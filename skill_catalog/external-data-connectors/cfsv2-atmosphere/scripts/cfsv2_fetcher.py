"""
cfsv2_fetcher.py
================
Fetch CFSv2 atmospheric forcing from the HYCOM OPeNDAP server for FVCOM
surface forcing (F07).

Forcing: F07 -- 10-m wind (U, V), surface pressure, and optionally other
         surface fluxes (longwave, shortwave, wind stress, precip, etc.).

Reference workflow
------------------
Old MATLAB:
  d_surface_forcing.m
    Sin.vartype = 'wind'     -> reads cfsv2-sec2_YYYY_01hr_uv-10m.nc
    Sin.vartype = 'pressure' -> reads cfsv2-sec_YYYY_01hr_sfcprs.nc
    Spatial subset: lat 36-41, lon 283-288
  fvcom_prepro/cfs2fvcom.m  -> write WRF-style regular-grid NC for FVCOM

Data source
-----------
  HYCOM THREDDS OPeNDAP server:
    https://tds.hycom.org/thredds/dodsC/datasets/force/ncep_cfsv2/netcdf/

  Three file-prefix patterns (all tried; first that connects wins):
    cfsv2-sec2_{YYYY}_01hr_{subdataset}.nc  (primary for uv-10m)
    cfsv2-sec_{YYYY}_01hr_{subdataset}.nc   (primary for sfcprs)
    cfsv2-sea_{YYYY}_01hr_{subdataset}.nc   (fallback)

  Subdatasets and corresponding product variables
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  uv-10m     wndewd (U10 m/s),  wndnwd (V10 m/s)
  sfcprs     airprs (Pa)
  dlwflx     dlwflx (W/m^2)
  dswsfc     dswflx (W/m^2)
  strblk     tauewd, taunwd  (N/m^2)
  TaqaQrQp   airtmp (K), vapmix (kg/kg), radflx (W/m^2), shwflx (W/m^2)
  precip     precip (kg m^-2 s^-1)
  surtmp     surtmp (K)

  Coordinate variables (all subdatasets):
    MT        -- time [days since 1900-12-31 00:00:00]
    Latitude  -- [deg N]
    Longitude -- [deg E, 0-360 convention]

Output format
-------------
  Spatial subset on the native CFSv2 regular grid (~0.2 deg resolution).
  Annual cache files: {cache_dir}/{subdataset}_{year}.nc
  Final forcing written by fvcom_writer.write_surface_forcing() in the
  WRF-style FVCOM METEO FORCING FILE format (XLAT/XLONG 2-D, Times char
  array, U10/V10 or air_pressure).  FVCOM performs its own internal
  interpolation from this regular grid to the unstructured mesh.

Python dependencies
-------------------
  xarray, pydap, netCDF4, numpy
  (pydap is required for HYCOM THREDDS OPeNDAP access)
"""

from __future__ import annotations

import logging
import sys
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

# Optional tqdm for chunk-level progress (graceful fallback to print)
try:
    from tqdm.std import tqdm as _Tqdm
except ImportError:  # pragma: no cover
    _Tqdm = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HYCOM_BASE = (
    "https://tds.hycom.org/thredds/dodsC/datasets/force/ncep_cfsv2/netcdf/"
)

# File-prefix candidates — tried in order for every (year, subdataset) pair
_URL_PREFIXES = ["cfsv2-sec2", "cfsv2-sec", "cfsv2-sea"]

# MT epoch: days since 1900-12-31 00:00:00
_MT_EPOCH = np.datetime64("1900-12-31T00:00:00", "s")

# Default spatial domain (Delaware Bay / Delaware River region)
DEFAULT_LON_RANGE: tuple[float, float] = (283.0, 288.0)  # 0-360 convention
DEFAULT_LAT_RANGE: tuple[float, float] = (36.0,  41.0)

# Subdataset → product variable names mapping
SUBDATASET_VARIABLES: dict[str, list[str]] = {
    "uv-10m":    ["wndewd", "wndnwd"],
    "sfcprs":    ["airprs"],
    "dlwflx":    ["dlwflx"],
    "dswsfc":    ["dswflx"],
    "strblk":    ["tauewd", "taunwd"],
    "TaqaQrQp":  ["airtmp", "vapmix", "radflx", "shwflx"],
    "precip":    ["precip"],
    "surtmp":    ["surtmp"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mt_to_datetime64(mt_values: np.ndarray) -> np.ndarray:
    """Convert MT float array (days since 1900-12-31) to datetime64[s]."""
    seconds = (np.asarray(mt_values, dtype=np.float64) * 86400).astype("int64")
    return _MT_EPOCH + seconds * np.timedelta64(1, "s")


def _try_cfsv2_url(year: int,
                   subdataset_name: str) -> tuple[str, xr.Dataset]:
    """
    Probe all three URL-prefix patterns and return (base_url, coords_ds).

    ``coords_ds`` contains only ``MT``, ``Latitude``, ``Longitude``
    (lightweight fetch used solely for index computations).
    Uses the pydap engine required for HYCOM THREDDS OPeNDAP access.

    Raises
    ------
    RuntimeError if none of the three URL templates succeed.
    """
    errors: list[str] = []
    for prefix in _URL_PREFIXES:
        base_url  = f"{_HYCOM_BASE}{prefix}_{year}_01hr_{subdataset_name}.nc"
        coord_url = f"{base_url}?MT,Latitude,Longitude"
        try:
            ds = xr.open_dataset(coord_url, engine="pydap", decode_times=False)
            _ = ds["Latitude"].values   # force first load to verify connectivity
            log.info("CFSv2 connected: %s", base_url)
            return base_url, ds
        except Exception as exc:
            errors.append(f"  {prefix}: {exc}")

    raise RuntimeError(
        f"Could not connect to CFSv2 {year}/{subdataset_name} via any URL prefix.\n"
        + "\n".join(errors)
    )


def _download_chunk(
    base_url: str,
    variables: list[str],
    t0_idx: int,
    t1_idx: int,
    lat0: int,
    lat1: int,
    lon0: int,
    lon1: int,
    max_retries: int = 5,
) -> xr.Dataset:
    """Download one spatial+time chunk with exponential-backoff retry."""
    var_parts = [
        f"{v}[{t0_idx}:1:{t1_idx}][{lat0}:1:{lat1}][{lon0}:1:{lon1}]"
        for v in variables
    ]
    coord_parts = [
        f"MT[{t0_idx}:1:{t1_idx}]",
        f"Latitude[{lat0}:1:{lat1}]",
        f"Longitude[{lon0}:1:{lon1}]",
    ]
    url = f"{base_url}?{','.join(var_parts + coord_parts)}"

    for attempt in range(max_retries):
        try:
            ds = xr.open_dataset(url, engine="pydap", decode_times=False)
            ds.load()   # materialise into memory before returning
            return ds
        except Exception as exc:
            wait = 2 ** attempt
            log.warning(
                "Chunk t[%d:%d] attempt %d/%d failed: %s — retry in %ds",
                t0_idx, t1_idx, attempt + 1, max_retries, exc, wait,
            )
            _time.sleep(wait)

    raise RuntimeError(
        f"Chunk t[{t0_idx}:{t1_idx}] failed after {max_retries} retries.\n"
        f"URL: {url}"
    )


# ---------------------------------------------------------------------------
# Internal helpers (continued)
# ---------------------------------------------------------------------------

def _sanitize_nc_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Drop any attribute whose value is a dict (not serializable to NetCDF).

    pydap occasionally returns nested dicts from the OPeNDAP DAS metadata
    (e.g. ``'Date': {'long_name': ..., 'next_Date': 20190101.0}``).
    xarray raises ``TypeError`` when attempting to write such values.
    """
    _VALID = (str, bytes, int, float, np.integer, np.floating, np.ndarray,
              list, tuple)
    ds.attrs = {k: v for k, v in ds.attrs.items() if isinstance(v, _VALID)}
    for var in ds.variables.values():
        var.attrs = {k: v for k, v in var.attrs.items() if isinstance(v, _VALID)}
    return ds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_cfsv2_year(
    year: int,
    subdataset_name: str,
    variables: list[str],
    lon_range: tuple[float, float] = DEFAULT_LON_RANGE,
    lat_range: tuple[float, float] = DEFAULT_LAT_RANGE,
    cache_dir: str | Path = ".",
    chunk_days: int = 30,
    max_retries: int = 5,
    overwrite: bool = False,
) -> Path:
    """
    Download one calendar year of CFSv2 data from the HYCOM OPeNDAP server.

    Downloads the spatial subset ``lat_range × lon_range`` in monthly-sized
    chunks to keep per-request payload manageable, concatenates along the
    ``MT`` dimension, and saves to ``{cache_dir}/{subdataset_name}_{year}.nc``.

    Parameters
    ----------
    year            : calendar year, e.g. 2018, 2019, or 2020
    subdataset_name : HYCOM subdataset key — one of the keys in
                      :data:`SUBDATASET_VARIABLES`, e.g. ``'uv-10m'``
    variables       : product variable names inside that subdataset,
                      e.g. ``['wndewd', 'wndnwd']``
    lon_range       : (lon_min, lon_max) in **0-360** convention [deg E]
    lat_range       : (lat_min, lat_max) [deg N]
    cache_dir       : output directory; created if it does not exist
    chunk_days      : approximate size of each OPeNDAP download slice [days]
                      (default 30 ≈ one calendar month)
    max_retries     : per-chunk exponential-backoff retries (default 5)
    overwrite       : re-download even if the cache file already exists
                      (default False)

    Returns
    -------
    ``pathlib.Path`` to the cached ``{subdataset_name}_{year}.nc`` file.

    Examples
    --------
    ::

        path = fetch_cfsv2_year(2019, 'uv-10m', ['wndewd', 'wndnwd'],
                                lon_range=(283, 288), lat_range=(36, 41),
                                cache_dir='data_raw/cfsv2')
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{subdataset_name}_{year}.nc"

    if out_path.exists() and not overwrite:
        print(f"[CACHE HIT] {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path

    print(f"[DOWNLOAD] CFSv2 {subdataset_name} {year} ...")
    base_url, coords_ds = _try_cfsv2_url(year, subdataset_name)

    lats = coords_ds["Latitude"].values.astype(float)
    lons = coords_ds["Longitude"].values.astype(float)
    mt   = coords_ds["MT"].values.astype(float)

    # Spatial index bounds (with 1-point padding for interpolation safety)
    lat0_i = max(0, int(np.searchsorted(lats, lat_range[0])) - 1)
    lat1_i = min(len(lats) - 1, int(np.searchsorted(lats, lat_range[1], side="right")))
    lon0_i = max(0, int(np.searchsorted(lons, lon_range[0])) - 1)
    lon1_i = min(len(lons) - 1, int(np.searchsorted(lons, lon_range[1], side="right")))

    print(
        f"  Grid: lat[{lat0_i}:{lat1_i}] = {lats[lat0_i]:.2f}°–{lats[lat1_i]:.2f}°N  "
        f"lon[{lon0_i}:{lon1_i}] = {lons[lon0_i]:.2f}°–{lons[lon1_i]:.2f}°E"
    )

    # Time index bounds for the requested year
    dt_arr  = _mt_to_datetime64(mt)
    y_start = np.datetime64(f"{year}-01-01T00:00:00", "s")
    y_end   = np.datetime64(f"{year}-12-31T23:59:59", "s")
    t_mask  = (dt_arr >= y_start) & (dt_arr <= y_end)
    t_idx   = np.where(t_mask)[0]

    if len(t_idx) == 0:
        raise ValueError(
            f"No MT time steps found for year {year} in the remote dataset."
        )

    # Download in chunks
    chunk_size = chunk_days * 24   # 1-hourly data
    chunk_bounds = list(range(0, len(t_idx), chunk_size))
    n_chunks = len(chunk_bounds)
    chunks: list[xr.Dataset] = []

    pbar = (
        _Tqdm(
            total=n_chunks,
            desc=f"  {subdataset_name} {year}",
            unit="chunk",
            leave=True,
            file=sys.stdout,
        )
        if _Tqdm is not None
        else None
    )

    for i_chunk, cs in enumerate(chunk_bounds):
        sub  = t_idx[cs : cs + chunk_size]
        t0i  = int(sub[0])
        t1i  = int(sub[-1])
        date_str = f"{str(dt_arr[t0i])[:10]} – {str(dt_arr[t1i])[:10]}"
        if pbar is not None:
            pbar.set_description(f"  {subdataset_name} {year}  [{date_str}]")
        else:
            print(
                f"  Chunk {i_chunk + 1}/{n_chunks}:"
                f" MT[{t0i}:{t1i}]"
                f" ({date_str})"
            )
        chunk_ds = _download_chunk(
            base_url, variables,
            t0i, t1i,
            lat0_i, lat1_i,
            lon0_i, lon1_i,
            max_retries=max_retries,
        )
        chunks.append(chunk_ds)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    ds_full = xr.concat(chunks, dim="MT")
    ds_full = _sanitize_nc_attrs(ds_full)   # strip pydap dict-attrs before write
    ds_full.to_netcdf(out_path)
    print(f"[OK] Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def fetch_wind_year(
    year: int,
    lon_range: tuple[float, float] = DEFAULT_LON_RANGE,
    lat_range: tuple[float, float] = DEFAULT_LAT_RANGE,
    cache_dir: str | Path = ".",
    **kwargs,
) -> Path:
    """
    Fetch CFSv2 10-m wind (``wndewd``, ``wndnwd``) for one calendar year.

    Convenience wrapper around :func:`fetch_cfsv2_year` with
    ``subdataset_name='uv-10m'``.

    Returns
    -------
    Path to ``uv-10m_{year}.nc`` in ``cache_dir``.
    """
    return fetch_cfsv2_year(
        year, "uv-10m", ["wndewd", "wndnwd"],
        lon_range=lon_range, lat_range=lat_range,
        cache_dir=cache_dir, **kwargs,
    )


def fetch_pressure_year(
    year: int,
    lon_range: tuple[float, float] = DEFAULT_LON_RANGE,
    lat_range: tuple[float, float] = DEFAULT_LAT_RANGE,
    cache_dir: str | Path = ".",
    **kwargs,
) -> Path:
    """
    Fetch CFSv2 surface pressure (``airprs``) for one calendar year.

    Convenience wrapper around :func:`fetch_cfsv2_year` with
    ``subdataset_name='sfcprs'``.

    Returns
    -------
    Path to ``sfcprs_{year}.nc`` in ``cache_dir``.
    """
    return fetch_cfsv2_year(
        year, "sfcprs", ["airprs"],
        lon_range=lon_range, lat_range=lat_range,
        cache_dir=cache_dir, **kwargs,
    )


def load_and_concat_years(
    subdataset_name: str,
    years: list[int],
    cache_dir: str | Path,
) -> xr.Dataset:
    """
    Load and concatenate annual cache files along the ``MT`` dimension.

    Parameters
    ----------
    subdataset_name : e.g. ``'uv-10m'`` or ``'sfcprs'``
    years           : list of years, e.g. ``[2018, 2019, 2020]``
    cache_dir       : directory containing ``{subdataset}_{year}.nc`` files

    Returns
    -------
    ``xr.Dataset`` with variables ``MT``, ``Latitude``, ``Longitude``, and
    all product variables, spanning the full requested time range.

    Raises
    ------
    FileNotFoundError if any annual cache file is missing.
    """
    cache_dir = Path(cache_dir)
    datasets: list[xr.Dataset] = []
    for yr in years:
        fpath = cache_dir / f"{subdataset_name}_{yr}.nc"
        if not fpath.exists():
            raise FileNotFoundError(
                f"Cache file not found: {fpath}\n"
                f"Run fetch_cfsv2_year({yr}, '{subdataset_name}', ...) first."
            )
        datasets.append(xr.open_dataset(fpath, decode_times=False))
    combined = xr.concat(datasets, dim="MT")
    return combined


# ---------------------------------------------------------------------------
# Deprecated backward-compatible stubs
# ---------------------------------------------------------------------------

def load_cfsv2_wind(nc_path, lon_range=DEFAULT_LON_RANGE,
                    lat_range=DEFAULT_LAT_RANGE):
    """
    Deprecated — local PNNL E:\\waterPACT\\ files no longer used.

    Use :func:`fetch_wind_year` to download from the HYCOM OPeNDAP server.
    """
    import warnings
    warnings.warn(
        "load_cfsv2_wind() is deprecated. Use fetch_wind_year() to download "
        "from the HYCOM OPeNDAP server instead.",
        DeprecationWarning, stacklevel=2,
    )
    raise NotImplementedError(
        "Local PNNL files not available. Use fetch_wind_year() instead."
    )


def load_cfsv2_pressure(nc_path, lon_range=DEFAULT_LON_RANGE,
                        lat_range=DEFAULT_LAT_RANGE):
    """
    Deprecated — local PNNL E:\\waterPACT\\ files no longer used.

    Use :func:`fetch_pressure_year` to download from the HYCOM OPeNDAP server.
    """
    import warnings
    warnings.warn(
        "load_cfsv2_pressure() is deprecated. Use fetch_pressure_year() to "
        "download from the HYCOM OPeNDAP server instead.",
        DeprecationWarning, stacklevel=2,
    )
    raise NotImplementedError(
        "Local PNNL files not available. Use fetch_pressure_year() instead."
    )


def regrid_to_fvcom(ds, variable, fvcom_lon, fvcom_lat):
    """
    Deprecated — no longer needed.

    CFSv2 forcing stays on the native regular grid.  FVCOM reads the
    WRF-style regular-grid forcing file directly and performs its own
    internal interpolation to the unstructured mesh.  Write output with
    :func:`~fvcom_writer.write_surface_forcing` on the native CFSv2 grid.
    """
    import warnings
    warnings.warn(
        "regrid_to_fvcom() is deprecated and no longer needed. "
        "FVCOM reads the WRF-style regular-grid forcing file directly.",
        DeprecationWarning, stacklevel=2,
    )
    raise NotImplementedError(
        "CFSv2 forcing stays on the regular grid; no node-level regridding needed."
    )
