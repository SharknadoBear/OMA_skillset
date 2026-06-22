"""
dbofs_fetcher.py
================
Fetch DBOFS (Delaware Bay Operational Forecast System) 3-D fields for
generating FVCOM initial condition files (F08).

Forcing component: F08 -- Initial temperature/salinity from DBOFS.

Design
------
*Streaming download-extract-delete* pattern (same as cbofs_fetcher.py):
for each target date, one DBOFS netCDF file is downloaded to a temporary
directory, the 3-D salt (and optionally temp) field is extracted and
interpolated to the FVCOM mesh, then the file is deleted.

NCEI archive
-------------
Base URL:
  https://www.ncei.noaa.gov/oa/prod-model/
  operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/
  access/delaware-bay-operational-forecast-system-dbofs/
File pattern:
  {YYYY}/{MM}/nos.dbofs.fields.f{FHH:03d}.{YYYYMMDD}.t{CC}z.nc
Available: 2014–present, ~48 MB per file, 4 cycles/day.

DBOFS grid info (similar to CBOFS — both are ROMS-based):
  Vtransform = 2, curvilinear rho-grid, N sigma layers.
  Variables: salt(ocean_time, s_rho, eta_rho, xi_rho),
             temp(ocean_time, s_rho, eta_rho, xi_rho),
             h(eta_rho, xi_rho), zeta(ocean_time, eta_rho, xi_rho),
             lon_rho(eta_rho, xi_rho), lat_rho(eta_rho, xi_rho),
             s_rho(s_rho), Cs_r(s_rho), hc

Python dependencies
-------------------
  numpy, netCDF4, scipy.spatial, requests
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import netCDF4 as nc4
except ImportError:
    raise ImportError("netCDF4 is required for dbofs_fetcher: pip install netCDF4")

try:
    import requests
except ImportError:
    raise ImportError("requests is required for dbofs_fetcher: pip install requests")

try:
    from scipy.spatial import cKDTree
except ImportError:
    raise ImportError("scipy is required for dbofs_fetcher: pip install scipy")

from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NCEI_DBOFS_BASE = (
    "https://www.ncei.noaa.gov/oa/prod-model/"
    "operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/"
    "access/delaware-bay-operational-forecast-system-dbofs/"
)

#: Available model cycles
ALL_CYCLES: tuple[str, ...] = ("t00z", "t06z", "t12z", "t18z")

#: Standard z-levels for FVCOM ITS file (from reference waterPACT_DRE_its.nc)
FVCOM_ITS_ZSL = np.array([
    0., -5., -10., -20., -30., -40., -50., -100., -150., -200.,
    -250., -300., -500., -700., -900., -1100., -1300., -1500.,
    -1700., -1900., -2100., -2300., -2500., -2700., -2900., -3100.,
    -5600., -6600., -7600., -8600.
], dtype=np.float32)

# ---------------------------------------------------------------------------
# HTTP setup (mirrors cbofs_fetcher.py)
# ---------------------------------------------------------------------------

_NCEI_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Connection": "keep-alive",
}

# Prefer PowerShell on Windows (WinHTTP/Schannel)
_PS_EXE: str | None = None
if os.name == "nt":
    _ps_candidate = (
        Path(os.environ.get("WINDIR", r"C:\Windows"))
        / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    )
    if _ps_candidate.is_file():
        _PS_EXE = str(_ps_candidate)
    else:
        _PS_EXE = shutil.which("powershell") or shutil.which("powershell.exe")

_CURL_EXE: str | None = shutil.which("curl") or shutil.which("curl.exe")
if _CURL_EXE is None and os.name == "nt":
    _win_curl = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "curl.exe"
    if _win_curl.is_file():
        _CURL_EXE = str(_win_curl)

# Persistent session for fallback requests
_SESSION = requests.Session()
_SESSION.headers.update(_NCEI_HEADERS)


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------

def _dbofs_url(date_str: str, cycle: str = "t00z", fhour: int = 0) -> str:
    """Build the NCEI prod-model download URL for one DBOFS field file.

    Parameters
    ----------
    date_str : ``'YYYY-MM-DD'`` date string
    cycle    : model cycle, one of ``'t00z'``, ``'t06z'``, ``'t12z'``, ``'t18z'``
    fhour    : forecast hour (0 = nowcast/analysis)

    Returns
    -------
    Full HTTPS URL string.
    """
    d = datetime.strptime(date_str, "%Y-%m-%d")
    yyyy = d.strftime("%Y")
    mm = d.strftime("%m")
    yyyymmdd = d.strftime("%Y%m%d")
    fname = f"nos.dbofs.fields.f{fhour:03d}.{yyyymmdd}.{cycle}.nc"
    return f"{NCEI_DBOFS_BASE}{yyyy}/{mm}/{fname}"


# ---------------------------------------------------------------------------
# HTTP downloader with retry (mirrors cbofs_fetcher._download_file)
# ---------------------------------------------------------------------------

def _download_file(url: str,
                   local_path: Path,
                   timeout: "int | tuple[int, int]" = (30, 180),
                   max_retries: int = 4) -> None:
    """Download *url* to *local_path* with retry + exponential backoff.

    Raises RuntimeError after all retries are exhausted.
    """
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    connect_t = timeout[0] if isinstance(timeout, tuple) else timeout
    read_t = timeout[1] if isinstance(timeout, tuple) else timeout
    max_time = connect_t + read_t

    for attempt in range(1, max_retries + 1):
        try:
            log.debug("DBOFS download attempt %d/%d: %s", attempt, max_retries, url)
            if _PS_EXE:
                ua = _NCEI_HEADERS["User-Agent"]
                out_str = str(local_path).replace('"', '`"')
                ps_cmd = (
                    "$ProgressPreference='SilentlyContinue'; "
                    f'Invoke-WebRequest -Uri "{url}" '
                    f'-OutFile "{out_str}" '
                    f'-UserAgent "{ua}" '
                    "-UseBasicParsing"
                )
                result = subprocess.run(
                    [_PS_EXE, "-NoProfile", "-NonInteractive",
                     "-Command", ps_cmd],
                    capture_output=True, timeout=max_time + 60,
                )
                if result.returncode != 0:
                    err = result.stderr.decode(errors="replace").strip()
                    raise RuntimeError(
                        f"PowerShell exit {result.returncode}: {err}"
                    )
            elif _CURL_EXE:
                cmd = [
                    _CURL_EXE, "-L", "-s", "-f",
                    "-o", str(local_path),
                    "--connect-timeout", str(connect_t),
                    "--max-time", str(max_time),
                    "-H", f"User-Agent: {_NCEI_HEADERS['User-Agent']}",
                    "-H", "Accept: */*",
                    url,
                ]
                result = subprocess.run(
                    cmd, capture_output=True, timeout=max_time + 30
                )
                if result.returncode != 0:
                    err = result.stderr.decode(errors="replace").strip()
                    raise RuntimeError(f"curl exit {result.returncode}: {err}")
            else:
                resp = _SESSION.get(url, stream=False, timeout=timeout)
                resp.raise_for_status()
                with open(local_path, "wb") as fh:
                    fh.write(resp.content)
            return  # success
        except Exception as exc:
            err_str = str(exc)
            # NoSuchKey = S3 object not found — don't retry, it's permanent
            if "NoSuchKey" in err_str or "404" in err_str:
                if local_path.exists():
                    local_path.unlink()
                raise RuntimeError(
                    f"DBOFS file not found (NoSuchKey/404): {url}"
                ) from exc
            if attempt == max_retries:
                raise RuntimeError(
                    f"DBOFS download failed after {max_retries} attempts: {url}\n"
                    f"  Last error: {exc}"
                ) from exc
            wait = 2 ** attempt
            log.warning("DBOFS download attempt %d/%d failed (%s); retrying in %ds",
                        attempt, max_retries, exc, wait)
            _time.sleep(wait)


# ---------------------------------------------------------------------------
# ROMS vertical coordinate computation
# ---------------------------------------------------------------------------

def roms_depths_2d(s_rho: np.ndarray,
                   Cs_r: np.ndarray,
                   hc: float,
                   h_2d: np.ndarray,
                   vtransform: int = 2) -> np.ndarray:
    """Compute physical depths at all ROMS rho-points (vectorized).

    Parameters
    ----------
    s_rho      : (N,) sigma coordinate values
    Cs_r       : (N,) stretching function values
    hc         : critical depth parameter
    h_2d       : (neta, nxi) bathymetry [m, positive]
    vtransform : ROMS Vtransform (1 or 2)

    Returns
    -------
    z : (N, neta, nxi) physical depths [m, negative-up].
        k=0 is near-bottom, k=N-1 is near-surface.
    """
    s = s_rho[:, None, None]   # (N, 1, 1)
    Cs = Cs_r[:, None, None]   # (N, 1, 1)
    h = h_2d[None, :, :]      # (1, neta, nxi)

    if vtransform == 2:
        z0 = (hc * s + h * Cs) / (hc + h)
        z = h * z0  # approximation with zeta=0 for initialization
    else:
        z0 = (s - Cs) * hc + Cs * h
        z = z0  # zeta=0
    return z.astype(np.float64)


# ---------------------------------------------------------------------------
# DBOFS field download
# ---------------------------------------------------------------------------

def fetch_dbofs_field(date_str: str,
                     cycle: str = "t00z",
                     fhour: int = 0,
                     work_dir: Optional[str | Path] = None) -> Path:
    """Download one DBOFS field file from the NCEI archive.

    Parameters
    ----------
    date_str : ISO date string ``'YYYY-MM-DD'``
    cycle    : model cycle (default ``'t00z'``)
    fhour    : forecast hour (default 0 = analysis/nowcast).
               If ``fhour=0`` (f000) fails, automatically retries with
               ``fhour=1`` (f001) since many months only have f001+.
    work_dir : directory to save to; defaults to system temp

    Returns
    -------
    Path to the downloaded file.  Caller is responsible for cleanup.
    """
    if work_dir is None:
        work_dir = Path(tempfile.gettempdir()) / "dbofs_cache"
    else:
        work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Try requested fhour first; if f000 fails, fall back to f001
    fhours_to_try = [fhour]
    if fhour == 0:
        fhours_to_try.append(1)  # f000 not available for most months

    for fh in fhours_to_try:
        url = _dbofs_url(date_str, cycle, fh)
        fname = url.rsplit("/", 1)[-1]
        local_path = work_dir / fname

        if local_path.exists() and local_path.stat().st_size > 1_000_000:
            log.info("DBOFS file already cached: %s", local_path.name)
            return local_path

        try:
            log.info("Downloading DBOFS: %s", fname)
            _download_file(url, local_path)
            return local_path
        except RuntimeError as exc:
            if fh == fhours_to_try[-1]:
                raise  # last attempt, propagate error
            log.warning("f%03d not available (%s); trying f%03d ...",
                        fh, str(exc)[:80], fhours_to_try[fhours_to_try.index(fh) + 1])
            # Clean up partial download
            if local_path.exists():
                local_path.unlink()
            continue

    # Should not reach here, but just in case
    raise RuntimeError(f"DBOFS download failed for {date_str} cycle={cycle}")


# ---------------------------------------------------------------------------
# DBOFS field extraction
# ---------------------------------------------------------------------------

def extract_dbofs_field(nc_path: str | Path,
                        variables: tuple[str, ...] = ("salt",),
                        ) -> dict:
    """Extract 3-D fields and grid info from a DBOFS file.

    Parameters
    ----------
    nc_path   : path to downloaded nos.dbofs.fields.*.nc
    variables : tuple of variable names to extract (default ``('salt',)``)

    Returns
    -------
    dict with keys:
      - 'lon_rho'  : (neta, nxi) float64 — longitude grid
      - 'lat_rho'  : (neta, nxi) float64 — latitude grid
      - 'h'        : (neta, nxi) float64 — bathymetry [m, positive]
      - 'mask_rho' : (neta, nxi) int — land mask (1=water, 0=land)
      - 's_rho'    : (N,) sigma coords
      - 'Cs_r'     : (N,) stretching function
      - 'hc'       : float — critical depth
      - 'vtransform': int
      - 'z'        : (N, neta, nxi) physical depths [m, negative-up]
      - 'salt'     : (N, neta, nxi) float64 (if requested)
      - 'temp'     : (N, neta, nxi) float64 (if requested)
      - 'ocean_time': np.datetime64
    """
    result = {}
    with nc4.Dataset(str(nc_path), "r") as ds:
        # Grid coordinates
        result["lon_rho"] = np.ma.filled(ds.variables["lon_rho"][:], np.nan).astype(np.float64)
        result["lat_rho"] = np.ma.filled(ds.variables["lat_rho"][:], np.nan).astype(np.float64)
        result["h"] = np.ma.filled(ds.variables["h"][:], 0.0).astype(np.float64)

        # Land mask
        if "mask_rho" in ds.variables:
            result["mask_rho"] = np.ma.filled(ds.variables["mask_rho"][:], 0).astype(np.int32)
        else:
            # Infer from salt: land points are masked
            raw = ds.variables["salt"][0, -1, :, :]  # surface layer
            result["mask_rho"] = (~np.ma.getmaskarray(raw)).astype(np.int32)

        # Sigma coordinate parameters
        s_rho = ds.variables["s_rho"][:].astype(np.float64)
        Cs_r = ds.variables["Cs_r"][:].astype(np.float64)
        hc = float(ds.variables["hc"][:])

        vtransform = 2
        if "Vtransform" in ds.variables:
            vtransform = int(ds.variables["Vtransform"][:])
        elif hasattr(ds, "Vtransform"):
            vtransform = int(ds.Vtransform)

        result["s_rho"] = s_rho
        result["Cs_r"] = Cs_r
        result["hc"] = hc
        result["vtransform"] = vtransform

        # Compute physical depths at all points (with zeta=0)
        result["z"] = roms_depths_2d(s_rho, Cs_r, hc, result["h"], vtransform)

        # Extract requested variables
        for var in variables:
            if var in ds.variables:
                raw = ds.variables[var][0, :, :, :]  # (N, neta, nxi)
                arr = np.ma.filled(raw, np.nan).astype(np.float64)
                result[var] = arr
            else:
                log.warning("Variable '%s' not found in DBOFS file", var)

        # Time
        ot_var = ds.variables["ocean_time"]
        ot_val = float(ot_var[0])
        ot_unit = getattr(ot_var, "units", "seconds since 2016-01-01")
        ot_cal = getattr(ot_var, "calendar", "gregorian")
        # Normalize non-standard calendar names
        if ot_cal == "gregorian_proleptic":
            ot_cal = "proleptic_gregorian"
        dt_cf = nc4.num2date(ot_val, ot_unit, ot_cal)
        result["ocean_time"] = np.datetime64(dt_cf.strftime("%Y-%m-%dT%H:%M:%S"), "s")

    return result


# ---------------------------------------------------------------------------
# Horizontal interpolation: DBOFS grid → FVCOM nodes
# ---------------------------------------------------------------------------

def interp_dbofs_to_fvcom(dbofs_data: dict,
                          fvcom_lon: np.ndarray,
                          fvcom_lat: np.ndarray,
                          fvcom_h: np.ndarray,
                          zsl: np.ndarray = FVCOM_ITS_ZSL,
                          variables: tuple[str, ...] = ("salt",),
                          n_nearest: int = 1,
                          ) -> dict:
    """Interpolate DBOFS 3-D fields to FVCOM nodes on z-levels.

    For each FVCOM node:
    1. Find nearest wet DBOFS rho-point (using KDTree on lon/lat).
    2. Get the ROMS vertical profile at that point.
    3. Interpolate vertically from ROMS physical depths to the
       standard z-levels (``zsl``).
    4. For z-levels below the local DBOFS seafloor, use bottom value.
       For z-levels above surface, use surface value.

    Parameters
    ----------
    dbofs_data : dict from :func:`extract_dbofs_field`
    fvcom_lon  : (nnode,) FVCOM node longitudes [degrees, can be negative]
    fvcom_lat  : (nnode,) FVCOM node latitudes [degrees]
    fvcom_h    : (nnode,) FVCOM bathymetry at each node [m, positive]
    zsl        : (ksl,) target z-levels [m, negative-up]
    variables  : which 3-D fields to interpolate
    n_nearest  : number of nearest neighbours (1 = nearest, >1 = IDW)

    Returns
    -------
    dict with:
      - 'salt' : (ksl, nnode) float32 — salinity on z-levels
      - 'temp' : (ksl, nnode) float32 — temperature on z-levels (if requested)
      - 'ocean_time' : np.datetime64
    """
    lon_rho = dbofs_data["lon_rho"]
    lat_rho = dbofs_data["lat_rho"]
    mask_rho = dbofs_data["mask_rho"]
    z_3d = dbofs_data["z"]       # (N, neta, nxi)

    # Flatten water points for KDTree
    water = mask_rho == 1
    lon_water = lon_rho[water]   # (nwater,)
    lat_water = lat_rho[water]

    # Build KDTree on (lat, lon) — approximate for nearby points
    tree = cKDTree(np.column_stack([lat_water, lon_water]))

    nnode = len(fvcom_lon)
    ksl = len(zsl)

    # Query nearest for all FVCOM nodes
    query_pts = np.column_stack([fvcom_lat, fvcom_lon])
    dists, indices = tree.query(query_pts, k=n_nearest)
    if n_nearest == 1:
        indices = indices[:, None]  # (nnode, 1) for uniform handling

    # Precompute flat indices for water points
    eta_water, xi_water = np.where(water)

    result = {"ocean_time": dbofs_data["ocean_time"]}

    for var in variables:
        if var not in dbofs_data:
            continue
        field_3d = dbofs_data[var]  # (N, neta, nxi)
        N = field_3d.shape[0]

        out = np.full((ksl, nnode), np.nan, dtype=np.float64)

        for i_node in range(nnode):
            # Get nearest water point index
            idx = indices[i_node, 0]
            j_eta = eta_water[idx]
            j_xi = xi_water[idx]

            # Vertical profile at that DBOFS point
            profile = field_3d[:, j_eta, j_xi]  # (N,) bottom→surface
            z_profile = z_3d[:, j_eta, j_xi]    # (N,) negative, bottom→surface

            # Skip if all NaN
            valid = np.isfinite(profile)
            if valid.sum() < 2:
                if valid.sum() == 1:
                    out[:, i_node] = profile[valid][0]
                continue

            z_valid = z_profile[valid]
            v_valid = profile[valid]

            # Ensure sorted ascending (bottom first = most negative)
            sort_idx = np.argsort(z_valid)
            z_valid = z_valid[sort_idx]
            v_valid = v_valid[sort_idx]

            # Interpolate to z-levels with nearest-neighbor clamping
            f_interp = interp1d(z_valid, v_valid, kind="linear",
                                bounds_error=False, fill_value=np.nan)
            interped = f_interp(zsl)

            # Clamp: above surface → surface value, below bottom → bottom value
            above = zsl > z_valid[-1]
            below = zsl < z_valid[0]
            interped[above] = v_valid[-1]  # surface value
            interped[below] = v_valid[0]   # bottom value

            out[:, i_node] = interped

        # Fill any remaining NaN nodes with nearest valid node (lateral fill)
        for k in range(ksl):
            row = out[k, :]
            nan_mask = ~np.isfinite(row)
            if nan_mask.any() and not nan_mask.all():
                valid_idx = np.where(~nan_mask)[0]
                nan_idx = np.where(nan_mask)[0]
                nearest = valid_idx[
                    np.argmin(np.abs(nan_idx[:, None] - valid_idx[None, :]), axis=1)
                ]
                out[k, nan_idx] = row[nearest]

        result[var] = out.astype(np.float32)

    return result


# ---------------------------------------------------------------------------
# High-level convenience: fetch + extract + interpolate in one call
# ---------------------------------------------------------------------------

def fetch_and_interp_dbofs(date_str: str,
                           fvcom_lon: np.ndarray,
                           fvcom_lat: np.ndarray,
                           fvcom_h: np.ndarray,
                           zsl: np.ndarray = FVCOM_ITS_ZSL,
                           variables: tuple[str, ...] = ("salt",),
                           cycle: str = "t00z",
                           fhour: int = 0,
                           work_dir: Optional[str | Path] = None,
                           keep_file: bool = False,
                           ) -> dict:
    """Fetch one DBOFS snapshot and interpolate to FVCOM nodes.

    Combines :func:`fetch_dbofs_field`, :func:`extract_dbofs_field`, and
    :func:`interp_dbofs_to_fvcom` into a single call.

    Parameters
    ----------
    date_str   : ISO date ``'YYYY-MM-DD'``
    fvcom_lon  : (nnode,) longitudes (can be negative, e.g. -75°)
    fvcom_lat  : (nnode,) latitudes
    fvcom_h    : (nnode,) bathymetry [m, positive]
    zsl        : (ksl,) target z-levels [m, negative-up]
    variables  : variables to extract/interpolate
    cycle      : DBOFS model cycle
    fhour      : forecast hour
    work_dir   : cache directory for the downloaded file
    keep_file  : if False, delete the downloaded file after extraction

    Returns
    -------
    dict with 'salt' and/or 'temp' arrays of shape (ksl, nnode),
    plus 'ocean_time' as np.datetime64.
    """
    nc_path = fetch_dbofs_field(date_str, cycle=cycle, fhour=fhour,
                                work_dir=work_dir)
    try:
        dbofs_data = extract_dbofs_field(nc_path, variables=variables)
        result = interp_dbofs_to_fvcom(dbofs_data, fvcom_lon, fvcom_lat,
                                       fvcom_h, zsl=zsl, variables=variables)
    finally:
        if not keep_file and nc_path.exists():
            nc_path.unlink()
            log.info("Deleted temporary DBOFS file: %s", nc_path.name)

    return result
