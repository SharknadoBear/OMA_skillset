"""
fvcom_writer.py
===============
Write FVCOM-compliant NetCDF forcing files (CF-1.8 convention).
Time variable uses Modified Julian Day epoch 1858-11-17 00:00:00 UTC,
matching FVCOM's internal time convention.

Reference workflow
------------------
Old MATLAB equivalents:
  - fvcom_prepro/write_FVCOM_elevtide.m  --> write_elevation_obc()
  - fvcom_prepro/write_FVCOM_tsobc.m     --> write_tsobc()
  - fvcom_prepro/write_FVCOM_river.m     --> write_river_nc()
  - fvcom_prepro/write_FVCOM_river_nml.m --> write_river_nml()
  - fvcom_prepro/write_FVCOM_forcing.m   --> write_surface_forcing()
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import netCDF4 as nc4
try:
    from .grid_utils import MJD_EPOCH
except ImportError:
    from grid_utils import MJD_EPOCH


# ---------------------------------------------------------------------------
# F01 / F02 / F04 -- Elevation / tidal OBC
# ---------------------------------------------------------------------------

def write_elevation_obc(out_path: str | Path,
                        obc_nodes: np.ndarray,
                        time_mjd: np.ndarray,
                        zeta: np.ndarray,
                        casename: str = "waterPACT") -> None:
    """
    Write FVCOM open-boundary elevation time series NetCDF.

    Replicates write_FVCOM_elevtide.m with strtime=T, inttime=T, floattime=T.

    Parameters
    ----------
    out_path  : output NetCDF file path
    obc_nodes : (nobc,) int32 — 1-based FVCOM node indices
    time_mjd  : (ntime,) float64 — Modified Julian Days
                 (epoch 1858-11-17 00:00:00 UTC)
    zeta      : (nobc, ntime) float32 — surface elevation [metres]
    casename  : FVCOM case name string (written as global attribute 'title')
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nobc, ntime = zeta.shape
    if len(obc_nodes) != nobc:
        raise ValueError(
            f"obc_nodes length {len(obc_nodes)} != zeta rows {nobc}"
        )
    if len(time_mjd) != ntime:
        raise ValueError(
            f"time_mjd length {len(time_mjd)} != zeta columns {ntime}"
        )

    # Time representations
    mjd_floor = np.floor(time_mjd).astype(np.int32)
    mjd_ms    = np.round((time_mjd - mjd_floor) * 86400000).astype(np.int32)

    # Time strings: "YYYY/MM/DD HH:MM:SS.ffffff"  (DateStrLen = 26)
    dt64         = MJD_EPOCH + (time_mjd * 86400).astype("int64") * np.timedelta64(1, "s")
    time_str_arr = np.zeros((ntime, 26), dtype="S1")
    for i, t in enumerate(dt64):
        s = str(t).replace("T", " ") + ".000000"   # 'YYYY-MM-DD HH:MM:SS.ffffff'
        s = s.replace("-", "/")                      # 'YYYY/MM/DD HH:MM:SS.ffffff'
        for j, ch in enumerate(s[:26]):
            time_str_arr[i, j] = ch.encode("ascii")

    with nc4.Dataset(out_path, "w", format="NETCDF3_CLASSIC") as nc:
        # Global attributes
        nc.type    = "FVCOM TIME SERIES ELEVATION FORCING FILE"
        nc.title   = casename
        nc.history = "File created with fvcom_writer.py — FVCOM preprocessing toolkit"

        # Dimensions
        nc.createDimension("nobc", nobc)
        nc.createDimension("time", None)         # UNLIMITED
        nc.createDimension("DateStrLen", 26)

        # obc_nodes
        v           = nc.createVariable("obc_nodes", "i4", ("nobc",))
        v.long_name = "Open Boundary Node Number"
        v.grid      = "obc_grid"
        v[:]        = obc_nodes.astype(np.int32)

        # iint
        v           = nc.createVariable("iint", "i4", ("time",))
        v.long_name = "internal mode iteration number"
        v[:]        = np.arange(1, ntime + 1, dtype=np.int32)

        # time (float MJD)
        v           = nc.createVariable("time", "f4", ("time",))
        v.long_name = "time"
        v.units     = "days since 1858-11-17 00:00:00"
        v.format    = "modified julian day (MJD)"
        v.time_zone = "UTC"
        v[:]        = time_mjd.astype(np.float32)

        # Itime (integer days)
        v           = nc.createVariable("Itime", "i4", ("time",))
        v.units     = "days since 1858-11-17 00:00:00"
        v.format    = "modified julian day (MJD)"
        v.time_zone = "UTC"
        v[:]        = mjd_floor

        # Itime2 (milliseconds since midnight)
        v           = nc.createVariable("Itime2", "i4", ("time",))
        v.units     = "msec since 00:00:00"
        v.time_zone = "UTC"
        v[:]        = mjd_ms

        # Times (character array)
        v           = nc.createVariable("Times", "S1", ("time", "DateStrLen"))
        v.time_zone = "UTC"
        v[:]        = time_str_arr

        # elevation — NETCDF3_CLASSIC requires the UNLIMITED dim (time) first.
        # We accept zeta as (nobc, ntime) and transpose when writing.
        v           = nc.createVariable("elevation", "f4", ("time", "nobc"))
        v.long_name = "Open Boundary Elevation"
        v.units     = "meters"
        v[:]        = zeta.T.astype(np.float32)   # (ntime, nobc)

    print(f"[OK] Written: {out_path}  ({nobc} OBC nodes, {ntime} time steps)")


# ---------------------------------------------------------------------------
# F03 / F05 -- T/S OBC
# ---------------------------------------------------------------------------

def write_tsobc(out_path: str | Path,
                obc_nodes: np.ndarray,
                time_mjd: np.ndarray,
                temp: np.ndarray,
                salt: np.ndarray,
                siglay: np.ndarray,
                siglev: np.ndarray,
                casename: str = "waterPACT") -> None:
    """
    Write FVCOM T/S open boundary condition file (NETCDF3_CLASSIC).

    The output format matches the reference file ``waterPACT_tsobc_2019.nc``
    produced by the old MATLAB preprocessing toolkit, with variable names
    ``obc_temp`` and ``obc_salinity`` (required by FVCOM).

    Parameters
    ----------
    out_path   : output file path
    obc_nodes  : (nobc,) 1-based FVCOM OBC node indices
    time_mjd   : (ntime,) Modified Julian Day time coordinate
    temp       : (ntime, nsiglay, nobc) potential temperature [°C]
    salt       : (ntime, nsiglay, nobc) salinity [PSU]
    siglay     : (nsiglay,) FVCOM sigma layer midpoints, range [-1, 0].
                 siglay[0] is just below the surface, siglay[-1] near the bed.
    siglev     : (nsiglev,) FVCOM sigma level interfaces, range [-1, 0].
                 nsiglev = nsiglay + 1.
    casename   : string stored in the ``title`` global attribute.

    Notes
    -----
    * Dimension order follows FVCOM convention: ``(time, siglay, nobc)``.
    * ``siglay`` and ``siglev`` are written as 2-D variables ``(siglay, nobc)``
      and ``(siglev, nobc)`` — the same sigma values are broadcast across all
      OBC nodes, matching FVCOM's internal storage.
    * Time encoding is identical to :func:`write_elevation_obc`:
      MJD float, integer day + millisecond-of-day, and character array.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    temp = np.asarray(temp, dtype=np.float32)
    salt = np.asarray(salt, dtype=np.float32)

    if temp.ndim != 3 or salt.ndim != 3:
        raise ValueError(
            f"temp and salt must be 3-D (ntime, nsiglay, nobc); "
            f"got shapes {temp.shape}, {salt.shape}"
        )

    ntime, nsiglay, nobc = temp.shape
    nsiglev = len(siglev)

    if len(obc_nodes) != nobc:
        raise ValueError(
            f"obc_nodes length {len(obc_nodes)} != nobc {nobc}"
        )
    if len(time_mjd) != ntime:
        raise ValueError(
            f"time_mjd length {len(time_mjd)} != ntime {ntime}"
        )
    if len(siglay) != nsiglay:
        raise ValueError(
            f"siglay length {len(siglay)} != nsiglay {nsiglay}"
        )

    # Time representations
    mjd_floor = np.floor(time_mjd).astype(np.int32)
    mjd_ms    = np.round((time_mjd - mjd_floor) * 86400000).astype(np.int32)

    dt64         = MJD_EPOCH + (time_mjd * 86400).astype("int64") * np.timedelta64(1, "s")
    time_str_arr = np.zeros((ntime, 26), dtype="S1")
    for i, t in enumerate(dt64):
        s = str(t).replace("T", " ") + ".000000"
        s = s.replace("-", "/")
        for j, ch in enumerate(s[:26]):
            time_str_arr[i, j] = ch.encode("ascii")

    # siglay / siglev broadcast to (nsiglay, nobc) and (nsiglev, nobc)
    siglay_2d = np.tile(siglay[:, np.newaxis], (1, nobc)).astype(np.float32)
    siglev_2d = np.tile(siglev[:, np.newaxis], (1, nobc)).astype(np.float32)

    with nc4.Dataset(out_path, "w", format="NETCDF3_CLASSIC") as nc:
        # Global attributes
        # FVCOM mod_force.F checks this global attribute literally.
        nc.type    = "FVCOM TIME SERIES OBC TS FILE"
        nc.title   = casename
        nc.history = "File created with fvcom_writer.py — FVCOM preprocessing toolkit"

        # Dimensions
        nc.createDimension("nobc",       nobc)
        nc.createDimension("siglay",     nsiglay)
        nc.createDimension("siglev",     nsiglev)
        nc.createDimension("time",       None)    # UNLIMITED
        nc.createDimension("DateStrLen", 26)

        # obc_nodes
        v           = nc.createVariable("obc_nodes", "i4", ("nobc",))
        v.long_name = "Open Boundary Node Number"
        v.grid      = "obc_grid"
        v[:]        = obc_nodes.astype(np.int32)

        # iint
        v           = nc.createVariable("iint", "i4", ("time",))
        v.long_name = "internal mode iteration number"
        v[:]        = np.arange(1, ntime + 1, dtype=np.int32)

        # time (float MJD)
        v           = nc.createVariable("time", "f4", ("time",))
        v.long_name = "time"
        v.units     = "days since 1858-11-17 00:00:00"
        v.format    = "modified julian day (MJD)"
        v.time_zone = "UTC"
        v[:]        = time_mjd.astype(np.float32)

        # Itime (integer days)
        v           = nc.createVariable("Itime", "i4", ("time",))
        v.units     = "days since 1858-11-17 00:00:00"
        v.format    = "modified julian day (MJD)"
        v.time_zone = "UTC"
        v[:]        = mjd_floor

        # Itime2 (milliseconds since midnight)
        v           = nc.createVariable("Itime2", "i4", ("time",))
        v.units     = "msec since 00:00:00"
        v.time_zone = "UTC"
        v[:]        = mjd_ms

        # Times (character array)
        v           = nc.createVariable("Times", "S1", ("time", "DateStrLen"))
        v.time_zone = "UTC"
        v[:]        = time_str_arr

        # siglay (2-D, broadcast across all OBC nodes)
        v           = nc.createVariable("siglay", "f4", ("siglay", "nobc"))
        v.long_name = "Sigma Layers"
        v.units     = "sigma_layers"
        v[:]        = siglay_2d

        # siglev (2-D)
        v           = nc.createVariable("siglev", "f4", ("siglev", "nobc"))
        v.long_name = "Sigma Levels"
        v.units     = "sigma_levels"
        v[:]        = siglev_2d

        # obc_temp
        v           = nc.createVariable("obc_temp", "f4",
                                        ("time", "siglay", "nobc"))
        v.long_name = "Open Boundary Temperature"
        v.units     = "Celsius"
        v[:]        = temp   # (ntime, nsiglay, nobc)

        # obc_salinity
        v           = nc.createVariable("obc_salinity", "f4",
                                        ("time", "siglay", "nobc"))
        v.long_name = "Open Boundary Salinity"
        v.units     = "PSU"
        v[:]        = salt   # (ntime, nsiglay, nobc)

    print(f"[OK] Written: {out_path}  "
          f"({nobc} OBC nodes, {nsiglay} siglay, {ntime} time steps)")


# ---------------------------------------------------------------------------
# F06 -- River forcing
# ---------------------------------------------------------------------------

def _coerce_tracer(arr: np.ndarray, name: str,
                   ntime: int, nrivers: int) -> np.ndarray:
    """Normalise a tracer array to shape (ntime, nrivers, nclass).

    Accepts 2-D ``(ntime, nrivers)`` (treated as 1 class) or
    3-D ``(ntime, nrivers, nclass)``.  Raises ValueError on mismatch.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim != 3 or arr.shape[:2] != (ntime, nrivers):
        raise ValueError(
            f"Tracer '{name}': expected shape ({ntime}, {nrivers}) or "
            f"({ntime}, {nrivers}, nclass), got {arr.shape}"
        )
    return arr


def write_river_nc(out_path: str | Path,
                   river_names: list[str],
                   time_mjd: np.ndarray,
                   flux: np.ndarray,
                   temp: np.ndarray,
                   salt: np.ndarray,
                   floc: np.ndarray | None = None,
                   plastic: np.ndarray | None = None,
                   extra_tracers: dict | None = None,
                   info1: str = "Delaware River Estuary",
                   info2: str = "Microplastic",
                   casename: str = "waterPACT") -> None:
    """
    Write FVCOM river forcing NetCDF (NETCDF3_CLASSIC).

    Replicates ``write_FVCOM_river.m`` from the FVCOM-toolbox.  Supports
    multiple sediment / microplastic classes and arbitrary extra tracers
    (fine sand, silt, dye, toxics, …) via the ``extra_tracers`` dict.

    Parameters
    ----------
    out_path       : output file path
    river_names    : list of river entry names, e.g. ``['DR_1', ..., 'SR_3']``
    time_mjd       : (ntime,) Modified Julian Days
    flux           : (ntime, nrivers) volume flux [m³/s], positive inflow
    temp           : (ntime, nrivers) river temperature [°C]
    salt           : (ntime, nrivers) river salinity [PSU]
    floc           : (ntime, nrivers) **or** (ntime, nrivers, nclass)
                     Floc / coarse-sand concentration [g/l], optional.
                     Written as ``coarse_sand_1``, ``coarse_sand_2``, …
    plastic        : (ntime, nrivers) **or** (ntime, nrivers, nclass)
                     Microplastic mass concentration [g/l = kg/m^3], optional.
                     Written as ``mp1``, ``mp2``, …
                     Pass a 3-D array for multi-class runs
                     (e.g. scenario_1 with 3 polymer types).
    extra_tracers  : dict mapping a variable-name **prefix** to an array of
                     shape (ntime, nrivers) **or** (ntime, nrivers, nclass).
                     Each prefix is written as ``{prefix}_1``, ``{prefix}_2``, …

                     Built-in prefixes and their FVCOM meaning
                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                     ``'coarse_sand'``  – additional coarse-sand class (same as floc)
                     ``'fine_sand'``    – fine-sand fraction
                     ``'silt'``         – silt fraction
                     ``'mud'``          – mud / cohesive sediment
                     ``'dye'``          – passive tracer / residence-time dye
                     ``'toxic'``        – generic contaminant class
                     Any string key is accepted; FVCOM reads whatever variable
                     names are present in the file.

                     Example — sensitivity run with 2 sediment classes::

                         extra_tracers = {
                             'fine_sand': fine_sand_arr,        # (ntime, nrivers)
                             'silt':      silt_arr,             # (ntime, nrivers)
                         }

                     Example — scenario_1 style (3 MP classes, no floc)::

                         plastic = mp_3class_arr               # (ntime, nrivers, 3)
                         # → writes mp1, mp2, mp3

    info1          : global attribute ``title``
    info2          : global attribute ``info``
    casename       : stored in ``casename`` global attribute
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flux = np.asarray(flux, dtype=np.float32)
    temp = np.asarray(temp, dtype=np.float32)
    salt = np.asarray(salt, dtype=np.float32)
    time_mjd = np.asarray(time_mjd, dtype=np.float64)

    ntime, nrivers = flux.shape
    if len(river_names) != nrivers:
        raise ValueError(
            f"river_names length {len(river_names)} != nrivers {nrivers}"
        )
    if len(time_mjd) != ntime:
        raise ValueError(
            f"time_mjd length {len(time_mjd)} != ntime {ntime}"
        )

    # Normalise optional tracers to (ntime, nrivers, nclass)
    if floc is not None:
        floc = _coerce_tracer(floc, "floc", ntime, nrivers)
    if plastic is not None:
        plastic = _coerce_tracer(plastic, "plastic", ntime, nrivers)
    if extra_tracers is not None:
        extra_tracers = {
            k: _coerce_tracer(v, k, ntime, nrivers)
            for k, v in extra_tracers.items()
        }

    # Time representations
    mjd_floor = np.floor(time_mjd).astype(np.int32)
    mjd_ms    = np.round((time_mjd - mjd_floor) * 86400000).astype(np.int32)

    dt64         = MJD_EPOCH + (time_mjd * 86400).astype("int64") * np.timedelta64(1, "s")
    time_str_arr = np.zeros((ntime, 26), dtype="S1")
    for i, t in enumerate(dt64):
        s = str(t).replace("T", " ") + ".000000"
        s = s.replace("-", "/")
        for j, ch in enumerate(s[:26]):
            time_str_arr[i, j] = ch.encode("ascii")

    with nc4.Dataset(out_path, "w", format="NETCDF3_CLASSIC") as nc:
        # Global attributes
        nc.type    = "FVCOM RIVER FORCING FILE"
        nc.title   = info1
        nc.info    = info2
        nc.history = "File created with fvcom_writer.py — FVCOM preprocessing toolkit"

        # Dimensions
        nc.createDimension("namelen",    80)
        nc.createDimension("rivers",     nrivers)
        nc.createDimension("time",       None)   # UNLIMITED
        nc.createDimension("DateStrLen", 26)

        # river_names  (rivers, namelen) — 80-char padded ASCII
        v = nc.createVariable("river_names", "S1", ("rivers", "namelen"))
        name_arr = np.zeros((nrivers, 80), dtype="S1")
        for r, name in enumerate(river_names):
            padded = f"{name:<80}"[:80]
            for j, ch in enumerate(padded):
                name_arr[r, j] = ch.encode("ascii")
        v[:] = name_arr

        # time (float MJD)
        v           = nc.createVariable("time", "f4", ("time",))
        v.long_name = "time"
        v.units     = "days since 1858-11-17 00:00:00"
        v.format    = "modified julian day (MJD)"
        v.time_zone = "UTC"
        v[:]        = time_mjd.astype(np.float32)

        # Itime (integer days)
        v           = nc.createVariable("Itime", "i4", ("time",))
        v.units     = "days since 1858-11-17 00:00:00"
        v.format    = "modified julian day (MJD)"
        v.time_zone = "UTC"
        v[:]        = mjd_floor

        # Itime2 (milliseconds since midnight)
        v           = nc.createVariable("Itime2", "i4", ("time",))
        v.units     = "msec since 00:00:00"
        v.time_zone = "UTC"
        v[:]        = mjd_ms

        # Times (character array)
        v           = nc.createVariable("Times", "S1", ("time", "DateStrLen"))
        v.time_zone = "UTC"
        v[:]        = time_str_arr

        # river_flux  (time, rivers)
        v           = nc.createVariable("river_flux", "f4", ("time", "rivers"))
        v.long_name = "river runoff volume flux"
        v.units     = "m^3s^-1"
        v[:]        = flux      # (ntime, nrivers)

        # river_temp  (time, rivers)
        v           = nc.createVariable("river_temp", "f4", ("time", "rivers"))
        v.long_name = "river runoff temperature"
        v.units     = "Celsius"
        v[:]        = temp

        # river_salt  (time, rivers)
        v           = nc.createVariable("river_salt", "f4", ("time", "rivers"))
        v.long_name = "river runoff salinity"
        v.units     = "PSU"
        v[:]        = salt

        # Optional: floc → coarse_sand_1, coarse_sand_2, ...  (time, rivers)
        if floc is not None:
            for i in range(floc.shape[2]):
                vname       = f"coarse_sand_{i + 1}"
                v           = nc.createVariable(vname, "f4", ("time", "rivers"))
                v.long_name = f"river runoff coarse sediment class {i + 1}"
                v.units     = "g/l"
                v[:]        = floc[:, :, i]

        # Optional: plastic → mp1, mp2, ...  (time, rivers)
        if plastic is not None:
            for i in range(plastic.shape[2]):
                vname       = f"mp{i + 1}"
                v           = nc.createVariable(vname, "f4", ("time", "rivers"))
                v.long_name = f"river runoff microplastic class {i + 1}"
                # FVCOM mod_force.F rejects "g/l" for plastic river variables.
                # The supplied values are g/L, numerically equal to kg/m^3.
                v.units     = "kg/m^3"
                v[:]        = plastic[:, :, i]

        # Optional: extra_tracers → {prefix}_1, {prefix}_2, ...  (time, rivers)
        if extra_tracers:
            for prefix, arr in extra_tracers.items():
                for i in range(arr.shape[2]):
                    vname       = f"{prefix}_{i + 1}"
                    v           = nc.createVariable(vname, "f4", ("time", "rivers"))
                    v.long_name = f"river runoff {prefix} class {i + 1}"
                    v.units     = "g/l"
                    v[:]        = arr[:, :, i]

    # Build summary of written tracer variables
    tracer_summary = []
    if floc    is not None: tracer_summary.append(f"coarse_sand x{floc.shape[2]}")
    if plastic is not None: tracer_summary.append(f"mp x{plastic.shape[2]}")
    if extra_tracers:
        for k, a in extra_tracers.items():
            tracer_summary.append(f"{k} x{a.shape[2]}")
    tracer_str = ("  tracers: " + ", ".join(tracer_summary)) if tracer_summary else ""
    print(f"[OK] Written: {out_path}  ({nrivers} rivers, {ntime} time steps){tracer_str}")


def write_river_nml(out_path: str | Path,
                    river_names: list[str],
                    river_nodes: list[int],
                    nc_file: str,
                    n_sigma: int = 20,
                    uniform: bool = True) -> None:
    """
    Write the FVCOM river namelist (``.nml``) companion file.

    Replicates ``write_FVCOM_river_nml.m`` output format.
    One ``&NML_RIVER`` block per river entry.

    Parameters
    ----------
    out_path    : output ``.nml`` file path
    river_names : list of river entry names (must match NetCDF ``river_names``)
    river_nodes : 1-based FVCOM node indices for each entry
    nc_file     : basename of the river NetCDF file (no directory path),
                  e.g. ``'waterPACT_riv_floc_MP_2018_2020.nc'``
    n_sigma     : number of FVCOM sigma layers (default 20).
                  Used to build uniform vertical distribution string.
    uniform     : if True (default), write uniform distribution
                  ``n_sigma*{1/n_sigma:.11f}``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(river_names) != len(river_nodes):
        raise ValueError(
            f"river_names length {len(river_names)} != "
            f"river_nodes length {len(river_nodes)}"
        )

    if uniform:
        frac = 1.0 / n_sigma
        vdist = f"{n_sigma}*{frac:.11f}"
    else:
        raise NotImplementedError(
            "Non-uniform vertical distribution not yet supported; "
            "edit the .nml file manually."
        )

    with open(out_path, "w") as fh:
        for name, node in zip(river_names, river_nodes):
            fh.write(" &NML_RIVER\n")
            fh.write(f"  RIVER_NAME          = '{name}',\n")
            fh.write(f"  RIVER_FILE          = '{nc_file}',\n")
            fh.write(f"  RIVER_GRID_LOCATION = {node},\n")
            fh.write(f"  RIVER_VERTICAL_DISTRIBUTION = {vdist}\n")
            fh.write("  /\n")

    print(f"[OK] Written: {out_path}  ({len(river_names)} river entries)")


# ---------------------------------------------------------------------------
# F07 -- Surface forcing (wind + pressure)
# ---------------------------------------------------------------------------

def write_surface_forcing(out_path: str | Path,
                          lat_1d: np.ndarray,
                          lon_1d: np.ndarray,
                          time_dt64: np.ndarray,
                          u10: np.ndarray | None = None,
                          v10: np.ndarray | None = None,
                          pressure: np.ndarray | None = None,
                          vartype: str = "wind",
                          casename: str = "waterPACT") -> None:
    """
    Write FVCOM WRF-style surface forcing file (NETCDF3_CLASSIC).

    Replicates the format of ``waterPACT_DRE_wind_2019.nc`` and
    ``waterPACT_DRE_pres_2019.nc`` produced by the MATLAB ``cfs2fvcom``
    toolbox.  FVCOM reads this file directly and performs its own
    internal interpolation from the regular grid to the unstructured
    mesh — no pre-interpolation to FVCOM nodes is required.

    Parameters
    ----------
    lat_1d    : (nlat,) 1-D latitude array [deg N], monotonically increasing
    lon_1d    : (nlon,) 1-D longitude array [deg E, **-180 to 180** convention],
                stored as XLONG in the output file
    time_dt64 : (ntime,) numpy ``datetime64[s]`` array
    u10       : (ntime, nlat, nlon) float32 — eastward 10-m wind [m/s]
                Required when ``vartype='wind'``.
    v10       : (ntime, nlat, nlon) float32 — northward 10-m wind [m/s]
                Required when ``vartype='wind'``.
    pressure  : (ntime, nlat, nlon) float32 — surface air pressure [Pa]
                Required when ``vartype='pressure'``.
    vartype   : ``'wind'`` — writes ``U10`` + ``V10``
                ``'pressure'`` — writes ``air_pressure``
    casename  : stored in the ``title`` global attribute (unused by FVCOM
                at runtime but useful for provenance)

    Notes
    -----
    * Output dimensions: ``south_north`` × ``west_east`` × ``time`` (UNLIMITED)
    * Coordinate variables: ``XLAT`` (south_north, west_east),
      ``XLONG`` (south_north, west_east)
    * Time variable: ``Times`` character array (time, DateStrLen=26)
      in ISO 8601 format ``YYYY-MM-DDTHH:MM:SS.000000``
    * FVCOM NML requirements: ``WIND_TYPE = 'speed'`` for the wind file;
      ``AIRPRESSURE_ON = T`` for the pressure file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lat_1d    = np.asarray(lat_1d,    dtype=np.float32).ravel()
    lon_1d    = np.asarray(lon_1d,    dtype=np.float32).ravel()
    time_dt64 = np.asarray(time_dt64, dtype="datetime64[s]")

    nlat  = len(lat_1d)
    nlon  = len(lon_1d)
    ntime = len(time_dt64)

    # 2-D coordinate meshes: XLAT (nlat, nlon), XLONG (nlat, nlon).
    # FVCOM's spherical bilinear interpolator expects gridded longitudes in
    # 0..360, even when the model grid itself is in western-hemisphere degrees.
    lon_1d_fvcom = np.mod(lon_1d, 360.0)
    xlat_2d  = np.tile(lat_1d[:, np.newaxis], (1, nlon))   # (nlat, nlon)
    xlong_2d = np.tile(lon_1d_fvcom[np.newaxis, :], (nlat, 1))   # (nlat, nlon)

    # Times character array  "YYYY-MM-DDTHH:MM:SS.000000"  (26 chars)
    time_str_arr = np.zeros((ntime, 26), dtype="S1")
    for i, t in enumerate(time_dt64):
        s = str(t) + ".000000"   # 19 + 7 = 26 chars
        for j, ch in enumerate(s[:26]):
            time_str_arr[i, j] = ch.encode("ascii")

    # Global attribute strings
    start_str = str(time_dt64[0]).replace("T", " ")
    end_str   = str(time_dt64[-1]).replace("T", " ")

    with nc4.Dataset(out_path, "w", format="NETCDF3_CLASSIC") as ds:
        # Global attributes
        ds.type        = "FVCOM METEO FORCING FILE"
        ds.title       = "CFS model forcing"
        ds.history     = "File created with fvcom_writer.py — FVCOM preprocessing toolkit"
        ds.source      = "wrf grid (structured) surface forcing"
        ds.START_DATE  = start_str
        ds.END_DATE    = end_str
        ds.Conventions = "CF-1.0"

        # Dimensions
        ds.createDimension("south_north", nlat)
        ds.createDimension("west_east",   nlon)
        ds.createDimension("time",        None)   # UNLIMITED
        ds.createDimension("DateStrLen",  26)

        # XLAT
        v             = ds.createVariable("XLAT",  "f4", ("south_north", "west_east"))
        v.long_name   = "latitude"
        v.description = "LATITUDE, SOUTH IS NEGATIVE"
        v.units       = "degrees_north"
        v.type        = "data"
        v[:]          = xlat_2d

        # XLONG
        v             = ds.createVariable("XLONG", "f4", ("south_north", "west_east"))
        v.long_name   = "longitude"
        v.description = "LONGITUDE, 0-360 DEGREES EAST"
        v.units       = "degrees_east"
        v.type        = "data"
        v[:]          = xlong_2d

        # Times
        v           = ds.createVariable("Times", "S1", ("time", "DateStrLen"))
        v.time_zone = "UTC"
        v[:]        = time_str_arr

        if vartype == "wind":
            if u10 is None or v10 is None:
                raise ValueError("u10 and v10 are required for vartype='wind'")
            u10_arr = np.asarray(u10, dtype=np.float32)
            v10_arr = np.asarray(v10, dtype=np.float32)
            if u10_arr.shape != (ntime, nlat, nlon):
                raise ValueError(
                    f"u10 shape {u10_arr.shape} != expected ({ntime}, {nlat}, {nlon})"
                )
            vu             = ds.createVariable("U10", "f4", ("time", "south_north", "west_east"))
            vu.long_name   = "Eastward Wind Velocity"
            vu.description = "U at 10 M"
            vu.units       = "m s-1"
            vu.grid        = "wrf_grid"
            vu.type        = "data"
            vu[:]          = u10_arr
            vv             = ds.createVariable("V10", "f4", ("time", "south_north", "west_east"))
            vv.long_name   = "Northward Wind Velocity"
            vv.description = "V at 10 M"
            vv.units       = "m s-1"
            vv.grid        = "wrf_grid"
            vv.type        = "data"
            vv[:]          = v10_arr
            print(f"[OK] Written: {out_path}  (vartype=wind, {ntime} steps, {nlat}×{nlon} grid)")

        elif vartype == "pressure":
            if pressure is None:
                raise ValueError("pressure is required for vartype='pressure'")
            pres_arr = np.asarray(pressure, dtype=np.float32)
            if pres_arr.shape != (ntime, nlat, nlon):
                raise ValueError(
                    f"pressure shape {pres_arr.shape} != expected ({ntime}, {nlat}, {nlon})"
                )
            vp             = ds.createVariable("air_pressure", "f4", ("time", "south_north", "west_east"))
            vp.long_name   = "Air Pressure"
            vp.description = "Sea surface airpressure"
            vp.units       = "Pa"
            vp.grid        = "wrf_grid"
            vp.coordinates = "lat lon"
            vp.type        = "data"
            vp[:]          = pres_arr
            print(f"[OK] Written: {out_path}  (vartype=pressure, {ntime} steps, {nlat}×{nlon} grid)")

        else:
            raise ValueError(f"vartype must be 'wind' or 'pressure'; got '{vartype}'")


# ---------------------------------------------------------------------------
# F08 -- Initial temperature/salinity
# ---------------------------------------------------------------------------

def write_initial_ts(out_path: str | Path,
                     time_mjd: float,
                     temp,
                     salt,
                     zsl: np.ndarray,
                     nnode: int,
                     casename: str = "waterPACT") -> None:
    """
    Write FVCOM initial T/S file on standard z-levels.

    Replicates the structure of ``waterPACT_DRE_its.nc``:
      dims: time(1, unlimited), ksl(30), node(68416), DateStrLen(26)
      vars: time, Itime, Itime2, Times, zsl(ksl), tsl(time,ksl,node), ssl(time,ksl,node)

    FVCOM reads this file when ``STARTUP_TS_TYPE = 'observed'`` and
    internally interpolates from z-levels to its sigma coordinate.

    Parameters
    ----------
    out_path : output NetCDF file path
    time_mjd : float — single Modified Julian Day for the initialization time
    temp     : (ksl, nnode) float32 array, OR scalar float for constant fill
    salt     : (ksl, nnode) float32 array, OR scalar float for constant fill
    zsl      : (ksl,) float32 — standard z-levels [m, negative-up, e.g. 0, -5, -10, ...]
    nnode    : number of FVCOM nodes (required when temp/salt is scalar)
    casename : FVCOM case name string
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ksl = len(zsl)

    # Handle scalar temp/salt → broadcast to (ksl, nnode)
    if np.isscalar(temp):
        temp_arr = np.full((ksl, nnode), float(temp), dtype=np.float32)
    else:
        temp_arr = np.asarray(temp, dtype=np.float32)
        if temp_arr.shape != (ksl, nnode):
            raise ValueError(
                f"temp shape {temp_arr.shape} != expected ({ksl}, {nnode})"
            )

    if np.isscalar(salt):
        salt_arr = np.full((ksl, nnode), float(salt), dtype=np.float32)
    else:
        salt_arr = np.asarray(salt, dtype=np.float32)
        if salt_arr.shape != (ksl, nnode):
            raise ValueError(
                f"salt shape {salt_arr.shape} != expected ({ksl}, {nnode})"
            )

    # Time encoding (single time step)
    mjd = float(time_mjd)
    mjd_floor = int(np.floor(mjd))
    mjd_ms = int(np.round((mjd - mjd_floor) * 86400000))

    # Time string
    dt64 = MJD_EPOCH + np.timedelta64(int(mjd * 86400), "s")
    s = str(dt64).replace("T", " ") + ".000000"
    s = s.replace("-", "/")
    time_str_arr = np.zeros((1, 26), dtype="S1")
    for j, ch in enumerate(s[:26]):
        time_str_arr[0, j] = ch.encode("ascii")

    with nc4.Dataset(out_path, "w", format="NETCDF3_CLASSIC") as nc:
        # Global attributes
        nc.title = "FVCOM Initial File"
        nc.history = ("File created with fvcom_writer.write_initial_ts — "
                      "FVCOM preprocessing toolkit")

        # Dimensions
        nc.createDimension("time", None)       # UNLIMITED
        nc.createDimension("ksl", ksl)
        nc.createDimension("node", nnode)
        nc.createDimension("DateStrLen", 26)

        # time (float MJD)
        v = nc.createVariable("time", "f4", ("time",))
        v.long_name = "time"
        v.units = "days since 0.0"
        v.time_zone = "none"
        v[:] = np.array([mjd], dtype=np.float32)

        # Itime
        v = nc.createVariable("Itime", "i4", ("time",))
        v.units = "days since 0.0"
        v.time_zone = "none"
        v[:] = np.array([mjd_floor], dtype=np.int32)

        # Itime2
        v = nc.createVariable("Itime2", "i4", ("time",))
        v.units = "msec since 00:00:00"
        v.time_zone = "none"
        v[:] = np.array([mjd_ms], dtype=np.int32)

        # Times
        v = nc.createVariable("Times", "S1", ("time", "DateStrLen"))
        v.time_zone = "UTC"
        v[:] = time_str_arr

        # zsl — standard z-levels
        v = nc.createVariable("zsl", "f4", ("ksl",))
        v.long_name = "standard z levels positive up"
        v.units = "m"
        v[:] = zsl.astype(np.float32)

        # ssl — initial salinity
        v = nc.createVariable("ssl", "f4", ("time", "ksl", "node"))
        v.long_name = "observed_salinity_profile"
        v.units = "PSU"
        v[0, :, :] = salt_arr

        # tsl — initial temperature
        v = nc.createVariable("tsl", "f4", ("time", "ksl", "node"))
        v.long_name = "observed_temperature_profile"
        v.units = "C"
        v[0, :, :] = temp_arr

    print(f"[OK] Written: {out_path}  (ksl={ksl}, node={nnode}, time_mjd={mjd:.2f})")
