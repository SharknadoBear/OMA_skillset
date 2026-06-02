"""
tpxo_tides.py
=============
TPXO v9.0 tidal harmonic loading, interpolation, and signal reconstruction.

Functions
---------
load_tpxo9                  : load amplitude/phase from h_tpxo9.v5a.nc
interp_tpxo_to_nodes        : bilinear complex-phasor interpolation to target lon/lat
reconstruct_tidal_signal    : single-node tidal reconstruction via UTide
reconstruct_tidal_all_nodes : vectorized batch reconstruction for all OBC nodes

Notes
-----
All longitudes use the [0, 360] convention internally; negative-longitude
inputs are converted automatically.

Phase interpolation uses complex phasors (A·cos(hp), A·sin(hp)) rather
than raw degrees to avoid the 0°/360° discontinuity artifact that afflicts
direct bilinear interpolation of the raw phase field.

TPXO download : https://www.tpxo.net/global  (free academic registration)
UTide source  : https://github.com/wesleybowman/UTide
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import netCDF4 as nc4
from scipy.interpolate import RegularGridInterpolator
try:
    from .grid_utils import datetime64_to_mjd, mjd_to_datetime64
except ImportError:
    from grid_utils import datetime64_to_mjd, mjd_to_datetime64


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TPXO_H_FILE = "h_tpxo9.v5a.nc"

# Doodson angular frequencies (degrees/hour) for all 22 TPXO9 v5a constituents.
# Used as fallback frequency source when UTide database lookup is unavailable.
CONST_FREQ_DEG_HR: dict[str, float] = {
    "M2":  28.9841042,
    "S2":  30.0000000,
    "N2":  28.4397295,
    "K2":  30.0821373,
    "2N2": 27.8953548,
    "MU2": 27.9682084,
    "NU2": 28.5125831,
    "L2":  29.5284789,
    "K1":  15.0410686,
    "O1":  13.9430356,
    "P1":  14.9589314,
    "Q1":  13.3986609,
    "2Q1": 12.8542862,
    "J1":  15.5854433,
    "OO1": 16.1391017,
    "S1":  15.0000000,
    "MF":   1.0980331,
    "MM":   0.5443747,
    "M4":  57.9682084,
    "MN4": 57.4238337,
    "MS4": 58.9841042,
    "M3":  43.4761563,
}


# ===========================================================================
# 1. TPXO loading
# ===========================================================================

def load_tpxo9(tpxo_dir: str | Path) -> dict:
    """
    Load harmonic amplitude and phase from the TPXO v9.0 elevation file.

    Parameters
    ----------
    tpxo_dir : directory containing h_tpxo9.v5a.nc.

    Returns
    -------
    dict with keys:
        lon     : (nlon,) float64 in [0, 360]
        lat     : (nlat,) float64 in [-90, 90]
        ha_m    : (ncon, nlat, nlon) float64 — amplitude in metres
        hp_deg  : (ncon, nlat, nlon) float64 — phase in degrees (Greenwich)
        names   : list of str — constituent names, upper-case
        nc_path : str — path of the file that was read
    """
    nc_path = Path(tpxo_dir) / TPXO_H_FILE
    if not nc_path.exists():
        raise FileNotFoundError(
            f"TPXO v9.0 file not found: {nc_path}\n"
            f"Download h_tpxo9.v5a.nc from https://www.tpxo.net/global "
            f"(free academic registration required)."
        )

    with nc4.Dataset(nc_path) as ds:
        lon_z = np.array(ds["lon_z"][:])
        lat_z = np.array(ds["lat_z"][:])
        con_raw = ds["con"][:]
        names = []
        for row in con_raw:
            try:
                s = row.tobytes().decode("utf-8", errors="ignore").strip().upper()
            except AttributeError:
                s = "".join(
                    c.decode("utf-8", errors="ignore") for c in row
                ).strip().upper()
            names.append(s)
        ha_raw = np.array(ds["ha"][:])
        hp_raw = np.array(ds["hp"][:])

    # Shape normalisation → (ncon, nlat, nlon)
    ncon = len(names)
    if ha_raw.ndim == 3:
        if ha_raw.shape[0] == ncon:
            ha, hp = ha_raw, hp_raw
        elif ha_raw.shape[2] == ncon:
            ha = ha_raw.transpose(2, 0, 1)
            hp = hp_raw.transpose(2, 0, 1)
        else:
            raise ValueError(
                f"Unexpected ha shape {ha_raw.shape} for {ncon} constituents."
            )
    else:
        raise ValueError(f"Expected 3-D ha array, got shape {ha_raw.shape}.")

    # Unit detection: netCDF4 auto-applies scale_factor; TPXO9 values end up
    # in metres (M2 global max ~8 m).  Values > 50 indicate mm storage.
    m2_idx = next((i for i, n in enumerate(names) if n.strip() == "M2"), 0)
    max_ha_m2 = float(np.nanmax(np.abs(ha[m2_idx])))
    if max_ha_m2 > 50.0:
        ha_m = ha / 1000.0
        print(f"[TPXO] ha in mm (M2 max={max_ha_m2:.1f}); converting to metres.")
    else:
        ha_m = ha.astype(float)
        print(f"[TPXO] ha in metres (M2 max={max_ha_m2:.4f} m).")

    ha_m[ha_m < -100] = np.nan
    hp_deg = np.where(np.abs(hp_raw) > 900, np.nan, hp_raw.astype(float))

    # Extract 1-D coordinate vectors — TPXO v9 stores lon_z/lat_z as 2-D
    # meshgrids (nlon, nlat).  Extract column/row slices to get 1-D arrays.
    lon_z = np.asarray(lon_z)
    lat_z = np.asarray(lat_z)
    dim1, dim2 = ha_m.shape[1], ha_m.shape[2]
    if lon_z.ndim == 2:
        lon_1d = lon_z[:, 0].ravel()
        lat_1d = lat_z[0, :].ravel()
    else:
        lon_1d = lon_z.ravel()
        lat_1d = lat_z.ravel()

    nlon_c, nlat_c = len(lon_1d), len(lat_1d)
    if dim1 == nlon_c and dim2 == nlat_c:
        ha_m   = ha_m.transpose(0, 2, 1)
        hp_deg = hp_deg.transpose(0, 2, 1)
        print(f"[TPXO] Transposed ha: (ncon, nlon={dim1}, nlat={dim2}) "
              f"\u2192 (ncon, nlat={dim2}, nlon={dim1})")

    print(f"[TPXO] Grid: {nlon_c} lon \u00d7 {nlat_c} lat  "
          f"(lon {lon_1d[0]:.2f}\u2013{lon_1d[-1]:.2f}, "
          f"lat {lat_1d[0]:.2f}\u2013{lat_1d[-1]:.2f})")
    return {
        "lon":     lon_1d,
        "lat":     lat_1d,
        "ha_m":    ha_m,
        "hp_deg":  hp_deg,
        "names":   names,
        "nc_path": str(nc_path),
    }


# ===========================================================================
# 2. Harmonic interpolation to target nodes
# ===========================================================================

def interp_tpxo_to_nodes(tpxo: dict,
                          target_lon: np.ndarray,
                          target_lat: np.ndarray,
                          constituents: list[str] | None = None) -> dict:
    """
    Bilinear interpolation of TPXO harmonics to target (lon, lat) positions.

    Uses complex-phasor interpolation — interpolates A·cos(phase) and
    A·sin(phase) separately — to avoid wrap-around artifacts at the 0°/360°
    discontinuity in the raw TPXO phase field.

    Parameters
    ----------
    tpxo         : output of load_tpxo9()
    target_lon   : (nnodes,) target longitudes [degrees, any convention]
    target_lat   : (nnodes,) target latitudes [degrees N]
    constituents : constituent names to use; None = all in file

    Returns
    -------
    dict with keys:
        amp   : (ncon, nnodes) float64 — interpolated amplitudes [m]
        phase : (ncon, nnodes) float64 — interpolated phases [deg, Greenwich]
        names : list of str — constituent names used
    """
    lon_tpxo  = tpxo["lon"]
    lat_tpxo  = tpxo["lat"]
    ha_m      = tpxo["ha_m"]
    hp_deg    = tpxo["hp_deg"]
    all_names = tpxo["names"]

    # Normalise target longitudes to [0, 360]
    tgt_lon_360 = np.where(target_lon < 0, target_lon + 360.0, target_lon)

    if constituents is None:
        use_idx   = list(range(len(all_names)))
        use_names = all_names
    else:
        constituents_up = [c.strip().upper() for c in constituents]
        use_idx   = [i for i, n in enumerate(all_names) if n in constituents_up]
        use_names = [all_names[i] for i in use_idx]
        missing   = set(constituents_up) - set(use_names)
        if missing:
            print(f"[TPXO] Constituents not found in file: {missing}")

    ncon   = len(use_idx)
    nnodes = len(target_lon)
    amp_out   = np.full((ncon, nnodes), np.nan)
    phase_out = np.full((ncon, nnodes), np.nan)

    query_pts = np.column_stack([target_lat, tgt_lon_360])

    for k, ci in enumerate(use_idx):
        ha_i = np.where(np.isnan(ha_m[ci]),    0.0, ha_m[ci])
        hp_i = np.where(np.isnan(hp_deg[ci]),  0.0, hp_deg[ci])
        re_part = ha_i * np.cos(np.radians(hp_i))
        im_part = ha_i * np.sin(np.radians(hp_i))

        re_out = RegularGridInterpolator(
            (lat_tpxo, lon_tpxo), re_part,
            method="linear", bounds_error=False, fill_value=np.nan
        )(query_pts)
        im_out = RegularGridInterpolator(
            (lat_tpxo, lon_tpxo), im_part,
            method="linear", bounds_error=False, fill_value=np.nan
        )(query_pts)

        amp_out[k]   = np.hypot(re_out, im_out)
        phase_out[k] = np.degrees(np.arctan2(im_out, re_out)) % 360.0

    return {"amp": amp_out, "phase": phase_out, "names": use_names}


# ===========================================================================
# 3. Tidal signal reconstruction
# ===========================================================================

def _mjd_to_utide_time(time_mjd: np.ndarray) -> np.ndarray:
    """Convert MJD float64 array to UTide gregorian days (days since 0000-12-31)."""
    from utide._time_conversion import _python_gregorian_datenum
    return _python_gregorian_datenum(mjd_to_datetime64(time_mjd))


def reconstruct_tidal_signal(amp_m: np.ndarray,
                              phase_deg: np.ndarray,
                              names: list[str],
                              time_mjd: np.ndarray,
                              lat: float) -> np.ndarray:
    """
    Reconstruct a tidal elevation time series from harmonic constants.

    Strategy 1 (primary): UTide — builds a complete coef structure matching
        utide.solve() output and calls utide.reconstruct().  Mid-point nodal
        corrections (nodsatlint=True) match the T-tide 'nodal' convention.
    Strategy 2 (fallback): pyTMD OTIS nodal corrections.
    Strategy 3 (last resort): simple harmonic sum, OTIS epoch, no nodal.

    Parameters
    ----------
    amp_m     : (ncon,) amplitude [metres]
    phase_deg : (ncon,) Greenwich phase lag [degrees]
    names     : constituent name strings (any case)
    time_mjd  : (ntime,) Modified Julian Days
    lat       : node latitude [degrees N]

    Returns
    -------
    eta : (ntime,) tidal elevation [metres]
    """
    names_up    = [n.strip().upper() for n in names]
    valid       = ~(np.isnan(amp_m) | np.isnan(phase_deg))
    v_amp       = amp_m[valid]
    v_phase     = phase_deg[valid]
    v_names_all = np.array(names_up)[valid]

    t_dt = mjd_to_datetime64(time_mjd)

    # ------------------------------------------------------------------
    # Strategy 1: UTide
    # ------------------------------------------------------------------
    try:
        import utide
        from utide._ut_constants import constit_index_dict
        from utide.constituent_selection import linearized_freqs
        from utide._time_conversion import _python_gregorian_datenum
        from utide.utilities import Bunch

        t_utide = _python_gregorian_datenum(t_dt)
        reftime = float(np.mean(t_utide))

        idx_keep  = [i for i, n in enumerate(v_names_all) if n in constit_index_dict]
        if not idx_keep:
            raise ValueError("No constituent names found in UTide database.")

        v_amp_k   = v_amp[idx_keep]
        v_phase_k = v_phase[idx_keep]
        v_names_k = v_names_all[idx_keep]
        v_lind_k  = np.array([constit_index_dict[n] for n in v_names_k], dtype=int)
        v_frq_k   = linearized_freqs(reftime)[v_lind_k]

        coef = Bunch(
            name = v_names_k,
            A    = v_amp_k.astype(float),
            g    = v_phase_k.astype(float),
            A_ci = np.zeros(len(v_amp_k)),
            g_ci = np.zeros(len(v_amp_k)),
            mean = 0.0,
            aux  = Bunch(
                lat     = float(lat),
                frq     = v_frq_k,
                lind    = v_lind_k,
                reftime = reftime,
                opt     = Bunch(
                    twodim     = False,
                    nodsatlint = True,
                    nodsatnone = False,
                    gwchlint   = False,
                    gwchnone   = False,
                    prefilt    = [],
                    notrend    = True,
                    nodiagn    = True,
                    conf_int   = False,
                ),
            ),
        )

        result = utide.reconstruct(t_dt, coef, verbose=False, min_SNR=0, min_PE=0)
        return np.asarray(result.h, dtype=float)

    except Exception as e:
        print(f"[WARN] UTide reconstruction failed ({e}); trying pyTMD fallback.")

    # ------------------------------------------------------------------
    # Strategy 2: pyTMD OTIS nodal corrections
    # ------------------------------------------------------------------
    try:
        import pyTMD.constituents
        t_mid_mjd  = np.array([np.mean(time_mjd)])
        cons_lower = [n.lower() for n in v_names_all]
        args       = pyTMD.constituents.arguments(
            t_mid_mjd, cons_lower, corrections="OTIS"
        )
        vu_mid = np.asarray(args[1]).ravel()
        F_mid  = np.ones(len(v_names_all))
        try:
            F_mid = np.asarray(
                pyTMD.constituents.nodal_modulation(
                    t_mid_mjd, cons_lower, corrections="OTIS"
                )
            ).ravel()
        except Exception:
            pass

        OTIS_EPOCH_MJD = 15019.0
        t_1900h  = (time_mjd - OTIS_EPOCH_MJD) * 24.0
        v_frq_fb = np.array(
            [CONST_FREQ_DEG_HR.get(n, np.nan) / 360.0 for n in v_names_all]
        )
        eta = np.zeros(len(time_mjd))
        for i in range(len(v_names_all)):
            eta += (F_mid[i] * v_amp[i]
                    * np.cos(2 * np.pi * v_frq_fb[i] * t_1900h
                             + vu_mid[i] - np.radians(v_phase[i])))
        return eta

    except Exception as e:
        print(f"[WARN] pyTMD fallback failed ({e}); using simple harmonic sum.")

    # ------------------------------------------------------------------
    # Strategy 3: simple harmonic sum, OTIS epoch, no nodal corrections
    # ------------------------------------------------------------------
    print("[WARNING] Using simple harmonic sum without nodal corrections.")
    OTIS_EPOCH_MJD = 15019.0
    t_1900h  = (time_mjd - OTIS_EPOCH_MJD) * 24.0
    v_frq_fb = np.array(
        [CONST_FREQ_DEG_HR.get(n, np.nan) / 360.0 for n in v_names_all]
    )
    eta = np.zeros(len(time_mjd))
    for i in range(len(v_names_all)):
        eta += v_amp[i] * np.cos(
            2 * np.pi * v_frq_fb[i] * t_1900h - np.radians(v_phase[i])
        )
    return eta


def reconstruct_tidal_all_nodes(
    amp_m: np.ndarray,
    phase_deg: np.ndarray,
    names: list[str],
    time_mjd: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """
    Vectorized tidal reconstruction for all nodes simultaneously.

    Computes the UTide basis matrix E(t) once using mid-point nodal corrections,
    then recovers all node time series in a single matrix multiply — ~200×
    faster than calling reconstruct_tidal_signal() in a loop.

    Parameters
    ----------
    amp_m     : (ncon, nnodes) amplitudes [metres]
    phase_deg : (ncon, nnodes) Greenwich phase lags [degrees]
    names     : ncon constituent name strings
    time_mjd  : (ntime,) Modified Julian Days
    lats      : (nnodes,) node latitudes [degrees N]

    Returns
    -------
    eta : (nnodes, ntime) float32 tidal elevation [metres]
    """
    from utide._ut_constants import constit_index_dict
    from utide.constituent_selection import linearized_freqs
    from utide._time_conversion import _python_gregorian_datenum
    from utide.harmonics import ut_E

    names_up = [n.strip().upper() for n in names]
    t_utide  = _python_gregorian_datenum(mjd_to_datetime64(time_mjd))
    reftime  = float(np.mean(t_utide))

    keep_idx = [i for i, n in enumerate(names_up) if n in constit_index_dict]
    names_k  = [names_up[i] for i in keep_idx]
    lind_k   = np.array([constit_index_dict[n] for n in names_k], dtype=int)
    frq_k    = linearized_freqs(reftime)[lind_k]

    amp_k   = np.where(np.isnan(amp_m[keep_idx, :]),     0.0, amp_m[keep_idx, :])    # (nc, nobc)
    phase_k = np.where(np.isnan(phase_deg[keep_idx, :]), 0.0, phase_deg[keep_idx, :])  # (nc, nobc)

    mean_lat = float(np.mean(lats))
    E = ut_E(t_utide, reftime, frq_k, lind_k, mean_lat,
             [True, False, False, False], [])   # (nt, nc)

    rpd = np.pi / 180.0
    AP  = 0.5 * amp_k * np.exp(-1j * phase_k * rpd)   # (nc, nobc)

    FIT = np.dot(E, AP) + np.dot(np.conj(E), np.conj(AP))
    return np.real(FIT).T.astype(np.float32)   # (nobc, nt)
