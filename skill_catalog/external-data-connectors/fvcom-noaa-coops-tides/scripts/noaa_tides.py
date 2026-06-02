"""
noaa_tides.py
=============
Fetch observations from the NOAA CO-OPS API.

Forcing components
------------------
  F04 -- water level (6-minute) at Chesapeake City (C&D Canal OBC)
  F05 -- salinity + water temperature (hourly) at Chesapeake City (canal T/S OBC)

Reference workflow
------------------
Old MATLAB:
  data_source/tide_chesapeake_city_YYYY.mat   -- pre-downloaded annual files
  d_boundary_forcing.m                        -- loads mat, applies to canal BC
  t_tide_const.mat                            -- harmonic constants from t_tide

Public functions
----------------
  fetch_noaa_waterlevel()      : 6-min water level -> pandas DataFrame + CSV cache
  fetch_noaa_ts()              : hourly salinity / temperature -> numpy dict + CSV cache
  get_station_info()           : station metadata (name, lat, lon, state)
  harmonics_from_record()      : utide harmonic analysis
  reconstruct_from_harmonics() : tidal reconstruction
  resample_to_fvcom_time()     : interpolate onto FVCOM MJD grid

Python dependencies
-------------------
  requests, pandas, numpy, utide

Station
-------
  NOAA CO-OPS station 8573927 -- Chesapeake City, MD (C&D Canal, active gauge)
  Datum: MSL (water level) / MLLW (salinity, temperature)
  Products: water_level (6-min), salinity (hourly), water_temperature (hourly)
"""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import requests


COOPS_API_BASE          = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
CHESAPEAKE_CITY_STATION = "8573927"
STATION_LAT             = 39.5267   # degrees N

# Maximum gap size that will be linearly interpolated (number of 6-min steps)
_MAX_GAP_STEPS = 30   # 30 × 6 min = 3 hours

# NOAA quality flags that indicate bad/missing data (comma-separated in the 'f' field)
_BAD_FLAGS = {"4", "8"}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _month_chunks(t_start: str, t_end: str):
    """
    Yield (begin_str, end_str) pairs covering [t_start, t_end] month-by-month.

    Each begin/end is formatted as 'YYYYMMDD HH:MM' for the CO-OPS API.
    The CO-OPS API limits 6-minute water_level requests to 31 days.
    """
    cur = pd.Timestamp(t_start).normalize()
    end = pd.Timestamp(t_end).normalize()
    while cur <= end:
        # last moment of the current month
        month_end = cur + pd.offsets.MonthEnd(0)
        chunk_end = min(month_end, end)
        yield (cur.strftime("%Y%m%d 00:00"),
               chunk_end.strftime("%Y%m%d 23:59"))
        cur = chunk_end + pd.Timedelta(days=1)


def _fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    In-place quality control and gap-filling on a CO-OPS DataFrame.

    Steps
    -----
    1. Replace any sample whose 'flags' field contains a bad-quality code
       (NOAA codes "4" = questionable, "8" = bad) with NaN.
    2. Linearly interpolate gaps ≤ _MAX_GAP_STEPS (3 hours).
    3. Gaps larger than _MAX_GAP_STEPS are left as NaN and a warning is printed.
    """
    # Step 1: flag-based masking
    def _has_bad_flag(f_str):
        if pd.isna(f_str):
            return False
        return bool(_BAD_FLAGS & set(str(f_str).split(",")))

    bad_mask = df["flags"].apply(_has_bad_flag)
    n_flagged = bad_mask.sum()
    if n_flagged:
        df.loc[bad_mask, "water_level"] = np.nan
        print(f"  [fill_gaps] flagged {n_flagged} samples as bad (codes 4/8)")

    # Step 2+3: identify NaN runs
    is_nan = df["water_level"].isna()
    if not is_nan.any():
        return df

    # Label contiguous NaN blocks
    change = is_nan != is_nan.shift()
    block_id = change.cumsum()
    nan_blocks = df[is_nan].groupby(block_id[is_nan])

    for bid, grp in nan_blocks:
        n = len(grp)
        i0 = grp.index[0]
        i1 = grp.index[-1]
        if n <= _MAX_GAP_STEPS:
            pass   # will be handled by interpolate() below
        else:
            t0 = df.loc[i0, "time"]
            t1 = df.loc[i1, "time"]
            print(f"  [fill_gaps] WARNING: gap of {n} steps ({n*6} min) "
                  f"from {t0} to {t1} — left as NaN")

    # Linear interpolation for all gaps; limit= caps it at _MAX_GAP_STEPS
    df["water_level"] = (df["water_level"]
                         .interpolate(method="linear", limit=_MAX_GAP_STEPS))
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_noaa_waterlevel(station_id: str,
                          t_start: str,
                          t_end: str,
                          cache_dir: str | Path,
                          product: str = "water_level",
                          datum: str = "MSL",
                          time_zone: str = "GMT") -> pd.DataFrame:
    """
    Fetch 6-minute water-level data from NOAA CO-OPS web API.

    Parameters
    ----------
    station_id : NOAA CO-OPS station number, e.g. '8573927'
    t_start    : ISO date string, e.g. '2018-01-01'
    t_end      : ISO date string, e.g. '2020-12-31'
    cache_dir  : directory for per-month CSV cache files
    product    : 'water_level' (verified) or 'predictions' (harmonic)
    datum      : vertical datum -- 'MSL' recommended for FVCOM compatibility
    time_zone  : 'GMT' (always use UTC)

    Returns
    -------
    pandas.DataFrame with columns:
        time         -- datetime64[ns, UTC]
        water_level  -- float64  [m], NaN where data unavailable / gap > 3 hr
        sigma        -- float64  [m], measurement uncertainty
        flags        -- str,     raw NOAA quality flags

    Notes
    -----
    * The CO-OPS API limits 6-minute water_level to 31-day windows.
      This function chunks automatically by calendar month.
    * Per-month CSVs are cached in ``cache_dir`` as
      ``noaa_{station_id}_{YYYYMM}.csv``.  Re-download is skipped if the
      file exists.
    * A 0.5-second pause is inserted between successive API calls to avoid
      throttling (mirrors MATLAB tena_coops_getAPIdata pause(0.5)).
    * Gaps ≤ 3 hours are linearly interpolated; larger gaps are left as NaN.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunks = list(_month_chunks(t_start, t_end))
    print(f"Fetching {len(chunks)} monthly chunks for station {station_id} "
          f"({t_start} → {t_end})")

    all_frames = []
    for i, (begin, end) in enumerate(chunks):
        yyyymm = begin[:6].replace(" ", "")[:6]   # 'YYYYMM'
        cache_file = cache_dir / f"noaa_{station_id}_{yyyymm}.csv"

        if cache_file.exists():
            df_chunk = pd.read_csv(cache_file, parse_dates=["time"])
            df_chunk["time"] = pd.to_datetime(df_chunk["time"], utc=True)
            all_frames.append(df_chunk)
            print(f"  [{i+1:02d}/{len(chunks)}] {yyyymm}  (cached, "
                  f"{len(df_chunk)} rows)")
            continue

        # Build request URL
        params = {
            "begin_date":   begin,
            "end_date":     end,
            "station":      station_id,
            "product":      product,
            "datum":        datum,
            "units":        "metric",
            "time_zone":    time_zone.lower(),
            "application":  "WaterPACT_DRE",
            "format":       "json",
        }
        resp = requests.get(COOPS_API_BASE, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if "error" in payload:
            msg = payload["error"].get("message", str(payload["error"]))
            raise RuntimeError(
                f"CO-OPS API error for {station_id} {yyyymm}: {msg}")

        records = payload.get("data", [])
        if not records:
            print(f"  [{i+1:02d}/{len(chunks)}] {yyyymm}  WARNING: empty response")
            _time.sleep(0.5)
            continue

        df_chunk = pd.DataFrame({
            "time":        [r["t"] for r in records],
            "water_level": [float(r["v"]) if r["v"] not in ("", None)
                            else np.nan for r in records],
            "sigma":       [float(r["s"]) if r.get("s") not in ("", None)
                            else np.nan for r in records],
            "flags":       [r.get("f", "") for r in records],
        })
        df_chunk["time"] = pd.to_datetime(df_chunk["time"],
                                          format="%Y-%m-%d %H:%M",
                                          utc=True)

        df_chunk.to_csv(cache_file, index=False)
        all_frames.append(df_chunk)
        print(f"  [{i+1:02d}/{len(chunks)}] {yyyymm}  ({len(df_chunk)} rows, "
              f"NaN={df_chunk['water_level'].isna().sum()})")
        _time.sleep(0.5)

    if not all_frames:
        raise RuntimeError("No data retrieved — check station ID and date range.")

    df = pd.concat(all_frames, ignore_index=True)
    df = df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)

    # Quality-control and gap-fill
    df = _fill_gaps(df)

    n_nan = df["water_level"].isna().sum()
    avail = 100.0 * (1.0 - n_nan / len(df))
    print(f"Done. Total rows: {len(df)}, NaN after fill: {n_nan} "
          f"({avail:.2f}% available)")
    return df


def harmonics_from_record(time: np.ndarray,
                          eta: np.ndarray,
                          lat: float = STATION_LAT) -> object:
    """
    Perform tidal harmonic analysis on a water-level record.

    Parameters
    ----------
    time : datetime64 array (or pandas DatetimeTZDtype)
    eta  : water-level array [m]
    lat  : station latitude in degrees N (needed for nodal corrections)

    Returns
    -------
    utide CoefStruct (pass directly to ``reconstruct_from_harmonics``)

    Notes
    -----
    Wraps ``utide.solve``.  Requires the ``utide`` package
    (``pip install utide``).
    """
    try:
        import utide
    except ImportError as exc:
        raise ImportError("utide is required: pip install utide") from exc

    t_pd = pd.to_datetime(time)
    t_hours = (t_pd.asi8 / 1e9 / 3600.0)   # ns → hours since Unix epoch

    # Remove NaN samples before analysis
    valid = ~np.isnan(eta)
    coef = utide.solve(t_hours[valid], eta[valid],
                       lat=lat, conf_int="linear", verbose=False)
    return coef


def reconstruct_from_harmonics(coef: object,
                                time: np.ndarray) -> np.ndarray:
    """
    Reconstruct a tidal water-level signal from harmonic constants.

    Parameters
    ----------
    coef : utide CoefStruct (output of ``harmonics_from_record``)
    time : datetime64 array for the reconstruction period

    Returns
    -------
    1-D float64 array of reconstructed water level [m]
    """
    try:
        import utide
    except ImportError as exc:
        raise ImportError("utide is required: pip install utide") from exc

    t_pd = pd.to_datetime(time)
    t_hours = (t_pd.asi8 / 1e9 / 3600.0)
    tide = utide.reconstruct(t_hours, coef, verbose=False)
    return tide.h


def resample_to_fvcom_time(df: pd.DataFrame,
                           time_mjd_fvcom: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate a NOAA water-level record onto the FVCOM time grid.

    Parameters
    ----------
    df              : DataFrame from ``fetch_noaa_waterlevel``
                      (must have columns 'time' (UTC datetime64) and
                      'water_level' (float, NaN-free or gap-filled)
    time_mjd_fvcom  : 1-D float64 array of FVCOM times in Modified Julian Days
                      (epoch 1858-11-17 00:00:00 UTC)

    Returns
    -------
    1-D float32 array, shape (ntime_fvcom,), water level [m] on FVCOM grid

    Raises
    ------
    ValueError
        If the FVCOM time window extends beyond the downloaded record
        (silent extrapolation is not allowed).
    """
    # Convert NOAA times to MJD for interpolation
    # MJD epoch: 1858-11-17 00:00:00 UTC
    mjd_epoch = pd.Timestamp("1858-11-17 00:00:00", tz="UTC")
    t_noaa_mjd = ((df["time"] - mjd_epoch).dt.total_seconds().values
                  / 86400.0)

    eta_noaa = df["water_level"].values.astype(np.float64)

    fvcom_start = time_mjd_fvcom[0]
    fvcom_end   = time_mjd_fvcom[-1]
    noaa_start  = np.nanmin(t_noaa_mjd)
    noaa_end    = np.nanmax(t_noaa_mjd)

    # Allow up to 1 minute of floating-point overshoot at either boundary
    # (NOAA timestamps truncated to minute → sub-minute rounding is expected).
    _MARGIN_MJD = 1.0 / 1440.0   # 1 minute
    if fvcom_start < noaa_start - _MARGIN_MJD or fvcom_end > noaa_end + _MARGIN_MJD:
        raise ValueError(
            f"FVCOM time window [{fvcom_start:.4f}, {fvcom_end:.4f}] MJD "
            f"extends beyond NOAA record [{noaa_start:.4f}, {noaa_end:.4f}] MJD. "
            "Extend the download range.")

    eta_fvcom = np.interp(time_mjd_fvcom, t_noaa_mjd, eta_noaa)
    return eta_fvcom.astype(np.float32)


# ---------------------------------------------------------------------------
# F05 -- Salinity and temperature from CO-OPS (station observations)
# ---------------------------------------------------------------------------

#: JSON response field name for each supported product
_TS_FIELD_MAP: dict[str, str] = {
    "salinity"          : "s",
    "water_temperature" : "v",
    "conductivity"      : "v",
}


def get_station_info(station_id: str) -> dict:
    """Return metadata for a CO-OPS station.

    Parameters
    ----------
    station_id : NOAA CO-OPS station ID string, e.g. ``'8573927'``

    Returns
    -------
    dict with keys: ``id``, ``name``, ``lat``, ``lon``, ``state``
    """
    url = (
        "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"
        f"/stations/{station_id}.json"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    meta = resp.json().get("stations", [{}])[0]
    return {
        "id"   : station_id,
        "name" : meta.get("name",  ""),
        "lat"  : meta.get("lat",   np.nan),
        "lon"  : meta.get("lng",   np.nan),
        "state": meta.get("state", ""),
    }


def fetch_noaa_ts(
        station_id: str = CHESAPEAKE_CITY_STATION,
        t_start: str    = "2018-01-01",
        t_end: str      = "2020-12-31",
        products: Sequence[str] = ("salinity", "water_temperature"),
        cache_dir: str | Path | None = None,
        interval: str  = "h",
        datum: str     = "MLLW",
        time_zone: str = "GMT") -> dict:
    """Fetch salinity and/or water temperature from the NOAA CO-OPS API.

    Designed for F05: canal T/S open boundary condition at Chesapeake City
    (station 8573927).  Uses the same CO-OPS API as
    :func:`fetch_noaa_waterlevel` but retrieves scalar oceanographic
    products (salinity, water_temperature) at hourly resolution.

    Parameters
    ----------
    station_id : CO-OPS station ID, default ``'8573927'`` (Chesapeake City)
    t_start, t_end : ISO date strings ``'YYYY-MM-DD'``
    products   : one or more of ``'salinity'``, ``'water_temperature'``,
                 ``'conductivity'``.  Each product issues one API call per
                 monthly chunk.
    cache_dir  : directory for per-month CSV cache files named
                 ``noaa_{station}_{product}_{YYYYMM}.csv``.
                 Pass ``None`` to disable caching.
    interval   : ``'h'`` (hourly, default) or ``'6'`` (6-minute)
    datum      : CO-OPS datum string (default ``'MLLW'``)
    time_zone  : always use ``'GMT'``

    Returns
    -------
    dict with keys:

    ``time_dt64``
        ndarray, shape ``(ntime,)``, dtype ``datetime64[s]``,
        regular hourly grid from ``t_start 00:00`` to ``t_end 23:00``.
    per-product arrays (same length as ``time_dt64``):
        ``salinity``          -- float64 [PSU],   NaN for unfilled gaps
        ``water_temperature`` -- float64 [deg C], NaN for unfilled gaps
        ``conductivity``      -- float64 [mS/cm], NaN for unfilled gaps

    Notes
    -----
    * Per-month CSVs are cached analogously to :func:`fetch_noaa_waterlevel`.
    * All NaN gaps are linearly interpolated (no 3-hour limit, because
      the oceanographic products have very few gaps in practice).
    * CO-OPS provides **surface bulk** observations only.  For the shallow
      (~5 m), well-mixed C&D Canal it is appropriate to broadcast surface
      T/S uniformly across all FVCOM sigma layers (see F05 notebook, Cell 9).
    """
    products = list(products)
    for p in products:
        if p not in _TS_FIELD_MAP:
            raise ValueError(
                f"Unknown product '{p}'.  Choose from: {list(_TS_FIELD_MAP)}"
            )

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    step_s = {"h": 3600, "6": 360}.get(interval, 3600)

    # Build regular time axis
    t_arr = np.arange(
        np.datetime64(t_start, "s"),
        np.datetime64(t_end,   "s") + np.timedelta64(step_s + 1, "s"),
        np.timedelta64(step_s, "s"),
        dtype="datetime64[s]",
    )
    t_end_dt = np.datetime64(t_end, "s") + np.timedelta64(86399, "s")
    t_arr = t_arr[t_arr <= t_end_dt]
    ntime = len(t_arr)

    out: dict = {"time_dt64": t_arr}
    for p in products:
        out[p] = np.full(ntime, np.nan)

    chunks = list(_month_chunks(t_start, t_end))

    for p in products:
        field = _TS_FIELD_MAP[p]
        print(f"Fetching {p} from station {station_id} "
              f"({t_start} -> {t_end}, {len(chunks)} monthly chunk(s))")

        for i, (begin, end) in enumerate(chunks):
            yyyymm = begin[:6]

            # --- CSV cache ---
            if cache_dir is not None:
                cache_file = cache_dir / f"noaa_{station_id}_{p}_{yyyymm}.csv"
                if cache_file.exists():
                    df_c = pd.read_csv(cache_file)
                    for _, row in df_c.iterrows():
                        try:
                            t_rec = np.datetime64(
                                str(row["time"]).replace(" ", "T"), "s")
                            idx = int(
                                (t_rec - t_arr[0]) / np.timedelta64(step_s, "s")
                            )
                            if 0 <= idx < ntime and pd.notna(row["value"]):
                                out[p][idx] = float(row["value"])
                        except (ValueError, OverflowError):
                            pass
                    print(f"  [{i+1:02d}/{len(chunks)}] {yyyymm} {p} (cached)")
                    continue

            # --- API request ---
            params = {
                "begin_date"  : begin,
                "end_date"    : end,
                "station"     : station_id,
                "product"     : p,
                "datum"       : datum,
                "units"       : "metric",
                "time_zone"   : time_zone.lower(),
                "interval"    : interval,
                "application" : "WaterPACT_DRE",
                "format"      : "json",
            }
            resp = requests.get(COOPS_API_BASE, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()

            if "error" in payload:
                msg = payload["error"].get("message", str(payload["error"]))
                print(f"  [{i+1:02d}/{len(chunks)}] {yyyymm} {p} WARNING: {msg}")
                _time.sleep(0.5)
                continue

            records    = payload.get("data", [])
            cache_rows: list[dict] = []
            n_ok       = 0

            for rec in records:
                try:
                    t_rec   = np.datetime64(rec["t"].replace(" ", "T"), "s")
                    val_str = rec.get(field, "").strip()
                    val     = (float(val_str)
                               if val_str and val_str != "-" else np.nan)
                    idx = int(
                        (t_rec - t_arr[0]) / np.timedelta64(step_s, "s")
                    )
                    if 0 <= idx < ntime:
                        out[p][idx] = val
                        if not np.isnan(val):
                            n_ok += 1
                    if cache_dir is not None:
                        cache_rows.append({"time": rec["t"], "value": val})
                except (ValueError, KeyError, OverflowError):
                    pass

            if cache_dir is not None and cache_rows:
                pd.DataFrame(cache_rows).to_csv(cache_file, index=False)

            print(f"  [{i+1:02d}/{len(chunks)}] {yyyymm} {p}  "
                  f"{n_ok}/{len(records)} valid")
            _time.sleep(0.5)

        # Linearly infill all NaN gaps
        col   = out[p]
        valid = np.isfinite(col)
        if valid.sum() >= 2 and (~valid).any():
            t_idx = np.arange(ntime, dtype=float)
            col[~valid] = np.interp(t_idx[~valid], t_idx[valid], col[valid])

        n_nan = int(np.sum(np.isnan(out[p])))
        pct   = 100.0 * n_nan / ntime
        lo    = np.nanmin(out[p])
        hi    = np.nanmax(out[p])
        print(f"  {p}: {ntime - n_nan}/{ntime} valid  "
              f"({pct:.1f}% missing)  range=[{lo:.3f}, {hi:.3f}]")

    return out
