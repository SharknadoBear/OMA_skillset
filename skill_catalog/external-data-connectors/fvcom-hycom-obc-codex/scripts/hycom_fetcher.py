"""
hycom_fetcher.py
================
HYCOM OPeNDAP data download for SSH (F02) and T/S (F03) offshore OBC.

Strategy
--------
A two-step OPeNDAP approach is used for robustness on the HYCOM THREDDS server:

  Step 1 – Lightweight coordinate fetch:
      ``xr.open_dataset(base_url)`` — opens lazily, fetches only lon/lat/time.
      Downloads only the axis arrays to determine integer index ranges.

  Step 2 – Targeted isel-based fetch:
      ``xr.open_dataset(base_url).isel(time=…, lat=…, lon=…).load()``
      Downloads only the data block needed, not the global dataset.
      (The netcdf4 OPeNDAP client does not support DAP2 constraint expressions
      appended as URL query strings.)

Each step is wrapped in an exponential-backoff retry loop to handle HYCOM
server instability.

After downloading one calendar month, the caller should immediately
interpolate to OBC nodes (``interp_ssh_to_obc``) and save to a compressed
NumPy checkpoint (``save_ssh_checkpoint``).  This avoids storing large
spatial grids in memory and allows the download to be resumed after a failure.

HYCOM Experiment Registry
-------------------------
All 9 known GOFS 3.1 experiments are encoded in ``HYCOM_EXPERIMENTS``.
``get_experiment_for_date(d)`` routes any date from 1994-01-01 to 2024-09-04
to the correct product/experiment, with overlap resolved by preferring the
experiment with the latest start date (newest product wins).

Reference workflow (old MATLAB)
--------------------------------
  data_source/d_HYCOM_boundary_ssh_data.m      -- monthly SSH download loop
  fvcom_prepro/get_HYCOM_forcing.m             -- OPeNDAP pull via ncread
  fvcom_prepro/interp_HYCOM2FVCOM.m            -- horizontal interpolation
  fvcom_prepro/write_FVCOM_elevtide.m          -- NetCDF output

Python dependencies
-------------------
  xarray, requests, numpy, scipy
"""

from __future__ import annotations

import time
import warnings
import calendar
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Optional, Sequence

# Suppress pydap's DAP-protocol auto-detection warning (HYCOM uses https://
# which triggers the warning; DAP2 is chosen automatically and works fine).
warnings.filterwarnings(
    "ignore",
    message="PyDAP was unable to determine the DAP protocol",
    category=UserWarning,
    module="pydap",
)


# ── ASCII progress log ────────────────────────────────────────────────────────
# All significant events (start/done/retry/fail) are appended here so the
# user can monitor progress in a plain text viewer (e.g. `Get-Content -Wait
# hycom_fetch.log` in PowerShell) independent of the notebook kernel.

_LOG_PATH: Optional[Path] = None   # set by fetch_hycom_ssh_month
_QUIET_STDOUT: bool = False        # set True by notebook loops to suppress stdout


def _log(msg: str, log_path: Optional[Path] = None) -> None:
    """Append a timestamped line to the ASCII log and optionally print.

    When ``_QUIET_STDOUT`` is True (e.g. notebook loop using clear_output),
    messages are written to the log file only — no stdout output.
    """
    path = log_path or _LOG_PATH
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    if not _QUIET_STDOUT:
        print(line)
    if path is not None:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

import numpy as np
import xarray as xr
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    PchipInterpolator,
    RegularGridInterpolator,
    interp1d,
)


# ── Notebook progress display ─────────────────────────────────────────────────

class NotebookProgressDisplay:
    """Refreshing single-block progress display for Jupyter notebooks.

    Replaces the cell output on every update so nothing stacks.  Falls back
    gracefully to plain ``print`` when IPython is not available (e.g. when
    called from a script or terminal).

    Parameters
    ----------
    total : int
        Total number of items to process.
    label : str
        Short title shown at the top of the block (default ``"Progress"``).
    bar_len : int
        Width of the ASCII progress bar in characters (default ``36``).
    n_recent : int
        Number of recent status messages to show (default ``8``).
    quiet_fetcher : bool
        If ``True``, sets ``_QUIET_STDOUT = True`` on this module so that
        per-chunk ``_log`` messages go to the log file only, not the display.
        Restored to ``False`` when ``close()`` is called (default ``True``).

    Example
    -------
    ::

        prog = NotebookProgressDisplay(total=36, label="HYCOM SSH download",
                                       quiet_fetcher=True)
        for i, (year, month) in enumerate(year_month_pairs):
            prog.update(i, current=f"{year}-{month:02d}", status="downloading …")
            # … do work …
            prog.log(f"{year}-{month:02d}: OK")
        prog.close(summary="All months done ✓")
    """

    def __init__(
        self,
        total: int,
        label: str = "Progress",
        bar_len: int = 36,
        n_recent: int = 8,
        quiet_fetcher: bool = True,
    ) -> None:
        self.total        = total
        self.label        = label
        self.bar_len      = bar_len
        self.n_recent     = n_recent
        self._recent: list[str] = []
        self._closed      = False

        # Suppress fetcher stdout so messages go to log file only
        global _QUIET_STDOUT
        self._prev_quiet  = _QUIET_STDOUT
        if quiet_fetcher:
            _QUIET_STDOUT = True

        # Detect IPython / Jupyter
        try:
            from IPython.display import clear_output as _co
            self._clear_output: Optional[object] = _co
        except ImportError:
            self._clear_output = None

    # ------------------------------------------------------------------
    def _bar(self, done: int) -> str:
        filled = int(self.bar_len * done / max(self.total, 1))
        return "█" * filled + "░" * (self.bar_len - filled)

    def _render(
        self,
        done: int,
        current: str = "",
        status: str = "",
        extra_lines: Optional[list] = None,
    ) -> None:
        pct = 100.0 * done / max(self.total, 1)
        lines = [
            f"{self.label}  [{self._bar(done)}]  {done}/{self.total}  ({pct:.0f}%)",
        ]
        if current:
            lines.append(f"  Current : {current}  {status}")
        if self._recent:
            lines.append("")
            lines.append("  Recent messages:")
            for m in self._recent[-self.n_recent:]:
                lines.append(f"    {m}")
        if extra_lines:
            lines.extend(extra_lines)

        if self._clear_output is not None:
            self._clear_output(wait=True)   # type: ignore[call-arg]
        print("\n".join(lines))

    # ------------------------------------------------------------------
    def update(self, done: int, current: str = "", status: str = "") -> None:
        """Refresh the display.  Call once per iteration *before* doing work."""
        if not self._closed:
            self._render(done, current=current, status=status)

    def log(self, msg: str) -> None:
        """Append *msg* to the rolling recent-messages list (no redraw)."""
        self._recent.append(msg)

    def close(self, summary: str = "") -> None:
        """Draw the final 100 % state and restore ``_QUIET_STDOUT``."""
        global _QUIET_STDOUT
        _QUIET_STDOUT = self._prev_quiet
        self._closed  = True
        extra = []
        if summary:
            extra = ["", f"  {summary}"]
        self._render(self.total, extra_lines=extra)


# ── HYCOM Experiment Registry ─────────────────────────────────────────────────
# Ordered oldest → newest. Overlap resolution: entry with the latest ``start``
# date takes priority when multiple experiments cover the same day.
#
# Source: https://tds.hycom.org/thredds/catalog.html  (verified 2026-05)
HYCOM_EXPERIMENTS: list[dict] = [
    {"product": "GLBv0.08", "expt": "expt_53.X",
     "start": date(1994,  1,  1), "end": date(2015, 12, 30)},
    {"product": "GLBv0.08", "expt": "expt_56.3",
     "start": date(2014,  7,  1), "end": date(2016,  4, 30)},
    {"product": "GLBv0.08", "expt": "expt_57.2",
     "start": date(2016,  5,  1), "end": date(2017,  1, 31)},
    {"product": "GLBv0.08", "expt": "expt_92.8",
     "start": date(2017,  2,  1), "end": date(2017,  5, 31)},
    {"product": "GLBv0.08", "expt": "expt_57.7",
     "start": date(2017,  6,  1), "end": date(2017,  9, 30)},
    {"product": "GLBv0.08", "expt": "expt_92.9",
     "start": date(2017, 10,  1), "end": date(2017, 12, 31)},
    {"product": "GLBv0.08", "expt": "expt_93.0",
     "start": date(2018,  1,  1), "end": date(2020,  2, 18)},
    {"product": "GLBu0.08", "expt": "expt_93.0",
     "start": date(2018,  9, 19), "end": date(2018, 12,  8)},
    {"product": "GLBy0.08", "expt": "expt_93.0",
     "start": date(2018, 12,  4), "end": date(2024,  9,  4)},
]

HYCOM_THREDDS_BASE = "https://tds.hycom.org/thredds/dodsC"

# Default Delaware shelf domain — [0, 360] longitude convention
DEFAULT_LON_RANGE = (283.0, 288.0)
DEFAULT_LAT_RANGE = (36.0,  41.0)

# HYCOM OPeNDAP variable metadata. The u/v aliases mirror the previous OMI
# knowledge-base reference, where product variables are water_u/water_v but
# short u/v names also appear in the dimension metadata.
HYCOM_2D_VARIABLES = {"surf_el"}
HYCOM_3D_VARIABLES = {"water_temp", "salinity", "water_u", "water_v"}
HYCOM_VARIABLE_ALIASES = {
    "ssh": "surf_el",
    "elev": "surf_el",
    "elevation": "surf_el",
    "temp": "water_temp",
    "temperature": "water_temp",
    "salt": "salinity",
    "sal": "salinity",
    "u": "water_u",
    "v": "water_v",
}
HYCOM_PRODUCT_VARIABLES = HYCOM_2D_VARIABLES | HYCOM_3D_VARIABLES


# ── Custom exception ──────────────────────────────────────────────────────────

class HycomDownloadError(RuntimeError):
    """Raised when all OPeNDAP retry attempts for a HYCOM URL are exhausted."""

    def __init__(self, url: str, attempts: int):
        super().__init__(
            f"HYCOM OPeNDAP download failed after {attempts} attempt(s).\n"
            f"URL: {url}"
        )
        self.url = url
        self.attempts = attempts


# ── Public routing helper ──────────────────────────────────────────────────────

def _get_experiment_entry_for_date(target_date: date) -> dict:
    """Return the selected HYCOM experiment registry entry for a date."""
    candidates = [
        e for e in HYCOM_EXPERIMENTS
        if e["start"] <= target_date <= e["end"]
    ]
    if not candidates:
        valid = (
            f"{HYCOM_EXPERIMENTS[0]['start']} to {HYCOM_EXPERIMENTS[-1]['end']}"
        )
        raise ValueError(
            f"No HYCOM experiment covers {target_date}. "
            f"Supported range: {valid}."
        )
    return max(candidates, key=lambda e: e["start"])


def get_experiment_for_date(target_date: date) -> tuple[str, str]:
    """
    Return the HYCOM (product, expt) pair that covers a given calendar date.

    When multiple experiments overlap the same date the one with the latest
    ``start`` date is returned (newest product takes priority over older ones).

    Parameters
    ----------
    target_date : datetime.date
        The calendar date to look up.

    Returns
    -------
    product : str
        HYCOM grid product identifier, e.g. ``'GLBy0.08'``.
    expt : str
        Experiment string, e.g. ``'expt_93.0'``.

    Raises
    ------
    ValueError
        If no HYCOM experiment in :data:`HYCOM_EXPERIMENTS` covers the date.

    Examples
    --------
    >>> from datetime import date
    >>> get_experiment_for_date(date(2019, 6, 1))
    ('GLBy0.08', 'expt_93.0')
    >>> get_experiment_for_date(date(1999, 1, 1))
    ('GLBv0.08', 'expt_53.X')
    >>> get_experiment_for_date(date(2018, 10, 15))
    ('GLBu0.08', 'expt_93.0')
    """
    best = _get_experiment_entry_for_date(target_date)
    return best["product"], best["expt"]


# ── Private helpers ────────────────────────────────────────────────────────────

def _build_base_url(target_date: date) -> str:
    """Return the THREDDS OPeNDAP base URL for the experiment covering *target_date*."""
    product, expt = get_experiment_for_date(target_date)
    return f"{HYCOM_THREDDS_BASE}/{product}/{expt}"


@dataclass(frozen=True)
class HycomDownloadRequest:
    """Reusable HYCOM request for driver-code composition.

    The request is intentionally small and serializable by ordinary Python
    callers. Variables may use canonical HYCOM names or the short aliases
    listed in HYCOM_VARIABLE_ALIASES.
    """

    start: str | date | datetime
    end: str | date | datetime
    variables: Sequence[str] = ("surf_el",)
    lon_range: tuple[float, float] = DEFAULT_LON_RANGE
    lat_range: tuple[float, float] = DEFAULT_LAT_RANGE
    depth_range: Optional[tuple[float, float]] = None
    max_depth: Optional[float] = 200.0
    points: Optional[Any] = None
    cache_dir: Optional[str | Path] = None
    max_retries: int = 5
    retry_delay: float = 10.0
    backoff: float = 2.0
    chunk_t: int = 20
    ssh_chunk_t: int = 50
    label: str = "hycom"


def _normalize_hycom_variables(variables: Sequence[str] | str | None) -> list[str]:
    """Normalize HYCOM variable aliases and validate supported product names."""
    if variables is None:
        raise ValueError("At least one HYCOM variable must be supplied.")
    raw_variables = [variables] if isinstance(variables, str) else list(variables)
    normalized: list[str] = []
    for raw in raw_variables:
        key = str(raw).strip()
        if not key:
            continue
        canonical = HYCOM_VARIABLE_ALIASES.get(key.lower(), key)
        if canonical not in HYCOM_PRODUCT_VARIABLES:
            supported = ", ".join(sorted(HYCOM_PRODUCT_VARIABLES | set(HYCOM_VARIABLE_ALIASES)))
            raise ValueError(f"Unsupported HYCOM variable '{raw}'. Supported: {supported}.")
        if canonical not in normalized:
            normalized.append(canonical)
    if not normalized:
        raise ValueError("At least one HYCOM variable must be supplied.")
    return normalized


def _parse_request_datetime(value: str | date | datetime, end_of_day: bool = False) -> datetime:
    """Parse request datetimes; date-only end values include the whole day."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
        if end_of_day:
            dt = dt + timedelta(days=1) - timedelta(seconds=1)
    else:
        text = str(value).strip()
        date_only = len(text) == 10 and text[4] == "-" and text[7] == "-"
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if date_only and end_of_day:
            dt = dt + timedelta(days=1) - timedelta(seconds=1)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt.replace(microsecond=0)


def _request_datetime_bounds(request: HycomDownloadRequest) -> tuple[datetime, datetime]:
    """Return validated start/end datetimes for a request."""
    start_dt = _parse_request_datetime(request.start, end_of_day=False)
    end_dt = _parse_request_datetime(request.end, end_of_day=True)
    if end_dt < start_dt:
        raise ValueError(f"HYCOM request end {end_dt} precedes start {start_dt}.")
    return start_dt, end_dt


def _end_of_day(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 23, 59, 59)


def _month_end(dt: datetime) -> datetime:
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    return datetime(dt.year, dt.month, last_day, 23, 59, 59)


def _selected_experiment_window_end(
    cursor: datetime,
    limit: datetime,
    entry: dict,
) -> datetime:
    """Shorten a window if HYCOM overlap priority changes on a later day."""
    selected = (entry["product"], entry["expt"])
    day = cursor.date() + timedelta(days=1)
    while datetime(day.year, day.month, day.day) <= limit:
        next_entry = _get_experiment_entry_for_date(day)
        next_selected = (next_entry["product"], next_entry["expt"])
        if next_selected != selected:
            return datetime(day.year, day.month, day.day) - timedelta(seconds=1)
        day += timedelta(days=1)
    return limit


def plan_hycom_chunks(request: HycomDownloadRequest) -> list[dict[str, Any]]:
    """Dry-run chunk plan for a HYCOM request without touching the network.

    The plan splits by month and HYCOM experiment boundary. Download chunking
    inside each planned window is still controlled by request.chunk_t and
    request.ssh_chunk_t.
    """
    variables = _normalize_hycom_variables(request.variables)
    start_dt, end_dt = _request_datetime_bounds(request)
    chunks: list[dict[str, Any]] = []

    cursor = start_dt
    while cursor <= end_dt:
        entry = _get_experiment_entry_for_date(cursor.date())
        candidate_end = min(end_dt, _end_of_day(entry["end"]), _month_end(cursor))
        window_end = _selected_experiment_window_end(cursor, candidate_end, entry)
        base_url = f"{HYCOM_THREDDS_BASE}/{entry['product']}/{entry['expt']}"
        chunks.append({
            "chunk_index": len(chunks) + 1,
            "start": cursor.isoformat(sep=" ", timespec="seconds"),
            "end": window_end.isoformat(sep=" ", timespec="seconds"),
            "product": entry["product"],
            "expt": entry["expt"],
            "base_url": base_url,
            "variables": tuple(variables),
            "lon_range": tuple(request.lon_range),
            "lat_range": tuple(request.lat_range),
            "depth_range": request.depth_range,
            "max_depth": request.max_depth,
            "chunk_t": request.chunk_t,
            "ssh_chunk_t": request.ssh_chunk_t,
        })
        cursor = window_end + timedelta(seconds=1)

    return chunks


def _retry(func, url: str, max_retries: int, retry_delay: float, backoff: float,
           log_path: Optional[Path] = None):
    """
    Call ``func(url)`` with exponential-backoff retry.

    Raises :class:`HycomDownloadError` after *max_retries* exhausted.
    """
    delay = retry_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func(url)
        except HycomDownloadError:
            raise  # don't swallow errors we already raised
        except Exception as exc:
            if attempt == max_retries:
                _log(f"  FAILED after {max_retries} attempt(s): {exc}", log_path)
                raise HycomDownloadError(url, max_retries) from exc
            _log(
                f"  attempt {attempt}/{max_retries} failed "
                f"({type(exc).__name__}: {exc}). Retrying in {delay:.0f} s …",
                log_path,
            )
            time.sleep(delay)
            delay *= backoff


def _fetch_coords(
    base_url: str,
    max_retries: int = 5,
    retry_delay: float = 10.0,
    backoff: float = 2.0,
) -> dict:
    """
    Fetch only coordinate arrays (lon, lat, time) from a HYCOM OPeNDAP endpoint.

    This lightweight first step avoids transferring variable data and gives
    the integer index ranges needed to build a targeted constraint URL.

    Parameters
    ----------
    base_url    : str
        HYCOM THREDDS base URL,
        e.g. ``'https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0'``.
    max_retries : int   Maximum retry attempts (default 5).
    retry_delay : float Initial wait in seconds between retries (default 10).
    backoff     : float Exponential multiplier applied after each failure (default 2).

    Returns
    -------
    dict with keys:
        ``'lon'``  – np.ndarray (nlon,) degrees East.
        ``'lat'``  – np.ndarray (nlat,) degrees North.
        ``'time'`` – np.ndarray (ntime,) datetime64[s].
    """
    # Use the DAP2 projected-variable URL to fetch only coord arrays.
    # engine='pydap' is required — the netcdf4 OPeNDAP client fails with
    # OSError -68 on this platform for all HYCOM THREDDS URLs.
    # The coord URL fetches only lon/lat/time metadata (no variable data).
    _log(f"  step 1 — fetching coordinates from {base_url.split('dodsC/')[-1]}")
    t0_wall = time.time()

    def _open(url):
        ds = xr.open_dataset(
            url + "?lon,lat,time",
            engine="pydap",
            decode_times=True,
        )
        result = {
            "lon":  ds["lon"].values.copy(),
            "lat":  ds["lat"].values.copy(),
            "time": ds["time"].values.astype("datetime64[s]").copy(),
        }
        ds.close()
        return result

    result = _retry(_open, base_url, max_retries, retry_delay, backoff)
    _log(f"  step 1 done ({time.time() - t0_wall:.1f} s)  "
         f"grid: {result['lon'].size} lon \u00d7 {result['lat'].size} lat "
         f"\u00d7 {result['time'].size} time")
    return result


def _fetch_depth_values(
    base_url: str,
    max_retries: int = 5,
    retry_delay: float = 10.0,
    backoff: float = 2.0,
) -> np.ndarray:
    """Fetch the HYCOM depth coordinate only."""
    _log("  fetching depth coordinate")

    def _open(url: str) -> np.ndarray:
        ds = xr.open_dataset(url + "?depth", engine="pydap", decode_times=False)
        depth_values = ds["depth"].values.astype(np.float64).copy()
        ds.close()
        return depth_values

    return _retry(_open, base_url, max_retries, retry_delay, backoff)


def _depth_index_range(
    hycom_depth: np.ndarray,
    depth_range: Optional[tuple[float, float]] = None,
    max_depth: Optional[float] = 200.0,
) -> tuple[int, int]:
    """Return inclusive HYCOM depth indices for a requested depth constraint."""
    if hycom_depth.size == 0:
        raise ValueError("HYCOM depth coordinate is empty.")

    if depth_range is not None:
        z0, z1 = sorted((float(depth_range[0]), float(depth_range[1])))
        mask = (hycom_depth >= z0) & (hycom_depth <= z1)
        if not mask.any():
            raise ValueError(f"No HYCOM depth levels found in [{z0}, {z1}] m.")
        idx = np.where(mask)[0]
        return int(idx[0]), int(idx[-1])

    if max_depth is None:
        return 0, int(len(hycom_depth) - 1)

    depth_mask = hycom_depth <= float(max_depth)
    if not depth_mask.any():
        return 0, 0
    depth1 = int(np.where(depth_mask)[0][-1])
    if depth1 + 1 < len(hycom_depth):
        depth1 += 1
    return 0, depth1


def _find_indices(
    coords: dict,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    t_start: np.datetime64,
    t_end: np.datetime64,
) -> tuple[int, int, int, int, int, int]:
    """
    Compute OPeNDAP integer index ranges for a spatial/temporal subsetting request.

    Handles longitude convention normalisation so that user input in either
    [0, 360] or [-180, 180] is correctly matched to the remote grid.

    Parameters
    ----------
    coords    : dict returned by :func:`_fetch_coords`.
    lon_range : (lon_min, lon_max) in any convention.
    lat_range : (lat_min, lat_max).
    t_start   : np.datetime64 – start of desired window (inclusive).
    t_end     : np.datetime64 – end of desired window (inclusive).

    Returns
    -------
    (t0, t1, lat0, lat1, lon0, lon1) – integer indices for the OPeNDAP
    constraint expression ``[start:1:end]``.
    """
    remote_lon  = coords["lon"]
    remote_lat  = coords["lat"]
    remote_time = coords["time"]

    lon_min_u, lon_max_u = float(lon_range[0]), float(lon_range[1])
    lat_min,   lat_max   = float(lat_range[0]), float(lat_range[1])

    # Normalise user longitude to match remote grid convention
    remote_uses_0_360 = np.all(remote_lon >= 0)
    if remote_uses_0_360:
        if lon_min_u < 0:
            lon_min_u += 360.0
        if lon_max_u < 0:
            lon_max_u += 360.0
    else:
        if lon_min_u > 180:
            lon_min_u -= 360.0
        if lon_max_u > 180:
            lon_max_u -= 360.0

    lon_idx  = np.where((remote_lon  >= lon_min_u) & (remote_lon  <= lon_max_u))[0]
    lat_idx  = np.where((remote_lat  >= lat_min)   & (remote_lat  <= lat_max))[0]
    time_idx = np.where(
        (remote_time >= t_start.astype("datetime64[s]")) &
        (remote_time <= t_end.astype("datetime64[s]"))
    )[0]

    if lon_idx.size == 0:
        raise ValueError(
            f"No HYCOM lon grid points found in [{lon_min_u}, {lon_max_u}]°E."
        )
    if lat_idx.size == 0:
        raise ValueError(
            f"No HYCOM lat grid points found in [{lat_min}, {lat_max}]°N."
        )
    if time_idx.size == 0:
        raise ValueError(
            f"No HYCOM time steps found in [{t_start}, {t_end}]."
        )

    return (
        int(time_idx[0]), int(time_idx[-1]),
        int(lat_idx[0]),  int(lat_idx[-1]),
        int(lon_idx[0]),  int(lon_idx[-1]),
    )


def _fetch_ssh_block(
    base_url: str,
    t0: int, t1: int,
    lat0: int, lat1: int,
    lon0: int, lon1: int,
    max_retries: int = 5,
    retry_delay: float = 10.0,
    backoff: float = 2.0,
    chunk_t: int = 50,
) -> xr.Dataset:
    """
    Download a targeted block of HYCOM surf_el in small time chunks.

    For each chunk the function first attempts a DAP2 constraint URL
    (``?surf_el[tc0:1:tc1][lat0:1:lat1][lon0:1:lon1],…``) so the HYCOM
    server pre-trims the data before sending.  If the server returns an
    I/O error for the constraint expression, the function falls back to
    the isel-based approach for the remainder of the download.

    Parameters
    ----------
    base_url        : HYCOM THREDDS base URL.
    t0, t1          : global time index range (inclusive).
    lat0, lat1      : latitude index range (inclusive).
    lon0, lon1      : longitude index range (inclusive).
    max_retries     : int
    retry_delay     : float  initial retry delay in seconds.
    backoff         : float  exponential backoff multiplier.
    chunk_t         : int    time steps per chunk (default 50 ≈ 6 days at 3-hourly).

    Returns
    -------
    xr.Dataset
        ``'surf_el'`` (time, lat, lon), decoded to float32 metres.
    """
    n_t   = t1 - t0 + 1
    n_lat = lat1 - lat0 + 1
    n_lon = lon1 - lon0 + 1

    chunk_starts = list(range(t0, t1 + 1, chunk_t))
    n_chunks     = len(chunk_starts)

    _log(
        f"  step 2 \u2014 surf_el [{n_t} \u00d7 {n_lat} \u00d7 {n_lon}]  "
        f"split into {n_chunks} chunk(s) of \u2264{chunk_t} steps"
    )
    t0_wall = time.time()

    def _fetch_one_chunk(url: str, tc0: int, tc1: int) -> xr.Dataset:
        """Fetch one time-chunk via DAP2 constraint URL with pydap engine."""
        constraint = (
            f"?surf_el[{tc0}:1:{tc1}][{lat0}:1:{lat1}][{lon0}:1:{lon1}]"
            f",time[{tc0}:1:{tc1}]"
            f",lat[{lat0}:1:{lat1}]"
            f",lon[{lon0}:1:{lon1}]"
        )
        ds = xr.open_dataset(
            url + constraint,
            engine="pydap",
            decode_times=True,
            mask_and_scale=True,
        )
        ds.load()
        return ds

    chunks: list[xr.Dataset] = []
    for i, tc0 in enumerate(chunk_starts, 1):
        tc1   = min(tc0 + chunk_t - 1, t1)
        t_wall = time.time()

        def _do_chunk(url: str, _tc0: int = tc0, _tc1: int = tc1) -> xr.Dataset:
            return _fetch_one_chunk(url, _tc0, _tc1)

        chunk_ds = _retry(_do_chunk, base_url, max_retries, retry_delay, backoff)
        chunks.append(chunk_ds)
        _log(
            f"  chunk {i}/{n_chunks}  t=[{tc0}:{tc1}]  "
            f"({tc1 - tc0 + 1} steps)  {time.time() - t_wall:.1f} s"
        )

    result = xr.concat(chunks, dim="time")
    _log(f"  step 2 done  total {time.time() - t0_wall:.1f} s")
    return result


# ── Public download function ───────────────────────────────────────────────────

def fetch_hycom_ssh_month(
    year: int,
    month: int,
    lon_range: tuple[float, float] = DEFAULT_LON_RANGE,
    lat_range: tuple[float, float] = DEFAULT_LAT_RANGE,
    cache_dir: Optional[str | Path] = None,
    max_retries: int = 5,
    retry_delay: float = 10.0,
    backoff: float = 2.0,
    chunk_t: int = 50,
) -> xr.Dataset:
    """
    Download one calendar month of HYCOM sea surface height (``surf_el``).

    The correct HYCOM experiment is selected automatically via
    :data:`HYCOM_EXPERIMENTS`. A two-step OPeNDAP strategy is used: first
    fetch coordinate arrays only (lightweight), then issue a targeted
    constraint-expression URL for the requested spatial/temporal block.
    Both steps are retried with exponential backoff.

    Parameters
    ----------
    year, month : int
        Calendar year and month (1–12).
    lon_range : (lon_min, lon_max)
        Longitude bounding box, degrees East [0, 360].
        Default: ``(283.0, 288.0)`` — Delaware shelf domain.
    lat_range : (lat_min, lat_max)
        Latitude bounding box, degrees North.
        Default: ``(36.0, 41.0)`` — Delaware shelf domain.
    cache_dir : path or None
        If provided, the ASCII progress log is written here as
        ``hycom_fetch.log``.  No raw NetCDF is saved — use
        :func:`save_ssh_checkpoint` for the interpolated-to-OBC checkpoint.
    max_retries : int   Maximum retry attempts (default 5).
    retry_delay : float Initial retry wait in seconds (default 10).
    backoff     : float Exponential backoff multiplier (default 2).
    chunk_t     : int   Time steps per download chunk (default 50 ≈ 6 days
                        at 3-hourly resolution).  Smaller values give more
                        frequent log updates and shorter per-request payloads.

    Returns
    -------
    xr.Dataset
        ``'surf_el'`` (time, lat, lon), float32, metres.

    Examples
    --------
    >>> ds = fetch_hycom_ssh_month(2019, 6)
    >>> ds['surf_el'].shape   # (ntime_june, nlat_subset, nlon_subset)
    """
    global _LOG_PATH

    first_day = date(year, month, 1)
    last_day  = date(year, month, calendar.monthrange(year, month)[1])

    base_url = _build_base_url(first_day)

    # Derive log path from cache_dir (or a sibling of it) if not already set
    if _LOG_PATH is None and cache_dir is not None:
        _LOG_PATH = Path(cache_dir) / "hycom_fetch.log"

    _log(f"── {year}-{month:02d} → {base_url.split('dodsC/')[-1]}")

    # Step 1 — lightweight coordinate fetch
    coords = _fetch_coords(base_url, max_retries, retry_delay, backoff)

    t_start = np.datetime64(first_day.isoformat(), "s")
    t_end   = np.datetime64(last_day.isoformat(),  "s") + np.timedelta64(23, "h")

    t0, t1, lat0, lat1, lon0, lon1 = _find_indices(
        coords, lon_range, lat_range, t_start, t_end
    )
    _log(
        f"  indices: t=[{t0}:{t1}] lat=[{lat0}:{lat1}] lon=[{lon0}:{lon1}]  "
        f"→ {t1 - t0 + 1} time steps"
    )

    # Step 2 — targeted data download in small chunks
    ds = _fetch_ssh_block(
        base_url, t0, t1, lat0, lat1, lon0, lon1,
        max_retries, retry_delay, backoff, chunk_t=chunk_t
    )

    return ds


# ── OBC interpolation ─────────────────────────────────────────────────────────

def interp_ssh_to_obc(
    ds_ssh: xr.Dataset,
    obc_lon: np.ndarray,
    obc_lat: np.ndarray,
    offset: float = 0.7671,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate HYCOM ``surf_el`` to FVCOM OBC node locations.

    A triangulation-based linear scheme
    (``scipy.interpolate.LinearNDInterpolator``) is used for interior points,
    with nearest-neighbour fallback for OBC nodes that lie outside the HYCOM
    convex hull (typically the most coastal nodes near the shoreline).

    An additive offset is applied after interpolation to correct for the
    mean sea-level bias between the HYCOM reference geoid and NAVD 88/MSL at
    the FVCOM boundary.

    Parameters
    ----------
    ds_ssh  : xr.Dataset
        Output of :func:`fetch_hycom_ssh_month`.  Must contain
        ``'surf_el'`` (time, lat, lon).
    obc_lon : np.ndarray  (nobc,)
        OBC node longitudes, degrees East [0, 360].
    obc_lat : np.ndarray  (nobc,)
        OBC node latitudes, degrees North.
    offset  : float
        Mean sea-level offset in metres to add to every interpolated value.
        Default ``0.7671 m`` — derived from a Cape May water-level
        calibration in the original MATLAB preprocessing (``dz0 = 0.7671``).

    Returns
    -------
    obc_ssh  : np.ndarray  (ntime, nobc) float32 — OBC SSH in metres, offset applied.
    time_mjd : np.ndarray  (ntime,) float64 — Modified Julian Day
                           (days since 1858-11-17 00:00:00 UTC).

    Examples
    --------
    >>> ds = fetch_hycom_ssh_month(2019, 6)
    >>> obc_ssh, time_mjd = interp_ssh_to_obc(ds, obc_lon, obc_lat)
    >>> obc_ssh.shape
    (248, 95)
    """
    try:
        from .grid_utils import datetime64_to_mjd
    except ImportError:
        from grid_utils import datetime64_to_mjd

    surf_el = ds_ssh["surf_el"].values        # (ntime, nlat, nlon)
    lons_2d, lats_2d = np.meshgrid(
        ds_ssh["lon"].values, ds_ssh["lat"].values
    )
    pts_flat   = np.column_stack([lons_2d.ravel(), lats_2d.ravel()])
    target_pts = np.column_stack([obc_lon, obc_lat])

    ntime = surf_el.shape[0]
    nobc  = len(obc_lon)
    obc_ssh = np.empty((ntime, nobc), dtype=np.float32)

    for i in range(ntime):
        vals = surf_el[i].ravel()
        valid = np.isfinite(vals)
        if valid.sum() < 4:
            obc_ssh[i] = np.nan
            continue

        lin_interp = LinearNDInterpolator(pts_flat[valid], vals[valid])
        result = lin_interp(target_pts)

        # Nearest-neighbour fallback for any extrapolated (NaN) nodes
        nan_mask = ~np.isfinite(result)
        if nan_mask.any():
            nn_interp = NearestNDInterpolator(pts_flat[valid], vals[valid])
            result[nan_mask] = nn_interp(target_pts[nan_mask])

        obc_ssh[i] = result.astype(np.float32)

    obc_ssh += np.float32(offset)

    time_np  = ds_ssh["time"].values.astype("datetime64[s]")
    time_mjd = datetime64_to_mjd(time_np)

    return obc_ssh, time_mjd


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _checkpoint_path(cache_dir: Path, year: int, month: int) -> Path:
    """Return the standard checkpoint filename for a given year/month."""
    return Path(cache_dir) / f"hycom_ssh_obc_{year}_{month:02d}.npz"


def save_ssh_checkpoint(
    obc_ssh: np.ndarray,
    time_mjd: np.ndarray,
    filepath: str | Path,
) -> Path:
    """
    Save an interpolated OBC SSH block to a compressed NumPy archive.

    Parameters
    ----------
    obc_ssh  : np.ndarray  (ntime, nobc) float32 — OBC SSH in metres.
    time_mjd : np.ndarray  (ntime,) float64 — Modified Julian Day.
    filepath : path
        Destination file, e.g.
        ``cache_dir / 'hycom_ssh_obc_2019_06.npz'``.
        Parent directory is created if it does not exist.

    Returns
    -------
    Path  Absolute path to the saved file.

    Examples
    --------
    >>> p = save_ssh_checkpoint(obc_ssh, time_mjd,
    ...                         'data_raw/hycom_ssh/hycom_ssh_obc_2019_06.npz')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(filepath), obc_ssh=obc_ssh, time_mjd=time_mjd)
    return filepath.resolve()


def load_ssh_checkpoints(
    cache_dir: str | Path,
    year_range: Optional[tuple[int, int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and chronologically concatenate all monthly OBC SSH checkpoint files.

    Parameters
    ----------
    cache_dir  : directory containing ``hycom_ssh_obc_YYYY_MM.npz`` files.
    year_range : (year_start, year_end) inclusive, or ``None`` to load all.

    Returns
    -------
    time_mjd : np.ndarray  (ntime_total,) float64 — sorted MJD time axis.
    obc_ssh  : np.ndarray  (ntime_total, nobc) float32.

    Raises
    ------
    FileNotFoundError  If no matching checkpoint files are found.

    Examples
    --------
    >>> time_mjd, obc_ssh = load_ssh_checkpoints(
    ...     'data_raw/hycom_ssh', year_range=(2018, 2020))
    >>> obc_ssh.shape
    (~26280, 95)
    """
    cache_dir = Path(cache_dir)
    files = sorted(cache_dir.glob("hycom_ssh_obc_*.npz"))

    if year_range is not None:
        y0, y1 = year_range
        files = [
            f for f in files
            if y0 <= int(f.stem.split("_")[3]) <= y1
        ]

    if not files:
        raise FileNotFoundError(
            f"No checkpoint files found in {cache_dir} "
            f"(year_range={year_range})."
        )

    time_list, ssh_list = [], []
    for f in files:
        data = np.load(f)
        time_list.append(data["time_mjd"])
        ssh_list.append(data["obc_ssh"])

    return np.concatenate(time_list), np.concatenate(ssh_list, axis=0)


# ── Time resampling ───────────────────────────────────────────────────────────

def resample_to_fvcom_time(
    time_mjd_hycom: np.ndarray,
    obc_ssh: np.ndarray,
    time_mjd_fvcom: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate OBC SSH from the HYCOM 3-hourly time grid to the
    FVCOM 6-minute time grid.

    Parameters
    ----------
    time_mjd_hycom : np.ndarray  (ntime_hycom,) float64 — HYCOM MJD axis.
    obc_ssh        : np.ndarray  (ntime_hycom, nobc) float32 — HYCOM SSH.
    time_mjd_fvcom : np.ndarray  (ntime_fvcom,) float64 — FVCOM MJD axis.

    Returns
    -------
    np.ndarray  (nobc, ntime_fvcom) float32
        SSH resampled to the FVCOM time grid, transposed to match the F01
        ``zeta`` array shape convention (node-major).

    Notes
    -----
    Linear extrapolation (``fill_value='extrapolate'``) is used at the
    small time windows at the ends of the record where the HYCOM and FVCOM
    axes do not overlap exactly.

    Examples
    --------
    >>> ssh_lf = resample_to_fvcom_time(time_mjd_hycom, obc_ssh, time_mjd_fvcom)
    >>> ssh_lf.shape
    (95, 263040)
    """
    interp_func = interp1d(
        time_mjd_hycom, obc_ssh,
        axis=0, kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return interp_func(time_mjd_fvcom).T.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# F03 — HYCOM Temperature / Salinity OBC
# ══════════════════════════════════════════════════════════════════════════════
#
# Reference MATLAB algorithm:
#   fvcom_prepro/interp_coarse_to_obc.m
#
# Horizontal: 16-nearest valid points → LinearNDInterpolator → IDW fallback
# Vertical:   depth-normalization + PchipInterpolator (extrap)
# Masking:    sentinel > 1.26e29 → NaN; salt < 0 → NaN; temp < -20 → NaN

# HYCOM standard depth levels (metres, positive-down)
HYCOM_Z_LEVELS = np.array([
    0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0,
    30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
    125.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0,
    900.0, 1000.0, 1250.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 5500.0,
], dtype=np.float64)

# Fill-value sentinel used in HYCOM NetCDF (values > this are invalid)
_HYCOM_FILL_THRESHOLD = 1.26e29


# ── Private helper: fetch a 3-D variable block ────────────────────────────────

def _fetch_ts_block(
    base_url: str,
    variables: list[str],
    t0: int, t1: int,
    depth0: int, depth1: int,
    lat0: int, lat1: int,
    lon0: int, lon1: int,
    max_retries: int = 5,
    retry_delay: float = 10.0,
    backoff: float = 2.0,
    chunk_t: int = 20,
) -> xr.Dataset:
    """
    Download a targeted block of HYCOM 3-D variables in time chunks.

    Uses DAP2 constraint expressions via the pydap engine.
    """
    n_t   = t1 - t0 + 1
    n_dep = depth1 - depth0 + 1
    n_lat = lat1 - lat0 + 1
    n_lon = lon1 - lon0 + 1

    chunk_starts = list(range(t0, t1 + 1, chunk_t))
    n_chunks     = len(chunk_starts)

    var_str = ",".join(variables)
    _log(
        f"  step 2 — {var_str} [{n_t}×{n_dep}×{n_lat}×{n_lon}]  "
        f"split into {n_chunks} chunk(s) of ≤{chunk_t} steps"
    )
    t0_wall = time.time()

    def _fetch_one_chunk(url: str, tc0: int, tc1: int) -> xr.Dataset:
        """Fetch one time-chunk via DAP2 constraint URL."""
        # Build constraint for each variable: var[t0:1:t1][d0:1:d1][lat0:1:lat1][lon0:1:lon1]
        var_constraints = ",".join(
            f"{v}[{tc0}:1:{tc1}][{depth0}:1:{depth1}][{lat0}:1:{lat1}][{lon0}:1:{lon1}]"
            for v in variables
        )
        constraint = (
            f"?{var_constraints}"
            f",time[{tc0}:1:{tc1}]"
            f",depth[{depth0}:1:{depth1}]"
            f",lat[{lat0}:1:{lat1}]"
            f",lon[{lon0}:1:{lon1}]"
        )
        ds = xr.open_dataset(
            url + constraint,
            engine="pydap",
            decode_times=True,
            mask_and_scale=True,
        )
        ds.load()
        return ds

    chunks: list[xr.Dataset] = []
    for i, tc0 in enumerate(chunk_starts, 1):
        tc1 = min(tc0 + chunk_t - 1, t1)
        t_wall = time.time()

        def _do_chunk(url: str, _tc0: int = tc0, _tc1: int = tc1) -> xr.Dataset:
            return _fetch_one_chunk(url, _tc0, _tc1)

        chunk_ds = _retry(_do_chunk, base_url, max_retries, retry_delay, backoff)
        chunks.append(chunk_ds)
        _log(
            f"    chunk {i}/{n_chunks}  t=[{tc0}:{tc1}]  "
            f"({tc1 - tc0 + 1} steps)  {time.time() - t_wall:.1f} s"
        )

    result = xr.concat(chunks, dim="time")
    _log(f"  step 2 done  total {time.time() - t0_wall:.1f} s")
    return result


# ── Public: fetch one month of HYCOM T/S ──────────────────────────────────────

def fetch_hycom_ts_month(
    year: int,
    month: int,
    lon_range: tuple[float, float] = DEFAULT_LON_RANGE,
    lat_range: tuple[float, float] = DEFAULT_LAT_RANGE,
    cache_dir: Optional[str | Path] = None,
    variables: Optional[list[str]] = None,
    max_retries: int = 5,
    retry_delay: float = 10.0,
    backoff: float = 2.0,
    chunk_t: int = 20,
    max_depth: float = 200.0,
) -> xr.Dataset:
    """
    Download one calendar month of HYCOM 3-D temperature and salinity.

    Parameters
    ----------
    year, month : int
        Calendar year and month (1–12).
    lon_range : (lon_min, lon_max)
        Degrees East [0, 360]. Default: Delaware shelf (283, 288).
    lat_range : (lat_min, lat_max)
        Degrees North. Default: (36, 41).
    cache_dir : path or None
        Directory for ASCII log.
    variables : list[str] or None
        HYCOM variable names. Default: ``['water_temp', 'salinity']``.
    max_retries : int   Maximum retry attempts (default 5).
    retry_delay : float Initial retry wait seconds (default 10).
    backoff     : float Exponential backoff multiplier (default 2).
    chunk_t     : int   Time steps per chunk (default 20 — 3-D data is large).
    max_depth   : float Maximum depth in metres to download (default 200 m).
                        HYCOM levels deeper than this are skipped.

    Returns
    -------
    xr.Dataset with variables (time, depth, lat, lon).
    """
    global _LOG_PATH

    if variables is None:
        variables = ["water_temp", "salinity"]
    variables = _normalize_hycom_variables(variables)
    three_d_variables = [v for v in variables if v in HYCOM_3D_VARIABLES]
    if len(three_d_variables) != len(variables):
        raise ValueError(
            "fetch_hycom_ts_month only supports 3-D HYCOM variables: "
            f"{sorted(HYCOM_3D_VARIABLES)}"
        )
    variables = three_d_variables

    first_day = date(year, month, 1)
    last_day  = date(year, month, calendar.monthrange(year, month)[1])
    base_url  = _build_base_url(first_day)

    if _LOG_PATH is None and cache_dir is not None:
        _LOG_PATH = Path(cache_dir) / "hycom_fetch.log"

    _log(f"── TS {year}-{month:02d} → {base_url.split('dodsC/')[-1]}")

    # Step 1 — lightweight coordinate fetch (reuse existing helper)
    coords = _fetch_coords(base_url, max_retries, retry_delay, backoff)

    # Also fetch depth coordinate
    _log("  fetching depth coordinate …")

    def _open_depth(url):
        return _fetch_depth_values(url, max_retries, retry_delay, backoff)

    hycom_depth = _open_depth(base_url)
    _log(f"  depth: {len(hycom_depth)} levels, max={hycom_depth.max():.0f} m")

    # Determine depth index range (only download levels ≤ max_depth + buffer)
    depth0, depth1 = _depth_index_range(hycom_depth, max_depth=max_depth)

    t_start = np.datetime64(first_day.isoformat(), "s")
    t_end   = np.datetime64(last_day.isoformat(), "s") + np.timedelta64(23, "h")

    t0, t1, lat0, lat1, lon0, lon1 = _find_indices(
        coords, lon_range, lat_range, t_start, t_end
    )
    _log(
        f"  indices: t=[{t0}:{t1}] depth=[{depth0}:{depth1}] "
        f"lat=[{lat0}:{lat1}] lon=[{lon0}:{lon1}]  "
        f"→ {t1 - t0 + 1} steps × {depth1 - depth0 + 1} levels"
    )

    # Step 2 — targeted data download
    ds = _fetch_ts_block(
        base_url, variables,
        t0, t1, depth0, depth1, lat0, lat1, lon0, lon1,
        max_retries, retry_delay, backoff, chunk_t=chunk_t,
    )

    # Apply masking: sentinel > 1.26e29, physical bounds
    for var in variables:
        if var not in ds:
            continue
        arr = ds[var].values
        arr[arr > _HYCOM_FILL_THRESHOLD] = np.nan
        if "salinity" in var.lower() or "salt" in var.lower():
            arr[arr < 0] = np.nan
        if "temp" in var.lower():
            arr[arr < -20] = np.nan
        ds[var].values = arr

    return ds


# ── Public: horizontal interpolation T/S to OBC nodes ────────────────────────

def _mask_hycom_variables(ds: xr.Dataset, variables: Sequence[str]) -> xr.Dataset:
    """Apply common HYCOM fill-value and simple physical masks in place."""
    for var in variables:
        if var not in ds:
            continue
        arr = ds[var].values
        arr[arr > _HYCOM_FILL_THRESHOLD] = np.nan
        lower_name = var.lower()
        if "salinity" in lower_name or "salt" in lower_name:
            arr[arr < 0] = np.nan
        if "temp" in lower_name:
            arr[arr < -20] = np.nan
        ds[var].values = arr
    return ds


def fetch_hycom(request: HycomDownloadRequest) -> xr.Dataset:
    """Fetch arbitrary HYCOM variables over a bounded date and space request.

    This driver composes the existing monthly/block helpers into a request-level
    API that can be reused by notebooks, Codex sessions, Hermes drivers, and
    FVCOM preprocessing scripts. It supports 2-D surf_el and 3-D temperature,
    salinity, water_u, and water_v fields.
    """
    global _LOG_PATH

    variables = _normalize_hycom_variables(request.variables)
    variables_2d = [v for v in variables if v in HYCOM_2D_VARIABLES]
    variables_3d = [v for v in variables if v in HYCOM_3D_VARIABLES]

    if _LOG_PATH is None and request.cache_dir is not None:
        _LOG_PATH = Path(request.cache_dir) / "hycom_fetch.log"

    datasets: list[xr.Dataset] = []
    for chunk in plan_hycom_chunks(request):
        base_url = str(chunk["base_url"])
        chunk_start = _parse_request_datetime(str(chunk["start"]))
        chunk_end = _parse_request_datetime(str(chunk["end"]))

        _log(
            f"-- {request.label} chunk {chunk['chunk_index']}: "
            f"{chunk['start']} to {chunk['end']} -> "
            f"{chunk['product']}/{chunk['expt']}"
        )

        coords = _fetch_coords(
            base_url,
            request.max_retries,
            request.retry_delay,
            request.backoff,
        )
        t0, t1, lat0, lat1, lon0, lon1 = _find_indices(
            coords,
            request.lon_range,
            request.lat_range,
            np.datetime64(chunk_start, "s"),
            np.datetime64(chunk_end, "s"),
        )

        pieces: list[xr.Dataset] = []
        if variables_2d:
            pieces.append(
                _fetch_ssh_block(
                    base_url,
                    t0, t1,
                    lat0, lat1,
                    lon0, lon1,
                    request.max_retries,
                    request.retry_delay,
                    request.backoff,
                    chunk_t=request.ssh_chunk_t,
                )
            )

        if variables_3d:
            hycom_depth = _fetch_depth_values(
                base_url,
                request.max_retries,
                request.retry_delay,
                request.backoff,
            )
            depth0, depth1 = _depth_index_range(
                hycom_depth,
                depth_range=request.depth_range,
                max_depth=request.max_depth,
            )
            pieces.append(
                _fetch_ts_block(
                    base_url,
                    variables_3d,
                    t0, t1,
                    depth0, depth1,
                    lat0, lat1,
                    lon0, lon1,
                    request.max_retries,
                    request.retry_delay,
                    request.backoff,
                    chunk_t=request.chunk_t,
                )
            )

        if not pieces:
            raise ValueError("No HYCOM variables were selected for download.")
        chunk_ds = pieces[0] if len(pieces) == 1 else xr.merge(pieces, compat="override")
        datasets.append(_mask_hycom_variables(chunk_ds, variables))

    return datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim="time")


def _coerce_points(points: Any) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Normalize point/station inputs into names, lon array, and lat array."""
    if points is None:
        raise ValueError("points must be supplied for HYCOM point extraction.")

    if isinstance(points, dict):
        names: list[str] = []
        lons: list[float] = []
        lats: list[float] = []
        for name, value in points.items():
            if isinstance(value, dict):
                lon = value.get("lon", value.get("longitude"))
                lat = value.get("lat", value.get("latitude"))
            else:
                lon, lat = value
            if lon is None or lat is None:
                raise ValueError(f"Point '{name}' must include lon and lat.")
            names.append(str(name))
            lons.append(float(lon))
            lats.append(float(lat))
        return names, np.asarray(lons, dtype=np.float64), np.asarray(lats, dtype=np.float64)

    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError("A single point must be [lon, lat].")
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must be a dict or an array-like of [lon, lat] rows.")
    names = [f"point_{i + 1}" for i in range(arr.shape[0])]
    return names, arr[:, 0], arr[:, 1]


def _normalize_target_lons_for_grid(target_lon: np.ndarray, remote_lon: np.ndarray) -> np.ndarray:
    """Convert target longitudes to the remote HYCOM grid convention."""
    result = target_lon.astype(np.float64).copy()
    if np.all(remote_lon >= 0):
        result = np.where(result < 0.0, result + 360.0, result)
    else:
        result = np.where(result > 180.0, result - 360.0, result)
    return result


def _interp_da_to_points(
    da: xr.DataArray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> xr.DataArray:
    """Interpolate one HYCOM DataArray to point locations."""
    lon_arr = da["lon"].values.astype(np.float64)
    lat_arr = da["lat"].values.astype(np.float64)
    interp_pts = np.column_stack([target_lat, target_lon])

    if "depth" in da.dims:
        data = da.transpose("time", "depth", "lat", "lon").values
        out = np.empty((data.shape[0], data.shape[1], len(target_lon)), dtype=np.float32)
        for t in range(data.shape[0]):
            for k in range(data.shape[1]):
                rgi = RegularGridInterpolator(
                    (lat_arr, lon_arr),
                    data[t, k],
                    method="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                out[t, k] = rgi(interp_pts).astype(np.float32)
        return xr.DataArray(
            out,
            dims=("time", "depth", "point"),
            coords={"time": da["time"], "depth": da["depth"]},
            name=da.name,
        )

    data = da.transpose("time", "lat", "lon").values
    out2d = np.empty((data.shape[0], len(target_lon)), dtype=np.float32)
    for t in range(data.shape[0]):
        rgi = RegularGridInterpolator(
            (lat_arr, lon_arr),
            data[t],
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        out2d[t] = rgi(interp_pts).astype(np.float32)
    return xr.DataArray(
        out2d,
        dims=("time", "point"),
        coords={"time": da["time"]},
        name=da.name,
    )


def fetch_hycom_points(
    request: HycomDownloadRequest,
    points: Optional[Any] = None,
) -> xr.Dataset:
    """Fetch a bounded HYCOM block and interpolate requested variables to points."""
    point_names, point_lon, point_lat = _coerce_points(points if points is not None else request.points)
    ds_grid = fetch_hycom(request)
    target_lon = _normalize_target_lons_for_grid(point_lon, ds_grid["lon"].values)

    ds_points = xr.Dataset(
        coords={
            "point": np.asarray(point_names, dtype=object),
            "lon": ("point", point_lon.astype(np.float64)),
            "lat": ("point", point_lat.astype(np.float64)),
        }
    )
    for var in _normalize_hycom_variables(request.variables):
        if var in ds_grid:
            ds_points[var] = _interp_da_to_points(ds_grid[var], target_lon, point_lat)
    return ds_points


def interp_ts_to_obc(
    ds_ts: xr.Dataset,
    obc_lon: np.ndarray,
    obc_lat: np.ndarray,
    variables: Optional[list[str]] = None,
) -> dict:
    """
    Horizontally interpolate HYCOM 3-D T/S fields to OBC node locations.

    Follows the algorithm of MATLAB ``interp_coarse_to_obc.m``:
    - Per depth level, per timestep: LinearNDInterpolator on valid points
    - NearestNDInterpolator fallback for NaN results (coastal nodes)
    - Surface guarantee: if surface returns NaN at any node, fill with nearest

    Since HYCOM is a regular lat-lon grid, we use RegularGridInterpolator
    for speed, falling back to scattered interpolation for masked levels.

    Parameters
    ----------
    ds_ts    : xr.Dataset with dims (time, depth, lat, lon).
    obc_lon  : (nobc,) longitudes in [0, 360].
    obc_lat  : (nobc,) latitudes.
    variables : list[str] — variable names to interpolate (default: all 4-D vars).

    Returns
    -------
    dict with keys:
        'temp'     : (ntime, ndepth, nobc) float32
        'salt'     : (ntime, ndepth, nobc) float32
        'depth'    : (ndepth,) float64 — HYCOM depth levels [m, positive-down]
        'time_mjd' : (ntime,) float64 — Modified Julian Day
    """
    try:
        from .grid_utils import datetime64_to_mjd
    except ImportError:
        from grid_utils import datetime64_to_mjd

    if variables is None:
        # Auto-detect 4-D variables
        variables = [v for v in ds_ts.data_vars if ds_ts[v].ndim == 4]

    lon_arr = ds_ts["lon"].values.astype(np.float64)
    lat_arr = ds_ts["lat"].values.astype(np.float64)
    dep_arr = ds_ts["depth"].values.astype(np.float64)

    nobc  = len(obc_lon)
    ntime = ds_ts.dims["time"]
    ndepth = len(dep_arr)

    target_pts = np.column_stack([obc_lon.astype(np.float64),
                                  obc_lat.astype(np.float64)])

    # Precompute meshgrid for scattered fallback
    lons_2d, lats_2d = np.meshgrid(lon_arr, lat_arr)
    pts_flat = np.column_stack([lons_2d.ravel(), lats_2d.ravel()])

    result = {}
    for var in variables:
        data4d = ds_ts[var].values  # (ntime, ndepth, nlat, nlon)
        obc_data = np.empty((ntime, ndepth, nobc), dtype=np.float32)

        for t in range(ntime):
            for k in range(ndepth):
                field_2d = data4d[t, k]  # (nlat, nlon)

                # Check how many valid points
                valid_mask = np.isfinite(field_2d)
                n_valid = valid_mask.sum()

                if n_valid < 4:
                    # Too few points — use nearest from whatever is available
                    if n_valid > 0:
                        vals_flat = field_2d.ravel()
                        v_mask = np.isfinite(vals_flat)
                        nn = NearestNDInterpolator(pts_flat[v_mask], vals_flat[v_mask])
                        obc_data[t, k] = nn(target_pts).astype(np.float32)
                    else:
                        obc_data[t, k] = np.nan
                    continue

                # Use RegularGridInterpolator for structured HYCOM grid
                # It requires a full rectangular grid — handle NaN via fill then fix
                field_filled = field_2d.copy()
                if not valid_mask.all():
                    # For points that are NaN (land/seafloor), fill temporarily
                    # with nearest valid value so RGI doesn't propagate NaN
                    from scipy.ndimage import distance_transform_edt
                    invalid = ~valid_mask
                    # indices of nearest valid point
                    _, nearest_idx = distance_transform_edt(
                        invalid, return_distances=True, return_indices=True
                    )
                    field_filled[invalid] = field_2d[
                        nearest_idx[0][invalid], nearest_idx[1][invalid]
                    ]

                try:
                    rgi = RegularGridInterpolator(
                        (lat_arr, lon_arr), field_filled,
                        method="linear", bounds_error=False,
                        fill_value=None,  # extrapolate
                    )
                    # RGI expects (lat, lon) order
                    interp_pts = np.column_stack([obc_lat, obc_lon])
                    vals = rgi(interp_pts).astype(np.float32)
                except Exception:
                    # Fallback to scattered interpolation
                    vals_flat = field_2d.ravel()
                    v_mask = np.isfinite(vals_flat)
                    lin = LinearNDInterpolator(pts_flat[v_mask], vals_flat[v_mask])
                    vals = lin(target_pts).astype(np.float32)
                    nan_mask = ~np.isfinite(vals)
                    if nan_mask.any():
                        nn = NearestNDInterpolator(pts_flat[v_mask], vals_flat[v_mask])
                        vals[nan_mask] = nn(target_pts[nan_mask]).astype(np.float32)

                obc_data[t, k] = vals

                # Surface guarantee: no NaN allowed at surface level (k=0)
                if k == 0:
                    nan_nodes = ~np.isfinite(obc_data[t, 0])
                    if nan_nodes.any():
                        vals_flat = field_2d.ravel()
                        v_mask = np.isfinite(vals_flat)
                        if v_mask.any():
                            nn = NearestNDInterpolator(
                                pts_flat[v_mask], vals_flat[v_mask]
                            )
                            obc_data[t, 0, nan_nodes] = nn(
                                target_pts[nan_nodes]
                            ).astype(np.float32)

        # Map variable name to canonical key
        if "temp" in var.lower():
            result["temp"] = obc_data
        elif "sal" in var.lower():
            result["salt"] = obc_data
        elif var.lower() == "water_u":
            result["u"] = obc_data
        elif var.lower() == "water_v":
            result["v"] = obc_data
        else:
            result[var] = obc_data

    # Time in MJD
    time_np = ds_ts["time"].values.astype("datetime64[s]")
    result["time_mjd"] = datetime64_to_mjd(time_np)
    result["depth"] = dep_arr

    return result


# ── F03 checkpoint helpers ────────────────────────────────────────────────────

def _ts_checkpoint_path(cache_dir: Path, year: int, month: int) -> Path:
    """Standard checkpoint filename for T/S OBC data."""
    return Path(cache_dir) / f"hycom_ts_obc_{year}_{month:02d}.npz"


def save_ts_checkpoint(
    temp_obc: np.ndarray,
    salt_obc: np.ndarray,
    time_mjd: np.ndarray,
    hycom_depth: np.ndarray,
    filepath: str | Path,
) -> Path:
    """
    Save horizontally-interpolated T/S OBC data to compressed NumPy archive.

    Parameters
    ----------
    temp_obc   : (ntime, ndepth, nobc) float32
    salt_obc   : (ntime, ndepth, nobc) float32
    time_mjd   : (ntime,) float64
    hycom_depth: (ndepth,) float64 — HYCOM depth levels [m]
    filepath   : destination path

    Returns
    -------
    Path to saved file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(filepath),
        temp_obc=temp_obc,
        salt_obc=salt_obc,
        time_mjd=time_mjd,
        hycom_depth=hycom_depth,
    )
    return filepath.resolve()


def load_ts_checkpoints(
    cache_dir: str | Path,
    year_range: Optional[tuple[int, int]] = None,
) -> dict:
    """
    Load and chronologically concatenate all monthly T/S OBC checkpoint files.

    Parameters
    ----------
    cache_dir  : directory containing ``hycom_ts_obc_YYYY_MM.npz`` files.
    year_range : (year_start, year_end) inclusive, or None for all.

    Returns
    -------
    dict with keys:
        'temp_obc'   : (ntime_total, ndepth, nobc) float32
        'salt_obc'   : (ntime_total, ndepth, nobc) float32
        'time_mjd'   : (ntime_total,) float64
        'hycom_depth': (ndepth,) float64
    """
    cache_dir = Path(cache_dir)
    files = sorted(cache_dir.glob("hycom_ts_obc_*.npz"))

    if year_range is not None:
        y0, y1 = year_range
        files = [
            f for f in files
            if y0 <= int(f.stem.split("_")[3]) <= y1
        ]

    if not files:
        raise FileNotFoundError(
            f"No T/S checkpoint files in {cache_dir} (year_range={year_range})."
        )

    time_list, temp_list, salt_list = [], [], []
    hycom_depth = None

    for f in files:
        data = np.load(f)
        time_list.append(data["time_mjd"])
        temp_list.append(data["temp_obc"])
        salt_list.append(data["salt_obc"])
        if hycom_depth is None:
            hycom_depth = data["hycom_depth"]

    return {
        "time_mjd":    np.concatenate(time_list),
        "temp_obc":    np.concatenate(temp_list, axis=0),
        "salt_obc":    np.concatenate(salt_list, axis=0),
        "hycom_depth": hycom_depth,
    }


# ── F03 vertical remapping ───────────────────────────────────────────────────

def remap_hycom_z_to_sigma(
    data_z: np.ndarray,
    hycom_depth: np.ndarray,
    node_depths: np.ndarray,
    sigma_levels: np.ndarray,
) -> np.ndarray:
    """
    Remap horizontally-interpolated HYCOM data from z-levels to FVCOM sigma.

    Follows the MATLAB ``interp_coarse_to_obc.m`` vertical algorithm:
    1. For each OBC node, compute FVCOM sigma target depths.
    2. Mask out below-seafloor NaN in source profile.
    3. Apply depth normalization to stretch coarse z-range to FVCOM z-range.
    4. Interpolate with PchipInterpolator (equivalent to MATLAB pchip+extrap).
    5. Post-interpolation: nearest-neighbor fill any remaining NaN across nodes.

    Parameters
    ----------
    data_z       : (ntime, ndepth_hycom, nobc) — values on HYCOM z-levels.
    hycom_depth  : (ndepth_hycom,) — HYCOM depth levels [m, positive-down].
    node_depths  : (nobc,) — FVCOM water depth at each OBC node [m, positive].
    sigma_levels : (nsiglay,) — FVCOM siglay values in [-1, 0].
                   siglay[0] ≈ 0 (near-surface), siglay[-1] ≈ -1 (near-bed).

    Returns
    -------
    data_sigma : (ntime, nsiglay, nobc) float32.
    """
    ntime, ndepth, nobc = data_z.shape
    nsiglay = len(sigma_levels)
    data_sigma = np.full((ntime, nsiglay, nobc), np.nan, dtype=np.float64)

    # Source z-levels (negative-up, for interpolation):  0, -2, -4, …
    z_hycom = -hycom_depth.astype(np.float64)  # shape (ndepth,), ≤ 0

    for i_node in range(nobc):
        h = float(node_depths[i_node])
        # FVCOM target depths (negative-up): sigma * h
        z_fvcom = sigma_levels.astype(np.float64) * h  # (nsiglay,), ≤ 0

        for t in range(ntime):
            profile = data_z[t, :, i_node].astype(np.float64)

            # Find valid (non-NaN) levels
            valid = np.isfinite(profile)
            n_valid = valid.sum()

            if n_valid < 2:
                # Not enough data — fill with single value or leave NaN
                if n_valid == 1:
                    data_sigma[t, :, i_node] = profile[valid][0]
                continue

            z_valid = z_hycom[valid]       # sorted: most negative first (deepest)
            v_valid = profile[valid]

            # Depth normalization (MATLAB interp_coarse_to_obc.m):
            # Stretch coarse z-range [A, B] to FVCOM z-range [C, D]
            A = z_valid[-1]   # shallowest coarse level (least negative, e.g. 0)
            B = z_valid[0]    # deepest coarse level (most negative)
            C = z_fvcom[0]    # shallowest FVCOM (least negative, near 0)
            D = z_fvcom[-1]   # deepest FVCOM (most negative, near -h)

            denom = A - B
            if abs(denom) < 1e-6:
                # All at same depth — fill constant
                data_sigma[t, :, i_node] = v_valid[0]
                continue

            norm_z = ((D - C) / denom) * (z_valid - B) + D
            # norm_z goes from D (deepest) to C (shallowest) — same direction as z_fvcom

            # Ensure monotonically increasing for PchipInterpolator
            if norm_z[0] > norm_z[-1]:
                norm_z = norm_z[::-1]
                v_valid = v_valid[::-1]

            # Remove any duplicates (can happen at boundaries)
            unique_mask = np.diff(norm_z, prepend=-np.inf) > 0
            norm_z = norm_z[unique_mask]
            v_valid = v_valid[unique_mask]

            if len(norm_z) < 2:
                data_sigma[t, :, i_node] = v_valid[0] if len(v_valid) > 0 else np.nan
                continue

            interp_fn = PchipInterpolator(norm_z, v_valid, extrapolate=False)
            result = interp_fn(z_fvcom)

            # Nearest-neighbor clamp for targets outside the source range
            # (PCHIP extrapolation diverges wildly near surface/bed)
            above = z_fvcom > norm_z[-1]  # above shallowest source level
            below = z_fvcom < norm_z[0]   # below deepest source level
            if above.any():
                result[above] = v_valid[-1]  # nearest = shallowest value
            if below.any():
                result[below] = v_valid[0]   # nearest = deepest value

            data_sigma[t, :, i_node] = result

    # Post-interpolation NaN fill: nearest-neighbor across nodes at each sigma level
    for t in range(ntime):
        for k in range(nsiglay):
            row = data_sigma[t, k, :]
            nan_mask = ~np.isfinite(row)
            if nan_mask.any() and not nan_mask.all():
                valid_idx = np.where(~nan_mask)[0]
                nan_idx = np.where(nan_mask)[0]
                # Find nearest valid node for each NaN node
                nearest = valid_idx[
                    np.argmin(np.abs(nan_idx[:, None] - valid_idx[None, :]), axis=1)
                ]
                data_sigma[t, k, nan_idx] = row[nearest]

    return data_sigma.astype(np.float32)
