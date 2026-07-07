"""
usgs_rivers_fetcher.py
======================
Fetch USGS NWIS streamflow and build FVCOM river forcing arrays (F06).

Forcing: F06 -- River volume flux, salinity, floc (coarse_sand), microplastic.

Reference workflow
------------------
Old MATLAB:
  data_source/2019_river/Trenton_river_2019_whole_year.txt   -- Delaware main
  data_source/2019_river/Schuylkill_river_2019_whole_year.txt
  d_river_forcing.m (river_forcing_floc_MP=1 block)
  fvcom_prepro/write_FVCOM_river.m
  fvcom_prepro/write_FVCOM_river_nml.m

River node mapping (waterPACT DRE, floc_MP config)
------------------
  DR_1 .. DR_6 : Delaware River inflow nodes
                 Main gauge: Trenton, NJ -- USGS 01463500
                 Flow split equally across 6 nodes (1/6 each)
  SR_1 .. SR_3 : Schuylkill River inflow nodes
                 Main gauge: Schuylkill at Philadelphia -- USGS 01474500
                 Flow split equally across 3 nodes (1/3 each)

Tracer formulas (floc_MP scenario)
-----------------------------------
  Floc (coarse_sand_1): Nash (1994) gives the mass load
      Q_s = 0.01 * Q^1.8 [metric tons/day], with Q in m^3/s.
      Dividing that load by the river volume discharge gives the
      equivalent concentration C_floc = 1.1574074e-4 * Q^0.8 [g/l].
      Source: Nash (1994) rating curve, McSweeney thesis for Delaware River
  Legacy MP (mp1): constant = 100 * 35.31 * 4.97e-10 ~ 1.754e-6 g/l
      Source: 2019 PE measurements (Delaware + Schuylkill, d_river_forcing.m).
      The P0-P7 analytical writer now defaults to DRBC microfiber forcing in
      microplastic_forcing.py, while preserving this value for reproducibility.

Python dependencies
-------------------
  dataretrieval (USGS NWIS), pandas, numpy
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# River node configuration (waterPACT DRE, floc_MP config)
# ---------------------------------------------------------------------------

#: River entry names as they appear in the FVCOM NML and NetCDF file.
RIVER_NAMES: list[str] = [
    "DR_1", "DR_2", "DR_3", "DR_4", "DR_5", "DR_6",
    "SR_1", "SR_2", "SR_3",
]

#: 1-based FVCOM node indices for each river entry (from waterPACT_river_floc_MP.nml).
RIVER_NODES: list[int] = [
    68412, 68416, 68413, 68414, 68415, 68410,   # DR_1..DR_6
    55093, 55092, 55075,                          # SR_1..SR_3
]

#: Equal-weight flow split for Delaware River nodes.
DELAWARE_NODE_WEIGHTS: list[float] = [1.0 / 6.0] * 6

#: Equal-weight flow split for Schuylkill River nodes.
SCHUYLKILL_NODE_WEIGHTS: list[float] = [1.0 / 3.0] * 3

# ---------------------------------------------------------------------------
# USGS gauge site numbers
# ---------------------------------------------------------------------------

USGS_SITES: dict[str, str] = {
    "Delaware_Trenton":     "01463500",
    "Schuylkill_Philly":    "01474500",
    # Additional gauges (not used in floc_MP config but available)
    "Brandywine_Cr":        "01481500",
    "Neshaminy_Cr":         "01465500",
    "Pennypack_Cr":         "01467048",
    "Cooper_R":             "01477120",
    "Assunpink_Cr":         "01464500",
    "NRancocas_Cr":         "01466500",
    "SRancocas_Cr":         "01467000",
    "Cohansey_R":           "01412800",
    "Maurice_R":            "01411500",
    "Frankford_Cr":         "01467087",
    "Mantua_Cr":            "01476500",
    "Shellpot_Cr":          "01476010",
}

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: ft³/s → m³/s conversion (USGS reports discharge in cfs).
_CFS_TO_M3S: float = 0.0283168

#: Legacy constant MP concentration for the original floc_MP scenario [g/l].
#: = 100 particles × 35.31 ft³ × 4.97e-10 kg/particle (2019 PE measurements).
MP_CONC_FLOC_MP: float = 100.0 * 35.31 * 4.97e-10   # ~1.754e-6 g/l


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_usgs_discharge(site_no: str,
                         t_start: str,
                         t_end: str,
                         cache_dir: str | Path,
                         param_code: str = "00060") -> pd.DataFrame:
    """
    Fetch daily mean discharge from USGS NWIS.

    Tries the live ``dataretrieval`` API first; on failure reads from a
    pre-downloaded CSV cache file ``{cache_dir}/{site_no}.csv``.

    CSV cache format (two-column, header row)::

        datetime,discharge_m3s
        2018-01-01,300.45
        ...

    Parameters
    ----------
    site_no    : USGS site number, e.g. ``'01463500'``
    t_start    : ISO date string, e.g. ``'2018-01-01'``
    t_end      : ISO date string, e.g. ``'2020-12-31'``
    cache_dir  : directory for storing / reading CSV cache files
    param_code : NWIS parameter code; ``'00060'`` = discharge [ft³/s]

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC, no TZ) and column
    ``'discharge_m3s'`` [m³/s].
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{site_no}.csv"

    # --- Try live NWIS API ---
    try:
        import warnings
        import dataretrieval.nwis as nwis
        log.info("Fetching USGS site %s from NWIS API (%s – %s) ...",
                 site_no, t_start, t_end)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            df_raw, _ = nwis.get_dv(
                sites=site_no,
                parameterCd=param_code,
                start=t_start,
                end=t_end,
            )
        if df_raw is None or df_raw.empty:
            raise ValueError(f"NWIS returned empty result for site {site_no}")

        # Column name is like '00060_Mean'; pick first numeric discharge column
        q_cols = [c for c in df_raw.columns
                  if param_code in str(c) and "cd" not in str(c).lower()]
        if not q_cols:
            raise KeyError(
                f"No discharge column found in NWIS result; "
                f"columns: {df_raw.columns.tolist()}"
            )
        q_col = q_cols[0]

        # Strip timezone if present (NWIS returns UTC-aware index)
        idx = df_raw.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert(None)

        df = pd.DataFrame(
            {"discharge_m3s": df_raw[q_col].values.astype(float) * _CFS_TO_M3S},
            index=idx,
        )
        df.index.name = "datetime"
        df = df.loc[t_start:t_end].copy()

        # Persist cache
        df.to_csv(cache_file)
        log.info("  Saved to cache %s  (%d daily values)", cache_file.name, len(df))
        return df

    except Exception as exc:
        log.warning(
            "NWIS API unavailable for site %s (%s); trying CSV cache ...",
            site_no, exc,
        )

    # --- Fall back to pre-downloaded CSV cache ---
    if not cache_file.exists():
        raise FileNotFoundError(
            f"NWIS API failed and no CSV cache found at: {cache_file}\n"
            "  Provide a two-column CSV with header [datetime, discharge_m3s]."
        )
    log.info("Reading CSV cache: %s", cache_file.name)
    df = pd.read_csv(cache_file, parse_dates=["datetime"], index_col="datetime")
    df.index = pd.DatetimeIndex(df.index)
    return df.loc[t_start:t_end, ["discharge_m3s"]].copy()


def discharge_to_hourly(df: pd.DataFrame,
                        t_start: str,
                        t_end: str) -> pd.Series:
    """
    Resample daily mean discharge to hourly by linear time-interpolation.

    USGS daily values are assigned to noon UTC (mid-day representative);
    linear interpolation fills the hourly grid; edge extrapolation uses
    nearest-neighbour (``limit_direction='both'``).

    Parameters
    ----------
    df      : DataFrame from :func:`fetch_usgs_discharge` (daily, DatetimeIndex)
    t_start : ISO date string for first hourly output step (00:00 UTC)
    t_end   : ISO date string for last  hourly output step (inclusive)

    Returns
    -------
    pd.Series of hourly discharge [m³/s] with DatetimeIndex.
    """
    # Shift daily value to noon of each day
    daily = df["discharge_m3s"].copy()
    daily.index = pd.DatetimeIndex(daily.index) + pd.Timedelta("12h")

    # Build hourly target grid
    hourly_idx = pd.date_range(t_start, t_end, freq="h")

    # Merge and interpolate
    combined_idx = daily.index.union(hourly_idx).sort_values()
    combined = daily.reindex(combined_idx).interpolate(
        method="time", limit_direction="both"
    )
    result = combined.reindex(hourly_idx)

    if result.isna().any():
        n_nan = int(result.isna().sum())
        log.warning("discharge_to_hourly: %d NaN values remain after interpolation "
                    "(check USGS data coverage for %s–%s)", n_nan, t_start, t_end)
        result = result.ffill().bfill()

    return result


# ---------------------------------------------------------------------------
# Flow distribution
# ---------------------------------------------------------------------------

def distribute_flow_to_nodes(Q_total: np.ndarray,
                              node_weights: list[float]) -> np.ndarray:
    """
    Distribute total river discharge to multiple FVCOM inflow nodes.

    Parameters
    ----------
    Q_total      : (ntime,) total discharge [m³/s]
    node_weights : fractional weights summing to 1.0

    Returns
    -------
    (ntime, nnodes) per-node discharge [m³/s]
    """
    weights = np.asarray(node_weights, dtype=np.float64)
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"node_weights must sum to 1.0; got {weights.sum():.8f}"
        )
    return np.outer(np.asarray(Q_total, dtype=np.float64), weights)


# ---------------------------------------------------------------------------
# Tracer concentration formulas
# ---------------------------------------------------------------------------

def compute_nash_sediment_load(q: np.ndarray) -> np.ndarray:
    """
    Compute sediment mass load using the Nash (1994) rating curve.

        Q_s = 0.01 * Q^1.8   [metric tons/day]

    Parameters
    ----------
    q : river discharge [m3/s]; any broadcastable shape

    Returns
    -------
    Array of same shape [metric tons/day].
    """
    return 0.01 * np.asarray(q, dtype=np.float64) ** 1.8


def compute_floc_concentration(q_node: np.ndarray) -> np.ndarray:
    """
    Compute load-derived floc concentration using the Nash (1994) curve.

        C_floc = 1.1574e-4 × q_node^0.8   [g/l]

    Nash gives mass load first; this function converts that load to
    concentration by dividing by volume discharge. For split-node river
    forcings, pass total gauge discharge and then assign the concentration to
    each split FVCOM river node.

    Parameters
    ----------
    q_node : per-node discharge [m³/s]; any broadcastable shape

    Returns
    -------
    Array of same shape [g/l].
    """
    q_node = np.asarray(q_node, dtype=np.float64)
    load = compute_nash_sediment_load(q_node)
    out = np.zeros_like(q_node, dtype=np.float64)
    np.divide(load * 1.0e6, q_node * 86400.0 * 1000.0, out=out,
              where=q_node > 0.0)
    return out


def compute_mp_concentration(Q: np.ndarray,
                              a: float,
                              b: float) -> np.ndarray:
    """
    Compute microplastic concentration using a power-law C-Q relationship.

        C_mp [g/l] = a × Q^b

    Parameters
    ----------
    Q : discharge array [m³/s]
    a : coefficient
    b : exponent

    Returns
    -------
    Array of same shape as Q [g/l].
    """
    return a * np.asarray(Q, dtype=np.float64) ** b
