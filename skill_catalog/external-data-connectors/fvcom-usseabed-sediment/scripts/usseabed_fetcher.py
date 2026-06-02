"""
usseabed_fetcher.py
===================
Fetch, clean, subset, summarize, and plot USGS usSEABED tabular data.

The module is intentionally transferable: it knows the official usSEABED data
release URL and generic table operations, but it does not hard-code a model
domain or river-boundary definition.  Project notebooks can pass an FVCOM mesh
and region node definitions when they need Delaware-specific products.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import requests

USSEABED_BASE_URL = (
    "https://cmgds.marine.usgs.gov/data/whcmsc/data-release/"
    "doi-P9H3LGWM/unpacked/usSEABED_EEZ"
)

USSEABED_TABLES: dict[str, str] = {
    "US9_EXT": "US9_EXT.csv",
    "US9_ONE": "US9_ONE.csv",
    "US9_CLC": "US9_CLC.csv",
    "US9_PRS": "US9_PRS.csv",
    "US9_SRC": "US9_SRC.csv",
}

DEFAULT_USECOLS = [
    "Latitude",
    "Longitude",
    "WaterDepth",
    "ObsvnTop",
    "ObsvnBot",
    "LocnName",
    "DataSetKey",
    "LocnKey",
    "ObsvnKey",
    "Device",
    "DataTypes",
    "Gravel",
    "Sand",
    "Mud",
    "Clay",
    "Grainsze",
    "Sorting",
    "Facies",
    "FolkCde",
    "Key",
    "ObsvnDate",
    "DateSrc",
]

NUMERIC_COLUMNS = [
    "Latitude",
    "Longitude",
    "WaterDepth",
    "ObsvnTop",
    "ObsvnBot",
    "DataSetKey",
    "LocnKey",
    "ObsvnKey",
    "Gravel",
    "Sand",
    "Mud",
    "Clay",
    "Grainsze",
    "Sorting",
    "Carbonate",
    "OrgCarbn",
    "Porosity",
    "PWaveVel",
]

FRACTION_COLUMNS = ["Gravel", "Sand", "Mud", "Clay"]


def normalize_table_name(table: str) -> str:
    """Normalize table identifiers such as ``EXT`` or ``US9_EXT.csv``."""
    name = str(table).upper().strip()
    if name.endswith(".CSV"):
        name = name[:-4]
    if not name.startswith("US9_"):
        name = f"US9_{name}"
    if name not in USSEABED_TABLES:
        valid = ", ".join(sorted(USSEABED_TABLES))
        raise ValueError(f"Unknown usSEABED table {table!r}; valid: {valid}")
    return name


def table_url(table: str) -> str:
    """Return the official USGS CSV URL for a usSEABED table."""
    table_name = normalize_table_name(table)
    return f"{USSEABED_BASE_URL}/{USSEABED_TABLES[table_name]}"


def download_usseabed_table(
    table: str,
    cache_dir: str | Path,
    overwrite: bool = False,
    chunk_size: int = 1024 * 1024,
    timeout: int = 60,
) -> Path:
    """Download a usSEABED CSV table to ``cache_dir`` if needed.

    Parameters
    ----------
    table
        Table identifier, for example ``"US9_EXT"`` or ``"ONE"``.
    cache_dir
        Local cache directory, typically ``Workspace/Preprocessing/data_raw/usseabed``.
    overwrite
        If true, re-download even when the cached CSV already exists.
    chunk_size
        Streaming download chunk size in bytes.
    timeout
        HTTP connect/read timeout passed to :mod:`requests`.
    """
    table_name = normalize_table_name(table)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / USSEABED_TABLES[table_name]

    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return out_path

    url = table_url(table_name)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    tmp_path.replace(out_path)
    return out_path


def clean_usseabed_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known numeric columns and replace usSEABED ``-99`` sentinels."""
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] <= -98, col] = np.nan
    return df


def load_usseabed_table(
    table: str,
    cache_dir: str | Path,
    usecols: Sequence[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Load a cached usSEABED CSV, downloading it first if necessary.

    ``bbox`` is ``(lon_min, lon_max, lat_min, lat_max)`` and is applied after
    loading and cleaning.
    """
    path = download_usseabed_table(table, cache_dir, overwrite=overwrite)
    df = pd.read_csv(path, usecols=usecols)
    df = clean_usseabed_dataframe(df)

    if bbox is not None:
        lon_min, lon_max, lat_min, lat_max = bbox
        mask = (
            df["Longitude"].between(lon_min, lon_max)
            & df["Latitude"].between(lat_min, lat_max)
        )
        df = df.loc[mask].copy()

    df["source_table"] = normalize_table_name(table)
    return df.reset_index(drop=True)


def add_silt_and_class_validity(
    df: pd.DataFrame,
    valid_sum_range: tuple[float, float] = (95.0, 105.0),
) -> pd.DataFrame:
    """Add derived silt and class-normalized fraction columns.

    Silt is derived only where ``Mud >= Clay``.  Normalized class fractions are
    computed only where ``Gravel + Sand + Mud`` is within ``valid_sum_range``.
    """
    df = df.copy()
    if {"Mud", "Clay"}.issubset(df.columns):
        df["Silt"] = np.where(df["Mud"] >= df["Clay"], df["Mud"] - df["Clay"], np.nan)
    else:
        df["Silt"] = np.nan

    required = ["Gravel", "Sand", "Mud", "Clay", "Silt"]
    if all(col in df.columns for col in required):
        total = df[["Gravel", "Sand", "Mud"]].sum(axis=1, min_count=3)
        valid = (
            total.between(valid_sum_range[0], valid_sum_range[1])
            & df[["Gravel", "Sand", "Clay", "Silt"]].notna().all(axis=1)
            & (df[["Gravel", "Sand", "Clay", "Silt"]] >= 0).all(axis=1)
        )
        denom = df.loc[valid, ["Gravel", "Sand", "Silt", "Clay"]].sum(axis=1)
        for col in ["Gravel", "Sand", "Silt", "Clay"]:
            out_col = f"{col}_norm"
            df[out_col] = np.nan
            df.loc[valid, out_col] = 100.0 * df.loc[valid, col] / denom
        df["valid_fraction_sum"] = valid
    else:
        df["valid_fraction_sum"] = False
    return df


def filter_to_fvcom_mesh(df: pd.DataFrame, mesh: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Keep only rows whose lon/lat points fall inside an FVCOM triangular mesh."""
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(mesh["lon"], mesh["lat"], mesh["tri"])
    finder = tri.get_trifinder()
    elements = finder(df["Longitude"].to_numpy(), df["Latitude"].to_numpy())
    out = df.loc[elements >= 0].copy()
    out["fvcom_element"] = elements[elements >= 0].astype(np.int32)
    return out.reset_index(drop=True)


def _centroid_for_nodes(mesh: Mapping[str, np.ndarray], nodes_1based: Sequence[int]) -> np.ndarray:
    idx = np.asarray(nodes_1based, dtype=int) - 1
    return np.array([np.nanmean(mesh["lon"][idx]), np.nanmean(mesh["lat"][idx])], dtype=float)


def _distance_km(lon: np.ndarray, lat: np.ndarray, target_lonlat: np.ndarray) -> np.ndarray:
    lat0 = np.deg2rad(np.nanmean(lat))
    scale = np.array([111.32 * np.cos(lat0), 111.32])
    points = np.column_stack([lon, lat])
    return np.linalg.norm((points - target_lonlat) * scale, axis=1)


def assign_nearest_boundary(
    df: pd.DataFrame,
    mesh: Mapping[str, np.ndarray],
    river_nodes: Mapping[str, Sequence[int]],
    region_col: str = "nearest_boundary",
) -> pd.DataFrame:
    """Assign each row to the nearest supplied boundary-node centroid."""
    if len(river_nodes) < 2:
        raise ValueError("river_nodes must contain at least two named node groups")

    out = df.copy()
    lon = out["Longitude"].to_numpy(dtype=float)
    lat = out["Latitude"].to_numpy(dtype=float)
    distance_cols = []

    for name, nodes in river_nodes.items():
        centroid = _centroid_for_nodes(mesh, nodes)
        col = f"distance_km_{name}"
        out[col] = _distance_km(lon, lat, centroid)
        out.attrs[f"centroid_{name}"] = centroid
        distance_cols.append(col)

    distances = out[distance_cols].to_numpy(dtype=float)
    nearest_idx = np.nanargmin(distances, axis=1)
    names = np.array(list(river_nodes.keys()), dtype=object)
    out[region_col] = names[nearest_idx]
    return out


def aggregate_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """Average repeated observations at each usSEABED location key."""
    if "LocnKey" not in df.columns:
        raise KeyError("aggregate_by_location requires a LocnKey column")

    df = add_silt_and_class_validity(df)
    numeric_cols = [
        col for col in [
            "Latitude", "Longitude", "WaterDepth", "Gravel", "Sand", "Mud",
            "Clay", "Silt", "Grainsze", "Sorting", "Gravel_norm", "Sand_norm",
            "Silt_norm", "Clay_norm",
        ] if col in df.columns
    ]
    first_cols = [
        col for col in [
            "source_table", "nearest_boundary", "Facies", "FolkCde", "LocnName",
            "DataSetKey",
        ] if col in df.columns
    ]

    agg = {col: "mean" for col in numeric_cols}
    agg.update({col: "first" for col in first_cols})
    agg["ObsvnKey"] = "count" if "ObsvnKey" in df.columns else "size"
    out = df.groupby("LocnKey", dropna=False).agg(agg).reset_index()
    out = out.rename(columns={"ObsvnKey": "observation_count"})
    out["valid_fraction_sum"] = out[["Gravel_norm", "Sand_norm", "Silt_norm", "Clay_norm"]].notna().all(axis=1)
    return out


def summarize_grain_fractions(
    df: pd.DataFrame,
    group_col: str | None = None,
    valid_sum_range: tuple[float, float] = (95.0, 105.0),
) -> pd.DataFrame:
    """Summarize grain-size fields for one table or grouped regions."""
    df = add_silt_and_class_validity(df, valid_sum_range=valid_sum_range)
    groups = [(None, df)] if group_col is None else list(df.groupby(group_col, dropna=False))
    rows = []

    for group_name, group in groups:
        valid = group[group["valid_fraction_sum"]]
        row = {
            "region": "whole_domain" if group_name is None else group_name,
            "observations": int(len(group)),
            "locations": int(group["LocnKey"].nunique(dropna=True)) if "LocnKey" in group.columns else np.nan,
            "valid_fraction_rows": int(len(valid)),
            "gravel_mean_pct": float(group["Gravel"].mean(skipna=True)) if "Gravel" in group else np.nan,
            "sand_mean_pct": float(group["Sand"].mean(skipna=True)) if "Sand" in group else np.nan,
            "mud_mean_pct": float(group["Mud"].mean(skipna=True)) if "Mud" in group else np.nan,
            "clay_mean_pct": float(group["Clay"].mean(skipna=True)) if "Clay" in group else np.nan,
            "silt_mean_pct": float(group["Silt"].mean(skipna=True)) if "Silt" in group else np.nan,
            "median_grainsize_phi": float(group["Grainsze"].median(skipna=True)) if "Grainsze" in group else np.nan,
            "mean_grainsize_phi": float(group["Grainsze"].mean(skipna=True)) if "Grainsze" in group else np.nan,
            "median_sorting_phi": float(group["Sorting"].median(skipna=True)) if "Sorting" in group else np.nan,
            "mean_sorting_phi": float(group["Sorting"].mean(skipna=True)) if "Sorting" in group else np.nan,
            "gravel_norm_pct": float(valid["Gravel_norm"].mean(skipna=True)) if len(valid) else np.nan,
            "sand_norm_pct": float(valid["Sand_norm"].mean(skipna=True)) if len(valid) else np.nan,
            "silt_norm_pct": float(valid["Silt_norm"].mean(skipna=True)) if len(valid) else np.nan,
            "clay_norm_pct": float(valid["Clay_norm"].mean(skipna=True)) if len(valid) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_samples_map(
    df: pd.DataFrame,
    mesh: Mapping[str, np.ndarray] | None = None,
    value_col: str | None = None,
    region_col: str | None = None,
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    s: float = 10.0,
    alpha: float = 0.85,
    draw_mesh_edges: bool = False,
):
    """Plot usSEABED sample points on lon/lat axes, optionally over an FVCOM mesh."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    if mesh is not None:
        tri = mtri.Triangulation(mesh["lon"], mesh["lat"], mesh["tri"])
        if "h" in mesh:
            ax.tripcolor(tri, mesh["h"], cmap="Greys", shading="flat", alpha=0.25)
        if draw_mesh_edges:
            ax.triplot(tri, color="0.75", linewidth=0.05, alpha=0.25)

    if value_col is not None:
        mappable = ax.scatter(
            df["Longitude"], df["Latitude"], c=df[value_col], cmap=cmap,
            s=s, alpha=alpha, edgecolor="none",
        )
    elif region_col is not None:
        codes, uniques = pd.factorize(df[region_col])
        mappable = ax.scatter(
            df["Longitude"], df["Latitude"], c=codes, cmap="tab10",
            s=s, alpha=alpha, edgecolor="none",
        )
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=str(name),
                       markerfacecolor=plt.cm.tab10(i % 10), markersize=7)
            for i, name in enumerate(uniques)
        ]
        ax.legend(handles=handles, loc="best", fontsize=8)
    else:
        mappable = ax.scatter(
            df["Longitude"], df["Latitude"], color="tab:blue",
            s=s, alpha=alpha, edgecolor="none",
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title)
    return ax, mappable


def plot_fraction_bars(summary: pd.DataFrame, ax=None, title: str | None = None):
    """Stacked bar plot for normalized gravel/sand/silt/clay fractions."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    cols = ["gravel_norm_pct", "sand_norm_pct", "silt_norm_pct", "clay_norm_pct"]
    labels = ["Gravel", "Sand", "Silt", "Clay"]
    colors = ["#8c6d31", "#d8b365", "#80cdc1", "#5e3c99"]
    x = np.arange(len(summary))
    bottom = np.zeros(len(summary))
    for col, label, color in zip(cols, labels, colors):
        vals = summary[col].fillna(0.0).to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(summary["region"], rotation=25, ha="right")
    ax.set_ylabel("Normalized fraction (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=4, fontsize=8)
    if title:
        ax.set_title(title)
    return ax
