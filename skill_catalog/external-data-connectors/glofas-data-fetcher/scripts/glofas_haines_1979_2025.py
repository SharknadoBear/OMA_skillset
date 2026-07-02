#!/usr/bin/env python3
"""Download GloFAS annual slices and validate Haines gage nearest pixels."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from shapely.geometry import Point

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET = "cems-glofas-historical"
VARIABLE = "river_discharge_in_the_last_24_hours"
AREA = [59.85, -136.30, 58.10, -134.50]
START_YEAR = 1979
END_YEAR = 2025
START_DATE = pd.Timestamp("1979-01-01")
END_DATE = pd.Timestamp("2025-12-31")
HUC_PATH = Path("workspace/hydropower/dhsvm_flownet/haines/source_wbd/wbd_huc12_intersecting_small_bbox.geojson")
FVCOM_PATH = Path(
    "workspace/hydropower/dhsvm_flownet/haines/source_fvcom_domain/fvcom_domain_matid1_epsg32608.geojson"
)
STREAM_VECTOR_PATH = Path("workspace/hydropower/dhsvm_flownet/haines/global_grass/stream_vector.gpkg")
LARGE_ROI_PATH = Path("workspace/hydropower/dhsvm_flownet/haines/source_wbd/haines_large_bbox.geojson")
SMALL_ROI_PATH = Path("workspace/hydropower/dhsvm_flownet/haines/source_wbd/haines_small_bbox.geojson")
NWIS_BBOX = "-136.057512,58.183646,-134.657707,59.702470"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def credential_status() -> dict[str, Any]:
    path = Path.home() / ".cdsapirc"
    if not path.exists():
        return {"exists": False, "has_url": False, "has_key": False, "path": str(path)}
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return {
        "exists": True,
        "has_url": any(line.strip().startswith("url:") for line in lines),
        "has_key": any(line.strip().startswith("key:") for line in lines),
        "path": str(path),
    }


def request_for_year(year: int) -> dict[str, Any]:
    return request_for_years([year])


def request_for_years(years: list[int]) -> dict[str, Any]:
    return {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": [VARIABLE],
        "hyear": [str(year) for year in years],
        "hmonth": [f"{month:02d}" for month in range(1, 13)],
        "hday": [f"{day:02d}" for day in range(1, 32)],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": AREA,
    }


def validate_zip(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    header = path.read_bytes()[:256]
    if header.lstrip().lower().startswith((b"<html", b"<!doctype html")):
        raise RuntimeError(f"{path} looks like HTML, not a ZIP payload.")
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"{path} is not a valid ZIP file.")
    with zipfile.ZipFile(path) as zf:
        members = [
            {"name": item.filename, "file_size": item.file_size, "compress_size": item.compress_size}
            for item in zf.infolist()
        ]
    nc_members = [item for item in members if item["name"].lower().endswith((".nc", ".nc4", ".cdf"))]
    if not nc_members:
        raise RuntimeError(f"{path} has no NetCDF member.")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "member_count": len(members),
        "members": members,
        "netcdf_members": nc_members,
    }


def preflight_storage(raw_dir: Path, years: list[int]) -> dict[str, Any]:
    existing_sizes = [p.stat().st_size for p in raw_dir.glob("cems_glofas_haines_*_dis24_v4.zip") if p.stat().st_size]
    legacy = Path("data/glofas_trials/cems_glofas_haines_1979_dis24_v4.zip")
    if legacy.exists():
        existing_sizes.append(legacy.stat().st_size)
    estimate_one = max(existing_sizes) if existing_sizes else 600_000
    estimated_total = estimate_one * len(years)
    target = raw_dir if raw_dir.exists() else raw_dir.parent
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    return {
        "raw_dir": str(raw_dir),
        "year_count": len(years),
        "estimate_one_year_bytes": int(estimate_one),
        "estimated_total_bytes": int(estimated_total),
        "required_free_bytes_4x": int(4 * estimated_total),
        "free_bytes": int(free),
        "passed": bool(free > 4 * estimated_total),
    }


def download_year(year: int, target: Path, *, force: bool = False) -> dict[str, Any]:
    return download_years([year], target, force=force)


def download_years(years: list[int], target: Path, *, force: bool = False) -> dict[str, Any]:
    if target.exists() and target.stat().st_size > 0 and not force:
        info = validate_zip(target)
        info.update({"years": years, "status": "existing"})
        return info
    status = credential_status()
    if not (status["exists"] and status["has_url"] and status["has_key"]):
        raise RuntimeError("EWDS credentials are not ready; check ~/.cdsapirc.")
    target.parent.mkdir(parents=True, exist_ok=True)
    import cdsapi

    label = str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"
    print(f"Requesting GloFAS {label} -> {target}")
    client = cdsapi.Client()
    client.retrieve(DATASET, request_for_years(years), str(target))
    info = validate_zip(target)
    info.update({"years": years, "status": "downloaded"})
    return info


def parse_covered_years(path: Path) -> list[int]:
    match = re.match(r"cems_glofas_haines_(\d{4})(?:_(\d{4}))?_dis24_v4\.zip$", path.name)
    if not match:
        return []
    start = int(match.group(1))
    end = int(match.group(2) or match.group(1))
    if end < start:
        return []
    return list(range(start, end + 1))


def chunk_missing_years(years: list[int], existing_coverage: set[int], chunk_size: int) -> list[list[int]]:
    missing = [year for year in years if year not in existing_coverage]
    chunks: list[list[int]] = []
    current: list[int] = []
    for year in missing:
        if current and (year != current[-1] + 1 or len(current) >= chunk_size):
            chunks.append(current)
            current = []
        current.append(year)
    if current:
        chunks.append(current)
    return chunks


def discover_existing_sources(raw_dir: Path, years: list[int]) -> tuple[list[Path], set[int], list[dict[str, Any]]]:
    wanted = set(years)
    sources: list[Path] = []
    coverage: set[int] = set()
    records: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("cems_glofas_haines_*_dis24_v4.zip")):
        covered = [year for year in parse_covered_years(path) if year in wanted]
        if not covered:
            continue
        try:
            info = validate_zip(path)
            sources.append(path)
            coverage.update(covered)
            records.append({"path": str(path), "years": covered, "status": "existing_valid", "bytes": info["bytes"]})
        except Exception as exc:
            records.append({"path": str(path), "years": covered, "status": "existing_invalid_ignored", "error": str(exc)})
    return sources, coverage, records


def extract_netcdf(zip_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str], dict[str, Any]]:
    info = validate_zip(zip_path)
    member = info["netcdf_members"][0]["name"]
    tmp = tempfile.TemporaryDirectory()
    out = Path(tmp.name) / Path(member).name
    with zipfile.ZipFile(zip_path) as zf:
        out.write_bytes(zf.read(member))
    return out, tmp, info


def read_glofas_zip(zip_path: Path) -> dict[str, Any]:
    nc_path, tmp, zip_info = extract_netcdf(zip_path)
    try:
        with Dataset(nc_path) as ds:
            if "dis24" not in ds.variables:
                raise RuntimeError(f"{zip_path} does not contain dis24.")
            lat = np.array(ds.variables["latitude"][:], dtype=float)
            lon = np.array(ds.variables["longitude"][:], dtype=float)
            lon = np.where(lon > 180.0, lon - 360.0, lon)
            time_name = "valid_time" if "valid_time" in ds.variables else "time"
            tvar = ds.variables[time_name]
            valid_time = pd.to_datetime([str(value) for value in num2date(tvar[:], tvar.units)])
            flow_date = valid_time - pd.Timedelta(days=1)
            qvar = ds.variables["dis24"]
            q = np.array(qvar[:], dtype=float)
            return {
                "zip_info": zip_info,
                "lat": lat,
                "lon": lon,
                "valid_time": valid_time,
                "flow_date": flow_date,
                "q": q,
                "q_units": getattr(qvar, "units", ""),
                "q_long_name": getattr(qvar, "long_name", ""),
            }
    finally:
        tmp.cleanup()


def load_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    return gpd.read_file(HUC_PATH).to_crs(4326), gpd.read_file(FVCOM_PATH).to_crs(4326)


def fetch_selected_nwis_gages(hucs: gpd.GeoDataFrame, fvcom: gpd.GeoDataFrame) -> pd.DataFrame:
    url = (
        "https://waterservices.usgs.gov/nwis/site/"
        f"?format=rdb&bBox={NWIS_BBOX}&parameterCd=00060&siteStatus=all&hasDataTypeCd=dv"
    )
    text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    sites = pd.read_csv(StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", dtype=str)
    inside_huc: list[bool] = []
    inside_fvcom: list[bool] = []
    for _, row in sites.iterrows():
        point = Point(float(row.dec_long_va), float(row.dec_lat_va))
        inside_huc.append(bool((hucs.geometry.contains(point) | hucs.geometry.touches(point)).any()))
        inside_fvcom.append(bool((fvcom.geometry.contains(point) | fvcom.geometry.touches(point)).any()))
    sites["inside_selected_huc12"] = inside_huc
    sites["inside_fvcom"] = inside_fvcom
    selected = sites[(sites.inside_selected_huc12) & (~sites.inside_fvcom)].copy().sort_values("site_no")
    return selected.reset_index(drop=True)


def fetch_nwis_observed(sites: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if sites.empty:
        return pd.DataFrame(columns=["site_no", "date", "obs_q_cfs", "obs_q_m3s"])
    site_list = ",".join(sites.site_no.astype(str).tolist())
    url = (
        "https://waterservices.usgs.gov/nwis/dv/"
        f"?format=rdb&sites={site_list}&parameterCd=00060&startDT={start.date()}&endDT={end.date()}&siteStatus=all"
    )
    text = urllib.request.urlopen(url, timeout=120).read().decode("utf-8")
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame(columns=["site_no", "date", "obs_q_cfs", "obs_q_m3s"])
    obs = pd.read_csv(StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", dtype=str)
    qcols = [column for column in obs.columns if column.endswith("_00060_00003")]
    if not qcols:
        return pd.DataFrame(columns=["site_no", "date", "obs_q_cfs", "obs_q_m3s"])
    obs = obs.rename(columns={qcols[0]: "obs_q_cfs", "datetime": "date"})
    obs["obs_q_cfs"] = pd.to_numeric(obs["obs_q_cfs"], errors="coerce")
    obs["obs_q_m3s"] = obs["obs_q_cfs"] * 0.028316846592
    return obs[["site_no", "date", "obs_q_cfs", "obs_q_m3s"]].copy()


def nearest_indices(lat: np.ndarray, lon: np.ndarray, sites: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for _, site in sites.iterrows():
        site_id = str(site.site_no)
        gage_lat = float(site.dec_lat_va)
        gage_lon = float(site.dec_long_va)
        ilat = int(np.argmin(np.abs(lat - gage_lat)))
        ilon = int(np.argmin(np.abs(lon - gage_lon)))
        out[site_id] = {
            "site_no": site_id,
            "station_nm": site.station_nm,
            "gage_lat": gage_lat,
            "gage_lon": gage_lon,
            "grid_lat": float(lat[ilat]),
            "grid_lon": float(lon[ilon]),
            "ilat": ilat,
            "ilon": ilon,
            "nearest_deg": float(math.hypot(float(lat[ilat]) - gage_lat, float(lon[ilon]) - gage_lon)),
        }
    return out


def metric_values(group: pd.DataFrame) -> dict[str, Any]:
    values = group[["glofas_q_m3s", "obs_q_m3s"]].dropna()
    values = values[np.isfinite(values.glofas_q_m3s) & np.isfinite(values.obs_q_m3s)]
    if values.empty:
        return {
            "n_pairs": 0,
            "date_start": None,
            "date_end": None,
            "r": np.nan,
            "kge": np.nan,
            "rmse_m3s": np.nan,
            "mae_m3s": np.nan,
            "bias_m3s": np.nan,
            "pbias_percent": np.nan,
            "nse": np.nan,
            "alpha": np.nan,
            "beta": np.nan,
            "glofas_mean_m3s": np.nan,
            "obs_mean_m3s": np.nan,
        }
    sim = values.glofas_q_m3s.to_numpy(float)
    obs = values.obs_q_m3s.to_numpy(float)
    diff = sim - obs
    r = float(np.corrcoef(sim, obs)[0, 1]) if len(values) > 1 and np.std(sim) > 0 and np.std(obs) > 0 else np.nan
    alpha = float(np.std(sim, ddof=0) / np.std(obs, ddof=0)) if np.std(obs, ddof=0) > 0 else np.nan
    beta = float(np.mean(sim) / np.mean(obs)) if np.mean(obs) != 0 else np.nan
    if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
        kge = float(1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    else:
        kge = np.nan
    denominator = float(np.sum((obs - np.mean(obs)) ** 2))
    nse = float(1.0 - np.sum(diff**2) / denominator) if denominator > 0 else np.nan
    return {
        "n_pairs": int(len(values)),
        "date_start": str(values.index.min().date()),
        "date_end": str(values.index.max().date()),
        "r": r,
        "kge": kge,
        "rmse_m3s": float(np.sqrt(np.mean(diff**2))),
        "mae_m3s": float(np.mean(np.abs(diff))),
        "bias_m3s": float(np.mean(diff)),
        "pbias_percent": float(100.0 * np.sum(diff) / np.sum(obs)) if np.sum(obs) != 0 else np.nan,
        "nse": nse,
        "alpha": alpha,
        "beta": beta,
        "glofas_mean_m3s": float(np.mean(sim)),
        "obs_mean_m3s": float(np.mean(obs)),
    }


def compute_metrics(merged: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    merged["date_dt"] = pd.to_datetime(merged["date"])
    rows: list[dict[str, Any]] = []
    for _, site in sites.iterrows():
        site_id = str(site.site_no)
        group = merged[merged.site_no == site_id].set_index("date_dt")
        metrics = metric_values(group)
        metrics.update({"site_no": site_id, "station_nm": site.station_nm})
        rows.append(metrics)
    return pd.DataFrame(rows)


def combine_sources(source_paths: list[Path], sites: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None
    q_sum: np.ndarray | None = None
    q_count: np.ndarray | None = None
    nearest: dict[str, dict[str, Any]] | None = None
    gage_rows: list[dict[str, Any]] = []
    source_status: list[dict[str, Any]] = []
    all_dates: list[pd.Timestamp] = []
    seen_dates: set[str] = set()
    units = ""
    long_name = ""
    for zip_path in sorted(source_paths, key=lambda path: (parse_covered_years(path) or [9999])[0]):
        data = read_glofas_zip(zip_path)
        if lat is None:
            lat = data["lat"]
            lon = data["lon"]
            q_sum = np.zeros(data["q"].shape[1:], dtype=float)
            q_count = np.zeros(data["q"].shape[1:], dtype=np.int64)
            nearest = nearest_indices(lat, lon, sites)
            units = data["q_units"]
            long_name = data["q_long_name"]
        elif not (np.array_equal(lat, data["lat"]) and np.array_equal(lon, data["lon"])):
            raise RuntimeError(f"Grid changed in {zip_path}")
        dates = data["flow_date"]
        mask = (dates >= START_DATE) & (dates <= END_DATE)
        dates = dates[mask]
        q = data["q"][mask, :, :]
        duplicate_dates = [date.date().isoformat() for date in dates if date.date().isoformat() in seen_dates]
        if duplicate_dates:
            raise RuntimeError(f"Duplicate flow dates found in {zip_path}: {duplicate_dates[:5]}")
        seen_dates.update(date.date().isoformat() for date in dates)
        finite = np.isfinite(q)
        q_sum += np.nansum(q, axis=0)
        q_count += finite.sum(axis=0)
        all_dates.extend(dates.to_list())
        assert nearest is not None
        for site_id, info in nearest.items():
            series = q[:, info["ilat"], info["ilon"]]
            for date_value, q_value in zip(dates, series):
                gage_rows.append(
                    {
                        "date": date_value.date().isoformat(),
                        "site_no": site_id,
                        "station_nm": info["station_nm"],
                        "gage_lat": info["gage_lat"],
                        "gage_lon": info["gage_lon"],
                        "grid_lat": info["grid_lat"],
                        "grid_lon": info["grid_lon"],
                        "nearest_deg": info["nearest_deg"],
                        "glofas_q_m3s": float(q_value) if np.isfinite(q_value) else np.nan,
                    }
                )
        source_status.append(
            {
                "zip": str(zip_path),
                "covered_years_from_name": parse_covered_years(zip_path),
                "flow_date_start": str(dates.min().date()) if len(dates) else None,
                "flow_date_end": str(dates.max().date()) if len(dates) else None,
                "day_count": int(len(dates)),
                "q_min_m3s": float(np.nanmin(q)),
                "q_max_m3s": float(np.nanmax(q)),
            }
        )
    assert lat is not None and lon is not None and q_sum is not None and q_count is not None
    mean_q = np.full(q_sum.shape, np.nan, dtype=float)
    np.divide(q_sum, q_count, out=mean_q, where=q_count > 0)
    pixel_rows: list[dict[str, Any]] = []
    for ilat, lat_value in enumerate(lat):
        for ilon, lon_value in enumerate(lon):
            pixel_rows.append(
                {
                    "lat": float(lat_value),
                    "lon": float(lon_value),
                    "mean_q_m3s": float(mean_q[ilat, ilon]) if np.isfinite(mean_q[ilat, ilon]) else np.nan,
                    "valid_day_count": int(q_count[ilat, ilon]),
                }
            )
    metadata = {
        "lat": lat,
        "lon": lon,
        "mean_q": mean_q,
        "q_count": q_count,
        "date_start": str(min(all_dates).date()),
        "date_end": str(max(all_dates).date()),
        "date_count": int(len(set(date.date().isoformat() for date in all_dates))),
        "units": units,
        "long_name": long_name,
        "source_status": source_status,
    }
    return pd.DataFrame(gage_rows), pd.DataFrame(pixel_rows), metadata


def plot_scatter_one(site_id: str, merged: pd.DataFrame, metrics: pd.Series, output: Path) -> None:
    data = merged[(merged.site_no == site_id) & merged.obs_q_m3s.notna() & merged.glofas_q_m3s.notna()].copy()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    if not data.empty:
        max_value = float(np.nanmax([data.obs_q_m3s.max(), data.glofas_q_m3s.max()]))
        ax.scatter(data.obs_q_m3s, data.glofas_q_m3s, s=8, alpha=0.28, color="#386cb0", edgecolors="none")
        ax.plot([0, max_value * 1.05], [0, max_value * 1.05], color="black", linewidth=1.0, linestyle="--")
        ax.set_xlim(0, max_value * 1.05)
        ax.set_ylim(0, max_value * 1.05)
        text = (
            f"n={int(metrics.n_pairs)}\n"
            f"KGE={metrics.kge:.2f}\nR={metrics.r:.2f}\n"
            f"RMSE={metrics.rmse_m3s:.1f} m3/s\nBias={metrics.bias_m3s:.1f} m3/s"
        )
    else:
        text = "No overlapping finite daily pairs"
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.85},
    )
    ax.set_xlabel("NWIS observed daily Q (m3/s)")
    ax.set_ylabel("GloFAS nearest-pixel daily Q (m3/s)")
    ax.set_title(f"{site_id} {metrics.station_nm}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_scatter_combined(merged: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    site_ids = metrics.site_no.astype(str).tolist()
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), dpi=180)
    axes_flat = axes.ravel()
    for ax, site_id in zip(axes_flat, site_ids):
        row = metrics[metrics.site_no == site_id].iloc[0]
        data = merged[(merged.site_no == site_id) & merged.obs_q_m3s.notna() & merged.glofas_q_m3s.notna()]
        if not data.empty:
            max_value = float(np.nanmax([data.obs_q_m3s.max(), data.glofas_q_m3s.max()]))
            ax.scatter(data.obs_q_m3s, data.glofas_q_m3s, s=4, alpha=0.22, color="#386cb0", edgecolors="none")
            ax.plot([0, max_value * 1.05], [0, max_value * 1.05], color="black", linewidth=0.8, linestyle="--")
            ax.set_xlim(0, max_value * 1.05)
            ax.set_ylim(0, max_value * 1.05)
            label = f"KGE={row.kge:.2f}, R={row.r:.2f}\nn={int(row.n_pairs)}"
        else:
            label = "no overlap"
        ax.text(0.04, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=7)
        ax.set_title(f"{site_id}\n{row.station_nm}", fontsize=8)
        ax.grid(True, alpha=0.22)
    for ax in axes_flat[len(site_ids) :]:
        ax.axis("off")
    fig.supxlabel("NWIS observed daily Q (m3/s)")
    fig.supylabel("GloFAS nearest-pixel daily Q (m3/s)")
    fig.suptitle("GloFAS v4.0 vs NWIS daily discharge, overlapping records", y=0.995)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_mean_map(pixel_df: pd.DataFrame, metadata: dict[str, Any], sites: pd.DataFrame, output: Path) -> None:
    hucs = gpd.read_file(HUC_PATH).to_crs(4326)
    fvcom = gpd.read_file(FVCOM_PATH).to_crs(4326)
    streams = gpd.read_file(STREAM_VECTOR_PATH).to_crs(4326)
    large_roi = gpd.read_file(LARGE_ROI_PATH).to_crs(4326)
    small_roi = gpd.read_file(SMALL_ROI_PATH).to_crs(4326)
    lon = metadata["lon"]
    lat = metadata["lat"]
    mean_q = metadata["mean_q"]
    plot_q = np.where(np.isfinite(mean_q) & (mean_q > 0), mean_q, np.nan)
    fig, ax = plt.subplots(figsize=(10.5, 9.5), dpi=190)
    mesh = ax.pcolormesh(lon, lat, plot_q, shading="nearest", cmap="viridis")
    streams.plot(ax=ax, color="#2b2b2b", linewidth=0.22, alpha=0.65, label="Extracted flow network", zorder=4)
    hucs.boundary.plot(ax=ax, color="white", linewidth=1.25, zorder=5)
    hucs.boundary.plot(ax=ax, color="black", linewidth=0.55, zorder=6, label="Selected HUC12")
    fvcom.boundary.plot(ax=ax, color="#e31a1c", linewidth=1.15, zorder=7, label="FVCOM boundary")
    large_roi.boundary.plot(ax=ax, color="#ff7f00", linewidth=1.45, linestyle="--", zorder=8, label="Large ROI")
    small_roi.boundary.plot(ax=ax, color="#00b4d8", linewidth=1.45, linestyle="--", zorder=9, label="Small ROI")
    ax.scatter(
        sites.dec_long_va.astype(float),
        sites.dec_lat_va.astype(float),
        s=34,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=10,
        label="NWIS gages",
    )
    for _, site in sites.iterrows():
        ax.text(float(site.dec_long_va) + 0.012, float(site.dec_lat_va) + 0.008, str(site.site_no), fontsize=6.5, zorder=11)
    ax.set_xlim(-136.32, -134.48)
    ax.set_ylim(58.08, 59.88)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("GloFAS v4.0 mean daily discharge (m3/s), 1979-2025")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean daily discharge (m3/s)")
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.88)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    years = list(range(args.start_year, args.end_year + 1))
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    storage = preflight_storage(args.raw_dir, years)
    if not storage["passed"] and not args.ignore_storage_gate:
        raise RuntimeError(f"Storage preflight failed: {storage}")

    source_paths, existing_coverage, existing_records = discover_existing_sources(args.raw_dir, years)
    download_status: list[dict[str, Any]] = existing_records.copy()
    chunks = chunk_missing_years(years, existing_coverage, args.chunk_size)
    for chunk in chunks:
        label = str(chunk[0]) if len(chunk) == 1 else f"{chunk[0]}_{chunk[-1]}"
        target = args.raw_dir / f"cems_glofas_haines_{label}_dis24_v4.zip"
        info = download_years(chunk, target, force=args.force_download)
        download_status.append(info)
        if target not in source_paths:
            source_paths.append(target)

    hucs, fvcom = load_boundaries()
    sites = fetch_selected_nwis_gages(hucs, fvcom)
    gage_ts, pixel_df, metadata = combine_sources(source_paths, sites)
    obs = fetch_nwis_observed(sites, START_DATE, END_DATE)
    merged = gage_ts.merge(obs, on=["site_no", "date"], how="left")
    metrics = compute_metrics(merged, sites)

    prefix = args.output_dir / "glofas_haines_1979_2025"
    sites_csv = prefix.with_name(prefix.name + "_selected_nwis_gages.csv")
    gage_csv = prefix.with_name(prefix.name + "_nearest_gage_timeseries.csv")
    obs_csv = prefix.with_name(prefix.name + "_nwis_observed_daily.csv")
    merged_csv = prefix.with_name(prefix.name + "_glofas_vs_nwis_daily.csv")
    metrics_csv = prefix.with_name(prefix.name + "_glofas_vs_nwis_metrics.csv")
    pixels_csv = prefix.with_name(prefix.name + "_pixel_mean_discharge.csv")
    map_png = prefix.with_name(prefix.name + "_mean_discharge_summary_map.png")
    scatter_png = prefix.with_name(prefix.name + "_gauge_scatter_all.png")
    summary_json = prefix.with_name(prefix.name + "_summary.json")

    sites.to_csv(sites_csv, index=False)
    gage_ts.to_csv(gage_csv, index=False)
    obs.to_csv(obs_csv, index=False)
    merged.to_csv(merged_csv, index=False)
    metrics.to_csv(metrics_csv, index=False)
    pixel_df.to_csv(pixels_csv, index=False)
    plot_scatter_combined(merged, metrics, scatter_png)
    scatter_dir = args.output_dir / "gauge_scatter"
    individual_scatter: dict[str, str] = {}
    for _, row in metrics.iterrows():
        site_id = str(row.site_no)
        out = scatter_dir / f"glofas_vs_nwis_{site_id}.png"
        plot_scatter_one(site_id, merged, row, out)
        individual_scatter[site_id] = str(out)
    plot_mean_map(pixel_df, metadata, sites, map_png)

    date_index = pd.date_range(START_DATE, END_DATE, freq="D")
    actual_dates = set(pd.to_datetime(gage_ts.date).dt.date.astype(str))
    missing_dates = [date.date().isoformat() for date in date_index if date.date().isoformat() not in actual_dates]
    summary = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "dataset": DATASET,
        "variable": VARIABLE,
        "area": AREA,
        "years": years,
        "storage_preflight": storage,
        "download_status": download_status,
        "source_status": metadata["source_status"],
        "date_start": metadata["date_start"],
        "date_end": metadata["date_end"],
        "expected_day_count": int(len(date_index)),
        "actual_day_count": metadata["date_count"],
        "missing_dates": missing_dates,
        "units": metadata["units"],
        "grid_shape": [int(len(metadata["lat"])), int(len(metadata["lon"]))],
        "selected_nwis_gage_count": int(len(sites)),
        "outputs": {
            "selected_nwis_gages_csv": str(sites_csv),
            "nearest_gage_timeseries_csv": str(gage_csv),
            "nwis_observed_daily_csv": str(obs_csv),
            "glofas_vs_nwis_daily_csv": str(merged_csv),
            "metrics_csv": str(metrics_csv),
            "pixel_mean_discharge_csv": str(pixels_csv),
            "mean_discharge_summary_map_png": str(map_png),
            "gauge_scatter_all_png": str(scatter_png),
            "individual_scatter_png": individual_scatter,
            "summary_json": str(summary_json),
        },
        "metrics": metrics.to_dict(orient="records"),
    }
    write_json(summary_json, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/glofas_trials/annual"))
    parser.add_argument("--output-dir", type=Path, default=Path("workspace/hydropower/glofas/outputs/glofas_haines_1979_2025"))
    parser.add_argument("--chunk-size", type=int, default=1, help="Number of missing years per CEMS request.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--ignore-storage-gate", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = run(parse_args())
    print(
        json.dumps(
            {
                "date_start": summary["date_start"],
                "date_end": summary["date_end"],
                "actual_day_count": summary["actual_day_count"],
                "selected_nwis_gage_count": summary["selected_nwis_gage_count"],
                "outputs": summary["outputs"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
