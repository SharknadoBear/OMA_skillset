#!/usr/bin/env python3
"""Download and QC a Haines-area GloFAS v4.0 1979 discharge trial."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime
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
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


DATASET = "cems-glofas-historical"
REQUEST: dict[str, Any] = {
    "system_version": ["version_4_0"],
    "hydrological_model": ["lisflood"],
    "product_type": ["consolidated"],
    "variable": ["river_discharge_in_the_last_24_hours"],
    "hyear": ["1979"],
    "hmonth": [f"{month:02d}" for month in range(1, 13)],
    "hday": [f"{day:02d}" for day in range(1, 32)],
    "data_format": "netcdf",
    "download_format": "zip",
    "area": [59.85, -136.30, 58.10, -134.50],
}
DEFAULT_TARGET = Path("data/glofas_trials/cems_glofas_haines_1979_dis24_v4.zip")
DEFAULT_OUTPUT_DIR = Path("workspace/hydropower/glofas/outputs")
HUC_PATH = Path("workspace/hydropower/dhsvm_flownet/haines/source_wbd/wbd_huc12_intersecting_small_bbox.geojson")
FVCOM_PATH = Path(
    "workspace/hydropower/dhsvm_flownet/haines/source_fvcom_domain/fvcom_domain_matid1_epsg32608.geojson"
)
NWIS_BBOX = "-136.057512,58.183646,-134.657707,59.702470"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _credential_status() -> dict[str, Any]:
    path = Path.home() / ".cdsapirc"
    if not path.exists():
        return {"path": str(path), "exists": False, "has_url": False, "has_key": False}
    # Do not return or print secret values.
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return {
        "path": str(path),
        "exists": True,
        "has_url": any(line.strip().startswith("url:") for line in lines),
        "has_key": any(line.strip().startswith("key:") for line in lines),
    }


def download_glofas(target: Path, force: bool = False) -> Path:
    status = _credential_status()
    if not (status["exists"] and status["has_url"] and status["has_key"]):
        raise RuntimeError(
            "EWDS credentials are not ready. Create C:\\Users\\huan111\\.cdsapirc with url and key entries."
        )
    if target.exists() and target.stat().st_size > 0 and not force:
        print(f"Using existing download: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Requesting {DATASET} to {target}")
    import cdsapi

    client = cdsapi.Client()
    client.retrieve(DATASET, REQUEST, str(target))
    return target


def validate_zip(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    header = path.read_bytes()[:256]
    if header.lstrip().lower().startswith((b"<html", b"<!doctype html")):
        raise RuntimeError(f"{path} looks like HTML, not a ZIP/NetCDF payload.")
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


def extract_first_netcdf(zip_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str], dict[str, Any]]:
    zinfo = validate_zip(zip_path)
    member_name = zinfo["netcdf_members"][0]["name"]
    tmp = tempfile.TemporaryDirectory()
    out = Path(tmp.name) / Path(member_name).name
    with zipfile.ZipFile(zip_path) as zf:
        out.write_bytes(zf.read(member_name))
    return out, tmp, zinfo


def read_glofas(path: Path) -> dict[str, Any]:
    with Dataset(path) as ds:
        if "dis24" not in ds.variables:
            raise RuntimeError(f"{path} does not contain dis24.")
        lat = np.array(ds.variables["latitude"][:], dtype=float)
        lon = np.array(ds.variables["longitude"][:], dtype=float)
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        time_name = "valid_time" if "valid_time" in ds.variables else "time"
        tvar = ds.variables[time_name]
        times = pd.to_datetime([str(value) for value in num2date(tvar[:], tvar.units)])
        qvar = ds.variables["dis24"]
        q = np.array(qvar[:], dtype=float)
        return {
            "lat": lat,
            "lon": lon,
            "times": times,
            "q": q,
            "q_units": getattr(qvar, "units", ""),
            "q_long_name": getattr(qvar, "long_name", ""),
            "time_name": time_name,
        }


def _load_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    hucs = gpd.read_file(HUC_PATH).to_crs(4326)
    fvcom = gpd.read_file(FVCOM_PATH).to_crs(4326)
    return hucs, fvcom


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
    return sites[(sites.inside_selected_huc12) & (~sites.inside_fvcom)].copy().sort_values("site_no")


def nearest_glofas_at_gages(data: dict[str, Any], sites: pd.DataFrame) -> pd.DataFrame:
    lat = data["lat"]
    lon = data["lon"]
    q = data["q"]
    times = data["times"]
    rows: list[dict[str, Any]] = []
    for _, row in sites.iterrows():
        gage_lat = float(row.dec_lat_va)
        gage_lon = float(row.dec_long_va)
        ilat = int(np.argmin(np.abs(lat - gage_lat)))
        ilon = int(np.argmin(np.abs(lon - gage_lon)))
        dist_deg = math.hypot(float(lat[ilat]) - gage_lat, float(lon[ilon]) - gage_lon)
        for tidx, stamp in enumerate(times):
            rows.append(
                {
                    "date": stamp.date().isoformat(),
                    "site_no": row.site_no,
                    "station_nm": row.station_nm,
                    "gage_lat": gage_lat,
                    "gage_lon": gage_lon,
                    "grid_lat": float(lat[ilat]),
                    "grid_lon": float(lon[ilon]),
                    "nearest_deg": dist_deg,
                    "glofas_q_m3s": float(q[tidx, ilat, ilon]) if np.isfinite(q[tidx, ilat, ilon]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def fetch_nwis_observed(sites: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if sites.empty:
        return pd.DataFrame(columns=["site_no", "date", "obs_q_cfs", "obs_q_m3s"])
    site_list = ",".join(sites.site_no.astype(str).tolist())
    url = (
        "https://waterservices.usgs.gov/nwis/dv/"
        f"?format=rdb&sites={site_list}&parameterCd=00060&startDT={start}&endDT={end}&siteStatus=all"
    )
    text = urllib.request.urlopen(url, timeout=90).read().decode("utf-8")
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame(columns=["site_no", "date", "obs_q_cfs", "obs_q_m3s"])
    obs = pd.read_csv(StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", dtype=str)
    qcols = [col for col in obs.columns if col.endswith("_00060_00003")]
    if not qcols:
        return pd.DataFrame(columns=["site_no", "date", "obs_q_cfs", "obs_q_m3s"])
    obs = obs.rename(columns={qcols[0]: "obs_q_cfs", "datetime": "date"})
    obs["obs_q_cfs"] = pd.to_numeric(obs["obs_q_cfs"], errors="coerce")
    obs["obs_q_m3s"] = obs["obs_q_cfs"] * 0.028316846592
    return obs[["site_no", "date", "obs_q_cfs", "obs_q_m3s"]].copy()


def compute_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    def one_metric(group: pd.DataFrame, obs_col: str) -> dict[str, Any]:
        values = group[["glofas_q_m3s", obs_col]].dropna()
        values = values[np.isfinite(values.glofas_q_m3s) & np.isfinite(values[obs_col])]
        if len(values) < 5:
            return {"n": int(len(values)), "corr": np.nan, "bias_m3s": np.nan, "rmse_m3s": np.nan, "ratio_mean": np.nan}
        diff = values.glofas_q_m3s - values[obs_col]
        return {
            "n": int(len(values)),
            "corr": float(values.glofas_q_m3s.corr(values[obs_col])),
            "bias_m3s": float(diff.mean()),
            "rmse_m3s": float(np.sqrt((diff**2).mean())),
            "ratio_mean": float(values.glofas_q_m3s.mean() / values[obs_col].mean())
            if values[obs_col].mean() != 0
            else np.nan,
            "glo_mean_m3s": float(values.glofas_q_m3s.mean()),
            "obs_mean_m3s": float(values[obs_col].mean()),
        }

    rows: list[dict[str, Any]] = []
    for site_no, group in merged.groupby("site_no", sort=True):
        info = group.iloc[0]
        for comparison, obs_col in [
            ("same_date", "obs_q_m3s"),
            ("obs_date_plus_1_to_valid_time", "obs_shiftprev_q_m3s"),
        ]:
            metric = one_metric(group, obs_col)
            metric.update(
                {
                    "site_no": site_no,
                    "station_nm": info.station_nm,
                    "comparison": comparison,
                    "grid_lat": info.grid_lat,
                    "grid_lon": info.grid_lon,
                }
            )
            rows.append(metric)
    return pd.DataFrame(rows)


def selected_huc_cell_summary(data: dict[str, Any], hucs: gpd.GeoDataFrame) -> pd.DataFrame:
    lat = data["lat"]
    lon = data["lon"]
    q = data["q"]
    rows: list[dict[str, Any]] = []
    for ilat, lat_value in enumerate(lat):
        for ilon, lon_value in enumerate(lon):
            point = Point(float(lon_value), float(lat_value))
            if bool((hucs.geometry.contains(point) | hucs.geometry.touches(point)).any()):
                series = q[:, ilat, ilon]
                finite = series[np.isfinite(series)]
                if finite.size:
                    mean_q = float(finite.mean())
                    max_q = float(finite.max())
                    min_q = float(finite.min())
                else:
                    mean_q = np.nan
                    max_q = np.nan
                    min_q = np.nan
                rows.append(
                    {
                        "lat": float(lat_value),
                        "lon": float(lon_value),
                        "mean_q_m3s": mean_q,
                        "max_q_m3s": max_q,
                        "min_q_m3s": min_q,
                    }
                )
    return pd.DataFrame(rows)


def plot_annual_mean(
    data: dict[str, Any], hucs: gpd.GeoDataFrame, fvcom: gpd.GeoDataFrame, sites: pd.DataFrame, output: Path
) -> None:
    valid_count = np.isfinite(data["q"]).sum(axis=0)
    annual_mean = np.full(data["q"].shape[1:], np.nan, dtype=float)
    np.divide(np.nansum(data["q"], axis=0), valid_count, out=annual_mean, where=valid_count > 0)
    plot_q = np.where(np.isfinite(annual_mean) & (annual_mean > 0), annual_mean, np.nan)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
    mesh = ax.pcolormesh(data["lon"], data["lat"], plot_q, shading="nearest", cmap="viridis")
    hucs.boundary.plot(ax=ax, color="black", linewidth=0.8, label="Selected HUC12")
    fvcom.boundary.plot(ax=ax, color="red", linewidth=0.9, label="FVCOM boundary")
    if not sites.empty:
        ax.scatter(
            sites.dec_long_va.astype(float),
            sites.dec_lat_va.astype(float),
            s=28,
            c="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
            label="NWIS daily Q gages",
        )
        for _, row in sites.iterrows():
            ax.text(
                float(row.dec_long_va) + 0.015,
                float(row.dec_lat_va) + 0.01,
                str(row.site_no),
                fontsize=6.5,
                color="black",
                zorder=6,
            )
    ax.set_xlim(-136.32, -134.48)
    ax.set_ylim(58.08, 59.88)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("GloFAS v4.0 1979 daily mean discharge, annual mean")
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label(f"{data['q_long_name']} ({data['q_units']})")
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_hydrographs(merged: pd.DataFrame, output: Path) -> None:
    site_order = sorted(merged.site_no.unique().tolist())
    if not site_order:
        return
    fig, axes = plt.subplots(len(site_order), 1, figsize=(10, 1.65 * len(site_order)), dpi=180, sharex=True)
    if len(site_order) == 1:
        axes = [axes]
    for ax, site in zip(axes, site_order):
        group = merged[merged.site_no == site].copy()
        group["date_dt"] = pd.to_datetime(group["date"])
        ax.plot(group.date_dt, group.glofas_q_m3s, color="#386cb0", linewidth=1.1, label="GloFAS nearest")
        if group.obs_q_m3s.notna().any():
            ax.plot(group.date_dt, group.obs_q_m3s, color="#d95f02", linewidth=0.9, label="NWIS obs")
        ax.set_ylabel("m3/s", fontsize=8)
        ax.set_title(f"{site} {group.station_nm.iloc[0]}", fontsize=8, loc="left")
        ax.grid(True, alpha=0.25, linewidth=0.5)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("GloFAS nearest-pixel hydrographs and NWIS overlap", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def run_qc(zip_path: Path, output_dir: Path, prefix: str) -> dict[str, Any]:
    nc_path, tmp, zip_info = extract_first_netcdf(zip_path)
    try:
        data = read_glofas(nc_path)
        hucs, fvcom = _load_boundaries()
        sites = fetch_selected_nwis_gages(hucs, fvcom)
        start = data["times"].min().strftime("%Y-%m-%d")
        end = data["times"].max().strftime("%Y-%m-%d")
        obs = fetch_nwis_observed(sites, start, end)
        nearest = nearest_glofas_at_gages(data, sites)
        merged = nearest.merge(obs, on=["site_no", "date"], how="left")
        obs_shift = obs.copy()
        if not obs_shift.empty:
            obs_shift["date"] = (pd.to_datetime(obs_shift["date"]) + pd.Timedelta(days=1)).dt.date.astype(str)
            obs_shift = obs_shift.rename(columns={"obs_q_m3s": "obs_shiftprev_q_m3s"})
            merged = merged.merge(obs_shift[["site_no", "date", "obs_shiftprev_q_m3s"]], on=["site_no", "date"], how="left")
        else:
            merged["obs_shiftprev_q_m3s"] = np.nan
        metrics = compute_metrics(merged)
        cells = selected_huc_cell_summary(data, hucs)

        output_dir.mkdir(parents=True, exist_ok=True)
        nearest_csv = output_dir / f"{prefix}_nearest_nwis_gage_timeseries.csv"
        obs_csv = output_dir / f"{prefix}_nwis_observed_daily.csv"
        merged_csv = output_dir / f"{prefix}_glofas_vs_nwis_daily.csv"
        metrics_csv = output_dir / f"{prefix}_glofas_vs_nwis_metrics.csv"
        cells_csv = output_dir / f"{prefix}_selected_huc12_cells_summary.csv"
        map_png = output_dir / f"{prefix}_annual_mean_overlay.png"
        hydro_png = output_dir / f"{prefix}_nearest_gage_hydrographs.png"
        summary_json = output_dir / f"{prefix}_summary.json"

        nearest.to_csv(nearest_csv, index=False)
        obs.to_csv(obs_csv, index=False)
        merged.to_csv(merged_csv, index=False)
        metrics.to_csv(metrics_csv, index=False)
        cells.to_csv(cells_csv, index=False)
        plot_annual_mean(data, hucs, fvcom, sites, map_png)
        plot_hydrographs(merged, hydro_png)

        q = data["q"]
        summary = {
            "source_zip": str(zip_path),
            "zip_info": zip_info,
            "time_start": start,
            "time_end": end,
            "time_count": int(len(data["times"])),
            "time_daily_first_steps_days": [
                int((data["times"][idx + 1] - data["times"][idx]).days) for idx in range(min(5, len(data["times"]) - 1))
            ],
            "lat_min": float(np.nanmin(data["lat"])),
            "lat_max": float(np.nanmax(data["lat"])),
            "lon_min": float(np.nanmin(data["lon"])),
            "lon_max": float(np.nanmax(data["lon"])),
            "shape": list(q.shape),
            "variable": "dis24",
            "units": data["q_units"],
            "long_name": data["q_long_name"],
            "finite_values": int(np.isfinite(q).sum()),
            "nonzero_values": int((np.isfinite(q) & (q > 0)).sum()),
            "q_min_m3s": float(np.nanmin(q)),
            "q_max_m3s": float(np.nanmax(q)),
            "selected_huc12_cell_centers": int(len(cells)),
            "selected_huc12_mean_q_min_m3s": float(cells.mean_q_m3s.min()) if not cells.empty else None,
            "selected_huc12_mean_q_max_m3s": float(cells.mean_q_m3s.max()) if not cells.empty else None,
            "nwis_selected_outside_fvcom_gages": int(len(sites)),
            "outputs": {
                "nearest_gage_timeseries_csv": str(nearest_csv),
                "nwis_observed_daily_csv": str(obs_csv),
                "glofas_vs_nwis_daily_csv": str(merged_csv),
                "glofas_vs_nwis_metrics_csv": str(metrics_csv),
                "selected_huc12_cells_summary_csv": str(cells_csv),
                "annual_mean_overlay_png": str(map_png),
                "nearest_gage_hydrographs_png": str(hydro_png),
                "summary_json": str(summary_json),
            },
            "metrics": metrics.to_dict(orient="records"),
        }
        _write_json(summary_json, summary)
        return summary
    finally:
        tmp.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-zip", type=Path, default=DEFAULT_TARGET, help="Raw GloFAS ZIP target path.")
    parser.add_argument("--input-zip", type=Path, help="Use an existing ZIP for QC instead of the target.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="QC output directory.")
    parser.add_argument("--prefix", default="glofas_haines_1979_dis24_v4", help="Output file prefix.")
    parser.add_argument("--skip-download", action="store_true", help="Run QC only against an existing ZIP.")
    parser.add_argument("--force-download", action="store_true", help="Overwrite/re-request target ZIP.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output_dir / f"{args.prefix}_request.json",
        {"dataset": DATASET, "request": REQUEST, "target_zip": str(args.target_zip), "credential_status": _credential_status()},
    )

    zip_path = args.input_zip or args.target_zip
    if not args.skip_download and args.input_zip is None:
        zip_path = download_glofas(args.target_zip, force=args.force_download)
    summary = run_qc(zip_path, args.output_dir, args.prefix)
    print(json.dumps({k: summary[k] for k in ["source_zip", "time_start", "time_end", "time_count", "units"]}, indent=2))
    print("QC outputs:")
    for value in summary["outputs"].values():
        print(f"  {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
