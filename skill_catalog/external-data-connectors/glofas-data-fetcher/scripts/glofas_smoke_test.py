#!/usr/bin/env python3
"""Small EWDS/CDSAPI GloFAS smoke test.

Downloads a compact GloFAS v4.0 historical river-discharge subset, validates
the ZIP/NetCDF payload, and writes request/inventory/CSV evidence without
printing credentials.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date


DATASET = "cems-glofas-historical"
VARIABLE = "river_discharge_in_the_last_24_hours"


def json_default(value: Any) -> Any:
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
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def credential_status() -> dict[str, Any]:
    path = Path.home() / ".cdsapirc"
    if not path.exists():
        return {"path": str(path), "exists": False, "has_url": False, "has_key": False}
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return {
        "path": str(path),
        "exists": True,
        "has_url": any(line.strip().startswith("url:") and len(line.split(":", 1)[1].strip()) > 0 for line in lines),
        "has_key": any(line.strip().startswith("key:") and len(line.split(":", 1)[1].strip()) > 0 for line in lines),
    }


def request_from_dates(start_date: str, end_date: str, area: list[float]) -> dict[str, Any]:
    dates = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
    if dates.empty:
        raise ValueError("Date range is empty.")
    if len({date.year for date in dates}) > 1:
        raise ValueError("Smoke test supports one calendar year per request.")
    return {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": [VARIABLE],
        "hyear": sorted({str(date.year) for date in dates}),
        "hmonth": sorted({f"{date.month:02d}" for date in dates}),
        "hday": sorted({f"{date.day:02d}" for date in dates}),
        "data_format": "netcdf",
        "download_format": "zip",
        "area": area,
    }


def validate_zip(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    header = path.read_bytes()[:256]
    if header.lstrip().lower().startswith((b"<html", b"<!doctype html")):
        raise RuntimeError(f"{path} looks like HTML, not ZIP data.")
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"{path} is not a valid ZIP file.")
    with zipfile.ZipFile(path) as archive:
        members = [
            {"name": item.filename, "file_size": item.file_size, "compress_size": item.compress_size}
            for item in archive.infolist()
        ]
    netcdf_members = [item for item in members if item["name"].lower().endswith((".nc", ".nc4", ".cdf"))]
    if not netcdf_members:
        raise RuntimeError(f"{path} contains no NetCDF member.")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "member_count": len(members),
        "members": members,
        "netcdf_members": netcdf_members,
    }


def extract_first_netcdf(zip_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str], dict[str, Any]]:
    zip_info = validate_zip(zip_path)
    member = zip_info["netcdf_members"][0]["name"]
    tmp = tempfile.TemporaryDirectory()
    nc_path = Path(tmp.name) / Path(member).name
    with zipfile.ZipFile(zip_path) as archive:
        nc_path.write_bytes(archive.read(member))
    return nc_path, tmp, zip_info


def inspect_netcdf(zip_path: Path, start_date: str, end_date: str, summary_csv: Path) -> dict[str, Any]:
    nc_path, tmp, zip_info = extract_first_netcdf(zip_path)
    try:
        with Dataset(nc_path) as ds:
            required = {"dis24", "latitude", "longitude"}
            missing = sorted(required.difference(ds.variables))
            if missing:
                raise RuntimeError(f"NetCDF is missing required variables: {missing}")
            time_name = "valid_time" if "valid_time" in ds.variables else "time" if "time" in ds.variables else None
            if time_name is None:
                raise RuntimeError("NetCDF has neither valid_time nor time.")
            tvar = ds.variables[time_name]
            valid_time = pd.to_datetime([str(value) for value in num2date(tvar[:], tvar.units)])
            flow_date = valid_time - pd.Timedelta(days=1)
            qvar = ds.variables["dis24"]
            q = np.array(qvar[:], dtype=float)
            lat = np.array(ds.variables["latitude"][:], dtype=float)
            lon = np.array(ds.variables["longitude"][:], dtype=float)
            lon = np.where(lon > 180.0, lon - 360.0, lon)
            target_mask = (flow_date >= pd.Timestamp(start_date)) & (flow_date <= pd.Timestamp(end_date))
            target_q = q[target_mask, :, :]
            rows: list[dict[str, Any]] = []
            for idx, stamp in enumerate(flow_date[target_mask]):
                plane = target_q[idx, :, :]
                finite = plane[np.isfinite(plane)]
                rows.append(
                    {
                        "flow_date": stamp.date().isoformat(),
                        "finite_count": int(finite.size),
                        "min_dis24": float(np.min(finite)) if finite.size else np.nan,
                        "mean_dis24": float(np.mean(finite)) if finite.size else np.nan,
                        "max_dis24": float(np.max(finite)) if finite.size else np.nan,
                    }
                )
            summary_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(summary_csv, index=False)
            finite_total = int(np.isfinite(target_q).sum()) if target_q.size else 0
            return {
                "zip": zip_info,
                "netcdf_member": zip_info["netcdf_members"][0]["name"],
                "dimensions": {name: int(len(dim)) for name, dim in ds.dimensions.items()},
                "variables": sorted(ds.variables.keys()),
                "time_name": time_name,
                "time_units": getattr(tvar, "units", ""),
                "valid_time_start": valid_time.min().isoformat() if len(valid_time) else None,
                "valid_time_end": valid_time.max().isoformat() if len(valid_time) else None,
                "flow_date_convention": "flow_date = valid_time - 1 day",
                "flow_date_start": flow_date.min().date().isoformat() if len(flow_date) else None,
                "flow_date_end": flow_date.max().date().isoformat() if len(flow_date) else None,
                "target_flow_date_start": start_date,
                "target_flow_date_end": end_date,
                "target_day_count": int(target_mask.sum()),
                "latitude_range": [float(np.min(lat)), float(np.max(lat))],
                "longitude_range": [float(np.min(lon)), float(np.max(lon))],
                "dis24_units": getattr(qvar, "units", ""),
                "dis24_shape": [int(value) for value in q.shape],
                "target_finite_value_count": finite_total,
                "target_has_finite_values": bool(finite_total > 0),
                "summary_csv": str(summary_csv),
            }
    finally:
        tmp.cleanup()


def download(request: dict[str, Any], target: Path, force: bool) -> None:
    if target.exists() and target.stat().st_size > 0 and not force:
        return
    status = credential_status()
    if not (status["exists"] and status["has_url"] and status["has_key"]):
        raise RuntimeError("EWDS credentials are not ready; check ~/.cdsapirc without printing its contents.")
    target.parent.mkdir(parents=True, exist_ok=True)
    import cdsapi

    client = cdsapi.Client()
    client.retrieve(DATASET, request, str(target))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="1979-01-01")
    parser.add_argument("--end-date", default="1979-01-03")
    parser.add_argument("--area", nargs=4, type=float, default=[59.85, -136.30, 58.10, -134.50])
    parser.add_argument("--output-dir", type=Path, default=Path("workspace/_runtime/glofas_skill_smoke"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request = request_from_dates(args.start_date, args.end_date, args.area)
    request_json = args.output_dir / "glofas_smoke_request.json"
    zip_path = args.output_dir / f"glofas_smoke_{args.start_date}_{args.end_date}.zip"
    inventory_json = args.output_dir / "glofas_smoke_inventory.json"
    summary_csv = args.output_dir / "glofas_smoke_dis24_summary.csv"
    write_json(request_json, {"dataset": DATASET, "request": request, "credential_status": credential_status()})
    download(request, zip_path, force=args.force)
    inventory = inspect_netcdf(zip_path, args.start_date, args.end_date, summary_csv)
    inventory.update({"dataset": DATASET, "request_json": str(request_json), "download_zip": str(zip_path)})
    write_json(inventory_json, inventory)
    if not inventory["target_has_finite_values"]:
        raise RuntimeError("Smoke test found no finite discharge values in target dates.")
    print(
        json.dumps(
            {
                "download_zip": str(zip_path),
                "inventory_json": str(inventory_json),
                "summary_csv": str(summary_csv),
                "target_day_count": inventory["target_day_count"],
                "target_finite_value_count": inventory["target_finite_value_count"],
                "dis24_units": inventory["dis24_units"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
