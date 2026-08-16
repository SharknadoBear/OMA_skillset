#!/usr/bin/env python3
"""Estimate requested data volume and choose local or Kestrel routing.

The estimator prefers explicit sizes and otherwise estimates a bounded CFSv2
window from its time range, geographic extent, products, and native grid
spacing. Unknown requests remain blocked rather than receiving a silent guess.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SIZE_KEYS_BYTES = ("estimated_bytes", "requested_bytes", "download_bytes", "size_bytes", "bytes")
SIZE_KEYS_MB = ("estimated_mb", "requested_mb", "download_mb", "size_mb")
SIZE_KEYS_GB = ("estimated_gb", "requested_gb", "download_gb", "size_gb")
SCRATCH_TEMPLATE = "/scratch/<username>/oma_external_data_connectors/{skill_name}/{run_id}"

PRODUCT_VARIABLE_COUNTS = {
    "uv-10m": 2,
    "sfcprs": 1,
    "dlwsfc": 1,
    "dlwflx": 1,
    "dswsfc": 1,
    "strblk": 2,
    "TaqaQrQp": 4,
    "precip": 1,
    "surtmp": 1,
}


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _explicit_size_bytes(obj: Any) -> float | None:
    if isinstance(obj, dict):
        total = 0.0
        found = False
        for key in SIZE_KEYS_BYTES:
            value = _number(obj.get(key))
            if value is not None:
                total += value
                found = True
        for key in SIZE_KEYS_MB:
            value = _number(obj.get(key))
            if value is not None:
                total += value * 1024**2
                found = True
        for key in SIZE_KEYS_GB:
            value = _number(obj.get(key))
            if value is not None:
                total += value * 1024**3
                found = True
        for list_key in ("sources", "tiles", "files", "chunks", "requests"):
            if isinstance(obj.get(list_key), list):
                for item in obj[list_key]:
                    item_size = _explicit_size_bytes(item)
                    if item_size is not None:
                        total += item_size
                        found = True
        return total if found else None
    if isinstance(obj, list):
        total = 0.0
        found = False
        for item in obj:
            item_size = _explicit_size_bytes(item)
            if item_size is not None:
                total += item_size
                found = True
        return total if found else None
    return None


def _utc(text: Any) -> datetime | None:
    if not text:
        return None
    value = str(text).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _bounded_cfsv2_estimate(request: dict[str, Any]) -> tuple[float | None, dict[str, Any] | None]:
    start = _utc(request.get("start") or request.get("start_utc"))
    end = _utc(request.get("end") or request.get("end_utc"))
    bbox = request.get("bbox") or request.get("bbox_0_360")
    products = request.get("subdatasets") or request.get("products") or request.get("subdataset")
    if isinstance(products, str):
        products = [products]
    if start is None or end is None or end < start or not isinstance(bbox, list) or len(bbox) != 4:
        return None, None
    if not products or any(str(value) not in PRODUCT_VARIABLE_COUNTS for value in products):
        return None, None
    try:
        west, south, east, north = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None, None
    if east < west or north < south:
        return None, None
    spacing = float(request.get("grid_spacing_degrees", 0.205))
    cadence_hours = float(request.get("cadence_hours", 1.0))
    if spacing <= 0 or cadence_hours <= 0:
        return None, None
    nlon = max(2, math.ceil((east - west) / spacing) + 3)
    nlat = max(2, math.ceil((north - south) / spacing) + 3)
    ntime = math.floor((end - start).total_seconds() / (cadence_hours * 3600.0)) + 1
    nvars = sum(PRODUCT_VARIABLE_COUNTS[str(value)] for value in products)
    values = ntime * nlat * nlon * nvars
    estimated = values * 4.0 * 1.35 + (ntime + nlat + nlon) * 8.0
    return estimated, {
        "method": "bounded_cfsv2_native_grid",
        "ntime": ntime,
        "nlat": nlat,
        "nlon": nlon,
        "variable_count": nvars,
        "float_bytes": 4,
        "overhead_factor": 1.35,
    }


def _local_free_bytes(path: Path) -> int:
    target = path if path.exists() else path.parent
    target.mkdir(parents=True, exist_ok=True)
    return int(shutil.disk_usage(target).free)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="Request or manifest JSON with bbox/time/source/size fields.")
    parser.add_argument("--run-dir", default=".", help="Local run/cache directory used for free-space checks.")
    parser.add_argument("--output", required=True, help="Estimate JSON to write.")
    parser.add_argument("--skill-name", default=None, help="Override detected skill name.")
    parser.add_argument("--run-id", default=None, help="Run identifier for Kestrel scratch planning.")
    args = parser.parse_args()

    request = _read_json(args.request)
    run_dir = Path(args.run_dir)
    skill_name = args.skill_name or Path(__file__).resolve().parents[1].name
    run_id = args.run_id or request.get("name") or request.get("run_id") or "default"
    estimated_bytes = _explicit_size_bytes(request)
    estimate_details: dict[str, Any] | None = None
    if estimated_bytes is not None:
        estimate_details = {"method": "explicit_manifest_size"}
    else:
        estimated_bytes, estimate_details = _bounded_cfsv2_estimate(request)
    local_free = _local_free_bytes(run_dir)

    if estimated_bytes is None:
        recommendation = "review_required"
        reason = "No explicit size estimate was found in the request manifest."
    elif local_free > 4 * estimated_bytes:
        recommendation = "local"
        reason = "Local free space is more than four times the estimated request size."
    else:
        recommendation = "kestrel"
        reason = "Local free space is not more than four times the estimated request size."

    result = {
        "schema_version": "external_data_estimate_v1",
        "skill_name": skill_name,
        "request_path": str(args.request) if args.request else None,
        "run_dir": str(run_dir),
        "estimated_requested_bytes": int(estimated_bytes) if estimated_bytes is not None else None,
        "estimated_requested_mb": round(estimated_bytes / 1024**2, 3) if estimated_bytes is not None else None,
        "estimate_details": estimate_details,
        "local_free_bytes": local_free,
        "local_free_gb": round(local_free / 1024**3, 3),
        "routing_recommendation": recommendation,
        "routing_reason": reason,
        "kestrel_scratch_path": SCRATCH_TEMPLATE.format(skill_name=skill_name, run_id=run_id),
        "download_gate": {
            "rule": "download locally only when local_free_bytes > 4 * estimated_requested_bytes",
            "unknown_estimate_policy": "review_required",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
