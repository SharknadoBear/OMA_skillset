#!/usr/bin/env python3
"""Estimate an NHD flowline REST request before download."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_LAYER_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6"


def request_json(url: str, params: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def parse_bbox(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise ValueError("--bbox requires MINLON MAXLON MINLAT MAXLAT")
    min_lon, max_lon, min_lat, max_lat = values
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(f"invalid bbox order: {values}")
    return min_lon, max_lon, min_lat, max_lat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("MINLON", "MAXLON", "MINLAT", "MAXLAT"))
    parser.add_argument("--layer-url", default=DEFAULT_LAYER_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-path", type=Path, default=Path("."))
    parser.add_argument("--bytes-per-feature", type=int, default=6000)
    parser.add_argument("--multiplier", type=float, default=4.0)
    args = parser.parse_args()

    min_lon, max_lon, min_lat, max_lat = parse_bbox(args.bbox)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    layer = request_json(args.layer_url, {"f": "json"})
    count = request_json(
        f"{args.layer_url}/query",
        {
            "f": "json",
            "where": "1=1",
            "returnCountOnly": "true",
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
        },
    )["count"]
    estimated_bytes = int(count * args.bytes_per_feature)
    required_bytes = int(estimated_bytes * args.multiplier)
    storage = shutil.disk_usage(args.storage_path.resolve())
    result = {
        "source": "The National Map NHD REST Flowline - Large Scale",
        "layer_url": args.layer_url,
        "layer_name": layer.get("name"),
        "geometry_type": layer.get("geometryType"),
        "max_record_count": layer.get("maxRecordCount"),
        "bbox": [min_lon, max_lon, min_lat, max_lat],
        "feature_count": int(count),
        "bytes_per_feature_assumption": args.bytes_per_feature,
        "estimated_requested_bytes": estimated_bytes,
        "required_bytes": required_bytes,
        "available_bytes": storage.free,
        "multiplier": args.multiplier,
        "passed_local_storage_gate": storage.free > required_bytes,
        "created": started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
