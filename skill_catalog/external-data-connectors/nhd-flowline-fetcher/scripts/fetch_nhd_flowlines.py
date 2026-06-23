#!/usr/bin/env python3
"""Fetch NHD flowlines from The National Map ArcGIS REST service."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape


DEFAULT_LAYER_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6"
DEFAULT_FIELDS = [
    "OBJECTID",
    "permanent_identifier",
    "gnis_name",
    "lengthkm",
    "reachcode",
    "ftype",
    "fcode",
    "innetwork",
    "mainpath",
    "visibilityfilter",
]


def parse_bbox(values: list[float]) -> tuple[float, float, float, float]:
    min_lon, max_lon, min_lat, max_lat = values
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(f"invalid bbox order: {values}")
    return min_lon, max_lon, min_lat, max_lat


def request_json(url: str, params: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def query_count(layer_url: str, bbox: tuple[float, float, float, float]) -> int:
    min_lon, max_lon, min_lat, max_lat = bbox
    data = request_json(
        f"{layer_url}/query",
        {
            "f": "json",
            "where": "1=1",
            "returnCountOnly": "true",
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
        },
    )
    return int(data["count"])


def fetch_page(
    layer_url: str,
    bbox: tuple[float, float, float, float],
    out_fields: str,
    offset: int,
    page_size: int,
) -> list[dict[str, Any]]:
    min_lon, max_lon, min_lat, max_lat = bbox
    data = request_json(
        f"{layer_url}/query",
        {
            "f": "geojson",
            "where": "1=1",
            "outFields": out_fields,
            "returnGeometry": "true",
            "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "orderByFields": "OBJECTID",
            "resultOffset": str(offset),
            "resultRecordCount": str(page_size),
        },
    )
    return data.get("features", [])


def features_to_gdf(features: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    rows: list[dict[str, Any]] = []
    geoms = []
    for feature in features:
        rows.append(feature.get("properties") or {})
        geoms.append(shape(feature["geometry"]) if feature.get("geometry") else None)
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs="EPSG:4326")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("MINLON", "MAXLON", "MINLAT", "MAXLAT"))
    parser.add_argument("--layer-url", default=DEFAULT_LAYER_URL)
    parser.add_argument("--out-gpkg", type=Path, required=True)
    parser.add_argument("--layer-name", default="nhd_flowline_context")
    parser.add_argument("--raw-geojson", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=2000)
    parser.add_argument("--out-fields", default=",".join(DEFAULT_FIELDS))
    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    layer = request_json(args.layer_url, {"f": "json"})
    count = query_count(args.layer_url, bbox)
    max_record_count = int(layer.get("maxRecordCount") or args.page_size)
    page_size = min(args.page_size, max_record_count)

    features: list[dict[str, Any]] = []
    page_counts: list[int] = []
    for offset in range(0, count, page_size):
        page = fetch_page(args.layer_url, bbox, args.out_fields, offset, page_size)
        features.extend(page)
        page_counts.append(len(page))
        print(f"offset={offset} fetched={len(page)} total={len(features)}")
        if not page:
            break

    gdf = features_to_gdf(features)
    args.out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(args.out_gpkg, layer=args.layer_name, driver="GPKG")
    if args.raw_geojson:
        args.raw_geojson.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(args.raw_geojson, driver="GeoJSON")

    manifest = {
        "source": "The National Map NHD REST Flowline - Large Scale",
        "layer_url": args.layer_url,
        "layer_name": layer.get("name"),
        "geometry_type": layer.get("geometryType"),
        "bbox": list(bbox),
        "estimated_count": count,
        "fetched_count": int(len(gdf)),
        "page_size": page_size,
        "page_counts": page_counts,
        "out_gpkg": str(args.out_gpkg),
        "out_layer": args.layer_name,
        "raw_geojson": str(args.raw_geojson) if args.raw_geojson else None,
        "out_fields": args.out_fields.split(","),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
