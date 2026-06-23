#!/usr/bin/env python3
"""Health-check fetched NHD flowline vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


REQUIRED_FIELDS = {"permanent_identifier", "gnis_name", "lengthkm", "reachcode", "ftype", "fcode", "innetwork"}


def parse_bbox(values: list[float]) -> tuple[float, float, float, float]:
    min_lon, max_lon, min_lat, max_lat = values
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError(f"invalid bbox order: {values}")
    return min_lon, max_lon, min_lat, max_lat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowlines", type=Path, required=True)
    parser.add_argument("--layer", default="nhd_flowline_context")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("MINLON", "MAXLON", "MINLAT", "MAXLAT"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bbox = parse_bbox(args.bbox)
    gdf = gpd.read_file(args.flowlines, layer=args.layer)
    gdf_ll = gdf.to_crs("EPSG:4326") if gdf.crs else gdf.set_crs("EPSG:4326")
    bbox_geom = gpd.GeoSeries([box(bbox[0], bbox[2], bbox[1], bbox[3])], crs="EPSG:4326").iloc[0]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest and args.manifest.exists() else {}
    missing_fields = sorted(REQUIRED_FIELDS - set(gdf.columns))
    valid = gdf.geometry.is_valid if len(gdf) else []
    intersects = gdf_ll.geometry.intersects(bbox_geom) if len(gdf_ll) else []
    health = {
        "flowlines": str(args.flowlines),
        "layer": args.layer,
        "bbox": list(bbox),
        "feature_count": int(len(gdf)),
        "crs": str(gdf.crs),
        "bounds_epsg4326": [float(v) for v in gdf_ll.total_bounds] if len(gdf_ll) else None,
        "missing_required_fields": missing_fields,
        "valid_geometry_count": int(valid.sum()) if len(gdf) else 0,
        "invalid_geometry_count": int((~valid).sum()) if len(gdf) else 0,
        "intersecting_bbox_count": int(intersects.sum()) if len(gdf_ll) else 0,
        "manifest_estimated_count": manifest.get("estimated_count"),
        "manifest_fetched_count": manifest.get("fetched_count"),
        "count_matches_manifest": (
            int(len(gdf)) == int(manifest["fetched_count"]) if manifest.get("fetched_count") is not None else None
        ),
        "passed": bool(len(gdf) > 0 and not missing_fields and (int(valid.sum()) == len(gdf)) and int(intersects.sum()) > 0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(health, indent=2), encoding="utf-8")
    print(json.dumps(health, indent=2))
    return 0 if health["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
