#!/usr/bin/env python3
"""Assign nearest model/statistic points to NHD flowlines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


DEFAULT_PROJECTED_CRS = "EPSG:3338"


def detect_stat_field(stats: pd.DataFrame, requested: str) -> str:
    if requested in stats.columns:
        return requested
    aliases = ["mean_discharge", "mean", "full_period_mean", "mean_cfs", "mean_m3s", "mean_m3_s"]
    for alias in aliases:
        if alias in stats.columns:
            return alias
    raise SystemExit(f"stat field {requested!r} not found; available columns: {list(stats.columns)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowlines", type=Path, required=True)
    parser.add_argument("--flowline-layer", default="nhd_flowline_context")
    parser.add_argument("--points-csv", type=Path, required=True)
    parser.add_argument("--stats-csv", type=Path)
    parser.add_argument("--stat-field", default="mean_discharge")
    parser.add_argument("--segment-id-field", default="segment_id")
    parser.add_argument("--point-lon-field", default="centroid_lon")
    parser.add_argument("--point-lat-field", default="centroid_lat")
    parser.add_argument("--out-gpkg", type=Path, required=True)
    parser.add_argument("--out-layer", default="nhd_flowline_nearest_prms")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--projected-crs", default=DEFAULT_PROJECTED_CRS)
    parser.add_argument("--distance-threshold-m", type=float, default=5000.0)
    args = parser.parse_args()

    flowlines = gpd.read_file(args.flowlines, layer=args.flowline_layer)
    points_df = pd.read_csv(args.points_csv)
    required = {args.segment_id_field, args.point_lon_field, args.point_lat_field}
    missing = sorted(required - set(points_df.columns))
    if missing:
        raise SystemExit(f"missing point columns: {missing}")

    stats_status = "missing"
    points_df[args.stat_field] = np.nan
    if args.stats_csv and args.stats_csv.exists():
        stats = pd.read_csv(args.stats_csv)
        if args.segment_id_field not in stats.columns:
            raise SystemExit(f"stats table missing {args.segment_id_field!r}")
        source_field = detect_stat_field(stats, args.stat_field)
        stats_status = f"joined:{source_field}"
        keep = stats[[args.segment_id_field, source_field]].rename(columns={source_field: args.stat_field})
        points_df = points_df.drop(columns=[args.stat_field], errors="ignore").merge(keep, on=args.segment_id_field, how="left")

    points = gpd.GeoDataFrame(
        points_df,
        geometry=[Point(xy) for xy in zip(points_df[args.point_lon_field], points_df[args.point_lat_field])],
        crs="EPSG:4326",
    )

    flowlines = flowlines.reset_index(drop=True).copy()
    flowlines["flowline_index"] = flowlines.index.astype(int)
    flow_proj = flowlines.to_crs(args.projected_crs)
    point_proj = points.to_crs(args.projected_crs)

    joined = gpd.sjoin_nearest(flow_proj, point_proj, how="left", distance_col="nearest_distance_m")
    joined = joined.sort_values(["flowline_index", "nearest_distance_m"]).drop_duplicates("flowline_index", keep="first")
    joined["nearest_assignment_method"] = "nearest PRMS point to NHD flowline"
    joined["nearest_distance_warning"] = joined["nearest_distance_m"] > args.distance_threshold_m
    joined["statistic_status"] = stats_status

    assigned = joined.to_crs(flowlines.crs)
    args.out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_file(args.out_gpkg, layer=args.out_layer, driver="GPKG")

    table = pd.DataFrame(assigned.drop(columns="geometry"))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    metadata = {
        "flowline_count": int(len(assigned)),
        "point_count": int(len(points)),
        "stats_status": stats_status,
        "stat_field": args.stat_field,
        "distance_threshold_m": args.distance_threshold_m,
        "warning_count": int(assigned["nearest_distance_warning"].sum()),
        "out_gpkg": str(args.out_gpkg),
        "out_layer": args.out_layer,
        "out_csv": str(args.out_csv),
    }
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
