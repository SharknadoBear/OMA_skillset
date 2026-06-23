#!/usr/bin/env python3
"""Map nearest-assigned NHD flowlines and optional statistic values."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, box


CORE_BBOX = (-135.7254414, -134.9430247, 58.8768858, 59.5301701)


def bbox_frame(bbox: tuple[float, float, float, float], name: str) -> gpd.GeoDataFrame:
    min_lon, max_lon, min_lat, max_lat = bbox
    return gpd.GeoDataFrame({"name": [name]}, geometry=[box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326")


def add_basemap(ax) -> str:
    try:
        import contextily as cx

        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, attribution_size=6, reset_extent=False)
        return "CartoDB Positron / OpenStreetMap contributors"
    except Exception as exc:  # pragma: no cover
        ax.text(0.01, 0.01, f"Basemap unavailable: {exc}", transform=ax.transAxes, fontsize=7, color="0.35")
        return "unavailable"


def make_points(points_csv: Path, lon_field: str, lat_field: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(points_csv).dropna(subset=[lon_field, lat_field]).copy()
    return gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df[lon_field], df[lat_field])], crs="EPSG:4326")


def setup_axes(gdf_ll: gpd.GeoDataFrame, bbox: tuple[float, float, float, float] | None):
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    if bbox:
        frame = bbox_frame(bbox, "context").to_crs("EPSG:3857")
        extent = frame.total_bounds
    else:
        extent = gdf_ll.to_crs("EPSG:3857").total_bounds
    pad_x = (extent[2] - extent[0]) * 0.045
    pad_y = (extent[3] - extent[1]) * 0.045
    ax.set_xlim(extent[0] - pad_x, extent[2] + pad_x)
    ax.set_ylim(extent[1] - pad_y, extent[3] + pad_y)
    return fig, ax


def draw_common(ax, assigned_m: gpd.GeoDataFrame, points_m: gpd.GeoDataFrame, bbox: tuple[float, float, float, float] | None, basemap_source: str) -> None:
    if bbox:
        bbox_frame(bbox, "context").to_crs("EPSG:3857").boundary.plot(
            ax=ax, color="#3f6f4d", linewidth=1.2, linestyle=(0, (5, 3)), zorder=3, label="context box"
        )
    bbox_frame(CORE_BBOX, "core ROI").to_crs("EPSG:3857").boundary.plot(
        ax=ax, color="#9f2f2f", linewidth=1.8, zorder=4, label="core ROI"
    )
    if not points_m.empty:
        points_m.plot(ax=ax, marker="o", markersize=28, color="#d55e00", edgecolor="white", linewidth=0.35, zorder=5, label="PRMS segment points")
    ax.set_axis_off()
    ax.text(0.01, 0.04, f"Basemap: {basemap_source}", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, color="#333333")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assigned-gpkg", type=Path, required=True)
    parser.add_argument("--assigned-layer", default="nhd_flowline_nearest_prms")
    parser.add_argument("--points-csv", type=Path, required=True)
    parser.add_argument("--stat-field", default="mean_discharge")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON", "MAXLON", "MINLAT", "MAXLAT"))
    parser.add_argument("--out-qc-map", type=Path, required=True)
    parser.add_argument("--out-value-map", type=Path)
    parser.add_argument("--point-lon-field", default="centroid_lon")
    parser.add_argument("--point-lat-field", default="centroid_lat")
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()

    assigned = gpd.read_file(args.assigned_gpkg, layer=args.assigned_layer)
    assigned_ll = assigned.to_crs("EPSG:4326")
    points = make_points(args.points_csv, args.point_lon_field, args.point_lat_field)
    assigned_m = assigned_ll.to_crs("EPSG:3857")
    points_m = points.to_crs("EPSG:3857")
    bbox = tuple(args.bbox) if args.bbox else None

    fig, ax = setup_axes(assigned_ll, bbox)
    basemap_source = add_basemap(ax)
    assigned_m.plot(ax=ax, color="#244d73", linewidth=0.75, alpha=0.65, zorder=2, label=f"NHD flowlines ({len(assigned_m)})")
    draw_common(ax, assigned_m, points_m, bbox, basemap_source)
    warning_count = int(assigned_m.get("nearest_distance_warning", pd.Series(dtype=bool)).sum())
    ax.set_title("NHD Flowlines With Nearest PRMS Segment Points", fontsize=14, pad=10)
    ax.text(
        0.01,
        0.98,
        f"{len(assigned_m)} NHD flowlines\n{len(points_m)} PRMS points\n{warning_count} assignments > threshold",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.86, "pad": 4},
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    args.out_qc_map.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_qc_map, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_qc_map}")

    if not args.out_value_map:
        return 0
    if args.stat_field not in assigned_m.columns or assigned_m[args.stat_field].isna().all():
        print(f"skipped value map: no non-null {args.stat_field!r} values")
        return 0

    fig, ax = setup_axes(assigned_ll, bbox)
    basemap_source = add_basemap(ax)
    assigned_m.plot(
        ax=ax,
        column=args.stat_field,
        cmap="viridis",
        linewidth=1.2,
        alpha=0.88,
        zorder=2,
        legend=True,
        legend_kwds={"label": args.stat_field},
    )
    draw_common(ax, assigned_m, points_m, bbox, basemap_source)
    ax.set_title(f"NHD Flowlines Colored By Assigned {args.stat_field}", fontsize=14, pad=10)
    args.out_value_map.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_value_map, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_value_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
