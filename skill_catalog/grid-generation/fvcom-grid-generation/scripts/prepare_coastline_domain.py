"""Prepare a coastline-aware FVCOM domain package for visual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from fvcom_grid_generation.bathymetry import load_bathymetry
from fvcom_grid_generation.coastline_domain import DomainPrepareConfig, prepare_coastline_domain
from fvcom_grid_generation.size_field import SizeFieldConfig, build_size_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a coastline-aware FVCOM domain and stop for visual review.")
    parser.add_argument("bathymetry", help="CUDEM/local bathymetry NetCDF or GeoTIFF.")
    parser.add_argument("coastline", help="CUSP coastline GeoPackage.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", default="coastline_domain")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    parser.add_argument("--reference-2dm", default=None, help="Optional reference mesh; first NS is used as open boundary.")
    parser.add_argument("--offshore-side", choices=["east", "west", "north", "south"], default=None)
    parser.add_argument(
        "--open-boundary-mode",
        choices=["auto", "ellipse", "bezier", "bbox-bow", "manual-line", "anchor-iterate"],
        default="auto",
        help="Non-reference open-boundary designer mode. Ignored when --reference-2dm is supplied.",
    )
    parser.add_argument("--manual-open-boundary", default=None, help="Line vector file used with --open-boundary-mode manual-line.")
    parser.add_argument("--ocean-direction", nargs=2, type=float, metavar=("DX", "DY"), help="Anchor-iterate ocean direction in projected map coordinates.")
    parser.add_argument("--anchor-seeds", nargs=4, type=float, metavar=("LON1", "LAT1", "LON2", "LAT2"), help="Anchor-iterate rough visual seed/probe coordinates.")
    parser.add_argument("--anchor-seed-json", default=None, help="JSON file containing ocean_direction and anchor_seeds for anchor-iterate mode.")
    parser.add_argument("--anchor-max-iterations", type=int, default=40)
    parser.add_argument("--anchor-step-factor", type=float, default=1.0, help="Initial anchor-iterate step as a multiple of target resolution.")
    parser.add_argument("--anchor-min-step-factor", type=float, default=0.1, help="Minimum anchor-iterate step as a multiple of target resolution.")
    parser.add_argument("--anchor-bbox-touch-fraction", type=float, default=0.02, help="Target gap from ocean-facing bbox side as a fraction of bbox width/height.")
    parser.add_argument("--open-boundary-candidate-rounds", type=int, default=3)
    parser.add_argument("--target-resolution-m", type=float, default=None)
    parser.add_argument("--max-nodes", type=int, default=500_000)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--gradation", type=float, default=0.15)
    parser.add_argument("--gradation-iterations", type=int, default=40)
    parser.add_argument("--max-mask-cells", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bbox = tuple(args.bbox) if args.bbox else None
    config = DomainPrepareConfig(
        target_resolution_m=args.target_resolution_m,
        max_nodes=args.max_nodes,
        min_depth_m=args.min_depth_m,
        offshore_side=args.offshore_side,
        max_mask_cells=args.max_mask_cells,
        gradation=args.gradation,
        gradation_iterations=args.gradation_iterations,
        open_boundary_mode=args.open_boundary_mode,
        manual_open_boundary=args.manual_open_boundary,
        ocean_direction=tuple(args.ocean_direction) if args.ocean_direction else None,
        anchor_seeds=tuple(args.anchor_seeds) if args.anchor_seeds else None,
        anchor_seed_json=args.anchor_seed_json,
        anchor_max_iterations=args.anchor_max_iterations,
        anchor_step_factor=args.anchor_step_factor,
        anchor_min_step_factor=args.anchor_min_step_factor,
        anchor_bbox_touch_fraction=args.anchor_bbox_touch_fraction,
        open_boundary_candidate_rounds=args.open_boundary_candidate_rounds,
    )
    prepared = prepare_coastline_domain(
        args.bathymetry,
        args.coastline,
        args.run_dir,
        args.name,
        bbox_wsen=bbox,
        reference_2dm=args.reference_2dm,
        config=config,
    )
    run_dir = Path(args.run_dir)
    metadata_path = run_dir / f"{args.name}_domain_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    size_outputs = _write_size_products(args.bathymetry, run_dir, args.name, metadata)
    review_png = _write_review_plot(run_dir, args.name, metadata)
    candidate_png = _write_candidate_contact_sheet(args.bathymetry, run_dir, args.name, metadata)
    anchor_outputs = _write_anchor_products(args.bathymetry, run_dir, args.name, metadata)
    metadata["outputs"].update(size_outputs)
    metadata["outputs"].update(anchor_outputs)
    metadata["outputs"]["domain_review_png"] = str(review_png)
    if candidate_png is not None:
        metadata["outputs"]["open_boundary_candidate_contact_sheet_png"] = str(candidate_png)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    review_path = Path(metadata["outputs"]["visual_review_json"])
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["image_paths"] = {"domain_review_png": str(review_png)}
    if candidate_png is not None:
        review["image_paths"]["open_boundary_candidate_contact_sheet_png"] = str(candidate_png)
    if "anchor_seed_map_png" in anchor_outputs:
        review["image_paths"]["anchor_seed_map_png"] = anchor_outputs["anchor_seed_map_png"]
    if "anchor_iteration_review_png" in anchor_outputs:
        review["image_paths"]["anchor_iteration_review_png"] = anchor_outputs["anchor_iteration_review_png"]
    if "anchor_iteration_review_plain_png" in anchor_outputs:
        review["image_paths"]["anchor_iteration_review_plain_png"] = anchor_outputs["anchor_iteration_review_plain_png"]
    review["vector_paths"] = {"domain_gpkg": metadata["outputs"]["domain_gpkg"]}
    if "anchor_points_geojson" in anchor_outputs:
        review["vector_paths"]["anchor_points_geojson"] = anchor_outputs["anchor_points_geojson"]
    anchor_data_paths = {
        key: value
        for key, value in anchor_outputs.items()
        if key in {"anchor_seed_json", "anchor_report_json"}
    }
    if anchor_data_paths:
        review["data_paths"] = anchor_data_paths
    review["notes"] = (
        "Agent or human visual review required before meshing. "
        "Candidate open-boundary arcs are algorithmically generated and scored; "
        "Codex/agent review must be recorded as agent visual inspection, not human scientific approval."
    )
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def _write_size_products(bathymetry: str, run_dir: Path, name: str, metadata: dict) -> dict[str, str]:
    bathy = load_bathymetry(bathymetry)
    target_resolution = float(metadata["target_resolution_m"])
    gradation = float((metadata.get("gradation_report") or {}).get("gradation", 0.15))
    config = SizeFieldConfig(min_size=target_resolution, gradation=gradation, gradation_iterations=40)
    size_field = build_size_field(bathy, config)

    nc = run_dir / f"{name}_size_fields.nc"
    xr.Dataset(
        {
            "raw_size_m": (("lat", "lon"), size_field.raw_size),
            "limited_size_m": (("lat", "lon"), size_field.size),
            "slope": (("lat", "lon"), size_field.slope),
        },
        coords={"lon": bathy.lon, "lat": bathy.lat},
        attrs={"gradation_report": json.dumps(size_field.gradation_report or {})},
    ).to_netcdf(nc)

    png = run_dir / f"{name}_size_fields.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    raw = axes[0].pcolormesh(bathy.lon, bathy.lat, size_field.raw_size, shading="auto")
    axes[0].set_title("Raw size field (m)")
    fig.colorbar(raw, ax=axes[0])
    limited = axes[1].pcolormesh(bathy.lon, bathy.lat, size_field.size, shading="auto")
    axes[1].set_title("Gradation-limited size field (m)")
    fig.colorbar(limited, ax=axes[1])
    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    fig.savefig(png, dpi=160)
    plt.close(fig)
    return {"size_fields_netcdf": str(nc), "size_fields_png": str(png)}


def _write_review_plot(run_dir: Path, name: str, metadata: dict) -> Path:
    import geopandas as gpd

    gpkg = Path(metadata["outputs"]["domain_gpkg"])
    domain = gpd.read_file(gpkg, layer="domain")
    open_boundary = gpd.read_file(gpkg, layer="open_boundary")
    land = gpd.read_file(gpkg, layer="land_boundary")
    try:
        candidates = gpd.read_file(gpkg, layer="open_boundary_candidates")
    except Exception:
        candidates = gpd.GeoDataFrame(geometry=[], crs=domain.crs)
    try:
        islands = gpd.read_file(gpkg, layer="island_boundary")
    except Exception:
        islands = gpd.GeoDataFrame(geometry=[], crs=domain.crs)
    try:
        coastline = gpd.read_file(gpkg, layer="coastline_candidates")
    except Exception:
        coastline = gpd.GeoDataFrame(geometry=[], crs=domain.crs)

    png = run_dir / f"{name}_domain_review.png"
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    domain.boundary.plot(ax=ax, color="black", linewidth=1.2, label="Prepared domain")
    if not candidates.empty:
        chosen = candidates[candidates.get("selected", False).astype(bool)] if "selected" in candidates else candidates.iloc[0:0]
        others = candidates.drop(chosen.index) if not chosen.empty else candidates
        if not others.empty:
            others.plot(ax=ax, color="0.55", linewidth=0.45, alpha=0.45, label="Open-boundary candidates")
    if not coastline.empty:
        coastline.plot(ax=ax, color="#00bcd4", linewidth=0.35, alpha=0.8, label="CUSP candidates")
    land.plot(ax=ax, color="black", linewidth=1.0, label="Land boundary")
    if not islands.empty:
        islands.plot(ax=ax, color="orange", linewidth=0.9, label="Island holes")
    open_boundary.plot(ax=ax, color="red", linewidth=2.0, label="Open boundary")
    west, south, east, north = metadata["bbox_wsen"]
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{name}: coastline-aware domain review")
    _legend_outside(fig, ax)
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return png


def _write_candidate_contact_sheet(bathymetry: str, run_dir: Path, name: str, metadata: dict) -> Path | None:
    import geopandas as gpd

    gpkg = Path(metadata["outputs"]["domain_gpkg"])
    try:
        candidates = gpd.read_file(gpkg, layer="open_boundary_candidates")
    except Exception:
        return None
    if candidates.empty:
        return None
    bathy = load_bathymetry(bathymetry)
    lon_plot, lat_plot, depth_plot = _downsample_for_plot(bathy.lon, bathy.lat, bathy.depth, max_cells=90_000)
    try:
        coastline = gpd.read_file(gpkg, layer="coastline_candidates")
    except Exception:
        coastline = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if len(coastline) > 10_000:
        coastline = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    domain = gpd.read_file(gpkg, layer="domain")
    sort_col = "score" if "score" in candidates else None
    if sort_col:
        candidates = candidates.sort_values(sort_col, ascending=False)
    top = candidates.head(9).reset_index(drop=True)
    n = len(top)
    cols = 3
    rows = int(np.ceil(n / cols))
    png = run_dir / f"{name}_open_boundary_candidates.png"
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False, constrained_layout=True)
    west, south, east, north = metadata["bbox_wsen"]
    for idx, ax in enumerate(axes.ravel()):
        if idx >= n:
            ax.axis("off")
            continue
        row = top.iloc[idx]
        depth = ax.pcolormesh(lon_plot, lat_plot, depth_plot, shading="auto", cmap="viridis", alpha=0.75)
        domain.boundary.plot(ax=ax, color="black", linewidth=0.7)
        if not coastline.empty:
            coastline.plot(ax=ax, color="#00bcd4", linewidth=0.25, alpha=0.6)
        gpd.GeoSeries([row.geometry], crs=candidates.crs).plot(ax=ax, color="red" if bool(row.get("selected", False)) else "orange", linewidth=2.0)
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{row.get('candidate_id', idx)}\n"
            f"{row.get('family', 'candidate')} score={float(row.get('score', 0.0)):.1f} "
            f"wet={float(row.get('wet_fraction', 0.0)):.2f}",
            fontsize=9,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(depth, ax=ax, label="Depth positive down (m)")
    fig.savefig(png, dpi=160)
    plt.close(fig)
    return png


def _write_anchor_products(bathymetry: str, run_dir: Path, name: str, metadata: dict) -> dict[str, str]:
    import geopandas as gpd
    from shapely.geometry import Point

    design = metadata.get("open_boundary_design") or {}
    if design.get("open_boundary_mode") != "anchor-iterate":
        return {}
    gpkg = Path(metadata["outputs"]["domain_gpkg"])
    bathy = load_bathymetry(bathymetry)
    lon_plot, lat_plot, depth_plot = _downsample_for_plot(bathy.lon, bathy.lat, bathy.depth, max_cells=90_000)
    try:
        coastline = gpd.read_file(gpkg, layer="coastline_candidates")
    except Exception:
        coastline = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if len(coastline) > 100_000:
        coastline = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    try:
        candidates = gpd.read_file(gpkg, layer="open_boundary_candidates")
    except Exception:
        candidates = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    west, south, east, north = metadata["bbox_wsen"]
    seed = design.get("seed") or {}
    seed_points = seed.get("anchor_seeds") or []
    direction = seed.get("ocean_direction") or []
    outer_points = (design.get("anchor_iteration") or {}).get("outer_envelope_preview_lonlat") or []
    outputs: dict[str, str] = {}

    seed_json = run_dir / f"{name}_anchor_seed.json"
    seed_payload = {
        "ocean_direction": direction,
        "anchor_seeds": seed_points,
        "reviewer": seed.get("reviewer", "cli"),
        "notes": seed.get("notes", ""),
        "visual_input_type": "agent_or_human_rough_seed",
        "exact_anchor_source": "coastline_geometry_intersections",
    }
    seed_json.write_text(json.dumps(seed_payload, indent=2), encoding="utf-8")
    outputs["anchor_seed_json"] = str(seed_json)

    seed_png = run_dir / f"{name}_anchor_seed_map.png"
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    depth = ax.pcolormesh(lon_plot, lat_plot, depth_plot, shading="auto", cmap="viridis", alpha=0.75)
    if not coastline.empty:
        coastline.plot(ax=ax, color="#00bcd4", linewidth=0.25, alpha=0.65, label="Coastline")
    if outer_points:
        outer_arr = np.asarray(outer_points, dtype=float)
        ax.scatter(outer_arr[:, 0], outer_arr[:, 1], color="#ff9800", s=2, alpha=0.55, label="Outer-envelope samples")
    if len(seed_points) == 4:
        ax.scatter([seed_points[0], seed_points[2]], [seed_points[1], seed_points[3]], color="red", s=40, label="Visual seed probes")
    if len(direction) == 2 and len(seed_points) == 4:
        cx = 0.5 * (seed_points[0] + seed_points[2])
        cy = 0.5 * (seed_points[1] + seed_points[3])
        scale = 0.12 * max(east - west, north - south)
        ax.arrow(
            cx,
            cy,
            float(direction[0]) * scale,
            float(direction[1]) * scale,
            facecolor="white",
            edgecolor="black",
            width=0.004,
            label="Ocean direction",
        )
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{name}: anchor visual seed map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    _legend_outside(fig, ax)
    fig.colorbar(depth, ax=ax, label="Depth positive down (m)")
    fig.savefig(seed_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs["anchor_seed_map_png"] = str(seed_png)

    review_png = run_dir / f"{name}_anchor_iteration_review.png"
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    depth = ax.pcolormesh(lon_plot, lat_plot, depth_plot, shading="auto", cmap="viridis", alpha=0.75)
    if not coastline.empty:
        coastline.plot(ax=ax, color="#00bcd4", linewidth=0.25, alpha=0.65, label="Coastline")
    if outer_points:
        outer_arr = np.asarray(outer_points, dtype=float)
        ax.scatter(outer_arr[:, 0], outer_arr[:, 1], color="#ff9800", s=2, alpha=0.55, label="Outer-envelope samples")
    if not candidates.empty:
        selected = candidates[candidates.get("selected", False).astype(bool)] if "selected" in candidates else candidates.iloc[0:0]
        others = candidates.drop(selected.index) if not selected.empty else candidates
        if not others.empty:
            others.plot(ax=ax, color="0.6", linewidth=0.4, alpha=0.45, label="Iteration arcs")
        if not selected.empty:
            selected.plot(ax=ax, color="red", linewidth=2.0, label="Selected/final arc")
    anchors = (design.get("anchor_iteration") or {}).get("anchor_points_lonlat") or []
    if anchors:
        ax.scatter([pt[0] for pt in anchors], [pt[1] for pt in anchors], color="yellow", edgecolor="black", s=60, label="Snapped anchors")
    touch_point = (design.get("selected_metrics") or {}).get("bbox_touch_point_lonlat") or (design.get("anchor_iteration") or {}).get("bbox_touch_point_lonlat")
    if touch_point and len(touch_point) == 2:
        ax.scatter([float(touch_point[0])], [float(touch_point[1])], color="magenta", edgecolor="black", s=55, marker="X", label="BBox-touch point")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{name}: anchor iteration review")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    _legend_outside(fig, ax)
    fig.colorbar(depth, ax=ax, label="Depth positive down (m)")
    fig.savefig(review_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs["anchor_iteration_review_png"] = str(review_png)

    plain_png = run_dir / f"{name}_anchor_iteration_review_plain.png"
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    if not coastline.empty:
        coastline.plot(ax=ax, color="#00bcd4", linewidth=0.25, alpha=0.8, label="Coastline")
    if outer_points:
        outer_arr = np.asarray(outer_points, dtype=float)
        ax.scatter(outer_arr[:, 0], outer_arr[:, 1], color="#ff9800", s=2, alpha=0.65, label="Outer-envelope samples")
    if not candidates.empty:
        selected = candidates[candidates.get("selected", False).astype(bool)] if "selected" in candidates else candidates.iloc[0:0]
        others = candidates.drop(selected.index) if not selected.empty else candidates
        if not others.empty:
            others.plot(ax=ax, color="0.6", linewidth=0.4, alpha=0.45, label="Iteration arcs")
        if not selected.empty:
            selected.plot(ax=ax, color="red", linewidth=2.0, label="Selected/final arc")
    if anchors:
        ax.scatter([pt[0] for pt in anchors], [pt[1] for pt in anchors], color="yellow", edgecolor="black", s=60, label="Snapped anchors")
    if touch_point and len(touch_point) == 2:
        ax.scatter([float(touch_point[0])], [float(touch_point[1])], color="magenta", edgecolor="black", s=55, marker="X", label="BBox-touch point")
    ax.plot([west, east, east, west, west], [south, south, north, north, south], color="red", linewidth=1.0, alpha=0.8, label="Requested bbox")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{name}: anchor iteration review (plain coastline)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    _legend_outside(fig, ax)
    fig.savefig(plain_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs["anchor_iteration_review_plain_png"] = str(plain_png)

    report = run_dir / f"{name}_anchor_report.json"
    report.write_text(json.dumps(design, indent=2), encoding="utf-8")
    outputs["anchor_report_json"] = str(report)

    anchors = (design.get("anchor_iteration") or {}).get("anchor_points_lonlat") or []
    if anchors:
        rows = [{"anchor_id": i + 1, "geometry": Point(float(pt[0]), float(pt[1]))} for i, pt in enumerate(anchors)]
        anchor_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        anchor_path = run_dir / f"{name}_anchor_points.geojson"
        anchor_gdf.to_file(anchor_path, driver="GeoJSON")
        outputs["anchor_points_geojson"] = str(anchor_path)
    return outputs


def _legend_outside(fig, ax, fontsize: int = 8) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label.startswith("_") or label in unique_labels:
            continue
        unique_handles.append(handle)
        unique_labels.append(label)
    if unique_handles:
        fig.legend(unique_handles, unique_labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=fontsize, frameon=True)


def _downsample_for_plot(lon, lat, field, max_cells: int = 90_000):
    field = np.asarray(field)
    stride = max(1, int(np.ceil(np.sqrt(field.size / max(max_cells, 1)))))
    return np.asarray(lon)[::stride], np.asarray(lat)[::stride], field[::stride, ::stride]


if __name__ == "__main__":
    main()
