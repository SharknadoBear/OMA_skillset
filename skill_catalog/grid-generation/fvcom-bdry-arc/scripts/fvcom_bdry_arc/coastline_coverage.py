"""Centered source-footprint audit for GSHHS-backed FVCOM topology."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import nearest_points, snap, unary_union

from .projection import LocalProjection, project_geometry, project_geometry_densified


MIN_COVERAGE_FACTOR = 2.0
SOURCE_FRAME_LENGTH_TOLERANCE_M = 1.0


def _sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
    try:
        value = gpd.read_file(path, layer=layer)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if value.crs is None:
        value = value.set_crs("EPSG:4326")
    return value.to_crs("EPSG:4326")


def _discover_manifest(gpkg: Path) -> Path | None:
    exact = gpkg.with_name(gpkg.name.replace("_gshhs_land.gpkg", "_gshhs_manifest.json"))
    if exact.is_file():
        return exact
    candidates = sorted(gpkg.parent.glob("*_gshhs_manifest.json"))
    return candidates[0] if len(candidates) == 1 else None


def _project_union(gdf: gpd.GeoDataFrame, projection: LocalProjection):
    geometries = []
    for geometry in gdf.geometry:
        if geometry is None or geometry.is_empty:
            continue
        projected = project_geometry(geometry, projection)
        if not projected.is_empty:
            geometries.append(projected)
    if not geometries:
        return GeometryCollection()
    merged = geometries[0]
    for geometry in geometries[1:]:
        merged = unary_union([merged, snap(geometry, merged, 0.01)])
    return merged


def _span(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    return max(float(bounds[2] - bounds[0]), 1.0), max(float(bounds[3] - bounds[1]), 1.0)


def audit_coastline_source_coverage(
    coastline_gpkg: str | Path,
    region_polygon_lonlat: Polygon,
    projection: LocalProjection,
    physical_coastline_xy,
    *,
    anchors_xy: list[Point] | None = None,
    delivered_boundary_xy=None,
    target_resolution_m: float = 250.0,
    output_dir: str | Path | None = None,
    name: str = "fvcom_boundary",
) -> dict[str, Any]:
    """Validate that no model boundary depends on a clipped source frame."""
    gpkg = Path(coastline_gpkg)
    try:
        available_layers = set(gpd.list_layers(gpkg)["name"].astype(str))
    except Exception:
        available_layers = set()
    required_layers = {
        "land_polygons",
        "coastline_lines",
        "source_footprint",
        "source_frame",
        "model_bbox",
    }
    missing_layers = sorted(required_layers.difference(available_layers))
    manifest_path = _discover_manifest(gpkg)
    manifest = _read_json(manifest_path)
    topology = manifest.get("topology_coverage") if isinstance(manifest, dict) else None
    source_layer = _read_layer(gpkg, "source_footprint")
    frame_layer = _read_layer(gpkg, "source_frame")
    model_layer = _read_layer(gpkg, "model_bbox")
    if frame_layer.empty and not source_layer.empty:
        frame_layer = gpd.GeoDataFrame(geometry=source_layer.geometry.boundary, crs="EPSG:4326")

    region_xy = project_geometry_densified(region_polygon_lonlat, projection)
    source_xy = _project_union(source_layer, projection)
    frame_xy = source_xy.boundary if not source_xy.is_empty else _project_union(frame_layer, projection)
    declared_model_xy = _project_union(model_layer, projection)
    model_xy = declared_model_xy if not declared_model_xy.is_empty else region_xy.envelope
    source_width, source_height = _span(source_xy.bounds) if not source_xy.is_empty else (0.0, 0.0)
    model_width, model_height = _span(model_xy.bounds)
    factor_x = source_width / model_width if source_width else 0.0
    factor_y = source_height / model_height if source_height else 0.0
    center_offset_x = abs(float(source_xy.centroid.x - model_xy.centroid.x)) if not source_xy.is_empty else float("inf")
    center_offset_y = abs(float(source_xy.centroid.y - model_xy.centroid.y)) if not source_xy.is_empty else float("inf")
    projected_centered = bool(
        center_offset_x <= 0.05 * model_width
        and center_offset_y <= 0.05 * model_height
    )
    declared_center_offset = topology.get("source_center_offset_lonlat") if topology else None
    declared_centered = bool(
        topology
        and topology.get("model_bbox_centrally_contained") is True
        and isinstance(declared_center_offset, (list, tuple))
        and len(declared_center_offset) == 2
        and max(abs(float(value)) for value in declared_center_offset) <= 1.0e-9
    )
    centered = bool(projected_centered or declared_centered)
    region_contained = bool(
        not source_xy.is_empty
        and source_xy.buffer(SOURCE_FRAME_LENGTH_TOLERANCE_M).covers(region_xy)
    )

    anchors = list(anchors_xy or [])
    anchor_frame_distances = [float(point.distance(frame_xy)) for point in anchors] if anchors and not frame_xy.is_empty else []
    anchor_coast_distances = [
        float(point.distance(physical_coastline_xy))
        for point in anchors
    ] if anchors and physical_coastline_xy is not None and not physical_coastline_xy.is_empty else []
    physical_landfall_tolerance_m = max(25.0, min(250.0, 0.50 * float(target_resolution_m)))
    source_dependency = 0.0
    if delivered_boundary_xy is not None and not delivered_boundary_xy.is_empty and not frame_xy.is_empty:
        try:
            source_dependency = float(delivered_boundary_xy.intersection(frame_xy).length)
        except Exception:
            source_dependency = float("inf")

    declared_eligible = bool(topology and topology.get("downstream_topology_eligible") is True)
    inferred_eligible = bool(
        factor_x >= MIN_COVERAGE_FACTOR
        and factor_y >= MIN_COVERAGE_FACTOR
        and centered
        and region_contained
    )
    failures: list[str] = []
    if missing_layers:
        failures.append("coastline_source_footprint_incomplete")
    if source_xy.is_empty or not region_contained:
        failures.append("coastline_source_footprint_incomplete")
    if factor_x < MIN_COVERAGE_FACTOR or factor_y < MIN_COVERAGE_FACTOR or not centered:
        failures.append("boundary_geometry_outside_coastline_coverage")
    if topology is not None and not declared_eligible:
        failures.append("coastline_source_manifest_not_topology_eligible")
    if source_dependency > SOURCE_FRAME_LENGTH_TOLERANCE_M:
        failures.append("coastline_source_frame_used_as_land_boundary")
    if anchor_frame_distances and min(anchor_frame_distances) <= SOURCE_FRAME_LENGTH_TOLERANCE_M:
        failures.append("coastline_clip_edge_landfall")
    if anchors and not anchor_coast_distances:
        failures.append("coastline_physical_landfall_evidence_missing")
    elif anchor_coast_distances and max(anchor_coast_distances) > physical_landfall_tolerance_m:
        failures.append("coastline_physical_landfall_evidence_missing")

    output_path = Path(output_dir) if output_dir else None
    whole_map = output_path / "coastline_source_coverage_map.png" if output_path else None
    zoom_map = output_path / "coastline_source_coverage_zoom.png" if output_path else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        _plot_coverage(
            whole_map,
            zoom_map,
            name,
            source_layer,
            source_xy,
            frame_xy,
            model_xy,
            region_xy,
            physical_coastline_xy,
            projection,
            anchors,
            delivered_boundary_xy,
            failures,
        )

    contract: dict[str, Any] = {
        "schema_version": "fvcom_coastline_source_coverage_v1",
        "final_status": "pass" if not failures and (declared_eligible or inferred_eligible) else "needs_review",
        "downstream_eligible": bool(not failures and (declared_eligible or inferred_eligible)),
        "coverage_provenance": "gshhs_manifest_v2" if topology is not None else "geometric_inference",
        "required_layers": sorted(required_layers),
        "missing_required_layers": missing_layers,
        "minimum_coverage_factor": MIN_COVERAGE_FACTOR,
        "coverage_factor_x": float(factor_x),
        "coverage_factor_y": float(factor_y),
        "center_offset_x_m": float(center_offset_x),
        "center_offset_y_m": float(center_offset_y),
        "model_bbox_centrally_contained": centered,
        "projected_model_bbox_centrally_contained": projected_centered,
        "manifest_model_bbox_centrally_contained": declared_centered,
        "manifest_source_center_offset_lonlat": declared_center_offset,
        "region_bpoly_covered": region_contained,
        "source_frame_dependency_length_m": float(source_dependency),
        "source_frame_dependency_limit_m": SOURCE_FRAME_LENGTH_TOLERANCE_M,
        "source_frame_metric_policy": "boundary_of_projected_source_footprint_union",
        "coordinate_policy": "native_longitudes_transformed_directly_without_longitude_warping",
        "projection_crs": projection.crs.to_string(),
        "anchor_to_source_frame_distance_m": anchor_frame_distances,
        "anchor_to_physical_coastline_distance_m": anchor_coast_distances,
        "physical_landfall_tolerance_m": float(physical_landfall_tolerance_m),
        "physical_coastline_only_landfalls": bool(
            not anchors
            or (
                bool(anchor_coast_distances)
                and max(anchor_coast_distances) <= physical_landfall_tolerance_m
            )
        ),
        "gshhs_topology_coverage": topology,
        "source_hashes": {
            "coastline_gpkg": _sha256(gpkg),
            "gshhs_manifest": _sha256(manifest_path),
        },
        "maps": {
            "whole_domain": {"path": str(whole_map) if whole_map else None, "sha256": _sha256(whole_map)},
            "source_edge_zoom": {"path": str(zoom_map) if zoom_map else None, "sha256": _sha256(zoom_map)},
        },
        "failure_taxonomy": list(dict.fromkeys(failures)),
    }
    if output_path:
        contract_path = output_path / "coastline_source_coverage.json"
        contract["contract_path"] = str(contract_path)
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


def _plot_coverage(
    whole_path: Path,
    zoom_path: Path,
    name: str,
    source_layer: gpd.GeoDataFrame,
    source_xy,
    frame_xy,
    model_xy,
    region_xy: Polygon,
    physical_coastline_xy,
    projection: LocalProjection,
    anchors: list[Point],
    delivered_boundary_xy,
    failures: list[str],
) -> None:
    physical = physical_coastline_xy if physical_coastline_xy is not None else GeometryCollection()
    delivered = delivered_boundary_xy if delivered_boundary_xy is not None else GeometryCollection()
    display_crs = projection.crs

    fig, ax = plt.subplots(figsize=(11, 9))
    if not source_xy.is_empty:
        gpd.GeoSeries([source_xy], crs=display_crs).plot(ax=ax, facecolor="#dbeafe", edgecolor="#dc2626", linewidth=1.2, alpha=0.18, label="GSHHS source footprint")
    if not frame_xy.is_empty:
        gpd.GeoSeries([frame_xy], crs=display_crs).plot(ax=ax, color="#dc2626", linewidth=1.4, linestyle="--", label="source frame")
    if not model_xy.is_empty:
        gpd.GeoSeries([model_xy], crs=display_crs).boundary.plot(ax=ax, color="#2563eb", linewidth=1.5, linestyle="-.", label="declared model bbox")
    gpd.GeoSeries([region_xy], crs=display_crs).boundary.plot(ax=ax, color="#111827", linewidth=2.0, label="RegionBPoly")
    if not physical.is_empty:
        gpd.GeoSeries([physical], crs=display_crs).plot(ax=ax, color="#166534", linewidth=0.8, label="physical GSHHS coastline")
    if not delivered.is_empty:
        gpd.GeoSeries([delivered], crs=display_crs).plot(ax=ax, color="#f97316", linewidth=2.0, label="delivered exterior")
    if anchors:
        gpd.GeoSeries(anchors, crs=display_crs).plot(ax=ax, color="#7c3aed", markersize=45, label="landfall anchors")
    ax.set_title(f"{name} coastline source coverage | {'PASS' if not failures else 'NEEDS REVIEW'}")
    ax.set_xlabel("Projected easting (m)")
    ax.set_ylabel("Projected northing (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(whole_path, dpi=180)
    plt.close(fig)

    focus = delivered if not delivered.is_empty else region_xy.boundary
    try:
        p_focus, p_frame = nearest_points(focus, frame_xy)
        minx, miny = min(p_focus.x, p_frame.x), min(p_focus.y, p_frame.y)
        maxx, maxy = max(p_focus.x, p_frame.x), max(p_focus.y, p_frame.y)
    except Exception:
        minx, miny, maxx, maxy = region_xy.bounds
    pad = max(1000.0, 0.10 * max(maxx - minx, maxy - miny, 1.0))
    fig, ax = plt.subplots(figsize=(10, 8))
    if not source_xy.is_empty:
        gpd.GeoSeries([source_xy], crs=display_crs).plot(ax=ax, facecolor="#dbeafe", edgecolor="#dc2626", linewidth=1.2, alpha=0.18)
    if not frame_xy.is_empty:
        gpd.GeoSeries([frame_xy], crs=display_crs).plot(ax=ax, color="#dc2626", linewidth=2.0, linestyle="--")
    gpd.GeoSeries([region_xy], crs=display_crs).boundary.plot(ax=ax, color="#111827", linewidth=1.8)
    if not physical.is_empty:
        gpd.GeoSeries([physical], crs=display_crs).plot(ax=ax, color="#166534", linewidth=0.9)
    if not delivered.is_empty:
        gpd.GeoSeries([delivered], crs=display_crs).plot(ax=ax, color="#f97316", linewidth=2.2)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_title(f"{name} closest approach to GSHHS source frame")
    ax.set_xlabel("Projected easting (m)")
    ax.set_ylabel("Projected northing (m)")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(zoom_path, dpi=180)
    plt.close(fig)


__all__ = ["audit_coastline_source_coverage"]
