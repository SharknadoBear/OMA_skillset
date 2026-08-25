"""GSHHS-derived RegionBPoly feedback for residual model-frame clipping."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import unary_union

from .projection import local_utm_projection, project_geometry, unproject_geometry


CLIP_FAILURES = {
    "unintended_frame_clip_nontrivial",
    "gshhs_coastline_incomplete_on_landward_boundary",
    "model_boundary_loop_needs_review",
    "residual_boundary_role_pending",
}

RESIDUAL_BOUNDARY_ROLES = {
    "solid_lagoon_closure",
    "secondary_tidal_obc",
    "invalid_geometry",
}

BOUNDARY_COMPLETENESS_ROUTES = {
    "adjust_bpoly",
    "retain_for_role_classification",
    "invalid_geometry",
}


def _line_parts(geometry) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    parts: list[LineString] = []
    for item in getattr(geometry, "geoms", []):
        parts.extend(_line_parts(item))
    return parts


def _outward_normal(side: LineString, polygon: Polygon) -> tuple[float, float]:
    p0, p1 = side.coords[0], side.coords[-1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    norm = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / norm, dx / norm
    midpoint = side.interpolate(0.5, normalized=True)
    centroid = polygon.centroid
    if nx * (midpoint.x - centroid.x) + ny * (midpoint.y - centroid.y) < 0.0:
        nx, ny = -nx, -ny
    return float(nx), float(ny)


def _water_continuation_evidence(
    line: LineString,
    side: LineString,
    polygon: Polygon,
    land_union,
    target_resolution_m: float,
) -> dict[str, Any]:
    """Probe the expanded GSHHS mask just outside one RegionBPoly side."""
    if land_union is None or land_union.is_empty:
        return {
            "status": "unavailable",
            "continues_outward": None,
            "wet_sample_fraction": None,
            "probe_distance_m": None,
        }
    nx, ny = _outward_normal(side, polygon)
    probe_distance = min(
        25_000.0,
        max(1_000.0, 2.0 * target_resolution_m, 0.5 * float(line.length)),
    )
    samples: list[bool] = []
    for along in (0.2, 0.5, 0.8):
        point = line.interpolate(along, normalized=True)
        for fraction in (0.25, 0.5, 1.0):
            sample = Point(
                point.x + nx * probe_distance * fraction,
                point.y + ny * probe_distance * fraction,
            )
            samples.append(not bool(land_union.buffer(0.25).covers(sample)))
    wet_fraction = float(sum(samples) / max(len(samples), 1))
    return {
        "status": "pass",
        "continues_outward": bool(wet_fraction >= 2.0 / 3.0),
        "wet_sample_fraction": wet_fraction,
        "probe_distance_m": float(probe_distance),
        "outward_unit_east_north": [nx, ny],
        "sample_count": len(samples),
    }


def _shoreline_tangent_angle_deg(point: Point, line: LineString, coastline, window_m: float) -> float | None:
    if coastline is None or coastline.is_empty:
        return None
    nearby = _line_parts(coastline.intersection(point.buffer(window_m)))
    if not nearby:
        return None
    selected = max(nearby, key=lambda item: item.length)
    if selected.length <= 0.0 or len(selected.coords) < 2:
        return None
    sx = float(selected.coords[-1][0] - selected.coords[0][0])
    sy = float(selected.coords[-1][1] - selected.coords[0][1])
    lx = float(line.coords[-1][0] - line.coords[0][0])
    ly = float(line.coords[-1][1] - line.coords[0][1])
    denom = math.hypot(sx, sy) * math.hypot(lx, ly)
    if denom <= 0.0:
        return None
    cosine = max(-1.0, min(1.0, abs((sx * lx + sy * ly) / denom)))
    return float(math.degrees(math.acos(cosine)))


def _required_feature_conflicts(
    region: dict[str, Any],
    line: LineString,
    projection,
    target_resolution_m: float,
) -> list[str]:
    conflicts: list[str] = []
    influence_m = max(5_000.0, 2.0 * target_resolution_m, 2.0 * float(line.length))
    features = (region.get("target_region_features") or {}).get("features", [])
    for feature in features:
        if not feature.get("required", False):
            continue
        category = str(feature.get("category", "")).lower()
        role = str(feature.get("role", "")).lower()
        if not any(token in f"{category} {role}" for token in ("river", "channel", "water", "target_region")):
            continue
        geometry = feature.get("geometry")
        if feature.get("type") != "point" or not isinstance(geometry, list) or len(geometry) < 2:
            continue
        point_xy = project_geometry(Point(float(geometry[0]), float(geometry[1])), projection)
        if point_xy.distance(line) <= influence_m:
            conflicts.append(str(feature.get("id") or feature.get("label") or "required_feature"))
    return sorted(set(conflicts))


def evaluate_open_exterior_metrics(
    unintended_length_m: float,
    landward_length_m: float,
    outer_length_m: float,
    tolerance_m: float,
) -> dict[str, Any]:
    """Evaluate the three independent open-exterior hard gates."""
    fraction = float(unintended_length_m / max(landward_length_m, 1.0))
    coverage = float(max(0.0, min(1.0, 1.0 - unintended_length_m / max(outer_length_m, 1.0))))
    absolute_pass = bool(unintended_length_m <= tolerance_m)
    fraction_pass = bool(fraction <= 0.001)
    coverage_pass = bool(coverage >= 0.999)
    return {
        "unintended_fraction": fraction,
        "intended_coverage": coverage,
        "length_gate": absolute_pass,
        "fraction_gate": fraction_pass,
        "coverage_gate": coverage_pass,
        "passed": bool(absolute_pass and fraction_pass and coverage_pass),
    }


def file_sha256(path: str | Path | None) -> str | None:
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


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_region_bpoly_arc_feedback(
    region_bpoly_json: str | Path,
    offshore_artifacts_json: str | Path,
    bdry_arc_gpkg: str | Path,
    coastline_gpkg: str | Path | None,
    loop_manifest_json: str | Path,
    run_dir: str | Path,
    source_manifest: dict[str, Any],
    *,
    frame_clip_policy: str,
    residual_boundary_policy: str = "strict-reject",
    frame_clip_tolerance_m: float | None,
    candidate_max_km: float = 100.0,
    adaptive_status: str = "pending",
    adaptive_failures: list[str] | None = None,
) -> dict[str, Any]:
    """Write a side-aware feedback artifact without changing the RegionBPoly."""
    region_path = Path(region_bpoly_json)
    offshore_path = Path(offshore_artifacts_json)
    package_path = Path(bdry_arc_gpkg)
    loop_path = Path(loop_manifest_json)
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    region = _read_json(region_path)
    offshore = _read_json(offshore_path)
    loop_manifest = _read_json(loop_path) if loop_path.exists() else {}
    points = _region_points(region)
    polygon_lonlat = Polygon(points)
    projection = local_utm_projection(tuple(float(v) for v in polygon_lonlat.bounds))
    polygon_xy = project_geometry(polygon_lonlat, projection)
    side_lines_xy = _side_lines(polygon_xy)

    frame_gdf = _read_layer(package_path, "frame_clip_boundary_arcs")
    open_gdf = _read_layer(package_path, "open_boundary_arc")
    wet_gdf = _read_layer(package_path, "wet_domain")
    frame_xy = _project_geometries(frame_gdf, projection)
    open_xy = _project_geometries(open_gdf, projection)
    open_union = unary_union(open_xy) if open_xy else GeometryCollection()
    open_length_for_bpoly_m = float(getattr(open_union, "length", 0.0))
    open_outside_bpoly_length_m = float(
        getattr(open_union.difference(polygon_xy), "length", 0.0)
        if not open_union.is_empty
        else 0.0
    )
    open_outside_bpoly_fraction = float(
        open_outside_bpoly_length_m / open_length_for_bpoly_m
        if open_length_for_bpoly_m > 0.0
        else 0.0
    )
    land_union_for_contract = _load_land_union(coastline_gpkg, projection)
    physical_coastline_for_contract = _load_physical_coastline_union(coastline_gpkg, projection)
    target_resolution_m = float(source_manifest.get("settings", {}).get("target_resolution_m", 250.0))
    tolerance_m = float(
        frame_clip_tolerance_m
        if frame_clip_tolerance_m is not None
        else max(250.0, 0.05 * target_resolution_m)
    )
    endpoint_tolerance_m = max(2.0 * tolerance_m, 0.10 * target_resolution_m)

    records: list[dict[str, Any]] = []
    geo_records: list[dict[str, Any]] = []
    for segment_id, line in enumerate(frame_xy):
        if line.is_empty or line.length <= 0.0:
            continue
        midpoint = line.interpolate(0.5, normalized=True)
        side_index = min(range(4), key=lambda idx: midpoint.distance(side_lines_xy[idx]))
        side = side_lines_xy[side_index]
        side_position = float(side.project(midpoint) / max(side.length, 1.0))
        open_overlap = 0.0
        open_distance = None
        if not open_union.is_empty:
            open_distance = float(line.distance(open_union))
            try:
                open_overlap = float(line.intersection(open_union.buffer(endpoint_tolerance_m)).length / line.length)
            except Exception:
                open_overlap = 0.0
        intentional = bool(open_overlap >= 0.95)
        line_lonlat = unproject_geometry(line, projection)
        landfall_tolerance_m = max(250.0, 0.25 * target_resolution_m)
        endpoint_distances = (
            [
                float(Point(line.coords[0]).distance(physical_coastline_for_contract)),
                float(Point(line.coords[-1]).distance(physical_coastline_for_contract)),
            ]
            if not physical_coastline_for_contract.is_empty
            else [None, None]
        )
        endpoint_buffers = Point(line.coords[0]).buffer(10.0).union(
            Point(line.coords[-1]).buffer(10.0)
        )
        line_interior = line.difference(endpoint_buffers)
        nonendpoint_land_crossing_m = float(
            line_interior.intersection(land_union_for_contract).length
            if not land_union_for_contract.is_empty
            else math.inf
        )
        nonendpoint_land_crossing_limit_m = 1.0e-6
        deterministic_solid_eligible = bool(
            line.is_simple
            and all(
                value is not None and value <= landfall_tolerance_m
                for value in endpoint_distances
            )
            and nonendpoint_land_crossing_m <= nonendpoint_land_crossing_limit_m
        )
        record = {
            "segment_id": int(segment_id),
            "side_index": int(side_index),
            "side_name": _side_name(region, side_index),
            "side_position": side_position,
            "length_m": float(line.length),
            "distance_to_open_boundary_m": open_distance,
            "open_boundary_overlap_fraction": open_overlap,
            "classification": "intentional_open_boundary" if intentional else "unintended_frame_clip",
            "role_status": "not_applicable" if intentional else "pending",
            "assigned_role": "intentional_open_boundary" if intentional else None,
            "solid_role_geometry": {
                "simple_nonbranching": bool(line.is_simple),
                "shoreline_bracketed_endpoints": bool(
                    all(
                        value is not None and value <= landfall_tolerance_m
                        for value in endpoint_distances
                    )
                ),
                "endpoint_distance_to_shoreline_m": endpoint_distances,
                "landfall_tolerance_m": float(landfall_tolerance_m),
                "nonendpoint_land_crossing_m": nonendpoint_land_crossing_m,
                "nonendpoint_land_crossing_limit_m": nonendpoint_land_crossing_limit_m,
                "wet_component_preserved": None,
                "eligible": deterministic_solid_eligible,
            },
            "boundary_completeness": {
                "status": "not_applicable" if intentional else "pending",
                "nontrivial": None,
                "same_side_nontrivial_count": None,
                "physical_water_continuation": None,
                "shoreline_tangent_angle_deg": [],
                "longitudinal_frame_supported_bar": None,
                "required_feature_conflicts": [],
                "automatic_trigger_reasons": [],
                "route": "intentional_open_boundary" if intentional else None,
            },
            "geometry_lonlat": [[float(x), float(y)] for x, y in line_lonlat.coords],
        }
        records.append(record)
        geo_records.append({
            **{
                key: value
                for key, value in record.items()
                if key not in {"geometry_lonlat", "solid_role_geometry"}
            },
            "geometry": line_lonlat,
        })

    unintended = [item for item in records if item["classification"] == "unintended_frame_clip"]
    nontrivial_limit_m = max(250.0, 0.05 * target_resolution_m)
    nontrivial_counts_by_side: dict[int, int] = {}
    for item in unintended:
        if float(item["length_m"]) > nontrivial_limit_m:
            side_index = int(item["side_index"])
            nontrivial_counts_by_side[side_index] = nontrivial_counts_by_side.get(side_index, 0) + 1

    automatic_truncation: list[dict[str, Any]] = []
    ambiguous_truncation: list[dict[str, Any]] = []
    for item, line in zip(
        unintended,
        [
            geometry
            for record, geometry in zip(records, frame_xy)
            if record.get("classification") == "unintended_frame_clip"
        ],
    ):
        side_index = int(item["side_index"])
        side = side_lines_xy[side_index]
        nontrivial = bool(float(item["length_m"]) > nontrivial_limit_m)
        continuation = _water_continuation_evidence(
            line,
            side,
            polygon_xy,
            land_union_for_contract,
            target_resolution_m,
        )
        tangent_window_m = max(500.0, 2.0 * target_resolution_m)
        tangent_angles = [
            value
            for value in (
                _shoreline_tangent_angle_deg(Point(line.coords[0]), line, physical_coastline_for_contract, tangent_window_m),
                _shoreline_tangent_angle_deg(Point(line.coords[-1]), line, physical_coastline_for_contract, tangent_window_m),
            )
            if value is not None
        ]
        longitudinal = bool(
            nontrivial
            and continuation.get("continues_outward") is True
            and tangent_angles
            and sum(tangent_angles) / len(tangent_angles) <= 35.0
        )
        feature_conflicts = _required_feature_conflicts(
            region,
            line,
            projection,
            target_resolution_m,
        )
        reasons: list[str] = []
        if nontrivial and nontrivial_counts_by_side.get(side_index, 0) >= 2:
            reasons.append("repeated_nontrivial_same_side_cut")
        if nontrivial and feature_conflicts and continuation.get("continues_outward") is True:
            reasons.append("required_feature_or_connectivity_truncation")
        if longitudinal:
            reasons.append("longitudinal_frame_supported_bar")
        completeness = item["boundary_completeness"]
        completeness.update(
            {
                "status": "adjust_bpoly" if reasons else "pending",
                "nontrivial": nontrivial,
                "nontrivial_limit_m": float(nontrivial_limit_m),
                "same_side_nontrivial_count": int(nontrivial_counts_by_side.get(side_index, 0)),
                "physical_water_continuation": continuation,
                "shoreline_tangent_angle_deg": tangent_angles,
                "longitudinal_frame_supported_bar": longitudinal,
                "required_feature_conflicts": feature_conflicts,
                "automatic_trigger_reasons": reasons,
                "route": "adjust_bpoly" if reasons else None,
            }
        )
        if reasons:
            automatic_truncation.append(item)
        elif nontrivial and continuation.get("continues_outward") is True:
            # A clearly transverse single closure may proceed to the ordinary
            # residual-role decision. Everything else is bound to a Codex map
            # decision rather than silently becoming a solid boundary.
            transverse = bool(tangent_angles and min(tangent_angles) >= 45.0)
            if transverse:
                completeness.update(
                    {
                        "status": "role_classification_ready",
                        "route": "retain_for_role_classification",
                        "automatic_clear_reason": "isolated_transverse_closure",
                    }
                )
            else:
                completeness["status"] = "agent_decision_required"
                ambiguous_truncation.append(item)
        else:
            completeness.update(
                {
                    "status": "role_classification_ready",
                    "route": "retain_for_role_classification",
                    "automatic_clear_reason": "numerical_or_no_outward_water_continuation",
                }
            )
    unintended_length_m = float(sum(item["length_m"] for item in unintended))
    wet_polygon_xy = _largest_projected_polygon(wet_gdf, projection)
    outer_length_m = float(wet_polygon_xy.exterior.length) if wet_polygon_xy is not None else 0.0
    open_length_m = float(getattr(open_union, "length", 0.0))
    landward_length_m = max(outer_length_m - open_length_m, 1.0)
    metric_result = evaluate_open_exterior_metrics(
        unintended_length_m, landward_length_m, outer_length_m, tolerance_m
    )
    unintended_fraction = metric_result["unintended_fraction"]
    intended_coverage = metric_result["intended_coverage"]
    length_gate = metric_result["length_gate"]
    fraction_gate = metric_result["fraction_gate"]
    coverage_gate = metric_result["coverage_gate"]
    diagnostic_pass = metric_result["passed"]
    gate_enabled = frame_clip_policy == "reject-unintended" and str(
        source_manifest.get("inputs", {}).get("coastline_source", "")
    ) == "gshhs"

    loop_failures = list(loop_manifest.get("failure_taxonomy", []))
    nonclip_loop_failures = [failure for failure in loop_failures if failure not in CLIP_FAILURES]
    arc_land_length_m = float(source_manifest.get("wet_domain", {}).get("arc_land_intersection_length_m", 0.0) or 0.0)
    wet_component_count = _wet_component_count(wet_gdf)
    for item in unintended:
        geometry = item["solid_role_geometry"]
        geometry["wet_component_preserved"] = wet_component_count == 1
        geometry["eligible"] = bool(geometry["eligible"] and wet_component_count == 1)
    expected_obc_count = int(
        region.get("expected_obc_count", offshore.get("expected_obc_count", 0 if region.get("boundary_policy") == "no_open_boundary" else 1))
    )
    delivered_obc_count = sum(1 for geom in open_gdf.geometry if geom is not None and not geom.is_empty)
    domain_type = str(region.get("domain_type", "")).lower()
    role_contract_enabled = bool(
        residual_boundary_policy == "solid-default"
        and frame_clip_policy != "report-only"
        and domain_type not in {"lake", "island", "archipelago"}
    )
    structural_failures: list[str] = []
    coastline_coverage = source_manifest.get("coastline_source_coverage") or {}
    coastline_coverage_required = str(
        source_manifest.get("inputs", {}).get("coastline_source", "")
    ) == "gshhs"
    if coastline_coverage_required and coastline_coverage.get("downstream_eligible") is not True:
        structural_failures.extend(
            coastline_coverage.get("failure_taxonomy")
            or ["coastline_source_footprint_incomplete"]
        )
    if wet_component_count != 1:
        structural_failures.append("wet_component_count_not_one")
    if delivered_obc_count != expected_obc_count and not (
        role_contract_enabled
        and delivered_obc_count < expected_obc_count
        and expected_obc_count - delivered_obc_count <= len(unintended)
    ):
        structural_failures.append("unexpected_open_boundary_count")
    if arc_land_length_m > 1.0e-6:
        structural_failures.append("open_boundary_intersects_land")
    coastal_single_obc = bool(
        expected_obc_count == 1
        and domain_type not in {"lake", "island", "archipelago"}
        and "obc_placement_policy" in source_manifest.get("settings", {})
    )
    if coastal_single_obc:
        if delivered_obc_count != 1 or len(open_xy) != 1:
            structural_failures.append("open_boundary_not_one_simple_arc")
        elif not bool(getattr(open_xy[0], "is_simple", False)):
            structural_failures.append("open_boundary_self_intersects_or_branches")
        land_boundary_for_contract = physical_coastline_for_contract
        if len(open_xy) == 1 and not land_boundary_for_contract.is_empty:
            landfall_tolerance_m = max(250.0, 0.25 * target_resolution_m)
            endpoint_hits = sum(
                Point(coord).distance(land_boundary_for_contract) <= landfall_tolerance_m
                for coord in (open_xy[0].coords[0], open_xy[0].coords[-1])
            )
            if endpoint_hits != 2:
                structural_failures.append("open_boundary_requires_exactly_two_coastline_landfalls")
    structural_failures.extend(nonclip_loop_failures)

    recommendation_sources = (
        automatic_truncation
        if automatic_truncation
        else ambiguous_truncation
        if ambiguous_truncation
        else unintended
    )
    recommendations = _candidate_recommendations(
        polygon_xy,
        region,
        recommendation_sources,
        land_union_for_contract,
        tolerance_m=tolerance_m,
        candidate_max_km=float(candidate_max_km),
    )
    if frame_clip_policy == "report-only":
        status = "input_needs_review"
    elif structural_failures:
        status = "input_needs_review"
    elif automatic_truncation:
        status = "adjust_bpoly" if recommendations else "input_needs_review"
    elif ambiguous_truncation:
        status = "boundary_completeness_decision_required"
    elif role_contract_enabled and unintended and not structural_failures:
        status = "assign_boundary_roles"
    elif not diagnostic_pass:
        status = "adjust_bpoly" if recommendations and not structural_failures else "input_needs_review"
    elif adaptive_status in {"needs_review", "failed"}:
        status = "input_needs_review"
    else:
        status = "pass"

    side_summaries = []
    for side_index in range(4):
        selected = [item for item in unintended if item["side_index"] == side_index]
        side_summaries.append(
            {
                "side_index": side_index,
                "side_name": _side_name(region, side_index),
                "unintended_segment_count": len(selected),
                "unintended_frame_clip_length_m": float(sum(item["length_m"] for item in selected)),
                "side_position_min": min((item["side_position"] for item in selected), default=None),
                "side_position_max": max((item["side_position"] for item in selected), default=None),
            }
        )

    completeness_failures = (
        ["region_bpoly_boundary_truncation_detected"]
        if automatic_truncation
        else ["region_bpoly_boundary_completeness_decision_required"]
        if ambiguous_truncation
        else []
    )
    feedback_failures = (
        list(dict.fromkeys(
            structural_failures
            + completeness_failures
            + (["residual_boundary_role_decision_required"] if unintended and not automatic_truncation and not ambiguous_truncation else [])
        ))
        if role_contract_enabled
        else _feedback_failures(
            diagnostic_pass,
            structural_failures,
            adaptive_status,
            adaptive_failures or [],
        )
    )
    feedback = {
        "schema_version": "region_bpoly_arc_feedback_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "diagnostic_status": "pass" if diagnostic_pass else "fail",
        "failure_taxonomy": feedback_failures,
        "policy": {
            "frame_clip_policy": frame_clip_policy,
            "residual_boundary_policy": residual_boundary_policy,
            "role_contract_enabled": role_contract_enabled,
            "gate_enabled": gate_enabled,
            "frame_clip_tolerance_m": tolerance_m,
            "unintended_fraction_threshold": 0.001,
            "intended_exterior_coverage_threshold": 0.999,
            "candidate_max_km": float(candidate_max_km),
            "semantics_policy": "geometry_only_no_feature_inference",
            "boundary_completeness_policy": "hybrid",
            "loop_scope": "land_boundary_only",
            "offshore_obc_bpoly_containment_required": False,
        },
        "inputs": {
            "region_bpoly_json": str(region_path),
            "offshore_artifacts_json": str(offshore_path),
            "bdry_arc_gpkg": str(package_path),
            "coastline_gpkg": str(coastline_gpkg) if coastline_gpkg else None,
            "model_boundary_loop_manifest": str(loop_path),
        },
        "input_sha256": {
            "region_bpoly_json": file_sha256(region_path),
            "offshore_artifacts_json": file_sha256(offshore_path),
            "bdry_arc_gpkg": file_sha256(package_path),
            "coastline_gpkg": file_sha256(coastline_gpkg),
            "model_boundary_loop_manifest": file_sha256(loop_path),
            "coastline_source_coverage": file_sha256(coastline_coverage.get("contract_path")),
        },
        "gshhs": {
            "source_version": source_manifest.get("inputs", {}).get("coastline_load", {}).get("source_version", "GSHHG 2.3.7"),
            "requested_resolution": source_manifest.get("settings", {}).get("gshhs_resolution"),
            "selected_resolution": source_manifest.get("inputs", {}).get("coastline_load", {}).get("gshhs_selected_resolution"),
            "levels": source_manifest.get("settings", {}).get("gshhs_levels"),
        },
        "coastline_source_coverage": coastline_coverage,
        "metrics": {
            "target_resolution_m": target_resolution_m,
            "wet_component_count": int(wet_component_count),
            "expected_obc_count": int(expected_obc_count),
            "delivered_obc_count": int(delivered_obc_count),
            "open_boundary_land_intersection_m": arc_land_length_m,
            "open_boundary_exterior_overlap_fraction": float(loop_manifest.get("qa", {}).get("open_boundary_exterior_overlap_fraction", 0.0) or 0.0),
            "outer_boundary_length_m": outer_length_m,
            "open_boundary_length_m": open_length_m,
            "open_boundary_outside_region_bpoly_length_m": open_outside_bpoly_length_m,
            "open_boundary_outside_region_bpoly_fraction": open_outside_bpoly_fraction,
            "landward_boundary_length_m": landward_length_m,
            "unintended_frame_clip_length_m": unintended_length_m,
            "unintended_frame_clip_fraction": unintended_fraction,
            "intended_land_open_exterior_coverage_fraction": intended_coverage,
            "length_gate_pass": length_gate,
            "fraction_gate_pass": fraction_gate,
            "coverage_gate_pass": coverage_gate,
            "model_loop_status": loop_manifest.get("final_status"),
            "adaptive_status": adaptive_status,
        },
        "frame_clip_segments": records,
        "boundary_completeness": {
            "schema_version": "fvcom_boundary_completeness_assessment_v1",
            "policy": "hybrid",
            "status": (
                "adjust_bpoly"
                if automatic_truncation
                else "agent_decision_required"
                if ambiguous_truncation
                else "pass"
            ),
            "nontrivial_limit_m": float(nontrivial_limit_m),
            "automatic_adjust_segment_ids": [int(item["segment_id"]) for item in automatic_truncation],
            "agent_decision_segment_ids": [int(item["segment_id"]) for item in ambiguous_truncation],
            "role_ready_segment_ids": [
                int(item["segment_id"])
                for item in unintended
                if item["boundary_completeness"].get("route") == "retain_for_role_classification"
            ],
            "decision_required": bool(ambiguous_truncation),
            "decision_status": "pending" if ambiguous_truncation else "not_required",
        },
        "side_summaries": side_summaries,
        "candidate_recommendations": recommendations,
        "adaptive": {
            "status": adaptive_status,
            "failure_taxonomy": list(adaptive_failures or []),
        },
    }

    segments_path = output_dir / "segments.geojson"
    if geo_records:
        gpd.GeoDataFrame(geo_records, geometry="geometry", crs="EPSG:4326").to_file(segments_path, driver="GeoJSON")
    else:
        segments_path.write_text('{"type":"FeatureCollection","features":[]}\n', encoding="utf-8")
    map_path = output_dir / "feedback_map.png"
    whole_map_context = _plot_feedback(
        map_path,
        polygon_lonlat,
        wet_gdf,
        open_gdf,
        frame_gdf,
        status,
        coastline_gpkg,
        region,
    )
    component_maps = _plot_residual_component_maps(
        output_dir,
        polygon_lonlat,
        wet_gdf,
        open_gdf,
        records,
        coastline_gpkg,
        region,
    )
    geography_usable = bool(
        whole_map_context.get("geography_usable") is True
        and all(record.get("geography_usable") is True for record in component_maps.values())
    )
    if not geography_usable:
        feedback["status"] = "input_needs_review"
        feedback["boundary_completeness"]["status"] = "needs_review"
        feedback["failure_taxonomy"] = list(dict.fromkeys(
            feedback.get("failure_taxonomy", [])
            + ["boundary_completeness_map_background_unusable"]
        ))
    map_failures = [] if geography_usable else ["boundary_completeness_map_background_unusable"]
    feedback["outputs"] = {
        "segments_geojson": str(segments_path),
        "feedback_map": str(map_path),
        "residual_component_maps": component_maps,
        "whole_map_context": whole_map_context,
    }
    feedback_path = output_dir / "region_bpoly_arc_feedback_v2.json"
    feedback_path.write_text(json.dumps(feedback, indent=2), encoding="utf-8")
    feedback["outputs"]["feedback_json"] = str(feedback_path)
    feedback_path.write_text(json.dumps(feedback, indent=2), encoding="utf-8")

    placement_family = str(
        source_manifest.get("wet_domain", {}).get("obc_placement_family")
        or source_manifest.get("settings", {}).get("obc_placement_policy")
        or "legacy-unspecified"
    )
    contract_path = output_dir / "open_exterior_contract.json"
    decision_path = output_dir / "open_exterior_agent_decision.json"
    report_only = frame_clip_policy == "report-only"
    coastline_coverage_pass = bool(
        not coastline_coverage_required
        or coastline_coverage.get("downstream_eligible") is True
    )
    metric_gate_pass = bool(diagnostic_pass and not structural_failures and coastline_coverage_pass)
    contract_schema = (
        "fvcom_open_exterior_contract_v3"
        if role_contract_enabled
        else "fvcom_open_exterior_contract_v1"
    )
    contract_failures = (
        list(dict.fromkeys(
            structural_failures
            + completeness_failures
            + map_failures
            + (["residual_boundary_role_decision_required"] if unintended else [])
            + ["open_exterior_agent_decision_required"]
        ))
        if role_contract_enabled
        else list(feedback["failure_taxonomy"])
    )
    contract = {
        "schema_version": contract_schema,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "final_status": "needs_review",
        "downstream_eligible": False,
        "obc_placement_policy": str(source_manifest.get("settings", {}).get("obc_placement_policy", "offshore-first")),
        "obc_placement_family": placement_family,
        "report_only": report_only,
        "residual_boundary_policy": {
            "mode": residual_boundary_policy,
            "default_role": "solid_lagoon_closure" if role_contract_enabled else None,
            "station_is_eligibility_not_automatic_obc": True,
            "requested_obc_count_controls_secondary_obc": True,
            "allowed_roles": sorted(RESIDUAL_BOUNDARY_ROLES),
        },
        "raw_residual_metrics": {
            "absolute_residual_length_m": unintended_length_m,
            "residual_fraction": unintended_fraction,
            "coastline_plus_obc_exterior_coverage": intended_coverage,
        },
        "boundary_lengths": {
            "outer_boundary_length_m": outer_length_m,
            "open_boundary_length_m": open_length_m,
            "open_boundary_outside_region_bpoly_length_m": open_outside_bpoly_length_m,
            "open_boundary_outside_region_bpoly_fraction": open_outside_bpoly_fraction,
            "landward_boundary_length_m": landward_length_m,
        },
        "hard_metrics": {
            "absolute_residual_length_m": unintended_length_m,
            "absolute_limit_m": tolerance_m,
            "absolute_gate_pass": length_gate,
            "residual_fraction": unintended_fraction,
            "fraction_limit": 0.001,
            "fraction_gate_pass": fraction_gate,
            "coastline_plus_obc_exterior_coverage": intended_coverage,
            "coverage_minimum": 0.999,
            "coverage_gate_pass": coverage_gate,
            "all_independent_metric_gates_pass": metric_gate_pass,
            "coastline_source_coverage_gate_pass": coastline_coverage_pass,
            "metric_subject": "unassigned_residual" if role_contract_enabled else "raw_residual",
        },
        "obc_geometry": {
            "expected_count": int(expected_obc_count),
            "delivered_count": int(delivered_obc_count),
            "simple_nonbranching": bool(all(getattr(geom, "is_simple", False) for geom in open_xy)),
            "nonendpoint_land_crossing_m": arc_land_length_m,
            "exterior_overlap_fraction": float(loop_manifest.get("qa", {}).get("open_boundary_exterior_overlap_fraction", 0.0) or 0.0),
        },
        "residual_components": records,
        "residual_role_summary": {
            "pending_count": int(len(unintended)) if role_contract_enabled else 0,
            "solid_lagoon_closure_count": 0,
            "secondary_tidal_obc_count": 0,
            "invalid_geometry_count": 0,
            "unassigned_residual_length_m": unintended_length_m if role_contract_enabled else 0.0,
            "assigned_solid_length_m": 0.0,
            "assigned_secondary_obc_length_m": 0.0,
        },
        "boundary_completeness": feedback["boundary_completeness"],
        "per_side": side_summaries,
        "coastline_source_coverage_required": coastline_coverage_required,
        "coastline_source_coverage": coastline_coverage,
        "source_hashes": dict(feedback["input_sha256"]),
        "map": {"path": str(map_path), "sha256": file_sha256(map_path), **whole_map_context},
        "component_maps": component_maps,
        "station_screen": {
            "required_for_secondary_tidal_obc": True,
            "status": "not_run",
            "path": None,
            "sha256": None,
        },
        "agent_decision": {
            "required": True,
            "status": "pending",
            "path": str(decision_path),
        },
        "failure_taxonomy": list(dict.fromkeys(
            contract_failures
            + (["diagnostic_only_report_only_policy"] if report_only else [])
            + (["open_exterior_agent_decision_required"] if metric_gate_pass and not report_only and not role_contract_enabled else [])
        )),
    }
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    pending_decision = {
        "schema_version": (
            "open_exterior_agent_decision_v3"
            if role_contract_enabled
            else "open_exterior_agent_decision_v1"
        ),
        "status": "pending",
        "decision_actor": {"kind": "codex_agent"},
        "assessed_contract_sha256": file_sha256(contract_path),
        "inspected_map_sha256": file_sha256(map_path),
        "bound_source_hashes": contract["source_hashes"],
        "inspected_coastline_coverage_map_sha256": (coastline_coverage.get("maps", {}).get("whole_domain", {}) or {}).get("sha256"),
        "inspected_coastline_coverage_zoom_sha256": (coastline_coverage.get("maps", {}).get("source_edge_zoom", {}) or {}).get("sha256"),
        "rationale": None,
        "residual_roles": [
            {
                "segment_id": int(item["segment_id"]),
                "role": None,
                "component_map_sha256": component_maps.get(str(item["segment_id"]), {}).get("sha256"),
                "no_artificial_bar": None,
                "no_protected_feature_conflict": None,
                "rationale": None,
            }
            for item in unintended
        ],
    }
    decision_path.write_text(json.dumps(pending_decision, indent=2), encoding="utf-8")
    completeness_decision_path = output_dir / "region_bpoly_boundary_completeness_decision_v1.json"
    pending_completeness_decision = {
        "schema_version": "region_bpoly_boundary_completeness_decision_v1",
        "status": "pending" if ambiguous_truncation else "not_required",
        "decision_actor": {"kind": "codex_agent"},
        "assessed_feedback_core_sha256": canonical_sha256(
            {
                "boundary_completeness": feedback["boundary_completeness"],
                "source_hashes": feedback["input_sha256"],
                "whole_map_sha256": file_sha256(map_path),
                "component_map_sha256": {
                    key: value.get("sha256") for key, value in component_maps.items()
                },
            }
        ),
        "bound_source_hashes": feedback["input_sha256"],
        "inspected_map_sha256": file_sha256(map_path),
        "component_decisions": [
            {
                "segment_id": int(item["segment_id"]),
                "route": None,
                "component_map_sha256": component_maps.get(str(item["segment_id"]), {}).get("sha256"),
                "rationale": None,
            }
            for item in ambiguous_truncation
        ],
    }
    completeness_decision_path.write_text(
        json.dumps(pending_completeness_decision, indent=2), encoding="utf-8"
    )
    feedback["open_exterior_contract"] = contract
    feedback["outputs"].update({
        "open_exterior_contract": str(contract_path),
        "open_exterior_agent_decision": str(decision_path),
        "open_exterior_review_map": str(map_path),
        "boundary_completeness_decision": str(completeness_decision_path),
    })
    feedback_path.write_text(json.dumps(feedback, indent=2), encoding="utf-8")
    return feedback


def _feedback_failures(
    diagnostic_pass: bool,
    structural_failures: list[str],
    adaptive_status: str,
    adaptive_failures: list[str],
) -> list[str]:
    failures = list(structural_failures)
    if not diagnostic_pass:
        failures.extend(
            [
                "unintended_frame_clip_nontrivial",
                "gshhs_coastline_incomplete_on_landward_boundary",
            ]
        )
    if adaptive_status in {"needs_review", "failed"}:
        failures.extend(adaptive_failures or ["adaptive_boundary_resolution_needs_review"])
    return list(dict.fromkeys(failures))


def _candidate_recommendations(
    polygon_xy: Polygon,
    region: dict[str, Any],
    unintended: list[dict[str, Any]],
    land_union,
    *,
    tolerance_m: float,
    candidate_max_km: float,
) -> list[dict[str, Any]]:
    by_side: dict[int, list[dict[str, Any]]] = {}
    for item in unintended:
        by_side.setdefault(int(item["side_index"]), []).append(item)
    if not by_side:
        return []
    recommendations: list[dict[str, Any]] = []
    coords = list(polygon_xy.exterior.coords)[:-1]
    centroid = polygon_xy.centroid
    for side_index, records in sorted(by_side.items(), key=lambda item: -sum(r["length_m"] for r in item[1])):
        p0 = coords[side_index]
        p1 = coords[(side_index + 1) % 4]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        norm = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / norm, dx / norm
        midpoint = Point(0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]))
        if nx * (midpoint.x - centroid.x) + ny * (midpoint.y - centroid.y) < 0.0:
            nx, ny = -nx, -ny
        mean_position = sum(record["side_position"] * record["length_m"] for record in records) / max(
            sum(record["length_m"] for record in records), 1.0
        )
        profile_specs = [("full_edge", 1.0, 1.0), ("start_taper", 1.0, 0.25), ("end_taper", 0.25, 1.0)]
        for profile, weight0, weight1 in profile_specs:
            distance_km, predicted_uncovered_m = _scan_outward_distance(
                p0,
                p1,
                nx,
                ny,
                weight0,
                weight1,
                land_union,
                tolerance_m=tolerance_m,
                candidate_max_km=candidate_max_km,
            )
            if distance_km is None:
                continue
            deltas = {
                str(side_index): [float(nx * distance_km * weight0), float(ny * distance_km * weight0)],
                str((side_index + 1) % 4): [float(nx * distance_km * weight1), float(ny * distance_km * weight1)],
            }
            recommendations.append(
                {
                    "candidate_id": f"side_{side_index}_{profile}_{int(round(distance_km * 1000.0)):06d}m",
                    "operation": "reshape",
                    "side_index": int(side_index),
                    "side_name": _side_name(region, side_index),
                    "profile": profile,
                    "displacement_km": float(distance_km),
                    "outward_unit_east_north": [float(nx), float(ny)],
                    "vertex_delta_km": deltas,
                    "source_unintended_frame_clip_length_m": float(sum(r["length_m"] for r in records)),
                    "source_segment_ids": [int(r["segment_id"]) for r in records],
                    "source_side_position_mean": float(mean_position),
                    "predicted_uncovered_side_length_m": float(predicted_uncovered_m),
                    "distance_search": {
                        "coarse_increment_km": 1.0,
                        "refinement_increment_km": 0.25,
                        "maximum_km": float(candidate_max_km),
                    },
                    "semantic_feature_changes": [],
                }
            )
    recommendations.sort(
        key=lambda item: (
            float(item.get("predicted_uncovered_side_length_m", float("inf"))),
            0 if item.get("profile") == "full_edge" else 1,
            float(item.get("displacement_km", float("inf"))),
        )
    )
    return recommendations[:12]


def _scan_outward_distance(
    p0,
    p1,
    nx: float,
    ny: float,
    weight0: float,
    weight1: float,
    land_union,
    *,
    tolerance_m: float,
    candidate_max_km: float,
) -> tuple[float | None, float]:
    if land_union is None or land_union.is_empty:
        return None, float("inf")

    def uncovered(distance_km: float) -> float:
        q0 = (p0[0] + nx * distance_km * 1000.0 * weight0, p0[1] + ny * distance_km * 1000.0 * weight0)
        q1 = (p1[0] + nx * distance_km * 1000.0 * weight1, p1[1] + ny * distance_km * 1000.0 * weight1)
        line = LineString([q0, q1])
        try:
            return float(line.difference(land_union).length)
        except Exception:
            return float(line.length)

    coarse = []
    distance = 1.0
    while distance <= candidate_max_km + 1.0e-9:
        value = uncovered(distance)
        coarse.append((distance, value))
        if value <= tolerance_m:
            break
        distance += 1.0
    if not coarse:
        return None, float("inf")
    coarse_choice = min(coarse, key=lambda item: (item[1] > tolerance_m, item[1], item[0]))
    if coarse_choice[1] <= tolerance_m:
        lower = max(0.25, coarse_choice[0] - 1.0)
        refined = []
        distance = lower
        while distance <= coarse_choice[0] + 1.0e-9:
            refined.append((distance, uncovered(distance)))
            distance += 0.25
        passing = [item for item in refined if item[1] <= tolerance_m]
        if passing:
            return min(passing, key=lambda item: item[0])
    return coarse_choice


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _region_points(region: dict[str, Any]) -> list[list[float]]:
    points = region.get("polygon_lonlat") or region.get("region_bpoly", {}).get("polygon_lonlat")
    if not points:
        raise ValueError("RegionBPoly feedback requires polygon_lonlat")
    points = [list(map(float, point)) for point in points]
    if len(points) == 5 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) != 4:
        raise ValueError("RegionBPoly feedback requires exactly four vertices")
    return points


def _side_name(region: dict[str, Any], side_index: int) -> str:
    labels = region.get("region_bpoly", {}).get("edge_labels") or region.get("edge_labels") or []
    return str(labels[side_index]) if len(labels) == 4 else f"side_{side_index}"


def _side_lines(polygon_xy: Polygon) -> list[LineString]:
    coords = list(polygon_xy.exterior.coords)[:-1]
    return [LineString([coords[index], coords[(index + 1) % 4]]) for index in range(4)]


def _read_layer(path: Path, layer: str) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(path, layer=layer)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    usable = [geometry is not None and not geometry.is_empty for geometry in gdf.geometry]
    if not any(usable):
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gdf.loc[usable].reset_index(drop=True)


def _load_land_union(path: str | Path | None, projection):
    if not path:
        return GeometryCollection()
    source = Path(path)
    if not source.exists():
        return GeometryCollection()
    land = _read_layer(source, "land_polygons")
    if land.empty:
        return GeometryCollection()
    polygons = []
    for geometry in land.geometry:
        if geometry is None or geometry.is_empty:
            continue
        projected = project_geometry(geometry, projection)
        if not projected.is_empty:
            polygons.append(projected)
    return unary_union(polygons) if polygons else GeometryCollection()


def _load_physical_coastline_union(path: str | Path | None, projection):
    if not path:
        return GeometryCollection()
    source = Path(path)
    if not source.exists():
        return GeometryCollection()
    coastline = _read_layer(source, "coastline_lines")
    if coastline.empty:
        return GeometryCollection()
    lines = _project_geometries(coastline, projection)
    return unary_union(lines) if lines else GeometryCollection()


def _project_geometries(gdf: gpd.GeoDataFrame, projection) -> list[LineString]:
    lines: list[LineString] = []
    for geometry in gdf.geometry:
        if geometry is None or geometry.is_empty:
            continue
        projected = project_geometry(geometry, projection)
        if isinstance(projected, LineString):
            lines.append(projected)
        elif hasattr(projected, "geoms"):
            lines.extend(part for part in projected.geoms if isinstance(part, LineString) and not part.is_empty)
    return lines


def _largest_projected_polygon(gdf: gpd.GeoDataFrame, projection) -> Polygon | None:
    polygons: list[Polygon] = []
    for geometry in gdf.geometry:
        if geometry is None or geometry.is_empty:
            continue
        projected = project_geometry(geometry, projection)
        if isinstance(projected, Polygon):
            polygons.append(projected)
        elif hasattr(projected, "geoms"):
            polygons.extend(part for part in projected.geoms if isinstance(part, Polygon) and not part.is_empty)
    return max(polygons, key=lambda item: item.area) if polygons else None


def _wet_component_count(gdf: gpd.GeoDataFrame) -> int:
    count = 0
    for geometry in gdf.geometry:
        if geometry is None or geometry.is_empty:
            continue
        if isinstance(geometry, Polygon):
            count += 1
        elif hasattr(geometry, "geoms"):
            count += sum(1 for part in geometry.geoms if isinstance(part, Polygon) and not part.is_empty)
    return count


def _plot_feedback(
    path: Path,
    region: Polygon,
    wet_gdf: gpd.GeoDataFrame,
    open_gdf: gpd.GeoDataFrame,
    frame_gdf: gpd.GeoDataFrame,
    status: str,
    coastline_gpkg: str | Path | None,
    region_doc: dict[str, Any],
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(11, 9))
    land = _read_layer(Path(coastline_gpkg), "land_polygons") if coastline_gpkg else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    coastline = _read_layer(Path(coastline_gpkg), "coastline_lines") if coastline_gpkg else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    source_frame = _read_layer(Path(coastline_gpkg), "source_frame") if coastline_gpkg else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    source_footprint = _read_layer(Path(coastline_gpkg), "source_footprint") if coastline_gpkg else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if not source_footprint.empty:
        source_footprint.plot(
            ax=ax,
            facecolor="#dff3ff",
            edgecolor="none",
            alpha=0.55,
            label="expanded source water context",
        )
    if not land.empty:
        land.plot(ax=ax, facecolor="#d8d1bd", edgecolor="#31572c", linewidth=0.45, alpha=0.75, label="expanded GSHHS land")
    if not coastline.empty:
        coastline.plot(ax=ax, color="#31572c", linewidth=0.7, alpha=0.9, label="physical coastline")
    if not source_frame.empty:
        source_frame.plot(ax=ax, color="#4361ee", linewidth=0.8, linestyle=":", label="GSHHS source frame")
    if not wet_gdf.empty:
        wet_gdf.plot(ax=ax, facecolor="#7cc6fe", edgecolor="#0b4f6c", linewidth=0.8, alpha=0.25)
    gpd.GeoSeries([region], crs="EPSG:4326").boundary.plot(ax=ax, color="#111111", linewidth=1.2, linestyle="--")
    if not open_gdf.empty:
        open_gdf.plot(ax=ax, color="#d00000", linewidth=2.2, label="open boundary")
    if not frame_gdf.empty:
        frame_gdf.plot(ax=ax, color="#ff8c00", linewidth=3.0, label="residual frame clip")
    ax.set_title(f"RegionBPoly arc feedback - {status}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    if not open_gdf.empty or not frame_gdf.empty:
        ax.legend(loc="best")
    _plot_required_features(ax, region_doc)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "geography_usable": bool(not land.empty and not coastline.empty),
        "background_kind": "expanded_gshhs_land_and_physical_coastline",
        "land_feature_count": int(len(land)),
        "coastline_feature_count": int(len(coastline)),
        "source_frame_feature_count": int(len(source_frame)),
    }


def _plot_required_features(ax, region_doc: dict[str, Any]) -> None:
    points = []
    labels = []
    for feature in (region_doc.get("target_region_features") or {}).get("features", []):
        geometry = feature.get("geometry")
        if not feature.get("required", False) or feature.get("type") != "point":
            continue
        if not isinstance(geometry, list) or len(geometry) < 2:
            continue
        points.append(Point(float(geometry[0]), float(geometry[1])))
        labels.append(str(feature.get("label") or feature.get("id") or "required feature"))
    if not points:
        return
    gpd.GeoSeries(points, crs="EPSG:4326").plot(
        ax=ax, color="#7209b7", marker="*", markersize=55, zorder=7, label="required feature"
    )
    for point, label in zip(points, labels):
        ax.annotate(label, (point.x, point.y), xytext=(3, 3), textcoords="offset points", fontsize=6, color="#4a148c")


def _plot_residual_component_maps(
    output_dir: Path,
    region: Polygon,
    wet_gdf: gpd.GeoDataFrame,
    open_gdf: gpd.GeoDataFrame,
    records: list[dict[str, Any]],
    coastline_gpkg: str | Path | None,
    region_doc: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Write one geographic, hash-bound zoom map per raw frame-water component."""
    land = _read_layer(Path(coastline_gpkg), "land_polygons") if coastline_gpkg else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    coastline = _read_layer(Path(coastline_gpkg), "coastline_lines") if coastline_gpkg else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    source_frame = _read_layer(Path(coastline_gpkg), "source_frame") if coastline_gpkg else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    source_footprint = _read_layer(Path(coastline_gpkg), "source_footprint") if coastline_gpkg else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    maps: dict[str, dict[str, Any]] = {}
    for item in records:
        if item.get("classification") != "unintended_frame_clip":
            continue
        segment_id = int(item["segment_id"])
        line = LineString(item["geometry_lonlat"])
        minx, miny, maxx, maxy = line.bounds
        span = max(maxx - minx, maxy - miny, 0.005)
        margin = max(0.02, 5.0 * span)
        extent = (minx - margin, miny - margin, maxx + margin, maxy + margin)
        fig, ax = plt.subplots(figsize=(9, 8))
        if not source_footprint.empty:
            source_footprint.plot(
                ax=ax,
                facecolor="#dff3ff",
                edgecolor="none",
                alpha=0.55,
                label="outside-frame wet continuation context",
            )
        if not land.empty:
            clipped = land.cx[extent[0] : extent[2], extent[1] : extent[3]]
            if not clipped.empty:
                clipped.plot(ax=ax, facecolor="#d8d1bd", edgecolor="#31572c", linewidth=0.8, alpha=0.9, label="GSHHS land")
        if not coastline.empty:
            clipped_coast = coastline.cx[extent[0] : extent[2], extent[1] : extent[3]]
            if not clipped_coast.empty:
                clipped_coast.plot(ax=ax, color="#31572c", linewidth=1.0, label="physical coastline")
        if not source_frame.empty:
            source_frame.plot(ax=ax, color="#4361ee", linewidth=0.7, linestyle=":", label="GSHHS source frame")
        if not wet_gdf.empty:
            wet_gdf.plot(ax=ax, facecolor="#8ecae6", edgecolor="#0b4f6c", linewidth=0.7, alpha=0.35, label="retained wet domain")
        gpd.GeoSeries([region], crs="EPSG:4326").boundary.plot(ax=ax, color="#111111", linewidth=1.2, linestyle="--", label="RegionBPoly frame")
        if not open_gdf.empty:
            open_gdf.plot(ax=ax, color="#d00000", linewidth=2.2, label="delivered OBC")
        gpd.GeoSeries([line], crs="EPSG:4326").plot(ax=ax, color="#ff8c00", linewidth=4.0, label="residual water segment")
        endpoints = [Point(line.coords[0]), Point(line.coords[-1])]
        gpd.GeoSeries(endpoints, crs="EPSG:4326").plot(ax=ax, color="#6a00f4", markersize=45, zorder=5, label="closure endpoints")
        _plot_required_features(ax, region_doc)
        ax.set_xlim(extent[0], extent[2])
        ax.set_ylim(extent[1], extent[3])
        ax.set_title(
            f"Residual component {segment_id} | {float(item['length_m']):.1f} m\n"
            "Classify as solid closure, secondary tidal OBC, or invalid geometry"
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)
        fig.tight_layout()
        map_path = output_dir / f"residual_component_{segment_id:03d}_role_map.png"
        fig.savefig(map_path, dpi=200)
        plt.close(fig)
        maps[str(segment_id)] = {
            "path": str(map_path),
            "sha256": file_sha256(map_path),
            "extent_lonlat": [float(value) for value in extent],
            "geography_usable": bool(not land.empty and not coastline.empty),
            "background_kind": "expanded_gshhs_land_and_physical_coastline",
        }
    return maps
