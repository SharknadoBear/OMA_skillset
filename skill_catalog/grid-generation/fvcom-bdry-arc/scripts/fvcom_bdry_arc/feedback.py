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

    recommendations = _candidate_recommendations(
        polygon_xy,
        region,
        unintended,
        land_union_for_contract,
        tolerance_m=tolerance_m,
        candidate_max_km=float(candidate_max_km),
    )
    if frame_clip_policy == "report-only":
        status = "input_needs_review"
    elif role_contract_enabled and unintended and not structural_failures:
        status = "assign_boundary_roles"
    elif not diagnostic_pass:
        status = "adjust_bpoly" if recommendations and not structural_failures else "input_needs_review"
    elif structural_failures:
        status = "input_needs_review"
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

    feedback_failures = (
        list(dict.fromkeys(
            structural_failures
            + (["residual_boundary_role_decision_required"] if unintended else [])
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
        "schema_version": "region_bpoly_arc_feedback_v1",
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
    _plot_feedback(map_path, polygon_lonlat, wet_gdf, open_gdf, frame_gdf, status)
    component_maps = _plot_residual_component_maps(
        output_dir,
        polygon_lonlat,
        wet_gdf,
        open_gdf,
        records,
        coastline_gpkg,
    )
    feedback["outputs"] = {
        "segments_geojson": str(segments_path),
        "feedback_map": str(map_path),
        "residual_component_maps": component_maps,
    }
    feedback_path = output_dir / "region_bpoly_arc_feedback_v1.json"
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
        "fvcom_open_exterior_contract_v2"
        if role_contract_enabled
        else "fvcom_open_exterior_contract_v1"
    )
    contract_failures = (
        list(dict.fromkeys(
            structural_failures
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
        "per_side": side_summaries,
        "coastline_source_coverage_required": coastline_coverage_required,
        "coastline_source_coverage": coastline_coverage,
        "source_hashes": dict(feedback["input_sha256"]),
        "map": {"path": str(map_path), "sha256": file_sha256(map_path)},
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
            "open_exterior_agent_decision_v2"
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
    feedback["open_exterior_contract"] = contract
    feedback["outputs"].update({
        "open_exterior_contract": str(contract_path),
        "open_exterior_agent_decision": str(decision_path),
        "open_exterior_review_map": str(map_path),
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
                    "candidate_id": f"side_{side_index}_{profile}_{int(round(distance_km)):03d}km",
                    "operation": "reshape",
                    "side_index": int(side_index),
                    "side_name": _side_name(region, side_index),
                    "profile": profile,
                    "displacement_km": float(distance_km),
                    "outward_unit_east_north": [float(nx), float(ny)],
                    "vertex_delta_km": deltas,
                    "source_unintended_frame_clip_length_m": float(sum(r["length_m"] for r in records)),
                    "source_side_position_mean": float(mean_position),
                    "predicted_uncovered_side_length_m": float(predicted_uncovered_m),
                    "distance_search": {
                        "coarse_increment_km": 5.0,
                        "refinement_increment_km": 1.0,
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
    distance = 5.0
    while distance <= candidate_max_km + 1.0e-9:
        value = uncovered(distance)
        coarse.append((distance, value))
        if value <= tolerance_m:
            break
        distance += 5.0
    if not coarse:
        return None, float("inf")
    coarse_choice = min(coarse, key=lambda item: (item[1] > tolerance_m, item[1], item[0]))
    if coarse_choice[1] <= tolerance_m:
        lower = max(1.0, coarse_choice[0] - 5.0)
        refined = []
        distance = lower
        while distance <= coarse_choice[0] + 1.0e-9:
            refined.append((distance, uncovered(distance)))
            distance += 1.0
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
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
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
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_residual_component_maps(
    output_dir: Path,
    region: Polygon,
    wet_gdf: gpd.GeoDataFrame,
    open_gdf: gpd.GeoDataFrame,
    records: list[dict[str, Any]],
    coastline_gpkg: str | Path | None,
) -> dict[str, dict[str, Any]]:
    """Write one geographic, hash-bound zoom map per raw frame-water component."""
    land = _read_layer(Path(coastline_gpkg), "land_polygons") if coastline_gpkg else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
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
        if not land.empty:
            clipped = land.cx[extent[0] : extent[2], extent[1] : extent[3]]
            if not clipped.empty:
                clipped.plot(ax=ax, facecolor="#d8d1bd", edgecolor="#31572c", linewidth=0.8, alpha=0.9, label="GSHHS land")
        if not wet_gdf.empty:
            wet_gdf.plot(ax=ax, facecolor="#8ecae6", edgecolor="#0b4f6c", linewidth=0.7, alpha=0.35, label="retained wet domain")
        gpd.GeoSeries([region], crs="EPSG:4326").boundary.plot(ax=ax, color="#111111", linewidth=1.2, linestyle="--", label="RegionBPoly frame")
        if not open_gdf.empty:
            open_gdf.plot(ax=ax, color="#d00000", linewidth=2.2, label="delivered OBC")
        gpd.GeoSeries([line], crs="EPSG:4326").plot(ax=ax, color="#ff8c00", linewidth=4.0, label="residual water segment")
        endpoints = [Point(line.coords[0]), Point(line.coords[-1])]
        gpd.GeoSeries(endpoints, crs="EPSG:4326").plot(ax=ax, color="#6a00f4", markersize=45, zorder=5, label="closure endpoints")
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
        }
    return maps
