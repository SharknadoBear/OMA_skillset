"""Topology postprocessor for FVCOM boundary-arc packages.

The implementation is intentionally conservative. It writes useful topology
evidence even when the coastline linework cannot support an autonomous pass.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from affine import Affine
from rasterio import features as rio_features
from scipy import ndimage
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, box, mapping, shape
from shapely.prepared import prep
from shapely.ops import linemerge, nearest_points, polygonize, substring, unary_union

from .boundary_loops import build_model_boundary_loops
from .boundary_resolution import boundary_resolution_config, build_boundary_resolution
from .feedback import build_region_bpoly_arc_feedback
from .projection import LocalProjection, local_utm_projection, project_geometry, unproject_geometry, unwrap_geometry_longitudes, unwrap_longitude


@dataclass(frozen=True)
class BdryArcConfig:
    """Run controls for the boundary-arc workflow."""

    mode: str = "execute"
    target_resolution_m: float = 250.0
    review_depth: str = "auto"
    coastline_buffer_km: float = 10.0
    seed_mode: str = "auto"
    manual_seed_json: str | None = None
    fetch_coastline: bool = False
    coastline_source: str = "gshhs"
    cusp_skill_dir: str | None = None
    gshhs_skill_dir: str | None = None
    gshhs_resolution: str = "f"
    gshhs_levels: str = "1"
    fallback_policy: str = "auto"
    topology_mode: str = "gshhs-vector"
    raster_resolution_m: float | None = None
    max_topology_iterations: int = 4
    convergence_area_frac: float = 0.01
    convergence_anchor_m: float | None = None
    progress_interval_s: float = 30.0
    heuristic_mode: str = "auto"
    topology_time_budget_s: float = 900.0
    boundary_resolution_profile: str = "legacy"
    frame_clip_policy: str = "reject-unintended"
    residual_boundary_policy: str = "solid-default"
    frame_clip_tolerance_m: float | None = None
    feedback_candidate_max_km: float = 100.0
    obc_placement_policy: str = "offshore-first"


def _resolve_heuristic_mode(cli_mode: str, run_mode: str) -> tuple[str, bool]:
    if cli_mode == "auto":
        resolved = "memory" if run_mode == "execute" else "unknown"
    else:
        resolved = cli_mode
    return resolved, resolved == "memory"


def run_bdry_arc(
    region_bpoly_json: str | Path,
    offshore_artifacts_json: str | Path,
    run_dir: str | Path,
    name: str,
    coastline_gpkg: str | Path | None = None,
    config: BdryArcConfig | None = None,
) -> dict[str, Any]:
    """Run the full boundary-arc postprocessor and write artifacts."""
    config = config or BdryArcConfig()
    if config.mode not in {"execute", "test"}:
        raise ValueError("--mode must be execute or test")
    if config.review_depth not in {"auto", "fast", "full"}:
        raise ValueError("--review-depth must be auto, fast, or full")
    if config.seed_mode not in {"auto", "manual-json"}:
        raise ValueError("--seed-mode must be auto or manual-json")
    if config.coastline_source not in {"gshhs", "generic-gpkg", "cusp-legacy"}:
        raise ValueError("--coastline-source must be gshhs, generic-gpkg, or cusp-legacy")
    if config.topology_mode not in {"gshhs-vector", "island-loop", "iterative-raster", "vector-only"}:
        raise ValueError("--topology-mode must be gshhs-vector, island-loop, iterative-raster, or vector-only")
    if config.heuristic_mode not in {"auto", "memory", "unknown"}:
        raise ValueError("--heuristic-mode must be auto, memory, or unknown")
    if config.boundary_resolution_profile not in {"legacy", "adaptive-coastal-v1", "adaptive-coastal-v2"}:
        raise ValueError("--boundary-resolution-profile must be legacy, adaptive-coastal-v1, or adaptive-coastal-v2")
    if config.frame_clip_policy not in {"reject-unintended", "report-only"}:
        raise ValueError("--frame-clip-policy must be reject-unintended or report-only")
    if config.residual_boundary_policy not in {"solid-default", "strict-reject"}:
        raise ValueError("--residual-boundary-policy must be solid-default or strict-reject")
    if config.obc_placement_policy not in {"offshore-first", "mouth-first"}:
        raise ValueError("--obc-placement-policy must be offshore-first or mouth-first")
    if config.frame_clip_tolerance_m is not None and config.frame_clip_tolerance_m < 0.0:
        raise ValueError("--frame-clip-tolerance-m must be nonnegative")
    if config.feedback_candidate_max_km <= 0.0:
        raise ValueError("--feedback-candidate-max-km must be positive")
    if config.topology_mode == "gshhs-vector" and config.coastline_source == "cusp-legacy":
        raise ValueError("--topology-mode gshhs-vector requires GSHHS/generic polygon-capable coastline input")
    resolved_heuristic_mode, place_memory_enabled = _resolve_heuristic_mode(config.heuristic_mode, config.mode)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_progress(
        run_dir,
        "run",
        "start",
        {
            "name": name,
            "topology_mode": config.topology_mode,
            "coastline_source": config.coastline_source,
            "gshhs_resolution_requested": config.gshhs_resolution,
            "heuristic_mode": resolved_heuristic_mode,
            "resolution_policy": "use requested GSHHS resolution only; do not downshift unless explicitly requested",
            "boundary_resolution_profile": config.boundary_resolution_profile,
            "frame_clip_policy": config.frame_clip_policy,
            "residual_boundary_policy": config.residual_boundary_policy,
            "obc_placement_policy": config.obc_placement_policy,
        },
    )
    visual_dir = run_dir / "intermediate" / "visual_review"
    if config.mode == "test":
        visual_dir.mkdir(parents=True, exist_ok=True)

    region = _read_json(region_bpoly_json)
    offshore = _read_json(offshore_artifacts_json)
    if _upstream_bpoly_unresolved(region):
        return _write_unresolved_upstream_manifest(
            region,
            offshore,
            run_dir,
            name,
            config,
            resolved_heuristic_mode,
            place_memory_enabled,
        )
    bpoly_lonlat_raw = _load_bpoly_polygon(region)
    bbox_wsen = tuple(float(v) for v in region.get("envelope_bbox") or bpoly_lonlat_raw.bounds)
    buffered_bbox = _buffer_bbox_lonlat(bbox_wsen, config.coastline_buffer_km)
    projection = local_utm_projection(buffered_bbox)
    longitude_origin = float(projection.longitude_origin or 0.0)
    bpoly_lonlat = unwrap_geometry_longitudes(bpoly_lonlat_raw, longitude_origin)
    bpoly_xy = project_geometry(bpoly_lonlat, projection).buffer(0)

    fetch_metadata: dict[str, Any] = {}
    if config.fetch_coastline:
        if config.coastline_source == "cusp-legacy":
            coastline_gpkg = _fetch_cusp_coastline(
                buffered_bbox,
                run_dir / "coastline",
                name,
                cusp_skill_dir=config.cusp_skill_dir,
                fallback_policy=config.fallback_policy,
            )
            fetch_metadata = {"fetched_with": "cusp-coastline", "coastline_source": "cusp-legacy"}
        else:
            coastline_gpkg, fetch_metadata = _fetch_gshhs_coastline(
                buffered_bbox,
                run_dir / "coastline",
                name,
                gshhs_skill_dir=config.gshhs_skill_dir,
                resolution=config.gshhs_resolution,
                levels=config.gshhs_levels,
                progress_interval_s=config.progress_interval_s,
            )
    if coastline_gpkg is None:
        raise ValueError("Provide --coastline-gpkg or use --fetch-coastline")

    _write_progress(run_dir, "load-coastline", "start", {"coastline_gpkg": str(coastline_gpkg)})
    coastline_raw, land_polygons_raw, coastline_load_meta = _load_coastline_product(
        coastline_gpkg,
        buffered_bbox,
        config.coastline_source,
    )
    if fetch_metadata:
        coastline_load_meta["fetch"] = fetch_metadata
    _write_progress(run_dir, "load-coastline", "done", {"feature_count": int(len(coastline_raw)), **coastline_load_meta})
    _write_progress(run_dir, "project-coastline", "start", {"target_crs": str(projection.crs)})
    coastline_raw = _unwrap_gdf_longitudes(coastline_raw, longitude_origin)
    land_polygons_raw = _unwrap_gdf_longitudes(land_polygons_raw, longitude_origin)
    coastline_xy = coastline_raw.to_crs(projection.crs) if not coastline_raw.empty else coastline_raw
    land_polygons_xy_gdf = land_polygons_raw.to_crs(projection.crs) if not land_polygons_raw.empty else land_polygons_raw
    raw_lines_xy = _flatten_lines(coastline_xy.geometry)
    land_polygons_xy = _flatten_polygons(land_polygons_xy_gdf.geometry)
    _write_progress(run_dir, "project-coastline", "done", {"line_count": int(len(raw_lines_xy))})
    audit = audit_coastline_topology(raw_lines_xy, bpoly_xy, config.target_resolution_m)
    _write_progress(run_dir, "audit", "done", {"line_count": audit["line_count"]})
    repaired_lines_xy, repair_meta, bridge_lines_xy = repair_coastline_graph(
        raw_lines_xy,
        bpoly_xy,
        config.target_resolution_m,
    )
    _write_progress(run_dir, "repair", "done", repair_meta)

    selected_side = unwrap_geometry_longitudes(_selected_side_line(offshore, region), longitude_origin)
    selected_side_xy = project_geometry(selected_side, projection)
    offshore_unit = _offshore_unit_vector(offshore, selected_side_xy)
    seed_xy, seed_meta = _resolve_seed(region, config, projection, bpoly_xy)
    forbidden_regions_lonlat: list[dict[str, Any]] = []
    forbidden_regions_xy: list[Polygon] = []
    lake_closed_branch = _uses_lake_closed_boundary(region, offshore)
    island_loop_branch = False if lake_closed_branch else _uses_island_loop_branch(region, offshore, config, place_memory_enabled)

    if lake_closed_branch:
        anchors = _lake_closed_boundary_reference_points(bpoly_xy, selected_side_xy, config.target_resolution_m)
    elif island_loop_branch:
        anchors = _island_loop_reference_points(selected_side_xy, bpoly_xy, config.target_resolution_m)
    elif config.topology_mode == "gshhs-vector":
        land_boundary_xy = unary_union(land_polygons_xy).boundary if land_polygons_xy else GeometryCollection()
        if land_boundary_xy.is_empty and repaired_lines_xy:
            land_boundary_xy = unary_union(repaired_lines_xy)
        anchors = _coastline_bpoly_anchor_points(
            land_boundary_xy,
            selected_side_xy,
            bpoly_xy,
            config.target_resolution_m,
        )
    else:
        anchors = _select_anchor_points(repaired_lines_xy, selected_side_xy, bpoly_xy, config.target_resolution_m)
    if lake_closed_branch:
        candidates = generate_lake_closed_boundary_candidates()
    elif island_loop_branch:
        candidates = generate_island_loop_candidates(
            bpoly_xy,
            selected_side_xy,
            offshore_unit,
            config.target_resolution_m,
        )
    else:
        candidates = generate_offshore_arc_candidates(
            anchors["start_xy"],
            anchors["end_xy"],
            offshore_unit,
            selected_side_xy,
            bpoly_xy,
            config.target_resolution_m,
            anchors=anchors,
        )
    if len(repaired_lines_xy) <= 20_000:
        coast_union_xy = unary_union(repaired_lines_xy) if repaired_lines_xy else GeometryCollection()
    else:
        coast_union_xy = GeometryCollection()
    if lake_closed_branch:
        scored = {"selected": candidates[0], "candidates": candidates}
    else:
        scored = (
            score_island_loop_candidates(candidates, bpoly_xy, config.target_resolution_m)
            if island_loop_branch
            else score_and_select_bdry_arc(candidates, coast_union_xy, bpoly_xy, config.target_resolution_m)
        )
    selected_arc_xy = scored["selected"]["geometry"]
    _write_progress(run_dir, "initial-arc", "done", {"candidate_count": len(scored["candidates"]), "selected": scored["selected"]["candidate_id"]})

    topology_mode_used = config.topology_mode
    topology_iterations: list[dict[str, Any]] = []
    if lake_closed_branch:
        topology_mode_used = "lake-closed-boundary"
        wet_result = extract_lake_closed_wet_domain(
            land_polygons_xy,
            bpoly_xy,
            seed_xy,
            config.target_resolution_m,
        )
        selected_arc_xy = wet_result.get("open_arc_xy", scored["selected"]["geometry"])
        scored["selected"]["geometry"] = selected_arc_xy
        _write_progress(run_dir, "lake-closed-boundary", "done", wet_result["metadata"])
    elif island_loop_branch:
        topology_mode_used = "island-loop"
        topology_selection = select_island_loop_topology(
            scored,
            land_polygons_xy,
            bpoly_xy,
            seed_xy,
            config.target_resolution_m,
            run_dir=run_dir,
            topology_time_budget_s=config.topology_time_budget_s,
        )
        scored = topology_selection["scored"]
        wet_result = topology_selection["wet_result"]
        selected_arc_xy = wet_result.get("open_arc_xy", scored["selected"]["geometry"])
        scored["selected"]["geometry"] = selected_arc_xy
        _write_progress(run_dir, "island-loop", "done", wet_result["metadata"])
    elif config.topology_mode == "gshhs-vector":
        _write_progress(run_dir, "gshhs-vector", "start", {"candidate_count": len(scored["candidates"])})
        topology_selection = select_gshhs_open_side_topology(
            scored,
            repaired_lines_xy,
            land_polygons_xy,
            bpoly_xy,
            seed_xy,
            config.target_resolution_m,
            anchors,
            run_dir=run_dir,
            topology_time_budget_s=config.topology_time_budget_s,
            obc_placement_policy=config.obc_placement_policy,
        )
        scored = topology_selection["scored"]
        wet_result = topology_selection["wet_result"]
        selected_arc_xy = wet_result.get("open_arc_xy", scored["selected"]["geometry"])
        scored["selected"]["geometry"] = selected_arc_xy
        _write_progress(run_dir, "gshhs-vector", "done", wet_result["metadata"])
    elif config.topology_mode == "iterative-raster":
        topology = iterative_raster_topology(
            repaired_lines_xy,
            selected_arc_xy,
            anchors,
            selected_side_xy,
            offshore_unit,
            bpoly_xy,
            seed_xy,
            config,
            visual_dir if config.mode == "test" else None,
            projection,
            name,
            bpoly_lonlat,
        )
        _write_progress(run_dir, "iterative-raster", "done", {"iterations": len(topology["iterations"])})
        repaired_lines_xy = topology["relevant_lines_xy"]
        selected_arc_xy = topology["selected_arc_xy"]
        anchors = topology["anchors"]
        scored = topology["scored"]
        wet_result = topology["wet_result"]
        topology_iterations = topology["iterations"]
        repair_meta["postclassification_retained_line_count"] = int(len(repaired_lines_xy))
        repair_meta["postclassification_dropped_line_count"] = int(max(len(raw_lines_xy) - len(repaired_lines_xy), 0))
    else:
        wet_result = extract_seeded_wet_domain(
            repaired_lines_xy,
            selected_arc_xy,
            bpoly_xy,
            seed_xy,
            forbidden_regions_xy,
            config.target_resolution_m,
        )
    if wet_result.get("metadata", {}).get("open_arc_trimmed_to_wet_exterior"):
        anchors = _promote_delivered_open_arc_landfalls(
            anchors,
            selected_arc_xy,
            land_boundary_xy,
            config.target_resolution_m,
        )
    if not lake_closed_branch and not island_loop_branch:
        trimmed = bool(wet_result.get("metadata", {}).get("open_arc_trimmed_to_wet_exterior"))
        if config.obc_placement_policy == "offshore-first":
            family = "compact-mouth-fallback" if trimmed else "complete-offshore"
        else:
            family = "compact-mouth" if trimmed else "complete-offshore-fallback"
        wet_result.setdefault("metadata", {})["obc_placement_policy"] = config.obc_placement_policy
        wet_result["metadata"]["obc_placement_family"] = family
        wet_result["metadata"]["offshore_family_attempted_first"] = config.obc_placement_policy == "offshore-first"
    final_status, failure_taxonomy = _final_status(scored, wet_result, anchors, forbidden_regions_xy)
    advisory_taxonomy: list[str] = []
    if wet_result.get("metadata", {}).get("open_arc_trimmed_to_wet_exterior"):
        advisory_taxonomy.append("source_open_arc_tail_trimmed_to_wet_exterior")
        if scored["selected"].get("metrics", {}).get("extra_coastline_intersection"):
            advisory_taxonomy.append("discarded_source_arc_tail_intersects_extra_coastline")
    resolution_policy = _gshhs_resolution_policy(config, coastline_load_meta)
    if resolution_policy["downgraded_without_explicit_request"]:
        final_status = "needs_review"
        if "gshhs_resolution_downgraded_without_explicit_request" not in failure_taxonomy:
            failure_taxonomy.append("gshhs_resolution_downgraded_without_explicit_request")

    selected_arc_lonlat = unproject_geometry(selected_arc_xy, projection)
    wet_domain_lonlat = unproject_geometry(wet_result["wet_domain_xy"], projection)
    anchors_lonlat = [
        unproject_geometry(Point(anchors["start_xy"]), projection),
        unproject_geometry(Point(anchors["end_xy"]), projection),
    ]
    candidates_gdf = _candidate_gdf(scored["candidates"], projection)
    raw_gdf = _lines_gdf(raw_lines_xy, projection, "raw_coastline")
    repaired_gdf = _lines_gdf(repaired_lines_xy + bridge_lines_xy, projection, "repaired_coastline")
    forbidden_gdf = _forbidden_gdf(forbidden_regions_lonlat)
    layers = _build_output_layers(
        wet_domain_lonlat,
        selected_arc_lonlat,
        anchors_lonlat,
        candidates_gdf,
        raw_gdf,
        repaired_gdf,
        forbidden_gdf,
        wet_result,
        projection,
    )
    outputs = _write_outputs(run_dir, layers, name)
    _write_progress(run_dir, "write-outputs", "done", outputs)

    if config.mode == "test":
        _write_review_maps(
            visual_dir,
            name,
            layers,
            bpoly_lonlat,
            candidates_gdf,
            selected_arc_lonlat,
            final_status,
        )
        if config.topology_mode == "gshhs-vector":
            _write_gshhs_review_maps(visual_dir, name, layers, bpoly_lonlat, final_status)
    final_map = run_dir / "bdry_arc_review_map.png"
    _plot_final_map(final_map, layers, bpoly_lonlat, final_status)

    manifest = {
        "schema_version": "fvcom_bdry_arc_manifest_v1",
        "name": name,
        "created_by": "fvcom-bdry-arc",
        "final_status": final_status,
        "failure_taxonomy": failure_taxonomy,
        "advisory_taxonomy": advisory_taxonomy,
        "inputs": {
            "region_bpoly_json": str(region_bpoly_json),
            "offshore_artifacts_json": str(offshore_artifacts_json),
            "coastline_gpkg": str(coastline_gpkg),
            "coastline_source": config.coastline_source,
            "coastline_load": coastline_load_meta,
        },
        "settings": {
            "mode": config.mode,
            "target_resolution_m": float(config.target_resolution_m),
            "review_depth": config.review_depth,
            "coastline_buffer_km": float(config.coastline_buffer_km),
            "seed_mode": config.seed_mode,
            "coastline_source": config.coastline_source,
            "gshhs_resolution": config.gshhs_resolution,
            "gshhs_levels": config.gshhs_levels,
            "topology_mode": config.topology_mode,
            "topology_mode_used": topology_mode_used,
            "heuristic_mode": resolved_heuristic_mode,
            "place_memory_enabled": place_memory_enabled,
            "lake_closed_boundary_branch": lake_closed_branch,
            "island_loop_branch": island_loop_branch,
            "raster_resolution_m": config.raster_resolution_m,
            "max_topology_iterations": int(config.max_topology_iterations),
            "convergence_area_frac": float(config.convergence_area_frac),
            "convergence_anchor_m": float(config.convergence_anchor_m or 2.0 * config.target_resolution_m),
            "progress_interval_s": float(config.progress_interval_s),
            "topology_time_budget_s": float(config.topology_time_budget_s),
            "boundary_resolution_profile": config.boundary_resolution_profile,
            "frame_clip_policy": config.frame_clip_policy,
            "residual_boundary_policy": config.residual_boundary_policy,
            "obc_placement_policy": config.obc_placement_policy,
            "frame_clip_tolerance_m": (
                float(config.frame_clip_tolerance_m)
                if config.frame_clip_tolerance_m is not None
                else float(max(250.0, 0.05 * config.target_resolution_m))
            ),
            "feedback_candidate_max_km": float(config.feedback_candidate_max_km),
        },
        "region_bpoly": {
            "domain_type": region.get("domain_type"),
            "domain_variant": region.get("domain_variant"),
            "canonical_region_key": region.get("qa", {})
            .get("bpoly_quality", {})
            .get("canonical_region_key"),
            "envelope_bbox": list(map(float, bbox_wsen)),
            "buffered_bbox": list(map(float, buffered_bbox)),
            "polygon_lonlat": list(mapping(bpoly_lonlat)["coordinates"][0]),
        },
        "offshore_side": {
            "selected_side_index": offshore.get("selected_side_index"),
            "selected_side_name": offshore.get("selected_side_name"),
            "offshore_azimuth_deg": offshore.get("offshore_azimuth_deg"),
            "selected_side_start_lonlat": offshore.get("selected_side_start_lonlat"),
            "selected_side_end_lonlat": offshore.get("selected_side_end_lonlat"),
        },
        "projection": {
            "crs": str(projection.crs),
            "epsg": projection.epsg,
            "longitude_origin": projection.longitude_origin,
        },
        "coastline_audit": audit,
        "repair": repair_meta,
        "seed": seed_meta,
        "anchors": {
            "source": anchors.get("source", "coastline"),
            "start_role": anchors.get("start_role", "start"),
            "end_role": anchors.get("end_role", "end"),
            "start_lonlat": [float(anchors_lonlat[0].x), float(anchors_lonlat[0].y)],
            "end_lonlat": [float(anchors_lonlat[1].x), float(anchors_lonlat[1].y)],
            "start_distance_to_side_endpoint_m": float(anchors["start_distance_m"]),
            "end_distance_to_side_endpoint_m": float(anchors["end_distance_m"]),
            "anchor_distance_m": float(Point(anchors["start_xy"]).distance(Point(anchors["end_xy"]))),
            "start_adjacent_side_index": anchors.get("start_adjacent_side_index"),
            "end_adjacent_side_index": anchors.get("end_adjacent_side_index"),
            "selected_side_index": anchors.get("selected_side_index"),
            "start_anchor_method": anchors.get("start_anchor_method"),
            "end_anchor_method": anchors.get("end_anchor_method"),
            "start_anchor_found": anchors.get("start_anchor_found"),
            "end_anchor_found": anchors.get("end_anchor_found"),
            "start_snap_distance_m": anchors.get("start_snap_distance_m"),
            "end_snap_distance_m": anchors.get("end_snap_distance_m"),
            "anchor_tolerance_m": anchors.get("anchor_tolerance_m"),
            "seaward_chain_lonlat": [
                [float(pt.x), float(pt.y)]
                for pt in (
                    unproject_geometry(Point(xy), projection)
                    for xy in anchors.get("seaward_chain_xy", [])
                )
            ],
        },
        "offshore_arc": {
            "selected_candidate_id": scored["selected"]["candidate_id"],
            "candidate_count": len(scored["candidates"]),
            "selected_score": scored["selected"]["score"],
            "selected_metrics": scored["selected"]["metrics"],
        },
        "wet_domain": wet_result["metadata"],
        "topology_iterations": topology_iterations,
        "qa": {
            "forbidden_region_count": int(len(forbidden_regions_xy)),
            "forbidden_region_overlap": wet_result["metadata"].get("forbidden_overlap", []),
            "status_rule": "pass only when anchors, selected arc, seeded polygon/raster connectivity, and component classification checks are clean",
            "gshhs_resolution_policy": resolution_policy,
        },
        "outputs": {
            **outputs,
            "bdry_arc_review_map": str(final_map),
            "visual_review_dir": str(visual_dir) if config.mode == "test" else None,
        },
    }
    manifest_path = run_dir / "bdry_arc_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

    loop_run_dir = run_dir / "model_boundary_loops"
    try:
        _write_progress(run_dir, "model-boundary-loops", "start", {"run_dir": str(loop_run_dir)})
        loop_manifest = build_model_boundary_loops(
            outputs["bdry_arc_package_gpkg"],
            manifest_path,
            loop_run_dir,
            name,
            target_resolution_m=config.target_resolution_m,
            min_island_area_m2=0.0,
            mode=config.mode,
        )
        _write_progress(run_dir, "model-boundary-loops", "done", {"final_status": loop_manifest.get("final_status")})
    except Exception as exc:
        _write_progress(run_dir, "model-boundary-loops", "failed", {"error": str(exc)})
        loop_run_dir.mkdir(parents=True, exist_ok=True)
        loop_manifest = {
            "schema_version": "fvcom_model_boundary_loops_v1",
            "name": name,
            "created_by": "fvcom-bdry-arc automatic loop builder",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "final_status": "needs_review",
            "failure_taxonomy": ["model_boundary_loop_build_failed"],
            "error": str(exc),
            "inputs": {
                "bdry_arc_gpkg": str(outputs["bdry_arc_package_gpkg"]),
                "manifest_json": str(manifest_path),
            },
            "outputs": {},
            "qa": {},
        }
        (loop_run_dir / "model_boundary_loop_manifest.json").write_text(
            json.dumps(_json_safe(loop_manifest), indent=2),
            encoding="utf-8",
        )

    loop_manifest_path = loop_run_dir / "model_boundary_loop_manifest.json"
    loop_outputs = dict(loop_manifest.get("outputs", {}))
    loop_outputs["model_boundary_loop_manifest"] = str(loop_manifest_path)
    loop_outputs["model_boundary_loop_dir"] = str(loop_run_dir)
    manifest["outputs"].update(loop_outputs)
    manifest["model_boundary_loops"] = {
        "final_status": loop_manifest.get("final_status"),
        "failure_taxonomy": loop_manifest.get("failure_taxonomy", []),
        "qa": loop_manifest.get("qa", {}),
        "outputs": loop_outputs,
    }
    if loop_manifest.get("final_status") != "pass":
        if manifest["final_status"] == "pass":
            manifest["final_status"] = "needs_review"
        if "model_boundary_loop_needs_review" not in manifest["failure_taxonomy"]:
            manifest["failure_taxonomy"].append("model_boundary_loop_needs_review")
    manifest["qa"]["model_boundary_loop_status_rule"] = (
        "run_bdry_arc automatically builds continuous model-boundary loops; "
        "the main manifest needs review when the loop package needs review"
    )
    feedback_dir = run_dir / "fb"
    feedback: dict[str, Any]
    feedback_failed = False
    try:
        feedback = build_region_bpoly_arc_feedback(
            region_bpoly_json,
            offshore_artifacts_json,
            outputs["bdry_arc_package_gpkg"],
            coastline_gpkg,
            loop_manifest_path,
            feedback_dir,
            manifest,
            frame_clip_policy=config.frame_clip_policy,
            residual_boundary_policy=config.residual_boundary_policy,
            frame_clip_tolerance_m=config.frame_clip_tolerance_m,
            candidate_max_km=config.feedback_candidate_max_km,
            adaptive_status=("disabled" if config.boundary_resolution_profile == "legacy" else "pending"),
        )
    except Exception as exc:
        feedback_failed = True
        feedback_dir.mkdir(parents=True, exist_ok=True)
        feedback = {
            "schema_version": "region_bpoly_arc_feedback_v1",
            "status": "input_needs_review",
            "diagnostic_status": "failed",
            "failure_taxonomy": ["region_bpoly_arc_feedback_failed"],
            "error": str(exc),
            "outputs": {},
        }
        feedback_path = feedback_dir / "region_bpoly_arc_feedback_v1.json"
        feedback_path.write_text(json.dumps(_json_safe(feedback), indent=2), encoding="utf-8")
        feedback["outputs"]["feedback_json"] = str(feedback_path)
    manifest["region_bpoly_arc_feedback"] = feedback
    manifest["outputs"].update(feedback.get("outputs", {}))
    open_contract = dict(feedback.get("open_exterior_contract", {}))
    manifest["open_exterior_contract"] = open_contract
    frame_gate_blocked = bool(not open_contract.get("downstream_eligible", False))
    if frame_gate_blocked:
        manifest["final_status"] = "needs_review"
        for failure in open_contract.get("failure_taxonomy", ["open_exterior_contract_not_downstream_eligible"]):
            if failure not in manifest["failure_taxonomy"]:
                manifest["failure_taxonomy"].append(failure)
    if feedback_failed:
        manifest["final_status"] = "needs_review"
        if "region_bpoly_arc_feedback_failed" not in manifest["failure_taxonomy"]:
            manifest["failure_taxonomy"].append("region_bpoly_arc_feedback_failed")

    adaptive_requested = config.boundary_resolution_profile in {"adaptive-coastal-v1", "adaptive-coastal-v2"}
    adaptive_can_run = bool(
        adaptive_requested
        and loop_outputs.get("model_boundary_loops_gpkg")
        and loop_manifest.get("final_status") == "pass"
        and not frame_gate_blocked
        and not feedback_failed
    )
    if adaptive_can_run:
        resolution_dir = run_dir / "boundary_resolution"
        _write_progress(run_dir, "boundary-resolution", "start", {"run_dir": str(resolution_dir)})
        try:
            resolution_manifest = build_boundary_resolution(
                loop_outputs["model_boundary_loops_gpkg"],
                loop_manifest_path,
                region_bpoly_json,
                coastline_gpkg,
                resolution_dir,
                name,
                boundary_resolution_config(config.boundary_resolution_profile),
            )
            manifest["boundary_resolution"] = resolution_manifest
            manifest["outputs"].update(resolution_manifest.get("outputs", {}))
            if resolution_manifest.get("final_status") != "pass":
                manifest["final_status"] = "needs_review"
                if "adaptive_boundary_resolution_needs_review" not in manifest["failure_taxonomy"]:
                    manifest["failure_taxonomy"].append("adaptive_boundary_resolution_needs_review")
            _write_progress(run_dir, "boundary-resolution", "done", {"final_status": resolution_manifest.get("final_status")})
        except Exception as exc:
            manifest["boundary_resolution"] = {
                "schema_version": (
                    "fvcom_boundary_resolution_manifest_v2"
                    if config.boundary_resolution_profile == "adaptive-coastal-v2"
                    else "fvcom_boundary_resolution_manifest_v1"
                ),
                "profile": config.boundary_resolution_profile,
                "final_status": "needs_review",
                "failure_taxonomy": ["adaptive_boundary_resolution_failed"],
                "error": str(exc),
                "outputs": {},
            }
            manifest["final_status"] = "needs_review"
            if "adaptive_boundary_resolution_failed" not in manifest["failure_taxonomy"]:
                manifest["failure_taxonomy"].append("adaptive_boundary_resolution_failed")
            _write_progress(run_dir, "boundary-resolution", "failed", {"error": str(exc)})
    elif adaptive_requested and frame_gate_blocked:
        manifest["boundary_resolution"] = {
            "profile": config.boundary_resolution_profile,
            "enabled": False,
            "final_status": "needs_review",
            "failure_taxonomy": ["blocked_by_region_bpoly_feedback"],
            "reason": (
                "Residual boundary roles and the hash-bound Codex map decision must be finalized before adaptive boundary resolution."
                if config.residual_boundary_policy == "solid-default"
                else "Residual GSHHS frame clipping must be resolved by RegionBPoly adjustment before adaptive boundary resolution."
            ),
            "outputs": {},
        }
        if "blocked_by_region_bpoly_feedback" not in manifest["failure_taxonomy"]:
            manifest["failure_taxonomy"].append("blocked_by_region_bpoly_feedback")
        _write_progress(run_dir, "boundary-resolution", "blocked", {"reason": "region_bpoly_feedback"})
    elif adaptive_requested:
        manifest["boundary_resolution"] = {
            "profile": config.boundary_resolution_profile,
            "enabled": False,
            "final_status": "needs_review",
            "failure_taxonomy": ["adaptive_boundary_resolution_prerequisite_failed"],
            "outputs": {},
        }
    else:
        manifest["boundary_resolution"] = {
            "profile": "legacy",
            "enabled": False,
            "reason": "legacy_profile_preserves_existing_boundary_workflow",
        }

    if not feedback_failed and not frame_gate_blocked:
        adaptive_manifest = manifest.get("boundary_resolution", {})
        adaptive_status = (
            "disabled"
            if not adaptive_requested
            else str(adaptive_manifest.get("final_status", "needs_review"))
        )
        feedback = build_region_bpoly_arc_feedback(
            region_bpoly_json,
            offshore_artifacts_json,
            outputs["bdry_arc_package_gpkg"],
            coastline_gpkg,
            loop_manifest_path,
            feedback_dir,
            manifest,
            frame_clip_policy=config.frame_clip_policy,
            residual_boundary_policy=config.residual_boundary_policy,
            frame_clip_tolerance_m=config.frame_clip_tolerance_m,
            candidate_max_km=config.feedback_candidate_max_km,
            adaptive_status=adaptive_status,
            adaptive_failures=list(adaptive_manifest.get("failure_taxonomy", [])),
        )
        manifest["region_bpoly_arc_feedback"] = feedback
        manifest["outputs"].update(feedback.get("outputs", {}))
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_progress(run_dir, "complete", "done", {"final_status": manifest["final_status"]})
    return manifest


def audit_coastline_topology(lines_xy: list[LineString], bpoly_xy: Polygon, target_resolution_m: float) -> dict[str, Any]:
    """Return basic linework health metrics."""
    endpoints: list[Point] = []
    closed_count = 0
    total_length = 0.0
    for line in lines_xy:
        if line.is_empty or len(line.coords) < 2:
            continue
        total_length += float(line.length)
        if line.is_ring or Point(line.coords[0]).distance(Point(line.coords[-1])) <= 1.0:
            closed_count += 1
        else:
            endpoints.extend([Point(line.coords[0]), Point(line.coords[-1])])

    snap_key_m = max(1.0, min(float(target_resolution_m), 100.0))
    endpoint_bins: dict[tuple[int, int], int] = {}
    for point in endpoints:
        key = (int(round(point.x / snap_key_m)), int(round(point.y / snap_key_m)))
        endpoint_bins[key] = endpoint_bins.get(key, 0) + 1
    dangling = sum(1 for count in endpoint_bins.values() if count == 1)

    polygon_count: int | None = 0
    if len(lines_xy) <= 20_000:
        try:
            merged = unary_union(lines_xy) if lines_xy else GeometryCollection()
            polygon_count = len(list(polygonize(merged)))
        except Exception:
            polygon_count = 0
    else:
        polygon_count = None

    coverage_fraction: float | None = 0.0
    if len(lines_xy) > 30_000:
        coverage_fraction = None
    elif lines_xy and not bpoly_xy.is_empty:
        inside_length = sum(float(line.intersection(bpoly_xy).length) for line in lines_xy)
        coverage_fraction = float(inside_length / max(total_length, 1.0))

    return {
        "line_count": int(len(lines_xy)),
        "closed_line_count": int(closed_count),
        "endpoint_count": int(len(endpoints)),
        "rounded_dangling_endpoint_count": int(dangling),
        "polygonize_face_count": polygon_count,
        "polygonize_policy": "skipped_large_line_count" if polygon_count is None else "computed",
        "total_line_length_m": float(total_length),
        "line_length_fraction_inside_bpoly": coverage_fraction,
        "coverage_policy": "skipped_large_line_count" if coverage_fraction is None else "computed",
        "endpoint_rounding_m": float(snap_key_m),
    }


def repair_coastline_graph(
    lines_xy: list[LineString],
    bpoly_xy: Polygon,
    target_resolution_m: float,
) -> tuple[list[LineString], dict[str, Any], list[LineString]]:
    """Add conservative endpoint bridges without rewriting source linework."""
    endpoints: list[dict[str, Any]] = []
    for idx, line in enumerate(lines_xy):
        coords = list(line.coords)
        if len(coords) < 2 or Point(coords[0]).distance(Point(coords[-1])) <= 1.0:
            continue
        endpoints.append({"line": idx, "which": "start", "point": Point(coords[0]), "tangent": _endpoint_tangent(coords, "start")})
        endpoints.append({"line": idx, "which": "end", "point": Point(coords[-1]), "tangent": _endpoint_tangent(coords, "end")})

    max_bridge = max(25.0, min(float(target_resolution_m) * 0.75, 250.0))
    if len(endpoints) > 10_000:
        metadata = {
            "strategy": "non_destructive_bridge_lines",
            "bridge_tolerance_m": float(max_bridge),
            "candidate_endpoint_count": int(len(endpoints)),
            "bridge_count": 0,
            "boundary_endpoint_policy": "do_not_bridge_endpoints_near_bpoly_boundary",
            "bridge_policy": "skipped_large_endpoint_count",
        }
        return list(lines_xy), metadata, []
    boundary_zone = bpoly_xy.boundary.buffer(max(2.0 * target_resolution_m, 500.0))
    bridges: list[LineString] = []
    used: set[int] = set()
    for i, item in enumerate(endpoints):
        if i in used or item["point"].within(boundary_zone):
            continue
        best_j = None
        best_d = max_bridge
        for j, other in enumerate(endpoints):
            if i == j or j in used or item["line"] == other["line"]:
                continue
            if other["point"].within(boundary_zone):
                continue
            dist = item["point"].distance(other["point"])
            if dist < best_d and _tangent_compatible(item["tangent"], other["tangent"]):
                best_d = dist
                best_j = j
        if best_j is not None:
            other = endpoints[best_j]
            bridge = LineString([item["point"], other["point"]])
            bridges.append(bridge)
            used.add(i)
            used.add(best_j)

    metadata = {
        "strategy": "non_destructive_bridge_lines",
        "bridge_tolerance_m": float(max_bridge),
        "candidate_endpoint_count": int(len(endpoints)),
        "bridge_count": int(len(bridges)),
        "boundary_endpoint_policy": "do_not_bridge_endpoints_near_bpoly_boundary",
        "bridge_policy": "computed",
    }
    return list(lines_xy) + bridges, metadata, bridges


def generate_offshore_arc_candidates(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    offshore_unit: np.ndarray,
    selected_side_xy: LineString,
    bpoly_xy: Polygon,
    target_resolution_m: float,
    anchors: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate open arcs by smoothly deforming the selected bpoly intent chain."""
    p0 = np.asarray(start_xy, dtype=float)
    p1 = np.asarray(end_xy, dtype=float)
    side_coords = list(selected_side_xy.coords)
    side0 = np.asarray(side_coords[0], dtype=float)
    side1 = np.asarray(side_coords[-1], dtype=float)
    if anchors and anchors.get("seaward_chain_xy"):
        chain = [np.asarray(xy, dtype=float) for xy in anchors["seaward_chain_xy"]]
        if len(chain) >= 4:
            side0 = chain[1]
            side1 = chain[-2]
    chord = p1 - p0
    chord_len = max(float(np.linalg.norm(chord)), 1.0)
    side_dist = 0.5 * (Point(p0).distance(selected_side_xy) + Point(p1).distance(selected_side_xy))
    chain_len = float(Point(p0).distance(Point(side0)) + Point(side0).distance(Point(side1)) + Point(side1).distance(Point(p1)))
    base_bow = max(4.0 * target_resolution_m, 0.08 * max(chain_len, chord_len), 0.35 * side_dist)
    factors = [0.0, 0.2, 0.4, 0.65, 0.9, 1.2]
    candidates: list[dict[str, Any]] = []
    for idx, factor in enumerate(factors, start=1):
        bow = base_bow * factor
        if anchors and anchors.get("seaward_chain_xy"):
            geometry = _seaward_chain_arc(p0, side0, side1, p1, offshore_unit, bow, n=160)
            family = "coastline_anchor_seaward_bpoly_chain"
            candidate_id = f"seaward_chain_{idx:02d}"
        else:
            geometry = _deformed_side_arc(side0, side1, p0, p1, offshore_unit, bow, n=128)
            family = "deformed_bpoly_side"
            candidate_id = f"side_warp_{idx:02d}"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "family": family,
                "bow_distance_m": float(bow),
                "control_chain_xy": anchors.get("seaward_chain_xy") if anchors else None,
                "geometry": geometry,
            }
        )
    return candidates


def generate_island_loop_candidates(
    bpoly_xy: Polygon,
    selected_side_xy: LineString,
    offshore_unit: np.ndarray,
    target_resolution_m: float,
) -> list[dict[str, Any]]:
    """Generate smooth closed offshore loops for island and archipelago domains."""
    coords = [(float(x), float(y)) for x, y in list(bpoly_xy.exterior.coords)[:-1]]
    if len(coords) < 3:
        raise ValueError("Island-loop topology requires a valid bpoly polygon")
    centroid = np.asarray([bpoly_xy.centroid.x, bpoly_xy.centroid.y], dtype=float)
    factors = [1.0, 1.04, 1.08, 1.14, 1.22]
    side_mid = np.asarray(selected_side_xy.interpolate(0.5, normalized=True).coords[0], dtype=float)
    max_span = max(bpoly_xy.bounds[2] - bpoly_xy.bounds[0], bpoly_xy.bounds[3] - bpoly_xy.bounds[1], 1.0)
    bow_base = max(4.0 * target_resolution_m, 0.04 * max_span)
    candidates: list[dict[str, Any]] = []
    for idx, factor in enumerate(factors, start=1):
        ring: list[tuple[float, float]] = []
        bow = bow_base * max(factor - 1.0, 0.0) * 3.0
        for coord in coords:
            vec = np.asarray(coord, dtype=float) - centroid
            pt = centroid + vec * factor
            distance_to_side = Point(float(pt[0]), float(pt[1])).distance(selected_side_xy)
            influence = max(0.0, 1.0 - distance_to_side / max(0.35 * max_span, 1.0))
            if influence > 0.0:
                side_vec = pt - side_mid
                along = float(np.dot(side_vec, offshore_unit))
                if along >= -0.25 * max_span:
                    pt = pt + offshore_unit * bow * influence
            ring.append((float(pt[0]), float(pt[1])))
        smooth_ring = _chaikin_closed_ring(ring, iterations=3)
        if smooth_ring[0] != smooth_ring[-1]:
            smooth_ring.append(smooth_ring[0])
        line = LineString(smooth_ring)
        candidates.append(
            {
                "candidate_id": f"island_loop_{idx:02d}",
                "family": "island_archipelago_closed_loop",
                "bow_distance_m": float(bow),
                "geometry": line,
            }
        )
    return candidates


def generate_lake_closed_boundary_candidates() -> list[dict[str, Any]]:
    """Return the no-open-boundary sentinel candidate for lake domains."""
    return [
        {
            "candidate_id": "lake_closed_boundary_no_open_arc",
            "family": "lake_closed_boundary",
            "bow_distance_m": 0.0,
            "geometry": LineString(),
            "score": 100.0,
            "metrics": {
                "extra_coastline_intersection": False,
                "closure_method": "lake_closed_boundary_no_open_arc",
                "open_arc_boundary_overlap_fraction": None,
            },
        }
    ]


def score_island_loop_candidates(
    candidates: list[dict[str, Any]],
    bpoly_xy: Polygon,
    target_resolution_m: float,
) -> dict[str, Any]:
    """Score island-loop candidates without coastline anchor assumptions."""
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        line = candidate["geometry"]
        frame = Polygon(line.coords).buffer(0)
        frame_valid = isinstance(frame, Polygon) and not frame.is_empty and frame.is_valid
        samples = _sample_line_points(line, max(line.length / 80.0, target_resolution_m))
        inside_fraction = sum(1 for pt in samples if bpoly_xy.buffer(4.0 * target_resolution_m).contains(pt)) / max(len(samples), 1)
        area_ratio = float(frame.area / max(bpoly_xy.area, 1.0)) if frame_valid else 0.0
        score = 100.0 * inside_fraction
        score += 50.0 if frame_valid else -200.0
        score -= abs(area_ratio - 1.15) * 25.0
        item = dict(candidate)
        item["score"] = float(score)
        item["metrics"] = {
            "inside_bpoly_buffer_fraction": float(inside_fraction),
            "frame_valid": bool(frame_valid),
            "frame_area_ratio_to_bpoly": float(area_ratio),
            "length_ratio": 1.0,
            "extra_coastline_intersection": False,
            "extra_intersection_length_m": 0.0,
        }
        scored.append(item)
    scored.sort(key=lambda obj: obj["score"], reverse=True)
    return {"selected": scored[0], "candidates": scored}


def score_and_select_bdry_arc(
    candidates: list[dict[str, Any]],
    coast_union_xy,
    bpoly_xy: Polygon,
    target_resolution_m: float,
) -> dict[str, Any]:
    """Score offshore arcs and select the best candidate."""
    scored: list[dict[str, Any]] = []
    endpoint_mask_factor = max(3.0 * target_resolution_m, 500.0)
    for candidate in candidates:
        line = candidate["geometry"]
        coords = list(line.coords)
        endpoint_mask = Point(coords[0]).buffer(endpoint_mask_factor).union(Point(coords[-1]).buffer(endpoint_mask_factor))
        inspect_line = line.difference(endpoint_mask)
        extra_intersection = False
        extra_intersection_length_m = 0.0
        if coast_union_xy is not None and not coast_union_xy.is_empty and not inspect_line.is_empty:
            inter = inspect_line.intersection(coast_union_xy.buffer(max(2.0, 0.02 * target_resolution_m)))
            extra_intersection = not inter.is_empty
            extra_intersection_length_m = float(getattr(inter, "length", 0.0))
        samples = _sample_line_points(line, max(line.length / 60.0, target_resolution_m))
        inside_fraction = sum(1 for pt in samples if bpoly_xy.buffer(target_resolution_m).contains(pt)) / max(len(samples), 1)
        chord = Point(coords[0]).distance(Point(coords[-1]))
        length_ratio = float(line.length / max(chord, 1.0))
        score = 100.0 * inside_fraction
        score += min(float(candidate["bow_distance_m"] / max(chord, 1.0)), 2.0) * 10.0
        score -= max(length_ratio - 1.8, 0.0) * 20.0
        if extra_intersection:
            score -= 80.0 + min(extra_intersection_length_m / max(target_resolution_m, 1.0), 80.0)
        item = dict(candidate)
        item["score"] = float(score)
        item["metrics"] = {
            "inside_bpoly_fraction": float(inside_fraction),
            "length_ratio": float(length_ratio),
            "extra_coastline_intersection": bool(extra_intersection),
            "extra_intersection_length_m": float(extra_intersection_length_m),
        }
        scored.append(item)
    scored.sort(key=lambda obj: obj["score"], reverse=True)
    return {"selected": scored[0], "candidates": scored}


def extract_seeded_wet_domain(
    coastline_lines_xy: list[LineString],
    offshore_arc_xy: LineString,
    bpoly_xy: Polygon,
    seed_xy: Point,
    forbidden_regions_xy: list[Polygon],
    target_resolution_m: float,
) -> dict[str, Any]:
    """Build a seed-selected wet-domain polygon from linework and arc."""
    source = "coastline_arc_polygonize"
    fallback_reason = None
    faces: list[Polygon] = []
    if len(coastline_lines_xy) > 30_000:
        fallback_reason = f"line_count_exceeds_v1_polygonize_limit: {len(coastline_lines_xy)}"
        faces = []
    else:
        try:
            linework = unary_union(coastline_lines_xy + [offshore_arc_xy])
            faces = [poly.buffer(0) for poly in polygonize(linework) if isinstance(poly, Polygon) and not poly.is_empty]
        except Exception as exc:
            fallback_reason = f"polygonize_failed: {exc}"
            faces = []

    chosen = _choose_seed_face(faces, seed_xy, bpoly_xy)
    if chosen is None and len(coastline_lines_xy) <= 30_000:
        try:
            framed = unary_union(coastline_lines_xy + [offshore_arc_xy, bpoly_xy.boundary])
            faces = [poly.buffer(0) for poly in polygonize(framed) if isinstance(poly, Polygon) and not poly.is_empty]
            chosen = _choose_seed_face(faces, seed_xy, bpoly_xy)
            source = "framed_polygonize"
        except Exception as exc:
            fallback_reason = f"framed_polygonize_failed: {exc}"
    if chosen is None:
        chosen = bpoly_xy.buffer(0)
        source = "fallback_bpoly_polygon"
        fallback_reason = fallback_reason or "no_polygonized_face_contains_seed"

    if not chosen.is_valid:
        chosen = chosen.buffer(0)
    if not chosen.intersects(bpoly_xy):
        chosen = chosen.intersection(bpoly_xy).buffer(0)
    if chosen.geom_type == "MultiPolygon":
        chosen = max(chosen.geoms, key=lambda geom: geom.area)

    min_hole_area = max((2.0 * target_resolution_m) ** 2, 1.0)
    kept_holes = []
    for ring in getattr(chosen, "interiors", []):
        hole = Polygon(ring)
        if hole.area >= min_hole_area and not hole.contains(seed_xy):
            kept_holes.append(ring.coords)
    domain = Polygon(chosen.exterior.coords, kept_holes).buffer(0)

    forbidden_overlap = []
    for idx, forbidden in enumerate(forbidden_regions_xy):
        overlap = domain.intersection(forbidden)
        if not overlap.is_empty and overlap.area > min_hole_area:
            forbidden_overlap.append({"index": idx, "area_m2": float(overlap.area)})

    metadata = {
        "source": source,
        "fallback_reason": fallback_reason,
        "face_count": int(len(faces)),
        "area_m2": float(domain.area),
        "perimeter_m": float(domain.length),
        "hole_count": int(len(getattr(domain, "interiors", []))),
        "seed_inside": bool(domain.buffer(max(1.0, 0.1 * target_resolution_m)).contains(seed_xy)),
        "forbidden_overlap": forbidden_overlap,
        "target_resolution_m": float(target_resolution_m),
        "method": "vector_polygonize_with_seeded_face_selection",
        "polygonize_limit": 30000,
    }
    return {"wet_domain_xy": domain, "metadata": metadata}


def select_gshhs_open_side_topology(
    scored: dict[str, Any],
    coastline_lines_xy: list[LineString],
    land_polygons_xy: list[Polygon],
    bpoly_xy: Polygon,
    seed_xy: Point,
    target_resolution_m: float,
    anchors: dict[str, Any] | None = None,
    run_dir: Path | None = None,
    topology_time_budget_s: float | None = None,
    obc_placement_policy: str = "offshore-first",
) -> dict[str, Any]:
    """Select the coastline-anchor seaward-chain deformation that creates the best domain."""
    evaluated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    start_time = time.monotonic()
    budget_exceeded = False
    if run_dir is not None:
        _write_progress(run_dir, "gshhs-vector", "land-union-start", {"land_polygon_count": len(land_polygons_xy)})
    land_union_xy = unary_union(land_polygons_xy).buffer(0) if land_polygons_xy else GeometryCollection()
    if run_dir is not None:
        _write_progress(run_dir, "gshhs-vector", "land-union-done", {"land_union_empty": land_union_xy.is_empty})
    for idx, candidate in enumerate(scored["candidates"], start=1):
        if run_dir is not None:
            _write_progress(
                run_dir,
                "gshhs-vector",
                "candidate-start",
                {"candidate_id": candidate.get("candidate_id"), "candidate_index": idx, "candidate_count": len(scored["candidates"])},
            )
        result = extract_gshhs_vector_wet_domain(
            coastline_lines_xy,
            land_polygons_xy,
            candidate["geometry"],
            bpoly_xy,
            seed_xy,
            target_resolution_m,
            anchors=anchors,
            land_union_xy=land_union_xy,
            obc_placement_policy=obc_placement_policy,
        )
        metadata = result["metadata"]
        metrics = dict(candidate.get("metrics", {}))
        metrics.update(
            {
                "closure_method": metadata.get("closure_method"),
                "open_arc_boundary_overlap_fraction": metadata.get("open_arc_boundary_overlap_fraction", 0.0),
                "land_boundary_overlap_fraction": metadata.get("land_boundary_overlap_fraction", 0.0),
                "frame_clip_boundary_length_m": metadata.get("frame_clip_boundary_length_m", 0.0),
                "deformed_frame_area_m2": metadata.get("deformed_frame_area_m2", 0.0),
            }
        )
        coords = list(candidate["geometry"].coords)
        chord = max(Point(coords[0]).distance(Point(coords[-1])), 1.0)
        bow_ratio = float(candidate.get("bow_distance_m", 0.0)) / chord
        topology_score = 0.0
        topology_score += 160.0 * float(metadata.get("open_arc_boundary_overlap_fraction", 0.0))
        if metadata.get("seed_inside"):
            topology_score += 140.0
        if metadata.get("deformed_frame_valid"):
            topology_score += 80.0
        topology_score += 35.0 * min(max(bow_ratio, 0.0) / 0.20, 1.0)
        topology_score -= max(0.98 - float(metadata.get("open_arc_boundary_overlap_fraction", 0.0)), 0.0) * 600.0
        topology_score -= max(float(metrics.get("length_ratio", 1.0)) - 1.25, 0.0) * 100.0
        if metadata.get("arc_land_intersection"):
            topology_score -= 180.0
            topology_score -= min(float(metadata.get("arc_land_intersection_length_m", 0.0)) / max(target_resolution_m, 1.0), 100.0) * 5.0
        if metadata.get("gshhs_missing_land_polygons"):
            topology_score -= 120.0

        item = dict(candidate)
        item["geometry"] = result.get("open_arc_xy", candidate["geometry"])
        item["metrics"] = metrics
        item["score"] = float(topology_score)
        result["metadata"]["candidate_id"] = candidate.get("candidate_id")
        evaluated.append((item, result))
        if run_dir is not None:
            _write_progress(
                run_dir,
                "gshhs-vector",
                "candidate-done",
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate_index": idx,
                    "candidate_count": len(scored["candidates"]),
                    "open_arc_boundary_overlap_fraction": metadata.get("open_arc_boundary_overlap_fraction", 0.0),
                    "seed_inside": metadata.get("seed_inside"),
                },
            )
        if topology_time_budget_s and topology_time_budget_s > 0 and time.monotonic() - start_time > topology_time_budget_s and evaluated:
            budget_exceeded = True
            if run_dir is not None:
                _write_progress(
                    run_dir,
                    "gshhs-vector",
                    "budget-exceeded",
                    {
                        "elapsed_seconds": time.monotonic() - start_time,
                        "topology_time_budget_s": topology_time_budget_s,
                        "evaluated_candidate_count": len(evaluated),
                    },
                )
            break

    evaluated.sort(key=lambda pair: pair[0]["score"], reverse=True)
    selected, wet_result = evaluated[0]
    if budget_exceeded:
        wet_result["metadata"]["topology_budget_exceeded"] = True
        wet_result["metadata"]["topology_time_budget_s"] = float(topology_time_budget_s or 0.0)
    candidates = [item for item, _result in evaluated]
    return {"scored": {"selected": selected, "candidates": candidates}, "wet_result": wet_result}


def select_island_loop_topology(
    scored: dict[str, Any],
    land_polygons_xy: list[Polygon],
    bpoly_xy: Polygon,
    seed_xy: Point,
    target_resolution_m: float,
    run_dir: Path | None = None,
    topology_time_budget_s: float | None = None,
) -> dict[str, Any]:
    """Select a closed island/archipelago offshore loop and water component."""
    evaluated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    start_time = time.monotonic()
    budget_exceeded = False
    if run_dir is not None:
        _write_progress(run_dir, "island-loop", "land-union-start", {"land_polygon_count": len(land_polygons_xy)})
    land_union_xy = unary_union(land_polygons_xy).buffer(0) if land_polygons_xy else GeometryCollection()
    if run_dir is not None:
        _write_progress(run_dir, "island-loop", "land-union-done", {"land_union_empty": land_union_xy.is_empty})
    for idx, candidate in enumerate(scored["candidates"], start=1):
        if run_dir is not None:
            _write_progress(
                run_dir,
                "island-loop",
                "candidate-start",
                {"candidate_id": candidate.get("candidate_id"), "candidate_index": idx, "candidate_count": len(scored["candidates"])},
            )
        result = extract_island_loop_wet_domain(
            land_polygons_xy,
            candidate["geometry"],
            bpoly_xy,
            seed_xy,
            target_resolution_m,
            land_union_xy=land_union_xy,
        )
        metadata = result["metadata"]
        metrics = dict(candidate.get("metrics", {}))
        metrics.update(
            {
                "closure_method": metadata.get("closure_method"),
                "open_arc_boundary_overlap_fraction": metadata.get("open_arc_boundary_overlap_fraction", 0.0),
                "land_patch_boundary_length_m": metadata.get("land_patch_boundary_length_m", 0.0),
                "frame_area_m2": metadata.get("deformed_frame_area_m2", 0.0),
            }
        )
        topology_score = float(candidate.get("score", 0.0))
        topology_score += 160.0 * float(metadata.get("open_arc_boundary_overlap_fraction", 0.0))
        if metadata.get("seed_inside"):
            topology_score += 140.0
        if metadata.get("deformed_frame_valid"):
            topology_score += 80.0
        patch_ratio = float(metadata.get("land_patch_boundary_fraction", 0.0))
        topology_score -= min(patch_ratio * 180.0, 120.0)
        if metadata.get("gshhs_missing_land_polygons"):
            topology_score -= 120.0
        item = dict(candidate)
        item["metrics"] = metrics
        item["score"] = float(topology_score)
        result["metadata"]["candidate_id"] = candidate.get("candidate_id")
        evaluated.append((item, result))
        if run_dir is not None:
            _write_progress(
                run_dir,
                "island-loop",
                "candidate-done",
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate_index": idx,
                    "candidate_count": len(scored["candidates"]),
                    "open_arc_boundary_overlap_fraction": metadata.get("open_arc_boundary_overlap_fraction", 0.0),
                    "land_patch_boundary_length_m": metadata.get("land_patch_boundary_length_m", 0.0),
                    "seed_inside": metadata.get("seed_inside"),
                },
            )
        if topology_time_budget_s and topology_time_budget_s > 0 and time.monotonic() - start_time > topology_time_budget_s and evaluated:
            budget_exceeded = True
            if run_dir is not None:
                _write_progress(
                    run_dir,
                    "island-loop",
                    "budget-exceeded",
                    {
                        "elapsed_seconds": time.monotonic() - start_time,
                        "topology_time_budget_s": topology_time_budget_s,
                        "evaluated_candidate_count": len(evaluated),
                    },
                )
            break
    evaluated.sort(key=lambda pair: pair[0]["score"], reverse=True)
    selected, wet_result = evaluated[0]
    if budget_exceeded:
        wet_result["metadata"]["topology_budget_exceeded"] = True
        wet_result["metadata"]["topology_time_budget_s"] = float(topology_time_budget_s or 0.0)
    candidates = [item for item, _result in evaluated]
    return {"scored": {"selected": selected, "candidates": candidates}, "wet_result": wet_result}


def extract_lake_closed_wet_domain(
    land_polygons_xy: list[Polygon],
    bpoly_xy: Polygon,
    seed_xy: Point,
    target_resolution_m: float,
) -> dict[str, Any]:
    """Build a closed lake-domain water component without any ocean open arc."""
    land_union = unary_union(land_polygons_xy).buffer(0) if land_polygons_xy else GeometryCollection()
    land_boundary = land_union.boundary if not land_union.is_empty else GeometryCollection()
    base_water = bpoly_xy.difference(land_union).buffer(0) if not land_union.is_empty else bpoly_xy.buffer(0)
    if not base_water.is_valid:
        base_water = base_water.buffer(0)
    water_seed_xy = seed_xy
    seed_snap_distance_m = 0.0
    seed_snapped = False
    if not base_water.is_empty and not base_water.buffer(1.0).contains(seed_xy):
        try:
            water_seed_xy = nearest_points(seed_xy, base_water)[1]
            seed_snap_distance_m = float(seed_xy.distance(water_seed_xy))
            seed_snapped = seed_snap_distance_m > 0.0
        except Exception:
            water_seed_xy = seed_xy
    domain = _choose_seed_component(base_water, water_seed_xy)
    fallback_reason: str | None = None
    if domain is None or domain.is_empty:
        fallback_reason = "seed_water_component_not_found"
        domain = bpoly_xy.buffer(0)
    if not domain.is_valid:
        domain = domain.buffer(0)
    if getattr(domain, "geom_type", "") != "Polygon":
        selected = _choose_seed_component(domain, water_seed_xy)
        domain = selected if selected is not None else bpoly_xy.buffer(0)

    open_arc = LineString()
    boundary_segments = _classify_domain_boundary_segments(
        domain,
        open_arc,
        land_boundary,
        bpoly_xy,
        target_resolution_m,
    )
    metadata = {
        "target_resolution_m": float(target_resolution_m),
        "source": "lake_closed_boundary_no_open_arc",
        "closure_method": "lake_closed_boundary_no_open_arc",
        "method": "lake_bpoly_minus_gshhs_land_union",
        "boundary_policy": "no_open_boundary",
        "no_ocean_open_boundary": True,
        "deformed_frame_valid": bool(bpoly_xy.is_valid and not bpoly_xy.is_empty),
        "deformed_frame_area_m2": float(bpoly_xy.area),
        "land_polygon_count": int(len(land_polygons_xy)),
        "coastline_line_count": 0,
        "gshhs_missing_land_polygons": bool(len(land_polygons_xy) == 0),
        "gshhs_missing_coastline_lines": False,
        "gshhs_polygonize_fallback_used": False,
        "fallback_reason": fallback_reason,
        "arc_land_intersection": False,
        "arc_land_intersection_length_m": 0.0,
        "open_arc_boundary_overlap_fraction": None,
        "land_boundary_overlap_fraction": float(boundary_segments["land_boundary_overlap_fraction"]),
        "frame_clip_boundary_length_m": float(boundary_segments["frame_clip_boundary_length_m"]),
        "area_m2": float(domain.area),
        "perimeter_m": float(domain.length),
        "hole_count": int(len(getattr(domain, "interiors", []))),
        "seed_inside": bool(domain.buffer(max(1.0, 0.1 * target_resolution_m)).contains(water_seed_xy)),
        "original_seed_inside": bool(domain.buffer(max(1.0, 0.1 * target_resolution_m)).contains(seed_xy)),
        "seed_snapped_to_gshhs_water": bool(seed_snapped),
        "seed_snap_distance_m": float(seed_snap_distance_m),
        "forbidden_overlap": [],
        "face_count": int(_polygon_count(base_water)),
    }
    return {
        "wet_domain_xy": domain,
        "open_arc_xy": open_arc,
        "deformed_frame_xy": bpoly_xy,
        "boundary_segments_xy": boundary_segments,
        "metadata": metadata,
    }


def extract_gshhs_vector_wet_domain(
    coastline_lines_xy: list[LineString],
    land_polygons_xy: list[Polygon],
    offshore_arc_xy: LineString,
    bpoly_xy: Polygon,
    seed_xy: Point,
    target_resolution_m: float,
    anchors: dict[str, Any] | None = None,
    land_union_xy=None,
    obc_placement_policy: str = "offshore-first",
) -> dict[str, Any]:
    """Build a GSHHS-first wet domain from a coastline-anchor deformed frame."""
    land_union = land_union_xy if land_union_xy is not None else (unary_union(land_polygons_xy).buffer(0) if land_polygons_xy else GeometryCollection())
    land_boundary = land_union.boundary if not land_union.is_empty else GeometryCollection()
    deformed_frame, frame_meta = _deformed_bpoly_frame(bpoly_xy, offshore_arc_xy, anchors)
    base_water = deformed_frame.difference(land_union).buffer(0) if not land_union.is_empty else deformed_frame.buffer(0)
    if not base_water.is_valid:
        base_water = base_water.buffer(0)

    water_seed_xy = seed_xy
    seed_snap_distance_m = 0.0
    seed_snapped = False
    if not base_water.is_empty:
        if not base_water.buffer(1.0).contains(seed_xy):
            try:
                water_seed_xy = nearest_points(seed_xy, base_water)[1]
                seed_snap_distance_m = float(seed_xy.distance(water_seed_xy))
                seed_snapped = seed_snap_distance_m > 0.0
            except Exception:
                water_seed_xy = seed_xy

    domain = _choose_seed_component(base_water, water_seed_xy)
    fallback_reason: str | None = None
    source = "coastline_anchor_seaward_bpoly_chain"
    if domain is None or domain.is_empty:
        source = "coastline_anchor_seed_water_component_not_found"
        fallback_reason = "seed_water_component_not_found"
        domain = deformed_frame.buffer(0)

    if not domain.is_valid:
        domain = domain.buffer(0)
    if getattr(domain, "geom_type", "") != "Polygon":
        selected = _choose_seed_component(domain, water_seed_xy)
        domain = selected if selected is not None else deformed_frame.buffer(0)

    source_open_arc_xy = offshore_arc_xy
    source_exterior_overlap = _line_fraction_near_boundary(
        source_open_arc_xy, domain.boundary, max(2.0, 0.02 * target_resolution_m)
    )
    use_complete_offshore = bool(
        obc_placement_policy == "offshore-first"
        and source_open_arc_xy.is_simple
        and source_exterior_overlap >= 0.999
    )
    if use_complete_offshore:
        delivered_open_arc_xy = source_open_arc_xy
        trim_report = {
            "open_arc_trimmed_to_wet_exterior": False,
            "open_arc_trim_reason": "complete_offshore_arc_owns_model_exterior",
            "source_open_arc_length_m": float(source_open_arc_xy.length),
            "delivered_open_arc_length_m": float(source_open_arc_xy.length),
            "discarded_source_open_arc_length_m": 0.0,
            "delivered_source_start_position_m": 0.0,
            "delivered_source_end_position_m": float(source_open_arc_xy.length),
        }
    else:
        delivered_open_arc_xy, trim_report = _normalize_open_arc_to_wet_exterior(
            source_open_arc_xy,
            domain,
            land_boundary,
            target_resolution_m,
        )
    trim_report["obc_placement_policy"] = obc_placement_policy
    trim_report["obc_placement_family"] = (
        "complete-offshore" if use_complete_offshore else "compact-mouth-fallback"
        if obc_placement_policy == "offshore-first" else "compact-mouth"
    )
    trim_report["source_open_arc_exterior_overlap_fraction"] = float(source_exterior_overlap)

    tolerance = max(2.0, 0.02 * target_resolution_m)
    source_arc_land = source_open_arc_xy.difference(
        Point(source_open_arc_xy.coords[0]).buffer(max(500.0, 3.0 * target_resolution_m)).union(
            Point(source_open_arc_xy.coords[-1]).buffer(max(500.0, 3.0 * target_resolution_m))
        )
    )
    source_arc_land_intersection = False
    source_arc_land_intersection_length_m = 0.0
    if not land_union.is_empty and not source_arc_land.is_empty:
        inter = source_arc_land.intersection(land_union.buffer(tolerance))
        source_arc_land_intersection = not inter.is_empty
        source_arc_land_intersection_length_m = float(getattr(inter, "length", 0.0))

    delivered_arc_land = delivered_open_arc_xy.difference(
        Point(delivered_open_arc_xy.coords[0]).buffer(max(500.0, 3.0 * target_resolution_m)).union(
            Point(delivered_open_arc_xy.coords[-1]).buffer(max(500.0, 3.0 * target_resolution_m))
        )
    )
    arc_land_intersection = False
    arc_land_intersection_length_m = 0.0
    if not land_union.is_empty and not delivered_arc_land.is_empty:
        inter = delivered_arc_land.intersection(land_union.buffer(tolerance))
        arc_land_intersection = not inter.is_empty
        arc_land_intersection_length_m = float(getattr(inter, "length", 0.0))

    boundary_segments = _classify_domain_boundary_segments(
        domain,
        delivered_open_arc_xy,
        land_boundary,
        deformed_frame,
        target_resolution_m,
    )
    open_overlap_fraction = _line_fraction_near_boundary(
        delivered_open_arc_xy,
        domain.boundary,
        max(tolerance, 0.1 * target_resolution_m),
    )
    source_open_overlap_fraction = _line_fraction_near_boundary(
        source_open_arc_xy,
        domain.boundary,
        max(tolerance, 0.1 * target_resolution_m),
    )

    metadata: dict[str, Any] = {
        "target_resolution_m": float(target_resolution_m),
        "face_count": int(_polygon_count(base_water)),
    }
    metadata.update(
        {
            "source": source,
            "closure_method": "coastline_anchor_seaward_bpoly_chain",
            "method": "coastline_anchor_seaward_bpoly_chain_minus_gshhs_land_union",
            **frame_meta,
            "land_polygon_count": int(len(land_polygons_xy)),
            "coastline_line_count": int(len(coastline_lines_xy)),
            "gshhs_missing_land_polygons": bool(len(land_polygons_xy) == 0),
            "gshhs_missing_coastline_lines": bool(len(coastline_lines_xy) == 0),
            "gshhs_polygonize_fallback_used": False,
            "fallback_reason": fallback_reason,
            "arc_land_intersection": bool(arc_land_intersection),
            "arc_land_intersection_length_m": float(arc_land_intersection_length_m),
            "source_arc_land_intersection": bool(source_arc_land_intersection),
            "source_arc_land_intersection_length_m": float(source_arc_land_intersection_length_m),
            "open_arc_boundary_overlap_fraction": float(open_overlap_fraction),
            "source_open_arc_boundary_overlap_fraction": float(source_open_overlap_fraction),
            **trim_report,
            "land_boundary_overlap_fraction": float(boundary_segments["land_boundary_overlap_fraction"]),
            "frame_clip_boundary_length_m": float(boundary_segments["frame_clip_boundary_length_m"]),
            "area_m2": float(domain.area),
            "perimeter_m": float(domain.length),
            "hole_count": int(len(getattr(domain, "interiors", []))),
            "seed_inside": bool(domain.buffer(max(1.0, 0.1 * target_resolution_m)).contains(water_seed_xy)),
            "original_seed_inside": bool(domain.buffer(max(1.0, 0.1 * target_resolution_m)).contains(seed_xy)),
            "seed_snapped_to_gshhs_water": bool(seed_snapped),
            "seed_snap_distance_m": float(seed_snap_distance_m),
            "forbidden_overlap": [],
        }
    )
    return {
        "wet_domain_xy": domain,
        "open_arc_xy": delivered_open_arc_xy,
        "deformed_frame_xy": deformed_frame,
        "boundary_segments_xy": boundary_segments,
        "metadata": metadata,
    }


def extract_island_loop_wet_domain(
    land_polygons_xy: list[Polygon],
    offshore_loop_xy: LineString,
    bpoly_xy: Polygon,
    seed_xy: Point,
    target_resolution_m: float,
    land_union_xy=None,
) -> dict[str, Any]:
    """Build an island/archipelago water domain from a closed offshore loop."""
    loop_xy = _ensure_closed_line(offshore_loop_xy)
    frame = Polygon(loop_xy.coords).buffer(0)
    frame_valid = isinstance(frame, Polygon) and not frame.is_empty and frame.is_valid
    if not frame_valid:
        frame = bpoly_xy.buffer(0)
    land_union = land_union_xy if land_union_xy is not None else (unary_union(land_polygons_xy).buffer(0) if land_polygons_xy else GeometryCollection())
    land_boundary = land_union.boundary if not land_union.is_empty else GeometryCollection()
    base_water = frame.difference(land_union).buffer(0) if not land_union.is_empty else frame.buffer(0)
    if not base_water.is_valid:
        base_water = base_water.buffer(0)

    water_seed_xy = seed_xy
    seed_snap_distance_m = 0.0
    seed_snapped = False
    if not base_water.is_empty and not base_water.buffer(1.0).contains(seed_xy):
        try:
            water_seed_xy = nearest_points(seed_xy, base_water)[1]
            seed_snap_distance_m = float(seed_xy.distance(water_seed_xy))
            seed_snapped = seed_snap_distance_m > 0.0
        except Exception:
            water_seed_xy = seed_xy

    domain = _choose_seed_component(base_water, water_seed_xy)
    fallback_reason: str | None = None
    if domain is None or domain.is_empty:
        fallback_reason = "seed_water_component_not_found"
        domain = frame.buffer(0)
    if not domain.is_valid:
        domain = domain.buffer(0)
    if getattr(domain, "geom_type", "") != "Polygon":
        selected = _choose_seed_component(domain, water_seed_xy)
        domain = selected if selected is not None else frame.buffer(0)

    tolerance = max(2.0, 0.02 * target_resolution_m)
    land_patch_geom = GeometryCollection()
    land_patch_lines: list[LineString] = []
    if not land_union.is_empty:
        try:
            land_patch_geom = loop_xy.intersection(land_union.buffer(tolerance))
            land_patch_lines = _line_parts(land_patch_geom)
        except Exception:
            land_patch_geom = GeometryCollection()
            land_patch_lines = []
    land_patch_length = _sum_line_length(land_patch_lines)
    loop_length = max(float(loop_xy.length), 1.0)
    boundary_segments = _classify_domain_boundary_segments(
        domain,
        loop_xy,
        land_boundary,
        frame,
        target_resolution_m,
    )
    boundary_segments["land_patch_boundary_arcs_xy"] = land_patch_lines
    boundary_segments["land_patch_boundary_length_m"] = float(land_patch_length)
    open_overlap_fraction = _line_fraction_near_boundary(loop_xy, domain.boundary, max(tolerance, 0.1 * target_resolution_m))
    metadata = {
        "target_resolution_m": float(target_resolution_m),
        "face_count": int(_polygon_count(base_water)),
        "source": "island_archipelago_offshore_loop",
        "closure_method": "island_archipelago_offshore_loop",
        "method": "closed_offshore_loop_minus_gshhs_land_union",
        "deformed_frame_valid": bool(frame_valid),
        "deformed_frame_area_m2": float(frame.area),
        "land_polygon_count": int(len(land_polygons_xy)),
        "coastline_line_count": 0,
        "gshhs_missing_land_polygons": bool(len(land_polygons_xy) == 0),
        "gshhs_missing_coastline_lines": False,
        "gshhs_polygonize_fallback_used": False,
        "fallback_reason": fallback_reason,
        "arc_land_intersection": bool(land_patch_length > 0.0),
        "arc_land_intersection_length_m": float(land_patch_length),
        "land_patch_policy": "land_patch",
        "island_blocker_land_patch_used": bool(land_patch_length > 0.0),
        "land_patch_boundary_length_m": float(land_patch_length),
        "land_patch_boundary_fraction": float(land_patch_length / loop_length),
        "open_arc_boundary_overlap_fraction": float(open_overlap_fraction),
        "land_boundary_overlap_fraction": float(boundary_segments["land_boundary_overlap_fraction"]),
        "frame_clip_boundary_length_m": float(boundary_segments["frame_clip_boundary_length_m"]),
        "area_m2": float(domain.area),
        "perimeter_m": float(domain.length),
        "hole_count": int(len(getattr(domain, "interiors", []))),
        "seed_inside": bool(domain.buffer(max(1.0, 0.1 * target_resolution_m)).contains(water_seed_xy)),
        "original_seed_inside": bool(domain.buffer(max(1.0, 0.1 * target_resolution_m)).contains(seed_xy)),
        "seed_snapped_to_gshhs_water": bool(seed_snapped),
        "seed_snap_distance_m": float(seed_snap_distance_m),
        "forbidden_overlap": [],
    }
    return {
        "wet_domain_xy": domain,
        "open_arc_xy": loop_xy,
        "deformed_frame_xy": frame,
        "boundary_segments_xy": boundary_segments,
        "metadata": metadata,
    }


def _deformed_bpoly_frame(
    bpoly_xy: Polygon,
    open_arc_xy: LineString,
    anchors: dict[str, Any] | None = None,
) -> tuple[Polygon, dict[str, Any]]:
    coords = list(bpoly_xy.exterior.coords)[:-1]
    arc_coords = [(float(x), float(y)) for x, y in open_arc_xy.coords]
    start = Point(arc_coords[0])
    end = Point(arc_coords[-1])
    if anchors and anchors.get("source") == "coastline_bpoly_intersection":
        selected_side_xy = anchors.get("selected_side_xy")
        frame, meta = _coastline_anchor_deformed_frame(
            coords,
            arc_coords,
            start,
            end,
            anchors,
            selected_side_xy if isinstance(selected_side_xy, LineString) else None,
        )
        if frame is not None:
            return frame, meta

    n = len(coords)
    start_idx = min(range(n), key=lambda idx: Point(coords[idx]).distance(start))
    end_idx = min(range(n), key=lambda idx: Point(coords[idx]).distance(end))
    start_distance = float(Point(coords[start_idx]).distance(start))
    end_distance = float(Point(coords[end_idx]).distance(end))
    orientation = "unresolved"
    ring: list[tuple[float, float]]
    if (start_idx + 1) % n == end_idx:
        orientation = "forward"
        ring = list(arc_coords)
        idx = end_idx
        while True:
            idx = (idx + 1) % n
            if idx == start_idx:
                break
            ring.append(tuple(coords[idx]))
    elif (end_idx + 1) % n == start_idx:
        orientation = "reverse"
        ring = list(reversed(arc_coords))
        idx = start_idx
        while True:
            idx = (idx + 1) % n
            if idx == end_idx:
                break
            ring.append(tuple(coords[idx]))
    else:
        ring = list(arc_coords) + [tuple(coord) for coord in coords if Point(coord).distance(start) > 1.0 and Point(coord).distance(end) > 1.0]

    frame = Polygon(ring)
    if not frame.is_valid:
        frame = frame.buffer(0)
    if getattr(frame, "geom_type", "") != "Polygon":
        frame = bpoly_xy.buffer(0)
        orientation = "fallback_original_bpoly"
    return frame, {
        "deformed_frame_valid": bool(frame.is_valid and not frame.is_empty),
        "deformed_frame_area_m2": float(frame.area),
        "deformed_frame_side_orientation": orientation,
        "open_arc_start_to_bpoly_vertex_m": start_distance,
        "open_arc_end_to_bpoly_vertex_m": end_distance,
    }


def _coastline_anchor_deformed_frame(
    coords: list[tuple[float, float]],
    arc_coords: list[tuple[float, float]],
    start: Point,
    end: Point,
    anchors: dict[str, Any],
    selected_side_xy: LineString | None,
) -> tuple[Polygon, dict[str, Any]] | tuple[None, dict[str, Any]]:
    records = _boundary_records_with_anchors(coords, anchors)
    if not records:
        return None, {}
    start_idx = next((idx for idx, item in enumerate(records) if item.get("label") == "start_anchor"), None)
    end_idx = next((idx for idx, item in enumerate(records) if item.get("label") == "end_anchor"), None)
    if start_idx is None or end_idx is None:
        return None, {}

    path_a = _boundary_path(records, start_idx, end_idx)
    path_b = _boundary_path(records, end_idx, start_idx)
    path_b_start_to_end = LineString(list(reversed(path_b.coords))) if len(path_b.coords) >= 2 else path_b
    selected_side = selected_side_xy or LineString([anchors["selected_side_start_corner_xy"], anchors["selected_side_end_corner_xy"]])
    tolerance = max(1.0, float(anchors.get("anchor_tolerance_m", 1.0)))
    overlap_a = _line_fraction_near_boundary(selected_side, path_a, tolerance)
    overlap_b = _line_fraction_near_boundary(selected_side, path_b_start_to_end, tolerance)
    if overlap_a <= overlap_b:
        landward_path_start_to_end = path_a
        orientation = "forward_landward_path"
        selected_side_overlap = overlap_a
    else:
        landward_path_start_to_end = path_b_start_to_end
        orientation = "reverse_landward_path"
        selected_side_overlap = overlap_b

    landward_end_to_start = list(reversed(list(landward_path_start_to_end.coords)))
    ring = _dedupe_consecutive_coords(list(arc_coords) + landward_end_to_start[1:])
    frame = Polygon(ring)
    if not frame.is_valid:
        frame = frame.buffer(0)
    if getattr(frame, "geom_type", "") != "Polygon" or frame.is_empty:
        return None, {}

    start_corner = Point(anchors["selected_side_start_corner_xy"])
    end_corner = Point(anchors["selected_side_end_corner_xy"])
    return frame, {
        "deformed_frame_valid": bool(frame.is_valid and not frame.is_empty),
        "deformed_frame_area_m2": float(frame.area),
        "deformed_frame_side_orientation": orientation,
        "landward_path_selected_side_overlap_fraction": float(selected_side_overlap),
        "open_arc_start_to_bpoly_vertex_m": float(start.distance(start_corner)),
        "open_arc_end_to_bpoly_vertex_m": float(end.distance(end_corner)),
        "seaward_chain_vertex_count": int(len(anchors.get("seaward_chain_xy", []))),
    }


def _boundary_records_with_anchors(
    coords: list[tuple[float, float]],
    anchors: dict[str, Any],
) -> list[dict[str, Any]]:
    n = len(coords)
    inserts: dict[int, list[dict[str, Any]]] = {}
    for label, side_key, xy_key in (
        ("start_anchor", "start_adjacent_side_index", "start_xy"),
        ("end_anchor", "end_adjacent_side_index", "end_xy"),
    ):
        side_idx = anchors.get(side_key)
        if side_idx is None:
            continue
        side_idx = int(side_idx) % n
        xy = tuple(float(v) for v in anchors[xy_key])
        seg = LineString([coords[side_idx], coords[(side_idx + 1) % n]])
        fraction = float(seg.project(Point(xy)) / max(seg.length, 1.0))
        inserts.setdefault(side_idx, []).append({"label": label, "coord": xy, "fraction": fraction})

    records: list[dict[str, Any]] = []
    for idx, coord in enumerate(coords):
        records.append({"label": f"vertex_{idx}", "coord": (float(coord[0]), float(coord[1]))})
        for item in sorted(inserts.get(idx, []), key=lambda obj: obj["fraction"]):
            records.append({"label": item["label"], "coord": item["coord"]})
    return records


def _boundary_path(records: list[dict[str, Any]], start_idx: int, end_idx: int) -> LineString:
    coords = [records[start_idx]["coord"]]
    idx = start_idx
    while idx != end_idx:
        idx = (idx + 1) % len(records)
        coords.append(records[idx]["coord"])
    return LineString(_dedupe_consecutive_coords(coords))


def _dedupe_consecutive_coords(coords: list[tuple[float, float]], tolerance: float = 1.0e-9) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for coord in coords:
        xy = (float(coord[0]), float(coord[1]))
        if not out or math.hypot(out[-1][0] - xy[0], out[-1][1] - xy[1]) > tolerance:
            out.append(xy)
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= tolerance:
        out.pop()
    return out


def _ensure_closed_line(line: LineString) -> LineString:
    coords = [(float(x), float(y)) for x, y in line.coords]
    if not coords:
        return line
    if Point(coords[0]).distance(Point(coords[-1])) > 1.0e-9:
        coords.append(coords[0])
    return LineString(coords)


def _chaikin_closed_ring(coords: list[tuple[float, float]], iterations: int = 2) -> list[tuple[float, float]]:
    ring = [(float(x), float(y)) for x, y in coords]
    if len(ring) < 3:
        return ring
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    for _ in range(max(0, int(iterations))):
        new_ring: list[tuple[float, float]] = []
        for idx, p0 in enumerate(ring):
            p1 = ring[(idx + 1) % len(ring)]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_ring.extend([q, r])
        ring = new_ring
    ring.append(ring[0])
    return ring


def _polygon_count(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    if isinstance(geom, Polygon):
        return 1
    if hasattr(geom, "geoms"):
        return sum(1 for part in geom.geoms if isinstance(part, Polygon) and not part.is_empty)
    return 0


def _line_fraction_near_boundary(line: LineString, boundary, tolerance_m: float) -> float:
    if line is None or line.is_empty or float(line.length) <= 0.0 or boundary is None or boundary.is_empty:
        return 0.0
    try:
        near = line.intersection(boundary.buffer(max(tolerance_m, 1.0)))
        return float(min(1.0, max(0.0, getattr(near, "length", 0.0) / max(line.length, 1.0))))
    except Exception:
        return 0.0


def _normalize_open_arc_to_wet_exterior(
    source_open_arc: LineString,
    domain: Polygon,
    land_boundary,
    target_resolution_m: float,
) -> tuple[LineString, dict[str, Any]]:
    """Trim landward source-arc tails to the contiguous delivered wet exterior."""
    report: dict[str, Any] = {
        "open_arc_trimmed_to_wet_exterior": False,
        "open_arc_trim_reason": "no_eligible_exterior_landfall_interval",
        "source_open_arc_length_m": float(getattr(source_open_arc, "length", 0.0)),
        "delivered_open_arc_length_m": float(getattr(source_open_arc, "length", 0.0)),
        "discarded_source_open_arc_length_m": 0.0,
        "delivered_start_landfall_distance_m": None,
        "delivered_end_landfall_distance_m": None,
    }
    if (
        source_open_arc is None
        or source_open_arc.is_empty
        or float(source_open_arc.length) <= 0.0
        or domain is None
        or domain.is_empty
        or land_boundary is None
        or land_boundary.is_empty
    ):
        return source_open_arc, report

    tolerance = max(2.0, 0.02 * float(target_resolution_m))
    landfall_tolerance = max(2.0 * tolerance, 0.10 * float(target_resolution_m))
    positions = [0.0, float(source_open_arc.length)]
    try:
        intersection = source_open_arc.intersection(land_boundary)
        positions.extend(_line_intersection_positions(source_open_arc, intersection))
    except Exception:
        pass
    positions = _unique_sorted_positions(positions, float(source_open_arc.length))
    candidates: list[dict[str, Any]] = []
    for start, end in zip(positions[:-1], positions[1:]):
        if end - start <= max(1.0e-6, 0.01 * tolerance):
            continue
        try:
            part = substring(source_open_arc, start, end)
        except Exception:
            continue
        if not isinstance(part, LineString) or part.is_empty or float(part.length) <= tolerance:
            continue
        overlap = _line_fraction_near_boundary(part, domain.boundary, tolerance)
        start_distance = float(Point(part.coords[0]).distance(land_boundary))
        end_distance = float(Point(part.coords[-1]).distance(land_boundary))
        candidates.append(
            {
                "line": part,
                "start_position_m": float(start),
                "end_position_m": float(end),
                "overlap_fraction": float(overlap),
                "start_landfall_distance_m": start_distance,
                "end_landfall_distance_m": end_distance,
                "landfall_pair": bool(
                    start_distance <= landfall_tolerance
                    and end_distance <= landfall_tolerance
                ),
            }
        )
    eligible = [
        item
        for item in candidates
        if item["landfall_pair"] and item["overlap_fraction"] >= 0.98
    ]
    if not eligible:
        report["open_arc_trim_candidate_count"] = int(len(candidates))
        return source_open_arc, report

    selected = max(
        eligible,
        key=lambda item: (
            float(item["line"].length),
            float(item["overlap_fraction"]),
        ),
    )
    delivered = selected["line"]
    discarded = max(0.0, float(source_open_arc.length) - float(delivered.length))
    trimmed = bool(discarded > tolerance)
    report.update(
        {
            "open_arc_trimmed_to_wet_exterior": trimmed,
            "open_arc_trim_reason": (
                "longest_contiguous_source_interval_on_wet_exterior_between_landfalls"
                if trimmed
                else "source_arc_already_matches_wet_exterior"
            ),
            "source_open_arc_length_m": float(source_open_arc.length),
            "delivered_open_arc_length_m": float(delivered.length),
            "discarded_source_open_arc_length_m": float(discarded),
            "delivered_source_start_position_m": float(selected["start_position_m"]),
            "delivered_source_end_position_m": float(selected["end_position_m"]),
            "delivered_open_arc_boundary_overlap_fraction": float(selected["overlap_fraction"]),
            "delivered_start_landfall_distance_m": float(selected["start_landfall_distance_m"]),
            "delivered_end_landfall_distance_m": float(selected["end_landfall_distance_m"]),
            "open_arc_trim_candidate_count": int(len(candidates)),
        }
    )
    return delivered, report


def _line_intersection_positions(line: LineString, geom) -> list[float]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [float(line.project(geom))]
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        if not coords:
            return []
        return [
            float(line.project(Point(coords[0]))),
            float(line.project(Point(coords[-1]))),
        ]
    if hasattr(geom, "geoms"):
        positions: list[float] = []
        for part in geom.geoms:
            positions.extend(_line_intersection_positions(line, part))
        return positions
    return []


def _unique_sorted_positions(values: list[float], length: float) -> list[float]:
    ordered = sorted(min(max(float(value), 0.0), float(length)) for value in values)
    unique: list[float] = []
    for value in ordered:
        if not unique or abs(value - unique[-1]) > 1.0e-6:
            unique.append(value)
    return unique


def _promote_delivered_open_arc_landfalls(
    anchors: dict[str, Any],
    delivered_open_arc: LineString,
    land_boundary,
    target_resolution_m: float,
) -> dict[str, Any]:
    """Replace provisional frame-side anchors with delivered wet-exterior landfalls."""
    if delivered_open_arc is None or delivered_open_arc.is_empty:
        return anchors
    out = dict(anchors)
    start = Point(delivered_open_arc.coords[0])
    end = Point(delivered_open_arc.coords[-1])
    tolerance = max(2.0, 0.10 * float(target_resolution_m))
    start_landfall_distance = float(start.distance(land_boundary)) if land_boundary is not None and not land_boundary.is_empty else float("inf")
    end_landfall_distance = float(end.distance(land_boundary)) if land_boundary is not None and not land_boundary.is_empty else float("inf")
    original_start = Point(out["start_xy"])
    original_end = Point(out["end_xy"])
    out.update(
        {
            "source_start_xy": out.get("start_xy"),
            "source_end_xy": out.get("end_xy"),
            "source_start_to_delivered_landfall_m": float(original_start.distance(start)),
            "source_end_to_delivered_landfall_m": float(original_end.distance(end)),
            "start_xy": (float(start.x), float(start.y)),
            "end_xy": (float(end.x), float(end.y)),
            "start_role": "wet_exterior_start_landfall",
            "end_role": "wet_exterior_end_landfall",
            "start_anchor_found": bool(start_landfall_distance <= tolerance),
            "end_anchor_found": bool(end_landfall_distance <= tolerance),
            "start_anchor_method": "wet_exterior_land_intersection",
            "end_anchor_method": "wet_exterior_land_intersection",
            "start_snap_distance_m": float(start_landfall_distance),
            "end_snap_distance_m": float(end_landfall_distance),
            "anchor_tolerance_m": float(tolerance),
            "seaward_chain_xy": [
                (float(start.x), float(start.y)),
                (float(end.x), float(end.y)),
            ],
        }
    )
    if out.get("selected_side_start_corner_xy") is not None:
        out["start_distance_m"] = float(start.distance(Point(out["selected_side_start_corner_xy"])))
    if out.get("selected_side_end_corner_xy") is not None:
        out["end_distance_m"] = float(end.distance(Point(out["selected_side_end_corner_xy"])))
    return out


def _classify_domain_boundary_segments(
    domain: Polygon,
    open_arc: LineString,
    land_boundary,
    frame_xy: Polygon,
    target_resolution_m: float,
) -> dict[str, Any]:
    tolerance = max(2.0, 0.02 * target_resolution_m)
    exterior = LineString(domain.exterior.coords) if isinstance(domain, Polygon) and not domain.is_empty else LineString()
    land_geom = GeometryCollection()
    if land_boundary is not None and not land_boundary.is_empty and not exterior.is_empty:
        land_geom = exterior.intersection(land_boundary.buffer(tolerance))
    frame_geom = GeometryCollection()
    if not exterior.is_empty and frame_xy is not None and not frame_xy.is_empty:
        frame_geom = exterior.intersection(frame_xy.boundary.buffer(tolerance))
        remove_near = GeometryCollection()
        if open_arc is not None and not open_arc.is_empty:
            remove_near = open_arc.buffer(max(2.0 * tolerance, 0.1 * target_resolution_m))
        if land_boundary is not None and not land_boundary.is_empty:
            land_near = land_boundary.buffer(max(2.0 * tolerance, 0.1 * target_resolution_m))
            remove_near = land_near if remove_near.is_empty else remove_near.union(land_near)
        if not remove_near.is_empty:
            try:
                frame_geom = frame_geom.difference(remove_near)
            except Exception:
                pass

    land_parts = _line_parts(land_geom)
    frame_parts = _line_parts(frame_geom)
    land_length = _sum_line_length(land_parts)
    frame_length = _sum_line_length(frame_parts)
    open_length = float(open_arc.length) if open_arc is not None and not open_arc.is_empty else 0.0
    denominator = max(float(exterior.length) - open_length, 1.0)
    return {
        "land_boundary_arcs_xy": land_parts,
        "frame_clip_boundary_arcs_xy": frame_parts,
        "land_boundary_length_m": float(land_length),
        "frame_clip_boundary_length_m": float(frame_length),
        "land_boundary_overlap_fraction": float(min(1.0, land_length / denominator)),
    }


def _line_parts(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if len(geom.coords) >= 2 and geom.length > 0.0 else []
    if isinstance(geom, MultiLineString):
        return [part for part in geom.geoms if len(part.coords) >= 2 and part.length > 0.0]
    if hasattr(geom, "geoms"):
        parts: list[LineString] = []
        for part in geom.geoms:
            parts.extend(_line_parts(part))
        return parts
    return []


def _sum_line_length(lines: list[LineString]) -> float:
    return float(sum(line.length for line in lines if line is not None and not line.is_empty))


def _choose_seed_component(geom, seed_xy: Point) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if geom.buffer(1.0).contains(seed_xy) or geom.intersects(seed_xy) else None
    if hasattr(geom, "geoms"):
        polygons = [part for part in geom.geoms if isinstance(part, Polygon) and not part.is_empty]
        if not polygons:
            return None
        containing = [poly for poly in polygons if poly.buffer(1.0).contains(seed_xy) or poly.intersects(seed_xy)]
        if containing:
            return max(containing, key=lambda poly: poly.area)
        return max(polygons, key=lambda poly: poly.area)
    return None


def iterative_raster_topology(
    coastline_lines_xy: list[LineString],
    initial_arc_xy: LineString,
    initial_anchors: dict[str, Any],
    selected_side_xy: LineString,
    offshore_unit: np.ndarray,
    bpoly_xy: Polygon,
    seed_xy: Point,
    config: BdryArcConfig,
    visual_dir: Path | None,
    projection: LocalProjection,
    name: str,
    bpoly_lonlat: Polygon,
) -> dict[str, Any]:
    """Use seeded raster connectivity to reduce large CUSP linework before final vector output."""
    target_resolution = float(config.target_resolution_m)
    resolutions = _raster_resolution_schedule(
        target_resolution,
        config.raster_resolution_m,
        max(1, int(config.max_topology_iterations)),
    )
    convergence_anchor_m = float(config.convergence_anchor_m or 2.0 * target_resolution)
    convergence_area_frac = float(config.convergence_area_frac)

    relevant_lines = list(coastline_lines_xy)
    selected_arc = initial_arc_xy
    anchors = dict(initial_anchors)
    scored = score_and_select_bdry_arc(
        generate_offshore_arc_candidates(
            anchors["start_xy"],
            anchors["end_xy"],
            offshore_unit,
            selected_side_xy,
            bpoly_xy,
            target_resolution,
        ),
        GeometryCollection(),
        bpoly_xy,
        target_resolution,
    )
    wet_result: dict[str, Any] | None = None
    previous_area: float | None = None
    previous_anchor_pair: tuple[Point, Point] | None = None
    iterations: list[dict[str, Any]] = []
    progress_run_dir = visual_dir.parents[1] if visual_dir is not None else None

    if visual_dir is not None:
        _plot_preliminary_arc_map(
            visual_dir / "preliminary_arc_map.png",
            name,
            coastline_lines_xy,
            bpoly_lonlat,
            unproject_geometry(selected_arc, projection),
            [
                unproject_geometry(Point(anchors["start_xy"]), projection),
                unproject_geometry(Point(anchors["end_xy"]), projection),
            ],
            projection,
        )

    for iter_index, resolution in enumerate(resolutions, start=1):
        if progress_run_dir is not None:
            _write_progress(progress_run_dir, "iterative-raster", "raster-fill-start", {"iteration": iter_index, "resolution_m": float(resolution), "line_count": len(relevant_lines)})
        raster = _raster_connectivity_fill(
            relevant_lines,
            selected_arc,
            bpoly_xy,
            seed_xy,
            float(resolution),
        )
        if progress_run_dir is not None:
            _write_progress(progress_run_dir, "iterative-raster", "raster-fill-done", raster["metadata"])
        wet_polygon = raster["wet_domain_xy"]
        if progress_run_dir is not None:
            _write_progress(progress_run_dir, "iterative-raster", "component-classification-start", {"iteration": iter_index})
        classified = _classify_relevant_lines(
            coastline_lines_xy,
            wet_polygon,
            selected_arc,
            target_resolution,
            float(resolution),
        )
        if progress_run_dir is not None:
            _write_progress(progress_run_dir, "iterative-raster", "component-classification-done", {"iteration": iter_index, "retained": classified["retained_count"], "dropped": classified["dropped_count"]})
        retained_lines = classified["retained_lines"]
        if len(retained_lines) >= 2:
            relevant_lines = retained_lines

        try:
            anchors = _select_anchor_points(relevant_lines, selected_side_xy, bpoly_xy, target_resolution)
        except Exception:
            pass
        candidates = generate_offshore_arc_candidates(
            anchors["start_xy"],
            anchors["end_xy"],
            offshore_unit,
            selected_side_xy,
            bpoly_xy,
            target_resolution,
        )
        coast_union = unary_union(relevant_lines) if len(relevant_lines) <= 20_000 and relevant_lines else GeometryCollection()
        scored = score_and_select_bdry_arc(candidates, coast_union, bpoly_xy, target_resolution)
        selected_arc = scored["selected"]["geometry"]
        if progress_run_dir is not None:
            _write_progress(progress_run_dir, "iterative-raster", "arc-update-done", {"iteration": iter_index, "selected": scored["selected"]["candidate_id"]})

        area = float(wet_polygon.area)
        anchor_pair = (Point(anchors["start_xy"]), Point(anchors["end_xy"]))
        anchor_shift = None
        area_change_frac = None
        converged = False
        if previous_area and previous_area > 0:
            area_change_frac = abs(area - previous_area) / previous_area
        if previous_anchor_pair is not None:
            anchor_shift = max(anchor_pair[0].distance(previous_anchor_pair[0]), anchor_pair[1].distance(previous_anchor_pair[1]))
        if area_change_frac is not None and anchor_shift is not None:
            converged = area_change_frac <= convergence_area_frac and anchor_shift <= convergence_anchor_m

        iteration_meta = {
            "iteration": iter_index,
            "resolution_m": float(resolution),
            "filled_area_m2": area,
            "filled_cell_count": raster["metadata"]["filled_cell_count"],
            "barrier_cell_count": raster["metadata"]["barrier_cell_count"],
            "rasterized_line_count": raster["metadata"].get("rasterized_line_count"),
            "retained_line_count": int(classified["retained_count"]),
            "dropped_line_count": int(classified["dropped_count"]),
            "retained_truncated_to_polygonize_limit": bool(classified.get("retained_truncated_to_polygonize_limit")),
            "area_change_frac": area_change_frac,
            "anchor_shift_m": anchor_shift,
            "converged": bool(converged),
        }
        iterations.append(iteration_meta)
        wet_result = {
            "wet_domain_xy": wet_polygon,
            "metadata": {
                **raster["metadata"],
                "source": "iterative_raster_connectivity",
                "method": "seeded_raster_fill_component_preclassification",
                "iteration": iter_index,
                "retained_line_count": int(classified["retained_count"]),
                "dropped_line_count": int(classified["dropped_count"]),
                "area_change_frac": area_change_frac,
                "anchor_shift_m": anchor_shift,
                "converged": bool(converged),
                "forbidden_overlap": [],
            },
        }

        if visual_dir is not None:
            _plot_raster_connectivity_map(
                visual_dir / f"raster_connectivity_iter_{iter_index:02d}.png",
                name,
                wet_polygon,
                selected_arc,
                bpoly_xy,
                seed_xy,
                projection,
                iteration_meta,
            )
            _plot_component_classification_map(
                visual_dir / f"component_classification_iter_{iter_index:02d}.png",
                name,
                classified["retained_lines"],
                classified["dropped_sample"],
                wet_polygon,
                selected_arc,
                bpoly_xy,
                projection,
                iteration_meta,
            )
            _write_progress(progress_run_dir, "iterative-raster", "iteration-plots-done", {"iteration": iter_index})

        previous_area = area
        previous_anchor_pair = anchor_pair
        if converged and iter_index >= 2:
            break

    if wet_result is None:
        wet_result = {
            "wet_domain_xy": bpoly_xy.buffer(0),
            "metadata": {
                "source": "fallback_bpoly_polygon",
                "fallback_reason": "iterative_raster_connectivity_failed_to_run",
                "area_m2": float(bpoly_xy.area),
                "perimeter_m": float(bpoly_xy.length),
                "hole_count": 0,
                "seed_inside": bool(bpoly_xy.contains(seed_xy)),
                "forbidden_overlap": [],
            },
        }

    vector_result = extract_seeded_wet_domain(relevant_lines, selected_arc, bpoly_xy, seed_xy, [], target_resolution)
    if vector_result["metadata"].get("source") != "fallback_bpoly_polygon" and vector_result["metadata"].get("seed_inside"):
        vector_result["metadata"]["source"] = "reduced_vector_polygonize_after_iterative_raster"
        vector_result["metadata"]["raster_preclassification_iterations"] = len(iterations)
        wet_result = vector_result
    return {
        "wet_result": wet_result,
        "selected_arc_xy": selected_arc,
        "anchors": anchors,
        "scored": scored,
        "relevant_lines_xy": relevant_lines,
        "iterations": iterations,
    }


def _raster_resolution_schedule(target_resolution_m: float, requested_resolution_m: float | None, max_iterations: int) -> list[float]:
    start = float(requested_resolution_m) if requested_resolution_m else max(8.0 * target_resolution_m, 2000.0)
    target = float(target_resolution_m)
    if max_iterations <= 1 or start <= target:
        return [target]
    values = np.geomspace(start, target, num=max_iterations)
    out = []
    for value in values:
        rounded = float(max(target, round(float(value) / 10.0) * 10.0))
        if not out or abs(out[-1] - rounded) > 1.0:
            out.append(rounded)
    if out[-1] != target:
        out.append(target)
    return out[:max_iterations]


def _raster_connectivity_fill(
    coastline_lines_xy: list[LineString],
    offshore_arc_xy: LineString,
    bpoly_xy: Polygon,
    seed_xy: Point,
    resolution_m: float,
) -> dict[str, Any]:
    minx, miny, maxx, maxy = bpoly_xy.bounds
    width = max(1, int(math.ceil((maxx - minx) / resolution_m)))
    height = max(1, int(math.ceil((maxy - miny) / resolution_m)))
    max_cells = 8_000_000
    if width * height > max_cells:
        scale = math.sqrt((width * height) / max_cells)
        resolution_m = float(resolution_m * scale)
        width = max(1, int(math.ceil((maxx - minx) / resolution_m)))
        height = max(1, int(math.ceil((maxy - miny) / resolution_m)))
    transform = Affine.translation(minx, maxy) * Affine.scale(resolution_m, -resolution_m)

    domain_mask = rio_features.rasterize(
        [(bpoly_xy, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    raster_lines = _raster_barrier_lines(coastline_lines_xy, resolution_m)
    barrier_shapes = [(line, 1) for line in raster_lines if line is not None and not line.is_empty]
    barrier_shapes.append((offshore_arc_xy, 1))
    barriers = rio_features.rasterize(
        barrier_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    barriers = ndimage.binary_dilation(barriers, structure=np.ones((3, 3), dtype=bool), iterations=1)
    water = domain_mask & ~barriers
    seed_cell = _seed_cell(seed_xy, transform, water)
    seed_mask = np.zeros_like(water, dtype=bool)
    if seed_cell is not None:
        seed_mask[seed_cell[0], seed_cell[1]] = True
        filled = ndimage.binary_propagation(seed_mask, structure=np.ones((3, 3), dtype=bool), mask=water)
    else:
        filled = np.zeros_like(water, dtype=bool)
    wet_polygon = _vectorize_mask_to_polygon(filled, transform, seed_xy, bpoly_xy)
    metadata = {
        "source": "iterative_raster_connectivity",
        "resolution_m": float(resolution_m),
        "raster_width": int(width),
        "raster_height": int(height),
        "raster_cell_count": int(width * height),
        "barrier_cell_count": int(np.count_nonzero(barriers & domain_mask)),
        "input_line_count": int(len(coastline_lines_xy)),
        "rasterized_line_count": int(len(raster_lines)),
        "water_cell_count": int(np.count_nonzero(water)),
        "filled_cell_count": int(np.count_nonzero(filled)),
        "seed_cell": [int(seed_cell[0]), int(seed_cell[1])] if seed_cell else None,
        "seed_inside": bool(seed_cell is not None and wet_polygon.buffer(resolution_m).contains(seed_xy)),
        "area_m2": float(wet_polygon.area),
        "perimeter_m": float(wet_polygon.length),
        "hole_count": int(len(getattr(wet_polygon, "interiors", []))),
    }
    return {"wet_domain_xy": wet_polygon, "metadata": metadata, "filled_mask": filled, "transform": transform}


def _raster_barrier_lines(lines_xy: list[LineString], resolution_m: float) -> list[LineString]:
    if len(lines_xy) <= 30_000:
        source = lines_xy
    else:
        min_length = max(0.75 * float(resolution_m), 150.0)
        source = [line for line in lines_xy if line.length >= min_length]
        if resolution_m >= 1500.0:
            cap = 5_000
        elif resolution_m >= 750.0:
            cap = 8_000
        else:
            cap = 14_000
        if len(source) > cap:
            source.sort(key=lambda line: line.length, reverse=True)
            source = source[:cap]
        if not source:
            source = lines_xy[:cap]
    tolerance = max(0.2 * float(resolution_m), 25.0)
    simplified = []
    for line in source:
        simple = line.simplify(tolerance, preserve_topology=False)
        if isinstance(simple, LineString) and not simple.is_empty and len(simple.coords) >= 2:
            simplified.append(simple)
        elif isinstance(simple, MultiLineString):
            simplified.extend(part for part in simple.geoms if len(part.coords) >= 2)
    return simplified or source


def _seed_cell(seed_xy: Point, transform: Affine, water: np.ndarray) -> tuple[int, int] | None:
    inv = ~transform
    col_f, row_f = inv * (seed_xy.x, seed_xy.y)
    row = int(math.floor(row_f))
    col = int(math.floor(col_f))
    height, width = water.shape
    if 0 <= row < height and 0 <= col < width and water[row, col]:
        return row, col
    for radius in range(1, 20):
        r0 = max(0, row - radius)
        r1 = min(height, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(width, col + radius + 1)
        candidates = np.argwhere(water[r0:r1, c0:c1])
        if candidates.size:
            best = min(candidates, key=lambda rc: (int(rc[0]) + r0 - row) ** 2 + (int(rc[1]) + c0 - col) ** 2)
            return int(best[0]) + r0, int(best[1]) + c0
    return None


def _vectorize_mask_to_polygon(mask: np.ndarray, transform: Affine, seed_xy: Point, fallback: Polygon) -> Polygon:
    if not np.any(mask):
        return fallback.buffer(0)
    polygons = []
    for geom, value in rio_features.shapes(mask.astype("uint8"), mask=mask, transform=transform):
        if int(value) == 1:
            poly = shape(geom).buffer(0)
            if isinstance(poly, Polygon) and not poly.is_empty:
                polygons.append(poly)
    if not polygons:
        return fallback.buffer(0)
    containing = [poly for poly in polygons if poly.buffer(1.0).contains(seed_xy)]
    if containing:
        return max(containing, key=lambda geom: geom.area).buffer(0)
    return max(polygons, key=lambda geom: geom.area).buffer(0)


def _classify_relevant_lines(
    lines_xy: list[LineString],
    wet_domain_xy: Polygon,
    offshore_arc_xy: LineString,
    target_resolution_m: float,
    raster_resolution_m: float,
) -> dict[str, Any]:
    buffer_m = max(3.0 * target_resolution_m, 2.0 * raster_resolution_m, 750.0)
    boundary_zone = wet_domain_xy.boundary.buffer(buffer_m)
    interior_zone = wet_domain_xy.buffer(max(target_resolution_m, raster_resolution_m))
    arc_zone = offshore_arc_xy.buffer(buffer_m)
    query_bounds = _expand_bounds(wet_domain_xy.bounds, buffer_m)
    arc_bounds = _expand_bounds(offshore_arc_xy.bounds, buffer_m)
    prepared_boundary = prep(boundary_zone)
    prepared_interior = prep(interior_zone)
    prepared_arc = prep(arc_zone)
    min_ring_area = max((2.0 * target_resolution_m) ** 2, 1.0)

    retained: list[LineString] = []
    dropped_sample: list[LineString] = []
    for idx, line in enumerate(lines_xy):
        if not (_bounds_intersect(line.bounds, query_bounds) or _bounds_intersect(line.bounds, arc_bounds)):
            if len(dropped_sample) < 400 and idx % max(1, len(lines_xy) // 400) == 0:
                dropped_sample.append(line)
            continue
        keep = False
        if prepared_boundary.intersects(line) or prepared_arc.intersects(line):
            keep = True
        elif _is_resolved_ring_in_domain(line, wet_domain_xy, min_ring_area, prepared_interior):
            keep = True
        elif line.length >= 4.0 * target_resolution_m and prepared_interior.intersects(line) and line.distance(wet_domain_xy.boundary) <= buffer_m:
            keep = True
        if keep:
            retained.append(line)
        elif len(dropped_sample) < 400 and idx % max(1, len(lines_xy) // 400) == 0:
            dropped_sample.append(line)
    if not retained:
        retained = list(lines_xy)
    truncated = False
    if len(retained) > 30_000:
        retained.sort(key=lambda line: line.length, reverse=True)
        retained = retained[:30_000]
        truncated = True
    return {
        "retained_lines": retained,
        "retained_count": int(len(retained)),
        "dropped_count": int(max(len(lines_xy) - len(retained), 0)),
        "dropped_sample": dropped_sample,
        "retained_truncated_to_polygonize_limit": truncated,
    }


def _expand_bounds(bounds: tuple[float, float, float, float], distance: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bounds
    return (minx - distance, miny - distance, maxx + distance, maxy + distance)


def _bounds_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _is_resolved_ring_in_domain(line: LineString, wet_domain_xy: Polygon, min_area_m2: float, prepared_interior) -> bool:
    if not (line.is_ring or Point(line.coords[0]).distance(Point(line.coords[-1])) <= 1.0):
        return False
    try:
        poly = Polygon(line.coords).buffer(0)
    except Exception:
        return False
    return isinstance(poly, Polygon) and poly.area >= min_area_m2 and prepared_interior.contains(poly.centroid)


def _fetch_gshhs_coastline(
    bbox_wsen: tuple[float, float, float, float],
    run_dir: Path,
    name: str,
    gshhs_skill_dir: str | None,
    resolution: str,
    levels: str,
    progress_interval_s: float = 30.0,
) -> tuple[Path, dict[str, Any]]:
    """Run the installed gshhs-coastline fetch script for a buffered bbox."""
    run_dir.mkdir(parents=True, exist_ok=True)
    skill_dir = _resolve_gshhs_skill_dir(gshhs_skill_dir)
    request_json = run_dir / "gshhs_request.json"
    request = {
        "schema_version": "gshhs_request_for_fvcom_bdry_arc_v1",
        "name": name,
        "bbox": list(map(float, bbox_wsen)),
        "resolution": resolution,
        "levels": levels,
        "source": "fvcom-bdry-arc --fetch-coastline",
    }
    request_json.write_text(json.dumps(request, indent=2), encoding="utf-8")
    estimate_json = run_dir / "download_estimate.json"
    estimate_script = skill_dir / "scripts" / "estimate_data_request.py"
    fetch_script = skill_dir / "scripts" / "fetch_gshhs_coastline.py"
    if estimate_script.exists():
        _run_subprocess_with_progress(
            [
                sys.executable,
                str(estimate_script),
                "--request",
                str(request_json),
                "--run-dir",
                str(run_dir),
                "--output",
                str(estimate_json),
                "--skill-name",
                "gshhs-coastline",
                "--run-id",
                name,
            ],
            run_dir,
            "fetch-gshhs-estimate",
            progress_interval_s,
        )
    _run_subprocess_with_progress(
        [
            sys.executable,
            str(fetch_script),
            "--bbox",
            *(str(v) for v in bbox_wsen),
            "--run-dir",
            str(run_dir),
            "--name",
            name,
            "--resolution",
            resolution,
            "--levels",
            levels,
            "--formats",
            "gpkg,geojson",
            "--allow-no-basemap",
            "--quiet",
        ],
        run_dir,
        "fetch-gshhs",
        progress_interval_s,
    )
    gpkg = run_dir / f"{name}_gshhs_land.gpkg"
    manifest = run_dir / f"{name}_gshhs_manifest.json"
    if not gpkg.exists():
        raise FileNotFoundError(f"GSHHS fetch did not create expected GeoPackage: {gpkg}")
    metadata: dict[str, Any] = {
        "fetched_with": "gshhs-coastline",
        "coastline_source": "gshhs",
        "gshhs_resolution": resolution,
        "gshhs_levels": levels,
        "gshhs_manifest_path": str(manifest) if manifest.exists() else None,
    }
    if manifest.exists():
        try:
            gshhs_manifest = _read_json(manifest)
            metadata["gshhs_selected_resolution"] = gshhs_manifest.get("source", {}).get("selected_resolution")
            metadata["gshhs_selected_levels"] = gshhs_manifest.get("source", {}).get("selected_levels")
        except Exception:
            pass
    return gpkg, metadata


def _fetch_cusp_coastline(
    bbox_wsen: tuple[float, float, float, float],
    run_dir: Path,
    name: str,
    cusp_skill_dir: str | None,
    fallback_policy: str,
) -> Path:
    """Run the installed cusp-coastline fetch script for a buffered bbox."""
    run_dir.mkdir(parents=True, exist_ok=True)
    skill_dir = _resolve_cusp_skill_dir(cusp_skill_dir)
    request_json = run_dir / "cusp_request.json"
    request = {
        "schema_version": "cusp_request_for_fvcom_bdry_arc_v1",
        "name": name,
        "bbox": list(map(float, bbox_wsen)),
        "estimated_mb": 300,
        "source": "fvcom-bdry-arc --fetch-coastline",
    }
    request_json.write_text(json.dumps(request, indent=2), encoding="utf-8")
    estimate_json = run_dir / "download_estimate.json"
    estimate_script = skill_dir / "scripts" / "estimate_data_request.py"
    fetch_script = skill_dir / "scripts" / "fetch_cusp_coastline.py"
    if estimate_script.exists():
        subprocess.run(
            [
                sys.executable,
                str(estimate_script),
                "--request",
                str(request_json),
                "--run-dir",
                str(run_dir),
                "--output",
                str(estimate_json),
                "--skill-name",
                "cusp-coastline",
                "--run-id",
                name,
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(fetch_script),
            "--bbox",
            *(str(v) for v in bbox_wsen),
            "--run-dir",
            str(run_dir),
            "--name",
            name,
            "--formats",
            "gpkg,geojson",
            "--region",
            "auto",
            "--fallback-policy",
            fallback_policy,
            "--allow-no-basemap",
            "--client-timeout-s",
            "600",
            "--quiet",
        ],
        check=True,
    )
    gpkg = run_dir / f"{name}_cusp_coastline.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(f"CUSP fetch did not create expected GeoPackage: {gpkg}")
    return gpkg


def _resolve_cusp_skill_dir(path: str | None) -> Path:
    if path:
        candidate = Path(path)
    else:
        codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        candidate = codex_home / "skills" / "cusp-coastline"
    if not candidate.exists():
        raise FileNotFoundError(f"Could not locate cusp-coastline skill: {candidate}")
    return candidate


def _resolve_gshhs_skill_dir(path: str | None) -> Path:
    if path:
        candidate = Path(path)
    else:
        codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        candidate = codex_home / "skills" / "gshhs-coastline"
    if not candidate.exists():
        raise FileNotFoundError(f"Could not locate gshhs-coastline skill: {candidate}")
    return candidate


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _upstream_bpoly_unresolved(region: dict[str, Any]) -> bool:
    status = str(region.get("final_status") or "").lower()
    domain_type = str(region.get("domain_type") or region.get("region_bpoly", {}).get("domain_type") or "").lower()
    coords = region.get("polygon_lonlat") or region.get("region_bpoly", {}).get("polygon_lonlat")
    return domain_type == "unresolved_autonomous_failure" or (status == "needs_review" and not coords)


def _write_unresolved_upstream_manifest(
    region: dict[str, Any],
    offshore: dict[str, Any],
    run_dir: Path,
    name: str,
    config: BdryArcConfig,
    resolved_heuristic_mode: str,
    place_memory_enabled: bool,
) -> dict[str, Any]:
    failure_taxonomy = ["upstream_region_bpoly_unresolved"]
    upstream_failures = (
        region.get("qa", {})
        .get("bpoly_quality", {})
        .get("failure_taxonomy", [])
    )
    for item in upstream_failures:
        code = item.get("code") if isinstance(item, dict) else str(item)
        if code and code not in failure_taxonomy:
            failure_taxonomy.append(str(code))
    manifest_path = run_dir / "bdry_arc_manifest.json"
    manifest = {
        "schema_version": "fvcom_bdry_arc_manifest_v1",
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_by": "fvcom-bdry-arc run_bdry_arc.py",
        "final_status": "needs_review",
        "failure_taxonomy": failure_taxonomy,
        "settings": {
            "mode": config.mode,
            "target_resolution_m": float(config.target_resolution_m),
            "review_depth": config.review_depth,
            "coastline_source": config.coastline_source,
            "topology_mode_requested": config.topology_mode,
            "topology_mode_used": "upstream-unresolved",
            "gshhs_resolution_requested": config.gshhs_resolution,
            "gshhs_levels": config.gshhs_levels,
            "heuristic_mode": resolved_heuristic_mode,
            "place_memory_enabled": bool(place_memory_enabled),
            "boundary_resolution_profile": config.boundary_resolution_profile,
            "frame_clip_policy": config.frame_clip_policy,
            "residual_boundary_policy": config.residual_boundary_policy,
            "obc_placement_policy": config.obc_placement_policy,
        },
        "inputs": {
            "region_name": region.get("name"),
            "region_final_status": region.get("final_status"),
            "region_domain_type": region.get("domain_type"),
            "offshore_boundary_policy": offshore.get("boundary_policy"),
        },
        "wet_domain": {
            "closure_method": "upstream_region_bpoly_unresolved",
            "area_m2": 0.0,
            "seed_inside": False,
        },
        "model_boundary_loops": {
            "final_status": "needs_review",
            "failure_taxonomy": ["upstream_region_bpoly_unresolved"],
        },
        "outputs": {
            "bdry_arc_manifest": str(manifest_path),
            "progress_state": str(run_dir / "bdry_arc_progress_state.json"),
            "progress_jsonl": str(run_dir / "bdry_arc_progress.jsonl"),
        },
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_progress(run_dir, "complete", "failed", {"failure_taxonomy": failure_taxonomy})
    return manifest


def _write_progress(run_dir: Path, stage: str, message: str, details: dict[str, Any] | None = None) -> None:
    now = datetime.now(timezone.utc)
    state_path = run_dir / "bdry_arc_progress_state.json"
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            state = _read_json(state_path)
        except Exception:
            state = {}
    started_unix = float(state.get("started_unix", now.timestamp()))
    elapsed = max(0.0, now.timestamp() - started_unix)
    percent = _progress_percent(stage, message)
    record = {
        "time_utc": now.isoformat(timespec="seconds"),
        "stage": stage,
        "message": message,
        "progress_percent": percent,
        "elapsed_seconds": elapsed,
        "details": _json_safe(details or {}),
    }
    with (run_dir / "bdry_arc_progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    state_doc = {
        "schema_version": "fvcom_bdry_arc_progress_state_v1",
        "started_utc": state.get("started_utc", now.isoformat(timespec="seconds")),
        "started_unix": started_unix,
        "updated_utc": now.isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "current_stage": stage,
        "current_message": message,
        "progress_percent": percent,
        "last_details": _json_safe(details or {}),
        "health_policy": "Progress is reported for slow stages; do not downshift GSHHS resolution unless the prompt or CLI explicitly requested it.",
    }
    state_path.write_text(json.dumps(_json_safe(state_doc), indent=2), encoding="utf-8")


def _run_subprocess_with_progress(
    cmd: list[str],
    run_dir: Path,
    stage: str,
    progress_interval_s: float,
) -> None:
    interval = max(float(progress_interval_s), 1.0)
    _write_progress(run_dir, stage, "start", {"command": _redacted_command(cmd)})
    proc = subprocess.Popen(cmd)
    last = time.monotonic()
    heartbeat_count = 0
    while proc.poll() is None:
        time.sleep(min(interval, 1.0))
        now = time.monotonic()
        if now - last >= interval:
            heartbeat_count += 1
            _write_progress(run_dir, stage, "running", {"heartbeat_count": heartbeat_count, "pid": proc.pid})
            last = now
    if proc.returncode != 0:
        _write_progress(run_dir, stage, "failed", {"returncode": proc.returncode})
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    _write_progress(run_dir, stage, "done", {"returncode": proc.returncode, "heartbeat_count": heartbeat_count})


def _redacted_command(cmd: list[str]) -> list[str]:
    return [str(part) for part in cmd]


def _progress_percent(stage: str, message: str) -> float:
    stage_points = {
        "run": 1.0,
        "fetch-gshhs-estimate": 5.0,
        "fetch-gshhs": 12.0,
        "load-coastline": 20.0,
        "project-coastline": 30.0,
        "audit": 36.0,
        "repair": 42.0,
        "initial-arc": 50.0,
        "gshhs-vector": 66.0,
        "island-loop": 66.0,
        "iterative-raster": 66.0,
        "write-outputs": 82.0,
        "model-boundary-loops": 92.0,
        "complete": 100.0,
    }
    base = stage_points.get(stage, 50.0)
    if message in {"done", "candidate-done"}:
        return min(base + 8.0, 99.0)
    if message in {"failed"}:
        return base
    if message in {"running"}:
        return min(base + 3.0, 98.0)
    return base


def _load_bpoly_polygon(region: dict[str, Any]) -> Polygon:
    coords = region.get("polygon_lonlat") or region.get("region_bpoly", {}).get("polygon_lonlat")
    if not coords:
        raise ValueError("region_bpoly_json does not contain polygon_lonlat")
    polygon = Polygon([(float(x), float(y)) for x, y in coords])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if not isinstance(polygon, Polygon) or polygon.is_empty:
        raise ValueError("region_bpoly_json polygon_lonlat is not a valid polygon")
    return polygon


def _buffer_bbox_lonlat(bbox_wsen: tuple[float, float, float, float], buffer_km: float) -> tuple[float, float, float, float]:
    west, south, east, north = bbox_wsen
    lat0 = 0.5 * (south + north)
    dlat = buffer_km / 110.54
    dlon = buffer_km / max(111.32 * math.cos(math.radians(lat0)), 20.0)
    buffered_west = west - dlon
    buffered_east = east + dlon
    crosses = east < west
    if crosses:
        return (_wrap_lon(buffered_west), south - dlat, _wrap_lon(buffered_east), north + dlat)
    return (max(-180.0, buffered_west), south - dlat, min(180.0, buffered_east), north + dlat)


def _load_coastline_product(
    path: str | Path,
    bbox_wsen: tuple[float, float, float, float],
    coastline_source: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    path = Path(path)
    layers = _list_vector_layers(path)
    coastline_layer = _choose_layer(layers, ("coastline_lines", "coastline", "shoreline", "lines"))
    land_layer = _choose_layer(layers, ("land_polygons", "land", "polygons"))
    if coastline_layer is None and layers:
        coastline_layer = layers[0]
    coastline = _read_vector_layer(path, bbox_wsen, coastline_layer)
    land = _read_vector_layer(path, bbox_wsen, land_layer) if land_layer else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if coastline.empty and not land.empty:
        coastline = gpd.GeoDataFrame(land.drop(columns="geometry", errors="ignore"), geometry=land.geometry.boundary, crs=land.crs)
    manifest_path = _discover_gshhs_manifest(path)
    metadata = {
        "available_layers": layers,
        "selected_coastline_layer": coastline_layer,
        "selected_land_layer": land_layer,
        "coastline_feature_count": int(len(coastline)),
        "land_polygon_feature_count": int(len(land)),
        "coastline_source": coastline_source,
        "gshhs_manifest_path": str(manifest_path) if manifest_path else None,
    }
    if manifest_path:
        try:
            gshhs_manifest = _read_json(manifest_path)
            metadata["gshhs_requested_resolution"] = gshhs_manifest.get("request", {}).get("resolution")
            metadata["gshhs_selected_resolution"] = gshhs_manifest.get("source", {}).get("selected_resolution")
            metadata["gshhs_selected_levels"] = gshhs_manifest.get("source", {}).get("selected_levels")
        except Exception:
            pass
    return coastline, land, metadata


def _load_coastline(path: str | Path, bbox_wsen: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    coastline, _land, _metadata = _load_coastline_product(path, bbox_wsen, "generic-gpkg")
    return coastline


def _list_vector_layers(path: Path) -> list[str]:
    try:
        import pyogrio

        return [str(name) for name in pyogrio.list_layers(path)[:, 0].tolist()]
    except Exception:
        return []


def _choose_layer(layers: list[str], preferred: tuple[str, ...]) -> str | None:
    lower = {name.lower(): name for name in layers}
    for item in preferred:
        if item.lower() in lower:
            return lower[item.lower()]
    return None


def _read_vector_layer(path: Path, bbox_wsen: tuple[float, float, float, float], layer: str | None) -> gpd.GeoDataFrame:
    path = Path(path)
    try:
        if _bbox_crosses_antimeridian(bbox_wsen):
            pieces = []
            west, south, east, north = bbox_wsen
            for split_bbox in ((west, south, 180.0, north), (-180.0, south, east, north)):
                if layer:
                    pieces.append(gpd.read_file(path, layer=layer, bbox=split_bbox))
                else:
                    pieces.append(gpd.read_file(path, bbox=split_bbox))
            pieces = [piece for piece in pieces if not piece.empty]
            gdf = gpd.GeoDataFrame(
                pd.concat(pieces, ignore_index=True),
                geometry="geometry",
                crs=pieces[0].crs if pieces else "EPSG:4326",
            ) if pieces else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        elif layer:
            gdf = gpd.read_file(path, layer=layer, bbox=bbox_wsen)
        else:
            gdf = gpd.read_file(path, bbox=bbox_wsen)
    except Exception:
        try:
            gdf = gpd.read_file(path, bbox=bbox_wsen)
        except Exception:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    if len(gdf) <= 20_000 and not _bbox_crosses_antimeridian(bbox_wsen):
        try:
            gdf = gpd.clip(gdf, gpd.GeoSeries([box(*bbox_wsen)], crs="EPSG:4326")).reset_index(drop=True)
        except Exception:
            pass
    return gdf


def _bbox_crosses_antimeridian(bbox_wsen: tuple[float, float, float, float]) -> bool:
    west, _south, east, _north = bbox_wsen
    return float(east) < float(west)


def _wrap_lon(lon: float) -> float:
    x = float(lon)
    while x > 180.0:
        x -= 360.0
    while x < -180.0:
        x += 360.0
    return x


def _unwrap_gdf_longitudes(gdf: gpd.GeoDataFrame, origin: float) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    out = gdf.copy()
    out = out.to_crs("EPSG:4326") if out.crs is not None else out.set_crs("EPSG:4326")
    out["geometry"] = [unwrap_geometry_longitudes(geom, origin) if geom is not None and not geom.is_empty else geom for geom in out.geometry]
    return out.set_crs("EPSG:4326", allow_override=True)


def _discover_gshhs_manifest(path: Path) -> Path | None:
    candidates = sorted(path.parent.glob("*_gshhs_manifest.json"))
    return candidates[0] if candidates else None


def _gshhs_resolution_policy(config: BdryArcConfig, coastline_load_meta: dict[str, Any]) -> dict[str, Any]:
    requested = str(config.gshhs_resolution or "f").lower()
    selected = str(
        coastline_load_meta.get("gshhs_selected_resolution")
        or coastline_load_meta.get("fetch", {}).get("gshhs_selected_resolution")
        or requested
    ).lower()
    manifest_requested = str(coastline_load_meta.get("gshhs_requested_resolution") or "").lower()
    explicit_manifest_selection = bool(
        manifest_requested
        and manifest_requested not in {"auto"}
        and manifest_requested == selected
    )
    explicit_lower_requested = requested not in {"auto", "f"} or explicit_manifest_selection
    downgraded_without_request = (
        requested == "f"
        and selected not in {"f", "full", ""}
        and not explicit_manifest_selection
    )
    return {
        "requested_resolution": requested,
        "upstream_manifest_requested_resolution": manifest_requested or None,
        "selected_resolution": selected,
        "default_resolution": "f",
        "explicit_lower_resolution_requested": bool(explicit_lower_requested),
        "downgraded_without_explicit_request": bool(downgraded_without_request),
        "policy": "default to GSHHS full resolution and never downshift to h/i/l/c unless the CLI or supplied GSHHS fetch manifest explicitly requested it",
    }


def _flatten_lines(geometries: Iterable[Any]) -> list[LineString]:
    lines: list[LineString] = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            if len(geom.coords) >= 2:
                lines.append(geom)
        elif isinstance(geom, MultiLineString):
            for part in geom.geoms:
                if len(part.coords) >= 2:
                    lines.append(part)
        elif isinstance(geom, Polygon):
            lines.append(LineString(geom.exterior.coords))
            for ring in geom.interiors:
                lines.append(LineString(ring.coords))
        elif hasattr(geom, "geoms"):
            lines.extend(_flatten_lines(geom.geoms))
    return lines


def _flatten_polygons(geometries: Iterable[Any]) -> list[Polygon]:
    polygons: list[Polygon] = []
    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            polygons.append(geom.buffer(0))
        elif hasattr(geom, "geoms"):
            polygons.extend(_flatten_polygons(geom.geoms))
    return [poly for poly in polygons if isinstance(poly, Polygon) and not poly.is_empty]


def _selected_side_line(offshore: dict[str, Any], region: dict[str, Any]) -> LineString:
    start = offshore.get("selected_side_start_lonlat")
    end = offshore.get("selected_side_end_lonlat")
    if start and end:
        return LineString([(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))])
    coords = region.get("polygon_lonlat") or region.get("region_bpoly", {}).get("polygon_lonlat")
    idx = int(offshore.get("selected_side_index", region.get("region_bpoly", {}).get("offshore_side_index", 0)))
    if not coords or idx >= len(coords) - 1:
        raise ValueError("Could not resolve selected offshore side from artifacts")
    return LineString([coords[idx], coords[idx + 1]])


def _offshore_unit_vector(offshore: dict[str, Any], selected_side_xy: LineString) -> np.ndarray:
    az = offshore.get("offshore_azimuth_deg")
    if az is not None:
        radians = math.radians(float(az))
        unit = np.asarray([math.sin(radians), math.cos(radians)], dtype=float)
    else:
        coords = np.asarray(selected_side_xy.coords, dtype=float)
        tangent = coords[-1] - coords[0]
        unit = np.asarray([tangent[1], -tangent[0]], dtype=float)
    norm = float(np.linalg.norm(unit))
    if not np.isfinite(norm) or norm <= 0:
        unit = np.asarray([1.0, 0.0], dtype=float)
    else:
        unit = unit / norm
    return unit


def _resolve_seed(region: dict[str, Any], config: BdryArcConfig, projection: LocalProjection, bpoly_xy: Polygon) -> tuple[Point, dict[str, Any]]:
    if config.seed_mode == "manual-json":
        if not config.manual_seed_json:
            raise ValueError("--manual-seed-json is required with --seed-mode manual-json")
        data = _read_json(config.manual_seed_json)
        lon = float(data["lon"])
        lat = float(data["lat"])
        seed = project_geometry(Point(lon, lat), projection)
        return seed, {"source": "manual-json", "lon": lon, "lat": lat, "manual_seed_json": config.manual_seed_json}
    for feature in region.get("target_region_features", {}).get("features", []):
        if not feature.get("required", True):
            continue
        geom = feature.get("geometry")
        if isinstance(geom, list) and len(geom) == 4:
            lon = 0.5 * (float(geom[0]) + float(geom[2]))
            lat = 0.5 * (float(geom[1]) + float(geom[3]))
            seed = project_geometry(Point(lon, lat), projection)
            if bpoly_xy.buffer(1000.0).contains(seed):
                return seed, {"source": "required_feature_center", "feature_id": feature.get("id"), "lon": lon, "lat": lat}
    centroid = bpoly_xy.centroid
    lonlat = unproject_geometry(centroid, projection)
    return centroid, {"source": "bpoly_centroid", "lon": float(lonlat.x), "lat": float(lonlat.y)}


def _default_forbidden_regions(region: dict[str, Any]) -> list[dict[str, Any]]:
    key = (
        region.get("qa", {})
        .get("bpoly_quality", {})
        .get("canonical_region_key")
    )
    name = str(region.get("name", "")).lower()
    request = str(region.get("target_region_features", {}).get("request_text", "")).lower()
    items: list[dict[str, Any]] = []
    if key == "delaware" or "delaware" in name or "delaware" in request:
        # Visually placed from the Delaware smoke map: cover the Chesapeake-side
        # wrong-region water context west of Delmarva without covering Delaware Bay.
        chesapeake_guard = Polygon(
            [
                (-77.65, 36.60),
                (-76.05, 36.60),
                (-75.82, 37.20),
                (-75.78, 37.78),
                (-75.94, 38.24),
                (-76.12, 38.72),
                (-76.34, 39.18),
                (-76.58, 39.65),
                (-77.65, 39.65),
                (-77.65, 36.60),
            ]
        )
        items.append(
            {
                "id": "chesapeake_bay_guard",
                "label": "Chesapeake Bay wrong-region guard, visually placed west of Delaware Bay",
                "geometry": chesapeake_guard,
            }
        )
    return items


def _select_anchor_points(
    lines_xy: list[LineString],
    selected_side_xy: LineString,
    bpoly_xy: Polygon,
    target_resolution_m: float,
) -> dict[str, Any]:
    coords = list(selected_side_xy.coords)
    targets = [Point(coords[0]), Point(coords[-1])]
    candidate_indices: set[int] = set()
    for target in targets:
        ranked = sorted(
            enumerate(lines_xy),
            key=lambda item: _bounds_distance_to_point(item[1].bounds, target),
        )
        for idx, _line in ranked[: min(750, len(ranked))]:
            candidate_indices.add(idx)

    search_poly = bpoly_xy.buffer(max(5.0 * target_resolution_m, 1000.0))
    samples: list[tuple[Point, int]] = []
    for idx in sorted(candidate_indices):
        line = lines_xy[idx]
        if not line.intersects(search_poly):
            continue
        spacing = max(5.0 * target_resolution_m, 1000.0)
        samples.extend((point, idx) for point in _sample_line_points(line, spacing))
        samples.append((Point(line.coords[0]), idx))
        samples.append((Point(line.coords[-1]), idx))
        for target in targets:
            try:
                samples.append((line.interpolate(line.project(target)), idx))
            except Exception:
                pass
    if not samples:
        raise ValueError("No coastline samples available for anchor selection")

    selected = []
    for target in targets:
        best = min(samples, key=lambda item: item[0].distance(target))
        selected.append(best)
    if selected[0][0].distance(selected[1][0]) < max(20.0 * target_resolution_m, 5000.0):
        target = targets[1]
        selected[1] = min(
            (item for item in samples if item[0].distance(selected[0][0]) >= max(20.0 * target_resolution_m, 5000.0)),
            key=lambda item: item[0].distance(target),
            default=selected[1],
        )
    return {
        "start_xy": (float(selected[0][0].x), float(selected[0][0].y)),
        "end_xy": (float(selected[1][0].x), float(selected[1][0].y)),
        "start_line_index": int(selected[0][1]),
        "end_line_index": int(selected[1][1]),
        "start_distance_m": float(selected[0][0].distance(targets[0])),
        "end_distance_m": float(selected[1][0].distance(targets[1])),
    }


def _uses_lake_closed_boundary(region: dict[str, Any], offshore: dict[str, Any]) -> bool:
    domain_type = str(region.get("domain_type") or region.get("region_bpoly", {}).get("domain_type") or "").lower()
    boundary_policies = {
        str(region.get("boundary_policy") or "").lower(),
        str(offshore.get("boundary_policy") or "").lower(),
    }
    return domain_type == "lake" or "no_open_boundary" in boundary_policies


def _uses_island_loop_branch(region: dict[str, Any], offshore: dict[str, Any], config: BdryArcConfig, place_memory_enabled: bool = True) -> bool:
    if config.topology_mode == "island-loop":
        return True
    if config.topology_mode != "gshhs-vector":
        return False
    domain_type = str(region.get("domain_type") or region.get("region_bpoly", {}).get("domain_type") or "").lower()
    boundary_policies = {
        str(region.get("boundary_policy") or "").lower(),
        str(offshore.get("boundary_policy") or "").lower(),
    }
    canonical = str(
        region.get("qa", {})
        .get("bpoly_quality", {})
        .get("canonical_region_key", "")
    ).lower()
    return (
        domain_type == "island"
        or "offshore_loop_no_land_anchors" in boundary_policies
        or (place_memory_enabled and canonical in {"hawaii_state", "hawaii_island", "aleutian"})
    )


def _lake_closed_boundary_reference_points(bpoly_xy: Polygon, selected_side_xy: LineString, target_resolution_m: float) -> dict[str, Any]:
    coords = list(selected_side_xy.coords)
    if len(coords) < 2:
        coords = list(bpoly_xy.exterior.coords)[:2]
    start = (float(coords[0][0]), float(coords[0][1]))
    end = (float(coords[-1][0]), float(coords[-1][1]))
    context = _selected_side_context(bpoly_xy, selected_side_xy)
    return {
        "source": "lake_closed_boundary_no_open_arc",
        "start_role": "lake_boundary_reference_start",
        "end_role": "lake_boundary_reference_end",
        "start_xy": start,
        "end_xy": end,
        "start_line_index": None,
        "end_line_index": None,
        "start_distance_m": 0.0,
        "end_distance_m": 0.0,
        "start_snap_distance_m": 0.0,
        "end_snap_distance_m": 0.0,
        "start_anchor_found": True,
        "end_anchor_found": True,
        "start_anchor_method": "lake_closed_boundary_no_open_arc",
        "end_anchor_method": "lake_closed_boundary_no_open_arc",
        "anchor_tolerance_m": float(max(target_resolution_m, 250.0)),
        "selected_side_index": int(context["selected_side_index"]),
        "start_adjacent_side_index": int(context["start_adjacent_side_index"]),
        "end_adjacent_side_index": int(context["end_adjacent_side_index"]),
        "selected_side_start_corner_xy": context["selected_side_start_corner_xy"],
        "selected_side_end_corner_xy": context["selected_side_end_corner_xy"],
        "selected_side_xy": selected_side_xy,
        "seaward_chain_xy": [],
        "closed_boundary": True,
        "no_ocean_open_boundary": True,
    }


def _island_loop_reference_points(selected_side_xy: LineString, bpoly_xy: Polygon, target_resolution_m: float) -> dict[str, Any]:
    coords = list(selected_side_xy.coords)
    start = (float(coords[0][0]), float(coords[0][1]))
    end = (float(coords[-1][0]), float(coords[-1][1]))
    context = _selected_side_context(bpoly_xy, selected_side_xy)
    return {
        "source": "offshore_loop_no_land_anchors",
        "start_role": "island_loop_reference_start",
        "end_role": "island_loop_reference_end",
        "start_xy": start,
        "end_xy": end,
        "start_line_index": None,
        "end_line_index": None,
        "start_distance_m": 0.0,
        "end_distance_m": 0.0,
        "start_snap_distance_m": 0.0,
        "end_snap_distance_m": 0.0,
        "start_anchor_found": True,
        "end_anchor_found": True,
        "start_anchor_method": "selected_bpoly_side_reference",
        "end_anchor_method": "selected_bpoly_side_reference",
        "anchor_tolerance_m": float(max(target_resolution_m, 250.0)),
        "selected_side_index": int(context["selected_side_index"]),
        "start_adjacent_side_index": int(context["start_adjacent_side_index"]),
        "end_adjacent_side_index": int(context["end_adjacent_side_index"]),
        "selected_side_start_corner_xy": context["selected_side_start_corner_xy"],
        "selected_side_end_corner_xy": context["selected_side_end_corner_xy"],
        "selected_side_xy": selected_side_xy,
        "seaward_chain_xy": [],
        "closed_loop": True,
    }


def _coastline_bpoly_anchor_points(
    coastline_boundary_xy,
    selected_side_xy: LineString,
    bpoly_xy: Polygon,
    target_resolution_m: float,
) -> dict[str, Any]:
    """Find open-boundary anchors where coastline intersects the two adjacent bpoly sides."""
    context = _selected_side_context(bpoly_xy, selected_side_xy)
    tolerance = max(float(target_resolution_m), 250.0)
    start_anchor = _find_side_coastline_anchor(
        context["start_adjacent_side_xy"],
        coastline_boundary_xy,
        Point(context["selected_side_start_corner_xy"]),
        tolerance,
    )
    end_anchor = _find_side_coastline_anchor(
        context["end_adjacent_side_xy"],
        coastline_boundary_xy,
        Point(context["selected_side_end_corner_xy"]),
        tolerance,
    )
    seaward_chain_xy = [
        start_anchor["xy"],
        context["selected_side_start_corner_xy"],
        context["selected_side_end_corner_xy"],
        end_anchor["xy"],
    ]
    return {
        "source": "coastline_bpoly_intersection",
        "start_role": "coastline_bpoly_start_anchor",
        "end_role": "coastline_bpoly_end_anchor",
        "start_xy": start_anchor["xy"],
        "end_xy": end_anchor["xy"],
        "start_line_index": None,
        "end_line_index": None,
        "start_distance_m": float(start_anchor["distance_to_corner_m"]),
        "end_distance_m": float(end_anchor["distance_to_corner_m"]),
        "start_snap_distance_m": float(start_anchor["snap_distance_m"]),
        "end_snap_distance_m": float(end_anchor["snap_distance_m"]),
        "start_anchor_found": bool(start_anchor["found"]),
        "end_anchor_found": bool(end_anchor["found"]),
        "start_anchor_method": start_anchor["method"],
        "end_anchor_method": end_anchor["method"],
        "anchor_tolerance_m": float(tolerance),
        "selected_side_index": int(context["selected_side_index"]),
        "start_adjacent_side_index": int(context["start_adjacent_side_index"]),
        "end_adjacent_side_index": int(context["end_adjacent_side_index"]),
        "selected_side_start_corner_xy": context["selected_side_start_corner_xy"],
        "selected_side_end_corner_xy": context["selected_side_end_corner_xy"],
        "selected_side_xy": selected_side_xy,
        "seaward_chain_xy": seaward_chain_xy,
    }


def _selected_side_context(bpoly_xy: Polygon, selected_side_xy: LineString) -> dict[str, Any]:
    coords = [(float(x), float(y)) for x, y in list(bpoly_xy.exterior.coords)[:-1]]
    if len(coords) < 3:
        raise ValueError("Region bpoly must contain at least three unique vertices")
    selected_coords = list(selected_side_xy.coords)
    start = (float(selected_coords[0][0]), float(selected_coords[0][1]))
    end = (float(selected_coords[-1][0]), float(selected_coords[-1][1]))
    start_pt = Point(start)
    end_pt = Point(end)
    n = len(coords)
    best_idx = 0
    best_distance = float("inf")
    best_reversed = False
    for idx in range(n):
        a = Point(coords[idx])
        b = Point(coords[(idx + 1) % n])
        forward = a.distance(start_pt) + b.distance(end_pt)
        reverse = a.distance(end_pt) + b.distance(start_pt)
        if min(forward, reverse) < best_distance:
            best_distance = min(forward, reverse)
            best_idx = idx
            best_reversed = reverse < forward
    ring_start_idx = best_idx
    ring_end_idx = (best_idx + 1) % n
    if best_reversed:
        start_adjacent_idx = (best_idx + 1) % n
        end_adjacent_idx = (best_idx - 1) % n
    else:
        start_adjacent_idx = (best_idx - 1) % n
        end_adjacent_idx = (best_idx + 1) % n
    return {
        "selected_side_index": best_idx,
        "selected_side_reversed_from_ring": bool(best_reversed),
        "selected_side_start_corner_xy": start,
        "selected_side_end_corner_xy": end,
        "selected_side_ring_start_index": ring_start_idx,
        "selected_side_ring_end_index": ring_end_idx,
        "start_adjacent_side_index": start_adjacent_idx,
        "end_adjacent_side_index": end_adjacent_idx,
        "start_adjacent_side_xy": LineString([coords[start_adjacent_idx], coords[(start_adjacent_idx + 1) % n]]),
        "end_adjacent_side_xy": LineString([coords[end_adjacent_idx], coords[(end_adjacent_idx + 1) % n]]),
        "selected_side_match_distance_m": float(best_distance),
    }


def _find_side_coastline_anchor(
    adjacent_side_xy: LineString,
    coastline_boundary_xy,
    offshore_corner_xy: Point,
    tolerance_m: float,
) -> dict[str, Any]:
    if coastline_boundary_xy is None or coastline_boundary_xy.is_empty:
        fallback = adjacent_side_xy.interpolate(adjacent_side_xy.project(offshore_corner_xy))
        return {
            "xy": (float(fallback.x), float(fallback.y)),
            "found": False,
            "method": "missing_coastline_boundary",
            "snap_distance_m": float("inf"),
            "distance_to_corner_m": float(fallback.distance(offshore_corner_xy)),
        }

    intersection = adjacent_side_xy.intersection(coastline_boundary_xy)
    candidates: list[dict[str, Any]] = []
    for point in _points_from_intersection(intersection, offshore_corner_xy):
        candidates.append(
            {
                "point": point,
                "found": True,
                "method": "exact_intersection",
                "snap_distance_m": 0.0,
                "distance_to_corner_m": float(point.distance(offshore_corner_xy)),
            }
        )
    try:
        near_geom = coastline_boundary_xy.intersection(adjacent_side_xy.buffer(tolerance_m))
    except Exception:
        near_geom = GeometryCollection()
    for coast_point in _points_from_intersection(near_geom, offshore_corner_xy):
        side_point = adjacent_side_xy.interpolate(adjacent_side_xy.project(coast_point))
        snap_distance = float(side_point.distance(coast_point))
        if snap_distance <= tolerance_m:
            candidates.append(
                {
                    "point": side_point,
                    "found": True,
                    "method": "snapped_within_tolerance" if snap_distance > 0.0 else "exact_intersection",
                    "snap_distance_m": snap_distance,
                    "distance_to_corner_m": float(side_point.distance(offshore_corner_xy)),
                }
            )
    if candidates:
        chosen_item = min(candidates, key=lambda item: (item["distance_to_corner_m"], item["snap_distance_m"]))
        chosen = chosen_item["point"]
        return {
            "xy": (float(chosen.x), float(chosen.y)),
            "found": bool(chosen_item["found"]),
            "method": chosen_item["method"],
            "snap_distance_m": float(chosen_item["snap_distance_m"]),
            "distance_to_corner_m": float(chosen_item["distance_to_corner_m"]),
        }

    try:
        side_point, coast_point = nearest_points(adjacent_side_xy, coastline_boundary_xy)
        snap_distance = float(side_point.distance(coast_point))
    except Exception:
        side_point = adjacent_side_xy.interpolate(adjacent_side_xy.project(offshore_corner_xy))
        snap_distance = float("inf")
    found = bool(snap_distance <= tolerance_m)
    return {
        "xy": (float(side_point.x), float(side_point.y)),
        "found": found,
        "method": "snapped_within_tolerance" if found else "nearest_exceeds_tolerance",
        "snap_distance_m": float(snap_distance),
        "distance_to_corner_m": float(side_point.distance(offshore_corner_xy)),
    }


def _points_from_intersection(geom, reference: Point) -> list[Point]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Point):
        return [geom]
    if isinstance(geom, LineString):
        try:
            return [nearest_points(reference, geom)[1]]
        except Exception:
            coords = list(geom.coords)
            return [Point(coords[0]), Point(coords[-1])] if coords else []
    if hasattr(geom, "geoms"):
        points: list[Point] = []
        for part in geom.geoms:
            points.extend(_points_from_intersection(part, reference))
        return points
    return []


def _bounds_distance_to_point(bounds: tuple[float, float, float, float], point: Point) -> float:
    minx, miny, maxx, maxy = bounds
    dx = max(minx - point.x, 0.0, point.x - maxx)
    dy = max(miny - point.y, 0.0, point.y - maxy)
    return math.hypot(dx, dy)


def _sample_line_points(line: LineString, spacing_m: float) -> list[Point]:
    if line.is_empty or line.length <= 0:
        return []
    n = max(2, int(math.ceil(line.length / max(spacing_m, 1.0))) + 1)
    return [line.interpolate(float(t), normalized=True) for t in np.linspace(0.0, 1.0, n)]


def _endpoint_tangent(coords: list[tuple[float, float]], which: str) -> np.ndarray:
    if which == "start":
        vec = np.asarray(coords[1], dtype=float) - np.asarray(coords[0], dtype=float)
    else:
        vec = np.asarray(coords[-2], dtype=float) - np.asarray(coords[-1], dtype=float)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else np.asarray([0.0, 0.0], dtype=float)


def _tangent_compatible(a: np.ndarray, b: np.ndarray) -> bool:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return False
    return abs(float(np.dot(a / na, b / nb))) > 0.35


def _deformed_side_arc(
    side0: np.ndarray,
    side1: np.ndarray,
    anchor0: np.ndarray,
    anchor1: np.ndarray,
    offshore_unit: np.ndarray,
    bow: float,
    n: int,
) -> LineString:
    coords: list[tuple[float, float]] = []
    start_offset = anchor0 - side0
    end_offset = anchor1 - side1
    for t in np.linspace(0.0, 1.0, n):
        side_point = side0 + t * (side1 - side0)
        endpoint_blend = (1.0 - t) * start_offset + t * end_offset
        offshore_bow = offshore_unit * bow * math.sin(math.pi * t)
        pt = side_point + endpoint_blend + offshore_bow
        coords.append((float(pt[0]), float(pt[1])))
    return LineString(coords)


def _seaward_chain_arc(
    anchor0: np.ndarray,
    side0: np.ndarray,
    side1: np.ndarray,
    anchor1: np.ndarray,
    offshore_unit: np.ndarray,
    bow: float,
    n: int,
) -> LineString:
    """Smooth the split-side plus offshore-side bpoly chain with fixed coastline anchors."""
    control0 = side0 + offshore_unit * bow
    control1 = side1 + offshore_unit * bow
    pts = []
    for t in np.linspace(0.0, 1.0, n):
        pt = (
            (1.0 - t) ** 3 * anchor0
            + 3.0 * (1.0 - t) ** 2 * t * control0
            + 3.0 * (1.0 - t) * t**2 * control1
            + t**3 * anchor1
        )
        pts.append(tuple(float(v) for v in pt))
    pts[0] = tuple(float(v) for v in anchor0)
    pts[-1] = tuple(float(v) for v in anchor1)
    return LineString(pts)


def _bezier_arc(p0: np.ndarray, p1: np.ndarray, offshore_unit: np.ndarray, bow: float, n: int) -> LineString:
    c0 = p0 + offshore_unit * bow
    c1 = p1 + offshore_unit * bow
    pts = []
    for t in np.linspace(0.0, 1.0, n):
        pt = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c0 + 3 * (1 - t) * t**2 * c1 + t**3 * p1
        pts.append(tuple(float(v) for v in pt))
    return LineString(pts)


def _bowed_arc(p0: np.ndarray, p1: np.ndarray, offshore_unit: np.ndarray, bow: float, n: int) -> LineString:
    pts = []
    for t in np.linspace(0.0, 1.0, n):
        pt = p0 + t * (p1 - p0) + offshore_unit * bow * math.sin(math.pi * t)
        pts.append(tuple(float(v) for v in pt))
    return LineString(pts)


def _choose_seed_face(faces: list[Polygon], seed_xy: Point, bpoly_xy: Polygon) -> Polygon | None:
    if not faces:
        return None
    containing = [face for face in faces if face.buffer(1.0).contains(seed_xy)]
    if containing:
        return max(containing, key=lambda face: face.area)
    near = [face for face in faces if face.intersects(seed_xy.buffer(2500.0))]
    if near:
        return max(near, key=lambda face: face.area)
    overlapping = [face.intersection(bpoly_xy).buffer(0) for face in faces if face.intersects(bpoly_xy)]
    overlapping = [face for face in overlapping if isinstance(face, Polygon) and not face.is_empty]
    return max(overlapping, key=lambda face: face.area) if overlapping else None


def _candidate_gdf(candidates: list[dict[str, Any]], projection: LocalProjection) -> gpd.GeoDataFrame:
    records = []
    for rank, item in enumerate(candidates, start=1):
        if item.get("geometry") is None or item["geometry"].is_empty:
            continue
        records.append(
            {
                "candidate_id": item["candidate_id"],
                "family": item["family"],
                "rank": rank,
                "score": float(item["score"]),
                "selected": rank == 1,
                "extra_intersection": bool(item["metrics"].get("extra_coastline_intersection")),
                "geometry": unproject_geometry(item["geometry"], projection),
            }
        )
    if not records:
        return gpd.GeoDataFrame({"candidate_id": [], "family": [], "rank": [], "score": [], "selected": [], "extra_intersection": []}, geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def _lines_gdf(lines_xy: list[LineString], projection: LocalProjection, segment_class: str) -> gpd.GeoDataFrame:
    records = [
        {"segment_class": segment_class, "line_id": idx, "geometry": unproject_geometry(line, projection)}
        for idx, line in enumerate(lines_xy)
        if line is not None and not line.is_empty
    ]
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def _forbidden_gdf(items: list[dict[str, Any]]) -> gpd.GeoDataFrame:
    records = [{"region_id": item["id"], "label": item["label"], "geometry": item["geometry"]} for item in items]
    if not records:
        return gpd.GeoDataFrame({"region_id": [], "label": []}, geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def _build_output_layers(
    wet_domain_lonlat,
    selected_arc_lonlat,
    anchors_lonlat: list[Point],
    candidates_gdf: gpd.GeoDataFrame,
    raw_gdf: gpd.GeoDataFrame,
    repaired_gdf: gpd.GeoDataFrame,
    forbidden_gdf: gpd.GeoDataFrame,
    wet_result: dict[str, Any],
    projection: LocalProjection,
) -> dict[str, gpd.GeoDataFrame]:
    wet_gdf = gpd.GeoDataFrame(
        [{"segment_class": "wet_domain", "final_status": "candidate", "geometry": wet_domain_lonlat}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    if selected_arc_lonlat is None or selected_arc_lonlat.is_empty:
        open_gdf = gpd.GeoDataFrame({"segment_class": []}, geometry=[], crs="EPSG:4326")
    else:
        open_gdf = gpd.GeoDataFrame(
            [{"segment_class": "open_boundary", "geometry": selected_arc_lonlat}],
            geometry="geometry",
            crs="EPSG:4326",
        )
    boundary_segments = wet_result.get("boundary_segments_xy", {})
    land_lines_xy = boundary_segments.get("land_boundary_arcs_xy", [])
    frame_lines_xy = boundary_segments.get("frame_clip_boundary_arcs_xy", [])
    land_patch_lines_xy = boundary_segments.get("land_patch_boundary_arcs_xy", [])
    if land_lines_xy:
        land_gdf = _lines_gdf(land_lines_xy, projection, "land_boundary")
    elif wet_result["metadata"].get("closure_method") in {"deformed_bpoly_frame_minus_gshhs_land", "coastline_anchor_seaward_bpoly_chain"}:
        land_gdf = gpd.GeoDataFrame({"segment_class": [], "line_id": []}, geometry=[], crs="EPSG:4326")
    else:
        land_gdf = gpd.GeoDataFrame(
            [{"segment_class": "land_boundary_candidate", "geometry": LineString(wet_domain_lonlat.exterior.coords)}],
            geometry="geometry",
            crs="EPSG:4326",
        )
    frame_gdf = (
        _lines_gdf(frame_lines_xy, projection, "frame_clip_boundary")
        if frame_lines_xy
        else gpd.GeoDataFrame({"segment_class": [], "line_id": []}, geometry=[], crs="EPSG:4326")
    )
    land_patch_gdf = (
        _lines_gdf(land_patch_lines_xy, projection, "land_patch_boundary")
        if land_patch_lines_xy
        else gpd.GeoDataFrame({"segment_class": [], "line_id": []}, geometry=[], crs="EPSG:4326")
    )
    island_records = [
        {"segment_class": "island_hole", "hole_id": idx, "geometry": LineString(ring.coords)}
        for idx, ring in enumerate(getattr(wet_domain_lonlat, "interiors", []))
    ]
    if island_records:
        island_gdf = gpd.GeoDataFrame(island_records, geometry="geometry", crs="EPSG:4326")
    else:
        island_gdf = gpd.GeoDataFrame({"segment_class": [], "hole_id": []}, geometry=[], crs="EPSG:4326")
    anchor_gdf = gpd.GeoDataFrame(
        [
            {
                "anchor_role": "coastline_bpoly_start_anchor"
                if wet_result["metadata"].get("closure_method") == "coastline_anchor_seaward_bpoly_chain"
                else (
                    "island_loop_reference_start"
                    if wet_result["metadata"].get("closure_method") == "island_archipelago_offshore_loop"
                    else (
                        "lake_boundary_reference_start"
                        if wet_result["metadata"].get("closure_method") == "lake_closed_boundary_no_open_arc"
                        else (
                            "bpoly_open_side_start"
                            if wet_result["metadata"].get("closure_method") == "deformed_bpoly_frame_minus_gshhs_land"
                            else "start"
                        )
                    )
                ),
                "geometry": anchors_lonlat[0],
            },
            {
                "anchor_role": "coastline_bpoly_end_anchor"
                if wet_result["metadata"].get("closure_method") == "coastline_anchor_seaward_bpoly_chain"
                else (
                    "island_loop_reference_end"
                    if wet_result["metadata"].get("closure_method") == "island_archipelago_offshore_loop"
                    else (
                        "lake_boundary_reference_end"
                        if wet_result["metadata"].get("closure_method") == "lake_closed_boundary_no_open_arc"
                        else (
                            "bpoly_open_side_end"
                            if wet_result["metadata"].get("closure_method") == "deformed_bpoly_frame_minus_gshhs_land"
                            else "end"
                        )
                    )
                ),
                "geometry": anchors_lonlat[1],
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    diag_geom = wet_result["wet_domain_xy"].centroid
    diag_gdf = gpd.GeoDataFrame(
        [
            {
                "diagnostic": "seeded_wet_domain_centroid",
                "source": wet_result["metadata"].get("source"),
                "geometry": unproject_geometry(diag_geom, projection),
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    return {
        "wet_domain": wet_gdf,
        "open_boundary_arc": open_gdf,
        "land_boundary_arcs": land_gdf,
        "frame_clip_boundary_arcs": frame_gdf,
        "land_patch_boundary_arcs": land_patch_gdf,
        "island_holes": island_gdf,
        "anchor_points": anchor_gdf,
        "candidate_arcs": candidates_gdf,
        "coastline_raw": raw_gdf,
        "coastline_repaired": repaired_gdf,
        "topology_diagnostics": diag_gdf,
        "forbidden_regions": forbidden_gdf,
    }


def _write_outputs(run_dir: Path, layers: dict[str, gpd.GeoDataFrame], name: str) -> dict[str, str]:
    gpkg = run_dir / "bdry_arc_package.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    for layer_name, gdf in layers.items():
        _write_layer(gpkg, layer_name, gdf)
    segments = []
    for layer_name in ("open_boundary_arc", "land_boundary_arcs", "frame_clip_boundary_arcs", "land_patch_boundary_arcs", "island_holes"):
        gdf = layers[layer_name]
        if gdf.empty:
            continue
        for _, row in gdf.iterrows():
            props = {key: _json_safe(value) for key, value in row.items() if key != "geometry"}
            props["layer"] = layer_name
            segments.append({"type": "Feature", "properties": props, "geometry": mapping(row.geometry)})
    segments_path = run_dir / "bdry_arc_segments.geojson"
    segments_path.write_text(
        json.dumps({"type": "FeatureCollection", "name": name, "features": segments}, indent=2),
        encoding="utf-8",
    )
    return {"bdry_arc_package_gpkg": str(gpkg), "bdry_arc_segments_geojson": str(segments_path)}


def _write_layer(gpkg: Path, layer_name: str, gdf: gpd.GeoDataFrame) -> None:
    if gdf.empty:
        gdf = gpd.GeoDataFrame(
            [{"empty_layer": True, "geometry": GeometryCollection()}],
            geometry="geometry",
            crs="EPSG:4326",
        )
    gdf.to_file(gpkg, layer=layer_name, driver="GPKG")


def _write_review_maps(
    visual_dir: Path,
    name: str,
    layers: dict[str, gpd.GeoDataFrame],
    bpoly_lonlat: Polygon,
    candidates_gdf: gpd.GeoDataFrame,
    selected_arc_lonlat: LineString,
    final_status: str,
) -> None:
    preliminary_path = visual_dir / "preliminary_arc_map.png"
    _plot_final_map(preliminary_path, layers, bpoly_lonlat, final_status)
    fig, ax = plt.subplots(figsize=(11, 9))
    _plot_sampled(layers["coastline_repaired"], ax, color="#59636e", linewidth=0.35, alpha=0.7)
    if not candidates_gdf.empty:
        candidates_gdf.plot(ax=ax, color="#9c7a00", linewidth=0.8, alpha=0.35)
        candidates_gdf[candidates_gdf["selected"] == True].plot(ax=ax, color="#c81d25", linewidth=2.0)
    gpd.GeoSeries([bpoly_lonlat], crs="EPSG:4326").boundary.plot(ax=ax, color="#111111", linewidth=1.4, linestyle="--")
    if selected_arc_lonlat is not None and not selected_arc_lonlat.is_empty:
        gpd.GeoSeries([selected_arc_lonlat], crs="EPSG:4326").plot(ax=ax, color="#c81d25", linewidth=2.2)
    layers["anchor_points"].plot(ax=ax, color="#ffdd57", edgecolor="#111111", markersize=36)
    ax.set_title(f"{name} arc candidate contact sheet")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(visual_dir / "arc_candidate_contact_sheet.png", dpi=180)
    plt.close(fig)


def _write_gshhs_review_maps(
    visual_dir: Path,
    name: str,
    layers: dict[str, gpd.GeoDataFrame],
    bpoly_lonlat: Polygon,
    final_status: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    if not layers["wet_domain"].empty:
        layers["wet_domain"].plot(ax=ax, facecolor="#7cc6fe", edgecolor="#0b4f6c", linewidth=1.1, alpha=0.28)
    _plot_sampled(layers["coastline_raw"], ax, color="#4d4d4d", linewidth=0.35, alpha=0.65)
    gpd.GeoSeries([bpoly_lonlat], crs="EPSG:4326").boundary.plot(ax=ax, color="#111111", linewidth=1.2, linestyle="--")
    if not layers["open_boundary_arc"].empty:
        layers["open_boundary_arc"].plot(ax=ax, color="#d00000", linewidth=2.2)
    if "land_patch_boundary_arcs" in layers and not layers["land_patch_boundary_arcs"].empty:
        layers["land_patch_boundary_arcs"].plot(ax=ax, color="#005ea8", linewidth=2.0, linestyle="--")
    layers["anchor_points"].plot(ax=ax, color="#ffdd57", edgecolor="#111111", markersize=40)
    ax.set_title(f"{name} GSHHS polygon topology - {final_status}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(visual_dir / "gshhs_polygon_topology_map.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 9))
    _plot_sampled(layers["coastline_repaired"], ax, color="#202020", linewidth=0.45, alpha=0.75)
    if not layers["open_boundary_arc"].empty:
        layers["open_boundary_arc"].plot(ax=ax, color="#d00000", linewidth=2.2)
    if "land_patch_boundary_arcs" in layers and not layers["land_patch_boundary_arcs"].empty:
        layers["land_patch_boundary_arcs"].plot(ax=ax, color="#005ea8", linewidth=2.0, linestyle="--")
    layers["anchor_points"].plot(ax=ax, color="#ffdd57", edgecolor="#111111", markersize=52)
    gpd.GeoSeries([bpoly_lonlat], crs="EPSG:4326").boundary.plot(ax=ax, color="#111111", linewidth=1.2, linestyle="--")
    ax.set_title(f"{name} GSHHS anchors and offshore arc")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(visual_dir / "gshhs_anchor_arc_map.png", dpi=180)
    plt.close(fig)


def _plot_final_map(path: Path, layers: dict[str, gpd.GeoDataFrame], bpoly_lonlat: Polygon, final_status: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    if not layers["wet_domain"].empty:
        layers["wet_domain"].plot(ax=ax, facecolor="#7cc6fe", edgecolor="#0b4f6c", linewidth=1.2, alpha=0.28)
    _plot_sampled(layers["coastline_raw"], ax, color="#4d4d4d", linewidth=0.25, alpha=0.45)
    _plot_sampled(layers["coastline_repaired"], ax, color="#202020", linewidth=0.45, alpha=0.8)
    gpd.GeoSeries([bpoly_lonlat], crs="EPSG:4326").boundary.plot(ax=ax, color="#111111", linewidth=1.2, linestyle="--")
    if not layers["open_boundary_arc"].empty:
        layers["open_boundary_arc"].plot(ax=ax, color="#d00000", linewidth=2.2)
    if "land_patch_boundary_arcs" in layers and not layers["land_patch_boundary_arcs"].empty:
        layers["land_patch_boundary_arcs"].plot(ax=ax, color="#005ea8", linewidth=2.0, linestyle="--")
    layers["anchor_points"].plot(ax=ax, color="#ffdd57", edgecolor="#111111", markersize=40)
    ax.set_title(f"fvcom-bdry-arc review map - {final_status}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_preliminary_arc_map(
    path: Path,
    name: str,
    lines_xy: list[LineString],
    bpoly_lonlat: Polygon,
    arc_lonlat: LineString,
    anchors_lonlat: list[Point],
    projection: LocalProjection,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    sampled = lines_xy[:: max(1, int(math.ceil(len(lines_xy) / 12_000)))]
    sampled_lonlat = [unproject_geometry(line, projection) for line in sampled]
    gpd.GeoSeries(sampled_lonlat, crs="EPSG:4326").plot(ax=ax, color="#59636e", linewidth=0.25, alpha=0.5)
    gpd.GeoSeries([bpoly_lonlat], crs="EPSG:4326").boundary.plot(ax=ax, color="#111111", linewidth=1.2, linestyle="--")
    gpd.GeoSeries([arc_lonlat], crs="EPSG:4326").plot(ax=ax, color="#d00000", linewidth=2.2)
    gpd.GeoSeries(anchors_lonlat, crs="EPSG:4326").plot(ax=ax, color="#ffdd57", edgecolor="#111111", markersize=40)
    ax.set_title(f"{name} preliminary offshore arc")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_raster_connectivity_map(
    path: Path,
    name: str,
    wet_polygon_xy: Polygon,
    arc_xy: LineString,
    bpoly_xy: Polygon,
    seed_xy: Point,
    projection: LocalProjection,
    metadata: dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    wet = unproject_geometry(wet_polygon_xy, projection)
    arc = unproject_geometry(arc_xy, projection)
    bpoly = unproject_geometry(bpoly_xy, projection)
    seed = unproject_geometry(seed_xy, projection)
    gpd.GeoSeries([wet], crs="EPSG:4326").plot(ax=ax, facecolor="#7cc6fe", edgecolor="#0b4f6c", linewidth=1.1, alpha=0.35)
    gpd.GeoSeries([bpoly], crs="EPSG:4326").boundary.plot(ax=ax, color="#111111", linewidth=1.2, linestyle="--")
    gpd.GeoSeries([arc], crs="EPSG:4326").plot(ax=ax, color="#d00000", linewidth=2.2)
    gpd.GeoSeries([seed], crs="EPSG:4326").plot(ax=ax, color="#22a06b", edgecolor="#111111", markersize=45)
    ax.set_title(f"{name} raster connectivity iter {metadata['iteration']} ({metadata['resolution_m']:.0f} m)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_component_classification_map(
    path: Path,
    name: str,
    retained_lines_xy: list[LineString],
    dropped_sample_xy: list[LineString],
    wet_polygon_xy: Polygon,
    arc_xy: LineString,
    bpoly_xy: Polygon,
    projection: LocalProjection,
    metadata: dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    wet = unproject_geometry(wet_polygon_xy, projection)
    arc = unproject_geometry(arc_xy, projection)
    bpoly = unproject_geometry(bpoly_xy, projection)
    if dropped_sample_xy:
        dropped = [unproject_geometry(line, projection) for line in dropped_sample_xy]
        gpd.GeoSeries(dropped, crs="EPSG:4326").plot(ax=ax, color="#b7b7b7", linewidth=0.25, alpha=0.35)
    if retained_lines_xy:
        sample = retained_lines_xy[:: max(1, int(math.ceil(len(retained_lines_xy) / 12_000)))]
        retained = [unproject_geometry(line, projection) for line in sample]
        gpd.GeoSeries(retained, crs="EPSG:4326").plot(ax=ax, color="#202020", linewidth=0.45, alpha=0.85)
    gpd.GeoSeries([wet], crs="EPSG:4326").boundary.plot(ax=ax, color="#0b4f6c", linewidth=1.0)
    gpd.GeoSeries([bpoly], crs="EPSG:4326").boundary.plot(ax=ax, color="#111111", linewidth=1.2, linestyle="--")
    gpd.GeoSeries([arc], crs="EPSG:4326").plot(ax=ax, color="#d00000", linewidth=2.2)
    ax.set_title(
        f"{name} components iter {metadata['iteration']}: "
        f"keep {metadata['retained_line_count']}, drop {metadata['dropped_line_count']}"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _final_status(
    scored: dict[str, Any],
    wet_result: dict[str, Any],
    anchors: dict[str, Any],
    forbidden_regions_xy: list[Polygon],
) -> tuple[str, list[str]]:
    failures: list[str] = []
    metrics = scored["selected"]["metrics"]
    metadata = wet_result["metadata"]
    if metrics.get("extra_coastline_intersection") and not metadata.get("open_arc_trimmed_to_wet_exterior"):
        failures.append("open_arc_intersects_extra_coastline")
    if metadata.get("source") == "fallback_bpoly_polygon":
        failures.append("seeded_wet_domain_polygonize_failed")
    if metadata.get("topology_budget_exceeded"):
        failures.append("large_gshhs_topology_budget_exceeded")
    if metadata.get("gshhs_missing_land_polygons"):
        failures.append("gshhs_missing_land_polygons")
    closure_method = metadata.get("closure_method")
    if closure_method == "lake_closed_boundary_no_open_arc":
        if not metadata.get("deformed_frame_valid"):
            failures.append("lake_closed_boundary_frame_invalid")
        if not metadata.get("seed_inside"):
            failures.append("seed_not_inside_wet_domain")
        status = "pass" if not failures else "needs_review"
        return status, failures
    if metadata.get("arc_land_intersection") and closure_method != "island_archipelago_offshore_loop":
        failures.append("gshhs_open_arc_crosses_land")
    if closure_method in {"deformed_bpoly_frame_minus_gshhs_land", "coastline_anchor_seaward_bpoly_chain", "island_archipelago_offshore_loop"}:
        if not metadata.get("deformed_frame_valid"):
            failures.append("deformed_bpoly_frame_invalid")
        threshold = 0.90 if closure_method == "island_archipelago_offshore_loop" and metadata.get("island_blocker_land_patch_used") else 0.98
        if float(metadata.get("open_arc_boundary_overlap_fraction", 0.0)) < threshold:
            failures.append("island_loop_not_on_final_boundary" if closure_method == "island_archipelago_offshore_loop" else "open_arc_not_on_final_boundary")
        if closure_method == "island_archipelago_offshore_loop" and float(metadata.get("land_patch_boundary_fraction", 0.0)) > 0.35:
            failures.append("island_loop_land_patch_too_large")
    if anchors.get("source") == "coastline_bpoly_intersection":
        if not anchors.get("start_anchor_found", False):
            failures.append("start_coastline_bpoly_anchor_missing")
        if not anchors.get("end_anchor_found", False):
            failures.append("end_coastline_bpoly_anchor_missing")
        tolerance = float(anchors.get("anchor_tolerance_m", 0.0))
        if float(anchors.get("start_snap_distance_m", 0.0)) > tolerance:
            failures.append("start_coastline_bpoly_anchor_snap_exceeds_tolerance")
        if float(anchors.get("end_snap_distance_m", 0.0)) > tolerance:
            failures.append("end_coastline_bpoly_anchor_snap_exceeds_tolerance")
    if not metadata.get("seed_inside"):
        failures.append("seed_not_inside_wet_domain")
    if metadata.get("forbidden_overlap"):
        failures.append("forbidden_wrong_region_overlap")
    if anchors.get("start_distance_m", 0.0) > 300_000 or anchors.get("end_distance_m", 0.0) > 300_000:
        failures.append("anchor_far_from_offshore_side_endpoint")
    status = "pass" if not failures else "needs_review"
    return status, failures


def _plot_sampled(gdf: gpd.GeoDataFrame, ax, max_features: int = 12_000, **kwargs) -> None:
    if gdf.empty:
        return
    if len(gdf) > max_features:
        stride = max(1, int(math.ceil(len(gdf) / max_features)))
        gdf.iloc[::stride].plot(ax=ax, **kwargs)
    else:
        gdf.plot(ax=ax, **kwargs)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__geo_interface__"):
        return mapping(value)
    return str(value)
