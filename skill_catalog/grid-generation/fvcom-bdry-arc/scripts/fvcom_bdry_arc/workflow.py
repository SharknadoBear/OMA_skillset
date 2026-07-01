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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from rasterio import features as rio_features
from scipy import ndimage
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, box, mapping, shape
from shapely.prepared import prep
from shapely.ops import linemerge, nearest_points, polygonize, unary_union

from .projection import LocalProjection, local_utm_projection, project_geometry, unproject_geometry


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
    if config.topology_mode not in {"gshhs-vector", "iterative-raster", "vector-only"}:
        raise ValueError("--topology-mode must be gshhs-vector, iterative-raster, or vector-only")
    if config.topology_mode == "gshhs-vector" and config.coastline_source == "cusp-legacy":
        raise ValueError("--topology-mode gshhs-vector requires GSHHS/generic polygon-capable coastline input")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = run_dir / "intermediate" / "visual_review"
    if config.mode == "test":
        visual_dir.mkdir(parents=True, exist_ok=True)

    region = _read_json(region_bpoly_json)
    offshore = _read_json(offshore_artifacts_json)
    bpoly_lonlat = _load_bpoly_polygon(region)
    bbox_wsen = tuple(float(v) for v in region.get("envelope_bbox") or bpoly_lonlat.bounds)
    buffered_bbox = _buffer_bbox_lonlat(bbox_wsen, config.coastline_buffer_km)
    projection = local_utm_projection(buffered_bbox)
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

    selected_side = _selected_side_line(offshore, region)
    selected_side_xy = project_geometry(selected_side, projection)
    offshore_unit = _offshore_unit_vector(offshore, selected_side_xy)
    seed_xy, seed_meta = _resolve_seed(region, config, projection, bpoly_xy)
    forbidden_regions_lonlat: list[dict[str, Any]] = []
    forbidden_regions_xy: list[Polygon] = []

    anchors = _select_anchor_points(repaired_lines_xy, selected_side_xy, bpoly_xy, config.target_resolution_m)
    candidates = generate_offshore_arc_candidates(
        anchors["start_xy"],
        anchors["end_xy"],
        offshore_unit,
        selected_side_xy,
        bpoly_xy,
        config.target_resolution_m,
    )
    if len(repaired_lines_xy) <= 20_000:
        coast_union_xy = unary_union(repaired_lines_xy) if repaired_lines_xy else GeometryCollection()
    else:
        coast_union_xy = GeometryCollection()
    scored = score_and_select_bdry_arc(candidates, coast_union_xy, bpoly_xy, config.target_resolution_m)
    selected_arc_xy = scored["selected"]["geometry"]
    _write_progress(run_dir, "initial-arc", "done", {"candidate_count": len(scored["candidates"]), "selected": scored["selected"]["candidate_id"]})

    topology_mode_used = config.topology_mode
    topology_iterations: list[dict[str, Any]] = []
    if config.topology_mode == "gshhs-vector":
        wet_result = extract_gshhs_vector_wet_domain(
            repaired_lines_xy,
            land_polygons_xy,
            selected_arc_xy,
            bpoly_xy,
            seed_xy,
            config.target_resolution_m,
        )
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
    final_status, failure_taxonomy = _final_status(scored, wet_result, anchors, forbidden_regions_xy)

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
            "raster_resolution_m": config.raster_resolution_m,
            "max_topology_iterations": int(config.max_topology_iterations),
            "convergence_area_frac": float(config.convergence_area_frac),
            "convergence_anchor_m": float(config.convergence_anchor_m or 2.0 * config.target_resolution_m),
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
        },
        "coastline_audit": audit,
        "repair": repair_meta,
        "seed": seed_meta,
        "anchors": {
            "start_lonlat": [float(anchors_lonlat[0].x), float(anchors_lonlat[0].y)],
            "end_lonlat": [float(anchors_lonlat[1].x), float(anchors_lonlat[1].y)],
            "start_distance_to_side_endpoint_m": float(anchors["start_distance_m"]),
            "end_distance_to_side_endpoint_m": float(anchors["end_distance_m"]),
            "anchor_distance_m": float(Point(anchors["start_xy"]).distance(Point(anchors["end_xy"]))),
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
        },
        "outputs": {
            **outputs,
            "bdry_arc_review_map": str(final_map),
            "visual_review_dir": str(visual_dir) if config.mode == "test" else None,
        },
    }
    manifest_path = run_dir / "bdry_arc_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
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
) -> list[dict[str, Any]]:
    """Generate Bezier and bowed offshore arc candidates."""
    p0 = np.asarray(start_xy, dtype=float)
    p1 = np.asarray(end_xy, dtype=float)
    chord = p1 - p0
    chord_len = max(float(np.linalg.norm(chord)), 1.0)
    side_dist = 0.5 * (Point(p0).distance(selected_side_xy) + Point(p1).distance(selected_side_xy))
    base_bow = max(8.0 * target_resolution_m, 0.30 * chord_len, side_dist + 0.10 * chord_len)
    factors = [0.55, 0.8, 1.1, 1.45]
    candidates: list[dict[str, Any]] = []
    for idx, factor in enumerate(factors, start=1):
        bow = base_bow * factor
        candidates.append(
            {
                "candidate_id": f"bezier_{idx:02d}",
                "family": "bezier",
                "bow_distance_m": float(bow),
                "geometry": _bezier_arc(p0, p1, offshore_unit, bow, n=96),
            }
        )
        candidates.append(
            {
                "candidate_id": f"bowed_{idx:02d}",
                "family": "bowed",
                "bow_distance_m": float(bow),
                "geometry": _bowed_arc(p0, p1, offshore_unit, bow, n=96),
            }
        )
    return candidates


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


def extract_gshhs_vector_wet_domain(
    coastline_lines_xy: list[LineString],
    land_polygons_xy: list[Polygon],
    offshore_arc_xy: LineString,
    bpoly_xy: Polygon,
    seed_xy: Point,
    target_resolution_m: float,
) -> dict[str, Any]:
    """Build a GSHHS-first wet domain from stable land polygons and the open arc."""
    land_union = unary_union(land_polygons_xy).buffer(0) if land_polygons_xy else GeometryCollection()
    water_seed_xy = seed_xy
    seed_snap_distance_m = 0.0
    seed_snapped = False
    if not land_union.is_empty:
        water_geom_for_seed = bpoly_xy.difference(land_union).buffer(0)
        if not water_geom_for_seed.is_empty and not water_geom_for_seed.buffer(1.0).contains(seed_xy):
            try:
                water_seed_xy = nearest_points(seed_xy, water_geom_for_seed)[1]
                seed_snap_distance_m = float(seed_xy.distance(water_seed_xy))
                seed_snapped = seed_snap_distance_m > 0.0
            except Exception:
                water_seed_xy = seed_xy
    result = extract_seeded_wet_domain(
        coastline_lines_xy,
        offshore_arc_xy,
        bpoly_xy,
        water_seed_xy,
        [],
        target_resolution_m,
    )
    domain = result["wet_domain_xy"]
    metadata = dict(result["metadata"])
    source = metadata.get("source")
    used_fallback = source == "fallback_bpoly_polygon" or not metadata.get("seed_inside") or not coastline_lines_xy or not land_polygons_xy
    fallback_reason = metadata.get("fallback_reason")

    if not land_union.is_empty and not domain.is_empty:
        clipped = _choose_seed_component(domain.difference(land_union).buffer(0), water_seed_xy)
        if clipped is not None and not clipped.is_empty:
            domain = clipped
        else:
            used_fallback = True
            fallback_reason = fallback_reason or "seed_component_lost_after_land_difference"

    if used_fallback:
        water_geom = bpoly_xy.difference(land_union).buffer(0) if not land_union.is_empty else bpoly_xy.buffer(0)
        fallback_domain = _choose_seed_component(water_geom, water_seed_xy)
        if fallback_domain is not None and not fallback_domain.is_empty:
            domain = fallback_domain
            source = "gshhs_bpoly_minus_land_fallback"
        else:
            source = "gshhs_seed_water_component_not_found"
            fallback_reason = fallback_reason or "seed_water_component_not_found"
            domain = bpoly_xy.buffer(0)

    if not domain.is_valid:
        domain = domain.buffer(0)
    if getattr(domain, "geom_type", "") != "Polygon":
        selected = _choose_seed_component(domain, seed_xy)
        domain = selected if selected is not None else bpoly_xy.buffer(0)

    tolerance = max(2.0, 0.02 * target_resolution_m)
    arc_land = offshore_arc_xy.difference(Point(offshore_arc_xy.coords[0]).buffer(max(500.0, 3.0 * target_resolution_m)).union(Point(offshore_arc_xy.coords[-1]).buffer(max(500.0, 3.0 * target_resolution_m))))
    arc_land_intersection = False
    arc_land_intersection_length_m = 0.0
    if not land_union.is_empty and not arc_land.is_empty:
        inter = arc_land.intersection(land_union.buffer(tolerance))
        arc_land_intersection = not inter.is_empty
        arc_land_intersection_length_m = float(getattr(inter, "length", 0.0))

    metadata.update(
        {
            "source": source if str(source).startswith("gshhs") else "gshhs_vector_polygonize",
            "method": "gshhs_vector_polygonize_with_land_union",
            "land_polygon_count": int(len(land_polygons_xy)),
            "coastline_line_count": int(len(coastline_lines_xy)),
            "gshhs_missing_land_polygons": bool(len(land_polygons_xy) == 0),
            "gshhs_missing_coastline_lines": bool(len(coastline_lines_xy) == 0),
            "gshhs_polygonize_fallback_used": bool(used_fallback or str(source).endswith("_fallback")),
            "fallback_reason": fallback_reason,
            "arc_land_intersection": bool(arc_land_intersection),
            "arc_land_intersection_length_m": float(arc_land_intersection_length_m),
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
    return {"wet_domain_xy": domain, "metadata": metadata}


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
                "gshhs-coastline",
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
            "--resolution",
            resolution,
            "--levels",
            levels,
            "--formats",
            "gpkg,geojson",
            "--allow-no-basemap",
            "--quiet",
        ],
        check=True,
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


def _write_progress(run_dir: Path, stage: str, message: str, details: dict[str, Any] | None = None) -> None:
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "message": message,
        "details": _json_safe(details or {}),
    }
    with (run_dir / "bdry_arc_progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


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
    return (west - dlon, south - dlat, east + dlon, north + dlat)


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
    metadata = {
        "available_layers": layers,
        "selected_coastline_layer": coastline_layer,
        "selected_land_layer": land_layer,
        "coastline_feature_count": int(len(coastline)),
        "land_polygon_feature_count": int(len(land)),
        "coastline_source": coastline_source,
        "gshhs_manifest_path": str(_discover_gshhs_manifest(path)) if _discover_gshhs_manifest(path) else None,
    }
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
        if layer:
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
    if len(gdf) <= 20_000:
        try:
            gdf = gpd.clip(gdf, gpd.GeoSeries([box(*bbox_wsen)], crs="EPSG:4326")).reset_index(drop=True)
        except Exception:
            pass
    return gdf


def _discover_gshhs_manifest(path: Path) -> Path | None:
    candidates = sorted(path.parent.glob("*_gshhs_manifest.json"))
    return candidates[0] if candidates else None


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
    open_gdf = gpd.GeoDataFrame(
        [{"segment_class": "open_boundary", "geometry": selected_arc_lonlat}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    land_gdf = gpd.GeoDataFrame(
        [{"segment_class": "land_boundary_candidate", "geometry": LineString(wet_domain_lonlat.exterior.coords)}],
        geometry="geometry",
        crs="EPSG:4326",
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
            {"anchor_role": "start", "geometry": anchors_lonlat[0]},
            {"anchor_role": "end", "geometry": anchors_lonlat[1]},
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
    for layer_name in ("open_boundary_arc", "land_boundary_arcs", "island_holes"):
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
    layers["open_boundary_arc"].plot(ax=ax, color="#d00000", linewidth=2.2)
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
    layers["open_boundary_arc"].plot(ax=ax, color="#d00000", linewidth=2.2)
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
    layers["open_boundary_arc"].plot(ax=ax, color="#d00000", linewidth=2.2)
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
    if metrics.get("extra_coastline_intersection"):
        failures.append("open_arc_intersects_extra_coastline")
    if wet_result["metadata"].get("source") == "fallback_bpoly_polygon":
        failures.append("seeded_wet_domain_polygonize_failed")
    if wet_result["metadata"].get("gshhs_missing_land_polygons"):
        failures.append("gshhs_missing_land_polygons")
    if wet_result["metadata"].get("gshhs_missing_coastline_lines"):
        failures.append("gshhs_missing_coastline_lines")
    if wet_result["metadata"].get("arc_land_intersection"):
        failures.append("gshhs_open_arc_crosses_land")
    if not wet_result["metadata"].get("seed_inside"):
        failures.append("seed_not_inside_wet_domain")
    if wet_result["metadata"].get("forbidden_overlap"):
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
