"""Algorithmic open-boundary candidate design for coastline-aware FVCOM domains."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from .bathymetry import BathymetryGrid
from .domain import infer_offshore_side
from .projection import LocalProjection, local_utm_projection, project_geometry, unproject_geometry, unproject_points


OPEN_BOUNDARY_MODES = ("auto", "ellipse", "bezier", "bbox-bow", "manual-line", "anchor-iterate")


@dataclass(frozen=True)
class OpenBoundaryDesignResult:
    """Selected open boundary, updated domain polygon, and candidate diagnostics."""

    domain_polygon: Polygon
    open_boundary: LineString
    candidates: gpd.GeoDataFrame
    metadata: dict


def design_open_boundary(
    wet_domain_polygon: Polygon,
    bathy: BathymetryGrid,
    bbox_wsen: tuple[float, float, float, float],
    coastline: gpd.GeoDataFrame,
    target_resolution_m: float,
    open_spacing_m: float,
    offshore_side: str | None = None,
    mode: str = "auto",
    manual_open_boundary: str | Path | None = None,
    ocean_direction: tuple[float, float] | None = None,
    anchor_seeds: tuple[float, float, float, float] | None = None,
    anchor_seed_json: str | Path | None = None,
    anchor_max_iterations: int = 40,
    anchor_step_factor: float = 1.0,
    anchor_min_step_factor: float = 0.1,
    anchor_bbox_touch_fraction: float = 0.02,
    max_rounds: int = 3,
    min_depth_m: float = 0.05,
) -> OpenBoundaryDesignResult:
    """Generate, score, and select a reproducible smooth offshore open boundary."""
    if mode not in OPEN_BOUNDARY_MODES:
        raise ValueError(f"open_boundary_mode must be one of {OPEN_BOUNDARY_MODES}")
    if mode == "manual-line" and not manual_open_boundary:
        raise ValueError("--manual-open-boundary is required when --open-boundary-mode manual-line is used.")
    projection = local_utm_projection(bbox_wsen)
    if mode == "anchor-iterate":
        seed = _resolve_anchor_seed(ocean_direction, anchor_seeds, anchor_seed_json)
        return _design_anchor_iterate(
            wet_domain_polygon,
            bathy,
            bbox_wsen,
            coastline,
            target_resolution_m,
            projection,
            seed,
            max_iterations=anchor_max_iterations,
            step_factor=anchor_step_factor,
            min_step_factor=anchor_min_step_factor,
            bbox_touch_fraction=anchor_bbox_touch_fraction,
            min_depth_m=min_depth_m,
        )

    wet_xy = project_geometry(wet_domain_polygon, projection).buffer(0)
    if not isinstance(wet_xy, Polygon):
        if isinstance(wet_xy, MultiPolygon):
            wet_xy = max(wet_xy.geoms, key=lambda geom: geom.area)
        else:
            raise ValueError("Wet-domain geometry must be polygonal for open-boundary design.")
    bbox_xy = project_geometry(box(*bbox_wsen), projection)
    coastline_xy = _coastline_union_xy(coastline, projection)

    if mode == "manual-line":
        offshore_side = offshore_side or infer_offshore_side(bathy)
        manual_xy = project_geometry(_load_manual_line(manual_open_boundary), projection)
        candidates = [_score_candidate("manual_001", "manual-line", 1, manual_xy, wet_xy, bathy, projection, coastline_xy, offshore_side, min_depth_m)]
    else:
        candidates = []
        families = ("bbox-bow", "ellipse", "bezier") if mode == "auto" else (mode,)
        inferred_side = infer_offshore_side(bathy)
        side_options = [offshore_side] if offshore_side else _ordered_side_options(inferred_side, mode)
        for side in side_options:
            for round_id in range(1, max(1, int(max_rounds)) + 1):
                candidates.extend(
                    _generate_candidate_lines(
                        bbox_xy.bounds,
                        side,
                        families=families,
                        open_spacing_m=open_spacing_m,
                        round_id=round_id,
                    )
                )
        candidates = [
            _score_candidate(cid, family, round_id, line_xy, wet_xy, bathy, projection, coastline_xy, side, min_depth_m)
            for cid, family, round_id, side, line_xy in candidates
        ]

    candidates.sort(key=lambda item: (item["score"], item["wet_fraction"], item["median_depth_m"]), reverse=True)
    selected = candidates[0]
    offshore_side = selected["offshore_side"]
    domain_xy = _domain_cut_by_candidate(wet_xy, bbox_xy.bounds, selected["line_xy"], offshore_side)
    domain_lonlat = unproject_geometry(domain_xy, projection).buffer(0)
    if not isinstance(domain_lonlat, Polygon):
        if isinstance(domain_lonlat, MultiPolygon):
            domain_lonlat = max(domain_lonlat.geoms, key=lambda geom: geom.area)
        else:
            raise ValueError("Open-boundary candidate did not produce a polygonal domain.")
    open_lonlat = unproject_geometry(selected["line_xy"], projection)
    candidates_gdf = _candidate_gdf(candidates, projection, selected["candidate_id"])

    metadata = {
        "open_boundary_mode": mode,
        "offshore_side": offshore_side,
        "selected_candidate_id": selected["candidate_id"],
        "selected_family": selected["family"],
        "design_status": "pass_candidate"
        if selected["wet_fraction"] >= 0.85 and selected["inside_fraction"] >= 0.85
        else "needs_visual_or_data_review",
        "candidate_count": int(len(candidates)),
        "candidate_rounds": int(max_rounds),
        "manual_open_boundary": str(manual_open_boundary) if manual_open_boundary else None,
        "scoring": {
            "wet_fraction_min_to_prefer": 0.85,
            "inside_fraction_min_to_prefer": 0.85,
            "score_formula": "100*wet + 25*inside + depth_bonus + coast_distance_bonus - curvature_penalty - endpoint_penalty",
        },
        "selected_metrics": _jsonable_candidate(selected),
    }
    if metadata["design_status"] != "pass_candidate":
        metadata["warning"] = (
            "Selected open-boundary candidate does not have strong wet-domain support. "
            "Inspect bathymetry coverage and candidate review images before meshing."
        )
    return OpenBoundaryDesignResult(domain_polygon=domain_lonlat, open_boundary=open_lonlat, candidates=candidates_gdf, metadata=metadata)


def _load_manual_line(path: str | Path | None) -> LineString:
    if path is None:
        raise ValueError("Missing manual open-boundary path.")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Manual open-boundary file is empty: {path}")
    geom = gdf.geometry.iloc[0]
    if isinstance(geom, LineString):
        return geom
    if hasattr(geom, "geoms"):
        lines = [item for item in geom.geoms if isinstance(item, LineString)]
        if lines:
            return max(lines, key=lambda line: line.length)
    raise ValueError("Manual open-boundary file must contain a LineString geometry.")


def _coastline_union_xy(coastline: gpd.GeoDataFrame, projection: LocalProjection):
    if coastline is None or coastline.empty:
        return None
    if coastline.crs is None:
        coast = coastline.set_crs("EPSG:4326")
    else:
        coast = coastline.to_crs("EPSG:4326")
    projected = [project_geometry(geom, projection) for geom in coast.geometry if geom is not None and not geom.is_empty]
    return unary_union(projected) if projected else None


def _resolve_anchor_seed(
    ocean_direction: tuple[float, float] | None,
    anchor_seeds: tuple[float, float, float, float] | None,
    anchor_seed_json: str | Path | None,
) -> dict:
    if anchor_seed_json:
        data = json.loads(Path(anchor_seed_json).read_text(encoding="utf-8"))
        ocean_direction = tuple(float(v) for v in data.get("ocean_direction", ocean_direction or ()))
        anchor_seeds = tuple(float(v) for v in data.get("anchor_seeds", anchor_seeds or ()))
        reviewer = data.get("reviewer", "unknown")
        notes = data.get("notes", "")
    else:
        reviewer = "cli"
        notes = ""
    if ocean_direction is None or len(ocean_direction) != 2:
        raise ValueError("--ocean-direction DX DY or anchor_seed_json.ocean_direction is required for anchor-iterate mode.")
    if anchor_seeds is None or len(anchor_seeds) != 4:
        raise ValueError("--anchor-seeds lon1 lat1 lon2 lat2 or anchor_seed_json.anchor_seeds is required for anchor-iterate mode.")
    direction = np.asarray(ocean_direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("ocean_direction must be a nonzero vector.")
    return {
        "ocean_direction": (float(direction[0] / norm), float(direction[1] / norm)),
        "anchor_seeds": tuple(float(v) for v in anchor_seeds),
        "reviewer": reviewer,
        "notes": notes,
    }


def _design_anchor_iterate(
    wet_domain_polygon: Polygon,
    bathy: BathymetryGrid,
    bbox_wsen: tuple[float, float, float, float],
    coastline: gpd.GeoDataFrame,
    target_resolution_m: float,
    projection: LocalProjection,
    seed: dict,
    max_iterations: int = 40,
    step_factor: float = 1.0,
    min_step_factor: float = 0.1,
    bbox_touch_fraction: float = 0.02,
    min_depth_m: float = 0.05,
) -> OpenBoundaryDesignResult:
    wet_xy = project_geometry(wet_domain_polygon, projection).buffer(0)
    if isinstance(wet_xy, MultiPolygon):
        wet_xy = max(wet_xy.geoms, key=lambda geom: geom.area)
    if not isinstance(wet_xy, Polygon):
        raise ValueError("Wet-domain geometry must be polygonal for anchor-iterate mode.")

    coastline_xy = _coastline_union_xy(coastline, projection)
    if coastline_xy is None or coastline_xy.is_empty:
        raise ValueError("Anchor iteration requires nonempty coastline linework.")

    bbox_xy = project_geometry(box(*bbox_wsen), projection)
    direction_xy = np.asarray(seed["ocean_direction"], dtype=float)
    direction_xy = direction_xy / max(float(np.linalg.norm(direction_xy)), 1.0e-12)
    seed_lonlat = np.asarray(
        [[seed["anchor_seeds"][0], seed["anchor_seeds"][1]], [seed["anchor_seeds"][2], seed["anchor_seeds"][3]]],
        dtype=float,
    )
    x, y = projection.to_xy.transform(seed_lonlat[:, 0], seed_lonlat[:, 1])
    probes = np.column_stack([x, y])
    step_factor = max(float(step_factor), 0.05)
    min_step_factor = max(float(min_step_factor), 0.01)
    bbox_touch_fraction = max(float(bbox_touch_fraction), 0.0)
    step = step_factor * target_resolution_m
    min_step = min(min_step_factor * target_resolution_m, step)
    duplicate_tol = max(1.0, 0.001 * target_resolution_m)
    side = _side_from_direction(direction_xy)
    target_gap = _bbox_touch_target_gap(bbox_xy.bounds, side, bbox_touch_fraction)
    outer_envelope = _build_outer_envelope_index(coastline_xy, direction_xy, target_resolution_m)
    outer_tolerance_m = 0.5 * target_resolution_m
    previous_count: int | None = None
    previous_signature: tuple | None = None
    best: dict | None = None
    history = []
    iteration_geoms = []
    locked = [False, False]
    repeated_state_at_min_step = 0
    stop_reason = "max_iterations"

    for iteration in range(1, max(1, int(max_iterations)) + 1):
        variant = _select_anchor_arc_variant(
            probes,
            direction_xy,
            bbox_xy.bounds,
            coastline_xy,
            bathy,
            projection,
            min_depth_m,
            duplicate_tol,
            target_gap,
            outer_envelope=outer_envelope,
            outer_tolerance_m=outer_tolerance_m,
        )
        arc = variant["arc"]
        intersections = variant["intersections"]
        count = len(intersections)
        wet_fraction = float(variant["wet_fraction"])
        median_depth = float(variant["median_depth_m"])
        closed_polygon, closed_valid = _closed_polygon_from_arc_and_side(arc, bbox_xy.bounds, side)
        valid_domain = bool(closed_valid and closed_polygon.intersects(wet_xy))
        endpoint_state = _classify_endpoint_anchors(arc, intersections, outer_envelope, outer_tolerance_m)
        record = {
            "iteration": iteration,
            "intersection_count": count,
            "step_m": float(step),
            "wet_fraction": float(wet_fraction),
            "median_depth_m": float(median_depth),
            "closed_polygon_valid": bool(valid_domain),
            "bow_factor": float(variant["bow_factor"]),
            "bbox_touch_distance_m": float(variant["bbox_touch_distance_m"]),
            "bbox_touch_error_m": float(variant["bbox_touch_error_m"]),
            "endpoint_tangent_error_deg": float(variant["endpoint_tangent_error_deg"]),
        }
        record.update(_endpoint_state_json(endpoint_state, projection))
        history.append(record)
        iteration_geoms.append(
            {
                "candidate_id": f"anchor_iter_{iteration:03d}",
                "family": "anchor-iterate",
                "round": iteration,
                "score": _anchor_score(count, valid_domain, wet_fraction, median_depth),
                "status": "converged" if count == 2 and valid_domain else "iterating",
                "wet_fraction": float(wet_fraction),
                "inside_fraction": 1.0 if valid_domain else 0.0,
                "median_depth_m": float(median_depth),
                "intersection_count": int(count),
                "step_m": float(step),
                "bow_factor": float(variant["bow_factor"]),
                "bbox_touch_distance_m": float(variant["bbox_touch_distance_m"]),
                "bbox_touch_error_m": float(variant["bbox_touch_error_m"]),
                "endpoint_tangent_error_deg": float(variant["endpoint_tangent_error_deg"]),
                "start_anchor_status": endpoint_state["start_anchor_status"],
                "end_anchor_status": endpoint_state["end_anchor_status"],
                "middle_extra_intersection_count": int(endpoint_state["middle_extra_intersection_count"]),
                "selected": False,
                "geometry": arc,
            }
        )
        if _endpoint_state_has_valid_pair(endpoint_state) and endpoint_state["middle_extra_intersection_count"] == 0 and count == 2:
            anchor_xy = np.asarray([endpoint_state["start_anchor_xy"], endpoint_state["end_anchor_xy"]], dtype=float)
            final_variant = _select_anchor_arc_variant(
                anchor_xy,
                direction_xy,
                bbox_xy.bounds,
                coastline_xy,
                bathy,
                projection,
                min_depth_m,
                duplicate_tol,
                target_gap,
                outer_envelope=outer_envelope,
                outer_tolerance_m=outer_tolerance_m,
            )
            trimmed = final_variant["arc"]
            final_intersections = final_variant["intersections"]
            final_count = len(final_intersections)
            final_endpoint_state = _classify_endpoint_anchors(trimmed, final_intersections, outer_envelope, outer_tolerance_m)
            outer_report = _endpoint_outer_report(final_endpoint_state)
            outer_pass = _endpoint_state_has_valid_pair(final_endpoint_state)
            final_middle_extra = int(final_endpoint_state["middle_extra_intersection_count"])
            final_closed, final_closed_valid = _closed_polygon_from_arc_and_side(trimmed, bbox_xy.bounds, side)
            final_valid_domain = bool(final_closed_valid and final_closed.intersects(wet_xy))
            history[-1]["final_rebuild_intersection_count"] = int(final_count)
            history[-1]["final_rebuild_bbox_touch_distance_m"] = float(final_variant["bbox_touch_distance_m"])
            history[-1]["final_rebuild_endpoint_tangent_error_deg"] = float(final_variant["endpoint_tangent_error_deg"])
            history[-1]["outer_envelope_anchor_pass"] = bool(outer_pass)
            history[-1]["outer_envelope_anchor_report"] = outer_report
            history[-1]["final_rebuild_middle_extra_intersection_count"] = final_middle_extra
            history[-1]["final_rebuild_start_anchor_status"] = final_endpoint_state["start_anchor_status"]
            history[-1]["final_rebuild_end_anchor_status"] = final_endpoint_state["end_anchor_status"]
            if (
                final_count == 2
                and final_middle_extra == 0
                and outer_pass
                and final_valid_domain
                and float(final_variant["endpoint_tangent_error_deg"]) <= 1.0e-6
            ):
                final_anchor_xy = np.asarray([final_endpoint_state["start_anchor_xy"], final_endpoint_state["end_anchor_xy"]], dtype=float)
                best = {
                    "status": "converged",
                    "arc_xy": trimmed,
                    "anchors_xy": final_anchor_xy,
                    "iterations": history,
                    "intersection_count": final_count,
                    "wet_fraction": float(final_variant["wet_fraction"]),
                    "median_depth_m": float(final_variant["median_depth_m"]),
                    "closed_polygon_valid": bool(final_valid_domain),
                    "stop_reason": "exactly_two_intersections",
                    "bow_factor": float(final_variant["bow_factor"]),
                    "bbox_touch_distance_m": float(final_variant["bbox_touch_distance_m"]),
                    "bbox_touch_error_m": float(final_variant["bbox_touch_error_m"]),
                    "bbox_touch_point_xy": final_variant["bbox_touch_point_xy"],
                    "endpoint_tangent_error_deg": float(final_variant["endpoint_tangent_error_deg"]),
                    "outer_envelope_anchor_report": outer_report,
                    "outer_envelope_anchor_pass": bool(outer_pass),
                    "start_anchor_status": final_endpoint_state["start_anchor_status"],
                    "end_anchor_status": final_endpoint_state["end_anchor_status"],
                    "middle_extra_intersection_count": final_middle_extra,
                }
                iteration_geoms[-1]["selected"] = True
                iteration_geoms[-1]["geometry"] = trimmed
                iteration_geoms[-1]["wet_fraction"] = float(final_variant["wet_fraction"])
                iteration_geoms[-1]["median_depth_m"] = float(final_variant["median_depth_m"])
                iteration_geoms[-1]["intersection_count"] = int(final_count)
                iteration_geoms[-1]["bow_factor"] = float(final_variant["bow_factor"])
                iteration_geoms[-1]["bbox_touch_distance_m"] = float(final_variant["bbox_touch_distance_m"])
                iteration_geoms[-1]["bbox_touch_error_m"] = float(final_variant["bbox_touch_error_m"])
                iteration_geoms[-1]["endpoint_tangent_error_deg"] = float(final_variant["endpoint_tangent_error_deg"])
                iteration_geoms[-1]["outer_envelope_anchor_pass"] = bool(outer_pass)
                break
            count = final_count
            intersections = final_intersections
            arc = trimmed
            endpoint_state = final_endpoint_state
            iteration_status = "final_rebuild_needs_iteration"
            iteration_geoms[-1]["status"] = iteration_status
            iteration_geoms[-1]["geometry"] = trimmed
            iteration_geoms[-1]["intersection_count"] = int(final_count)
            iteration_geoms[-1]["wet_fraction"] = float(final_variant["wet_fraction"])
            iteration_geoms[-1]["median_depth_m"] = float(final_variant["median_depth_m"])
            iteration_geoms[-1]["bow_factor"] = float(final_variant["bow_factor"])
            iteration_geoms[-1]["bbox_touch_distance_m"] = float(final_variant["bbox_touch_distance_m"])
            iteration_geoms[-1]["bbox_touch_error_m"] = float(final_variant["bbox_touch_error_m"])
            iteration_geoms[-1]["endpoint_tangent_error_deg"] = float(final_variant["endpoint_tangent_error_deg"])
            iteration_geoms[-1]["outer_envelope_anchor_pass"] = bool(outer_pass)

        if previous_count is not None and ((previous_count == 0 and count > 2) or (previous_count > 2 and count == 0)):
            step = max(min_step, 0.5 * step)

        movement_actions: list[str] = []
        for idx, prefix in enumerate(("start", "end")):
            status_key = f"{prefix}_anchor_status"
            point_key = f"{prefix}_anchor_xy"
            status_value = endpoint_state[status_key]
            anchor_point = endpoint_state[point_key]
            if status_value == "valid_outer_envelope" and anchor_point is not None:
                probes[idx] = np.asarray(anchor_point, dtype=float)
                locked[idx] = True
                movement_actions.append(f"freeze_{prefix}")
            elif status_value == "missing":
                locked[idx] = False
                probes[idx] = probes[idx] - direction_xy * step
                movement_actions.append(f"move_{prefix}_landward")
            else:
                locked[idx] = False
                probes[idx] = probes[idx] + direction_xy * step
                movement_actions.append(f"move_{prefix}_oceanward")

        if endpoint_state["middle_extra_intersection_count"] > 0:
            step = max(min_step, 0.5 * step)
            movement_actions.append("reduce_step_for_middle_extras")

        record["start_locked"] = bool(locked[0])
        record["end_locked"] = bool(locked[1])
        record["movement_action"] = "_".join(movement_actions) if movement_actions else "none"

        signature = _endpoint_state_signature(endpoint_state, count)
        if previous_signature == signature and step <= min_step:
            repeated_state_at_min_step += 1
        else:
            repeated_state_at_min_step = 0
        previous_count = count
        previous_signature = signature
        if repeated_state_at_min_step >= 5:
            stop_reason = "stalled_endpoint_motion"
            break

    if best is None:
        variant = _select_anchor_arc_variant(
            probes,
            direction_xy,
            bbox_xy.bounds,
            coastline_xy,
            bathy,
            projection,
            min_depth_m,
            duplicate_tol,
            target_gap,
            outer_envelope=outer_envelope,
            outer_tolerance_m=outer_tolerance_m,
        )
        arc = variant["arc"]
        intersections = variant["intersections"]
        endpoint_state = _classify_endpoint_anchors(arc, intersections, outer_envelope, outer_tolerance_m)
        anchor_xy = np.empty((0, 2))
        best = {
            "status": "needs_visual_seed",
            "arc_xy": arc,
            "anchors_xy": anchor_xy,
            "iterations": history,
            "intersection_count": int(len(intersections)),
            "wet_fraction": float(variant["wet_fraction"]),
            "median_depth_m": float(variant["median_depth_m"]),
            "closed_polygon_valid": False,
            "bow_factor": float(variant["bow_factor"]),
            "bbox_touch_distance_m": float(variant["bbox_touch_distance_m"]),
            "bbox_touch_error_m": float(variant["bbox_touch_error_m"]),
            "bbox_touch_point_xy": variant["bbox_touch_point_xy"],
            "endpoint_tangent_error_deg": float(variant["endpoint_tangent_error_deg"]),
            "outer_envelope_anchor_report": _endpoint_outer_report(endpoint_state),
            "outer_envelope_anchor_pass": _endpoint_state_has_valid_pair(endpoint_state),
            "start_anchor_status": endpoint_state["start_anchor_status"],
            "end_anchor_status": endpoint_state["end_anchor_status"],
            "middle_extra_intersection_count": int(endpoint_state["middle_extra_intersection_count"]),
            "stop_reason": stop_reason,
        }
        iteration_geoms.append(
            {
                "candidate_id": "anchor_iter_final_failed",
                "family": "anchor-iterate",
                "round": len(history) + 1,
                "score": _anchor_score(len(intersections), False, variant["wet_fraction"], variant["median_depth_m"]),
                "status": "needs_visual_seed",
                "wet_fraction": float(variant["wet_fraction"]),
                "inside_fraction": 0.0,
                "median_depth_m": float(variant["median_depth_m"]),
                "intersection_count": int(len(intersections)),
                "step_m": float(step),
                "bow_factor": float(variant["bow_factor"]),
                "bbox_touch_distance_m": float(variant["bbox_touch_distance_m"]),
                "bbox_touch_error_m": float(variant["bbox_touch_error_m"]),
                "endpoint_tangent_error_deg": float(variant["endpoint_tangent_error_deg"]),
                "start_anchor_status": endpoint_state["start_anchor_status"],
                "end_anchor_status": endpoint_state["end_anchor_status"],
                "middle_extra_intersection_count": int(endpoint_state["middle_extra_intersection_count"]),
                "selected": True,
                "geometry": arc,
            }
        )

    arc_xy = best["arc_xy"]
    open_lonlat = unproject_geometry(arc_xy, projection)
    domain_xy = _domain_cut_by_candidate(wet_xy, bbox_xy.bounds, arc_xy, side)
    domain_lonlat = unproject_geometry(domain_xy, projection).buffer(0)
    if isinstance(domain_lonlat, MultiPolygon):
        domain_lonlat = max(domain_lonlat.geoms, key=lambda geom: geom.area)
    candidates_gdf = _anchor_iteration_gdf(iteration_geoms, projection)
    anchors_lonlat = unproject_points(best["anchors_xy"], projection) if len(best["anchors_xy"]) else np.empty((0, 2))
    status = "pass_candidate" if best["status"] == "converged" and best["wet_fraction"] >= 0.85 else "needs_visual_or_data_review"
    metadata = {
        "open_boundary_mode": "anchor-iterate",
        "offshore_side": side,
        "selected_candidate_id": "anchor_iter_final",
        "selected_family": "anchor-iterate",
        "design_status": status,
        "candidate_count": int(len(iteration_geoms)),
        "candidate_rounds": int(len(history)),
        "seed": {
            "ocean_direction": [float(direction_xy[0]), float(direction_xy[1])],
            "anchor_seeds": [float(v) for v in seed["anchor_seeds"]],
            "reviewer": seed.get("reviewer"),
            "notes": seed.get("notes", ""),
        },
        "anchor_iteration": {
            "status": best["status"],
            "stop_reason": best["stop_reason"],
            "intersection_count": int(best["intersection_count"]),
            "max_iterations": int(max_iterations),
            "initial_step_m": float(step_factor * target_resolution_m),
            "min_step_m": float(min_step),
            "duplicate_intersection_tolerance_m": float(duplicate_tol),
            "coastline_intersection_source": "full_unpruned",
            "coastline_pruning": "none",
            "outer_envelope_tolerance_m": float(outer_tolerance_m),
            "outer_envelope_sample_spacing_m": float(outer_envelope.get("sample_spacing_m", 0.0)),
            "outer_envelope_bin_size_m": float(outer_envelope.get("bin_size_m", 0.0)),
            "outer_envelope_preview_lonlat": unproject_points(outer_envelope["preview_points_xy"], projection).tolist()
            if len(outer_envelope.get("preview_points_xy", []))
            else [],
            "outer_envelope_anchor_pass": bool(best["outer_envelope_anchor_pass"]),
            "outer_envelope_anchor_report": best["outer_envelope_anchor_report"],
            "step_factor": float(step_factor),
            "min_step_factor": float(min_step_factor),
            "bbox_touch_fraction": float(bbox_touch_fraction),
            "bbox_touch_target_gap_m": float(target_gap),
            "bow_factor": float(best["bow_factor"]),
            "bbox_touch_distance_m": float(best["bbox_touch_distance_m"]),
            "bbox_touch_error_m": float(best["bbox_touch_error_m"]),
            "bbox_touch_point_lonlat": unproject_points(np.asarray([best["bbox_touch_point_xy"]], dtype=float), projection)[0].tolist(),
            "endpoint_tangent_error_deg": float(best["endpoint_tangent_error_deg"]),
            "outer_envelope_anchor_pass": bool(best["outer_envelope_anchor_pass"]),
            "closed_polygon_valid": bool(best["closed_polygon_valid"]),
            "wet_fraction": float(best["wet_fraction"]),
            "median_depth_m": float(best["median_depth_m"]),
            "start_anchor_status": best.get("start_anchor_status"),
            "end_anchor_status": best.get("end_anchor_status"),
            "middle_extra_intersection_count": int(best.get("middle_extra_intersection_count", 0)),
            "anchor_points_lonlat": anchors_lonlat.tolist(),
            "history": history,
        },
        "selected_metrics": {
            "candidate_id": "anchor_iter_final",
            "family": "anchor-iterate",
            "score": _anchor_score(best["intersection_count"], best["closed_polygon_valid"], best["wet_fraction"], best["median_depth_m"]),
            "wet_fraction": float(best["wet_fraction"]),
            "inside_fraction": 1.0 if best["closed_polygon_valid"] else 0.0,
            "median_depth_m": float(best["median_depth_m"]),
            "intersection_count": int(best["intersection_count"]),
            "offshore_side": side,
            "bow_factor": float(best["bow_factor"]),
            "bbox_touch_distance_m": float(best["bbox_touch_distance_m"]),
            "bbox_touch_error_m": float(best["bbox_touch_error_m"]),
            "bbox_touch_point_lonlat": unproject_points(np.asarray([best["bbox_touch_point_xy"]], dtype=float), projection)[0].tolist(),
            "endpoint_tangent_error_deg": float(best["endpoint_tangent_error_deg"]),
            "outer_envelope_anchor_pass": bool(best["outer_envelope_anchor_pass"]),
        },
    }
    if status != "pass_candidate":
        metadata["warning"] = (
            "Anchor-iterate did not produce a fully accepted open boundary. "
            "Inspect the anchor iteration review and supply a new visual seed if needed."
        )
    return OpenBoundaryDesignResult(domain_polygon=domain_lonlat, open_boundary=open_lonlat, candidates=candidates_gdf, metadata=metadata)


def _pruned_coastline_union_xy(coastline: gpd.GeoDataFrame, projection: LocalProjection, min_length_m: float):
    if coastline is None or coastline.empty:
        return None
    coast = coastline.set_crs("EPSG:4326") if coastline.crs is None else coastline.to_crs("EPSG:4326")
    lines = []
    for geom in coast.geometry:
        for line in _iter_lines(geom):
            projected = project_geometry(line, projection)
            if projected.length >= min_length_m:
                lines.append(projected)
    return unary_union(lines) if lines else None


def _iter_lines(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            yield line
    elif isinstance(geom, GeometryCollection):
        for item in geom.geoms:
            yield from _iter_lines(item)


def _build_outer_envelope_index(linework, ocean_direction: np.ndarray, target_resolution_m: float) -> dict:
    sample_spacing = max(25.0, min(0.25 * float(target_resolution_m), 500.0))
    points = _sample_linework_points_xy(linework, sample_spacing)
    if points.size == 0:
        return {
            "points_xy": np.empty((0, 2)),
            "preview_points_xy": np.empty((0, 2)),
            "max_s_by_bin": {},
            "q_min": 0.0,
            "bin_size_m": float(target_resolution_m),
            "sample_spacing_m": sample_spacing,
            "direction": np.asarray(ocean_direction, dtype=float),
            "perpendicular": np.asarray([0.0, 1.0]),
        }
    direction = np.asarray(ocean_direction, dtype=float)
    direction = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=float)
    s = points @ direction
    q = points @ perpendicular
    q_min = float(np.nanmin(q))
    bin_size = max(float(target_resolution_m), sample_spacing)
    bins = np.floor((q - q_min) / bin_size).astype(int)
    max_s_by_bin: dict[int, float] = {}
    for bin_id in np.unique(bins):
        mask = bins == bin_id
        max_s_by_bin[int(bin_id)] = float(np.nanmax(s[mask]))
    preview_tolerance = 0.5 * float(target_resolution_m)
    outer_mask = np.asarray([max_s_by_bin[int(bin_id)] - s[idx] <= preview_tolerance for idx, bin_id in enumerate(bins)], dtype=bool)
    preview = points[outer_mask]
    if len(preview) > 1500:
        pick = np.linspace(0, len(preview) - 1, 1500).astype(int)
        preview = preview[pick]
    return {
        "points_xy": points,
        "preview_points_xy": preview,
        "max_s_by_bin": max_s_by_bin,
        "q_min": q_min,
        "bin_size_m": float(bin_size),
        "sample_spacing_m": float(sample_spacing),
        "direction": direction,
        "perpendicular": perpendicular,
    }


def _sample_linework_points_xy(linework, spacing_m: float) -> np.ndarray:
    points = []
    for line in _iter_lines(linework):
        if line.length <= 0.0:
            coords = list(line.coords)
            if coords:
                points.append(coords[0])
            continue
        n = max(2, int(np.ceil(line.length / max(spacing_m, 1.0))) + 1)
        for distance in np.linspace(0.0, line.length, n):
            point = line.interpolate(float(distance))
            points.append((point.x, point.y))
    return np.asarray(points, dtype=float) if points else np.empty((0, 2), dtype=float)


def _anchor_outer_envelope_report(anchor_xy: np.ndarray, envelope: dict, tolerance_m: float) -> list[dict]:
    anchors = np.asarray(anchor_xy, dtype=float)
    if anchors.size == 0:
        return []
    direction = np.asarray(envelope["direction"], dtype=float)
    perpendicular = np.asarray(envelope["perpendicular"], dtype=float)
    q_min = float(envelope["q_min"])
    bin_size = max(float(envelope["bin_size_m"]), 1.0)
    max_s_by_bin: dict[int, float] = envelope["max_s_by_bin"]
    report = []
    for idx, point in enumerate(anchors):
        s = float(point @ direction)
        q = float(point @ perpendicular)
        bin_id = int(np.floor((q - q_min) / bin_size))
        local_bins = [item for item in (bin_id - 1, bin_id, bin_id + 1) if item in max_s_by_bin]
        if local_bins:
            local_max_s = max(max_s_by_bin[item] for item in local_bins)
            offshore_deficit = max(0.0, float(local_max_s - s))
            passed = bool(offshore_deficit <= tolerance_m)
        else:
            local_max_s = float("nan")
            offshore_deficit = float("inf")
            passed = False
        report.append(
            {
                "anchor_index": int(idx + 1),
                "bin_id": int(bin_id),
                "outer_envelope_pass": passed,
                "passed": passed,
                "offshore_deficit_m": float(offshore_deficit),
                "local_outer_projection_m": float(local_max_s) if np.isfinite(local_max_s) else None,
                "anchor_projection_m": float(s),
                "tolerance_m": float(tolerance_m),
            }
        )
    return report


def _classify_endpoint_anchors(arc: LineString, intersections: list[Point], outer_envelope: dict, outer_tolerance_m: float) -> dict:
    items = []
    length = max(float(arc.length), 1.0e-12)
    for point in intersections:
        distance = float(arc.project(point))
        s_norm = float(distance / length)
        xy = np.asarray([point.x, point.y], dtype=float)
        report = _anchor_outer_envelope_report(np.asarray([xy]), outer_envelope, outer_tolerance_m)
        outer = report[0] if report else {
            "outer_envelope_pass": False,
            "passed": False,
            "offshore_deficit_m": float("inf"),
            "tolerance_m": float(outer_tolerance_m),
        }
        if s_norm <= 0.18:
            zone = "start"
        elif s_norm >= 0.82:
            zone = "end"
        else:
            zone = "middle"
        items.append(
            {
                "point": point,
                "xy": xy,
                "s_norm": s_norm,
                "zone": zone,
                "outer_report": outer,
                "outer_envelope_pass": bool(outer.get("passed", outer.get("outer_envelope_pass", False))),
                "offshore_deficit_m": float(outer.get("offshore_deficit_m", float("inf"))),
            }
        )
    start_items = sorted([item for item in items if item["zone"] == "start"], key=lambda item: item["s_norm"])
    end_items = sorted([item for item in items if item["zone"] == "end"], key=lambda item: item["s_norm"], reverse=True)
    middle_items = [item for item in items if item["zone"] == "middle"]
    start = _classify_endpoint_zone(start_items, "start", outer_tolerance_m)
    end = _classify_endpoint_zone(end_items, "end", outer_tolerance_m)
    return {
        "items": items,
        "start_zone_count": len(start_items),
        "end_zone_count": len(end_items),
        "middle_extra_intersection_count": len(middle_items),
        "start_anchor_status": start["status"],
        "end_anchor_status": end["status"],
        "start_anchor_xy": start["anchor_xy"],
        "end_anchor_xy": end["anchor_xy"],
        "start_anchor_report": start["anchor_report"],
        "end_anchor_report": end["anchor_report"],
    }


def _classify_endpoint_zone(items: list[dict], endpoint_name: str, outer_tolerance_m: float) -> dict:
    if not items:
        return {"status": "missing", "anchor_xy": None, "anchor_report": None}
    outer_items = [item for item in items if item["outer_envelope_pass"]]
    anchor = outer_items[0] if outer_items else items[0]
    if len(items) > 1:
        status = "blocked_by_extras"
    elif anchor["outer_envelope_pass"]:
        status = "valid_outer_envelope"
    elif anchor["offshore_deficit_m"] <= 2.0 * float(outer_tolerance_m):
        status = "candidate"
    else:
        status = "invalid_inner"
    report = dict(anchor["outer_report"] or {})
    report["endpoint"] = endpoint_name
    report["arc_fraction"] = float(anchor["s_norm"])
    return {"status": status, "anchor_xy": np.asarray(anchor["xy"], dtype=float), "anchor_report": report}


def _endpoint_state_has_valid_pair(endpoint_state: dict) -> bool:
    return (
        endpoint_state.get("start_anchor_status") == "valid_outer_envelope"
        and endpoint_state.get("end_anchor_status") == "valid_outer_envelope"
        and endpoint_state.get("start_anchor_xy") is not None
        and endpoint_state.get("end_anchor_xy") is not None
    )


def _endpoint_outer_report(endpoint_state: dict) -> list[dict]:
    report = []
    for key in ("start_anchor_report", "end_anchor_report"):
        item = endpoint_state.get(key)
        if item:
            report.append(item)
    return report


def _endpoint_state_signature(endpoint_state: dict, intersection_count: int) -> tuple:
    return (
        int(intersection_count),
        endpoint_state.get("start_anchor_status"),
        endpoint_state.get("end_anchor_status"),
        int(endpoint_state.get("start_zone_count", 0)),
        int(endpoint_state.get("end_zone_count", 0)),
        int(endpoint_state.get("middle_extra_intersection_count", 0)),
    )


def _endpoint_state_json(endpoint_state: dict, projection: LocalProjection) -> dict:
    data = {
        "start_anchor_status": endpoint_state["start_anchor_status"],
        "end_anchor_status": endpoint_state["end_anchor_status"],
        "start_zone_count": int(endpoint_state["start_zone_count"]),
        "end_zone_count": int(endpoint_state["end_zone_count"]),
        "middle_extra_intersection_count": int(endpoint_state["middle_extra_intersection_count"]),
    }
    for prefix in ("start", "end"):
        xy = endpoint_state.get(f"{prefix}_anchor_xy")
        report = endpoint_state.get(f"{prefix}_anchor_report")
        if xy is not None:
            lonlat = unproject_points(np.asarray([xy], dtype=float), projection)[0]
            data[f"{prefix}_anchor_lonlat"] = [float(lonlat[0]), float(lonlat[1])]
        else:
            data[f"{prefix}_anchor_lonlat"] = None
        if report:
            data[f"{prefix}_anchor_outer_deficit_m"] = report.get("offshore_deficit_m")
            data[f"{prefix}_anchor_arc_fraction"] = report.get("arc_fraction")
        else:
            data[f"{prefix}_anchor_outer_deficit_m"] = None
            data[f"{prefix}_anchor_arc_fraction"] = None
    return data


def _select_anchor_arc_variant(
    probes: np.ndarray,
    ocean_direction: np.ndarray,
    bounds: tuple[float, float, float, float],
    coastline_xy,
    bathy: BathymetryGrid,
    projection: LocalProjection,
    min_depth_m: float,
    duplicate_tol: float,
    target_gap_m: float,
    outer_envelope: dict | None = None,
    outer_tolerance_m: float = 0.0,
) -> dict:
    variants = []
    side = _side_from_direction(ocean_direction)
    for bow_factor in (0.35, 0.55, 0.8, 1.1, 1.45, 1.85, 2.35):
        variant = _directional_bezier_variant(probes[0], probes[1], ocean_direction, bounds, bow_factor=bow_factor, target_gap_m=target_gap_m)
        arc = variant["arc"]
        intersections = _line_intersections_on_arc(arc, coastline_xy, duplicate_tol)
        count = len(intersections)
        wet_fraction, median_depth = _arc_bathy_stats(arc, bathy, projection, min_depth_m)
        closed_polygon, closed_valid = _closed_polygon_from_arc_and_side(arc, bounds, side)
        touch_distance, touch_point = _bbox_touch_info(arc, bounds, side)
        touch_error = abs(touch_distance - target_gap_m)
        samples = _sample_line_xy(arc, 96)
        score = _anchor_variant_score(
            count,
            closed_valid,
            wet_fraction,
            median_depth,
            touch_error,
            touch_distance,
            variant["endpoint_tangent_error_deg"],
            _curvature_penalty(samples),
            bounds,
            side,
        )
        endpoint_state = None
        if outer_envelope is not None:
            endpoint_state = _classify_endpoint_anchors(arc, intersections, outer_envelope, outer_tolerance_m)
            valid_count = int(endpoint_state["start_anchor_status"] == "valid_outer_envelope") + int(
                endpoint_state["end_anchor_status"] == "valid_outer_envelope"
            )
            blocked_count = int(endpoint_state["start_anchor_status"] == "blocked_by_extras") + int(
                endpoint_state["end_anchor_status"] == "blocked_by_extras"
            )
            score += 95.0 * valid_count
            score -= 35.0 * int(endpoint_state["middle_extra_intersection_count"])
            score -= 25.0 * blocked_count
        variants.append(
            {
                "arc": arc,
                "intersections": intersections,
                "score": float(score),
                "intersection_count": int(count),
                "wet_fraction": float(wet_fraction),
                "median_depth_m": float(median_depth),
                "closed_polygon_valid": bool(closed_valid and not closed_polygon.is_empty),
                "bow_factor": float(bow_factor),
                "bbox_touch_distance_m": float(touch_distance),
                "bbox_touch_error_m": float(touch_error),
                "bbox_touch_point_xy": np.asarray([touch_point.x, touch_point.y], dtype=float),
                "endpoint_tangent_error_deg": float(variant["endpoint_tangent_error_deg"]),
                "endpoint_state": endpoint_state,
            }
        )
    variants.sort(key=lambda item: item["score"], reverse=True)
    return variants[0]


def _bezier_from_probes(
    p0: np.ndarray,
    p3: np.ndarray,
    ocean_direction: np.ndarray,
    bounds: tuple[float, float, float, float],
    bow_factor: float = 1.0,
    bbox_touch_fraction: float = 0.02,
) -> LineString:
    side = _side_from_direction(np.asarray(ocean_direction, dtype=float))
    target_gap_m = _bbox_touch_target_gap(bounds, side, bbox_touch_fraction)
    return _directional_bezier_variant(p0, p3, ocean_direction, bounds, bow_factor=bow_factor, target_gap_m=target_gap_m)["arc"]


def _directional_bezier_variant(
    p0: np.ndarray,
    p3: np.ndarray,
    ocean_direction: np.ndarray,
    bounds: tuple[float, float, float, float],
    bow_factor: float,
    target_gap_m: float,
) -> dict:
    p0 = np.asarray(p0, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    direction = np.asarray(ocean_direction, dtype=float)
    direction = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    span = max(width, height)
    side = _side_from_direction(direction)
    axis_extent = width if side in {"east", "west"} else height
    endpoint_gap = max(_bbox_touch_distance_for_points(np.vstack([p0, p3]), bounds, side) - target_gap_m, 0.0)
    chord = max(float(np.linalg.norm(p3 - p0)), 1.0)
    base_length = max(0.20 * chord, 0.20 * axis_extent, endpoint_gap)
    control_length = max(1.0, bow_factor * base_length)
    p1 = p0 + control_length * direction
    p2 = p3 + control_length * direction
    t = np.linspace(0.0, 1.0, 128)
    pts = (
        ((1.0 - t) ** 3)[:, None] * p0
        + (3.0 * ((1.0 - t) ** 2) * t)[:, None] * p1
        + (3.0 * (1.0 - t) * (t**2))[:, None] * p2
        + (t**3)[:, None] * p3
    )
    line = LineString(pts)
    return {
        "arc": line,
        "control_length_0_m": float(control_length),
        "control_length_1_m": float(control_length),
        "endpoint_tangent_error_deg": _endpoint_tangent_error_degrees(p0, p1, p2, p3, direction),
    }


def _bbox_touch_target_gap(bounds: tuple[float, float, float, float], side: str, bbox_touch_fraction: float) -> float:
    minx, miny, maxx, maxy = bounds
    axis_extent = max(maxx - minx, 1.0) if side in {"east", "west"} else max(maxy - miny, 1.0)
    return float(max(bbox_touch_fraction, 0.0) * axis_extent)


def _bbox_touch_distance_for_points(points: np.ndarray, bounds: tuple[float, float, float, float], side: str) -> float:
    minx, miny, maxx, maxy = bounds
    points = np.asarray(points, dtype=float)
    if side == "east":
        return float(maxx - np.nanmax(points[:, 0]))
    if side == "west":
        return float(np.nanmin(points[:, 0]) - minx)
    if side == "north":
        return float(maxy - np.nanmax(points[:, 1]))
    if side == "south":
        return float(np.nanmin(points[:, 1]) - miny)
    raise ValueError("offshore_side must be east, west, north, or south")


def _bbox_touch_info(arc: LineString, bounds: tuple[float, float, float, float], side: str) -> tuple[float, Point]:
    samples = _sample_line_xy(arc, 128)
    minx, miny, maxx, maxy = bounds
    if side == "east":
        idx = int(np.nanargmax(samples[:, 0]))
        return float(maxx - samples[idx, 0]), Point(float(samples[idx, 0]), float(samples[idx, 1]))
    if side == "west":
        idx = int(np.nanargmin(samples[:, 0]))
        return float(samples[idx, 0] - minx), Point(float(samples[idx, 0]), float(samples[idx, 1]))
    if side == "north":
        idx = int(np.nanargmax(samples[:, 1]))
        return float(maxy - samples[idx, 1]), Point(float(samples[idx, 0]), float(samples[idx, 1]))
    if side == "south":
        idx = int(np.nanargmin(samples[:, 1]))
        return float(samples[idx, 1] - miny), Point(float(samples[idx, 0]), float(samples[idx, 1]))
    raise ValueError("offshore_side must be east, west, north, or south")


def _endpoint_tangent_error_degrees(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, ocean_direction: np.ndarray) -> float:
    direction = np.asarray(ocean_direction, dtype=float)
    direction = direction / max(float(np.linalg.norm(direction)), 1.0e-12)
    start = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    end = np.asarray(p2, dtype=float) - np.asarray(p3, dtype=float)
    return float(max(_angle_between_degrees(start, direction), _angle_between_degrees(end, direction)))


def _angle_between_degrees(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 180.0
    cosang = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def _anchor_variant_score(
    count: int,
    valid_domain: bool,
    wet_fraction: float,
    median_depth: float,
    bbox_touch_error_m: float,
    bbox_touch_distance_m: float,
    endpoint_tangent_error_deg: float,
    curvature: float,
    bounds: tuple[float, float, float, float],
    side: str,
) -> float:
    minx, miny, maxx, maxy = bounds
    axis_extent = max(maxx - minx, 1.0) if side in {"east", "west"} else max(maxy - miny, 1.0)
    if count == 2:
        count_score = 220.0
    elif count == 1:
        count_score = 85.0
    elif count == 0:
        count_score = 55.0
    else:
        count_score = max(20.0, 120.0 - 25.0 * min(count - 2, 8))
    touch_penalty = 65.0 * min(bbox_touch_error_m / max(0.08 * axis_extent, 1.0), 4.0)
    outside_penalty = 120.0 * min(max(-bbox_touch_distance_m, 0.0) / max(0.03 * axis_extent, 1.0), 4.0)
    tangent_penalty = 4.0 * min(endpoint_tangent_error_deg, 45.0)
    return float(
        count_score
        + 25.0 * float(valid_domain)
        + 60.0 * wet_fraction
        + min(max(median_depth, 0.0) / 10.0, 12.0)
        - touch_penalty
        - outside_penalty
        - tangent_penalty
        - 15.0 * curvature
    )


def _line_intersections_on_arc(arc: LineString, linework, tolerance_m: float) -> list[Point]:
    intersection = arc.intersection(linework)
    points = list(_iter_points(intersection))
    if not points:
        return []
    points.sort(key=lambda point: arc.project(point))
    unique = []
    for point in points:
        if unique and point.distance(unique[-1]) <= tolerance_m:
            continue
        unique.append(point)
    return unique


def _iter_points(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Point):
        yield geom
    elif isinstance(geom, MultiPoint):
        for point in geom.geoms:
            yield point
    elif isinstance(geom, LineString):
        coords = list(geom.coords)
        if coords:
            yield Point(coords[0])
            yield Point(coords[-1])
    elif isinstance(geom, (MultiLineString, GeometryCollection)):
        for item in geom.geoms:
            yield from _iter_points(item)


def _trim_arc_between_intersections(arc: LineString, intersections: list[Point]) -> LineString:
    distances = sorted(float(arc.project(point)) for point in intersections[:2])
    d0, d1 = distances
    samples = np.linspace(d0, d1, 96)
    pts = [arc.interpolate(float(distance)) for distance in samples]
    return LineString([(point.x, point.y) for point in pts])


def _arc_bathy_stats(arc: LineString, bathy: BathymetryGrid, projection: LocalProjection, min_depth_m: float) -> tuple[float, float]:
    xy = _sample_line_xy(arc, 96)
    lonlat = unproject_points(xy, projection)
    depths = bathy.sample(lonlat[:, 0], lonlat[:, 1], fill_value=np.nan)
    finite = depths[np.isfinite(depths)]
    wet_fraction = float(np.mean(np.isfinite(depths) & (depths > min_depth_m)))
    median_depth = float(np.nanmedian(finite)) if finite.size else 0.0
    return wet_fraction, median_depth


def _closed_polygon_from_arc_and_side(arc: LineString, bounds: tuple[float, float, float, float], side: str) -> tuple[Polygon, bool]:
    try:
        polygon = _landward_cap_polygon(bounds, arc, side)
        return polygon, bool(polygon.is_valid and not polygon.is_empty and polygon.area > 0.0)
    except Exception:
        return Polygon(), False


def _side_from_direction(direction_xy: np.ndarray) -> str:
    dx, dy = float(direction_xy[0]), float(direction_xy[1])
    if abs(dx) >= abs(dy):
        return "east" if dx >= 0.0 else "west"
    return "north" if dy >= 0.0 else "south"


def _anchor_score(count: int, valid_domain: bool, wet_fraction: float, median_depth: float) -> float:
    count_score = 100.0 if count == 2 else max(0.0, 40.0 - 15.0 * abs(count - 2))
    return float(count_score + 25.0 * float(valid_domain) + 50.0 * wet_fraction + min(max(median_depth, 0.0) / 10.0, 10.0))


def _anchor_iteration_gdf(records: list[dict], projection: LocalProjection) -> gpd.GeoDataFrame:
    rows = []
    for record in records:
        row = dict(record)
        row["geometry"] = unproject_geometry(row["geometry"], projection)
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326") if rows else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def _generate_candidate_lines(
    bounds: tuple[float, float, float, float],
    side: str,
    families: tuple[str, ...],
    open_spacing_m: float,
    round_id: int,
) -> list[tuple[str, str, int, str, LineString]]:
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    side_length = height if side in {"east", "west"} else width
    n_points = max(8, int(np.ceil(side_length / max(open_spacing_m, 1.0))) + 1)
    span = max(width, height)
    round_scale = 1.0 + 0.35 * (round_id - 1)
    margin_fracs = (0.0, 0.04)
    inset_fracs = (0.04 * round_scale, 0.08 * round_scale, 0.14 * round_scale)
    records = []
    serial = 1
    for family in families:
        for margin_frac in margin_fracs:
            for inset_frac in inset_fracs:
                if family == "bbox-bow":
                    line = _sine_or_ellipse_line(bounds, side, n_points, inset_frac, 0.85 * inset_frac, margin_frac, "sine")
                elif family == "ellipse":
                    line = _sine_or_ellipse_line(bounds, side, n_points, inset_frac, 0.95 * inset_frac, margin_frac, "ellipse")
                elif family == "bezier":
                    line = _bezier_line(bounds, side, n_points, inset_frac, 0.95 * inset_frac, margin_frac)
                else:  # pragma: no cover - guarded by CLI choices
                    raise ValueError(f"Unsupported open-boundary family: {family}")
                cid = f"{side}_{family.replace('-', '_')}_r{round_id}_{serial:03d}"
                records.append((cid, family, round_id, side, line))
                serial += 1
    return records


def _ordered_side_options(inferred_side: str, mode: str) -> list[str]:
    if mode != "auto":
        return [inferred_side]
    sides = [inferred_side, "east", "south", "north", "west"]
    ordered = []
    for side in sides:
        if side not in ordered:
            ordered.append(side)
    return ordered


def _side_vectors(side: str) -> tuple[np.ndarray, np.ndarray]:
    if side == "east":
        return np.asarray([0.0, 1.0]), np.asarray([1.0, 0.0])
    if side == "west":
        return np.asarray([0.0, -1.0]), np.asarray([-1.0, 0.0])
    if side == "north":
        return np.asarray([-1.0, 0.0]), np.asarray([0.0, 1.0])
    if side == "south":
        return np.asarray([1.0, 0.0]), np.asarray([0.0, -1.0])
    raise ValueError("offshore_side must be east, west, north, or south")


def _side_endpoints(bounds: tuple[float, float, float, float], side: str, inset_frac: float, margin_frac: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    span = max(width, height)
    margin = margin_frac * (height if side in {"east", "west"} else width)
    inset = min(inset_frac * span, 0.35 * (width if side in {"east", "west"} else height))
    if side == "east":
        return np.asarray([maxx - inset, miny + margin]), np.asarray([maxx - inset, maxy - margin]), inset, span
    if side == "west":
        return np.asarray([minx + inset, maxy - margin]), np.asarray([minx + inset, miny + margin]), inset, span
    if side == "north":
        return np.asarray([maxx - margin, maxy - inset]), np.asarray([minx + margin, maxy - inset]), inset, span
    return np.asarray([minx + margin, miny + inset]), np.asarray([maxx - margin, miny + inset]), inset, span


def _sine_or_ellipse_line(
    bounds: tuple[float, float, float, float],
    side: str,
    n_points: int,
    inset_frac: float,
    bow_frac: float,
    margin_frac: float,
    profile: str,
) -> LineString:
    p0, p1, _inset, span = _side_endpoints(bounds, side, inset_frac, margin_frac)
    tangent, normal = _side_vectors(side)
    t = np.linspace(0.0, 1.0, n_points)
    along = np.linalg.norm(p1 - p0)
    if profile == "ellipse":
        shape = np.sqrt(np.maximum(0.0, 1.0 - (2.0 * t - 1.0) ** 2))
    else:
        shape = np.sin(np.pi * t)
    bow = bow_frac * span
    pts = p0 + np.outer(t, tangent) * along + np.outer(shape, normal) * bow
    return LineString(pts)


def _bezier_line(
    bounds: tuple[float, float, float, float],
    side: str,
    n_points: int,
    inset_frac: float,
    bow_frac: float,
    margin_frac: float,
) -> LineString:
    p0, p3, _inset, span = _side_endpoints(bounds, side, inset_frac, margin_frac)
    tangent, normal = _side_vectors(side)
    length = np.linalg.norm(p3 - p0)
    bow = bow_frac * span
    p1 = p0 + tangent * (0.32 * length) + normal * bow
    p2 = p3 - tangent * (0.32 * length) + normal * bow
    t = np.linspace(0.0, 1.0, n_points)
    pts = (
        ((1.0 - t) ** 3)[:, None] * p0
        + (3.0 * ((1.0 - t) ** 2) * t)[:, None] * p1
        + (3.0 * (1.0 - t) * (t**2))[:, None] * p2
        + (t**3)[:, None] * p3
    )
    return LineString(pts)


def _score_candidate(
    candidate_id: str,
    family: str,
    round_id: int,
    line_xy: LineString,
    wet_xy: Polygon,
    bathy: BathymetryGrid,
    projection: LocalProjection,
    coastline_xy,
    offshore_side: str,
    min_depth_m: float,
) -> dict:
    samples = _sample_line_xy(line_xy, 96)
    lonlat = unproject_points(samples, projection)
    depths = bathy.sample(lonlat[:, 0], lonlat[:, 1], fill_value=np.nan)
    wet_fraction = float(np.mean(np.isfinite(depths) & (depths > min_depth_m)))
    inside_fraction = float(np.mean([wet_xy.buffer(1.0).covers(Point(float(x), float(y))) for x, y in samples]))
    finite_depths = depths[np.isfinite(depths)]
    median_depth = float(np.nanmedian(finite_depths)) if finite_depths.size else 0.0
    minimum_depth = float(np.nanmin(finite_depths)) if finite_depths.size else 0.0
    coast_distance = float(line_xy.distance(coastline_xy)) if coastline_xy is not None and not coastline_xy.is_empty else float(line_xy.distance(wet_xy.boundary))
    endpoint_attachment = max(float(Point(line_xy.coords[0]).distance(wet_xy.boundary)), float(Point(line_xy.coords[-1]).distance(wet_xy.boundary)))
    curvature = _curvature_penalty(samples)
    depth_bonus = min(max(median_depth, 0.0) / 20.0, 10.0)
    coast_bonus = 20.0 * min(max(coast_distance, 0.0) / 25_000.0, 1.0)
    endpoint_penalty = min(endpoint_attachment / 5000.0, 12.0)
    score = 100.0 * wet_fraction + 25.0 * inside_fraction + depth_bonus + coast_bonus - 10.0 * curvature - endpoint_penalty
    status = "preferred" if wet_fraction >= 0.85 and inside_fraction >= 0.85 else "usable_with_review"
    return {
        "candidate_id": candidate_id,
        "family": family,
        "round": int(round_id),
        "line_xy": line_xy,
        "score": float(score),
        "status": status,
        "wet_fraction": wet_fraction,
        "inside_fraction": inside_fraction,
        "median_depth_m": median_depth,
        "min_depth_m": minimum_depth,
        "min_coast_distance_m": coast_distance,
        "endpoint_attachment_m": float(endpoint_attachment),
        "curvature_penalty": float(curvature),
        "offshore_side": offshore_side,
    }


def _sample_line_xy(line: LineString, n: int) -> np.ndarray:
    distances = np.linspace(0.0, line.length, max(2, int(n)))
    points = [line.interpolate(float(distance)) for distance in distances]
    return np.asarray([[point.x, point.y] for point in points], dtype=float)


def _curvature_penalty(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    a = points[1:-1] - points[:-2]
    b = points[2:] - points[1:-1]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    valid = (na > 0.0) & (nb > 0.0)
    if not np.any(valid):
        return 0.0
    cosang = np.sum(a[valid] * b[valid], axis=1) / (na[valid] * nb[valid])
    angles = np.arccos(np.clip(cosang, -1.0, 1.0))
    return float(np.nanmean(np.abs(angles)) / np.pi)


def _domain_cut_by_candidate(wet_xy: Polygon, bounds: tuple[float, float, float, float], line_xy: LineString, side: str) -> Polygon:
    cap = _landward_cap_polygon(bounds, line_xy, side)
    clipped = wet_xy.intersection(cap).buffer(0)
    if clipped.is_empty:
        clipped = wet_xy
    if isinstance(clipped, MultiPolygon):
        clipped = max(clipped.geoms, key=lambda geom: geom.area)
    if not isinstance(clipped, Polygon):
        raise ValueError("Candidate open-boundary cut did not produce a polygon.")
    return clipped.buffer(0)


def _landward_cap_polygon(bounds: tuple[float, float, float, float], line_xy: LineString, side: str) -> Polygon:
    minx, miny, maxx, maxy = bounds
    coords = list(line_xy.coords)
    if side == "east":
        ring = coords + [(minx, maxy), (minx, miny)]
    elif side == "west":
        ring = coords + [(maxx, miny), (maxx, maxy)]
    elif side == "north":
        ring = coords + [(minx, miny), (maxx, miny)]
    elif side == "south":
        ring = coords + [(maxx, maxy), (minx, maxy)]
    else:
        raise ValueError("offshore_side must be east, west, north, or south")
    return Polygon(ring).buffer(0)


def _candidate_gdf(candidates: list[dict], projection: LocalProjection, selected_id: str) -> gpd.GeoDataFrame:
    rows = []
    for item in candidates:
        row = _jsonable_candidate(item)
        row["selected"] = item["candidate_id"] == selected_id
        row["geometry"] = unproject_geometry(item["line_xy"], projection)
        rows.append(row)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _jsonable_candidate(item: dict) -> dict:
    return {
        key: value
        for key, value in item.items()
        if key != "line_xy" and isinstance(value, (str, int, float, bool, type(None)))
    }


def candidate_summary_json(candidates: gpd.GeoDataFrame) -> str:
    """Serialize candidate metrics without geometries for metadata sidecars."""
    rows = []
    for _, row in candidates.drop(columns="geometry").iterrows():
        rows.append({key: (float(value) if isinstance(value, np.floating) else value) for key, value in row.items()})
    return json.dumps(rows, indent=2)
