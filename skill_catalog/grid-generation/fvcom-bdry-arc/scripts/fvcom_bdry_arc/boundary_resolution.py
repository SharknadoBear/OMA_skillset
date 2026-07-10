"""Adaptive open-boundary and island resolution for FVCOM boundary packages.

This module is a clean-room, opt-in postprocessor.  It never changes the
legacy model-boundary-loop package.  Instead it writes a separate package
containing a resolved wet-domain polygon and explicit ordered constraint
nodes for downstream meshing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box, mapping
from shapely.ops import nearest_points, substring, unary_union
from shapely.prepared import prep

from .projection import local_utm_projection, project_geometry, unproject_geometry


@dataclass(frozen=True)
class BoundaryResolutionConfig:
    """Controls for the balanced adaptive coastal profile."""

    profile: str = "adaptive-coastal-v1"
    land_spacing_m: float = 150.0
    mission_spacing_m: float = 150.0
    open_anchor_spacing_m: float = 500.0
    open_central_spacing_m: float = 8000.0
    gradation: float = 0.15
    compact_spacing_m: float = 500.0
    irregular_spacing_m: float = 400.0
    elongated_spacing_m: float = 300.0
    complex_spacing_m: float = 300.0
    mission_buffer_m: float = 10_000.0
    min_vertices: int = 8
    area_budget_fraction: float = 0.005
    per_feature_area_tolerance: float = 0.02
    centroid_tolerance_fraction: float = 0.10
    hausdorff_tolerance_fraction: float = 0.50
    repair_sample_spacing_m: float = 250.0
    repair_land_clearance_m: float = 25.0


def analyze_boundary_resolution(
    model_boundary_loops_gpkg: str | Path,
    region_bpoly_json: str | Path | None = None,
    config: BoundaryResolutionConfig | None = None,
) -> dict[str, Any]:
    """Return non-mutating boundary and island resolution diagnostics."""
    config = config or BoundaryResolutionConfig()
    package = _load_loop_package(Path(model_boundary_loops_gpkg))
    projection = package["projection"]
    domain_xy: Polygon = package["domain_xy"]
    islands_xy: list[Polygon] = package["islands_xy"]
    mission_xy = _mission_geometry(region_bpoly_json, projection, config.mission_buffer_m)
    metrics = _island_metrics(islands_xy, domain_xy, mission_xy, config)
    return {
        "schema_version": "fvcom_boundary_resolution_analysis_v1",
        "profile": config.profile,
        "source": str(model_boundary_loops_gpkg),
        "island_count": len(metrics),
        "source_island_area_m2": float(sum(item["area_m2"] for item in metrics)),
        "source_island_perimeter_m": float(sum(item["perimeter_m"] for item in metrics)),
        "source_island_vertex_count": int(sum(item["source_vertex_count"] for item in metrics)),
        "class_counts": _count_by(metrics, "shape_class"),
        "protected_count": int(sum(bool(item["protected_mission"]) for item in metrics)),
        "subgrid_count": int(sum(item["shape_class"] == "subgrid_fragment" for item in metrics)),
        "islands": metrics,
    }


def build_boundary_resolution(
    model_boundary_loops_gpkg: str | Path,
    model_boundary_loop_manifest: str | Path | None,
    region_bpoly_json: str | Path | None,
    coastline_gpkg: str | Path | None,
    run_dir: str | Path,
    name: str,
    config: BoundaryResolutionConfig | None = None,
) -> dict[str, Any]:
    """Build an adaptive boundary-resolution package without touching legacy outputs."""
    config = config or BoundaryResolutionConfig()
    if config.profile != "adaptive-coastal-v1":
        raise ValueError("Boundary resolution builder requires profile adaptive-coastal-v1")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(model_boundary_loops_gpkg)
    package = _load_loop_package(source_path)
    projection = package["projection"]
    source_domain: Polygon = package["domain_xy"]
    islands_xy: list[Polygon] = package["islands_xy"]
    mission_xy = _mission_geometry(region_bpoly_json, projection, config.mission_buffer_m)
    land_union = _load_land_union(coastline_gpkg, projection)

    open_xy, landward_xy = _canonical_open_and_landward(package["segments_xy"], source_domain)
    repaired_open, repair_report = _repair_open_arc(open_xy, source_domain, land_union, config)
    outer_shell = _compose_shell(repaired_open, landward_xy, source_domain)
    shell_polygon = Polygon(outer_shell)
    if not shell_polygon.is_valid:
        shell_polygon = shell_polygon.buffer(0)
    if not isinstance(shell_polygon, Polygon) or shell_polygon.is_empty:
        raise ValueError("Adaptive open-arc repair did not produce a valid exterior polygon")

    source_metrics = _island_metrics(islands_xy, source_domain, mission_xy, config)
    topologized, action_report = _apply_subgrid_actions(
        shell_polygon,
        islands_xy,
        source_metrics,
        mission_xy,
        config,
    )
    resolved_islands, resolved_records = _generalize_islands(topologized, mission_xy, config)

    open_nodes, open_h = _sample_open_arc(repaired_open, config)
    land_nodes = _sample_line(landward_xy, config.land_spacing_m, include_end=True)
    outer_nodes = open_nodes + land_nodes[1:-1]
    outer_kinds = ["open"] * len(open_nodes) + ["land"] * max(0, len(land_nodes) - 2)
    outer_h = open_h + [float(config.land_spacing_m)] * max(0, len(land_nodes) - 2)
    outer_nodes = _deduplicate_ring(outer_nodes)
    outer_kinds = outer_kinds[: len(outer_nodes)]
    outer_h = outer_h[: len(outer_nodes)]

    island_chains: list[list[tuple[float, float]]] = []
    island_targets: list[float] = []
    for record, polygon in zip(resolved_records, resolved_islands):
        target = float(record["target_spacing_m"])
        if record["protected_mission"]:
            chain = list(polygon.exterior.coords)[:-1]
            candidate = polygon
        else:
            chain = _sample_closed_ring(polygon, target, config.min_vertices)
            candidate = Polygon(chain)
            attempts = 0
            while (
                (not candidate.is_valid or candidate.is_empty or abs(candidate.area / max(polygon.area, 1.0) - 1.0) > config.per_feature_area_tolerance)
                and attempts < 5
            ):
                target *= 0.5
                chain = _sample_closed_ring(polygon, target, config.min_vertices)
                candidate = Polygon(chain)
                attempts += 1
            if not candidate.is_valid or candidate.is_empty:
                chain = list(polygon.exterior.coords)[:-1]
                candidate = polygon
        record["final_target_spacing_m"] = float(target)
        record["resolved_vertex_count"] = int(len(chain))
        record["resolved_area_m2"] = float(candidate.area)
        island_chains.append(chain)
        island_targets.append(float(target))

    resolved_domain = Polygon(outer_nodes, holes=island_chains)
    if not resolved_domain.is_valid:
        resolved_domain = resolved_domain.buffer(0)
    resolved_domain = _select_polygon(resolved_domain, source_domain.representative_point())
    if not isinstance(resolved_domain, Polygon) or resolved_domain.is_empty:
        raise ValueError("Resolved boundary nodes do not form a valid wet-domain polygon")

    sampled_open = LineString(open_nodes)
    endpoint_mask = Point(open_nodes[0]).buffer(max(2.0 * config.repair_sample_spacing_m, 500.0)).union(
        Point(open_nodes[-1]).buffer(max(2.0 * config.repair_sample_spacing_m, 500.0))
    )
    sampled_land_length = 0.0
    if land_union is not None and not land_union.is_empty:
        sampled_land_length = float(sampled_open.difference(endpoint_mask).intersection(land_union).length)
    exterior = LineString(resolved_domain.exterior.coords)
    exterior_tolerance = max(0.01, 1.0e-7 * max(float(sampled_open.length), 1.0))
    exterior_off_length = float(sampled_open.difference(exterior.buffer(exterior_tolerance)).length)
    exterior_overlap = float(max(0.0, 1.0 - exterior_off_length / max(float(sampled_open.length), 1.0)))

    node_records: list[dict[str, Any]] = []
    chain_summaries: list[dict[str, Any]] = []
    _append_node_chain(node_records, chain_summaries, 0, outer_nodes, outer_kinds, outer_h, projection)
    for chain_id, (chain, target) in enumerate(zip(island_chains, island_targets), start=1):
        _append_node_chain(
            node_records,
            chain_summaries,
            chain_id,
            chain,
            ["island"] * len(chain),
            [target] * len(chain),
            projection,
        )

    gpkg = run_dir / "boundary_resolution.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    _write_resolution_layers(
        gpkg,
        resolved_domain,
        repaired_open,
        resolved_islands,
        islands_xy,
        node_records,
        source_metrics,
        resolved_records,
        projection,
    )
    diagnostics_path = run_dir / "boundary_resolution_diagnostics.json"
    node_geojson_path = run_dir / "boundary_resolution_nodes.geojson"
    review_map = run_dir / "boundary_resolution_review_map.png"
    diagnostics = {
        "schema_version": "fvcom_boundary_resolution_diagnostics_v1",
        "source_analysis": {
            "island_count": len(source_metrics),
            "class_counts": _count_by(source_metrics, "shape_class"),
            "protected_count": int(sum(bool(item["protected_mission"]) for item in source_metrics)),
        },
        "open_arc_repair": repair_report,
        "topology_actions": action_report,
        "resolved_islands": resolved_records,
        "chains": chain_summaries,
    }
    diagnostics_path.write_text(json.dumps(_json_safe(diagnostics), indent=2), encoding="utf-8")
    node_geojson_path.write_text(json.dumps(_node_geojson(node_records), indent=2), encoding="utf-8")
    _plot_review(review_map, source_domain, resolved_domain, repaired_open, mission_xy, projection, source_metrics)

    open_count = int(sum(item["boundary_kind"] == "open" for item in node_records))
    island_count = int(sum(item["boundary_kind"] == "island" for item in node_records))
    topology_area_fraction = float(action_report["cumulative_absolute_area_change_m2"] / max(action_report["source_island_area_m2"], 1.0))
    failures: list[str] = []
    if not repair_report["land_free"]:
        failures.append("adaptive_open_arc_intersects_land")
    if sampled_land_length > 1.0e-6:
        failures.append("sampled_open_boundary_intersects_land")
    if not repair_report["anchors_preserved"]:
        failures.append("adaptive_open_arc_anchor_shift")
    if exterior_overlap < 1.0 - 1.0e-9:
        failures.append("sampled_open_boundary_not_on_exterior")
    if topology_area_fraction > config.area_budget_fraction + 1.0e-12:
        failures.append("island_topology_area_budget_exceeded")
    if not resolved_domain.is_valid:
        failures.append("resolved_domain_invalid")
    manifest = {
        "schema_version": "fvcom_boundary_resolution_manifest_v1",
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_by": "fvcom-bdry-arc boundary_resolution.py",
        "profile": config.profile,
        "final_status": "pass" if not failures else "needs_review",
        "failure_taxonomy": failures,
        "inputs": {
            "model_boundary_loops_gpkg": str(source_path),
            "model_boundary_loop_manifest": str(model_boundary_loop_manifest) if model_boundary_loop_manifest else None,
            "region_bpoly_json": str(region_bpoly_json) if region_bpoly_json else None,
            "coastline_gpkg": str(coastline_gpkg) if coastline_gpkg else None,
        },
        "settings": _json_safe(config.__dict__),
        "qa": {
            "open_boundary_node_count": open_count,
            "island_boundary_node_count": island_count,
            "total_boundary_node_count": int(len(node_records)),
            "resolved_island_count": int(len(resolved_islands)),
            "source_island_count": int(len(islands_xy)),
            "topology_absolute_area_change_fraction": topology_area_fraction,
            "protected_mission_operation_count": int(action_report["protected_operation_count"]),
            "open_arc_land_intersection_m": float(max(repair_report["land_intersection_length_m"], sampled_land_length)),
            "open_arc_exterior_overlap_fraction": exterior_overlap,
            "resolved_domain_valid": bool(resolved_domain.is_valid),
        },
        "chains": chain_summaries,
        "outputs": {
            "boundary_resolution_gpkg": str(gpkg),
            "boundary_resolution_diagnostics_json": str(diagnostics_path),
            "boundary_resolution_nodes_geojson": str(node_geojson_path),
            "boundary_resolution_review_map": str(review_map),
            "boundary_resolution_manifest": str(run_dir / "boundary_resolution_manifest.json"),
        },
    }
    manifest_path = run_dir / "boundary_resolution_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def _load_loop_package(path: Path) -> dict[str, Any]:
    layers = set(gpd.list_layers(path)["name"])
    domain_gdf = gpd.read_file(path, layer="model_domain_polygon")
    domain_lonlat = next(geom for geom in domain_gdf.geometry if isinstance(geom, Polygon) and not geom.is_empty)
    if domain_gdf.crs is not None:
        domain_lonlat = gpd.GeoSeries([domain_lonlat], crs=domain_gdf.crs).to_crs("EPSG:4326").iloc[0]
    projection = local_utm_projection(tuple(float(v) for v in domain_lonlat.bounds))
    domain_xy = project_geometry(domain_lonlat, projection).buffer(0)
    segments = gpd.read_file(path, layer="model_outer_boundary_segments").to_crs("EPSG:4326")
    segment_records = []
    for _, row in segments.sort_values("sequence_id").iterrows():
        segment_records.append(
            {
                "sequence_id": int(row.sequence_id),
                "segment_class": str(row.segment_class),
                "geometry": project_geometry(row.geometry, projection),
            }
        )
    islands_xy: list[Polygon] = []
    if "island_boundary_polygons" in layers:
        islands = gpd.read_file(path, layer="island_boundary_polygons").to_crs("EPSG:4326")
        islands_xy = [project_geometry(geom, projection).buffer(0) for geom in islands.geometry if isinstance(geom, Polygon) and not geom.is_empty]
    return {
        "projection": projection,
        "domain_xy": _select_polygon(domain_xy, domain_xy.representative_point()),
        "segments_xy": segment_records,
        "islands_xy": islands_xy,
    }


def _mission_geometry(region_bpoly_json: str | Path | None, projection, buffer_m: float):
    if not region_bpoly_json or not Path(region_bpoly_json).exists():
        return GeometryCollection()
    doc = json.loads(Path(region_bpoly_json).read_text(encoding="utf-8-sig"))
    ingredients = doc.get("target_region_features", {}).get("features", [])
    if not ingredients:
        ingredients = doc.get("qa", {}).get("ingredient_coverage", {}).get("ingredients", [])
    if not ingredients:
        retained = doc.get("qa", {}).get("target_region_features", {}).get("retained_geojson_path")
        if retained and Path(retained).exists():
            feature_doc = json.loads(Path(retained).read_text(encoding="utf-8-sig"))
            ingredients = [
                {**feature.get("properties", {}), "geometry": feature.get("geometry")}
                for feature in feature_doc.get("features", [])
            ]
    polygons = []
    for item in ingredients:
        role = str(item.get("role", ""))
        if role not in {"target_water_body", "river_input_context"}:
            continue
        geometry = item.get("geometry")
        if isinstance(geometry, dict) and geometry.get("type") == "Polygon":
            polygons.append(project_geometry(Polygon(geometry["coordinates"][0]), projection))
        elif isinstance(geometry, (list, tuple)) and len(geometry) == 4:
            poly = box(*map(float, geometry))
            polygons.append(project_geometry(poly, projection))
    return unary_union(polygons).buffer(float(buffer_m)) if polygons else GeometryCollection()


def _load_land_union(coastline_gpkg: str | Path | None, projection):
    if not coastline_gpkg or not Path(coastline_gpkg).exists():
        return GeometryCollection()
    path = Path(coastline_gpkg)
    layers = set(gpd.list_layers(path)["name"])
    layer = "land_polygons" if "land_polygons" in layers else next(iter(layers), None)
    if layer is None:
        return GeometryCollection()
    gdf = gpd.read_file(path, layer=layer).to_crs("EPSG:4326")
    return unary_union([project_geometry(geom, projection) for geom in gdf.geometry if geom is not None and not geom.is_empty])


def _canonical_open_and_landward(records: list[dict[str, Any]], domain: Polygon) -> tuple[LineString, LineString]:
    records = sorted(records, key=lambda item: item["sequence_id"])
    flags = [item["segment_class"] == "open_boundary" for item in records]
    if not any(flags):
        raise ValueError("No open_boundary segments are available for adaptive coastal resolution")
    if all(flags):
        line = LineString(domain.exterior.coords)
        return line, LineString([line.coords[-1], line.coords[0]])
    pivot = next(idx for idx, flag in enumerate(flags) if not flag)
    order = list(range(pivot + 1, len(records))) + list(range(0, pivot + 1))
    groups: list[list[int]] = []
    current: list[int] = []
    for idx in order:
        if flags[idx]:
            current.append(idx)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    group = max(groups, key=len)
    open_coords = _ordered_segment_coords([records[idx]["geometry"] for idx in group])
    used = set(group)
    complement_order = []
    end = group[-1]
    idx = (end + 1) % len(records)
    while idx not in used:
        complement_order.append(idx)
        idx = (idx + 1) % len(records)
    land_coords = _ordered_segment_coords([records[idx]["geometry"] for idx in complement_order])
    if np.linalg.norm(np.asarray(land_coords[0]) - np.asarray(open_coords[-1])) > np.linalg.norm(np.asarray(land_coords[-1]) - np.asarray(open_coords[-1])):
        land_coords.reverse()
    return LineString(open_coords), LineString(land_coords)


def _ordered_segment_coords(lines: Iterable[LineString]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for line in lines:
        coords = [(float(x), float(y)) for x, y in line.coords]
        if not out:
            out.extend(coords)
            continue
        if np.linalg.norm(np.asarray(out[-1]) - np.asarray(coords[-1])) < np.linalg.norm(np.asarray(out[-1]) - np.asarray(coords[0])):
            coords.reverse()
        if np.linalg.norm(np.asarray(out[-1]) - np.asarray(coords[0])) < 1.0e-5:
            out.extend(coords[1:])
        else:
            out.extend(coords)
    return out


def _repair_open_arc(open_line: LineString, domain: Polygon, land_union, config: BoundaryResolutionConfig) -> tuple[LineString, dict[str, Any]]:
    spacing = float(config.repair_sample_spacing_m)
    coords = _sample_line(open_line, spacing, include_end=True)
    original_start = Point(coords[0])
    original_end = Point(coords[-1])
    if land_union is None or land_union.is_empty:
        return LineString(coords), {
            "method": "sampled_no_land_polygons",
            "anchors_preserved": True,
            "land_free": True,
            "land_intersection_length_m": 0.0,
            "moved_point_count": 0,
        }
    corridor = open_line.buffer(max(5000.0, 20.0 * spacing))
    local_land = land_union.intersection(corridor)
    forbidden = local_land.buffer(max(1.0, float(config.repair_land_clearance_m)))
    prepared_forbidden = prep(forbidden)
    prepared_domain = prep(domain.buffer(1.0))
    bad = [idx for idx, xy in enumerate(coords) if idx not in {0, len(coords) - 1} and (prepared_forbidden.contains(Point(xy)) or not prepared_domain.covers(Point(xy)))]
    moved = set()
    array = np.asarray(coords, dtype=float)
    for idx in bad:
        tangent = array[min(idx + 1, len(array) - 1)] - array[max(idx - 1, 0)]
        norm = float(np.linalg.norm(tangent))
        if norm <= 1.0e-12:
            continue
        tangent /= norm
        normals = (np.asarray([-tangent[1], tangent[0]]), np.asarray([tangent[1], -tangent[0]]))
        selected = None
        for normal in normals:
            for distance in (50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 5000.0):
                candidate = array[idx] + distance * normal
                point = Point(float(candidate[0]), float(candidate[1]))
                if prepared_domain.covers(point) and not prepared_forbidden.contains(point):
                    selected = candidate
                    break
            if selected is not None:
                break
        if selected is not None:
            coords[idx] = (float(selected[0]), float(selected[1]))
            moved.add(idx)
    # Smooth a small neighborhood of every repaired sample while keeping anchors fixed.
    active = set()
    for idx in moved:
        active.update(range(max(1, idx - 4), min(len(coords) - 1, idx + 5)))
    arr = np.asarray(coords, dtype=float)
    for _ in range(12):
        trial = arr.copy()
        for idx in sorted(active):
            candidate = 0.25 * arr[idx - 1] + 0.50 * arr[idx] + 0.25 * arr[idx + 1]
            point = Point(float(candidate[0]), float(candidate[1]))
            if prepared_domain.covers(point) and not prepared_forbidden.contains(point):
                trial[idx] = candidate
        arr = trial
    arr[0] = [original_start.x, original_start.y]
    arr[-1] = [original_end.x, original_end.y]
    repaired = LineString(arr)
    endpoint_mask = original_start.buffer(max(2.0 * spacing, 500.0)).union(original_end.buffer(max(2.0 * spacing, 500.0)))
    inspected = repaired.difference(endpoint_mask)
    intersection = inspected.intersection(local_land)
    land_length = float(getattr(intersection, "length", 0.0))
    return repaired, {
        "method": "deterministic_interior_clearance_line_search",
        "sample_spacing_m": spacing,
        "anchors_preserved": bool(Point(repaired.coords[0]).distance(original_start) < 1.0e-8 and Point(repaired.coords[-1]).distance(original_end) < 1.0e-8),
        "land_free": bool(land_length <= 1.0e-6),
        "land_intersection_length_m": land_length,
        "moved_point_count": int(len(moved)),
        "source_length_m": float(open_line.length),
        "repaired_length_m": float(repaired.length),
    }


def _compose_shell(open_line: LineString, landward: LineString, source_domain: Polygon) -> list[tuple[float, float]]:
    open_coords = list(open_line.coords)
    land_coords = list(landward.coords)
    if np.linalg.norm(np.asarray(land_coords[0]) - np.asarray(open_coords[-1])) > np.linalg.norm(np.asarray(land_coords[-1]) - np.asarray(open_coords[-1])):
        land_coords.reverse()
    coords = open_coords + land_coords[1:]
    if np.linalg.norm(np.asarray(coords[0]) - np.asarray(coords[-1])) > 1.0e-7:
        coords.append(coords[0])
    polygon = Polygon(coords)
    if polygon.is_valid and polygon.contains(source_domain.representative_point()):
        return [(float(x), float(y)) for x, y in coords]
    reversed_coords = list(reversed(open_coords)) + list(reversed(land_coords))[1:]
    if np.linalg.norm(np.asarray(reversed_coords[0]) - np.asarray(reversed_coords[-1])) > 1.0e-7:
        reversed_coords.append(reversed_coords[0])
    return [(float(x), float(y)) for x, y in reversed_coords]


def _island_metrics(islands: list[Polygon], domain: Polygon, mission, config: BoundaryResolutionConfig) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    outer = LineString(domain.exterior.coords)
    for idx, polygon in enumerate(islands):
        polygon = polygon.buffer(0)
        area = float(polygon.area)
        perimeter = float(polygon.length)
        diameter = float(2.0 * math.sqrt(area / math.pi)) if area > 0.0 else 0.0
        compactness = float(4.0 * math.pi * area / max(perimeter * perimeter, 1.0e-12))
        solidity = float(area / max(polygon.convex_hull.area, 1.0e-12))
        rectangle = polygon.minimum_rotated_rectangle
        rect_coords = np.asarray(rectangle.exterior.coords, dtype=float)[:4]
        edges = np.linalg.norm(np.roll(rect_coords, -1, axis=0) - rect_coords, axis=1)
        width = float(np.min(edges)) if len(edges) else 0.0
        length = float(np.max(edges)) if len(edges) else 0.0
        aspect = float(length / max(width, 1.0e-12))
        gap = float(polygon.distance(outer))
        gap_line = _nearest_connector(polygon, outer)
        for other_idx, other in enumerate(islands):
            if other_idx != idx:
                distance = float(polygon.distance(other))
                if distance < gap:
                    gap = distance
                    gap_line = _nearest_connector(polygon, other)
        protected_island = bool(mission is not None and not mission.is_empty and polygon.intersects(mission))
        protected_gap = bool(mission is not None and not mission.is_empty and gap_line.intersects(mission))
        protected = bool(protected_island or protected_gap)
        base_h = float(config.mission_spacing_m if protected else config.compact_spacing_m)
        if diameter < 2.0 * base_h or width < 1.5 * base_h:
            shape_class = "subgrid_fragment"
        elif aspect >= 3.0 and solidity >= 0.70:
            shape_class = "elongated_barrier"
        elif solidity < 0.72 or compactness < 0.12:
            shape_class = "complex_concave"
        elif compactness >= 0.42 and aspect < 2.5 and solidity >= 0.85:
            shape_class = "compact"
        else:
            shape_class = "irregular"
        simplified = polygon.simplify(0.25 * base_h, preserve_topology=True)
        results.append(
            {
                "island_id": int(idx),
                "area_m2": area,
                "perimeter_m": perimeter,
                "equivalent_diameter_m": diameter,
                "compactness": compactness,
                "normalized_complexity": float(1.0 / math.sqrt(max(compactness, 1.0e-12))),
                "minimum_rectangle_width_m": width,
                "minimum_rectangle_length_m": length,
                "aspect_ratio": aspect,
                "solidity": solidity,
                "nearest_gap_m": gap,
                "protected_mission": protected,
                "protected_island": protected_island,
                "protected_gap": protected_gap,
                "shape_class": shape_class,
                "source_vertex_count": int(len(polygon.exterior.coords) - 1),
                "simplified_area_ratio": float(simplified.area / max(area, 1.0)),
                "simplified_perimeter_ratio": float(simplified.length / max(perimeter, 1.0)),
                "action": "retain",
                "reason": "resolved_or_protected",
            }
        )
    return results


def _apply_subgrid_actions(shell: Polygon, islands: list[Polygon], metrics: list[dict[str, Any]], mission, config: BoundaryResolutionConfig) -> tuple[Polygon, dict[str, Any]]:
    source_area = float(sum(poly.area for poly in islands))
    budget = float(config.area_budget_fraction * source_area)
    dropped: set[int] = set()
    bridges = []
    cumulative = 0.0
    protected_operations = 0
    actions: list[dict[str, Any]] = []
    outer = LineString(shell.exterior.coords)
    current_water = Polygon(shell.exterior.coords, holes=[list(poly.exterior.coords) for poly in islands])
    merge_targets: set[int] = set()
    candidates = sorted((item for item in metrics if item["shape_class"] == "subgrid_fragment"), key=lambda item: (item["area_m2"], item["island_id"]))
    for item in candidates:
        idx = int(item["island_id"])
        polygon = islands[idx]
        if item["protected_mission"]:
            item["action"] = "retain_protected"
            item["reason"] = "mission_region_or_gap_protection"
            protected_operations += 0
            continue
        if idx in merge_targets:
            item["action"] = "retain_merge_dependency"
            item["reason"] = "larger_landmass_receives_prior_subgrid_bridge"
            continue
        target_h = float(config.compact_spacing_m)
        nearest_geom = outer
        nearest_id: int | None = None
        gap = float(polygon.distance(outer))
        for other_idx, other in enumerate(islands):
            if other_idx == idx or other_idx in dropped or other.area <= polygon.area:
                continue
            distance = float(polygon.distance(other))
            if distance < gap:
                gap = distance
                nearest_geom = other
                nearest_id = other_idx
        if gap < target_h:
            a, b = nearest_points(polygon, nearest_geom)
            width = max(2.0, min(0.15 * target_h, 0.25 * gap + 1.0))
            start = np.asarray([a.x, a.y], dtype=float)
            end = np.asarray([b.x, b.y], dtype=float)
            vector = end - start
            norm = float(np.linalg.norm(vector))
            if norm > 1.0e-9:
                vector /= norm
                start -= 2.0 * width * vector
                end += 2.0 * width * vector
            bridge = LineString([start, end]).buffer(width, cap_style=2)
            previous_bridges = unary_union(bridges) if bridges else GeometryCollection()
            delta = float(bridge.difference(previous_bridges).intersection(shell).area)
            action = "merge_to_mainland" if nearest_id is None else "merge_to_island"
        else:
            bridge = None
            delta = float(polygon.area)
            action = "drop_subgrid"
        if cumulative + delta > budget + 1.0e-9:
            item["action"] = "retain_budget_limited"
            item["reason"] = "aggregate_area_budget_exhausted"
            continue
        if bridge is not None:
            unintended = [
                other_idx
                for other_idx, other in enumerate(islands)
                if other_idx not in {idx, nearest_id} and other_idx not in dropped and bridge.intersects(other)
            ]
            if unintended or (nearest_id is not None and bridge.intersects(outer)):
                item["action"] = "retain_topology_guard"
                item["reason"] = "bridge_creates_unintended_land_contact"
                continue
        trial_dropped = set(dropped)
        trial_bridges = list(bridges)
        if bridge is None:
            trial_dropped.add(idx)
        else:
            trial_bridges.append(bridge)
        trial_holes = [list(poly.exterior.coords) for other_idx, poly in enumerate(islands) if other_idx not in trial_dropped]
        trial_water = Polygon(shell.exterior.coords, holes=trial_holes)
        if trial_bridges:
            trial_water = trial_water.difference(unary_union(trial_bridges))
        if not isinstance(trial_water, Polygon) or trial_water.is_empty or not trial_water.is_valid:
            item["action"] = "retain_topology_guard"
            item["reason"] = "operation_invalid_or_disconnects_wet_domain"
            continue
        changed = current_water.symmetric_difference(trial_water)
        if mission is not None and not mission.is_empty and changed.intersects(mission):
            item["action"] = "retain_protected"
            item["reason"] = "operation_changes_protected_mission_water"
            continue
        cumulative += delta
        item["action"] = action
        item["reason"] = "unprotected_subgrid_resolution_rule"
        if bridge is None:
            dropped.add(idx)
        else:
            bridges.append(bridge)
            if nearest_id is not None:
                merge_targets.add(int(nearest_id))
        current_water = trial_water
        actions.append({"island_id": idx, "action": action, "area_change_m2": delta, "nearest_gap_m": gap, "merge_target_island_id": nearest_id})

    return current_water, {
        "policy": "balanced_protected_auto_merge_drop",
        "source_island_area_m2": source_area,
        "area_budget_m2": budget,
        "cumulative_absolute_area_change_m2": cumulative,
        "cumulative_absolute_area_change_fraction": float(cumulative / max(source_area, 1.0)),
        "dropped_count": int(len(dropped)),
        "bridge_count": int(len(bridges)),
        "protected_operation_count": int(protected_operations),
        "actions": actions,
    }


def _generalize_islands(domain: Polygon, mission, config: BoundaryResolutionConfig) -> tuple[list[Polygon], list[dict[str, Any]]]:
    islands = [Polygon(ring).buffer(0) for ring in domain.interiors]
    resolved: list[Polygon] = []
    records: list[dict[str, Any]] = []
    for idx, polygon in enumerate(islands):
        area = float(polygon.area)
        perimeter = float(polygon.length)
        compactness = float(4.0 * math.pi * area / max(perimeter * perimeter, 1.0e-12))
        solidity = float(area / max(polygon.convex_hull.area, 1.0e-12))
        rect = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords, dtype=float)[:4]
        edge = np.linalg.norm(np.roll(rect, -1, axis=0) - rect, axis=1)
        width = float(np.min(edge))
        aspect = float(np.max(edge) / max(width, 1.0e-12))
        diameter = float(2.0 * math.sqrt(area / math.pi))
        outer = LineString(domain.exterior.coords)
        gap = float(polygon.distance(outer))
        gap_line = _nearest_connector(polygon, outer)
        for other_idx, other in enumerate(islands):
            if other_idx == idx:
                continue
            distance = float(polygon.distance(other))
            if distance < gap:
                gap = distance
                gap_line = _nearest_connector(polygon, other)
        protected_island = bool(mission is not None and not mission.is_empty and polygon.intersects(mission))
        protected_gap = bool(mission is not None and not mission.is_empty and gap_line.intersects(mission))
        protected = bool(protected_island or protected_gap)
        source_orientation = _principal_orientation_deg(polygon)
        if protected:
            shape_class = "protected_mission"
            target = min(float(config.mission_spacing_m), max(1.0, 0.25 * gap)) if gap > 0.0 else float(config.mission_spacing_m)
        elif aspect >= 3.0 and solidity >= 0.70:
            shape_class = "elongated_barrier"
            target = config.elongated_spacing_m
        elif solidity < 0.72 or compactness < 0.12:
            shape_class = "complex_concave"
            target = config.complex_spacing_m
        elif compactness >= 0.42 and aspect < 2.5 and solidity >= 0.85:
            shape_class = "compact"
            target = config.compact_spacing_m
        else:
            shape_class = "irregular"
            target = config.irregular_spacing_m
        tolerance = 0.0 if protected else 0.25 * float(target)
        accepted = polygon
        if not protected:
            for _ in range(8):
                candidate = polygon.simplify(tolerance, preserve_topology=True).buffer(0)
                if isinstance(candidate, Polygon) and not candidate.is_empty:
                    area_error = abs(candidate.area / max(polygon.area, 1.0) - 1.0)
                    centroid_shift = float(candidate.centroid.distance(polygon.centroid))
                    hausdorff = float(candidate.hausdorff_distance(polygon))
                    orientation_error = _principal_orientation_difference_deg(source_orientation, _principal_orientation_deg(candidate))
                    orientation_stable = bool(aspect < 1.25 or orientation_error <= 5.0)
                    if area_error <= config.per_feature_area_tolerance and centroid_shift <= config.centroid_tolerance_fraction * target and hausdorff <= config.hausdorff_tolerance_fraction * target and orientation_stable:
                        accepted = candidate
                        break
                tolerance *= 0.5
        resolved_orientation = _principal_orientation_deg(accepted)
        resolved.append(accepted)
        records.append(
            {
                "resolved_island_id": int(idx),
                "shape_class": shape_class,
                "protected_mission": protected,
                "protected_island": protected_island,
                "protected_gap": protected_gap,
                "nearest_gap_m": gap,
                "source_area_m2": area,
                "generalized_area_m2": float(accepted.area),
                "generalized_area_error_fraction": float(abs(accepted.area / max(area, 1.0) - 1.0)),
                "equivalent_diameter_m": diameter,
                "minimum_rectangle_width_m": width,
                "compactness": compactness,
                "solidity": solidity,
                "aspect_ratio": aspect,
                "source_principal_orientation_deg": source_orientation,
                "resolved_principal_orientation_deg": resolved_orientation,
                "principal_orientation_change_deg": _principal_orientation_difference_deg(source_orientation, resolved_orientation),
                "target_spacing_m": float(target),
                "accepted_simplification_tolerance_m": float(tolerance),
            }
        )
    return resolved, records


def _sample_open_arc(line: LineString, config: BoundaryResolutionConfig) -> tuple[list[tuple[float, float]], list[float]]:
    length = float(line.length)
    positions = [0.0]
    while positions[-1] < length:
        s = positions[-1]
        h = min(config.open_central_spacing_m, config.open_anchor_spacing_m + config.gradation * min(s, max(0.0, length - s)))
        positions.append(min(length, s + max(1.0, float(h))))
        if positions[-1] >= length:
            break
    for _ in range(12):
        added: list[float] = []
        for start, end in zip(positions[:-1], positions[1:]):
            chord = LineString([line.interpolate(start), line.interpolate(end)])
            section = substring(line, start, end)
            local_h = min(
                config.open_central_spacing_m,
                config.open_anchor_spacing_m + config.gradation * min(start, max(0.0, length - end)),
            )
            if float(section.hausdorff_distance(chord)) > 0.10 * max(float(local_h), 1.0):
                added.append(0.5 * (start + end))
        if not added:
            break
        positions = sorted(set(positions + added))
    coords = []
    sizes = []
    for s in positions:
        point = line.interpolate(float(s))
        coords.append((float(point.x), float(point.y)))
        sizes.append(float(min(config.open_central_spacing_m, config.open_anchor_spacing_m + config.gradation * min(s, max(0.0, length - s)))))
    return coords, sizes


def _principal_orientation_deg(polygon: Polygon) -> float:
    rectangle = np.asarray(polygon.minimum_rotated_rectangle.exterior.coords, dtype=float)[:4]
    vectors = np.roll(rectangle, -1, axis=0) - rectangle
    lengths = np.linalg.norm(vectors, axis=1)
    vector = vectors[int(np.argmax(lengths))]
    return float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)


def _nearest_connector(first, second) -> LineString:
    start, end = nearest_points(first, second)
    return LineString([(float(start.x), float(start.y)), (float(end.x), float(end.y))])


def _principal_orientation_difference_deg(first: float, second: float) -> float:
    delta = abs(float(first) - float(second)) % 180.0
    return float(min(delta, 180.0 - delta))


def _sample_line(line: LineString, spacing: float, include_end: bool) -> list[tuple[float, float]]:
    length = float(line.length)
    n = max(1, int(math.ceil(length / max(float(spacing), 1.0))))
    positions = np.linspace(0.0, length, n + 1)
    if not include_end:
        positions = positions[:-1]
    return [(float(line.interpolate(float(s)).x), float(line.interpolate(float(s)).y)) for s in positions]


def _sample_closed_ring(polygon: Polygon, spacing: float, minimum: int) -> list[tuple[float, float]]:
    line = LineString(polygon.exterior.coords)
    n = max(int(minimum), int(math.ceil(line.length / max(float(spacing), 1.0))))
    return [(float(line.interpolate(i * line.length / n).x), float(line.interpolate(i * line.length / n).y)) for i in range(n)]


def _deduplicate_ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for xy in coords:
        if not out or np.linalg.norm(np.asarray(out[-1]) - np.asarray(xy)) > 1.0e-7:
            out.append((float(xy[0]), float(xy[1])))
    if len(out) > 1 and np.linalg.norm(np.asarray(out[0]) - np.asarray(out[-1])) <= 1.0e-7:
        out.pop()
    return out


def _append_node_chain(records, summaries, chain_id, coords, kinds, sizes, projection) -> None:
    start = len(records)
    for pos, (xy, kind, size) in enumerate(zip(coords, kinds, sizes)):
        point = unproject_geometry(Point(float(xy[0]), float(xy[1])), projection)
        records.append(
            {
                "node_index_zero_based": int(len(records)),
                "chain_id": int(chain_id),
                "chain_position": int(pos),
                "boundary_kind": str(kind),
                "target_spacing_m": float(size),
                "is_hard_anchor": bool(chain_id == 0 and kind == "open" and pos in {0, len(coords) - 1}),
                "geometry": point,
            }
        )
    summaries.append(
        {
            "chain_id": int(chain_id),
            "kind": "outer" if chain_id == 0 else "island",
            "node_count": int(len(coords)),
            "start_node_index_zero_based": int(start),
            "end_node_index_zero_based": int(len(records) - 1),
        }
    )


def _write_resolution_layers(gpkg, domain, open_line, islands, source_islands, node_records, source_metrics, resolved_records, projection) -> None:
    domain_ll = unproject_geometry(domain, projection)
    open_ll = unproject_geometry(open_line, projection)
    gpd.GeoDataFrame([{"profile": "adaptive-coastal-v1", "geometry": domain_ll}], geometry="geometry", crs="EPSG:4326").to_file(gpkg, layer="resolved_domain_polygon", driver="GPKG")
    gpd.GeoDataFrame([{"segment_class": "open_boundary", "geometry": open_ll}], geometry="geometry", crs="EPSG:4326").to_file(gpkg, layer="resolved_open_boundary", driver="GPKG")
    island_rows = []
    for idx, polygon in enumerate(islands):
        record = resolved_records[idx] if idx < len(resolved_records) else {}
        island_rows.append({**{k: _json_safe(v) for k, v in record.items()}, "geometry": unproject_geometry(polygon, projection)})
    if island_rows:
        gpd.GeoDataFrame(island_rows, geometry="geometry", crs="EPSG:4326").to_file(gpkg, layer="resolved_island_polygons", driver="GPKG")
    node_gdf = gpd.GeoDataFrame(node_records, geometry="geometry", crs="EPSG:4326")
    node_gdf.to_file(gpkg, layer="boundary_nodes", driver="GPKG")
    diagnostic_rows = []
    for idx, record in enumerate(source_metrics):
        geometry = unproject_geometry(source_islands[idx], projection) if idx < len(source_islands) else None
        diagnostic_rows.append({**{k: _json_safe(v) for k, v in record.items()}, "geometry": geometry})
    if diagnostic_rows:
        gpd.GeoDataFrame(diagnostic_rows, geometry="geometry", crs="EPSG:4326").to_file(gpkg, layer="island_diagnostics", driver="GPKG")


def _node_geojson(records: list[dict[str, Any]]) -> dict[str, Any]:
    features = []
    for record in records:
        props = {key: _json_safe(value) for key, value in record.items() if key != "geometry"}
        features.append({"type": "Feature", "properties": props, "geometry": mapping(record["geometry"])})
    return {"type": "FeatureCollection", "features": features}


def _plot_review(path, source_domain, resolved_domain, open_line, mission, projection, metrics) -> None:
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    gpd.GeoSeries([unproject_geometry(source_domain, projection)], crs="EPSG:4326").boundary.plot(ax=ax, color="#9aa0a6", linewidth=0.5, label="legacy")
    gpd.GeoSeries([unproject_geometry(resolved_domain, projection)], crs="EPSG:4326").boundary.plot(ax=ax, color="#16537e", linewidth=0.8, label="resolved")
    gpd.GeoSeries([unproject_geometry(open_line, projection)], crs="EPSG:4326").plot(ax=ax, color="#d00000", linewidth=2.0, label="resolved OBC")
    if mission is not None and not mission.is_empty:
        gpd.GeoSeries([unproject_geometry(mission, projection)], crs="EPSG:4326").boundary.plot(ax=ax, color="#7b2cbf", linewidth=0.8, linestyle="--", label="protected mission")
    ax.set_title(f"Adaptive coastal boundary resolution: {len(metrics)} source islands")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _select_polygon(geometry, reference: Point) -> Polygon:
    if isinstance(geometry, Polygon):
        return geometry
    parts = [part for part in getattr(geometry, "geoms", []) if isinstance(part, Polygon) and not part.is_empty]
    if not parts:
        return Polygon()
    containing = [part for part in parts if part.buffer(1.0e-8).covers(reference)]
    return max(containing or parts, key=lambda item: item.area)


def _count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        value = str(record.get(key, "unknown"))
        result[value] = result.get(value, 0) + 1
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
