"""Adaptive open-boundary and island resolution for FVCOM boundary packages.

This module is a clean-room, opt-in postprocessor.  It never changes the
legacy model-boundary-loop package.  Instead it writes a separate package
containing a resolved wet-domain polygon and explicit ordered constraint
nodes for downstream meshing.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely import STRtree
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


@dataclass(frozen=True)
class BoundaryResolutionV2Config(BoundaryResolutionConfig):
    """Opt-in controls for feature-anchored, passage-aware boundary sampling."""

    profile: str = "adaptive-coastal-v2"
    sharp_turn_threshold_deg: float = 35.0
    spit_turn_threshold_deg: float = 70.0
    anchor_chord_error_fraction: float = 0.20
    anchor_min_separation_factor: float = 0.75
    protected_elements_across: int = 4
    unprotected_elements_across: int = 3
    passage_search_spacing_m: float = 300.0
    passage_max_width_m: float = 5000.0
    passage_min_along_separation_m: float = 1500.0
    passage_min_spacing_m: float = 150.0


def boundary_resolution_config(profile: str) -> BoundaryResolutionConfig:
    """Return the profile-specific configuration without changing v1 settings."""
    if profile == "adaptive-coastal-v2":
        return BoundaryResolutionV2Config()
    if profile == "adaptive-coastal-v1":
        return BoundaryResolutionConfig()
    raise ValueError(f"Unsupported adaptive boundary-resolution profile: {profile}")


def _passage_gate_taxonomy(
    passage_report: dict[str, Any],
    use_v2: bool = True,
) -> tuple[list[str], list[str]]:
    """Separate topology-critical protected passages from review-only findings."""
    failures: list[str] = []
    advisories: list[str] = []
    if not use_v2:
        return failures, advisories
    if int(passage_report.get("protected_unresolved_count", 0)) > 0:
        failures.append("protected_passage_underresolved")
    if int(passage_report.get("unprotected_unresolved_count", 0)) > 0:
        advisories.append("unprotected_passage_underresolved")
    return failures, advisories


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
    reuse_boundary_resolution_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Build an adaptive boundary-resolution package without touching legacy outputs."""
    config = config or BoundaryResolutionConfig()
    if config.profile not in {"adaptive-coastal-v1", "adaptive-coastal-v2"}:
        raise ValueError("Boundary resolution builder requires profile adaptive-coastal-v1 or adaptive-coastal-v2")
    use_v2 = config.profile == "adaptive-coastal-v2"
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

    if reuse_boundary_resolution_manifest is not None:
        if not use_v2:
            raise ValueError("Resolved-topology reuse is only supported by adaptive-coastal-v2")
        source_metrics, action_report, resolved_islands, resolved_records = _load_reused_v1_topology(
            reuse_boundary_resolution_manifest,
            projection,
        )
    else:
        source_metrics = _island_metrics(islands_xy, source_domain, mission_xy, config)
        topologized, action_report = _apply_subgrid_actions(
            shell_polygon,
            islands_xy,
            source_metrics,
            mission_xy,
            config,
        )
        resolved_islands, resolved_records = _generalize_islands(topologized, mission_xy, config)

    passage_report = {
        "policy": "not_enabled_for_profile",
        "passages": [],
        "passage_count": 0,
        "protected_unresolved_count": 0,
        "unprotected_unresolved_count": 0,
        "automatic_topology_operation_count": 0,
    }
    land_controls: list[dict[str, Any]] = []
    island_target_overrides: dict[int, float] = {}
    if use_v2:
        passage_domain = Polygon(shell_polygon.exterior.coords, holes=[list(poly.exterior.coords) for poly in resolved_islands])
        passage_report, land_controls, island_target_overrides = _inventory_narrow_passages(
            landward_xy,
            resolved_islands,
            passage_domain,
            mission_xy,
            config,
            projection,
        )
        open_nodes, open_h, open_meta, open_sampling = _sample_open_arc_v2(repaired_open, config)
        land_nodes, land_h, land_meta, land_sampling = _sample_landward_v2(landward_xy, land_controls, config)
        outer_entries = _deduplicate_node_entries(
            [
                {
                    "xy": xy,
                    "boundary_kind": "open",
                    "target_spacing_m": h,
                    **meta,
                }
                for xy, h, meta in zip(open_nodes, open_h, open_meta)
            ]
            + [
                {
                    "xy": xy,
                    "boundary_kind": "land",
                    "target_spacing_m": h,
                    **meta,
                }
                for xy, h, meta in zip(land_nodes[1:-1], land_h[1:-1], land_meta[1:-1])
            ]
        )
        target_gradation_conditioning = _enforce_delivered_target_gradation(
            outer_entries,
            float(config.gradation),
        )
        outer_nodes = [item["xy"] for item in outer_entries]
        outer_kinds = [str(item["boundary_kind"]) for item in outer_entries]
        outer_h = [float(item["target_spacing_m"]) for item in outer_entries]
        outer_meta = outer_entries
    else:
        open_nodes, open_h = _sample_open_arc(repaired_open, config)
        land_nodes = _sample_line(landward_xy, config.land_spacing_m, include_end=True)
        outer_nodes = open_nodes + land_nodes[1:-1]
        outer_kinds = ["open"] * len(open_nodes) + ["land"] * max(0, len(land_nodes) - 2)
        outer_h = open_h + [float(config.land_spacing_m)] * max(0, len(land_nodes) - 2)
        outer_nodes = _deduplicate_ring(outer_nodes)
        outer_kinds = outer_kinds[: len(outer_nodes)]
        outer_h = outer_h[: len(outer_nodes)]
        outer_meta = []
        open_sampling = {}
        land_sampling = {}
        target_gradation_conditioning = {}

    island_chains: list[list[tuple[float, float]]] = []
    island_targets: list[float] = []
    for resolved_index, (record, polygon) in enumerate(zip(resolved_records, resolved_islands)):
        target = float(record["target_spacing_m"])
        if use_v2 and resolved_index in island_target_overrides:
            record["passage_harmonized_target_spacing_m"] = float(island_target_overrides[resolved_index])
            target = min(target, float(island_target_overrides[resolved_index]))
        if record["protected_mission"]:
            chain = _densify_closed_ring_vertices(polygon, target)
            candidate = Polygon(chain)
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
    if use_v2:
        _append_v2_outer_chain(node_records, chain_summaries, outer_meta, projection)
    else:
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
        config.profile,
        passage_report.get("passages", []) if use_v2 else None,
    )
    diagnostics_path = run_dir / "boundary_resolution_diagnostics.json"
    node_geojson_path = run_dir / "boundary_resolution_nodes.geojson"
    review_map = run_dir / "boundary_resolution_review_map.png"
    diagnostics = {
        "schema_version": "fvcom_boundary_resolution_diagnostics_v2" if use_v2 else "fvcom_boundary_resolution_diagnostics_v1",
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
    if use_v2:
        diagnostics["boundary_sampling"] = {
            "profile": config.profile,
            "open": open_sampling,
            "land": land_sampling,
            "junctions": _junction_diagnostics(outer_meta, config),
            "delivered_target_gradation_conditioning": target_gradation_conditioning,
        }
        diagnostics["channel_passages"] = passage_report
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
    spacing_qa = _boundary_spacing_qa(outer_nodes, outer_h) if use_v2 else {}
    hard_anchor_count = int(sum(bool(item.get("is_hard_anchor")) for item in node_records))
    landfall_hard_anchor_count = int(
        sum(item.get("anchor_type") == "open_landfall" and bool(item.get("is_hard_anchor")) for item in node_records)
    )
    if use_v2 and landfall_hard_anchor_count != 2:
        failures.append("open_landfall_hard_anchor_count_invalid")
    passage_failures, advisories = _passage_gate_taxonomy(passage_report, use_v2=use_v2)
    failures.extend(passage_failures)
    if use_v2 and float(spacing_qa.get("maximum_edge_to_target_ratio", 0.0)) > 1.55 + 1.0e-9:
        failures.append("boundary_edge_to_target_ratio_exceeded")
    if use_v2 and float(spacing_qa.get("maximum_target_gradation", 0.0)) > float(config.gradation) + 1.0e-9:
        failures.append("boundary_target_gradation_exceeded")
    manifest = {
        "schema_version": "fvcom_boundary_resolution_manifest_v2" if use_v2 else "fvcom_boundary_resolution_manifest_v1",
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "created_by": "fvcom-bdry-arc boundary_resolution.py",
        "profile": config.profile,
        "final_status": "pass" if not failures else "needs_review",
        "failure_taxonomy": failures,
        "advisory_taxonomy": advisories,
        "inputs": {
            "model_boundary_loops_gpkg": str(source_path),
            "model_boundary_loop_manifest": str(model_boundary_loop_manifest) if model_boundary_loop_manifest else None,
            "region_bpoly_json": str(region_bpoly_json) if region_bpoly_json else None,
            "coastline_gpkg": str(coastline_gpkg) if coastline_gpkg else None,
            "reused_boundary_resolution_manifest": (
                str(reuse_boundary_resolution_manifest) if reuse_boundary_resolution_manifest else None
            ),
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
            **(
                {
                    **spacing_qa,
                    "hard_anchor_count": hard_anchor_count,
                    "open_landfall_hard_anchor_count": landfall_hard_anchor_count,
                    "passage_count": int(passage_report["passage_count"]),
                    "protected_underresolved_passage_count": int(passage_report["protected_unresolved_count"]),
                    "unprotected_underresolved_passage_count": int(passage_report["unprotected_unresolved_count"]),
                    "automatic_passage_topology_operation_count": int(
                        passage_report["automatic_topology_operation_count"]
                    ),
                }
                if use_v2
                else {}
            ),
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


def _load_reused_v1_topology(
    manifest_path: str | Path,
    projection,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Polygon], list[dict[str, Any]]]:
    """Load an accepted v1 island topology for a v2-only prevention rebuild."""
    manifest_path = Path(manifest_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if document.get("profile") != "adaptive-coastal-v1" or document.get("final_status") != "pass":
        raise ValueError("Topology reuse requires an accepted adaptive-coastal-v1 boundary-resolution manifest")
    gpkg = Path(document["outputs"]["boundary_resolution_gpkg"])
    diagnostics_path = Path(document["outputs"]["boundary_resolution_diagnostics_json"])
    layers = set(gpd.list_layers(gpkg)["name"])
    required = {"resolved_island_polygons", "island_diagnostics"}
    if missing := required - layers:
        raise ValueError(f"Reusable v1 package is missing layers: {sorted(missing)}")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8-sig"))

    island_gdf = gpd.read_file(gpkg, layer="resolved_island_polygons").to_crs("EPSG:4326")
    if "resolved_island_id" in island_gdf.columns:
        island_gdf = island_gdf.sort_values("resolved_island_id")
    resolved_islands = [
        project_geometry(geometry, projection).buffer(0)
        for geometry in island_gdf.geometry
        if isinstance(geometry, Polygon) and not geometry.is_empty
    ]
    resolved_records = [dict(item) for item in diagnostics.get("resolved_islands", [])]
    if len(resolved_islands) != len(resolved_records):
        raise ValueError(
            "Reusable v1 package has inconsistent resolved-island geometry and diagnostic counts: "
            f"{len(resolved_islands)} != {len(resolved_records)}"
        )

    source_gdf = gpd.read_file(gpkg, layer="island_diagnostics")
    if "island_id" in source_gdf.columns:
        source_gdf = source_gdf.sort_values("island_id")
    source_metrics = [
        {
            str(column): row[column]
            for column in source_gdf.columns
            if column != "geometry"
        }
        for _, row in source_gdf.iterrows()
    ]
    action_report = dict(diagnostics.get("topology_actions", {}))
    required_actions = {
        "source_island_area_m2",
        "cumulative_absolute_area_change_m2",
        "protected_operation_count",
    }
    if missing := required_actions - set(action_report):
        raise ValueError(f"Reusable v1 topology report is missing fields: {sorted(missing)}")
    action_report["reused_from_manifest"] = str(manifest_path)
    action_report["reuse_policy"] = "accepted_v1_topology_v2_prevention_rebuild"
    return source_metrics, action_report, resolved_islands, resolved_records


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
    cleaned_islands = [polygon.buffer(0) for polygon in islands]
    island_tree = STRtree(cleaned_islands) if cleaned_islands else None
    for idx, polygon in enumerate(cleaned_islands):
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
        nearest = _nearest_other_island(idx, polygon, cleaned_islands, island_tree)
        if nearest is not None and nearest[1] < gap:
            other_idx, gap = nearest
            gap_line = _nearest_connector(polygon, cleaned_islands[other_idx])
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
    island_tree = STRtree(islands) if islands else None
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
        nearest = _nearest_other_island(idx, polygon, islands, island_tree)
        if nearest is not None and nearest[1] < gap:
            other_idx, gap = nearest
            gap_line = _nearest_connector(polygon, islands[other_idx])
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


def _nearest_other_island(
    index: int,
    polygon: Polygon,
    islands: list[Polygon],
    tree: STRtree | None,
) -> tuple[int, float] | None:
    """Return the exact nearest distinct island using the spatial index."""
    if tree is None or len(islands) < 2:
        return None
    indices, distances = tree.query_nearest(polygon, exclusive=True, return_distance=True)
    candidates = sorted(
        (float(distance), int(other_index))
        for other_index, distance in zip(np.asarray(indices).ravel(), np.asarray(distances).ravel())
        if int(other_index) != int(index)
    )
    if candidates:
        distance, other_index = candidates[0]
        return int(other_index), float(distance)
    # Degenerate duplicate geometries can be excluded together by GEOS.
    # Preserve exact behavior with a rare linear fallback for that case only.
    fallback = [
        (float(polygon.distance(other)), int(other_index))
        for other_index, other in enumerate(islands)
        if int(other_index) != int(index)
    ]
    if not fallback:
        return None
    distance, other_index = min(fallback)
    return int(other_index), float(distance)


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


def _inventory_narrow_passages(
    landward: LineString,
    islands: list[Polygon],
    domain: Polygon,
    mission,
    config: BoundaryResolutionConfig,
    projection,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, float]]:
    """Conservatively inventory wet connectors between nonlocal boundary banks.

    The inventory may lower paired sampling targets, but it never closes a
    channel or changes geographic topology. Ambiguous or unresolvable cases
    are retained and exposed as review gates.
    """
    max_width = float(getattr(config, "passage_max_width_m", 5000.0))
    search_spacing = float(getattr(config, "passage_search_spacing_m", 300.0))
    min_along = float(getattr(config, "passage_min_along_separation_m", 1500.0))
    min_spacing = float(getattr(config, "passage_min_spacing_m", config.land_spacing_m))
    raw_candidates: list[dict[str, Any]] = []

    # Same-chain search captures opposite banks of a narrow inlet/channel.
    sample_count = min(1200, max(3, int(math.ceil(float(landward.length) / max(search_spacing, 1.0))) + 1))
    sample_s = np.linspace(0.0, float(landward.length), sample_count)
    sample_xy = np.asarray([[landward.interpolate(float(s)).x, landward.interpolate(float(s)).y] for s in sample_s], dtype=float)
    sample_tangent = np.asarray(
        [_line_tangent_at(landward, float(s), search_spacing) for s in sample_s],
        dtype=float,
    )
    if sample_count >= 3:
        distances = np.linalg.norm(sample_xy[:, None, :] - sample_xy[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        neighbor_count = min(64, sample_count - 1)
        for first in range(sample_count):
            nearby = np.argpartition(distances[first], neighbor_count - 1)[:neighbor_count]
            for second in sorted((int(value) for value in nearby), key=lambda value: distances[first, value]):
                if second <= first or abs(float(sample_s[second] - sample_s[first])) < min_along:
                    continue
                width = float(distances[first, second])
                if not (1.0 < width <= max_width):
                    continue
                tangent_a = sample_tangent[first]
                tangent_b = sample_tangent[second]
                connector_unit = (sample_xy[second] - sample_xy[first]) / max(width, 1.0e-12)
                if abs(float(np.dot(tangent_a, tangent_b))) < 0.50:
                    continue
                if abs(float(np.dot(tangent_a, connector_unit))) > 0.70 or abs(float(np.dot(tangent_b, connector_unit))) > 0.70:
                    continue
                # Geometry-domain intersection is substantially more expensive
                # than the local tangent screen on a long crenulated shoreline.
                # Defer it until the vector tests identify a plausible opposite
                # bank, preserving the same conservative acceptance contract.
                connector = LineString([sample_xy[first], sample_xy[second]])
                if not _wet_connector_is_conservative(connector, domain):
                    continue
                raw_candidates.append(
                    {
                        "bank_a": "land",
                        "bank_b": "land",
                        "position_a_m": float(sample_s[first]),
                        "position_b_m": float(sample_s[second]),
                        "island_a": None,
                        "island_b": None,
                        "width_m": width,
                        "connector": connector,
                    }
                )
                break

    # Cross-component nearest connectors cover island/mainland and island/island gaps.
    components: list[tuple[str, int | None, Any]] = [("land", None, landward)] + [
        ("island", index, LineString(polygon.exterior.coords)) for index, polygon in enumerate(islands)
    ]
    component_geometries = [item[2] for item in components]
    component_tree = STRtree(component_geometries)
    indexed_component_pair_count = 0
    all_component_pair_count = len(components) * max(0, len(components) - 1) // 2
    for first, geometry_a in enumerate(component_geometries):
        nearby = component_tree.query(geometry_a, predicate="dwithin", distance=max_width)
        for second in sorted(int(value) for value in nearby if int(value) > first):
            indexed_component_pair_count += 1
            kind_a, island_a, geometry_a = components[first]
            kind_b, island_b, geometry_b = components[second]
            point_a, point_b = nearest_points(geometry_a, geometry_b)
            width = float(point_a.distance(point_b))
            if not (1.0 < width <= max_width):
                continue
            connector = LineString([point_a, point_b])
            if not _wet_connector_is_conservative(connector, domain):
                continue
            raw_candidates.append(
                {
                    "bank_a": kind_a,
                    "bank_b": kind_b,
                    "position_a_m": float(landward.project(point_a)) if kind_a == "land" else float(geometry_a.project(point_a)),
                    "position_b_m": float(landward.project(point_b)) if kind_b == "land" else float(geometry_b.project(point_b)),
                    "island_a": island_a,
                    "island_b": island_b,
                    "width_m": width,
                    "connector": connector,
                }
            )

    # Keep one narrow representative per local bank-pair neighborhood.
    accepted: list[dict[str, Any]] = []
    for candidate in sorted(raw_candidates, key=lambda item: item["width_m"]):
        duplicate = False
        for prior in accepted:
            same_components = {
                (candidate["bank_a"], candidate["island_a"]),
                (candidate["bank_b"], candidate["island_b"]),
            } == {
                (prior["bank_a"], prior["island_a"]),
                (prior["bank_b"], prior["island_b"]),
            }
            endpoint_distance = float(candidate["connector"].hausdorff_distance(prior["connector"]))
            if same_components and endpoint_distance <= max(500.0, min(candidate["width_m"], prior["width_m"])):
                duplicate = True
                break
        if not duplicate:
            accepted.append(candidate)

    passages: list[dict[str, Any]] = []
    land_controls: list[dict[str, Any]] = []
    island_targets: dict[int, float] = {}
    protected_unresolved = 0
    unprotected_unresolved = 0
    for passage_id, candidate in enumerate(accepted):
        connector = candidate.pop("connector")
        protected = bool(mission is not None and not mission.is_empty and connector.intersects(mission))
        elements = int(
            getattr(config, "protected_elements_across", 4)
            if protected
            else getattr(config, "unprotected_elements_across", 3)
        )
        required_h = float(candidate["width_m"] / max(elements, 1))
        unresolved = bool(required_h < min_spacing - 1.0e-9)
        if unresolved and protected:
            protected_unresolved += 1
        elif unresolved:
            unprotected_unresolved += 1
        action = "retain_needs_review" if unresolved else "harmonize_paired_spacing"
        if not unresolved:
            if candidate["bank_a"] == "land":
                land_controls.append(
                    {
                        "passage_id": int(passage_id),
                        "source_position_m": float(candidate["position_a_m"]),
                        "target_spacing_m": required_h,
                    }
                )
            elif candidate["island_a"] is not None:
                island_id = int(candidate["island_a"])
                island_targets[island_id] = min(island_targets.get(island_id, math.inf), required_h)
            if candidate["bank_b"] == "land":
                land_controls.append(
                    {
                        "passage_id": int(passage_id),
                        "source_position_m": float(candidate["position_b_m"]),
                        "target_spacing_m": required_h,
                    }
                )
            elif candidate["island_b"] is not None:
                island_id = int(candidate["island_b"])
                island_targets[island_id] = min(island_targets.get(island_id, math.inf), required_h)
        connector_ll = unproject_geometry(connector, projection)
        passages.append(
            {
                "passage_id": int(passage_id),
                **candidate,
                "protected_mission": protected,
                "required_elements_across": elements,
                "required_target_spacing_m": required_h,
                "minimum_permitted_spacing_m": min_spacing,
                "resolvable_at_minimum_spacing": not unresolved,
                "action": action,
                "automatic_topology_change": False,
                "connector_lonlat": [[float(x), float(y)] for x, y in connector_ll.coords],
            }
        )
    return (
        {
            "policy": "conservative_inventory_harmonize_only_no_topology_closure",
            "passage_count": int(len(passages)),
            "protected_unresolved_count": int(protected_unresolved),
            "unprotected_unresolved_count": int(unprotected_unresolved),
            "automatic_topology_operation_count": 0,
            "search_spacing_m": search_spacing,
            "maximum_inventory_width_m": max_width,
            "minimum_permitted_spacing_m": min_spacing,
            "all_component_pair_count": int(all_component_pair_count),
            "spatially_indexed_component_pair_count": int(indexed_component_pair_count),
            "passages": passages,
        },
        land_controls,
        island_targets,
    )


def _wet_connector_is_conservative(connector: LineString, domain: Polygon) -> bool:
    if connector.is_empty or connector.length <= 1.0:
        return False
    buffered = domain.buffer(2.0)
    if not buffered.covers(connector):
        return False
    for fraction in np.linspace(0.1, 0.9, 9):
        if not domain.buffer(0.25).covers(connector.interpolate(float(fraction), normalized=True)):
            return False
    return True


def _line_tangent_at(line: LineString, position: float, scale: float) -> np.ndarray:
    half = max(1.0, 0.5 * float(scale))
    start = line.interpolate(max(0.0, float(position) - half))
    end = line.interpolate(min(float(line.length), float(position) + half))
    vector = np.asarray([end.x - start.x, end.y - start.y], dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-12 else np.asarray([1.0, 0.0])


def _sample_open_arc_v2(
    line: LineString,
    config: BoundaryResolutionConfig,
) -> tuple[list[tuple[float, float]], list[float], list[dict[str, Any]], dict[str, Any]]:
    """Sample one OBC chain while retaining stable source-feature vertices exactly."""
    length = float(line.length)

    def target(s: float) -> float:
        return float(
            min(
                config.open_central_spacing_m,
                config.open_anchor_spacing_m + config.gradation * min(max(0.0, s), max(0.0, length - s)),
            )
        )

    anchors = _stable_feature_anchors(line, target, config)
    positions = _equidistributed_positions(line, anchors, target)
    # Curvature/chord control is retained, but every added point is explicitly
    # non-anchor and source anchors remain exact.
    for _ in range(12):
        added: list[float] = []
        for start, end in zip(positions[:-1], positions[1:]):
            chord = LineString([line.interpolate(start), line.interpolate(end)])
            section = substring(line, start, end)
            local_h = min(target(start), target(end), target(0.5 * (start + end)))
            if float(section.hausdorff_distance(chord)) > 0.10 * max(float(local_h), 1.0):
                added.append(0.5 * (start + end))
        if not added:
            break
        positions = sorted(set(positions + added))
    coords, sizes, metadata = _sample_records(line, positions, anchors, target, "open")
    return coords, sizes, metadata, {
        "method": "anchor_interval_metric_equidistribution_with_chord_guard",
        "source_length_m": length,
        "node_count": len(coords),
        "feature_anchor_count": int(sum(item["anchor_type"] != "open_landfall" for item in anchors)),
        "hard_anchor_count": int(len(anchors)),
        "anchors": anchors,
    }


def _sample_landward_v2(
    line: LineString,
    controls: list[dict[str, Any]],
    config: BoundaryResolutionConfig,
) -> tuple[list[tuple[float, float]], list[float], list[dict[str, Any]], dict[str, Any]]:
    """Sample land boundary with shared landfall targets and passage controls."""
    length = float(line.length)

    def target(s: float) -> float:
        distance_from_junction = min(max(0.0, s), max(0.0, length - s))
        value = max(
            float(config.land_spacing_m),
            float(config.open_anchor_spacing_m) - float(config.gradation) * distance_from_junction,
        )
        for control in controls:
            control_s = float(control["source_position_m"])
            control_h = float(control["target_spacing_m"])
            value = min(value, control_h + float(config.gradation) * abs(float(s) - control_s))
        return float(value)

    anchors = _stable_feature_anchors(line, target, config)
    positions = _equidistributed_positions(line, anchors, target)
    coords, sizes, metadata = _sample_records(line, positions, anchors, target, "land")
    return coords, sizes, metadata, {
        "method": "anchor_interval_metric_equidistribution_with_shared_junction_target",
        "source_length_m": length,
        "node_count": len(coords),
        "feature_anchor_count": int(sum(item["anchor_type"] != "open_landfall" for item in anchors)),
        "hard_anchor_count": int(len(anchors)),
        "junction_target_spacing_m": float(config.open_anchor_spacing_m),
        "interior_land_target_spacing_m": float(config.land_spacing_m),
        "gradation": float(config.gradation),
        "junction_transition_length_m": float(
            max(0.0, config.open_anchor_spacing_m - config.land_spacing_m) / max(float(config.gradation), 1.0e-12)
        ),
        "passage_control_count": int(len(controls)),
        "passage_controls": controls,
        "anchors": anchors,
    }


def _stable_feature_anchors(line: LineString, target, config: BoundaryResolutionConfig) -> list[dict[str, Any]]:
    """Return endpoints plus non-noisy sharp-turn/spit anchors on a source chain."""
    coords = np.asarray(line.coords, dtype=float)
    if len(coords) < 2:
        return []
    edge_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    length = float(line.length)
    candidates: list[dict[str, Any]] = []
    threshold = float(getattr(config, "sharp_turn_threshold_deg", 35.0))
    spit_threshold = float(getattr(config, "spit_turn_threshold_deg", 70.0))
    chord_fraction = float(getattr(config, "anchor_chord_error_fraction", 0.20))
    for idx in range(1, len(coords) - 1):
        incoming = coords[idx] - coords[idx - 1]
        outgoing = coords[idx + 1] - coords[idx]
        turn = _turn_angle_deg(incoming, outgoing)
        wide_idx0 = max(0, idx - 2)
        wide_idx1 = min(len(coords) - 1, idx + 2)
        wide_turn = _turn_angle_deg(coords[idx] - coords[wide_idx0], coords[wide_idx1] - coords[idx])
        chord = LineString([coords[wide_idx0], coords[wide_idx1]])
        chord_error = float(Point(coords[idx]).distance(chord))
        local_h = max(1.0, float(target(float(cumulative[idx]))))
        stable = bool(turn >= threshold and (wide_turn >= 0.65 * threshold or chord_error >= chord_fraction * local_h))
        if not stable:
            continue
        anchor_type = "spit_tip" if turn >= spit_threshold and chord_error >= chord_fraction * local_h else "sharp_turn"
        candidates.append(
            {
                "source_position_m": float(cumulative[idx]),
                "anchor_type": anchor_type,
                "source_vertex_index": int(idx),
                "turn_angle_deg": float(turn),
                "multiscale_turn_angle_deg": float(wide_turn),
                "chord_error_m": chord_error,
                "local_target_spacing_m": local_h,
                "score": float(max(turn / max(threshold, 1.0), chord_error / max(chord_fraction * local_h, 1.0))),
            }
        )
    selected: list[dict[str, Any]] = []
    selected_positions: list[float] = []
    for candidate in sorted(candidates, key=lambda item: (-item["score"], item["source_position_m"])):
        separation = float(getattr(config, "anchor_min_separation_factor", 0.75)) * candidate["local_target_spacing_m"]
        position = float(candidate["source_position_m"])
        insertion = bisect_left(selected_positions, position)
        neighbors = selected_positions[max(0, insertion - 1) : min(len(selected_positions), insertion + 1)]
        if all(abs(position - prior) >= separation for prior in neighbors):
            selected_positions.insert(insertion, position)
            selected.append(candidate)
    endpoints = [
        {
            "source_position_m": 0.0,
            "anchor_type": "open_landfall",
            "source_vertex_index": 0,
            "turn_angle_deg": 0.0,
            "multiscale_turn_angle_deg": 0.0,
            "chord_error_m": 0.0,
            "local_target_spacing_m": float(target(0.0)),
            "score": math.inf,
        },
        {
            "source_position_m": length,
            "anchor_type": "open_landfall",
            "source_vertex_index": int(len(coords) - 1),
            "turn_angle_deg": 0.0,
            "multiscale_turn_angle_deg": 0.0,
            "chord_error_m": 0.0,
            "local_target_spacing_m": float(target(length)),
            "score": math.inf,
        },
    ]
    result = endpoints + selected
    result.sort(key=lambda item: item["source_position_m"])
    for anchor_id, item in enumerate(result):
        item["anchor_id"] = f"{item['anchor_type']}_{anchor_id:04d}"
        item.pop("score", None)
    return result


def _turn_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norm <= 1.0e-12:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _equidistributed_positions(line: LineString, anchors: list[dict[str, Any]], target) -> list[float]:
    """Equidistribute integral(ds/h) independently between retained anchors."""
    positions: list[float] = []
    anchor_positions = sorted(set(float(item["source_position_m"]) for item in anchors))
    for interval_index, (start, end) in enumerate(zip(anchor_positions[:-1], anchor_positions[1:])):
        if end - start <= 1.0e-9:
            continue
        probe_count = min(2049, max(33, int(math.ceil((end - start) / 25.0)) + 1))
        probe = np.linspace(start, end, probe_count)
        weight = np.asarray([1.0 / max(float(target(float(s))), 1.0) for s in probe], dtype=float)
        cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (weight[:-1] + weight[1:]) * np.diff(probe))))
        interval_count = max(1, int(math.ceil(float(cumulative[-1]) - 1.0e-12)))
        desired = np.linspace(0.0, float(cumulative[-1]), interval_count + 1)
        local = np.interp(desired, cumulative, probe).tolist()
        if interval_index:
            local = local[1:]
        positions.extend(float(value) for value in local)
    if not positions:
        positions = [0.0, float(line.length)]
    positions[0] = 0.0
    positions[-1] = float(line.length)
    return positions


def _sample_records(
    line: LineString,
    positions: list[float],
    anchors: list[dict[str, Any]],
    target,
    source_chain: str,
) -> tuple[list[tuple[float, float]], list[float], list[dict[str, Any]]]:
    coords: list[tuple[float, float]] = []
    sizes: list[float] = []
    metadata: list[dict[str, Any]] = []
    ordered_anchors = sorted(anchors, key=lambda item: float(item["source_position_m"]))
    anchor_positions = np.asarray(
        [float(item["source_position_m"]) for item in ordered_anchors],
        dtype=float,
    )
    for position in positions:
        point = line.interpolate(float(position))
        match = None
        if len(anchor_positions):
            insertion = int(np.searchsorted(anchor_positions, float(position), side="left"))
            for candidate in (insertion - 1, insertion):
                if 0 <= candidate < len(ordered_anchors) and abs(anchor_positions[candidate] - float(position)) <= 1.0e-6:
                    match = ordered_anchors[candidate]
                    break
        coords.append((float(point.x), float(point.y)))
        sizes.append(float(target(float(position))))
        metadata.append(
            {
                "is_hard_anchor": bool(match is not None),
                "anchor_type": str(match["anchor_type"]) if match else "",
                "anchor_id": str(match["anchor_id"]) if match else "",
                "source_chain": source_chain,
                "source_position_m": float(position),
            }
        )
    return coords, sizes, metadata


def _deduplicate_node_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate coordinates without separating boundary metadata from nodes."""
    out: list[dict[str, Any]] = []
    for entry in entries:
        item = dict(entry)
        item["xy"] = (float(item["xy"][0]), float(item["xy"][1]))
        if out and np.linalg.norm(np.asarray(out[-1]["xy"]) - np.asarray(item["xy"])) <= 1.0e-7:
            out[-1]["is_hard_anchor"] = bool(out[-1].get("is_hard_anchor") or item.get("is_hard_anchor"))
            if not out[-1].get("anchor_type"):
                out[-1]["anchor_type"] = item.get("anchor_type", "")
                out[-1]["anchor_id"] = item.get("anchor_id", "")
            out[-1]["target_spacing_m"] = min(float(out[-1]["target_spacing_m"]), float(item["target_spacing_m"]))
            continue
        out.append(item)
    if len(out) > 1 and np.linalg.norm(np.asarray(out[0]["xy"]) - np.asarray(out[-1]["xy"])) <= 1.0e-7:
        out[0]["is_hard_anchor"] = bool(out[0].get("is_hard_anchor") or out[-1].get("is_hard_anchor"))
        out[0]["target_spacing_m"] = min(float(out[0]["target_spacing_m"]), float(out[-1]["target_spacing_m"]))
        out.pop()
    return out


def _append_v2_outer_chain(records, summaries, entries: list[dict[str, Any]], projection) -> None:
    start = len(records)
    for pos, item in enumerate(entries):
        point = unproject_geometry(Point(item["xy"]), projection)
        records.append(
            {
                "node_index_zero_based": int(len(records)),
                "chain_id": 0,
                "chain_position": int(pos),
                "boundary_kind": str(item["boundary_kind"]),
                "target_spacing_m": float(item["target_spacing_m"]),
                "is_hard_anchor": bool(item.get("is_hard_anchor", False)),
                "anchor_type": str(item.get("anchor_type", "")),
                "anchor_id": str(item.get("anchor_id", "")),
                "source_chain": str(item.get("source_chain", "")),
                "source_position_m": float(item.get("source_position_m", 0.0)),
                "geometry": point,
            }
        )
    summaries.append(
        {
            "chain_id": 0,
            "kind": "outer",
            "node_count": int(len(entries)),
            "start_node_index_zero_based": int(start),
            "end_node_index_zero_based": int(len(records) - 1),
            "hard_anchor_count": int(sum(bool(item.get("is_hard_anchor")) for item in entries)),
            "open_landfall_hard_anchor_count": int(
                sum(bool(item.get("is_hard_anchor")) and item.get("anchor_type") == "open_landfall" for item in entries)
            ),
        }
    )


def _boundary_spacing_qa(coords: list[tuple[float, float]], sizes: list[float]) -> dict[str, Any]:
    if len(coords) < 2:
        return {"maximum_edge_to_target_ratio": 0.0, "p95_edge_to_target_ratio": 0.0, "maximum_target_gradation": 0.0}
    points = np.asarray(coords, dtype=float)
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    target = np.asarray(sizes, dtype=float)
    ratios = lengths / np.maximum(np.minimum(target, np.roll(target, -1)), 1.0)
    gradation = np.abs(np.roll(target, -1) - target) / np.maximum(lengths, 1.0)
    return {
        "maximum_edge_to_target_ratio": float(np.max(ratios)),
        "p95_edge_to_target_ratio": float(np.percentile(ratios, 95.0)),
        "maximum_target_gradation": float(np.max(gradation)),
    }


def _enforce_delivered_target_gradation(
    entries: list[dict[str, Any]],
    gradation: float,
) -> dict[str, Any]:
    """Project targets onto the actual chord-length Lipschitz constraints."""
    if len(entries) < 2 or gradation <= 0.0:
        return {"adjusted_node_count": 0, "iteration_count": 0, "maximum_gradient": 0.0}
    points = np.asarray([item["xy"] for item in entries], dtype=float)
    raw = np.asarray([item["target_spacing_m"] for item in entries], dtype=float)
    target = raw.copy()
    effective_gradation = float(gradation) * (1.0 - 1.0e-4)
    fixed = np.asarray(
        [item.get("anchor_type") == "open_landfall" for item in entries],
        dtype=bool,
    )
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    iteration_count = 0
    for iteration_count in range(1, 1001):
        changed = False
        for index in range(len(entries)):
            following = (index + 1) % len(entries)
            limit = effective_gradation * max(float(lengths[index]), 1.0)
            difference = float(target[following] - target[index])
            if abs(difference) <= limit + 1.0e-10:
                continue
            sign = 1.0 if difference > 0.0 else -1.0
            if fixed[index] and fixed[following]:
                raise ValueError("Fixed landfall targets cannot satisfy delivered boundary gradation")
            if fixed[index]:
                target[following] = target[index] + sign * limit
            elif fixed[following]:
                target[index] = target[following] - sign * limit
            else:
                midpoint = 0.5 * (target[index] + target[following])
                target[index] = midpoint - 0.5 * sign * limit
                target[following] = midpoint + 0.5 * sign * limit
            changed = True
        if not changed:
            break
    gradients = np.abs(np.roll(target, -1) - target) / np.maximum(lengths, 1.0)
    maximum = float(np.max(gradients))
    if maximum > float(gradation) + 1.0e-8:
        raise ValueError(
            "Delivered target-gradation projection did not converge: "
            f"{maximum} > {gradation}"
        )
    for item, value in zip(entries, target):
        item["target_spacing_m"] = float(max(value, 1.0))
    adjusted = np.abs(target - raw) > 1.0e-9
    return {
        "method": "anchor_preserving_actual_chord_lipschitz_projection",
        "requested_gradation": float(gradation),
        "effective_projection_gradation": effective_gradation,
        "fixed_landfall_count": int(np.count_nonzero(fixed)),
        "adjusted_node_count": int(np.count_nonzero(adjusted)),
        "maximum_adjustment_m": float(np.max(np.abs(target - raw))),
        "iteration_count": int(iteration_count),
        "maximum_gradient": maximum,
    }


def _junction_diagnostics(entries: list[dict[str, Any]], config: BoundaryResolutionConfig) -> list[dict[str, Any]]:
    diagnostics = []
    count = len(entries)
    for index, item in enumerate(entries):
        if item.get("anchor_type") != "open_landfall":
            continue
        neighbors = [entries[(index - 1) % count], entries[(index + 1) % count]]
        land_neighbor = next((neighbor for neighbor in neighbors if neighbor.get("boundary_kind") == "land"), None)
        diagnostics.append(
            {
                "node_index_zero_based": int(index),
                "hard_anchor": bool(item.get("is_hard_anchor")),
                "shared_target_spacing_m": float(item["target_spacing_m"]),
                "expected_shared_target_spacing_m": float(config.open_anchor_spacing_m),
                "adjacent_land_target_spacing_m": float(land_neighbor["target_spacing_m"]) if land_neighbor else None,
                "adjacent_land_edge_length_m": float(
                    Point(item["xy"]).distance(Point(land_neighbor["xy"]))
                )
                if land_neighbor
                else None,
            }
        )
    return diagnostics


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


def _densify_closed_ring_vertices(polygon: Polygon, spacing: float) -> list[tuple[float, float]]:
    """Densify exact source segments while retaining every original vertex."""
    coords = list(polygon.exterior.coords)
    out: list[tuple[float, float]] = []
    for start, end in zip(coords[:-1], coords[1:]):
        start_xy = np.asarray(start, dtype=float)
        end_xy = np.asarray(end, dtype=float)
        length = float(np.linalg.norm(end_xy - start_xy))
        count = max(1, int(math.ceil(length / max(float(spacing), 1.0))))
        for index in range(count):
            fraction = float(index / count)
            point = (1.0 - fraction) * start_xy + fraction * end_xy
            if not out or np.linalg.norm(np.asarray(out[-1]) - point) > 1.0e-9:
                out.append((float(point[0]), float(point[1])))
    return out


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


def _write_resolution_layers(
    gpkg,
    domain,
    open_line,
    islands,
    source_islands,
    node_records,
    source_metrics,
    resolved_records,
    projection,
    profile: str = "adaptive-coastal-v1",
    passages: list[dict[str, Any]] | None = None,
) -> None:
    domain_ll = unproject_geometry(domain, projection)
    open_ll = unproject_geometry(open_line, projection)
    gpd.GeoDataFrame([{"profile": profile, "geometry": domain_ll}], geometry="geometry", crs="EPSG:4326").to_file(gpkg, layer="resolved_domain_polygon", driver="GPKG")
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
    if passages:
        passage_rows = []
        for record in passages:
            coords = record.get("connector_lonlat", [])
            if len(coords) < 2:
                continue
            passage_rows.append(
                {
                    **{key: _json_safe(value) for key, value in record.items() if key != "connector_lonlat"},
                    "geometry": LineString(coords),
                }
            )
        if passage_rows:
            gpd.GeoDataFrame(passage_rows, geometry="geometry", crs="EPSG:4326").to_file(
                gpkg, layer="passage_diagnostics", driver="GPKG"
            )


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
