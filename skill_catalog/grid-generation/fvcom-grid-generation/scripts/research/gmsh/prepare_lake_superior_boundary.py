#!/usr/bin/env python3
"""Prepare a closed Lake Superior adaptive-v2 boundary package from GSHHG.

GSHHG 2.3.7 represents Lake Superior and the downstream upper Great Lakes in
one level-2 feature. This research helper preserves the Superior shoreline and
all contained level-3 islands, then selects one deterministic St. Marys
numerical land gate from the source wet geometry. Candidate gates must be
shoreline-snapped, stay entirely in water, touch no L3 island, and separate a
Lake Superior reference point from a Lake Huron reference point. The gate is
not an open boundary: the delivered contract has exactly zero OBC chains.

The implementation intentionally reuses the tested Lake Ontario cache,
packaging, and provenance helpers.  All Superior-specific topology decisions
and all rewritten artifacts are implemented here; the Ontario helper is not
modified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely import make_valid
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import split, unary_union

import prepare_lake_ontario_boundary as base


DEFAULT_BBOX = (-92.25, 46.25, -83.95, 49.15)
LAKE_REFERENCE = (-87.20, 47.50)
DOWNSTREAM_REFERENCE = (-82.50, 44.80)
OUTLET_REFERENCE = (-84.4545, 46.47775)
OUTLET_CONTEXT_BBOX = (-84.90, 46.35, -84.15, 46.62)
GATE_SEARCH_BBOX = (-84.90, 46.46, -84.20, 46.51)
PROJECTED_CRS = "EPSG:32616"
GATE_SCAN_STEP_DEG = 0.00025
MINIMUM_GATE_LENGTH_M = 2_000.0
MAXIMUM_GATE_LENGTH_M = 25_000.0
MINIMUM_RETAINED_AREA_KM2 = 75_000.0
MAXIMUM_RETAINED_AREA_KM2 = 90_000.0
OUTLET_CLOSURE_TOLERANCE_DEG = 1.0e-7


_original_select_lake = base._select_lake
_original_select_islands = base._select_islands
_original_boundary_nodes = base._boundary_nodes
_closure_coords: tuple[tuple[float, float], tuple[float, float]] | None = None
_gate_level3: gpd.GeoDataFrame | None = None
_selected_island_source_records: list[dict[str, Any]] = []


def _deep_replace(value: Any) -> Any:
    """Replace Ontario-specific labels without changing generic provenance."""
    replacements = (
        ("lake_ontario", "lake_superior"),
        ("Lake Ontario", "Lake Superior"),
        ("LAKE ONTARIO", "LAKE SUPERIOR"),
        ("st_lawrence", "st_marys"),
        ("St. Lawrence", "St. Marys"),
        ("St Lawrence", "St Marys"),
    )
    if isinstance(value, str):
        updated = value
        for old, new in replacements:
            updated = updated.replace(old, new)
        return updated
    if isinstance(value, dict):
        return {key: _deep_replace(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_replace(item) for item in value]
    return value


def _select_lake_superior(level2: gpd.GeoDataFrame) -> tuple[Polygon, dict[str, Any]]:
    """Select the shortest accepted water-only gate in the St. Marys search."""
    global _closure_coords
    if _gate_level3 is None:
        raise RuntimeError("Lake Superior gate selection requires frozen GSHHG L3.")
    source, selection = _original_select_lake(level2)
    superior_reference = Point(*LAKE_REFERENCE)
    downstream_reference = Point(*DOWNSTREAM_REFERENCE)
    if not source.covers(superior_reference) or not source.covers(
        downstream_reference
    ):
        raise RuntimeError(
            "Selected joined GSHHG L2 source must contain both Superior and "
            "downstream reference points."
        )

    relevant_islands: list[Polygon] = []
    for geometry in _gate_level3.geometry:
        for polygon in base._polygon_parts(geometry):
            if source.intersects(polygon):
                relevant_islands.append(polygon)
    island_land = (
        unary_union(relevant_islands)
        if relevant_islands
        else Polygon()
    )
    wet_source = make_valid(source.difference(island_land))
    scan_south = float(GATE_SEARCH_BBOX[1])
    scan_north = float(GATE_SEARCH_BBOX[3])
    scan_count = int(
        round((scan_north - scan_south) / GATE_SCAN_STEP_DEG)
    ) + 1
    candidates: list[dict[str, Any]] = []
    evaluated_segment_count = 0
    for latitude in np.linspace(scan_south, scan_north, scan_count):
        scan = LineString(
            [
                (GATE_SEARCH_BBOX[0], float(latitude)),
                (GATE_SEARCH_BBOX[2], float(latitude)),
            ]
        )
        intersection = source.intersection(scan)
        segments = (
            [intersection]
            if isinstance(intersection, LineString)
            else [
                value
                for value in getattr(intersection, "geoms", [])
                if isinstance(value, LineString)
            ]
        )
        for segment in segments:
            evaluated_segment_count += 1
            if segment.is_empty or len(segment.coords) < 2:
                continue
            start = tuple(map(float, segment.coords[0]))
            end = tuple(map(float, segment.coords[-1]))
            if (
                start[0] < GATE_SEARCH_BBOX[0]
                or end[0] > GATE_SEARCH_BBOX[2]
            ):
                continue
            if not segment.intersection(island_land).is_empty:
                continue
            if segment.difference(source).length > 1.0e-10:
                continue
            if (
                Point(start).distance(source.boundary)
                > OUTLET_CLOSURE_TOLERANCE_DEG
                or Point(end).distance(source.boundary)
                > OUTLET_CLOSURE_TOLERANCE_DEG
            ):
                continue
            segment_xy = (
                gpd.GeoSeries([segment], crs="EPSG:4326")
                .to_crs(PROJECTED_CRS)
                .iloc[0]
            )
            length_m = float(segment_xy.length)
            if not (
                MINIMUM_GATE_LENGTH_M
                <= length_m
                <= MAXIMUM_GATE_LENGTH_M
            ):
                continue
            extension = max(
                0.002,
                0.02 * abs(float(end[0]) - float(start[0])),
            )
            splitter = LineString(
                [
                    (float(start[0]) - extension, float(latitude)),
                    (float(end[0]) + extension, float(latitude)),
                ]
            )
            wet_parts = [
                value
                for value in split(wet_source, splitter).geoms
                if isinstance(value, Polygon) and value.area > 1.0e-10
            ]
            superior_wet = next(
                (
                    value
                    for value in wet_parts
                    if value.covers(superior_reference)
                ),
                None,
            )
            downstream_wet = next(
                (
                    value
                    for value in wet_parts
                    if value.covers(downstream_reference)
                ),
                None,
            )
            if (
                superior_wet is None
                or downstream_wet is None
                or superior_wet is downstream_wet
            ):
                continue
            retained_area_km2 = float(
                gpd.GeoSeries([superior_wet], crs="EPSG:4326")
                .to_crs(PROJECTED_CRS)
                .iloc[0]
                .area
                / 1.0e6
            )
            if not (
                MINIMUM_RETAINED_AREA_KM2
                <= retained_area_km2
                <= MAXIMUM_RETAINED_AREA_KM2
            ):
                continue
            source_parts = [
                value
                for value in split(source, splitter).geoms
                if isinstance(value, Polygon) and value.area > 1.0e-10
            ]
            superior_shell = next(
                (
                    value
                    for value in source_parts
                    if value.covers(superior_reference)
                ),
                None,
            )
            downstream_shell = next(
                (
                    value
                    for value in source_parts
                    if value.covers(downstream_reference)
                ),
                None,
            )
            if (
                superior_shell is None
                or downstream_shell is None
                or superior_shell is downstream_shell
            ):
                continue
            candidates.append(
                {
                    "length_m": length_m,
                    "latitude_deg": float(latitude),
                    "start": start,
                    "end": end,
                    "retained_area_km2_before_island_subtraction": (
                        float(
                            gpd.GeoSeries(
                                [superior_shell],
                                crs="EPSG:4326",
                            )
                            .to_crs(PROJECTED_CRS)
                            .iloc[0]
                            .area
                            / 1.0e6
                        )
                    ),
                    "retained_wet_area_km2": retained_area_km2,
                    "shell": superior_shell,
                }
            )
    if not candidates:
        raise RuntimeError(
            "No shoreline-snapped, water-only St. Marys gate separated Lake "
            "Superior from the downstream reference."
        )
    selected = min(
        candidates,
        key=lambda value: (
            float(value["length_m"]),
            float(value["latitude_deg"]),
            float(value["start"][0]),
        ),
    )
    superior = selected.pop("shell")
    closure_vertices = (selected["start"], selected["end"])
    _closure_coords = closure_vertices
    selection.update(
        {
            "selected_source_bounds": list(map(float, source.bounds)),
            "selected_source_interior_ring_count": len(source.interiors),
            "complete_source_feature_retained": False,
            "retained_lake": "Lake Superior",
            "source_feature_semantics": (
                "GSHHG L2 feature joins Lake Superior to the downstream upper "
                "Great Lakes through the St. Marys waterway"
            ),
            "retained_shell_bounds": list(map(float, superior.bounds)),
            "retained_shell_component_count": 1,
            "gate_search": {
                "bbox_wsen": list(map(float, GATE_SEARCH_BBOX)),
                "scan_orientation": "constant_latitude_water_slices",
                "scan_step_deg": float(GATE_SCAN_STEP_DEG),
                "evaluated_segment_count": int(evaluated_segment_count),
                "accepted_candidate_count": int(len(candidates)),
                "selection_policy": (
                    "shortest accepted projected water cross-section; "
                    "deterministic latitude/longitude tie break"
                ),
                "minimum_gate_length_m": float(MINIMUM_GATE_LENGTH_M),
                "maximum_gate_length_m": float(MAXIMUM_GATE_LENGTH_M),
                "retained_area_range_km2": [
                    float(MINIMUM_RETAINED_AREA_KM2),
                    float(MAXIMUM_RETAINED_AREA_KM2),
                ],
                "superior_reference_lonlat": list(LAKE_REFERENCE),
                "downstream_reference_lonlat": list(
                    DOWNSTREAM_REFERENCE
                ),
                "shoreline_snapped_endpoints": True,
                "source_water_overlap_fraction": 1.0,
                "l3_land_intersection_count": 0,
                "separates_superior_and_downstream_references": True,
            },
            "outlet_closure": {
                "name": "st_marys_closed_lake_numerical_boundary",
                "kind": "land",
                "is_open_boundary": False,
                "latitude_deg": float(selected["latitude_deg"]),
                "endpoints_lonlat": [
                    [float(lon), float(lat)] for lon, lat in closure_vertices
                ],
                "length_m": float(selected["length_m"]),
                "retained_wet_area_km2": float(
                    selected["retained_wet_area_km2"]
                ),
                "source_vertex_count": 0,
                "artificial_hard_anchor_count": 2,
                "shoreline_snapped": True,
                "source_water_overlap_fraction": 1.0,
                "l3_land_intersection_count": 0,
            },
        }
    )
    return superior, selection


def _select_islands_superior(
    level3: gpd.GeoDataFrame,
    lake: Polygon,
) -> tuple[list[Polygon], list[dict[str, Any]]]:
    """Capture the exact frozen L3 source inventory retained as holes."""
    global _selected_island_source_records
    islands, records = _original_select_islands(level3, lake)
    _selected_island_source_records = [dict(value) for value in records]
    return islands, records


def _boundary_nodes_superior(
    domain: Polygon,
    outlet_point: Point,
) -> tuple[gpd.GeoDataFrame, float, list[dict[str, Any]]]:
    """Mark both numerical-closure endpoints as non-source hard anchors."""
    nodes, target_spacing, chains = _original_boundary_nodes(domain, outlet_point)
    if _closure_coords is None:
        raise RuntimeError("Lake Superior closure vertices were not established.")
    closure_indices: list[int] = []
    for index, row in nodes.iterrows():
        point = row.geometry
        if any(
            math.hypot(point.x - lon, point.y - lat)
            <= OUTLET_CLOSURE_TOLERANCE_DEG
            for lon, lat in _closure_coords
        ):
            closure_indices.append(int(index))
    if len(closure_indices) != 2:
        raise RuntimeError(
            "Boundary package must contain both St. Marys closure endpoints; "
            f"found {len(closure_indices)}."
        )
    exterior_mask = nodes["chain_id"] == 0
    nodes.loc[exterior_mask, "is_hard_anchor"] = False
    nodes.loc[exterior_mask, "outlet_context_anchor"] = False
    nodes.loc[closure_indices, "is_hard_anchor"] = True
    nodes.loc[closure_indices, "outlet_context_anchor"] = True
    nodes.loc[closure_indices, "is_source_vertex"] = False
    nodes.loc[closure_indices, "source_level"] = 0
    nodes.loc[closure_indices, "source_vertex_index"] = None
    chains[0]["hard_anchor_count"] = 2
    return nodes, target_spacing, chains


def _plot_evidence_superior(
    output_path: Path,
    level1_context: gpd.GeoDataFrame,
    domain: Polygon,
    islands: list[Polygon],
    outlet_point: Point,
    metrics: dict[str, Any],
) -> None:
    figure, axis = plt.subplots(figsize=(13.5, 7.5), constrained_layout=True)
    if not level1_context.empty:
        level1_context.plot(
            ax=axis,
            color="#d9d2c3",
            edgecolor="#81796b",
            linewidth=0.30,
        )
    gpd.GeoDataFrame(geometry=[domain], crs="EPSG:4326").plot(
        ax=axis,
        color="#70b7d5",
        edgecolor="#174f72",
        linewidth=0.70,
    )
    if islands:
        gpd.GeoDataFrame(geometry=islands, crs="EPSG:4326").plot(
            ax=axis,
            color="#d9d2c3",
            edgecolor="#5f4b32",
            linewidth=0.50,
        )
    if _closure_coords is None:
        raise RuntimeError("Cannot plot an undefined St. Marys closure.")
    closure = gpd.GeoDataFrame(
        {"kind": ["closed_lake_numerical_land_boundary"]},
        geometry=[LineString(_closure_coords)],
        crs="EPSG:4326",
    )
    closure.plot(
        ax=axis,
        color="#b91c1c",
        linewidth=2.4,
        label="St. Marys numerical closure (land; not OBC)",
    )
    axis.scatter(
        [coord[0] for coord in _closure_coords],
        [coord[1] for coord in _closure_coords],
        s=38,
        marker="o",
        color="#b91c1c",
        edgecolors="white",
        linewidths=0.5,
        zorder=5,
        label="closure hard anchors",
    )
    axis.scatter(
        [outlet_point.x],
        [outlet_point.y],
        s=54,
        marker="*",
        color="#7f1d1d",
        edgecolors="white",
        linewidths=0.5,
        zorder=6,
        label="outlet-context midpoint",
    )
    axis.set_xlim(DEFAULT_BBOX[0], DEFAULT_BBOX[2])
    axis.set_ylim(DEFAULT_BBOX[1], DEFAULT_BBOX[3])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(
        "Lake Superior closed-lake boundary: GSHHG L2 shoreline with L3 islands\n"
        f"{metrics['wet_area_km2']:.1f} km² wet area; "
        f"{metrics['island_hole_count']} island holes; 0 OBC chains"
    )
    axis.legend(loc="lower left", frameon=True)
    axis.grid(True, linewidth=0.25, alpha=0.35)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    gate_path = output_path.with_name("lake_superior_gate_evidence.png")
    gate_figure, gate_axis = plt.subplots(
        figsize=(11.5, 7.5),
        constrained_layout=True,
    )
    if not level1_context.empty:
        level1_context.plot(
            ax=gate_axis,
            color="#d9d2c3",
            edgecolor="#81796b",
            linewidth=0.45,
        )
    gpd.GeoDataFrame(
        geometry=[domain],
        crs="EPSG:4326",
    ).plot(
        ax=gate_axis,
        color="#70b7d5",
        edgecolor="#174f72",
        linewidth=0.85,
    )
    if _gate_level3 is not None and not _gate_level3.empty:
        gate_islands = gpd.clip(
            _gate_level3,
            gpd.GeoDataFrame(
                geometry=[box(*OUTLET_CONTEXT_BBOX)],
                crs="EPSG:4326",
            ),
            keep_geom_type=True,
        )
        if not gate_islands.empty:
            gate_islands.plot(
                ax=gate_axis,
                color="#d9d2c3",
                edgecolor="#5f4b32",
                linewidth=0.65,
            )
    if _closure_coords is None:
        raise RuntimeError("Cannot plot an undefined St. Marys closure.")
    gate_length_km = float(
        gpd.GeoSeries(
            [LineString(_closure_coords)],
            crs="EPSG:4326",
        )
        .to_crs(PROJECTED_CRS)
        .iloc[0]
        .length
        / 1000.0
    )
    gate_axis.plot(
        [value[0] for value in _closure_coords],
        [value[1] for value in _closure_coords],
        color="#b91c1c",
        linewidth=3.0,
        label=(
            "selected water-only numerical land gate "
            f"({gate_length_km:.2f} km)"
        ),
    )
    gate_axis.scatter(
        [value[0] for value in _closure_coords],
        [value[1] for value in _closure_coords],
        s=52,
        color="#b91c1c",
        edgecolors="white",
        linewidths=0.7,
        zorder=6,
        label="shoreline-snapped hard anchors",
    )
    gate_axis.set_xlim(OUTLET_CONTEXT_BBOX[0], OUTLET_CONTEXT_BBOX[2])
    gate_axis.set_ylim(OUTLET_CONTEXT_BBOX[1], OUTLET_CONTEXT_BBOX[3])
    gate_axis.set_aspect("equal", adjustable="box")
    gate_axis.set_xlabel("Longitude")
    gate_axis.set_ylabel("Latitude")
    gate_axis.set_title(
        "Lake Superior outlet gate evidence\n"
        "zero GSHHG L3 land intersection; Superior/Huron references separated"
    )
    gate_axis.legend(loc="best", frameon=True)
    gate_axis.grid(True, linewidth=0.25, alpha=0.35)
    gate_figure.savefig(gate_path, dpi=200)
    plt.close(gate_figure)


def _write_package_superior(
    output_dir: Path,
    domain: Polygon,
    islands: list[Polygon],
    nodes: gpd.GeoDataFrame,
    level1_context: gpd.GeoDataFrame,
    lake_source: Polygon,
    outlet_point: Point,
) -> tuple[Path, Path]:
    """Write a self-describing Superior package without mutating it afterward."""
    gpkg_path = output_dir / "boundary_resolution.gpkg"
    if gpkg_path.exists():
        raise FileExistsError(
            f"{gpkg_path} already exists. Use a fresh preparation directory "
            "to keep runs immutable."
        )
    gpd.GeoDataFrame(
        {"name": ["lake_superior_closed_wet_domain"]},
        geometry=[domain],
        crs="EPSG:4326",
    ).to_file(gpkg_path, layer="resolved_domain_polygon", driver="GPKG")
    base._empty_open_boundary().to_file(
        gpkg_path,
        layer="resolved_open_boundary",
        driver="GPKG",
    )
    gpd.GeoDataFrame(
        {"boundary_kind": ["land"]},
        geometry=[LineString(domain.exterior.coords)],
        crs="EPSG:4326",
    ).to_file(gpkg_path, layer="resolved_land_boundary", driver="GPKG")
    if islands:
        gpd.GeoDataFrame(
            {
                "island_id": list(range(1, len(islands) + 1)),
                "source_level": [3] * len(islands),
            },
            geometry=islands,
            crs="EPSG:4326",
        ).to_file(
            gpkg_path,
            layer="resolved_island_polygons",
            driver="GPKG",
        )
    nodes.to_file(gpkg_path, layer="boundary_nodes", driver="GPKG")
    gpd.GeoDataFrame(
        {"source_level": [2], "complete_feature": [False]},
        geometry=[lake_source],
        crs="EPSG:4326",
    ).to_file(
        gpkg_path,
        layer="source_l2_lake_polygon",
        driver="GPKG",
    )
    if not level1_context.empty:
        level1_context.to_file(
            gpkg_path,
            layer="source_l1_land_context",
            driver="GPKG",
        )
    gpd.GeoDataFrame(
        {
            "name": ["st_marys_outlet_context"],
            "is_open_boundary": [False],
            "protected_context": [True],
        },
        geometry=[outlet_point],
        crs="EPSG:4326",
    ).to_file(gpkg_path, layer="outlet_context", driver="GPKG")
    if _closure_coords is None:
        raise RuntimeError("Cannot write an undefined St. Marys closure.")
    closure_xy = (
        gpd.GeoSeries(
            [LineString(_closure_coords)],
            crs="EPSG:4326",
        )
        .to_crs(PROJECTED_CRS)
        .iloc[0]
    )
    gpd.GeoDataFrame(
        {
            "name": ["st_marys_numerical_land_gate"],
            "boundary_kind": ["land"],
            "is_open_boundary": [False],
            "shoreline_snapped": [True],
            "l3_land_crossing": [False],
            "length_m": [float(closure_xy.length)],
        },
        geometry=[LineString(_closure_coords)],
        crs="EPSG:4326",
    ).to_file(gpkg_path, layer="numerical_land_gate", driver="GPKG")
    nodes_geojson = output_dir / "boundary_resolution_nodes.geojson"
    nodes.to_file(nodes_geojson, driver="GeoJSON")
    return gpkg_path, nodes_geojson


def _rewrite_request_artifacts(
    output_dir: Path,
    request: dict[str, Any],
    estimate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = _deep_replace(request)
    estimate = _deep_replace(estimate)
    request["case_id"] = "lake_superior"
    request["clip_policy"] = (
        "retain the Lake Superior portion of the joined upper-Great-Lakes L2 "
        "feature using the shortest deterministic shoreline-snapped, "
        "water-only St. Marys gate that separates Superior from Huron"
    )
    request["open_boundary_policy"] = (
        "closed lake; the St. Marys closure is land and exactly zero OBC chains"
    )
    request["level_semantics"]["2"] = (
        "Lake Superior wet shell retained from the joined upper-Great-Lakes L2 feature"
    )
    estimate["case_id"] = "lake_superior"
    base._write_json(output_dir / "gshhs_request.json", request)
    base._write_json(output_dir / "download_estimate.json", estimate)
    return request, estimate


def _rewrite_package_artifacts(
    output_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    if result["manifest"].get("final_status") != "pass":
        raise RuntimeError(
            "Base closed-lake preparation did not pass; refusing to rewrite "
            "its status."
        )
    report = _deep_replace(result["report"])
    manifest = _deep_replace(result["manifest"])
    old_map = output_dir / "lake_ontario_boundary_evidence.png"
    new_map = output_dir / "lake_superior_boundary_evidence.png"
    old_report = output_dir / "lake_ontario_topology_report.json"
    new_report = output_dir / "lake_superior_topology_report.json"
    if old_map.exists():
        old_map.replace(new_map)
    if old_report.exists():
        old_report.replace(new_report)

    nodes = gpd.read_file(output_dir / "boundary_resolution_nodes.geojson")
    source_node_count = int(nodes["is_source_vertex"].astype(bool).sum())
    hard_anchor_count = int(nodes["is_hard_anchor"].astype(bool).sum())
    selection = report["selection"]
    closure = selection["outlet_closure"]
    island_inventory = [
        {
            "source_row": int(value["source_row"]),
            "source_id": str(value["source_id"]),
            "source_part": int(value["source_part"]),
            "bounds": list(map(float, value["bounds"])),
        }
        for value in _selected_island_source_records
    ]
    report.update(
        {
            "schema_version": "lake_superior_closed_boundary_report_v1",
            "case_id": "lake_superior",
            "preparation_status": "pass",
        }
    )
    report["boundary_contract"].update(
        {
            "source_vertices_preserved": True,
            "source_vertex_count": source_node_count,
            "artificial_outlet_closure_vertex_count": 2,
            "hard_anchor_count": hard_anchor_count,
            "outlet_closure_kind": "land",
            "outlet_closure_is_open_boundary": False,
        }
    )
    report["topology"].update(
        {
            "outlet_context_reference_lonlat": list(OUTLET_REFERENCE),
            "downstream_separation_reference_lonlat": list(
                DOWNSTREAM_REFERENCE
            ),
            "st_marys_closure_length_m": closure["length_m"],
            "st_marys_closure_endpoints_lonlat": closure["endpoints_lonlat"],
            "st_marys_closure_shoreline_snapped": True,
            "st_marys_closure_l3_land_intersection_count": 0,
            "open_boundary_chain_count": 0,
            "open_boundary_node_count": 0,
        }
    )
    report["gshhg_l3_island_source_inventory"] = {
        "record_count": int(len(island_inventory)),
        "records": island_inventory,
        "source_ids": sorted(
            {str(value["source_id"]) for value in island_inventory}
        ),
    }
    report["outputs"]["boundary_resolution_review_map"] = str(new_map)
    report["outputs"]["outlet_gate_evidence_map"] = str(
        output_dir / "lake_superior_gate_evidence.png"
    )
    report["outputs"]["topology_report"] = str(new_report)
    report["scope_exclusions"] = [
        "No bathymetry was fetched by the boundary-preparation step.",
        "No shared schema, shared module, or existing case manifest was modified.",
        "No regional mesh was generated by the boundary-preparation step.",
    ]

    manifest.update(
        {
            "name": "lake_superior_closed_lake_gshhg_h_l2_l3",
            "created_by": "research/gmsh/prepare_lake_superior_boundary.py",
            "final_status": "pass",
        }
    )
    manifest["advisory_taxonomy"] = [
        "gshhg_l2_upper_great_lakes_feature_closed_at_st_marys"
    ]
    manifest["inputs"].update(
        {
            "selection_reference_lonlat": list(LAKE_REFERENCE),
            "downstream_separation_reference_lonlat": list(
                DOWNSTREAM_REFERENCE
            ),
            "outlet_reference_lonlat": list(OUTLET_REFERENCE),
            "st_marys_gate_search_bbox_wsen": list(GATE_SEARCH_BBOX),
            "st_marys_closure_latitude_deg": float(
                closure["latitude_deg"]
            ),
        }
    )
    manifest["settings"].update(
        {
            "complete_level_2_feature_retained": False,
            "retained_lake": "Lake Superior",
            "outlet_closure_kind": "land",
            "outlet_closure_is_open_boundary": False,
            "outlet_closure_endpoint_policy": "two_artificial_hard_anchors",
            "outlet_closure_selection_policy": (
                "shortest deterministic shoreline-snapped water-only gate"
            ),
        }
    )
    manifest["qa"].update(
        {
            "hard_anchor_count": hard_anchor_count,
            "source_boundary_node_count": source_node_count,
            "artificial_outlet_closure_vertex_count": 2,
            "outlet_context_preserved": True,
            "st_marys_closure_length_m": closure["length_m"],
            "st_marys_closure_shoreline_snapped": True,
            "st_marys_closure_l3_land_intersection_count": 0,
            "gshhg_l3_island_source_record_count": int(
                len(island_inventory)
            ),
            "open_boundary_chain_count": 0,
            "open_boundary_node_count": 0,
        }
    )
    manifest["gshhg_l3_island_source_inventory"] = {
        "record_count": int(len(island_inventory)),
        "records": island_inventory,
        "source_ids": sorted(
            {str(value["source_id"]) for value in island_inventory}
        ),
    }
    manifest["outputs"]["boundary_resolution_review_map"] = str(new_map)
    manifest["outputs"]["outlet_gate_evidence_map"] = str(
        output_dir / "lake_superior_gate_evidence.png"
    )
    manifest["outputs"]["boundary_resolution_diagnostics_json"] = str(new_report)

    manifest_path = output_dir / "boundary_resolution_manifest.json"
    base._write_json(manifest_path, manifest)
    base._write_json(new_report, report)
    return {
        "manifest": manifest,
        "report": report,
        "manifest_path": str(manifest_path),
        "report_path": str(new_report),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cache-dir",
        default="Workspace/Preprocessing/fvcom-gshhs-coastline/cache/gshhg",
    )
    parser.add_argument(
        "--resolution",
        choices=("h",),
        default="h",
        help="Frozen topology contract currently supports GSHHG high resolution.",
    )
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit the official GSHHG download only when the cache is incomplete.",
    )
    return parser


def _require_fresh_output_directory(output_dir: Path) -> None:
    """Reject any existing destination before a preparation artifact is written."""
    if output_dir.exists():
        raise FileExistsError(
            "Lake Superior boundary output directory must not already exist: "
            f"{output_dir}"
        )


def main() -> int:
    global _gate_level3
    args = _parser().parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (workspace_root / output_dir).resolve()
    _require_fresh_output_directory(output_dir)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (workspace_root / cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    bbox_values = tuple(float(value) for value in args.bbox)

    base.DEFAULT_BBOX = DEFAULT_BBOX
    base.LAKE_REFERENCE = LAKE_REFERENCE
    base.OUTLET_REFERENCE = OUTLET_REFERENCE
    base.OUTLET_CONTEXT_BBOX = OUTLET_CONTEXT_BBOX
    base.PROJECTED_CRS = PROJECTED_CRS
    base._select_lake = _select_lake_superior
    base._select_islands = _select_islands_superior
    base._boundary_nodes = _boundary_nodes_superior
    base._plot_evidence = _plot_evidence_superior
    base._write_package = _write_package_superior

    request, estimate = base._write_request_and_estimate(
        output_dir,
        cache_dir,
        args.resolution,
        bbox_values,
    )
    _, estimate = _rewrite_request_artifacts(
        output_dir,
        request,
        estimate,
    )
    if args.estimate_only:
        print(json.dumps(estimate, indent=2))
        return 0

    source_manifest = base._ensure_sources(
        cache_dir,
        args.resolution,
        estimate,
        allow_download=bool(args.allow_download),
    )
    base._write_json(
        output_dir / "gshhg_source_cache_manifest.json",
        source_manifest,
    )
    _gate_level3 = base._read_level(
        base._source_paths(cache_dir, args.resolution, 3)[0],
        bbox_values,
    )
    result = base._prepare(
        output_dir,
        cache_dir,
        args.resolution,
        bbox_values,
        source_manifest,
    )
    result = _rewrite_package_artifacts(output_dir, result)
    topology = result["report"]["topology"]
    if topology["wet_component_count"] != 1:
        raise RuntimeError("Lake Superior preparation did not retain one wet component.")
    if topology["island_hole_count"] != len(_selected_island_source_records):
        raise RuntimeError(
            "Resolved island holes do not match the frozen GSHHG L3 source "
            f"inventory: holes={topology['island_hole_count']}, "
            f"records={len(_selected_island_source_records)}."
        )
    if topology["open_boundary_chain_count"] != 0:
        raise RuntimeError("Closed Lake Superior preparation created an OBC.")
    print(
        json.dumps(
            {
                "status": "pass",
                "manifest": result["manifest_path"],
                "report": result["report_path"],
                "topology": topology,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
