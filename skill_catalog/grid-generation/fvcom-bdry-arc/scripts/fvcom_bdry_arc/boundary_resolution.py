"""Adaptive v2 open-boundary and island resolution for FVCOM packages.

The builder preserves the base model-boundary-loop package and writes a
separate resolved wet-domain polygon plus explicit ordered constraints.
"""

from __future__ import annotations

import atexit
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
import heapq
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely import STRtree
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box, mapping
from shapely.ops import nearest_points, substring, unary_union
from shapely.prepared import prep

from .projection import (
    local_utm_projection,
    project_geometry,
    project_geometry_densified,
    projection_from_manifest,
    unproject_geometry,
)


@dataclass(frozen=True)
class BoundaryResolutionConfig:
    """Controls for the sole Adaptive v2 boundary-resolution profile."""

    profile: str = "adaptive-coastal-v2"
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


    sharp_turn_threshold_deg: float = 35.0
    spit_turn_threshold_deg: float = 70.0
    anchor_chord_error_fraction: float = 0.20
    anchor_min_separation_factor: float = 0.75
    protected_elements_across: int = 4
    unprotected_elements_across: int = 3
    passage_search_spacing_m: float = 300.0
    passage_max_width_m: float = 5000.0
    passage_min_along_separation_m: float = 1500.0
    passage_min_spacing_m: float | None = None
    progress_interval_s: float = 5.0


BoundaryResolutionV2Config = BoundaryResolutionConfig


ProgressCallback = Callable[[int, int, dict[str, Any] | None], None]


_PROGRESS_PHASE_RANGES: dict[str, tuple[float, float]] = {
    "start": (0.0, 1.0),
    "load_inputs": (1.0, 8.0),
    "outer_boundary": (8.0, 12.0),
    "source_island_metrics": (12.0, 30.0),
    "subgrid_topology": (30.0, 50.0),
    "island_generalization": (50.0, 65.0),
    "passage_inventory": (65.0, 80.0),
    "boundary_sampling": (80.0, 92.0),
    "write_outputs": (92.0, 96.0),
    "quality_gates": (96.0, 99.0),
    "complete": (100.0, 100.0),
}


class _BoundaryResolutionProgress:
    """Persist auditable, monotonic Adaptive v2 progress and item counts."""

    def __init__(self, run_dir: Path, interval_s: float) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "boundary_resolution_progress.jsonl"
        self.state_path = self.run_dir / "boundary_resolution_progress_state.json"
        self.started_utc = datetime.now(timezone.utc)
        self.started_monotonic = time.monotonic()
        self.interval_s = max(0.0, float(interval_s))
        self.last_write_monotonic = float("-inf")
        self.last_overall_percent = 0.0
        self.heartbeat_count = 0
        self.last_record: dict[str, Any] | None = None
        self.closed = False
        self._exit_handler = self._record_process_exit
        atexit.register(self._exit_handler)

    def emit(
        self,
        phase: str,
        message: str,
        processed_count: int = 0,
        total_count: int = 0,
        details: dict[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> None:
        now_monotonic = time.monotonic()
        processed = max(0, int(processed_count))
        total = max(0, int(total_count))
        if total > 0:
            processed = min(processed, total)
            fraction = float(processed / total)
        else:
            fraction = 1.0 if message in {"done", "complete"} else 0.0
        start, end = _PROGRESS_PHASE_RANGES.get(phase, (0.0, 99.0))
        overall = float(start + (end - start) * fraction)
        overall = max(self.last_overall_percent, min(100.0, overall))
        terminal = message in {"start", "done", "complete", "failed", "cancelled"}
        if not force and not terminal and now_monotonic - self.last_write_monotonic < self.interval_s:
            return
        self.heartbeat_count += 1
        now_utc = datetime.now(timezone.utc)
        elapsed = max(0.0, now_monotonic - self.started_monotonic)
        record = {
            "schema_version": "fvcom_boundary_resolution_progress_v1",
            "sequence": int(self.heartbeat_count),
            "time_utc": now_utc.isoformat(timespec="seconds"),
            "elapsed_seconds": float(elapsed),
            "phase": str(phase),
            "message": str(message),
            "processed_count": processed,
            "total_count": total,
            "phase_percent": float(100.0 * fraction),
            "overall_percent": overall,
            "details": _json_safe(details or {}),
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        state = {
            "schema_version": "fvcom_boundary_resolution_progress_state_v1",
            "started_utc": self.started_utc.isoformat(timespec="seconds"),
            "updated_utc": record["time_utc"],
            "elapsed_seconds": record["elapsed_seconds"],
            "current_phase": record["phase"],
            "current_message": record["message"],
            "processed_count": processed,
            "total_count": total,
            "phase_percent": record["phase_percent"],
            "overall_percent": overall,
            "heartbeat_count": int(self.heartbeat_count),
            "last_details": record["details"],
            "health_policy": (
                "Counts report completed topology items; percentage is monotonic across weighted scientific phases. "
                "Do not infer geometry acceptance until the resolution manifest is written."
            ),
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)
        self.last_write_monotonic = now_monotonic
        self.last_overall_percent = overall
        self.last_record = record
        if message in {"complete", "failed", "cancelled"}:
            self.closed = True
            atexit.unregister(self._exit_handler)

    def _record_process_exit(self) -> None:
        """Write one terminal failure/cancellation event when a run exits unfinished."""
        if self.closed or self.last_record is None:
            return
        previous = self.last_record
        details = dict(previous.get("details") or {})
        last_exception = getattr(sys, "last_exc", None)
        if last_exception is not None and not isinstance(last_exception, (KeyboardInterrupt, SystemExit)):
            message = "failed"
            details["failure_reason"] = "unhandled_exception"
            details["exception_type"] = type(last_exception).__name__
            details["exception_message"] = str(last_exception)
        else:
            message = "cancelled"
            details["cancellation_reason"] = (
                "keyboard_interrupt" if isinstance(last_exception, KeyboardInterrupt)
                else "process_exit_before_completion"
            )
        self.emit(
            str(previous["phase"]),
            message,
            int(previous["processed_count"]),
            int(previous["total_count"]),
            details,
            force=True,
        )

    def callback(self, phase: str) -> ProgressCallback:
        def report(processed: int, total: int, details: dict[str, Any] | None = None) -> None:
            done = total >= 0 and processed >= total
            self.emit(
                phase,
                "done" if done else "running",
                processed,
                total,
                details,
                force=done,
            )

        return report


def boundary_resolution_config(profile: str | None = None) -> BoundaryResolutionConfig:
    """Return the sole v2 configuration; reject removed generation profiles."""
    if profile not in {None, "adaptive-coastal-v2"}:
        raise ValueError("Boundary resolution only supports adaptive-coastal-v2")
    return BoundaryResolutionConfig()


def _passage_gate_taxonomy(
    passage_report: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Separate topology-critical protected passages from review-only findings."""
    failures: list[str] = []
    advisories: list[str] = []
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
    if config.profile != "adaptive-coastal-v2":
        raise ValueError("Boundary resolution only supports adaptive-coastal-v2")
    package = _load_loop_package(Path(model_boundary_loops_gpkg))
    projection = package["projection"]
    domain_xy: Polygon = package["domain_xy"]
    islands_xy: list[Polygon] = package["islands_xy"]
    mission_xy = _mission_geometry(region_bpoly_json, projection, config.mission_buffer_m)
    metrics = _island_metrics(islands_xy, domain_xy, mission_xy, config)
    return {
        "schema_version": "fvcom_boundary_resolution_analysis_v2",
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
    """Build the Adaptive v2 package without changing base boundary outputs."""
    config = config or BoundaryResolutionConfig()
    if config.profile != "adaptive-coastal-v2":
        raise ValueError("Boundary resolution only supports adaptive-coastal-v2")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress = _BoundaryResolutionProgress(run_dir, config.progress_interval_s)
    progress.emit("start", "start", 0, 1, {"name": name, "profile": config.profile}, force=True)
    source_path = Path(model_boundary_loops_gpkg)
    progress.emit("load_inputs", "start", 0, 3, {"source": str(source_path)}, force=True)
    package = _load_loop_package(source_path, model_boundary_loop_manifest)
    progress.emit(
        "load_inputs",
        "running",
        1,
        3,
        {"source_island_count": len(package["islands_xy"])},
        force=True,
    )
    projection = package["projection"]
    source_domain: Polygon = package["domain_xy"]
    islands_xy: list[Polygon] = package["islands_xy"]
    mission_xy = _mission_geometry(region_bpoly_json, projection, config.mission_buffer_m)
    progress.emit("load_inputs", "running", 2, 3, {"mission_geometry_loaded": mission_xy is not None}, force=True)
    land_union = _load_land_union(coastline_gpkg, projection)
    progress.emit(
        "load_inputs",
        "done",
        3,
        3,
        {"land_union_loaded": land_union is not None and not land_union.is_empty},
        force=True,
    )

    progress.emit("outer_boundary", "start", 0, 1, force=True)
    boundary_parts, open_lineage, repair_report, shell_polygon = _prepare_outer_boundary_parts(
        package["segments_xy"],
        source_domain,
        package.get("delivered_open_chains_xy", []),
        package.get("delivered_open_source_layer"),
        land_union,
        config,
    )
    progress.emit(
        "outer_boundary",
        "done",
        1,
        1,
        {"boundary_part_count": len(boundary_parts), "land_free": repair_report.get("land_free")},
        force=True,
    )
    open_parts = [item for item in boundary_parts if item["boundary_kind"] == "open"]
    land_parts = [item for item in boundary_parts if item["boundary_kind"] == "land"]
    progress.emit(
        "source_island_metrics",
        "start",
        0,
        len(islands_xy),
        {"island_count": len(islands_xy)},
        force=True,
    )
    source_metrics = _island_metrics(
        islands_xy,
        source_domain,
        mission_xy,
        config,
        progress=progress.callback("source_island_metrics"),
    )
    subgrid_count = int(sum(item["shape_class"] == "subgrid_fragment" for item in source_metrics))
    progress.emit(
        "subgrid_topology",
        "start",
        0,
        subgrid_count,
        {"candidate_count": subgrid_count},
        force=True,
    )
    topologized, action_report = _apply_subgrid_actions(
        shell_polygon,
        islands_xy,
        source_metrics,
        mission_xy,
        config,
        progress=progress.callback("subgrid_topology"),
    )
    generalization_count = len(topologized.interiors)
    progress.emit(
        "island_generalization",
        "start",
        0,
        generalization_count,
        {"island_count": generalization_count},
        force=True,
    )
    resolved_islands, resolved_records = _generalize_islands(
        topologized,
        mission_xy,
        config,
        progress=progress.callback("island_generalization"),
    )

    passage_report = {
        "policy": "conservative_inventory_harmonize_only_no_topology_closure",
        "passages": [],
        "passage_count": 0,
        "protected_unresolved_count": 0,
        "unprotected_unresolved_count": 0,
        "automatic_topology_operation_count": 0,
    }
    land_controls: list[dict[str, Any]] = []
    island_target_overrides: dict[int, float] = {}
    passage_domain = Polygon(shell_polygon.exterior.coords, holes=[list(poly.exterior.coords) for poly in resolved_islands])
    progress.emit(
        "passage_inventory",
        "start",
        0,
        len(land_parts) + len(resolved_islands),
        {"land_component_count": len(land_parts), "island_component_count": len(resolved_islands)},
        force=True,
    )
    passage_report, land_controls, island_target_overrides = _inventory_narrow_passages(
        [item["geometry"] for item in land_parts],
        resolved_islands,
        passage_domain,
        mission_xy,
        config,
        projection,
        progress=progress.callback("passage_inventory"),
    )
    outer_entries: list[dict[str, Any]] = []
    open_sampling: list[dict[str, Any]] = []
    land_sampling: list[dict[str, Any]] = []
    sampled_open_chains: list[dict[str, Any]] = []
    sampling_total = len(boundary_parts) + len(resolved_islands)
    sampling_processed = 0
    progress.emit(
        "boundary_sampling",
        "start",
        0,
        sampling_total,
        {"outer_part_count": len(boundary_parts), "island_count": len(resolved_islands)},
        force=True,
    )
    for part in boundary_parts:
        line = part["geometry"]
        if part["boundary_kind"] == "open":
            if part.get("is_closed"):
                nodes, targets, metadata, sampling = _sample_closed_open_loop_v2(
                    line,
                    config,
                    land_union=land_union,
                )
            else:
                nodes, targets, metadata, sampling = _sample_open_arc_v2(
                    line,
                    config,
                    land_union=land_union,
                )
            obc_id = int(part["obc_id"])
            for item in metadata:
                item["obc_id"] = obc_id
            sampling.update({"obc_id": obc_id, "is_closed": bool(part.get("is_closed"))})
            sampling["source_direction_reversed_for_exterior"] = bool(
                part.get("source_direction_reversed", False)
            )
            open_sampling.append(sampling)
            sampled_open_chains.append(
                {
                    "obc_id": obc_id,
                    "is_closed": bool(part.get("is_closed")),
                    "nodes": list(nodes),
                    "targets": list(targets),
                    "metadata": [dict(item) for item in metadata],
                    "source_geometry": line,
                    "source_direction_reversed_for_exterior": bool(
                        part.get("source_direction_reversed", False)
                    ),
                }
            )
        else:
            land_id = int(part["land_id"])
            controls = [item for item in land_controls if int(item.get("land_id", -1)) == land_id]
            nodes, targets, metadata, sampling = _sample_landward_v2(line, controls, config)
            for item in metadata:
                item["obc_id"] = None
                item["land_id"] = land_id
            sampling.update({"land_id": land_id})
            land_sampling.append(sampling)
        entries = [
            {
                "xy": xy,
                "boundary_kind": str(part["boundary_kind"]),
                "target_spacing_m": h,
                **meta,
            }
            for xy, h, meta in zip(nodes, targets, metadata)
        ]
        outer_entries.extend(entries)
        sampling_processed += 1
        progress.emit(
            "boundary_sampling",
            "running",
            sampling_processed,
            sampling_total,
            {"substage": "outer_boundary", "boundary_kind": str(part["boundary_kind"])},
        )
    outer_entries = _deduplicate_node_entries(outer_entries)
    target_gradation_conditioning = _enforce_delivered_target_gradation(
        outer_entries,
        float(config.gradation),
    )
    outer_nodes = [item["xy"] for item in outer_entries]
    outer_kinds = [str(item["boundary_kind"]) for item in outer_entries]
    outer_h = [float(item["target_spacing_m"]) for item in outer_entries]
    outer_meta = outer_entries

    island_chains: list[list[tuple[float, float]]] = []
    island_targets: list[float] = []
    for resolved_index, (record, polygon) in enumerate(zip(resolved_records, resolved_islands)):
        target = float(record["target_spacing_m"])
        if resolved_index in island_target_overrides:
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
        sampling_processed += 1
        progress.emit(
            "boundary_sampling",
            "done" if sampling_processed >= sampling_total else "running",
            sampling_processed,
            sampling_total,
            {"substage": "island", "island_index": int(resolved_index)},
            force=sampling_processed >= sampling_total,
        )

    resolved_domain = Polygon(outer_nodes, holes=island_chains)
    if not resolved_domain.is_valid:
        resolved_domain = resolved_domain.buffer(0)
    resolved_domain = _select_polygon(resolved_domain, source_domain.representative_point())
    if not isinstance(resolved_domain, Polygon) or resolved_domain.is_empty:
        raise ValueError("Resolved boundary nodes do not form a valid wet-domain polygon")

    sampled_open_geometries = [
        LineString(item["nodes"] + ([item["nodes"][0]] if item["is_closed"] and item["nodes"][0] != item["nodes"][-1] else []))
        for item in sampled_open_chains
    ]
    sampled_land_length = 0.0
    if land_union is not None and not land_union.is_empty:
        for item, sampled_open in zip(sampled_open_chains, sampled_open_geometries):
            if item["is_closed"]:
                interior = sampled_open
            else:
                endpoint_mask = Point(item["nodes"][0]).buffer(
                    max(2.0 * config.repair_sample_spacing_m, 500.0)
                ).union(
                    Point(item["nodes"][-1]).buffer(
                        max(2.0 * config.repair_sample_spacing_m, 500.0)
                    )
                )
                interior = sampled_open.difference(endpoint_mask)
            sampled_land_length += float(interior.intersection(land_union).length)
    exterior = LineString(resolved_domain.exterior.coords)
    sampled_open_length = float(sum(line.length for line in sampled_open_geometries))
    exterior_tolerance = max(0.01, 1.0e-7 * max(sampled_open_length, 1.0))
    exterior_off_length = float(
        sum(line.difference(exterior.buffer(exterior_tolerance)).length for line in sampled_open_geometries)
    )
    exterior_overlap = float(max(0.0, 1.0 - exterior_off_length / max(sampled_open_length, 1.0)))

    node_records: list[dict[str, Any]] = []
    chain_summaries: list[dict[str, Any]] = []
    _append_v2_outer_chain(node_records, chain_summaries, outer_meta, projection)
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

    progress.emit("write_outputs", "start", 0, 4, force=True)
    gpkg = run_dir / "boundary_resolution.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    _write_resolution_layers(
        gpkg,
        resolved_domain,
        [
            item.get("output_geometry", item["geometry"])
            for item in sorted(open_parts, key=lambda item: int(item["obc_id"]))
        ],
        resolved_islands,
        islands_xy,
        node_records,
        source_metrics,
        resolved_records,
        projection,
        config.profile,
        passage_report.get("passages", []),
    )
    progress.emit("write_outputs", "running", 1, 4, {"artifact": str(gpkg)}, force=True)
    diagnostics_path = run_dir / "boundary_resolution_diagnostics.json"
    node_geojson_path = run_dir / "boundary_resolution_nodes.geojson"
    review_map = run_dir / "boundary_resolution_review_map.png"
    diagnostics = {
        "schema_version": "fvcom_boundary_resolution_diagnostics_v2",
        "source_analysis": {
            "island_count": len(source_metrics),
            "class_counts": _count_by(source_metrics, "shape_class"),
            "protected_count": int(sum(bool(item["protected_mission"]) for item in source_metrics)),
        },
        "open_arc_repair": repair_report,
        "open_boundary_lineage": open_lineage,
        "topology_actions": action_report,
        "resolved_islands": resolved_records,
        "chains": chain_summaries,
    }
    diagnostics["boundary_sampling"] = {
        "profile": config.profile,
        "open": open_sampling,
        "land": land_sampling,
        "junctions": _junction_diagnostics(outer_meta, config),
        "delivered_target_gradation_conditioning": target_gradation_conditioning,
    }
    diagnostics["channel_passages"] = passage_report
    diagnostics_path.write_text(json.dumps(_json_safe(diagnostics), indent=2), encoding="utf-8")
    progress.emit("write_outputs", "running", 2, 4, {"artifact": str(diagnostics_path)}, force=True)
    node_geojson_path.write_text(json.dumps(_node_geojson(node_records), indent=2), encoding="utf-8")
    progress.emit("write_outputs", "running", 3, 4, {"artifact": str(node_geojson_path)}, force=True)
    _plot_review(review_map, source_domain, resolved_domain, [item["geometry"] for item in open_parts], mission_xy, projection, source_metrics)
    progress.emit("write_outputs", "done", 4, 4, {"artifact": str(review_map)}, force=True)

    progress.emit("quality_gates", "start", 0, 1, force=True)
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
    spacing_qa = _boundary_spacing_qa(outer_nodes, outer_h)
    hard_anchor_count = int(sum(bool(item.get("is_hard_anchor")) for item in node_records))
    landfall_hard_anchor_count = int(
        sum(item.get("anchor_type") == "open_landfall" and bool(item.get("is_hard_anchor")) for item in node_records)
    )
    open_chain_summaries = _open_chain_qa(
        sampled_open_chains,
        sampled_open_geometries,
        exterior,
        land_union,
        node_records,
        config,
    )
    open_chain_summaries.sort(key=lambda item: int(item["obc_id"]))
    delivered_obc_count = int(len(open_chain_summaries))
    expected_obc_count = _expected_obc_count(model_boundary_loop_manifest, delivered_obc_count)
    if delivered_obc_count != expected_obc_count:
        failures.append("unexpected_open_boundary_count")
    closed_chains = [item for item in open_chain_summaries if item["is_closed"]]
    if closed_chains:
        if delivered_obc_count != 1 or len(closed_chains) != 1:
            failures.append("closed_open_boundary_count_invalid")
        for item in closed_chains:
            if item["open_landfall_hard_anchor_count"] != 0:
                failures.append("closed_open_boundary_has_landfall_anchor")
            if item["open_loop_seam_hard_anchor_count"] != 1:
                failures.append("open_loop_seam_hard_anchor_count_invalid")
            if item["open_loop_balance_hard_anchor_count"] != 1:
                failures.append("open_loop_balance_hard_anchor_count_invalid")
    else:
        for item in open_chain_summaries:
            if item["open_landfall_hard_anchor_count"] != 2:
                failures.append(f"obc_{item['obc_id']}_landfall_hard_anchor_count_invalid")
    passage_failures, advisories = _passage_gate_taxonomy(passage_report)
    failures.extend(passage_failures)
    if float(spacing_qa.get("maximum_edge_to_target_ratio", 0.0)) > 1.55 + 1.0e-9:
        failures.append("boundary_edge_to_target_ratio_exceeded")
    if float(spacing_qa.get("maximum_target_gradation", 0.0)) > float(config.gradation) + 1.0e-9:
        failures.append("boundary_target_gradation_exceeded")
    manifest = {
        "schema_version": "fvcom_boundary_resolution_manifest_v2",
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
        },
        "settings": _json_safe(config.__dict__),
        "projection": {
            "crs": projection.crs.to_string(),
            "epsg": projection.epsg,
            "longitude_origin": projection.longitude_origin,
            "coordinate_policy": "native_longitudes_transformed_directly_without_longitude_warping",
        },
        "qa": {
            "open_boundary_node_count": open_count,
            "expected_obc_count": expected_obc_count,
            "delivered_obc_count": delivered_obc_count,
            "closed_obc_count": int(len(closed_chains)),
            "island_boundary_node_count": island_count,
            "total_boundary_node_count": int(len(node_records)),
            "resolved_island_count": int(len(resolved_islands)),
            "source_island_count": int(len(islands_xy)),
            "topology_absolute_area_change_fraction": topology_area_fraction,
            "protected_mission_operation_count": int(action_report["protected_operation_count"]),
            "open_arc_land_intersection_m": float(max(repair_report["land_intersection_length_m"], sampled_land_length)),
            "open_arc_exterior_overlap_fraction": exterior_overlap,
            "open_arc_source": open_lineage["source"],
            "exact_delivered_open_boundary_length_m": open_lineage.get(
                "delivered_open_boundary_length_m"
            ),
            "proximity_classified_open_boundary_length_m": open_lineage.get(
                "proximity_classified_open_boundary_length_m"
            ),
            "proximity_classified_excess_length_m": open_lineage.get(
                "proximity_classified_excess_length_m"
            ),
            "resolved_domain_valid": bool(resolved_domain.is_valid),
            **spacing_qa,
            "hard_anchor_count": hard_anchor_count,
            "open_landfall_hard_anchor_count": landfall_hard_anchor_count,
            "open_loop_seam_hard_anchor_count": int(
                sum(item["open_loop_seam_hard_anchor_count"] for item in open_chain_summaries)
            ),
            "open_loop_balance_hard_anchor_count": int(
                sum(item["open_loop_balance_hard_anchor_count"] for item in open_chain_summaries)
            ),
            "passage_count": int(passage_report["passage_count"]),
            "protected_underresolved_passage_count": int(passage_report["protected_unresolved_count"]),
            "unprotected_underresolved_passage_count": int(passage_report["unprotected_unresolved_count"]),
            "passage_minimum_spacing_policy": passage_report["minimum_spacing_policy"],
            "passage_minimum_permitted_spacing_m": float(passage_report["minimum_permitted_spacing_m"]),
            "passage_minimum_controlling_passage_id": passage_report["minimum_spacing_controlling_passage_id"],
            "minimum_protected_passage_width_m": passage_report["minimum_protected_passage_width_m"],
            "automatic_passage_topology_operation_count": int(passage_report["automatic_topology_operation_count"]),
        },
        "chains": chain_summaries,
        "open_boundary_chains": open_chain_summaries,
        "outputs": {
            "boundary_resolution_gpkg": str(gpkg),
            "boundary_resolution_diagnostics_json": str(diagnostics_path),
            "boundary_resolution_nodes_geojson": str(node_geojson_path),
            "boundary_resolution_review_map": str(review_map),
            "boundary_resolution_manifest": str(run_dir / "boundary_resolution_manifest.json"),
            "boundary_resolution_progress_jsonl": str(progress.jsonl_path),
            "boundary_resolution_progress_state": str(progress.state_path),
        },
    }
    manifest_path = run_dir / "boundary_resolution_manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    progress.emit(
        "quality_gates",
        "done",
        1,
        1,
        {"final_status": manifest["final_status"], "failure_count": len(failures)},
        force=True,
    )
    progress.emit(
        "complete",
        "complete",
        1,
        1,
        {"final_status": manifest["final_status"], "manifest": str(manifest_path)},
        force=True,
    )
    return manifest


def _load_loop_package(
    path: Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    layers = set(gpd.list_layers(path)["name"])
    domain_gdf = gpd.read_file(path, layer="model_domain_polygon")
    domain_lonlat = next(geom for geom in domain_gdf.geometry if isinstance(geom, Polygon) and not geom.is_empty)
    if domain_gdf.crs is not None:
        domain_lonlat = gpd.GeoSeries([domain_lonlat], crs=domain_gdf.crs).to_crs("EPSG:4326").iloc[0]
    manifest_file = Path(manifest_path) if manifest_path else path.with_name("model_boundary_loop_manifest.json")
    manifest = (
        json.loads(manifest_file.read_text(encoding="utf-8-sig"))
        if manifest_file.is_file()
        else {}
    )
    fallback_projection = _compact_projection(domain_lonlat)
    projection = projection_from_manifest(
        manifest,
        _compact_bbox(domain_lonlat),
    ) if manifest.get("projection") else fallback_projection
    domain_xy = _project_compact(domain_lonlat, projection).buffer(0)
    segments = gpd.read_file(path, layer="model_outer_boundary_segments").to_crs("EPSG:4326")
    segment_records = []
    for _, row in segments.sort_values("sequence_id").iterrows():
        segment_records.append(
            {
                "sequence_id": int(row.sequence_id),
                "segment_class": str(row.segment_class),
                "geometry": _project_compact(row.geometry, projection),
            }
        )
    islands_xy: list[Polygon] = []
    if "island_boundary_polygons" in layers:
        islands = gpd.read_file(path, layer="island_boundary_polygons").to_crs("EPSG:4326")
        islands_xy = [_project_compact(geom, projection).buffer(0) for geom in islands.geometry if isinstance(geom, Polygon) and not geom.is_empty]
    delivered_open_chains_xy: list[dict[str, Any]] = []
    delivered_open_source_layer = None
    for layer in ("delivered_open_boundary_arc", "source_open_boundary_arc"):
        if layer not in layers:
            continue
        delivered = gpd.read_file(path, layer=layer).to_crs("EPSG:4326")
        if "obc_id" in delivered.columns:
            delivered = delivered.sort_values("obc_id")
        next_obc_id = 0
        for _, row in delivered.iterrows():
            geom = row.geometry
            parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
            base_obc_id = int(row.obc_id) if "obc_id" in delivered.columns else next_obc_id
            for offset, part in enumerate(parts):
                if not isinstance(part, LineString) or part.is_empty:
                    continue
                obc_id = base_obc_id if len(parts) == 1 else base_obc_id + offset
                projected_part = _project_compact(part, projection)
                delivered_open_chains_xy.append(
                    {
                        "obc_id": int(obc_id),
                        "geometry": projected_part,
                        "is_closed": bool(
                            projected_part.is_ring
                            or Point(projected_part.coords[0]).distance(Point(projected_part.coords[-1])) <= 1.0
                        ),
                    }
                )
            next_obc_id = max(next_obc_id, base_obc_id + max(1, len(parts)))
        if delivered_open_chains_xy:
            delivered_open_source_layer = layer
            break
    delivered_open_chains_xy.sort(key=lambda item: int(item["obc_id"]))
    for expected_id, item in enumerate(delivered_open_chains_xy):
        item["obc_id"] = int(expected_id)
    return {
        "projection": projection,
        "domain_xy": _select_polygon(domain_xy, domain_xy.representative_point()),
        "segments_xy": segment_records,
        "islands_xy": islands_xy,
        "delivered_open_chains_xy": delivered_open_chains_xy,
        "delivered_open_source_layer": delivered_open_source_layer,
    }


def _compact_projection(domain_lonlat: Polygon):
    """Choose a local projection from the minimum-span circular longitude frame."""
    longitudes = sorted(
        set((float(lon) % 360.0) for lon, _ in domain_lonlat.exterior.coords[:-1])
    )
    if not longitudes:
        raise ValueError("Model domain has no usable exterior coordinates")
    if len(longitudes) == 1:
        origin = ((longitudes[0] + 180.0) % 360.0) - 180.0
        span = 0.0
    else:
        gaps = [
            (
                (longitudes[(index + 1) % len(longitudes)] - longitudes[index]) % 360.0,
                index,
            )
            for index in range(len(longitudes))
        ]
        _, gap_index = max(gaps, key=lambda item: (item[0], -item[1]))
        start = longitudes[(gap_index + 1) % len(longitudes)]
        span = (longitudes[gap_index] - start) % 360.0
        origin = ((start + 0.5 * span + 180.0) % 360.0) - 180.0
    lats = [float(lat) for _lon, lat in domain_lonlat.exterior.coords]
    west = _wrap_lon(origin - 0.5 * span)
    east = _wrap_lon(origin + 0.5 * span)
    return local_utm_projection((west, min(lats), east, max(lats)))


def _compact_bbox(domain_lonlat: Polygon) -> tuple[float, float, float, float]:
    projection = _compact_projection(domain_lonlat)
    origin = float(projection.longitude_origin or 0.0)
    longitudes = [float(lon) for lon, _lat in domain_lonlat.exterior.coords[:-1]]
    circular = [((lon - origin + 180.0) % 360.0) - 180.0 for lon in longitudes]
    lats = [float(lat) for _lon, lat in domain_lonlat.exterior.coords]
    west = _wrap_lon(origin + min(circular))
    east = _wrap_lon(origin + max(circular))
    return west, min(lats), east, max(lats)


def _wrap_lon(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def _project_compact(geometry, projection):
    return project_geometry(geometry, projection)


def _prepare_outer_boundary_parts(
    records: list[dict[str, Any]],
    domain: Polygon,
    delivered_chains: list[dict[str, Any]],
    delivered_source_layer: str | None,
    land_union,
    config: BoundaryResolutionConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], Polygon]:
    """Return cyclic open/land parts without collapsing distinct OBC chains."""
    exterior = LineString(domain.exterior.coords)
    tolerance_m = max(5.0, 1.0e-6 * float(exterior.length))
    classified = _classified_open_chains(records, domain)
    source_chains = delivered_chains or classified
    if not source_chains:
        raise ValueError("No delivered open-boundary chain is available for Adaptive v2")
    if any(bool(item.get("is_closed")) for item in source_chains):
        if len(source_chains) != 1:
            raise ValueError("A closed offshore OBC cannot be mixed with other OBC chains")
        source = source_chains[0]["geometry"]
        overlap = _line_overlap_fraction(source, exterior, tolerance_m)
        if overlap < 1.0 - 1.0e-9 or abs(float(source.length) - float(exterior.length)) > tolerance_m:
            raise ValueError("Closed delivered OBC does not match the model exterior")
        closed = _rotate_ring_to_canonical_seam(exterior)
        land_intersection = float(
            closed.intersection(land_union).length
            if land_union is not None and not land_union.is_empty
            else 0.0
        )
        repair = {
            "method": "closed_offshore_loop_no_landfall_repair",
            "chain_count": 1,
            "land_free": bool(land_intersection <= 1.0e-6),
            "anchors_preserved": True,
            "land_intersection_length_m": land_intersection,
            "chains": [
                {
                    "obc_id": 0,
                    "is_closed": True,
                    "land_free": bool(land_intersection <= 1.0e-6),
                    "anchors_preserved": True,
                    "land_intersection_length_m": land_intersection,
                }
            ],
        }
        lineage = {
            "source": "exact_delivered_open_boundary_arc" if delivered_chains else "classified_closed_exterior_loop",
            "source_layer": delivered_source_layer or "model_outer_boundary_segments",
            "exact_delivered_geometry_available": bool(delivered_chains),
            "delivered_obc_count": 1,
            "delivered_open_boundary_length_m": float(closed.length),
            "proximity_classified_open_boundary_length_m": float(sum(item["geometry"].length for item in classified)),
            "proximity_classified_excess_length_m": float(
                max(0.0, sum(item["geometry"].length for item in classified) - float(closed.length))
            ),
            "exterior_overlap_fraction": overlap,
            "chains": [{"obc_id": 0, "is_closed": True, "length_m": float(closed.length)}],
        }
        return ([{"boundary_kind": "open", "obc_id": 0, "is_closed": True, "geometry": closed}], lineage, repair, domain)

    prepared: list[dict[str, Any]] = []
    for fallback_id, item in enumerate(source_chains):
        obc_id = int(item.get("obc_id", fallback_id))
        normalized, endpoint_snap = _normalize_open_chain_endpoints_on_exterior(
            item["geometry"],
            exterior,
            max(tolerance_m, float(config.repair_sample_spacing_m)),
        )
        oriented, start_m, end_m, overlap, reversed_from_source = _orient_open_chain_on_exterior(
            normalized, exterior, tolerance_m
        )
        repaired, report = _repair_open_arc(oriented, domain, land_union, config)
        report["delivered_endpoint_normalization"] = endpoint_snap
        prepared.append(
            {
                "boundary_kind": "open",
                "obc_id": obc_id,
                "is_closed": False,
                "geometry": repaired,
                "output_geometry": (
                    LineString(list(repaired.coords)[::-1]) if reversed_from_source else repaired
                ),
                "source_direction_reversed": bool(reversed_from_source),
                "source_geometry": oriented,
                "start_m": start_m,
                "end_m": end_m,
                "wraps": bool(end_m < start_m),
                "overlap": overlap,
                "repair": report,
            }
        )
    prepared.sort(key=lambda item: float(item["start_m"]))
    wrap_indices = [idx for idx, item in enumerate(prepared) if item["wraps"]]
    if len(wrap_indices) > 1:
        raise ValueError("Delivered OBC intervals overlap across the exterior seam")
    if wrap_indices:
        pivot = wrap_indices[0]
        prepared = prepared[pivot:] + prepared[:pivot]
    for first_index, first in enumerate(prepared):
        for second in prepared[first_index + 1 :]:
            overlap_length = float(
                first["source_geometry"].intersection(second["source_geometry"].buffer(tolerance_m)).length
            )
            if overlap_length > 2.0 * tolerance_m:
                raise ValueError("Delivered OBC intervals overlap on the model exterior")

    parts: list[dict[str, Any]] = []
    land_id = 0
    for index, item in enumerate(prepared):
        parts.append({key: value for key, value in item.items() if key not in {"start_m", "end_m", "wraps", "overlap", "repair", "source_geometry"}})
        following = prepared[(index + 1) % len(prepared)]
        gap = _cyclic_exterior_substring(exterior, float(item["end_m"]), float(following["start_m"]))
        if float(gap.length) > tolerance_m:
            parts.append(
                {
                    "boundary_kind": "land",
                    "land_id": int(land_id),
                    "is_closed": False,
                    "geometry": gap,
                }
            )
            land_id += 1
    shell_coords = _ordered_part_coords([item["geometry"] for item in parts])
    shell_polygon = Polygon(shell_coords)
    if not shell_polygon.is_valid:
        shell_polygon = shell_polygon.buffer(0)
    if not isinstance(shell_polygon, Polygon) or shell_polygon.is_empty:
        raise ValueError("Adaptive v2 OBC repair did not produce a valid exterior polygon")
    repair_records = [
        {"obc_id": int(item["obc_id"]), "is_closed": False, **_json_safe(item["repair"])}
        for item in prepared
    ]
    repair = {
        "method": "per_obc_fixed_landfall_water_side_repair",
        "chain_count": int(len(prepared)),
        "land_free": bool(all(item.get("land_free") for item in repair_records)),
        "anchors_preserved": bool(all(item.get("anchors_preserved") for item in repair_records)),
        "land_intersection_length_m": float(sum(float(item.get("land_intersection_length_m", 0.0)) for item in repair_records)),
        "chains": repair_records,
    }
    delivered_length = float(sum(item["geometry"].length for item in prepared))
    classified_length = float(sum(item["geometry"].length for item in classified))
    lineage = {
        "source": "exact_delivered_open_boundary_arc" if delivered_chains else "proximity_classified_model_outer_boundary_segments",
        "source_layer": delivered_source_layer or "model_outer_boundary_segments",
        "exact_delivered_geometry_available": bool(delivered_chains),
        "delivered_obc_count": int(len(prepared)),
        "delivered_open_boundary_length_m": delivered_length,
        "proximity_classified_open_boundary_length_m": classified_length,
        "proximity_classified_excess_length_m": float(max(0.0, classified_length - delivered_length)),
        "exterior_overlap_fraction": float(
            sum(item["overlap"] * item["geometry"].length for item in prepared) / max(delivered_length, 1.0)
        ),
        "chains": [
            {
                "obc_id": int(item["obc_id"]),
                "is_closed": False,
                "length_m": float(item["geometry"].length),
                "exterior_overlap_fraction": float(item["overlap"]),
            }
            for item in prepared
        ],
    }
    return parts, lineage, repair, shell_polygon


def _classified_open_chains(records: list[dict[str, Any]], domain: Polygon) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: int(item["sequence_id"]))
    flags = [item["segment_class"] == "open_boundary" for item in ordered]
    if not any(flags):
        return []
    if all(flags):
        return [{"obc_id": 0, "is_closed": True, "geometry": LineString(domain.exterior.coords)}]
    pivot = next(index for index, flag in enumerate(flags) if not flag)
    traversal = list(range(pivot + 1, len(ordered))) + list(range(0, pivot + 1))
    groups: list[list[int]] = []
    current: list[int] = []
    for index in traversal:
        if flags[index]:
            current.append(index)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return [
        {
            "obc_id": int(obc_id),
            "is_closed": False,
            "geometry": LineString(_ordered_segment_coords([ordered[index]["geometry"] for index in group])),
        }
        for obc_id, group in enumerate(groups)
    ]


def _orient_open_chain_on_exterior(
    line: LineString,
    exterior: LineString,
    tolerance_m: float,
) -> tuple[LineString, float, float, float, bool]:
    if not line.is_simple or line.is_ring or len(line.coords) < 2:
        raise ValueError("A coastal delivered OBC must be one simple nonclosed LineString")
    overlap = _line_overlap_fraction(line, exterior, tolerance_m)
    if overlap < 1.0 - 1.0e-9:
        raise ValueError("Delivered OBC is not completely on the model exterior")
    start_m = float(exterior.project(Point(line.coords[0])))
    end_m = float(exterior.project(Point(line.coords[-1])))
    forward = _cyclic_exterior_substring(exterior, start_m, end_m)
    reverse = LineString(list(_cyclic_exterior_substring(exterior, end_m, start_m).coords)[::-1])
    forward_match = _line_overlap_fraction(line, forward, tolerance_m)
    reverse_match = _line_overlap_fraction(line, reverse, tolerance_m)
    if max(forward_match, reverse_match) < 1.0 - 1.0e-9:
        raise ValueError("Delivered OBC endpoints do not isolate one exterior interval")
    if forward_match >= reverse_match:
        return line, start_m, end_m, overlap, False
    return LineString(list(line.coords)[::-1]), end_m, start_m, overlap, True


def _normalize_open_chain_endpoints_on_exterior(
    line: LineString,
    exterior: LineString,
    snap_limit_m: float,
) -> tuple[LineString, dict[str, Any]]:
    """Snap only QA-bounded OBC endpoints to the canonical model exterior.

    The open-exterior contract permits a physical landfall endpoint to be a
    small distance from the polygonized model exterior. Adaptive v2 samples
    the canonical exterior, so normalize those endpoint-only offsets before
    enforcing complete-chain overlap. Interior deviations remain untouched
    and are still rejected by ``_orient_open_chain_on_exterior``.
    """
    if not isinstance(line, LineString) or line.is_empty or len(line.coords) < 2:
        raise ValueError("A delivered OBC requires at least two coordinates")
    coordinates = [(float(x), float(y)) for x, y in line.coords]
    distances = [
        float(Point(coordinates[0]).distance(exterior)),
        float(Point(coordinates[-1]).distance(exterior)),
    ]
    limit = max(0.0, float(snap_limit_m))
    if max(distances) > limit + 1.0e-9:
        raise ValueError(
            "Delivered OBC endpoint is too far from the model exterior: "
            f"maximum {max(distances):.3f} m exceeds {limit:.3f} m"
        )
    for index in (0, -1):
        point = Point(coordinates[index])
        projected = exterior.interpolate(exterior.project(point))
        coordinates[index] = (float(projected.x), float(projected.y))
    normalized = LineString(coordinates)
    return normalized, {
        "method": "bounded_endpoint_projection_to_model_exterior",
        "endpoint_snap_distance_m": distances,
        "maximum_endpoint_snap_distance_m": float(max(distances)),
        "endpoint_snap_limit_m": limit,
        "normalized": bool(max(distances) > 1.0e-8),
        "interior_coordinates_changed": False,
    }


def _rotate_ring_to_canonical_seam(line: LineString) -> LineString:
    coords = [(float(x), float(y)) for x, y in list(line.coords)[:-1]]
    if len(coords) < 3:
        raise ValueError("Closed offshore OBC requires at least three distinct vertices")
    seam_index = min(range(len(coords)), key=lambda index: (coords[index][0], coords[index][1], index))
    rotated = coords[seam_index:] + coords[:seam_index]
    return LineString(rotated + [rotated[0]])


def _ordered_part_coords(lines: list[LineString]) -> list[tuple[float, float]]:
    join_tolerance_m = 5.0
    coords: list[tuple[float, float]] = []
    for line in lines:
        part = [(float(x), float(y)) for x, y in line.coords]
        if not coords:
            coords.extend(part)
        elif np.linalg.norm(np.asarray(coords[-1]) - np.asarray(part[0])) <= join_tolerance_m:
            part[0] = coords[-1]
            coords.extend(part[1:])
        else:
            raise ValueError("Adaptive v2 exterior parts are not continuously ordered")
    if coords:
        closure_gap = np.linalg.norm(np.asarray(coords[0]) - np.asarray(coords[-1]))
        if closure_gap <= join_tolerance_m:
            coords[-1] = coords[0]
        else:
            raise ValueError("Adaptive v2 exterior parts do not close continuously")
    return coords


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
            polygons.append(project_geometry_densified(Polygon(geometry["coordinates"][0]), projection))
        elif isinstance(geometry, (list, tuple)) and len(geometry) == 4:
            poly = box(*map(float, geometry))
            polygons.append(project_geometry_densified(poly, projection))
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
    return unary_union([_project_compact(geom, projection) for geom in gdf.geometry if geom is not None and not geom.is_empty])


def _canonical_open_and_landward(
    records: list[dict[str, Any]],
    domain: Polygon,
    delivered_open: LineString | None = None,
    delivered_source_layer: str | None = None,
) -> tuple[LineString, LineString, dict[str, Any]]:
    classified_open, classified_landward = _classified_open_and_landward(records, domain)
    classified_length = float(classified_open.length)
    if delivered_open is None or delivered_open.is_empty:
        return classified_open, classified_landward, {
            "source": "proximity_classified_model_outer_boundary_segments",
            "source_layer": "model_outer_boundary_segments",
            "exact_delivered_geometry_available": False,
            "delivered_open_boundary_length_m": classified_length,
            "proximity_classified_open_boundary_length_m": classified_length,
            "proximity_classified_excess_length_m": 0.0,
            "exterior_overlap_fraction": 1.0,
        }
    exact_open, exact_landward, exact_report = _exact_delivered_open_and_landward(
        delivered_open,
        domain,
    )
    exact_report.update(
        {
            "source": "exact_delivered_open_boundary_arc",
            "source_layer": delivered_source_layer,
            "exact_delivered_geometry_available": True,
            "delivered_open_boundary_length_m": float(exact_open.length),
            "proximity_classified_open_boundary_length_m": classified_length,
            "proximity_classified_excess_length_m": float(
                max(0.0, classified_length - float(exact_open.length))
            ),
        }
    )
    return exact_open, exact_landward, exact_report


def _classified_open_and_landward(
    records: list[dict[str, Any]],
    domain: Polygon,
) -> tuple[LineString, LineString]:
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


def _exact_delivered_open_and_landward(
    delivered_open: LineString,
    domain: Polygon,
) -> tuple[LineString, LineString, dict[str, Any]]:
    """Pair an exact delivered OBC with the complementary domain exterior."""
    if not delivered_open.is_simple or len(delivered_open.coords) < 2:
        raise ValueError("Exact delivered open-boundary geometry is not a simple LineString")
    exterior = LineString(domain.exterior.coords)
    tolerance_m = max(5.0, 1.0e-6 * float(exterior.length))
    exterior_overlap = float(
        delivered_open.intersection(exterior.buffer(tolerance_m)).length
        / max(float(delivered_open.length), 1.0)
    )
    if exterior_overlap < 1.0 - 1.0e-9:
        raise ValueError("Exact delivered open-boundary geometry is not on the model exterior")

    if delivered_open.is_ring:
        if abs(float(delivered_open.length) - float(exterior.length)) > tolerance_m:
            raise ValueError("Closed delivered open-boundary geometry does not match the model exterior")
        junction = delivered_open.coords[0]
        return delivered_open, LineString([junction, junction]), {
            "exterior_overlap_fraction": exterior_overlap,
            "matched_exterior_interval_fraction": 1.0,
            "matched_exterior_direction": "closed_exterior_loop",
            "landward_boundary_length_m": 0.0,
            "start_junction_gap_m": 0.0,
            "end_junction_gap_m": 0.0,
        }

    start_distance = float(exterior.project(Point(delivered_open.coords[0])))
    end_distance = float(exterior.project(Point(delivered_open.coords[-1])))
    forward = _cyclic_exterior_substring(exterior, start_distance, end_distance)
    backward = _cyclic_exterior_substring(exterior, end_distance, start_distance)
    reverse_backward = LineString(list(backward.coords)[::-1])
    forward_match = _line_overlap_fraction(delivered_open, forward, tolerance_m)
    backward_match = _line_overlap_fraction(delivered_open, reverse_backward, tolerance_m)
    if max(forward_match, backward_match) < 1.0 - 1.0e-9:
        raise ValueError("Exact delivered open-boundary anchors do not isolate one exterior interval")

    if forward_match >= backward_match:
        landward = backward
        matched_direction = "exterior_forward"
        matched_fraction = forward_match
    else:
        landward = LineString(list(forward.coords)[::-1])
        matched_direction = "exterior_reverse"
        matched_fraction = backward_match
    land_coords = list(landward.coords)
    land_coords[0] = delivered_open.coords[-1]
    land_coords[-1] = delivered_open.coords[0]
    landward = LineString(land_coords)
    return delivered_open, landward, {
        "exterior_overlap_fraction": exterior_overlap,
        "matched_exterior_interval_fraction": float(matched_fraction),
        "matched_exterior_direction": matched_direction,
        "landward_boundary_length_m": float(landward.length),
        "start_junction_gap_m": float(
            Point(delivered_open.coords[-1]).distance(Point(landward.coords[0]))
        ),
        "end_junction_gap_m": float(
            Point(landward.coords[-1]).distance(Point(delivered_open.coords[0]))
        ),
    }


def _cyclic_exterior_substring(
    exterior: LineString,
    start_distance: float,
    end_distance: float,
) -> LineString:
    if end_distance >= start_distance:
        return substring(exterior, start_distance, end_distance)
    first = substring(exterior, start_distance, exterior.length)
    second = substring(exterior, 0.0, end_distance)
    return LineString(list(first.coords) + list(second.coords)[1:])


def _line_overlap_fraction(line: LineString, reference: LineString, tolerance_m: float) -> float:
    return float(
        line.intersection(reference.buffer(tolerance_m)).length
        / max(float(line.length), 1.0)
    )


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


def _island_metrics(
    islands: list[Polygon],
    domain: Polygon,
    mission,
    config: BoundaryResolutionConfig,
    *,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
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
        if progress is not None:
            progress(
                idx + 1,
                len(cleaned_islands),
                {"island_id": int(idx), "shape_class": shape_class},
            )
    if progress is not None and not cleaned_islands:
        progress(0, 0, {"island_count": 0})
    return results


def _apply_subgrid_actions(
    shell: Polygon,
    islands: list[Polygon],
    metrics: list[dict[str, Any]],
    mission,
    config: BoundaryResolutionConfig,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[Polygon, dict[str, Any]]:
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
    for candidate_position, item in enumerate(candidates):
        idx = int(item["island_id"])
        if progress is not None:
            progress(
                candidate_position,
                len(candidates),
                {"next_island_id": idx, "accepted_action_count": len(actions)},
            )
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

    if progress is not None:
        progress(
            len(candidates),
            len(candidates),
            {"accepted_action_count": len(actions)},
        )

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


def _generalize_islands(
    domain: Polygon,
    mission,
    config: BoundaryResolutionConfig,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[list[Polygon], list[dict[str, Any]]]:
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
        if progress is not None:
            progress(
                idx + 1,
                len(islands),
                {"island_id": int(idx), "protected_mission": protected},
            )
    if progress is not None and not islands:
        progress(0, 0, {"island_count": 0})
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
    landward: list[LineString] | LineString,
    islands: list[Polygon],
    domain: Polygon,
    mission,
    config: BoundaryResolutionConfig,
    projection,
    *,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, float]]:
    """Conservatively inventory wet connectors between nonlocal boundary banks.

    The inventory may lower paired sampling targets, but it never closes a
    channel or changes geographic topology. Ambiguous or unresolvable cases
    are retained and exposed as review gates.
    """
    max_width = float(getattr(config, "passage_max_width_m", 5000.0))
    search_spacing = float(getattr(config, "passage_search_spacing_m", 300.0))
    min_along = float(getattr(config, "passage_min_along_separation_m", 1500.0))
    raw_candidates: list[dict[str, Any]] = []
    land_lines = [landward] if isinstance(landward, LineString) else list(landward)
    land_lines = [line for line in land_lines if isinstance(line, LineString) and not line.is_empty and line.length > 1.0]
    # These exact cover geometries depend only on the accepted domain.  Large
    # archipelagos make buffering the many-hole polygon expensive, so prepare
    # each one once instead of rebuilding it for every connector sample.
    connector_domain_cover = prep(domain.buffer(2.0))
    connector_sample_cover = prep(domain.buffer(0.25))

    # Same-chain search captures opposite banks of a narrow inlet/channel.
    for land_id, land_line in enumerate(land_lines):
        sample_count = min(1200, max(3, int(math.ceil(float(land_line.length) / max(search_spacing, 1.0))) + 1))
        sample_s = np.linspace(0.0, float(land_line.length), sample_count)
        sample_xy = np.asarray([[land_line.interpolate(float(s)).x, land_line.interpolate(float(s)).y] for s in sample_s], dtype=float)
        sample_tangent = np.asarray(
            [_line_tangent_at(land_line, float(s), search_spacing) for s in sample_s],
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
                    connector = LineString([sample_xy[first], sample_xy[second]])
                    if not _wet_connector_is_conservative(
                        connector,
                        domain,
                        buffered_domain_cover=connector_domain_cover,
                        sample_domain_cover=connector_sample_cover,
                    ):
                        continue
                    raw_candidates.append(
                        {
                            "bank_a": "land",
                            "bank_b": "land",
                            "land_a": int(land_id),
                            "land_b": int(land_id),
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
    components: list[tuple[str, int | None, Any]] = [
        ("land", index, line) for index, line in enumerate(land_lines)
    ] + [
        ("island", index, LineString(polygon.exterior.coords)) for index, polygon in enumerate(islands)
    ]
    component_geometries = [item[2] for item in components]
    # Use envelopes for the broad phase so GEOS does not repeatedly evaluate
    # exact distances between highly detailed whole-island coastlines.  An
    # axis-aligned envelope expanded by max_width contains every geometry
    # whose exact distance can be <= max_width; the unchanged nearest-points
    # and wet-connector checks below reject the conservative false positives.
    component_envelopes = [geometry.envelope for geometry in component_geometries]
    component_tree = STRtree(component_envelopes)
    indexed_component_pair_count = 0
    all_component_pair_count = len(components) * max(0, len(components) - 1) // 2
    if progress is not None:
        progress(
            0,
            len(component_geometries),
            {"substage": "cross_component", "raw_candidate_count": len(raw_candidates)},
        )
    for first, geometry_a in enumerate(component_geometries):
        min_x, min_y, max_x, max_y = geometry_a.bounds
        search_envelope = box(
            min_x - max_width,
            min_y - max_width,
            max_x + max_width,
            max_y + max_width,
        )
        nearby = component_tree.query(search_envelope)
        for second in sorted(int(value) for value in nearby if int(value) > first):
            indexed_component_pair_count += 1
            kind_a, component_a, geometry_a = components[first]
            kind_b, component_b, geometry_b = components[second]
            point_a, point_b = nearest_points(geometry_a, geometry_b)
            width = float(point_a.distance(point_b))
            if not (1.0 < width <= max_width):
                continue
            connector = LineString([point_a, point_b])
            if not _wet_connector_is_conservative(
                connector,
                domain,
                buffered_domain_cover=connector_domain_cover,
                sample_domain_cover=connector_sample_cover,
            ):
                continue
            raw_candidates.append(
                {
                    "bank_a": kind_a,
                    "bank_b": kind_b,
                    "land_a": int(component_a) if kind_a == "land" and component_a is not None else None,
                    "land_b": int(component_b) if kind_b == "land" and component_b is not None else None,
                    "position_a_m": float(geometry_a.project(point_a)),
                    "position_b_m": float(geometry_b.project(point_b)),
                    "island_a": int(component_a) if kind_a == "island" and component_a is not None else None,
                    "island_b": int(component_b) if kind_b == "island" and component_b is not None else None,
                    "width_m": width,
                    "connector": connector,
                }
            )
        if progress is not None:
            progress(
                first + 1,
                len(component_geometries),
                {
                    "substage": "cross_component",
                    "indexed_component_pair_count": indexed_component_pair_count,
                    "raw_candidate_count": len(raw_candidates),
                },
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

    passage_candidates: list[dict[str, Any]] = []
    for passage_id, candidate in enumerate(accepted):
        connector = candidate["connector"]
        protected = bool(mission is not None and not mission.is_empty and connector.intersects(mission))
        elements = int(
            getattr(config, "protected_elements_across", 4)
            if protected
            else getattr(config, "unprotected_elements_across", 3)
        )
        required_h = float(candidate["width_m"] / max(elements, 1))
        passage_candidates.append(
            {
                **candidate,
                "passage_id": int(passage_id),
                "protected_mission": protected,
                "required_elements_across": elements,
                "required_target_spacing_m": required_h,
            }
        )

    configured_min_spacing = getattr(config, "passage_min_spacing_m", None)
    controlling_passage: dict[str, Any] | None = None
    if configured_min_spacing is not None:
        min_spacing = float(configured_min_spacing)
        if not math.isfinite(min_spacing) or min_spacing <= 0.0:
            raise ValueError("passage_min_spacing_m must be positive and finite when explicitly configured")
        min_spacing_policy = "explicit_configuration"
    else:
        protected_candidates = [item for item in passage_candidates if item["protected_mission"]]
        if protected_candidates:
            controlling_passage = min(
                protected_candidates,
                key=lambda item: (
                    float(item["required_target_spacing_m"]),
                    float(item["width_m"]),
                    int(item["passage_id"]),
                ),
            )
            min_spacing = float(controlling_passage["required_target_spacing_m"])
            min_spacing_policy = "adaptive_from_minimum_protected_passage_width"
        else:
            min_spacing = float(config.land_spacing_m)
            min_spacing_policy = "configured_land_spacing_no_protected_passage"

    passages: list[dict[str, Any]] = []
    land_controls: list[dict[str, Any]] = []
    island_targets: dict[int, float] = {}
    protected_unresolved = 0
    unprotected_unresolved = 0
    for candidate in passage_candidates:
        passage_id = int(candidate["passage_id"])
        connector = candidate["connector"]
        protected = bool(candidate["protected_mission"])
        elements = int(candidate["required_elements_across"])
        required_h = float(candidate["required_target_spacing_m"])
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
                        "land_id": int(candidate.get("land_a", 0) or 0),
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
                        "land_id": int(candidate.get("land_b", 0) or 0),
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
                **{key: value for key, value in candidate.items() if key != "connector"},
                "protected_mission": protected,
                "required_elements_across": elements,
                "required_target_spacing_m": required_h,
                "minimum_permitted_spacing_m": min_spacing,
                "minimum_spacing_policy": min_spacing_policy,
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
            "minimum_spacing_policy": min_spacing_policy,
            "configured_minimum_spacing_override_m": (
                float(configured_min_spacing) if configured_min_spacing is not None else None
            ),
            "minimum_spacing_controlling_passage_id": (
                int(controlling_passage["passage_id"]) if controlling_passage is not None else None
            ),
            "minimum_protected_passage_width_m": (
                float(controlling_passage["width_m"]) if controlling_passage is not None else None
            ),
            "minimum_protected_passage_required_spacing_m": (
                float(controlling_passage["required_target_spacing_m"])
                if controlling_passage is not None
                else None
            ),
            "component_pair_index_policy": "expanded_envelope_broad_phase_then_exact_distance_and_wet_connector",
            "wet_connector_domain_buffer_policy": "exact_domain_buffers_prepared_once_per_inventory",
            "all_component_pair_count": int(all_component_pair_count),
            "spatially_indexed_component_pair_count": int(indexed_component_pair_count),
            "passages": passages,
        },
        land_controls,
        island_targets,
    )


def _wet_connector_is_conservative(
    connector: LineString,
    domain: Polygon,
    *,
    buffered_domain_cover=None,
    sample_domain_cover=None,
) -> bool:
    if connector.is_empty or connector.length <= 1.0:
        return False
    buffered_cover = buffered_domain_cover or prep(domain.buffer(2.0))
    if not buffered_cover.covers(connector):
        return False
    sample_cover = sample_domain_cover or prep(domain.buffer(0.25))
    for fraction in np.linspace(0.1, 0.9, 9):
        if not sample_cover.covers(connector.interpolate(float(fraction), normalized=True)):
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
    *,
    land_union=None,
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
    positions, land_safety = _refine_sampling_positions_for_land_safety(
        line,
        positions,
        land_union,
        is_closed=False,
        endpoint_exclusion_m=max(2.0 * config.repair_sample_spacing_m, 500.0),
    )
    coords, sizes, metadata = _sample_records(line, positions, anchors, target, "open")
    return coords, sizes, metadata, {
        "method": "anchor_interval_metric_equidistribution_with_chord_and_land_safety_guards",
        "source_length_m": length,
        "node_count": len(coords),
        "feature_anchor_count": int(sum(item["anchor_type"] != "open_landfall" for item in anchors)),
        "hard_anchor_count": int(len(anchors)),
        "land_safety_refinement": land_safety,
        "anchors": anchors,
    }


def _sample_closed_open_loop_v2(
    line: LineString,
    config: BoundaryResolutionConfig,
    *,
    land_union=None,
) -> tuple[list[tuple[float, float]], list[float], list[dict[str, Any]], dict[str, Any]]:
    """Sample a cyclic island/archipelago OBC with deterministic numerical anchors."""
    if not line.is_ring:
        raise ValueError("Closed-loop sampling requires an exact ring")
    length = float(line.length)
    balance_m = 0.5 * length

    def cyclic_distance(first: float, second: float) -> float:
        direct = abs(float(first) - float(second))
        return min(direct, max(0.0, length - direct))

    def target(position: float) -> float:
        distance = min(cyclic_distance(position, 0.0), cyclic_distance(position, balance_m))
        return float(
            min(
                config.open_central_spacing_m,
                config.open_anchor_spacing_m + config.gradation * distance,
            )
        )

    feature_anchors = [
        item
        for item in _stable_feature_anchors(line, target, config)
        if 1.0e-6 < float(item["source_position_m"]) < length - 1.0e-6
    ]
    anchors = [
        {
            "source_position_m": 0.0,
            "anchor_type": "open_loop_seam",
            "anchor_id": "open_loop_seam_0000",
            "source_vertex_index": 0,
            "local_target_spacing_m": float(target(0.0)),
        },
        {
            "source_position_m": balance_m,
            "anchor_type": "open_loop_balance",
            "anchor_id": "open_loop_balance_0000",
            "source_vertex_index": None,
            "local_target_spacing_m": float(target(balance_m)),
        },
        *feature_anchors,
        {
            "source_position_m": length,
            "anchor_type": "open_loop_seam",
            "anchor_id": "open_loop_seam_0000",
            "source_vertex_index": len(line.coords) - 1,
            "local_target_spacing_m": float(target(length)),
        },
    ]
    anchors.sort(key=lambda item: float(item["source_position_m"]))
    positions = _equidistributed_positions(line, anchors, target)
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
    positions, land_safety = _refine_sampling_positions_for_land_safety(
        line,
        positions,
        land_union,
        is_closed=True,
        endpoint_exclusion_m=0.0,
    )
    coords, sizes, metadata = _sample_records(line, positions, anchors, target, "open_loop")
    return coords, sizes, metadata, {
        "method": "cyclic_seam_balance_metric_equidistribution_with_chord_and_land_safety_guards",
        "source_length_m": length,
        "node_count_including_periodic_closure": len(coords),
        "hard_anchor_count": 2,
        "open_landfall_hard_anchor_count": 0,
        "open_loop_seam_hard_anchor_count": 1,
        "open_loop_balance_hard_anchor_count": 1,
        "canonical_seam_xy": list(coords[0]),
        "balance_position_m": balance_m,
        "land_safety_refinement": land_safety,
        "anchors": anchors,
    }


def _refine_sampling_positions_for_land_safety(
    line: LineString,
    positions: list[float],
    land_union,
    *,
    is_closed: bool,
    endpoint_exclusion_m: float,
    maximum_iterations: int = 24,
) -> tuple[list[float], dict[str, Any]]:
    """Bisect only sampled chords that shortcut through physical land.

    The accepted source OBC remains authoritative.  Refinement inserts points
    interpolated on that exact source chain until every delivered chord is
    land-free.  Coastal landfall neighborhoods use the same exclusion radius
    as the downstream nonendpoint-land QA; closed offshore loops have no such
    exception.
    """
    ordered = sorted(set(float(value) for value in positions))
    initial_count = len(ordered)
    if land_union is None or land_union.is_empty or len(ordered) < 2:
        return ordered, {
            "applied": False,
            "iteration_count": 0,
            "added_node_count": 0,
            "remaining_unsafe_chord_count": 0,
            "remaining_land_intersection_m": 0.0,
            "endpoint_exclusion_m": float(0.0 if is_closed else endpoint_exclusion_m),
        }

    endpoint_mask = None
    if not is_closed and endpoint_exclusion_m > 0.0:
        endpoint_mask = Point(line.coords[0]).buffer(float(endpoint_exclusion_m)).union(
            Point(line.coords[-1]).buffer(float(endpoint_exclusion_m))
        )
    prepared_land = prep(land_union)

    def unsafe_intervals(values: list[float]) -> tuple[list[float], float, int]:
        additions: list[float] = []
        intersection_length = 0.0
        unresolved = 0
        for start, end in zip(values[:-1], values[1:]):
            chord = LineString([line.interpolate(start), line.interpolate(end)])
            interior = chord if endpoint_mask is None else chord.difference(endpoint_mask)
            if interior.is_empty or not prepared_land.intersects(interior):
                continue
            overlap = float(interior.intersection(land_union).length)
            if overlap <= 1.0e-6:
                continue
            intersection_length += overlap
            midpoint = 0.5 * (float(start) + float(end))
            if end - start > 1.0e-6 and midpoint not in values:
                additions.append(midpoint)
            else:
                unresolved += 1
        return additions, float(intersection_length), int(unresolved)

    iteration_count = 0
    for iteration in range(int(maximum_iterations)):
        additions, _intersection_length, _unresolved = unsafe_intervals(ordered)
        if not additions:
            break
        ordered = sorted(set(ordered + additions))
        iteration_count = iteration + 1
    remaining, remaining_length, unresolved = unsafe_intervals(ordered)
    return ordered, {
        "applied": True,
        "iteration_count": int(iteration_count),
        "added_node_count": int(len(ordered) - initial_count),
        "remaining_unsafe_chord_count": int(len(remaining) + unresolved),
        "remaining_land_intersection_m": float(remaining_length),
        "endpoint_exclusion_m": float(0.0 if is_closed else endpoint_exclusion_m),
        "maximum_iterations": int(maximum_iterations),
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
            if out[-1].get("obc_id") is None and item.get("obc_id") is not None:
                out[-1]["obc_id"] = int(item["obc_id"])
                out[-1]["boundary_kind"] = "open"
            if not out[-1].get("anchor_type"):
                out[-1]["anchor_type"] = item.get("anchor_type", "")
                out[-1]["anchor_id"] = item.get("anchor_id", "")
            out[-1]["target_spacing_m"] = min(float(out[-1]["target_spacing_m"]), float(item["target_spacing_m"]))
            continue
        out.append(item)
    if len(out) > 1 and np.linalg.norm(np.asarray(out[0]["xy"]) - np.asarray(out[-1]["xy"])) <= 1.0e-7:
        out[0]["is_hard_anchor"] = bool(out[0].get("is_hard_anchor") or out[-1].get("is_hard_anchor"))
        if out[0].get("obc_id") is None and out[-1].get("obc_id") is not None:
            out[0]["obc_id"] = int(out[-1]["obc_id"])
            out[0]["boundary_kind"] = "open"
        if not out[0].get("anchor_type"):
            out[0]["anchor_type"] = out[-1].get("anchor_type", "")
            out[0]["anchor_id"] = out[-1].get("anchor_id", "")
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
                "obc_id": int(item["obc_id"]) if item.get("obc_id") is not None else None,
                "land_id": int(item["land_id"]) if item.get("land_id") is not None else None,
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
            "open_boundary_chain_count": int(len({item.get("obc_id") for item in entries if item.get("obc_id") is not None})),
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


def _open_chain_spacing_qa(
    coords: list[tuple[float, float]],
    sizes: list[float],
    closed: bool,
) -> dict[str, Any]:
    points = np.asarray(coords, dtype=float)
    target = np.asarray(sizes, dtype=float)
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= 1.0e-7:
        points = points[:-1]
        target = target[:-1]
    if len(points) < 2:
        return {
            "maximum_edge_to_target_ratio": 0.0,
            "p95_edge_to_target_ratio": 0.0,
            "maximum_target_gradation": 0.0,
        }
    if closed:
        end = np.roll(points, -1, axis=0)
        following_target = np.roll(target, -1)
    else:
        end = points[1:]
        points = points[:-1]
        following_target = target[1:]
        target = target[:-1]
    lengths = np.linalg.norm(end - points, axis=1)
    ratios = lengths / np.maximum(np.minimum(target, following_target), 1.0)
    gradation = np.abs(following_target - target) / np.maximum(lengths, 1.0)
    return {
        "maximum_edge_to_target_ratio": float(np.max(ratios)),
        "p95_edge_to_target_ratio": float(np.percentile(ratios, 95.0)),
        "maximum_target_gradation": float(np.max(gradation)),
    }


def _open_chain_qa(
    sampled: list[dict[str, Any]],
    geometries: list[LineString],
    exterior: LineString,
    land_union,
    node_records: list[dict[str, Any]],
    config: BoundaryResolutionConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item, geometry in zip(sampled, geometries):
        metadata = list(item["metadata"])
        anchor_ids: dict[str, dict[str, Any]] = {}
        for record in metadata:
            if record.get("is_hard_anchor") and record.get("anchor_id"):
                anchor_ids[str(record["anchor_id"])] = record
        tolerance = max(0.01, 1.0e-7 * max(float(geometry.length), 1.0))
        off_length = float(geometry.difference(exterior.buffer(tolerance)).length)
        overlap = float(max(0.0, 1.0 - off_length / max(float(geometry.length), 1.0)))
        if land_union is None or land_union.is_empty:
            land_intersection = 0.0
        elif item["is_closed"]:
            land_intersection = float(geometry.intersection(land_union).length)
        else:
            endpoints = Point(geometry.coords[0]).buffer(
                max(2.0 * config.repair_sample_spacing_m, 500.0)
            ).union(
                Point(geometry.coords[-1]).buffer(
                    max(2.0 * config.repair_sample_spacing_m, 500.0)
                )
            )
            land_intersection = float(geometry.difference(endpoints).intersection(land_union).length)
        obc_id = int(item["obc_id"])
        node_indices = [
            int(record["node_index_zero_based"])
            for record in node_records
            if record.get("obc_id") == obc_id
        ]
        if item.get("source_direction_reversed_for_exterior"):
            node_indices = list(reversed(node_indices))
        summary = {
            "obc_id": obc_id,
            "is_closed": bool(item["is_closed"]),
            "source_direction_reversed_for_exterior": bool(
                item.get("source_direction_reversed_for_exterior", False)
            ),
            "node_count": int(len(node_indices)),
            "node_sequence_zero_based": node_indices,
            "source_length_m": float(item["source_geometry"].length),
            "sampled_length_m": float(geometry.length),
            "exterior_overlap_fraction": overlap,
            "nonendpoint_land_intersection_m": land_intersection,
            "hard_anchor_count": int(len(anchor_ids)),
            "open_landfall_hard_anchor_count": int(
                sum(record.get("anchor_type") == "open_landfall" for record in anchor_ids.values())
            ),
            "open_loop_seam_hard_anchor_count": int(
                sum(record.get("anchor_type") == "open_loop_seam" for record in anchor_ids.values())
            ),
            "open_loop_balance_hard_anchor_count": int(
                sum(record.get("anchor_type") == "open_loop_balance" for record in anchor_ids.values())
            ),
            **_open_chain_spacing_qa(item["nodes"], item["targets"], bool(item["is_closed"])),
        }
        output.append(summary)
    return output


def _expected_obc_count(manifest_path: str | Path | None, delivered: int) -> int:
    if not manifest_path or not Path(manifest_path).is_file():
        return int(delivered)
    document = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    value = document.get("qa", {}).get("expected_obc_count")
    if value is None:
        value = document.get("open_boundary_lineage", {}).get("expected_obc_count")
    return int(delivered if value is None else value)


def _enforce_delivered_target_gradation(
    entries: list[dict[str, Any]],
    gradation: float,
) -> dict[str, Any]:
    """Project targets exactly onto cyclic chord-length Lipschitz constraints."""
    if len(entries) < 2 or gradation <= 0.0:
        return {"adjusted_node_count": 0, "iteration_count": 0, "maximum_gradient": 0.0}
    points = np.asarray([item["xy"] for item in entries], dtype=float)
    raw = np.asarray([item["target_spacing_m"] for item in entries], dtype=float)
    effective_gradation = float(gradation) * (1.0 - 1.0e-4)
    fixed = np.asarray(
        [item.get("anchor_type") in {"open_landfall", "open_loop_seam", "open_loop_balance"} for item in entries],
        dtype=bool,
    )
    lengths = np.maximum(
        np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1),
        1.0,
    )
    edge_costs = effective_gradation * lengths
    count = len(entries)

    # Fixed-anchor cones provide exact feasible lower/upper bounds on a cycle.
    # Clipping raw targets to those bounds prevents the shortest-path closure
    # below from moving a hard anchor. Incompatible fixed anchors are detected
    # directly instead of being hidden by an iteration limit.
    lower = np.full(count, 1.0, dtype=float)
    upper = np.full(count, np.inf, dtype=float)
    positions = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    perimeter = float(np.sum(lengths))
    fixed_indices = np.flatnonzero(fixed)
    for anchor_index in fixed_indices:
        delta = np.abs(positions - positions[int(anchor_index)])
        cycle_distance = np.minimum(delta, perimeter - delta)
        lower = np.maximum(
            lower,
            float(raw[int(anchor_index)]) - effective_gradation * cycle_distance,
        )
        upper = np.minimum(
            upper,
            float(raw[int(anchor_index)]) + effective_gradation * cycle_distance,
        )
    if np.any(lower > upper + 1.0e-8):
        raise ValueError("Fixed Adaptive v2 anchors cannot satisfy delivered boundary gradation")

    target = np.maximum(np.minimum(raw, upper), lower)
    target[fixed] = raw[fixed]

    # Multi-source Dijkstra computes min_j(target_j + g*d_cycle(i,j)), the
    # greatest graph-Lipschitz minorant of the clipped targets. This is exact
    # for the cyclic boundary graph and converges independently of node count.
    queue = [(float(value), int(index)) for index, value in enumerate(target)]
    heapq.heapify(queue)
    relaxation_count = 0
    while queue:
        value, index = heapq.heappop(queue)
        if value > float(target[index]) + 1.0e-10:
            continue
        neighbors = (
            ((index - 1) % count, float(edge_costs[(index - 1) % count])),
            ((index + 1) % count, float(edge_costs[index])),
        )
        for neighbor, cost in neighbors:
            candidate = value + cost
            if candidate >= float(target[neighbor]) - 1.0e-10:
                continue
            if fixed[neighbor]:
                if candidate < float(raw[neighbor]) - 1.0e-8:
                    raise ValueError("Fixed Adaptive v2 anchors cannot satisfy delivered boundary gradation")
                continue
            target[neighbor] = candidate
            relaxation_count += 1
            heapq.heappush(queue, (candidate, neighbor))

    if np.any(np.abs(target[fixed] - raw[fixed]) > 1.0e-8):
        raise ValueError("Adaptive v2 target-gradation projection moved a fixed anchor")
    gradients = np.abs(np.roll(target, -1) - target) / lengths
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
        "method": "anchor_preserving_cycle_shortest_path_lipschitz_projection",
        "requested_gradation": float(gradation),
        "effective_projection_gradation": effective_gradation,
        "fixed_anchor_count": int(np.count_nonzero(fixed)),
        "adjusted_node_count": int(np.count_nonzero(adjusted)),
        "maximum_adjustment_m": float(np.max(np.abs(target - raw))),
        "iteration_count": 1,
        "relaxation_count": int(relaxation_count),
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
    open_lines,
    islands,
    source_islands,
    node_records,
    source_metrics,
    resolved_records,
    projection,
    profile: str = "adaptive-coastal-v2",
    passages: list[dict[str, Any]] | None = None,
) -> None:
    domain_ll = unproject_geometry(domain, projection)
    gpd.GeoDataFrame([{"profile": profile, "geometry": domain_ll}], geometry="geometry", crs="EPSG:4326").to_file(gpkg, layer="resolved_domain_polygon", driver="GPKG")
    open_rows = [
        {
            "segment_class": "open_boundary",
            "obc_id": int(obc_id),
            "is_closed": bool(line.is_ring),
            "geometry": unproject_geometry(line, projection),
        }
        for obc_id, line in enumerate(open_lines)
    ]
    gpd.GeoDataFrame(open_rows, geometry="geometry", crs="EPSG:4326").to_file(
        gpkg, layer="resolved_open_boundary", driver="GPKG"
    )
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


def _plot_review(path, source_domain, resolved_domain, open_lines, mission, projection, metrics) -> None:
    native_source = unproject_geometry(source_domain, projection)
    minx, _miny, maxx, _maxy = native_source.bounds
    projected_display = float(maxx - minx) > 180.0
    display_crs = projection.crs if projected_display else "EPSG:4326"
    source_display = source_domain if projected_display else native_source
    resolved_display = resolved_domain if projected_display else unproject_geometry(resolved_domain, projection)
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    gpd.GeoSeries([source_display], crs=display_crs).boundary.plot(ax=ax, color="#9aa0a6", linewidth=0.5, label="base")
    gpd.GeoSeries([resolved_display], crs=display_crs).boundary.plot(ax=ax, color="#16537e", linewidth=0.8, label="resolved")
    if open_lines:
        gpd.GeoSeries(
            open_lines if projected_display else [unproject_geometry(line, projection) for line in open_lines],
            crs=display_crs,
        ).plot(ax=ax, color="#d00000", linewidth=2.0, label="resolved OBC")
    if mission is not None and not mission.is_empty:
        mission_display = mission if projected_display else unproject_geometry(mission, projection)
        gpd.GeoSeries([mission_display], crs=display_crs).boundary.plot(ax=ax, color="#7b2cbf", linewidth=0.8, linestyle="--", label="protected mission")
    ax.set_title(f"Adaptive coastal boundary resolution: {len(metrics)} source islands")
    ax.set_xlabel("Projected easting (m)" if projected_display else "Longitude")
    ax.set_ylabel("Projected northing (m)" if projected_display else "Latitude")
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
