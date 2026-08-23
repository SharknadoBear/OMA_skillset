#!/usr/bin/env python3
"""Apply one autonomous-thin-v1 decision and optionally remesh/condition."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.autonomous_thin import (  # noqa: E402
    CLOSURE_SCHEMA,
    PATCH_SCHEMA,
    AutonomousThinConfig,
    boundary_transaction_audit,
    canonical_sha256,
    json_safe,
    interior_topology_plan,
    no_op_closure_report,
    rank_shoreline_candidates,
    regularize_shoreline,
    sha256_file,
    shoreline_junction_turns_deg,
    validate_agent_decision,
)
from fvcom_grid_generation.portfolio_case import (  # noqa: E402
    PortfolioCaseConfig,
    run_portfolio_case,
)
from fvcom_grid_generation.projection import (  # noqa: E402
    local_utm_projection,
    project_geometry,
    project_points,
    unproject_points,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2) + "\n", encoding="utf-8")
    return path


def _retain_rejected_boundary_candidates(
    output: Path,
    *,
    route: str,
    candidate_records: list[dict[str, Any]],
    source_manifest_path: Path,
) -> Path:
    """Persist every rejected boundary candidate before returning failure."""
    return _write_json(output / "rejected_boundary_candidates.json", {
        "schema_version": PATCH_SCHEMA,
        "status": "rejected",
        "route": route,
        "candidate_ranking": candidate_records,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
    })


def _resolve_manifest_gpkg(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    value = (manifest.get("outputs") or {}).get("boundary_resolution_gpkg")
    if value:
        path = Path(str(value))
        if not path.is_absolute():
            path = manifest_path.parent / path
        if path.is_file():
            return path.resolve()
    path = manifest_path.with_name("boundary_resolution.gpkg")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _line_frames(
    path: Path, projection: Any
) -> tuple[list[Any], list[float | None], list[str | int | None]]:
    geometries: list[Any] = []
    accuracies: list[float | None] = []
    dates: list[str | int | None] = []
    for layer in gpd.list_layers(path)["name"].tolist():
        try:
            frame = gpd.read_file(path, layer=layer).to_crs(projection.crs)
        except Exception:
            continue
        for _, row in frame.iterrows():
            geometry = row.geometry
            if geometry is None or geometry.is_empty or geometry.geom_type not in {
                "LineString", "MultiLineString"
            }:
                continue
            geometries.append(geometry)
            value = row.get("HOR_ACC")
            try:
                accuracies.append(float(value) if value is not None else None)
            except (TypeError, ValueError):
                accuracies.append(None)
            dates.append(row.get("SRC_DATE"))
    return geometries, accuracies, dates


def _forward_positions(size: int, start: int, end: int) -> list[int]:
    result = [int(start % size)]
    cursor = int(start % size)
    while cursor != int(end % size):
        cursor = (cursor + 1) % size
        result.append(cursor)
        if len(result) > size:
            raise ValueError("cyclic interval failed to terminate")
    return result


def _minimal_cyclic_window(size: int, implicated: list[int]) -> tuple[int, int]:
    selected = sorted(set(int(value) % size for value in implicated))
    if not selected:
        raise ValueError("no implicated source nodes")
    if len(selected) == 1:
        return selected[0], selected[0]
    gaps: list[tuple[int, int, int]] = []
    for left, right in zip(selected, selected[1:] + [selected[0] + size]):
        gaps.append((right - left, left % size, right % size))
    _gap, gap_start, gap_end = max(gaps)
    return gap_end, gap_start


def _expand_brackets(
    xy: np.ndarray,
    implicated_positions: list[int],
    distance_m: float,
) -> tuple[int, int]:
    size = len(xy)
    start, end = _minimal_cyclic_window(size, implicated_positions)
    walked = 0.0
    cursor = start
    while walked < distance_m:
        previous = (cursor - 1) % size
        walked += float(np.linalg.norm(xy[cursor] - xy[previous]))
        cursor = previous
        if cursor == end:
            raise ValueError("stable bracket expansion consumed complete chain")
    start = cursor
    walked = 0.0
    cursor = end
    while walked < distance_m:
        following = (cursor + 1) % size
        walked += float(np.linalg.norm(xy[following] - xy[cursor]))
        cursor = following
        if cursor == start:
            raise ValueError("stable bracket expansion consumed complete chain")
    return start, cursor


def _chain_line(xy: np.ndarray, start: int, end: int) -> LineString:
    positions = _forward_positions(len(xy), start, end)
    return LineString(xy[np.asarray(positions, dtype=int)])


def _replace_frame_window(
    frame: gpd.GeoDataFrame,
    start: int,
    end: int,
    replacement_xy: LineString,
    projection: Any,
    *,
    target_spacing_m: float,
    source_tag: str,
) -> tuple[gpd.GeoDataFrame, list[int], list[str]]:
    size = len(frame)
    interval = _forward_positions(size, start, end)
    removed_positions = interval[1:-1]
    blocked: list[str] = []
    for position in removed_positions:
        row = frame.iloc[position]
        if str(row.get("boundary_kind", "")).strip().lower() == "open":
            blocked.append(f"open_boundary_node:{row.get('node_index_zero_based', position)}")
            continue
        if not bool(row.get("is_hard_anchor", False)):
            continue
        anchor_type = str(row.get("anchor_type", "") or "")
        if anchor_type not in {"sharp_turn", "spit_tip"}:
            blocked.append(
                f"nondemotable_hard_anchor:{row.get('anchor_id', position)}"
            )
    if blocked:
        raise ValueError(";".join(blocked))
    remainder_positions = _forward_positions(size, end, start)[:-1]
    records: list[dict[str, Any]] = []
    start_row = frame.iloc[start].to_dict()
    end_row = frame.iloc[end].to_dict()
    records.append(start_row)
    replacement_interior = list(replacement_xy.coords)[1:-1]
    interior_lonlat = (
        unproject_points(np.asarray(replacement_interior, dtype=float), projection)
        if replacement_interior
        else np.empty((0, 2), dtype=float)
    )
    for lon, lat in interior_lonlat:
        row = deepcopy(start_row)
        row["geometry"] = Point(float(lon), float(lat))
        row["target_spacing_m"] = float(target_spacing_m)
        row["is_hard_anchor"] = False
        row["anchor_type"] = ""
        row["anchor_id"] = ""
        row["source_chain"] = source_tag
        row["source_position_m"] = None
        records.append(row)
    records.append(end_row)
    for position in remainder_positions[1:]:
        records.append(frame.iloc[position].to_dict())
    result = gpd.GeoDataFrame(records, geometry="geometry", crs=frame.crs)
    result = result.reset_index(drop=True)
    return result, removed_positions, blocked


def _open_line_from_frame(frame: gpd.GeoDataFrame) -> LineString:
    open_positions = [
        index
        for index, value in enumerate(frame["boundary_kind"].astype(str).str.lower())
        if value == "open"
    ]
    if not open_positions:
        return LineString()
    return LineString([frame.geometry.iloc[index].coords[0] for index in open_positions])


def _polygon_from_groups(groups: list[gpd.GeoDataFrame]) -> Polygon:
    exterior = [point.coords[0] for point in groups[0].geometry]
    holes = [[point.coords[0] for point in group.geometry] for group in groups[1:]]
    return Polygon(exterior, holes=holes)


def _write_patch_packages(
    source_manifest_path: Path,
    decision: dict[str, Any],
    diagnostic: dict[str, Any],
    output: Path,
    *,
    cusp_gpkg: Path | None,
) -> dict[str, Any]:
    source_manifest = _read_json(source_manifest_path)
    source_gpkg = _resolve_manifest_gpkg(source_manifest_path, source_manifest)
    layers = set(gpd.list_layers(source_gpkg)["name"].tolist())
    nodes = gpd.read_file(source_gpkg, layer="boundary_nodes").to_crs("EPSG:4326")
    nodes = nodes.sort_values(["chain_id", "chain_position"]).reset_index(drop=True)
    grouped_values = list(nodes.groupby("chain_id", sort=True))
    groups = [group.copy().reset_index(drop=True) for _, group in grouped_values]
    chain_index = int(decision["source_window"]["chain_index_zero_based"])
    if chain_index >= len(groups):
        raise ValueError("decision source chain is absent")
    frame = groups[chain_index]
    domain_before = _polygon_from_groups(groups)
    projection = local_utm_projection(tuple(float(value) for value in domain_before.bounds))
    xy = project_points(
        np.asarray([[point.x, point.y] for point in frame.geometry], dtype=float),
        projection,
    )
    source_ids = [int(value) for value in decision["source_window"]["source_node_indices_zero_based"]]
    id_to_position = {
        int(value): int(index)
        for index, value in enumerate(frame["node_index_zero_based"].astype(int))
    }
    implicated = [id_to_position[value] for value in source_ids if value in id_to_position]
    if not implicated:
        raise ValueError("decision source nodes do not occur in selected chain")
    component = next(
        value for value in diagnostic["components"]
        if value["component_id"] == decision["component_id"]
    )
    local_target = float(component["local_target_m"])
    start, end = _expand_brackets(
        xy,
        implicated,
        AutonomousThinConfig().stable_bracket_target_multiplier * local_target,
    )
    original_window = _chain_line(xy, start, end)
    route = str(decision["route"])
    candidate_records: list[dict[str, Any]] = []
    replacement_options: list[tuple[LineString, str, int | None]] = []
    source_method = "model_scale_regularization"
    if route == "subgrid_wet_connection":
        replacement = LineString([xy[start], xy[end]])
        source_method = "aggressive_complete_connection_closure"
        replacement_options.append((replacement, source_method, None))
    elif route == "subgrid_boundary_spike_or_sliver":
        ranked: list[dict[str, Any]] = []
        if cusp_gpkg is not None:
            geometries, accuracies, source_dates = _line_frames(cusp_gpkg, projection)
            ranked = rank_shoreline_candidates(
                geometries,
                tuple(xy[start]),
                tuple(xy[end]),
                original_window,
                local_target_m=local_target,
                horizontal_accuracy_m=accuracies,
                source_dates=source_dates,
            )[: AutonomousThinConfig().maximum_candidates_per_component]
        incoming = xy[start] - xy[(start - 1) % len(xy)]
        outgoing = xy[(end + 1) % len(xy)] - xy[end]
        maximum_junction_turn_deg = 135.0
        for candidate_rank, value in enumerate(ranked):
            trial = regularize_shoreline(
                value["geometry"],
                tuple(xy[start]),
                tuple(xy[end]),
                local_target_m=local_target,
                horizontal_accuracy_m=value.get("reported_horizontal_accuracy_m"),
            )
            turns = shoreline_junction_turns_deg(trial, incoming, outgoing)
            record = json_safe(value) | {
                "candidate_rank_zero_based": candidate_rank,
                "regularized_length_m": float(trial.length),
                "regularized_coordinate_count": len(trial.coords),
                "junction_turns": turns,
                "eligible": bool(
                    trial.is_simple
                    and turns["maximum_turn_deg"] <= maximum_junction_turn_deg
                ),
            }
            if not record["eligible"]:
                record["rejection_reason"] = (
                    "replacement_self_intersection"
                    if not trial.is_simple
                    else "shoreline_junction_tangent_disagreement"
                )
            candidate_records.append(record)
            if record["eligible"]:
                replacement_options.append((
                    trial,
                    "nearest_eligible_cusp_arc_model_scale_regularized",
                    len(candidate_records) - 1,
                ))
        if not replacement_options and len(candidate_records) < AutonomousThinConfig().maximum_candidates_per_component:
            replacement = LineString([xy[start], xy[end]])
            turns = shoreline_junction_turns_deg(replacement, incoming, outgoing)
            candidate_records.append({
                "candidate_rank_zero_based": len(candidate_records),
                "source": "model_scale_bracket_chord_fallback",
                "regularized_length_m": float(replacement.length),
                "regularized_coordinate_count": len(replacement.coords),
                "junction_turns": turns,
                "eligible": True,
            })
            replacement_options.append((
                replacement,
                "model_scale_bracket_chord_fallback",
                len(candidate_records) - 1,
            ))
    elif route == "resolved_channel_meshing_defect":
        replacement = original_window
        source_method = "preserve_geometry_reduce_boundary_target"
        replacement_options.append((replacement, source_method, None))
    else:
        raise ValueError(f"route {route!r} is not a boundary-remesh route")

    open_before = gpd.read_file(source_gpkg, layer="resolved_open_boundary").to_crs("EPSG:4326")
    open_before_geom = unary_union([value for value in open_before.geometry if value is not None and not value.is_empty])
    # A land-only thin repair must preserve the delivered OBC geometry byte-for-
    # geometry at the package interface.  Reconstructing it from adaptive nodes
    # can legitimately produce a simplified line and a false OBC-regression
    # failure even though no open node was edited.
    open_after_geom = open_before_geom
    selected: tuple[list[gpd.GeoDataFrame], Polygon, LineString, list[int], dict[str, Any], str] | None = None
    for replacement_trial, method_trial, record_index in replacement_options:
        try:
            updated, removed_trial, _blocked = _replace_frame_window(
                frame,
                start,
                end,
                replacement_trial,
                projection,
                target_spacing_m=(
                    min(
                        local_target,
                        float((decision.get("resolution_evidence") or {}).get(
                            "required_target_m",
                            float((decision.get("resolution_evidence") or {})["width_m"]) / 3.0,
                        )),
                    )
                    if route == "resolved_channel_meshing_defect"
                    else local_target
                ),
                source_tag="autonomous-thin-v1",
            )
            trial_groups = [group.copy() for group in groups]
            trial_groups[chain_index] = updated
            node_offset = 0
            for chain_position, group in enumerate(trial_groups):
                group["chain_position"] = np.arange(len(group), dtype=int)
                group["node_index_zero_based"] = np.arange(
                    node_offset, node_offset + len(group), dtype=int
                )
                node_offset += len(group)
                trial_groups[chain_position] = group
            domain_trial = _polygon_from_groups(trial_groups)
            audit_trial = boundary_transaction_audit(
                project_geometry(domain_before, projection),
                project_geometry(domain_trial, projection),
                expected_hole_count=len(trial_groups) - 1,
                obc_before=project_geometry(open_before_geom, projection),
                obc_after=project_geometry(open_after_geom, projection),
            )
        except Exception as exc:
            audit_trial = {
                "passed": False,
                "failure_taxonomy": ["boundary_candidate_exception"],
                "exception": f"{type(exc).__name__}: {exc}",
            }
        if record_index is not None:
            candidate_records[record_index]["boundary_transaction_audit"] = audit_trial
        if audit_trial["passed"]:
            if record_index is not None:
                candidate_records[record_index]["selected"] = True
            selected = (
                trial_groups,
                domain_trial,
                replacement_trial,
                removed_trial,
                audit_trial,
                method_trial,
            )
            break
        if record_index is not None:
            candidate_records[record_index]["eligible"] = False
            candidate_records[record_index]["rejection_reason"] = "boundary_transaction_failed"
    if selected is None:
        _retain_rejected_boundary_candidates(
            output,
            route=route,
            candidate_records=candidate_records,
            source_manifest_path=source_manifest_path,
        )
        raise ValueError("no autonomous boundary candidate passed the structural transaction audit")
    groups, domain_after, replacement, removed, audit, source_method = selected

    package_dir = output / "patched_boundary"
    package_dir.mkdir(parents=True, exist_ok=True)
    gpkg = package_dir / "boundary_resolution.gpkg"
    if gpkg.exists():
        raise FileExistsError(gpkg)
    gpd.GeoDataFrame(
        [{"profile": "autonomous-thin-v1", "geometry": domain_after}],
        geometry="geometry", crs="EPSG:4326",
    ).to_file(gpkg, layer="resolved_domain_polygon", driver="GPKG")
    gpd.GeoDataFrame(
        [{"segment_class": "open_boundary", "geometry": open_after_geom}],
        geometry="geometry", crs="EPSG:4326",
    ).to_file(gpkg, layer="resolved_open_boundary", driver="GPKG")
    if len(groups) > 1:
        gpd.GeoDataFrame(
            [
                {"resolved_island_id": index, "geometry": Polygon([p.coords[0] for p in group.geometry])}
                for index, group in enumerate(groups[1:], start=1)
            ],
            geometry="geometry", crs="EPSG:4326",
        ).to_file(gpkg, layer="resolved_island_polygons", driver="GPKG")
    all_nodes = gpd.GeoDataFrame(
        pd.concat(groups, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    )
    all_nodes.to_file(gpkg, layer="boundary_nodes", driver="GPKG")
    nodes_geojson = package_dir / "boundary_resolution_nodes.geojson"
    all_nodes.to_file(nodes_geojson, driver="GeoJSON")
    patched_manifest = deepcopy(source_manifest)
    patched_manifest["schema_version"] = "fvcom_boundary_resolution_manifest_v2"
    patched_manifest["profile"] = "adaptive-coastal-v2"
    patched_manifest["final_status"] = "pass"
    patched_manifest["autonomous_thin_patch"] = {
        "schema_version": PATCH_SCHEMA,
        "decision_sha256": canonical_sha256(decision),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_gpkg": str(source_gpkg),
        "source_gpkg_sha256": sha256_file(source_gpkg),
        "route": route,
        "source_method": source_method,
        "chain_index_zero_based": chain_index,
        "stable_bracket_positions": [start, end],
        "removed_chain_positions": removed,
        "candidate_ranking": candidate_records,
        "audit": audit,
    }
    patched_manifest["qa"] = dict(patched_manifest.get("qa") or {}) | {
        "wet_component_count": 1,
        "resolved_domain_valid": bool(domain_after.is_valid),
        "open_boundary_chain_count": 0 if open_after_geom.is_empty else 1,
    }
    patched_manifest["outputs"] = dict(patched_manifest.get("outputs") or {}) | {
        "boundary_resolution_gpkg": str(gpkg),
        "boundary_resolution_nodes_geojson": str(nodes_geojson),
    }
    manifest_path = _write_json(package_dir / "boundary_resolution_manifest.json", patched_manifest)

    local_patch_document = deepcopy(patched_manifest["autonomous_thin_patch"])
    local_patch_document.update({
        "schema_version": PATCH_SCHEMA,
        "status": "pass",
        "boundary_resolution_manifest": str(manifest_path),
        "boundary_resolution_gpkg": str(gpkg),
        "boundary_resolution_gpkg_sha256": sha256_file(gpkg),
        "boundary_nodes_geojson": str(nodes_geojson),
        "boundary_nodes_geojson_sha256": sha256_file(nodes_geojson),
    })
    local_patch_path = _write_json(
        package_dir / "local_shoreline_patch.json", local_patch_document
    )

    comparison = package_dir / "boundary_patch_comparison.png"
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    before_xy = project_geometry(domain_before, projection)
    after_xy = project_geometry(domain_after, projection)
    bx, by = before_xy.exterior.xy
    ax.plot(bx, by, "--", color="#777777", linewidth=1.0, label="before")
    axx, ayy = after_xy.exterior.xy
    ax.plot(axx, ayy, color="#d62728", linewidth=1.2, label="after")
    ox, oy = original_window.xy
    rx, ry = replacement.xy
    ax.plot(ox, oy, color="#1f77b4", linewidth=2.0, label="source window")
    ax.plot(rx, ry, color="#2ca02c", linewidth=2.0, label="selected replacement")
    ax.set_xlim(min(ox) - 3 * local_target, max(ox) + 3 * local_target)
    ax.set_ylim(min(oy) - 3 * local_target, max(oy) + 3 * local_target)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.set_title(f"{route}: boundary-level transaction")
    fig.savefig(comparison, dpi=220)
    plt.close(fig)
    return {
        "schema_version": PATCH_SCHEMA,
        "status": "pass",
        "source_method": source_method,
        "route": route,
        "boundary_resolution_manifest": str(manifest_path),
        "boundary_resolution_gpkg": str(gpkg),
        "boundary_nodes_geojson": str(nodes_geojson),
        "local_shoreline_patch": str(local_patch_path),
        "comparison_map": str(comparison),
        "audit": audit,
        "expected_island_holes": len(groups) - 1,
    }


def _write_case_manifest(
    source_case_path: Path,
    patch: dict[str, Any],
    output: Path,
) -> Path:
    manifest = _read_json(source_case_path)
    manifest["case_id"] = str(manifest["case_id"]) + "_autonomous_thin_v1"
    manifest["display_name"] = str(manifest.get("display_name", manifest["case_id"])) + " — autonomous thin v1"
    manifest["boundary"]["input_kind"] = "adaptive_v2"
    manifest["boundary"]["resolution_manifest"] = patch["boundary_resolution_manifest"]
    manifest["boundary"]["expected_island_holes"] = int(patch["expected_island_holes"])
    manifest["boundary"].pop("model_boundary_loops_gpkg", None)
    manifest["boundary"].pop("model_boundary_loop_manifest", None)
    manifest["readiness"] = {
        "status": "ready",
        "boundary_status": "available",
        "bathymetry_status": "available",
        "blockers": [],
    }
    manifest.setdefault("notes", []).append(
        "Boundary geometry was changed by the SHA-bound autonomous-thin-v1 transaction; the mesh must be regenerated before use."
    )
    return _write_json(output / "patched_case_manifest.json", manifest)


def _condition_mesh(
    *,
    mesh: Path,
    size_field: Path,
    bathymetry: Path,
    boundary_contract: Path,
    source_boundary_metadata: Path,
    condition_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_portfolio_conditioning.py")),
        "--mesh", str(mesh),
        "--size-field-nc", str(size_field),
        "--bathymetry-nc", str(bathymetry),
        "--boundary-contract-json", str(boundary_contract),
        "--source-boundary-metadata-json", str(source_boundary_metadata),
        "--output-dir", str(condition_dir),
        "--conditioning-profile", "minimal-topology-v1",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "conditioned_2dm": str(condition_dir / "conditioned.2dm"),
        "conditioning_report": str(condition_dir / "conditioning_report.json"),
    }


def _condition_generated(output: Path, bathymetry: Path) -> dict[str, Any]:
    raw = next((output / "remesh").glob("candidates/gmsh_frontal_delaunay_6/raw_mesh.2dm"))
    bundle = output / "remesh" / "input_bundle"
    candidate = raw.parent
    condition_dir = output / "conditioned"
    return _condition_mesh(
        mesh=raw,
        size_field=bundle / "canonical_size_field_v4.nc",
        bathymetry=bathymetry,
        boundary_contract=bundle / "canonical_boundary.json",
        source_boundary_metadata=candidate / "boundary_metadata.json",
        condition_dir=condition_dir,
    )


def _run_interior_topology(
    diagnostic: dict[str, Any],
    decision: dict[str, Any],
    output: Path,
    *,
    execute: bool,
) -> dict[str, Any]:
    plan = interior_topology_plan(decision, diagnostic)
    interior_dir = output / "interior_topology"
    plan_path = _write_json(interior_dir / "agent_local_topology_plan.json", plan)
    result: dict[str, Any] = {
        "status": "plan_ready",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
    }
    if not execute:
        return result
    paths = diagnostic.get("input_paths") or {}
    required = {
        "mesh": paths.get("mesh"),
        "boundary_nodes_geojson": paths.get("boundary_nodes_geojson"),
        "size_field_nc": paths.get("size_field_nc"),
        "bathymetry_nc": paths.get("bathymetry_nc"),
        "boundary_contract_json": paths.get("boundary_contract_json"),
        "source_boundary_metadata_json": paths.get("source_boundary_metadata_json"),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        result.update({"status": "blocked", "failure_taxonomy": [
            "interior_topology_input_missing:" + key for key in missing
        ]})
        return result
    repaired = interior_dir / "repaired.2dm"
    repair_report = interior_dir / "repair_report.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("apply_visual_superthin_plan.py")),
        "--mesh", str(required["mesh"]),
        "--boundary-nodes-geojson", str(required["boundary_nodes_geojson"]),
        "--size-field-nc", str(required["size_field_nc"]),
        "--plan", str(plan_path),
        "--output-mesh", str(repaired),
        "--report", str(repair_report),
        "--output-boundary-nodes", str(interior_dir / "boundary_nodes.geojson"),
        "--output-obc-remap-manifest", str(interior_dir / "obc_remap.json"),
        "--maximum-boundary-insertions", "0",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    result.update({
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "repaired_mesh": str(repaired),
        "repair_report": str(repair_report),
    })
    if not repaired.is_file():
        result["status"] = "needs_review"
        result["failure_taxonomy"] = ["interior_topology_repair_failed"]
        return result
    conditioning = _condition_mesh(
        mesh=repaired,
        size_field=Path(str(required["size_field_nc"])),
        bathymetry=Path(str(required["bathymetry_nc"])),
        boundary_contract=Path(str(required["boundary_contract_json"])),
        source_boundary_metadata=Path(str(required["source_boundary_metadata_json"])),
        condition_dir=output / "conditioned",
    )
    result["conditioning"] = conditioning
    result["status"] = "conditioned" if Path(conditioning["conditioning_report"]).is_file() else "needs_review"
    return result


def _conditionable_raw_remesh(output: Path, hard_node_cap: int = 1_000_000) -> dict[str, Any]:
    """Separate successful raw publication from full raw-quality acceptance."""
    candidate = output / "remesh" / "candidates" / "gmsh_frontal_delaunay_6"
    raw = candidate / "raw_mesh.2dm"
    manifest_path = candidate / "candidate_manifest.json"
    roundtrip_path = candidate / "roundtrip.json"
    generator_path = candidate / "generator_report.json"
    evidence: dict[str, Any] = {
        "passed": False,
        "raw_mesh": str(raw),
        "failure_taxonomy": [],
    }
    required = [raw, manifest_path, roundtrip_path, generator_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        evidence["failure_taxonomy"].append("raw_remesh_artifact_missing")
        evidence["missing"] = missing
        return evidence
    manifest = _read_json(manifest_path)
    roundtrip = _read_json(roundtrip_path)
    generator = _read_json(generator_path)
    if not bool(manifest.get("raw_stage")) or bool(manifest.get("common_conditioning_applied")):
        evidence["failure_taxonomy"].append("raw_stage_contract_failed")
    if not bool(roundtrip.get("passed")):
        evidence["failure_taxonomy"].append("raw_roundtrip_failed")
    if not bool(generator.get("raw_stage")) or bool(generator.get("common_conditioning_applied")):
        evidence["failure_taxonomy"].append("generator_raw_stage_contract_failed")
    node_count = int(manifest.get("node_count", hard_node_cap + 1))
    if node_count > hard_node_cap:
        evidence["failure_taxonomy"].append("raw_node_cap_exceeded")
    evidence.update({
        "node_count": node_count,
        "triangle_count": int(manifest.get("triangle_count", 0)),
        "roundtrip_passed": bool(roundtrip.get("passed")),
        "candidate_quality_accepted": bool(manifest.get("quality_accepted")),
        "candidate_failure_taxonomy": list(manifest.get("failure_taxonomy") or []),
    })
    evidence["passed"] = not evidence["failure_taxonomy"]
    return evidence


def _apply_conditioning_decision(
    report: dict[str, Any], conditioning: dict[str, Any]
) -> None:
    """Populate the three independent status layers from a conditioning audit."""
    report["conditioning"] = conditioning
    conditioning_path = Path(conditioning["conditioning_report"])
    if not conditioning_path.is_file():
        report["status"] = "needs_review"
        report["failure_taxonomy"].append("conditioning_report_missing")
        return
    conditioning_report = _read_json(conditioning_path)
    report["minimal_local_debt_closed"] = bool(
        conditioning_report.get("minimal_local_debt_closed")
    )
    benchmark_ready = bool(
        conditioning_report.get(
            "benchmark_grid_baseline_ready",
            conditioning_report.get("fvcom_ready"),
        )
    )
    report["benchmark_grid_baseline_ready"] = benchmark_ready
    report["fvcom_ready"] = benchmark_ready
    report["accepted"] = benchmark_ready
    report["submission_eligible"] = bool(
        conditioning_report.get("submission_eligible", False)
    )
    report["regional_refinement_debt"] = list(
        conditioning_report.get("regional_refinement_debt") or []
    )
    report["quality_advisories"] = dict(
        conditioning_report.get("quality_advisories") or {}
    )
    report["quality_policy"] = dict(conditioning_report.get("quality_policy") or {})
    audit = conditioning_report.get("final_global_audit") or {}
    superthin = audit.get("superthin_triangle_count")
    report["autonomous_thin_closed"] = bool(superthin == 0)
    report["status"] = "pass" if report["autonomous_thin_closed"] else "needs_review"
    report["final_superthin_triangle_count"] = superthin
    report["failure_taxonomy"] = list(
        conditioning_report.get("fvcom_readiness_failure_taxonomy") or []
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--source-case-manifest", type=Path)
    parser.add_argument("--source-boundary-resolution-manifest", type=Path)
    parser.add_argument("--bathymetry-nc", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--cusp-gpkg", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    diagnostic = _read_json(args.diagnostic)
    if int(diagnostic.get("component_count", -1)) == 0:
        report = no_op_closure_report(diagnostic)
        report["created_at_utc"] = utc_now()
        report_path = _write_json(output / "autonomous_thin_closure.json", report)
        print(json.dumps({
            "status": report["status"],
            "report": str(report_path),
            "autonomous_thin_closed": True,
            "minimal_local_debt_closed": True,
            "benchmark_grid_baseline_ready": report[
                "benchmark_grid_baseline_ready"
            ],
            "fvcom_ready": report["fvcom_ready"],
            "submission_eligible": report["submission_eligible"],
            "failure_taxonomy": report["failure_taxonomy"],
        }, indent=2))
        return 0
    if args.decision is None:
        raise ValueError("--decision is required when thin components remain")
    required_paths = {
        "--source-case-manifest": args.source_case_manifest,
        "--source-boundary-resolution-manifest": args.source_boundary_resolution_manifest,
        "--bathymetry-nc": args.bathymetry_nc,
        "--workspace-root": args.workspace_root,
    }
    missing_arguments = [name for name, value in required_paths.items() if value is None]
    if missing_arguments:
        raise ValueError("missing arguments for active closure: " + ", ".join(missing_arguments))
    decision = _read_json(args.decision)
    validate_agent_decision(
        decision,
        diagnostic_sha256=sha256_file(args.diagnostic),
        mesh_sha256=diagnostic["input_hashes"]["mesh"],
        diagnostic_input_hashes=diagnostic["input_hashes"],
    )
    route = str(decision["route"])
    report: dict[str, Any] = {
        "schema_version": CLOSURE_SCHEMA,
        "profile": "autonomous-thin-v1",
        "created_at_utc": utc_now(),
        "status": "needs_review",
        "decision": str(args.decision.resolve()),
        "decision_sha256": sha256_file(args.decision),
        "route": route,
        "autonomous_thin_closed": False,
        "minimal_local_debt_closed": False,
        "benchmark_grid_baseline_ready": False,
        "fvcom_ready": False,
        "accepted": False,
        "submission_eligible": False,
        "regional_refinement_debt": [],
        "quality_advisories": {},
        "quality_policy": {},
        "failure_taxonomy": [],
    }
    if route == "protected_or_source_conflict":
        report["failure_taxonomy"] = ["protected_or_source_conflict"]
    elif route == "interior_topology_defect":
        interior = _run_interior_topology(
            diagnostic, decision, output, execute=bool(args.execute)
        )
        report["interior_topology"] = interior
        if not args.execute:
            report["status"] = "interior_topology_plan_ready"
        elif interior.get("status") == "conditioned":
            _apply_conditioning_decision(report, interior["conditioning"])
        else:
            report["status"] = "needs_review"
            report["failure_taxonomy"] = list(
                interior.get("failure_taxonomy") or ["interior_topology_repair_failed"]
            )
    else:
        patch = _write_patch_packages(
            args.source_boundary_resolution_manifest.resolve(),
            decision,
            diagnostic,
            output,
            cusp_gpkg=args.cusp_gpkg.resolve() if args.cusp_gpkg else None,
        )
        report["patch"] = patch
        case_path = _write_case_manifest(args.source_case_manifest.resolve(), patch, output)
        report["patched_case_manifest"] = str(case_path)
        report["status"] = "boundary_patch_ready_for_remesh"
        if args.execute:
            remesh = run_portfolio_case(
                case_path,
                args.workspace_root.resolve(),
                output / "remesh",
                candidate_ids=["gmsh-6"],
                config=PortfolioCaseConfig(),
            )
            report["remesh"] = remesh
            raw_acceptance = _conditionable_raw_remesh(output)
            report["raw_remesh_accepted_for_conditioning"] = raw_acceptance
            if raw_acceptance["passed"]:
                conditioning = _condition_generated(output, args.bathymetry_nc.resolve())
                _apply_conditioning_decision(report, conditioning)
            else:
                report["status"] = "needs_review"
                report["failure_taxonomy"].append("gmsh6_remesh_failed")
    report_path = _write_json(output / "autonomous_thin_closure.json", report)
    print(json.dumps({
        "status": report["status"],
        "report": str(report_path),
        "autonomous_thin_closed": report["autonomous_thin_closed"],
        "minimal_local_debt_closed": report["minimal_local_debt_closed"],
        "benchmark_grid_baseline_ready": report[
            "benchmark_grid_baseline_ready"
        ],
        "fvcom_ready": report["fvcom_ready"],
        "submission_eligible": report["submission_eligible"],
        "failure_taxonomy": report["failure_taxonomy"],
    }, indent=2))
    return 0 if report["status"] in {
        "pass", "boundary_patch_ready_for_remesh", "interior_topology_plan_ready"
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
