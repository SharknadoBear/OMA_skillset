#!/usr/bin/env python3
"""Repair thin FVCOM triangles while preserving the land and open-boundary shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.postprocess import boundary_chains_from_mesh  # noqa: E402
from fvcom_grid_generation.local_topology import AggressiveConditioningConfig, condition_mesh_aggressive  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection, project_points, unproject_points  # noqa: E402
from fvcom_grid_generation.regional_conditioning import (  # noqa: E402
    SpringRelaxConfig,
    ThinTriangleRepairConfig,
    repair_thin_triangles,
)
from fvcom_grid_generation.sms_2dm import read_2dm, write_2dm  # noqa: E402
from fvcom_grid_generation.systematic_v6 import SystematicV6LoopConfig, run_systematic_v6_loop  # noqa: E402
from condition_mesh_local import (  # noqa: E402
    _boundary_geojson,
    _boundary_metadata,
    _remap_depths,
    _serialized_roundtrip_audit,
    _target_sizes,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="Input FVCOM/SMS 2DM mesh.")
    parser.add_argument("--output-mesh", "--output", dest="output_mesh", required=True, help="Repaired output 2DM mesh.")
    parser.add_argument("--report", required=True, help="JSON repair report.")
    parser.add_argument("--name", help="Output MESHNAME; defaults to <input-name>_thin_repaired.")
    parser.add_argument(
        "--thin-repair-profile",
        choices=("guarded-v1", "systematic-v2", "systematic-v3", "systematic-v5", "systematic-v6", "none"),
        default="guarded-v1",
        help="Choose guarded-v1, systematic-v2, boundary-adaptive systematic-v3, locked-star systematic-v5, or none.",
    )
    parser.add_argument("--systematic-v3-obc-policy", choices=("preserve", "redistribute"), default="redistribute")
    parser.add_argument("--systematic-v5-max-star-transactions", type=int, default=256)
    parser.add_argument(
        "--systematic-v5-connectivity-restriction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--systematic-v5-max-connectivity-transactions",
        type=int,
        default=32,
    )
    parser.add_argument("--systematic-v5-wall-time-s", type=float, default=21600.0)
    parser.add_argument(
        "--systematic-v5-boundary-window-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--systematic-v6-total-iterations", type=int, default=1000)
    parser.add_argument("--systematic-v6-max-cycles", type=int, default=12)
    parser.add_argument("--systematic-v6-max-closure-rounds", type=int, default=8)
    parser.add_argument("--systematic-v6-wall-time-s", type=float, default=28800.0)
    parser.add_argument("--systematic-v6-final-audit-reserve-s", type=float, default=3600.0)
    parser.add_argument("--boundary-nodes-geojson")
    parser.add_argument("--output-boundary-nodes")
    parser.add_argument("--output-obc-remap-manifest")
    parser.add_argument("--boundary-resolution-manifest")
    parser.add_argument("--size-field-nc")
    parser.add_argument("--target-spacing-m", type=float)
    parser.add_argument(
        "--region-bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Optional lon/lat bbox limiting defect detection and repair.",
    )
    parser.add_argument("--quality-threshold", type=float, default=0.25, help="Repair triangles with equilateral quality below this value.")
    parser.add_argument("--min-angle-deg", type=float, default=20.0, help="Repair triangles with minimum angle below this value.")
    parser.add_argument("--max-passes", type=int, default=2, help="Maximum repair passes.")
    parser.add_argument("--max-flips", type=int, default=200, help="Maximum accepted edge flips.")
    parser.add_argument("--max-insertions", type=int, default=50, help="Maximum shape-preserving edge midpoint insertions.")
    parser.add_argument("--split-target-factor", type=float, default=1.25, help="Split a long edge only above this multiple of local target size.")
    parser.add_argument("--relax-quality-threshold", type=float, default=0.40)
    parser.add_argument("--relax-min-angle-deg", type=float, default=28.0)
    parser.add_argument("--relax-ring-layers", type=int, default=3)
    parser.add_argument("--relax-iterations", type=int, default=20)
    parser.add_argument("--relax-damping", type=float, default=0.35)
    parser.add_argument("--relax-max-step-fraction", type=float, default=0.15)
    parser.add_argument("--relax-shape-weight", type=float, default=0.20)
    parser.add_argument("--relax-force-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of existing output/report files.")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    output_path = Path(args.output_mesh)
    report_path = Path(args.report)
    _validate_paths(mesh_path, output_path, report_path, overwrite=bool(args.overwrite))
    _validate_controls(args, parser)

    if args.thin_repair_profile == "none":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mesh_path, output_path)
        document = {
            "schema_version": "fvcom_thin_triangle_repair_cli_v2",
            "profile": "none",
            "input_mesh": str(mesh_path),
            "output_mesh": str(output_path),
            "repair": {"enabled": False, "reason": "thin_repair_profile_none", "edit_count": 0},
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        print(json.dumps({"output_mesh": str(output_path), "report": str(report_path)}, indent=2))
        return 0

    mesh = read_2dm(mesh_path)
    projection = local_utm_projection(_mesh_bbox(mesh.nodes_lonlat))
    nodes_xy = project_points(mesh.nodes_lonlat, projection)
    triangles_zero = np.asarray(mesh.triangles, dtype=int) - 1
    open_zero = np.asarray(mesh.open_boundary_nodes, dtype=int) - 1
    chains = boundary_chains_from_mesh(mesh.triangles)
    fixed = _fixed_boundary_mask(len(nodes_xy), chains)
    region_bbox_xy = _project_bbox(args.region_bbox, projection) if args.region_bbox else None

    if args.thin_repair_profile in {"systematic-v2", "systematic-v3", "systematic-v5", "systematic-v6"}:
        systematic_profile = str(args.thin_repair_profile)
        if args.region_bbox:
            parser.error(f"--region-bbox is not supported by {systematic_profile}; connected components define local patches")
        chains, kinds, hard, explicit_targets = _boundary_metadata(
            len(nodes_xy),
            triangles_zero,
            open_zero,
            args.boundary_nodes_geojson,
            args.boundary_resolution_manifest,
        )
        fixed = _fixed_boundary_mask(len(nodes_xy), chains)
        targets = _target_sizes(mesh.nodes_lonlat, nodes_xy, triangles_zero, args.size_field_nc)
        if args.target_spacing_m is not None:
            if args.target_spacing_m <= 0.0:
                parser.error("--target-spacing-m must be positive")
            targets[:] = float(args.target_spacing_m)
        explicit = np.isfinite(explicit_targets) & (explicit_targets > 0.0)
        targets[explicit] = explicit_targets[explicit]
        topology_config = AggressiveConditioningConfig(
                thin_repair_profile=systematic_profile,
                systematic_v3_obc_policy=str(args.systematic_v3_obc_policy),
                systematic_gate_scope=(
                    "loop-end" if systematic_profile in {"systematic-v5", "systematic-v6"} else "candidate"
                ),
                systematic_v5_max_star_transactions_per_round=int(
                    args.systematic_v5_max_star_transactions
                ),
                systematic_v5_enable_connectivity_restriction=bool(
                    args.systematic_v5_connectivity_restriction
                ),
                systematic_v5_max_connectivity_transactions_per_round=int(
                    args.systematic_v5_max_connectivity_transactions
                ),
                systematic_v5_enable_boundary_window_fallback=bool(
                    args.systematic_v5_boundary_window_fallback
                ),
                deadline_monotonic_s=(
                    time.perf_counter() + float(args.systematic_v5_wall_time_s)
                    if systematic_profile in {"systematic-v5", "systematic-v6"}
                    else None
                ),
                max_rounds=int(args.max_passes),
                enable_pruning=False,
                enable_thin_repair=True,
                enable_valence_repair=bool(systematic_profile == "systematic-v6"),
                max_prunes_per_round=0,
                max_valence_removals_per_round=(
                    500 if systematic_profile == "systematic-v6" else 0
                ),
            )
        if systematic_profile == "systematic-v6":
            systematic = run_systematic_v6_loop(
                nodes_xy,
                triangles_zero,
                fixed,
                chains,
                open_zero,
                target_spacing_m=targets,
                boundary_kinds=kinds,
                hard_anchor_mask=hard,
                topology_config=topology_config,
                loop_config=SystematicV6LoopConfig(
                    maximum_closure_rounds=int(
                        args.systematic_v6_max_closure_rounds
                    ),
                    maximum_relaxation_cycles=int(
                        args.systematic_v6_max_cycles
                    ),
                    total_relaxation_iterations=int(
                        args.systematic_v6_total_iterations
                    ),
                    wall_clock_seconds=float(
                        args.systematic_v6_wall_time_s
                    ),
                    final_audit_reserve_seconds=float(
                        args.systematic_v6_final_audit_reserve_s
                    ),
                ),
            )
        else:
            systematic = condition_mesh_aggressive(
                nodes_xy,
                triangles_zero,
                fixed,
                chains,
                open_zero,
                target_spacing_m=targets,
                boundary_kinds=kinds,
                hard_anchor_mask=hard,
                config=topology_config,
            )
        depths = _remap_depths(mesh.depths, nodes_xy, systematic.nodes_xy, systematic.node_lineage)
        output_name = args.name or f"{mesh.mesh_name}_{systematic_profile.replace('-', '_')}"
        output_mesh = write_2dm(
            output_path,
            unproject_points(systematic.nodes_xy, projection),
            depths,
            systematic.triangles + 1,
            systematic.open_boundary_nodes_zero_based + 1,
            mesh_name=output_name,
        )
        roundtrip = _serialized_roundtrip_audit(output_mesh, systematic, projection)
        boundary_output = Path(args.output_boundary_nodes) if args.output_boundary_nodes else None
        obc_remap_output = Path(args.output_obc_remap_manifest) if args.output_obc_remap_manifest else None
        if boundary_output is not None:
            boundary_output.parent.mkdir(parents=True, exist_ok=True)
            boundary_output.write_text(
                json.dumps(
                    _boundary_geojson(
                        unproject_points(systematic.nodes_xy, projection),
                        systematic.constraint_chains,
                        systematic.open_boundary_nodes_zero_based,
                        systematic.boundary_kinds,
                        systematic.hard_anchor_mask,
                        systematic.node_lineage,
                        systematic.target_spacing_m,
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
        if obc_remap_output is not None:
            obc_remap_output.parent.mkdir(parents=True, exist_ok=True)
            obc_remap_output.write_text(
                json.dumps(_json_safe(systematic.obc_remap_manifest), indent=2),
                encoding="utf-8",
            )
        lineage_to_current = {
            int(lineage): int(index)
            for index, lineage in enumerate(np.asarray(systematic.node_lineage, dtype=int))
            if int(lineage) >= 0
        }
        original_boundary = np.where(fixed)[0]
        if systematic_profile == "systematic-v2" and any(
            int(node) not in lineage_to_current for node in original_boundary
        ):
            raise RuntimeError("systematic-v2 removed an original protected boundary node")
        surviving_boundary = [int(node) for node in original_boundary if int(node) in lineage_to_current]
        current_boundary = np.asarray([lineage_to_current[node] for node in surviving_boundary], dtype=int)
        boundary_shift = (
            _maximum_shift(nodes_xy[np.asarray(surviving_boundary, dtype=int)], systematic.nodes_xy[current_boundary])
            if surviving_boundary
            else 0.0
        )
        delivered_open_lineage = np.asarray(
            [systematic.node_lineage[int(node)] for node in systematic.open_boundary_nodes_zero_based],
            dtype=int,
        )
        _require_constraints(systematic.triangles, systematic.constraint_chains, systematic.open_boundary_nodes_zero_based)
        document = {
            "schema_version": "fvcom_thin_triangle_repair_cli_v2",
            "profile": systematic_profile,
            "input_mesh": str(mesh_path),
            "output_mesh": str(output_mesh),
            "output_boundary_nodes": str(boundary_output) if boundary_output is not None else None,
            "output_obc_remap_manifest": str(obc_remap_output) if obc_remap_output is not None else None,
            "projection_epsg": int(projection.epsg),
            "boundary_metadata_supplied": bool(args.boundary_nodes_geojson),
            "size_field_supplied": bool(args.size_field_nc),
            "original_boundary_coordinate_max_shift_m": float(boundary_shift),
            "obc_order_preserved": bool(np.array_equal(open_zero, delivered_open_lineage)),
            "obc_remap_manifest": systematic.obc_remap_manifest,
            "serialized_roundtrip": roundtrip,
            "repair": systematic.report,
            "edit_ledger": systematic.edit_ledger,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_json_safe(document), indent=2), encoding="utf-8")
        print(json.dumps({"output_mesh": str(output_mesh), "report": str(report_path)}, indent=2))
        return 0 if systematic.report["superthin_gate_passed"] and roundtrip["passed"] else 2

    relaxation_config = SpringRelaxConfig(
        enabled=True,
        quality_threshold=float(args.relax_quality_threshold),
        min_angle_deg=float(args.relax_min_angle_deg),
        ring_layers=int(args.relax_ring_layers),
        iterations=int(args.relax_iterations),
        damping=float(args.relax_damping),
        max_step_fraction=float(args.relax_max_step_fraction),
        shape_weight=float(args.relax_shape_weight),
        force_tolerance=float(args.relax_force_tolerance),
    )
    config = ThinTriangleRepairConfig(
        enabled=True,
        quality_threshold=float(args.quality_threshold),
        min_angle_deg=float(args.min_angle_deg),
        max_passes=int(args.max_passes),
        max_flips=int(args.max_flips),
        max_insertions=int(args.max_insertions),
        split_target_factor=float(args.split_target_factor),
        relaxation_config=relaxation_config,
    )
    result = repair_thin_triangles(
        nodes_xy,
        triangles_zero,
        fixed,
        chains,
        open_zero,
        target_spacing_m=None,
        region_bbox_xy=region_bbox_xy,
        config=config,
    )

    repaired_xy = np.asarray(result.nodes_xy, dtype=float)
    repaired_triangles = np.asarray(result.triangles, dtype=int)
    repaired_fixed = np.asarray(result.fixed_node_mask, dtype=bool)
    repaired_chains = [list(map(int, chain)) for chain in result.constraint_chains]
    repaired_open = np.asarray(result.open_boundary_nodes_zero_based, dtype=int)
    if len(repaired_xy) < len(nodes_xy) or not np.allclose(repaired_xy[: len(nodes_xy)][fixed], nodes_xy[fixed], atol=1.0e-8, rtol=0.0):
        raise RuntimeError("Thin-triangle repair moved, removed, or renumbered an original protected boundary node")
    boundary_shift = _maximum_shift(nodes_xy[fixed], repaired_xy[: len(nodes_xy)][fixed])
    if not _is_ordered_subsequence(open_zero.tolist(), repaired_open.tolist()):
        raise RuntimeError("Thin-triangle repair did not preserve original OBC node order")
    _require_constraints(repaired_triangles, repaired_chains, repaired_open)

    parent_edges = _normalize_parent_edges(result.inserted_parent_edges, original_node_count=len(nodes_xy))
    boundary_insertion_deviation = _boundary_insertion_deviation(repaired_xy, repaired_fixed, parent_edges)
    if boundary_insertion_deviation > 1.0e-7:
        raise RuntimeError(
            f"An inserted protected node deviates from its parent boundary edge by {boundary_insertion_deviation:.6g} m"
        )
    depths = _assign_inserted_depths(mesh.depths, len(repaired_xy), parent_edges)
    output_name = args.name or f"{mesh.mesh_name}_thin_repaired"
    output_mesh = write_2dm(
        output_path,
        unproject_points(repaired_xy, projection),
        depths,
        repaired_triangles + 1,
        repaired_open + 1,
        mesh_name=output_name,
    )
    document = {
        "schema_version": "fvcom_thin_triangle_repair_cli_v1",
        "input_mesh": str(mesh_path),
        "output_mesh": str(output_mesh),
        "projection_epsg": int(projection.epsg),
        "region_bbox_lonlat": list(map(float, args.region_bbox)) if args.region_bbox else None,
        "original_node_count": int(len(nodes_xy)),
        "repaired_node_count": int(len(repaired_xy)),
        "inserted_node_count": int(len(repaired_xy) - len(nodes_xy)),
        "inserted_parent_edges": parent_edges,
        "inserted_depth_method": "arithmetic_mean_of_recorded_parent_edge_depths",
        "protected_boundary_node_count": int(np.count_nonzero(repaired_fixed)),
        "original_boundary_coordinate_max_shift_m": float(boundary_shift),
        "original_boundary_coordinates_unchanged": bool(boundary_shift <= 1.0e-8),
        "inserted_boundary_parent_edge_max_deviation_m": float(boundary_insertion_deviation),
        "boundary_polyline_shape_unchanged": bool(boundary_shift <= 1.0e-8 and boundary_insertion_deviation <= 1.0e-7),
        "obc_order_preserved": True,
        "settings": {
            **{key: value for key, value in config.__dict__.items() if key != "relaxation_config"},
            "relaxation_config": relaxation_config.__dict__,
        },
        "repair": result.report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_json_safe(document), indent=2), encoding="utf-8")
    print(json.dumps({"output_mesh": str(output_mesh), "report": str(report_path)}, indent=2))
    return 0


def _mesh_bbox(nodes_lonlat: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.min(nodes_lonlat[:, 0])),
        float(np.min(nodes_lonlat[:, 1])),
        float(np.max(nodes_lonlat[:, 0])),
        float(np.max(nodes_lonlat[:, 1])),
    )


def _project_bbox(values: list[float], projection: Any) -> tuple[float, float, float, float]:
    west, south, east, north = map(float, values)
    if not west < east or not south < north:
        raise ValueError("--region-bbox requires WEST < EAST and SOUTH < NORTH")
    corners = project_points(
        np.asarray([[west, south], [west, north], [east, south], [east, north]], dtype=float),
        projection,
    )
    return (
        float(np.min(corners[:, 0])),
        float(np.min(corners[:, 1])),
        float(np.max(corners[:, 0])),
        float(np.max(corners[:, 1])),
    )


def _fixed_boundary_mask(node_count: int, chains: list[list[int]]) -> np.ndarray:
    fixed = np.zeros(int(node_count), dtype=bool)
    for chain in chains:
        fixed[np.asarray(chain, dtype=int)] = True
    return fixed


def _normalize_parent_edges(values: Any, *, original_node_count: int) -> list[tuple[int, int, int]]:
    if isinstance(values, dict):
        records = [(int(node), int(edge[0]), int(edge[1])) for node, edge in values.items()]
    else:
        records = [tuple(map(int, record)) for record in values]
    if any(len(record) != 3 for record in records):
        raise RuntimeError("inserted_parent_edges must contain (new_node, parent_a, parent_b) records")
    records = sorted(records)
    expected = list(range(int(original_node_count), int(original_node_count) + len(records)))
    if [record[0] for record in records] != expected:
        raise RuntimeError("Every appended node must have exactly one ordered parent-edge record")
    return records


def _assign_inserted_depths(original: np.ndarray, node_count: int, parent_edges: list[tuple[int, int, int]]) -> np.ndarray:
    original = np.asarray(original, dtype=float)
    depths = np.full(int(node_count), np.nan, dtype=float)
    depths[: len(original)] = original
    for new_node, parent_a, parent_b in parent_edges:
        if not (0 <= parent_a < new_node and 0 <= parent_b < new_node):
            raise RuntimeError(f"Invalid parent edge for inserted node {new_node}: {(parent_a, parent_b)}")
        if not np.isfinite(depths[parent_a]) or not np.isfinite(depths[parent_b]):
            raise RuntimeError(f"Parent depths are unavailable for inserted node {new_node}")
        depths[new_node] = 0.5 * (depths[parent_a] + depths[parent_b])
    if not np.all(np.isfinite(depths)):
        raise RuntimeError("A repaired node is missing its parent-edge depth assignment")
    return depths


def _boundary_insertion_deviation(
    points: np.ndarray,
    fixed: np.ndarray,
    parent_edges: list[tuple[int, int, int]],
) -> float:
    deviations: list[float] = []
    for new_node, parent_a, parent_b in parent_edges:
        if not fixed[new_node]:
            continue
        if not fixed[parent_a] or not fixed[parent_b]:
            return float("inf")
        a = points[parent_a]
        b = points[parent_b]
        edge = b - a
        denominator = float(np.dot(edge, edge))
        if denominator <= 1.0e-24:
            deviations.append(float(np.linalg.norm(points[new_node] - a)))
            continue
        fraction = float(np.dot(points[new_node] - a, edge) / denominator)
        if fraction < -1.0e-12 or fraction > 1.0 + 1.0e-12:
            return float("inf")
        closest = a + np.clip(fraction, 0.0, 1.0) * edge
        deviations.append(float(np.linalg.norm(points[new_node] - closest)))
    return float(max(deviations, default=0.0))


def _is_ordered_subsequence(original: list[int], candidate: list[int]) -> bool:
    iterator = iter(candidate)
    return all(any(value == expected for value in iterator) for expected in original)


def _require_constraints(triangles_zero: np.ndarray, chains: list[list[int]], open_zero: np.ndarray) -> None:
    edges = {
        tuple(sorted((int(a), int(b))))
        for tri in np.asarray(triangles_zero, dtype=int)
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
    }
    missing = []
    for chain in chains:
        for index, a in enumerate(chain):
            edge = tuple(sorted((int(a), int(chain[(index + 1) % len(chain)]))))
            if edge not in edges:
                missing.append(edge)
    missing_open = [
        tuple(sorted((int(a), int(b))))
        for a, b in zip(open_zero[:-1], open_zero[1:])
        if tuple(sorted((int(a), int(b)))) not in edges
    ]
    if missing or missing_open:
        raise RuntimeError(f"Constraint audit failed: {len(missing)} protected and {len(missing_open)} OBC edges missing")


def _maximum_shift(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(after - before, axis=1))) if len(before) else 0.0


def _validate_paths(mesh: Path, output: Path, report: Path, *, overwrite: bool) -> None:
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    if output.resolve() == mesh.resolve():
        raise ValueError("Refusing to overwrite the input mesh")
    if output.resolve() == report.resolve():
        raise ValueError("--output-mesh and --report must be different paths")
    existing = [path for path in (output, report) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output(s): {existing}")


def _validate_controls(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    for name in ("quality_threshold", "relax_quality_threshold"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    for name in ("min_angle_deg", "relax_min_angle_deg"):
        if not 0.0 < getattr(args, name) < 60.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 60")
    if min(args.max_passes, args.max_flips, args.max_insertions, args.relax_ring_layers, args.relax_iterations) < 0:
        parser.error("pass, operation, ring, and iteration budgets must be nonnegative")
    if args.systematic_v5_max_connectivity_transactions < 0:
        parser.error(
            "--systematic-v5-max-connectivity-transactions must be nonnegative"
        )
    if (
        args.systematic_v6_total_iterations < 0
        or args.systematic_v6_max_cycles < 0
        or args.systematic_v6_max_closure_rounds < 0
        or args.systematic_v6_wall_time_s <= 0.0
        or args.systematic_v6_final_audit_reserve_s < 0.0
    ):
        parser.error("systematic-v6 controls must be nonnegative with positive wall time")
    if args.split_target_factor <= 0.0 or args.relax_max_step_fraction <= 0.0 or args.relax_force_tolerance <= 0.0:
        parser.error("split factor, relaxation step fraction, and force tolerance must be positive")
    if not 0.0 < args.relax_damping <= 1.0 or args.relax_shape_weight < 0.0:
        parser.error("relaxation damping must be in (0, 1] and shape weight must be nonnegative")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
