#!/usr/bin/env python3
"""Repair thin FVCOM triangles while preserving the land and open-boundary shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.postprocess import boundary_chains_from_mesh  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection, project_points, unproject_points  # noqa: E402
from fvcom_grid_generation.regional_conditioning import (  # noqa: E402
    SpringRelaxConfig,
    ThinTriangleRepairConfig,
    repair_thin_triangles,
)
from fvcom_grid_generation.sms_2dm import read_2dm, write_2dm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="Input FVCOM/SMS 2DM mesh.")
    parser.add_argument("--output-mesh", "--output", dest="output_mesh", required=True, help="Repaired output 2DM mesh.")
    parser.add_argument("--report", required=True, help="JSON repair report.")
    parser.add_argument("--name", help="Output MESHNAME; defaults to <input-name>_thin_repaired.")
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

    mesh = read_2dm(mesh_path)
    projection = local_utm_projection(_mesh_bbox(mesh.nodes_lonlat))
    nodes_xy = project_points(mesh.nodes_lonlat, projection)
    triangles_zero = np.asarray(mesh.triangles, dtype=int) - 1
    open_zero = np.asarray(mesh.open_boundary_nodes, dtype=int) - 1
    chains = boundary_chains_from_mesh(mesh.triangles)
    fixed = _fixed_boundary_mask(len(nodes_xy), chains)
    region_bbox_xy = _project_bbox(args.region_bbox, projection) if args.region_bbox else None

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
