#!/usr/bin/env python3
"""Apply boundary-fixed spring relaxation to all or part of an FVCOM 2DM mesh."""

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
from fvcom_grid_generation.regional_conditioning import SpringRelaxConfig, relax_mesh_spring  # noqa: E402
from fvcom_grid_generation.sms_2dm import read_2dm, write_2dm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="Input FVCOM/SMS 2DM mesh.")
    parser.add_argument("--output-mesh", "--output", dest="output_mesh", required=True, help="Relaxed output 2DM mesh.")
    parser.add_argument("--report", required=True, help="JSON relaxation report.")
    parser.add_argument("--name", help="Output MESHNAME; defaults to <input-name>_relaxed.")
    parser.add_argument(
        "--region-bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Optional lon/lat bbox; nodes outside its buffered graph patch remain fixed.",
    )
    parser.add_argument("--quality-threshold", type=float, default=0.40, help="Seed triangles with equilateral quality below this value.")
    parser.add_argument("--min-angle-deg", type=float, default=28.0, help="Seed triangles with minimum angle below this value.")
    parser.add_argument("--ring-layers", type=int, default=3, help="Triangle-neighbor rings around seeded defects.")
    parser.add_argument("--iterations", type=int, default=20, help="Maximum damped spring iterations.")
    parser.add_argument("--damping", type=float, default=0.35, help="Spring update damping in (0, 1].")
    parser.add_argument("--max-step-fraction", type=float, default=0.15, help="Maximum step as a fraction of local rest length.")
    parser.add_argument("--shape-weight", type=float, default=0.20, help="Weight of the triangle-shape regularization force.")
    parser.add_argument("--force-tolerance", type=float, default=1.0e-3, help="Normalized force-residual convergence tolerance.")
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

    config = SpringRelaxConfig(
        enabled=True,
        quality_threshold=float(args.quality_threshold),
        min_angle_deg=float(args.min_angle_deg),
        ring_layers=int(args.ring_layers),
        iterations=int(args.iterations),
        damping=float(args.damping),
        max_step_fraction=float(args.max_step_fraction),
        shape_weight=float(args.shape_weight),
        force_tolerance=float(args.force_tolerance),
    )
    result = relax_mesh_spring(
        nodes_xy,
        triangles_zero,
        fixed,
        target_spacing_m=None,
        constraint_chains=chains,
        open_boundary_nodes_zero_based=open_zero,
        region_bbox_xy=region_bbox_xy,
        seed_triangle_mask=None,
        config=config,
    )

    relaxed_xy = np.asarray(result.nodes_xy, dtype=float)
    if relaxed_xy.shape != nodes_xy.shape:
        raise RuntimeError("Spring relaxation must not add, remove, or renumber mesh nodes")
    boundary_shift = _maximum_shift(nodes_xy[fixed], relaxed_xy[fixed])
    if boundary_shift > 1.0e-8:
        raise RuntimeError(f"Relaxation moved a protected boundary node by {boundary_shift:.6g} m")
    _require_constraints(triangles_zero, chains, open_zero)

    output_name = args.name or f"{mesh.mesh_name}_relaxed"
    output_mesh = write_2dm(
        output_path,
        unproject_points(relaxed_xy, projection),
        mesh.depths,
        mesh.triangles,
        mesh.open_boundary_nodes,
        mesh_name=output_name,
    )
    document = {
        "schema_version": "fvcom_mesh_spring_relaxation_cli_v1",
        "input_mesh": str(mesh_path),
        "output_mesh": str(output_mesh),
        "projection_epsg": int(projection.epsg),
        "region_bbox_lonlat": list(map(float, args.region_bbox)) if args.region_bbox else None,
        "protected_boundary_node_count": int(np.count_nonzero(fixed)),
        "boundary_coordinate_max_shift_m": float(boundary_shift),
        "all_boundary_coordinates_unchanged": bool(boundary_shift <= 1.0e-8),
        "obc_order_preserved": True,
        "active_node_count": int(np.count_nonzero(np.asarray(result.active_node_mask, dtype=bool))),
        "settings": config.__dict__,
        "relaxation": result.report,
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
    if not 0.0 <= args.quality_threshold <= 1.0:
        parser.error("--quality-threshold must be between 0 and 1")
    if not 0.0 < args.min_angle_deg < 60.0:
        parser.error("--min-angle-deg must be between 0 and 60")
    if args.ring_layers < 0 or args.iterations < 0:
        parser.error("--ring-layers and --iterations must be nonnegative")
    if not 0.0 < args.damping <= 1.0:
        parser.error("--damping must be in (0, 1]")
    if args.max_step_fraction <= 0.0 or args.shape_weight < 0.0 or args.force_tolerance <= 0.0:
        parser.error("step fraction and force tolerance must be positive; shape weight must be nonnegative")


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
