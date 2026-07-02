"""Pure-Python OceanMesh-style constrained Delaunay refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import Point

from .boundary import BoundaryNodes
from .projection import unproject_points
from .size_field import SizeField


@dataclass(frozen=True)
class MeshConfig:
    max_constraint_iterations: int = 8
    refine_iterations: int = 3
    smooth_iterations: int = 8
    max_interior_points: int = 80_000
    max_refine_insertions_per_iter: int = 1500
    size_overrun_factor: float = 1.55
    min_angle_refine_deg: float = 24.0


@dataclass
class MeshResult:
    nodes_xy: np.ndarray
    nodes_lonlat: np.ndarray
    triangles: np.ndarray
    open_boundary_nodes: np.ndarray
    boundary_node_count: int
    fixed_node_mask: np.ndarray
    report: dict[str, Any]


ProgressCallback = Callable[[str, float, dict[str, Any] | None], None]


def generate_mesh(
    boundary: BoundaryNodes,
    size_field: SizeField,
    config: MeshConfig,
    progress_callback: ProgressCallback | None = None,
) -> MeshResult:
    """Generate a constrained Delaunay mesh with boundary midpoint recovery."""
    _progress(progress_callback, "seed_boundary_nodes", 0.0, {"boundary_node_count": int(len(boundary.xy))})
    points = [tuple(xy) for xy in boundary.xy]
    kinds = list(boundary.kinds)
    chains = [list(chain) for chain in boundary.constraint_chains]
    open_nodes = list(boundary.open_boundary_indices)

    interior = _interior_seed_points(boundary, size_field, config)
    _progress(progress_callback, "seed_interior_points", 0.08, {"interior_seed_count": int(len(interior))})
    for xy in interior:
        points.append((float(xy[0]), float(xy[1])))
        kinds.append("interior")

    constraint_report: dict[str, Any] = {}
    triangles = np.empty((0, 3), dtype=int)
    for iteration in range(config.max_constraint_iterations + 1):
        arr = np.asarray(points, dtype=float)
        triangles = _triangulate_filtered(arr, boundary)
        missing = _missing_constraint_edges(triangles, chains)
        constraint_report = {
            "iterations": int(iteration),
            "missing_constraint_edge_count": int(len(missing)),
            "boundary_constraint_recovered": bool(len(missing) == 0),
        }
        _progress(
            progress_callback,
            "recover_boundary_constraints",
            0.10 + 0.30 * min(iteration + 1, config.max_constraint_iterations + 1) / max(config.max_constraint_iterations + 1, 1),
            constraint_report,
        )
        if not missing:
            break
        pending = sorted(missing[: max(1, 2_000)], key=lambda item: (item[0], item[1]), reverse=True)
        for chain_idx, edge_pos in pending:
            chain = chains[chain_idx]
            a = chain[edge_pos]
            b = chain[(edge_pos + 1) % len(chain)]
            midpoint = 0.5 * (arr[a] + arr[b])
            kind = kinds[a] if kinds[a] == kinds[b] else "land"
            new_idx = len(points)
            points.append((float(midpoint[0]), float(midpoint[1])))
            kinds.append(kind)
            chain.insert(edge_pos + 1, new_idx)
            if kind == "open":
                _insert_open_node(open_nodes, a, b, new_idx)

    points_arr = np.asarray(points, dtype=float)
    boundary_count = len([kind for kind in kinds if kind != "interior"])
    fixed_mask = np.asarray([kind != "interior" for kind in kinds], dtype=bool)

    for refine_iter in range(config.refine_iterations):
        triangles = _triangulate_filtered(points_arr, boundary)
        additions = _refinement_points(points_arr, triangles, boundary, size_field, config)
        _progress(
            progress_callback,
            "refine_triangles",
            0.45 + 0.30 * (refine_iter + 1) / max(config.refine_iterations, 1),
            {
                "iteration": int(refine_iter + 1),
                "total_iterations": int(config.refine_iterations),
                "triangle_count": int(len(triangles)),
                "added_node_count": int(len(additions)),
            },
        )
        if not len(additions):
            break
        points_arr = np.vstack([points_arr, additions])
        fixed_mask = np.concatenate([fixed_mask, np.zeros(len(additions), dtype=bool)])
        kinds.extend(["interior"] * len(additions))

    triangles = _triangulate_filtered(points_arr, boundary)
    _progress(progress_callback, "smooth_interior_nodes", 0.82, {"node_count": int(len(points_arr)), "triangle_count": int(len(triangles))})
    points_arr = _smooth_interior(points_arr, triangles, fixed_mask, boundary, iterations=config.smooth_iterations)
    triangles = _triangulate_filtered(points_arr, boundary)
    triangles = _orient_ccw(points_arr, triangles)
    lonlat = unproject_points(points_arr, boundary.projection)
    open_boundary_nodes = np.asarray([idx + 1 for idx in open_nodes if idx < len(points_arr)], dtype=int)
    report = {
        "schema_version": "fvcom_python_oceanmesh_mesh_v1",
        "backend": "scipy_delaunay_clean_room",
        "node_count": int(len(points_arr)),
        "triangle_count": int(len(triangles)),
        "boundary_node_count": int(boundary_count),
        "open_boundary_node_count": int(len(open_boundary_nodes)),
        "constraint_recovery": constraint_report,
        "refine_iterations": int(config.refine_iterations),
        "smooth_iterations": int(config.smooth_iterations),
    }
    _progress(progress_callback, "mesh_complete", 1.0, report)
    return MeshResult(
        nodes_xy=points_arr,
        nodes_lonlat=lonlat,
        triangles=triangles + 1,
        open_boundary_nodes=open_boundary_nodes,
        boundary_node_count=boundary_count,
        fixed_node_mask=fixed_mask,
        report=report,
    )


def _progress(callback: ProgressCallback | None, message: str, fraction: float, extra: dict[str, Any] | None = None) -> None:
    if callback is not None:
        callback(message, float(fraction), extra)


def _interior_seed_points(boundary: BoundaryNodes, size_field: SizeField, config: MeshConfig) -> np.ndarray:
    domain = boundary.domain_polygon_xy
    minx, miny, maxx, maxy = domain.bounds
    area = max(float(domain.area), 1.0)
    spacing = max(float(np.nanmedian(size_field.size)), float(np.sqrt(area / max(config.max_interior_points, 1))))
    spacing = max(spacing, 10.0)
    xs = np.arange(minx + 0.5 * spacing, maxx, spacing)
    ys = np.arange(miny + 0.5 * spacing, maxy, spacing * np.sqrt(3.0) / 2.0)
    pts = []
    for row, y in enumerate(ys):
        offset = 0.5 * spacing if row % 2 else 0.0
        for x in xs + offset:
            point = Point(float(x), float(y))
            if domain.contains(point):
                pts.append((float(x), float(y)))
            if len(pts) >= config.max_interior_points:
                break
        if len(pts) >= config.max_interior_points:
            break
    return np.asarray(pts, dtype=float)


def _triangulate_filtered(points: np.ndarray, boundary: BoundaryNodes) -> np.ndarray:
    if len(points) < 3:
        return np.empty((0, 3), dtype=int)
    unique, inverse = np.unique(np.round(points, 8), axis=0, return_inverse=True)
    if len(unique) < 3:
        return np.empty((0, 3), dtype=int)
    tri = Delaunay(unique)
    simplices_unique = np.asarray(tri.simplices, dtype=int)
    simplices = np.asarray([[np.where(inverse == idx)[0][0] for idx in simplex] for simplex in simplices_unique], dtype=int)
    kept = []
    domain = boundary.domain_polygon_xy
    for simplex in simplices:
        coords = points[simplex]
        centroid = Point(float(coords[:, 0].mean()), float(coords[:, 1].mean()))
        if domain.contains(centroid):
            kept.append(simplex)
    return np.asarray(kept, dtype=int)


def _missing_constraint_edges(triangles: np.ndarray, chains: list[list[int]]) -> list[tuple[int, int]]:
    edge_set: set[tuple[int, int]] = set()
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_set.add(tuple(sorted((int(a), int(b)))))
    missing = []
    for chain_idx, chain in enumerate(chains):
        if len(chain) < 2:
            continue
        for pos, a in enumerate(chain):
            b = chain[(pos + 1) % len(chain)]
            if tuple(sorted((int(a), int(b)))) not in edge_set:
                missing.append((chain_idx, pos))
    return missing


def _insert_open_node(open_nodes: list[int], a: int, b: int, new_idx: int) -> None:
    for i in range(len(open_nodes) - 1):
        if open_nodes[i] == a and open_nodes[i + 1] == b:
            open_nodes.insert(i + 1, new_idx)
            return
    open_nodes.append(new_idx)


def _refinement_points(points: np.ndarray, triangles: np.ndarray, boundary: BoundaryNodes, size_field: SizeField, config: MeshConfig) -> np.ndarray:
    additions = []
    if not len(triangles):
        return np.empty((0, 2), dtype=float)
    tree = cKDTree(points)
    for tri in triangles:
        coords = points[tri]
        max_edge = _max_edge_length(coords)
        min_angle = _triangle_angles(coords).min()
        centroid = coords.mean(axis=0)
        lonlat = unproject_points(np.asarray([centroid]), boundary.projection)[0]
        target = float(size_field.sample(np.asarray([lonlat[0]]), np.asarray([lonlat[1]]))[0])
        if max_edge <= config.size_overrun_factor * target and min_angle >= config.min_angle_refine_deg:
            continue
        candidate = _circumcenter(coords)
        if candidate is None or not boundary.domain_polygon_xy.contains(Point(float(candidate[0]), float(candidate[1]))):
            a, b = _longest_edge(coords)
            candidate = 0.5 * (a + b)
        if not boundary.domain_polygon_xy.contains(Point(float(candidate[0]), float(candidate[1]))):
            continue
        distance, _ = tree.query(candidate)
        if distance < max(0.25 * target, 1.0):
            continue
        additions.append(candidate)
        if len(additions) >= config.max_refine_insertions_per_iter:
            break
    return np.asarray(additions, dtype=float)


def _smooth_interior(points: np.ndarray, triangles: np.ndarray, fixed: np.ndarray, boundary: BoundaryNodes, iterations: int) -> np.ndarray:
    out = points.copy()
    adjacency = [set() for _ in range(len(out))]
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            adjacency[int(a)].add(int(b))
            adjacency[int(b)].add(int(a))
    for _ in range(max(0, iterations)):
        updated = out.copy()
        for idx, neighbors in enumerate(adjacency):
            if fixed[idx] or not neighbors:
                continue
            target = out[list(neighbors)].mean(axis=0)
            candidate = 0.45 * out[idx] + 0.55 * target
            if boundary.domain_polygon_xy.contains(Point(float(candidate[0]), float(candidate[1]))):
                updated[idx] = candidate
        out = updated
    return out


def _orient_ccw(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    out = triangles.copy()
    for i, tri in enumerate(out):
        coords = points[tri]
        area2 = np.cross(coords[1] - coords[0], coords[2] - coords[0])
        if area2 < 0:
            out[i] = [tri[0], tri[2], tri[1]]
    return out


def _max_edge_length(coords: np.ndarray) -> float:
    return float(max(np.linalg.norm(coords[0] - coords[1]), np.linalg.norm(coords[1] - coords[2]), np.linalg.norm(coords[2] - coords[0])))


def _triangle_angles(coords: np.ndarray) -> np.ndarray:
    lengths = np.asarray([
        np.linalg.norm(coords[1] - coords[2]),
        np.linalg.norm(coords[0] - coords[2]),
        np.linalg.norm(coords[0] - coords[1]),
    ])
    angles = []
    for i in range(3):
        a = lengths[i]
        b = lengths[(i + 1) % 3]
        c = lengths[(i + 2) % 3]
        denom = max(2.0 * b * c, 1.0e-12)
        val = np.clip((b * b + c * c - a * a) / denom, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(val)))
    return np.asarray(angles, dtype=float)


def _circumcenter(coords: np.ndarray) -> np.ndarray | None:
    ax, ay = coords[0]
    bx, by = coords[1]
    cx, cy = coords[2]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1.0e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return np.asarray([ux, uy], dtype=float)


def _longest_edge(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = ((coords[0], coords[1]), (coords[1], coords[2]), (coords[2], coords[0]))
    return max(edges, key=lambda item: float(np.linalg.norm(item[0] - item[1])))
