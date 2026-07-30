"""Clean-room OceanMesh-style geometric and topology metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class EdgeTopology:
    edge_to_triangles: dict[tuple[int, int], list[int]]
    node_neighbors: list[set[int]]
    triangle_neighbor_count: np.ndarray
    boundary_edges: list[tuple[int, int]]
    nonmanifold_edges: list[tuple[int, int]]
    connected_component_sizes: list[int]


def triangle_geometry(points: np.ndarray, triangles: np.ndarray) -> dict[str, np.ndarray]:
    """Return signed area, edge lengths, angles, and equilateral quality."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    if not len(triangles):
        empty = np.empty(0, dtype=float)
        return {
            "signed_area": empty,
            "area": empty,
            "edge_lengths": np.empty((0, 3), dtype=float),
            "angles_deg": np.empty((0, 3), dtype=float),
            "quality": empty,
        }
    coords = points[triangles]
    v10 = coords[:, 1] - coords[:, 0]
    v20 = coords[:, 2] - coords[:, 0]
    area2 = v10[:, 0] * v20[:, 1] - v10[:, 1] * v20[:, 0]
    lengths = np.column_stack(
        [
            np.linalg.norm(coords[:, 1] - coords[:, 2], axis=1),
            np.linalg.norm(coords[:, 0] - coords[:, 2], axis=1),
            np.linalg.norm(coords[:, 0] - coords[:, 1], axis=1),
        ]
    )
    angles = np.empty_like(lengths)
    for idx in range(3):
        a = lengths[:, idx]
        b = lengths[:, (idx + 1) % 3]
        c = lengths[:, (idx + 2) % 3]
        denom = np.maximum(2.0 * b * c, 1.0e-30)
        cosine = np.clip((b * b + c * c - a * a) / denom, -1.0, 1.0)
        angles[:, idx] = np.degrees(np.arccos(cosine))
    quality = 2.0 * np.sqrt(3.0) * np.abs(area2) / np.maximum(np.sum(lengths * lengths, axis=1), 1.0e-30)
    return {
        "signed_area": 0.5 * area2,
        "area": 0.5 * np.abs(area2),
        "edge_lengths": lengths,
        "angles_deg": angles,
        "quality": quality,
    }


def build_edge_topology(node_count: int, triangles: np.ndarray) -> EdgeTopology:
    """Build edge, neighbor, connected-component, and manifold topology."""
    triangles = np.asarray(triangles, dtype=int)
    edge_to_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)
    node_neighbors = [set() for _ in range(int(node_count))]
    for triangle_index, tri in enumerate(triangles):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a = int(a)
            b = int(b)
            edge = tuple(sorted((a, b)))
            edge_to_triangles[edge].append(int(triangle_index))
            node_neighbors[a].add(b)
            node_neighbors[b].add(a)

    triangle_neighbors = [set() for _ in range(len(triangles))]
    for attached in edge_to_triangles.values():
        if len(attached) == 2:
            a, b = attached
            triangle_neighbors[a].add(b)
            triangle_neighbors[b].add(a)
    neighbor_count = np.asarray([len(values) for values in triangle_neighbors], dtype=int)

    seen: set[int] = set()
    component_sizes: list[int] = []
    for start in range(len(triangles)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in triangle_neighbors[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component_sizes.append(size)

    return EdgeTopology(
        edge_to_triangles=dict(edge_to_triangles),
        node_neighbors=node_neighbors,
        triangle_neighbor_count=neighbor_count,
        boundary_edges=[edge for edge, attached in edge_to_triangles.items() if len(attached) == 1],
        nonmanifold_edges=[edge for edge, attached in edge_to_triangles.items() if len(attached) > 2],
        connected_component_sizes=sorted(component_sizes, reverse=True),
    )


def chain_edges(chains: Iterable[Iterable[int]]) -> set[tuple[int, int]]:
    """Return closed, undirected edges for model-boundary constraint chains."""
    edges: set[tuple[int, int]] = set()
    for chain_values in chains:
        chain = [int(value) for value in chain_values]
        if len(chain) < 2:
            continue
        for position, a in enumerate(chain):
            b = chain[(position + 1) % len(chain)]
            edges.add(tuple(sorted((a, b))))
    return edges


def constraint_integrity(
    topology: EdgeTopology,
    constraint_chains: Iterable[Iterable[int]] | None,
    open_boundary_nodes_zero_based: Iterable[int] | None,
    open_boundary_chains_zero_based: Iterable[Iterable[int]] | None = None,
    open_boundary_cyclic: Iterable[bool] | None = None,
) -> dict[str, Any]:
    """Audit protected chains and zero, one, or many ordered ocean OBCs."""
    mesh_edges = set(topology.edge_to_triangles)
    mesh_boundary_edges = set(topology.boundary_edges)
    protected = chain_edges(
        constraint_chains if constraint_chains is not None else []
    )
    missing = sorted(protected - mesh_edges)
    if open_boundary_chains_zero_based is None:
        legacy_nodes = [
            int(value)
            for value in (
                open_boundary_nodes_zero_based
                if open_boundary_nodes_zero_based is not None
                else []
            )
        ]
        open_chains = [legacy_nodes] if legacy_nodes else []
    else:
        open_chains = [
            [int(value) for value in chain_values]
            for chain_values in open_boundary_chains_zero_based
        ]
    cyclic_values = [
        bool(value)
        for value in (
            open_boundary_cyclic
            if open_boundary_cyclic is not None
            else []
        )
    ]
    if len(cyclic_values) < len(open_chains):
        cyclic_values.extend([False] * (len(open_chains) - len(cyclic_values)))

    chain_reports: list[dict[str, Any]] = []
    all_missing_open: list[tuple[int, int]] = []
    node_chain_membership: dict[int, list[int]] = defaultdict(list)
    for chain_index, open_nodes in enumerate(open_chains):
        cyclic = bool(cyclic_values[chain_index])
        minimum_node_count = 3 if cyclic else 2
        for node in set(open_nodes):
            node_chain_membership[node].append(chain_index)
        pairs = [tuple(sorted((a, b))) for a, b in zip(open_nodes[:-1], open_nodes[1:])]
        closing_edge: tuple[int, int] | None = None
        if cyclic and len(open_nodes) > 1:
            closing_edge = tuple(sorted((open_nodes[-1], open_nodes[0])))
            pairs.append(closing_edge)
        # An OBC pair must be an exterior mesh edge (one attached triangle);
        # accepting an arbitrary interior edge would hide a topology error.
        missing_open = [edge for edge in pairs if edge not in mesh_boundary_edges]
        all_missing_open.extend(missing_open)
        chain_reports.append(
            {
                "chain_index": int(chain_index),
                "cyclic": cyclic,
                "node_count": int(len(open_nodes)),
                "unique_node_count": int(len(set(open_nodes))),
                "minimum_node_count": minimum_node_count,
                "minimum_node_count_satisfied": bool(
                    len(open_nodes) >= minimum_node_count
                ),
                "missing_pair_count": int(len(missing_open)),
                "missing_pairs": [list(edge) for edge in missing_open[:100]],
                "cyclic_closure_edge": list(closing_edge) if closing_edge else None,
                "cyclic_closure_present": bool(
                    closing_edge is None or closing_edge in mesh_boundary_edges
                ),
                "ordered": bool(
                    len(open_nodes) >= minimum_node_count
                    and len(open_nodes) == len(set(open_nodes))
                    and not missing_open
                ),
            }
        )
    flattened_open = [node for chain in open_chains for node in chain]
    shared_open_nodes = sorted(
        node for node, memberships in node_chain_membership.items() if len(memberships) > 1
    )
    return {
        "protected_edge_count": int(len(protected)),
        "missing_protected_edge_count": int(len(missing)),
        "missing_protected_edges": [list(edge) for edge in missing[:100]],
        "open_boundary_chain_count": int(len(open_chains)),
        "open_boundary_chains": chain_reports,
        "open_boundary_node_count": int(len(flattened_open)),
        "open_boundary_unique_node_count": int(len(set(flattened_open))),
        "open_boundary_missing_pair_count": int(len(all_missing_open)),
        "open_boundary_missing_pairs": [list(edge) for edge in all_missing_open[:100]],
        "open_boundary_shared_node_count": int(len(shared_open_nodes)),
        "open_boundary_shared_nodes": shared_open_nodes[:100],
        "all_protected_edges_present": bool(not missing),
        "open_boundary_ordered": bool(
            not shared_open_nodes and all(report["ordered"] for report in chain_reports)
        ),
    }


def compute_mesh_metrics(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    depths: np.ndarray | None = None,
    constraint_chains: Iterable[Iterable[int]] | None = None,
    open_boundary_nodes_zero_based: Iterable[int] | None = None,
    open_boundary_chains_zero_based: Iterable[Iterable[int]] | None = None,
    open_boundary_cyclic: Iterable[bool] | None = None,
    target_size_by_triangle: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute OceanMesh, FVCOM, topology, and constraint metrics."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    geometry = triangle_geometry(points, triangles)
    topology = build_edge_topology(len(points), triangles)
    quality = geometry["quality"]
    angles = geometry["angles_deg"]
    areas = geometry["area"]
    signed_area = geometry["signed_area"]
    min_angles = np.min(angles, axis=1) if len(angles) else np.empty(0)
    max_angles = np.max(angles, axis=1) if len(angles) else np.empty(0)
    valence = np.asarray([len(values) for values in topology.node_neighbors], dtype=int)
    boundary_degree: dict[int, int] = defaultdict(int)
    for a, b in topology.boundary_edges:
        boundary_degree[a] += 1
        boundary_degree[b] += 1
    boundary_triangles = sorted(
        {triangle for edge in topology.boundary_edges for triangle in topology.edge_to_triangles[edge]}
    )

    area_changes: list[float] = []
    bathy_slopes: list[float] = []
    depth_values = np.asarray(depths, dtype=float) if depths is not None else None
    for (a, b), attached in topology.edge_to_triangles.items():
        if len(attached) == 2:
            area_a = float(areas[attached[0]])
            area_b = float(areas[attached[1]])
            area_changes.append(abs(area_a - area_b) / max(area_a, area_b, 1.0e-30))
        if depth_values is not None:
            distance = max(float(np.linalg.norm(points[a] - points[b])), 1.0e-30)
            bathy_slopes.append(abs(float(depth_values[a] - depth_values[b])) / distance)

    q_mean = float(np.mean(quality)) if len(quality) else 0.0
    q_std = float(np.std(quality)) if len(quality) else 0.0
    result: dict[str, Any] = {
        "schema_version": "fvcom_mesh_metrics_v2",
        "node_count": int(len(points)),
        "triangle_count": int(len(triangles)),
        "oceanmesh_quality": {
            "q_min": _minimum(quality),
            "q_mean": q_mean,
            "q_std": q_std,
            "q_l3_sigma": float(q_mean - 3.0 * q_std),
            "q_quantiles": _quantiles(quality),
            "count_q_below_0_10": int(np.sum(quality < 0.10)),
            "count_q_below_0_25": int(np.sum(quality < 0.25)),
            "boundary_count_q_below_0_25": int(np.sum(quality[boundary_triangles] < 0.25)) if boundary_triangles else 0,
        },
        "angles": {
            "min_angle_deg": _minimum(min_angles),
            "max_angle_deg": _maximum(max_angles),
            "min_angle_quantiles_deg": _quantiles(min_angles),
            "count_min_angle_below_5": int(np.sum(min_angles < 5.0)),
            "count_min_angle_below_20": int(np.sum(min_angles < 20.0)),
            "count_min_angle_below_30": int(np.sum(min_angles < 30.0)),
        },
        "topology": {
            "connected_component_count": int(len(topology.connected_component_sizes)),
            "connected_component_sizes": topology.connected_component_sizes[:20],
            "boundary_edge_count": int(len(topology.boundary_edges)),
            "boundary_node_count": int(len(boundary_degree)),
            "boundary_degree_anomaly_count": int(sum(value != 2 for value in boundary_degree.values())),
            "max_boundary_degree": int(max(boundary_degree.values(), default=0)),
            "singly_connected_triangle_count": int(np.sum(topology.triangle_neighbor_count == 1)),
            "nonmanifold_edge_count": int(len(topology.nonmanifold_edges)),
            "unused_node_count": int(len(points) - len(np.unique(triangles))) if len(triangles) else int(len(points)),
            "nonpositive_signed_area_count": int(np.sum(signed_area <= 0.0)),
        },
        "valence": {
            "definition": "unique_vertex_neighbors",
            "max_node_valence": int(np.max(valence)) if len(valence) else 0,
            "count_valence_above_6": int(np.sum(valence > 6)),
            "count_valence_above_8": int(np.sum(valence > 8)),
            "quantiles": _quantiles(valence.astype(float)),
        },
        "max_adjacent_area_change": float(max(area_changes, default=0.0)),
        "max_bathymetric_slope": float(max(bathy_slopes, default=0.0)) if depths is not None else None,
        "constraint_integrity": constraint_integrity(
            topology,
            constraint_chains,
            open_boundary_nodes_zero_based,
            open_boundary_chains_zero_based,
            open_boundary_cyclic,
        ),
    }
    if depth_values is not None:
        result["depths"] = {
            "finite": bool(np.all(np.isfinite(depth_values))),
            "positive": bool(len(depth_values) and float(np.nanmin(depth_values)) > 0.0),
            "minimum_m": float(np.nanmin(depth_values)) if len(depth_values) else None,
        }
    if target_size_by_triangle is not None:
        target = np.asarray(target_size_by_triangle, dtype=float)
        valid_target = bool(
            target.ndim == 1
            and len(target) == len(triangles)
            and np.all(np.isfinite(target))
            and np.all(target > 0.0)
        )
        result["size_error_l_over_h"] = {
            "valid": valid_target,
            "target_count": int(target.size),
            "maximum": 0.0,
            "quantiles": _quantiles(np.empty(0)),
            "count_above_1_55": 0,
            "count_above_2_0": 0,
        }
        if valid_target and len(triangles):
            size_error = np.max(geometry["edge_lengths"], axis=1) / target
            result["size_error_l_over_h"].update(
                {
                    "maximum": _maximum(size_error),
                    "quantiles": _quantiles(size_error),
                    "count_above_1_55": int(np.sum(size_error > 1.55)),
                    "count_above_2_0": int(np.sum(size_error > 2.0)),
                }
            )
    return result


def element_metric_arrays(points: np.ndarray, triangles: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-element arrays for QA GeoPackage output."""
    geometry = triangle_geometry(points, triangles)
    return {
        "quality_q": geometry["quality"],
        "min_angle_deg": np.min(geometry["angles_deg"], axis=1),
        "max_angle_deg": np.max(geometry["angles_deg"], axis=1),
        "area_m2": geometry["area"],
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"p01": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    quantiles = np.quantile(values, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {key: float(value) for key, value in zip(("p01", "p05", "p50", "p95", "p99"), quantiles)}


def _minimum(values: np.ndarray) -> float:
    return float(np.min(values)) if len(values) else 0.0


def _maximum(values: np.ndarray) -> float:
    return float(np.max(values)) if len(values) else 0.0
