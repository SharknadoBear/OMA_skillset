#!/usr/bin/env python3
"""Deterministic core helpers for FVCOM regional mesh-refinement adapters."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


@dataclass(frozen=True)
class PatchTopology:
    selection: np.ndarray
    selected_nodes: np.ndarray
    boundary_edges: np.ndarray
    physical_edges: np.ndarray
    stitch_edges: np.ndarray
    loops: tuple[tuple[int, ...], ...]
    topology_closure_elements: int


@dataclass(frozen=True)
class BoundarySplitResult:
    points: np.ndarray
    loops: tuple[tuple[int, ...], ...]
    boundary_edges: np.ndarray
    lineage: tuple[dict[str, object], ...]


def unique_edges(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(triangles, dtype=np.int64)
    edges = np.sort(np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
    return np.unique(edges, axis=0, return_counts=True)


def signed_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    vertices = np.asarray(points, dtype=float)[np.asarray(triangles, dtype=np.int64)]
    first = vertices[:, 1] - vertices[:, 0]
    second = vertices[:, 2] - vertices[:, 0]
    return 0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])


def orient_positive(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    result = np.asarray(triangles, dtype=np.int64).copy()
    flip = signed_areas(points, result) < 0.0
    result[flip, 1], result[flip, 2] = result[flip, 2].copy(), result[flip, 1].copy()
    return result


def triangle_quality(points: np.ndarray, triangles: np.ndarray) -> dict[str, np.ndarray]:
    vertices = np.asarray(points, dtype=float)[np.asarray(triangles, dtype=np.int64)]
    edge = np.column_stack(
        (
            np.linalg.norm(vertices[:, 2] - vertices[:, 1], axis=1),
            np.linalg.norm(vertices[:, 0] - vertices[:, 2], axis=1),
            np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1),
        )
    )
    area = np.abs(signed_areas(points, triangles))
    quality = 4.0 * np.sqrt(3.0) * area / np.sum(edge * edge, axis=1)

    def angle(opposite: np.ndarray, side_a: np.ndarray, side_b: np.ndarray) -> np.ndarray:
        cosine = (side_a * side_a + side_b * side_b - opposite * opposite) / (2.0 * side_a * side_b)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    angles = np.column_stack(
        (
            angle(edge[:, 0], edge[:, 1], edge[:, 2]),
            angle(edge[:, 1], edge[:, 0], edge[:, 2]),
            angle(edge[:, 2], edge[:, 0], edge[:, 1]),
        )
    )
    return {"edge_length": edge, "area": area, "quality": quality, "minimum_angle": np.min(angles, axis=1)}


def quality_summary(points: np.ndarray, triangles: np.ndarray) -> dict[str, object]:
    triangles = orient_positive(points, triangles)
    metrics = triangle_quality(points, triangles)
    edges, counts = unique_edges(triangles)
    valence = np.bincount(edges.ravel(), minlength=len(points))
    return {
        "nodes": int(len(points)),
        "elements": int(len(triangles)),
        "nonpositive_elements": int(np.sum(signed_areas(points, triangles) <= 0.0)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "maximum_valence": int(np.max(valence)),
        "q_min": float(np.min(metrics["quality"])),
        "minimum_angle_deg": float(np.min(metrics["minimum_angle"])),
        "q_below_0p10": int(np.sum(metrics["quality"] < 0.10)),
        "angle_below_5deg": int(np.sum(metrics["minimum_angle"] < 5.0)),
    }


def _loop_area(points: np.ndarray, loop: Iterable[int]) -> float:
    indices = np.asarray(tuple(loop), dtype=np.int64)
    xy = points[indices]
    return 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1)))


def order_boundary_loops(points: np.ndarray, boundary_edges: np.ndarray) -> tuple[tuple[int, ...], ...]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in np.asarray(boundary_edges, dtype=np.int64):
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))
    bad = [node for node, neighbors in adjacency.items() if len(neighbors) != 2]
    if bad:
        raise ValueError(f"boundary graph is not degree two at nodes {bad[:10]}")
    remaining = {tuple(sorted((int(a), int(b)))) for a, b in boundary_edges}
    loops: list[list[int]] = []
    while remaining:
        start, second = min(remaining)
        remaining.remove((start, second))
        loop = [start, second]
        previous, current = start, second
        while current != start:
            nxt = next(node for node in adjacency[current] if node != previous)
            edge = tuple(sorted((current, nxt)))
            if edge not in remaining:
                if nxt == start:
                    break
                raise ValueError("boundary loop revisited an edge")
            remaining.remove(edge)
            loop.append(nxt)
            previous, current = current, nxt
        if loop[-1] == start:
            loop.pop()
        loops.append(loop)
    loops.sort(key=lambda item: abs(_loop_area(points, item)), reverse=True)
    for index, loop in enumerate(loops):
        area = _loop_area(points, loop)
        if (index == 0 and area < 0.0) or (index > 0 and area > 0.0):
            loops[index] = list(reversed(loop))
    return tuple(tuple(loop) for loop in loops)


def close_element_selection(triangles: np.ndarray, selection: np.ndarray, maximum_passes: int = 50) -> tuple[np.ndarray, int]:
    triangles = np.asarray(triangles, dtype=np.int64)
    result = np.asarray(selection, dtype=bool).copy()
    additions_total = 0
    for _ in range(maximum_passes):
        edges, counts = unique_edges(triangles[result])
        boundary = edges[counts == 1]
        degree = np.bincount(boundary.ravel(), minlength=int(np.max(triangles)) + 1)
        ambiguous = np.flatnonzero((degree > 0) & (degree != 2))
        if len(ambiguous) == 0:
            return result, additions_total
        additions = np.any(np.isin(triangles, ambiguous), axis=1) & ~result
        if not np.any(additions):
            raise ValueError("element-aligned patch cannot be topologically closed")
        additions_total += int(np.sum(additions))
        result[additions] = True
    raise ValueError("patch topology closure exceeded the pass limit")


def classify_patch(points: np.ndarray, triangles: np.ndarray, selection: np.ndarray) -> PatchTopology:
    selection, closure = close_element_selection(triangles, selection)
    patch_triangles = np.asarray(triangles, dtype=np.int64)[selection]
    selected_nodes = np.unique(patch_triangles)
    patch_edges, patch_counts = unique_edges(patch_triangles)
    boundary = patch_edges[patch_counts == 1]
    all_edges, all_counts = unique_edges(triangles)
    physical_set = {tuple(map(int, edge)) for edge in all_edges[all_counts == 1]}
    physical_mask = np.asarray([tuple(map(int, edge)) in physical_set for edge in boundary])
    return PatchTopology(
        selection=selection,
        selected_nodes=selected_nodes,
        boundary_edges=boundary,
        physical_edges=boundary[physical_mask],
        stitch_edges=boundary[~physical_mask],
        loops=order_boundary_loops(points, boundary),
        topology_closure_elements=closure,
    )


def graph_lower_envelope(node_count: int, edges: np.ndarray, points: np.ndarray, initial: np.ndarray, gradation: float) -> np.ndarray:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(node_count)]
    for a, b in np.asarray(edges, dtype=np.int64):
        length = float(np.linalg.norm(points[int(a)] - points[int(b)]))
        adjacency[int(a)].append((int(b), length))
        adjacency[int(b)].append((int(a), length))
    values = np.asarray(initial, dtype=float).copy()
    queue = [(float(value), index) for index, value in enumerate(values) if np.isfinite(value)]
    heapq.heapify(queue)
    while queue:
        value, node = heapq.heappop(queue)
        if value > values[node] + 1e-12:
            continue
        for neighbor, length in adjacency[node]:
            candidate = value + gradation * length
            if candidate + 1e-12 < values[neighbor]:
                values[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return values


def stitch_field_audit(stitch_edges: np.ndarray, incumbent: np.ndarray, target: np.ndarray) -> dict[str, float | bool]:
    nodes = np.unique(np.asarray(stitch_edges, dtype=np.int64))
    ratio = np.maximum(target[nodes] / incumbent[nodes], incumbent[nodes] / target[nodes])
    p95, maximum = float(np.quantile(ratio, 0.95)), float(np.max(ratio))
    return {"p95": p95, "maximum": maximum, "pass": bool(p95 <= 1.5 and maximum <= 2.0)}


def boundary_size_audit(
    points: np.ndarray,
    boundary_edges: np.ndarray,
    target_size: np.ndarray,
    p95_limit: float = 1.5,
    maximum_limit: float = 2.0,
) -> dict[str, object]:
    edges = np.asarray(boundary_edges, dtype=np.int64)
    length = np.linalg.norm(np.asarray(points, dtype=float)[edges[:, 1]] - np.asarray(points, dtype=float)[edges[:, 0]], axis=1)
    h_mid = 0.5 * (np.asarray(target_size, dtype=float)[edges[:, 0]] + np.asarray(target_size, dtype=float)[edges[:, 1]])
    if np.any(h_mid <= 0.0):
        raise ValueError("boundary target sizes must be positive")
    ratio = length / h_mid
    p95 = float(np.quantile(ratio, 0.95))
    maximum = float(np.max(ratio))
    return {
        "edge_count": int(len(edges)),
        "length_m": length,
        "h_mid_m": h_mid,
        "l_over_h": ratio,
        "p95": p95,
        "maximum": maximum,
        "p95_limit": float(p95_limit),
        "maximum_limit": float(maximum_limit),
        "pass": bool(p95 <= p95_limit and maximum <= maximum_limit),
    }


def absorb_incident_edge_stars(
    triangles: np.ndarray,
    selection: np.ndarray,
    selected_edges: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Absorb complete wet-element stars for selected source edges, then close."""
    triangles = np.asarray(triangles, dtype=np.int64)
    result = np.asarray(selection, dtype=bool).copy()
    before = int(np.sum(result))
    for a, b in np.asarray(selected_edges, dtype=np.int64):
        result[np.any(triangles == int(a), axis=1) & np.any(triangles == int(b), axis=1)] = True
    result, _ = close_element_selection(triangles, result)
    return result, int(np.sum(result) - before)


def split_selected_boundary_chords(
    points: np.ndarray,
    loops: Iterable[Iterable[int]],
    target_size: np.ndarray,
    selected_edges: Iterable[tuple[int, int]],
    maximum_edge_to_target_ratio: float = 1.5,
    protected_nodes: Iterable[int] = (),
) -> BoundarySplitResult:
    """Split selected loop chords exactly while preserving every source vertex."""
    if maximum_edge_to_target_ratio <= 0.0:
        raise ValueError("maximum_edge_to_target_ratio must be positive")
    source_points = np.asarray(points, dtype=float)
    target = np.asarray(target_size, dtype=float)
    selected = {tuple(sorted(map(int, edge))) for edge in selected_edges}
    protected = {int(value) for value in protected_nodes}
    loop_collection = tuple(tuple(map(int, values)) for values in loops)
    delivered_points = [row.copy() for row in source_points]
    delivered_loops: list[tuple[int, ...]] = []
    lineage: list[dict[str, object]] = []
    delivered_edges: list[tuple[int, int]] = []
    for loop_index, loop in enumerate(loop_collection):
        augmented: list[int] = []
        for index, node in enumerate(loop):
            nxt = loop[(index + 1) % len(loop)]
            augmented.append(node)
            key = tuple(sorted((node, nxt)))
            if key not in selected:
                continue
            if node in protected or nxt in protected:
                raise ValueError(f"selected boundary edge {node}-{nxt} is incident to a protected node")
            length = float(np.linalg.norm(source_points[nxt] - source_points[node]))
            h_mid = 0.5 * float(target[node] + target[nxt])
            if h_mid <= 0.0:
                raise ValueError("boundary target sizes must be positive")
            segments = int(np.ceil(length / (maximum_edge_to_target_ratio * h_mid)))
            for segment_index in range(1, max(segments, 1)):
                fraction = segment_index / float(segments)
                coordinate = (1.0 - fraction) * source_points[node] + fraction * source_points[nxt]
                new_node = len(delivered_points)
                delivered_points.append(coordinate)
                augmented.append(new_node)
                lineage.append(
                    {
                        "new_node": new_node,
                        "parent_a": node,
                        "parent_b": nxt,
                        "fraction": fraction,
                        "loop_index": loop_index,
                        "geometry_error": 0.0,
                    }
                )
        delivered_loops.append(tuple(augmented))
        delivered_edges.extend(
            (augmented[index], augmented[(index + 1) % len(augmented)]) for index in range(len(augmented))
        )
    missing = selected - {
        tuple(sorted((loop[index], loop[(index + 1) % len(loop)])))
        for loop in loop_collection
        for index in range(len(loop))
    }
    if missing:
        raise ValueError(f"selected edges are not present on supplied loops: {sorted(missing)[:10]}")
    return BoundarySplitResult(
        points=np.asarray(delivered_points, dtype=float),
        loops=tuple(delivered_loops),
        boundary_edges=np.asarray(delivered_edges, dtype=np.int64),
        lineage=tuple(lineage),
    )


def boundary_lineage_audit(
    source_points: np.ndarray,
    result: BoundarySplitResult,
) -> dict[str, object]:
    errors: list[float] = []
    for item in result.lineage:
        expected = (
            (1.0 - float(item["fraction"])) * source_points[int(item["parent_a"])]
            + float(item["fraction"]) * source_points[int(item["parent_b"])]
        )
        errors.append(float(np.linalg.norm(result.points[int(item["new_node"])] - expected)))
    return {
        "inserted_nodes": len(result.lineage),
        "maximum_geometry_error": max(errors, default=0.0),
        "source_vertices_preserved": bool(np.array_equal(result.points[: len(source_points)], source_points)),
        "pass": bool(max(errors, default=0.0) == 0.0 and np.array_equal(result.points[: len(source_points)], source_points)),
    }


def deterministic_new_ids(source_node_count: int, removable_ids: np.ndarray, new_points: np.ndarray) -> np.ndarray:
    removable = np.sort(np.asarray(removable_ids, dtype=np.int64))
    order = np.lexsort((new_points[:, 0], new_points[:, 1]))
    if len(new_points) < len(removable):
        raise ValueError("new patch has fewer interior nodes than the vacated source IDs")
    extra = len(new_points) - len(removable)
    assigned = np.concatenate((removable, np.arange(source_node_count, source_node_count + extra, dtype=np.int64)))
    result = np.empty(len(new_points), dtype=np.int64)
    result[order] = assigned
    return result


def seam_audit(triangles: np.ndarray, stitch_edges: np.ndarray, physical_edges: np.ndarray) -> dict[str, object]:
    edges, counts = unique_edges(triangles)
    lookup = {tuple(map(int, edge)): int(count) for edge, count in zip(edges, counts)}
    bad_stitch = [tuple(map(int, edge)) for edge in stitch_edges if lookup.get(tuple(sorted(map(int, edge))), 0) != 2]
    bad_physical = [tuple(map(int, edge)) for edge in physical_edges if lookup.get(tuple(sorted(map(int, edge))), 0) != 1]
    return {"bad_stitch_edges": bad_stitch, "bad_physical_edges": bad_physical, "pass": not bad_stitch and not bad_physical}


def protected_node_audit(source_points: np.ndarray, delivered_points: np.ndarray, protected_ids_one_based: Iterable[int]) -> dict[str, object]:
    indices = np.asarray(tuple(protected_ids_one_based), dtype=np.int64) - 1
    error = np.linalg.norm(source_points[indices] - delivered_points[indices], axis=1)
    return {"node_ids": (indices + 1).tolist(), "maximum_coordinate_error": float(np.max(error)), "pass": bool(np.max(error) == 0.0)}


def interpolate_bathymetry(source_xy: np.ndarray, source_depth: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    linear = LinearNDInterpolator(np.asarray(source_xy, dtype=float), np.asarray(source_depth, dtype=float), fill_value=np.nan)
    result = np.asarray(linear(np.asarray(query_xy, dtype=float)), dtype=float)
    missing = ~np.isfinite(result)
    if np.any(missing):
        nearest = NearestNDInterpolator(np.asarray(source_xy, dtype=float), np.asarray(source_depth, dtype=float))
        result[missing] = nearest(np.asarray(query_xy, dtype=float)[missing])
    return result


def scenario_depth(baseline: np.ndarray, inside_mask: np.ndarray, distance_inside: np.ndarray, target_m: float, slope_ratio: float = 3.0) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=float)
    result = baseline.copy()
    inside = np.asarray(inside_mask, dtype=bool)
    result[inside] = np.maximum(
        baseline[inside],
        np.minimum(float(target_m), baseline[inside] + np.asarray(distance_inside, dtype=float)[inside] / float(slope_ratio)),
    )
    return result


def canonical_hash(*arrays: np.ndarray, metadata: dict[str, object] | None = None) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    if metadata is not None:
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()
