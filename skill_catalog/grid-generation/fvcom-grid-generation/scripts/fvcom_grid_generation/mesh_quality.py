"""FVCOM/SMS mesh quality diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class QualityThresholds:
    min_angle: float = 30.0
    max_angle: float = 130.0
    max_slope: float = 0.1
    max_area_change: float = 0.5
    max_connecting_elements: int = 8
    max_open_boundary_normal_deviation: float = 30.0


def evaluate_mesh_quality(
    nodes: np.ndarray,
    depths: np.ndarray,
    triangles: np.ndarray,
    open_boundary: np.ndarray | None = None,
    thresholds: QualityThresholds | None = None,
) -> dict:
    """Return SMS/FVCOM quality metrics and failed-element/node indices."""
    thresholds = thresholds or QualityThresholds()
    xy = lonlat_to_local_xy(nodes)
    tri0 = np.asarray(triangles, dtype=int) - 1
    areas = triangle_areas(xy, tri0)
    angles = triangle_angles(xy, tri0)
    connectivity = node_connectivity(len(nodes), tri0)
    slopes = triangle_depth_slopes(xy, depths, tri0)
    area_change = adjacent_area_change(tri0, areas)
    normality = open_boundary_normality(xy, tri0, open_boundary)

    failed_angle = np.flatnonzero(
        (np.nanmin(angles, axis=1) < thresholds.min_angle)
        | (np.nanmax(angles, axis=1) > thresholds.max_angle)
    ) + 1
    failed_slope = np.flatnonzero(slopes > thresholds.max_slope) + 1
    failed_area_change = np.flatnonzero(area_change > thresholds.max_area_change) + 1
    failed_connectivity = np.flatnonzero(connectivity > thresholds.max_connecting_elements) + 1
    failed_normality = np.asarray([], dtype=int)
    if normality["deviation_deg"].size:
        failed_normality = normality["segment_ids"][
            normality["deviation_deg"] > thresholds.max_open_boundary_normal_deviation
        ]

    return {
        "n_nodes": int(len(nodes)),
        "n_triangles": int(len(triangles)),
        "min_angle": float(np.nanmin(angles)),
        "max_angle": float(np.nanmax(angles)),
        "max_slope": float(np.nanmax(slopes)),
        "max_area_change": float(np.nanmax(area_change)) if area_change.size else 0.0,
        "max_connecting_elements": int(np.nanmax(connectivity)) if connectivity.size else 0,
        "open_boundary_max_normal_deviation": float(np.nanmax(normality["deviation_deg"]))
        if normality["deviation_deg"].size
        else 0.0,
        "failed_angle_elements": failed_angle,
        "failed_slope_elements": failed_slope,
        "failed_area_change_elements": failed_area_change,
        "failed_connectivity_nodes": failed_connectivity,
        "failed_open_boundary_segments": failed_normality,
    }


def lonlat_to_local_xy(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=float)
    lon = nodes[:, 0]
    lat = nodes[:, 1]
    lon0 = float(np.nanmean(lon))
    lat0 = float(np.nanmean(lat))
    x = EARTH_RADIUS_M * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = EARTH_RADIUS_M * np.radians(lat - lat0)
    return np.column_stack([x, y])


def triangle_areas(xy: np.ndarray, tri0: np.ndarray) -> np.ndarray:
    p = xy[tri0]
    return 0.5 * (
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )


def triangle_angles(xy: np.ndarray, tri0: np.ndarray) -> np.ndarray:
    p = xy[tri0]
    lengths = np.stack(
        [
            np.linalg.norm(p[:, 1] - p[:, 2], axis=1),
            np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
            np.linalg.norm(p[:, 0] - p[:, 1], axis=1),
        ],
        axis=1,
    )
    a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
    angles = np.empty((len(tri0), 3), dtype=float)
    angles[:, 0] = _law_of_cosines(b, c, a)
    angles[:, 1] = _law_of_cosines(a, c, b)
    angles[:, 2] = 180.0 - angles[:, 0] - angles[:, 1]
    return angles


def _law_of_cosines(a: np.ndarray, b: np.ndarray, opposite: np.ndarray) -> np.ndarray:
    denom = np.maximum(2.0 * a * b, 1.0e-12)
    cosang = np.clip((a * a + b * b - opposite * opposite) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def node_connectivity(n_nodes: int, tri0: np.ndarray) -> np.ndarray:
    counts = np.zeros(n_nodes, dtype=int)
    for node in tri0.ravel():
        counts[node] += 1
    return counts


def triangle_depth_slopes(xy: np.ndarray, depths: np.ndarray, tri0: np.ndarray) -> np.ndarray:
    p = xy[tri0]
    h = np.asarray(depths, dtype=float)[tri0]
    max_depth_range = np.nanmax(h, axis=1) - np.nanmin(h, axis=1)
    edges = np.stack(
        [
            np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
            np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
            np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
        ],
        axis=1,
    )
    return max_depth_range / np.maximum(np.nanmax(edges, axis=1), 1.0)


def adjacent_area_change(tri0: np.ndarray, areas: np.ndarray) -> np.ndarray:
    edge_to_tri: dict[tuple[int, int], list[int]] = {}
    for tid, tri in enumerate(tri0):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_to_tri.setdefault(edge, []).append(tid)
    change = np.zeros(len(tri0), dtype=float)
    abs_area = np.abs(areas)
    for tids in edge_to_tri.values():
        if len(tids) != 2:
            continue
        a0, a1 = abs_area[tids[0]], abs_area[tids[1]]
        metric = abs(a0 - a1) / max(a0, a1, 1.0)
        change[tids[0]] = max(change[tids[0]], metric)
        change[tids[1]] = max(change[tids[1]], metric)
    return change


def open_boundary_normality(xy: np.ndarray, tri0: np.ndarray, open_boundary: np.ndarray | None) -> dict:
    if open_boundary is None or len(open_boundary) < 2:
        return {"segment_ids": np.asarray([], dtype=int), "deviation_deg": np.asarray([], dtype=float)}
    ob0 = np.asarray(open_boundary, dtype=int) - 1
    edge_to_tri: dict[tuple[int, int], list[int]] = {}
    for tid, tri in enumerate(tri0):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_tri.setdefault(tuple(sorted((int(a), int(b)))), []).append(tid)

    seg_ids = []
    deviations = []
    for sid, (a, b) in enumerate(zip(ob0[:-1], ob0[1:]), start=1):
        tids = edge_to_tri.get(tuple(sorted((int(a), int(b)))), [])
        if not tids:
            continue
        tri = tri0[tids[0]]
        others = [node for node in tri if node not in {a, b}]
        if not others:
            continue
        tangent = xy[b] - xy[a]
        inward = xy[others[0]] - 0.5 * (xy[a] + xy[b])
        if np.linalg.norm(tangent) == 0.0 or np.linalg.norm(inward) == 0.0:
            continue
        cosang = np.dot(tangent, inward) / (np.linalg.norm(tangent) * np.linalg.norm(inward))
        angle_to_tangent = abs(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
        deviation = abs(90.0 - angle_to_tangent)
        seg_ids.append(sid)
        deviations.append(deviation)
    return {
        "segment_ids": np.asarray(seg_ids, dtype=int),
        "deviation_deg": np.asarray(deviations, dtype=float),
    }
