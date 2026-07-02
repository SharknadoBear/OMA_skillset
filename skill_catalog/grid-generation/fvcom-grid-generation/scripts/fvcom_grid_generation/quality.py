"""FVCOM mesh-quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    min_angle_deg: float = 30.0
    max_angle_deg: float = 130.0
    max_bathy_slope: float = 0.1
    max_area_change: float = 0.5
    max_node_valence: int = 8


def evaluate_mesh_quality(
    nodes_xy: np.ndarray,
    depths: np.ndarray,
    triangles_1based: np.ndarray,
    open_boundary_nodes: np.ndarray,
    constraint_report: dict[str, Any],
    thresholds: QualityThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate FVCOM-oriented mesh quality gates."""
    thresholds = thresholds or QualityThresholds()
    tris = np.asarray(triangles_1based, dtype=int) - 1
    angles = []
    areas = []
    slopes = []
    valence = np.zeros(len(nodes_xy), dtype=int)
    edge_to_tri: dict[tuple[int, int], list[int]] = {}
    for tidx, tri in enumerate(tris):
        coords = nodes_xy[tri]
        tri_angles = _triangle_angles(coords)
        angles.extend(tri_angles.tolist())
        area = abs(float(np.cross(coords[1] - coords[0], coords[2] - coords[0]))) / 2.0
        areas.append(area)
        for node in tri:
            valence[int(node)] += 1
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_tri.setdefault(tuple(sorted((int(a), int(b)))), []).append(tidx)
            dz = abs(float(depths[a] - depths[b]))
            dist = max(float(np.linalg.norm(nodes_xy[a] - nodes_xy[b])), 1.0)
            slopes.append(dz / dist)
    area_changes = []
    for tids in edge_to_tri.values():
        if len(tids) == 2:
            a1 = areas[tids[0]]
            a2 = areas[tids[1]]
            area_changes.append(abs(a1 - a2) / max(a1, a2, 1.0e-12))
    failures = []
    min_angle = float(np.nanmin(angles)) if angles else 0.0
    max_angle = float(np.nanmax(angles)) if angles else 0.0
    max_slope = float(np.nanmax(slopes)) if slopes else 0.0
    max_area_change = float(np.nanmax(area_changes)) if area_changes else 0.0
    max_valence = int(np.nanmax(valence)) if valence.size else 0
    if min_angle < thresholds.min_angle_deg:
        failures.append("min_angle_below_threshold")
    if max_angle > thresholds.max_angle_deg:
        failures.append("max_angle_above_threshold")
    if max_slope > thresholds.max_bathy_slope:
        failures.append("bathymetric_slope_above_threshold")
    if max_area_change > thresholds.max_area_change:
        failures.append("adjacent_area_change_above_threshold")
    if max_valence > thresholds.max_node_valence:
        failures.append("node_valence_above_threshold")
    if not np.all(np.isfinite(depths)) or float(np.nanmin(depths)) <= 0.0:
        failures.append("nonpositive_or_nan_depth")
    if open_boundary_nodes.size == 0:
        failures.append("missing_open_boundary_nodestring")
    if not constraint_report.get("boundary_constraint_recovered", False):
        failures.append("boundary_constraint_not_recovered")
    return {
        "schema_version": "fvcom_mesh_quality_v1",
        "node_count": int(len(nodes_xy)),
        "triangle_count": int(len(tris)),
        "open_boundary_node_count": int(open_boundary_nodes.size),
        "min_angle_deg": min_angle,
        "max_angle_deg": max_angle,
        "max_bathymetric_slope": max_slope,
        "max_adjacent_area_change": max_area_change,
        "max_node_valence": max_valence,
        "thresholds": thresholds.__dict__,
        "constraint_recovery": constraint_report,
        "failure_taxonomy": failures,
        "accepted": not failures,
    }


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
