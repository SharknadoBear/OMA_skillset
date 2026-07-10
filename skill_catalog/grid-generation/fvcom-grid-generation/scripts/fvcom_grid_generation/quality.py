"""FVCOM and OceanMesh-style mesh-quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import compute_mesh_metrics


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
    *,
    constraint_chains: list[list[int]] | None = None,
    target_size_by_triangle: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate final FVCOM gates plus OceanMesh and topology diagnostics."""
    thresholds = thresholds or QualityThresholds()
    triangles = np.asarray(triangles_1based, dtype=int) - 1
    open_zero = (np.asarray(open_boundary_nodes, dtype=int) - 1).tolist()
    metrics = compute_mesh_metrics(
        np.asarray(nodes_xy, dtype=float),
        triangles,
        depths=np.asarray(depths, dtype=float),
        constraint_chains=constraint_chains,
        open_boundary_nodes_zero_based=open_zero,
        target_size_by_triangle=target_size_by_triangle,
    )
    min_angle = float(metrics["angles"]["min_angle_deg"])
    max_angle = float(metrics["angles"]["max_angle_deg"])
    max_slope = float(metrics["max_bathymetric_slope"] or 0.0)
    max_area_change = float(metrics["max_adjacent_area_change"])
    max_valence = int(metrics["valence"]["max_node_valence"])
    topology = metrics["topology"]
    integrity = metrics["constraint_integrity"]
    depth_report = metrics["depths"]

    failures: list[str] = []
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
    if not depth_report["finite"] or not depth_report["positive"]:
        failures.append("nonpositive_or_nan_depth")
    if len(open_zero) == 0:
        failures.append("missing_open_boundary_nodestring")
    elif not integrity["open_boundary_ordered"]:
        failures.append("open_boundary_nodestring_not_ordered_on_mesh")
    if not constraint_report.get("boundary_constraint_recovered", False):
        failures.append("boundary_constraint_not_recovered")
    if not integrity["all_protected_edges_present"]:
        failures.append("protected_boundary_constraint_missing")
    if topology["connected_component_count"] != 1:
        failures.append("multiple_mesh_components")
    if topology["nonmanifold_edge_count"]:
        failures.append("nonmanifold_edges_present")
    if topology["boundary_degree_anomaly_count"]:
        failures.append("boundary_not_traversable")
    if topology["singly_connected_triangle_count"]:
        failures.append("singly_connected_elements_present")
    if topology["nonpositive_signed_area_count"]:
        failures.append("nonpositive_triangle_area")

    return {
        "schema_version": "fvcom_mesh_quality_v2",
        "node_count": int(metrics["node_count"]),
        "triangle_count": int(metrics["triangle_count"]),
        "open_boundary_node_count": int(len(open_zero)),
        "min_angle_deg": min_angle,
        "max_angle_deg": max_angle,
        "max_bathymetric_slope": max_slope,
        "max_adjacent_area_change": max_area_change,
        "max_node_valence": max_valence,
        "thresholds": thresholds.__dict__,
        "constraint_recovery": constraint_report,
        "oceanmesh_quality": metrics["oceanmesh_quality"],
        "angle_statistics": metrics["angles"],
        "topology": topology,
        "valence": metrics["valence"],
        "constraint_integrity": integrity,
        "size_error_l_over_h": metrics.get("size_error_l_over_h"),
        "depths": depth_report,
        "failure_taxonomy": failures,
        "accepted": not failures,
    }
