"""FVCOM and OceanMesh-style mesh-quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import compute_mesh_metrics


@dataclass(frozen=True)
class QualityThresholds:
    min_q_l3_sigma: float = 0.75
    min_angle_deg: float = 30.0
    max_angle_deg: float = 130.0
    max_bathy_slope: float = 0.1
    max_area_change: float = 0.5
    max_node_valence: int = 8
    max_size_error_p95: float = 1.55
    max_size_error: float = 2.0


def evaluate_mesh_quality(
    nodes_xy: np.ndarray,
    depths: np.ndarray,
    triangles_1based: np.ndarray,
    open_boundary_nodes: np.ndarray,
    constraint_report: dict[str, Any],
    thresholds: QualityThresholds | None = None,
    *,
    constraint_chains: list[list[int]] | None = None,
    open_boundary_chains: list[list[int]] | None = None,
    open_boundary_cyclic: list[bool] | None = None,
    require_open_boundary: bool = True,
    expected_open_boundary_count: int | None = None,
    enforce_size_error: bool = False,
    enforce_no_unused_nodes: bool = False,
    target_size_by_triangle: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate final FVCOM gates plus OceanMesh and topology diagnostics."""
    thresholds = thresholds or QualityThresholds()
    triangles = np.asarray(triangles_1based, dtype=int) - 1
    open_zero = (np.asarray(open_boundary_nodes, dtype=int) - 1).tolist()
    open_chains_zero = None
    if open_boundary_chains is not None:
        open_chains_zero = [
            (np.asarray(values, dtype=int) - 1).tolist()
            for values in open_boundary_chains
        ]
    metrics = compute_mesh_metrics(
        np.asarray(nodes_xy, dtype=float),
        triangles,
        depths=np.asarray(depths, dtype=float),
        constraint_chains=constraint_chains,
        open_boundary_nodes_zero_based=open_zero,
        open_boundary_chains_zero_based=open_chains_zero,
        open_boundary_cyclic=open_boundary_cyclic,
        target_size_by_triangle=target_size_by_triangle,
    )
    min_angle = float(metrics["angles"]["min_angle_deg"])
    max_angle = float(metrics["angles"]["max_angle_deg"])
    max_slope = float(metrics["max_bathymetric_slope"] or 0.0)
    max_area_change = float(metrics["max_adjacent_area_change"])
    max_valence = int(metrics["valence"]["max_node_valence"])
    q_l3_sigma = float(metrics["oceanmesh_quality"].get("q_l3_sigma", float("-inf")))
    topology = metrics["topology"]
    integrity = metrics["constraint_integrity"]
    depth_report = metrics["depths"]

    failures: list[str] = []
    if (
        int(metrics["oceanmesh_quality"].get("count_q_below_0_10", 0)) > 0
        or int(metrics["angles"].get("count_min_angle_below_5", 0)) > 0
    ):
        failures.append("superthin_elements_present")
    if not np.isfinite(q_l3_sigma) or q_l3_sigma <= float(thresholds.min_q_l3_sigma):
        failures.append("q_l3_sigma_below_threshold")
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
    open_chain_count = int(integrity["open_boundary_chain_count"])
    open_node_count = int(integrity["open_boundary_node_count"])
    if expected_open_boundary_count is not None:
        if open_chain_count != int(expected_open_boundary_count):
            failures.append("open_boundary_chain_count_mismatch")
    elif require_open_boundary and open_chain_count == 0:
        failures.append("missing_open_boundary_nodestring")
    if open_chain_count and not integrity["open_boundary_ordered"]:
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
    if enforce_no_unused_nodes and topology["unused_node_count"]:
        failures.append("unused_mesh_nodes_present")
    size_error = metrics.get("size_error_l_over_h")
    if enforce_size_error:
        if not size_error:
            failures.append("missing_target_size_error_diagnostic")
        elif not bool(size_error.get("valid", False)):
            failures.append("target_size_by_triangle_invalid")
        else:
            p95 = float(size_error["quantiles"]["p95"])
            maximum = float(size_error["maximum"])
            if p95 > thresholds.max_size_error_p95:
                failures.append("target_size_l_over_h_p95_above_threshold")
            if maximum > thresholds.max_size_error:
                failures.append("target_size_l_over_h_max_above_threshold")

    return {
        "schema_version": "fvcom_mesh_quality_v2",
        "node_count": int(metrics["node_count"]),
        "triangle_count": int(metrics["triangle_count"]),
        "open_boundary_chain_count": open_chain_count,
        "open_boundary_node_count": open_node_count,
        "open_boundary_required": bool(require_open_boundary),
        "expected_open_boundary_chain_count": (
            int(expected_open_boundary_count)
            if expected_open_boundary_count is not None
            else None
        ),
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
