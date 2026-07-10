"""Before/after mesh-quality comparison helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compare_quality_documents(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Build the controlled geometric-improvement comparison contract."""
    fields = {
        "q_l3_sigma": (
            before["oceanmesh_quality"]["q_l3_sigma"],
            after["oceanmesh_quality"]["q_l3_sigma"],
            "increase",
        ),
        "q_p01": (
            before["oceanmesh_quality"]["q_quantiles"]["p01"],
            after["oceanmesh_quality"]["q_quantiles"]["p01"],
            "increase",
        ),
        "min_angle_p01": (
            before["angle_statistics"]["min_angle_quantiles_deg"]["p01"],
            after["angle_statistics"]["min_angle_quantiles_deg"]["p01"],
            "increase",
        ),
        "count_q_below_0_25": (
            before["oceanmesh_quality"]["count_q_below_0_25"],
            after["oceanmesh_quality"]["count_q_below_0_25"],
            "decrease",
        ),
        "count_min_angle_below_30": (
            before["angle_statistics"]["count_min_angle_below_30"],
            after["angle_statistics"]["count_min_angle_below_30"],
            "decrease",
        ),
        "boundary_degree_anomalies": (
            before["topology"]["boundary_degree_anomaly_count"],
            after["topology"]["boundary_degree_anomaly_count"],
            "decrease",
        ),
        "singly_connected": (
            before["topology"]["singly_connected_triangle_count"],
            after["topology"]["singly_connected_triangle_count"],
            "decrease",
        ),
        "count_valence_above_8": (
            before["valence"]["count_valence_above_8"],
            after["valence"]["count_valence_above_8"],
            "decrease",
        ),
    }
    metrics: dict[str, Any] = {}
    for name, (old, new, direction) in fields.items():
        metrics[name] = {
            "before": float(old),
            "after": float(new),
            "delta": float(new - old),
            "direction": direction,
            "improved": bool(new > old if direction == "increase" else new < old),
        }
    invariants = {
        "protected_boundaries_preserved": bool(after["constraint_integrity"]["all_protected_edges_present"]),
        "open_boundary_ordered": bool(after["constraint_integrity"]["open_boundary_ordered"]),
        "single_component": bool(after["topology"]["connected_component_count"] == 1),
        "manifold": bool(after["topology"]["nonmanifold_edge_count"] == 0),
        "nonpositive_area_not_increased": bool(
            after["topology"]["nonpositive_signed_area_count"]
            <= before["topology"]["nonpositive_signed_area_count"]
        ),
        "finite_positive_depths": bool(after.get("depths", {}).get("finite") and after.get("depths", {}).get("positive")),
    }
    return {
        "schema_version": "fvcom_mesh_quality_comparison_v1",
        "metrics": metrics,
        "invariants": invariants,
        "strict_improvement_count": int(sum(value["improved"] for value in metrics.values())),
        "all_invariants_pass": bool(all(invariants.values())),
        "all_requested_metrics_improved": bool(all(value["improved"] for value in metrics.values())),
    }


def write_quality_comparison_plot(
    path: str | Path,
    before: dict[str, Any],
    after: dict[str, Any],
    title: str,
) -> Path:
    """Write a compact before/after quality-tail and defect plot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    quality_names = ["q L3sigma", "q p01", "angle p01 / 60"]
    quality_before = [
        before["oceanmesh_quality"]["q_l3_sigma"],
        before["oceanmesh_quality"]["q_quantiles"]["p01"],
        before["angle_statistics"]["min_angle_quantiles_deg"]["p01"] / 60.0,
    ]
    quality_after = [
        after["oceanmesh_quality"]["q_l3_sigma"],
        after["oceanmesh_quality"]["q_quantiles"]["p01"],
        after["angle_statistics"]["min_angle_quantiles_deg"]["p01"] / 60.0,
    ]
    defect_names = ["q<0.25", "angle<30", "singly", "valence>8", "boundary degree"]
    defect_before = [
        before["oceanmesh_quality"]["count_q_below_0_25"],
        before["angle_statistics"]["count_min_angle_below_30"],
        before["topology"]["singly_connected_triangle_count"],
        before["valence"]["count_valence_above_8"],
        before["topology"]["boundary_degree_anomaly_count"],
    ]
    defect_after = [
        after["oceanmesh_quality"]["count_q_below_0_25"],
        after["angle_statistics"]["count_min_angle_below_30"],
        after["topology"]["singly_connected_triangle_count"],
        after["valence"]["count_valence_above_8"],
        after["topology"]["boundary_degree_anomaly_count"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x_quality = np.arange(len(quality_names))
    axes[0].bar(x_quality - 0.18, quality_before, width=0.36, label="preclean", color="#9aa4af")
    axes[0].bar(x_quality + 0.18, quality_after, width=0.36, label="postclean", color="#1f77b4")
    axes[0].set_xticks(x_quality, quality_names)
    axes[0].set_ylabel("normalized quality")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    x_defect = np.arange(len(defect_names))
    axes[1].bar(x_defect - 0.18, defect_before, width=0.36, label="preclean", color="#d98b8b")
    axes[1].bar(x_defect + 0.18, defect_after, width=0.36, label="postclean", color="#2ca02c")
    axes[1].set_xticks(x_defect, defect_names, rotation=20, ha="right")
    axes[1].set_ylabel("defect count")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.suptitle(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path
