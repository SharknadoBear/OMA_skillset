#!/usr/bin/env python3
"""Offline tests for the reusable FVCOM regional-refinement core."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refinement_core import (  # noqa: E402
    absorb_incident_edge_stars,
    boundary_lineage_audit,
    boundary_size_audit,
    canonical_hash,
    classify_patch,
    deterministic_new_ids,
    interpolate_bathymetry,
    protected_node_audit,
    quality_summary,
    scenario_depth,
    seam_audit,
    split_selected_boundary_chords,
    stitch_field_audit,
)


def structured_mesh(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray([(float(x), float(y)) for y in range(ny + 1) for x in range(nx + 1)])
    triangles = []
    for y in range(ny):
        for x in range(nx):
            lower_left = y * (nx + 1) + x
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            triangles.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
    return points, np.asarray(triangles, dtype=np.int64)


def test_interior_patch() -> dict:
    points, triangles = structured_mesh(6, 6)
    centroid = np.mean(points[triangles], axis=1)
    selection = (centroid[:, 0] >= 2) & (centroid[:, 0] <= 4) & (centroid[:, 1] >= 2) & (centroid[:, 1] <= 4)
    patch = classify_patch(points, triangles, selection)
    assert len(patch.loops) == 1
    assert len(patch.physical_edges) == 0
    assert len(patch.stitch_edges) > 0
    return {"loops": len(patch.loops), "stitch_edges": len(patch.stitch_edges)}


def test_coastline_island_patch() -> dict:
    points, triangles = structured_mesh(6, 6)
    centroid = np.mean(points[triangles], axis=1)
    island = (centroid[:, 0] > 2) & (centroid[:, 0] < 4) & (centroid[:, 1] > 2) & (centroid[:, 1] < 4)
    wet_triangles = triangles[~island]
    patch = classify_patch(points, wet_triangles, np.ones(len(wet_triangles), dtype=bool))
    assert len(patch.loops) == 2
    assert len(patch.stitch_edges) == 0
    assert len(patch.physical_edges) == len(patch.boundary_edges)
    return {"loops": len(patch.loops), "physical_edges": len(patch.physical_edges)}


def test_protected_ids_and_numbering() -> dict:
    points, _ = structured_mesh(3, 3)
    delivered = points.copy()
    audit = protected_node_audit(points, delivered, [1, 2, 15, 16])
    assert audit["pass"]
    new_points = np.asarray([[2.2, 1.1], [0.5, 0.7], [1.5, 1.5], [2.5, 2.5]])
    first = deterministic_new_ids(len(points), np.asarray([5, 9]), new_points)
    second = deterministic_new_ids(len(points), np.asarray([9, 5]), new_points.copy())
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == [5, 9, 16, 17]
    return {"protected": audit["node_ids"], "assigned_ids_zero_based": first.tolist()}


def test_seam_rejection_and_field_gate() -> dict:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]])
    rejected = seam_audit(triangles[:1], np.asarray([[0, 2]]), np.empty((0, 2), dtype=np.int64))
    assert not rejected["pass"]
    accepted = seam_audit(triangles, np.asarray([[0, 2]]), np.asarray([[0, 1], [1, 2], [2, 3], [0, 3]]))
    assert accepted["pass"]
    field = stitch_field_audit(np.asarray([[0, 2]]), np.ones(4), np.asarray([1.1, 1.0, 1.2, 1.0]))
    assert field["pass"]
    return {"rejection_detected": True, "field_ratio": field}


def test_bathymetry_and_scenario() -> dict:
    source_xy = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    source_depth = 2.0 + source_xy[:, 0] + 2.0 * source_xy[:, 1]
    query = np.asarray([[0.25, 0.5], [0.75, 0.25], [2.0, 2.0]])
    depth = interpolate_bathymetry(source_xy, source_depth, query)
    assert np.allclose(depth[:2], [3.25, 3.25])
    assert np.all(np.isfinite(depth))
    baseline = np.asarray([10.0, 12.0, 20.0])
    scenario = scenario_depth(baseline, np.asarray([True, True, False]), np.asarray([3.0, 12.0, 0.0]), 13.716)
    assert np.allclose(scenario, [11.0, 13.716, 20.0])
    return {"interpolated": depth.tolist(), "scenario": scenario.tolist()}


def test_repeatable_hash_and_quality() -> dict:
    points, triangles = structured_mesh(4, 4)
    metadata = {"algorithm": 6, "seed": 1}
    first = canonical_hash(points, triangles, metadata=metadata)
    second = canonical_hash(points.copy(), triangles.copy(), metadata=dict(metadata))
    assert first == second
    quality = quality_summary(points, triangles)
    assert quality["nonmanifold_edges"] == 0
    assert quality["nonpositive_elements"] == 0
    return {"sha256": first, "quality": quality}


def test_exact_chord_splitting_and_lineage() -> dict:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    target = np.full(4, 0.2)
    first = split_selected_boundary_chords(points, [(0, 1, 2, 3)], target, [(0, 1)], 1.5)
    second = split_selected_boundary_chords(points.copy(), [(0, 1, 2, 3)], target.copy(), [(1, 0)], 1.5)
    assert len(first.lineage) == 3
    assert np.array_equal(first.points, second.points)
    audit = boundary_lineage_audit(points, first)
    assert audit["pass"]
    delivered_target = np.r_[target, np.full(len(first.lineage), 0.2)]
    size = boundary_size_audit(first.points, first.boundary_edges[:4], delivered_target)
    assert size["pass"]
    assert canonical_hash(first.points, first.boundary_edges) == canonical_hash(second.points, second.boundary_edges)
    return {"inserted_nodes": len(first.lineage), "lineage": audit, "boundary_p95": size["p95"]}


def test_island_split_and_protected_rejection() -> dict:
    points = np.asarray(
        [[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0], [1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 1.0]]
    )
    loops = [(0, 1, 2, 3), (4, 5, 6, 7)]
    target = np.full(8, 0.8)
    result = split_selected_boundary_chords(points, loops, target, [(0, 1), (4, 5)], 1.5)
    assert len(result.loops) == 2
    assert boundary_lineage_audit(points, result)["pass"]
    rejected = False
    try:
        split_selected_boundary_chords(points, loops, target, [(0, 1)], 1.5, protected_nodes=[0])
    except ValueError:
        rejected = True
    assert rejected
    return {"outer_nodes": len(result.loops[0]), "island_nodes": len(result.loops[1]), "protected_rejection": rejected}


def test_augmented_incident_star() -> dict:
    _, triangles = structured_mesh(3, 3)
    selection = np.zeros(len(triangles), dtype=bool)
    selected_edge = np.asarray([[0, 1]], dtype=np.int64)
    augmented, additions = absorb_incident_edge_stars(triangles, selection, selected_edge)
    assert additions > 0
    assert np.any(augmented)
    patch = classify_patch(structured_mesh(3, 3)[0], triangles, augmented)
    assert len(patch.loops) == 1
    return {"selected_elements": int(np.sum(augmented)), "additions": additions}


def main() -> int:
    tests = {
        "interior_patch": test_interior_patch,
        "coastline_touching_patch_with_island": test_coastline_island_patch,
        "protected_ids_and_deterministic_numbering": test_protected_ids_and_numbering,
        "seam_rejection_and_field_gate": test_seam_rejection_and_field_gate,
        "bathymetry_interpolation": test_bathymetry_and_scenario,
        "repeatable_hashes": test_repeatable_hash_and_quality,
        "exact_chord_splitting_and_lineage": test_exact_chord_splitting_and_lineage,
        "island_split_and_protected_rejection": test_island_split_and_protected_rejection,
        "augmented_incident_star": test_augmented_incident_star,
    }
    results = {}
    for name, function in tests.items():
        results[name] = function()
    print(json.dumps({"status": "PASS", "tests": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
