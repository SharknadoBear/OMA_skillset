#!/usr/bin/env python3
"""Synthetic tests for agent-reviewed visual superthin repair."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.visual_superthin import (  # noqa: E402
    PLAN_SCHEMA,
    _action_candidates,
    _apply_candidate,
    _force_remove_restricted_replacement_edges,
    _quality_advisory_counts,
    _visual_acceptance_failures,
    _visual_quality_advisories,
    apply_visual_superthin_plan,
    create_visual_state,
    validate_visual_plan,
    visual_component_inventory,
)
from fvcom_grid_generation.local_topology import (  # noqa: E402
    _expand_triangle_patch,
    _inventory_superthin_components,
    _obc_remap_manifest,
    _ordered_patch_boundary,
)
from fvcom_grid_generation.metrics import build_edge_topology  # noqa: E402


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]], np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.50, 0.015],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=int,
    )
    fixed = np.asarray([True, True, True, True, False], dtype=bool)
    chains = [[0, 1, 2, 3]]
    open_nodes = np.asarray([], dtype=int)
    targets = np.ones(len(points), dtype=float)
    return points, triangles, fixed, chains, open_nodes, targets


def _reviewed_plan(component_id: str, mesh_hash: str = "A" * 64) -> dict:
    return {
        "schema_version": PLAN_SCHEMA,
        "input_mesh_sha256": mesh_hash,
        "review": {
            "status": "reviewed",
            "reviewed_by": "Codex synthetic reviewer",
            "reviewed_at_utc": "2026-07-21T00:00:00+00:00",
            "manageable": True,
            "visual_evidence": ["synthetic_component.png"],
            "observations": "A movable interior apex creates one flat boundary-front triangle.",
        },
        "component": {
            "component_id": component_id,
            "classification": "fixed-boundary-fan",
        },
        "actions": [
            {
                "tool": "constrained_retriangulation",
                "patch_rings": [1, 2, 4],
                "maximum_support_nodes": 0,
                "remove_movable_component_nodes": True,
                "local_relaxation": False,
            }
        ],
        "restricted_lineage_edges": [],
        "acceptance": {
            "require_strict_superthin_reduction": True,
            "allow_valence_change": True,
            "preserve_quality_tails": True,
            "preserve_existing_boundary_coordinates": True,
        },
    }


def test_plan_review_and_hash_are_mandatory() -> None:
    plan = _reviewed_plan("thin-0-synthetic")
    validate_visual_plan(plan, input_sha256="A" * 64)
    stale = deepcopy(plan)
    stale["input_mesh_sha256"] = "B" * 64
    try:
        validate_visual_plan(stale, input_sha256="A" * 64)
    except ValueError as exc:
        assert "stale" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("stale plan was accepted")
    pending = deepcopy(plan)
    pending["review"]["status"] = "pending_visual_review"
    try:
        validate_visual_plan(pending, input_sha256="A" * 64)
    except ValueError as exc:
        assert "review.status" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unreviewed plan was accepted")


def test_one_component_transaction_is_deterministic() -> None:
    points, triangles, fixed, chains, open_nodes, targets = _fixture()
    state, config, _ = create_visual_state(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
    )
    components = visual_component_inventory(state, config)
    assert len(components) == 1
    plan = _reviewed_plan(components[0]["component_id"])
    first = apply_visual_superthin_plan(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
        plan=plan,
    )
    second = apply_visual_superthin_plan(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
        plan=plan,
    )
    assert first.report["accepted"] is True
    assert first.report["before"]["superthin_triangle_count"] > 0
    assert first.report["after"]["superthin_triangle_count"] == 0
    assert np.array_equal(first.nodes_xy, second.nodes_xy)
    assert np.array_equal(first.triangles, second.triangles)
    assert np.array_equal(first.nodes_xy[:4], points[:4])
    assert first.report["after"]["count_valence_above_limit"] == 0
    assert first.report["post_acceptance_atlas_required"] is True
    assert first.report["rerank_required_before_next_component"] is True


def test_restricted_diagonal_is_not_reintroduced() -> None:
    points, triangles, fixed, chains, open_nodes, targets = _fixture()
    state, config, _ = create_visual_state(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
    )
    component = _inventory_superthin_components(state, config)[0]
    plan = _reviewed_plan(component["component_id"])
    result = apply_visual_superthin_plan(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
        plan=plan,
        restricted_lineage_edges={(0, 2)},
    )
    delivered = {
        tuple(sorted((int(triangle[index]), int(triangle[(index + 1) % 3]))))
        for triangle in result.triangles
        for index in range(3)
    }
    assert (0, 2) not in delivered
    assert result.report["after"]["restricted_edge_violation_count"] == 0


def test_forbidden_delaunay_diagonal_is_force_flipped() -> None:
    points = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [1.8, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    fixed = np.zeros(4, dtype=bool)
    state, config, _ = create_visual_state(
        points,
        triangles,
        fixed,
        [],
        np.asarray([], dtype=int),
        target_spacing_m=np.ones(4, dtype=float),
        restricted_lineage_edges={(0, 2)},
    )
    from fvcom_grid_generation.local_topology import _allowed_edge_policy

    delivered, flips, illegal = _force_remove_restricted_replacement_edges(
        state,
        triangles,
        locked_edges=set(),
        policy=_allowed_edge_policy(state, config),
        max_flips=2,
    )
    edges = {
        tuple(sorted((int(triangle[index]), int(triangle[(index + 1) % 3]))))
        for triangle in delivered
        for index in range(3)
    }
    assert flips == 1
    assert illegal == []
    assert (0, 2) not in edges
    assert (1, 3) in edges


def test_collapsed_front_support_spokes_are_executed() -> None:
    points, triangles, fixed, chains, open_nodes, targets = _fixture()
    state, config, _ = create_visual_state(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
    )
    component = visual_component_inventory(state, config)[0]
    plan = _reviewed_plan(component["component_id"])
    plan["actions"] = [
        {
            "tool": "inward_front_support",
            "patch_rings": [2],
            "maximum_support_nodes": 1,
            "candidate_geometry": "superthin_longest_edges",
            "height_fractions": [0.5],
            "maximum_candidate_points": 2,
            "maximum_pair_candidates": 0,
            "remove_movable_component_nodes": True,
            "local_relaxation": False,
        }
    ]
    result = apply_visual_superthin_plan(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
        plan=plan,
    )
    attempts = result.report["attempts"]
    assert attempts
    assert any(
        int(attempt.get("evidence", {}).get("locked_support_spoke_count", 0)) > 0
        for attempt in attempts
    )


def test_minmax_cavity_action_is_bounded_and_deterministic() -> None:
    points, triangles, fixed, chains, open_nodes, targets = _fixture()
    state, config, _ = create_visual_state(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
    )
    component = visual_component_inventory(state, config)[0]
    plan = _reviewed_plan(component["component_id"])
    plan["actions"] = [
        {
            "tool": "minmax_cavity_triangulation",
            "patch_rings": [1, 2],
            "maximum_support_nodes": 0,
            "local_relaxation": False,
        }
    ]
    first = apply_visual_superthin_plan(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
        plan=plan,
    )
    second = apply_visual_superthin_plan(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
        plan=plan,
    )
    assert len(first.report["attempts"]) <= 2
    assert np.array_equal(first.triangles, second.triangles)
    assert first.report["attempts"][0]["tool"] == "minmax_cavity_triangulation"


def test_failed_visual_route_rolls_back_atomically() -> None:
    points, triangles, fixed, chains, open_nodes, targets = _fixture()
    state, config, _ = create_visual_state(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
    )
    component = visual_component_inventory(state, config)[0]
    plan = _reviewed_plan(component["component_id"])
    plan["actions"] = [
        {
            "tool": "passage_centerline_support",
            "patch_rings": [1],
            "maximum_support_nodes": 1,
            "local_relaxation": False,
        }
    ]
    result = apply_visual_superthin_plan(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
        plan=plan,
    )
    assert result.report["accepted"] is False
    assert result.report["status"] == "visual_route_infeasible"
    assert np.array_equal(result.nodes_xy, points)
    assert np.array_equal(result.triangles, triangles)
    assert result.report["post_acceptance_atlas_required"] is False


def test_class_2_tail_regressions_are_advisory_only() -> None:
    points, triangles, fixed, chains, open_nodes, targets = _fixture()
    state, config, initial_components = create_visual_state(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
    )
    components = _inventory_superthin_components(state, config)
    assert len(components) == 1
    from fvcom_grid_generation.local_topology import _audit_state, _summary

    before = _summary(state, config)
    after = dict(before)
    after["superthin_triangle_count"] = 0
    after["superthin_severity_sum"] = 0.0
    after["q_l3_sigma"] = float(before["q_l3_sigma"]) - 1.0e-4
    after["area_transition_count_above_0_50"] = (
        int(before["area_transition_count_above_0_50"]) + 2
    )
    invariant_ok, invariants, _ = _audit_state(state, config, initial_components)
    failures = _visual_acceptance_failures(
        state,
        state,
        components[0],
        components,
        before,
        after,
        invariant_ok,
        invariants,
        tolerance=1.0e-9,
    )
    advisories = _visual_quality_advisories(before, after, tolerance=1.0e-9)
    assert failures == []
    assert advisories == [
        "area_transition_count_above_0_50_regression",
        "q_l3_sigma_regression",
    ]
    assert _quality_advisory_counts(
        [{"quality_advisories": advisories}, {"quality_advisories": advisories[:1]}]
    ) == {
        "area_transition_count_above_0_50_regression": 2,
        "q_l3_sigma_regression": 1,
    }


def test_obc_source_arc_insertion_preserves_original_subsequence() -> None:
    points, triangles, fixed, chains, _, targets = _fixture()
    open_nodes = np.asarray([0, 1], dtype=int)
    state, config, _ = create_visual_state(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
    )
    component = _inventory_superthin_components(state, config)[0]
    topology = build_edge_topology(len(state.points), state.triangles)
    patch = _expand_triangle_patch(
        state.triangles,
        topology,
        component["triangle_indices"],
        2,
    )
    ring = _ordered_patch_boundary(state.triangles, patch)
    assert ring is not None
    candidates = _action_candidates(
        state,
        component,
        patch,
        ring,
        "source_arc_insertion",
        {"tool": "source_arc_insertion"},
        1,
    )
    assert candidates
    trial = state.clone()
    changed, failures, evidence = _apply_candidate(
        trial,
        component,
        patch,
        ring,
        candidates[0],
        config,
    )
    assert changed, failures
    assert evidence["boundary_insertion_count"] == 1
    retained = [
        int(trial.lineage[index])
        for index in trial.open_nodes
        if int(trial.lineage[index]) >= 0
    ]
    assert retained == [0, 1]
    manifest = _obc_remap_manifest(trial)
    assert manifest["obc_forcing_compatible"] is False
    assert manifest["forcing_invalidation_required"] is True


def test_passage_support_discovers_two_patch_banks() -> None:
    points = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 0.04], [2.0, 0.04]],
        dtype=float,
    )
    triangles = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=int)
    fixed = np.ones(4, dtype=bool)
    chains = [[0, 1], [2, 3]]
    state, config, _ = create_visual_state(
        points,
        triangles,
        fixed,
        chains,
        np.asarray([], dtype=int),
        target_spacing_m=np.ones(4, dtype=float),
    )
    components = _inventory_superthin_components(state, config)
    assert components
    component = components[0]
    topology = build_edge_topology(len(state.points), state.triangles)
    patch = _expand_triangle_patch(
        state.triangles,
        topology,
        component["triangle_indices"],
        1,
    )
    ring = _ordered_patch_boundary(state.triangles, patch)
    assert ring is not None
    candidates = _action_candidates(
        state,
        component,
        patch,
        ring,
        "passage_centerline_support",
        {"tool": "passage_centerline_support", "lock_support_spokes": False},
        1,
    )
    assert candidates
    assert all(candidate["support_count"] == 1 for candidate in candidates)


def main() -> int:
    tests = [
        test_plan_review_and_hash_are_mandatory,
        test_one_component_transaction_is_deterministic,
        test_restricted_diagonal_is_not_reintroduced,
        test_forbidden_delaunay_diagonal_is_force_flipped,
        test_collapsed_front_support_spokes_are_executed,
        test_minmax_cavity_action_is_bounded_and_deterministic,
        test_failed_visual_route_rolls_back_atomically,
        test_class_2_tail_regressions_are_advisory_only,
        test_obc_source_arc_insertion_preserves_original_subsequence,
        test_passage_support_discovers_two_patch_banks,
    ]
    for test in tests:
        test()
    print(f"visual superthin selftests passed ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
