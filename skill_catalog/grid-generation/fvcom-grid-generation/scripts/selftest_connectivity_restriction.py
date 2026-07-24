"""Deterministic tests for systematic-V5 allowed-edge restriction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from fvcom_grid_generation.connectivity_restriction import (  # noqa: E402
    AllowedEdgePolicy,
    ConnectivityRestrictionConfig,
    audit_superthin_connectivity,
    restricted_edge_violation_records,
)
from fvcom_grid_generation.local_topology import (  # noqa: E402
    AggressiveConditioningConfig,
    condition_mesh_aggressive,
)


def _hairpin_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[list[int]],
    list[str],
    np.ndarray,
]:
    points = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, -3.0],
            [4.0, -3.0],
            [4.0, -0.1],
            [6.0, 0.0],
            [6.0, 5.0],
            [0.0, 5.0],
            [3.0, 2.0],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [1, 2, 3],
            [1, 3, 4],
            [1, 4, 5],
            [0, 1, 8],
            [1, 5, 8],
            [5, 6, 8],
            [6, 7, 8],
            [7, 0, 8],
        ],
        dtype=int,
    )
    fixed = np.asarray([True] * 8 + [False], dtype=bool)
    chains = [list(range(8))]
    kinds = ["land"] * 8 + ["interior"]
    hard = np.zeros(len(points), dtype=bool)
    hard[[1, 4]] = True
    return points, triangles, fixed, chains, kinds, hard


def _config() -> AggressiveConditioningConfig:
    return AggressiveConditioningConfig(
        thin_repair_profile="systematic-v5",
        systematic_gate_scope="loop-end",
        systematic_v5_connectivity_only=True,
        systematic_v5_enable_connectivity_restriction=True,
        systematic_v5_max_connectivity_transactions_per_round=8,
        systematic_v5_max_connectivity_candidates_per_component=8,
        systematic_v5_enable_boundary_window_fallback=False,
        enable_pruning=False,
        enable_thin_repair=True,
        enable_valence_repair=False,
        max_rounds=1,
        max_prunes_per_round=0,
        max_valence_removals_per_round=0,
        micro_relax_cycles=0,
    )


def test_policy_classifies_hairpin_shortcut_by_arc_span() -> None:
    points, triangles, _, chains, _, _ = _hairpin_fixture()
    policy = AllowedEdgePolicy(
        points,
        np.ones(len(points), dtype=float),
        chains,
        np.arange(len(points), dtype=int),
        config=ConnectivityRestrictionConfig(),
    )
    evidence = policy.same_chain_shortcut_evidence((1, 4))
    assert evidence["is_shortcut"] is True
    assert evidence["arc_chord_ratio"] > 3.0
    assert evidence["arc_target_ratio"] > 3.0
    geometry_mask = np.zeros(len(triangles), dtype=bool)
    geometry_mask[2] = True
    candidates = policy.candidate_records(
        triangles,
        [2],
        geometry_mask,
    )
    assert candidates[0]["lineage_edge"] == [1, 4]


def test_policy_keeps_local_nonshortcut_boundary_diagonal() -> None:
    points, _, _, chains, _, _ = _hairpin_fixture()
    policy = AllowedEdgePolicy(
        points,
        np.full(len(points), 2.0),
        chains,
        np.arange(len(points), dtype=int),
    )
    evidence = policy.same_chain_shortcut_evidence((3, 5))
    assert evidence["is_shortcut"] is False
    assert policy.is_allowed(
        (3, 5),
        reject_same_chain_shortcuts=True,
    )


def test_protected_chain_edge_overrides_an_accidental_restriction() -> None:
    points, _, _, chains, _, _ = _hairpin_fixture()
    policy = AllowedEdgePolicy(
        points,
        np.ones(len(points), dtype=float),
        chains,
        np.arange(len(points), dtype=int),
        restricted_lineage_edges={(2, 3)},
    )
    assert policy.is_allowed((2, 3))
    assert policy.same_chain_shortcut_evidence((2, 3))["protected"]


def test_read_only_audit_reports_ranked_causal_shortcut() -> None:
    points, triangles, _, chains, _, _ = _hairpin_fixture()
    report = audit_superthin_connectivity(
        points,
        triangles,
        np.ones(len(points), dtype=float),
        chains,
    )
    assert (
        report["schema_version"]
        == "fvcom_superthin_connectivity_restriction_v1"
    )
    assert report["superthin_triangle_count"] == 1
    assert report["superthin_component_count"] == 1
    candidate = report["components"][0]["candidate_edges"][0]
    assert candidate["lineage_edge"] == [1, 4]
    assert candidate["same_chain_shortcut"]["is_shortcut"] is True


def test_topology_only_restriction_closes_hairpin_superthin() -> None:
    points, triangles, fixed, chains, kinds, hard = _hairpin_fixture()
    result = condition_mesh_aggressive(
        points,
        triangles,
        fixed,
        chains,
        np.asarray([0, 1], dtype=int),
        target_spacing_m=np.ones(len(points), dtype=float),
        boundary_kinds=kinds,
        hard_anchor_mask=hard,
        config=_config(),
    )
    assert result.report["before"]["superthin_triangle_count"] >= 1
    assert result.report["after"]["superthin_triangle_count"] == 0
    assert (1, 4) in result.restricted_lineage_edges
    delivered_edges = {
        tuple(sorted((int(a), int(b))))
        for triangle in result.triangles
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        )
    }
    assert (1, 4) not in delivered_edges
    assert np.array_equal(result.nodes_xy, points)
    assert result.constraint_chains == chains
    assert result.open_boundary_nodes_zero_based.tolist() == [0, 1]
    stage = result.report["rounds"][0]["aggressive_thin_repair"]
    connectivity = stage["connectivity_restriction_initial"]
    assert connectivity["accepted"] == 1
    assert connectivity["restricted_edge_violation_count"] == 0
    assert result.report["invariants"]["all_protected_edges_present"] is True
    assert result.report["invariants"]["open_boundary_ordered"] is True


def test_persisted_restriction_survives_a_second_closure() -> None:
    points, triangles, fixed, chains, kinds, hard = _hairpin_fixture()
    first = condition_mesh_aggressive(
        points,
        triangles,
        fixed,
        chains,
        np.asarray([0, 1], dtype=int),
        target_spacing_m=np.ones(len(points), dtype=float),
        boundary_kinds=kinds,
        hard_anchor_mask=hard,
        config=_config(),
    )
    second = condition_mesh_aggressive(
        first.nodes_xy,
        first.triangles,
        first.fixed_node_mask,
        first.constraint_chains,
        first.open_boundary_nodes_zero_based,
        target_spacing_m=first.target_spacing_m,
        boundary_kinds=first.boundary_kinds,
        hard_anchor_mask=first.hard_anchor_mask,
        restricted_lineage_edges=first.restricted_lineage_edges,
        config=_config(),
    )
    assert second.report["after"]["restricted_edge_violation_count"] == 0
    assert (1, 4) in second.restricted_lineage_edges
    assert np.array_equal(first.triangles, second.triangles)


def test_topology_only_full_v5_preserves_exact_obc_membership() -> None:
    points, triangles, fixed, chains, kinds, hard = _hairpin_fixture()
    config = replace(
        _config(),
        systematic_v5_connectivity_only=False,
        systematic_v5_max_star_transactions_per_round=16,
    )
    result = condition_mesh_aggressive(
        points,
        triangles,
        fixed,
        chains,
        np.asarray([0, 1], dtype=int),
        target_spacing_m=np.ones(len(points), dtype=float),
        boundary_kinds=kinds,
        hard_anchor_mask=hard,
        config=config,
    )
    assert (
        result.open_boundary_nodes_zero_based.tolist()
        == [0, 1]
    )
    assert (
        result.report["invariants"][
            "open_boundary_membership_unchanged"
        ]
        is True
    )


def test_impossible_transaction_rolls_back_exactly() -> None:
    points, triangles, fixed, chains, kinds, hard = _hairpin_fixture()
    points = points[:8]
    triangles = triangles[:3]
    config = replace(
        _config(),
        systematic_v5_max_connectivity_candidates_per_component=1,
    )
    result = condition_mesh_aggressive(
        points,
        triangles,
        fixed[:8],
        chains,
        np.asarray([0, 1], dtype=int),
        target_spacing_m=np.ones(len(points), dtype=float),
        boundary_kinds=kinds[:8],
        hard_anchor_mask=hard[:8],
        config=config,
    )
    assert result.report["after"]["superthin_triangle_count"] == 1
    assert result.restricted_lineage_edges == set()
    assert np.array_equal(result.nodes_xy, points)
    assert np.array_equal(result.triangles, triangles)


def test_restriction_is_exactly_repeatable() -> None:
    points, triangles, fixed, chains, kinds, hard = _hairpin_fixture()
    kwargs = {
        "target_spacing_m": np.ones(len(points), dtype=float),
        "boundary_kinds": kinds,
        "hard_anchor_mask": hard,
        "config": _config(),
    }
    first = condition_mesh_aggressive(
        points,
        triangles,
        fixed,
        chains,
        np.asarray([0, 1], dtype=int),
        **kwargs,
    )
    second = condition_mesh_aggressive(
        points,
        triangles,
        fixed,
        chains,
        np.asarray([0, 1], dtype=int),
        **kwargs,
    )
    assert np.array_equal(first.nodes_xy, second.nodes_xy)
    assert np.array_equal(first.triangles, second.triangles)
    assert first.restricted_lineage_edges == second.restricted_lineage_edges


def test_shared_restricted_edge_audit_is_lineage_stable() -> None:
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    lineage = np.asarray([100, 101, 102, 103], dtype=int)
    records = restricted_edge_violation_records(
        triangles,
        lineage,
        {(100, 102)},
    )
    assert records == [
        {
            "edge": [0, 2],
            "lineage_edge": [100, 102],
            "attached_triangle_indices": [0, 1],
        }
    ]


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "connectivity-restriction selftests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
