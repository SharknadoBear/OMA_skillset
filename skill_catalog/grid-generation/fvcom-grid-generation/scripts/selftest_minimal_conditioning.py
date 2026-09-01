#!/usr/bin/env python3
"""Focused tests for the minimal-topology-v1 conditioning contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fvcom_grid_generation.local_topology as local_topology  # noqa: E402
from fvcom_grid_generation.local_topology import (  # noqa: E402
    AggressiveConditioningConfig,
    condition_mesh_aggressive,
)
from fvcom_grid_generation.metrics import chain_edges  # noqa: E402
from fvcom_grid_generation.portfolio_conditioning import (  # noqa: E402
    PortfolioConditioningConfig,
    _minimal_report_only_deltas,
    _primary_topology_config,
    _stage_regressions,
)


def _minimal_config(**overrides: object) -> AggressiveConditioningConfig:
    base = AggressiveConditioningConfig(
        profile_name="minimal-topology-v1",
        stage_order="valence-before-thin",
        enable_pruning=False,
        enable_thin_repair=True,
        enable_valence_repair=True,
        max_rounds=4,
        max_prunes_per_round=0,
        boundary_edit_policy="none",
        max_boundary_edits_per_round=0,
        max_boundary_welds_per_round=0,
        max_boundary_ear_removals_per_round=0,
        micro_relax_cycles=0,
    )
    return replace(base, **overrides)


def _nine_spoke_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[list[int]],
    list[str],
]:
    count = 9
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    ring = np.column_stack((np.cos(angles), np.sin(angles)))
    points = np.vstack((ring, np.asarray([[0.0, 0.0]])))
    triangles = np.asarray(
        [[count, index, (index + 1) % count] for index in range(count)],
        dtype=int,
    )
    return (
        points,
        triangles,
        np.asarray([True] * count + [False]),
        [list(range(count))],
        ["land"] * count + ["interior"],
    )


def _run(
    points: np.ndarray,
    triangles: np.ndarray,
    fixed: np.ndarray,
    chains: list[list[int]],
    kinds: list[str],
    targets: np.ndarray,
    config: AggressiveConditioningConfig | None = None,
):
    return condition_mesh_aggressive(
        points,
        triangles,
        fixed,
        chains,
        np.empty(0, dtype=int),
        target_spacing_m=targets,
        boundary_kinds=kinds,
        hard_anchor_mask=fixed,
        config=config or _minimal_config(),
    )


def _canonical_triangles(values: np.ndarray) -> list[tuple[int, int, int]]:
    return sorted(
        tuple(sorted(map(int, triangle)))
        for triangle in np.asarray(values, dtype=int)
    )


def test_auto_resolver_is_minimal_and_disables_broad_edits() -> None:
    resolved = _primary_topology_config(
        PortfolioConditioningConfig(conditioning_profile="auto"),
        effective_profile="minimal-topology-v1",
    )
    assert resolved.profile_name == "minimal-topology-v1"
    assert resolved.stage_order == "valence-before-thin"
    assert not resolved.enable_pruning
    assert resolved.boundary_edit_policy == "none"
    assert resolved.max_boundary_edits_per_round == 0
    assert resolved.max_boundary_welds_per_round == 0
    assert resolved.max_boundary_ear_removals_per_round == 0
    assert resolved.enable_fixed_hard_fan_arc_refinement
    assert resolved.max_fixed_hard_fan_arc_refinements_per_round == 8
    assert resolved.micro_relax_cycles == 0
    assert resolved.topology_escrow_enabled


def _stage_audit() -> dict[str, object]:
    return {
        "core_passed": True,
        "core_failures": [],
        "singly_connected_triangle_count": 0,
        "superthin_triangle_count": 0,
        "count_valence_above_8": 1,
        "valence_excess_above_8": 1,
        "maximum_valence": 9,
        "area_transition_defect_count": 10,
        "l_over_h_count_above_1_55": 0,
        "q_min": 0.50,
        "minimum_angle_deg": 30.0,
        "q_p01": 0.80,
        "q_l3_sigma": 0.85,
        "maximum_adjacent_area_change": 0.70,
        "l_over_h_p95": 1.00,
        "l_over_h_maximum": 1.20,
    }


def test_minimal_outer_gate_reports_q_p01_and_area_count_only() -> None:
    before = _stage_audit()
    after = dict(before)
    after.update(
        {
            "count_valence_above_8": 0,
            "valence_excess_above_8": 0,
            "area_transition_defect_count": 15,
            "q_p01": 0.799,
        }
    )
    assert _stage_regressions(before, after, minimal_policy=True) == []
    deltas = _minimal_report_only_deltas(before, after)
    assert deltas["q_p01"]["regressed_under_previous_gate"]
    assert deltas["area_transition_defect_count_above_0_50"][
        "regressed_under_previous_gate"
    ]


def test_legacy_outer_gate_preserves_q_p01_and_area_count_vetoes() -> None:
    before = _stage_audit()
    after = dict(before)
    after.update(
        {
            "area_transition_defect_count": 11,
            "q_p01": 0.799,
        }
    )
    assert set(_stage_regressions(before, after)) == {
        "area_transition_defect_count_regressed",
        "q_p01_regressed",
    }


def test_minimal_outer_gate_keeps_priority_vetoes_only() -> None:
    before = _stage_audit()
    after = dict(before)
    after.update(
        {
            "superthin_triangle_count": 1,
            "q_min": 0.24,
            "minimum_angle_deg": 19.0,
            "q_l3_sigma": 0.849,
            "maximum_adjacent_area_change": 0.71,
            "l_over_h_p95": 1.01,
        }
    )
    failures = set(_stage_regressions(before, after, minimal_policy=True))
    assert failures == {"superthin_triangle_count_regressed"}


def test_valence_only_repair_closes_without_boundary_movement() -> None:
    points, triangles, fixed, chains, kinds = _nine_spoke_fixture()
    result = _run(
        points,
        triangles,
        fixed,
        chains,
        kinds,
        np.ones(len(points), dtype=float),
    )
    assert result.report["before"]["count_valence_above_limit"] == 1
    assert result.report["after"]["count_valence_above_limit"] == 0
    assert result.report["after"]["superthin_triangle_count"] == 0
    assert result.report["minimal_local_debt_closed"]
    assert result.report["edit_counts"].get(
        "high-valence-cavity-remove",
        0,
    ) == 1
    assert np.array_equal(result.nodes_xy[:9], points[:9])
    assert chain_edges(result.constraint_chains) == chain_edges(chains)


def test_superthin_only_repair_is_an_atomic_flip() -> None:
    points = np.asarray(
        [[-1.0, 0.0], [0.0, -0.01], [1.0, 0.0], [0.0, 2.0]],
        dtype=float,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    fixed = np.ones(4, dtype=bool)
    chains = [[0, 1, 2, 3]]
    result = _run(
        points,
        triangles,
        fixed,
        chains,
        ["land"] * 4,
        np.asarray([0.25, 0.25, 0.25, 0.50]),
    )
    assert result.report["before"]["superthin_triangle_count"] == 1
    assert result.report["after"]["superthin_triangle_count"] == 0
    assert result.report["minimal_local_debt_closed"]
    assert result.report["edit_counts"] == {"superthin-edge-flip": 1}
    assert len(result.triangles) == len(triangles)
    assert np.array_equal(result.nodes_xy, points)
    assert result.constraint_chains == chains


def test_protected_superthin_boundary_is_reported_not_deleted() -> None:
    points = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0e-4]],
        dtype=float,
    )
    triangles = np.asarray([[0, 1, 2]], dtype=int)
    fixed = np.ones(3, dtype=bool)
    result = _run(
        points,
        triangles,
        fixed,
        [[0, 1, 2]],
        ["land"] * 3,
        np.ones(3, dtype=float),
    )
    assert not result.report["minimal_local_debt_closed"]
    assert result.report["after"]["superthin_triangle_count"] == 1
    assert len(result.triangles) == 1
    assert np.array_equal(result.nodes_xy, points)
    assert _canonical_triangles(result.triangles) == [(0, 1, 2)]


def _fixed_hard_fan_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[list[int]],
    list[str],
]:
    points = np.asarray(
        [
            [0.0, 0.0],
            [-202.53342621243792, -139.4223343185149],
            [-442.683838472527, -46.290730671491474],
            [-284.64227076625684, -162.47637594863772],
            [-46.94825545203639, -121.90163250686601],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [[1, 4, 0], [2, 3, 1], [3, 4, 1]],
        dtype=int,
    )
    fixed = np.ones(len(points), dtype=bool)
    return (
        points,
        triangles,
        fixed,
        [[0, 1, 2, 3, 4]],
        ["fixed_topological_boundary"] * len(points),
    )


def test_fixed_hard_non_obc_fan_gets_one_bounded_arc_refinement() -> None:
    points, triangles, fixed, chains, kinds = _fixed_hard_fan_fixture()
    result = _run(
        points,
        triangles,
        fixed,
        chains,
        kinds,
        np.full(len(points), 1375.0, dtype=float),
        _minimal_config(
            enable_fixed_hard_fan_arc_refinement=True,
            max_fixed_hard_fan_arc_refinements_per_round=1,
            max_superthin_flips_per_round=0,
            max_collapses_per_round=0,
            max_valence_removals_per_round=0,
        ),
    )
    assert result.report["before"]["superthin_triangle_count"] == 1
    assert result.report["after"]["superthin_triangle_count"] == 0
    assert result.report["after"]["maximum_valence"] <= 8
    assert result.report["minimal_local_debt_closed"]
    assert len(result.nodes_xy) == len(points) + 1
    assert np.array_equal(result.nodes_xy[: len(points)], points)
    assert result.open_boundary_nodes_zero_based.size == 0
    assert result.report["edit_counts"] == {
        "minimal-fixed-hard-fan-source-arc-refinement": 1,
        "minimal-fixed-hard-fan-transaction-accepted": 1,
    }


def test_fixed_hard_fan_refinement_never_changes_an_obc() -> None:
    points, triangles, fixed, chains, kinds = _fixed_hard_fan_fixture()
    open_nodes = np.asarray([0, 1, 2, 3, 4], dtype=int)
    result = condition_mesh_aggressive(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=np.full(len(points), 1375.0, dtype=float),
        boundary_kinds=["open"] * len(points),
        hard_anchor_mask=fixed,
        config=_minimal_config(
            enable_fixed_hard_fan_arc_refinement=True,
            max_fixed_hard_fan_arc_refinements_per_round=1,
            max_superthin_flips_per_round=0,
            max_collapses_per_round=0,
            max_valence_removals_per_round=0,
        ),
    )
    assert result.report["after"]["superthin_triangle_count"] == 1
    assert np.array_equal(result.nodes_xy, points)
    assert np.array_equal(result.open_boundary_nodes_zero_based, open_nodes)
    assert result.edit_ledger == []


def _cross_chain_passage_state() -> local_topology._State:
    points = np.asarray(
        [
            [0.0, 0.0],
            [0.05, 0.0],
            [0.05, 1.0],
            [-2.0, -2.0],
            [-2.0, 2.0],
            [2.0, 0.5],
        ],
        dtype=float,
    )
    return local_topology._State(
        points=points,
        triangles=np.asarray([[0, 1, 2]], dtype=int),
        fixed=np.ones(len(points), dtype=bool),
        targets=np.full(len(points), 10.0, dtype=float),
        chains=[[0, 3, 4], [1, 2, 5]],
        open_nodes=np.empty(0, dtype=int),
        kinds=["land", "island", "island", "land", "land", "island"],
        hard=np.asarray([True, False, False, False, False, False]),
        lineage=np.arange(len(points), dtype=int),
        source_points=points.copy(),
        source_chains=[[0, 3, 4], [1, 2, 5]],
    )


def test_cross_chain_passage_has_one_exact_non_obc_midpoint_route() -> None:
    state = _cross_chain_passage_state()
    components = local_topology._inventory_superthin_components(
        state,
        _minimal_config(),
    )
    assert len(components) == 1
    component = components[0]
    assert component["classification"] == "under-resolved-passage"
    assert component["gap_over_h"] < 0.25
    assert local_topology._fixed_hard_fan_ineligibility(state, component) == []
    assert local_topology._automatic_source_arc_edge_is_eligible(
        state, (1, 2), require_hard=False
    )
    assert not local_topology._automatic_source_arc_edge_is_eligible(
        state, (1, 2), require_hard=True
    )


def test_cross_chain_midpoint_route_rejects_obc_and_nonlocal_passage() -> None:
    state = _cross_chain_passage_state()
    component = local_topology._inventory_superthin_components(
        state,
        _minimal_config(),
    )[0]
    state.open_nodes = np.asarray([1], dtype=int)
    assert "component_touches_open_boundary" in local_topology._fixed_hard_fan_ineligibility(
        state, component
    )
    state.open_nodes = np.empty(0, dtype=int)
    component = dict(component)
    component["gap_over_h"] = 0.5
    assert "passage_not_strictly_sub_resolution" in local_topology._fixed_hard_fan_ineligibility(
        state, component
    )


def test_expired_deadline_stops_before_first_transaction() -> None:
    points, triangles, fixed, chains, kinds = _nine_spoke_fixture()
    result = _run(
        points,
        triangles,
        fixed,
        chains,
        kinds,
        np.ones(len(points), dtype=float),
        _minimal_config(deadline_monotonic_s=time.perf_counter() - 1.0),
    )
    assert result.report["deadline_reached"]
    assert result.report["rounds"] == []
    assert result.report["after"]["count_valence_above_limit"] == 1
    assert not result.report["minimal_local_debt_closed"]
    assert np.array_equal(result.nodes_xy, points)
    assert _canonical_triangles(result.triangles) == _canonical_triangles(
        triangles
    )


def test_deterministic_replay_matches_connectivity_and_lineage() -> None:
    fixture = _nine_spoke_fixture()
    first = _run(
        *fixture,
        np.ones(len(fixture[0]), dtype=float),
    )
    second = _run(
        *fixture,
        np.ones(len(fixture[0]), dtype=float),
    )
    assert np.array_equal(first.nodes_xy, second.nodes_xy)
    assert np.array_equal(first.triangles, second.triangles)
    assert np.array_equal(first.node_lineage, second.node_lineage)
    assert first.edit_ledger == second.edit_ledger


TESTS: tuple[Callable[[], None], ...] = (
    test_auto_resolver_is_minimal_and_disables_broad_edits,
    test_minimal_outer_gate_reports_q_p01_and_area_count_only,
    test_legacy_outer_gate_preserves_q_p01_and_area_count_vetoes,
    test_minimal_outer_gate_keeps_priority_vetoes_only,
    test_valence_only_repair_closes_without_boundary_movement,
    test_superthin_only_repair_is_an_atomic_flip,
    test_protected_superthin_boundary_is_reported_not_deleted,
    test_fixed_hard_non_obc_fan_gets_one_bounded_arc_refinement,
    test_fixed_hard_fan_refinement_never_changes_an_obc,
    test_cross_chain_passage_has_one_exact_non_obc_midpoint_route,
    test_cross_chain_midpoint_route_rejects_obc_and_nonlocal_passage,
    test_expired_deadline_stops_before_first_transaction,
    test_deterministic_replay_matches_connectivity_and_lineage,
)


def main() -> int:
    failures: list[tuple[str, BaseException]] = []
    for test in TESTS:
        try:
            test()
        except BaseException as exc:
            failures.append((test.__name__, exc))
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    if failures:
        return 1
    print("All minimal-conditioning tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
