"""Focused synthetic regression tests for atomic thin/valence conditioning v2."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import fvcom_grid_generation.local_topology as local_topology  # noqa: E402
from fvcom_grid_generation.local_topology import (  # noqa: E402
    AggressiveConditioningConfig,
    condition_mesh_aggressive,
)
from fvcom_grid_generation.metrics import build_edge_topology, triangle_geometry  # noqa: E402


def _thin_only_config(**overrides: Any) -> AggressiveConditioningConfig:
    config = AggressiveConditioningConfig(
        max_rounds=1,
        enable_pruning=False,
        enable_thin_repair=True,
        enable_valence_repair=False,
        max_prunes_per_round=0,
        max_boundary_ear_removals_per_round=0,
        max_boundary_welds_per_round=0,
        max_superthin_flips_per_round=0,
        max_collapses_per_round=0,
        max_boundary_edits_per_round=0,
        max_valence_removals_per_round=0,
        micro_relax_cycles=0,
    )
    return replace(config, **overrides)


def _weld_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]], list[str]]:
    # Node 4 lies just inside the coarse bottom source arc.  The surrounding
    # interior triangulation gives every retained boundary triangle two
    # neighbors after the weld, so a successful edit creates no FVCOM ear.
    points = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 5.0],
            [0.0, 5.0],
            [5.0, 0.1],
            [2.5, 2.0],
            [7.5, 2.0],
            [5.0, 4.0],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [0, 1, 4],
            [0, 4, 5],
            [0, 5, 3],
            [3, 5, 7],
            [3, 7, 2],
            [2, 7, 6],
            [2, 6, 1],
            [1, 6, 4],
            [4, 6, 7],
            [4, 7, 5],
        ],
        dtype=int,
    )
    fixed = np.asarray([True, True, True, True, False, False, False, False])
    chains = [[0, 1, 2, 3]]
    kinds = ["land", "land", "land", "land", "interior", "interior", "interior", "interior"]
    return points, triangles, fixed, chains, kinds


def _ear_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]], list[str]]:
    # Triangle (0, 2, 1) is an exterior redundant ear.  Its free chord (0, 2)
    # is the current mesh boundary while protected source arcs (0, 1), (1, 2)
    # are each backed by a retained wet triangle.
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, -0.01],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [0, 2, 1],
            [0, 1, 5],
            [1, 2, 5],
            [2, 3, 5],
            [3, 4, 5],
            [4, 0, 5],
        ],
        dtype=int,
    )
    fixed = np.asarray([True, True, True, True, True, False])
    chains = [[0, 1, 2, 3, 4]]
    kinds = ["land", "land", "land", "land", "land", "interior"]
    return points, triangles, fixed, chains, kinds


def _run_fixture(
    fixture: tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]], list[str]],
    config: AggressiveConditioningConfig,
    *,
    hard: np.ndarray | None = None,
    sampler: Any = None,
):
    points, triangles, fixed, chains, kinds = fixture
    return condition_mesh_aggressive(
        points,
        triangles,
        fixed,
        chains,
        np.empty(0, dtype=int),
        target_spacing_m=np.full(len(points), 2.0),
        boundary_kinds=kinds,
        hard_anchor_mask=np.zeros(len(points), dtype=bool) if hard is None else hard,
        target_spacing_sampler=sampler,
        config=config,
    )


def _canonical_triangles(triangles: np.ndarray) -> list[tuple[int, int, int]]:
    return sorted(tuple(sorted(map(int, tri))) for tri in np.asarray(triangles, dtype=int))


def test_mode_safe_thin_disable_is_exact_noop() -> None:
    points, triangles, *_ = _weld_fixture()
    config = replace(
        _thin_only_config(),
        enable_thin_repair=False,
    )
    result = _run_fixture(_weld_fixture(), config)
    assert np.array_equal(result.nodes_xy, points)
    assert _canonical_triangles(result.triangles) == _canonical_triangles(triangles)
    round_report = result.report["rounds"][0]
    assert round_report["aggressive_thin_repair"]["reason"] == "thin_repair_disabled"
    assert round_report["accepted_operation_count"] == 0


def test_compaction_discards_only_stale_boundary_kind_tail() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=float)
    state = local_topology._State(
        points=points.copy(),
        triangles=np.asarray([[0, 1, 2]], dtype=int),
        fixed=np.asarray([True, True, True, False]),
        targets=np.ones(4, dtype=float),
        chains=[[0, 1, 2]],
        open_nodes=np.empty(0, dtype=int),
        kinds=["land", "land", "land", "interior", "stale-tail"],
        hard=np.zeros(4, dtype=bool),
        lineage=np.arange(4, dtype=int),
        source_points=points.copy(),
        source_chains=[[0, 1, 2]],
    )
    mapping = local_topology._compact(state)
    assert mapping.tolist() == [0, 1, 2, -1]
    assert len(state.points) == len(state.fixed) == len(state.targets) == len(state.hard) == len(state.lineage) == 3
    assert state.kinds == ["land", "land", "land"]


def test_redundant_ear_removal_preserves_boundary_and_uses_actual_area() -> None:
    points, triangles, *_ = _ear_fixture()
    expected_area = float(triangle_geometry(points, triangles[[0]])["area"][0])
    result = _run_fixture(
        _ear_fixture(),
        _thin_only_config(
            max_boundary_ear_removals_per_round=2,
            maximum_domain_area_change_fraction=0.01,
        ),
    )
    assert result.report["edit_counts"].get("boundary-ear-remove", 0) == 1
    assert len(result.triangles) == len(triangles) - 1
    assert np.isclose(result.report["cumulative_boundary_area_change_m2"], expected_area)
    assert result.report["invariants"]["boundary_traversable"] is True
    assert result.report["invariants"]["new_singly_connected_triangle_count"] == 0
    assert result.report["invariants"]["all_protected_edges_present"] is True


def test_redundant_ear_area_budget_rejection_is_reported() -> None:
    points, triangles, *_ = _ear_fixture()
    result = _run_fixture(
        _ear_fixture(),
        _thin_only_config(
            max_boundary_ear_removals_per_round=1,
            maximum_domain_area_change_fraction=1.0e-5,
        ),
    )
    assert np.array_equal(result.nodes_xy, points)
    assert _canonical_triangles(result.triangles) == _canonical_triangles(triangles)
    rejected = result.report["rounds"][0]["aggressive_thin_repair"]["rejected_cases"]
    assert any("domain_area_budget" in case["failures"] for case in rejected)


def test_source_arc_weld_succeeds_and_resamples_eulerian_target() -> None:
    sampler = lambda xy: np.full(len(xy), 3.0, dtype=float)
    result = _run_fixture(
        _weld_fixture(),
        _thin_only_config(
            max_boundary_welds_per_round=2,
            maximum_domain_area_change_fraction=0.02,
        ),
        sampler=sampler,
    )
    assert result.report["edit_counts"].get("boundary-arc-weld", 0) == 1
    assert np.allclose(result.nodes_xy[4], [5.0, 0.0])
    assert np.isclose(result.target_spacing_m[4], 3.0)
    assert result.constraint_chains[0][:3] == [0, 4, 1]
    assert result.report["invariants"]["new_singly_connected_triangle_count"] == 0
    assert result.report["invariants"]["boundary_degree_anomaly_count"] == 0
    assert result.report["after"]["superthin_triangle_count"] == 0


def test_weld_hard_anchor_distance_and_channel_guards_are_reported() -> None:
    base = _thin_only_config(max_boundary_welds_per_round=1)
    hard = np.zeros(8, dtype=bool)
    hard[0] = True
    anchored = _run_fixture(_weld_fixture(), base, hard=hard)
    anchored_rejections = anchored.report["rounds"][0]["aggressive_thin_repair"]["screened_rejections"]
    assert any("hard_anchor_buffer" in case["failures"] for case in anchored_rejections)

    distant = _run_fixture(
        _weld_fixture(),
        replace(base, boundary_weld_max_distance_fraction=0.01),
    )
    distant_rejections = distant.report["rounds"][0]["aggressive_thin_repair"]["screened_rejections"]
    assert any("weld_distance_fraction" in case["failures"] for case in distant_rejections)

    points, triangles, fixed, chains, kinds = _weld_fixture()
    kinds[0] = kinds[1] = "channel-bank"
    channel = _run_fixture((points, triangles, fixed, chains, kinds), base)
    channel_rejections = channel.report["rounds"][0]["aggressive_thin_repair"]["screened_rejections"]
    assert any(
        "under_resolved_channel_or_junction_requires_upstream_review" in case["failures"]
        for case in channel_rejections
    )
    assert channel.report["edit_counts"].get("boundary-arc-weld", 0) == 0


def test_systematic_v2_closes_fixed_boundary_fan_without_moving_anchors() -> None:
    points, triangles, fixed, chains, kinds = _weld_fixture()
    hard = np.zeros(len(points), dtype=bool)
    hard[[0, 1]] = True
    result = condition_mesh_aggressive(
        points,
        triangles,
        fixed,
        chains,
        np.empty(0, dtype=int),
        target_spacing_m=np.full(len(points), 20.0),
        boundary_kinds=kinds,
        hard_anchor_mask=hard,
        config=AggressiveConditioningConfig(
            thin_repair_profile="systematic-v2",
            max_rounds=1,
            enable_pruning=False,
            enable_valence_repair=False,
            max_prunes_per_round=0,
            max_valence_removals_per_round=0,
            micro_relax_cycles=0,
        ),
    )
    assert result.report["before"]["superthin_triangle_count"] == 1
    assert result.report["after"]["superthin_triangle_count"] == 0
    assert result.report["edit_counts"].get("systematic-superthin-component-cavity", 0) == 1
    lineage_to_current = {int(value): index for index, value in enumerate(result.node_lineage)}
    assert np.array_equal(result.nodes_xy[lineage_to_current[0]], points[0])
    assert np.array_equal(result.nodes_xy[lineage_to_current[1]], points[1])
    assert result.report["invariants"]["all_protected_edges_present"] is True


def test_systematic_component_inventory_classifies_under_resolved_passage() -> None:
    points = np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 0.1]], dtype=float)
    state = local_topology._State(
        points=points.copy(),
        triangles=np.asarray([[0, 1, 2]], dtype=int),
        fixed=np.ones(3, dtype=bool),
        targets=np.full(3, 1.0),
        chains=[[0, 1], [2]],
        open_nodes=np.empty(0, dtype=int),
        kinds=["island", "island", "island"],
        hard=np.zeros(3, dtype=bool),
        lineage=np.arange(3, dtype=int),
        source_points=points.copy(),
        source_chains=[[0, 1], [2]],
    )
    components = local_topology._inventory_superthin_components(
        state,
        AggressiveConditioningConfig(thin_repair_profile="systematic-v2"),
    )
    assert len(components) == 1
    assert components[0]["classification"] == "under-resolved-passage"
    assert components[0]["local_feature_target_m"] <= 0.05 + 1.0e-12


def _systematic_v3_weld_config(*, obc_policy: str = "redistribute") -> AggressiveConditioningConfig:
    return AggressiveConditioningConfig(
        thin_repair_profile="systematic-v3",
        systematic_v3_obc_policy=obc_policy,
        max_rounds=1,
        enable_pruning=False,
        enable_valence_repair=False,
        max_prunes_per_round=0,
        max_valence_removals_per_round=0,
        systematic_max_components_per_round=0,
        max_boundary_ear_removals_per_round=0,
        max_boundary_welds_per_round=2,
        max_superthin_flips_per_round=0,
        max_collapses_per_round=0,
        max_boundary_edits_per_round=0,
        micro_relax_cycles=0,
        maximum_domain_area_change_fraction=0.02,
    )


def test_systematic_v3_weld_redistributes_obc_and_emits_remap() -> None:
    points, triangles, fixed, chains, kinds = _weld_fixture()
    kinds = ["open"] * 4 + kinds[4:]
    hard = np.zeros(len(points), dtype=bool)
    hard[[0, 1]] = True
    result = condition_mesh_aggressive(
        points,
        triangles,
        fixed,
        chains,
        np.asarray([0, 1], dtype=int),
        target_spacing_m=np.full(len(points), 2.0),
        boundary_kinds=kinds,
        hard_anchor_mask=hard,
        config=_systematic_v3_weld_config(),
    )
    assert result.report["after"]["superthin_triangle_count"] == 0
    assert result.report["edit_counts"].get("boundary-arc-weld", 0) == 1
    assert result.open_boundary_nodes_zero_based.tolist() == [0, 4, 1]
    assert result.obc_remap_manifest["original_obc_count"] == 2
    assert result.obc_remap_manifest["delivered_obc_count"] == 3
    assert result.obc_remap_manifest["obc_forcing_compatible"] is False
    assert result.obc_remap_manifest["forcing_invalidation_required"] is True


def test_systematic_v3_preserve_policy_rolls_back_obc_count_change() -> None:
    points, triangles, fixed, chains, kinds = _weld_fixture()
    kinds = ["open"] * 4 + kinds[4:]
    hard = np.zeros(len(points), dtype=bool)
    hard[[0, 1]] = True
    result = condition_mesh_aggressive(
        points,
        triangles,
        fixed,
        chains,
        np.asarray([0, 1], dtype=int),
        target_spacing_m=np.full(len(points), 2.0),
        boundary_kinds=kinds,
        hard_anchor_mask=hard,
        config=_systematic_v3_weld_config(obc_policy="preserve"),
    )
    assert result.open_boundary_nodes_zero_based.tolist() == [0, 1]
    v3 = result.report["rounds"][0]["aggressive_thin_repair"]["boundary_adaptive_v3"]
    assert "obc_count_changed_under_preserve_policy" in v3["guarded_boundary_ladder"]["rollback_failures"]


def test_systematic_v3_weld_can_snap_to_unchanged_hard_anchor() -> None:
    points, triangles, fixed, chains, kinds = _weld_fixture()
    points[4] = [0.25, 0.01]
    hard = np.zeros(len(points), dtype=bool)
    hard[0] = True
    state = local_topology._State(
        points=points.copy(),
        triangles=triangles.copy(),
        fixed=fixed.copy(),
        targets=np.full(len(points), 2.0),
        chains=[chain.copy() for chain in chains],
        open_nodes=np.empty(0, dtype=int),
        kinds=kinds.copy(),
        hard=hard.copy(),
        lineage=np.arange(len(points), dtype=int),
        source_points=points.copy(),
        source_chains=[chain.copy() for chain in chains],
        source_open_nodes=np.empty(0, dtype=int),
        source_kinds=kinds.copy(),
        initial_domain_area_m2=50.0,
    )
    changed, failures = local_topology._weld_vertex_to_boundary_arc(
        state,
        (0, 1),
        4,
        0,
        AggressiveConditioningConfig(
            thin_repair_profile="systematic-v3",
            maximum_domain_area_change_fraction=0.2,
        ),
    )
    assert changed and not failures
    assert np.array_equal(state.points[0], points[0])
    assert 4 not in set(map(int, state.lineage))
    assert state.ledger[-1]["operation"] == "boundary-arc-weld-snap"
    assert state.ledger[-1]["target_is_hard_anchor"] is True
    assert state.ledger[-1]["intermediate_degenerate_triangles_removed"] >= 1


def test_systematic_v3_source_arc_slide_keeps_hard_endpoints_and_zero_normal_offset() -> None:
    points, triangles, fixed, chains, kinds = _weld_fixture()
    hard = np.zeros(len(points), dtype=bool)
    hard[[0, 2]] = True
    state = local_topology._State(
        points=points.copy(),
        triangles=triangles.copy(),
        fixed=fixed.copy(),
        targets=np.full(len(points), 2.0),
        chains=[chain.copy() for chain in chains],
        open_nodes=np.empty(0, dtype=int),
        kinds=kinds.copy(),
        hard=hard.copy(),
        lineage=np.arange(len(points), dtype=int),
        source_points=points.copy(),
        source_chains=[chain.copy() for chain in chains],
        source_open_nodes=np.empty(0, dtype=int),
        source_kinds=kinds.copy(),
        initial_domain_area_m2=50.0,
    )
    changed, evidence = local_topology._apply_source_arc_windows_v3(
        state,
        [{"chain_index": 0, "nodes": [0, 1, 2], "boundary_kind": "land"}],
        AggressiveConditioningConfig(
            thin_repair_profile="systematic-v3",
            maximum_domain_area_change_fraction=0.2,
        ),
    )
    assert changed and evidence
    assert np.array_equal(state.points[0], points[0])
    assert np.array_equal(state.points[2], points[2])
    assert not np.array_equal(state.points[1], points[1])
    assert local_topology._maximum_boundary_source_arc_deviation(state) <= 1.0e-9


def _nine_spoke_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]], list[str]]:
    count = 9
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    ring = np.column_stack((np.cos(angles), np.sin(angles)))
    points = np.vstack([ring, np.asarray([[0.0, 0.0]])])
    triangles = np.asarray([[count, index, (index + 1) % count] for index in range(count)], dtype=int)
    return (
        points,
        triangles,
        np.asarray([True] * count + [False]),
        [list(range(count))],
        ["land"] * count + ["interior"],
    )


def test_systematic_v5_rebuilds_complete_retained_center_star() -> None:
    points = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 0.01],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=int,
    )
    result = condition_mesh_aggressive(
        points,
        triangles,
        np.asarray([True, True, True, True, False]),
        [[0, 1, 2, 3]],
        np.empty(0, dtype=int),
        target_spacing_m=np.full(len(points), 1.5),
        boundary_kinds=["land", "land", "land", "land", "interior"],
        hard_anchor_mask=np.asarray([True, True, True, True, False]),
        config=_thin_only_config(
            thin_repair_profile="systematic-v5",
            systematic_gate_scope="loop-end",
            systematic_v5_max_star_transactions_per_round=8,
        ),
    )
    assert result.report["after"]["superthin_triangle_count"] == 0
    assert result.report["after"]["nonpositive_signed_area_count"] == 0
    operations = [
        entry
        for entry in result.edit_ledger
        if entry.get("operation") == "systematic-v5-complete-locked-star-reconstruction"
    ]
    assert operations
    assert operations[0]["old_star_triangle_count"] == 4
    assert operations[0]["replacement_triangle_count"] == 4
    assert operations[0]["candidate"].startswith("center-relocation")
    assert np.array_equal(result.nodes_xy[:4], points[:4])


def test_systematic_v5_transaction_rolls_back_impossible_fixed_star() -> None:
    points = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 0.01],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=int,
    )
    result = condition_mesh_aggressive(
        points,
        triangles,
        np.ones(len(points), dtype=bool),
        [[0, 1, 2, 3]],
        np.empty(0, dtype=int),
        target_spacing_m=np.full(len(points), 1.5),
        boundary_kinds=["land", "land", "land", "land", "interior"],
        hard_anchor_mask=np.ones(len(points), dtype=bool),
        config=_thin_only_config(
            thin_repair_profile="systematic-v5",
            systematic_gate_scope="loop-end",
            systematic_v5_max_star_transactions_per_round=4,
        ),
    )
    assert np.array_equal(result.nodes_xy, points)
    assert _canonical_triangles(result.triangles) == _canonical_triangles(triangles)
    assert result.report["after"]["superthin_triangle_count"] > 0
    stage = result.report["rounds"][0]["aggressive_thin_repair"]
    assert stage["closure_succeeded"] is False
    assert stage["blocked_components"]


def test_valence_created_sliver_rolls_back_when_thin_branch_is_disabled() -> None:
    original_repair = local_topology._repair_high_valence
    degraded: dict[str, Any] = {}

    def repair_then_create_sliver(state, config, initial_components):
        report = original_repair(state, config, initial_components)
        movable = np.where(~state.fixed)[0]
        assert len(movable)
        node = int(movable[0])
        degraded["node"] = node
        degraded["coordinate"] = state.points[node].copy()
        neighbor = min(build_edge_topology(len(state.points), state.triangles).node_neighbors[node])
        state.points[node] = state.points[int(neighbor)] + 1.0e-5 * (state.points[node] - state.points[int(neighbor)])
        return report

    local_topology._repair_high_valence = repair_then_create_sliver
    try:
        fixture = _nine_spoke_fixture()
        points, triangles, fixed, chains, kinds = fixture
        result = condition_mesh_aggressive(
            points,
            triangles,
            fixed,
            chains,
            np.empty(0, dtype=int),
            target_spacing_m=np.full(len(points), 1.5),
            boundary_kinds=kinds,
            hard_anchor_mask=np.zeros(len(points), dtype=bool),
            config=AggressiveConditioningConfig(
                max_rounds=1,
                enable_pruning=False,
                enable_thin_repair=False,
                enable_valence_repair=True,
                max_prunes_per_round=0,
                micro_relax_cycles=0,
            ),
        )
    finally:
        local_topology._repair_high_valence = original_repair
    transaction = result.report["rounds"][0]["valence_thin_atomic_transaction"]
    assert degraded
    assert transaction["rolled_back"] is True
    assert any(
        gate in transaction["rejected_gates"]
        for gate in ("new_superthin_triangles", "superthin_severity_regression", "positive_signed_areas")
    )
    assert np.array_equal(result.nodes_xy, points)
    assert _canonical_triangles(result.triangles) == _canonical_triangles(triangles)


def test_valence_created_sliver_is_repaired_inside_same_transaction() -> None:
    original_valence = local_topology._repair_high_valence
    original_thin = local_topology._repair_superthin
    degraded: dict[str, Any] = {}

    def repair_then_create_sliver(state, config, initial_components):
        report = original_valence(state, config, initial_components)
        movable = np.where(~state.fixed)[0]
        assert len(movable)
        node = int(movable[0])
        degraded["node"] = node
        degraded["coordinate"] = state.points[node].copy()
        neighbor = min(build_edge_topology(len(state.points), state.triangles).node_neighbors[node])
        state.points[node] = state.points[int(neighbor)] + 1.0e-5 * (state.points[node] - state.points[int(neighbor)])
        return report

    def restore_created_sliver(state, config, initial_components):
        before = local_topology._summary(state, config)
        if degraded:
            state.points[int(degraded["node"])] = degraded["coordinate"]
            return {"accepted": 1, "rejected": 0, "before": before, "after": local_topology._summary(state, config)}
        return {"accepted": 0, "rejected": 0, "before": before, "after": before}

    local_topology._repair_high_valence = repair_then_create_sliver
    local_topology._repair_superthin = restore_created_sliver
    try:
        points, triangles, fixed, chains, kinds = _nine_spoke_fixture()
        result = condition_mesh_aggressive(
            points,
            triangles,
            fixed,
            chains,
            np.empty(0, dtype=int),
            target_spacing_m=np.full(len(points), 1.5),
            boundary_kinds=kinds,
            hard_anchor_mask=np.zeros(len(points), dtype=bool),
            config=AggressiveConditioningConfig(
                max_rounds=1,
                enable_pruning=False,
                enable_thin_repair=True,
                enable_valence_repair=True,
                max_prunes_per_round=0,
                max_boundary_ear_removals_per_round=1,
                max_boundary_welds_per_round=0,
                max_superthin_flips_per_round=0,
                max_collapses_per_round=0,
                max_boundary_edits_per_round=0,
                micro_relax_cycles=0,
            ),
        )
    finally:
        local_topology._repair_high_valence = original_valence
        local_topology._repair_superthin = original_thin
    transaction = result.report["rounds"][0]["valence_thin_atomic_transaction"]
    assert transaction["accepted"] is True
    assert transaction["rolled_back"] is False
    assert transaction["accepted_operation_count"] >= 2
    assert result.report["after"]["count_valence_above_limit"] == 0
    assert result.report["after"]["superthin_triangle_count"] == 0


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"local-topology-v2 selftests passed ({len(tests)} tests)")


if __name__ == "__main__":
    main()
