#!/usr/bin/env python3
"""Deterministic policy and transaction smoke tests for systematic V6."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from fvcom_grid_generation.local_topology import (  # noqa: E402
    AggressiveConditioningConfig,
)
from fvcom_grid_generation.systematic_v6 import (  # noqa: E402
    SystematicV6LoopConfig,
    _champion_rejection_gates,
    _closure_round_rejection_gates,
    _compose_lineage,
    _relaxation_entry_eligible,
    _terminal_source_contract_audit,
    _validate_loop_config,
    run_systematic_v6_loop,
)
from fvcom_grid_generation.systematic_v6_policy import (  # noqa: E402
    ADAPTIVE_GATE_POLICY,
    ADAPTIVE_POLICY_LADDER,
    EVIDENCE_RETRY_CEILINGS,
    GATE_POLICY_PRESETS,
    SOFT_TOPOLOGY_GATE_POLICY,
    STRICT_GATE_POLICY,
    TOPOLOGY_ESCROW_GATE_POLICY,
    TOPOLOGY_PRIORITY_GATE_POLICY,
    apply_gate_policy,
    build_evidence_retry_policy,
    resolve_gate_policy_stages,
    topology_policy_overrides,
)
from fvcom_grid_generation.visual_superthin import create_visual_state  # noqa: E402


def _fan() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[list[int]],
    np.ndarray,
]:
    points = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=int,
    )
    fixed = np.asarray([True, True, True, True, False], dtype=bool)
    return points, triangles, fixed, [[0, 1, 2, 3]], np.empty(0, dtype=int)


def _run_fan(**loop_overrides: object):
    points, triangles, fixed, chains, open_nodes = _fan()
    loop = SystematicV6LoopConfig(
        maximum_closure_rounds=1,
        valence_rounds_per_closure=1,
        maximum_relaxation_cycles=0,
        total_relaxation_iterations=0,
        final_audit_reserve_seconds=0.0,
        wall_clock_seconds=30.0,
        passage_removal_enabled=False,
    )
    loop = replace(loop, **loop_overrides)
    result = run_systematic_v6_loop(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=np.full(len(points), 1.5),
        boundary_kinds=["land"] * 4 + ["interior"],
        hard_anchor_mask=fixed,
        topology_config=AggressiveConditioningConfig(
            thin_repair_profile="systematic-v5",
            enable_pruning=False,
            enable_valence_repair=True,
            max_valence=8,
            max_valence_removals_per_round=50,
            micro_relax_cycles=0,
        ),
        loop_config=loop,
    )
    return points, triangles, fixed, result


def test_zero_debt_profile_preserves_source_contract() -> None:
    points, triangles, fixed, result = _run_fan()
    report = result.report
    assert report["v6_zero_debt_pass"] is True
    assert report["terminal_source_contract_audit"]["passed"] is True
    assert report["cumulative_relaxation_iterations"] == 0
    assert report["maximum_closed_q_l3_sigma"] == report["after"]["q_l3_sigma"]
    assert report["maximum_raw_relaxation_q_l3_sigma"] is None
    assert np.array_equal(result.nodes_xy[fixed], points[fixed])
    assert np.array_equal(result.triangles, triangles)


def test_relaxation_candidate_with_insufficient_gain_rolls_back_exactly() -> None:
    points, triangles, fixed, result = _run_fan(
        maximum_relaxation_cycles=1,
        total_relaxation_iterations=3,
        burst_ladder=(3,),
        maximum_burst=3,
        checkpoint_interval=1,
        minimum_champion_gain=1.0,
        target_q_l3_sigma=2.0,
    )
    assert result.report["committed_cycle_count"] == 0
    assert result.report["maximum_raw_relaxation_q_l3_sigma"] is not None
    assert np.array_equal(result.nodes_xy, points)
    assert np.array_equal(result.triangles, triangles)
    assert np.array_equal(result.nodes_xy[fixed], points[fixed])


def test_relaxation_gate_rejects_thin_or_valence_debt() -> None:
    _, _, _, result = _run_fan()
    baseline = dict(result.report["after"])
    thin = dict(baseline)
    thin["superthin_triangle_count"] = 1
    valence = dict(baseline)
    valence["count_valence_above_limit"] = 1
    valence["valence_excess_sum"] = 1
    assert "zero_thin_zero_valence" in _champion_rejection_gates(
        baseline,
        thin,
        minimum_gain=-1.0,
    )
    assert "zero_thin_zero_valence" in _champion_rejection_gates(
        baseline,
        valence,
        minimum_gain=-1.0,
    )
    assert _relaxation_entry_eligible(thin) is False
    assert _relaxation_entry_eligible(valence) is False


def test_strict_gate_rejects_one_l_over_h_tail_defect() -> None:
    _, _, _, result = _run_fan()
    before = dict(result.report["after"])
    after = dict(before)
    after["l_over_h_count_above_1_55"] += 1
    gates = _closure_round_rejection_gates(
        before,
        after,
        loop_config=SystematicV6LoopConfig(
            closure_gate_policy=STRICT_GATE_POLICY
        ),
    )
    assert "l_over_h_tail_regression" in gates


def test_topology_priority_spends_bounded_soft_debt_only_for_progress() -> None:
    _, _, _, result = _run_fan()
    before = dict(result.report["after"])
    before["superthin_triangle_count"] = 2
    before["superthin_severity_sum"] = 2.0
    after = dict(before)
    after["superthin_triangle_count"] = 1
    after["superthin_severity_sum"] = 1.0
    after["l_over_h_count_above_1_55"] += 1
    config = SystematicV6LoopConfig(
        closure_gate_policy=TOPOLOGY_PRIORITY_GATE_POLICY,
        closure_max_l_over_h_count_increase=1,
        closure_max_minimum_angle_p01_decrease=0.5,
    )
    assert not _closure_round_rejection_gates(
        before,
        after,
        loop_config=config,
        soft_baseline=before,
    )
    no_progress = dict(before)
    no_progress["l_over_h_count_above_1_55"] += 1
    assert "soft_debt_without_primary_progress" in _closure_round_rejection_gates(
        before,
        no_progress,
        loop_config=config,
        soft_baseline=before,
    )
    over_budget = dict(after)
    over_budget["l_over_h_count_above_1_55"] += 1
    assert (
        "l_over_h_count_soft_budget_exceeded"
        in _closure_round_rejection_gates(
            before,
            over_budget,
            loop_config=config,
            soft_baseline=before,
        )
    )
    angle_over_budget = dict(after)
    angle_over_budget["minimum_angle_p01_deg"] = (
        float(before["minimum_angle_p01_deg"]) - 0.6
    )
    assert (
        "minimum_angle_p01_soft_budget_exceeded"
        in _closure_round_rejection_gates(
            before,
            angle_over_budget,
            loop_config=config,
            soft_baseline=before,
        )
    )


def test_topology_escrow_accepts_bounded_valence_trade_and_thin_repayment() -> None:
    _, _, _, result = _run_fan()
    baseline = dict(result.report["after"])
    baseline.update(
        {
            "superthin_triangle_count": 3,
            "superthin_severity_sum": 5.023,
            "count_valence_above_limit": 309,
            "valence_excess_sum": 326,
            "maximum_valence": 11,
        }
    )
    valence_trade = dict(baseline)
    valence_trade.update(
        {
            "superthin_triangle_count": 14,
            "superthin_severity_sum": 13.714,
            "count_valence_above_limit": 8,
            "valence_excess_sum": 11,
            "maximum_valence": 10,
        }
    )
    config = SystematicV6LoopConfig(
        closure_gate_policy=TOPOLOGY_ESCROW_GATE_POLICY,
    )
    assert not _closure_round_rejection_gates(
        baseline,
        valence_trade,
        loop_config=config,
        soft_baseline=baseline,
    )

    thin_repayment = dict(valence_trade)
    thin_repayment.update(
        {
            "superthin_triangle_count": 8,
            "superthin_severity_sum": 9.0,
            "count_valence_above_limit": 16,
            "valence_excess_sum": 25,
            "maximum_valence": 11,
        }
    )
    assert not _closure_round_rejection_gates(
        valence_trade,
        thin_repayment,
        loop_config=config,
        soft_baseline=baseline,
    )


def test_topology_escrow_rejects_unbounded_or_dual_debt() -> None:
    _, _, _, result = _run_fan()
    baseline = dict(result.report["after"])
    baseline.update(
        {
            "superthin_triangle_count": 3,
            "superthin_severity_sum": 5.0,
            "count_valence_above_limit": 100,
            "valence_excess_sum": 120,
            "maximum_valence": 11,
        }
    )
    config = SystematicV6LoopConfig(
        closure_gate_policy=TOPOLOGY_ESCROW_GATE_POLICY,
    )
    too_thin = dict(baseline)
    too_thin.update(
        {
            "superthin_triangle_count": 26,
            "superthin_severity_sum": 20.0,
            "count_valence_above_limit": 60,
            "valence_excess_sum": 80,
        }
    )
    assert (
        "topology_escrow_superthin_count_exceeded"
        in _closure_round_rejection_gates(
            baseline,
            too_thin,
            loop_config=config,
            soft_baseline=baseline,
        )
    )
    dual = dict(baseline)
    dual.update(
        {
            "superthin_triangle_count": 4,
            "superthin_severity_sum": 6.0,
            "count_valence_above_limit": 101,
            "valence_excess_sum": 121,
        }
    )
    assert (
        "topology_escrow_dual_debt_regression"
        in _closure_round_rejection_gates(
            baseline,
            dual,
            loop_config=config,
            soft_baseline=baseline,
        )
    )


def test_gate_policy_validation_rejects_invalid_escrow_fraction() -> None:
    try:
        _validate_loop_config(
            SystematicV6LoopConfig(
                closure_gate_policy=TOPOLOGY_ESCROW_GATE_POLICY,
                escrow_large_valence_reduction_fraction=1.1,
            )
        )
    except ValueError as exc:
        assert "within [0, 1]" in str(exc)
    else:
        raise AssertionError("invalid escrow fraction was accepted")


def test_fixed_policy_presets_and_adaptive_order_are_reusable() -> None:
    resolved = resolve_gate_policy_stages(ADAPTIVE_GATE_POLICY)
    assert tuple(resolved["stage_order"]) == ADAPTIVE_POLICY_LADDER
    assert resolved["stages"][0]["name"] == STRICT_GATE_POLICY
    assert resolved["stages"][2]["name"] == SOFT_TOPOLOGY_GATE_POLICY
    soft = apply_gate_policy(
        SystematicV6LoopConfig(),
        SOFT_TOPOLOGY_GATE_POLICY,
    )
    assert soft.closure_gate_policy == TOPOLOGY_PRIORITY_GATE_POLICY
    assert soft.closure_max_l_over_h_count_increase == 8
    assert (
        topology_policy_overrides(SOFT_TOPOLOGY_GATE_POLICY)[
            "max_valence_l_over_h_count_increase"
        ]
        == 4
    )


def test_generic_v6_defaults_exclude_case_passage_authority() -> None:
    config = SystematicV6LoopConfig()
    assert config.passage_removal_enabled is False
    assert config.allow_authorized_topology_delta is False
    assert config.known_passage_node_ids_1based == ()
    driver = (
        SCRIPT_DIR
        / "research"
        / "delaware"
        / "run_systematic_v6_overnight.py"
    ).read_text(encoding="utf-8")
    assert "DELAWARE_PASSAGE_NODE_IDS_1BASED" in driver
    assert "known_passage_node_ids_1based" in driver
    assert "(95, 106911, 106926)" in driver


def test_deadline_safe_exit_returns_exact_valid_baseline() -> None:
    points, triangles, fixed, result = _run_fan(
        wall_clock_seconds=0.0,
        deadline_monotonic_s=time.perf_counter(),
    )
    assert np.array_equal(result.nodes_xy, points)
    assert np.array_equal(result.triangles, triangles)
    assert np.array_equal(result.nodes_xy[fixed], points[fixed])
    assert result.report["terminal_source_contract_audit"]["passed"] is True
    assert result.report["work_deadline_reached"] is True


def test_lineage_rebase_is_deterministic_for_created_nodes() -> None:
    previous = np.asarray([100, 101, 102], dtype=int)
    delivered = np.asarray([2, -1, 0, -1, -2], dtype=int)
    first = _compose_lineage(previous, delivered)
    second = _compose_lineage(previous, delivered)
    assert np.array_equal(first, second)
    assert first.tolist() == [102, -1, 100, -1, -2]


def test_terminal_contract_rejects_obc_loss_and_hard_anchor_motion() -> None:
    points, triangles, fixed, chains, _ = _fan()
    state, _, _ = create_visual_state(
        points,
        triangles,
        fixed,
        chains,
        np.asarray([0, 1], dtype=int),
        target_spacing_m=np.full(len(points), 1.5),
        boundary_kinds=["open", "open", "land", "land", "interior"],
        hard_anchor_mask=fixed,
    )
    assert _terminal_source_contract_audit(state)["passed"] is True
    changed_obc = state.clone()
    changed_obc.open_nodes = np.asarray([1, 0], dtype=int)
    assert _terminal_source_contract_audit(changed_obc)["passed"] is False
    changed_hard = state.clone()
    changed_hard.points[0, 0] += 0.1
    assert _terminal_source_contract_audit(changed_hard)["passed"] is False


def test_evidence_retry_derives_one_bounded_soft_budget() -> None:
    _, _, _, result = _run_fan()
    baseline = dict(result.report["after"])
    before = dict(baseline)
    before.update(
        {
            "superthin_triangle_count": 2,
            "superthin_severity_sum": 2.0,
            "count_valence_above_limit": 4,
            "valence_excess_sum": 5,
        }
    )
    trial = dict(before)
    trial.update(
        {
            "superthin_triangle_count": 1,
            "superthin_severity_sum": 1.0,
            "l_over_h_count_above_1_55": (
                int(baseline["l_over_h_count_above_1_55"]) + 10
            ),
        }
    )
    decision = build_evidence_retry_policy(
        baseline,
        before,
        trial,
        ["l_over_h_count_soft_budget_exceeded"],
        GATE_POLICY_PRESETS["topology-escrow-v1"],
    )
    assert decision["eligible"] is True
    assert decision["policy"]["closure_l_over_h_count_increase"] == 12
    assert decision["policy"]["valence_l_over_h_count_increase"] == 5
    assert (
        decision["policy"]["closure_l_over_h_count_increase"]
        <= EVIDENCE_RETRY_CEILINGS[
            "closure_l_over_h_count_increase"
        ]
    )


def test_evidence_retry_rejects_hard_gate_or_missing_progress() -> None:
    _, _, _, result = _run_fan()
    baseline = dict(result.report["after"])
    before = dict(baseline)
    before["superthin_triangle_count"] = 2
    before["superthin_severity_sum"] = 2.0
    improved = dict(before)
    improved["superthin_triangle_count"] = 1
    improved["superthin_severity_sum"] = 1.0
    base_policy = GATE_POLICY_PRESETS["topology-escrow-v1"]
    hard = build_evidence_retry_policy(
        baseline,
        before,
        improved,
        ["nonmanifold_topology"],
        base_policy,
    )
    assert hard["eligible"] is False
    assert hard["reason"] == "nonsoft_rejection_gate"
    no_progress = build_evidence_retry_policy(
        baseline,
        before,
        before,
        ["l_over_h_count_soft_budget_exceeded"],
        base_policy,
    )
    assert no_progress["eligible"] is False
    assert no_progress["reason"] == "no_primary_topology_progress"


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"systematic-v6 selftests passed ({len(tests)} tests)")


if __name__ == "__main__":
    main()
