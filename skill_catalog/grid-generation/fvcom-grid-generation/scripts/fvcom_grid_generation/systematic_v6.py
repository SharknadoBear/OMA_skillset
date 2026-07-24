"""Research-only V6 passage/valence/interaction conditioning loop.

V6 deliberately keeps the ordinary automatic profiles unchanged.  It reuses
the complete V5 topology-preserving closure, enters whole-passage removal only
after that closure plateaus, clears valence and any resulting thin debt as one
outer transaction, and compares relaxation candidates only at zero-thin,
zero-valence checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Any, Iterable

import numpy as np

from .interaction_relaxation import InteractionRelaxationConfig, relax_mesh_interaction
from .local_topology import (
    AggressiveConditioningConfig,
    LocalTopologyResult,
    _State,
    _boundary_graph_audit,
    _inventory_superthin_components,
    _summary,
    condition_mesh_aggressive,
)
from .metrics import build_edge_topology, chain_edges
from .thin_passage import (
    ThinPassageRemovalConfig,
    infer_passage_removal_candidates,
    try_remove_thin_passage,
)
from .systematic_v6_policy import (
    GATE_POLICIES,
    STRICT_GATE_POLICY,
    TOPOLOGY_ESCROW_GATE_POLICY,
    TOPOLOGY_PRIORITY_GATE_POLICY,
)
from .visual_superthin import create_visual_state


REPORT_SCHEMA = "fvcom_systematic_v6_loop_v1"


@dataclass(frozen=True)
class SystematicV6LoopConfig:
    enabled: bool = True
    maximum_closure_rounds: int = 8
    valence_rounds_per_closure: int = 4
    maximum_relaxation_cycles: int = 12
    total_relaxation_iterations: int = 1000
    burst_ladder: tuple[int, ...] = (10, 25, 50, 100)
    maximum_burst: int = 100
    checkpoint_interval: int = 10
    superthin_trigger: int = 25
    wall_clock_seconds: float = 28800.0
    final_audit_reserve_seconds: float = 3600.0
    minimum_champion_gain: float = 1.0e-4
    ladder_advance_gain: float = 1.0e-3
    target_q_l3_sigma: float = 0.75
    plateau_gain: float = 1.0e-5
    closure_gate_policy: str = STRICT_GATE_POLICY
    closure_max_q_l3_sigma_decrease: float = 0.0
    closure_max_q_p01_decrease: float = 0.0
    closure_max_minimum_angle_p01_decrease: float = 0.0
    closure_max_l_over_h_count_increase: int = 0
    closure_max_l_over_h_p95_increase: float = 0.0
    closure_max_l_over_h_maximum_increase: float = 0.0
    closure_max_area_transition_count_increase: int = 0
    escrow_maximum_superthin_count: int = 25
    escrow_maximum_superthin_severity: float = 25.0
    escrow_maximum_valence: int = 12
    escrow_maximum_valence_count_rebound: int = 8
    escrow_maximum_valence_excess_rebound: int = 16
    escrow_large_valence_threshold: int = 16
    escrow_large_valence_reduction_fraction: float = 0.25
    escrow_large_valence_minimum_reduction: int = 32
    escrow_maximum_alternating_rounds: int = 3
    passage_removal_enabled: bool = False
    allow_authorized_topology_delta: bool = False
    known_passage_node_ids_1based: tuple[tuple[int, ...], ...] = ()
    passage_patch_rings: int = 4
    passage_overresolved_edge_fraction: float = 0.55
    passage_maximum_inferred_bank_nodes: int = 8
    passage_maximum_removed_triangles: int = 128
    deadline_monotonic_s: float | None = None


def run_systematic_v6_loop(
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    fixed_node_mask: np.ndarray,
    constraint_chains: list[list[int]],
    open_boundary_nodes_zero_based: np.ndarray,
    *,
    target_spacing_m: np.ndarray,
    boundary_kinds: list[str] | None = None,
    hard_anchor_mask: np.ndarray | None = None,
    node_lineage: np.ndarray | None = None,
    source_contract: dict[str, Any] | None = None,
    target_spacing_sampler: Any = None,
    restricted_lineage_edges: set[tuple[int, int]] | None = None,
    topology_config: AggressiveConditioningConfig | None = None,
    loop_config: SystematicV6LoopConfig | None = None,
    closure_soft_baseline: dict[str, Any] | None = None,
) -> LocalTopologyResult:
    """Run V6 closure and interaction relaxation on one frozen mesh.

    ``closure_soft_baseline`` lets an external adaptive driver keep one
    cumulative soft-debt ledger while it advances through successively looser
    conditioning policies.  Post-relaxation closure still anchors itself to
    the current exact-zero champion.
    """
    loop_config = loop_config or SystematicV6LoopConfig()
    _validate_loop_config(loop_config)
    topology_config = topology_config or AggressiveConditioningConfig(
        thin_repair_profile="systematic-v5"
    )
    started = time.perf_counter()
    hard_deadline = _effective_deadline(started, loop_config)
    work_deadline = max(
        started,
        hard_deadline - max(0.0, float(loop_config.final_audit_reserve_seconds)),
    )
    state, _, _ = create_visual_state(
        nodes_xy,
        triangles,
        fixed_node_mask,
        constraint_chains,
        open_boundary_nodes_zero_based,
        target_spacing_m=target_spacing_m,
        boundary_kinds=boundary_kinds,
        hard_anchor_mask=hard_anchor_mask,
        node_lineage=(
            np.arange(len(nodes_xy), dtype=int)
            if node_lineage is None
            else np.asarray(node_lineage, dtype=int)
        ),
        restricted_lineage_edges=restricted_lineage_edges,
    )
    state.target_sampler = target_spacing_sampler
    if source_contract is not None:
        _restore_source_contract(state, source_contract)
    initial_source = _source_contract(state)
    initial = _summary(state, topology_config)

    state, closure_report = _establish_zero_debt(
        state,
        topology_config,
        loop_config,
        work_deadline,
        soft_baseline_override=closure_soft_baseline,
    )
    champion = _summary(state, topology_config)
    maximum_closed_q_l3_sigma: float | None = (
        float(champion["q_l3_sigma"])
        if _relaxation_entry_eligible(champion)
        else None
    )
    maximum_raw_relaxation_q_l3_sigma: float | None = None
    cycles: list[dict[str, Any]] = []
    ladder = tuple(
        max(1, min(int(value), int(loop_config.maximum_burst)))
        for value in loop_config.burst_ladder
    ) or (1,)
    ladder_index = 0
    cumulative_iterations = 0
    committed_cycles = 0
    improving_zero_checkpoints = 0
    status = str(closure_report["status"])

    if _relaxation_entry_eligible(champion) and bool(loop_config.enabled):
        status = "zero_debt_champion"
        for cycle_index in range(max(0, int(loop_config.maximum_relaxation_cycles))):
            if _deadline_reached(work_deadline):
                status = "terminal_audit_reserve"
                break
            if cumulative_iterations >= int(loop_config.total_relaxation_iterations):
                status = "iteration_limit"
                break
            if float(champion["q_l3_sigma"]) > float(loop_config.target_q_l3_sigma):
                status = "quality_target_reached"
                break
            burst = min(
                int(ladder[ladder_index]),
                int(loop_config.maximum_burst),
                int(loop_config.total_relaxation_iterations) - cumulative_iterations,
            )
            interaction = relax_mesh_interaction(
                state.points,
                state.triangles,
                state.fixed | state.hard,
                target_spacing_m=state.targets,
                config=InteractionRelaxationConfig(
                    iterations=burst,
                    checkpoint_interval=min(
                        max(1, int(loop_config.checkpoint_interval)),
                        burst,
                    ),
                    superthin_trigger=int(loop_config.superthin_trigger),
                    plateau_gain=float(loop_config.plateau_gain),
                    deadline_monotonic_s=work_deadline,
                ),
            )
            checkpoint_q = [
                float(checkpoint.metrics["q_l3_sigma"])
                for checkpoint in interaction.checkpoints
            ]
            if checkpoint_q:
                raw_maximum = max(checkpoint_q)
                maximum_raw_relaxation_q_l3_sigma = (
                    raw_maximum
                    if maximum_raw_relaxation_q_l3_sigma is None
                    else max(maximum_raw_relaxation_q_l3_sigma, raw_maximum)
                )
            cumulative_iterations += int(interaction.iterations_completed)
            eligible = [
                checkpoint
                for checkpoint in interaction.checkpoints
                if int(checkpoint.iteration) > 0
                and int(checkpoint.metrics["superthin_triangle_count"])
                <= int(loop_config.superthin_trigger)
            ]
            cycle: dict[str, Any] = {
                "cycle": int(cycle_index + 1),
                "burst": int(burst),
                "champion_before": champion,
                "interaction": interaction.report,
                "committed": False,
            }
            if not eligible:
                cycle["rejection_gates"] = ["no_checkpoint_within_thin_trigger"]
                cycles.append(cycle)
                ladder_index = max(0, ladder_index - 1)
                status = "interaction_no_eligible_checkpoint"
                continue
            selected = max(
                eligible,
                key=lambda item: (
                    float(item.metrics["q_l3_sigma"]),
                    -int(item.metrics["superthin_triangle_count"]),
                    int(item.iteration),
                ),
            )
            candidate = state.clone()
            candidate.points = np.asarray(selected.nodes_xy, dtype=float).copy()
            candidate, candidate_closure = _establish_zero_debt(
                candidate,
                topology_config,
                loop_config,
                work_deadline,
            )
            candidate_summary = _summary(candidate, topology_config)
            gain = float(candidate_summary["q_l3_sigma"]) - float(
                champion["q_l3_sigma"]
            )
            rejection_gates = _champion_rejection_gates(
                champion,
                candidate_summary,
                minimum_gain=float(loop_config.minimum_champion_gain),
            )
            cycle.update(
                {
                    "selected_checkpoint_iteration": int(selected.iteration),
                    "selected_checkpoint_metrics": selected.metrics,
                    "closure": candidate_closure,
                    "candidate_after": candidate_summary,
                    "candidate_q_l3_gain": float(gain),
                    "rejection_gates": rejection_gates,
                }
            )
            if rejection_gates:
                cycles.append(cycle)
                ladder_index = max(0, ladder_index - 1)
                status = (
                    "quality_plateau"
                    if gain < float(loop_config.minimum_champion_gain)
                    else "candidate_gate_failure"
                )
                continue
            state = candidate
            _restore_source_contract(state, initial_source)
            champion = candidate_summary
            maximum_closed_q_l3_sigma = (
                float(champion["q_l3_sigma"])
                if maximum_closed_q_l3_sigma is None
                else max(
                    maximum_closed_q_l3_sigma,
                    float(champion["q_l3_sigma"]),
                )
            )
            committed_cycles += 1
            improving_zero_checkpoints += 1
            cycle["committed"] = True
            cycles.append(cycle)
            if gain >= float(loop_config.ladder_advance_gain):
                ladder_index = min(len(ladder) - 1, ladder_index + 1)
            if float(champion["q_l3_sigma"]) > float(loop_config.target_q_l3_sigma):
                status = "quality_target_reached"
                break
        else:
            status = "cycle_limit"

    final = _summary(state, topology_config)
    source_contract_audit = _terminal_source_contract_audit(state)
    relaxation_entry_failures = _relaxation_entry_failures(final)
    if not bool(source_contract_audit["passed"]):
        relaxation_entry_failures.append("source_contract")
    relaxation_entry_failures = sorted(set(relaxation_entry_failures))
    residual = _serializable_component_inventory(state, topology_config)
    standard_ready = bool(
        _zero_debt(final)
        and float(final["q_l3_sigma"]) > float(loop_config.target_q_l3_sigma)
        and int(final["connected_component_count"]) == 1
        and _structural_summary_valid(final)
        and bool(source_contract_audit["passed"])
    )
    authorized_ready = bool(
        _zero_debt(final)
        and float(final["q_l3_sigma"]) > float(loop_config.target_q_l3_sigma)
        and _structural_summary_valid(final)
        and bool(source_contract_audit["passed"])
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "profile": "systematic-v6",
        "settings": asdict(loop_config),
        "topology_settings": asdict(topology_config),
        "status": status,
        "before": initial,
        "after": final,
        "initial_closure": closure_report,
        "cycles": cycles,
        "cumulative_relaxation_iterations": int(cumulative_iterations),
        "maximum_closed_q_l3_sigma": maximum_closed_q_l3_sigma,
        "maximum_raw_relaxation_q_l3_sigma": (
            maximum_raw_relaxation_q_l3_sigma
        ),
        "delivered_q_l3_sigma": float(final["q_l3_sigma"]),
        "committed_cycle_count": int(committed_cycles),
        "improving_zero_debt_checkpoint_count": int(improving_zero_checkpoints),
        "steady_quality_growth_demonstrated": bool(
            improving_zero_checkpoints >= 3
        ),
        "accepted": bool(
            _zero_debt(final)
            and _structural_summary_valid(final)
            and source_contract_audit["passed"]
        ),
        "fvcom_valence_gate_passed": bool(
            int(final["count_valence_above_limit"]) == 0
        ),
        "superthin_gate_passed": bool(
            int(final["superthin_triangle_count"]) == 0
        ),
        "terminal_topology_gate_passed": bool(
            _zero_debt(final)
            and _structural_summary_valid(final)
            and source_contract_audit["passed"]
        ),
        "relaxation_entry_gate_passed": bool(not relaxation_entry_failures),
        "relaxation_entry_rejection_gates": relaxation_entry_failures,
        "v6_zero_debt_pass": bool(_zero_debt(final)),
        "v6_quality_target_pass": bool(
            float(final["q_l3_sigma"]) > float(loop_config.target_q_l3_sigma)
        ),
        "authorized_topology_smoke_ready": authorized_ready,
        "standard_catalog_ready": standard_ready,
        "standard_catalog_multiple_component_warning": bool(
            int(final["connected_component_count"]) != 1
        ),
        "terminal_source_contract_audit": source_contract_audit,
        "residual_components": residual,
        "failure_hypotheses": (
            _failure_hypotheses(residual) if not _zero_debt(final) else []
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "work_deadline_reached": bool(_deadline_reached(work_deadline)),
        "hard_deadline_reached": bool(_deadline_reached(hard_deadline)),
    }
    return _state_result(state, report)


def _establish_zero_debt(
    state: _State,
    topology_config: AggressiveConditioningConfig,
    loop_config: SystematicV6LoopConfig,
    deadline: float,
    *,
    soft_baseline_override: dict[str, Any] | None = None,
) -> tuple[_State, dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    status = "closure_round_limit"
    soft_baseline = (
        _summary(state, topology_config)
        if soft_baseline_override is None
        else dict(soft_baseline_override)
    )
    topology_debt_states = {_topology_debt_signature(soft_baseline)}
    escrow_alternating_rounds = 0
    for round_index in range(max(0, int(loop_config.maximum_closure_rounds))):
        if _deadline_reached(deadline):
            status = "terminal_audit_reserve"
            break
        before_state = state.clone()
        before = _summary(state, topology_config)
        state, thin_report = _condition_state(
            state,
            topology_config,
            deadline,
            enable_valence=False,
            max_rounds=1,
        )
        state, first_passage = _passage_sweep(
            state,
            topology_config,
            loop_config,
            deadline,
        )
        state, valence_report = _condition_state(
            state,
            topology_config,
            deadline,
            enable_valence=True,
            max_rounds=int(loop_config.valence_rounds_per_closure),
        )
        state, second_passage = _passage_sweep(
            state,
            topology_config,
            loop_config,
            deadline,
        )
        after = _summary(state, topology_config)
        rejection_gates = _closure_round_rejection_gates(
            before,
            after,
            loop_config=loop_config,
            soft_baseline=soft_baseline,
        )
        escrow_direction = _topology_escrow_direction(before, after)
        if (
            str(loop_config.closure_gate_policy)
            == TOPOLOGY_ESCROW_GATE_POLICY
            and escrow_direction != "nonregressing"
        ):
            if (
                escrow_alternating_rounds
                >= int(loop_config.escrow_maximum_alternating_rounds)
            ):
                rejection_gates.append("topology_escrow_round_limit")
            debt_signature = _topology_debt_signature(after)
            if debt_signature in topology_debt_states and not _zero_debt(after):
                rejection_gates.append("topology_escrow_repeated_debt_state")
        rejection_gates = sorted(set(rejection_gates))
        accepted = not rejection_gates
        round_report = {
            "round": int(round_index + 1),
            "before": before,
            "topology_preserving_closure": thin_report,
            "passage_sweep_before_valence": first_passage,
            "valence_and_thin_closure": valence_report,
            "passage_sweep_after_valence": second_passage,
            "trial_after": after,
            "accepted": bool(accepted),
            "rejection_gates": rejection_gates,
            "topology_escrow_direction": escrow_direction,
            "topology_escrow_alternating_rounds_before": int(
                escrow_alternating_rounds
            ),
            "soft_debt_budget": _soft_debt_budget_report(
                soft_baseline,
                after,
                loop_config,
            ),
        }
        if not accepted:
            state = before_state
            round_report["rolled_back"] = True
            round_report["after"] = _summary(state, topology_config)
            rounds.append(round_report)
            status = "closure_round_gate_failure"
            break
        round_report["rolled_back"] = False
        round_report["after"] = after
        topology_debt_states.add(_topology_debt_signature(after))
        if (
            str(loop_config.closure_gate_policy)
            == TOPOLOGY_ESCROW_GATE_POLICY
            and escrow_direction != "nonregressing"
        ):
            escrow_alternating_rounds += 1
        round_report["topology_escrow_alternating_rounds_after"] = int(
            escrow_alternating_rounds
        )
        rounds.append(round_report)
        if _zero_debt(after):
            status = "zero_debt"
            break
        if not _strict_closure_progress(before, after):
            status = "closure_plateau"
            break
    return state, {
        "schema_version": "fvcom_systematic_v6_zero_debt_closure_v1",
        "status": status,
        "gate_policy": str(loop_config.closure_gate_policy),
        "soft_debt_baseline": soft_baseline,
        "soft_debt_budget": _soft_debt_budget_report(
            soft_baseline,
            _summary(state, topology_config),
            loop_config,
        ),
        "topology_escrow_alternating_rounds": int(escrow_alternating_rounds),
        "rounds": rounds,
        "after": _summary(state, topology_config),
    }


def _condition_state(
    state: _State,
    topology_config: AggressiveConditioningConfig,
    deadline: float,
    *,
    enable_valence: bool,
    max_rounds: int,
) -> tuple[_State, dict[str, Any]]:
    base_lineage = np.asarray(state.lineage, dtype=int).copy()
    current_restrictions = _global_restrictions_to_current(
        base_lineage,
        state.restricted_lineage_edges,
    )
    config = replace(
        topology_config,
        thin_repair_profile="systematic-v5",
        systematic_gate_scope="loop-end",
        systematic_v5_enable_boundary_window_fallback=False,
        enable_pruning=False,
        enable_thin_repair=True,
        enable_valence_repair=bool(enable_valence),
        max_rounds=max(1, int(max_rounds)),
        max_prunes_per_round=0,
        max_valence_removals_per_round=(
            max(1, int(topology_config.max_valence_removals_per_round))
            if enable_valence
            else 0
        ),
        deadline_monotonic_s=deadline,
    )
    result = condition_mesh_aggressive(
        state.points,
        state.triangles,
        state.fixed,
        state.chains,
        state.open_nodes,
        target_spacing_m=state.targets,
        boundary_kinds=state.kinds,
        hard_anchor_mask=state.hard,
        target_spacing_sampler=state.target_sampler,
        restricted_lineage_edges=current_restrictions,
        config=config,
    )
    composed = _compose_lineage(base_lineage, result.node_lineage)
    restrictions = _compose_restrictions(
        base_lineage,
        result.restricted_lineage_edges,
    )
    delivered = _state_from_result(
        state,
        result,
        composed,
        restrictions,
    )
    return delivered, result.report


def _passage_sweep(
    state: _State,
    topology_config: AggressiveConditioningConfig,
    loop_config: SystematicV6LoopConfig,
    deadline: float,
) -> tuple[_State, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    accepted = 0
    blocked: set[tuple[int, ...]] = set()
    if not bool(loop_config.passage_removal_enabled):
        return state, {
            "enabled": False,
            "accepted": 0,
            "attempts": [],
            "status": "disabled",
        }
    passage_config = ThinPassageRemovalConfig(
        patch_rings=int(loop_config.passage_patch_rings),
        overresolved_edge_fraction=float(
            loop_config.passage_overresolved_edge_fraction
        ),
        maximum_inferred_bank_nodes=int(
            loop_config.passage_maximum_inferred_bank_nodes
        ),
        maximum_removed_triangles=int(
            loop_config.passage_maximum_removed_triangles
        ),
        maximum_dangling_peel_rounds=0,
    )
    while not _deadline_reached(deadline):
        components = _inventory_superthin_components(state, topology_config)
        if not components:
            break
        progressed = False
        for component in components:
            lineage_key = tuple(
                sorted(
                    int(state.lineage[int(node)])
                    for node in np.unique(
                        state.triangles[
                            np.asarray(component["triangle_indices"], dtype=int)
                        ]
                    )
                )
            )
            if lineage_key in blocked:
                continue
            candidates: list[list[int]] = []
            evidence: dict[str, Any] = {
                "mode": "systematic-v6-last-resort",
                "topology_preserving_routes_exhausted": True,
            }
            for node_ids in loop_config.known_passage_node_ids_1based:
                lineages = [int(value) - 1 for value in node_ids]
                if (
                    set(lineages).issubset(set(map(int, state.lineage)))
                    and set(lineages) & set(lineage_key)
                ):
                    candidates.append(lineages)
                    evidence.setdefault("known_candidate_ids_1based", []).append(
                        list(map(int, node_ids))
                    )
            try:
                inferred, inferred_evidence = infer_passage_removal_candidates(
                    state,
                    component,
                    config=passage_config,
                )
                candidates.extend(inferred)
                evidence["bilateral_inference"] = inferred_evidence
            except (ValueError, RuntimeError) as exc:
                evidence["bilateral_inference_failure"] = str(exc)
            candidates = _deduplicate_candidates(candidates)
            if not candidates:
                blocked.add(lineage_key)
                records.append(
                    {
                        "component_id": str(component["component_id"]),
                        "accepted": False,
                        "status": "no_bounded_passage_candidate",
                        "component_lineages": list(lineage_key),
                        "inference": evidence,
                    }
                )
                continue
            trial, report = try_remove_thin_passage(
                state,
                str(component["component_id"]),
                candidates,
                expected_boundary_component_delta=None,
                expected_wet_component_delta=None,
                human_approved=True,
                allow_authorized_topology_delta=bool(
                    loop_config.allow_authorized_topology_delta
                ),
                inference_evidence=evidence,
                config=passage_config,
            )
            report["component_lineages"] = list(lineage_key)
            records.append(report)
            if bool(report["accepted"]):
                state = _rebase_state(trial)
                accepted += 1
                blocked.clear()
                progressed = True
                break
            blocked.add(lineage_key)
        if not progressed:
            break
    return state, {
        "enabled": True,
        "accepted": int(accepted),
        "attempts": records,
        "status": (
            "zero_superthin"
            if int(_summary(state, topology_config)["superthin_triangle_count"]) == 0
            else "passage_plateau"
        ),
    }


def _state_from_result(
    previous: _State,
    result: LocalTopologyResult,
    lineage: np.ndarray,
    restrictions: set[tuple[int, int]],
) -> _State:
    delivered, _, _ = create_visual_state(
        result.nodes_xy,
        result.triangles,
        result.fixed_node_mask,
        result.constraint_chains,
        result.open_boundary_nodes_zero_based,
        target_spacing_m=result.target_spacing_m,
        boundary_kinds=result.boundary_kinds,
        hard_anchor_mask=result.hard_anchor_mask,
        node_lineage=lineage,
        restricted_lineage_edges=restrictions,
    )
    delivered.target_sampler = previous.target_sampler
    delivered.ledger = [
        *[dict(value) for value in previous.ledger],
        *[dict(value) for value in result.edit_ledger],
    ]
    delivered.cumulative_boundary_area_change_m2 = float(
        previous.cumulative_boundary_area_change_m2
    )
    _restore_source_contract(delivered, _source_contract(previous))
    return delivered


def _rebase_state(state: _State) -> _State:
    source = _source_contract(state)
    rebased, _, _ = create_visual_state(
        state.points,
        state.triangles,
        state.fixed,
        state.chains,
        state.open_nodes,
        target_spacing_m=state.targets,
        boundary_kinds=state.kinds,
        hard_anchor_mask=state.hard,
        node_lineage=state.lineage,
        restricted_lineage_edges=state.restricted_lineage_edges,
    )
    rebased.target_sampler = state.target_sampler
    rebased.ledger = [dict(value) for value in state.ledger]
    rebased.cumulative_boundary_area_change_m2 = float(
        state.cumulative_boundary_area_change_m2
    )
    _restore_source_contract(rebased, source)
    return rebased


def _source_contract(state: _State) -> dict[str, Any]:
    return {
        "source_points": np.asarray(state.source_points, dtype=float).copy(),
        "source_chains": [list(map(int, chain)) for chain in state.source_chains],
        "source_open_nodes": np.asarray(state.source_open_nodes, dtype=int).copy(),
        "source_kinds": list(state.source_kinds),
        "source_hard_anchor_lineage": np.asarray(
            state.source_hard_anchor_lineage,
            dtype=int,
        ).copy(),
    }


def _restore_source_contract(state: _State, source: dict[str, Any]) -> None:
    state.source_points = np.asarray(source["source_points"], dtype=float).copy()
    state.source_chains = [
        list(map(int, chain)) for chain in source["source_chains"]
    ]
    state.source_open_nodes = np.asarray(
        source["source_open_nodes"],
        dtype=int,
    ).copy()
    state.source_kinds = list(source["source_kinds"])
    state.source_hard_anchor_lineage = np.asarray(
        source["source_hard_anchor_lineage"],
        dtype=int,
    ).copy()


def _state_result(state: _State, report: dict[str, Any]) -> LocalTopologyResult:
    return LocalTopologyResult(
        nodes_xy=np.asarray(state.points, dtype=float).copy(),
        triangles=np.asarray(state.triangles, dtype=int).copy(),
        fixed_node_mask=np.asarray(state.fixed, dtype=bool).copy(),
        target_spacing_m=np.asarray(state.targets, dtype=float).copy(),
        constraint_chains=[list(map(int, chain)) for chain in state.chains],
        open_boundary_nodes_zero_based=np.asarray(state.open_nodes, dtype=int).copy(),
        boundary_kinds=list(state.kinds),
        hard_anchor_mask=np.asarray(state.hard, dtype=bool).copy(),
        node_lineage=np.asarray(state.lineage, dtype=int).copy(),
        restricted_lineage_edges=set(state.restricted_lineage_edges),
        report=report,
        edit_ledger=[dict(value) for value in state.ledger],
        obc_remap_manifest={
            "original_open_boundary_source_lineage": [
                int(value) for value in state.source_open_nodes
            ],
            "delivered_open_boundary_source_lineage": [
                int(state.lineage[int(node)]) for node in state.open_nodes
            ],
            "source_lineage_order_unchanged": bool(
                [
                    int(state.lineage[int(node)])
                    for node in state.open_nodes
                ]
                == [int(value) for value in state.source_open_nodes]
            ),
        },
    )


def _terminal_source_contract_audit(state: _State) -> dict[str, Any]:
    lineage_to_node = {
        int(value): int(index) for index, value in enumerate(state.lineage)
    }
    source_hard = list(
        map(int, np.asarray(state.source_hard_anchor_lineage, dtype=int))
    )
    missing_hard = [
        int(value) for value in source_hard if int(value) not in lineage_to_node
    ]
    moved_hard = [
        int(value)
        for value in source_hard
        if int(value) in lineage_to_node
        and not np.array_equal(
            state.points[lineage_to_node[int(value)]],
            state.source_points[int(value)],
        )
    ]
    source_open = list(map(int, np.asarray(state.source_open_nodes, dtype=int)))
    delivered_open = [
        int(state.lineage[int(node)])
        for node in np.asarray(state.open_nodes, dtype=int)
    ]
    passed = bool(
        not missing_hard
        and not moved_hard
        and delivered_open == source_open
    )
    return {
        "passed": passed,
        "source_hard_anchor_count": int(len(source_hard)),
        "missing_hard_anchor_lineages": missing_hard,
        "moved_hard_anchor_lineages": moved_hard,
        "source_open_boundary_count": int(len(source_open)),
        "delivered_open_boundary_count": int(len(delivered_open)),
        "source_open_boundary_lineage": source_open,
        "delivered_open_boundary_lineage": delivered_open,
        "open_boundary_lineage_order_unchanged": bool(
            delivered_open == source_open
        ),
    }


def _compose_lineage(previous: np.ndarray, delivered: np.ndarray) -> np.ndarray:
    previous = np.asarray(previous, dtype=int)
    delivered = np.asarray(delivered, dtype=int)
    output = np.empty(len(delivered), dtype=int)
    next_negative = min(
        int(np.min(previous)) if len(previous) else 0,
        0,
    ) - 1
    created: dict[int, int] = {}
    for index, value in enumerate(delivered):
        value = int(value)
        if 0 <= value < len(previous):
            output[index] = int(previous[value])
        else:
            if value not in created:
                created[value] = int(next_negative)
                next_negative -= 1
            output[index] = int(created[value])
    return output


def _global_restrictions_to_current(
    lineage: np.ndarray,
    restrictions: Iterable[tuple[int, int]],
) -> set[tuple[int, int]]:
    inverse = {int(value): int(index) for index, value in enumerate(lineage)}
    return {
        tuple(sorted((inverse[int(left)], inverse[int(right)])))
        for left, right in restrictions
        if int(left) in inverse and int(right) in inverse
    }


def _compose_restrictions(
    previous: np.ndarray,
    restrictions: Iterable[tuple[int, int]],
) -> set[tuple[int, int]]:
    previous = np.asarray(previous, dtype=int)
    return {
        tuple(sorted((int(previous[int(left)]), int(previous[int(right)]))))
        for left, right in restrictions
        if 0 <= int(left) < len(previous) and 0 <= int(right) < len(previous)
    }


def _deduplicate_candidates(values: Iterable[Iterable[int]]) -> list[list[int]]:
    output: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for value in values:
        key = tuple(sorted(set(map(int, value))))
        if key and key not in seen:
            output.append(list(key))
            seen.add(key)
    return output


def _closure_round_rejection_gates(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    loop_config: SystematicV6LoopConfig | None = None,
    soft_baseline: dict[str, Any] | None = None,
) -> list[str]:
    loop_config = loop_config or SystematicV6LoopConfig()
    policy = str(loop_config.closure_gate_policy)
    soft_baseline = before if soft_baseline is None else soft_baseline
    failures: list[str] = []
    if not _structural_summary_valid(after):
        failures.append("structural_invariants")
    if policy == TOPOLOGY_ESCROW_GATE_POLICY:
        failures.extend(
            _topology_escrow_rejection_gates(
                before,
                after,
                soft_baseline,
                loop_config,
            )
        )
    else:
        if int(after["superthin_triangle_count"]) > int(
            before["superthin_triangle_count"]
        ):
            failures.append("superthin_count_regression")
        if int(after["count_valence_above_limit"]) > int(
            before["count_valence_above_limit"]
        ):
            failures.append("valence_count_regression")
        if int(after["valence_excess_sum"]) > int(before["valence_excess_sum"]):
            failures.append("valence_excess_regression")
    if policy == STRICT_GATE_POLICY:
        if float(after["q_l3_sigma"]) + 1.0e-9 < float(before["q_l3_sigma"]):
            failures.append("q_l3_sigma_regression")
        if float(after["q_p01"]) + 1.0e-9 < float(before["q_p01"]):
            failures.append("q_p01_regression")
        if int(after["l_over_h_count_above_1_55"]) > int(
            before["l_over_h_count_above_1_55"]
        ):
            failures.append("l_over_h_tail_regression")
        if int(after["area_transition_count_above_0_50"]) > int(
            before["area_transition_count_above_0_50"]
        ):
            failures.append("area_transition_regression")
        return failures
    if float(after["q_l3_sigma"]) + 1.0e-9 < (
        float(soft_baseline["q_l3_sigma"])
        - float(loop_config.closure_max_q_l3_sigma_decrease)
    ):
        failures.append("q_l3_sigma_soft_budget_exceeded")
    if float(after["q_p01"]) + 1.0e-9 < (
        float(soft_baseline["q_p01"])
        - float(loop_config.closure_max_q_p01_decrease)
    ):
        failures.append("q_p01_soft_budget_exceeded")
    if float(after.get("minimum_angle_p01_deg", 0.0)) + 1.0e-9 < (
        float(soft_baseline.get("minimum_angle_p01_deg", 0.0))
        - float(loop_config.closure_max_minimum_angle_p01_decrease)
    ):
        failures.append("minimum_angle_p01_soft_budget_exceeded")
    if int(after["l_over_h_count_above_1_55"]) > (
        int(soft_baseline["l_over_h_count_above_1_55"])
        + int(loop_config.closure_max_l_over_h_count_increase)
    ):
        failures.append("l_over_h_count_soft_budget_exceeded")
    if float(after["l_over_h_p95"]) > (
        float(soft_baseline["l_over_h_p95"])
        + float(loop_config.closure_max_l_over_h_p95_increase)
        + 1.0e-12
    ):
        failures.append("l_over_h_p95_soft_budget_exceeded")
    if float(after["l_over_h_maximum"]) > (
        float(soft_baseline["l_over_h_maximum"])
        + float(loop_config.closure_max_l_over_h_maximum_increase)
        + 1.0e-12
    ):
        failures.append("l_over_h_maximum_soft_budget_exceeded")
    if int(after["area_transition_count_above_0_50"]) > (
        int(soft_baseline["area_transition_count_above_0_50"])
        + int(loop_config.closure_max_area_transition_count_increase)
    ):
        failures.append("area_transition_soft_budget_exceeded")
    if (
        _soft_debt_regressed(before, after)
        and not _strict_closure_progress(before, after)
    ):
        failures.append("soft_debt_without_primary_progress")
    return failures


def _topology_escrow_rejection_gates(
    before: dict[str, Any],
    after: dict[str, Any],
    baseline: dict[str, Any],
    loop_config: SystematicV6LoopConfig,
) -> list[str]:
    """Bound alternating thin/valence debt without weakening hard gates."""
    failures: list[str] = []
    before_thin = int(before["superthin_triangle_count"])
    after_thin = int(after["superthin_triangle_count"])
    before_severity = float(before.get("superthin_severity_sum", 0.0))
    after_severity = float(after.get("superthin_severity_sum", 0.0))
    before_valence = int(before["count_valence_above_limit"])
    after_valence = int(after["count_valence_above_limit"])
    before_excess = int(before["valence_excess_sum"])
    after_excess = int(after["valence_excess_sum"])

    thin_regressed = bool(
        after_thin > before_thin
        or after_severity > before_severity + 1.0e-10
    )
    valence_regressed = bool(
        after_valence > before_valence or after_excess > before_excess
    )
    if thin_regressed and valence_regressed:
        failures.append("topology_escrow_dual_debt_regression")
        return failures

    maximum_valence_cap = max(
        int(loop_config.escrow_maximum_valence),
        int(baseline.get("maximum_valence", 0)),
    )
    if int(after.get("maximum_valence", 0)) > maximum_valence_cap:
        failures.append("topology_escrow_maximum_valence_exceeded")

    if thin_regressed:
        required_valence_reduction = 1
        if before_valence > int(loop_config.escrow_large_valence_threshold):
            required_valence_reduction = min(
                int(
                    math.ceil(
                        float(loop_config.escrow_large_valence_reduction_fraction)
                        * float(before_valence)
                    )
                ),
                int(loop_config.escrow_large_valence_minimum_reduction),
            )
            required_valence_reduction = max(1, required_valence_reduction)
        if before_valence - after_valence < required_valence_reduction:
            failures.append(
                "topology_escrow_insufficient_valence_count_reduction"
            )
        if after_excess >= before_excess:
            failures.append(
                "topology_escrow_valence_excess_not_reduced"
            )
        if after_thin > int(loop_config.escrow_maximum_superthin_count):
            failures.append("topology_escrow_superthin_count_exceeded")
        if (
            after_severity
            > float(loop_config.escrow_maximum_superthin_severity) + 1.0e-10
        ):
            failures.append("topology_escrow_superthin_severity_exceeded")
    elif valence_regressed:
        thin_repaid = bool(
            after_thin < before_thin
            or after_severity + 1.0e-10 < before_severity
        )
        if not thin_repaid:
            failures.append("topology_escrow_thin_debt_not_repaid")
        if (
            after_valence - before_valence
            > int(loop_config.escrow_maximum_valence_count_rebound)
        ):
            failures.append("topology_escrow_valence_count_rebound_exceeded")
        if (
            after_excess - before_excess
            > int(loop_config.escrow_maximum_valence_excess_rebound)
        ):
            failures.append("topology_escrow_valence_excess_rebound_exceeded")
        if after_valence > int(baseline["count_valence_above_limit"]):
            failures.append("topology_escrow_valence_count_baseline_exceeded")
        if after_excess > int(baseline["valence_excess_sum"]):
            failures.append("topology_escrow_valence_excess_baseline_exceeded")
    return failures


def _topology_escrow_direction(
    before: dict[str, Any],
    after: dict[str, Any],
) -> str:
    thin_regressed = bool(
        int(after["superthin_triangle_count"])
        > int(before["superthin_triangle_count"])
        or float(after.get("superthin_severity_sum", 0.0))
        > float(before.get("superthin_severity_sum", 0.0)) + 1.0e-10
    )
    valence_regressed = bool(
        int(after["count_valence_above_limit"])
        > int(before["count_valence_above_limit"])
        or int(after["valence_excess_sum"]) > int(before["valence_excess_sum"])
    )
    if thin_regressed and valence_regressed:
        return "dual_regression"
    if thin_regressed:
        return "valence_trade"
    if valence_regressed:
        return "thin_repayment"
    return "nonregressing"


def _topology_debt_signature(summary: dict[str, Any]) -> tuple[int, float, int, int]:
    return (
        int(summary["superthin_triangle_count"]),
        round(float(summary.get("superthin_severity_sum", 0.0)), 8),
        int(summary["count_valence_above_limit"]),
        int(summary["valence_excess_sum"]),
    )


def _soft_debt_budget_report(
    baseline: dict[str, Any],
    after: dict[str, Any],
    loop_config: SystematicV6LoopConfig,
) -> dict[str, Any]:
    """Report soft-debt use against one fixed closure-segment baseline."""
    budget = {
        "q_l3_sigma_decrease": float(
            loop_config.closure_max_q_l3_sigma_decrease
        ),
        "q_p01_decrease": float(loop_config.closure_max_q_p01_decrease),
        "minimum_angle_p01_decrease": float(
            loop_config.closure_max_minimum_angle_p01_decrease
        ),
        "l_over_h_count_increase": int(
            loop_config.closure_max_l_over_h_count_increase
        ),
        "l_over_h_p95_increase": float(
            loop_config.closure_max_l_over_h_p95_increase
        ),
        "l_over_h_maximum_increase": float(
            loop_config.closure_max_l_over_h_maximum_increase
        ),
        "area_transition_count_increase": int(
            loop_config.closure_max_area_transition_count_increase
        ),
    }
    used = {
        "q_l3_sigma_decrease": max(
            0.0,
            float(baseline["q_l3_sigma"]) - float(after["q_l3_sigma"]),
        ),
        "q_p01_decrease": max(
            0.0,
            float(baseline["q_p01"]) - float(after["q_p01"]),
        ),
        "minimum_angle_p01_decrease": max(
            0.0,
            float(baseline.get("minimum_angle_p01_deg", 0.0))
            - float(after.get("minimum_angle_p01_deg", 0.0)),
        ),
        "l_over_h_count_increase": max(
            0,
            int(after["l_over_h_count_above_1_55"])
            - int(baseline["l_over_h_count_above_1_55"]),
        ),
        "l_over_h_p95_increase": max(
            0.0,
            float(after["l_over_h_p95"]) - float(baseline["l_over_h_p95"]),
        ),
        "l_over_h_maximum_increase": max(
            0.0,
            float(after["l_over_h_maximum"])
            - float(baseline["l_over_h_maximum"]),
        ),
        "area_transition_count_increase": max(
            0,
            int(after["area_transition_count_above_0_50"])
            - int(baseline["area_transition_count_above_0_50"]),
        ),
    }
    within = {
        key: bool(float(used[key]) <= float(budget[key]) + 1.0e-12)
        for key in budget
    }
    return {
        "policy": str(loop_config.closure_gate_policy),
        "budget": budget,
        "used": used,
        "remaining": {
            key: max(0.0, float(budget[key]) - float(used[key]))
            for key in budget
        },
        "within_budget": within,
        "passed": bool(all(within.values())),
    }


def _soft_debt_regressed(
    before: dict[str, Any],
    after: dict[str, Any],
) -> bool:
    return bool(
        float(after["q_l3_sigma"]) + 1.0e-9 < float(before["q_l3_sigma"])
        or float(after["q_p01"]) + 1.0e-9 < float(before["q_p01"])
        or float(after.get("minimum_angle_p01_deg", 0.0)) + 1.0e-9
        < float(before.get("minimum_angle_p01_deg", 0.0))
        or int(after["l_over_h_count_above_1_55"])
        > int(before["l_over_h_count_above_1_55"])
        or float(after["l_over_h_p95"]) > float(before["l_over_h_p95"]) + 1.0e-12
        or float(after["l_over_h_maximum"])
        > float(before["l_over_h_maximum"]) + 1.0e-12
        or int(after["area_transition_count_above_0_50"])
        > int(before["area_transition_count_above_0_50"])
    )


def _champion_rejection_gates(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    minimum_gain: float,
) -> list[str]:
    failures = _closure_round_rejection_gates(before, after)
    if not _zero_debt(after):
        failures.append("zero_thin_zero_valence")
    if float(after["q_l3_sigma"]) - float(before["q_l3_sigma"]) < minimum_gain:
        failures.append("minimum_q_l3_gain")
    if float(after.get("minimum_angle_p01_deg", 0.0)) + 1.0e-8 < float(
        before.get("minimum_angle_p01_deg", 0.0)
    ):
        failures.append("minimum_angle_p01_regression")
    if int(after["singly_connected_triangle_count"]) > int(
        before["singly_connected_triangle_count"]
    ):
        failures.append("singly_connected_regression")
    if int(after["boundary_degree_anomaly_count"]) > int(
        before["boundary_degree_anomaly_count"]
    ):
        failures.append("boundary_degree_regression")
    return sorted(set(failures))


def _strict_closure_progress(
    before: dict[str, Any],
    after: dict[str, Any],
) -> bool:
    return bool(
        int(after["superthin_triangle_count"])
        < int(before["superthin_triangle_count"])
        or float(after.get("superthin_severity_sum", 0.0)) + 1.0e-10
        < float(before.get("superthin_severity_sum", 0.0))
        or int(after["count_valence_above_limit"])
        < int(before["count_valence_above_limit"])
        or int(after["valence_excess_sum"]) < int(before["valence_excess_sum"])
    )


def _zero_debt(summary: dict[str, Any]) -> bool:
    return bool(
        int(summary["superthin_triangle_count"]) == 0
        and int(summary["count_valence_above_limit"]) == 0
        and int(summary.get("restricted_edge_violation_count", 1)) == 0
    )


def _relaxation_entry_failures(summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if int(summary["superthin_triangle_count"]) != 0:
        failures.append("superthin_debt")
    if int(summary["count_valence_above_limit"]) != 0:
        failures.append("valence_debt")
    if int(summary.get("restricted_edge_violation_count", 1)) != 0:
        failures.append("restricted_edge_debt")
    if int(summary["nonpositive_signed_area_count"]) != 0:
        failures.append("nonpositive_signed_areas")
    if int(summary["nonmanifold_edge_count"]) != 0:
        failures.append("nonmanifold_edges")
    if int(summary["boundary_degree_anomaly_count"]) != 0:
        failures.append("boundary_degree_anomalies")
    return failures


def _relaxation_entry_eligible(summary: dict[str, Any]) -> bool:
    return bool(not _relaxation_entry_failures(summary))


def _structural_summary_valid(summary: dict[str, Any]) -> bool:
    return bool(
        int(summary["nonpositive_signed_area_count"]) == 0
        and int(summary["nonmanifold_edge_count"]) == 0
        and int(summary["boundary_degree_anomaly_count"]) == 0
        and int(summary.get("restricted_edge_violation_count", 1)) == 0
    )


def _serializable_component_inventory(
    state: _State,
    topology_config: AggressiveConditioningConfig,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for component in _inventory_superthin_components(state, topology_config):
        record = {
            key: value
            for key, value in component.items()
            if key != "triangle_indices"
        }
        record["triangle_indices_zero_based"] = [
            int(value) for value in component["triangle_indices"]
        ]
        output.append(record)
    return output


def _failure_hypotheses(
    components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lineages = [
        {
            "component_id": str(value.get("component_id")),
            "classification": str(value.get("classification", "unknown")),
            "node_lineage": list(map(int, value.get("node_lineage", []))),
        }
        for value in components
    ]
    return [
        {
            "hypothesis": "expanded_coupled_minmax_cavity",
            "execute": False,
            "patch_ring_ladder": [8, 12],
            "objective": "joint zero-superthin and true-neighbor valence<=8",
            "protected_chord_partition": True,
            "affected_components": lineages,
        },
        {
            "hypothesis": "upstream_boundary_topography_adjustment",
            "execute": False,
            "routes": [
                "bilateral physical-channel regularization to at least four elements across",
                "scientifically dispensable wet-corridor closure",
                "OBC source-arc reparameterization with regenerated inward front",
            ],
            "affected_components": lineages,
            "requires_human_review": True,
        },
    ]


def _effective_deadline(started: float, config: SystematicV6LoopConfig) -> float:
    local = float(started + max(0.0, float(config.wall_clock_seconds)))
    if config.deadline_monotonic_s is None:
        return local
    return min(local, float(config.deadline_monotonic_s))


def _validate_loop_config(config: SystematicV6LoopConfig) -> None:
    if str(config.closure_gate_policy) not in GATE_POLICIES:
        raise ValueError(
            "closure_gate_policy must be one of "
            + ", ".join(map(str, GATE_POLICIES))
        )
    nonnegative = {
        "closure_max_q_l3_sigma_decrease": (
            config.closure_max_q_l3_sigma_decrease
        ),
        "closure_max_q_p01_decrease": config.closure_max_q_p01_decrease,
        "closure_max_minimum_angle_p01_decrease": (
            config.closure_max_minimum_angle_p01_decrease
        ),
        "closure_max_l_over_h_count_increase": (
            config.closure_max_l_over_h_count_increase
        ),
        "closure_max_l_over_h_p95_increase": (
            config.closure_max_l_over_h_p95_increase
        ),
        "closure_max_l_over_h_maximum_increase": (
            config.closure_max_l_over_h_maximum_increase
        ),
        "closure_max_area_transition_count_increase": (
            config.closure_max_area_transition_count_increase
        ),
        "escrow_maximum_superthin_count": (
            config.escrow_maximum_superthin_count
        ),
        "escrow_maximum_superthin_severity": (
            config.escrow_maximum_superthin_severity
        ),
        "escrow_maximum_valence": config.escrow_maximum_valence,
        "escrow_maximum_valence_count_rebound": (
            config.escrow_maximum_valence_count_rebound
        ),
        "escrow_maximum_valence_excess_rebound": (
            config.escrow_maximum_valence_excess_rebound
        ),
        "escrow_large_valence_threshold": (
            config.escrow_large_valence_threshold
        ),
        "escrow_large_valence_reduction_fraction": (
            config.escrow_large_valence_reduction_fraction
        ),
        "escrow_large_valence_minimum_reduction": (
            config.escrow_large_valence_minimum_reduction
        ),
        "escrow_maximum_alternating_rounds": (
            config.escrow_maximum_alternating_rounds
        ),
    }
    invalid = [name for name, value in nonnegative.items() if float(value) < 0.0]
    if invalid:
        raise ValueError(
            "closure soft-debt budgets must be nonnegative: "
            + ", ".join(invalid)
        )
    if float(config.escrow_large_valence_reduction_fraction) > 1.0:
        raise ValueError(
            "escrow_large_valence_reduction_fraction must be within [0, 1]"
        )


def _deadline_reached(deadline: float) -> bool:
    return bool(time.perf_counter() >= float(deadline))
