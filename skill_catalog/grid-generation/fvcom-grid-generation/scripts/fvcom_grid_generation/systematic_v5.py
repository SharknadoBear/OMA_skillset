"""Adaptive zero-debt relaxation/locked-star closure loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
from typing import Any

import numpy as np

from .interaction_relaxation import (
    InteractionRelaxationConfig,
    relax_mesh_interaction,
)
from .local_topology import (
    AggressiveConditioningConfig,
    LocalTopologyResult,
    condition_mesh_aggressive,
)


@dataclass(frozen=True)
class SystematicV5LoopConfig:
    enabled: bool = True
    total_iterations: int = 1000
    maximum_cycles: int = 6
    burst_ladder: tuple[int, ...] = (10, 25, 50, 100)
    maximum_burst: int = 250
    superthin_trigger: int = 25
    checkpoint_interval: int = 10
    wall_clock_seconds: float = 21600.0
    minimum_champion_gain: float = 1.0e-4
    ladder_advance_gain: float = 1.0e-3
    plateau_gain: float = 1.0e-5
    target_q_l3_sigma: float = 0.75
    maximum_failed_minimum_bursts: int = 2
    maximum_small_net_gains: int = 2
    deadline_monotonic_s: float | None = None


def run_systematic_v5_loop(
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    fixed_node_mask: np.ndarray,
    constraint_chains: list[list[int]],
    open_boundary_nodes_zero_based: np.ndarray,
    *,
    target_spacing_m: np.ndarray,
    boundary_kinds: list[str] | None = None,
    hard_anchor_mask: np.ndarray | None = None,
    target_spacing_sampler: Any = None,
    restricted_lineage_edges: set[tuple[int, int]] | None = None,
    topology_config: AggressiveConditioningConfig | None = None,
    loop_config: SystematicV5LoopConfig | None = None,
) -> LocalTopologyResult:
    """Establish zero debt, relax, close, and retain only improving champions."""
    loop_config = loop_config or SystematicV5LoopConfig()
    topology_config = topology_config or AggressiveConditioningConfig(
        thin_repair_profile="systematic-v5"
    )
    started = time.perf_counter()
    effective_deadline = _effective_deadline(started, loop_config)
    closure_config = replace(
        topology_config,
        thin_repair_profile="systematic-v5",
        systematic_gate_scope="loop-end",
        enable_pruning=False,
        enable_thin_repair=True,
        enable_valence_repair=False,
        max_rounds=1,
        max_prunes_per_round=0,
        max_valence_removals_per_round=0,
        deadline_monotonic_s=effective_deadline,
    )
    initial_closure_started = time.perf_counter()
    champion = condition_mesh_aggressive(
        nodes_xy,
        triangles,
        fixed_node_mask,
        constraint_chains,
        open_boundary_nodes_zero_based,
        target_spacing_m=target_spacing_m,
        boundary_kinds=boundary_kinds,
        hard_anchor_mask=hard_anchor_mask,
        target_spacing_sampler=target_spacing_sampler,
        restricted_lineage_edges=restricted_lineage_edges,
        config=closure_config,
    )
    initial_closure_seconds = float(time.perf_counter() - initial_closure_started)
    champion_lineage = np.asarray(champion.node_lineage, dtype=int).copy()
    champion_restrictions_global = set(
        champion.restricted_lineage_edges
    )
    champion_restrictions_current = (
        _restrictions_to_delivered_indices(champion)
    )
    champion_summary = dict(champion.report["after"])
    initial_zero = _closure_is_structurally_valid(champion, None)
    cycles: list[dict[str, Any]] = []
    cumulative_iterations = 0
    committed_cycles = 0
    failed_minimum_bursts = 0
    small_net_gains = 0
    improving_zero_checkpoints = 0
    lineage_recurrence: dict[str, int] = {}
    ladder = tuple(
        max(1, min(int(value), int(loop_config.maximum_burst)))
        for value in loop_config.burst_ladder
    ) or (1,)
    ladder_index = 0
    status = "initial_closure_failed"
    if initial_zero and bool(loop_config.enabled):
        status = "running"
        for cycle_index in range(max(0, int(loop_config.maximum_cycles))):
            if _deadline_reached(effective_deadline):
                status = "time_limit"
                break
            if cumulative_iterations >= int(loop_config.total_iterations):
                status = "iteration_limit"
                break
            if float(champion_summary["q_l3_sigma"]) > float(loop_config.target_q_l3_sigma):
                status = "quality_target_reached"
                break
            burst = min(
                int(ladder[ladder_index]),
                int(loop_config.maximum_burst),
                int(loop_config.total_iterations) - cumulative_iterations,
            )
            interaction = relax_mesh_interaction(
                champion.nodes_xy,
                champion.triangles,
                champion.fixed_node_mask | champion.hard_anchor_mask,
                target_spacing_m=champion.target_spacing_m,
                config=InteractionRelaxationConfig(
                    iterations=int(burst),
                    checkpoint_interval=min(
                        max(1, int(loop_config.checkpoint_interval)),
                        int(burst),
                    ),
                    superthin_trigger=int(loop_config.superthin_trigger),
                    plateau_gain=float(loop_config.plateau_gain),
                    deadline_monotonic_s=effective_deadline,
                ),
            )
            cumulative_iterations += int(interaction.iterations_completed)
            eligible = [
                checkpoint
                for checkpoint in interaction.checkpoints
                if int(checkpoint.iteration) > 0
                and int(checkpoint.metrics["superthin_triangle_count"])
                <= int(loop_config.superthin_trigger)
            ]
            cycle_report: dict[str, Any] = {
                "cycle": int(cycle_index + 1),
                "burst": int(burst),
                "ladder_index_before": int(ladder_index),
                "champion_before": champion_summary,
                "interaction": interaction.report,
                "committed": False,
            }
            if not eligible:
                ladder_index = max(0, ladder_index - 1)
                failed_minimum_bursts += int(ladder_index == 0)
                cycle_report["rejection_gates"] = ["no_checkpoint_within_thin_trigger"]
                cycle_report["ladder_index_after"] = int(ladder_index)
                cycles.append(cycle_report)
                if failed_minimum_bursts >= int(loop_config.maximum_failed_minimum_bursts):
                    status = "failed_minimum_bursts"
                    break
                continue
            selected = max(
                eligible,
                key=lambda checkpoint: (
                    float(checkpoint.metrics["q_l3_sigma"]),
                    -int(checkpoint.metrics["superthin_triangle_count"]),
                    int(checkpoint.iteration),
                ),
            )
            closure_started = time.perf_counter()
            candidate = condition_mesh_aggressive(
                selected.nodes_xy,
                champion.triangles,
                champion.fixed_node_mask,
                champion.constraint_chains,
                champion.open_boundary_nodes_zero_based,
                target_spacing_m=champion.target_spacing_m,
                boundary_kinds=champion.boundary_kinds,
                hard_anchor_mask=champion.hard_anchor_mask,
                target_spacing_sampler=target_spacing_sampler,
                restricted_lineage_edges=champion_restrictions_current,
                config=closure_config,
            )
            closure_seconds = float(time.perf_counter() - closure_started)
            candidate_lineage = _compose_lineage(
                champion_lineage,
                candidate.node_lineage,
            )
            candidate_restrictions_current = (
                _restrictions_to_delivered_indices(candidate)
            )
            candidate_restrictions_global = {
                tuple(
                    sorted(
                        (
                            int(champion_lineage[left]),
                            int(champion_lineage[right]),
                        )
                    )
                )
                for left, right in candidate.restricted_lineage_edges
                if 0 <= int(left) < len(champion_lineage)
                and 0 <= int(right) < len(champion_lineage)
            }
            candidate_summary = dict(candidate.report["after"])
            closure_stage = _v5_closure_stage(candidate.report)
            recurring = []
            for component_id, count in (
                (closure_stage or {}).get("recurrence_counts", {})
            ).items():
                lineage_recurrence[str(component_id)] = (
                    lineage_recurrence.get(str(component_id), 0) + int(count)
                )
                if lineage_recurrence[str(component_id)] > 1:
                    recurring.append(str(component_id))
            structural = _closure_is_structurally_valid(
                candidate,
                champion_summary,
            )
            gain = float(candidate_summary["q_l3_sigma"]) - float(
                champion_summary["q_l3_sigma"]
            )
            rejection_gates: list[str] = []
            if not structural:
                rejection_gates.append("zero_debt_structural_closure")
            if gain < float(loop_config.minimum_champion_gain):
                rejection_gates.append("champion_q_l3_gain")
            if recurring:
                rejection_gates.append("recurring_repaired_lineage")
            cycle_report.update(
                {
                    "selected_checkpoint_iteration": int(selected.iteration),
                    "selected_checkpoint_metrics": selected.metrics,
                    "closure_runtime_seconds": closure_seconds,
                    "closure": candidate.report,
                    "candidate_after": candidate_summary,
                    "candidate_q_l3_gain": float(gain),
                    "recurring_components": recurring,
                    "rejection_gates": rejection_gates,
                }
            )
            if rejection_gates:
                ladder_index = max(0, ladder_index - 1)
                failed_minimum_bursts += int(ladder_index == 0)
                small_net_gains += int(gain < float(loop_config.minimum_champion_gain))
                cycle_report["ladder_index_after"] = int(ladder_index)
                cycles.append(cycle_report)
                if failed_minimum_bursts >= int(loop_config.maximum_failed_minimum_bursts):
                    status = "failed_minimum_bursts"
                    break
                if small_net_gains >= int(loop_config.maximum_small_net_gains):
                    status = "quality_plateau"
                    break
                continue
            champion = candidate
            champion_lineage = candidate_lineage
            champion_restrictions_current = (
                candidate_restrictions_current
            )
            champion_restrictions_global.update(
                candidate_restrictions_global
            )
            champion.node_lineage = candidate_lineage.copy()
            champion_summary = candidate_summary
            committed_cycles += 1
            improving_zero_checkpoints += 1
            failed_minimum_bursts = 0
            if gain >= float(loop_config.ladder_advance_gain):
                ladder_index = min(len(ladder) - 1, ladder_index + 1)
            cycle_report["committed"] = True
            cycle_report["ladder_index_after"] = int(ladder_index)
            cycles.append(cycle_report)
            if improving_zero_checkpoints >= 3:
                status = "steady_quality_growth_demonstrated"
            if float(champion_summary["q_l3_sigma"]) > float(loop_config.target_q_l3_sigma):
                status = "quality_target_reached"
                break
        else:
            status = "cycle_limit"
    elif initial_zero:
        status = "closure_only"
    report = dict(champion.report)
    report["systematic_v5_loop"] = {
        "schema_version": "fvcom_systematic_v5_loop_v1",
        "settings": asdict(loop_config),
        "status": status,
        "initial_closure_succeeded": bool(initial_zero),
        "initial_closure_runtime_seconds": initial_closure_seconds,
        "cumulative_relaxation_iterations": int(cumulative_iterations),
        "committed_cycle_count": int(committed_cycles),
        "improving_zero_debt_checkpoint_count": int(improving_zero_checkpoints),
        "steady_quality_growth_demonstrated": bool(improving_zero_checkpoints >= 3),
        "lineage_recurrence": dict(sorted(lineage_recurrence.items())),
        "cycles": cycles,
        "champion": champion_summary,
        "deadline_reached": bool(_deadline_reached(effective_deadline)),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    champion.report = report
    champion.node_lineage = champion_lineage
    champion.restricted_lineage_edges = (
        champion_restrictions_global
    )
    return champion


def _closure_is_structurally_valid(
    result: LocalTopologyResult,
    baseline: dict[str, Any] | None,
) -> bool:
    after = result.report["after"]
    invariants = result.report.get("invariants", {})
    singly_limit = (
        int(baseline["singly_connected_triangle_count"])
        if baseline is not None
        else int(result.report["before"]["singly_connected_triangle_count"])
    )
    anomaly_limit = (
        int(baseline["boundary_degree_anomaly_count"])
        if baseline is not None
        else int(result.report["before"]["boundary_degree_anomaly_count"])
    )
    quality_baseline = (
        baseline if baseline is not None else result.report["before"]
    )
    return bool(
        result.report.get("accepted")
        and int(after["superthin_triangle_count"]) == 0
        and int(after["nonpositive_signed_area_count"]) == 0
        and int(after["nonmanifold_edge_count"]) == 0
        and int(after.get("restricted_edge_violation_count", 1)) == 0
        and int(after["connected_component_count"])
        == int(result.report["before"]["connected_component_count"])
        and int(after["singly_connected_triangle_count"]) <= singly_limit
        and int(after["boundary_degree_anomaly_count"]) <= anomaly_limit
        and int(after["count_valence_above_limit"])
        <= int(quality_baseline["count_valence_above_limit"])
        and int(after["valence_excess_sum"])
        <= int(quality_baseline["valence_excess_sum"])
        and float(after["q_l3_sigma"]) + 1.0e-9
        >= float(quality_baseline["q_l3_sigma"])
        and float(after["q_p01"]) + 1.0e-9
        >= float(quality_baseline["q_p01"])
        and int(after["l_over_h_count_above_1_55"])
        <= int(quality_baseline["l_over_h_count_above_1_55"])
        and int(after["area_transition_count_above_0_50"])
        <= int(quality_baseline["area_transition_count_above_0_50"])
        and bool(invariants.get("all_protected_edges_present", False))
        and bool(invariants.get("open_boundary_ordered", False))
        and int(invariants.get("missing_hard_anchor_count", 1)) == 0
        and int(invariants.get("moved_hard_anchor_count", 1)) == 0
    )


def _v5_closure_stage(report: dict[str, Any]) -> dict[str, Any] | None:
    for round_report in report.get("rounds", []):
        stage = round_report.get("aggressive_thin_repair")
        if isinstance(stage, dict) and stage.get("profile") == "systematic-v5":
            return stage
    return None


def _compose_lineage(
    previous: np.ndarray,
    delivered_to_previous: np.ndarray,
) -> np.ndarray:
    previous = np.asarray(previous, dtype=int)
    delivered_to_previous = np.asarray(delivered_to_previous, dtype=int)
    output = np.empty(len(delivered_to_previous), dtype=int)
    next_negative = min(int(np.min(previous)) if len(previous) else 0, 0) - 1
    negative_map: dict[int, int] = {}
    for index, value in enumerate(delivered_to_previous):
        value = int(value)
        if 0 <= value < len(previous):
            output[index] = int(previous[value])
        else:
            if value not in negative_map:
                negative_map[value] = int(next_negative)
                next_negative -= 1
            output[index] = negative_map[value]
    return output


def _restrictions_to_delivered_indices(
    result: LocalTopologyResult,
) -> set[tuple[int, int]]:
    """Remap one closure's source-lineage restrictions after compaction."""
    inverse: dict[int, int] = {}
    for delivered, source in enumerate(
        np.asarray(result.node_lineage, dtype=int)
    ):
        source = int(source)
        if source >= 0 and source not in inverse:
            inverse[source] = int(delivered)
    return {
        tuple(sorted((inverse[int(left)], inverse[int(right)])))
        for left, right in result.restricted_lineage_edges
        if int(left) in inverse and int(right) in inverse
    }


def _effective_deadline(
    started: float,
    config: SystematicV5LoopConfig,
) -> float:
    local = float(started + max(0.0, float(config.wall_clock_seconds)))
    if config.deadline_monotonic_s is None:
        return local
    return min(local, float(config.deadline_monotonic_s))


def _deadline_reached(deadline: float) -> bool:
    return bool(time.perf_counter() >= float(deadline))
