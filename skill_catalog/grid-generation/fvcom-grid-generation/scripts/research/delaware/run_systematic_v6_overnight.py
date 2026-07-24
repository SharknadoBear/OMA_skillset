#!/usr/bin/env python3
"""Reproduce the Delaware adaptive exact-zero conditioning experiment.

The driver progressively tests strict, topology-priority, bounded-soft, and
topology-escrow closure policies on both frozen pilots. It starts interaction
relaxation only from an exact zero-superthin, zero-valence, zero-restricted-edge
checkpoint and preserves the final audit window.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon


SCRIPT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from condition_mesh_local import (  # noqa: E402
    _bbox,
    _boundary_geojson,
    _boundary_metadata,
    _json_safe,
    _remap_depths,
    _serialized_roundtrip_audit,
    _target_sizes,
)
from fvcom_grid_generation.local_topology import (  # noqa: E402
    AggressiveConditioningConfig,
)
from fvcom_grid_generation import local_topology as topology  # noqa: E402
from fvcom_grid_generation.metrics import (  # noqa: E402
    build_edge_topology,
    chain_edges,
)
from fvcom_grid_generation.projection import (  # noqa: E402
    local_utm_projection,
    project_points,
    unproject_points,
)
from fvcom_grid_generation.sms_2dm import read_2dm, write_2dm  # noqa: E402
from fvcom_grid_generation.systematic_v6 import (  # noqa: E402
    SystematicV6LoopConfig,
    run_systematic_v6_loop,
)
from fvcom_grid_generation.systematic_v6_policy import (  # noqa: E402
    ADAPTIVE_GATE_POLICY,
    FIXED_GATE_POLICIES,
    build_evidence_retry_policy,
    resolve_gate_policy_stages,
)
from fvcom_grid_generation.visual_superthin import (  # noqa: E402
    create_visual_state,
)


EXPECTED_INPUTS = {
    "iteration3": {
        "mesh": (
            "Workspace/Preprocessing/fvcom-grid-generation/runs/"
            "delaware_superthin_connectivity_v1_20260719/"
            "iteration3_full_v5_closure_r3.2dm"
        ),
        "boundary": (
            "Workspace/Preprocessing/fvcom-grid-generation/runs/"
            "delaware_superthin_connectivity_v1_20260719/"
            "iteration3_full_v5_boundary_nodes_r3.geojson"
        ),
        "sha256": (
            "1893321C48D5BD819D8360F868879B7A7565D41F2BB7487BAB1958FCF85518E1"
        ),
    },
    "r5": {
        "mesh": (
            "Workspace/Preprocessing/fvcom-grid-generation/runs/"
            "delaware_superthin_connectivity_v1_20260719/"
            "r5_full_v5_closure_r3.2dm"
        ),
        "boundary": (
            "Workspace/Preprocessing/fvcom-grid-generation/runs/"
            "delaware_superthin_connectivity_v1_20260719/"
            "r5_full_v5_boundary_nodes_r3.geojson"
        ),
        "sha256": (
            "3AF9781E15AB26E7AC0385A7445E50879948E17EA3FB274EBCD42AC7C0888624"
        ),
    },
}
SIZE_FIELD = (
    "Workspace/Preprocessing/fvcom-grid-generation/runs/"
    "delaware_v2_prevention_only_20260713_r5/size_field.nc"
)
DELAWARE_PASSAGE_NODE_IDS_1BASED = ((95, 106911, 106926),)
THIN16_COMPONENT = {
    "component_id": "thin-16-7abd9f8f29",
    "triangle_indices_zero_based": (170, 200595),
    "node_lineage": (16, 17, 109929, 109930, 109931),
    "expected_support_counts": tuple(range(2, 9)),
}


@dataclass
class Pilot:
    name: str
    mesh_path: Path
    boundary_path: Path
    mesh: Any
    projection: Any
    points: np.ndarray
    triangles: np.ndarray
    fixed: np.ndarray
    chains: list[list[int]]
    open_nodes: np.ndarray
    targets: np.ndarray
    kinds: list[str]
    hard: np.ndarray
    lineage: np.ndarray
    restrictions: set[tuple[int, int]]
    source_contract: dict[str, Any]
    closure_soft_baseline: dict[str, Any] | None = None
    report: dict[str, Any] = field(default_factory=dict)
    segment_paths: list[str] = field(default_factory=list)
    segment_summaries: list[dict[str, Any]] = field(default_factory=list)
    passage_replays: list[dict[str, Any]] = field(default_factory=list)
    policy_history: list[dict[str, Any]] = field(default_factory=list)
    relaxation_history: list[dict[str, Any]] = field(default_factory=list)
    active: bool = False
    relaxation_entry_achieved: bool = False
    relaxation_entry_segment_index: int | None = None
    relaxation_gate_policy: dict[str, Any] | None = None
    policy_ladder_exhausted: bool = False
    ladder_index: int = 0
    consecutive_improving: int = 0
    maximum_closed_q_l3_sigma: float = float("-inf")
    maximum_closed_q_l3_segment_index: int | None = None
    maximum_raw_relaxation_q_l3_sigma: float = float("-inf")
    maximum_raw_relaxation_segment_index: int | None = None
    maximum_raw_relaxation_iteration: int | None = None
    evidence_retry_decision: dict[str, Any] = field(default_factory=dict)
    extended_cavity_decision: dict[str, Any] = field(default_factory=dict)


class Journal:
    def __init__(self, path: Path):
        self.path = path

    def emit(self, event: str, **values: Any) -> None:
        record = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "event": str(event),
            **_json_safe(values),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def _resolve_gate_policy(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    overrides = {
        "valence_l_over_h_count_increase": (
            args.valence_l_over_h_count_budget
        ),
        "closure_q_l3_sigma_decrease": args.closure_q_l3_decrease_budget,
        "closure_q_p01_decrease": args.closure_q_p01_decrease_budget,
        "closure_minimum_angle_p01_decrease": (
            args.closure_minimum_angle_p01_decrease_budget
        ),
        "closure_l_over_h_count_increase": (
            args.closure_l_over_h_count_budget
        ),
        "closure_l_over_h_p95_increase": args.closure_l_over_h_p95_budget,
        "closure_l_over_h_maximum_increase": (
            args.closure_l_over_h_maximum_budget
        ),
        "closure_area_transition_count_increase": (
            args.closure_area_transition_count_budget
        ),
    }
    try:
        return resolve_gate_policy_stages(
            str(args.gate_policy),
            overrides,
        )
    except ValueError as exc:
        parser.error(str(exc))
        raise


def _last_rejected_closure_round(
    report: dict[str, Any],
) -> dict[str, Any] | None:
    rounds = report.get("initial_closure", {}).get("rounds", [])
    rejected = [
        value
        for value in rounds
        if value.get("rejection_gates")
        and not bool(value.get("accepted", False))
    ]
    return rejected[-1] if rejected else None


def _evidence_retry_decision(
    pilot: Pilot,
    base_policy: dict[str, Any],
) -> dict[str, Any]:
    rejected = _last_rejected_closure_round(pilot.report)
    if rejected is None:
        return {
            "schema_version": "fvcom_systematic_v6_evidence_retry_v1",
            "eligible": False,
            "reason": "no_rejected_closure_round",
        }
    baseline = pilot.closure_soft_baseline or rejected.get("before", {})
    return build_evidence_retry_policy(
        baseline,
        rejected["before"],
        rejected["trial_after"],
        list(rejected.get("rejection_gates", [])),
        base_policy,
    )


def _structural_after_valid(report: dict[str, Any]) -> bool:
    after = report.get("after", {})
    return bool(
        int(after.get("nonpositive_signed_area_count", 1)) == 0
        and int(after.get("nonmanifold_edge_count", 1)) == 0
        and int(after.get("boundary_degree_anomaly_count", 1)) == 0
        and int(after.get("restricted_edge_violation_count", 1)) == 0
        and bool(
            report.get("terminal_source_contract_audit", {}).get(
                "passed",
                False,
            )
        )
    )


def _extended_cavity_decision(
    pilot: Pilot,
    base_policy: dict[str, Any],
) -> dict[str, Any]:
    closure_status = str(
        pilot.report.get("initial_closure", {}).get("status", "")
    )
    eligible = bool(
        not _exact_zero_debt(pilot.report)
        and _structural_after_valid(pilot.report)
        and closure_status in {"closure_plateau", "closure_round_limit"}
    )
    decision: dict[str, Any] = {
        "schema_version": "fvcom_systematic_v6_extended_cavity_v1",
        "eligible": eligible,
        "closure_status": closure_status,
        "reason": (
            "bounded_no_legal_candidate_fallback"
            if eligible
            else "closure_not_at_safe_topology_plateau"
        ),
    }
    if eligible:
        policy = dict(base_policy)
        policy.update(
            {
                "name": "topology-extended-cavity-v1",
                "patch_ring_ladder": (1, 2, 4, 8, 12),
            }
        )
        decision["policy"] = policy
    return decision


def _exact_zero_debt(report: dict[str, Any]) -> bool:
    after = report.get("after", {})
    return bool(
        report.get("v6_zero_debt_pass", False)
        and int(after.get("superthin_triangle_count", -1)) == 0
        and int(after.get("count_valence_above_limit", -1)) == 0
        and int(after.get("restricted_edge_violation_count", -1)) == 0
    )


def _track_quality_maxima(pilot: Pilot, segment_index: int) -> None:
    after = pilot.report.get("after", {})
    closed_value = float(after.get("q_l3_sigma", float("-inf")))
    if (
        _exact_zero_debt(pilot.report)
        and closed_value > float(pilot.maximum_closed_q_l3_sigma)
    ):
        pilot.maximum_closed_q_l3_sigma = closed_value
        pilot.maximum_closed_q_l3_segment_index = int(segment_index)
    for cycle in pilot.report.get("cycles", []):
        interaction = cycle.get("interaction", {})
        for checkpoint in interaction.get("checkpoint_metrics", []):
            raw_value = float(
                checkpoint.get("q_l3_sigma", float("-inf"))
            )
            if raw_value > float(
                pilot.maximum_raw_relaxation_q_l3_sigma
            ):
                pilot.maximum_raw_relaxation_q_l3_sigma = raw_value
                pilot.maximum_raw_relaxation_segment_index = int(
                    segment_index
                )
                pilot.maximum_raw_relaxation_iteration = int(
                    checkpoint.get("iteration", 0)
                )


def _record_relaxation_entry(
    pilot: Pilot,
    stage_policy: dict[str, Any],
    segment_index: int,
    journal: Journal,
) -> None:
    pilot.relaxation_entry_achieved = True
    pilot.relaxation_entry_segment_index = int(segment_index)
    pilot.relaxation_gate_policy = dict(stage_policy)
    pilot.active = True
    journal.emit(
        "pilot_relaxation_entry_achieved",
        pilot=pilot.name,
        stage=stage_policy["name"],
        segment_index=int(segment_index),
        after=pilot.report.get("after", {}),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wall-time-s", type=float, default=28800.0)
    parser.add_argument("--final-audit-reserve-s", type=float, default=3600.0)
    parser.add_argument(
        "--minimum-relaxation-reserve-s",
        type=float,
        default=1800.0,
    )
    parser.add_argument("--pilot-closure-s", type=float, default=6000.0)
    parser.add_argument("--relaxation-segment-s", type=float, default=900.0)
    parser.add_argument(
        "--gate-policy",
        choices=(*FIXED_GATE_POLICIES, ADAPTIVE_GATE_POLICY),
        default=ADAPTIVE_GATE_POLICY,
        help=(
            "adaptive-topology-v1 progresses through strict, +1 L/h, soft "
            "topology, and topology-escrow closure before relaxation"
        ),
    )
    parser.add_argument("--valence-l-over-h-count-budget", type=int)
    parser.add_argument("--closure-q-l3-decrease-budget", type=float)
    parser.add_argument("--closure-q-p01-decrease-budget", type=float)
    parser.add_argument(
        "--closure-minimum-angle-p01-decrease-budget",
        type=float,
    )
    parser.add_argument("--closure-l-over-h-count-budget", type=int)
    parser.add_argument("--closure-l-over-h-p95-budget", type=float)
    parser.add_argument("--closure-l-over-h-maximum-budget", type=float)
    parser.add_argument("--closure-area-transition-count-budget", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.wall_time_s <= 0.0:
        parser.error("--wall-time-s must be positive")
    if not 0.0 <= args.final_audit_reserve_s < args.wall_time_s:
        parser.error("--final-audit-reserve-s must be within the wall time")
    non_audit_window = (
        float(args.wall_time_s) - float(args.final_audit_reserve_s)
    )
    if not 0.0 <= args.minimum_relaxation_reserve_s < non_audit_window:
        parser.error(
            "--minimum-relaxation-reserve-s must be within the non-audit "
            "work window"
        )
    gate_policy = _resolve_gate_policy(args, parser)
    gate_policy["minimum_relaxation_reserve_seconds"] = float(
        args.minimum_relaxation_reserve_s
    )

    workspace = Path(args.workspace_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    journal = Journal(run_dir / "progress.jsonl")
    started = time.perf_counter()
    hard_deadline = started + float(args.wall_time_s)
    work_deadline = hard_deadline - float(args.final_audit_reserve_s)
    exit_code = 1
    pilots: list[Pilot] = []
    try:
        journal.emit(
            "run_started",
            workspace_root=workspace,
            run_dir=run_dir,
            wall_time_s=float(args.wall_time_s),
            final_audit_reserve_s=float(args.final_audit_reserve_s),
            minimum_relaxation_reserve_s=float(
                args.minimum_relaxation_reserve_s
            ),
            work_deadline_monotonic_s=work_deadline,
            gate_policy=gate_policy,
        )
        pilots, preflight = _preflight(workspace, journal)
        _write_json(run_dir / "preflight.json", preflight)
        journal.emit("preflight_passed", pilots=[value.name for value in pilots])

        policy_stages = list(gate_policy["stages"])
        segment_index = 0
        closure_deadline_reached = False
        for stage_index, stage_policy in enumerate(policy_stages):
            for pilot_position, pilot in enumerate(pilots):
                if pilot.relaxation_entry_achieved:
                    continue
                remaining = work_deadline - time.perf_counter()
                if remaining <= 0.0:
                    closure_deadline_reached = True
                    break
                closure_remaining = max(
                    0.0,
                    remaining - float(args.minimum_relaxation_reserve_s),
                )
                if closure_remaining <= 0.0:
                    closure_deadline_reached = True
                    break
                current_pending = sum(
                    not value.relaxation_entry_achieved
                    for value in pilots[pilot_position:]
                )
                future_pending = (
                    sum(
                        not value.relaxation_entry_achieved
                        for value in pilots
                    )
                    * max(0, len(policy_stages) - stage_index - 1)
                )
                fair_share = closure_remaining / max(
                    1,
                    current_pending + future_pending,
                )
                budget = min(float(args.pilot_closure_s), fair_share)
                phase = (
                    "initial_zero_debt_closure"
                    if not bool(gate_policy["adaptive"])
                    else (
                        f"adaptive_closure_{stage_index + 1}_"
                        f"{stage_policy['name']}"
                    )
                )
                _run_segment(
                    pilot,
                    run_dir,
                    journal,
                    phase=phase,
                    segment_index=segment_index,
                    budget_s=budget,
                    maximum_cycles=0,
                    total_iterations=0,
                    burst=10,
                    work_deadline=work_deadline,
                    gate_policy=stage_policy,
                )
                exact_entry = _exact_zero_debt(pilot.report)
                stage_record = {
                    "stage_index": int(stage_index),
                    "segment_index": int(segment_index),
                    "preset": str(stage_policy["name"]),
                    "engine_policy": str(stage_policy["engine_name"]),
                    "escrow_enabled": bool(
                        stage_policy["escrow_enabled"]
                    ),
                    "budget_seconds": float(budget),
                    "status": str(pilot.report.get("status", "")),
                    "accepted_or_provisional_state_carried_forward": True,
                    "exact_zero_debt_entry": bool(exact_entry),
                    "cumulative_soft_baseline": (
                        pilot.closure_soft_baseline
                    ),
                    "before": pilot.report.get("before", {}),
                    "after": pilot.report.get("after", {}),
                }
                pilot.policy_history.append(stage_record)
                _track_quality_maxima(pilot, segment_index)
                if exact_entry:
                    _record_relaxation_entry(
                        pilot,
                        stage_policy,
                        segment_index,
                        journal,
                    )
                segment_index += 1
            if closure_deadline_reached:
                break

        if bool(gate_policy["adaptive"]) and not closure_deadline_reached:
            retry_candidates: list[
                tuple[Pilot, dict[str, Any]]
            ] = []
            for pilot in pilots:
                if pilot.relaxation_entry_achieved:
                    continue
                decision = _evidence_retry_decision(
                    pilot,
                    policy_stages[-1],
                )
                decision["attempted"] = False
                decision["completed"] = not bool(
                    decision.get("eligible", False)
                )
                pilot.evidence_retry_decision = decision
                journal.emit(
                    "pilot_evidence_retry_decision",
                    pilot=pilot.name,
                    decision=decision,
                )
                if bool(decision.get("eligible", False)):
                    retry_candidates.append((pilot, decision))

            for retry_position, (pilot, decision) in enumerate(
                retry_candidates
            ):
                remaining = work_deadline - time.perf_counter()
                closure_remaining = max(
                    0.0,
                    remaining - float(args.minimum_relaxation_reserve_s),
                )
                if closure_remaining <= 0.0:
                    closure_deadline_reached = True
                    decision["interrupted"] = True
                    decision["reason"] = (
                        "minimum_relaxation_reserve_reached"
                    )
                    break
                budget = min(
                    float(args.pilot_closure_s),
                    closure_remaining
                    / max(1, len(retry_candidates) - retry_position),
                )
                stage_policy = dict(decision["policy"])
                before_retry = pilot.report.get("after", {})
                _run_segment(
                    pilot,
                    run_dir,
                    journal,
                    phase=(
                        "adaptive_closure_5_"
                        "topology-escrow-evidence-v1"
                    ),
                    segment_index=segment_index,
                    budget_s=budget,
                    maximum_cycles=0,
                    total_iterations=0,
                    burst=10,
                    work_deadline=work_deadline,
                    gate_policy=stage_policy,
                )
                exact_entry = _exact_zero_debt(pilot.report)
                decision.update(
                    {
                        "attempted": True,
                        "completed": True,
                        "segment_index": int(segment_index),
                        "exact_zero_debt_entry": bool(exact_entry),
                    }
                )
                pilot.policy_history.append(
                    {
                        "stage_index": len(policy_stages),
                        "segment_index": int(segment_index),
                        "preset": str(stage_policy["name"]),
                        "engine_policy": str(
                            stage_policy["engine_name"]
                        ),
                        "escrow_enabled": bool(
                            stage_policy["escrow_enabled"]
                        ),
                        "budget_seconds": float(budget),
                        "status": str(
                            pilot.report.get("status", "")
                        ),
                        "accepted_or_provisional_state_carried_forward": (
                            True
                        ),
                        "exact_zero_debt_entry": bool(exact_entry),
                        "cumulative_soft_baseline": (
                            pilot.closure_soft_baseline
                        ),
                        "before": before_retry,
                        "after": pilot.report.get("after", {}),
                        "evidence_retry_decision": decision,
                    }
                )
                _track_quality_maxima(pilot, segment_index)
                if exact_entry:
                    _record_relaxation_entry(
                        pilot,
                        stage_policy,
                        segment_index,
                        journal,
                    )
                segment_index += 1

        if bool(gate_policy["adaptive"]) and not closure_deadline_reached:
            cavity_candidates: list[
                tuple[Pilot, dict[str, Any]]
            ] = []
            for pilot in pilots:
                if pilot.relaxation_entry_achieved:
                    continue
                retry_policy = pilot.evidence_retry_decision.get(
                    "policy",
                    policy_stages[-1],
                )
                decision = _extended_cavity_decision(
                    pilot,
                    retry_policy,
                )
                decision["attempted"] = False
                decision["completed"] = not bool(
                    decision.get("eligible", False)
                )
                pilot.extended_cavity_decision = decision
                journal.emit(
                    "pilot_extended_cavity_decision",
                    pilot=pilot.name,
                    decision=decision,
                )
                if bool(decision.get("eligible", False)):
                    cavity_candidates.append((pilot, decision))

            for cavity_position, (pilot, decision) in enumerate(
                cavity_candidates
            ):
                remaining = work_deadline - time.perf_counter()
                closure_remaining = max(
                    0.0,
                    remaining - float(args.minimum_relaxation_reserve_s),
                )
                if closure_remaining <= 0.0:
                    closure_deadline_reached = True
                    decision["interrupted"] = True
                    decision["reason"] = (
                        "minimum_relaxation_reserve_reached"
                    )
                    break
                budget = min(
                    float(args.pilot_closure_s),
                    closure_remaining
                    / max(1, len(cavity_candidates) - cavity_position),
                )
                stage_policy = dict(decision["policy"])
                before_cavity = pilot.report.get("after", {})
                _run_segment(
                    pilot,
                    run_dir,
                    journal,
                    phase=(
                        "adaptive_closure_6_"
                        "topology-extended-cavity-v1"
                    ),
                    segment_index=segment_index,
                    budget_s=budget,
                    maximum_cycles=0,
                    total_iterations=0,
                    burst=10,
                    work_deadline=work_deadline,
                    gate_policy=stage_policy,
                )
                exact_entry = _exact_zero_debt(pilot.report)
                decision.update(
                    {
                        "attempted": True,
                        "completed": True,
                        "segment_index": int(segment_index),
                        "exact_zero_debt_entry": bool(exact_entry),
                    }
                )
                pilot.policy_history.append(
                    {
                        "stage_index": len(policy_stages) + 1,
                        "segment_index": int(segment_index),
                        "preset": str(stage_policy["name"]),
                        "engine_policy": str(
                            stage_policy["engine_name"]
                        ),
                        "escrow_enabled": bool(
                            stage_policy["escrow_enabled"]
                        ),
                        "budget_seconds": float(budget),
                        "status": str(
                            pilot.report.get("status", "")
                        ),
                        "accepted_or_provisional_state_carried_forward": (
                            True
                        ),
                        "exact_zero_debt_entry": bool(exact_entry),
                        "cumulative_soft_baseline": (
                            pilot.closure_soft_baseline
                        ),
                        "before": before_cavity,
                        "after": pilot.report.get("after", {}),
                        "extended_cavity_decision": decision,
                    }
                )
                _track_quality_maxima(pilot, segment_index)
                if exact_entry:
                    _record_relaxation_entry(
                        pilot,
                        stage_policy,
                        segment_index,
                        journal,
                    )
                segment_index += 1

        expected_stage_names = [
            str(value["name"]) for value in policy_stages
        ]
        for pilot in pilots:
            attempted_stage_names = [
                str(value["preset"]) for value in pilot.policy_history
            ]
            static_ladder_complete = (
                attempted_stage_names[: len(expected_stage_names)]
                == expected_stage_names
            )
            dynamic_decisions_complete = bool(
                pilot.evidence_retry_decision.get("completed", False)
                and pilot.extended_cavity_decision.get("completed", False)
            )
            pilot.policy_ladder_exhausted = bool(
                not pilot.relaxation_entry_achieved
                and static_ladder_complete
                and dynamic_decisions_complete
            )
            if pilot.relaxation_entry_achieved:
                continue
            pilot.active = False
            journal.emit(
                (
                    "pilot_policy_ladder_exhausted"
                    if pilot.policy_ladder_exhausted
                    else "pilot_policy_ladder_interrupted"
                ),
                pilot=pilot.name,
                reason=(
                    "exact_zero_debt_not_reached"
                    if pilot.policy_ladder_exhausted
                    else "work_deadline_reached"
                ),
                attempted_stages=attempted_stage_names,
                expected_stages=expected_stage_names,
                evidence_retry_decision=pilot.evidence_retry_decision,
                extended_cavity_decision=(
                    pilot.extended_cavity_decision
                ),
                after=pilot.report.get("after", {}),
            )

        ladder = (10, 25, 50, 100)
        while time.perf_counter() < work_deadline:
            eligible = [
                pilot
                for pilot in pilots
                if pilot.active and pilot.relaxation_entry_achieved
            ]
            if not eligible:
                break
            progressed = False
            for pilot in eligible:
                remaining = work_deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                burst = ladder[pilot.ladder_index]
                prior_q = float(pilot.report["after"]["q_l3_sigma"])
                budget = min(float(args.relaxation_segment_s), remaining)
                _run_segment(
                    pilot,
                    run_dir,
                    journal,
                    phase="alternating_relaxation",
                    segment_index=segment_index,
                    budget_s=budget,
                    maximum_cycles=1,
                    total_iterations=burst,
                    burst=burst,
                    work_deadline=work_deadline,
                    gate_policy=(
                        pilot.relaxation_gate_policy or policy_stages[-1]
                    ),
                )
                progressed = True
                committed = int(
                    pilot.report.get("committed_cycle_count", 0)
                ) > 0
                gain = (
                    float(pilot.report["after"]["q_l3_sigma"]) - prior_q
                )
                exact_zero_debt = _exact_zero_debt(pilot.report)
                _track_quality_maxima(pilot, segment_index)
                pilot.relaxation_history.append(
                    {
                        "segment_index": int(segment_index),
                        "burst": int(burst),
                        "committed": bool(committed),
                        "q_l3_sigma_before": float(prior_q),
                        "q_l3_sigma_after": float(
                            pilot.report["after"]["q_l3_sigma"]
                        ),
                        "q_l3_sigma_gain": float(gain),
                        "maximum_closed_q_l3_sigma_so_far": (
                            float(pilot.maximum_closed_q_l3_sigma)
                            if np.isfinite(
                                pilot.maximum_closed_q_l3_sigma
                            )
                            else None
                        ),
                        "maximum_raw_relaxation_q_l3_sigma_so_far": (
                            float(
                                pilot.maximum_raw_relaxation_q_l3_sigma
                            )
                            if np.isfinite(
                                pilot.maximum_raw_relaxation_q_l3_sigma
                            )
                            else None
                        ),
                        "exact_zero_debt_checkpoint": bool(
                            exact_zero_debt
                        ),
                        "status": str(pilot.report.get("status", "")),
                    }
                )
                if committed and gain >= 1.0e-4:
                    pilot.consecutive_improving += 1
                else:
                    pilot.consecutive_improving = 0
                pilot.ladder_index = (
                    pilot.ladder_index + 1
                ) % len(ladder)
                if not exact_zero_debt:
                    pilot.active = False
                    journal.emit(
                        "pilot_removed_from_rotation",
                        pilot=pilot.name,
                        reason="post_relaxation_zero_debt_failure",
                        after=pilot.report.get("after", {}),
                    )
                elif bool(pilot.report.get("v6_quality_target_pass", False)):
                    pilot.active = False
                    journal.emit(
                        "pilot_quality_target_reached",
                        pilot=pilot.name,
                        consecutive_improving_zero_debt_checkpoints=(
                            pilot.consecutive_improving
                        ),
                        after=pilot.report.get("after", {}),
                    )
                segment_index += 1
            if not progressed:
                break

        journal.emit(
            "final_audit_started",
            work_deadline_reached=time.perf_counter() >= work_deadline,
            remaining_hard_seconds=max(0.0, hard_deadline - time.perf_counter()),
        )
        pilot_verdicts = [
            _finalize_pilot(pilot, run_dir, workspace, journal)
            for pilot in pilots
        ]
        summary = _run_summary(
            pilot_verdicts,
            preflight,
            started,
            hard_deadline,
            gate_policy,
        )
        _write_json(run_dir / "acceptance_summary.json", summary)
        _write_markdown(run_dir / "acceptance_summary.md", summary)
        exit_code = 0
        journal.emit(
            "run_completed",
            runtime_seconds=float(time.perf_counter() - started),
            verdicts=summary["verdicts"],
        )
    except Exception as exc:
        journal.emit(
            "run_failed",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        _write_json(
            run_dir / "fatal_error.json",
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_seconds": float(time.perf_counter() - started),
            },
        )
    return exit_code


def _lineage_chain_sequences(state: Any) -> list[list[int]]:
    return [
        [int(state.lineage[int(node)]) for node in chain]
        for chain in state.chains
    ]


def _probe_r5_thin16(pilot: Pilot) -> dict[str, Any]:
    """Verify the real r5 OBC-front support route before the clocked run."""
    if pilot.name != "r5":
        raise ValueError("thin-16 probe requires the frozen r5 pilot")
    state, config, _ = create_visual_state(
        pilot.points,
        pilot.triangles,
        pilot.fixed,
        pilot.chains,
        pilot.open_nodes,
        target_spacing_m=pilot.targets,
        boundary_kinds=pilot.kinds,
        hard_anchor_mask=pilot.hard,
        node_lineage=pilot.lineage,
        restricted_lineage_edges=pilot.restrictions,
    )
    config = replace(
        config,
        systematic_v5_enable_boundary_window_fallback=False,
        systematic_v5_max_inward_front_support_points=8,
        systematic_v5_max_lawson_flips_per_transaction=0,
        micro_relax_cycles=0,
    )
    triangle_indices = tuple(
        map(int, THIN16_COMPONENT["triangle_indices_zero_based"])
    )
    if any(
        index < 0 or index >= len(state.triangles)
        for index in triangle_indices
    ):
        raise ValueError("thin-16 triangle index is outside the frozen r5 mesh")
    observed_lineage = sorted(
        {
            int(state.lineage[int(node)])
            for index in triangle_indices
            for node in state.triangles[int(index)]
        }
    )
    expected_lineage = sorted(map(int, THIN16_COMPONENT["node_lineage"]))
    if observed_lineage != expected_lineage:
        raise ValueError(
            "thin-16 lineage mismatch: "
            f"expected {expected_lineage}, observed {observed_lineage}"
        )

    expected_counts = list(
        map(int, THIN16_COMPONENT["expected_support_counts"])
    )
    selected: tuple[int, int, list[int], list[dict[str, Any]]] | None = None
    screened: list[dict[str, Any]] = []
    for triangle_index in triangle_indices:
        triangle = list(map(int, state.triangles[int(triangle_index)]))
        coordinates = state.points[np.asarray(triangle, dtype=int)]
        opposite_lengths = np.asarray(
            [
                np.linalg.norm(coordinates[1] - coordinates[2]),
                np.linalg.norm(coordinates[0] - coordinates[2]),
                np.linalg.norm(coordinates[0] - coordinates[1]),
            ],
            dtype=float,
        )
        center = int(triangle[int(np.argmax(opposite_lengths))])
        incident = np.where(
            np.any(state.triangles == int(center), axis=1)
        )[0]
        ring = topology._ordered_one_ring(
            state.triangles[incident],
            center,
        )
        modes = (
            []
            if ring is None
            else [
                value
                for value in topology._locked_star_modes(
                    state,
                    center,
                    ring,
                    triangle_index,
                    config,
                )
                if value["name"] == "inward-front-multi-support"
                and not value.get("generation_failures")
            ]
        )
        counts = [
            int(value.get("requested_support_node_count", -1))
            for value in modes
        ]
        screened.append(
            {
                "triangle_index_zero_based": int(triangle_index),
                "triangle_lineage": [
                    int(state.lineage[int(node)]) for node in triangle
                ],
                "causal_center_lineage": int(state.lineage[center]),
                "ring_lineage": (
                    []
                    if ring is None
                    else [int(state.lineage[int(node)]) for node in ring]
                ),
                "support_counts": counts,
            }
        )
        if counts == expected_counts and ring is not None:
            selected = (triangle_index, center, ring, modes)
            break
    if selected is None:
        raise RuntimeError(
            "thin-16 did not enumerate the required support counts 2..8"
        )

    triangle_index, center, ring, modes = selected
    repeat_modes = [
        value
        for value in topology._locked_star_modes(
            state,
            center,
            ring,
            triangle_index,
            config,
        )
        if value["name"] == "inward-front-multi-support"
        and not value.get("generation_failures")
    ]
    repeat_by_count = {
        int(value["requested_support_node_count"]): value
        for value in repeat_modes
    }
    polygon = Polygon(state.points[np.asarray(ring, dtype=int)])
    source_open_lineage = [
        int(state.lineage[int(node)]) for node in state.open_nodes
    ]
    source_chain_lineage = _lineage_chain_sequences(state)
    source_boundary_coordinates = {
        int(state.lineage[int(node)]): state.points[int(node)].copy()
        for chain in state.chains
        for node in chain
    }
    candidates: list[dict[str, Any]] = []
    for mode in modes:
        support_count = int(mode["requested_support_node_count"])
        support = np.asarray(mode["coordinates"], dtype=float)
        repeat = np.asarray(
            repeat_by_count[support_count]["coordinates"],
            dtype=float,
        )
        strictly_inside = bool(
            np.all(
                contains_xy(
                    polygon,
                    support[:, 0],
                    support[:, 1],
                )
            )
        )
        generated_deterministically = bool(np.array_equal(support, repeat))
        first = state.clone()
        first_changed, first_failures, first_evidence = (
            topology._reconstruct_locked_star_candidate(
                first,
                center=center,
                triangle_index=triangle_index,
                mode=mode,
                config=config,
            )
        )
        second = state.clone()
        second_changed, second_failures, _ = (
            topology._reconstruct_locked_star_candidate(
                second,
                center=center,
                triangle_index=triangle_index,
                mode=mode,
                config=config,
            )
        )
        reconstructed_deterministically = bool(
            first_changed
            and second_changed
            and not first_failures
            and not second_failures
            and np.array_equal(first.points, second.points)
            and np.array_equal(first.triangles, second.triangles)
        )
        delivered_open_lineage = (
            [int(first.lineage[int(node)]) for node in first.open_nodes]
            if first_changed
            else []
        )
        delivered_chain_lineage = (
            _lineage_chain_sequences(first) if first_changed else []
        )
        lineage_to_node = (
            {
                int(value): int(index)
                for index, value in enumerate(first.lineage)
                if int(value) >= 0
            }
            if first_changed
            else {}
        )
        boundary_coordinates_exact = bool(
            first_changed
            and all(
                lineage in lineage_to_node
                and np.array_equal(
                    first.points[lineage_to_node[lineage]],
                    coordinate,
                )
                for lineage, coordinate in source_boundary_coordinates.items()
            )
        )
        protected_edges_present = bool(
            first_changed
            and chain_edges(first.chains).issubset(
                topology._edge_set(first.triangles)
            )
        )
        edge_topology = (
            build_edge_topology(len(first.points), first.triangles)
            if first_changed
            else None
        )
        manifold = bool(
            edge_topology is not None
            and all(
                len(attached) <= 2
                for attached in edge_topology.edge_to_triangles.values()
            )
        )
        positive_area = bool(
            first_changed
            and np.all(
                topology.triangle_geometry(
                    first.points,
                    first.triangles,
                )["signed_area"]
                > topology._area_tolerance(first.points, first.triangles)
            )
        )
        global_delaunay_used = bool(
            mode.get("support_generation_evidence", {}).get(
                "global_delaunay_used",
                True,
            )
            or first_evidence.get("global_delaunay_used", True)
        )
        passed = bool(
            strictly_inside
            and generated_deterministically
            and reconstructed_deterministically
            and delivered_open_lineage == source_open_lineage
            and delivered_chain_lineage == source_chain_lineage
            and boundary_coordinates_exact
            and protected_edges_present
            and manifold
            and positive_area
            and not global_delaunay_used
        )
        candidates.append(
            {
                "support_count": support_count,
                "strictly_inside": strictly_inside,
                "generated_deterministically": generated_deterministically,
                "reconstructed_deterministically": (
                    reconstructed_deterministically
                ),
                "open_boundary_lineage_unchanged": bool(
                    delivered_open_lineage == source_open_lineage
                ),
                "boundary_chain_lineage_unchanged": bool(
                    delivered_chain_lineage == source_chain_lineage
                ),
                "boundary_coordinates_exact": boundary_coordinates_exact,
                "protected_edges_present": protected_edges_present,
                "manifold": manifold,
                "positive_area": positive_area,
                "global_delaunay_used": global_delaunay_used,
                "passed": passed,
            }
        )
    report = {
        "schema_version": "fvcom_systematic_v6_thin16_probe_v1",
        "component_id": str(THIN16_COMPONENT["component_id"]),
        "input_mesh": str(pilot.mesh_path),
        "input_sha256": _sha256(pilot.mesh_path),
        "expected_node_lineage": expected_lineage,
        "observed_node_lineage": observed_lineage,
        "screened_causal_stars": screened,
        "selected_triangle_index_zero_based": int(triangle_index),
        "selected_center_lineage": int(state.lineage[center]),
        "expected_support_counts": expected_counts,
        "candidates": candidates,
        "passed": bool(
            [value["support_count"] for value in candidates]
            == expected_counts
            and all(value["passed"] for value in candidates)
        ),
    }
    if not report["passed"]:
        raise RuntimeError("thin-16 real-mesh support probe failed")
    return report


def _preflight(workspace: Path, journal: Journal) -> tuple[list[Pilot], dict[str, Any]]:
    size_path = workspace / SIZE_FIELD
    if not size_path.is_file():
        raise FileNotFoundError(size_path)
    validation = _run_validation_suite(journal)
    pilots: list[Pilot] = []
    checks: list[dict[str, Any]] = []
    for name, spec in EXPECTED_INPUTS.items():
        mesh_path = workspace / str(spec["mesh"])
        boundary_path = workspace / str(spec["boundary"])
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        if not boundary_path.is_file():
            raise FileNotFoundError(boundary_path)
        digest = _sha256(mesh_path)
        passed = digest == str(spec["sha256"])
        checks.append(
            {
                "pilot": name,
                "mesh": str(mesh_path),
                "expected_sha256": str(spec["sha256"]),
                "actual_sha256": digest,
                "passed": passed,
            }
        )
        if not passed:
            raise ValueError(f"frozen input hash mismatch for {name}")
        mesh = read_2dm(mesh_path)
        projection = local_utm_projection(_bbox(mesh.nodes_lonlat))
        points = project_points(mesh.nodes_lonlat, projection)
        triangles = np.asarray(mesh.triangles, dtype=int) - 1
        open_nodes = np.asarray(mesh.open_boundary_nodes, dtype=int) - 1
        chains, kinds, hard, explicit_targets = _boundary_metadata(
            len(points),
            triangles,
            open_nodes,
            str(boundary_path),
            None,
        )
        fixed = np.zeros(len(points), dtype=bool)
        for chain in chains:
            fixed[np.asarray(chain, dtype=int)] = True
        targets = _target_sizes(
            mesh.nodes_lonlat,
            points,
            triangles,
            str(size_path),
        )
        explicit = np.isfinite(explicit_targets) & (explicit_targets > 0.0)
        targets[explicit] = explicit_targets[explicit]
        lineage = np.arange(len(points), dtype=int)
        source_contract = {
            "source_points": points.copy(),
            "source_chains": [list(map(int, chain)) for chain in chains],
            "source_open_nodes": open_nodes.copy(),
            "source_kinds": list(kinds),
            "source_hard_anchor_lineage": np.where(hard)[0].astype(int),
        }
        pilots.append(
            Pilot(
                name=name,
                mesh_path=mesh_path,
                boundary_path=boundary_path,
                mesh=mesh,
                projection=projection,
                points=points,
                triangles=triangles,
                fixed=fixed,
                chains=chains,
                open_nodes=open_nodes,
                targets=targets,
                kinds=kinds,
                hard=hard,
                lineage=lineage,
                restrictions=set(),
                source_contract=source_contract,
            )
        )
        journal.emit(
            "pilot_preflight",
            pilot=name,
            node_count=len(points),
            triangle_count=len(triangles),
            obc_node_count=len(open_nodes),
            hard_anchor_count=int(np.count_nonzero(hard)),
            sha256=digest,
        )
    r5_pilot = next(value for value in pilots if value.name == "r5")
    thin16_probe = _probe_r5_thin16(r5_pilot)
    journal.emit(
        "thin16_probe",
        passed=thin16_probe["passed"],
        selected_triangle_index_zero_based=thin16_probe[
            "selected_triangle_index_zero_based"
        ],
        selected_center_lineage=thin16_probe["selected_center_lineage"],
        support_counts=[
            value["support_count"] for value in thin16_probe["candidates"]
        ],
    )
    return pilots, {
        "schema_version": "fvcom_systematic_v6_overnight_preflight_v1",
        "size_field": str(size_path),
        "size_field_sha256": _sha256(size_path),
        "input_checks": checks,
        "unit_validation": validation,
        "thin16_probe": thin16_probe,
        "python": sys.version,
        "numpy": np.__version__,
        "implementation_sha256": {
            name: _sha256(SCRIPT_DIR / name)
            for name in (
                "fvcom_grid_generation/systematic_v6.py",
                "fvcom_grid_generation/systematic_v6_policy.py",
                "fvcom_grid_generation/local_topology.py",
                "fvcom_grid_generation/thin_passage.py",
                "research/delaware/run_systematic_v6_overnight.py",
                "diagnose_superthin_components.py",
                "selftest_local_topology_v5_extensions.py",
                "selftest_systematic_v6.py",
            )
        },
    }


def _run_validation_suite(journal: Journal) -> list[dict[str, Any]]:
    tests = (
        "selftest_connectivity_restriction.py",
        "selftest_systematic_v5.py",
        "selftest_local_topology_v2.py",
        "selftest_local_topology_v5_extensions.py",
        "selftest_visual_superthin.py",
        "selftest_boundary_contract_v2.py",
        "selftest_size_field_v2.py",
        "selftest_fvcom_grid.py",
        "selftest_systematic_v6.py",
    )
    records: list[dict[str, Any]] = []
    for name in tests:
        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / name)],
            cwd=str(SCRIPT_DIR.parent),
            capture_output=True,
            text=True,
            check=False,
        )
        record = {
            "test": name,
            "returncode": int(process.returncode),
            "runtime_seconds": float(time.perf_counter() - started),
            "stdout": process.stdout,
            "stderr": process.stderr,
            "passed": bool(process.returncode == 0),
        }
        records.append(record)
        journal.emit(
            "unit_validation",
            test=name,
            passed=record["passed"],
            runtime_seconds=record["runtime_seconds"],
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"unit validation failed: {name}: {process.stderr}"
            )
    return records


def _supported_dataclass_kwargs(
    dataclass_type: Any,
    values: dict[str, Any],
) -> dict[str, Any]:
    fields = getattr(dataclass_type, "__dataclass_fields__", {})
    return {key: value for key, value in values.items() if key in fields}


def _run_segment(
    pilot: Pilot,
    run_dir: Path,
    journal: Journal,
    *,
    phase: str,
    segment_index: int,
    budget_s: float,
    maximum_cycles: int,
    total_iterations: int,
    burst: int,
    work_deadline: float,
    gate_policy: dict[str, Any],
) -> None:
    segment_started = time.perf_counter()
    deadline = min(work_deadline, segment_started + max(0.001, float(budget_s)))
    journal.emit(
        "segment_started",
        pilot=pilot.name,
        phase=phase,
        segment_index=int(segment_index),
        budget_s=float(budget_s),
        burst=int(burst),
        gate_policy=gate_policy,
    )
    topology_values = {
        "thin_repair_profile": "systematic-v5",
        "systematic_v5_enable_connectivity_restriction": True,
        "systematic_v5_max_connectivity_transactions_per_round": 32,
        "systematic_v5_enable_boundary_window_fallback": False,
        "systematic_v5_patch_ring_ladder": tuple(
            map(
                int,
                gate_policy.get(
                    "patch_ring_ladder",
                    (1, 2, 4),
                ),
            )
        ),
        "enable_pruning": False,
        "enable_thin_repair": True,
        "enable_valence_repair": True,
        "max_rounds": 4,
        "max_prunes_per_round": 0,
        "max_valence": 8,
        "max_valence_removals_per_round": 500,
        "max_valence_flip_batch": 64,
        "max_valence_cluster_merges_per_round": 25,
        "max_valence_l_over_h_count_increase": int(
            gate_policy["valence_l_over_h_count_increase"]
        ),
        "topology_escrow_enabled": bool(
            gate_policy["escrow_enabled"]
        ),
        "topology_escrow_maximum_superthin_count": int(
            gate_policy.get("escrow_maximum_superthin_count", 25)
        ),
        "topology_escrow_maximum_superthin_severity": float(
            gate_policy.get("escrow_maximum_superthin_severity", 25.0)
        ),
        "topology_escrow_maximum_valence": int(
            gate_policy.get("escrow_maximum_valence", 12)
        ),
        "micro_relax_cycles": 3,
        "deadline_monotonic_s": deadline,
    }
    loop_values = {
        "maximum_closure_rounds": 8 if "closure" in phase else 4,
        "valence_rounds_per_closure": 4,
        "maximum_relaxation_cycles": int(maximum_cycles),
        "total_relaxation_iterations": int(total_iterations),
        "burst_ladder": (int(burst),),
        "maximum_burst": int(burst),
        "checkpoint_interval": min(10, max(1, int(burst))),
        "wall_clock_seconds": max(0.001, float(budget_s)),
        "final_audit_reserve_seconds": 0.0,
        "closure_gate_policy": str(gate_policy["engine_name"]),
        "closure_max_q_l3_sigma_decrease": float(
            gate_policy["closure_q_l3_sigma_decrease"]
        ),
        "closure_max_q_p01_decrease": float(
            gate_policy["closure_q_p01_decrease"]
        ),
        "closure_max_minimum_angle_p01_decrease": float(
            gate_policy["closure_minimum_angle_p01_decrease"]
        ),
        "closure_max_l_over_h_count_increase": int(
            gate_policy["closure_l_over_h_count_increase"]
        ),
        "closure_max_l_over_h_p95_increase": float(
            gate_policy["closure_l_over_h_p95_increase"]
        ),
        "closure_max_l_over_h_maximum_increase": float(
            gate_policy["closure_l_over_h_maximum_increase"]
        ),
        "closure_max_area_transition_count_increase": int(
            gate_policy["closure_area_transition_count_increase"]
        ),
        "escrow_maximum_superthin_count": int(
            gate_policy.get("escrow_maximum_superthin_count", 25)
        ),
        "escrow_maximum_superthin_severity": float(
            gate_policy.get("escrow_maximum_superthin_severity", 25.0)
        ),
        "escrow_maximum_valence": int(
            gate_policy.get("escrow_maximum_valence", 12)
        ),
        "escrow_maximum_valence_count_rebound": int(
            gate_policy.get("escrow_maximum_valence_count_rebound", 8)
        ),
        "escrow_maximum_valence_excess_rebound": int(
            gate_policy.get("escrow_maximum_valence_excess_rebound", 16)
        ),
        "passage_removal_enabled": True,
        "allow_authorized_topology_delta": True,
        "known_passage_node_ids_1based": (
            DELAWARE_PASSAGE_NODE_IDS_1BASED
        ),
        "deadline_monotonic_s": deadline,
    }
    result = run_systematic_v6_loop(
        pilot.points,
        pilot.triangles,
        pilot.fixed,
        pilot.chains,
        pilot.open_nodes,
        target_spacing_m=pilot.targets,
        boundary_kinds=pilot.kinds,
        hard_anchor_mask=pilot.hard,
        node_lineage=pilot.lineage,
        source_contract=pilot.source_contract,
        closure_soft_baseline=pilot.closure_soft_baseline,
        restricted_lineage_edges=pilot.restrictions,
        topology_config=AggressiveConditioningConfig(
            **_supported_dataclass_kwargs(
                AggressiveConditioningConfig,
                topology_values,
            )
        ),
        loop_config=SystematicV6LoopConfig(
            **_supported_dataclass_kwargs(
                SystematicV6LoopConfig,
                loop_values,
            )
        ),
    )
    pilot.points = result.nodes_xy
    pilot.triangles = result.triangles
    pilot.fixed = result.fixed_node_mask
    pilot.chains = result.constraint_chains
    pilot.open_nodes = result.open_boundary_nodes_zero_based
    pilot.targets = result.target_spacing_m
    pilot.kinds = result.boundary_kinds
    pilot.hard = result.hard_anchor_mask
    pilot.lineage = result.node_lineage
    pilot.restrictions = set(result.restricted_lineage_edges)
    pilot.report = result.report
    if pilot.closure_soft_baseline is None:
        candidate_baseline = (
            result.report.get("initial_closure", {}).get(
                "soft_debt_baseline"
            )
        )
        if isinstance(candidate_baseline, dict):
            pilot.closure_soft_baseline = dict(candidate_baseline)
    pilot.passage_replays.extend(
        _collect_passage_replays(result.report, phase, segment_index)
    )
    segment_path = (
        run_dir
        / pilot.name
        / "segments"
        / f"{segment_index:04d}_{phase}.json"
    )
    _write_json(segment_path, result.report)
    pilot.segment_paths.append(str(segment_path))
    segment_summary = {
        "segment_index": int(segment_index),
        "phase": phase,
        "runtime_seconds": float(time.perf_counter() - segment_started),
        "status": str(result.report.get("status", "")),
        "gate_policy": gate_policy,
        "committed_cycle_count": int(
            result.report.get("committed_cycle_count", 0)
        ),
        "v6_zero_debt_pass": bool(
            result.report.get("v6_zero_debt_pass", False)
        ),
        "exact_zero_debt_pass": bool(_exact_zero_debt(result.report)),
        "v6_quality_target_pass": bool(
            result.report.get("v6_quality_target_pass", False)
        ),
        "after": result.report.get("after", {}),
        "report": str(segment_path),
    }
    pilot.segment_summaries.append(segment_summary)
    journal.emit("segment_completed", pilot=pilot.name, **segment_summary)


def _finalize_pilot(
    pilot: Pilot,
    run_dir: Path,
    workspace: Path,
    journal: Journal,
) -> dict[str, Any]:
    pilot_dir = run_dir / pilot.name
    pilot_dir.mkdir(parents=True, exist_ok=True)
    zero_debt = bool(_exact_zero_debt(pilot.report))
    artifact_label = "champion" if zero_debt else "provisional"
    output_mesh = (
        pilot_dir
        / f"{pilot.name}_systematic_v6_{artifact_label}.2dm"
    )
    lonlat = unproject_points(pilot.points, pilot.projection)
    depths = _remap_depths(
        pilot.mesh.depths,
        project_points(pilot.mesh.nodes_lonlat, pilot.projection),
        pilot.points,
        pilot.lineage,
    )
    write_2dm(
        output_mesh,
        lonlat,
        depths,
        pilot.triangles + 1,
        pilot.open_nodes + 1,
        mesh_name=(
            f"delaware_{pilot.name}_systematic_v6_{artifact_label}"
        ),
    )
    boundary_output = pilot_dir / "boundary_nodes.geojson"
    _write_json(
        boundary_output,
        _boundary_geojson(
            lonlat,
            pilot.chains,
            pilot.open_nodes,
            pilot.kinds,
            pilot.hard,
            pilot.lineage,
            pilot.targets,
        ),
    )
    lineage_output = pilot_dir / "node_lineage.json"
    _write_json(lineage_output, {"node_lineage": pilot.lineage})
    restriction_output = pilot_dir / "restricted_lineage_edges.json"
    _write_json(
        restriction_output,
        {
            "restricted_lineage_edges": [
                list(map(int, edge)) for edge in sorted(pilot.restrictions)
            ]
        },
    )
    obc_output = pilot_dir / "obc_remap_manifest.json"
    _write_json(
        obc_output,
        {
            "source_open_boundary_lineage": list(
                map(int, pilot.source_contract["source_open_nodes"])
            ),
            "delivered_open_boundary_lineage": [
                int(pilot.lineage[int(node)]) for node in pilot.open_nodes
            ],
        },
    )
    result_view = _ResultView(pilot)
    roundtrip = _serialized_roundtrip_audit(
        output_mesh,
        result_view,
        pilot.projection,
    )
    quality_output = pilot_dir / "independent_mesh_quality.json"
    quality_process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "analyze_mesh_quality.py"),
            "--mesh",
            str(output_mesh),
            "--output",
            str(quality_output),
        ],
        cwd=str(SCRIPT_DIR.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    atlas_output = None
    if not zero_debt:
        atlas_output = pilot_dir / "failure_atlas"
        atlas_process = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "diagnose_superthin_components.py"),
                "--mesh",
                str(output_mesh),
                "--boundary-nodes-geojson",
                str(boundary_output),
                "--size-field-nc",
                str(workspace / SIZE_FIELD),
                "--output-dir",
                str(atlas_output),
                "--conditioning-report",
                str(pilot.segment_paths[-1]),
                "--node-lineage-json",
                str(lineage_output),
                "--restricted-lineage-json",
                str(restriction_output),
                "--ring-ladder",
                "1",
                "2",
                "4",
                "8",
                "12",
            ],
            cwd=str(SCRIPT_DIR.parent),
            capture_output=True,
            text=True,
            check=False,
        )
        _write_json(
            pilot_dir / "failure_atlas_process.json",
            {
                "returncode": int(atlas_process.returncode),
                "stdout": atlas_process.stdout,
                "stderr": atlas_process.stderr,
            },
        )
        _write_json(
            pilot_dir / "human_handoff_hypotheses.json",
            _human_handoff_hypotheses(pilot, lonlat),
        )
    report_output = pilot_dir / "pilot_report.json"
    report = {
        "schema_version": "fvcom_systematic_v6_overnight_pilot_v1",
        "pilot": pilot.name,
        "input_mesh": str(pilot.mesh_path),
        "input_sha256": _sha256(pilot.mesh_path),
        "output_mesh": str(output_mesh),
        "output_sha256": _sha256(output_mesh),
        "artifact_status": {
            "label": artifact_label,
            "exact_zero_debt": bool(zero_debt),
            "promotable_closed_checkpoint": bool(zero_debt),
            "provisional_nonzero_debt": bool(not zero_debt),
        },
        "segments": pilot.segment_summaries,
        "segment_reports": pilot.segment_paths,
        "policy_stage_history": pilot.policy_history,
        "relaxation_stage_history": pilot.relaxation_history,
        "evidence_retry_decision": pilot.evidence_retry_decision,
        "extended_cavity_decision": pilot.extended_cavity_decision,
        "cumulative_closure_soft_baseline": (
            pilot.closure_soft_baseline
        ),
        "relaxation_entry": {
            "achieved": bool(pilot.relaxation_entry_achieved),
            "segment_index": pilot.relaxation_entry_segment_index,
            "gate_policy": pilot.relaxation_gate_policy,
            "exact_zero_debt_required": True,
        },
        "policy_ladder_exhausted": bool(pilot.policy_ladder_exhausted),
        "maximum_closed_q_l3_sigma": (
            float(pilot.maximum_closed_q_l3_sigma)
            if np.isfinite(pilot.maximum_closed_q_l3_sigma)
            else None
        ),
        "maximum_closed_q_l3_segment_index": (
            pilot.maximum_closed_q_l3_segment_index
        ),
        "maximum_raw_relaxation_q_l3_sigma": (
            float(pilot.maximum_raw_relaxation_q_l3_sigma)
            if np.isfinite(pilot.maximum_raw_relaxation_q_l3_sigma)
            else None
        ),
        "maximum_raw_relaxation_segment_index": (
            pilot.maximum_raw_relaxation_segment_index
        ),
        "maximum_raw_relaxation_iteration": (
            pilot.maximum_raw_relaxation_iteration
        ),
        "terminal_conditioning": pilot.report,
        "deterministic_passage_replays": pilot.passage_replays,
        "deterministic_passage_replay_pass": bool(
            all(value["passed"] for value in pilot.passage_replays)
        ),
        "serialized_roundtrip": roundtrip,
        "independent_quality_process": {
            "returncode": int(quality_process.returncode),
            "stdout": quality_process.stdout,
            "stderr": quality_process.stderr,
            "output": str(quality_output),
        },
        "failure_atlas": str(atlas_output) if atlas_output else None,
        "verdicts": {
            "v6_zero_debt_pass": zero_debt,
            "v6_quality_target_pass": bool(
                pilot.report.get("v6_quality_target_pass", False)
            ),
            "authorized_topology_smoke_ready": bool(
                pilot.report.get("authorized_topology_smoke_ready", False)
                and roundtrip["passed"]
            ),
            "standard_catalog_ready": bool(
                pilot.report.get("standard_catalog_ready", False)
                and roundtrip["passed"]
            ),
        },
    }
    _write_json(report_output, report)
    journal.emit(
        "pilot_finalized",
        pilot=pilot.name,
        report=report_output,
        verdicts=report["verdicts"],
        serialized_roundtrip=roundtrip,
    )
    return report


def _collect_passage_replays(
    document: Any,
    phase: str,
    segment_index: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if (
                value.get("schema_version") == "fvcom_thin_passage_removal_v1"
                and bool(value.get("accepted", False))
            ):
                selected = value.get("selected_candidate_index")
                attempts = value.get("attempts", [])
                record = (
                    attempts[int(selected)]
                    if selected is not None and int(selected) < len(attempts)
                    else {}
                )
                replay = record.get("deterministic_replay", {})
                output.append(
                    {
                        "phase": phase,
                        "segment_index": int(segment_index),
                        "path": path,
                        "component_id": str(value.get("component_id", "")),
                        "passed": bool(replay.get("passed", False)),
                        "requested_node_lineages": record.get(
                            "requested_node_lineages",
                            [],
                        ),
                        "replay": replay,
                    }
                )
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(document, "")
    return output


class _ResultView:
    def __init__(self, pilot: Pilot):
        self.nodes_xy = pilot.points
        self.triangles = pilot.triangles
        self.open_boundary_nodes_zero_based = pilot.open_nodes


def _human_handoff_hypotheses(
    pilot: Pilot,
    lonlat: np.ndarray,
) -> dict[str, Any]:
    topology = build_edge_topology(len(pilot.points), pilot.triangles)
    valence = np.asarray(
        [len(values) for values in topology.node_neighbors],
        dtype=int,
    )
    bad_valence = np.where(valence > 8)[0]
    residual = pilot.report.get("residual_components", [])
    lineages = sorted(
        {
            int(value)
            for component in residual
            for value in component.get("node_lineage", [])
        }
        | {int(pilot.lineage[int(node)]) for node in bad_valence}
    )
    current_nodes = [
        int(index)
        for index, value in enumerate(pilot.lineage)
        if int(value) in set(lineages)
    ]
    geometry = (
        {
            "longitude_min": float(np.min(lonlat[current_nodes, 0])),
            "longitude_max": float(np.max(lonlat[current_nodes, 0])),
            "latitude_min": float(np.min(lonlat[current_nodes, 1])),
            "latitude_max": float(np.max(lonlat[current_nodes, 1])),
        }
        if current_nodes
        else None
    )
    common = {
        "execute": False,
        "source_node_lineages_zero_based": lineages,
        "source_node_ids_1based": [
            int(value) + 1 for value in lineages if int(value) >= 0
        ],
        "affected_geographic_bbox": geometry,
        "mission_conflicts": [
            "hard anchors remain immutable",
            "ordered 87-node OBC remains immutable unless Bear authorizes regeneration",
            "wet-component changes require explicit scientific interpretation",
        ],
        "validation_gates": [
            "zero superthin triangles",
            "maximum true-neighbor valence <= 8",
            "zero restricted-edge violations",
            "positive manifold geometry and degree-two boundary loops",
            "exact hard-anchor and ordered OBC contracts",
            "12-decimal serialization round trip",
        ],
    }
    return {
        "schema_version": "fvcom_systematic_v6_human_handoff_v1",
        "pilot": pilot.name,
        "stop_after_proposals": True,
        "hypotheses": [
            {
                **common,
                "name": "expanded_coupled_cavity",
                "method": (
                    "8/12-ring target- and valence-constrained min-max "
                    "reconstruction split along protected chords"
                ),
                "expected_topology_change": (
                    "local connectivity and optional distributed-Steiner "
                    "nodes; no intended wet-component change"
                ),
            },
            {
                **common,
                "name": "upstream_boundary_topography_adjustment",
                "method": (
                    "regenerate the source boundary: widen/regularize a "
                    "physical channel to four elements across, close a "
                    "dispensable wet corridor, or reparameterize the OBC fan"
                ),
                "expected_topology_change": (
                    "boundary, wet-component, and local area distribution "
                    "may change and must be reviewed scientifically"
                ),
            },
        ],
    }


def _run_summary(
    pilot_reports: list[dict[str, Any]],
    preflight: dict[str, Any],
    started: float,
    hard_deadline: float,
    gate_policy: dict[str, Any],
) -> dict[str, Any]:
    verdict_names = (
        "v6_zero_debt_pass",
        "v6_quality_target_pass",
        "authorized_topology_smoke_ready",
        "standard_catalog_ready",
    )
    verdicts = {
        name: bool(
            pilot_reports
            and all(report["verdicts"][name] for report in pilot_reports)
        )
        for name in verdict_names
    }
    return {
        "schema_version": "fvcom_systematic_v6_overnight_summary_v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": float(time.perf_counter() - started),
        "hard_deadline_reached": bool(time.perf_counter() >= hard_deadline),
        "gate_policy": gate_policy,
        "preflight": preflight,
        "pilots": [
            {
                "pilot": report["pilot"],
                "report": report["output_mesh"],
                "artifact_status": report["artifact_status"],
                "verdicts": report["verdicts"],
                "after": report["terminal_conditioning"].get("after", {}),
                "failure_atlas": report["failure_atlas"],
                "policy_stage_history": report["policy_stage_history"],
                "evidence_retry_decision": report[
                    "evidence_retry_decision"
                ],
                "extended_cavity_decision": report[
                    "extended_cavity_decision"
                ],
                "relaxation_stage_history": (
                    report["relaxation_stage_history"]
                ),
                "relaxation_entry": report["relaxation_entry"],
                "policy_ladder_exhausted": report[
                    "policy_ladder_exhausted"
                ],
                "maximum_closed_q_l3_sigma": report[
                    "maximum_closed_q_l3_sigma"
                ],
                "maximum_closed_q_l3_segment_index": report[
                    "maximum_closed_q_l3_segment_index"
                ],
                "maximum_raw_relaxation_q_l3_sigma": report[
                    "maximum_raw_relaxation_q_l3_sigma"
                ],
                "maximum_raw_relaxation_segment_index": report[
                    "maximum_raw_relaxation_segment_index"
                ],
                "maximum_raw_relaxation_iteration": report[
                    "maximum_raw_relaxation_iteration"
                ],
            }
            for report in pilot_reports
        ],
        "verdicts": verdicts,
        "catalog_note": (
            "Standard catalog readiness remains false when an accepted "
            "authorized topology result has multiple wet components."
        ),
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Systematic V6 Adaptive Exact-Zero Delaware Experiment",
        "",
        f"Completed: {summary['completed_utc']}",
        "",
        "## Gate policy",
        "",
        f"- Driver policy: `{summary['gate_policy']['name']}`",
        (
            "- Closure stage order: "
            + ", ".join(
                f"`{value}`"
                for value in summary["gate_policy"]["stage_order"]
            )
        ),
        (
            "- Minimum relaxation reserve inside the work window: "
            f"{summary['gate_policy']['minimum_relaxation_reserve_seconds']} s"
        ),
        "- Relaxation entry requires exact zero superthin, valence, and restricted-edge debt.",
        "",
        "## Verdicts",
        "",
    ]
    lines.extend(
        f"- `{key}`: **{str(value).lower()}**"
        for key, value in summary["verdicts"].items()
    )
    lines.extend(["", "## Pilots", ""])
    for pilot in summary["pilots"]:
        after = pilot["after"]
        lines.extend(
            [
                f"### {pilot['pilot']}",
                "",
                f"- Superthin triangles: {after.get('superthin_triangle_count')}",
                f"- Valence violations: {after.get('count_valence_above_limit')}",
                f"- Maximum valence: {after.get('maximum_valence')}",
                f"- `q_l3_sigma`: {after.get('q_l3_sigma')}",
                f"- Wet components: {after.get('connected_component_count')}",
                (
                    "- Final artifact label: "
                    f"`{pilot['artifact_status']['label']}`"
                ),
                (
                    "- Relaxation entry achieved: "
                    f"{str(pilot['relaxation_entry']['achieved']).lower()}"
                ),
                (
                    "- Policy ladder exhausted without entry: "
                    f"{str(pilot['policy_ladder_exhausted']).lower()}"
                ),
                (
                    "- Maximum closed `q_l3_sigma` (promotable): "
                    f"{pilot.get('maximum_closed_q_l3_sigma')} "
                    "(segment "
                    f"{pilot.get('maximum_closed_q_l3_segment_index')})"
                ),
                (
                    "- Maximum raw relaxation `q_l3_sigma` "
                    "(diagnostic only): "
                    f"{pilot.get('maximum_raw_relaxation_q_l3_sigma')} "
                    "(segment "
                    f"{pilot.get('maximum_raw_relaxation_segment_index')}, "
                    "iteration "
                    f"{pilot.get('maximum_raw_relaxation_iteration')})"
                ),
                f"- Failure atlas: {pilot.get('failure_atlas')}",
                "",
                "Policy stages:",
                "",
            ]
        )
        for stage in pilot["policy_stage_history"]:
            stage_after = stage.get("after", {})
            lines.append(
                "- "
                f"`{stage['preset']}` via `{stage['engine_policy']}`: "
                f"status `{stage['status']}`, "
                f"superthin {stage_after.get('superthin_triangle_count')}, "
                "valence "
                f"{stage_after.get('count_valence_above_limit')}, "
                "exact entry "
                f"{str(stage['exact_zero_debt_entry']).lower()}"
            )
        lines.extend(["", "Relaxation bursts:", ""])
        if pilot["relaxation_stage_history"]:
            for stage in pilot["relaxation_stage_history"]:
                lines.append(
                    "- "
                    f"segment {stage['segment_index']}, burst "
                    f"{stage['burst']}: `q_l3_sigma` "
                    f"{stage['q_l3_sigma_after']} "
                    f"(gain {stage['q_l3_sigma_gain']}), exact zero debt "
                    f"{str(stage['exact_zero_debt_checkpoint']).lower()}"
                )
        else:
            lines.append("- None; exact relaxation entry was not available.")
        lines.append("")
    lines.extend(["## Catalog note", "", summary["catalog_note"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
