"""Reusable gate policies for research-only systematic V6 conditioning."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping


STRICT_GATE_POLICY = "strict-v6"
TOPOLOGY_PRIORITY_GATE_POLICY = "topology-priority-v1"
SOFT_TOPOLOGY_GATE_POLICY = "soft-topology-v1"
TOPOLOGY_ESCROW_GATE_POLICY = "topology-escrow-v1"
ADAPTIVE_GATE_POLICY = "adaptive-topology-v1"

GATE_POLICIES = (
    STRICT_GATE_POLICY,
    TOPOLOGY_PRIORITY_GATE_POLICY,
    TOPOLOGY_ESCROW_GATE_POLICY,
)
FIXED_GATE_POLICIES = (
    STRICT_GATE_POLICY,
    TOPOLOGY_PRIORITY_GATE_POLICY,
    SOFT_TOPOLOGY_GATE_POLICY,
    TOPOLOGY_ESCROW_GATE_POLICY,
)
ADAPTIVE_POLICY_LADDER = FIXED_GATE_POLICIES

POLICY_BUDGET_KEYS = (
    "valence_l_over_h_count_increase",
    "closure_q_l3_sigma_decrease",
    "closure_q_p01_decrease",
    "closure_minimum_angle_p01_decrease",
    "closure_l_over_h_count_increase",
    "closure_l_over_h_p95_increase",
    "closure_l_over_h_maximum_increase",
    "closure_area_transition_count_increase",
)

GATE_POLICY_PRESETS: dict[str, dict[str, Any]] = {
    STRICT_GATE_POLICY: {
        "name": STRICT_GATE_POLICY,
        "engine_name": STRICT_GATE_POLICY,
        "escrow_enabled": False,
        "valence_l_over_h_count_increase": 0,
        "closure_q_l3_sigma_decrease": 0.0,
        "closure_q_p01_decrease": 0.0,
        "closure_minimum_angle_p01_decrease": 0.0,
        "closure_l_over_h_count_increase": 0,
        "closure_l_over_h_p95_increase": 0.0,
        "closure_l_over_h_maximum_increase": 0.0,
        "closure_area_transition_count_increase": 0,
    },
    TOPOLOGY_PRIORITY_GATE_POLICY: {
        "name": TOPOLOGY_PRIORITY_GATE_POLICY,
        "engine_name": TOPOLOGY_PRIORITY_GATE_POLICY,
        "escrow_enabled": False,
        "valence_l_over_h_count_increase": 1,
        "closure_q_l3_sigma_decrease": 0.0,
        "closure_q_p01_decrease": 0.0,
        "closure_minimum_angle_p01_decrease": 0.0,
        "closure_l_over_h_count_increase": 1,
        "closure_l_over_h_p95_increase": 0.0,
        "closure_l_over_h_maximum_increase": 0.0,
        "closure_area_transition_count_increase": 0,
    },
    SOFT_TOPOLOGY_GATE_POLICY: {
        "name": SOFT_TOPOLOGY_GATE_POLICY,
        "engine_name": TOPOLOGY_PRIORITY_GATE_POLICY,
        "escrow_enabled": False,
        "valence_l_over_h_count_increase": 4,
        "closure_q_l3_sigma_decrease": 0.005,
        "closure_q_p01_decrease": 0.005,
        "closure_minimum_angle_p01_decrease": 0.5,
        "closure_l_over_h_count_increase": 8,
        "closure_l_over_h_p95_increase": 0.02,
        "closure_l_over_h_maximum_increase": 0.05,
        "closure_area_transition_count_increase": 100,
    },
    TOPOLOGY_ESCROW_GATE_POLICY: {
        "name": TOPOLOGY_ESCROW_GATE_POLICY,
        "engine_name": TOPOLOGY_ESCROW_GATE_POLICY,
        "escrow_enabled": True,
        "escrow_maximum_superthin_count": 25,
        "escrow_maximum_superthin_severity": 25.0,
        "escrow_maximum_valence": 12,
        "escrow_maximum_valence_count_rebound": 8,
        "escrow_maximum_valence_excess_rebound": 16,
        "valence_l_over_h_count_increase": 4,
        "closure_q_l3_sigma_decrease": 0.005,
        "closure_q_p01_decrease": 0.005,
        "closure_minimum_angle_p01_decrease": 0.5,
        "closure_l_over_h_count_increase": 8,
        "closure_l_over_h_p95_increase": 0.02,
        "closure_l_over_h_maximum_increase": 0.05,
        "closure_area_transition_count_increase": 100,
    },
}

EVIDENCE_RETRY_SOFT_GATES = frozenset(
    {
        "q_l3_sigma_soft_budget_exceeded",
        "q_p01_soft_budget_exceeded",
        "minimum_angle_p01_soft_budget_exceeded",
        "l_over_h_count_soft_budget_exceeded",
        "l_over_h_p95_soft_budget_exceeded",
        "l_over_h_maximum_soft_budget_exceeded",
        "area_transition_soft_budget_exceeded",
        "topology_escrow_superthin_count_exceeded",
        "topology_escrow_superthin_severity_exceeded",
        "topology_escrow_valence_count_rebound_exceeded",
        "topology_escrow_valence_excess_rebound_exceeded",
    }
)
EVIDENCE_RETRY_CEILINGS = {
    "valence_l_over_h_count_increase": 6,
    "closure_q_l3_sigma_decrease": 0.010,
    "closure_q_p01_decrease": 0.010,
    "closure_minimum_angle_p01_decrease": 1.0,
    "closure_l_over_h_count_increase": 16,
    "closure_l_over_h_p95_increase": 0.04,
    "closure_l_over_h_maximum_increase": 0.10,
    "closure_area_transition_count_increase": 200,
    "escrow_maximum_superthin_count": 32,
    "escrow_maximum_superthin_severity": 32.0,
    "escrow_maximum_valence_count_rebound": 12,
    "escrow_maximum_valence_excess_rebound": 24,
}


def gate_policy_preset(name: str) -> dict[str, Any]:
    """Return an independent fixed-policy preset."""
    try:
        return dict(GATE_POLICY_PRESETS[str(name)])
    except KeyError as exc:
        raise ValueError(
            "systematic V6 gate policy must be one of "
            + ", ".join(FIXED_GATE_POLICIES)
        ) from exc


def resolve_gate_policy_stages(
    requested: str,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one fixed policy or the research adaptive policy ladder."""
    requested = str(requested)
    overrides = {
        str(key): value
        for key, value in (overrides or {}).items()
        if value is not None
    }
    unknown = sorted(set(overrides) - set(POLICY_BUDGET_KEYS))
    if unknown:
        raise ValueError(
            "unsupported gate-policy overrides: " + ", ".join(unknown)
        )
    stage_names = (
        ADAPTIVE_POLICY_LADDER
        if requested == ADAPTIVE_GATE_POLICY
        else (requested,)
    )
    stages: list[dict[str, Any]] = []
    for stage_name in stage_names:
        policy = gate_policy_preset(stage_name)
        if requested != ADAPTIVE_GATE_POLICY or stage_name != STRICT_GATE_POLICY:
            policy.update(overrides)
        invalid = [
            key
            for key in POLICY_BUDGET_KEYS
            if float(policy[key]) < 0.0
        ]
        if invalid:
            raise ValueError(
                "gate-policy budgets must be nonnegative: "
                + ", ".join(invalid)
            )
        if str(policy["engine_name"]) not in GATE_POLICIES:
            raise ValueError(
                f"unsupported systematic-V6 engine policy: "
                f"{policy['engine_name']}"
            )
        stages.append(policy)
    return {
        "name": requested,
        "adaptive": bool(requested == ADAPTIVE_GATE_POLICY),
        "stage_order": [str(value["name"]) for value in stages],
        "stages": stages,
        "overrides": overrides,
    }


def loop_policy_overrides(name: str) -> dict[str, Any]:
    """Map a fixed preset onto ``SystematicV6LoopConfig`` field names."""
    policy = gate_policy_preset(name)
    return {
        "closure_gate_policy": str(policy["engine_name"]),
        "closure_max_q_l3_sigma_decrease": float(
            policy["closure_q_l3_sigma_decrease"]
        ),
        "closure_max_q_p01_decrease": float(
            policy["closure_q_p01_decrease"]
        ),
        "closure_max_minimum_angle_p01_decrease": float(
            policy["closure_minimum_angle_p01_decrease"]
        ),
        "closure_max_l_over_h_count_increase": int(
            policy["closure_l_over_h_count_increase"]
        ),
        "closure_max_l_over_h_p95_increase": float(
            policy["closure_l_over_h_p95_increase"]
        ),
        "closure_max_l_over_h_maximum_increase": float(
            policy["closure_l_over_h_maximum_increase"]
        ),
        "closure_max_area_transition_count_increase": int(
            policy["closure_area_transition_count_increase"]
        ),
        "escrow_maximum_superthin_count": int(
            policy.get("escrow_maximum_superthin_count", 25)
        ),
        "escrow_maximum_superthin_severity": float(
            policy.get("escrow_maximum_superthin_severity", 25.0)
        ),
        "escrow_maximum_valence": int(
            policy.get("escrow_maximum_valence", 12)
        ),
        "escrow_maximum_valence_count_rebound": int(
            policy.get("escrow_maximum_valence_count_rebound", 8)
        ),
        "escrow_maximum_valence_excess_rebound": int(
            policy.get("escrow_maximum_valence_excess_rebound", 16)
        ),
    }


def topology_policy_overrides(name: str) -> dict[str, Any]:
    """Map a fixed preset onto ``AggressiveConditioningConfig`` fields."""
    policy = gate_policy_preset(name)
    return {
        "max_valence_l_over_h_count_increase": int(
            policy["valence_l_over_h_count_increase"]
        ),
        "topology_escrow_enabled": bool(policy["escrow_enabled"]),
        "topology_escrow_maximum_superthin_count": int(
            policy.get("escrow_maximum_superthin_count", 25)
        ),
        "topology_escrow_maximum_superthin_severity": float(
            policy.get("escrow_maximum_superthin_severity", 25.0)
        ),
        "topology_escrow_maximum_valence": int(
            policy.get("escrow_maximum_valence", 12)
        ),
    }


def apply_gate_policy(loop_config: Any, name: str) -> Any:
    """Return a dataclass copy with one fixed policy applied."""
    return replace(loop_config, **loop_policy_overrides(name))


def build_evidence_retry_policy(
    baseline: Mapping[str, Any],
    before: Mapping[str, Any],
    trial: Mapping[str, Any],
    rejection_gates: list[str],
    base_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one bounded soft-only retry from an observed rejected round."""
    gates = sorted(set(map(str, rejection_gates)))
    decision: dict[str, Any] = {
        "schema_version": "fvcom_systematic_v6_evidence_retry_v1",
        "eligible": False,
        "rejection_gates": gates,
        "ceilings": dict(EVIDENCE_RETRY_CEILINGS),
    }
    if not gates:
        decision["reason"] = "no_rejected_closure_round"
        return decision
    nonsoft = sorted(set(gates) - EVIDENCE_RETRY_SOFT_GATES)
    if nonsoft:
        decision.update(
            {
                "reason": "nonsoft_rejection_gate",
                "nonsoft_rejection_gates": nonsoft,
            }
        )
        return decision

    thin_regressed = bool(
        int(trial["superthin_triangle_count"])
        > int(before["superthin_triangle_count"])
        or float(trial.get("superthin_severity_sum", 0.0))
        > float(before.get("superthin_severity_sum", 0.0)) + 1.0e-10
    )
    valence_regressed = bool(
        int(trial["count_valence_above_limit"])
        > int(before["count_valence_above_limit"])
        or int(trial["valence_excess_sum"])
        > int(before["valence_excess_sum"])
    )
    primary_progress = bool(
        int(trial["superthin_triangle_count"])
        < int(before["superthin_triangle_count"])
        or float(trial.get("superthin_severity_sum", 0.0)) + 1.0e-10
        < float(before.get("superthin_severity_sum", 0.0))
        or int(trial["count_valence_above_limit"])
        < int(before["count_valence_above_limit"])
        or int(trial["valence_excess_sum"])
        < int(before["valence_excess_sum"])
    )
    decision.update(
        {
            "primary_topology_progress": primary_progress,
            "thin_debt_regressed": thin_regressed,
            "valence_debt_regressed": valence_regressed,
        }
    )
    if not primary_progress:
        decision["reason"] = "no_primary_topology_progress"
        return decision
    if thin_regressed and valence_regressed:
        decision["reason"] = "dual_primary_debt_regression"
        return decision

    required = {
        "closure_q_l3_sigma_decrease": max(
            0.0, float(baseline["q_l3_sigma"]) - float(trial["q_l3_sigma"])
        ),
        "closure_q_p01_decrease": max(
            0.0, float(baseline["q_p01"]) - float(trial["q_p01"])
        ),
        "closure_minimum_angle_p01_decrease": max(
            0.0,
            float(baseline.get("minimum_angle_p01_deg", 0.0))
            - float(trial.get("minimum_angle_p01_deg", 0.0)),
        ),
        "closure_l_over_h_count_increase": max(
            0,
            int(trial["l_over_h_count_above_1_55"])
            - int(baseline["l_over_h_count_above_1_55"]),
        ),
        "closure_l_over_h_p95_increase": max(
            0.0,
            float(trial["l_over_h_p95"]) - float(baseline["l_over_h_p95"]),
        ),
        "closure_l_over_h_maximum_increase": max(
            0.0,
            float(trial["l_over_h_maximum"])
            - float(baseline["l_over_h_maximum"]),
        ),
        "closure_area_transition_count_increase": max(
            0,
            int(trial["area_transition_count_above_0_50"])
            - int(baseline["area_transition_count_above_0_50"]),
        ),
        "escrow_maximum_superthin_count": int(
            trial["superthin_triangle_count"]
        ),
        "escrow_maximum_superthin_severity": float(
            trial.get("superthin_severity_sum", 0.0)
        ),
        "escrow_maximum_valence_count_rebound": max(
            0,
            int(trial["count_valence_above_limit"])
            - int(before["count_valence_above_limit"]),
        ),
        "escrow_maximum_valence_excess_rebound": max(
            0,
            int(trial["valence_excess_sum"])
            - int(before["valence_excess_sum"]),
        ),
    }
    decision["required_budget"] = required
    policy = dict(base_policy)
    policy.update(
        {
            "name": "topology-escrow-evidence-v1",
            "engine_name": TOPOLOGY_ESCROW_GATE_POLICY,
            "escrow_enabled": True,
        }
    )
    exceeded: list[str] = []

    def promote_float(
        key: str,
        gate: str,
        margin: float,
        quantum: float,
    ) -> None:
        if gate not in gates:
            return
        target = max(
            float(policy[key]),
            _round_up(float(required[key]) + margin, quantum),
        )
        if target > float(EVIDENCE_RETRY_CEILINGS[key]) + 1.0e-12:
            exceeded.append(key)
        else:
            policy[key] = target

    def promote_int(
        key: str,
        gate: str,
        margin: int,
        quantum: int = 1,
    ) -> None:
        if gate not in gates:
            return
        raw = int(required[key]) + int(margin)
        target = max(
            int(policy[key]),
            int(math.ceil(raw / max(1, quantum)) * max(1, quantum)),
        )
        if target > int(EVIDENCE_RETRY_CEILINGS[key]):
            exceeded.append(key)
        else:
            policy[key] = target

    promote_float(
        "closure_q_l3_sigma_decrease",
        "q_l3_sigma_soft_budget_exceeded",
        0.001,
        0.001,
    )
    promote_float(
        "closure_q_p01_decrease",
        "q_p01_soft_budget_exceeded",
        0.001,
        0.001,
    )
    promote_float(
        "closure_minimum_angle_p01_decrease",
        "minimum_angle_p01_soft_budget_exceeded",
        0.1,
        0.1,
    )
    promote_int(
        "closure_l_over_h_count_increase",
        "l_over_h_count_soft_budget_exceeded",
        2,
    )
    promote_float(
        "closure_l_over_h_p95_increase",
        "l_over_h_p95_soft_budget_exceeded",
        0.005,
        0.005,
    )
    promote_float(
        "closure_l_over_h_maximum_increase",
        "l_over_h_maximum_soft_budget_exceeded",
        0.01,
        0.01,
    )
    promote_int(
        "closure_area_transition_count_increase",
        "area_transition_soft_budget_exceeded",
        25,
        25,
    )
    promote_int(
        "escrow_maximum_superthin_count",
        "topology_escrow_superthin_count_exceeded",
        2,
    )
    promote_float(
        "escrow_maximum_superthin_severity",
        "topology_escrow_superthin_severity_exceeded",
        2.0,
        1.0,
    )
    promote_int(
        "escrow_maximum_valence_count_rebound",
        "topology_escrow_valence_count_rebound_exceeded",
        1,
    )
    promote_int(
        "escrow_maximum_valence_excess_rebound",
        "topology_escrow_valence_excess_rebound_exceeded",
        2,
    )
    if any("l_over_h" in value for value in gates):
        policy["valence_l_over_h_count_increase"] = min(
            int(EVIDENCE_RETRY_CEILINGS["valence_l_over_h_count_increase"]),
            int(policy["valence_l_over_h_count_increase"]) + 1,
        )
    if exceeded:
        decision.update(
            {
                "reason": "evidence_retry_ceiling_exceeded",
                "ceiling_failures": sorted(set(exceeded)),
            }
        )
        return decision
    decision.update(
        {
            "eligible": True,
            "reason": "soft_only_primary_progress",
            "policy": policy,
        }
    )
    return decision


def _round_up(value: float, quantum: float) -> float:
    return float(
        math.ceil((float(value) - 1.0e-12) / float(quantum))
        * float(quantum)
    )
