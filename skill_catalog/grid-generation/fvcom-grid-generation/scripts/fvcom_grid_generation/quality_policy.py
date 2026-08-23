"""Authoritative benchmark-first FVCOM grid-quality policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "fvcom_grid_quality_policy_v1"
REQUIRED_BUCKETS = {
    "benchmark_baseline",
    "regional_refinement_debt",
    "quality_advisories",
}
REQUIRED_THRESHOLDS = {
    "maximum_node_valence",
    "superthin_quality_below",
    "superthin_minimum_angle_below_deg",
    "regional_q_l3_sigma_target_above",
    "regional_minimum_angle_target_deg",
    "regional_maximum_angle_target_deg",
    "regional_maximum_adjacent_area_change",
    "regional_maximum_bathymetric_slope",
    "regional_target_size_l_over_h_p95",
    "regional_target_size_l_over_h_maximum",
    "roundtrip_coordinate_tolerance_m",
    "planning_node_limit",
    "hard_node_limit",
}


def policy_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "references"
        / "fvcom_grid_quality_policy_v1.json"
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_quality_policy(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path).resolve() if path is not None else policy_path()
    document = json.loads(source.read_text(encoding="utf-8-sig"))
    allowed = {
        "schema_version",
        "policy_id",
        "description",
        "priority_order",
        "thresholds",
        "buckets",
        "submission_preconditions",
        "compatibility_aliases",
        "mesher_bakeoff_adapter",
        "transaction_policy",
    }
    unknown_top = sorted(set(document) - allowed)
    missing_top = sorted(allowed - set(document))
    if unknown_top or missing_top:
        raise ValueError(
            f"quality policy top-level contract mismatch; missing={missing_top}, unknown={unknown_top}"
        )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("quality policy schema_version is not fvcom_grid_quality_policy_v1")
    if not isinstance(document.get("policy_id"), str) or not document["policy_id"]:
        raise ValueError("quality policy requires a nonempty policy_id")
    if set(document.get("buckets", {})) != REQUIRED_BUCKETS:
        raise ValueError("quality policy must define exactly the three standard buckets")
    if document.get("priority_order") != [
        "absolute_structural_invariants",
        "valence_debt",
        "superthin_debt",
        "regional_refinement_debt",
        "quality_advisories",
    ]:
        raise ValueError("quality policy priority order is invalid")
    if document.get("compatibility_aliases") != {
        "fvcom_ready": "benchmark_grid_baseline_ready",
        "accepted": "benchmark_grid_baseline_ready",
    }:
        raise ValueError("quality policy compatibility aliases are invalid")

    thresholds = document.get("thresholds", {})
    if set(thresholds) != REQUIRED_THRESHOLDS:
        raise ValueError("quality policy threshold contract is incomplete or has unknown keys")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in thresholds.values()
    ):
        raise ValueError("quality policy thresholds must be numeric")
    if (
        int(thresholds["maximum_node_valence"]) != 8
        or float(thresholds["superthin_quality_below"]) != 0.10
        or float(thresholds["superthin_minimum_angle_below_deg"]) != 5.0
        or int(thresholds["planning_node_limit"])
        >= int(thresholds["hard_node_limit"])
    ):
        raise ValueError("quality policy hard thresholds are invalid")

    seen_exact: dict[str, str] = {}
    seen_prefix: dict[str, str] = {}
    for bucket_name in ("benchmark_baseline", "regional_refinement_debt"):
        bucket = document["buckets"][bucket_name]
        if bool(bucket.get("blocking")) != (bucket_name == "benchmark_baseline"):
            raise ValueError(f"invalid blocking flag for {bucket_name}")
        criteria = bucket.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            raise ValueError(f"{bucket_name} requires nonempty criteria")
        criterion_ids: set[str] = set()
        for criterion in criteria:
            unknown_criterion = sorted(
                set(criterion) - {"id", "priority", "failure_codes", "failure_prefixes"}
            )
            if unknown_criterion:
                raise ValueError(
                    f"unknown criterion keys in {bucket_name}: {unknown_criterion}"
                )
            criterion_id = str(criterion.get("id", ""))
            if not criterion_id or criterion_id in criterion_ids:
                raise ValueError(f"duplicate or missing criterion id in {bucket_name}")
            criterion_ids.add(criterion_id)
            for code in criterion.get("failure_codes", []):
                code = str(code)
                if not code or code in seen_exact:
                    raise ValueError(f"failure code is empty or assigned more than once: {code}")
                seen_exact[code] = bucket_name
            for prefix in criterion.get("failure_prefixes", []):
                prefix = str(prefix)
                if not prefix or prefix in seen_prefix:
                    raise ValueError(f"failure prefix is empty or assigned more than once: {prefix}")
                seen_prefix[prefix] = bucket_name
    prefixes = list(seen_prefix)
    if any(
        left != right and (left.startswith(right) or right.startswith(left))
        for index, left in enumerate(prefixes)
        for right in prefixes[index + 1 :]
    ):
        raise ValueError("quality policy failure prefixes overlap")
    advisory_bucket = document["buckets"]["quality_advisories"]
    if set(advisory_bucket) != {"blocking", "metric_paths"}:
        raise ValueError("quality_advisories contract has unknown or missing keys")
    if bool(advisory_bucket.get("blocking")) or not advisory_bucket.get("metric_paths"):
        raise ValueError("quality_advisories requires metric_paths")
    submission = document.get("submission_preconditions", {})
    if not submission.get("failure_codes") or not submission.get("required_status"):
        raise ValueError("submission_preconditions are incomplete")
    overlap = sorted(set(submission["failure_codes"]) & set(seen_exact))
    if overlap:
        raise ValueError(f"submission-only codes also occur in quality buckets: {overlap}")
    adapter = document.get("mesher_bakeoff_adapter", {})
    if (
        adapter.get("accepted_path") != "benchmark_grid_baseline_ready"
        or adapter.get("failure_taxonomy_path") != "failure_taxonomy"
        or not adapter.get("hard_gate_ids")
        or not adapter.get("metric_paths")
    ):
        raise ValueError("mesher_bakeoff_adapter is incomplete")
    transaction = document.get("transaction_policy", {})
    required_transaction = {
        "absolute_invariants_always_block",
        "valence_tuple",
        "superthin_tuple",
        "temporary_superthin_escrow_after_valence_improvement",
        "class_2_or_3_regression_may_rollback",
        "isolated_triangle_deletion_allowed",
    }
    if set(transaction) != required_transaction:
        raise ValueError("transaction_policy contract is incomplete or has unknown keys")
    if (
        transaction["absolute_invariants_always_block"] is not True
        or transaction["temporary_superthin_escrow_after_valence_improvement"] is not True
        or transaction["class_2_or_3_regression_may_rollback"] is not False
        or transaction["isolated_triangle_deletion_allowed"] is not False
        or transaction["valence_tuple"]
        != ["count_valence_above_8", "valence_excess_above_8", "maximum_valence"]
        or transaction["superthin_tuple"]
        != ["superthin_triangle_count", "superthin_severity_sum"]
    ):
        raise ValueError("transaction_policy semantics are invalid")

    return {
        **document,
        "source_path": str(source),
        "sha256": sha256_file(source),
        "_exact_code_buckets": seen_exact,
        "_prefix_buckets": seen_prefix,
    }


def public_policy_binding(policy: dict[str, Any] | None = None) -> dict[str, str]:
    selected = policy or load_quality_policy()
    return {
        "schema_version": str(selected["schema_version"]),
        "policy_id": str(selected["policy_id"]),
        "sha256": str(selected["sha256"]),
        "path": "references/fvcom_grid_quality_policy_v1.json",
    }


def classify_failure_codes(
    codes: Iterable[str],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = policy or load_quality_policy()
    exact = selected["_exact_code_buckets"]
    prefixes = selected["_prefix_buckets"]
    submission_codes = set(selected["submission_preconditions"]["failure_codes"])
    output: dict[str, list[str]] = {
        "benchmark_baseline": [],
        "regional_refinement_debt": [],
        "submission_preconditions": [],
        "unclassified": [],
    }
    for raw_code in codes:
        code = str(raw_code)
        if code in exact:
            output[exact[code]].append(code)
            continue
        matches = sorted({bucket for prefix, bucket in prefixes.items() if code.startswith(prefix)})
        if len(matches) == 1:
            output[matches[0]].append(code)
        elif code in submission_codes:
            output["submission_preconditions"].append(code)
        else:
            output["unclassified"].append(code)
    for key in output:
        output[key] = sorted(set(output[key]))
    # Unknown findings fail closed instead of silently becoming advisory debt.
    output["benchmark_baseline"] = sorted(
        set(output["benchmark_baseline"] + output["unclassified"])
    )
    return {**output, "policy": public_policy_binding(selected)}


def _criterion_for_code(code: str, bucket_name: str, policy: dict[str, Any]) -> str:
    for criterion in policy["buckets"][bucket_name].get("criteria", []):
        if code in criterion.get("failure_codes", []):
            return str(criterion["id"])
        if any(code.startswith(str(prefix)) for prefix in criterion.get("failure_prefixes", [])):
            return str(criterion["id"])
    return "unclassified_fail_closed"


def regional_debt_records(
    codes: Iterable[str],
    policy: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    selected = policy or load_quality_policy()
    return [
        {
            "code": code,
            "criterion": _criterion_for_code(code, "regional_refinement_debt", selected),
            "decision_role": "nonblocking_regional_refinement",
        }
        for code in sorted(set(map(str, codes)))
    ]


def apply_quality_policy(
    quality: dict[str, Any],
    findings: Iterable[str],
    *,
    advisories: dict[str, Any] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = policy or load_quality_policy()
    classified = classify_failure_codes(findings, selected)
    baseline = classified["benchmark_baseline"]
    ready = not baseline
    quality["quality_policy"] = public_policy_binding(selected)
    quality["baseline_failure_taxonomy"] = baseline
    quality["regional_refinement_debt"] = regional_debt_records(
        classified["regional_refinement_debt"], selected
    )
    quality["quality_advisories"] = dict(advisories or {})
    quality["unclassified_quality_findings"] = classified["unclassified"]
    quality["benchmark_grid_baseline_ready"] = ready
    quality["failure_taxonomy"] = baseline
    quality["fvcom_ready"] = ready
    quality["accepted"] = ready
    return quality


__all__ = [
    "SCHEMA_VERSION",
    "apply_quality_policy",
    "classify_failure_codes",
    "load_quality_policy",
    "policy_path",
    "public_policy_binding",
    "regional_debt_records",
    "sha256_file",
]
