#!/usr/bin/env python3
"""Offline regression tests for the benchmark-first quality policy."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.portfolio_conditioning import _stage_regressions
from fvcom_grid_generation.quality_policy import (
    apply_quality_policy,
    classify_failure_codes,
    load_quality_policy,
    public_policy_binding,
)


def audit(
    *,
    valence_count: int,
    valence_excess: int,
    maximum_valence: int,
    superthin: int,
    core_passed: bool = True,
    singly: int = 0,
    q_min: float = 0.8,
    q_l3: float = 0.9,
    angle: float = 40.0,
    area: float = 0.2,
    l_p95: float = 1.0,
    l_max: float = 1.1,
) -> dict:
    return {
        "core_passed": core_passed,
        "core_failures": [] if core_passed else ["nonmanifold_edges"],
        "count_valence_above_8": valence_count,
        "valence_excess_above_8": valence_excess,
        "maximum_valence": maximum_valence,
        "superthin_triangle_count": superthin,
        "singly_connected_triangle_count": singly,
        "q_min": q_min,
        "q_p01": q_min,
        "q_l3_sigma": q_l3,
        "minimum_angle_deg": angle,
        "maximum_adjacent_area_change": area,
        "area_transition_defect_count": 0,
        "l_over_h_count_above_1_55": 0,
        "l_over_h_p95": l_p95,
        "l_over_h_maximum": l_max,
    }


def test_policy_contract_and_unique_assignment() -> None:
    policy = load_quality_policy()
    assert policy["schema_version"] == "fvcom_grid_quality_policy_v1"
    assert public_policy_binding(policy)["sha256"] == policy["sha256"]
    exact = policy["_exact_code_buckets"]
    assert exact["node_valence_above_threshold"] == "benchmark_baseline"
    assert exact["superthin_elements_present"] == "benchmark_baseline"
    assert exact["adjacent_area_change_above_threshold"] == (
        "regional_refinement_debt"
    )
    assert exact["singly_connected_elements_present"] == (
        "regional_refinement_debt"
    )


def test_malformed_and_duplicate_policy_fail_closed() -> None:
    source = Path(load_quality_policy()["source_path"])
    original = json.loads(source.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        malformed = dict(original)
        malformed["unexpected"] = True
        path = root / "malformed.json"
        path.write_text(json.dumps(malformed), encoding="utf-8")
        try:
            load_quality_policy(path)
        except ValueError:
            pass
        else:
            raise AssertionError("unknown policy fields must fail")

        duplicate = json.loads(source.read_text(encoding="utf-8"))
        duplicate["buckets"]["regional_refinement_debt"]["criteria"][0][
            "failure_codes"
        ].append("node_valence_above_threshold")
        path = root / "duplicate.json"
        path.write_text(json.dumps(duplicate), encoding="utf-8")
        try:
            load_quality_policy(path)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate code assignment must fail")


def test_bucket_decisions_and_unknown_fail_closed() -> None:
    quality: dict = {}
    apply_quality_policy(
        quality,
        [
            "q_l3_sigma_below_threshold",
            "adjacent_area_change_above_threshold",
            "singly_connected_elements_present",
        ],
    )
    assert quality["benchmark_grid_baseline_ready"]
    assert quality["fvcom_ready"] and quality["accepted"]
    assert quality["failure_taxonomy"] == []
    assert len(quality["regional_refinement_debt"]) == 3

    blocked: dict = {}
    apply_quality_policy(blocked, ["node_valence_above_threshold"])
    assert not blocked["benchmark_grid_baseline_ready"]
    assert blocked["failure_taxonomy"] == ["node_valence_above_threshold"]

    unknown = classify_failure_codes(["future_unclassified_finding"])
    assert unknown["unclassified"] == ["future_unclassified_finding"]
    assert unknown["benchmark_baseline"] == ["future_unclassified_finding"]

    submission = classify_failure_codes(["open_boundary_forcing_missing"])
    assert submission["benchmark_baseline"] == []
    assert submission["submission_preconditions"] == [
        "open_boundary_forcing_missing"
    ]


def test_delaware_40_node_priority_regression() -> None:
    before = audit(
        valence_count=40,
        valence_excess=40,
        maximum_valence=9,
        superthin=3,
        singly=0,
        q_min=0.11,
        q_l3=0.85,
        angle=5.1,
        area=0.90,
        l_p95=1.2,
        l_max=1.5,
    )
    after = audit(
        valence_count=0,
        valence_excess=0,
        maximum_valence=8,
        superthin=3,
        singly=17,
        q_min=0.02,
        q_l3=0.50,
        angle=2.0,
        area=0.99,
        l_p95=3.0,
        l_max=7.0,
    )
    assert _stage_regressions(before, after, minimal_policy=True) == []


def test_superthin_priority_and_structural_rollback() -> None:
    before = audit(
        valence_count=0,
        valence_excess=0,
        maximum_valence=8,
        superthin=4,
    )
    after = audit(
        valence_count=0,
        valence_excess=0,
        maximum_valence=8,
        superthin=2,
        singly=40,
        q_min=0.01,
        q_l3=0.1,
        angle=1.0,
        area=0.99,
        l_p95=5.0,
        l_max=8.0,
    )
    assert _stage_regressions(before, after, minimal_policy=True) == []

    valence_regression = dict(after)
    valence_regression.update(
        count_valence_above_8=1,
        valence_excess_above_8=1,
        maximum_valence=9,
    )
    assert "valence_debt_regressed" in _stage_regressions(
        before, valence_regression, minimal_policy=True
    )

    structural = dict(after)
    structural["core_passed"] = False
    structural["core_failures"] = ["nonmanifold_edges"]
    assert "terminal_core:nonmanifold_edges" in _stage_regressions(
        before, structural, minimal_policy=True
    )


def main() -> None:
    tests = [
        test_policy_contract_and_unique_assignment,
        test_malformed_and_duplicate_policy_fail_closed,
        test_bucket_decisions_and_unknown_fail_closed,
        test_delaware_40_node_priority_regression,
        test_superthin_priority_and_structural_rollback,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("All benchmark-first grid-quality policy tests passed.")


if __name__ == "__main__":
    main()
