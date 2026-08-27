#!/usr/bin/env python3
"""Focused tests for Grid Generation's reviewed Adaptive-v2 gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.open_exterior import (  # noqa: E402
    validate_grid_boundary_gate,
)


def resolution_fixture(root: Path, **qa_overrides: object) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    package = root / "boundary_resolution.gpkg"
    loops = root / "model_boundary_loops.gpkg"
    package.write_bytes(b"adaptive-v2-package")
    loops.write_bytes(b"source-loops")
    qa = {
        "resolved_domain_valid": True,
        "open_arc_land_intersection_m": 0.0,
        "open_arc_exterior_overlap_fraction": 1.0,
        "protected_underresolved_passage_count": 0,
        "maximum_edge_to_target_ratio": 1.10,
        "p95_edge_to_target_ratio": 1.02,
        "maximum_target_gradation": 0.15,
    }
    qa.update(qa_overrides)
    manifest = {
        "schema_version": "fvcom_boundary_resolution_manifest_v2",
        "profile": "adaptive-coastal-v2",
        "final_status": "pass",
        "failure_taxonomy": [],
        "inputs": {"model_boundary_loops_gpkg": str(loops)},
        "outputs": {"boundary_resolution_gpkg": str(package)},
        "qa": qa,
    }
    path = root / "boundary_resolution_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_strict_policy_still_requires_open_exterior_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        resolution = resolution_fixture(Path(tmp))
        report = validate_grid_boundary_gate(
            None,
            resolution,
            policy="strict",
        )
        assert report["passed"] is False
        assert "open_exterior_contract_missing" in report["failure_taxonomy"]


def test_reviewed_policy_accepts_exact_passing_adaptive_v2() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        resolution = resolution_fixture(Path(tmp))
        report = validate_grid_boundary_gate(
            None,
            resolution,
            policy="reviewed-adaptive-v2",
        )
        assert report["passed"] is True, report
        assert report["decision_basis"] == "reviewed_adaptive_v2_package"
        assert report["boundary_resolution"]["artifacts"][
            "boundary_resolution_manifest"
        ]["sha256"]
        assert report["advisory_taxonomy"] == [
            "upstream_open_exterior:open_exterior_contract_missing"
        ]


def test_reviewed_policy_cannot_waive_resolved_land_crossing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        resolution = resolution_fixture(
            Path(tmp),
            open_arc_land_intersection_m=0.1,
        )
        report = validate_grid_boundary_gate(
            None,
            resolution,
            policy="reviewed-adaptive-v2",
        )
        assert report["passed"] is False
        assert (
            "reviewed_boundary_open_arc_intersects_land"
            in report["failure_taxonomy"]
        )


def test_reviewed_policy_rejects_mismatched_arc_lineage() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        resolution = resolution_fixture(root / "selected")
        other = resolution_fixture(root / "other")
        arc = root / "bdry_arc_manifest.json"
        arc.write_text(
            json.dumps(
                {"outputs": {"boundary_resolution_manifest": str(other)}}
            ),
            encoding="utf-8",
        )
        report = validate_grid_boundary_gate(
            arc,
            resolution,
            policy="reviewed-adaptive-v2",
        )
        assert report["passed"] is False
        assert (
            "reviewed_boundary_resolution_lineage_mismatch"
            in report["failure_taxonomy"]
        )


def main() -> int:
    tests = [
        test_strict_policy_still_requires_open_exterior_contract,
        test_reviewed_policy_accepts_exact_passing_adaptive_v2,
        test_reviewed_policy_cannot_waive_resolved_land_crossing,
        test_reviewed_policy_rejects_mismatched_arc_lineage,
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} reviewed boundary-gate tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
