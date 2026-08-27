#!/usr/bin/env python3
"""Focused tests for Grid Generation's reviewed Adaptive-v2 gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.open_exterior import (  # noqa: E402
    validate_grid_boundary_gate,
    validate_open_exterior_contract,
)
from fvcom_grid_generation.gmsh_experiment import check_case_readiness  # noqa: E402


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


def test_strict_contract_accepts_existing_workspace_relative_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        contract_dir = root / "stage" / "open_exterior"
        evidence_dir = root / "evidence"
        contract_dir.mkdir(parents=True)
        evidence_dir.mkdir()
        review_map = evidence_dir / "review.png"
        review_map.write_bytes(b"review-map")
        source_hashes = {"candidate": "fixture-sha"}
        decision = evidence_dir / "decision.json"
        decision.write_text(
            json.dumps(
                {
                    "decision_actor": {"kind": "codex_agent"},
                    "inspected_map_sha256": hashlib.sha256(
                        review_map.read_bytes()
                    ).hexdigest(),
                    "bound_source_hashes": source_hashes,
                }
            ),
            encoding="utf-8",
        )
        contract = contract_dir / "contract.json"
        contract.write_text(
            json.dumps(
                {
                    "schema_version": "fvcom_open_exterior_contract_v1",
                    "report_only": False,
                    "downstream_eligible": True,
                    "hard_metrics": {
                        "absolute_gate_pass": True,
                        "fraction_gate_pass": True,
                        "coverage_gate_pass": True,
                        "absolute_limit_m": 250.0,
                    },
                    "agent_decision": {
                        "status": "pass",
                        "path": "evidence/decision.json",
                        "sha256": hashlib.sha256(
                            decision.read_bytes()
                        ).hexdigest(),
                    },
                    "map": {"path": "evidence/review.png"},
                    "source_hashes": source_hashes,
                    "obc_geometry": {
                        "expected_count": 0,
                        "delivered_count": 0,
                        "simple_nonbranching": True,
                        "nonendpoint_land_crossing_m": 0.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        original_cwd = Path.cwd()
        try:
            os.chdir(root)
            report = validate_open_exterior_contract(
                contract,
                require_topology_coverage=False,
            )
        finally:
            os.chdir(original_cwd)
        assert report["passed"] is True, report


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


def test_reviewed_policy_accepts_hash_identical_relocated_arc_lineage() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        recorded = resolution_fixture(root / "recorded")
        selected_dir = root / "selected"
        selected_dir.mkdir()
        selected = selected_dir / "boundary_resolution_manifest.json"
        shutil.copy2(recorded, selected)
        arc = root / "bdry_arc_manifest.json"
        arc.write_text(
            json.dumps(
                {"outputs": {"boundary_resolution_manifest": str(recorded)}}
            ),
            encoding="utf-8",
        )
        report = validate_grid_boundary_gate(
            arc,
            selected,
            policy="reviewed-adaptive-v2",
        )
        assert report["passed"] is True, report
        lineage = report["boundary_resolution"]["artifacts"][
            "boundary_resolution_lineage"
        ]
        assert lineage["mode"] == "sha256_identical_relocation"
        assert lineage["recorded_path"] == str(recorded.resolve())
        assert lineage["selected_path"] == str(selected.resolve())
        assert lineage["sha256"]


def test_gmsh_preflight_consumes_reviewed_policy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        resolution = resolution_fixture(root / "boundary")
        case = root / "case.json"
        case.write_text(
            json.dumps(
                {
                    "schema_version": "gmsh_fvcom_case_v1",
                    "case_id": "reviewed_gate_fixture",
                    "boundary": {
                        "input_kind": "adaptive_v2",
                        "resolution_manifest": str(resolution),
                        "open_exterior_gate_policy": "reviewed-adaptive-v2",
                    },
                    "bathymetry": {"netcdf": None},
                }
            ),
            encoding="utf-8",
        )
        readiness = check_case_readiness(case, root)
        assert readiness["grid_boundary_gate"]["passed"] is True
        assert (
            "upstream_open_exterior:open_exterior_contract_missing"
            in readiness["warnings"]
        )
        assert "boundary_resolution_manifest" in readiness["input_hashes"]


def main() -> int:
    tests = [
        test_strict_policy_still_requires_open_exterior_contract,
        test_strict_contract_accepts_existing_workspace_relative_evidence,
        test_reviewed_policy_accepts_exact_passing_adaptive_v2,
        test_reviewed_policy_cannot_waive_resolved_land_crossing,
        test_reviewed_policy_rejects_mismatched_arc_lineage,
        test_reviewed_policy_accepts_hash_identical_relocated_arc_lineage,
        test_gmsh_preflight_consumes_reviewed_policy,
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} reviewed boundary-gate tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
