#!/usr/bin/env python3
"""Offline tests for standardized projects and open-exterior revalidation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.grid_project import (
    DEFAULT_MESHER_POLICY,
    init_project,
    promote,
    publish,
    validate,
)
from fvcom_grid_generation.open_exterior import sha256_file, validate_open_exterior_contract


def write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def contract_fixture(root: Path, *, report_only: bool = False) -> Path:
    map_path = write(root / "open_map.png", "map")
    decision_path = root / "decision.json"
    source_hashes = {"region": "abc", "coastline": "def"}
    decision = {
        "schema_version": "open_exterior_agent_decision_v1",
        "status": "pass",
        "decision_actor": {"kind": "codex_agent"},
        "inspected_map_sha256": sha256_file(map_path),
        "bound_source_hashes": source_hashes,
    }
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    contract = {
        "schema_version": "fvcom_open_exterior_contract_v1",
        "report_only": report_only,
        "downstream_eligible": not report_only,
        "hard_metrics": {
            "absolute_gate_pass": True,
            "fraction_gate_pass": True,
            "coverage_gate_pass": True,
            "absolute_limit_m": 250.0,
        },
        "obc_geometry": {
            "expected_count": 1,
            "delivered_count": 1,
            "simple_nonbranching": True,
            "nonendpoint_land_crossing_m": 0.0,
        },
        "source_hashes": source_hashes,
        "map": {"path": str(map_path), "sha256": sha256_file(map_path)},
        "agent_decision": {
            "status": "pass",
            "path": str(decision_path),
            "sha256": sha256_file(decision_path),
        },
    }
    path = root / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


def companions(root: Path) -> dict[str, Path]:
    values = {
        "mesh_quality": write(root / "08_audit" / "_work" / "quality.json", "{}"),
        "mesh_conditioning": write(root / "08_audit" / "_work" / "conditioning.json", "{}"),
        "boundary_nodes": write(root / "08_audit" / "_work" / "boundary.geojson", '{"type":"FeatureCollection","features":[]}'),
        "obc_remap_manifest": write(root / "08_audit" / "_work" / "obc.json", "{}"),
        "roundtrip_audit": write(root / "08_audit" / "_work" / "roundtrip.json", "{}"),
        "mesh_review_map": write(root / "08_audit" / "_work" / "review.png", "png"),
    }
    return values


def gmsh6_candidate(raw_mesh: Path, *, candidate_id: str = "gmsh_frontal_delaunay_6") -> Path:
    attempt = raw_mesh.parent
    report = {
        "schema_version": "fvcom_portfolio_generator_report_v1",
        "backend": "gmsh" if candidate_id == "gmsh_frontal_delaunay_6" else "scipy",
        "algorithm": 6 if candidate_id == "gmsh_frontal_delaunay_6" else None,
        "algorithm_name": "Frontal-Delaunay" if candidate_id == "gmsh_frontal_delaunay_6" else "clean-room",
        "thread_count": 1,
        "random_seed": 1,
        "native_smoothing_steps": 8,
        "algorithm_fallback_enabled": False,
        "raw_stage": True,
        "common_conditioning_applied": False,
    }
    report_path = attempt / "generator_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    candidate = {
        "schema_version": "fvcom_mesher_candidate_manifest_v1",
        "candidate_id": candidate_id,
        "raw_stage": True,
        "common_conditioning_applied": False,
        "artifacts": {
            "sms_2dm": {"path": str(raw_mesh), "sha256": sha256_file(raw_mesh)},
            "generator_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        },
    }
    candidate_path = attempt / "candidate_manifest.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    return candidate_path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        guarded = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "run_fvcom_grid.py"),
                "--run-dir",
                str(base / "cleanroom_guard"),
                "--name",
                "cleanroom_guard",
                "--mode",
                "execute",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert guarded.returncode == 2
        assert "Gmsh Frontal-Delaunay algorithm 6" in guarded.stderr
        contract = contract_fixture(base / "evidence")
        assert validate_open_exterior_contract(contract)["passed"]
        report_only = contract_fixture(base / "diagnostic", report_only=True)
        assert not validate_open_exterior_contract(report_only)["passed"]

        project = base / "nonready"
        initialized = init_project(project, "nonready")
        assert initialized["mesher_policy"] == DEFAULT_MESHER_POLICY
        assert init_project(project, "nonready")["name"] == "nonready"
        status = publish(
            project,
            mesh=None,
            companions={},
            fvcom_ready=False,
            submission_eligible=False,
            obc_status="failed",
            forcing_status="unknown",
            failures=["boundary_failed"],
        )
        assert status["state"] == "failed_pre_mesh"
        source = write(project / "06_raw_mesh" / "_work" / "attempt1" / "raw_mesh.2dm", "MESH2D\n")
        candidate = gmsh6_candidate(source)
        try:
            promote(project, "06_raw_mesh", source, "raw_mesh.2dm")
        except ValueError as exc:
            assert "generator-manifest" in str(exc)
        else:
            raise AssertionError("raw promotion without Gmsh provenance must fail")
        promote(
            project,
            "06_raw_mesh",
            source,
            "raw_mesh.2dm",
            generator_manifest=candidate,
        )
        assert (project / "06_raw_mesh" / "raw_mesh_manifest.json").is_file()
        manifest_text = (project / "project_manifest.json").read_text(encoding="utf-8")
        assert str(project) not in manifest_text
        status = publish(
            project,
            mesh=project / "06_raw_mesh" / "raw_mesh.2dm",
            companions=companions(project),
            fvcom_ready=False,
            submission_eligible=False,
            obc_status="pass",
            forcing_status="incompatible",
            failures=["forcing_incompatible"],
            open_exterior_source=contract,
        )
        assert (project / "final" / "fvcom_grid.2dm").is_file()
        assert not validate(project, require_submission_ready=True)["passed"]

        ready = base / "ready"
        init_project(ready, "ready")
        publish(
            ready,
            mesh=None,
            companions={},
            fvcom_ready=False,
            submission_eligible=False,
            obc_status="pending",
            forcing_status="pending",
            failures=[],
        )
        raw_source = write(ready / "06_raw_mesh" / "_work" / "gmsh6" / "raw_mesh.2dm", "MESH2D\n")
        raw_candidate = gmsh6_candidate(raw_source)
        promote(
            ready,
            "06_raw_mesh",
            raw_source,
            "raw_mesh.2dm",
            generator_manifest=raw_candidate,
        )
        conditioned = write(ready / "07_conditioning" / "_work" / "mesh.2dm", "MESH2D\n")
        promote(ready, "07_conditioning", conditioned, "conditioned_mesh.2dm")
        publish(
            ready,
            mesh=ready / "07_conditioning" / "conditioned_mesh.2dm",
            companions=companions(ready),
            fvcom_ready=True,
            submission_eligible=True,
            obc_status="pass",
            forcing_status="compatible",
            failures=[],
            open_exterior_source=contract,
        )
        assert validate(ready, require_submission_ready=True)["passed"]

        rejected = base / "cleanroom"
        init_project(rejected, "cleanroom")
        control_mesh = write(rejected / "06_raw_mesh" / "_work" / "control" / "raw_mesh.2dm", "MESH2D\n")
        control_candidate = gmsh6_candidate(control_mesh, candidate_id="clean_room_raw")
        try:
            promote(
                rejected,
                "06_raw_mesh",
                control_mesh,
                "raw_mesh.2dm",
                generator_manifest=control_candidate,
            )
        except ValueError as exc:
            assert "operational Gmsh-6" in str(exc)
        else:
            raise AssertionError("clean-room control must not enter an operational project")
    print("passed standardized project and open-exterior tests")


if __name__ == "__main__":
    main()
