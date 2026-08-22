#!/usr/bin/env python3
"""Offline tests for standardized projects and open-exterior revalidation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.grid_project import init_project, promote, publish, validate
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


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        contract = contract_fixture(base / "evidence")
        assert validate_open_exterior_contract(contract)["passed"]
        report_only = contract_fixture(base / "diagnostic", report_only=True)
        assert not validate_open_exterior_contract(report_only)["passed"]

        project = base / "nonready"
        init_project(project, "nonready")
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
        source = write(project / "06_raw_mesh" / "_work" / "attempt1.2dm", "MESH2D\n")
        promote(project, "06_raw_mesh", source, "raw_mesh.2dm")
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
        raw = write(ready / "07_conditioning" / "_work" / "mesh.2dm", "MESH2D\n")
        promote(ready, "07_conditioning", raw, "conditioned_mesh.2dm")
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
    print("passed standardized project and open-exterior tests")


if __name__ == "__main__":
    main()
