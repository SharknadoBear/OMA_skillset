#!/usr/bin/env python3
"""Offline tests for standardized projects and open-exterior revalidation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.grid_project import (
    DEFAULT_MESHER_POLICY,
    init_project,
    promote,
    publish,
    validate,
)
from fvcom_grid_generation.open_exterior import sha256_file, validate_open_exterior_contract
from fvcom_grid_generation.quality_policy import public_policy_binding
from fvcom_grid_generation.sms_2dm import write_2dm


def write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def coverage_fixture(root: Path) -> tuple[dict, str]:
    whole = write(root / "coverage_map.png", "coverage map")
    zoom = write(root / "coverage_zoom.png", "coverage zoom")
    path = root / "coastline_source_coverage.json"
    coverage = {
        "schema_version": "fvcom_coastline_source_coverage_v1",
        "downstream_eligible": True,
        "coverage_factor_x": 3.0,
        "coverage_factor_y": 3.0,
        "model_bbox_centrally_contained": True,
        "region_bpoly_covered": True,
        "source_frame_dependency_length_m": 0.0,
        "source_frame_dependency_limit_m": 1.0,
        "physical_coastline_only_landfalls": True,
        "contract_path": str(path),
        "maps": {
            "whole_domain": {"path": str(whole), "sha256": sha256_file(whole)},
            "source_edge_zoom": {"path": str(zoom), "sha256": sha256_file(zoom)},
        },
    }
    path.write_text(json.dumps(coverage), encoding="utf-8")
    return coverage, sha256_file(path)


def contract_fixture(root: Path, *, report_only: bool = False) -> Path:
    map_path = write(root / "open_map.png", "map")
    decision_path = root / "decision.json"
    coverage, coverage_hash = coverage_fixture(root)
    source_hashes = {"region": "abc", "coastline": "def", "coastline_source_coverage": coverage_hash}
    decision = {
        "schema_version": "open_exterior_agent_decision_v1",
        "status": "pass",
        "decision_actor": {"kind": "codex_agent"},
        "inspected_map_sha256": sha256_file(map_path),
        "bound_source_hashes": source_hashes,
        "inspected_coastline_coverage_map_sha256": coverage["maps"]["whole_domain"]["sha256"],
        "inspected_coastline_coverage_zoom_sha256": coverage["maps"]["source_edge_zoom"]["sha256"],
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
        "coastline_source_coverage_required": True,
        "coastline_source_coverage": coverage,
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


def role_contract_fixture(root: Path, *, stale_component_map: bool = False) -> Path:
    map_path = write(root / "open_map.png", "whole map")
    component_map = write(root / "residual_map.png", "component map")
    coverage, coverage_hash = coverage_fixture(root)
    source_hashes = {"region": "abc", "coastline": "def", "coastline_source_coverage": coverage_hash}
    decision_path = root / "decision_v2.json"
    decision = {
        "schema_version": "open_exterior_agent_decision_v2",
        "status": "pass",
        "decision_actor": {"kind": "codex_agent"},
        "inspected_map_sha256": sha256_file(map_path),
        "bound_source_hashes": source_hashes,
        "inspected_coastline_coverage_map_sha256": coverage["maps"]["whole_domain"]["sha256"],
        "inspected_coastline_coverage_zoom_sha256": coverage["maps"]["source_edge_zoom"]["sha256"],
        "residual_roles": [{
            "segment_id": 0,
            "role": "solid_lagoon_closure",
            "component_map_sha256": sha256_file(component_map),
            "no_artificial_bar": True,
            "no_protected_feature_conflict": True,
        }],
    }
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    contract = {
        "schema_version": "fvcom_open_exterior_contract_v2",
        "report_only": False,
        "downstream_eligible": True,
        "hard_metrics": {
            "absolute_gate_pass": True,
            "fraction_gate_pass": True,
            "coverage_gate_pass": True,
            "all_independent_metric_gates_pass": True,
            "absolute_limit_m": 250.0,
            "absolute_residual_length_m": 0.0,
            "metric_subject": "unassigned_residual",
        },
        "obc_geometry": {
            "expected_count": 1,
            "delivered_count": 1,
            "simple_nonbranching": True,
            "nonendpoint_land_crossing_m": 0.0,
        },
        "coastline_source_coverage_required": True,
        "coastline_source_coverage": coverage,
        "residual_role_summary": {
            "pending_count": 0,
            "unassigned_residual_length_m": 0.0,
            "solid_lagoon_closure_count": 1,
            "secondary_tidal_obc_count": 0,
        },
        "residual_components": [{
            "segment_id": 0,
            "classification": "unintended_frame_clip",
            "assigned_role": "solid_lagoon_closure",
            "role_status": "accepted",
            "solid_role_geometry": {"eligible": True},
            "agent_geometry_confirmation": {"no_artificial_bar": True, "no_protected_feature_conflict": True},
            "forcing_eligibility": None,
        }],
        "component_maps": {"0": {"path": str(component_map), "sha256": ("0" * 64 if stale_component_map else sha256_file(component_map))}},
        "source_hashes": source_hashes,
        "map": {"path": str(map_path), "sha256": sha256_file(map_path)},
        "agent_decision": {"status": "pass", "path": str(decision_path), "sha256": sha256_file(decision_path)},
    }
    path = root / "contract_v2.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


def unsupported_v3_contract_fixture(root: Path) -> Path:
    path = role_contract_fixture(root)
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract["schema_version"] = "fvcom_open_exterior_contract_v3"
    path = root / "contract_v3.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


def reviewed_resolution_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    package = write(root / "boundary_resolution.gpkg", "package")
    loops = write(root / "model_boundary_loops.gpkg", "loops")
    path = root / "boundary_resolution_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "fvcom_boundary_resolution_manifest_v2",
                "profile": "adaptive-coastal-v2",
                "final_status": "pass",
                "failure_taxonomy": [],
                "inputs": {"model_boundary_loops_gpkg": str(loops)},
                "outputs": {"boundary_resolution_gpkg": str(package)},
                "qa": {
                    "resolved_domain_valid": True,
                    "open_arc_land_intersection_m": 0.0,
                    "open_arc_exterior_overlap_fraction": 1.0,
                    "protected_underresolved_passage_count": 0,
                    "maximum_edge_to_target_ratio": 1.1,
                    "p95_edge_to_target_ratio": 1.0,
                    "maximum_target_gradation": 0.15,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def mesh_fixture(path: Path, *, open_boundary: bool = True) -> Path:
    chains = [[1, 2]] if open_boundary else []
    return write_2dm(
        path,
        np.asarray(
            [[-75.2, 39.0], [-75.0, 39.0], [-75.0, 39.2], [-75.2, 39.2]],
            dtype=float,
        ),
        np.asarray([2.0, 4.0, 8.0, 6.0], dtype=float),
        np.asarray([[1, 2, 3], [1, 3, 4]], dtype=int),
        np.empty(0, dtype=int),
        mesh_name="project fixture",
        open_boundary_chains=chains,
        open_boundary_ids=([1] if chains else []),
    )


def companions(
    root: Path,
    *,
    benchmark_ready: bool = True,
    findings: list[str] | None = None,
) -> dict[str, Path]:
    findings = list(findings or [])
    quality = {
        "schema_version": "fvcom_mesh_quality_v3",
        "quality_policy": public_policy_binding(),
        "oceanmesh_quality": {"q_l3_sigma": 0.812345},
        "all_quality_findings": findings,
        "benchmark_grid_baseline_ready": benchmark_ready,
        "fvcom_ready": benchmark_ready,
        "accepted": benchmark_ready,
        "failure_taxonomy": findings if not benchmark_ready else [],
        "regional_refinement_debt": [],
    }
    values = {
        "mesh_quality": write(
            root / "08_audit" / "_work" / "quality.json",
            json.dumps(quality),
        ),
        "mesh_conditioning": write(root / "08_audit" / "_work" / "conditioning.json", "{}"),
        "boundary_nodes": write(root / "08_audit" / "_work" / "boundary.geojson", '{"type":"FeatureCollection","features":[]}'),
        "obc_remap_manifest": write(root / "08_audit" / "_work" / "obc.json", "{}"),
        "roundtrip_audit": write(root / "08_audit" / "_work" / "roundtrip.json", "{}"),
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
        missing_coverage = contract_fixture(base / "missing_coverage")
        missing_doc = json.loads(missing_coverage.read_text(encoding="utf-8"))
        missing_doc.pop("coastline_source_coverage", None)
        missing_doc.pop("coastline_source_coverage_required", None)
        missing_coverage.write_text(json.dumps(missing_doc), encoding="utf-8")
        missing_audit = validate_open_exterior_contract(missing_coverage)
        assert "coastline_source_coverage_missing" in missing_audit["failure_taxonomy"]
        stale_coverage = contract_fixture(base / "stale_coverage")
        stale_doc = json.loads(stale_coverage.read_text(encoding="utf-8"))
        Path(stale_doc["coastline_source_coverage"]["contract_path"]).write_text("stale", encoding="utf-8")
        stale_coverage_audit = validate_open_exterior_contract(stale_coverage)
        assert "coastline_source_coverage_hash_stale" in stale_coverage_audit["failure_taxonomy"]
        role_contract = role_contract_fixture(base / "role_evidence")
        role_audit = validate_open_exterior_contract(role_contract)
        assert role_audit["passed"], role_audit
        unsupported_v3 = unsupported_v3_contract_fixture(base / "unsupported_v3")
        unsupported_v3_audit = validate_open_exterior_contract(unsupported_v3)
        assert not unsupported_v3_audit["passed"]
        assert "open_exterior_contract_schema_unsupported" in unsupported_v3_audit["failure_taxonomy"]
        stale_role_contract = role_contract_fixture(base / "stale_role_evidence", stale_component_map=True)
        stale_audit = validate_open_exterior_contract(stale_role_contract)
        assert not stale_audit["passed"]
        assert "residual_boundary_component_map_stale" in stale_audit["failure_taxonomy"]
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
        source = mesh_fixture(
            project / "06_raw_mesh" / "_work" / "attempt1" / "raw_mesh.2dm"
        )
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
            failures=["node_valence_above_threshold"],
            open_exterior_source=contract,
            basemap_provider="offline",
        )
        assert (project / "final" / "fvcom_grid.2dm").is_file()
        assert not validate(project, require_submission_ready=True)["passed"]

        reviewed_project = base / "reviewed_boundary"
        init_project(reviewed_project, "reviewed_boundary")
        reviewed_resolution = reviewed_resolution_fixture(base / "reviewed_evidence")
        reviewed_status = publish(
            reviewed_project,
            mesh=None,
            companions={},
            fvcom_ready=False,
            submission_eligible=False,
            obc_status="pass",
            forcing_status="missing",
            failures=[],
            boundary_resolution_source=reviewed_resolution,
            boundary_gate_policy="reviewed-adaptive-v2",
            basemap_provider="offline",
        )
        assert reviewed_status["open_exterior_audit"]["passed"] is True
        assert str(base) not in json.dumps(reviewed_status["open_exterior_audit"])

        ready = base / "ready"
        current_contract = role_contract_fixture(base / "current_project_contract")
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
        raw_source = mesh_fixture(
            ready / "06_raw_mesh" / "_work" / "gmsh6" / "raw_mesh.2dm"
        )
        raw_candidate = gmsh6_candidate(raw_source)
        promote(
            ready,
            "06_raw_mesh",
            raw_source,
            "raw_mesh.2dm",
            generator_manifest=raw_candidate,
        )
        conditioned = mesh_fixture(
            ready / "07_conditioning" / "_work" / "mesh.2dm"
        )
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
            open_exterior_source=current_contract,
            basemap_provider="offline",
        )
        assert validate(ready, require_benchmark_ready=True)["passed"]
        assert validate(ready, require_submission_ready=True)["passed"]
        assert (ready / "08_audit" / "mesh_review_map.png").is_file()
        assert (ready / "final" / "mesh_review_map_manifest.json").is_file()

        rejected = base / "cleanroom"
        init_project(rejected, "cleanroom")
        control_mesh = mesh_fixture(
            rejected / "06_raw_mesh" / "_work" / "control" / "raw_mesh.2dm"
        )
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
