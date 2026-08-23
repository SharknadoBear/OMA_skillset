#!/usr/bin/env python3
"""Regression tests for independent open-exterior gates and policy defaults."""

from pathlib import Path
import json
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_bdry_arc.feedback import evaluate_open_exterior_metrics
from fvcom_bdry_arc.workflow import BdryArcConfig
import finalize_open_exterior_decision as finalizer


def main() -> None:
    # Tampa-style northern/western bars fail every metric.
    tampa = evaluate_open_exterior_metrics(111_193.677, 903_200.0, 916_100.0, 250.0)
    assert not tampa["passed"]
    assert not tampa["length_gate"] and not tampa["fraction_gate"] and not tampa["coverage_gate"]

    # A small absolute residual cannot waive fractional/coverage gates.
    compact = evaluate_open_exterior_metrics(200.0, 10_000.0, 10_100.0, 250.0)
    assert compact["length_gate"]
    assert not compact["fraction_gate"]
    assert not compact["coverage_gate"]
    assert not compact["passed"]

    clean = evaluate_open_exterior_metrics(0.0, 100_000.0, 110_000.0, 250.0)
    assert clean["passed"]
    defaults = BdryArcConfig()
    assert defaults.frame_clip_policy == "reject-unintended"
    assert defaults.residual_boundary_policy == "solid-default"
    assert defaults.obc_placement_policy == "offshore-first"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        evidence = root / "fb"
        evidence.mkdir()
        review_map = evidence / "feedback_map.png"
        review_map.write_bytes(b"map")
        decision_path = evidence / "open_exterior_agent_decision.json"
        decision_path.write_text("{}", encoding="utf-8")
        contract_path = evidence / "open_exterior_contract.json"
        contract_path.write_text(json.dumps({
            "schema_version": "fvcom_open_exterior_contract_v1",
            "report_only": False,
            "hard_metrics": {
                "absolute_gate_pass": True,
                "fraction_gate_pass": True,
                "coverage_gate_pass": True,
                "all_independent_metric_gates_pass": True,
            },
            "source_hashes": {"region": "abc"},
            "map": {"path": str(review_map)},
            "failure_taxonomy": ["open_exterior_agent_decision_required"],
        }), encoding="utf-8")
        manifest_path = root / "bdry_arc_manifest.json"
        manifest_path.write_text(json.dumps({
            "final_status": "needs_review",
            "failure_taxonomy": ["open_exterior_agent_decision_required"],
            "settings": {"boundary_resolution_profile": "legacy"},
            "outputs": {
                "open_exterior_contract": str(contract_path),
                "open_exterior_review_map": str(review_map),
                "open_exterior_agent_decision": str(decision_path),
            },
            "region_bpoly_arc_feedback": {},
        }), encoding="utf-8")
        result = finalizer.finalize(manifest_path, "pass", "Inspected full map; no frame bar remains.", resume_adaptive=False)
        assert result["open_exterior_contract"]["downstream_eligible"] is True
        assert result["final_status"] == "pass"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        evidence = root / "fb"
        evidence.mkdir()
        review_map = evidence / "feedback_map.png"
        review_map.write_bytes(b"map")
        decision_path = evidence / "open_exterior_agent_decision.json"
        decision_path.write_text('{"status":"pending"}', encoding="utf-8")
        contract_path = evidence / "open_exterior_contract.json"
        contract_path.write_text(json.dumps({
            "schema_version": "fvcom_open_exterior_contract_v1",
            "report_only": False,
            "hard_metrics": {
                "absolute_gate_pass": True,
                "fraction_gate_pass": True,
                "coverage_gate_pass": True,
                "all_independent_metric_gates_pass": True,
            },
            "source_hashes": {},
            "failure_taxonomy": ["open_exterior_agent_decision_required"],
        }), encoding="utf-8")
        manifest_path = root / "bdry_arc_manifest.json"
        manifest_path.write_text(json.dumps({
            "name": "atomic",
            "final_status": "needs_review",
            "failure_taxonomy": ["open_exterior_agent_decision_required"],
            "settings": {"boundary_resolution_profile": "adaptive-coastal-v2"},
            "inputs": {"region_bpoly_json": "region", "coastline_gpkg": "coast"},
            "outputs": {
                "open_exterior_contract": str(contract_path),
                "open_exterior_review_map": str(review_map),
                "open_exterior_agent_decision": str(decision_path),
                "model_boundary_loops_gpkg": "loops",
                "model_boundary_loop_manifest": "loop_manifest",
            },
        }), encoding="utf-8")
        before = (contract_path.read_bytes(), decision_path.read_bytes(), manifest_path.read_bytes())
        original = finalizer.build_boundary_resolution
        finalizer.build_boundary_resolution = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic interruption"))
        try:
            try:
                finalizer.finalize(manifest_path, "pass", "Inspected map.", resume_adaptive=True)
            except RuntimeError as exc:
                assert "synthetic interruption" in str(exc)
            else:
                raise AssertionError("synthetic interruption was not propagated")
        finally:
            finalizer.build_boundary_resolution = original
        assert before == (contract_path.read_bytes(), decision_path.read_bytes(), manifest_path.read_bytes())

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        evidence = root / "fb"
        evidence.mkdir()
        review_map = evidence / "feedback_map.png"
        component_map = evidence / "residual_component_000_role_map.png"
        review_map.write_bytes(b"whole map")
        component_map.write_bytes(b"component map")
        decision_path = evidence / "open_exterior_agent_decision.json"
        decision_path.write_text('{"status":"pending"}', encoding="utf-8")
        contract_path = evidence / "open_exterior_contract.json"
        contract_path.write_text(json.dumps({
            "schema_version": "fvcom_open_exterior_contract_v2",
            "report_only": False,
            "downstream_eligible": False,
            "hard_metrics": {
                "absolute_residual_length_m": 831.118,
                "absolute_limit_m": 250.0,
                "absolute_gate_pass": False,
                "residual_fraction": 0.0008,
                "fraction_limit": 0.001,
                "fraction_gate_pass": True,
                "coastline_plus_obc_exterior_coverage": 0.9992,
                "coverage_minimum": 0.999,
                "coverage_gate_pass": True,
                "all_independent_metric_gates_pass": False,
                "metric_subject": "unassigned_residual",
            },
            "raw_residual_metrics": {"absolute_residual_length_m": 831.118},
            "boundary_lengths": {"outer_boundary_length_m": 1_100_000.0, "landward_boundary_length_m": 1_000_000.0},
            "obc_geometry": {"expected_count": 1, "delivered_count": 1, "simple_nonbranching": True, "nonendpoint_land_crossing_m": 0.0},
            "residual_components": [{
                "segment_id": 0,
                "classification": "unintended_frame_clip",
                "length_m": 831.118,
                "role_status": "pending",
                "assigned_role": None,
                "solid_role_geometry": {"eligible": True},
            }],
            "component_maps": {"0": {"path": str(component_map), "sha256": finalizer.sha256_file(component_map)}},
            "source_hashes": {"region": "abc"},
            "map": {"path": str(review_map), "sha256": finalizer.sha256_file(review_map)},
            "failure_taxonomy": ["residual_boundary_role_decision_required", "open_exterior_agent_decision_required"],
        }), encoding="utf-8")
        manifest_path = root / "bdry_arc_manifest.json"
        manifest_path.write_text(json.dumps({
            "final_status": "needs_review",
            "failure_taxonomy": ["residual_boundary_role_decision_required", "blocked_by_region_bpoly_feedback"],
            "settings": {"boundary_resolution_profile": "legacy"},
            "outputs": {
                "open_exterior_contract": str(contract_path),
                "open_exterior_review_map": str(review_map),
                "open_exterior_agent_decision": str(decision_path),
            },
            "region_bpoly_arc_feedback": {"failure_taxonomy": ["residual_boundary_role_decision_required"], "outputs": {}},
        }), encoding="utf-8")
        result = finalizer.finalize(
            manifest_path,
            "pass",
            "Inspected whole and component maps; the lagoon closure is simple and no artificial bar remains.",
            resume_adaptive=False,
        )
        resolved = result["open_exterior_contract"]
        assert resolved["schema_version"] == "fvcom_open_exterior_contract_v2"
        assert resolved["raw_residual_metrics"]["absolute_residual_length_m"] == 831.118
        assert resolved["hard_metrics"]["absolute_residual_length_m"] == 0.0
        assert resolved["residual_components"][0]["assigned_role"] == "solid_lagoon_closure"
        assert resolved["downstream_eligible"] is True

        pending_hash = finalizer.sha256_file(contract_path)
        station_screen = evidence / "station_screen.json"
        station_screen.write_text(json.dumps({
            "schema_version": "noaa_coops_tidal_station_screen_v1",
            "source_contract_sha256": pending_hash,
            "eligible_station_count": 1,
            "components": [{"segment_id": 0, "candidates": [{"station_id": "8570283", "name": "Ocean City Inlet", "distance_km": 11.5, "eligible_for_residual_obc": True}]}],
        }), encoding="utf-8")
        # The requested count is already occupied by the Atlantic OBC, so a
        # nearby station cannot silently create a second opening.
        try:
            finalizer.finalize(
                manifest_path,
                "pass",
                "Inspected maps.",
                resume_adaptive=False,
                residual_roles={0: "secondary_tidal_obc"},
                station_screen_path=station_screen,
            )
        except ValueError as exc:
            assert "Requested OBC count" in str(exc) or "stale" in str(exc)
        else:
            raise AssertionError("secondary OBC must respect the requested count")
    print("passed strict and solid-default open-exterior policy tests")


if __name__ == "__main__":
    main()
