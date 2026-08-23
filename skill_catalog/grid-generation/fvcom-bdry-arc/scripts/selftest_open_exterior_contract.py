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
    print("passed 6 strict open-exterior policy tests")


if __name__ == "__main__":
    main()
