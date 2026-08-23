#!/usr/bin/env python3
"""Record a hash-bound Codex open-exterior decision and resume boundary QA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_bdry_arc.boundary_resolution import (  # noqa: E402
    boundary_resolution_config,
    build_boundary_resolution,
)


DECISION_FAILURES = {
    "open_exterior_agent_decision_required",
    "open_exterior_contract_not_downstream_eligible",
    "blocked_by_region_bpoly_feedback",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def _existing_adaptive_resolution(run_dir: Path, profile: str) -> dict | None:
    path = run_dir / "boundary_resolution" / "boundary_resolution_manifest.json"
    if not path.is_file():
        return None
    value = read_json(path)
    if value.get("profile") != profile:
        return None
    required_outputs = [
        value.get("outputs", {}).get("boundary_resolution_gpkg"),
        value.get("outputs", {}).get("boundary_resolution_nodes_geojson"),
    ]
    if not all(item and Path(item).is_file() for item in required_outputs):
        return None
    return value


def finalize(manifest_path: Path, decision: str, rationale: str, *, resume_adaptive: bool) -> dict:
    manifest = read_json(manifest_path)
    contract_path = Path(manifest.get("outputs", {}).get("open_exterior_contract", ""))
    map_path = Path(manifest.get("outputs", {}).get("open_exterior_review_map", ""))
    decision_path = Path(manifest.get("outputs", {}).get("open_exterior_agent_decision", ""))
    if not contract_path.is_file() or not map_path.is_file() or not decision_path.parent.is_dir():
        raise ValueError("boundary manifest does not resolve a complete open-exterior evidence package")
    contract = read_json(contract_path)
    if contract.get("schema_version") != "fvcom_open_exterior_contract_v1":
        raise ValueError("unsupported open-exterior contract")
    metrics = contract.get("hard_metrics", {})
    hard_pass = bool(
        metrics.get("absolute_gate_pass")
        and metrics.get("fraction_gate_pass")
        and metrics.get("coverage_gate_pass")
        and metrics.get("all_independent_metric_gates_pass")
    )
    if decision == "pass" and (not hard_pass or contract.get("report_only")):
        raise ValueError("Codex cannot pass failed hard metrics or a report-only package")
    if not rationale.strip():
        raise ValueError("a concise visual rationale is required")

    assessed_hash = sha256_file(contract_path)
    decision_doc = {
        "schema_version": "open_exterior_agent_decision_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": decision,
        "decision_actor": {"kind": "codex_agent"},
        "assessed_contract_sha256": assessed_hash,
        "inspected_map_sha256": sha256_file(map_path),
        "bound_source_hashes": contract.get("source_hashes", {}),
        "rationale": rationale.strip(),
        "confirmation": {
            "whole_domain_map_inspected": True,
            "no_artificial_frame_supported_strip": decision == "pass",
        },
    }
    eligible = bool(decision == "pass" and hard_pass and not contract.get("report_only"))
    # Adaptive construction can be expensive.  Finish or recover it before
    # publishing the decision/contract so interruption never leaves a passing
    # child contract under a needs-review parent manifest.
    resolution = None
    if eligible and resume_adaptive:
        profile = str(manifest.get("settings", {}).get("boundary_resolution_profile", "legacy"))
        if profile in {"adaptive-coastal-v1", "adaptive-coastal-v2"}:
            resolution = _existing_adaptive_resolution(manifest_path.parent, profile)
            if resolution is None:
                outputs = manifest.get("outputs", {})
                resolution = build_boundary_resolution(
                    outputs["model_boundary_loops_gpkg"],
                    outputs["model_boundary_loop_manifest"],
                    manifest["inputs"]["region_bpoly_json"],
                    manifest["inputs"]["coastline_gpkg"],
                    manifest_path.parent / "boundary_resolution",
                    str(manifest.get("name", "fvcom_boundary")),
                    boundary_resolution_config(profile),
                )

    atomic_json(decision_path, decision_doc)
    decision_hash = sha256_file(decision_path)
    contract["agent_decision"] = {
        "required": True,
        "status": decision,
        "path": str(decision_path),
        "sha256": decision_hash,
        "assessed_contract_sha256": assessed_hash,
        "inspected_map_sha256": decision_doc["inspected_map_sha256"],
    }
    contract["downstream_eligible"] = eligible
    contract["final_status"] = "pass" if eligible else "needs_review"
    failures = [f for f in contract.get("failure_taxonomy", []) if f not in DECISION_FAILURES]
    if not eligible and "open_exterior_agent_decision_rejected" not in failures:
        failures.append("open_exterior_agent_decision_rejected")
    contract["failure_taxonomy"] = failures
    atomic_json(contract_path, contract)

    manifest["open_exterior_contract"] = contract
    if isinstance(manifest.get("region_bpoly_arc_feedback"), dict):
        manifest["region_bpoly_arc_feedback"]["open_exterior_contract"] = contract
    manifest["failure_taxonomy"] = [
        f for f in manifest.get("failure_taxonomy", []) if f not in DECISION_FAILURES
    ]
    if not eligible:
        if "open_exterior_agent_decision_rejected" not in manifest["failure_taxonomy"]:
            manifest["failure_taxonomy"].append("open_exterior_agent_decision_rejected")
        manifest["final_status"] = "needs_review"
    elif resolution is not None:
        manifest["boundary_resolution"] = resolution
        manifest["outputs"].update(resolution.get("outputs", {}))
        if resolution.get("final_status") != "pass":
            manifest["failure_taxonomy"].append("adaptive_boundary_resolution_needs_review")
    if eligible and not manifest.get("failure_taxonomy"):
        manifest["final_status"] = "pass"
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdry-arc-manifest", required=True, type=Path)
    parser.add_argument("--decision", required=True, choices=("pass", "needs_review"))
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--resume-adaptive", action="store_true")
    args = parser.parse_args()
    result = finalize(
        args.bdry_arc_manifest.resolve(),
        args.decision,
        args.rationale,
        resume_adaptive=args.resume_adaptive,
    )
    print(json.dumps({
        "final_status": result.get("final_status"),
        "downstream_eligible": result.get("open_exterior_contract", {}).get("downstream_eligible"),
    }, indent=2))
    return 0 if result.get("open_exterior_contract", {}).get("downstream_eligible") else 2


if __name__ == "__main__":
    raise SystemExit(main())
