#!/usr/bin/env python3
"""Finalize a hash-bound Codex decision for ambiguous RegionBPoly residual cuts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROUTES = {
    "adjust_bpoly",
    "retain_for_role_classification",
    "invalid_geometry",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _feedback_core(feedback: dict[str, Any]) -> dict[str, Any]:
    component_maps = feedback.get("outputs", {}).get("residual_component_maps", {})
    return {
        "boundary_completeness": feedback.get("boundary_completeness"),
        "source_hashes": feedback.get("input_sha256"),
        "whole_map_sha256": (
            feedback.get("open_exterior_contract", {}).get("map", {}).get("sha256")
            or feedback.get("outputs", {}).get("whole_map_sha256")
        ),
        "component_map_sha256": {
            str(key): value.get("sha256") for key, value in component_maps.items()
        },
    }


def _parse_routes(values: list[str]) -> dict[int, str]:
    routes: dict[int, str] = {}
    for value in values:
        try:
            raw_id, route = value.split("=", 1)
            segment_id = int(raw_id)
        except Exception as exc:
            raise ValueError("--route must use SEGMENT_ID=ROUTE") from exc
        if route not in ROUTES:
            raise ValueError(f"Unsupported boundary-completeness route: {route}")
        if segment_id in routes:
            raise ValueError(f"Duplicate route for segment {segment_id}")
        routes[segment_id] = route
    return routes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdry-arc-manifest", required=True)
    parser.add_argument("--route", action="append", default=[], metavar="SEGMENT_ID=ROUTE")
    parser.add_argument("--rationale", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.bdry_arc_manifest).resolve()
    manifest = _read(manifest_path)
    feedback_path = Path(manifest.get("outputs", {}).get("feedback_json", "")).resolve()
    if not feedback_path.is_file():
        raise FileNotFoundError("Boundary manifest does not reference feedback JSON")
    feedback = _read(feedback_path)
    if feedback.get("schema_version") != "region_bpoly_arc_feedback_v2":
        raise ValueError("Boundary-completeness decisions require region_bpoly_arc_feedback_v2")
    completeness = feedback.get("boundary_completeness") or {}
    expected_ids = {int(value) for value in completeness.get("agent_decision_segment_ids", [])}
    routes = _parse_routes(args.route)
    if set(routes) != expected_ids:
        raise ValueError(
            f"Routes must cover exactly the pending segments: expected {sorted(expected_ids)}, got {sorted(routes)}"
        )

    decision_path = Path(feedback.get("outputs", {}).get("boundary_completeness_decision", "")).resolve()
    if not decision_path.is_file():
        raise FileNotFoundError("Pending boundary-completeness decision is missing")
    pending = _read(decision_path)
    expected_core_hash = _canonical_sha256(_feedback_core(feedback))
    if pending.get("assessed_feedback_core_sha256") != expected_core_hash:
        raise ValueError("Boundary-completeness decision template is stale")

    component_maps = feedback.get("outputs", {}).get("residual_component_maps", {})
    records = {int(item["segment_id"]): item for item in feedback.get("frame_clip_segments", [])}
    component_decisions = []
    for segment_id in sorted(routes):
        item = records.get(segment_id)
        if item is None:
            raise ValueError(f"Residual segment is missing: {segment_id}")
        map_record = component_maps.get(str(segment_id), {})
        map_path = Path(map_record.get("path", ""))
        if not map_path.is_file() or _sha256(map_path) != map_record.get("sha256"):
            raise ValueError(f"Residual component map is missing or stale: {segment_id}")
        route = routes[segment_id]
        item.setdefault("boundary_completeness", {}).update(
            {
                "status": "resolved",
                "route": route,
                "decision_actor": {"kind": "codex_agent"},
            }
        )
        component_decisions.append(
            {
                "segment_id": segment_id,
                "route": route,
                "component_map_sha256": map_record.get("sha256"),
                "rationale": args.rationale,
            }
        )

    selected_routes = set(routes.values())
    if "invalid_geometry" in selected_routes:
        next_status = "input_needs_review"
        completeness_status = "invalid_geometry"
        failure = "region_bpoly_boundary_geometry_invalid"
    elif "adjust_bpoly" in selected_routes:
        next_status = "adjust_bpoly"
        completeness_status = "adjust_bpoly"
        failure = "region_bpoly_boundary_truncation_detected"
        adjust_ids = {segment_id for segment_id, route in routes.items() if route == "adjust_bpoly"}
        feedback["candidate_recommendations"] = [
            candidate
            for candidate in feedback.get("candidate_recommendations", [])
            if adjust_ids.intersection(int(value) for value in candidate.get("source_segment_ids", []))
        ]
        if not feedback["candidate_recommendations"]:
            next_status = "input_needs_review"
            failure = "region_bpoly_boundary_adjustment_candidate_missing"
    else:
        next_status = "assign_boundary_roles" if feedback.get("frame_clip_segments") else "pass"
        completeness_status = "pass"
        failure = None

    decision = {
        "schema_version": "region_bpoly_boundary_completeness_decision_v1",
        "status": "accepted",
        "decision_actor": {"kind": "codex_agent"},
        "assessed_feedback_core_sha256": expected_core_hash,
        "bound_source_hashes": feedback.get("input_sha256"),
        "inspected_map_sha256": feedback.get("open_exterior_contract", {}).get("map", {}).get("sha256"),
        "component_decisions": component_decisions,
        "rationale": args.rationale,
    }
    _atomic_json(decision_path, decision)
    decision_hash = _sha256(decision_path)

    completeness.update(
        {
            "status": completeness_status,
            "decision_required": True,
            "decision_status": "accepted",
            "decision_path": str(decision_path),
            "decision_sha256": decision_hash,
        }
    )
    feedback["boundary_completeness"] = completeness
    feedback["status"] = next_status
    failures = [
        value
        for value in feedback.get("failure_taxonomy", [])
        if value != "region_bpoly_boundary_completeness_decision_required"
    ]
    if failure and failure not in failures:
        failures.append(failure)
    if not failure and next_status == "assign_boundary_roles" and "residual_boundary_role_decision_required" not in failures:
        failures.append("residual_boundary_role_decision_required")
    feedback["failure_taxonomy"] = failures
    contract = feedback.get("open_exterior_contract") or {}
    contract_records = {
        int(item.get("segment_id", -1)): item
        for item in contract.get("residual_components", [])
    }
    for segment_id, route in routes.items():
        if segment_id in contract_records:
            contract_records[segment_id].setdefault("boundary_completeness", {}).update(
                {
                    "status": "resolved",
                    "route": route,
                    "decision_actor": {"kind": "codex_agent"},
                }
            )
    contract["boundary_completeness"] = completeness
    contract["failure_taxonomy"] = [
        value
        for value in contract.get("failure_taxonomy", [])
        if value != "region_bpoly_boundary_completeness_decision_required"
    ]
    if failure and failure not in contract["failure_taxonomy"]:
        contract["failure_taxonomy"].append(failure)
    feedback["open_exterior_contract"] = contract
    _atomic_json(Path(feedback["outputs"]["open_exterior_contract"]), contract)
    _atomic_json(feedback_path, feedback)

    manifest["region_bpoly_arc_feedback"] = feedback
    manifest["open_exterior_contract"] = contract
    manifest.setdefault("outputs", {})["boundary_completeness_decision"] = str(decision_path)
    manifest["outputs"]["boundary_completeness_decision_sha256"] = decision_hash
    _atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": next_status,
                "decision": str(decision_path),
                "decision_sha256": decision_hash,
                "feedback_json": str(feedback_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
