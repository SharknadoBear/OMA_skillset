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
    "model_boundary_loop_needs_review",
    "residual_boundary_role_decision_required",
    "residual_boundary_role_pending",
    "unintended_frame_clip_nontrivial",
    "gshhs_coastline_incomplete_on_landward_boundary",
}

RESIDUAL_ROLES = {
    "solid_lagoon_closure",
    "secondary_tidal_obc",
    "invalid_geometry",
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


def _load_station_screen(path: Path | None, assessed_contract_sha256: str) -> tuple[dict | None, dict]:
    if path is None:
        return None, {"status": "not_run", "path": None, "sha256": None}
    if not path.is_file():
        raise ValueError(f"NOAA CO-OPS station screen does not exist: {path}")
    screen = read_json(path)
    if screen.get("schema_version") != "noaa_coops_tidal_station_screen_v1":
        raise ValueError("unsupported NOAA CO-OPS station-screen schema")
    if screen.get("source_contract_sha256") != assessed_contract_sha256:
        raise ValueError("NOAA CO-OPS station screen is stale for this open-exterior contract")
    return screen, {
        "status": "pass",
        "path": str(path),
        "sha256": sha256_file(path),
        "eligible_station_count": int(screen.get("eligible_station_count", 0)),
    }


def _eligible_station_for_segment(screen: dict | None, segment_id: int) -> dict | None:
    if not screen:
        return None
    component = next(
        (item for item in screen.get("components", []) if int(item.get("segment_id", -1)) == segment_id),
        None,
    )
    if not component:
        return None
    return next(
        (item for item in component.get("candidates", []) if item.get("eligible_for_residual_obc") is True),
        None,
    )


def _apply_residual_roles(
    contract: dict,
    *,
    assessed_contract_sha256: str,
    decision: str,
    requested_roles: dict[int, str],
    station_screen_path: Path | None,
) -> bool:
    screen, screen_binding = _load_station_screen(station_screen_path, assessed_contract_sha256)
    contract["station_screen"] = screen_binding
    components = list(contract.get("residual_components") or [])
    expected_obc = int(contract.get("obc_geometry", {}).get("expected_count", 0) or 0)
    source_delivered_obc = int(contract.get("obc_geometry", {}).get("delivered_count", 0) or 0)
    assigned_solid = 0.0
    assigned_secondary = 0.0
    pending = 0
    invalid = 0
    secondary_count = 0
    for component in components:
        if component.get("classification") == "intentional_open_boundary":
            continue
        segment_id = int(component.get("segment_id", -1))
        role = requested_roles.get(segment_id, "solid_lagoon_closure")
        if role not in RESIDUAL_ROLES:
            raise ValueError(f"Unsupported residual role for segment {segment_id}: {role}")
        geometry = component.get("solid_role_geometry", {})
        if role == "solid_lagoon_closure" and geometry.get("eligible") is not True:
            if decision == "pass":
                raise ValueError(f"Residual segment {segment_id} is not geometrically eligible for a solid closure")
            pending += 1
            continue
        station = None
        if role == "secondary_tidal_obc":
            station = _eligible_station_for_segment(screen, segment_id)
            if station is None and decision == "pass":
                raise ValueError(f"Residual segment {segment_id} has no eligible, hydraulically connected NOAA CO-OPS station")
            if source_delivered_obc + secondary_count >= expected_obc and decision == "pass":
                raise ValueError(
                    "Requested OBC count does not permit another boundary; a nearby gauge is eligibility evidence only"
                )
            secondary_count += 1
            assigned_secondary += float(component.get("length_m", 0.0) or 0.0)
        elif role == "invalid_geometry":
            invalid += 1
        else:
            assigned_solid += float(component.get("length_m", 0.0) or 0.0)
        component["assigned_role"] = role
        component["role_status"] = "accepted" if decision == "pass" and role != "invalid_geometry" else "needs_review"
        component["agent_geometry_confirmation"] = {
            "no_artificial_bar": bool(decision == "pass" and role != "invalid_geometry"),
            "no_protected_feature_conflict": bool(decision == "pass" and role != "invalid_geometry"),
        }
        component["forcing_eligibility"] = (
            {
                "provider": "NOAA CO-OPS",
                "station_id": station.get("station_id"),
                "station_name": station.get("name"),
                "distance_km": station.get("distance_km"),
                "eligibility_only": True,
            }
            if station
            else None
        )
    unassigned_length = float(
        sum(
            float(item.get("length_m", 0.0) or 0.0)
            for item in components
            if item.get("classification") != "intentional_open_boundary"
            and item.get("role_status") not in {"accepted", "needs_review"}
        )
    )
    lengths = contract.get("boundary_lengths", {})
    landward_length = max(float(lengths.get("landward_boundary_length_m", 1.0) or 1.0), 1.0)
    outer_length = max(float(lengths.get("outer_boundary_length_m", 1.0) or 1.0), 1.0)
    tolerance = float(contract.get("hard_metrics", {}).get("absolute_limit_m", 250.0) or 250.0)
    fraction = unassigned_length / landward_length
    coverage = max(0.0, min(1.0, 1.0 - unassigned_length / outer_length))
    hard = contract.setdefault("hard_metrics", {})
    hard.update({
        "absolute_residual_length_m": unassigned_length,
        "absolute_gate_pass": unassigned_length <= tolerance,
        "residual_fraction": fraction,
        "fraction_gate_pass": fraction <= float(hard.get("fraction_limit", 0.001)),
        "coastline_plus_obc_exterior_coverage": coverage,
        "coverage_gate_pass": coverage >= float(hard.get("coverage_minimum", 0.999)),
        "metric_subject": "unassigned_residual",
    })
    hard["all_independent_metric_gates_pass"] = bool(
        hard["absolute_gate_pass"] and hard["fraction_gate_pass"] and hard["coverage_gate_pass"]
    )
    contract["residual_role_summary"] = {
        "pending_count": pending,
        "solid_lagoon_closure_count": int(sum(item.get("assigned_role") == "solid_lagoon_closure" for item in components)),
        "secondary_tidal_obc_count": secondary_count,
        "invalid_geometry_count": invalid,
        "unassigned_residual_length_m": unassigned_length,
        "assigned_solid_length_m": assigned_solid,
        "assigned_secondary_obc_length_m": assigned_secondary,
    }
    contract["obc_geometry"]["source_delivered_count"] = source_delivered_obc
    contract["obc_geometry"]["delivered_count"] = source_delivered_obc + secondary_count
    return bool(
        decision == "pass"
        and pending == 0
        and invalid == 0
        and hard["all_independent_metric_gates_pass"]
        and contract["obc_geometry"]["delivered_count"] == expected_obc
    )


def _parse_roles(values: list[str]) -> dict[int, str]:
    roles: dict[int, str] = {}
    for value in values:
        try:
            segment, role = value.split("=", 1)
            segment_id = int(segment)
        except Exception as exc:
            raise ValueError("--residual-role must use SEGMENT_ID=ROLE") from exc
        if segment_id in roles:
            raise ValueError(f"Duplicate residual role for segment {segment_id}")
        roles[segment_id] = role
    return roles


def _verify_component_maps(contract: dict) -> None:
    maps = contract.get("component_maps", {})
    for component in contract.get("residual_components", []):
        if component.get("classification") == "intentional_open_boundary":
            continue
        segment_id = str(component.get("segment_id"))
        record = maps.get(segment_id) or {}
        path = Path(record.get("path", ""))
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            raise ValueError(f"Residual component {segment_id} map is missing or stale")


def _clear_loop_role_failures(manifest: dict) -> None:
    loop_path = Path(manifest.get("outputs", {}).get("model_boundary_loop_manifest", ""))
    if not loop_path.is_file():
        return
    loop = read_json(loop_path)
    loop["failure_taxonomy"] = [
        item for item in loop.get("failure_taxonomy", []) if item not in DECISION_FAILURES
    ]
    if not loop["failure_taxonomy"]:
        loop["final_status"] = "pass"
    atomic_json(loop_path, loop)
    if isinstance(manifest.get("model_boundary_loops"), dict):
        manifest["model_boundary_loops"]["final_status"] = loop.get("final_status")
        manifest["model_boundary_loops"]["failure_taxonomy"] = loop.get("failure_taxonomy", [])


def _bind_resolution_contract(resolution: dict, contract_path: Path) -> None:
    if not resolution:
        return
    contract = read_json(contract_path)
    solid_components = [
        {
            "segment_id": int(item.get("segment_id", -1)),
            "role": item.get("assigned_role"),
            "length_m": float(item.get("length_m", 0.0) or 0.0),
            "geometry_lonlat": item.get("geometry_lonlat"),
        }
        for item in contract.get("residual_components", [])
        if item.get("assigned_role") == "solid_lagoon_closure"
    ]
    resolution["open_exterior_contract_binding"] = {
        "schema_version": contract.get("schema_version"),
        "path": str(contract_path),
        "sha256": sha256_file(contract_path),
        "boundary_role_policy": "solid-default",
    }
    resolution["solid_boundary_roles"] = {
        "classification": "fixed_landward_chain",
        "component_count": len(solid_components),
        "total_length_m": float(sum(item["length_m"] for item in solid_components)),
        "components": solid_components,
    }
    resolution.setdefault("qa", {})["solid_residual_boundary_component_count"] = len(solid_components)
    resolution["qa"]["solid_residual_boundary_length_m"] = float(
        sum(item["length_m"] for item in solid_components)
    )
    resolution.setdefault("outputs", {})["open_exterior_contract"] = str(contract_path)
    output = Path(resolution.get("outputs", {}).get("boundary_resolution_manifest", ""))
    if output.name and output.parent.is_dir():
        atomic_json(output, resolution)


def finalize(
    manifest_path: Path,
    decision: str,
    rationale: str,
    *,
    resume_adaptive: bool,
    residual_roles: dict[int, str] | None = None,
    station_screen_path: Path | None = None,
) -> dict:
    manifest = read_json(manifest_path)
    contract_path = Path(manifest.get("outputs", {}).get("open_exterior_contract", ""))
    map_path = Path(manifest.get("outputs", {}).get("open_exterior_review_map", ""))
    decision_path = Path(manifest.get("outputs", {}).get("open_exterior_agent_decision", ""))
    if not contract_path.is_file() or not map_path.is_file() or not decision_path.parent.is_dir():
        raise ValueError("boundary manifest does not resolve a complete open-exterior evidence package")
    contract = read_json(contract_path)
    schema = contract.get("schema_version")
    if schema not in {"fvcom_open_exterior_contract_v1", "fvcom_open_exterior_contract_v2"}:
        raise ValueError("unsupported open-exterior contract")
    assessed_hash = sha256_file(contract_path)
    role_pass = True
    if schema == "fvcom_open_exterior_contract_v2":
        if decision == "pass":
            _verify_component_maps(contract)
        role_pass = _apply_residual_roles(
            contract,
            assessed_contract_sha256=assessed_hash,
            decision=decision,
            requested_roles=dict(residual_roles or {}),
            station_screen_path=station_screen_path,
        )
    metrics = contract.get("hard_metrics", {})
    hard_pass = bool(
        metrics.get("absolute_gate_pass")
        and metrics.get("fraction_gate_pass")
        and metrics.get("coverage_gate_pass")
        and metrics.get("all_independent_metric_gates_pass")
    )
    if decision == "pass" and (not hard_pass or not role_pass or contract.get("report_only")):
        raise ValueError("Codex cannot pass failed hard metrics or a report-only package")
    if not rationale.strip():
        raise ValueError("a concise visual rationale is required")

    decision_doc = {
        "schema_version": "open_exterior_agent_decision_v2" if schema.endswith("v2") else "open_exterior_agent_decision_v1",
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
        "residual_roles": [
            {
                "segment_id": int(item.get("segment_id", -1)),
                "role": item.get("assigned_role"),
                "component_map_sha256": (contract.get("component_maps", {}).get(str(item.get("segment_id")), {}) or {}).get("sha256"),
                "no_artificial_bar": bool(item.get("agent_geometry_confirmation", {}).get("no_artificial_bar", False)),
                "no_protected_feature_conflict": bool(item.get("agent_geometry_confirmation", {}).get("no_protected_feature_conflict", False)),
            }
            for item in contract.get("residual_components", [])
            if item.get("classification") != "intentional_open_boundary"
        ],
    }
    eligible = bool(
        decision == "pass"
        and hard_pass
        and role_pass
        and not contract.get("report_only")
    )
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
        "residual_roles": decision_doc["residual_roles"],
    }
    contract["downstream_eligible"] = eligible
    contract["final_status"] = "pass" if eligible else "needs_review"
    failures = [f for f in contract.get("failure_taxonomy", []) if f not in DECISION_FAILURES]
    if not eligible and "open_exterior_agent_decision_rejected" not in failures:
        failures.append("open_exterior_agent_decision_rejected")
    contract["failure_taxonomy"] = failures
    atomic_json(contract_path, contract)

    if eligible:
        _clear_loop_role_failures(manifest)
    if resolution is not None:
        _bind_resolution_contract(resolution, contract_path)

    manifest["open_exterior_contract"] = contract
    if isinstance(manifest.get("region_bpoly_arc_feedback"), dict):
        feedback = manifest["region_bpoly_arc_feedback"]
        feedback["open_exterior_contract"] = contract
        feedback["role_resolution_status"] = "pass" if eligible else "needs_review"
        if eligible:
            feedback["status"] = "pass"
            feedback["failure_taxonomy"] = [
                item for item in feedback.get("failure_taxonomy", []) if item not in DECISION_FAILURES
            ]
        feedback_path = Path(feedback.get("outputs", {}).get("feedback_json", ""))
        if feedback_path.name and feedback_path.parent.is_dir():
            atomic_json(feedback_path, feedback)
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
    parser.add_argument(
        "--residual-role",
        action="append",
        default=[],
        metavar="SEGMENT_ID=ROLE",
        help="Override the solid default for one residual component.",
    )
    parser.add_argument("--station-screen-json", type=Path)
    args = parser.parse_args()
    result = finalize(
        args.bdry_arc_manifest.resolve(),
        args.decision,
        args.rationale,
        resume_adaptive=args.resume_adaptive,
        residual_roles=_parse_roles(args.residual_role),
        station_screen_path=(args.station_screen_json.resolve() if args.station_screen_json else None),
    )
    print(json.dumps({
        "final_status": result.get("final_status"),
        "downstream_eligible": result.get("open_exterior_contract", {}).get("downstream_eligible"),
    }, indent=2))
    return 0 if result.get("open_exterior_contract", {}).get("downstream_eligible") else 2


if __name__ == "__main__":
    raise SystemExit(main())
