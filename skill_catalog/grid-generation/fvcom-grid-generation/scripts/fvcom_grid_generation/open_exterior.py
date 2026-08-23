"""Strict downstream validation for FVCOM open-exterior contracts v1 and v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _resolve(value: str | Path | None, parent: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (parent / path).resolve()


def discover_open_exterior_contract(source: str | Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Follow boundary-resolution/loop/arc lineage to the strict contract."""
    current = Path(source).resolve()
    seen: set[Path] = set()
    for _ in range(8):
        if current in seen or not current.is_file():
            break
        seen.add(current)
        doc = _read(current)
        if doc.get("schema_version") in {"fvcom_open_exterior_contract_v1", "fvcom_open_exterior_contract_v2"}:
            return doc, current
        embedded = doc.get("open_exterior_contract")
        if isinstance(embedded, dict) and embedded.get("schema_version") in {"fvcom_open_exterior_contract_v1", "fvcom_open_exterior_contract_v2"}:
            candidate = _resolve(doc.get("outputs", {}).get("open_exterior_contract"), current.parent)
            return (embedded, candidate if candidate and candidate.is_file() else current)
        candidates = [
            doc.get("outputs", {}).get("open_exterior_contract"),
            doc.get("inputs", {}).get("manifest_json"),
            doc.get("inputs", {}).get("model_boundary_loop_manifest"),
            (doc.get("boundary") or {}).get("open_exterior_contract"),
            (doc.get("boundary") or {}).get("resolution_manifest"),
            (doc.get("boundary") or {}).get("model_boundary_loop_manifest"),
        ]
        next_path = next(
            (path for value in candidates if (path := _resolve(value, current.parent)) and path.is_file()),
            None,
        )
        if next_path is None:
            break
        current = next_path
    return None, None


def validate_open_exterior_contract(source: str | Path, *, required: bool = True) -> dict[str, Any]:
    contract, contract_path = discover_open_exterior_contract(source)
    failures: list[str] = []
    if contract is None:
        if required:
            failures.append("open_exterior_contract_missing")
        return {"passed": not failures, "failure_taxonomy": failures, "contract_path": None}
    if contract.get("report_only"):
        failures.append("diagnostic_only_report_only_policy")
    schema = contract.get("schema_version")
    if schema not in {"fvcom_open_exterior_contract_v1", "fvcom_open_exterior_contract_v2"}:
        failures.append("open_exterior_contract_schema_unsupported")
    metrics = contract.get("hard_metrics", {})
    for field, failure in (
        ("absolute_gate_pass", "open_exterior_absolute_gate_failed"),
        ("fraction_gate_pass", "open_exterior_fraction_gate_failed"),
        ("coverage_gate_pass", "open_exterior_coverage_gate_failed"),
    ):
        if metrics.get(field) is not True:
            failures.append(failure)
    if contract.get("downstream_eligible") is not True:
        failures.append("open_exterior_not_downstream_eligible")
    decision = contract.get("agent_decision", {})
    if decision.get("status") != "pass":
        failures.append("open_exterior_agent_decision_missing_or_rejected")
    decision_path = _resolve(decision.get("path"), contract_path.parent if contract_path else Path.cwd())
    map_path = _resolve((contract.get("map") or {}).get("path"), contract_path.parent if contract_path else Path.cwd())
    if not decision_path or not decision_path.is_file():
        failures.append("open_exterior_agent_decision_file_missing")
    elif decision.get("sha256") != sha256_file(decision_path):
        failures.append("open_exterior_agent_decision_hash_stale")
    else:
        decision_doc = _read(decision_path)
        if decision_doc.get("decision_actor", {}).get("kind") != "codex_agent":
            failures.append("open_exterior_decision_actor_invalid")
        if not map_path or not map_path.is_file():
            failures.append("open_exterior_map_missing")
        elif decision_doc.get("inspected_map_sha256") != sha256_file(map_path):
            failures.append("open_exterior_map_hash_stale")
        if decision_doc.get("bound_source_hashes") != contract.get("source_hashes"):
            failures.append("open_exterior_source_hash_binding_stale")
        if schema == "fvcom_open_exterior_contract_v2":
            _validate_residual_roles(
                contract,
                contract_path,
                decision_doc,
                failures,
            )
    obc = contract.get("obc_geometry", {})
    expected = int(obc.get("expected_count", 0) or 0)
    if int(obc.get("delivered_count", 0) or 0) != expected:
        failures.append("open_exterior_obc_count_invalid")
    if expected > 0:
        if obc.get("simple_nonbranching") is not True:
            failures.append("open_exterior_obc_not_simple_nonbranching")
        if float(obc.get("nonendpoint_land_crossing_m", 0.0) or 0.0) > float(metrics.get("absolute_limit_m", 250.0)):
            failures.append("open_exterior_obc_land_crossing")
    return {
        "passed": not failures,
        "failure_taxonomy": list(dict.fromkeys(failures)),
        "contract_path": str(contract_path) if contract_path else None,
        "contract_sha256": sha256_file(contract_path) if contract_path and contract_path.is_file() else None,
        "obc_placement_family": contract.get("obc_placement_family"),
        "contract_schema": schema,
        "residual_role_summary": contract.get("residual_role_summary"),
    }


def _validate_residual_roles(
    contract: dict[str, Any],
    contract_path: Path,
    decision_doc: dict[str, Any],
    failures: list[str],
) -> None:
    summary = contract.get("residual_role_summary", {})
    if int(summary.get("pending_count", -1)) != 0:
        failures.append("residual_boundary_role_pending")
    if float(summary.get("unassigned_residual_length_m", -1.0)) > 1.0e-9:
        failures.append("residual_boundary_unassigned")
    decision_roles = {
        int(item.get("segment_id", -1)): item
        for item in decision_doc.get("residual_roles", [])
    }
    component_maps = contract.get("component_maps", {})
    for component in contract.get("residual_components", []):
        if component.get("classification") == "intentional_open_boundary":
            continue
        segment_id = int(component.get("segment_id", -1))
        role = component.get("assigned_role")
        if role not in {"solid_lagoon_closure", "secondary_tidal_obc"}:
            failures.append("residual_boundary_role_invalid_or_missing")
            continue
        if component.get("role_status") != "accepted":
            failures.append("residual_boundary_role_not_accepted")
        decision = decision_roles.get(segment_id, {})
        if decision.get("role") != role:
            failures.append("residual_boundary_role_decision_mismatch")
        if decision.get("no_artificial_bar") is not True:
            failures.append("residual_boundary_artificial_bar_not_cleared")
        if decision.get("no_protected_feature_conflict") is not True:
            failures.append("residual_boundary_protected_conflict_not_cleared")
        map_record = component_maps.get(str(segment_id), {})
        map_path = _resolve(map_record.get("path"), contract_path.parent)
        if not map_path or not map_path.is_file():
            failures.append("residual_boundary_component_map_missing")
        elif map_record.get("sha256") != sha256_file(map_path):
            failures.append("residual_boundary_component_map_stale")
        elif decision.get("component_map_sha256") != map_record.get("sha256"):
            failures.append("residual_boundary_component_map_not_bound_to_decision")
        if role == "solid_lagoon_closure":
            geometry = component.get("solid_role_geometry", {})
            if geometry.get("eligible") is not True:
                failures.append("residual_solid_boundary_geometry_invalid")
            if component.get("forcing_eligibility") is not None:
                failures.append("residual_solid_boundary_has_forcing_lineage")
        else:
            if not component.get("forcing_eligibility", {}).get("station_id"):
                failures.append("residual_secondary_obc_station_missing")
    if int(summary.get("secondary_tidal_obc_count", 0) or 0) > 0:
        screen = contract.get("station_screen", {})
        screen_path = _resolve(screen.get("path"), contract_path.parent)
        if screen.get("status") != "pass" or not screen_path or not screen_path.is_file():
            failures.append("residual_secondary_obc_station_screen_missing")
        elif screen.get("sha256") != sha256_file(screen_path):
            failures.append("residual_secondary_obc_station_screen_stale")


__all__ = ["discover_open_exterior_contract", "validate_open_exterior_contract"]
