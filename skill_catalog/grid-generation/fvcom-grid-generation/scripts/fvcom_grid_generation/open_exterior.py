"""Strict downstream validation for active FVCOM open-exterior contracts v1-v2."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


GRID_BOUNDARY_GATE_POLICIES = ("strict", "reviewed-adaptive-v2")


def _finite_number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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
        if doc.get("schema_version") in {"fvcom_open_exterior_contract_v1", "fvcom_open_exterior_contract_v2", "fvcom_open_exterior_contract_v3"}:
            return doc, current
        embedded = doc.get("open_exterior_contract")
        if isinstance(embedded, dict) and embedded.get("schema_version") in {"fvcom_open_exterior_contract_v1", "fvcom_open_exterior_contract_v2", "fvcom_open_exterior_contract_v3"}:
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


def validate_open_exterior_contract(
    source: str | Path,
    *,
    required: bool = True,
    require_topology_coverage: bool = True,
) -> dict[str, Any]:
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
    coverage = contract.get("coastline_source_coverage") or {}
    coverage_required = bool(
        require_topology_coverage
        and int((contract.get("obc_geometry") or {}).get("expected_count", 0) or 0) > 0
    )
    decision = contract.get("agent_decision", {})
    decision_doc: dict[str, Any] = {}
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
    if coverage_required:
        if contract.get("coastline_source_coverage_required") is not True or not coverage:
            failures.append("coastline_source_coverage_missing")
        else:
            _validate_coastline_source_coverage(
                coverage,
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
        "coastline_source_coverage": coverage,
    }


def validate_grid_boundary_gate(
    open_exterior_source: str | Path | None,
    boundary_resolution_source: str | Path | None,
    *,
    policy: str = "strict",
    strict_contract_required: bool = True,
) -> dict[str, Any]:
    """Evaluate Grid Generation's explicit downstream boundary gate.

    ``strict`` preserves the active open-exterior contract decision.  The
    ``reviewed-adaptive-v2`` compatibility policy is intentionally narrower:
    it may replace missing or rejected upstream review evidence only when the
    exact Adaptive-v2 package passes its immutable, geometry-facing checks.
    Mesh construction still performs the full loaded-node boundary contract.
    """
    if policy not in GRID_BOUNDARY_GATE_POLICIES:
        raise ValueError(
            "boundary gate policy must be one of "
            + ", ".join(GRID_BOUNDARY_GATE_POLICIES)
        )
    strict_audit = validate_open_exterior_contract(
        open_exterior_source or "",
        required=bool(strict_contract_required),
    )
    if policy == "strict" or strict_audit["passed"]:
        return {
            "schema_version": "fvcom_grid_boundary_gate_v1",
            "policy": policy,
            "passed": bool(strict_audit["passed"]),
            "decision_basis": "strict_open_exterior_contract",
            "failure_taxonomy": list(strict_audit["failure_taxonomy"]),
            "advisory_taxonomy": [],
            "strict_open_exterior_audit": strict_audit,
            "boundary_resolution": None,
        }

    failures: list[str] = []
    resolution_path = (
        Path(boundary_resolution_source).expanduser().resolve()
        if boundary_resolution_source
        else None
    )
    resolution: dict[str, Any] = {}
    if resolution_path is None or not resolution_path.is_file():
        failures.append("reviewed_boundary_resolution_manifest_missing")
    else:
        try:
            resolution = _read(resolution_path)
        except Exception:
            failures.append("reviewed_boundary_resolution_manifest_unreadable")

    if resolution:
        if resolution.get("schema_version") != "fvcom_boundary_resolution_manifest_v2":
            failures.append("reviewed_boundary_resolution_schema_invalid")
        if resolution.get("profile") != "adaptive-coastal-v2":
            failures.append("reviewed_boundary_resolution_profile_invalid")
        if resolution.get("final_status") != "pass":
            failures.append("reviewed_boundary_resolution_not_pass")
        if list(resolution.get("failure_taxonomy") or []):
            failures.append("reviewed_boundary_resolution_has_failures")

    qa = resolution.get("qa") or {}
    if resolution and qa.get("resolved_domain_valid") is not True:
        failures.append("reviewed_boundary_domain_invalid")
    if resolution and _finite_number(
        qa.get("open_arc_land_intersection_m"), float("inf")
    ) > 1.0e-6:
        failures.append("reviewed_boundary_open_arc_intersects_land")
    if resolution and _finite_number(
        qa.get("open_arc_exterior_overlap_fraction"), 0.0
    ) < 0.999:
        failures.append("reviewed_boundary_open_arc_not_on_exterior")
    if resolution and _integer(
        qa.get("protected_underresolved_passage_count"), -1
    ) != 0:
        failures.append("reviewed_boundary_protected_passage_underresolved")
    if resolution and _finite_number(
        qa.get("maximum_edge_to_target_ratio"), float("inf")
    ) > 1.55:
        failures.append("reviewed_boundary_edge_to_target_maximum_exceeded")
    if resolution and _finite_number(
        qa.get("p95_edge_to_target_ratio"), float("inf")
    ) > 1.55:
        failures.append("reviewed_boundary_edge_to_target_p95_exceeded")
    if resolution and _finite_number(
        qa.get("maximum_target_gradation"), float("inf")
    ) > 0.20 + 1.0e-9:
        failures.append("reviewed_boundary_target_gradation_exceeded")

    artifacts: dict[str, Any] = {}
    if resolution_path and resolution_path.is_file():
        artifacts["boundary_resolution_manifest"] = {
            "path": str(resolution_path),
            "sha256": sha256_file(resolution_path),
        }
    if resolution:
        gpkg_path = _resolve(
            (resolution.get("outputs") or {}).get("boundary_resolution_gpkg"),
            resolution_path.parent,
        )
        loops_path = _resolve(
            (resolution.get("inputs") or {}).get("model_boundary_loops_gpkg"),
            resolution_path.parent,
        )
        for key, path, failure in (
            (
                "boundary_resolution_gpkg",
                gpkg_path,
                "reviewed_boundary_resolution_package_missing",
            ),
            (
                "model_boundary_loops_gpkg",
                loops_path,
                "reviewed_boundary_source_loops_missing",
            ),
        ):
            if path is None or not path.is_file():
                failures.append(failure)
            else:
                artifacts[key] = {"path": str(path), "sha256": sha256_file(path)}

    arc_path = Path(open_exterior_source).expanduser().resolve() if open_exterior_source else None
    if arc_path and arc_path.is_file() and resolution_path and resolution_path.is_file():
        try:
            arc_doc = _read(arc_path)
        except Exception:
            failures.append("reviewed_boundary_arc_manifest_unreadable")
        else:
            recorded = _resolve(
                (arc_doc.get("outputs") or {}).get("boundary_resolution_manifest"),
                arc_path.parent,
            )
            if recorded is not None and recorded != resolution_path:
                failures.append("reviewed_boundary_resolution_lineage_mismatch")
            artifacts["bdry_arc_manifest"] = {
                "path": str(arc_path),
                "sha256": sha256_file(arc_path),
            }

    return {
        "schema_version": "fvcom_grid_boundary_gate_v1",
        "policy": policy,
        "passed": not failures,
        "decision_basis": "reviewed_adaptive_v2_package",
        "failure_taxonomy": list(dict.fromkeys(failures)),
        "advisory_taxonomy": [
            f"upstream_open_exterior:{value}"
            for value in strict_audit["failure_taxonomy"]
        ],
        "strict_open_exterior_audit": strict_audit,
        "boundary_resolution": {
            "schema_version": resolution.get("schema_version"),
            "profile": resolution.get("profile"),
            "final_status": resolution.get("final_status"),
            "qa": {
                key: qa.get(key)
                for key in (
                    "resolved_domain_valid",
                    "open_arc_land_intersection_m",
                    "open_arc_exterior_overlap_fraction",
                    "protected_underresolved_passage_count",
                    "maximum_edge_to_target_ratio",
                    "p95_edge_to_target_ratio",
                    "maximum_target_gradation",
                )
            },
            "artifacts": artifacts,
        },
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


def _validate_coastline_source_coverage(
    coverage: dict[str, Any],
    contract: dict[str, Any],
    contract_path: Path,
    decision_doc: dict[str, Any],
    failures: list[str],
) -> None:
    if coverage.get("schema_version") != "fvcom_coastline_source_coverage_v1":
        failures.append("coastline_source_coverage_schema_unsupported")
    if coverage.get("downstream_eligible") is not True:
        failures.append("coastline_source_footprint_incomplete")
    if float(coverage.get("coverage_factor_x", 0.0) or 0.0) < 2.0 or float(
        coverage.get("coverage_factor_y", 0.0) or 0.0
    ) < 2.0:
        failures.append("boundary_geometry_outside_coastline_coverage")
    if coverage.get("model_bbox_centrally_contained") is not True or coverage.get("region_bpoly_covered") is not True:
        failures.append("coastline_source_footprint_incomplete")
    if float(coverage.get("source_frame_dependency_length_m", float("inf"))) > float(
        coverage.get("source_frame_dependency_limit_m", 1.0)
    ):
        failures.append("coastline_source_frame_used_as_land_boundary")
    if coverage.get("physical_coastline_only_landfalls") is not True:
        failures.append("coastline_clip_edge_landfall")
    coverage_path = _resolve(coverage.get("contract_path"), contract_path.parent)
    expected_hash = (contract.get("source_hashes") or {}).get("coastline_source_coverage")
    if not coverage_path or not coverage_path.is_file():
        failures.append("coastline_source_coverage_file_missing")
    elif expected_hash != sha256_file(coverage_path):
        failures.append("coastline_source_coverage_hash_stale")
    for key, decision_key in (
        ("whole_domain", "inspected_coastline_coverage_map_sha256"),
        ("source_edge_zoom", "inspected_coastline_coverage_zoom_sha256"),
    ):
        record = (coverage.get("maps") or {}).get(key) or {}
        map_path = _resolve(record.get("path"), contract_path.parent)
        if not map_path or not map_path.is_file():
            failures.append("coastline_source_coverage_map_missing")
        elif record.get("sha256") != sha256_file(map_path):
            failures.append("coastline_source_coverage_map_stale")
        elif decision_doc.get(decision_key) != record.get("sha256"):
            failures.append("coastline_source_coverage_map_not_bound_to_decision")


__all__ = [
    "GRID_BOUNDARY_GATE_POLICIES",
    "discover_open_exterior_contract",
    "validate_grid_boundary_gate",
    "validate_open_exterior_contract",
]
