"""Autonomous, evidence-bound classification and shoreline patch utilities.

The module deliberately operates on boundary geometry, not isolated mesh
triangles.  It supplies deterministic measurements and transaction guards;
Codex supplies the visual classification in a hash-bound decision document.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import substring

from .quality_policy import (
    apply_quality_policy,
    classify_failure_codes,
    load_quality_policy,
    public_policy_binding,
)


DIAGNOSTIC_SCHEMA = "fvcom_thin_component_diagnostic_v2"
DECISION_SCHEMA = "fvcom_agent_thin_decision_v1"
PATCH_SCHEMA = "fvcom_local_shoreline_patch_v1"
CLOSURE_SCHEMA = "fvcom_autonomous_thin_closure_v1"

ROUTES = {
    "interior_topology_defect",
    "resolved_channel_meshing_defect",
    "subgrid_boundary_spike_or_sliver",
    "subgrid_wet_connection",
    "protected_or_source_conflict",
}


@dataclass(frozen=True)
class AutonomousThinConfig:
    minimum_elements_across: int = 3
    cusp_buffer_minimum_m: float = 1_000.0
    cusp_buffer_maximum_m: float = 5_000.0
    cusp_target_multiplier: float = 10.0
    cusp_component_multiplier: float = 2.0
    stable_bracket_target_multiplier: float = 2.0
    maximum_candidates_per_component: int = 3
    maximum_remesh_cycles: int = 3
    planning_node_limit: int = 900_000
    hard_node_limit: int = 1_000_000


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_cusp_buffer_m(
    local_target_m: float,
    component_diameter_m: float,
    config: AutonomousThinConfig | None = None,
) -> float:
    """Return the bounded, scale-relative local CUSP request buffer."""
    cfg = config or AutonomousThinConfig()
    value = max(
        cfg.cusp_target_multiplier * float(local_target_m),
        cfg.cusp_component_multiplier * float(component_diameter_m),
        cfg.cusp_buffer_minimum_m,
    )
    return float(min(cfg.cusp_buffer_maximum_m, value))


def resolution_feasibility(
    width_m: float | None,
    local_target_m: float,
    bathymetry_floor_m: float | None,
    estimated_nodes_at_required_size: int | None,
    config: AutonomousThinConfig | None = None,
) -> dict[str, Any]:
    """Evaluate whether a wet feature can support the required cross cells."""
    cfg = config or AutonomousThinConfig()
    if width_m is None or not np.isfinite(width_m) or width_m <= 0.0:
        return {
            "known": False,
            "resolvable": None,
            "required_target_m": None,
            "reason": "width_unavailable",
        }
    required = float(width_m) / float(cfg.minimum_elements_across)
    floor_ok = bool(
        bathymetry_floor_m is None
        or required >= float(bathymetry_floor_m) - 1.0e-9
    )
    budget_ok = bool(
        estimated_nodes_at_required_size is None
        or int(estimated_nodes_at_required_size) <= cfg.planning_node_limit
    )
    current_ok = bool(float(width_m) / max(float(local_target_m), 1.0e-30) >= cfg.minimum_elements_across)
    return {
        "known": True,
        "resolvable": bool(floor_ok and budget_ok),
        "resolved_at_current_target": current_ok,
        "width_m": float(width_m),
        "local_target_m": float(local_target_m),
        "width_over_target": float(width_m) / max(float(local_target_m), 1.0e-30),
        "minimum_elements_across": int(cfg.minimum_elements_across),
        "required_target_m": required,
        "bathymetry_floor_m": (
            float(bathymetry_floor_m) if bathymetry_floor_m is not None else None
        ),
        "estimated_nodes_at_required_size": (
            int(estimated_nodes_at_required_size)
            if estimated_nodes_at_required_size is not None
            else None
        ),
        "floor_ok": floor_ok,
        "budget_ok": budget_ok,
        "reason": "resolvable" if floor_ok and budget_ok else "resolution_infeasible",
    }


def validate_agent_decision(
    decision: dict[str, Any],
    *,
    diagnostic_sha256: str,
    mesh_sha256: str,
    diagnostic_input_hashes: dict[str, Any] | None = None,
) -> None:
    """Validate a Codex visual decision and reject stale or human-gated plans."""
    if decision.get("schema_version") != DECISION_SCHEMA:
        raise ValueError(f"decision schema must be {DECISION_SCHEMA}")
    if str(decision.get("diagnostic_sha256", "")).lower() != diagnostic_sha256.lower():
        raise ValueError("stale agent decision: diagnostic SHA-256 differs")
    if str(decision.get("input_mesh_sha256", "")).lower() != mesh_sha256.lower():
        raise ValueError("stale agent decision: mesh SHA-256 differs")
    actor = decision.get("decision_actor")
    if not isinstance(actor, dict) or actor.get("kind") != "codex_agent":
        raise ValueError("decision_actor.kind must be codex_agent")
    if not str(actor.get("identifier", "")).strip():
        raise ValueError("decision_actor.identifier is required")
    route = str(decision.get("route", ""))
    if route not in ROUTES:
        raise ValueError(f"unsupported autonomous thin route: {route!r}")
    cycle_index = int(decision.get("cycle_index_zero_based", 0))
    if cycle_index < 0 or cycle_index >= AutonomousThinConfig().maximum_remesh_cycles:
        raise ValueError("agent decision exceeds the maximum remesh-cycle count")
    evidence = decision.get("visual_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("at least one inspected visual-evidence record is required")
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("visual evidence entries must be objects")
        if not str(item.get("path", "")).strip() or not str(item.get("sha256", "")).strip():
            raise ValueError("visual evidence requires path and SHA-256")
        path = Path(str(item["path"]))
        if not path.is_file() or sha256_file(path).lower() != str(item["sha256"]).lower():
            raise ValueError(f"visual evidence is absent or stale: {path}")
    if not str(decision.get("observations", "")).strip():
        raise ValueError("agent observations are required")
    if diagnostic_input_hashes is not None:
        bound = decision.get("bound_input_hashes")
        if not isinstance(bound, dict) or canonical_sha256(bound) != canonical_sha256(
            diagnostic_input_hashes
        ):
            raise ValueError("stale agent decision: bound input hashes differ")
    if route in {
        "resolved_channel_meshing_defect",
        "subgrid_boundary_spike_or_sliver",
        "subgrid_wet_connection",
    }:
        source = decision.get("source_window")
        if not isinstance(source, dict):
            raise ValueError("boundary routes require source_window")
        if int(source.get("chain_index_zero_based", -1)) < 0:
            raise ValueError("source_window requires a nonnegative chain index")
        indices = source.get("source_node_indices_zero_based")
        if not isinstance(indices, list) or not indices:
            raise ValueError("source_window requires implicated source nodes")
        protected = decision.get("protected_feature_check")
        if not isinstance(protected, dict):
            raise ValueError("boundary routes require protected_feature_check")
        for key in (
            "obc_touched",
            "explicit_mission_feature_touched",
            "forcing_anchor_touched",
        ):
            if protected.get(key) is not False:
                raise ValueError(f"boundary route has protected or unresolved lineage: {key}")
        if route == "subgrid_boundary_spike_or_sliver":
            if not diagnostic_input_hashes or not diagnostic_input_hashes.get("cusp_gpkg"):
                raise ValueError("shoreline-correction route requires hash-bound CUSP evidence")
        if diagnostic_input_hashes and not diagnostic_input_hashes.get("gshhs_gpkg"):
            raise ValueError("boundary route requires hash-bound GSHHS evidence")
    if route == "resolved_channel_meshing_defect":
        plan = decision.get("resolution_evidence")
        if not isinstance(plan, dict):
            raise ValueError("resolved-channel route requires resolution_evidence")
        feasible = resolution_feasibility(
            plan.get("width_m"),
            plan.get("local_target_m"),
            plan.get("bathymetry_floor_m"),
            plan.get("estimated_nodes_at_required_size"),
        )
        if not feasible.get("resolvable"):
            raise ValueError("resolved-channel route is not resolution-feasible")
    if route == "subgrid_wet_connection":
        plan = decision.get("resolution_evidence")
        if not isinstance(plan, dict):
            raise ValueError("subgrid-connection route requires resolution_evidence")
        feasible = resolution_feasibility(
            plan.get("width_m"),
            plan.get("local_target_m"),
            plan.get("bathymetry_floor_m"),
            plan.get("estimated_nodes_at_required_size"),
        )
        if feasible.get("resolvable") is not False:
            raise ValueError("subgrid connection has not been proven resolution-infeasible")
    if decision.get("human_review_required") is True:
        raise ValueError("autonomous-thin-v1 must not introduce a human-review gate")


def decision_template(
    diagnostic: dict[str, Any],
    diagnostic_path: str | Path,
    component: dict[str, Any],
) -> dict[str, Any]:
    """Create a pending agent decision template bound to current evidence."""
    diagnostic_path = Path(diagnostic_path)
    images = [
        value
        for value in (
            diagnostic.get("whole_domain_map"),
            component.get("decision_diagram"),
            component.get("image"),
        )
        if value and Path(str(value)).is_file()
    ]
    return {
        "schema_version": DECISION_SCHEMA,
        "status": "agent_visual_classification_pending",
        "diagnostic": str(diagnostic_path),
        "diagnostic_sha256": sha256_file(diagnostic_path),
        "input_mesh_sha256": diagnostic["input_hashes"]["mesh"],
        "bound_input_hashes": diagnostic["input_hashes"],
        "component_id": component["component_id"],
        "cycle_index_zero_based": 0,
        "decision_actor": {"kind": "codex_agent", "identifier": ""},
        "route": "",
        "observations": "",
        "visual_evidence": [
            {"path": str(path), "sha256": sha256_file(path)} for path in images
        ],
        "source_window": {
            "chain_index_zero_based": component.get("source_chain_index_zero_based"),
            "source_node_indices_zero_based": component.get(
                "source_node_indices_zero_based", []
            ),
        },
        "protected_feature_check": {
            "obc_touched": bool(component.get("touches_open_boundary", False)),
            "explicit_mission_feature_touched": None,
            "forcing_anchor_touched": None,
        },
    }


def no_op_closure_report(diagnostic: dict[str, Any]) -> dict[str, Any]:
    """Return the deterministic autonomous-thin decision for zero components."""
    if int(diagnostic.get("component_count", -1)) != 0:
        raise ValueError("no-op closure requires a zero-component diagnostic")
    if int(diagnostic.get("superthin_triangle_count", -1)) != 0:
        raise ValueError("no-op closure requires zero superthin triangles")
    benchmark_ready = bool(
        diagnostic.get(
            "benchmark_grid_baseline_ready",
            diagnostic.get("fvcom_ready", False),
        )
    )
    return {
        "schema_version": CLOSURE_SCHEMA,
        "profile": "autonomous-thin-v1",
        "status": "pass",
        "route": "no_op",
        "autonomous_thin_closed": True,
        "minimal_local_debt_closed": True,
        "benchmark_grid_baseline_ready": benchmark_ready,
        "fvcom_ready": benchmark_ready,
        "accepted": benchmark_ready,
        "submission_eligible": bool(diagnostic.get("submission_eligible", False)),
        "regional_refinement_debt": list(
            diagnostic.get("regional_refinement_debt") or []
        ),
        "quality_advisories": dict(diagnostic.get("quality_advisories") or {}),
        "quality_policy": dict(
            diagnostic.get("quality_policy") or public_policy_binding()
        ),
        "failure_taxonomy": list(diagnostic.get("fvcom_readiness_failure_taxonomy") or []),
        "input_hashes": dict(diagnostic.get("input_hashes") or {}),
    }


def interior_topology_plan(
    decision: dict[str, Any], diagnostic: dict[str, Any]
) -> dict[str, Any]:
    """Translate a Codex interior classification to the existing local tool."""
    if decision.get("route") != "interior_topology_defect":
        raise ValueError("interior topology plan requires the interior route")
    component_id = str(decision.get("component_id", ""))
    component = next(
        (value for value in diagnostic.get("components", []) if value.get("component_id") == component_id),
        None,
    )
    if component is None:
        raise ValueError("interior component is absent from diagnostic")
    return {
        "schema_version": "fvcom_visual_superthin_repair_plan_v1",
        "input_mesh_sha256": diagnostic["input_hashes"]["mesh"],
        "review": {
            "status": "reviewed",
            "reviewed_by": decision["decision_actor"]["identifier"],
            "reviewer_kind": "codex_agent",
            "observations": decision["observations"],
            "visual_evidence": [value["path"] for value in decision["visual_evidence"]],
            "manageable": True,
        },
        "component": {"component_id": component_id},
        "actions": [
            {
                "tool": "constrained_retriangulation",
                "patch_rings": [1, 2, 4],
                "maximum_support_nodes": 0,
                "local_relaxation": False,
            },
            {
                "tool": "minmax_cavity_triangulation",
                "patch_rings": [1, 2, 4],
                "maximum_support_nodes": 0,
                "local_relaxation": False,
            },
        ],
        "acceptance": {
            "require_strict_superthin_reduction": True,
            "fixed_boundary_coordinates": True,
            "preserve_restricted_edges": True,
            "isolated_triangle_deletion_permitted": False,
        },
    }


def _iter_lines(geometry: Any) -> Iterable[LineString]:
    if isinstance(geometry, LineString):
        if not geometry.is_empty:
            yield geometry
    elif isinstance(geometry, MultiLineString):
        for value in geometry.geoms:
            if not value.is_empty:
                yield value


def _candidate_subline(line: LineString, start: Point, end: Point) -> LineString | None:
    a = float(line.project(start))
    b = float(line.project(end))
    if abs(a - b) <= 1.0e-9:
        return None
    piece = substring(line, min(a, b), max(a, b))
    if not isinstance(piece, LineString) or piece.length <= 0.0:
        return None
    coords = list(piece.coords)
    if a > b:
        coords.reverse()
    return LineString(coords)


def rank_shoreline_candidates(
    geometries: Iterable[Any],
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    original_window: LineString,
    *,
    local_target_m: float,
    horizontal_accuracy_m: Iterable[float | None] | None = None,
    source_dates: Iterable[str | int | None] | None = None,
) -> list[dict[str, Any]]:
    """Rank CUSP arcs that span stable brackets without source-specific rules."""
    start = Point(start_xy)
    end = Point(end_xy)
    values = list(geometries)
    accuracies = list(horizontal_accuracy_m or [None] * len(values))
    dates = list(source_dates or [None] * len(values))
    if len(accuracies) < len(values):
        accuracies.extend([None] * (len(values) - len(accuracies)))
    if len(dates) < len(values):
        dates.extend([None] * (len(values) - len(dates)))

    def _date_value(value: str | int | None) -> int | None:
        digits = "".join(character for character in str(value or "") if character.isdigit())
        if len(digits) == 4:
            digits += "0101"
        if len(digits) < 8:
            return None
        return int(digits[:8])

    parsed_dates = [_date_value(value) for value in dates]
    finite_dates = [value for value in parsed_dates if value is not None]
    newest_date = max(finite_dates) if finite_dates else None
    oldest_date = min(finite_dates) if finite_dates else None
    records: list[dict[str, Any]] = []
    maximum_snap = max(500.0, 2.0 * float(local_target_m))
    original_vector = np.asarray(original_window.coords[-1], dtype=float) - np.asarray(
        original_window.coords[0], dtype=float
    )
    original_norm = max(float(np.linalg.norm(original_vector)), 1.0e-30)
    for source_index, (geometry, accuracy, source_date, parsed_date) in enumerate(
        zip(values, accuracies, dates, parsed_dates)
    ):
        for part_index, line in enumerate(_iter_lines(geometry)):
            piece = _candidate_subline(line, start, end)
            if piece is None:
                continue
            start_gap = float(Point(piece.coords[0]).distance(start))
            end_gap = float(Point(piece.coords[-1]).distance(end))
            if max(start_gap, end_gap) > maximum_snap:
                continue
            vector = np.asarray(piece.coords[-1], dtype=float) - np.asarray(
                piece.coords[0], dtype=float
            )
            norm = max(float(np.linalg.norm(vector)), 1.0e-30)
            tangent_cosine = float(np.clip(np.dot(vector, original_vector) / (norm * original_norm), -1.0, 1.0))
            tangent_penalty = 1.0 - tangent_cosine
            hausdorff = float(piece.hausdorff_distance(original_window))
            reported_accuracy = (
                float(accuracy)
                if accuracy is not None and np.isfinite(float(accuracy))
                else None
            )
            score = (
                max(start_gap, end_gap) / maximum_snap
                + 0.25 * tangent_penalty
                + 0.10 * hausdorff / max(float(local_target_m), 1.0)
                + 0.05 * (reported_accuracy or 0.0) / max(float(local_target_m), 1.0)
                + 0.05 * (
                    (newest_date - parsed_date) / max(newest_date - oldest_date, 1)
                    if parsed_date is not None and newest_date is not None and oldest_date is not None
                    else 0.5
                )
            )
            records.append(
                {
                    "source_feature_index": int(source_index),
                    "source_part_index": int(part_index),
                    "score": float(score),
                    "start_gap_m": start_gap,
                    "end_gap_m": end_gap,
                    "tangent_cosine": tangent_cosine,
                    "hausdorff_to_original_m": hausdorff,
                    "reported_horizontal_accuracy_m": reported_accuracy,
                    "source_date": str(source_date) if source_date is not None else None,
                    "geometry": piece,
                }
            )
    records.sort(
        key=lambda item: (
            float(item["score"]),
            int(item["source_feature_index"]),
            int(item["source_part_index"]),
        )
    )
    return records


def _resample_line(line: LineString, spacing_m: float) -> LineString:
    length = float(line.length)
    if length <= 0.0:
        raise ValueError("cannot resample an empty line")
    count = max(1, int(math.ceil(length / max(float(spacing_m), 1.0e-9))))
    distances = np.linspace(0.0, length, count + 1)
    return LineString([line.interpolate(float(value)).coords[0] for value in distances])


def regularize_shoreline(
    candidate: LineString,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    local_target_m: float,
    horizontal_accuracy_m: float | None = None,
) -> LineString:
    """Snap, simplify, and resample a source arc at the model scale."""
    tolerance = max(
        float(horizontal_accuracy_m or 0.0),
        0.25 * float(local_target_m),
    )
    coords = [tuple(map(float, start_xy)), *list(candidate.coords)[1:-1], tuple(map(float, end_xy))]
    line = LineString(coords).simplify(tolerance, preserve_topology=False)
    coords = [tuple(map(float, start_xy)), *list(line.coords)[1:-1], tuple(map(float, end_xy))]
    return _resample_line(LineString(coords), float(local_target_m))


def shoreline_junction_turns_deg(
    replacement: LineString,
    incoming_vector_xy: tuple[float, float] | np.ndarray,
    outgoing_vector_xy: tuple[float, float] | np.ndarray,
) -> dict[str, float]:
    """Measure directional turns where a replacement rejoins its source chain."""
    coords = np.asarray(replacement.coords, dtype=float)
    if len(coords) < 2:
        raise ValueError("replacement shoreline needs at least two coordinates")

    def _angle(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
            return 180.0
        cosine = float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    start_turn = _angle(np.asarray(incoming_vector_xy, dtype=float), coords[1] - coords[0])
    end_turn = _angle(coords[-1] - coords[-2], np.asarray(outgoing_vector_xy, dtype=float))
    return {
        "start_turn_deg": start_turn,
        "end_turn_deg": end_turn,
        "maximum_turn_deg": max(start_turn, end_turn),
    }


def replace_cyclic_window(
    coordinates: np.ndarray,
    start_index: int,
    end_index: int,
    replacement: LineString,
) -> tuple[np.ndarray, list[int]]:
    """Replace the forward cyclic interval (start,end) and retain brackets."""
    values = np.asarray(coordinates, dtype=float)
    size = len(values)
    if size < 4:
        raise ValueError("a boundary chain needs at least four vertices")
    start_index %= size
    end_index %= size
    removed: list[int] = []
    cursor = (start_index + 1) % size
    while cursor != end_index:
        removed.append(cursor)
        cursor = (cursor + 1) % size
        if len(removed) >= size - 1:
            raise ValueError("replacement interval consumes the complete chain")
    keep_after: list[int] = []
    cursor = end_index
    while cursor != start_index:
        keep_after.append(cursor)
        cursor = (cursor + 1) % size
    interior_values = list(replacement.coords)[1:-1]
    interior = (
        np.asarray(interior_values, dtype=float)
        if interior_values
        else np.empty((0, 2), dtype=float)
    )
    result = np.vstack(
        [
            values[start_index][None, :],
            interior,
            values[np.asarray(keep_after, dtype=int)],
        ]
    )
    return result, removed


def boundary_transaction_audit(
    before_domain: Polygon,
    after_domain: Polygon,
    *,
    expected_hole_count: int,
    obc_before: LineString | MultiLineString,
    obc_after: LineString | MultiLineString,
    protected_points: Iterable[Point] = (),
) -> dict[str, Any]:
    """Audit the structural invariants available before remeshing."""
    failures: list[str] = []
    if after_domain.is_empty or not after_domain.is_valid or after_domain.area <= 0.0:
        failures.append("patched_domain_invalid")
    if len(after_domain.interiors) != int(expected_hole_count):
        failures.append("island_hole_count_changed")
    if obc_before.hausdorff_distance(obc_after) > 0.01:
        failures.append("open_boundary_geometry_changed")
    lost = [index for index, point in enumerate(protected_points) if not after_domain.covers(point)]
    if lost:
        failures.append("protected_feature_lost")
    try:
        symmetric = before_domain.symmetric_difference(after_domain)
        symmetric_area = float(symmetric.area)
    except Exception:
        symmetric_area = float("nan")
    return {
        "passed": not failures,
        "failure_taxonomy": failures,
        "before_area_m2": float(before_domain.area),
        "after_area_m2": float(after_domain.area),
        "signed_area_delta_m2": float(after_domain.area - before_domain.area),
        "absolute_symmetric_difference_m2": symmetric_area,
        "relative_symmetric_difference": float(
            symmetric_area / max(before_domain.area, 1.0e-30)
        ),
        "hole_count_before": int(len(before_domain.interiors)),
        "hole_count_after": int(len(after_domain.interiors)),
        "protected_points_lost": lost,
    }


def closure_acceptance(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    expected_open_boundary_count: int,
    roundtrip_passed: bool,
) -> dict[str, Any]:
    """Apply the non-negotiable post-remesh autonomous closure gates."""
    findings: list[str] = []
    before_thin = int(before.get("superthin_triangle_count", 0))
    after_thin = int(after.get("superthin_triangle_count", before_thin))
    if before_thin and after_thin >= before_thin:
        findings.append("superthin_repair_not_monotonic")
    if int(after.get("connected_component_count", 1)) != 1:
        findings.append("multiple_mesh_components")
    if int(after.get("singly_connected_triangle_count", 0)) != 0:
        findings.append("singly_connected_elements_present")
    if int(after.get("nonmanifold_edge_count", 0)) != 0:
        findings.append("nonmanifold_edges_present")
    if int(after.get("nonpositive_area_count", 0)) != 0:
        findings.append("nonpositive_triangle_area")
    if int(after.get("open_boundary_chain_count", expected_open_boundary_count)) != int(
        expected_open_boundary_count
    ):
        findings.append("open_boundary_chain_count_mismatch")
    if not bool(after.get("open_boundary_ordered", True)):
        findings.append("open_boundary_nodestring_not_ordered_on_mesh")
    if not bool(after.get("forcing_compatible", True)):
        findings.append("open_boundary_forcing_incompatible")
    if not roundtrip_passed:
        findings.append("2dm_roundtrip_failed")
    if int(after.get("count_valence_above_8", 0)) > 0:
        findings.append("node_valence_above_threshold")
    if after_thin > 0:
        findings.append("superthin_elements_present")
    policy = load_quality_policy()
    result = apply_quality_policy(
        {}, findings, advisories={"closure_after": dict(after)}, policy=policy
    )
    classified = classify_failure_codes(findings, policy)
    submission_failures = list(classified["submission_preconditions"])
    result.update({
        "passed": bool(result["benchmark_grid_baseline_ready"]),
        "submission_eligible": bool(
            result["benchmark_grid_baseline_ready"] and not submission_failures
        ),
        "submission_failure_taxonomy": submission_failures,
        "superthin_before": before_thin,
        "superthin_after": after_thin,
        "autonomous_thin_closed": bool(
            result["benchmark_grid_baseline_ready"] and after_thin == 0
        ),
    })
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items() if key != "geometry"}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def config_dict(config: AutonomousThinConfig | None = None) -> dict[str, Any]:
    return asdict(config or AutonomousThinConfig())
