"""Agent-reviewed, one-component FVCOM superthin repair.

This module is deliberately separate from the automatic conditioning profiles.
It applies exactly one visually reviewed component plan, evaluates bounded local
topology candidates, and commits at most one globally audited transaction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Iterable

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon
import triangle as tr

from .local_topology import (
    AggressiveConditioningConfig,
    LocalTopologyResult,
    _State,
    _allowed_edge_policy,
    _audit_state,
    _boundary_graph_audit,
    _compact,
    _expand_triangle_patch,
    _failure_counts,
    _find_chain_edge,
    _inventory_superthin_components,
    _ledger_counts,
    _lawson_legalize_locked_patch,
    _mesh_area,
    _micro_relax,
    _new_lineage_ids,
    _obc_remap_manifest,
    _ordered_patch_boundary,
    _orient_ccw,
    _patch_perimeter_edges,
    _restricted_edge_violation_records,
    _reconstruct_connectivity_patch_v1,
    _signed_mesh_area,
    _summary,
)
from .metrics import build_edge_topology, chain_edges, triangle_geometry


PLAN_SCHEMA = "fvcom_visual_superthin_repair_plan_v1"
REPORT_SCHEMA = "fvcom_visual_superthin_repair_v1"


@dataclass(frozen=True)
class VisualSuperthinConfig:
    patch_ring_ladder: tuple[int, ...] = (1, 2, 4)
    maximum_support_nodes: int = 2
    maximum_boundary_insertions: int = 2
    micro_relax_cycles: int = 2
    minimum_quality_tolerance: float = 1.0e-9


def validate_visual_plan(plan: dict[str, Any], *, input_sha256: str) -> None:
    """Validate review evidence, input identity, and bounded route syntax."""
    if str(plan.get("schema_version")) != PLAN_SCHEMA:
        raise ValueError(f"visual plan schema must be {PLAN_SCHEMA}")
    recorded_hash = str(plan.get("input_mesh_sha256", "")).upper()
    if recorded_hash != str(input_sha256).upper():
        raise ValueError(
            "stale visual repair plan: input mesh SHA-256 does not match"
        )
    review = plan.get("review")
    if not isinstance(review, dict):
        raise ValueError("visual plan requires a review object")
    if str(review.get("status", "")).lower() != "reviewed":
        raise ValueError("visual plan review.status must be reviewed")
    if not str(review.get("reviewed_by", "")).strip():
        raise ValueError("visual plan requires review.reviewed_by")
    if not str(review.get("observations", "")).strip():
        raise ValueError("visual plan requires review.observations")
    evidence = review.get("visual_evidence")
    if not isinstance(evidence, list) or not evidence or not all(
        str(value).strip() for value in evidence
    ):
        raise ValueError("visual plan requires at least one visual evidence path")
    if review.get("manageable") is not True:
        raise ValueError("visual plan must explicitly mark the residual set manageable")
    component = plan.get("component")
    if not isinstance(component, dict) or not str(component.get("component_id", "")):
        raise ValueError("visual plan requires exactly one component_id")
    actions = plan.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("visual plan requires a non-empty actions list")
    allowed = {
        "constrained_retriangulation",
        "inward_front_support",
        "passage_centerline_support",
        "source_arc_insertion",
        "minmax_cavity_triangulation",
    }
    for action in actions:
        if not isinstance(action, dict) or str(action.get("tool")) not in allowed:
            raise ValueError(f"unsupported visual repair action: {action!r}")
        rings = action.get("patch_rings", [1, 2, 4])
        if not isinstance(rings, list) or not rings or any(
            int(value) not in {1, 2, 3, 4} for value in rings
        ):
            raise ValueError("action patch_rings must be a non-empty subset of 1..4")
        if int(action.get("maximum_support_nodes", 2)) not in {0, 1, 2}:
            raise ValueError("maximum_support_nodes must be 0, 1, or 2")
    acceptance = plan.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError("visual plan requires an acceptance object")
    if acceptance.get("require_strict_superthin_reduction") is not True:
        raise ValueError("visual plans must require strict superthin reduction")


def create_visual_state(
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    fixed_node_mask: np.ndarray,
    constraint_chains: list[list[int]],
    open_boundary_nodes_zero_based: np.ndarray,
    *,
    target_spacing_m: np.ndarray,
    boundary_kinds: list[str] | None = None,
    hard_anchor_mask: np.ndarray | None = None,
    node_lineage: np.ndarray | None = None,
    restricted_lineage_edges: set[tuple[int, int]] | None = None,
) -> tuple[_State, AggressiveConditioningConfig, int]:
    """Create the same audited state used by local-topology conditioning."""
    points = np.asarray(nodes_xy, dtype=float).copy()
    tris = _orient_ccw(points, np.asarray(triangles, dtype=int).copy())
    fixed = np.asarray(fixed_node_mask, dtype=bool).copy()
    targets = np.asarray(target_spacing_m, dtype=float).copy()
    if len(points) != len(fixed) or len(points) != len(targets):
        raise ValueError("points, fixed mask, and targets must have equal length")
    kinds = list(boundary_kinds or ["interior"] * len(points))[: len(points)]
    if len(kinds) < len(points):
        kinds.extend(["interior"] * (len(points) - len(kinds)))
    hard = np.asarray(
        hard_anchor_mask
        if hard_anchor_mask is not None
        else np.zeros(len(points), dtype=bool),
        dtype=bool,
    ).copy()
    lineage = np.asarray(
        node_lineage if node_lineage is not None else np.arange(len(points)),
        dtype=int,
    ).copy()
    chains = [list(map(int, chain)) for chain in constraint_chains]
    open_nodes = np.asarray(open_boundary_nodes_zero_based, dtype=int).copy()
    topology = build_edge_topology(len(points), tris)
    boundary = _boundary_graph_audit(topology)
    protected = chain_edges(chains)
    state = _State(
        points=points,
        triangles=tris,
        fixed=fixed,
        targets=targets,
        chains=chains,
        open_nodes=open_nodes,
        kinds=kinds,
        hard=hard,
        lineage=lineage,
        source_points=points.copy(),
        source_chains=[chain.copy() for chain in chains],
        source_open_nodes=open_nodes.copy(),
        source_kinds=kinds.copy(),
        source_hard_anchor_lineage=lineage[np.where(hard)[0]].astype(int),
        initial_domain_area_m2=max(_signed_mesh_area(points, tris), 1.0e-30),
        initial_boundary_component_count=int(boundary["component_count"]),
        initial_boundary_degree_anomaly_count=int(boundary["degree_anomaly_count"]),
        initial_singly_connected_triangle_count=int(
            np.count_nonzero(topology.triangle_neighbor_count == 1)
        ),
        initial_protected_not_boundary_count=int(
            sum(len(topology.edge_to_triangles.get(edge, [])) != 1 for edge in protected)
        ),
        restricted_lineage_edges={
            tuple(sorted(map(int, edge)))
            for edge in (restricted_lineage_edges or set())
        },
    )
    config = AggressiveConditioningConfig(
        thin_repair_profile="systematic-v5",
        systematic_gate_scope="loop-end",
        systematic_v5_enable_boundary_window_fallback=True,
        enable_pruning=False,
        enable_thin_repair=True,
        enable_valence_repair=False,
        max_rounds=1,
        max_prunes_per_round=0,
        max_valence_removals_per_round=0,
    )
    return state, config, int(len(topology.connected_component_sizes))


def visual_component_inventory(
    state: _State,
    config: AggressiveConditioningConfig,
) -> list[dict[str, Any]]:
    """Return serializable component records in visual-review priority order."""
    return [
        {key: value for key, value in item.items() if key != "triangle_indices"}
        | {"triangle_indices_zero_based": list(map(int, item["triangle_indices"]))}
        for item in _inventory_superthin_components(state, config)
    ]


def apply_visual_superthin_plan(
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    fixed_node_mask: np.ndarray,
    constraint_chains: list[list[int]],
    open_boundary_nodes_zero_based: np.ndarray,
    *,
    target_spacing_m: np.ndarray,
    plan: dict[str, Any],
    boundary_kinds: list[str] | None = None,
    hard_anchor_mask: np.ndarray | None = None,
    node_lineage: np.ndarray | None = None,
    restricted_lineage_edges: set[tuple[int, int]] | None = None,
    visual_config: VisualSuperthinConfig | None = None,
) -> LocalTopologyResult:
    """Apply one reviewed component plan and commit at most one transaction."""
    visual_config = visual_config or VisualSuperthinConfig()
    state, config, initial_components = create_visual_state(
        nodes_xy,
        triangles,
        fixed_node_mask,
        constraint_chains,
        open_boundary_nodes_zero_based,
        target_spacing_m=target_spacing_m,
        boundary_kinds=boundary_kinds,
        hard_anchor_mask=hard_anchor_mask,
        node_lineage=node_lineage,
        restricted_lineage_edges=restricted_lineage_edges,
    )
    before = _summary(state, config)
    before_components = _inventory_superthin_components(state, config)
    component_id = str(plan["component"]["component_id"])
    matching = [item for item in before_components if item["component_id"] == component_id]
    if not matching:
        raise ValueError(f"visual component is absent or stale: {component_id}")
    selected_component = matching[0]
    attempts: list[dict[str, Any]] = []
    best: tuple[tuple[float, ...], _State, dict[str, Any]] | None = None

    for action_index, action in enumerate(plan["actions"]):
        tool = str(action["tool"])
        rings_values = [int(value) for value in action.get("patch_rings", [1, 2, 4])]
        support_limit = min(
            int(visual_config.maximum_support_nodes),
            int(action.get("maximum_support_nodes", visual_config.maximum_support_nodes)),
        )
        topology = build_edge_topology(len(state.points), state.triangles)
        for rings in rings_values:
            patch = _expand_triangle_patch(
                state.triangles,
                topology,
                selected_component["triangle_indices"],
                rings,
            )
            ring = _ordered_patch_boundary(state.triangles, patch)
            if ring is None:
                attempts.append(
                    _attempt_record(
                        action_index,
                        tool,
                        rings,
                        0,
                        ["non_simple_patch_boundary"],
                    )
                )
                continue
            candidates = _action_candidates(
                state,
                selected_component,
                patch,
                ring,
                tool,
                action,
                support_limit,
            )
            for candidate_index, candidate in enumerate(candidates):
                trial = state.clone()
                changed, failures, evidence = _apply_candidate(
                    trial,
                    selected_component,
                    patch,
                    ring,
                    candidate,
                    config,
                )
                record = _attempt_record(
                    action_index,
                    tool,
                    rings,
                    candidate_index,
                    failures,
                    evidence=evidence,
                )
                if not changed:
                    attempts.append(record)
                    continue
                if bool(action.get("local_relaxation", True)):
                    relax_config = config.__class__(
                        **{
                            **asdict(config),
                            "micro_relax_cycles": int(visual_config.micro_relax_cycles),
                        }
                    )
                    _micro_relax(
                        trial,
                        replacement_seed_nodes=trial.last_affected,
                        config=relax_config,
                    )
                elif int(evidence.get("local_superthin_after", 0)) >= int(
                    evidence.get("local_superthin_before", 0)
                ):
                    record["failures"] = ["superthin_debt_not_strictly_reduced"]
                    attempts.append(record)
                    continue
                ok, invariants, after = _audit_state(trial, config, initial_components)
                failures = _visual_acceptance_failures(
                    state,
                    trial,
                    selected_component,
                    before_components,
                    before,
                    after,
                    ok,
                    invariants,
                    tolerance=float(visual_config.minimum_quality_tolerance),
                )
                advisories = _visual_quality_advisories(
                    before,
                    after,
                    tolerance=float(visual_config.minimum_quality_tolerance),
                )
                record.update(
                    {
                        "invariants": invariants,
                        "after": after,
                        "failures": failures,
                        "quality_advisories": advisories,
                    }
                )
                attempts.append(record)
                if failures:
                    continue
                score = (
                    float(after["superthin_triangle_count"]),
                    float(after["superthin_severity_sum"]),
                    -float(after["q_min"]),
                    -float(after["minimum_angle_deg"]),
                    float(after["l_over_h_count_above_1_55"]),
                    float(after["area_transition_count_above_0_50"]),
                    float(candidate.get("support_count", 0)),
                    float(candidate.get("boundary_insertion_count", 0)),
                    float(rings),
                    float(action_index),
                    float(candidate_index),
                )
                record["candidate_score"] = list(score)
                if best is None or score < best[0]:
                    best = (score, trial, record)
        if best is not None:
            break

    accepted = best is not None
    if accepted:
        _, state, selected_record = best
        selected_record["accepted"] = True
        state.ledger.append(
            {
                "operation": "agent-reviewed-visual-superthin-transaction",
                "component_id": component_id,
                "reviewed_by": str(plan["review"]["reviewed_by"]),
                "visual_evidence": list(plan["review"]["visual_evidence"]),
                "tool": str(selected_record["tool"]),
                "patch_rings": int(selected_record["patch_rings"]),
                "candidate_index": int(selected_record["candidate_index"]),
                "superthin_before": int(before["superthin_triangle_count"]),
                "superthin_after": int(_summary(state, config)["superthin_triangle_count"]),
            }
        )
    after = _summary(state, config)
    invariant_ok, invariants, _ = _audit_state(state, config, initial_components)
    remaining = visual_component_inventory(state, config)
    obc = _obc_remap_manifest(state)
    zero = int(after["superthin_triangle_count"]) == 0
    if zero and bool(obc["forcing_invalidation_required"]):
        status = "visual_zero_superthin_pass_forcing_remap_required"
    elif zero:
        status = "visual_zero_superthin_pass"
    elif accepted:
        status = "visual_component_reduced"
    else:
        status = "visual_route_infeasible"
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "accepted": bool(accepted and invariant_ok),
        "review": dict(plan["review"]),
        "component": {key: value for key, value in selected_component.items() if key != "triangle_indices"},
        "settings": asdict(visual_config),
        "before": before,
        "after": after,
        "attempts": attempts,
        "failure_counts": _failure_counts(attempts),
        "quality_advisory_counts": _quality_advisory_counts(attempts),
        "remaining_components": remaining,
        "post_acceptance_atlas_required": bool(accepted),
        "next_visual_plan_requires_new_mesh_hash": bool(accepted),
        "rerank_required_before_next_component": bool(accepted),
        "invariants": invariants,
        "restricted_lineage_edges": [
            list(map(int, edge)) for edge in sorted(state.restricted_lineage_edges)
        ],
        "edit_count": int(len(state.ledger)),
        "edit_counts": _ledger_counts(state.ledger),
        "obc_remap_manifest": obc,
        "fvcom_ready": False,
        "fvcom_ready_reason": "visual experiment does not close valence or the q_l3_sigma target",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    return LocalTopologyResult(
        nodes_xy=state.points,
        triangles=_orient_ccw(state.points, state.triangles),
        fixed_node_mask=state.fixed,
        target_spacing_m=state.targets,
        constraint_chains=[chain.copy() for chain in state.chains],
        open_boundary_nodes_zero_based=state.open_nodes.copy(),
        boundary_kinds=state.kinds.copy(),
        hard_anchor_mask=state.hard.copy(),
        node_lineage=state.lineage.copy(),
        restricted_lineage_edges=set(state.restricted_lineage_edges),
        report=report,
        edit_ledger=[entry.copy() for entry in state.ledger],
        obc_remap_manifest=obc,
    )


def preview_action_points(
    state: _State,
    component: dict[str, Any],
    *,
    patch_rings: int = 2,
) -> dict[str, list[list[float]]]:
    """Return bounded candidate points for atlas overlays without editing."""
    topology = build_edge_topology(len(state.points), state.triangles)
    patch = _expand_triangle_patch(
        state.triangles,
        topology,
        component["triangle_indices"],
        int(patch_rings),
    )
    ring = _ordered_patch_boundary(state.triangles, patch)
    if ring is None:
        return {}
    output: dict[str, list[list[float]]] = {}
    for tool in ("inward_front_support", "passage_centerline_support", "source_arc_insertion"):
        candidates = _action_candidates(
            state,
            component,
            patch,
            ring,
            tool,
            {"tool": tool},
            2,
        )
        points: list[list[float]] = []
        for candidate in candidates:
            points.extend(
                [list(map(float, value)) for value in candidate.get("support_points", [])]
            )
            points.extend(
                [list(map(float, value["coordinate"])) for value in candidate.get("boundary_insertions", [])]
            )
        unique: list[list[float]] = []
        for point in points:
            if not any(np.linalg.norm(np.asarray(point) - np.asarray(old)) < 1.0e-8 for old in unique):
                unique.append(point)
        output[tool] = unique[:12]
    return output


def _action_candidates(
    state: _State,
    component: dict[str, Any],
    patch: np.ndarray,
    ring: list[int],
    tool: str,
    action: dict[str, Any],
    support_limit: int,
) -> list[dict[str, Any]]:
    polygon = Polygon(state.points[np.asarray(ring, dtype=int)])
    representative = np.asarray(
        [polygon.representative_point().x, polygon.representative_point().y],
        dtype=float,
    )
    if tool == "constrained_retriangulation":
        return [
            {
                "support_points": [],
                "support_count": 0,
                "boundary_insertions": [],
                "boundary_insertion_count": 0,
                "remove_movable_component_nodes": bool(
                    action.get("remove_movable_component_nodes", True)
                ),
            }
        ]
    if tool == "minmax_cavity_triangulation":
        return [
            {
                "triangulation_method": "protected_chord_quality_ear",
                "support_points": [],
                "support_count": 0,
                "boundary_insertions": [],
                "boundary_insertion_count": 0,
                "remove_movable_component_nodes": False,
            }
        ]
    if tool == "inward_front_support":
        protected = chain_edges(state.chains)
        protected_component_edges = {
            edge
            for triangle_index in component["triangle_indices"]
            for edge in _triangle_edges(state.triangles[int(triangle_index)])
            if edge in protected
        }
        front_specs: list[tuple[tuple[int, int], np.ndarray, bool]] = []
        if str(action.get("candidate_geometry", "protected_front")) == (
            "superthin_longest_edges"
        ):
            geometry = triangle_geometry(
                state.points,
                state.triangles[
                    np.asarray(component["triangle_indices"], dtype=int)
                ],
            )
            ordering = sorted(
                range(len(component["triangle_indices"])),
                key=lambda index: (
                    float(geometry["quality"][index]),
                    float(np.min(geometry["angles_deg"][index])),
                    int(index),
                ),
            )
            for local_index in ordering:
                triangle = state.triangles[
                    int(component["triangle_indices"][local_index])
                ]
                edges = _triangle_edges(triangle)
                edge = max(
                    edges,
                    key=lambda value: (
                        float(
                            np.linalg.norm(
                                state.points[value[1]] - state.points[value[0]]
                            )
                        ),
                        value,
                    ),
                )
                opposite = next(
                    int(value) for value in triangle if int(value) not in edge
                )
                front_specs.append(
                    (edge, state.points[opposite].copy(), True)
                )
        known_edges = {edge for edge, _, _ in front_specs}
        front_specs.extend(
            (edge, representative.copy(), False)
            for edge in sorted(protected_component_edges)
            if edge not in known_edges
        )
        fractions = [
            float(value)
            for value in action.get("height_fractions", [0.35, 0.50, 0.75, 0.8660254])
        ]
        raw: list[Any] = []
        connect_front = str(
            action.get("candidate_geometry", "protected_front")
        ) == "superthin_longest_edges"
        for edge, toward, extend_past_opposite in front_specs:
            midpoint = 0.5 * (state.points[edge[0]] + state.points[edge[1]])
            direction = toward - midpoint
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-12:
                continue
            length = float(np.linalg.norm(state.points[edge[1]] - state.points[edge[0]]))
            for fraction in fractions:
                distance = float(fraction) * length
                if not extend_past_opposite:
                    distance = min(distance, 0.90 * norm)
                point = midpoint + distance * direction / norm
                if _strictly_inside(polygon, point):
                    if connect_front:
                        raw.append(
                            {
                                "point": point,
                                "connect_to": list(map(int, edge)),
                            }
                        )
                    else:
                        raw.append(point)
        return _support_candidate_sets(raw, support_limit, action)
    if tool == "passage_centerline_support":
        raw = _passage_midpoints(state, component, patch, polygon)
        return _support_candidate_sets(raw, support_limit, action)
    if tool == "source_arc_insertion":
        candidates: list[dict[str, Any]] = []
        protected = chain_edges(state.chains)
        component_edges = {
            edge
            for triangle_index in component["triangle_indices"]
            for edge in _triangle_edges(state.triangles[int(triangle_index)])
            if edge in protected and _find_chain_edge(state.chains, edge) is not None
        }
        for edge in sorted(component_edges)[:2]:
            point = 0.5 * (state.points[edge[0]] + state.points[edge[1]])
            direction = representative - point
            norm = float(np.linalg.norm(direction))
            support: list[list[float]] = []
            if norm > 1.0e-12 and support_limit > 0:
                height = 0.50 * float(np.linalg.norm(state.points[edge[1]] - state.points[edge[0]]))
                inside = point + min(height, 0.85 * norm) * direction / norm
                if _strictly_inside(polygon, inside):
                    support = [list(map(float, inside))]
            candidates.append(
                {
                    "support_points": support,
                    "support_count": len(support),
                    "boundary_insertions": [
                        {"edge": list(map(int, edge)), "coordinate": list(map(float, point))}
                    ],
                    "boundary_insertion_count": 1,
                    "remove_movable_component_nodes": bool(
                        action.get("remove_movable_component_nodes", False)
                    ),
                }
            )
        return candidates
    return []


def _support_candidate_sets(
    raw: Iterable[Any],
    limit: int,
    action: dict[str, Any],
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            value = np.asarray(item["point"], dtype=float)
            connect_to = list(map(int, item.get("connect_to", [])))
        else:
            value = np.asarray(item, dtype=float)
            connect_to = []
        if not any(
            float(np.linalg.norm(value - old["point"])) <= 1.0e-7
            for old in unique
        ):
            unique.append({"point": value, "connect_to": connect_to})
    point_limit = max(
        1,
        int(action.get("maximum_candidate_points", max(8, int(limit) * 4))),
    )
    unique = unique[:point_limit]
    groups: list[list[np.ndarray]] = []
    if int(limit) == 0:
        groups = [[]]
    else:
        groups.extend([[value] for value in unique])
        if int(limit) >= 2:
            pair_limit = max(0, int(action.get("maximum_pair_candidates", 12)))
            groups.extend(
                [
                    list(pair)
                    for pair in list(combinations(unique[:8], 2))[:pair_limit]
                ]
            )
    return [
        {
            "support_points": [
                list(map(float, value["point"])) for value in group
            ],
            "support_connect_to": [
                (
                    list(map(int, value["connect_to"]))
                    if bool(action.get("lock_support_spokes", True))
                    else []
                )
                for value in group
            ],
            "support_count": len(group),
            "boundary_insertions": [],
            "boundary_insertion_count": 0,
            "remove_movable_component_nodes": bool(
                action.get("remove_movable_component_nodes", True)
            ),
        }
        for group in groups
    ]


def _passage_midpoints(
    state: _State,
    component: dict[str, Any],
    patch: np.ndarray,
    polygon: Polygon,
) -> list[Any]:
    chain_ids = list(map(int, component.get("boundary_chain_ids", [])))
    patch_nodes = set(map(int, np.unique(state.triangles[np.asarray(patch, dtype=int)])))
    local_chain_ids = [
        int(chain_index)
        for chain_index, chain in enumerate(state.chains)
        if any(int(node) in patch_nodes for node in chain)
    ]
    for chain_index in local_chain_ids:
        if chain_index not in chain_ids:
            chain_ids.append(chain_index)
    if len(chain_ids) < 2:
        return []
    membership: dict[int, set[int]] = {}
    for chain_index in chain_ids:
        for node in state.chains[chain_index]:
            if int(node) in patch_nodes:
                membership.setdefault(int(node), set()).add(int(chain_index))
    pairs: list[tuple[float, int, int, np.ndarray]] = []
    nodes = sorted(membership)
    for offset, left in enumerate(nodes):
        for right in nodes[offset + 1 :]:
            if not membership[left].isdisjoint(membership[right]):
                continue
            midpoint = 0.5 * (state.points[left] + state.points[right])
            if not _strictly_inside(polygon, midpoint):
                continue
            pairs.append(
                (
                    float(np.linalg.norm(state.points[left] - state.points[right])),
                    int(left),
                    int(right),
                    midpoint,
                )
            )
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for _, left, right, midpoint in sorted(pairs, key=lambda value: (value[0], value[1], value[2])):
        if left in used and right in used:
            continue
        if any(
            float(np.linalg.norm(midpoint - old["point"])) <= 1.0e-7
            for old in selected
        ):
            continue
        selected.append(
            {
                "point": midpoint,
                "connect_to": [int(left), int(right)],
            }
        )
        used.update((left, right))
    return selected[:8]


def _apply_candidate(
    state: _State,
    component: dict[str, Any],
    patch: np.ndarray,
    ring: list[int],
    candidate: dict[str, Any],
    config: AggressiveConditioningConfig,
) -> tuple[bool, list[str], dict[str, Any]]:
    if str(candidate.get("triangulation_method", "")) == (
        "protected_chord_quality_ear"
    ):
        forbidden = next(
            iter(sorted(state.restricted_lineage_edges)),
            (-10_000_002, -10_000_001),
        )
        changed, failures, evidence = _reconstruct_connectivity_patch_v1(
            state,
            patch,
            forbidden_lineage_edge=forbidden,
            config=config,
        )
        if changed:
            before_debt = evidence.get("local_debt_before", {})
            after_debt = evidence.get("local_debt_after", {})
            evidence.update(
                {
                    "support_count": 0,
                    "boundary_insertion_count": 0,
                    "local_superthin_before": int(
                        before_debt.get("superthin_triangle_count", 0)
                    ),
                    "local_superthin_after": int(
                        after_debt.get("superthin_triangle_count", 0)
                    ),
                    "triangulation_method": "protected_chord_quality_ear",
                }
            )
        return changed, failures, evidence
    old_triangles = state.triangles[np.asarray(patch, dtype=int)].copy()
    old_area = float(np.sum(triangle_geometry(state.points, old_triangles)["area"]))
    original_perimeter = _patch_perimeter_edges(state.triangles, patch)
    inserted_boundary_nodes: list[int] = []
    replaced_perimeter = set(original_perimeter)
    for item in candidate.get("boundary_insertions", []):
        edge = tuple(sorted(map(int, item["edge"])))
        node = _append_node(
            state,
            np.asarray(item["coordinate"], dtype=float),
            fixed=True,
            kind=str(state.kinds[edge[0]]),
            hard=False,
            target=float(0.5 * (state.targets[edge[0]] + state.targets[edge[1]])),
        )
        membership = _find_chain_edge(state.chains, edge)
        if membership is None:
            return False, ["source_arc_edge_not_in_chain"], {}
        chain_index, position = membership
        chain = state.chains[int(chain_index)]
        left = int(chain[position])
        right = int(chain[(position + 1) % len(chain)])
        chain.insert(int(position + 1), int(node))
        _insert_open_node(state, left, right, node)
        replaced_perimeter.discard(edge)
        if edge in original_perimeter:
            replaced_perimeter.update(
                {tuple(sorted((left, node))), tuple(sorted((node, right)))}
            )
        inserted_boundary_nodes.append(int(node))
    support_nodes = [
        _append_node(
            state,
            np.asarray(point, dtype=float),
            fixed=False,
            kind="interior",
            hard=False,
            target=_support_target(state, component),
        )
        for point in candidate.get("support_points", [])
    ]
    patch_nodes = set(map(int, np.unique(old_triangles)))
    component_nodes = set(
        map(
            int,
            np.unique(
                state.triangles[np.asarray(component["triangle_indices"], dtype=int)]
            ),
        )
    )
    removable: set[int] = set()
    if bool(candidate.get("remove_movable_component_nodes", True)):
        patch_set = set(map(int, np.asarray(patch, dtype=int)))
        for node in component_nodes:
            if state.fixed[node] or node in set(ring):
                continue
            incident = set(map(int, np.where(np.any(state.triangles == int(node), axis=1))[0]))
            if incident.issubset(patch_set):
                removable.add(int(node))
    retained = sorted(patch_nodes - removable)
    extra = [*inserted_boundary_nodes, *support_nodes]
    vertex_nodes = [*retained, *extra]
    if len(vertex_nodes) < 3 or len(set(vertex_nodes)) != len(vertex_nodes):
        return False, ["invalid_candidate_node_set"], {}
    lookup = {node: index for index, node in enumerate(vertex_nodes)}
    segments = set(replaced_perimeter)
    support_connect_to = list(candidate.get("support_connect_to", []))
    for support_node, endpoints in zip(support_nodes, support_connect_to):
        for endpoint in map(int, endpoints):
            if endpoint in retained:
                segments.add(tuple(sorted((int(support_node), endpoint))))
    protected = chain_edges(state.chains)
    old_edges = _edge_set(old_triangles)
    segments.update(
        edge for edge in protected if edge in old_edges or any(node in edge for node in inserted_boundary_nodes)
    )
    missing_segment_nodes = [edge for edge in segments if edge[0] not in lookup or edge[1] not in lookup]
    if missing_segment_nodes:
        return False, ["candidate_segment_node_missing"], {"missing_segments": missing_segment_nodes[:20]}
    local_segments = np.asarray([[lookup[a], lookup[b]] for a, b in sorted(segments)], dtype=int)
    data = {
        "vertices": state.points[np.asarray(vertex_nodes, dtype=int)],
        "segments": local_segments,
    }
    try:
        output = tr.triangulate(data, "pQY")
    except Exception as exc:  # pragma: no cover - library-specific diagnostics
        return False, ["constrained_triangle_exception"], {"exception": str(exc)}
    if "triangles" not in output or "vertices" not in output:
        return False, ["constrained_triangle_empty"], {}
    out_vertices = np.asarray(output["vertices"], dtype=float)
    if len(out_vertices) != len(vertex_nodes) or not np.allclose(
        out_vertices,
        data["vertices"],
        rtol=0.0,
        atol=1.0e-9,
    ):
        return False, ["unexpected_triangle_steiner_or_vertex_change"], {
            "input_vertex_count": len(vertex_nodes),
            "output_vertex_count": len(out_vertices),
        }
    replacement = np.asarray(vertex_nodes, dtype=int)[np.asarray(output["triangles"], dtype=int)]
    replacement = _orient_ccw(state.points, replacement)
    geometry = triangle_geometry(state.points, replacement)
    if np.any(geometry["signed_area"] <= 0.0):
        return False, ["nonpositive_candidate_triangle"], {}
    new_area = float(np.sum(geometry["area"]))
    if abs(new_area - old_area) > 1.0e-8 * max(old_area, 1.0):
        return False, ["visual_patch_area_mismatch"], {
            "old_area_m2": old_area,
            "new_area_m2": new_area,
        }
    policy = _allowed_edge_policy(state, config)
    replacement, forced_flip_count, unremovable_illegal = (
        _force_remove_restricted_replacement_edges(
            state,
            replacement,
            locked_edges=segments,
            policy=policy,
            max_flips=16,
        )
    )
    if unremovable_illegal:
        return False, ["visual_candidate_uses_restricted_edge"], {
            "illegal_edges": [
                list(map(int, value)) for value in unremovable_illegal[:20]
            ],
            "forced_restricted_flip_count": int(forced_flip_count),
        }
    replacement_edges = _edge_set(replacement)
    if not segments.issubset(replacement_edges):
        return False, ["visual_constrained_segment_missing"], {}
    illegal = [
        edge
        for edge in replacement_edges
        if not policy.is_allowed(edge, reject_same_chain_shortcuts=False)
    ]
    if illegal:
        return False, ["visual_candidate_uses_restricted_edge"], {
            "illegal_edges": [list(map(int, value)) for value in illegal[:20]]
        }
    keep = np.ones(len(state.triangles), dtype=bool)
    keep[np.asarray(patch, dtype=int)] = False
    outside_count = int(np.count_nonzero(keep))
    state.triangles = _orient_ccw(
        state.points,
        np.vstack([state.triangles[keep], replacement]),
    )
    replacement_ids = set(
        range(outside_count, outside_count + len(replacement))
    )
    lawson_flip_count = _lawson_legalize_locked_patch(
        state,
        replacement_ids,
        segments | protected,
        max_flips=128,
        edge_allowed=lambda edge: policy.is_allowed(
            edge,
            reject_same_chain_shortcuts=False,
        ),
    )
    post_legalization_illegal = [
        edge
        for edge in _edge_set(
            state.triangles[np.asarray(sorted(replacement_ids), dtype=int)]
        )
        if not policy.is_allowed(edge, reject_same_chain_shortcuts=False)
    ]
    if post_legalization_illegal:
        return False, ["restricted_edge_present_after_legalization"], {
            "illegal_edges": [
                list(map(int, value)) for value in post_legalization_illegal[:20]
            ],
            "forced_restricted_flip_count": int(forced_flip_count),
            "lawson_flip_count": int(lawson_flip_count),
        }
    delivered_replacement = state.triangles[
        np.asarray(sorted(replacement_ids), dtype=int)
    ]
    local_superthin_before = _superthin_count(old_triangles, state.points, config)
    local_superthin_after = _superthin_count(
        delivered_replacement,
        state.points,
        config,
    )
    state.last_affected = sorted(set(map(int, np.unique(delivered_replacement))))
    state.ledger.append(
        {
            "operation": "visual-constrained-patch-reconstruction",
            "component_id": str(component["component_id"]),
            "patch_triangle_count": int(len(patch)),
            "replacement_triangle_count": int(len(replacement)),
            "removed_movable_node_count": int(len(removable)),
            "inserted_support_node_count": int(len(support_nodes)),
            "inserted_boundary_node_count": int(len(inserted_boundary_nodes)),
            "locked_support_spoke_count": int(
                sum(len(value) for value in support_connect_to)
            ),
            "forced_restricted_flip_count": int(forced_flip_count),
            "lawson_flip_count": int(lawson_flip_count),
            "local_superthin_before": int(local_superthin_before),
            "local_superthin_after": int(local_superthin_after),
        }
    )
    _compact(state)
    return True, [], {
        "support_count": int(len(support_nodes)),
        "boundary_insertion_count": int(len(inserted_boundary_nodes)),
        "locked_support_spoke_count": int(
            sum(len(value) for value in support_connect_to)
        ),
        "removed_movable_node_count": int(len(removable)),
        "replacement_triangle_count": int(len(replacement)),
        "forced_restricted_flip_count": int(forced_flip_count),
        "lawson_flip_count": int(lawson_flip_count),
        "local_superthin_before": int(local_superthin_before),
        "local_superthin_after": int(local_superthin_after),
        "old_area_m2": old_area,
        "new_area_m2": new_area,
    }


def _superthin_count(
    triangles: np.ndarray,
    points: np.ndarray,
    config: AggressiveConditioningConfig,
) -> int:
    geometry = triangle_geometry(points, np.asarray(triangles, dtype=int))
    return int(
        np.count_nonzero(
            (geometry["quality"] < float(config.superthin_quality_threshold))
            | (
                np.min(geometry["angles_deg"], axis=1)
                < float(config.superthin_min_angle_deg)
            )
        )
    )


def _force_remove_restricted_replacement_edges(
    state: _State,
    replacement: np.ndarray,
    *,
    locked_edges: set[tuple[int, int]],
    policy: Any,
    max_flips: int,
) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    """Flip forbidden local diagonals even when they are Delaunay-preferred."""
    delivered = _orient_ccw(state.points, np.asarray(replacement, dtype=int).copy())
    flips = 0
    for _ in range(max(0, int(max_flips))):
        edge_to_triangles: dict[tuple[int, int], list[int]] = {}
        for triangle_index, triangle in enumerate(delivered):
            for edge in _triangle_edges(triangle):
                edge_to_triangles.setdefault(edge, []).append(int(triangle_index))
        illegal = sorted(
            edge
            for edge in edge_to_triangles
            if not policy.is_allowed(edge, reject_same_chain_shortcuts=False)
        )
        if not illegal:
            return delivered, int(flips), []
        changed = False
        for edge in illegal:
            attached = edge_to_triangles.get(edge, [])
            if edge in locked_edges or len(attached) != 2:
                continue
            first, second = map(int, attached)
            c_values = [
                int(value)
                for value in delivered[first]
                if int(value) not in edge
            ]
            d_values = [
                int(value)
                for value in delivered[second]
                if int(value) not in edge
            ]
            if (
                len(c_values) != 1
                or len(d_values) != 1
                or c_values[0] == d_values[0]
            ):
                continue
            c, d = c_values[0], d_values[0]
            alternative = tuple(sorted((c, d)))
            if (
                alternative in edge_to_triangles
                or alternative in locked_edges
                or not policy.is_allowed(
                    alternative,
                    reject_same_chain_shortcuts=False,
                )
            ):
                continue
            pair = _orient_ccw(
                state.points,
                np.asarray(
                    [[c, d, int(edge[0])], [d, c, int(edge[1])]],
                    dtype=int,
                ),
            )
            pair_geometry = triangle_geometry(state.points, pair)
            if np.any(pair_geometry["signed_area"] <= 0.0):
                continue
            delivered[first] = pair[0]
            delivered[second] = pair[1]
            flips += 1
            changed = True
            break
        if not changed:
            return delivered, int(flips), illegal
    remaining = sorted(
        edge
        for edge in _edge_set(delivered)
        if not policy.is_allowed(edge, reject_same_chain_shortcuts=False)
    )
    return delivered, int(flips), remaining


def _visual_acceptance_failures(
    before_state: _State,
    after_state: _State,
    selected_component: dict[str, Any],
    before_components: list[dict[str, Any]],
    before: dict[str, Any],
    after: dict[str, Any],
    invariant_ok: bool,
    invariants: dict[str, Any],
    *,
    tolerance: float,
) -> list[str]:
    failures: list[str] = []
    if not invariant_ok:
        failures.extend(
            key
            for key, value in invariants.items()
            if (isinstance(value, bool) and not value)
            or (key.endswith("_count") and isinstance(value, int) and value > 0 and key in {
                "nonmanifold_edge_count",
                "unused_node_count",
                "new_singly_connected_triangle_count",
                "duplicate_triangle_count",
                "repeated_node_triangle_count",
                "missing_hard_anchor_count",
                "moved_hard_anchor_count",
                "restricted_edge_violation_count",
            })
        )
    if not (
        int(after["superthin_triangle_count"]) < int(before["superthin_triangle_count"])
        and float(after["superthin_severity_sum"]) <= float(before["superthin_severity_sum"]) + tolerance
    ):
        failures.append("superthin_debt_not_strictly_reduced")
    if _restricted_edge_violation_records(after_state):
        failures.append("restricted_edge_violation")
    before_fixed = {
        int(lineage): before_state.points[index].copy()
        for index, lineage in enumerate(before_state.lineage)
        if bool(before_state.fixed[index]) and int(lineage) >= 0
    }
    after_by_lineage = {
        int(lineage): after_state.points[index]
        for index, lineage in enumerate(after_state.lineage)
        if int(lineage) >= 0
    }
    if any(
        lineage not in after_by_lineage
        or not np.array_equal(point, after_by_lineage[lineage])
        for lineage, point in before_fixed.items()
    ):
        failures.append("existing_boundary_coordinate_changed")
    source_open = list(map(int, before_state.lineage[before_state.open_nodes]))
    delivered_open = list(map(int, after_state.lineage[after_state.open_nodes]))
    retained = [value for value in delivered_open if value in set(source_open)]
    if retained != source_open:
        failures.append("original_obc_order_or_membership_changed")
    selected_lineage = set(map(int, selected_component["node_lineage"]))
    before_unselected_ids = {
        str(item["component_id"])
        for item in before_components
        if str(item["component_id"]) != str(selected_component["component_id"])
    }
    after_components = _inventory_superthin_components(
        after_state,
        AggressiveConditioningConfig(thin_repair_profile="systematic-v5"),
    )
    for item in after_components:
        lineage = set(map(int, item["node_lineage"]))
        if lineage.isdisjoint(selected_lineage) and str(item["component_id"]) not in before_unselected_ids:
            failures.append("new_superthin_component_elsewhere")
            break
    return sorted(set(failures))


def _visual_quality_advisories(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    tolerance: float,
) -> list[str]:
    """Record nonblocking Class-2/3 changes for a visual transaction.

    The benchmark-first policy permits these representation-dependent tails to
    move in either direction.  Exact structure, valence, superthin debt,
    constraints, OBC lineage, and boundary-coordinate checks remain blocking in
    ``_visual_acceptance_failures``.
    """
    findings: list[str] = []
    guarded = (
        ("q_min", tolerance),
        ("q_p01", tolerance),
        ("q_l3_sigma", tolerance),
        ("minimum_angle_p01_deg", 1.0e-3),
    )
    for key, slack in guarded:
        if float(after[key]) + float(slack) < float(before[key]):
            findings.append(f"{key}_regression")
    for key in (
        "l_over_h_count_above_1_55",
        "area_transition_count_above_0_50",
    ):
        if int(after[key]) > int(before[key]):
            findings.append(f"{key}_regression")
    return sorted(set(findings))


def _quality_advisory_counts(attempts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attempt in attempts:
        for finding in attempt.get("quality_advisories", []):
            key = str(finding)
            counts[key] = int(counts.get(key, 0) + 1)
    return dict(sorted(counts.items()))


def _append_node(
    state: _State,
    coordinate: np.ndarray,
    *,
    fixed: bool,
    kind: str,
    hard: bool,
    target: float,
) -> int:
    node = len(state.points)
    state.points = np.vstack([state.points, np.asarray(coordinate, dtype=float)])
    state.fixed = np.concatenate([state.fixed, np.asarray([bool(fixed)], dtype=bool)])
    state.targets = np.concatenate([state.targets, np.asarray([float(target)], dtype=float)])
    state.kinds.append(str(kind))
    state.hard = np.concatenate([state.hard, np.asarray([bool(hard)], dtype=bool)])
    state.lineage = np.concatenate([state.lineage, _new_lineage_ids(state, 1)])
    return int(node)


def _insert_open_node(state: _State, left: int, right: int, node: int) -> None:
    values = list(map(int, state.open_nodes))
    for index, (a, b) in enumerate(zip(values[:-1], values[1:])):
        if (a, b) == (left, right):
            values.insert(index + 1, int(node))
            state.open_nodes = np.asarray(values, dtype=int)
            return
        if (a, b) == (right, left):
            values.insert(index + 1, int(node))
            state.open_nodes = np.asarray(values, dtype=int)
            return


def _support_target(state: _State, component: dict[str, Any]) -> float:
    value = component.get("local_feature_target_m")
    if value is not None and np.isfinite(float(value)) and float(value) > 0.0:
        return float(value)
    nodes = np.asarray(
        sorted(
            set(
                map(
                    int,
                    np.unique(
                        state.triangles[np.asarray(component["triangle_indices"], dtype=int)]
                    ),
                )
            )
        ),
        dtype=int,
    )
    return float(np.median(state.targets[nodes]))


def _strictly_inside(polygon: Polygon, point: np.ndarray) -> bool:
    return bool(
        contains_xy(
            polygon,
            np.asarray([float(point[0])]),
            np.asarray([float(point[1])]),
        )[0]
    )


def _triangle_edges(triangle: Iterable[int]) -> list[tuple[int, int]]:
    a, b, c = map(int, triangle)
    return [tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))]


def _edge_set(triangles: np.ndarray) -> set[tuple[int, int]]:
    return {
        edge
        for triangle in np.asarray(triangles, dtype=int)
        for edge in _triangle_edges(triangle)
    }


def _attempt_record(
    action_index: int,
    tool: str,
    rings: int,
    candidate_index: int,
    failures: list[str],
    *,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "action_index": int(action_index),
        "tool": str(tool),
        "patch_rings": int(rings),
        "candidate_index": int(candidate_index),
        "accepted": False,
        "failures": list(map(str, failures)),
        "evidence": dict(evidence or {}),
    }
