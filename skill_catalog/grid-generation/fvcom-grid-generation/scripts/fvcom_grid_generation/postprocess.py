"""Constraint-preserving OceanMesh-style post-generation refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .metrics import build_edge_topology, chain_edges, compute_mesh_metrics, triangle_geometry


ProgressCallback = Callable[[str, float, dict[str, Any] | None], None]


@dataclass(frozen=True)
class PostprocessConfig:
    profile: str = "rpw2019"
    boundary_policy: str = "protect-all"
    max_passes: int = 8
    connectivity_limit: int | None = None
    boundary_quality_cutoff: float = 0.25
    disjoint_area_fraction: float = 0.25
    max_flip_passes: int = 6
    max_cavity_removals: int = 24
    max_interior_collapses: int = 12


@dataclass
class PostprocessResult:
    nodes_xy: np.ndarray
    triangles: np.ndarray
    fixed_node_mask: np.ndarray
    constraint_chains: list[list[int]]
    open_boundary_nodes: np.ndarray
    report: dict[str, Any]
    history: list[dict[str, Any]]


@dataclass
class _MeshState:
    points: np.ndarray
    triangles: np.ndarray
    fixed: np.ndarray
    chains: list[list[int]]
    open_nodes: list[int]


def postprocess_mesh(
    nodes_xy: np.ndarray,
    triangles_1based: np.ndarray,
    fixed_node_mask: np.ndarray,
    constraint_chains: list[list[int]],
    open_boundary_nodes_1based: np.ndarray,
    config: PostprocessConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PostprocessResult:
    """Apply a clean-room cleanup profile while preserving protected boundaries."""
    config = config or PostprocessConfig()
    if config.profile not in {"none", "rpw2019", "projection-medium"}:
        raise ValueError(f"Unsupported postprocess profile: {config.profile}")
    if config.boundary_policy not in {"protect-all", "protect-open"}:
        raise ValueError(f"Unsupported boundary policy: {config.boundary_policy}")
    connectivity_limit = int(
        config.connectivity_limit
        if config.connectivity_limit is not None
        else (6 if config.profile == "rpw2019" else 8)
    )
    state = _MeshState(
        points=np.asarray(nodes_xy, dtype=float).copy(),
        triangles=np.asarray(triangles_1based, dtype=int).copy() - 1,
        fixed=np.asarray(fixed_node_mask, dtype=bool).copy(),
        chains=[list(map(int, chain)) for chain in constraint_chains],
        open_nodes=(np.asarray(open_boundary_nodes_1based, dtype=int) - 1).tolist(),
    )
    if config.boundary_policy == "protect-open":
        open_set = set(state.open_nodes)
        state.fixed = np.asarray([index in open_set for index in range(len(state.points))], dtype=bool)
    original_boundary_coordinates = [state.points[np.asarray(chain, dtype=int)].copy() for chain in state.chains]
    history: list[dict[str, Any]] = []
    before = compute_mesh_metrics(
        state.points,
        state.triangles,
        constraint_chains=state.chains,
        open_boundary_nodes_zero_based=state.open_nodes,
    )

    if config.profile == "none":
        report = _build_report(config, connectivity_limit, before, before, original_boundary_coordinates, state, "disabled", history)
        return _result(state, report, history)

    _progress(progress_callback, "fix_consistency", 0.05, None)
    outcome = _execute_guarded(state, lambda: _fix_consistency(state), "consistency")
    _record_outcome(history, "fix_consistency", outcome)

    stop_reason = "profile_complete"
    if config.profile == "rpw2019":
        _progress(progress_callback, "make_boundaries_traversable", 0.20, None)
        outcome = _execute_guarded(
            state,
            lambda: _repair_boundary_quality_by_flips(state, config, purpose="traversability"),
            "traversability",
        )
        _record_outcome(history, "make_boundaries_traversable", outcome)

        _progress(progress_callback, "repair_singly_connected", 0.40, None)
        def _repair_singly() -> int:
            operations = _remove_unprotected_singly(state, max_rounds=8)
            return operations + _repair_boundary_quality_by_flips(state, config, purpose="singly_connected")

        outcome = _execute_guarded(state, _repair_singly, "singly_connected")
        _record_outcome(history, "repair_singly_connected", outcome)

        _progress(progress_callback, "bound_connectivity", 0.62, {"limit": connectivity_limit})
        outcome = _execute_guarded(
            state,
            lambda: _bound_connectivity(state, connectivity_limit, config),
            "connectivity",
        )
        _record_outcome(history, "bound_connectivity", outcome)

        _progress(progress_callback, "direct_implicit_smoothing", 0.82, None)
        outcome = _execute_guarded(state, lambda: _direct_implicit_smooth(state), "smoothing")
        _record_outcome(history, "direct_implicit_smoothing", outcome)
    else:
        for pass_index in range(max(1, int(config.max_passes))):
            pass_changes = 0
            _progress(
                progress_callback,
                "projection_medium_pass",
                0.10 + 0.80 * pass_index / max(int(config.max_passes), 1),
                {"pass": pass_index + 1},
            )
            outcome = _execute_guarded(
                state,
                lambda: _repair_boundary_quality_by_flips(state, config, purpose="poor_boundary"),
                "quality_tail",
            )
            pass_changes += outcome["committed_operations"]
            _record_outcome(history, f"pass_{pass_index + 1}_poor_boundary_repair", outcome)

            outcome = _execute_guarded(state, lambda: _collapse_interior_thin_edges(state, config), "quality_tail")
            pass_changes += outcome["committed_operations"]
            _record_outcome(history, f"pass_{pass_index + 1}_interior_thin_collapse", outcome)

            outcome = _execute_guarded(
                state,
                lambda: _repair_boundary_quality_by_flips(state, config, purpose="traversability"),
                "traversability",
            )
            pass_changes += outcome["committed_operations"]
            _record_outcome(history, f"pass_{pass_index + 1}_traversability", outcome)

            outcome = _execute_guarded(
                state,
                lambda: _bound_connectivity(state, connectivity_limit, config),
                "connectivity",
            )
            pass_changes += outcome["committed_operations"]
            _record_outcome(history, f"pass_{pass_index + 1}_connectivity", outcome)

            outcome = _execute_guarded(state, lambda: _direct_implicit_smooth(state), "smoothing")
            pass_changes += outcome["committed_operations"]
            _record_outcome(history, f"pass_{pass_index + 1}_smoothing", outcome)
            current_q = history[-1]["after_full"]["oceanmesh_quality"]["q_min"]
            if current_q >= config.boundary_quality_cutoff:
                stop_reason = "minimum_quality_target_reached"
                break
            if pass_changes == 0:
                stop_reason = "protected_defect_or_no_change"
                break
        else:
            stop_reason = "maximum_passes_reached"

    _progress(progress_callback, "postprocess_complete", 1.0, {"stop_reason": stop_reason})
    after = _metrics(state)
    report = _build_report(
        config,
        connectivity_limit,
        before,
        after,
        original_boundary_coordinates,
        state,
        stop_reason,
        history,
    )
    return _result(state, report, history)


def boundary_chains_from_mesh(triangles_1based: np.ndarray) -> list[list[int]]:
    """Extract deterministic boundary loops from a 1-based triangulation."""
    triangles = np.asarray(triangles_1based, dtype=int) - 1
    topology = build_edge_topology(int(np.max(triangles)) + 1 if len(triangles) else 0, triangles)
    adjacency: dict[int, list[int]] = {}
    for a, b in topology.boundary_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    remaining = {tuple(sorted(edge)) for edge in topology.boundary_edges}
    chains: list[list[int]] = []
    while remaining:
        first = min(remaining)
        start, current = first
        chain = [start]
        previous = -1
        while True:
            chain.append(current)
            remaining.discard(tuple(sorted((chain[-2], current))))
            candidates = [value for value in sorted(adjacency.get(current, [])) if value != previous]
            next_node = next((value for value in candidates if tuple(sorted((current, value))) in remaining), None)
            if next_node is None or next_node == start:
                if next_node == start:
                    remaining.discard(tuple(sorted((current, start))))
                break
            previous, current = current, next_node
        if len(chain) > 2:
            chains.append(chain)
    return chains


def _result(state: _MeshState, report: dict[str, Any], history: list[dict[str, Any]]) -> PostprocessResult:
    return PostprocessResult(
        nodes_xy=state.points,
        triangles=state.triangles + 1,
        fixed_node_mask=state.fixed,
        constraint_chains=state.chains,
        open_boundary_nodes=np.asarray(state.open_nodes, dtype=int) + 1,
        report=report,
        history=history,
    )


def _metrics(state: _MeshState) -> dict[str, Any]:
    return compute_mesh_metrics(
        state.points,
        state.triangles,
        constraint_chains=state.chains,
        open_boundary_nodes_zero_based=state.open_nodes,
    )


def _clone_state(state: _MeshState) -> _MeshState:
    return _MeshState(
        points=state.points.copy(),
        triangles=state.triangles.copy(),
        fixed=state.fixed.copy(),
        chains=[chain.copy() for chain in state.chains],
        open_nodes=state.open_nodes.copy(),
    )


def _restore_state(state: _MeshState, snapshot: _MeshState) -> None:
    state.points = snapshot.points
    state.triangles = snapshot.triangles
    state.fixed = snapshot.fixed
    state.chains = snapshot.chains
    state.open_nodes = snapshot.open_nodes


def _execute_guarded(state: _MeshState, operation: Callable[[], int], focus: str) -> dict[str, Any]:
    before = _metrics(state)
    snapshot = _clone_state(state)
    attempted = int(operation())
    candidate = _metrics(state)
    accepted, decision = _stage_acceptance(before, candidate, attempted, focus)
    if not accepted:
        _restore_state(state, snapshot)
        return {
            "before": before,
            "after": before,
            "attempted_operations": attempted,
            "committed_operations": 0,
            "rolled_back": bool(attempted),
            "decision": decision,
        }
    return {
        "before": before,
        "after": candidate,
        "attempted_operations": attempted,
        "committed_operations": attempted,
        "rolled_back": False,
        "decision": decision,
    }


def _stage_acceptance(
    before: dict[str, Any],
    after: dict[str, Any],
    attempted: int,
    focus: str,
) -> tuple[bool, str]:
    if attempted == 0:
        return True, "no_operation"
    before_q = before["oceanmesh_quality"]
    after_q = after["oceanmesh_quality"]
    before_angles = before["angles"]
    after_angles = after["angles"]
    invariant_ok = (
        after["constraint_integrity"]["all_protected_edges_present"]
        and after["constraint_integrity"]["open_boundary_ordered"]
        and after["topology"]["connected_component_count"] <= before["topology"]["connected_component_count"]
        and after["topology"]["nonmanifold_edge_count"] <= before["topology"]["nonmanifold_edge_count"]
        and after["topology"]["nonpositive_signed_area_count"] <= before["topology"]["nonpositive_signed_area_count"]
        and after["topology"]["boundary_degree_anomaly_count"] <= before["topology"]["boundary_degree_anomaly_count"]
    )
    quality_nonregression = (
        after_q["q_l3_sigma"] + 1.0e-12 >= before_q["q_l3_sigma"]
        and after_q["q_quantiles"]["p01"] + 1.0e-12 >= before_q["q_quantiles"]["p01"]
        and after_q["count_q_below_0_25"] <= before_q["count_q_below_0_25"]
        and after_angles["min_angle_quantiles_deg"]["p01"] + 1.0e-12
        >= before_angles["min_angle_quantiles_deg"]["p01"]
        and after_angles["count_min_angle_below_30"] <= before_angles["count_min_angle_below_30"]
    )
    if not invariant_ok:
        return False, "rollback_invariant_regression"
    if not quality_nonregression:
        return False, "rollback_quality_tail_regression"
    improvement = {
        "consistency": True,
        "quality_tail": (
            after_q["q_l3_sigma"] > before_q["q_l3_sigma"]
            or after_q["q_quantiles"]["p01"] > before_q["q_quantiles"]["p01"]
            or after_q["count_q_below_0_25"] < before_q["count_q_below_0_25"]
        ),
        "traversability": (
            after["topology"]["boundary_degree_anomaly_count"]
            < before["topology"]["boundary_degree_anomaly_count"]
            or after["topology"]["singly_connected_triangle_count"]
            < before["topology"]["singly_connected_triangle_count"]
            or after_q["count_q_below_0_25"] < before_q["count_q_below_0_25"]
        ),
        "singly_connected": (
            after["topology"]["singly_connected_triangle_count"]
            < before["topology"]["singly_connected_triangle_count"]
        ),
        "connectivity": (
            after["valence"]["count_valence_above_8"] < before["valence"]["count_valence_above_8"]
            or after["valence"]["max_node_valence"] < before["valence"]["max_node_valence"]
        ),
        "smoothing": (
            after_q["q_l3_sigma"] > before_q["q_l3_sigma"]
            or after_q["q_quantiles"]["p01"] > before_q["q_quantiles"]["p01"]
            or after_angles["min_angle_quantiles_deg"]["p01"]
            > before_angles["min_angle_quantiles_deg"]["p01"]
        ),
    }.get(focus, False)
    return (True, "committed_nonregressing_improvement") if improvement else (False, "rollback_no_target_improvement")


def _record_outcome(history: list[dict[str, Any]], stage: str, outcome: dict[str, Any]) -> None:
    _record(
        history,
        stage,
        int(outcome["committed_operations"]),
        outcome["before"],
        outcome["after"],
        attempted_operations=int(outcome["attempted_operations"]),
        rolled_back=bool(outcome["rolled_back"]),
        decision=str(outcome["decision"]),
    )


def _record(
    history: list[dict[str, Any]],
    stage: str,
    operations: int,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    attempted_operations: int | None = None,
    rolled_back: bool = False,
    decision: str = "committed",
) -> None:
    history.append(
        {
            "stage": stage,
            "operations": int(operations),
            "attempted_operations": int(attempted_operations if attempted_operations is not None else operations),
            "changed": bool(operations),
            "rolled_back": bool(rolled_back),
            "decision": decision,
            "before": _headline(before),
            "after": _headline(after),
            "after_full": after,
        }
    )


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": metrics["node_count"],
        "triangles": metrics["triangle_count"],
        "q_min": metrics["oceanmesh_quality"]["q_min"],
        "q_l3_sigma": metrics["oceanmesh_quality"]["q_l3_sigma"],
        "q_p01": metrics["oceanmesh_quality"]["q_quantiles"]["p01"],
        "count_q_below_0_25": metrics["oceanmesh_quality"]["count_q_below_0_25"],
        "count_angle_below_30": metrics["angles"]["count_min_angle_below_30"],
        "singly_connected": metrics["topology"]["singly_connected_triangle_count"],
        "boundary_degree_anomalies": metrics["topology"]["boundary_degree_anomaly_count"],
        "max_valence": metrics["valence"]["max_node_valence"],
        "count_valence_above_8": metrics["valence"]["count_valence_above_8"],
    }


def _build_report(
    config: PostprocessConfig,
    connectivity_limit: int,
    before: dict[str, Any],
    after: dict[str, Any],
    original_boundary_coordinates: list[np.ndarray],
    state: _MeshState,
    stop_reason: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    shifts: list[float] = []
    for original, chain in zip(original_boundary_coordinates, state.chains):
        current = state.points[np.asarray(chain, dtype=int)]
        if original.shape == current.shape and len(original):
            shifts.append(float(np.max(np.linalg.norm(current - original, axis=1))))
        else:
            shifts.append(float("inf"))
    protected_defects = int(after["oceanmesh_quality"]["boundary_count_q_below_0_25"])
    return {
        "schema_version": "fvcom_mesh_postprocess_v1",
        "profile": config.profile,
        "boundary_policy": config.boundary_policy,
        "stage_order": [entry["stage"] for entry in history],
        "profile_defaults": {
            "disjoint_area_fraction": float(config.disjoint_area_fraction),
            "boundary_quality_cutoff": float(config.boundary_quality_cutoff),
            "connectivity_limit": int(connectivity_limit),
            "max_passes": int(config.max_passes),
            "singly_connected_policy": "exhaustive" if config.profile == "rpw2019" else "disabled_by_profile",
        },
        "stop_reason": stop_reason,
        "boundary_coordinate_max_shift_m": float(max(shifts, default=0.0)),
        "all_boundary_coordinates_unchanged": bool(max(shifts, default=0.0) <= 1.0e-10),
        "protected_boundary_quality_defect_count": protected_defects,
        "before": before,
        "after": after,
        "improvement": _comparison(before, after),
        "operation_count": int(sum(entry["operations"] for entry in history)),
    }


def _comparison(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "q_l3_sigma": (before["oceanmesh_quality"]["q_l3_sigma"], after["oceanmesh_quality"]["q_l3_sigma"], "increase"),
        "q_p01": (before["oceanmesh_quality"]["q_quantiles"]["p01"], after["oceanmesh_quality"]["q_quantiles"]["p01"], "increase"),
        "min_angle_p01": (before["angles"]["min_angle_quantiles_deg"]["p01"], after["angles"]["min_angle_quantiles_deg"]["p01"], "increase"),
        "count_q_below_0_25": (before["oceanmesh_quality"]["count_q_below_0_25"], after["oceanmesh_quality"]["count_q_below_0_25"], "decrease"),
        "count_angle_below_30": (before["angles"]["count_min_angle_below_30"], after["angles"]["count_min_angle_below_30"], "decrease"),
        "singly_connected": (before["topology"]["singly_connected_triangle_count"], after["topology"]["singly_connected_triangle_count"], "decrease"),
        "boundary_degree_anomalies": (before["topology"]["boundary_degree_anomaly_count"], after["topology"]["boundary_degree_anomaly_count"], "decrease"),
        "count_valence_above_8": (before["valence"]["count_valence_above_8"], after["valence"]["count_valence_above_8"], "decrease"),
    }
    output: dict[str, Any] = {}
    for name, (old, new, direction) in keys.items():
        output[name] = {
            "before": float(old),
            "after": float(new),
            "delta": float(new - old),
            "improved": bool(new > old if direction == "increase" else new < old),
        }
    output["strict_improvement_count"] = int(sum(value["improved"] for value in output.values() if isinstance(value, dict)))
    return output


def _fix_consistency(state: _MeshState) -> int:
    before_count = len(state.triangles)
    state.triangles = _orient_triangles(state.points, state.triangles)
    if len(state.triangles):
        sorted_rows = np.sort(state.triangles, axis=1)
        _, unique_indices = np.unique(sorted_rows, axis=0, return_index=True)
        state.triangles = state.triangles[np.sort(unique_indices)]
    protected = chain_edges(state.chains)
    geometry = triangle_geometry(state.points, state.triangles)
    keep = np.ones(len(state.triangles), dtype=bool)
    for index in np.where(geometry["area"] <= 1.0e-12)[0]:
        tri = state.triangles[index]
        edges = {tuple(sorted((int(a), int(b)))) for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))}
        if not edges.intersection(protected):
            keep[index] = False
    state.triangles = state.triangles[keep]
    _compact_state(state)
    return int(before_count - len(state.triangles))


def _repair_boundary_quality_by_flips(state: _MeshState, config: PostprocessConfig, purpose: str) -> int:
    operations = 0
    for _ in range(max(1, int(config.max_flip_passes))):
        topology = build_edge_topology(len(state.points), state.triangles)
        geometry = triangle_geometry(state.points, state.triangles)
        protected = chain_edges(state.chains)
        selected_triangles: set[int] = set()
        selected_new_edges: set[tuple[int, int]] = set()
        updates: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for edge, attached in sorted(topology.edge_to_triangles.items()):
            if len(attached) != 2 or edge in protected:
                continue
            first, second = attached
            if first in selected_triangles or second in selected_triangles:
                continue
            first_edges = _triangle_edges(state.triangles[first])
            second_edges = _triangle_edges(state.triangles[second])
            boundary_bad = (
                (geometry["quality"][first] < config.boundary_quality_cutoff and bool(first_edges & protected))
                or (geometry["quality"][second] < config.boundary_quality_cutoff and bool(second_edges & protected))
            )
            singly_bad = (
                topology.triangle_neighbor_count[first] == 1 or topology.triangle_neighbor_count[second] == 1
            )
            if purpose in {"poor_boundary", "traversability"} and not boundary_bad:
                continue
            if purpose == "singly_connected" and not singly_bad:
                continue
            candidate = _edge_flip_candidate(state.points, state.triangles, edge, attached)
            if candidate is None:
                continue
            new_first, new_second, old_min, new_min = candidate
            if new_min <= old_min + 1.0e-12:
                continue
            opposite_edge = tuple(sorted((int(new_first[0]), int(new_first[1]))))
            if opposite_edge in topology.edge_to_triangles or opposite_edge in selected_new_edges:
                continue
            updates.append((first, second, new_first, new_second))
            selected_triangles.update((first, second))
            selected_new_edges.add(opposite_edge)
        if not updates:
            break
        for first, second, new_first, new_second in updates:
            state.triangles[first] = new_first
            state.triangles[second] = new_second
        operations += len(updates)
    return operations


def _remove_unprotected_singly(state: _MeshState, max_rounds: int) -> int:
    operations = 0
    for _ in range(max_rounds):
        topology = build_edge_topology(len(state.points), state.triangles)
        protected = chain_edges(state.chains)
        remove: list[int] = []
        for index in np.where(topology.triangle_neighbor_count == 1)[0]:
            if not (_triangle_edges(state.triangles[index]) & protected):
                remove.append(int(index))
        if not remove:
            break
        state.triangles = np.delete(state.triangles, remove, axis=0)
        operations += len(remove)
        _compact_state(state)
    return operations


def _collapse_interior_thin_edges(state: _MeshState, config: PostprocessConfig) -> int:
    operations = 0
    for _ in range(max(0, int(config.max_interior_collapses))):
        geometry = triangle_geometry(state.points, state.triangles)
        protected = chain_edges(state.chains)
        candidate_edge: tuple[int, int] | None = None
        for index in np.argsort(geometry["quality"]):
            if geometry["quality"][index] >= config.boundary_quality_cutoff:
                break
            tri = state.triangles[index]
            if _triangle_edges(tri) & protected:
                continue
            coords = state.points[tri]
            edges = [
                (int(tri[0]), int(tri[1]), np.linalg.norm(coords[0] - coords[1])),
                (int(tri[1]), int(tri[2]), np.linalg.norm(coords[1] - coords[2])),
                (int(tri[2]), int(tri[0]), np.linalg.norm(coords[2] - coords[0])),
            ]
            for a, b, _ in sorted(edges, key=lambda item: item[2]):
                if not state.fixed[a] and not state.fixed[b] and tuple(sorted((a, b))) not in protected:
                    candidate_edge = (a, b)
                    break
            if candidate_edge is not None:
                break
        if candidate_edge is None or not _try_interior_collapse(state, candidate_edge):
            break
        operations += 1
    return operations


def _try_interior_collapse(state: _MeshState, edge: tuple[int, int]) -> bool:
    a, b = edge
    old_metrics = _metrics(state)
    trial = _MeshState(
        points=state.points.copy(),
        triangles=state.triangles.copy(),
        fixed=state.fixed.copy(),
        chains=[chain.copy() for chain in state.chains],
        open_nodes=state.open_nodes.copy(),
    )
    trial.points[a] = 0.5 * (trial.points[a] + trial.points[b])
    trial.triangles[trial.triangles == b] = a
    keep = np.asarray([len(set(map(int, tri))) == 3 for tri in trial.triangles], dtype=bool)
    trial.triangles = _orient_triangles(trial.points, trial.triangles[keep])
    geometry = triangle_geometry(trial.points, trial.triangles)
    if np.any(geometry["signed_area"] <= 1.0e-12):
        return False
    _compact_state(trial)
    new_metrics = _metrics(trial)
    if not new_metrics["constraint_integrity"]["all_protected_edges_present"]:
        return False
    if new_metrics["oceanmesh_quality"]["q_l3_sigma"] + 1.0e-12 < old_metrics["oceanmesh_quality"]["q_l3_sigma"]:
        return False
    state.points, state.triangles, state.fixed, state.chains, state.open_nodes = (
        trial.points,
        trial.triangles,
        trial.fixed,
        trial.chains,
        trial.open_nodes,
    )
    return True


def _bound_connectivity(state: _MeshState, limit: int, config: PostprocessConfig) -> int:
    operations = _connectivity_flip_passes(state, limit, max_passes=config.max_flip_passes)
    operations += _remove_high_valence_cavities(state, limit, max_removals=config.max_cavity_removals)
    return operations


def _connectivity_flip_passes(state: _MeshState, limit: int, max_passes: int) -> int:
    operations = 0
    for _ in range(max_passes):
        topology = build_edge_topology(len(state.points), state.triangles)
        valence = np.asarray([len(values) for values in topology.node_neighbors], dtype=int)
        protected = chain_edges(state.chains)
        selected_triangles: set[int] = set()
        updates: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for edge, attached in sorted(topology.edge_to_triangles.items()):
            if len(attached) != 2 or edge in protected or max(valence[list(edge)]) <= limit:
                continue
            first, second = attached
            if first in selected_triangles or second in selected_triangles:
                continue
            candidate = _edge_flip_candidate(state.points, state.triangles, edge, attached)
            if candidate is None:
                continue
            new_first, new_second, old_min, new_min = candidate
            a, b = edge
            c = next(int(value) for value in state.triangles[first] if int(value) not in edge)
            d = next(int(value) for value in state.triangles[second] if int(value) not in edge)
            new_edge = tuple(sorted((c, d)))
            if new_edge in topology.edge_to_triangles:
                continue
            before_score = sum(max(0, int(valence[node]) - limit) for node in (a, b, c, d))
            simulated = {a: valence[a] - 1, b: valence[b] - 1, c: valence[c] + 1, d: valence[d] + 1}
            after_score = sum(max(0, int(simulated[node]) - limit) for node in (a, b, c, d))
            if after_score >= before_score or new_min + 1.0e-12 < 0.90 * old_min:
                continue
            updates.append((first, second, new_first, new_second))
            selected_triangles.update((first, second))
        if not updates:
            break
        for first, second, new_first, new_second in updates:
            state.triangles[first] = new_first
            state.triangles[second] = new_second
        operations += len(updates)
    return operations


def _remove_high_valence_cavities(state: _MeshState, limit: int, max_removals: int) -> int:
    operations = 0
    for _ in range(max_removals):
        topology = build_edge_topology(len(state.points), state.triangles)
        valence = np.asarray([len(values) for values in topology.node_neighbors], dtype=int)
        candidates = [index for index in np.argsort(-valence) if valence[index] > limit and not state.fixed[index]]
        if not candidates:
            break
        accepted = False
        for node in candidates[:100]:
            incident = np.where(np.any(state.triangles == int(node), axis=1))[0]
            ring = _ordered_one_ring(state.triangles[incident], int(node))
            if ring is None or len(ring) < 3:
                continue
            new_triangles = _ear_clip(state.points, ring)
            if new_triangles is None:
                continue
            old_quality = triangle_geometry(state.points, state.triangles[incident])["quality"]
            new_triangles = _orient_triangles(state.points, np.asarray(new_triangles, dtype=int))
            new_quality = triangle_geometry(state.points, new_triangles)["quality"]
            if np.min(new_quality) + 1.0e-12 < 0.80 * np.min(old_quality):
                continue
            trial = _MeshState(
                points=state.points.copy(),
                triangles=np.vstack([np.delete(state.triangles, incident, axis=0), new_triangles]),
                fixed=state.fixed.copy(),
                chains=[chain.copy() for chain in state.chains],
                open_nodes=state.open_nodes.copy(),
            )
            _compact_state(trial)
            trial_metrics = _metrics(trial)
            if not trial_metrics["constraint_integrity"]["all_protected_edges_present"]:
                continue
            if trial_metrics["valence"]["count_valence_above_8"] > _metrics(state)["valence"]["count_valence_above_8"]:
                continue
            state.points, state.triangles, state.fixed, state.chains, state.open_nodes = (
                trial.points,
                trial.triangles,
                trial.fixed,
                trial.chains,
                trial.open_nodes,
            )
            operations += 1
            accepted = True
            break
        if not accepted:
            break
    return operations


def _direct_implicit_smooth(state: _MeshState) -> int:
    topology = build_edge_topology(len(state.points), state.triangles)
    free_nodes = np.where(~state.fixed)[0]
    if not len(free_nodes):
        return 0
    row_for_node = {int(node): row for row, node in enumerate(free_nodes)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros((len(free_nodes), 2), dtype=float)
    for row, node in enumerate(free_nodes):
        neighbors = sorted(topology.node_neighbors[int(node)])
        if not neighbors:
            rows.append(row)
            cols.append(row)
            data.append(1.0)
            rhs[row] = state.points[node]
            continue
        rows.append(row)
        cols.append(row)
        data.append(float(len(neighbors)))
        for neighbor in neighbors:
            if neighbor in row_for_node:
                rows.append(row)
                cols.append(row_for_node[neighbor])
                data.append(-1.0)
            else:
                rhs[row] += state.points[neighbor]
    matrix = coo_matrix((data, (rows, cols)), shape=(len(free_nodes), len(free_nodes))).tocsr()
    try:
        target = np.column_stack([spsolve(matrix, rhs[:, 0]), spsolve(matrix, rhs[:, 1])])
    except Exception:
        return 0
    if not np.all(np.isfinite(target)):
        return 0
    before_geometry = triangle_geometry(state.points, state.triangles)
    before_q = before_geometry["quality"]
    before_l3 = float(np.mean(before_q) - 3.0 * np.std(before_q))
    before_p01 = float(np.quantile(before_q, 0.01))
    for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
        candidate = state.points.copy()
        candidate[free_nodes] = (1.0 - alpha) * state.points[free_nodes] + alpha * target
        geometry = triangle_geometry(candidate, state.triangles)
        if np.any(geometry["signed_area"] <= 1.0e-12):
            continue
        q = geometry["quality"]
        l3 = float(np.mean(q) - 3.0 * np.std(q))
        p01 = float(np.quantile(q, 0.01))
        if l3 + 1.0e-12 >= before_l3 and p01 + 1.0e-12 >= before_p01:
            moved = int(np.sum(np.linalg.norm(candidate - state.points, axis=1) > 1.0e-9))
            state.points = candidate
            return moved
    return 0


def _edge_flip_candidate(
    points: np.ndarray,
    triangles: np.ndarray,
    edge: tuple[int, int],
    attached: list[int],
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    a, b = edge
    first, second = attached
    c_values = [int(value) for value in triangles[first] if int(value) not in edge]
    d_values = [int(value) for value in triangles[second] if int(value) not in edge]
    if len(c_values) != 1 or len(d_values) != 1 or c_values[0] == d_values[0]:
        return None
    c, d = c_values[0], d_values[0]
    old = triangles[[first, second]]
    new = _orient_triangles(points, np.asarray([[c, d, a], [d, c, b]], dtype=int))
    old_geometry = triangle_geometry(points, old)
    new_geometry = triangle_geometry(points, new)
    if np.any(new_geometry["signed_area"] <= 1.0e-12):
        return None
    old_area = float(np.sum(old_geometry["area"]))
    new_area = float(np.sum(new_geometry["area"]))
    if abs(new_area - old_area) > 1.0e-8 * max(old_area, 1.0):
        return None
    return new[0], new[1], float(np.min(old_geometry["quality"])), float(np.min(new_geometry["quality"]))


def _ordered_one_ring(incident_triangles: np.ndarray, node: int) -> list[int] | None:
    adjacency: dict[int, list[int]] = {}
    for tri in incident_triangles:
        others = [int(value) for value in tri if int(value) != node]
        if len(others) != 2:
            return None
        a, b = others
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if not adjacency or any(len(set(values)) != 2 for values in adjacency.values()):
        return None
    start = min(adjacency)
    ring = [start]
    previous = -1
    current = start
    for _ in range(len(adjacency) + 1):
        candidates = sorted(set(adjacency[current]))
        next_node = candidates[0] if candidates[0] != previous else candidates[1]
        if next_node == start:
            return ring if len(ring) == len(adjacency) else None
        ring.append(next_node)
        previous, current = current, next_node
    return None


def _ear_clip(points: np.ndarray, ring: list[int]) -> list[list[int]] | None:
    vertices = ring.copy()
    polygon = points[np.asarray(vertices, dtype=int)]
    area2 = float(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - np.roll(polygon[:, 0], -1) * polygon[:, 1]))
    if area2 < 0.0:
        vertices.reverse()
    output: list[list[int]] = []
    while len(vertices) > 3:
        ears: list[tuple[float, int, list[int]]] = []
        for index, current in enumerate(vertices):
            previous = vertices[index - 1]
            following = vertices[(index + 1) % len(vertices)]
            triangle = [previous, current, following]
            geometry = triangle_geometry(points, np.asarray([triangle], dtype=int))
            if geometry["signed_area"][0] <= 1.0e-12:
                continue
            if any(
                _point_in_triangle(points[value], points[triangle])
                for value in vertices
                if value not in triangle
            ):
                continue
            ears.append((float(geometry["quality"][0]), index, triangle))
        if not ears:
            return None
        _, index, triangle = max(ears, key=lambda item: (item[0], -item[1]))
        output.append(triangle)
        vertices.pop(index)
    output.append(vertices)
    return output


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    a, b, c = triangle
    v0 = c - a
    v1 = b - a
    v2 = point - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1.0e-30:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return u > 1.0e-12 and v > 1.0e-12 and u + v < 1.0 - 1.0e-12


def _compact_state(state: _MeshState) -> None:
    used = np.zeros(len(state.points), dtype=bool)
    if len(state.triangles):
        used[state.triangles.ravel()] = True
    keep = used | state.fixed
    mapping = np.full(len(state.points), -1, dtype=int)
    mapping[keep] = np.arange(int(np.sum(keep)))
    state.points = state.points[keep]
    state.fixed = state.fixed[keep]
    state.triangles = mapping[state.triangles]
    state.chains = [[int(mapping[value]) for value in chain] for chain in state.chains]
    state.open_nodes = [int(mapping[value]) for value in state.open_nodes]


def _orient_triangles(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    output = np.asarray(triangles, dtype=int).copy()
    if not len(output):
        return output
    coords = points[output]
    area2 = (
        (coords[:, 1, 0] - coords[:, 0, 0]) * (coords[:, 2, 1] - coords[:, 0, 1])
        - (coords[:, 1, 1] - coords[:, 0, 1]) * (coords[:, 2, 0] - coords[:, 0, 0])
    )
    flip = np.where(area2 < 0.0)[0]
    if len(flip):
        output[flip] = output[flip][:, [0, 2, 1]]
    return output


def _triangle_edges(triangle: np.ndarray) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(triangle[0]), int(triangle[1])))),
        tuple(sorted((int(triangle[1]), int(triangle[2])))),
        tuple(sorted((int(triangle[2]), int(triangle[0])))),
    }


def _progress(callback: ProgressCallback | None, message: str, fraction: float, extra: dict[str, Any] | None) -> None:
    if callback is not None:
        callback(message, float(fraction), extra)
