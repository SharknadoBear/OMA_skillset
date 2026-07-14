"""Aggressive, transactional local topology conditioning for FVCOM meshes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Callable

import numpy as np

from .metrics import build_edge_topology, chain_edges, constraint_integrity, triangle_geometry
from .regional_conditioning import SpringRelaxConfig, _edge_flip_candidate, relax_mesh_spring


@dataclass(frozen=True)
class AggressiveConditioningConfig:
    enabled: bool = True
    enable_pruning: bool = True
    enable_thin_repair: bool = True
    enable_valence_repair: bool = True
    max_rounds: int = 4
    quality_threshold: float = 0.25
    min_angle_deg: float = 20.0
    superthin_quality_threshold: float = 0.10
    superthin_min_angle_deg: float = 5.0
    collapse_l_over_h: float = 0.50
    max_collapses_per_round: int = 100
    max_boundary_edits_per_round: int = 25
    max_superthin_flips_per_round: int = 100
    max_boundary_welds_per_round: int = 25
    max_boundary_ear_removals_per_round: int = 25
    boundary_weld_max_distance_fraction: float = 0.50
    boundary_weld_max_altitude_to_arc_fraction: float = 0.20
    boundary_weld_land_max_distance_m: float = 5.0
    boundary_weld_open_max_distance_m: float = 250.0
    boundary_weld_anchor_buffer_segments: int = 1
    boundary_weld_junction_buffer_segments: int = 2
    boundary_weld_channel_clearance_fraction: float = 1.50
    boundary_weld_forbidden_kind_tokens: tuple[str, ...] = ("channel", "junction", "landfall")
    max_prunes_per_round: int = 500
    prune_l_over_h: float = 1.25
    prune_min_quality: float = 0.40
    prune_min_angle_deg: float = 28.0
    max_valence: int = 8
    max_valence_removals_per_round: int = 500
    max_valence_flip_batch: int = 64
    max_valence_cluster_merges_per_round: int = 25
    max_valence_l_over_h_count_increase: int = 0
    valence_node_lineage_filter: tuple[int, ...] = ()
    boundary_edit_policy: str = "kind-aware-envelope"
    land_boundary_max_deviation_m: float = 5.0
    open_boundary_max_deviation_m: float = 250.0
    land_boundary_deviation_fraction: float = 0.03
    open_boundary_deviation_fraction: float = 0.05
    maximum_domain_area_change_fraction: float = 1.0e-4
    micro_relax_cycles: int = 3
    micro_relax_iterations: int = 6
    micro_relax_ring_layers: int = 2
    micro_relax_damping: float = 0.30
    micro_relax_max_step_fraction: float = 0.08
    micro_relax_shape_weight: float = 0.25


@dataclass
class LocalTopologyResult:
    nodes_xy: np.ndarray
    triangles: np.ndarray
    fixed_node_mask: np.ndarray
    target_spacing_m: np.ndarray
    constraint_chains: list[list[int]]
    open_boundary_nodes_zero_based: np.ndarray
    boundary_kinds: list[str]
    hard_anchor_mask: np.ndarray
    node_lineage: np.ndarray
    report: dict[str, Any]
    edit_ledger: list[dict[str, Any]]


@dataclass
class _State:
    points: np.ndarray
    triangles: np.ndarray
    fixed: np.ndarray
    targets: np.ndarray
    chains: list[list[int]]
    open_nodes: np.ndarray
    kinds: list[str]
    hard: np.ndarray
    lineage: np.ndarray
    source_points: np.ndarray
    source_chains: list[list[int]]
    source_hard_anchor_lineage: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    target_sampler: Callable[[np.ndarray], np.ndarray] | None = None
    initial_domain_area_m2: float = 0.0
    initial_boundary_component_count: int = 0
    initial_boundary_degree_anomaly_count: int = 0
    initial_singly_connected_triangle_count: int = 0
    initial_protected_not_boundary_count: int = 0
    ledger: list[dict[str, Any]] = field(default_factory=list)
    cumulative_boundary_area_change_m2: float = 0.0
    last_affected: list[int] = field(default_factory=list)

    def clone(self) -> "_State":
        return _State(
            points=self.points.copy(),
            triangles=self.triangles.copy(),
            fixed=self.fixed.copy(),
            targets=self.targets.copy(),
            chains=[chain.copy() for chain in self.chains],
            open_nodes=self.open_nodes.copy(),
            kinds=self.kinds.copy(),
            hard=self.hard.copy(),
            lineage=self.lineage.copy(),
            source_points=self.source_points.copy(),
            source_chains=[chain.copy() for chain in self.source_chains],
            source_hard_anchor_lineage=self.source_hard_anchor_lineage.copy(),
            target_sampler=self.target_sampler,
            initial_domain_area_m2=float(self.initial_domain_area_m2),
            initial_boundary_component_count=int(self.initial_boundary_component_count),
            initial_boundary_degree_anomaly_count=int(self.initial_boundary_degree_anomaly_count),
            initial_singly_connected_triangle_count=int(self.initial_singly_connected_triangle_count),
            initial_protected_not_boundary_count=int(self.initial_protected_not_boundary_count),
            ledger=[entry.copy() for entry in self.ledger],
            cumulative_boundary_area_change_m2=float(self.cumulative_boundary_area_change_m2),
            last_affected=self.last_affected.copy(),
        )


def condition_mesh_aggressive(
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    fixed_node_mask: np.ndarray,
    constraint_chains: list[list[int]],
    open_boundary_nodes_zero_based: np.ndarray,
    *,
    target_spacing_m: np.ndarray,
    boundary_kinds: list[str] | None = None,
    hard_anchor_mask: np.ndarray | None = None,
    target_spacing_sampler: Callable[[np.ndarray], np.ndarray] | None = None,
    config: AggressiveConditioningConfig | None = None,
) -> LocalTopologyResult:
    """Apply target-aware pruning, aggressive thin repair, and hard valence repair."""
    config = config or AggressiveConditioningConfig()
    points = np.asarray(nodes_xy, dtype=float).copy()
    tris = _orient_ccw(points, np.asarray(triangles, dtype=int).copy())
    fixed = np.asarray(fixed_node_mask, dtype=bool).copy()
    targets = _normalize_targets(target_spacing_m, len(points))
    kinds = list(boundary_kinds or ["interior"] * len(points))
    if len(kinds) < len(points):
        kinds.extend(["interior"] * (len(points) - len(kinds)))
    elif len(kinds) > len(points):
        # ``kinds`` is descriptive metadata rather than mesh geometry.  A
        # previously rolled-back insertion can leave stale tail labels in a
        # caller-owned list; active node indices always occupy the leading
        # ``len(points)`` entries.  Normalize that harmless tail here while
        # retaining strict one-value-per-node checks for all numeric fields.
        kinds = kinds[: len(points)]
    hard = np.asarray(hard_anchor_mask if hard_anchor_mask is not None else np.zeros(len(points), dtype=bool), dtype=bool).copy()
    if len(hard) != len(points):
        raise ValueError("hard_anchor_mask must have one value per node")
    chains = [list(map(int, chain)) for chain in constraint_chains]
    initial_topology = build_edge_topology(len(points), tris)
    initial_boundary = _boundary_graph_audit(initial_topology)
    initial_protected = chain_edges(chains)
    state = _State(
        points=points,
        triangles=tris,
        fixed=fixed,
        targets=targets,
        chains=chains,
        open_nodes=np.asarray(open_boundary_nodes_zero_based, dtype=int).copy(),
        kinds=kinds,
        hard=hard,
        lineage=np.arange(len(points), dtype=int),
        source_points=points.copy(),
        source_chains=[chain.copy() for chain in chains],
        source_hard_anchor_lineage=np.where(hard)[0].astype(int),
        target_sampler=target_spacing_sampler,
        initial_domain_area_m2=max(_signed_mesh_area(points, tris), 1.0e-30),
        initial_boundary_component_count=int(initial_boundary["component_count"]),
        initial_boundary_degree_anomaly_count=int(initial_boundary["degree_anomaly_count"]),
        initial_singly_connected_triangle_count=int(np.count_nonzero(initial_topology.triangle_neighbor_count == 1)),
        initial_protected_not_boundary_count=int(
            sum(len(initial_topology.edge_to_triangles.get(edge, [])) != 1 for edge in initial_protected)
        ),
    )
    initial = _summary(state, config)
    initial_components = int(initial["connected_component_count"])
    rounds: list[dict[str, Any]] = []
    if config.enabled:
        for round_index in range(max(0, int(config.max_rounds))):
            before_round = _summary(state, config)
            prune = (
                _prune_redundant_vertices(state, config, initial_components)
                if config.enable_pruning and int(config.max_prunes_per_round) > 0
                else _disabled_stage(state, config, "pruning_disabled")
            )
            thin = (
                _repair_superthin(state, config, initial_components)
                if config.enable_thin_repair and _thin_budget(config) > 0
                else _disabled_stage(state, config, "thin_repair_disabled")
            )
            valence, post_valence_thin, compound = _repair_valence_thin_atomic(
                state,
                config,
                initial_components,
            )
            after_round = _summary(state, config)
            operations = int(
                prune["accepted"]
                + thin["accepted"]
                + valence["accepted"]
                + post_valence_thin["accepted"]
            )
            rounds.append(
                {
                    "round": int(round_index + 1),
                    "before": before_round,
                    "redundant_vertex_pruning": prune,
                    "aggressive_thin_repair": thin,
                    "high_valence_repair": valence,
                    "post_valence_thin_repair": post_valence_thin,
                    "valence_thin_atomic_transaction": compound,
                    "after": after_round,
                    "accepted_operation_count": operations,
                }
            )
            if operations == 0:
                break
            if after_round["count_valence_above_limit"] == 0 and after_round["superthin_triangle_count"] == 0:
                break
    final = _summary(state, config)
    hard_gate = bool(final["count_valence_above_limit"] == 0)
    superthin_gate = bool(final["superthin_triangle_count"] == 0)
    report = {
        "schema_version": "fvcom_aggressive_local_conditioning_v2",
        "profile": "aggressive-local-v2",
        "settings": asdict(config),
        "accepted": bool(_state_invariants(state, initial_components)[0]),
        "fvcom_valence_gate_passed": hard_gate,
        "superthin_gate_passed": superthin_gate,
        "terminal_topology_gate_passed": bool(hard_gate and superthin_gate),
        "before": initial,
        "after": final,
        "rounds": rounds,
        "edit_count": int(len(state.ledger)),
        "edit_counts": _ledger_counts(state.ledger),
        "cumulative_boundary_area_change_m2": float(state.cumulative_boundary_area_change_m2),
        "cumulative_boundary_area_change_fraction": float(
            state.cumulative_boundary_area_change_m2 / max(_mesh_area(points, tris), 1.0e-30)
        ),
        "invariants": _state_invariants(state, initial_components)[1],
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
        report=report,
        edit_ledger=[entry.copy() for entry in state.ledger],
    )


def _thin_budget(config: AggressiveConditioningConfig) -> int:
    return int(
        max(0, int(config.max_boundary_ear_removals_per_round))
        + max(0, int(config.max_boundary_welds_per_round))
        + max(0, int(config.max_superthin_flips_per_round))
        + max(0, int(config.max_collapses_per_round))
        + max(0, int(config.max_boundary_edits_per_round))
    )


def _disabled_stage(
    state: _State,
    config: AggressiveConditioningConfig,
    reason: str,
) -> dict[str, Any]:
    summary = _summary(state, config)
    return {
        "accepted": 0,
        "rejected": 0,
        "disabled": True,
        "reason": str(reason),
        "before": summary,
        "after": summary,
    }


def _repair_valence_thin_atomic(
    state: _State,
    config: AggressiveConditioningConfig,
    initial_components: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Close valence and any resulting extreme-shape debt as one transaction.

    A valence edit is useful only if its newly created superthin debt can be
    repaid immediately.  The transaction therefore snapshots before valence,
    optionally invokes the thin queue, and rolls back both branches when the
    final global audit fails.  Disabling thin repair is mode-safe: valence may
    still proceed, but only when it creates no extreme-shape regression.
    """
    before = _summary(state, config)
    if not config.enable_valence_repair or int(config.max_valence_removals_per_round) <= 0:
        disabled = _disabled_stage(state, config, "valence_repair_disabled")
        post = _disabled_stage(state, config, "no_valence_transaction")
        return disabled, post, {
            "attempted": False,
            "accepted": True,
            "rolled_back": False,
            "rejected_gates": [],
            "before": before,
            "after": before,
        }

    snapshot = state.clone()
    valence = _repair_high_valence(state, config, initial_components)
    if int(valence.get("accepted", 0)) <= 0:
        post = _disabled_stage(state, config, "no_accepted_valence_edit")
        after = _summary(state, config)
        return valence, post, {
            "attempted": bool(valence.get("attempted_count", 0)),
            "accepted": True,
            "rolled_back": False,
            "rejected_gates": [],
            "before": before,
            "after": after,
        }

    post = (
        _repair_superthin(state, config, initial_components)
        if config.enable_thin_repair and _thin_budget(config) > 0
        else _disabled_stage(state, config, "thin_repair_disabled")
    )
    ok, invariants, after = _audit_state(state, config, initial_components)
    rejected_gates = _compound_valence_thin_failures(before, after, ok, invariants, config)
    accepted_edits = int(valence.get("accepted", 0)) + int(post.get("accepted", 0))
    if rejected_gates:
        attempted_valence = int(valence.get("accepted", 0))
        attempted_thin = int(post.get("accepted", 0))
        _restore(state, snapshot)
        valence = dict(valence)
        valence.update(
            {
                "accepted": 0,
                "rolled_back_operation_count": attempted_valence,
                "transaction_rolled_back": True,
                "after": _summary(state, config),
            }
        )
        post = dict(post)
        post.update(
            {
                "accepted": 0,
                "rolled_back_operation_count": attempted_thin,
                "transaction_rolled_back": True,
                "after": _summary(state, config),
            }
        )
        return valence, post, {
            "attempted": True,
            "accepted": False,
            "rolled_back": True,
            "rolled_back_operation_count": accepted_edits,
            "rejected_gates": rejected_gates,
            "invariants": invariants,
            "before": before,
            "trial": after,
            "after": _summary(state, config),
        }
    return valence, post, {
        "attempted": True,
        "accepted": True,
        "rolled_back": False,
        "accepted_operation_count": accepted_edits,
        "rejected_gates": [],
        "invariants": invariants,
        "before": before,
        "after": after,
    }


def _compound_valence_thin_failures(
    before: dict[str, Any],
    after: dict[str, Any],
    invariants_ok: bool,
    invariants: dict[str, Any],
    config: AggressiveConditioningConfig,
) -> list[str]:
    failures: list[str] = []
    if not invariants_ok:
        failures.extend(_failed_invariant_names(invariants))
    if not _nonregression(
        before,
        after,
        purpose="valence",
        max_l_over_h_count_increase=int(config.max_valence_l_over_h_count_increase),
    ):
        failures.append("valence_quality_target_nonregression")
    if after["superthin_triangle_count"] > before["superthin_triangle_count"]:
        failures.append("new_superthin_triangles")
    if after["superthin_severity_sum"] > before["superthin_severity_sum"] + 1.0e-10:
        failures.append("superthin_severity_regression")
    if after["singly_connected_triangle_count"] > before["singly_connected_triangle_count"]:
        failures.append("new_singly_connected_triangles")
    if after["boundary_degree_anomaly_count"] > before["boundary_degree_anomaly_count"]:
        failures.append("new_boundary_degree_anomalies")
    if after["boundary_component_count"] != before["boundary_component_count"]:
        failures.append("boundary_traversability_component_change")
    return sorted(set(failures))


def inventory_high_valence(
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    *,
    constraint_chains: list[list[int]] | None = None,
    fixed_node_mask: np.ndarray | None = None,
    boundary_kinds: list[str] | None = None,
    hard_anchor_mask: np.ndarray | None = None,
    node_lineage: np.ndarray | None = None,
    max_valence: int = 8,
) -> dict[str, Any]:
    """Inventory every unique-neighbor valence violation in one topology scan."""
    points = np.asarray(nodes_xy, dtype=float)
    tris = _orient_ccw(points, np.asarray(triangles, dtype=int))
    node_count = len(points)
    fixed = np.asarray(fixed_node_mask if fixed_node_mask is not None else np.zeros(node_count, dtype=bool), dtype=bool)
    hard = np.asarray(hard_anchor_mask if hard_anchor_mask is not None else np.zeros(node_count, dtype=bool), dtype=bool)
    kinds = list(boundary_kinds or ["interior"] * node_count)
    lineage = np.asarray(node_lineage if node_lineage is not None else np.arange(node_count), dtype=int)
    state = _State(
        points=points.copy(),
        triangles=tris.copy(),
        fixed=fixed.copy(),
        targets=np.ones(node_count, dtype=float),
        chains=[list(map(int, chain)) for chain in (constraint_chains or [])],
        open_nodes=np.empty(0, dtype=int),
        kinds=kinds,
        hard=hard.copy(),
        lineage=lineage.copy(),
        source_points=points.copy(),
        source_chains=[list(map(int, chain)) for chain in (constraint_chains or [])],
    )
    topology = build_edge_topology(node_count, tris)
    geometry = triangle_geometry(points, tris)
    incident_by_node = _incident_triangle_lists(node_count, tris)
    valence = np.asarray([len(values) for values in topology.node_neighbors], dtype=int)
    protected = chain_edges(state.chains)
    boundary_nodes = {int(value) for edge in topology.boundary_edges for value in edge}
    violating_nodes = {int(value) for value in np.where(valence > int(max_valence))[0]}
    violation_components: list[list[int]] = []
    unseen = set(violating_nodes)
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(int(current))
            following = sorted((topology.node_neighbors[current] & violating_nodes) & unseen)
            for value in following:
                unseen.remove(int(value))
                stack.append(int(value))
        violation_components.append(sorted(component))
    violation_components.sort(key=lambda values: (-len(values), values[0]))
    component_lookup = {
        int(node): (int(component_id), int(len(component)))
        for component_id, component in enumerate(violation_components)
        for node in component
    }
    records: list[dict[str, Any]] = []
    for node in np.where(valence > int(max_valence))[0]:
        node = int(node)
        incident = np.asarray(incident_by_node[node], dtype=int)
        ring = _ordered_one_ring(tris[incident], node) if len(incident) else None
        flip = _best_valence_flip(state, node, int(max_valence), topology, valence, protected=protected)
        violating_neighbors = sorted(int(value) for value in topology.node_neighbors[node] if valence[int(value)] > int(max_valence))
        if flip is not None:
            route = "edge_flip"
        elif fixed[node] and hard[node]:
            route = "hard_boundary_blocked"
        elif fixed[node]:
            route = "guarded_boundary_cavity"
        elif ring is None:
            route = "unordered_interior_ring"
        elif violating_neighbors:
            route = "interior_cluster_cavity"
        else:
            route = "interior_cavity"
        local_quality = geometry["quality"][incident] if len(incident) else np.empty(0)
        local_angles = np.min(geometry["angles_deg"][incident], axis=1) if len(incident) else np.empty(0)
        local_edges = geometry["edge_lengths"][incident] if len(incident) else np.empty((0, 3))
        records.append(
            {
                "node_index_zero_based": node,
                "node_id_1based": node + 1,
                "node_lineage": int(lineage[node]),
                "x_m": float(points[node, 0]),
                "y_m": float(points[node, 1]),
                "valence": int(valence[node]),
                "excess_above_limit": int(valence[node] - int(max_valence)),
                "incident_triangle_count": int(len(incident)),
                "is_mesh_boundary": bool(node in boundary_nodes),
                "is_fixed": bool(fixed[node]),
                "is_hard_anchor": bool(hard[node]),
                "boundary_kind": str(kinds[node]) if node < len(kinds) else "interior",
                "ordered_one_ring": bool(ring is not None),
                "legal_flip_available": bool(flip is not None),
                "repair_route_hint": route,
                "minimum_incident_quality": float(np.min(local_quality)) if len(local_quality) else float("nan"),
                "minimum_incident_angle_deg": float(np.min(local_angles)) if len(local_angles) else float("nan"),
                "median_incident_edge_m": float(np.median(local_edges)) if len(local_edges) else float("nan"),
                "maximum_neighbor_valence": int(max((valence[value] for value in topology.node_neighbors[node]), default=0)),
                "violating_neighbor_count": int(len(violating_neighbors)),
                "violating_neighbor_ids_1based": [int(value) + 1 for value in violating_neighbors],
                "violation_component_id": int(component_lookup[node][0]),
                "violation_component_size": int(component_lookup[node][1]),
            }
        )
    route_counts: dict[str, int] = {}
    for record in records:
        key = str(record["repair_route_hint"])
        route_counts[key] = route_counts.get(key, 0) + 1
    return {
        "max_valence_allowed": int(max_valence),
        "violation_count": int(len(records)),
        "maximum_valence": int(np.max(valence)) if len(valence) else 0,
        "squared_excess_sum": int(np.sum(np.maximum(0, valence - int(max_valence)) ** 2)),
        "route_counts": route_counts,
        "violation_component_count": int(len(violation_components)),
        "maximum_violation_component_size": int(max((len(values) for values in violation_components), default=0)),
        "violation_component_size_histogram": {
            str(size): int(sum(1 for values in violation_components if len(values) == size))
            for size in sorted({len(values) for values in violation_components})
        },
        "records": records,
    }


def _prune_redundant_vertices(state: _State, config: AggressiveConditioningConfig, initial_components: int) -> dict[str, Any]:
    stage_before = _summary(state, config)
    baseline = stage_before
    accepted = 0
    rejected = 0
    remaining_budget = max(0, int(config.max_prunes_per_round))
    while remaining_budget > 0:
        topology = build_edge_topology(len(state.points), state.triangles)
        valence = np.asarray([len(values) for values in topology.node_neighbors], dtype=int)
        existing_edges = set(topology.edge_to_triangles)
        incident_by_node = _incident_triangle_lists(len(state.points), state.triangles)
        candidates: list[tuple[float, int, list[int], np.ndarray, np.ndarray]] = []
        for node, neighbors in enumerate(topology.node_neighbors):
            if state.fixed[node] or len(neighbors) not in (3, 4):
                continue
            incident = np.asarray(incident_by_node[node], dtype=int)
            ring = _ordered_one_ring(state.triangles[incident], int(node))
            if ring is None or len(ring) not in (3, 4):
                continue
            replacement = _best_prune_triangulation(
                state,
                node,
                ring,
                config,
                valence=valence,
                existing_edges=existing_edges,
            )
            if replacement is None:
                continue
            geometry = triangle_geometry(state.points, replacement)
            h = _triangle_targets(state.targets, replacement)
            redundancy = float(np.max(geometry["edge_lengths"], axis=1).max() / max(float(np.nanmedian(h)), 1.0e-12))
            candidates.append((redundancy, int(node), ring, replacement, incident))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1]))
        selected: list[tuple[float, int, list[int], np.ndarray, np.ndarray]] = []
        occupied_nodes: set[int] = set()
        occupied_triangles: set[int] = set()
        batch_limit = min(50, remaining_budget)
        for candidate in candidates:
            _, _node, ring, _replacement, incident = candidate
            patch_nodes = set(map(int, ring)) | {int(_node)}
            patch_triangles = set(map(int, incident.tolist()))
            if patch_nodes & occupied_nodes or patch_triangles & occupied_triangles:
                continue
            selected.append(candidate)
            occupied_nodes.update(patch_nodes)
            occupied_triangles.update(patch_triangles)
            if len(selected) >= batch_limit:
                break
        if not selected:
            break
        snapshot = state.clone()
        keep = np.ones(len(state.triangles), dtype=bool)
        additions: list[np.ndarray] = []
        affected_nodes: set[int] = set()
        for _, node, ring, replacement, incident in selected:
            keep[incident] = False
            additions.append(replacement)
            affected_nodes.update(map(int, ring))
            state.ledger.append(
                {
                    "operation": f"degree-{len(ring)}-vertex-prune",
                    "removed_original_node": int(state.lineage[node]),
                    "node_before_compaction": int(node),
                }
            )
        state.triangles = _orient_ccw(state.points, np.vstack([state.triangles[keep], *additions]))
        state.last_affected = sorted(affected_nodes)
        _compact(state)
        _micro_relax(state, replacement_seed_nodes=state.last_affected, config=config)
        ok, _ = _state_invariants(state, initial_components)
        after_trial = _summary(state, config)
        if not ok or not _nonregression(baseline, after_trial, purpose="prune"):
            _restore(state, snapshot)
            rejected += len(selected)
            break
        accepted += len(selected)
        remaining_budget -= len(selected)
        baseline = after_trial
    return {"accepted": int(accepted), "rejected": int(rejected), "before": stage_before, "after": _summary(state, config)}


def _best_prune_triangulation(
    state: _State,
    node: int,
    ring: list[int],
    config: AggressiveConditioningConfig,
    *,
    valence: np.ndarray,
    existing_edges: set[tuple[int, int]],
) -> np.ndarray | None:
    if len(ring) == 3:
        options = [np.asarray([ring], dtype=int)]
    else:
        options = [
            np.asarray([[ring[0], ring[1], ring[2]], [ring[0], ring[2], ring[3]]], dtype=int),
            np.asarray([[ring[1], ring[2], ring[3]], [ring[1], ring[3], ring[0]]], dtype=int),
        ]
    candidates: list[tuple[tuple[float, ...], np.ndarray]] = []
    for option in options:
        option = _orient_ccw(state.points, option)
        geometry = triangle_geometry(state.points, option)
        if np.any(geometry["signed_area"] <= _area_tolerance(state.points, option)):
            continue
        h = _triangle_targets(state.targets, option)
        l_over_h = np.max(geometry["edge_lengths"], axis=1) / np.maximum(h, 1.0e-12)
        min_angle = np.min(geometry["angles_deg"], axis=1)
        if (
            float(np.max(l_over_h)) > float(config.prune_l_over_h)
            or float(np.min(geometry["quality"])) < float(config.prune_min_quality)
            or float(np.min(min_angle)) < float(config.prune_min_angle_deg)
        ):
            continue
        simulated = valence.copy()
        simulated[np.asarray(ring, dtype=int)] -= 1
        for edge in _edge_set(option):
            if edge not in existing_edges:
                simulated[list(edge)] += 1
        if int(np.max(simulated[np.asarray(ring, dtype=int)])) > int(config.max_valence):
            continue
        score = (
            float(np.max(l_over_h)),
            -float(np.min(geometry["quality"])),
            -float(np.min(min_angle)),
        )
        candidates.append((score, option))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _repair_superthin(state: _State, config: AggressiveConditioningConfig, initial_components: int) -> dict[str, Any]:
    stage_before = _summary(state, config)
    before = stage_before
    ear_remove_count = 0
    boundary_weld_count = 0
    flip_count = 0
    collapse_count = 0
    boundary_split_count = 0
    boundary_remove_count = 0
    rejected = 0
    rejected_cases: list[dict[str, Any]] = []
    screened_rejections: list[dict[str, Any]] = []
    # Remove a redundant exterior ear only when each protected side remains
    # represented by another wet triangle.  This is the safe direct-removal
    # case requested for a near-collinear boundary closure.
    blocked_ears: set[int] = set()
    for _ in range(max(0, int(config.max_boundary_ear_removals_per_round))):
        candidate = _select_boundary_ear_triangle(state, config, blocked_ears)
        if candidate is None:
            break
        snapshot = state.clone()
        tri_nodes = state.triangles[int(candidate)].copy()
        signed_area_before = _signed_mesh_area(state.points, state.triangles)
        state.triangles = np.delete(state.triangles, int(candidate), axis=0)
        state.last_affected = list(map(int, tri_nodes))
        actual_area_change = abs(_signed_mesh_area(state.points, state.triangles) - signed_area_before)
        budget_ok = _boundary_area_budget_allows(state, actual_area_change, config)
        ok, invariant_report, trial = _audit_state(state, config, initial_components)
        if not budget_ok or not ok or not _nonregression(before, trial, purpose="thin"):
            _restore(state, snapshot)
            blocked_ears.add(int(candidate))
            rejected += 1
            rejection = _thin_rejection(
                "boundary-ear-remove",
                int(candidate),
                before,
                trial,
                ok,
                invariant_report=invariant_report,
            )
            if not budget_ok:
                rejection["failures"] = sorted(set([*rejection["failures"], "domain_area_budget"]))
            rejection["actual_signed_domain_area_change_m2"] = float(actual_area_change)
            rejected_cases.append(rejection)
            continue
        state.cumulative_boundary_area_change_m2 += actual_area_change
        state.ledger.append(
            {
                "operation": "boundary-ear-remove",
                "triangle_original_node_lineage": [int(state.lineage[node]) for node in tri_nodes],
                "actual_signed_domain_area_change_m2": float(actual_area_change),
            }
        )
        ear_remove_count += 1
        before = trial
        blocked_ears.clear()

    # If a coarse protected arc is the base of a sliver, weld the opposite
    # free vertex to its perpendicular projection on the arc and insert it in
    # the constraint chain.  The old sliver disappears and the arc gains the
    # resolution demanded by the local geometry.
    blocked_welds: set[tuple[int, int, int]] = set()
    for _ in range(max(0, int(config.max_boundary_welds_per_round))):
        proposal, screening = _select_boundary_weld(state, config, blocked_welds)
        screened_rejections.extend(screening)
        if proposal is None:
            break
        edge, node, triangle_index = proposal
        snapshot = state.clone()
        changed, construction_failures = _weld_vertex_to_boundary_arc(state, edge, node, triangle_index, config)
        if not changed:
            _restore(state, snapshot)
            blocked_welds.add((int(edge[0]), int(edge[1]), int(node)))
            rejected += 1
            rejected_cases.append(
                {
                    "operation": "boundary-arc-weld",
                    "payload": [int(edge[0]), int(edge[1]), int(node)],
                    "failures": construction_failures or ["weld_construction_failed"],
                }
            )
            continue
        _micro_relax(state, replacement_seed_nodes=state.last_affected, config=config)
        stabilized = _stabilize_after_boundary_removal(state, config)
        if stabilized:
            state.ledger.append(
                {"operation": "boundary-weld-local-valence-stabilize", "followup_edit_count": int(stabilized)}
            )
        ok, invariant_report, trial = _audit_state(state, config, initial_components)
        if not ok or not _nonregression(before, trial, purpose="thin"):
            _restore(state, snapshot)
            blocked_welds.add((int(edge[0]), int(edge[1]), int(node)))
            rejected += 1
            rejected_cases.append(
                _thin_rejection(
                    "boundary-arc-weld",
                    (*edge, node),
                    before,
                    trial,
                    ok,
                    invariant_report=invariant_report,
                )
            )
            continue
        boundary_weld_count += 1
        before = trial
        blocked_welds.clear()
    # First use the cheapest connectivity edit.  Unlike thin-repair-v1's
    # disjoint global batch, this loop re-evaluates the extreme-tail component
    # after every accepted flip and can therefore work through a zipper created
    # by a preceding valence transaction.
    blocked_flips: set[tuple[int, int]] = set()
    for _ in range(max(0, int(config.max_superthin_flips_per_round))):
        proposal = _select_superthin_flip(state, config, blocked_flips)
        if proposal is None:
            break
        edge, first, second, new_first, new_second = proposal
        snapshot = state.clone()
        state.triangles[int(first)] = new_first
        state.triangles[int(second)] = new_second
        state.triangles = _orient_ccw(state.points, state.triangles)
        state.last_affected = sorted(set(map(int, np.unique(state.triangles[[int(first), int(second)]]))))
        _micro_relax(state, replacement_seed_nodes=state.last_affected, config=config)
        ok, invariant_report, trial = _audit_state(state, config, initial_components)
        if not ok or not _nonregression(before, trial, purpose="thin"):
            _restore(state, snapshot)
            blocked_flips.add(tuple(edge))
            rejected += 1
            rejected_cases.append(
                _thin_rejection(
                    "superthin-edge-flip", edge, before, trial, ok, invariant_report=invariant_report
                )
            )
            continue
        state.ledger.append(
            {
                "operation": "superthin-edge-flip",
                "parent_edge_original_nodes": [int(state.lineage[edge[0]]), int(state.lineage[edge[1]])],
            }
        )
        flip_count += 1
        before = trial
        blocked_flips.clear()
    # Interior pair collapse: select the worst legal pair and re-evaluate after each edit.
    blocked_collapses: set[tuple[int, int]] = set()
    for _ in range(max(0, int(config.max_collapses_per_round))):
        candidate = _select_collapse_edge(state, config, blocked_collapses)
        if candidate is None:
            break
        edge = candidate
        snapshot = state.clone()
        if not _collapse_edge(state, edge):
            _restore(state, snapshot)
            blocked_collapses.add(tuple(edge))
            rejected += 1
            rejected_cases.append(
                {
                    "operation": "interior-edge-collapse",
                    "payload": list(map(int, edge)),
                    "failures": ["edit_construction_failed"],
                }
            )
            continue
        _micro_relax(state, replacement_seed_nodes=state.last_affected, config=config)
        ok, invariant_report, trial = _audit_state(state, config, initial_components)
        if not ok or not _nonregression(before, trial, purpose="thin"):
            _restore(state, snapshot)
            blocked_collapses.add(tuple(edge))
            rejected += 1
            rejected_cases.append(
                _thin_rejection(
                    "interior-edge-collapse", edge, before, trial, ok, invariant_report=invariant_report
                )
            )
            continue
        collapse_count += 1
        before = trial
        blocked_collapses.clear()
    # Boundary branch: one edit at a time, using the longest-side rule.
    if config.boundary_edit_policy != "none":
        blocked_boundary: set[tuple[str, tuple[int, ...]]] = set()
        for _ in range(max(0, int(config.max_boundary_edits_per_round))):
            proposal = _select_boundary_edit(state, config, blocked_boundary)
            if proposal is None:
                break
            snapshot = state.clone()
            operation, payload = proposal
            changed = _split_boundary_edge(state, payload, config) if operation == "split" else _remove_boundary_vertex(state, int(payload), config)
            if not changed:
                _restore(state, snapshot)
                blocked_boundary.add(_proposal_key(operation, payload))
                rejected += 1
                rejected_cases.append(
                    {
                        "operation": f"boundary-{operation}",
                        "payload": list(_proposal_key(operation, payload)[1]),
                        "failures": ["edit_construction_failed"],
                    }
                )
                continue
            _micro_relax(state, replacement_seed_nodes=state.last_affected, config=config)
            ok, invariant_report, trial = _audit_state(state, config, initial_components)
            if not ok or not _nonregression(before, trial, purpose="thin"):
                _restore(state, snapshot)
                blocked_boundary.add(_proposal_key(operation, payload))
                rejected += 1
                rejected_cases.append(
                    _thin_rejection(
                        f"boundary-{operation}",
                        payload,
                        before,
                        trial,
                        ok,
                        invariant_report=invariant_report,
                    )
                )
                continue
            if operation == "split":
                boundary_split_count += 1
            else:
                boundary_remove_count += 1
            before = trial
            blocked_boundary.clear()
    return {
        "accepted": int(
            ear_remove_count
            + boundary_weld_count
            + flip_count
            + collapse_count
            + boundary_split_count
            + boundary_remove_count
        ),
        "boundary_ear_removals": int(ear_remove_count),
        "boundary_arc_welds": int(boundary_weld_count),
        "superthin_edge_flips": int(flip_count),
        "interior_pair_collapses": int(collapse_count),
        "boundary_edge_splits": int(boundary_split_count),
        "boundary_vertex_removals": int(boundary_remove_count),
        "rejected": int(rejected),
        "rejected_cases": rejected_cases,
        "screened_rejections": _deduplicate_rejections(screened_rejections),
        "audit_scope": "candidate_patch_precheck_plus_global_transaction_audit",
        "before": stage_before,
        "after": _summary(state, config),
    }


def _select_boundary_ear_triangle(
    state: _State,
    config: AggressiveConditioningConfig,
    excluded: set[int],
) -> int | None:
    topology = build_edge_topology(len(state.points), state.triangles)
    geometry = triangle_geometry(state.points, state.triangles)
    min_angles = np.min(geometry["angles_deg"], axis=1)
    superthin = (geometry["quality"] < float(config.superthin_quality_threshold)) | (
        min_angles < float(config.superthin_min_angle_deg)
    )
    protected = chain_edges(state.chains)
    ordering = sorted(
        np.where(superthin)[0],
        key=lambda index: (float(geometry["quality"][index]), float(min_angles[index]), int(index)),
    )
    for index in ordering:
        if int(index) in excluded:
            continue
        tri = state.triangles[int(index)]
        if not bool(np.all(state.fixed[tri])):
            continue
        kind_values = " ".join(str(state.kinds[int(node)]).lower() for node in tri)
        if any(
            str(token).lower() in kind_values
            for token in config.boundary_weld_forbidden_kind_tokens
            if str(token)
        ):
            # Closing or shaving an under-resolved channel is an upstream
            # semantic decision, never an automatic triangle-deletion route.
            continue
        edges = _triangle_edge_keys(tri)
        protected_edges = [edge for edge in edges if edge in protected]
        free_edges = [edge for edge in edges if edge not in protected]
        if len(protected_edges) < 2 or len(free_edges) != 1:
            continue
        # The protected chain must survive on the retained wet side, and the
        # unprotected chord must be an exterior edge belonging only to this ear.
        if not all(len(topology.edge_to_triangles.get(edge, [])) >= 2 for edge in protected_edges):
            continue
        if len(topology.edge_to_triangles.get(free_edges[0], [])) != 1:
            continue
        return int(index)
    return None


def _select_boundary_weld(
    state: _State,
    config: AggressiveConditioningConfig,
    excluded: set[tuple[int, int, int]],
) -> tuple[tuple[tuple[int, int], int, int] | None, list[dict[str, Any]]]:
    topology = build_edge_topology(len(state.points), state.triangles)
    geometry = triangle_geometry(state.points, state.triangles)
    min_angles = np.min(geometry["angles_deg"], axis=1)
    superthin = (geometry["quality"] < float(config.superthin_quality_threshold)) | (
        min_angles < float(config.superthin_min_angle_deg)
    )
    protected = chain_edges(state.chains)
    proposals: list[tuple[float, float, tuple[int, int], int, int]] = []
    screened: list[dict[str, Any]] = []
    for index in np.where(superthin)[0]:
        tri = state.triangles[int(index)]
        for edge in _triangle_edge_keys(tri):
            if edge not in protected or len(topology.edge_to_triangles.get(edge, [])) != 1:
                continue
            opposite = [int(node) for node in tri if int(node) not in edge]
            if len(opposite) != 1:
                continue
            node = opposite[0]
            key = (int(edge[0]), int(edge[1]), int(node))
            if key in excluded:
                continue
            weld_geometry, failures = _boundary_weld_geometry(state, edge, node, config)
            if failures or weld_geometry is None:
                screened.append(
                    {
                        "operation": "boundary-arc-weld",
                        "payload": [int(edge[0]), int(edge[1]), int(node)],
                        "triangle_index": int(index),
                        "failures": failures or ["invalid_source_arc_projection"],
                    }
                )
                continue
            _, _, distance, h = weld_geometry
            proposals.append((float(geometry["quality"][index]), distance / h, edge, node, int(index)))
    if not proposals:
        return None, screened
    _, _, edge, node, index = min(proposals, key=lambda value: (value[0], value[1], value[4]))
    return (edge, int(node), int(index)), screened


def _weld_vertex_to_boundary_arc(
    state: _State,
    edge: tuple[int, int],
    node: int,
    triangle_index: int,
    config: AggressiveConditioningConfig,
) -> tuple[bool, list[str]]:
    chain_pos = _find_chain_edge(state.chains, edge)
    weld_geometry, failures = _boundary_weld_geometry(state, edge, node, config)
    if chain_pos is None or failures or weld_geometry is None:
        return False, failures or ["constraint_chain_edge_not_found"]
    chain_index, position = chain_pos
    a, b = map(int, edge)
    fraction, projection, distance, _ = weld_geometry
    topology = build_edge_topology(len(state.points), state.triangles)
    attached = topology.edge_to_triangles.get(tuple(sorted(edge)), [])
    if attached != [int(triangle_index)] and set(attached) != {int(triangle_index)}:
        return False, ["source_arc_not_exterior_edge"]
    tri = state.triangles[int(triangle_index)]
    if int(node) not in tri or not {a, b}.issubset(set(map(int, tri))):
        return False, ["source_triangle_mismatch"]
    signed_area_before = _signed_mesh_area(state.points, state.triangles)
    affected = set(map(int, topology.node_neighbors[int(node)]))
    affected.update((a, b, int(node)))
    old_point = state.points[int(node)].copy()
    state.points[int(node)] = projection
    state.fixed[int(node)] = True
    state.hard[int(node)] = False
    state.kinds[int(node)] = state.kinds[a]
    state.targets[int(node)] = _sample_target_at(state, projection, fallback=_edge_target(state.targets, edge))
    state.chains[chain_index].insert(position + 1, int(node))
    open_values = state.open_nodes.tolist()
    for open_position, (left, right) in enumerate(zip(open_values[:-1], open_values[1:])):
        if {int(left), int(right)} == {a, b}:
            open_values.insert(open_position + 1, int(node))
            state.open_nodes = np.asarray(open_values, dtype=int)
            break
    state.triangles = np.delete(state.triangles, int(triangle_index), axis=0)
    state.triangles = _orient_ccw(state.points, state.triangles)
    actual_area_change = abs(_signed_mesh_area(state.points, state.triangles) - signed_area_before)
    if not _boundary_area_budget_allows(state, actual_area_change, config):
        return False, ["domain_area_budget"]
    state.last_affected = sorted(affected)
    state.cumulative_boundary_area_change_m2 += actual_area_change
    state.ledger.append(
        {
            "operation": "boundary-arc-weld",
            "welded_original_node": int(state.lineage[int(node)]),
            "parent_edge_original_nodes": [int(state.lineage[a]), int(state.lineage[b])],
            "boundary_kind": state.kinds[a],
            "projection_fraction": fraction,
            "weld_distance_m": distance,
            "original_coordinate_xy": [float(old_point[0]), float(old_point[1])],
            "actual_signed_domain_area_change_m2": float(actual_area_change),
            "target_spacing_resampled": bool(state.target_sampler is not None),
        }
    )
    return True, []


def _boundary_weld_geometry(
    state: _State,
    edge: tuple[int, int],
    node: int,
    config: AggressiveConditioningConfig,
) -> tuple[tuple[float, np.ndarray, float, float] | None, list[str]]:
    """Project to the immutable source arc and return all semantic guard failures."""
    failures: list[str] = []
    a, b = map(int, edge)
    if not (0 <= int(node) < len(state.points)) or state.fixed[int(node)] or state.hard[int(node)]:
        failures.append("weld_vertex_fixed_or_hard")
    if not _same_boundary_kind(state, edge):
        failures.append("mixed_boundary_kind")
    source_edge = _source_arc_edge(state, edge)
    if source_edge is None:
        failures.append("source_arc_edge_not_found")
        return None, failures
    source_a, source_b = source_edge
    vector = source_b - source_a
    denominator = float(np.dot(vector, vector))
    if denominator <= 1.0e-20:
        failures.append("degenerate_source_arc")
        return None, failures
    fraction = float(np.dot(state.points[int(node)] - source_a, vector) / denominator)
    if not 0.02 < fraction < 0.98:
        failures.append("projection_outside_source_arc_interior")
    projection = source_a + fraction * vector
    distance = float(np.linalg.norm(state.points[int(node)] - projection))
    h = max(_edge_target(state.targets, edge), 1.0e-12)
    arc_length = float(np.sqrt(denominator))
    if distance > float(config.boundary_weld_max_distance_fraction) * h:
        failures.append("weld_distance_fraction")
    if distance > float(config.boundary_weld_max_altitude_to_arc_fraction) * arc_length:
        failures.append("weld_altitude_to_arc")
    kind = str(state.kinds[a]).lower()
    absolute_limit = (
        float(config.boundary_weld_open_max_distance_m)
        if kind in {"open", "frame"}
        else float(config.boundary_weld_land_max_distance_m)
    )
    if distance > absolute_limit:
        failures.append("weld_absolute_distance")
    tokens = tuple(str(value).lower() for value in config.boundary_weld_forbidden_kind_tokens)
    if any(token and token in kind for token in tokens):
        failures.append("under_resolved_channel_or_junction_requires_upstream_review")
    chain_pos = _find_chain_edge(state.chains, edge)
    if chain_pos is None:
        failures.append("constraint_chain_edge_not_found")
    else:
        chain_index, position = chain_pos
        chain = state.chains[chain_index]
        anchor_buffer = max(0, int(config.boundary_weld_anchor_buffer_segments))
        nearby = _cyclic_chain_window(chain, position, anchor_buffer)
        if any(state.hard[value] for value in nearby):
            failures.append("hard_anchor_buffer")
        junction_buffer = max(0, int(config.boundary_weld_junction_buffer_segments))
        junction_nodes = _cyclic_chain_window(chain, position, junction_buffer)
        junction_kinds = {str(state.kinds[value]) for value in junction_nodes}
        if len(junction_kinds) > 1:
            failures.append("boundary_kind_junction_buffer")
    clearance = _minimum_remote_boundary_clearance(state, projection, edge)
    if np.isfinite(clearance) and clearance < float(config.boundary_weld_channel_clearance_fraction) * h:
        failures.append("narrow_channel_semantic_guard")
    return (fraction, projection, distance, h), sorted(set(failures))


def _select_superthin_flip(
    state: _State,
    config: AggressiveConditioningConfig,
    excluded: set[tuple[int, int]],
) -> tuple[tuple[int, int], int, int, np.ndarray, np.ndarray] | None:
    topology = build_edge_topology(len(state.points), state.triangles)
    geometry = triangle_geometry(state.points, state.triangles)
    min_angles = np.min(geometry["angles_deg"], axis=1)
    superthin = (geometry["quality"] < float(config.superthin_quality_threshold)) | (
        min_angles < float(config.superthin_min_angle_deg)
    )
    protected = chain_edges(state.chains)
    ordering = sorted(
        np.where(superthin)[0],
        key=lambda index: (float(geometry["quality"][index]), float(min_angles[index]), int(index)),
    )
    for index in ordering:
        tri = state.triangles[int(index)]
        for edge in sorted(_triangle_edge_keys(tri)):
            if edge in excluded or edge in protected:
                continue
            attached = list(map(int, topology.edge_to_triangles.get(edge, [])))
            if len(attached) != 2:
                continue
            first, second = attached
            candidate = _edge_flip_candidate(state.points, state.triangles, edge, first, second)
            if candidate is None:
                continue
            new_first, new_second, old_q, new_q, old_angle, new_angle, new_edge = candidate
            if new_edge in topology.edge_to_triangles:
                continue
            if new_q <= old_q + 1.0e-8 or new_angle <= old_angle + 0.05:
                continue
            return edge, first, second, new_first, new_second
    return None


def _proposal_key(operation: str, payload: Any) -> tuple[str, tuple[int, ...]]:
    if isinstance(payload, tuple):
        values = tuple(map(int, payload))
    elif isinstance(payload, (list, np.ndarray)):
        values = tuple(map(int, payload))
    else:
        values = (int(payload),)
    return str(operation), values


def _thin_rejection(
    operation: str,
    payload: Any,
    before: dict[str, Any],
    trial: dict[str, Any],
    invariants_ok: bool,
    *,
    invariant_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    if not invariants_ok:
        failures.extend(_failed_invariant_names(invariant_report or {}))
    if trial["superthin_triangle_count"] > before["superthin_triangle_count"]:
        failures.append("superthin_count_increase")
    if trial["thin_severity_sum"] >= before["thin_severity_sum"] - 1.0e-10:
        failures.append("no_thin_severity_drop")
    if trial["q_min"] + 1.0e-12 < before["q_min"]:
        failures.append("q_min_regression")
    if trial["q_p01"] + 1.0e-9 < before["q_p01"]:
        failures.append("q_p01_regression")
    if trial["minimum_angle_deg"] + 1.0e-8 < before["minimum_angle_deg"]:
        failures.append("minimum_angle_regression")
    if trial["l_over_h_p95"] > 1.001 * max(before["l_over_h_p95"], 1.0e-12):
        failures.append("l_over_h_p95_regression")
    if trial["l_over_h_count_above_1_55"] > before["l_over_h_count_above_1_55"]:
        failures.append("l_over_h_excess_count_increase")
    if trial["count_valence_above_limit"] > before["count_valence_above_limit"]:
        failures.append("valence_gate_regression")
    if trial["singly_connected_triangle_count"] > before["singly_connected_triangle_count"]:
        failures.append("new_singly_connected_triangles")
    if trial["boundary_degree_anomaly_count"] > before["boundary_degree_anomaly_count"]:
        failures.append("new_boundary_degree_anomalies")
    if trial["boundary_component_count"] != before["boundary_component_count"]:
        failures.append("boundary_traversability_component_change")
    return {
        "operation": str(operation),
        "payload": list(_proposal_key(operation, payload)[1]),
        "failures": failures or ["compound_nonregression_gate"],
        "before": _summary_from(before),
        "trial": _summary_from(trial),
    }


def _select_collapse_edge(
    state: _State,
    config: AggressiveConditioningConfig,
    excluded: set[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    topology = build_edge_topology(len(state.points), state.triangles)
    geometry = triangle_geometry(state.points, state.triangles)
    min_angles = np.min(geometry["angles_deg"], axis=1)
    superthin = (geometry["quality"] < float(config.superthin_quality_threshold)) | (min_angles < float(config.superthin_min_angle_deg))
    protected = chain_edges(state.chains)
    candidates: list[tuple[float, tuple[int, int]]] = []
    for edge, attached in topology.edge_to_triangles.items():
        if excluded and tuple(edge) in excluded:
            continue
        if len(attached) != 2 or edge in protected or state.fixed[edge[0]] or state.fixed[edge[1]]:
            continue
        if not all(bool(superthin[int(index)]) for index in attached):
            continue
        if not _link_condition(topology, edge, attached):
            continue
        h = _edge_target(state.targets, edge)
        length = float(np.linalg.norm(state.points[edge[0]] - state.points[edge[1]]))
        if not np.isfinite(h) or length / max(h, 1.0e-12) > float(config.collapse_l_over_h):
            continue
        candidates.append((length / max(h, 1.0e-12), edge))
    return min(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def _collapse_edge(state: _State, edge: tuple[int, int]) -> bool:
    a, b = map(int, edge)
    midpoint = 0.5 * (state.points[a] + state.points[b])
    state.points[a] = midpoint
    state.targets[a] = _edge_target(state.targets, edge)
    state.triangles[state.triangles == b] = a
    keep = np.asarray([len(set(map(int, tri))) == 3 for tri in state.triangles], dtype=bool)
    state.triangles = _orient_ccw(state.points, state.triangles[keep])
    state.ledger.append(
        {
            "operation": "interior-edge-collapse",
            "kept_original_node": int(state.lineage[a]),
            "removed_original_node": int(state.lineage[b]),
            "parent_edge_original_nodes": [int(state.lineage[a]), int(state.lineage[b])],
        }
    )
    state.last_affected = sorted(set(map(int, np.unique(state.triangles[np.any(state.triangles == a, axis=1)]))))
    _compact(state)
    return True


def _select_boundary_edit(
    state: _State,
    config: AggressiveConditioningConfig,
    excluded: set[tuple[str, tuple[int, ...]]] | None = None,
) -> tuple[str, Any] | None:
    topology = build_edge_topology(len(state.points), state.triangles)
    geometry = triangle_geometry(state.points, state.triangles)
    min_angles = np.min(geometry["angles_deg"], axis=1)
    superthin = np.where(
        (geometry["quality"] < float(config.superthin_quality_threshold))
        | (min_angles < float(config.superthin_min_angle_deg))
    )[0]
    protected = chain_edges(state.chains)
    ordering = sorted(superthin, key=lambda index: (float(geometry["quality"][index]), float(min_angles[index]), int(index)))
    for index in ordering:
        tri = state.triangles[int(index)]
        edges = [
            (tuple(sorted((int(tri[1]), int(tri[2])))), float(geometry["edge_lengths"][index, 0])),
            (tuple(sorted((int(tri[0]), int(tri[2])))), float(geometry["edge_lengths"][index, 1])),
            (tuple(sorted((int(tri[0]), int(tri[1])))), float(geometry["edge_lengths"][index, 2])),
        ]
        longest = max(edges, key=lambda item: item[1])[0]
        if longest in protected:
            attached = topology.edge_to_triangles.get(longest, [])
            if len(attached) == 1 and _same_boundary_kind(state, longest):
                proposal = ("split", longest)
                if not excluded or _proposal_key(*proposal) not in excluded:
                    return proposal
            continue
        if config.boundary_edit_policy == "split-only":
            continue
        candidates = [int(node) for node in tri if state.fixed[int(node)] and not state.hard[int(node)]]
        candidates = [node for node in candidates if _boundary_removal_allowed(state, node, config)]
        if candidates:
            for node in sorted(candidates, key=lambda value: _boundary_deviation(state, value)):
                proposal = ("remove", node)
                if not excluded or _proposal_key(*proposal) not in excluded:
                    return proposal
    return None


def _split_boundary_edge(state: _State, edge: tuple[int, int], config: AggressiveConditioningConfig) -> bool:
    topology = build_edge_topology(len(state.points), state.triangles)
    attached = topology.edge_to_triangles.get(tuple(sorted(edge)), [])
    if len(attached) != 1:
        return False
    chain_pos = _find_chain_edge(state.chains, edge)
    if chain_pos is None:
        return False
    chain_index, position = chain_pos
    a, b = map(int, edge)
    tri_index = int(attached[0])
    opposite = [int(value) for value in state.triangles[tri_index] if int(value) not in edge]
    if len(opposite) != 1:
        return False
    new_node = len(state.points)
    state.points = np.vstack([state.points, 0.5 * (state.points[a] + state.points[b])])
    state.fixed = np.concatenate([state.fixed, np.asarray([True])])
    state.targets = np.concatenate([state.targets, np.asarray([_edge_target(state.targets, edge)])])
    state.kinds.append(state.kinds[a])
    state.hard = np.concatenate([state.hard, np.asarray([False])])
    state.lineage = np.concatenate([state.lineage, _new_lineage_ids(state, 1)])
    state.chains[chain_index].insert(position + 1, int(new_node))
    if a in set(state.open_nodes.tolist()) and b in set(state.open_nodes.tolist()):
        open_values = state.open_nodes.tolist()
        for index, (left, right) in enumerate(zip(open_values[:-1], open_values[1:])):
            if {int(left), int(right)} == {a, b}:
                open_values.insert(index + 1, int(new_node))
                state.open_nodes = np.asarray(open_values, dtype=int)
                break
    keep = np.ones(len(state.triangles), dtype=bool)
    keep[tri_index] = False
    c = opposite[0]
    additions = np.asarray([[a, new_node, c], [new_node, b, c]], dtype=int)
    state.triangles = _orient_ccw(state.points, np.vstack([state.triangles[keep], additions]))
    state.last_affected = [a, b, c, int(new_node)]
    state.ledger.append(
        {
            "operation": "boundary-edge-split",
            "new_node_before_compaction": int(new_node),
            "parent_edge_original_nodes": [int(state.lineage[a]), int(state.lineage[b])],
            "boundary_kind": state.kinds[a],
        }
    )
    return True


def _remove_boundary_vertex(state: _State, node: int, config: AggressiveConditioningConfig) -> bool:
    membership = _find_chain_node(state.chains, int(node))
    if membership is None:
        return False
    chain_index, position = membership
    chain = state.chains[chain_index]
    previous = int(chain[position - 1])
    following = int(chain[(position + 1) % len(chain)])
    incident = np.where(np.any(state.triangles == int(node), axis=1))[0]
    fan = _ordered_boundary_fan(state.triangles[incident], int(node), previous, following)
    if fan is None or len(fan) < 3:
        return False
    replacement = _triangulate_ring_greedy(state.points, fan, None, int(config.max_valence))
    if replacement is None:
        return False
    area_change = 0.5 * abs(float(np.cross(state.points[node] - state.points[previous], state.points[following] - state.points[previous])))
    if (
        state.cumulative_boundary_area_change_m2 + area_change
        > float(config.maximum_domain_area_change_fraction) * max(_mesh_area(state.points, state.triangles), 1.0e-30)
    ):
        return False
    keep = np.ones(len(state.triangles), dtype=bool)
    keep[incident] = False
    state.triangles = _orient_ccw(state.points, np.vstack([state.triangles[keep], replacement]))
    removed_original = int(state.lineage[node])
    chain.pop(position)
    state.open_nodes = state.open_nodes[state.open_nodes != int(node)]
    state.cumulative_boundary_area_change_m2 += area_change
    state.ledger.append(
        {
            "operation": "boundary-vertex-remove",
            "removed_original_node": removed_original,
            "boundary_kind": state.kinds[node],
            "chord_deviation_m": float(_point_segment_distance(state.points[node], state.points[previous], state.points[following])),
            "local_area_change_m2": float(area_change),
        }
    )
    state.last_affected = [int(value) for value in fan]
    _compact(state)
    return True


def _stabilize_after_boundary_removal(
    state: _State,
    config: AggressiveConditioningConfig,
    *,
    max_edits: int = 4,
) -> int:
    """Repair valence transferred to the interior by a boundary-node removal.

    Removing a boundary fan can reduce the selected node's valence while adding
    diagonals to one or more fan nodes.  Auditing those as separate transactions
    rejects the useful first edit because the hard violation has merely moved.
    Keep the operation atomic and clear any newly overloaded node in the
    affected patch before the global acceptance gate is evaluated.  Boundary
    coordinates remain fixed during the preferred spoke-flip follow-up.
    """
    accepted = 0
    affected_lineage = {
        int(state.lineage[node])
        for node in state.last_affected
        if 0 <= int(node) < len(state.lineage)
    }
    for _ in range(max(0, int(max_edits))):
        topology = build_edge_topology(len(state.points), state.triangles)
        valence = np.asarray([len(values) for values in topology.node_neighbors], dtype=int)
        affected = {
            int(node)
            for node, lineage in enumerate(state.lineage)
            if int(lineage) in affected_lineage
        }
        local = set(affected)
        for node in affected:
            local.update(map(int, topology.node_neighbors[node]))
        candidates = [
            int(node)
            for node in local
            if valence[node] > int(config.max_valence)
        ]
        if not candidates:
            break
        node = max(candidates, key=lambda value: (int(valence[value]), -int(value)))
        changed, _ = _attempt_high_valence_edit(
            state,
            node,
            topology,
            valence,
            config,
            skip_boundary_removal=True,
        )
        if not changed:
            break
        accepted += 1
        affected_lineage.update(
            int(state.lineage[value])
            for value in state.last_affected
            if 0 <= int(value) < len(state.lineage)
        )
    return int(accepted)


def _repair_high_valence(state: _State, config: AggressiveConditioningConfig, initial_components: int) -> dict[str, Any]:
    before = _summary(state, config)
    topology = build_edge_topology(len(state.points), state.triangles)
    flip_batches = _repair_valence_flip_batches(
        state,
        config,
        initial_components,
        topology=topology,
        baseline=before,
        budget=max(0, int(config.max_valence_removals_per_round)),
    )
    cluster_merges = _repair_valence_cluster_cavities(
        state,
        config,
        initial_components,
        topology=flip_batches["topology"],
        baseline=flip_batches["baseline"],
        node_budget=max(0, int(config.max_valence_removals_per_round) - len(flip_batches["accepted_node_lineage"])),
    )
    accepted = int(flip_batches["accepted"] + cluster_merges["accepted"])
    rejected = 0
    blocked_lineage: set[int] = set()
    attempted_lineage: list[int] = list(flip_batches["accepted_node_lineage"]) + list(cluster_merges["attempted_node_lineage"])
    rejected_cases: list[dict[str, Any]] = []
    topology = cluster_merges["topology"]
    before = cluster_merges["baseline"]
    filter_values = set(map(int, config.valence_node_lineage_filter))
    remaining_budget = max(0, int(config.max_valence_removals_per_round) - len(attempted_lineage))
    for _ in range(remaining_budget):
        valence = np.asarray([len(values) for values in topology.node_neighbors], dtype=int)
        candidates = np.asarray(
            [
                int(node)
                for node in np.where(valence > int(config.max_valence))[0]
                if _lineage_key(state, int(node)) not in blocked_lineage
                and (not filter_values or _lineage_key(state, int(node)) in filter_values)
            ],
            dtype=int,
        )
        if not len(candidates):
            break
        node = int(max(candidates, key=lambda value: (int(valence[value]), -int(value))))
        node_key = _lineage_key(state, node)
        attempted_lineage.append(node_key)
        snapshot = state.clone()
        changed, operation = _attempt_high_valence_edit(state, node, topology, valence, config)
        if not changed:
            rejected += 1
            blocked_lineage.add(node_key)
            rejected_cases.append(
                {
                    "node_lineage": int(node_key),
                    "node_index_zero_based": int(node),
                    "valence": int(valence[node]),
                    "reason": "no_legal_local_edit",
                    "attempted_operation": operation,
                }
            )
            continue
        _micro_relax(state, replacement_seed_nodes=state.last_affected, config=config)
        trial_geometry = triangle_geometry(state.points, state.triangles)
        trial_topology = build_edge_topology(len(state.points), state.triangles)
        ok, invariant_report = _state_invariants(
            state,
            initial_components,
            geometry=trial_geometry,
            topology=trial_topology,
        )
        trial = _summary(state, config, geometry=trial_geometry, topology=trial_topology)
        nonregression = _nonregression(
            before,
            trial,
            purpose="valence",
            max_l_over_h_count_increase=int(config.max_valence_l_over_h_count_increase),
        )
        if (not ok or not nonregression) and (
            operation == "high-valence-edge-flip" or operation.startswith("boundary-vertex-remove")
        ):
            _restore(state, snapshot)
            alternative_changed, alternative_operation = _attempt_high_valence_edit(
                state,
                node,
                topology,
                valence,
                config,
                skip_flip=True,
                skip_boundary_removal=operation.startswith("boundary-vertex-remove"),
            )
            if alternative_changed:
                _micro_relax(state, replacement_seed_nodes=state.last_affected, config=config)
                alternative_geometry = triangle_geometry(state.points, state.triangles)
                alternative_topology = build_edge_topology(len(state.points), state.triangles)
                alternative_ok, alternative_invariants = _state_invariants(
                    state,
                    initial_components,
                    geometry=alternative_geometry,
                    topology=alternative_topology,
                )
                alternative_trial = _summary(
                    state,
                    config,
                    geometry=alternative_geometry,
                    topology=alternative_topology,
                )
                alternative_nonregression = _nonregression(
                    before,
                    alternative_trial,
                    purpose="valence",
                    max_l_over_h_count_increase=int(config.max_valence_l_over_h_count_increase),
                )
                if alternative_ok and alternative_nonregression:
                    accepted += 1
                    before = alternative_trial
                    topology = alternative_topology
                    continue
                invariant_report = alternative_invariants
                trial = alternative_trial
                ok = alternative_ok
                nonregression = alternative_nonregression
                operation = f"high-valence-edge-flip->{alternative_operation}"
            else:
                operation = f"high-valence-edge-flip->{alternative_operation}"
        if not ok or not nonregression:
            _restore(state, snapshot)
            rejected += 1
            blocked_lineage.add(node_key)
            rejected_cases.append(
                {
                    "node_lineage": int(node_key),
                    "node_index_zero_based": int(node),
                    "valence": int(valence[node]),
                    "reason": "invariant_gate" if not ok else "quality_target_nonregression_gate",
                    "attempted_operation": operation,
                    "before": before,
                    "trial": trial,
                    "invariants": invariant_report,
                }
            )
            continue
        accepted += 1
        before = trial
        topology = trial_topology
    final = _summary(state, config, topology=topology)
    remaining = int(final["count_valence_above_limit"])
    return {
        "accepted": int(accepted),
        "rejected": int(rejected),
        "attempted_count": int(len(attempted_lineage)),
        "attempted_node_lineage": attempted_lineage,
        "rejected_cases": rejected_cases,
        "flip_batches": flip_batches["batches"],
        "cluster_merges": cluster_merges["transactions"],
        "remaining_violation_count": remaining,
        "budget_exhausted_with_unattempted_nodes": bool(
            len(attempted_lineage) >= int(config.max_valence_removals_per_round) and remaining > len(blocked_lineage)
        ),
        "after": final,
    }


def _repair_valence_flip_batches(
    state: _State,
    config: AggressiveConditioningConfig,
    initial_components: int,
    *,
    topology: Any,
    baseline: dict[str, Any],
    budget: int,
) -> dict[str, Any]:
    accepted = 0
    accepted_lineage: list[int] = []
    batches: list[dict[str, Any]] = []
    maximum_batch = max(0, int(config.max_valence_flip_batch))
    while accepted < int(budget) and maximum_batch > 0:
        valence = np.asarray([len(values) for values in topology.node_neighbors], dtype=int)
        filter_values = set(map(int, config.valence_node_lineage_filter))
        violators = [
            int(node)
            for node in np.where(valence > int(config.max_valence))[0]
            if not filter_values or _lineage_key(state, int(node)) in filter_values
        ]
        if not violators:
            break
        protected = chain_edges(state.chains)
        proposals: list[tuple[tuple[float, ...], int, tuple[Any, ...]]] = []
        for node in violators:
            option = _best_valence_flip(
                state,
                node,
                int(config.max_valence),
                topology,
                valence,
                protected=protected,
            )
            if option is not None:
                proposals.append(((-float(valence[node]), *option[0], float(node)), node, option))
        if not proposals:
            break
        proposals.sort(key=lambda item: item[0])
        disjoint: list[tuple[int, tuple[Any, ...]]] = []
        occupied_triangles: set[int] = set()
        occupied_nodes: set[int] = set()
        for _, node, option in proposals:
            _, first, second, new_first, new_second, _ = option
            patch_triangles = {int(first), int(second)}
            patch_nodes = set(map(int, np.concatenate([new_first, new_second, state.triangles[[first, second]].ravel()])))
            if patch_triangles & occupied_triangles or patch_nodes & occupied_nodes:
                continue
            disjoint.append((int(node), option))
            occupied_triangles.update(patch_triangles)
            occupied_nodes.update(patch_nodes)
            if len(disjoint) >= min(maximum_batch, int(budget) - accepted):
                break
        if not disjoint:
            break
        trial_count = len(disjoint)
        batch_accepted = False
        while trial_count >= 2:
            selected = disjoint[:trial_count]
            snapshot = state.clone()
            for node, option in selected:
                _, first, second, new_first, new_second, new_edge = option
                state.triangles[int(first)] = np.asarray(new_first, dtype=int)
                state.triangles[int(second)] = np.asarray(new_second, dtype=int)
                state.ledger.append(
                    {
                        "operation": "high-valence-edge-flip",
                        "node_original_id": int(state.lineage[node]),
                        "valence_before": int(valence[node]),
                        "new_edge_original_nodes": [int(state.lineage[new_edge[0]]), int(state.lineage[new_edge[1]])],
                        "batch_size": int(trial_count),
                    }
                )
            geometry = triangle_geometry(state.points, state.triangles)
            trial_topology = build_edge_topology(len(state.points), state.triangles)
            ok, invariant_report = _state_invariants(
                state,
                initial_components,
                geometry=geometry,
                topology=trial_topology,
            )
            trial_summary = _summary(state, config, geometry=geometry, topology=trial_topology)
            nonregression = _nonregression(
                baseline,
                trial_summary,
                purpose="valence",
                max_l_over_h_count_increase=int(config.max_valence_l_over_h_count_increase),
            )
            if ok and nonregression:
                lineage = [int(state.lineage[node]) for node, _ in selected]
                accepted += int(trial_count)
                accepted_lineage.extend(lineage)
                batches.append(
                    {
                        "accepted": True,
                        "operation_count": int(trial_count),
                        "node_lineage": lineage,
                        "before": baseline,
                        "after": trial_summary,
                    }
                )
                baseline = trial_summary
                topology = trial_topology
                batch_accepted = True
                break
            _restore(state, snapshot)
            batches.append(
                {
                    "accepted": False,
                    "operation_count": int(trial_count),
                    "reason": "invariant_gate" if not ok else "quality_target_nonregression_gate",
                    "invariants": invariant_report,
                }
            )
            trial_count //= 2
        if not batch_accepted:
            break
    return {
        "accepted": int(accepted),
        "accepted_node_lineage": accepted_lineage,
        "batches": batches,
        "topology": topology,
        "baseline": baseline,
    }


def _repair_valence_cluster_cavities(
    state: _State,
    config: AggressiveConditioningConfig,
    initial_components: int,
    *,
    topology: Any,
    baseline: dict[str, Any],
    node_budget: int,
) -> dict[str, Any]:
    accepted = 0
    processed_nodes = 0
    attempted_lineage: list[int] = []
    transactions: list[dict[str, Any]] = []
    blocked: set[tuple[int, ...]] = set()
    filter_values = set(map(int, config.valence_node_lineage_filter))
    for _ in range(max(0, int(config.max_valence_cluster_merges_per_round))):
        if processed_nodes >= int(node_budget):
            break
        valence = np.asarray([len(values) for values in topology.node_neighbors], dtype=int)
        violations = {int(value) for value in np.where(valence > int(config.max_valence))[0]}
        components: list[list[int]] = []
        unseen = set(violations)
        while unseen:
            start = min(unseen)
            unseen.remove(start)
            stack = [start]
            component: list[int] = []
            while stack:
                current = stack.pop()
                component.append(int(current))
                following = sorted((topology.node_neighbors[current] & violations) & unseen)
                for value in following:
                    unseen.remove(int(value))
                    stack.append(int(value))
            if 2 <= len(component) <= 8 and not any(state.fixed[node] for node in component):
                lineage_key = tuple(sorted(int(state.lineage[node]) for node in component))
                if lineage_key not in blocked and (not filter_values or set(lineage_key).issubset(filter_values)):
                    components.append(sorted(component))
        if not components:
            break
        components.sort(key=lambda values: (-sum(int(valence[node] - config.max_valence) for node in values), -len(values), values[0]))
        cluster = components[0]
        lineage_key = tuple(sorted(int(state.lineage[node]) for node in cluster))
        if processed_nodes + len(cluster) > int(node_budget):
            break
        snapshot = state.clone()
        cavity = _ordered_cluster_cavity(state.triangles, set(cluster))
        if cavity is None:
            blocked.add(lineage_key)
            attempted_lineage.extend(lineage_key)
            processed_nodes += len(cluster)
            transactions.append({"accepted": False, "node_lineage": list(lineage_key), "reason": "non_simple_cluster_cavity"})
            continue
        incident, ring = cavity
        replacement, steiner_nodes = _distributed_cluster_cavity(
            state,
            cluster,
            ring,
            incident,
            valence,
            topology,
            int(config.max_valence),
        )
        attempted_lineage.extend(lineage_key)
        processed_nodes += len(cluster)
        if replacement is None:
            _restore(state, snapshot)
            blocked.add(lineage_key)
            transactions.append({"accepted": False, "node_lineage": list(lineage_key), "reason": "no_quality_safe_cluster_cavity"})
            continue
        keep = np.ones(len(state.triangles), dtype=bool)
        keep[incident] = False
        state.triangles = _orient_ccw(state.points, np.vstack([state.triangles[keep], replacement]))
        state.ledger.append(
            {
                "operation": "high-valence-cluster-cavity-merge",
                "removed_original_nodes": list(lineage_key),
                "cluster_size": int(len(cluster)),
                "replacement_triangle_count": int(len(replacement)),
                "steiner_node_count": int(len(steiner_nodes)),
            }
        )
        state.last_affected = [int(value) for value in ring] + steiner_nodes
        _compact(state)
        _micro_relax(state, replacement_seed_nodes=state.last_affected, config=config)
        geometry = triangle_geometry(state.points, state.triangles)
        trial_topology = build_edge_topology(len(state.points), state.triangles)
        ok, invariant_report = _state_invariants(
            state,
            initial_components,
            geometry=geometry,
            topology=trial_topology,
        )
        trial_summary = _summary(state, config, geometry=geometry, topology=trial_topology)
        nonregression = _nonregression(
            baseline,
            trial_summary,
            purpose="valence",
            max_l_over_h_count_increase=int(config.max_valence_l_over_h_count_increase),
        )
        if not ok or not nonregression:
            _restore(state, snapshot)
            blocked.add(lineage_key)
            transactions.append(
                {
                    "accepted": False,
                    "node_lineage": list(lineage_key),
                    "reason": "invariant_gate" if not ok else "quality_target_nonregression_gate",
                    "invariants": invariant_report,
                    "trial": trial_summary,
                }
            )
            continue
        accepted += 1
        baseline = trial_summary
        topology = trial_topology
        transactions.append(
            {
                "accepted": True,
                "node_lineage": list(lineage_key),
                "cluster_size": int(len(cluster)),
                "steiner_node_count": int(len(steiner_nodes)),
                "after": trial_summary,
            }
        )
    return {
        "accepted": int(accepted),
        "processed_node_count": int(processed_nodes),
        "attempted_node_lineage": attempted_lineage,
        "transactions": transactions,
        "topology": topology,
        "baseline": baseline,
    }


def _ordered_cluster_cavity(triangles: np.ndarray, cluster: set[int]) -> tuple[np.ndarray, list[int]] | None:
    incident = np.where(np.any(np.isin(triangles, np.asarray(sorted(cluster), dtype=int)), axis=1))[0]
    if not len(incident):
        return None
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in np.asarray(triangles, dtype=int)[incident]:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if not adjacency or any(len(values) != 2 for values in adjacency.values()) or any(node in cluster for node in adjacency):
        return None
    start = min(adjacency)
    ring = [start]
    previous = -1
    current = start
    for _ in range(len(adjacency) + 1):
        choices = sorted(value for value in adjacency[current] if value != previous)
        if not choices:
            return None
        following = choices[0]
        if following == start:
            break
        if following in ring:
            return None
        ring.append(int(following))
        previous, current = current, following
    if len(ring) != len(adjacency):
        return None
    return np.asarray(incident, dtype=int), ring


def _attempt_high_valence_edit(
    state: _State,
    node: int,
    topology: Any,
    valence: np.ndarray,
    config: AggressiveConditioningConfig,
    *,
    skip_flip: bool = False,
    skip_boundary_removal: bool = False,
) -> tuple[bool, str]:
    changed = False if skip_flip else _try_valence_flip(state, node, int(config.max_valence), topology=topology, valence=valence)
    if changed:
        return True, "high-valence-edge-flip"
    if state.fixed[node]:
        if not skip_boundary_removal and not state.hard[node] and _boundary_removal_allowed(state, node, config):
            if _remove_boundary_vertex(state, node, config):
                stabilized = _stabilize_after_boundary_removal(state, config)
                if stabilized:
                    state.ledger.append(
                        {
                            "operation": "boundary-removal-local-valence-stabilize",
                            "followup_edit_count": int(stabilized),
                        }
                    )
                    return True, "boundary-vertex-remove+local-valence-stabilize"
                return True, "boundary-vertex-remove"
        membership = _find_chain_node(state.chains, int(node))
        if membership is None:
            return False, "fixed-node-not-in-constraint-chain"
        chain_index, position = membership
        chain = state.chains[chain_index]
        previous = int(chain[position - 1])
        following = int(chain[(position + 1) % len(chain)])
        incident = np.where(np.any(state.triangles == int(node), axis=1))[0]
        fan = _ordered_boundary_fan(state.triangles[incident], int(node), previous, following)
        if fan is None or len(fan) < 3:
            return False, "unordered-boundary-fan"
        replacement, steiner_nodes = _distributed_boundary_fan(
            state,
            node,
            fan,
            valence,
            topology,
            int(config.max_valence),
        )
        if replacement is None:
            return False, "no_quality_safe_boundary_fan"
        keep = np.ones(len(state.triangles), dtype=bool)
        keep[incident] = False
        state.triangles = _orient_ccw(state.points, np.vstack([state.triangles[keep], replacement]))
        state.ledger.append(
            {
                "operation": "high-valence-boundary-fan-redistribute",
                "node_original_id": int(state.lineage[node]),
                "valence_before": int(valence[node]),
                "replacement_triangle_count": int(len(replacement)),
                "steiner_node_count": int(len(steiner_nodes)),
                "boundary_kind": str(state.kinds[node]),
            }
        )
        state.last_affected = [int(value) for value in fan] + [int(node)] + steiner_nodes
        return True, "high-valence-boundary-fan-redistribute"
    incident = np.where(np.any(state.triangles == int(node), axis=1))[0]
    ring = _ordered_one_ring(state.triangles[incident], node)
    if ring is None:
        return False, "unordered-one-ring"
    replacement = _triangulate_ring_greedy(state.points, ring, valence, int(config.max_valence), removed_node=node)
    old_geometry = triangle_geometry(state.points, state.triangles[incident])
    use_plain = False
    if replacement is not None:
        new_geometry = triangle_geometry(state.points, replacement)
        h = _triangle_targets(state.targets, replacement)
        l_over_h = np.max(new_geometry["edge_lengths"], axis=1) / np.maximum(h, 1.0e-12)
        use_plain = bool(
            float(np.min(new_geometry["quality"])) + 1.0e-12 >= float(np.min(old_geometry["quality"]))
            and float(np.max(l_over_h)) <= 1.55
        )
    steiner_nodes: list[int] = []
    if not use_plain:
        replacement, steiner_nodes = _distributed_steiner_cavity(state, node, ring, valence, int(config.max_valence))
    if replacement is None:
        return False, "no_quality_safe_cavity"
    keep = np.ones(len(state.triangles), dtype=bool)
    keep[incident] = False
    state.triangles = _orient_ccw(state.points, np.vstack([state.triangles[keep], replacement]))
    state.ledger.append(
        {
            "operation": "high-valence-cavity-remove",
            "removed_original_node": int(state.lineage[node]),
            "valence_before": int(valence[node]),
            "replacement_triangle_count": int(len(replacement)),
            "steiner_node_count": int(len(steiner_nodes)),
        }
    )
    state.last_affected = [int(value) for value in ring] + steiner_nodes
    _compact(state)
    return True, "high-valence-cavity-remove"


def _lineage_key(state: _State, node: int) -> int:
    """Return a compaction-stable key for an original node or a unique inserted node."""
    return int(state.lineage[int(node)])


def _try_valence_flip(
    state: _State,
    node: int,
    limit: int,
    *,
    topology: Any | None = None,
    valence: np.ndarray | None = None,
) -> bool:
    topology = topology if topology is not None else build_edge_topology(len(state.points), state.triangles)
    valence = np.asarray(valence, dtype=int) if valence is not None else np.asarray([len(values) for values in topology.node_neighbors], dtype=int)
    option = _best_valence_flip(state, node, limit, topology, valence)
    if option is None:
        return False
    _, first, second, new_first, new_second, new_edge = option
    state.triangles[first] = new_first
    state.triangles[second] = new_second
    state.last_affected = sorted(set(map(int, np.unique(state.triangles[[first, second]]))))
    state.ledger.append(
        {
            "operation": "high-valence-edge-flip",
            "node_original_id": int(state.lineage[node]),
            "valence_before": int(valence[node]),
            "new_edge_original_nodes": [int(state.lineage[new_edge[0]]), int(state.lineage[new_edge[1]])],
        }
    )
    return True


def _best_valence_flip(
    state: _State,
    node: int,
    limit: int,
    topology: Any,
    valence: np.ndarray,
    protected: set[tuple[int, int]] | None = None,
) -> tuple[tuple[float, ...], int, int, np.ndarray, np.ndarray, tuple[int, int]] | None:
    protected = protected if protected is not None else chain_edges(state.chains)
    options: list[tuple[tuple[float, ...], int, int, np.ndarray, np.ndarray, tuple[int, int]]] = []
    for neighbor in sorted(topology.node_neighbors[int(node)]):
        edge = tuple(sorted((int(node), int(neighbor))))
        attached = topology.edge_to_triangles.get(edge, [])
        if edge in protected or len(attached) != 2:
            continue
        first, second = map(int, attached)
        c_values = [int(value) for value in state.triangles[first] if int(value) not in edge]
        d_values = [int(value) for value in state.triangles[second] if int(value) not in edge]
        if len(c_values) != 1 or len(d_values) != 1 or c_values[0] == d_values[0]:
            continue
        c, d = c_values[0], d_values[0]
        new_edge = tuple(sorted((c, d)))
        if new_edge in topology.edge_to_triangles:
            continue
        new_pair = _orient_ccw(state.points, np.asarray([[c, d, edge[0]], [d, c, edge[1]]], dtype=int))
        old_geometry = triangle_geometry(state.points, state.triangles[[first, second]])
        new_geometry = triangle_geometry(state.points, new_pair)
        if np.any(new_geometry["signed_area"] <= _area_tolerance(state.points, new_pair)):
            continue
        predicted_c = int(valence[c] + 1)
        predicted_d = int(valence[d] + 1)
        if predicted_c > int(limit) or predicted_d > int(limit):
            continue
        old_min = float(np.min(old_geometry["quality"]))
        new_min = float(np.min(new_geometry["quality"]))
        if new_min + 1.0e-12 < 0.90 * old_min:
            continue
        score = (float(max(predicted_c, predicted_d)), -new_min, float(c), float(d))
        options.append((score, first, second, new_pair[0], new_pair[1], new_edge))
    return min(options, key=lambda item: item[0]) if options else None


def _triangulate_ring_greedy(
    points: np.ndarray,
    ring: list[int],
    valence: np.ndarray | None,
    limit: int,
    *,
    removed_node: int | None = None,
) -> np.ndarray | None:
    if len(ring) < 3:
        return None
    vertices = ring.copy()
    polygon = points[np.asarray(vertices, dtype=int)]
    area2 = float(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - np.roll(polygon[:, 0], -1) * polygon[:, 1]))
    if area2 < 0.0:
        vertices.reverse()
    simulated = np.asarray(valence, dtype=int).copy() if valence is not None else np.zeros(len(points), dtype=int)
    if valence is not None and removed_node is not None:
        simulated[np.asarray(vertices, dtype=int)] -= 1
    existing = {tuple(sorted((int(vertices[i]), int(vertices[(i + 1) % len(vertices)])))) for i in range(len(vertices))}
    output: list[list[int]] = []
    while len(vertices) > 3:
        ears: list[tuple[tuple[float, ...], int, list[int], tuple[int, int]]] = []
        for index, current in enumerate(vertices):
            previous = int(vertices[index - 1])
            following = int(vertices[(index + 1) % len(vertices)])
            triangle = [previous, int(current), following]
            geometry = triangle_geometry(points, np.asarray([triangle], dtype=int))
            if geometry["signed_area"][0] <= _area_tolerance(points, np.asarray([triangle], dtype=int)):
                continue
            if any(_point_in_triangle(points[value], points[np.asarray(triangle, dtype=int)]) for value in vertices if value not in triangle):
                continue
            diagonal = tuple(sorted((previous, following)))
            add = 0 if diagonal in existing else 1
            pred_prev = int(simulated[previous] + add)
            pred_next = int(simulated[following] + add)
            excess = max(0, pred_prev - int(limit)) ** 2 + max(0, pred_next - int(limit)) ** 2
            min_angle = float(np.min(geometry["angles_deg"]))
            quality = float(geometry["quality"][0])
            score = (float(excess), float(max(pred_prev, pred_next)), -quality, -min_angle, float(index))
            ears.append((score, index, triangle, diagonal))
        if not ears:
            return None
        _, index, triangle, diagonal = min(ears, key=lambda item: item[0])
        if diagonal not in existing:
            simulated[list(diagonal)] += 1
            existing.add(diagonal)
        output.append(triangle)
        vertices.pop(index)
    output.append(vertices)
    result = _orient_ccw(points, np.asarray(output, dtype=int))
    if np.any(triangle_geometry(points, result)["signed_area"] <= _area_tolerance(points, result)):
        return None
    return result


def _distributed_steiner_cavity(
    state: _State,
    node: int,
    ring: list[int],
    valence: np.ndarray,
    limit: int,
) -> tuple[np.ndarray | None, list[int]]:
    """Partition an overloaded star among two to eight centroidal nodes."""
    n = len(ring)
    if n < 5:
        return None, []
    ring_points = state.points[np.asarray(ring, dtype=int)]
    centered = ring_points - np.mean(ring_points, axis=0)
    covariance = centered.T @ centered
    try:
        _, eigenvectors = np.linalg.eigh(covariance)
        axis = np.asarray(eigenvectors[:, -1], dtype=float)
    except np.linalg.LinAlgError:
        axis = np.asarray([1.0, 0.0])
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    minimum_radius = float(np.min(np.linalg.norm(ring_points - state.points[node], axis=1)))
    h_node = max(float(state.targets[node]), 1.0e-12)
    old_area = float(np.sum(triangle_geometry(state.points, state.triangles[np.any(state.triangles == int(node), axis=1)])["area"]))
    existing = _edge_set(state.triangles)
    candidates: list[tuple[tuple[float, ...], np.ndarray, np.ndarray]] = []
    maximum_steiner = min(8, max(2, int(np.ceil(n / 2.0))))
    for count in range(2, maximum_steiner + 1):
        if int(np.ceil(n / count)) + 3 > int(limit):
            continue
        for fraction in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
            radius = min(float(fraction) * minimum_radius, 0.30 * h_node)
            for start in range(n):
                ordered = ring[start:] + ring[:start]
                base = n // count
                remainder = n % count
                sector_sizes = [base + (1 if index < remainder else 0) for index in range(count)]
                cuts = [0]
                for size in sector_sizes:
                    cuts.append(cuts[-1] + int(size))
                sector_directions: list[np.ndarray] = []
                for sector in range(count):
                    values = [ordered[index % n] for index in range(cuts[sector], cuts[sector + 1] + 1)]
                    centroid = np.mean(state.points[np.asarray(values, dtype=int)], axis=0)
                    direction = centroid - state.points[node]
                    norm = max(float(np.linalg.norm(direction)), 1.0e-12)
                    sector_directions.append(direction / norm)
                steiner_points = state.points[node] + radius * np.asarray(sector_directions)
                trial_points = np.vstack([state.points, steiner_points])
                steiner_ids = list(range(len(state.points), len(state.points) + count))
                additions: list[list[int]] = []
                for sector, steiner in enumerate(steiner_ids):
                    for index in range(cuts[sector], cuts[sector + 1]):
                        additions.append([int(steiner), int(ordered[index]), int(ordered[(index + 1) % n])])
                    cut_node = int(ordered[cuts[sector] % n])
                    additions.append([int(steiner_ids[sector - 1]), int(steiner), cut_node])
                for index in range(1, count - 1):
                    additions.append([int(steiner_ids[0]), int(steiner_ids[index]), int(steiner_ids[index + 1])])
                candidate = _orient_ccw(trial_points, np.asarray(additions, dtype=int))
                geometry = triangle_geometry(trial_points, candidate)
                if np.any(geometry["signed_area"] <= _area_tolerance(trial_points, candidate)):
                    continue
                if abs(float(np.sum(geometry["area"])) - old_area) > 1.0e-8 * max(old_area, 1.0):
                    continue
                simulated = np.asarray(valence, dtype=int).copy()
                simulated[np.asarray(ring, dtype=int)] -= 1
                new_edges = _edge_set(candidate)
                local_new_neighbors = {steiner: set() for steiner in steiner_ids}
                for edge in new_edges:
                    if edge not in existing:
                        for endpoint in edge:
                            if endpoint < len(simulated):
                                simulated[endpoint] += 1
                    for steiner in steiner_ids:
                        if steiner in edge:
                            local_new_neighbors[steiner].update(value for value in edge if value != steiner)
                if int(np.max(simulated[np.asarray(ring, dtype=int)])) > int(limit):
                    continue
                local_new_valence = {key: len(values) for key, values in local_new_neighbors.items()}
                if max(local_new_valence.values(), default=0) > int(limit):
                    continue
                target_values = np.concatenate([state.targets, np.full(count, h_node)])
                h = _triangle_targets(target_values, candidate)
                l_over_h = np.max(geometry["edge_lengths"], axis=1) / np.maximum(h, 1.0e-12)
                score = (
                    float(max(np.max(simulated[np.asarray(ring, dtype=int)]), max(local_new_valence.values(), default=0))),
                    float(np.count_nonzero(l_over_h > 1.55)),
                    -float(np.min(geometry["quality"])),
                    float(np.max(l_over_h)),
                    float(count),
                    float(radius),
                    float(start),
                )
                candidates.append((score, steiner_points, candidate))
    if not candidates:
        return None, []
    _, steiner_points, candidate = min(candidates, key=lambda item: item[0])
    new_ids = list(range(len(state.points), len(state.points) + len(steiner_points)))
    state.points = np.vstack([state.points, steiner_points])
    state.fixed = np.concatenate([state.fixed, np.zeros(len(steiner_points), dtype=bool)])
    state.targets = np.concatenate([state.targets, np.full(len(steiner_points), h_node)])
    state.kinds.extend(["interior"] * len(steiner_points))
    state.hard = np.concatenate([state.hard, np.zeros(len(steiner_points), dtype=bool)])
    state.lineage = np.concatenate([state.lineage, _new_lineage_ids(state, len(steiner_points))])
    return candidate, new_ids


def _distributed_boundary_fan(
    state: _State,
    node: int,
    fan: list[int],
    valence: np.ndarray,
    topology: Any,
    limit: int,
) -> tuple[np.ndarray | None, list[int]]:
    """Keep a boundary node fixed while distributing its interior fan connectivity."""
    n = len(fan)
    if n < 3:
        return None, []
    incident = np.where(np.any(state.triangles == int(node), axis=1))[0]
    incident_set = set(map(int, incident.tolist()))
    old_area = float(np.sum(triangle_geometry(state.points, state.triangles[incident])["area"]))
    minimum_radius = float(np.min(np.linalg.norm(state.points[np.asarray(fan, dtype=int)] - state.points[node], axis=1)))
    h_node = max(float(state.targets[node]), 1.0e-12)
    existing_edges = set(topology.edge_to_triangles)
    removed_edges = {
        edge
        for edge, attached in topology.edge_to_triangles.items()
        if attached and set(map(int, attached)).issubset(incident_set)
    }
    surviving_edges = existing_edges - removed_edges
    edge_count = n - 1
    candidates: list[tuple[tuple[float, ...], np.ndarray, np.ndarray, np.ndarray]] = []
    for count in range(2, min(8, edge_count) + 1):
        if count + 2 > int(limit):
            continue
        for interior_cuts in combinations(range(1, edge_count), count - 1):
            cuts = [0, *map(int, interior_cuts), edge_count]
            sizes = [cuts[index + 1] - cuts[index] for index in range(count)]
            if min(sizes) <= 0 or max(sizes) + 4 > int(limit):
                continue
            directions: list[np.ndarray] = []
            for sector in range(count):
                values = fan[cuts[sector] : cuts[sector + 1] + 1]
                centroid = np.mean(state.points[np.asarray(values, dtype=int)], axis=0)
                direction = centroid - state.points[node]
                norm = max(float(np.linalg.norm(direction)), 1.0e-12)
                directions.append(direction / norm)
            for fraction in (0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.36, 0.42):
                radius = min(float(fraction) * minimum_radius, 0.40 * h_node)
                sector_points = state.points[node] + radius * np.asarray(directions)
                mean_direction = np.mean(np.asarray(directions), axis=0)
                mean_direction /= max(float(np.linalg.norm(mean_direction)), 1.0e-12)
                center_point = state.points[node] + 0.45 * radius * mean_direction
                steiner_points = np.vstack([sector_points, center_point])
                steiner_targets = _interpolate_local_targets(state, steiner_points, [int(node), *fan])
                trial_points = np.vstack([state.points, steiner_points])
                steiner_ids = list(range(len(state.points), len(state.points) + len(steiner_points)))
                sector_ids = steiner_ids[:count]
                center_id = int(steiner_ids[-1])
                additions: list[list[int]] = []
                for sector, steiner in enumerate(sector_ids):
                    for index in range(cuts[sector], cuts[sector + 1]):
                        additions.append([int(steiner), int(fan[index]), int(fan[index + 1])])
                    if sector > 0:
                        additions.append([int(sector_ids[sector - 1]), int(steiner), int(fan[cuts[sector]])])
                additions.append([int(node), int(fan[0]), int(sector_ids[0])])
                additions.append([int(node), int(sector_ids[0]), center_id])
                for sector in range(count - 1):
                    additions.append([center_id, int(sector_ids[sector]), int(sector_ids[sector + 1])])
                additions.append([int(node), center_id, int(sector_ids[-1])])
                additions.append([int(node), int(sector_ids[-1]), int(fan[-1])])
                candidate = _orient_ccw(trial_points, np.asarray(additions, dtype=int))
                geometry = triangle_geometry(trial_points, candidate)
                if np.any(geometry["signed_area"] <= _area_tolerance(trial_points, candidate)):
                    continue
                if abs(float(np.sum(geometry["area"])) - old_area) > 1.0e-8 * max(old_area, 1.0):
                    continue
                simulated = np.concatenate(
                    [np.asarray(valence, dtype=int).copy(), np.zeros(len(steiner_points), dtype=int)]
                )
                for edge in removed_edges:
                    simulated[list(edge)] -= 1
                for edge in _edge_set(candidate):
                    if edge not in surviving_edges:
                        simulated[list(edge)] += 1
                local_nodes = np.asarray([int(node), *map(int, fan), *steiner_ids], dtype=int)
                if int(np.max(simulated[local_nodes])) > int(limit):
                    continue
                target_values = np.concatenate([state.targets, steiner_targets])
                h = _triangle_targets(target_values, candidate)
                l_over_h = np.max(geometry["edge_lengths"], axis=1) / np.maximum(h, 1.0e-12)
                score = (
                    float(np.max(simulated[local_nodes])),
                    float(np.count_nonzero(l_over_h > 1.55)),
                    -float(np.min(geometry["quality"])),
                    float(np.max(l_over_h)),
                    float(len(steiner_points)),
                    float(radius),
                )
                candidates.append((score, steiner_points, steiner_targets, candidate))
    if not candidates:
        return None, []
    _, steiner_points, steiner_targets, candidate = min(candidates, key=lambda item: item[0])
    new_ids = list(range(len(state.points), len(state.points) + len(steiner_points)))
    state.points = np.vstack([state.points, steiner_points])
    state.fixed = np.concatenate([state.fixed, np.zeros(len(steiner_points), dtype=bool)])
    state.targets = np.concatenate([state.targets, steiner_targets])
    state.kinds.extend(["interior"] * len(steiner_points))
    state.hard = np.concatenate([state.hard, np.zeros(len(steiner_points), dtype=bool)])
    state.lineage = np.concatenate([state.lineage, _new_lineage_ids(state, len(steiner_points))])
    return candidate, new_ids


def _distributed_cluster_cavity(
    state: _State,
    cluster: list[int],
    ring: list[int],
    incident: np.ndarray,
    valence: np.ndarray,
    topology: Any,
    limit: int,
) -> tuple[np.ndarray | None, list[int]]:
    """Replace a connected interior zipper cluster with a distributed local net."""
    n = len(ring)
    if n < 5:
        return None, []
    center = np.mean(state.points[np.asarray(cluster, dtype=int)], axis=0)
    old_area = float(np.sum(triangle_geometry(state.points, state.triangles[incident])["area"]))
    minimum_radius = float(np.min(np.linalg.norm(state.points[np.asarray(ring, dtype=int)] - center, axis=1)))
    h_cluster = max(float(np.median(state.targets[np.asarray(cluster, dtype=int)])), 1.0e-12)
    incident_set = set(map(int, np.asarray(incident, dtype=int).tolist()))
    existing_edges = set(topology.edge_to_triangles)
    removed_edges = {
        edge
        for edge, attached in topology.edge_to_triangles.items()
        if attached and set(map(int, attached)).issubset(incident_set)
    }
    surviving_edges = existing_edges - removed_edges
    candidates: list[tuple[tuple[float, ...], np.ndarray, np.ndarray, np.ndarray]] = []
    minimum_count = max(2, min(len(cluster), 6))
    for count in range(minimum_count, min(8, max(minimum_count, int(np.ceil(n / 2.0)))) + 1):
        if int(np.ceil(n / count)) + 3 > int(limit):
            continue
        for fraction in (0.10, 0.15, 0.20, 0.25, 0.30, 0.36, 0.42):
            radius = min(float(fraction) * minimum_radius, 0.40 * h_cluster)
            for start in range(n):
                ordered = ring[start:] + ring[:start]
                base = n // count
                remainder = n % count
                sizes = [base + (1 if index < remainder else 0) for index in range(count)]
                cuts = [0]
                for size in sizes:
                    cuts.append(cuts[-1] + int(size))
                directions: list[np.ndarray] = []
                for sector in range(count):
                    values = [ordered[index % n] for index in range(cuts[sector], cuts[sector + 1] + 1)]
                    centroid = np.mean(state.points[np.asarray(values, dtype=int)], axis=0)
                    direction = centroid - center
                    norm = max(float(np.linalg.norm(direction)), 1.0e-12)
                    directions.append(direction / norm)
                sector_points = center + radius * np.asarray(directions)
                steiner_points = np.vstack([sector_points, center])
                steiner_targets = _interpolate_local_targets(state, steiner_points, [*cluster, *ring])
                trial_points = np.vstack([state.points, steiner_points])
                steiner_ids = list(range(len(state.points), len(state.points) + len(steiner_points)))
                sector_ids = steiner_ids[:count]
                center_id = int(steiner_ids[-1])
                additions: list[list[int]] = []
                for sector, steiner in enumerate(sector_ids):
                    for index in range(cuts[sector], cuts[sector + 1]):
                        additions.append([int(steiner), int(ordered[index]), int(ordered[(index + 1) % n])])
                    cut_node = int(ordered[cuts[sector] % n])
                    additions.append([int(sector_ids[sector - 1]), int(steiner), cut_node])
                for index in range(count):
                    additions.append([center_id, int(sector_ids[index]), int(sector_ids[(index + 1) % count])])
                candidate = _orient_ccw(trial_points, np.asarray(additions, dtype=int))
                geometry = triangle_geometry(trial_points, candidate)
                if np.any(geometry["signed_area"] <= _area_tolerance(trial_points, candidate)):
                    continue
                if abs(float(np.sum(geometry["area"])) - old_area) > 1.0e-8 * max(old_area, 1.0):
                    continue
                simulated = np.concatenate([np.asarray(valence, dtype=int).copy(), np.zeros(len(steiner_points), dtype=int)])
                for edge in removed_edges:
                    simulated[list(edge)] -= 1
                for edge in _edge_set(candidate):
                    if edge not in surviving_edges:
                        simulated[list(edge)] += 1
                local_nodes = np.asarray([*map(int, ring), *steiner_ids], dtype=int)
                if int(np.max(simulated[local_nodes])) > int(limit):
                    continue
                target_values = np.concatenate([state.targets, steiner_targets])
                h = _triangle_targets(target_values, candidate)
                l_over_h = np.max(geometry["edge_lengths"], axis=1) / np.maximum(h, 1.0e-12)
                score = (
                    float(np.max(simulated[local_nodes])),
                    float(np.count_nonzero(l_over_h > 1.55)),
                    float(max(0, len(steiner_points) - len(cluster))),
                    float(len(steiner_points)),
                    -float(np.min(geometry["quality"])),
                    float(np.max(l_over_h)),
                    float(radius),
                    float(start),
                )
                candidates.append((score, steiner_points, steiner_targets, candidate))
    if not candidates:
        return None, []
    _, steiner_points, steiner_targets, candidate = min(candidates, key=lambda item: item[0])
    new_ids = list(range(len(state.points), len(state.points) + len(steiner_points)))
    state.points = np.vstack([state.points, steiner_points])
    state.fixed = np.concatenate([state.fixed, np.zeros(len(steiner_points), dtype=bool)])
    state.targets = np.concatenate([state.targets, steiner_targets])
    state.kinds.extend(["interior"] * len(steiner_points))
    state.hard = np.concatenate([state.hard, np.zeros(len(steiner_points), dtype=bool)])
    state.lineage = np.concatenate([state.lineage, _new_lineage_ids(state, len(steiner_points))])
    return candidate, new_ids


def _micro_relax(state: _State, replacement_seed_nodes: list[int], config: AggressiveConditioningConfig) -> None:
    if config.micro_relax_cycles <= 0 or config.micro_relax_iterations <= 0 or not len(state.triangles):
        return
    seed_nodes = {int(value) for value in replacement_seed_nodes if 0 <= int(value) < len(state.points)}
    if not seed_nodes:
        return
    topology = build_edge_topology(len(state.points), state.triangles)
    distance = {node: 0 for node in seed_nodes}
    frontier = set(seed_nodes)
    patch_layers = max(1, int(config.micro_relax_ring_layers)) + 2
    for layer in range(1, patch_layers + 1):
        following = {
            int(neighbor)
            for node in frontier
            for neighbor in topology.node_neighbors[int(node)]
            if int(neighbor) not in distance
        }
        for node in following:
            distance[node] = int(layer)
        frontier = following
        if not frontier:
            break
    patch_nodes = np.asarray(sorted(distance), dtype=int)
    in_patch = np.zeros(len(state.points), dtype=bool)
    in_patch[patch_nodes] = True
    triangle_mask = np.all(in_patch[state.triangles], axis=1)
    global_triangle_ids = np.where(triangle_mask)[0]
    if not len(global_triangle_ids):
        return
    mapping = np.full(len(state.points), -1, dtype=int)
    mapping[patch_nodes] = np.arange(len(patch_nodes), dtype=int)
    local_triangles = mapping[state.triangles[global_triangle_ids]]
    local_fixed = state.fixed[patch_nodes].copy()
    anchor_layer = max(distance.values())
    for local_node, global_node in enumerate(patch_nodes):
        if distance[int(global_node)] >= anchor_layer or any(not in_patch[int(value)] for value in topology.node_neighbors[int(global_node)]):
            local_fixed[local_node] = True
    local_seed_nodes = {int(mapping[node]) for node in seed_nodes if mapping[node] >= 0}
    seed_mask = np.asarray([any(int(node) in local_seed_nodes for node in tri) for tri in local_triangles], dtype=bool)
    if not np.any(seed_mask):
        return
    local_points = state.points[patch_nodes].copy()
    for _ in range(max(0, int(config.micro_relax_cycles))):
        if state.target_sampler is not None:
            state.targets[patch_nodes] = _sample_targets(state, local_points, fallback=state.targets[patch_nodes])
        relaxed = relax_mesh_spring(
            local_points,
            local_triangles,
            local_fixed,
            target_spacing_m=state.targets[patch_nodes],
            constraint_chains=[],
            open_boundary_nodes_zero_based=np.empty(0, dtype=int),
            seed_triangle_mask=seed_mask,
            config=SpringRelaxConfig(
                enabled=True,
                quality_threshold=0.40,
                min_angle_deg=28.0,
                ring_layers=int(config.micro_relax_ring_layers),
                iterations=int(config.micro_relax_iterations),
                damping=float(config.micro_relax_damping),
                max_step_fraction=float(config.micro_relax_max_step_fraction),
                shape_weight=float(config.micro_relax_shape_weight),
                force_tolerance=1.0e-3,
            ),
        )
        if not relaxed.report.get("accepted"):
            break
        local_points = relaxed.nodes_xy
        state.points[patch_nodes] = local_points
    if state.target_sampler is not None:
        state.targets[patch_nodes] = _sample_targets(state, state.points[patch_nodes], fallback=state.targets[patch_nodes])


def _summary(
    state: _State,
    config: AggressiveConditioningConfig,
    *,
    geometry: dict[str, np.ndarray] | None = None,
    topology: Any | None = None,
) -> dict[str, Any]:
    geometry = geometry if geometry is not None else triangle_geometry(state.points, state.triangles)
    topology = topology if topology is not None else build_edge_topology(len(state.points), state.triangles)
    quality = geometry["quality"]
    min_angles = np.min(geometry["angles_deg"], axis=1) if len(state.triangles) else np.empty(0)
    valence = np.asarray([len(values) for values in topology.node_neighbors], dtype=int)
    h = _triangle_targets(state.targets, state.triangles)
    l_over_h = np.max(geometry["edge_lengths"], axis=1) / np.maximum(h, 1.0e-12) if len(state.triangles) else np.empty(0)
    q_mean = float(np.mean(quality)) if len(quality) else 0.0
    q_std = float(np.std(quality)) if len(quality) else 0.0
    superthin = (quality < float(config.superthin_quality_threshold)) | (min_angles < float(config.superthin_min_angle_deg))
    thin = (quality < float(config.quality_threshold)) | (min_angles < float(config.min_angle_deg))
    boundary_audit = _boundary_graph_audit(topology)
    protected = chain_edges(state.chains)
    protected_not_boundary = sum(len(topology.edge_to_triangles.get(edge, [])) != 1 for edge in protected)
    superthin_severity = float(
        np.sum(
            np.maximum(
                0.0,
                (float(config.superthin_quality_threshold) - quality)
                / max(float(config.superthin_quality_threshold), 1.0e-12),
            )
            ** 2
        )
        + np.sum(
            np.maximum(
                0.0,
                (float(config.superthin_min_angle_deg) - min_angles)
                / max(float(config.superthin_min_angle_deg), 1.0e-12),
            )
            ** 2
        )
    )
    return {
        "node_count": int(len(state.points)),
        "triangle_count": int(len(state.triangles)),
        "q_min": float(np.min(quality)) if len(quality) else 0.0,
        "q_p01": float(np.quantile(quality, 0.01)) if len(quality) else 0.0,
        "q_mean": q_mean,
        "q_l3_sigma": float(q_mean - 3.0 * q_std),
        "minimum_angle_deg": float(np.min(min_angles)) if len(min_angles) else 0.0,
        "minimum_angle_p01_deg": float(np.quantile(min_angles, 0.01)) if len(min_angles) else 0.0,
        "thin_triangle_count": int(np.count_nonzero(thin)),
        "superthin_triangle_count": int(np.count_nonzero(superthin)),
        "superthin_severity_sum": superthin_severity,
        "thin_severity_sum": float(
            np.sum(np.maximum(0.0, (float(config.quality_threshold) - quality) / max(float(config.quality_threshold), 1.0e-12)) ** 2)
            + np.sum(np.maximum(0.0, (float(config.min_angle_deg) - min_angles) / max(float(config.min_angle_deg), 1.0e-12)) ** 2)
        ),
        "maximum_valence": int(np.max(valence)) if len(valence) else 0,
        "count_valence_above_limit": int(np.count_nonzero(valence > int(config.max_valence))),
        "valence_excess_sum": int(np.sum(np.maximum(0, valence - int(config.max_valence)) ** 2)),
        "l_over_h_p95": float(np.quantile(l_over_h, 0.95)) if len(l_over_h) else 0.0,
        "l_over_h_maximum": float(np.max(l_over_h)) if len(l_over_h) else 0.0,
        "l_over_h_count_above_1_55": int(np.count_nonzero(l_over_h > 1.55)),
        "connected_component_count": int(len(topology.connected_component_sizes)),
        "nonmanifold_edge_count": int(len(topology.nonmanifold_edges)),
        "nonpositive_signed_area_count": int(np.count_nonzero(geometry["signed_area"] <= _area_tolerance(state.points, state.triangles))),
        "singly_connected_triangle_count": int(np.count_nonzero(topology.triangle_neighbor_count == 1)),
        "boundary_degree_anomaly_count": int(boundary_audit["degree_anomaly_count"]),
        "boundary_component_count": int(boundary_audit["component_count"]),
        "protected_edge_not_boundary_count": int(protected_not_boundary),
    }


def _summary_from(value: dict[str, Any]) -> dict[str, Any]:
    return dict(value)


def _nonregression(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    purpose: str,
    max_l_over_h_count_increase: int = 0,
) -> bool:
    topology_ok = bool(
        after["singly_connected_triangle_count"] <= before["singly_connected_triangle_count"]
        and after["boundary_degree_anomaly_count"] <= before["boundary_degree_anomaly_count"]
        and after["boundary_component_count"] == before["boundary_component_count"]
        and after["protected_edge_not_boundary_count"] <= before["protected_edge_not_boundary_count"]
    )
    if purpose == "valence":
        # The FVCOM connectivity cap is a hard structural gate.  A perfectly
        # regular nine-spoke star cannot be reduced to valence eight without
        # changing its quality distribution, so use absolute local-quality
        # floors while still protecting target-size excess.
        return bool(
            topology_ok
            and
            (after["count_valence_above_limit"] < before["count_valence_above_limit"] or after["valence_excess_sum"] < before["valence_excess_sum"])
            and after["q_min"] + 1.0e-12 >= min(before["q_min"], 0.25)
            and after["minimum_angle_deg"] + 1.0e-8 >= min(before["minimum_angle_deg"], 20.0)
            and after["l_over_h_p95"] <= max(1.55, 1.001 * max(before["l_over_h_p95"], 1.0e-12))
            and after["l_over_h_count_above_1_55"]
            <= before["l_over_h_count_above_1_55"] + max(0, int(max_l_over_h_count_increase))
        )
    if purpose == "thin":
        defect_improved = bool(
            after["superthin_triangle_count"] < before["superthin_triangle_count"]
            or after["thin_triangle_count"] < before["thin_triangle_count"]
            or after["thin_severity_sum"] < before["thin_severity_sum"] - 1.0e-10
        )
        return bool(
            defect_improved
            and topology_ok
            and after["superthin_triangle_count"] <= before["superthin_triangle_count"]
            and after["q_min"] + 1.0e-12 >= before["q_min"]
            and after["q_p01"] + 1.0e-9 >= before["q_p01"]
            and after["minimum_angle_deg"] + 1.0e-8 >= before["minimum_angle_deg"]
            and after["l_over_h_p95"] <= 1.001 * max(before["l_over_h_p95"], 1.0e-12)
            and after["l_over_h_count_above_1_55"] <= before["l_over_h_count_above_1_55"]
            and after["count_valence_above_limit"] <= before["count_valence_above_limit"]
        )
    size_ok = bool(
        after["l_over_h_maximum"] <= 1.001 * max(before["l_over_h_maximum"], 1.0e-12)
        if purpose == "prune"
        else after["l_over_h_p95"] <= 1.001 * max(before["l_over_h_p95"], 1.0e-12)
    )
    common = bool(
        topology_ok
        and after["q_l3_sigma"] + 1.0e-9 >= before["q_l3_sigma"]
        and after["q_p01"] + 1.0e-9 >= before["q_p01"]
        and after["minimum_angle_p01_deg"] + 1.0e-3 >= before["minimum_angle_p01_deg"]
        and size_ok
        and after["l_over_h_count_above_1_55"] <= before["l_over_h_count_above_1_55"]
    )
    if not common:
        return False
    if purpose == "prune":
        return bool(after["node_count"] < before["node_count"] and after["count_valence_above_limit"] <= before["count_valence_above_limit"])
    return True


def _state_invariants(
    state: _State,
    initial_components: int,
    *,
    geometry: dict[str, np.ndarray] | None = None,
    topology: Any | None = None,
) -> tuple[bool, dict[str, Any]]:
    geometry = geometry if geometry is not None else triangle_geometry(state.points, state.triangles)
    topology = topology if topology is not None else build_edge_topology(len(state.points), state.triangles)
    integrity = constraint_integrity(topology, state.chains, state.open_nodes.tolist())
    positive = bool(len(state.triangles) and np.all(geometry["signed_area"] > _area_tolerance(state.points, state.triangles)))
    boundary_audit = _boundary_graph_audit(topology)
    singly_connected = int(np.count_nonzero(topology.triangle_neighbor_count == 1))
    protected = chain_edges(state.chains)
    protected_not_boundary = int(sum(len(topology.edge_to_triangles.get(edge, [])) != 1 for edge in protected))
    canonical_triangles = [tuple(sorted(map(int, tri))) for tri in np.asarray(state.triangles, dtype=int)]
    duplicate_triangles = int(len(canonical_triangles) - len(set(canonical_triangles)))
    repeated_node_triangles = int(sum(len(set(values)) != 3 for values in canonical_triangles))
    chain_node_range_ok = bool(
        all(0 <= int(node) < len(state.points) for chain in state.chains for node in chain)
    )
    chain_unique_nodes = bool(all(len(chain) == len(set(map(int, chain))) for chain in state.chains))
    lineage_to_node = {int(value): int(index) for index, value in enumerate(state.lineage)}
    missing_hard = [int(value) for value in state.source_hard_anchor_lineage if int(value) not in lineage_to_node]
    moved_hard = [
        int(value)
        for value in state.source_hard_anchor_lineage
        if int(value) in lineage_to_node
        and not np.array_equal(state.points[lineage_to_node[int(value)]], state.source_points[int(value)])
    ]
    report = {
        "positive_signed_areas": positive,
        "all_protected_edges_present": bool(not state.chains or integrity["all_protected_edges_present"]),
        "open_boundary_ordered": bool(not len(state.open_nodes) or integrity["open_boundary_ordered"]),
        "connected_component_count": int(len(topology.connected_component_sizes)),
        "nonmanifold_edge_count": int(len(topology.nonmanifold_edges)),
        "unused_node_count": int(len(state.points) - len(np.unique(state.triangles))) if len(state.triangles) else int(len(state.points)),
        "singly_connected_triangle_count": singly_connected,
        "new_singly_connected_triangle_count": int(
            max(0, singly_connected - int(state.initial_singly_connected_triangle_count))
        ),
        "boundary_degree_anomaly_count": int(boundary_audit["degree_anomaly_count"]),
        "boundary_component_count": int(boundary_audit["component_count"]),
        "boundary_traversable": bool(
            boundary_audit["component_count"] == int(state.initial_boundary_component_count)
            and boundary_audit["degree_anomaly_count"] <= int(state.initial_boundary_degree_anomaly_count)
        ),
        "protected_edge_not_boundary_count": protected_not_boundary,
        "duplicate_triangle_count": duplicate_triangles,
        "repeated_node_triangle_count": repeated_node_triangles,
        "chain_node_range_ok": chain_node_range_ok,
        "chain_unique_nodes": chain_unique_nodes,
        "missing_hard_anchor_count": int(len(missing_hard)),
        "moved_hard_anchor_count": int(len(moved_hard)),
        "missing_hard_anchor_lineage": missing_hard[:100],
        "moved_hard_anchor_lineage": moved_hard[:100],
        "constraint_integrity": integrity,
    }
    ok = bool(
        positive
        and report["all_protected_edges_present"]
        and report["open_boundary_ordered"]
        and report["connected_component_count"] == int(initial_components)
        and report["nonmanifold_edge_count"] == 0
        and report["unused_node_count"] == 0
        and report["new_singly_connected_triangle_count"] == 0
        and report["boundary_traversable"]
        and report["protected_edge_not_boundary_count"] <= int(state.initial_protected_not_boundary_count)
        and report["duplicate_triangle_count"] == 0
        and report["repeated_node_triangle_count"] == 0
        and report["chain_node_range_ok"]
        and report["chain_unique_nodes"]
        and report["missing_hard_anchor_count"] == 0
        and report["moved_hard_anchor_count"] == 0
    )
    return ok, report


def _audit_state(
    state: _State,
    config: AggressiveConditioningConfig,
    initial_components: int,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Share one geometry/topology build across the transaction's global audit."""
    geometry = triangle_geometry(state.points, state.triangles)
    topology = build_edge_topology(len(state.points), state.triangles)
    ok, invariants = _state_invariants(
        state,
        initial_components,
        geometry=geometry,
        topology=topology,
    )
    summary = _summary(state, config, geometry=geometry, topology=topology)
    return ok, invariants, summary


def _failed_invariant_names(report: dict[str, Any]) -> list[str]:
    boolean_gates = {
        "positive_signed_areas": report.get("positive_signed_areas"),
        "all_protected_edges_present": report.get("all_protected_edges_present"),
        "open_boundary_ordered": report.get("open_boundary_ordered"),
        "boundary_traversable": report.get("boundary_traversable"),
        "chain_node_range_ok": report.get("chain_node_range_ok"),
        "chain_unique_nodes": report.get("chain_unique_nodes"),
    }
    failures = [name for name, value in boolean_gates.items() if value is False]
    count_gates = {
        "nonmanifold_edges": report.get("nonmanifold_edge_count", 0),
        "unused_nodes": report.get("unused_node_count", 0),
        "new_singly_connected_triangles": report.get("new_singly_connected_triangle_count", 0),
        "duplicate_triangles": report.get("duplicate_triangle_count", 0),
        "repeated_node_triangles": report.get("repeated_node_triangle_count", 0),
        "missing_hard_anchors": report.get("missing_hard_anchor_count", 0),
        "moved_hard_anchors": report.get("moved_hard_anchor_count", 0),
    }
    failures.extend(name for name, value in count_gates.items() if int(value) > 0)
    return sorted(set(failures or ["structural_invariant"]))


def _compact(state: _State) -> np.ndarray:
    node_count = len(state.points)
    numeric_lengths = {
        "fixed": len(state.fixed),
        "targets": len(state.targets),
        "hard": len(state.hard),
        "lineage": len(state.lineage),
    }
    invalid = {name: length for name, length in numeric_lengths.items() if length != node_count}
    if invalid:
        raise ValueError(
            "Local-topology node metadata length mismatch before compaction: "
            f"points={node_count}, fields={invalid}"
        )
    if len(state.kinds) < node_count:
        raise ValueError(
            "Local-topology boundary-kind metadata is missing active node values before compaction: "
            f"points={node_count}, kinds={len(state.kinds)}"
        )
    # Descriptive kind entries beyond the active point array are stale tail
    # metadata.  They cannot be referenced by triangles or chains and are
    # safe to discard before applying the compaction mapping.
    active_kinds = state.kinds[:node_count]
    referenced = set(map(int, np.unique(state.triangles)))
    referenced.update(int(value) for chain in state.chains for value in chain)
    keep = np.asarray([index in referenced for index in range(node_count)], dtype=bool)
    mapping = np.full(node_count, -1, dtype=int)
    mapping[np.where(keep)[0]] = np.arange(int(np.count_nonzero(keep)), dtype=int)
    state.points = state.points[keep]
    state.fixed = state.fixed[keep]
    state.targets = state.targets[keep]
    state.hard = state.hard[keep]
    state.lineage = state.lineage[keep]
    state.kinds = [kind for kind, retain in zip(active_kinds, keep, strict=True) if retain]
    state.triangles = mapping[state.triangles]
    state.chains = [[int(mapping[node]) for node in chain if mapping[node] >= 0] for chain in state.chains]
    state.open_nodes = np.asarray([int(mapping[node]) for node in state.open_nodes if mapping[node] >= 0], dtype=int)
    state.last_affected = [int(mapping[node]) for node in state.last_affected if 0 <= node < len(mapping) and mapping[node] >= 0]
    return mapping


def _restore(state: _State, snapshot: _State) -> None:
    state.points = snapshot.points
    state.triangles = snapshot.triangles
    state.fixed = snapshot.fixed
    state.targets = snapshot.targets
    state.chains = snapshot.chains
    state.open_nodes = snapshot.open_nodes
    state.kinds = snapshot.kinds
    state.hard = snapshot.hard
    state.lineage = snapshot.lineage
    state.source_points = snapshot.source_points
    state.source_chains = snapshot.source_chains
    state.source_hard_anchor_lineage = snapshot.source_hard_anchor_lineage
    state.target_sampler = snapshot.target_sampler
    state.initial_domain_area_m2 = snapshot.initial_domain_area_m2
    state.initial_boundary_component_count = snapshot.initial_boundary_component_count
    state.initial_boundary_degree_anomaly_count = snapshot.initial_boundary_degree_anomaly_count
    state.initial_singly_connected_triangle_count = snapshot.initial_singly_connected_triangle_count
    state.initial_protected_not_boundary_count = snapshot.initial_protected_not_boundary_count
    state.ledger = snapshot.ledger
    state.cumulative_boundary_area_change_m2 = snapshot.cumulative_boundary_area_change_m2
    state.last_affected = snapshot.last_affected


def _ordered_one_ring(incident_triangles: np.ndarray, node: int) -> list[int] | None:
    adjacency: dict[int, set[int]] = {}
    for tri in incident_triangles:
        others = [int(value) for value in tri if int(value) != int(node)]
        if len(others) != 2:
            return None
        a, b = others
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    if not adjacency or any(len(values) != 2 for values in adjacency.values()):
        return None
    start = min(adjacency)
    ring = [start]
    previous = -1
    current = start
    for _ in range(len(adjacency) + 1):
        candidates = sorted(adjacency[current])
        next_node = candidates[0] if candidates[0] != previous else candidates[1]
        if next_node == start:
            return ring if len(ring) == len(adjacency) else None
        ring.append(next_node)
        previous, current = current, next_node
    return None


def _ordered_boundary_fan(incident_triangles: np.ndarray, node: int, previous: int, following: int) -> list[int] | None:
    adjacency: dict[int, set[int]] = {}
    for tri in incident_triangles:
        others = [int(value) for value in tri if int(value) != int(node)]
        if len(others) != 2:
            return None
        a, b = others
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    if previous not in adjacency or following not in adjacency:
        return None
    path = [int(previous)]
    prior = -1
    current = int(previous)
    for _ in range(len(adjacency) + 1):
        if current == int(following):
            return path
        options = [value for value in sorted(adjacency[current]) if value != prior]
        if not options:
            return None
        next_node = options[0]
        path.append(int(next_node))
        prior, current = current, int(next_node)
    return None


def _boundary_removal_allowed(state: _State, node: int, config: AggressiveConditioningConfig) -> bool:
    if node < 0 or node >= len(state.points) or not state.fixed[node] or state.hard[node]:
        return False
    membership = _find_chain_node(state.chains, node)
    if membership is None:
        return False
    chain_index, position = membership
    chain = state.chains[chain_index]
    if len(chain) <= 3:
        return False
    previous = int(chain[position - 1])
    following = int(chain[(position + 1) % len(chain)])
    kind = state.kinds[node]
    if state.kinds[previous] != kind or state.kinds[following] != kind:
        return False
    # Preserve the ends of the non-cyclic OBC nodestring.
    open_values = state.open_nodes.tolist()
    if node in open_values and (open_values.index(node) in {0, len(open_values) - 1}):
        return False
    deviation = _point_segment_distance(state.points[node], state.points[previous], state.points[following])
    h = max(float(state.targets[node]), 1.0e-12)
    if kind in {"open", "frame"}:
        limit = min(float(config.open_boundary_max_deviation_m), float(config.open_boundary_deviation_fraction) * h)
    else:
        limit = min(float(config.land_boundary_max_deviation_m), float(config.land_boundary_deviation_fraction) * h)
    return bool(deviation <= limit)


def _boundary_deviation(state: _State, node: int) -> float:
    membership = _find_chain_node(state.chains, int(node))
    if membership is None:
        return float("inf")
    chain_index, position = membership
    chain = state.chains[chain_index]
    return _point_segment_distance(state.points[node], state.points[chain[position - 1]], state.points[chain[(position + 1) % len(chain)]])


def _find_chain_edge(chains: list[list[int]], edge: tuple[int, int]) -> tuple[int, int] | None:
    target = set(map(int, edge))
    for chain_index, chain in enumerate(chains):
        for position, a in enumerate(chain):
            b = chain[(position + 1) % len(chain)]
            if {int(a), int(b)} == target:
                return int(chain_index), int(position)
    return None


def _find_chain_node(chains: list[list[int]], node: int) -> tuple[int, int] | None:
    for chain_index, chain in enumerate(chains):
        if int(node) in chain:
            return int(chain_index), int(chain.index(int(node)))
    return None


def _same_boundary_kind(state: _State, edge: tuple[int, int]) -> bool:
    return bool(state.fixed[edge[0]] and state.fixed[edge[1]] and state.kinds[edge[0]] == state.kinds[edge[1]])


def _source_arc_edge(state: _State, edge: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    a, b = map(int, edge)
    source_a = int(state.lineage[a]) if 0 <= a < len(state.lineage) else -1
    source_b = int(state.lineage[b]) if 0 <= b < len(state.lineage) else -1
    if source_a < 0 or source_b < 0 or source_a >= len(state.source_points) or source_b >= len(state.source_points):
        return None
    target = {source_a, source_b}
    if not any(
        {int(left), int(right)} == target
        for chain in state.source_chains
        for left, right in zip(chain, [*chain[1:], chain[0]])
    ):
        return None
    return state.source_points[source_a].copy(), state.source_points[source_b].copy()


def _cyclic_chain_window(chain: list[int], edge_position: int, radius: int) -> set[int]:
    if not chain:
        return set()
    radius = max(0, int(radius))
    positions = range(int(edge_position) - radius, int(edge_position) + 2 + radius)
    return {int(chain[position % len(chain)]) for position in positions}


def _minimum_remote_boundary_clearance(
    state: _State,
    point: np.ndarray,
    active_edge: tuple[int, int],
) -> float:
    active_nodes = set(map(int, active_edge))
    distances: list[float] = []
    for chain in state.chains:
        for left, right in zip(chain, [*chain[1:], chain[0]]):
            edge = {int(left), int(right)}
            if edge & active_nodes:
                continue
            distances.append(
                _point_segment_distance(point, state.points[int(left)], state.points[int(right)])
            )
    return float(min(distances)) if distances else float("inf")


def _boundary_graph_audit(topology: Any) -> dict[str, int]:
    adjacency: dict[int, set[int]] = {}
    for a, b in topology.boundary_edges:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    anomaly_count = int(sum(len(values) != 2 for values in adjacency.values()))
    unseen = set(adjacency)
    component_count = 0
    while unseen:
        component_count += 1
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        while stack:
            current = stack.pop()
            following = adjacency[current] & unseen
            unseen.difference_update(following)
            stack.extend(following)
    return {
        "node_count": int(len(adjacency)),
        "degree_anomaly_count": anomaly_count,
        "component_count": int(component_count),
    }


def _boundary_area_budget_allows(
    state: _State,
    actual_change_m2: float,
    config: AggressiveConditioningConfig,
) -> bool:
    budget = float(config.maximum_domain_area_change_fraction) * max(float(state.initial_domain_area_m2), 1.0e-30)
    return bool(
        np.isfinite(actual_change_m2)
        and actual_change_m2 >= 0.0
        and state.cumulative_boundary_area_change_m2 + float(actual_change_m2) <= budget + 1.0e-12
    )


def _sample_target_at(state: _State, point: np.ndarray, *, fallback: float) -> float:
    return float(_sample_targets(state, np.asarray(point, dtype=float).reshape(1, 2), fallback=np.asarray([fallback]))[0])


def _sample_targets(state: _State, points: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    fallback_values = np.asarray(fallback, dtype=float).reshape(-1)
    if state.target_sampler is None:
        return fallback_values.copy()
    try:
        sampled = np.asarray(state.target_sampler(np.asarray(points, dtype=float)), dtype=float).reshape(-1)
    except Exception:
        return fallback_values.copy()
    if len(sampled) == 1 and len(fallback_values) > 1:
        sampled = np.full(len(fallback_values), float(sampled[0]), dtype=float)
    if len(sampled) != len(fallback_values):
        return fallback_values.copy()
    return np.where(np.isfinite(sampled) & (sampled > 0.0), sampled, fallback_values)


def _deduplicate_rejections(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for value in values:
        key = (
            str(value.get("operation", "")),
            tuple(map(int, value.get("payload", []))),
            tuple(map(str, value.get("failures", []))),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _link_condition(topology: Any, edge: tuple[int, int], attached: list[int]) -> bool:
    a, b = map(int, edge)
    common = set(topology.node_neighbors[a]) & set(topology.node_neighbors[b])
    opposite: set[int] = set()
    for tri_index in attached:
        # edge_to_triangles is paired with the caller's triangle array; common
        # neighbors are exactly the two opposite vertices for a legal interior collapse.
        for value in topology.node_neighbors[a] & topology.node_neighbors[b]:
            opposite.add(int(value))
    return bool(len(attached) == 2 and len(common) == 2 and common == opposite)


def _triangle_targets(targets: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    values = np.asarray(targets, dtype=float)[np.asarray(triangles, dtype=int)]
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = 3.0 / np.sum(1.0 / np.maximum(values, 1.0e-12), axis=1)
    return np.where(np.isfinite(harmonic) & (harmonic > 0.0), harmonic, np.nanmedian(values, axis=1))


def _edge_target(targets: np.ndarray, edge: tuple[int, int]) -> float:
    values = np.asarray([targets[int(edge[0])], targets[int(edge[1])]], dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if not len(values):
        return 1.0
    if len(values) == 1:
        return float(values[0])
    return float(2.0 / np.sum(1.0 / values))


def _interpolate_local_targets(state: _State, points: np.ndarray, source_nodes: list[int]) -> np.ndarray:
    sources = np.asarray(sorted(set(map(int, source_nodes))), dtype=int)
    if not len(sources):
        return np.full(len(points), float(np.nanmedian(state.targets)), dtype=float)
    source_points = state.points[sources]
    source_targets = state.targets[sources]
    output = np.empty(len(points), dtype=float)
    for index, point in enumerate(np.asarray(points, dtype=float)):
        distance = np.linalg.norm(source_points - point, axis=1)
        nearest = np.argsort(distance)[: min(8, len(distance))]
        if distance[nearest[0]] <= 1.0e-10:
            output[index] = float(source_targets[nearest[0]])
            continue
        weights = 1.0 / np.maximum(distance[nearest], 1.0e-9) ** 2
        output[index] = float(np.sum(weights * source_targets[nearest]) / np.sum(weights))
    fallback = float(np.nanmedian(source_targets[np.isfinite(source_targets) & (source_targets > 0.0)]))
    return np.where(np.isfinite(output) & (output > 0.0), output, fallback)


def _normalize_targets(values: np.ndarray, node_count: int) -> np.ndarray:
    targets = np.asarray(values, dtype=float).copy()
    if len(targets) != int(node_count):
        raise ValueError("target_spacing_m must have one value per node")
    fallback = float(np.nanmedian(targets[np.isfinite(targets) & (targets > 0.0)])) if np.any(np.isfinite(targets) & (targets > 0.0)) else 1.0
    return np.where(np.isfinite(targets) & (targets > 0.0), targets, fallback)


def _new_lineage_ids(state: _State, count: int) -> np.ndarray:
    """Allocate stable negative identities for newly inserted local nodes."""
    count = max(0, int(count))
    if count == 0:
        return np.empty(0, dtype=int)
    lowest = min(int(np.min(state.lineage)) if len(state.lineage) else 0, 0)
    return np.arange(lowest - 1, lowest - count - 1, -1, dtype=int)


def _edge_set(triangles: np.ndarray) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(a), int(b))))
        for tri in np.asarray(triangles, dtype=int)
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
    }


def _triangle_edge_keys(triangle: np.ndarray) -> set[tuple[int, int]]:
    a, b, c = map(int, np.asarray(triangle, dtype=int))
    return {tuple(sorted(edge)) for edge in ((a, b), (b, c), (c, a))}


def _incident_triangle_lists(node_count: int, triangles: np.ndarray) -> list[list[int]]:
    incident = [[] for _ in range(int(node_count))]
    for triangle_index, tri in enumerate(np.asarray(triangles, dtype=int)):
        for node in tri:
            incident[int(node)].append(int(triangle_index))
    return incident


def _orient_ccw(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=int).copy()
    if not len(triangles):
        return triangles.reshape((-1, 3))
    signed = triangle_geometry(points, triangles)["signed_area"]
    flip = signed < 0.0
    if np.any(flip):
        triangles[flip, 1], triangles[flip, 2] = triangles[flip, 2].copy(), triangles[flip, 1].copy()
    return triangles


def _area_tolerance(points: np.ndarray, triangles: np.ndarray) -> float:
    if not len(triangles):
        return 0.0
    lengths = triangle_geometry(points, triangles)["edge_lengths"]
    scale2 = float(np.max(lengths, initial=0.0)) ** 2
    return max(1.0e-12, 1.0e-14 * scale2)


def _mesh_area(points: np.ndarray, triangles: np.ndarray) -> float:
    return float(np.sum(triangle_geometry(points, triangles)["area"])) if len(triangles) else 0.0


def _signed_mesh_area(points: np.ndarray, triangles: np.ndarray) -> float:
    return float(np.sum(triangle_geometry(points, triangles)["signed_area"])) if len(triangles) else 0.0


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    edge = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    denominator = float(np.dot(edge, edge))
    if denominator <= 1.0e-24:
        return float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(a, dtype=float)))
    fraction = float(np.dot(np.asarray(point, dtype=float) - np.asarray(a, dtype=float), edge) / denominator)
    closest = np.asarray(a, dtype=float) + np.clip(fraction, 0.0, 1.0) * edge
    return float(np.linalg.norm(np.asarray(point, dtype=float) - closest))


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    a, b, c = np.asarray(triangle, dtype=float)
    v0 = c - a
    v1 = b - a
    v2 = np.asarray(point, dtype=float) - a
    denominator = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(float(denominator)) <= 1.0e-20:
        return False
    u = (v2[0] * v1[1] - v1[0] * v2[1]) / denominator
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / denominator
    return bool(u > 1.0e-12 and v > 1.0e-12 and u + v < 1.0 - 1.0e-12)


def _ledger_counts(ledger: list[dict[str, Any]]) -> dict[str, int]:
    output: dict[str, int] = {}
    for entry in ledger:
        name = str(entry.get("operation", "unknown"))
        output[name] = output.get(name, 0) + 1
    return output
