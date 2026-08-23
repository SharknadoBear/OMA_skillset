"""Aggressive, transactional local topology conditioning for FVCOM meshes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
from itertools import combinations
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial import Delaunay, QhullError
from shapely import contains_xy
from shapely.geometry import LineString, Polygon

from .connectivity_restriction import (
    AllowedEdgePolicy,
    ConnectivityRestrictionConfig,
    restricted_edge_violation_records,
)
from .metrics import build_edge_topology, chain_edges, constraint_integrity, triangle_geometry
from .regional_conditioning import SpringRelaxConfig, _edge_flip_candidate, relax_mesh_spring


@dataclass(frozen=True)
class AggressiveConditioningConfig:
    enabled: bool = True
    profile_name: str = "aggressive-local-v2"
    stage_order: str = "thin-before-valence"
    enable_pruning: bool = True
    enable_thin_repair: bool = True
    enable_valence_repair: bool = True
    thin_repair_profile: str = "guarded-v1"
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
    systematic_max_components_per_round: int = 24
    systematic_min_patch_rings: int = 2
    systematic_max_patch_rings: int = 4
    systematic_max_support_points: int = 8
    systematic_min_passage_elements: int = 2
    systematic_v3_obc_policy: str = "redistribute"
    systematic_v3_boundary_window_radius: int = 4
    systematic_v3_max_candidates_per_component: int = 12
    systematic_v3_weld_snap_fraction: float = 0.15
    systematic_v3_passage_clearance_tolerance_m: float = 0.50
    systematic_gate_scope: str = "candidate"
    systematic_collapse_welds_per_round: int = 0
    systematic_v5_max_star_transactions_per_round: int = 256
    systematic_v5_max_inward_front_support_points: int = 8
    systematic_v5_max_lawson_flips_per_transaction: int = 128
    systematic_v5_patch_ring_ladder: tuple[int, ...] = (1, 2, 4)
    systematic_v5_enable_boundary_window_fallback: bool = True
    systematic_v5_enable_connectivity_restriction: bool = True
    systematic_v5_connectivity_only: bool = False
    systematic_v5_max_connectivity_transactions_per_round: int = 32
    systematic_v5_max_connectivity_candidates_per_component: int = 8
    systematic_v5_connectivity_shortcut_arc_chord_ratio: float = 3.0
    systematic_v5_connectivity_shortcut_arc_target_ratio: float = 3.0
    systematic_v5_audit_reserve_seconds: float = 120.0
    deadline_monotonic_s: float | None = None
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
    topology_escrow_enabled: bool = False
    topology_escrow_maximum_superthin_count: int = 25
    topology_escrow_maximum_superthin_severity: float = 25.0
    topology_escrow_maximum_valence: int = 12
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
    restricted_lineage_edges: set[tuple[int, int]]
    report: dict[str, Any]
    edit_ledger: list[dict[str, Any]]
    obc_remap_manifest: dict[str, Any]


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
    source_open_nodes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    source_kinds: list[str] = field(default_factory=list)
    source_hard_anchor_lineage: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    target_sampler: Callable[[np.ndarray], np.ndarray] | None = None
    initial_domain_area_m2: float = 0.0
    initial_boundary_component_count: int = 0
    initial_boundary_degree_anomaly_count: int = 0
    initial_singly_connected_triangle_count: int = 0
    initial_protected_not_boundary_count: int = 0
    restricted_lineage_edges: set[tuple[int, int]] = field(default_factory=set)
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
            source_open_nodes=self.source_open_nodes.copy(),
            source_kinds=self.source_kinds.copy(),
            source_hard_anchor_lineage=self.source_hard_anchor_lineage.copy(),
            target_sampler=self.target_sampler,
            initial_domain_area_m2=float(self.initial_domain_area_m2),
            initial_boundary_component_count=int(self.initial_boundary_component_count),
            initial_boundary_degree_anomaly_count=int(self.initial_boundary_degree_anomaly_count),
            initial_singly_connected_triangle_count=int(self.initial_singly_connected_triangle_count),
            initial_protected_not_boundary_count=int(self.initial_protected_not_boundary_count),
            restricted_lineage_edges=set(self.restricted_lineage_edges),
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
    restricted_lineage_edges: set[tuple[int, int]] | None = None,
    config: AggressiveConditioningConfig | None = None,
) -> LocalTopologyResult:
    """Apply target-aware pruning, aggressive thin repair, and hard valence repair."""
    config = config or AggressiveConditioningConfig()
    if str(config.stage_order) not in {
        "thin-before-valence",
        "valence-before-thin",
    }:
        raise ValueError(
            "stage_order must be thin-before-valence or valence-before-thin"
        )
    if str(config.thin_repair_profile) not in {
        "guarded-v1",
        "systematic-v2",
        "systematic-v3",
        "systematic-v5",
    }:
        raise ValueError(
            "thin_repair_profile must be guarded-v1, systematic-v2, systematic-v3, or systematic-v5"
        )
    if str(config.systematic_v3_obc_policy) not in {"preserve", "redistribute"}:
        raise ValueError("systematic_v3_obc_policy must be preserve or redistribute")
    if str(config.systematic_gate_scope) not in {"candidate", "loop-end"}:
        raise ValueError("systematic_gate_scope must be candidate or loop-end")
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
        source_open_nodes=np.asarray(open_boundary_nodes_zero_based, dtype=int).copy(),
        source_kinds=kinds.copy(),
        source_hard_anchor_lineage=np.where(hard)[0].astype(int),
        target_sampler=target_spacing_sampler,
        initial_domain_area_m2=max(_signed_mesh_area(points, tris), 1.0e-30),
        initial_boundary_component_count=int(initial_boundary["component_count"]),
        initial_boundary_degree_anomaly_count=int(initial_boundary["degree_anomaly_count"]),
        initial_singly_connected_triangle_count=int(np.count_nonzero(initial_topology.triangle_neighbor_count == 1)),
        initial_protected_not_boundary_count=int(
            sum(len(initial_topology.edge_to_triangles.get(edge, [])) != 1 for edge in initial_protected)
        ),
        restricted_lineage_edges={
            tuple(sorted(map(int, edge)))
            for edge in (restricted_lineage_edges or set())
        },
    )
    initial = _summary(state, config)
    initial_components = int(initial["connected_component_count"])
    rounds: list[dict[str, Any]] = []
    if config.enabled:
        for round_index in range(max(0, int(config.max_rounds))):
            if _deadline_reached(config):
                break
            before_round = _summary(state, config)
            prune = (
                _prune_redundant_vertices(state, config, initial_components)
                if config.enable_pruning and int(config.max_prunes_per_round) > 0
                else _disabled_stage(state, config, "pruning_disabled")
            )
            if str(config.stage_order) == "valence-before-thin":
                valence, post_valence_thin, compound = (
                    _repair_valence_thin_atomic(
                        state,
                        config,
                        initial_components,
                    )
                )
                thin = (
                    _repair_superthin(state, config, initial_components)
                    if config.enable_thin_repair and _thin_budget(config) > 0
                    else _disabled_stage(state, config, "thin_repair_disabled")
                )
                (
                    terminal_valence,
                    terminal_post_valence_thin,
                    terminal_compound,
                ) = _repair_valence_thin_atomic(
                    state,
                    config,
                    initial_components,
                )
            else:
                thin = (
                    _repair_superthin(state, config, initial_components)
                    if config.enable_thin_repair and _thin_budget(config) > 0
                    else _disabled_stage(state, config, "thin_repair_disabled")
                )
                valence, post_valence_thin, compound = (
                    _repair_valence_thin_atomic(
                        state,
                        config,
                        initial_components,
                    )
                )
                terminal_valence = _disabled_stage(
                    state,
                    config,
                    "terminal_valence_scan_not_selected",
                )
                terminal_post_valence_thin = _disabled_stage(
                    state,
                    config,
                    "terminal_valence_scan_not_selected",
                )
                terminal_compound = {
                    "attempted": False,
                    "accepted": True,
                    "rolled_back": False,
                    "rejected_gates": [],
                    "before": _summary(state, config),
                    "after": _summary(state, config),
                }
            after_round = _summary(state, config)
            operations = int(
                prune["accepted"]
                + thin["accepted"]
                + valence["accepted"]
                + post_valence_thin["accepted"]
                + terminal_valence["accepted"]
                + terminal_post_valence_thin["accepted"]
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
                    "terminal_high_valence_repair": terminal_valence,
                    "terminal_post_valence_thin_repair": (
                        terminal_post_valence_thin
                    ),
                    "terminal_valence_thin_atomic_transaction": (
                        terminal_compound
                    ),
                    "after": after_round,
                    "accepted_operation_count": operations,
                }
            )
            if operations == 0:
                break
            if after_round["count_valence_above_limit"] == 0 and after_round["superthin_triangle_count"] == 0:
                break
    final = _summary(state, config)
    final_invariants_ok, final_invariants, _ = _audit_state(
        state,
        config,
        initial_components,
    )
    hard_gate = bool(final["count_valence_above_limit"] == 0)
    superthin_gate = bool(final["superthin_triangle_count"] == 0)
    minimal_local_debt_closed = bool(
        final_invariants_ok
        and hard_gate
        and superthin_gate
        and int(final["restricted_edge_violation_count"]) == 0
    )
    report = {
        "schema_version": "fvcom_aggressive_local_conditioning_v2",
        "profile": str(config.profile_name),
        "settings": asdict(config),
        "accepted": bool(final_invariants_ok),
        "fvcom_valence_gate_passed": hard_gate,
        "superthin_gate_passed": superthin_gate,
        "minimal_local_debt_closed": minimal_local_debt_closed,
        "terminal_topology_gate_passed": minimal_local_debt_closed,
        "deadline_reached": bool(_deadline_reached(config)),
        "before": initial,
        "after": final,
        "rounds": rounds,
        "edit_count": int(len(state.ledger)),
        "edit_counts": _ledger_counts(state.ledger),
        "cumulative_boundary_area_change_m2": float(state.cumulative_boundary_area_change_m2),
        "cumulative_boundary_area_change_fraction": float(
            state.cumulative_boundary_area_change_m2 / max(_mesh_area(points, tris), 1.0e-30)
        ),
        "invariants": final_invariants,
        "obc_remap_manifest": _obc_remap_manifest(state),
        "restricted_lineage_edges": [
            list(map(int, edge))
            for edge in sorted(state.restricted_lineage_edges)
        ],
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
        obc_remap_manifest=_obc_remap_manifest(state),
    )


def _thin_budget(config: AggressiveConditioningConfig) -> int:
    return int(
        max(0, int(config.max_boundary_ear_removals_per_round))
        + max(0, int(config.max_boundary_welds_per_round))
        + max(0, int(config.max_superthin_flips_per_round))
        + max(0, int(config.max_collapses_per_round))
        + max(0, int(config.max_boundary_edits_per_round))
        + max(0, int(config.systematic_collapse_welds_per_round))
        + (
            max(0, int(config.systematic_v5_max_star_transactions_per_round))
            if str(config.thin_repair_profile).lower() == "systematic-v5"
            else 0
        )
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

    escrow_state: _State | None = None
    escrow_ok = False
    escrow_invariants: dict[str, Any] = {}
    escrow_summary: dict[str, Any] = {}
    escrow_rejected_gates: list[str] = []
    if bool(config.topology_escrow_enabled):
        escrow_state = state.clone()
        (
            escrow_ok,
            escrow_invariants,
            escrow_summary,
        ) = _audit_state(
            escrow_state,
            config,
            initial_components,
        )
        escrow_rejected_gates = _topology_escrow_failures(
            escrow_state,
            before,
            escrow_summary,
            escrow_ok,
            escrow_invariants,
            config,
        )

    post = (
        _repair_superthin(state, config, initial_components)
        if config.enable_thin_repair and _thin_budget(config) > 0
        else _disabled_stage(state, config, "thin_repair_disabled")
    )
    ok, invariants, after = _audit_state(state, config, initial_components)
    rejected_gates = _compound_valence_thin_failures(before, after, ok, invariants, config)
    accepted_edits = int(valence.get("accepted", 0)) + int(post.get("accepted", 0))
    post_counterproductive = bool(
        config.topology_escrow_enabled
        and _post_thin_counterproductive(
            escrow_summary,
            after,
            post,
        )
    )
    escrow_trigger_reasons = [
        *(
            ["final_compound_transaction_failed"]
            if rejected_gates
            else []
        ),
        *(
            ["post_thin_counterproductive"]
            if post_counterproductive
            else []
        ),
    ]
    if (
        escrow_state is not None
        and escrow_trigger_reasons
        and not escrow_rejected_gates
    ):
        attempted_thin = int(post.get("accepted", 0))
        _restore(state, escrow_state)
        valence = dict(valence)
        valence.update(
            {
                "transaction_rolled_back": False,
                "topology_escrow_retained": True,
                "provisional_escrow_acceptance": True,
                "after": _summary(state, config),
            }
        )
        post = dict(post)
        post.update(
            {
                "accepted": 0,
                "rolled_back_operation_count": attempted_thin,
                "transaction_rolled_back": True,
                "topology_escrow_post_thin_rolled_back": True,
                "provisional_escrow_acceptance": False,
                "after": _summary(state, config),
            }
        )
        return valence, post, {
            "attempted": True,
            "accepted": True,
            "accepted_via": "valence_only_midpoint_escrow",
            "provisional_escrow_accepted": True,
            "rolled_back": False,
            "post_thin_rolled_back": True,
            "accepted_operation_count": int(
                valence.get("accepted", 0)
            ),
            "rejected_gates": [],
            "final_candidate_rejected_gates": rejected_gates,
            "escrow_trigger_reasons": escrow_trigger_reasons,
            "escrow_rejected_gates": [],
            "escrow_invariants": escrow_invariants,
            "before": before,
            "valence_only_midpoint": escrow_summary,
            "final_trial": after,
            "after": _summary(state, config),
        }
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
            **(
                {
                    "provisional_escrow_accepted": False,
                    "escrow_trigger_reasons": (
                        escrow_trigger_reasons
                    ),
                    "escrow_rejected_gates": (
                        escrow_rejected_gates
                    ),
                    "escrow_invariants": escrow_invariants,
                    "valence_only_midpoint": escrow_summary,
                }
                if bool(config.topology_escrow_enabled)
                else {}
            ),
        }
    if (
        bool(config.topology_escrow_enabled)
        and post_counterproductive
    ):
        # A counterproductive post-thin result can reach this branch only when
        # the provisional midpoint itself failed its escrow contract.
        # Preserve the ordinary compound result and retain the failed escrow
        # evidence rather than silently changing strict behavior.
        return valence, post, {
            "attempted": True,
            "accepted": True,
            "rolled_back": False,
            "accepted_operation_count": accepted_edits,
            "rejected_gates": [],
            "provisional_escrow_accepted": False,
            "escrow_trigger_reasons": escrow_trigger_reasons,
            "escrow_rejected_gates": escrow_rejected_gates,
            "escrow_invariants": escrow_invariants,
            "before": before,
            "valence_only_midpoint": escrow_summary,
            "after": after,
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


def _topology_escrow_failures(
    state: _State,
    before: dict[str, Any],
    midpoint: dict[str, Any],
    invariants_ok: bool,
    invariants: dict[str, Any],
    config: AggressiveConditioningConfig,
) -> list[str]:
    """Audit an opt-in valence-only midpoint without requiring zero thin debt."""
    failures: list[str] = []
    if not invariants_ok:
        failures.extend(_failed_invariant_names(invariants))
    before_valence = (
        int(before["count_valence_above_limit"]),
        int(before["valence_excess_sum"]),
        int(before["maximum_valence"]),
    )
    midpoint_valence = (
        int(midpoint["count_valence_above_limit"]),
        int(midpoint["valence_excess_sum"]),
        int(midpoint["maximum_valence"]),
    )
    if not midpoint_valence < before_valence:
        failures.append("escrow_valence_tuple_not_strictly_improved")
    maximum_valence = max(
        int(before["maximum_valence"]),
        max(0, int(config.topology_escrow_maximum_valence)),
    )
    if int(midpoint["maximum_valence"]) > maximum_valence:
        failures.append("escrow_maximum_valence_exceeded")
    if int(midpoint["superthin_triangle_count"]) > max(
        0,
        int(config.topology_escrow_maximum_superthin_count),
    ):
        failures.append("escrow_superthin_count_exceeded")
    if float(midpoint["superthin_severity_sum"]) > max(
        0.0,
        float(config.topology_escrow_maximum_superthin_severity),
    ):
        failures.append("escrow_superthin_severity_exceeded")
    if int(midpoint["boundary_component_count"]) != int(
        before["boundary_component_count"]
    ):
        failures.append("escrow_boundary_component_change")
    if int(midpoint["boundary_degree_anomaly_count"]) > int(
        before["boundary_degree_anomaly_count"]
    ):
        failures.append("escrow_boundary_degree_regression")
    if int(midpoint["connected_component_count"]) != int(
        before["connected_component_count"]
    ):
        failures.append("escrow_wet_component_change")
    if int(midpoint["protected_edge_not_boundary_count"]) > int(
        before["protected_edge_not_boundary_count"]
    ):
        failures.append("escrow_protected_edge_regression")
    if int(midpoint.get("restricted_edge_violation_count", 0)) != 0:
        failures.append("escrow_restricted_edge_violation")
    if _maximum_boundary_source_arc_deviation(state) > 1.0e-6:
        failures.append("escrow_boundary_vertex_off_source_arc")
    if not _boundary_loops_simple(state):
        failures.append("escrow_boundary_self_intersection")
    if len(state.source_open_nodes):
        source_open = tuple(
            map(
                int,
                np.asarray(state.source_open_nodes, dtype=int),
            )
        )
        if _open_boundary_lineage_sequence(state) != source_open:
            failures.append("escrow_open_boundary_lineage_changed")
    return sorted(set(failures))


def _post_thin_counterproductive(
    midpoint: dict[str, Any],
    after: dict[str, Any],
    post: dict[str, Any],
) -> bool:
    """Return true when accepted thin work fails to improve its causal debt."""
    if not midpoint or int(post.get("accepted", 0)) <= 0:
        return False
    thin_before = (
        int(midpoint["superthin_triangle_count"]),
        float(midpoint["superthin_severity_sum"]),
    )
    thin_after = (
        int(after["superthin_triangle_count"]),
        float(after["superthin_severity_sum"]),
    )
    valence_before = (
        int(midpoint["count_valence_above_limit"]),
        int(midpoint["valence_excess_sum"]),
        int(midpoint["maximum_valence"]),
    )
    valence_after = (
        int(after["count_valence_above_limit"]),
        int(after["valence_excess_sum"]),
        int(after["maximum_valence"]),
    )
    return bool(
        not thin_after < thin_before
        or valence_after > valence_before
        or int(after["boundary_degree_anomaly_count"])
        > int(midpoint["boundary_degree_anomaly_count"])
        or int(after["boundary_component_count"])
        != int(midpoint["boundary_component_count"])
    )


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
        source_open_nodes=np.empty(0, dtype=int),
        source_kinds=kinds.copy(),
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
    """Run the selected extreme-tail repair profile.

    ``systematic-v2`` deliberately keeps every existing model-boundary node
    fixed. ``systematic-v3`` retains that ladder and then adds transactional
    source-arc welding, sliding, and boundary-node redistribution.
    ``systematic-v5`` removes and reconstructs the complete causal node star
    before accepting a collapse; distributional engineering gates are left to
    the enclosing zero-debt relaxation/closure cycle.
    """
    profile = str(config.thin_repair_profile).lower()
    if profile not in {"guarded-v1", "systematic-v2", "systematic-v3", "systematic-v5"}:
        raise ValueError(
            "thin_repair_profile must be guarded-v1, systematic-v2, systematic-v3, or systematic-v5"
        )
    if profile == "guarded-v1":
        return _repair_superthin_guarded(state, config, initial_components)
    if profile == "systematic-v5":
        return _repair_superthin_systematic_v5(state, config, initial_components)
    stage_before = _summary(state, config)
    interior_only = replace(
        config,
        max_boundary_ear_removals_per_round=0,
        max_boundary_welds_per_round=0,
        max_boundary_edits_per_round=0,
    )
    cheap = _repair_superthin_guarded(state, interior_only, initial_components)
    systematic = _repair_superthin_components(state, config, initial_components)
    boundary_adaptive = (
        _repair_superthin_boundary_adaptive_v3(state, config, initial_components)
        if profile == "systematic-v3"
        else None
    )
    return {
        "profile": profile,
        "accepted": (
            int(cheap.get("accepted", 0))
            + int(systematic.get("accepted", 0))
            + int((boundary_adaptive or {}).get("accepted", 0))
        ),
        "rejected": (
            int(cheap.get("rejected", 0))
            + int(systematic.get("rejected", 0))
            + int((boundary_adaptive or {}).get("rejected", 0))
        ),
        "cheap_interior_ladder": cheap,
        "systematic_components": systematic,
        "boundary_adaptive_v3": boundary_adaptive,
        "component_classifications": [
            *systematic.get("component_classifications", []),
            *((boundary_adaptive or {}).get("component_classifications", [])),
        ],
        "candidate_attempts": [
            *systematic.get("candidate_attempts", []),
            *((boundary_adaptive or {}).get("candidate_attempts", [])),
        ],
        "blocked_components": (
            (boundary_adaptive or {}).get("blocked_components", systematic.get("blocked_components", []))
        ),
        "runtime_seconds": (
            float(systematic.get("runtime_seconds", 0.0))
            + float((boundary_adaptive or {}).get("runtime_seconds", 0.0))
        ),
        "before": stage_before,
        "after": _summary(state, config),
    }


def _thin_transaction_passes(
    before: dict[str, Any],
    after: dict[str, Any],
    invariants_ok: bool,
    config: AggressiveConditioningConfig,
) -> bool:
    """Apply only structural and debt gates inside a v4 loop transaction."""
    if not invariants_ok:
        return False
    if str(config.systematic_gate_scope) != "loop-end":
        return bool(_nonregression(before, after, purpose="thin"))
    debt_before = (
        int(before["superthin_triangle_count"]),
        float(before["superthin_severity_sum"]),
    )
    debt_after = (
        int(after["superthin_triangle_count"]),
        float(after["superthin_severity_sum"]),
    )
    return bool(debt_after < debt_before)


def _repair_superthin_guarded(state: _State, config: AggressiveConditioningConfig, initial_components: int) -> dict[str, Any]:
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
        if _deadline_reached(config):
            break
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
        if not budget_ok or not _thin_transaction_passes(before, trial, ok, config):
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
        if _deadline_reached(config):
            break
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
        if not _thin_transaction_passes(before, trial, ok, config):
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
        if _deadline_reached(config):
            break
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
        if not _thin_transaction_passes(before, trial, ok, config):
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
        if _deadline_reached(config):
            break
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
        if not _thin_transaction_passes(before, trial, ok, config):
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
            if _deadline_reached(config):
                break
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
            if not _thin_transaction_passes(before, trial, ok, config):
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


def _repair_superthin_components(
    state: _State,
    config: AggressiveConditioningConfig,
    initial_components: int,
) -> dict[str, Any]:
    """Repair connected extreme-tail debt with deterministic local cavities."""
    started = time.perf_counter()
    stage_before = _summary(state, config)
    initial_summary = dict(stage_before)
    classifications: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    blocked: dict[str, dict[str, Any]] = {}
    accepted = 0
    for _ in range(max(0, int(config.systematic_max_components_per_round))):
        if _deadline_reached(config):
            break
        components = _inventory_superthin_components(state, config)
        if not components:
            break
        available = [component for component in components if component["component_id"] not in blocked]
        if not available:
            break
        component = available[0]
        classifications.append({key: value for key, value in component.items() if key != "triangle_indices"})
        best: tuple[tuple[float, ...], _State, dict[str, Any]] | None = None
        component_attempts: list[dict[str, Any]] = []
        topology = build_edge_topology(len(state.points), state.triangles)
        for rings in range(
            max(0, int(config.systematic_min_patch_rings)),
            max(0, int(config.systematic_max_patch_rings)) + 1,
        ):
            if _deadline_reached(config):
                break
            patch = _expand_triangle_patch(state.triangles, topology, component["triangle_indices"], rings)
            ring = _ordered_patch_boundary(state.triangles, patch)
            if ring is None:
                component_attempts.append(
                    {
                        "component_id": component["component_id"],
                        "patch_rings": int(rings),
                        "accepted": False,
                        "failures": ["non_simple_patch_boundary"],
                    }
                )
                continue
            support_groups, support_evidence = _systematic_support_groups(
                state,
                component,
                patch,
                ring,
                config,
            )
            for support_index, support in enumerate(support_groups):
                if _deadline_reached(config):
                    break
                trial = state.clone()
                changed, construction = _retriangulate_patch_with_support(
                    trial,
                    patch,
                    ring,
                    support,
                    component,
                    config,
                )
                record: dict[str, Any] = {
                    "component_id": component["component_id"],
                    "classification": component["classification"],
                    "patch_rings": int(rings),
                    "patch_triangle_count": int(len(patch)),
                    "support_group": int(support_index),
                    "support_point_count": int(len(support)),
                    "support_evidence": support_evidence,
                    "accepted": False,
                }
                if not changed:
                    record["failures"] = construction or ["cavity_construction_failed"]
                    component_attempts.append(record)
                    continue
                _micro_relax(trial, replacement_seed_nodes=trial.last_affected, config=config)
                ok, invariant_report, trial_summary = _audit_state(trial, config, initial_components)
                loop_end_scope = str(config.systematic_gate_scope) == "loop-end"
                nonregression = loop_end_scope or _nonregression(stage_before, trial_summary, purpose="thin")
                failures: list[str] = []
                if loop_end_scope and not ok:
                    failures = _failed_invariant_names(invariant_report)
                elif not ok or not nonregression:
                    failures = _thin_rejection(
                        "systematic-component-cavity",
                        tuple(component["node_lineage"]),
                        stage_before,
                        trial_summary,
                        ok,
                        invariant_report=invariant_report,
                    )["failures"]
                if not nonregression:
                    failures = sorted(set([*failures, "global_thin_nonregression_gate"]))
                debt_before = (
                    int(stage_before["superthin_triangle_count"]),
                    float(stage_before["superthin_severity_sum"]),
                )
                debt_after = (
                    int(trial_summary["superthin_triangle_count"]),
                    float(trial_summary["superthin_severity_sum"]),
                )
                if not debt_after < debt_before:
                    failures = sorted(set([*failures, "superthin_debt_not_reduced"]))
                record["trial"] = _summary_from(trial_summary)
                record["invariants"] = invariant_report
                record["failures"] = failures
                if failures:
                    component_attempts.append(record)
                    continue
                score = (
                    float(trial_summary["superthin_triangle_count"]),
                    float(trial_summary["superthin_severity_sum"]),
                    -float(trial_summary["q_min"]),
                    -float(trial_summary["minimum_angle_deg"]),
                    float(trial_summary["count_valence_above_limit"]),
                    float(trial_summary["l_over_h_count_above_1_55"]),
                    float(trial_summary["area_transition_count_above_0_50"]),
                    float(len(support)),
                    float(rings),
                    float(support_index),
                )
                record["failures"] = []
                record["candidate_score"] = list(score)
                component_attempts.append(record)
                if best is None or score < best[0]:
                    best = (score, trial, record)
        attempts.extend(component_attempts)
        if best is None:
            blocked[component["component_id"]] = {
                **{key: value for key, value in component.items() if key != "triangle_indices"},
                "attempt_count": int(len(component_attempts)),
                "failure_counts": _failure_counts(component_attempts),
            }
            continue
        _, selected, selected_record = best
        before_count = int(stage_before["superthin_triangle_count"])
        _restore(state, selected)
        selected_record["accepted"] = True
        state.ledger.append(
            {
                "operation": "systematic-superthin-component-cavity",
                "component_id": component["component_id"],
                "classification": component["classification"],
                "source_triangle_lineage": component["node_lineage"],
                "patch_rings": int(selected_record["patch_rings"]),
                "support_point_count": int(selected_record["support_point_count"]),
                "local_feature_target_m": component.get("local_feature_target_m"),
                "superthin_before": before_count,
                "superthin_after": int(_summary(state, config)["superthin_triangle_count"]),
            }
        )
        accepted += 1
        stage_before = _summary(state, config)
        blocked.clear()
    final_components = _inventory_superthin_components(state, config)
    for component in final_components:
        blocked.setdefault(
            component["component_id"],
            {key: value for key, value in component.items() if key != "triangle_indices"},
        )
    return {
        "schema_version": "fvcom_systematic_thin_repair_v2",
        "profile": "systematic-v2",
        "accepted": int(accepted),
        "rejected": int(sum(not bool(item.get("accepted")) for item in attempts)),
        "component_classifications": classifications,
        "candidate_attempts": attempts,
        "blocked_components": list(blocked.values()),
        "before": initial_summary,
        "after": _summary(state, config),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _connectivity_restriction_config(
    config: AggressiveConditioningConfig,
) -> ConnectivityRestrictionConfig:
    return ConnectivityRestrictionConfig(
        enabled=bool(config.systematic_v5_enable_connectivity_restriction),
        maximum_transactions=max(
            0,
            int(config.systematic_v5_max_connectivity_transactions_per_round),
        ),
        maximum_candidates_per_component=max(
            0,
            int(config.systematic_v5_max_connectivity_candidates_per_component),
        ),
        patch_ring_ladder=tuple(
            int(value)
            for value in config.systematic_v5_patch_ring_ladder
        ),
        shortcut_arc_chord_ratio=float(
            config.systematic_v5_connectivity_shortcut_arc_chord_ratio
        ),
        shortcut_arc_target_ratio=float(
            config.systematic_v5_connectivity_shortcut_arc_target_ratio
        ),
    )


def _allowed_edge_policy(
    state: _State,
    config: AggressiveConditioningConfig,
) -> AllowedEdgePolicy:
    return AllowedEdgePolicy(
        state.points,
        state.targets,
        state.chains,
        state.lineage,
        restricted_lineage_edges=state.restricted_lineage_edges,
        config=_connectivity_restriction_config(config),
    )


def _repair_superthin_connectivity_v1(
    state: _State,
    config: AggressiveConditioningConfig,
    initial_components: int,
) -> dict[str, Any]:
    """Remove causal superthin connectivity without moving or adding nodes."""
    started = time.perf_counter()
    before = _summary(state, config)
    settings = asdict(_connectivity_restriction_config(config))
    if (
        not bool(config.systematic_v5_enable_connectivity_restriction)
        or int(config.systematic_v5_max_connectivity_transactions_per_round)
        <= 0
    ):
        return {
            "schema_version": "fvcom_superthin_connectivity_restriction_v1",
            "profile": "superthin-connectivity-v1",
            "enabled": False,
            "reason": "connectivity_restriction_disabled",
            "settings": settings,
            "accepted": 0,
            "rejected": 0,
            "candidate_attempts": [],
            "blocked_components": [],
            "restricted_lineage_edges": [
                list(map(int, edge))
                for edge in sorted(state.restricted_lineage_edges)
            ],
            "restricted_edge_violation_count": int(
                len(_restricted_edge_violation_records(state))
            ),
            "before": before,
            "after": before,
            "runtime_seconds": float(time.perf_counter() - started),
        }

    accepted = 0
    rejected = 0
    attempts: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    blocked: dict[str, dict[str, Any]] = {}
    transaction_limit = max(
        0,
        int(config.systematic_v5_max_connectivity_transactions_per_round),
    )
    audit_config = replace(config, systematic_gate_scope="loop-end")
    passage_baseline = _passage_clearance_inventory(state, config)
    for _ in range(transaction_limit):
        if _deadline_reached(
            config,
            reserve_seconds=float(config.systematic_v5_audit_reserve_seconds),
        ):
            break
        components = _inventory_superthin_components(state, config)
        if not components:
            break
        component = next(
            (
                value
                for value in components
                if str(value["component_id"]) not in blocked
            ),
            None,
        )
        if component is None:
            break
        component_id = str(component["component_id"])
        classifications.append(
            {
                key: value
                for key, value in component.items()
                if key != "triangle_indices"
            }
        )
        geometry = triangle_geometry(state.points, state.triangles)
        minimum_angles = np.min(geometry["angles_deg"], axis=1)
        superthin_mask = (
            geometry["quality"]
            < float(config.superthin_quality_threshold)
        ) | (
            minimum_angles
            < float(config.superthin_min_angle_deg)
        )
        policy = _allowed_edge_policy(state, config)
        records = policy.candidate_records(
            state.triangles,
            component["triangle_indices"],
            superthin_mask,
        )
        component_attempts: list[dict[str, Any]] = []
        best: tuple[tuple[float, ...], _State, dict[str, Any]] | None = None
        before_summary = _summary(state, config)
        for record in records:
            if _deadline_reached(
                config,
                reserve_seconds=float(config.systematic_v5_audit_reserve_seconds),
            ):
                break
            edge = tuple(map(int, record["edge"]))
            patches = _connectivity_patch_candidates_v1(
                state,
                component,
                edge,
                config,
            )
            for patch_mode, patch in patches:
                if _deadline_reached(
                    config,
                    reserve_seconds=float(config.systematic_v5_audit_reserve_seconds),
                ):
                    break
                trial = state.clone()
                lineage_edge = tuple(
                    sorted(map(int, record["lineage_edge"]))
                )
                trial.restricted_lineage_edges.add(lineage_edge)
                changed, failures, evidence = (
                    _reconstruct_connectivity_patch_v1(
                        trial,
                        patch,
                        forbidden_lineage_edge=lineage_edge,
                        config=config,
                    )
                )
                attempt: dict[str, Any] = {
                    "component_id": component_id,
                    "classification": component["classification"],
                    "candidate_rank": int(record["candidate_rank"]),
                    "edge": list(map(int, edge)),
                    "lineage_edge": list(map(int, lineage_edge)),
                    "edge_evidence": {
                        key: value
                        for key, value in record.items()
                        if key
                        not in {
                            "edge",
                            "lineage_edge",
                            "candidate_rank",
                        }
                    },
                    "patch_mode": str(patch_mode),
                    "patch_triangle_count": int(len(patch)),
                    "accepted": False,
                }
                if not changed:
                    attempt["failures"] = failures or [
                        "connectivity_patch_construction_failed"
                    ]
                    component_attempts.append(attempt)
                    rejected += 1
                    continue
                local_before = evidence["local_debt_before"]
                local_after = evidence["local_debt_after"]
                local_debt_before = (
                    int(local_before["superthin_triangle_count"]),
                    float(local_before["superthin_severity_sum"]),
                )
                local_debt_after = (
                    int(local_after["superthin_triangle_count"]),
                    float(local_after["superthin_severity_sum"]),
                )
                strict_isolated_transaction = bool(
                    config.systematic_v5_connectivity_only
                )
                local_debt_is_admissible = (
                    local_debt_after < local_debt_before
                    if strict_isolated_transaction
                    else local_debt_after[0] <= local_debt_before[0]
                )
                if not local_debt_is_admissible:
                    attempt.update(
                        {
                            "evidence": evidence,
                            "failures": [
                                "replacement_does_not_reduce_patch_superthin"
                                if strict_isolated_transaction
                                else "replacement_increases_patch_superthin_count"
                            ],
                        }
                    )
                    component_attempts.append(attempt)
                    rejected += 1
                    continue
                ok, invariant_report, trial_summary = _audit_state(
                    trial,
                    audit_config,
                    initial_components,
                )
                hard_failures = _v5_hard_gate_failures(
                    state,
                    trial,
                    config,
                    passage_baseline,
                )
                gate_failures = (
                    []
                    if ok
                    else _failed_invariant_names(invariant_report)
                )
                gate_failures = sorted(
                    set([*gate_failures, *hard_failures])
                )
                if len(trial.points) != len(state.points):
                    gate_failures.append("node_count_changed")
                elif not np.array_equal(trial.points, state.points):
                    gate_failures.append("node_coordinates_changed")
                if trial.chains != state.chains:
                    gate_failures.append("boundary_chain_changed")
                if not np.array_equal(trial.open_nodes, state.open_nodes):
                    gate_failures.append("open_boundary_membership_changed")
                if (
                    strict_isolated_transaction
                    and not _connectivity_nonregression(
                        before_summary,
                        trial_summary,
                    )
                ):
                    gate_failures.append(
                        "global_connectivity_nonregression_gate"
                    )
                global_debt_before = (
                    int(before_summary["superthin_triangle_count"]),
                    float(before_summary["superthin_severity_sum"]),
                )
                global_debt_after = (
                    int(trial_summary["superthin_triangle_count"]),
                    float(trial_summary["superthin_severity_sum"]),
                )
                if strict_isolated_transaction:
                    if not global_debt_after < global_debt_before:
                        gate_failures.append(
                            "global_superthin_debt_not_reduced"
                        )
                elif global_debt_after[0] > global_debt_before[0]:
                    gate_failures.append(
                        "global_superthin_count_increased_during_transfer"
                    )
                gate_failures = sorted(set(gate_failures))
                attempt.update(
                    {
                        "evidence": evidence,
                        "trial": _summary_from(trial_summary),
                        "invariants": invariant_report,
                        "failures": gate_failures,
                    }
                )
                component_attempts.append(attempt)
                if gate_failures:
                    rejected += 1
                    continue
                score = (
                    float(trial_summary["superthin_triangle_count"]),
                    float(trial_summary["superthin_severity_sum"]),
                    float(record["candidate_rank"]),
                    -float(trial_summary["q_min"]),
                    -float(trial_summary["q_l3_sigma"]),
                    float(
                        trial_summary[
                            "l_over_h_count_above_1_55"
                        ]
                    ),
                    float(
                        trial_summary[
                            "area_transition_count_above_0_50"
                        ]
                    ),
                    float(len(patch)),
                )
                attempt["candidate_score"] = list(score)
                if best is None or score < best[0]:
                    best = (score, trial, attempt)
            if (
                best is not None
                and int(best[2]["candidate_rank"])
                == int(record["candidate_rank"])
            ):
                # Candidate ranking is part of the policy.  Finish the
                # preferred edge's complete patch ladder, then commit its
                # best admissible reconstruction without screening
                # lower-ranked causal hypotheses.
                break
        attempts.extend(component_attempts)
        if best is None:
            blocked[component_id] = {
                **{
                    key: value
                    for key, value in component.items()
                    if key != "triangle_indices"
                },
                "attempt_count": int(len(component_attempts)),
                "failure_counts": _failure_counts(component_attempts),
            }
            continue
        _, winner, winner_record = best
        _restore(state, winner)
        winner_record["accepted"] = True
        winner_record["failures"] = []
        accepted += 1
        blocked.pop(component_id, None)

    after = _summary(state, config)
    return {
        "schema_version": "fvcom_superthin_connectivity_restriction_v1",
        "profile": "superthin-connectivity-v1",
        "enabled": True,
        "settings": settings,
        "closure_succeeded": bool(
            after["superthin_triangle_count"] == 0
        ),
        "accepted": int(accepted),
        "rejected": int(rejected),
        "component_classifications": classifications,
        "candidate_attempts": attempts,
        "blocked_components": list(blocked.values()),
        "restricted_lineage_edges": [
            list(map(int, edge))
            for edge in sorted(state.restricted_lineage_edges)
        ],
        "restricted_edge_violation_count": int(
            len(_restricted_edge_violation_records(state))
        ),
        "deadline_reached": bool(_deadline_reached(config)),
        "before": before,
        "after": after,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _connectivity_patch_candidates_v1(
    state: _State,
    component: dict[str, Any],
    edge: tuple[int, int],
    config: AggressiveConditioningConfig,
) -> list[tuple[str, np.ndarray]]:
    topology = build_edge_topology(len(state.points), state.triangles)
    attached = set(
        map(int, topology.edge_to_triangles.get(tuple(sorted(edge)), []))
    )
    component_triangles = set(
        map(int, component.get("triangle_indices", []))
    )
    seeds = attached | component_triangles
    patches: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()

    def append(name: str, values: Iterable[int]) -> None:
        key = tuple(sorted(set(map(int, values))))
        if not key or key in seen or len(key) > 5000:
            return
        seen.add(key)
        patches.append((name, np.asarray(key, dtype=int)))

    append("component-edge", seeds)
    endpoint_stars = set(seeds)
    for node in edge:
        endpoint_stars.update(
            map(
                int,
                np.where(
                    np.any(state.triangles == int(node), axis=1)
                )[0],
            )
        )
    append("coupled-endpoint-stars", endpoint_stars)
    for rings in tuple(
        int(value)
        for value in config.systematic_v5_patch_ring_ladder
    ):
        if rings <= 0:
            continue
        expanded = _expand_triangle_patch(
            state.triangles,
            topology,
            sorted(seeds),
            rings,
        )
        append(f"expanded-{rings}-ring", expanded)
    return patches


def _reconstruct_connectivity_patch_v1(
    state: _State,
    patch: np.ndarray,
    *,
    forbidden_lineage_edge: tuple[int, int],
    config: AggressiveConditioningConfig,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Retriangulate one constrained patch with all coordinates unchanged."""
    patch = np.asarray(sorted(set(map(int, patch))), dtype=int)
    if not len(patch):
        return False, ["empty_connectivity_patch"], {}
    old_triangles = state.triangles[patch].copy()
    old_nodes = sorted(set(map(int, np.unique(old_triangles))))
    protected = chain_edges(state.chains)
    groups = _patch_groups_split_by_protected_chords(
        state.triangles,
        patch,
        protected,
    )
    if not groups:
        return False, ["connectivity_patch_has_no_constrained_faces"], {}
    policy = _allowed_edge_policy(state, config)

    def edge_allowed(edge: tuple[int, int]) -> bool:
        return bool(
            policy.is_allowed(
                edge,
                reject_same_chain_shortcuts=False,
            )
        )

    replacements: list[np.ndarray] = []
    face_evidence: list[dict[str, Any]] = []
    for group in groups:
        ring = _ordered_patch_boundary(state.triangles, group)
        if ring is None:
            return False, ["non_simple_constrained_subface"], {}
        ring_polygon = Polygon(
            state.points[np.asarray(ring, dtype=int)]
        )
        if not ring_polygon.is_valid or ring_polygon.area <= 0.0:
            return False, ["invalid_constrained_subface_polygon"], {}
        group_nodes = sorted(
            set(
                map(
                    int,
                    np.unique(state.triangles[group]),
                )
            )
        )
        interior_nodes = sorted(set(group_nodes) - set(ring))
        replacement = _triangulate_ring_with_existing_hub_v1(
            state.points,
            ring,
            interior_nodes,
            edge_allowed=edge_allowed,
        )
        triangulation_method = "existing-interior-hub"
        nodes_to_insert = [
            node
            for node in interior_nodes
            if replacement is None
            or node not in set(map(int, np.unique(replacement)))
        ]
        if replacement is None:
            replacement = _triangulate_ring_greedy(
                state.points,
                ring,
                None,
                max(int(config.max_valence), 8),
                edge_allowed=edge_allowed,
            )
            triangulation_method = "allowed-edge-ear"
            nodes_to_insert = interior_nodes
        if replacement is None:
            return False, ["allowed_edge_ear_triangulation_failed"], {}
        replacement, insertion_failures = (
            _insert_existing_patch_nodes_v1(
                state.points,
                replacement,
                nodes_to_insert,
                edge_allowed=edge_allowed,
            )
        )
        if insertion_failures:
            return False, insertion_failures, {}
        replacements.append(replacement)
        face_evidence.append(
            {
                "source_triangle_count": int(len(group)),
                "ring_node_count": int(len(ring)),
                "interior_node_count": int(len(interior_nodes)),
                "replacement_triangle_count": int(len(replacement)),
                "triangulation_method": triangulation_method,
            }
        )
    replacement = _orient_ccw(
        state.points,
        np.vstack(replacements),
    )
    replacement_nodes = set(map(int, np.unique(replacement)))
    if replacement_nodes != set(old_nodes):
        return False, ["connectivity_patch_node_set_changed"], {}
    replacement_edges = _edge_set(replacement)
    if forbidden_lineage_edge in {
        tuple(
            sorted(
                (
                    int(state.lineage[a]),
                    int(state.lineage[b]),
                )
            )
        )
        for a, b in replacement_edges
    }:
        return False, ["forbidden_lineage_edge_reintroduced"], {}
    old_geometry = triangle_geometry(state.points, old_triangles)
    replacement_geometry = triangle_geometry(
        state.points,
        replacement,
    )
    if np.any(
        replacement_geometry["signed_area"]
        <= _area_tolerance(state.points, replacement)
    ):
        return False, ["nonpositive_connectivity_replacement"], {}
    old_area = float(np.sum(old_geometry["area"]))
    new_area = float(np.sum(replacement_geometry["area"]))
    if abs(new_area - old_area) > 1.0e-8 * max(old_area, 1.0):
        return False, ["connectivity_patch_area_mismatch"], {}
    old_patch_edges = _edge_set(old_triangles)
    protected_inside = protected & old_patch_edges
    if not protected_inside.issubset(replacement_edges):
        return False, ["protected_chord_missing_from_replacement"], {}
    patch_perimeter = _patch_perimeter_edges(
        state.triangles,
        patch,
    )
    if not patch_perimeter.issubset(replacement_edges):
        return False, ["connectivity_patch_perimeter_changed"], {}
    keep = np.ones(len(state.triangles), dtype=bool)
    keep[patch] = False
    outside = state.triangles[keep]
    state.triangles = _orient_ccw(
        state.points,
        np.vstack([outside, replacement]),
    )
    replacement_ids = set(
        range(len(outside), len(outside) + len(replacement))
    )
    flip_count = _lawson_legalize_locked_patch(
        state,
        replacement_ids,
        patch_perimeter | protected,
        max_flips=max(
            0,
            int(config.systematic_v5_max_lawson_flips_per_transaction),
        ),
        edge_allowed=edge_allowed,
    )
    delivered_edges = _edge_set(state.triangles)
    if not patch_perimeter.issubset(delivered_edges):
        return False, ["locked_connectivity_perimeter_changed"], {}
    if not protected.issubset(delivered_edges):
        return False, ["protected_edge_missing"], {}
    if _restricted_edge_violation_records(state):
        return False, ["restricted_edge_present_after_reconstruction"], {}
    state.last_affected = old_nodes
    delivered_patch_geometry = triangle_geometry(
        state.points,
        state.triangles[
            np.asarray(sorted(replacement_ids), dtype=int)
        ],
    )
    evidence = {
        "candidate": "topology-only-coupled-connectivity",
        "forbidden_lineage_edge": list(
            map(int, forbidden_lineage_edge)
        ),
        "source_patch_triangle_count": int(len(patch)),
        "replacement_triangle_count": int(len(replacement)),
        "patch_node_count": int(len(old_nodes)),
        "constrained_subface_count": int(len(groups)),
        "constrained_subfaces": face_evidence,
        "protected_chord_count": int(len(protected_inside)),
        "lawson_flip_count": int(flip_count),
        "node_coordinate_change_m": 0.0,
        "boundary_node_set_changed": False,
        "old_patch_area_m2": float(old_area),
        "new_patch_area_m2": float(new_area),
        "local_debt_before": _geometry_superthin_debt(
            old_geometry,
            config,
        ),
        "local_debt_after": _geometry_superthin_debt(
            delivered_patch_geometry,
            config,
        ),
    }
    state.ledger.append(
        {
            "operation": "systematic-v5-superthin-connectivity-restriction",
            **evidence,
        }
    )
    return True, [], evidence


def _patch_groups_split_by_protected_chords(
    triangles: np.ndarray,
    patch: np.ndarray,
    protected: set[tuple[int, int]],
) -> list[np.ndarray]:
    patch_set = set(map(int, np.asarray(patch, dtype=int)))
    edge_to_patch: dict[tuple[int, int], list[int]] = {}
    for triangle_index in sorted(patch_set):
        for edge in _triangle_edge_keys(
            triangles[int(triangle_index)]
        ):
            edge_to_patch.setdefault(edge, []).append(
                int(triangle_index)
            )
    adjacency: dict[int, set[int]] = {
        index: set()
        for index in patch_set
    }
    for edge, attached in edge_to_patch.items():
        if edge in protected or len(attached) != 2:
            continue
        left, right = map(int, attached)
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(patch_set)
    groups: list[np.ndarray] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        group: list[int] = []
        while stack:
            current = stack.pop()
            group.append(int(current))
            for following in sorted(
                adjacency[current] & remaining,
                reverse=True,
            ):
                remaining.remove(following)
                stack.append(int(following))
        groups.append(
            np.asarray(sorted(group), dtype=int)
        )
    return groups


def _patch_perimeter_edges(
    triangles: np.ndarray,
    patch: np.ndarray,
) -> set[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for triangle in np.asarray(triangles, dtype=int)[
        np.asarray(patch, dtype=int)
    ]:
        for edge in _triangle_edge_keys(triangle):
            counts[edge] = counts.get(edge, 0) + 1
    return {
        edge
        for edge, count in counts.items()
        if count == 1
    }


def _triangulate_ring_with_existing_hub_v1(
    points: np.ndarray,
    ring: list[int],
    interior_nodes: Iterable[int],
    *,
    edge_allowed: Callable[[tuple[int, int]], bool],
) -> np.ndarray | None:
    """Use an unchanged interior node as a constrained polygon fan hub."""
    if len(ring) < 3:
        return None
    polygon = Polygon(points[np.asarray(ring, dtype=int)])
    if not polygon.is_valid or polygon.area <= 0.0:
        return None
    span = np.ptp(points[np.asarray(ring, dtype=int)], axis=0)
    tolerance = max(1.0e-12, 1.0e-12 * float(np.max(span)))
    covered_polygon = polygon.buffer(tolerance)
    candidates: list[tuple[tuple[float, ...], np.ndarray]] = []
    for hub in sorted(set(map(int, interior_nodes))):
        fan = _orient_ccw(
            points,
            np.asarray(
                [
                    [hub, int(ring[index]), int(ring[(index + 1) % len(ring)])]
                    for index in range(len(ring))
                ],
                dtype=int,
            ),
        )
        geometry = triangle_geometry(points, fan)
        if np.any(
            geometry["signed_area"]
            <= _area_tolerance(points, fan)
        ):
            continue
        if any(
            not covered_polygon.covers(
                Polygon(points[np.asarray(triangle, dtype=int)])
            )
            for triangle in fan
        ):
            continue
        if any(
            not edge_allowed(edge)
            for edge in _edge_set(fan)
        ):
            continue
        area = float(np.sum(geometry["area"]))
        if abs(area - float(polygon.area)) > 1.0e-8 * max(
            float(polygon.area),
            1.0,
        ):
            continue
        score = (
            -float(np.min(geometry["quality"])),
            -float(np.min(geometry["angles_deg"])),
            float(hub),
        )
        candidates.append((score, fan))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _insert_existing_patch_nodes_v1(
    points: np.ndarray,
    triangles: np.ndarray,
    nodes: Iterable[int],
    *,
    edge_allowed: Callable[[tuple[int, int]], bool],
) -> tuple[np.ndarray, list[str]]:
    """Insert existing fixed-coordinate nodes into a polygon triangulation."""
    delivered = _orient_ccw(
        points,
        np.asarray(triangles, dtype=int),
    )
    for node in sorted(set(map(int, nodes))):
        if node in set(map(int, np.unique(delivered))):
            continue
        containing: list[
            tuple[int, np.ndarray]
        ] = []
        for triangle_index, triangle in enumerate(delivered):
            barycentric = _barycentric_coordinates(
                points[int(node)],
                points[np.asarray(triangle, dtype=int)],
            )
            if barycentric is not None and np.all(
                barycentric >= -1.0e-10
            ):
                containing.append(
                    (int(triangle_index), barycentric)
                )
        if not containing:
            return delivered, [
                "existing_patch_node_outside_retriangulation"
            ]
        edge_hit: tuple[int, int] | None = None
        for triangle_index, barycentric in containing:
            zero = np.where(np.abs(barycentric) <= 1.0e-10)[0]
            if len(zero):
                triangle = delivered[int(triangle_index)]
                opposite = int(zero[0])
                edge_hit = tuple(
                    sorted(
                        map(
                            int,
                            np.delete(triangle, opposite),
                        )
                    )
                )
                break
        if edge_hit is None:
            triangle_index = int(containing[0][0])
            triangle = list(
                map(int, delivered[triangle_index])
            )
            new_edges = [
                tuple(sorted((node, vertex)))
                for vertex in triangle
            ]
            if any(
                not edge_allowed(edge)
                for edge in new_edges
            ):
                return delivered, [
                    "existing_node_insertion_creates_restricted_edge"
                ]
            replacement = np.asarray(
                [
                    [node, triangle[0], triangle[1]],
                    [node, triangle[1], triangle[2]],
                    [node, triangle[2], triangle[0]],
                ],
                dtype=int,
            )
            delivered = np.vstack(
                [
                    np.delete(
                        delivered,
                        triangle_index,
                        axis=0,
                    ),
                    replacement,
                ]
            )
            delivered = _orient_ccw(points, delivered)
            continue
        attached = [
            index
            for index, triangle in enumerate(delivered)
            if set(edge_hit).issubset(
                set(map(int, triangle))
            )
        ]
        if not attached or len(attached) > 2:
            return delivered, [
                "existing_node_edge_insertion_nonmanifold"
            ]
        replacement_rows: list[list[int]] = []
        for triangle_index in attached:
            triangle = list(
                map(int, delivered[int(triangle_index)])
            )
            opposite_values = [
                value
                for value in triangle
                if value not in edge_hit
            ]
            if len(opposite_values) != 1:
                return delivered, [
                    "existing_node_edge_insertion_invalid_triangle"
                ]
            opposite = int(opposite_values[0])
            replacement_rows.extend(
                [
                    [node, int(edge_hit[0]), opposite],
                    [node, opposite, int(edge_hit[1])],
                ]
            )
        new_edges = {
            tuple(sorted((node, int(edge_hit[0])))),
            tuple(sorted((node, int(edge_hit[1])))),
            *(
                tuple(
                    sorted(
                        (
                            node,
                            int(
                                next(
                                    value
                                    for value in delivered[index]
                                    if int(value)
                                    not in edge_hit
                                )
                            ),
                        )
                    )
                )
                for index in attached
            ),
        }
        if any(
            not edge_allowed(edge)
            for edge in new_edges
        ):
            return delivered, [
                "existing_node_edge_insertion_creates_restricted_edge"
            ]
        keep = np.ones(len(delivered), dtype=bool)
        keep[np.asarray(attached, dtype=int)] = False
        delivered = np.vstack(
            [
                delivered[keep],
                np.asarray(replacement_rows, dtype=int),
            ]
        )
        delivered = _orient_ccw(points, delivered)
    geometry = triangle_geometry(points, delivered)
    if np.any(
        geometry["signed_area"]
        <= _area_tolerance(points, delivered)
    ):
        return delivered, [
            "existing_node_insertion_nonpositive_geometry"
        ]
    return delivered, []


def _barycentric_coordinates(
    point: np.ndarray,
    triangle: np.ndarray,
) -> np.ndarray | None:
    a, b, c = np.asarray(triangle, dtype=float)
    matrix = np.column_stack((b - a, c - a))
    try:
        uv = np.linalg.solve(
            matrix,
            np.asarray(point, dtype=float) - a,
        )
    except np.linalg.LinAlgError:
        return None
    return np.asarray(
        [1.0 - float(uv[0]) - float(uv[1]), uv[0], uv[1]],
        dtype=float,
    )


def _failure_counts(
    attempts: Iterable[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attempt in attempts:
        for failure in attempt.get("failures", []):
            key = str(failure)
            counts[key] = counts.get(key, 0) + 1
    return dict(
        sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )
    )


def _repair_superthin_systematic_v5(
    state: _State,
    config: AggressiveConditioningConfig,
    initial_components: int,
) -> dict[str, Any]:
    """Restrict causal edges, run locked stars, and finish with closure."""
    started = time.perf_counter()
    before = _summary(state, config)
    connectivity_initial = _repair_superthin_connectivity_v1(
        state,
        config,
        initial_components,
    )
    if bool(config.systematic_v5_connectivity_only):
        after = _summary(state, config)
        return {
            "schema_version": "fvcom_systematic_thin_repair_v5",
            "profile": "systematic-v5",
            "closure_succeeded": bool(
                after["superthin_triangle_count"] == 0
            ),
            "accepted": int(
                connectivity_initial.get("accepted", 0)
            ),
            "rejected": int(
                connectivity_initial.get("rejected", 0)
            ),
            "connectivity_restriction_initial": connectivity_initial,
            "connectivity_restriction_recheck": None,
            "locked_star_initial": None,
            "component_cavity_recovery": None,
            "boundary_adaptive_recovery": None,
            "locked_star_final": None,
            "component_classifications": list(
                connectivity_initial.get(
                    "component_classifications",
                    [],
                )
            ),
            "candidate_attempts": list(
                connectivity_initial.get("candidate_attempts", [])
            ),
            "blocked_components": list(
                connectivity_initial.get("blocked_components", [])
            ),
            "recurrence_counts": {},
            "restricted_lineage_edges": [
                list(map(int, edge))
                for edge in sorted(state.restricted_lineage_edges)
            ],
            "restricted_edge_violation_count": int(
                len(_restricted_edge_violation_records(state))
            ),
            "deadline_reached": bool(_deadline_reached(config)),
            "before": before,
            "after": after,
            "runtime_seconds": float(
                time.perf_counter() - started
            ),
        }
    first = _repair_superthin_locked_star_v5(state, config, initial_components)
    cavity: dict[str, Any] | None = None
    connectivity_recheck: dict[str, Any] | None = None
    boundary: dict[str, Any] | None = None
    second: dict[str, Any] | None = None
    residual = _inventory_superthin_components(state, config)
    if residual and not _deadline_reached(
        config,
        reserve_seconds=float(config.systematic_v5_audit_reserve_seconds),
    ):
        cavity = _repair_superthin_components(
            state,
            replace(
                config,
                thin_repair_profile="systematic-v2",
                systematic_gate_scope="loop-end",
                systematic_min_patch_rings=1,
                systematic_max_patch_rings=4,
                systematic_max_components_per_round=min(
                    16,
                    max(1, int(config.systematic_max_components_per_round)),
                ),
                micro_relax_cycles=min(1, int(config.micro_relax_cycles)),
                micro_relax_iterations=min(2, int(config.micro_relax_iterations)),
                micro_relax_ring_layers=2,
            ),
            initial_components,
        )
        residual = _inventory_superthin_components(state, config)
    if (
        residual
        and bool(config.systematic_v5_enable_connectivity_restriction)
        and not _deadline_reached(
            config,
            reserve_seconds=float(config.systematic_v5_audit_reserve_seconds),
        )
    ):
        connectivity_recheck = _repair_superthin_connectivity_v1(
            state,
            replace(
                config,
                systematic_v5_max_connectivity_transactions_per_round=max(
                    1,
                    int(
                        config.systematic_v5_max_connectivity_transactions_per_round
                    )
                    - int(connectivity_initial.get("accepted", 0)),
                ),
            ),
            initial_components,
        )
        residual = _inventory_superthin_components(state, config)
    has_boundary_residual = any(
        bool(component.get("boundary_chain_ids"))
        for component in residual
    )
    if (
        has_boundary_residual
        and bool(config.systematic_v5_enable_boundary_window_fallback)
        and not _deadline_reached(
            config,
            reserve_seconds=float(config.systematic_v5_audit_reserve_seconds),
        )
    ):
        boundary_config = replace(
            config,
            thin_repair_profile="systematic-v3",
            systematic_gate_scope="loop-end",
            systematic_max_components_per_round=min(
                8,
                max(1, int(config.systematic_max_components_per_round)),
            ),
            systematic_v3_max_candidates_per_component=min(
                8,
                max(1, int(config.systematic_v3_max_candidates_per_component)),
            ),
            systematic_collapse_welds_per_round=0,
            max_boundary_welds_per_round=min(
                8,
                max(1, int(config.max_boundary_welds_per_round)),
            ),
            max_boundary_edits_per_round=min(
                8,
                max(1, int(config.max_boundary_edits_per_round)),
            ),
            micro_relax_cycles=min(1, int(config.micro_relax_cycles)),
            micro_relax_iterations=min(2, int(config.micro_relax_iterations)),
            micro_relax_ring_layers=2,
        )
        boundary = _repair_superthin_boundary_adaptive_v3(
            state,
            boundary_config,
            initial_components,
        )
    if (
        int(_summary(state, config)["superthin_triangle_count"]) > 0
        and not _deadline_reached(
            config,
            reserve_seconds=float(config.systematic_v5_audit_reserve_seconds),
        )
    ):
        second = _repair_superthin_locked_star_v5(
            state,
            replace(
                config,
                systematic_v5_max_star_transactions_per_round=max(
                    1,
                    int(config.systematic_v5_max_star_transactions_per_round)
                    - int(first.get("accepted", 0)),
                ),
            ),
            initial_components,
        )
    after = _summary(state, config)
    reports = [
        value
        for value in (
            connectivity_initial,
            first,
            cavity,
            connectivity_recheck,
            boundary,
            second,
        )
        if value is not None
    ]
    return {
        "schema_version": "fvcom_systematic_thin_repair_v5",
        "profile": "systematic-v5",
        "closure_succeeded": bool(
            after["superthin_triangle_count"] == 0
            and after["restricted_edge_violation_count"] == 0
            and _connectivity_nonregression(before, after)
        ),
        "accepted": int(sum(int(value.get("accepted", 0)) for value in reports)),
        "rejected": int(sum(int(value.get("rejected", 0)) for value in reports)),
        "connectivity_restriction_initial": connectivity_initial,
        "locked_star_initial": first,
        "component_cavity_recovery": cavity,
        "connectivity_restriction_recheck": connectivity_recheck,
        "boundary_adaptive_recovery": boundary,
        "locked_star_final": second,
        "component_classifications": [
            item
            for value in reports
            for item in value.get("component_classifications", [])
        ],
        "candidate_attempts": [
            item
            for value in reports
            for item in value.get("candidate_attempts", [])
        ],
        "blocked_components": (
            (
                second
                or boundary
                or connectivity_recheck
                or cavity
                or first
                or connectivity_initial
            ).get("blocked_components", [])
        ),
        "recurrence_counts": {
            str(key): int(count)
            for value in (first, second)
            if value is not None
            for key, count in value.get("recurrence_counts", {}).items()
        },
        "restricted_lineage_edges": [
            list(map(int, edge))
            for edge in sorted(state.restricted_lineage_edges)
        ],
        "restricted_edge_violation_count": int(
            len(_restricted_edge_violation_records(state))
        ),
        "deadline_reached": bool(_deadline_reached(config)),
        "before": before,
        "after": after,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _repair_superthin_locked_star_v5(
    state: _State,
    config: AggressiveConditioningConfig,
    initial_components: int,
) -> dict[str, Any]:
    """Exhaust superthin debt through complete, atomic node-star replacement.

    V4 moved an apex and repaired only the two triangles adjacent to its
    altitude base.  Every other face incident on that apex retained its old
    connectivity.  V5 locks the complete one-ring perimeter, removes every
    face in the old fan, and installs a fully legal replacement before the
    trial can be audited or committed.
    """
    started = time.perf_counter()
    initial_summary = _summary(state, config)
    current_summary = dict(initial_summary)
    attempts: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    blocked: dict[str, dict[str, Any]] = {}
    accepted = 0
    recurrence: dict[str, int] = {}
    audit_config = replace(config, systematic_gate_scope="loop-end")
    transaction_limit = max(0, int(config.systematic_v5_max_star_transactions_per_round))
    passage_baseline = _passage_clearance_inventory(state, config)
    for _ in range(transaction_limit):
        if _deadline_reached(
            config,
            reserve_seconds=float(config.systematic_v5_audit_reserve_seconds),
        ):
            break
        components = _inventory_superthin_components(state, config)
        if not components:
            break
        component = next(
            (value for value in components if value["component_id"] not in blocked),
            None,
        )
        if component is None:
            break
        component_id = str(component["component_id"])
        recurrence[component_id] = recurrence.get(component_id, 0) + 1
        classifications.append(
            {key: value for key, value in component.items() if key != "triangle_indices"}
        )
        geometry = triangle_geometry(state.points, state.triangles)
        minimum_angles = np.min(geometry["angles_deg"], axis=1)
        triangle_order = sorted(
            map(int, component["triangle_indices"]),
            key=lambda index: (
                float(geometry["quality"][index]),
                float(minimum_angles[index]),
                int(index),
            ),
        )
        best: tuple[tuple[float, ...], _State, dict[str, Any]] | None = None
        component_attempts: list[dict[str, Any]] = []
        pending_candidates: list[
            tuple[tuple[float, ...], _State, dict[str, Any], dict[str, Any]]
        ] = []
        screened_centers: set[int] = set()
        for triangle_index in triangle_order:
            if _deadline_reached(
                config,
                reserve_seconds=float(config.systematic_v5_audit_reserve_seconds),
            ):
                break
            triangle = list(map(int, state.triangles[int(triangle_index)]))
            protected = chain_edges(state.chains)
            triangle_coordinates = state.points[np.asarray(triangle, dtype=int)]
            opposite_lengths = np.asarray(
                [
                    np.linalg.norm(triangle_coordinates[1] - triangle_coordinates[2]),
                    np.linalg.norm(triangle_coordinates[0] - triangle_coordinates[2]),
                    np.linalg.norm(triangle_coordinates[0] - triangle_coordinates[1]),
                ],
                dtype=float,
            )
            # The minimum-angle vertex opposite the longest edge is the
            # unique shortest-altitude causal apex.  Testing the other two
            # vertices only repeats non-causal global audits.
            causal_center = int(triangle[int(np.argmax(opposite_lengths))])
            for center in [causal_center]:
                if _deadline_reached(
                    config,
                    reserve_seconds=float(config.systematic_v5_audit_reserve_seconds),
                ):
                    break
                if int(center) in screened_centers:
                    continue
                screened_centers.add(int(center))
                incident = np.where(np.any(state.triangles == int(center), axis=1))[0]
                ring = _ordered_one_ring(state.triangles[incident], int(center))
                if ring is None:
                    boundary_trial = state.clone()
                    changed, construction, evidence = _reconstruct_locked_boundary_fan_candidate(
                        boundary_trial,
                        center=int(center),
                        triangle_index=int(triangle_index),
                        config=config,
                    )
                    if changed:
                        mode = {"name": "boundary-fan-ear-retriangulation"}
                        quick = _v5_quick_metrics_from_evidence(
                            current_summary,
                            evidence,
                        )
                        record = {
                            "component_id": component_id,
                            "classification": component["classification"],
                            "triangle_index": int(triangle_index),
                            "center_lineage": int(state.lineage[int(center)]),
                            "candidate": str(mode["name"]),
                            "accepted": False,
                            "star_triangle_count": int(len(incident)),
                            "ring_node_count": int(evidence.get("patch_ring_node_count", 0)),
                            "patch_rings": 1,
                            "evidence": evidence,
                            "quick_trial": quick,
                        }
                        debt_before = (
                            int(current_summary["superthin_triangle_count"]),
                            float(current_summary["superthin_severity_sum"]),
                        )
                        debt_after = (
                            int(quick["superthin_triangle_count"]),
                            float(quick["superthin_severity_sum"]),
                        )
                        if debt_after < debt_before:
                            quick_score = _v5_quick_candidate_score(
                                quick,
                                evidence,
                                mode,
                            )
                            record["quick_candidate_score"] = list(quick_score)
                            pending_candidates.append(
                                (quick_score, boundary_trial, record, mode)
                            )
                        else:
                            record["failures"] = ["superthin_debt_not_reduced"]
                            component_attempts.append(record)
                        continue
                    component_attempts.append(
                        {
                            "component_id": component_id,
                            "operation": "locked-star-screen",
                            "triangle_index": int(triangle_index),
                            "center_lineage": int(state.lineage[int(center)]),
                            "accepted": False,
                            "failures": construction or ["open_or_nonmanifold_node_star"],
                        }
                    )
                    continue
                incident_edges = _edge_set(state.triangles[incident])
                protected_chords = sorted(
                    edge
                    for edge in protected & incident_edges
                    if int(center) in edge
                )
                if protected_chords:
                    component_attempts.append(
                        {
                            "component_id": component_id,
                            "operation": "locked-star-screen",
                            "triangle_index": int(triangle_index),
                            "center_lineage": int(state.lineage[int(center)]),
                            "accepted": False,
                            "failures": ["center_has_protected_incident_chord"],
                        }
                    )
                    continue
                modes = _locked_star_modes(
                    state,
                    int(center),
                    ring,
                    int(triangle_index),
                    config,
                )
                for mode in modes:
                    if _deadline_reached(
                        config,
                        reserve_seconds=float(config.systematic_v5_audit_reserve_seconds),
                    ):
                        break
                    trial = state.clone()
                    changed, construction, evidence = _reconstruct_locked_star_candidate(
                        trial,
                        center=int(center),
                        triangle_index=int(triangle_index),
                        mode=mode,
                        config=config,
                    )
                    record: dict[str, Any] = {
                        "component_id": component_id,
                        "classification": component["classification"],
                        "triangle_index": int(triangle_index),
                        "center_lineage": int(state.lineage[int(center)]),
                        "candidate": str(mode["name"]),
                        "accepted": False,
                        "star_triangle_count": int(len(incident)),
                        "ring_node_count": int(len(ring)),
                        "patch_rings": 1,
                    }
                    if str(mode["name"]) == "inward-front-multi-support":
                        requested_coordinates = np.asarray(
                            mode.get(
                                "coordinates",
                                np.empty((0, 2), dtype=float),
                            ),
                            dtype=float,
                        ).reshape((-1, 2))
                        record.update(
                            {
                                "requested_support_node_count": int(
                                    mode.get(
                                        "requested_support_node_count",
                                        len(requested_coordinates),
                                    )
                                ),
                                "requested_support_coordinates_xy": [
                                    list(map(float, point))
                                    for point in requested_coordinates
                                ],
                                "support_generation_evidence": dict(
                                    mode.get(
                                        "support_generation_evidence",
                                        {},
                                    )
                                ),
                            }
                        )
                    if not changed:
                        record["failures"] = construction or ["locked_star_construction_failed"]
                        if evidence:
                            record["construction_evidence"] = evidence
                        component_attempts.append(record)
                        continue
                    quick = _v5_quick_metrics_from_evidence(
                        current_summary,
                        evidence,
                    )
                    debt_before = (
                        int(current_summary["superthin_triangle_count"]),
                        float(current_summary["superthin_severity_sum"]),
                    )
                    debt_after = (
                        int(quick["superthin_triangle_count"]),
                        float(quick["superthin_severity_sum"]),
                    )
                    if not debt_after < debt_before:
                        record.update(
                            {
                                "evidence": evidence,
                                "quick_trial": quick,
                                "failures": ["superthin_debt_not_reduced"],
                            }
                        )
                        component_attempts.append(record)
                        continue
                    quick_score = _v5_quick_candidate_score(
                        quick,
                        evidence,
                        mode,
                    )
                    record.update(
                        {
                            "evidence": evidence,
                            "quick_trial": quick,
                            "quick_candidate_score": list(quick_score),
                        }
                    )
                    pending_candidates.append((quick_score, trial, record, mode))
        for pending_index, (_, trial, record, mode) in enumerate(
            sorted(pending_candidates, key=lambda item: item[0])
        ):
            if _deadline_reached(
                config,
                reserve_seconds=float(config.systematic_v5_audit_reserve_seconds),
            ):
                break
            _micro_relax(
                trial,
                replacement_seed_nodes=trial.last_affected,
                config=replace(
                    config,
                    micro_relax_cycles=min(1, int(config.micro_relax_cycles)),
                    micro_relax_iterations=min(2, int(config.micro_relax_iterations)),
                    micro_relax_ring_layers=2,
                ),
            )
            hard_ok, hard_report, trial_summary = _audit_state(
                trial,
                audit_config,
                initial_components,
            )
            extra_failures = _v5_hard_gate_failures(
                state,
                trial,
                config,
                passage_baseline,
            )
            failures = [] if hard_ok else _failed_invariant_names(hard_report)
            failures = sorted(set([*failures, *extra_failures]))
            debt_before = (
                int(current_summary["superthin_triangle_count"]),
                float(current_summary["superthin_severity_sum"]),
            )
            debt_after = (
                int(trial_summary["superthin_triangle_count"]),
                float(trial_summary["superthin_severity_sum"]),
            )
            if not debt_after < debt_before:
                failures = sorted(set([*failures, "superthin_debt_not_reduced"]))
            # These gates are deferred across relaxation, but a closure
            # checkpoint is not eligible to become the champion if it adds
            # either defect.
            if int(trial_summary["singly_connected_triangle_count"]) > int(
                current_summary["singly_connected_triangle_count"]
            ):
                failures = sorted(set([*failures, "new_singly_connected_triangles"]))
            if int(trial_summary["boundary_degree_anomaly_count"]) > int(
                current_summary["boundary_degree_anomaly_count"]
            ):
                failures = sorted(set([*failures, "new_boundary_degree_anomalies"]))
            record.update(
                {
                    "global_audit_rank": int(pending_index),
                    "trial": _summary_from(trial_summary),
                    "invariants": hard_report,
                    "failures": failures,
                }
            )
            component_attempts.append(record)
            if failures:
                continue
            score = _v5_candidate_score(
                trial_summary,
                record["evidence"],
                mode,
            )
            record["candidate_score"] = list(score)
            best = (score, trial, record)
            break
        if best is not None:
            audited_ids = {id(item) for item in component_attempts}
            for _, _, record, _ in pending_candidates:
                if id(record) in audited_ids:
                    continue
                record["failures"] = ["lower_ranked_candidate_not_globally_audited"]
                component_attempts.append(record)
        if best is None:
            expanded_best, expanded_attempts = _try_expanded_locked_patches_v5(
                state,
                component,
                config,
                audit_config,
                initial_components,
                current_summary,
                passage_baseline,
            )
            component_attempts.extend(expanded_attempts)
            if expanded_best is not None:
                best = expanded_best
        attempts.extend(component_attempts)
        if best is None:
            blocked[component_id] = {
                **{key: value for key, value in component.items() if key != "triangle_indices"},
                "attempt_count": int(len(component_attempts)),
                "failure_counts": _failure_counts(component_attempts),
            }
            continue
        _, selected, selected_record = best
        before_debt = (
            int(current_summary["superthin_triangle_count"]),
            float(current_summary["superthin_severity_sum"]),
        )
        _restore(state, selected)
        selected_record["accepted"] = True
        current_summary = _summary(state, config)
        state.ledger.append(
            {
                "operation": "systematic-v5-locked-star-transaction",
                "component_id": component_id,
                "classification": component["classification"],
                "candidate": selected_record["candidate"],
                "patch_rings": int(selected_record.get("patch_rings", 1)),
                "debt_before": list(before_debt),
                "debt_after": [
                    int(current_summary["superthin_triangle_count"]),
                    float(current_summary["superthin_severity_sum"]),
                ],
                "evidence": selected_record.get("evidence", {}),
            }
        )
        accepted += 1
        blocked.clear()
    residual = _inventory_superthin_components(state, config)
    for component in residual:
        blocked.setdefault(
            str(component["component_id"]),
            {key: value for key, value in component.items() if key != "triangle_indices"},
        )
    final_summary = _summary(state, config)
    return {
        "schema_version": "fvcom_systematic_thin_repair_v5",
        "profile": "systematic-v5",
        "closure_succeeded": bool(final_summary["superthin_triangle_count"] == 0),
        "accepted": int(accepted),
        "rejected": int(sum(not bool(item.get("accepted")) for item in attempts)),
        "component_classifications": classifications,
        "candidate_attempts": attempts,
        "blocked_components": list(blocked.values()),
        "recurrence_counts": dict(sorted(recurrence.items())),
        "deadline_reached": bool(_deadline_reached(config)),
        "before": initial_summary,
        "after": final_summary,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _locked_inward_front_support_modes(
    state: _State,
    center: int,
    ring: list[int],
    base: tuple[int, int],
    config: AggressiveConditioningConfig,
) -> list[dict[str, Any]]:
    """Build deterministic multi-node wet-side fronts behind a protected base.

    The complete locked ring is left unchanged.  Candidate points occupy one
    line parallel to the causal protected edge and strictly inside the patch;
    the caller evaluates every 2..N group on an independent state clone.
    """
    center = int(center)
    base = tuple(sorted(map(int, base)))
    requested_limit = max(
        0,
        int(config.systematic_v5_max_inward_front_support_points),
    )
    support_limit = min(8, requested_limit)
    common: dict[str, Any] = {
        "requested_max_support_point_count": int(requested_limit),
        "bounded_max_support_point_count": int(support_limit),
        "hard_support_point_cap": 8,
        "base_node_indices_zero_based": list(map(int, base)),
        "base_node_lineage": [
            int(state.lineage[int(node)]) for node in base
        ],
        "base_coordinates_xy": [
            list(map(float, state.points[int(node)])) for node in base
        ],
        "base_boundary_kinds": [
            str(state.kinds[int(node)]) for node in base
        ],
        "triangulation_method": "deterministic-ear-plus-point-insertion",
        "global_delaunay_used": False,
    }
    open_edges = {
        tuple(sorted((int(left), int(right))))
        for left, right in zip(
            np.asarray(state.open_nodes, dtype=int)[:-1],
            np.asarray(state.open_nodes, dtype=int)[1:],
        )
    }
    common["base_is_open_boundary_edge"] = bool(base in open_edges)

    def failed_mode(*failures: str) -> dict[str, Any]:
        return {
            "name": "inward-front-multi-support",
            "coordinates": np.empty((0, 2), dtype=float),
            "requested_support_node_count": 0,
            "generation_failures": list(map(str, failures)),
            "support_generation_evidence": dict(common),
        }

    if support_limit < 2:
        return [failed_mode("inward_front_support_limit_below_two")]
    if len(ring) < 3 or len(set(map(int, ring))) != len(ring):
        return [failed_mode("invalid_locked_ring_for_inward_front")]
    ring_edges = {
        tuple(
            sorted(
                (
                    int(ring[index]),
                    int(ring[(index + 1) % len(ring)]),
                )
            )
        )
        for index in range(len(ring))
    }
    if base not in ring_edges:
        return [failed_mode("protected_base_not_on_locked_ring")]
    polygon = Polygon(state.points[np.asarray(ring, dtype=int)])
    if not polygon.is_valid or polygon.area <= 0.0:
        return [failed_mode("invalid_locked_ring_polygon")]
    start = np.asarray(state.points[int(base[0])], dtype=float)
    end = np.asarray(state.points[int(base[1])], dtype=float)
    tangent = end - start
    base_length = float(np.linalg.norm(tangent))
    if not np.isfinite(base_length) or base_length <= 1.0e-12:
        return [failed_mode("degenerate_protected_base")]
    tangent /= base_length
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    midpoint = 0.5 * (start + end)
    representative = np.asarray(
        [
            float(polygon.representative_point().x),
            float(polygon.representative_point().y),
        ],
        dtype=float,
    )
    if float(np.dot(representative - midpoint, normal)) < 0.0:
        normal *= -1.0
    representative_depth = float(
        np.dot(representative - midpoint, normal)
    )
    if representative_depth <= 1.0e-12:
        center_depth = float(
            np.dot(state.points[center] - midpoint, normal)
        )
        if center_depth < 0.0:
            normal *= -1.0
            center_depth *= -1.0
        representative_depth = center_depth
    positive_targets = state.targets[
        np.asarray([*ring, center], dtype=int)
    ]
    positive_targets = positive_targets[
        np.isfinite(positive_targets) & (positive_targets > 0.0)
    ]
    local_target = (
        float(np.median(positive_targets))
        if len(positive_targets)
        else base_length
    )
    scale = max(
        base_length,
        float(np.sqrt(max(float(polygon.area), 1.0e-30))),
        1.0,
    )
    tolerance = max(1.0e-10 * scale, 1.0e-9)
    desired_depth = min(
        0.30 * base_length,
        0.65 * max(representative_depth, tolerance),
        0.50 * max(local_target, tolerance),
    )
    desired_depth = max(desired_depth, 16.0 * tolerance)
    common.update(
        {
            "base_length_m": float(base_length),
            "wet_side_unit_normal": list(map(float, normal)),
            "representative_point_xy": list(map(float, representative)),
            "representative_normal_depth_m": float(
                representative_depth
            ),
            "local_target_m": float(local_target),
            "desired_front_depth_m": float(desired_depth),
            "strict_inside_tolerance_m": float(tolerance),
        }
    )

    modes: list[dict[str, Any]] = []
    ring_points = state.points[np.asarray(ring, dtype=int)]
    for support_count in range(2, support_limit + 1):
        fractions = np.arange(
            1,
            support_count + 1,
            dtype=float,
        ) / float(support_count + 1)
        depth = float(desired_depth)
        coordinates = np.empty((0, 2), dtype=float)
        generation_failures: list[str] = []
        for _ in range(32):
            base_points = (
                (1.0 - fractions[:, None]) * start
                + fractions[:, None] * end
            )
            trial = base_points + depth * normal
            inside = np.asarray(
                contains_xy(polygon, trial[:, 0], trial[:, 1]),
                dtype=bool,
            )
            separated = bool(
                len(trial) == 0
                or np.all(
                    np.min(
                        np.linalg.norm(
                            trial[:, None, :] - ring_points[None, :, :],
                            axis=2,
                        ),
                        axis=1,
                    )
                    > tolerance
                )
            )
            if bool(np.all(inside)) and separated:
                coordinates = trial
                break
            depth *= 0.5
            if depth <= 8.0 * tolerance:
                break
        if len(coordinates) != support_count:
            generation_failures.append(
                "unable_to_place_strictly_inside_inward_front"
            )
        evidence = {
            **common,
            "support_point_count": int(support_count),
            "front_depth_m": float(depth),
            "tangential_fractions": list(map(float, fractions)),
            "support_coordinates_xy": [
                list(map(float, point)) for point in coordinates
            ],
            "strictly_inside_locked_ring": bool(
                len(coordinates) == support_count
            ),
        }
        modes.append(
            {
                "name": "inward-front-multi-support",
                "coordinates": np.asarray(
                    coordinates,
                    dtype=float,
                ).reshape((-1, 2)),
                "requested_support_node_count": int(support_count),
                "generation_failures": generation_failures,
                "support_generation_evidence": evidence,
            }
        )
    return modes


def _locked_star_modes(
    state: _State,
    center: int,
    ring: list[int],
    triangle_index: int,
    config: AggressiveConditioningConfig,
) -> list[dict[str, Any]]:
    """Return the deterministic V5 candidate ladder for one closed star."""
    triangle = list(map(int, state.triangles[int(triangle_index)]))
    other = [value for value in triangle if value != int(center)]
    if len(other) != 2:
        return []
    base = tuple(sorted(map(int, other)))
    coordinates = state.points[np.asarray(triangle, dtype=int)]
    opposite_lengths = np.asarray(
        [
            np.linalg.norm(coordinates[1] - coordinates[2]),
            np.linalg.norm(coordinates[0] - coordinates[2]),
            np.linalg.norm(coordinates[0] - coordinates[1]),
        ],
        dtype=float,
    )
    incenter = np.sum(coordinates * opposite_lengths[:, None], axis=0) / max(
        float(np.sum(opposite_lengths)),
        1.0e-12,
    )
    ring_points = state.points[np.asarray(ring, dtype=int)]
    weights = 1.0 / np.maximum(state.targets[np.asarray(ring, dtype=int)], 1.0e-12)
    weighted_centroid = np.sum(ring_points * weights[:, None], axis=0) / float(np.sum(weights))
    polygon = Polygon(ring_points)
    representative = np.asarray(
        [polygon.representative_point().x, polygon.representative_point().y],
        dtype=float,
    )
    base_vector = state.points[base[1]] - state.points[base[0]]
    denominator = float(np.dot(base_vector, base_vector))
    projection = (
        state.points[base[0]].copy()
        if denominator <= 1.0e-20
        else state.points[base[0]]
        + float(
            np.clip(
                np.dot(state.points[int(center)] - state.points[base[0]], base_vector)
                / denominator,
                0.02,
                0.98,
            )
        )
        * base_vector
    )
    nearest_ring = int(
        min(
            ring,
            key=lambda node: (
                float(np.linalg.norm(state.points[int(node)] - projection)),
                int(node),
            ),
        )
    )
    modes: list[dict[str, Any]] = []
    if float(np.linalg.norm(state.points[nearest_ring] - projection)) <= (
        float(config.systematic_v3_weld_snap_fraction)
        * max(float(state.targets[int(center)]), 1.0e-12)
    ):
        modes.append(
            {
                "name": "contract-existing",
                "target_node": nearest_ring,
            }
        )
    modes.extend([
        {"name": "center-elimination"},
        {
            "name": "center-relocation-altitude",
            "coordinate": 0.88 * projection + 0.12 * representative,
        },
        {"name": "center-relocation-incenter", "coordinate": incenter},
        {
            "name": "center-relocation-offcenter",
            "coordinate": 0.55 * incenter + 0.45 * representative,
        },
        {
            "name": "center-relocation-monitor-centroid",
            "coordinate": 0.70 * weighted_centroid + 0.30 * representative,
        },
    ])
    protected = chain_edges(state.chains)
    if (
        base in protected
        and not state.fixed[int(center)]
        and not state.hard[int(center)]
        and bool(config.systematic_v5_enable_boundary_window_fallback)
    ):
        relaxed = replace(
            config,
            boundary_weld_max_distance_fraction=max(
                2.0,
                float(config.boundary_weld_max_distance_fraction),
            ),
            boundary_weld_max_altitude_to_arc_fraction=max(
                1.0,
                float(config.boundary_weld_max_altitude_to_arc_fraction),
            ),
            boundary_weld_land_max_distance_m=max(
                1000.0,
                float(config.boundary_weld_land_max_distance_m),
            ),
            boundary_weld_open_max_distance_m=max(
                1000.0,
                float(config.boundary_weld_open_max_distance_m),
            ),
            boundary_weld_anchor_buffer_segments=0,
            boundary_weld_junction_buffer_segments=0,
            boundary_weld_channel_clearance_fraction=0.0,
            boundary_weld_forbidden_kind_tokens=(),
        )
        weld, failures = _boundary_weld_geometry(state, base, int(center), relaxed)
        if weld is not None and not failures:
            fraction, arc_projection, distance, h = weld
            snap_target = min(
                base,
                key=lambda node: (
                    float(np.linalg.norm(arc_projection - state.points[int(node)])),
                    int(node),
                ),
            )
            snap_distance = float(
                np.linalg.norm(arc_projection - state.points[int(snap_target)])
            )
            if snap_distance <= float(config.systematic_v3_weld_snap_fraction) * max(h, 1.0e-12):
                modes.insert(
                    0,
                    {
                        "name": "boundary-snap-existing",
                        "base": base,
                        "target_node": int(snap_target),
                        "coordinate": state.points[int(snap_target)].copy(),
                        "projection_fraction": float(fraction),
                        "weld_distance_m": float(distance),
                    },
                )
            else:
                modes.insert(
                    0,
                    {
                        "name": "boundary-source-arc-insertion",
                        "base": base,
                        "coordinate": np.asarray(arc_projection, dtype=float),
                        "projection_fraction": float(fraction),
                        "weld_distance_m": float(distance),
                    },
                )
    if (
        base in protected
        and not state.fixed[int(center)]
        and not state.hard[int(center)]
        and _find_chain_node(state.chains, int(center)) is None
    ):
        modes.extend(
            _locked_inward_front_support_modes(
                state,
                int(center),
                ring,
                base,
                config,
            )
        )
    modes.append({"name": "support-node-fallback", "coordinate": representative})
    return modes


def _reconstruct_locked_boundary_fan_candidate(
    state: _State,
    *,
    center: int,
    triangle_index: int,
    config: AggressiveConditioningConfig,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Rebuild every face of an open boundary-node star without moving it."""
    center = int(center)
    membership = _find_chain_node(state.chains, center)
    if membership is None:
        return False, ["open_star_center_not_on_boundary_chain"], {}
    chain_index, position = membership
    chain = state.chains[int(chain_index)]
    if len(chain) < 3:
        return False, ["boundary_chain_too_short"], {}
    previous = int(chain[(position - 1) % len(chain)])
    following = int(chain[(position + 1) % len(chain)])
    incident = np.where(np.any(state.triangles == center, axis=1))[0]
    fan = _ordered_boundary_fan(
        state.triangles[incident],
        center,
        previous,
        following,
    )
    if fan is None:
        fan = _ordered_boundary_fan(
            state.triangles[incident],
            center,
            following,
            previous,
        )
    if fan is None or len(fan) < 2:
        return False, ["non_simple_boundary_fan"], {}
    patch_ring = [center, *map(int, fan)]
    if len(set(patch_ring)) != len(patch_ring):
        return False, ["repeated_boundary_fan_ring_node"], {}
    old_fan = state.triangles[incident].copy()
    old_geometry = triangle_geometry(state.points, old_fan)
    replacement = _triangulate_ring_greedy(
        state.points,
        patch_ring,
        None,
        max(int(config.max_valence), 8),
    )
    if replacement is None:
        return False, ["deterministic_boundary_fan_ear_triangulation_failed"], {}
    replacement_geometry = triangle_geometry(state.points, replacement)
    if np.any(
        replacement_geometry["signed_area"]
        <= _area_tolerance(state.points, replacement)
    ):
        return False, ["nonpositive_boundary_fan_replacement"], {}
    old_area = float(np.sum(old_geometry["area"]))
    new_area = float(np.sum(replacement_geometry["area"]))
    if abs(new_area - old_area) > 1.0e-8 * max(old_area, 1.0):
        return False, ["boundary_fan_area_mismatch"], {}
    perimeter = {
        tuple(
            sorted(
                (
                    int(patch_ring[index]),
                    int(patch_ring[(index + 1) % len(patch_ring)]),
                )
            )
        )
        for index in range(len(patch_ring))
    }
    protected = chain_edges(state.chains)
    protected_inside = protected & _edge_set(old_fan)
    if not protected_inside.issubset(perimeter):
        return False, ["boundary_fan_contains_protected_internal_chord"], {}
    keep = np.ones(len(state.triangles), dtype=bool)
    keep[incident] = False
    outside = state.triangles[keep]
    state.triangles = _orient_ccw(state.points, np.vstack([outside, replacement]))
    replacement_ids = set(range(len(outside), len(outside) + len(replacement)))
    flip_count = _lawson_legalize_locked_patch(
        state,
        replacement_ids,
        perimeter | protected,
        max_flips=max(
            0,
            int(config.systematic_v5_max_lawson_flips_per_transaction),
        ),
    )
    delivered_edges = _edge_set(state.triangles)
    if not perimeter.issubset(delivered_edges):
        return False, ["locked_boundary_fan_perimeter_changed"], {}
    if not protected.issubset(delivered_edges):
        return False, ["protected_edge_missing"], {}
    state.last_affected = sorted(set(map(int, patch_ring)))
    delivered_patch_geometry = triangle_geometry(
        state.points,
        state.triangles[np.asarray(sorted(replacement_ids), dtype=int)],
    )
    evidence = {
        "candidate": "boundary-fan-ear-retriangulation",
        "center_original_lineage": int(state.lineage[center]),
        "center_is_hard_anchor": bool(state.hard[center]),
        "source_triangle_index": int(triangle_index),
        "old_star_triangle_count": int(len(incident)),
        "replacement_triangle_count": int(len(replacement)),
        "patch_ring_node_count": int(len(patch_ring)),
        "locked_ring_original_lineage": [
            int(state.lineage[node]) for node in patch_ring
        ],
        "lawson_flip_count": int(flip_count),
        "inserted_support_node_count": 0,
        "old_star_area_m2": float(old_area),
        "new_star_area_m2": float(new_area),
        "local_debt_before": _geometry_superthin_debt(old_geometry, config),
        "local_debt_after": _geometry_superthin_debt(delivered_patch_geometry, config),
    }
    state.ledger.append(
        {
            "operation": "systematic-v5-complete-locked-boundary-fan-reconstruction",
            **evidence,
        }
    )
    return True, [], evidence


def _reconstruct_locked_star_candidate(
    state: _State,
    *,
    center: int,
    triangle_index: int,
    mode: dict[str, Any],
    config: AggressiveConditioningConfig,
) -> tuple[bool, list[str], dict[str, Any]]:
    center = int(center)
    incident = np.where(np.any(state.triangles == center, axis=1))[0]
    ring = _ordered_one_ring(state.triangles[incident], center)
    if ring is None:
        return False, ["open_or_nonmanifold_node_star"], {}
    if state.hard[center] or state.fixed[center] or _find_chain_node(state.chains, center) is not None:
        if str(mode["name"]) not in {
            "center-relocation-altitude",
            "center-relocation-incenter",
            "center-relocation-offcenter",
            "center-relocation-monitor-centroid",
        }:
            return False, ["center_fixed_or_hard"], {}
        return False, ["fixed_center_relocation_forbidden"], {}
    old_fan = state.triangles[incident].copy()
    old_geometry = triangle_geometry(state.points, old_fan)
    old_area = float(np.sum(old_geometry["area"]))
    old_coordinate = state.points[center].copy()
    protected_before = chain_edges(state.chains)
    old_fan_edges = _edge_set(old_fan)
    internal_protected = {
        edge
        for edge in protected_before & old_fan_edges
        if center in edge
    }
    if internal_protected:
        return False, ["center_has_protected_incident_chord"], {}
    base_replacement = _triangulate_ring_greedy(
        state.points,
        ring,
        None,
        max(int(config.max_valence), 8),
        removed_node=center,
    )
    if base_replacement is None:
        return False, ["deterministic_ear_triangulation_failed"], {}
    name = str(mode["name"])
    inserted_support = 0
    support_coordinates = np.empty((0, 2), dtype=float)
    support_front_evidence: dict[str, Any] = {}
    boundary_inserted = False
    replacement: np.ndarray
    if name.startswith("center-relocation"):
        coordinate = np.asarray(mode["coordinate"], dtype=float)
        polygon = Polygon(state.points[np.asarray(ring, dtype=int)])
        if not bool(contains_xy(polygon, np.asarray([coordinate[0]]), np.asarray([coordinate[1]]))[0]):
            return False, ["relocated_center_outside_locked_ring"], {}
        state.points[center] = coordinate
        replacement = _orient_ccw(
            state.points,
            np.asarray(
                [
                    [center, int(ring[index]), int(ring[(index + 1) % len(ring)])]
                    for index in range(len(ring))
                ],
                dtype=int,
            ),
        )
    elif name == "boundary-source-arc-insertion":
        base = tuple(sorted(map(int, mode["base"])))
        chain_position = _find_chain_edge(state.chains, base)
        if chain_position is None:
            return False, ["constraint_chain_edge_not_found"], {}
        matching = [
            index
            for index, triangle in enumerate(base_replacement)
            if set(base).issubset(set(map(int, triangle)))
        ]
        if len(matching) != 1:
            return False, ["boundary_base_not_unique_in_ring_replacement"], {}
        split_index = int(matching[0])
        third = [value for value in base_replacement[split_index] if int(value) not in base]
        if len(third) != 1:
            return False, ["boundary_split_opposite_not_unique"], {}
        state.points[center] = np.asarray(mode["coordinate"], dtype=float)
        state.fixed[center] = True
        state.hard[center] = False
        state.kinds[center] = state.kinds[base[0]]
        state.targets[center] = _sample_target_at(
            state,
            state.points[center],
            fallback=_edge_target(state.targets, base),
        )
        replacement = np.vstack(
            [
                np.delete(base_replacement, split_index, axis=0),
                np.asarray(
                    [
                        [base[0], center, int(third[0])],
                        [center, base[1], int(third[0])],
                    ],
                    dtype=int,
                ),
            ]
        )
        chain_index, position = chain_position
        chain = state.chains[int(chain_index)]
        left = int(chain[position])
        right = int(chain[(position + 1) % len(chain)])
        insert_position = int(position + 1)
        chain.insert(insert_position, center)
        open_values = state.open_nodes.tolist()
        for open_position, (open_left, open_right) in enumerate(
            zip(open_values[:-1], open_values[1:])
        ):
            if int(open_left) == left and int(open_right) == right:
                open_values.insert(open_position + 1, center)
                state.open_nodes = np.asarray(open_values, dtype=int)
                break
            if int(open_left) == right and int(open_right) == left:
                open_values.insert(open_position + 1, center)
                state.open_nodes = np.asarray(open_values, dtype=int)
                break
        boundary_inserted = True
    elif name == "inward-front-multi-support":
        support_front_evidence = dict(
            mode.get("support_generation_evidence", {})
        )
        generation_failures = list(
            map(str, mode.get("generation_failures", []))
        )
        try:
            support_coordinates = np.asarray(
                mode.get(
                    "coordinates",
                    np.empty((0, 2), dtype=float),
                ),
                dtype=float,
            ).reshape((-1, 2))
        except ValueError:
            support_coordinates = np.empty((0, 2), dtype=float)
            generation_failures.append(
                "invalid_inward_front_support_coordinate_shape"
            )
        requested_count = int(
            mode.get(
                "requested_support_node_count",
                len(support_coordinates),
            )
        )
        support_cap = min(
            8,
            max(
                0,
                int(
                    config.systematic_v5_max_inward_front_support_points
                ),
            ),
        )
        failure_evidence = {
            "candidate": name,
            "requested_support_node_count": int(requested_count),
            "support_coordinate_count": int(len(support_coordinates)),
            "support_coordinates_xy": [
                list(map(float, point))
                for point in support_coordinates
            ],
            "bounded_max_support_point_count": int(support_cap),
            "support_generation_evidence": support_front_evidence,
            "triangulation_method": (
                "deterministic-ear-plus-point-insertion"
            ),
            "global_delaunay_used": False,
        }
        if generation_failures:
            return False, sorted(set(generation_failures)), failure_evidence
        if requested_count != len(support_coordinates):
            return False, [
                "inward_front_support_count_mismatch"
            ], failure_evidence
        if not 2 <= len(support_coordinates) <= support_cap:
            return False, [
                "inward_front_support_count_out_of_bounds"
            ], failure_evidence
        if not np.all(np.isfinite(support_coordinates)):
            return False, [
                "nonfinite_inward_front_support_coordinate"
            ], failure_evidence
        polygon = Polygon(state.points[np.asarray(ring, dtype=int)])
        if not polygon.is_valid or polygon.area <= 0.0:
            return False, [
                "invalid_locked_ring_for_inward_front_support"
            ], failure_evidence
        strictly_inside = np.asarray(
            contains_xy(
                polygon,
                support_coordinates[:, 0],
                support_coordinates[:, 1],
            ),
            dtype=bool,
        )
        if not bool(np.all(strictly_inside)):
            failure_evidence[
                "strictly_inside_locked_ring"
            ] = list(map(bool, strictly_inside))
            return False, [
                "inward_front_support_outside_locked_ring"
            ], failure_evidence
        span = np.ptp(
            state.points[np.asarray(ring, dtype=int)],
            axis=0,
        )
        coordinate_tolerance = max(
            1.0e-10 * max(float(np.max(span)), 1.0),
            1.0e-9,
        )
        pair_distances = np.linalg.norm(
            support_coordinates[:, None, :]
            - support_coordinates[None, :, :],
            axis=2,
        )
        np.fill_diagonal(pair_distances, np.inf)
        if float(np.min(pair_distances)) <= coordinate_tolerance:
            return False, [
                "duplicate_inward_front_support_coordinate"
            ], failure_evidence
        support_start = len(state.points)
        support_ids = np.arange(
            support_start,
            support_start + len(support_coordinates),
            dtype=int,
        )
        fallback_target = float(
            np.median(state.targets[np.asarray(ring, dtype=int)])
        )
        support_targets = np.asarray(
            [
                _sample_target_at(
                    state,
                    point,
                    fallback=fallback_target,
                )
                for point in support_coordinates
            ],
            dtype=float,
        )
        state.points = np.vstack(
            [state.points, support_coordinates]
        )
        state.fixed = np.concatenate(
            [
                state.fixed,
                np.zeros(len(support_coordinates), dtype=bool),
            ]
        )
        state.targets = np.concatenate(
            [state.targets, support_targets]
        )
        state.kinds.extend(
            ["interior"] * len(support_coordinates)
        )
        state.hard = np.concatenate(
            [
                state.hard,
                np.zeros(len(support_coordinates), dtype=bool),
            ]
        )
        state.lineage = np.concatenate(
            [
                state.lineage,
                _new_lineage_ids(
                    state,
                    len(support_coordinates),
                ),
            ]
        )
        allowed_policy = _allowed_edge_policy(state, config)
        replacement, insertion_failures = (
            _insert_existing_patch_nodes_v1(
                state.points,
                base_replacement,
                support_ids,
                edge_allowed=lambda edge: allowed_policy.is_allowed(
                    edge,
                    reject_same_chain_shortcuts=False,
                ),
            )
        )
        if insertion_failures:
            failure_evidence["point_insertion_failures"] = list(
                map(str, insertion_failures)
            )
            return False, [
                f"inward_front_{failure}"
                for failure in insertion_failures
            ], failure_evidence
        disallowed_edges = [
            list(map(int, edge))
            for edge in sorted(_edge_set(replacement))
            if not allowed_policy.is_allowed(
                edge,
                reject_same_chain_shortcuts=False,
            )
        ]
        if disallowed_edges:
            failure_evidence[
                "disallowed_replacement_edges"
            ] = disallowed_edges
            return False, [
                "inward_front_support_creates_restricted_edge"
            ], failure_evidence
        used_support = set(map(int, np.unique(replacement))) & set(
            map(int, support_ids)
        )
        if used_support != set(map(int, support_ids)):
            failure_evidence[
                "unused_support_node_indices_zero_based"
            ] = sorted(
                set(map(int, support_ids)) - used_support
            )
            return False, [
                "inward_front_support_node_unused"
            ], failure_evidence
        inserted_support = int(len(support_coordinates))
    elif name == "support-node-fallback":
        coordinate = np.asarray(mode["coordinate"], dtype=float)
        polygon = Polygon(state.points[np.asarray(ring, dtype=int)])
        if not bool(contains_xy(polygon, np.asarray([coordinate[0]]), np.asarray([coordinate[1]]))[0]):
            return False, ["support_node_outside_locked_ring"], {}
        support = len(state.points)
        state.points = np.vstack([state.points, coordinate])
        state.fixed = np.concatenate([state.fixed, np.zeros(1, dtype=bool)])
        state.targets = np.concatenate(
            [
                state.targets,
                np.asarray(
                    [
                        _sample_target_at(
                            state,
                            coordinate,
                            fallback=float(np.median(state.targets[np.asarray(ring, dtype=int)])),
                        )
                    ],
                    dtype=float,
                ),
            ]
        )
        state.kinds.append("interior")
        state.hard = np.concatenate([state.hard, np.zeros(1, dtype=bool)])
        state.lineage = np.concatenate([state.lineage, _new_lineage_ids(state, 1)])
        replacement = _orient_ccw(
            state.points,
            np.asarray(
                [
                    [support, int(ring[index]), int(ring[(index + 1) % len(ring)])]
                    for index in range(len(ring))
                ],
                dtype=int,
            ),
        )
        inserted_support = 1
    else:
        # Both contraction to an existing ring node and explicit center
        # elimination produce the same locked-ring triangulation.  Their
        # distinct labels retain the causal decision in the transaction log.
        replacement = base_replacement
    replacement = _orient_ccw(state.points, replacement)
    replacement_geometry = triangle_geometry(state.points, replacement)
    if np.any(
        replacement_geometry["signed_area"]
        <= _area_tolerance(state.points, replacement)
    ):
        return False, ["nonpositive_locked_star_replacement"], {}
    new_area = float(np.sum(replacement_geometry["area"]))
    if abs(new_area - old_area) > 1.0e-8 * max(old_area, 1.0):
        return False, ["locked_star_area_mismatch"], {
            "old_star_area_m2": old_area,
            "new_star_area_m2": new_area,
        }
    keep = np.ones(len(state.triangles), dtype=bool)
    keep[incident] = False
    outside = state.triangles[keep]
    state.triangles = _orient_ccw(
        state.points,
        np.vstack([outside, replacement]),
    )
    replacement_ids = set(
        range(len(outside), len(outside) + len(replacement))
    )
    ring_edges = {
        tuple(sorted((int(ring[index]), int(ring[(index + 1) % len(ring)]))))
        for index in range(len(ring))
    }
    if boundary_inserted:
        base = tuple(sorted(map(int, mode["base"])))
        ring_edges.discard(base)
        ring_edges.update(
            {
                tuple(sorted((base[0], center))),
                tuple(sorted((center, base[1]))),
            }
        )
    protected_after = chain_edges(state.chains)
    locked_edges = ring_edges | protected_after
    flip_count = _lawson_legalize_locked_patch(
        state,
        replacement_ids,
        locked_edges,
        max_flips=max(
            0,
            int(config.systematic_v5_max_lawson_flips_per_transaction),
        ),
    )
    delivered_edges = _edge_set(state.triangles)
    if not ring_edges.issubset(delivered_edges):
        return False, ["locked_patch_perimeter_changed"], {}
    if not protected_after.issubset(delivered_edges):
        return False, ["protected_edge_missing"], {}
    canonical = [tuple(sorted(map(int, tri))) for tri in state.triangles]
    if len(canonical) != len(set(canonical)):
        return False, ["duplicate_triangle_after_star_reconstruction"], {}
    state.last_affected = sorted(
        set(map(int, np.unique(replacement)))
        | set(map(int, ring))
    )
    delivered_patch_geometry = triangle_geometry(
        state.points,
        state.triangles[np.asarray(sorted(replacement_ids), dtype=int)],
    )
    evidence = {
        "candidate": name,
        "center_original_lineage": int(state.lineage[center]),
        "source_triangle_index": int(triangle_index),
        "old_star_triangle_count": int(len(incident)),
        "replacement_triangle_count": int(len(replacement)),
        "locked_ring_original_lineage": [
            int(state.lineage[node]) for node in ring
        ],
        "protected_chord_count": int(
            len(protected_before & old_fan_edges)
        ),
        "lawson_flip_count": int(flip_count),
        "inserted_support_node_count": int(inserted_support),
        "support_coordinates_xy": [
            list(map(float, point))
            for point in support_coordinates
        ],
        "support_generation_evidence": support_front_evidence,
        "support_triangulation_method": (
            "deterministic-ear-plus-point-insertion"
            if name == "inward-front-multi-support"
            else None
        ),
        "global_delaunay_used": False,
        "boundary_node_inserted": bool(boundary_inserted),
        "old_coordinate_xy": list(map(float, old_coordinate)),
        "new_coordinate_xy": (
            list(map(float, state.points[center]))
            if center < len(state.points)
            else None
        ),
        "old_star_area_m2": float(old_area),
        "new_star_area_m2": float(new_area),
        "local_debt_before": _geometry_superthin_debt(old_geometry, config),
        "local_debt_after": _geometry_superthin_debt(delivered_patch_geometry, config),
        "intermediate_degenerate_triangles_removed": int(
            1 if name.startswith("boundary-") else 0
        ),
    }
    state.ledger.append(
        {
            "operation": "systematic-v5-complete-locked-star-reconstruction",
            **evidence,
        }
    )
    _compact(state)
    return True, [], evidence


def _try_expanded_locked_patches_v5(
    state: _State,
    component: dict[str, Any],
    config: AggressiveConditioningConfig,
    audit_config: AggressiveConditioningConfig,
    initial_components: int,
    before_summary: dict[str, Any],
    passage_baseline: dict[tuple[int, int], float],
) -> tuple[
    tuple[tuple[float, ...], _State, dict[str, Any]] | None,
    list[dict[str, Any]],
]:
    """Expand a failed causal star through the configured 1→2→4 ring ladder."""
    attempts: list[dict[str, Any]] = []
    best: tuple[tuple[float, ...], _State, dict[str, Any]] | None = None
    topology = build_edge_topology(len(state.points), state.triangles)
    for rings in tuple(int(value) for value in config.systematic_v5_patch_ring_ladder):
        if rings <= 1 or _deadline_reached(config):
            continue
        patch = _expand_triangle_patch(
            state.triangles,
            topology,
            component["triangle_indices"],
            rings,
        )
        ring = _ordered_patch_boundary(state.triangles, patch)
        record: dict[str, Any] = {
            "component_id": component["component_id"],
            "classification": component["classification"],
            "candidate": "expanded-ring-elimination",
            "patch_rings": int(rings),
            "patch_triangle_count": int(len(patch)),
            "accepted": False,
        }
        if ring is None:
            record["failures"] = ["non_simple_patch_boundary"]
            attempts.append(record)
            continue
        trial = state.clone()
        changed, failures, evidence = _reconstruct_expanded_ring_v5(
            trial,
            patch,
            ring,
            config,
        )
        if not changed:
            record["failures"] = failures or ["expanded_patch_construction_failed"]
            attempts.append(record)
            continue
        _micro_relax(
            trial,
            replacement_seed_nodes=trial.last_affected,
            config=replace(
                config,
                micro_relax_cycles=min(1, int(config.micro_relax_cycles)),
                micro_relax_iterations=min(2, int(config.micro_relax_iterations)),
                micro_relax_ring_layers=2,
            ),
        )
        ok, invariant_report, trial_summary = _audit_state(
            trial,
            audit_config,
            initial_components,
        )
        hard_failures = _v5_hard_gate_failures(
            state,
            trial,
            config,
            passage_baseline,
        )
        all_failures = [] if ok else _failed_invariant_names(invariant_report)
        all_failures = sorted(set([*all_failures, *hard_failures]))
        debt_before = (
            int(before_summary["superthin_triangle_count"]),
            float(before_summary["superthin_severity_sum"]),
        )
        debt_after = (
            int(trial_summary["superthin_triangle_count"]),
            float(trial_summary["superthin_severity_sum"]),
        )
        if not debt_after < debt_before:
            all_failures = sorted(set([*all_failures, "superthin_debt_not_reduced"]))
        record.update(
            {
                "evidence": evidence,
                "trial": _summary_from(trial_summary),
                "invariants": invariant_report,
                "failures": all_failures,
            }
        )
        attempts.append(record)
        if all_failures:
            continue
        score = _v5_candidate_score(
            trial_summary,
            evidence,
            {"name": "expanded-ring-elimination"},
        )
        record["candidate_score"] = list(score)
        if best is None or score < best[0]:
            best = (score, trial, record)
    return best, attempts


def _reconstruct_expanded_ring_v5(
    state: _State,
    patch: np.ndarray,
    ring: list[int],
    config: AggressiveConditioningConfig,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Deterministically eliminate a movable expanded cavity and ear-triangulate it."""
    patch = np.asarray(patch, dtype=int)
    patch_nodes = set(map(int, np.unique(state.triangles[patch])))
    interior = sorted(patch_nodes - set(map(int, ring)))
    protected = chain_edges(state.chains)
    protected_inside = protected & _edge_set(state.triangles[patch])
    perimeter = {
        tuple(sorted((int(ring[index]), int(ring[(index + 1) % len(ring)]))))
        for index in range(len(ring))
    }
    if any(
        state.fixed[node]
        or state.hard[node]
        or _find_chain_node(state.chains, int(node)) is not None
        for node in interior
    ):
        return False, ["expanded_patch_contains_immutable_interior_node"], {}
    if not protected_inside.issubset(perimeter):
        return False, ["expanded_patch_contains_protected_chord"], {}
    replacement = _triangulate_ring_greedy(
        state.points,
        ring,
        None,
        max(int(config.max_valence), 8),
    )
    if replacement is None:
        return False, ["deterministic_ear_triangulation_failed"], {}
    old_geometry = triangle_geometry(state.points, state.triangles[patch])
    replacement_geometry = triangle_geometry(state.points, replacement)
    old_area = float(np.sum(old_geometry["area"]))
    new_area = float(np.sum(replacement_geometry["area"]))
    if abs(new_area - old_area) > 1.0e-8 * max(old_area, 1.0):
        return False, ["expanded_patch_area_mismatch"], {}
    keep = np.ones(len(state.triangles), dtype=bool)
    keep[patch] = False
    outside = state.triangles[keep]
    state.triangles = _orient_ccw(state.points, np.vstack([outside, replacement]))
    replacement_ids = set(range(len(outside), len(outside) + len(replacement)))
    flip_count = _lawson_legalize_locked_patch(
        state,
        replacement_ids,
        perimeter | protected,
        max_flips=max(
            0,
            int(config.systematic_v5_max_lawson_flips_per_transaction),
        ),
    )
    state.last_affected = sorted(set(map(int, ring)))
    delivered_patch_geometry = triangle_geometry(
        state.points,
        state.triangles[np.asarray(sorted(replacement_ids), dtype=int)],
    )
    evidence = {
        "candidate": "expanded-ring-elimination",
        "patch_triangle_count": int(len(patch)),
        "replacement_triangle_count": int(len(replacement)),
        "removed_movable_node_count": int(len(interior)),
        "locked_ring_original_lineage": [int(state.lineage[node]) for node in ring],
        "lawson_flip_count": int(flip_count),
        "old_patch_area_m2": float(old_area),
        "new_patch_area_m2": float(new_area),
        "inserted_support_node_count": 0,
        "local_debt_before": _geometry_superthin_debt(old_geometry, config),
        "local_debt_after": _geometry_superthin_debt(delivered_patch_geometry, config),
    }
    state.ledger.append(
        {
            "operation": "systematic-v5-expanded-locked-patch-reconstruction",
            **evidence,
        }
    )
    _compact(state)
    return True, [], evidence


def _lawson_legalize_locked_patch(
    state: _State,
    patch_triangle_ids: set[int],
    locked_edges: set[tuple[int, int]],
    *,
    max_flips: int,
    edge_allowed: Callable[[tuple[int, int]], bool] | None = None,
) -> int:
    """Apply deterministic protected-edge-safe Lawson flips inside one patch."""
    flips = 0
    patch_triangle_ids = {
        int(value)
        for value in patch_triangle_ids
        if 0 <= int(value) < len(state.triangles)
    }
    outside_ids = [
        index
        for index in range(len(state.triangles))
        if int(index) not in patch_triangle_ids
    ]
    outside_edges = (
        _edge_set(state.triangles[np.asarray(outside_ids, dtype=int)])
        if outside_ids
        else set()
    )
    for _ in range(max(0, int(max_flips))):
        local_edges: dict[tuple[int, int], list[int]] = {}
        for triangle_id in sorted(patch_triangle_ids):
            for edge in _triangle_edge_keys(state.triangles[int(triangle_id)]):
                local_edges.setdefault(edge, []).append(int(triangle_id))
        selected: tuple[int, int, np.ndarray] | None = None
        for edge in sorted(local_edges):
            attached = list(map(int, local_edges[edge]))
            if edge in locked_edges or len(attached) != 2:
                continue
            first, second = attached
            c_values = [
                int(value)
                for value in state.triangles[first]
                if int(value) not in edge
            ]
            d_values = [
                int(value)
                for value in state.triangles[second]
                if int(value) not in edge
            ]
            if len(c_values) != 1 or len(d_values) != 1 or c_values[0] == d_values[0]:
                continue
            c, d = c_values[0], d_values[0]
            new_edge = tuple(sorted((c, d)))
            if (
                new_edge in local_edges
                or new_edge in outside_edges
                or new_edge in locked_edges
                or (
                    edge_allowed is not None
                    and not bool(edge_allowed(new_edge))
                )
            ):
                continue
            if not _delaunay_edge_is_illegal(
                state.points,
                int(edge[0]),
                int(edge[1]),
                c,
                d,
            ):
                continue
            replacement = _orient_ccw(
                state.points,
                np.asarray(
                    [[c, d, int(edge[0])], [d, c, int(edge[1])]],
                    dtype=int,
                ),
            )
            geometry = triangle_geometry(state.points, replacement)
            if np.any(
                geometry["signed_area"]
                <= _area_tolerance(state.points, replacement)
            ):
                continue
            selected = (first, second, replacement)
            break
        if selected is None:
            break
        first, second, replacement = selected
        state.triangles[first] = replacement[0]
        state.triangles[second] = replacement[1]
        flips += 1
    return int(flips)


def _delaunay_edge_is_illegal(
    points: np.ndarray,
    a: int,
    b: int,
    c: int,
    d: int,
) -> bool:
    """Return True when d lies strictly inside circumcircle(a,b,c)."""
    pa = np.asarray(points[int(a)], dtype=float)
    pb = np.asarray(points[int(b)], dtype=float)
    pc = np.asarray(points[int(c)], dtype=float)
    pd = np.asarray(points[int(d)], dtype=float)
    ab = pb - pa
    ac = pc - pa
    orient = float(ab[0] * ac[1] - ab[1] * ac[0])
    if abs(orient) <= 1.0e-20:
        return False
    shifted = np.vstack([pa - pd, pb - pd, pc - pd])
    squared = np.sum(shifted * shifted, axis=1)
    determinant = float(
        np.linalg.det(
            np.column_stack([shifted[:, 0], shifted[:, 1], squared])
        )
    )
    signed = determinant if orient > 0.0 else -determinant
    scale = max(
        float(np.max(np.linalg.norm(shifted, axis=1))) ** 4,
        1.0,
    )
    return bool(signed > 1.0e-13 * scale)


def _v5_hard_gate_failures(
    before_state: _State,
    trial: _State,
    config: AggressiveConditioningConfig,
    passage_baseline: dict[tuple[int, int], float],
) -> list[str]:
    failures: list[str] = []
    if not np.all(np.isfinite(trial.points)):
        failures.append("nonfinite_coordinates")
    boundary_changed = _v5_boundary_changed(before_state, trial)
    open_boundary_changed = (
        _open_boundary_lineage_sequence(before_state)
        != _open_boundary_lineage_sequence(trial)
    )
    if not bool(config.systematic_v5_enable_boundary_window_fallback):
        if boundary_changed:
            failures.append("topology_only_boundary_changed")
        if open_boundary_changed:
            failures.append("topology_only_open_boundary_changed")
    if boundary_changed:
        if not _boundary_loops_simple(trial):
            failures.append("boundary_loop_invalid_or_self_intersecting")
        if _maximum_boundary_source_arc_deviation(trial) > 1.0e-7:
            failures.append("boundary_off_source_arc")
    area_change = abs(
        _signed_mesh_area(trial.points, trial.triangles)
        - _signed_mesh_area(before_state.points, before_state.triangles)
    )
    if area_change > float(config.maximum_domain_area_change_fraction) * max(
        float(trial.initial_domain_area_m2),
        1.0e-30,
    ):
        failures.append("domain_area_budget")
    if boundary_changed:
        clearance_tolerance = float(config.systematic_v3_passage_clearance_tolerance_m)
        for pair, baseline in passage_baseline.items():
            delivered = _chain_pair_clearance(trial, *pair)
            if not np.isfinite(delivered):
                failures.append("passage_bank_identity")
                continue
            if delivered + clearance_tolerance < float(baseline):
                failures.append("passage_clearance_loss")
    return sorted(set(failures))


def _v5_boundary_changed(before_state: _State, trial: _State) -> bool:
    """Compare boundary lineage/order/coordinates without assuming stable indices."""
    def signature(state: _State) -> tuple[tuple[tuple[int, ...], ...], dict[int, tuple[float, float]]]:
        chains = tuple(
            tuple(int(state.lineage[int(node)]) for node in chain)
            for chain in state.chains
        )
        coordinates = {
            int(state.lineage[int(node)]): (
                float(state.points[int(node), 0]),
                float(state.points[int(node), 1]),
            )
            for chain in state.chains
            for node in chain
        }
        return chains, coordinates

    return bool(signature(before_state) != signature(trial))


def _open_boundary_lineage_sequence(
    state: _State,
) -> tuple[int, ...]:
    return tuple(
        int(state.lineage[int(node)])
        for node in np.asarray(state.open_nodes, dtype=int)
    )


def _geometry_superthin_debt(
    geometry: dict[str, np.ndarray],
    config: AggressiveConditioningConfig,
) -> dict[str, Any]:
    quality = np.asarray(geometry["quality"], dtype=float)
    minimum_angles = np.min(np.asarray(geometry["angles_deg"], dtype=float), axis=1)
    superthin = (
        (quality < float(config.superthin_quality_threshold))
        | (minimum_angles < float(config.superthin_min_angle_deg))
    )
    severity = float(
        np.sum(
            np.maximum(
                0.0,
                (
                    float(config.superthin_quality_threshold) - quality
                )
                / max(float(config.superthin_quality_threshold), 1.0e-12),
            )
            ** 2
        )
        + np.sum(
            np.maximum(
                0.0,
                (
                    float(config.superthin_min_angle_deg) - minimum_angles
                )
                / max(float(config.superthin_min_angle_deg), 1.0e-12),
            )
            ** 2
        )
    )
    return {
        "superthin_triangle_count": int(np.count_nonzero(superthin)),
        "superthin_severity_sum": severity,
        "q_min": float(np.min(quality)) if len(quality) else 0.0,
        "minimum_angle_deg": float(np.min(minimum_angles)) if len(minimum_angles) else 0.0,
    }


def _v5_quick_metrics_from_evidence(
    baseline: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Rank a star from its changed faces; all other global debt is constant."""
    before = evidence.get("local_debt_before", {})
    after = evidence.get("local_debt_after", {})
    return {
        "superthin_triangle_count": int(
            int(baseline["superthin_triangle_count"])
            - int(before.get("superthin_triangle_count", 0))
            + int(after.get("superthin_triangle_count", 0))
        ),
        "superthin_severity_sum": float(
            float(baseline["superthin_severity_sum"])
            - float(before.get("superthin_severity_sum", 0.0))
            + float(after.get("superthin_severity_sum", 0.0))
        ),
        "q_min": float(after.get("q_min", baseline["q_min"])),
        "minimum_angle_deg": float(
            after.get("minimum_angle_deg", baseline["minimum_angle_deg"])
        ),
    }


def _v5_quick_candidate_score(
    summary: dict[str, Any],
    evidence: dict[str, Any],
    mode: dict[str, Any],
) -> tuple[float, ...]:
    return (
        float(summary["superthin_triangle_count"]),
        float(summary["superthin_severity_sum"]),
        -float(summary["q_min"]),
        -float(summary["minimum_angle_deg"]),
        float(evidence.get("inserted_support_node_count", 0)),
        float(_v5_mode_order(str(mode.get("name")))),
    )


def _v5_mode_order(name: str) -> int:
    return {
        "boundary-snap-existing": 0,
        "boundary-source-arc-insertion": 1,
        "boundary-fan-ear-retriangulation": 2,
        "contract-existing": 3,
        "center-elimination": 4,
        "center-relocation-altitude": 5,
        "center-relocation-incenter": 6,
        "center-relocation-offcenter": 7,
        "center-relocation-monitor-centroid": 8,
        "expanded-ring-elimination": 9,
        "inward-front-multi-support": 10,
        "support-node-fallback": 11,
    }.get(str(name), 99)


def _v5_candidate_score(
    summary: dict[str, Any],
    evidence: dict[str, Any],
    mode: dict[str, Any],
) -> tuple[float, ...]:
    return (
        float(summary["superthin_triangle_count"]),
        float(summary["superthin_severity_sum"]),
        float(summary["singly_connected_triangle_count"]),
        float(summary["boundary_degree_anomaly_count"]),
        -float(summary["q_min"]),
        -float(summary["minimum_angle_deg"]),
        float(summary["valence_excess_sum"]),
        float(summary["l_over_h_count_above_1_55"]),
        float(summary["area_transition_count_above_0_50"]),
        float(evidence.get("inserted_support_node_count", 0)),
        float(_v5_mode_order(str(mode.get("name")))),
    )


def _select_interior_skeleton_collapse(
    state: _State,
    config: AggressiveConditioningConfig,
    excluded: set[tuple[int, int, int]],
) -> tuple[tuple[tuple[int, int], int, int] | None, list[dict[str, Any]]]:
    """Select an interior sliver apex for contraction to its altitude base."""
    topology = build_edge_topology(len(state.points), state.triangles)
    geometry = triangle_geometry(state.points, state.triangles)
    minimum_angles = np.min(geometry["angles_deg"], axis=1)
    superthin = (geometry["quality"] < float(config.superthin_quality_threshold)) | (
        minimum_angles < float(config.superthin_min_angle_deg)
    )
    protected = chain_edges(state.chains)
    proposals: list[tuple[float, float, float, tuple[int, int], int, int]] = []
    screened: list[dict[str, Any]] = []
    for triangle_index in np.where(superthin)[0]:
        triangle = list(map(int, state.triangles[int(triangle_index)]))
        altitude_cases: list[tuple[float, tuple[int, int], int]] = []
        for apex in triangle:
            base_values = [value for value in triangle if value != apex]
            edge = tuple(sorted(map(int, base_values)))
            length = float(np.linalg.norm(state.points[edge[1]] - state.points[edge[0]]))
            altitude_cases.append((length, edge, int(apex)))
        _, edge, apex = max(altitude_cases, key=lambda item: (item[0], item[1], -item[2]))
        key = (int(edge[0]), int(edge[1]), int(apex))
        failures: list[str] = []
        if key in excluded:
            continue
        if edge in protected:
            failures.append("skeleton_base_is_protected_boundary")
        attached = list(map(int, topology.edge_to_triangles.get(edge, [])))
        if len(attached) != 2 or int(triangle_index) not in attached:
            failures.append("skeleton_base_not_internal_two_sided_edge")
        if bool(state.fixed[int(apex)]) or bool(state.hard[int(apex)]):
            failures.append("skeleton_apex_fixed_or_hard")
        if _find_chain_node(state.chains, int(apex)) is not None:
            failures.append("skeleton_apex_is_boundary_node")
        base_vector = state.points[edge[1]] - state.points[edge[0]]
        denominator = float(np.dot(base_vector, base_vector))
        if denominator <= 1.0e-20:
            failures.append("degenerate_skeleton_base")
            fraction = 0.0
            distance = float("inf")
        else:
            fraction = float(
                np.dot(state.points[int(apex)] - state.points[edge[0]], base_vector)
                / denominator
            )
            projection = state.points[edge[0]] + fraction * base_vector
            distance = float(np.linalg.norm(state.points[int(apex)] - projection))
            if not 0.02 < fraction < 0.98:
                failures.append("skeleton_projection_outside_edge_interior")
        if failures:
            screened.append(
                {
                    "operation": "systematic-v4-interior-skeleton-collapse",
                    "payload": [int(edge[0]), int(edge[1]), int(apex)],
                    "triangle_index": int(triangle_index),
                    "failures": sorted(set(failures)),
                }
            )
            continue
        proposals.append(
            (
                float(geometry["quality"][int(triangle_index)]),
                float(minimum_angles[int(triangle_index)]),
                float(distance),
                edge,
                int(apex),
                int(triangle_index),
            )
        )
    if not proposals:
        return None, screened
    _, _, _, edge, apex, triangle_index = min(
        proposals,
        key=lambda item: (item[0], item[1], item[2], item[5], item[3], item[4]),
    )
    return (edge, int(apex), int(triangle_index)), screened


def _contract_apex_to_internal_edge(
    state: _State,
    edge: tuple[int, int],
    apex: int,
    triangle_index: int,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Contract a sliver apex to an internal edge and retain a two-sided line."""
    a, b = map(int, edge)
    apex = int(apex)
    topology = build_edge_topology(len(state.points), state.triangles)
    attached = list(map(int, topology.edge_to_triangles.get(tuple(sorted(edge)), [])))
    if len(attached) != 2 or int(triangle_index) not in attached:
        return False, ["skeleton_base_not_internal_two_sided_edge"], {}
    source_triangle = set(map(int, state.triangles[int(triangle_index)]))
    if source_triangle != {a, b, apex}:
        return False, ["skeleton_source_triangle_mismatch"], {}
    if state.fixed[apex] or state.hard[apex] or _find_chain_node(state.chains, apex) is not None:
        return False, ["skeleton_apex_fixed_or_hard"], {}
    base_vector = state.points[b] - state.points[a]
    denominator = float(np.dot(base_vector, base_vector))
    if denominator <= 1.0e-20:
        return False, ["degenerate_skeleton_base"], {}
    fraction = float(np.dot(state.points[apex] - state.points[a], base_vector) / denominator)
    if not 0.02 < fraction < 0.98:
        return False, ["skeleton_projection_outside_edge_interior"], {}
    projection = state.points[a] + fraction * base_vector
    original_coordinate = state.points[apex].copy()
    affected = set(map(int, topology.node_neighbors[apex]))
    affected.update(map(int, topology.node_neighbors[a]))
    affected.update(map(int, topology.node_neighbors[b]))
    affected.update((a, b, apex))
    state.points[apex] = projection

    ordinary: list[np.ndarray] = []
    split: list[np.ndarray] = []
    for index, triangle in enumerate(np.asarray(state.triangles, dtype=int)):
        values = set(map(int, triangle))
        if int(index) == int(triangle_index):
            continue
        if {a, b}.issubset(values):
            others = sorted(values - {a, b})
            if len(others) != 1 or int(others[0]) == apex:
                return False, ["skeleton_adjacent_triangle_mismatch"], {}
            other = int(others[0])
            split.extend(
                [
                    np.asarray([a, apex, other], dtype=int),
                    np.asarray([apex, b, other], dtype=int),
                ]
            )
        else:
            ordinary.append(np.asarray(triangle, dtype=int))
    ordinary_array = np.asarray(ordinary, dtype=int)
    tolerance = _area_tolerance(state.points, ordinary_array)
    removed_degenerate = 1
    if len(ordinary_array):
        ordinary_geometry = triangle_geometry(state.points, ordinary_array)
        if np.any(ordinary_geometry["signed_area"] < -tolerance):
            return False, ["skeleton_contraction_inverted_adjacent_triangle"], {}
        keep = ordinary_geometry["signed_area"] > tolerance
        removed_degenerate += int(np.count_nonzero(~keep))
        ordinary_array = ordinary_array[keep]
    split_array = _orient_ccw(state.points, np.asarray(split, dtype=int))
    if len(split_array):
        split_geometry = triangle_geometry(state.points, split_array)
        if np.any(split_geometry["signed_area"] <= _area_tolerance(state.points, split_array)):
            return False, ["skeleton_split_created_nonpositive_triangle"], {}
    combined = np.vstack([ordinary_array, split_array]) if len(ordinary_array) else split_array
    canonical: set[tuple[int, int, int]] = set()
    unique: list[np.ndarray] = []
    duplicate_count = 0
    for triangle in combined:
        key = tuple(sorted(map(int, triangle)))
        if len(set(key)) != 3:
            removed_degenerate += 1
            continue
        if key in canonical:
            duplicate_count += 1
            continue
        canonical.add(key)
        unique.append(np.asarray(triangle, dtype=int))
    if not unique:
        return False, ["skeleton_contraction_removed_complete_mesh"], {}
    state.triangles = np.asarray(unique, dtype=int)
    delivered_topology = build_edge_topology(len(state.points), state.triangles)
    retained_edges = (tuple(sorted((a, apex))), tuple(sorted((apex, b))))
    if any(len(delivered_topology.edge_to_triangles.get(value, [])) != 2 for value in retained_edges):
        return False, ["collapsed_internal_line_not_two_sided"], {}
    state.last_affected = sorted(affected)
    evidence = {
        "source_triangle": int(triangle_index),
        "source_triangle_original_lineage": [
            int(state.lineage[value]) for value in sorted(source_triangle)
        ],
        "apex_original_lineage": int(state.lineage[apex]),
        "base_original_lineage": [int(state.lineage[a]), int(state.lineage[b])],
        "projection_fraction": float(fraction),
        "original_coordinate_xy": list(map(float, original_coordinate)),
        "projection_coordinate_xy": list(map(float, projection)),
        "intermediate_degenerate_triangles_removed": int(removed_degenerate),
        "duplicate_triangles_removed": int(duplicate_count),
        "retained_internal_line_edges": [list(map(int, value)) for value in retained_edges],
    }
    state.ledger.append(
        {
            "operation": "systematic-v4-interior-skeleton-collapse",
            **evidence,
        }
    )
    return True, [], evidence


def _repair_skeleton_boundary_welds_loop(
    state: _State,
    config: AggressiveConditioningConfig,
    initial_components: int,
) -> dict[str, Any]:
    """Collapse boundary-directed slivers while deferring engineering gates.

    This research-only ladder keeps structural/source-arc gates inside each
    transaction, but leaves distributional quality gates to the surrounding
    relaxation-collapse loop.
    """
    started = time.perf_counter()
    baseline_state = state.clone()
    passage_baseline = _passage_clearance_inventory(state, config)
    attempts: list[dict[str, Any]] = []
    excluded: set[tuple[int, int, int]] = set()
    excluded_interior: set[tuple[int, int, int]] = set()
    accepted = 0
    collapse_config = replace(
        config,
        boundary_weld_max_distance_fraction=max(1.0, float(config.boundary_weld_max_distance_fraction)),
        boundary_weld_max_altitude_to_arc_fraction=max(
            0.50,
            float(config.boundary_weld_max_altitude_to_arc_fraction),
        ),
        boundary_weld_land_max_distance_m=max(250.0, float(config.boundary_weld_land_max_distance_m)),
        boundary_weld_open_max_distance_m=max(500.0, float(config.boundary_weld_open_max_distance_m)),
        boundary_weld_anchor_buffer_segments=0,
        boundary_weld_junction_buffer_segments=1,
        boundary_weld_channel_clearance_fraction=0.0,
        boundary_weld_forbidden_kind_tokens=(),
    )
    for _ in range(max(0, int(config.systematic_collapse_welds_per_round))):
        if _deadline_reached(config):
            break
        proposal, screened = _select_boundary_weld(state, collapse_config, excluded)
        attempts.extend(screened)
        operation = "systematic-v4-skeleton-boundary-weld"
        if proposal is None:
            proposal, screened = _select_interior_skeleton_collapse(
                state,
                collapse_config,
                excluded_interior,
            )
            attempts.extend(screened)
            operation = "systematic-v4-interior-skeleton-collapse"
        if proposal is None:
            break
        edge, node, triangle_index = proposal
        key = (int(edge[0]), int(edge[1]), int(node))
        if operation == "systematic-v4-skeleton-boundary-weld":
            excluded.add(key)
        else:
            excluded_interior.add(key)
        before = _summary(state, config)
        trial = state.clone()
        construction_evidence: dict[str, Any] = {}
        if operation == "systematic-v4-skeleton-boundary-weld":
            changed, construction_failures = _weld_vertex_to_boundary_arc(
                trial,
                edge,
                int(node),
                int(triangle_index),
                collapse_config,
            )
        else:
            changed, construction_failures, construction_evidence = _contract_apex_to_internal_edge(
                trial,
                edge,
                int(node),
                int(triangle_index),
            )
        record: dict[str, Any] = {
            "operation": operation,
            "payload": [int(edge[0]), int(edge[1]), int(node)],
            "triangle_index": int(triangle_index),
            "collapse_target": construction_evidence,
            "accepted": False,
        }
        if not changed:
            record["failures"] = construction_failures or ["skeleton_weld_construction_failed"]
            attempts.append(record)
            continue
        _micro_relax(trial, replacement_seed_nodes=trial.last_affected, config=config)
        if _deadline_reached(config):
            record["failures"] = ["deadline_before_candidate_audit"]
            attempts.append(record)
            break
        ok, invariant_report, after = _audit_state(trial, config, initial_components)
        failures = _v3_trial_failures(
            trial,
            before,
            after,
            ok,
            invariant_report,
            None,
            None,
            config,
        )
        failures.extend(
            _v3_global_boundary_failures(
                trial,
                baseline_state,
                passage_baseline,
                config,
            )
        )
        record["before"] = _summary_from(before)
        record["trial"] = _summary_from(after)
        record["invariants"] = invariant_report
        record["failures"] = sorted(set(failures))
        attempts.append(record)
        if failures:
            continue
        _restore(state, trial)
        record["accepted"] = True
        accepted += 1
        excluded.clear()
        excluded_interior.clear()
    return {
        "schema_version": "fvcom_systematic_skeleton_weld_loop_v1",
        "accepted": int(accepted),
        "rejected": int(sum(not bool(item.get("accepted")) for item in attempts)),
        "candidate_attempts": attempts,
        "deadline_reached": bool(_deadline_reached(config)),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _repair_superthin_boundary_adaptive_v3(
    state: _State,
    config: AggressiveConditioningConfig,
    initial_components: int,
) -> dict[str, Any]:
    """Close residual boundary debt with source-arc-only transactions."""
    started = time.perf_counter()
    initial_summary = _summary(state, config)
    attempts: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []
    accepted = 0

    passage_baseline = _passage_clearance_inventory(state, config)
    boundary_snapshot = state.clone()
    boundary_config = replace(
        config,
        max_boundary_ear_removals_per_round=0,
        max_boundary_welds_per_round=min(8, max(0, int(config.max_boundary_welds_per_round))),
        max_boundary_edits_per_round=min(8, max(0, int(config.max_boundary_edits_per_round))),
        max_superthin_flips_per_round=0,
        max_collapses_per_round=0,
        boundary_weld_anchor_buffer_segments=0,
        boundary_weld_junction_buffer_segments=1,
        boundary_weld_channel_clearance_fraction=0.0,
        boundary_weld_forbidden_kind_tokens=(),
    )
    guarded = _repair_superthin_guarded(state, boundary_config, initial_components)
    guarded_failures = _v3_global_boundary_failures(
        state,
        boundary_snapshot,
        passage_baseline,
        config,
    )
    if guarded_failures:
        _restore(state, boundary_snapshot)
        attempts.append(
            {
                "operation": "systematic-v3-boundary-weld-redistribution-ladder",
                "accepted": False,
                "failures": guarded_failures,
                "nested_report": guarded,
            }
        )
        guarded = {
            **guarded,
            "accepted": 0,
            "rolled_back": True,
            "rollback_failures": guarded_failures,
        }
    else:
        accepted += int(guarded.get("accepted", 0))
        attempts.append(
            {
                "operation": "systematic-v3-boundary-weld-redistribution-ladder",
                "accepted": bool(guarded.get("accepted", 0)),
                "failures": [],
                "nested_report": guarded,
            }
        )

    skeleton_welds = (
        _repair_skeleton_boundary_welds_loop(state, config, initial_components)
        if str(config.systematic_gate_scope) == "loop-end"
        and int(config.systematic_collapse_welds_per_round) > 0
        else {
            "schema_version": "fvcom_systematic_skeleton_weld_loop_v1",
            "accepted": 0,
            "rejected": 0,
            "candidate_attempts": [],
            "deadline_reached": bool(_deadline_reached(config)),
            "reason": "disabled",
        }
    )
    accepted += int(skeleton_welds.get("accepted", 0))
    attempts.extend(skeleton_welds.get("candidate_attempts", []))

    blocked: dict[str, dict[str, Any]] = {}
    for _ in range(max(0, int(config.systematic_max_components_per_round))):
        if _deadline_reached(config):
            break
        components = _inventory_superthin_components(state, config)
        boundary_components = [
            component
            for component in components
            if component.get("boundary_chain_ids")
            and component["component_id"] not in blocked
        ]
        if not boundary_components:
            break
        component = boundary_components[0]
        classifications.append({key: value for key, value in component.items() if key != "triangle_indices"})
        raw_windows = _component_boundary_windows_v3(state, component, config)
        windows_by_chain: dict[int, list[dict[str, Any]]] = {}
        for window in raw_windows:
            windows_by_chain.setdefault(int(window["chain_index"]), []).append(window)
        windows: list[dict[str, Any]] = []
        for chain_index in sorted(windows_by_chain):
            ranked = sorted(
                windows_by_chain[chain_index],
                key=lambda item: (
                    -len(item.get("component_nodes", [])),
                    len(item.get("nodes", [])),
                    tuple(item.get("nodes", [])),
                ),
            )
            windows.extend(ranked[:2])
        base_window_groups: list[list[dict[str, Any]]] = [[window] for window in windows]
        if component["classification"] == "under-resolved-passage":
            by_chain: dict[int, dict[str, Any]] = {}
            for window in windows:
                by_chain.setdefault(int(window["chain_index"]), window)
            chain_ids = list(map(int, component.get("boundary_chain_ids", [])))
            if all(chain_id in by_chain for chain_id in chain_ids[:2]):
                base_window_groups.insert(0, [by_chain[chain_ids[0]], by_chain[chain_ids[1]]])
        slide_budget = max(1, int(config.systematic_v3_max_candidates_per_component) // 2)
        candidate_windows = [
            {"windows": group, "slide_fraction": fraction}
            for fraction in (0.25, 0.50, 0.75, 1.0)
            for group in base_window_groups
        ][:slide_budget]
        best: tuple[tuple[float, ...], _State, dict[str, Any]] | None = None
        component_attempts: list[dict[str, Any]] = []
        stage_before = _summary(state, config)
        passage_before = _component_passage_clearance(state, component)
        for candidate_spec in candidate_windows:
            if _deadline_reached(config):
                break
            window_group = candidate_spec["windows"]
            slide_fraction = float(candidate_spec["slide_fraction"])
            trial = state.clone()
            changed, evidence = _apply_source_arc_windows_v3(
                trial,
                window_group,
                config,
                slide_fraction=slide_fraction,
            )
            record: dict[str, Any] = {
                "component_id": component["component_id"],
                "classification": component["classification"],
                "operation": "systematic-v3-source-arc-slide",
                "boundary_windows": evidence,
                "slide_fraction": slide_fraction,
                "accepted": False,
            }
            if not changed:
                record["failures"] = ["source_arc_window_no_change"]
                component_attempts.append(record)
                continue
            _micro_relax(trial, replacement_seed_nodes=trial.last_affected, config=config)
            ok, invariant_report, trial_summary = _audit_state(trial, config, initial_components)
            passage_after = _component_passage_clearance(trial, component)
            failures = _v3_trial_failures(
                trial,
                stage_before,
                trial_summary,
                ok,
                invariant_report,
                passage_before,
                passage_after,
                config,
            )
            record["trial"] = _summary_from(trial_summary)
            record["invariants"] = invariant_report
            record["passage_clearance_before_m"] = passage_before
            record["passage_clearance_after_m"] = passage_after
            record["failures"] = failures
            if failures:
                component_attempts.append(record)
                continue
            score = _v3_candidate_score(trial_summary, evidence, state, trial)
            record["candidate_score"] = list(score)
            component_attempts.append(record)
            if best is None or score < best[0]:
                best = (score, trial, record)
        remaining_budget = max(
            0,
            int(config.systematic_v3_max_candidates_per_component) - len(component_attempts),
        )
        for operation, payload in _component_boundary_redistribution_candidates_v3(
            state,
            component,
            config,
        )[:remaining_budget]:
            if _deadline_reached(config):
                break
            trial = state.clone()
            if operation == "remove":
                changed = _remove_boundary_vertex(trial, int(payload), config)
            else:
                changed = _split_boundary_edge(trial, tuple(map(int, payload)), config)
            record = {
                "component_id": component["component_id"],
                "classification": component["classification"],
                "operation": f"systematic-v3-boundary-{operation}",
                "payload": list(_proposal_key(operation, payload)[1]),
                "accepted": False,
            }
            if not changed:
                record["failures"] = ["boundary_redistribution_construction_failed"]
                component_attempts.append(record)
                continue
            _micro_relax(trial, replacement_seed_nodes=trial.last_affected, config=config)
            ok, invariant_report, trial_summary = _audit_state(trial, config, initial_components)
            passage_after = _component_passage_clearance(trial, component)
            failures = _v3_trial_failures(
                trial,
                stage_before,
                trial_summary,
                ok,
                invariant_report,
                passage_before,
                passage_after,
                config,
            )
            record["trial"] = _summary_from(trial_summary)
            record["invariants"] = invariant_report
            record["passage_clearance_before_m"] = passage_before
            record["passage_clearance_after_m"] = passage_after
            record["failures"] = failures
            if failures:
                component_attempts.append(record)
                continue
            evidence = [{"operation": operation, "payload": record["payload"]}]
            score = _v3_candidate_score(trial_summary, evidence, state, trial)
            record["candidate_score"] = list(score)
            component_attempts.append(record)
            if best is None or score < best[0]:
                best = (score, trial, record)
        attempts.extend(component_attempts)
        if best is None:
            blocked[component["component_id"]] = {
                **{key: value for key, value in component.items() if key != "triangle_indices"},
                "attempt_count": int(len(component_attempts)),
                "failure_counts": _failure_counts(component_attempts),
            }
            continue
        _, selected, selected_record = best
        _restore(state, selected)
        selected_record["accepted"] = True
        accepted += 1
        blocked.clear()

    final_components = _inventory_superthin_components(state, config)
    for component in final_components:
        blocked.setdefault(
            component["component_id"],
            {key: value for key, value in component.items() if key != "triangle_indices"},
        )
    return {
        "schema_version": "fvcom_systematic_thin_repair_v3",
        "profile": "systematic-v3",
        "settings": {
            "boundary_motion": "source-arc-only",
            "obc_policy": str(config.systematic_v3_obc_policy),
            "window_radius": int(config.systematic_v3_boundary_window_radius),
            "weld_snap_fraction": float(config.systematic_v3_weld_snap_fraction),
            "passage_clearance_tolerance_m": float(config.systematic_v3_passage_clearance_tolerance_m),
        },
        "accepted": int(accepted),
        "rejected": int(sum(not bool(item.get("accepted")) for item in attempts)),
        "guarded_boundary_ladder": guarded,
        "skeleton_boundary_welds": skeleton_welds,
        "component_classifications": classifications,
        "candidate_attempts": attempts,
        "blocked_components": list(blocked.values()),
        "obc_remap_manifest": _obc_remap_manifest(state),
        "before": initial_summary,
        "after": _summary(state, config),
        "runtime_seconds": float(time.perf_counter() - started),
        "deadline_reached": bool(_deadline_reached(config)),
    }


def _component_boundary_windows_v3(
    state: _State,
    component: dict[str, Any],
    config: AggressiveConditioningConfig,
) -> list[dict[str, Any]]:
    component_nodes = set(
        map(
            int,
            np.unique(state.triangles[np.asarray(component["triangle_indices"], dtype=int)]),
        )
    )
    radius = max(1, int(config.systematic_v3_boundary_window_radius))
    windows: dict[tuple[int, tuple[int, ...]], dict[str, Any]] = {}
    for node in sorted(component_nodes):
        membership = _find_chain_node(state.chains, node)
        if membership is None:
            continue
        chain_index, position = membership
        chain = state.chains[chain_index]
        kind = str(state.kinds[node])
        positions = [int(position)]
        for direction in (-1, 1):
            current = int(position)
            side: list[int] = []
            for _ in range(radius):
                following = (current + direction) % len(chain)
                candidate = int(chain[following])
                if str(state.kinds[candidate]) != kind:
                    break
                side.append(int(following))
                current = following
                if bool(state.hard[candidate]):
                    break
            if direction < 0:
                positions = list(reversed(side)) + positions
            else:
                positions.extend(side)
        ordered_nodes = [int(chain[index]) for index in positions]
        if len(ordered_nodes) < 3:
            continue
        movable = [
            value
            for value in ordered_nodes[1:-1]
            if not bool(state.hard[value]) and str(state.kinds[value]) == kind
        ]
        if not movable:
            continue
        key = (int(chain_index), tuple(ordered_nodes))
        windows[key] = {
            "chain_index": int(chain_index),
            "nodes": ordered_nodes,
            "component_nodes": sorted(component_nodes & set(ordered_nodes)),
            "boundary_kind": kind,
        }
    return [windows[key] for key in sorted(windows)]


def _component_boundary_redistribution_candidates_v3(
    state: _State,
    component: dict[str, Any],
    config: AggressiveConditioningConfig,
) -> list[tuple[str, Any]]:
    triangle_indices = np.asarray(component["triangle_indices"], dtype=int)
    component_nodes = sorted(set(map(int, np.unique(state.triangles[triangle_indices]))))
    open_set = set(map(int, state.open_nodes))
    proposals: list[tuple[str, Any]] = []
    for node in component_nodes:
        if not state.fixed[node] or state.hard[node]:
            continue
        if (
            str(config.systematic_v3_obc_policy) == "preserve"
            and node in open_set
        ):
            continue
        if _boundary_removal_allowed(state, node, config):
            proposals.append(("remove", int(node)))
    protected = chain_edges(state.chains)
    component_edges = sorted(
        set(
            edge
            for triangle_index in triangle_indices
            for edge in _triangle_edge_keys(state.triangles[int(triangle_index)])
            if edge in protected
        )
    )
    for edge in component_edges:
        if (
            str(config.systematic_v3_obc_policy) == "preserve"
            and edge[0] in open_set
            and edge[1] in open_set
        ):
            continue
        proposals.append(("split", edge))
    return proposals


def _apply_source_arc_windows_v3(
    state: _State,
    windows: list[dict[str, Any]],
    config: AggressiveConditioningConfig,
    *,
    slide_fraction: float = 1.0,
) -> tuple[bool, list[dict[str, Any]]]:
    signed_area_before = _signed_mesh_area(state.points, state.triangles)
    moved: set[int] = set()
    evidence: list[dict[str, Any]] = []
    for window in windows:
        chain_index = int(window["chain_index"])
        nodes = list(map(int, window["nodes"]))
        if len(nodes) < 3:
            continue
        positions = _target_equalized_source_arc_positions(state, chain_index, nodes)
        if positions is None:
            continue
        changes: list[dict[str, Any]] = []
        for node, target_point in zip(nodes[1:-1], positions, strict=True):
            if bool(state.hard[node]):
                continue
            old = state.points[node].copy()
            point = _source_arc_fractional_move(
                state,
                chain_index,
                old,
                target_point,
                float(slide_fraction),
            )
            if point is None:
                continue
            if np.linalg.norm(old - point) <= 1.0e-9:
                continue
            state.points[node] = point
            state.targets[node] = _sample_target_at(state, point, fallback=float(state.targets[node]))
            moved.add(int(node))
            changes.append(
                {
                    "node_lineage": int(state.lineage[node]),
                    "old_xy": [float(old[0]), float(old[1])],
                    "new_xy": [float(point[0]), float(point[1])],
                }
            )
        if changes:
            evidence.append(
                {
                    "chain_index": chain_index,
                    "boundary_kind": str(window["boundary_kind"]),
                    "slide_fraction": float(slide_fraction),
                    "window_node_lineage": [int(state.lineage[node]) for node in nodes],
                    "moved_nodes": changes,
                }
            )
    if not moved:
        return False, evidence
    actual_area_change = abs(_signed_mesh_area(state.points, state.triangles) - signed_area_before)
    if not _boundary_area_budget_allows(state, actual_area_change, config):
        return False, [*evidence, {"failure": "domain_area_budget"}]
    affected = set(moved)
    topology = build_edge_topology(len(state.points), state.triangles)
    for node in moved:
        affected.update(map(int, topology.node_neighbors[int(node)]))
    state.last_affected = sorted(affected)
    state.cumulative_boundary_area_change_m2 += actual_area_change
    state.ledger.append(
        {
            "operation": "systematic-v3-source-arc-slide",
            "windows": evidence,
            "actual_signed_domain_area_change_m2": float(actual_area_change),
        }
    )
    return True, evidence


def _v3_trial_failures(
    trial: _State,
    before: dict[str, Any],
    after: dict[str, Any],
    invariants_ok: bool,
    invariant_report: dict[str, Any],
    passage_before: float | None,
    passage_after: float | None,
    config: AggressiveConditioningConfig,
) -> list[str]:
    failures: list[str] = []
    if not invariants_ok:
        failures.extend(_failed_invariant_names(invariant_report))
    debt_before = (int(before["superthin_triangle_count"]), float(before["superthin_severity_sum"]))
    debt_after = (int(after["superthin_triangle_count"]), float(after["superthin_severity_sum"]))
    if not debt_after < debt_before:
        failures.append("superthin_debt_not_reduced")
    if str(config.systematic_gate_scope) != "loop-end":
        if after["q_min"] + 1.0e-12 < before["q_min"]:
            failures.append("q_min_regression")
        if after["q_p01"] + 1.0e-6 < before["q_p01"]:
            failures.append("q_p01_regression")
        if after["q_l3_sigma"] + 1.0e-8 < before["q_l3_sigma"]:
            failures.append("q_l3_sigma_regression")
        if after["minimum_angle_deg"] + 1.0e-8 < before["minimum_angle_deg"]:
            failures.append("minimum_angle_regression")
        if after["minimum_angle_p01_deg"] + 1.0e-3 < before["minimum_angle_p01_deg"]:
            failures.append("minimum_angle_p01_regression")
        if after["l_over_h_p95"] > 1.001 * max(before["l_over_h_p95"], 1.0e-12):
            failures.append("l_over_h_p95_regression")
        if after["l_over_h_count_above_1_55"] > before["l_over_h_count_above_1_55"]:
            failures.append("l_over_h_excess_count_increase")
        if after["area_transition_count_above_0_50"] > before["area_transition_count_above_0_50"]:
            failures.append("area_transition_excess_count_increase")
        if after["count_valence_above_limit"] > before["count_valence_above_limit"]:
            failures.append("valence_gate_regression")
        if after["singly_connected_triangle_count"] > before["singly_connected_triangle_count"]:
            failures.append("new_singly_connected_triangles")
    if after["boundary_degree_anomaly_count"] > before["boundary_degree_anomaly_count"]:
        failures.append("new_boundary_degree_anomalies")
    if after["boundary_component_count"] != before["boundary_component_count"]:
        failures.append("boundary_traversability_component_change")
    if _maximum_boundary_source_arc_deviation(trial) > 1.0e-6:
        failures.append("boundary_vertex_off_source_arc")
    if not _boundary_loops_simple(trial):
        failures.append("boundary_self_intersection")
    if (
        passage_before is not None
        and passage_after is not None
        and passage_after + float(config.systematic_v3_passage_clearance_tolerance_m) < passage_before
    ):
        failures.append("passage_clearance_regression")
    if str(config.systematic_v3_obc_policy) == "preserve":
        manifest = _obc_remap_manifest(trial)
        if int(manifest["delivered_obc_count"]) != int(manifest["original_obc_count"]):
            failures.append("obc_count_changed_under_preserve_policy")
    return sorted(set(failures))


def _v3_global_boundary_failures(
    trial: _State,
    before_state: _State,
    passage_baseline: dict[tuple[int, int], float],
    config: AggressiveConditioningConfig,
) -> list[str]:
    failures: list[str] = []
    if _maximum_boundary_source_arc_deviation(trial) > 1.0e-6:
        failures.append("boundary_vertex_off_source_arc")
    if not _boundary_loops_simple(trial):
        failures.append("boundary_self_intersection")
    current = _passage_clearance_inventory(trial, config)
    for pair, baseline in passage_baseline.items():
        if pair in current and current[pair] + float(config.systematic_v3_passage_clearance_tolerance_m) < baseline:
            failures.append(f"passage_clearance_regression:{pair[0]}-{pair[1]}")
    if str(config.systematic_v3_obc_policy) == "preserve":
        if len(trial.open_nodes) != len(before_state.open_nodes):
            failures.append("obc_count_changed_under_preserve_policy")
    return sorted(set(failures))


def _v3_candidate_score(
    summary: dict[str, Any],
    evidence: list[dict[str, Any]],
    before_state: _State,
    trial: _State,
) -> tuple[float, ...]:
    moved_count = int(
        sum(len(window.get("moved_nodes", [])) for window in evidence)
    )
    return (
        float(summary["superthin_triangle_count"]),
        float(summary["superthin_severity_sum"]),
        -float(summary["q_min"]),
        -float(summary["minimum_angle_deg"]),
        float(_maximum_boundary_source_arc_deviation(trial)),
        float(summary["count_valence_above_limit"]),
        float(summary["l_over_h_count_above_1_55"]),
        float(summary["area_transition_count_above_0_50"]),
        float(abs(len(trial.open_nodes) - len(before_state.open_nodes))),
        float(moved_count),
    )


def _inventory_superthin_components(state: _State, config: AggressiveConditioningConfig) -> list[dict[str, Any]]:
    geometry = triangle_geometry(state.points, state.triangles)
    minimum_angles = np.min(geometry["angles_deg"], axis=1)
    bad = np.where(
        (geometry["quality"] < float(config.superthin_quality_threshold))
        | (minimum_angles < float(config.superthin_min_angle_deg))
    )[0]
    if not len(bad):
        return []
    bad_set = set(map(int, bad))
    node_to_bad: dict[int, list[int]] = {}
    for triangle_index in bad_set:
        for node in state.triangles[triangle_index]:
            node_to_bad.setdefault(int(node), []).append(int(triangle_index))
    unseen = set(bad_set)
    raw_components: list[list[int]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(int(current))
            neighbors: set[int] = set()
            for node in state.triangles[current]:
                neighbors.update(node_to_bad.get(int(node), []))
            for following in sorted(neighbors & unseen, reverse=True):
                unseen.remove(int(following))
                stack.append(int(following))
        raw_components.append(sorted(component))
    chain_membership: dict[int, set[int]] = {}
    for chain_index, chain in enumerate(state.chains):
        for node in chain:
            chain_membership.setdefault(int(node), set()).add(int(chain_index))
    output: list[dict[str, Any]] = []
    for triangles in raw_components:
        nodes = sorted(set(map(int, np.unique(state.triangles[np.asarray(triangles, dtype=int)]))))
        lineage = sorted(map(int, state.lineage[np.asarray(nodes, dtype=int)]))
        digest = hashlib.sha1(",".join(map(str, lineage)).encode("ascii")).hexdigest()[:10]
        fixed_nodes = [node for node in nodes if bool(state.fixed[node])]
        hard_nodes = [node for node in nodes if bool(state.hard[node])]
        chain_ids = sorted(set().union(*(chain_membership.get(node, set()) for node in fixed_nodes))) if fixed_nodes else []
        passage_width = float("nan")
        if len(chain_ids) >= 2:
            candidates: list[float] = []
            for first, second in combinations(chain_ids, 2):
                left = [node for node in fixed_nodes if first in chain_membership.get(node, set())]
                right = [node for node in fixed_nodes if second in chain_membership.get(node, set())]
                if left and right:
                    delta = state.points[np.asarray(left, dtype=int)][:, None, :] - state.points[np.asarray(right, dtype=int)][None, :, :]
                    candidates.append(float(np.min(np.linalg.norm(delta, axis=2))))
            if candidates:
                passage_width = float(min(candidates))
        target = float(np.median(state.targets[np.asarray(nodes, dtype=int)])) if nodes else float("nan")
        gap_ratio = passage_width / max(target, 1.0e-12) if np.isfinite(passage_width) else float("nan")
        if len(chain_ids) >= 2 and np.isfinite(gap_ratio) and gap_ratio < 1.5:
            classification = "under-resolved-passage"
        elif fixed_nodes and hard_nodes:
            classification = "fixed-boundary-hard-anchor-fan"
        elif fixed_nodes:
            classification = "fixed-boundary-fan"
        elif not fixed_nodes:
            classification = "interior-connectivity-transition"
        else:
            classification = "mixed"
        severity = float(
            np.sum(
                np.maximum(0.0, float(config.superthin_quality_threshold) - geometry["quality"][triangles])
                + np.maximum(0.0, float(config.superthin_min_angle_deg) - minimum_angles[triangles])
            )
        )
        output.append(
            {
                "component_id": f"thin-{lineage[0]}-{digest}",
                "classification": classification,
                "triangle_indices": triangles,
                "triangle_count": int(len(triangles)),
                "node_lineage": lineage,
                "fixed_node_count": int(len(fixed_nodes)),
                "hard_anchor_count": int(len(hard_nodes)),
                "boundary_chain_ids": chain_ids,
                "passage_width_m": passage_width if np.isfinite(passage_width) else None,
                "gap_over_h": gap_ratio if np.isfinite(gap_ratio) else None,
                "local_feature_target_m": (
                    min(target, passage_width / max(1, int(config.systematic_min_passage_elements)))
                    if np.isfinite(passage_width)
                    else None
                ),
                "minimum_quality": float(np.min(geometry["quality"][triangles])),
                "minimum_angle_deg": float(np.min(minimum_angles[triangles])),
                "severity": severity,
            }
        )
    output.sort(key=lambda item: (-float(item["severity"]), int(item["node_lineage"][0])))
    return output


def _expand_triangle_patch(triangles: np.ndarray, topology: Any, seeds: list[int], rings: int) -> np.ndarray:
    selected = set(map(int, seeds))
    frontier = set(selected)
    for _ in range(max(0, int(rings))):
        following: set[int] = set()
        for triangle_index in frontier:
            tri = np.asarray(triangles, dtype=int)[int(triangle_index)]
            for edge in _triangle_edge_keys(tri):
                following.update(map(int, topology.edge_to_triangles.get(edge, [])))
        following -= selected
        selected.update(following)
        frontier = following
        if not frontier:
            break
    return np.asarray(sorted(selected), dtype=int)


def _ordered_patch_boundary(triangles: np.ndarray, patch: np.ndarray) -> list[int] | None:
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in np.asarray(triangles, dtype=int)[np.asarray(patch, dtype=int)]:
        for edge in _triangle_edge_keys(tri):
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if not adjacency or any(len(values) != 2 for values in adjacency.values()):
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
    return ring


def _systematic_support_groups(
    state: _State,
    component: dict[str, Any],
    patch: np.ndarray,
    ring: list[int],
    config: AggressiveConditioningConfig,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    polygon = Polygon(state.points[np.asarray(ring, dtype=int)])
    if not polygon.is_valid or polygon.area <= 0.0:
        return [np.empty((0, 2), dtype=float)], {"reason": "invalid_patch_polygon"}
    patch_nodes = sorted(set(map(int, np.unique(state.triangles[np.asarray(patch, dtype=int)]))))
    existing = state.points[np.asarray(patch_nodes, dtype=int)]
    supports: list[np.ndarray] = []
    protected = chain_edges(state.chains)
    patch_centroid = np.asarray([polygon.representative_point().x, polygon.representative_point().y], dtype=float)
    for triangle_index in component["triangle_indices"]:
        tri = state.triangles[int(triangle_index)]
        protected_edges = [edge for edge in _triangle_edge_keys(tri) if edge in protected]
        for edge in protected_edges:
            a, b = map(int, edge)
            midpoint = 0.5 * (state.points[a] + state.points[b])
            direction = patch_centroid - midpoint
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-12:
                continue
            edge_length = float(np.linalg.norm(state.points[b] - state.points[a]))
            if component["classification"] == "fixed-boundary-hard-anchor-fan":
                height = 0.15 * edge_length
            elif component["classification"] == "under-resolved-passage":
                passage_width = component.get("passage_width_m")
                width_cap = 0.40 * float(passage_width) if passage_width is not None else float("inf")
                height = min(0.25 * edge_length, width_cap)
            else:
                height = 0.30 * edge_length
            supports.append(midpoint + min(height, 0.75 * norm) * direction / norm)
        coords = state.points[np.asarray(tri, dtype=int)]
        lengths = np.asarray(
            [
                np.linalg.norm(coords[1] - coords[2]),
                np.linalg.norm(coords[0] - coords[2]),
                np.linalg.norm(coords[0] - coords[1]),
            ],
            dtype=float,
        )
        incenter = np.sum(coords * lengths[:, None], axis=0) / max(float(np.sum(lengths)), 1.0e-12)
        supports.append(0.35 * incenter + 0.65 * patch_centroid)
    if component["classification"] == "under-resolved-passage":
        chain_ids = list(map(int, component.get("boundary_chain_ids", [])))
        membership: dict[int, set[int]] = {}
        for chain_index in chain_ids:
            for node in state.chains[chain_index]:
                membership.setdefault(int(node), set()).add(chain_index)
        pairs: list[tuple[float, np.ndarray]] = []
        fixed_nodes = [node for node in patch_nodes if state.fixed[node] and node in membership]
        for a, b in combinations(fixed_nodes, 2):
            if membership[a].isdisjoint(membership[b]):
                distance = float(np.linalg.norm(state.points[a] - state.points[b]))
                pairs.append((distance, 0.5 * (state.points[a] + state.points[b])))
        passage_support = [
            midpoint
            for _, midpoint in sorted(pairs, key=lambda item: item[0])[: max(2, int(config.systematic_max_support_points))]
        ]
        supports = [*passage_support, *supports]
    supports.append(patch_centroid)
    tolerance = max(np.sqrt(max(float(polygon.area), 1.0e-30)) * 1.0e-8, 1.0e-6)
    unique: list[np.ndarray] = []
    for point in supports:
        point = np.asarray(point, dtype=float)
        if not bool(contains_xy(polygon, np.asarray([point[0]]), np.asarray([point[1]]))[0]):
            continue
        if len(existing) and float(np.min(np.linalg.norm(existing - point, axis=1))) <= tolerance:
            continue
        if any(float(np.linalg.norm(value - point)) <= tolerance for value in unique):
            continue
        unique.append(point)
    unique = unique[: max(0, int(config.systematic_max_support_points))]
    empty = np.empty((0, 2), dtype=float)
    all_support = np.asarray(unique, dtype=float) if unique else empty
    if component["classification"] == "under-resolved-passage":
        groups = [all_support]
    elif component["classification"] in {"fixed-boundary-fan", "fixed-boundary-hard-anchor-fan"}:
        groups = [all_support, empty] if len(all_support) else [empty]
    else:
        groups = [empty]
        if len(all_support):
            groups.append(np.asarray([all_support[0]], dtype=float))
        if len(all_support) > 1:
            groups.append(all_support)
    return groups, {
        "candidate_support_point_count": int(len(unique)),
        "passage_width_m": component.get("passage_width_m"),
        "local_feature_target_m": component.get("local_feature_target_m"),
    }


def _retriangulate_patch_with_support(
    state: _State,
    patch: np.ndarray,
    ring: list[int],
    support: np.ndarray,
    component: dict[str, Any],
    config: AggressiveConditioningConfig,
) -> tuple[bool, list[str]]:
    patch = np.asarray(patch, dtype=int)
    patch_nodes = sorted(set(map(int, np.unique(state.triangles[patch]))))
    component_nodes = set(
        map(
            int,
            np.unique(state.triangles[np.asarray(component["triangle_indices"], dtype=int)]),
        )
    )
    removable_component_nodes = {
        node for node in component_nodes if not state.fixed[node] and node not in set(ring)
    }
    patch_nodes = [node for node in patch_nodes if node not in removable_component_nodes]
    local_points = state.points[np.asarray(patch_nodes, dtype=int)].copy()
    support = np.asarray(support, dtype=float).reshape((-1, 2))
    local_input = np.vstack([local_points, support]) if len(support) else local_points
    if len(local_input) < 3:
        return False, ["insufficient_patch_points"]
    try:
        simplices = np.asarray(Delaunay(local_input).simplices, dtype=int)
    except (QhullError, ValueError):
        return False, ["local_delaunay_failed"]
    polygon = Polygon(state.points[np.asarray(ring, dtype=int)])
    centroids = np.mean(local_input[simplices], axis=1)
    inside = np.asarray(contains_xy(polygon, centroids[:, 0], centroids[:, 1]), dtype=bool)
    simplices = simplices[inside]
    if not len(simplices):
        return False, ["no_triangles_inside_patch"]
    new_ids = list(range(len(state.points), len(state.points) + len(support)))
    local_to_global = np.asarray([*patch_nodes, *new_ids], dtype=int)
    replacement = local_to_global[simplices]
    trial_points = np.vstack([state.points, support]) if len(support) else state.points.copy()
    replacement = _orient_ccw(trial_points, replacement)
    ring_edges = {
        tuple(sorted((int(ring[index]), int(ring[(index + 1) % len(ring)]))))
        for index in range(len(ring))
    }
    replacement_edges = _edge_set(replacement)
    if not ring_edges.issubset(replacement_edges):
        return False, ["patch_boundary_edge_missing"]
    protected_inside = chain_edges(state.chains) & _edge_set(state.triangles[patch])
    if not protected_inside.issubset(replacement_edges):
        return False, ["protected_edge_missing"]
    old_geometry = triangle_geometry(state.points, state.triangles[patch])
    old_area = float(np.sum(old_geometry["area"]))
    new_geometry = triangle_geometry(trial_points, replacement)
    if np.any(new_geometry["signed_area"] <= _area_tolerance(trial_points, replacement)):
        return False, ["nonpositive_replacement_area"]
    new_area = float(np.sum(new_geometry["area"]))
    if abs(new_area - old_area) > 1.0e-8 * max(old_area, 1.0):
        return False, ["patch_area_mismatch"]
    minimum_angles = np.min(new_geometry["angles_deg"], axis=1)
    old_minimum_angles = np.min(old_geometry["angles_deg"], axis=1)
    old_superthin = int(
        np.count_nonzero(
            (old_geometry["quality"] < float(config.superthin_quality_threshold))
            | (old_minimum_angles < float(config.superthin_min_angle_deg))
        )
    )
    new_superthin = int(
        np.count_nonzero(
            (new_geometry["quality"] < float(config.superthin_quality_threshold))
            | (minimum_angles < float(config.superthin_min_angle_deg))
        )
    )
    if new_superthin >= old_superthin:
        return False, ["replacement_does_not_reduce_patch_superthin"]
    if len(support):
        feature_target = component.get("local_feature_target_m")
        fallback = float(np.median(state.targets[np.asarray(patch_nodes, dtype=int)]))
        targets = np.asarray(
            [
                _sample_target_at(state, point, fallback=fallback)
                for point in support
            ],
            dtype=float,
        )
        if feature_target is not None and np.isfinite(float(feature_target)):
            targets = np.minimum(targets, float(feature_target))
        state.points = trial_points
        state.fixed = np.concatenate([state.fixed, np.zeros(len(support), dtype=bool)])
        state.targets = np.concatenate([state.targets, targets])
        state.kinds.extend(["interior"] * len(support))
        state.hard = np.concatenate([state.hard, np.zeros(len(support), dtype=bool)])
        state.lineage = np.concatenate([state.lineage, _new_lineage_ids(state, len(support))])
    keep = np.ones(len(state.triangles), dtype=bool)
    keep[patch] = False
    state.triangles = _orient_ccw(state.points, np.vstack([state.triangles[keep], replacement]))
    state.last_affected = sorted(set(map(int, np.unique(replacement))))
    state.ledger.append(
        {
            "operation": "systematic-cavity-retriangulation",
            "component_id": component["component_id"],
            "classification": component["classification"],
            "patch_triangle_count": int(len(patch)),
            "replacement_triangle_count": int(len(replacement)),
            "inserted_support_node_count": int(len(support)),
            "removed_movable_component_node_count": int(len(removable_component_nodes)),
            "local_feature_target_m": component.get("local_feature_target_m"),
        }
    )
    _compact(state)
    return True, []


def _failure_counts(attempts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attempt in attempts:
        for failure in attempt.get("failures", []):
            counts[str(failure)] = counts.get(str(failure), 0) + 1
    return dict(sorted(counts.items()))


def _connectivity_nonregression(
    before: dict[str, Any],
    after: dict[str, Any],
) -> bool:
    """Apply the topology-only acceptance contract for one restriction."""
    topology_ok = bool(
        after["singly_connected_triangle_count"]
        <= before["singly_connected_triangle_count"]
        and after["boundary_degree_anomaly_count"]
        <= before["boundary_degree_anomaly_count"]
        and after["boundary_component_count"]
        == before["boundary_component_count"]
        and after["protected_edge_not_boundary_count"]
        <= before["protected_edge_not_boundary_count"]
    )
    return bool(
        topology_ok
        and after["superthin_triangle_count"]
        <= before["superthin_triangle_count"]
        and after["q_min"] + 1.0e-12 >= before["q_min"]
        and after["q_p01"] + 1.0e-9 >= before["q_p01"]
        and after["q_l3_sigma"] + 1.0e-9
        >= before["q_l3_sigma"]
        and after["minimum_angle_deg"] + 1.0e-8
        >= before["minimum_angle_deg"]
        and after["l_over_h_count_above_1_55"]
        <= before["l_over_h_count_above_1_55"]
        and after["area_transition_count_above_0_50"]
        <= before["area_transition_count_above_0_50"]
        and after["count_valence_above_limit"]
        <= before["count_valence_above_limit"]
        and after["valence_excess_sum"]
        <= before["valence_excess_sum"]
    )


def _deadline_reached(config: AggressiveConditioningConfig, *, reserve_seconds: float = 0.0) -> bool:
    deadline = config.deadline_monotonic_s
    return bool(
        deadline is not None
        and time.perf_counter() + max(0.0, float(reserve_seconds)) >= float(deadline)
    )


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
    if str(config.thin_repair_profile) == "systematic-v3":
        snap_limit = float(config.systematic_v3_weld_snap_fraction) * max(
            _edge_target(state.targets, edge),
            1.0e-12,
        )
        snap_target = min(
            (a, b),
            key=lambda value: float(np.linalg.norm(projection - state.points[int(value)])),
        )
        snap_distance = float(np.linalg.norm(projection - state.points[int(snap_target)]))
        if snap_distance <= snap_limit:
            state.triangles[state.triangles == int(node)] = int(snap_target)
            keep = np.asarray(
                [len(set(map(int, values))) == 3 for values in state.triangles],
                dtype=bool,
            )
            state.triangles = _orient_ccw(state.points, state.triangles[keep])
            actual_area_change = abs(_signed_mesh_area(state.points, state.triangles) - signed_area_before)
            if not _boundary_area_budget_allows(state, actual_area_change, config):
                return False, ["domain_area_budget"]
            state.last_affected = sorted(affected)
            state.cumulative_boundary_area_change_m2 += actual_area_change
            state.ledger.append(
                {
                    "operation": "boundary-arc-weld-snap",
                    "welded_original_node": int(state.lineage[int(node)]),
                    "target_original_node": int(state.lineage[int(snap_target)]),
                    "target_is_hard_anchor": bool(state.hard[int(snap_target)]),
                    "parent_edge_original_nodes": [int(state.lineage[a]), int(state.lineage[b])],
                    "projection_fraction": float(fraction),
                    "weld_distance_m": float(distance),
                    "snap_distance_m": float(snap_distance),
                    "intermediate_degenerate_triangles_removed": int(np.count_nonzero(~keep)),
                    "actual_signed_domain_area_change_m2": float(actual_area_change),
                }
            )
            _compact(state)
            return True, []
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
        if str(config.thin_repair_profile) != "systematic-v3" and any(state.hard[value] for value in nearby):
            failures.append("hard_anchor_buffer")
        junction_buffer = max(0, int(config.boundary_weld_junction_buffer_segments))
        junction_nodes = _cyclic_chain_window(chain, position, junction_buffer)
        junction_kinds = {str(state.kinds[value]) for value in junction_nodes}
        if len(junction_kinds) > 1:
            failures.append("boundary_kind_junction_buffer")
    clearance_fraction = float(config.boundary_weld_channel_clearance_fraction)
    if clearance_fraction > 0.0:
        clearance = _minimum_remote_boundary_clearance(state, projection, edge)
        if np.isfinite(clearance) and clearance < clearance_fraction * h:
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
    if trial["area_transition_count_above_0_50"] > before["area_transition_count_above_0_50"]:
        failures.append("area_transition_excess_count_increase")
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
    new_point = 0.5 * (state.points[a] + state.points[b])
    if str(config.thin_repair_profile) == "systematic-v3":
        source_midpoint = _source_arc_midpoint(state, chain_index, a, b)
        if source_midpoint is not None:
            new_point = source_midpoint
    state.points = np.vstack([state.points, new_point])
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
    previous_to_node = state.points[node] - state.points[previous]
    previous_to_following = (
        state.points[following] - state.points[previous]
    )
    area_change = 0.5 * abs(
        float(
            previous_to_node[0] * previous_to_following[1]
            - previous_to_node[1] * previous_to_following[0]
        )
    )
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
            allow_new_singly_connected=(
                str(config.profile_name) == "minimal-topology-v1"
            ),
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
                    allow_new_singly_connected=(
                        str(config.profile_name) == "minimal-topology-v1"
                    ),
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
                allow_new_singly_connected=(
                    str(config.profile_name) == "minimal-topology-v1"
                ),
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
            allow_new_singly_connected=(
                str(config.profile_name) == "minimal-topology-v1"
            ),
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
    edge_allowed: Callable[[tuple[int, int]], bool] | None = None,
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
        active_polygon = Polygon(
            points[np.asarray(vertices, dtype=int)]
        )
        span = np.ptp(
            points[np.asarray(vertices, dtype=int)],
            axis=0,
        )
        coverage_tolerance = max(
            1.0e-12,
            1.0e-12 * float(np.max(span)),
        )
        if not active_polygon.is_valid or active_polygon.area <= 0.0:
            return None
        covered_polygon = active_polygon.buffer(coverage_tolerance)
        ears: list[tuple[tuple[float, ...], int, list[int], tuple[int, int]]] = []
        for index, current in enumerate(vertices):
            previous = int(vertices[index - 1])
            following = int(vertices[(index + 1) % len(vertices)])
            triangle = [previous, int(current), following]
            geometry = triangle_geometry(points, np.asarray([triangle], dtype=int))
            if geometry["signed_area"][0] <= _area_tolerance(points, np.asarray([triangle], dtype=int)):
                continue
            ear_polygon = Polygon(
                points[np.asarray(triangle, dtype=int)]
            )
            if (
                not ear_polygon.is_valid
                or not covered_polygon.covers(ear_polygon)
            ):
                continue
            if any(_point_in_triangle(points[value], points[np.asarray(triangle, dtype=int)]) for value in vertices if value not in triangle):
                continue
            diagonal = tuple(sorted((previous, following)))
            if (
                diagonal not in existing
                and edge_allowed is not None
                and not bool(edge_allowed(diagonal))
            ):
                continue
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


def _restricted_edge_violation_records(
    state: _State,
    *,
    topology: Any | None = None,
) -> list[dict[str, Any]]:
    return restricted_edge_violation_records(
        state.triangles,
        state.lineage,
        state.restricted_lineage_edges,
        edge_to_triangles=(
            None if topology is None else topology.edge_to_triangles
        ),
    )


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
    area_transition = _area_transition_tail(geometry["area"], topology)
    restricted_violations = _restricted_edge_violation_records(
        state,
        topology=topology,
    )
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
        "area_transition_count_above_0_50": int(area_transition["count_above_0_50"]),
        "area_transition_maximum": float(area_transition["maximum"]),
        "connected_component_count": int(len(topology.connected_component_sizes)),
        "nonmanifold_edge_count": int(len(topology.nonmanifold_edges)),
        "nonpositive_signed_area_count": int(np.count_nonzero(geometry["signed_area"] <= _area_tolerance(state.points, state.triangles))),
        "singly_connected_triangle_count": int(np.count_nonzero(topology.triangle_neighbor_count == 1)),
        "boundary_degree_anomaly_count": int(boundary_audit["degree_anomaly_count"]),
        "boundary_component_count": int(boundary_audit["component_count"]),
        "protected_edge_not_boundary_count": int(protected_not_boundary),
        "restricted_edge_violation_count": int(
            len(restricted_violations)
        ),
    }


def _summary_from(value: dict[str, Any]) -> dict[str, Any]:
    return dict(value)


def _area_transition_tail(areas: np.ndarray, topology: Any) -> dict[str, Any]:
    values: list[float] = []
    area_values = np.asarray(areas, dtype=float)
    for attached in topology.edge_to_triangles.values():
        if len(attached) != 2:
            continue
        first, second = map(int, attached)
        denominator = max(float(area_values[first]), float(area_values[second]), 1.0e-30)
        values.append(abs(float(area_values[first]) - float(area_values[second])) / denominator)
    array = np.asarray(values, dtype=float)
    return {
        "count_above_0_50": int(np.count_nonzero(array > 0.50)),
        "maximum": float(np.max(array)) if len(array) else 0.0,
    }


def _nonregression(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    purpose: str,
    max_l_over_h_count_increase: int = 0,
) -> bool:
    topology_ok = bool(
        after["boundary_degree_anomaly_count"] <= before["boundary_degree_anomaly_count"]
        and after["boundary_component_count"] == before["boundary_component_count"]
        and after["protected_edge_not_boundary_count"] <= before["protected_edge_not_boundary_count"]
    )
    if purpose == "valence":
        # The FVCOM connectivity cap is a hard structural gate.  A perfectly
        # regular nine-spoke star cannot be reduced to valence eight without
        # changing its quality distribution. Benchmark-first conditioning
        # therefore gates on structure and valence improvement, not regional
        # refinement metrics such as angle tails, area transition, or L/h.
        before_valence = (
            int(before["count_valence_above_limit"]),
            int(before["valence_excess_sum"]),
            int(before["maximum_valence"]),
        )
        after_valence = (
            int(after["count_valence_above_limit"]),
            int(after["valence_excess_sum"]),
            int(after["maximum_valence"]),
        )
        return bool(topology_ok and after_valence < before_valence)
    if purpose == "thin":
        defect_improved = bool(
            after["superthin_triangle_count"] < before["superthin_triangle_count"]
            or after["thin_triangle_count"] < before["thin_triangle_count"]
            or after["thin_severity_sum"] < before["thin_severity_sum"] - 1.0e-10
        )
        before_valence = (
            int(before["count_valence_above_limit"]),
            int(before["valence_excess_sum"]),
            int(before["maximum_valence"]),
        )
        after_valence = (
            int(after["count_valence_above_limit"]),
            int(after["valence_excess_sum"]),
            int(after["maximum_valence"]),
        )
        return bool(
            defect_improved
            and topology_ok
            and after["superthin_triangle_count"] <= before["superthin_triangle_count"]
            and after_valence <= before_valence
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
    allow_new_singly_connected: bool = False,
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
    restricted_violations = _restricted_edge_violation_records(
        state,
        topology=topology,
    )
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
        "restricted_edge_violation_count": int(
            len(restricted_violations)
        ),
        "restricted_edge_violations": restricted_violations[:100],
        "constraint_integrity": integrity,
    }
    ok = bool(
        positive
        and report["all_protected_edges_present"]
        and report["open_boundary_ordered"]
        and report["connected_component_count"] == int(initial_components)
        and report["nonmanifold_edge_count"] == 0
        and report["unused_node_count"] == 0
        and (allow_new_singly_connected or report["new_singly_connected_triangle_count"] == 0)
        and report["boundary_traversable"]
        and report["protected_edge_not_boundary_count"] <= int(state.initial_protected_not_boundary_count)
        and report["duplicate_triangle_count"] == 0
        and report["repeated_node_triangle_count"] == 0
        and report["chain_node_range_ok"]
        and report["chain_unique_nodes"]
        and report["missing_hard_anchor_count"] == 0
        and report["moved_hard_anchor_count"] == 0
        and report["restricted_edge_violation_count"] == 0
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
        allow_new_singly_connected=(
            str(config.profile_name) == "minimal-topology-v1"
            or str(config.systematic_gate_scope) == "loop-end"
        ),
    )
    if (
        str(config.thin_repair_profile) == "systematic-v5"
        and not bool(
            config.systematic_v5_enable_boundary_window_fallback
        )
    ):
        source_open = tuple(
            map(
                int,
                np.asarray(state.source_open_nodes, dtype=int),
            )
        )
        delivered_open = _open_boundary_lineage_sequence(state)
        unchanged = bool(delivered_open == source_open)
        invariants["open_boundary_membership_unchanged"] = unchanged
        invariants["source_open_boundary_count"] = int(
            len(source_open)
        )
        invariants["delivered_open_boundary_count"] = int(
            len(delivered_open)
        )
        ok = bool(ok and unchanged)
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
        "open_boundary_membership_unchanged": report.get(
            "open_boundary_membership_unchanged"
        ),
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
        "restricted_edges": report.get(
            "restricted_edge_violation_count",
            0,
        ),
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
    state.source_open_nodes = snapshot.source_open_nodes
    state.source_kinds = snapshot.source_kinds
    state.source_hard_anchor_lineage = snapshot.source_hard_anchor_lineage
    state.target_sampler = snapshot.target_sampler
    state.initial_domain_area_m2 = snapshot.initial_domain_area_m2
    state.initial_boundary_component_count = snapshot.initial_boundary_component_count
    state.initial_boundary_degree_anomaly_count = snapshot.initial_boundary_degree_anomaly_count
    state.initial_singly_connected_triangle_count = snapshot.initial_singly_connected_triangle_count
    state.initial_protected_not_boundary_count = snapshot.initial_protected_not_boundary_count
    state.restricted_lineage_edges = set(snapshot.restricted_lineage_edges)
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


def _source_chain_curve(state: _State, chain_index: int) -> tuple[np.ndarray, np.ndarray, float] | None:
    if not 0 <= int(chain_index) < len(state.source_chains):
        return None
    source_chain = list(map(int, state.source_chains[int(chain_index)]))
    if len(source_chain) < 2:
        return None
    coordinates = state.source_points[np.asarray([*source_chain, source_chain[0]], dtype=int)]
    lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 1.0e-12:
        return None
    return coordinates, cumulative, total


def _project_to_source_arc(
    state: _State,
    chain_index: int,
    point: np.ndarray,
) -> tuple[float, np.ndarray, float] | None:
    curve = _source_chain_curve(state, chain_index)
    if curve is None:
        return None
    coordinates, cumulative, total = curve
    best: tuple[float, float, np.ndarray] | None = None
    for index, (left, right) in enumerate(zip(coordinates[:-1], coordinates[1:])):
        vector = right - left
        denominator = float(np.dot(vector, vector))
        fraction = (
            0.0
            if denominator <= 1.0e-20
            else float(np.clip(np.dot(np.asarray(point, dtype=float) - left, vector) / denominator, 0.0, 1.0))
        )
        projection = left + fraction * vector
        distance = float(np.linalg.norm(np.asarray(point, dtype=float) - projection))
        s = float(cumulative[index] + fraction * (cumulative[index + 1] - cumulative[index]))
        candidate = (distance, s, projection)
        if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
            best = candidate
    if best is None:
        return None
    return float(best[1] % total), best[2].copy(), float(best[0])


def _interpolate_source_arc(state: _State, chain_index: int, s: float) -> np.ndarray | None:
    curve = _source_chain_curve(state, chain_index)
    if curve is None:
        return None
    coordinates, cumulative, total = curve
    value = float(s % total)
    index = int(np.searchsorted(cumulative, value, side="right") - 1)
    index = min(max(index, 0), len(coordinates) - 2)
    span = float(cumulative[index + 1] - cumulative[index])
    fraction = 0.0 if span <= 1.0e-20 else (value - float(cumulative[index])) / span
    return coordinates[index] + float(fraction) * (coordinates[index + 1] - coordinates[index])


def _source_arc_midpoint(
    state: _State,
    chain_index: int,
    first: int,
    second: int,
) -> np.ndarray | None:
    chain = state.chains[int(chain_index)]
    edge_pos = _find_chain_edge([chain], (int(first), int(second)))
    if edge_pos is None:
        return None
    _, position = edge_pos
    left = int(chain[position])
    right = int(chain[(position + 1) % len(chain)])
    left_projection = _project_to_source_arc(state, chain_index, state.points[left])
    right_projection = _project_to_source_arc(state, chain_index, state.points[right])
    curve = _source_chain_curve(state, chain_index)
    if left_projection is None or right_projection is None or curve is None:
        return None
    total = float(curve[2])
    delta = (float(right_projection[0]) - float(left_projection[0])) % total
    return _interpolate_source_arc(state, chain_index, float(left_projection[0]) + 0.5 * delta)


def _target_equalized_source_arc_positions(
    state: _State,
    chain_index: int,
    nodes: list[int],
) -> list[np.ndarray] | None:
    if len(nodes) < 3:
        return None
    curve = _source_chain_curve(state, chain_index)
    left = _project_to_source_arc(state, chain_index, state.points[int(nodes[0])])
    right = _project_to_source_arc(state, chain_index, state.points[int(nodes[-1])])
    if curve is None or left is None or right is None:
        return None
    total = float(curve[2])
    delta = (float(right[0]) - float(left[0])) % total
    if delta <= 1.0e-9 or delta >= 0.90 * total:
        return None
    sample_count = 129
    fractions = np.linspace(0.0, 1.0, sample_count)
    h_left = max(float(state.targets[int(nodes[0])]), 1.0e-12)
    h_right = max(float(state.targets[int(nodes[-1])]), 1.0e-12)
    h = (1.0 - fractions) * h_left + fractions * h_right
    ds = delta / float(sample_count - 1)
    density = 1.0 / np.maximum(h, 1.0e-12)
    metric = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[:-1] + density[1:]) * ds)]
    )
    if metric[-1] <= 1.0e-20:
        return None
    output: list[np.ndarray] = []
    count = len(nodes) - 2
    for rank in range(1, count + 1):
        target_metric = float(metric[-1]) * float(rank) / float(count + 1)
        fraction = float(np.interp(target_metric, metric, fractions))
        point = _interpolate_source_arc(
            state,
            chain_index,
            float(left[0]) + fraction * delta,
        )
        if point is None:
            return None
        output.append(np.asarray(point, dtype=float))
    return output


def _source_arc_fractional_move(
    state: _State,
    chain_index: int,
    current_point: np.ndarray,
    target_point: np.ndarray,
    fraction: float,
) -> np.ndarray | None:
    current = _project_to_source_arc(state, chain_index, current_point)
    target = _project_to_source_arc(state, chain_index, target_point)
    curve = _source_chain_curve(state, chain_index)
    if current is None or target is None or curve is None:
        return None
    total = float(curve[2])
    delta = float(target[0]) - float(current[0])
    if delta > 0.5 * total:
        delta -= total
    elif delta < -0.5 * total:
        delta += total
    return _interpolate_source_arc(
        state,
        chain_index,
        float(current[0]) + float(np.clip(fraction, 0.0, 1.0)) * delta,
    )


def _maximum_boundary_source_arc_deviation(state: _State) -> float:
    maximum = 0.0
    for chain_index, chain in enumerate(state.chains):
        for node in chain:
            lineage = int(state.lineage[int(node)])
            if (
                0 <= lineage < len(state.source_points)
                and np.array_equal(state.points[int(node)], state.source_points[lineage])
            ):
                continue
            projection = _project_to_source_arc(state, chain_index, state.points[int(node)])
            if projection is None:
                return float("inf")
            maximum = max(maximum, float(projection[2]))
    return float(maximum)


def _boundary_loops_simple(state: _State) -> bool:
    for chain in state.chains:
        if len(chain) < 3:
            return False
        coordinates = state.points[np.asarray([*chain, chain[0]], dtype=int)]
        line = LineString(coordinates)
        if not line.is_valid or not line.is_simple:
            return False
    return True


def _chain_pair_clearance(state: _State, first: int, second: int) -> float:
    if not (0 <= int(first) < len(state.chains) and 0 <= int(second) < len(state.chains)):
        return float("nan")
    left = state.chains[int(first)]
    right = state.chains[int(second)]
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    left_line = LineString(state.points[np.asarray([*left, left[0]], dtype=int)])
    right_line = LineString(state.points[np.asarray([*right, right[0]], dtype=int)])
    return float(left_line.distance(right_line))


def _component_passage_clearance(state: _State, component: dict[str, Any]) -> float | None:
    chain_ids = list(map(int, component.get("boundary_chain_ids", [])))
    if len(chain_ids) < 2:
        return None
    values = [
        _chain_pair_clearance(state, first, second)
        for first, second in combinations(chain_ids, 2)
    ]
    finite = [value for value in values if np.isfinite(value)]
    return float(min(finite)) if finite else None


def _passage_clearance_inventory(
    state: _State,
    config: AggressiveConditioningConfig,
) -> dict[tuple[int, int], float]:
    output: dict[tuple[int, int], float] = {}
    for component in _inventory_superthin_components(state, config):
        chain_ids = list(map(int, component.get("boundary_chain_ids", [])))
        if len(chain_ids) < 2:
            continue
        for first, second in combinations(chain_ids, 2):
            pair = tuple(sorted((int(first), int(second))))
            value = _chain_pair_clearance(state, *pair)
            if np.isfinite(value):
                output[pair] = float(value)
    return output


def _obc_remap_manifest(state: _State) -> dict[str, Any]:
    source_open = list(map(int, np.asarray(state.source_open_nodes, dtype=int)))
    source_set = set(source_open)
    delivered = list(map(int, np.asarray(state.open_nodes, dtype=int)))
    delivered_lineage = [int(state.lineage[node]) for node in delivered]
    source_positions: dict[int, tuple[int, float, float]] = {}
    for source_node in source_open:
        for chain_index, chain in enumerate(state.source_chains):
            if int(source_node) in set(map(int, chain)):
                projection = _project_to_source_arc(
                    state,
                    chain_index,
                    state.source_points[int(source_node)],
                )
                curve = _source_chain_curve(state, chain_index)
                if projection is not None and curve is not None:
                    source_positions[int(source_node)] = (
                        int(chain_index),
                        float(projection[0]),
                        float(curve[2]),
                    )
                break
    entries: list[dict[str, Any]] = []
    for order, node in enumerate(delivered):
        lineage = int(state.lineage[node])
        membership = _find_chain_node(state.chains, int(node))
        chain_index = int(membership[0]) if membership is not None else -1
        projection = (
            _project_to_source_arc(state, chain_index, state.points[node])
            if chain_index >= 0
            else None
        )
        source_coordinate = (
            state.source_points[lineage]
            if 0 <= lineage < len(state.source_points)
            else None
        )
        moved = bool(
            source_coordinate is not None
            and not np.allclose(state.points[node], source_coordinate, atol=1.0e-9, rtol=0.0)
        )
        if lineage in source_set:
            status = "slid" if moved else "retained"
        elif lineage >= 0:
            status = "redistributed"
        else:
            status = "inserted"
        candidates = [
            (abs(float(source_s) - float(projection[0])), source_node)
            for source_node, (source_chain, source_s, _) in source_positions.items()
            if projection is not None and int(source_chain) == chain_index
        ]
        nearest = [int(value[1]) for value in sorted(candidates)[:2]]
        entries.append(
            {
                "delivered_order_zero_based": int(order),
                "delivered_node_index_zero_based": int(node),
                "status": status,
                "source_node_lineage": lineage if lineage >= 0 else None,
                "bracketing_original_obc_lineage": nearest,
                "constraint_chain_id": chain_index,
                "source_arc_position_m": float(projection[0]) if projection is not None else None,
                "source_arc_fraction": (
                    float(projection[0] / _source_chain_curve(state, chain_index)[2])
                    if projection is not None and _source_chain_curve(state, chain_index) is not None
                    else None
                ),
                "coordinate_xy": [float(state.points[node, 0]), float(state.points[node, 1])],
            }
        )
    compatible = bool(
        len(delivered) == len(source_open)
        and delivered_lineage == source_open
        and all(entry["status"] == "retained" for entry in entries)
    )
    return {
        "schema_version": "fvcom_obc_remap_manifest_v1",
        "original_obc_count": int(len(source_open)),
        "delivered_obc_count": int(len(delivered)),
        "obc_forcing_compatible": compatible,
        "forcing_invalidation_required": not compatible,
        "orientation_preserved": bool(
            not source_open
            or not delivered_lineage
            or (
                delivered_lineage[0] == source_open[0]
                and delivered_lineage[-1] == source_open[-1]
            )
        ),
        "removed_original_obc_lineage": [
            int(node) for node in source_open if int(node) not in set(delivered_lineage)
        ],
        "delivered_nodes": entries,
    }


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
