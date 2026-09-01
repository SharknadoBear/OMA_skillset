"""Pure-Python OceanMesh-style constrained Delaunay refinement."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from shapely import contains_xy
from shapely.geometry import MultiLineString, Point
from shapely.ops import linemerge

from .boundary import BoundaryNodes
from .metrics import build_edge_topology, constraint_integrity, triangle_geometry
from .node_budget import DEFAULT_MAX_INTERIOR_POINTS
from .local_topology import AggressiveConditioningConfig, condition_mesh_aggressive
from .systematic_v5 import SystematicV5LoopConfig, run_systematic_v5_loop
from .systematic_v6 import SystematicV6LoopConfig, run_systematic_v6_loop
from .systematic_v6_policy import (
    loop_policy_overrides,
    topology_policy_overrides,
)
from .projection import unproject_points
from .regional_conditioning import (
    AreaTransitionRelaxConfig,
    SpringRelaxConfig,
    ThinTriangleRepairConfig,
    relax_mesh_area_transitions,
    relax_mesh_spring,
    repair_thin_triangles,
)
from .size_field import SizeField, boundary_front_seed_points


@dataclass(frozen=True)
class MeshConfig:
    max_constraint_iterations: int = 8
    refine_iterations: int = 3
    smooth_iterations: int = 8
    max_interior_points: int = DEFAULT_MAX_INTERIOR_POINTS
    max_refine_insertions_per_iter: int = 1500
    size_overrun_factor: float = 1.55
    min_angle_refine_deg: float = 24.0
    adaptive_seed: bool = False
    boundary_front_seeding: bool = True
    regional_spring_relaxation: bool = True
    spring_relax_iterations: int = 20
    spring_relax_quality_threshold: float = 0.40
    spring_relax_min_angle_deg: float = 28.0
    spring_relax_ring_layers: int = 3
    spring_relax_shape_weight: float = 0.20
    thin_triangle_repair: bool = True
    thin_repair_profile: str = "guarded-v1"
    systematic_v3_obc_policy: str = "redistribute"
    systematic_v5_total_iterations: int = 1000
    systematic_v5_max_cycles: int = 6
    systematic_v5_max_burst: int = 250
    systematic_v5_thin_trigger: int = 25
    systematic_v5_checkpoint_interval: int = 10
    systematic_v5_wall_time_s: float = 21600.0
    systematic_v5_connectivity_restriction: bool = True
    systematic_v5_max_connectivity_transactions: int = 32
    systematic_v6_total_iterations: int = 1000
    systematic_v6_max_cycles: int = 12
    systematic_v6_max_closure_rounds: int = 8
    systematic_v6_max_burst: int = 100
    systematic_v6_checkpoint_interval: int = 10
    systematic_v6_wall_time_s: float = 28800.0
    systematic_v6_final_audit_reserve_s: float = 3600.0
    systematic_v6_gate_policy: str = "strict-v6"
    systematic_v6_passage_removal: bool = False
    thin_triangle_quality_threshold: float = 0.25
    thin_triangle_min_angle_deg: float = 20.0
    thin_triangle_max_passes: int = 2
    thin_triangle_max_flips: int = 200
    thin_triangle_max_insertions: int = 50
    area_transition_relaxation: bool = True
    area_transition_max_patches: int = 12
    area_transition_area_change_threshold: float = 0.50
    area_transition_target_gradient_threshold: float = 0.10
    conditioning_profile: str = "auto"
    minimal_conditioning_wall_time_s: float = 3_600.0
    aggressive_conditioning_rounds: int = 4
    aggressive_boundary_edit_policy: str = "kind-aware-envelope"
    aggressive_max_prunes_per_round: int = 500
    aggressive_max_valence_repairs_per_round: int = 500


@dataclass
class MeshResult:
    nodes_xy: np.ndarray
    nodes_lonlat: np.ndarray
    triangles: np.ndarray
    open_boundary_nodes: np.ndarray
    boundary_node_count: int
    fixed_node_mask: np.ndarray
    target_spacing_m: np.ndarray
    constraint_chains: list[list[int]]
    boundary_kinds: list[str]
    hard_anchor_mask: np.ndarray
    node_lineage: np.ndarray
    report: dict[str, Any]


ProgressCallback = Callable[[str, float, dict[str, Any] | None], None]
SizeSamplerXY = Callable[[np.ndarray], np.ndarray]


def generate_mesh(
    boundary: BoundaryNodes,
    size_field: SizeField,
    config: MeshConfig,
    progress_callback: ProgressCallback | None = None,
    size_sampler_xy: SizeSamplerXY | None = None,
) -> MeshResult:
    """Generate a constrained Delaunay mesh with boundary midpoint recovery."""
    declared_open_boundaries = boundary.open_boundaries or []
    explicit_named_multi = bool(
        len(declared_open_boundaries) > 1
        and any(
            not str(chain.chain_id).startswith("obc_")
            for chain in declared_open_boundaries
        )
    )
    merged_open_boundary = (
        linemerge(boundary.open_boundary_xy)
        if isinstance(boundary.open_boundary_xy, MultiLineString)
        else boundary.open_boundary_xy
    )
    multi_geometry = bool(
        isinstance(merged_open_boundary, MultiLineString)
        and len(merged_open_boundary.geoms) > 1
    )
    cyclic_unsupported = bool(
        any(chain.cyclic for chain in declared_open_boundaries)
        and (
            boundary.adaptive_resolution
            or any(
                not str(chain.chain_id).startswith("obc_")
                for chain in declared_open_boundaries
            )
        )
    )
    if (
        explicit_named_multi
        or multi_geometry
        or cyclic_unsupported
    ):
        raise ValueError(
            "The production clean-room mesher currently supports at most one "
            "noncyclic open-boundary chain. Use the research Gmsh route for "
            "multiple or cyclic OBC contracts."
        )
    if str(config.thin_repair_profile) not in {
        "guarded-v1",
        "systematic-v2",
        "systematic-v3",
        "systematic-v5",
        "systematic-v6",
        "none",
    }:
        raise ValueError(
            "thin_repair_profile must be guarded-v1, systematic-v2, "
            "systematic-v3, systematic-v5, systematic-v6, or none"
        )
    if str(config.systematic_v3_obc_policy) not in {"preserve", "redistribute"}:
        raise ValueError("systematic_v3_obc_policy must be preserve or redistribute")
    def _sample_size_field_targets(
        sample_points_xy: np.ndarray,
    ) -> np.ndarray:
        sample_points = np.asarray(sample_points_xy, dtype=float)
        if size_sampler_xy is not None:
            values = np.asarray(size_sampler_xy(sample_points), dtype=float)
        else:
            sampled_lonlat = unproject_points(
                sample_points,
                boundary.projection,
            )
            values = np.asarray(
                size_field.sample(
                    sampled_lonlat[:, 0],
                    sampled_lonlat[:, 1],
                ),
                dtype=float,
            )
        values = values.reshape(-1)
        if (
            len(values) != len(sample_points)
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
        ):
            raise ValueError(
                "size sampler must return one finite positive target per point"
            )
        return values

    _progress(progress_callback, "seed_boundary_nodes", 0.0, {"boundary_node_count": int(len(boundary.xy))})
    points = [tuple(xy) for xy in boundary.xy]
    kinds = list(boundary.kinds)
    point_targets = [float(value) for value in boundary.target_spacing_m]
    hard_anchors = list(
        np.asarray(
            boundary.hard_anchor_mask if boundary.hard_anchor_mask is not None else np.zeros(len(boundary.xy), dtype=bool),
            dtype=bool,
        )
    )
    chains = [list(chain) for chain in boundary.constraint_chains]

    interior = _interior_seed_points(
        boundary,
        size_field,
        config,
        _sample_size_field_targets,
    )
    _progress(progress_callback, "seed_interior_points", 0.08, {"interior_seed_count": int(len(interior))})
    for xy in interior:
        points.append((float(xy[0]), float(xy[1])))
        kinds.append("interior")
        point_targets.append(float("nan"))
        hard_anchors.append(False)

    constraint_report: dict[str, Any] = {}
    triangles = np.empty((0, 3), dtype=int)
    for iteration in range(config.max_constraint_iterations + 1):
        arr = np.asarray(points, dtype=float)
        triangles = _triangulate_filtered(arr, boundary)
        missing = _missing_constraint_edges(triangles, chains)
        constraint_report = {
            "iterations": int(iteration),
            "missing_constraint_edge_count": int(len(missing)),
            "boundary_constraint_recovered": bool(len(missing) == 0),
        }
        _progress(
            progress_callback,
            "recover_boundary_constraints",
            0.10 + 0.30 * min(iteration + 1, config.max_constraint_iterations + 1) / max(config.max_constraint_iterations + 1, 1),
            constraint_report,
        )
        if not missing:
            break
        pending = sorted(missing[: max(1, 2_000)], key=lambda item: (item[0], item[1]), reverse=True)
        for chain_idx, edge_pos in pending:
            chain = chains[chain_idx]
            a = chain[edge_pos]
            b = chain[(edge_pos + 1) % len(chain)]
            midpoint = 0.5 * (arr[a] + arr[b])
            kind = kinds[a] if kinds[a] == kinds[b] else "land"
            new_idx = len(points)
            points.append((float(midpoint[0]), float(midpoint[1])))
            kinds.append(kind)
            point_targets.append(float(0.5 * (point_targets[a] + point_targets[b])))
            hard_anchors.append(False)
            chain.insert(edge_pos + 1, new_idx)
    points_arr = np.asarray(points, dtype=float)
    fixed_mask = np.asarray([kind != "interior" for kind in kinds], dtype=bool)

    for refine_iter in range(config.refine_iterations):
        triangles = _triangulate_filtered(points_arr, boundary)
        additions = _refinement_points(
            points_arr,
            triangles,
            boundary,
            size_field,
            config,
            _sample_size_field_targets,
        )
        _progress(
            progress_callback,
            "refine_triangles",
            0.45 + 0.30 * (refine_iter + 1) / max(config.refine_iterations, 1),
            {
                "iteration": int(refine_iter + 1),
                "total_iterations": int(config.refine_iterations),
                "triangle_count": int(len(triangles)),
                "added_node_count": int(len(additions)),
            },
        )
        if not len(additions):
            break
        points_arr = np.vstack([points_arr, additions])
        fixed_mask = np.concatenate([fixed_mask, np.zeros(len(additions), dtype=bool)])
        kinds.extend(["interior"] * len(additions))
        point_targets.extend([float("nan")] * len(additions))
        hard_anchors.extend([False] * len(additions))

    triangles = _triangulate_filtered(points_arr, boundary)
    _progress(progress_callback, "smooth_interior_nodes", 0.82, {"node_count": int(len(points_arr)), "triangle_count": int(len(triangles))})
    points_arr = _smooth_interior(points_arr, triangles, fixed_mask, boundary, iterations=config.smooth_iterations)
    # Refinement insertions and the final Delaunay rebuild can invalidate a
    # previously recovered concave-boundary edge.  Re-run midpoint recovery
    # after smoothing so the quality report describes the delivered mesh, not
    # the pre-refinement triangulation.
    initial_constraint_report = dict(constraint_report)
    points = [tuple(xy) for xy in points_arr]
    final_iteration = 0
    for final_iteration in range(config.max_constraint_iterations + 1):
        points_arr = np.asarray(points, dtype=float)
        triangles = _triangulate_filtered(points_arr, boundary)
        missing = _missing_constraint_edges(triangles, chains)
        if not missing:
            break
        pending = sorted(missing[: max(1, 2_000)], key=lambda item: (item[0], item[1]), reverse=True)
        for chain_idx, edge_pos in pending:
            chain = chains[chain_idx]
            a = chain[edge_pos]
            b = chain[(edge_pos + 1) % len(chain)]
            midpoint = 0.5 * (points_arr[a] + points_arr[b])
            kind = kinds[a] if kinds[a] == kinds[b] else "land"
            new_idx = len(points)
            points.append((float(midpoint[0]), float(midpoint[1])))
            kinds.append(kind)
            point_targets.append(float(0.5 * (point_targets[a] + point_targets[b])))
            hard_anchors.append(False)
            chain.insert(edge_pos + 1, new_idx)
    points_arr = np.asarray(points, dtype=float)
    triangles = _triangulate_filtered(points_arr, boundary)
    final_missing = _missing_constraint_edges(triangles, chains)
    constraint_report = {
        "iterations": int(initial_constraint_report.get("iterations", 0)) + int(final_iteration),
        "initial_iterations": int(initial_constraint_report.get("iterations", 0)),
        "final_iterations": int(final_iteration),
        "missing_constraint_edge_count": int(len(final_missing)),
        "boundary_constraint_recovered": bool(len(final_missing) == 0),
        "final_recovery_applied": True,
    }
    _progress(progress_callback, "finalize_boundary_constraints", 0.88, constraint_report)
    triangles = _orient_ccw(points_arr, triangles)
    fixed_mask = np.asarray([kind != "interior" for kind in kinds], dtype=bool)
    open_nodes = _ordered_boundary_kind_group(chains[0] if chains else [], kinds, "open")

    # Guarded generation-time conditioning starts only after protected-edge
    # recovery.  Neither stage is allowed to invoke a global retriangulation.
    preconditioning = {
        "points": points_arr.copy(),
        "triangles": triangles.copy(),
        "fixed": fixed_mask.copy(),
        "chains": [chain.copy() for chain in chains],
        "open_nodes": np.asarray(open_nodes, dtype=int),
        "kinds": kinds.copy(),
        "point_targets": np.asarray(point_targets, dtype=float).copy(),
        "hard_anchors": np.asarray(hard_anchors, dtype=bool).copy(),
    }
    sampled_targets = _sample_size_field_targets(points_arr)
    stored_targets = np.asarray(point_targets, dtype=float)
    conditioning_targets = np.where(np.isfinite(stored_targets) & (stored_targets > 0.0), stored_targets, sampled_targets)

    def _sample_conditioning_targets(sample_points_xy: np.ndarray) -> np.ndarray:
        """Sample Eulerian interior targets while retaining explicit fixed-boundary targets."""
        sample_points_xy = np.asarray(sample_points_xy, dtype=float)
        values = _sample_size_field_targets(sample_points_xy)
        node_aligned = bool(
            len(values) == len(fixed_mask)
            and len(conditioning_targets) == len(fixed_mask)
            and np.allclose(sample_points_xy[fixed_mask], points_arr[fixed_mask], rtol=0.0, atol=1.0e-10)
        )
        if node_aligned:
            values = np.asarray(values, dtype=float).copy()
            values[fixed_mask] = np.asarray(conditioning_targets, dtype=float)[fixed_mask]
        return values
    effective_conditioning_profile = str(config.conditioning_profile)
    if effective_conditioning_profile == "auto":
        effective_conditioning_profile = "minimal-topology-v1"
    if effective_conditioning_profile not in {
        "minimal-topology-v1",
        "guarded-v1",
        "aggressive-local-v2",
        "none",
    }:
        raise ValueError(
            "conditioning_profile must be auto, minimal-topology-v1, "
            "guarded-v1, aggressive-local-v2, or none"
        )
    minimal_profile = effective_conditioning_profile == "minimal-topology-v1"
    spring_config = SpringRelaxConfig(
        enabled=bool(config.regional_spring_relaxation) and not minimal_profile,
        quality_threshold=float(config.spring_relax_quality_threshold),
        min_angle_deg=float(config.spring_relax_min_angle_deg),
        ring_layers=int(config.spring_relax_ring_layers),
        iterations=int(config.spring_relax_iterations),
        shape_weight=float(config.spring_relax_shape_weight),
    )
    _progress(progress_callback, "regional_spring_relaxation", 0.91, {"enabled": spring_config.enabled})
    spring_result = relax_mesh_spring(
        points_arr,
        triangles,
        fixed_mask,
        target_spacing_m=conditioning_targets,
        constraint_chains=chains,
        open_boundary_nodes_zero_based=np.asarray(open_nodes, dtype=int),
        config=spring_config,
    )
    points_arr = spring_result.nodes_xy

    thin_config = ThinTriangleRepairConfig(
        enabled=bool(config.thin_triangle_repair) and not minimal_profile,
        quality_threshold=float(config.thin_triangle_quality_threshold),
        min_angle_deg=float(config.thin_triangle_min_angle_deg),
        max_passes=int(config.thin_triangle_max_passes),
        max_flips=int(config.thin_triangle_max_flips),
        max_insertions=int(config.thin_triangle_max_insertions),
        relaxation_config=SpringRelaxConfig(
            enabled=bool(config.regional_spring_relaxation) and not minimal_profile,
            quality_threshold=float(config.spring_relax_quality_threshold),
            min_angle_deg=float(config.spring_relax_min_angle_deg),
            ring_layers=int(config.spring_relax_ring_layers),
            iterations=min(max(int(config.spring_relax_iterations), 0), 15),
            shape_weight=float(config.spring_relax_shape_weight),
        ),
    )
    _progress(progress_callback, "thin_triangle_repair", 0.95, {"enabled": thin_config.enabled})
    thin_result = repair_thin_triangles(
        points_arr,
        triangles,
        fixed_mask,
        chains,
        np.asarray(open_nodes, dtype=int),
        target_spacing_m=conditioning_targets,
        config=thin_config,
    )
    points_arr = thin_result.nodes_xy
    triangles = _orient_ccw(points_arr, thin_result.triangles)
    fixed_mask = thin_result.fixed_node_mask
    chains = [chain.copy() for chain in thin_result.constraint_chains]
    open_nodes = thin_result.open_boundary_nodes_zero_based.tolist()
    conditioning_targets = thin_result.target_spacing_m
    if len(points_arr) > len(kinds):
        kinds.extend(["interior"] * (len(points_arr) - len(kinds)))
    if len(points_arr) > len(hard_anchors):
        hard_anchors.extend([False] * (len(points_arr) - len(hard_anchors)))
    point_targets = conditioning_targets.tolist()

    aggressive_result = None
    node_lineage = np.arange(len(points_arr), dtype=int)
    restricted_edges_current: set[tuple[int, int]] = set()
    if effective_conditioning_profile in {
        "aggressive-local-v2",
        "minimal-topology-v1",
    }:
        _progress(progress_callback, "aggressive_local_topology", 0.965, {"profile": effective_conditioning_profile})
        aggressive_result = condition_mesh_aggressive(
            points_arr,
            triangles,
            fixed_mask,
            chains,
            np.asarray(open_nodes, dtype=int),
            target_spacing_m=conditioning_targets,
            boundary_kinds=kinds,
            hard_anchor_mask=np.asarray(hard_anchors, dtype=bool),
            target_spacing_sampler=_sample_conditioning_targets,
            config=AggressiveConditioningConfig(
                enabled=True,
                profile_name=effective_conditioning_profile,
                stage_order=(
                    "valence-before-thin"
                    if minimal_profile
                    else "thin-before-valence"
                ),
                enable_pruning=not minimal_profile,
                enable_thin_repair=(
                    True
                    if minimal_profile
                    else str(config.thin_repair_profile) != "none"
                ),
                thin_repair_profile=(
                    "guarded-v1"
                    if minimal_profile or str(config.thin_repair_profile) == "none"
                    else (
                        "systematic-v5"
                        if str(config.thin_repair_profile) == "systematic-v6"
                        else str(config.thin_repair_profile)
                    )
                ),
                systematic_v3_obc_policy=str(config.systematic_v3_obc_policy),
                systematic_v5_enable_connectivity_restriction=bool(
                    config.systematic_v5_connectivity_restriction
                ),
                systematic_v5_max_connectivity_transactions_per_round=int(
                    config.systematic_v5_max_connectivity_transactions
                ),
                max_rounds=int(config.aggressive_conditioning_rounds),
                boundary_edit_policy=(
                    "none"
                    if minimal_profile
                    else str(config.aggressive_boundary_edit_policy)
                ),
                max_boundary_edits_per_round=(0 if minimal_profile else 25),
                enable_fixed_hard_fan_arc_refinement=bool(minimal_profile),
                max_fixed_hard_fan_arc_refinements_per_round=(
                    8 if minimal_profile else 0
                ),
                max_boundary_welds_per_round=(0 if minimal_profile else 25),
                max_boundary_ear_removals_per_round=(0 if minimal_profile else 25),
                max_prunes_per_round=(
                    0
                    if minimal_profile
                    else int(config.aggressive_max_prunes_per_round)
                ),
                max_valence_removals_per_round=int(config.aggressive_max_valence_repairs_per_round),
                micro_relax_cycles=(0 if minimal_profile else 3),
                deadline_monotonic_s=(
                    time.perf_counter()
                    + float(config.minimal_conditioning_wall_time_s)
                    if minimal_profile
                    else None
                ),
            ),
        )
        points_arr = aggressive_result.nodes_xy
        triangles = _orient_ccw(points_arr, aggressive_result.triangles)
        fixed_mask = aggressive_result.fixed_node_mask
        chains = [chain.copy() for chain in aggressive_result.constraint_chains]
        open_nodes = aggressive_result.open_boundary_nodes_zero_based.tolist()
        conditioning_targets = aggressive_result.target_spacing_m
        point_targets = conditioning_targets.tolist()
        kinds = aggressive_result.boundary_kinds.copy()
        hard_anchors = aggressive_result.hard_anchor_mask.tolist()
        node_lineage = aggressive_result.node_lineage.copy()
        restricted_edges_current = _restrictions_to_delivered_indices(
            aggressive_result.restricted_lineage_edges,
            aggressive_result.node_lineage,
        )

    area_transition_config = AreaTransitionRelaxConfig(
        enabled=bool(config.area_transition_relaxation) and not minimal_profile,
        max_patches=int(config.area_transition_max_patches),
        raw_area_change_threshold=float(config.area_transition_area_change_threshold),
        target_gradient_threshold=float(config.area_transition_target_gradient_threshold),
    )
    _progress(
        progress_callback,
        "area_transition_relaxation",
        0.97,
        {"enabled": area_transition_config.enabled, "max_patches": int(area_transition_config.max_patches)},
    )
    area_transition_result = relax_mesh_area_transitions(
        points_arr,
        triangles,
        fixed_mask,
        target_spacing_sampler=_sample_conditioning_targets,
        constraint_chains=chains,
        open_boundary_nodes_zero_based=np.asarray(open_nodes, dtype=int),
        config=area_transition_config,
    )
    points_arr = area_transition_result.nodes_xy
    conditioning_targets = area_transition_result.target_spacing_m
    point_targets = conditioning_targets.tolist()

    terminal_thin_result = None
    if (
        not minimal_profile
        and aggressive_result is not None
        and str(config.thin_repair_profile) in {
        "systematic-v2",
        "systematic-v3",
        "systematic-v5",
        "systematic-v6",
        }
    ):
        terminal_profile = str(config.thin_repair_profile)
        _progress(progress_callback, "terminal_systematic_thin_repair", 0.975, {"profile": terminal_profile})
        previous_lineage = np.asarray(node_lineage, dtype=int).copy()
        terminal_topology_policy = (
            topology_policy_overrides(
                str(config.systematic_v6_gate_policy)
            )
            if terminal_profile == "systematic-v6"
            else {}
        )
        terminal_config = AggressiveConditioningConfig(
            enabled=True,
            thin_repair_profile=terminal_profile,
            systematic_v3_obc_policy=str(config.systematic_v3_obc_policy),
            systematic_v5_enable_connectivity_restriction=bool(
                config.systematic_v5_connectivity_restriction
            ),
            systematic_v5_max_connectivity_transactions_per_round=int(
                config.systematic_v5_max_connectivity_transactions
            ),
            max_rounds=1,
            enable_pruning=False,
            enable_thin_repair=True,
            enable_valence_repair=bool(terminal_profile == "systematic-v6"),
            max_prunes_per_round=0,
            max_valence_removals_per_round=(
                int(config.aggressive_max_valence_repairs_per_round)
                if terminal_profile == "systematic-v6"
                else 0
            ),
            **terminal_topology_policy,
        )
        if terminal_profile == "systematic-v6":
            terminal_thin_result = run_systematic_v6_loop(
                points_arr,
                triangles,
                fixed_mask,
                chains,
                np.asarray(open_nodes, dtype=int),
                target_spacing_m=conditioning_targets,
                boundary_kinds=kinds,
                hard_anchor_mask=np.asarray(hard_anchors, dtype=bool),
                target_spacing_sampler=_sample_conditioning_targets,
                restricted_lineage_edges=restricted_edges_current,
                topology_config=terminal_config,
                loop_config=SystematicV6LoopConfig(
                    maximum_closure_rounds=int(
                        config.systematic_v6_max_closure_rounds
                    ),
                    maximum_relaxation_cycles=int(
                        config.systematic_v6_max_cycles
                    ),
                    total_relaxation_iterations=int(
                        config.systematic_v6_total_iterations
                    ),
                    maximum_burst=int(config.systematic_v6_max_burst),
                    checkpoint_interval=int(
                        config.systematic_v6_checkpoint_interval
                    ),
                    wall_clock_seconds=float(
                        config.systematic_v6_wall_time_s
                    ),
                    final_audit_reserve_seconds=float(
                        config.systematic_v6_final_audit_reserve_s
                    ),
                    passage_removal_enabled=bool(
                        config.systematic_v6_passage_removal
                    ),
                    allow_authorized_topology_delta=bool(
                        config.systematic_v6_passage_removal
                    ),
                    **loop_policy_overrides(
                        str(config.systematic_v6_gate_policy)
                    ),
                ),
            )
        elif terminal_profile == "systematic-v5":
            terminal_thin_result = run_systematic_v5_loop(
                points_arr,
                triangles,
                fixed_mask,
                chains,
                np.asarray(open_nodes, dtype=int),
                target_spacing_m=conditioning_targets,
                boundary_kinds=kinds,
                hard_anchor_mask=np.asarray(hard_anchors, dtype=bool),
                target_spacing_sampler=_sample_conditioning_targets,
                restricted_lineage_edges=restricted_edges_current,
                topology_config=terminal_config,
                loop_config=SystematicV5LoopConfig(
                    total_iterations=int(config.systematic_v5_total_iterations),
                    maximum_cycles=int(config.systematic_v5_max_cycles),
                    maximum_burst=int(config.systematic_v5_max_burst),
                    superthin_trigger=int(config.systematic_v5_thin_trigger),
                    checkpoint_interval=int(config.systematic_v5_checkpoint_interval),
                    wall_clock_seconds=float(config.systematic_v5_wall_time_s),
                ),
            )
        else:
            terminal_thin_result = condition_mesh_aggressive(
                points_arr,
                triangles,
                fixed_mask,
                chains,
                np.asarray(open_nodes, dtype=int),
                target_spacing_m=conditioning_targets,
                boundary_kinds=kinds,
                hard_anchor_mask=np.asarray(hard_anchors, dtype=bool),
                target_spacing_sampler=_sample_conditioning_targets,
                config=terminal_config,
            )
        points_arr = terminal_thin_result.nodes_xy
        triangles = _orient_ccw(points_arr, terminal_thin_result.triangles)
        fixed_mask = terminal_thin_result.fixed_node_mask
        chains = [chain.copy() for chain in terminal_thin_result.constraint_chains]
        open_nodes = terminal_thin_result.open_boundary_nodes_zero_based.tolist()
        conditioning_targets = terminal_thin_result.target_spacing_m
        point_targets = conditioning_targets.tolist()
        kinds = terminal_thin_result.boundary_kinds.copy()
        hard_anchors = terminal_thin_result.hard_anchor_mask.tolist()
        node_lineage = _compose_lineage(previous_lineage, terminal_thin_result.node_lineage)

    terminal_audit = _conditioning_terminal_audit(
        points_arr,
        triangles,
        fixed_mask,
        chains,
        np.asarray(open_nodes, dtype=int),
        preconditioning["points"],
        preconditioning["fixed"],
        node_lineage=node_lineage,
        original_hard=preconditioning["hard_anchors"],
        allow_boundary_motion=str(config.thin_repair_profile) in {"systematic-v3", "systematic-v5"},
    )
    baseline_audit = _conditioning_terminal_audit(
        preconditioning["points"],
        preconditioning["triangles"],
        preconditioning["fixed"],
        preconditioning["chains"],
        preconditioning["open_nodes"],
        preconditioning["points"],
        preconditioning["fixed"],
        node_lineage=np.arange(len(preconditioning["points"]), dtype=int),
        original_hard=preconditioning["hard_anchors"],
        allow_boundary_motion=False,
    )
    if aggressive_result is not None and not aggressive_result.report.get("fvcom_valence_gate_passed", False):
        terminal_audit["fvcom_valence_gate_passed"] = False
    else:
        terminal_audit["fvcom_valence_gate_passed"] = True
    terminal_rollback = bool(baseline_audit["passed"] and not terminal_audit["passed"])
    if terminal_rollback:
        points_arr = preconditioning["points"]
        triangles = preconditioning["triangles"]
        fixed_mask = preconditioning["fixed"]
        chains = preconditioning["chains"]
        open_nodes = preconditioning["open_nodes"].tolist()
        kinds = preconditioning["kinds"]
        point_targets = preconditioning["point_targets"].tolist()
        conditioning_targets = preconditioning["point_targets"]
        hard_anchors = preconditioning["hard_anchors"].tolist()
        node_lineage = np.arange(len(points_arr), dtype=int)
        terminal_audit = baseline_audit
    obc_remap_manifest = _generation_obc_remap_manifest(
        preconditioning,
        np.asarray(points_arr, dtype=float),
        chains,
        np.asarray(open_nodes, dtype=int),
        np.asarray(node_lineage, dtype=int),
    )
    terminal_stage_name = (
        f"{str(config.thin_repair_profile)}-terminal"
        if terminal_thin_result is not None
        else "systematic-thin-terminal-disabled"
    )
    minimal_local_debt_closed = bool(
        minimal_profile
        and aggressive_result is not None
        and aggressive_result.report.get(
            "minimal_local_debt_closed",
            False,
        )
        and terminal_audit["passed"]
        and not terminal_rollback
    )
    conditioning_report = {
        "schema_version": "fvcom_generation_conditioning_v5",
        "profile": effective_conditioning_profile,
        "requested_profile": str(config.conditioning_profile),
        "minimal_local_debt_closed": minimal_local_debt_closed,
        "stage_order": (
            [
                "spring-relaxation-disabled",
                "broad-thin-repair-disabled",
                "valence-first-local-transactions",
                "immediate-post-valence-superthin-cleanup",
                "residual-superthin-component-repair",
                "terminal-valence-and-superthin-scan",
                "area-transition-relaxation-disabled",
                "terminal-constraint-audit",
            ]
            if minimal_profile
            else [
                "spring-relax-v1",
                "thin-repair-v1",
                (
                    "aggressive-local-v2"
                    if aggressive_result is not None
                    else "aggressive-local-disabled"
                ),
                "area-transition-relax-v1",
                terminal_stage_name,
                "terminal-constraint-audit",
            ]
        ),
        "spring_relaxation": spring_result.report,
        "thin_triangle_repair": thin_result.report,
        "aggressive_local_topology": aggressive_result.report if aggressive_result is not None else {"enabled": False},
        "mesh_edit_ledger": [
            *(aggressive_result.edit_ledger if aggressive_result is not None else []),
            *(terminal_thin_result.edit_ledger if terminal_thin_result is not None else []),
        ],
        "area_transition_relaxation": area_transition_result.report,
        "terminal_systematic_thin_repair": (
            terminal_thin_result.report if terminal_thin_result is not None else {"enabled": False}
        ),
        "baseline_terminal_audit": baseline_audit,
        "terminal_audit": terminal_audit,
        "terminal_rollback_applied": terminal_rollback,
        "obc_remap_manifest": obc_remap_manifest,
    }
    final_missing = _missing_constraint_edges(triangles, chains)
    constraint_report["missing_constraint_edge_count"] = int(len(final_missing))
    constraint_report["boundary_constraint_recovered"] = bool(len(final_missing) == 0)
    constraint_report["conditioning_terminal_audit"] = terminal_audit
    _progress(progress_callback, "terminal_constraint_audit", 0.98, terminal_audit)

    boundary_count = int(np.count_nonzero(fixed_mask))
    lonlat = unproject_points(points_arr, boundary.projection)
    open_boundary_nodes = np.asarray([idx + 1 for idx in open_nodes if idx < len(points_arr)], dtype=int)
    report = {
        "schema_version": "fvcom_python_oceanmesh_mesh_v3",
        "backend": "scipy_delaunay_clean_room",
        "node_count": int(len(points_arr)),
        "triangle_count": int(len(triangles)),
        "boundary_node_count": int(boundary_count),
        "open_boundary_node_count": int(len(open_boundary_nodes)),
        "constraint_chain_count": int(len(chains)),
        "open_boundary_rebuilt_from_exterior_chain": True,
        "constraint_recovery": constraint_report,
        "size_sampling_mode": (
            "projected_callback"
            if size_sampler_xy is not None
            else "size_field_raster"
        ),
        "refine_iterations": int(config.refine_iterations),
        "smooth_iterations": int(config.smooth_iterations),
        "conditioning": conditioning_report,
    }
    _progress(progress_callback, "mesh_complete", 1.0, report)
    return MeshResult(
        nodes_xy=points_arr,
        nodes_lonlat=lonlat,
        triangles=triangles + 1,
        open_boundary_nodes=open_boundary_nodes,
        boundary_node_count=boundary_count,
        fixed_node_mask=fixed_mask,
        target_spacing_m=np.asarray(point_targets, dtype=float),
        constraint_chains=chains,
        boundary_kinds=kinds,
        hard_anchor_mask=np.asarray(hard_anchors, dtype=bool),
        node_lineage=np.asarray(node_lineage, dtype=int),
        report=report,
    )


def _conditioning_terminal_audit(
    points: np.ndarray,
    triangles: np.ndarray,
    fixed: np.ndarray,
    chains: list[list[int]],
    open_nodes_zero_based: np.ndarray,
    original_points: np.ndarray,
    original_fixed: np.ndarray,
    *,
    node_lineage: np.ndarray | None = None,
    original_hard: np.ndarray | None = None,
    allow_boundary_motion: bool = False,
) -> dict[str, Any]:
    """Audit the delivered conditioned mesh without changing connectivity."""
    geometry = triangle_geometry(points, triangles)
    topology = build_edge_topology(len(points), triangles)
    integrity = constraint_integrity(topology, chains, np.asarray(open_nodes_zero_based, dtype=int).tolist())
    original_count = len(original_points)
    original_mask = np.asarray(original_fixed, dtype=bool)
    lineage = np.asarray(node_lineage if node_lineage is not None else np.arange(len(points), dtype=int), dtype=int)
    surviving = np.where((lineage >= 0) & (lineage < original_count))[0]
    surviving_original = lineage[surviving]
    boundary_survivors = surviving[original_mask[surviving_original]] if len(surviving) else np.empty(0, dtype=int)
    boundary_original = lineage[boundary_survivors] if len(boundary_survivors) else np.empty(0, dtype=int)
    boundary_shift = (
        float(np.max(np.linalg.norm(points[boundary_survivors] - original_points[boundary_original], axis=1)))
        if len(boundary_survivors)
        else 0.0
    )
    hard_mask = np.asarray(original_hard if original_hard is not None else np.zeros(original_count, dtype=bool), dtype=bool)
    surviving_original_set = set(map(int, surviving_original.tolist()))
    missing_hard = [int(index) for index in np.where(hard_mask)[0] if int(index) not in surviving_original_set]
    hard_survivors = surviving[hard_mask[surviving_original]] if len(surviving) else np.empty(0, dtype=int)
    hard_original = lineage[hard_survivors] if len(hard_survivors) else np.empty(0, dtype=int)
    hard_shift = (
        float(np.max(np.linalg.norm(points[hard_survivors] - original_points[hard_original], axis=1)))
        if len(hard_survivors)
        else 0.0
    )
    positive = bool(len(triangles) and np.all(geometry["signed_area"] > 0.0))
    constraints_ok = bool(not chains or integrity["all_protected_edges_present"])
    obc_ok = bool(not len(open_nodes_zero_based) or integrity["open_boundary_ordered"])
    one_component = bool(len(topology.connected_component_sizes) == 1)
    manifold = bool(not topology.nonmanifold_edges)
    passed = bool(
        positive
        and constraints_ok
        and obc_ok
        and one_component
        and manifold
        and (bool(allow_boundary_motion) or boundary_shift <= 1.0e-10)
        and not missing_hard
        and hard_shift <= 1.0e-10
    )
    return {
        "passed": passed,
        "positive_signed_areas": positive,
        "all_protected_edges_present": constraints_ok,
        "open_boundary_ordered": obc_ok,
        "connected_component_count": int(len(topology.connected_component_sizes)),
        "nonmanifold_edge_count": int(len(topology.nonmanifold_edges)),
        "original_boundary_coordinate_max_shift_m": boundary_shift,
        "boundary_motion_policy": "source-arc-audited" if allow_boundary_motion else "fixed",
        "surviving_original_boundary_node_count": int(len(boundary_survivors)),
        "removed_original_boundary_node_count": int(np.count_nonzero(original_mask) - len(boundary_survivors)),
        "missing_hard_anchor_count": int(len(missing_hard)),
        "missing_hard_anchors": missing_hard[:100],
        "hard_anchor_coordinate_max_shift_m": hard_shift,
        "constraint_integrity": integrity,
    }


def _progress(callback: ProgressCallback | None, message: str, fraction: float, extra: dict[str, Any] | None = None) -> None:
    if callback is not None:
        callback(message, float(fraction), extra)


def _interior_seed_points(
    boundary: BoundaryNodes,
    size_field: SizeField,
    config: MeshConfig,
    size_sampler_xy: SizeSamplerXY | None = None,
) -> np.ndarray:
    if config.adaptive_seed or boundary.adaptive_resolution:
        return _adaptive_quadtree_seed_points(
            boundary,
            size_field,
            config,
            size_sampler_xy,
        )
    domain = boundary.domain_polygon_xy
    minx, miny, maxx, maxy = domain.bounds
    area = max(float(domain.area), 1.0)
    spacing = max(float(np.nanmedian(size_field.size)), float(np.sqrt(area / max(config.max_interior_points, 1))))
    spacing = max(spacing, 10.0)
    xs = np.arange(minx + 0.5 * spacing, maxx, spacing)
    ys = np.arange(miny + 0.5 * spacing, maxy, spacing * np.sqrt(3.0) / 2.0)
    pts = []
    boundary_tree = cKDTree(boundary.xy) if len(boundary.xy) else None
    for row, y in enumerate(ys):
        offset = 0.5 * spacing if row % 2 else 0.0
        for x in xs + offset:
            point = Point(float(x), float(y))
            if not domain.contains(point):
                continue
            nearest_boundary_target = spacing
            if boundary_tree is not None:
                _, nearest_boundary = boundary_tree.query([x, y])
                nearest_boundary_target = float(
                    boundary.target_spacing_m[int(nearest_boundary)]
                )
            # A regular lattice can land numerically on a straight boundary.
            # Keeping a physical clearance prevents an interior duplicate from
            # hiding the protected boundary vertex during Delaunay uniquing.
            boundary_clearance = float(domain.boundary.distance(point))
            if boundary_clearance < 0.20 * min(
                float(spacing),
                nearest_boundary_target,
            ):
                continue
            pts.append((float(x), float(y)))
            if len(pts) >= config.max_interior_points:
                break
        if len(pts) >= config.max_interior_points:
            break
    return np.asarray(pts, dtype=float)


def _adaptive_quadtree_seed_points(
    boundary: BoundaryNodes,
    size_field: SizeField,
    config: MeshConfig,
    size_sampler_xy: SizeSamplerXY | None = None,
) -> np.ndarray:
    """Create deterministic variable-density seeds from local target size."""
    domain = boundary.domain_polygon_xy
    minx, miny, maxx, maxy = domain.bounds
    span = max(maxx - minx, maxy - miny)
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    stack: list[tuple[float, float, float, int]] = [(cx, cy, span, 0)]
    leaves: list[tuple[float, float, float]] = []
    max_depth = 18
    while stack:
        x, y, width, depth = stack.pop()
        half = 0.5 * width
        cell = Point(float(x), float(y)).buffer(0.5 * width, cap_style=3)
        if not domain.intersects(cell):
            continue
        sample_xy = np.asarray(
            [
                [x, y],
                [x - 0.4 * width, y - 0.4 * width],
                [x + 0.4 * width, y - 0.4 * width],
                [x - 0.4 * width, y + 0.4 * width],
                [x + 0.4 * width, y + 0.4 * width],
            ],
            dtype=float,
        )
        # The root quadtree is the square bounding the projected domain, so
        # its center/corner probes can lie well outside the model polygon.
        # Adaptive-v2 coverage is intentionally strict; never ask the size
        # field to extrapolate for those irrelevant probes.  Retain probes
        # that are actually inside the domain and use a representative point
        # of the cell/domain overlap when the regular probes all miss a thin
        # intersection.
        inside = np.asarray(contains_xy(domain, sample_xy[:, 0], sample_xy[:, 1]), dtype=bool)
        if np.any(inside):
            sample_xy = sample_xy[inside]
        else:
            overlap = domain.intersection(cell)
            if overlap.is_empty:
                continue
            representative = overlap.representative_point()
            sample_xy = np.asarray([[representative.x, representative.y]], dtype=float)
        if size_sampler_xy is not None:
            targets = np.asarray(size_sampler_xy(sample_xy), dtype=float)
        else:
            lonlat = unproject_points(sample_xy, boundary.projection)
            targets = size_field.sample(lonlat[:, 0], lonlat[:, 1])
        target = max(10.0, float(np.nanmin(targets)))
        if width > 1.25 * target and depth < max_depth:
            quarter = 0.25 * width
            child = 0.5 * width
            # Reverse push order preserves deterministic southwest-to-northeast traversal.
            for dx, dy in ((quarter, quarter), (-quarter, quarter), (quarter, -quarter), (-quarter, -quarter)):
                stack.append((x + dx, y + dy, child, depth + 1))
            continue
        leaves.append((x, y, target))
        if len(leaves) > int(config.max_interior_points) * 3:
            raise ValueError("Adaptive quadtree seed estimate exceeds --max-interior-points")
    boundary_tree = cKDTree(boundary.xy) if len(boundary.xy) else None
    front = np.empty((0, 2), dtype=float)
    if bool(config.boundary_front_seeding) and str(getattr(boundary, "resolution_profile", "")) == "adaptive-coastal-v2":
        front, _ = boundary_front_seed_points(boundary)
    accepted: list[tuple[float, float]] = [tuple(map(float, value)) for value in np.asarray(front, dtype=float)]
    accepted_tree = cKDTree(np.asarray(accepted, dtype=float)) if accepted else None
    for x, y, target in sorted(leaves, key=lambda item: (item[1], item[0])):
        point = Point(float(x), float(y))
        if not domain.contains(point):
            continue
        boundary_distance = float(domain.boundary.distance(point))
        nearest_boundary_target = target
        if boundary_tree is not None:
            _, nearest_boundary = boundary_tree.query([x, y])
            nearest_boundary_target = float(boundary.target_spacing_m[int(nearest_boundary)])
        if boundary_distance < 0.20 * min(float(target), nearest_boundary_target):
            continue
        if accepted_tree is not None and float(accepted_tree.query([x, y])[0]) < 0.35 * min(float(target), nearest_boundary_target):
            continue
        if accepted:
            recent = np.asarray(accepted[-512:], dtype=float)
            if float(np.min(np.linalg.norm(recent - np.asarray([x, y]), axis=1))) < 0.35 * min(
                float(target), nearest_boundary_target
            ):
                continue
        accepted.append((float(x), float(y)))
        if len(accepted) % 512 == 0:
            accepted_tree = cKDTree(np.asarray(accepted, dtype=float))
        if len(accepted) > int(config.max_interior_points):
            raise ValueError("Adaptive quadtree seeds exceed --max-interior-points; raise the cap or coarsen the profile")
    return np.asarray(accepted, dtype=float)


def _restrictions_to_delivered_indices(
    restrictions: set[tuple[int, int]],
    node_lineage: np.ndarray,
) -> set[tuple[int, int]]:
    inverse = {
        int(source): int(delivered)
        for delivered, source in enumerate(
            np.asarray(node_lineage, dtype=int)
        )
        if int(source) >= 0
    }
    return {
        tuple(sorted((inverse[int(left)], inverse[int(right)])))
        for left, right in restrictions
        if int(left) in inverse and int(right) in inverse
    }


def _compose_lineage(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Compose a conditioning-stage lineage map, preserving unique inserted IDs."""
    previous = np.asarray(previous, dtype=int)
    current = np.asarray(current, dtype=int)
    next_inserted = min(int(np.min(previous)) if len(previous) else 0, 0) - 1
    output = np.empty(len(current), dtype=int)
    for index, value in enumerate(current):
        if 0 <= int(value) < len(previous):
            output[index] = int(previous[int(value)])
        else:
            output[index] = int(next_inserted)
            next_inserted -= 1
    return output


def _triangulate_filtered(points: np.ndarray, boundary: BoundaryNodes) -> np.ndarray:
    if len(points) < 3:
        return np.empty((0, 3), dtype=int)
    unique, inverse = np.unique(np.round(points, 8), axis=0, return_inverse=True)
    if len(unique) < 3:
        return np.empty((0, 3), dtype=int)
    tri = Delaunay(unique)
    simplices_unique = np.asarray(tri.simplices, dtype=int)
    simplices = np.asarray([[np.where(inverse == idx)[0][0] for idx in simplex] for simplex in simplices_unique], dtype=int)
    kept = []
    domain = boundary.domain_polygon_xy
    for simplex in simplices:
        coords = points[simplex]
        centroid = Point(float(coords[:, 0].mean()), float(coords[:, 1].mean()))
        if domain.contains(centroid):
            kept.append(simplex)
    return np.asarray(kept, dtype=int)


def _missing_constraint_edges(triangles: np.ndarray, chains: list[list[int]]) -> list[tuple[int, int]]:
    edge_set: set[tuple[int, int]] = set()
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_set.add(tuple(sorted((int(a), int(b)))))
    missing = []
    for chain_idx, chain in enumerate(chains):
        if len(chain) < 2:
            continue
        for pos, a in enumerate(chain):
            b = chain[(pos + 1) % len(chain)]
            if tuple(sorted((int(a), int(b)))) not in edge_set:
                missing.append((chain_idx, pos))
    return missing


def _ordered_boundary_kind_group(chain: list[int], kinds: list[str], target_kind: str) -> list[int]:
    """Return the longest cyclic contiguous kind group in chain order."""
    if not chain:
        return []
    flags = [kinds[index] == target_kind for index in chain]
    if all(flags):
        return list(chain)
    if not any(flags):
        return []
    pivot = next(index for index, flag in enumerate(flags) if not flag)
    ordered = chain[pivot + 1 :] + chain[: pivot + 1]
    groups: list[list[int]] = []
    current: list[int] = []
    for node in ordered:
        if kinds[node] == target_kind:
            current.append(node)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return max(groups, key=lambda values: (len(values), -values[0])) if groups else []


def _generation_obc_remap_manifest(
    baseline: dict[str, Any],
    delivered_points: np.ndarray,
    delivered_chains: list[list[int]],
    delivered_open: np.ndarray,
    delivered_lineage: np.ndarray,
) -> dict[str, Any]:
    source_points = np.asarray(baseline["points"], dtype=float)
    source_chains = [list(map(int, chain)) for chain in baseline["chains"]]
    source_open = list(map(int, np.asarray(baseline["open_nodes"], dtype=int)))
    delivered = list(map(int, np.asarray(delivered_open, dtype=int)))
    source_set = set(source_open)

    def chain_for_node(chains: list[list[int]], node: int) -> int | None:
        for chain_index, chain in enumerate(chains):
            if int(node) in chain:
                return int(chain_index)
        return None

    def source_curve(chain_index: int) -> tuple[np.ndarray, np.ndarray, float] | None:
        if not 0 <= chain_index < len(source_chains):
            return None
        chain = source_chains[chain_index]
        if len(chain) < 2:
            return None
        coordinates = source_points[np.asarray([*chain, chain[0]], dtype=int)]
        lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        total = float(cumulative[-1])
        return (coordinates, cumulative, total) if total > 1.0e-12 else None

    def project(chain_index: int, point: np.ndarray) -> tuple[float, float] | None:
        curve = source_curve(chain_index)
        if curve is None:
            return None
        coordinates, cumulative, total = curve
        best: tuple[float, float] | None = None
        for index, (left, right) in enumerate(zip(coordinates[:-1], coordinates[1:])):
            vector = right - left
            denominator = float(np.dot(vector, vector))
            fraction = (
                0.0
                if denominator <= 1.0e-20
                else float(np.clip(np.dot(point - left, vector) / denominator, 0.0, 1.0))
            )
            projected = left + fraction * vector
            distance = float(np.linalg.norm(point - projected))
            s = float(cumulative[index] + fraction * (cumulative[index + 1] - cumulative[index]))
            if best is None or (distance, s) < (best[1], best[0]):
                best = (s % total, distance)
        return best

    source_positions: dict[int, tuple[int, float]] = {}
    for source_node in source_open:
        chain_index = chain_for_node(source_chains, source_node)
        projected = project(chain_index, source_points[source_node]) if chain_index is not None else None
        if chain_index is not None and projected is not None:
            source_positions[source_node] = (chain_index, projected[0])

    entries: list[dict[str, Any]] = []
    delivered_lineages: list[int] = []
    for order, node in enumerate(delivered):
        lineage = int(delivered_lineage[node])
        delivered_lineages.append(lineage)
        chain_index = chain_for_node(delivered_chains, node)
        projected = project(chain_index, delivered_points[node]) if chain_index is not None else None
        moved = bool(
            0 <= lineage < len(source_points)
            and not np.allclose(delivered_points[node], source_points[lineage], atol=1.0e-9, rtol=0.0)
        )
        status = (
            "slid"
            if lineage in source_set and moved
            else "retained"
            if lineage in source_set
            else "redistributed"
            if lineage >= 0
            else "inserted"
        )
        nearest = sorted(
            (
                (abs(source_s - projected[0]), source_node)
                for source_node, (source_chain, source_s) in source_positions.items()
                if projected is not None and source_chain == chain_index
            )
        )[:2]
        curve = source_curve(chain_index) if chain_index is not None else None
        entries.append(
            {
                "delivered_order_zero_based": int(order),
                "delivered_node_index_zero_based": int(node),
                "status": status,
                "source_node_lineage": lineage if lineage >= 0 else None,
                "bracketing_original_obc_lineage": [int(value[1]) for value in nearest],
                "constraint_chain_id": chain_index,
                "source_arc_position_m": float(projected[0]) if projected is not None else None,
                "source_arc_fraction": (
                    float(projected[0] / curve[2]) if projected is not None and curve is not None else None
                ),
                "coordinate_xy": [float(delivered_points[node, 0]), float(delivered_points[node, 1])],
            }
        )
    compatible = bool(
        len(delivered) == len(source_open)
        and delivered_lineages == source_open
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
            or not delivered_lineages
            or (
                delivered_lineages[0] == source_open[0]
                and delivered_lineages[-1] == source_open[-1]
            )
        ),
        "removed_original_obc_lineage": [
            int(node) for node in source_open if int(node) not in set(delivered_lineages)
        ],
        "delivered_nodes": entries,
    }


def _refinement_points(
    points: np.ndarray,
    triangles: np.ndarray,
    boundary: BoundaryNodes,
    size_field: SizeField,
    config: MeshConfig,
    size_sampler_xy: SizeSamplerXY | None = None,
) -> np.ndarray:
    additions = []
    if not len(triangles):
        return np.empty((0, 2), dtype=float)
    tree = cKDTree(points)
    triangle_coords = points[triangles]
    centroids = triangle_coords.mean(axis=1)
    if size_sampler_xy is not None:
        targets = np.asarray(size_sampler_xy(centroids), dtype=float)
    else:
        centroid_lonlat = unproject_points(centroids, boundary.projection)
        targets = size_field.sample(
            centroid_lonlat[:, 0],
            centroid_lonlat[:, 1],
        )
    for tri, coords, centroid, sampled_target in zip(triangles, triangle_coords, centroids, targets, strict=True):
        max_edge = _max_edge_length(coords)
        min_angle = _triangle_angles(coords).min()
        target = float(sampled_target)
        if max_edge <= config.size_overrun_factor * target and min_angle >= config.min_angle_refine_deg:
            continue
        if boundary.adaptive_resolution and max_edge <= 0.95 * target:
            # Do not chase a low angle by adding already-undersized nodes.
            continue
        candidate = _circumcenter(coords)
        if candidate is None or not boundary.domain_polygon_xy.contains(Point(float(candidate[0]), float(candidate[1]))):
            a, b = _longest_edge(coords)
            candidate = 0.5 * (a + b)
        candidate_point = Point(float(candidate[0]), float(candidate[1]))
        if not boundary.domain_polygon_xy.contains(candidate_point):
            continue
        # A fallback midpoint can be numerically just inside a protected
        # boundary edge.  Reject that pseudo-interior point before it can
        # split the edge without joining the boundary constraint chain.
        if float(boundary.domain_polygon_xy.boundary.distance(candidate_point)) < max(
            0.10 * target,
            1.0,
        ):
            continue
        distance, _ = tree.query(candidate)
        if distance < max(0.25 * target, 1.0):
            continue
        additions.append(candidate)
        if len(additions) >= config.max_refine_insertions_per_iter:
            break
    return np.asarray(additions, dtype=float)


def _smooth_interior(points: np.ndarray, triangles: np.ndarray, fixed: np.ndarray, boundary: BoundaryNodes, iterations: int) -> np.ndarray:
    out = points.copy()
    adjacency = [set() for _ in range(len(out))]
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            adjacency[int(a)].add(int(b))
            adjacency[int(b)].add(int(a))
    for _ in range(max(0, iterations)):
        updated = out.copy()
        for idx, neighbors in enumerate(adjacency):
            if fixed[idx] or not neighbors:
                continue
            target = out[list(neighbors)].mean(axis=0)
            candidate = 0.45 * out[idx] + 0.55 * target
            if boundary.domain_polygon_xy.contains(Point(float(candidate[0]), float(candidate[1]))):
                updated[idx] = candidate
        out = updated
    return out


def _orient_ccw(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    out = triangles.copy()
    for i, tri in enumerate(out):
        coords = points[tri]
        first = coords[1] - coords[0]
        second = coords[2] - coords[0]
        area2 = first[0] * second[1] - first[1] * second[0]
        if area2 < 0:
            out[i] = [tri[0], tri[2], tri[1]]
    return out


def _max_edge_length(coords: np.ndarray) -> float:
    return float(max(np.linalg.norm(coords[0] - coords[1]), np.linalg.norm(coords[1] - coords[2]), np.linalg.norm(coords[2] - coords[0])))


def _triangle_angles(coords: np.ndarray) -> np.ndarray:
    lengths = np.asarray([
        np.linalg.norm(coords[1] - coords[2]),
        np.linalg.norm(coords[0] - coords[2]),
        np.linalg.norm(coords[0] - coords[1]),
    ])
    angles = []
    for i in range(3):
        a = lengths[i]
        b = lengths[(i + 1) % 3]
        c = lengths[(i + 2) % 3]
        denom = max(2.0 * b * c, 1.0e-12)
        val = np.clip((b * b + c * c - a * a) / denom, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(val)))
    return np.asarray(angles, dtype=float)


def _circumcenter(coords: np.ndarray) -> np.ndarray | None:
    ax, ay = coords[0]
    bx, by = coords[1]
    cx, cy = coords[2]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1.0e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return np.asarray([ux, uy], dtype=float)


def _longest_edge(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = ((coords[0], coords[1]), (coords[1], coords[2]), (coords[2], coords[0]))
    return max(edges, key=lambda item: float(np.linalg.norm(item[0] - item[1])))
