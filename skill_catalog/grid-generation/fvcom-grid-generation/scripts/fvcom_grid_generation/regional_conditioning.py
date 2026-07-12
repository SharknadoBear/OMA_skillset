"""Regional, boundary-fixed conditioning for FVCOM triangular meshes.

All public functions use zero-based node indices.  Spring relaxation keeps
connectivity fixed.  Thin-triangle repair uses local edge flips and midpoint
splits only; it never invokes a global Delaunay rebuild.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .metrics import build_edge_topology, chain_edges, constraint_integrity, triangle_geometry


@dataclass(frozen=True)
class SpringRelaxConfig:
    enabled: bool = True
    quality_threshold: float = 0.40
    min_angle_deg: float = 28.0
    ring_layers: int = 3
    iterations: int = 20
    damping: float = 0.35
    max_step_fraction: float = 0.15
    shape_weight: float = 0.20
    force_tolerance: float = 1.0e-3


@dataclass
class SpringRelaxResult:
    nodes_xy: np.ndarray
    report: dict[str, Any]
    active_node_mask: np.ndarray


@dataclass(frozen=True)
class AreaTransitionRelaxConfig:
    enabled: bool = True
    max_patches: int = 12
    raw_area_change_threshold: float = 0.50
    target_gradient_threshold: float = 0.10
    high_gradient_area_change_threshold: float = 0.375
    normalized_log_jump_threshold: float = float(np.log(1.5))
    minimum_severity_reduction_fraction: float = 0.02
    l_over_h_threshold: float = 1.55
    l_over_h_relative_tolerance: float = 1.0e-3
    max_total_displacement_fraction: float = 0.25
    max_attempt_multiplier: int = 3
    spring_config: SpringRelaxConfig = field(
        default_factory=lambda: SpringRelaxConfig(
            ring_layers=3,
            iterations=20,
            damping=0.30,
            max_step_fraction=0.08,
            shape_weight=0.20,
        )
    )


@dataclass
class AreaTransitionRelaxResult:
    nodes_xy: np.ndarray
    target_spacing_m: np.ndarray
    report: dict[str, Any]
    active_node_mask: np.ndarray


@dataclass(frozen=True)
class ThinTriangleRepairConfig:
    enabled: bool = True
    quality_threshold: float = 0.25
    min_angle_deg: float = 20.0
    max_passes: int = 2
    max_flips: int = 200
    max_insertions: int = 50
    split_target_factor: float = 1.25
    relaxation_config: SpringRelaxConfig = field(default_factory=SpringRelaxConfig)


@dataclass
class ThinTriangleRepairResult:
    nodes_xy: np.ndarray
    triangles: np.ndarray
    fixed_node_mask: np.ndarray
    target_spacing_m: np.ndarray
    constraint_chains: list[list[int]]
    open_boundary_nodes_zero_based: np.ndarray
    inserted_parent_edges: list[tuple[int, int, int]]
    report: dict[str, Any]


def relax_mesh_spring(
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    fixed_node_mask: np.ndarray,
    *,
    target_spacing_m: np.ndarray | None = None,
    constraint_chains: list[list[int]] | None = None,
    open_boundary_nodes_zero_based: np.ndarray | None = None,
    region_bbox_xy: tuple[float, float, float, float] | None = None,
    seed_triangle_mask: np.ndarray | None = None,
    config: SpringRelaxConfig | None = None,
) -> SpringRelaxResult:
    """Relax defect-selected graph patches as a finite-rest-length spring net.

    Physical boundary nodes are represented by ``fixed_node_mask`` and are
    never moved.  Nodes outside the selected graph patch also remain unchanged
    and act as an outer anchoring halo.
    """
    config = config or SpringRelaxConfig()
    points = np.asarray(nodes_xy, dtype=float).copy()
    tris = np.asarray(triangles, dtype=int).copy()
    fixed = np.asarray(fixed_node_mask, dtype=bool).copy()
    _validate_mesh_arrays(points, tris, fixed)
    chains = [list(map(int, chain)) for chain in (constraint_chains or [])]
    open_nodes = np.asarray(open_boundary_nodes_zero_based if open_boundary_nodes_zero_based is not None else [], dtype=int)
    targets = _normalized_targets(target_spacing_m, len(points))
    before = _quality_summary(points, tris, config.quality_threshold, config.min_angle_deg)

    if not config.enabled or not len(tris) or int(config.iterations) <= 0:
        return SpringRelaxResult(points, _spring_noop_report(config, before, "disabled"), np.zeros(len(points), dtype=bool))

    geometry = triangle_geometry(points, tris)
    min_angles = np.min(geometry["angles_deg"], axis=1)
    if seed_triangle_mask is None:
        seeds = (geometry["quality"] < float(config.quality_threshold)) | (min_angles < float(config.min_angle_deg))
    else:
        seeds = np.asarray(seed_triangle_mask, dtype=bool).copy()
        if seeds.shape != (len(tris),):
            raise ValueError("seed_triangle_mask must have one value per triangle")
    if region_bbox_xy is not None:
        minx, miny, maxx, maxy = map(float, region_bbox_xy)
        centroids = points[tris].mean(axis=1)
        in_region = (
            (centroids[:, 0] >= minx)
            & (centroids[:, 0] <= maxx)
            & (centroids[:, 1] >= miny)
            & (centroids[:, 1] <= maxy)
        )
        seeds &= in_region
    if not np.any(seeds):
        return SpringRelaxResult(points, _spring_noop_report(config, before, "no_selected_defects"), np.zeros(len(points), dtype=bool))

    topology = build_edge_topology(len(points), tris)
    candidate_seed_count = int(np.count_nonzero(seeds))
    seeds = _worst_seed_component(seeds, geometry, topology, config)
    seed_nodes = np.unique(tris[np.where(seeds)[0]].ravel())
    mobility = _graph_mobility(topology.node_neighbors, seed_nodes, fixed, max(0, int(config.ring_layers)))
    active = mobility > 0.0
    if not np.any(active):
        return SpringRelaxResult(points, _spring_noop_report(config, before, "no_movable_nodes"), active)

    affected = np.any(active[tris], axis=1)
    edges = np.asarray(
        [edge for edge in sorted(topology.edge_to_triangles) if active[edge[0]] or active[edge[1]]],
        dtype=int,
    )
    if not len(edges) or not np.any(affected):
        return SpringRelaxResult(points, _spring_noop_report(config, before, "empty_active_patch"), active)

    node_sizes = _infer_node_sizes(points, topology.node_neighbors, targets)
    rest_lengths = _rest_lengths(points, edges, node_sizes)
    original_points = points.copy()
    original_fixed = points[fixed].copy()
    initial_energy = _conditioning_energy(points, tris, edges, rest_lengths, affected, float(config.shape_weight))
    current_energy = float(initial_energy)
    initial_local = _quality_summary(points, tris[affected], config.quality_threshold, config.min_angle_deg)
    history: list[dict[str, Any]] = []
    accepted_iterations = 0
    force_residual = float("inf")
    stop_reason = "iteration_budget_reached"

    for iteration in range(max(0, int(config.iterations))):
        force, diagonal = _spring_shape_force(
            points,
            tris,
            edges,
            rest_lengths,
            affected,
            float(config.shape_weight),
        )
        delta = np.zeros_like(points)
        movable = active & ~fixed & (diagonal > 0.0)
        delta[movable] = float(config.damping) * mobility[movable, None] * force[movable] / diagonal[movable, None]
        cap = np.maximum(float(config.max_step_fraction) * node_sizes, 1.0e-9)
        norm = np.linalg.norm(delta, axis=1)
        scale = np.minimum(1.0, cap / np.maximum(norm, 1.0e-30))
        delta *= scale[:, None]
        force_residual = float(np.max(np.linalg.norm(delta[movable], axis=1) / np.maximum(node_sizes[movable], 1.0e-9))) if np.any(movable) else 0.0
        if force_residual <= float(config.force_tolerance):
            stop_reason = "force_balance_tolerance"
            break

        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = points + float(alpha) * delta
            candidate[fixed] = original_fixed
            candidate_geometry = triangle_geometry(candidate, tris[affected])
            if np.any(candidate_geometry["signed_area"] <= _area_tolerance(points, tris[affected])):
                continue
            candidate_energy = _conditioning_energy(candidate, tris, edges, rest_lengths, affected, float(config.shape_weight))
            if not np.isfinite(candidate_energy) or candidate_energy >= current_energy - 1.0e-12 * max(abs(current_energy), 1.0):
                continue
            candidate_local = _quality_summary(candidate, tris[affected], config.quality_threshold, config.min_angle_deg)
            if not _local_step_nonregression(initial_local, candidate_local):
                continue
            points = candidate
            energy_drop = current_energy - float(candidate_energy)
            current_energy = float(candidate_energy)
            accepted_iterations += 1
            history.append(
                {
                    "iteration": int(iteration + 1),
                    "alpha": float(alpha),
                    "energy": current_energy,
                    "energy_drop": float(energy_drop),
                    "normalized_max_step": force_residual,
                }
            )
            accepted = True
            break
        if not accepted:
            stop_reason = "no_legal_backtracked_step"
            break

    after = _quality_summary(points, tris, config.quality_threshold, config.min_angle_deg)
    after_local = _quality_summary(points, tris[affected], config.quality_threshold, config.min_angle_deg)
    trial_after = dict(after)
    trial_after_local = dict(after_local)
    fixed_shift = _maximum_shift(original_points[fixed], points[fixed])
    integrity = constraint_integrity(build_edge_topology(len(points), tris), chains, open_nodes.tolist())
    stage_ok = (
        accepted_iterations > 0
        and current_energy < initial_energy - 1.0e-12 * max(abs(initial_energy), 1.0)
        and _global_nonregression(before, after)
        and _local_nonregression(initial_local, after_local)
        and fixed_shift <= 1.0e-10
        and (not chains or bool(integrity["all_protected_edges_present"]))
        and (not len(open_nodes) or bool(integrity["open_boundary_ordered"]))
    )
    if not stage_ok:
        points = original_points
        after = before
        after_local = initial_local
        current_energy = initial_energy
        reason = "rollback_transaction_guard" if accepted_iterations else stop_reason
        accepted_iterations = 0
    else:
        reason = f"accepted_{stop_reason}"

    report = {
        "schema_version": "fvcom_spring_relaxation_v1",
        "profile": "spring-relax-v1",
        "accepted": bool(stage_ok),
        "reason": reason,
        "settings": _config_dict(config),
        "candidate_seed_triangle_count": candidate_seed_count,
        "selected_seed_triangle_count": int(np.count_nonzero(seeds)),
        "active_node_count": int(np.count_nonzero(active)),
        "movable_node_count": int(np.count_nonzero(active & ~fixed)),
        "affected_triangle_count": int(np.count_nonzero(affected)),
        "accepted_iteration_count": int(accepted_iterations),
        "initial_energy": float(initial_energy),
        "final_energy": float(current_energy),
        "relative_energy_reduction": float((initial_energy - current_energy) / max(abs(initial_energy), 1.0e-30)),
        "normalized_force_residual": float(force_residual if np.isfinite(force_residual) else 0.0),
        "boundary_coordinate_max_shift_m": float(_maximum_shift(original_points[fixed], points[fixed])),
        "before": before,
        "after": after,
        "trial_after_before_rollback": trial_after,
        "local_before": initial_local,
        "local_after": after_local,
        "trial_local_after_before_rollback": trial_after_local,
        "constraint_integrity": integrity,
        "history": history,
    }
    return SpringRelaxResult(nodes_xy=points, report=report, active_node_mask=active)


def relax_mesh_area_transitions(
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    fixed_node_mask: np.ndarray,
    *,
    target_spacing_sampler: Callable[[np.ndarray], np.ndarray],
    constraint_chains: list[list[int]] | None = None,
    open_boundary_nodes_zero_based: np.ndarray | None = None,
    region_bbox_xy: tuple[float, float, float, float] | None = None,
    config: AreaTransitionRelaxConfig | None = None,
) -> AreaTransitionRelaxResult:
    """Sequentially relax excessive adjacent-area transitions.

    The target sampler is evaluated in physical ``x/y`` space before every
    outer patch.  This keeps target spacing Eulerian while the inner spring
    solve remains a small, transactional, fixed-connectivity operation.
    """
    config = config or AreaTransitionRelaxConfig()
    points = np.asarray(nodes_xy, dtype=float).copy()
    tris = np.asarray(triangles, dtype=int).copy()
    fixed = np.asarray(fixed_node_mask, dtype=bool).copy()
    _validate_mesh_arrays(points, tris, fixed)
    _validate_area_transition_config(config)
    chains = [list(map(int, chain)) for chain in (constraint_chains or [])]
    open_nodes = np.asarray(open_boundary_nodes_zero_based if open_boundary_nodes_zero_based is not None else [], dtype=int)
    original_points = points.copy()
    original_fixed = points[fixed].copy()
    topology = build_edge_topology(len(points), tris)
    initial_component_count = len(topology.connected_component_sizes)
    sampler_call_count = 0

    def sample_targets(locations: np.ndarray) -> np.ndarray:
        nonlocal sampler_call_count
        sampler_call_count += 1
        values = np.asarray(target_spacing_sampler(np.asarray(locations, dtype=float)), dtype=float)
        if values.shape != (len(locations),):
            raise ValueError("target_spacing_sampler must return one value per x/y location")
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("target_spacing_sampler returned a nonfinite or nonpositive spacing")
        return values

    original_targets = sample_targets(points)
    before, pairs = _area_transition_summary(
        points,
        tris,
        topology,
        sample_targets,
        config,
        region_bbox_xy,
    )
    before_quality = _quality_summary(
        points,
        tris,
        config.spring_config.quality_threshold,
        config.spring_config.min_angle_deg,
    )
    current_summary = before
    current_pairs = pairs
    active_union = np.zeros(len(points), dtype=bool)
    rejected_edges: set[tuple[int, int]] = set()
    patch_reports: list[dict[str, Any]] = []
    accepted_patch_count = 0
    attempted_patch_count = 0
    stop_reason = "no_selected_area_transition_defects"

    can_run = bool(config.enabled and len(tris) and int(config.max_patches) > 0)
    if not config.enabled or int(config.max_patches) <= 0:
        stop_reason = "disabled"
    elif not len(tris):
        stop_reason = "empty_mesh"

    maximum_attempts = max(int(config.max_patches), int(config.max_patches) * int(config.max_attempt_multiplier))
    while can_run and accepted_patch_count < int(config.max_patches) and attempted_patch_count < maximum_attempts:
        candidates = [pair for pair in current_pairs if pair["triggered"] and tuple(pair["edge"]) not in rejected_edges]
        if not candidates:
            if accepted_patch_count:
                stop_reason = "transition_targets_satisfied"
            elif attempted_patch_count:
                stop_reason = "no_legal_transition_patch"
            else:
                stop_reason = "no_selected_area_transition_defects"
            break
        selected = max(
            candidates,
            key=lambda pair: (
                float(pair["score"]),
                float(pair["area_change"]),
                float(pair["normalized_log_jump"]),
                -int(pair["edge"][0]),
                -int(pair["edge"][1]),
            ),
        )
        edge = tuple(map(int, selected["edge"]))
        attached = list(map(int, selected["triangles"]))
        seed_mask = np.zeros(len(tris), dtype=bool)
        seed_mask[np.asarray(attached, dtype=int)] = True
        node_targets = sample_targets(points)
        spring_result = relax_mesh_spring(
            points,
            tris,
            fixed,
            target_spacing_m=node_targets,
            constraint_chains=chains,
            open_boundary_nodes_zero_based=open_nodes,
            seed_triangle_mask=seed_mask,
            config=config.spring_config,
        )
        attempted_patch_count += 1
        patch_report: dict[str, Any] = {
            "attempt": int(attempted_patch_count),
            "selected_pair": _public_transition_pair(selected),
            "accepted": False,
            "spring": spring_result.report,
            "before": current_summary,
        }
        if not spring_result.report.get("accepted", False):
            rejected_edges.add(edge)
            patch_report["reason"] = "spring_transaction_rejected"
            patch_report["gate_failures"] = [str(spring_result.report.get("reason", "spring_rejected"))]
            patch_reports.append(patch_report)
            continue

        candidate_points = np.asarray(spring_result.nodes_xy, dtype=float)
        candidate_summary, candidate_pairs = _area_transition_summary(
            candidate_points,
            tris,
            topology,
            sample_targets,
            config,
            region_bbox_xy,
        )
        candidate_quality = _quality_summary(
            candidate_points,
            tris,
            config.spring_config.quality_threshold,
            config.spring_config.min_angle_deg,
        )
        displacement_fraction = _maximum_displacement_fraction(original_points, candidate_points, original_targets)
        gate_failures = _area_transition_candidate_failures(
            candidate_points,
            tris,
            fixed,
            chains,
            open_nodes,
            original_points,
            original_fixed,
            initial_component_count,
            before,
            current_summary,
            candidate_summary,
            before_quality,
            candidate_quality,
            displacement_fraction,
            config,
        )
        patch_report.update(
            {
                "after": candidate_summary,
                "shape_after": candidate_quality,
                "maximum_total_displacement_over_h": float(displacement_fraction),
                "gate_failures": gate_failures,
            }
        )
        if gate_failures:
            rejected_edges.add(edge)
            patch_report["reason"] = "area_transition_transaction_rejected"
            patch_reports.append(patch_report)
            continue

        points = candidate_points
        current_summary = candidate_summary
        current_pairs = candidate_pairs
        active_union |= np.asarray(spring_result.active_node_mask, dtype=bool)
        accepted_patch_count += 1
        patch_report["accepted"] = True
        patch_report["reason"] = "accepted"
        patch_reports.append(patch_report)
    else:
        if can_run and accepted_patch_count >= int(config.max_patches):
            stop_reason = "patch_budget_reached"
        elif can_run and attempted_patch_count >= maximum_attempts:
            stop_reason = "attempt_budget_reached"

    trial_points = points.copy()
    trial_after = current_summary
    trial_quality = _quality_summary(
        trial_points,
        tris,
        config.spring_config.quality_threshold,
        config.spring_config.min_angle_deg,
    )
    trial_displacement_fraction = _maximum_displacement_fraction(original_points, trial_points, original_targets)
    final_gate_failures = (
        _area_transition_final_failures(
            trial_points,
            tris,
            fixed,
            chains,
            open_nodes,
            original_points,
            original_fixed,
            initial_component_count,
            before,
            trial_after,
            before_quality,
            trial_quality,
            trial_displacement_fraction,
            config,
        )
        if accepted_patch_count > 0
        else []
    )
    stage_accepted = bool(accepted_patch_count > 0 and not final_gate_failures)
    if accepted_patch_count > 0 and final_gate_failures:
        points = original_points
        current_summary = before
        active_union[:] = False
        reason = "rollback_final_stage_guard"
    elif stage_accepted:
        reason = f"accepted_{stop_reason}"
    else:
        reason = stop_reason

    final_targets = sample_targets(points)
    after_quality = _quality_summary(
        points,
        tris,
        config.spring_config.quality_threshold,
        config.spring_config.min_angle_deg,
    )
    final_topology = build_edge_topology(len(points), tris)
    integrity = constraint_integrity(final_topology, chains, open_nodes.tolist())
    boundary_shift = _maximum_shift(original_points[fixed], points[fixed])
    final_displacement_fraction = _maximum_displacement_fraction(original_points, points, original_targets)
    report = {
        "schema_version": "fvcom_area_transition_relaxation_v1",
        "profile": "area-transition-relax-v1",
        "accepted": bool(stage_accepted),
        "reason": reason,
        "settings": _config_dict(config),
        "target_sampling": {
            "frame": "eulerian_xy",
            "call_count": int(sampler_call_count),
            "resampled_before_each_outer_patch": True,
        },
        "attempted_patch_count": int(attempted_patch_count),
        "trial_accepted_patch_count": int(accepted_patch_count),
        "applied_patch_count": int(accepted_patch_count if stage_accepted else 0),
        "rejected_pair_count": int(len(rejected_edges)),
        "active_node_count": int(np.count_nonzero(active_union)),
        "boundary_coordinate_max_shift_m": float(boundary_shift),
        "maximum_total_displacement_over_h": float(final_displacement_fraction),
        "trial_maximum_total_displacement_over_h": float(trial_displacement_fraction),
        "before": before,
        "after": current_summary,
        "trial_after_before_rollback": trial_after,
        "shape_before": before_quality,
        "shape_after": after_quality,
        "trial_shape_after_before_rollback": trial_quality,
        "final_gate_failures": final_gate_failures,
        "constraint_integrity": integrity,
        "connected_component_count": int(len(final_topology.connected_component_sizes)),
        "nonmanifold_edge_count": int(len(final_topology.nonmanifold_edges)),
        "patches": patch_reports,
    }
    return AreaTransitionRelaxResult(
        nodes_xy=points,
        target_spacing_m=final_targets,
        report=report,
        active_node_mask=active_union,
    )


def repair_thin_triangles(
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    fixed_node_mask: np.ndarray,
    constraint_chains: list[list[int]],
    open_boundary_nodes_zero_based: np.ndarray,
    *,
    target_spacing_m: np.ndarray | None = None,
    region_bbox_xy: tuple[float, float, float, float] | None = None,
    config: ThinTriangleRepairConfig | None = None,
) -> ThinTriangleRepairResult:
    """Repair severe triangles with protected-edge-safe local operations."""
    config = config or ThinTriangleRepairConfig()
    points = np.asarray(nodes_xy, dtype=float).copy()
    tris = _orient_ccw(points, np.asarray(triangles, dtype=int).copy())
    fixed = np.asarray(fixed_node_mask, dtype=bool).copy()
    _validate_mesh_arrays(points, tris, fixed)
    chains = [list(map(int, chain)) for chain in constraint_chains]
    open_nodes = np.asarray(open_boundary_nodes_zero_based, dtype=int).copy()
    targets = _normalized_targets(target_spacing_m, len(points))
    original_points = points.copy()
    original_fixed = points[fixed].copy()
    initial_topology = build_edge_topology(len(points), tris)
    initial_components = len(initial_topology.connected_component_sizes)
    before = _quality_summary(points, tris, config.quality_threshold, config.min_angle_deg)
    inserted: list[tuple[int, int, int]] = []

    if not config.enabled or not len(tris):
        return _thin_result(points, tris, fixed, targets, chains, open_nodes, inserted, config, before, before, False, "disabled", 0, 0, None)

    flip_snapshot = tris.copy()
    flip_count = 0
    for _ in range(max(0, int(config.max_passes))):
        remaining = max(0, int(config.max_flips) - flip_count)
        if remaining <= 0:
            break
        updates = _quality_flip_batch(points, tris, chains, config, region_bbox_xy, remaining)
        if not updates:
            break
        for first, second, new_first, new_second in updates:
            tris[first] = new_first
            tris[second] = new_second
        flip_count += len(updates)

    after_flips = _quality_summary(points, tris, config.quality_threshold, config.min_angle_deg)
    if flip_count and not _repair_transaction_ok(
        points,
        tris,
        fixed,
        chains,
        open_nodes,
        original_points,
        initial_components,
        before,
        after_flips,
        require_improvement=True,
    ):
        tris = flip_snapshot
        flip_count = 0
        after_flips = before

    insertion_snapshot = (points.copy(), tris.copy(), fixed.copy(), targets.copy())
    insertion_before = _quality_summary(points, tris, config.quality_threshold, config.min_angle_deg)
    selected_edges = _select_split_edges(
        points,
        tris,
        targets,
        chains,
        config,
        region_bbox_xy,
        max(0, int(config.max_insertions)),
    )
    relaxation_report: dict[str, Any] | None = None
    if selected_edges:
        points, tris, fixed, targets, inserted = _apply_edge_splits(points, tris, fixed, targets, selected_edges)
        inserted_nodes = {record[0] for record in inserted}
        seed_mask = np.asarray([any(int(node) in inserted_nodes for node in tri) for tri in tris], dtype=bool)
        relaxed = relax_mesh_spring(
            points,
            tris,
            fixed,
            target_spacing_m=targets,
            constraint_chains=chains,
            open_boundary_nodes_zero_based=open_nodes,
            seed_triangle_mask=seed_mask,
            config=config.relaxation_config,
        )
        points = relaxed.nodes_xy
        relaxation_report = relaxed.report
        insertion_after = _quality_summary(points, tris, config.quality_threshold, config.min_angle_deg)
        insertion_ok = _repair_transaction_ok(
            points,
            tris,
            fixed,
            chains,
            open_nodes,
            original_points,
            initial_components,
            insertion_before,
            insertion_after,
            require_improvement=True,
        )
        if not insertion_ok:
            points, tris, fixed, targets = insertion_snapshot
            inserted = []
            relaxation_report = {"accepted": False, "reason": "split_bundle_rolled_back", "trial": relaxation_report}

    after = _quality_summary(points, tris, config.quality_threshold, config.min_angle_deg)
    operation_count = int(flip_count + len(inserted))
    accepted = operation_count > 0 and _repair_transaction_ok(
        points,
        tris,
        fixed,
        chains,
        open_nodes,
        original_points,
        initial_components,
        before,
        after,
        require_improvement=True,
    )
    if not accepted and operation_count:
        points = original_points
        tris = np.asarray(triangles, dtype=int).copy()
        fixed = np.asarray(fixed_node_mask, dtype=bool).copy()
        targets = _normalized_targets(target_spacing_m, len(points))
        inserted = []
        flip_count = 0
        after = before
        reason = "rollback_transaction_guard"
    elif accepted:
        reason = "quality_improved"
    else:
        reason = "no_legal_improving_operation"

    return _thin_result(
        points,
        tris,
        fixed,
        targets,
        chains,
        open_nodes,
        inserted,
        config,
        before,
        after,
        accepted,
        reason,
        flip_count,
        len(inserted),
        relaxation_report,
    )


def _spring_noop_report(config: SpringRelaxConfig, summary: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": "fvcom_spring_relaxation_v1",
        "profile": "spring-relax-v1",
        "accepted": False,
        "reason": reason,
        "settings": _config_dict(config),
        "selected_seed_triangle_count": 0,
        "active_node_count": 0,
        "movable_node_count": 0,
        "affected_triangle_count": 0,
        "accepted_iteration_count": 0,
        "initial_energy": 0.0,
        "final_energy": 0.0,
        "relative_energy_reduction": 0.0,
        "normalized_force_residual": 0.0,
        "boundary_coordinate_max_shift_m": 0.0,
        "before": summary,
        "after": summary,
        "local_before": summary,
        "local_after": summary,
        "constraint_integrity": {},
        "history": [],
    }


def _thin_result(
    points: np.ndarray,
    tris: np.ndarray,
    fixed: np.ndarray,
    targets: np.ndarray,
    chains: list[list[int]],
    open_nodes: np.ndarray,
    inserted: list[tuple[int, int, int]],
    config: ThinTriangleRepairConfig,
    before: dict[str, Any],
    after: dict[str, Any],
    accepted: bool,
    reason: str,
    flips: int,
    insertions: int,
    relaxation_report: dict[str, Any] | None,
) -> ThinTriangleRepairResult:
    topology = build_edge_topology(len(points), tris)
    integrity = constraint_integrity(topology, chains, open_nodes.tolist())
    report = {
        "schema_version": "fvcom_thin_triangle_repair_v1",
        "profile": "thin-repair-v1",
        "accepted": bool(accepted),
        "reason": reason,
        "settings": {
            key: value
            for key, value in _config_dict(config).items()
            if key != "relaxation_config"
        },
        "relaxation_settings": _config_dict(config.relaxation_config),
        "edge_flip_count": int(flips),
        "edge_split_count": int(insertions),
        "inserted_parent_edges": [list(map(int, record)) for record in inserted],
        "before": before,
        "after": after,
        "constraint_integrity": integrity,
        "connected_component_count": int(len(topology.connected_component_sizes)),
        "nonmanifold_edge_count": int(len(topology.nonmanifold_edges)),
        "regional_relaxation": relaxation_report,
    }
    return ThinTriangleRepairResult(
        nodes_xy=np.asarray(points, dtype=float),
        triangles=_orient_ccw(points, tris),
        fixed_node_mask=np.asarray(fixed, dtype=bool),
        target_spacing_m=np.asarray(targets, dtype=float),
        constraint_chains=[chain.copy() for chain in chains],
        open_boundary_nodes_zero_based=np.asarray(open_nodes, dtype=int),
        inserted_parent_edges=inserted.copy(),
        report=report,
    )


def _quality_flip_batch(
    points: np.ndarray,
    tris: np.ndarray,
    chains: list[list[int]],
    config: ThinTriangleRepairConfig,
    region_bbox_xy: tuple[float, float, float, float] | None,
    budget: int,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    topology = build_edge_topology(len(points), tris)
    geometry = triangle_geometry(points, tris)
    min_angles = np.min(geometry["angles_deg"], axis=1)
    bad = (geometry["quality"] < float(config.quality_threshold)) | (min_angles < float(config.min_angle_deg))
    bad &= _triangle_region_mask(points, tris, region_bbox_xy)
    protected = chain_edges(chains)
    candidate_edges: set[tuple[int, int]] = set()
    ordering = sorted(np.where(bad)[0], key=lambda index: (float(geometry["quality"][index]), float(min_angles[index]), int(index)))
    for index in ordering:
        candidate_edges.update(_triangle_edges(tris[int(index)]))
    selected_triangles: set[int] = set()
    selected_new_edges: set[tuple[int, int]] = set()
    updates: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for edge in sorted(candidate_edges):
        attached = topology.edge_to_triangles.get(edge, [])
        if edge in protected or len(attached) != 2:
            continue
        first, second = map(int, attached)
        if first in selected_triangles or second in selected_triangles:
            continue
        candidate = _edge_flip_candidate(points, tris, edge, first, second)
        if candidate is None:
            continue
        new_first, new_second, old_min_q, new_min_q, old_min_angle, new_min_angle, new_edge = candidate
        if new_edge in topology.edge_to_triangles or new_edge in selected_new_edges:
            continue
        if new_min_q <= old_min_q + 1.0e-8 or new_min_angle <= old_min_angle + 0.05:
            continue
        updates.append((first, second, new_first, new_second))
        selected_triangles.update((first, second))
        selected_new_edges.add(new_edge)
        if len(updates) >= int(budget):
            break
    return updates


def _edge_flip_candidate(
    points: np.ndarray,
    tris: np.ndarray,
    edge: tuple[int, int],
    first: int,
    second: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, tuple[int, int]] | None:
    a, b = edge
    c_values = [int(value) for value in tris[first] if int(value) not in edge]
    d_values = [int(value) for value in tris[second] if int(value) not in edge]
    if len(c_values) != 1 or len(d_values) != 1 or c_values[0] == d_values[0]:
        return None
    c, d = c_values[0], d_values[0]
    new_edge = tuple(sorted((c, d)))
    old_pair = tris[[first, second]]
    new_pair = _orient_ccw(points, np.asarray([[c, d, a], [d, c, b]], dtype=int))
    old_geometry = triangle_geometry(points, old_pair)
    new_geometry = triangle_geometry(points, new_pair)
    if np.any(new_geometry["signed_area"] <= _area_tolerance(points, new_pair)):
        return None
    old_area = float(np.sum(old_geometry["area"]))
    new_area = float(np.sum(new_geometry["area"]))
    if abs(new_area - old_area) > 1.0e-8 * max(old_area, 1.0):
        return None
    return (
        new_pair[0],
        new_pair[1],
        float(np.min(old_geometry["quality"])),
        float(np.min(new_geometry["quality"])),
        float(np.min(old_geometry["angles_deg"])),
        float(np.min(new_geometry["angles_deg"])),
        new_edge,
    )


def _select_split_edges(
    points: np.ndarray,
    tris: np.ndarray,
    targets: np.ndarray,
    chains: list[list[int]],
    config: ThinTriangleRepairConfig,
    region_bbox_xy: tuple[float, float, float, float] | None,
    budget: int,
) -> list[tuple[tuple[int, int], list[int]]]:
    if budget <= 0:
        return []
    topology = build_edge_topology(len(points), tris)
    geometry = triangle_geometry(points, tris)
    min_angles = np.min(geometry["angles_deg"], axis=1)
    bad = (geometry["quality"] < float(config.quality_threshold)) | (min_angles < float(config.min_angle_deg))
    bad &= _triangle_region_mask(points, tris, region_bbox_xy)
    protected = chain_edges(chains)
    selected_triangles: set[int] = set()
    selected_edges: list[tuple[tuple[int, int], list[int]]] = []
    ordering = sorted(np.where(bad)[0], key=lambda index: (float(geometry["quality"][index]), float(min_angles[index]), int(index)))
    for index in ordering:
        tri = tris[int(index)]
        edges = sorted(_triangle_edges(tri), key=lambda edge: (-float(np.linalg.norm(points[edge[0]] - points[edge[1]])), edge))
        for edge in edges:
            attached = list(map(int, topology.edge_to_triangles.get(edge, [])))
            if edge in protected or len(attached) != 2 or any(item in selected_triangles for item in attached):
                continue
            length = float(np.linalg.norm(points[edge[0]] - points[edge[1]]))
            edge_target = _edge_target(targets, edge)
            if np.isfinite(edge_target):
                if length <= float(config.split_target_factor) * edge_target:
                    continue
            else:
                local_lengths = geometry["edge_lengths"][int(index)]
                if length <= 1.8 * max(float(np.min(local_lengths)), 1.0e-12):
                    continue
            if not _split_improves_patch(points, tris, edge, attached, config):
                continue
            selected_edges.append((edge, attached))
            selected_triangles.update(attached)
            break
        if len(selected_edges) >= int(budget):
            break
    return selected_edges


def _split_improves_patch(
    points: np.ndarray,
    tris: np.ndarray,
    edge: tuple[int, int],
    attached: list[int],
    config: ThinTriangleRepairConfig,
) -> bool:
    midpoint = 0.5 * (points[edge[0]] + points[edge[1]])
    trial_points = np.vstack([points, midpoint])
    new_node = len(points)
    new_tris: list[list[int]] = []
    for tri_index in attached:
        tri = tris[int(tri_index)]
        opposite = [int(value) for value in tri if int(value) not in edge]
        if len(opposite) != 1:
            return False
        c = opposite[0]
        new_tris.extend([[edge[0], new_node, c], [new_node, edge[1], c]])
    old_geometry = triangle_geometry(points, tris[np.asarray(attached, dtype=int)])
    new_array = _orient_ccw(trial_points, np.asarray(new_tris, dtype=int))
    new_geometry = triangle_geometry(trial_points, new_array)
    if np.any(new_geometry["signed_area"] <= _area_tolerance(trial_points, new_array)):
        return False
    old_bad = _geometry_bad_count(old_geometry, config.quality_threshold, config.min_angle_deg)
    new_bad = _geometry_bad_count(new_geometry, config.quality_threshold, config.min_angle_deg)
    return bool(
        new_bad < old_bad
        or (
            new_bad == old_bad
            and float(np.min(new_geometry["quality"])) > float(np.min(old_geometry["quality"])) + 1.0e-8
            and float(np.min(new_geometry["angles_deg"])) > float(np.min(old_geometry["angles_deg"])) + 0.05
        )
    )


def _apply_edge_splits(
    points: np.ndarray,
    tris: np.ndarray,
    fixed: np.ndarray,
    targets: np.ndarray,
    selected: list[tuple[tuple[int, int], list[int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    out_points = points.copy()
    out_fixed = fixed.copy()
    out_targets = targets.copy()
    remove: set[int] = set()
    additions: list[list[int]] = []
    records: list[tuple[int, int, int]] = []
    for edge, attached in selected:
        a, b = map(int, edge)
        new_node = len(out_points)
        out_points = np.vstack([out_points, 0.5 * (points[a] + points[b])])
        out_fixed = np.concatenate([out_fixed, np.asarray([False])])
        out_targets = np.concatenate([out_targets, np.asarray([_edge_target(out_targets, (a, b))])])
        records.append((int(new_node), a, b))
        for tri_index in attached:
            remove.add(int(tri_index))
            opposite = [int(value) for value in tris[int(tri_index)] if int(value) not in edge]
            if len(opposite) != 1:
                raise ValueError("Cannot split an edge whose attached triangle has invalid connectivity")
            c = opposite[0]
            additions.extend([[a, new_node, c], [new_node, b, c]])
    keep = np.asarray([index not in remove for index in range(len(tris))], dtype=bool)
    out_tris = np.vstack([tris[keep], np.asarray(additions, dtype=int)])
    out_tris = _orient_ccw(out_points, out_tris)
    return out_points, out_tris, out_fixed, out_targets, records


def _repair_transaction_ok(
    points: np.ndarray,
    tris: np.ndarray,
    fixed: np.ndarray,
    chains: list[list[int]],
    open_nodes: np.ndarray,
    original_points: np.ndarray,
    initial_components: int,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    require_improvement: bool,
) -> bool:
    geometry = triangle_geometry(points, tris)
    if np.any(geometry["signed_area"] <= _area_tolerance(points, tris)):
        return False
    topology = build_edge_topology(len(points), tris)
    if topology.nonmanifold_edges or len(topology.connected_component_sizes) != int(initial_components):
        return False
    integrity = constraint_integrity(topology, chains, open_nodes.tolist())
    if chains and not integrity["all_protected_edges_present"]:
        return False
    if len(open_nodes) and not integrity["open_boundary_ordered"]:
        return False
    original_fixed_mask = np.asarray(fixed[: len(original_points)], dtype=bool)
    if _maximum_shift(original_points[original_fixed_mask], points[: len(original_points)][original_fixed_mask]) > 1.0e-10:
        return False
    if not _global_nonregression(before, after):
        return False
    if require_improvement and not _defect_improved(before, after):
        return False
    return True


def _conditioning_energy(
    points: np.ndarray,
    tris: np.ndarray,
    edges: np.ndarray,
    rest_lengths: np.ndarray,
    affected: np.ndarray,
    shape_weight: float,
) -> float:
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    spring = 0.5 * np.sum(((lengths - rest_lengths) / np.maximum(rest_lengths, 1.0e-12)) ** 2)
    quality = triangle_geometry(points, tris[affected])["quality"]
    shape = 0.5 * float(shape_weight) * np.sum((1.0 - quality) ** 2)
    return float(spring + shape)


def _spring_shape_force(
    points: np.ndarray,
    tris: np.ndarray,
    edges: np.ndarray,
    rest_lengths: np.ndarray,
    affected: np.ndarray,
    shape_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    force = np.zeros_like(points)
    diagonal = np.zeros(len(points), dtype=float)
    for (a, b), rest in zip(edges, rest_lengths, strict=True):
        a = int(a)
        b = int(b)
        direction = points[b] - points[a]
        length = max(float(np.linalg.norm(direction)), 1.0e-12)
        stiffness = 1.0 / max(float(rest) ** 2, 1.0e-24)
        edge_force = stiffness * (length - float(rest)) * direction / length
        force[a] += edge_force
        force[b] -= edge_force
        diagonal[a] += stiffness
        diagonal[b] += stiffness
    if shape_weight > 0.0:
        for tri in tris[np.where(affected)[0]]:
            ids = np.asarray(tri, dtype=int)
            coords = points[ids]
            area2 = float(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
            area = 0.5 * area2
            if area <= 0.0:
                continue
            lengths2 = (
                float(np.dot(coords[1] - coords[2], coords[1] - coords[2]))
                + float(np.dot(coords[0] - coords[2], coords[0] - coords[2]))
                + float(np.dot(coords[0] - coords[1], coords[0] - coords[1]))
            )
            if lengths2 <= 1.0e-24:
                continue
            quality = 4.0 * np.sqrt(3.0) * area / lengths2
            grad_area = np.asarray(
                [
                    0.5 * _rotate90(coords[2] - coords[1]),
                    0.5 * _rotate90(coords[0] - coords[2]),
                    0.5 * _rotate90(coords[1] - coords[0]),
                ]
            )
            grad_sum = np.asarray(
                [
                    2.0 * (2.0 * coords[0] - coords[1] - coords[2]),
                    2.0 * (2.0 * coords[1] - coords[2] - coords[0]),
                    2.0 * (2.0 * coords[2] - coords[0] - coords[1]),
                ]
            )
            grad_quality = 4.0 * np.sqrt(3.0) * (lengths2 * grad_area - area * grad_sum) / (lengths2 * lengths2)
            shape_force = float(shape_weight) * (1.0 - quality) * grad_quality
            for local, node in enumerate(ids):
                force[int(node)] += shape_force[local]
                diagonal[int(node)] += float(shape_weight) / max(lengths2, 1.0e-24)
    return force, diagonal


def _graph_mobility(
    neighbors: list[set[int]],
    seed_nodes: np.ndarray,
    fixed: np.ndarray,
    ring_layers: int,
) -> np.ndarray:
    distance = np.full(len(neighbors), -1, dtype=int)
    queue: deque[int] = deque()
    for node in sorted(map(int, seed_nodes)):
        distance[node] = 0
        queue.append(node)
    while queue:
        node = queue.popleft()
        if distance[node] >= ring_layers:
            continue
        for neighbor in sorted(neighbors[node]):
            if distance[neighbor] < 0:
                distance[neighbor] = distance[node] + 1
                queue.append(int(neighbor))
    mobility = np.zeros(len(neighbors), dtype=float)
    selected = distance >= 0
    if ring_layers <= 0:
        mobility[selected] = 1.0
    else:
        mobility[selected] = 0.5 * (1.0 + np.cos(np.pi * distance[selected] / float(ring_layers + 1)))
    mobility[fixed] = 0.0
    return mobility


def _worst_seed_component(
    seeds: np.ndarray,
    geometry: dict[str, np.ndarray],
    topology: Any,
    config: SpringRelaxConfig,
) -> np.ndarray:
    """Select one edge-connected defect component for a single regional pass."""
    selected = set(map(int, np.where(seeds)[0]))
    adjacency: dict[int, set[int]] = {index: set() for index in selected}
    for attached in topology.edge_to_triangles.values():
        if len(attached) == 2:
            a, b = map(int, attached)
            if a in selected and b in selected:
                adjacency[a].add(b)
                adjacency[b].add(a)
    min_angles = np.min(geometry["angles_deg"], axis=1)
    quality = geometry["quality"]
    components: list[list[int]] = []
    unseen = set(selected)
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    if not components:
        return seeds

    def score(component: list[int]) -> tuple[float, int, int]:
        ids = np.asarray(component, dtype=int)
        q_deficit = np.maximum(
            0.0,
            (float(config.quality_threshold) - quality[ids]) / max(float(config.quality_threshold), 1.0e-12),
        )
        angle_deficit = np.maximum(
            0.0,
            (float(config.min_angle_deg) - min_angles[ids]) / max(float(config.min_angle_deg), 1.0e-12),
        )
        return (float(np.sum(q_deficit * q_deficit + angle_deficit * angle_deficit)), len(component), -component[0])

    chosen = max(components, key=score)
    out = np.zeros_like(seeds, dtype=bool)
    out[np.asarray(chosen, dtype=int)] = True
    return out


def _infer_node_sizes(points: np.ndarray, neighbors: list[set[int]], targets: np.ndarray) -> np.ndarray:
    sizes = np.asarray(targets, dtype=float).copy()
    all_lengths: list[float] = []
    local = np.full(len(points), np.nan, dtype=float)
    for node, node_neighbors in enumerate(neighbors):
        lengths = [float(np.linalg.norm(points[node] - points[neighbor])) for neighbor in node_neighbors]
        lengths = [value for value in lengths if np.isfinite(value) and value > 0.0]
        if lengths:
            local[node] = float(np.median(lengths))
            all_lengths.extend(lengths)
    fallback = float(np.median(all_lengths)) if all_lengths else 1.0
    local = np.where(np.isfinite(local) & (local > 0.0), local, fallback)
    sizes = np.where(np.isfinite(sizes) & (sizes > 0.0), sizes, local)
    return np.maximum(sizes, 1.0e-9)


def _rest_lengths(points: np.ndarray, edges: np.ndarray, node_sizes: np.ndarray) -> np.ndarray:
    current = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    a = node_sizes[edges[:, 0]]
    b = node_sizes[edges[:, 1]]
    harmonic = 2.0 / np.maximum(1.0 / a + 1.0 / b, 1.0e-30)
    valid = np.isfinite(harmonic) & (harmonic > 0.0) & np.isfinite(current) & (current > 0.0)
    scale = float(np.median(current[valid] / harmonic[valid])) if np.any(valid) else 1.0
    desired = scale * harmonic
    return np.clip(desired, 0.50 * np.maximum(current, 1.0e-9), 2.0 * np.maximum(current, 1.0e-9))


def _area_transition_summary(
    points: np.ndarray,
    tris: np.ndarray,
    topology: Any,
    sample_targets: Callable[[np.ndarray], np.ndarray],
    config: AreaTransitionRelaxConfig,
    region_bbox_xy: tuple[float, float, float, float] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    geometry = triangle_geometry(points, tris)
    areas = np.asarray(geometry["area"], dtype=float)
    if not len(tris):
        summary = {
            "shared_edge_count": 0,
            "candidate_pair_count": 0,
            "raw_trigger_pair_count": 0,
            "high_gradient_trigger_pair_count": 0,
            "maximum_adjacent_area_change": 0.0,
            "adjacent_area_change_p95": 0.0,
            "adjacent_area_change_above_threshold_count": 0,
            "target_gradient_maximum": 0.0,
            "target_gradient_p95": 0.0,
            "normalized_log_jump_maximum": 0.0,
            "normalized_log_jump_p95": 0.0,
            "normalized_log_jump_above_threshold_count": 0,
            "transition_severity_sum": 0.0,
            "l_over_h": _l_over_h_summary(np.empty(0, dtype=float), config.l_over_h_threshold),
            "worst_pairs": [],
        }
        return summary, []

    centroids = np.mean(points[tris], axis=1)
    targets = sample_targets(centroids)
    target_area = (np.sqrt(3.0) / 4.0) * targets * targets
    normalized_area = np.maximum(areas, 1.0e-30) / np.maximum(target_area, 1.0e-30)
    l_over_h = np.max(geometry["edge_lengths"], axis=1) / np.maximum(targets, 1.0e-30)
    pairs: list[dict[str, Any]] = []
    for edge, attached_values in sorted(topology.edge_to_triangles.items()):
        if len(attached_values) != 2:
            continue
        first, second = map(int, attached_values)
        area_first = float(areas[first])
        area_second = float(areas[second])
        area_change = abs(area_first - area_second) / max(area_first, area_second, 1.0e-30)
        centroid_distance = max(float(np.linalg.norm(centroids[first] - centroids[second])), 1.0e-12)
        target_gradient = abs(float(targets[first] - targets[second])) / centroid_distance
        normalized_log_jump = abs(float(np.log(normalized_area[first]) - np.log(normalized_area[second])))
        raw_trigger = bool(area_change > float(config.raw_area_change_threshold))
        high_gradient_trigger = bool(
            target_gradient > float(config.target_gradient_threshold)
            and area_change > float(config.high_gradient_area_change_threshold)
            and normalized_log_jump > float(config.normalized_log_jump_threshold)
        )
        pair_center = 0.5 * (centroids[first] + centroids[second])
        if region_bbox_xy is None:
            in_region = True
        else:
            minx, miny, maxx, maxy = map(float, region_bbox_xy)
            in_region = bool(minx <= pair_center[0] <= maxx and miny <= pair_center[1] <= maxy)
        raw_excess = max(0.0, area_change - float(config.raw_area_change_threshold))
        high_area_excess = max(0.0, area_change - float(config.high_gradient_area_change_threshold))
        normalized_excess = max(0.0, normalized_log_jump - float(config.normalized_log_jump_threshold))
        severity = raw_excess * raw_excess
        if high_gradient_trigger:
            severity += high_area_excess * high_area_excess + normalized_excess * normalized_excess
        score = severity + 1.0e-6 * area_change
        pairs.append(
            {
                "edge": tuple(map(int, edge)),
                "triangles": (first, second),
                "area_change": float(area_change),
                "target_gradient": float(target_gradient),
                "normalized_log_jump": float(normalized_log_jump),
                "raw_trigger": raw_trigger,
                "high_gradient_trigger": high_gradient_trigger,
                "in_region": bool(in_region),
                "triggered": bool(in_region and (raw_trigger or high_gradient_trigger)),
                "severity": float(severity),
                "score": float(score),
            }
        )

    area_changes = np.asarray([pair["area_change"] for pair in pairs], dtype=float)
    target_gradients = np.asarray([pair["target_gradient"] for pair in pairs], dtype=float)
    normalized_jumps = np.asarray([pair["normalized_log_jump"] for pair in pairs], dtype=float)
    triggered = [pair for pair in pairs if pair["triggered"]]
    worst = sorted(
        triggered if triggered else pairs,
        key=lambda pair: (float(pair["score"]), float(pair["area_change"]), -int(pair["edge"][0]), -int(pair["edge"][1])),
        reverse=True,
    )[:20]
    summary = {
        "shared_edge_count": int(len(pairs)),
        "candidate_pair_count": int(len(triggered)),
        "raw_trigger_pair_count": int(sum(bool(pair["in_region"] and pair["raw_trigger"]) for pair in pairs)),
        "high_gradient_trigger_pair_count": int(
            sum(bool(pair["in_region"] and pair["high_gradient_trigger"]) for pair in pairs)
        ),
        "maximum_adjacent_area_change": _maximum_or_zero(area_changes),
        "adjacent_area_change_p95": _quantile_or_zero(area_changes, 0.95),
        "adjacent_area_change_above_threshold_count": int(
            np.sum(area_changes > float(config.raw_area_change_threshold))
        ),
        "target_gradient_maximum": _maximum_or_zero(target_gradients),
        "target_gradient_p95": _quantile_or_zero(target_gradients, 0.95),
        "normalized_log_jump_maximum": _maximum_or_zero(normalized_jumps),
        "normalized_log_jump_p95": _quantile_or_zero(normalized_jumps, 0.95),
        "normalized_log_jump_above_threshold_count": int(
            np.sum(normalized_jumps > float(config.normalized_log_jump_threshold))
        ),
        "transition_severity_sum": float(sum(float(pair["severity"]) for pair in pairs)),
        "l_over_h": _l_over_h_summary(l_over_h, config.l_over_h_threshold),
        "worst_pairs": [_public_transition_pair(pair) for pair in worst],
    }
    return summary, pairs


def _public_transition_pair(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge": list(map(int, pair["edge"])),
        "triangles": list(map(int, pair["triangles"])),
        "area_change": float(pair["area_change"]),
        "target_gradient": float(pair["target_gradient"]),
        "normalized_log_jump": float(pair["normalized_log_jump"]),
        "raw_trigger": bool(pair["raw_trigger"]),
        "high_gradient_trigger": bool(pair["high_gradient_trigger"]),
        "in_region": bool(pair["in_region"]),
        "triggered": bool(pair["triggered"]),
        "severity": float(pair["severity"]),
        "score": float(pair["score"]),
    }


def _l_over_h_summary(values: np.ndarray, threshold: float) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {
        "definition": "maximum_triangle_edge_over_eulerian_centroid_target",
        "maximum": _maximum_or_zero(values),
        "p95": _quantile_or_zero(values, 0.95),
        "median": _quantile_or_zero(values, 0.50),
        "count_above_threshold": int(np.sum(values > float(threshold))),
        "threshold": float(threshold),
    }


def _area_transition_candidate_failures(
    points: np.ndarray,
    tris: np.ndarray,
    fixed: np.ndarray,
    chains: list[list[int]],
    open_nodes: np.ndarray,
    original_points: np.ndarray,
    original_fixed: np.ndarray,
    initial_component_count: int,
    stage_before: dict[str, Any],
    current_before: dict[str, Any],
    candidate_after: dict[str, Any],
    stage_quality: dict[str, Any],
    candidate_quality: dict[str, Any],
    displacement_fraction: float,
    config: AreaTransitionRelaxConfig,
) -> list[str]:
    failures = _area_transition_mesh_failures(
        points,
        tris,
        fixed,
        chains,
        open_nodes,
        original_points,
        original_fixed,
        initial_component_count,
    )
    tolerance = 1.0e-12
    if not _global_nonregression(stage_quality, candidate_quality):
        failures.append("stage_shape_tail_regression")
    if candidate_after["maximum_adjacent_area_change"] > current_before["maximum_adjacent_area_change"] + tolerance:
        failures.append("adjacent_area_change_maximum_regression")
    if (
        candidate_after["adjacent_area_change_above_threshold_count"]
        > current_before["adjacent_area_change_above_threshold_count"]
    ):
        failures.append("new_raw_area_transition_defect")
    if candidate_after["candidate_pair_count"] > current_before["candidate_pair_count"]:
        failures.append("new_triggered_transition_pair")
    severity_drop = float(current_before["transition_severity_sum"] - candidate_after["transition_severity_sum"])
    required_drop = float(config.minimum_severity_reduction_fraction) * max(
        float(current_before["transition_severity_sum"]), 1.0e-12
    )
    count_improved = bool(
        candidate_after["candidate_pair_count"] < current_before["candidate_pair_count"]
        or candidate_after["adjacent_area_change_above_threshold_count"]
        < current_before["adjacent_area_change_above_threshold_count"]
    )
    if not count_improved and severity_drop + tolerance < required_drop:
        failures.append("insufficient_transition_severity_reduction")
    failures.extend(_area_transition_stage_baseline_failures(stage_before, candidate_after, config))
    if displacement_fraction > float(config.max_total_displacement_fraction) + tolerance:
        failures.append("total_displacement_above_target_fraction")
    return sorted(set(failures))


def _area_transition_final_failures(
    points: np.ndarray,
    tris: np.ndarray,
    fixed: np.ndarray,
    chains: list[list[int]],
    open_nodes: np.ndarray,
    original_points: np.ndarray,
    original_fixed: np.ndarray,
    initial_component_count: int,
    stage_before: dict[str, Any],
    stage_after: dict[str, Any],
    stage_quality: dict[str, Any],
    final_quality: dict[str, Any],
    displacement_fraction: float,
    config: AreaTransitionRelaxConfig,
) -> list[str]:
    failures = _area_transition_mesh_failures(
        points,
        tris,
        fixed,
        chains,
        open_nodes,
        original_points,
        original_fixed,
        initial_component_count,
    )
    if not _global_nonregression(stage_quality, final_quality):
        failures.append("stage_shape_tail_regression")
    failures.extend(_area_transition_stage_baseline_failures(stage_before, stage_after, config))
    if stage_after["transition_severity_sum"] >= stage_before["transition_severity_sum"] - 1.0e-12:
        failures.append("transition_severity_not_improved")
    if displacement_fraction > float(config.max_total_displacement_fraction) + 1.0e-12:
        failures.append("total_displacement_above_target_fraction")
    return sorted(set(failures))


def _area_transition_stage_baseline_failures(
    stage_before: dict[str, Any],
    candidate_after: dict[str, Any],
    config: AreaTransitionRelaxConfig,
) -> list[str]:
    failures: list[str] = []
    tolerance = 1.0e-12
    if candidate_after["maximum_adjacent_area_change"] > stage_before["maximum_adjacent_area_change"] + tolerance:
        failures.append("stage_area_transition_maximum_regression")
    if (
        candidate_after["adjacent_area_change_above_threshold_count"]
        > stage_before["adjacent_area_change_above_threshold_count"]
    ):
        failures.append("stage_raw_area_transition_count_regression")
    if candidate_after["candidate_pair_count"] > stage_before["candidate_pair_count"]:
        failures.append("stage_triggered_transition_count_regression")
    if (
        candidate_after["normalized_log_jump_above_threshold_count"]
        > stage_before["normalized_log_jump_above_threshold_count"]
    ):
        failures.append("stage_normalized_transition_count_regression")
    normalized_tolerance = 1.0e-9
    if candidate_after["normalized_log_jump_p95"] > stage_before["normalized_log_jump_p95"] + normalized_tolerance:
        failures.append("stage_normalized_transition_p95_regression")
    before_lh = stage_before["l_over_h"]
    after_lh = candidate_after["l_over_h"]
    relative_tolerance = max(0.0, float(config.l_over_h_relative_tolerance))
    if after_lh["maximum"] > before_lh["maximum"] * (1.0 + relative_tolerance) + 1.0e-12:
        failures.append("stage_l_over_h_maximum_regression")
    if after_lh["p95"] > before_lh["p95"] * (1.0 + relative_tolerance) + 1.0e-12:
        failures.append("stage_l_over_h_p95_regression")
    if after_lh["count_above_threshold"] > before_lh["count_above_threshold"]:
        failures.append("stage_l_over_h_count_regression")
    return failures


def _area_transition_mesh_failures(
    points: np.ndarray,
    tris: np.ndarray,
    fixed: np.ndarray,
    chains: list[list[int]],
    open_nodes: np.ndarray,
    original_points: np.ndarray,
    original_fixed: np.ndarray,
    initial_component_count: int,
) -> list[str]:
    failures: list[str] = []
    geometry = triangle_geometry(points, tris)
    if np.any(geometry["signed_area"] <= _area_tolerance(points, tris)):
        failures.append("nonpositive_triangle_area")
    topology = build_edge_topology(len(points), tris)
    if topology.nonmanifold_edges:
        failures.append("nonmanifold_edge_introduced")
    if len(topology.connected_component_sizes) != int(initial_component_count):
        failures.append("connected_component_count_changed")
    integrity = constraint_integrity(topology, chains, open_nodes.tolist())
    if chains and not integrity["all_protected_edges_present"]:
        failures.append("protected_edge_missing")
    if len(open_nodes) and not integrity["open_boundary_ordered"]:
        failures.append("open_boundary_order_changed")
    if _maximum_shift(original_fixed, points[fixed]) > 1.0e-10:
        failures.append("fixed_boundary_coordinate_changed")
    if len(points) != len(original_points):
        failures.append("node_count_changed")
    return failures


def _maximum_displacement_fraction(before: np.ndarray, after: np.ndarray, targets: np.ndarray) -> float:
    if not len(before):
        return 0.0
    displacement = np.linalg.norm(np.asarray(after, dtype=float) - np.asarray(before, dtype=float), axis=1)
    return float(np.max(displacement / np.maximum(np.asarray(targets, dtype=float), 1.0e-12)))


def _quantile_or_zero(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, float(quantile))) if len(values) else 0.0


def _maximum_or_zero(values: np.ndarray) -> float:
    return float(np.max(values)) if len(values) else 0.0


def _quality_summary(points: np.ndarray, tris: np.ndarray, quality_threshold: float, min_angle_deg: float) -> dict[str, Any]:
    geometry = triangle_geometry(points, np.asarray(tris, dtype=int))
    quality = geometry["quality"]
    min_angles = np.min(geometry["angles_deg"], axis=1) if len(geometry["angles_deg"]) else np.empty(0, dtype=float)
    q_mean = float(np.mean(quality)) if len(quality) else 0.0
    q_std = float(np.std(quality)) if len(quality) else 0.0
    return {
        "triangle_count": int(len(quality)),
        "q_min": float(np.min(quality)) if len(quality) else 0.0,
        "q_p01": float(np.quantile(quality, 0.01)) if len(quality) else 0.0,
        "q_mean": q_mean,
        "q_l3_sigma": float(q_mean - 3.0 * q_std),
        "minimum_angle_min_deg": float(np.min(min_angles)) if len(min_angles) else 0.0,
        "minimum_angle_p01_deg": float(np.quantile(min_angles, 0.01)) if len(min_angles) else 0.0,
        "quality_below_threshold_count": int(np.sum(quality < float(quality_threshold))),
        "angle_below_threshold_count": int(np.sum(min_angles < float(min_angle_deg))),
        "thin_triangle_count": int(np.sum((quality < float(quality_threshold)) | (min_angles < float(min_angle_deg)))),
        "thin_severity_sum": float(
            np.sum(np.maximum(0.0, (float(quality_threshold) - quality) / max(float(quality_threshold), 1.0e-12)) ** 2)
            + np.sum(np.maximum(0.0, (float(min_angle_deg) - min_angles) / max(float(min_angle_deg), 1.0e-12)) ** 2)
        ),
    }


def _global_nonregression(before: dict[str, Any], after: dict[str, Any]) -> bool:
    quality_tolerance = 1.0e-9
    angle_tolerance_deg = 1.0e-3
    return bool(
        after["q_min"] + quality_tolerance >= before["q_min"]
        and after["minimum_angle_min_deg"] + 1.0e-6 >= before["minimum_angle_min_deg"]
        and after["q_l3_sigma"] + quality_tolerance >= before["q_l3_sigma"]
        and after["q_p01"] + quality_tolerance >= before["q_p01"]
        and after["minimum_angle_p01_deg"] + angle_tolerance_deg >= before["minimum_angle_p01_deg"]
        and after["quality_below_threshold_count"] <= before["quality_below_threshold_count"]
        and after["angle_below_threshold_count"] <= before["angle_below_threshold_count"]
    )


def _local_step_nonregression(before: dict[str, Any], after: dict[str, Any]) -> bool:
    # A spring patch can temporarily redistribute the identity of the worst
    # element before converging.  Permit at most a two-percent intermediate
    # dip; the final transaction gate below still requires exact nonregression.
    tolerance = 1.0e-10
    return bool(
        after["q_min"] + tolerance >= 0.98 * before["q_min"]
        and after["minimum_angle_min_deg"] + tolerance >= 0.98 * before["minimum_angle_min_deg"]
        and after["thin_triangle_count"] <= before["thin_triangle_count"]
    )


def _local_nonregression(before: dict[str, Any], after: dict[str, Any]) -> bool:
    tolerance = 1.0e-10
    return bool(
        after["q_min"] + tolerance >= 0.98 * before["q_min"]
        and after["minimum_angle_min_deg"] + tolerance >= 0.98 * before["minimum_angle_min_deg"]
        and after["thin_triangle_count"] <= before["thin_triangle_count"]
        and after["thin_severity_sum"] <= before["thin_severity_sum"] + tolerance
    )


def _defect_improved(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return bool(
        after["thin_triangle_count"] < before["thin_triangle_count"]
        or (
            after["thin_triangle_count"] == before["thin_triangle_count"]
            and after["thin_severity_sum"] < before["thin_severity_sum"] - 1.0e-10
            and after["q_min"] >= before["q_min"] - 1.0e-10
            and after["minimum_angle_min_deg"] >= before["minimum_angle_min_deg"] - 1.0e-10
        )
    )


def _triangle_region_mask(
    points: np.ndarray,
    tris: np.ndarray,
    region_bbox_xy: tuple[float, float, float, float] | None,
) -> np.ndarray:
    if region_bbox_xy is None:
        return np.ones(len(tris), dtype=bool)
    minx, miny, maxx, maxy = map(float, region_bbox_xy)
    centroids = points[tris].mean(axis=1)
    return (
        (centroids[:, 0] >= minx)
        & (centroids[:, 0] <= maxx)
        & (centroids[:, 1] >= miny)
        & (centroids[:, 1] <= maxy)
    )


def _geometry_bad_count(geometry: dict[str, np.ndarray], quality_threshold: float, min_angle_deg: float) -> int:
    min_angles = np.min(geometry["angles_deg"], axis=1) if len(geometry["angles_deg"]) else np.empty(0)
    return int(np.sum((geometry["quality"] < float(quality_threshold)) | (min_angles < float(min_angle_deg))))


def _edge_target(targets: np.ndarray, edge: tuple[int, int]) -> float:
    values = np.asarray([targets[int(edge[0])], targets[int(edge[1])]], dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if not len(values):
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    return float(2.0 / np.sum(1.0 / values))


def _triangle_edges(tri: np.ndarray) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(tri[0]), int(tri[1])))),
        tuple(sorted((int(tri[1]), int(tri[2])))),
        tuple(sorted((int(tri[2]), int(tri[0])))),
    }


def _orient_ccw(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    out = np.asarray(tris, dtype=int).copy()
    if not len(out):
        return out.reshape((0, 3))
    coords = points[out]
    area2 = (coords[:, 1, 0] - coords[:, 0, 0]) * (coords[:, 2, 1] - coords[:, 0, 1]) - (
        coords[:, 1, 1] - coords[:, 0, 1]
    ) * (coords[:, 2, 0] - coords[:, 0, 0])
    negative = area2 < 0.0
    out[negative, 1], out[negative, 2] = out[negative, 2].copy(), out[negative, 1].copy()
    return out


def _area_tolerance(points: np.ndarray, tris: np.ndarray) -> float:
    if not len(tris):
        return 1.0e-14
    lengths = triangle_geometry(points, tris)["edge_lengths"]
    scale = float(np.median(lengths[lengths > 0.0])) if np.any(lengths > 0.0) else 1.0
    return max(1.0e-14, 1.0e-12 * scale * scale)


def _normalized_targets(values: np.ndarray | None, node_count: int) -> np.ndarray:
    if values is None:
        return np.full(int(node_count), np.nan, dtype=float)
    out = np.asarray(values, dtype=float).copy()
    if out.shape != (int(node_count),):
        raise ValueError("target_spacing_m must have one value per node")
    return out


def _validate_area_transition_config(config: AreaTransitionRelaxConfig) -> None:
    if int(config.max_patches) < 0:
        raise ValueError("max_patches must be nonnegative")
    if int(config.max_attempt_multiplier) < 1:
        raise ValueError("max_attempt_multiplier must be at least one")
    if not 0.0 <= float(config.high_gradient_area_change_threshold) < 1.0:
        raise ValueError("high_gradient_area_change_threshold must lie in [0, 1)")
    if not 0.0 <= float(config.raw_area_change_threshold) < 1.0:
        raise ValueError("raw_area_change_threshold must lie in [0, 1)")
    if float(config.high_gradient_area_change_threshold) > float(config.raw_area_change_threshold):
        raise ValueError("high_gradient_area_change_threshold must not exceed raw_area_change_threshold")
    if float(config.target_gradient_threshold) < 0.0:
        raise ValueError("target_gradient_threshold must be nonnegative")
    if float(config.normalized_log_jump_threshold) < 0.0:
        raise ValueError("normalized_log_jump_threshold must be nonnegative")
    if not 0.0 <= float(config.minimum_severity_reduction_fraction) < 1.0:
        raise ValueError("minimum_severity_reduction_fraction must lie in [0, 1)")
    if float(config.l_over_h_threshold) <= 0.0:
        raise ValueError("l_over_h_threshold must be positive")
    if float(config.l_over_h_relative_tolerance) < 0.0:
        raise ValueError("l_over_h_relative_tolerance must be nonnegative")
    if float(config.max_total_displacement_fraction) <= 0.0:
        raise ValueError("max_total_displacement_fraction must be positive")


def _validate_mesh_arrays(points: np.ndarray, tris: np.ndarray, fixed: np.ndarray) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("nodes_xy must have shape (node, 2)")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("triangles must have shape (element, 3)")
    if fixed.shape != (len(points),):
        raise ValueError("fixed_node_mask must have one value per node")
    if len(tris) and (int(np.min(tris)) < 0 or int(np.max(tris)) >= len(points)):
        raise ValueError("triangles contain an out-of-range node index")


def _rotate90(vector: np.ndarray) -> np.ndarray:
    return np.asarray([-float(vector[1]), float(vector[0])], dtype=float)


def _maximum_shift(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(after - before, axis=1))) if len(before) else 0.0


def _config_dict(config: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in config.__dict__.items():
        result[key] = _config_dict(value) if hasattr(value, "__dict__") else value
    return result
