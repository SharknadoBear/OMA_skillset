"""Pure-Python OceanMesh-style constrained Delaunay refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import Point

from .boundary import BoundaryNodes
from .metrics import build_edge_topology, constraint_integrity, triangle_geometry
from .local_topology import AggressiveConditioningConfig, condition_mesh_aggressive
from .projection import unproject_points
from .regional_conditioning import (
    AreaTransitionRelaxConfig,
    SpringRelaxConfig,
    ThinTriangleRepairConfig,
    relax_mesh_area_transitions,
    relax_mesh_spring,
    repair_thin_triangles,
)
from .size_field import SizeField


@dataclass(frozen=True)
class MeshConfig:
    max_constraint_iterations: int = 8
    refine_iterations: int = 3
    smooth_iterations: int = 8
    max_interior_points: int = 80_000
    max_refine_insertions_per_iter: int = 1500
    size_overrun_factor: float = 1.55
    min_angle_refine_deg: float = 24.0
    adaptive_seed: bool = False
    regional_spring_relaxation: bool = True
    spring_relax_iterations: int = 20
    spring_relax_quality_threshold: float = 0.40
    spring_relax_min_angle_deg: float = 28.0
    spring_relax_ring_layers: int = 3
    spring_relax_shape_weight: float = 0.20
    thin_triangle_repair: bool = True
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


def generate_mesh(
    boundary: BoundaryNodes,
    size_field: SizeField,
    config: MeshConfig,
    progress_callback: ProgressCallback | None = None,
) -> MeshResult:
    """Generate a constrained Delaunay mesh with boundary midpoint recovery."""
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

    interior = _interior_seed_points(boundary, size_field, config)
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
        additions = _refinement_points(points_arr, triangles, boundary, size_field, config)
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
    def _sample_size_field_targets(sample_points_xy: np.ndarray) -> np.ndarray:
        sampled_lonlat = unproject_points(np.asarray(sample_points_xy, dtype=float), boundary.projection)
        return size_field.sample(sampled_lonlat[:, 0], sampled_lonlat[:, 1])

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
    spring_config = SpringRelaxConfig(
        enabled=bool(config.regional_spring_relaxation),
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
        enabled=bool(config.thin_triangle_repair),
        quality_threshold=float(config.thin_triangle_quality_threshold),
        min_angle_deg=float(config.thin_triangle_min_angle_deg),
        max_passes=int(config.thin_triangle_max_passes),
        max_flips=int(config.thin_triangle_max_flips),
        max_insertions=int(config.thin_triangle_max_insertions),
        relaxation_config=SpringRelaxConfig(
            enabled=bool(config.regional_spring_relaxation),
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

    effective_conditioning_profile = str(config.conditioning_profile)
    if effective_conditioning_profile == "auto":
        effective_conditioning_profile = "aggressive-local-v2" if boundary.adaptive_resolution else "guarded-v1"
    if effective_conditioning_profile not in {"guarded-v1", "aggressive-local-v2", "none"}:
        raise ValueError("conditioning_profile must be auto, guarded-v1, aggressive-local-v2, or none")
    aggressive_result = None
    node_lineage = np.arange(len(points_arr), dtype=int)
    if effective_conditioning_profile == "aggressive-local-v2":
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
            config=AggressiveConditioningConfig(
                enabled=True,
                max_rounds=int(config.aggressive_conditioning_rounds),
                boundary_edit_policy=str(config.aggressive_boundary_edit_policy),
                max_prunes_per_round=int(config.aggressive_max_prunes_per_round),
                max_valence_removals_per_round=int(config.aggressive_max_valence_repairs_per_round),
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

    area_transition_config = AreaTransitionRelaxConfig(
        enabled=bool(config.area_transition_relaxation),
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
    conditioning_report = {
        "schema_version": "fvcom_generation_conditioning_v3",
        "profile": effective_conditioning_profile,
        "stage_order": [
            "spring-relax-v1",
            "thin-repair-v1",
            "aggressive-local-v2" if aggressive_result is not None else "aggressive-local-disabled",
            "area-transition-relax-v1",
            "terminal-constraint-audit",
        ],
        "spring_relaxation": spring_result.report,
        "thin_triangle_repair": thin_result.report,
        "aggressive_local_topology": aggressive_result.report if aggressive_result is not None else {"enabled": False},
        "mesh_edit_ledger": aggressive_result.edit_ledger if aggressive_result is not None else [],
        "area_transition_relaxation": area_transition_result.report,
        "baseline_terminal_audit": baseline_audit,
        "terminal_audit": terminal_audit,
        "terminal_rollback_applied": terminal_rollback,
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
    positive = bool(len(triangles) and np.all(geometry["signed_area"] > 0.0))
    constraints_ok = bool(not chains or integrity["all_protected_edges_present"])
    obc_ok = bool(not len(open_nodes_zero_based) or integrity["open_boundary_ordered"])
    one_component = bool(len(topology.connected_component_sizes) == 1)
    manifold = bool(not topology.nonmanifold_edges)
    passed = bool(positive and constraints_ok and obc_ok and one_component and manifold and boundary_shift <= 1.0e-10 and not missing_hard)
    return {
        "passed": passed,
        "positive_signed_areas": positive,
        "all_protected_edges_present": constraints_ok,
        "open_boundary_ordered": obc_ok,
        "connected_component_count": int(len(topology.connected_component_sizes)),
        "nonmanifold_edge_count": int(len(topology.nonmanifold_edges)),
        "original_boundary_coordinate_max_shift_m": boundary_shift,
        "surviving_original_boundary_node_count": int(len(boundary_survivors)),
        "removed_original_boundary_node_count": int(np.count_nonzero(original_mask) - len(boundary_survivors)),
        "missing_hard_anchor_count": int(len(missing_hard)),
        "missing_hard_anchors": missing_hard[:100],
        "constraint_integrity": integrity,
    }


def _progress(callback: ProgressCallback | None, message: str, fraction: float, extra: dict[str, Any] | None = None) -> None:
    if callback is not None:
        callback(message, float(fraction), extra)


def _interior_seed_points(boundary: BoundaryNodes, size_field: SizeField, config: MeshConfig) -> np.ndarray:
    if config.adaptive_seed or boundary.adaptive_resolution:
        return _adaptive_quadtree_seed_points(boundary, size_field, config)
    domain = boundary.domain_polygon_xy
    minx, miny, maxx, maxy = domain.bounds
    area = max(float(domain.area), 1.0)
    spacing = max(float(np.nanmedian(size_field.size)), float(np.sqrt(area / max(config.max_interior_points, 1))))
    spacing = max(spacing, 10.0)
    xs = np.arange(minx + 0.5 * spacing, maxx, spacing)
    ys = np.arange(miny + 0.5 * spacing, maxy, spacing * np.sqrt(3.0) / 2.0)
    pts = []
    for row, y in enumerate(ys):
        offset = 0.5 * spacing if row % 2 else 0.0
        for x in xs + offset:
            point = Point(float(x), float(y))
            if domain.contains(point):
                pts.append((float(x), float(y)))
            if len(pts) >= config.max_interior_points:
                break
        if len(pts) >= config.max_interior_points:
            break
    return np.asarray(pts, dtype=float)


def _adaptive_quadtree_seed_points(boundary: BoundaryNodes, size_field: SizeField, config: MeshConfig) -> np.ndarray:
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
    accepted: list[tuple[float, float]] = []
    for x, y, target in sorted(leaves, key=lambda item: (item[1], item[0])):
        point = Point(float(x), float(y))
        if not domain.contains(point):
            continue
        if boundary_tree is not None and float(boundary_tree.query([x, y])[0]) < 0.35 * target:
            continue
        accepted.append((float(x), float(y)))
        if len(accepted) > int(config.max_interior_points):
            raise ValueError("Adaptive quadtree seeds exceed --max-interior-points; raise the cap or coarsen the profile")
    return np.asarray(accepted, dtype=float)


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


def _refinement_points(points: np.ndarray, triangles: np.ndarray, boundary: BoundaryNodes, size_field: SizeField, config: MeshConfig) -> np.ndarray:
    additions = []
    if not len(triangles):
        return np.empty((0, 2), dtype=float)
    tree = cKDTree(points)
    triangle_coords = points[triangles]
    centroids = triangle_coords.mean(axis=1)
    centroid_lonlat = unproject_points(centroids, boundary.projection)
    targets = size_field.sample(centroid_lonlat[:, 0], centroid_lonlat[:, 1])
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
        if not boundary.domain_polygon_xy.contains(Point(float(candidate[0]), float(candidate[1]))):
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
        area2 = np.cross(coords[1] - coords[0], coords[2] - coords[0])
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
