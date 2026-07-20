"""Fixed-connectivity edge-angle-barrier relaxation for systematic V5.

The interaction engine intentionally never performs a global Delaunay rebuild.
It moves only explicitly movable vertices, checks positive geometry after every
trial step, and leaves connectivity changes to guarded local topology tools.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

import numpy as np

from .metrics import build_edge_topology, triangle_geometry


@dataclass(frozen=True)
class InteractionRelaxationConfig:
    iterations: int = 25
    checkpoint_interval: int = 10
    superthin_quality_threshold: float = 0.10
    superthin_min_angle_deg: float = 5.0
    superthin_trigger: int = 25
    edge_weight: float = 0.22
    angle_weight: float = 0.38
    area_barrier_weight: float = 0.55
    target_quality: float = 0.72
    minimum_normalized_area: float = 0.025
    damping: float = 0.30
    maximum_step_fraction: float = 0.06
    minimum_step_scale: float = 1.0 / 128.0
    maximum_rejected_steps: int = 3
    q_l3_decrease_tolerance: float = 2.0e-6
    plateau_gain: float = 1.0e-5
    plateau_checkpoints: int = 2
    deadline_monotonic_s: float | None = None


@dataclass
class InteractionCheckpoint:
    iteration: int
    nodes_xy: np.ndarray
    metrics: dict[str, Any]


@dataclass
class InteractionRelaxationResult:
    nodes_xy: np.ndarray
    iterations_completed: int
    checkpoints: list[InteractionCheckpoint]
    report: dict[str, Any]


def relax_mesh_interaction(
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    fixed_node_mask: np.ndarray,
    *,
    target_spacing_m: np.ndarray,
    config: InteractionRelaxationConfig | None = None,
) -> InteractionRelaxationResult:
    """Run a bounded interaction burst without changing connectivity."""
    config = config or InteractionRelaxationConfig()
    points = np.asarray(nodes_xy, dtype=float).copy()
    triangles = np.asarray(triangles, dtype=int)
    fixed = np.asarray(fixed_node_mask, dtype=bool)
    targets = np.asarray(target_spacing_m, dtype=float)
    if len(points) != len(fixed) or len(points) != len(targets):
        raise ValueError("points, fixed_node_mask, and target_spacing_m must have matching lengths")
    if np.any(~np.isfinite(points)) or np.any(~np.isfinite(targets)) or np.any(targets <= 0.0):
        raise ValueError("interaction relaxation requires finite coordinates and positive targets")
    topology = build_edge_topology(len(points), triangles)
    edges = np.asarray(sorted(topology.edge_to_triangles), dtype=int)
    started = time.perf_counter()
    initial = interaction_metrics(points, triangles, config)
    current = dict(initial)
    checkpoints = [InteractionCheckpoint(0, points.copy(), dict(initial))]
    rejected_steps = 0
    rejection_streak = 0
    stop_reason = "iteration_limit"
    accepted_scales: list[float] = []
    checkpoint_gains: list[float] = []
    for iteration in range(1, max(0, int(config.iterations)) + 1):
        if _deadline_reached(config):
            stop_reason = "deadline"
            break
        displacement = _interaction_displacement(
            points,
            triangles,
            edges,
            targets,
            fixed,
            config,
        )
        if not np.any(np.linalg.norm(displacement, axis=1) > 0.0):
            stop_reason = "stationary"
            break
        accepted = False
        scale = 1.0
        candidate = points
        trial_metrics = current
        while scale + 1.0e-15 >= float(config.minimum_step_scale):
            if _deadline_reached(config):
                stop_reason = "deadline"
                break
            trial = points + scale * displacement
            trial[fixed] = points[fixed]
            geometry = triangle_geometry(trial, triangles)
            tolerance = _area_tolerance(trial)
            if np.all(np.isfinite(trial)) and np.all(geometry["signed_area"] > tolerance):
                metrics = interaction_metrics(
                    trial,
                    triangles,
                    config,
                    geometry=geometry,
                )
                objective_improved = bool(
                    float(metrics["objective"])
                    < float(current["objective"]) - 1.0e-12
                )
                q_safe = bool(
                    float(metrics["q_l3_sigma"])
                    + float(config.q_l3_decrease_tolerance)
                    >= float(current["q_l3_sigma"])
                )
                if objective_improved and q_safe:
                    accepted = True
                    candidate = trial
                    trial_metrics = metrics
                    break
            scale *= 0.5
        if not accepted:
            rejected_steps += 1
            rejection_streak += 1
            if stop_reason == "deadline":
                break
            if rejection_streak >= max(1, int(config.maximum_rejected_steps)):
                stop_reason = "three_rejected_steps"
                break
            continue
        points = candidate
        current = dict(trial_metrics)
        accepted_scales.append(float(scale))
        rejection_streak = 0
        checkpoint_due = bool(
            iteration % max(1, int(config.checkpoint_interval)) == 0
            or iteration == int(config.iterations)
        )
        if checkpoint_due:
            previous_q = float(checkpoints[-1].metrics["q_l3_sigma"])
            checkpoint_gains.append(float(current["q_l3_sigma"]) - previous_q)
            checkpoints.append(
                InteractionCheckpoint(
                    int(iteration),
                    points.copy(),
                    dict(current),
                )
            )
            if (
                len(checkpoint_gains) >= max(1, int(config.plateau_checkpoints))
                and all(
                    value < float(config.plateau_gain)
                    for value in checkpoint_gains[-int(config.plateau_checkpoints) :]
                )
            ):
                stop_reason = "q_l3_plateau"
                break
        if int(current["superthin_triangle_count"]) >= int(config.superthin_trigger):
            if not checkpoint_due:
                checkpoints.append(
                    InteractionCheckpoint(
                        int(iteration),
                        points.copy(),
                        dict(current),
                    )
                )
            stop_reason = "superthin_trigger"
            break
    completed = int(checkpoints[-1].iteration)
    if completed == 0 and accepted_scales:
        completed = int(min(int(config.iterations), len(accepted_scales) + rejected_steps))
    final = interaction_metrics(points, triangles, config)
    return InteractionRelaxationResult(
        nodes_xy=points,
        iterations_completed=completed,
        checkpoints=checkpoints,
        report={
            "schema_version": "fvcom_interaction_relaxation_v1",
            "engine": "fixed-connectivity-edge-angle-barrier",
            "global_delaunay_rebuild": False,
            "settings": asdict(config),
            "stop_reason": stop_reason,
            "iterations_completed": int(completed),
            "accepted_step_count": int(len(accepted_scales)),
            "rejected_step_count": int(rejected_steps),
            "accepted_step_scales": accepted_scales,
            "before": initial,
            "after": final,
            "checkpoint_metrics": [
                {
                    "iteration": int(checkpoint.iteration),
                    **checkpoint.metrics,
                }
                for checkpoint in checkpoints
            ],
            "runtime_seconds": float(time.perf_counter() - started),
        },
    )


def interaction_metrics(
    points: np.ndarray,
    triangles: np.ndarray,
    config: InteractionRelaxationConfig,
    *,
    geometry: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    geometry = (
        geometry
        if geometry is not None
        else triangle_geometry(np.asarray(points, dtype=float), np.asarray(triangles, dtype=int))
    )
    quality = np.asarray(geometry["quality"], dtype=float)
    minimum_angles = (
        np.min(np.asarray(geometry["angles_deg"], dtype=float), axis=1)
        if len(quality)
        else np.empty(0, dtype=float)
    )
    q_mean = float(np.mean(quality)) if len(quality) else 0.0
    q_std = float(np.std(quality)) if len(quality) else 0.0
    q_l3 = float(q_mean - 3.0 * q_std)
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
    quality_debt = float(
        np.mean(
            np.maximum(0.0, float(config.target_quality) - quality) ** 2
        )
    ) if len(quality) else 0.0
    signed_area = np.asarray(geometry["signed_area"], dtype=float)
    area_scale = max(float(np.median(np.abs(signed_area))) if len(signed_area) else 1.0, 1.0e-30)
    area_barrier = float(
        np.mean(
            np.maximum(
                0.0,
                float(config.minimum_normalized_area)
                - signed_area / area_scale,
            )
            ** 2
        )
    ) if len(signed_area) else 0.0
    objective = float(
        -q_l3
        + float(config.angle_weight) * quality_debt
        + float(config.area_barrier_weight) * area_barrier
    )
    return {
        "q_min": float(np.min(quality)) if len(quality) else 0.0,
        "q_p01": float(np.quantile(quality, 0.01)) if len(quality) else 0.0,
        "q_mean": q_mean,
        "q_l3_sigma": q_l3,
        "minimum_angle_deg": float(np.min(minimum_angles)) if len(minimum_angles) else 0.0,
        "minimum_angle_p01_deg": (
            float(np.quantile(minimum_angles, 0.01))
            if len(minimum_angles)
            else 0.0
        ),
        "superthin_triangle_count": int(np.count_nonzero(superthin)),
        "superthin_severity_sum": severity,
        "nonpositive_signed_area_count": int(
            np.count_nonzero(signed_area <= _area_tolerance(np.asarray(points, dtype=float)))
        ),
        "objective": objective,
    }


def _interaction_displacement(
    points: np.ndarray,
    triangles: np.ndarray,
    edges: np.ndarray,
    targets: np.ndarray,
    fixed: np.ndarray,
    config: InteractionRelaxationConfig,
) -> np.ndarray:
    displacement = np.zeros_like(points)
    weight = np.zeros(len(points), dtype=float)
    if len(edges):
        delta = points[edges[:, 1]] - points[edges[:, 0]]
        length = np.linalg.norm(delta, axis=1)
        rest = 0.5 * (targets[edges[:, 0]] + targets[edges[:, 1]])
        unit = delta / np.maximum(length[:, None], 1.0e-12)
        ratio = np.clip((length - rest) / np.maximum(rest, 1.0e-12), -2.0, 2.0)
        force = (
            float(config.edge_weight)
            * ratio[:, None]
            * unit
            * rest[:, None]
        )
        np.add.at(displacement, edges[:, 0], force)
        np.add.at(displacement, edges[:, 1], -force)
        np.add.at(weight, edges[:, 0], float(config.edge_weight))
        np.add.at(weight, edges[:, 1], float(config.edge_weight))
    geometry = triangle_geometry(points, triangles)
    quality = np.asarray(geometry["quality"], dtype=float)
    minimum_angles = np.min(np.asarray(geometry["angles_deg"], dtype=float), axis=1)
    shape_debt = np.maximum(0.0, float(config.target_quality) - quality)
    shape_debt += np.maximum(
        0.0,
        float(config.superthin_min_angle_deg) - minimum_angles,
    ) / max(float(config.superthin_min_angle_deg), 1.0)
    for local_apex in range(3):
        apex = triangles[:, local_apex]
        left = triangles[:, (local_apex + 1) % 3]
        right = triangles[:, (local_apex + 2) % 3]
        base = points[right] - points[left]
        base_length = np.linalg.norm(base, axis=1)
        midpoint = 0.5 * (points[left] + points[right])
        normal = np.column_stack((-base[:, 1], base[:, 0]))
        normal /= np.maximum(base_length[:, None], 1.0e-12)
        side = np.sign(
            np.sum((points[apex] - midpoint) * normal, axis=1)
        )
        side[side == 0.0] = 1.0
        ideal = midpoint + (
            side * (np.sqrt(3.0) / 2.0) * base_length
        )[:, None] * normal
        target = (
            targets[apex] + targets[left] + targets[right]
        ) / 3.0
        correction = (
            float(config.angle_weight)
            * (0.10 + shape_debt)[:, None]
            * (ideal - points[apex])
        )
        np.add.at(displacement, apex, correction)
        np.add.at(
            weight,
            apex,
            float(config.angle_weight) * (0.10 + shape_debt),
        )
    signed_area = np.asarray(geometry["signed_area"], dtype=float)
    triangle_target = np.mean(targets[triangles], axis=1)
    normalized_area = signed_area / np.maximum(triangle_target * triangle_target, 1.0e-12)
    barrier_debt = np.maximum(
        0.0,
        float(config.minimum_normalized_area) - normalized_area,
    )
    if np.any(barrier_debt > 0.0):
        a = points[triangles[:, 0]]
        b = points[triangles[:, 1]]
        c = points[triangles[:, 2]]
        area_gradients = (
            np.column_stack((b[:, 1] - c[:, 1], c[:, 0] - b[:, 0])),
            np.column_stack((c[:, 1] - a[:, 1], a[:, 0] - c[:, 0])),
            np.column_stack((a[:, 1] - b[:, 1], b[:, 0] - a[:, 0])),
        )
        for local_node, gradient in enumerate(area_gradients):
            node = triangles[:, local_node]
            norm = np.linalg.norm(gradient, axis=1)
            correction = (
                float(config.area_barrier_weight)
                * barrier_debt[:, None]
                * gradient
                / np.maximum(norm[:, None], 1.0e-12)
                * triangle_target[:, None]
            )
            np.add.at(displacement, node, correction)
            np.add.at(
                weight,
                node,
                float(config.area_barrier_weight) * barrier_debt,
            )
    movable = ~fixed
    displacement[movable] /= np.maximum(weight[movable, None], 1.0)
    displacement[fixed] = 0.0
    maximum = (
        float(config.maximum_step_fraction)
        * np.maximum(targets, 1.0e-12)
    )
    norm = np.linalg.norm(displacement, axis=1)
    factor = np.minimum(1.0, maximum / np.maximum(norm, 1.0e-12))
    displacement *= factor[:, None]
    displacement *= float(config.damping)
    displacement[fixed] = 0.0
    return displacement


def _area_tolerance(points: np.ndarray) -> float:
    if not len(points):
        return 1.0e-30
    span = np.ptp(np.asarray(points, dtype=float), axis=0)
    scale = max(float(np.max(span)), 1.0)
    return max(1.0e-16 * scale * scale, 1.0e-30)


def _deadline_reached(config: InteractionRelaxationConfig) -> bool:
    return bool(
        config.deadline_monotonic_s is not None
        and time.perf_counter() >= float(config.deadline_monotonic_s)
    )
