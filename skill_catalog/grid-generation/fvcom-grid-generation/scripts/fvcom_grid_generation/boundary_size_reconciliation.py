"""Deterministic reconciliation of a 1-D boundary with a 2-D size field.

The reconciler is deliberately generator-neutral.  It retains every source
boundary vertex and inserts points only on the corresponding straight source
segment.  It samples the already-gradated two-dimensional field directly on
the boundary.  The backward-compatible ``minimum`` mode forms

``h_gamma(t) = min((1 - t) * h_a + t * h_b, H(x(t)))``.

The ``sampled_field`` mode instead uses ``h_gamma(t) = H(x(t))`` so the
delivered boundary follows the same two-dimensional field without retaining a
finer source-target discontinuity.

The provisional targets on every closed constraint chain are then replaced by
their cyclic one-dimensional lower gradation envelope.  Each segment is split
by equidistributing its sampled metric integral ``integral(ds / h_gamma)`` and
by a deterministic split-until-edge-compliant pass.  This is direct sampled
field reconciliation; it is not the wet-domain distance min-plus construction
``inf_x(H(x) + g*d_wet(x, Gamma(s)))``.  No mesher-specific geometry or
filesystem state is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import json
from typing import Any, Callable

import numpy as np

from .boundary import BoundaryNodes, OpenBoundaryChain, normalized_open_boundaries
from .projection import unproject_points


SizeSampler = Callable[[np.ndarray], np.ndarray]

_LINEAGE_KEYS = {
    "reconciliation_source_node_index_zero_based",
    "reconciliation_source_segment_start_zero_based",
    "reconciliation_source_segment_end_zero_based",
    "reconciliation_source_chain_index_zero_based",
    "reconciliation_source_segment_position_zero_based",
    "reconciliation_segment_interpolation_weight",
    "reconciliation_chain_normalized_arclength",
    "reconciliation_inserted",
    "reconciliation_source_target_spacing_m",
    "reconciliation_sampled_size_field_m",
}


@dataclass(frozen=True)
class BoundarySizeReconciliationConfig:
    """Numerical policy for deterministic metric-length subdivision."""

    target_metric_edge: float = 1.0
    minimum_quadrature_points: int = 33
    quadrature_target_fraction: float = 0.25
    maximum_quadrature_points: int = 4097
    compatibility_factor: float = 2.0
    maximum_spacing_gradient: float = 0.20
    maximum_boundary_l_over_h: float = 1.55
    enforce_sampled_field_compatibility: bool = False
    target_combination: str = "minimum"
    edge_tolerance: float = 1.0e-10
    sampler_id: str = "projected_size_sampler"


@dataclass(frozen=True)
class BoundarySizeReconciliationResult:
    """Reconciled boundary and its pre-meshing audit."""

    boundary: BoundaryNodes
    audit: dict[str, Any]

    @property
    def report(self) -> dict[str, Any]:
        """Backward-neutral JSON-ready name for the reconciliation audit."""

        return self.audit

    def edge_midpoint_records(self) -> list[dict[str, Any]]:
        """Return target and source lineage for every closed boundary edge."""

        metadata = self.boundary.metadata or {}
        segment = np.asarray(
            metadata["reconciliation_source_segment_position_zero_based"],
            dtype=int,
        )
        weight = np.asarray(
            metadata["reconciliation_segment_interpolation_weight"],
            dtype=float,
        )
        field = np.asarray(
            metadata["reconciliation_sampled_size_field_m"],
            dtype=float,
        )
        records: list[dict[str, Any]] = []
        for chain_index, raw_chain in enumerate(self.boundary.constraint_chains):
            chain = [int(value) for value in raw_chain]
            for edge_position, (start, end) in enumerate(
                zip(chain, chain[1:] + chain[:1])
            ):
                same_segment = bool(segment[start] == segment[end])
                end_weight = float(weight[end]) if same_segment else 1.0
                midpoint = 0.5 * (
                    self.boundary.xy[start] + self.boundary.xy[end]
                )
                target = 0.5 * float(
                    self.boundary.target_spacing_m[start]
                    + self.boundary.target_spacing_m[end]
                )
                length = float(
                    np.linalg.norm(
                        self.boundary.xy[end] - self.boundary.xy[start]
                    )
                )
                records.append(
                    {
                        "chain_index_zero_based": int(chain_index),
                        "edge_position_zero_based": int(edge_position),
                        "start_node_index_zero_based": int(start),
                        "end_node_index_zero_based": int(end),
                        "midpoint_xy": midpoint.tolist(),
                        "target_spacing_m": target,
                        "sampled_size_field_m": 0.5
                        * float(field[start] + field[end]),
                        "edge_length_m": length,
                        "edge_length_over_target": length / target,
                        "source_segment_position_zero_based": int(
                            segment[start]
                        ),
                        "source_segment_midpoint_weight": 0.5
                        * float(weight[start] + end_weight),
                    }
                )
        return records


def audit_reconciled_boundary_size_field(
    boundary: BoundaryNodes,
    size_sampler: SizeSampler,
    *,
    config: BoundarySizeReconciliationConfig | None = None,
) -> dict[str, Any]:
    """Audit a delivered boundary against its final rebuilt 2-D size field.

    Every closed-loop edge, including the last-to-first edge, is sampled at
    both endpoints and its midpoint.  Factor compatibility is always a hard
    gate here; this differs from the provisional-field diagnostic emitted by
    :func:`reconcile_boundary_size_field`.
    """

    policy = config or BoundarySizeReconciliationConfig()
    _validate_inputs(boundary, policy)
    node_field = _sample_size_field(size_sampler, np.asarray(boundary.xy))
    edge_lengths: list[float] = []
    edge_l_over_h: list[float] = []
    target_gradients: list[float] = []
    adjacent_target_ratios: list[float] = []
    midpoint_targets: list[float] = []
    midpoint_fields: list[float] = []
    chain_reports: list[dict[str, Any]] = []

    for chain_index, raw_chain in enumerate(boundary.constraint_chains):
        chain = [int(value) for value in raw_chain]
        starts = np.asarray(chain, dtype=int)
        ends = np.asarray(chain[1:] + chain[:1], dtype=int)
        midpoint_xy = 0.5 * (boundary.xy[starts] + boundary.xy[ends])
        midpoint_field = _sample_size_field(size_sampler, midpoint_xy)
        midpoint_target = 0.5 * (
            boundary.target_spacing_m[starts]
            + boundary.target_spacing_m[ends]
        )
        lengths = np.linalg.norm(
            boundary.xy[ends] - boundary.xy[starts],
            axis=1,
        )
        if np.any(lengths <= policy.edge_tolerance):
            raise ValueError(
                f"Delivered constraint chain {chain_index} has a degenerate edge"
            )
        h_gamma = np.minimum.reduce(
            (
                np.minimum(
                    boundary.target_spacing_m[starts],
                    node_field[starts],
                ),
                np.minimum(
                    boundary.target_spacing_m[ends],
                    node_field[ends],
                ),
                np.minimum(midpoint_target, midpoint_field),
            )
        )
        ratios = lengths / h_gamma
        gradients = np.abs(
            boundary.target_spacing_m[ends]
            - boundary.target_spacing_m[starts]
        ) / lengths
        adjacent = np.maximum(
            boundary.target_spacing_m[starts],
            boundary.target_spacing_m[ends],
        ) / np.minimum(
            boundary.target_spacing_m[starts],
            boundary.target_spacing_m[ends],
        )
        edge_lengths.extend(lengths.tolist())
        edge_l_over_h.extend(ratios.tolist())
        target_gradients.extend(gradients.tolist())
        adjacent_target_ratios.extend(adjacent.tolist())
        midpoint_targets.extend(midpoint_target.tolist())
        midpoint_fields.extend(midpoint_field.tolist())
        chain_reports.append(
            {
                "chain_index_zero_based": int(chain_index),
                "edge_count": int(len(chain)),
                "maximum_edge_l_over_h_gamma": float(np.max(ratios)),
                "maximum_target_gradient": float(np.max(gradients)),
            }
        )

    midpoint_target_array = np.asarray(midpoint_targets, dtype=float)
    midpoint_field_array = np.asarray(midpoint_fields, dtype=float)
    endpoint_ratio = (
        np.asarray(boundary.target_spacing_m, dtype=float) / node_field
    )
    midpoint_ratio = midpoint_target_array / midpoint_field_array
    compatibility_ratio = np.concatenate((endpoint_ratio, midpoint_ratio))
    lower = 1.0 / policy.compatibility_factor
    upper = policy.compatibility_factor
    endpoint_incompatible = (
        (endpoint_ratio < lower) | (endpoint_ratio > upper)
    )
    midpoint_incompatible = (
        (midpoint_ratio < lower) | (midpoint_ratio > upper)
    )
    incompatible = np.concatenate(
        (endpoint_incompatible, midpoint_incompatible)
    )
    maximum_lh = max(edge_l_over_h, default=0.0)
    maximum_gradient = max(target_gradients, default=0.0)
    failures: list[str] = []
    if maximum_lh > policy.maximum_boundary_l_over_h + policy.edge_tolerance:
        failures.append("boundary_edge_l_over_h")
    if maximum_gradient > (
        policy.maximum_spacing_gradient + policy.edge_tolerance
    ):
        failures.append("boundary_target_gradation")
    if np.any(incompatible):
        failures.append("boundary_field_factor_compatibility")

    digest = hashlib.sha256()
    for values in (
        np.asarray(boundary.xy, dtype="<f8"),
        np.asarray(boundary.target_spacing_m, dtype="<f8"),
        np.asarray(node_field, dtype="<f8"),
        np.asarray(midpoint_field_array, dtype="<f8"),
    ):
        digest.update(values.tobytes(order="C"))
    return {
        "schema_version": "fvcom_reconciled_boundary_final_field_audit_v1",
        "status": "pass" if not failures else "needs_review",
        "passed": bool(not failures),
        "target_size_attribution_valid": bool(not failures),
        "failure_taxonomy": failures,
        "boundary_node_count": int(len(boundary.xy)),
        "constraint_chain_count": int(len(boundary.constraint_chains)),
        "boundary_edge_count": int(len(edge_lengths)),
        "chains": chain_reports,
        "boundary_edge_length_m": _summary(edge_lengths),
        "boundary_edge_l_over_h_gamma": _summary(edge_l_over_h),
        "adjacent_target_gradient": _summary(target_gradients),
        "adjacent_target_ratio": _summary(adjacent_target_ratios),
        "boundary_to_final_field_ratio": _summary(
            compatibility_ratio.tolist()
        ),
        "factor_compatibility": {
            "factor": float(policy.compatibility_factor),
            "lower_ratio": float(lower),
            "upper_ratio": float(upper),
            "endpoint_incompatible_count": int(
                np.count_nonzero(endpoint_incompatible)
            ),
            "midpoint_incompatible_count": int(
                np.count_nonzero(midpoint_incompatible)
            ),
            "incompatible_sample_count": int(np.count_nonzero(incompatible)),
            "sample_count": int(len(compatibility_ratio)),
            "passed": bool(not np.any(incompatible)),
            "enforced_as_hard_gate": True,
        },
        "thresholds": {
            "maximum_boundary_l_over_h": float(
                policy.maximum_boundary_l_over_h
            ),
            "maximum_spacing_gradient": float(
                policy.maximum_spacing_gradient
            ),
            "compatibility_factor": float(policy.compatibility_factor),
            "target_combination": str(policy.target_combination),
        },
        "reproducibility": {
            "algorithm": "final_boundary_field_endpoint_midpoint_audit_v2",
            "target_combination": str(policy.target_combination),
            "sampler_id": str(policy.sampler_id),
            "sample_contract_sha256": digest.hexdigest(),
        },
    }


@dataclass(frozen=True)
class _SegmentPlan:
    chain_index: int
    segment_position: int
    start: int
    end: int
    parameters: np.ndarray
    targets: np.ndarray
    field_targets: np.ndarray
    metric_integral: float
    quadrature_count: int
    quadrature_parameters: np.ndarray
    quadrature_targets: np.ndarray
    quadrature_field_targets: np.ndarray


def reconcile_boundary_size_field(
    boundary: BoundaryNodes,
    size_sampler: SizeSampler,
    *,
    config: BoundarySizeReconciliationConfig | None = None,
) -> BoundarySizeReconciliationResult:
    """Reconcile boundary spacing with a projected 2-D size sampler.

    ``size_sampler`` receives an ``(N, 2)`` projected-coordinate array and
    must return one finite positive target per row.  Source constraint chains
    are treated as closed loops and must not repeat their first node.

    A call is one authoritative-source reconciliation pass.  A fixed-point
    driver must retain the original boundary, rebuild the 2-D field from this
    result, and call this function again with that original boundary and the
    rebuilt sampler.  Feeding a reconciled result back as the source is
    rejected so inserted nodes cannot silently replace original-segment
    lineage.
    """

    policy = config or BoundarySizeReconciliationConfig()
    _validate_inputs(boundary, policy)
    source_count = len(boundary.xy)
    source_targets = np.asarray(boundary.target_spacing_m, dtype=float)

    plans: list[_SegmentPlan] = []
    edge_keys: set[tuple[int, int]] = set()
    source_membership = np.full(source_count, -1, dtype=int)
    for chain_index, raw_chain in enumerate(boundary.constraint_chains):
        chain = [int(value) for value in raw_chain]
        for node in chain:
            if source_membership[node] >= 0:
                raise ValueError(
                    "A source boundary node may belong to only one constraint chain"
                )
            source_membership[node] = chain_index
        for position, start in enumerate(chain):
            end = chain[(position + 1) % len(chain)]
            if (start, end) in edge_keys or (end, start) in edge_keys:
                raise ValueError("Duplicate source constraint edge")
            edge_keys.add((start, end))
            plans.append(
                _plan_segment(
                    boundary,
                    start,
                    end,
                    chain_index,
                    position,
                    size_sampler,
                    policy,
                )
            )
    plans = _apply_closed_chain_gradation(
        boundary,
        plans,
        size_sampler,
        policy,
    )
    if np.any(source_membership < 0):
        missing = np.flatnonzero(source_membership < 0)
        raise ValueError(
            f"Every source boundary node must belong to a constraint chain; "
            f"missing {missing[:10].tolist()}"
        )

    plan_lookup = {
        (plan.chain_index, plan.segment_position): plan for plan in plans
    }
    output_xy: list[np.ndarray] = []
    output_kinds: list[str] = []
    output_targets: list[float] = []
    output_field_targets: list[float] = []
    output_hard: list[bool] = []
    output_source_node: list[int] = []
    output_segment_start: list[int] = []
    output_segment_end: list[int] = []
    output_chain_index: list[int] = []
    output_segment_position: list[int] = []
    output_weight: list[float] = []
    output_chain_fraction: list[float] = []
    output_inserted: list[bool] = []
    output_source_target: list[float] = []
    source_to_output = np.full(source_count, -1, dtype=int)
    edge_sequences: dict[tuple[int, int], list[int]] = {}
    new_chains: list[list[int]] = []
    source_hard = np.asarray(
        boundary.hard_anchor_mask
        if boundary.hard_anchor_mask is not None
        else np.zeros(source_count, dtype=bool),
        dtype=bool,
    )

    for chain_index, raw_chain in enumerate(boundary.constraint_chains):
        chain = [int(value) for value in raw_chain]
        chain_lengths = np.asarray(
            [
                np.linalg.norm(
                    np.asarray(boundary.xy[chain[(i + 1) % len(chain)]])
                    - np.asarray(boundary.xy[chain[i]])
                )
                for i in range(len(chain))
            ],
            dtype=float,
        )
        total_length = float(np.sum(chain_lengths))
        starts = np.concatenate(([0.0], np.cumsum(chain_lengths[:-1])))
        new_chain: list[int] = []
        for position, start in enumerate(chain):
            end = chain[(position + 1) % len(chain)]
            plan = plan_lookup[(chain_index, position)]
            sequence: list[int] = []
            for local_index, (parameter, target, field_target) in enumerate(
                zip(plan.parameters, plan.targets, plan.field_targets)
            ):
                # The endpoint at t=1 belongs to the following segment and is
                # therefore added only when that segment starts.
                if local_index == len(plan.parameters) - 1:
                    continue
                parameter = float(parameter)
                if parameter <= policy.edge_tolerance:
                    if source_to_output[start] < 0:
                        output_index = len(output_xy)
                        source_to_output[start] = output_index
                        output_xy.append(np.asarray(boundary.xy[start], dtype=float).copy())
                        output_kinds.append(str(boundary.kinds[start]))
                        output_targets.append(float(target))
                        output_field_targets.append(float(field_target))
                        output_hard.append(bool(source_hard[start]))
                        output_source_node.append(int(start))
                        output_inserted.append(False)
                    else:
                        output_index = int(source_to_output[start])
                else:
                    output_index = len(output_xy)
                    point = (
                        (1.0 - parameter) * np.asarray(boundary.xy[start], dtype=float)
                        + parameter * np.asarray(boundary.xy[end], dtype=float)
                    )
                    output_xy.append(point)
                    output_kinds.append(
                        _inserted_segment_kind(
                            str(boundary.kinds[start]),
                            str(boundary.kinds[end]),
                        )
                    )
                    output_targets.append(float(target))
                    output_field_targets.append(float(field_target))
                    output_hard.append(False)
                    output_source_node.append(-1)
                    output_inserted.append(True)
                output_segment_start.append(int(start))
                output_segment_end.append(int(end))
                output_chain_index.append(int(chain_index))
                output_segment_position.append(int(position))
                output_weight.append(parameter)
                output_chain_fraction.append(
                    float((starts[position] + parameter * chain_lengths[position]) / total_length)
                    if total_length > policy.edge_tolerance
                    else 0.0
                )
                output_source_target.append(
                    float(
                        (1.0 - parameter) * source_targets[start]
                        + parameter * source_targets[end]
                    )
                )
                new_chain.append(output_index)
                sequence.append(output_index)
            edge_sequences[(start, end)] = sequence
        new_chains.append(new_chain)

    xy = np.asarray(output_xy, dtype=float)
    target_spacing = np.asarray(output_targets, dtype=float)
    field_targets = np.asarray(output_field_targets, dtype=float)
    metadata = _rebuild_metadata(
        boundary,
        np.asarray(output_source_node, dtype=int),
        np.asarray(output_segment_start, dtype=int),
        np.asarray(output_segment_end, dtype=int),
        np.asarray(output_weight, dtype=float),
    )
    metadata.update(
        {
            "reconciliation_source_node_index_zero_based": np.asarray(
                output_source_node, dtype=int
            ),
            "reconciliation_source_segment_start_zero_based": np.asarray(
                output_segment_start, dtype=int
            ),
            "reconciliation_source_segment_end_zero_based": np.asarray(
                output_segment_end, dtype=int
            ),
            "reconciliation_source_chain_index_zero_based": np.asarray(
                output_chain_index, dtype=int
            ),
            "reconciliation_source_segment_position_zero_based": np.asarray(
                output_segment_position, dtype=int
            ),
            "reconciliation_segment_interpolation_weight": np.asarray(
                output_weight, dtype=float
            ),
            "reconciliation_chain_normalized_arclength": np.asarray(
                output_chain_fraction, dtype=float
            ),
            "reconciliation_inserted": np.asarray(output_inserted, dtype=bool),
            "reconciliation_source_target_spacing_m": np.asarray(
                output_source_target, dtype=float
            ),
            "reconciliation_sampled_size_field_m": field_targets.copy(),
        }
    )

    open_boundaries = _rebuild_open_boundaries(
        normalized_open_boundaries(boundary),
        source_to_output,
        edge_sequences,
    )
    open_indices = _ordered_unique(
        [int(node) for chain in open_boundaries for node in chain.node_indices]
    )
    exterior_indices = (
        list(new_chains[0]) if boundary.constraint_chains else []
    )
    reconciled = BoundaryNodes(
        xy=xy,
        lonlat=unproject_points(xy, boundary.projection),
        kinds=output_kinds,
        target_spacing_m=target_spacing,
        exterior_indices=exterior_indices,
        open_boundary_indices=open_indices,
        constraint_chains=new_chains,
        domain_polygon_xy=boundary.domain_polygon_xy,
        open_boundary_xy=boundary.open_boundary_xy,
        land_boundary_xy=boundary.land_boundary_xy,
        island_polygons_xy=list(boundary.island_polygons_xy),
        projection=boundary.projection,
        hard_anchor_mask=np.asarray(output_hard, dtype=bool),
        adaptive_resolution=bool(boundary.adaptive_resolution),
        source_resolution_manifest=boundary.source_resolution_manifest,
        resolution_profile=str(boundary.resolution_profile),
        metadata=metadata,
        passage_diagnostics=list(boundary.passage_diagnostics or []),
        open_boundaries=open_boundaries,
    )
    audit = _build_audit(
        boundary,
        reconciled,
        plans,
        field_targets,
        source_to_output,
        policy,
    )
    return BoundarySizeReconciliationResult(boundary=reconciled, audit=audit)


def _validate_inputs(
    boundary: BoundaryNodes, policy: BoundarySizeReconciliationConfig
) -> None:
    xy = np.asarray(boundary.xy, dtype=float)
    targets = np.asarray(boundary.target_spacing_m, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0:
        raise ValueError("boundary.xy must be a non-empty (N, 2) array")
    if len(boundary.lonlat) != len(xy) or len(boundary.kinds) != len(xy):
        raise ValueError("Boundary coordinate/kind lengths differ")
    if targets.shape != (len(xy),) or np.any(~np.isfinite(targets)) or np.any(targets <= 0):
        raise ValueError("Boundary targets must be finite, positive, and node-aligned")
    if not boundary.constraint_chains:
        raise ValueError("At least one closed constraint chain is required")
    for chain in boundary.constraint_chains:
        values = [int(value) for value in chain]
        if len(values) < 3 or len(values) != len(set(values)):
            raise ValueError(
                "Constraint chains must contain at least three unique nodes "
                "without repeating the first node"
            )
        if min(values) < 0 or max(values) >= len(xy):
            raise ValueError("Constraint chain node index is out of range")
    if policy.target_metric_edge <= 0.0:
        raise ValueError("target_metric_edge must be positive")
    if policy.minimum_quadrature_points < 3:
        raise ValueError("minimum_quadrature_points must be at least three")
    if policy.maximum_quadrature_points < policy.minimum_quadrature_points:
        raise ValueError("maximum_quadrature_points is smaller than the minimum")
    if policy.quadrature_target_fraction <= 0.0:
        raise ValueError("quadrature_target_fraction must be positive")
    if policy.compatibility_factor < 1.0:
        raise ValueError("compatibility_factor must be at least one")
    if policy.maximum_spacing_gradient <= 0.0:
        raise ValueError("maximum_spacing_gradient must be positive")
    if policy.maximum_boundary_l_over_h <= 0.0:
        raise ValueError("maximum_boundary_l_over_h must be positive")
    if policy.target_combination not in {"minimum", "sampled_field"}:
        raise ValueError(
            "target_combination must be 'minimum' or 'sampled_field'"
        )
    if (
        not np.isfinite(policy.maximum_spacing_gradient)
        or policy.maximum_spacing_gradient <= 0.0
    ):
        raise ValueError("maximum_spacing_gradient must be finite and positive")


def _sample_size_field(size_sampler: SizeSampler, points: np.ndarray) -> np.ndarray:
    query = np.asarray(points, dtype=float)
    sample_xy = getattr(size_sampler, "sample_xy", None)
    raw = sample_xy(query) if callable(sample_xy) else size_sampler(query)
    values = np.asarray(raw, dtype=float)
    if values.ndim == 0 and len(points) == 1:
        values = values.reshape(1)
    values = values.reshape(-1)
    if len(values) != len(points):
        raise ValueError("size_sampler must return one value per input point")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("size_sampler returned a non-finite or non-positive target")
    return values


def _plan_segment(
    boundary: BoundaryNodes,
    start: int,
    end: int,
    chain_index: int,
    position: int,
    size_sampler: SizeSampler,
    policy: BoundarySizeReconciliationConfig,
) -> _SegmentPlan:
    a = np.asarray(boundary.xy[start], dtype=float)
    b = np.asarray(boundary.xy[end], dtype=float)
    length = float(np.linalg.norm(b - a))
    if length <= policy.edge_tolerance:
        raise ValueError(f"Degenerate boundary segment {start}->{end}")
    source_a = float(boundary.target_spacing_m[start])
    source_b = float(boundary.target_spacing_m[end])
    # Use a nested sequence so the first sampled minimum H is retained while
    # it drives further quadrature refinement.
    interval_count = max(2, int(policy.minimum_quadrature_points) - 1)
    maximum_intervals = max(2, int(policy.maximum_quadrature_points) - 1)
    while True:
        q = np.linspace(0.0, 1.0, interval_count + 1, dtype=float)
        points = a[None, :] + q[:, None] * (b - a)[None, :]
        field = _sample_size_field(size_sampler, points)
        source = (1.0 - q) * source_a + q * source_b
        h_gamma = _combine_boundary_and_field_targets(
            source,
            field,
            policy.target_combination,
        )
        sampled_minimum = float(np.min(h_gamma))
        requested_intervals = max(
            interval_count,
            int(
                np.ceil(
                    length
                    / max(
                        policy.quadrature_target_fraction * sampled_minimum,
                        policy.edge_tolerance,
                    )
                )
            ),
        )
        if requested_intervals <= interval_count or interval_count >= maximum_intervals:
            break
        interval_count = min(
            maximum_intervals,
            2 * interval_count,
        )
    provisional = _SegmentPlan(
        chain_index=int(chain_index),
        segment_position=int(position),
        start=int(start),
        end=int(end),
        parameters=np.asarray([0.0, 1.0], dtype=float),
        targets=np.asarray([h_gamma[0], h_gamma[-1]], dtype=float),
        field_targets=np.asarray([field[0], field[-1]], dtype=float),
        metric_integral=0.0,
        quadrature_count=int(len(q)),
        quadrature_parameters=q,
        quadrature_targets=h_gamma,
        quadrature_field_targets=field,
    )
    return _finalize_segment_plan(
        boundary,
        provisional,
        h_gamma,
        size_sampler,
        policy,
    )


def _combine_boundary_and_field_targets(
    source: np.ndarray,
    field: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Return the declared effective one-dimensional boundary target."""

    if mode == "minimum":
        return np.minimum(source, field)
    if mode == "sampled_field":
        return np.asarray(field, dtype=float).copy()
    raise ValueError(f"unsupported target-combination mode {mode!r}")


def _apply_closed_chain_gradation(
    boundary: BoundaryNodes,
    plans: list[_SegmentPlan],
    size_sampler: SizeSampler,
    policy: BoundarySizeReconciliationConfig,
) -> list[_SegmentPlan]:
    """Apply one cyclic lower Lipschitz envelope to each constraint chain."""

    lookup = {
        (plan.chain_index, plan.segment_position): plan for plan in plans
    }
    output: list[_SegmentPlan] = []
    for chain_index, raw_chain in enumerate(boundary.constraint_chains):
        chain = [int(value) for value in raw_chain]
        chain_plans = [
            lookup[(chain_index, position)] for position in range(len(chain))
        ]
        lengths = np.asarray(
            [
                np.linalg.norm(
                    boundary.xy[chain[(position + 1) % len(chain)]]
                    - boundary.xy[chain[position]]
                )
                for position in range(len(chain))
            ],
            dtype=float,
        )
        starts = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
        total_length = float(np.sum(lengths))
        sample_positions: list[float] = []
        raw_targets: list[float] = []
        slices: list[slice] = []
        for position, plan in enumerate(chain_plans):
            begin = len(sample_positions)
            q = np.asarray(plan.quadrature_parameters, dtype=float)
            h = np.asarray(plan.quadrature_targets, dtype=float)
            sample_positions.extend(
                (
                    starts[position] + lengths[position] * q[:-1]
                ).tolist()
            )
            raw_targets.extend(h[:-1].tolist())
            slices.append(slice(begin, len(sample_positions)))
        envelope = _cyclic_lower_gradation_envelope(
            np.asarray(sample_positions, dtype=float),
            np.asarray(raw_targets, dtype=float),
            total_length,
            policy.maximum_spacing_gradient,
        )
        for position, plan in enumerate(chain_plans):
            current = np.asarray(envelope[slices[position]], dtype=float)
            following = np.asarray(
                envelope[slices[(position + 1) % len(chain_plans)]],
                dtype=float,
            )
            segment_envelope = np.concatenate((current, [following[0]]))
            output.append(
                _finalize_segment_plan(
                    boundary,
                    plan,
                    segment_envelope,
                    size_sampler,
                    policy,
                )
            )
    return output


def _cyclic_lower_gradation_envelope(
    positions: np.ndarray,
    raw_targets: np.ndarray,
    total_length: float,
    maximum_gradient: float,
) -> np.ndarray:
    """Compute ``min_j(raw[j] + g*d_cycle(i, j))`` by graph Dijkstra."""

    count = len(raw_targets)
    if count < 3:
        raise ValueError("A closed gradation chain requires at least three samples")
    result = np.asarray(raw_targets, dtype=float).copy()
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    for index in range(count - 1):
        distance = float(positions[index + 1] - positions[index])
        weight = maximum_gradient * distance
        adjacency[index].append((index + 1, weight))
        adjacency[index + 1].append((index, weight))
    closure_distance = float(total_length - positions[-1] + positions[0])
    closure_weight = maximum_gradient * closure_distance
    adjacency[-1].append((0, closure_weight))
    adjacency[0].append((count - 1, closure_weight))
    queue = [(float(value), int(index)) for index, value in enumerate(result)]
    heapq.heapify(queue)
    while queue:
        value, index = heapq.heappop(queue)
        if value > result[index] + 1.0e-12:
            continue
        for neighbour, weight in adjacency[index]:
            candidate = value + weight
            if candidate + 1.0e-12 < result[neighbour]:
                result[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return result


def _finalize_segment_plan(
    boundary: BoundaryNodes,
    plan: _SegmentPlan,
    quadrature_targets: np.ndarray,
    size_sampler: SizeSampler,
    policy: BoundarySizeReconciliationConfig,
) -> _SegmentPlan:
    """Equidistribute the metric, then split until every edge is compliant."""

    q = np.asarray(plan.quadrature_parameters, dtype=float)
    h = np.asarray(quadrature_targets, dtype=float)
    a = np.asarray(boundary.xy[plan.start], dtype=float)
    b = np.asarray(boundary.xy[plan.end], dtype=float)
    length = float(np.linalg.norm(b - a))
    increments = 0.5 * (1.0 / h[:-1] + 1.0 / h[1:])
    increments *= length * np.diff(q)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    metric_integral = float(cumulative[-1])
    segment_count = max(
        1,
        int(np.ceil(metric_integral / policy.target_metric_edge)),
    )
    parameters = np.interp(
        np.linspace(0.0, metric_integral, segment_count + 1),
        cumulative,
        q,
    )
    parameters[0] = 0.0
    parameters[-1] = 1.0
    for _ in range(64):
        targets = np.interp(parameters, q, h)
        midpoints = 0.5 * (parameters[:-1] + parameters[1:])
        midpoint_targets = np.interp(midpoints, q, h)
        ratios = (
            length
            * np.diff(parameters)
            / np.minimum(
                np.minimum(targets[:-1], targets[1:]),
                midpoint_targets,
            )
        )
        violating = np.flatnonzero(
            ratios > policy.target_metric_edge + policy.edge_tolerance
        )
        if not len(violating):
            break
        parameters = np.asarray(
            sorted(
                set(
                    parameters.tolist()
                    + [float(midpoints[index]) for index in violating]
                )
            ),
            dtype=float,
        )
    else:
        raise RuntimeError(
            f"Boundary segment {plan.start}->{plan.end} did not reach the "
            "edge/target constraint"
        )
    delivered_points = a[None, :] + parameters[:, None] * (b - a)[None, :]
    delivered_field = _sample_size_field(size_sampler, delivered_points)
    delivered_target = np.interp(parameters, q, h)
    return _SegmentPlan(
        chain_index=plan.chain_index,
        segment_position=plan.segment_position,
        start=plan.start,
        end=plan.end,
        parameters=parameters,
        targets=delivered_target,
        field_targets=delivered_field,
        metric_integral=metric_integral,
        quadrature_count=int(len(q)),
        quadrature_parameters=q,
        quadrature_targets=h,
        quadrature_field_targets=np.asarray(
            plan.quadrature_field_targets,
            dtype=float,
        ),
    )


def _inserted_segment_kind(start_kind: str, end_kind: str) -> str:
    """Keep mixed open/solid segments solid; originals retain exact kinds."""

    open_kinds = {"open", "open_boundary"}
    start_open = start_kind.lower() in open_kinds
    end_open = end_kind.lower() in open_kinds
    if start_open and end_open:
        return start_kind
    if start_kind == end_kind:
        return start_kind
    return "land"


def _rebuild_metadata(
    boundary: BoundaryNodes,
    source_node: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    weight: np.ndarray,
) -> dict[str, np.ndarray]:
    source_metadata = boundary.metadata or {}
    collisions = sorted(_LINEAGE_KEYS.intersection(source_metadata))
    if collisions:
        raise ValueError(
            "Boundary metadata already contains reconciliation-reserved keys: "
            + ", ".join(collisions)
        )
    output: dict[str, np.ndarray] = {}
    for name, raw_values in source_metadata.items():
        values = np.asarray(raw_values)
        if len(values) != len(boundary.xy):
            raise ValueError(f"Boundary metadata {name!r} is not node-aligned")
        rebuilt: list[Any] = []
        for exact, start, end, t in zip(
            source_node, segment_start, segment_end, weight
        ):
            if exact >= 0:
                rebuilt.append(values[int(exact)])
                continue
            # Source metadata can contain anchor flags, categorical identities,
            # integer node IDs, and other source-only semantics.  There is no
            # safe generic interpolation rule.  Inserted-node lineage lives in
            # the explicit reconciliation_* fields below, so use JSON ``null``
            # here instead of fabricating or duplicating a source identity.
            rebuilt.append(None)
        output[str(name)] = np.asarray(rebuilt, dtype=object)
    return output


def _rebuild_open_boundaries(
    source_chains: list[OpenBoundaryChain],
    source_to_output: np.ndarray,
    edge_sequences: dict[tuple[int, int], list[int]],
) -> list[OpenBoundaryChain]:
    rebuilt: list[OpenBoundaryChain] = []
    for chain in source_chains:
        source = [int(value) for value in chain.node_indices]
        if not source:
            raise ValueError(f"Open boundary {chain.chain_id!r} is empty")
        delivered: list[int] = []
        pair_count = len(source) if chain.cyclic else max(0, len(source) - 1)
        for position in range(pair_count):
            a = source[position]
            b = source[(position + 1) % len(source)]
            if (a, b) in edge_sequences:
                edge = list(edge_sequences[(a, b)])
            elif (b, a) in edge_sequences:
                # Reversing [b, inserts...] gives [inserts..., b]; prepend the
                # explicit a node and omit the repeated b endpoint.
                reverse_full = list(reversed(edge_sequences[(b, a)]))
                edge = [int(source_to_output[a]), *reverse_full[:-1]]
            else:
                raise ValueError(
                    f"Open boundary {chain.chain_id!r} contains non-adjacent "
                    f"constraint nodes {a}->{b}"
                )
            delivered.extend(edge)
        if not chain.cyclic:
            delivered.append(int(source_to_output[source[-1]]))
        delivered = _ordered_unique(delivered)
        rebuilt.append(
            OpenBoundaryChain(
                chain_id=str(chain.chain_id),
                node_indices=tuple(delivered),
                kind=str(chain.kind),
                cyclic=bool(chain.cyclic),
                orientation=str(chain.orientation),
            )
        )
    return rebuilt


def _build_audit(
    source: BoundaryNodes,
    delivered: BoundaryNodes,
    plans: list[_SegmentPlan],
    field_targets: np.ndarray,
    source_to_output: np.ndarray,
    policy: BoundarySizeReconciliationConfig,
) -> dict[str, Any]:
    edge_l_over_h: list[float] = []
    gradients: list[float] = []
    adjacent_ratios: list[float] = []
    for plan in plans:
        segment_length = float(
            np.linalg.norm(source.xy[plan.end] - source.xy[plan.start])
        )
        for index in range(len(plan.parameters) - 1):
            ta = float(plan.parameters[index])
            tb = float(plan.parameters[index + 1])
            ha = float(plan.targets[index])
            hb = float(plan.targets[index + 1])
            midpoint = 0.5 * (ta + tb)
            hm = float(
                np.interp(
                    midpoint,
                    plan.quadrature_parameters,
                    plan.quadrature_targets,
                )
            )
            length = segment_length * (tb - ta)
            h_min = min(ha, hm, hb)
            edge_l_over_h.append(length / h_min)
            gradients.append(
                abs(hb - ha) / length
            )
            adjacent_ratios.append(
                max(ha, hb) / min(ha, hb)
            )
    compatibility_ratio = np.asarray(delivered.target_spacing_m) / field_targets
    lower = 1.0 / policy.compatibility_factor
    upper = policy.compatibility_factor
    incompatible = (compatibility_ratio < lower) | (compatibility_ratio > upper)
    all_source_vertices_exact = bool(
        np.all(
            np.linalg.norm(
                delivered.xy[source_to_output] - np.asarray(source.xy, dtype=float),
                axis=1,
            )
            <= policy.edge_tolerance
        )
    )
    hard = np.asarray(
        source.hard_anchor_mask
        if source.hard_anchor_mask is not None
        else np.zeros(len(source.xy), dtype=bool),
        dtype=bool,
    )
    delivered_hard = np.asarray(delivered.hard_anchor_mask, dtype=bool)
    hard_anchors_exact = bool(
        np.all(delivered_hard[source_to_output[hard]]) and all_source_vertices_exact
    )
    source_digest = hashlib.sha256()
    for values in (
        np.asarray(source.xy, dtype="<f8"),
        np.asarray(source.target_spacing_m, dtype="<f8"),
        hard.astype(np.uint8),
    ):
        source_digest.update(values.tobytes(order="C"))
    source_digest.update(
        json.dumps(
            {
                "kinds": list(map(str, source.kinds)),
                "constraint_chains": [
                    list(map(int, chain)) for chain in source.constraint_chains
                ],
                "open_boundaries": [
                    {
                        "chain_id": chain.chain_id,
                        "node_indices": list(chain.node_indices),
                        "kind": chain.kind,
                        "cyclic": chain.cyclic,
                        "orientation": chain.orientation,
                    }
                    for chain in normalized_open_boundaries(source)
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    reconciled_sampler_digest = hashlib.sha256()
    field_sampler_digest = hashlib.sha256()
    for plan in plans:
        parameter_bytes = np.asarray(
            plan.quadrature_parameters, dtype="<f8"
        ).tobytes(order="C")
        reconciled_sampler_digest.update(parameter_bytes)
        reconciled_sampler_digest.update(
            np.asarray(plan.quadrature_targets, dtype="<f8").tobytes(order="C")
        )
        field_sampler_digest.update(parameter_bytes)
        field_sampler_digest.update(
            np.asarray(plan.quadrature_field_targets, dtype="<f8").tobytes(
                order="C"
            )
        )
    policy_digest = hashlib.sha256(
        json.dumps(
            {
                "target_metric_edge": policy.target_metric_edge,
                "minimum_quadrature_points": policy.minimum_quadrature_points,
                "quadrature_target_fraction": policy.quadrature_target_fraction,
                "maximum_quadrature_points": policy.maximum_quadrature_points,
                "compatibility_factor": policy.compatibility_factor,
                "maximum_spacing_gradient": policy.maximum_spacing_gradient,
                "maximum_boundary_l_over_h": (
                    policy.maximum_boundary_l_over_h
                ),
                "enforce_sampled_field_compatibility": (
                    policy.enforce_sampled_field_compatibility
                ),
                "target_combination": policy.target_combination,
                "edge_tolerance": policy.edge_tolerance,
                "sampler_id": policy.sampler_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    maximum_lh = max(edge_l_over_h, default=0.0)
    maximum_gradient = max(gradients, default=0.0)
    failures: list[str] = []
    if not all_source_vertices_exact:
        failures.append("source_boundary_vertex_shift")
    if not hard_anchors_exact:
        failures.append("hard_anchor_loss")
    if policy.enforce_sampled_field_compatibility and np.any(incompatible):
        failures.append("boundary_field_factor_compatibility")
    if (
        maximum_gradient
        > policy.maximum_spacing_gradient + policy.edge_tolerance
    ):
        failures.append("adjacent_target_gradation")
    if maximum_lh > policy.target_metric_edge + policy.edge_tolerance:
        failures.append("boundary_edge_metric_length")
    return {
        "schema_version": "fvcom_boundary_size_reconciliation_v1",
        "status": "pass" if not failures else "needs_review",
        "passed": bool(not failures),
        "failure_taxonomy": failures,
        "source_boundary_node_count": int(len(source.xy)),
        "delivered_boundary_node_count": int(len(delivered.xy)),
        "inserted_boundary_node_count": int(len(delivered.xy) - len(source.xy)),
        "source_constraint_chain_count": int(len(source.constraint_chains)),
        "open_boundary_chain_count": int(len(delivered.open_boundaries or [])),
        "all_source_vertices_exact": all_source_vertices_exact,
        "hard_anchors_exact": hard_anchors_exact,
        "boundary_edge_l_over_h_gamma": _summary(edge_l_over_h),
        "adjacent_target_gradient": _summary(gradients),
        "adjacent_target_ratio": _summary(adjacent_ratios),
        "boundary_to_sampled_field_ratio": _summary(
            compatibility_ratio.tolist()
        ),
        "factor_compatibility": {
            "factor": float(policy.compatibility_factor),
            "lower_ratio": float(lower),
            "upper_ratio": float(upper),
            "incompatible_node_count": int(np.count_nonzero(incompatible)),
            "passed": bool(not np.any(incompatible)),
            "enforced_as_hard_gate": bool(
                policy.enforce_sampled_field_compatibility
            ),
            "role": (
                "diagnostic_against_provisional_sampled_field"
                if not policy.enforce_sampled_field_compatibility
                else "hard_check_against_declared_final_field"
            ),
        },
        "segment_metric_integral": _summary(
            [plan.metric_integral for plan in plans]
        ),
        "quadrature": {
            "minimum_points": int(policy.minimum_quadrature_points),
            "maximum_points": int(policy.maximum_quadrature_points),
            "used_points": _summary(
                [float(plan.quadrature_count) for plan in plans]
            ),
            "target_fraction": float(policy.quadrature_target_fraction),
        },
        "thresholds": {
            "target_metric_edge": float(policy.target_metric_edge),
            "maximum_spacing_gradient": float(policy.maximum_spacing_gradient),
            "comparison_tolerance": float(policy.edge_tolerance),
            "maximum_boundary_l_over_h": float(
                policy.maximum_boundary_l_over_h
            ),
            "compatibility_factor": float(policy.compatibility_factor),
            "enforce_sampled_field_compatibility": bool(
                policy.enforce_sampled_field_compatibility
            ),
            "target_combination": str(policy.target_combination),
        },
        "reproducibility": {
            "algorithm": (
                "boundary_target_combination_cyclic_lower_gradation_"
                "metric_equidistribution_v2"
            ),
            "method_scope": (
                (
                    "boundary_target_follows_sampled_gradated_2d_field"
                    if policy.target_combination == "sampled_field"
                    else "boundary_target_is_minimum_of_source_and_"
                    "sampled_gradated_2d_field"
                )
                + "; not_wet_distance_min_plus"
            ),
            "sampler_id": str(policy.sampler_id),
            "source_contract_sha256": source_digest.hexdigest(),
            "policy_sha256": policy_digest,
            "sampled_size_field_sha256": field_sampler_digest.hexdigest(),
            "sampled_reconciled_target_sha256": (
                reconciled_sampler_digest.hexdigest()
            ),
        },
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {
            "count": 0,
            "minimum": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "maximum": 0.0,
        }
    return {
        "count": int(len(array)),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(np.max(array)),
    }


def _ordered_unique(values: list[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        if value not in seen:
            output.append(int(value))
            seen.add(int(value))
    return output
