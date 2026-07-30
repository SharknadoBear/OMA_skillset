"""Read-only, edge-aware target-size transition diagnostics.

The legacy FVCOM target-size diagnostic assigns one target to each triangle.
That is useful as a whole-mesh summary, but it can hide a discontinuity between
the delivered one-dimensional boundary discretization and the adjacent
two-dimensional size field.  This module instead evaluates every unique mesh
edge.  Interior edges use the conservative minimum of the two endpoint and
midpoint field samples.  Constraint edges use the corresponding boundary
target, when supplied, and otherwise fall back to the two-dimensional field.

All node indices accepted by this module are zero based.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA = "fvcom_edge_size_audit_v1"
DEFAULT_THRESHOLDS = (1.55, 2.0)
STRATA = ("boundary", "first_ring", "transition", "true_interior")


SizeSampler = Callable[[np.ndarray], np.ndarray | Sequence[float] | float]


def audit_edge_target_sizes(
    points_xy: np.ndarray,
    triangles: np.ndarray,
    constraint_chains: Sequence[Sequence[int] | Mapping[str, Any]],
    size_sampler: SizeSampler,
    *,
    boundary_target_by_node: np.ndarray | Mapping[int, float] | None = None,
    boundary_target_sampler: SizeSampler | None = None,
    transition_graph_rings: int = 2,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    boundary_gradation_limit: float = 0.20,
    boundary_field_ratio_limit: float = 2.0,
    boundary_first_ring_p95_limit: float = 1.5,
    boundary_first_ring_maximum_limit: float = 2.0,
) -> dict[str, Any]:
    """Audit unique-edge ``L/h`` and boundary-to-interior transition strata.

    Triangle graph distance is measured from triangles containing an explicit
    constraint-chain edge.  Distance-zero triangles form the ``boundary``
    stratum; distance one is ``first_ring``; the requested number of subsequent
    graph rings form ``transition``; all remaining triangles are
    ``true_interior``.  Explicit constraint edges are ``boundary`` edges.
    Other edges touching graph distances zero or one are ``first_ring`` edges.

    Parameters
    ----------
    points_xy, triangles
        Mesh coordinates and zero-based first-order triangle connectivity.
    constraint_chains
        Ordered zero-based chain nodes.  A mapping form accepts ``node_ids`` or
        ``nodes`` and optional ``cyclic`` and ``chain_id`` fields.
    size_sampler
        Callable receiving an ``(n, 2)`` array and returning target sizes.
    boundary_target_by_node, boundary_target_sampler
        Optional one-dimensional boundary targets.  Node data can be an array
        of mesh-node length (non-boundary entries may be NaN) or an index-value
        mapping.  When both forms are supplied their conservative minimum is
        used.  Boundary edges fall back to ``size_sampler`` if neither yields a
        valid positive target.
    transition_graph_rings
        Number of triangle graph rings after ``first_ring`` assigned to the
        transition stratum.
    thresholds
        Strict ``L/h`` exceedance thresholds reported for every stratum. The
        smallest value is the hard all-triangle p95 limit and the largest is
        the hard all-triangle maximum limit.
    boundary_gradation_limit
        Maximum advisory adjacent-edge target and realized-size slope,
        ``abs(delta h) / distance_between_edge_midpoints``.
    boundary_field_ratio_limit
        Hard symmetric-ratio limit between the conservative explicit boundary
        target and conservative two-dimensional field target sampled on the
        same constraint edge. The generic default is ``2.0``; the raw mesher
        portfolio currently overrides it with its stricter common policy.
    boundary_first_ring_p95_limit, boundary_first_ring_maximum_limit
        Hard limits on the realized symmetric ratio between each constraint
        edge length and the equilateral-side characteristic length of every
        incident boundary-row triangle.
    """

    points = np.asarray(points_xy, dtype=float)
    tris = np.asarray(triangles, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (n, 2)")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("triangles must have shape (m, 3)")
    if len(points) == 0 or len(tris) == 0:
        raise ValueError("points_xy and triangles must be non-empty")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_xy must be finite")
    if np.any(tris < 0) or np.any(tris >= len(points)):
        raise ValueError("triangles contain an out-of-range node index")
    if np.any(
        (tris[:, 0] == tris[:, 1])
        | (tris[:, 1] == tris[:, 2])
        | (tris[:, 2] == tris[:, 0])
    ):
        raise ValueError("triangles contain repeated node indices")
    if int(transition_graph_rings) != transition_graph_rings or transition_graph_rings < 0:
        raise ValueError("transition_graph_rings must be a non-negative integer")
    threshold_values = _validate_thresholds(thresholds)
    if not np.isfinite(boundary_gradation_limit) or boundary_gradation_limit < 0.0:
        raise ValueError("boundary_gradation_limit must be finite and non-negative")
    if (
        not np.isfinite(boundary_field_ratio_limit)
        or boundary_field_ratio_limit <= 1.0
    ):
        raise ValueError(
            "boundary_field_ratio_limit must be finite and greater than one"
        )
    if (
        not np.isfinite(boundary_first_ring_p95_limit)
        or boundary_first_ring_p95_limit < 1.0
    ):
        raise ValueError(
            "boundary_first_ring_p95_limit must be finite and at least one"
        )
    if (
        not np.isfinite(boundary_first_ring_maximum_limit)
        or boundary_first_ring_maximum_limit < 1.0
    ):
        raise ValueError(
            "boundary_first_ring_maximum_limit must be finite and at least one"
        )
    if boundary_first_ring_p95_limit > boundary_first_ring_maximum_limit:
        raise ValueError(
            "boundary_first_ring_p95_limit cannot exceed "
            "boundary_first_ring_maximum_limit"
        )

    chains = _normalize_chains(constraint_chains, len(points))
    constraint_edges = {
        edge
        for chain in chains
        for edge in chain["edges"]
    }
    if not constraint_edges:
        raise ValueError("at least one constraint-chain edge is required")

    edges, edge_triangles, triangle_edges = _mesh_edges(tris)
    edge_lookup = {edge: index for index, edge in enumerate(edges)}
    missing_edges = sorted(edge for edge in constraint_edges if edge not in edge_lookup)
    if missing_edges:
        preview = ", ".join(str(edge) for edge in missing_edges[:5])
        raise ValueError(f"constraint-chain edges are absent from the mesh: {preview}")

    triangle_distance = _triangle_graph_distance(
        len(tris),
        edge_triangles,
        triangle_edges,
        {edge_lookup[edge] for edge in constraint_edges},
    )
    triangle_strata = np.asarray(
        [
            _triangle_stratum(distance, int(transition_graph_rings))
            for distance in triangle_distance
        ],
        dtype=object,
    )

    edge_nodes = np.asarray(edges, dtype=np.int64)
    endpoint_a = points[edge_nodes[:, 0]]
    endpoint_b = points[edge_nodes[:, 1]]
    midpoints = 0.5 * (endpoint_a + endpoint_b)
    lengths = np.linalg.norm(endpoint_b - endpoint_a, axis=1)
    field_samples = _sample_triplets(size_sampler, endpoint_a, endpoint_b, midpoints)
    field_targets = _positive_row_min(field_samples)

    boundary_node_targets = _boundary_node_target_array(
        boundary_target_by_node,
        len(points),
    )
    boundary_samples = np.full((len(edges), 3), np.nan, dtype=float)
    if boundary_target_sampler is not None:
        boundary_samples = _sample_triplets(
            boundary_target_sampler,
            endpoint_a,
            endpoint_b,
            midpoints,
        )
    if boundary_node_targets is not None:
        boundary_samples[:, 0] = _finite_positive_min(
            boundary_samples[:, 0],
            boundary_node_targets[edge_nodes[:, 0]],
        )
        boundary_samples[:, 1] = _finite_positive_min(
            boundary_samples[:, 1],
            boundary_node_targets[edge_nodes[:, 1]],
        )
        interpolated = 0.5 * (
            boundary_node_targets[edge_nodes[:, 0]]
            + boundary_node_targets[edge_nodes[:, 1]]
        )
        boundary_samples[:, 2] = _finite_positive_min(
            boundary_samples[:, 2],
            interpolated,
        )
    boundary_targets = _positive_row_min(boundary_samples)
    boundary_samples_complete = np.all(
        np.isfinite(boundary_samples) & (boundary_samples > 0.0),
        axis=1,
    )
    field_samples_complete = np.all(
        np.isfinite(field_samples) & (field_samples > 0.0),
        axis=1,
    )

    is_constraint = np.asarray(
        [edge in constraint_edges for edge in edges],
        dtype=bool,
    )
    selected_targets = field_targets.copy()
    boundary_has_own_target = is_constraint & np.isfinite(boundary_targets)
    selected_targets[boundary_has_own_target] = boundary_targets[
        boundary_has_own_target
    ]
    valid_ratio = (
        np.isfinite(lengths)
        & (lengths > 0.0)
        & np.isfinite(selected_targets)
        & (selected_targets > 0.0)
    )
    ratios = np.full(len(edges), np.nan, dtype=float)
    ratios[valid_ratio] = lengths[valid_ratio] / selected_targets[valid_ratio]

    edge_strata = np.asarray(
        [
            _edge_stratum(
                index,
                is_constraint[index],
                edge_triangles,
                triangle_distance,
                int(transition_graph_rings),
            )
            for index in range(len(edges))
        ],
        dtype=object,
    )

    triangle_ratios = np.full(len(tris), np.nan, dtype=float)
    for triangle_index, incident_edges in enumerate(triangle_edges):
        values = ratios[np.asarray(incident_edges, dtype=np.int64)]
        finite = values[np.isfinite(values)]
        if len(finite):
            triangle_ratios[triangle_index] = float(np.max(finite))

    edge_reports = {
        stratum: _ratio_summary(ratios[edge_strata == stratum], threshold_values)
        for stratum in STRATA
    }
    edge_reports["all"] = _ratio_summary(ratios, threshold_values)
    triangle_reports = {
        stratum: _ratio_summary(
            triangle_ratios[triangle_strata == stratum],
            threshold_values,
        )
        for stratum in STRATA
    }
    triangle_reports["all"] = _ratio_summary(triangle_ratios, threshold_values)
    target_size_gate = _target_size_hard_gate(
        triangle_reports["all"],
        edge_summary=edge_reports["all"],
        p95_limit=float(threshold_values[0]),
        maximum_limit=float(threshold_values[-1]),
    )
    boundary_field_interface = _boundary_field_interface_diagnostic(
        is_constraint,
        boundary_samples,
        field_samples,
        float(boundary_field_ratio_limit),
    )
    boundary_first_ring = _boundary_first_ring_continuity(
        chains,
        edge_lookup,
        edge_triangles,
        lengths,
        points,
        tris,
        p95_limit=float(boundary_first_ring_p95_limit),
        maximum_limit=float(boundary_first_ring_maximum_limit),
    )
    failures = sorted(
        set(
            list(boundary_field_interface["failure_taxonomy"])
            + list(boundary_first_ring["failure_taxonomy"])
            + list(target_size_gate["failure_taxonomy"])
        )
    )

    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "passed": bool(
            boundary_field_interface["passed"]
            and boundary_first_ring["passed"]
            and target_size_gate["passed"]
        ),
        "failure_taxonomy": failures,
        "method": {
            "edge_target_sampling": "minimum_of_endpoints_and_midpoint",
            "triangle_l_over_h": "maximum_of_three_incident_edge_ratios",
            "boundary_target_precedence": (
                "boundary_target_then_2d_field_fallback"
            ),
            "triangle_graph_distance_seed": (
                "triangles_incident_to_constraint_chain_edges"
            ),
            "transition_graph_rings": int(transition_graph_rings),
            "thresholds": threshold_values,
            "strict_threshold_comparison": True,
            "boundary_gradation_limit": float(boundary_gradation_limit),
            "boundary_field_ratio_limit": float(boundary_field_ratio_limit),
            "boundary_first_ring_characteristic_length": (
                "equilateral_side_with_incident_triangle_area_"
                "sqrt_4A_over_sqrt3"
            ),
            "boundary_first_ring_p95_limit": float(
                boundary_first_ring_p95_limit
            ),
            "boundary_first_ring_maximum_limit": float(
                boundary_first_ring_maximum_limit
            ),
        },
        "counts": {
            "nodes": int(len(points)),
            "triangles": int(len(tris)),
            "unique_edges": int(len(edges)),
            "constraint_chains": int(len(chains)),
            "constraint_edges": int(np.count_nonzero(is_constraint)),
            "boundary_edges_with_own_target": int(
                np.count_nonzero(boundary_has_own_target)
            ),
            "edges_with_invalid_target_or_length": int(
                np.count_nonzero(~valid_ratio)
            ),
        },
        "edge_l_over_h": edge_reports,
        "triangle_l_over_h": triangle_reports,
        "target_size_hard_gate": target_size_gate,
        "boundary_field_interface": boundary_field_interface,
        "boundary_first_ring_realized_continuity": boundary_first_ring,
        "stratum_counts": {
            "edges": {
                stratum: int(np.count_nonzero(edge_strata == stratum))
                for stratum in STRATA
            },
            "triangles": {
                stratum: int(np.count_nonzero(triangle_strata == stratum))
                for stratum in STRATA
            },
        },
        "boundary_diagnostics": _boundary_diagnostics(
            chains,
            edge_lookup,
            lengths,
            selected_targets,
            ratios,
            points,
            float(boundary_gradation_limit),
            threshold_values,
        ),
    }


def _validate_thresholds(thresholds: Sequence[float]) -> list[float]:
    values = sorted({float(value) for value in thresholds})
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("thresholds must contain finite positive values")
    return values


def _normalize_chains(
    raw_chains: Sequence[Sequence[int] | Mapping[str, Any]],
    node_count: int,
) -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    for chain_index, raw in enumerate(raw_chains):
        if isinstance(raw, Mapping):
            raw_nodes = raw.get("node_ids", raw.get("nodes"))
            if raw_nodes is None:
                raise ValueError(
                    f"constraint chain {chain_index} lacks node_ids or nodes"
                )
            chain_id = str(raw.get("chain_id", chain_index))
            cyclic = bool(raw.get("cyclic", False))
        else:
            raw_nodes = raw
            chain_id = str(chain_index)
            cyclic = False
        nodes = [int(value) for value in raw_nodes]
        if len(nodes) >= 2 and nodes[0] == nodes[-1]:
            nodes.pop()
            cyclic = True
        if len(nodes) < 2:
            raise ValueError(f"constraint chain {chain_id} has fewer than two nodes")
        if any(node < 0 or node >= node_count for node in nodes):
            raise ValueError(f"constraint chain {chain_id} has an invalid node index")
        pairs = list(zip(nodes[:-1], nodes[1:]))
        if cyclic:
            pairs.append((nodes[-1], nodes[0]))
        if any(a == b for a, b in pairs):
            raise ValueError(f"constraint chain {chain_id} has a zero-length edge")
        chains.append(
            {
                "chain_id": chain_id,
                "nodes": nodes,
                "cyclic": cyclic,
                "edges": [_canonical_edge(a, b) for a, b in pairs],
            }
        )
    return chains


def _mesh_edges(
    triangles: np.ndarray,
) -> tuple[
    list[tuple[int, int]],
    list[list[int]],
    list[list[int]],
]:
    edge_to_triangles: dict[tuple[int, int], list[int]] = {}
    triangle_edge_keys: list[list[tuple[int, int]]] = []
    for triangle_index, (a, b, c) in enumerate(triangles.tolist()):
        keys = [
            _canonical_edge(a, b),
            _canonical_edge(b, c),
            _canonical_edge(c, a),
        ]
        triangle_edge_keys.append(keys)
        for key in keys:
            edge_to_triangles.setdefault(key, []).append(triangle_index)
    nonmanifold = [
        edge for edge, owners in edge_to_triangles.items() if len(owners) > 2
    ]
    if nonmanifold:
        raise ValueError(
            f"mesh has {len(nonmanifold)} non-manifold edges with >2 triangles"
        )
    edges = sorted(edge_to_triangles)
    lookup = {edge: index for index, edge in enumerate(edges)}
    edge_triangles = [edge_to_triangles[edge] for edge in edges]
    triangle_edges = [
        [lookup[key] for key in keys]
        for keys in triangle_edge_keys
    ]
    return edges, edge_triangles, triangle_edges


def _triangle_graph_distance(
    triangle_count: int,
    edge_triangles: Sequence[Sequence[int]],
    triangle_edges: Sequence[Sequence[int]],
    constraint_edge_indices: set[int],
) -> np.ndarray:
    adjacency = [set() for _ in range(triangle_count)]
    for owners in edge_triangles:
        if len(owners) == 2:
            left, right = owners
            adjacency[left].add(right)
            adjacency[right].add(left)
    seeds = sorted(
        {
            triangle
            for edge_index in constraint_edge_indices
            for triangle in edge_triangles[edge_index]
        }
    )
    distance = np.full(triangle_count, -1, dtype=np.int64)
    queue: deque[int] = deque()
    for seed in seeds:
        distance[seed] = 0
        queue.append(seed)
    while queue:
        triangle = queue.popleft()
        for neighbor in sorted(adjacency[triangle]):
            if distance[neighbor] < 0:
                distance[neighbor] = distance[triangle] + 1
                queue.append(neighbor)
    # Disconnected components are true interior relative to the supplied
    # constraints.  Use a large finite sentinel to keep JSON-free internals.
    distance[distance < 0] = np.iinfo(np.int32).max
    return distance


def _triangle_stratum(distance: int, transition_rings: int) -> str:
    if distance == 0:
        return "boundary"
    if distance == 1:
        return "first_ring"
    if distance <= 1 + transition_rings:
        return "transition"
    return "true_interior"


def _edge_stratum(
    edge_index: int,
    is_constraint: bool,
    edge_triangles: Sequence[Sequence[int]],
    triangle_distance: np.ndarray,
    transition_rings: int,
) -> str:
    if is_constraint:
        return "boundary"
    owner_distances = triangle_distance[
        np.asarray(edge_triangles[edge_index], dtype=np.int64)
    ]
    distance = int(np.min(owner_distances))
    if distance <= 1:
        return "first_ring"
    if distance <= 1 + transition_rings:
        return "transition"
    return "true_interior"


def _sample_triplets(
    sampler: SizeSampler,
    endpoint_a: np.ndarray,
    endpoint_b: np.ndarray,
    midpoint: np.ndarray,
) -> np.ndarray:
    stacked = np.vstack((endpoint_a, endpoint_b, midpoint))
    sampled = _sample(sampler, stacked)
    edge_count = len(endpoint_a)
    return np.column_stack(
        (
            sampled[:edge_count],
            sampled[edge_count : 2 * edge_count],
            sampled[2 * edge_count :],
        )
    )


def _sample(sampler: SizeSampler, xy: np.ndarray) -> np.ndarray:
    values = np.asarray(sampler(np.asarray(xy, dtype=float)), dtype=float)
    if values.ndim == 0:
        values = np.full(len(xy), float(values), dtype=float)
    values = values.reshape(-1)
    if len(values) != len(xy):
        raise ValueError(
            "size sampler must return one value per requested coordinate"
        )
    return values


def _positive_row_min(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & (values > 0.0)
    safe = np.where(valid, values, np.inf)
    result = np.min(safe, axis=1)
    result[~np.any(valid, axis=1)] = np.nan
    return result


def _finite_positive_min(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    stacked = np.column_stack((left, right))
    return _positive_row_min(stacked)


def _boundary_node_target_array(
    source: np.ndarray | Mapping[int, float] | None,
    node_count: int,
) -> np.ndarray | None:
    if source is None:
        return None
    result = np.full(node_count, np.nan, dtype=float)
    if isinstance(source, Mapping):
        for raw_index, raw_value in source.items():
            index = int(raw_index)
            if index < 0 or index >= node_count:
                raise ValueError("boundary target mapping has an invalid node index")
            result[index] = float(raw_value)
        return result
    values = np.asarray(source, dtype=float).reshape(-1)
    if len(values) != node_count:
        raise ValueError(
            "boundary_target_by_node array must have one value per mesh node"
        )
    return values.copy()


def _ratio_summary(
    raw_values: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    values = np.asarray(raw_values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    summary: dict[str, Any] = {
        "count": int(len(values)),
        "finite_count": int(len(finite)),
        "invalid_count": int(len(values) - len(finite)),
        "quantiles": _quantiles(finite),
        "maximum": float(np.max(finite)) if len(finite) else None,
        "threshold_exceedance_counts": {
            _threshold_key(threshold): int(np.count_nonzero(finite > threshold))
            for threshold in thresholds
        },
    }
    return summary


def _target_size_hard_gate(
    triangle_summary: Mapping[str, Any],
    *,
    edge_summary: Mapping[str, Any],
    p95_limit: float,
    maximum_limit: float,
) -> dict[str, Any]:
    """Evaluate all-triangle limits while requiring every edge to be valid."""

    p95 = triangle_summary.get("quantiles", {}).get("p95")
    maximum = triangle_summary.get("maximum")
    invalid_edge_count = int(edge_summary.get("invalid_count", 0))
    invalid_triangle_count = int(
        triangle_summary.get("invalid_count", 0)
    )
    invalid = bool(
        invalid_edge_count > 0
        or invalid_triangle_count > 0
        or p95 is None
        or maximum is None
    )
    p95_exceeded = bool(
        p95 is not None and float(p95) > float(p95_limit)
    )
    maximum_exceeded = bool(
        maximum is not None and float(maximum) > float(maximum_limit)
    )
    failures: list[str] = []
    if invalid:
        failures.append("edge_aware_target_size_invalid")
    if p95_exceeded:
        failures.append("edge_aware_target_size_p95_exceeded")
    if maximum_exceeded:
        failures.append("edge_aware_target_size_maximum_exceeded")
    return {
        "definition": (
            "all_edges_valid_and_all_triangle_maximum_incident_edge_l_over_h"
        ),
        "p95": p95,
        "maximum": maximum,
        "limits": {
            "p95": float(p95_limit),
            "maximum": float(maximum_limit),
            "comparison": "inclusive_limit_passes",
        },
        "invalid_count": invalid_edge_count + invalid_triangle_count,
        "invalid_edge_count": invalid_edge_count,
        "invalid_triangle_count": invalid_triangle_count,
        "p95_exceeded": p95_exceeded,
        "maximum_exceeded": maximum_exceeded,
        "passed": not failures,
        "failure_taxonomy": failures,
    }


def _boundary_field_interface_diagnostic(
    is_constraint: np.ndarray,
    boundary_samples: np.ndarray,
    field_samples: np.ndarray,
    ratio_limit: float,
) -> dict[str, Any]:
    """Compare 1-D and 2-D targets pointwise on each constraint edge."""

    constraint = np.asarray(is_constraint, dtype=bool)
    boundary_values = np.asarray(boundary_samples, dtype=float)
    field_values = np.asarray(field_samples, dtype=float)
    if (
        boundary_values.shape != (len(constraint), 3)
        or field_values.shape != (len(constraint), 3)
    ):
        raise ValueError(
            "boundary and field interface samples must have shape (n_edges, 3)"
        )
    boundary_valid = np.isfinite(boundary_values) & (
        boundary_values > 0.0
    )
    field_valid = np.isfinite(field_values) & (field_values > 0.0)
    boundary_complete = constraint & np.all(boundary_valid, axis=1)
    field_complete = constraint & np.all(field_valid, axis=1)
    evaluated = boundary_complete & field_complete

    pointwise_ratio = np.full(boundary_values.shape, np.nan, dtype=float)
    pointwise_boundary_over_field = np.full(
        boundary_values.shape,
        np.nan,
        dtype=float,
    )
    pointwise_field_over_boundary = np.full(
        boundary_values.shape,
        np.nan,
        dtype=float,
    )
    pointwise_boundary_over_field[evaluated] = (
        boundary_values[evaluated] / field_values[evaluated]
    )
    pointwise_field_over_boundary[evaluated] = (
        field_values[evaluated] / boundary_values[evaluated]
    )
    pointwise_ratio[evaluated] = np.maximum(
        pointwise_boundary_over_field[evaluated],
        pointwise_field_over_boundary[evaluated],
    )
    edge_ratio = np.full(len(constraint), np.nan, dtype=float)
    edge_ratio[evaluated] = np.max(pointwise_ratio[evaluated], axis=1)
    exceeds = evaluated & (edge_ratio > float(ratio_limit))
    boundary_coarser = evaluated & np.any(
        pointwise_boundary_over_field > float(ratio_limit),
        axis=1,
    )
    boundary_finer = evaluated & np.any(
        pointwise_field_over_boundary > float(ratio_limit),
        axis=1,
    )
    missing_boundary = constraint & ~boundary_complete
    invalid_field = constraint & ~field_complete

    failures: list[str] = []
    if np.any(missing_boundary):
        failures.append("boundary_field_interface_boundary_target_incomplete")
    if np.any(invalid_field):
        failures.append("boundary_field_interface_2d_target_incomplete")
    if np.any(exceeds):
        failures.append("boundary_field_interface_ratio_limit_exceeded")

    ratio_summary = _value_summary(edge_ratio[evaluated])
    ratio_summary["limit"] = float(ratio_limit)
    edge_exceedance_count = int(np.count_nonzero(exceeds))
    ratio_summary["above_limit_count"] = edge_exceedance_count
    pointwise_summary = _value_summary(pointwise_ratio[evaluated].reshape(-1))
    pointwise_summary["limit"] = float(ratio_limit)
    sample_exceedance_count = int(
        np.count_nonzero(pointwise_ratio[evaluated] > float(ratio_limit))
    )
    pointwise_summary["above_limit_count"] = sample_exceedance_count
    limit_passed = bool(
        not np.any(missing_boundary)
        and not np.any(invalid_field)
        and not np.any(exceeds)
    )
    return {
        "definition": (
            "per-edge maximum of max(h_gamma/H, H/h_gamma) evaluated "
            "independently at endpoint_a, endpoint_b, and midpoint"
        ),
        "sample_locations": ["endpoint_a", "endpoint_b", "midpoint"],
        "reduction": "maximum_pointwise_symmetric_ratio",
        "strict_limit_comparison": True,
        "ratio_limit": float(ratio_limit),
        "factor_two_limit": float(ratio_limit),
        "constraint_edge_count": int(np.count_nonzero(constraint)),
        "evaluated_edge_count": int(np.count_nonzero(evaluated)),
        "incomplete_boundary_target_edge_count": int(
            np.count_nonzero(missing_boundary)
        ),
        "incomplete_2d_field_target_edge_count": int(
            np.count_nonzero(invalid_field)
        ),
        "boundary_coarser_than_field_count": int(
            np.count_nonzero(boundary_coarser)
        ),
        "boundary_finer_than_field_count": int(
            np.count_nonzero(boundary_finer)
        ),
        "symmetric_ratio": ratio_summary,
        "pointwise_symmetric_ratio": pointwise_summary,
        "ratio_limit_exceedance_count": edge_exceedance_count,
        "ratio_limit_sample_exceedance_count": sample_exceedance_count,
        "ratio_limit_passed": limit_passed,
        # Retain these top-level fields because archived comparison consumers
        # read them directly. They are aliases only: their values always
        # follow the configured ratio limit, which need not equal two.
        "factor_two_exceedance_count": edge_exceedance_count,
        "factor_two_sample_exceedance_count": sample_exceedance_count,
        "factor_two_passed": limit_passed,
        "legacy_compatibility_aliases": {
            "deprecated": True,
            "semantic_note": (
                "factor_two names are historical aliases; values follow "
                "the configured ratio_limit and do not imply a limit of 2"
            ),
            "aliases": {
                "factor_two_limit": "ratio_limit",
                "factor_two_exceedance_count": (
                    "ratio_limit_exceedance_count"
                ),
                "factor_two_sample_exceedance_count": (
                    "ratio_limit_sample_exceedance_count"
                ),
                "factor_two_passed": "ratio_limit_passed",
            },
        },
        "failure_taxonomy": failures,
        "passed": not failures,
    }


def _boundary_first_ring_continuity(
    chains: Sequence[Mapping[str, Any]],
    edge_lookup: Mapping[tuple[int, int], int],
    edge_triangles: Sequence[Sequence[int]],
    edge_lengths: np.ndarray,
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    p95_limit: float,
    maximum_limit: float,
) -> dict[str, Any]:
    """Compare realized boundary edges with their incident triangle scale."""

    triangle_xy = np.asarray(points, dtype=float)[
        np.asarray(triangles, dtype=np.int64)
    ]
    twice_area = np.abs(
        (triangle_xy[:, 1, 0] - triangle_xy[:, 0, 0])
        * (triangle_xy[:, 2, 1] - triangle_xy[:, 0, 1])
        - (triangle_xy[:, 1, 1] - triangle_xy[:, 0, 1])
        * (triangle_xy[:, 2, 0] - triangle_xy[:, 0, 0])
    )
    triangle_area = 0.5 * twice_area
    characteristic = np.sqrt(
        4.0 * triangle_area / np.sqrt(3.0)
    )

    global_values: list[float] = []
    global_pairs: set[tuple[int, int]] = set()
    global_invalid_pairs: set[tuple[int, int]] = set()
    chain_reports: list[dict[str, Any]] = []
    chain_p95_exceedance_count = 0
    chain_maximum_exceedance_count = 0

    for chain in chains:
        values: list[float] = []
        invalid_count = 0
        incident_count = 0
        for edge in chain["edges"]:
            edge_index = int(edge_lookup[edge])
            owners = edge_triangles[edge_index]
            if not owners:
                invalid_count += 1
                continue
            for raw_triangle_index in owners:
                triangle_index = int(raw_triangle_index)
                incident_count += 1
                pair = (edge_index, triangle_index)
                edge_length = float(edge_lengths[edge_index])
                triangle_length = float(characteristic[triangle_index])
                if (
                    not np.isfinite(edge_length)
                    or edge_length <= 0.0
                    or not np.isfinite(triangle_length)
                    or triangle_length <= 0.0
                ):
                    invalid_count += 1
                    global_invalid_pairs.add(pair)
                    continue
                ratio = _symmetric_ratio(edge_length, triangle_length)
                values.append(ratio)
                if pair not in global_pairs:
                    global_pairs.add(pair)
                    global_values.append(ratio)

        summary = _value_summary(values)
        p95 = summary["quantiles"]["p95"]
        maximum = summary["maximum"]
        p95_exceeded = bool(
            p95 is not None and float(p95) > float(p95_limit)
        )
        maximum_exceeded = bool(
            maximum is not None and float(maximum) > float(maximum_limit)
        )
        if p95_exceeded:
            chain_p95_exceedance_count += 1
        if maximum_exceeded:
            chain_maximum_exceedance_count += 1
        chain_failures: list[str] = []
        if invalid_count:
            chain_failures.append(
                "boundary_first_ring_characteristic_length_invalid"
            )
        if p95_exceeded:
            chain_failures.append(
                "boundary_first_ring_realized_ratio_p95_exceeded"
            )
        if maximum_exceeded:
            chain_failures.append(
                "boundary_first_ring_realized_ratio_maximum_exceeded"
            )
        chain_reports.append(
            {
                "chain_id": str(chain["chain_id"]),
                "cyclic": bool(chain["cyclic"]),
                "constraint_edge_count": int(len(chain["edges"])),
                "incident_triangle_count": int(incident_count),
                "invalid_incident_count": int(invalid_count),
                "symmetric_ratio": summary,
                "p95_exceeded": p95_exceeded,
                "maximum_exceeded": maximum_exceeded,
                "passed": not chain_failures,
                "failure_taxonomy": chain_failures,
            }
        )

    global_summary = _value_summary(global_values)
    global_p95 = global_summary["quantiles"]["p95"]
    global_maximum = global_summary["maximum"]
    global_p95_exceeded = bool(
        global_p95 is not None and float(global_p95) > float(p95_limit)
    )
    global_maximum_exceeded = bool(
        global_maximum is not None
        and float(global_maximum) > float(maximum_limit)
    )
    invalid_count = int(len(global_invalid_pairs))
    failures: list[str] = []
    if invalid_count or any(
        int(record["invalid_incident_count"]) > 0
        for record in chain_reports
    ):
        failures.append("boundary_first_ring_characteristic_length_invalid")
    if global_p95_exceeded or chain_p95_exceedance_count:
        failures.append("boundary_first_ring_realized_ratio_p95_exceeded")
    if global_maximum_exceeded or chain_maximum_exceedance_count:
        failures.append(
            "boundary_first_ring_realized_ratio_maximum_exceeded"
        )

    return {
        "definition": (
            "symmetric ratio max(L_boundary/h_incident, "
            "h_incident/L_boundary) for each constraint edge and each "
            "incident boundary-row triangle"
        ),
        "first_ring_interpretation": (
            "triangle directly incident to a constraint edge"
        ),
        "triangle_characteristic_length": (
            "equilateral side with equal area: sqrt(4*A/sqrt(3))"
        ),
        "limits": {
            "p95": float(p95_limit),
            "maximum": float(maximum_limit),
            "comparison": "inclusive_limit_passes",
        },
        "global": {
            "constraint_edge_count": int(
                len(
                    {
                        int(edge_lookup[edge])
                        for chain in chains
                        for edge in chain["edges"]
                    }
                )
            ),
            "incident_triangle_count": int(
                len(global_pairs | global_invalid_pairs)
            ),
            "invalid_incident_count": invalid_count,
            "symmetric_ratio": global_summary,
            "p95_exceeded": global_p95_exceeded,
            "maximum_exceeded": global_maximum_exceeded,
            "chain_p95_exceedance_count": int(
                chain_p95_exceedance_count
            ),
            "chain_maximum_exceedance_count": int(
                chain_maximum_exceedance_count
            ),
        },
        "chains": chain_reports,
        "passed": not failures,
        "failure_taxonomy": failures,
    }


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {
            "minimum": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    probabilities = (0.0, 0.01, 0.05, 0.50, 0.95, 0.99)
    labels = ("minimum", "p01", "p05", "p50", "p95", "p99")
    sampled = np.quantile(values, probabilities)
    return {
        label: float(value)
        for label, value in zip(labels, sampled)
    }


def _threshold_key(value: float) -> str:
    return f"above_{value:g}".replace(".", "_")


def _boundary_diagnostics(
    chains: Sequence[Mapping[str, Any]],
    edge_lookup: Mapping[tuple[int, int], int],
    lengths: np.ndarray,
    targets: np.ndarray,
    ratios: np.ndarray,
    points: np.ndarray,
    gradation_limit: float,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    chain_reports: list[dict[str, Any]] = []
    adjacent_length_ratios: list[float] = []
    adjacent_target_ratios: list[float] = []
    target_gradations: list[float] = []
    realized_gradations: list[float] = []

    for chain in chains:
        edge_indices = np.asarray(
            [edge_lookup[edge] for edge in chain["edges"]],
            dtype=np.int64,
        )
        chain_lengths = lengths[edge_indices]
        chain_targets = targets[edge_indices]
        chain_ratios = ratios[edge_indices]
        edge_midpoints = np.asarray(
            [
                0.5 * (points[a] + points[b])
                for a, b in chain["edges"]
            ],
            dtype=float,
        )
        adjacent_pairs = list(zip(range(len(edge_indices) - 1), range(1, len(edge_indices))))
        if bool(chain["cyclic"]) and len(edge_indices) > 1:
            adjacent_pairs.append((len(edge_indices) - 1, 0))
        local_length_ratios: list[float] = []
        local_target_ratios: list[float] = []
        local_target_gradations: list[float] = []
        local_realized_gradations: list[float] = []
        for left, right in adjacent_pairs:
            midpoint_distance = float(
                np.linalg.norm(edge_midpoints[right] - edge_midpoints[left])
            )
            length_ratio = _symmetric_ratio(
                chain_lengths[left],
                chain_lengths[right],
            )
            target_ratio = _symmetric_ratio(
                chain_targets[left],
                chain_targets[right],
            )
            target_gradation = _normalized_delta(
                chain_targets[left],
                chain_targets[right],
                midpoint_distance,
            )
            realized_gradation = _normalized_delta(
                chain_lengths[left],
                chain_lengths[right],
                midpoint_distance,
            )
            for value, local, aggregate in (
                (length_ratio, local_length_ratios, adjacent_length_ratios),
                (target_ratio, local_target_ratios, adjacent_target_ratios),
                (target_gradation, local_target_gradations, target_gradations),
                (
                    realized_gradation,
                    local_realized_gradations,
                    realized_gradations,
                ),
            ):
                if np.isfinite(value):
                    local.append(float(value))
                    aggregate.append(float(value))
        chain_reports.append(
            {
                "chain_id": str(chain["chain_id"]),
                "cyclic": bool(chain["cyclic"]),
                "edge_count": int(len(edge_indices)),
                "edge_l_over_h": _ratio_summary(chain_ratios, thresholds),
                "adjacent_edge_length_ratio": _value_summary(
                    local_length_ratios
                ),
                "adjacent_edge_target_ratio": _value_summary(
                    local_target_ratios
                ),
                "target_gradation": _gradation_summary(
                    local_target_gradations,
                    gradation_limit,
                ),
                "realized_edge_length_gradation": _gradation_summary(
                    local_realized_gradations,
                    gradation_limit,
                ),
            }
        )

    return {
        "chains": chain_reports,
        "aggregate": {
            "adjacent_edge_length_ratio": _value_summary(
                adjacent_length_ratios
            ),
            "adjacent_edge_target_ratio": _value_summary(
                adjacent_target_ratios
            ),
            "target_gradation": _gradation_summary(
                target_gradations,
                gradation_limit,
            ),
            "realized_edge_length_gradation": _gradation_summary(
                realized_gradations,
                gradation_limit,
            ),
        },
    }


def _value_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return {
        "count": int(len(finite)),
        "quantiles": _quantiles(finite),
        "maximum": float(np.max(finite)) if len(finite) else None,
    }


def _gradation_summary(
    values: Sequence[float],
    limit: float,
) -> dict[str, Any]:
    summary = _value_summary(values)
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    summary["limit"] = float(limit)
    summary["above_limit_count"] = int(np.count_nonzero(finite > limit))
    return summary


def _symmetric_ratio(left: float, right: float) -> float:
    if (
        not np.isfinite(left)
        or not np.isfinite(right)
        or left <= 0.0
        or right <= 0.0
    ):
        return float("nan")
    return float(max(left, right) / min(left, right))


def _normalized_delta(left: float, right: float, distance: float) -> float:
    if (
        not np.isfinite(left)
        or not np.isfinite(right)
        or not np.isfinite(distance)
        or left <= 0.0
        or right <= 0.0
        or distance <= 0.0
    ):
        return float("nan")
    return float(abs(right - left) / distance)


def _canonical_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


__all__ = [
    "DEFAULT_THRESHOLDS",
    "SCHEMA",
    "STRATA",
    "audit_edge_target_sizes",
]
