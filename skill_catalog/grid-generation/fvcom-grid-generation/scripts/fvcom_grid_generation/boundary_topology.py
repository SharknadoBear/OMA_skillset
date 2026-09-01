"""Deterministic Grid-owned compensation for pre-mesh island topology.

The Adaptive-v2 source package remains immutable.  This module removes whole
island chains that are not strictly inside the exterior and merges island
groups whose separation is no larger than the finest participating boundary
spacing.  Exterior and OBC geometry are never edited here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import nearest_points, unary_union

from .projection import LocalProjection, unproject_geometry


SCHEMA_VERSION = "fvcom_grid_boundary_topology_compensation_v1"


@dataclass(frozen=True)
class DeliveredIsland:
    """One compensated island chain and its source lineage."""

    xy: np.ndarray
    target_spacing_m: np.ndarray
    source_indices: tuple[int, ...]
    source_chain_ids: tuple[str, ...]
    unchanged: bool


@dataclass(frozen=True)
class BoundaryTopologyCompensation:
    """Normalized topology plus non-serialized plotting evidence."""

    exterior_xy: np.ndarray
    source_islands_xy: tuple[np.ndarray, ...]
    delivered_islands: tuple[DeliveredIsland, ...]
    wet_domain_xy: Polygon
    report: dict[str, Any]


@dataclass(frozen=True)
class _IslandState:
    polygon: Polygon
    xy: np.ndarray
    target_spacing_m: np.ndarray
    source_indices: tuple[int, ...]
    source_chain_ids: tuple[str, ...]
    source_h_m: tuple[float, ...]
    unchanged: bool

    @property
    def h_m(self) -> float:
        return float(min(self.source_h_m))


def _ring_xy(values: Iterable[Iterable[float]]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Boundary ring coordinates must have shape (n, 2)")
    if len(array) > 1 and np.allclose(array[0], array[-1], rtol=0.0, atol=1.0e-9):
        array = array[:-1]
    if len(array) < 3 or not np.all(np.isfinite(array)):
        raise ValueError("Boundary ring requires at least three finite vertices")
    return np.asarray(array, dtype=float)


def _geometry_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_binding(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        return {"path": str(candidate), "exists": False, "sha256": None}
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(candidate), "exists": True, "sha256": digest.hexdigest()}


def _positive_minimum(values: np.ndarray, source_chain_id: str) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array) & (array > 0.0)]
    if not len(finite):
        raise ValueError(
            f"Island chain {source_chain_id!r} has no finite positive target_spacing_m"
        )
    return float(np.min(finite))


def _strictly_inside(exterior: Polygon, island: Polygon) -> bool:
    return bool(
        island.is_valid
        and not island.is_empty
        and island.area > 0.0
        and exterior.contains(island)
        and not exterior.boundary.intersects(island)
    )


def _eligible_edges(states: list[_IslandState]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for left_index, left in enumerate(states):
        for right_index in range(left_index + 1, len(states)):
            right = states[right_index]
            gap = float(left.polygon.distance(right.polygon))
            threshold = float(min(left.h_m, right.h_m))
            epsilon = max(1.0e-8, threshold * 1.0e-10)
            if gap <= threshold + epsilon:
                edges.append(
                    {
                        "left": int(left_index),
                        "right": int(right_index),
                        "gap_m": gap,
                        "threshold_m": threshold,
                        "normalized_gap": gap / threshold,
                        "left_source_indices": list(left.source_indices),
                        "right_source_indices": list(right.source_indices),
                    }
                )
    return sorted(
        edges,
        key=lambda item: (
            float(item["normalized_gap"]),
            float(item["gap_m"]),
            tuple(item["left_source_indices"]),
            tuple(item["right_source_indices"]),
        ),
    )


def _connected_components(count: int, edges: list[dict[str, Any]]) -> list[list[int]]:
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for edge in edges:
        union(int(edge["left"]), int(edge["right"]))
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return sorted(groups.values(), key=lambda values: min(values))


def _bridge_union(states: list[_IslandState], edges: list[dict[str, Any]]) -> Polygon:
    """Join one connected island group without removing any source land area."""

    source_union = unary_union([state.polygon for state in states])
    if isinstance(source_union, Polygon):
        return source_union

    bridges = []
    for edge in edges:
        left = states[int(edge["left"])].polygon
        right = states[int(edge["right"])].polygon
        point_left, point_right = nearest_points(left, right)
        threshold = float(edge["threshold_m"])
        gap = float(edge["gap_m"])
        # The eligibility decision is exactly the half-buffer rule.  The
        # retained bridge is narrower than that decision envelope and is
        # unioned with all original land, so no source island is eroded.
        bridge_half_width = max(1.0e-6, min(0.05 * threshold, 0.25 * max(gap, 1.0e-6)))
        connector = LineString([point_left, point_right])
        if connector.length <= 1.0e-9:
            bridges.append(point_left.buffer(bridge_half_width))
        else:
            bridges.append(connector.buffer(bridge_half_width, cap_style=3, join_style=2))
    merged = unary_union([source_union, *bridges]).buffer(0)
    if isinstance(merged, MultiPolygon):
        # A point-touch can survive a zero-width topological union.  Use the
        # declared half-buffer envelope only for the still-disconnected parts.
        radius = 0.5 * min(float(edge["threshold_m"]) for edge in edges)
        radius += max(1.0e-6, radius * 1.0e-9)
        merged = unary_union([state.polygon for state in states]).buffer(
            radius, join_style=2
        ).buffer(-radius, join_style=2)
        merged = unary_union([merged, source_union]).buffer(0)
    if not isinstance(merged, Polygon) or merged.is_empty or not merged.is_valid:
        raise ValueError("Eligible island group could not be reconstructed as one valid polygon")
    return merged


def _resample_ring(polygon: Polygon, spacing_m: float) -> np.ndarray:
    ring = LineString(polygon.exterior.coords)
    length = float(ring.length)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("Merged island has a nonpositive perimeter")
    count = max(3, int(math.ceil(length / max(float(spacing_m), 1.0e-9))))
    distances = np.linspace(0.0, length, count, endpoint=False)
    values = np.asarray(
        [[float(ring.interpolate(value).x), float(ring.interpolate(value).y)] for value in distances],
        dtype=float,
    )
    if len(values) < 3 or len(np.unique(values, axis=0)) < 3:
        raise ValueError("Merged island resampling produced fewer than three unique vertices")
    return values


def _merge_component(
    member_states: list[_IslandState],
    component_edges: list[dict[str, Any]],
) -> _IslandState:
    polygon = _bridge_union(member_states, component_edges)
    source_indices = tuple(
        sorted(index for state in member_states for index in state.source_indices)
    )
    source_chain_ids = tuple(
        chain_id
        for index, chain_id in sorted(
            (
                (source_index, chain_id)
                for state in member_states
                for source_index, chain_id in zip(state.source_indices, state.source_chain_ids)
            ),
            key=lambda item: item[0],
        )
    )
    source_h = tuple(
        h
        for state in sorted(member_states, key=lambda value: min(value.source_indices))
        for h in state.source_h_m
    )
    h_m = float(min(source_h))
    xy = _resample_ring(polygon, h_m)
    return _IslandState(
        polygon=Polygon(xy),
        xy=xy,
        target_spacing_m=np.full(len(xy), h_m, dtype=float),
        source_indices=source_indices,
        source_chain_ids=source_chain_ids,
        source_h_m=source_h,
        unchanged=False,
    )


def _domain_checks(exterior: Polygon, islands: list[_IslandState]) -> tuple[Polygon, dict[str, Any]]:
    for state in islands:
        if not _strictly_inside(exterior, state.polygon):
            raise ValueError("Compensated island is not strictly contained by the exterior")
    for left_index, left in enumerate(islands):
        for right in islands[left_index + 1 :]:
            if left.polygon.distance(right.polygon) <= 1.0e-8:
                raise ValueError("Compensated island holes are not mutually disjoint")
    wet = Polygon(
        exterior.exterior.coords,
        holes=[state.polygon.exterior.coords for state in islands],
    )
    valid = bool(
        isinstance(wet, Polygon)
        and wet.is_valid
        and not wet.is_empty
        and wet.area > 0.0
        and len(wet.interiors) == len(islands)
    )
    if not valid:
        raise ValueError("Compensated island chains do not reconstruct one valid wet polygon")
    return wet, {
        "wet_domain_valid": True,
        "wet_component_count": 1,
        "island_chain_count": int(len(islands)),
        "polygon_hole_count": int(len(wet.interiors)),
        "exact_chain_hole_agreement": bool(len(wet.interiors) == len(islands)),
        "holes_strictly_contained": True,
        "holes_mutually_disjoint": True,
    }


def normalize_boundary_topology(
    exterior_xy: Iterable[Iterable[float]],
    island_xy: Iterable[Iterable[Iterable[float]]],
    island_target_spacing_m: Iterable[Iterable[float]],
    *,
    source_chain_ids: Iterable[str] | None = None,
    reference_holes_xy: Iterable[Iterable[Iterable[float]]] | None = None,
    source_resolution_manifest: str | Path | None = None,
    source_boundary_gpkg: str | Path | None = None,
    protected_exterior_contract: dict[str, Any] | None = None,
) -> BoundaryTopologyCompensation:
    """Apply the two authorized Grid compensation classes deterministically."""

    exterior_array = _ring_xy(exterior_xy)
    exterior = Polygon(exterior_array)
    if not exterior.is_valid or exterior.is_empty or exterior.area <= 0.0:
        raise ValueError("Source exterior is not one valid positive-area polygon")
    source_arrays = tuple(_ring_xy(values) for values in island_xy)
    target_arrays = tuple(np.asarray(list(values), dtype=float) for values in island_target_spacing_m)
    if len(source_arrays) != len(target_arrays):
        raise ValueError("Island-chain and target-spacing counts differ")
    ids = tuple(
        str(value)
        for value in (
            source_chain_ids
            if source_chain_ids is not None
            else [f"island_{index + 1:03d}" for index in range(len(source_arrays))]
        )
    )
    if len(ids) != len(source_arrays) or len(set(ids)) != len(ids):
        raise ValueError("Island source-chain IDs must be unique and complete")
    reference_arrays = (
        tuple(_ring_xy(values) for values in reference_holes_xy)
        if reference_holes_xy is not None
        else ()
    )
    reference_polygons = tuple(Polygon(values) for values in reference_arrays)
    if any(
        polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0
        for polygon in reference_polygons
    ):
        raise ValueError("Resolved-domain reference holes must be valid positive-area polygons")

    states: list[_IslandState] = []
    removed_actions: list[dict[str, Any]] = []
    for index, (chain_id, xy, targets) in enumerate(zip(ids, source_arrays, target_arrays)):
        if len(targets) != len(xy):
            raise ValueError(f"Island chain {chain_id!r} target count differs from its vertices")
        polygon = Polygon(xy)
        if not polygon.is_valid or polygon.is_empty or polygon.area <= 0.0:
            raise ValueError(f"Island chain {chain_id!r} is independently invalid")
        h_m = _positive_minimum(targets, chain_id)
        if not _strictly_inside(exterior, polygon):
            outside_area = float(polygon.difference(exterior).area)
            boundary_contact = polygon.intersection(exterior.boundary)
            removed_actions.append(
                {
                    "action_type": "remove_exterior_conflict",
                    "source_indices_one_based": [int(index + 1)],
                    "source_chain_ids": [chain_id],
                    "minimum_target_spacing_m": h_m,
                    "source_area_m2": float(polygon.area),
                    "removed_area_m2": float(polygon.area),
                    "outside_exterior_area_m2": outside_area,
                    "exterior_contact_length_m": float(boundary_contact.length),
                    "exterior_contact_area_m2": float(boundary_contact.area),
                    "source_geometry_sha256": _geometry_sha256(xy),
                    "source_bounds_xy": [float(value) for value in polygon.bounds],
                    "reason": "complete_island_touches_crosses_or_extends_outside_exterior",
                }
            )
            continue
        states.append(
            _IslandState(
                polygon=polygon,
                xy=xy,
                target_spacing_m=targets.copy(),
                source_indices=(index,),
                source_chain_ids=(chain_id,),
                source_h_m=(h_m,),
                unchanged=True,
            )
        )

    merge_actions: list[dict[str, Any]] = []
    source_reference_hole: dict[int, int | None] = {}
    if reference_polygons:
        for state in states:
            source_index = int(state.source_indices[0])
            overlaps = [float(state.polygon.intersection(hole).area) for hole in reference_polygons]
            selected = int(np.argmax(overlaps)) if overlaps and max(overlaps) > 0.0 else None
            if selected is None:
                distances = [float(state.polygon.distance(hole)) for hole in reference_polygons]
                nearest = int(np.argmin(distances)) if distances else None
                if nearest is not None and distances[nearest] <= state.h_m + 1.0e-8:
                    selected = nearest
            source_reference_hole[source_index] = selected

    edges = _eligible_edges(states)
    if reference_polygons:
        edges = [
            edge
            for edge in edges
            if (
                source_reference_hole.get(int(edge["left_source_indices"][0])) is not None
                and source_reference_hole.get(int(edge["left_source_indices"][0]))
                == source_reference_hole.get(int(edge["right_source_indices"][0]))
            )
        ]
    components = _connected_components(len(states), edges)
    next_states: list[_IslandState] = []
    for component in components:
        members = [states[index] for index in component]
        if len(members) == 1:
            next_states.append(members[0])
            continue
        member_set = set(component)
        local_index = {value: index for index, value in enumerate(component)}
        component_edges = [
            {
                **edge,
                "left": local_index[int(edge["left"])],
                "right": local_index[int(edge["right"])],
            }
            for edge in edges
            if int(edge["left"]) in member_set and int(edge["right"]) in member_set
        ]
        merged = _merge_component(members, component_edges)
        source_area = float(sum(state.polygon.area for state in members))
        source_union_area = float(unary_union([state.polygon for state in members]).area)
        action = {
            "action_type": "merge_island_group",
            "source_indices_one_based": [int(value + 1) for value in merged.source_indices],
            "source_chain_ids": list(merged.source_chain_ids),
            "reference_hole_index_one_based": (
                int(source_reference_hole[merged.source_indices[0]] + 1)
                if reference_polygons
                else None
            ),
            "minimum_target_spacing_m": float(merged.h_m),
            "source_area_sum_m2": source_area,
            "source_union_area_m2": source_union_area,
            "delivered_area_m2": float(merged.polygon.area),
            "added_land_area_m2": float(max(0.0, merged.polygon.area - source_union_area)),
            "lost_land_area_m2": float(max(0.0, source_union_area - merged.polygon.area)),
            "resampled_vertex_count": int(len(merged.xy)),
            "delivered_geometry_sha256": _geometry_sha256(merged.xy),
            "delivered_bounds_xy": [float(value) for value in merged.polygon.bounds],
            "eligible_pair_evidence": [
                {
                    "left_source_indices_one_based": [
                        int(value + 1) for value in edge["left_source_indices"]
                    ],
                    "right_source_indices_one_based": [
                        int(value + 1) for value in edge["right_source_indices"]
                    ],
                    "gap_m": float(edge["gap_m"]),
                    "threshold_m": float(edge["threshold_m"]),
                    "criterion_passed": bool(
                        float(edge["gap_m"]) <= float(edge["threshold_m"]) + 1.0e-8
                    ),
                }
                for edge in component_edges
            ],
            "reason": "overlap_touch_or_sub_resolution_gap_with_same_resolved_s2_hole",
        }
        if not _strictly_inside(exterior, merged.polygon):
            action["post_merge_action"] = "remove_exterior_conflict"
            removed_actions.append(
                {
                    "action_type": "remove_exterior_conflict",
                    "source_indices_one_based": action["source_indices_one_based"],
                    "source_chain_ids": action["source_chain_ids"],
                    "minimum_target_spacing_m": action["minimum_target_spacing_m"],
                    "source_area_m2": action["delivered_area_m2"],
                    "removed_area_m2": action["delivered_area_m2"],
                    "source_geometry_sha256": action["delivered_geometry_sha256"],
                    "source_bounds_xy": action["delivered_bounds_xy"],
                    "reason": "merged_island_touches_crosses_or_extends_outside_exterior",
                }
            )
        else:
            next_states.append(merged)
        merge_actions.append(action)
    states = sorted(next_states, key=lambda value: min(value.source_indices))

    wet_domain, validity = _domain_checks(exterior, states)
    if reference_polygons and len(states) != len(reference_polygons):
        raise ValueError(
            "Authorized compensation does not reproduce the resolved S2 hole count: "
            f"{len(states)} delivered versus {len(reference_polygons)} reference holes"
        )
    delivered: list[DeliveredIsland] = []
    source_to_delivered: list[dict[str, Any]] = []
    for delivered_index, state in enumerate(states, start=1):
        delivered.append(
            DeliveredIsland(
                xy=state.xy.copy(),
                target_spacing_m=state.target_spacing_m.copy(),
                source_indices=state.source_indices,
                source_chain_ids=state.source_chain_ids,
                unchanged=bool(state.unchanged and len(state.source_indices) == 1),
            )
        )
        source_to_delivered.append(
            {
                "delivered_chain_index_one_based": int(delivered_index),
                "source_indices_one_based": [int(value + 1) for value in state.source_indices],
                "source_chain_ids": list(state.source_chain_ids),
                "unchanged": bool(state.unchanged and len(state.source_indices) == 1),
                "delivered_vertex_count": int(len(state.xy)),
                "delivered_minimum_target_spacing_m": float(state.h_m),
                "delivered_geometry_sha256": _geometry_sha256(state.xy),
            }
        )
    removed_source_indices = sorted(
        {
            int(value - 1)
            for action in removed_actions
            for value in action["source_indices_one_based"]
        }
    )
    for source_index in removed_source_indices:
        source_to_delivered.append(
            {
                "delivered_chain_index_one_based": None,
                "source_indices_one_based": [int(source_index + 1)],
                "source_chain_ids": [ids[source_index]],
                "unchanged": False,
                "disposition": "removed_exterior_conflict",
            }
        )
    source_to_delivered.sort(key=lambda value: min(value["source_indices_one_based"]))

    source_land_area = float(sum(Polygon(values).area for values in source_arrays))
    delivered_land_area = float(sum(Polygon(value.xy).area for value in delivered))
    exterior_contract_hash = _canonical_sha256(protected_exterior_contract or {})
    report = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "grid_owned_deterministic_island_topology_normalizer_v1",
        "status": "pass",
        "changed": bool(removed_actions or merge_actions),
        "source_bindings": {
            "boundary_resolution_manifest": _file_binding(source_resolution_manifest),
            "boundary_resolution_gpkg": _file_binding(source_boundary_gpkg),
        },
        "policy": {
            "exterior_conflicts_first": True,
            "exterior_conflict_action": "remove_complete_island_chain",
            "island_pair_threshold": "min(h_i,h_j)",
            "h_i_definition": "minimum_finite_positive_target_spacing_m",
            "buffer_equivalence": "each_side_threshold_over_two",
            "merge_resampling_spacing": "finest_member_spacing",
            "resolved_s2_hole_guard": bool(reference_polygons),
            "upstream_loop_or_visual_review": False,
        },
        "counts": {
            "source_island_chain_count": int(len(source_arrays)),
            "resolved_s2_reference_hole_count": int(len(reference_polygons)),
            "delivered_island_chain_count": int(len(delivered)),
            "removed_source_chain_count": int(len(removed_source_indices)),
            "merge_action_count": int(len(merge_actions)),
            "merged_source_chain_count": int(
                len(
                    {
                        value
                        for action in merge_actions
                        for value in action["source_indices_one_based"]
                    }
                )
            ),
            "net_chain_count_change": int(len(delivered) - len(source_arrays)),
        },
        "areas_m2": {
            "source_island_area_sum": source_land_area,
            "delivered_island_area_sum": delivered_land_area,
            "net_land_area_change": float(delivered_land_area - source_land_area),
            "wet_domain_area": float(wet_domain.area),
        },
        "actions": [*removed_actions, *merge_actions],
        "source_to_delivered_chains": source_to_delivered,
        "validity": validity,
        "invariants": {
            "exterior_coordinates_unchanged": True,
            "source_exterior_geometry_sha256": _geometry_sha256(exterior_array),
            "delivered_exterior_geometry_sha256": _geometry_sha256(exterior_array),
            "protected_exterior_contract_sha256_before": exterior_contract_hash,
            "protected_exterior_contract_sha256_after": exterior_contract_hash,
            "obc_coordinates_order_ids_segmentation_hard_anchors_unchanged": True,
        },
    }
    report["report_content_sha256"] = _canonical_sha256(report)
    return BoundaryTopologyCompensation(
        exterior_xy=exterior_array.copy(),
        source_islands_xy=source_arrays,
        delivered_islands=tuple(delivered),
        wet_domain_xy=wet_domain,
        report=report,
    )


def expected_hole_count_matches(
    expected: int | None,
    compensation: BoundaryTopologyCompensation,
) -> tuple[bool, str]:
    """Accept an expectation bound to either immutable source or delivery."""

    if expected is None:
        return True, "not_declared"
    source = int(compensation.report["counts"]["source_island_chain_count"])
    delivered = int(compensation.report["counts"]["delivered_island_chain_count"])
    if int(expected) == source:
        return True, "matched_source_before_authorized_compensation"
    if int(expected) == delivered:
        return True, "matched_delivered_after_authorized_compensation"
    return False, "mismatch_source_and_delivered_counts"


def _action_bounds(compensation: BoundaryTopologyCompensation, action: dict[str, Any]) -> tuple[float, float, float, float]:
    indices = [int(value) - 1 for value in action.get("source_indices_one_based", [])]
    polygons = [Polygon(compensation.source_islands_xy[index]) for index in indices]
    if action.get("delivered_bounds_xy"):
        minx, miny, maxx, maxy = map(float, action["delivered_bounds_xy"])
        polygons.append(Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]))
    union = unary_union(polygons)
    return tuple(float(value) for value in union.bounds)


def _plot_ring(ax, xy: np.ndarray, **kwargs: Any) -> None:
    values = np.vstack([np.asarray(xy, dtype=float), np.asarray(xy, dtype=float)[0]])
    ax.plot(values[:, 0], values[:, 1], **kwargs)


def write_boundary_topology_compensation(
    output_dir: str | Path,
    compensation: BoundaryTopologyCompensation,
    projection: LocalProjection,
) -> dict[str, str]:
    """Write the hash-bound report plus overview and action-zoom maps."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "boundary_topology_compensation.json"
    overview_path = output / "boundary_topology_compensation_overview.png"
    zoom_path = output / "boundary_topology_compensation_zoom.png"
    report_path.write_text(
        json.dumps(compensation.report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_ring(ax, compensation.exterior_xy, color="black", linewidth=1.1, label="unchanged exterior")
    for index, values in enumerate(compensation.source_islands_xy):
        _plot_ring(
            ax,
            values,
            color="#d95f5f",
            linewidth=0.6,
            alpha=0.55,
            label="source island chains" if index == 0 else None,
        )
    for index, island in enumerate(compensation.delivered_islands):
        _plot_ring(
            ax,
            island.xy,
            color="#1769aa",
            linewidth=0.9,
            alpha=0.9,
            label="delivered island chains" if index == 0 else None,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Grid pre-mesh island topology compensation\n"
        f"{len(compensation.source_islands_xy)} source → "
        f"{len(compensation.delivered_islands)} delivered chains"
    )
    ax.set_xlabel(f"projected x (m), {projection.crs.to_string()}")
    ax.set_ylabel("projected y (m)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(overview_path, dpi=180)
    plt.close(fig)

    actions = list(compensation.report.get("actions") or [])
    cols = 2
    rows = max(1, int(math.ceil(max(1, len(actions)) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5.5 * rows), squeeze=False)
    flat = list(axes.flat)
    if not actions:
        flat[0].text(0.5, 0.5, "No island topology compensation was required", ha="center", va="center")
        flat[0].set_axis_off()
    for ax, action_index, action in zip(flat, range(1, len(actions) + 1), actions):
        indices = [int(value) - 1 for value in action.get("source_indices_one_based", [])]
        for source_index in indices:
            _plot_ring(ax, compensation.source_islands_xy[source_index], color="#d62728", linewidth=1.3)
        source_set = set(indices)
        for island in compensation.delivered_islands:
            if source_set.intersection(island.source_indices):
                _plot_ring(ax, island.xy, color="#1f77b4", linewidth=1.5)
        minx, miny, maxx, maxy = _action_bounds(compensation, action)
        span = max(maxx - minx, maxy - miny, float(action.get("minimum_target_spacing_m", 1.0)))
        pad = max(0.15 * span, float(action.get("minimum_target_spacing_m", 1.0)))
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{action_index}. {action.get('action_type')} | "
            f"chains {action.get('source_indices_one_based')}"
        )
        ax.set_xlabel("projected x (m)")
        ax.set_ylabel("projected y (m)")
    for ax in flat[len(actions) if actions else 1 :]:
        ax.set_axis_off()
    fig.suptitle("Source (red) and compensated delivery (blue)")
    fig.tight_layout()
    fig.savefig(zoom_path, dpi=180)
    plt.close(fig)

    return {
        "report_json": str(report_path),
        "overview_map": str(overview_path),
        "zoom_map": str(zoom_path),
    }


__all__ = [
    "BoundaryTopologyCompensation",
    "DeliveredIsland",
    "SCHEMA_VERSION",
    "expected_hole_count_matches",
    "normalize_boundary_topology",
    "write_boundary_topology_compensation",
]
