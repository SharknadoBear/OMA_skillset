"""Allowed-edge policy for topology-only superthin connectivity closure.

The policy deliberately separates classification from mesh mutation.  It
identifies unprotected edges that participate in the current superthin debt,
records same-chain source-arc shortcuts, and carries accepted restrictions by
source-node lineage so compaction cannot silently make an illegal connection
legal again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .metrics import chain_edges, triangle_geometry


@dataclass(frozen=True)
class ConnectivityRestrictionConfig:
    """Conservative systematic-V5 connectivity policy."""

    enabled: bool = True
    maximum_transactions: int = 32
    maximum_candidates_per_component: int = 8
    patch_ring_ladder: tuple[int, ...] = (1, 2, 4)
    shortcut_arc_chord_ratio: float = 3.0
    shortcut_arc_target_ratio: float = 3.0


class AllowedEdgePolicy:
    """Classify and audit edges using stable source-node lineage."""

    def __init__(
        self,
        points: np.ndarray,
        target_spacing_m: np.ndarray,
        constraint_chains: Iterable[Iterable[int]],
        node_lineage: np.ndarray,
        *,
        restricted_lineage_edges: Iterable[Iterable[int]] = (),
        config: ConnectivityRestrictionConfig | None = None,
    ) -> None:
        self.points = np.asarray(points, dtype=float)
        self.targets = np.asarray(target_spacing_m, dtype=float)
        self.chains = [list(map(int, chain)) for chain in constraint_chains]
        self.lineage = np.asarray(node_lineage, dtype=int)
        self.config = config or ConnectivityRestrictionConfig()
        self.protected_edges = chain_edges(self.chains)
        self.restricted_lineage_edges = {
            _edge_key(int(values[0]), int(values[1]))
            for values in restricted_lineage_edges
            if len(values) >= 2 and int(values[0]) != int(values[1])
        }
        self._memberships: dict[int, list[tuple[int, int]]] = {}
        self._chain_cumulative: list[np.ndarray] = []
        self._chain_totals: list[float] = []
        for chain_index, chain in enumerate(self.chains):
            for position, node in enumerate(chain):
                self._memberships.setdefault(int(node), []).append(
                    (int(chain_index), int(position))
                )
            if len(chain) < 2:
                self._chain_cumulative.append(np.asarray([0.0], dtype=float))
                self._chain_totals.append(0.0)
                continue
            coordinates = self.points[np.asarray([*chain, chain[0]], dtype=int)]
            lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
            cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
            self._chain_cumulative.append(cumulative)
            self._chain_totals.append(float(cumulative[-1]))

    def lineage_edge(self, edge: Iterable[int]) -> tuple[int, int]:
        a, b = map(int, edge)
        return _edge_key(int(self.lineage[a]), int(self.lineage[b]))

    def is_persistently_restricted(self, edge: Iterable[int]) -> bool:
        return self.lineage_edge(edge) in self.restricted_lineage_edges

    def is_allowed(
        self,
        edge: Iterable[int],
        *,
        reject_same_chain_shortcuts: bool = False,
    ) -> bool:
        key = _edge_key(*map(int, edge))
        if key in self.protected_edges:
            return True
        if self.is_persistently_restricted(key):
            return False
        if reject_same_chain_shortcuts:
            return not bool(self.same_chain_shortcut_evidence(key)["is_shortcut"])
        return True

    def same_chain_shortcut_evidence(
        self,
        edge: Iterable[int],
    ) -> dict[str, Any]:
        a, b = map(int, edge)
        chord = float(np.linalg.norm(self.points[b] - self.points[a]))
        h = _harmonic_target(float(self.targets[a]), float(self.targets[b]))
        shared = sorted(
            set(self._memberships.get(a, []))
            and {
                left[0]
                for left in self._memberships.get(a, [])
            }
            & {
                right[0]
                for right in self._memberships.get(b, [])
            }
        )
        best: dict[str, Any] | None = None
        for chain_index in shared:
            left_positions = [
                position
                for index, position in self._memberships.get(a, [])
                if index == chain_index
            ]
            right_positions = [
                position
                for index, position in self._memberships.get(b, [])
                if index == chain_index
            ]
            chain = self.chains[int(chain_index)]
            cumulative = self._chain_cumulative[int(chain_index)]
            total = float(self._chain_totals[int(chain_index)])
            for left in left_positions:
                for right in right_positions:
                    position_span = abs(int(right) - int(left))
                    cyclic_span = min(
                        position_span,
                        max(0, len(chain) - position_span),
                    )
                    forward = abs(float(cumulative[int(right)] - cumulative[int(left)]))
                    arc = min(forward, max(0.0, total - forward))
                    evidence = {
                        "chain_index": int(chain_index),
                        "left_position": int(left),
                        "right_position": int(right),
                        "cyclic_position_span": int(cyclic_span),
                        "source_arc_span_m": float(arc),
                        "chord_length_m": float(chord),
                        "target_spacing_m": float(h),
                        "arc_chord_ratio": float(arc / max(chord, 1.0e-12)),
                        "arc_target_ratio": float(arc / max(h, 1.0e-12)),
                    }
                    if best is None or (
                        float(evidence["arc_chord_ratio"]),
                        float(evidence["arc_target_ratio"]),
                        int(evidence["cyclic_position_span"]),
                    ) > (
                        float(best["arc_chord_ratio"]),
                        float(best["arc_target_ratio"]),
                        int(best["cyclic_position_span"]),
                    ):
                        best = evidence
        if best is None:
            best = {
                "chain_index": None,
                "left_position": None,
                "right_position": None,
                "cyclic_position_span": None,
                "source_arc_span_m": None,
                "chord_length_m": float(chord),
                "target_spacing_m": float(h),
                "arc_chord_ratio": None,
                "arc_target_ratio": None,
            }
        adjacent = _edge_key(a, b) in self.protected_edges
        best["protected"] = bool(adjacent)
        best["lineage_edge"] = list(self.lineage_edge((a, b)))
        best["is_shortcut"] = bool(
            not adjacent
            and best["chain_index"] is not None
            and int(best["cyclic_position_span"] or 0) > 1
            and float(best["arc_chord_ratio"] or 0.0)
            >= float(self.config.shortcut_arc_chord_ratio)
            and float(best["arc_target_ratio"] or 0.0)
            >= float(self.config.shortcut_arc_target_ratio)
        )
        return best

    def candidate_records(
        self,
        triangles: np.ndarray,
        component_triangle_indices: Iterable[int],
        superthin_mask: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Rank causal unprotected edges touching one superthin component."""
        triangles = np.asarray(triangles, dtype=int)
        superthin_mask = np.asarray(superthin_mask, dtype=bool)
        component = sorted(set(map(int, component_triangle_indices)))
        edge_to_component_triangles: dict[tuple[int, int], set[int]] = {}
        for triangle_index in component:
            triangle = triangles[int(triangle_index)]
            for edge in _triangle_edges(triangle):
                if edge in self.protected_edges:
                    continue
                edge_to_component_triangles.setdefault(edge, set()).add(
                    int(triangle_index)
                )
        records: list[dict[str, Any]] = []
        for edge, attached_component in edge_to_component_triangles.items():
            attached_global = np.where(
                np.count_nonzero(
                    np.isin(triangles, np.asarray(edge, dtype=int)),
                    axis=1,
                )
                == 2
            )[0]
            attached_superthin = [
                int(index)
                for index in attached_global
                if bool(superthin_mask[int(index)])
            ]
            a, b = edge
            length = float(np.linalg.norm(self.points[b] - self.points[a]))
            target = _harmonic_target(
                float(self.targets[a]),
                float(self.targets[b]),
            )
            shortcut = self.same_chain_shortcut_evidence(edge)
            l_over_h = float(length / max(target, 1.0e-12))
            size_extremeness = max(
                l_over_h,
                1.0 / max(l_over_h, 1.0e-12),
            )
            priority = (
                0 if bool(shortcut["is_shortcut"]) else 1,
                -int(len(attached_superthin)),
                -float(size_extremeness),
                self.lineage_edge(edge),
            )
            records.append(
                {
                    "edge": list(map(int, edge)),
                    "lineage_edge": list(self.lineage_edge(edge)),
                    "component_triangle_indices": sorted(attached_component),
                    "attached_triangle_indices": list(
                        map(int, attached_global.tolist())
                    ),
                    "attached_superthin_triangle_indices": attached_superthin,
                    "length_m": float(length),
                    "target_spacing_m": float(target),
                    "l_over_h": float(l_over_h),
                    "same_chain_shortcut": shortcut,
                    "_priority": priority,
                }
            )
        records.sort(key=lambda record: record["_priority"])
        limit = max(0, int(self.config.maximum_candidates_per_component))
        output = records[:limit]
        for rank, record in enumerate(output, start=1):
            record.pop("_priority", None)
            record["candidate_rank"] = int(rank)
        return output

    def restricted_violation_records(
        self,
        triangles: np.ndarray,
    ) -> list[dict[str, Any]]:
        return restricted_edge_violation_records(
            triangles,
            self.lineage,
            self.restricted_lineage_edges,
        )


def restricted_edge_violation_records(
    triangles: np.ndarray,
    node_lineage: np.ndarray,
    restricted_lineage_edges: Iterable[Iterable[int]],
    *,
    edge_to_triangles: Mapping[tuple[int, int], Iterable[int]] | None = None,
) -> list[dict[str, Any]]:
    """Audit forbidden source-lineage edges in one delivered mesh."""
    triangles = np.asarray(triangles, dtype=int)
    lineage = np.asarray(node_lineage, dtype=int)
    restricted = {
        _edge_key(int(values[0]), int(values[1]))
        for values in restricted_lineage_edges
        if len(values) >= 2 and int(values[0]) != int(values[1])
    }
    if not restricted:
        return []
    if edge_to_triangles is None:
        attached: dict[tuple[int, int], list[int]] = {}
        for triangle_index, triangle in enumerate(triangles):
            for edge in _triangle_edges(triangle):
                attached.setdefault(edge, []).append(int(triangle_index))
    else:
        attached = {
            _edge_key(*map(int, edge)): list(map(int, triangle_indices))
            for edge, triangle_indices in edge_to_triangles.items()
        }
    records: list[dict[str, Any]] = []
    for edge in sorted(attached):
        a, b = edge
        lineage_edge = _edge_key(int(lineage[a]), int(lineage[b]))
        if lineage_edge not in restricted:
            continue
        records.append(
            {
                "edge": list(map(int, edge)),
                "lineage_edge": list(map(int, lineage_edge)),
                "attached_triangle_indices": list(
                    map(int, attached[edge])
                ),
            }
        )
    return records


def audit_superthin_connectivity(
    points: np.ndarray,
    triangles: np.ndarray,
    target_spacing_m: np.ndarray,
    constraint_chains: Iterable[Iterable[int]],
    *,
    node_lineage: np.ndarray | None = None,
    restricted_lineage_edges: Iterable[Iterable[int]] = (),
    quality_threshold: float = 0.10,
    minimum_angle_deg: float = 5.0,
    config: ConnectivityRestrictionConfig | None = None,
) -> dict[str, Any]:
    """Inventory components and ranked causal edges without changing a mesh."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    lineage = np.asarray(
        node_lineage
        if node_lineage is not None
        else np.arange(len(points), dtype=int),
        dtype=int,
    )
    policy = AllowedEdgePolicy(
        points,
        target_spacing_m,
        constraint_chains,
        lineage,
        restricted_lineage_edges=restricted_lineage_edges,
        config=config,
    )
    geometry = triangle_geometry(points, triangles)
    minimum_angles = np.min(geometry["angles_deg"], axis=1)
    mask = (
        geometry["quality"] < float(quality_threshold)
    ) | (
        minimum_angles < float(minimum_angle_deg)
    )
    components = _triangle_components(
        triangles,
        np.where(mask)[0],
    )
    records: list[dict[str, Any]] = []
    for component_index, component in enumerate(components, start=1):
        candidates = policy.candidate_records(
            triangles,
            component,
            mask,
        )
        records.append(
            {
                "component_index": int(component_index),
                "triangle_indices": list(map(int, component)),
                "triangle_ids_1based": [
                    int(value) + 1 for value in component
                ],
                "minimum_quality": float(
                    np.min(
                        geometry["quality"][
                            np.asarray(component, dtype=int)
                        ]
                    )
                ),
                "minimum_angle_deg": float(
                    np.min(
                        minimum_angles[
                            np.asarray(component, dtype=int)
                        ]
                    )
                ),
                "candidate_edges": candidates,
            }
        )
    violations = policy.restricted_violation_records(triangles)
    return {
        "schema_version": "fvcom_superthin_connectivity_restriction_v1",
        "mode": "audit",
        "superthin_quality_threshold": float(quality_threshold),
        "superthin_minimum_angle_deg": float(minimum_angle_deg),
        "superthin_triangle_count": int(np.count_nonzero(mask)),
        "superthin_component_count": int(len(components)),
        "components": records,
        "restricted_lineage_edges": [
            list(map(int, edge))
            for edge in sorted(policy.restricted_lineage_edges)
        ],
        "restricted_edge_violations": violations,
        "restricted_edge_violation_count": int(len(violations)),
    }


def _triangle_components(
    triangles: np.ndarray,
    selected: Iterable[int],
) -> list[list[int]]:
    selected_set = set(map(int, selected))
    edge_to_selected: dict[tuple[int, int], list[int]] = {}
    for triangle_index in sorted(selected_set):
        for edge in _triangle_edges(triangles[int(triangle_index)]):
            edge_to_selected.setdefault(edge, []).append(
                int(triangle_index)
            )
    adjacency = {
        index: set()
        for index in selected_set
    }
    for attached in edge_to_selected.values():
        for left in attached:
            adjacency[left].update(
                value for value in attached if value != left
            )
    remaining = set(selected_set)
    components: list[list[int]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack = [seed]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(int(current))
            following = adjacency[current] & remaining
            remaining.difference_update(following)
            stack.extend(sorted(following, reverse=True))
        components.append(sorted(component))
    return components


def _harmonic_target(left: float, right: float) -> float:
    left = max(float(left), 1.0e-12)
    right = max(float(right), 1.0e-12)
    return float(2.0 / (1.0 / left + 1.0 / right))


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return tuple(sorted((int(a), int(b))))


def _triangle_edges(triangle: np.ndarray) -> tuple[tuple[int, int], ...]:
    a, b, c = map(int, triangle)
    return (_edge_key(a, b), _edge_key(b, c), _edge_key(c, a))


def _mesh_edges(triangles: np.ndarray) -> set[tuple[int, int]]:
    return {
        edge
        for triangle in np.asarray(triangles, dtype=int)
        for edge in _triangle_edges(triangle)
    }
