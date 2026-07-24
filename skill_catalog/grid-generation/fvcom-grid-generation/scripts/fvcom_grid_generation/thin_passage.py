"""Human-guided removal of an artificial, under-resolved wet passage.

This research-only operator deliberately changes boundary topology.  It is
kept separate from the normal conditioning profiles, which preserve passage
connectivity and all existing boundary nodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .local_topology import (
    AggressiveConditioningConfig,
    _State,
    _boundary_graph_audit,
    _compact,
    _expand_triangle_patch,
    _inventory_superthin_components,
    _restricted_edge_violation_records,
    _summary,
)
from .metrics import build_edge_topology, chain_edges, constraint_integrity, triangle_geometry
from .postprocess import boundary_chains_from_mesh


REPORT_SCHEMA = "fvcom_thin_passage_removal_v1"


@dataclass(frozen=True)
class ThinPassageRemovalConfig:
    patch_rings: int = 4
    overresolved_edge_fraction: float = 0.55
    maximum_inferred_bank_nodes: int = 8
    maximum_removed_triangles: int = 128
    maximum_dangling_peel_rounds: int = 8
    quality_tolerance: float = 1.0e-9


def infer_passage_removal_candidates(
    state: _State,
    component: dict[str, Any],
    *,
    config: ThinPassageRemovalConfig | None = None,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Infer source-lineage node sets from a causal component and bank spacing.

    The learned pattern is a normally spaced bracket on each bank surrounding
    a short run of over-resolved boundary edges.  Internal vertices of that run
    are removed together with a movable component apex.  Returned values are
    source lineages, so the plan remains stable after earlier compaction.
    """
    config = config or ThinPassageRemovalConfig()
    topology = build_edge_topology(len(state.points), state.triangles)
    patch = _expand_triangle_patch(
        state.triangles,
        topology,
        component["triangle_indices"],
        int(config.patch_rings),
    )
    patch_nodes = set(map(int, np.unique(state.triangles[patch])))
    component_nodes = set(
        map(
            int,
            np.unique(
                state.triangles[
                    np.asarray(component["triangle_indices"], dtype=int)
                ]
            ),
        )
    )
    component_chain_ids = sorted(
        {
            chain_index
            for chain_index, chain in enumerate(state.chains)
            if component_nodes & set(map(int, chain))
        }
    )
    if not component_chain_ids:
        raise ValueError("passage inference requires a component touching a boundary chain")

    movable_component_nodes = sorted(
        node for node in component_nodes if not bool(state.fixed[node])
    )
    focus_nodes = movable_component_nodes or sorted(component_nodes)
    focus = np.mean(state.points[np.asarray(focus_nodes, dtype=int)], axis=0)

    opposing: list[tuple[float, int]] = []
    for chain_index, chain in enumerate(state.chains):
        if chain_index in component_chain_ids or not (patch_nodes & set(map(int, chain))):
            continue
        distance = min(
            _point_segment_distance(
                focus,
                state.points[int(left)],
                state.points[int(right)],
            )
            for left, right in zip(chain, [*chain[1:], chain[0]])
        )
        opposing.append((float(distance), int(chain_index)))
    if not opposing:
        raise ValueError("no opposing passage bank occurs inside the selected patch")
    opposing_distance, opposing_chain_id = min(opposing)
    chain = list(map(int, state.chains[opposing_chain_id]))
    if len(chain) < 4:
        raise ValueError("opposing bank is too short to define retained brackets")

    ratios = []
    distances = []
    for left, right in zip(chain, [*chain[1:], chain[0]]):
        length = float(np.linalg.norm(state.points[left] - state.points[right]))
        target = _harmonic_target(state.targets[left], state.targets[right])
        ratios.append(length / max(target, 1.0e-12))
        distances.append(
            _point_segment_distance(focus, state.points[left], state.points[right])
        )
    low = np.asarray(ratios, dtype=float) < float(config.overresolved_edge_fraction)
    low_indices = np.where(low)[0]
    if not len(low_indices):
        raise ValueError("opposing bank has no bounded over-resolved edge run")
    seed_edge = int(min(low_indices, key=lambda value: (distances[int(value)], int(value))))
    run_edges = _cyclic_true_run(low, seed_edge)
    run_nodes = [chain[run_edges[0]]]
    run_nodes.extend(chain[(edge + 1) % len(chain)] for edge in run_edges)
    internal_nodes = run_nodes[1:-1]
    if not internal_nodes:
        raise ValueError("over-resolved bank run has no removable internal node")
    if len(internal_nodes) > int(config.maximum_inferred_bank_nodes):
        raise ValueError("inferred bank-removal window exceeds its node budget")
    if any(state.hard[node] for node in internal_nodes):
        raise ValueError("inferred bank-removal window crosses a hard anchor")
    if any(node in set(map(int, state.open_nodes)) for node in internal_nodes):
        raise ValueError("automatic passage removal may not alter the OBC")

    bank_lineages = [int(state.lineage[node]) for node in internal_nodes]
    movable_lineages = [int(state.lineage[node]) for node in movable_component_nodes]
    primary = sorted(set(bank_lineages + movable_lineages))
    candidates = [primary]
    if bank_lineages and sorted(bank_lineages) != primary:
        candidates.append(sorted(set(bank_lineages)))
    evidence = {
        "component_chain_ids": component_chain_ids,
        "opposing_chain_id": int(opposing_chain_id),
        "opposing_bank_distance_m": float(opposing_distance),
        "retained_bracket_lineages": [
            int(state.lineage[run_nodes[0]]),
            int(state.lineage[run_nodes[-1]]),
        ],
        "retained_bracket_node_ids_1based_source": [
            int(state.lineage[run_nodes[0]]) + 1,
            int(state.lineage[run_nodes[-1]]) + 1,
        ],
        "overresolved_run_edge_l_over_h": [float(ratios[index]) for index in run_edges],
        "inferred_bank_node_lineages": bank_lineages,
        "inferred_bank_node_ids_1based_source": [value + 1 for value in bank_lineages],
        "movable_component_lineages": movable_lineages,
        "movable_component_node_ids_1based_source": [value + 1 for value in movable_lineages],
        "resolution_match_test": {
            "mode": "diagnostic_only_no_insertion",
            "passed_as_physical_passage": False,
            "reason": "opposing bank contains a target-relative over-resolution run at the causal apex",
            "overresolved_edge_fraction": float(config.overresolved_edge_fraction),
        },
    }
    return candidates, evidence


def try_remove_thin_passage(
    state: _State,
    component_id: str,
    candidate_lineage_sets: list[list[int]],
    *,
    expected_boundary_component_delta: int | None,
    expected_wet_component_delta: int | None,
    human_approved: bool,
    allow_authorized_topology_delta: bool = False,
    inference_evidence: dict[str, Any] | None = None,
    config: ThinPassageRemovalConfig | None = None,
) -> tuple[_State, dict[str, Any]]:
    """Test bounded node-star deletions and commit the first passing cut."""
    config = config or ThinPassageRemovalConfig()
    conditioning = AggressiveConditioningConfig(
        thin_repair_profile="systematic-v5",
        enable_pruning=False,
        enable_valence_repair=False,
    )
    components = _inventory_superthin_components(state, conditioning)
    matching = [item for item in components if str(item["component_id"]) == str(component_id)]
    if len(matching) != 1:
        raise ValueError(f"passage component is absent or stale: {component_id}")
    selected = matching[0]
    before = _summary(state, conditioning)
    attempts: list[dict[str, Any]] = []
    accepted_state: _State | None = None
    accepted_record: dict[str, Any] | None = None

    for index, requested_lineages in enumerate(candidate_lineage_sets):
        trial = state.clone()
        record: dict[str, Any] = {
            "candidate_index": int(index),
            "requested_node_lineages": sorted(set(map(int, requested_lineages))),
            "requested_node_ids_1based_source": [
                value + 1 for value in sorted(set(map(int, requested_lineages)))
            ],
        }
        try:
            edit = _delete_node_stars(
                trial,
                selected,
                requested_lineages,
                config=config,
            )
            after = _summary(trial, conditioning)
            audit = _passage_audit(
                state,
                trial,
                before,
                after,
                expected_boundary_component_delta=expected_boundary_component_delta,
                expected_wet_component_delta=expected_wet_component_delta,
                allow_authorized_topology_delta=bool(
                    allow_authorized_topology_delta
                ),
                quality_tolerance=float(config.quality_tolerance),
            )
            record.update({"edit": edit, "after": after, "audit": audit})
            record["failures"] = list(audit["failures"])
        except (ValueError, RuntimeError) as exc:
            record["failures"] = [str(exc)]
        if not record["failures"]:
            replay = state.clone()
            try:
                _delete_node_stars(
                    replay,
                    selected,
                    requested_lineages,
                    config=config,
                )
                replay_after = _summary(replay, conditioning)
                replay_audit = _passage_audit(
                    state,
                    replay,
                    before,
                    replay_after,
                    expected_boundary_component_delta=(
                        expected_boundary_component_delta
                    ),
                    expected_wet_component_delta=expected_wet_component_delta,
                    allow_authorized_topology_delta=bool(
                        allow_authorized_topology_delta
                    ),
                    quality_tolerance=float(config.quality_tolerance),
                )
                replay_matches = bool(
                    np.array_equal(replay.points, trial.points)
                    and np.array_equal(replay.triangles, trial.triangles)
                    and np.array_equal(replay.lineage, trial.lineage)
                    and replay.chains == trial.chains
                    and np.array_equal(replay.open_nodes, trial.open_nodes)
                )
                record["deterministic_replay"] = {
                    "passed": bool(replay_matches and replay_audit["passed"]),
                    "state_matches": replay_matches,
                    "audit": replay_audit,
                }
                if not record["deterministic_replay"]["passed"]:
                    record["failures"].append(
                        "deterministic_passage_replay_mismatch"
                    )
            except (ValueError, RuntimeError) as exc:
                record["deterministic_replay"] = {
                    "passed": False,
                    "failure": str(exc),
                }
                record["failures"].append(
                    "deterministic_passage_replay_failed"
                )
        attempts.append(record)
        if not record["failures"]:
            accepted_state = trial
            accepted_record = record
            break

    accepted = accepted_state is not None
    delivered = accepted_state if accepted else state
    after = _summary(delivered, conditioning)
    if accepted_record is not None:
        accepted_record["accepted"] = True
        delivered.ledger.append(
            {
                "operation": "human-guided-whole-thin-passage-removal",
                "component_id": str(component_id),
                "human_approved": bool(human_approved),
                "removed_node_lineages": list(accepted_record["edit"]["removed_node_lineages"]),
                "removed_triangle_count": int(accepted_record["edit"]["removed_triangle_count"]),
                "removed_area_m2": float(accepted_record["edit"]["removed_area_m2"]),
                "superthin_before": int(before["superthin_triangle_count"]),
                "superthin_after": int(after["superthin_triangle_count"]),
                "boundary_component_delta": int(
                    after["boundary_component_count"] - before["boundary_component_count"]
                ),
            }
        )
    report = {
        "schema_version": REPORT_SCHEMA,
        "component_id": str(component_id),
        "accepted": bool(accepted),
        "status": "passage_removed" if accepted else "passage_removal_infeasible",
        "human_approved": bool(human_approved),
        "allow_authorized_topology_delta": bool(
            allow_authorized_topology_delta
        ),
        "expected_boundary_component_delta": expected_boundary_component_delta,
        "expected_wet_component_delta": expected_wet_component_delta,
        "before": before,
        "after": after,
        "inference": inference_evidence or {},
        "attempts": attempts,
        "selected_candidate_index": (
            int(accepted_record["candidate_index"]) if accepted_record is not None else None
        ),
    }
    return delivered, report


def _delete_node_stars(
    state: _State,
    selected_component: dict[str, Any],
    requested_lineages: list[int],
    *,
    config: ThinPassageRemovalConfig,
) -> dict[str, Any]:
    requested = sorted(set(map(int, requested_lineages)))
    lineage_to_node = {int(value): int(index) for index, value in enumerate(state.lineage)}
    missing = [value for value in requested if value not in lineage_to_node]
    if missing:
        raise ValueError(f"requested passage nodes are stale or absent: {missing}")
    nodes = sorted(lineage_to_node[value] for value in requested)
    if any(bool(state.hard[node]) for node in nodes):
        raise ValueError("passage removal may not delete a hard anchor")
    open_set = set(map(int, state.open_nodes))
    if any(node in open_set for node in nodes):
        raise ValueError("passage removal may not delete an OBC node")

    node_mask = np.isin(state.triangles, np.asarray(nodes, dtype=int))
    incident = np.where(np.any(node_mask, axis=1))[0]
    selected_indices = set(map(int, selected_component["triangle_indices"]))
    if not selected_indices.issubset(set(map(int, incident))):
        raise ValueError("candidate does not remove the complete selected superthin component")
    if not len(incident):
        raise ValueError("candidate has no incident triangle star")
    if len(incident) > int(config.maximum_removed_triangles):
        raise ValueError("candidate exceeds the removed-triangle budget")

    geometry = triangle_geometry(state.points, state.triangles)
    topology_before = build_edge_topology(len(state.points), state.triangles)
    baseline_singly = {
        tuple(sorted(int(state.lineage[node]) for node in state.triangles[index]))
        for index in np.where(topology_before.triangle_neighbor_count <= 1)[0]
    }
    removed_area = float(np.sum(geometry["area"][incident]))
    removed_triangle_lineages = [
        sorted(int(state.lineage[node]) for node in state.triangles[index])
        for index in incident
    ]
    keep = np.ones(len(state.triangles), dtype=bool)
    keep[incident] = False
    state.triangles = state.triangles[keep]
    peeled = _peel_new_dangling_triangles(
        state,
        baseline_lineage_triangles=baseline_singly,
        maximum_rounds=int(config.maximum_dangling_peel_rounds),
    )
    if len(peeled):
        removed_area += float(
            np.sum(triangle_geometry(state.points, peeled)["area"])
        )

    precompact_topology = build_edge_topology(len(state.points), state.triangles)
    boundary = _boundary_graph_audit(precompact_topology)
    if int(boundary["degree_anomaly_count"]) != 0:
        raise ValueError("passage cut creates a non-traversable boundary graph")
    chains = boundary_chains_from_mesh(state.triangles + 1)
    if not chains:
        raise ValueError("passage cut has no valid delivered boundary loop")
    state.chains = [list(map(int, chain)) for chain in chains]
    delivered_boundary = {int(node) for chain in state.chains for node in chain}
    state.fixed = np.asarray(
        [index in delivered_boundary for index in range(len(state.points))],
        dtype=bool,
    )
    for node in delivered_boundary:
        if str(state.kinds[node]) == "interior":
            state.kinds[node] = "passage_cut"
    state.last_affected = sorted(delivered_boundary & set(map(int, np.unique(state.triangles))))
    _compact(state)
    if any(value in set(map(int, state.lineage)) for value in requested):
        raise RuntimeError("requested passage node survived node-star deletion")
    return {
        "removed_node_lineages": requested,
        "removed_node_ids_1based_source": [value + 1 for value in requested],
        "removed_triangle_count": int(len(incident) + len(peeled)),
        "removed_incident_triangle_count": int(len(incident)),
        "peeled_dangling_triangle_count": int(len(peeled)),
        "removed_area_m2": float(removed_area),
        "removed_triangle_lineages": removed_triangle_lineages,
        "delivered_boundary_chain_count": int(len(state.chains)),
    }


def _peel_new_dangling_triangles(
    state: _State,
    *,
    baseline_lineage_triangles: set[tuple[int, int, int]],
    maximum_rounds: int,
) -> np.ndarray:
    removed: list[np.ndarray] = []
    for _ in range(max(0, int(maximum_rounds))):
        topology = build_edge_topology(len(state.points), state.triangles)
        dangling = [
            int(index)
            for index in np.where(topology.triangle_neighbor_count <= 1)[0]
            if tuple(
                sorted(
                    int(state.lineage[node])
                    for node in state.triangles[int(index)]
                )
            )
            not in baseline_lineage_triangles
        ]
        if not dangling:
            break
        values = state.triangles[np.asarray(dangling, dtype=int)].copy()
        removed.extend(values)
        keep = np.ones(len(state.triangles), dtype=bool)
        keep[np.asarray(dangling, dtype=int)] = False
        state.triangles = state.triangles[keep]
    return np.asarray(removed, dtype=int).reshape((-1, 3)) if removed else np.empty((0, 3), dtype=int)


def _passage_audit(
    before_state: _State,
    after_state: _State,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    expected_boundary_component_delta: int | None,
    expected_wet_component_delta: int | None,
    allow_authorized_topology_delta: bool,
    quality_tolerance: float,
) -> dict[str, Any]:
    topology = build_edge_topology(len(after_state.points), after_state.triangles)
    geometry = triangle_geometry(after_state.points, after_state.triangles)
    boundary = _boundary_graph_audit(topology)
    integrity = constraint_integrity(
        topology,
        after_state.chains,
        after_state.open_nodes.tolist(),
    )
    lineage_to_node = {
        int(value): int(index) for index, value in enumerate(after_state.lineage)
    }
    missing_hard = [
        int(value)
        for value in before_state.source_hard_anchor_lineage
        if int(value) not in lineage_to_node
    ]
    moved_hard = [
        int(value)
        for value in before_state.source_hard_anchor_lineage
        if int(value) in lineage_to_node
        and not np.array_equal(
            after_state.points[lineage_to_node[int(value)]],
            before_state.source_points[int(value)],
        )
    ]
    canonical = [tuple(sorted(map(int, values))) for values in after_state.triangles]
    before_open_lineage = [
        int(before_state.lineage[int(node)])
        for node in np.asarray(before_state.open_nodes, dtype=int)
    ]
    after_open_lineage = [
        int(after_state.lineage[int(node)])
        for node in np.asarray(after_state.open_nodes, dtype=int)
    ]
    failures: list[str] = []
    gates = {
        "positive_signed_areas": bool(len(geometry["signed_area"]) and np.all(geometry["signed_area"] > 0.0)),
        "manifold_edges": bool(not topology.nonmanifold_edges),
        "boundary_degree_two": bool(int(boundary["degree_anomaly_count"]) == 0),
        "all_delivered_boundary_edges_present": bool(integrity["all_protected_edges_present"]),
        "open_boundary_ordered": bool(integrity["open_boundary_ordered"]),
        "open_boundary_lineage_unchanged": bool(
            after_open_lineage == before_open_lineage
        ),
        "hard_anchors_preserved": bool(not missing_hard and not moved_hard),
        "no_unused_nodes": bool(len(np.unique(after_state.triangles)) == len(after_state.points)),
        "no_duplicate_triangles": bool(len(canonical) == len(set(canonical))),
        "no_repeated_node_triangles": bool(all(len(set(values)) == 3 for values in canonical)),
        "restricted_edges_absent": bool(not _restricted_edge_violation_records(after_state, topology=topology)),
        "superthin_strictly_reduced": bool(after["superthin_triangle_count"] < before["superthin_triangle_count"]),
        "superthin_severity_nonincreasing": bool(after["superthin_severity_sum"] <= before["superthin_severity_sum"] + quality_tolerance),
        "q_min_nonregression": bool(after["q_min"] + quality_tolerance >= before["q_min"]),
        "q_p01_nonregression": bool(after["q_p01"] + quality_tolerance >= before["q_p01"]),
        "minimum_angle_nonregression": bool(after["minimum_angle_deg"] + 1.0e-8 >= before["minimum_angle_deg"]),
        "l_over_h_tail_nonincreasing": bool(after["l_over_h_count_above_1_55"] <= before["l_over_h_count_above_1_55"]),
        "area_transition_nonincreasing": bool(after["area_transition_count_above_0_50"] <= before["area_transition_count_above_0_50"]),
    }
    if bool(allow_authorized_topology_delta):
        gates["authorized_boundary_component_delta"] = True
    elif expected_boundary_component_delta is not None:
        gates["intended_boundary_component_delta"] = bool(
            int(after["boundary_component_count"] - before["boundary_component_count"])
            == int(expected_boundary_component_delta)
        )
    if bool(allow_authorized_topology_delta):
        gates["authorized_wet_component_delta"] = True
    elif expected_wet_component_delta is not None:
        gates["intended_wet_component_delta"] = bool(
            int(after["connected_component_count"] - before["connected_component_count"])
            == int(expected_wet_component_delta)
        )
    else:
        gates["wet_component_count_preserved"] = bool(
            after["connected_component_count"] == before["connected_component_count"]
        )
    failures.extend(name for name, passed in gates.items() if not passed)
    return {
        "passed": bool(not failures),
        "failures": failures,
        "gates": gates,
        "boundary_component_delta": int(after["boundary_component_count"] - before["boundary_component_count"]),
        "wet_component_delta": int(after["connected_component_count"] - before["connected_component_count"]),
        "missing_hard_anchor_lineages": missing_hard,
        "moved_hard_anchor_lineages": moved_hard,
        "obc_node_count": int(len(after_state.open_nodes)),
        "obc_source_lineage_before": before_open_lineage,
        "obc_source_lineage_after": after_open_lineage,
        "restricted_edge_violations": _restricted_edge_violation_records(after_state, topology=topology),
    }


def _cyclic_true_run(values: np.ndarray, seed: int) -> list[int]:
    flags = np.asarray(values, dtype=bool)
    if not bool(flags[int(seed)]):
        raise ValueError("run seed must select a true edge")
    count = len(flags)
    run = [int(seed)]
    cursor = (int(seed) - 1) % count
    while bool(flags[cursor]) and cursor not in run:
        run.insert(0, int(cursor))
        cursor = (cursor - 1) % count
    cursor = (int(seed) + 1) % count
    while bool(flags[cursor]) and cursor not in run:
        run.append(int(cursor))
        cursor = (cursor + 1) % count
    return run


def _point_segment_distance(point: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    direction = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    denominator = float(np.dot(direction, direction))
    if denominator <= 1.0e-30:
        return float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(left, dtype=float)))
    fraction = float(np.dot(np.asarray(point, dtype=float) - np.asarray(left, dtype=float), direction) / denominator)
    projected = np.asarray(left, dtype=float) + np.clip(fraction, 0.0, 1.0) * direction
    return float(np.linalg.norm(np.asarray(point, dtype=float) - projected))


def _harmonic_target(first: float, second: float) -> float:
    values = np.asarray([first, second], dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if not len(values):
        return 1.0
    if len(values) == 1:
        return float(values[0])
    return float(2.0 / np.sum(1.0 / values))


def config_as_dict(config: ThinPassageRemovalConfig) -> dict[str, Any]:
    return asdict(config)
