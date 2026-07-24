#!/usr/bin/env python3
"""Focused deterministic tests for V5 inward-front support and escrow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from fvcom_grid_generation import local_topology as topology  # noqa: E402
from fvcom_grid_generation.metrics import (  # noqa: E402
    build_edge_topology,
    chain_edges,
)
from fvcom_grid_generation.visual_superthin import (  # noqa: E402
    create_visual_state,
)


def _boundary_star():
    points = np.asarray(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
            [2.0, 0.15],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=int,
    )
    fixed = np.asarray([True, True, True, True, False], dtype=bool)
    hard = np.asarray([True, True, True, True, False], dtype=bool)
    state, config, components = create_visual_state(
        points,
        triangles,
        fixed,
        [[0, 1, 2, 3]],
        np.asarray([0, 1], dtype=int),
        target_spacing_m=np.full(len(points), 2.0),
        boundary_kinds=["open", "open", "land", "land", "interior"],
        hard_anchor_mask=hard,
    )
    config = replace(
        config,
        systematic_v5_enable_boundary_window_fallback=False,
        systematic_v5_max_inward_front_support_points=4,
        systematic_v5_max_lawson_flips_per_transaction=0,
        micro_relax_cycles=0,
    )
    incident = np.where(np.any(state.triangles == 4, axis=1))[0]
    ring = topology._ordered_one_ring(state.triangles[incident], 4)
    assert ring is not None
    return state, config, components, ring


def _front_modes(state, config, ring):
    return [
        value
        for value in topology._locked_star_modes(
            state,
            4,
            ring,
            0,
            config,
        )
        if value["name"] == "inward-front-multi-support"
        and not value.get("generation_failures")
    ]


def test_inward_front_groups_are_bounded_strictly_inside_and_deterministic() -> None:
    state, config, _, ring = _boundary_star()
    first = _front_modes(state, config, ring)
    second = _front_modes(state, config, ring)
    assert [
        int(value["requested_support_node_count"]) for value in first
    ] == [2, 3, 4]
    assert len(first) == len(second)
    polygon = Polygon(state.points[np.asarray(ring, dtype=int)])
    for left, right in zip(first, second):
        assert np.array_equal(left["coordinates"], right["coordinates"])
        coordinates = np.asarray(left["coordinates"], dtype=float)
        assert np.all(
            contains_xy(
                polygon,
                coordinates[:, 0],
                coordinates[:, 1],
            )
        )
        evidence = left["support_generation_evidence"]
        assert evidence["strictly_inside_locked_ring"] is True
        assert evidence["global_delaunay_used"] is False
        assert evidence["base_is_open_boundary_edge"] is True

    capped = _front_modes(
        state,
        replace(
            config,
            systematic_v5_max_inward_front_support_points=99,
        ),
        ring,
    )
    assert max(
        int(value["requested_support_node_count"]) for value in capped
    ) == 8


def _apply_three_support_candidate():
    state, config, components, ring = _boundary_star()
    mode = next(
        value
        for value in _front_modes(state, config, ring)
        if int(value["requested_support_node_count"]) == 3
    )
    original_delaunay = topology.Delaunay

    def forbidden_delaunay(*_args, **_kwargs):
        raise AssertionError("the inward-front route invoked Delaunay")

    topology.Delaunay = forbidden_delaunay
    try:
        changed, failures, evidence = (
            topology._reconstruct_locked_star_candidate(
                state,
                center=4,
                triangle_index=0,
                mode=mode,
                config=config,
            )
        )
    finally:
        topology.Delaunay = original_delaunay
    assert changed, failures
    return state, config, components, evidence


def test_inward_front_uses_three_supports_and_preserves_hard_contract() -> None:
    first, config, components, evidence = _apply_three_support_candidate()
    second, _, _, second_evidence = _apply_three_support_candidate()
    assert evidence["inserted_support_node_count"] == 3
    assert len(evidence["support_coordinates_xy"]) == 3
    assert evidence["global_delaunay_used"] is False
    assert evidence["support_triangulation_method"] == (
        "deterministic-ear-plus-point-insertion"
    )
    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.triangles, second.triangles)
    assert evidence["support_coordinates_xy"] == (
        second_evidence["support_coordinates_xy"]
    )

    lineage_to_node = {
        int(value): int(index)
        for index, value in enumerate(first.lineage)
    }
    assert len(first.chains[0]) == 4
    assert first.open_nodes.tolist() == [0, 1]
    assert [
        int(first.lineage[int(node)]) for node in first.open_nodes
    ] == [0, 1]
    for lineage in range(4):
        node = lineage_to_node[lineage]
        assert np.array_equal(
            first.points[node],
            first.source_points[lineage],
        )
        assert bool(first.fixed[node])
        assert bool(first.hard[node])
    created = np.where(first.lineage < 0)[0]
    assert len(created) == 3
    assert not np.any(first.fixed[created])
    assert not np.any(first.hard[created])
    assert all(first.kinds[int(node)] == "interior" for node in created)

    delivered = build_edge_topology(len(first.points), first.triangles)
    assert chain_edges(first.chains).issubset(
        set(delivered.edge_to_triangles)
    )
    assert not delivered.nonmanifold_edges
    geometry = topology.triangle_geometry(
        first.points,
        first.triangles,
    )
    assert np.all(geometry["signed_area"] > 0.0)
    ok, invariants, _ = topology._audit_state(
        first,
        config,
        components,
    )
    assert ok, invariants
    assert invariants["open_boundary_membership_unchanged"] is True
    assert invariants["restricted_edge_violation_count"] == 0

    original, _, _, ring = _boundary_star()
    bad_mode = next(
        value
        for value in _front_modes(original, config, ring)
        if int(value["requested_support_node_count"]) == 3
    )
    bad_mode = dict(bad_mode)
    bad_coordinates = np.asarray(
        bad_mode["coordinates"],
        dtype=float,
    ).copy()
    bad_coordinates[0] = np.asarray([20.0, 20.0])
    bad_mode["coordinates"] = bad_coordinates
    changed, failures, failure_evidence = (
        topology._reconstruct_locked_star_candidate(
            original,
            center=4,
            triangle_index=0,
            mode=bad_mode,
            config=config,
        )
    )
    assert not changed
    assert "inward_front_support_outside_locked_ring" in failures
    assert failure_evidence["requested_support_node_count"] == 3
    assert len(failure_evidence["support_coordinates_xy"]) == 3


def _valence_fan():
    count = 10
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    inner = np.column_stack([np.cos(angles), np.sin(angles)])
    outer = 2.0 * inner
    points = np.vstack(
        [inner, outer, np.asarray([[0.0, 0.0]])]
    )
    center = 2 * count
    center_fan = [
        [index, (index + 1) % count, center]
        for index in range(count)
    ]
    outer_strip: list[list[int]] = []
    for index in range(count):
        following = (index + 1) % count
        outer_strip.extend(
            [
                [index, count + index, count + following],
                [index, count + following, following],
            ]
        )
    triangles = np.asarray([*center_fan, *outer_strip], dtype=int)
    fixed = np.asarray(
        [False] * count + [True] * count + [False],
        dtype=bool,
    )
    hard = np.asarray(
        [False] * count + [True] * count + [False],
        dtype=bool,
    )
    state, config, components = create_visual_state(
        points,
        triangles,
        fixed,
        [list(range(count, 2 * count))],
        np.asarray([count, count + 1, count + 2], dtype=int),
        target_spacing_m=np.full(len(points), 10.0),
        boundary_kinds=["interior"] * count
        + ["open", "open", "open"]
        + ["land"] * (count - 3)
        + ["interior"],
        hard_anchor_mask=hard,
    )
    config = replace(
        config,
        systematic_v5_enable_boundary_window_fallback=False,
        enable_valence_repair=True,
        enable_thin_repair=True,
        max_valence_removals_per_round=10,
        max_valence_l_over_h_count_increase=0,
        micro_relax_cycles=0,
    )
    return state, config, components


def _fake_valence_with_bounded_sliver(state, config, _components):
    before = topology._summary(state, config)
    delivered_topology = build_edge_topology(
        len(state.points),
        state.triangles,
    )
    center = int(
        max(
            range(len(state.points)),
            key=lambda node: len(
                delivered_topology.node_neighbors[int(node)]
            ),
        )
    )
    ring = list(range(10))
    replacement = topology._triangulate_ring_greedy(
        state.points,
        ring,
        None,
        max(int(config.max_valence), 8),
        removed_node=center,
    )
    assert replacement is not None
    replacement = np.asarray(replacement, dtype=int)
    split = list(map(int, replacement[0]))
    midpoint = 0.5 * (
        state.points[split[0]] + state.points[split[1]]
    )
    support_coordinate = midpoint + 1.0e-4 * (
        state.points[split[2]] - midpoint
    )
    support = len(state.points)
    state.points = np.vstack([state.points, support_coordinate])
    state.fixed = np.concatenate([state.fixed, [False]])
    state.targets = np.concatenate([state.targets, [10.0]])
    state.kinds.append("interior")
    state.hard = np.concatenate([state.hard, [False]])
    state.lineage = np.concatenate(
        [state.lineage, topology._new_lineage_ids(state, 1)]
    )
    outside = state.triangles[
        ~np.any(state.triangles == int(center), axis=1)
    ]
    replacement = np.vstack(
        [
            replacement[1:],
            np.asarray(
                [
                    [support, split[0], split[1]],
                    [support, split[1], split[2]],
                    [support, split[2], split[0]],
                ],
                dtype=int,
            ),
        ]
    )
    state.triangles = topology._orient_ccw(
        state.points,
        np.vstack([outside, replacement]),
    )
    state.last_affected = sorted(
        set(map(int, np.unique(state.triangles)))
    )
    state.ledger.append({"operation": "synthetic-valence-midpoint"})
    topology._compact(state)
    after = topology._summary(state, config)
    assert after["count_valence_above_limit"] < (
        before["count_valence_above_limit"]
    )
    assert after["valence_excess_sum"] < before["valence_excess_sum"]
    assert after["superthin_triangle_count"] > 0
    return {
        "accepted": 1,
        "rejected": 0,
        "attempted_count": 1,
        "before": before,
        "after": after,
    }


def _fake_counterproductive_post(state, config, _components):
    summary = topology._summary(state, config)
    state.ledger.append({"operation": "synthetic-counterproductive-post"})
    return {
        "accepted": 1,
        "rejected": 0,
        "before": summary,
        "after": summary,
    }


def _run_escrow(enabled: bool):
    state, config, components = _valence_fan()
    config = replace(
        config,
        topology_escrow_enabled=bool(enabled),
        topology_escrow_maximum_superthin_count=25,
        topology_escrow_maximum_superthin_severity=25.0,
        topology_escrow_maximum_valence=12,
    )
    original_points = state.points.copy()
    original_triangles = state.triangles.copy()
    original_valence = topology._repair_high_valence
    original_thin = topology._repair_superthin
    topology._repair_high_valence = _fake_valence_with_bounded_sliver
    topology._repair_superthin = _fake_counterproductive_post
    try:
        valence, post, transaction = (
            topology._repair_valence_thin_atomic(
                state,
                config,
                components,
            )
        )
    finally:
        topology._repair_high_valence = original_valence
        topology._repair_superthin = original_thin
    return (
        state,
        config,
        valence,
        post,
        transaction,
        original_points,
        original_triangles,
    )


def test_valence_midpoint_escrow_is_opt_in_bounded_and_explicit() -> None:
    (
        strict_state,
        _,
        strict_valence,
        strict_post,
        strict_transaction,
        original_points,
        original_triangles,
    ) = _run_escrow(False)
    assert strict_transaction["accepted"] is False
    assert strict_transaction["rolled_back"] is True
    assert "provisional_escrow_accepted" not in strict_transaction
    assert np.array_equal(strict_state.points, original_points)
    assert np.array_equal(strict_state.triangles, original_triangles)
    assert strict_valence["accepted"] == 0
    assert strict_post["accepted"] == 0

    (
        escrow_state,
        escrow_config,
        escrow_valence,
        escrow_post,
        transaction,
        _,
        _,
    ) = _run_escrow(True)
    assert transaction["accepted"] is True
    assert transaction["accepted_via"] == "valence_only_midpoint_escrow"
    assert transaction["provisional_escrow_accepted"] is True
    assert transaction["post_thin_rolled_back"] is True
    assert transaction["escrow_rejected_gates"] == []
    assert "final_compound_transaction_failed" in (
        transaction["escrow_trigger_reasons"]
    )
    assert "post_thin_counterproductive" in (
        transaction["escrow_trigger_reasons"]
    )
    midpoint = transaction["valence_only_midpoint"]
    assert midpoint["count_valence_above_limit"] == 0
    assert midpoint["valence_excess_sum"] == 0
    assert midpoint["maximum_valence"] <= 12
    assert 0 < midpoint["superthin_triangle_count"] <= (
        escrow_config.topology_escrow_maximum_superthin_count
    )
    assert midpoint["superthin_severity_sum"] <= (
        escrow_config.topology_escrow_maximum_superthin_severity
    )
    assert escrow_valence["topology_escrow_retained"] is True
    assert escrow_post["topology_escrow_post_thin_rolled_back"] is True
    assert escrow_post["accepted"] == 0
    assert any(
        value["operation"] == "synthetic-valence-midpoint"
        for value in escrow_state.ledger
    )
    assert all(
        value["operation"] != "synthetic-counterproductive-post"
        for value in escrow_state.ledger
    )
    ok, invariants, _ = topology._audit_state(
        escrow_state,
        escrow_config,
        1,
    )
    assert ok, invariants
    assert invariants["open_boundary_membership_unchanged"] is True
    assert invariants["restricted_edge_violation_count"] == 0


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "local-topology V5 extension selftests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
