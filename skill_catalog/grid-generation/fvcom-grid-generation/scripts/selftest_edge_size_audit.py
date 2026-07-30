#!/usr/bin/env python
"""Focused tests for the edge-aware target-size transition audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fvcom_grid_generation.edge_size_audit import (  # noqa: E402
    SCHEMA,
    audit_edge_target_sizes,
)


def _strip_mesh(cell_count: int = 6) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            (float(column), float(row))
            for column in range(cell_count + 1)
            for row in (0, 1)
        ],
        dtype=float,
    )
    triangles: list[tuple[int, int, int]] = []
    for column in range(cell_count):
        lower_left = 2 * column
        upper_left = lower_left + 1
        lower_right = lower_left + 2
        upper_right = lower_left + 3
        triangles.extend(
            [
                (lower_left, lower_right, upper_left),
                (lower_right, upper_right, upper_left),
            ]
        )
    return points, np.asarray(triangles, dtype=np.int64)


def _test_strata_and_conservative_sampling() -> None:
    points, triangles = _strip_mesh()

    def sampler(xy: np.ndarray) -> np.ndarray:
        # Along horizontal edges, the left endpoint is the conservative target.
        return 1.0 + xy[:, 0]

    boundary_targets = np.full(len(points), np.nan)
    boundary_targets[[0, 1]] = 0.5
    report = audit_edge_target_sizes(
        points,
        triangles,
        [[0, 1]],
        sampler,
        boundary_target_by_node=boundary_targets,
        transition_graph_rings=2,
    )
    assert report["schema"] == SCHEMA
    assert report["counts"]["boundary_edges_with_own_target"] == 1
    assert report["stratum_counts"]["triangles"] == {
        "boundary": 1,
        "first_ring": 1,
        "transition": 2,
        "true_interior": 8,
    }
    assert report["stratum_counts"]["edges"]["boundary"] == 1
    assert report["stratum_counts"]["edges"]["first_ring"] > 0
    assert report["stratum_counts"]["edges"]["transition"] > 0
    assert report["stratum_counts"]["edges"]["true_interior"] > 0
    # Boundary edge length is one and its requested target is 0.5.
    boundary = report["edge_l_over_h"]["boundary"]
    assert np.isclose(boundary["maximum"], 2.0)
    assert boundary["threshold_exceedance_counts"]["above_1_55"] == 1
    assert boundary["threshold_exceedance_counts"]["above_2"] == 0
    # Triangle values are the maximum of incident unique-edge values.
    assert (
        report["triangle_l_over_h"]["boundary"]["maximum"]
        >= report["edge_l_over_h"]["boundary"]["maximum"]
    )


def _test_chain_gradation_and_cyclic_normalization() -> None:
    points, triangles = _strip_mesh(cell_count=3)
    bottom_chain = {
        "chain_id": "bottom",
        "nodes": [0, 2, 4, 6],
        "cyclic": False,
    }
    targets = {0: 1.0, 2: 1.0, 4: 2.0, 6: 2.0}
    report = audit_edge_target_sizes(
        points,
        triangles,
        [bottom_chain],
        lambda xy: np.full(len(xy), 4.0),
        boundary_target_by_node=targets,
        transition_graph_rings=0,
        boundary_gradation_limit=0.2,
    )
    diagnostics = report["boundary_diagnostics"]
    chain = diagnostics["chains"][0]
    assert chain["chain_id"] == "bottom"
    assert chain["edge_count"] == 3
    assert chain["adjacent_edge_target_ratio"]["maximum"] > 1.0
    assert chain["target_gradation"]["above_limit_count"] == 1
    assert report["stratum_counts"]["triangles"]["transition"] == 0

    # Repeated-first-node syntax is normalized to a cyclic chain.  A square
    # split into two triangles supplies every requested perimeter edge.
    square_points = np.asarray(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    )
    square_triangles = np.asarray([(0, 1, 2), (0, 2, 3)])
    cyclic = audit_edge_target_sizes(
        square_points,
        square_triangles,
        [{"chain_id": "outer", "nodes": [0, 1, 2, 3, 0]}],
        lambda xy: np.ones(len(xy)),
    )
    assert cyclic["counts"]["constraint_edges"] == 4
    assert cyclic["boundary_diagnostics"]["chains"][0]["cyclic"] is True


def _test_boundary_field_interface_factor_two_pass_and_failure() -> None:
    points = np.asarray(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        dtype=float,
    )
    triangles = np.asarray([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
    closed_loop = [
        {
            "chain_id": "closed_outer",
            "nodes": [0, 1, 2, 3],
            "cyclic": True,
        }
    ]
    field = lambda xy: np.ones(len(xy), dtype=float)

    # Equality at the factor-two limit passes because the comparison is strict.
    passing = audit_edge_target_sizes(
        points,
        triangles,
        closed_loop,
        field,
        boundary_target_sampler=lambda xy: np.full(len(xy), 2.0),
    )
    interface = passing["boundary_field_interface"]
    assert interface["constraint_edge_count"] == 4
    assert interface["evaluated_edge_count"] == 4
    assert np.isclose(interface["symmetric_ratio"]["maximum"], 2.0)
    assert interface["factor_two_exceedance_count"] == 0
    assert interface["factor_two_passed"] is True
    assert passing["passed"] is True
    assert passing["failure_taxonomy"] == []

    failing = audit_edge_target_sizes(
        points,
        triangles,
        closed_loop,
        field,
        boundary_target_sampler=lambda xy: np.full(len(xy), 3.0),
    )
    interface = failing["boundary_field_interface"]
    assert np.isclose(interface["symmetric_ratio"]["maximum"], 3.0)
    assert interface["factor_two_exceedance_count"] == 4
    assert interface["boundary_coarser_than_field_count"] == 4
    assert interface["boundary_finer_than_field_count"] == 0
    assert interface["factor_two_passed"] is False
    assert failing["passed"] is False
    assert failing["failure_taxonomy"] == [
        "boundary_field_interface_factor_two_jump"
    ]


def _test_boundary_field_interface_is_pointwise_not_minimum_of_triplets() -> None:
    points = np.asarray(
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        dtype=float,
    )
    triangles = np.asarray([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
    boundary_targets = np.full(len(points), np.nan, dtype=float)
    boundary_targets[[0, 1]] = [1.0, 100.0]

    report = audit_edge_target_sizes(
        points,
        triangles,
        [{"chain_id": "edge", "nodes": [0, 1], "cyclic": False}],
        lambda xy: 100.0 - 99.0 * xy[:, 0],
        boundary_target_by_node=boundary_targets,
    )
    interface = report["boundary_field_interface"]
    # Both triplets have minimum 1.0, but the independent endpoint ratios are
    # 100.  Comparing reduced minima would therefore be a dangerous false pass.
    assert np.isclose(interface["symmetric_ratio"]["maximum"], 100.0)
    assert interface["factor_two_exceedance_count"] == 1
    assert interface["factor_two_sample_exceedance_count"] == 2
    assert report["passed"] is False


def main() -> int:
    tests = [
        ("strata_and_conservative_sampling", _test_strata_and_conservative_sampling),
        ("chain_gradation_and_cyclic_normalization", _test_chain_gradation_and_cyclic_normalization),
        (
            "boundary_field_interface_factor_two_pass_and_failure",
            _test_boundary_field_interface_factor_two_pass_and_failure,
        ),
        (
            "boundary_field_interface_is_pointwise_not_minimum_of_triplets",
            _test_boundary_field_interface_is_pointwise_not_minimum_of_triplets,
        ),
    ]
    passed = 0
    for name, test in tests:
        test()
        passed += 1
        print(json.dumps({"test": name, "status": "pass"}, sort_keys=True))
    print(json.dumps({"passed": passed, "total": len(tests)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
