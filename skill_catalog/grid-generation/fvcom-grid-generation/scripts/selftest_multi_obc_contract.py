#!/usr/bin/env python3
"""Focused tests for zero-, single-, multi-, and cyclic OBC contracts."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
from shapely.geometry import LineString, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.boundary import (  # noqa: E402
    BoundaryNodes,
    OpenBoundaryChain,
    _manifest_open_boundary_chains,
    evaluate_boundary_contract_v2,
)
from fvcom_grid_generation.metrics import (  # noqa: E402
    build_edge_topology,
    constraint_integrity,
)
from fvcom_grid_generation.mesh import MeshConfig, generate_mesh  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection  # noqa: E402
from fvcom_grid_generation.quality import evaluate_mesh_quality  # noqa: E402
from fvcom_grid_generation.sms_2dm import read_2dm, write_2dm  # noqa: E402
from fvcom_grid_generation.workflow import GridConfig  # noqa: E402


def _square_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    triangles_zero_based = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    return points, triangles_zero_based, triangles_zero_based + 1


def _boundary_nodes(
    *,
    kinds: list[str],
    hard: list[bool],
    open_indices: list[int],
    open_boundaries: list[OpenBoundaryChain] | None,
) -> BoundaryNodes:
    points, _triangles_zero, _triangles_one = _square_mesh()
    open_line = (
        LineString(points[np.asarray(open_indices, dtype=int)])
        if len(open_indices) >= 2
        else LineString()
    )
    return BoundaryNodes(
        xy=points,
        lonlat=np.zeros_like(points),
        kinds=kinds,
        target_spacing_m=np.full(len(points), 2.0, dtype=float),
        exterior_indices=[0, 1, 2, 3],
        open_boundary_indices=open_indices,
        constraint_chains=[[0, 1, 2, 3]],
        domain_polygon_xy=Polygon(points),
        open_boundary_xy=open_line,
        land_boundary_xy=LineString([points[0], points[1], points[2], points[3], points[0]]),
        island_polygons_xy=[],
        projection=local_utm_projection((-75.0, 39.0, -74.0, 40.0)),
        hard_anchor_mask=np.asarray(hard, dtype=bool),
        adaptive_resolution=True,
        resolution_profile="adaptive-coastal-v2",
        open_boundaries=open_boundaries,
    )


def _quality(
    target_size_by_triangle: np.ndarray | None,
    *,
    require_open_boundary: bool = True,
    open_boundary_nodes: np.ndarray | None = None,
    open_boundary_chains: list[list[int]] | None = None,
) -> dict:
    points, _triangles_zero, triangles_one = _square_mesh()
    legacy = (
        np.asarray([1, 2], dtype=int)
        if open_boundary_nodes is None
        else np.asarray(open_boundary_nodes, dtype=int)
    )
    return evaluate_mesh_quality(
        points,
        np.ones(len(points), dtype=float),
        triangles_one,
        legacy,
        {"boundary_constraint_recovered": True},
        constraint_chains=[[0, 1, 2, 3]],
        open_boundary_chains=open_boundary_chains,
        require_open_boundary=require_open_boundary,
        enforce_size_error=True,
        target_size_by_triangle=target_size_by_triangle,
    )


def test_legacy_single_nodestring_roundtrip() -> None:
    points, _triangles_zero, triangles_one = _square_mesh()
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "single.2dm"
        write_2dm(
            output,
            points,
            np.ones(len(points), dtype=float),
            triangles_one,
            np.asarray([1, 2], dtype=int),
            mesh_name="single",
        )
        mesh = read_2dm(output)
    assert mesh.mesh_name == "single"
    assert len(mesh.open_boundary_chains) == 1
    assert mesh.open_boundary_chains[0].tolist() == [1, 2]
    assert mesh.open_boundary_nodes.tolist() == [1, 2]


def test_two_nodestrings_roundtrip_without_flattening() -> None:
    points, _triangles_zero, triangles_one = _square_mesh()
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "two.2dm"
        write_2dm(
            output,
            points,
            np.ones(len(points), dtype=float),
            triangles_one,
            np.empty(0, dtype=int),
            open_boundary_chains=([1, 2], [4, 3]),
        )
        mesh = read_2dm(output)
    assert len(mesh.open_boundary_chains) == 2
    assert [chain.tolist() for chain in mesh.open_boundary_chains] == [[1, 2], [4, 3]]
    assert mesh.open_boundary_nodes.tolist() == [1, 2]


def test_numpy_iterables_and_unique_nodestring_ids() -> None:
    points, triangles_zero, triangles_one = _square_mesh()
    topology = build_edge_topology(len(points), triangles_zero)
    report = constraint_integrity(
        topology,
        np.asarray([[0, 1, 2, 3]], dtype=int),
        np.asarray([0, 1], dtype=int),
        open_boundary_cyclic=np.asarray([False], dtype=bool),
    )
    assert report["open_boundary_ordered"] is True

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "numpy_ids.2dm"
        write_2dm(
            output,
            points,
            np.ones(len(points), dtype=float),
            triangles_one,
            np.empty(0, dtype=int),
            open_boundary_chains=([1, 2], [4, 3]),
            open_boundary_ids=np.asarray([7, 8], dtype=int),
        )
        mesh = read_2dm(output)
        assert mesh.open_boundary_ids == (7, 8)

        try:
            write_2dm(
                output,
                points,
                np.ones(len(points), dtype=float),
                triangles_one,
                np.empty(0, dtype=int),
                open_boundary_chains=([1, 2], [4, 3]),
                open_boundary_ids=[7, 7],
            )
        except ValueError as exc:
            assert "unique positive ID" in str(exc)
        else:
            raise AssertionError("duplicate nodestring IDs were not rejected")


def test_lake_zero_nodestring_roundtrip_and_quality() -> None:
    points, _triangles_zero, triangles_one = _square_mesh()
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "lake.2dm"
        write_2dm(
            output,
            points,
            np.ones(len(points), dtype=float),
            triangles_one,
            np.empty(0, dtype=int),
        )
        mesh = read_2dm(output)
    assert not mesh.open_boundary_chains
    assert mesh.open_boundary_nodes.size == 0

    closed = _quality(
        np.ones(2, dtype=float),
        require_open_boundary=False,
        open_boundary_nodes=np.empty(0, dtype=int),
        open_boundary_chains=[],
    )
    assert "missing_open_boundary_nodestring" not in closed["failure_taxonomy"]
    required = _quality(
        np.ones(2, dtype=float),
        require_open_boundary=True,
        open_boundary_nodes=np.empty(0, dtype=int),
        open_boundary_chains=[],
    )
    assert "missing_open_boundary_nodestring" in required["failure_taxonomy"]
    explicit_zero = evaluate_mesh_quality(
        points,
        np.ones(len(points), dtype=float),
        triangles_one,
        np.empty(0, dtype=int),
        {"boundary_constraint_recovered": True},
        constraint_chains=[[0, 1, 2, 3]],
        open_boundary_chains=[],
        require_open_boundary=True,
        expected_open_boundary_count=0,
    )
    assert "missing_open_boundary_nodestring" not in explicit_zero["failure_taxonomy"]


def test_cyclic_nodestring_omits_repeat_and_audits_closure() -> None:
    points, triangles_zero, triangles_one = _square_mesh()
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "cyclic.2dm"
        write_2dm(
            output,
            points,
            np.ones(len(points), dtype=float),
            triangles_one,
            np.empty(0, dtype=int),
            open_boundary_chains=([1, 2, 3, 4, 1],),
        )
        mesh = read_2dm(output)
    assert mesh.open_boundary_chains[0].tolist() == [1, 2, 3, 4]

    topology = build_edge_topology(len(points), triangles_zero)
    valid = constraint_integrity(
        topology,
        [[0, 1, 2, 3]],
        None,
        [[0, 1, 2, 3]],
        [True],
    )
    assert valid["open_boundary_ordered"] is True
    assert valid["open_boundary_missing_pair_count"] == 0

    diagonal_closure = constraint_integrity(
        topology,
        [[0, 1, 2, 3]],
        None,
        [[0, 1, 2]],
        [True],
    )
    assert diagonal_closure["open_boundary_ordered"] is False
    assert diagonal_closure["open_boundary_missing_pairs"] == [[0, 2]]

    two_node_cyclic = constraint_integrity(
        topology,
        [[0, 1, 2, 3]],
        None,
        [[0, 1]],
        [True],
    )
    assert two_node_cyclic["open_boundary_ordered"] is False
    assert (
        two_node_cyclic["open_boundary_chains"][0][
            "minimum_node_count_satisfied"
        ]
        is False
    )


def test_multi_chain_integrity_has_no_phantom_bridge() -> None:
    points, triangles_zero, _triangles_one = _square_mesh()
    topology = build_edge_topology(len(points), triangles_zero)
    plural = constraint_integrity(
        topology,
        [[0, 1, 2, 3]],
        None,
        [[0, 1], [3, 2]],
        [False, False],
    )
    assert plural["open_boundary_chain_count"] == 2
    assert plural["open_boundary_ordered"] is True
    assert plural["open_boundary_missing_pair_count"] == 0

    flattened = constraint_integrity(
        topology,
        [[0, 1, 2, 3]],
        [0, 1, 3, 2],
    )
    assert flattened["open_boundary_ordered"] is False
    assert flattened["open_boundary_missing_pairs"] == [[1, 3]]


def test_interior_edge_cannot_be_an_open_boundary() -> None:
    points, triangles_zero, _triangles_one = _square_mesh()
    topology = build_edge_topology(len(points), triangles_zero)
    report = constraint_integrity(
        topology,
        [[0, 1, 2, 3]],
        None,
        [[0, 2]],
        [False],
    )
    assert (0, 2) in topology.edge_to_triangles
    assert len(topology.edge_to_triangles[(0, 2)]) == 2
    assert report["open_boundary_ordered"] is False
    assert report["open_boundary_missing_pairs"] == [[0, 2]]


def test_target_size_p95_and_maximum_gates() -> None:
    def debt_codes(report: dict) -> set[str]:
        return {
            str(value["code"])
            for value in report.get("regional_refinement_debt", [])
        }

    passing = _quality(np.ones(2, dtype=float))
    assert "target_size_l_over_h_p95_above_threshold" not in debt_codes(passing)
    assert "target_size_l_over_h_max_above_threshold" not in debt_codes(passing)

    p95_failure = _quality(np.full(2, 0.8, dtype=float))
    assert p95_failure["benchmark_grid_baseline_ready"]
    assert "target_size_l_over_h_p95_above_threshold" in debt_codes(p95_failure)
    assert "target_size_l_over_h_max_above_threshold" not in debt_codes(p95_failure)

    maximum_failure = _quality(np.full(2, 0.6, dtype=float))
    assert "target_size_l_over_h_p95_above_threshold" in debt_codes(maximum_failure)
    assert "target_size_l_over_h_max_above_threshold" in debt_codes(maximum_failure)

    missing = _quality(None)
    assert "missing_target_size_error_diagnostic" in debt_codes(missing)


def test_boundary_contract_exact_counts_and_cyclic_policy() -> None:
    legacy = _boundary_nodes(
        kinds=["open", "open", "land", "land"],
        hard=[True, True, False, False],
        open_indices=[0, 1],
        open_boundaries=None,
    )
    assert evaluate_boundary_contract_v2(legacy)["passed"] is True

    two = _boundary_nodes(
        kinds=["open", "open", "open", "open"],
        hard=[True, True, True, True],
        open_indices=[0, 1, 2, 3],
        open_boundaries=[
            OpenBoundaryChain("race", (0, 1)),
            OpenBoundaryChain("east_river", (2, 3)),
        ],
    )
    exact = evaluate_boundary_contract_v2(two, expected_open_boundary_count=2)
    assert exact["passed"] is True, exact
    try:
        generate_mesh(two, None, MeshConfig())  # type: ignore[arg-type]
    except ValueError as exc:
        assert "research Gmsh route" in str(exc)
    else:
        raise AssertionError("production mesher silently flattened two OBC chains")
    mismatch = evaluate_boundary_contract_v2(two, expected_open_boundary_count=1)
    assert "open_boundary_chain_count_mismatch" in mismatch["failure_taxonomy"]

    lake = _boundary_nodes(
        kinds=["land", "land", "land", "land"],
        hard=[False, False, False, False],
        open_indices=[],
        open_boundaries=[],
    )
    assert evaluate_boundary_contract_v2(lake, expected_open_boundary_count=0)["passed"] is True
    assert "open_boundary_chain_count_mismatch" in evaluate_boundary_contract_v2(
        lake,
        expected_open_boundary_count=1,
    )["failure_taxonomy"]

    hawaii = _boundary_nodes(
        kinds=["open", "open", "open", "open"],
        hard=[False, False, False, False],
        open_indices=[0, 1, 2, 3],
        open_boundaries=[
            OpenBoundaryChain(
                "hawaii_offshore",
                (0, 1, 2, 3),
                kind="cyclic_offshore",
                cyclic=True,
            )
        ],
    )
    cyclic = evaluate_boundary_contract_v2(hawaii, expected_open_boundary_count=1)
    assert cyclic["passed"] is True, cyclic
    assert "open_boundary_landfall_anchor_missing" not in cyclic["failure_taxonomy"]


def test_v2_manifest_chains_are_consumed_without_concatenation() -> None:
    assert GridConfig().boundary_resolution_profile == "adaptive-coastal-v2"
    manifest = {
        "profile": "adaptive-coastal-v2",
        "open_boundary_chains": [
            {
                "obc_id": 8,
                "is_closed": False,
                "node_sequence_zero_based": [50, 40],
            },
            {
                "obc_id": 3,
                "is_closed": False,
                "node_sequence_zero_based": [10, 20],
            },
        ],
    }
    chains = _manifest_open_boundary_chains(
        manifest,
        {10: 0, 20: 1, 30: 2, 40: 3, 50: 4},
        ["open", "open", "land", "open", "open"],
    )
    assert [chain.chain_id for chain in chains] == ["obc_003", "obc_008"]
    assert [chain.node_indices for chain in chains] == [(0, 1), (4, 3)]
    assert all(chain.orientation == "source" for chain in chains)


def main() -> int:
    tests = [
        test_legacy_single_nodestring_roundtrip,
        test_two_nodestrings_roundtrip_without_flattening,
        test_numpy_iterables_and_unique_nodestring_ids,
        test_lake_zero_nodestring_roundtrip_and_quality,
        test_cyclic_nodestring_omits_repeat_and_audits_closure,
        test_multi_chain_integrity_has_no_phantom_bridge,
        test_interior_edge_cannot_be_an_open_boundary,
        test_target_size_p95_and_maximum_gates,
        test_boundary_contract_exact_counts_and_cyclic_policy,
        test_v2_manifest_chains_are_consumed_without_concatenation,
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} multi-OBC contract tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
