#!/usr/bin/env python3
"""Standalone regression tests for the research-only Gmsh backend.

Run this file directly with the isolated experiment interpreter.  It uses
temporary directories, requires no pytest installation, and leaves no mesh
artifacts in the source tree.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from typing import Callable

import numpy as np
from shapely.geometry import LineString, Polygon


def _load_backend():
    backend_path = (
        Path(__file__).resolve().parent
        / "fvcom_grid_generation"
        / "gmsh_backend.py"
    )
    module_name = "_fvcom_grid_generation_gmsh_backend_selftest"
    spec = importlib.util.spec_from_file_location(module_name, backend_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Gmsh backend from {backend_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BACKEND = _load_backend()


def _expect_raises(
    error_type: type[BaseException],
    action: Callable[[], object],
    message_fragment: str | None = None,
) -> BaseException:
    try:
        action()
    except error_type as exc:
        if message_fragment is not None:
            assert message_fragment.lower() in str(exc).lower(), str(exc)
        return exc
    raise AssertionError(f"Expected {error_type.__name__} was not raised")


def _outer_loop(
    segment_kinds: tuple[str, str, str, str],
    *,
    side_m: float = 60_000.0,
) -> object:
    return BACKEND.SourceLoop(
        loop_id="exterior",
        xy=np.asarray(
            [
                [0.0, 0.0],
                [side_m, 0.0],
                [side_m, side_m],
                [0.0, side_m],
            ],
            dtype=float,
        ),
        segment_kinds=segment_kinds,
        source_vertex_ids=("outer_v0", "outer_v1", "outer_v2", "outer_v3"),
        role="exterior",
    )


def _center_island(*, side_m: float = 60_000.0) -> object:
    low = side_m * 0.40
    high = side_m * 0.60
    return BACKEND.SourceLoop(
        loop_id="center_island",
        xy=np.asarray(
            [
                [low, low],
                [high, low],
                [high, high],
                [low, high],
            ],
            dtype=float,
        ),
        segment_kinds=("island", "island", "island", "island"),
        source_vertex_ids=(
            "island_v0",
            "island_v1",
            "island_v2",
            "island_v3",
        ),
        role="island",
        island_id="center",
    )


def _triangle_edges(triangles_1based: np.ndarray) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for triangle in np.asarray(triangles_1based, dtype=np.int64):
        for left, right in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edges.add(tuple(sorted((int(left), int(right)))))
    return edges


def _assert_positive_ccw_triangles(result: object) -> None:
    triangles = np.asarray(result.triangles_1based, dtype=np.int64) - 1
    nodes = np.asarray(result.nodes_xy, dtype=float)
    a = nodes[triangles[:, 0]]
    b = nodes[triangles[:, 1]]
    c = nodes[triangles[:, 2]]
    twice_area = (
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    assert np.all(twice_area > 0.0), float(np.min(twice_area))


def _assert_source_vertices_retained(
    source_loops: tuple[object, ...],
    result: object,
) -> None:
    expected_ids = {
        source_id
        for loop in source_loops
        for source_id in loop.source_vertex_ids
    }
    assert set(result.source_vertex_node_ids) == expected_ids
    for loop in source_loops:
        for source_id, expected_xy in zip(loop.source_vertex_ids, loop.xy):
            node_id = int(result.source_vertex_node_ids[source_id])
            actual_xy = result.nodes_xy[node_id - 1]
            assert np.allclose(actual_xy, expected_xy, rtol=0.0, atol=1.0e-7)


def _assert_boundary_lineage(
    source_loops: tuple[object, ...],
    result: object,
) -> None:
    source_by_id = {loop.loop_id: loop for loop in source_loops}
    delivered_by_id = {loop.loop_id: loop for loop in result.delivered_loops}
    assert set(delivered_by_id) == set(source_by_id)

    for loop_id, delivered in delivered_by_id.items():
        source = source_by_id[loop_id]
        assert delivered.node_ids
        assert len(delivered.node_ids) == len(set(delivered.node_ids))
        assert len(delivered.node_ids) == len(delivered.lineage)
        fractions = []
        for item in delivered.lineage:
            assert item.loop_id == loop_id
            assert 0 <= item.source_segment_index < len(source.xy)
            assert 0.0 <= item.interpolation_weight <= 1.0
            assert 0.0 <= item.loop_normalized_arclength < 1.0
            assert (
                item.source_segment_kind
                == source.segment_kinds[item.source_segment_index]
            )
            start = source.xy[item.source_segment_index]
            end = source.xy[(item.source_segment_index + 1) % len(source.xy)]
            reconstructed = (
                (1.0 - item.interpolation_weight) * start
                + item.interpolation_weight * end
            )
            actual = result.nodes_xy[item.mesh_node_id - 1]
            assert np.allclose(actual, reconstructed, rtol=0.0, atol=1.0e-6)
            fractions.append(item.loop_normalized_arclength)
        assert np.all(np.diff(np.asarray(fractions)) > 0.0)
        assert sum(item.is_source_vertex for item in delivered.lineage) == len(
            source.xy
        )


def _validate_gate_has_zero_land_crossing(
    gate_xy: np.ndarray,
    land_polygons_xy: tuple[np.ndarray, ...],
    *,
    tolerance_m: float = 1.0e-9,
) -> None:
    """Reject a prospective exchange gate with positive-length land overlap."""

    gate = LineString(np.asarray(gate_xy, dtype=float))
    if gate.is_empty or not gate.is_simple or gate.length <= tolerance_m:
        raise ValueError("exchange gate is empty, zero-length, or non-simple")
    for index, coordinates in enumerate(land_polygons_xy):
        land = Polygon(np.asarray(coordinates, dtype=float))
        if not land.is_valid or land.is_empty:
            raise ValueError(f"land polygon {index} is invalid")
        overlap_length = float(gate.intersection(land).length)
        if overlap_length > tolerance_m:
            raise ValueError(
                f"exchange gate crosses land polygon {index} for "
                f"{overlap_length:.6f} m"
            )


def test_threshold_field_direction() -> None:
    config = BACKEND.GmshConfig(
        h_uniform_m=2_000.0,
        near_size_m=8_000.0,
        dist_min_m=10_000.0,
        dist_max_m=70_000.0,
    )
    values = BACKEND.threshold_target_size(
        np.asarray([0.0, 10_000.0, 70_000.0]),
        config,
    )
    assert np.array_equal(values, np.asarray([8_000.0, 8_000.0, 2_000.0]))
    assert values[0] == values[1] > values[2]

    # Exercise Gmsh's actual Distance/Threshold implementation, not only the
    # analytical mirror above.  The left boundary is the OBC: element length
    # must decrease from the 8 km near band, through the linear transition,
    # to the 2 km far field.
    exterior = BACKEND.SourceLoop(
        loop_id="threshold_direction_exterior",
        xy=np.asarray(
            [
                [0.0, 0.0],
                [160_000.0, 0.0],
                [160_000.0, 40_000.0],
                [0.0, 40_000.0],
            ],
            dtype=float,
        ),
        segment_kinds=("land", "land", "land", "open"),
        source_vertex_ids=(
            "threshold_v0",
            "threshold_v1",
            "threshold_v2",
            "threshold_v3",
        ),
        role="exterior",
    )
    geometry = BACKEND.GmshGeometry(
        exterior=exterior,
        open_boundaries=(
            BACKEND.SourceOpenBoundary(
                chain_id="left_obc",
                exterior_segment_indices=(3,),
                kind="ocean_exchange",
            ),
        ),
    )
    with tempfile.TemporaryDirectory(prefix="gmsh_threshold_direction_") as temporary:
        result = BACKEND.run_gmsh_attempt(
            geometry,
            config,
            Path(temporary) / "threshold_direction.msh",
        )

    triangles = np.asarray(result.triangles_1based, dtype=int) - 1
    triangle_xy = np.asarray(result.nodes_xy, dtype=float)[triangles]
    centroids_x = np.mean(triangle_xy[:, :, 0], axis=1)
    edge_lengths = np.stack(
        [
            np.linalg.norm(triangle_xy[:, 1] - triangle_xy[:, 0], axis=1),
            np.linalg.norm(triangle_xy[:, 2] - triangle_xy[:, 1], axis=1),
            np.linalg.norm(triangle_xy[:, 0] - triangle_xy[:, 2], axis=1),
        ],
        axis=1,
    )
    longest_edges = np.max(edge_lengths, axis=1)
    near_median = float(np.median(longest_edges[centroids_x <= 10_000.0]))
    transition_median = float(
        np.median(
            longest_edges[
                (centroids_x >= 35_000.0) & (centroids_x <= 45_000.0)
            ]
        )
    )
    far_median = float(np.median(longest_edges[centroids_x >= 90_000.0]))
    assert near_median > 1.15 * transition_median, (
        near_median,
        transition_median,
        far_median,
    )
    assert transition_median > 1.15 * far_median, (
        near_median,
        transition_median,
        far_median,
    )
    assert near_median > 2.0 * far_median, (
        near_median,
        transition_median,
        far_median,
    )


def test_single_coastal_obc() -> None:
    exterior = _outer_loop(("open", "land", "land", "land"))
    geometry = BACKEND.GmshGeometry(
        exterior=exterior,
        open_boundaries=(
            BACKEND.SourceOpenBoundary(
                chain_id="coastal",
                exterior_segment_indices=(0,),
                kind="ocean_exchange",
            ),
        ),
    )
    config = BACKEND.GmshConfig(h_uniform_m=6_000.0)
    with tempfile.TemporaryDirectory(prefix="gmsh_single_obc_") as temporary:
        msh_path = Path(temporary) / "single_coastal.msh"
        preflight = BACKEND.measure_boundary_mesh(geometry, config)
        result = BACKEND.run_gmsh_attempt(geometry, config, msh_path)
        assert msh_path.is_file()
        assert preflight.boundary_node_count == result.boundary_node_count_1d
        assert len(result.open_boundaries) == 1
        delivered = result.open_boundaries[0]
        assert delivered.node_ids[0] == result.source_vertex_node_ids["outer_v0"]
        assert delivered.node_ids[-1] == result.source_vertex_node_ids["outer_v1"]
        assert len(delivered.node_ids) > 2
        chain_fraction = np.asarray(
            [item.chain_normalized_arclength for item in delivered.lineage],
            dtype=float,
        )
        assert chain_fraction[0] == 0.0
        assert np.isclose(chain_fraction[-1], 1.0)
        assert np.all(np.diff(chain_fraction) > 0.0)
        assert {"WET_DOMAIN", "OBC_coastal", "LAND_COASTLINE"} <= set(
            result.physical_groups
        )
        _assert_source_vertices_retained((exterior,), result)
        _assert_boundary_lineage((exterior,), result)
        _assert_positive_ccw_triangles(result)
        assert np.isfinite(result.element_quality.sicn.minimum)
        assert np.isfinite(result.element_quality.gamma.minimum)


def test_two_disjoint_obcs_with_island_hole() -> None:
    exterior = _outer_loop(("open", "land", "open", "land"))
    island = _center_island()
    geometry = BACKEND.GmshGeometry(
        exterior=exterior,
        holes=(island,),
        open_boundaries=(
            BACKEND.SourceOpenBoundary(
                chain_id="south_exchange",
                exterior_segment_indices=(0,),
                kind="ocean_exchange",
            ),
            BACKEND.SourceOpenBoundary(
                chain_id="north_exchange",
                exterior_segment_indices=(2,),
                kind="ocean_exchange",
            ),
        ),
    )
    with tempfile.TemporaryDirectory(prefix="gmsh_two_obc_") as temporary:
        result = BACKEND.run_gmsh_attempt(
            geometry,
            BACKEND.GmshConfig(h_uniform_m=7_500.0),
            Path(temporary) / "two_obc.msh",
        )
        assert [item.chain_id for item in result.open_boundaries] == [
            "south_exchange",
            "north_exchange",
        ]
        assert set(result.open_boundaries[0].node_ids).isdisjoint(
            result.open_boundaries[1].node_ids
        )
        assert {
            "WET_DOMAIN",
            "LAND_COASTLINE",
            "OBC_south_exchange",
            "OBC_north_exchange",
            "ISLAND_center",
        } <= set(result.physical_groups)
        _assert_source_vertices_retained((exterior, island), result)
        _assert_boundary_lineage((exterior, island), result)
        _assert_positive_ccw_triangles(result)


def test_closed_lake_zero_obc() -> None:
    exterior = _outer_loop(("land", "land", "land", "land"))
    island = _center_island()
    geometry = BACKEND.GmshGeometry(exterior=exterior, holes=(island,))
    config = BACKEND.GmshConfig(
        h_uniform_m=7_500.0,
        constant_field=True,
    )
    distances = np.asarray([0.0, 10_000.0, 70_000.0])
    constant_values = np.full(distances.shape, config.h_uniform_m)
    assert np.array_equal(constant_values, np.asarray([7_500.0] * 3))
    with tempfile.TemporaryDirectory(prefix="gmsh_closed_lake_") as temporary:
        result = BACKEND.run_gmsh_attempt(
            geometry,
            config,
            Path(temporary) / "closed_lake.msh",
        )
        assert result.open_boundaries == ()
        assert not any(name.startswith("OBC_") for name in result.physical_groups)
        assert set(result.physical_groups) == {
            "WET_DOMAIN",
            "LAND_COASTLINE",
            "ISLAND_center",
        }
        _assert_source_vertices_retained((exterior, island), result)
        _assert_boundary_lineage((exterior, island), result)
        _assert_positive_ccw_triangles(result)


def test_cyclic_all_open_with_hole_and_repeatability() -> None:
    exterior = _outer_loop(("open", "open", "open", "open"))
    island = _center_island()
    geometry = BACKEND.GmshGeometry(
        exterior=exterior,
        holes=(island,),
        open_boundaries=(
            BACKEND.SourceOpenBoundary(
                chain_id="cyclic_offshore",
                exterior_segment_indices=(0, 1, 2, 3),
                kind="offshore_ring",
                cyclic=True,
            ),
        ),
    )
    config = BACKEND.GmshConfig(h_uniform_m=7_500.0)
    with tempfile.TemporaryDirectory(prefix="gmsh_cyclic_") as temporary:
        root = Path(temporary)
        first_path = root / "cyclic_first.msh"
        second_path = root / "cyclic_second.msh"
        first = BACKEND.run_gmsh_attempt(geometry, config, first_path)
        second = BACKEND.run_gmsh_attempt(geometry, config, second_path)

        chain = first.open_boundaries[0]
        assert chain.cyclic
        assert chain.node_ids[0] != chain.node_ids[-1]
        assert len(chain.node_ids) == len(set(chain.node_ids))
        assert tuple(chain.node_ids) == tuple(
            first.delivered_open_boundaries_1based["cyclic_offshore"]
        )
        assert tuple(sorted((chain.node_ids[-1], chain.node_ids[0]))) in (
            _triangle_edges(first.triangles_1based)
        )
        source_nodes = {
            first.source_vertex_node_ids[source_id]
            for source_id in exterior.source_vertex_ids
        }
        assert source_nodes <= set(chain.node_ids)
        chain_fractions = np.asarray(
            [item.chain_normalized_arclength for item in chain.lineage],
            dtype=float,
        )
        assert chain_fractions[0] == 0.0
        assert np.all(chain_fractions < 1.0)
        assert np.all(np.diff(chain_fractions) > 0.0)
        assert "OBC_cyclic_offshore" in first.physical_groups
        assert "LAND_COASTLINE" not in first.physical_groups
        assert "ISLAND_center" in first.physical_groups

        assert np.array_equal(first.nodes_xy, second.nodes_xy)
        assert np.array_equal(first.triangles_1based, second.triangles_1based)
        assert first.gmsh_node_tags == second.gmsh_node_tags
        assert (
            first.delivered_open_boundaries_1based
            == second.delivered_open_boundaries_1based
        )
        first_hash = hashlib.sha256(first_path.read_bytes()).hexdigest()
        second_hash = hashlib.sha256(second_path.read_bytes()).hexdigest()
        assert first_hash == second_hash
        _assert_source_vertices_retained((exterior, island), first)
        _assert_boundary_lineage((exterior, island), first)
        _assert_positive_ccw_triangles(first)


def test_land_crossing_gate_and_invalid_chain_rejected() -> None:
    barrier_land = np.asarray(
        [
            [25_000.0, -5_000.0],
            [35_000.0, -5_000.0],
            [35_000.0, 5_000.0],
            [25_000.0, 5_000.0],
        ]
    )
    _expect_raises(
        ValueError,
        lambda: _validate_gate_has_zero_land_crossing(
            np.asarray([[0.0, 0.0], [60_000.0, 0.0]]),
            (barrier_land,),
        ),
        "crosses land",
    )
    _validate_gate_has_zero_land_crossing(
        np.asarray([[0.0, 10_000.0], [60_000.0, 10_000.0]]),
        (barrier_land,),
    )

    exterior = _outer_loop(("open", "land", "open", "land"))
    invalid_chain = BACKEND.SourceOpenBoundary(
        chain_id="noncontiguous",
        exterior_segment_indices=(0, 2),
    )
    _expect_raises(
        ValueError,
        lambda: BACKEND.GmshGeometry(
            exterior=exterior,
            open_boundaries=(invalid_chain,),
        ),
        "not contiguous",
    )


TESTS: tuple[Callable[[], None], ...] = (
    test_threshold_field_direction,
    test_single_coastal_obc,
    test_two_disjoint_obcs_with_island_hole,
    test_closed_lake_zero_obc,
    test_cyclic_all_open_with_hole_and_repeatability,
    test_land_crossing_gate_and_invalid_chain_rejected,
)


def main() -> int:
    failures: list[tuple[str, BaseException]] = []
    for test in TESTS:
        try:
            test()
        except BaseException as exc:
            failures.append((test.__name__, exc))
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    if failures:
        print(f"{len(failures)} of {len(TESTS)} Gmsh experiment tests failed")
        return 1
    print(f"All {len(TESTS)} Gmsh experiment tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
