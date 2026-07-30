#!/usr/bin/env python3
"""Focused standalone tests for the regional raw portfolio case runner."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from typing import Callable

import numpy as np
from shapely.geometry import Polygon


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fvcom_grid_generation.bathymetry import BathymetryGrid, coarsen_for_size_field
from fvcom_grid_generation.gmsh_experiment import PreparedCase, prepare_case
from fvcom_grid_generation.portfolio_case import (
    DEFAULT_PRIMARY_CANDIDATE,
    BoundaryTraceSizeSampler,
    GMSH_CANDIDATE_ALGORITHMS,
    PortfolioCaseConfig,
    ProjectedSizeSampler,
    _CandidateMesh,
    _apply_case_budget_targets,
    _execute_candidate,
    _load_canonical_boundary,
    _preflight_report,
    _prepared_case_on_boundary,
    _reconciliation_changed_obc_sequence,
    _reconcile_boundary_and_size_field,
    _roundtrip_report,
    _run_clean_room_candidate,
    _run_gmsh_candidate,
    _sampler_adjusted_node_budget,
    _scientific_bundle_sha256,
    _size_field_config,
    _validate_output_path_budget,
    _source_obc_node_indices,
    _write_raw_metric_comparison,
    capability_routing,
    normalize_candidate_ids,
    run_portfolio_case,
)
from fvcom_grid_generation.projection import (
    local_utm_projection,
    project_points,
)
from fvcom_grid_generation.size_field import (
    build_size_field,
    estimate_node_budget,
)
from run_mesher_portfolio_case import build_parser


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


def _synthetic_case(*, open_boundaries: tuple[object, ...] = ()) -> PreparedCase:
    lon = np.linspace(-76.10, -76.04, 25)
    lat = np.linspace(43.20, 43.24, 21)
    projection = local_utm_projection(
        (float(lon[0]), float(lat[0]), float(lon[-1]), float(lat[-1]))
    )
    exterior_lonlat = np.asarray(
        [
            [-76.09, 43.21],
            [-76.05, 43.21],
            [-76.05, 43.23],
            [-76.09, 43.23],
        ],
        dtype=float,
    )
    exterior_xy = project_points(exterior_lonlat, projection)
    domain = Polygon(exterior_lonlat)
    bathy = BathymetryGrid(
        lon=lon,
        lat=lat,
        depth=np.full((len(lat), len(lon)), 12.0, dtype=float),
        source_path="synthetic_bathymetry",
    )
    manifest = {
        "schema_version": "gmsh_fvcom_case_v1",
        "case_id": "synthetic_lake",
        "display_name": "Synthetic Lake",
        "boundary": {
            "input_kind": "model_loops_v1",
            "expected_open_boundary_count": len(open_boundaries),
        },
    }
    return PreparedCase(
        manifest=manifest,
        manifest_path=Path("synthetic_case.json"),
        workspace_root=Path.cwd(),
        projection=projection,
        exterior_xy=exterior_xy,
        holes_xy=(),
        exterior_segment_kinds=("land", "land", "land", "land"),
        hard_anchor_vertex_indices=(),
        open_boundaries=open_boundaries,
        source_domain_lonlat=domain,
        bathymetry=bathy,
        input_paths={},
        boundary_revalidation={},
        manifest_sha256="synthetic",
    )


def _uniform_size_field(prepared: PreparedCase, size_m: float = 1_000.0):
    shape = prepared.bathymetry.depth.shape
    return SimpleNamespace(
        lon=prepared.bathymetry.lon,
        lat=prepared.bathymetry.lat,
        size=np.full(shape, size_m, dtype=float),
        coverage_mask=np.ones(shape, dtype=bool),
        domain_mask=np.ones(shape, dtype=bool),
        report={
            "schema_version": "fvcom_size_field_v4",
            "sampling_interface": {
                "schema_version": "fvcom_wet_mask_sampling_v2",
                "method": (
                    "bilinear_all_positive_weight_active_else_"
                    "highest_weight_positive_active_else_coarsest_covered"
                ),
            },
        },
    )


def test_candidate_normalization_and_capability_routing() -> None:
    assert normalize_candidate_ids(["clean-room", "gmsh-1", "gmsh-5", "gmsh-6"]) == (
        "clean_room_raw",
        "gmsh_meshadapt_1",
        "gmsh_delaunay_5",
        "gmsh_frontal_delaunay_6",
    )
    assert GMSH_CANDIDATE_ALGORITHMS == {
        "gmsh_meshadapt_1": 1,
        "gmsh_delaunay_5": 5,
        "gmsh_frontal_delaunay_6": 6,
    }
    assert normalize_candidate_ids(None)[0] == DEFAULT_PRIMARY_CANDIDATE
    _expect_raises(
        ValueError,
        lambda: normalize_candidate_ids(["unknown"]),
        "unknown candidate",
    )
    closed = capability_routing(SimpleNamespace(open_boundaries=()))
    assert closed["default_raw_candidate"] == DEFAULT_PRIMARY_CANDIDATE
    assert (
        closed["candidates"][DEFAULT_PRIMARY_CANDIDATE]["policy_role"]
        == "research_default_raw"
    )
    assert closed["candidates"]["clean_room_raw"]["supported"]
    plural = capability_routing(
        SimpleNamespace(
            open_boundaries=(
                SimpleNamespace(cyclic=False),
                SimpleNamespace(cyclic=False),
            )
        )
    )
    assert not plural["candidates"]["clean_room_raw"]["supported"]
    assert plural["candidates"]["gmsh_delaunay_5"]["supported"]
    cyclic = capability_routing(
        SimpleNamespace(open_boundaries=(SimpleNamespace(cyclic=True),))
    )
    assert not cyclic["candidates"]["clean_room_raw"]["supported"]
    assert cyclic["candidates"]["gmsh_frontal_delaunay_6"]["supported"]


def test_source_obc_node_ordering() -> None:
    forward = SimpleNamespace(
        chain_id="forward",
        exterior_segment_indices=(1, 2),
        orientation="source",
        cyclic=False,
    )
    reverse = SimpleNamespace(
        chain_id="reverse",
        exterior_segment_indices=(2, 1),
        orientation="reverse",
        cyclic=False,
    )
    cyclic = SimpleNamespace(
        chain_id="cyclic",
        exterior_segment_indices=(0, 1, 2, 3),
        orientation="source",
        cyclic=True,
    )
    assert _source_obc_node_indices(forward, 4) == (1, 2, 3)
    assert _source_obc_node_indices(reverse, 4) == (3, 2, 1)
    assert _source_obc_node_indices(cyclic, 4) == (0, 1, 2, 3)


def test_partial_forward_reverse_obc_rebase_and_segment_kind_mask() -> None:
    forward = SimpleNamespace(
        chain_id="west_exchange",
        kind="exchange",
        cyclic=False,
        orientation="source",
        exterior_segment_indices=(0,),
    )
    reverse = SimpleNamespace(
        chain_id="east_exchange",
        kind="exchange",
        cyclic=False,
        orientation="reverse",
        exterior_segment_indices=(2,),
    )
    prepared = _synthetic_case(open_boundaries=(forward, reverse))
    boundary, _ = _load_canonical_boundary(prepared, PortfolioCaseConfig())
    assert [tuple(chain.node_indices) for chain in boundary.open_boundaries] == [
        (0, 1),
        (3, 2),
    ]

    rebased = _prepared_case_on_boundary(prepared, boundary)
    assert [
        (
            chain.chain_id,
            chain.kind,
            chain.cyclic,
            chain.orientation,
            chain.exterior_segment_indices,
        )
        for chain in rebased.open_boundaries
    ] == [
        ("west_exchange", "exchange", False, "source", (0,)),
        ("east_exchange", "exchange", False, "reverse", (2,)),
    ]
    assert rebased.exterior_segment_kinds == (
        "open",
        "land",
        "open",
        "land",
    )


def test_cyclic_obc_rebase_includes_last_to_first_segment() -> None:
    cyclic = SimpleNamespace(
        chain_id="offshore_cycle",
        kind="offshore",
        cyclic=True,
        orientation="source",
        exterior_segment_indices=(0, 1, 2, 3),
    )
    prepared = _synthetic_case(open_boundaries=(cyclic,))
    boundary, _ = _load_canonical_boundary(prepared, PortfolioCaseConfig())
    delivered = boundary.open_boundaries[0]
    assert tuple(delivered.node_indices) == (0, 1, 2, 3)

    rebased = _prepared_case_on_boundary(prepared, boundary)
    rebased_chain = rebased.open_boundaries[0]
    assert rebased_chain.exterior_segment_indices == (0, 1, 2, 3)
    assert rebased_chain.exterior_segment_indices[-1] == (
        len(rebased.exterior_xy) - 1
    )
    assert (
        delivered.node_indices[-1],
        delivered.node_indices[0],
    ) == (3, 0)
    assert rebased.exterior_segment_kinds == ("open",) * 4


def test_reconciliation_lineage_controls_forcing_compatibility() -> None:
    forward = SimpleNamespace(
        chain_id="west_exchange",
        kind="exchange",
        cyclic=False,
        orientation="source",
        exterior_segment_indices=(0,),
    )
    reverse = SimpleNamespace(
        chain_id="east_exchange",
        kind="exchange",
        cyclic=False,
        orientation="reverse",
        exterior_segment_indices=(2,),
    )
    prepared = _synthetic_case(open_boundaries=(forward, reverse))
    boundary, _ = _load_canonical_boundary(prepared, PortfolioCaseConfig())
    node_count = len(boundary.xy)
    unchanged = replace(
        boundary,
        metadata={
            "reconciliation_inserted": np.zeros(node_count, dtype=bool),
            "reconciliation_source_node_index_zero_based": np.arange(
                node_count,
                dtype=int,
            ),
        },
    )
    assert not _reconciliation_changed_obc_sequence(prepared, unchanged)

    inserted_mask = np.zeros(node_count, dtype=bool)
    inserted_mask[3] = True
    inserted_lineage = np.arange(node_count, dtype=int)
    inserted_lineage[3] = -1
    inserted = replace(
        boundary,
        metadata={
            "reconciliation_inserted": inserted_mask,
            "reconciliation_source_node_index_zero_based": inserted_lineage,
        },
    )
    assert _reconciliation_changed_obc_sequence(prepared, inserted)

    reordered_lineage = np.arange(node_count, dtype=int)
    reordered_lineage[[0, 1]] = reordered_lineage[[1, 0]]
    reordered = replace(
        boundary,
        metadata={
            "reconciliation_inserted": np.zeros(node_count, dtype=bool),
            "reconciliation_source_node_index_zero_based": reordered_lineage,
        },
    )
    assert _reconciliation_changed_obc_sequence(prepared, reordered)


def test_case_budget_targets_and_geometry_forced_edge_count() -> None:
    forward = SimpleNamespace(
        chain_id="west_exchange",
        kind="exchange",
        cyclic=False,
        orientation="source",
        exterior_segment_indices=(0,),
    )
    prepared = _synthetic_case(open_boundaries=(forward,))
    boundary, _ = _load_canonical_boundary(prepared, PortfolioCaseConfig())
    exterior = boundary.constraint_chains[0]
    edge_lengths = np.asarray(
        [
            np.linalg.norm(
                boundary.xy[end] - boundary.xy[start]
            )
            for start, end in zip(
                exterior,
                exterior[1:] + exterior[:1],
            )
        ],
        dtype=float,
    )
    solid_target = float(2.0 * np.max(edge_lengths))
    open_target = float(0.5 * np.min(edge_lengths))
    updated, report = _apply_case_budget_targets(
        boundary,
        {
            "applied": True,
            "solid_and_island_target_m": solid_target,
            "open_boundary_target_m": open_target,
        },
        geometry_continuity=False,
    )
    assert np.allclose(
        updated.target_spacing_m,
        [open_target, open_target, solid_target, solid_target],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert report["open_boundary_node_count"] == 2
    assert report["geometry_forced_subgrid_edge_count"] == 1
    assert np.array_equal(
        updated.metadata["case_budget_target_spacing_m"],
        updated.target_spacing_m,
    )

    geometry_updated, geometry_report = _apply_case_budget_targets(
        boundary,
        {
            "applied": True,
            "solid_and_island_target_m": solid_target,
            "open_boundary_target_m": open_target,
        },
        geometry_continuity=True,
        geometry_metric_ratio=0.80,
    )
    expected_geometry = np.full(len(boundary.xy), np.inf, dtype=float)
    for start, end in zip(
        exterior,
        exterior[1:] + exterior[:1],
    ):
        value = (
            np.linalg.norm(boundary.xy[end] - boundary.xy[start])
            / 0.80
        )
        expected_geometry[start] = min(
            expected_geometry[start],
            value,
        )
        expected_geometry[end] = min(
            expected_geometry[end],
            value,
        )
    assert np.allclose(
        geometry_updated.target_spacing_m,
        np.minimum(updated.target_spacing_m, expected_geometry),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert geometry_report["geometry_continuity_applied"]
    assert geometry_report["geometry_limited_node_count"] > 0
    assert np.array_equal(
        geometry_updated.metadata[
            "geometry_continuity_target_spacing_m"
        ],
        expected_geometry,
    )

    independent_updated, independent_report = (
        _apply_case_budget_targets(
            boundary,
            {"applied": False},
            geometry_continuity=True,
            geometry_metric_ratio=1.0,
        )
    )
    expected_independent_geometry = np.full(
        len(boundary.xy),
        np.inf,
        dtype=float,
    )
    for start, end in zip(
        exterior,
        exterior[1:] + exterior[:1],
    ):
        length = np.linalg.norm(
            boundary.xy[end] - boundary.xy[start]
        )
        expected_independent_geometry[start] = min(
            expected_independent_geometry[start],
            length,
        )
        expected_independent_geometry[end] = min(
            expected_independent_geometry[end],
            length,
        )
    assert independent_report["applied"] is False
    assert independent_report["geometry_continuity_applied"] is True
    assert np.allclose(
        independent_updated.target_spacing_m,
        np.minimum(
            boundary.target_spacing_m,
            expected_independent_geometry,
        ),
    )


def test_portfolio_config_reconciliation_validation() -> None:
    valid = PortfolioCaseConfig(
        boundary_reconciliation_max_iterations=1,
        boundary_metric_edge=0.25,
        boundary_field_compatibility_factor=1.000001,
    )
    assert valid.boundary_reconciliation_max_iterations == 1
    assert valid.boundary_metric_edge > 0.0
    assert valid.boundary_field_compatibility_factor > 1.0
    assert valid.boundary_target_combination == "sampled_field"
    assert valid.boundary_geometry_continuity
    assert valid.boundary_geometry_metric_ratio == 1.0
    assert valid.gradation == 0.10
    assert valid.boundary_trace_samples_per_target == 4.0
    assert valid.boundary_trace_nearest_sample_count == 16
    for value in (1.0, 0.0, -1.0, np.nan):
        _expect_raises(
            ValueError,
            lambda value=value: PortfolioCaseConfig(
                boundary_field_compatibility_factor=value
            ),
            "finite and > 1",
        )
    for value in (0.0, -1.0, np.nan):
        _expect_raises(
            ValueError,
            lambda value=value: PortfolioCaseConfig(
                boundary_metric_edge=value
            ),
            "finite and positive",
        )
    _expect_raises(
        ValueError,
        lambda: PortfolioCaseConfig(
            boundary_reconciliation_max_iterations=0
        ),
        "must be positive",
    )
    _expect_raises(
        ValueError,
        lambda: PortfolioCaseConfig(
            boundary_target_combination="implicit"
        ),
        "must be 'minimum' or 'sampled_field'",
    )
    for value in (1.0, 0.0, np.nan):
        _expect_raises(
            ValueError,
            lambda value=value: PortfolioCaseConfig(
                boundary_trace_samples_per_target=value
            ),
            "must be finite and at least two",
        )
    _expect_raises(
        ValueError,
        lambda: PortfolioCaseConfig(
            boundary_trace_nearest_sample_count=0
        ),
        "must be positive",
    )
    for value in (0.0, -1.0, 1.00001, np.nan):
        _expect_raises(
            ValueError,
            lambda value=value: PortfolioCaseConfig(
                boundary_geometry_metric_ratio=value
            ),
            "in (0, 1]",
        )
    _expect_raises(
        ValueError,
        lambda: _size_field_config(
            PortfolioCaseConfig(
                land_spacing_m=2_000.0,
                maximum_size_m=1_000.0,
            )
        ),
        "exceeds maximum_size_m",
    )


def test_portfolio_cli_continuity_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--case-manifest",
            "case.json",
            "--output-dir",
            "fresh-output",
        ]
    )
    assert args.gradation == 0.10
    assert args.boundary_geometry_metric_ratio == 1.0
    assert args.boundary_reconciliation_max_iterations == 8
    assert args.boundary_field_compatibility_factor == 1.5
    assert args.boundary_target_combination == "sampled_field"
    assert not args.disable_boundary_geometry_continuity
    assert not args.disable_case_budget_spacing_policy


def test_cached_projected_size_sampler() -> None:
    prepared = _synthetic_case()
    lon_grid, lat_grid = np.meshgrid(
        prepared.bathymetry.lon,
        prepared.bathymetry.lat,
    )
    values = 1_000.0 + 10.0 * (lon_grid - lon_grid.min()) + 5.0 * (
        lat_grid - lat_grid.min()
    )
    field = SimpleNamespace(
        lon=prepared.bathymetry.lon,
        lat=prepared.bathymetry.lat,
        size=values,
        coverage_mask=np.ones(values.shape, dtype=bool),
        domain_mask=np.ones(values.shape, dtype=bool),
    )
    sampler = ProjectedSizeSampler(field, prepared.projection)
    lonlat = np.asarray([[-76.08, 43.22], [-76.06, 43.225]])
    xy = project_points(lonlat, prepared.projection)
    sampled = sampler.sample_xy(xy)
    expected = 1_000.0 + 10.0 * (lonlat[:, 0] - lon_grid.min()) + 5.0 * (
        lonlat[:, 1] - lat_grid.min()
    )
    assert np.allclose(sampled, expected, rtol=0.0, atol=1.0e-7)
    assert np.isclose(sampler(float(xy[0, 0]), float(xy[0, 1])), expected[0])
    outside = project_points(np.asarray([[-75.0, 43.22]]), prepared.projection)
    _expect_raises(
        ValueError,
        lambda: sampler.sample_xy(outside),
        "strict coverage",
    )


def test_boundary_trace_sampler_exact_trace_and_normal_release() -> None:
    prepared = _synthetic_case()
    boundary, _ = _load_canonical_boundary(prepared, PortfolioCaseConfig())
    square_xy = np.asarray(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
        ],
        dtype=float,
    )
    fine_boundary = replace(
        boundary,
        xy=square_xy,
        target_spacing_m=np.full(4, 10.0, dtype=float),
    )

    class ConstantBaseSampler:
        def sample_xy(self, values_xy):
            values = np.asarray(values_xy, dtype=float)
            return np.full(len(values), 100.0, dtype=float)

    sampler = BoundaryTraceSizeSampler(
        ConstantBaseSampler(),
        fine_boundary,
        gradation=0.20,
        samples_per_target=4.0,
        nearest_sample_count=16,
    )
    midpoints = 0.5 * (
        square_xy + np.roll(square_xy, -1, axis=0)
    )
    trace_values = sampler.sample_xy(np.vstack([square_xy, midpoints]))
    assert np.allclose(trace_values, 10.0, rtol=0.0, atol=1.0e-12)

    normal_distance = np.asarray(
        [0.0, 50.0, 200.0, 450.0, 600.0],
        dtype=float,
    )
    outward_normal_points = np.column_stack(
        [
            np.full(len(normal_distance), 50.0),
            -normal_distance,
        ]
    )
    released = sampler.sample_xy(outward_normal_points)
    expected = np.minimum(100.0, 10.0 + 0.20 * normal_distance)
    assert np.allclose(released, expected, rtol=0.0, atol=1.0e-12)
    assert np.all(np.diff(released) >= -1.0e-12)
    assert released[-1] == 100.0

    assert sampler.report["endpoint_midpoint_exact_by_construction"]
    assert sampler.report["distance_metric"] == "straight_euclidean"
    assert sampler.report["barrier_aware"] is False

    # Operational counters are candidate-local: preflight and candidate order
    # cannot change another candidate's immutable sampling evidence.
    sampler.reset_operational_counters()
    sampler.sample_xy(outward_normal_points[:2])
    first_order_a = {
        key: sampler.report[key]
        for key in (
            "sample_query_count",
            "expanded_sample_query_count",
            "maximum_neighbors_examined",
        )
    }
    sampler.reset_operational_counters()
    sampler.sample_xy(outward_normal_points[2:])
    first_order_b = {
        key: sampler.report[key]
        for key in (
            "sample_query_count",
            "expanded_sample_query_count",
            "maximum_neighbors_examined",
        )
    }
    sampler.reset_operational_counters()
    sampler.sample_xy(outward_normal_points[2:])
    single_candidate_b = {
        key: sampler.report[key]
        for key in (
            "sample_query_count",
            "expanded_sample_query_count",
            "maximum_neighbors_examined",
        )
    }
    sampler.reset_operational_counters()
    sampler.sample_xy(outward_normal_points[:2])
    reversed_order_a = {
        key: sampler.report[key]
        for key in (
            "sample_query_count",
            "expanded_sample_query_count",
            "maximum_neighbors_examined",
        )
    }
    assert first_order_a == reversed_order_a
    assert first_order_b == single_candidate_b


def test_boundary_trace_adaptive_search_finds_distant_fine_source() -> None:
    prepared = _synthetic_case()
    boundary, _ = _load_canonical_boundary(prepared, PortfolioCaseConfig())
    square_xy = np.asarray(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
        ],
        dtype=float,
    )
    mixed = replace(
        boundary,
        xy=square_xy,
        target_spacing_m=np.asarray(
            [1.0, 100.0, 100.0, 100.0],
            dtype=float,
        ),
    )

    class ConstantBaseSampler:
        def sample_xy(self, values_xy):
            return np.full(len(np.asarray(values_xy)), 1_000.0)

    sampler = BoundaryTraceSizeSampler(
        ConstantBaseSampler(),
        mixed,
        gradation=0.01,
        samples_per_target=4.0,
        nearest_sample_count=1,
    )
    query = np.asarray([[100.0, 50.0]], dtype=float)
    actual = float(sampler.sample_xy(query)[0])
    brute_force = float(
        np.min(
            sampler._targets
            + 0.01
            * np.linalg.norm(
                sampler._points - query[0],
                axis=1,
            )
        )
    )
    assert np.isclose(actual, brute_force, rtol=0.0, atol=1.0e-12)
    assert sampler.report["expanded_sample_query_count"] > 0
    assert sampler.report["maximum_neighbors_examined"] > 1
    assert sampler.report["exact_discrete_trace_sample_minimum"]


def test_boundary_trace_query_chunking_preserves_values_and_counters() -> None:
    prepared = _synthetic_case()
    boundary, _ = _load_canonical_boundary(prepared, PortfolioCaseConfig())
    mixed = replace(
        boundary,
        xy=np.asarray(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 100.0],
            ],
            dtype=float,
        ),
        target_spacing_m=np.asarray(
            [1.0, 100.0, 100.0, 100.0],
            dtype=float,
        ),
    )

    class ConstantBaseSampler:
        def sample_xy(self, values_xy):
            return np.full(len(np.asarray(values_xy)), 1_000.0)

    common = {
        "gradation": 0.01,
        "samples_per_target": 4.0,
        "nearest_sample_count": 1,
    }
    chunked = BoundaryTraceSizeSampler(
        ConstantBaseSampler(),
        mixed,
        query_chunk_size=2,
        **common,
    )
    unchunked = BoundaryTraceSizeSampler(
        ConstantBaseSampler(),
        mixed,
        query_chunk_size=1_000,
        **common,
    )
    query = np.column_stack(
        [
            np.linspace(5.0, 95.0, 11),
            np.linspace(95.0, 5.0, 11),
        ]
    )
    chunked_values = chunked.sample_xy(query)
    unchunked_values = unchunked.sample_xy(query)
    assert np.allclose(
        chunked_values,
        unchunked_values,
        rtol=0.0,
        atol=1.0e-12,
    )
    for key in (
        "sample_query_count",
        "expanded_sample_query_count",
        "maximum_neighbors_examined",
    ):
        assert chunked.report[key] == unchunked.report[key]
    assert chunked.report["query_chunk_size"] == 2
    assert chunked.report["memory_bounded_query_chunks"]


def test_boundary_trace_metric_sampling_bounds_extreme_endpoint_ratio() -> None:
    prepared = _synthetic_case()
    boundary, _ = _load_canonical_boundary(
        prepared,
        PortfolioCaseConfig(),
    )
    extreme = replace(
        boundary,
        xy=np.asarray(
            [
                [0.0, 0.0],
                [1_000_000.0, 0.0],
                [1_000_000.0, 1_000_000.0],
                [0.0, 1_000_000.0],
            ],
            dtype=float,
        ),
        target_spacing_m=np.asarray(
            [1.0, 1_000_000.0, 1_000_000.0, 1_000_000.0],
            dtype=float,
        ),
    )

    class ConstantBaseSampler:
        def sample_xy(self, values_xy):
            return np.full(len(np.asarray(values_xy)), 2_000_000.0)

    sampler = BoundaryTraceSizeSampler(
        ConstantBaseSampler(),
        extreme,
        gradation=0.10,
        samples_per_target=4.0,
        nearest_sample_count=16,
        maximum_total_sample_count=1_000,
    )
    assert sampler.report["boundary_sample_count"] < 1_000
    assert sampler.report["sample_distribution"] == (
        "linear_endpoint_target_metric_equidistribution"
    )
    _expect_raises(
        ValueError,
        lambda: BoundaryTraceSizeSampler(
            ConstantBaseSampler(),
            extreme,
            gradation=0.10,
            samples_per_target=4.0,
            nearest_sample_count=16,
            maximum_total_sample_count=10,
        ),
        "safety limit",
    )


def test_boundary_trace_single_edge_cap_fails_before_large_allocation() -> None:
    prepared = _synthetic_case()
    boundary, _ = _load_canonical_boundary(
        prepared,
        PortfolioCaseConfig(),
    )
    pathological = replace(
        boundary,
        xy=np.asarray(
            [
                [0.0, 0.0],
                [1.0e12, 0.0],
                [1.0e12, 1.0],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
        target_spacing_m=np.ones(4, dtype=float),
    )

    class ConstantBaseSampler:
        def sample_xy(self, values_xy):
            return np.full(len(np.asarray(values_xy)), 2.0e12)

    _expect_raises(
        ValueError,
        lambda: BoundaryTraceSizeSampler(
            ConstantBaseSampler(),
            pathological,
            gradation=0.10,
            samples_per_target=4.0,
            nearest_sample_count=16,
            maximum_total_sample_count=1_000,
        ),
        "before allocation",
    )


def test_sampler_adjusted_preflight_counts_trace_refinement() -> None:
    prepared = _synthetic_case()
    boundary, _ = _load_canonical_boundary(prepared, PortfolioCaseConfig())
    traced_boundary = replace(
        boundary,
        target_spacing_m=np.full(len(boundary.xy), 100.0, dtype=float),
    )
    shape = prepared.bathymetry.depth.shape
    raster = np.full(shape, 1_000.0, dtype=float)

    class ConstantBaseSampler:
        def sample_xy(self, values_xy):
            return np.full(len(np.asarray(values_xy)), 1_000.0)

    sampler = BoundaryTraceSizeSampler(
        ConstantBaseSampler(),
        traced_boundary,
        gradation=0.15,
        samples_per_target=4.0,
        nearest_sample_count=2,
    )
    field = SimpleNamespace(
        lon=prepared.bathymetry.lon,
        lat=prepared.bathymetry.lat,
        size=raster,
        coverage_mask=np.ones(shape, dtype=bool),
        domain_mask=np.ones(shape, dtype=bool),
    )
    base = estimate_node_budget(
        field.lon,
        field.lat,
        field.size,
        coverage_mask=field.coverage_mask,
        domain_mask=field.domain_mask,
    )
    field.report = {
        "schema_version": "fvcom_size_field_v4",
        "node_budget_estimate": base,
    }
    adjusted = _sampler_adjusted_node_budget(
        field,
        traced_boundary,
        sampler,
        chunk_size=100,
    )
    assert adjusted["trace_reduced_cell_count"] > 0
    assert adjusted["estimated_interior_node_count"] > (
        base["estimated_interior_node_count"]
    )
    assert adjusted["callback_schema"] == (
        "fvcom_boundary_trace_sampler_v2"
    )
    assert adjusted["schema_version"] == (
        "fvcom_sampler_adjusted_node_budget_v3"
    )
    assert adjusted["trace_lipschitz_lower_bound_applied"]
    assert adjusted["inactive_cells_excluded_from_raster_stencil"]
    assert adjusted["trace_cell_radius_upper_bound_m"] > 0.0
    pilot = _preflight_report(
        field,
        traced_boundary,
        PortfolioCaseConfig(
            preflight_node_limit=1_000_000,
            hard_node_limit=1_000_001,
        ),
        sampler=sampler,
    )
    callback_total = int(pilot["estimated_total_node_count"])
    raster_total = callback_total - (
        int(pilot["final_callback_estimated_interior_node_count"])
        - int(pilot["raster_only_estimated_interior_node_count"])
    )
    assert callback_total > raster_total
    separating_limit = (callback_total + raster_total) // 2
    gated = _preflight_report(
        field,
        traced_boundary,
        PortfolioCaseConfig(
            preflight_node_limit=separating_limit,
            hard_node_limit=max(separating_limit + 1, 1_000_000),
        ),
        sampler=sampler,
    )
    assert raster_total <= separating_limit
    assert gated["estimated_total_node_count"] > separating_limit
    assert gated["passed"] is False


def test_boundary_trace_is_authoritative_without_wet_support() -> None:
    prepared = _synthetic_case()
    boundary, _ = _load_canonical_boundary(
        prepared,
        PortfolioCaseConfig(),
    )
    traced = replace(
        boundary,
        target_spacing_m=np.full(len(boundary.xy), 8_000.0),
    )

    class UnsupportedFineBase:
        report = {
            "schema_version": "fvcom_projected_size_sampler_v2"
        }

        def sample_xy_with_active_support(self, values_xy):
            count = len(np.asarray(values_xy))
            return np.full(count, 1_375.0), np.zeros(count, dtype=bool)

        def sample_xy(self, values_xy):
            return self.sample_xy_with_active_support(values_xy)[0]

    sampler = BoundaryTraceSizeSampler(
        UnsupportedFineBase(),
        traced,
        gradation=0.10,
        samples_per_target=4.0,
        nearest_sample_count=2,
    )
    delivered = sampler.sample_xy(traced.xy[[0]])
    assert np.isclose(delivered[0], 8_000.0)
    assert sampler.report[
        "base_query_without_positive_active_support_count"
    ] == 1


def test_preflight_stencil_excludes_inactive_dry_values() -> None:
    prepared = _synthetic_case()
    boundary, _ = _load_canonical_boundary(
        prepared,
        PortfolioCaseConfig(),
    )
    shape = prepared.bathymetry.depth.shape
    raster = np.full(shape, 100.0, dtype=float)
    active = np.zeros(shape, dtype=bool)
    active[9:12, 11:14] = True
    raster[active] = 1_000.0
    field = SimpleNamespace(
        lon=prepared.bathymetry.lon,
        lat=prepared.bathymetry.lat,
        size=raster,
        coverage_mask=np.ones(shape, dtype=bool),
        domain_mask=active,
        report={},
    )

    class ConstantActiveSampler:
        def sample_xy(self, values_xy):
            return np.full(len(np.asarray(values_xy)), 1_000.0)

    base = estimate_node_budget(
        field.lon,
        field.lat,
        field.size,
        coverage_mask=field.coverage_mask,
        domain_mask=field.domain_mask,
    )
    adjusted = _sampler_adjusted_node_budget(
        field,
        boundary,
        ConstantActiveSampler(),
    )
    assert adjusted["estimated_interior_node_count"] == (
        base["estimated_interior_node_count"]
    )
    assert adjusted["stencil_reduced_cell_count"] == 0


def test_trace_lipschitz_preflight_bounds_dense_subcell_quadrature() -> None:
    prepared = _synthetic_case()
    boundary, _ = _load_canonical_boundary(
        prepared,
        PortfolioCaseConfig(),
    )
    traced = replace(
        boundary,
        # Keep the trace fine enough relative to this fixture's raster-cell
        # radius to force the adaptive subcell integration branch.
        target_spacing_m=np.full(len(boundary.xy), 10.0),
    )
    shape = prepared.bathymetry.depth.shape
    raster = np.full(shape, 1_000.0, dtype=float)
    active = np.zeros(shape, dtype=bool)
    active[1:-1, 1:-1] = True
    field = SimpleNamespace(
        lon=prepared.bathymetry.lon,
        lat=prepared.bathymetry.lat,
        size=raster,
        coverage_mask=np.ones(shape, dtype=bool),
        domain_mask=active,
        report={},
    )

    class ConstantBaseSampler:
        def sample_xy(self, values_xy):
            return np.full(len(np.asarray(values_xy)), 1_000.0)

    sampler = BoundaryTraceSizeSampler(
        ConstantBaseSampler(),
        traced,
        gradation=0.10,
        samples_per_target=4.0,
        nearest_sample_count=2,
    )
    conservative = _sampler_adjusted_node_budget(
        field,
        traced,
        sampler,
    )
    assert conservative["trace_subcell_query_count"] > 0
    assert any(
        int(subdivisions) > 1 and int(cell_count) > 0
        for subdivisions, cell_count in conservative[
            "trace_subdivision_count_by_axis"
        ].items()
    )

    fractions = np.linspace(-0.5, 0.5, 9)
    queries: list[list[float]] = []
    cell_ids: list[int] = []
    lon_width = np.gradient(np.asarray(field.lon, dtype=float))
    lat_width = np.gradient(np.asarray(field.lat, dtype=float))
    for row, column in np.argwhere(active):
        cell_id = int(row * shape[1] + column)
        for fy in fractions:
            for fx in fractions:
                queries.append(
                    [
                        float(field.lon[column] + fx * lon_width[column]),
                        float(field.lat[row] + fy * lat_width[row]),
                    ]
                )
                cell_ids.append(cell_id)
    query_xy = project_points(
        np.asarray(queries, dtype=float),
        traced.projection,
    )
    dense_h = sampler.sample_xy(query_xy)
    inverse_square_sum = np.bincount(
        np.asarray(cell_ids, dtype=int),
        weights=1.0 / np.square(dense_h),
        minlength=int(np.prod(shape)),
    ).reshape(shape)
    counts = np.bincount(
        np.asarray(cell_ids, dtype=int),
        minlength=int(np.prod(shape)),
    ).reshape(shape)
    equivalent_size = raster.copy()
    equivalent_size[active] = 1.0 / np.sqrt(
        inverse_square_sum[active] / counts[active]
    )
    dense = estimate_node_budget(
        field.lon,
        field.lat,
        equivalent_size,
        coverage_mask=field.coverage_mask,
        domain_mask=field.domain_mask,
    )
    assert conservative["estimated_interior_node_count_float"] >= (
        dense["estimated_interior_node_count_float"] * (1.0 - 1.0e-12)
    )


def test_raw_comparison_exports_realized_boundary_continuity() -> None:
    with tempfile.TemporaryDirectory(
        prefix="portfolio_comparison_continuity_"
    ) as temporary:
        output = Path(temporary)
        quality_path = output / "quality.json"
        quality_path.write_text(
            json.dumps(
                {
                    "accepted": False,
                    "edge_aware_size_error": {
                        "boundary_field_interface": {
                            "factor_two_exceedance_count": 2,
                        },
                        "boundary_first_ring_realized_continuity": {
                            "passed": False,
                            "global": {
                                "symmetric_ratio": {
                                    "quantiles": {"p95": 1.61},
                                    "maximum": 2.25,
                                },
                                "chain_p95_exceedance_count": 3,
                                "chain_maximum_exceedance_count": 1,
                            },
                        },
                    },
                    "failure_taxonomy": [
                        "boundary_first_ring_continuity_p95_exceeded"
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "candidate_id": "gmsh_frontal_delaunay_6",
            "status": "needs_review",
            "artifacts": {
                "quality_json": {
                    "path": str(quality_path),
                }
            },
        }
        routing = {
            "candidates": {
                "gmsh_frontal_delaunay_6": {
                    "supported": True,
                    "reasons": [],
                }
            }
        }
        _write_raw_metric_comparison(
            output,
            [manifest],
            routing,
            "fixture_bundle_sha256",
        )
        payload = json.loads(
            (output / "raw_metric_comparison.json").read_text(
                encoding="utf-8"
            )
        )
        row = payload["supported_candidate_metrics"][0]
        assert row[
            "boundary_field_interface_ratio_limit_exceedance_count"
        ] == 2
        assert row["boundary_first_ring_continuity_passed"] is False
        assert row["boundary_first_ring_continuity_p95"] == 1.61
        assert row["boundary_first_ring_continuity_maximum"] == 2.25
        assert row["boundary_first_ring_chain_p95_exceedance_count"] == 3
        assert row[
            "boundary_first_ring_chain_maximum_exceedance_count"
        ] == 1
        csv_header = (output / "raw_metric_comparison.csv").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        assert "boundary_first_ring_continuity_p95" in csv_header


def test_scientific_bundle_hash_excludes_paths_and_measurement_detail() -> None:
    config = PortfolioCaseConfig(size_field_max_cells=40_000)
    field_config = _size_field_config(config)
    base_preflight = {
        "canonical_size_field_schema": "fvcom_size_field_v4",
        "estimated_interior_node_count": 100,
        "explicit_source_boundary_node_count": 20,
        "gmsh_measured_boundary_node_count": 20,
        "common_boundary_node_count": 20,
        "boundary_front_seed_count": 5,
        "estimated_total_node_count": 125,
        "preflight_node_limit": 900_000,
        "hard_node_limit": 1_000_000,
        "passed": True,
    }
    first, _ = _scientific_bundle_sha256(
        case_id="case",
        source_hashes={
            "bathymetry": {
                "path": "C:/first/location.nc",
                "sha256": "a" * 64,
                "bytes": 1,
            }
        },
        canonical_boundary_sha256="c" * 64,
        canonical_field_sha256="b" * 64,
        projection_epsg=32618,
        portfolio_config=config,
        size_field_config=field_config,
        preflight=base_preflight,
    )
    changed_detail = dict(base_preflight)
    changed_detail["gmsh_measured_boundary_node_count"] = None
    second, _ = _scientific_bundle_sha256(
        case_id="case",
        source_hashes={
            "bathymetry": {
                "path": "D:/different/location.nc",
                "sha256": "a" * 64,
                "bytes": 999,
            }
        },
        canonical_boundary_sha256="c" * 64,
        canonical_field_sha256="b" * 64,
        projection_epsg=32618,
        portfolio_config=config,
        size_field_config=field_config,
        preflight=changed_detail,
    )
    assert first == second
    changed_boundary, _ = _scientific_bundle_sha256(
        case_id="case",
        source_hashes={
            "bathymetry": {
                "path": "D:/different/location.nc",
                "sha256": "a" * 64,
                "bytes": 999,
            }
        },
        canonical_boundary_sha256="d" * 64,
        canonical_field_sha256="b" * 64,
        projection_epsg=32618,
        portfolio_config=config,
        size_field_config=field_config,
        preflight=changed_detail,
    )
    assert first != changed_boundary
    minimum_mode, _ = _scientific_bundle_sha256(
        case_id="case",
        source_hashes={
            "bathymetry": {
                "path": "C:/first/location.nc",
                "sha256": "a" * 64,
                "bytes": 1,
            }
        },
        canonical_boundary_sha256="c" * 64,
        canonical_field_sha256="b" * 64,
        projection_epsg=32618,
        portfolio_config=replace(
            config,
            boundary_target_combination="minimum",
        ),
        size_field_config=field_config,
        preflight=base_preflight,
    )
    assert first != minimum_mode


def test_zero_and_plural_nodestring_roundtrip() -> None:
    prepared = _synthetic_case()
    nodes_lonlat = np.asarray(prepared.source_domain_lonlat.exterior.coords[:-1])
    nodes_xy = project_points(nodes_lonlat, prepared.projection)
    base = dict(
        nodes_xy=nodes_xy,
        nodes_lonlat=nodes_lonlat,
        triangles_1based=np.asarray([[1, 2, 3], [1, 3, 4]], dtype=int),
        depths=np.full(4, 12.0),
        constraint_chains_zero=[[0, 1, 2, 3]],
        open_boundary_cyclic=[],
        constraint_report={"boundary_constraint_recovered": True},
        boundary_metadata={
            "all_source_vertices_retained": True,
            "hard_anchors_retained": True,
        },
        generator_report={},
        extra_quality={},
    )
    with tempfile.TemporaryDirectory(prefix="portfolio_roundtrip_") as temporary:
        root = Path(temporary)
        closed_mesh = _CandidateMesh(
            open_boundary_chains_1based=[],
            **base,
        )
        closed = _roundtrip_report(
            root / "closed.2dm",
            prepared,
            closed_mesh,
            "closed",
        )
        assert closed["passed"]
        assert closed["open_boundary_chain_count"] == 0
        assert closed["nodestring_ids"] == []

        plural_prepared = SimpleNamespace(
            manifest=prepared.manifest,
            projection=prepared.projection,
            open_boundaries=(object(), object()),
        )
        plural_mesh = _CandidateMesh(
            open_boundary_chains_1based=[[1, 2], [3, 4]],
            **{**base, "open_boundary_cyclic": [False, False]},
        )
        plural = _roundtrip_report(
            root / "plural.2dm",
            plural_prepared,
            plural_mesh,
            "plural",
        )
        assert plural["passed"]
        assert plural["open_boundary_chain_count"] == 2
        assert plural["nodestring_ids"] == [1, 2]


def test_real_small_gmsh_candidate_samples_immutable_bathymetry() -> None:
    try:
        import gmsh
    except Exception:
        print("SKIP real Gmsh candidate: gmsh is unavailable")
        return
    if str(gmsh.__version__) != "4.15.2":
        print(f"SKIP real Gmsh candidate: found gmsh {gmsh.__version__}")
        return
    prepared = _synthetic_case()
    field = _uniform_size_field(prepared)
    sampler = ProjectedSizeSampler(
        field,
        prepared.projection,
    )
    boundary, _ = _load_canonical_boundary(
        prepared,
        PortfolioCaseConfig(),
    )
    with tempfile.TemporaryDirectory(prefix="portfolio_real_gmsh_") as temporary:
        root = Path(temporary)
        mesh = _run_gmsh_candidate(
            "gmsh_frontal_delaunay_6",
            prepared,
            sampler,
            root,
        )
        assert (root / "raw_mesh.msh").is_file()
        assert mesh.generator_report["algorithm"] == 6
        assert mesh.generator_report["algorithm_name"] == "Frontal-Delaunay"
        assert mesh.generator_report["boundary_node_count_1d"] == 4
        assert mesh.generator_report["boundary_discretization_mode"] == (
            "preserve_source_segments_two_endpoints"
        )
        assert mesh.generator_report["source_boundary_vertex_count"] == 4
        assert mesh.generator_report["delivered_boundary_node_count"] == 4
        assert mesh.boundary_metadata[
            "boundary_discretization_matched_to_source"
        ]
        assert mesh.open_boundary_chains_1based == []
        assert np.allclose(mesh.depths, 12.0, rtol=0.0, atol=1.0e-10)
        report = _roundtrip_report(
            root / "synthetic.2dm",
            prepared,
            mesh,
            "gmsh_frontal_delaunay_6",
        )
        assert report["passed"]
        assert report["open_boundary_chain_count"] == 0

        import fvcom_grid_generation.portfolio_case as portfolio_case_module

        original_runner = portfolio_case_module._run_gmsh_candidate
        portfolio_case_module._run_gmsh_candidate = (
            lambda *_args, **_kwargs: mesh
        )
        try:
            manifest = _execute_candidate(
                "gmsh_frontal_delaunay_6",
                root,
                prepared,
                boundary,
                field,
                sampler,
                PortfolioCaseConfig(),
                "fixture_bundle_sha256",
            )
        finally:
            portfolio_case_module._run_gmsh_candidate = original_runner
        delivered_artifact = manifest["artifacts"]["node_budget_delivered"]
        delivered_path = Path(delivered_artifact["path"])
        assert delivered_path.name == "node_budget_delivered.json"
        assert delivered_path.is_file()
        assert delivered_artifact["sha256"] == hashlib.sha256(
            delivered_path.read_bytes()
        ).hexdigest()
        delivered = json.loads(delivered_path.read_text(encoding="utf-8"))
        assert delivered["passed"]
        quality = json.loads(
            Path(manifest["artifacts"]["quality_json"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        assert quality["canonical_size_sampling"]["schema_version"] == (
            "fvcom_projected_size_sampler_v2"
        )
        assert quality["canonical_size_sampling"][
            "sampling_interface_schema_version"
        ] == "fvcom_wet_mask_sampling_v2"


def test_gmsh_common_locked_boundary_mismatch_is_rejected() -> None:
    from fvcom_grid_generation import gmsh_backend

    prepared = _synthetic_case()
    boundary, _ = _load_canonical_boundary(prepared, PortfolioCaseConfig())
    sampler = ProjectedSizeSampler(
        _uniform_size_field(prepared),
        prepared.projection,
    )
    lineage = tuple(
        SimpleNamespace(
            mesh_node_id=index + 1,
            gmsh_node_tag=index + 1,
            loop_id=f"{prepared.manifest['case_id']}:exterior",
            source_segment_index=index,
            interpolation_weight=0.0,
            loop_normalized_arclength=float(index) / len(boundary.xy),
            chain_normalized_arclength=None,
            is_source_vertex=True,
        )
        for index in range(len(boundary.xy))
    )
    delivered_loop = SimpleNamespace(
        loop_id=f"{prepared.manifest['case_id']}:exterior",
        role="exterior",
        island_id=None,
        source_orientation="counterclockwise",
        node_ids=tuple(range(1, len(boundary.xy) + 1)),
        lineage=lineage,
    )
    fake_result = SimpleNamespace(
        nodes_xy=np.asarray(boundary.xy, dtype=float),
        delivered_loops=(delivered_loop,),
        open_boundaries=(),
        source_vertex_node_ids={
            f"{prepared.manifest['case_id']}:exterior:{index}": index + 1
            for index in range(len(boundary.xy))
        },
        boundary_node_count_1d=len(boundary.xy) + 1,
        boundary_discretization_mode="preserve_source_segments_two_endpoints",
    )
    original = gmsh_backend.run_gmsh_attempt
    gmsh_backend.run_gmsh_attempt = lambda *_args, **_kwargs: fake_result
    try:
        with tempfile.TemporaryDirectory(
            prefix="portfolio_common_locked_guard_"
        ) as temporary:
            _expect_raises(
                ValueError,
                lambda: _run_gmsh_candidate(
                    "gmsh_frontal_delaunay_6",
                    prepared,
                    sampler,
                    Path(temporary),
                    boundary=boundary,
                ),
                "COMMON_LOCKED boundary invariant failed",
            )
    finally:
        gmsh_backend.run_gmsh_attempt = original


def test_real_small_clean_room_candidate_uses_same_raw_contract() -> None:
    prepared = _synthetic_case()
    config = PortfolioCaseConfig(
        size_field_max_cells=10_000,
        land_spacing_m=1_000.0,
        open_spacing_m=1_000.0,
        maximum_size_m=2_000.0,
        clean_room_refine_iterations=0,
        clean_room_smooth_iterations=0,
    )
    boundary, _report = _load_canonical_boundary(prepared, config)
    field = build_size_field(
        prepared.bathymetry,
        boundary,
        _size_field_config(config),
    )
    sampler = BoundaryTraceSizeSampler(
        ProjectedSizeSampler(field, boundary.projection),
        boundary,
        gradation=config.gradation,
    )
    mesh = _run_clean_room_candidate(
        prepared,
        boundary,
        field,
        config,
        sampler=sampler,
    )
    assert mesh.generator_report["backend"] == "scipy_delaunay_clean_room"
    assert mesh.generator_report["boundary_discretization_mode"] == (
        "constraint_midpoint_recovery"
    )
    assert mesh.generator_report["source_boundary_vertex_count"] == 4
    assert mesh.generator_report["delivered_boundary_node_count"] >= 4
    assert (
        mesh.boundary_metadata["boundary_discretization_matched_to_source"]
        == (
            mesh.generator_report["delivered_boundary_node_count"]
            == mesh.generator_report["source_boundary_vertex_count"]
        )
    )
    assert mesh.generator_report["raw_stage"]
    assert not mesh.generator_report["common_conditioning_applied"]
    assert mesh.generator_report["canonical_size_callback_used"]
    assert (
        mesh.generator_report["mesh_report"]["size_sampling_mode"]
        == "projected_callback"
    )
    assert mesh.open_boundary_chains_1based == []
    assert np.allclose(mesh.depths, 12.0, rtol=0.0, atol=1.0e-10)
    with tempfile.TemporaryDirectory(
        prefix="portfolio_real_clean_room_"
    ) as temporary:
        report = _roundtrip_report(
            Path(temporary) / "synthetic_clean.2dm",
            prepared,
            mesh,
            "clean_room_raw",
        )
    assert report["passed"]
    assert report["open_boundary_chain_count"] == 0


def test_common_boundary_field_fixed_point_and_gmsh_rebase() -> None:
    prepared = _synthetic_case()
    config = PortfolioCaseConfig(
        size_field_max_cells=10_000,
        land_spacing_m=1_000.0,
        open_spacing_m=1_000.0,
        maximum_size_m=1_000.0,
        boundary_reconciliation_max_iterations=5,
        clean_room_refine_iterations=0,
        clean_room_smooth_iterations=0,
    )
    source, _ = _load_canonical_boundary(prepared, config)
    delivered, field, sampler, report = _reconcile_boundary_and_size_field(
        prepared.bathymetry,
        source,
        _size_field_config(config),
        config,
    )
    assert report["passed"], report
    assert report["converged_iteration"] is not None
    assert len(delivered.xy) > len(source.xy)
    assert report["final_boundary_field_audit"]["passed"]
    assert field.report["schema_version"] == "fvcom_size_field_v4"
    frozen_reconciliation_sampler = json.loads(
        json.dumps(report["boundary_trace_sampler"])
    )
    assert np.all(np.isfinite(sampler.sample_xy(delivered.xy)))
    sampler.reset_operational_counters()
    sampler.sample_xy(delivered.xy[:2])
    assert report["boundary_trace_sampler"] == frozen_reconciliation_sampler
    rebased = _prepared_case_on_boundary(prepared, delivered)
    assert len(rebased.exterior_xy) == len(delivered.constraint_chains[0])
    assert len(rebased.holes_xy) == len(delivered.constraint_chains) - 1
    assert rebased.open_boundaries == ()


def test_case_budget_reconciliation_follows_spatial_field_without_jump() -> None:
    import fvcom_grid_generation.portfolio_case as portfolio_case_module

    prepared = _synthetic_case()
    config = PortfolioCaseConfig(
        size_field_max_cells=10_000,
        land_spacing_m=100.0,
        open_spacing_m=100.0,
        maximum_size_m=2_000.0,
        boundary_reconciliation_max_iterations=3,
        boundary_geometry_continuity=False,
        use_case_budget_spacing_policy=True,
    )
    source, _ = _load_canonical_boundary(prepared, config)
    lon_grid, lat_grid = np.meshgrid(
        prepared.bathymetry.lon,
        prepared.bathymetry.lat,
    )
    lon_fraction = (
        (lon_grid - float(np.min(lon_grid)))
        / float(np.ptp(lon_grid))
    )
    lat_fraction = (
        (lat_grid - float(np.min(lat_grid)))
        / float(np.ptp(lat_grid))
    )
    spatial_size = 800.0 + 300.0 * lon_fraction + 100.0 * lat_fraction

    def build_spatial_field(*_args, **_kwargs):
        return SimpleNamespace(
            lon=prepared.bathymetry.lon,
            lat=prepared.bathymetry.lat,
            size=spatial_size,
            coverage_mask=np.ones(spatial_size.shape, dtype=bool),
            report={
                "schema_version": "fvcom_size_field_v4",
                "node_budget_estimate": {},
                "sampling_interface": {
                    "schema_version": "fvcom_wet_mask_sampling_v2",
                    "method": (
                        "bilinear_all_positive_weight_active_else_"
                        "highest_weight_positive_active_else_coarsest_covered"
                    ),
                },
            },
        )

    original_builder = portfolio_case_module.build_size_field
    portfolio_case_module.build_size_field = build_spatial_field
    try:
        delivered, _field, sampler, report = (
            _reconcile_boundary_and_size_field(
                prepared.bathymetry,
                source,
                _size_field_config(config),
                config,
            )
        )
    finally:
        portfolio_case_module.build_size_field = original_builder

    source_field_values = sampler.sample_xy(source.xy)
    assert isinstance(sampler, ProjectedSizeSampler)
    assert sampler.report["sampling_interface_schema_version"] == (
        "fvcom_wet_mask_sampling_v2"
    )
    assert report["boundary_trace_sampler"] is None
    assert "explicitly disabled" in report["method_scope"]
    assert np.all(source.target_spacing_m < source_field_values)
    assert report["passed"], report
    assert report["policy"]["target_combination"] == "sampled_field"

    inserted = np.asarray(
        delivered.metadata["reconciliation_inserted"],
        dtype=bool,
    )
    source_node = np.asarray(
        delivered.metadata[
            "reconciliation_source_node_index_zero_based"
        ],
        dtype=int,
    )
    original_output_indices: list[int] = []
    for source_index in range(len(source.xy)):
        matches = np.flatnonzero(
            (~inserted) & (source_node == source_index)
        )
        assert len(matches) == 1
        output_index = int(matches[0])
        original_output_indices.append(output_index)
        assert np.allclose(
            delivered.xy[output_index],
            source.xy[source_index],
            rtol=0.0,
            atol=1.0e-10,
        )
    assert np.all(
        delivered.target_spacing_m[
            np.asarray(original_output_indices, dtype=int)
        ]
        > source.target_spacing_m
    )

    final_audit = report["final_boundary_field_audit"]
    factor_audit = final_audit["factor_compatibility"]
    assert final_audit["passed"], final_audit
    assert factor_audit["passed"]
    assert factor_audit["incompatible_sample_count"] == 0
    assert factor_audit["endpoint_incompatible_count"] == 0
    assert factor_audit["midpoint_incompatible_count"] == 0
    ratio = final_audit["boundary_to_final_field_ratio"]
    assert ratio["minimum"] >= factor_audit["lower_ratio"] - 1.0e-12
    assert ratio["maximum"] <= factor_audit["upper_ratio"] + 1.0e-12


def _find_workspace_root() -> Path | None:
    for candidate in [Path.cwd(), *Path.cwd().parents, *SCRIPTS.parents]:
        if (candidate / "Workspace").is_dir() and (
            candidate / "Agent_skill_dev"
        ).is_dir():
            return candidate
    return None


def test_lake_ontario_adaptive_boundary_and_fresh_v4_smoke() -> None:
    workspace = _find_workspace_root()
    if workspace is None:
        print("SKIP Lake Ontario smoke: workspace root is unavailable")
        return
    case_manifest = (
        workspace
        / "Agent_skill_dev"
        / "skill_catalog"
        / "grid-generation"
        / "fvcom-grid-generation"
        / "scripts"
        / "research"
        / "gmsh"
        / "cases"
        / "05_lake_ontario.json"
    )
    if not case_manifest.exists():
        print("SKIP Lake Ontario smoke: case manifest is unavailable")
        return
    prepared = prepare_case(case_manifest, workspace)
    config = PortfolioCaseConfig(size_field_max_cells=40_000)
    boundary, report = _load_canonical_boundary(prepared, config)
    assert report["node_count"] == 661
    assert report["constraint_chain_count"] == 13
    assert report["open_boundary_count"] == 0
    assert boundary.open_boundaries == []
    size_bathy = coarsen_for_size_field(
        prepared.bathymetry,
        max_cells=40_000,
    )
    field = build_size_field(
        size_bathy,
        boundary,
        _size_field_config(config),
    )
    assert field.report["schema_version"] == "fvcom_size_field_v4"
    sampler = ProjectedSizeSampler(field, prepared.projection)
    values = sampler.sample_xy(boundary.xy)
    assert len(values) == 661
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


def test_existing_output_is_rejected_before_case_loading() -> None:
    with tempfile.TemporaryDirectory(prefix="portfolio_fresh_guard_") as temporary:
        root = Path(temporary)
        _expect_raises(
            FileExistsError,
            lambda: run_portfolio_case(
                root / "missing_case.json",
                root,
                root,
                candidate_ids=["clean-room"],
            ),
            "must not already exist",
        )


def test_windows_output_path_budget_fails_before_meshing() -> None:
    if os.name != "nt":
        return
    _expect_raises(
        ValueError,
        lambda: _validate_output_path_budget(
            Path("C:/") / ("long_portfolio_root_" * 14)
        ),
        "shorten --output-dir",
    )


TESTS: tuple[Callable[[], None], ...] = (
    test_candidate_normalization_and_capability_routing,
    test_source_obc_node_ordering,
    test_partial_forward_reverse_obc_rebase_and_segment_kind_mask,
    test_cyclic_obc_rebase_includes_last_to_first_segment,
    test_reconciliation_lineage_controls_forcing_compatibility,
    test_case_budget_targets_and_geometry_forced_edge_count,
    test_portfolio_config_reconciliation_validation,
    test_portfolio_cli_continuity_defaults,
    test_cached_projected_size_sampler,
    test_boundary_trace_sampler_exact_trace_and_normal_release,
    test_boundary_trace_adaptive_search_finds_distant_fine_source,
    test_boundary_trace_query_chunking_preserves_values_and_counters,
    test_boundary_trace_metric_sampling_bounds_extreme_endpoint_ratio,
    test_boundary_trace_single_edge_cap_fails_before_large_allocation,
    test_sampler_adjusted_preflight_counts_trace_refinement,
    test_boundary_trace_is_authoritative_without_wet_support,
    test_preflight_stencil_excludes_inactive_dry_values,
    test_trace_lipschitz_preflight_bounds_dense_subcell_quadrature,
    test_raw_comparison_exports_realized_boundary_continuity,
    test_scientific_bundle_hash_excludes_paths_and_measurement_detail,
    test_zero_and_plural_nodestring_roundtrip,
    test_real_small_gmsh_candidate_samples_immutable_bathymetry,
    test_gmsh_common_locked_boundary_mismatch_is_rejected,
    test_real_small_clean_room_candidate_uses_same_raw_contract,
    test_common_boundary_field_fixed_point_and_gmsh_rebase,
    test_case_budget_reconciliation_follows_spatial_field_without_jump,
    test_lake_ontario_adaptive_boundary_and_fresh_v4_smoke,
    test_existing_output_is_rejected_before_case_loading,
    test_windows_output_path_budget_fails_before_meshing,
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
        print(f"{len(failures)} of {len(TESTS)} portfolio case tests failed")
        return 1
    print(f"All {len(TESTS)} portfolio case tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
