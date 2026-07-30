#!/usr/bin/env python3
"""Focused checks for deterministic boundary/size-field reconciliation."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon

from fvcom_grid_generation.boundary import BoundaryNodes, OpenBoundaryChain
from fvcom_grid_generation.boundary_size_reconciliation import (
    BoundarySizeReconciliationConfig,
    audit_reconciled_boundary_size_field,
    reconcile_boundary_size_field,
)
from fvcom_grid_generation.projection import (
    local_utm_projection,
    unproject_points,
)


def _boundary(
    points: list[tuple[float, float]],
    *,
    kinds: list[str] | None = None,
    targets: list[float] | None = None,
    open_boundaries: list[OpenBoundaryChain] | None = None,
    hard: list[bool] | None = None,
) -> BoundaryNodes:
    xy = np.asarray(points, dtype=float)
    projection = local_utm_projection((-76.0, 42.0, -75.0, 43.0))
    polygon = Polygon(xy)
    kinds = kinds or ["land"] * len(points)
    targets = targets or [10.0] * len(points)
    open_indices = [
        int(node)
        for chain in (open_boundaries or [])
        for node in chain.node_indices
    ]
    return BoundaryNodes(
        xy=xy,
        lonlat=unproject_points(xy, projection),
        kinds=list(kinds),
        target_spacing_m=np.asarray(targets, dtype=float),
        exterior_indices=list(range(len(points))),
        open_boundary_indices=open_indices,
        constraint_chains=[list(range(len(points)))],
        domain_polygon_xy=polygon,
        open_boundary_xy=LineString(),
        land_boundary_xy=LineString(polygon.exterior.coords),
        island_polygons_xy=[],
        projection=projection,
        hard_anchor_mask=np.asarray(
            hard if hard is not None else [False] * len(points), dtype=bool
        ),
        adaptive_resolution=True,
        resolution_profile="adaptive-coastal-v2",
        metadata={
            "semantic_id": np.asarray(
                [f"node-{index}" for index in range(len(points))], dtype=object
            ),
            "is_hard_anchor": np.asarray(
                hard if hard is not None else [False] * len(points), dtype=bool
            ),
            "is_source_vertex": np.ones(len(points), dtype=bool),
            "source_vertex_index": np.arange(len(points), dtype=int),
            "outlet_context_anchor": np.asarray(
                hard if hard is not None else [False] * len(points), dtype=bool
            ),
        },
        open_boundaries=open_boundaries,
    )


def _constant(value: float):
    return lambda points: np.full(len(points), float(value), dtype=float)


def _assert_originals_exact(source: BoundaryNodes, delivered: BoundaryNodes) -> None:
    lineage = delivered.metadata[
        "reconciliation_source_node_index_zero_based"
    ]
    for source_index, source_point in enumerate(source.xy):
        matches = np.flatnonzero(lineage == source_index)
        assert len(matches) == 1
        assert np.array_equal(delivered.xy[matches[0]], source_point)
        assert (
            delivered.metadata["semantic_id"][matches[0]]
            == source.metadata["semantic_id"][source_index]
        )


def test_closed_loop_coarse_to_fine_and_exact_lineage() -> None:
    source = _boundary(
        [(0, 0), (40, 0), (40, 40), (0, 40)],
        targets=[20, 20, 20, 20],
        hard=[True, False, True, False],
    )
    result = reconcile_boundary_size_field(source, _constant(5.0))
    assert len(result.boundary.xy) == 32
    assert not result.boundary.open_boundaries
    assert result.audit["inserted_boundary_node_count"] == 28
    assert result.audit["factor_compatibility"]["passed"]
    assert result.audit["all_source_vertices_exact"]
    assert result.audit["hard_anchors_exact"]
    assert result.audit["passed"]
    _assert_originals_exact(source, result.boundary)
    inserted = result.boundary.metadata["reconciliation_inserted"]
    assert not np.any(result.boundary.hard_anchor_mask[inserted])
    for name in (
        "is_hard_anchor",
        "is_source_vertex",
        "source_vertex_index",
        "outlet_context_anchor",
        "semantic_id",
    ):
        assert all(value is None for value in result.boundary.metadata[name][inserted])


def test_single_and_multiple_obc_identity_order() -> None:
    points = [
        (0, 0),
        (20, 0),
        (40, 0),
        (40, 20),
        (40, 40),
        (20, 40),
        (0, 40),
        (0, 20),
    ]
    single = OpenBoundaryChain(
        chain_id="atlantic",
        node_indices=(0, 1, 2),
        kind="exchange",
        cyclic=False,
        orientation="forward",
    )
    one = _boundary(
        points,
        kinds=["open", "open", "open", "land", "land", "land", "land", "land"],
        open_boundaries=[single],
        hard=[True, False, True, False, False, False, False, False],
    )
    one_result = reconcile_boundary_size_field(one, _constant(5.0))
    rebuilt = one_result.boundary.open_boundaries[0]
    assert rebuilt.chain_id == "atlantic"
    assert rebuilt.kind == "exchange"
    assert not rebuilt.cyclic
    assert rebuilt.orientation == "forward"
    source_lineage = one_result.boundary.metadata[
        "reconciliation_source_node_index_zero_based"
    ][list(rebuilt.node_indices)]
    assert source_lineage[0] == 0 and source_lineage[-1] == 2
    assert np.array_equal(source_lineage[source_lineage >= 0], [0, 1, 2])
    original_output = {
        int(source_index): index
        for index, source_index in enumerate(
            one_result.boundary.metadata[
                "reconciliation_source_node_index_zero_based"
            ]
        )
        if source_index >= 0
    }
    assert [
        one_result.boundary.kinds[original_output[index]]
        for index in range(len(points))
    ] == one.kinds
    assert one_result.boundary.hard_anchor_mask[
        original_output[0]
    ] and one_result.boundary.hard_anchor_mask[original_output[2]]

    second = OpenBoundaryChain(
        chain_id="east_river",
        node_indices=(6, 5, 4),
        kind="exchange",
        cyclic=False,
        orientation="reverse",
    )
    multiple = _boundary(points, open_boundaries=[single, second])
    multiple_result = reconcile_boundary_size_field(
        multiple, _constant(5.0)
    )
    assert [
        chain.chain_id for chain in multiple_result.boundary.open_boundaries
    ] == ["atlantic", "east_river"]
    assert [
        chain.orientation
        for chain in multiple_result.boundary.open_boundaries
    ] == ["forward", "reverse"]
    assert len(set(multiple_result.boundary.open_boundary_indices)) == len(
        multiple_result.boundary.open_boundary_indices
    )


def test_cyclic_obc_omits_repeated_first_and_preserves_orientation() -> None:
    cyclic = OpenBoundaryChain(
        chain_id="hawaii_offshore",
        node_indices=(0, 1, 2, 3),
        kind="cyclic_offshore",
        cyclic=True,
        orientation="counterclockwise",
    )
    source = _boundary(
        [(0, 0), (20, 0), (20, 20), (0, 20)],
        kinds=["open"] * 4,
        open_boundaries=[cyclic],
    )
    result = reconcile_boundary_size_field(source, _constant(5.0))
    rebuilt = result.boundary.open_boundaries[0]
    assert rebuilt.cyclic
    assert rebuilt.orientation == "counterclockwise"
    assert rebuilt.node_indices[0] != rebuilt.node_indices[-1]
    assert tuple(rebuilt.node_indices) == tuple(
        result.boundary.constraint_chains[0]
    )


def test_deterministic_repeat_and_spatial_adaptation() -> None:
    source = _boundary(
        [(0, 0), (40, 0), (40, 20), (0, 20)],
        targets=[20, 20, 20, 20],
    )

    def sampler(points: np.ndarray) -> np.ndarray:
        return np.where(points[:, 0] <= 20.0, 4.0, 10.0)

    policy = BoundarySizeReconciliationConfig(sampler_id="step_fixture_v1")
    first = reconcile_boundary_size_field(source, sampler, config=policy)
    second = reconcile_boundary_size_field(source, sampler, config=policy)
    assert np.array_equal(first.boundary.xy, second.boundary.xy)
    assert np.array_equal(
        first.boundary.target_spacing_m, second.boundary.target_spacing_m
    )
    assert first.audit["reproducibility"] == second.audit["reproducibility"]
    assert (
        first.audit["reproducibility"]["sampler_id"] == "step_fixture_v1"
    )
    left_target = np.median(
        first.boundary.target_spacing_m[first.boundary.xy[:, 0] < 20.0]
    )
    right_target = np.median(
        first.boundary.target_spacing_m[first.boundary.xy[:, 0] > 20.0]
    )
    assert left_target < right_target
    _assert_originals_exact(source, first.boundary)


def test_sampled_field_mode_can_smoothly_coarsen_source_targets() -> None:
    source = _boundary(
        [(0, 0), (40, 0), (40, 40), (0, 40)],
        targets=[5, 5, 5, 5],
        hard=[True, False, True, False],
    )
    minimum = reconcile_boundary_size_field(
        source,
        _constant(20.0),
        config=BoundarySizeReconciliationConfig(
            target_combination="minimum",
        ),
    )
    field_matched = reconcile_boundary_size_field(
        source,
        _constant(20.0),
        config=BoundarySizeReconciliationConfig(
            target_combination="sampled_field",
        ),
    )
    assert len(field_matched.boundary.xy) < len(minimum.boundary.xy)
    assert np.allclose(field_matched.boundary.target_spacing_m, 20.0)
    assert field_matched.audit["factor_compatibility"]["passed"]
    assert field_matched.audit["passed"]
    assert (
        minimum.audit["reproducibility"]["policy_sha256"]
        != field_matched.audit["reproducibility"]["policy_sha256"]
    )
    assert (
        field_matched.audit["thresholds"]["target_combination"]
        == "sampled_field"
    )
    final = audit_reconciled_boundary_size_field(
        field_matched.boundary,
        _constant(20.0),
        config=BoundarySizeReconciliationConfig(
            target_combination="sampled_field",
        ),
    )
    assert final["thresholds"]["target_combination"] == "sampled_field"
    assert (
        final["reproducibility"]["target_combination"]
        == "sampled_field"
    )
    _assert_originals_exact(source, field_matched.boundary)


def test_mixed_open_land_edge_inserts_only_solid_nodes() -> None:
    source = _boundary(
        [(0, 0), (40, 0), (40, 40), (0, 40)],
        kinds=["open", "land", "land", "land"],
        targets=[20, 20, 20, 20],
        hard=[True, True, False, False],
    )
    result = reconcile_boundary_size_field(source, _constant(5.0))
    metadata = result.boundary.metadata
    inserted = metadata["reconciliation_inserted"]
    starts = metadata["reconciliation_source_segment_start_zero_based"]
    ends = metadata["reconciliation_source_segment_end_zero_based"]
    mixed = inserted & (starts == 0) & (ends == 1)
    assert np.count_nonzero(mixed) > 0
    assert set(np.asarray(result.boundary.kinds, dtype=object)[mixed]) == {"land"}
    source_lineage = metadata["reconciliation_source_node_index_zero_based"]
    original_zero = int(np.flatnonzero(source_lineage == 0)[0])
    original_one = int(np.flatnonzero(source_lineage == 1)[0])
    assert result.boundary.kinds[original_zero] == "open"
    assert result.boundary.kinds[original_one] == "land"


def test_sharp_field_drop_gets_cyclic_lower_gradation_envelope() -> None:
    source = _boundary(
        [(0, 0), (1000, 0), (1000, 1000), (0, 1000)],
        targets=[500, 500, 500, 500],
    )

    class SharpSampler:
        def sample_xy(self, points: np.ndarray) -> np.ndarray:
            distance = np.linalg.norm(
                np.asarray(points, dtype=float)
                - np.asarray([500.0, 0.0]),
                axis=1,
            )
            return np.where(distance <= 20.0, 50.0, 500.0)

    policy = BoundarySizeReconciliationConfig(
        minimum_quadrature_points=33,
        quadrature_target_fraction=0.25,
        maximum_spacing_gradient=0.20,
        enforce_sampled_field_compatibility=False,
        sampler_id="sharp_drop_sample_xy_fixture",
    )
    result = reconcile_boundary_size_field(
        source,
        SharpSampler(),
        config=policy,
    )
    assert result.audit["adjacent_target_gradient"]["maximum"] <= 0.20 + 1.0e-12
    assert (
        result.audit["boundary_edge_l_over_h_gamma"]["maximum"]
        <= policy.target_metric_edge + policy.edge_tolerance
    )
    assert result.audit["factor_compatibility"]["incompatible_node_count"] > 0
    assert not result.audit["factor_compatibility"]["enforced_as_hard_gate"]
    assert result.audit["passed"]
    assert "not_wet_distance_min_plus" in result.audit["reproducibility"]["method_scope"]
    midpoint_records = result.edge_midpoint_records()
    assert len(midpoint_records) == len(result.boundary.xy)
    assert midpoint_records[-1]["end_node_index_zero_based"] == 0
    final_against_provisional = audit_reconciled_boundary_size_field(
        result.boundary,
        SharpSampler(),
        config=policy,
    )
    assert not final_against_provisional["passed"]
    assert "boundary_field_factor_compatibility" in (
        final_against_provisional["failure_taxonomy"]
    )


def test_final_rebuilt_field_audit_endpoint_midpoint_contract() -> None:
    source = _boundary(
        [(0, 0), (40, 0), (40, 40), (0, 40)],
        targets=[20, 20, 20, 20],
    )
    result = reconcile_boundary_size_field(source, _constant(5.0))
    final = audit_reconciled_boundary_size_field(
        result.boundary,
        _constant(5.0),
    )
    assert final["passed"]
    assert final["boundary_edge_count"] == len(result.boundary.xy)
    assert final["factor_compatibility"]["incompatible_sample_count"] == 0
    assert final["boundary_edge_l_over_h_gamma"]["maximum"] <= 1.55
    assert final["adjacent_target_gradient"]["maximum"] <= 0.20


if __name__ == "__main__":
    tests = [
        test_closed_loop_coarse_to_fine_and_exact_lineage,
        test_single_and_multiple_obc_identity_order,
        test_cyclic_obc_omits_repeated_first_and_preserves_orientation,
        test_deterministic_repeat_and_spatial_adaptation,
        test_sampled_field_mode_can_smoothly_coarsen_source_targets,
        test_mixed_open_land_edge_inserts_only_solid_nodes,
        test_sharp_field_drop_gets_cyclic_lower_gradation_envelope,
        test_final_rebuilt_field_audit_endpoint_midpoint_contract,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")
