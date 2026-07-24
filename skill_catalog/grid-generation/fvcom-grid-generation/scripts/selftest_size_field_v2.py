"""Focused synthetic tests for the opt-in FVCOM size-field v2 internals."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import xarray as xr
from shapely.geometry import LineString, Point, Polygon


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from fvcom_grid_generation.bathymetry import BathymetryGrid
from fvcom_grid_generation.boundary import BoundaryNodes
from fvcom_grid_generation.projection import local_utm_projection, project_geometry, project_points
from fvcom_grid_generation.size_field import (
    SizeFieldConfig,
    SizeFieldSemantics,
    apply_gradation_limit,
    boundary_front_seed_points,
    build_size_field,
    estimate_node_budget,
    segment_boundary_distance_size,
    write_size_field,
)
from fvcom_grid_generation.workflow import _size_field_semantics_v2


def _square_case() -> tuple[BathymetryGrid, BoundaryNodes]:
    lon = np.asarray([-75.01, -75.005, -75.0, -74.995, -74.99], dtype=float)
    lat = np.asarray([39.0, 39.0025, 39.005, 39.0075, 39.01], dtype=float)
    depth = np.full((len(lat), len(lon)), 20.0, dtype=float)
    bathy = BathymetryGrid(lon=lon, lat=lat, depth=depth)
    projection = local_utm_projection((-75.01, 39.0, -74.99, 39.01))
    corners_lonlat = np.asarray(
        [[-75.01, 39.0], [-74.99, 39.0], [-74.99, 39.01], [-75.01, 39.01]],
        dtype=float,
    )
    xy = project_points(corners_lonlat, projection)
    polygon_lonlat = Polygon(corners_lonlat)
    polygon_xy = project_geometry(polygon_lonlat, projection)
    return bathy, BoundaryNodes(
        xy=xy,
        lonlat=corners_lonlat,
        kinds=["land", "land", "open", "open"],
        target_spacing_m=np.asarray([100.0, 300.0, 1000.0, 1000.0], dtype=float),
        exterior_indices=[0, 1, 2, 3],
        open_boundary_indices=[2, 3],
        constraint_chains=[[0, 1, 2, 3]],
        domain_polygon_xy=polygon_xy,
        open_boundary_xy=LineString(xy[[2, 3]]),
        land_boundary_xy=LineString(xy[[0, 1, 2, 3, 0]]),
        island_polygons_xy=[],
        projection=projection,
        hard_anchor_mask=np.asarray([True, True, False, False], dtype=bool),
        adaptive_resolution=True,
    )


def _v2_config(**overrides: object) -> SizeFieldConfig:
    values: dict[str, object] = {
        "land_spacing_m": 100.0,
        "open_spacing_m": 100.0,
        "max_size_m": 2000.0,
        "gradation": 0.15,
        "adaptive_boundary": True,
        "bathymetry_gradient_policy": "off",
        "size_field_profile": "adaptive-coastal-v2",
    }
    values.update(overrides)
    return SizeFieldConfig(**values)


def test_segment_target_interpolation() -> None:
    bathy, boundary = _square_case()
    size, distance, target, source = segment_boundary_distance_size(
        bathy,
        boundary,
        _v2_config(gradation=0.0),
    )
    assert size.shape == bathy.depth.shape
    assert distance[1, 2] > 0.0
    assert 180.0 < target[1, 2] < 220.0, target[1, 2]
    assert size[1, 2] == target[1, 2]
    assert source[1, 2] == 1


def test_hard_priority_over_junction_floor() -> None:
    bathy, boundary = _square_case()
    shape = bathy.depth.shape
    junction = np.zeros(shape, dtype=bool)
    junction[2, 2] = True
    channel = np.full(shape, np.inf, dtype=float)
    channel[2, 2] = 50.0
    field = build_size_field(
        bathy,
        boundary,
        _v2_config(junction_floor_m=800.0),
        semantics=SizeFieldSemantics(junction_mask=junction, channel_size_m=channel),
    )
    assert field.report["schema_version"] == "fvcom_size_field_v2"
    assert field.soft_size is not None and field.soft_size[2, 2] == 800.0
    assert field.hard_size is not None and field.hard_size[2, 2] == 50.0
    assert field.raw_size[2, 2] == 50.0
    assert field.source_attribution is not None and field.source_attribution[2, 2] == 5
    assert field.report["junction"]["hard_override_cell_count"] == 1


def test_strict_and_explicit_coverage_sampling() -> None:
    bathy, boundary = _square_case()
    strict = build_size_field(bathy, boundary, _v2_config())
    try:
        strict.sample(np.asarray([-75.02]), np.asarray([39.005]))
    except ValueError as exc:
        assert "outside explicit coverage" in str(exc)
    else:
        raise AssertionError("strict v2 sampling should reject an uncovered point")

    nearest = build_size_field(bathy, boundary, _v2_config(coverage_policy="nearest"))
    sampled = nearest.sample(np.asarray([-75.02]), np.asarray([39.005]))
    assert sampled.shape == (1,) and np.isfinite(sampled[0])

    _, uncovered_boundary = _square_case()
    uncovered_boundary.lonlat[2, 0] = -74.98
    try:
        build_size_field(bathy, uncovered_boundary, _v2_config())
    except ValueError as exc:
        assert "boundary is outside explicit" in str(exc)
    else:
        raise AssertionError("strict v2 construction should reject an uncovered boundary node")


def test_automatic_land_open_junction_grade() -> None:
    bathy, boundary = _square_case()
    field = build_size_field(bathy, boundary, _v2_config())
    automatic = field.report["junction"]["automatic"]
    assert automatic["junction_point_count"] == 2
    assert automatic["cell_count"] > 0
    assert field.junction_mask is not None and np.any(field.junction_mask)
    assert field.domain_mask is not None and np.any(field.domain_mask)


def test_passage_layer_becomes_hard_channel_constraint() -> None:
    bathy, boundary = _square_case()
    center = np.mean(boundary.xy, axis=0)
    boundary.passage_diagnostics = [
        {
            "action": "harmonize_paired_spacing",
            "resolvable_at_minimum_spacing": True,
            "required_target_spacing_m": 50.0,
            "width_m": 400.0,
            "geometry_xy": LineString([center + [-200.0, 0.0], center + [200.0, 0.0]]),
        }
    ]
    semantics = _size_field_semantics_v2(bathy, boundary)
    assert semantics.junction_mask is None
    assert semantics.junction_floor_m is None
    assert semantics.channel_size_m is not None
    assert np.nanmin(np.asarray(semantics.channel_size_m, dtype=float)) == 50.0
    field = build_size_field(bathy, boundary, _v2_config(), semantics=semantics)
    assert field.hard_size is not None
    assert np.nanmin(field.hard_size) == 50.0
    assert field.report["source_attribution_cell_counts"]["channel_hard"] > 0


def test_eight_neighbor_metric_gradation() -> None:
    lon = np.asarray([-75.0, -74.99, -74.98], dtype=float)
    lat = np.asarray([39.0, 39.01, 39.02], dtype=float)
    raw = np.full((3, 3), 1000.0, dtype=float)
    raw[1, 1] = 100.0
    limited, report = apply_gradation_limit(lon, lat, raw, 0.10, connectivity=8)
    assert report["connectivity"] == 8
    assert report["max_neighbor_gradation"] <= 0.100000001
    assert limited[0, 0] < raw[0, 0]


def test_node_budget_scaling() -> None:
    lon = np.linspace(-75.0, -74.9, 11)
    lat = np.linspace(39.0, 39.1, 11)
    fine = estimate_node_budget(lon, lat, np.full((11, 11), 500.0))
    coarse = estimate_node_budget(lon, lat, np.full((11, 11), 1000.0))
    ratio = fine["estimated_interior_node_count_float"] / coarse["estimated_interior_node_count_float"]
    assert abs(ratio - 4.0) < 1.0e-12


def test_boundary_front_and_anchor_seeds() -> None:
    _, boundary = _square_case()
    boundary.target_spacing_m[:] = 200.0
    seeds, report = boundary_front_seed_points(boundary)
    assert len(seeds) >= 4
    assert report["hard_anchor_bisector_count"] == 2
    assert all(boundary.domain_polygon_xy.contains(Point(*point)) for point in seeds)


def test_v2_persistence_and_v1_default() -> None:
    bathy, boundary = _square_case()
    v2 = build_size_field(bathy, boundary, _v2_config())
    with tempfile.TemporaryDirectory(prefix="size-field-v2-") as temp_dir:
        nc_path = Path(temp_dir) / "size_field.nc"
        png_path = Path(temp_dir) / "size_field.png"
        write_size_field(v2, nc_path, png_path)
        with xr.open_dataset(nc_path) as dataset:
            assert dataset.attrs["schema_version"] == "fvcom_size_field_v2"
            assert "soft_priority_mesh_size_m" in dataset
            assert "size_source_attribution" in dataset
            assert "size_field_coverage_mask" in dataset

    v1 = build_size_field(
        bathy,
        boundary,
        SizeFieldConfig(
            land_spacing_m=100.0,
            open_spacing_m=100.0,
            max_size_m=2000.0,
            adaptive_boundary=True,
            bathymetry_gradient_policy="off",
        ),
    )
    assert v1.report["schema_version"] == "fvcom_size_field_v1"
    assert v1.report["gradation"]["method"] == "priority_queue_lower_envelope"
    outside = v1.sample(np.asarray([-76.0]), np.asarray([40.0]))
    assert outside[0] == np.nanmax(v1.size)


def main() -> None:
    tests = [
        test_segment_target_interpolation,
        test_hard_priority_over_junction_floor,
        test_strict_and_explicit_coverage_sampling,
        test_automatic_land_open_junction_grade,
        test_passage_layer_becomes_hard_channel_constraint,
        test_eight_neighbor_metric_gradation,
        test_node_budget_scaling,
        test_boundary_front_and_anchor_seeds,
        test_v2_persistence_and_v1_default,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("size-field v2 self-test passed")


if __name__ == "__main__":
    main()
