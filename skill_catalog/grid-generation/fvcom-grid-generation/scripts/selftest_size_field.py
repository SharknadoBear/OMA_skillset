"""Synthetic regression tests for the unified FVCOM size-field algorithm."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString, Point, Polygon
import xarray as xr


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from fvcom_grid_generation.bathymetry import BathymetryGrid
from fvcom_grid_generation.boundary import BoundaryNodes
from fvcom_grid_generation.projection import (
    local_utm_projection,
    project_geometry,
    project_points,
)
from fvcom_grid_generation.size_field import (
    ChannelFlowline,
    SizeFieldConfig,
    _grid_spacing_m,
    apply_gradation_limit,
    boundary_background_size,
    boundary_front_seed_points,
    build_size_field,
    estimate_node_budget,
    oceanmesh_feature_size,
    write_size_field,
)


def _square_case(
    *,
    kinds: list[str] | None = None,
    targets: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    count: int = 21,
) -> tuple[BathymetryGrid, BoundaryNodes]:
    lon = np.linspace(-75.01, -74.99, count)
    lat = np.linspace(39.0, 39.02, count)
    if depth is None:
        depth = np.full((count, count), 20.0, dtype=float)
    bathy = BathymetryGrid(lon=lon, lat=lat, depth=np.asarray(depth, dtype=float))
    projection = local_utm_projection((-75.01, 39.0, -74.99, 39.02))
    corners_lonlat = np.asarray(
        [[-75.01, 39.0], [-74.99, 39.0], [-74.99, 39.02], [-75.01, 39.02]],
        dtype=float,
    )
    xy = project_points(corners_lonlat, projection)
    polygon_lonlat = Polygon(corners_lonlat)
    polygon_xy = project_geometry(polygon_lonlat, projection)
    boundary_kinds = kinds or ["land", "open", "open", "land"]
    boundary_targets = (
        np.asarray(targets, dtype=float)
        if targets is not None
        else np.asarray([100.0, 1000.0, 1000.0, 100.0], dtype=float)
    )
    open_indices = [
        index for index, kind in enumerate(boundary_kinds) if kind == "open"
    ]
    return bathy, BoundaryNodes(
        xy=xy,
        lonlat=corners_lonlat,
        kinds=boundary_kinds,
        target_spacing_m=boundary_targets,
        exterior_indices=[0, 1, 2, 3],
        open_boundary_indices=open_indices,
        constraint_chains=[[0, 1, 2, 3]],
        domain_polygon_xy=polygon_xy,
        open_boundary_xy=LineString(xy[[1, 2]]) if open_indices else LineString(),
        land_boundary_xy=LineString(xy[[3, 0]]),
        island_polygons_xy=[],
        projection=projection,
        hard_anchor_mask=np.asarray([False, True, True, False], dtype=bool),
        adaptive_resolution=True,
    )


def _config(**overrides: object) -> SizeFieldConfig:
    values: dict[str, object] = {
        "land_spacing_m": 100.0,
        "open_spacing_m": 1000.0,
        "max_size_m": 5000.0,
        "gradation": 0.20,
        "slope_elements": 10.0,
        "coastal_distance_m": 20_000.0,
        "feature_elements": 3.0,
        "wavelength_period_s": 44_714.0,
        "wavelength_elements": 20.0,
    }
    values.update(overrides)
    return SizeFieldConfig(**values)


def test_open_land_endpoint_and_monotone_transition() -> None:
    bathy, boundary = _square_case()
    background, d_open, d_land, phi, report = boundary_background_size(
        bathy,
        boundary,
        _config(),
    )
    middle = len(bathy.lat) // 2
    transect = background[middle, :]
    assert abs(transect[0] - 100.0) < 1.0e-6
    assert abs(transect[-1] - 1000.0) < 1.0e-6
    assert np.all(np.diff(transect) >= -1.0e-9)
    assert abs(phi[middle, 0] - 1.0) < 1.0e-7
    assert abs(phi[middle, -1]) < 1.0e-7
    assert np.all(np.isfinite(d_open))
    assert np.all(np.isfinite(d_land))
    assert report["method"] == "open_land_log_smoothstep"


def test_coincident_landfall_definition() -> None:
    bathy, boundary = _square_case()
    background, _, _, phi, report = boundary_background_size(
        bathy,
        boundary,
        _config(),
    )
    bottom = 0
    open_landfall = len(bathy.lon) - 1
    assert abs(phi[bottom, open_landfall] - 0.5) < 1.0e-12
    assert abs(background[bottom, open_landfall] - 1000.0) < 1.0e-6
    assert report["open_segment_count"] > 0
    assert report["land_segment_count"] > 0


def test_closed_domain_land_distance_background() -> None:
    bathy, boundary = _square_case(
        kinds=["land"] * 4,
        targets=np.full(4, 200.0),
    )
    background, d_open, d_land, phi, report = boundary_background_size(
        bathy,
        boundary,
        _config(land_spacing_m=150.0),
    )
    assert report["method"] == "closed_domain_land_distance"
    assert np.all(np.isinf(d_open))
    assert np.all(np.isnan(phi))
    assert abs(background[0, len(bathy.lon) // 2] - 200.0) < 0.1
    assert background[len(bathy.lat) // 2, len(bathy.lon) // 2] > 200.0
    assert np.all(np.isfinite(d_land))


def test_shallow_slope_is_active() -> None:
    count = 21
    x = np.linspace(0.0, 1.0, count)
    depth = np.tile(2.0 + 18.0 * x, (count, 1))
    bathy, boundary = _square_case(depth=depth, count=count)
    field = build_size_field(
        bathy,
        boundary,
        _config(),
        domain_mask=np.ones_like(depth, dtype=bool),
    )
    shallow = field.coastal_mask & (field.depth <= 20.0)
    assert np.any(shallow)
    assert np.all(np.isfinite(field.slope_size[shallow]))
    assert field.report["shallow_slope_active_cell_count"] == int(
        np.count_nonzero(shallow)
    )


def test_coastal_mask_excludes_offshore_candidates() -> None:
    bathy, boundary = _square_case()
    field = build_size_field(
        bathy,
        boundary,
        _config(coastal_distance_m=100.0),
        domain_mask=np.ones_like(bathy.depth, dtype=bool),
    )
    middle = len(bathy.lat) // 2
    assert not field.coastal_mask[middle, middle]
    assert field.source_attribution[middle, middle] == 1
    assert np.isnan(field.feature_size[middle, middle])
    assert np.isnan(field.slope_size[middle, middle])


def test_feature_and_wavelength_formulas() -> None:
    count = 41
    bathy, boundary = _square_case(count=count)
    wet = np.zeros((count, count), dtype=bool)
    wet[2:-2, 8:-8] = True
    fallback = np.full((count, count), 5000.0, dtype=float)
    feature, report = oceanmesh_feature_size(
        bathy,
        boundary,
        wet,
        fallback,
        _config(feature_elements=3.0),
    )
    assert report["fallback_used"] is False
    dx_m, dy_m = _grid_spacing_m(bathy.lon, bathy.lat)
    wet_distance = distance_transform_edt(
        np.pad(wet, 1, constant_values=False),
        sampling=(dy_m, dx_m),
    )[1:-1, 1:-1]
    row = count // 2
    col = count // 2
    assert abs(feature[row, col] - 2.0 * wet_distance[row, col] / 3.0) < 1.0e-6

    field = build_size_field(
        bathy,
        boundary,
        _config(),
        domain_mask=np.ones_like(bathy.depth, dtype=bool),
    )
    expected = 44_714.0 * np.sqrt(9.807 * 20.0) / 20.0
    assert np.allclose(
        field.wavelength_size[field.coastal_mask],
        expected,
    )


def test_optional_channel_and_highest_order_attribution() -> None:
    bathy, boundary = _square_case()
    center_y = float(np.mean(boundary.xy[:, 1]))
    left_x = float(np.min(boundary.xy[:, 0]))
    right_x = float(np.max(boundary.xy[:, 0]))
    line = LineString([(left_x, center_y), (right_x, center_y)])
    field = build_size_field(
        bathy,
        boundary,
        _config(channel_min_size_m=75.0),
        flowlines=[
            ChannelFlowline(line, 2),
            ChannelFlowline(line, 5),
        ],
        domain_mask=np.ones_like(bathy.depth, dtype=bool),
    )
    row = len(bathy.lat) // 2
    col = len(bathy.lon) // 2
    assert field.channel_size[row, col] == 75.0
    assert field.channel_seg_order[row, col] == 5
    assert field.source_attribution[row, col] == 5
    assert field.report["channel"]["segorder_changes_size"] is False


def test_cfl_is_report_only() -> None:
    bathy, boundary = _square_case()
    mask = np.ones_like(bathy.depth, dtype=bool)
    baseline = build_size_field(
        bathy,
        boundary,
        _config(target_timestep_s="auto"),
        domain_mask=mask,
    )
    diagnostic = build_size_field(
        bathy,
        boundary,
        _config(target_timestep_s=1_000_000.0),
        domain_mask=mask,
    )
    assert np.array_equal(baseline.size, diagnostic.size)
    assert diagnostic.report["cfl"]["mode"] == "diagnostic_only"
    assert diagnostic.report["cfl"]["cfl_modifies_size"] is False
    assert diagnostic.report["cfl"]["cells_below_target_timestep"] > 0
    expected_dt = (
        diagnostic.report["cfl"]["cfl"]
        * float(np.nanmin(diagnostic.size))
        / np.sqrt(9.807 * 20.0)
    )
    assert abs(diagnostic.report["cfl"]["recommended_timestep_s"] - expected_dt) < 1.0e-9
    assert "external_mode_sqrt" in diagnostic.report["cfl"]["wave_speed_assumption"]


def test_strict_coverage_sampling() -> None:
    bathy, boundary = _square_case()
    field = build_size_field(
        bathy,
        boundary,
        _config(),
        domain_mask=np.ones_like(bathy.depth, dtype=bool),
    )
    try:
        field.sample(np.asarray([-75.02]), np.asarray([39.01]))
    except ValueError as exc:
        assert "outside explicit coverage" in str(exc)
    else:
        raise AssertionError("Sampling outside the explicit grid must fail")


def test_gradation_never_coarsens_and_uses_eight_neighbors() -> None:
    lon = np.asarray([-75.0, -74.99, -74.98], dtype=float)
    lat = np.asarray([39.0, 39.01, 39.02], dtype=float)
    raw = np.full((3, 3), 1000.0, dtype=float)
    raw[1, 1] = 100.0
    limited, report = apply_gradation_limit(lon, lat, raw, 0.10)
    assert report["connectivity"] == 8
    assert report["never_coarsened"] is True
    assert report["max_neighbor_gradation"] <= 0.100000001
    assert np.all(limited <= raw + 1.0e-9)


def test_node_budget_and_boundary_front_seeds() -> None:
    lon = np.linspace(-75.0, -74.9, 11)
    lat = np.linspace(39.0, 39.1, 11)
    fine = estimate_node_budget(lon, lat, np.full((11, 11), 500.0))
    coarse = estimate_node_budget(lon, lat, np.full((11, 11), 1000.0))
    ratio = (
        fine["estimated_interior_node_count_float"]
        / coarse["estimated_interior_node_count_float"]
    )
    assert abs(ratio - 4.0) < 1.0e-12

    _, boundary = _square_case()
    boundary.target_spacing_m[:] = 200.0
    seeds, report = boundary_front_seed_points(boundary)
    assert len(seeds) >= 4
    assert report["hard_anchor_bisector_count"] == 2
    assert all(boundary.domain_polygon_xy.contains(Point(*point)) for point in seeds)


def test_netcdf_roundtrip_and_schema() -> None:
    bathy, boundary = _square_case()
    field = build_size_field(
        bathy,
        boundary,
        _config(),
        domain_mask=np.ones_like(bathy.depth, dtype=bool),
    )
    with tempfile.TemporaryDirectory(prefix="size-field-v3-") as temp_dir:
        nc_path = Path(temp_dir) / "size_field.nc"
        png_path = Path(temp_dir) / "size_field.png"
        write_size_field(field, nc_path, png_path)
        with xr.open_dataset(nc_path) as dataset:
            assert dataset.attrs["schema_version"] == "fvcom_size_field_v3"
            assert dataset.attrs["coverage_policy"] == "strict"
            assert "background_mesh_size_m" in dataset
            assert "oceanmesh_feature_mesh_size_m" in dataset
            assert "m2_wavelength_mesh_size_m" in dataset
            assert "channel_seg_order" in dataset
            assert "raw_size_source_attribution" in dataset
            assert np.allclose(dataset["mesh_size_m"].values, field.size)
        assert png_path.exists() and png_path.stat().st_size > 0


def main() -> None:
    tests = [
        test_open_land_endpoint_and_monotone_transition,
        test_coincident_landfall_definition,
        test_closed_domain_land_distance_background,
        test_shallow_slope_is_active,
        test_coastal_mask_excludes_offshore_candidates,
        test_feature_and_wavelength_formulas,
        test_optional_channel_and_highest_order_attribution,
        test_cfl_is_report_only,
        test_strict_coverage_sampling,
        test_gradation_never_coarsens_and_uses_eight_neighbors,
        test_node_budget_and_boundary_front_seeds,
        test_netcdf_roundtrip_and_schema,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("unified size-field self-test passed")


if __name__ == "__main__":
    main()
