"""Synthetic regression tests for the FVCOM hydraulic-skeleton size field."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon
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
    SizeFieldConfig,
    _wet_graph_distance_and_labels,
    apply_gradation_limit,
    build_size_field,
    write_size_field,
)


def _rectangular_estuary_case(
    *,
    depth: np.ndarray | None = None,
    nx: int = 61,
    ny: int = 31,
) -> tuple[BathymetryGrid, BoundaryNodes]:
    """Return a long wet rectangle with a western OBC and three solid sides."""
    west, east = -122.60, -122.42
    south, north = 37.78, 37.82
    lon = np.linspace(west, east, nx)
    lat = np.linspace(south, north, ny)
    if depth is None:
        depth = np.full((ny, nx), 20.0, dtype=float)
    bathy = BathymetryGrid(
        lon=lon,
        lat=lat,
        depth=np.asarray(depth, dtype=float),
    )

    corners_lonlat = np.asarray(
        [
            [west, south],
            [east, south],
            [east, north],
            [west, north],
        ],
        dtype=float,
    )
    projection = local_utm_projection((west, south, east, north))
    xy = project_points(corners_lonlat, projection)
    polygon_lonlat = Polygon(corners_lonlat)
    return bathy, BoundaryNodes(
        xy=xy,
        lonlat=corners_lonlat,
        kinds=["open", "land", "land", "open"],
        target_spacing_m=np.asarray([3000.0, 250.0, 250.0, 3000.0]),
        exterior_indices=[0, 1, 2, 3],
        open_boundary_indices=[3, 0],
        constraint_chains=[[0, 1, 2, 3]],
        domain_polygon_xy=project_geometry(polygon_lonlat, projection),
        open_boundary_xy=LineString(xy[[3, 0]]),
        land_boundary_xy=LineString(xy[[0, 1, 2, 3]]),
        island_polygons_xy=[],
        projection=projection,
        hard_anchor_mask=np.asarray([True, False, False, True], dtype=bool),
        adaptive_resolution=True,
    )


def _config(**overrides: object) -> SizeFieldConfig:
    values: dict[str, object] = {
        "land_spacing_m": 250.0,
        "open_spacing_m": 3000.0,
        "max_size_m": 5000.0,
        "gradation": 0.20,
        "slope_elements": 10.0,
        "coastal_distance_m": 20_000.0,
        "hydraulic_elements_across_min": 3.0,
        "hydraulic_elements_across_max": 8.0,
        "hydraulic_max_width_m": 10_000.0,
        "hydraulic_bank_angle_deg": 110.0,
        "hydraulic_longitudinal_gradation": 0.10,
        "hydraulic_corridor_width_factor": 0.55,
        "obc_hold_distance_m": 1000.0,
        "obc_transition_distance_m": 6000.0,
    }
    values.update(overrides)
    return SizeFieldConfig(**values)


def _build(
    bathy: BathymetryGrid,
    boundary: BoundaryNodes,
    **config_overrides: object,
):
    return build_size_field(
        bathy,
        boundary,
        _config(**config_overrides),
        domain_mask=np.ones_like(bathy.depth, dtype=bool),
    )


def test_hydraulic_skeleton_follows_opposing_solid_banks() -> None:
    bathy, boundary = _rectangular_estuary_case()
    flat = _build(bathy, boundary)

    lon2, lat2 = np.meshgrid(bathy.lon, bathy.lat)
    displaced_trough = 12.0 + 45.0 * (
        1.0
        - np.exp(
            -(
                (lat2 - 37.811) / 0.0025
            )
            ** 2
        )
    )
    displaced_bathy, displaced_boundary = _rectangular_estuary_case(
        depth=displaced_trough
    )
    displaced = _build(displaced_bathy, displaced_boundary)

    assert flat.report["hydraulic_skeleton"]["status"] == "complete"
    assert flat.report["hydraulic_skeleton"]["open_boundary_used_as_bank"] is False
    assert np.count_nonzero(flat.hydraulic_skeleton_mask) >= 3
    assert np.array_equal(
        flat.hydraulic_skeleton_mask,
        displaced.hydraulic_skeleton_mask,
    )
    skeleton_rows, _ = np.nonzero(flat.hydraulic_skeleton_mask)
    skeleton_lat = bathy.lat[skeleton_rows]
    assert abs(float(np.median(skeleton_lat)) - float(np.mean(bathy.lat))) <= (
        1.5 * float(np.diff(bathy.lat).mean())
    )
    assert np.all(
        np.isfinite(
            flat.hydraulic_cross_section_area_m2[
                flat.hydraulic_skeleton_mask
            ]
        )
    )
    assert np.any(flat.hydraulic_corridor_mask)
    assert np.any(flat.source_attribution == 3)


def test_bathymetric_slope_target_is_active() -> None:
    _, baseline_boundary = _rectangular_estuary_case()
    ny, nx = 31, 61
    depth = np.repeat(np.linspace(1.0, 81.0, ny)[:, None], nx, axis=1)
    bathy, boundary = _rectangular_estuary_case(depth=depth, nx=nx, ny=ny)
    assert np.allclose(boundary.xy, baseline_boundary.xy)
    field = _build(bathy, boundary)

    shallow = field.coastal_mask & (field.depth <= 50.0)
    assert np.any(shallow)
    assert np.all(np.isfinite(field.slope_size[shallow]))
    assert field.report["shallow_slope_active_cell_count"] == int(
        np.count_nonzero(shallow)
    )
    assert np.any(field.source_attribution == 2)


def test_wet_obc_target_hold_and_quintic_log_transfer() -> None:
    bathy, boundary = _rectangular_estuary_case()
    field = _build(bathy, boundary)
    active = field.domain_mask & field.coverage_mask

    assert np.all(np.isfinite(field.wet_obc_distance_m[active]))
    assert np.allclose(field.wet_obc_target_m[active], 3000.0)
    middle = len(bathy.lat) // 2
    row_distance = field.wet_obc_distance_m[middle]
    row_fraction = field.transition_fraction[middle]
    assert np.all(np.diff(row_distance) >= -1.0e-9)
    assert np.all(np.diff(row_fraction) >= -1.0e-12)

    hold = active & (field.wet_obc_distance_m <= 1000.0 + 1.0e-9)
    assert np.any(hold)
    assert np.allclose(
        field.obc_transition_size[hold],
        field.wet_obc_target_m[hold],
    )

    transferring = active & (field.transition_fraction > 0.0) & (
        field.transition_fraction < 1.0
    )
    assert np.any(transferring)
    index = int(np.flatnonzero(transferring.ravel())[len(np.flatnonzero(transferring)) // 2])
    alpha = float(field.transition_fraction.ravel()[index])
    expected = np.exp(
        (1.0 - alpha) * np.log(field.wet_obc_target_m.ravel()[index])
        + alpha * np.log(field.nearshore_size.ravel()[index])
    )
    assert abs(field.obc_transition_size.ravel()[index] - expected) < 1.0e-9

    report = field.report["open_boundary_transition"]
    assert report["method"] == "wet_distance_quintic_log_authority_transfer"
    assert report["hold_distance_m"] == 1000.0
    assert report["effective_transition_distance_m"] == 6000.0
    assert report["wet_distance_reachable_cell_count"] == int(
        np.count_nonzero(active)
    )


def test_wet_distance_does_not_cross_a_dry_barrier() -> None:
    wet = np.ones((7, 7), dtype=bool)
    wet[:6, 3] = False
    source = np.zeros_like(wet)
    source[2, 0] = True
    distance, labels = _wet_graph_distance_and_labels(
        wet,
        source,
        1.0,
        1.0,
    )
    assert np.all(np.isinf(distance[:6, 3]))
    assert distance[2, 6] > 8.0
    assert labels[2, 6] == 14


def test_unreachable_raster_component_retains_nearshore_target() -> None:
    bathy, boundary = _rectangular_estuary_case()
    domain = np.ones_like(bathy.depth, dtype=bool)
    domain[:, 30] = False
    field = build_size_field(
        bathy,
        boundary,
        _config(),
        domain_mask=domain,
    )
    unreachable = domain & ~np.isfinite(field.wet_obc_distance_m)
    assert np.any(unreachable)
    assert np.all(np.isfinite(field.size[unreachable]))
    assert np.allclose(
        field.obc_transition_size[unreachable],
        field.nearshore_size[unreachable],
    )
    report = field.report["open_boundary_transition"]
    assert report["wet_distance_unreachable_cell_count"] == int(
        np.count_nonzero(unreachable)
    )
    assert report["unreachable_cell_policy"].startswith(
        "retain_nearshore_target"
    )


def test_gradation_never_coarsens_and_uses_eight_neighbors() -> None:
    lon = np.asarray([-122.60, -122.59, -122.58], dtype=float)
    lat = np.asarray([37.78, 37.79, 37.80], dtype=float)
    raw = np.full((3, 3), 1000.0, dtype=float)
    raw[1, 1] = 100.0
    limited, report = apply_gradation_limit(lon, lat, raw, 0.10)
    assert report["connectivity"] == 8
    assert report["never_coarsened"] is True
    assert report["max_neighbor_gradation"] <= 0.100000001
    assert np.all(limited <= raw + 1.0e-9)


def test_strict_coverage_sampling() -> None:
    bathy, boundary = _rectangular_estuary_case()
    field = _build(bathy, boundary)
    try:
        field.sample(np.asarray([-122.70]), np.asarray([37.80]))
    except ValueError as exc:
        assert "outside explicit coverage" in str(exc)
    else:
        raise AssertionError("Sampling outside explicit coverage must fail")


def test_netcdf_v4_and_component_maps() -> None:
    bathy, boundary = _rectangular_estuary_case()
    field = _build(bathy, boundary)
    with tempfile.TemporaryDirectory(prefix="size-field-v4-") as temp_dir:
        nc_path = Path(temp_dir) / "size_field.nc"
        png_path = Path(temp_dir) / "size_field.png"
        written_nc, written_png, components_png = write_size_field(
            field,
            nc_path,
            png_path,
        )
        assert written_nc == nc_path
        assert written_png == png_path
        assert components_png == Path(temp_dir) / "size_field_components.png"
        with xr.open_dataset(nc_path) as dataset:
            assert dataset.attrs["schema_version"] == "fvcom_size_field_v4"
            assert dataset.attrs["coverage_policy"] == "strict"
            expected_variables = {
                "mesh_size_m",
                "solid_boundary_background_mesh_size_m",
                "bathymetry_slope_mesh_size_m",
                "hydraulic_corridor_mesh_size_m",
                "nearshore_mesh_size_m",
                "obc_transition_mesh_size_m",
                "wet_obc_distance_m",
                "wet_obc_source_target_m",
                "hydraulic_skeleton_mask",
                "hydraulic_bank_width_m",
                "hydraulic_importance",
                "hydraulic_storage_ranking_area_m2",
                "hydraulic_cross_section_area_m2",
                "nearshore_size_source_attribution",
            }
            assert expected_variables.issubset(dataset.data_vars)
            assert np.allclose(dataset["mesh_size_m"].values, field.size)
        for path in (png_path, components_png):
            assert path.exists() and path.stat().st_size > 0


def main() -> None:
    tests = [
        test_hydraulic_skeleton_follows_opposing_solid_banks,
        test_bathymetric_slope_target_is_active,
        test_wet_obc_target_hold_and_quintic_log_transfer,
        test_wet_distance_does_not_cross_a_dry_barrier,
        test_unreachable_raster_component_retains_nearshore_target,
        test_gradation_never_coarsens_and_uses_eight_neighbors,
        test_strict_coverage_sampling,
        test_netcdf_v4_and_component_maps,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("hydraulic-skeleton size-field self-test passed")


if __name__ == "__main__":
    main()
