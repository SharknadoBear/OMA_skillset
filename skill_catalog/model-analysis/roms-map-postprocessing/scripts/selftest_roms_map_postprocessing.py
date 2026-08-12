#!/usr/bin/env python3
"""Offline synthetic tests for ROMS loading, transforms, and static rendering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from netCDF4 import Dataset
from PIL import Image

from roms_map_tools import build_wet_cell_footprints
from roms_map_postprocessing import main as map_main
from roms_output import (
    destagger_u_to_rho,
    destagger_v_to_rho,
    inspect_inputs,
    load_current_series,
    load_scalar_series,
    roms_depths,
    rotate_to_earth,
)


ETA, XI, LEVELS = 4, 5, 3
S_RHO = np.array([-5.0 / 6.0, -0.5, -1.0 / 6.0])
S_W = np.array([-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0])


def _grid():
    y, x = np.mgrid[:ETA, :XI]
    lon = -76.0 + 0.08 * x + 0.005 * y * y
    lat = 38.0 + 0.07 * y + 0.004 * x * y
    mask = np.ones((ETA, XI), dtype=np.int8)
    mask[0, 0] = 0
    h = 8.0 + 0.7 * x + 0.4 * y
    return lon, lat, mask, h


def _set_angle_metadata(variable):
    variable.units = "radians"
    variable.standard_name = "grid_angle_of_rotation_from_east_to_y"
    variable.long_name = "angle between XI-axis and EAST"


def write_raw(path: Path, hours, *, vtransform=1, reverse=False, angle=np.pi * 0.0,
              include_v=True, include_angle=True, include_mask=True, missing_layer=False):
    lon, lat, mask, h = _grid()
    s_rho = S_RHO[::-1] if reverse else S_RHO
    s_w = S_W[::-1] if reverse else S_W
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        for name, size in (("ocean_time", None), ("s_rho", LEVELS), ("s_w", LEVELS + 1),
                           ("eta_rho", ETA), ("xi_rho", XI), ("eta_u", ETA), ("xi_u", XI - 1),
                           ("eta_v", ETA - 1), ("xi_v", XI)):
            ds.createDimension(name, size)
        time = ds.createVariable("ocean_time", "f8", ("ocean_time",))
        time.units = "seconds since 1970-01-01 00:00:00 UTC"
        time[:] = np.asarray(hours) * 3600.0 + 1784505600.0
        for name, values, dimensions in (
            ("lon_rho", lon, ("eta_rho", "xi_rho")), ("lat_rho", lat, ("eta_rho", "xi_rho")),
            ("mask_rho", mask, ("eta_rho", "xi_rho")), ("h", h, ("eta_rho", "xi_rho")),
            ("mask_u", mask[:, :-1] * mask[:, 1:], ("eta_u", "xi_u")),
            ("mask_v", mask[:-1, :] * mask[1:, :], ("eta_v", "xi_v")),
            ("s_rho", s_rho, ("s_rho",)), ("Cs_r", s_rho, ("s_rho",)),
            ("s_w", s_w, ("s_w",)), ("Cs_w", s_w, ("s_w",)),
        ):
            if name == "mask_rho" and not include_mask:
                continue
            ds.createVariable(name, "f8" if name not in {"mask_rho", "mask_u", "mask_v"} else "i1", dimensions)[:] = values
        if include_angle:
            angle_variable = ds.createVariable("angle", "f8", ("eta_rho", "xi_rho"))
            angle_variable[:] = angle
            _set_angle_metadata(angle_variable)
        ds.createVariable("hc", "f8")[:] = 2.0
        ds.createVariable("Vtransform", "i4")[:] = vtransform
        ds.createVariable("Vstretching", "i4")[:] = 1
        count = len(hours)
        zeta = ds.createVariable("zeta", "f8", ("ocean_time", "eta_rho", "xi_rho"), fill_value=1.0e37)
        zeta.long_name, zeta.units = "water surface elevation", "m"
        salt = ds.createVariable("salt", "f8", ("ocean_time", "s_rho", "eta_rho", "xi_rho"), fill_value=1.0e37)
        salt.long_name = "practical salinity"
        u = ds.createVariable("u", "f8", ("ocean_time", "s_rho", "eta_u", "xi_u"), fill_value=1.0e37)
        u.standard_name, u.units = "sea_water_x_velocity", "m s-1"
        v = None
        if include_v:
            v = ds.createVariable("v", "f8", ("ocean_time", "s_rho", "eta_v", "xi_v"), fill_value=1.0e37)
            v.standard_name, v.units = "sea_water_y_velocity", "m s-1"
        for local, hour in enumerate(hours):
            zeta[local] = 0.05 * np.sin(hour / 3.0)
            canonical_salt = np.stack([10.0 + hour * 0.01 + k * 10.0 + 0.02 * lon for k in range(LEVELS)])
            canonical_u = np.stack([1.0 + hour * 0.01 + k * 0.1 + np.zeros((ETA, XI - 1)) for k in range(LEVELS)])
            canonical_v = np.stack([2.0 + hour * 0.01 + k * 0.2 + np.zeros((ETA - 1, XI)) for k in range(LEVELS)])
            salt[local] = canonical_salt[::-1] if reverse else canonical_salt
            u[local] = canonical_u[::-1] if reverse else canonical_u
            if v is not None:
                v[local] = canonical_v[::-1] if reverse else canonical_v
        if missing_layer:
            salt[0, 1, 2, 2] = salt._FillValue


def write_compact(path: Path, raw_paths):
    salinity = load_scalar_series(raw_paths, variable="salinity", layer="surface")
    current = load_current_series(raw_paths, layer="surface")
    grid = salinity.grid
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        for name, size in (("time", len(salinity.times)), ("s_rho", len(grid.s_rho)),
                           ("s_w", len(grid.s_w)), ("eta_rho", ETA), ("xi_rho", XI)):
            ds.createDimension(name, size)
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "seconds since 1970-01-01 00:00:00 UTC"
        time[:] = salinity.times.astype("int64") / 1.0e9
        for name, values, dimensions in (
            ("lon_rho", grid.lon, ("eta_rho", "xi_rho")), ("lat_rho", grid.lat, ("eta_rho", "xi_rho")),
            ("mask_rho", grid.mask, ("eta_rho", "xi_rho")), ("h", grid.h, ("eta_rho", "xi_rho")),
            ("angle", grid.angle, ("eta_rho", "xi_rho")), ("s_rho", grid.s_rho, ("s_rho",)),
            ("Cs_r", grid.cs_r, ("s_rho",)), ("s_w", grid.s_w, ("s_w",)), ("Cs_w", grid.cs_w, ("s_w",)),
        ):
            variable = ds.createVariable(name, "f8", dimensions)
            variable[:] = values
            if name == "angle":
                _set_angle_metadata(variable)
        # CBOFS compact files advertise these as global attributes; DBOFS
        # compact files use scalar variables (covered by the raw/movie tests).
        ds.setncattr("hc", grid.hc)
        ds.setncattr("Vtransform", grid.vtransform)
        ds.setncattr("Vstretching", grid.vstretching)
        for name, values, standard_name in (
            ("salinity_surface", salinity.values, "sea_water_practical_salinity"),
            ("eastward_velocity_surface", current.east, "eastward_sea_water_velocity"),
            ("northward_velocity_surface", current.north, "northward_sea_water_velocity"),
            ("current_speed_surface", current.speed, "sea_water_speed"),
        ):
            var = ds.createVariable(name, "f8", ("time", "eta_rho", "xi_rho"), fill_value=1.0e37)
            var.standard_name = standard_name
            var[:] = values
        ds.vector_provenance = "earth_relative_on_rho_grid"
        ds.derived_vector_reference = "earth_relative_on_rho_grid"
        ds.schema_version = "roms_compact_fields_v1"
        ds.velocity_processing = "vertically reduce native u/v, wet-aware destagger to rho, rotate by angle"


def write_permuted_raw(path: Path, hours):
    """Write legal ROMS variables with deliberately noncanonical dimension order."""

    hours = np.asarray(hours)
    lon, lat, mask, h = _grid()
    zeta_values = np.stack([np.full((ETA, XI), 0.05 * np.sin(hour / 3.0)) for hour in hours])
    salt_values = np.stack([
        np.stack([10.0 + hour * 0.01 + level * 10.0 + 0.02 * lon for level in range(LEVELS)])
        for hour in hours
    ])
    u_values = np.stack([
        np.stack([1.0 + hour * 0.01 + level * 0.1 + np.zeros((ETA, XI - 1))
                  for level in range(LEVELS)]) for hour in hours
    ])
    v_values = np.stack([
        np.stack([2.0 + hour * 0.01 + level * 0.2 + np.zeros((ETA - 1, XI))
                  for level in range(LEVELS)]) for hour in hours
    ])
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        for name, size in (("ocean_time", None), ("s_rho", LEVELS), ("s_w", LEVELS + 1),
                           ("eta_rho", ETA), ("xi_rho", XI), ("eta_u", ETA), ("xi_u", XI - 1),
                           ("eta_v", ETA - 1), ("xi_v", XI)):
            ds.createDimension(name, size)
        time = ds.createVariable("ocean_time", "f8", ("ocean_time",))
        time.units = "seconds since 1970-01-01 00:00:00 UTC"
        time[:] = hours * 3600.0 + 1784505600.0
        for name, values, dimensions in (
            ("lon_rho", lon.T, ("xi_rho", "eta_rho")),
            ("lat_rho", lat.T, ("xi_rho", "eta_rho")),
            ("mask_rho", mask.T, ("xi_rho", "eta_rho")),
            ("h", h.T, ("xi_rho", "eta_rho")),
            ("angle", np.zeros_like(h).T, ("xi_rho", "eta_rho")),
            ("mask_u", (mask[:, :-1] * mask[:, 1:]).T, ("xi_u", "eta_u")),
            ("mask_v", (mask[:-1, :] * mask[1:, :]).T, ("xi_v", "eta_v")),
            ("s_rho", S_RHO, ("s_rho",)), ("Cs_r", S_RHO, ("s_rho",)),
            ("s_w", S_W, ("s_w",)), ("Cs_w", S_W, ("s_w",)),
        ):
            dtype = "i1" if name.startswith("mask_") else "f8"
            variable = ds.createVariable(name, dtype, dimensions)
            variable[:] = values
            if name == "angle":
                _set_angle_metadata(variable)
        ds.createVariable("hc", "f8")[:] = 2.0
        ds.createVariable("Vtransform", "i4")[:] = 1
        ds.createVariable("Vstretching", "i4")[:] = 1
        zeta = ds.createVariable("zeta", "f8", ("xi_rho", "ocean_time", "eta_rho"))
        salt = ds.createVariable("salt", "f8", ("xi_rho", "s_rho", "ocean_time", "eta_rho"), fill_value=1.0e37)
        u = ds.createVariable("u", "f8", ("xi_u", "ocean_time", "eta_u", "s_rho"), fill_value=1.0e37)
        v = ds.createVariable("v", "f8", ("s_rho", "xi_v", "eta_v", "ocean_time"), fill_value=1.0e37)
        salt.long_name = "practical salinity"
        u.standard_name, u.units = "sea_water_x_velocity", "m s-1"
        v.standard_name, v.units = "sea_water_y_velocity", "m s-1"
        zeta[:] = np.transpose(zeta_values, (2, 0, 1))
        salt[:] = np.transpose(salt_values, (3, 1, 0, 2))
        u[:] = np.transpose(u_values, (3, 0, 2, 1))
        v[:] = np.transpose(v_values, (1, 3, 2, 0))


def _expect_error(callable_, text):
    try:
        callable_()
    except Exception as error:
        assert text.lower() in str(error).lower(), (text, error)
    else:
        raise AssertionError(f"Expected an error containing {text!r}.")


def _packed_seam_footprint_regression():
    """Prove dry packed coordinates cannot stretch an independent wet footprint."""

    x = np.arange(5, dtype=float)
    lon = np.vstack((
        -76.0 + 0.01 * x,
        -76.0 + 0.01 * x,
        -74.0 + 0.01 * x,  # unrelated packed coordinates: this row is dry
        -74.0 + 0.01 * x,
    ))
    lat = np.vstack((
        np.full(5, 38.00),
        np.full(5, 38.01),
        np.full(5, 39.00),
        np.full(5, 39.01),
    ))
    mask = np.ones((4, 5), dtype=np.int8)
    mask[2, :] = 0
    footprints = build_wet_cell_footprints(lon, lat, mask, angle=np.zeros_like(lon))
    wet = mask == 1
    centers = footprints.polygons.mean(axis=1)
    expected_centers = np.column_stack((lon[wet], lat[wet]))
    assert len(footprints.polygons) == int(np.count_nonzero(wet)) == 15
    assert np.allclose(centers, expected_centers, atol=1.0e-12)
    assert np.max(np.ptp(footprints.polygons[:, :, 0], axis=1)) < 0.02
    assert np.max(np.ptp(footprints.polygons[:, :, 1], axis=1)) < 0.02
    assert footprints.maximum_span_km < 2.0
    assert int(np.count_nonzero(footprints.fallback_cells)) == 5
    return footprints


def main() -> int:
    regression_checks = []
    with tempfile.TemporaryDirectory(prefix="roms_map_selftest_") as temporary:
        root = Path(temporary)
        raw = [root / f"raw_{index}.nc" for index in range(3)]
        write_raw(raw[0], range(0, 8))
        write_raw(raw[1], range(7, 16))
        write_raw(raw[2], range(16, 24))

        noaa_roundoff = root / "noaa_cs_w_roundoff.nc"
        write_raw(noaa_roundoff, [0])
        with Dataset(noaa_roundoff, "r+") as dataset:
            cs_w = dataset.variables["Cs_w"]
            cs_w.valid_min = -1.0
            cs_w.valid_max = 0.0
            cs_w[-1] = np.nextafter(0.0, 1.0)
        noaa_series = load_scalar_series([noaa_roundoff], variable="salinity", layer="surface")
        assert np.isfinite(noaa_series.grid.cs_w).all()
        assert noaa_series.grid.cs_w[-1] > 0.0
        regression_checks.append("coordinate_valid_range_roundoff_preserved")

        packed_footprints = _packed_seam_footprint_regression()
        regression_checks.append("packed_dry_seam_independent_wet_cell_footprints")

        inspection = inspect_inputs(raw)
        assert inspection["geometry"]["angle_units"] == "radians"
        assert inspection["geometry"]["angle_convention"] == "xi_axis_counterclockwise_from_east"
        assert inspection["combined_time"]["raw_record_count"] == 25
        assert inspection["combined_time"]["unique_record_count"] == 24
        surface = load_scalar_series(raw, variable="salinity", layer="surface")
        assert surface.values.shape == (24, ETA, XI)
        assert surface.duplicate_times_removed == 1
        assert np.all(np.diff(surface.times.astype("int64")) > 0)
        current = load_current_series(raw, layer="surface")
        wet = current.grid.mask == 1
        assert np.allclose(current.east[:, wet], 1.2 + np.arange(24)[:, None] * 0.01)
        assert np.allclose(current.north[:, wet], 2.4 + np.arange(24)[:, None] * 0.01)
        assert np.allclose(current.speed, np.hypot(current.east, current.north), equal_nan=True)

        ordered, permuted = root / "ordered.nc", root / "permuted.nc"
        write_raw(ordered, [0, 1])
        write_permuted_raw(permuted, [0, 1])
        ordered_surface = load_scalar_series([ordered], variable="salinity", layer="surface")
        permuted_surface = load_scalar_series([permuted], variable="salinity", layer="surface")
        ordered_average = load_scalar_series([ordered], variable="salinity", layer="depth_average")
        permuted_average = load_scalar_series([permuted], variable="salinity", layer="depth_average")
        ordered_current = load_current_series([ordered], layer="depth_average")
        permuted_current = load_current_series([permuted], layer="depth_average")
        assert ordered_surface.grid.geometry_sha256 == permuted_surface.grid.geometry_sha256
        assert np.allclose(ordered_surface.values, permuted_surface.values, equal_nan=True)
        assert np.allclose(ordered_average.values, permuted_average.values, equal_nan=True)
        assert np.allclose(ordered_current.east, permuted_current.east, equal_nan=True)
        assert np.allclose(ordered_current.north, permuted_current.north, equal_nan=True)
        regression_checks.append("arbitrary_named_dimension_order")

        depth_average = load_scalar_series([raw[0]], variable="salinity", layer="depth_average")
        assert np.isfinite(depth_average.values[:, wet]).all()
        missing = root / "missing.nc"
        write_raw(missing, [0], missing_layer=True)
        missing_average = load_scalar_series([missing], variable="salinity", layer="depth_average")
        assert np.isfinite(missing_average.values[0, 2, 2])

        reversed_path = root / "reversed.nc"
        write_raw(reversed_path, [0], reverse=True)
        reversed_surface = load_scalar_series([reversed_path], variable="salinity", layer="surface")
        assert np.allclose(reversed_surface.values[0, wet], surface.values[0, wet])

        v2_path = root / "vtransform2.nc"
        write_raw(v2_path, [0], vtransform=2)
        v2_average = load_scalar_series([v2_path], variable="salinity", layer="depth_average")
        assert np.isfinite(v2_average.values[0, wet]).all()
        lon, lat, mask, h = _grid()
        zeta = np.full((2, ETA, XI), 0.25)
        for transform in (1, 2):
            levels = roms_depths(h, zeta, S_W, S_W, 2.0, transform)
            thickness = np.abs(np.diff(levels, axis=1))
            assert np.allclose(np.sum(thickness, axis=1), h[None] + zeta)

        u_native = np.arange(ETA * (XI - 1), dtype=float).reshape(ETA, XI - 1)
        v_native = np.arange((ETA - 1) * XI, dtype=float).reshape(ETA - 1, XI)
        assert destagger_u_to_rho(u_native, (ETA, XI)).shape == (ETA, XI)
        assert destagger_v_to_rho(v_native, (ETA, XI)).shape == (ETA, XI)
        east, north = rotate_to_earth(np.ones((ETA, XI)), 2 * np.ones((ETA, XI)), np.full((ETA, XI), np.pi / 2))
        assert np.allclose(east, -2.0) and np.allclose(north, 1.0)

        rotated = root / "rotated.nc"
        write_raw(rotated, [0], angle=np.pi / 2)
        rotated_current = load_current_series([rotated], layer="surface")
        assert np.allclose(rotated_current.east[0, wet], -2.4)
        assert np.allclose(rotated_current.north[0, wet], 1.2)

        missing_v = root / "missing_v.nc"
        write_raw(missing_v, [0], include_v=False)
        _expect_error(lambda: load_current_series([missing_v]), "paired")
        missing_angle = root / "missing_angle.nc"
        write_raw(missing_angle, [0], include_angle=False)
        _expect_error(lambda: inspect_inputs([missing_angle]), "angle")
        missing_mask = root / "missing_mask.nc"
        write_raw(missing_mask, [0], include_mask=False)
        _expect_error(lambda: inspect_inputs([missing_mask]), "mask_rho")
        regression_checks.append("mask_rho_required")

        invalid_angle_units = root / "invalid_angle_units.nc"
        write_raw(invalid_angle_units, [0])
        with Dataset(invalid_angle_units, "r+") as ds:
            ds.variables["angle"].units = "degrees"
        _expect_error(lambda: inspect_inputs([invalid_angle_units]), "radian units")
        ambiguous_angle = root / "ambiguous_angle.nc"
        write_raw(ambiguous_angle, [0])
        with Dataset(ambiguous_angle, "r+") as ds:
            variable = ds.variables["angle"]
            variable.delncattr("standard_name")
            variable.long_name = "grid orientation"
        _expect_error(lambda: inspect_inputs([ambiguous_angle]), "ambiguous semantics")
        out_of_range_angle = root / "out_of_range_angle.nc"
        write_raw(out_of_range_angle, [0], angle=2.0 * np.pi + 0.01)
        _expect_error(lambda: inspect_inputs([out_of_range_angle]), "radian range")
        regression_checks.append("strict_angle_metadata_and_range")

        nonbinary_rho = root / "nonbinary_rho.nc"
        write_raw(nonbinary_rho, [0])
        with Dataset(nonbinary_rho, "r+") as ds:
            ds.variables["mask_rho"][1, 1] = 2
        _expect_error(lambda: inspect_inputs([nonbinary_rho]), "binary 0/1")
        nonbinary_u = root / "nonbinary_u.nc"
        write_raw(nonbinary_u, [0])
        with Dataset(nonbinary_u, "r+") as ds:
            ds.variables["mask_u"][1, 1] = 2
        _expect_error(lambda: load_current_series([nonbinary_u]), "binary 0/1")
        regression_checks.append("strict_binary_rho_and_native_masks")
        _expect_error(lambda: load_scalar_series([raw[0], v2_path], variable="salinity"), "drift")

        compact = root / "compact.nc"
        write_compact(compact, raw)
        compact_salinity = load_scalar_series([compact], variable="salinity", layer="surface")
        compact_current = load_current_series([compact], layer="surface")
        assert np.allclose(compact_salinity.values, surface.values, equal_nan=True)
        assert np.allclose(compact_current.speed, current.speed, equal_nan=True)
        assert compact_current.resolution["vector_provenance"] == "earth_relative_on_rho_grid"
        assert compact_current.resolution["compact_current_provenance"]["schema_version"] == "roms_compact_fields_v1"
        assert "velocity_processing" in compact_current.resolution["compact_current_provenance"]
        assert compact_current.resolution["input_kind"] == "compact"
        assert compact_current.resolution["angle_units"] == "radians"
        assert compact_current.resolution["angle_convention"] == "xi_axis_counterclockwise_from_east"
        assert compact_current.resolution["verified_speed_variable"] == "current_speed_surface"
        regression_checks.append("compact_current_provenance_preserved")

        ambiguous_raw = root / "ambiguous_raw_vectors.nc"
        write_raw(ambiguous_raw, [0])
        with Dataset(ambiguous_raw, "r+") as ds:
            for name in ("eastward_velocity_surface", "northward_velocity_surface"):
                ds.createVariable(name, "f8", ("ocean_time", "eta_rho", "xi_rho"))[:] = 1.0
        _expect_error(lambda: load_current_series([ambiguous_raw]), "requires schema_version")

        bad_provenance = root / "bad_compact_provenance.nc"
        write_compact(bad_provenance, raw)
        with Dataset(bad_provenance, "r+") as ds:
            ds.delncattr("derived_vector_reference")
            ds.delncattr("vector_provenance")
        _expect_error(lambda: load_current_series([bad_provenance]), "earth-relative vector provenance")

        bad_speed = root / "bad_compact_speed.nc"
        write_compact(bad_speed, raw)
        with Dataset(bad_speed, "r+") as ds:
            ds.variables["current_speed_surface"][0, 1, 1] += 0.25
        _expect_error(
            lambda: load_scalar_series([bad_speed], variable="current_speed", layer="surface"),
            "hypot(east,north)",
        )
        regression_checks.append("strict_compact_vector_provenance_and_speed_pair")

        inspection_json = root / "inspection.json"
        assert map_main(["inspect", "--input", *(str(path) for path in raw), "--output", str(inspection_json)]) == 0
        assert json.loads(inspection_json.read_text())["combined_time"]["unique_record_count"] == 24
        protected_hash = hashlib.sha256(raw[0].read_bytes()).hexdigest()
        assert map_main(["inspect", "--input", str(raw[0]), "--output", str(raw[0])]) == 2
        assert hashlib.sha256(raw[0].read_bytes()).hexdigest() == protected_hash
        regression_checks.append("inspect_input_collision_rejected")
        with Dataset(raw[0], "r+") as ds:
            ds.variables["salt"].units = "first-source-unit"
            ds.variables["salt"].long_name = "first-source salinity"
        with Dataset(raw[1], "r+") as ds:
            ds.variables["salt"].units = "selected-source-unit"
            ds.variables["salt"].long_name = "selected-source salinity"
        salinity_png, salinity_json = root / "salinity.png", root / "salinity.json"
        map_args = ["map", "--input", *(str(path) for path in raw), "--variable", "salinity",
                    "--time", "2026-07-20T12:00:00Z", "--layer", "surface", "--limits-scope", "series",
                    "--output", str(salinity_png), "--report", str(salinity_json), "--dpi", "80"]
        assert map_main(map_args) == 0
        current_png, current_json = root / "current.png", root / "current.json"
        current_args = ["map", "--input", str(compact), "--variable", "current_speed", "--time-index", "12",
                        "--layer", "surface", "--quiver", "current", "--quiver-stride", "2",
                        "--vmin", "0", "--vmax", "5", "--output", str(current_png),
                        "--report", str(current_json), "--dpi", "80"]
        assert map_main(current_args) == 0
        for png in (salinity_png, current_png):
            with Image.open(png) as image:
                assert image.width > 100 and image.height > 100
                assert np.asarray(image.convert("RGB")).std() > 1.0
        salinity_manifest = json.loads(salinity_json.read_text())
        current_manifest = json.loads(current_json.read_text())
        assert salinity_manifest["rendering"]["finite_wet_coverage"] >= 0.95
        assert salinity_manifest["rendering"]["style"] == "wet_cells"
        assert salinity_manifest["rendering"]["wet_mask_rule"] == "mask_rho==1_and_finite_scalar"
        assert salinity_manifest["rendering"]["rendered_cells"] == salinity_manifest["rendering"]["finite_wet_cells"]
        assert salinity_manifest["rendering"]["footprint_method"] == packed_footprints.spacing_method
        assert salinity_manifest["rendering"]["footprint_maximum_span_km"] > 0
        assert current_manifest["rendering"]["quiver_count"] > 0
        assert current_manifest["selection"]["resolution"]["earth_relative_on_rho_grid"] is True
        assert current_manifest["selection"]["angle_units"] == "radians"
        assert current_manifest["selection"]["angle_convention"] == "xi_axis_counterclockwise_from_east"
        assert Path(salinity_manifest["selection"]["source"]) == raw[1].resolve()
        assert salinity_manifest["selection"]["units"] == "selected-source-unit"
        assert salinity_manifest["selection"]["long_name"] == "selected-source salinity"
        assert current_manifest["selection"]["long_name"] == "Earth-relative current speed"
        regression_checks.append("selected_record_metadata_and_current_label")

        unsafe_png, unsafe_json = root / "unsafe.png", root / "unsafe.json"
        assert map_main([
            "map", "--input", str(raw[0]), "--variable", "salinity", "--time-index", "0",
            "--layer", "surface", "--style", "pcolormesh", "--vmin", "0", "--vmax", "40",
            "--output", str(unsafe_png), "--report", str(unsafe_json),
        ]) == 2
        assert not unsafe_png.exists() and not unsafe_json.exists()
        regression_checks.append("masked_center_pcolormesh_rejected")

        all_nan = root / "all_nan.nc"
        write_raw(all_nan, [0])
        with Dataset(all_nan, "r+") as ds:
            variable = ds.variables["salt"]
            variable[:] = variable._FillValue
        rejected_png, rejected_json = root / "rejected.png", root / "rejected.json"
        assert map_main([
            "map", "--input", str(all_nan), "--variable", "salinity", "--time-index", "0",
            "--layer", "surface", "--vmin", "0", "--vmax", "40",
            "--output", str(rejected_png), "--report", str(rejected_json),
        ]) == 2
        assert not rejected_png.exists() and not rejected_json.exists()
        regression_checks.append("explicit_limits_zero_wet_coverage_rejected")

        schema = json.loads((Path(__file__).parents[1] / "references" / "map_manifest.schema.json").read_text())
        assert set(schema["required"]) <= set(salinity_manifest)
        assert {"source", "source_record_index", "original_time_utc"} <= set(schema["properties"]["selection"]["required"])
        assert schema["properties"]["rendering"]["properties"]["finite_wet_coverage"]["exclusiveMinimum"] == 0
        assert salinity_manifest["rendering"]["finite_wet_coverage"] > 0
        regression_checks.append("strengthened_map_manifest_schema")

        package_root = Path(__file__).resolve().parents[1]
        sibling = package_root.parent / "roms-movie-postprocessing" / "scripts"
        if not sibling.is_dir():
            sibling = package_root.parents[2] / "roms-movie-postprocessing" / "prototype" / "roms-movie-postprocessing" / "scripts"
        parity = {}
        if sibling.is_dir():
            for name in ("roms_output.py", "roms_map_tools.py"):
                left = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                right = hashlib.sha256((sibling / name).read_bytes()).hexdigest()
                assert left == right, f"Shared core drift: {name}"
                parity[name] = left
        print(json.dumps({"status": "pass", "tests": 44, "unique_frames": 24,
                          "regression_checks": regression_checks,
                          "raw_compact_parity": True, "shared_module_sha256": parity}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
