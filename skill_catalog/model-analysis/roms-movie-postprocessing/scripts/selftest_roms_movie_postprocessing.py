#!/usr/bin/env python3
"""Offline raw/compact and 24-frame GIF tests for ROMS movie postprocessing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from netCDF4 import Dataset
from PIL import Image

from roms_movie_postprocessing import _display_label, create_gif, main as movie_main
from roms_output import load_current_series, load_scalar_series


ETA, XI = 4, 5
S_RHO = np.array([-5.0 / 6.0, -0.5, -1.0 / 6.0])
S_W = np.array([-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0])


def _set_angle_metadata(variable):
    variable.units = "radians"
    variable.standard_name = "grid_angle_of_rotation_from_east_to_y"
    variable.long_name = "angle between XI-axis and EAST"


def write_raw(path: Path, start_hour: int, stop_hour: int):
    hours = np.arange(start_hour, stop_hour)
    y, x = np.mgrid[:ETA, :XI]
    lon = -75.5 + x * 0.08 + y * 0.004
    lat = 38.2 + y * 0.07 + x * y * 0.002
    mask = np.ones((ETA, XI), dtype=np.int8)
    mask[0, 0] = 0
    h = 10.0 + x + 0.5 * y
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        for name, size in (("ocean_time", None), ("s_rho", 3), ("s_w", 4),
                           ("eta_rho", ETA), ("xi_rho", XI), ("eta_u", ETA),
                           ("xi_u", XI - 1), ("eta_v", ETA - 1), ("xi_v", XI)):
            ds.createDimension(name, size)
        time = ds.createVariable("ocean_time", "f8", ("ocean_time",))
        time.units = "seconds since 1970-01-01 00:00:00 UTC"
        time[:] = 1784505600.0 + hours * 3600.0
        for name, values, dimensions, dtype in (
            ("lon_rho", lon, ("eta_rho", "xi_rho"), "f8"),
            ("lat_rho", lat, ("eta_rho", "xi_rho"), "f8"),
            ("mask_rho", mask, ("eta_rho", "xi_rho"), "i1"),
            ("mask_u", mask[:, :-1] * mask[:, 1:], ("eta_u", "xi_u"), "i1"),
            ("mask_v", mask[:-1] * mask[1:], ("eta_v", "xi_v"), "i1"),
            ("h", h, ("eta_rho", "xi_rho"), "f8"),
            ("angle", np.full_like(h, 0.2), ("eta_rho", "xi_rho"), "f8"),
            ("s_rho", S_RHO, ("s_rho",), "f8"), ("Cs_r", S_RHO, ("s_rho",), "f8"),
            ("s_w", S_W, ("s_w",), "f8"), ("Cs_w", S_W, ("s_w",), "f8"),
        ):
            variable = ds.createVariable(name, dtype, dimensions)
            variable[:] = values
            if name == "angle":
                _set_angle_metadata(variable)
        ds.createVariable("hc", "f8")[:] = 2.0
        ds.createVariable("Vtransform", "i4")[:] = 1
        ds.createVariable("Vstretching", "i4")[:] = 1
        zeta = ds.createVariable("zeta", "f8", ("ocean_time", "eta_rho", "xi_rho"))
        salt = ds.createVariable("salt", "f8", ("ocean_time", "s_rho", "eta_rho", "xi_rho"), fill_value=1.0e37)
        salt.long_name = "practical salinity"
        u = ds.createVariable("u", "f8", ("ocean_time", "s_rho", "eta_u", "xi_u"), fill_value=1.0e37)
        v = ds.createVariable("v", "f8", ("ocean_time", "s_rho", "eta_v", "xi_v"), fill_value=1.0e37)
        u.standard_name, v.standard_name = "sea_water_x_velocity", "sea_water_y_velocity"
        for local, hour in enumerate(hours):
            zeta[local] = 0.04 * np.sin(hour / 4)
            for level in range(3):
                salt[local, level] = 15 + 5 * level + 0.03 * hour + 0.02 * x + 0.01 * y
                u[local, level] = 0.3 + 0.1 * level + 0.015 * hour + np.zeros((ETA, XI - 1))
                v[local, level] = 0.2 + 0.05 * level - 0.004 * hour + np.zeros((ETA - 1, XI))


def write_compact(path: Path, raw_paths):
    salinity = load_scalar_series(raw_paths, variable="salinity", layer="surface")
    current = load_current_series(raw_paths, layer="surface")
    grid = salinity.grid
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        for name, size in (("time", len(salinity.times)), ("s_rho", 3), ("s_w", 4),
                           ("eta_rho", ETA), ("xi_rho", XI)):
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
        ds.createVariable("hc", "f8")[:] = grid.hc
        ds.createVariable("Vtransform", "i4")[:] = grid.vtransform
        ds.createVariable("Vstretching", "i4")[:] = grid.vstretching
        for name, values in (
            ("salinity_surface", salinity.values), ("eastward_velocity_surface", current.east),
            ("northward_velocity_surface", current.north), ("current_speed_surface", current.speed),
        ):
            ds.createVariable(name, "f8", ("time", "eta_rho", "xi_rho"), fill_value=1.0e37)[:] = values
        ds.schema_version = "roms_compact_fields_v1"
        ds.vector_reference = "earth_relative_on_rho_grid"
        ds.velocity_processing = "vertical view on native C-grid; finite-aware destagger to rho; angle rotation"


def main() -> int:
    regression_checks = []
    with tempfile.TemporaryDirectory(prefix="roms_movie_selftest_") as temporary:
        root = Path(temporary)
        raw_a, raw_b = root / "raw_a.nc", root / "raw_b.nc"
        write_raw(raw_a, 0, 13)
        write_raw(raw_b, 12, 24)
        raw_paths = [raw_a, raw_b]

        noaa_roundoff = root / "noaa_cs_w_roundoff.nc"
        write_raw(noaa_roundoff, 0, 1)
        with Dataset(noaa_roundoff, "r+") as dataset:
            cs_w = dataset.variables["Cs_w"]
            cs_w.valid_min = -1.0
            cs_w.valid_max = 0.0
            cs_w[-1] = np.nextafter(0.0, 1.0)
        noaa_series = load_scalar_series([noaa_roundoff], variable="salinity", layer="surface")
        assert np.isfinite(noaa_series.grid.cs_w).all()
        assert noaa_series.grid.cs_w[-1] > 0.0
        regression_checks.append("coordinate_valid_range_roundoff_preserved")

        compact = root / "compact.nc"
        write_compact(compact, raw_paths)

        inspection = root / "inspection.json"
        assert movie_main(["inspect", "--input", str(raw_a), str(raw_b), "--output", str(inspection)]) == 0
        assert json.loads(inspection.read_text())["combined_time"]["unique_record_count"] == 24
        inspection_payload = json.loads(inspection.read_text())
        assert inspection_payload["geometry"]["angle_units"] == "radians"
        assert inspection_payload["geometry"]["angle_convention"] == "xi_axis_counterclockwise_from_east"
        protected_hash = hashlib.sha256(raw_a.read_bytes()).hexdigest()
        assert movie_main(["inspect", "--input", str(raw_a), "--output", str(raw_a)]) == 2
        assert hashlib.sha256(raw_a.read_bytes()).hexdigest() == protected_hash
        regression_checks.append("inspect_input_collision_rejected")

        input_collision_output = root / "input_collision.gif"
        try:
            create_gif([raw_a], variable="salinity", output=input_collision_output, report=raw_a,
                       dpi=55, figure_size=(4.0, 3.5))
        except ValueError as error:
            assert "distinct" in str(error).lower()
        else:
            raise AssertionError("Expected report/input collision rejection.")
        assert not input_collision_output.exists()
        same_target = root / "same_target.gif"
        try:
            create_gif([raw_a], variable="salinity", output=same_target, report=same_target,
                       dpi=55, figure_size=(4.0, 3.5))
        except ValueError as error:
            assert "distinct" in str(error).lower()
        else:
            raise AssertionError("Expected GIF/report collision rejection.")
        assert not same_target.exists()
        gif_input = root / "netcdf_input.gif"
        write_raw(gif_input, 0, 2)
        gif_input_hash = hashlib.sha256(gif_input.read_bytes()).hexdigest()
        try:
            create_gif([gif_input], variable="salinity", output=gif_input, report=root / "unused.json",
                       dpi=55, figure_size=(4.0, 3.5))
        except ValueError as error:
            assert "distinct" in str(error).lower()
        else:
            raise AssertionError("Expected GIF/input collision rejection.")
        assert hashlib.sha256(gif_input.read_bytes()).hexdigest() == gif_input_hash
        regression_checks.append("gif_report_and_input_collisions_rejected_before_writes")

        salinity_gif, salinity_report = root / "salinity.gif", root / "salinity.json"
        salinity = create_gif(
            raw_paths, variable="salinity", layer="surface", start="2026-07-20T00:00:00Z",
            end_exclusive="2026-07-21T00:00:00Z", fps=4, quantiles=(2, 98),
            output=salinity_gif, report=salinity_report, dpi=55, figure_size=(4.0, 3.5))
        assert salinity["selection"]["frame_count"] == 24
        assert salinity["selection"]["duplicate_times_removed"] == 1
        assert salinity["selection"]["distinct_rendered_frame_count"] == 24
        assert salinity["coverage"]["minimum_finite_wet_fraction"] >= 0.95
        assert salinity["rendering"]["temporary_frames_cleaned"] is True

        current_gif, current_report = root / "current.gif", root / "current.json"
        current = create_gif([compact], variable="current_speed", layer="surface", fps=5,
                             vmin=0.0, vmax=1.5, output=current_gif, report=current_report,
                             dpi=55, figure_size=(4.0, 3.5))
        assert current["selection"]["frame_count"] == 24
        assert current["selection"]["distinct_rendered_frame_count"] == 24
        assert current["fixed_color_limits"]["method"] == "explicit"
        assert current["resolved"]["earth_relative_on_rho_grid"] is True
        assert current["resolved"]["vector_reference"] == "earth_relative_on_rho_grid"
        assert current["resolved"]["compact_current_provenance"]["schema_version"] == "roms_compact_fields_v1"
        assert current["resolved"]["verified_speed_variable"] == "current_speed_surface"
        assert current["resolved"]["angle_units"] == "radians"
        assert current["resolved"]["angle_convention"] == "xi_axis_counterclockwise_from_east"
        assert current["grid"]["angle_units"] == "radians"
        assert current["grid"]["angle_convention"] == "xi_axis_counterclockwise_from_east"
        assert _display_label("current_speed") == "Earth-relative current speed"
        regression_checks.append("compact_current_provenance_and_current_label")
        for path in (salinity_gif, current_gif):
            with Image.open(path) as image:
                assert image.n_frames == 24 and image.width > 100 and image.height > 100

        invalid_angle = root / "invalid_angle.nc"
        write_raw(invalid_angle, 0, 2)
        with Dataset(invalid_angle, "r+") as ds:
            ds.variables["angle"].units = "degrees"
        try:
            load_scalar_series([invalid_angle], variable="salinity", layer="surface")
        except ValueError as error:
            assert "radian units" in str(error).lower()
        else:
            raise AssertionError("Expected invalid angle-unit rejection.")

        bad_speed = root / "bad_speed.nc"
        write_compact(bad_speed, raw_paths)
        with Dataset(bad_speed, "r+") as ds:
            ds.variables["current_speed_surface"][0, 1, 1] += 0.25
        try:
            load_scalar_series([bad_speed], variable="current_speed", layer="surface")
        except ValueError as error:
            assert "hypot(east,north)" in str(error)
        else:
            raise AssertionError("Expected inconsistent compact speed rejection.")
        regression_checks.append("strict_angle_and_compact_speed_gates")

        identical = root / "identical.nc"
        write_raw(identical, 0, 2)
        with Dataset(identical, "r+") as ds:
            ds.variables["salt"][1] = ds.variables["salt"][0]
        identical_gif, identical_report = root / "identical.gif", root / "identical.json"
        try:
            identical_manifest = create_gif(
                [identical], variable="salinity", layer="surface", fps=4, vmin=0, vmax=40,
                output=identical_gif, report=identical_report, dpi=55, figure_size=(4.0, 3.5),
                title_template="constant title")
        except RuntimeError as error:
            assert "expected 2" in str(error)
            assert not identical_gif.exists() and not identical_report.exists()
            identical_behavior = "atomic_failure_without_output"
        else:
            assert identical_manifest["selection"]["frame_count"] == 2
            assert identical_manifest["output"]["frame_count"] == 2
            with Image.open(identical_gif) as image:
                assert image.n_frames == 2
            identical_behavior = "declared_frame_count_preserved"
        regression_checks.append(f"identical_frames_{identical_behavior}")

        movie_schema = json.loads((Path(__file__).parents[1] / "references" / "movie_manifest.schema.json").read_text())
        assert set(movie_schema["required"]) <= set(current)
        assert movie_schema["properties"]["coverage"]["properties"]["minimum_finite_wet_fraction"]["exclusiveMinimum"] == 0
        assert movie_schema["properties"]["rendering"]["properties"]["temporary_frames_cleaned"]["const"] is True
        assert len(current["frames"]) == current["output"]["frame_count"] == current["selection"]["frame_count"]
        regression_checks.append("movie_manifest_schema")

        package_root = Path(__file__).resolve().parents[1]
        sibling = package_root.parent / "roms-map-postprocessing" / "scripts"
        if not sibling.is_dir():
            sibling = package_root.parents[2] / "roms-map-postprocessing" / "prototype" / "roms-map-postprocessing" / "scripts"
        parity = {}
        for name in ("roms_output.py", "roms_map_tools.py"):
            left = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            right = hashlib.sha256((sibling / name).read_bytes()).hexdigest()
            assert left == right, f"Shared core drift: {name}"
            parity[name] = left
        print(json.dumps({"status": "pass", "raw_gif_frames": 24, "compact_gif_frames": 24,
                          "distinct_frames": 24, "regression_checks": regression_checks,
                          "shared_module_sha256": parity}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
