#!/usr/bin/env python3
"""Offline EFDC loader, vertical, footprint, and static-map regression tests."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile

import numpy as np
from netCDF4 import Dataset
from PIL import Image

import efdc_map_postprocessing as map_module
from efdc_map_tools import build_wet_cell_footprints, plot_efdc_scalar
from efdc_map_postprocessing import main as map_main
from efdc_output import load_scalar_series, normalize_times, sigma_weights


def validate_schema_instance(schema, value, path="$"):
    """Validate the Draft-2020-12 subset used by bundled evidence schemas."""
    if "const" in schema:
        assert value == schema["const"], f"{path}: expected const {schema['const']!r}"
    if "enum" in schema:
        assert value in schema["enum"], f"{path}: not in enum"
    kind = schema.get("type")
    types = kind if isinstance(kind, list) else [kind] if kind else []
    checks = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
              "string": lambda v: isinstance(v, str), "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool), "null": lambda v: v is None}
    if types:
        assert any(checks[item](value) for item in types), f"{path}: wrong type"
    if isinstance(value, dict):
        for required in schema.get("required", []):
            assert required in value, f"{path}: missing {required}"
        for name, child in schema.get("properties", {}).items():
            if name in value:
                validate_schema_instance(child, value[name], f"{path}.{name}")
    if isinstance(value, list):
        assert len(value) >= schema.get("minItems", 0), f"{path}: too few items"
        if "items" in schema:
            for index, child_value in enumerate(value):
                validate_schema_instance(schema["items"], child_value, f"{path}[{index}]")
    if isinstance(value, (int, float)) and "minimum" in schema:
        assert value >= schema["minimum"], f"{path}: below minimum"
    if isinstance(value, str) and "pattern" in schema:
        assert re.search(schema["pattern"], value), f"{path}: pattern mismatch"


SIGMA = np.array([0.75, 0.0, 0.25], dtype=np.float32)  # deliberately reversed/permuted


def write_raw(path: Path, hours=(0, 1, 2), *, outside_value=False, bad_vector=False, drift=False,
              missing_layer=False):
    ny, nx = 5, 7
    y, x = np.mgrid[:ny, :nx]
    lon = -81.7 + 0.02 * x + 0.002 * y * y
    lat = 29.6 + 0.025 * y + 0.001 * x * y
    if drift:
        lon[2, 2] += 0.1
    mask = np.zeros((ny, nx), dtype=np.float32)
    mask[1:4, 1:6] = 5.0
    mask[2, 3] = 0.0  # land hole
    mask[0, 5] = 5.0  # one-cell-wide branch, attached only along logical eta
    mask[0, 0] = -85.48875
    # Packed dry coordinates are deliberately corrupt and must be ignored.
    lon[mask != 5] = 0.0
    lat[mask != 5] = 0.1
    depth = np.where(mask == 5, 4.0 + x + y, 0.0102)
    with Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        for name, size in (("time", len(hours)), ("sigma", 3), ("ny", ny), ("nx", nx)):
            ds.createDimension(name, size)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "seconds since 1970-01-01 00:00:00 UTC"
        t[:] = [h * 3600 + 1800 + (20 if i % 2 else -20) for i, h in enumerate(hours)]
        for name, values in (("lon", lon), ("lat", lat), ("mask", mask), ("depth", depth)):
            ds.createVariable(name, "f4", ("ny", "nx"))[:] = values
        ds.createVariable("sigma", "f4", ("sigma",))[:] = SIGMA
        shape = (len(hours), 3, ny, nx)
        wet = mask == 5
        salt_values = np.full(shape, -99999.0, dtype=np.float32)
        u_values = np.full(shape, -99999.0, dtype=np.float32)
        v_values = np.full(shape, -99999.0, dtype=np.float32)
        # Storage-order values: sigma .75 -> 30, sigma 0 -> 10, sigma .25 -> 20.
        for ti in range(len(hours)):
            for k, base in enumerate((30.0, 10.0, 20.0)):
                salt_values[ti, k, wet] = base + ti
                u_values[ti, k, wet] = (k + 1) + ti
                v_values[ti, k, wet] = 2 * (k + 1) + ti
        if outside_value:
            salt_values[0, 0, 1, 0] = 7.0
        if missing_layer:
            salt_values[0, 0, 1, 1] = -99999.0
        for name, values, standard in (
            ("salt", salt_values, "sea_water_practical_salinity"),
            ("u", u_values, "eastward_sea_water_velocity"),
            ("v", v_values, "northward_sea_water_velocity"),
        ):
            var = ds.createVariable(name, "f4", ("time", "sigma", "ny", "nx"), fill_value=-99999.0)
            var.missing_value = -99999.0
            var.set_auto_mask(False)
            var.standard_name = "grid_x_velocity" if bad_vector and name == "u" else standard
            var.units = "ppt" if name == "salt" else "m s-1"
            var[:] = values
        for name, standard, scale in (("air_u", "eastward_wind", 1.0), ("air_v", "northward_wind", 2.0)):
            values = np.stack([scale * (1 + x + y + ti) for ti in range(len(hours))]).astype(np.float32)
            var = ds.createVariable(name, "f4", ("time", "ny", "nx"))
            var.standard_name = standard
            var.units = "m s-1"
            var[:] = values


def write_compact(path: Path, source: Path, *, bad_wet_mask=False, outside_value=False):
    sal = load_scalar_series([source], variable="salinity", layer="surface")
    cur = load_scalar_series([source], variable="current_speed", layer="surface")
    east = load_scalar_series([source], variable="u", layer="surface")
    north = load_scalar_series([source], variable="v", layer="surface")
    with Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        nt, ny, nx = sal.values.shape
        for name, size in (("time", nt), ("sigma", len(sal.grid.sigma)), ("ny", ny), ("nx", nx)):
            ds.createDimension(name, size)
        ds.schema_version = "efdc_compact_fields_v1"
        ds.vertical_method = "efdc_layer_top_sigma_with_bed_edge_1"
        ds.vector_components = "collocated earth-relative"
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "seconds since 1970-01-01 00:00:00 UTC"
        t[:] = sal.times.astype("datetime64[s]").astype(np.int64)
        for name, values in (("lon", sal.grid.lon), ("lat", sal.grid.lat),
                             ("mask", sal.grid.source_mask), ("depth", sal.grid.depth)):
            ds.createVariable(name, "f4", ("ny", "nx"))[:] = values
        wet_values = sal.grid.mask.copy()
        if bad_wet_mask:
            wet_values[1, 1] = 0
        ds.createVariable("wet_mask", "i1", ("ny", "nx"))[:] = wet_values
        ds.createVariable("sigma", "f4", ("sigma",))[:] = sal.grid.sigma
        arrays = {
            "salinity_surface": sal.values.copy(),
            "current_speed_surface": cur.values.copy(),
            "eastward_velocity_surface": east.values.copy(),
            "northward_velocity_surface": north.values.copy(),
        }
        if outside_value:
            arrays["salinity_surface"][0, 1, 0] = 4.0
        metadata = {
            "salinity_surface": ("sea_water_practical_salinity", "ppt"),
            "current_speed_surface": ("sea_water_speed", "m s-1"),
            "eastward_velocity_surface": ("eastward_sea_water_velocity", "m s-1"),
            "northward_velocity_surface": ("northward_sea_water_velocity", "m s-1"),
        }
        for name, values in arrays.items():
            var = ds.createVariable(name, "f4", ("time", "ny", "nx"), fill_value=-99999)
            var.standard_name, var.units = metadata[name]
            var[:] = np.ma.masked_invalid(values)


def expect_error(fn, text):
    try:
        fn()
    except Exception as exc:
        assert text.lower() in str(exc).lower(), str(exc)
    else:
        raise AssertionError(f"Expected error containing {text!r}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="efdc_map_selftest_") as td:
        root = Path(td)
        raw = root / "raw.nc"
        write_raw(raw)
        expected_weights = np.array([0.25, 0.25, 0.5])
        np.testing.assert_allclose(sigma_weights(SIGMA), expected_weights)
        normalized, offsets, cadence = normalize_times(
            np.array(["2026-07-20T00:29:40", "2026-07-20T01:30:20"], dtype="datetime64[s]")
        )
        assert cadence == 3600 and list(offsets) == [20.0, -20.0]
        assert str(normalized[0]).startswith("2026-07-20T00:30")
        salinity = load_scalar_series([raw], variable="salinity", layer="depth_average")
        wet = salinity.grid.mask == 1
        # 30*.25 + 10*.25 + 20*.5 = 20
        np.testing.assert_allclose(salinity.values[0][wet], 20.0)
        speed = load_scalar_series([raw], variable="current_speed", layer="depth_average")
        mean_u = 1 * .25 + 2 * .25 + 3 * .5
        mean_v = 2 * .25 + 4 * .25 + 6 * .5
        np.testing.assert_allclose(speed.values[0][wet], np.hypot(mean_u, mean_v))
        wind = load_scalar_series([raw], variable="wind_speed")
        assert np.isfinite(wind.values[:, wet]).all() and np.isnan(wind.values[:, ~wet]).all()
        assert speed.resolution["vertical_method"] == "efdc_layer_top_sigma_with_bed_edge_1"
        footprints = build_wet_cell_footprints(salinity.grid.lon, salinity.grid.lat, salinity.grid.mask)
        assert len(footprints.polygons) == 15
        assert np.all(footprints.polygons[:, :, 0] < -81.0)  # dry zero never affects polygons
        png, report = root / "map.png", root / "map.json"
        assert map_main([
            "map", "--input", str(raw), "--variable", "current_speed", "--time-index", "1",
            "--layer", "surface", "--quiver", "current", "--quiver-stride", "2",
            "--limits-scope", "series", "--output", str(png), "--report", str(report),
        ]) == 0
        manifest = json.loads(report.read_text())
        schema_path = Path(__file__).resolve().parent.parent / "references" / "map_manifest.schema.json"
        schema = json.loads(schema_path.read_text())
        assert schema["$schema"].endswith("2020-12/schema")
        validate_schema_instance(schema, manifest)
        assert manifest["rendering"]["style"] == "wet_cells"
        assert manifest["rendering"]["wet_cells"] == 15
        assert manifest["rendering"]["quiver_count"] > 0
        with Image.open(png) as image:
            assert image.width > 100 and image.height > 100 and np.asarray(image).std() > 2
        later = root / "later.nc"
        write_raw(later, hours=(3, 4))
        later_png, later_report = root / "later.png", root / "later.json"
        assert map_main([
            "map", "--input", str(raw), "--input", str(later), "--variable", "salinity",
            "--time", "1970-01-01T04:30:00Z", "--layer", "surface",
            "--output", str(later_png), "--report", str(later_report),
        ]) == 0
        later_manifest = json.loads(later_report.read_text())
        assert Path(later_manifest["selection"]["source_record_path"]) == later.resolve()
        assert later_manifest["selection"]["long_name"] == "Practical salinity"
        assert later_manifest["selection"]["units"] == "ppt"
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        try:
            expect_error(lambda: plot_efdc_scalar(
                ax, lon=salinity.grid.lon, lat=salinity.grid.lat, mask=salinity.grid.mask,
                values=salinity.values[0], vmin=10, vmax=30, method="pcolormesh",
            ), "unsafe")
        finally:
            plt.close(fig)
        outside = root / "outside.nc"
        write_raw(outside, outside_value=True)
        expect_error(lambda: load_scalar_series([outside], variable="salinity"), "outside source mask == 5")
        bad_vector = root / "bad_vector.nc"
        write_raw(bad_vector, bad_vector=True)
        expect_error(lambda: load_scalar_series([bad_vector], variable="current_speed"), "collocated earth-relative")
        drift = root / "drift.nc"
        write_raw(drift, hours=(3,), drift=True)
        expect_error(lambda: load_scalar_series([raw, drift], variable="salinity"), "geometry drift")
        missing = root / "missing.nc"
        write_raw(missing, missing_layer=True)
        missing_mean = load_scalar_series([missing], variable="salinity", layer="depth_average")
        # At [1,1], bottom .75 layer is absent; remaining .25/.5 weights renormalize.
        np.testing.assert_allclose(missing_mean.values[0, 1, 1], (10*.25 + 20*.5) / .75)
        np.testing.assert_allclose(load_scalar_series([raw], variable="salinity", layer="surface").values[0][wet], 10)
        np.testing.assert_allclose(load_scalar_series([raw], variable="salinity", layer="bottom").values[0][wet], 30)
        compact = root / "compact.nc"
        write_compact(compact, raw)
        compact_sal = load_scalar_series([compact], variable="salinity", layer="surface")
        compact_cur = load_scalar_series([compact], variable="current_speed", layer="surface")
        np.testing.assert_allclose(compact_sal.values, load_scalar_series([raw], variable="salinity", layer="surface").values, equal_nan=True)
        np.testing.assert_allclose(compact_cur.values, load_scalar_series([raw], variable="current_speed", layer="surface").values, equal_nan=True)
        bad_compact_mask = root / "bad_compact_mask.nc"
        write_compact(bad_compact_mask, raw, bad_wet_mask=True)
        expect_error(lambda: load_scalar_series([bad_compact_mask], variable="salinity"), "disagrees")
        outside_compact = root / "outside_compact.nc"
        write_compact(outside_compact, raw, outside_value=True)
        expect_error(lambda: load_scalar_series([outside_compact], variable="salinity"), "outside source mask == 5")
        protected = root / "protected.png"
        protected.write_bytes(b"preexisting-map")
        original_save = map_module.save_efdc_scalar_map
        def fail_save(output, **kwargs):
            Path(output).write_bytes(b"partial")
            raise RuntimeError("injected map save failure")
        map_module.save_efdc_scalar_map = fail_save
        try:
            assert map_main([
                "map", "--input", str(raw), "--variable", "salinity", "--time-index", "0",
                "--output", str(protected), "--report", str(root / "protected.json"),
            ]) == 2
        finally:
            map_module.save_efdc_scalar_map = original_save
        assert protected.read_bytes() == b"preexisting-map"
        assert not protected.with_name("protected.tmp.png").exists()
        print(json.dumps({"status": "pass", "checks": 34, "wet_cells": 15}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
