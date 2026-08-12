#!/usr/bin/env python3
"""Offline synthetic tests for POM map postprocessing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from netCDF4 import Dataset
from PIL import Image, ImageStat

try:
    from .pom_map_postprocessing import main
    from .pom_map_tools import quantile_limits
    from .pom_output import (
        inspect_inputs,
        load_scalar_series,
        normalize_times,
        resolve_layer_index,
        sigma_weights,
    )
except ImportError:
    from pom_map_postprocessing import main
    from pom_map_tools import quantile_limits
    from pom_output import inspect_inputs, load_scalar_series, normalize_times, resolve_layer_index, sigma_weights


def _expect_error(kind, callback, contains: str | None = None) -> None:
    try:
        callback()
    except kind as exc:
        if contains and contains.lower() not in str(exc).lower():
            raise AssertionError(f"Expected {contains!r} in {exc!r}.") from exc
    else:
        raise AssertionError(f"Expected {kind.__name__}.")


def _make_fixture(
    path: Path,
    hours: list[float],
    *,
    cycle_bias: float = 0.0,
    lon_shift: float = 0.0,
    include_v: bool = True,
    vector_metadata: str = "long_name",
    include_ready_speed: bool = False,
) -> None:
    ny, nx, nz = 5, 6, 4
    y, x = np.mgrid[0:ny, 0:nx]
    lon = -74.2 + x * 0.055 + y * 0.004 + lon_shift
    lat = 40.4 + y * 0.044 + x * 0.002
    mask = np.ones((ny, nx), dtype=np.int8)
    mask[0, 0] = 0
    mask[1, 0] = 0
    sigma = np.asarray([1.0, 0.6, 0.2, 0.0], dtype=np.float32)  # Reversed on purpose.

    with Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("time", len(hours))
        ds.createDimension("sigma", nz)
        ds.createDimension("ny", ny)
        ds.createDimension("nx", nx)
        ds.source_system = "NYOFS"
        ds.source_model = "POM"
        ds.source_grid = "coarse"
        ds.grid_type = "curvilinear"
        if vector_metadata == "global":
            ds.vector_components = "earth_relative"

        time = ds.createVariable("time", "f4", ("time",))
        time.units = "days since 2008-01-01 00:00:00 +00:00"
        time[:] = np.asarray(hours, dtype=np.float64) / 24.0
        ds.createVariable("lon", "f8", ("ny", "nx"))[:] = lon
        ds.createVariable("lat", "f8", ("ny", "nx"))[:] = lat
        ds.createVariable("mask", "i1", ("ny", "nx"))[:] = mask
        ds.createVariable("depth", "f4", ("ny", "nx"))[:] = 4.0 + y + 0.5 * x
        sigma_var = ds.createVariable("sigma", "f4", ("sigma",))
        sigma_var.positive = "down"
        sigma_var[:] = sigma

        zeta = ds.createVariable("zeta", "f4", ("time", "ny", "nx"), fill_value=-99999.0)
        air_u = ds.createVariable("air_u", "f4", ("time", "ny", "nx"), fill_value=-99999.0)
        air_v = ds.createVariable("air_v", "f4", ("time", "ny", "nx"), fill_value=-99999.0)
        u = ds.createVariable("u", "f4", ("time", "sigma", "ny", "nx"), fill_value=-99999.0)
        v = ds.createVariable("v", "f4", ("time", "sigma", "ny", "nx"), fill_value=-99999.0) if include_v else None
        zeta.units = "m"
        zeta.long_name = "water surface elevation"
        u.units = "m s-1"
        if v is not None:
            v.units = "m s-1"
        air_u.units = air_v.units = "m s-1"
        if vector_metadata == "cf":
            u.standard_name = "eastward_sea_water_velocity"
            if v is not None:
                v.standard_name = "northward_sea_water_velocity"
            air_u.standard_name = "eastward_wind"
            air_v.standard_name = "northward_wind"
        elif vector_metadata == "long_name":
            u.long_name = "Eastward Water Velocity"
            if v is not None:
                v.long_name = "Northward Water Velocity"
            air_u.long_name = "Eastward Wind Velocity"
            air_v.long_name = "Northward Wind Velocity"

        ready = None
        if include_ready_speed:
            ready = ds.createVariable("current_speed_surface", "f4", ("time", "ny", "nx"), fill_value=-99999.0)
            ready.units = "m s-1"
            ready.long_name = "surface current speed"

        spatial = x * 0.02 + y * 0.03
        for local, hour in enumerate(hours):
            nominal = int(round(hour))
            zeta[local] = cycle_bias + nominal + spatial
            air_u[local] = 3.0 + nominal + spatial
            air_v[local] = 4.0
            for k, base in enumerate((4.0, 3.0, 2.0, 1.0)):
                u[local, k] = cycle_bias + base + nominal * 0.1 + spatial
                if v is not None:
                    v[local, k] = 0.0
            if ready is not None:
                ready[local] = 50.0 + nominal + spatial
        u[0, 1, 2, 2] = -99999.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_nonblank_png(path: Path) -> None:
    assert path.stat().st_size > 1000
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width > 100 and image.height > 100
        low, high = ImageStat.Stat(image.convert("L")).extrema[0]
        assert low < high


def run_selftest() -> dict[str, object]:
    checks: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pom_map_selftest_") as temporary_name:
        root = Path(temporary_name)
        first = root / "nyofs.t05z.fields.nowcast.nc"
        second = root / "nyofs.t11z.fields.nowcast.nc"
        ready = root / "nyofs.compact.nc"
        drift = root / "nyofs.drift.nc"
        missing_v = root / "nyofs.no_v.nc"
        ambiguous = root / "nyofs.grid_relative_unknown.nc"
        jitter = 14.0625 / 3600.0
        _make_fixture(first, [0.0, 1.0 - jitter, 2.0 + jitter])
        _make_fixture(second, [2.0 + jitter, 3.0 - jitter, 4.0 + jitter], cycle_bias=100.0)
        _make_fixture(ready, [0.0, 1.0], include_ready_speed=True)
        _make_fixture(drift, [5.0], lon_shift=0.001)
        _make_fixture(missing_v, [0.0], include_v=False)
        _make_fixture(ambiguous, [0.0], vector_metadata="none")

        sigma = np.asarray([1.0, 0.6, 0.2, 0.0])
        np.testing.assert_allclose(sigma_weights(sigma), [0.2, 0.4, 0.3, 0.1])
        assert resolve_layer_index(sigma, "surface") == 3
        assert resolve_layer_index(sigma, "near_surface") == 2
        assert resolve_layer_index(sigma, "bottom") == 0
        checks.append("reversed_sigma_dynamic_views_and_weights")

        boundary_raw = np.asarray(
            [
                np.datetime64("2008-01-01T00:01:00", "ns"),
                np.datetime64("2008-01-01T01:01:01", "ns"),
            ]
        )
        normalized, offsets, cadence = normalize_times(boundary_raw, tolerance_seconds=60.0)
        assert cadence == 3600
        assert normalized[0] == np.datetime64("2008-01-01T00:00:00")
        assert normalized[1] == boundary_raw[1]
        np.testing.assert_allclose(offsets, [-60.0, 0.0])
        checks.append("inclusive_sixty_second_snap_boundary")

        surface = load_scalar_series(
            [second, first],
            variable="current_speed",
            layer="surface",
            start="2008-01-01T00:00:00Z",
            end_exclusive="2008-01-01T05:00:00Z",
        )
        assert len(surface.times) == 5
        assert surface.duplicate_times_removed == 1
        assert Path(surface.record_sources[2]).name == first.name
        assert surface.resolution["resolved_mode"] == "magnitude"
        assert surface.resolution["normalized_cadence_seconds"] == 3600
        assert np.max(np.abs(surface.time_offsets_seconds)) < 15.0
        checks.append("raw_concat_jitter_snap_dedup_and_earthward_magnitude")

        averaged = load_scalar_series([first], variable="current_speed", layer="depth_average")
        np.testing.assert_allclose(averaged.values[0, 3, 3], 2.7 + 0.06 + 0.09, atol=1.0e-5)
        expected_missing = (0.2 * 4.0 + 0.3 * 2.0 + 0.1 * 1.0) / 0.6 + 0.04 + 0.06
        np.testing.assert_allclose(averaged.values[0, 2, 2], expected_missing, atol=1.0e-5)
        assert np.isnan(averaged.values[:, 0, 0]).all()
        checks.append("finite_aware_depth_average_and_land_mask")

        direct = load_scalar_series([ready], variable="current_speed", layer="surface")
        assert direct.resolution["resolved_mode"] == "direct"
        assert direct.resolution["source_variables"] == ["current_speed_surface"]
        np.testing.assert_allclose(direct.values[0, 3, 3], 50.0 + 0.06 + 0.09, atol=1.0e-5)
        checks.append("compact_ready_magnitude_resolution")

        _expect_error(ValueError, lambda: load_scalar_series([first, drift], variable="zeta"), "drift")
        _expect_error(KeyError, lambda: load_scalar_series([missing_v], variable="current_speed"), "paired")
        _expect_error(ValueError, lambda: load_scalar_series([ambiguous], variable="current_speed"), "earth-relative")
        checks.append("geometry_unpaired_and_unproven_vector_rejections")

        inspection = inspect_inputs([first])
        assert inspection["status"] == "pass"
        assert inspection["geometry"]["shape"] == [5, 6]
        assert inspection["geometry"]["surface_sigma_index"] == 3
        inspect_path = root / "inspection.json"
        assert main(["inspect", "--input", str(first), "--output", str(inspect_path)]) == 0
        inspection_cli = json.loads(inspect_path.read_text(encoding="utf-8"))
        assert inspection_cli["schema_version"] == "pom_inspection_v1"
        assert inspection_cli["files"][0]["sha256"] == _sha256(first)
        checks.append("library_and_cli_inspection")

        current_png = root / "current_surface.png"
        current_json = root / "current_surface.json"
        assert main(
            [
                "map", "--input", str(first), "--variable", "current_speed",
                "--time", "2008-01-01T01:00:00Z", "--layer", "surface",
                "--limits-scope", "series", "--quiver", "current", "--quiver-stride", "2",
                "--output", str(current_png), "--report", str(current_json), "--dpi", "72",
            ]
        ) == 0
        _assert_nonblank_png(current_png)
        manifest = json.loads(current_json.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "pom_map_manifest_v1"
        assert manifest["output"]["sha256"] == _sha256(current_png)
        assert manifest["selection"]["normalized_time_utc"] == "2008-01-01T01:00:00Z"
        assert manifest["selection"]["resolution"]["resolved_mode"] == "magnitude"
        assert manifest["quiver"]["mode"] == "current"
        assert manifest["rendering"]["quiver_count"] == 8
        assert manifest["rendering"]["finite_wet_coverage"] == 1.0
        assert not list(root.glob("*.tmp.png"))
        checks.append("static_current_map_quiver_manifest_and_atomic_cleanup")

        zeta_png = root / "zeta.png"
        zeta_json = root / "zeta.json"
        assert main(
            [
                "map", "--input", str(first), "--variable", "zeta", "--time-index", "0",
                "--layer", "surface", "--style", "contourf", "--vmin", "-1", "--vmax", "2",
                "--output", str(zeta_png), "--report", str(zeta_json), "--dpi", "72",
            ]
        ) == 0
        _assert_nonblank_png(zeta_png)
        zeta_manifest = json.loads(zeta_json.read_text(encoding="utf-8"))
        assert zeta_manifest["rendering"]["vmin"] == -1.0
        assert zeta_manifest["rendering"]["vmax"] == 2.0
        assert zeta_manifest["rendering"]["style"] == "contourf"
        checks.append("contourf_explicit_limits_and_time_index")

        bad_png = root / "bad.png"
        bad_json = root / "bad.json"
        assert main(
            [
                "map", "--input", str(ambiguous), "--variable", "zeta", "--time-index", "0",
                "--quiver", "current", "--output", str(bad_png), "--report", str(bad_json),
            ]
        ) == 2
        assert not bad_png.exists() and not bad_json.exists()
        limits = quantile_limits(np.asarray([1.0, 1.0, np.nan]))
        assert limits[0] < 1.0 < limits[1]
        checks.append("quiver_vector_provenance_gate_and_constant_limits")

    return {"status": "pass", "check_count": len(checks), "checks": checks}


if __name__ == "__main__":
    print(json.dumps(run_selftest(), indent=2))
