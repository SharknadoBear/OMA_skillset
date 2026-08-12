#!/usr/bin/env python3
"""Offline synthetic tests for the POM movie postprocessing skill."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from netCDF4 import Dataset
from PIL import Image, ImageStat

try:
    from .pom_map_tools import quantile_limits, save_pom_scalar_map
    from .pom_movie_postprocessing import create_gif, inspect_inputs
    from .pom_output import load_scalar_series, resolve_layer_index, sigma_weights
except ImportError:
    from pom_map_tools import quantile_limits, save_pom_scalar_map
    from pom_movie_postprocessing import create_gif, inspect_inputs
    from pom_output import load_scalar_series, resolve_layer_index, sigma_weights


def _expect_error(kind, callback, contains: str | None = None) -> None:
    try:
        callback()
    except kind as exc:
        if contains is not None and contains.lower() not in str(exc).lower():
            raise AssertionError(f"Expected {kind.__name__} containing {contains!r}; received {exc!r}") from exc
    else:
        raise AssertionError(f"Expected {kind.__name__}.")


def _make_fixture(
    path: Path,
    hours: list[float],
    *,
    cycle_bias: float = 0.0,
    lon_shift: float = 0.0,
    include_v: bool = True,
    vector_metadata: bool = True,
) -> None:
    ny, nx, nz = 4, 5, 4
    y, x = np.mgrid[0:ny, 0:nx]
    lon = -74.1 + x * 0.055 + y * 0.003 + lon_shift
    lat = 40.45 + y * 0.045 + x * 0.002
    mask = np.ones((ny, nx), dtype=np.int8)
    mask[0, 0] = 0
    depth = 4.0 + y + 0.5 * x
    sigma = np.asarray([1.0, 0.6, 0.2, 0.0], dtype=np.float32)  # Bottom-to-surface on purpose.

    with Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("time", len(hours))
        ds.createDimension("sigma", nz)
        ds.createDimension("ny", ny)
        ds.createDimension("nx", nx)
        ds.source_system = "NYOFS"
        ds.source_model = "POM"
        ds.source_grid = "coarse"
        ds.grid_type = "curvilinear"
        if vector_metadata:
            ds.vector_components = "earth_relative"

        time = ds.createVariable("time", "f4", ("time",))
        time.units = "days since 2008-01-01  0:00:00 00:00"
        time[:] = np.asarray(hours, dtype=np.float64) / 24.0
        ds.createVariable("lon", "f8", ("ny", "nx"))[:] = lon
        ds.createVariable("lat", "f8", ("ny", "nx"))[:] = lat
        ds.createVariable("mask", "i1", ("ny", "nx"))[:] = mask
        ds.createVariable("depth", "f4", ("ny", "nx"))[:] = depth
        sigma_var = ds.createVariable("sigma", "f4", ("sigma",))
        sigma_var.positive = "down"
        sigma_var[:] = sigma

        zeta = ds.createVariable("zeta", "f4", ("time", "ny", "nx"), fill_value=-99999.0)
        air_u = ds.createVariable("air_u", "f4", ("time", "ny", "nx"), fill_value=-99999.0)
        air_v = ds.createVariable("air_v", "f4", ("time", "ny", "nx"), fill_value=-99999.0)
        u = ds.createVariable("u", "f4", ("time", "sigma", "ny", "nx"), fill_value=-99999.0)
        if include_v:
            v = ds.createVariable("v", "f4", ("time", "sigma", "ny", "nx"), fill_value=-99999.0)
        u.standard_name = "eastward_sea_water_velocity" if vector_metadata else ""
        if include_v:
            v.standard_name = "northward_sea_water_velocity" if vector_metadata else ""
        air_u.standard_name = "eastward_wind"
        air_v.standard_name = "northward_wind"

        shape_2d = x * 0.02 + y * 0.03
        for local, hour in enumerate(hours):
            nominal_hour = int(round(hour))
            zeta[local] = cycle_bias + nominal_hour + shape_2d
            air_u[local] = 3.0 + nominal_hour + shape_2d
            air_v[local] = 4.0
            for k, base in enumerate((4.0, 3.0, 2.0, 1.0)):
                u[local, k] = cycle_bias + base + nominal_hour * 0.1 + shape_2d
                if include_v:
                    v[local, k] = 0.0
        # Exercise finite-aware wet-layer renormalization and source fill handling.
        u[0, 1, 1, 1] = -99999.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_selftest() -> dict[str, object]:
    checks: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pom_movie_selftest_") as tmp_name:
        root = Path(tmp_name)
        first = root / "nyofs.t05z.fields.nowcast.nc"
        second = root / "nyofs.t11z.fields.nowcast.nc"
        drift = root / "nyofs.grid_drift.nc"
        missing_v = root / "nyofs.no_v.nc"
        ambiguous = root / "nyofs.ambiguous_vectors.nc"
        # ±14.0625-second live-style timing jitter; the second file overlaps hour 2.
        jitter = 14.0625 / 3600.0
        _make_fixture(first, [0.0, 1.0 - jitter, 2.0 + jitter])
        _make_fixture(second, [2.0 + jitter, 3.0 - jitter, 4.0 + jitter], cycle_bias=100.0)
        _make_fixture(drift, [5.0], lon_shift=0.001)
        _make_fixture(missing_v, [0.0], include_v=False)
        _make_fixture(ambiguous, [0.0], vector_metadata=False)

        sigma = np.asarray([1.0, 0.6, 0.2, 0.0])
        weights = sigma_weights(sigma)
        np.testing.assert_allclose(weights, [0.2, 0.4, 0.3, 0.1], atol=1.0e-12)
        assert resolve_layer_index(sigma, "surface") == 3
        assert resolve_layer_index(sigma, "near_surface") == 2
        assert resolve_layer_index(sigma, "bottom") == 0
        checks.append("reversed_sigma_and_trapezoid_weights")

        # Supply files out of order. Sorting and duplicate preference must still be deterministic.
        surface = load_scalar_series(
            [second, first],
            variable="current_speed",
            layer="surface",
            start="2008-01-01T00:00:00Z",
            end_exclusive="2008-01-01T05:00:00Z",
        )
        assert len(surface.times) == 5
        assert surface.duplicate_times_removed == 1
        assert surface.resolution["surface_sigma_index"] == 3
        assert surface.resolution["bottom_sigma_index"] == 0
        assert surface.resolution["normalized_cadence_seconds"] == 3600
        assert float(np.max(np.abs(surface.time_offsets_seconds))) <= 15.0
        expected_times = np.arange(5, dtype="timedelta64[h]") + np.datetime64("2008-01-01T00")
        np.testing.assert_array_equal(surface.times.astype("datetime64[h]"), expected_times)
        # Duplicate hour 2 must come from the earlier t05 aggregate, not cycle_bias=100.
        assert Path(surface.record_sources[2]).name == first.name
        np.testing.assert_allclose(surface.values[2, 2, 2], 1.0 + 0.2 + 0.04 + 0.06, atol=1.0e-5)
        checks.append("jitter_snap_sort_dedup_and_preceding_cycle_preference")

        bottom = load_scalar_series([first], variable="current_speed", layer="bottom")
        explicit = load_scalar_series([first], variable="current_speed", layer="index:3")
        np.testing.assert_allclose(bottom.values[0, 2, 2], 4.0 + 0.04 + 0.06, atol=1.0e-5)
        np.testing.assert_allclose(explicit.values, surface.values[:3], atol=1.0e-5, equal_nan=True)
        checks.append("surface_bottom_and_explicit_sigma_views")

        depth_average = load_scalar_series([first], variable="current_speed", layer="depth_average")
        # Unmodified wet cell: 0.2*4 + 0.4*3 + 0.3*2 + 0.1*1 = 2.7, plus spatial field.
        np.testing.assert_allclose(depth_average.values[0, 2, 2], 2.7 + 0.04 + 0.06, atol=1.0e-5)
        # Missing k=1 at (1,1): renormalize [0.2,0.3,0.1] over their 0.6 sum.
        expected_missing = (0.2 * 4.0 + 0.3 * 2.0 + 0.1 * 1.0) / 0.6 + 0.02 + 0.03
        np.testing.assert_allclose(depth_average.values[0, 1, 1], expected_missing, atol=1.0e-5)
        assert np.isnan(depth_average.values[:, 0, 0]).all()
        checks.append("finite_aware_depth_average_and_land_mask")

        wind = load_scalar_series([first], variable="wind_speed", layer="surface")
        np.testing.assert_allclose(wind.values[0, 2, 2], np.hypot(3.0 + 0.04 + 0.06, 4.0), atol=1.0e-5)
        checks.append("derived_wind_speed")

        _expect_error(
            ValueError,
            lambda: load_scalar_series([first, drift], variable="zeta", layer="surface"),
            "drift",
        )
        _expect_error(
            KeyError,
            lambda: load_scalar_series([missing_v], variable="current_speed", layer="surface"),
            "paired",
        )
        _expect_error(
            ValueError,
            lambda: load_scalar_series([ambiguous], variable="current_speed", layer="surface"),
            "earth-relative",
        )
        checks.append("geometry_unpaired_and_grid_relative_rejections")

        inspection = inspect_inputs([first, second])
        assert inspection["pom"]["status"] == "pass"
        assert inspection["pom"]["geometry"]["surface_sigma_index"] == 3
        assert inspection["pom"]["combined_time"]["duplicate_record_count"] == 1
        checks.append("inspection_report")

        limits = quantile_limits(np.asarray([1.0, 1.0, np.nan]))
        assert limits[0] < 1.0 < limits[1]
        map_path = root / "representative.png"
        result = save_pom_scalar_map(
            map_path,
            lon=surface.grid.lon,
            lat=surface.grid.lat,
            mask=surface.grid.mask,
            values=surface.values[0],
            vmin=float(np.nanmin(surface.values)),
            vmax=float(np.nanmax(surface.values)),
            title="Synthetic POM",
            colorbar_label="current speed",
            dpi=72,
        )
        assert result.finite_wet_fraction == 1.0
        with Image.open(map_path) as image:
            extrema = ImageStat.Stat(image.convert("L")).extrema[0]
            assert extrema[0] < extrema[1]
        checks.append("nonblank_shared_map_render")

        gif_path = root / "current_speed_surface.gif"
        manifest_path = root / "current_speed_surface.json"
        manifest = create_gif(
            [second, first],
            variable="current_speed",
            layer="surface",
            start="2008-01-01T00:00:00Z",
            end_exclusive="2008-01-01T05:00:00Z",
            fps=4,
            output=gif_path,
            report=manifest_path,
            dpi=72,
        )
        assert manifest["schema_version"] == "pom_movie_manifest_v1"
        assert manifest["selection"]["frame_count"] == 5
        assert manifest["selection"]["duplicate_times_removed"] == 1
        assert manifest["selection"]["unique_monotonic"] is True
        assert manifest["rendering"]["temporary_frames_cleaned"] is True
        assert manifest["fixed_color_limits"]["method"] == "full_series_quantiles"
        assert manifest["output"]["sha256"] == _sha256(gif_path)
        assert manifest["coverage"]["minimum_finite_wet_fraction"] == 1.0
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["output"]["frame_count"] == 5
        with Image.open(gif_path) as gif:
            assert gif.n_frames == 5
            assert gif.info.get("loop") == 0
            gif.seek(0)
            first_frame = np.asarray(gif.convert("RGB"), dtype=np.int16)
            gif.seek(4)
            last_frame = np.asarray(gif.convert("RGB"), dtype=np.int16)
            assert np.mean(np.abs(first_frame - last_frame)) > 0.1
        checks.append("fixed_scale_five_frame_gif_manifest_and_cleanup")

        explicit_gif = root / "explicit.gif"
        explicit_manifest = create_gif(
            [first],
            variable="zeta",
            layer="surface",
            start="2008-01-01T00:00:00Z",
            end_exclusive="2008-01-01T02:00:00Z",
            vmin=-1.0,
            vmax=5.0,
            fps=2,
            output=explicit_gif,
            dpi=60,
        )
        assert explicit_manifest["fixed_color_limits"]["method"] == "explicit"
        assert explicit_manifest["fixed_color_limits"]["vmin"] == -1.0
        assert explicit_manifest["output"]["frame_count"] == 2
        _expect_error(
            ValueError,
            lambda: create_gif(
                [first],
                variable="zeta",
                output=root / "bad.gif",
                vmin=0.0,
            ),
            "both",
        )
        checks.append("explicit_limits_and_argument_validation")

    return {"status": "pass", "check_count": len(checks), "checks": checks}


if __name__ == "__main__":
    print(json.dumps(run_selftest(), indent=2))
