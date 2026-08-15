from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

try:
    import netCDF4 as nc4
    import numpy as np

    from waterlevel_core import (
        Boundary,
        SourceSeries,
        build_target_times,
        iso_utc,
        read_boundary_2dm,
        read_source,
        sha256_file,
        spatial_interpolate,
        temporal_interpolate,
        write_fvcom_forcing,
        write_json_atomic,
    )
    from waterlevel_validation import validate_forcing
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency {exc.name!r}. Install numpy, scipy, netCDF4, and matplotlib on this equipment."
    ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run synthetic tests for FVCOM boundary water-level forcing, optionally on a real 2DM mesh."
    )
    parser.add_argument("--work-dir", help="Retain test products in this directory; otherwise use a temporary directory")
    parser.add_argument("--mesh-2dm", help="Optional real geographic 2DM mesh for a 400-day forward test")
    parser.add_argument("--open-ns", nargs="+", type=int, default=[1], help="Open nodestring ids for --mesh-2dm")
    return parser


def _write_small_mesh(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "MESH2D",
                'MESHNAME "synthetic_two_arc"',
                "E3T 1 1 2 3 1",
                "E3T 2 2 4 3 1",
                "E3T 3 3 4 5 1",
                "E3T 4 4 6 5 1",
                "ND 1 -75.2 38.0 10",
                "ND 2 -75.0 38.0 10",
                "ND 3 -74.8 38.0 10",
                "ND 4 -75.0 38.2 10",
                "ND 5 -74.8 38.2 10",
                "ND 6 -75.0 38.4 10",
                "NS 1 2",
                "NS -3 1",
                "NS 5 -6 2",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )


def _synthetic_signal(hours: np.ndarray) -> np.ndarray:
    centered_hours = hours - np.mean(hours)
    centered_days = centered_hours / 24.0
    elapsed_days = (hours - hours[0]) / 24.0
    return (
        0.55 * np.cos(2 * np.pi * centered_hours / 12.4206)
        + 0.16 * np.cos(2 * np.pi * centered_hours / 12.0)
        + 0.11 * np.cos(2 * np.pi * centered_hours / 23.9345)
        + 0.08 * np.cos(2 * np.pi * centered_hours / 25.8193)
        + 0.13 * np.cos(2 * np.pi * centered_days / 7.0)
        + 0.07 * np.cos(2 * np.pi * centered_days / 30.0)
        + 0.08 * np.cos(2 * np.pi * centered_days / 200.0)
        + 0.006 * elapsed_days / 365.2425
    )


def _write_direct_source(path: Path, boundary: Boundary) -> None:
    hours = np.arange(0, 220 * 24 + 1, dtype=np.float64)
    times = hours.astype(np.int64) * 3_600_000 + int(np.datetime64("2020-01-01T00:00:00", "ms").astype(np.int64))
    permutation = np.arange(len(boundary.node_ids))[::-1]
    values = _synthetic_signal(hours)[:, None] + np.arange(len(boundary.node_ids))[None, :] * 0.01
    with nc4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.createDimension("time", len(times))
        dataset.createDimension("point", len(boundary.node_ids))
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 2020-01-01 00:00:00"
        time.calendar = "standard"
        time[:] = hours
        nodes = dataset.createVariable("obc_nodes", "i4", ("point",))
        nodes[:] = boundary.node_ids[permutation]
        elevation = dataset.createVariable("elevation", "f4", ("time", "point"))
        elevation.units = "cm"
        elevation[:] = (values[:, permutation] * 100.0).astype(np.float32)


def _write_station_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "station_id", "lon", "lat", "water_level"])
        writer.writeheader()
        for hour in range(4):
            stamp = str(np.datetime64("2020-01-01T00:00:00") + np.timedelta64(hour, "h")) + "Z"
            writer.writerow({"time": stamp, "station_id": "west", "lon": -75.2, "lat": 38.0, "water_level": hour})
            writer.writerow({"time": stamp, "station_id": "east", "lon": -74.8, "lat": 38.2, "water_level": hour + 2})


def _write_dateline_grid(path: Path) -> None:
    with nc4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.createDimension("time", 3)
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 3)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 2020-01-01 00:00:00"
        time[:] = [0, 1, 2]
        lon = dataset.createVariable("lon", "f8", ("lon",))
        lat = dataset.createVariable("lat", "f8", ("lat",))
        lon[:] = [179.0, 180.0, 181.0]
        lat[:] = [-1.0, 1.0]
        zeta = dataset.createVariable("zeta", "f4", ("time", "lat", "lon"), fill_value=-9999.0)
        zeta.units = "meters"
        block = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 10.0
        block[:, 0, 0] = -9999.0
        zeta[:] = block


def run_core_tests(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    mesh = root / "synthetic.2dm"
    _write_small_mesh(mesh)
    boundary = read_boundary_2dm(mesh, [1, 2])
    assert boundary.node_ids.tolist() == [1, 2, 3, 5, 6]
    assert [len(arc) for arc in boundary.arcs] == [3, 2]

    direct_path = root / "direct_source.nc"
    _write_direct_source(direct_path, boundary)
    direct = read_source(direct_path)
    assert iso_utc(direct.times_ms[[59 * 24]])[0] == "2020-02-29T00:00:00.000Z"
    mapped, direct_stats = spatial_interpolate(direct, boundary)
    assert direct_stats["method"] == "direct_node_id"
    assert mapped.shape == (220 * 24 + 1, 5)
    assert np.allclose(mapped[0] - mapped[0, 0], np.arange(5) * 0.01, atol=1e-6)

    single = SourceSeries(direct.times_ms, direct.values[:, :1], "single_series", "meters")
    try:
        spatial_interpolate(single, boundary)
    except ValueError as exc:
        assert "broadcast" in str(exc)
    else:
        raise AssertionError("Single-series broadcast was not protected")
    broadcast, _ = spatial_interpolate(single, boundary, broadcast_single=True)
    assert broadcast.shape == mapped.shape

    station_csv = root / "stations.csv"
    _write_station_csv(station_csv)
    stations = read_source(station_csv, units_override="cm")
    station_values, station_stats = spatial_interpolate(stations, boundary, max_nearest_km=100.0)
    assert station_stats["method"] == "station_idw"
    assert np.all(np.isfinite(station_values))
    assert 0.049 < np.max(np.abs(station_values)) < 0.051

    dateline_path = root / "dateline.nc"
    _write_dateline_grid(dateline_path)
    dateline = read_source(dateline_path)
    dateline_boundary = Boundary(
        node_ids=np.asarray([1, 2]),
        lon=np.asarray([179.5, -178.5]),
        lat=np.asarray([0.0, 0.0]),
        arcs=(np.asarray([0, 1]),),
        source="synthetic dateline",
    )
    dateline_values, dateline_stats = spatial_interpolate(dateline, dateline_boundary, max_nearest_km=200.0)
    assert np.all(np.isfinite(dateline_values))
    assert dateline_stats["fallback_target_indices"] == [0, 1]
    try:
        spatial_interpolate(dateline, dateline_boundary, max_nearest_km=50.0)
    except ValueError as exc:
        assert "nearest-source limit" in str(exc)
    else:
        raise AssertionError("A target outside the reviewed nearest-wet limit was accepted")

    resample_end = direct.times_ms[48]
    resample_times = build_target_times(
        direct.times_ms,
        start=iso_utc(direct.times_ms[[0]])[0],
        end=iso_utc(np.asarray([resample_end]))[0],
        dt_seconds=7200.0,
    )
    resampled, resample_stats = temporal_interpolate(direct.times_ms, mapped, resample_times)
    assert resampled.shape == (25, len(boundary.node_ids))
    assert resample_stats["target_cadence_seconds"] == 7200.0
    try:
        temporal_interpolate(direct.times_ms, mapped, direct.times_ms - 1)
    except ValueError as exc:
        assert "extrapolation" in str(exc)
    else:
        raise AssertionError("Unapproved temporal extrapolation was accepted")

    irregular = direct.times_ms.copy()
    irregular[10:] += 10 * 3_600_000
    try:
        build_target_times(irregular)
    except ValueError as exc:
        assert "Irregular" in str(exc)
    else:
        raise AssertionError("Irregular time axis was accepted without a target grid")
    try:
        temporal_interpolate(irregular, mapped, irregular, max_gap_factor=3.0)
    except ValueError as exc:
        assert "gap" in str(exc)
    else:
        raise AssertionError("Large temporal gap was not rejected")

    bad_output = root / "must_not_exist.nc"
    try:
        write_fvcom_forcing(
            bad_output,
            boundary,
            direct.times_ms,
            mapped[:, :-1],
            case_name="bad",
            vertical_datum="unspecified",
            source_name=direct_path.name,
            source_sha256="0" * 64,
            spatial_method="test",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid forcing shape was accepted")
    assert not bad_output.exists()

    output = root / "synthetic_forcing.nc"
    write_fvcom_forcing(
        output,
        boundary,
        direct.times_ms,
        mapped,
        case_name="synthetic",
        vertical_datum="MSL synthetic",
        source_name=direct_path.name,
        source_sha256=sha256_file(direct_path),
        spatial_method=direct_stats["method"],
    )
    qa = root / "synthetic_qa"
    validation = validate_forcing(output, boundary, qa)
    assert validation["status"] == "pass"
    assert validation["cadence_seconds"] == 3600.0
    assert validation["prominent_spectral_peaks"]["tidal"] is not None
    for artifact in validation["qa_artifacts"].values():
        assert Path(artifact).is_file()
    return {
        "status": "pass",
        "boundary_nodes": len(boundary.node_ids),
        "output": str(output.resolve()),
        "qa": validation,
    }


def _write_real_grid_source(path: Path, boundary: Boundary) -> None:
    hours = np.arange(0, 400 * 24 + 1, dtype=np.float64)
    lon = np.linspace(boundary.lon.min() - 0.25, boundary.lon.max() + 0.25, 6)
    lat = np.linspace(boundary.lat.min() - 0.25, boundary.lat.max() + 0.25, 6)
    base = _synthetic_signal(hours).astype(np.float32)
    spatial = (
        0.015 * (lat[:, None] - np.mean(lat))
        + 0.008 * np.cos(np.radians((lon[None, :] - np.mean(lon)) * 20.0))
    ).astype(np.float32)
    with nc4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.createDimension("time", len(hours))
        dataset.createDimension("lat", len(lat))
        dataset.createDimension("lon", len(lon))
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 2020-01-01 00:00:00"
        time.calendar = "standard"
        time[:] = hours
        lon_var = dataset.createVariable("lon", "f8", ("lon",))
        lat_var = dataset.createVariable("lat", "f8", ("lat",))
        lon_var[:] = lon
        lat_var[:] = lat
        elevation = dataset.createVariable("water_level", "f4", ("time", "lat", "lon"), fill_value=-9999.0)
        elevation.units = "meters"
        elevation[:] = base[:, None, None] + spatial[None, :, :]


def run_real_mesh_test(root: Path, mesh_path: Path, open_ns: list[int]) -> dict[str, object]:
    boundary = read_boundary_2dm(mesh_path, open_ns)
    source_path = root / "san_francisco_combined_waterlevel_source.nc"
    _write_real_grid_source(source_path, boundary)
    source = read_source(source_path)
    spatial, spatial_stats = spatial_interpolate(source, boundary)
    target = build_target_times(source.times_ms)
    elevation, temporal_stats = temporal_interpolate(source.times_ms, spatial, target)
    output = root / "san_francisco_boundary_waterlevel_forcing.nc"
    write_fvcom_forcing(
        output,
        boundary,
        target,
        elevation,
        case_name="san_francisco_boundary_waterlevel_skill_test",
        vertical_datum="synthetic MSL",
        source_name=source_path.name,
        source_sha256=sha256_file(source_path),
        spatial_method=spatial_stats["method"],
    )
    validation = validate_forcing(output, boundary, root / "san_francisco_qa")
    peaks = validation["prominent_spectral_peaks"]
    assert 11.0 < peaks["tidal"]["period_hours"] < 13.0
    assert 150.0 < peaks["subtidal"]["period_hours"] < 185.0
    assert peaks["vlf_slr_scale"]["period_hours"] > 90.0 * 24.0
    assert abs(validation["linear_trend_mm_per_year"]["boundary_median"] - 6.0) < 1.5
    report = {
        "status": "pass",
        "mesh": str(mesh_path.resolve()),
        "open_nodestrings": open_ns,
        "boundary_node_count": len(boundary.node_ids),
        "source": str(source_path.resolve()),
        "forcing": str(output.resolve()),
        "spatial": spatial_stats,
        "temporal": temporal_stats,
        "validation": validation,
    }
    write_json_atomic(root / "san_francisco_test_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.work_dir:
        root = Path(args.work_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="fvcom-boundary-waterlevel-")
        root = Path(temp_context.name)
    try:
        core_report = run_core_tests(root / "core")
        if args.mesh_2dm:
            real_report = run_real_mesh_test(root, Path(args.mesh_2dm), args.open_ns)
            print(f"[PASS] Real-mesh test: {real_report['forcing']}")
        print(f"[PASS] Core self-test: {core_report['output']}")
        if args.work_dir:
            print(f"[PASS] Retained test products: {root}")
        return 0
    finally:
        if temp_context is not None:
            temp_context.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
