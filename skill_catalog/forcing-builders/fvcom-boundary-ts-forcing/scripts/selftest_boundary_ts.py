from __future__ import annotations

import argparse
import tempfile
from contextlib import nullcontext
from pathlib import Path

try:
    import matplotlib  # noqa: F401
    import netCDF4 as nc4
    import numpy as np
    import scipy  # noqa: F401

    from ts_core import (
        build_target_times,
        iso_utc,
        read_boundary_2dm,
        read_boundary_dat,
        read_sigma_ready_source,
        repair_missing,
        sha256_file,
        temporal_resample,
        validate_sigma,
        write_fvcom_ts_forcing,
        write_json_atomic,
    )
    from ts_validation import validate_forcing
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency {exc.name!r}. Install numpy, scipy, netCDF4, and matplotlib on this equipment before continuing."
    ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic tests for the FVCOM boundary temperature/salinity forcing skill.")
    parser.add_argument("--work-dir", help="Retain products here; otherwise use a temporary directory")
    return parser


def _write_mesh(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "MESH2D",
                'MESHNAME "synthetic_two_arc_ts"',
                "E3T 1 1 2 3 1",
                "E3T 2 2 4 3 1",
                "E3T 3 3 4 5 1",
                "E3T 4 4 6 5 1",
                "ND 1 -123.2 37.2 20",
                "ND 2 -123.0 37.3 35",
                "ND 3 -122.8 37.4 50",
                "ND 4 -122.7 37.6 70",
                "ND 5 -122.6 37.8 90",
                "ND 6 -122.5 38.0 110",
                "NS 1 2",
                "NS -3 1",
                "NS 5 -6 2",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )


def _write_dat_files(grd: Path, obc: Path) -> None:
    grd.write_text(
        "\n".join(
            (
                "Node Number = 4",
                "Cell Number = 2",
                "1 1 2 3 1",
                "2 2 4 3 1",
                "1 -123.0 37.0 20",
                "2 -122.9 37.0 30",
                "3 -123.0 37.1 40",
                "4 -122.9 37.1 50",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    obc.write_text("OBC Node Number = 3\n1 1 1\n2 2 1\n3 4 1\n", encoding="utf-8", newline="\n")


def _write_source(path: Path, boundary, *, surface_first: bool = False, two_dimensional_sigma: bool = False) -> tuple[np.ndarray, np.ndarray]:
    hours = np.arange(73, dtype=np.float64)
    source_nodes = boundary.node_ids[::-1]
    siglev = np.linspace(0.0, -1.0, 5) if surface_first else np.linspace(-1.0, 0.0, 5)
    siglay = 0.5 * (siglev[:-1] + siglev[1:])
    base_temp = (
        12.0
        + 2.0 * np.sin(2.0 * np.pi * hours[:, None, None] / 24.0)
        + 4.0 * (siglay[None, :, None] + 1.0)
        + 0.15 * np.arange(len(boundary.node_ids))[None, None, :]
    )
    base_salt = (
        31.0
        + 0.5 * np.cos(2.0 * np.pi * hours[:, None, None] / 36.0)
        + 1.2 * (-siglay[None, :, None])
        + 0.08 * np.arange(len(boundary.node_ids))[None, None, :]
    )
    temperature = np.broadcast_to(base_temp, (len(hours), len(siglay), len(boundary.node_ids))).copy()
    salinity = np.broadcast_to(base_salt, temperature.shape).copy()
    temperature[10:14, 0, 0] = np.nan
    temperature[:, 2, 0] = np.nan
    temperature[:, :, 1] = np.nan
    salinity[20:23, 1, 2] = np.nan
    salinity[:, 3, 2] = np.nan
    salinity[:, :, 4] = np.nan
    permutation = np.arange(len(source_nodes))[::-1]
    with nc4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.createDimension("time", len(hours))
        dataset.createDimension("sigma", len(siglay))
        dataset.createDimension("interface", len(siglev))
        dataset.createDimension("point", len(source_nodes))
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 2020-02-28 00:00:00"
        time.calendar = "standard"
        time[:] = hours
        nodes = dataset.createVariable("node_id", "i4", ("point",))
        nodes[:] = source_nodes
        temp = dataset.createVariable("temperature", "f8", ("point", "time", "sigma"), fill_value=-9999.0)
        temp.units = "Kelvin"
        temp[:] = np.moveaxis(temperature[:, :, permutation] + 273.15, [0, 1, 2], [1, 2, 0])
        salt = dataset.createVariable("salinity", "f8", ("time", "sigma", "point"), fill_value=-9999.0)
        salt.units = "PSU"
        salt[:] = salinity[:, :, permutation]
        if two_dimensional_sigma:
            layer = dataset.createVariable("siglay", "f8", ("point", "sigma"))
            interface = dataset.createVariable("siglev", "f8", ("interface", "point"))
            layer[:] = np.repeat(siglay[None, :], len(source_nodes), axis=0)
            interface[:] = np.repeat(siglev[:, None], len(source_nodes), axis=1)
        else:
            layer = dataset.createVariable("siglay", "f8", ("sigma",))
            interface = dataset.createVariable("siglev", "f8", ("interface",))
            layer[:] = siglay
            interface[:] = siglev
    return temperature, salinity


def run(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    mesh = root / "synthetic.2dm"
    _write_mesh(mesh)
    boundary = read_boundary_2dm(mesh, [1, 2])
    assert boundary.node_ids.tolist() == [1, 2, 3, 5, 6]
    assert [len(arc) for arc in boundary.arcs] == [3, 2]
    assert boundary.depth_m.tolist() == [20.0, 35.0, 50.0, 90.0, 110.0]
    grd, obc = root / "synthetic_grd.dat", root / "synthetic_obc.dat"
    _write_dat_files(grd, obc)
    dat_boundary = read_boundary_dat(grd, obc)
    assert dat_boundary.node_ids.tolist() == [1, 2, 4]
    assert len(dat_boundary.arcs) == 1

    source_path = root / "sigma_ready_source.nc"
    original_temp, original_salt = _write_source(source_path, boundary)
    source = read_sigma_ready_source(source_path, boundary)
    assert source.sigma_orientation == "bottom_to_surface"
    assert iso_utc(source.times_ms[[24]])[0] == "2020-02-29T00:00:00.000Z"
    assert np.allclose(source.temperature_c[np.isfinite(original_temp)], original_temp[np.isfinite(original_temp)])
    temperature, temperature_codes, temperature_report = repair_missing(source.temperature_c, source.times_ms, source.siglay, boundary, "temperature")
    salinity, salinity_codes, salinity_report = repair_missing(source.salinity, source.times_ms, source.siglay, boundary, "salinity")
    assert np.all(np.isfinite(temperature)) and np.all(np.isfinite(salinity))
    assert all(value > 0 for value in temperature_report["method_counts"].values())
    assert all(value > 0 for value in salinity_report["method_counts"].values())
    assert np.array_equal(temperature[np.isfinite(source.temperature_c)], source.temperature_c[np.isfinite(source.temperature_c)])
    assert np.array_equal(salinity[np.isfinite(source.salinity)], source.salinity[np.isfinite(source.salinity)])

    surface_source = root / "surface_first_2d_sigma.nc"
    _write_source(surface_source, boundary, surface_first=True, two_dimensional_sigma=True)
    surface = read_sigma_ready_source(surface_source, boundary)
    assert surface.sigma_orientation == "surface_to_bottom"
    assert validate_sigma(surface.siglay, surface.siglev) == "surface_to_bottom"

    target_times = build_target_times(
        source.times_ms,
        start=iso_utc(source.times_ms[[0]])[0],
        end=iso_utc(source.times_ms[[-1]])[0],
        dt_seconds=7200.0,
    )
    temperature_out, time_report = temporal_resample(source.times_ms, temperature, target_times)
    salinity_out, _ = temporal_resample(source.times_ms, salinity, target_times)
    assert temperature_out.shape[0] == 37 and time_report["target_cadence_seconds"] == 7200.0
    try:
        temporal_resample(source.times_ms, temperature, source.times_ms - 1)
    except ValueError as exc:
        assert "extrapolation" in str(exc)
    else:
        raise AssertionError("Temporal extrapolation was accepted")
    irregular = source.times_ms.copy()
    irregular[20:] += 10 * 3_600_000
    try:
        build_target_times(irregular)
    except ValueError as exc:
        assert "Irregular" in str(exc)
    else:
        raise AssertionError("Irregular time was accepted without explicit target time")
    try:
        temporal_resample(irregular, temperature, np.asarray([irregular[0], irregular[-1]]), max_gap_factor=3.0)
    except ValueError as exc:
        assert "gap" in str(exc)
    else:
        raise AssertionError("Excessive source gap was accepted")

    impossible = np.ones_like(source.temperature_c)
    impossible[:, :, boundary.arcs[1]] = np.nan
    try:
        repair_missing(impossible, source.times_ms, source.siglay, boundary, "impossible_temperature")
    except ValueError as exc:
        assert "cannot be repaired" in str(exc)
    else:
        raise AssertionError("Repair crossed a disconnected boundary arc")

    single_time = np.ones_like(source.temperature_c)
    single_time[:, 0, 0] = np.nan
    single_time[5, 0, 0] = 7.25
    single_time_repaired, single_time_codes, _ = repair_missing(
        single_time, source.times_ms, source.siglay, boundary, "single_time_value"
    )
    assert np.all(single_time_repaired[:, 0, 0] == 7.25)
    assert np.all(single_time_codes[np.arange(len(source.times_ms)) != 5, 0, 0] == 1)

    single_vertical = np.ones_like(source.temperature_c)
    single_vertical[:, :, 0] = np.nan
    single_vertical[:, 1, 0] = 6.5
    single_vertical_repaired, single_vertical_codes, _ = repair_missing(
        single_vertical, source.times_ms, source.siglay, boundary, "single_vertical_value"
    )
    assert np.all(single_vertical_repaired[:, :, 0] == 6.5)
    assert np.all(single_vertical_codes[:, [0, 2, 3], 0] == 2)

    single_spatial = np.ones_like(source.temperature_c)
    single_spatial[:, :, boundary.arcs[0][1:]] = np.nan
    single_spatial_repaired, single_spatial_codes, _ = repair_missing(
        single_spatial, source.times_ms, source.siglay, boundary, "single_spatial_value"
    )
    assert np.all(single_spatial_repaired[:, :, boundary.arcs[0]] == 1.0)
    assert np.all(single_spatial_codes[:, :, boundary.arcs[0][1:]] == 3)

    invalid_siglev = source.siglev.copy()
    invalid_siglev[2, 0] = invalid_siglev[1, 0]
    try:
        validate_sigma(source.siglay, invalid_siglev)
    except ValueError as exc:
        assert "monotonic" in str(exc) or "between" in str(exc)
    else:
        raise AssertionError("Invalid sigma coordinates were accepted")

    output = root / "synthetic_tsobc.nc"
    write_fvcom_ts_forcing(
        output,
        boundary,
        target_times,
        source.siglay,
        source.siglev,
        temperature_out,
        salinity_out,
        case_name="synthetic_ts",
        source_name=source_path,
        source_sha256=sha256_file(source_path),
        sigma_orientation=source.sigma_orientation,
    )
    qa_dir = root / "synthetic_qa"
    repairs = {"temperature": temperature_report, "salinity": salinity_report}
    validation = validate_forcing(
        output,
        boundary,
        qa_dir,
        repair_codes={"temperature": temperature_codes, "salinity": salinity_codes},
        repair_times_ms=source.times_ms,
        repair_reports=repairs,
    )
    assert validation["status"] == "pass_with_repairs"
    assert validation["structural_checks"]["zero_nan_gate_pass"]
    for artifact in validation["qa_artifacts"].values():
        assert Path(artifact).is_file()
    bad_output = root / "must_not_exist.nc"
    try:
        write_fvcom_ts_forcing(
            bad_output,
            boundary,
            target_times,
            source.siglay,
            source.siglev,
            temperature_out[:, :, :-1],
            salinity_out,
            case_name="bad",
            source_name=source_path,
            source_sha256="0" * 64,
            sigma_orientation=source.sigma_orientation,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid forcing shape was accepted")
    assert not bad_output.exists()
    report = {"status": "pass", "output": str(output.resolve()), "validation": validation}
    write_json_atomic(root / "selftest_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = nullcontext(Path(args.work_dir).resolve()) if args.work_dir else tempfile.TemporaryDirectory(prefix="fvcom_boundary_ts_")
    with context as location:
        root = Path(location)
        report = run(root)
        print(f"[PASS] Synthetic FVCOM boundary T/S tests: {report['output']}")
        if args.work_dir:
            print(f"[PASS] Retained self-test products: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
