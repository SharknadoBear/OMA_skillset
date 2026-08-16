#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import tempfile
from contextlib import nullcontext
from pathlib import Path

try:
    import matplotlib  # noqa: F401
    import netCDF4 as nc4
    import numpy as np
    import scipy  # noqa: F401

    import surface_fluxes_core as core
    from surface_fluxes_validation import validate_forcing_files
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency {exc.name!r}. Install numpy, scipy, netCDF4, and matplotlib on this equipment before continuing."
    ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic tests for modular FVCOM surface flux forcing.")
    parser.add_argument("--work-dir", help="Retain products here; otherwise use a temporary directory")
    return parser


def _mesh_files(root: Path) -> tuple[Path, Path]:
    mesh = root / "synthetic.2dm"
    mesh.write_text(
        "\n".join(
            (
                "MESH2D",
                "E3T 10 1 2 3 1",
                "E3T 20 2 4 3 1",
                "E3T 30 3 4 5 1",
                "E3T 40 4 6 5 1",
                "ND 1 -75.4 38.0 5",
                "ND 2 -75.1 38.0 8",
                "ND 3 -75.4 38.3 12",
                "ND 4 -75.1 38.3 16",
                "ND 5 -75.4 38.6 20",
                "ND 6 -75.1 38.6 25",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    grd = root / "synthetic_grd.dat"
    grd.write_text(
        "\n".join(
            (
                "Node Number = 6",
                "Cell Number = 4",
                "10 1 2 3 1",
                "20 2 4 3 1",
                "30 3 4 5 1",
                "40 4 6 5 1",
                "1 -75.4 38.0 5",
                "2 -75.1 38.0 8",
                "3 -75.4 38.3 12",
                "4 -75.1 38.3 16",
                "5 -75.4 38.6 20",
                "6 -75.1 38.6 25",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    return mesh, grd


def _base_fields(times: int, width: int) -> dict[str, np.ndarray]:
    hour = np.arange(times, dtype=np.float64)[:, None]
    space = np.linspace(0.0, 1.0, width)[None, :]
    return {
        "eastward_wind": 4.0 + np.sin(2 * np.pi * hour / 24.0) + space,
        "northward_wind": -1.0 + np.cos(2 * np.pi * hour / 18.0) - 0.2 * space,
        "eastward_stress": 0.08 + 0.02 * np.sin(2 * np.pi * hour / 24.0) + 0.01 * space,
        "northward_stress": -0.03 + 0.01 * np.cos(2 * np.pi * hour / 18.0) - 0.005 * space,
        "net_shortwave": np.maximum(0.0, 450.0 * np.sin(np.pi * (hour % 24) / 24.0)) + 10.0 * space,
        "total_net_heat_flux": 80.0 + 120.0 * np.sin(2 * np.pi * hour / 24.0) + 5.0 * space,
        "air_temperature": 291.15 + 4.0 * np.sin(2 * np.pi * hour / 24.0) + space,
        "relative_humidity": 0.70 + 0.08 * np.cos(2 * np.pi * hour / 24.0) - 0.02 * space,
        "absolute_air_pressure": 1012.0 + 5.0 * np.sin(2 * np.pi * hour / 36.0) + 0.5 * space,
        "downward_longwave": 330.0 + 10.0 * np.cos(2 * np.pi * hour / 24.0) + space,
        "downward_shortwave": np.maximum(0.0, 600.0 * np.sin(np.pi * (hour % 24) / 24.0)) + 5.0 * space,
        "precipitation": np.maximum(0.0, 2.0e-4 * np.sin(2 * np.pi * hour / 17.0)) + 1.0e-5 * space,
        "evaporation": 3.0 + 0.4 * np.sin(2 * np.pi * hour / 24.0) + 0.1 * space,
    }


UNITS = {
    "eastward_wind": "m s-1",
    "northward_wind": "m s-1",
    "eastward_stress": "Pa",
    "northward_stress": "Pa",
    "net_shortwave": "W m-2",
    "total_net_heat_flux": "W m-2",
    "air_temperature": "Kelvin",
    "relative_humidity": "1",
    "absolute_air_pressure": "hPa",
    "downward_longwave": "W m-2",
    "downward_shortwave": "W m-2",
    "precipitation": "kg m-2 s-1",
    "evaporation": "mm day-1",
}


def _write_source(path: Path, layout: str, mesh: core.MeshGeometry) -> None:
    nt, ny, nx = 49, 4, 5
    with nc4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.pressure_reference = "absolute"
        dataset.createDimension("time", nt)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "hours since 2020-02-28 00:00:00"
        time.calendar = "standard"
        time[:] = np.arange(nt)
        if layout == "structured":
            dataset.createDimension("y", ny)
            dataset.createDimension("x", nx)
            lat_axis = np.linspace(37.8, 38.8, ny)
            lon_axis = np.linspace(-75.8, -74.7, nx)
            lon, lat = np.meshgrid(lon_axis, lat_axis)
            lat_var = dataset.createVariable("latitude", "f8", ("y", "x"))
            lon_var = dataset.createVariable("longitude", "f8", ("y", "x"))
            lat_var.units, lon_var.units = "degrees_north", "degrees_east"
            lat_var[:], lon_var[:] = lat, lon
            node_fields = _base_fields(nt, ny * nx)
            element_fields = node_fields
            dims = ("time", "y", "x")
        else:
            dataset.createDimension("node", len(mesh.node_ids))
            dataset.createDimension("nele", len(mesh.element_ids))
            node_ids = dataset.createVariable("node_id", "i4", ("node",))
            element_ids = dataset.createVariable("element_id", "i4", ("nele",))
            node_ids[:], element_ids[:] = mesh.node_ids, mesh.element_ids
            node_fields = _base_fields(nt, len(mesh.node_ids))
            element_fields = _base_fields(nt, len(mesh.element_ids))
            dims = None
        for role in UNITS:
            is_element = role in {"eastward_wind", "northward_wind", "eastward_stress", "northward_stress"}
            values = element_fields[role] if is_element else node_fields[role]
            if layout == "structured":
                values = values.reshape(nt, ny, nx)
                variable_dims = dims
            else:
                variable_dims = ("time", "nele" if is_element else "node")
            variable = dataset.createVariable(role, "f8", variable_dims, fill_value=-9999.0)
            variable.units = UNITS[role]
            if role == "absolute_air_pressure":
                variable.pressure_reference = "absolute"
            variable[:] = values


def _all_subsets() -> list[tuple[str, ...]]:
    return [
        subset
        for size in range(1, len(core.PACKAGES) + 1)
        for subset in itertools.combinations(core.PACKAGES, size)
    ]


def run(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    mesh_path, grd_path = _mesh_files(root)
    mesh = core.read_mesh_2dm(mesh_path)
    mesh_grd = core.read_mesh_grd(grd_path)
    assert np.array_equal(mesh.node_ids, mesh_grd.node_ids)
    assert np.array_equal(mesh.element_ids, mesh_grd.element_ids)
    assert np.allclose(mesh.lonc, mesh_grd.lonc)

    structured_source = root / "prepared_structured.nc"
    native_source = root / "prepared_native.nc"
    _write_source(structured_source, "structured", mesh)
    _write_source(native_source, "fvcom", mesh)
    combinations = 0
    representative_files: dict[str, list[Path]] = {"structured": [], "fvcom": []}
    for layout, source, geometry in (
        ("structured", structured_source, None),
        ("fvcom", native_source, mesh),
    ):
        for index, packages in enumerate(_all_subsets()):
            data = core.read_prepared_netcdf(
                source,
                layout=layout,
                packages=packages,
                mesh=geometry,
                pressure_reference="absolute",
            )
            for file_layout in ("combined", "split"):
                result = core.write_prepared_bundle(
                    data,
                    root / "combinations" / layout / file_layout / f"case_{index:02d}",
                    case_name=f"{layout}_{file_layout}_{index:02d}",
                    packages=packages,
                    file_layout=file_layout,
                )
                assert all(path.is_file() for path in result.files.values())
                combinations += 1
                if file_layout == "combined" and set(packages) == set(core.PACKAGES):
                    representative_files[layout].extend(result.files.values())

    direct_reports = {
        "structured": validate_forcing_files(
            representative_files["structured"], root / "qa_direct" / "structured"
        ),
        "fvcom": validate_forcing_files(
            representative_files["fvcom"], root / "qa_direct" / "fvcom", mesh=mesh
        ),
    }
    assert all(
        report["status"] == "pass" and report["structural_checks"]["zero_nan_gate"]
        for report in direct_reports.values()
    )
    try:
        validate_forcing_files(
            [representative_files["structured"][0], representative_files["fvcom"][0]],
            root / "qa_mixed_contract_failure",
        )
    except ValueError as exc:
        assert "mix structured and FVCOM-native" in str(exc)
    else:
        raise AssertionError("Mixed structured/native bundle was accepted")
    percent_values, percent_units, _ = core.convert_field(
        "relative_humidity", np.asarray([50.0]), "percent", None
    )
    assert percent_units == "percent" and float(percent_values[0]) == 50.0

    bulk = core.read_prepared_netcdf(
        structured_source,
        layout="structured",
        packages=core.PACKAGES,
        wind_mode="speed",
        heat_mode="bulk",
        pressure_reference="absolute",
    )
    assert np.allclose(bulk.fields["air_temperature"].mean(), _base_fields(49, 20)["air_temperature"].mean() - 273.15)
    assert 60.0 < bulk.fields["relative_humidity"].mean() < 80.0
    assert 100_000.0 < bulk.fields["absolute_air_pressure"].mean() < 103_000.0
    assert bulk.fields["precipitation"].max() < 1.0e-6
    assert bulk.fields["evaporation"].mean() < 1.0e-7

    coare26 = core.write_prepared_bundle(
        bulk,
        root / "bulk_coare26",
        case_name="bulk_coare26",
        packages=core.PACKAGES,
        heat_mode="bulk",
        coare_version="COARE26Z",
    )
    assert len(coare26.files) == 1
    bulk_report = validate_forcing_files(coare26.files.values(), root / "qa_bulk")
    assert bulk_report["status"] == "pass"

    coare40 = core.write_prepared_bundle(
        bulk,
        root / "bulk_coare40",
        case_name="bulk_coare40",
        packages=core.PACKAGES,
        heat_mode="bulk",
        coare_version="COARE40VN",
    )
    assert len(coare40.files) == 4
    with nc4.Dataset(coare40.package_files["heat"]) as dataset:
        assert dataset.variables["air_pressure"].units == "hPa"
        assert 900.0 < float(np.mean(dataset.variables["air_pressure"][:])) < 1100.0
    with nc4.Dataset(coare40.package_files["pressure"]) as dataset:
        assert dataset.variables["air_pressure"].units == "Pa"
        assert 90_000.0 < float(np.mean(dataset.variables["air_pressure"][:])) < 110_000.0
    try:
        core.write_prepared_bundle(
            bulk,
            root / "unsafe",
            case_name="unsafe",
            packages=core.PACKAGES,
            heat_mode="bulk",
            coare_version="COARE40VN",
            file_layout="combined",
        )
    except ValueError as exc:
        assert "unsafe" in str(exc)
    else:
        raise AssertionError("Unsafe COARE40VN combined pressure file was accepted")

    heat_only = core.read_prepared_netcdf(
        structured_source,
        layout="structured",
        packages=["heat"],
        heat_mode="bulk",
        pressure_reference="absolute",
    )
    try:
        core.write_prepared_bundle(
            heat_only,
            root / "bulk_without_wind",
            case_name="bulk_without_wind",
            packages=["heat"],
            heat_mode="bulk",
        )
    except ValueError as exc:
        assert "wind" in str(exc).lower()
    else:
        raise AssertionError("Bulk heat without wind-speed confirmation was accepted")
    confirmed = core.write_prepared_bundle(
        heat_only,
        root / "bulk_external_wind",
        case_name="bulk_external_wind",
        packages=["heat"],
        heat_mode="bulk",
        external_wind_speed=True,
    )
    assert len(confirmed.files) == 1

    stress = core.read_prepared_netcdf(
        native_source,
        layout="fvcom",
        packages=["wind"],
        wind_mode="stress",
        mesh=mesh,
    )
    stress_result = core.write_prepared_bundle(
        stress,
        root / "stress",
        case_name="stress",
        packages=["wind"],
        wind_mode="stress",
        file_layout="split",
    )
    assert stress_result.package_files["wind"].is_file()

    times = np.arange(np.datetime64("2020-01-01T00:00:00"), np.datetime64("2020-01-01T04:00:00"), np.timedelta64(1, "h"))
    api_fields = {
        "absolute_air_pressure": np.full((len(times), 2, 3), 101_325.0),
    }
    api_result = core.write_surface_forcing_bundle(
        root / "python_api",
        "python_api",
        layout="structured",
        times_utc=times,
        fields=api_fields,
        units={"absolute_air_pressure": "Pa"},
        packages=["pressure"],
        lat=np.linspace(38.0, 39.0, 2),
        lon=np.linspace(-76.0, -74.0, 3),
        pressure_reference="absolute",
    )
    assert api_result.package_files["pressure"].is_file()
    irregular_times = np.asarray(
        ["2020-01-01T00:00:00", "2020-01-01T01:00:00", "2020-01-01T03:00:00"],
        dtype="datetime64[s]",
    )
    irregular_result = core.write_surface_forcing_bundle(
        root / "irregular_api",
        "irregular",
        layout="structured",
        times_utc=irregular_times,
        fields={"absolute_air_pressure": np.full((3, 2, 3), 101_325.0)},
        units={"absolute_air_pressure": "Pa"},
        packages=["pressure"],
        lat=np.linspace(38.0, 39.0, 2),
        lon=np.linspace(-76.0, -74.0, 3),
        pressure_reference="absolute",
    )
    irregular_report = validate_forcing_files(
        irregular_result.files.values(), root / "qa_irregular"
    )
    assert not irregular_report["files"][0]["regular_time"]
    try:
        core.write_prepared_bundle(
            core.prepare_arrays(
                layout="structured",
                times_utc=times,
                fields=api_fields,
                units={"absolute_air_pressure": "Pa"},
                packages=["pressure"],
                lat=np.linspace(38.0, 39.0, 2),
                lon=np.linspace(-76.0, -74.0, 3),
                pressure_reference="absolute",
            ),
            root / "coverage_failure",
            case_name="coverage_failure",
            packages=["pressure"],
            model_start_ms=int(np.datetime64("2019-12-31T23:00:00", "ms").astype(np.int64)),
        )
    except ValueError as exc:
        assert "model start" in str(exc)
    else:
        raise AssertionError("Insufficient model coverage was accepted")
    try:
        core.prepare_arrays(
            layout="structured",
            times_utc=times,
            fields=api_fields,
            units={"absolute_air_pressure": "Pa"},
            packages=["pressure"],
            lat=np.linspace(38.0, 39.0, 2),
            lon=np.linspace(-76.0, -74.0, 3),
        )
    except ValueError as exc:
        assert "pressure_reference" in str(exc)
    else:
        raise AssertionError("Pressure without an absolute reference was accepted")
    try:
        core.prepare_arrays(
            layout="structured",
            times_utc=["2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"],
            fields={"absolute_air_pressure": np.ones((2, 2, 3)) * 101_325},
            units={"absolute_air_pressure": "Pa"},
            packages=["pressure"],
            lat=np.linspace(38.0, 39.0, 2),
            lon=np.linspace(-76.0, -74.0, 3),
            pressure_reference="absolute",
        )
    except ValueError as exc:
        assert "increasing" in str(exc)
    else:
        raise AssertionError("Duplicate time was accepted")
    bad = api_fields["absolute_air_pressure"].copy()
    bad[0, 0, 0] = np.nan
    failed_output = root / "nan_failure" / "nan_aip.nc"
    try:
        core.write_surface_forcing_bundle(
            failed_output.parent,
            "nan",
            layout="structured",
            times_utc=times,
            fields={"absolute_air_pressure": bad},
            units={"absolute_air_pressure": "Pa"},
            packages=["pressure"],
            lat=np.linspace(38.0, 39.0, 2),
            lon=np.linspace(-76.0, -74.0, 3),
            pressure_reference="absolute",
        )
    except ValueError as exc:
        assert "NaN" in str(exc)
    else:
        raise AssertionError("NaN source was accepted")
    assert not failed_output.exists()

    atomic_directory = root / "staged_failure"
    original_write_one = core._write_one
    staged_calls = 0

    def fail_second_staged_write(*args, **kwargs):
        nonlocal staged_calls
        staged_calls += 1
        original_write_one(*args, **kwargs)
        if staged_calls == 2:
            raise RuntimeError("synthetic staged-write failure")

    core._write_one = fail_second_staged_write
    try:
        core.write_prepared_bundle(
            bulk,
            atomic_directory,
            case_name="atomic_failure",
            packages=core.PACKAGES,
            heat_mode="bulk",
            file_layout="split",
        )
    except RuntimeError as exc:
        assert "staged-write failure" in str(exc)
    else:
        raise AssertionError("Synthetic staged-write failure did not propagate")
    finally:
        core._write_one = original_write_one
    assert not list(atomic_directory.glob("*.nc"))
    assert not list(atomic_directory.glob("*.tmp"))

    mismatch = root / "node_mismatch.nc"
    _write_source(mismatch, "fvcom", mesh)
    with nc4.Dataset(mismatch, "a") as dataset:
        dataset.variables["node_id"][0] = 999
    try:
        core.read_prepared_netcdf(mismatch, layout="fvcom", packages=["pressure"], mesh=mesh, pressure_reference="absolute")
    except ValueError as exc:
        assert "node_id order" in str(exc)
    else:
        raise AssertionError("Native node mismatch was accepted")

    report = {
        "status": "pass",
        "package_combinations_tested": combinations,
        "layouts": ["structured", "fvcom"],
        "direct_validation": direct_reports,
        "bulk_validation": bulk_report,
        "coare40_split_files": [str(path.resolve()) for path in coare40.files.values()],
        "python_api_output": str(api_result.package_files["pressure"].resolve()),
        "irregular_time_output": str(irregular_result.package_files["pressure"].resolve()),
    }
    core.write_json_atomic(root / "selftest_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = nullcontext(Path(args.work_dir).resolve()) if args.work_dir else tempfile.TemporaryDirectory(prefix="fvcom_surface_fluxes_")
    with context as location:
        root = Path(location)
        report = run(root)
        print(
            f"[PASS] FVCOM surface flux self-test: "
            f"{report['package_combinations_tested']} package/spatial/file-layout combinations"
        )
        if args.work_dir:
            print(f"[PASS] Retained products: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
