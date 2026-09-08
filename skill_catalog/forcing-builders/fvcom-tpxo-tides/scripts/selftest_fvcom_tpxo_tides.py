#!/usr/bin/env python3
"""Offline synthetic tests for production FVCOM TPXO tide forcing."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import netCDF4 as nc4
import numpy as np

from build_fvcom_tides import (
    GateError,
    build,
    build_time_axis,
    load_boundary_points,
    load_harmonics,
    parse_utc,
    validate_forcing,
    RECONSTRUCTION_RULE_VERSION,
)
from prepare_obc_points import ordered_nodes_sha256, prepare


NAMES = [
    "M2", "S2", "N2", "K2", "2N2", "MU2", "NU2", "L2", "K1", "O1", "P1",
    "Q1", "2Q1", "J1", "OO1", "S1", "MF", "MM", "M4", "MN4", "MS4", "M3",
]


def write_point_product(path: Path, ids: list[str]) -> None:
    with nc4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.product_schema = "tpxo9v5_harmonics_v1"
        ds.createDimension("constituent", 22)
        ds.createDimension("point", 2)
        ds.createVariable("constituent_name", str, ("constituent",))[:] = np.asarray(NAMES, dtype=object)
        ds.createVariable("target_id", str, ("point",))[:] = np.asarray(ids, dtype=object)
        ds.createVariable("longitude", "f8", ("point",))[:] = [-94.8, -94.7]
        ds.createVariable("latitude", "f8", ("point",))[:] = [28.8, 28.9]
        phase = np.deg2rad(np.asarray([359.0, 1.0]))
        coefficient = np.empty((22, 2), dtype=np.complex128)
        for index in range(22):
            coefficient[index] = (100.0 + index) * np.exp(-1j * phase)
        real = ds.createVariable("elevation_real", "f4", ("constituent", "point"))
        imag = ds.createVariable("elevation_imaginary", "f4", ("constituent", "point"))
        real.units = imag.units = "mm"
        real[:] = coefficient.real
        imag[:] = coefficient.imag
        flag = ds.createVariable("elevation_interpolation_flag", "i1", ("constituent", "point"))
        flag[:] = 0


def write_native_product(path: Path) -> None:
    with nc4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.product_schema = "tpxo9v5_harmonics_v1"
        ds.createDimension("constituent", 22)
        ds.createDimension("latitude_z", 2)
        ds.createDimension("longitude_z", 2)
        ds.createVariable("constituent_name", str, ("constituent",))[:] = np.asarray(NAMES, dtype=object)
        ds.createVariable("latitude_z", "f8", ("latitude_z",))[:] = [28.0, 29.0]
        ds.createVariable("longitude_z", "f8", ("longitude_z",))[:] = [265.0, 266.0]
        coefficient = np.ones((22, 2, 2), dtype=np.complex128) * (0.2 + 0.1j)
        coefficient[:, 1, 1] = np.nan + 1j * np.nan
        real = ds.createVariable("elevation_real", "f4", ("constituent", "latitude_z", "longitude_z"))
        imag = ds.createVariable("elevation_imaginary", "f4", ("constituent", "latitude_z", "longitude_z"))
        real.units = imag.units = "m"
        real[:] = coefficient.real
        imag[:] = coefficient.imag


def run() -> None:
    with tempfile.TemporaryDirectory(prefix="fvcom_tpxo_selftest_") as temp:
        root = Path(temp)
        mesh = root / "grid.2dm"
        mesh.write_text(
            "MESH2D\nE3T 1 42 7 9 1\nND 42 -94.8 28.8 -2\n"
            "ND 7 -94.7 28.9 -3\nND 9 -94.6 28.7 -4\n",
            encoding="utf-8",
        )
        mesh_sha = hashlib.sha256(mesh.read_bytes()).hexdigest()
        contract = root / "grid_contract.json"
        contract.write_text(
            json.dumps(
                {
                    "schema_version": "fvcom_grid_delivery_contract_v1",
                    "mesh": {
                        "path": "grid.2dm",
                        "sha256": mesh_sha,
                        "node_count": 3,
                        "triangle_count": 1,
                        "coordinate_crs": "EPSG:4326",
                    },
                    "open_boundaries": [
                        {
                            "obc_id": "obc_001",
                            "node_ids": [42, 7],
                            "node_order_sha256": ordered_nodes_sha256([42, 7]),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        points_csv = root / "obc_points.csv"
        points_report = prepare(contract, points_csv, root / "obc_points_manifest.json")
        assert points_report["status"] == "ready" and points_report["obc_node_count"] == 2
        assert load_boundary_points(points_csv).node_ids.tolist() == [42, 7]
        harmonics = root / "harmonics.nc"
        write_point_product(harmonics, ["42", "7"])
        args = argparse.Namespace(
            harmonics=str(harmonics),
            obc_points=str(points_csv),
            obc_dat=None,
            start=parse_utc("2025-03-25T00:00:00Z"),
            end=parse_utc("2025-03-25T01:00:00Z"),
            interval_minutes=6,
            expected_constituents=22,
            constituent_count_explanation=None,
            nearest_wet_max_deg=1.0,
            case_name="synthetic_tpxo",
            output=str(root / "elevtide.nc"),
            diagnostics=str(root / "diagnostics.nc"),
            report=str(root / "report.json"),
        )
        report = build(args)
        assert report["status"] == "ready"
        assert report["constituent_count"] == 22 and report["time_count"] == 11
        with nc4.Dataset(args.output) as ds:
            assert ds.reconstruction_rule_version == RECONSTRUCTION_RULE_VERSION
            assert json.loads(ds.utide_ngflags_json) == [False, False, False, False]
            assert ds.builder_sha256 == report['hashes']['builder_sha256']
            assert ds.utide_version == report['provenance']['utide_version']
            assert np.dtype(ds["time"].dtype).itemsize == 8
            assert np.array_equal(ds["obc_nodes"][:], [42, 7])
            assert ds["elevation"].shape == (11, 2)
            assert np.all(np.diff(ds["time"][:]) > 0)
            encoded_times = [str(value) for value in nc4.chartostring(ds["Times"][:])]
            assert encoded_times[0] == "2025/03/25 00:00:00.000000"
            assert encoded_times[-1] == "2025/03/25 01:00:00.000000"
        with nc4.Dataset(args.diagnostics) as ds:
            phase = np.asarray(ds["elevation_phase_lag"][0], dtype=float)
            assert np.allclose(phase, [359.0, 1.0], atol=1e-4)
            amplitude = np.asarray(ds["elevation_amplitude"][0], dtype=float)
            assert np.allclose(amplitude, 0.1, atol=1e-6), "millimetres must convert to metres"
        assert json.loads(Path(args.report).read_text(encoding="utf-8"))["health"]["status"] == "pass"
        original_forcing = Path(args.output).read_bytes()
        _, short_mjd = build_time_axis(args.start, args.end, 6)
        for attribute, bad_value in [('reconstruction_rule_version', 'midpoint'),
                                     ('utide_ngflags_json', '[true, false, false, false]'),
                                     ('builder_sha256', '0' * 64)]:
            with nc4.Dataset(args.output, 'a') as ds:
                ds.setncattr(attribute, bad_value)
            assert validate_forcing(args.output, load_boundary_points(points_csv), short_mjd, 360)['status'] == 'fail'
            Path(args.output).write_bytes(original_forcing)
        assert validate_forcing(args.output, load_boundary_points(points_csv), short_mjd, 360)['status'] == 'pass'
        full_times, full_mjd = build_time_axis(
            parse_utc("2025-03-25T00:00:00Z"),
            parse_utc("2025-05-01T00:00:00Z"),
            6,
        )
        assert full_times.size == full_mjd.size == 8881
        assert str(full_times[0]) == "2025-03-25T00:00:00"
        assert str(full_times[-1]) == "2025-05-01T00:00:00"

        mismatched = root / "mismatched.nc"
        write_point_product(mismatched, ["7", "42"])
        try:
            load_harmonics(mismatched, load_boundary_points(points_csv), 1.0)
        except GateError as exc:
            assert exc.code == "node_order_mismatch"
        else:
            raise AssertionError("node-order mismatch was not rejected")

        native = root / "native.nc"
        write_native_product(native)
        native_points = root / "native_points.csv"
        native_points.write_text(
            "node_id,longitude,latitude\n1,-94.0,29.0\n",
            encoding="utf-8",
        )
        native_harmonics = load_harmonics(native, load_boundary_points(native_points), 2.0)
        assert np.all(native_harmonics.flags == 1), "dry native corner must use nearest-wet fallback"
        assert np.all(np.isfinite(native_harmonics.coefficient_m))
    print("PASS: FVCOM TPXO tide forcing synthetic tests")


if __name__ == "__main__":
    run()
