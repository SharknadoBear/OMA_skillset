"""Offline transport and scientific tests. Temporary payloads are explicitly synthetic."""
import argparse
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import zipfile

import numpy as np
import xarray as xr

import era5_core as core


class SimulatedCDS:
    def __init__(self, mode="zip", size_limit=None, transient=0, interrupt=False):
        self.mode, self.size_limit = mode, size_limit
        self.transient, self.interrupt = transient, interrupt
        self.calls = 0

    def retrieve(self, request, target):
        self.calls += 1
        if self.interrupt:
            Path(target).write_bytes(b"incomplete")
            self.interrupt = False
            raise KeyboardInterrupt()
        if self.transient:
            self.transient -= 1
            raise core.TransientError("simulated network failure")
        year, month = request["year"][0], request["month"][0]
        times = np.array([np.datetime64(f"{year}-{month}-{day}T{hour}", "ns") for day in request["day"] for hour in request["time"]])
        if self.size_limit and len(times) > self.size_limit:
            raise core.SizeLimitError("simulated cost limit")
        n, w, s, e = request["area"]
        lat = np.arange(n, s - .01, -.25)
        lon = np.arange(w, e + .01, .25) % 360
        shape = (len(times), len(lat), len(lon))
        base = np.arange(np.prod(shape), dtype="float32").reshape(shape) / 10000
        fields = {}
        names = {v[0]: v[1] for p in core.PRODUCTS.values() for v in p}
        for variable in request["variable"]:
            name = names[variable]
            values = base + {"u10": 2, "v10": -3, "msl": 100000}[name]
            fields[name] = (("valid_time", "latitude", "longitude"), values, {"units": "Pa" if name == "msl" else "m s**-1", "source_test": "synthetic"})
        ds = xr.Dataset(fields, coords={"valid_time": times, "latitude": lat, "longitude": lon,
                                       "expver": ("valid_time", ["0001"] * len(times)), "number": 0})
        if self.mode == "missing_hour":
            ds = ds.isel(valid_time=slice(1, None))
        elif self.mode == "duplicate_hour":
            ds = ds.assign_coords(valid_time=np.concatenate((times[:1], times[:-1])))
        elif self.mode == "missing_field":
            ds = ds.drop_vars("v10")
        elif self.mode == "units":
            ds.msl.attrs["units"] = "hPa"
        elif self.mode == "nan":
            ds.u10.values.flat[0] = np.nan
        elif self.mode == "wrong_grid":
            ds = ds.assign_coords(latitude=lat + .01)
        elif self.mode == "expver":
            ds = ds.drop_vars("expver").expand_dims(expver=[1, 5])
        elif self.mode == "wind_orientation":
            ds.u10.attrs["GRIB_uvRelativeToGrid"] = 1
        elif self.mode == "wrong_height":
            ds.u10.attrs["GRIB_level"] = 100
        elif self.mode == "wrong_parameter":
            ds.msl.attrs["GRIB_paramId"] = 134
        elif self.mode == "corrupt_zip":
            Path(target).write_bytes(b"PK\x03\x04broken")
            return
        elif self.mode == "html":
            Path(target).write_text("<html>server error</html>")
            return
        if self.mode == "netcdf":
            ds.to_netcdf(target, engine="netcdf4")
            return
        with tempfile.TemporaryDirectory(dir=Path(target).parent) as temporary:
            nc = Path(temporary) / "source.nc"
            ds.to_netcdf(nc, engine="netcdf4")
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                if self.mode == "unsafe_zip":
                    archive.write(nc, "../../escape.nc")
                elif self.mode == "split_members":
                    for name in ds.data_vars:
                        nc_part = Path(temporary) / (name + ".nc")
                        ds[[name]].to_netcdf(nc_part, engine="netcdf4")
                        archive.write(nc_part, nc_part.name)
                else:
                    archive.write(nc, "data.nc")


class ERA5Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.request = {"schema_version": "era5_request_v1", "start": "2025-08-30T00:00:00Z", "end": "2025-09-03T00:00:00Z",
                        "products": ["wind_10m", "mean_sea_level_pressure"], "bbox": [-121.1, 34.8, -120.6, 35.2], "halo_cells": 1}
        self.request_path = self.root / "request.json"

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, request=None, name="run"):
        core.write_json(self.request_path, request or self.request)
        root = self.root / name
        core.save_plan(self.request_path, root)
        return root

    def test_monthly_plan_and_api_selection(self):
        plan = core.build_plan(self.request)
        self.assertEqual((plan["total_hours"], plan["chunk_count"]), (96, 2))
        self.assertEqual([c["hours"] for c in plan["chunks"]], [48, 48])
        self.assertEqual(plan["request"]["area"], [35.5, -121.5, 34.5, -120.25])
        for chunk in plan["chunks"]:
            self.assertEqual(len(chunk["request"]["day"]) * len(chunk["request"]["time"]), chunk["hours"])

    def test_mesh_bounds_all_nodes_and_halo(self):
        mesh = self.root / "fort.14"
        mesh.write_text("geographic mesh\n1 4\n1 -121.191237894441 34.74508269552 1\n2 -120.610194456324 35.260432 1\n3 -120.9 35.0 1\n4 -121.0 35.1 -4\n1 3 1 2 3\n")
        request = {k: v for k, v in self.request.items() if k != "bbox"}
        request["mesh"] = {"path": "fort.14", "crs": "EPSG:4326"}
        plan = core.build_plan(request, self.root)
        self.assertEqual(plan["request"]["area"], [35.75, -121.5, 34.25, -120.25])
        self.assertEqual(plan["request"]["source_mesh"]["nodes"], 4)
        self.assertEqual(plan["request"]["shape"], [7, 6])

    def test_longitude_normalization(self):
        other = dict(self.request, bbox=[238.9, 34.8, 239.4, 35.2])
        self.assertEqual(core.build_plan(other)["request"]["area"], core.build_plan(self.request)["request"]["area"])

    def test_partial_days_exact_cartesian_hours(self):
        plan = core.build_plan(dict(self.request, start="2025-01-31T22:00:00Z", end="2025-02-02T02:00:00Z"))
        self.assertEqual(plan["total_hours"], 28)
        self.assertEqual([c["hours"] for c in plan["chunks"]], [2, 24, 2])
        times = np.concatenate([core.expected_times(c) for c in plan["chunks"]])
        self.assertTrue(np.all(np.diff(times) == np.timedelta64(1, "h")))

    def test_multiyear_and_leap_days(self):
        plan = core.build_plan(dict(self.request, start="2011-01-01T00:00:00Z", end="2026-01-01T00:00:00Z"))
        self.assertEqual((plan["chunk_count"], plan["total_hours"]), (180, 131496))
        self.assertEqual(sum(c["hours"] for c in plan["chunks"]), 131496)
        for first, second in zip(plan["chunks"], plan["chunks"][1:]):
            self.assertEqual(first["end"], second["start"])
        leap = core.build_plan(dict(self.request, start="2024-02-28T00:00:00Z", end="2024-03-01T00:00:00Z"))
        self.assertEqual(leap["total_hours"], 48)
        self.assertEqual(leap["chunks"][0]["request"]["day"], ["28", "29"])

    def test_invalid_requests(self):
        bad = [dict(self.request, products=[]), dict(self.request, products=["heat_flux"]),
               dict(self.request, products=["wind_10m", "wind_10m"]), dict(self.request, halo_cells=True),
               dict(self.request, bbox=[179, 34, -179, 36]), dict(self.request, key="secret"),
               dict(self.request, start="2025-08-30T00:30:00Z"), dict(self.request, start="2025-08-30T00:00:00"),
               dict(self.request, end=self.request["start"]), dict(self.request, mesh={"path": "fort.14", "crs": "EPSG:32610"})]
        for request in bad:
            with self.subTest(request=request), self.assertRaises(core.RequestError):
                core.build_plan(request)

    def test_netcdf_zip_and_split_member_formats(self):
        for mode in ("zip", "netcdf", "split_members"):
            with self.subTest(mode=mode):
                root = self.prepare(name=mode)
                result = core.run(root, SimulatedCDS(mode))
                self.assertTrue(result["health"]["pass_all"])
                with xr.open_dataset(next((root / "fields").glob("*.nc"))) as ds:
                    self.assertTrue(np.all(np.diff(ds.latitude) > 0))
                    self.assertTrue(np.all(ds.longitude < 0))
                    self.assertEqual(set(ds.data_vars), {"u10", "v10", "msl"})
                    self.assertEqual(set(ds.expver.values), {"0001"})

    def test_controlled_resume_and_no_transfer_repeat(self):
        root = self.prepare()
        transport = SimulatedCDS()
        first = core.run(root, transport, max_chunks=1)
        self.assertFalse(first["health"]["pass_all"])
        self.assertEqual(first["health"]["completed_hours"], 48)
        second = core.run(root, transport)
        self.assertTrue(second["health"]["pass_all"])
        self.assertEqual((second["new_chunks"], second["reused_chunks"]), (1, 1))
        third = core.run(root, transport)
        self.assertEqual((third["new_chunks"], third["reused_chunks"], transport.calls), (0, 2, 2))

    def test_interrupted_transfer_reacquired(self):
        root = self.prepare()
        transport = SimulatedCDS(interrupt=True)
        with self.assertRaises(KeyboardInterrupt):
            core.run(root, transport)
        self.assertFalse((root / ".era5.lock").exists())
        result = core.run(root, transport)
        self.assertTrue(result["health"]["pass_all"])
        self.assertEqual(transport.calls, 3)

    def test_corrupt_committed_output_not_skipped(self):
        root = self.prepare()
        transport = SimulatedCDS()
        core.run(root, transport)
        next((root / "fields").glob("*.nc")).write_bytes(b"corrupt")
        with self.assertRaises(core.ValidationError):
            core.run(root, transport)
        self.assertEqual(transport.calls, 2)
        self.assertFalse(core.health(root)["pass_all"])

    def test_changed_request_and_plan_rejected(self):
        root = self.prepare()
        core.write_json(self.request_path, dict(self.request, halo_cells=2))
        with self.assertRaises(core.RequestError):
            core.save_plan(self.request_path, root)
        plan = core.read_json(root / "download_plan.json")
        plan["chunks"][0]["request"]["area"][0] += .25
        core.write_json(root / "download_plan.json", plan)
        with self.assertRaises(core.ValidationError):
            core.run(root, SimulatedCDS())

    def test_scientific_and_payload_failures(self):
        for mode in ("missing_hour", "duplicate_hour", "missing_field", "units", "nan", "wrong_grid", "expver", "html", "unsafe_zip", "wind_orientation", "wrong_height", "wrong_parameter", "corrupt_zip"):
            with self.subTest(mode=mode):
                root = self.prepare(name=mode)
                with self.assertRaises(core.ValidationError):
                    core.run(root, SimulatedCDS(mode))
                self.assertFalse(core.health(root)["pass_all"])

    def test_transient_retry_and_bounded_failure(self):
        root = self.prepare()
        transport = SimulatedCDS(transient=1)
        sleeps = []
        self.assertTrue(core.run(root, transport, sleep=sleeps.append)["health"]["pass_all"])
        self.assertEqual(sleeps, [2])
        transport = SimulatedCDS(transient=10)
        with self.assertRaises(core.TransientError):
            core.run(self.prepare(name="failed"), transport, sleep=lambda _: None)
        self.assertEqual(transport.calls, 3)

    def test_size_limits_subdivide_and_resume(self):
        root = self.prepare()
        transport = SimulatedCDS(size_limit=24)
        result = core.run(root, transport)
        self.assertTrue(result["health"]["pass_all"])
        self.assertEqual(result["health"]["expected_chunks"], 4)
        calls = transport.calls
        self.assertEqual(core.run(root, transport)["new_chunks"], 0)
        self.assertEqual(transport.calls, calls)

    def test_active_lock_blocks_second_writer(self):
        root = self.prepare()
        with core.run_lock(root):
            with self.assertRaises(core.RequestError):
                core.run(root, SimulatedCDS())

    def test_access_failure_is_not_retried(self):
        class Denied:
            calls = 0
            def retrieve(self, request, target):
                self.calls += 1
                raise core.AccessError("simulated licence rejection")
        backend = Denied()
        with self.assertRaises(core.AccessError):
            core.run(self.prepare(), backend, sleep=lambda _: None)
        self.assertEqual(backend.calls, 1)

    def test_real_transport_error_classification_and_redaction(self):
        import requests
        backend = object.__new__(core.CDSBackend)
        backend.legacy = Mock()
        backend.legacy.key = "synthetic-secret"
        response = requests.Response()
        response.status_code = 403
        backend.legacy.retrieve.side_effect = requests.HTTPError("cost limits exceeded", response=response)
        with self.assertRaises(core.SizeLimitError):
            backend.retrieve({}, self.root / "unused")
        backend.legacy.retrieve.side_effect = requests.HTTPError("licence not accepted", response=response)
        with self.assertRaises(core.AccessError):
            backend.retrieve({}, self.root / "unused")
        response.status_code = 503
        backend.legacy.retrieve.side_effect = requests.HTTPError("unavailable", response=response)
        with self.assertRaises(core.TransientError):
            backend.retrieve({}, self.root / "unused")
        backend.legacy.retrieve.side_effect = RuntimeError("token=synthetic-secret https://example.test/result?signature=sensitive")
        with self.assertRaises(core.RequestError) as raised:
            backend.retrieve({}, self.root / "unused")
        self.assertNotIn("synthetic-secret", str(raised.exception))
        self.assertNotIn("signature=", str(raised.exception))

    def test_changed_mesh_invalidates_saved_plan(self):
        mesh = self.root / "fort.14"
        mesh.write_text("mesh\n1 3\n1 -121 35 1\n2 -120.75 35 1\n3 -121 35.25 1\n1 3 1 2 3\n")
        request = {k: v for k, v in self.request.items() if k != "bbox"}
        request["mesh"] = {"path": "fort.14", "crs": "EPSG:4326"}
        root = self.prepare(request)
        mesh.write_text(mesh.read_text().replace("35.25", "35.3"))
        with self.assertRaises(core.RequestError):
            core.run(root, SimulatedCDS())

    def test_snapshot_is_exactly_one_hour(self):
        core.write_json(self.request_path, self.request)
        result = core.snapshot(self.request_path, self.root / "smoke", backend=SimulatedCDS())
        self.assertTrue(result["health"]["pass_all"])
        self.assertEqual(result["health"]["completed_hours"], 1)

    def test_optional_streaming_annual_assembly(self):
        request = dict(self.request, start="2024-01-01T00:00:00Z", end="2025-01-01T00:00:00Z", bbox=[-121, 35, -120.75, 35.25], halo_cells=0)
        root = self.prepare(request)
        self.assertTrue(core.run(root, SimulatedCDS())["health"]["pass_all"])
        result = core.assemble_year(root, 2024, self.root / "annual.nc")
        self.assertEqual(result["hours"], 8784)
        with xr.open_dataset(self.root / "annual.nc") as ds:
            self.assertEqual(ds.sizes["time"], 8784)
            self.assertTrue(np.isfinite(ds.msl).all())
        with self.assertRaises(core.RequestError):
            core.assemble_year(root, 2024, self.root / "annual.nc")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ERA5Tests))
    report = {"type": "offline_simulated_transport_tests", "at_utc": core.now(), "tests_run": result.testsRun,
              "pass_all": result.wasSuccessful(), "failures": len(result.failures), "errors": len(result.errors),
              "failure_details": [{"test": str(test), "traceback": message} for test, message in result.failures + result.errors],
              "runtime": core.versions(), "live_download_claim": False}
    core.write_json(args.output, report)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
