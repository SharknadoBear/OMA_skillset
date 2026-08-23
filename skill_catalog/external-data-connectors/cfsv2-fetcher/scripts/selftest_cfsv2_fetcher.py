#!/usr/bin/env python3
"""Offline regression tests for the CFSv2 fetcher and compatibility API."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from cfsv2_fetcher import (
    _plan_gate,
    build_cfsv2_plan,
    cfsv2_airprs_to_absolute_pa,
    fetch_cfsv2_plan,
    fetch_cfsv2_window,
    fetch_wind_year,
    health_cfsv2,
    inventory_cfsv2,
    normalize_subdataset,
    validate_plan,
)


class Cfsv2FetcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        times = np.arange(
            np.datetime64("2020-01-01T00", "h"),
            np.datetime64("2021-01-01T00", "h"),
            np.timedelta64(1, "h"),
        )
        mt = (times.astype("datetime64[ms]") - np.datetime64("1900-12-31T00:00:00", "ms")) / np.timedelta64(1, "D")
        lat = np.array([35.0, 36.0, 37.0], dtype=np.float32)
        lon = np.array([282.0, 283.0, 284.0, 285.0], dtype=np.float32)
        base = np.arange(times.size, dtype=np.float32)[:, None, None]
        wndewd = base + np.zeros((times.size, lat.size, lon.size), dtype=np.float32)
        wndnwd = -base + np.ones_like(wndewd)
        dataset = xr.Dataset(
            {
                "wndewd": (("MT", "Latitude", "Longitude"), wndewd, {"units": "m s-1"}),
                "wndnwd": (("MT", "Latitude", "Longitude"), wndnwd, {"units": "m s-1"}),
            },
            coords={
                "MT": ("MT", mt.astype(np.float64), {"units": "days since 1900-12-31 00:00:00"}),
                "Latitude": ("Latitude", lat, {"units": "degrees_north"}),
                "Longitude": ("Longitude", lon, {"units": "degrees_east"}),
            },
        )
        self.source = self.root / "cfsv2_uv_2020.nc"
        dataset.to_netcdf(self.source, engine="netcdf4")
        dataset.close()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def request(self, output: str = "subset.nc") -> dict:
        return {
            "start": "2020-02-01T00:00:00Z",
            "end": "2020-02-01T12:00:00Z",
            "subdataset": "uv-10m",
            "variables": ["wndewd", "wndnwd"],
            "bbox": [282.5, 35.5, 284.5, 37.0],
            "source_url": str(self.source),
            "chunk_hours": 4,
            "retry_delay_seconds": 0,
            "output": str(self.root / output),
        }

    def test_inventory_alias_and_pressure(self) -> None:
        inventory = inventory_cfsv2(2020, "uv-10m", source_url=str(self.source))
        self.assertIn("wndewd", inventory["variables"])
        self.assertEqual(normalize_subdataset("dlwflx"), "dlwsfc")
        converted = cfsv2_airprs_to_absolute_pa([13.2])
        self.assertAlmostEqual(float(converted[0]), 101320.0)

    def test_plan_fetch_resume_and_health(self) -> None:
        run = self.root / "run"
        plan = build_cfsv2_plan(
            self.request(),
            run_dir=run,
            probe_override={
                "method": "test",
                "conservative_bytes_per_second": 1_000_000,
                "median_request_seconds": 0.01,
            },
            free_bytes_override=10**9,
        )
        self.assertGreater(len(plan["chunks"]), 1)
        self.assertNotIn(Path.home().name.lower(), json.dumps(plan).lower())
        result = fetch_cfsv2_plan(plan, run_dir=run, open_monitor=False)
        self.assertTrue(health_cfsv2(result["output"], plan["request"])["passed"])
        status = json.loads((self.root / "run" / "download_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "complete")
        second = fetch_cfsv2_plan(plan, run_dir=run, open_monitor=False)
        self.assertEqual(second["sha256"], result["sha256"])

    def test_backward_window_and_annual_apis(self) -> None:
        window = fetch_cfsv2_window(
            np.datetime64("2020-03-01T00"),
            np.datetime64("2020-03-01T03"),
            "uv-10m",
            lon_range=(282.5, 284.5),
            lat_range=(35.5, 37.0),
            output=self.root / "compat-window.nc",
            run_dir=self.root / "compat-window-run",
            source_url=str(self.source),
            open_monitor=False,
        )
        self.assertTrue(window.exists())
        annual = fetch_wind_year(
            2020,
            lon_range=(282.5, 284.5),
            lat_range=(35.5, 37.0),
            cache_dir=self.root / "annual",
            chunk_days=100,
            source_url=str(self.source),
            run_dir=self.root / "annual-run",
            open_monitor=False,
        )
        with xr.open_dataset(annual, decode_times=False) as dataset:
            self.assertEqual(dataset.sizes["MT"], 8784)

    def test_legacy_wrong_era_delegates_to_v2_router(self) -> None:
        routed_output = self.root / "cfsr.nc"
        routed_output.touch()
        with patch("cfsv2_fetcher.execute_atmospheric_request", return_value={"model": "cfsr", "output": str(routed_output)} ) as execute:
            result = fetch_cfsv2_window(
                "2008-01-01T00:00:00Z", "2008-01-01T00:00:00Z", "sfcprs",
                lon_range=(283.0, 283.3), lat_range=(38.0, 38.3),
                output=self.root / "legacy.nc", run_dir=self.root / "routed", open_monitor=False,
            )
        self.assertEqual(result, routed_output)
        self.assertEqual(execute.call_args.args[1]["products"], ["surface_pressure"])

    def test_gate_stale_plan_and_coverage(self) -> None:
        self.assertEqual(_plan_gate(1, 100, 599)["state"], "ready")
        self.assertEqual(_plan_gate(1, 100, 600)["state"], "long_run_monitor_required")
        self.assertEqual(_plan_gate(1, 100, float("nan"))["state"], "blocked")
        self.assertEqual(_plan_gate(100, 400, 1)["state"], "blocked")
        plan = build_cfsv2_plan(
            self.request(),
            run_dir=self.root / "hash-plan",
            probe_override={"conservative_bytes_per_second": 1000, "median_request_seconds": 0},
            free_bytes_override=10**9,
        )
        changed = copy.deepcopy(plan)
        changed["chunks"][0]["expected_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_plan(changed)
        missing = self.request()
        missing["start"] = "2019-01-01T00:00:00Z"
        missing["end"] = "2019-01-01T01:00:00Z"
        with self.assertRaisesRegex(ValueError, "No CFSv2 records|coverage"):
            build_cfsv2_plan(
                missing,
                run_dir=self.root / "missing",
                probe_override={"conservative_bytes_per_second": 1000, "median_request_seconds": 0},
                free_bytes_override=10**9,
            )
        failed_probe = self.request()
        with self.assertRaisesRegex(Exception, "positive"):
            build_cfsv2_plan(
                failed_probe,
                run_dir=self.root / "failed-probe",
                probe_override={"conservative_bytes_per_second": 0},
                free_bytes_override=10**9,
            )

    def test_local_source_template_is_portable(self) -> None:
        request = self.request("template-output.nc")
        request["source_url"] = str(self.root / "cfsv2_uv_{year}.nc")
        plan = build_cfsv2_plan(
            request,
            run_dir=self.root / "template-run",
            probe_override={
                "conservative_bytes_per_second": 1_000_000,
                "median_request_seconds": 0.01,
            },
            free_bytes_override=10**9,
        )
        self.assertIn("{year}", plan["request"]["source_url"])
        self.assertNotIn(Path.home().name.lower(), json.dumps(plan).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
