#!/usr/bin/env python3
"""Offline regression tests for the model-neutral HYCOM fetcher."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import xarray as xr

from download_monitor import DownloadStatus, atomic_write_json, monitor_html, safe_message, serve_monitor
from hycom_fetcher import (
    _plan_gate,
    build_hycom_plan,
    fetch_hycom_plan,
    health_hycom,
    inventory_hycom,
    validate_plan,
)


class HycomFetcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        time = np.arange("2020-01-01", "2020-01-07", dtype="datetime64[D]")
        depth = np.array([0.0, 5.0, 15.0], dtype=np.float32)
        lat = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
        lon = np.arange(0.0, 360.0, 10.0, dtype=np.float32)
        shape3 = (time.size, depth.size, lat.size, lon.size)
        shape2 = (time.size, lat.size, lon.size)
        temperature = np.arange(np.prod(shape3), dtype=np.float32).reshape(shape3) / 100.0
        mixed = np.arange(np.prod(shape2), dtype=np.float32).reshape(shape2)
        temperature[0, 0, 0, 0] = np.nan
        dataset = xr.Dataset(
            {
                "temperature": (("time", "depth", "lat", "lon"), temperature, {"units": "degree_C"}),
                "mixed_layer_thickness": (("time2", "lat", "lon"), mixed, {"units": "m"}),
            },
            coords={
                "time": ("time", time, {"standard_name": "time", "axis": "T"}),
                "time2": ("time2", time, {"standard_name": "time", "axis": "T"}),
                "depth": ("depth", depth, {"standard_name": "depth", "axis": "Z", "positive": "down"}),
                "lat": ("lat", lat, {"standard_name": "latitude", "axis": "Y", "units": "degrees_north"}),
                "lon": ("lon", lon, {"standard_name": "longitude", "axis": "X", "units": "degrees_east"}),
            },
            attrs={"title": "synthetic arbitrary-variable HYCOM source"},
        )
        self.source = self.root / "source.nc"
        dataset.to_netcdf(self.source, engine="netcdf4")
        dataset.close()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def request(self, *, points: bool = False) -> dict:
        request = {
            "source": str(self.source),
            "variables": ["temperature", "mixed_layer_thickness"],
            "start": "2020-01-02T00:00:00Z",
            "end": "2020-01-05T00:00:00Z",
            "bbox": [350.0, -5.0, 20.0, 5.0],
            "depth": [0.0, 15.0],
            "chunk_target_mib": 0.001,
            "retry_delay_seconds": 0,
            "output": str(self.root / ("points.nc" if points else "subset.nc")),
        }
        if points:
            request["bbox"] = [0.0, -5.0, 20.0, 5.0]
            request["points"] = [{"name": "A", "lon": 10.0, "lat": 0.0}]
        return request

    def plan(self, request: dict, run_dir: Path | None = None) -> dict:
        return build_hycom_plan(
            request,
            run_dir=run_dir or self.root / "run",
            probe_override={
                "method": "test",
                "conservative_bytes_per_second": 1_000_000.0,
                "median_request_seconds": 0.01,
            },
            free_bytes_override=10**9,
        )

    def test_inventory_arbitrary_variables_and_coordinates(self) -> None:
        inventory = inventory_hycom(str(self.source))
        self.assertIn("mixed_layer_thickness", inventory["variables"])
        self.assertEqual(inventory["coordinate_roles"]["longitude"], "lon")
        self.assertEqual(inventory["coordinates"]["depth"]["maximum"], 15.0)
        self.assertEqual(
            inventory["variable_coordinate_roles"]["mixed_layer_thickness"]["time_dimension"],
            "time2",
        )

    def test_plan_dateline_chunking_fetch_resume_and_health(self) -> None:
        request = self.request()
        run = self.root / "run"
        plan = self.plan(request, run)
        self.assertNotIn(Path.home().name.lower(), json.dumps(plan).lower())
        self.assertGreaterEqual(len({chunk["segment"] for chunk in plan["chunks"]}), 2)
        self.assertEqual(plan["gate"]["state"], "ready")
        validate_plan(plan)
        result = fetch_hycom_plan(plan, run_dir=run, open_monitor=False)
        self.assertTrue(Path(result["output"]).exists())
        health = health_hycom(result["output"], request=request)
        self.assertTrue(health["passed"])
        status = json.loads((run / "download_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "complete")
        with xr.open_dataset(result["output"]) as output:
            self.assertEqual(output.sizes["time"], 4)
            self.assertEqual(output.sizes["time2"], 4)
            self.assertEqual(output["lon"].values.tolist(), [350.0, 0.0, 10.0, 20.0])
        second = fetch_hycom_plan(plan, run_dir=run, open_monitor=False)
        self.assertEqual(second["sha256"], result["sha256"])

    def test_generic_point_sampling(self) -> None:
        request = self.request(points=True)
        run = self.root / "point-run"
        plan = self.plan(request, run)
        result = fetch_hycom_plan(plan, run_dir=run, open_monitor=False)
        with xr.open_dataset(result["output"]) as output:
            self.assertEqual(output.sizes["point"], 1)
            self.assertEqual(str(output["point"].values[0]), "A")

    def test_gates_hashes_and_unknown_variable(self) -> None:
        self.assertEqual(_plan_gate(1, 100, 599)["state"], "ready")
        self.assertEqual(_plan_gate(1, 100, 600)["state"], "long_run_monitor_required")
        self.assertEqual(_plan_gate(1, 100, float("nan"))["state"], "blocked")
        self.assertEqual(_plan_gate(100, 400, 1)["state"], "blocked")
        plan = self.plan(self.request())
        changed = copy.deepcopy(plan)
        changed["chunks"][0]["expected_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_plan(changed)
        bad = self.request()
        bad["variables"] = ["not_a_variable"]
        with self.assertRaisesRegex(ValueError, "not found"):
            self.plan(bad)
        with self.assertRaisesRegex(Exception, "positive"):
            build_hycom_plan(
                self.request(),
                run_dir=self.root / "failed-probe",
                probe_override={"conservative_bytes_per_second": 0},
                free_bytes_override=10**9,
            )

    def test_status_html_and_terminal_state(self) -> None:
        status_path = self.root / "status" / "download_status.json"
        status = DownloadStatus(
            status_path,
            connector="hycom-fetcher",
            request_hash="abc",
            total_chunks=2,
            expected_bytes=100,
            estimate_seconds=600,
            heartbeat_seconds=60,
        )
        status.start()
        status.update(completed_chunks=1, completed_bytes=50, active_chunk="chunk-2")
        status.finish("complete", "done")
        saved = json.loads(status_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["state"], "complete")
        page = monitor_html()
        self.assertIn("download_status.json", page)
        self.assertNotIn("join('\n')", page)
        self.assertIn("127.0.0.1", __import__("inspect").getsource(__import__("download_monitor")))

    def test_loopback_server_browser_hook_and_redaction(self) -> None:
        run = self.root / "server"
        atomic_write_json(run / "download_status.json", {"state": "complete"})
        with mock.patch("download_monitor.webbrowser.open", return_value=True) as opener:
            self.assertEqual(
                serve_monitor(
                    run,
                    port=0,
                    open_browser=True,
                    terminal_grace_seconds=0,
                    max_hours=0.01,
                ),
                0,
            )
            opener.assert_called_once()
        server = json.loads((run / "monitor_server.json").read_text(encoding="utf-8"))
        self.assertEqual(server["host"], "127.0.0.1")
        redacted = safe_message("https://example.test/data?token=secret C:\\Users\\name\\private.nc")
        self.assertNotIn("secret", redacted)
        self.assertNotIn("Users", redacted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
