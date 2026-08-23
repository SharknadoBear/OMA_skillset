#!/usr/bin/env python3
"""Offline tests for CFSR request, gate, resume, and health contracts."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np

from cfsr_fetcher import (
    DownloadStatus,
    _atomic_canonical,
    _download_ncei_unit,
    _plan_gate,
    health,
    month_keys,
    normalize_request,
    parse_utc,
)


class RangeHandler(BaseHTTPRequestHandler):
    payload = bytes((index % 251 for index in range(1024 * 1024 + 17)))

    def do_GET(self) -> None:  # noqa: N802
        start = 0
        if self.headers.get("Range"):
            start = int(self.headers["Range"].split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(self.payload)-1}/{len(self.payload)}")
        else:
            self.send_response(200)
        data = self.payload[start:]
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_: object) -> None:
        return


class CfsrFetcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def request(self) -> dict:
        return normalize_request({
            "start":"2008-01-01T00:00:00Z", "end":"2009-01-01T00:00:00Z",
            "product":"surface_pressure", "bbox":[279.1,31.6,296.1,45.8], "provider":"auto",
        })

    def test_request_months_and_gate(self) -> None:
        request = self.request()
        self.assertEqual(month_keys(parse_utc(request["start"]), parse_utc(request["end"])), [f"2008{month:02d}" for month in range(1,13)] + ["200901"])
        self.assertEqual(_plan_gate(100, 401, 1)["state"], "ready")
        self.assertEqual(_plan_gate(100, 400, 1)["state"], "blocked")
        self.assertEqual(_plan_gate(100, 1000, 600)["state"], "long_run_monitor_required")
        with self.assertRaisesRegex(ValueError, "prmsl|surface_pressure"):
            normalize_request({**request, "product":"prmsl"})

    def test_range_resume(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            raw = self.root / "raw"
            raw.mkdir()
            partial = raw / "pressfc.gdas.test.grb2.part"
            partial.write_bytes(RangeHandler.payload[:12345])
            status = DownloadStatus(self.root / "status.json", provider="ncei", expected_bytes=len(RangeHandler.payload), completed_bytes=0)
            request = self.request()
            request.update({"max_retries":2,"retry_delay_seconds":0.0,"backoff":1.0})
            result = _download_ncei_unit({"id":"test","url":f"http://127.0.0.1:{server.server_port}/file","bytes":len(RangeHandler.payload)}, raw, status, 0, request)
            self.assertEqual(result.read_bytes(), RangeHandler.payload)
        finally:
            server.shutdown()
            server.server_close()

    def test_canonical_health(self) -> None:
        path = self.root / "pressure.nc"
        times = np.arange(0, 4 * 3600, 3600, dtype=np.int64) + int(parse_utc("2008-01-01T00:00:00Z").timestamp())
        lat = np.array([30.0, 40.0, 50.0])
        lon = np.array([278.0, 288.0, 298.0])
        pressure = np.empty((4,3,3), dtype=np.float32)
        for index in range(4):
            pressure[index] = 100000 + index + np.arange(9).reshape(3,3)
        _atomic_canonical(path, times, lat, lon, pressure, {"source_provider":"test"})
        request = normalize_request({"start":"2008-01-01T00:00:00Z","end":"2008-01-01T03:00:00Z","bbox":[279.0,31.0,296.0,45.0]})
        report = health(path, request)
        self.assertTrue(report["passed"])
        self.assertEqual(report["dimensions"]["time"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
