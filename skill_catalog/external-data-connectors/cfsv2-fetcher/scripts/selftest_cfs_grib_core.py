#!/usr/bin/env python3
"""Offline contract tests for the shared CFS atmospheric core."""

from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from cfs_grib_core import (
    CfsAtmosphericError,
    ERA_SPLIT,
    _match_field,
    _download_unit,
    _plan_gate,
    _write_canonical,
    classify_era,
    health,
    hycom_eligibility,
    ncei_inventory,
    normalize_request,
    fetch_hycom,
)
from download_monitor import DownloadStatus


class CatalogHandler(BaseHTTPRequestHandler):
    file_bytes = b"GRIB" + bytes(4092)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/root/catalog.xml":
            body = b'<catalog xmlns:xlink="http://www.w3.org/1999/xlink"><catalogRef xlink:href="200801/catalog.xml"/></catalog>'
        elif self.path == "/root/200801/catalog.xml":
            body = b'<catalog><dataset urlPath="model/200801/pressfc.gdas.200801.grb2"/><dataset urlPath="model/200801/pressfc.gdas.200801.l.grb2"/></catalog>'
        elif self.path == "/files/200801/pressfc.gdas.200801.grb2":
            start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            body = self.file_bytes[start:]
            self.send_response(206 if self.headers.get("Range") else 200)
            self.send_header("Content-Range", f"bytes {start}-{len(self.file_bytes)-1}/{len(self.file_bytes)}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802
        if self.path != "/files/200801/pressfc.gdas.200801.grb2":
            self.send_error(404); return
        self.send_response(200); self.send_header("Content-Length", str(len(self.file_bytes))); self.send_header("Accept-Ranges", "bytes"); self.end_headers()

    def log_message(self, *_: object) -> None:
        return


class SharedCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_era_gate_and_crossing(self) -> None:
        before = datetime(2011, 3, 31, 23, tzinfo=timezone.utc)
        after = ERA_SPLIT
        self.assertEqual([x[0] for x in classify_era(before, after)], ["cfsr", "cfsv2"])
        request = normalize_request({"start": "2011-04-01T00:00:00Z", "end": "2011-04-01T00:00:00Z", "products": ["wind_10m"], "bbox": [283, 38, 283.3, 38.3]}, "cfsv2")
        self.assertEqual(request["schema_version"], "cfs_atmospheric_request_v2")
        with self.assertRaises(CfsAtmosphericError):
            from cfs_grib_core import validate_in_era
            validate_in_era(request, "cfsr")

    def test_storage_and_provider_lock_eligibility(self) -> None:
        self.assertEqual(_plan_gate(100, 401, 1)["state"], "ready")
        self.assertEqual(_plan_gate(100, 400, 1)["state"], "blocked")
        self.assertTrue(hycom_eligibility("cfsv2", ["wind_10m", "surface_pressure"])["eligible"])
        self.assertFalse(hycom_eligibility("cfsv2", ["air_temperature_2m"])["eligible"])

    def test_catalog_full_resolution_selection(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), CatalogHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            request = normalize_request({
                "start": "2008-01-01T00:00:00Z", "end": "2008-01-01T00:00:00Z", "products": ["surface_pressure"], "bbox": [283, 38, 283.3, 38.3],
                "catalog_root": f"{base}/root/catalog.xml", "month_catalog_template": f"{base}/root/{{yyyymm}}/catalog.xml", "ncei_file_template": f"{base}/files/{{yyyymm}}/{{stem}}.gdas.{{yyyymm}}.grb2",
            }, "cfsr")
            result = ncei_inventory("cfsr", request)
            self.assertEqual(result["coverage"]["first_month"], "200801")
            self.assertEqual(len(result["source_units"]), 1)
            self.assertNotIn(".l.", result["source_units"][0]["url"])
        finally:
            server.shutdown(); server.server_close()

    def test_shared_http_range_resume(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), CatalogHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/files/200801/pressfc.gdas.200801.grb2"
            raw = self.root / "raw"; raw.mkdir()
            partial = raw / "pressfc.gdas.200801.grb2.part"
            partial.write_bytes(CatalogHandler.file_bytes[:123])
            request = {"max_retries": 2}
            status = DownloadStatus(self.root / "resume_status.json")
            result = _download_unit({"id": "200801:surface_pressure", "url": url, "bytes": len(CatalogHandler.file_bytes)}, raw, request, status)
            self.assertEqual(result.read_bytes(), CatalogHandler.file_bytes)
            status.finish("complete", "fixture complete")
        finally:
            server.shutdown(); server.server_close()

    def test_field_validation_and_canonical_health(self) -> None:
        meta = {"short_name": "UGRD", "name": "U-component of wind", "type_of_level": "heightAboveGround", "level": 10}
        spec = {"short": {"ugrd"}, "contains": ("u-component of wind",), "level": {"heightaboveground"}, "value": 10}
        self.assertTrue(_match_field(meta, spec))
        path = self.root / "fields.nc"
        _write_canonical(path, [datetime(2019, 7, 1, tzinfo=timezone.utc)], np.array([38.0, 38.2]), np.array([283.0, 283.2]), {"absolute_air_pressure": np.full((1, 2, 2), 101325.0)}, {"absolute_air_pressure": {"canonical_units": "Pa", "source_units": "Pa", "product_definition_template": 0}}, {"source_provider": "test", "model": "cfsv2"})
        request = normalize_request({"start": "2019-07-01T00:00:00Z", "end": "2019-07-01T00:00:00Z", "products": ["surface_pressure"], "bbox": [283, 38, 283.3, 38.3]}, "cfsv2")
        self.assertTrue(health(path, request)["passed"])

    def test_whole_request_hycom_fixture(self) -> None:
        import xarray as xr
        moment = np.datetime64("2019-07-01T00:00:00", "s")
        mt = float((moment - np.datetime64("1900-12-31T00:00:00", "s")) / np.timedelta64(1, "D"))
        fixture = self.root / "hycom.nc"
        xr.Dataset(
            {
                "wndewd": (("MT", "Latitude", "Longitude"), np.ones((1, 2, 2)), {"units": "m s-1"}),
                "wndnwd": (("MT", "Latitude", "Longitude"), -np.ones((1, 2, 2)), {"units": "m s-1"}),
            },
            coords={"MT": [mt], "Latitude": [38.0, 38.2], "Longitude": [283.0, 283.2]},
        ).to_netcdf(fixture, engine="netcdf4")
        request = normalize_request({"start": "2019-07-01T00:00:00Z", "end": "2019-07-01T00:00:00Z", "products": ["wind_10m"], "bbox": [283, 38, 283.2, 38.2], "provider": "hycom"}, "cfsv2")
        status = DownloadStatus(self.root / "status.json")
        with patch("cfs_grib_core._open_hycom", side_effect=lambda *_args, **_kwargs: (xr.open_dataset(fixture, engine="netcdf4", decode_times=False), "https://example.test/hycom.nc")):
            output = self.root / "hycom_fields.nc"
            fetch_hycom("cfsv2", {"request": request, "request_hash": "fixture"}, self.root, output, status)
        self.assertTrue(health(output, request)["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
