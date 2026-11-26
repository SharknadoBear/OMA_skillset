#!/usr/bin/env python3
"""Offline contract tests for the shared CFS atmospheric core."""

from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from io import BytesIO
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import netCDF4 as nc

from cfs_grib_core import (
    CfsAtmosphericError,
    ERA_SPLIT,
    PRODUCTS,
    _convert,
    _match_field,
    _selection_score,
    _native_field_grids,
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
    def test_cycle_boundary_prefers_nonzero_mean_and_requires_physical_units(self):
        spec=PRODUCTS["precipitation_rate"]["fields"][0]
        row=dict(step_type="avg",step_range="0",forecast_time=0)
        self.assertGreater(_selection_score(dict(row,step_range="0-6"),spec,100),_selection_score(row,spec,100))
        for product,units in (("precipitation_rate","kg m**-2 s**-1"),("upward_shortwave_surface_flux","W m**-2")):
            name=PRODUCTS[product]["fields"][0]["name"]
            values,_=_convert(np.array([2.]),dict(units=units),name)
            np.testing.assert_array_equal(values,[2.])
            with self.assertRaisesRegex(CfsAtmosphericError,"native units"):
                _convert(np.array([2.]),dict(units="J m-2"),name)

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

    def test_mean_sea_level_pressure(self) -> None:
        self.assertEqual(PRODUCTS["mean_sea_level_pressure"]["stem"], "prmsl")
        spec = PRODUCTS["mean_sea_level_pressure"]["fields"][0]
        meta = {"short_name": "msl", "name": "Mean sea level pressure", "type_of_level": "meanSea", "units": "hPa"}
        self.assertTrue(_match_field(meta, spec))
        self.assertFalse(_match_field(dict(meta, type_of_level="surface"), spec))
        self.assertFalse(_match_field(dict(meta, short_name="sp"), spec))
        actual, _ = _convert(np.array([1013.25]), meta, spec["name"])
        np.testing.assert_allclose(actual, [101325.0])
        self.assertFalse(hycom_eligibility("cfsv2", ["wind_10m", "mean_sea_level_pressure"])["eligible"])

    def test_pressure_and_wind_preserve_their_native_grids(self) -> None:
        wind = (np.array([46.0, 47.0, 48.0]), np.array([234.0, 235.0, 236.0]))
        pressure = (np.array([45.0, 49.0]), np.array([233.0, 237.0]))
        grids = dict(air_pressure_at_mean_sea_level=pressure, eastward_wind=wind, northward_wind=wind)
        main, secondary = _native_field_grids(grids)
        self.assertIs(main, wind)
        self.assertIs(secondary, pressure)
        self.assertIsNone(_native_field_grids({"air_pressure_at_mean_sea_level": pressure})[1])
        with self.assertRaisesRegex(CfsAtmosphericError, "primary native grid"):
            _native_field_grids(dict(grids, northward_wind=pressure))
        values = {name: (lat[:, None] + lon[None, :])[None] for name, (lat, lon) in grids.items()}
        metadata = {name: {"canonical_units": "Pa" if name.startswith("air_pressure") else "m/s"}
                    for name in grids}
        output = self.root / "native-grids.nc"
        _write_canonical(output, [datetime(2008, 1, 1, tzinfo=timezone.utc)], *main, values,
                         metadata, {"source_provider": "ncei"}, mslp_grid=secondary)
        self.assertCountEqual(health(output)["variables"], grids)
        self.assertTrue(health(output)["passed"])
        with nc.Dataset(output) as dataset:
            self.assertEqual(dataset["air_pressure_at_mean_sea_level"].dimensions,
                             ("time", "mslp_latitude", "mslp_longitude"))
            np.testing.assert_array_equal(dataset["mslp_longitude"][:], pressure[1])
            for name in grids:
                np.testing.assert_array_equal(dataset[name][:], values[name])

    def test_ncei_object_store_when_default_tds_unavailable(self) -> None:
        model = "cfsr"
        date = "2008-01-01T00:00:00Z"
        request = normalize_request({"start":date,"end":date,"products":["wind_10m","mean_sea_level_pressure"],
                                     "bbox":[234.0,44.0,238.0,48.0]}, model)
        header = b"GRIB" + bytes(3) + bytes([2]) + (120).to_bytes(8, "big")
        def reply(*_, **__):
            response = MagicMock()
            response.__enter__.return_value = response
            response.status_code = 206
            response.headers = {"Content-Range":"bytes 0-15/120"}
            response.raw = BytesIO(header)
            return response
        with patch("cfs_grib_core._tds_inventory", side_effect=CfsAtmosphericError("HTTP 503")), \
                patch("cfs_grib_core._request", side_effect=reply):
            inv = ncei_inventory(model, request)
            self.assertEqual(inv["transport"], "ncei_https_object_store")
            self.assertEqual(inv["coverage"]["scope"], "requested_months")
            self.assertEqual(inv["total_bytes"], 240)
            self.assertIn("/prmsl.gdas.", inv["source_units"][1]["url"])
            with self.assertRaisesRegex(CfsAtmosphericError, "503"):
                ncei_inventory(model, dict(request, catalog_root="https://example.org/catalog.xml"))
            header = b"<html>error page"
            with self.assertRaisesRegex(CfsAtmosphericError, "Invalid GRIB2"):
                ncei_inventory(model, request)

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
