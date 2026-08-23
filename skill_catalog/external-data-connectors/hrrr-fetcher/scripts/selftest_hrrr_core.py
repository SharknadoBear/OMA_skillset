#!/usr/bin/env python3
"""Offline contract, transfer, provider-lock, grid, rotation, and schema tests."""

from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import requests

from download_monitor import DownloadStatus, atomic_write_json
from hrrr_core import (
    DOMAIN_INFO,
    HrrrError,
    SCHEMA_MANIFEST,
    _bbox_subset,
    _coalesce_messages,
    _download_object,
    _download_range,
    _expand_products,
    _requirements,
    _rotate_winds,
    _valid_grib,
    _write_output,
    build_inventory,
    build_plan,
    hash_payload,
    health_run,
    hrrr_version,
    normalize_request,
    object_key,
    parse_idx,
    parse_period,
    parse_utc,
    select_idx,
)
from hrrr_fetcher import main


def fake_grib(marker: int, size: int = 80) -> bytes:
    if size < 20:
        raise ValueError(size)
    return b"GRIB" + bytes([marker, 0, 0, 2]) + size.to_bytes(8, "big") + bytes([marker]) * (size - 20) + b"7777"


class RangeFixture(BaseHTTPRequestHandler):
    first = fake_grib(11, 80)
    second = fake_grib(22, 92)
    payload = first + second
    idx = (
        f"1:0:d=2024011512:UGRD:10 m above ground:anl:\n"
        f"2:{len(first)}:d=2024011512:VGRD:10 m above ground:anl:\n"
    ).encode()
    fail_aws_full = True

    def do_GET(self) -> None:  # noqa: N802
        provider = self.path.strip("/").split("/", 1)[0]
        if self.path.endswith(".idx"):
            if provider == "missing":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.idx)))
            self.end_headers()
            self.wfile.write(self.idx)
            return
        range_header = self.headers.get("Range")
        if provider == "ignored":
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
            return
        if not range_header:
            self.send_error(400)
            return
        start_text, end_text = range_header.split("=", 1)[1].split("-", 1)
        start = int(start_text)
        end = int(end_text) if end_text else len(self.payload) - 1
        if provider == "aws" and self.fail_aws_full and (start, end) != (0, 0):
            self.send_error(503)
            return
        body = self.payload[start : end + 1]
        if provider == "truncated" and len(body) > 2:
            body = body[:-2]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.payload)}")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", '"fixture"')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        return


class HrrrCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RangeFixture)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()

    @staticmethod
    def analysis(products: list[object] | None = None, **extra: object) -> dict:
        payload = {
            "schema_version": "hrrr_request_v1",
            "domain": "conus",
            "mode": "analysis",
            "start": "2024-01-15T12:00:00Z",
            "end": "2024-01-15T12:00:00Z",
            "products": products or ["wind_10m"],
            "bbox": [-77.2, 36.8, -74.5, 39.8],
            "provider_order": ["aws", "gcp"],
            "max_retries": 0,
        }
        payload.update(extra)
        return payload

    def test_archive_cadence_version_and_filenames(self) -> None:
        self.assertEqual(hrrr_version(parse_utc("2014-07-30T18:00:00Z")), "pre-operational")
        self.assertEqual(hrrr_version(parse_utc("2014-09-30T00:00:00Z")), "v1")
        self.assertEqual(hrrr_version(parse_utc("2016-08-23T00:00:00Z")), "v2")
        self.assertEqual(hrrr_version(parse_utc("2018-07-12T00:00:00Z")), "v3")
        self.assertEqual(hrrr_version(parse_utc("2020-12-02T00:00:00Z")), "v4")
        with self.assertRaisesRegex(ValueError, "archive begins"):
            normalize_request({**self.analysis(), "start": "2014-07-30T17:00:00Z", "end": "2014-07-30T17:00:00Z"})
        with self.assertRaisesRegex(ValueError, "Alaska analysis cycles"):
            normalize_request({**self.analysis(), "domain": "alaska", "start": "2024-01-15T13:00:00Z", "end": "2024-01-15T13:00:00Z"})
        with self.assertRaisesRegex(ValueError, "exact UTC hours"):
            normalize_request({"domain": "conus", "mode": "forecast", "cycle_start": "2024-01-15T12:30:00Z", "forecast_periods": ["PT1H"], "products": ["surface_pressure"]})
        cycle = datetime(2024, 1, 15, 12, tzinfo=timezone.utc)
        self.assertEqual(object_key("conus", cycle, "wrfsfc", 0), "hrrr.20240115/conus/hrrr.t12z.wrfsfcf00.grib2")
        self.assertEqual(object_key("alaska", cycle, "wrfnat", 3), "hrrr.20240115/alaska/hrrr.t12z.wrfnatf03.ak.grib2")
        self.assertEqual((DOMAIN_INFO["conus"]["nx"], DOMAIN_INFO["conus"]["ny"]), (1799, 1059))
        self.assertEqual((DOMAIN_INFO["alaska"]["nx"], DOMAIN_INFO["alaska"]["ny"]), (1299, 919))

    def test_forecast_periods_subhourly_alias_and_limits(self) -> None:
        request = normalize_request({
            "domain": "conus",
            "mode": "forecast",
            "cycle_start": "2024-01-15T12:00:00Z",
            "forecast_periods": ["PT15M", "PT30M", "PT45M", "PT1H"],
            "products": [{"alias": "wind_10m", "family": "wrfsubh"}],
        })
        requirements = _requirements(request)
        self.assertEqual({row["lead_hour"] for row in requirements}, {1})
        self.assertEqual(len(requirements[0]["targets"]), 8)
        self.assertEqual(parse_period("PT18H"), 1080)
        with self.assertRaisesRegex(ValueError, "through PT18H"):
            _requirements(normalize_request({**request, "forecast_periods": ["PT18H15M"]}))
        with self.assertRaisesRegex(ValueError, "No source objects"):
            _requirements(normalize_request({**request, "forecast_periods": ["PT15M"], "products": ["wind_10m"]}))
        with self.assertRaisesRegex(ValueError, "forecast-only"):
            _requirements(normalize_request(self.analysis([{"alias": "wind_10m", "family": "wrfsubh"}])))
        with self.assertRaisesRegex(ValueError, "require vector_group"):
            _expand_products([{"family": "wrfprs", "short_name": "UGRD", "level_text": "850 mb"}])
        with self.assertRaisesRegex(ValueError, "Duplicate source selector"):
            _expand_products(["surface_pressure", "surface_pressure"])
        paired = _expand_products([
            {"family": "wrfprs", "short_name": "UGRD", "level_text": "850 mb", "output_name": "u_wind", "vector_group": "pressure_wind"},
            {"family": "wrfprs", "short_name": "VGRD", "level_text": "850 mb", "output_name": "v_wind", "vector_group": "pressure_wind"},
        ])
        self.assertFalse(paired[0]["rotate_to_earth"])

    def test_idx_line_endings_selection_ambiguity_and_coalescing(self) -> None:
        text = RangeFixture.idx.decode().replace("\n", "\r\n")
        rows = parse_idx(text, len(RangeFixture.payload))
        targets = _expand_products(["wind_10m"])
        for target in targets:
            target["forecast_period_minutes"] = 0
            target["expected_step"] = "anl"
        u = select_idx(rows, targets[0])
        v = select_idx(rows, targets[1])
        self.assertEqual(len(_coalesce_messages([u, v])), 1)
        with self.assertRaisesRegex(HrrrError, "Ambiguous"):
            select_idx([rows[0], {**rows[0], "message_number": 3}], targets[0])
        with self.assertRaisesRegex(HrrrError, "strictly increasing"):
            parse_idx("1:0:d=x:TMP:surface:anl:\r2:0:d=x:TMP:surface:anl:\r", 10)

    def test_bbox_and_dateline(self) -> None:
        latitude = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=float)
        longitude = np.asarray([[179, -179, -170], [179, -179, -170]], dtype=float)
        ys, xs, mask = _bbox_subset(latitude, longitude, [178, -1, -178, 2], 0)
        self.assertEqual((ys.start, ys.stop, xs.start, xs.stop), (0, 2, 0, 2))
        self.assertTrue(mask.all())
        request = normalize_request({**self.analysis(), "bbox": [0, -90, 360, 90], "longitude_convention": "0_360"})
        self.assertEqual(request["bbox"], [-180.0, -90.0, 180.0, 90.0])

    def test_range_resume_ignored_and_truncated(self) -> None:
        request = normalize_request(self.analysis(provider_order=["aws"], max_retries=0))
        status = DownloadStatus(self.root / "range_status.json", request_hash="fixture", total_chunks=1, expected_bytes=80)
        destination = self.root / "resume.grib2"
        destination.with_suffix(".grib2.part").write_bytes(RangeFixture.first[:13])
        with requests.Session() as session:
            result = _download_range(session, f"{self.base}/gcp/object", 0, len(RangeFixture.first) - 1, destination, request, status)
            self.assertTrue(_valid_grib(destination, len(RangeFixture.first)))
            self.assertEqual(result["sha256"], __import__("hashlib").sha256(RangeFixture.first).hexdigest())
            with self.assertRaisesRegex(HrrrError, "Invalid ranged response"):
                _download_range(session, f"{self.base}/ignored/object", 0, 79, self.root / "ignored.grib2", request, status)
            with self.assertRaisesRegex(HrrrError, "has .* expected"):
                _download_range(session, f"{self.base}/truncated/object", 0, 79, self.root / "truncated.grib2", request, status)

    def test_inventory_storage_provider_switch_and_object_lock(self) -> None:
        provider_roots = {
            "aws": f"{self.base}/aws",
            "gcp": f"{self.base}/gcp",
            "azure": f"{self.base}/missing",
            "nomads": f"{self.base}/missing",
        }
        with patch("hrrr_core.PROVIDERS", provider_roots):
            inventory = build_inventory(self.analysis())
            self.assertEqual(inventory["objects"][0]["provider_lock"], "aws")
            blocked = build_plan(self.analysis(), self.root, free_bytes_override=1)
            self.assertEqual(blocked["gate"]["state"], "blocked")
            ready = build_plan(self.analysis(), self.root, free_bytes_override=10**12)
            self.assertEqual(ready["gate"]["state"], "ready")
            status = DownloadStatus(
                self.root / "object_status.json",
                request_hash=inventory["request_hash"],
                total_chunks=2,
                expected_bytes=len(RangeFixture.payload),
            )
            with requests.Session() as session:
                state, messages, attempts = _download_object(
                    session, inventory["objects"][0], inventory["request"], self.root / "raw", status
                )
            self.assertEqual(state["provider"], "gcp")
            self.assertEqual([row["state"] for row in attempts], ["failed", "complete"])
            self.assertEqual(len(messages), 2)
            self.assertTrue(all(_valid_grib(Path(row["path"]), row["bytes"]) for row in messages))
            self.assertTrue((Path(messages[0]["path"]).parent / "source.idx").exists())
            with requests.Session() as session:
                resumed, _, resumed_attempts = _download_object(
                    session, inventory["objects"][0], inventory["request"], self.root / "raw", status
                )
            self.assertEqual(resumed["provider"], "gcp")
            self.assertEqual(resumed_attempts[0]["state"], "resumed_complete")

    def test_vector_rotation_and_analysis_schema_health(self) -> None:
        scratch_u = self.root / "u.npy"
        scratch_v = self.root / "v.npy"
        np.save(scratch_u, np.ones((2, 3), dtype=np.float32))
        np.save(scratch_v, np.zeros((2, 3), dtype=np.float32))
        metadata = {"uv_relative_to_grid": True, "units": "m s-1", "name": "wind", "valid_time": "2024-01-15T12:00:00Z"}
        records = [
            {"cadence_group": "hourly", "cycle": "2024-01-15T12:00:00Z", "forecast_period_minutes": 0, "target": {"output_name": "eastward_wind_10m", "vector_group": "wind", "component": "u", "rotate_to_earth": True}, "metadata": dict(metadata), "vertical_dimension": None, "vertical_value": None, "scratch_path": str(scratch_u)},
            {"cadence_group": "hourly", "cycle": "2024-01-15T12:00:00Z", "forecast_period_minutes": 0, "target": {"output_name": "northward_wind_10m", "vector_group": "wind", "component": "v", "rotate_to_earth": True}, "metadata": dict(metadata), "vertical_dimension": None, "vertical_value": None, "scratch_path": str(scratch_v)},
        ]
        grid = {
            "x": np.arange(3.0), "y": np.arange(2.0),
            "latitude": np.ones((2, 3)), "longitude": np.ones((2, 3)), "bbox_mask": np.ones((2, 3), dtype=bool),
            "basis_ex": np.zeros((2, 3)), "basis_nx": np.ones((2, 3)), "basis_ey": -np.ones((2, 3)), "basis_ny": np.zeros((2, 3)),
            "crs_wkt": "LOCAL_CS[\"fixture\"]",
        }
        _rotate_winds(records, grid)
        self.assertTrue(np.allclose(np.load(scratch_u), 0))
        self.assertTrue(np.allclose(np.load(scratch_v), 1))
        request = normalize_request(self.analysis())
        output = _write_output(self.root / "hrrr_fields.nc", request, hash_payload(request), records, grid, [{"provider": "fixture"}])
        manifest = {
            "schema_version": SCHEMA_MANIFEST,
            "request_hash": hash_payload(request),
            "plan_hash": "fixture",
            "created_utc": "2024-01-15T12:00:00Z",
            "domain": "conus",
            "mode": "analysis",
            "provider_locks": [{"provider": "fixture"}],
            "provider_attempts": [],
            "raw_messages": [],
            "outputs": [output],
            "records": [
                {
                    "output_name": row["target"]["output_name"],
                    "cycle": row["cycle"],
                    "forecast_period_minutes": row["forecast_period_minutes"],
                    "cadence_group": row["cadence_group"],
                }
                for row in records
            ],
        }
        manifest["manifest_hash"] = hash_payload(manifest)
        atomic_write_json(self.root / "run_manifest.json", manifest)
        report = health_run(self.root)
        self.assertTrue(report["passed"], report["issues"])

    def test_forecast_schema_and_valid_time_health(self) -> None:
        import netCDF4

        request = normalize_request({
            "domain": "conus",
            "mode": "forecast",
            "cycle_start": "2024-01-15T12:00:00Z",
            "forecast_periods": ["PT1H"],
            "products": ["surface_pressure"],
        })
        scratch = self.root / "pressure.npy"
        np.save(scratch, np.full((2, 3), 101325.0, dtype=np.float32))
        records = [{
            "cadence_group": "hourly",
            "cycle": "2024-01-15T12:00:00Z",
            "forecast_period_minutes": 60,
            "target": {"output_name": "surface_air_pressure"},
            "metadata": {"units": "Pa", "name": "Surface pressure"},
            "vertical_dimension": None,
            "vertical_value": None,
            "scratch_path": str(scratch),
        }]
        grid = {
            "x": np.arange(3.0), "y": np.arange(2.0),
            "latitude": np.ones((2, 3)), "longitude": np.ones((2, 3)), "bbox_mask": np.ones((2, 3), dtype=bool),
            "crs_wkt": "LOCAL_CS[\"fixture\"]", "x_slice": [0, 3], "y_slice": [0, 2],
        }
        output = _write_output(self.root / "hrrr_fields.nc", request, hash_payload(request), records, grid, [{"provider": "fixture"}])
        with netCDF4.Dataset(output["path"]) as dataset:
            self.assertEqual(dataset.variables["valid_time"].shape, (1, 1))
            self.assertEqual(int(dataset.variables["forecast_period"][0]), 60)
        manifest = {
            "schema_version": SCHEMA_MANIFEST,
            "request_hash": hash_payload(request), "plan_hash": "fixture", "created_utc": "2024-01-15T12:00:00Z",
            "domain": "conus", "mode": "forecast", "provider_locks": [], "provider_attempts": [], "gaps": [],
            "raw_messages": [], "outputs": [output],
            "records": [{"output_name": "surface_air_pressure", "cycle": "2024-01-15T12:00:00Z", "forecast_period_minutes": 60, "cadence_group": "hourly"}],
        }
        manifest["manifest_hash"] = hash_payload(manifest)
        atomic_write_json(self.root / "run_manifest.json", manifest)
        self.assertTrue(health_run(self.root)["passed"])

    def test_cli_products_and_machine_error(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(["products"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["schema_version"], "hrrr_product_catalog_v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
