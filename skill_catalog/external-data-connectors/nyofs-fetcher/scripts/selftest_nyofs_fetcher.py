#!/usr/bin/env python3
"""Offline self-tests for the NYOFS connector, POM extraction, and health gate."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import nyofs_fetcher as nf  # noqa: E402

UTC = timezone.utc


def _request(
    *,
    start: str = "2026-07-20T00:00:00Z",
    end: str = "2026-07-21T00:00:00Z",
    grid: str = "coarse",
    product: str = "fields",
    guidance: str = "nowcast",
    missing_policy: str = "error",
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "nyofs_request_v1",
        "start_utc": start,
        "end_utc_exclusive": end,
        "grid": grid,
        "product": product,
        "guidance": guidance,
        "missing_policy": missing_policy,
        "cache_policy": "keep",
        "max_workers": 2,
    }
    if product == "fields":
        value["variables"] = ["zeta", "u", "v", "air_u", "air_v"]
        value["vertical_views"] = ["surface", "near_surface", "bottom", "depth_average", 2]
    if guidance == "forecast":
        value["run_cycle_utc"] = "2026-07-20T05:00:00Z"
    return value


def _object(key: str, size: int = 100, etag: str = "etag") -> dict[str, Any]:
    parsed = nf.parse_object_key(key)
    if parsed is None:
        raise AssertionError(key)
    return {
        **parsed,
        "size": size,
        "etag": etag,
        "last_modified": "2026-07-20T06:00:00.000Z",
        "storage_class": "STANDARD",
        "url": "https://example.invalid/" + key,
    }


class FakeResponse:
    def __init__(self, content: bytes, *, status_code: int = 200, headers: dict[str, str] | None = None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        for offset in range(0, len(self.content), max(1, chunk_size)):
            yield self.content[offset : offset + chunk_size]


class PaginationSession:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, *, params: dict[str, Any], timeout: float):
        self.calls.append(dict(params))
        token = params.get("continuation-token")
        if token is None:
            xml = b"""<?xml version='1.0' encoding='UTF-8'?>
            <ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>
              <IsTruncated>true</IsTruncated><NextContinuationToken>next-token</NextContinuationToken>
              <Contents><Key>nyofs/netcdf/2026/07/20/a.nc</Key><LastModified>2026-07-20T00:00:00Z</LastModified><ETag>\"one\"</ETag><Size>7</Size><StorageClass>STANDARD</StorageClass></Contents>
            </ListBucketResult>"""
        else:
            xml = b"""<?xml version='1.0' encoding='UTF-8'?>
            <ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>
              <IsTruncated>false</IsTruncated>
              <Contents><Key>nyofs/netcdf/2026/07/20/b.nc</Key><LastModified>2026-07-20T00:00:00Z</LastModified><ETag>\"two-2\"</ETag><Size>9</Size><StorageClass>STANDARD</StorageClass></Contents>
            </ListBucketResult>"""
        return FakeResponse(xml)


class ResumeSession:
    def __init__(self, payload: bytes, etag: str, fail_once: bool = True):
        self.payload = payload
        self.etag = etag
        self.fail_once = fail_once
        self.calls: list[dict[str, str]] = []

    def get(self, url: str, *, headers: dict[str, str], stream: bool, timeout: float):
        self.calls.append(dict(headers))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("simulated interrupted transfer")
        start = int(headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        content = self.payload[start:]
        return FakeResponse(
            content,
            status_code=206 if start else 200,
            headers={"ETag": f'"{self.etag}"', "Content-Length": str(len(content))},
        )


def _write_fields(path: Path, start: datetime, *, geometry_delta: float = 0.0, reverse_sigma: bool = True) -> None:
    import netCDF4
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("nx", 5)
        ds.createDimension("ny", 4)
        ds.createDimension("sigma", 4)
        ds.createDimension("time", 6)
        ds.setncatts({"model": "POM", "grid_type": "curvilinear", "file_type": "Full_Grid"})
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = "days since 2008-01-01 00:00:00 UTC"
        time_var.standard_name = "time"
        decoded = [start + timedelta(hours=index, seconds=14 if index % 2 == 0 else -14) for index in range(6)]
        time_var[:] = netCDF4.date2num(decoded, time_var.units)
        yy, xx = np.mgrid[:4, :5]
        lon = -75.0 + xx * 0.1 + yy * 0.01
        lat = 40.0 + yy * 0.1 + xx * 0.005
        lon[1, 1] += geometry_delta
        mask = np.ones((4, 5), dtype=np.float32)
        mask[0, 0] = 0
        depth = 10 + yy + xx
        sigma = np.asarray([1.0, 0.6, 0.2, 0.0] if reverse_sigma else [0.0, 0.2, 0.6, 1.0], dtype=np.float32)
        for name, values, dims in (
            ("lon", lon, ("ny", "nx")),
            ("lat", lat, ("ny", "nx")),
            ("mask", mask, ("ny", "nx")),
            ("depth", depth, ("ny", "nx")),
            ("sigma", sigma, ("sigma",)),
        ):
            variable = ds.createVariable(name, "f4", dims)
            variable[:] = values
        ds.variables["sigma"].positive = "down"
        ds.variables["depth"].positive = "down"
        fill = np.float32(-99999.0)
        for name, dims, standard_name in (
            ("zeta", ("time", "ny", "nx"), "sea_surface_elevation"),
            ("air_u", ("time", "ny", "nx"), "eastward_wind"),
            ("air_v", ("time", "ny", "nx"), "northward_wind"),
            ("u", ("time", "sigma", "ny", "nx"), "eastward_sea_water_velocity"),
            ("v", ("time", "sigma", "ny", "nx"), "northward_sea_water_velocity"),
        ):
            variable = ds.createVariable(name, "f4", dims, fill_value=fill)
            variable.missing_value = fill
            variable.standard_name = standard_name
            variable.units = "m/s" if name != "zeta" else "m"
            shape = tuple(len(ds.dimensions[dim]) for dim in dims)
            data = np.empty(shape, dtype=np.float32)
            if len(shape) == 3:
                for t in range(6):
                    data[t] = (0.1 * t if name == "zeta" else (2.0 if name == "air_u" else -1.0))
                data[:, 0, 0] = fill
            else:
                for t in range(6):
                    for level in range(4):
                        data[t, level] = (0.2 * t + level if name == "u" else 0.5 + 0.1 * level)
                data[:, :, 0, 0] = fill
                if name == "u":
                    data[0, 1, 1, 1] = fill
            variable[:] = data


def _write_stations(path: Path) -> None:
    import netCDF4

    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("time", 61)
        ds.createDimension("station", 2)
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = "days since 2008-01-01 00:00:00 UTC"
        start = datetime(2026, 7, 19, 23, tzinfo=UTC)
        decoded = [
            start + timedelta(minutes=6 * index, seconds=14 if index % 2 == 0 else -14)
            for index in range(61)
        ]
        time_var[:] = netCDF4.date2num(decoded, time_var.units)
        lon = ds.createVariable("lon", "f4", ("station",))
        lat = ds.createVariable("lat", "f4", ("station",))
        lon[:] = [-74.0, -73.9]
        lat[:] = [40.5, 40.6]


def _make_synthetic_run(root: Path, *, geometry_drift: bool = False) -> tuple[dict[str, Any], Path]:
    request = _request(end="2026-07-20T12:00:00Z")
    request_path = root / "request.json"
    nf.write_json_atomic(request_path, request)
    objects: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for cycle, start, drift in (
        (5, datetime(2026, 7, 20, 0, tzinfo=UTC), 0.0),
        (11, datetime(2026, 7, 20, 6, tzinfo=UTC), 0.02 if geometry_drift else 0.0),
    ):
        key = f"nyofs/netcdf/2026/07/20/nyofs.t{cycle:02d}z.20260720.fields.nowcast.nc"
        source = _object(key, etag=f"synthetic-{cycle}-2")
        path = root / "cache" / "raw" / f"nyofs.t{cycle:02d}z.20260720.fields.nowcast.nc"
        _write_fields(path, start, geometry_delta=drift)
        source["size"] = path.stat().st_size
        digest = nf._sha256(path)
        source["url"] = "https://example.invalid/" + key
        objects.append(source)
        outcomes.append(
            {
                "key": key,
                "url": source["url"],
                "local_path": str(path.resolve()),
                "status": "downloaded",
                "size": source["size"],
                "etag": source["etag"],
                "sha256": digest,
                "resumed": False,
                "resumed_from_bytes": 0,
                "retry_count": 0,
                "source": source,
            }
        )
        nf.write_json_atomic(
            nf._download_sidecar(path),
            {"size": source["size"], "etag": source["etag"], "sha256": digest},
        )
    nf.write_json_atomic(
        root / "download_estimate.json",
        {
            "schema_version": "nyofs_download_estimate_v1",
            "request": nf.validate_request(request),
            "objects": objects,
            "object_count": len(objects),
            "total_bytes": sum(item["size"] for item in objects),
        },
    )
    nf.write_json_atomic(
        root / "fetch_manifest.json",
        {"schema_version": "nyofs_fetch_manifest_v1", "request": nf.validate_request(request), "outcomes": outcomes},
    )
    return request, request_path


class RequestAndCatalogTests(unittest.TestCase):
    def test_validation_defaults_and_boundaries(self):
        normalized = nf.validate_request(_request())
        self.assertEqual(normalized["grid"], "coarse")
        self.assertEqual(normalized["variables"], ["zeta", "u", "v", "air_u", "air_v"])
        invalid = _request(product="stations")
        invalid["variables"] = ["zeta"]
        with self.assertRaisesRegex(ValueError, "passthrough"):
            nf.validate_request(invalid)
        unpaired = _request()
        unpaired["variables"] = ["zeta", "u"]
        with self.assertRaisesRegex(ValueError, "together"):
            nf.validate_request(unpaired)
        nowcast = _request()
        nowcast["run_cycle_utc"] = "2026-07-20T05:00:00Z"
        with self.assertRaisesRegex(ValueError, "only"):
            nf.validate_request(nowcast)
        forecast = _request(guidance="forecast")
        forecast["run_cycle_utc"] = "2026-07-20T06:00:00Z"
        with self.assertRaisesRegex(ValueError, "05, 11, 17, or 23"):
            nf.validate_request(forecast)

    def test_current_and_actual_legacy_aggregate_names(self):
        current = nf.parse_object_key("nyofs/netcdf/2026/07/20/nyofs_fg.t11z.20260720.fields.nowcast.nc")
        self.assertEqual((current["grid"], current["run_time"], current["naming"]), ("fine", "2026-07-20T11:00:00Z", "current_aggregate"))
        legacy = nf.parse_object_key("nyofs/netcdf/202607/nos.nyofs.fields.nowcast.20260720.t11z.nc")
        self.assertEqual((legacy["grid"], legacy["guidance"], legacy["naming"]), ("coarse", "nowcast", "legacy_aggregate"))
        self.assertIsNone(nf.parse_object_key("nyofs/netcdf/202607/nos.nyofs.fields.n001.20260720.t11z.nc"))

    def test_forecast_nominal_spans_split_by_product(self):
        fields = nf.parse_object_key("nyofs/netcdf/2026/07/20/nyofs.t05z.20260720.fields.forecast.nc")
        stations = nf.parse_object_key("nyofs/netcdf/2026/07/20/nyofs.t05z.20260720.stations.forecast.nc")
        self.assertEqual(fields["expected_start_utc"], "2026-07-20T06:00:00Z")
        self.assertEqual(fields["expected_end_utc_exclusive"], "2026-07-22T12:00:00Z")
        self.assertEqual(stations["expected_start_utc"], "2026-07-20T05:00:00Z")
        self.assertEqual(stations["expected_end_utc_exclusive"], "2026-07-22T11:06:00Z")

    def test_s3_xml_pagination_and_multipart_etag(self):
        session = PaginationSession()
        objects = nf.list_s3_objects("nyofs/netcdf/", session=session, endpoint="https://example.invalid")
        self.assertEqual([item["size"] for item in objects], [7, 9])
        self.assertEqual(objects[1]["etag"], "two-2")
        self.assertEqual(session.calls[1]["continuation-token"], "next-token")

    def test_one_day_selection_deduplication_and_exact_bytes(self):
        objects: list[dict[str, Any]] = []
        for grid_name, model, size in (("coarse", "nyofs", 10), ("fine", "nyofs_fg", 20)):
            for cycle in (5, 11, 17, 23):
                current = _object(f"nyofs/netcdf/2026/07/20/{model}.t{cycle:02d}z.20260720.fields.nowcast.nc", size=size)
                legacy = _object(f"nyofs/netcdf/202607/nos.{model}.fields.nowcast.20260720.t{cycle:02d}z.nc", size=999)
                objects.extend([legacy, current])
        request = nf.validate_request(_request(grid="both"))
        selection = nf.select_objects(request, objects)
        self.assertEqual(len(selection["selected"]), 8)
        self.assertEqual(len(selection["duplicate_objects"]), 8)
        self.assertTrue(all(item["naming"] == "current_aggregate" for item in selection["selected"]))
        self.assertEqual(selection["duplicate_times"], [])
        self.assertEqual(selection["nominal_time_count"], 48)
        self.assertEqual(selection["nominal_time_count_by_grid"], {"coarse": 24, "fine": 24})
        with tempfile.TemporaryDirectory() as folder:
            report = nf.plan_request(request, folder, objects=objects)
        self.assertEqual(report["total_bytes"], 4 * 10 + 4 * 20)
        self.assertEqual(report["missing_times"], [])
        self.assertEqual(report["duplicate_times"], [])
        self.assertEqual(report["nominal_time_count"], 48)

    def test_missing_cycle_fails_under_error_and_records_under_skip(self):
        objects = [
            _object(f"nyofs/netcdf/2026/07/20/nyofs.t{cycle:02d}z.20260720.fields.nowcast.nc")
            for cycle in (5, 11, 17)
        ]
        with self.assertRaisesRegex(RuntimeError, "missing"):
            nf.select_objects(nf.validate_request(_request()), objects)
        skipped = nf.select_objects(nf.validate_request(_request(missing_policy="skip")), objects)
        self.assertEqual(len(skipped["missing_times"]), 6)


class DownloadTests(unittest.TestCase):
    def test_resume_retry_multipart_etag_and_cache_hit(self):
        payload = b"abcdefghijklmnopqrstuvwxyz"
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "object.nc"
            partial = destination.with_name(destination.name + ".part")
            partial.write_bytes(payload[:7])
            item = {
                "key": "nyofs/netcdf/test/object.nc",
                "url": "https://example.invalid/object.nc",
                "size": len(payload),
                "etag": "opaque-multipart-3",
                "last_modified": "2026-01-01T00:00:00Z",
            }
            nf.write_json_atomic(
                nf._partial_sidecar(destination),
                {
                    "schema_version": "nyofs_partial_object_v1",
                    "key": item["key"],
                    "url": item["url"],
                    "size": item["size"],
                    "etag": item["etag"],
                },
            )
            session = ResumeSession(payload, item["etag"], fail_once=True)
            result = nf.download_object(item, destination, session=session, max_attempts=2, chunk_size=5)
            self.assertEqual(result["status"], "downloaded")
            self.assertTrue(result["resumed"])
            self.assertEqual(result["resumed_from_bytes"], 7)
            self.assertEqual(result["retry_count"], 1)
            self.assertEqual(destination.read_bytes(), payload)
            sidecar = json.loads(nf._download_sidecar(destination).read_text())
            self.assertTrue(sidecar["etag_is_multipart"])
            cached = nf.download_object(item, destination, session=ResumeSession(payload, item["etag"], fail_once=False))
            self.assertEqual(cached["status"], "cache_hit")

    def test_stale_partial_provenance_is_not_resumed(self):
        payload = b"new-payload"
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "object.nc"
            destination.with_name(destination.name + ".part").write_bytes(b"old")
            nf.write_json_atomic(
                nf._partial_sidecar(destination),
                {"key": "old-key", "url": "https://old.invalid", "size": 3, "etag": "old"},
            )
            item = {
                "key": "nyofs/netcdf/test/new.nc",
                "url": "https://example.invalid/new.nc",
                "size": len(payload),
                "etag": "new-etag",
                "last_modified": "2026-01-01T00:00:00Z",
            }
            session = ResumeSession(payload, item["etag"], fail_once=False)
            result = nf.download_object(item, destination, session=session, max_attempts=1)
            self.assertFalse(result["resumed"])
            self.assertEqual(session.calls, [{}])
            self.assertEqual(destination.read_bytes(), payload)


class POMExtractionTests(unittest.TestCase):
    def test_reversed_sigma_and_missing_layer_weighted_answers(self):
        import numpy as np

        sigma = np.asarray([1.0, 0.6, 0.2, 0.0])
        weights = nf.sigma_trapezoid_weights(sigma)
        np.testing.assert_allclose(weights, [0.2, 0.4, 0.3, 0.1], atol=1e-12)
        values = np.asarray([[[[0.0]], [[np.nan]], [[2.0]], [[3.0]]]])
        result = nf.weighted_vertical_average(values, weights, np.asarray([[True]]), axis=1)
        expected = (0.2 * 0.0 + 0.3 * 2.0 + 0.1 * 3.0) / (0.2 + 0.3 + 0.1)
        self.assertAlmostEqual(float(result[0, 0, 0]), expected)

    def test_synthetic_extract_and_full_health_gate(self):
        import netCDF4
        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            request, request_path = _make_synthetic_run(root)
            extraction = nf.extract_request(request_path, root)
            self.assertEqual(len(extraction["outputs"]), 1)
            output = Path(extraction["outputs"][0]["path"])
            self.assertTrue(output.is_file())
            with netCDF4.Dataset(output) as ds:
                self.assertEqual(ds.schema_version, "nyofs_compact_fields_v1")
                self.assertEqual(len(ds.dimensions["time"]), 12)
                self.assertEqual(ds.variables["lon"].dimensions, ("y", "x"))
                for name in (
                    "u_surface",
                    "u_near_surface",
                    "u_bottom",
                    "u_depth_average",
                    "u_sigma_2",
                    "current_speed_depth_average",
                    "wind_speed",
                ):
                    self.assertIn(name, ds.variables)
                times = nf._compact_times(ds)
                self.assertEqual(times[0], datetime(2026, 7, 20, 0, tzinfo=UTC))
                self.assertEqual(times[-1], datetime(2026, 7, 20, 11, tzinfo=UTC))
                speed = np.ma.asarray(ds.variables["current_speed_surface"][:]).filled(np.nan)
                u = np.ma.asarray(ds.variables["u_surface"][:]).filled(np.nan)
                v = np.ma.asarray(ds.variables["v_surface"][:]).filled(np.nan)
                np.testing.assert_allclose(speed, np.hypot(u, v), equal_nan=True)
            health = nf.evaluate_health(request_path, root)
            self.assertEqual(health["status"], "pass", health["critical_findings"])
            self.assertEqual(health["critical_findings"], [])

    def test_geometry_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, request_path = _make_synthetic_run(root, geometry_drift=True)
            with self.assertRaisesRegex(RuntimeError, "geometry drift"):
                nf.extract_request(request_path, root)

    def test_stations_reject_extraction(self):
        request = nf.validate_request(_request(product="stations"))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "passthrough"):
                nf.extract_request(request, folder)

    def test_station_passthrough_health_and_six_minute_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            request = _request(
                start="2026-07-20T00:00:00Z",
                end="2026-07-20T01:00:00Z",
                product="stations",
            )
            request_path = root / "request.json"
            nf.write_json_atomic(request_path, request)
            key = "nyofs/netcdf/2026/07/20/nyofs.t05z.20260720.stations.nowcast.nc"
            source = _object(key, etag="station-etag")
            path = root / "cache" / "raw" / Path(key).name
            _write_stations(path)
            source["size"] = path.stat().st_size
            digest = nf._sha256(path)
            outcome = {
                "key": key,
                "url": source["url"],
                "local_path": str(path.resolve()),
                "status": "downloaded",
                "size": source["size"],
                "etag": source["etag"],
                "sha256": digest,
                "source": source,
            }
            nf.write_json_atomic(
                root / "download_estimate.json",
                {"schema_version": "nyofs_download_estimate_v1", "objects": [source]},
            )
            nf.write_json_atomic(root / "fetch_manifest.json", {"outcomes": [outcome]})
            nf.write_json_atomic(
                nf._download_sidecar(path),
                {"size": source["size"], "etag": source["etag"], "sha256": digest},
            )
            health = nf.evaluate_health(request_path, root)
            self.assertEqual(health["status"], "pass", health["critical_findings"])
            grid = health["raw_source_consistency"]["coarse"]
            self.assertEqual(grid["unique_requested_time_count"], 10)
            self.assertEqual(grid["expected_time_count"], 10)
            self.assertEqual(grid["missing_times"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
