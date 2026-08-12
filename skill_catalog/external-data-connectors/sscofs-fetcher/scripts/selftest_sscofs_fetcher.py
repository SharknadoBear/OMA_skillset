#!/usr/bin/env python3
"""Offline regression tests for the SSCOFS fetcher and health checker.

No NOAA object is downloaded.  S3 listing/download behavior is exercised with
in-memory HTTP responses, and extraction is tested with a tiny FVCOM NetCDF
whose sigma interfaces run from bottom to surface.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_download_health as health  # noqa: E402
import estimate_data_request as estimator  # noqa: E402
import sscofs_fetcher as core  # noqa: E402


UTC = timezone.utc


def _get(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _iso_value(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    text = str(value)
    return text.replace("+00:00", "Z")


def _call(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Pass only supported keyword arguments across small API refinements."""
    signature = inspect.signature(function)
    accepts_kwargs = any(item.kind == item.VAR_KEYWORD for item in signature.parameters.values())
    selected = kwargs if accepts_kwargs else {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(*args, **selected)


def _normalize_request(mapping: dict[str, Any]) -> dict[str, Any]:
    result = core.validate_request(mapping)
    return mapping if result is None else result


def _selection_objects(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        for child in value:
            if isinstance(child, list):
                return child
    if isinstance(value, Mapping):
        for key in ("objects", "selected_objects", "selected", "files"):
            child = value.get(key)
            if isinstance(child, list):
                return child
    child = _get(value, "objects", "selected_objects", "selected", default=None)
    if isinstance(child, list):
        return child
    raise AssertionError(f"Could not locate selected objects in {type(value).__name__}")


def _recursive_values(value: Any, key_names: set[str]) -> list[Any]:
    result: list[Any] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in key_names:
                result.append(child)
            result.extend(_recursive_values(child, key_names))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.extend(_recursive_values(child, key_names))
    elif hasattr(value, "__dict__"):
        result.extend(_recursive_values(vars(value), key_names))
    return result


def _base_request(start: str, end: str, **updates: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "schema_version": "sscofs_request_v1",
        "start_utc": start,
        "end_utc_exclusive": end,
        "product": "fields",
        "guidance": "nowcast",
        "variables": ["salinity", "u", "v"],
        "vertical_views": ["surface", "bottom", "depth_average"],
        "missing_policy": "error",
        "cache_policy": "keep",
        "max_workers": 4,
    }
    if "source_policy" in updates:
        request["schema_version"] = "sscofs_request_v2"
    request.update(updates)
    return _normalize_request(request)


def _s3_xml(entries: list[tuple[str, int, str]], truncated: bool = False, token: str | None = None) -> bytes:
    contents = "".join(
        "<Contents>"
        f"<Key>{key}</Key><LastModified>2026-07-20T00:00:00.000Z</LastModified>"
        f"<ETag>&quot;{etag}&quot;</ETag><Size>{size}</Size><StorageClass>STANDARD</StorageClass>"
        "</Contents>"
        for key, size, etag in entries
    )
    next_token = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<IsTruncated>{str(truncated).lower()}</IsTruncated>{contents}{next_token}"
        "</ListBucketResult>"
    ).encode("utf-8")


class FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200, headers: dict[str, str] | None = None):
        self.content = body
        self.text = body.decode("utf-8", errors="replace")
        self.status_code = status
        self.headers = headers or {"Content-Length": str(len(body))}
        self.reason = "OK" if status < 400 else "ERROR"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 8192):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def close(self) -> None:
        return None


class PagingSession:
    def __init__(self, pages: list[bytes]):
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        index = len(self.calls) - 1
        return FakeResponse(self.pages[index])


class DownloadSession:
    def __init__(self, payload: bytes, etag: str, *, forbid_network: bool = False):
        self.payload = payload
        self.etag = etag
        self.forbid_network = forbid_network
        self.get_calls: list[dict[str, Any]] = []
        self.head_calls: list[dict[str, Any]] = []

    def head(self, url: str, **kwargs: Any) -> FakeResponse:
        if self.forbid_network:
            raise AssertionError("validated cache hit unexpectedly used the network")
        self.head_calls.append({"url": url, **kwargs})
        return FakeResponse(b"", headers={"Content-Length": str(len(self.payload)), "ETag": f'"{self.etag}"'})

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if self.forbid_network:
            raise AssertionError("validated cache hit unexpectedly used the network")
        self.get_calls.append({"url": url, **kwargs})
        headers = kwargs.get("headers") or {}
        range_header = headers.get("Range") or headers.get("range")
        start = int(range_header.split("=")[1].split("-")[0]) if range_header else 0
        body = self.payload[start:]
        response_headers = {
            "Content-Length": str(len(body)),
            "ETag": f'"{self.etag}"',
        }
        if start:
            response_headers["Content-Range"] = f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}"
        return FakeResponse(body, status=206 if start else 200, headers=response_headers)

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        return self.head(url, **kwargs) if method.upper() == "HEAD" else self.get(url, **kwargs)


def _write_fixture(path: Path) -> dict[str, float]:
    import numpy as np
    from netCDF4 import Dataset, date2num

    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("DateStrLen", 20)
        ds.createDimension("node", 4)
        ds.createDimension("nele", 2)
        ds.createDimension("three", 3)
        ds.createDimension("siglay", 3)
        ds.createDimension("siglev", 4)
        ds.createVariable("lon", "f8", ("node",))[:] = [0.0, 1.0, 1.0, 0.0]
        ds.createVariable("lat", "f8", ("node",))[:] = [0.0, 0.0, 1.0, 1.0]
        ds.createVariable("nv", "i4", ("three", "nele"))[:] = [[1, 1], [2, 3], [3, 4]]
        levels = np.array([-1.0, -0.5, -0.2, 0.0], dtype=float)[:, None]
        ds.createVariable("siglev", "f8", ("siglev", "node"))[:] = np.repeat(levels, 4, axis=1)
        layers = np.array([-0.75, -0.35, -0.10], dtype=float)[:, None]
        ds.createVariable("siglay", "f8", ("siglay", "node"))[:] = np.repeat(layers, 4, axis=1)
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = "hours since 2026-07-20 00:00:00 +00:00"
        time_var.calendar = "standard"
        time_var[:] = date2num([datetime(2026, 7, 20, tzinfo=UTC)], time_var.units, time_var.calendar)
        times = ds.createVariable("Times", "S1", ("time", "DateStrLen"))
        times[:] = np.array([list("2026-07-20T00:00:00Z")], dtype="S1")
        wet_nodes = ds.createVariable("wet_nodes", "i1", ("time", "node"))
        wet_nodes[:] = [[1, 1, 0, 1]]
        wet_cells = ds.createVariable("wet_cells", "i1", ("time", "nele"))
        wet_cells[:] = [[1, 0]]
        salinity = ds.createVariable("salinity", "f8", ("time", "siglay", "node"), fill_value=-9999.0)
        salt = np.array(
            [[10.0, 10.0, 99.0, 10.0], [20.0, np.nan, 99.0, 20.0], [30.0, 30.0, 99.0, 30.0]],
            dtype=float,
        )
        salinity[0, :, :] = salt
        u = ds.createVariable("u", "f8", ("time", "siglay", "nele"), fill_value=-9999.0)
        v = ds.createVariable("v", "f8", ("time", "siglay", "nele"), fill_value=-9999.0)
        u[0, :, :] = [[1.0, 99.0], [2.0, 99.0], [3.0, 99.0]]
        v[0, :, :] = [[2.0, 99.0], [1.0, 99.0], [0.0, 99.0]]
    return {
        "salinity_average_complete": 17.0,
        "salinity_average_missing": 11.0 / 0.7,
        "u_average": 1.7,
        "v_average": 1.3,
        "speed_average": math.hypot(1.7, 1.3),
    }


def _write_station_times(path: Path, start: datetime) -> None:
    import numpy as np
    from netCDF4 import Dataset, date2num
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as ds:
        ds.createDimension("time", 61)
        ds.createDimension("DateStrLen", 20)
        time_var = ds.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since 1970-01-01 00:00:00 +00:00"
        times = [start + timedelta(minutes=6 * index) for index in range(61)]
        time_var[:] = date2num(times, time_var.units)
        chars = ds.createVariable("Times", "S1", ("time", "DateStrLen"))
        chars[:] = np.asarray([list(item.strftime("%Y-%m-%dT%H:%M:%SZ")) for item in times], dtype="S1")


class KeyAndCatalogTests(unittest.TestCase):
    def test_health_source_summary_is_archive_specific(self) -> None:
        aws = core._decorate_object({
            **core.parse_object_key("sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.fields.n003.nc"),
            "source_id": "aws_operational", "size": 1, "size_bytes": 1,
            "etag": "aws", "last_modified": "2026-07-20T06:00:00Z",
        })
        ncei_key = core.archive_sources.get_source_descriptor("ncei_long_term", "sscofs")["root_prefix"] + \
            "2026/07/nos.sscofs.fields.n004.20260720.t03z.nc"
        ncei = core._decorate_object({
            **core.parse_object_key(ncei_key), "source_id": "ncei_long_term",
            "size": 1, "size_bytes": 1, "etag": "ncei",
            "last_modified": "2026-07-21T06:00:00Z",
        }, "ncei_long_term")
        summary = health._source_summary({"objects": [aws, ncei]}, {})
        self.assertNotIn("bucket", summary)
        self.assertEqual(set(summary["archives"]), {"aws_operational", "ncei_long_term"})
        self.assertNotEqual(summary["archives"]["aws_operational"]["container"],
                            summary["archives"]["ncei_long_term"]["container"])

    def test_estimate_hook_preserves_v1_migration_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            request = {
                "schema_version": "sscofs_request_v1", "start_utc": "2026-07-20T00:00:00Z",
                "end_utc_exclusive": "2026-07-20T01:00:00Z", "product": "fields", "guidance": "nowcast",
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            captured: dict[str, Any] = {}
            def fake_plan(value, run_dir=None):
                captured.update(value)
                return {"schema_version": "sscofs_download_estimate_v2"}
            with mock.patch.object(estimator, "plan_request", side_effect=fake_plan):
                self.assertEqual(estimator.main([
                    "--request", str(request_path), "--run-dir", str(root),
                    "--output", str(root / "estimate.json"),
                ]), 0)
            self.assertEqual(captured["schema_version"], "sscofs_request_v1")

    def test_ordered_fallback_complete_gap_and_listing_error(self) -> None:
        request = _base_request("2026-07-20T00:00:00Z", "2026-07-20T01:00:00Z")
        aws = core.parse_object_key("sscofs.t03z.20260720.fields.n003.nc")
        ncei = core.parse_object_key("nos.sscofs.fields.n003.20260720.t03z.nc")
        aws.update({"source_id": "aws_operational", "size": 5, "size_bytes": 5, "etag": "aws"})
        ncei.update({"source_id": "ncei_long_term", "size": 6, "size_bytes": 6, "etag": "ncei"})
        with mock.patch.object(core, "_discover_source", return_value=[aws]) as discover:
            objects, trace = core.discover_objects(request, with_trace=True)
        self.assertEqual([call.args[1] for call in discover.call_args_list], ["aws_operational"])
        self.assertFalse(trace["fallback_triggered"])
        self.assertEqual(objects[0]["etag"], "aws")
        calls: list[str] = []
        def gap_then_fill(req, source_id, **kwargs):
            calls.append(source_id)
            return [] if source_id == "aws_operational" else [ncei]
        with mock.patch.object(core, "_discover_source", side_effect=gap_then_fill):
            objects, trace = core.discover_objects(request, with_trace=True)
        self.assertEqual(calls, ["aws_operational", "ncei_long_term"])
        self.assertTrue(trace["fallback_triggered"])
        self.assertEqual(objects[0]["source_id"], "ncei_long_term")
        with mock.patch.object(core, "_discover_source", side_effect=RuntimeError("AWS listing failed")):
            with self.assertRaisesRegex(RuntimeError, "AWS listing failed"):
                core.discover_objects(request)

    def test_cross_archive_same_semantics_prefers_aws_despite_etag(self) -> None:
        request = _base_request("2026-07-20T00:00:00Z", "2026-07-20T01:00:00Z")
        aws = core.parse_object_key("sscofs.t03z.20260720.fields.n003.nc")
        ncei = core.parse_object_key("nos.sscofs.fields.n003.20260720.t03z.nc")
        aws.update({"source_id": "aws_operational", "size": 5, "size_bytes": 5, "etag": "aws-version"})
        ncei.update({"source_id": "ncei_long_term", "size": 7, "size_bytes": 7, "etag": "ncei-version"})
        selected = core.select_objects(request, [ncei, aws])["objects"]
        self.assertEqual(selected[0]["source_id"], "aws_operational")

    def test_aws_n000_triggers_ncei_lookup_for_preferred_n006(self) -> None:
        request = _base_request("2026-07-19T21:00:00Z", "2026-07-19T22:00:00Z")
        aws = core.parse_object_key("sscofs.t03z.20260720.fields.n000.nc")
        ncei = core.parse_object_key("nos.sscofs.fields.n006.20260719.t21z.nc")
        aws.update({"source_id": "aws_operational", "size": 5, "size_bytes": 5})
        ncei.update({"source_id": "ncei_long_term", "size": 6, "size_bytes": 6})
        def discover(req, source_id, **kwargs):
            return [aws] if source_id == "aws_operational" else [ncei]
        with mock.patch.object(core, "_discover_source", side_effect=discover):
            objects, trace = core.discover_objects(request, with_trace=True)
        self.assertTrue(trace["fallback_triggered"])
        self.assertEqual(core.select_objects(request, objects)["objects"][0]["lead_hour"], 6)

    def test_aws_n006_satisfies_scientific_precedence_without_ncei_lookup(self) -> None:
        request = _base_request("2026-07-19T21:00:00Z", "2026-07-19T22:00:00Z")
        n000 = core.parse_object_key("sscofs.t03z.20260720.fields.n000.nc")
        n006 = core.parse_object_key("sscofs.t21z.20260719.fields.n006.nc")
        for item, size in ((n000, 5), (n006, 6)):
            item.update({"source_id": "aws_operational", "size": size, "size_bytes": size})
        with mock.patch.object(core, "_discover_source", return_value=[n000, n006]) as discover:
            objects, trace = core.discover_objects(request, with_trace=True)
        self.assertEqual([call.args[1] for call in discover.call_args_list], ["aws_operational"])
        self.assertFalse(trace["fallback_triggered"])
        self.assertEqual(trace["scientific_precedence_before_fallback"], [])
        self.assertEqual(core.select_objects(request, objects)["objects"][0]["lead_hour"], 6)

    def test_out_of_window_n000_does_not_trigger_ncei_lookup(self) -> None:
        request = _base_request("2026-07-20T03:00:00Z", "2026-07-20T04:00:00Z")
        selected = core.parse_object_key("sscofs.t03z.20260720.fields.n006.nc")
        outside = core.parse_object_key("sscofs.t03z.20260720.fields.n000.nc")
        for item, size in ((selected, 5), (outside, 6)):
            item.update({"source_id": "aws_operational", "size": size, "size_bytes": size})
        with mock.patch.object(core, "_discover_source", return_value=[selected, outside]) as discover:
            objects, trace = core.discover_objects(request, with_trace=True)
        self.assertEqual([call.args[1] for call in discover.call_args_list], ["aws_operational"])
        self.assertFalse(trace["fallback_triggered"])
        self.assertEqual(trace["scientific_precedence_before_fallback"], [])
        self.assertEqual(core.select_objects(request, objects)["selected_count"], 1)

    def test_station_boundary_queries_ncei_preceding_terminal(self) -> None:
        request = core.validate_request({
            "schema_version": "sscofs_request_v2",
            "start_utc": "2026-07-20T21:00:00Z",
            "end_utc_exclusive": "2026-07-20T21:06:00Z",
            "product": "stations", "guidance": "nowcast",
        })
        following = core.parse_object_key("sscofs.t03z.20260721.stations.nowcast.nc")
        preceding = core.parse_object_key("nos.sscofs.stations.nowcast.20260720.t21z.nc")
        following.update({"source_id": "aws_operational", "size": 5, "size_bytes": 5})
        preceding.update({"source_id": "ncei_long_term", "size": 6, "size_bytes": 6})
        calls: list[str] = []
        def discover(req, source_id, **kwargs):
            calls.append(source_id)
            return [following] if source_id == "aws_operational" else [preceding]
        with mock.patch.object(core, "_discover_source", side_effect=discover):
            objects, trace = core.discover_objects(request, with_trace=True)
        self.assertEqual(calls, ["aws_operational", "ncei_long_term"])
        self.assertTrue(trace["fallback_triggered"])
        selected = core.select_objects(request, objects)["objects"]
        self.assertEqual([item["run_time_utc"] for item in selected], [
            "2026-07-20T21:00:00Z", "2026-07-21T03:00:00Z",
        ])

    def test_fetch_parser_rejects_cache_and_route_overrides(self) -> None:
        parser = core.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["fetch", "--plan", "plan.json", "--run-dir", "run", "--cache-dir", "elsewhere"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["fetch", "--plan", "plan.json", "--run-dir", "run", "--force-route"])

    def test_fetch_requires_plan_path_and_fallback_trace_is_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(TypeError, "plan path"):
                core.fetch_request({"schema_version": "sscofs_download_estimate_v2"}, folder)
        request = _base_request(
            "2026-07-20T00:00:00Z", "2026-07-20T01:00:00Z", source_policy="aws_then_ncei"
        )
        selected = [{"source_id": "ncei_long_term"}]
        trace = {
            "policy": "aws_then_ncei", "aws": {"status": "success"},
            "ncei": {"status": "success"}, "fallback_triggered": True,
            "coverage_before_fallback": {"missing_times": []},
            "scientific_precedence_before_fallback": [],
        }
        self.assertTrue(core.validate_fallback_decision(request, trace, selected))
        trace["scientific_precedence_before_fallback"] = ["2026-07-20T00:00:00Z"]
        self.assertEqual(core.validate_fallback_decision(request, trace, selected), [])

    def test_interior_station_boundary_requires_scientific_fallback(self) -> None:
        request = core.validate_request({
            "schema_version": "sscofs_request_v1",
            "start_utc": "2026-07-20T15:00:00Z",
            "end_utc_exclusive": "2026-07-20T22:00:00Z",
            "product": "stations", "guidance": "nowcast",
        })
        following = core.parse_object_key(
            "sscofs/netcdf/2026/07/21/sscofs.t03z.20260721.stations.nowcast.nc"
        )
        self.assertIn("2026-07-20T21:00:00Z", core._scientific_fallback_times(request, [following]))

    def test_v1_migrates_and_v2_source_policy_validates(self) -> None:
        legacy = _base_request("2026-07-20T00:00:00Z", "2026-07-20T01:00:00Z")
        self.assertEqual(legacy["schema_version"], "sscofs_request_v2")
        self.assertEqual(legacy["source_policy"], "aws_then_ncei")
        explicit = dict(legacy, source_policy="ncei_only")
        self.assertEqual(core.validate_request(explicit)["source_policy"], "ncei_only")
        with self.assertRaisesRegex(ValueError, "v2-only"):
            core.validate_request({
                "schema_version": "sscofs_request_v1",
                "start_utc": "2026-07-20T00:00:00Z",
                "end_utc_exclusive": "2026-07-20T01:00:00Z",
                "product": "fields",
                "guidance": "nowcast",
                "source_policy": "ncei_only",
            })
        with self.assertRaises(ValueError):
            core.validate_request(dict(explicit, source_policy="other"))
        schema = json.loads((SCRIPT_DIR.parent / "references" / "request.schema.json").read_text())
        self.assertTrue(any(
            item.get("if", {}).get("properties", {}).get("schema_version", {}).get("const") == "sscofs_request_v1"
            and item.get("then", {}).get("not", {}).get("required") == ["source_policy"]
            for item in schema.get("allOf", [])
        ))

    def test_aws_preference_follows_n006_scientific_preference(self) -> None:
        request = _base_request("2026-07-19T21:00:00Z", "2026-07-19T22:00:00Z")
        aws = core.parse_object_key("sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.fields.n000.nc")
        ncei_root = core.archive_sources.get_source_descriptor("ncei_long_term", "sscofs")["root_prefix"]
        ncei = core.parse_object_key(ncei_root + "2026/07/nos.sscofs.fields.n006.20260719.t21z.nc")
        self.assertIsNotNone(aws)
        self.assertIsNotNone(ncei)
        aws.update({"source_id": "aws_operational", "size": 2, "size_bytes": 2})
        ncei.update({"source_id": "ncei_long_term", "size": 3, "size_bytes": 3})
        selected = core.select_objects(request, [aws, ncei])["objects"]
        self.assertEqual(selected[0]["source_id"], "ncei_long_term")

    def test_v2_plan_has_source_and_digest_bindings(self) -> None:
        request = _base_request("2026-07-19T21:00:00Z", "2026-07-19T22:00:00Z", source_policy="aws_only")
        key = "sscofs/netcdf/2026/07/19/sscofs.t21z.20260719.fields.n006.nc"
        descriptor = core.archive_sources.get_source_descriptor("aws_operational", "sscofs")
        item = {**descriptor, **core.parse_object_key(key), "source_id": "aws_operational", "size": 7, "size_bytes": 7,
                "etag": "opaque", "last_modified": "2026-07-20T00:00:00Z", "url": core.archive_sources.canonical_object_url("aws_operational", "sscofs", key)}
        with tempfile.TemporaryDirectory() as folder:
            plan = core.plan_request(request, folder, objects=[item])
        self.assertEqual(plan["schema_version"], "sscofs_download_estimate_v2")
        self.assertEqual(plan["source_totals"]["aws_operational"], {"object_count": 1, "bytes": 7})
        self.assertEqual(len(plan["selected_objects_sha256"]), 64)

    def test_nested_and_legacy_key_parsing(self) -> None:
        nested = core.parse_object_key("sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.fields.n003.nc")
        legacy = core.parse_object_key("sscofs/netcdf/202607/sscofs.t03z.20260720.fields.n003.nc")
        for parsed in (nested, legacy):
            self.assertEqual(_get(parsed, "product"), "fields")
            self.assertEqual(_get(parsed, "guidance"), "nowcast")
            self.assertEqual(int(_get(parsed, "lead_hour", "lead")), 3)
            self.assertEqual(_iso_value(_get(parsed, "run_time_utc", "run_time")), "2026-07-20T03:00:00Z")
            self.assertEqual(_iso_value(_get(parsed, "valid_time_utc", "valid_time")), "2026-07-20T00:00:00Z")

    def test_forecast_formula_run_date_discovery_and_passthrough_guard(self) -> None:
        parsed = core.parse_object_key(
            "sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.fields.f072.nc"
        )
        self.assertEqual(_get(parsed, "guidance"), "forecast")
        self.assertEqual(_iso_value(_get(parsed, "valid_time_utc", "valid_time")), "2026-07-23T03:00:00Z")
        request = _base_request(
            "2026-07-23T03:00:00Z",
            "2026-07-23T04:00:00Z",
            guidance="forecast",
            run_cycle_utc="2026-07-20T03:00:00Z",
        )
        prefixes = core._discovery_prefixes(request)
        self.assertIn("sscofs/netcdf/2026/07/20/", prefixes)
        with self.assertRaises(ValueError):
            core.validate_request(
                {
                    "schema_version": "sscofs_request_v1",
                    "start_utc": "2026-07-20T00:00:00Z",
                    "end_utc_exclusive": "2026-07-20T01:00:00Z",
                    "product": "stations",
                    "guidance": "nowcast",
                    "variables": ["salinity"],
                }
            )

    def test_s3_xml_pagination(self) -> None:
        pages = [
            _s3_xml(
                [("sscofs/netcdf/2026/07/19/sscofs.t21z.20260719.fields.n006.nc", 7, "etag-a")],
                truncated=True,
                token="page-2",
            ),
            _s3_xml(
                [("sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.fields.n001.nc", 11, "etag-b")]
            ),
        ]
        session = PagingSession(pages)
        objects = list(_call(core.list_s3_objects, "sscofs/netcdf/2026/07/", session=session))
        self.assertEqual(len(objects), 2)
        self.assertEqual(len(session.calls), 2)
        second_params = session.calls[1].get("params") or {}
        token = second_params.get("continuation-token") or second_params.get("ContinuationToken")
        self.assertEqual(token, "page-2")
        self.assertEqual(int(_get(objects[0], "size", "size_bytes")), 7)

    def test_duplicate_resolution_missing_policy_and_estimate(self) -> None:
        entries = [
            ("sscofs/netcdf/2026/07/19/sscofs.t21z.20260719.fields.n006.nc", 7, "a"),
            ("sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.fields.n000.nc", 9, "b"),
            ("sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.fields.n001.nc", 11, "c"),
        ]
        objects = list(_call(core.list_s3_objects, "sscofs/netcdf/", session=PagingSession([_s3_xml(entries)])))
        request = _base_request("2026-07-19T21:00:00Z", "2026-07-19T23:00:00Z")
        selection = core.select_objects(request, objects)
        selected = _selection_objects(selection)
        self.assertEqual(len(selected), 2)
        keys = [str(_get(item, "key", "object_key")) for item in selected]
        self.assertTrue(any("t21z.20260719.fields.n006" in key for key in keys))
        self.assertFalse(any("fields.n000" in key for key in keys))
        with tempfile.TemporaryDirectory(prefix="sscofs-plan-") as temp:
            plan = _call(core.plan_request, request, run_dir=Path(temp), objects=objects)
        totals = [int(value) for value in _recursive_values(plan, {"total_bytes", "request_bytes"}) if isinstance(value, (int, float))]
        self.assertIn(18, totals)
        routes = [str(value).lower() for value in _recursive_values(plan, {"routing_decision", "route", "recommended_route"})]
        self.assertTrue(any("local" in value for value in routes))

        incomplete = _base_request("2026-07-19T21:00:00Z", "2026-07-20T00:00:00Z")
        with self.assertRaises((ValueError, RuntimeError)):
            core.select_objects(incomplete, objects)
        skip = _base_request(
            "2026-07-19T21:00:00Z", "2026-07-20T00:00:00Z", missing_policy="skip"
        )
        skip_result = core.select_objects(skip, objects)
        self.assertEqual(len(_selection_objects(skip_result)), 2)
        missing = _recursive_values(skip_result, {"missing_times", "missing", "missing_valid_times"})
        self.assertTrue(missing)

    def test_four_times_free_space_route(self) -> None:
        huge = 10**18
        entries = [
            ("sscofs/netcdf/2026/07/19/sscofs.t21z.20260719.fields.n006.nc", huge, "a"),
        ]
        objects = list(_call(core.list_s3_objects, "sscofs/netcdf/", session=PagingSession([_s3_xml(entries)])))
        request = _base_request("2026-07-19T21:00:00Z", "2026-07-19T22:00:00Z")
        with tempfile.TemporaryDirectory(prefix="sscofs-route-") as temp:
            plan = _call(core.plan_request, request, run_dir=Path(temp), objects=objects)
        routes = [str(value).lower() for value in _recursive_values(plan, {"routing_decision", "route", "recommended_route"})]
        self.assertTrue(any("kestrel" in value for value in routes))


class DownloadTests(unittest.TestCase):
    def test_verified_legacy_aws_cache_is_reused_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            key = "sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.fields.n003.nc"
            item = core._decorate_object({
                **core.parse_object_key(key), "source_id": "aws_operational",
                "size": 1, "size_bytes": 1, "etag": "legacy-etag",
                "last_modified": "2026-07-20T06:00:00Z",
            })
            legacy = core._legacy_aws_destination(run, item)
            legacy.parent.mkdir(parents=True)
            _write_fixture(legacy)
            item["size"] = item["size_bytes"] = legacy.stat().st_size
            digest = core._sha256(legacy)
            core.write_json_atomic(core._download_sidecar(legacy), {
                "schema_version": "sscofs_cached_object_v1", "key": key, "url": item["url"],
                "size_bytes": item["size_bytes"], "etag": item["etag"],
                "last_modified": item["last_modified"], "sha256": digest,
            })
            result = core._legacy_aws_cache_result(item, legacy, validate_netcdf=True)
            self.assertIsNotNone(result)
            self.assertTrue(result["legacy_cache_reused"])
            ncei = dict(item, source_id="ncei_long_term")
            self.assertIsNone(core._legacy_aws_cache_result(ncei, legacy))

    def test_stale_partial_source_identity_is_discarded(self) -> None:
        payload, etag = b"replacement", "new-etag"
        key = "sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.stations.nowcast.nc"
        objects = list(_call(core.list_s3_objects, "sscofs/netcdf/", session=PagingSession([_s3_xml([(key, len(payload), etag)])])))
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / Path(key).name
            Path(str(destination) + ".part").write_bytes(b"old")
            core.write_json_atomic(core._partial_sidecar(destination), {"schema_version": "sscofs_partial_object_v2", "source_identity": "stale"})
            session = DownloadSession(payload, etag)
            core.download_object(objects[0], destination, session=session, max_retries=0)
            self.assertEqual((session.get_calls[0].get("headers") or {}), {})
            self.assertEqual(destination.read_bytes(), payload)

    def test_resume_cache_hit_and_multipart_etag(self) -> None:
        payload = b"anonymous-sscofs-payload"
        etag = "0123456789abcdef0123456789abcdef-2"  # multipart: never an MD5 checksum
        key = "sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.stations.nowcast.nc"
        objects = list(
            _call(
                core.list_s3_objects,
                "sscofs/netcdf/",
                session=PagingSession([_s3_xml([(key, len(payload), etag)])]),
            )
        )
        self.assertEqual(len(objects), 1)
        with tempfile.TemporaryDirectory(prefix="sscofs-download-") as temp:
            destination = Path(temp) / Path(key).name
            part = Path(str(destination) + ".part")
            part.write_bytes(payload[:5])
            core.write_json_atomic(
                core._partial_sidecar(destination),
                {
                    "schema_version": "sscofs_partial_object_v2",
                    "source_id": objects[0].get("source_id", "aws_operational"),
                    "source_identity": core.archive_sources.source_identity_digest({**objects[0], "source_id": objects[0].get("source_id", "aws_operational")}),
                    "key": objects[0]["key"], "url": objects[0]["url"],
                    "size_bytes": len(payload), "etag": etag,
                    "last_modified": objects[0].get("last_modified"),
                },
            )
            session = DownloadSession(payload, etag)
            result = _call(core.download_object, objects[0], destination, session=session, max_retries=1)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(part.exists())
            range_headers = [
                (call.get("headers") or {}).get("Range") for call in session.get_calls
            ]
            self.assertIn("bytes=5-", range_headers)
            self.assertEqual(
                _get(result, "sha256", "sha256_hex", default=hashlib.sha256(payload).hexdigest()),
                hashlib.sha256(payload).hexdigest(),
            )
            # A complete size/ETag-backed cache entry must be idempotent.
            cache_result = _call(
                core.download_object,
                objects[0],
                destination,
                session=DownloadSession(payload, etag, forbid_network=True),
                max_retries=1,
            )
            cache_status = str(_get(cache_result, "status", "outcome", default="cache_hit")).lower()
            cache_flag = bool(_get(cache_result, "cache_hit", default=False))
            self.assertTrue(cache_flag or "cache" in cache_status)


class VerticalAndExtractionTests(unittest.TestCase):
    def test_mixed_archive_extraction_provenance_and_drift(self) -> None:
        from netCDF4 import Dataset
        with tempfile.TemporaryDirectory(prefix="sscofs-mixed-") as folder:
            root = Path(folder)
            aws_path, ncei_path = root / "aws.nc", root / "ncei.nc"
            _write_fixture(aws_path)
            _write_fixture(ncei_path)
            with Dataset(ncei_path, "a") as ds:
                ds.variables["time"][:] = [1.0]
                ds.variables["Times"][:] = __import__("numpy").asarray(
                    [list("2026-07-20T01:00:00Z")], dtype="S1"
                )
            request = _base_request("2026-07-20T00:00:00Z", "2026-07-20T02:00:00Z")
            aws_key = "sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.fields.n003.nc"
            ncei_key = core.archive_sources.get_source_descriptor("ncei_long_term", "sscofs")["root_prefix"] + \
                "2026/07/nos.sscofs.fields.n004.20260720.t03z.nc"
            sources = [
                {"local_path": str(aws_path), "source_id": "aws_operational", "key": aws_key, "url": "aws-url"},
                {"local_path": str(ncei_path), "source_id": "ncei_long_term", "key": ncei_key, "url": "ncei-url"},
            ]
            result = core.extract_fields([aws_path, ncei_path], request, root / "mixed.nc", source_records=sources)
            self.assertEqual(result["source_summary"], {"aws_operational": 1, "ncei_long_term": 1})
            self.assertEqual({item["source_id"] for item in result["source_records"]}, {"aws_operational", "ncei_long_term"})
            with Dataset(result["output_path"]) as ds:
                self.assertEqual(json.loads(ds.source_summary_json), result["source_summary"])
                self.assertEqual(health._decode_char_rows(ds.variables["source_archive"][:]),
                                 [item["source_id"] for item in result["source_records"]])
                self.assertEqual(health._decode_char_rows(ds.variables["source_key"][:]),
                                 [item["key"] for item in result["source_records"]])
            report, critical = health._verify_v2_extraction_provenance(
                result,
                [{"source_id": item["source_id"], "key": item["key"], "url": item["url"]} for item in sources],
                {"records": [{**item, "status": "downloaded"} for item in sources]}, root,
            )
            self.assertFalse(critical, critical)
            self.assertEqual(report["source_summary"], result["source_summary"])

            drift = root / "ncei-drift.nc"
            _write_fixture(drift)
            with Dataset(drift, "a") as ds:
                ds.variables["time"][:] = [1.0]
                ds.variables["Times"][:] = __import__("numpy").asarray(
                    [list("2026-07-20T01:00:00Z")], dtype="S1"
                )
                ds.variables["lon"][0] = 0.25
            with self.assertRaisesRegex(ValueError, "geometry|topology"):
                core.extract_fields(
                    [aws_path, drift], request, root / "drift.nc",
                    source_records=[sources[0], {**sources[1], "local_path": str(drift)}],
                )

    def test_v2_health_binds_custom_plan_and_rejects_duplicate_records(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sscofs-v2-health-") as folder:
            run = Path(folder)
            request = _base_request("2026-07-20T00:00:00Z", "2026-07-20T01:00:00Z")
            request_path = run / "request.json"
            core.write_json_atomic(request_path, request)
            key = "sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.fields.n003.nc"
            seed = core._decorate_object({
                **core.parse_object_key(key), "source_id": "aws_operational",
                "size": 1, "size_bytes": 1, "etag": "fixture-etag",
                "last_modified": "2026-07-20T06:00:00Z",
            })
            raw = run / "cache" / "raw" / core.archive_sources.cache_relpath(seed)
            raw.parent.mkdir(parents=True)
            _write_fixture(raw)
            source = core._decorate_object({**seed, "size": raw.stat().st_size, "size_bytes": raw.stat().st_size})
            digest = core._sha256(raw)
            core.write_json_atomic(core._download_sidecar(raw), {
                "schema_version": "sscofs_cached_object_v2", "source_id": "aws_operational",
                "source_identity": core.archive_sources.source_identity_digest(source),
                "key": key, "url": source["url"], "size_bytes": source["size_bytes"],
                "etag": source["etag"], "last_modified": source["last_modified"],
                "etag_semantics": "opaque_provenance", "sha256": digest,
            })
            plan = core.plan_request(request, run, objects=[source])
            plan["source_discovery"] = {
                "policy": "aws_then_ncei", "aws": {"status": "success", "object_count": 1},
                "ncei": {"status": "not_requested", "object_count": 0},
                "fallback_triggered": False, "coverage_before_fallback": {"missing_times": []},
                "scientific_precedence_before_fallback": [],
            }
            custom_plan = run / "review" / "custom_plan.json"
            custom_plan.parent.mkdir()
            core.write_json_atomic(custom_plan, plan)
            record = {
                "key": key, "url": source["url"], "local_path": str(raw.resolve()),
                "status": "downloaded", "cache_hit": False, "size_bytes": source["size_bytes"],
                "etag": source["etag"], "sha256": digest, "source_id": "aws_operational",
                "source_identity": core.archive_sources.source_identity_digest(source),
                "valid_time_utc": "2026-07-20T00:00:00Z", "run_time_utc": source["run_time_utc"],
                "product": "fields", "guidance": "nowcast", "source": source,
            }
            manifest = {
                "schema_version": "sscofs_fetch_manifest_v2", "request": request,
                "estimate_path": str(custom_plan.resolve()), "reviewed_plan_sha256": core._sha256(custom_plan),
                "normalized_request_sha256": plan["normalized_request_sha256"],
                "selected_objects_sha256": plan["selected_objects_sha256"],
                "selected_object_count_binding": 1, "selected_total_bytes_binding": source["size_bytes"],
                "selected_object_count": 1, "successful_object_count": 1, "failure_count": 0,
                "downloaded_count": 1, "cache_hit_count": 0, "records": [record], "complete": True,
                "source_totals": plan["source_totals"],
            }
            core.write_json_atomic(run / "fetch_manifest.json", manifest)
            compact = run / "compact.nc"
            extraction = core.extract_fields([raw], request, compact, source_records=[record])
            from netCDF4 import Dataset
            with Dataset(compact, "a") as ds:
                ds.variables["salinity"][0, 1, 1] = 20.0
            extraction["sha256"] = core._sha256(compact)
            extraction["size_bytes"] = compact.stat().st_size
            core.write_json_atomic(run / "extraction_manifest.json", extraction)
            # No canonical estimate exists: health must follow the manifest's reviewed path.
            (run / "download_estimate.json").unlink(missing_ok=True)
            report = health.evaluate_health(request_path, run, run / "health.json", run / "plots")
            self.assertFalse(report["critical_caveats"], report["critical_caveats"])
            tampered = json.loads(json.dumps(manifest))
            tampered["records"].append(dict(record))
            core.write_json_atomic(run / "fetch_manifest.json", tampered)
            report = health.evaluate_health(request_path, run, run / "health-tampered.json", run / "plots2")
            self.assertTrue(any("duplicate" in item or "cardinality" in item for item in report["critical_caveats"]), report["critical_caveats"])

    def test_station_health_records_preceding_cycle_terminal_winner(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            preceding = root / "preceding.nc"
            following = root / "following.nc"
            _write_station_times(preceding, datetime(2026, 7, 20, 15, tzinfo=UTC))
            _write_station_times(following, datetime(2026, 7, 20, 21, tzinfo=UTC))
            manifest = {"records": [
                {"local_path": str(preceding), "key": "preceding-terminal.nc",
                 "source_id": "ncei_long_term", "run_time_utc": "2026-07-20T21:00:00Z"},
                {"local_path": str(following), "key": "following-initial.nc",
                 "source_id": "aws_operational", "run_time_utc": "2026-07-21T03:00:00Z"},
            ]}
            times, report, _ = health._aggregate_passthrough_times([preceding, following], manifest)
            boundary = next(
                item for item in report["selected_time_records"]
                if item["normalized_time_utc"] == "2026-07-20T21:00:00Z"
            )
            self.assertIn(datetime(2026, 7, 20, 21, tzinfo=UTC), times)
            self.assertEqual(boundary["source_key"], "preceding-terminal.nc")
            self.assertEqual(boundary["candidate_count"], 2)

    def test_cache_cleanup_is_deferred_and_preserves_fetch_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            raw = run / "cache" / "raw" / "source.nc"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"payload")
            digest = core._sha256(raw)
            core.write_json_atomic(core._download_sidecar(raw), {"fixture": True})
            core.write_json_atomic(run / "fetch_manifest.json", {
                "schema_version": "sscofs_fetch_manifest_v2",
                "records": [{"status": "cache_hit", "local_path": str(raw), "key": "key",
                             "source_id": "aws_operational", "sha256": digest}],
            })
            # Extraction no longer calls cleanup. Health invokes this helper only
            # after its complete integrity/provenance gate passes.
            self.assertTrue(raw.is_file())
            cleanup = core._delete_raw_after_extract(run)
            self.assertFalse(raw.exists())
            self.assertEqual(cleanup["deleted_file_count"], 1)
            manifest = core._read_json(run / "fetch_manifest.json")
            self.assertEqual(manifest["records"][0]["status"], "cache_hit")
            self.assertTrue((run / "cache_cleanup.json").is_file())

    def test_v2_extraction_provenance_is_exact_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sscofs-provenance-") as temp:
            run_dir = Path(temp)
            source = run_dir / "raw_fields.nc"
            _write_fixture(source)
            request = _base_request("2026-07-20T00:00:00Z", "2026-07-20T01:00:00Z")
            key = "sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.fields.n003.nc"
            url = core.archive_sources.canonical_object_url("aws_operational", "sscofs", key)
            source_record = {
                "source_id": "aws_operational",
                "key": key,
                "url": url,
                "local_path": str(source.resolve()),
                "status": "downloaded",
            }
            extraction = core.extract_fields(
                [source], request, run_dir / "compact_fields.nc", source_records=[source_record]
            )
            report, critical = health._verify_v2_extraction_provenance(
                extraction,
                [{"source_id": "aws_operational", "key": key, "url": url}],
                {"records": [source_record]},
                run_dir,
            )
            self.assertFalse(critical, critical)
            self.assertTrue(report["exact_plan_fetch_extraction_match"])
            self.assertTrue(report["compact_provenance_matches"])

            forged = json.loads(json.dumps(extraction))
            forged["source_records"][0]["key"] = key.replace("n003", "n004")
            _, forged_critical = health._verify_v2_extraction_provenance(
                forged,
                [{"source_id": "aws_operational", "key": key, "url": url}],
                {"records": [source_record]},
                run_dir,
            )
            self.assertTrue(any("exactly match" in item for item in forged_critical), forged_critical)
            self.assertTrue(any("compact" in item for item in forged_critical), forged_critical)

            with Path(extraction["output_path"]).open("ab") as handle:
                handle.write(b"tamper")
            _, hash_critical = health._verify_v2_extraction_provenance(
                extraction,
                [{"source_id": "aws_operational", "key": key, "url": url}],
                {"records": [source_record]},
                run_dir,
            )
            self.assertTrue(any("SHA-256" in item for item in hash_critical), hash_critical)
            self.assertTrue(any("byte size" in item for item in hash_critical), hash_critical)

    def test_reversed_sigma_missing_layer_and_dry_mask_math(self) -> None:
        import numpy as np

        siglev = np.array([[-1.0, -1.0], [-0.5, -0.5], [-0.2, -0.2], [0.0, 0.0]])
        weights = np.asarray(core.thickness_weights(siglev), dtype=float)
        self.assertEqual(weights.shape, (3, 2))
        np.testing.assert_allclose(weights[:, 0], [0.5, 0.3, 0.2], atol=1.0e-12)
        data = np.array([[[10.0, 10.0], [20.0, np.nan], [30.0, 30.0]]])
        wet = np.array([[1, 0]], dtype=bool)
        average = np.asarray(core.weighted_vertical_average(data, weights, wet_mask=wet), dtype=float)
        self.assertAlmostEqual(float(average.reshape(-1)[0]), 17.0, places=12)
        self.assertTrue(np.isnan(average.reshape(-1)[1]))
        renormalized = np.asarray(core.weighted_vertical_average(data, weights), dtype=float)
        self.assertAlmostEqual(float(renormalized.reshape(-1)[1]), 11.0 / 0.7, places=12)

    def test_near_surface_and_explicit_sigma_views(self) -> None:
        from netCDF4 import Dataset

        with tempfile.TemporaryDirectory(prefix="sscofs-views-") as temp:
            source = Path(temp) / "raw_fields.nc"
            _write_fixture(source)
            # Keep this direct-view fixture fully finite over wet support; the
            # separate math/extraction tests exercise missing-layer averaging.
            with Dataset(source, "a") as ds:
                ds.variables["salinity"][0, 1, 1] = 20.0
            request = _base_request(
                "2026-07-20T00:00:00Z",
                "2026-07-20T01:00:00Z",
                vertical_views=["near_surface", 1],
            )
            output = Path(temp) / "views.nc"
            _call(core.extract_fields, [source], request, output)
            with Dataset(output) as ds:
                self.assertEqual(int(ds.getncattr("surface_sigma_index")), 2)
                self.assertEqual(int(ds.getncattr("near_surface_sigma_index")), 1)
                self.assertEqual(int(ds.getncattr("bottom_sigma_index")), 0)
                required = {
                    "salinity_near_surface", "u_near_surface", "v_near_surface",
                    "current_speed_near_surface", "salinity_sigma_001", "u_sigma_001",
                    "v_sigma_001", "current_speed_sigma_001",
                }
                self.assertTrue(required.issubset(ds.variables))
                self.assertAlmostEqual(float(ds.variables["salinity_near_surface"][0, 0]), 20.0, places=5)
                self.assertAlmostEqual(float(ds.variables["salinity_sigma_001"][0, 0]), 20.0, places=5)
                self.assertAlmostEqual(float(ds.variables["u_near_surface"][0, 0]), 2.0, places=5)
                self.assertAlmostEqual(float(ds.variables["v_near_surface"][0, 0]), 1.0, places=5)
                self.assertAlmostEqual(
                    float(ds.variables["current_speed_near_surface"][0, 0]), math.hypot(2.0, 1.0), places=5
                )

    def test_synthetic_fvcom_extract_and_health(self) -> None:
        import numpy as np
        from netCDF4 import Dataset

        with tempfile.TemporaryDirectory(prefix="sscofs-fixture-") as temp:
            run_dir = Path(temp)
            source = run_dir / "raw_fields.nc"
            expected = _write_fixture(source)
            request = _base_request("2026-07-20T00:00:00Z", "2026-07-20T01:00:00Z")
            compact = run_dir / "compact_fields.nc"
            _call(core.extract_fields, [source], request, compact)
            self.assertTrue(compact.is_file())
            with Dataset(compact, "r") as ds:
                required = {
                    "salinity", "u", "v",
                    "salinity_surface", "salinity_bottom", "salinity_depth_average",
                    "u_surface", "v_surface", "current_speed_surface",
                    "u_bottom", "v_bottom", "current_speed_bottom",
                    "u_depth_average", "v_depth_average", "current_speed_depth_average",
                }
                self.assertTrue(required.issubset(ds.variables))
                self.assertAlmostEqual(float(ds.variables["salinity_surface"][0, 0]), 30.0, places=10)
                self.assertAlmostEqual(float(ds.variables["salinity_bottom"][0, 0]), 10.0, places=10)
                self.assertAlmostEqual(
                    float(ds.variables["salinity_depth_average"][0, 0]),
                    expected["salinity_average_complete"],
                    places=5,
                )
                self.assertAlmostEqual(
                    float(ds.variables["salinity_depth_average"][0, 1]),
                    expected["salinity_average_missing"],
                    places=5,
                )
                self.assertAlmostEqual(float(ds.variables["u_depth_average"][0, 0]), expected["u_average"], places=5)
                self.assertAlmostEqual(float(ds.variables["v_depth_average"][0, 0]), expected["v_average"], places=5)
                self.assertAlmostEqual(
                    float(ds.variables["current_speed_depth_average"][0, 0]),
                    expected["speed_average"],
                    places=5,
                )
                # Dry values may be masked or NaN, but must not become plausible source data.
                dry = ds.variables["salinity_depth_average"][0, 2]
                self.assertTrue(np.ma.is_masked(dry) or not np.isfinite(float(dry)))

            # The missing source layer above proves finite-layer
            # renormalization.  Fill only the compact source copy before the
            # strict >=95% health gate; derived answers remain unchanged.
            with Dataset(compact, "a") as ds:
                ds.variables["salinity"][0, 1, 1] = 20.0

            request_path = run_dir / "request.json"
            request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
            source_key = "sscofs/netcdf/2026/07/20/sscofs.t03z.20260720.fields.n003.nc"
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            estimate = {
                "schema_version": "sscofs_download_estimate_v1",
                "objects": [{"key": source_key, "size": source.stat().st_size, "etag": "fixture-etag"}],
                "total_bytes": source.stat().st_size,
            }
            manifest = {
                "schema_version": "sscofs_fetch_manifest_v1",
                "objects": [{
                    "key": source_key,
                    "local_path": str(source),
                    "size": source.stat().st_size,
                    "etag": "fixture-etag",
                    "sha256": source_hash,
                    "valid_time_utc": "2026-07-20T00:00:00Z",
                    "status": "downloaded",
                }],
            }
            (run_dir / "download_estimate.json").write_text(
                json.dumps(estimate, indent=2) + "\n", encoding="utf-8"
            )
            (run_dir / "fetch_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            report = health.evaluate_health(
                request_path,
                run_dir,
                run_dir / "health.json",
                run_dir / "plots",
            )
            self.assertFalse(report["critical_caveats"], report["critical_caveats"])
            self.assertTrue(report["acceptance"]["passed"])
            plots = report["selected_netcdf"].get("plots", [])
            self.assertEqual(len(plots), 6)
            self.assertTrue(all(Path(item["path"]).is_file() for item in plots))


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
