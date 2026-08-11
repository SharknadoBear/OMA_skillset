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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import check_download_health as health  # noqa: E402
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


class KeyAndCatalogTests(unittest.TestCase):
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
