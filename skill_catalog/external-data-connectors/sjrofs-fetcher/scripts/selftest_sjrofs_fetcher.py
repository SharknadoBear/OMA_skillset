#!/usr/bin/env python3
"""Offline regression tests for SJROFS discovery, transfer, EFDC extraction, and QA."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sjrofs_fetcher as sf

UTC = timezone.utc


def request(*, start="2026-07-20T00:00:00Z", end="2026-07-21T00:00:00Z",
            product="fields", guidance="nowcast", policy="aws_then_ncei",
            version="sjrofs_request_v2", missing="error") -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": version, "start_utc": start, "end_utc_exclusive": end,
        "product": product, "guidance": guidance, "missing_policy": missing,
        "cache_policy": "keep", "max_workers": 2,
    }
    if version.endswith("v2"):
        result["source_policy"] = policy
    if product == "fields":
        result["variables"] = ["zeta", "salt", "u", "v"]
        result["vertical_views"] = ["surface", "near_surface", "bottom", "depth_average", 3]
    if guidance == "forecast":
        result["run_cycle_utc"] = "2026-07-20T05:00:00Z"
    return result


def source(key: str, *, source_id="aws_operational", size=100, etag="opaque-2") -> dict[str, Any]:
    parsed = sf.parse_object_key(key)
    if parsed is None:
        raise AssertionError(key)
    descriptor = sf.archive_sources.get_source_descriptor(source_id, "sjrofs")
    result = {
        **descriptor, **parsed, "source_id": source_id, "size": size, "etag": etag,
        "last_modified": "2026-07-20T06:00:00Z",
        "url": sf.archive_sources.canonical_object_url(source_id, "sjrofs", key),
    }
    return sf._decorate_source(result, source_id)


class FakeResponse:
    def __init__(self, content=b"", status_code=200, headers=None):
        self.content, self.status_code = content, status_code
        self.headers = dict(headers or {})
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
    def iter_content(self, chunk_size):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset:offset + chunk_size]
    def close(self):
        return None


class ResumeSession:
    def __init__(self, payload, etag):
        self.payload, self.etag, self.calls = payload, etag, []
    def get(self, url, *, headers, stream, timeout):
        self.calls.append(dict(headers))
        start = int(headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        content = self.payload[start:]
        headers_out = {"ETag": f'"{self.etag}"', "Content-Length": str(len(content))}
        if start:
            headers_out["Content-Range"] = f"bytes {start}-{len(self.payload)-1}/{len(self.payload)}"
        return FakeResponse(content, 206 if start else 200, headers_out)


def write_fields(path: Path, first: datetime, *, reverse=True, geometry_delta=0.0,
                 invalid_mask=False, dry_value=False, invalid_vector=False,
                 atmospheric_dry=False):
    import netCDF4
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        for name, size in (("nx", 8), ("ny", 6), ("sigma", 6), ("time", 6)):
            ds.createDimension(name, size)
        ds.setncatts({"model": "EFDC_3D", "grid_type": "curvilinear", "source": "SJROFS"})
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2008-01-01 00:00:00 UTC"
        decoded = [first + timedelta(hours=i, seconds=14 if i % 2 == 0 else -14) for i in range(6)]
        time[:] = netCDF4.date2num(decoded, time.units)
        yy, xx = np.mgrid[:6, :8]
        lon = -81.8 + 0.02 * xx + 0.004 * yy
        lat = 29.5 + 0.03 * yy + 0.002 * xx
        lon[2, 2] += geometry_delta
        mask = np.zeros((6, 8), dtype="f4")
        mask[1:5, 1:7] = 5
        mask[5, 7] = -85.48875
        if invalid_mask:
            mask[0, 1] = 2
        depth = 2.0 + yy + xx / 10
        lon[mask != 5], lat[mask != 5], depth[mask != 5] = 0, 0, 0.01
        sigma = np.asarray([0, .167, .334, .501, .668, .833], dtype="f4")
        if reverse:
            sigma = sigma[::-1]
        for name, values, dims in (
            ("lon", lon, ("ny", "nx")), ("lat", lat, ("ny", "nx")),
            ("mask", mask, ("ny", "nx")), ("depth", depth, ("ny", "nx")),
            ("sigma", sigma, ("sigma",)),
        ):
            var = ds.createVariable(name, "f4", dims)
            var[:] = values
        ds.variables["sigma"].positive = "down"
        fill = np.float32(-99999)
        specs = (
            ("zeta", ("time", "ny", "nx"), "sea_surface_height_above_geoid", "m"),
            ("salt", ("time", "sigma", "ny", "nx"), "sea_water_practical_salinity", "1e-3"),
            ("u", ("time", "sigma", "ny", "nx"), "grid_eastward_sea_water_velocity" if invalid_vector else "eastward_sea_water_velocity", "m s-1"),
            ("v", ("time", "sigma", "ny", "nx"), "northward_sea_water_velocity", "m s-1"),
            ("air_u", ("time", "ny", "nx"), "eastward_wind", "m s-1"),
            ("air_v", ("time", "ny", "nx"), "northward_wind", "m s-1"),
        )
        wet = mask == 5
        for name, dims, standard_name, units in specs:
            var = ds.createVariable(name, "f4", dims, fill_value=fill)
            var.missing_value, var.standard_name, var.units = fill, standard_name, units
            shape = tuple(len(ds.dimensions[d]) for d in dims)
            data = np.full(shape, fill, dtype="f4")
            if len(shape) == 3:
                for t in range(6):
                    data[t, wet] = .1 * t
                    if atmospheric_dry and name in {"air_u", "air_v"}:
                        data[t, ~wet] = 2.0
            else:
                for t in range(6):
                    for k, top in enumerate(sigma):
                        value = 30 + 2 * top if name == "salt" else (.1 + top if name == "u" else .2 + .5 * top)
                        data[t, k, wet] = value + .01 * t
                data[0, 2, 2, 2] = fill
            if dry_value:
                data[(0,) * (data.ndim - 2) + (0, 0)] = 7
            var[:] = data


def synthetic_run(root: Path, *, drift=False) -> tuple[dict[str, Any], Path]:
    req = request(end="2026-07-20T12:00:00Z")
    req_path = root / "request.json"
    sf.write_json_atomic(req_path, req)
    objects, outcomes = [], []
    for cycle, first, delta in (
        (5, datetime(2026, 7, 20, 0, 30, tzinfo=UTC), 0),
        (11, datetime(2026, 7, 20, 6, 30, tzinfo=UTC), .02 if drift else 0),
    ):
        key = f"sjrofs/netcdf/2026/07/20/sjrofs.t{cycle:02}z.20260720.fields.nowcast.nc"
        placeholder = source(key, etag=f"synthetic-{cycle}-2")
        path = sf._destination_for_object(root, placeholder)
        write_fields(path, first, geometry_delta=delta)
        item = source(key, size=path.stat().st_size, etag=f"synthetic-{cycle}-2")
        digest = sf._sha256(path)
        objects.append(item)
        outcomes.append({
            "key": key, "url": item["url"], "local_path": str(path.resolve()),
            "status": "downloaded", "size": item["size"], "etag": item["etag"],
            "sha256": digest, "resumed": False, "resumed_from_bytes": 0,
            "retry_count": 0, "source": item, "source_id": item["source_id"],
            "source_identity": item["source_identity"],
        })
        sf.write_json_atomic(sf._download_sidecar(path), {
            "schema_version": "sjrofs_cached_object_v2", "key": key, "url": item["url"],
            "size": item["size"], "etag": item["etag"], "last_modified": item["last_modified"],
            "sha256": digest, "source_id": item["source_id"], "source_identity": item["source_identity"],
            "etag_semantics": "opaque_provenance",
        })
    estimate = sf.plan_request(req, root, objects=objects)
    estimate["source_discovery"] = {
        "policy": "aws_then_ncei", "aws": {"status": "success", "object_count": 2},
        "ncei": {"status": "not_requested", "object_count": 0},
        "fallback_triggered": False, "coverage_before_fallback": [],
        "scientific_precedence_before_fallback": [],
    }
    estimate_path = root / "download_estimate.json"
    sf.write_json_atomic(estimate_path, estimate)
    manifest = {
        "schema_version": "sjrofs_fetch_manifest_v2", "request": sf.validate_request(req),
        "estimate_path": str(estimate_path.resolve()), "reviewed_plan_sha256": sf._sha256(estimate_path),
        "normalized_request_sha256": estimate["normalized_request_sha256"],
        "selected_objects_sha256": estimate["selected_objects_sha256"],
        "selected_object_count_binding": 2, "selected_total_bytes_binding": estimate["total_bytes"],
        "outcomes": outcomes, "counts": {"objects": 2, "downloaded": 2, "cache_hits": 0, "failed": 0, "resumed": 0},
        "source_discovery": estimate["source_discovery"], "source_totals": estimate["source_totals"],
    }
    sf.write_json_atomic(root / "fetch_manifest.json", manifest)
    return req, req_path


class RequestAndSelectionTests(unittest.TestCase):
    def test_v1_migration_defaults_and_station_rejections(self):
        normalized = sf.validate_request(request(version="sjrofs_request_v1"))
        self.assertEqual(normalized["schema_version"], "sjrofs_request_v2")
        self.assertEqual(normalized["source_policy"], "aws_then_ncei")
        self.assertEqual(normalized["variables"], ["zeta", "salt", "u", "v"])
        self.assertNotIn("grid", normalized)
        with self.assertRaisesRegex(ValueError, "passthrough"):
            sf.validate_request({**request(product="stations"), "variables": ["salt"]})
        with self.assertRaisesRegex(ValueError, "unknown request"):
            sf.validate_request({**request(), "grid": "coarse"})

    def test_forecast_cycle_and_ncei_capability(self):
        with self.assertRaisesRegex(ValueError, "run_cycle_utc"):
            sf.validate_request({**request(guidance="forecast"), "run_cycle_utc": None})
        with self.assertRaisesRegex(ValueError, "field-forecast"):
            sf.discover_objects(sf.validate_request(request(guidance="forecast", policy="ncei_only")))

    def test_key_layouts_and_discrete_half_hour_selection(self):
        current = sf.parse_object_key("sjrofs/netcdf/2026/07/20/sjrofs.t05z.20260720.fields.nowcast.nc")
        legacy = sf.parse_object_key("sjrofs/netcdf/202607/nos.sjrofs.fields.nowcast.20260720.t05z.nc")
        self.assertEqual((current["layout"], legacy["layout"]), ("daily", "monthly"))
        self.assertEqual(current["expected_start_utc"], "2026-07-20T00:30:00Z")
        req = sf.validate_request(request(end="2026-07-20T06:00:00Z", policy="aws_only"))
        prior = source("sjrofs/netcdf/2026/07/19/sjrofs.t23z.20260719.fields.nowcast.nc")
        today = source("sjrofs/netcdf/2026/07/20/sjrofs.t05z.20260720.fields.nowcast.nc")
        selected = sf.select_objects(req, [prior, today])
        self.assertEqual([item["key"] for item in selected["selected"]], [today["key"]])
        self.assertEqual(selected["nominal_time_count"], 6)
        self.assertFalse(selected["missing_times"])

    def test_exact_one_day_four_objects_and_24_points(self):
        req = sf.validate_request(request(policy="aws_only"))
        items = [source(f"sjrofs/netcdf/2026/07/20/sjrofs.t{h:02}z.20260720.fields.nowcast.nc") for h in (5, 11, 17, 23)]
        selected = sf.select_objects(req, items)
        self.assertEqual(len(selected["selected"]), 4)
        self.assertEqual(selected["nominal_time_count"], 24)
        self.assertEqual(selected["missing_times"], [])

    def test_duplicate_ranking_and_conflict(self):
        req = sf.validate_request(request(end="2026-07-20T06:00:00Z"))
        daily = source("sjrofs/netcdf/2026/07/20/sjrofs.t05z.20260720.fields.nowcast.nc")
        monthly = source("sjrofs/netcdf/202607/sjrofs.t05z.20260720.fields.nowcast.nc")
        self.assertEqual(sf.select_objects(req, [monthly, daily])["selected"][0]["key"], daily["key"])
        conflict = dict(daily, key=daily["key"] + "-alias", size=101)
        with self.assertRaisesRegex(RuntimeError, "same-rank"):
            sf.select_objects(req, [daily, conflict])

    def test_ordered_fallback_and_discovery_error(self):
        req = sf.validate_request(request(end="2026-07-20T06:00:00Z"))
        aws = source("sjrofs/netcdf/2026/07/20/sjrofs.t05z.20260720.fields.nowcast.nc")
        with mock.patch.object(sf, "_discover_source", return_value=[aws]) as discover:
            result, trace = sf.discover_objects(req, with_trace=True)
        self.assertEqual([call.args[1] for call in discover.call_args_list], ["aws_operational"])
        self.assertFalse(trace["fallback_triggered"])
        nroot = sf.archive_sources.get_source_descriptor("ncei_long_term", "sjrofs")["root_prefix"]
        ncei = source(nroot + "2020/01/nos.sjrofs.fields.nowcast.20200101.t05z.nc", source_id="ncei_long_term")
        historical = sf.validate_request(request(start="2020-01-01T00:00:00Z", end="2020-01-01T06:00:00Z"))
        with mock.patch.object(sf, "_discover_source", side_effect=[[], [ncei]]):
            result, trace = sf.discover_objects(historical, with_trace=True)
        self.assertTrue(trace["fallback_triggered"])
        self.assertEqual(result[0]["source_id"], "ncei_long_term")
        with mock.patch.object(sf, "_discover_source", side_effect=RuntimeError("listing failed")):
            with self.assertRaisesRegex(RuntimeError, "listing failed"):
                sf.discover_objects(req)

    def test_plan_gate_and_fetch_requires_reviewed_path(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(TypeError, "plan path"):
                sf.fetch_request({}, folder)
            req = request(end="2026-07-20T06:00:00Z", policy="aws_only")
            item = source("sjrofs/netcdf/2026/07/20/sjrofs.t05z.20260720.fields.nowcast.nc", size=13_110_380)
            plan = sf.plan_request(req, folder, objects=[item])
            self.assertEqual(plan["total_bytes"], 13_110_380)
            self.assertEqual(plan["required_free_bytes"], 52_441_520)


class TransferAndMathTests(unittest.TestCase):
    def test_layer_top_weights_reversed_and_known_average(self):
        import numpy as np
        sigma = np.array([.833, .668, .501, .334, .167, 0])
        weights = sf.efdc_layer_top_weights(sigma)
        expected_sorted = np.diff(np.r_[np.sort(sigma), 1.0])
        self.assertTrue(np.allclose(weights[np.argsort(sigma)], expected_sorted))
        values = (10 + 2 * sigma).reshape(1, 6, 1, 1)
        result = sf.weighted_vertical_average(values, weights, np.ones((1, 1), bool), axis=1)
        self.assertAlmostEqual(float(result[0, 0, 0]), float(np.sum((10 + 2 * sigma) * weights)))
        values[0, 2, 0, 0] = np.nan
        finite = np.isfinite(values[0, :, 0, 0])
        expected = float(np.sum(values[0, finite, 0, 0] * weights[finite]) / weights[finite].sum())
        self.assertAlmostEqual(float(sf.weighted_vertical_average(values, weights, np.ones((1, 1), bool), axis=1)[0, 0, 0]), expected)

    def test_invalid_sigma_fails_closed(self):
        for sigma in ([.1, .5], [0, .5, .5], [0, .5, 1]):
            with self.assertRaises(ValueError):
                sf.efdc_layer_top_weights(sigma)

    def test_resume_multipart_etag_and_cache_hit(self):
        with tempfile.TemporaryDirectory() as folder:
            source_file = Path(folder) / "source.nc"
            write_fields(source_file, datetime(2026, 7, 20, 0, 30, tzinfo=UTC))
            payload = source_file.read_bytes()
            path = Path(folder) / "sample.nc"
            item = {"key": "fixture", "url": "https://example.invalid/fixture", "size": len(payload), "etag": "opaque-3", "last_modified": "now"}
            part = path.with_name(path.name + ".part")
            part.write_bytes(payload[:70])
            sf.write_json_atomic(path.with_name(path.name + ".part.json"), {"schema_version": "fixture", **item, "source_id": None, "source_identity": None})
            result = sf.download_object(item, path, session=ResumeSession(payload, "opaque-3"), max_attempts=1, chunk_size=13)
            self.assertTrue(result["resumed"])
            self.assertEqual(path.read_bytes(), payload)
            sidecar = json.loads(sf._download_sidecar(path).read_text())
            self.assertTrue(sidecar["etag_is_multipart"])

    def test_corrupt_complete_partial_fails_before_promotion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.nc"
            payload = b"not-netcdf"
            item = {"key": "fixture", "url": "https://example.invalid/fixture", "size": len(payload), "etag": "opaque", "last_modified": "now"}
            part = path.with_name(path.name + ".part")
            part.write_bytes(payload)
            sf.write_json_atomic(path.with_name(path.name + ".part.json"), {"schema_version": "fixture", **item, "source_id": None, "source_identity": None})
            with self.assertRaisesRegex(RuntimeError, "not a valid"):
                sf.download_object(item, path)
            self.assertFalse(path.exists())

    def test_mask_vector_and_dry_footprint_fail_closed(self):
        import netCDF4
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, options, pattern in (
                ("mask.nc", {"invalid_mask": True}, "active-water code"),
                ("dry.nc", {"dry_value": True}, "outside source mask"),
                ("vector.nc", {"invalid_vector": True}, "not advertised"),
            ):
                path = root / name
                write_fields(path, datetime(2026, 7, 20, 0, 30, tzinfo=UTC), **options)
                with netCDF4.Dataset(path) as ds:
                    if name == "vector.nc":
                        with self.assertRaisesRegex(RuntimeError, pattern):
                            sf._validate_vector_metadata(ds.variables["u"], ds.variables["v"])
                    else:
                        if name == "dry.nc":
                            geometry = sf._geometry(ds)
                            with self.assertRaisesRegex(ValueError, pattern):
                                sf._validate_dynamic_wet_footprint(ds, geometry)
                        else:
                            with self.assertRaisesRegex(ValueError, pattern):
                                sf._geometry(ds)

    def test_atmospheric_dry_values_are_permitted_but_extraction_masks_them(self):
        import netCDF4
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "atmos.nc"
            write_fields(path, datetime(2026, 7, 20, 0, 30, tzinfo=UTC), atmospheric_dry=True)
            with netCDF4.Dataset(path) as ds:
                geometry = sf._geometry(ds)
                sf._validate_dynamic_wet_footprint(ds, geometry)
                air = sf._read_dynamic_record(ds.variables["air_u"], 0, geometry)
                self.assertFalse(np.isfinite(air[~geometry["wet"]]).any())


class ExtractionAndHealthTests(unittest.TestCase):
    def test_synthetic_extract_and_health(self):
        import netCDF4
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            req, req_path = synthetic_run(root)
            extraction = sf.extract_request(req_path, root)
            self.assertEqual(len(extraction["outputs"]), 1)
            output = Path(extraction["outputs"][0]["path"])
            with netCDF4.Dataset(output) as ds:
                self.assertEqual(ds.schema_version, "efdc_compact_fields_v1")
                self.assertEqual(ds.vertical_method, "efdc_layer_top_sigma_with_bed_edge_1")
                self.assertEqual(len(ds.dimensions["time"]), 12)
                self.assertEqual(np.unique(ds.variables["mask"][:]).tolist(), [-85.4887466430664, 0.0, 5.0])
                self.assertEqual(int(ds.variables["wet_mask"][:].sum()), 24)
                self.assertIn("salinity_depth_average", ds.variables)
                self.assertIn("eastward_velocity_surface", ds.variables)
                u = np.ma.asarray(ds.variables["eastward_velocity_depth_average"][:]).filled(np.nan)
                v = np.ma.asarray(ds.variables["northward_velocity_depth_average"][:]).filled(np.nan)
                speed = np.ma.asarray(ds.variables["current_speed_depth_average"][:]).filled(np.nan)
                self.assertTrue(np.allclose(speed, np.hypot(u, v), equal_nan=True))
            health = sf.evaluate_health(req_path, root)
            self.assertEqual(health["status"], "pass", health["critical_findings"])

    def test_geometry_drift_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, req_path = synthetic_run(root, drift=True)
            with self.assertRaisesRegex(RuntimeError, "geometry drift"):
                sf.extract_request(req_path, root)

    def test_plan_tampering_blocks_extract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, req_path = synthetic_run(root)
            plan_path = root / "download_estimate.json"
            plan = json.loads(plan_path.read_text())
            plan["total_bytes"] += 1
            sf.write_json_atomic(plan_path, plan)
            with self.assertRaisesRegex(RuntimeError, "plan"):
                sf.extract_request(req_path, root)

    def test_outcome_source_tampering_blocks_extract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, req_path = synthetic_run(root)
            manifest_path = root / "fetch_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["outcomes"][0]["source"]["run_time"] = "2026-07-20T11:00:00Z"
            sf.write_json_atomic(manifest_path, manifest)
            with self.assertRaisesRegex(RuntimeError, "source descriptor"):
                sf.extract_request(req_path, root)

    def test_health_request_cannot_expand_cache_deletion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            req, req_path = synthetic_run(root)
            sf.extract_request(req_path, root)
            delete_request = dict(req, cache_policy="delete_after_extract")
            with self.assertRaisesRegex(RuntimeError, "differs from the reviewed plan"):
                sf.evaluate_health(delete_request, root)
            self.assertTrue(any((root / "cache" / "raw").rglob("*.nc")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
