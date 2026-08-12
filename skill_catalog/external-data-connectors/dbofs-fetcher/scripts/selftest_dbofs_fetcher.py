#!/usr/bin/env python3
"""Offline regression tests for the self-contained DBOFS connector."""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import netCDF4
import numpy as np

HERE = Path(__file__).resolve().parent

import dbofs_fetcher as dbofs
import roms_aws_core as core
from roms_aws_core import download_object, list_s3_objects, verify_transfers, write_json_atomic
from roms_processing import (
    compact_health,
    destagger_to_rho,
    extract_fields,
    geometry_snapshot,
    layer_thickness,
    raw_consistency,
    roms_depths,
    rotate_to_earth,
    weighted_vertical_average,
)

UTC = timezone.utc


def request(**changes):
    value = {
        "schema_version": "dbofs_request_v1",
        "start_utc": "2026-07-20T00:00:00Z",
        "end_utc_exclusive": "2026-07-20T02:00:00Z",
        "product": "fields",
        "guidance": "nowcast",
        "variables": ["zeta", "salt", "u", "v"],
        "vertical_views": ["surface", "bottom", "depth_average"],
        "missing_policy": "error",
        "cache_policy": "keep",
        "max_workers": 2,
    }
    value.update(changes)
    return value


def object_item(key: str, size: int = 10, etag: str = "abc-4"):
    parsed = dbofs.parse_object_key(key)
    assert parsed is not None
    return {
        **parsed,
        "size": size,
        "etag": etag,
        "last_modified": "2026-07-20T00:00:00Z",
        "url": core.S3_ENDPOINT + "/" + key,
    }


def write_fixture(
    path: Path,
    hour: int,
    *,
    vtransform: int = 1,
    reversed_sigma: bool = False,
    angle: float = 0.0,
    omit_mask: str | None = None,
):
    eta, xi, layers = 3, 4, 3
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        for name, size in (("ocean_time", 1), ("eta_rho", eta), ("xi_rho", xi),
                           ("eta_u", eta), ("xi_u", xi - 1), ("eta_v", eta - 1),
                           ("xi_v", xi), ("s_rho", layers), ("s_w", layers + 1)):
            ds.createDimension(name, size)
        lon = -76 + np.arange(xi)[None, :] * 0.05 + np.arange(eta)[:, None] * 0.002
        lat = 38 + np.arange(eta)[:, None] * 0.05 + np.arange(xi)[None, :] * 0.001
        h = np.full((eta, xi), 10.0)
        mask = np.ones((eta, xi), dtype=np.int8); mask[0, 0] = 0
        mask_u = mask[:, :-1] * mask[:, 1:]
        mask_v = mask[:-1, :] * mask[1:, :]
        static = {
            "lon_rho": lon, "lat_rho": lat, "h": h,
            "angle": np.full_like(h, angle), "mask_rho": mask,
            "lon_u": 0.5 * (lon[:, :-1] + lon[:, 1:]), "lat_u": 0.5 * (lat[:, :-1] + lat[:, 1:]), "mask_u": mask_u,
            "lon_v": 0.5 * (lon[:-1, :] + lon[1:, :]), "lat_v": 0.5 * (lat[:-1, :] + lat[1:, :]), "mask_v": mask_v,
        }
        dims = {
            "lon_rho": ("eta_rho", "xi_rho"), "lat_rho": ("eta_rho", "xi_rho"),
            "h": ("eta_rho", "xi_rho"), "angle": ("eta_rho", "xi_rho"), "mask_rho": ("eta_rho", "xi_rho"),
            "lon_u": ("eta_u", "xi_u"), "lat_u": ("eta_u", "xi_u"), "mask_u": ("eta_u", "xi_u"),
            "lon_v": ("eta_v", "xi_v"), "lat_v": ("eta_v", "xi_v"), "mask_v": ("eta_v", "xi_v"),
        }
        for name, value in static.items():
            if name == omit_mask:
                continue
            variable = ds.createVariable(name, "i1" if name.startswith("mask_") else "f8", dims[name])
            variable[:] = value
            if name == "angle":
                variable.units = "radians"
                variable.standard_name = "grid_angle_of_rotation_from_east_to_y"
                variable.long_name = "angle between XI-axis and EAST"
        sigma_rho = np.array([-0.8, -0.4, -0.1]); sigma_w = np.array([-1.0, -0.6, -0.2, 0.0])
        salt_values = np.array([10.0, 20.0, 30.0]); u_values = np.array([1.0, 2.0, 3.0])
        if reversed_sigma:
            sigma_rho, sigma_w = sigma_rho[::-1], sigma_w[::-1]
            salt_values, u_values = salt_values[::-1], u_values[::-1]
        for name, dims1, values in (("s_rho", ("s_rho",), sigma_rho), ("Cs_r", ("s_rho",), sigma_rho),
                                    ("s_w", ("s_w",), sigma_w), ("Cs_w", ("s_w",), sigma_w)):
            variable = ds.createVariable(name, "f8", dims1)
            if name == "Cs_w":
                values = values.copy()
                values[int(np.argmin(np.abs(sigma_w)))] = 2.8901245e-17
                variable.valid_max = 0.0
            variable[:] = values
        ds.createVariable("hc", "f8").assignValue(1.0)
        ds.createVariable("Vtransform", "i4").assignValue(vtransform)
        ds.createVariable("Vstretching", "i4").assignValue(1)
        time_var = ds.createVariable("ocean_time", "f8", ("ocean_time",))
        time_var.units = "seconds since 1970-01-01 00:00:00 UTC"
        time_var[:] = datetime(2026, 7, 20, hour, tzinfo=UTC).timestamp() + 30
        zeta = ds.createVariable("zeta", "f4", ("ocean_time", "eta_rho", "xi_rho"), fill_value=1e37)
        zeta.standard_name = "sea_surface_height_above_geoid"; zeta.units = "m"
        zeta.grid = "grid"; zeta.location = "face"
        zeta[0] = np.where(mask, 0.2, 1e37)
        salt = ds.createVariable("salt", "f4", ("ocean_time", "s_rho", "eta_rho", "xi_rho"), fill_value=1e37)
        salt.standard_name = "sea_water_practical_salinity"
        salt.grid = "grid"; salt.location = "face"
        salt_data = np.stack([np.full((eta, xi), value) for value in salt_values])
        salt_data[:, mask == 0] = 1e37
        salt_data[1, 2, 2] = 1e37  # finite-layer renormalization case
        salt[0] = salt_data
        u = ds.createVariable("u", "f4", ("ocean_time", "s_rho", "eta_u", "xi_u"), fill_value=1e37)
        v = ds.createVariable("v", "f4", ("ocean_time", "s_rho", "eta_v", "xi_v"), fill_value=1e37)
        u.standard_name = "sea_water_x_velocity"; v.standard_name = "sea_water_y_velocity"
        u.units = v.units = "m s-1"
        u.grid = v.grid = "grid"; u.location = "edge1"; v.location = "edge2"
        u_data = np.stack([np.full((eta, xi - 1), value) for value in u_values])
        v_data = np.zeros((layers, eta - 1, xi))
        u_data[:, mask_u == 0] = 1e37; v_data[:, mask_v == 0] = 1e37
        u[0] = u_data; v[0] = v_data
        ds.source_key = f"dbofs.t06z.20260720.fields.n00{hour + 1}.nc"


def write_station_fixture(path: Path):
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("ocean_time", 2)
        ds.createDimension("station", 2)
        time_var = ds.createVariable("ocean_time", "f8", ("ocean_time",))
        time_var.units = "seconds since 1970-01-01 00:00:00 UTC"
        time_var[:] = [
            datetime(2026, 7, 19, 23, 54, tzinfo=UTC).timestamp(),
            datetime(2026, 7, 20, 0, 0, tzinfo=UTC).timestamp(),
        ]
        zeta = ds.createVariable("zeta", "f4", ("ocean_time", "station"), fill_value=1e37)
        zeta[:] = [[0.1, 0.2], [0.2, 0.3]]


def netcdf_payload() -> bytes:
    """Return a minimal, fully openable NetCDF payload for transfer tests."""
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "payload.nc"
        with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
            dataset.createDimension("record", 1)
            value = dataset.createVariable("value", "f4", ("record",))
            value[:] = [1.0]
        return path.read_bytes()


class FakeResponse:
    def __init__(self, content=b"", *, status=200, headers=None):
        self.content = content
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start:start + chunk_size]

    def close(self):
        return None


class PaginationSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        key = "dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.n006.nc"
        token = params.get("continuation-token") if params else None
        suffix = "2" if token else "1"
        truncated = "false" if token else "true"
        next_token = "" if token else "<NextContinuationToken>next</NextContinuationToken>"
        xml = f"""<ListBucketResult><IsTruncated>{truncated}</IsTruncated>{next_token}<Contents><Key>{key}.{suffix}</Key><LastModified>2026-01-01T00:00:00Z</LastModified><ETag>\"etag-{suffix}\"</ETag><Size>{suffix}</Size></Contents></ListBucketResult>"""
        return FakeResponse(xml.encode())


class DownloadSession:
    def __init__(self, payload: bytes, etag: str):
        self.payload, self.etag, self.headers_seen = payload, etag, []

    def get(self, url, headers=None, stream=True, timeout=None):
        headers = headers or {}; self.headers_seen.append(headers)
        start = int(headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        data = self.payload[start:]
        return FakeResponse(data, status=206 if start else 200, headers={"Content-Length": str(len(data)), "ETag": self.etag})


class ContractTests(unittest.TestCase):
    def test_package_import_surface(self):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [sys.executable, "-c", "import scripts; assert 'fetch_plan' in scripts.__all__"],
            cwd=HERE.parent, env=environment, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_request_validation_and_passthrough_rejection(self):
        normalized = dbofs.validate_request(request(variables=["salinity", "u", "v", "zeta"]))
        self.assertIn("salt", normalized["variables"])
        with self.assertRaisesRegex(ValueError, "passthrough-only"):
            dbofs.validate_request(request(product="stations"))
        forecast = request(guidance="forecast", run_cycle_utc="2026-07-20T06:00:00Z")
        self.assertEqual(dbofs.validate_request(forecast)["run_cycle_utc"], "2026-07-20T06:00:00Z")
        with self.assertRaisesRegex(ValueError, "required"):
            dbofs.validate_request(request(guidance="forecast"))

    def test_current_legacy_and_time_formulas(self):
        current = dbofs.parse_object_key("dbofs/netcdf/2026/07/20/dbofs.t06z.20260720.fields.n001.nc")
        legacy = dbofs.parse_object_key("dbofs/netcdf/202607/nos.dbofs.fields.n006.20260720.t00z.nc")
        forecast = dbofs.parse_object_key("dbofs/netcdf/2026/07/20/dbofs.t06z.20260720.fields.f003.nc")
        station = dbofs.parse_object_key("dbofs/netcdf/2026/07/20/dbofs.t06z.20260720.stations.nowcast.nc")
        self.assertEqual(current["valid_time"], "2026-07-20T01:00:00Z")
        self.assertEqual(legacy["valid_time"], "2026-07-20T00:00:00Z")
        self.assertEqual(forecast["valid_time"], "2026-07-20T09:00:00Z")
        self.assertEqual(station["expected_start_utc"], "2026-07-20T00:00:00Z")
        self.assertIsNotNone(dbofs.parse_object_key("dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.f048.nc"))
        for invalid in (
            "dbofs/netcdf/2026/07/20/dbofs.t03z.20260720.fields.n001.nc",
            "dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.n000.nc",
            "dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.n007.nc",
            "dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.f000.nc",
            "dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.f049.nc",
        ):
            self.assertIsNone(dbofs.parse_object_key(invalid), invalid)

    def test_selection_duplicate_rank_and_conflict(self):
        req = request(end_utc_exclusive="2026-07-20T01:00:00Z")
        current = object_item("dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.n006.nc")
        legacy = object_item("dbofs/netcdf/202607/nos.dbofs.fields.n006.20260720.t00z.nc")
        result = dbofs.select_objects(req, [legacy, current])
        self.assertEqual(result["selected"][0]["key"], current["key"])
        clone = dict(current); clone["key"] = current["key"].replace("2026/07/20", "2026/07/21"); clone["etag"] = "other"
        with self.assertRaisesRegex(RuntimeError, "equal-rank conflicting"):
            dbofs.select_objects(req, [current, clone])

    def test_s3_pagination(self):
        session = PaginationSession()
        result = list_s3_objects("dbofs/netcdf/", session=session, endpoint="https://example.test")
        self.assertEqual(len(result), 2)
        self.assertEqual(session.calls[1]["continuation-token"], "next")

    def test_station_boundary_prefers_preceding_cycle_terminal(self):
        req = request(
            product="stations", variables=None, vertical_views=None,
            end_utc_exclusive="2026-07-20T00:06:00Z",
        )
        req.pop("variables"); req.pop("vertical_views")
        before = object_item("dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.stations.nowcast.nc")
        after = object_item("dbofs/netcdf/2026/07/20/dbofs.t06z.20260720.stations.nowcast.nc")
        selected = dbofs.select_objects(req, [before, after])
        self.assertEqual([item["key"] for item in selected["selected"]], [before["key"]])
        self.assertEqual(selected["duplicate_times"], [])

    def test_missing_policy_and_regulargrid_passthrough(self):
        with self.assertRaisesRegex(RuntimeError, "missing 1"):
            dbofs.select_objects(request(), [object_item("dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.n006.nc")])
        skipped = dbofs.select_objects(request(missing_policy="skip"), [object_item("dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.n006.nc")])
        self.assertEqual(skipped["missing_times"], ["2026-07-20T01:00:00Z"])
        regular = request(product="regulargrid")
        regular.pop("variables"); regular.pop("vertical_views")
        normalized = dbofs.validate_request(regular)
        self.assertNotIn("variables", normalized)

    def test_exact_plan(self):
        keys = [
            "dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.n006.nc",
            "dbofs/netcdf/2026/07/20/dbofs.t06z.20260720.fields.n001.nc",
        ]
        with tempfile.TemporaryDirectory() as td:
            report = dbofs.plan_request(request(), td, objects=[object_item(key, size=31 + i) for i, key in enumerate(keys)])
            self.assertEqual(report["total_bytes"], 63)
            self.assertEqual(report["required_free_bytes"], 252)
            self.assertEqual(report["routing_decision"], "local")
            unknown = dbofs.plan_request(
                request(end_utc_exclusive="2026-07-20T01:00:00Z"), td,
                objects=[object_item(keys[0], size=0)],
            )
            self.assertIsNone(unknown["total_bytes"])
            self.assertEqual(unknown["routing_decision"], "review")

            missing_etag = dbofs.plan_request(
                request(end_utc_exclusive="2026-07-20T01:00:00Z"), td,
                objects=[object_item(keys[0], size=31, etag="")],
            )
            self.assertEqual(missing_etag["routing_decision"], "review")
            self.assertEqual(missing_etag["routing_reason"], "source_metadata_incomplete")
            self.assertEqual(missing_etag["incomplete_source_metadata"][0]["field"], "etag")

    def test_direct_fetch_disabled_and_legacy_conflict(self):
        with self.assertRaisesRegex(RuntimeError, "review a plan"):
            dbofs.fetch_request(request(), "unused")
        current = object_item("dbofs/netcdf/2026/07/20/dbofs.t06z.20260720.fields.n006.nc")
        conflict = dict(current, key="dbofs/netcdf/2026/07/21/dbofs.t06z.20260720.fields.n006.nc", size=99, etag="other")
        original = dbofs.list_s3_objects
        try:
            dbofs.list_s3_objects = lambda prefix: [current, conflict] if "/2026/07/20/" in prefix else []
            with self.assertRaisesRegex(RuntimeError, "equal-rank conflicting"):
                dbofs._legacy_object("2026-07-20", "t06z", 0)
        finally:
            dbofs.list_s3_objects = original


class TransferTests(unittest.TestCase):
    def test_resume_multipart_and_cache_hit(self):
        payload = netcdf_payload()
        item = {"key": "dbofs/netcdf/test.nc", "url": "https://example.test/test.nc", "size": len(payload), "etag": "opaque-4", "last_modified": "now"}
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / "test.nc"
            destination.with_name("test.nc.part").write_bytes(payload[:5])
            write_json_atomic(destination.with_name("test.nc.part.json"), {
                "schema_version": "dbofs_partial_object_v1", "key": item["key"],
                "url": item["url"], "size": item["size"], "etag": item["etag"],
            })
            session = DownloadSession(payload, item["etag"])
            first = download_object(item, destination, session=session, chunk_size=3, schema_prefix="dbofs")
            self.assertTrue(first["resumed"]); self.assertEqual(first["resumed_from_bytes"], 5)
            self.assertEqual(session.headers_seen[0]["Range"], "bytes=5-")
            sidecar = json.loads(destination.with_name("test.nc.download.json").read_text())
            self.assertTrue(sidecar["etag_is_multipart"])
            second = download_object(item, destination, session=session, schema_prefix="dbofs")
            self.assertEqual(second["status"], "cache_hit")

    def test_corrupt_complete_partial_is_reset_then_redownloaded(self):
        payload = netcdf_payload()
        corrupt = b"X" * len(payload)
        item = {
            "key": "dbofs/netcdf/test.nc",
            "url": "https://example.test/test.nc",
            "size": len(payload),
            "etag": "opaque-etag",
            "last_modified": "now",
        }
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "test.nc"
            part = destination.with_name("test.nc.part")
            part.write_bytes(corrupt)
            write_json_atomic(destination.with_name("test.nc.part.json"), {
                "schema_version": "dbofs_partial_object_v1",
                "key": item["key"], "url": item["url"], "size": len(payload),
                "etag": item["etag"],
            })
            session = DownloadSession(payload, item["etag"])
            result = download_object(
                item, destination, session=session, max_attempts=2,
                schema_prefix="dbofs",
            )
            self.assertEqual(result["status"], "downloaded")
            self.assertEqual(result["retry_count"], 1)
            self.assertFalse(result["resumed"])
            self.assertEqual(result["resumed_from_bytes"], 0)
            self.assertTrue(result["discarded_invalid_partial"])
            self.assertEqual(len(session.headers_seen), 1)
            self.assertNotIn("Range", session.headers_seen[0])
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(part.exists())
            self.assertFalse(destination.with_name("test.nc.part.json").exists())

    def test_approved_plan_manifest_and_hash_gate(self):
        payload = netcdf_payload()
        item = object_item(
            "dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.n006.nc",
            size=len(payload), etag="opaque-etag",
        )
        req = request(end_utc_exclusive="2026-07-20T01:00:00Z")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_path = root / "approved.json"
            dbofs.plan_request(req, root, objects=[item], output=plan_path)
            manifest = dbofs.fetch_plan(plan_path, root, session=DownloadSession(payload, item["etag"]))
            self.assertEqual(manifest["approved_plan"]["path"], str(plan_path.resolve()))
            self.assertEqual(verify_transfers(root, dbofs.validate_request(req))["status"], "pass")
            plan_path.write_text("{}", encoding="utf-8")
            failed = verify_transfers(root, dbofs.validate_request(req))
            self.assertEqual(failed["status"], "fail")
            self.assertTrue(any("approved plan" in finding for finding in failed["failures"]))

    def test_fetch_recomputes_storage_gate_and_source_scope(self):
        payload = netcdf_payload()
        key = "dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.n006.nc"
        item = object_item(key, size=len(payload), etag="opaque-etag")
        req = request(end_utc_exclusive="2026-07-20T01:00:00Z")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = dbofs.plan_request(req, root, objects=[item])
            bad_required = json.loads(json.dumps(plan))
            bad_required["required_free_bytes"] += 1
            with self.assertRaisesRegex(RuntimeError, "exactly four times"):
                dbofs.fetch_plan(bad_required, root, session=DownloadSession(payload, item["etag"]))
            with mock.patch.object(
                core.shutil, "disk_usage",
                return_value=type("Usage", (), {"free": plan["required_free_bytes"]})(),
            ):
                with self.assertRaisesRegex(RuntimeError, "immediately before transfer"):
                    dbofs.fetch_plan(plan, root, session=DownloadSession(payload, item["etag"]))

            mutations = {
                "wrong prefix": {"key": key.replace("dbofs/netcdf/", "other/netcdf/")},
                "wrong daily date": {"key": key.replace("2026/07/20", "2026/07/19")},
                "wrong monthly month": {
                    "key": "dbofs/netcdf/202606/nos.dbofs.fields.n006.20260720.t00z.nc",
                },
                "wrong url": {"url": "https://example.test/evil.nc"},
                "missing etag": {"etag": ""},
                "missing last modified": {"last_modified": ""},
            }
            for label, changes in mutations.items():
                tampered = json.loads(json.dumps(plan))
                tampered["objects"][0].update(changes)
                with self.subTest(label=label):
                    with self.assertRaisesRegex(RuntimeError, "approved plan"):
                        dbofs.fetch_plan(
                            tampered, root,
                            session=DownloadSession(payload, item["etag"]),
                        )

    def test_transfer_response_etag_is_mandatory_and_exact(self):
        payload = netcdf_payload()
        item = {
            "key": "dbofs/netcdf/test.nc", "url": "https://example.test/test.nc",
            "size": len(payload), "etag": "planned", "last_modified": "now",
        }
        with tempfile.TemporaryDirectory() as temporary:
            for response_etag, finding in (("", "no ETag"), ("changed", "ETag changed")):
                destination = Path(temporary) / f"{response_etag or 'missing'}.nc"
                result = download_object(
                    item, destination,
                    session=DownloadSession(payload, response_etag),
                    max_attempts=1, schema_prefix="dbofs",
                )
                self.assertEqual(result["status"], "failed")
                self.assertTrue(any(finding in error for error in result["errors"]), result)

    def test_health_revalidates_approved_plan_contract_and_selected_keys(self):
        payload = netcdf_payload()
        item = object_item(
            "dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.n006.nc",
            size=len(payload), etag="opaque-etag",
        )
        req = request(end_utc_exclusive="2026-07-20T01:00:00Z")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "approved.json"
            manifest_path = root / "fetch_manifest.json"
            original_plan = dbofs.plan_request(req, root, objects=[item], output=plan_path)
            original_manifest = dbofs.fetch_plan(
                plan_path, root, session=DownloadSession(payload, item["etag"]),
            )

            def verify_mutation(mutated_plan, expected_finding, *, manifest_schema=None):
                write_json_atomic(plan_path, mutated_plan)
                mutated_manifest = json.loads(json.dumps(original_manifest))
                mutated_manifest["approved_plan"]["sha256"] = core.sha256_file(plan_path)
                if manifest_schema is not None:
                    mutated_manifest["approved_plan"]["schema_version"] = manifest_schema
                write_json_atomic(manifest_path, mutated_manifest)
                report = verify_transfers(root, dbofs.validate_request(req))
                self.assertEqual(report["status"], "fail")
                self.assertTrue(
                    any(expected_finding in finding for finding in report["failures"]),
                    report["failures"],
                )

            bad_schema = json.loads(json.dumps(original_plan))
            bad_schema["schema_version"] = "dbofs_download_estimate_v0"
            verify_mutation(bad_schema, "schema_version")

            verify_mutation(
                original_plan, "schema_version",
                manifest_schema="dbofs_download_estimate_v0",
            )

            bad_request = json.loads(json.dumps(original_plan))
            bad_request["request"]["missing_policy"] = "skip"
            verify_mutation(bad_request, "request does not match")

            bad_keys = json.loads(json.dumps(original_plan))
            bad_keys["objects"][0]["key"] = (
                "dbofs/netcdf/2026/07/20/dbofs.t06z.20260720.fields.n001.nc"
            )
            verify_mutation(bad_keys, "selected keys")

    def test_corrupt_exact_size_cache_with_matching_sha_is_redownloaded(self):
        payload = netcdf_payload()
        corrupt = b"X" * len(payload)
        item = {
            "key": "dbofs/netcdf/test.nc",
            "url": "https://example.test/test.nc",
            "size": len(payload),
            "etag": "opaque-etag",
            "last_modified": "now",
        }
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "test.nc"
            destination.write_bytes(corrupt)
            corrupt_digest = core.sha256_file(destination)
            write_json_atomic(destination.with_name("test.nc.download.json"), {
                "schema_version": "dbofs_cached_object_v1",
                "key": item["key"], "url": item["url"], "size": len(corrupt),
                "etag": item["etag"], "sha256": corrupt_digest,
            })
            result = download_object(
                item,
                destination,
                session=DownloadSession(payload, item["etag"]),
                max_attempts=1,
                schema_prefix="dbofs",
            )
            self.assertEqual(result["status"], "downloaded")
            self.assertNotEqual(result["sha256"], corrupt_digest)
            self.assertEqual(destination.read_bytes(), payload)
            core._validate_netcdf_payload(destination)

    def test_existing_transfer_lock_fails_fast_without_touching_partial(self):
        payload = netcdf_payload()
        item = {
            "key": "dbofs/netcdf/test.nc",
            "url": "https://example.test/test.nc",
            "size": len(payload),
            "etag": "opaque-etag",
            "last_modified": "now",
        }
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "test.nc"
            lock = destination.with_name("test.nc.transfer.lock")
            lock.write_text("owned by another process", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "another transfer owns"):
                download_object(
                    item, destination, session=DownloadSession(payload, item["etag"]),
                    max_attempts=1, schema_prefix="dbofs",
                )
            self.assertFalse(destination.with_name("test.nc.part").exists())
            self.assertTrue(lock.is_file())

    def test_lock_cleanup_preserves_replacement_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "test.nc"
            lock = destination.with_name("test.nc.transfer.lock")
            real_close = core.os.close

            def close_then_replace(descriptor):
                real_close(descriptor)
                lock.unlink()
                lock.write_text(json.dumps({"owner_token": "replacement-owner"}),
                                encoding="utf-8")

            with mock.patch.object(core.os, "close", side_effect=close_then_replace):
                with core._destination_lock(destination):
                    pass
            self.assertTrue(lock.is_file())
            self.assertEqual(
                json.loads(lock.read_text(encoding="utf-8"))["owner_token"],
                "replacement-owner",
            )

    def test_cache_is_rechecked_after_lock_acquisition(self):
        payload = netcdf_payload()
        item = {
            "key": "dbofs/netcdf/test.nc",
            "url": "https://example.test/test.nc",
            "size": len(payload),
            "etag": "opaque-etag",
            "last_modified": "now",
        }
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "test.nc"
            session = DownloadSession(payload, item["etag"])
            original_lock = core._destination_lock

            @contextmanager
            def completing_lock(_destination):
                destination.write_bytes(payload)
                write_json_atomic(destination.with_name("test.nc.download.json"), {
                    "schema_version": "dbofs_cached_object_v1",
                    "key": item["key"], "url": item["url"], "size": len(payload),
                    "etag": item["etag"], "sha256": core.sha256_file(destination),
                })
                yield destination.with_name("test.nc.transfer.lock")

            try:
                core._destination_lock = completing_lock
                result = download_object(
                    item, destination, session=session, max_attempts=1,
                    schema_prefix="dbofs",
                )
            finally:
                core._destination_lock = original_lock
            self.assertEqual(result["status"], "cache_hit")
            self.assertEqual(session.headers_seen, [])

    def test_netcdf_metadata_validation_is_serialized(self):
        payload = netcdf_payload()
        state = {"active": 0, "maximum": 0, "opens": 0}
        state_lock = threading.Lock()

        class BlockingDataset:
            def __init__(self, *_args, **_kwargs):
                self.dimensions = {}
                self.variables = {}

            def __enter__(self):
                with state_lock:
                    state["active"] += 1
                    state["opens"] += 1
                    state["maximum"] = max(state["maximum"], state["active"])
                time.sleep(0.02)
                return self

            def __exit__(self, *_args):
                with state_lock:
                    state["active"] -= 1

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test.nc"
            source.write_bytes(payload)
            with mock.patch.object(netCDF4, "Dataset", BlockingDataset):
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [pool.submit(core._validate_netcdf_payload, source)
                               for _ in range(8)]
                    for future in futures:
                        future.result()
        self.assertEqual(state["opens"], 8)
        self.assertEqual(state["maximum"], 1)


class RomsMathTests(unittest.TestCase):
    def test_transforms_thickness_and_missing_layer_average(self):
        h = np.array([[10.0]]); zeta = np.array([[0.2]])
        sr = np.array([-0.8, -0.4, -0.1]); sw = np.array([-1.0, -0.6, -0.2, 0.0])
        for transform in (1, 2):
            z = roms_depths(sr, sr, 1.0, h, zeta, transform)
            self.assertEqual(z.shape, (3, 1, 1))
            thickness, closure = layer_thickness(sw, sw, 1.0, h, zeta, transform)
            self.assertLess(closure, 1e-10)
            self.assertAlmostEqual(float(thickness.sum()), 10.2, places=8)
            data = np.array([[[10.0]], [[np.nan]], [[30.0]]])
            average = weighted_vertical_average(data, thickness)
            expected = (10 * thickness[0, 0, 0] + 30 * thickness[2, 0, 0]) / (thickness[0, 0, 0] + thickness[2, 0, 0])
            self.assertAlmostEqual(float(average[0, 0]), float(expected), places=8)

    def test_destagger_and_rotation(self):
        u = np.full((2, 3), 2.0); v = np.zeros((1, 4)); mask = np.ones((2, 4), dtype=bool)
        u_rho, v_rho = destagger_to_rho(u, v, mask)
        east, north, speed = rotate_to_earth(u_rho, v_rho, np.full(mask.shape, math.pi / 2))
        self.assertTrue(np.allclose(east, 0, atol=1e-12)); self.assertTrue(np.allclose(north, 2))
        self.assertTrue(np.allclose(speed, 2))

    def test_strict_geometry_angle_masks_and_vertical_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.nc"
            write_fixture(valid, 0)
            with netCDF4.Dataset(valid) as dataset:
                snapshot = geometry_snapshot(dataset)
            self.assertEqual(
                snapshot["angle_contract"]["convention"],
                "xi_axis_counterclockwise_from_east_radians",
            )

            cases = []
            for name in (
                "angle_units", "angle_semantics", "angle_nonfinite", "mask_binary",
                "sigma_nonfinite", "stretching_nonmonotone", "vtransform",
                "vstretching", "hc",
            ):
                path = root / f"{name}.nc"
                write_fixture(path, 0)
                cases.append((name, path))
            with netCDF4.Dataset(cases[0][1], "r+") as ds:
                ds.variables["angle"].units = "degrees"
            with netCDF4.Dataset(cases[1][1], "r+") as ds:
                ds.variables["angle"].standard_name = "unknown_angle"
                ds.variables["angle"].long_name = "grid rotation"
            with netCDF4.Dataset(cases[2][1], "r+") as ds:
                ds.variables["angle"][1, 1] = np.nan
            with netCDF4.Dataset(cases[3][1], "r+") as ds:
                ds.variables["mask_u"][1, 1] = 2
            with netCDF4.Dataset(cases[4][1], "r+") as ds:
                ds.variables["s_rho"][1] = np.nan
            with netCDF4.Dataset(cases[5][1], "r+") as ds:
                ds.variables["Cs_r"][:] = [-0.8, -0.8, -0.1]
            with netCDF4.Dataset(cases[6][1], "r+") as ds:
                ds.variables["Vtransform"].assignValue(3)
            with netCDF4.Dataset(cases[7][1], "r+") as ds:
                ds.variables["Vstretching"].assignValue(0)
            with netCDF4.Dataset(cases[8][1], "r+") as ds:
                ds.variables["hc"].assignValue(np.nan)
            expected = {
                "angle_units": "angle units", "angle_semantics": "angle metadata",
                "angle_nonfinite": "angle must be finite", "mask_binary": "binary 0/1",
                "sigma_nonfinite": "finite values", "stretching_nonmonotone": "monotonic",
                "vtransform": "Vtransform must be 1 or 2", "vstretching": "positive integer",
                "hc": "finite scalar",
            }
            for name, path in cases:
                with self.subTest(name=name):
                    with netCDF4.Dataset(path) as dataset:
                        with self.assertRaisesRegex(ValueError, expected[name]):
                            geometry_snapshot(dataset)

            missing = root / "missing_mask.nc"
            write_fixture(missing, 0, omit_mask="mask_v")
            with netCDF4.Dataset(missing) as dataset:
                with self.assertRaisesRegex(ValueError, "missing variables: mask_v"):
                    geometry_snapshot(dataset)


class ExtractionTests(unittest.TestCase):
    def test_synthetic_extract_reversed_sigma_health_and_legacy_depth(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            paths = [directory / "hour0.nc", directory / "hour1.nc"]
            write_fixture(paths[0], 0, vtransform=1, reversed_sigma=True, angle=math.pi / 2)
            write_fixture(paths[1], 1, vtransform=1, reversed_sigma=True, angle=math.pi / 2)
            output = directory / "dbofs_fields.nc"
            manifest = extract_fields(request(), paths, output, dbofs.CONFIG)
            self.assertEqual(len(manifest["records"]), 2)
            self.assertLess(manifest["maximum_absolute_thickness_closure_m"], 1e-10)
            with netCDF4.Dataset(output) as ds:
                self.assertEqual(ds.schema_version, "roms_compact_fields_v1")
                self.assertEqual(ds.angle_convention, "xi_axis_counterclockwise_from_east_radians")
                self.assertEqual(
                    ds.variables["angle"].angle_convention,
                    "xi_axis_counterclockwise_from_east_radians",
                )
                self.assertTrue(np.allclose(ds.variables["salinity_surface"][:, 1:, 1:], 30.0))
                self.assertTrue(np.allclose(ds.variables["current_speed_surface"][:, 1:, 1:], 3.0))
                self.assertTrue(np.allclose(ds.variables["eastward_velocity_surface"][:, 1:, 1:], 0.0, atol=1e-6))
                self.assertTrue(np.allclose(ds.variables["northward_velocity_surface"][:, 1:, 1:], 3.0))
                # Unequal W-level thickness gives 18 for complete salinity columns.
                self.assertTrue(np.allclose(ds.variables["salinity_depth_average"][:, 1, 1], 18.0, atol=1e-5))
            health = compact_health(output, dbofs.validate_request(request()))
            self.assertEqual(health["status"], "pass", health["critical_findings"])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                legacy = dbofs.roms_depths_2d(np.array([-0.5]), np.array([-0.5]), 1.0, np.array([[10.0]]), vtransform=1)
            self.assertEqual(legacy.shape, (1, 1, 1)); self.assertTrue(any(item.category is DeprecationWarning for item in caught))
            self.assertTrue(dbofs._dbofs_url("2026-07-20", "t06z", 0).endswith("dbofs.t06z.20260720.fields.n006.nc"))

    def test_requested_variable_schema_expectations_and_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first.nc", root / "second.nc"
            write_fixture(first, 0)
            write_fixture(second, 1)
            with netCDF4.Dataset(second, "r+") as dataset:
                dataset.variables["salt"].grid = "different-grid"
            with self.assertRaisesRegex(ValueError, "schema/dimension drift.*salt"):
                extract_fields(request(), [first, second], root / "drift.nc", dbofs.CONFIG)

            wrong = root / "wrong_dimensions.nc"
            write_fixture(wrong, 0)
            with netCDF4.Dataset(wrong, "r+") as dataset:
                temp = dataset.createVariable(
                    "temp", "f4", ("ocean_time", "eta_u", "xi_u"), fill_value=1e37,
                )
                temp.grid = "grid"; temp.location = "edge1"; temp[:] = 1.0
            with self.assertRaisesRegex(ValueError, "incompatible ROMS dimensions"):
                extract_fields(
                    request(variables=["temp"]), [wrong], root / "wrong.nc", dbofs.CONFIG,
                )

    def test_unaligned_exact_times_requested_variables_and_native_health(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "hour0.nc"
            write_fixture(source, 0)
            req = request(
                start_utc="2026-07-19T23:50:00Z",
                end_utc_exclusive="2026-07-20T00:10:00Z",
                vertical_views=["surface"],
            )
            output = root / "compact.nc"
            manifest = extract_fields(req, [source], output, dbofs.CONFIG)
            self.assertEqual(len(manifest["records"]), 1)
            self.assertEqual(len(manifest["finite_wet_coverage"]["u_native_surface"]), 1)
            healthy = compact_health(output, dbofs.validate_request(req))
            self.assertEqual(healthy["status"], "pass", healthy["critical_findings"])
            self.assertIn("u_native_surface", healthy["variables"])
            wrong_time = compact_health(output, dbofs.validate_request(request(
                start_utc="2026-07-20T01:00:00Z", end_utc_exclusive="2026-07-20T02:00:00Z",
                vertical_views=["surface"],
            )))
            self.assertEqual(wrong_time["status"], "fail")
            missing_temp = compact_health(output, dbofs.validate_request(request(
                start_utc="2026-07-19T23:50:00Z", end_utc_exclusive="2026-07-20T00:10:00Z",
                variables=["temp"], vertical_views=["surface"],
            )))
            self.assertEqual(missing_temp["status"], "fail")
            self.assertTrue(any("requested variables" in item for item in missing_temp["critical_findings"]))

    def test_manifest_discovered_custom_output_health(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fixture = root / "fixture.nc"
            write_fixture(fixture, 0)
            payload = fixture.read_bytes()
            req = request(end_utc_exclusive="2026-07-20T01:00:00Z", vertical_views=["surface"])
            item = object_item(
                "dbofs/netcdf/2026/07/20/dbofs.t00z.20260720.fields.n006.nc",
                size=len(payload), etag="fixture-etag",
            )
            plan_path = root / "download_estimate.json"
            dbofs.plan_request(req, root, objects=[item], output=plan_path)
            dbofs.fetch_plan(plan_path, root, session=DownloadSession(payload, item["etag"]))
            custom = root / "custom-name.nc"
            dbofs.extract_request(req, root, output=custom)
            report = dbofs.evaluate_health(req, root)
            self.assertEqual(report["status"], "pass", report["critical_findings"])
            self.assertEqual(report["compact_output"]["path"], str(custom.resolve()))

    def test_station_cropped_selection_artifact_and_exact_window(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "dbofs.t00z.20260720.stations.nowcast.nc"
            artifact = root / "station_cropped_selection.json"
            write_station_fixture(source)
            req = dbofs.validate_request({
                "schema_version": "dbofs_request_v1",
                "start_utc": "2026-07-20T00:00:00Z",
                "end_utc_exclusive": "2026-07-20T00:06:00Z",
                "product": "stations", "guidance": "nowcast",
                "missing_policy": "error", "cache_policy": "keep", "max_workers": 1,
            })
            report = raw_consistency([source], req, selection_output=artifact)
            self.assertEqual(report["status"], "pass", report["critical_findings"])
            selected = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(selected["expected_times_utc"], ["2026-07-20T00:00:00Z"])
            self.assertEqual(len(selected["selected_records"]), 1)
            wrong = dbofs.validate_request({
                **req, "start_utc": "2030-01-01T00:00:00Z",
                "end_utc_exclusive": "2030-01-01T00:06:00Z",
            })
            self.assertEqual(raw_consistency([source], wrong)["status"], "fail")


if __name__ == "__main__":
    unittest.main(verbosity=2)
