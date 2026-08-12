#!/usr/bin/env python3
"""Offline regression tests for the CBOFS AWS connector and ROMS core."""

from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cbofs_fetcher as cbofs
import roms_aws_core as core
import roms_processing as roms


def request(**updates):
    value = {
        "schema_version": "cbofs_request_v1",
        "start_utc": "2026-07-20T00:00:00Z",
        "end_utc_exclusive": "2026-07-20T02:00:00Z",
        "product": "fields",
        "guidance": "nowcast",
        "variables": ["zeta", "salt", "u", "v"],
        "vertical_views": ["surface"],
        "missing_policy": "error",
        "cache_policy": "keep",
        "max_workers": 2,
    }
    value.update(updates)
    return value


def object_for(key: str, size: int = 100, etag: str = "opaque-8"):
    parsed = cbofs.parse_object_key(key)
    if parsed is None:
        raise AssertionError(key)
    return {**parsed, "size": size, "etag": etag, "last_modified": "2026-07-21T00:00:00Z",
            "url": core.S3_ENDPOINT + "/" + key}


def authorize_aws_fixture_plan(path: Path) -> dict:
    plan = core.read_json(path)
    plan["source_attempts"] = [{"source_id": "aws_operational", "status": "success"}]
    core.write_json_atomic(path, plan)
    return plan


class FakeResponse:
    def __init__(self, content=b"", *, status=200, headers=None):
        self.content = content
        self.status_code = status
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1024):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset:offset + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        params = kwargs.get("params")
        if (isinstance(params, dict) and params.get("list-type") == "2"
                and str(params.get("prefix", "")).lower().endswith(".nc")):
            key = str(params["prefix"])
            response = self.responses[0]
            etag = str(response.headers.get("ETag", ""))
            size = len(response.content)
            xml = (
                "<ListBucketResult>"
                f"<Contents><Key>{key}</Key><LastModified>2026-07-21T00:00:00Z</LastModified>"
                f"<ETag>{etag}</ETag><Size>{size}</Size></Contents>"
                "<IsTruncated>false</IsTruncated></ListBucketResult>"
            ).encode()
            return FakeResponse(xml)
        if not self.responses:
            raise AssertionError("unexpected request")
        return self.responses.pop(0)


def s3_xml(keys, *, truncated=False, token=None):
    contents = "".join(
        f"<Contents><Key>{key}</Key><LastModified>2026-07-20T00:00:00Z</LastModified>"
        f"<ETag>&quot;etag-{index}&quot;</ETag><Size>{index + 1}</Size>"
        f"<StorageClass>STANDARD</StorageClass></Contents>"
        for index, key in enumerate(keys)
    )
    continuation = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return (f"<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
            f"{contents}<IsTruncated>{str(truncated).lower()}</IsTruncated>{continuation}"
            f"</ListBucketResult>").encode()


def create_fixture(path: Path, stamp: datetime, *, vtransform=1, lon_offset=0.0,
                   reverse_sigma=False, missing_layer=False,
                   angle_units="radians", angle_standard_name="grid_angle_of_rotation_from_east_to_y",
                   angle_long_name="angle between XI-axis and EAST", angle_value=0.0,
                   angle_on_u_grid=False, source_calendar="gregorian"):
    import netCDF4
    import numpy as np
    eta, xi, n = 4, 5, 3
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("ocean_time", 1)
        ds.createDimension("s_rho", n)
        ds.createDimension("s_w", n + 1)
        ds.createDimension("eta_rho", eta)
        ds.createDimension("xi_rho", xi)
        ds.createDimension("eta_u", eta)
        ds.createDimension("xi_u", xi - 1)
        ds.createDimension("eta_v", eta - 1)
        ds.createDimension("xi_v", xi)
        y, x = np.mgrid[0:eta, 0:xi]
        lon = -77 + x * .1 + y * .01 + lon_offset
        lat = 36 + y * .1 + x * .005
        mask = np.ones((eta, xi), dtype=np.int8)
        mask[0, 0] = 0
        h = 10 + x + y
        angle = np.full_like(lon, angle_value)
        for name, dims, values, dtype in (
            ("lon_rho", ("eta_rho", "xi_rho"), lon, "f8"),
            ("lat_rho", ("eta_rho", "xi_rho"), lat, "f8"),
            ("h", ("eta_rho", "xi_rho"), h, "f8"),
            ("angle", (("eta_u", "xi_u") if angle_on_u_grid else ("eta_rho", "xi_rho")),
             (angle[:, :-1] if angle_on_u_grid else angle), "f8"),
            ("mask_rho", ("eta_rho", "xi_rho"), mask, "i1"),
            ("lon_u", ("eta_u", "xi_u"), (lon[:, :-1] + lon[:, 1:]) / 2, "f8"),
            ("lat_u", ("eta_u", "xi_u"), (lat[:, :-1] + lat[:, 1:]) / 2, "f8"),
            ("mask_u", ("eta_u", "xi_u"), mask[:, :-1] * mask[:, 1:], "i1"),
            ("lon_v", ("eta_v", "xi_v"), (lon[:-1] + lon[1:]) / 2, "f8"),
            ("lat_v", ("eta_v", "xi_v"), (lat[:-1] + lat[1:]) / 2, "f8"),
            ("mask_v", ("eta_v", "xi_v"), mask[:-1] * mask[1:], "i1"),
        ):
            variable = ds.createVariable(name, dtype, dims)
            variable[:] = values
            if name == "angle":
                variable.units = angle_units
                if angle_standard_name is not None:
                    variable.standard_name = angle_standard_name
                if angle_long_name is not None:
                    variable.long_name = angle_long_name
        s_rho = np.array([-.8333333333, -.5, -.1666666667])
        s_w = np.array([-1., -.6666666667, -.3333333333, 0.])
        if reverse_sigma:
            s_rho, s_w = s_rho[::-1], s_w[::-1]
        cs_w = s_w.copy()
        cs_w[np.argmin(np.abs(cs_w))] = 2.890124535033231e-17
        for name, dims, values in (("s_rho", ("s_rho",), s_rho),
                                   ("Cs_r", ("s_rho",), s_rho),
                                   ("s_w", ("s_w",), s_w),
                                   ("Cs_w", ("s_w",), cs_w)):
            coordinate = ds.createVariable(name, "f8", dims)
            coordinate[:] = values
            coordinate.valid_min = -1.0
            coordinate.valid_max = 0.0
        ds.Vtransform = vtransform
        ds.Vstretching = 1
        ds.hc = 2.0
        time = ds.createVariable("ocean_time", "f8", ("ocean_time",))
        time.units = "seconds since 1970-01-01 00:00:00 UTC"
        time.calendar = source_calendar
        time[:] = stamp.replace(tzinfo=timezone.utc).timestamp() + 20.0
        zeta = ds.createVariable("zeta", "f4", ("ocean_time", "eta_rho", "xi_rho"), fill_value=1e37)
        zeta.standard_name = "sea_surface_height_above_geoid"
        zeta.units = "m"
        zeta[0] = np.where(mask == 1, .2, 1e37)
        salt = ds.createVariable("salt", "f4", ("ocean_time", "s_rho", "eta_rho", "xi_rho"), fill_value=1e37)
        salt.standard_name = "sea_water_practical_salinity"
        u = ds.createVariable("u", "f4", ("ocean_time", "s_rho", "eta_u", "xi_u"), fill_value=1e37)
        u.standard_name = "sea_water_x_velocity"
        u.units = "m s-1"
        v = ds.createVariable("v", "f4", ("ocean_time", "s_rho", "eta_v", "xi_v"), fill_value=1e37)
        v.standard_name = "sea_water_y_velocity"
        v.units = "m s-1"
        layer_values = [10., 20., 30.] if not reverse_sigma else [30., 20., 10.]
        u_values = [1., 2., 3.] if not reverse_sigma else [3., 2., 1.]
        for layer in range(n):
            salt[0, layer] = np.where(mask == 1, layer_values[layer], 1e37)
            u[0, layer] = np.where(mask[:, :-1] * mask[:, 1:] == 1, u_values[layer], 1e37)
            v[0, layer] = np.where(mask[:-1] * mask[1:] == 1, 0., 1e37)
        if missing_layer:
            salt[0, 1, 2, 2] = 1e37


def create_run_fixture(root: Path, req: dict | None = None):
    normalized = cbofs.validate_request(req or request(end_utc_exclusive="2026-07-20T01:00:00Z"))
    key = "cbofs/netcdf/2026/07/20/cbofs.t00z.20260720.fields.n006.nc"
    source = root / "cache" / "raw" / "2026" / "07" / "20" / Path(key).name
    source.parent.mkdir(parents=True, exist_ok=True)
    create_fixture(source, datetime(2026, 7, 20, 0))
    size, digest = source.stat().st_size, core.sha256_file(source)
    item = core._decorate_source(cbofs.CONFIG, object_for(key, size=size, etag="fixture-etag"))
    core.write_json_atomic(source.with_name(source.name + ".download.json"), {
        "schema_version": "roms_cached_object_v1", "model": "cbofs",
        "source_id": item["source_id"], "source_identity": item["source_identity"],
        "key": key, "url": item["url"], "size": size, "etag": item["etag"],
        "etag_is_multipart": False, "etag_semantics": "opaque_provenance",
        "last_modified": item["last_modified"], "sha256": digest,
        "completed_utc": "2026-07-20T00:00:00Z",
    })
    outcome = {"key": key, "url": item["url"], "local_path": str(source.resolve()),
               "status": "downloaded", "size": size, "etag": item["etag"],
               "sha256": digest, "resumed": False, "resumed_from_bytes": 0,
               "retry_count": 0, "source": item}
    plan_path = root / "download_estimate.json"
    plan = {
        "schema_version": "cbofs_download_estimate_v2", "request": normalized,
        "request_sha256": core.canonical_json_sha256(normalized),
        "objects_sha256": core.source_objects_sha256([item]),
        "objects": [item], "object_count": 1, "total_bytes": size,
        "source_totals": {
            "aws_operational": {"object_count": 1, "bytes": size},
            "ncei_long_term": {"object_count": 0, "bytes": 0},
        },
        "source": {"policy": normalized["source_policy"],
                   "access": "anonymous_https_listobjectsv2"},
        "source_attempts": [{"source_id": "aws_operational", "status": "success"}],
    }
    core.write_json_atomic(plan_path, plan)
    core.write_json_atomic(root / "fetch_manifest.json", {
        "schema_version": "cbofs_fetch_manifest_v2", "created_utc": "2026-07-20T00:00:00Z",
        "source_policy": normalized["source_policy"],
        "request": normalized,
        "approved_plan": {
            "path": str(plan_path.resolve()), "sha256": core.sha256_file(plan_path),
            "schema_version": plan["schema_version"],
            "request_sha256": core.canonical_json_sha256(normalized),
            "objects_sha256": core.source_objects_sha256([item]),
            "object_count": 1, "total_bytes": size,
        },
        "outcomes": [outcome],
        "counts": {"objects": 1, "downloaded": 1, "cache_hits": 0, "failed": 0, "resumed": 0},
        "source_provenance": {"bucket": core.BUCKET, "endpoint": core.S3_ENDPOINT,
                              "access": "anonymous_https"},
    })
    return normalized, source


def extract_bound_run(root: Path, req: dict, output: Path):
    paths, binding = core.verified_manifest_inputs(
        cbofs.CONFIG, root / "fetch_manifest.json", request=req, run_dir=root)
    return cbofs.extract_request(req, paths, output, provenance=binding)


def netcdf_payload() -> bytes:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "fixture.nc"
        create_fixture(path, datetime(2026, 7, 20, 1))
        return path.read_bytes()


class RequestAndInventoryTests(unittest.TestCase):
    def test_v1_migrates_to_v2_and_source_policy_is_strict(self):
        normalized = cbofs.validate_request(request())
        self.assertEqual(normalized["schema_version"], "cbofs_request_v2")
        self.assertEqual(normalized["source_policy"], "aws_then_ncei")
        with self.assertRaisesRegex(ValueError, "source_policy"):
            cbofs.validate_request(request(source_policy="silent_fallback"))

    def test_ordered_fallback_queries_ncei_only_for_aws_gaps(self):
        req = cbofs.validate_request(request(
            schema_version="cbofs_request_v2", start_utc="2026-07-20T01:00:00Z",
            end_utc_exclusive="2026-07-20T02:00:00Z"))
        aws = core._decorate_source(cbofs.CONFIG, object_for(
            "cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.fields.n001.nc"))
        calls = []
        def complete(_config, _request, source_id, **_kwargs):
            calls.append(source_id)
            return ([aws] if source_id == "aws_operational" else []), {
                "source_id": source_id, "status": "success", "prefixes": [],
                "object_count": 1 if source_id == "aws_operational" else 0, "error": None}
        with mock.patch.object(core, "_discover_one_source", side_effect=complete):
            _, evidence = core.discover_objects_with_evidence(cbofs.CONFIG, req)
        self.assertEqual(calls, ["aws_operational"])
        self.assertFalse(evidence["fallback_triggered"])

    def test_request_contract_and_aliases(self):
        validated = cbofs.validate_request(request(variables=["salinity", "u", "v"]))
        self.assertEqual(validated["variables"], ["salt", "u", "v"])
        with self.assertRaisesRegex(ValueError, "passthrough-only"):
            cbofs.validate_request(request(product="stations", variables=["salt"]))
        with self.assertRaisesRegex(ValueError, "required"):
            cbofs.validate_request(request(guidance="forecast"))
        forecast = request(guidance="forecast", run_cycle_utc="2026-07-20T00:00:00Z")
        self.assertEqual(cbofs.validate_request(forecast)["run_cycle_utc"], "2026-07-20T00:00:00Z")

    def test_filename_parsing_and_time_formula(self):
        n001 = cbofs.parse_object_key("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.fields.n001.nc")
        n006 = cbofs.parse_object_key("cbofs/netcdf/202607/nos.cbofs.fields.n006.20260720.t06z.nc")
        f001 = cbofs.parse_object_key("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.fields.f001.nc")
        station = cbofs.parse_object_key("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.stations.nowcast.nc")
        regular = cbofs.parse_object_key("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.regulargrid.n006.nc")
        self.assertEqual(n001["valid_time"], "2026-07-20T01:00:00Z")
        self.assertEqual(n006["valid_time"], "2026-07-20T06:00:00Z")
        self.assertEqual(f001["valid_time"], "2026-07-20T07:00:00Z")
        self.assertTrue(station["aggregate"])
        self.assertEqual(regular["product"], "regulargrid")

    def test_invalid_cycles_leads_and_product_packaging_are_rejected(self):
        invalid = [
            "cbofs.t03z.20260720.fields.n006.nc",
            "cbofs.t06z.20260720.fields.n000.nc",
            "cbofs.t06z.20260720.fields.n007.nc",
            "cbofs.t06z.20260720.fields.f000.nc",
            "cbofs.t99z.20260720.fields.n006.nc",
            "cbofs.t06z.20260230.fields.n006.nc",
            "cbofs.t06z.20260720.stations.n006.nc",
            "cbofs.t06z.20260720.fields.nowcast.nc",
        ]
        for name in invalid:
            with self.subTest(name=name):
                self.assertIsNone(cbofs.parse_object_key(f"cbofs/netcdf/2026/07/20/{name}"))

    def test_s3_pagination(self):
        session = FakeSession([
            FakeResponse(s3_xml(["cbofs/a"], truncated=True, token="next")),
            FakeResponse(s3_xml(["cbofs/b"])),
        ])
        values = core.list_s3_objects("cbofs/", session=session, endpoint="https://example.invalid")
        self.assertEqual([item["key"] for item in values], ["cbofs/a", "cbofs/b"])
        self.assertEqual(session.calls[1][1]["params"]["continuation-token"], "next")

    def test_station_boundary_prefers_preceding_cycle(self):
        import netCDF4
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for cycle in (0, 6):
                path = root / f"cbofs.t{cycle:02d}z.20260720.stations.nowcast.nc"
                with netCDF4.Dataset(path, "w") as dataset:
                    dataset.createDimension("ocean_time", 1)
                    time = dataset.createVariable("ocean_time", "f8", ("ocean_time",))
                    time.units = "seconds since 1970-01-01 00:00:00 UTC"
                    time[:] = datetime(2026, 7, 20, 6, tzinfo=timezone.utc).timestamp()
                paths.append(path)
            station_request = cbofs.validate_request({
                "schema_version": "cbofs_request_v1",
                "start_utc": "2026-07-20T06:00:00Z",
                "end_utc_exclusive": "2026-07-20T06:06:00Z",
                "product": "stations", "guidance": "nowcast",
                "missing_policy": "error", "cache_policy": "keep", "max_workers": 1,
            })
            audit = core.audit_time_records(cbofs.CONFIG, station_request, paths)
            self.assertEqual(audit["selected_count"], 1)
            self.assertIn("t00z", audit["duplicate_records"][0]["preferred"])

    def test_station_boundary_triggers_ncei_discovery_for_preceding_terminal(self):
        station_request = cbofs.validate_request({
            "schema_version": "cbofs_request_v2",
            "start_utc": "2026-07-20T00:00:00Z",
            "end_utc_exclusive": "2026-07-20T00:06:00Z",
            "product": "stations", "guidance": "nowcast",
            "source_policy": "aws_then_ncei", "missing_policy": "error",
            "cache_policy": "keep", "max_workers": 1,
        })
        aws = core._decorate_source(
            cbofs.CONFIG,
            object_for("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.stations.nowcast.nc"),
            "aws_operational",
        )
        descriptor = core.archive_sources.get_source_descriptor("ncei_long_term", "cbofs")
        ncei_key = (
            descriptor["root_prefix"]
            + "2026/07/nos.cbofs.stations.nowcast.20260720.t00z.nc"
        )
        parsed = cbofs.parse_object_key(ncei_key)
        self.assertIsNotNone(parsed)
        ncei = core._decorate_source(
            cbofs.CONFIG,
            {**parsed, "size": 101, "etag": "ncei-station",
             "last_modified": "2026-07-22T00:00:00Z"},
            "ncei_long_term",
        )

        def discover(_config, _request, source_id, **_kwargs):
            values = [aws] if source_id == "aws_operational" else [ncei]
            return values, {"source_id": source_id, "status": "success",
                            "prefixes": [], "object_count": 1, "error": None}

        with mock.patch.object(core, "_discover_one_source", side_effect=discover) as mocked:
            combined, evidence = core.discover_objects_with_evidence(
                cbofs.CONFIG, station_request)
        self.assertEqual(mocked.call_count, 2)
        self.assertTrue(evidence["fallback_triggered"])
        self.assertEqual(evidence["fallback_reason"],
                         "aws_scientific_precedence_unresolved")
        self.assertEqual(evidence["scientific_precedence_before_fallback"],
                         ["2026-07-20T00:00:00Z"])
        selection = cbofs.select_objects(station_request, combined)
        self.assertEqual([item["key"] for item in selection["selected"]], [ncei_key])

    def test_v1_cannot_override_migrated_source_policy(self):
        with self.assertRaisesRegex(ValueError, "v1 always migrates"):
            cbofs.validate_request(request(source_policy="ncei_only"))
        explicit_v2 = cbofs.validate_request(request(
            schema_version="cbofs_request_v2", source_policy="ncei_only"))
        self.assertEqual(explicit_v2["source_policy"], "ncei_only")

    def test_duplicate_preference_and_conflict(self):
        current = object_for("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.fields.n001.nc")
        legacy = object_for("cbofs/netcdf/202607/nos.cbofs.fields.n001.20260720.t06z.nc")
        selection = cbofs.select_objects(request(start_utc="2026-07-20T01:00:00Z"), [legacy, current])
        self.assertEqual(selection["selected"][0]["key"], current["key"])
        legacy_peer = object_for("cbofs/netcdf/202607/cbofs.fields.n001.20260720.t06z.nc")
        identical = cbofs.select_objects(request(start_utc="2026-07-20T01:00:00Z"),
                                         [legacy, legacy_peer])
        self.assertEqual(identical["selected"][0]["key"], min(legacy["key"], legacy_peer["key"]))
        self.assertEqual(identical["duplicate_objects"][0]["rejection_reason"],
                         "same_rank_identical_remote_metadata")
        conflict = dict(legacy_peer, etag="different-etag")
        with self.assertRaisesRegex(RuntimeError, "same-rank"):
            cbofs.select_objects(request(start_utc="2026-07-20T01:00:00Z"), [legacy, conflict])

    def test_missing_policy_and_exact_estimate(self):
        one = object_for("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.fields.n001.nc", 123)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            cbofs.select_objects(request(start_utc="2026-07-20T01:00:00Z",
                                         end_utc_exclusive="2026-07-20T03:00:00Z"), [one])
        with tempfile.TemporaryDirectory() as temporary:
            report = cbofs.plan_request(request(start_utc="2026-07-20T01:00:00Z",
                                                end_utc_exclusive="2026-07-20T02:00:00Z"),
                                        temporary, objects=[one])
            self.assertEqual(report["total_bytes"], 123)
            self.assertEqual(report["source_totals"], {
                "aws_operational": {"object_count": 1, "bytes": 123},
                "ncei_long_term": {"object_count": 0, "bytes": 0},
            })
            self.assertEqual(report["required_free_bytes"], 492)
            self.assertEqual(report["routing_decision"], "local")

    def test_mixed_source_plan_totals_are_exact(self):
        aws = object_for(
            "cbofs/netcdf/2026/07/20/cbofs.t00z.20260720.fields.n006.nc", 11)
        descriptor = core.archive_sources.get_source_descriptor("ncei_long_term", "cbofs")
        key = descriptor["root_prefix"] + "2026/07/nos.cbofs.fields.n001.20260720.t06z.nc"
        ncei = {**cbofs.parse_object_key(key), "source_id": "ncei_long_term",
                "size": 13, "etag": "ncei", "last_modified": "2026-07-21T00:00:00Z"}
        with tempfile.TemporaryDirectory() as temporary:
            report = cbofs.plan_request(request(
                schema_version="cbofs_request_v2", source_policy="aws_then_ncei"),
                temporary, objects=[aws, ncei])
        self.assertEqual(report["source_totals"], {
            "aws_operational": {"object_count": 1, "bytes": 11},
            "ncei_long_term": {"object_count": 1, "bytes": 13},
        })
        self.assertEqual(report["total_bytes"], 24)

    def test_non_positive_or_unknown_sizes_never_approve_local_fetch(self):
        for bad_size in (None, -1, 0, True, "100"):
            with self.subTest(size=bad_size), tempfile.TemporaryDirectory() as temporary:
                item = object_for("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.fields.n001.nc")
                item["size"] = bad_size
                report = cbofs.plan_request(
                    request(start_utc="2026-07-20T01:00:00Z",
                            end_utc_exclusive="2026-07-20T02:00:00Z"),
                    temporary, objects=[item])
                self.assertEqual(report["routing_decision"], "review")
                self.assertIsNone(report["total_bytes"])
                self.assertIsNone(report["required_free_bytes"])

    def test_source_scope_and_remote_identity_gate_local_plans(self):
        good = object_for("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.fields.n001.nc")
        req = request(start_utc="2026-07-20T01:00:00Z",
                      end_utc_exclusive="2026-07-20T02:00:00Z")
        with tempfile.TemporaryDirectory() as temporary:
            missing_etag = dict(good, etag="")
            report = cbofs.plan_request(req, temporary, objects=[missing_etag])
            self.assertEqual(report["routing_decision"], "review")
            self.assertIn("ETag", report["incomplete_source_metadata"][0]["reason"])
        with tempfile.TemporaryDirectory() as temporary:
            missing_modified = dict(good, last_modified="")
            report = cbofs.plan_request(req, temporary, objects=[missing_modified])
            self.assertEqual(report["routing_decision"], "review")
            self.assertIn("Last-Modified", report["incomplete_source_metadata"][0]["reason"])
        with tempfile.TemporaryDirectory() as temporary:
            wrong_url = dict(good, url="https://example.invalid/object.nc")
            with self.assertRaisesRegex(ValueError, "exact NOAA archive URL"):
                cbofs.plan_request(req, temporary, objects=[wrong_url])
        with tempfile.TemporaryDirectory() as temporary:
            bad_key = dict(good,
                           key="cbofs/netcdf/2026/07/19/cbofs.t06z.20260720.fields.n001.nc")
            bad_key["url"] = core.S3_ENDPOINT + "/" + bad_key["key"]
            with self.assertRaisesRegex(ValueError, "layout/date"):
                cbofs.plan_request(req, temporary, objects=[bad_key])
        wrong_prefix = dict(good, key=good["key"].replace("cbofs/netcdf/", "dbofs/netcdf/", 1))
        wrong_prefix["url"] = core.S3_ENDPOINT + "/" + wrong_prefix["key"]
        with self.assertRaisesRegex(ValueError, "must start"):
            core.validate_source_object(cbofs.CONFIG, wrong_prefix)

    def test_direct_request_to_transfer_is_disabled(self):
        with self.assertRaisesRegex(RuntimeError, "direct request-to-transfer"):
            cbofs.fetch_request(request(), ".", objects=[])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cbofs.build_parser().parse_args([
                "fetch", "--request", "request.json", "--run-dir", "run",
            ])

    def test_injected_object_plan_cannot_authorize_transfer(self):
        item = object_for(
            "cbofs/netcdf/2026/07/20/cbofs.t00z.20260720.fields.n006.nc")
        req = request(end_utc_exclusive="2026-07-20T01:00:00Z")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "approved.json"
            cbofs.plan_request(req, temporary, objects=[item], output=path)
            with self.assertRaisesRegex(RuntimeError, "discovery evidence"):
                cbofs.fetch_plan(path, temporary)


class DownloadTests(unittest.TestCase):
    def test_partial_sidecar_source_identity_drift_restarts_from_zero(self):
        payload = netcdf_payload()
        raw = object_for(
            "cbofs/netcdf/2026/07/20/cbofs.t00z.20260720.fields.n006.nc",
            len(payload), "opaque-etag",
        )
        item = core._decorate_source(cbofs.CONFIG, raw, "aws_operational")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "field.nc"
            destination.with_name("field.nc.part").write_bytes(payload[:7])
            core.write_json_atomic(destination.with_name("field.nc.part.json"), {
                "schema_version": "roms_partial_object_v1",
                "source_id": item["source_id"], "source_identity": "stale-identity",
                "key": item["key"], "url": item["url"], "size": item["size"],
                "etag": item["etag"], "last_modified": "stale-modified",
            })
            response = FakeResponse(
                payload, headers={"Content-Length": str(len(payload)), "ETag": item["etag"]})
            session = FakeSession([response])
            result = cbofs.download_object(item, destination, session=session)
            self.assertFalse(result["resumed"])
            self.assertNotIn("Range", session.calls[0][1]["headers"])

    def test_fetch_relists_exact_key_and_rejects_changed_remote_metadata(self):
        payload = netcdf_payload()
        item = object_for(
            "cbofs/netcdf/2026/07/20/cbofs.t00z.20260720.fields.n006.nc",
            len(payload), "approved-etag",
        )
        req = request(end_utc_exclusive="2026-07-20T01:00:00Z")
        with tempfile.TemporaryDirectory() as temporary:
            plan_path = Path(temporary) / "approved.json"
            plan = cbofs.plan_request(req, temporary, objects=[item], output=plan_path)
            plan = authorize_aws_fixture_plan(plan_path)
            changed = dict(plan["objects"][0], etag="changed-etag")
            changed["source_identity"] = core.archive_sources.source_identity_digest(changed)
            with mock.patch.object(
                core.archive_sources, "list_objects_v2", return_value=[changed],
            ) as relist:
                with self.assertRaisesRegex(RuntimeError, "remote ETag differs"):
                    cbofs.fetch_plan(plan_path, temporary, session=FakeSession([]))
            relist.assert_called_once_with(
                "aws_operational", "cbofs", item["key"],
                session=mock.ANY, max_keys=2,
            )

    def test_fetch_rejects_in_memory_plan_mapping(self):
        with self.assertRaisesRegex(RuntimeError, "reviewed plan file path"):
            cbofs.fetch_plan({}, ".")

    def test_verified_legacy_aws_cache_is_reused_in_place(self):
        payload = netcdf_payload()
        raw = object_for(
            "cbofs/netcdf/2026/07/20/cbofs.t00z.20260720.fields.n006.nc",
            len(payload), "opaque-etag",
        )
        req = request(end_utc_exclusive="2026-07-20T01:00:00Z")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "approved.json"
            plan = cbofs.plan_request(req, root, objects=[raw], output=plan_path)
            plan = authorize_aws_fixture_plan(plan_path)
            item = plan["objects"][0]
            legacy = root / "cache" / "raw" / "2026" / "07" / "20" / Path(item["key"]).name
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(payload)
            core.write_json_atomic(legacy.with_name(legacy.name + ".download.json"), {
                "schema_version": "roms_cached_object_v1", "model": "cbofs",
                "key": item["key"], "url": item["url"], "size": len(payload),
                "etag": item["etag"], "last_modified": item["last_modified"],
                "etag_semantics": "opaque_provenance", "sha256": core.sha256_file(legacy),
            })
            manifest = cbofs.fetch_plan(plan_path, root, session=FakeSession([]))
            outcome = manifest["outcomes"][0]
            self.assertEqual(outcome["cache_location"], "legacy_aws_v1")
            self.assertEqual(Path(outcome["local_path"]), legacy.resolve())
            self.assertEqual(core.verified_manifest_inputs(
                cbofs.CONFIG, root / "fetch_manifest.json", request=req, run_dir=root)[0],
                [legacy.resolve()])

    def test_resume_multipart_and_cache_hit(self):
        payload = netcdf_payload()
        item = object_for("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.fields.n001.nc",
                          len(payload), "multipart-token-8")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "field.nc"
            part = destination.with_name(destination.name + ".part")
            split = len(payload) // 3
            part.write_bytes(payload[:split])
            core.write_json_atomic(destination.with_name(destination.name + ".part.json"), {
                "schema_version": "roms_partial_object_v1", "source_id": None,
                "source_identity": None, "key": item["key"], "url": item["url"],
                "size": len(payload), "etag": item["etag"],
                "last_modified": item["last_modified"],
            })
            response = FakeResponse(payload[split:], status=206,
                                    headers={"Content-Length": str(len(payload) - split),
                                             "ETag": item["etag"],
                                             "Content-Range": f"bytes {split}-{len(payload)-1}/{len(payload)}"})
            session = FakeSession([response])
            result = cbofs.download_object(item, destination, session=session)
            self.assertTrue(result["resumed"])
            self.assertTrue(response.closed)
            self.assertEqual(destination.read_bytes(), payload)
            metadata = core.read_json(destination.with_name(destination.name + ".download.json"))
            self.assertTrue(metadata["etag_is_multipart"])
            self.assertEqual(metadata["etag_semantics"], "opaque_provenance")
            cached = cbofs.download_object(item, destination, session=FakeSession([]))
            self.assertEqual(cached["status"], "cache_hit")

    def test_complete_partial_is_promoted_without_network(self):
        payload = netcdf_payload()
        item = object_for("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.fields.n001.nc",
                          len(payload), "opaque-etag")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "field.nc"
            destination.with_name(destination.name + ".part").write_bytes(payload)
            core.write_json_atomic(destination.with_name(destination.name + ".part.json"), {
                "schema_version": "roms_partial_object_v1", "source_id": None,
                "source_identity": None, "key": item["key"],
                "url": item["url"], "size": len(payload), "etag": item["etag"],
                "last_modified": item["last_modified"],
            })
            result = cbofs.download_object(item, destination, session=FakeSession([]))
            self.assertEqual(result["status"], "downloaded")
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(destination.with_name(destination.name + ".part").exists())

    def test_corrupt_complete_partial_is_reset_then_redownloaded(self):
        payload = netcdf_payload()
        corrupt = b"X" * len(payload)
        item = object_for("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.fields.n001.nc",
                          len(payload), "opaque-etag")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "field.nc"
            part = destination.with_name(destination.name + ".part")
            part.write_bytes(corrupt)
            core.write_json_atomic(destination.with_name(destination.name + ".part.json"), {
                "schema_version": "roms_partial_object_v1", "source_id": None,
                "source_identity": None, "key": item["key"],
                "url": item["url"], "size": len(payload), "etag": item["etag"],
                "last_modified": item["last_modified"],
            })
            response = FakeResponse(payload, headers={"Content-Length": str(len(payload)),
                                                      "ETag": item["etag"]})
            session = FakeSession([response])
            result = cbofs.download_object(item, destination, session=session, max_attempts=2)
            self.assertEqual(result["status"], "downloaded")
            self.assertEqual(result["retry_count"], 1)
            self.assertFalse(result["resumed"])
            self.assertEqual(result["resumed_from_bytes"], 0)
            self.assertTrue(result["discarded_invalid_partial"])
            self.assertEqual(len(session.calls), 1)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(part.exists())
            self.assertFalse(destination.with_name(destination.name + ".part.json").exists())

    def test_corrupt_exact_size_cache_with_matching_sha_is_redownloaded(self):
        payload = netcdf_payload()
        corrupt = b"X" * len(payload)
        item = object_for("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.fields.n001.nc",
                          len(payload), "opaque-etag")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "field.nc"
            destination.write_bytes(corrupt)
            corrupt_digest = core.sha256_file(destination)
            core.write_json_atomic(destination.with_name(destination.name + ".download.json"), {
                "schema_version": "roms_cached_object_v1", "model": "cbofs",
                "key": item["key"], "url": item["url"], "size": len(corrupt),
                "etag": item["etag"], "sha256": core.sha256_file(destination),
            })
            response = FakeResponse(payload, headers={"Content-Length": str(len(payload)),
                                                      "ETag": item["etag"]})
            result = cbofs.download_object(item, destination,
                                           session=FakeSession([response]), max_attempts=1)
            self.assertEqual(result["status"], "downloaded")
            self.assertNotEqual(result["sha256"], corrupt_digest)
            self.assertEqual(result["sha256"], core.sha256_file(destination))
            self.assertEqual(destination.read_bytes(), payload)
            core._validate_netcdf_payload(destination)
            self.assertTrue(response.closed)

    def test_netcdf_metadata_open_validation_is_serialized(self):
        state_lock = core.threading.Lock()
        barrier = core.threading.Barrier(4)

        class TrackingDataset:
            active = 0
            maximum_active = 0

            def __init__(self, _path):
                with state_lock:
                    type(self).active += 1
                    type(self).maximum_active = max(
                        type(self).maximum_active, type(self).active)
                # Make overlap deterministic if the production lock is absent.
                core.time.sleep(0.02)
                self.dimensions = {}
                self.variables = {}

            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _traceback):
                with state_lock:
                    type(self).active -= 1

        class FakeNetCDF4:
            Dataset = TrackingDataset

        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index in range(8):
                path = Path(temporary) / f"field_{index}.nc"
                path.write_bytes(b"\x89HDF\r\n\x1a\n")
                paths.append(path)

            def validate(path):
                barrier.wait()
                core._validate_netcdf_payload(path)

            with mock.patch.object(core, "_netcdf_modules",
                                   return_value=(FakeNetCDF4, None)):
                with core.concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    list(executor.map(validate, paths))
        self.assertEqual(TrackingDataset.maximum_active, 1)

    def test_fetch_manifest_binds_reviewed_plan(self):
        payload = netcdf_payload()
        item = object_for("cbofs/netcdf/2026/07/20/cbofs.t00z.20260720.fields.n006.nc",
                          len(payload), "opaque-etag")
        req = request(end_utc_exclusive="2026-07-20T01:00:00Z", max_workers=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "approved.json"
            cbofs.plan_request(req, root, objects=[item], output=plan_path)
            authorize_aws_fixture_plan(plan_path)
            manifest = cbofs.fetch_plan(
                plan_path, root,
                session=FakeSession([FakeResponse(
                    payload, headers={"Content-Length": str(len(payload)), "ETag": item["etag"]})]),
            )
            approved = manifest["approved_plan"]
            self.assertEqual(approved["path"], str(plan_path.resolve()))
            self.assertEqual(approved["sha256"], core.sha256_file(plan_path))
            self.assertEqual(core.verify_approved_plan_provenance(cbofs.CONFIG, manifest), [])
            plan_path.write_text("{}", encoding="utf-8")
            failures = core.verify_approved_plan_provenance(cbofs.CONFIG, manifest)
            self.assertTrue(any("SHA-256 mismatch" in finding for finding in failures))

    def test_fallback_evidence_is_required_at_fetch_and_health(self):
        payload = netcdf_payload()
        descriptor = core.archive_sources.get_source_descriptor("ncei_long_term", "cbofs")
        key = descriptor["root_prefix"] + "2020/01/nos.cbofs.fields.n006.20200101.t00z.nc"
        parsed = cbofs.parse_object_key(key)
        ncei = {**parsed, "source_id": "ncei_long_term",
                "size": len(payload), "etag": "ncei-etag",
                "last_modified": "2026-07-21T00:00:00Z"}
        req = request(
            schema_version="cbofs_request_v2", source_policy="aws_then_ncei",
            start_utc="2020-01-01T00:00:00Z",
            end_utc_exclusive="2020-01-01T01:00:00Z", max_workers=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); plan_path = root / "approved.json"
            plan = cbofs.plan_request(req, root, objects=[ncei], output=plan_path)
            with self.assertRaisesRegex(RuntimeError, "fallback decision"):
                cbofs.fetch_plan(plan_path, root, session=FakeSession([]))
            plan.update({
                "source_attempts": [
                    {"source_id": "aws_operational", "status": "success"},
                    {"source_id": "ncei_long_term", "status": "success"},
                ],
                "fallback_triggered": True,
                "coverage_before_fallback": ["2020-01-01T00:00:00Z"],
            })
            core.write_json_atomic(plan_path, plan)
            manifest = cbofs.fetch_plan(
                plan_path, root,
                session=FakeSession([FakeResponse(
                    payload, headers={"Content-Length": str(len(payload)), "ETag": ncei["etag"]})]),
            )
            self.assertEqual(core.verify_approved_plan_provenance(cbofs.CONFIG, manifest), [])
            plan["fallback_triggered"] = False
            core.write_json_atomic(plan_path, plan)
            manifest["approved_plan"]["sha256"] = core.sha256_file(plan_path)
            core.write_json_atomic(root / "fetch_manifest.json", manifest)
            failures = core.verify_approved_plan_provenance(cbofs.CONFIG, manifest)
            self.assertTrue(any("fallback" in finding for finding in failures), failures)

    def test_transfer_requires_exact_response_etag(self):
        payload = netcdf_payload()
        item = object_for("cbofs/netcdf/2026/07/20/cbofs.t00z.20260720.fields.n006.nc",
                          len(payload), "planned-etag")
        for headers, message in (
                ({"Content-Length": str(len(payload))}, "no ETag"),
                ({"Content-Length": str(len(payload)), "ETag": "different-etag"},
                 "ETag changed")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                outcome = cbofs.download_object(
                    item, Path(temporary) / "field.nc",
                    session=FakeSession([FakeResponse(payload, headers=headers)]),
                    max_attempts=1)
                self.assertEqual(outcome["status"], "failed")
                self.assertTrue(any(message in error for error in outcome["errors"]))

    def test_existing_transfer_lock_fails_fast_without_touching_partial(self):
        payload = netcdf_payload()
        item = object_for("cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.fields.n001.nc",
                          len(payload), "opaque-etag")
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "field.nc"
            lock = destination.with_name(destination.name + ".transfer.lock")
            lock.write_text("owned by another process", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "another transfer owns"):
                cbofs.download_object(item, destination, session=FakeSession([]), max_attempts=1)
            self.assertFalse(destination.with_name(destination.name + ".part").exists())
            self.assertTrue(lock.is_file())

    def test_lock_cleanup_preserves_replacement_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "field.nc"
            lock = destination.with_name(destination.name + ".transfer.lock")
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
            self.assertEqual(json.loads(lock.read_text(encoding="utf-8"))["owner_token"],
                             "replacement-owner")


class RomsMathTests(unittest.TestCase):
    def test_vtransform_one_and_two_close(self):
        import numpy as np
        s = np.array([-1., -.5, 0.])
        h = np.array([[10.]])
        zeta = np.array([[1.]])
        for transform in (1, 2):
            z = roms.roms_depths(s, s, h, zeta, 2., transform)
            self.assertAlmostEqual(float(z[-1, 0, 0]), 1.0)
            self.assertAlmostEqual(float(z[0, 0, 0]), -10.0)
            thickness = roms.layer_thickness(s, s, h, zeta, 2., transform)
            self.assertAlmostEqual(float(thickness.sum()), 11.0)

    def test_missing_layer_renormalizes(self):
        import numpy as np
        data = np.array([[[1.]], [[np.nan]], [[5.]]])
        weights = np.array([[[1.]], [[2.]], [[1.]]])
        self.assertAlmostEqual(float(roms.weighted_vertical_average(data, weights)[0, 0]), 3.0)

    def test_destagger_and_rotation(self):
        import numpy as np
        u = roms.destagger_u(np.ones((2, 2)))
        v = roms.destagger_v(np.zeros((1, 3)))
        east, north, speed = roms.rotate_to_earth(u, v, np.full((2, 3), np.pi / 2))
        self.assertTrue(np.allclose(east, 0, atol=1e-12))
        self.assertTrue(np.allclose(north, 1))
        self.assertTrue(np.allclose(speed, 1))

    def test_legacy_depth_wrapper_warns(self):
        import numpy as np
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            depth = cbofs.roms_station_depths(np.array([-1., 0.]), np.array([-1., 0.]), 2., 10., 1., 1)
        self.assertEqual(depth.shape, (2,))
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))
        self.assertTrue(cbofs._legacy_field_url("2026-07-20", "t00z", 0).endswith("fields.n006.nc"))

    def test_legacy_fetch_hands_off_written_plan_path(self):
        item = object_for(
            "cbofs/netcdf/2026/07/20/cbofs.t00z.20260720.fields.n006.nc")
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(cbofs, "discover_objects", return_value=[item]), \
                mock.patch.object(cbofs, "plan_request") as planner, \
                mock.patch.object(cbofs, "fetch_plan", return_value={
                    "outcomes": [{"local_path": str(Path(temporary) / "field.nc")}]
                }) as fetcher:
            result = cbofs._legacy_fetch_one(
                "2026-07-20", "t00z", 0, Path(temporary))
        plan_path = Path(temporary) / "download_estimate.json"
        self.assertEqual(result, Path(temporary) / "field.nc")
        self.assertEqual(planner.call_args.kwargs["output"], plan_path)
        self.assertEqual(fetcher.call_args.args[0], plan_path)


class ExtractionTests(unittest.TestCase):
    def test_historical_calendar_alias_decodes_and_preserves_provenance(self):
        import netCDF4
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "historical.nc"
            create_fixture(source, datetime(2026, 7, 20, 0),
                           source_calendar="gregorian_proleptic")
            with netCDF4.Dataset(source) as dataset:
                decoded = core.decode_times(dataset)
            self.assertEqual(decoded[0]["source_calendar"], "gregorian_proleptic")
            self.assertEqual(decoded[0]["decoder_calendar"], "proleptic_gregorian")
            self.assertTrue(decoded[0]["calendar_alias_applied"])
            report = cbofs.extract_request(
                request(end_utc_exclusive="2026-07-20T01:00:00Z"), [source],
                root / "compact.nc")
            self.assertEqual(report["records"][0]["source_calendar"],
                             "gregorian_proleptic")
            self.assertEqual(report["source_time_metadata"][0]["decoder_calendar"],
                             "proleptic_gregorian")
            with netCDF4.Dataset(root / "compact.nc") as dataset:
                metadata = json.loads(dataset.source_time_metadata_json)
                self.assertEqual(metadata[0]["source_calendar"], "gregorian_proleptic")
            unsupported = root / "unsupported-calendar.nc"
            create_fixture(unsupported, datetime(2026, 7, 20, 0),
                           source_calendar="not_a_cf_calendar")
            with netCDF4.Dataset(unsupported) as dataset:
                with self.assertRaises(ValueError):
                    core.decode_times(dataset)

    def test_matching_mixed_archive_extraction_provenance_and_drift_rejection(self):
        import netCDF4
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); aws_path = root / "aws.nc"; ncei_path = root / "ncei.nc"
            create_fixture(aws_path, datetime(2026, 7, 20, 0))
            create_fixture(ncei_path, datetime(2026, 7, 20, 1))
            aws_desc = core.archive_sources.get_source_descriptor("aws_operational", "cbofs")
            ncei_desc = core.archive_sources.get_source_descriptor("ncei_long_term", "cbofs")
            binding = {
                "schema_version": "cbofs_verified_fetch_binding_v2", "verified": True,
                "request_sha256": core.canonical_json_sha256(cbofs.validate_request(request())),
                "fetch_manifest_path": str(root / "fetch_manifest.json"),
                "fetch_manifest_sha256": "a" * 64,
                "approved_plan_path": str(root / "download_estimate.json"),
                "approved_plan_sha256": "b" * 64,
                "objects": [
                    {**aws_desc, "source_archive": "aws_operational",
                     "key": "cbofs/netcdf/2026/07/20/cbofs.t00z.20260720.fields.n006.nc",
                     "url": core.archive_sources.canonical_object_url("aws_operational", "cbofs", "cbofs/netcdf/2026/07/20/cbofs.t00z.20260720.fields.n006.nc"),
                     "local_path": str(aws_path.resolve())},
                    {**ncei_desc, "source_archive": "ncei_long_term",
                     "key": ncei_desc["root_prefix"] + "2026/07/nos.cbofs.fields.n001.20260720.t06z.nc",
                     "url": core.archive_sources.canonical_object_url("ncei_long_term", "cbofs", ncei_desc["root_prefix"] + "2026/07/nos.cbofs.fields.n001.20260720.t06z.nc"),
                     "local_path": str(ncei_path.resolve())},
                ],
            }
            output = root / "mixed.nc"
            report = cbofs.extract_request(request(), [aws_path, ncei_path], output,
                                           provenance=binding)
            self.assertEqual(report["source_provenance"]["archive_count"], 2)
            self.assertEqual({item["source_archive"] for item in report["records"]},
                             {"aws_operational", "ncei_long_term"})
            extraction_manifest = core.read_json(root / "extraction_manifest.json")
            self.assertEqual(extraction_manifest["schema_version"],
                             "cbofs_extraction_manifest_v2")
            with netCDF4.Dataset(ncei_path, "a") as ds:
                ds.variables["lon_rho"][:] += 0.25
            with self.assertRaisesRegex((ValueError, RuntimeError), "geometry.*drift"):
                cbofs.extract_request(request(), [aws_path, ncei_path], root / "bad.nc",
                                      provenance=binding)

    def test_extract_reversed_sigma_and_health(self):
        import netCDF4
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first.nc", root / "second.nc"
            create_fixture(first, datetime(2026, 7, 20, 0), reverse_sigma=True, missing_layer=True)
            create_fixture(second, datetime(2026, 7, 20, 1), reverse_sigma=True)
            output = root / "cbofs_fields.nc"
            req = request(vertical_views=["surface", "bottom", "depth_average"])
            report = cbofs.extract_request(req, [second, first], output)
            self.assertEqual(report["record_count"], 2)
            self.assertEqual(report["status"], "healthy")
            self.assertLess(report["max_thickness_relative_error"], 1e-6)
            with netCDF4.Dataset(output) as dataset:
                self.assertEqual(dataset.schema_version, "roms_compact_fields_v1")
                self.assertIn("native_u_v_grid_relative", dataset.vector_provenance)
                self.assertEqual(dataset.derived_vector_reference, "earth_relative_on_rho_grid")
                self.assertEqual(len(dataset.dimensions["time"]), 2)
                salinity = np.ma.filled(dataset.variables["salinity_surface"][:], np.nan)
                speed = np.ma.filled(dataset.variables["current_speed_surface"][:], np.nan)
                average = np.ma.filled(dataset.variables["current_speed_depth_average"][:], np.nan)
                self.assertAlmostEqual(float(np.nanmedian(salinity)), 30., places=5)
                self.assertAlmostEqual(float(np.nanmedian(speed)), 3., places=5)
                self.assertAlmostEqual(float(np.nanmedian(average)), 2., places=5)
                self.assertEqual(dataset.angle_convention, roms.ANGLE_CONVENTION)
                self.assertEqual(dataset.variables["angle"].angle_convention,
                                 roms.ANGLE_CONVENTION)
                self.assertEqual(dataset.variables["angle"].units, "radians")

    def test_angle_units_semantics_values_and_metadata_drift_are_rejected(self):
        import netCDF4
        import numpy as np
        cases = [
            {"angle_units": "degrees"},
            {"angle_standard_name": None, "angle_long_name": "grid orientation"},
            {"angle_value": 2 * np.pi + 0.01},
            {"angle_value": np.nan},
            {"angle_on_u_grid": True},
        ]
        for updates in cases:
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "bad-angle.nc"
                create_fixture(path, datetime(2026, 7, 20, 0), **updates)
                with netCDF4.Dataset(path) as dataset:
                    with self.assertRaisesRegex(RuntimeError, "angle"):
                        roms.read_geometry(dataset)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first.nc", root / "second.nc"
            create_fixture(first, datetime(2026, 7, 20, 0))
            create_fixture(second, datetime(2026, 7, 20, 1), angle_units="rad")
            with self.assertRaisesRegex(RuntimeError, "geometry/schema drift.*angle_units"):
                cbofs.extract_request(request(), [first, second], root / "bad.nc")

    def test_explicit_missing_layer_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.nc"
            create_fixture(source, datetime(2026, 7, 20, 0), missing_layer=True)
            req = request(end_utc_exclusive="2026-07-20T01:00:00Z", vertical_views=[1])
            report = cbofs.extract_request(req, [source], root / "compact.nc")
            self.assertEqual(report["status"], "critical")
            self.assertIn("below 95%", report["critical"][0])

    def test_geometry_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first.nc", root / "second.nc"
            create_fixture(first, datetime(2026, 7, 20, 0))
            create_fixture(second, datetime(2026, 7, 20, 1), lon_offset=.1)
            with self.assertRaisesRegex(RuntimeError, "geometry/schema drift"):
                cbofs.extract_request(request(), [first, second], root / "bad.nc")

    def test_missing_or_nonfinite_vertical_metadata_is_rejected(self):
        import netCDF4
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.nc"
            create_fixture(missing, datetime(2026, 7, 20, 0))
            with netCDF4.Dataset(missing, "a") as dataset:
                dataset.delncattr("Vstretching")
            with netCDF4.Dataset(missing) as dataset:
                with self.assertRaisesRegex(RuntimeError, "Vstretching"):
                    roms.read_geometry(dataset)
            nonfinite = root / "nonfinite.nc"
            create_fixture(nonfinite, datetime(2026, 7, 20, 0))
            with netCDF4.Dataset(nonfinite, "a") as dataset:
                dataset.variables["Cs_r"][1] = np.nan
            with netCDF4.Dataset(nonfinite) as dataset:
                with self.assertRaisesRegex(RuntimeError, "Cs_r"):
                    roms.read_geometry(dataset)

    def test_passthrough_extract_rejected(self):
        with self.assertRaisesRegex(ValueError, "passthrough"):
            cbofs.extract_request({"schema_version": "cbofs_request_v1",
                                   "start_utc": "2026-07-20T00:00:00Z",
                                   "end_utc_exclusive": "2026-07-20T01:00:00Z",
                                   "product": "stations", "guidance": "nowcast"},
                                  ["unused.nc"], "unused-output.nc")


class HealthAndProvenanceTests(unittest.TestCase):
    def test_existing_v1_station_evidence_remains_readable(self):
        run = HERE.parents[2] / "runs" / "station_smoke"
        if not (run / "fetch_manifest.json").is_file():
            self.skipTest("retained v1 station evidence is unavailable")
        req = core.read_json(run / "request.json")
        report = cbofs.evaluate_health(req, run)
        self.assertEqual(report["status"], "healthy", report)
        self.assertTrue(any("legacy v1" in item for item in report["warnings"]))

    def test_health_is_fail_closed_without_manifest_or_fields_compact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            req = request(end_utc_exclusive="2026-07-20T01:00:00Z")
            report = cbofs.evaluate_health(req, root, [])
            self.assertEqual(report["status"], "critical")
            self.assertTrue(any("fetch_manifest" in item for item in report["critical"]))
            self.assertTrue(any("compact" in item for item in report["critical"]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            req, _ = create_run_fixture(root)
            report = cbofs.evaluate_health(req, root, [])
            self.assertEqual(report["status"], "critical")
            self.assertTrue(any("compact" in item for item in report["critical"]))

    def test_compact_provenance_and_hash_are_verified(self):
        import netCDF4
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            req, source = create_run_fixture(root)
            compact = root / "cbofs_fields.nc"
            extract_bound_run(root, req, compact)
            report = cbofs.evaluate_health(req, root, [compact])
            self.assertEqual(report["status"], "healthy")
            with netCDF4.Dataset(compact) as dataset:
                self.assertEqual(json.loads(dataset.request_json), req)
                key = str(netCDF4.chartostring(dataset.variables["source_key"][:])[0]).rstrip("\x00")
                self.assertEqual(key, "cbofs/netcdf/2026/07/20/cbofs.t00z.20260720.fields.n006.nc")
                self.assertEqual(dataset.variables["salinity_surface"].coordinates,
                                 "lon_rho lat_rho")
                self.assertNotIn("time", dataset.variables["salinity_surface"].ncattrs())
                self.assertNotIn("ocean_time", dataset.variables["salinity_surface"].coordinates)
                self.assertEqual(dataset.input_provenance_mode, "verified_fetch_manifest")
                self.assertEqual(dataset.angle_convention, roms.ANGLE_CONVENTION)
            extraction = core.read_json(compact.with_suffix(".health.json"))
            self.assertEqual(extraction["fetch_binding"]["verified"], True)
            self.assertEqual(extraction["angle_metadata"]["convention"],
                             roms.ANGLE_CONVENTION)
            with compact.open("ab") as stream:
                stream.write(b"tamper")
            report = cbofs.evaluate_health(req, root, [compact])
            self.assertEqual(report["status"], "critical")
            self.assertTrue(any("compact integrity" in item for item in report["critical"]))

    def test_health_rejects_tampered_approved_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            req, source = create_run_fixture(root)
            compact = root / "cbofs_fields.nc"
            extract_bound_run(root, req, compact)
            (root / "download_estimate.json").write_text("{}", encoding="utf-8")
            report = cbofs.evaluate_health(req, root, [compact])
            self.assertEqual(report["status"], "critical")
            self.assertTrue(any("approved-plan provenance" in item
                                for item in report["critical"]))

    def test_manifest_paths_and_health_reject_tampered_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            req, source = create_run_fixture(root)
            compact = root / "cbofs_fields.nc"
            extract_bound_run(root, req, compact)
            sidecar_path = source.with_name(source.name + ".download.json")
            sidecar = core.read_json(sidecar_path)
            sidecar["etag"] = "tampered-etag"
            core.write_json_atomic(sidecar_path, sidecar)
            with self.assertRaisesRegex(RuntimeError, "sidecar ETag mismatch"):
                core.manifest_paths(cbofs.CONFIG, root, request=req)
            report = cbofs.evaluate_health(req, root, [compact])
            self.assertEqual(report["status"], "critical")
            self.assertTrue(any("sidecar ETag mismatch" in item
                                for item in report["critical"]))

    def test_every_sidecar_identity_field_is_enforced(self):
        updates = {
            "key": "cbofs/netcdf/2026/07/20/cbofs.t06z.20260720.fields.n001.nc",
            "url": "https://example.invalid/tampered.nc",
            "size": 1,
            "sha256": "0" * 64,
            "etag": "tampered-etag",
        }
        labels = {"etag": "ETag", "sha256": "SHA-256"}
        for name, value in updates.items():
            with self.subTest(field=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                req, source = create_run_fixture(root)
                sidecar_path = source.with_name(source.name + ".download.json")
                sidecar = core.read_json(sidecar_path)
                sidecar[name] = value
                core.write_json_atomic(sidecar_path, sidecar)
                with self.assertRaisesRegex(RuntimeError, f"sidecar .*{labels.get(name, name)}"):
                    core.manifest_paths(cbofs.CONFIG, root, request=req)

    def test_cli_manifest_and_run_dir_extraction_embed_verified_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            req, _ = create_run_fixture(root)
            request_path = root / "request.json"
            core.write_json_atomic(request_path, req)
            for mode, value in (("--run-dir", root),
                                ("--manifest", root / "fetch_manifest.json")):
                output = root / ("run-dir.nc" if mode == "--run-dir" else "manifest.nc")
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(cbofs.main([
                        "extract", "--request", str(request_path), mode, str(value),
                        "--output", str(output),
                    ]), 0)
                report = core.read_json(output.with_suffix(".health.json"))
                self.assertEqual(report["input_provenance_mode"],
                                 "verified_fetch_manifest")
                self.assertEqual(report["fetch_binding"]["request_sha256"],
                                 core.canonical_json_sha256(req))

    def test_unbound_extraction_cannot_pass_full_run_health(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            req, source = create_run_fixture(root)
            compact = root / "unbound.nc"
            report = cbofs.extract_request(req, [source], compact)
            self.assertEqual(report["input_provenance_mode"], "explicit_unbound_inputs")
            health = cbofs.evaluate_health(req, root, [compact])
            self.assertEqual(health["status"], "critical")
            self.assertTrue(any("compact integrity" in item for item in health["critical"]))

    def test_verified_manifest_inputs_reject_request_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            req, _ = create_run_fixture(root)
            mismatched = dict(req, start_utc="2026-07-19T23:00:00Z")
            with self.assertRaisesRegex(RuntimeError, "request does not match"):
                core.verified_manifest_inputs(cbofs.CONFIG, root / "fetch_manifest.json",
                                              request=mismatched, run_dir=root)

    def test_delete_after_extract_runs_after_health_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            req = request(end_utc_exclusive="2026-07-20T01:00:00Z",
                          cache_policy="delete_after_extract")
            req, source = create_run_fixture(root, req)
            compact = root / "cbofs_fields.nc"
            extract_bound_run(root, req, compact)
            self.assertTrue(source.is_file())
            first = cbofs.evaluate_health(req, root, [compact])
            self.assertEqual(first["status"], "healthy")
            self.assertFalse(source.exists())
            self.assertTrue((root / "cache_cleanup.json").is_file())
            second = cbofs.evaluate_health(req, root, [compact])
            self.assertEqual(second["status"], "healthy")
            self.assertEqual(second["transfers"][0]["state"],
                             "intentionally_deleted_after_health")


if __name__ == "__main__":
    unittest.main(verbosity=2)
