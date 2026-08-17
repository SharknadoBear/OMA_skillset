#!/usr/bin/env python3
"""Offline regression tests for argo-fetcher, plus the frozen Guam parity gate."""

from __future__ import annotations

import copy
import gzip
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

import argo_fetcher as af


def expect_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except (af.ArgoError, ValueError, FileNotFoundError):
        return
    raise AssertionError(f"Expected failure from {function.__name__}")


def write_index(path: Path, product: str, rows: list[dict[str, object]], update: str = "20260602000000") -> None:
    columns = ["file", "date", "latitude", "longitude", "ocean", "profiler_type", "institution", "date_update"]
    if product in {"bio", "synthetic"}:
        columns = ["file", "date", "latitude", "longitude", "ocean", "profiler_type", "institution", "parameters", "parameter_data_mode", "date_update"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("# Title : synthetic Argo index\n")
        handle.write(f"# Date of update : {update}\n")
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(column, "")) for column in columns) + "\n")


def _put_chars(variable, texts: list[str], width: int) -> None:
    matrix = np.full((len(texts), width), b" ", dtype="S1")
    for index, text in enumerate(texts):
        encoded = text.encode("ascii")[:width]
        matrix[index, : len(encoded)] = np.frombuffer(encoded, dtype="S1")
    variable[:] = matrix


def write_profile(
    path: Path,
    *,
    product: str = "core",
    update: str = "20260602000000",
    n_prof: int = 2,
    n_levels: int = 3,
    parameters: tuple[str, ...] | None = None,
    finite: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = parameters or (("PRES", "TEMP", "PSAL") if product == "core" else ("PRES", "DOXY", "CHLA"))
    with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("N_PROF", n_prof)
        dataset.createDimension("N_LEVELS", n_levels)
        dataset.createDimension("N_PARAM", len(params))
        dataset.createDimension("STRING8", 8)
        dataset.createDimension("STRING14", 14)
        dataset.createDimension("STRING16", 16)
        date_update = dataset.createVariable("DATE_UPDATE", "S1", ("STRING14",))
        date_update[:] = np.frombuffer(update.encode("ascii"), dtype="S1")
        platform = dataset.createVariable("PLATFORM_NUMBER", "S1", ("N_PROF", "STRING8"))
        _put_chars(platform, ["5900001", "5900001"][:n_prof], 8)
        data_mode = dataset.createVariable("DATA_MODE", "S1", ("N_PROF",))
        data_mode[:] = np.asarray([b"R", b"A"][:n_prof], dtype="S1")
        parameter_mode = dataset.createVariable("PARAMETER_DATA_MODE", "S1", ("N_PROF", "N_PARAM"))
        parameter_mode[:] = np.asarray([[b"R"] * len(params), [b"D"] * len(params)][:n_prof], dtype="S1")
        station = dataset.createVariable("STATION_PARAMETERS", "S1", ("N_PROF", "N_PARAM", "STRING16"))
        station_values = np.full((n_prof, len(params), 16), b" ", dtype="S1")
        for profile in range(n_prof):
            for index, name in enumerate(params):
                encoded = name.encode("ascii")
                station_values[profile, index, : len(encoded)] = np.frombuffer(encoded, dtype="S1")
        station[:] = station_values
        latitude = dataset.createVariable("LATITUDE", "f8", ("N_PROF",))
        longitude = dataset.createVariable("LONGITUDE", "f8", ("N_PROF",))
        latitude[:] = np.linspace(10.0, 10.2, n_prof)
        longitude[:] = np.linspace(170.0, 170.2, n_prof)
        juld = dataset.createVariable("JULD", "f8", ("N_PROF",))
        juld.units = "days since 1950-01-01 00:00:00 UTC"
        juld.calendar = "gregorian"
        juld[:] = np.arange(n_prof) + 27000
        for qc_name in ("POSITION_QC", "JULD_QC"):
            variable = dataset.createVariable(qc_name, "S1", ("N_PROF",))
            variable[:] = np.asarray([b"1", b"2"][:n_prof], dtype="S1")
        for parameter in params:
            raw = dataset.createVariable(parameter, "f4", ("N_PROF", "N_LEVELS"), fill_value=99999.0)
            adjusted = dataset.createVariable(f"{parameter}_ADJUSTED", "f4", ("N_PROF", "N_LEVELS"), fill_value=99999.0)
            error = dataset.createVariable(f"{parameter}_ADJUSTED_ERROR", "f4", ("N_PROF", "N_LEVELS"), fill_value=99999.0)
            values = np.arange(n_prof * n_levels, dtype="f4").reshape(n_prof, n_levels) if finite else np.full((n_prof, n_levels), 99999.0, dtype="f4")
            raw[:] = values
            adjusted[:] = values
            error[:] = np.full((n_prof, n_levels), 0.1, dtype="f4")
            qc = dataset.createVariable(f"{parameter}_QC", "S1", ("N_PROF", "N_LEVELS"))
            qc[:] = np.asarray([[b"1"] * n_levels, [b"3"] * n_levels][:n_prof], dtype="S1")


def write_empty_profile(path: Path, update: str = "20260602000000") -> None:
    with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("N_PROF", 0)
        dataset.createDimension("N_LEVELS", 0)
        dataset.createDimension("STRING14", 14)
        date_update = dataset.createVariable("DATE_UPDATE", "S1", ("STRING14",))
        date_update[:] = np.frombuffer(update.encode("ascii"), dtype="S1")


def base_rows() -> dict[str, list[dict[str, object]]]:
    common = {"ocean": "P", "profiler_type": "846", "institution": "AO", "date_update": "20260602000000"}
    return {
        "core": [
            {**common, "file": "aoml/5900001/profiles/R5900001_001.nc", "date": "20250101000000", "latitude": 10, "longitude": 170},
            {**common, "file": "aoml/5900001/profiles/D5900001_002.nc", "date": "20250102000000", "latitude": 0, "longitude": -179},
            {**common, "file": "coriolis/5900002/profiles/R5900002_001.nc", "date": "20250103000000", "latitude": 20, "longitude": -160},
        ],
        "synthetic": [
            {**common, "file": "coriolis/5900002/profiles/SR5900002_001.nc", "date": "20250101000000", "latitude": 10, "longitude": 170, "parameters": "PRES TEMP PSAL DOXY CHLA", "parameter_data_mode": "RRRRR"},
        ],
        "bio": [
            {**common, "file": "aoml/5900001/profiles/BD5900001_001.nc", "date": "20250101000000", "latitude": 10, "longitude": 170, "parameters": "PRES DOXY CHLA", "parameter_data_mode": "RDD"},
            {**common, "file": "aoml/5900001/profiles/BR5900001_002.nc", "date": "20250102000000", "latitude": 11, "longitude": 171, "parameters": "PRES DOXY", "parameter_data_mode": "RR"},
        ],
    }


def make_indexes(root: Path, rows: dict[str, list[dict[str, object]]] | None = None) -> None:
    for product, values in (rows or base_rows()).items():
        write_index(root / af.PRODUCTS[product], product, values)


def test_requests_and_selection(root: Path) -> None:
    indexes = root / "indexes"
    make_indexes(indexes)
    request = {"schema": af.REQUEST_SCHEMA, "products": ["core"], "start": "2025-01-01T00:00:00Z", "end": "2025-01-02T00:00:00Z", "bbox": [169, 9, 171, 11]}
    frame = af.load_index(indexes / af.PRODUCTS["core"], "core")
    assert af.select_profiles(frame, request, "core")["file"].tolist() == ["aoml/5900001/profiles/R5900001_001.nc"]
    dateline = {**request, "start": "2025-01-02T00:00:00Z", "bbox": [175, -5, -175, 5]}
    assert len(af.select_profiles(frame, dateline, "core")) == 1
    filtered = {**request, "wmos": [5900001], "dacs": ["aoml"], "file_modes": ["R"]}
    assert len(af.select_profiles(frame, filtered, "core")) == 1

    polygon = root / "polygon.geojson"
    polygon.write_text(json.dumps({"type": "Polygon", "coordinates": [[[170, 10], [171, 10], [171, 11], [170, 11], [170, 10]]]}), encoding="utf-8")
    poly_request = {**request, "geojson": str(polygon)}
    poly_request.pop("bbox")
    assert len(af.select_profiles(frame, poly_request, "core")) == 1  # boundary counts inside
    multi = root / "multi.geojson"
    multi.write_text(json.dumps({"type": "MultiPolygon", "coordinates": [[[[169, 9], [171, 9], [171, 11], [169, 11], [169, 9]]], [[[-180, -1], [-178, -1], [-178, 1], [-180, 1], [-180, -1]]]]}), encoding="utf-8")
    multi_request = {**request, "geojson": str(multi), "end": "2025-01-02T00:00:00Z"}
    multi_request.pop("bbox")
    assert len(af.select_profiles(frame, multi_request, "core")) == 2

    mesh = root / "mesh.2dm"
    mesh.write_text("MESH2D\nND 1 169 9 0\nND 2 171 9 0\nND 3 171 11 0\nND 4 169 11 0\nE4Q 1 1 2 3 4 1\n", encoding="utf-8")
    mesh_request = {**request, "mesh_2dm": str(mesh)}
    mesh_request.pop("bbox")
    assert len(af.select_profiles(frame, mesh_request, "core")) == 1
    mesh_360 = root / "mesh_360.2dm"
    mesh_360.write_text("MESH2D\nND 1 180 -1 0\nND 2 182 -1 0\nND 3 182 1 0\nND 4 180 1 0\nE4Q 1 1 2 3 4 1\n", encoding="utf-8")
    mesh_360_request = {**request, "start": "2025-01-02", "end": "2025-01-02", "mesh_2dm": str(mesh_360)}
    mesh_360_request.pop("bbox")
    assert len(af.select_profiles(frame, mesh_360_request, "core")) == 1

    bio = af.load_index(indexes / af.PRODUCTS["bio"], "bio")
    bgc = {"schema": af.REQUEST_SCHEMA, "products": ["bio"], "start": "2025-01-01", "end": "2025-01-03", "global": True, "parameters": ["DOXY", "CHLA"], "parameter_match": "all"}
    assert len(af.select_profiles(bio, bgc, "bio")) == 1
    bgc["parameter_match"] = "any"
    assert len(af.select_profiles(bio, bgc, "bio")) == 2
    assert list(zip(bio.iloc[0]["parameters"].split(), bio.iloc[0]["parameter_data_mode"])) == [("PRES", "R"), ("DOXY", "D"), ("CHLA", "D")]

    for invalid in (
        {"schema": af.REQUEST_SCHEMA, "products": ["core"], "start": "2025-01-01", "end": "2025-01-02"},
        {**request, "global": True},
        {**request, "end": "2024-01-01"},
        {**request, "file_modes": ["A"]},
        {**request, "products": ["core"], "parameters": ["DOXY"]},
        {**request, "bbox": [0, 10, 1, 5]},
    ):
        expect_error(af.validate_request, invalid)


def test_plan_and_revision(root: Path) -> None:
    indexes = root / "plan_indexes"
    make_indexes(indexes)
    request = {"schema": af.REQUEST_SCHEMA, "products": ["core", "bio", "synthetic"], "start": "2025-01-01", "end": "2025-01-03", "global": True}
    all_files = [str(row["file"]) for values in base_rows().values() for row in values]
    sizes = {name: 1000 + index for index, name in enumerate(all_files)}
    run = root / "plan_run"
    plan = af.build_download_plan(request, run, index_dir=indexes, size_lookup=sizes, probe_result={"ok": True, "bytes_per_second": 1000, "bytes": 100, "seconds": 0.1}, free_bytes_override=10_000_000)
    af.verify_plan(plan)
    assert plan["selection_count"] == 6 and not plan["blocked"]
    assert plan["estimate"]["method"] == "exact_head"
    assert (run / "selection.json").is_file() and (run / "selection.csv").is_file()
    tampered = copy.deepcopy(plan)
    tampered["selection_count"] += 1
    expect_error(af.verify_plan, tampered)
    expired = copy.deepcopy(plan)
    expired["expires_utc"] = "2000-01-01T00:00:00Z"
    expired.pop("plan_hash")
    expired["plan_hash"] = af.sha256_value(expired)
    expect_error(af.verify_plan, expired)
    blocked = af.build_download_plan(request, root / "blocked", index_dir=indexes, size_lookup=sizes, probe_result={"ok": True, "bytes_per_second": 1000}, free_bytes_override=4 * sum(sizes.values()))
    assert blocked["blocked"] and not blocked["storage"]["passes"]
    empty_request = {"schema": af.REQUEST_SCHEMA, "products": ["core"], "start": "1900-01-01", "end": "1900-01-02", "global": True}
    empty = af.build_download_plan(empty_request, root / "empty", index_dir=indexes, size_lookup={}, probe_result={"ok": True, "bytes_per_second": None}, free_bytes_override=1_000_000)
    assert empty["blocked"] and "selection_empty" in empty["block_reasons"]

    rows = base_rows()
    rows["core"][0]["date_update"] = "20260603000000"
    write_index(indexes / af.PRODUCTS["core"], "core", rows["core"], update="20260603000000")
    expect_error(af.recheck_selected_rows, plan, root / "cache", index_dir=indexes)

    many = []
    for number in range(130):
        many.append({"product": "core" if number % 2 else "bio", "dac": "aoml" if number % 3 else "coriolis", "file_mode": "R" if number % 5 else "D", "file": f"dac/{number}"})
    assert af.deterministic_sample(many) == af.deterministic_sample(list(reversed(many)))
    assert len(af.deterministic_sample(many)) == 64


class IndexHandler(BaseHTTPRequestHandler):
    payload = b""

    def log_message(self, *_args) -> None:
        return

    def _headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.send_header("ETag", hashlib_sha(self.payload))
        self.send_header("Last-Modified", "Wed, 03 Jun 2026 00:00:00 GMT")
        self.end_headers()

    def do_HEAD(self) -> None:
        self._headers()

    def do_GET(self) -> None:
        self._headers()
        self.wfile.write(self.payload)


def test_index_refresh_and_offline(root: Path) -> None:
    source = root / af.PRODUCTS["core"]
    write_index(source, "core", base_rows()["core"][:1])
    IndexHandler.payload = source.read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", 0), IndexHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    old_primary = af.PRIMARY_BASE
    af.PRIMARY_BASE = f"http://127.0.0.1:{server.server_port}"
    cache = root / "cache"
    try:
        path, first = af.ensure_index("core", cache, timeout=2)
        original_hash = first["sha256"]
        changed_rows = base_rows()["core"][:2]
        write_index(source, "core", changed_rows, update="20260603000000")
        IndexHandler.payload = source.read_bytes()
        path, second = af.ensure_index("core", cache, timeout=2)
        assert second["sha256"] != original_hash and len(af.load_index(path, "core")) == 2
    finally:
        server.shutdown()
        server.server_close()
    try:
        _, offline = af.ensure_index("core", cache, allow_stale_offline=True, timeout=0.2)
        assert offline["offline_stale_used"] is True
        expect_error(af.ensure_index, "core", cache, timeout=0.2)
    finally:
        af.PRIMARY_BASE = old_primary


class MirrorHandler(BaseHTTPRequestHandler):
    payload = b""
    fail_primary = True

    def log_message(self, *_args) -> None:
        return

    def do_HEAD(self) -> None:
        if self.path.startswith("/primary") and self.fail_primary:
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.send_header("ETag", hashlib_sha(self.payload))
        self.send_header("Last-Modified", "Wed, 03 Jun 2026 00:00:00 GMT")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/primary") and self.fail_primary:
            self.send_error(503)
            return
        start = 0
        if self.headers.get("Range"):
            start = int(self.headers["Range"].split("=")[1].split("-")[0])
        body = self.payload[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Last-Modified", "Wed, 03 Jun 2026 00:00:00 GMT")
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(self.payload)-1}/{len(self.payload)}")
        self.end_headers()
        self.wfile.write(body)


def hashlib_sha(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def test_transport_resume_fallback_and_cache(root: Path) -> None:
    source = root / "source.nc"
    write_profile(source, product="bio", parameters=("PRES", "DOXY", "CHLA"))
    MirrorHandler.payload = source.read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", 0), MirrorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    old_primary, old_s3 = af.PRIMARY_BASE, af.S3_BASE
    af.PRIMARY_BASE = f"http://127.0.0.1:{server.server_port}/primary"
    af.S3_BASE = f"http://127.0.0.1:{server.server_port}/fallback"
    try:
        relative = "aoml/5900001/profiles/BD5900001_001.nc"
        row = {"product": "bio", "file": relative, "date_update": "20260602000000", "local_path": af.safe_relative_profile_path(relative).as_posix()}
        run = root / "transport"
        partial = run / row["local_path"]
        partial = partial.with_name(partial.name + ".part")
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(MirrorHandler.payload[:200])
        result = af._download_one(row, run, timeout=5, retries=5, existing_record=None, sleep=lambda _seconds: None)
        assert result["status"] == "downloaded" and result["mirror"] == "argo_s3" and not partial.exists()
        record = result
        second = af._download_one(row, run, timeout=5, retries=5, existing_record=record, sleep=lambda _seconds: None)
        assert second["status"] == "reused_validated"

        stale = root / "stale.nc"
        write_profile(stale, product="bio", update="20250101000000", parameters=("PRES", "DOXY", "CHLA"))
        MirrorHandler.payload = stale.read_bytes()
        row2 = {**row, "file": "aoml/5900001/profiles/BD5900001_002.nc", "local_path": af.safe_relative_profile_path("aoml/5900001/profiles/BD5900001_002.nc").as_posix()}
        rejected = af._download_one(row2, run, timeout=5, retries=5, existing_record=None, sleep=lambda _seconds: None)
        assert rejected["status"] == "failed" and "older" in rejected["error"]
        assert Path(run / row2["local_path"]).with_name(Path(row2["local_path"]).name + ".part").exists()
    finally:
        af.PRIMARY_BASE, af.S3_BASE = old_primary, old_s3
        server.shutdown()
        server.server_close()


def make_health_run(root: Path, *, product: str = "bio", parameters: tuple[str, ...] = ("PRES", "DOXY", "CHLA"), update: str = "20260602000000", finite: bool = True) -> tuple[dict, Path]:
    indexes = root / "indexes"
    rows = base_rows()
    for key in list(rows):
        if key != product:
            del rows[key]
    rows[product] = rows[product][:1]
    rows[product][0]["date_update"] = "20260602000000"
    make_indexes(indexes, rows)
    relative = str(rows[product][0]["file"])
    request = {"schema": af.REQUEST_SCHEMA, "products": [product], "start": "2025-01-01", "end": "2025-01-02", "global": True}
    if product != "core":
        request["parameters"] = ["DOXY", "CHLA"]
        request["parameter_match"] = "all"
    run = root / "run"
    plan = af.build_download_plan(request, run, index_dir=indexes, size_lookup={relative: 1000}, probe_result={"ok": True, "bytes_per_second": 1000}, free_bytes_override=1_000_000)
    native = run / plan["selected_rows"][0]["local_path"]
    write_profile(native, product=product, update=update, parameters=parameters, finite=finite)
    af.atomic_write_csv(run / "download_manifest.csv", [{"product": product, "file": relative, "status": "downloaded", "attempts": 1, "mirror": "fixture", "bytes": native.stat().st_size, "sha256": af.sha256_file(native), "index_date_update": "20260602000000", "internal_date_update": update, "local_path": plan["selected_rows"][0]["local_path"], "error": ""}])
    return plan, run


def test_health(root: Path) -> None:
    plan, run = make_health_run(root / "valid")
    health = af.health_check(plan, run)
    assert health["status"] == "pass" and health["profiles"] == 2
    assert set(health["summary"]["data_mode_counts"]) == {"R", "A"}
    assert health["summary"]["parameter_data_mode_counts"]["D"] > 0
    assert health["files"][0]["parameter_data_mode_by_parameter"]["CHLA"]["D"] > 0
    assert len(health["plots"]) == 3 and (run / "profile_inventory.csv").is_file()

    stale_plan, stale_run = make_health_run(root / "stale", update="20250101000000")
    assert af.health_check(stale_plan, stale_run)["status"] == "fail"
    missing_plan, missing_run = make_health_run(root / "missing", parameters=("PRES", "DOXY"))
    assert any(item["code"] == "requested_parameter_absent" for item in af.health_check(missing_plan, missing_run)["failures"])
    finite_plan, finite_run = make_health_run(root / "finite", finite=False)
    assert any(item["code"] == "no_finite_requested_observations" for item in af.health_check(finite_plan, finite_run)["failures"])
    empty_plan, empty_run = make_health_run(root / "empty")
    empty_path = empty_run / empty_plan["selected_rows"][0]["local_path"]
    write_empty_profile(empty_path)
    empty_manifest = list(af._manifest_map(empty_run / "download_manifest.csv").values())
    empty_manifest[0]["bytes"] = empty_path.stat().st_size
    empty_manifest[0]["sha256"] = af.sha256_file(empty_path)
    af.atomic_write_csv(empty_run / "download_manifest.csv", empty_manifest)
    assert any(item["code"] == "empty_profile_or_level_dimension" for item in af.health_check(empty_plan, empty_run)["failures"])
    corrupt_plan, corrupt_run = make_health_run(root / "corrupt")
    corrupt_path = corrupt_run / corrupt_plan["selected_rows"][0]["local_path"]
    corrupt_path.write_bytes(b"not-netcdf")
    manifest = list(af._manifest_map(corrupt_run / "download_manifest.csv").values())
    manifest[0]["bytes"] = corrupt_path.stat().st_size
    manifest[0]["sha256"] = af.sha256_file(corrupt_path)
    af.atomic_write_csv(corrupt_run / "download_manifest.csv", manifest)
    assert any(item["code"] == "netcdf_open_failure" for item in af.health_check(corrupt_plan, corrupt_run)["failures"])


def test_guam_parity() -> str:
    base = Path.home() / "OneDrive - PNNL" / "Desktop" / "OTEC_guam"
    index = base / "Workspace" / "Data" / "argo" / "index" / af.PRODUCTS["core"]
    mesh = base / "Resources" / "hgrid_Deception_test_run.2dm"
    manifest = base / "Workspace" / "Data" / "argo" / "argo_guam_manifest.csv"
    if not (index.is_file() and mesh.is_file() and manifest.is_file()):
        return "skipped_fixture_unavailable"
    frame = af.load_index(index, "core")
    request = {
        "schema": af.REQUEST_SCHEMA, "products": ["core"],
        "start": "1993-01-01T00:00:00Z", "end": "2022-12-31T23:59:59Z",
        "mesh_2dm": str(mesh),
    }
    actual = sorted(af.select_profiles(frame, request, "core")["file"].tolist())
    expected = sorted(pd.read_csv(manifest, dtype=str)["file"].tolist())
    assert len(actual) == 2257, f"Guam selection count changed: {len(actual)}"
    assert actual == expected, f"Guam path parity changed: {len(set(actual) ^ set(expected))} differing paths"
    return "passed_2257_paths"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="argo_fetcher_selftest_") as directory:
        root = Path(directory)
        test_requests_and_selection(root / "selection")
        test_plan_and_revision(root / "planning")
        test_index_refresh_and_offline(root / "indexes")
        test_transport_resume_fallback_and_cache(root / "transport")
        test_health(root / "health")
        parity = test_guam_parity()
    print(json.dumps({"status": "pass", "tests": 5, "guam_parity": parity}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
