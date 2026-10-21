"""Regional ERA5 acquisition. Credentials never enter portable requests or reports."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import tempfile
import time
import uuid
import zipfile

import numpy as np
import xarray as xr

UTC = dt.timezone.utc
HOUR = dt.timedelta(hours=1)
DATASET = "reanalysis-era5-single-levels"
STEP = 0.25
PRODUCTS = {
    "wind_10m": [("10m_u_component_of_wind", "u10"), ("10m_v_component_of_wind", "v10")],
    "mean_sea_level_pressure": [("mean_sea_level_pressure", "msl")],
}
PACKAGES = ["cdsapi", "ecmwf-datastores-client", "numpy", "xarray", "netCDF4", "requests", "PyYAML", "matplotlib"]


class ERA5Error(Exception):
    pass


class RequestError(ERA5Error):
    pass


class ValidationError(ERA5Error):
    pass


class AccessError(ERA5Error):
    pass


class SizeLimitError(ERA5Error):
    pass


class TransientError(ERA5Error):
    pass


def now():
    return dt.datetime.now(UTC).isoformat()


def iso(value):
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc(value):
    try:
        result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise RequestError("Endpoints must be ISO-8601 UTC hours") from exc
    if result.tzinfo is None or result.utcoffset() != dt.timedelta(0) or result.minute or result.second or result.microsecond:
        raise RequestError("Endpoints must be whole hours with UTC Z or +00:00")
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def replace(source, target):
    for attempt in range(6):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.15 * (attempt + 1))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def safe_message(exc, secret=None):
    text = str(exc)
    if secret:
        text = text.replace(secret, "[redacted]")
    text = re.sub(r"https?://\S+", "[redacted-url]", text)
    text = re.sub(r"(?i)(authorization|token|key)\s*[:=]\s*[^\s,;]+", r"\1=[redacted]", text)
    return text[:700]


class CDSBackend:
    def __init__(self):
        import cdsapi
        # Quiet callbacks prevent service messages or signed result URLs from entering logs.
        try:
            self.legacy = cdsapi.Client(quiet=True, debug=False, progress=False, timeout=60, retry_max=1,
                                       info_callback=lambda *a: None, warning_callback=lambda *a: None,
                                       error_callback=lambda *a: None, debug_callback=lambda *a: None)
            self.client = self.legacy.client
            self.client.log_callback = lambda *a, **k: None
        except Exception as exc:
            raise AccessError("CDS client configuration failed; check .cdsapirc without displaying its contents") from None
        if self.legacy.url.rstrip("/") != "https://cds.climate.copernicus.eu/api":
            raise AccessError("The configured endpoint is not CDS; provide CDSAPI_URL/CDSAPI_KEY without rewriting an EWDS/ADS config")

    def access(self):
        try:
            self.client.check_authentication()
            accepted = {(r["id"], r["revision"]) for r in self.client.get_accepted_licences()}
            form = self.client.get_collection(DATASET).form
            required = [r for widget in form if widget.get("type") == "LicenceWidget"
                        for r in widget.get("details", {}).get("licences", [])]
            missing = [{"id": r["id"], "revision": r["revision"]} for r in required
                       if (r["id"], r["revision"]) not in accepted]
            return {"authentication": "passed", "required_licences": [{"id": r["id"], "revision": r["revision"]} for r in required],
                    "missing_licences": missing, "ready": not missing}
        except Exception as exc:
            raise AccessError("CDS authentication/licence inspection failed: " + safe_message(exc, self.legacy.key)) from None

    def retrieve(self, request, target):
        try:
            self.legacy.retrieve(DATASET, request, str(target))
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            message = str(exc).lower()
            if any(word in message for word in ("cost limits", "request too large", "too many fields", "maximum number", "selection limit", "cost limit")):
                raise SizeLimitError("CDS request-size limit") from None
            if status in (401, 403) or "licen" in message or "unauthor" in message:
                raise AccessError("CDS rejected access; verify token and accepted dataset licence") from None
            import requests
            if status in (408, 429, 500, 502, 503, 504) or isinstance(exc, (requests.ConnectionError, requests.Timeout)):
                raise TransientError("CDS transient failure: " + type(exc).__name__) from None
            raise RequestError("CDS acquisition failed: " + safe_message(exc, self.legacy.key)) from None


def check_runtime(access=False):
    runtime = {"at_utc": now(), "python": os.sys.version.split()[0], "packages": versions()}
    runtime["runtime_ready"] = all(runtime["packages"].values())
    if access:
        try:
            runtime["access"] = CDSBackend().access()
        except AccessError as exc:
            runtime["access"] = {"ready": False, "error": str(exc)}
    return runtime


def mesh_bounds(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            stream.readline()
            ne, nn = map(int, stream.readline().split()[:2])
            if nn < 3 or ne < 1:
                raise ValueError("invalid mesh counts")
            nodes = np.loadtxt(stream, max_rows=nn, usecols=(0, 1, 2))
        if nodes.shape != (nn, 3) or not np.array_equal(nodes[:, 0], np.arange(1, nn + 1)):
            raise ValueError("node IDs must be 1..NP in order")
        if not np.isfinite(nodes).all() or np.any(np.abs(nodes[:, 2]) > 90) or np.any((nodes[:, 1] < -180) | (nodes[:, 1] > 360)):
            raise ValueError("invalid geographic node coordinates")
        lon = (nodes[:, 1] + 180) % 360 - 180
        bounds = [float(lon.min()), float(nodes[:, 2].min()), float(lon.max()), float(nodes[:, 2].max())]
        if bounds[2] - bounds[0] > 180:
            raise ValueError("mesh crosses the dateline; make two bbox requests")
        return bounds, nn
    except (OSError, ValueError, IndexError) as exc:
        raise RequestError("Cannot read geographic fort.14: " + str(exc)) from exc


def normalize_request(request, base_dir=None):
    base_dir = Path(base_dir or ".")
    if not isinstance(request, dict):
        raise RequestError("Request must be a JSON object")
    unknown = set(request) - {"schema_version", "start", "end", "products", "mesh", "bbox", "halo_cells"}
    if unknown:
        raise RequestError("Unknown request keys: " + ", ".join(sorted(unknown)))
    if request.get("schema_version") != "era5_request_v1":
        raise RequestError("Expected schema_version era5_request_v1")
    start, end = utc(request.get("start")), utc(request.get("end"))
    if start >= end or start.year < 1940:
        raise RequestError("Require start < end and dates from 1940 onward")
    products = request.get("products", ["wind_10m", "mean_sea_level_pressure"])
    if not isinstance(products, list) or not products or any(not isinstance(p, str) or p not in PRODUCTS for p in products):
        raise RequestError("Use wind_10m and/or mean_sea_level_pressure")
    if len(set(products)) != len(products):
        raise RequestError("Repeated products")
    halo = request.get("halo_cells", 1)
    if isinstance(halo, bool) or not isinstance(halo, int) or halo < 0:
        raise RequestError("halo_cells must be a nonnegative integer")
    if ("mesh" in request) == ("bbox" in request):
        raise RequestError("Specify exactly one of mesh or bbox")
    source_mesh = None
    if "mesh" in request:
        mesh = request["mesh"]
        if not isinstance(mesh, dict) or set(mesh) != {"path", "crs"} or mesh["crs"] != "EPSG:4326":
            raise RequestError("mesh requires path and explicit crs EPSG:4326")
        path = (base_dir / mesh["path"]).resolve()
        bounds, count = mesh_bounds(path)
        source_mesh = {"path": str(path), "sha256": file_hash(path), "nodes": count, "crs": "EPSG:4326"}
    else:
        try:
            bounds = list(map(float, request["bbox"]))
        except (ValueError, TypeError):
            raise RequestError("bbox must be four numeric W,S,E,N coordinates") from None
        if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
            raise RequestError("bbox must be four finite coordinates")
        w, s, e, n = bounds
        if w < -180 or w > 360 or e < -180 or e > 360 or s < -90 or n > 90 or s >= n:
            raise RequestError("Invalid geographic bbox")
        w, e = (w + 180) % 360 - 180, (e + 180) % 360 - 180
        if w >= e or e - w > 180:
            raise RequestError("Dateline/wrapped or >180-degree bbox unsupported; make two requests")
        bounds = [w, s, e, n]
    w, s, e, n = bounds
    area = [min(90.0, math.ceil((n + halo * STEP) / STEP) * STEP),
            math.floor((w - halo * STEP) / STEP) * STEP,
            max(-90.0, math.floor((s - halo * STEP) / STEP) * STEP),
            math.ceil((e + halo * STEP) / STEP) * STEP]
    if area[1] < -180 or area[3] >= 180:
        raise RequestError("Padded area reaches the dateline; reduce halo or make two requests")
    shape = [round((area[0] - area[2]) / STEP) + 1, round((area[3] - area[1]) / STEP) + 1]
    if min(shape) < 2:
        raise RequestError("Area requires at least two source points on each axis")
    result = {"schema_version": "era5_normalized_request_v1", "start": iso(start), "end": iso(end),
              "products": sorted(products), "bbox": bounds, "halo_cells": halo, "area": area,
              "grid_degrees": STEP, "shape": shape, "dataset": DATASET, "source_mesh": source_mesh}
    return result


def variables(request):
    return [v for p in request["products"] for v in PRODUCTS[p]]


def make_chunk(start, end, request):
    times = [start + i * HOUR for i in range(int((end - start) / HOUR))]
    days = sorted({t.strftime("%d") for t in times})
    hours = sorted({t.strftime("%H:00") for t in times})
    if len(days) * len(hours) != len(times):
        raise RequestError("Nonrectangular CDS day/time selection")
    return {"id": start.strftime("%Y%m%dT%H") + "_" + end.strftime("%Y%m%dT%H"), "start": iso(start), "end": iso(end),
            "hours": len(times), "request": {"product_type": ["reanalysis"], "variable": [v[0] for v in variables(request)],
            "year": [start.strftime("%Y")], "month": [start.strftime("%m")], "day": days, "time": hours,
            "data_format": "netcdf", "download_format": "zip", "area": request["area"]}}


def monthly_chunks(request):
    start, end = utc(request["start"]), utc(request["end"])
    chunks = []
    while start < end:
        month_end = start.replace(year=start.year + 1, month=1, day=1, hour=0) if start.month == 12 else start.replace(month=start.month + 1, day=1, hour=0)
        stop = min(month_end, end)
        if start.hour and start.date() < stop.date():
            partial_end = start.replace(hour=0) + dt.timedelta(days=1)
            chunks.append(make_chunk(start, partial_end, request))
            start = partial_end
        full_end = stop.replace(hour=0) if stop.hour and start.date() < stop.date() else stop
        if start < full_end:
            chunks.append(make_chunk(start, full_end, request))
            start = full_end
        if start < stop:
            chunks.append(make_chunk(start, stop, request))
            start = stop
    return chunks


def build_plan(request, base_dir=None, run_dir=None):
    normalized = normalize_request(request, base_dir)
    chunks = monthly_chunks(normalized)
    hours = int((utc(normalized["end"]) - utc(normalized["start"])) / HOUR)
    core = {"schema_version": "era5_download_plan_v1", "request": normalized, "request_hash": digest(normalized),
            "chunks": chunks, "total_hours": hours, "chunk_count": len(chunks)}
    payload = hours * math.prod(normalized["shape"]) * len(variables(normalized)) * 4
    storage = max(128 * 1024**2, payload * 12 + len(chunks) * 1024**2)
    core["estimate"] = {"float32_payload_bytes": payload, "required_free_bytes": storage,
                        "note": "Payload excludes container overhead; storage includes retained raw, canonical, scratch and margin"}
    core["plan_hash"] = digest(core)
    target = Path(run_dir or ".").resolve()
    while not target.exists():
        target = target.parent
    free = shutil.disk_usage(target).free
    core["storage"] = {"free_bytes": free, "ready": free >= storage}
    core["created_utc"] = now()
    return core


def verify_plan(plan):
    if plan.get("schema_version") != "era5_download_plan_v1":
        raise ValidationError("Unknown plan schema")
    content = {k: v for k, v in plan.items() if k not in ("plan_hash", "storage", "created_utc")}
    if digest(content) != plan["plan_hash"] or digest(plan["request"]) != plan["request_hash"]:
        raise ValidationError("Plan identity mismatch")


def save_plan(request_path, run_dir, snapshot=False):
    request_path, root = Path(request_path).resolve(), Path(run_dir).resolve()
    request = read_json(request_path)
    if snapshot:
        request["end"] = iso(utc(request["start"]) + HOUR)
    plan = build_plan(request, request_path.parent, root)
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        existing = root / "download_plan.json"
        if existing.exists() and read_json(existing).get("plan_hash") != plan["plan_hash"]:
            raise RequestError("Run directory belongs to another request; use a new directory")
        # Store the path resolved against its original request location.
        saved = dict(request)
        if "mesh" in saved:
            saved["mesh"] = dict(saved["mesh"], path=plan["request"]["source_mesh"]["path"])
        write_json(root / "request.json", saved)
        write_json(existing, plan)
    return plan


def process_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied: conservatively assume alive.
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextlib.contextmanager
def run_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".era5.lock"
    record = {"pid": os.getpid(), "host": socket.gethostname(), "token": uuid.uuid4().hex, "at_utc": now()}
    for attempt in range(2):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                json.dump(record, stream)
            break
        except FileExistsError:
            existing = read_json(path)
            if attempt == 0 and existing.get("host") == record["host"] and not process_alive(int(existing["pid"])):
                path.unlink()
                continue
            raise RequestError("Another process or host owns this run directory") from None
    try:
        yield
    finally:
        if path.exists() and read_json(path).get("token") == record["token"]:
            path.unlink()


def expected_times(chunk):
    start = np.datetime64(utc(chunk["start"]).replace(tzinfo=None), "ns")
    return start + np.arange(chunk["hours"]) * np.timedelta64(1, "h")


def normalize_dataset(dataset):
    ds = dataset.copy()
    rename = {k: v for k, v in (("valid_time", "time"), ("lat", "latitude"), ("lon", "longitude")) if k in ds and v not in ds}
    ds = ds.rename(rename)
    for name in ("time", "latitude", "longitude"):
        if name not in ds.coords or ds[name].ndim != 1:
            raise ValidationError("Missing or non-1D coordinate: " + name)
    if "expver" in ds.dims:
        if ds.sizes["expver"] != 1:
            raise ValidationError("Conflicting expver branches at common timestamps; no implicit blending")
        ds = ds.isel(expver=0, drop=False)
    if "number" in ds.dims:
        if ds.sizes["number"] != 1:
            raise ValidationError("Ensemble data are outside the reanalysis contract")
        ds = ds.isel(number=0, drop=False)
    ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
    return ds.sortby(["time", "latitude", "longitude"])


def validate_fields(ds, chunk, request):
    for name in ("time", "latitude", "longitude"):
        if name not in ds.coords or ds[name].ndim != 1:
            raise ValidationError("Invalid coordinate " + name)
    actual = np.asarray(ds.time.values).astype("datetime64[ns]")
    if not np.array_equal(actual, expected_times(chunk)):
        raise ValidationError("Missing, duplicate or unintended UTC hours")
    area = request["area"]
    lat = np.arange(request["shape"][0]) * STEP + area[2]
    lon = np.arange(request["shape"][1]) * STEP + area[1]
    if ds.latitude.size != len(lat) or ds.longitude.size != len(lon) or not np.allclose(ds.latitude.values, lat, atol=1e-7, rtol=0) or not np.allclose(ds.longitude.values, lon, atol=1e-7, rtol=0):
        raise ValidationError("Source lattice does not match requested 0.25-degree interpolation support")
    report = {}
    for _, name in variables(request):
        if name not in ds:
            raise ValidationError("Missing requested field " + name)
        field = ds[name]
        if set(field.dims) != {"time", "latitude", "longitude"}:
            raise ValidationError("Unexpected dimensions for " + name)
        unit = str(field.attrs.get("units", ""))
        allowed = {"Pa", "pascal", "pascals"} if name == "msl" else {"m s**-1", "m s-1", "m s^-1", "m/s", "m s^{-1}"}
        if unit not in allowed:
            raise ValidationError("Wrong source units for " + name + ": " + unit)
        expected_parameter = {"u10": 165, "v10": 166, "msl": 151}[name]
        if "GRIB_paramId" in field.attrs and int(field.attrs["GRIB_paramId"]) != expected_parameter:
            raise ValidationError("Wrong source parameter for " + name)
        if field.attrs.get("GRIB_stepType", "instant") != "instant":
            raise ValidationError("Non-instantaneous source field " + name)
        if name in ("u10", "v10") and int(field.attrs.get("GRIB_uvRelativeToGrid", 0)) != 0:
            raise ValidationError("Wind must be earth-relative; no implicit rotation")
        if name in ("u10", "v10") and "GRIB_level" in field.attrs and float(field.attrs["GRIB_level"]) != 10:
            raise ValidationError("Wind is not at 10 m")
        values = np.asarray(field.values)
        if not np.isfinite(values).all() or (name == "msl" and np.any(values <= 0)):
            raise ValidationError("Nonfinite or invalid values in " + name)
        report[name] = {"units": unit, "min": float(values.min()), "max": float(values.max()), "finite_values": int(values.size)}
    if "expver" in ds.dims:
        raise ValidationError("Unresolved expver dimension")
    if "expver" in ds:
        versions_present = sorted(set(map(str, np.asarray(ds.expver.values).ravel())))
    else:
        versions_present = []
    return {"hours": len(actual), "shape": request["shape"], "fields": report,
            "expver": versions_present, "version_origin": "metadata retained" if versions_present else "unknown; source NetCDF lacks expver"}


def payload_members(payload, scratch, limit):
    with Path(payload).open("rb") as stream:
        signature = stream.read(8)
    if signature.startswith(b"PK"):
        with zipfile.ZipFile(payload) as archive:
            members = [r for r in archive.infolist() if not r.is_dir()]
            if not members or sum(r.file_size for r in members) > limit:
                raise ValidationError("Empty ZIP or excessive uncompressed payload")
            result = []
            for index, member in enumerate(members):
                name = PurePosixPath(member.filename.replace("\\", "/"))
                if name.is_absolute() or ".." in name.parts or any(":" in part for part in name.parts) or not member.filename.lower().endswith(".nc"):
                    raise ValidationError("Unsafe or non-NetCDF ZIP member")
                target = Path(scratch) / f"member_{index:03d}.nc"
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                with target.open("rb") as stream:
                    head = stream.read(8)
                if not (head.startswith(b"CDF") or head == b"\x89HDF\r\n\x1a\n"):
                    raise ValidationError("ZIP member is not NetCDF")
                result.append(target)
            return result
    if signature.startswith(b"CDF") or signature == b"\x89HDF\r\n\x1a\n":
        if Path(payload).stat().st_size > limit:
            raise ValidationError("Excessive NetCDF payload")
        return [Path(payload)]
    raise ValidationError("Response is neither ZIP nor NetCDF (possibly an HTML/error body)")


def canonicalize(payload, output, chunk, request):
    output = Path(output)
    expected_bytes = chunk["hours"] * math.prod(request["shape"]) * len(variables(request)) * 4
    with tempfile.TemporaryDirectory(prefix="era5_unpack_", dir=output.parent) as temporary:
        try:
            paths = payload_members(payload, temporary, max(16 * 1024**2, expected_bytes * 40))
        except (zipfile.BadZipFile, RuntimeError):
            raise ValidationError("Invalid or encrypted ZIP response") from None
        sources = []
        for path in paths:
            with xr.open_dataset(path, engine="netcdf4") as source:
                sources.append(normalize_dataset(source.load()))
        try:
            ds = xr.merge(sources, join="exact", compat="equals", combine_attrs="drop_conflicts")
        except (ValueError, xr.MergeError):
            raise ValidationError("Source member coordinates or overlapping fields conflict") from None
        qc = validate_fields(ds, chunk, request)
        names = [n for _, n in variables(request)]
        # Keep expver and scalar number coordinates attached to selected fields.
        ds = ds[names].transpose("time", "latitude", "longitude", missing_dims="ignore")
        ds.attrs.update(schema_version="era5_fields_v1", Conventions="CF-1.10", source_dataset=DATASET,
                        request_hash=digest(request), raw_sha256=file_hash(payload), time_convention="UTC; start inclusive, end exclusive")
        ds.latitude.attrs.update(units="degrees_north", standard_name="latitude")
        ds.longitude.attrs.update(units="degrees_east", standard_name="longitude")
        for name in names:
            if name in ("u10", "v10"):
                ds[name].attrs["source_standard_name"] = ds[name].attrs.get("standard_name", "not provided")
                ds[name].attrs["standard_name"] = "eastward_wind" if name == "u10" else "northward_wind"
                ds[name].attrs["height_above_ground_m"] = 10.0
            ds[name].encoding = {}
        encoding = {name: {"zlib": True, "complevel": 4} for name in names}
        part = output.with_name(output.name + ".part")
        ds.to_netcdf(part, engine="netcdf4", encoding=encoding)
        with xr.open_dataset(part, engine="netcdf4") as written:
            validate_fields(written, chunk, request)
            for name in names:
                if not np.array_equal(ds[name].values, written[name].values):
                    raise ValidationError("Canonical serialization changed source values")
        replace(part, output)
        return qc


def verify_chunk(root, record, chunk, request):
    if record.get("state") != "complete" or record.get("chunk") != chunk:
        raise ValidationError("Incomplete or mismatched chunk record")
    raw, output = Path(root) / record["raw_path"], Path(root) / record["output_path"]
    if not raw.is_file() or not output.is_file() or file_hash(raw) != record["raw_sha256"] or file_hash(output) != record["output_sha256"]:
        raise ValidationError("Committed chunk checksum/presence failure: " + chunk["id"])
    with xr.open_dataset(output, engine="netcdf4") as ds:
        if ds.attrs.get("schema_version") != "era5_fields_v1" or ds.attrs.get("request_hash") != digest(request) or ds.attrs.get("raw_sha256") != record["raw_sha256"]:
            raise ValidationError("Canonical provenance/schema mismatch")
        return validate_fields(ds, chunk, request)


def leaf_chunks(plan, manifest):
    def leaves(chunk):
        children = manifest.get("splits", {}).get(chunk["id"])
        if children is None:
            return [chunk]
        if not children or children[0]["start"] != chunk["start"] or children[-1]["end"] != chunk["end"] or sum(c["hours"] for c in children) != chunk["hours"]:
            raise ValidationError("Invalid adaptive chunk coverage")
        for i, child in enumerate(children):
            if i and children[i - 1]["end"] != child["start"]:
                raise ValidationError("Gap in adaptive chunks")
            if child != make_chunk(utc(child["start"]), utc(child["end"]), plan["request"]):
                raise ValidationError("Adaptive chunk request mismatch")
        return [r for child in children for r in leaves(child)]
    return [leaf for chunk in plan["chunks"] for leaf in leaves(chunk)]


def health(run_dir, write=True):
    root = Path(run_dir)
    report = {"schema_version": "era5_health_report_v1", "at_utc": now(), "pass_all": False, "errors": [], "completed_hours": 0}
    try:
        plan, manifest = read_json(root / "download_plan.json"), read_json(root / "run_manifest.json")
        verify_plan(plan)
        if manifest.get("schema_version") != "era5_run_manifest_v1" or manifest.get("plan_hash") != plan["plan_hash"]:
            raise ValidationError("Manifest/plan identity mismatch")
        source_mesh = plan["request"]["source_mesh"]
        if source_mesh and file_hash(source_mesh["path"]) != source_mesh["sha256"]:
            raise ValidationError("Bound mesh changed since planning")
        leaves = leaf_chunks(plan, manifest)
        qc = []
        for chunk in leaves:
            record = manifest["chunks"].get(chunk["id"])
            if not record or record.get("state") != "complete":
                report["errors"].append("Incomplete chunk: " + chunk["id"])
                continue
            qc.append(verify_chunk(root, record, chunk, plan["request"]))
            report["completed_hours"] += chunk["hours"]
        report.update(expected_hours=plan["total_hours"], completed_chunks=len(qc), expected_chunks=len(leaves),
                      request_hash=plan["request_hash"], plan_hash=plan["plan_hash"], chunk_qc=qc)
        if report["completed_hours"] != plan["total_hours"]:
            report["errors"].append("Incomplete requested period")
        report["pass_all"] = not report["errors"]
    except Exception as exc:
        report["errors"].append(safe_message(exc))
    if write:
        write_json(root / "health_check.json", report)
    return report


def run(run_dir, backend=None, max_chunks=None, retries=3, sleep=time.sleep):
    root = Path(run_dir).resolve()
    if not isinstance(retries, int) or isinstance(retries, bool) or not 1 <= retries <= 3:
        raise RequestError("Acquisition retries must be an integer from 1 through 3")
    with run_lock(root):
        plan = read_json(root / "download_plan.json")
        verify_plan(plan)
        # Rebuild from stored input: changes to source mesh cannot be silently accepted.
        rebuilt = build_plan(read_json(root / "request.json"), root, root)
        if rebuilt["plan_hash"] != plan["plan_hash"]:
            raise RequestError("Run inputs/mesh changed; use a new run directory")
        if shutil.disk_usage(root).free < plan["estimate"]["required_free_bytes"]:
            raise RequestError("Insufficient free storage for this plan")
        if max_chunks is not None and max_chunks < 1:
            raise RequestError("max_chunks must be positive")
        path = root / "run_manifest.json"
        manifest = read_json(path) if path.exists() else {"schema_version": "era5_run_manifest_v1", "plan_hash": plan["plan_hash"],
                    "request_hash": plan["request_hash"], "created_utc": now(), "runtime": versions(), "chunks": {}, "splits": {}}
        if manifest.get("schema_version") != "era5_run_manifest_v1" or manifest.get("plan_hash") != plan["plan_hash"]:
            raise ValidationError("Manifest belongs to another plan")
        for name in ("raw", "fields", "requests"):
            (root / name).mkdir(exist_ok=True)
        new_count, reused_count = 0, 0
        transport = backend

        def status(state, **extra):
            write_json(root / "status.json", {"schema_version": "era5_status_v1", "at_utc": now(), "state": state,
                        "plan_hash": plan["plan_hash"], "new_chunks": new_count, "reused_chunks": reused_count, **extra})

        try:
            queue = leaf_chunks(plan, manifest)
            while queue:
                chunk = queue.pop(0)
                record = manifest["chunks"].get(chunk["id"])
                if record and record.get("state") == "complete":
                    verify_chunk(root, record, chunk, plan["request"])
                    reused_count += 1
                    status("reusing_verified_chunk", chunk=chunk["id"])
                    continue
                if max_chunks is not None and new_count >= max_chunks:
                    status("stopped_between_chunks", next_chunk=chunk["id"])
                    break
                if transport is None:
                    transport = CDSBackend()
                    access = transport.access()
                    write_json(root / "access_check.json", access)
                    if not access["ready"]:
                        raise AccessError("Accept required licences in CDS: " + json.dumps(access["missing_licences"]))
                raw = root / "raw" / (chunk["id"] + ".payload")
                part = raw.with_name(raw.name + ".part")
                output = root / "fields" / (chunk["id"] + ".nc")
                write_json(root / "requests" / (chunk["id"] + ".json"), {"dataset": DATASET, "request": chunk["request"]})
                manifest["chunks"][chunk["id"]] = {"state": "acquiring", "chunk": chunk, "at_utc": now()}
                write_json(path, manifest)
                split = False
                for attempt in range(retries):
                    status("waiting_for_cds_or_transfer", chunk=chunk["id"], attempt=attempt + 1)
                    part.unlink(missing_ok=True)
                    try:
                        transport.retrieve(chunk["request"], part)
                        replace(part, raw)
                        break
                    except SizeLimitError:
                        if chunk["hours"] <= 1:
                            raise
                        begin, end = utc(chunk["start"]), utc(chunk["end"])
                        middle = begin + (chunk["hours"] // 2) * HOUR
                        if begin.date() != end.date():
                            midnight = middle.replace(hour=0)
                            if begin < midnight < end:
                                middle = midnight
                        # monthly_chunks handles partial days in the two intervals.
                        children = []
                        for a, b in ((begin, middle), (middle, end)):
                            sub = dict(plan["request"], start=iso(a), end=iso(b))
                            children.extend(monthly_chunks(sub))
                        manifest["splits"][chunk["id"]] = children
                        manifest["chunks"][chunk["id"]]["state"] = "split"
                        write_json(path, manifest)
                        queue = children + queue
                        split = True
                        break
                    except TransientError:
                        if attempt + 1 >= retries:
                            raise
                        status("retry_backoff", chunk=chunk["id"], attempt=attempt + 1)
                        sleep(2 ** (attempt + 1))
                if split:
                    continue
                status("validating", chunk=chunk["id"])
                qc = canonicalize(raw, output, chunk, plan["request"])
                manifest["chunks"][chunk["id"]] = {"state": "complete", "chunk": chunk, "at_utc": now(),
                    "raw_path": str(raw.relative_to(root)).replace("\\", "/"), "raw_sha256": file_hash(raw),
                    "output_path": str(output.relative_to(root)).replace("\\", "/"), "output_sha256": file_hash(output), "qc": qc}
                write_json(path, manifest)
                new_count += 1
                status("chunk_complete", chunk=chunk["id"])
            report = health(root)
            manifest["updated_utc"] = now()
            manifest["state"] = "complete" if report["pass_all"] else "partial"
            write_json(path, manifest)
            status(manifest["state"], completed_hours=report["completed_hours"], expected_hours=plan["total_hours"])
            return {"new_chunks": new_count, "reused_chunks": reused_count, "health": report}
        except BaseException as exc:
            manifest["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["updated_utc"] = now()
            write_json(path, manifest)
            status(manifest["state"], error=safe_message(exc))
            raise


def snapshot(request_path, run_dir, **kwargs):
    save_plan(request_path, run_dir, snapshot=True)
    return run(run_dir, **kwargs)


def assemble_year(run_dir, year, output):
    import netCDF4
    root, output = Path(run_dir), Path(output)
    if output.exists() or Path(str(output) + ".manifest.json").exists():
        raise RequestError("Annual output already exists")
    with run_lock(root):
        if not health(root)["pass_all"]:
            raise ValidationError("Annual assembly requires a healthy complete run")
        plan, manifest = read_json(root / "download_plan.json"), read_json(root / "run_manifest.json")
        start, end = dt.datetime(year, 1, 1, tzinfo=UTC), dt.datetime(year + 1, 1, 1, tzinfo=UTC)
        if utc(plan["request"]["start"]) > start or utc(plan["request"]["end"]) < end:
            raise RequestError("Run does not contain the entire requested calendar year")
        records = [manifest["chunks"][c["id"]] for c in leaf_chunks(plan, manifest) if utc(c["start"]) < end and utc(c["end"]) > start]
        output.parent.mkdir(parents=True, exist_ok=True)
        part = output.with_name(output.name + ".part")
        sources, offset = [], 0
        try:
            with netCDF4.Dataset(part, "w") as target:
                target.setncatts({"Conventions": "CF-1.10", "schema_version": "era5_annual_fields_v1", "source_dataset": DATASET, "source_plan_hash": plan["plan_hash"]})
                target.createDimension("time", None)
                for name, length in zip(("latitude", "longitude"), plan["request"]["shape"]):
                    target.createDimension(name, length)
                clock = target.createVariable("time", "i8", ("time",))
                clock.units, clock.calendar = "seconds since 1970-01-01 00:00:00", "proleptic_gregorian"
                names = [n for _, n in variables(plan["request"])]
                for record in records:
                    with xr.open_dataset(root / record["output_path"], engine="netcdf4") as src:
                        sel = src.sel(time=slice(np.datetime64(start.replace(tzinfo=None)), np.datetime64(end.replace(tzinfo=None)) - np.timedelta64(1, "h"))).load()
                        count = sel.sizes["time"]
                        if offset == 0:
                            for axis in ("latitude", "longitude"):
                                coord = target.createVariable(axis, "f8", (axis,))
                                coord[:] = sel[axis].values
                                coord.setncatts(dict(sel[axis].attrs))
                            for name in names:
                                field = target.createVariable(name, sel[name].dtype, ("time", "latitude", "longitude"), zlib=True, complevel=4)
                                field.setncatts({k: v for k, v in sel[name].attrs.items() if k != "_FillValue"})
                        clock[offset:offset + count] = sel.time.values.astype("datetime64[s]").astype("int64")
                        for name in names:
                            target[name][offset:offset + count] = sel[name].values
                        offset += count
                        sources.append({"path": record["output_path"], "sha256": record["output_sha256"], "qc": record["qc"]})
            expected = int((end - start) / HOUR)
            with netCDF4.Dataset(part) as ds:
                expected_clock = np.datetime64(start.replace(tzinfo=None), "s").astype("int64") + np.arange(expected) * 3600
                if offset != expected or not np.array_equal(ds["time"][:], expected_clock):
                    raise ValidationError("Annual assembly clock mismatch")
                position = 0
                for record in records:
                    with xr.open_dataset(root / record["output_path"], engine="netcdf4") as original:
                        selected = original.sel(time=slice(np.datetime64(start.replace(tzinfo=None)), np.datetime64(end.replace(tzinfo=None)) - np.timedelta64(1, "h")))
                        count = selected.sizes["time"]
                        for name in names:
                            if not np.array_equal(ds[name][position:position + count], selected[name].values):
                                raise ValidationError("Annual assembly changed field values: " + name)
                        position += count
            replace(part, output)
            report = {"year": year, "hours": offset, "exact_source_values_verified": True, "output_sha256": file_hash(output), "sources": sources, "at_utc": now()}
            write_json(str(output) + ".manifest.json", report)
            return report
        finally:
            part.unlink(missing_ok=True)
