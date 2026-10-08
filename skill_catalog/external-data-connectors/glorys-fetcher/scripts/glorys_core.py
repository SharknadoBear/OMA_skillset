"""Bounded, source-preserving GLORYS12 acquisition through the official Toolbox."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import socket
import time
import uuid

import numpy as np
import xarray as xr

UTC = dt.timezone.utc
PRODUCT = "GLOBAL_MULTIYEAR_PHY_001_030"
DAILY = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
MONTHLY = "cmems_mod_glo_phy_my_0.083deg_P1M-m"
ALIASES = {"water_level": ["zos"], "temperature": ["thetao"],
           "salinity": ["so"], "currents_3d": ["uo", "vo"]}
PACKAGES = ["copernicusmarine", "numpy", "xarray", "h5netcdf", "h5py", "dask", "PyYAML", "matplotlib"]


class GLORYSError(Exception): pass
class RequestError(GLORYSError): pass
class AccessError(GLORYSError): pass
class ValidationError(GLORYSError): pass
class TransientError(GLORYSError): pass


def now(): return dt.datetime.now(UTC).isoformat()


def iso(value):
    if isinstance(value, (str, np.datetime64)):
        return np.datetime_as_string(np.datetime64(str(value).replace("Z", ""), "us"), unit="us").rstrip("0").rstrip(".") + "Z"
    return value.astimezone(UTC).isoformat(timespec="microseconds" if value.microsecond else "seconds").replace("+00:00", "Z")


def utc(value):
    try:
        v = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if v.tzinfo is None or v.utcoffset() != dt.timedelta(0): raise ValueError()
        return v
    except (AttributeError, ValueError):
        raise RequestError("Dates must be ISO-8601 UTC with Z or +00:00") from None


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""): h.update(block)
    return h.hexdigest()


def replace(src, dst):
    for attempt in range(6):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 5: raise
            time.sleep(.15 * (attempt + 1))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as f: return json.load(f)


def json_value(v):
    if isinstance(v, dict): return {str(k): json_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)): return [json_value(x) for x in v]
    if isinstance(v, np.generic): return json_value(v.item())
    if isinstance(v, (str, int, bool)) or v is None: return v
    if isinstance(v, float): return v if math.isfinite(v) else str(v)
    return str(v)


@contextlib.contextmanager
def quiet_sdk():
    """Do not put SDK service URLs or authentication responses into run evidence."""
    names = ["copernicusmarine", "boto3", "botocore", "urllib3", "fsspec", "s3fs"]
    levels = {n: logging.getLogger(n).level for n in names}
    for n in names: logging.getLogger(n).setLevel(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try: yield
        finally:
            for n, level in levels.items(): logging.getLogger(n).setLevel(level)


def credentials_present():
    directory = Path(os.environ.get("COPERNICUSMARINE_CREDENTIALS_DIRECTORY", Path.home() / ".copernicusmarine"))
    return ((bool(os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")) and
             bool(os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD"))) or
            (directory / ".copernicusmarine-credentials").is_file())


def check_runtime(access=False):
    versions = {}
    for name in PACKAGES:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    result = {"at_utc": now(), "python": os.sys.version.split()[0], "packages": versions,
              "runtime_ready": all(versions.values()) and versions["copernicusmarine"] == "2.5.0" and os.sys.version_info[:2] == (3, 11)}
    if access:
        if not credentials_present():
            result["access"] = {"ready": False, "error": "Local Marine credentials missing; run copernicusmarine login"}
        else:
            import copernicusmarine as cm
            try:
                with quiet_sdk(): ready = bool(cm.login(check_credentials_valid=True))
                result["access"] = {"ready": ready, "error": None if ready else "Marine authentication rejected"}
            except Exception:
                result["access"] = {"ready": False, "error": "Marine authentication check failed"}
    return result


def classify_failure(exc):
    status = getattr(getattr(exc, "response", None), "status_code", None)
    name = type(exc).__name__.lower()
    text = str(exc).lower()  # Classification only; never persist raw server messages.
    if status in (401, 403) or any(x in name + text for x in ("credential", "unauthor", "forbidden", "authentication")):
        return AccessError("Marine authentication rejected; configure Toolbox credentials locally")
    if status in (408, 429, 500, 502, 503, 504) or any(x in name for x in ("timeout", "connection", "network")):
        return TransientError("Marine transient transport failure (" + type(exc).__name__ + ")")
    return RequestError("Marine request failed (" + type(exc).__name__ + "); inspect request and catalogue coverage")


class OfficialBackend:
    def __init__(self, service="arco-geo-series"):
        if service not in ("arco-geo-series", "arco-time-series"): raise RequestError("Unsupported ARCO service")
        import copernicusmarine as cm
        self.cm = cm
        self.service = service

    def catalogue(self, dataset_id, dataset_version=None):
        try:
            with quiet_sdk():
                catalogue = self.cm.describe(dataset_id=dataset_id, show_all_versions=bool(dataset_version), disable_progress_bar=True, raise_on_error=True)
            product = next(p for p in catalogue.products if p.product_id == PRODUCT)
            dataset = next(d for d in product.datasets if d.dataset_id == dataset_id)
            version = next((v for v in dataset.versions if v.label == dataset_version), None) if dataset_version else dataset.versions[0]
            if version is None: raise RequestError("Frozen dataset version is unavailable")
            part = next((p for p in version.parts if p.name == "default"), version.parts[0])
            service = next(s for s in part.services if s.service_name == self.service)
            return dataset, version, part, service
        except GLORYSError: raise
        except Exception as exc: raise classify_failure(exc) from None

    def inventory(self, dataset_id, dataset_version=None):
        dataset, version, part, service = self.catalogue(dataset_id, dataset_version)
        public = {"schema_version": "glorys_inventory_v1", "at_utc": now(), "product_id": PRODUCT,
                  "dataset_id": dataset_id, "dataset_version": version.label, "dataset_part": part.name,
                  "service": self.service, "toolbox_version": self.cm.__version__,
                  "catalogue_variables": [json_value(v.model_dump()) for v in service.variables],
                  "coordinates_verified": False}
        if not credentials_present():
            public["access_note"] = "Catalogue inspected; exact source coordinates require local Toolbox authentication"
            return public
        try:
            with quiet_sdk():
                ds = self.cm.open_dataset(dataset_id=dataset_id, dataset_version=version.label,
                    dataset_part=part.name, service=self.service, raise_if_updating=True)
            with ds:
                public["coordinates"] = {name: [iso(x) for x in a.values] if name == "time" else json_value(a.values)
                                         for name, a in ds.coords.items() if name in ("time", "depth", "latitude", "longitude")}
                public["coordinate_attributes"] = {n: json_value(ds[n].attrs) for n in public["coordinates"]}
                public["variables"] = {name: {"dimensions": list(a.dims), "dtype": str(a.dtype),
                    "attributes": json_value(a.attrs)} for name, a in ds.data_vars.items()}
                public["source_attributes"] = json_value(ds.attrs)
                public["coordinates_verified"] = True
            return public
        except Exception as exc: raise classify_failure(exc) from None

    def subset(self, plan, chunk, target, dry_run=False):
        if not credentials_present(): raise AccessError("Local Marine credentials missing; run copernicusmarine login")
        c = plan["coordinates"]
        args = dict(dataset_id=plan["source"]["dataset_id"], dataset_version=plan["source"]["dataset_version"],
                    dataset_part=plan["source"]["dataset_part"], service=plan["source"]["service"],
                    variables=plan["request"]["variables"], minimum_longitude=c["longitude"][0],
                    maximum_longitude=c["longitude"][-1], minimum_latitude=c["latitude"][0],
                    maximum_latitude=c["latitude"][-1], start_datetime=chunk["times"][0],
                    end_datetime=chunk["times"][-1], coordinates_selection_method="inside",
                    output_directory=str(Path(target).parent), output_filename=Path(target).name,
                    file_format="netcdf", overwrite=False, disable_progress_bar=True,
                    raise_if_updating=True, dry_run=dry_run)
        if "depth" in c: args.update(minimum_depth=c["depth"][0], maximum_depth=c["depth"][-1])
        try:
            with quiet_sdk(): result = self.cm.subset(**args)
            if dry_run:
                data = result.model_dump(mode="json") if hasattr(result, "model_dump") else {}
                # Only numeric estimates are portable; never persist URLs or credentials.
                estimates = {k: v for k, v in data.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                for key in ("file_size", "data_transfer_size"):
                    size = getattr(result, key, None)
                    if isinstance(size, (int, float)): estimates[key + "_MB"] = size
                return estimates
            if not Path(target).is_file(): raise ValidationError("Toolbox did not publish the requested temporary file")
        except GLORYSError: raise
        except Exception as exc: raise classify_failure(exc) from None


def inventory(dataset_id=DAILY, output=None, backend=None, dataset_version=None, service="arco-geo-series"):
    if dataset_id not in (DAILY, MONTHLY): raise RequestError("v1 accepts the GLORYS12 daily or monthly dataset")
    backend = backend or OfficialBackend(service=service)
    for attempt in range(1, 4):
        try:
            result = backend.inventory(dataset_id, dataset_version=dataset_version)
            break
        except TransientError:
            if attempt == 3: raise
            time.sleep(attempt * 2)
    if output: write_json(output, result)
    return result


def mesh_bounds(path):
    try:
        with Path(path).open(encoding="utf-8") as f:
            f.readline(); ne, nn = map(int, f.readline().split()[:2])
            a = np.loadtxt(f, max_rows=nn, usecols=(0, 1, 2))
        if ne < 1 or nn < 3 or a.shape != (nn, 3) or not np.array_equal(a[:, 0], np.arange(1, nn + 1)):
            raise ValueError()
        if not np.isfinite(a).all() or np.any(np.abs(a[:, 2]) > 90) or np.any(np.abs(a[:, 1]) > 360): raise ValueError()
        lon = (a[:, 1] + 180) % 360 - 180
        if lon.max() - lon.min() > 180: raise RequestError("Dateline crossing: use two explicit regional requests")
        return [float(lon.min()), float(a[:, 2].min()), float(lon.max()), float(a[:, 2].max())], nn
    except RequestError: raise
    except (OSError, ValueError, IndexError): raise RequestError("Invalid geographic fort.14 nodes or ordered IDs") from None


def normalize_request(request, base_dir="."):
    allowed = {"schema_version", "dataset_id", "variables", "start", "end", "bbox", "mesh", "depth", "halo_cells", "chunk_target_mib", "service", "chunk_mode"}
    if set(request) - allowed: raise RequestError("Unknown request fields: " + ", ".join(sorted(set(request) - allowed)))
    if request.get("schema_version") != "glorys_request_v1": raise RequestError("Expected glorys_request_v1")
    dataset = request.get("dataset_id", DAILY)
    if dataset not in (DAILY, MONTHLY): raise RequestError("Unsupported GLORYS12 dataset ID")
    start, end = utc(request.get("start")), utc(request.get("end"))
    if end <= start: raise RequestError("End must follow start")
    fields = request.get("variables")
    if not isinstance(fields, list) or not fields or any(not isinstance(v, str) or not v for v in fields):
        raise RequestError("Provide a nonempty variable-name list")
    variables = []
    for field in fields:
        for name in ALIASES.get(field, [field]):
            if name not in variables: variables.append(name)
    if ("mesh" in request) == ("bbox" in request): raise RequestError("Provide exactly one of mesh or bbox")
    result = {"schema_version": "glorys_request_v1", "dataset_id": dataset, "variables": variables,
              "start": iso(start), "end": iso(end)}
    for key, choices in (("service", ("arco-geo-series", "arco-time-series")), ("chunk_mode", ("calendar_month", "bounded_batch"))):
        if key in request:
            if request[key] not in choices: raise RequestError("Unsupported " + key)
            result[key] = request[key]
    if "mesh" in request:
        m = request["mesh"]
        if not isinstance(m, dict) or set(m) != {"path", "crs"} or m["crs"] != "EPSG:4326":
            raise RequestError("Mesh requires path and crs=EPSG:4326")
        path = (Path(base_dir) / m["path"]).resolve()
        bounds, nodes = mesh_bounds(path)
        result["mesh"] = {"sha256": file_hash(path), "nodes": nodes, "crs": "EPSG:4326"}
    else: bounds = request["bbox"]
    try:
        b = [float(x) for x in bounds]
        if len(b) != 4 or not all(math.isfinite(x) for x in b) or not (-180 <= b[0] < b[2] <= 180 and -90 <= b[1] < b[3] <= 90):
            raise ValueError()
    except (ValueError, TypeError): raise RequestError("bbox must be [west,south,east,north] without dateline crossing") from None
    result["bbox"] = b
    halo = request.get("halo_cells", 1)
    if type(halo) is not int or halo < 0: raise RequestError("halo_cells must be a nonnegative integer")
    result["halo_cells"] = halo
    if "depth" in request:
        try:
            z = [float(x) for x in request["depth"]]
            if len(z) != 2 or not all(math.isfinite(x) for x in z) or not 0 <= z[0] <= z[1]: raise ValueError()
        except (ValueError, TypeError): raise RequestError("depth must be positive-down [minimum,maximum] metres") from None
        result["depth"] = z
    target = request.get("chunk_target_mib", 64)
    if type(target) is not int or not 1 <= target <= 1024: raise RequestError("chunk_target_mib must be an integer in 1..1024")
    result["chunk_target_mib"] = target
    return result


def enclosing_axis(values, lo, hi, halo):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or a.size < 2 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0):
        raise ValidationError("Source spatial coordinate must be finite, increasing and one-dimensional")
    if lo < a[0] or hi > a[-1]: raise RequestError("Requested region extends outside source coverage")
    left = max(0, int(np.searchsorted(a, lo, side="right")) - 1 - halo)
    right = min(a.size - 1, int(np.searchsorted(a, hi, side="left")) + halo)
    return a[left:right + 1].tolist()


def build_plan(request, base_dir=".", run_dir=".", backend=None, source_inventory=None):
    r = normalize_request(request, base_dir)
    inv = source_inventory or inventory(r["dataset_id"], backend=backend, service=r.get("service", "arco-geo-series"))
    if not inv.get("coordinates_verified"): raise AccessError("Exact coordinate inventory pending local Toolbox authentication")
    if inv["dataset_id"] != r["dataset_id"]: raise ValidationError("Inventory/request dataset mismatch")
    if "service" in r and r["service"] != inv["service"]: raise ValidationError("Inventory/request service mismatch")
    fields = inv["variables"]
    for name in r["variables"]:
        if name not in fields: raise RequestError("Variable absent from selected source: " + name)
        if set(fields[name]["dimensions"]) - {"time", "depth", "latitude", "longitude"}:
            raise RequestError("Unsupported dimensions for source field: " + name)
        if not np.issubdtype(np.dtype(fields[name]["dtype"]), np.number): raise RequestError("Field is not numeric: " + name)
    original = inv["coordinates"]
    times = [utc(x) for x in original["time"]]
    if not times or any(b <= a for a, b in zip(times, times[1:])): raise ValidationError("Source time coordinate is not strictly increasing")
    start, end = utc(r["start"]), utc(r["end"])
    first_period = times[0].replace(hour=0, minute=0, second=0, microsecond=0)
    if r["dataset_id"] == MONTHLY: first_period = first_period.replace(day=1)
    last_period = times[-1].replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)
    if r["dataset_id"] == MONTHLY:
        last_period = (times[-1].replace(day=1, hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=32)).replace(day=1)
    if start < first_period or end > last_period: raise RequestError("Time request extends outside source averaging-period coverage")
    selected = [x for x in times if start <= x < end]
    if not selected: raise RequestError("Requested period contains no source timestamps")
    if r["dataset_id"] == DAILY:
        phase = times[0] - first_period
        expected = []; clock = start.replace(hour=0, minute=0, second=0, microsecond=0) + phase
        while clock < end:
            if clock >= start: expected.append(clock)
            clock += dt.timedelta(days=1)
        if selected != expected: raise ValidationError("Source daily time gaps or inconsistent timestamp phase")
    else:
        if any((b.year * 12 + b.month) - (a.year * 12 + a.month) != 1 for a, b in zip(selected, selected[1:])):
            raise ValidationError("Source monthly time gap")
        month = start.replace(day=1,hour=0,minute=0,second=0,microsecond=0)
        while month < end:
            following = (month + dt.timedelta(days=32)).replace(day=1)
            if start <= month and following <= end and not any(month <= value < following for value in selected):
                raise ValidationError("A fully requested month has no source record")
            month = following
    b = r["bbox"]
    coords = {"longitude": enclosing_axis(original["longitude"], b[0], b[2], r["halo_cells"]),
              "latitude": enclosing_axis(original["latitude"], b[1], b[3], r["halo_cells"]),
              "time": [iso(x) for x in selected]}
    if any("depth" in fields[n]["dimensions"] for n in r["variables"]):
        depth = np.asarray(original.get("depth", []), dtype=float)
        if depth.size == 0 or not np.isfinite(depth).all() or np.any(np.diff(depth) <= 0): raise ValidationError("Invalid source depth coordinate")
        low, high = r.get("depth", [float(depth[0]), float(depth[-1])])
        coords["depth"] = depth[(depth >= low) & (depth <= high)].tolist()
        if not coords["depth"]: raise RequestError("No source depth level within requested bounds")
    per_time = 0
    for n in r["variables"]:
        dims = fields[n]["dimensions"]
        per_time += math.prod(len(coords[d]) for d in dims if d != "time") * np.dtype(fields[n]["dtype"]).itemsize
    limit = r["chunk_target_mib"] * 1048576
    if per_time > limit: raise RequestError("One source timestamp exceeds chunk memory budget; reduce area/depth or increase chunk_target_mib")
    max_times = max(1, limit // max(per_time, 1))
    chunks = []
    batched = r.get("chunk_mode") == "bounded_batch"
    for timestamp in coords["time"]:
        month = timestamp[:7]
        if not chunks or (not batched and chunks[-1]["month"] != month) or len(chunks[-1]["times"]) >= max_times:
            chunks.append({"id": f"batch_{len(chunks):04d}", "times": []} if batched else
                          {"id": f"{month}_{len(chunks):04d}", "month": month, "times": []})
        chunks[-1]["times"].append(timestamp)
    source = {k: inv[k] for k in ("product_id", "dataset_id", "dataset_version", "dataset_part", "service", "toolbox_version")}
    source["metadata_sha256"] = digest({"source": source, "variables": {n: fields[n] for n in r["variables"]}, "coordinates": coords,
                                          "coordinate_attributes": inv.get("coordinate_attributes", {})})
    decoded = per_time * len(selected)
    existing = Path(run_dir).resolve()
    while not existing.exists(): existing = existing.parent
    free = shutil.disk_usage(existing).free
    peak = 4 * decoded + 64 * 1048576
    plan = {"schema_version": "glorys_plan_v1", "request": r, "request_hash": digest(r), "source": source,
            "coordinates": coords, "variables": {n: fields[n] for n in r["variables"]},
            "coordinate_attributes": {n: inv.get("coordinate_attributes", {}).get(n, {}) for n in coords},
            "chunks": chunks, "estimate": {"decoded_bytes": decoded, "peak_disk_bytes": peak,
            "network_transfer_bytes": None, "network_note": "ARCO block transfer can exceed final subset size"},
            "storage": {"free_bytes": free, "ready": free >= peak},
            "time_note": "Source timestamps preserved; daily means described by producer as noon-centered; no clock shifting",
            "selected_counts": {k: len(v) for k, v in coords.items()}}
    plan["plan_hash"] = plan_identity(plan)
    return plan, inv


def plan_identity(plan):
    return digest({k: v for k, v in plan.items() if k not in ("plan_hash", "storage", "toolbox_estimate", "created_utc", "selected_counts")})


def save_plan(request_path, run_dir, backend=None):
    p = Path(request_path).resolve(); run_dir = Path(run_dir).resolve()
    plan, inv = build_plan(read_json(p), p.parent, run_dir, backend=backend)
    run_dir.mkdir(parents=True, exist_ok=True)
    current = run_dir / "download_plan.json"
    if current.exists() and read_json(current)["plan_hash"] != plan["plan_hash"]:
        raise ValidationError("Run directory is bound to another request/source; use a new directory")
    if "mesh" in read_json(p):
        original = read_json(p)["mesh"]["path"]
        mesh = (p.parent / original).resolve()
        write_json(run_dir / "mesh_reference.json", {"path": os.path.relpath(mesh, run_dir), "sha256": file_hash(mesh)})
    write_json(run_dir / "request.json", plan["request"])
    write_json(run_dir / "inventory.json", inv)
    write_json(current, plan)
    return plan


def validate_file(path, plan, clocks):
    """Independent decoded read, bounded to one chunk. No fill, clip or interpolation."""
    try:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            if set(ds.data_vars) != set(plan["variables"]): raise ValidationError("Output fields differ from plan")
            for name, selected in plan["coordinates"].items():
                want = clocks if name == "time" else selected
                if name not in ds.coords: raise ValidationError("Missing coordinate: " + name)
                actual = [iso(v) for v in ds[name].values] if name == "time" else ds[name].values.tolist()
                if actual != want: raise ValidationError("Output coordinate/clock mismatch: " + name)
                for key, value in plan["coordinate_attributes"].get(name, {}).items():
                    if json_value(ds[name].attrs.get(key)) != value: raise ValidationError("Coordinate attribute changed: " + name + "/" + key)
            diagnostics = {}
            for name, spec in plan["variables"].items():
                a = ds[name]
                if list(a.dims) != spec["dimensions"]: raise ValidationError("Field dimensions changed: " + name)
                if str(a.dtype) != spec["dtype"]: raise ValidationError("Decoded field dtype changed: " + name)
                for key, value in spec["attributes"].items():
                    if json_value(a.attrs.get(key)) != value: raise ValidationError("Field attribute changed: " + name + "/" + key)
                values = a.values
                if np.isinf(values).any(): raise ValidationError("Infinite field value: " + name)
                valid = np.isfinite(values)
                if name in ("zos", "thetao", "so", "uo", "vo", "bottomT") and not valid.any():
                    raise ValidationError("No valid ocean coverage for " + name)
                if "time" in a.dims and name in ("zos", "thetao", "so", "uo", "vo"):
                    axes = tuple(i for i, d in enumerate(a.dims) if d != "time")
                    if not valid.any(axis=axes).all(): raise ValidationError("An entire source record lacks ocean coverage: " + name)
                lo = float(values[valid].min()) if valid.any() else None
                hi = float(values[valid].max()) if valid.any() else None
                # Warnings are evidence, never edits to the published source values.
                limits = {"zos": (-10, 10), "thetao": (-4, 45), "bottomT": (-4, 45), "so": (0, 50), "uo": (-10, 10), "vo": (-10, 10)}
                excursion = name in limits and valid.any() and (lo < limits[name][0] or hi > limits[name][1])
                diagnostics[name] = {"dimensions": list(a.dims), "shape": list(a.shape), "units": a.attrs.get("units"),
                    "valid_count": int(valid.sum()), "missing_count": int((~valid).sum()), "minimum": lo, "maximum": hi,
                    "mask_sha256": hashlib.sha256(np.packbits(~valid).tobytes()).hexdigest(),
                    "values_sha256": hashlib.sha256(np.where(valid, values, 0).tobytes()).hexdigest(),
                    "plausibility_excursion": bool(excursion)}
            if "uo" in ds and "vo" in ds:
                if ds.uo.dims != ds.vo.dims or ds.uo.shape != ds.vo.shape: raise ValidationError("Horizontal vector grids are misaligned")
                diagnostics["vector_alignment"] = {"same_grid": True,
                    "mask_difference_count": int(np.count_nonzero(np.isfinite(ds.uo.values) != np.isfinite(ds.vo.values)))}
            return {"clocks": clocks, "fields": diagnostics, "source_attributes": json_value(ds.attrs)}
    except GLORYSError: raise
    except Exception as exc: raise ValidationError("Unreadable or invalid NetCDF (" + type(exc).__name__ + ")") from None


def chunk_paths(run_dir, chunk):
    file = Path(run_dir) / "chunks" / (chunk["id"] + ".nc")
    return file, file.with_suffix(".receipt.json")


def receipt_identity(plan):
    return {"plan_hash": plan["plan_hash"], "request_hash": plan["request_hash"], "source": plan["source"]}


def verify_mesh_reference(run_dir, plan):
    if "mesh" not in plan["request"]: return
    path = Path(run_dir) / "mesh_reference.json"
    if not path.exists(): raise ValidationError("Mesh reference is missing")
    m = read_json(path)
    if file_hash(Path(run_dir) / m["path"]) != m["sha256"] or m["sha256"] != plan["request"]["mesh"]["sha256"]:
        raise ValidationError("Mesh changed after planning")


def verify_committed(run_dir, plan, chunk):
    path, record = chunk_paths(run_dir, chunk)
    if not record.exists(): return None
    try:
        r = read_json(record)
        if any(r.get(k) != v for k, v in receipt_identity(plan).items()): raise ValidationError("Committed chunk belongs to another request/source")
        if r["times"] != chunk["times"] or not path.exists() or file_hash(path) != r["sha256"]:
            raise ValidationError("Committed chunk is missing or corrupt: " + chunk["id"])
        diagnostics = validate_file(path, plan, chunk["times"])
        if diagnostics != r["validation"]: raise ValidationError("Committed chunk validation changed: " + chunk["id"])
        return r
    except GLORYSError: raise
    except Exception: raise ValidationError("Invalid chunk receipt: " + chunk["id"]) from None


@contextlib.contextmanager
def run_lock(run_dir):
    lock = Path(run_dir) / "writer.lock"
    try:
        with lock.open("x", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "at_utc": now()}, f)
    except FileExistsError:
        raise ValidationError("Run has a writer.lock; confirm its recorded process is stopped before removing the lock") from None
    try: yield
    finally: lock.unlink(missing_ok=True)


def frozen_source_check(plan, backend):
    inv = inventory(plan["source"]["dataset_id"], backend=backend, dataset_version=plan["source"]["dataset_version"], service=plan["source"]["service"])
    if not inv.get("coordinates_verified"): raise AccessError("Exact source access is unavailable")
    original = inv["coordinates"]
    selected = plan["coordinates"]
    for coord, values in selected.items():
        axis = original.get(coord, [])
        if coord == "time":
            current = [value for value in axis if utc(values[0]) <= utc(value) <= utc(values[-1])]
        else: current = [value for value in axis if values[0] <= value <= values[-1]]
        if current != values: raise ValidationError("Frozen source coordinates changed: " + coord)
    source = {k: inv[k] for k in ("product_id", "dataset_id", "dataset_version", "dataset_part", "service", "toolbox_version")}
    source["metadata_sha256"] = digest({"source": source, "variables": {n: inv["variables"].get(n) for n in plan["variables"]},
        "coordinates": selected, "coordinate_attributes": inv.get("coordinate_attributes", {})})
    if source != plan["source"]: raise ValidationError("Frozen source identity or variable metadata changed")
    return inv


def assemble(run_dir, plan):
    """Append decoded chunks to unlimited time; only one chunk is held in memory."""
    import h5netcdf
    run_dir = Path(run_dir)
    final = run_dir / "subset.nc"
    temporary = run_dir / ("subset." + uuid.uuid4().hex + ".part.nc")
    offset = 0
    try:
        for index, chunk in enumerate(plan["chunks"]):
            r = verify_committed(run_dir, plan, chunk)
            if not r: raise ValidationError("Cannot assemble incomplete chunks")
            path, _ = chunk_paths(run_dir, chunk)
            with xr.open_dataset(path, engine="h5netcdf") as source:
                ds = source.load()
                if index == 0:
                    for name in ds.variables: ds[name].encoding = {}
                    encoding = {"time": {"units": "seconds since 1970-01-01", "calendar": "proleptic_gregorian", "dtype": "int64"}}
                    ds.to_netcdf(temporary, engine="h5netcdf", unlimited_dims=["time"], encoding=encoding)
                else:
                    with h5netcdf.File(temporary, "a") as out:
                        count = ds.sizes["time"]
                        out.resize_dimension("time", offset + count)
                        for name, a in ds.variables.items():
                            if "time" not in a.dims: continue
                            values = a.values
                            if name == "time": values = values.astype("datetime64[s]").astype("int64")
                            target = [slice(None)] * a.ndim
                            target[a.dims.index("time")] = slice(offset, offset + count)
                            out.variables[name][tuple(target)] = values
                offset += ds.sizes["time"]
        replace(temporary, final)
        return final
    finally: temporary.unlink(missing_ok=True)


def fetch(run_dir, backend=None, max_chunks=None, retry_delay=2.0):
    run_dir = Path(run_dir).resolve()
    plan = read_json(run_dir / "download_plan.json")
    if plan_identity(plan) != plan["plan_hash"]: raise ValidationError("Plan was altered after selection")
    verify_mesh_reference(run_dir, plan)
    backend = backend or OfficialBackend(service=plan["source"]["service"])
    result = {"at_utc": now(), "plan_hash": plan["plan_hash"], "new_chunks": 0, "reused_chunks": 0, "complete": False}
    with run_lock(run_dir):
        try:
            frozen_source_check(plan, backend)
            (run_dir / "chunks").mkdir(exist_ok=True)
            reusable = [verify_committed(run_dir, plan, c) for c in plan["chunks"]]
            if (not all(reusable) or not (run_dir / "subset.nc").exists()) and shutil.disk_usage(run_dir).free < plan["estimate"]["peak_disk_bytes"]:
                raise RequestError("Insufficient conservative peak disk headroom")
            for chunk in plan["chunks"]:
                existing = verify_committed(run_dir, plan, chunk)
                if existing:
                    result["reused_chunks"] += 1
                    continue
                if max_chunks is not None and result["new_chunks"] >= max_chunks: break
                path, record = chunk_paths(run_dir, chunk)
                # A file without a receipt was never committed. Reacquire it.
                path.unlink(missing_ok=True)
                for stale in path.parent.glob(path.stem + ".*.part.nc"):
                    middle = stale.name[len(path.stem)+1:-len(".part.nc")]
                    if re.fullmatch("[0-9a-f]{32}", middle): stale.unlink()
                for attempt in range(1, 4):
                    temporary = path.with_name(path.stem + "." + uuid.uuid4().hex + ".part.nc")
                    try:
                        backend.subset(plan, chunk, temporary)
                        validation = validate_file(temporary, plan, chunk["times"])
                        receipt = {**receipt_identity(plan), "times": chunk["times"], "sha256": file_hash(temporary),
                            "bytes": temporary.stat().st_size, "committed_utc": now(), "attempts": attempt, "validation": validation}
                        replace(temporary, path)
                        write_json(record, receipt)
                        result["new_chunks"] += 1
                        break
                    except TransientError:
                        if attempt == 3: raise
                        time.sleep(retry_delay * attempt)
                    finally: temporary.unlink(missing_ok=True)
                result["completed_chunks"] = result["new_chunks"] + result["reused_chunks"]
                result["state"] = "acquiring"
                write_json(run_dir / "progress.json", result)
            receipts = []
            for chunk in plan["chunks"]:
                r = verify_committed(run_dir, plan, chunk)
                if r: receipts.append({"file": "chunks/" + chunk["id"] + ".nc", **r})
            result["completed_chunks"] = len(receipts)
            if len(receipts) == len(plan["chunks"]):
                final = run_dir / "subset.nc"
                old = read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
                if old.get("assembled") and final.exists():
                    if file_hash(final) != old["assembled"]["sha256"]: raise ValidationError("Committed assembly is corrupt")
                else: assemble(run_dir, plan)
                result["complete"] = True
            manifest = {"schema_version": "glorys_manifest_v1", **receipt_identity(plan), "at_utc": now(),
                "request": plan["request"], "source_grid": "Published regular latitude/longitude grid; not native NEMO C grid",
                "chunks": receipts, "complete": result["complete"], "runtime": check_runtime(),
                "assembled": {"file": "subset.nc", "sha256": file_hash(run_dir / "subset.nc")} if result["complete"] else None}
            write_json(run_dir / "manifest.json", manifest)
            if result["complete"]: health(run_dir)
            result["state"] = "complete" if result["complete"] else "paused_after_chunk_limit"
            write_json(run_dir / "progress.json", result)
            return result
        except GLORYSError as exc:
            result.update(state="failed", error_type=type(exc).__name__, error=str(exc))
            write_json(run_dir / "progress.json", result)
            raise
        except Exception as exc:
            message = "Local acquisition or assembly failed (" + type(exc).__name__ + ")"
            result.update(state="failed", error_type="ValidationError", error=message)
            write_json(run_dir / "progress.json", result)
            raise ValidationError(message) from None


def health(run_dir, output=None):
    """Read receipts, all chunk values, and assembled slices independently of acquisition."""
    run_dir = Path(run_dir)
    plan = read_json(run_dir / "download_plan.json")
    if plan_identity(plan) != plan["plan_hash"]: raise ValidationError("Plan integrity failed")
    verify_mesh_reference(run_dir, plan)
    manifest = read_json(run_dir / "manifest.json")
    if any(manifest.get(k) != v for k, v in receipt_identity(plan).items()): raise ValidationError("Manifest identity differs from plan")
    complete = len(manifest["chunks"]) == len(plan["chunks"]) and manifest["complete"]
    warnings = []; records = []
    checked = 0
    ds = None
    try:
        if complete:
            final = run_dir / manifest["assembled"]["file"]
            if file_hash(final) != manifest["assembled"]["sha256"]: raise ValidationError("Assembly hash mismatch")
            ds = xr.open_dataset(final, engine="h5netcdf")
            if [iso(v) for v in ds.time.values] != plan["coordinates"]["time"]: raise ValidationError("Assembly clock mismatch")
            if set(ds.data_vars) != set(plan["variables"]): raise ValidationError("Assembly variable mismatch")
            for coord in plan["coordinates"]:
                if coord != "time" and ds[coord].values.tolist() != plan["coordinates"][coord]: raise ValidationError("Assembly grid mismatch")
        offset = 0
        for chunk in plan["chunks"]:
            r = verify_committed(run_dir, plan, chunk)
            if not r: continue
            records.append({"file": "chunks/" + chunk["id"] + ".nc", **r})
            checked += 1
            for name, diagnostic in r["validation"]["fields"].items():
                if diagnostic.get("plausibility_excursion"): warnings.append({"chunk": chunk["id"], "variable": name, "minimum": diagnostic["minimum"], "maximum": diagnostic["maximum"]})
            if ds is not None:
                with xr.open_dataset(chunk_paths(run_dir, chunk)[0], engine="h5netcdf") as native:
                    for name in native.variables:
                        a = ds[name].isel(time=slice(offset, offset + len(chunk["times"]))) if "time" in native[name].dims else ds[name]
                        if a.dims != native[name].dims or a.dtype != native[name].dtype or json_value(a.attrs) != json_value(native[name].attrs):
                            raise ValidationError("Assembly changed dimensions/attributes: " + name)
                        x, y = a.values, native[name].values
                        if np.issubdtype(x.dtype, np.number):
                            equal = np.array_equal(x, y, equal_nan=True)
                        else: equal = np.array_equal(x, y)
                        if not equal: raise ValidationError("Assembly changed values or missing patterns: " + name)
                offset += len(chunk["times"])
        if records != manifest["chunks"]: raise ValidationError("Manifest chunk provenance differs from independently verified receipts")
    finally:
        if ds is not None: ds.close()
    report = {"schema_version": "glorys_health_v1", "at_utc": now(), "plan_hash": plan["plan_hash"],
        "pass": bool(complete and checked == len(plan["chunks"])), "complete": bool(complete), "verified_chunks": checked,
        "source_clocks": plan["coordinates"]["time"], "selected_counts": {k: len(v) for k, v in plan["coordinates"].items()},
        "checks": ["hashes", "exact source clocks", "coordinates", "fields and attributes", "decoded values and missing patterns", "horizontal vector grids", "valid ocean coverage"],
        "plausibility_excursions": warnings}
    write_json(output or run_dir / "health_report.json", report)
    return report


def inspect(path):
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        result = {"file": str(Path(path).resolve()), "sha256": file_hash(path), "dimensions": dict(ds.sizes),
            "clocks": [iso(v) for v in ds.time.values] if "time" in ds else [],
            "coordinates": {n: {"count": a.size, "first": iso(a.values[0]) if n == "time" else json_value(a.values[0]), "last": iso(a.values[-1]) if n == "time" else json_value(a.values[-1]), "attributes": json_value(a.attrs)} for n, a in ds.coords.items()},
            "variables": {n: {"dimensions": list(a.dims), "dtype": str(a.dtype), "attributes": json_value(a.attrs)} for n, a in ds.data_vars.items()}}
    return result


def snapshot(request_path, run_dir, backend=None):
    """Acquire the first actual source record in a request into its own directory."""
    path = Path(request_path).resolve()
    request = read_json(path)
    backend = backend or OfficialBackend(service=request.get("service", "arco-geo-series"))
    plan, inv = build_plan(request, path.parent, run_dir, backend=backend)
    first = utc(plan["coordinates"]["time"][0])
    request["start"] = iso(first)
    request["end"] = iso(first + dt.timedelta(seconds=1))
    if "mesh" in request: request["mesh"]["path"] = str((path.parent / request["mesh"]["path"]).resolve())
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    frozen = run_dir / "snapshot_request.json"
    write_json(frozen, request)
    save_plan(frozen, run_dir, backend=backend)
    return fetch(run_dir, backend=backend)
