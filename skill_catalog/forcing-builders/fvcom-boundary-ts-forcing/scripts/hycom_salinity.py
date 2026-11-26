"""Read the ESPC-D-V02 three-hourly, salinity-only archive from hycom-fetcher."""

import hashlib
import json
import re
from pathlib import Path

import netCDF4
import numpy as np

from ts_core import decode_netcdf_times, sha256_file


def content_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def load_hycom(directory: Path):
    plan_path = directory / "download_plan.json"
    qa_path = directory / "health_check.json"
    plan = json.loads(plan_path.read_bytes())
    qa = json.loads(qa_path.read_bytes())
    if (
        plan.get("schema_version") != "hycom_download_plan_v1"
        or qa.get("schema_version") != "hycom_health_v1"
    ):
        raise ValueError("A checked hycom-fetcher plan and health report are required")
    stored_hash = plan.pop("plan_hash")
    if content_hash(plan) != stored_hash or content_hash(plan["request"]) != plan["request_hash"]:
        raise ValueError("HYCOM plan or request identity changed")
    source_url = plan["request"]["source"]
    if not re.fullmatch(r"https://tds\.hycom\.org/thredds/dodsC/ESPC-D-V02/s3z/[0-9]{4}", source_url):
        raise ValueError("Require the public annual ESPC-D-V02/s3z salinity archive")
    name = qa["input_name"]
    if Path(name).name != name:
        raise ValueError("Expected a local subset basename")
    source = directory / name
    if (
        not qa.get("passed")
        or sha256_file(source) != qa["sha256"]
        or "salinity" not in plan["request"]["variables"]
    ):
        raise ValueError("Incomplete or changed HYCOM salinity product")
    with netCDF4.Dataset(source) as ds:
        salt = ds["salinity"]
        if (
            getattr(ds, "connector", None) != "hycom-fetcher"
            or getattr(ds, "request_hash", None) != plan["request_hash"]
            or str(getattr(ds, "field_type", "")).lower() != "instantaneous"
            or salt.dimensions != ("time", "depth", "lat", "lon")
            or salt.units.lower() != "psu"
            or ds["depth"].positive.lower() != "down"
            or ds["depth"].units != "m"
        ):
            raise ValueError("HYCOM field, dimensions, units or source record identity differs")
        st, _, _ = decode_netcdf_times(ds)
        lon, lat, depth = (np.asarray(ds[k][:], dtype=float) for k in ("lon", "lat", "depth"))
        # Keep the subset contiguous in signed longitude using the same permutation for data.
        lon = (lon + 180.0) % 360.0 - 180.0
        ix = np.argsort(lon)
        lon = lon[ix]
        salinity = np.ma.filled(np.ma.asarray(salt[:], dtype=float), np.nan)[..., ix]
        if len(lon) < 2 or np.max(np.diff(lon)) > 1.0 or not np.all(np.diff(st) == 10_800_000):
            raise ValueError("Require a contiguous regional subset on the native three-hour UTC axis")
    provenance = dict(
        product="HYCOM ESPC-D-V02",
        source_url=source_url,
        field="salinity",
        plan_hash=stored_hash,
        request_hash=plan["request_hash"],
        source_inventory_hash=plan["source_inventory_hash"],
    )
    files = dict(source=source, hycom_plan=plan_path, hycom_health=qa_path)
    return lon, lat, depth, st, salinity, provenance, files
