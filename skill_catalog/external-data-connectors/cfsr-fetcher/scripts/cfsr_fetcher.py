#!/usr/bin/env python3
"""NCEI-first, resumable NCEP CFSR surface-pressure fetcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import netCDF4 as nc4
import numpy as np
import requests

try:
    from .download_monitor import DownloadStatus, atomic_write_json, launch_monitor, write_monitor_html
except ImportError:
    from download_monitor import DownloadStatus, atomic_write_json, launch_monitor, write_monitor_html


NCEI_CATALOG = "https://www.ncei.noaa.gov/thredds/catalog/model-cfs_reanl_ts/{yyyymm}/catalog.xml"
NCEI_FILE = "https://www.ncei.noaa.gov/thredds/fileServer/model-cfs_reanl_ts/{yyyymm}/pressfc.gdas.{yyyymm}.grb2"
HYCOM_FILE = "https://tds.hycom.org/thredds/dodsC/datasets/force/ncep_cfsr/netcdf/cfsr-sec_{year}_01hr_sfcprs.nc"
MT_EPOCH = np.datetime64("1900-12-31T00:00:00", "s")


class CfsrFetcherError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    result = datetime.fromisoformat(text)
    if result.tzinfo is None:
        raise ValueError(f"Timestamp {value!r} must include UTC or an offset")
    return result.astimezone(timezone.utc).replace(microsecond=0)


def public_url(value: str) -> str:
    parsed = urlsplit(str(value))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Expected a public HTTP(S) source, got {value!r}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Source URLs cannot contain credentials, query strings, or fragments")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def month_keys(start: datetime, end: datetime) -> list[str]:
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    final = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    values: list[str] = []
    while cursor <= final:
        values.append(cursor.strftime("%Y%m"))
        cursor = datetime(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1, tzinfo=timezone.utc)
    return values


def normalize_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    start = parse_utc(str(payload.get("start") or payload.get("start_utc")))
    end = parse_utc(str(payload.get("end") or payload.get("end_utc")))
    if end < start:
        raise ValueError("Request end precedes start")
    product = str(payload.get("product", "surface_pressure"))
    if product != "surface_pressure":
        raise ValueError("The initial CFSR connector supports only surface_pressure")
    bbox = payload.get("bbox") or payload.get("bbox_0_360")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bbox must be [west,south,east,north]")
    west, south, east, north = [float(value) for value in bbox]
    if not (0 <= west < east <= 360 and -90 <= south < north <= 90):
        raise ValueError("bbox must be increasing and use 0-360 longitude")
    provider = str(payload.get("provider", "auto")).lower()
    if provider not in {"auto", "ncei", "hycom"}:
        raise ValueError("provider must be auto, ncei, or hycom")
    order = [str(value).lower() for value in payload.get("provider_order", ["ncei", "hycom"])]
    if any(value not in {"ncei", "hycom"} for value in order) or len(set(order)) != len(order):
        raise ValueError("provider_order must contain unique ncei/hycom values")
    if provider != "auto":
        order = [provider]
    output = str(payload.get("output") or "cfsr_surface_pressure.nc")
    return {
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "product": product,
        "bbox": [west, south, east, north],
        "halo_cells": max(0, int(payload.get("halo_cells", 1))),
        "provider": provider,
        "provider_order": order,
        "chunk_hours": max(1, int(payload.get("chunk_hours", 168))),
        "max_retries": max(1, int(payload.get("max_retries", 5))),
        "retry_delay_seconds": max(0.0, float(payload.get("retry_delay_seconds", 5.0))),
        "backoff": max(1.0, float(payload.get("backoff", 2.0))),
        "output": output,
        "ncei_catalog_template": public_url(str(payload.get("ncei_catalog_template", NCEI_CATALOG))),
        "ncei_file_template": public_url(str(payload.get("ncei_file_template", NCEI_FILE))),
        "hycom_source_template": public_url(str(payload.get("hycom_source_template", HYCOM_FILE))),
    }


def _request(session: requests.Session, method: str, url: str, *, retries: int = 3, **kwargs: Any) -> requests.Response:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.request(method, url, timeout=(30, 120), allow_redirects=True, **kwargs)
            if response.status_code >= 500 or response.status_code in {408, 429}:
                raise requests.HTTPError(f"HTTP {response.status_code} for {url}", response=response)
            response.raise_for_status()
            return response
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(min(30.0, 2.0 ** (attempt - 1)))
    raise CfsrFetcherError(str(last)) from last


def ncei_inventory(request: Mapping[str, Any], *, session: requests.Session | None = None) -> dict[str, Any]:
    own = session is None
    client = session or requests.Session()
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    try:
        for key in month_keys(parse_utc(str(request["start"])), parse_utc(str(request["end"]))):
            url = str(request["ncei_file_template"]).format(yyyymm=key, year=key[:4], month=key[4:])
            response = _request(client, "HEAD", url, retries=3)
            length = int(response.headers.get("Content-Length", "0"))
            if length <= 0:
                raise CfsrFetcherError(f"NCEI did not report a positive Content-Length for {key}")
            rows.append({
                "id": key,
                "url": public_url(url),
                "bytes": length,
                "last_modified": response.headers.get("Last-Modified"),
                "accept_ranges": response.headers.get("Accept-Ranges", "").lower() == "bytes",
            })
    finally:
        if own:
            client.close()
    if not all(row["accept_ranges"] for row in rows):
        raise CfsrFetcherError("One or more NCEI files do not advertise byte-range resume")
    return {
        "schema_version": "cfsr_inventory_v1",
        "provider": "ncei",
        "source_product": "NCEP CFS Reanalysis Time Series pressfc",
        "source_units": rows,
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _require_eccodes() -> Any:
    try:
        import eccodes
    except ImportError as exc:
        raise CfsrFetcherError("eccodes is required to decode native NCEI GRIB2") from exc
    return eccodes


def _decode_time(eccodes: Any, message: Any) -> datetime:
    date = int(eccodes.codes_get(message, "validityDate"))
    value = int(eccodes.codes_get(message, "validityTime"))
    return datetime.strptime(f"{date:08d}{value:04d}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def _regular_grid(eccodes: Any, message: Any, decoded_values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ni = int(eccodes.codes_get(message, "Ni"))
    nj = int(eccodes.codes_get(message, "Nj"))
    values = (
        np.asarray(eccodes.codes_get_array(message, "values"), dtype=np.float64)
        if decoded_values is None
        else np.asarray(decoded_values, dtype=np.float64).reshape(-1)
    )
    lats = np.asarray(eccodes.codes_get_array(message, "latitudes"), dtype=np.float64)
    lons = np.mod(np.asarray(eccodes.codes_get_array(message, "longitudes"), dtype=np.float64), 360.0)
    if not (values.size == lats.size == lons.size == ni * nj):
        raise CfsrFetcherError("GRIB grid dimensions do not match decoded arrays")
    candidates = [(nj, ni, False), (ni, nj, True)]
    for rows, cols, transpose in candidates:
        lat2 = lats.reshape(rows, cols)
        lon2 = lons.reshape(rows, cols)
        data = values.reshape(rows, cols)
        if transpose:
            lat2, lon2, data = lat2.T, lon2.T, data.T
        if np.allclose(lat2, lat2[:, :1], atol=1e-7) and np.allclose(lon2, lon2[:1, :], atol=1e-7):
            return lat2[:, 0], lon2[0, :], data
    raise CfsrFetcherError("NCEI GRIB grid is not a separable native latitude/longitude grid")


def _pressure_pa(eccodes: Any, message: Any, values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    units = str(eccodes.codes_get(message, "units")).strip()
    name = str(eccodes.codes_get(message, "name"))
    short_name = str(eccodes.codes_get(message, "shortName"))
    level = str(eccodes.codes_get(message, "typeOfLevel"))
    if "pressure" not in name.lower() or level.lower() not in {"surface", "sfc"}:
        raise CfsrFetcherError(f"Unexpected GRIB field {name!r} at {level!r}")
    key = units.lower().replace(" ", "")
    if key in {"pa", "pascal", "pascals"}:
        pressure = values
        conversion = "identity_Pa"
    elif key in {"hpa", "mb", "mbar", "hectopascal", "hectopascals"}:
        pressure = values * 100.0
        conversion = "hPa_times_100_to_Pa"
    else:
        raise CfsrFetcherError(f"Unsupported NCEI pressure units {units!r}")
    return pressure, {"source_name": name, "source_short_name": short_name, "source_units": units, "conversion": conversion, "type_of_level": level}


def _bbox_indices(axis: np.ndarray, low: float, high: float, halo: int) -> np.ndarray:
    found = np.flatnonzero(np.isfinite(axis) & (axis >= low) & (axis <= high))
    if found.size == 0:
        raise CfsrFetcherError("Requested bbox lies outside source coverage")
    return np.arange(max(0, int(found.min()) - halo), min(axis.size, int(found.max()) + halo + 1))


def ncei_smoke(request: Mapping[str, Any], inventory: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    client = requests.Session()
    unit = inventory["source_units"][0]
    url = str(unit["url"])
    started = time.monotonic()
    header = _request(client, "GET", url, headers={"Range": "bytes=0-15"}, retries=3).content
    if len(header) != 16 or header[:4] != b"GRIB" or int(header[7]) != 2:
        raise CfsrFetcherError("NCEI smoke response is not a GRIB2 message")
    message_bytes = int.from_bytes(header[8:16], "big")
    if message_bytes <= 16 or message_bytes > int(unit["bytes"]):
        raise CfsrFetcherError("Invalid GRIB2 message length in NCEI smoke response")
    response = _request(client, "GET", url, headers={"Range": f"bytes=0-{message_bytes - 1}"}, retries=3)
    payload = response.content
    smoke_dir = run_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    path = smoke_dir / f"{unit['id']}_first_message.grb2"
    temporary = path.with_suffix(".part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    eccodes = _require_eccodes()
    try:
        import rasterio
    except ImportError as exc:
        raise CfsrFetcherError("rasterio/GDAL with GRIB JPEG2000 support is required") from exc
    with rasterio.open(path) as raster:
        raster_values = raster.read(1)
    with path.open("rb") as stream:
        message = eccodes.codes_grib_new_from_file(stream)
        if message is None:
            raise CfsrFetcherError("ecCodes could not decode the NCEI smoke message")
        try:
            lat, lon, values = _regular_grid(eccodes, message, raster_values)
            pressure, metadata = _pressure_pa(eccodes, message, values)
            timestamp = _decode_time(eccodes, message)
        finally:
            eccodes.codes_release(message)
    west, south, east, north = request["bbox"]
    yi = _bbox_indices(lat, south, north, int(request["halo_cells"]))
    xi = _bbox_indices(lon, west, east, int(request["halo_cells"]))
    subset = pressure[np.ix_(yi, xi)]
    if not np.all(np.isfinite(subset)) or float(subset.min()) < 50_000 or float(subset.max()) > 120_000:
        raise CfsrFetcherError("NCEI smoke pressure is non-finite or implausible")
    elapsed = max(time.monotonic() - started, 1e-6)
    return {
        "schema_version": "cfsr_ncei_smoke_v1",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "message_bytes": message_bytes,
        "elapsed_seconds": round(elapsed, 6),
        "bytes_per_second": round(message_bytes / elapsed, 3),
        "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "global_shape": [int(lat.size), int(lon.size)],
        "subset_shape": [int(yi.size), int(xi.size)],
        "latitude_range": [float(lat[yi].min()), float(lat[yi].max())],
        "longitude_range": [float(lon[xi].min()), float(lon[xi].max())],
        "pressure_minimum_pa": float(subset.min()),
        "pressure_maximum_pa": float(subset.max()),
        **metadata,
    }


def _plan_gate(transfer_bytes: int, free_bytes: int, conservative_seconds: float) -> dict[str, Any]:
    passed = free_bytes > 4 * transfer_bytes
    if not passed:
        state, reason = "blocked", "Local free space is not greater than four times the raw transfer size."
    elif not math.isfinite(conservative_seconds) or conservative_seconds <= 0:
        state, reason = "blocked", "A positive duration estimate was not established."
    elif conservative_seconds >= 600:
        state, reason = "long_run_monitor_required", "Conservative duration is at least ten minutes."
    else:
        state, reason = "ready", "Time and storage gates passed."
    return {"state": state, "reason": reason, "storage_passed": passed, "storage_rule": "free_bytes > 4 * raw_transfer_bytes", "monitor_threshold_seconds": 600}


def build_ncei_plan(request: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    inventory = ncei_inventory(request)
    smoke = ncei_smoke(request, inventory, run_dir)
    probe_bytes = min(16 * 1024 * 1024, int(inventory["source_units"][0]["bytes"]))
    client = requests.Session()
    probe_started = time.monotonic()
    probe_response = _request(
        client,
        "GET",
        str(inventory["source_units"][0]["url"]),
        headers={"Range": f"bytes=0-{probe_bytes - 1}"},
        retries=3,
    )
    received = len(probe_response.content)
    probe_elapsed = max(time.monotonic() - probe_started, 1e-6)
    if received != probe_bytes:
        raise CfsrFetcherError(f"NCEI transfer probe returned {received} bytes; expected {probe_bytes}")
    transfer_probe = {
        "method":"HTTP Range bounded transfer",
        "bytes":received,
        "elapsed_seconds":round(probe_elapsed,6),
        "bytes_per_second":round(received / probe_elapsed,3),
    }
    rate = max(1.0, float(transfer_probe["bytes_per_second"]))
    transfer = int(inventory["total_bytes"])
    central = transfer / rate + 60 * len(inventory["source_units"])
    conservative = central * 1.5 + 300 * len(inventory["source_units"])
    free = int(shutil.disk_usage(run_dir).free)
    expected_hours = int((parse_utc(str(request["end"])) - parse_utc(str(request["start"]))).total_seconds() // 3600) + 1
    return {
        "provider": "ncei",
        "inventory": inventory,
        "smoke": smoke,
        "transfer_probe": transfer_probe,
        "source_units": inventory["source_units"],
        "raw_transfer_bytes": transfer,
        "free_bytes": free,
        "expected_hours": expected_hours,
        "duration_estimate_seconds": {"central": round(central, 3), "conservative": round(conservative, 3)},
        "gate": _plan_gate(transfer, free, conservative),
    }


def _mt_to_epoch_seconds(values: np.ndarray) -> np.ndarray:
    return (MT_EPOCH + np.rint(np.asarray(values, dtype=np.float64) * 86400.0).astype("timedelta64[s]")).astype("datetime64[s]").astype(np.int64)


def build_hycom_plan(request: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    try:
        import xarray as xr
    except ImportError as exc:
        raise CfsrFetcherError("xarray is required for HYCOM fallback") from exc
    start, end = parse_utc(str(request["start"])), parse_utc(str(request["end"]))
    source_units: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    transfer = 0
    started_probe = time.monotonic()
    probe_bytes = 0
    for year in range(start.year, end.year + 1):
        url = str(request["hycom_source_template"]).format(year=year)
        with xr.open_dataset(url, engine="netcdf4", decode_times=False, cache=False) as dataset:
            for name in ("MT", "Latitude", "Longitude", "airprs"):
                if name not in dataset:
                    raise CfsrFetcherError(f"HYCOM CFSR source is missing {name}")
            epoch = _mt_to_epoch_seconds(dataset["MT"].values)
            start_s, end_s = int(start.timestamp()), int(end.timestamp())
            ti = np.flatnonzero((epoch >= start_s) & (epoch <= end_s))
            if ti.size == 0:
                continue
            lat = np.asarray(dataset["Latitude"].values, float)
            lon = np.asarray(dataset["Longitude"].values, float)
            west, south, east, north = request["bbox"]
            yi = _bbox_indices(lat, south, north, int(request["halo_cells"]))
            xi = _bbox_indices(lon, west, east, int(request["halo_cells"]))
            source_units.append({"id": str(year), "url": public_url(url), "records": int(ti.size)})
            for offset in range(0, ti.size, int(request["chunk_hours"])):
                part = ti[offset:offset + int(request["chunk_hours"])]
                count = int(part.size * yi.size * xi.size * np.dtype(dataset["airprs"].dtype).itemsize)
                chunks.append({"id": f"{year}-{offset:05d}", "url": public_url(url), "time": [int(part[0]), int(part[-1]) + 1], "latitude": [int(yi[0]), int(yi[-1]) + 1], "longitude": [int(xi[0]), int(xi[-1]) + 1], "bytes": count})
                transfer += count
            if probe_bytes == 0:
                sample = dataset["airprs"].isel(MT=slice(int(ti[0]), int(ti[0]) + 1), Latitude=slice(int(yi[0]), min(int(yi[0]) + 16, int(yi[-1]) + 1)), Longitude=slice(int(xi[0]), min(int(xi[0]) + 16, int(xi[-1]) + 1))).load()
                probe_bytes = int(sample.nbytes)
    if not chunks:
        raise CfsrFetcherError("HYCOM CFSR has no records in the requested interval")
    probe_elapsed = max(time.monotonic() - started_probe, 1e-6)
    rate = max(1.0, probe_bytes / probe_elapsed)
    central = transfer / rate + 5 * len(chunks)
    conservative = central * 1.5 + 5 * len(chunks)
    free = int(shutil.disk_usage(run_dir).free)
    expected_hours = int((end - start).total_seconds() // 3600) + 1
    return {"provider":"hycom","source_units":source_units,"chunks":chunks,"raw_transfer_bytes":transfer,"free_bytes":free,"expected_hours":expected_hours,"duration_estimate_seconds":{"central":round(central,3),"conservative":round(conservative,3)},"gate":_plan_gate(transfer,free,conservative)}


def build_plan(payload: Mapping[str, Any], run_dir: str | Path) -> dict[str, Any]:
    request = normalize_request(payload)
    directory = Path(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []
    provider_plan: dict[str, Any] | None = None
    for provider in request["provider_order"]:
        try:
            provider_plan = build_ncei_plan(request, directory) if provider == "ncei" else build_hycom_plan(request, directory)
            break
        except Exception as exc:
            errors.append({"provider": provider, "error": f"{type(exc).__name__}: {exc}"})
    if provider_plan is None:
        raise CfsrFetcherError("No provider passed inventory/smoke: " + " | ".join(f"{row['provider']}: {row['error']}" for row in errors))
    plan = {
        "schema_version": "cfsr_download_plan_v1",
        "created_utc": utc_now(),
        "expires_utc": (datetime.now(timezone.utc) + timedelta(hours=24)).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "request": request,
        "request_hash": hash_payload(request),
        "provider_attempt_errors": errors,
        **provider_plan,
    }
    content = dict(plan)
    plan["plan_hash"] = hash_payload(content)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != "cfsr_download_plan_v1":
        raise ValueError("Unsupported CFSR plan schema")
    supplied = str(plan.get("plan_hash", ""))
    content = dict(plan)
    content.pop("plan_hash", None)
    if supplied != hash_payload(content):
        raise ValueError("CFSR plan hash mismatch")
    if str(plan.get("request_hash")) != hash_payload(plan.get("request")):
        raise ValueError("CFSR request hash mismatch")
    if datetime.now(timezone.utc) > parse_utc(str(plan["expires_utc"])):
        raise ValueError("CFSR plan expired; rerun estimate")
    if plan.get("gate", {}).get("state") == "blocked":
        raise ValueError(f"CFSR plan blocked: {plan['gate'].get('reason')}")


def _atomic_canonical(path: Path, times: np.ndarray, lat: np.ndarray, lon: np.ndarray, pressure: np.ndarray, attrs: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with nc4.Dataset(temporary, "w", format="NETCDF4_CLASSIC") as output:
            output.createDimension("time", int(times.size))
            output.createDimension("latitude", int(lat.size))
            output.createDimension("longitude", int(lon.size))
            tv = output.createVariable("time", "f8", ("time",))
            tv.units = "seconds since 1970-01-01 00:00:00 UTC"
            tv.calendar = "standard"
            tv[:] = times.astype(np.float64)
            yv = output.createVariable("latitude", "f8", ("latitude",))
            xv = output.createVariable("longitude", "f8", ("longitude",))
            yv.units = "degrees_north"
            xv.units = "degrees_east"
            yv[:] = lat
            xv[:] = lon
            pv = output.createVariable("absolute_air_pressure", "f4", ("time", "latitude", "longitude"), zlib=True, complevel=3, shuffle=True, chunksizes=(min(168, int(times.size)), int(lat.size), int(lon.size)))
            pv.units = "Pa"
            pv.pressure_reference = "absolute"
            pv.coordinates = "latitude longitude"
            pv[:] = np.asarray(pressure, dtype=np.float32)
            for key, value in attrs.items():
                output.setncattr(str(key), value if isinstance(value, (str, int, float)) else json.dumps(value, sort_keys=True))
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _valid_checkpoint(path: Path, request_hash: str, source_sha: str) -> bool:
    if not path.is_file():
        return False
    try:
        with nc4.Dataset(path) as dataset:
            return str(getattr(dataset, "request_hash", "")) == request_hash and str(getattr(dataset, "source_sha256", "")) == source_sha and "absolute_air_pressure" in dataset.variables
    except Exception:
        return False


def _download_ncei_unit(unit: Mapping[str, Any], raw_dir: Path, status: DownloadStatus, base_completed: int, request: Mapping[str, Any]) -> Path:
    destination = raw_dir / f"pressfc.gdas.{unit['id']}.grb2"
    expected = int(unit["bytes"])
    if destination.is_file() and destination.stat().st_size == expected:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists() and partial.stat().st_size > expected:
        raise CfsrFetcherError(f"Partial file exceeds source size: {partial}")
    client = requests.Session()
    last_error: Exception | None = None
    for attempt in range(1, int(request["max_retries"]) + 1):
        try:
            start = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={start}-"}
            with client.get(str(unit["url"]), headers=headers, stream=True, timeout=(30, 180), allow_redirects=True) as response:
                if start and response.status_code == 200:
                    partial.unlink(missing_ok=True)
                    start = 0
                    raise CfsrFetcherError("Server ignored Range resume; safely restarting this source unit")
                if response.status_code not in {200, 206}:
                    response.raise_for_status()
                mode = "ab" if start else "wb"
                with partial.open(mode) as stream:
                    last_update = time.monotonic()
                    for block in response.iter_content(1024 * 1024):
                        if not block:
                            continue
                        stream.write(block)
                        if time.monotonic() - last_update >= 1:
                            current = partial.stat().st_size
                            status.update(active_source=str(unit["id"]), completed_bytes=base_completed + current, attempts=int(status.data.get("attempts", 0)) + 1)
                            last_update = time.monotonic()
            if partial.stat().st_size != expected:
                raise CfsrFetcherError(f"Size mismatch for {unit['id']}: {partial.stat().st_size} != {expected}")
            os.replace(partial, destination)
            return destination
        except Exception as exc:
            last_error = exc
            if attempt < int(request["max_retries"]):
                status.update(message=f"Retrying {unit['id']}: {type(exc).__name__}: {exc}", retries=int(status.data.get("retries", 0)) + 1)
                time.sleep(float(request["retry_delay_seconds"]) * float(request["backoff"]) ** (attempt - 1))
    raise CfsrFetcherError(f"NCEI unit {unit['id']} failed: {last_error}") from last_error


def _decode_ncei_month(raw: Path, checkpoint: Path, request: Mapping[str, Any], request_hash: str, status: DownloadStatus) -> tuple[Path, int]:
    source_sha = sha256_file(raw)
    if _valid_checkpoint(checkpoint, request_hash, source_sha):
        with nc4.Dataset(checkpoint) as dataset:
            return checkpoint, int(dataset.dimensions["time"].size)
    eccodes = _require_eccodes()
    start, end = parse_utc(str(request["start"])), parse_utc(str(request["end"]))
    match = re.search(r"(\d{6})\.grb2$", raw.name)
    if not match:
        raise CfsrFetcherError(f"Cannot infer source month from {raw.name}")
    source_month = match.group(1)
    west, south, east, north = request["bbox"]
    records: list[dict[str, Any]] = []
    signature_counts: dict[tuple[Any, ...], int] = {}
    with raw.open("rb") as stream:
        band = 0
        while True:
            message = eccodes.codes_grib_new_from_file(stream)
            if message is None:
                break
            band += 1
            try:
                timestamp = _decode_time(eccodes, message)
                signature = (
                    int(eccodes.codes_get(message, "Ni")),
                    int(eccodes.codes_get(message, "Nj")),
                    round(float(eccodes.codes_get(message, "longitudeOfLastGridPointInDegrees")), 6),
                    round(float(eccodes.codes_get(message, "latitudeOfFirstGridPointInDegrees")), 6),
                    round(float(eccodes.codes_get(message, "latitudeOfLastGridPointInDegrees")), 6),
                )
                # Count signatures across the complete monthly file.  The
                # inclusive terminal month can contribute only one requested
                # timestamp, where the two step-zero encodings would otherwise
                # tie and make the selected native grid depend on message
                # order.  The dominant full-file signature is the consistent
                # hourly forecast grid used by all preceding checkpoints.
                signature_counts[signature] = signature_counts.get(signature, 0) + 1
                if timestamp < start or timestamp > end or timestamp.strftime("%Y%m") != source_month:
                    continue
                step = int(eccodes.codes_get(message, "step"))
                row = {"band": band, "timestamp": int(timestamp.timestamp()), "step": step, "signature": signature}
                records.append(row)
            finally:
                eccodes.codes_release(message)
    if not records:
        raise CfsrFetcherError(f"No requested records found in {raw.name}")
    canonical_signature = max(signature_counts, key=signature_counts.get)
    canonical_records = [row for row in records if row["signature"] == canonical_signature]
    chosen: dict[int, dict[str, Any]] = {}
    for row in canonical_records:
        existing = chosen.get(int(row["timestamp"]))
        if existing is None or int(row["step"]) < int(existing["step"]):
            chosen[int(row["timestamp"])] = row
    selected_rows = sorted(chosen.values(), key=lambda row: int(row["timestamp"]))
    selected_bands = {int(row["band"]): row for row in selected_rows}
    times: list[int] = []
    fields: list[np.ndarray] = []
    selected_lat: np.ndarray | None = None
    selected_lon: np.ndarray | None = None
    metadata: dict[str, Any] | None = None
    try:
        import rasterio
    except ImportError as exc:
        raise CfsrFetcherError("rasterio/GDAL with GRIB JPEG2000 support is required") from exc
    with rasterio.open(raw) as raster, raw.open("rb") as stream:
        band = 0
        while True:
            message = eccodes.codes_grib_new_from_file(stream)
            if message is None:
                break
            band += 1
            try:
                if band not in selected_bands:
                    continue
                raster_values = raster.read(band)
                lat, lon, values = _regular_grid(eccodes, message, raster_values)
                pressure, current = _pressure_pa(eccodes, message, values)
                yi = _bbox_indices(lat, south, north, int(request["halo_cells"]))
                xi = _bbox_indices(lon, west, east, int(request["halo_cells"]))
                if selected_lat is None:
                    selected_lat, selected_lon, metadata = lat[yi], lon[xi], current
                    metadata.update({
                        "record_selection":"dominant forecast-grid signature; shortest forecast lead for duplicate valid times",
                        "canonical_grid_signature":canonical_signature,
                    })
                elif not np.allclose(selected_lat, lat[yi], rtol=0, atol=1e-10) or not np.allclose(selected_lon, lon[xi], rtol=0, atol=1e-10):
                    raise CfsrFetcherError("Canonical NCEI grid changed within a monthly file")
                fields.append(pressure[np.ix_(yi, xi)].astype(np.float32))
                times.append(int(selected_bands[band]["timestamp"]))
                if len(fields) % 24 == 0:
                    status.update(decoded_hours=int(status.data.get("decoded_hours", 0)) + 24)
            finally:
                eccodes.codes_release(message)
        if band != raster.count:
            raise CfsrFetcherError(f"ecCodes/GDAL message-count mismatch for {raw.name}: {band} != {raster.count}")
    if not fields or selected_lat is None or selected_lon is None or metadata is None:
        raise CfsrFetcherError(f"No requested records decoded from {raw.name}")
    order = np.argsort(np.asarray(times, dtype=np.int64))
    times = [times[int(index)] for index in order]
    data = np.stack([fields[int(index)] for index in order])
    metadata["source_message_count"] = int(band)
    metadata["selected_record_count"] = int(len(times))
    metadata["discarded_duplicate_or_alternate_grid_records"] = int(len(records) - len(times))
    _atomic_canonical(checkpoint, np.asarray(times, dtype=np.int64), selected_lat, selected_lon, data, {
        "connector": "cfsr-fetcher", "source_provider": "ncei", "source_product": "NCEP CFS Reanalysis Time Series pressfc", "source_file": raw.name, "source_sha256": source_sha, "request_hash": request_hash, **metadata,
    })
    return checkpoint, len(times)


def _combine_checkpoints(checkpoints: Sequence[Path], destination: Path, request: Mapping[str, Any], request_hash: str, provider: str) -> None:
    times: list[np.ndarray] = []
    fields: list[np.ndarray] = []
    lat: np.ndarray | None = None
    lon: np.ndarray | None = None
    sources: list[str] = []
    for path in checkpoints:
        with nc4.Dataset(path) as dataset:
            current_lat = np.asarray(dataset.variables["latitude"][:], dtype=np.float64)
            current_lon = np.asarray(dataset.variables["longitude"][:], dtype=np.float64)
            if lat is None:
                lat, lon = current_lat, current_lon
            elif not np.array_equal(lat, current_lat) or not np.array_equal(lon, current_lon):
                raise CfsrFetcherError("Provider checkpoints do not share an exact native grid")
            times.append(np.asarray(dataset.variables["time"][:], dtype=np.int64))
            fields.append(np.asarray(dataset.variables["absolute_air_pressure"][:], dtype=np.float32))
            sources.append(str(getattr(dataset, "source_file", path.name)))
    combined_times = np.concatenate(times)
    combined_fields = np.concatenate(fields, axis=0)
    order = np.argsort(combined_times)
    combined_times, combined_fields = combined_times[order], combined_fields[order]
    if np.unique(combined_times).size != combined_times.size:
        raise CfsrFetcherError("Provider checkpoints contain duplicate timestamps")
    expected = np.arange(int(parse_utc(str(request["start"])).timestamp()), int(parse_utc(str(request["end"])).timestamp()) + 1, 3600, dtype=np.int64)
    if not np.array_equal(combined_times, expected):
        missing = np.setdiff1d(expected, combined_times)
        raise CfsrFetcherError(f"Provider output does not match exact hourly axis; missing {missing[:5].tolist()}")
    assert lat is not None and lon is not None
    _atomic_canonical(destination, combined_times, lat, lon, combined_fields, {
        "connector":"cfsr-fetcher", "source_provider":provider, "source_product":"NCEP Climate Forecast System Reanalysis", "request_hash":request_hash, "requested_start_utc":request["start"], "requested_end_utc":request["end"], "requested_bbox_0_360":request["bbox"], "halo_cells":int(request["halo_cells"]), "source_files":sources, "pressure_reference":"absolute",
    })


def fetch_ncei(plan: Mapping[str, Any], run_dir: Path, destination: Path, status: DownloadStatus, cleanup_raw: bool) -> None:
    raw_dir = run_dir / "ncei" / "raw"
    checkpoint_dir = run_dir / "ncei" / "checkpoints"
    raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed_bytes = 0
    checkpoints: list[Path] = []
    decoded = 0
    for index, unit in enumerate(plan["source_units"], 1):
        status.update(active_source=str(unit["id"]), completed_bytes=completed_bytes, message=f"Acquiring NCEI {unit['id']}")
        raw = _download_ncei_unit(unit, raw_dir, status, completed_bytes, plan["request"])
        completed_bytes += int(unit["bytes"])
        source_sha = sha256_file(raw)
        status.update(completed_bytes=completed_bytes, completed_chunks=index, last_success_utc=utc_now(), message=f"Verified NCEI {unit['id']} ({source_sha[:12]})")
        checkpoint = checkpoint_dir / f"{plan['request_hash'][:12]}-{unit['id']}.nc"
        checkpoint, count = _decode_ncei_month(raw, checkpoint, plan["request"], str(plan["request_hash"]), status)
        checkpoints.append(checkpoint)
        decoded += count
        status.update(decoded_hours=decoded, message=f"Decoded {unit['id']}: {count} requested hours")
    _combine_checkpoints(checkpoints, destination, plan["request"], str(plan["request_hash"]), "ncei")
    if cleanup_raw:
        for raw in raw_dir.glob("*.grb2"):
            raw.unlink()


def fetch_hycom(plan: Mapping[str, Any], run_dir: Path, destination: Path, status: DownloadStatus) -> None:
    try:
        import xarray as xr
    except ImportError as exc:
        raise CfsrFetcherError("xarray is required for HYCOM fallback") from exc
    checkpoint_dir = run_dir / "hycom" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[Path] = []
    completed = 0
    decoded = 0
    for index, chunk in enumerate(plan["chunks"], 1):
        checkpoint = checkpoint_dir / f"{plan['request_hash'][:12]}-{chunk['id']}.nc"
        if not checkpoint.is_file():
            last: Exception | None = None
            for attempt in range(1, int(plan["request"]["max_retries"]) + 1):
                try:
                    status.update(active_source=chunk["id"], message=f"Downloading HYCOM {chunk['id']} attempt {attempt}")
                    with xr.open_dataset(chunk["url"], engine="netcdf4", decode_times=False, cache=False) as dataset:
                        piece = dataset[["airprs"]].isel(MT=slice(*chunk["time"]), Latitude=slice(*chunk["latitude"]), Longitude=slice(*chunk["longitude"])).load()
                        epoch = _mt_to_epoch_seconds(piece["MT"].values)
                        lat = np.asarray(piece["Latitude"].values, float)
                        lon = np.asarray(piece["Longitude"].values, float)
                        raw = np.asarray(piece["airprs"].values, dtype=np.float64)
                        units = str(piece["airprs"].attrs.get("units", "hPa")).lower()
                        departure = raw if units in {"hpa", "mb", "mbar"} else raw / 100.0
                        pressure = (departure + 1000.0) * 100.0
                    _atomic_canonical(checkpoint, epoch, lat, lon, pressure, {"connector":"cfsr-fetcher","source_provider":"hycom","source_file":chunk["url"],"source_variable":"airprs","source_units":units,"conversion":"(airprs + 1000 hPa) * 100 Pa/hPa","request_hash":plan["request_hash"]})
                    last = None
                    break
                except Exception as exc:
                    last = exc
                    if attempt < int(plan["request"]["max_retries"]):
                        status.update(message=f"Retrying HYCOM {chunk['id']}: {exc}", retries=int(status.data.get("retries", 0)) + 1)
                        time.sleep(float(plan["request"]["retry_delay_seconds"]) * float(plan["request"]["backoff"]) ** (attempt - 1))
            if last is not None:
                raise CfsrFetcherError(f"HYCOM chunk {chunk['id']} failed: {last}") from last
        checkpoints.append(checkpoint)
        completed += int(chunk["bytes"])
        with nc4.Dataset(checkpoint) as dataset:
            decoded += int(dataset.dimensions["time"].size)
        status.update(completed_chunks=index, completed_bytes=completed, decoded_hours=decoded, last_success_utc=utc_now())
    _combine_checkpoints(checkpoints, destination, plan["request"], str(plan["request_hash"]), "hycom")


def health(path: str | Path, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source = Path(path).resolve()
    checks: list[dict[str, Any]] = []
    with nc4.Dataset(source) as dataset:
        required = ["time", "latitude", "longitude", "absolute_air_pressure"]
        for name in required:
            checks.append({"check": f"variable:{name}", "passed": name in dataset.variables})
        times = np.asarray(dataset.variables["time"][:], dtype=np.int64)
        lat = np.asarray(dataset.variables["latitude"][:], dtype=float)
        lon = np.asarray(dataset.variables["longitude"][:], dtype=float)
        pressure = np.asarray(dataset.variables["absolute_air_pressure"][:], dtype=np.float32)
        checks.append({"check":"time_strict_hourly","passed":times.size < 2 or bool(np.all(np.diff(times) == 3600))})
        checks.append({"check":"finite_pressure","passed":bool(np.all(np.isfinite(pressure)))})
        pmin, pmax = float(np.nanmin(pressure)), float(np.nanmax(pressure))
        checks.append({"check":"plausible_absolute_pressure","passed":pmin >= 50_000 and pmax <= 120_000,"minimum_pa":pmin,"maximum_pa":pmax})
        frozen = int(np.count_nonzero(np.nanmax(pressure, axis=0) == np.nanmin(pressure, axis=0)))
        checks.append({"check":"zero_temporally_frozen_cells","passed":frozen == 0,"count":frozen})
        checks.append({"check":"nonempty_native_grid","passed":lat.size > 1 and lon.size > 1})
        if request:
            normalized = normalize_request(request)
            expected = np.arange(int(parse_utc(normalized["start"]).timestamp()), int(parse_utc(normalized["end"]).timestamp()) + 1, 3600, dtype=np.int64)
            checks.append({"check":"exact_requested_time_axis","passed":bool(np.array_equal(times, expected)),"expected_records":int(expected.size),"actual_records":int(times.size)})
            west, south, east, north = normalized["bbox"]
            checks.append({"check":"bbox_coverage","passed":float(lon.min()) <= west and float(lon.max()) >= east and float(lat.min()) <= south and float(lat.max()) >= north})
        payload = {"schema_version":"cfsr_health_v1","path":str(source),"bytes":source.stat().st_size,"sha256":sha256_file(source),"source_provider":str(getattr(dataset,"source_provider","unknown")),"dimensions":{"time":int(times.size),"latitude":int(lat.size),"longitude":int(lon.size)},"checks":checks}
    payload["passed"] = all(bool(row["passed"]) for row in checks)
    if not payload["passed"]:
        raise CfsrFetcherError("CFSR output failed health validation")
    return payload


def fetch_plan(plan: Mapping[str, Any], run_dir: str | Path, *, output: str | Path | None = None, open_monitor: bool = True, cleanup_raw: bool = False) -> dict[str, Any]:
    validate_plan(plan)
    directory = Path(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = Path(output).resolve() if output else (directory / str(plan["request"]["output"])).resolve()
    provider = str(plan["provider"])
    atomic_write_json(directory / "source_provider_lock.json", {"schema_version":"cfsr_provider_lock_v1","provider":provider,"request_hash":plan["request_hash"],"locked_utc":utc_now(),"mixed_providers":False})
    write_monitor_html(directory)
    status = DownloadStatus(directory / "download_status.json", connector="cfsr-fetcher", provider=provider, request_hash=plan["request_hash"], total_chunks=len(plan.get("source_units", plan.get("chunks", []))), completed_chunks=0, expected_bytes=int(plan["raw_transfer_bytes"]), completed_bytes=0, expected_hours=int(plan["expected_hours"]), decoded_hours=0, retries=0, attempts=0, active_source=None, artifacts={"monitor":"download_monitor.html","health":"health_check.json","output":destination.name})
    monitor = None
    if float(plan["duration_estimate_seconds"]["conservative"]) >= 600:
        monitor = launch_monitor(directory, open_browser=open_monitor) if open_monitor else {"launched":False,"html":str(directory / "download_monitor.html")}
    status.start(f"Starting whole-run provider {provider}")
    try:
        if provider == "ncei":
            fetch_ncei(plan, directory, destination, status, cleanup_raw)
        elif provider == "hycom":
            fetch_hycom(plan, directory, destination, status)
        else:
            raise CfsrFetcherError(f"Unsupported locked provider {provider}")
        report = health(destination, plan["request"])
        atomic_write_json(directory / "health_check.json", report)
        status.update(completed_bytes=int(plan["raw_transfer_bytes"]), completed_chunks=len(plan.get("source_units", plan.get("chunks", []))), decoded_hours=int(plan["expected_hours"]), message="Health checks passed")
        status.finish("complete", "CFSR acquisition and health validation completed")
        return {"schema_version":"cfsr_fetch_result_v1","provider":provider,"output":str(destination),"sha256":report["sha256"],"health":str(directory / "health_check.json"),"status":str(directory / "download_status.json"),"monitor":monitor}
    except BaseException as exc:
        status.finish("cancelled" if isinstance(exc, KeyboardInterrupt) else "failed", f"{type(exc).__name__}: {exc}")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--request", type=Path, required=True)
    inventory.add_argument("--output", type=Path)
    estimate = sub.add_parser("estimate")
    estimate.add_argument("--request", type=Path, required=True)
    estimate.add_argument("--run-dir", type=Path, required=True)
    estimate.add_argument("--output", type=Path)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--plan", type=Path, required=True)
    fetch.add_argument("--run-dir", type=Path, required=True)
    fetch.add_argument("--output", type=Path)
    fetch.add_argument("--resume", action="store_true")
    fetch.add_argument("--cleanup-raw", action="store_true")
    fetch.add_argument("--no-open-monitor", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    check = sub.add_parser("health")
    check.add_argument("--input", type=Path, required=True)
    check.add_argument("--request", type=Path)
    check.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "inventory":
        request = normalize_request(json.loads(args.request.read_text(encoding="utf-8-sig")))
        attempts: list[dict[str, Any]] = []
        result = None
        for provider in request["provider_order"]:
            try:
                result = ncei_inventory(request) if provider == "ncei" else {"schema_version":"cfsr_inventory_v1","provider":"hycom","note":"HYCOM inventory is established during estimate"}
                break
            except Exception as exc:
                attempts.append({"provider":provider,"error":f"{type(exc).__name__}: {exc}"})
        if result is None:
            raise CfsrFetcherError("All inventory providers failed")
        result["provider_attempt_errors"] = attempts
        if args.output:
            atomic_write_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "estimate":
        request_payload = json.loads(args.request.read_text(encoding="utf-8-sig"))
        plan = build_plan(request_payload, args.run_dir)
        output = args.output or args.run_dir / "download_plan.json"
        atomic_write_json(output, plan)
        atomic_write_json(args.run_dir / "request.json", normalize_request(request_payload))
        print(json.dumps(plan, indent=2))
    elif args.command == "fetch":
        plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
        print(json.dumps(fetch_plan(plan, args.run_dir, output=args.output, open_monitor=not args.no_open_monitor, cleanup_raw=args.cleanup_raw), indent=2))
    elif args.command == "status":
        path = args.run_dir / "download_status.json"
        print(path.read_text(encoding="utf-8-sig") if path.exists() else json.dumps({"state":"not_started"}, indent=2))
    elif args.command == "health":
        request = json.loads(args.request.read_text(encoding="utf-8-sig")) if args.request else None
        report = health(args.input, request)
        if args.output:
            atomic_write_json(args.output, report)
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
