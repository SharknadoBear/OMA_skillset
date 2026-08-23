#!/usr/bin/env python3
"""Inventory, estimate, fetch, resume, and health-check bounded CFSv2 fields."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import xarray as xr

try:
    from .download_monitor import DownloadStatus, atomic_write_json, launch_monitor, safe_message, write_monitor_html
except ImportError:
    from download_monitor import DownloadStatus, atomic_write_json, launch_monitor, safe_message, write_monitor_html

try:
    from .cfs_grib_core import (
        SCHEMA_PLAN as ATMOSPHERIC_PLAN_SCHEMA,
        ERA_SPLIT,
        build_plan as build_atmospheric_plan,
        execute_request as execute_atmospheric_request,
        fetch_plan as fetch_atmospheric_plan,
        health as health_atmospheric,
        hycom_eligibility,
        ncei_inventory as inventory_ncei_atmospheric,
        normalize_request as normalize_atmospheric_request,
        runtime_preflight,
        legacy_cross_era_warning,
    )
except ImportError:
    from cfs_grib_core import (
        SCHEMA_PLAN as ATMOSPHERIC_PLAN_SCHEMA,
        ERA_SPLIT,
        build_plan as build_atmospheric_plan,
        execute_request as execute_atmospheric_request,
        fetch_plan as fetch_atmospheric_plan,
        health as health_atmospheric,
        hycom_eligibility,
        ncei_inventory as inventory_ncei_atmospheric,
        normalize_request as normalize_atmospheric_request,
        runtime_preflight,
        legacy_cross_era_warning,
    )


HYCOM_CFSV2_BASE = "https://tds.hycom.org/thredds/dodsC/datasets/force/ncep_cfsv2/netcdf/"
URL_PREFIXES = ("cfsv2-sec2", "cfsv2-sec", "cfsv2-sea")
MT_EPOCH = np.datetime64("1900-12-31T00:00:00", "ms")
DEFAULT_LON_RANGE: tuple[float, float] = (283.0, 288.0)
DEFAULT_LAT_RANGE: tuple[float, float] = (36.0, 41.0)

SUBDATASET_VARIABLES: dict[str, list[str]] = {
    "uv-10m": ["wndewd", "wndnwd"],
    "sfcprs": ["airprs"],
    "dlwsfc": ["dlwflx"],
    "dswsfc": ["dswflx"],
    "strblk": ["tauewd", "taunwd"],
    "wndspd": ["wndspd"],
    "TaqaQrQp": ["airtmp", "vapmix", "radflx", "shwflx"],
    "precip": ["precip"],
    "surtmp": ["surtmp"],
}
SUBDATASET_ALIASES = {"dlwflx": "dlwsfc"}
CFSV2_PRESSURE_BASE_HPA = 1000.0
LEGACY_SUBDATASET_PRODUCTS = {
    "uv-10m": "wind_10m",
    "sfcprs": "surface_pressure",
    "dlwsfc": "downward_longwave_surface_flux",
    "dswsfc": "downward_shortwave_surface_flux",
    "strblk": "surface_wind_stress",
    "precip": "precipitation_rate",
    "surtmp": "surface_temperature",
}


class Cfsv2FetcherError(RuntimeError):
    """Raised for CFSv2 request, transfer, or validation failures."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_subdataset(name: str) -> str:
    value = SUBDATASET_ALIASES.get(str(name), str(name))
    if value not in SUBDATASET_VARIABLES:
        choices = ", ".join(sorted(set(SUBDATASET_VARIABLES) | set(SUBDATASET_ALIASES)))
        raise ValueError(f"Unknown CFSv2 subdataset {name!r}; choose one of: {choices}")
    return value


def _parse_utc(value: str | np.datetime64) -> np.datetime64:
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("Timestamp cannot be NaT")
        return value.astype("datetime64[ms]")
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp {value!r} has no timezone; use Z or an explicit offset")
    return np.datetime64(parsed.astimezone(timezone.utc).replace(tzinfo=None), "ms")


def _request_time_text(value: str | np.datetime64) -> str:
    if isinstance(value, np.datetime64):
        return str(value.astype("datetime64[ms]")) + "Z"
    return str(value)


def _mt_to_datetime64(values: np.ndarray) -> np.ndarray:
    milliseconds = np.rint(np.asarray(values, dtype=np.float64) * 86_400_000.0).astype(np.int64)
    return MT_EPOCH + milliseconds.astype("timedelta64[ms]")


def cfsv2_airprs_to_absolute_pa(
    values: Any,
    *,
    source_units: str = "hPa",
    base_hpa: float = CFSV2_PRESSURE_BASE_HPA,
) -> np.ndarray:
    """Convert the HYCOM CFSv2 pressure departure from 1000 hPa to absolute Pa."""
    units = str(source_units).strip().lower().replace(" ", "")
    data = np.asanyarray(values, dtype=np.float64)
    if units in {"hpa", "hectopascal", "hectopascals", "mb", "mbar"}:
        departure_hpa = data
    elif units in {"pa", "pascal", "pascals"}:
        departure_hpa = data / 100.0
    else:
        raise ValueError(f"Unsupported CFSv2 airprs units {source_units!r}; expected hPa or Pa")
    result = (departure_hpa + float(base_hpa)) * 100.0
    if not np.all(np.isfinite(result)):
        raise ValueError("Converted CFSv2 pressure contains non-finite values")
    return result


def _public_source(value: str) -> str:
    parsed = urlsplit(str(value))
    if parsed.scheme in {"http", "https"}:
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("CFSv2 source URLs must not contain credentials, queries, or fragments")
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CFSv2 source not found: {value}")
    return str(path.resolve())


def _source_descriptor(value: str) -> tuple[str, str]:
    source = _public_source(value)
    if source.startswith(("http://", "https://")):
        return "remote", source
    return "local", Path(source).name


def _candidate_urls(year: int, subdataset: str) -> list[str]:
    name = normalize_subdataset(subdataset)
    return [f"{HYCOM_CFSV2_BASE}{prefix}_{year}_01hr_{name}.nc" for prefix in URL_PREFIXES]


def _open_source(source: str, *, decode_times: bool = False) -> xr.Dataset:
    value = _public_source(source)
    engine = "netcdf4" if value.startswith(("http://", "https://")) else None
    return xr.open_dataset(value, engine=engine, decode_times=decode_times, mask_and_scale=True, cache=False)


def resolve_cfsv2_source(
    year: int,
    subdataset: str,
    *,
    source_url: str | None = None,
) -> str:
    """Resolve an explicit source or the first available official prefix."""
    if source_url:
        candidate = str(source_url).format(year=year, subdataset=normalize_subdataset(subdataset))
        return _public_source(candidate)
    errors: list[str] = []
    for candidate in _candidate_urls(year, subdataset):
        try:
            with _open_source(candidate) as dataset:
                _ = dataset["Latitude"].values[:1]
            return candidate
        except Exception as exc:
            errors.append(f"{Path(candidate).name}: {safe_message(exc)}")
    raise Cfsv2FetcherError(
        f"No official CFSv2 source opened for {year}/{subdataset}: " + " | ".join(errors)
    )


def inventory_cfsv2(
    year: int,
    subdataset: str,
    *,
    source_url: str | None = None,
) -> dict[str, Any]:
    name = normalize_subdataset(subdataset)
    started = time.monotonic()
    source = resolve_cfsv2_source(year, name, source_url=source_url)
    with _open_source(source) as dataset:
        times = _mt_to_datetime64(np.asarray(dataset["MT"].values, dtype=np.float64))
        lat = np.asarray(dataset["Latitude"].values, dtype=float)
        lon = np.asarray(dataset["Longitude"].values, dtype=float)
        variables = {
            var: {
                "dimensions": list(data.dims),
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "units": data.attrs.get("units"),
                "standard_name": data.attrs.get("standard_name"),
                "long_name": data.attrs.get("long_name"),
            }
            for var, data in dataset.data_vars.items()
        }
        payload = {
            "schema_version": "cfsv2_inventory_v1",
            "year": int(year),
            "subdataset": name,
            "source": _source_descriptor(source)[1],
            "source_kind": _source_descriptor(source)[0],
            "dimensions": {dim: int(size) for dim, size in dataset.sizes.items()},
            "variables": variables,
            "time": {
                "name": "MT",
                "minimum": str(times.min()) + "Z",
                "maximum": str(times.max()) + "Z",
            },
            "latitude": {"name": "Latitude", "minimum": float(lat.min()), "maximum": float(lat.max())},
            "longitude": {"name": "Longitude", "minimum": float(lon.min()), "maximum": float(lon.max())},
        }
    payload["metadata_open_seconds"] = round(time.monotonic() - started, 3)
    payload["inventory_hash"] = _hash_payload(
        {key: payload[key] for key in ("year", "subdataset", "source", "dimensions", "variables")}
    )
    return payload


def _bounds(values: np.ndarray, requested: tuple[float, float], label: str) -> tuple[int, int]:
    array = np.asarray(values, dtype=float)
    if requested[0] > requested[1]:
        raise ValueError(f"{label} range must be increasing")
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"CFSv2 {label} must be a nonempty 1-D coordinate")
    ascending = np.all(np.diff(array) > 0)
    descending = np.all(np.diff(array) < 0)
    if not (ascending or descending):
        raise ValueError(f"CFSv2 {label} coordinate must be monotonic")
    mask = np.isfinite(array) & (array >= requested[0]) & (array <= requested[1])
    found = np.flatnonzero(mask)
    if found.size == 0:
        raise ValueError(f"Requested {label} range is outside source coverage")
    return max(0, int(found.min()) - 1), min(array.size, int(found.max()) + 2)


def _normalize_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    start = payload.get("start") or payload.get("start_utc")
    end = payload.get("end") or payload.get("end_utc")
    subdataset = normalize_subdataset(str(payload.get("subdataset", "")))
    if start is None or end is None:
        raise ValueError("CFSv2 request requires start and end")
    start_time, end_time = _parse_utc(str(start)), _parse_utc(str(end))
    if end_time < start_time:
        raise ValueError("CFSv2 request end precedes start")
    bbox = payload.get("bbox") or payload.get("bbox_0_360")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("CFSv2 bbox must be [west, south, east, north]")
    west, south, east, north = (float(value) for value in bbox)
    if west > east or south > north:
        raise ValueError("CFSv2 bbox must be increasing in its native 0-360 convention")
    variables = payload.get("variables") or SUBDATASET_VARIABLES[subdataset]
    if not isinstance(variables, list) or not variables:
        raise ValueError("CFSv2 variables must be a non-empty list")
    variables = list(dict.fromkeys(str(value) for value in variables))
    chunk_hours = int(payload.get("chunk_hours", 168))
    if chunk_hours < 1:
        raise ValueError("chunk_hours must be positive")
    return {
        "start": str(start),
        "end": str(end),
        "subdataset": subdataset,
        "variables": variables,
        "bbox": [west, south, east, north],
        "source_url": payload.get("source_url"),
        "chunk_hours": chunk_hours,
        "max_retries": int(payload.get("max_retries", 5)),
        "retry_delay_seconds": float(payload.get("retry_delay_seconds", 2.0)),
        "backoff": float(payload.get("backoff", 2.0)),
        "output": payload.get("output"),
    }


def _variable_bytes(dataset: xr.Dataset, variables: Sequence[str], indexers: Mapping[str, Sequence[int]]) -> int:
    total = 0
    for name in variables:
        data = dataset[name]
        count = 1
        for dim, size in data.sizes.items():
            start, stop = indexers.get(dim, (0, size))
            count *= int(stop) - int(start)
        total += count * max(1, np.dtype(data.dtype).itemsize)
    return int(total)


def _download_selection(
    source: str,
    variables: Sequence[str],
    indexers: Mapping[str, Sequence[int]],
) -> xr.Dataset:
    with _open_source(source) as dataset:
        selected = dataset[list(variables)].isel(
            {dim: slice(int(bounds[0]), int(bounds[1])) for dim, bounds in indexers.items()}
        )
        return selected.load()


def _timing_probe(
    chunk: Mapping[str, Any], variables: Sequence[str], *, repeats: int = 2
) -> dict[str, Any]:
    indexers = {
        dim: [int(bounds[0]), min(int(bounds[1]), int(bounds[0]) + (24 if dim == "MT" else 64))]
        for dim, bounds in chunk["indexers"].items()
    }
    rates: list[float] = []
    elapsed_values: list[float] = []
    byte_values: list[int] = []
    for _ in range(max(1, repeats)):
        started = time.monotonic()
        sample = _download_selection(str(chunk["source"]), variables, indexers)
        elapsed = max(1e-6, time.monotonic() - started)
        size = int(sum(data.nbytes for data in sample.data_vars.values()))
        sample.close()
        if size <= 0:
            raise Cfsv2FetcherError("CFSv2 timing probe returned zero bytes")
        elapsed_values.append(elapsed)
        byte_values.append(size)
        rates.append(size / elapsed)
    return {
        "method": "bounded_source_probe_v1",
        "repeats": len(rates),
        "probe_indexers": indexers,
        "elapsed_seconds": [round(value, 6) for value in elapsed_values],
        "bytes": byte_values,
        "bytes_per_second": [round(value, 3) for value in rates],
        "conservative_bytes_per_second": round(min(rates), 3),
        "median_request_seconds": round(float(np.median(elapsed_values)), 6),
    }


def _plan_gate(expected_bytes: int, free_bytes: int, conservative_seconds: float) -> dict[str, Any]:
    enough = free_bytes > 4 * expected_bytes
    if not enough:
        state, reason = "blocked", "Local free space is not greater than four times the request size."
    elif not math.isfinite(conservative_seconds) or conservative_seconds < 0:
        state, reason = "blocked", "A bounded timing estimate could not be established."
    elif conservative_seconds >= 600:
        state, reason = "long_run_monitor_required", "The conservative duration estimate is at least 600 seconds."
    else:
        state, reason = "ready", "The time and storage gates passed."
    return {
        "state": state,
        "reason": reason,
        "monitor_threshold_seconds": 600,
        "storage_rule": "local_free_bytes > 4 * estimated_requested_bytes",
        "storage_passed": enough,
    }


def build_cfsv2_plan(
    request_payload: Mapping[str, Any],
    *,
    run_dir: str | Path,
    probe_repeats: int = 2,
    probe_override: Mapping[str, Any] | None = None,
    free_bytes_override: int | None = None,
) -> dict[str, Any]:
    request = _normalize_request(request_payload)
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    start = _parse_utc(request["start"])
    end = _parse_utc(request["end"])
    years = range(int(str(start)[:4]), int(str(end)[:4]) + 1)
    chunks: list[dict[str, Any]] = []
    inventories: list[dict[str, Any]] = []
    selected_time_parts: list[np.ndarray] = []
    for year in years:
        source = resolve_cfsv2_source(year, request["subdataset"], source_url=request["source_url"])
        with _open_source(source) as dataset:
            for coordinate in ("MT", "Latitude", "Longitude"):
                if coordinate not in dataset:
                    raise ValueError(f"CFSv2 source is missing {coordinate}")
            missing = [name for name in request["variables"] if name not in dataset.data_vars]
            if missing:
                raise ValueError(f"Variables not present in {year}/{request['subdataset']}: {missing}")
            inventories.append(
                {
                    "inventory_hash": _hash_payload(
                        {
                            "year": year,
                            "subdataset": request["subdataset"],
                            "source": _source_descriptor(source)[1],
                            "dimensions": {dim: int(size) for dim, size in dataset.sizes.items()},
                            "variables": {
                                name: {
                                    "dimensions": list(dataset[name].dims),
                                    "shape": list(dataset[name].shape),
                                    "dtype": str(dataset[name].dtype),
                                }
                                for name in request["variables"]
                            },
                        }
                    )
                }
            )
            times = _mt_to_datetime64(np.asarray(dataset["MT"].values, dtype=float))
            selected = np.flatnonzero((times >= start) & (times <= end))
            if selected.size == 0:
                continue
            selected_time_parts.append(times[selected])
            if np.any(np.diff(selected) != 1):
                raise ValueError("Selected CFSv2 time indices are not contiguous")
            west, south, east, north = request["bbox"]
            lat_start, lat_stop = _bounds(dataset["Latitude"].values, (south, north), "latitude")
            lon_start, lon_stop = _bounds(dataset["Longitude"].values, (west, east), "longitude")
            for offset in range(0, selected.size, request["chunk_hours"]):
                subset = selected[offset : offset + request["chunk_hours"]]
                indexers = {
                    "MT": [int(subset[0]), int(subset[-1]) + 1],
                    "Latitude": [lat_start, lat_stop],
                    "Longitude": [lon_start, lon_stop],
                }
                chunks.append(
                    {
                        "id": f"y{year}-t{len(chunks) + 1:05d}",
                        "year": year,
                        "source": _public_source(source),
                        "indexers": indexers,
                        "expected_bytes": _variable_bytes(dataset, request["variables"], indexers),
                    }
                )
    if not chunks:
        raise ValueError("No CFSv2 records intersect the requested time window")
    selected_times = np.unique(np.concatenate(selected_time_parts))
    cadence = (
        np.median(np.diff(selected_times)).astype("timedelta64[ms]")
        if selected_times.size > 1
        else np.timedelta64(0, "ms")
    )
    if selected_times[0] > start or selected_times[-1] + cadence < end:
        raise ValueError(
            f"CFSv2 source coverage {selected_times[0]} through {selected_times[-1]} "
            f"does not cover requested {start} through {end}"
        )
    expected_bytes = int(sum(chunk["expected_bytes"] for chunk in chunks))
    probe = dict(probe_override) if probe_override is not None else _timing_probe(
        chunks[0], request["variables"], repeats=probe_repeats
    )
    rate = float(probe.get("conservative_bytes_per_second", 0))
    if not math.isfinite(rate) or rate <= 0:
        raise Cfsv2FetcherError("CFSv2 timing probe did not establish a positive rate")
    latency = float(probe.get("median_request_seconds", 0))
    central = expected_bytes / rate + len(chunks) * min(5.0, max(0.0, latency))
    conservative = central * 1.5 + len(chunks) * 5.0
    free_bytes = int(free_bytes_override) if free_bytes_override is not None else int(shutil.disk_usage(run_path).free)
    serialized_request = dict(request)
    if request.get("source_url"):
        raw_source = str(request["source_url"])
        if raw_source.startswith(("http://", "https://")):
            explicit = _public_source(raw_source)
        elif "{year}" in raw_source or "{subdataset}" in raw_source:
            template = Path(raw_source).expanduser()
            explicit = str(template if template.is_absolute() else template.resolve())
        else:
            explicit = _public_source(raw_source)
        serialized_request["source_url"] = (
            explicit
            if explicit.startswith(("http://", "https://"))
            else os.path.relpath(explicit, run_path)
        )
    if serialized_request.get("output"):
        requested_output = Path(str(serialized_request["output"])).expanduser()
        if not requested_output.is_absolute():
            requested_output = run_path / requested_output
        serialized_request["output"] = os.path.relpath(
            requested_output.resolve(), run_path
        )
    for chunk in chunks:
        absolute_source = str(chunk["source"])
        if absolute_source.startswith(("http://", "https://")):
            chunk["source_kind"] = "remote"
        else:
            chunk["source_kind"] = "local-relative"
            chunk["source"] = os.path.relpath(absolute_source, run_path)
    request_hash = _hash_payload(serialized_request)
    plan: dict[str, Any] = {
        "schema_version": "cfsv2_download_plan_v1",
        "connector": "cfsv2-fetcher",
        "created_utc": _utc_now(),
        "expires_utc": (datetime.now(timezone.utc) + timedelta(hours=24)).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "request": serialized_request,
        "request_hash": request_hash,
        "source_inventory_hashes": [item["inventory_hash"] for item in inventories],
        "chunks": chunks,
        "estimated_requested_bytes": expected_bytes,
        "estimated_working_bytes": expected_bytes * 4,
        "local_free_bytes": free_bytes,
        "timing_probe": probe,
        "duration_estimate_seconds": {
            "low": round(central * 0.75, 3),
            "central": round(central, 3),
            "conservative": round(conservative, 3),
        },
        "gate": _plan_gate(expected_bytes, free_bytes, conservative),
        "routing_recommendation": "local" if free_bytes > 4 * expected_bytes else "kestrel",
    }
    plan["plan_hash"] = _hash_payload(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != "cfsv2_download_plan_v1":
        raise ValueError("Unsupported CFSv2 plan schema")
    supplied = str(plan.get("plan_hash", ""))
    content = dict(plan)
    content.pop("plan_hash", None)
    if not supplied or supplied != _hash_payload(content):
        raise ValueError("CFSv2 plan hash mismatch; the plan is stale or edited")
    if str(plan.get("request_hash")) != _hash_payload(plan.get("request")):
        raise ValueError("CFSv2 request hash mismatch")
    expires = datetime.fromisoformat(str(plan["expires_utc"]).replace("Z", "+00:00"))
    if datetime.now(timezone.utc) > expires:
        raise ValueError("CFSv2 plan expired; run estimate again")
    if plan.get("gate", {}).get("state") == "blocked":
        raise ValueError(f"CFSv2 download gate is blocked: {plan['gate'].get('reason')}")


def _sanitize_dataset(dataset: xr.Dataset) -> xr.Dataset:
    valid = (str, bytes, int, float, np.integer, np.floating, np.ndarray, list, tuple)
    result = dataset.copy(deep=False)
    result.attrs = {key: value for key, value in result.attrs.items() if isinstance(value, valid)}
    for variable in result.variables.values():
        variable.attrs = {key: value for key, value in variable.attrs.items() if isinstance(value, valid)}
        variable.encoding = {}
    for name in result.data_vars:
        if np.issubdtype(result[name].dtype, np.floating):
            result[name] = result[name].astype(np.float32)
            result[name].encoding = {}
    return result


def _atomic_netcdf(dataset: xr.Dataset, output: str | Path) -> Path:
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(handle)
    try:
        _sanitize_dataset(dataset).to_netcdf(temporary, engine="netcdf4", format="NETCDF4_CLASSIC")
        with xr.open_dataset(temporary, decode_times=False) as check:
            if any(int(size) <= 0 for size in check.sizes.values()):
                raise Cfsv2FetcherError("Staged CFSv2 output contains an empty dimension")
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def _valid_checkpoint(path: Path, variables: Sequence[str], request_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path, decode_times=False) as dataset:
            return str(dataset.attrs.get("request_hash")) == request_hash and all(name in dataset for name in variables)
    except Exception:
        return False


def health_cfsv2(input_path: str | Path, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path = Path(input_path).resolve()
    checks: list[dict[str, Any]] = []
    with xr.open_dataset(path, decode_times=False) as dataset:
        variables = list((request or {}).get("variables") or dataset.data_vars)
        for name in variables:
            present = name in dataset.data_vars
            checks.append({"check": f"variable:{name}", "passed": present})
            if present:
                values = np.asarray(dataset[name].values)
                finite = np.isfinite(values) if np.issubdtype(values.dtype, np.number) else np.ones(values.shape, bool)
                checks.append({
                    "check": f"finite:{name}",
                    "passed": bool(finite.any()),
                    "finite_fraction": round(float(finite.mean()), 6),
                })
        times = _mt_to_datetime64(np.asarray(dataset["MT"].values, dtype=float))
        checks.append({"check": "time_monotonic", "passed": times.size < 2 or bool(np.all(np.diff(times) > np.timedelta64(0, "ms")))})
        if request and times.size:
            start = _parse_utc(str(request["start"]))
            end = _parse_utc(str(request["end"]))
            cadence = np.median(np.diff(times)).astype("timedelta64[ms]") if times.size > 1 else np.timedelta64(0, "ms")
            checks.append({
                "check": "requested_time_coverage",
                "passed": bool(times[0] <= start and times[-1] + cadence >= end),
                "actual_start": str(times[0]),
                "actual_end": str(times[-1]),
            })
        checks.append({"check": "nonempty_dimensions", "passed": all(int(size) > 0 for size in dataset.sizes.values())})
        payload = {
            "schema_version": "cfsv2_health_v1",
            "input_name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
            "variables": list(dataset.data_vars),
            "checks": checks,
        }
    payload["passed"] = all(bool(check["passed"]) for check in checks)
    if not payload["passed"]:
        raise Cfsv2FetcherError("CFSv2 output failed health validation")
    return payload


def fetch_cfsv2_plan(
    plan: Mapping[str, Any],
    *,
    run_dir: str | Path,
    output: str | Path | None = None,
    open_monitor: bool = True,
    cleanup_chunks: bool = False,
) -> dict[str, Any]:
    validate_plan(plan)
    destination_value = output or plan.get("request", {}).get("output")
    if not destination_value:
        raise ValueError("CFSv2 output is required in the request or fetch command")
    run_path = Path(run_dir).resolve()
    destination = (
        Path(output).expanduser().resolve()
        if output is not None
        else (run_path / str(destination_value)).resolve()
    )
    chunks_dir = run_path / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    variables = list(plan["request"]["variables"])
    request_hash = str(plan["request_hash"])
    status = DownloadStatus(
        run_path / "download_status.json",
        connector="cfsv2-fetcher",
        request_hash=request_hash,
        total_chunks=len(plan["chunks"]),
        expected_bytes=int(plan["estimated_requested_bytes"]),
        estimate_seconds=float(plan["duration_estimate_seconds"]["conservative"]),
        artifacts={"monitor": "download_monitor.html", "health": "health_check.json", "output": destination.name},
    )
    monitor = None
    if float(plan["duration_estimate_seconds"]["conservative"]) >= 600:
        monitor = (
            launch_monitor(run_path, open_browser=True)
            if open_monitor
            else {"launched": False, "html": str(write_monitor_html(run_path)), "reason": "monitor launch disabled"}
        )
        print(json.dumps({"monitor": monitor}, indent=2))
    status.start()
    completed: list[Path] = []
    completed_bytes = 0
    retries = 0
    try:
        for chunk in plan["chunks"]:
            chunk_id = str(chunk["id"])
            checkpoint = chunks_dir / f"{request_hash[:12]}-{chunk_id}.nc"
            source = str(chunk["source"])
            if chunk.get("source_kind") == "local-relative":
                source = str((run_path / source).resolve())
                if not Path(source).exists():
                    raise FileNotFoundError(
                        "The plan's relative local source is unavailable from this run directory; rerun estimate here"
                    )
            if not _valid_checkpoint(checkpoint, variables, request_hash):
                last_error: Exception | None = None
                for attempt in range(1, int(plan["request"]["max_retries"]) + 1):
                    status.update(
                        active_chunk=chunk_id,
                        attempts=int(status.data.get("attempts", 0)) + 1,
                        message=f"Downloading {chunk_id}, attempt {attempt}",
                    )
                    try:
                        piece = _download_selection(source, variables, chunk["indexers"])
                        try:
                            piece.attrs.update({
                                "connector": "cfsv2-fetcher",
                                "request_hash": request_hash,
                                "chunk_id": chunk_id,
                                "source_name": Path(source).name if not source.startswith(("http://", "https://")) else _public_source(source),
                                "subdataset": plan["request"]["subdataset"],
                            })
                            _atomic_netcdf(piece, checkpoint)
                        finally:
                            piece.close()
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt >= int(plan["request"]["max_retries"]):
                            break
                        retries += 1
                        status.update(retries=retries, message=f"Retrying {chunk_id}: {safe_message(exc)}")
                        time.sleep(float(plan["request"]["retry_delay_seconds"]) * float(plan["request"]["backoff"]) ** (attempt - 1))
                if last_error is not None:
                    status.update(failed_chunks=int(status.data.get("failed_chunks", 0)) + 1)
                    raise Cfsv2FetcherError(f"Chunk {chunk_id} failed: {safe_message(last_error)}") from last_error
            completed.append(checkpoint)
            completed_bytes += int(chunk["expected_bytes"])
            status.update(
                completed_chunks=len(completed),
                completed_bytes=completed_bytes,
                active_chunk=None,
                retries=retries,
                message=f"Completed or reused {chunk_id}",
            )
        opened = [xr.open_dataset(path, decode_times=False).load() for path in completed]
        combined = xr.concat(
            opened,
            dim="MT",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        ) if len(opened) > 1 else opened[0]
        combined.attrs.update({
            "connector": "cfsv2-fetcher",
            "request_hash": request_hash,
            "requested_start_utc": plan["request"]["start"],
            "requested_end_utc": plan["request"]["end"],
            "requested_bbox_0_360": json.dumps(plan["request"]["bbox"]),
            "subdataset": plan["request"]["subdataset"],
            "source_names": "\n".join(
                dict.fromkeys(
                    Path(str(chunk["source"])).name
                    if chunk.get("source_kind") == "local-relative"
                    else str(chunk["source"])
                    for chunk in plan["chunks"]
                )
            ),
        })
        try:
            _atomic_netcdf(combined, destination)
        finally:
            combined.close()
            for item in opened:
                if item is not combined:
                    item.close()
        health = health_cfsv2(destination, plan["request"])
        atomic_write_json(run_path / "health_check.json", health)
        status.update(completed_bytes=int(plan["estimated_requested_bytes"]), message="Health checks passed")
        status.finish("complete", "Download and health validation completed")
        if cleanup_chunks:
            for path in completed:
                path.unlink(missing_ok=True)
            try:
                chunks_dir.rmdir()
            except OSError:
                pass
        return {
            "schema_version": "cfsv2_fetch_result_v1",
            "output": str(destination),
            "health": str(run_path / "health_check.json"),
            "status": str(run_path / "download_status.json"),
            "monitor": monitor,
            "request_hash": request_hash,
            "sha256": health["sha256"],
        }
    except BaseException as exc:
        status.finish("cancelled" if isinstance(exc, KeyboardInterrupt) else "failed", safe_message(exc))
        raise


def fetch_cfsv2_window(
    start: str | np.datetime64,
    end: str | np.datetime64,
    subdataset_name: str,
    variables: list[str] | None = None,
    *,
    lon_range: tuple[float, float] = DEFAULT_LON_RANGE,
    lat_range: tuple[float, float] = DEFAULT_LAT_RANGE,
    output: str | Path,
    chunk_hours: int = 168,
    max_retries: int = 5,
    overwrite: bool = False,
    run_dir: str | Path | None = None,
    source_url: str | None = None,
    open_monitor: bool = True,
) -> Path:
    """Backward-compatible bounded API with an automatic estimate gate."""
    destination = Path(output).resolve()
    if destination.exists() and not overwrite:
        return destination
    start_value, end_value = _parse_utc(start), _parse_utc(end)
    if start_value < np.datetime64(ERA_SPLIT.replace(tzinfo=None), "ms"):
        subdataset = normalize_subdataset(subdataset_name)
        product = LEGACY_SUBDATASET_PRODUCTS.get(subdataset)
        if product is None:
            raise Cfsv2FetcherError(f"Legacy {subdataset!r} has no scientifically exact cross-era NCEI mapping; use an explicit v2 product request")
        routed_run = Path(run_dir).resolve() if run_dir else destination.parent
        routed = execute_atmospheric_request(
            "cfsv2",
            {
                "start": _request_time_text(start), "end": _request_time_text(end),
                "products": [product], "bbox": [lon_range[0], lat_range[0], lon_range[1], lat_range[1]],
                "max_retries": max_retries, "output": destination.name,
            },
            routed_run, __file__, snapshot=start_value == end_value, open_monitor=open_monitor,
        )
        routed_path = Path(routed["output"])
        if routed.get("model") == "cfs-family":
            legacy_cross_era_warning()
        return routed_path
    request = {
        "start": _request_time_text(start),
        "end": _request_time_text(end),
        "subdataset": subdataset_name,
        "variables": variables,
        "bbox": [lon_range[0], lat_range[0], lon_range[1], lat_range[1]],
        "chunk_hours": chunk_hours,
        "max_retries": max_retries,
        "source_url": source_url,
        "output": str(destination),
    }
    run_path = Path(run_dir).resolve() if run_dir else destination.parent / f"{destination.stem}_run"
    plan = build_cfsv2_plan(request, run_dir=run_path)
    atomic_write_json(run_path / "download_plan.json", plan)
    result = fetch_cfsv2_plan(plan, run_dir=run_path, open_monitor=open_monitor)
    return Path(result["output"])


def fetch_cfsv2_year(
    year: int,
    subdataset_name: str,
    variables: list[str],
    lon_range: tuple[float, float] = DEFAULT_LON_RANGE,
    lat_range: tuple[float, float] = DEFAULT_LAT_RANGE,
    cache_dir: str | Path = ".",
    chunk_days: int = 30,
    max_retries: int = 5,
    overwrite: bool = False,
    **kwargs: Any,
) -> Path:
    return fetch_cfsv2_window(
        f"{year}-01-01T00:00:00Z",
        f"{year}-12-31T23:59:59Z",
        subdataset_name,
        variables,
        lon_range=lon_range,
        lat_range=lat_range,
        output=Path(cache_dir) / f"{subdataset_name}_{year}.nc",
        chunk_hours=max(1, int(chunk_days)) * 24,
        max_retries=max_retries,
        overwrite=overwrite,
        **kwargs,
    )


def fetch_wind_year(year: int, lon_range=DEFAULT_LON_RANGE, lat_range=DEFAULT_LAT_RANGE, cache_dir=".", **kwargs: Any) -> Path:
    return fetch_cfsv2_year(year, "uv-10m", ["wndewd", "wndnwd"], lon_range, lat_range, cache_dir, **kwargs)


def fetch_pressure_year(year: int, lon_range=DEFAULT_LON_RANGE, lat_range=DEFAULT_LAT_RANGE, cache_dir=".", **kwargs: Any) -> Path:
    return fetch_cfsv2_year(year, "sfcprs", ["airprs"], lon_range, lat_range, cache_dir, **kwargs)


def load_and_concat_years(subdataset_name: str, years: list[int], cache_dir: str | Path) -> xr.Dataset:
    datasets = [xr.open_dataset(Path(cache_dir) / f"{subdataset_name}_{year}.nc", decode_times=False) for year in years]
    return xr.concat(datasets, dim="MT")


def load_cfsv2_wind(*_args: Any, **_kwargs: Any) -> None:
    raise NotImplementedError("Use fetch_wind_year() or fetch_cfsv2_window()")


def load_cfsv2_pressure(*_args: Any, **_kwargs: Any) -> None:
    raise NotImplementedError("Use fetch_pressure_year() or fetch_cfsv2_window()")


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory", help="Inspect an NCEI v2 request or a legacy HYCOM subdataset")
    inventory.add_argument("--request")
    inventory.add_argument("--year", type=int)
    inventory.add_argument("--subdataset", choices=sorted(set(SUBDATASET_VARIABLES) | set(SUBDATASET_ALIASES)))
    inventory.add_argument("--source-url")
    inventory.add_argument("--output")
    estimate = subparsers.add_parser("estimate", help="Build a timed, hash-bound download plan")
    estimate.add_argument("--request", required=True)
    estimate.add_argument("--run-dir", required=True)
    estimate.add_argument("--output", required=True)
    estimate.add_argument("--probe-repeats", type=int, default=2)
    fetch = subparsers.add_parser("fetch", help="Execute a validated CFSv2 plan")
    fetch.add_argument("--plan", required=True)
    fetch.add_argument("--run-dir", required=True)
    fetch.add_argument("--output")
    fetch.add_argument("--no-open-monitor", action="store_true")
    fetch.add_argument("--cleanup-chunks", action="store_true")
    fetch.add_argument("--cleanup-raw", action="store_true")
    window = subparsers.add_parser("window", help="Deprecated alias: estimate and fetch a bounded window")
    window.add_argument("--start", required=True)
    window.add_argument("--end", required=True)
    window.add_argument("--subdataset", required=True, choices=sorted(set(SUBDATASET_VARIABLES) | set(SUBDATASET_ALIASES)))
    window.add_argument("--variables", nargs="+")
    window.add_argument("--lon-min", type=float, required=True)
    window.add_argument("--lon-max", type=float, required=True)
    window.add_argument("--lat-min", type=float, required=True)
    window.add_argument("--lat-max", type=float, required=True)
    window.add_argument("--output", required=True)
    window.add_argument("--report")
    window.add_argument("--run-dir")
    window.add_argument("--source-url")
    window.add_argument("--chunk-hours", type=int, default=168)
    window.add_argument("--max-retries", type=int, default=5)
    window.add_argument("--overwrite", action="store_true")
    window.add_argument("--no-open-monitor", action="store_true")
    health = subparsers.add_parser("health", help="Health-check a fetched CFSv2 NetCDF")
    health.add_argument("--input", required=True)
    health.add_argument("--request")
    health.add_argument("--output", required=True)
    pressure = subparsers.add_parser("pressure-to-pa", help="Convert CFSv2 pressure departure to absolute Pa")
    pressure.add_argument("--value", type=float, required=True)
    pressure.add_argument("--units", default="hPa", choices=("hPa", "Pa"))
    snapshot = subparsers.add_parser("snapshot", help="Fetch one exact UTC NCEI/HYCOM snapshot through the v2 contract")
    snapshot.add_argument("--request", required=True)
    snapshot.add_argument("--run-dir", required=True)
    snapshot.add_argument("--no-route", action="store_true")
    snapshot.add_argument("--no-open-monitor", action="store_true")
    run = subparsers.add_parser("run", help="Route, estimate, fetch, and validate a v2 request")
    run.add_argument("--request", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--snapshot", action="store_true")
    run.add_argument("--no-route", action="store_true")
    run.add_argument("--no-open-monitor", action="store_true")
    runtime = subparsers.add_parser("runtime", help="Emit a machine-readable GRIB runtime preflight")
    runtime.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        if args.request:
            request = normalize_atmospheric_request(_read_json(args.request), "cfsv2")
            if request["provider"] == "hycom":
                payload = {"schema_version": "cfs_hycom_inventory_v2", "model": "cfsv2", **hycom_eligibility("cfsv2", request["products"])}
            else:
                payload = inventory_ncei_atmospheric("cfsv2", request)
        else:
            if args.year is None or args.subdataset is None:
                raise ValueError("inventory requires --request, or both --year and --subdataset")
            payload = inventory_cfsv2(args.year, args.subdataset, source_url=args.source_url)
        if args.output:
            atomic_write_json(args.output, payload)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "estimate":
        request_payload = _read_json(args.request)
        if request_payload.get("schema_version") == "cfs_atmospheric_request_v2" or "products" in request_payload or "product" in request_payload:
            payload = build_atmospheric_plan("cfsv2", request_payload, args.run_dir)
        else:
            payload = build_cfsv2_plan(request_payload, run_dir=args.run_dir, probe_repeats=args.probe_repeats)
        atomic_write_json(args.output, payload)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "fetch":
        plan = _read_json(args.plan)
        if plan.get("schema_version") == ATMOSPHERIC_PLAN_SCHEMA:
            payload = fetch_atmospheric_plan("cfsv2", plan, args.run_dir, output=args.output, open_monitor=not args.no_open_monitor, cleanup_raw=args.cleanup_raw)
        else:
            payload = fetch_cfsv2_plan(plan, run_dir=args.run_dir, output=args.output, open_monitor=not args.no_open_monitor, cleanup_chunks=args.cleanup_chunks)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "window":
        path = fetch_cfsv2_window(
            args.start, args.end, args.subdataset, args.variables,
            lon_range=(args.lon_min, args.lon_max), lat_range=(args.lat_min, args.lat_max),
            output=args.output, chunk_hours=args.chunk_hours, max_retries=args.max_retries,
            overwrite=args.overwrite, run_dir=args.run_dir, source_url=args.source_url,
            open_monitor=not args.no_open_monitor,
        )
        if args.report:
            health = health_cfsv2(path)
            atomic_write_json(args.report, {"schema_version": "cfsv2_window_fetch_v2", "output": str(path), "health": health})
        print(path)
        return 0
    if args.command == "health":
        request = _read_json(args.request) if args.request else None
        payload = health_atmospheric(args.input, request) if request and ("products" in request or request.get("schema_version") == "cfs_atmospheric_request_v2") else health_cfsv2(args.input, request)
        atomic_write_json(args.output, payload)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command in {"snapshot", "run"}:
        payload = execute_atmospheric_request(
            "cfsv2", _read_json(args.request), args.run_dir, __file__,
            snapshot=(args.command == "snapshot" or bool(getattr(args, "snapshot", False))),
            no_route=bool(args.no_route), open_monitor=not args.no_open_monitor,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "runtime":
        payload = runtime_preflight()
        if args.output:
            atomic_write_json(args.output, payload)
        print(json.dumps(payload, indent=2))
        return 0 if payload["passed"] else 2
    value = float(cfsv2_airprs_to_absolute_pa([args.value], source_units=args.units)[0])
    print(json.dumps({"airprs_departure": args.value, "source_units": args.units, "absolute_pressure_pa": value}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
