#!/usr/bin/env python3
"""Inventory, estimate, fetch, point-sample, and health-check native HYCOM data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import xarray as xr

try:
    from .download_monitor import DownloadStatus, atomic_write_json, launch_monitor, safe_message, write_monitor_html
except ImportError:  # direct script execution
    from download_monitor import DownloadStatus, atomic_write_json, launch_monitor, safe_message, write_monitor_html


SOURCE_ALIASES = {
    "gofs-latest": "https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest",
    "espc-d-v02-latest": "https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest",
    "gofs-3.1-2018-2024": "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0",
    "gofs-3.0-reanalysis": "https://tds.hycom.org/thredds/dodsC/GLBu0.08/reanalysis",
}

ROLE_CANDIDATES = {
    "time": ("time", "tau", "mt", "ocean_time"),
    "longitude": ("lon", "longitude", "xlon", "x"),
    "latitude": ("lat", "latitude", "ylat", "y"),
    "depth": ("depth", "lev", "level", "z"),
}

ROLE_STANDARD_NAMES = {
    "time": {"time"},
    "longitude": {"longitude", "grid_longitude"},
    "latitude": {"latitude", "grid_latitude"},
    "depth": {"depth", "sea_floor_depth_below_geoid"},
}


class HycomFetcherError(RuntimeError):
    """Raised for request, source, transfer, or validation failures."""


@dataclass(frozen=True)
class HycomRequest:
    """Serializable model-neutral HYCOM subset request."""

    source: str
    variables: tuple[str, ...]
    start: str | None = None
    end: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    depth: tuple[float, float] | None = None
    coordinate_overrides: Mapping[str, str] | None = None
    dimension_bounds: Mapping[str, Sequence[Any]] | None = None
    points: Sequence[Mapping[str, Any]] | None = None
    chunk_target_mib: float = 32.0
    max_retries: int = 5
    retry_delay_seconds: float = 5.0
    backoff: float = 2.0
    output: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_source(source: str) -> str:
    parsed = urlsplit(str(source))
    if parsed.scheme in {"http", "https"}:
        if parsed.username or parsed.password:
            raise ValueError("Credentials must not be embedded in a HYCOM source URL")
        if parsed.query or parsed.fragment:
            raise ValueError("HYCOM source URLs with query strings or fragments are not accepted")
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return str(Path(source).expanduser().resolve())


def _source_descriptor(source: str) -> tuple[str, str]:
    resolved = resolve_source(source)
    if resolved.startswith(("http://", "https://")):
        return "remote", _public_source(resolved)
    return "local", Path(resolved).name


def resolve_source(source: str) -> str:
    """Resolve a public alias or an explicit local/OPeNDAP source."""
    value = str(source).strip()
    if value in SOURCE_ALIASES:
        return SOURCE_ALIASES[value]
    if value.startswith(("http://", "https://")):
        return _public_source(value)
    path = Path(value).expanduser()
    if not path.exists():
        choices = ", ".join(sorted(SOURCE_ALIASES))
        raise FileNotFoundError(f"HYCOM source not found: {value}. Known aliases: {choices}")
    return str(path.resolve())


def _open_dataset(source: str, *, decode_times: bool = False) -> xr.Dataset:
    return xr.open_dataset(
        resolve_source(source), decode_times=decode_times, mask_and_scale=True, cache=False
    )


def _time_values(variable: xr.DataArray) -> np.ndarray:
    """Decode standard CF time or HYCOM's ``hours since analysis`` convention."""
    values = np.asarray(variable.values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[ms]")
    units = str(variable.attrs.get("units", "")).strip()
    calendar = str(variable.attrs.get("calendar", "standard"))
    if "since analysis" in units.lower():
        origin = variable.attrs.get("time_origin")
        if not origin:
            raise ValueError(
                f"Time coordinate {variable.name!r} uses 'since analysis' without time_origin"
            )
        unit = units.split("since", 1)[0].strip()
        units = f"{unit} since {origin}"
    if " since " not in units.lower():
        raise ValueError(
            f"Numeric time coordinate {variable.name!r} lacks decodable '<unit> since <origin>' units"
        )
    try:
        decoded = xr.coding.times.decode_cf_datetime(values, units, calendar=calendar)
    except Exception as exc:
        raise ValueError(
            f"Could not decode time coordinate {variable.name!r} with units {units!r}"
        ) from exc
    return np.asarray(decoded).astype("datetime64[ms]")


def _parse_overrides(values: Iterable[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"Coordinate override must be ROLE=NAME, got {value!r}")
        role, name = value.split("=", 1)
        role = role.strip().lower()
        if role not in ROLE_CANDIDATES or not name.strip():
            raise ValueError(f"Unsupported coordinate role {role!r}")
        result[role] = name.strip()
    return result


def discover_coordinates(
    dataset: xr.Dataset, overrides: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Discover time, longitude, latitude, and depth coordinates from CF metadata."""
    overrides = {str(k).lower(): str(v) for k, v in (overrides or {}).items()}
    available = set(dataset.variables)
    result: dict[str, str] = {}
    for role, name in overrides.items():
        if role not in ROLE_CANDIDATES:
            raise ValueError(f"Unknown coordinate role {role!r}")
        if name not in available:
            raise ValueError(f"Coordinate override {role}={name} is not in the source")
        result[role] = name

    for role in ROLE_CANDIDATES:
        if role in result:
            continue
        for name in dataset.variables:
            variable = dataset[name]
            standard_name = str(variable.attrs.get("standard_name", "")).lower()
            axis = str(variable.attrs.get("axis", "")).upper()
            positive = str(variable.attrs.get("positive", "")).lower()
            if standard_name in ROLE_STANDARD_NAMES[role]:
                result[role] = name
                break
            if role == "time" and axis == "T":
                result[role] = name
                break
            if role == "longitude" and axis == "X":
                result[role] = name
                break
            if role == "latitude" and axis == "Y":
                result[role] = name
                break
            if role == "depth" and (axis == "Z" or positive in {"up", "down"}):
                result[role] = name
                break
        if role in result:
            continue
        candidates = ROLE_CANDIDATES[role]
        for name in dataset.variables:
            if name.lower() in candidates:
                result[role] = name
                break
    return result


def discover_time_coordinates(dataset: xr.Dataset) -> dict[str, str]:
    """Return the best decodable time coordinate for every time-like dimension."""
    candidates: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for name in dataset.variables:
        variable = dataset[name]
        if variable.ndim != 1:
            continue
        dim = variable.dims[0]
        standard_name = str(variable.attrs.get("standard_name", "")).lower()
        axis = str(variable.attrs.get("axis", "")).upper()
        units = str(variable.attrs.get("units", "")).lower()
        lower = name.lower()
        is_time = (
            np.issubdtype(variable.dtype, np.datetime64)
            or standard_name == "time"
            or axis == "T"
            or " since " in units
            or lower in ROLE_CANDIDATES["time"]
            or lower.startswith("time")
        )
        if not is_time:
            continue
        score = 0 if name == dim else 1 if standard_name == "time" else 2 if axis == "T" else 3
        candidates[dim].append((score, name))
    return {dim: sorted(values)[0][1] for dim, values in candidates.items()}


def discover_variable_coordinates(
    dataset: xr.Dataset,
    variable_names: Sequence[str],
    global_roles: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """Map each data variable to the coordinate names that actually span it."""
    time_by_dim = discover_time_coordinates(dataset)
    result: dict[str, dict[str, str]] = {}
    for variable_name in variable_names:
        data = dataset[variable_name]
        roles: dict[str, str] = {}
        for dim in data.dims:
            if dim in time_by_dim:
                roles["time"] = time_by_dim[dim]
                roles["time_dimension"] = dim
                break
        for role in ("longitude", "latitude", "depth"):
            coordinate = global_roles.get(role)
            if coordinate and all(dim in data.dims for dim in dataset[coordinate].dims):
                roles[role] = coordinate
        result[variable_name] = roles
    return result


def _scalar_json(value: Any) -> Any:
    item = np.asarray(value).reshape(-1)[0]
    if np.issubdtype(np.asarray(item).dtype, np.datetime64):
        return str(np.datetime64(item, "ms")) + "Z"
    if isinstance(item, np.generic):
        item = item.item()
    if isinstance(item, (float, int, str, bool)) or item is None:
        return item
    return str(item)


def _coordinate_summary(variable: xr.DataArray) -> dict[str, Any]:
    values = np.asarray(variable.values)
    payload: dict[str, Any] = {
        "dimensions": list(variable.dims),
        "shape": list(variable.shape),
        "dtype": str(variable.dtype),
        "units": variable.attrs.get("units"),
        "standard_name": variable.attrs.get("standard_name"),
    }
    if values.size:
        if np.issubdtype(values.dtype, np.datetime64):
            finite = values[~np.isnat(values)]
        elif np.issubdtype(values.dtype, np.number):
            finite = values[np.isfinite(values)]
        else:
            finite = values.reshape(-1)
        if finite.size:
            payload["minimum"] = _scalar_json(np.min(finite))
            payload["maximum"] = _scalar_json(np.max(finite))
    return payload


def inventory_hycom(
    source: str,
    *,
    coordinate_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect a source without downloading its data arrays."""
    resolved = resolve_source(source)
    opened = time.monotonic()
    with _open_dataset(resolved) as dataset:
        roles = discover_coordinates(dataset, coordinate_overrides)
        variables = {
            name: {
                "dimensions": list(variable.dims),
                "shape": list(variable.shape),
                "dtype": str(variable.dtype),
                "units": variable.attrs.get("units"),
                "standard_name": variable.attrs.get("standard_name"),
                "long_name": variable.attrs.get("long_name"),
            }
            for name, variable in dataset.data_vars.items()
        }
        coordinates = {
            role: {"name": name, **_coordinate_summary(dataset[name])}
            for role, name in roles.items()
        }
        if "time" in roles:
            decoded_time = _time_values(dataset[roles["time"]])
            coordinates["time"]["minimum"] = _scalar_json(decoded_time.min())
            coordinates["time"]["maximum"] = _scalar_json(decoded_time.max())
            coordinates["time"]["decoded_from_units"] = dataset[roles["time"]].attrs.get("units")
        time_coordinates: dict[str, Any] = {}
        for dim, name in discover_time_coordinates(dataset).items():
            values = _time_values(dataset[name])
            time_coordinates[dim] = {
                "name": name,
                "size": int(values.size),
                "minimum": _scalar_json(values.min()),
                "maximum": _scalar_json(values.max()),
                "source_units": dataset[name].attrs.get("units"),
            }
        variable_roles = discover_variable_coordinates(
            dataset, list(dataset.data_vars), roles
        )
        payload = {
            "schema_version": "hycom_inventory_v1",
            "source": _source_descriptor(resolved)[1],
            "source_kind": _source_descriptor(resolved)[0],
            "source_alias": source if source in SOURCE_ALIASES else None,
            "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
            "coordinate_roles": roles,
            "time_coordinates": time_coordinates,
            "variable_coordinate_roles": variable_roles,
            "coordinates": coordinates,
            "variables": variables,
            "global_attributes": {
                key: value
                for key, value in dataset.attrs.items()
                if isinstance(value, (str, int, float))
            },
        }
    payload["metadata_open_seconds"] = round(time.monotonic() - opened, 3)
    payload["inventory_hash"] = _hash_payload(
        {
            "source": payload["source"],
            "dimensions": payload["dimensions"],
            "coordinate_roles": payload["coordinate_roles"],
            "time_coordinates": payload["time_coordinates"],
            "variable_coordinate_roles": payload["variable_coordinate_roles"],
            "variables": payload["variables"],
        }
    )
    return payload


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _normalize_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = str(payload.get("source", "")).strip()
    raw_variables = payload.get("variables")
    if not source:
        raise ValueError("Request field 'source' is required")
    if not isinstance(raw_variables, list) or not raw_variables:
        raise ValueError("Request field 'variables' must be a non-empty JSON list")
    variables = list(dict.fromkeys(str(value).strip() for value in raw_variables if str(value).strip()))
    if not variables:
        raise ValueError("At least one non-empty variable name is required")
    request: dict[str, Any] = {
        "source": source,
        "variables": variables,
        "start": payload.get("start") or payload.get("start_utc"),
        "end": payload.get("end") or payload.get("end_utc"),
        "bbox": payload.get("bbox"),
        "depth": payload.get("depth") or payload.get("depth_range"),
        "coordinate_overrides": dict(payload.get("coordinate_overrides") or {}),
        "dimension_bounds": dict(payload.get("dimension_bounds") or {}),
        "points": payload.get("points"),
        "chunk_target_mib": float(payload.get("chunk_target_mib", 32.0)),
        "max_retries": int(payload.get("max_retries", 5)),
        "retry_delay_seconds": float(payload.get("retry_delay_seconds", 5.0)),
        "backoff": float(payload.get("backoff", 2.0)),
        "output": payload.get("output"),
    }
    if (request["start"] is None) != (request["end"] is None):
        raise ValueError("Both start and end are required when either is supplied")
    if request["bbox"] is not None:
        if not isinstance(request["bbox"], list) or len(request["bbox"]) != 4:
            raise ValueError("bbox must be [west, south, east, north]")
        request["bbox"] = [float(value) for value in request["bbox"]]
        if request["bbox"][3] < request["bbox"][1]:
            raise ValueError("bbox north must not be south of bbox south")
    if request["depth"] is not None:
        if not isinstance(request["depth"], list) or len(request["depth"]) != 2:
            raise ValueError("depth must be [minimum, maximum]")
        request["depth"] = [float(value) for value in request["depth"]]
        if request["depth"][1] < request["depth"][0]:
            raise ValueError("depth maximum must not precede depth minimum")
    if request["chunk_target_mib"] <= 0:
        raise ValueError("chunk_target_mib must be positive")
    if request["max_retries"] < 1 or request["backoff"] < 1 or request["retry_delay_seconds"] < 0:
        raise ValueError("Retry settings are invalid")
    if request["points"] is not None and not isinstance(request["points"], list):
        raise ValueError("points must be a JSON list of objects with lon and lat")
    return request


def _slice_from_mask(mask: np.ndarray, label: str) -> tuple[int, int]:
    found = np.flatnonzero(mask)
    if found.size == 0:
        raise ValueError(f"Requested {label} does not intersect the source coordinate")
    return int(found.min()), int(found.max()) + 1


def _bound_mask(values: np.ndarray, low: Any, high: Any) -> np.ndarray:
    if np.issubdtype(values.dtype, np.datetime64):
        lo = np.datetime64(str(low).replace("Z", ""))
        hi = np.datetime64(str(high).replace("Z", ""))
        if hi < lo:
            raise ValueError("time end precedes time start")
        return (~np.isnat(values)) & (values >= lo) & (values <= hi)
    numeric = values.astype(np.float64)
    return np.isfinite(numeric) & (numeric >= float(low)) & (numeric <= float(high))


def _normalize_lon(value: float, source_values: np.ndarray) -> float:
    finite = source_values[np.isfinite(source_values)]
    if finite.size == 0:
        raise ValueError("Longitude coordinate contains no finite values")
    if float(np.nanmin(finite)) >= 0.0 and float(np.nanmax(finite)) > 180.0:
        return float(value) % 360.0
    return ((float(value) + 180.0) % 360.0) - 180.0


def _spatial_segments(
    dataset: xr.Dataset,
    roles: Mapping[str, str],
    bbox: Sequence[float] | None,
) -> list[dict[str, list[int]]]:
    if bbox is None:
        return [{}]
    if "longitude" not in roles or "latitude" not in roles:
        raise ValueError("A bbox requires discovered longitude and latitude coordinates")
    lon_name, lat_name = roles["longitude"], roles["latitude"]
    lon = np.asarray(dataset[lon_name].values)
    lat = np.asarray(dataset[lat_name].values)
    west, south, east, north = (float(value) for value in bbox)
    norm_west = _normalize_lon(west, lon)
    norm_east = _normalize_lon(east, lon)
    lon_ranges = (
        [(norm_west, norm_east)]
        if norm_west <= norm_east
        else [(norm_west, float(np.nanmax(lon))), (float(np.nanmin(lon)), norm_east)]
    )
    segments: list[dict[str, list[int]]] = []
    if lon.ndim == lat.ndim == 1:
        lat_start, lat_stop = _slice_from_mask(_bound_mask(lat, south, north), "latitude range")
        for lo, hi in lon_ranges:
            lon_start, lon_stop = _slice_from_mask(_bound_mask(lon, lo, hi), "longitude range")
            segments.append(
                {
                    dataset[lat_name].dims[0]: [lat_start, lat_stop],
                    dataset[lon_name].dims[0]: [lon_start, lon_stop],
                }
            )
        return segments
    lon_dims = dataset[lon_name].dims
    lat_dims = dataset[lat_name].dims
    if lon.ndim != 2 or lat.ndim != 2 or lon_dims != lat_dims:
        raise ValueError("Only rectilinear or matching 2-D curvilinear lon/lat coordinates are supported")
    for lo, hi in lon_ranges:
        mask = _bound_mask(lon, lo, hi) & _bound_mask(lat, south, north)
        rows, columns = np.where(mask)
        if rows.size == 0:
            raise ValueError("Requested bbox does not intersect the curvilinear source grid")
        segments.append(
            {
                lon_dims[0]: [int(rows.min()), int(rows.max()) + 1],
                lon_dims[1]: [int(columns.min()), int(columns.max()) + 1],
            }
        )
    return segments


def _selection_indexers(
    dataset: xr.Dataset,
    request: Mapping[str, Any],
    roles: Mapping[str, str],
) -> tuple[dict[str, list[int]], list[dict[str, list[int]]]]:
    indexers: dict[str, list[int]] = {}
    if request.get("depth") is not None:
        if "depth" not in roles:
            raise ValueError("Depth bounds were supplied but no depth coordinate was discovered")
        variable = dataset[roles["depth"]]
        if variable.ndim != 1:
            raise ValueError("The depth coordinate must be one-dimensional")
        indexers[variable.dims[0]] = list(
            _slice_from_mask(
                _bound_mask(np.asarray(variable.values), *request["depth"]), "depth range"
            )
        )
    for name, bounds in request.get("dimension_bounds", {}).items():
        if name not in dataset.variables:
            raise ValueError(f"dimension_bounds coordinate {name!r} is not in the source")
        variable = dataset[name]
        if variable.ndim != 1 or len(bounds) != 2:
            raise ValueError(f"dimension_bounds[{name!r}] requires a 1-D coordinate and two bounds")
        indexers[variable.dims[0]] = list(
            _slice_from_mask(_bound_mask(np.asarray(variable.values), bounds[0], bounds[1]), name)
        )
    return indexers, _spatial_segments(dataset, roles, request.get("bbox"))


def _selection_size(
    dataset: xr.Dataset, variables: Sequence[str], indexers: Mapping[str, Sequence[int]]
) -> int:
    total = 0
    for name in variables:
        variable = dataset[name]
        count = 1
        for dim, size in variable.sizes.items():
            start, stop = indexers.get(dim, (0, size))
            count *= max(0, int(stop) - int(start))
        try:
            itemsize = np.dtype(variable.dtype).itemsize
        except TypeError:
            itemsize = 8
        total += count * max(1, itemsize)
    return int(total)


def _merged_indexers(*parts: Mapping[str, Sequence[int]]) -> dict[str, list[int]]:
    merged: dict[str, list[int]] = {}
    for part in parts:
        for name, bounds in part.items():
            if name in merged:
                merged[name] = [max(merged[name][0], int(bounds[0])), min(merged[name][1], int(bounds[1]))]
            else:
                merged[name] = [int(bounds[0]), int(bounds[1])]
    if any(stop <= start for start, stop in merged.values()):
        raise ValueError("Combined selection is empty")
    return merged


def _make_chunks(
    dataset: xr.Dataset,
    variables: Sequence[str],
    base: Mapping[str, Sequence[int]],
    segments: Sequence[Mapping[str, Sequence[int]]],
    time_dim: str | None,
    target_bytes: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    time_bounds = base.get(time_dim, (0, dataset.sizes[time_dim])) if time_dim else (0, 1)
    t0, t1 = int(time_bounds[0]), int(time_bounds[1])
    representative = _merged_indexers(base, segments[0])
    full_size = _selection_size(dataset, variables, representative)
    ntime = max(1, t1 - t0) if time_dim else 1
    steps = ntime if not time_dim else max(1, int(target_bytes / max(1, full_size / ntime)))
    for segment_index, segment in enumerate(segments):
        cursor_values = range(t0, t1, steps) if time_dim else (0,)
        for cursor in cursor_values:
            time_part = {time_dim: [cursor, min(t1, cursor + steps)]} if time_dim else {}
            indexers = _merged_indexers(base, segment, time_part)
            chunk_id = f"s{segment_index + 1:02d}-t{len(chunks) + 1:05d}"
            chunks.append(
                {
                    "id": chunk_id,
                    "segment": segment_index,
                    "indexers": indexers,
                    "expected_bytes": _selection_size(dataset, variables, indexers),
                }
            )
    return chunks


def _probe_indexers(indexers: Mapping[str, Sequence[int]], roles: Mapping[str, str], dataset: xr.Dataset) -> dict[str, list[int]]:
    coordinate_dims = {
        role: dataset[name].dims
        for role, name in roles.items()
        if name in dataset.variables
    }
    caps: dict[str, int] = {}
    for role, dims in coordinate_dims.items():
        for dim in dims:
            caps[dim] = min(caps.get(dim, 10**9), 1 if role == "time" else 4 if role == "depth" else 96)
    result: dict[str, list[int]] = {}
    for dim, bounds in indexers.items():
        start, stop = int(bounds[0]), int(bounds[1])
        result[dim] = [start, min(stop, start + caps.get(dim, 4))]
    return result


def _load_selection(
    source: str,
    variables: Sequence[str],
    indexers: Mapping[str, Sequence[int]],
) -> xr.Dataset:
    with _open_dataset(source) as dataset:
        selected = dataset[list(variables)]
        selected = selected.isel(
            {
                name: slice(int(bounds[0]), int(bounds[1]))
                for name, bounds in indexers.items()
                if name in selected.dims
            }
        )
        return selected.load()


def probe_source(
    source: str,
    variables: Sequence[str],
    indexers: Mapping[str, Sequence[int]],
    roles: Mapping[str, str],
    *,
    repeats: int = 2,
) -> dict[str, Any]:
    """Run bounded representative reads for transfer-time calibration."""
    timings: list[float] = []
    rates: list[float] = []
    bytes_read: list[int] = []
    with _open_dataset(source) as metadata:
        probe_indexers = _probe_indexers(indexers, roles, metadata)
    for _ in range(max(1, int(repeats))):
        started = time.monotonic()
        sample = _load_selection(source, variables, probe_indexers)
        elapsed = max(1e-6, time.monotonic() - started)
        transferred = int(sum(variable.nbytes for variable in sample.data_vars.values()))
        sample.close()
        if transferred <= 0:
            raise HycomFetcherError("Timed HYCOM probe returned zero data bytes")
        timings.append(elapsed)
        bytes_read.append(transferred)
        rates.append(transferred / elapsed)
    return {
        "method": "bounded_source_probe_v1",
        "probe_indexers": probe_indexers,
        "repeats": len(timings),
        "elapsed_seconds": [round(value, 6) for value in timings],
        "bytes": bytes_read,
        "bytes_per_second": [round(value, 3) for value in rates],
        "conservative_bytes_per_second": round(float(min(rates)), 3),
        "median_request_seconds": round(float(np.median(timings)), 6),
    }


def _plan_gate(expected_bytes: int, free_bytes: int, conservative_seconds: float) -> dict[str, Any]:
    enough = free_bytes > 4 * expected_bytes
    if not enough:
        state = "blocked"
        reason = "Local free space is not greater than four times the estimated request size."
    elif not math.isfinite(conservative_seconds) or conservative_seconds < 0:
        state = "blocked"
        reason = "A bounded timing estimate could not be established."
    elif conservative_seconds >= 600.0:
        state = "long_run_monitor_required"
        reason = "The conservative duration estimate is at least 600 seconds."
    else:
        state = "ready"
        reason = "The time and storage gates passed."
    return {
        "state": state,
        "reason": reason,
        "monitor_threshold_seconds": 600,
        "storage_rule": "local_free_bytes > 4 * estimated_requested_bytes",
        "storage_passed": enough,
    }


def build_hycom_plan(
    request_payload: Mapping[str, Any],
    *,
    run_dir: str | Path,
    probe_repeats: int = 2,
    probe_override: Mapping[str, Any] | None = None,
    free_bytes_override: int | None = None,
) -> dict[str, Any]:
    """Create a hash-bound request plan after schema, time, and storage estimation."""
    request = _normalize_request(request_payload)
    resolved = resolve_source(request["source"])
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    with _open_dataset(resolved) as dataset:
        roles = discover_coordinates(dataset, request["coordinate_overrides"])
        missing = [name for name in request["variables"] if name not in dataset.data_vars]
        if missing:
            raise ValueError(f"Variables not found in source: {missing}")
        base, segments = _selection_indexers(dataset, request, roles)
        variable_roles = discover_variable_coordinates(dataset, request["variables"], roles)
        longitude_dim = dataset[roles["longitude"]].dims[-1] if "longitude" in roles else None
        variable_groups: dict[str | None, list[str]] = defaultdict(list)
        for variable_name in request["variables"]:
            variable_groups[variable_roles[variable_name].get("time_dimension")].append(variable_name)
        chunks: list[dict[str, Any]] = []
        for group_index, (time_dim, group_variables) in enumerate(variable_groups.items(), start=1):
            group_base = dict(base)
            if request.get("start") is not None:
                if time_dim is None:
                    time_like_dims = [dim for dim in dataset[group_variables[0]].dims if dim.lower().startswith("time")]
                    if time_like_dims:
                        raise ValueError(
                            f"No decodable time coordinate was found for {group_variables}"
                        )
                else:
                    time_coordinate = variable_roles[group_variables[0]]["time"]
                    group_base[time_dim] = list(
                        _slice_from_mask(
                            _bound_mask(
                                _time_values(dataset[time_coordinate]),
                                request["start"],
                                request["end"],
                            ),
                            f"time range for {time_coordinate}",
                        )
                    )
            group_chunks = _make_chunks(
                dataset,
                group_variables,
                group_base,
                segments,
                time_dim,
                int(request["chunk_target_mib"] * 1024**2),
            )
            for chunk in group_chunks:
                chunk["id"] = f"g{group_index:02d}-{chunk['id']}"
                chunk["variables"] = list(group_variables)
                chunk["time_group"] = time_dim
            chunks.extend(group_chunks)
        inventory = inventory_hycom(resolved, coordinate_overrides=request["coordinate_overrides"])
    if not chunks:
        raise ValueError("The request generated no download chunks")
    expected_bytes = int(sum(chunk["expected_bytes"] for chunk in chunks))
    probe = dict(probe_override) if probe_override is not None else probe_source(
        resolved,
        chunks[0]["variables"],
        chunks[0]["indexers"],
        roles,
        repeats=probe_repeats,
    )
    rate = float(probe.get("conservative_bytes_per_second", 0.0))
    if not math.isfinite(rate) or rate <= 0:
        raise HycomFetcherError("Bounded timing probe did not yield a positive transfer rate")
    latency = float(probe.get("median_request_seconds", 0.0))
    central = expected_bytes / rate + len(chunks) * min(5.0, max(0.0, latency))
    conservative = central * 1.5 + len(chunks) * 5.0
    free_bytes = int(free_bytes_override) if free_bytes_override is not None else int(shutil.disk_usage(run_path).free)
    source_kind = "remote" if resolved.startswith(("http://", "https://")) else "local-relative"
    source_reference = _public_source(resolved) if source_kind == "remote" else os.path.relpath(resolved, run_path)
    request_for_hash = {**request, "source": source_reference}
    if request_for_hash.get("output"):
        requested_output = Path(str(request_for_hash["output"])).expanduser()
        if not requested_output.is_absolute():
            requested_output = run_path / requested_output
        request_for_hash["output"] = os.path.relpath(
            requested_output.resolve(), run_path
        )
    request_hash = _hash_payload(request_for_hash)
    plan: dict[str, Any] = {
        "schema_version": "hycom_download_plan_v1",
        "connector": "hycom-fetcher",
        "created_utc": _utc_now(),
        "expires_utc": (
            datetime.now(timezone.utc) + timedelta(hours=24)
        ).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "request": request_for_hash,
        "request_hash": request_hash,
        "source_kind": source_kind,
        "source_inventory_hash": inventory["inventory_hash"],
        "coordinate_roles": roles,
        "variable_coordinate_roles": variable_roles,
        "time_dimensions": sorted(
            {str(chunk["time_group"]) for chunk in chunks if chunk.get("time_group")}
        ),
        "longitude_dimension": longitude_dim,
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
    if plan.get("schema_version") != "hycom_download_plan_v1":
        raise ValueError("Unsupported HYCOM plan schema")
    supplied = str(plan.get("plan_hash", ""))
    content = dict(plan)
    content.pop("plan_hash", None)
    if not supplied or supplied != _hash_payload(content):
        raise ValueError("HYCOM plan hash mismatch; the plan is stale or has been edited")
    if str(plan.get("request_hash", "")) != _hash_payload(plan.get("request")):
        raise ValueError("HYCOM request hash mismatch")
    expires = datetime.fromisoformat(str(plan["expires_utc"]).replace("Z", "+00:00"))
    if datetime.now(timezone.utc) > expires:
        raise ValueError("HYCOM plan has expired; run estimate again")
    if plan.get("gate", {}).get("state") == "blocked":
        raise ValueError(f"HYCOM download gate is blocked: {plan['gate'].get('reason')}")


def _clean_attr(value: Any) -> Any:
    if isinstance(value, (str, bytes, int, float, np.number, np.ndarray, list, tuple)):
        return value
    return json.dumps(value, sort_keys=True, default=_json_default)


def _sanitize_dataset(dataset: xr.Dataset) -> xr.Dataset:
    result = dataset.copy(deep=False)
    result.attrs = {str(key): _clean_attr(value) for key, value in result.attrs.items()}
    for name in result.variables:
        result[name].attrs = {
            str(key): _clean_attr(value) for key, value in result[name].attrs.items()
        }
    return result


def _atomic_netcdf(dataset: xr.Dataset, output: str | Path) -> Path:
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    try:
        _sanitize_dataset(dataset).to_netcdf(temporary, engine="netcdf4", format="NETCDF4")
        with xr.open_dataset(temporary, decode_times=False) as check:
            if any(int(size) <= 0 for size in check.sizes.values()):
                raise HycomFetcherError("Staged NetCDF contains an empty dimension")
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def _download_chunk(
    plan: Mapping[str, Any],
    chunk: Mapping[str, Any],
    output: Path,
    source: str,
) -> Path:
    dataset = _load_selection(
        source,
        list(chunk["variables"]),
        chunk["indexers"],
    )
    try:
        dataset.attrs.update(
            {
                "connector": "hycom-fetcher",
                "request_hash": str(plan["request_hash"]),
                "chunk_id": str(chunk["id"]),
                "source_name": Path(source).name if not source.startswith(("http://", "https://")) else _public_source(source),
            }
        )
        return _atomic_netcdf(dataset, output)
    finally:
        dataset.close()


def _valid_checkpoint(path: Path, variables: Sequence[str], request_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path, decode_times=False) as dataset:
            return (
                str(dataset.attrs.get("request_hash")) == request_hash
                and all(name in dataset.data_vars for name in variables)
                and all(int(size) > 0 for size in dataset.sizes.values())
            )
    except Exception:
        return False


def _combine_chunks(
    plan: Mapping[str, Any], chunk_paths: Mapping[str, Path]
) -> xr.Dataset:
    groups: dict[tuple[str, int], list[tuple[Mapping[str, Any], Path]]] = defaultdict(list)
    by_id = {str(chunk["id"]): chunk for chunk in plan["chunks"]}
    for chunk_id, path in chunk_paths.items():
        chunk = by_id[chunk_id]
        groups[(str(chunk.get("time_group") or "__static__"), int(chunk["segment"]))].append((chunk, path))
    assembled_by_time: dict[str, list[tuple[int, xr.Dataset]]] = defaultdict(list)
    for (time_group, segment_id), entries in sorted(groups.items()):
        opened: list[xr.Dataset] = []
        for chunk, path in sorted(
            entries,
            key=lambda pair: pair[0]["indexers"].get(time_group, [0, 1])[0]
            if time_group != "__static__"
            else 0,
        ):
            opened.append(xr.open_dataset(path, decode_times=False).load())
        if time_group != "__static__" and len(opened) > 1:
            segment = xr.concat(
                opened,
                dim=time_group,
                data_vars="minimal",
                coords="minimal",
                compat="override",
                combine_attrs="override",
            )
        else:
            segment = opened[0]
        for item in opened:
            if item is not segment:
                item.close()
        assembled_by_time[time_group].append((segment_id, segment))
    time_products: list[xr.Dataset] = []
    lon_role = plan.get("coordinate_roles", {}).get("longitude")
    for time_group, segment_items in assembled_by_time.items():
        segment_items.sort(key=lambda item: item[0])
        segments = [item[1] for item in segment_items]
        if len(segments) == 1:
            time_products.append(segments[0])
            continue
        if not lon_role:
            raise HycomFetcherError("Multiple spatial segments require a longitude coordinate")
        lon_dim = segments[0][lon_role].dims[-1]
        product = xr.concat(
            segments,
            dim=lon_dim,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            combine_attrs="override",
        )
        for segment in segments:
            if segment is not product:
                segment.close()
        time_products.append(product)
    if len(time_products) == 1:
        return time_products[0]
    result = xr.merge(
        time_products,
        compat="override",
        join="outer",
        combine_attrs="override",
    )
    for product in time_products:
        if product is not result:
            product.close()
    return result


def _point_sample(dataset: xr.Dataset, plan: Mapping[str, Any]) -> xr.Dataset:
    points = plan.get("request", {}).get("points")
    if not points:
        return dataset
    roles = plan.get("coordinate_roles", {})
    lon_name, lat_name = roles.get("longitude"), roles.get("latitude")
    if not lon_name or not lat_name or dataset[lon_name].ndim != 1 or dataset[lat_name].ndim != 1:
        raise ValueError("Generic point sampling requires rectilinear 1-D longitude and latitude")
    names: list[str] = []
    lons: list[float] = []
    lats: list[float] = []
    source_lon = np.asarray(dataset[lon_name].values, dtype=float)
    for index, point in enumerate(points):
        if not isinstance(point, Mapping) or "lon" not in point or "lat" not in point:
            raise ValueError("Each point requires lon and lat")
        names.append(str(point.get("name", f"point_{index + 1}")))
        lons.append(_normalize_lon(float(point["lon"]), source_lon))
        lats.append(float(point["lat"]))
    target_lon = xr.DataArray(np.asarray(lons), dims="point")
    target_lat = xr.DataArray(np.asarray(lats), dims="point")
    sampled = dataset.interp({lon_name: target_lon, lat_name: target_lat}, method="linear")
    sampled = sampled.assign_coords(
        point=("point", names),
        requested_longitude=("point", [float(point["lon"]) for point in points]),
        requested_latitude=("point", lats),
    )
    sampled.attrs.update(dataset.attrs)
    sampled.attrs["sampling_method"] = "xarray linear interpolation on native rectilinear grid"
    return sampled


def health_hycom(
    input_path: str | Path,
    *,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Check structure, requested variables, time monotonicity, and finite coverage."""
    path = Path(input_path).resolve()
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"HYCOM output is missing or empty: {path}")
    checks: list[dict[str, Any]] = []
    with xr.open_dataset(path, decode_times=False) as dataset:
        variables = list((request or {}).get("variables") or dataset.data_vars)
        for name in variables:
            present = name in dataset.data_vars
            checks.append({"check": f"variable:{name}", "passed": present})
            if not present:
                continue
            values = np.asarray(dataset[name].values)
            if np.issubdtype(values.dtype, np.number):
                finite = np.isfinite(values)
                count = int(finite.sum())
                checks.append(
                    {
                        "check": f"finite:{name}",
                        "passed": count > 0,
                        "finite_count": count,
                        "total_count": int(values.size),
                        "finite_fraction": round(count / max(1, values.size), 6),
                    }
                )
        roles = discover_coordinates(dataset, (request or {}).get("coordinate_overrides"))
        used_dims = {
            dim
            for name in variables
            if name in dataset.data_vars
            for dim in dataset[name].dims
        }
        for time_dim, time_name in discover_time_coordinates(dataset).items():
            if time_dim not in used_dims:
                continue
            values = _time_values(dataset[time_name])
            if values.size < 2:
                monotonic = True
            elif np.issubdtype(values.dtype, np.datetime64):
                monotonic = bool(np.all(np.diff(values) > np.timedelta64(0, "ns")))
            elif np.issubdtype(values.dtype, np.number):
                monotonic = bool(np.all(np.diff(values.astype(float)) > 0))
            else:
                monotonic = all(values[index] < values[index + 1] for index in range(values.size - 1))
            checks.append(
                {
                    "check": f"time_monotonic:{time_name}",
                    "passed": monotonic,
                    "time_dimension": time_dim,
                }
            )
        checks.append(
            {
                "check": "nonempty_dimensions",
                "passed": all(int(size) > 0 for size in dataset.sizes.values()),
            }
        )
        payload = {
            "schema_version": "hycom_health_v1",
            "input_name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
            "variables": list(dataset.data_vars),
            "coordinate_roles": roles,
            "checks": checks,
        }
    payload["passed"] = all(bool(item["passed"]) for item in checks)
    if not payload["passed"]:
        raise HycomFetcherError("HYCOM output failed one or more health checks")
    return payload


def fetch_hycom_plan(
    plan: Mapping[str, Any],
    *,
    output: str | Path | None = None,
    run_dir: str | Path,
    open_monitor: bool = True,
    cleanup_chunks: bool = False,
) -> dict[str, Any]:
    """Execute a validated plan with chunk checkpoints and persistent status."""
    validate_plan(plan)
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    destination_value = output or plan.get("request", {}).get("output")
    if not destination_value:
        raise ValueError("An output path is required in the request or fetch command")
    destination = (
        Path(output).expanduser().resolve()
        if output is not None
        else (run_path / str(destination_value)).resolve()
    )
    source = str(plan["request"]["source"])
    if plan.get("source_kind") == "local-relative":
        source = str((run_path / source).resolve())
        if not Path(source).exists():
            raise FileNotFoundError(
                "The plan's relative local source is unavailable from this run directory; rerun estimate here"
            )
    chunks_dir = run_path / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    request_hash = str(plan["request_hash"])
    status = DownloadStatus(
        run_path / "download_status.json",
        connector="hycom-fetcher",
        request_hash=request_hash,
        total_chunks=len(plan["chunks"]),
        expected_bytes=int(plan["estimated_requested_bytes"]),
        estimate_seconds=float(plan["duration_estimate_seconds"]["conservative"]),
        artifacts={
            "monitor": "download_monitor.html",
            "health": "health_check.json",
            "output": destination.name,
        },
    )
    monitor: dict[str, Any] | None = None
    if float(plan["duration_estimate_seconds"]["conservative"]) >= 600.0:
        monitor = (
            launch_monitor(run_path, open_browser=True)
            if open_monitor
            else {"launched": False, "html": str(write_monitor_html(run_path)), "reason": "monitor launch disabled"}
        )
        print(json.dumps({"monitor": monitor}, indent=2))
    status.start()
    completed: dict[str, Path] = {}
    completed_bytes = 0
    retries = 0
    try:
        for chunk in plan["chunks"]:
            chunk_id = str(chunk["id"])
            checkpoint = chunks_dir / f"{request_hash[:12]}-{chunk_id}.nc"
            if _valid_checkpoint(checkpoint, list(chunk["variables"]), request_hash):
                completed[chunk_id] = checkpoint
                completed_bytes += int(chunk["expected_bytes"])
                status.update(
                    completed_chunks=len(completed),
                    completed_bytes=completed_bytes,
                    retries=retries,
                    message=f"Reused checkpoint {chunk_id}",
                )
                continue
            last_error: Exception | None = None
            for attempt in range(1, int(plan["request"]["max_retries"]) + 1):
                status.update(
                    active_chunk=chunk_id,
                    attempts=int(status.data.get("attempts", 0)) + 1,
                    message=f"Downloading {chunk_id}, attempt {attempt}",
                )
                try:
                    _download_chunk(plan, chunk, checkpoint, source)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt >= int(plan["request"]["max_retries"]):
                        break
                    retries += 1
                    status.update(retries=retries, message=f"Retrying {chunk_id}: {safe_message(exc)}")
                    delay = float(plan["request"]["retry_delay_seconds"]) * (
                        float(plan["request"]["backoff"]) ** (attempt - 1)
                    )
                    time.sleep(delay)
            if last_error is not None:
                status.update(failed_chunks=int(status.data.get("failed_chunks", 0)) + 1)
                raise HycomFetcherError(f"Chunk {chunk_id} failed: {safe_message(last_error)}") from last_error
            completed[chunk_id] = checkpoint
            completed_bytes += int(chunk["expected_bytes"])
            status.update(
                completed_chunks=len(completed),
                completed_bytes=completed_bytes,
                active_chunk=None,
                retries=retries,
                message=f"Completed {chunk_id}",
            )
        combined = _combine_chunks(plan, completed)
        try:
            final = _point_sample(combined, plan)
            final.attrs.update(
                {
                    "connector": "hycom-fetcher",
                    "request_hash": request_hash,
                    "source_name": Path(source).name if not source.startswith(("http://", "https://")) else _public_source(source),
                    "plan_created_utc": str(plan["created_utc"]),
                }
            )
            _atomic_netcdf(final, destination)
            if final is not combined:
                final.close()
        finally:
            combined.close()
        health = health_hycom(destination, request=plan["request"])
        atomic_write_json(run_path / "health_check.json", health)
        status.update(completed_bytes=int(plan["estimated_requested_bytes"]), message="Health checks passed")
        status.finish("complete", "Download and health validation completed")
        if cleanup_chunks:
            for path in completed.values():
                path.unlink(missing_ok=True)
            try:
                chunks_dir.rmdir()
            except OSError:
                pass
        return {
            "schema_version": "hycom_fetch_result_v1",
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


def _write_payload(path: str | Path | None, payload: Mapping[str, Any]) -> None:
    if path:
        atomic_write_json(path, dict(payload))
    print(json.dumps(payload, indent=2, default=_json_default))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Inspect a HYCOM source schema")
    inventory.add_argument("--source", required=True, help="Alias, local NetCDF, or public OPeNDAP URL")
    inventory.add_argument("--coordinate", action="append", help="Coordinate override ROLE=NAME")
    inventory.add_argument("--output", help="Inventory JSON path; otherwise print only")

    estimate = subparsers.add_parser("estimate", help="Build a hash-bound timed download plan")
    estimate.add_argument("--request", required=True, help="Request JSON")
    estimate.add_argument("--run-dir", required=True)
    estimate.add_argument("--output", required=True, help="Plan JSON")
    estimate.add_argument("--probe-repeats", type=int, default=2)

    fetch = subparsers.add_parser("fetch", help="Execute a validated plan")
    fetch.add_argument("--plan", required=True)
    fetch.add_argument("--run-dir", required=True)
    fetch.add_argument("--output", help="Override request output")
    fetch.add_argument("--no-open-monitor", action="store_true")
    fetch.add_argument("--cleanup-chunks", action="store_true")

    health = subparsers.add_parser("health", help="Health-check a fetched NetCDF")
    health.add_argument("--input", required=True)
    health.add_argument("--request", help="Original request JSON")
    health.add_argument("--output", required=True, help="Health JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        payload = inventory_hycom(
            args.source, coordinate_overrides=_parse_overrides(args.coordinate)
        )
        _write_payload(args.output, payload)
        return 0
    if args.command == "estimate":
        payload = build_hycom_plan(
            _read_json(args.request),
            run_dir=args.run_dir,
            probe_repeats=args.probe_repeats,
        )
        _write_payload(args.output, payload)
        return 0
    if args.command == "fetch":
        payload = fetch_hycom_plan(
            _read_json(args.plan),
            output=args.output,
            run_dir=args.run_dir,
            open_monitor=not args.no_open_monitor,
            cleanup_chunks=args.cleanup_chunks,
        )
        print(json.dumps(payload, indent=2))
        return 0
    request = _normalize_request(_read_json(args.request)) if args.request else None
    payload = health_hycom(args.input, request=request)
    _write_payload(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
