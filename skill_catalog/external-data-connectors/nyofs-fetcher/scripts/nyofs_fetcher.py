#!/usr/bin/env python3
"""Plan, fetch, inspect, and extract NOAA NYOFS POM data from public AWS."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence
from urllib.parse import quote

BUCKET = "noaa-nos-ofs-pds"
S3_ENDPOINT = f"https://{BUCKET}.s3.amazonaws.com"
SCHEMA_VERSION = "nyofs_request_v1"
COMPACT_SCHEMA_VERSION = "nyofs_compact_fields_v1"
CYCLE_HOURS = {5, 11, 17, 23}
DEFAULT_VARIABLES = ["zeta", "u", "v", "air_u", "air_v"]
DEFAULT_VIEWS: list[str | int] = ["surface", "bottom", "depth_average"]
UTC = timezone.utc

_CURRENT_RE = re.compile(
    r"^(?P<model>nyofs(?:_fg)?)\.t(?P<hour>\d{2})z\."
    r"(?P<date>\d{8})\.(?P<product>fields|stations)\."
    r"(?P<guidance>nowcast|forecast)\.nc$",
    re.IGNORECASE,
)
_LEGACY_RE = re.compile(
    r"^(?:nos\.)?(?P<model>nyofs(?:_fg)?)\."
    r"(?P<product>fields|stations)\.(?P<guidance>nowcast|forecast)\."
    r"(?P<date>\d{8})\.t(?P<hour>\d{2})z\.nc$",
    re.IGNORECASE,
)


def _requests_module():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("requests is required for public HTTPS access") from exc
    return requests


def _netcdf_modules():
    try:
        import netCDF4
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("netCDF4 and numpy are required for NetCDF processing") from exc
    return netCDF4, np


def _parse_utc(value: Any, name: str = "timestamp") -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty ISO-8601 string")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include an explicit UTC offset")
    return parsed.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _timestamp(value: datetime) -> float:
    return value.astimezone(UTC).timestamp()


def _json_clean(value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_clean(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _json_clean(value.item())
        if isinstance(value, np.ndarray):
            return _json_clean(value.tolist())
    except ImportError:
        pass
    return value


def write_json_atomic(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_clean(value), indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _view_suffix(view: str | int) -> str:
    return f"sigma_{view}" if isinstance(view, int) else str(view)


def load_request(path: str | Path) -> dict[str, Any]:
    mapping = _read_json(path)
    if not isinstance(mapping, Mapping):
        raise ValueError("request root must be a JSON object")
    return validate_request(mapping)


def validate_request(mapping: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "start_utc",
        "end_utc_exclusive",
        "grid",
        "product",
        "guidance",
        "run_cycle_utc",
        "variables",
        "vertical_views",
        "missing_policy",
        "cache_policy",
        "max_workers",
    }
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown request properties: {', '.join(unknown)}")
    if mapping.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    start = _parse_utc(mapping.get("start_utc"), "start_utc")
    end = _parse_utc(mapping.get("end_utc_exclusive"), "end_utc_exclusive")
    if end <= start:
        raise ValueError("end_utc_exclusive must be later than start_utc")
    grid = mapping.get("grid", "coarse")
    if grid not in {"coarse", "fine", "both"}:
        raise ValueError("grid must be coarse, fine, or both")
    product = mapping.get("product")
    if product not in {"fields", "stations"}:
        raise ValueError("product must be fields or stations")
    guidance = mapping.get("guidance")
    if guidance not in {"nowcast", "forecast"}:
        raise ValueError("guidance must be nowcast or forecast")
    run_cycle: datetime | None = None
    if guidance == "forecast":
        if "run_cycle_utc" not in mapping:
            raise ValueError("run_cycle_utc is required for forecast requests")
        run_cycle = _parse_utc(mapping["run_cycle_utc"], "run_cycle_utc")
        if run_cycle.minute or run_cycle.second or run_cycle.microsecond or run_cycle.hour not in CYCLE_HOURS:
            raise ValueError("run_cycle_utc must be exactly 05, 11, 17, or 23 UTC")
    elif "run_cycle_utc" in mapping:
        raise ValueError("run_cycle_utc is permitted only for forecast requests")

    if product == "stations":
        if "variables" in mapping or "vertical_views" in mapping:
            raise ValueError("stations is passthrough-only; variables and vertical_views are not allowed")
        variables: list[str] = []
        views: list[str | int] = []
    else:
        raw_variables = mapping.get("variables", DEFAULT_VARIABLES)
        if (
            not isinstance(raw_variables, list)
            or not raw_variables
            or any(not isinstance(v, str) or not v.strip() for v in raw_variables)
        ):
            raise ValueError("variables must be a non-empty array of variable names")
        variables = [v.strip() for v in raw_variables]
        if len(set(variables)) != len(variables):
            raise ValueError("variables must be unique")
        if ("u" in variables) != ("v" in variables):
            raise ValueError("u and v must be requested together so velocity pairing can be verified")
        raw_views = mapping.get("vertical_views", DEFAULT_VIEWS)
        if not isinstance(raw_views, list) or not raw_views:
            raise ValueError("vertical_views must be a non-empty array")
        views = []
        valid_named = {"surface", "near_surface", "bottom", "depth_average"}
        for view in raw_views:
            if isinstance(view, bool) or not isinstance(view, (str, int)):
                raise ValueError("vertical views must be named strings or non-negative sigma indices")
            if isinstance(view, int) and view < 0:
                raise ValueError("explicit sigma indices must be non-negative")
            if isinstance(view, str) and view not in valid_named:
                raise ValueError(f"unsupported vertical view: {view!r}")
            if view in views:
                raise ValueError("vertical_views must be unique")
            views.append(view)

    missing_policy = mapping.get("missing_policy", "error")
    if missing_policy not in {"error", "skip"}:
        raise ValueError("missing_policy must be error or skip")
    cache_policy = mapping.get("cache_policy", "keep")
    if cache_policy not in {"keep", "delete_after_extract"}:
        raise ValueError("cache_policy must be keep or delete_after_extract")
    max_workers = mapping.get("max_workers", 4)
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")

    normalized: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "start_utc": _iso(start),
        "end_utc_exclusive": _iso(end),
        "grid": grid,
        "product": product,
        "guidance": guidance,
        "missing_policy": missing_policy,
        "cache_policy": cache_policy,
        "max_workers": max_workers,
    }
    if run_cycle is not None:
        normalized["run_cycle_utc"] = _iso(run_cycle)
    if product == "fields":
        normalized["variables"] = variables
        normalized["vertical_views"] = views
    return normalized


def _layout_for_key(key: str) -> str:
    if re.search(r"/\d{4}/\d{2}/\d{2}/", key):
        return "daily"
    if re.search(r"/\d{6}/", key):
        return "monthly"
    return "unknown"


def parse_object_key(key: str) -> dict[str, Any] | None:
    name = Path(key).name
    match = _CURRENT_RE.match(name)
    legacy = False
    if match is None:
        match = _LEGACY_RE.match(name)
        legacy = match is not None
    if match is None:
        return None
    values = match.groupdict()
    run = datetime.strptime(values["date"] + values["hour"], "%Y%m%d%H").replace(tzinfo=UTC)
    model = values["model"].lower()
    grid = "fine" if model.endswith("_fg") else "coarse"
    product = values["product"].lower()
    if legacy:
        guidance = values["guidance"].lower()
        lead = None
        valid = None
        aggregate = True
        if guidance == "nowcast":
            hours_back = 5 if product == "fields" else 6
            expected_start = run - timedelta(hours=hours_back)
            expected_end = run + (timedelta(hours=1) if product == "fields" else timedelta(minutes=6))
        else:
            if product == "fields":
                expected_start = run + timedelta(hours=1)
                expected_end = run + timedelta(hours=55)
            else:
                expected_start = run
                expected_end = run + timedelta(hours=54, minutes=6)
    else:
        guidance = values["guidance"].lower()
        lead = None
        valid = None
        aggregate = True
        if guidance == "nowcast":
            hours_back = 5 if product == "fields" else 6
            expected_start = run - timedelta(hours=hours_back)
            expected_end = run + (timedelta(hours=1) if product == "fields" else timedelta(minutes=6))
        else:
            if product == "fields":
                expected_start = run + timedelta(hours=1)
                expected_end = run + timedelta(hours=55)
            else:
                expected_start = run
                expected_end = run + timedelta(hours=54, minutes=6)
    return {
        "key": key,
        "name": name,
        "layout": _layout_for_key(key),
        "naming": "legacy_aggregate" if legacy else "current_aggregate",
        "aggregate": aggregate,
        "grid": grid,
        "model": model,
        "product": product,
        "guidance": guidance,
        "run_time": _iso(run),
        "cycle_hour": run.hour,
        "lead": lead,
        "valid_time": _iso(valid),
        "expected_start_utc": _iso(expected_start),
        "expected_end_utc_exclusive": _iso(expected_end),
    }


def _xml_text(node: ET.Element, name: str) -> str | None:
    for child in node:
        if child.tag.rsplit("}", 1)[-1] == name:
            return child.text
    return None


def list_s3_objects(
    prefix: str,
    *,
    session: Any | None = None,
    endpoint: str = S3_ENDPOINT,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    requests = _requests_module()
    client = session or requests.Session()
    token: str | None = None
    objects: list[dict[str, Any]] = []
    while True:
        params: dict[str, Any] = {"list-type": "2", "prefix": prefix, "max-keys": 1000}
        if token:
            params["continuation-token"] = token
        response = client.get(endpoint + "/", params=params, timeout=timeout)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for node in root.iter():
            if node.tag.rsplit("}", 1)[-1] != "Contents":
                continue
            key = _xml_text(node, "Key")
            size = _xml_text(node, "Size")
            if not key or size is None:
                continue
            objects.append(
                {
                    "key": key,
                    "size": int(size),
                    "etag": (_xml_text(node, "ETag") or "").strip('"'),
                    "last_modified": _xml_text(node, "LastModified"),
                    "storage_class": _xml_text(node, "StorageClass"),
                    "url": endpoint + "/" + quote(key, safe="/"),
                }
            )
        truncated = next(
            (node.text for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "IsTruncated"),
            "false",
        )
        if str(truncated).lower() != "true":
            break
        token = next(
            (node.text for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "NextContinuationToken"),
            None,
        )
        if not token:
            raise RuntimeError("S3 listing is truncated but has no continuation token")
    return objects


def _days(start: datetime, end: datetime) -> Iterable[datetime]:
    cursor = datetime(start.year, start.month, start.day, tzinfo=UTC)
    final = datetime(end.year, end.month, end.day, tzinfo=UTC)
    while cursor <= final:
        yield cursor
        cursor += timedelta(days=1)


def _month_starts(start: datetime, end: datetime) -> Iterable[datetime]:
    cursor = datetime(start.year, start.month, 1, tzinfo=UTC)
    final = datetime(end.year, end.month, 1, tzinfo=UTC)
    while cursor <= final:
        yield cursor
        cursor = datetime(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1, tzinfo=UTC)


def discovery_prefixes(request: Mapping[str, Any]) -> list[str]:
    if request["guidance"] == "forecast":
        center = _parse_utc(request["run_cycle_utc"])
        first = center - timedelta(days=1)
        last = center + timedelta(days=1)
    else:
        first = _parse_utc(request["start_utc"]) - timedelta(days=1)
        last = _parse_utc(request["end_utc_exclusive"]) + timedelta(days=1)
    daily = [f"nyofs/netcdf/{d:%Y/%m/%d}/" for d in _days(first, last)]
    monthly = [f"nyofs/netcdf/{m:%Y%m}/" for m in _month_starts(first, last)]
    return daily + monthly


def discover_objects(
    request: Mapping[str, Any],
    *,
    session: Any | None = None,
    endpoint: str = S3_ENDPOINT,
) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for prefix in discovery_prefixes(request):
        for raw in list_s3_objects(prefix, session=session, endpoint=endpoint):
            parsed = parse_object_key(raw["key"])
            if parsed is None:
                continue
            item = {**raw, **parsed}
            if item["product"] != request["product"] or item["guidance"] != request["guidance"]:
                continue
            if request["grid"] != "both" and item["grid"] != request["grid"]:
                continue
            found[item["key"]] = item
    return sorted(found.values(), key=lambda x: (x["run_time"], x["grid"], x["key"]))


def _preference(item: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        1 if item.get("naming") == "current_aggregate" else 0,
        1 if item.get("layout") == "daily" else 0,
        str(item.get("key", "")),
    )


def _overlaps(item: Mapping[str, Any], start: datetime, end: datetime) -> bool:
    item_start = _parse_utc(item["expected_start_utc"])
    item_end = _parse_utc(item["expected_end_utc_exclusive"])
    return item_start < end and item_end > start


def _expected_times(start: datetime, end: datetime, step_seconds: int) -> list[datetime]:
    first_epoch = math.ceil(start.timestamp() / step_seconds) * step_seconds
    cursor = datetime.fromtimestamp(first_epoch, tz=UTC)
    result: list[datetime] = []
    while cursor < end:
        result.append(cursor)
        cursor += timedelta(seconds=step_seconds)
    return result


def _nominal_item_times(item: Mapping[str, Any], product: str) -> list[datetime]:
    start = _parse_utc(item["expected_start_utc"])
    end = _parse_utc(item["expected_end_utc_exclusive"])
    step = 3600 if product == "fields" else 360
    return _expected_times(start, end, step)


def select_objects(request: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    start = _parse_utc(request["start_utc"])
    end = _parse_utc(request["end_utc_exclusive"])
    candidates: list[dict[str, Any]] = []
    for raw in objects:
        item = dict(raw)
        if item.get("product") != request["product"] or item.get("guidance") != request["guidance"]:
            continue
        if request["grid"] != "both" and item.get("grid") != request["grid"]:
            continue
        if request["guidance"] == "forecast":
            if item.get("run_time") != request["run_cycle_utc"]:
                continue
        if _overlaps(item, start, end):
            candidates.append(item)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in candidates:
        if item.get("aggregate"):
            identity = (item["grid"], item["product"], item["guidance"], item["run_time"])
        else:
            identity = (item["grid"], item["product"], item["guidance"], item.get("valid_time"))
        grouped.setdefault(identity, []).append(item)
    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for group in grouped.values():
        winner = max(group, key=_preference)
        selected.append(winner)
        duplicates.extend(item for item in group if item is not winner)
    selected.sort(key=lambda x: (x["grid"], x["run_time"], x["key"]))

    expected = _expected_times(start, end, 3600 if request["product"] == "fields" else 360)
    requested_grids = ["coarse", "fine"] if request["grid"] == "both" else [request["grid"]]
    coverage: dict[tuple[str, datetime], list[str]] = {}
    for item in selected:
        for stamp in _nominal_item_times(item, request["product"]):
            if start <= stamp < end:
                coverage.setdefault((item["grid"], stamp), []).append(item["key"])
    missing = [
        {"grid": grid, "time_utc": _iso(stamp)}
        for grid in requested_grids
        for stamp in expected
        if (grid, stamp) not in coverage
    ]
    duplicate_times = {
        (grid, stamp): keys
        for (grid, stamp), keys in coverage.items()
        if len(keys) > 1
    }
    if missing and request["missing_policy"] == "error":
        preview = ", ".join(f"{item['grid']} {item['time_utc']}" for item in missing[:8])
        raise RuntimeError(f"source inventory is missing {len(missing)} required timestamps: {preview}")
    if not selected:
        raise RuntimeError("source inventory did not contain any matching NYOFS objects")
    return {
        "selected": selected,
        "duplicate_objects": duplicates,
        "missing_times": missing,
        "duplicate_times": [
            {"grid": grid, "time_utc": _iso(stamp), "keys": keys}
            for (grid, stamp), keys in sorted(duplicate_times.items())
        ],
        "nominal_time_count": len(expected) * len(requested_grids),
        "nominal_time_count_by_grid": {grid: len(expected) for grid in requested_grids},
        "coverage_note": "filename-derived forecast spans are nominal; downloaded NetCDF time coordinates are authoritative",
    }


def _disk_free(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return int(shutil.disk_usage(path).free)


def inventory_request(
    request: Mapping[str, Any] | str | Path,
    run_dir: str | Path,
    *,
    objects: Sequence[Mapping[str, Any]] | None = None,
    session: Any | None = None,
    endpoint: str = S3_ENDPOINT,
) -> dict[str, Any]:
    normalized = load_request(request) if isinstance(request, (str, Path)) else validate_request(request)
    discovered = [dict(item) for item in (objects if objects is not None else discover_objects(normalized, session=session, endpoint=endpoint))]
    report = {
        "schema_version": "nyofs_inventory_v1",
        "created_utc": _iso(datetime.now(UTC)),
        "request": normalized,
        "prefixes": discovery_prefixes(normalized),
        "object_count": len(discovered),
        "objects": discovered,
        "source": {"bucket": BUCKET, "endpoint": endpoint, "access": "anonymous_https_listobjectsv2"},
    }
    write_json_atomic(Path(run_dir) / "inventory.json", report)
    return report


def plan_request(
    request: Mapping[str, Any] | str | Path,
    run_dir: str | Path,
    *,
    output: str | Path | None = None,
    objects: Sequence[Mapping[str, Any]] | None = None,
    session: Any | None = None,
    endpoint: str = S3_ENDPOINT,
) -> dict[str, Any]:
    normalized = load_request(request) if isinstance(request, (str, Path)) else validate_request(request)
    discovered = [dict(item) for item in (objects if objects is not None else discover_objects(normalized, session=session, endpoint=endpoint))]
    selection = select_objects(normalized, discovered)
    incomplete = [item["key"] for item in selection["selected"] if not isinstance(item.get("size"), int) or item["size"] < 0]
    total = sum(int(item.get("size", 0)) for item in selection["selected"])
    run_path = Path(run_dir)
    free = _disk_free(run_path)
    required = 4 * total
    if incomplete:
        route = "review"
        decision = "estimate_incomplete"
    elif free > required:
        route = "local"
        decision = "local_free_bytes_exceeds_four_times_exact_request_bytes"
    else:
        route = "kestrel"
        decision = "local_free_space_gate_failed"
    report = {
        "schema_version": "nyofs_download_estimate_v1",
        "created_utc": _iso(datetime.now(UTC)),
        "request": normalized,
        "source": {"bucket": BUCKET, "endpoint": endpoint, "access": "anonymous_https_listobjectsv2"},
        "objects": selection["selected"],
        "object_count": len(selection["selected"]),
        "total_bytes": total,
        "total_gib": total / 1024**3,
        "incomplete_size_keys": incomplete,
        "missing_times": selection["missing_times"],
        "duplicate_times": selection["duplicate_times"],
        "duplicate_objects": selection["duplicate_objects"],
        "nominal_time_count": selection["nominal_time_count"],
        "nominal_time_count_by_grid": selection["nominal_time_count_by_grid"],
        "coverage_note": selection["coverage_note"],
        "local_free_bytes": free,
        "required_free_bytes": required,
        "routing_decision": route,
        "routing_reason": decision,
        "kestrel_stage_hint": "/scratch/yhuang168/oma_external_data_connectors/nyofs-fetcher/<run-id>",
    }
    write_json_atomic(output or run_path / "download_estimate.json", report)
    return report


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _clean_etag(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _download_sidecar(destination: Path) -> Path:
    return destination.with_name(destination.name + ".download.json")


def _partial_sidecar(destination: Path) -> Path:
    return destination.with_name(destination.name + ".part.json")


def _destination_for_key(run_dir: Path, key: str) -> Path:
    prefix = "nyofs/netcdf/"
    relative = key[len(prefix) :] if key.startswith(prefix) else Path(key).name
    parts = Path(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts) or Path(relative).is_absolute():
        raise ValueError(f"unsafe S3 key path: {key!r}")
    destination = run_dir / "cache" / "raw"
    for part in parts:
        destination /= part
    return destination


def _cache_result(item: Mapping[str, Any], destination: Path) -> dict[str, Any] | None:
    sidecar = _download_sidecar(destination)
    if not destination.is_file() or not sidecar.is_file():
        return None
    try:
        metadata = _read_json(sidecar)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    expected_size = int(item.get("size", -1))
    if destination.stat().st_size != expected_size or int(metadata.get("size", -2)) != expected_size:
        return None
    expected_etag = _clean_etag(item.get("etag"))
    if expected_etag and _clean_etag(metadata.get("etag")) != expected_etag:
        return None
    expected_hash = str(metadata.get("sha256", ""))
    if len(expected_hash) != 64:
        return None
    actual_hash = _sha256(destination)
    if actual_hash != expected_hash:
        return None
    return {
        "key": item["key"],
        "url": item["url"],
        "local_path": str(destination.resolve()),
        "status": "cache_hit",
        "size": expected_size,
        "etag": expected_etag,
        "sha256": actual_hash,
        "resumed": False,
        "resumed_from_bytes": 0,
        "retry_count": 0,
        "source": dict(item),
    }


def download_object(
    item: Mapping[str, Any],
    destination: str | Path,
    *,
    session: Any | None = None,
    timeout: float = 120.0,
    max_attempts: int = 4,
    chunk_size: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    requests = _requests_module()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = _cache_result(item, destination)
    if cached is not None:
        return cached
    expected_size = int(item.get("size", -1))
    if expected_size < 0:
        raise RuntimeError(f"object has no exact non-negative size: {item.get('key')}")
    expected_etag = _clean_etag(item.get("etag"))
    partial = destination.with_name(destination.name + ".part")
    partial_metadata_path = _partial_sidecar(destination)
    partial_metadata = {
        "schema_version": "nyofs_partial_object_v1",
        "key": item["key"],
        "url": item["url"],
        "size": expected_size,
        "etag": expected_etag,
    }
    if partial.exists():
        try:
            existing_partial_metadata = _read_json(partial_metadata_path)
        except Exception:
            existing_partial_metadata = None
        if not isinstance(existing_partial_metadata, Mapping) or any(
            existing_partial_metadata.get(name) != partial_metadata[name]
            for name in ("key", "url", "size", "etag")
        ):
            partial.unlink(missing_ok=True)
            partial_metadata_path.unlink(missing_ok=True)
    if not partial.exists():
        write_json_atomic(partial_metadata_path, partial_metadata)
    resumed_from = partial.stat().st_size if partial.exists() else 0
    if resumed_from > expected_size:
        partial.unlink()
        resumed_from = 0
    if resumed_from == expected_size:
        digest = _sha256(partial)
        os.replace(partial, destination)
        partial_metadata_path.unlink(missing_ok=True)
        metadata = {
            "schema_version": "nyofs_cached_object_v1",
            "key": item["key"],
            "url": item["url"],
            "size": expected_size,
            "etag": expected_etag,
            "etag_is_multipart": "-" in expected_etag,
            "last_modified": item.get("last_modified"),
            "sha256": digest,
            "completed_utc": _iso(datetime.now(UTC)),
        }
        write_json_atomic(_download_sidecar(destination), metadata)
        return {
            "key": item["key"],
            "url": item["url"],
            "local_path": str(destination.resolve()),
            "status": "downloaded",
            "size": expected_size,
            "etag": expected_etag,
            "sha256": digest,
            "resumed": True,
            "resumed_from_bytes": resumed_from,
            "retry_count": 0,
            "source": dict(item),
        }
    client = session or requests.Session()
    errors: list[str] = []
    for attempt in range(max_attempts):
        current = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={current}-"} if current else {}
        try:
            response = client.get(item["url"], headers=headers, stream=True, timeout=timeout)
            response.raise_for_status()
            if current and response.status_code != 206:
                partial.unlink(missing_ok=True)
                current = 0
            mode = "ab" if current and response.status_code == 206 else "wb"
            response_etag = _clean_etag(response.headers.get("ETag"))
            if expected_etag and response_etag and response_etag != expected_etag:
                raise RuntimeError(f"ETag changed during transfer: {response_etag} != {expected_etag}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                expected_response = expected_size - current if mode == "ab" else expected_size
                if int(content_length) != expected_response:
                    raise RuntimeError(
                        f"Content-Length mismatch: {content_length} != {expected_response}"
                    )
            with partial.open(mode) as stream:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        stream.write(chunk)
            actual_size = partial.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(f"downloaded size mismatch: {actual_size} != {expected_size}")
            digest = _sha256(partial)
            os.replace(partial, destination)
            partial_metadata_path.unlink(missing_ok=True)
            metadata = {
                "schema_version": "nyofs_cached_object_v1",
                "key": item["key"],
                "url": item["url"],
                "size": expected_size,
                "etag": expected_etag,
                "etag_is_multipart": "-" in expected_etag,
                "last_modified": item.get("last_modified"),
                "sha256": digest,
                "completed_utc": _iso(datetime.now(UTC)),
            }
            write_json_atomic(_download_sidecar(destination), metadata)
            return {
                "key": item["key"],
                "url": item["url"],
                "local_path": str(destination.resolve()),
                "status": "downloaded",
                "size": expected_size,
                "etag": expected_etag,
                "sha256": digest,
                "resumed": resumed_from > 0,
                "resumed_from_bytes": resumed_from,
                "retry_count": attempt,
                "source": dict(item),
            }
        except Exception as exc:  # preserve .part for a later retry/run
            errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            if attempt + 1 < max_attempts:
                time.sleep(min(2**attempt, 4))
    return {
        "key": item.get("key"),
        "url": item.get("url"),
        "local_path": str(destination.resolve()),
        "status": "failed",
        "size": expected_size,
        "etag": expected_etag,
        "resumed": resumed_from > 0,
        "resumed_from_bytes": resumed_from,
        "retry_count": max(0, max_attempts - 1),
        "errors": errors,
        "source": dict(item),
    }


def fetch_request(
    request: Mapping[str, Any] | str | Path,
    run_dir: str | Path,
    *,
    objects: Sequence[Mapping[str, Any]] | None = None,
    session: Any | None = None,
    endpoint: str = S3_ENDPOINT,
) -> dict[str, Any]:
    normalized = load_request(request) if isinstance(request, (str, Path)) else validate_request(request)
    run_path = Path(run_dir)
    estimate = plan_request(normalized, run_path, objects=objects, session=session, endpoint=endpoint)
    if estimate["routing_decision"] != "local":
        raise RuntimeError(
            f"local fetch is not approved by estimate: {estimate['routing_decision']} "
            f"({estimate['routing_reason']})"
        )
    outcomes: list[dict[str, Any]] = []

    def transfer(item: Mapping[str, Any]) -> dict[str, Any]:
        return download_object(item, _destination_for_key(run_path, str(item["key"])), session=session)

    with concurrent.futures.ThreadPoolExecutor(max_workers=normalized["max_workers"]) as pool:
        futures = [pool.submit(transfer, item) for item in estimate["objects"]]
        for future in concurrent.futures.as_completed(futures):
            outcomes.append(future.result())
    outcomes.sort(key=lambda x: str(x.get("key")))
    failures = [item for item in outcomes if item["status"] == "failed"]
    manifest = {
        "schema_version": "nyofs_fetch_manifest_v1",
        "created_utc": _iso(datetime.now(UTC)),
        "request": normalized,
        "estimate_path": str((run_path / "download_estimate.json").resolve()),
        "outcomes": outcomes,
        "counts": {
            "objects": len(outcomes),
            "downloaded": sum(item["status"] == "downloaded" for item in outcomes),
            "cache_hits": sum(item["status"] == "cache_hit" for item in outcomes),
            "failed": len(failures),
            "resumed": sum(bool(item.get("resumed")) for item in outcomes),
        },
        "source_provenance": {"bucket": BUCKET, "endpoint": endpoint, "access": "anonymous_https"},
    }
    write_json_atomic(run_path / "fetch_manifest.json", manifest)
    if failures:
        raise RuntimeError(f"{len(failures)} NYOFS object transfers failed; inspect fetch_manifest.json")
    return manifest


def _as_utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    return datetime(
        int(value.year),
        int(value.month),
        int(value.day),
        int(value.hour),
        int(value.minute),
        int(value.second),
        int(getattr(value, "microsecond", 0)),
        tzinfo=UTC,
    )


@dataclass(frozen=True)
class TimeRecord:
    original: datetime
    normalized: datetime
    adjustment_seconds: float
    raw_value: float


def normalize_time(value: datetime, cadence_seconds: int, tolerance_seconds: float = 60.0) -> tuple[datetime, float]:
    epoch = value.timestamp()
    nominal_epoch = round(epoch / cadence_seconds) * cadence_seconds
    nominal = datetime.fromtimestamp(nominal_epoch, tz=UTC)
    adjustment = epoch - nominal_epoch
    return (nominal if abs(adjustment) <= tolerance_seconds else value, adjustment if abs(adjustment) <= tolerance_seconds else 0.0)


def decode_times(ds: Any, product: str) -> list[TimeRecord]:
    netCDF4, np = _netcdf_modules()
    if "time" not in ds.variables:
        raise ValueError("NetCDF file has no time variable")
    variable = ds.variables["time"]
    units = getattr(variable, "units", None)
    if not units:
        raise ValueError("NetCDF time variable has no units")
    calendar = getattr(variable, "calendar", "standard")
    raw = np.asarray(variable[:], dtype=float).reshape(-1)
    decoded = netCDF4.num2date(
        raw,
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=False,
    )
    cadence = 3600 if product == "fields" else 360
    result: list[TimeRecord] = []
    for raw_value, decoded_value in zip(raw, decoded):
        original = _as_utc_datetime(decoded_value)
        normalized, adjustment = normalize_time(original, cadence)
        result.append(TimeRecord(original, normalized, adjustment, float(raw_value)))
    return result


def _attribute_dict(variable: Any) -> dict[str, Any]:
    return {name: _json_clean(variable.getncattr(name)) for name in variable.ncattrs()}


def _array_digest(value: Any) -> str:
    _, np = _netcdf_modules()
    array = np.ma.asarray(value)
    data = np.asarray(array.filled(np.nan), dtype=np.float64)
    return hashlib.sha256(data.tobytes(order="C")).hexdigest()


def inspect_file(path: str | Path, product: str | None = None) -> dict[str, Any]:
    netCDF4, np = _netcdf_modules()
    source = Path(path)
    parsed = parse_object_key(source.name)
    resolved_product = product or (parsed["product"] if parsed else "fields")
    with netCDF4.Dataset(source) as ds:
        times = decode_times(ds, resolved_product)
        variables = {
            name: {
                "dtype": str(variable.dtype),
                "dimensions": list(variable.dimensions),
                "shape": list(variable.shape),
                "attributes": _attribute_dict(variable),
            }
            for name, variable in ds.variables.items()
        }
        geometry: dict[str, Any] = {}
        for name in ("lon", "lat", "mask", "depth", "sigma"):
            if name in ds.variables:
                values = np.ma.asarray(ds.variables[name][:])
                finite = np.asarray(values.filled(np.nan), dtype=float)
                geometry[name] = {
                    "shape": list(values.shape),
                    "digest": _array_digest(values),
                    "finite_count": int(np.isfinite(finite).sum()),
                    "minimum": float(np.nanmin(finite)) if np.isfinite(finite).any() else None,
                    "maximum": float(np.nanmax(finite)) if np.isfinite(finite).any() else None,
                }
                if name == "mask":
                    geometry[name]["unique_values"] = [float(x) for x in np.unique(finite[np.isfinite(finite)])]
        return {
            "path": str(source.resolve()),
            "size": source.stat().st_size,
            "data_model": ds.data_model,
            "dimensions": {name: len(dimension) for name, dimension in ds.dimensions.items()},
            "global_attributes": {name: _json_clean(ds.getncattr(name)) for name in ds.ncattrs()},
            "variables": variables,
            "times": [
                {
                    "raw_value": record.raw_value,
                    "original_utc": _iso(record.original),
                    "normalized_utc": _iso(record.normalized),
                    "original_minus_normalized_seconds": record.adjustment_seconds,
                }
                for record in times
            ],
            "geometry": geometry,
        }


def _manifest_paths(run_dir: Path) -> list[Path]:
    manifest_path = run_dir / "fetch_manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = _read_json(manifest_path)
    paths: list[Path] = []
    for outcome in manifest.get("outcomes", []):
        if outcome.get("status") not in {"downloaded", "cache_hit"}:
            continue
        candidate = Path(outcome["local_path"])
        if candidate.is_file():
            paths.append(candidate)
    return paths


def inspect_request(request: Mapping[str, Any] | str | Path, run_dir: str | Path) -> dict[str, Any]:
    normalized = load_request(request) if isinstance(request, (str, Path)) else validate_request(request)
    run_path = Path(run_dir)
    paths = _manifest_paths(run_path)
    if not paths:
        raise RuntimeError("fetch_manifest.json contains no available downloaded files")
    files = [inspect_file(path, normalized["product"]) for path in paths]
    report = {
        "schema_version": "nyofs_inspection_v1",
        "created_utc": _iso(datetime.now(UTC)),
        "request": normalized,
        "file_count": len(files),
        "files": files,
    }
    write_json_atomic(run_path / "inspection.json", report)
    return report


def sigma_trapezoid_weights(sigma: Any):
    """Return normalized trapezoidal point weights in the source sigma order."""
    _, np = _netcdf_modules()
    values = np.asarray(sigma, dtype=float).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("sigma must contain at least two finite points")
    order = np.argsort(values)
    sorted_values = values[order]
    differences = np.diff(sorted_values)
    if np.any(differences <= 0):
        raise ValueError("sigma points must be unique and monotonic after sorting")
    sorted_weights = np.empty(values.size, dtype=float)
    sorted_weights[0] = differences[0] / 2.0
    sorted_weights[-1] = differences[-1] / 2.0
    if values.size > 2:
        sorted_weights[1:-1] = (sorted_values[2:] - sorted_values[:-2]) / 2.0
    weights = np.empty_like(sorted_weights)
    weights[order] = np.abs(sorted_weights)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("sigma trapezoid weights have a non-positive sum")
    return weights / total


def weighted_vertical_average(data: Any, weights: Any, wet_mask: Any | None = None, axis: int = 1):
    """Average a vertical axis, renormalizing over finite wet layers."""
    _, np = _netcdf_modules()
    values = np.ma.asarray(data, dtype=float).filled(np.nan)
    axis = axis % values.ndim
    layer_weights = np.asarray(weights, dtype=float).reshape(-1)
    if values.shape[axis] != layer_weights.size:
        raise ValueError("vertical data length does not match sigma weights")
    reshape = [1] * values.ndim
    reshape[axis] = layer_weights.size
    broadcast_weights = layer_weights.reshape(reshape)
    valid = np.isfinite(values)
    if wet_mask is not None:
        wet = np.asarray(wet_mask, dtype=bool)
        expected_horizontal = values.shape[:axis] + values.shape[axis + 1 :]
        if wet.shape == expected_horizontal[-wet.ndim :]:
            wet_shape = [1] * values.ndim
            wet_shape[-wet.ndim :] = wet.shape
            valid &= wet.reshape(wet_shape)
        else:
            raise ValueError("wet mask shape is incompatible with vertical field")
    denominator = np.sum(np.where(valid, broadcast_weights, 0.0), axis=axis)
    numerator = np.sum(np.where(valid, values * broadcast_weights, 0.0), axis=axis)
    result = np.full(denominator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _geometry(ds: Any) -> dict[str, Any]:
    _, np = _netcdf_modules()
    required = ("lon", "lat", "mask", "depth", "sigma")
    missing = [name for name in required if name not in ds.variables]
    if missing:
        raise ValueError(f"POM fields file is missing geometry variables: {', '.join(missing)}")
    lon_var = ds.variables["lon"]
    if len(lon_var.dimensions) != 2:
        raise ValueError("lon must be a two-dimensional curvilinear coordinate")
    ydim, xdim = lon_var.dimensions
    arrays = {name: np.ma.asarray(ds.variables[name][:]).filled(np.nan) for name in required}
    shape = arrays["lon"].shape
    if arrays["lat"].shape != shape or arrays["mask"].shape != shape or arrays["depth"].shape != shape:
        raise ValueError("lon, lat, mask, and depth must have identical two-dimensional shapes")
    mask_finite = arrays["mask"][np.isfinite(arrays["mask"])]
    if not set(float(x) for x in np.unique(mask_finite)).issubset({0.0, 1.0}):
        raise ValueError("NYOFS mask contains values other than 0 and 1")
    wet = np.isclose(arrays["mask"], 1.0)
    if not wet.any():
        raise ValueError("NYOFS mask has no wet cells")
    if not np.isfinite(arrays["lon"][wet]).all() or not np.isfinite(arrays["lat"][wet]).all():
        raise ValueError("NYOFS wet cells contain non-finite coordinates")
    sigma = np.asarray(arrays["sigma"], dtype=float).reshape(-1)
    sigma_trapezoid_weights(sigma)
    return {
        "lon": np.asarray(arrays["lon"], dtype=float),
        "lat": np.asarray(arrays["lat"], dtype=float),
        "mask": np.asarray(arrays["mask"], dtype=float),
        "wet": wet,
        "depth": np.asarray(arrays["depth"], dtype=float),
        "sigma": sigma,
        "ydim": ydim,
        "xdim": xdim,
    }


def _assert_geometry(reference: Mapping[str, Any], candidate: Mapping[str, Any], path: Path) -> None:
    _, np = _netcdf_modules()
    for name in ("lon", "lat", "mask", "depth", "sigma"):
        left = np.asarray(reference[name])
        right = np.asarray(candidate[name])
        if left.shape != right.shape or not np.allclose(left, right, rtol=1e-6, atol=1e-7, equal_nan=True):
            raise RuntimeError(f"NYOFS geometry drift for {name} in {path}")


def _canonical_dynamic_dimensions(variable: Any, geometry: Mapping[str, Any]) -> tuple[str, ...]:
    dimensions = tuple(name for name in variable.dimensions if name != "time")
    ydim, xdim = geometry["ydim"], geometry["xdim"]
    if dimensions == (ydim, xdim):
        return ("y", "x")
    if dimensions == ("sigma", ydim, xdim):
        return ("sigma", "y", "x")
    raise ValueError(
        f"variable {variable.name!r} has unsupported POM dimensions {variable.dimensions}; "
        f"expected time,{ydim},{xdim} with optional sigma"
    )


def _read_dynamic_record(variable: Any, time_index: int, geometry: Mapping[str, Any]):
    _, np = _netcdf_modules()
    if "time" not in variable.dimensions:
        raise ValueError(f"requested source variable {variable.name!r} has no time dimension")
    axis = variable.dimensions.index("time")
    index: list[Any] = [slice(None)] * variable.ndim
    index[axis] = time_index
    values = np.ma.asarray(variable[tuple(index)], dtype=float).filled(np.nan)
    if axis != 0:
        # The time axis is removed; move remaining axes only if a future source changes ordering.
        remaining = list(variable.dimensions)
        remaining.pop(axis)
        target = ["sigma", geometry["ydim"], geometry["xdim"]] if "sigma" in remaining else [geometry["ydim"], geometry["xdim"]]
        values = np.transpose(values, [remaining.index(name) for name in target])
    wet = geometry["wet"]
    if values.ndim == 2:
        values = np.where(wet, values, np.nan)
    elif values.ndim == 3:
        values = np.where(wet[None, :, :], values, np.nan)
    return values


def _field_records(
    request: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
    grid: str,
) -> list[dict[str, Any]]:
    netCDF4, _ = _netcdf_modules()
    start = _parse_utc(request["start_utc"])
    end = _parse_utc(request["end_utc_exclusive"])
    candidates: dict[datetime, list[dict[str, Any]]] = {}
    for outcome in outcomes:
        if outcome.get("status") not in {"downloaded", "cache_hit"}:
            continue
        source = outcome.get("source") or parse_object_key(str(outcome.get("key", ""))) or {}
        if source.get("grid") != grid or source.get("product") != "fields":
            continue
        path = Path(outcome["local_path"])
        with netCDF4.Dataset(path) as ds:
            time_variable = ds.variables.get("time")
            source_time_units = str(getattr(time_variable, "units", ""))
            source_time_calendar = str(getattr(time_variable, "calendar", "standard"))
            for index, record in enumerate(decode_times(ds, "fields")):
                if start <= record.normalized < end:
                    candidates.setdefault(record.normalized, []).append(
                        {
                            "path": path,
                            "time_index": index,
                            "time": record.normalized,
                            "original_time": record.original,
                            "time_adjustment_seconds": record.adjustment_seconds,
                            "raw_time_value": record.raw_value,
                            "source_time_units": source_time_units,
                            "source_time_calendar": source_time_calendar,
                            "run_time": _parse_utc(source["run_time"]),
                            "key": source.get("key", outcome.get("key")),
                        }
                    )
    records: list[dict[str, Any]] = []
    for stamp, group in candidates.items():
        # For a boundary duplicate, the preceding (earlier) cycle owns the terminal record.
        winner = min(group, key=lambda item: (item["run_time"], str(item["key"])))
        records.append(winner)
    records.sort(key=lambda item: item["time"])
    expected = _expected_times(start, end, 3600)
    available = {item["time"] for item in records}
    missing = [stamp for stamp in expected if stamp not in available]
    if missing and request["missing_policy"] == "error":
        raise RuntimeError(f"downloaded {grid} fields are missing {len(missing)} requested hourly records")
    if not records:
        raise RuntimeError(f"downloaded manifest has no {grid} fields in the requested window")
    return records


def _copy_attributes(source: Any, destination: Any) -> None:
    for name in source.ncattrs():
        if name in {"_FillValue", "missing_value", "scale_factor", "add_offset"}:
            continue
        try:
            destination.setncattr(name, source.getncattr(name))
        except (TypeError, ValueError):
            destination.setncattr(name, str(source.getncattr(name)))


def _view_indices(sigma: Any, views: Sequence[str | int]) -> dict[str, int | None]:
    _, np = _netcdf_modules()
    values = np.asarray(sigma, dtype=float).reshape(-1)
    by_surface_distance = np.argsort(np.abs(values))
    result: dict[str, int | None] = {}
    for view in views:
        suffix = _view_suffix(view)
        if view == "surface":
            result[suffix] = int(by_surface_distance[0])
        elif view == "near_surface":
            if values.size < 2:
                raise ValueError("near_surface requires at least two sigma layers")
            result[suffix] = int(by_surface_distance[1])
        elif view == "bottom":
            result[suffix] = int(by_surface_distance[-1])
        elif view == "depth_average":
            result[suffix] = None
        else:
            index = int(view)
            if index >= values.size:
                raise ValueError(f"explicit sigma index {index} exceeds available layer count {values.size}")
            result[suffix] = index
    return result


def _validate_vector_metadata(u_var: Any, v_var: Any) -> None:
    if u_var.dimensions != v_var.dimensions:
        raise RuntimeError("u and v dimensions differ; grid staggering is unsupported")
    u_standard = str(getattr(u_var, "standard_name", "")).lower()
    v_standard = str(getattr(v_var, "standard_name", "")).lower()
    u_long = str(getattr(u_var, "long_name", "")).lower()
    v_long = str(getattr(v_var, "long_name", "")).lower()
    if u_standard and u_standard != "eastward_sea_water_velocity":
        raise RuntimeError(f"u is not advertised as earthward east velocity: {u_standard}")
    if v_standard and v_standard != "northward_sea_water_velocity":
        raise RuntimeError(f"v is not advertised as earthward north velocity: {v_standard}")
    if not u_standard and "eastward" not in u_long:
        raise RuntimeError("u lacks eastward velocity metadata; grid-relative vectors are unsupported")
    if not v_standard and "northward" not in v_long:
        raise RuntimeError("v lacks northward velocity metadata; grid-relative vectors are unsupported")


def _write_char_rows(variable: Any, strings: Sequence[str], width: int) -> None:
    _, np = _netcdf_modules()
    encoded = np.full((len(strings), width), b" ", dtype="S1")
    for row, text in enumerate(strings):
        raw = text.encode("utf-8")[:width]
        if raw:
            encoded[row, : len(raw)] = np.frombuffer(raw, dtype="S1")
    variable[:] = encoded


def _extract_grid(
    request: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
    grid: str,
    run_dir: Path,
) -> dict[str, Any]:
    netCDF4, np = _netcdf_modules()
    records = _field_records(request, outcomes, grid)
    unique_paths = sorted({Path(item["path"]) for item in records}, key=str)
    datasets: dict[Path, Any] = {}
    try:
        for path in unique_paths:
            datasets[path] = netCDF4.Dataset(path)
        first_ds = datasets[Path(records[0]["path"])]
        reference_geometry = _geometry(first_ds)
        requested = list(request["variables"])
        missing = [name for name in requested if name not in first_ds.variables]
        if missing:
            raise RuntimeError(f"requested variables are absent from {grid} fields: {', '.join(missing)}")
        source_schema: dict[str, dict[str, Any]] = {}
        for name in requested:
            variable = first_ds.variables[name]
            dimensions = _canonical_dynamic_dimensions(variable, reference_geometry)
            source_schema[name] = {
                "dimensions": dimensions,
                "source_dimensions": tuple(variable.dimensions),
                "dtype": str(variable.dtype),
                "attributes": _attribute_dict(variable),
            }
        if "u" in requested and "v" in requested:
            _validate_vector_metadata(first_ds.variables["u"], first_ds.variables["v"])
        for path, ds in datasets.items():
            geometry = _geometry(ds)
            _assert_geometry(reference_geometry, geometry, path)
            for name, schema in source_schema.items():
                if name not in ds.variables:
                    raise RuntimeError(f"schema drift: {name!r} is missing from {path}")
                variable = ds.variables[name]
                if tuple(variable.dimensions) != schema["source_dimensions"] or str(variable.dtype) != schema["dtype"]:
                    raise RuntimeError(f"schema drift for {name!r} in {path}")
            if "u" in requested and "v" in requested:
                _validate_vector_metadata(ds.variables["u"], ds.variables["v"])

        arrays: dict[str, Any] = {}
        for name in requested:
            arrays[name] = np.asarray(
                [
                    _read_dynamic_record(
                        datasets[Path(record["path"])].variables[name],
                        int(record["time_index"]),
                        reference_geometry,
                    )
                    for record in records
                ],
                dtype=np.float32,
            )
        sigma = reference_geometry["sigma"]
        weights = sigma_trapezoid_weights(sigma)
        views = _view_indices(sigma, request["vertical_views"])
        derived: dict[str, Any] = {}
        for name, values in arrays.items():
            if source_schema[name]["dimensions"] != ("sigma", "y", "x"):
                continue
            for suffix, layer_index in views.items():
                if layer_index is None:
                    result = weighted_vertical_average(values, weights, reference_geometry["wet"], axis=1)
                else:
                    result = values[:, int(layer_index), :, :]
                derived[f"{name}_{suffix}"] = np.asarray(result, dtype=np.float32)
        if "u" in arrays and "v" in arrays:
            for suffix in views:
                u_name, v_name = f"u_{suffix}", f"v_{suffix}"
                if u_name in derived and v_name in derived:
                    derived[f"current_speed_{suffix}"] = np.hypot(derived[u_name], derived[v_name]).astype(np.float32)
        if "air_u" in arrays and "air_v" in arrays:
            if arrays["air_u"].shape != arrays["air_v"].shape:
                raise RuntimeError("air_u and air_v dimensions differ")
            derived["wind_speed"] = np.hypot(arrays["air_u"], arrays["air_v"]).astype(np.float32)

        compact_dir = run_dir / "compact"
        compact_dir.mkdir(parents=True, exist_ok=True)
        destination = compact_dir / f"nyofs_{grid}_fields.nc"
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.unlink(missing_ok=True)
        fill_value = np.float32(-99999.0)
        with netCDF4.Dataset(temporary, "w", format="NETCDF4_CLASSIC") as output:
            output.createDimension("time", len(records))
            output.createDimension("sigma", len(sigma))
            output.createDimension("y", reference_geometry["lon"].shape[0])
            output.createDimension("x", reference_geometry["lon"].shape[1])
            source_width = max(1, max(len(str(record["key"]).encode("utf-8")) for record in records))
            output.createDimension("source_key_strlen", source_width)
            time_units_width = max(1, max(len(record["source_time_units"].encode("utf-8")) for record in records))
            calendar_width = max(1, max(len(record["source_time_calendar"].encode("utf-8")) for record in records))
            output.createDimension("source_time_units_strlen", time_units_width)
            output.createDimension("source_time_calendar_strlen", calendar_width)
            output.setncatts(
                {
                    "schema_version": COMPACT_SCHEMA_VERSION,
                    "title": f"NOAA NYOFS {grid} POM compact fields",
                    "Conventions": "CF-1.10",
                    "source_system": "NYOFS",
                    "source_model": "POM",
                    "source_grid": grid,
                    "grid_type": "curvilinear",
                    "vector_components": "earth_relative",
                    "time_coverage_start": request["start_utc"],
                    "time_coverage_end_exclusive": request["end_utc_exclusive"],
                    "source_keys_json": json.dumps(sorted({str(record["key"]) for record in records})),
                    "created_utc": _iso(datetime.now(UTC)),
                    "history": "Created by OMA nyofs-fetcher",
                }
            )
            epoch_units = "seconds since 1970-01-01 00:00:00 UTC"
            time_var = output.createVariable("time", "f8", ("time",))
            time_var.setncatts({"standard_name": "time", "long_name": "normalized nominal time", "units": epoch_units, "calendar": "proleptic_gregorian"})
            time_var[:] = [_timestamp(record["time"]) for record in records]
            original_var = output.createVariable("original_time", "f8", ("time",))
            original_var.setncatts({"long_name": "decoded source time before nominal normalization", "units": epoch_units, "calendar": "proleptic_gregorian"})
            original_var[:] = [_timestamp(record["original_time"]) for record in records]
            offset_var = output.createVariable("original_time_offset_seconds", "f8", ("time",))
            offset_var.long_name = "original decoded time minus normalized time"
            offset_var.units = "s"
            offset_var[:] = [record["time_adjustment_seconds"] for record in records]
            raw_time_var = output.createVariable("source_time_value", "f8", ("time",))
            raw_time_var.long_name = "numeric source time coordinate before decoding"
            raw_time_var[:] = [record["raw_time_value"] for record in records]
            units_var = output.createVariable("source_time_units", "S1", ("time", "source_time_units_strlen"))
            _write_char_rows(units_var, [record["source_time_units"] for record in records], time_units_width)
            calendar_var = output.createVariable("source_time_calendar", "S1", ("time", "source_time_calendar_strlen"))
            _write_char_rows(calendar_var, [record["source_time_calendar"] for record in records], calendar_width)
            cycle_var = output.createVariable("source_cycle_time", "f8", ("time",))
            cycle_var.units = epoch_units
            cycle_var.calendar = "proleptic_gregorian"
            cycle_var[:] = [_timestamp(record["run_time"]) for record in records]
            key_var = output.createVariable("source_key", "S1", ("time", "source_key_strlen"))
            key_var.long_name = "public AWS source object key"
            _write_char_rows(key_var, [str(record["key"]) for record in records], source_width)

            for name, data, units, long_name in (
                ("lon", reference_geometry["lon"], "degrees_east", "longitude"),
                ("lat", reference_geometry["lat"], "degrees_north", "latitude"),
                ("depth", reference_geometry["depth"], "m", "positive-down bathymetry"),
            ):
                variable = output.createVariable(name, "f4", ("y", "x"), zlib=True, complevel=4, shuffle=True, fill_value=fill_value)
                variable.units = units
                variable.long_name = long_name
                variable[:] = np.where(np.isfinite(data), data, fill_value).astype(np.float32)
            mask_var = output.createVariable("mask", "i1", ("y", "x"), zlib=True, complevel=4, shuffle=True)
            mask_var.setncatts({"long_name": "POM wet mask", "flag_values": np.asarray([0, 1], dtype=np.int8), "flag_meanings": "land wet"})
            mask_var[:] = reference_geometry["wet"].astype(np.int8)
            sigma_var = output.createVariable("sigma", "f4", ("sigma",))
            sigma_var.setncatts({"long_name": "POM sigma point coordinate", "positive": "down", "units": "1"})
            sigma_var[:] = np.asarray(sigma, dtype=np.float32)

            for name, values in arrays.items():
                dimensions = ("time",) + tuple(source_schema[name]["dimensions"])
                variable = output.createVariable(name, "f4", dimensions, zlib=True, complevel=4, shuffle=True, fill_value=fill_value)
                _copy_attributes(first_ds.variables[name], variable)
                variable.coordinates = "time sigma lat lon" if "sigma" in dimensions else "time lat lon"
                variable[:] = np.where(np.isfinite(values), values, fill_value).astype(np.float32)
            for name, values in derived.items():
                variable = output.createVariable(name, "f4", ("time", "y", "x"), zlib=True, complevel=4, shuffle=True, fill_value=fill_value)
                variable.coordinates = "time lat lon"
                if name.startswith("current_speed"):
                    variable.setncatts({"long_name": name.replace("_", " "), "units": "m s-1", "standard_name": "sea_water_speed"})
                elif name == "wind_speed":
                    variable.setncatts({"long_name": "wind speed", "units": "m s-1", "standard_name": "wind_speed"})
                else:
                    base = name.split("_", 1)[0]
                    if base in first_ds.variables:
                        _copy_attributes(first_ds.variables[base], variable)
                    variable.long_name = name.replace("_", " ")
                variable[:] = np.where(np.isfinite(values), values, fill_value).astype(np.float32)
        os.replace(temporary, destination)
        return {
            "grid": grid,
            "path": str(destination.resolve()),
            "size": destination.stat().st_size,
            "sha256": _sha256(destination),
            "record_count": len(records),
            "start_utc": _iso(records[0]["time"]),
            "end_utc": _iso(records[-1]["time"]),
            "variables": sorted(list(arrays) + list(derived)),
            "source_keys": sorted({str(record["key"]) for record in records}),
            "sigma_weights": [float(x) for x in weights],
        }
    finally:
        for ds in datasets.values():
            ds.close()


def extract_request(request: Mapping[str, Any] | str | Path, run_dir: str | Path) -> dict[str, Any]:
    normalized = load_request(request) if isinstance(request, (str, Path)) else validate_request(request)
    if normalized["product"] != "fields":
        raise ValueError("extract is available only for product: fields; stations is passthrough-only")
    run_path = Path(run_dir)
    manifest_path = run_path / "fetch_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("fetch_manifest.json is required before extraction")
    fetch_manifest = _read_json(manifest_path)
    grids = ["coarse", "fine"] if normalized["grid"] == "both" else [normalized["grid"]]
    outputs = [_extract_grid(normalized, fetch_manifest.get("outcomes", []), grid, run_path) for grid in grids]
    report = {
        "schema_version": "nyofs_extraction_manifest_v1",
        "created_utc": _iso(datetime.now(UTC)),
        "request": normalized,
        "outputs": outputs,
        "cache_policy": normalized["cache_policy"],
        "raw_cache_deleted": False,
        "raw_cache_deletion_note": "delete_after_extract is deferred until check_download_health.py passes",
    }
    write_json_atomic(run_path / "extraction_manifest.json", report)
    return report


def _compact_times(ds: Any) -> list[datetime]:
    netCDF4, np = _netcdf_modules()
    variable = ds.variables["time"]
    decoded = netCDF4.num2date(
        np.asarray(variable[:], dtype=float),
        units=variable.units,
        calendar=getattr(variable, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=False,
    )
    return [_as_utc_datetime(value) for value in decoded]


def _finite_values(variable: Any):
    _, np = _netcdf_modules()
    return np.ma.asarray(variable[:], dtype=float).filled(np.nan)


def _raw_consistency(
    outcomes: Sequence[Mapping[str, Any]], request: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    netCDF4, np = _netcdf_modules()
    critical: list[str] = []
    warnings: list[str] = []
    by_grid: dict[str, list[dict[str, Any]]] = {"coarse": [], "fine": []}
    reference: dict[str, dict[str, Any]] = {}
    start = _parse_utc(request["start_utc"])
    end = _parse_utc(request["end_utc_exclusive"])
    cadence = 3600 if request["product"] == "fields" else 360
    for outcome in outcomes:
        if outcome.get("status") not in {"downloaded", "cache_hit"}:
            continue
        source = outcome.get("source") or {}
        grid = source.get("grid")
        if grid not in by_grid:
            continue
        path = Path(outcome["local_path"])
        try:
            with netCDF4.Dataset(path) as ds:
                records = decode_times(ds, request["product"])
                by_grid[grid].append(
                    {
                        "path": str(path),
                        "key": outcome.get("key"),
                        "count": len(records),
                        "times": records,
                        "run_time": _parse_utc(source["run_time"]),
                    }
                )
                if request["product"] == "fields":
                    candidate = _geometry(ds)
                    if grid in reference:
                        try:
                            _assert_geometry(reference[grid], candidate, path)
                        except RuntimeError as exc:
                            critical.append(str(exc))
                    else:
                        reference[grid] = candidate
        except Exception as exc:
            critical.append(f"cannot inspect raw object {path}: {type(exc).__name__}: {exc}")
    requested_grids = ["coarse", "fine"] if request["grid"] == "both" else [request["grid"]]
    summary: dict[str, Any] = {}
    expected = _expected_times(start, end, cadence)
    for grid in requested_grids:
        candidates: dict[datetime, list[dict[str, Any]]] = {}
        for item in by_grid[grid]:
            for record in item["times"]:
                if start <= record.normalized < end:
                    candidates.setdefault(record.normalized, []).append(
                        {"record": record, "run_time": item["run_time"], "key": item["key"]}
                    )
        unique = sorted(candidates)
        missing = [stamp for stamp in expected if stamp not in candidates]
        duplicate_count = sum(max(0, len(group) - 1) for group in candidates.values())
        nonmonotonic = any(right <= left for left, right in zip(unique, unique[1:]))
        summary[grid] = {
            "object_count": len(by_grid[grid]),
            "unique_requested_time_count": len(unique),
            "expected_time_count": len(expected),
            "missing_times": [_iso(stamp) for stamp in missing],
            "source_duplicate_record_count": duplicate_count,
            "deduplication": "preceding cycle terminal record wins",
            "monotonic": not nonmonotonic,
            "first_time_utc": _iso(unique[0]) if unique else None,
            "last_time_utc": _iso(unique[-1]) if unique else None,
        }
        if not by_grid[grid]:
            critical.append(f"no downloaded {grid} objects are available")
        if missing and request["missing_policy"] == "error":
            critical.append(f"raw {grid} coverage is missing {len(missing)} requested timestamps")
        if nonmonotonic:
            critical.append(f"raw {grid} normalized times are not strictly monotonic")
        offsets = [
            abs(entry["record"].adjustment_seconds)
            for group in candidates.values()
            for entry in group
        ]
        if offsets and max(offsets) > 60:
            warnings.append(f"raw {grid} contains source time jitter above 60 seconds")
    return summary, critical, warnings


def _check_compact(
    output: Mapping[str, Any],
    request: Mapping[str, Any],
    plots_dir: Path | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    netCDF4, np = _netcdf_modules()
    path = Path(output["path"])
    critical: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {"path": str(path), "grid": output.get("grid")}
    if not path.is_file():
        critical.append(f"compact output does not exist: {path}")
        return report, critical, warnings
    expected_hash = output.get("sha256")
    actual_hash = _sha256(path)
    report.update({"size": path.stat().st_size, "sha256": actual_hash, "hash_matches_manifest": actual_hash == expected_hash})
    if expected_hash and actual_hash != expected_hash:
        critical.append(f"compact output hash differs from extraction manifest: {path}")
    with netCDF4.Dataset(path) as ds:
        report["schema_version"] = getattr(ds, "schema_version", None)
        if report["schema_version"] != COMPACT_SCHEMA_VERSION:
            critical.append(f"unexpected compact schema version in {path}")
        required_geometry = {
            "lon",
            "lat",
            "mask",
            "depth",
            "sigma",
            "time",
            "original_time",
            "original_time_offset_seconds",
            "source_time_value",
            "source_time_units",
            "source_time_calendar",
            "source_cycle_time",
            "source_key",
        }
        absent_geometry = sorted(required_geometry - set(ds.variables))
        if absent_geometry:
            critical.append(f"compact output lacks required coordinates: {', '.join(absent_geometry)}")
            return report, critical, warnings
        lon = _finite_values(ds.variables["lon"])
        lat = _finite_values(ds.variables["lat"])
        mask = _finite_values(ds.variables["mask"])
        depth = _finite_values(ds.variables["depth"])
        sigma = np.asarray(ds.variables["sigma"][:], dtype=float)
        wet = np.isclose(mask, 1.0)
        mask_values = sorted(float(value) for value in np.unique(mask[np.isfinite(mask)]))
        geometry = {
            "shape": list(lon.shape),
            "mask_values": mask_values,
            "wet_cells": int(wet.sum()),
            "coordinate_finite_wet_fraction": float((np.isfinite(lon[wet]) & np.isfinite(lat[wet])).mean()) if wet.any() else 0.0,
            "positive_depth_wet_fraction": float((depth[wet] > 0).mean()) if wet.any() else 0.0,
            "sigma": [float(value) for value in sigma],
        }
        try:
            weights = sigma_trapezoid_weights(sigma)
            geometry["sigma_weights"] = [float(value) for value in weights]
            geometry["sigma_weight_sum"] = float(weights.sum())
        except ValueError as exc:
            critical.append(f"invalid sigma coordinate in {path}: {exc}")
        report["geometry"] = geometry
        if lon.ndim != 2 or lat.shape != lon.shape or mask.shape != lon.shape or depth.shape != lon.shape:
            critical.append(f"compact POM geometry shapes are inconsistent in {path}")
        if not set(mask_values).issubset({0.0, 1.0}) or not wet.any():
            critical.append(f"compact POM mask is invalid in {path}")
        if geometry["coordinate_finite_wet_fraction"] < 1.0:
            critical.append(f"compact wet coordinates are non-finite in {path}")

        times = _compact_times(ds)
        expected_times = _expected_times(_parse_utc(request["start_utc"]), _parse_utc(request["end_utc_exclusive"]), 3600)
        unique = len(set(times)) == len(times)
        monotonic = all(right > left for left, right in zip(times, times[1:]))
        missing = [stamp for stamp in expected_times if stamp not in set(times)]
        report["time"] = {
            "count": len(times),
            "expected_count": len(expected_times),
            "unique": unique,
            "monotonic": monotonic,
            "missing_times": [_iso(stamp) for stamp in missing],
            "first_utc": _iso(times[0]) if times else None,
            "last_utc": _iso(times[-1]) if times else None,
            "max_abs_original_offset_seconds": float(np.nanmax(np.abs(_finite_values(ds.variables["original_time_offset_seconds"])))) if times else None,
        }
        if not unique or not monotonic:
            critical.append(f"compact times are duplicate or non-monotonic in {path}")
        if missing and request["missing_policy"] == "error":
            critical.append(f"compact output is missing {len(missing)} requested hours in {path}")

        variable_checks: dict[str, Any] = {}
        requested = list(request["variables"])
        sigma_sources: list[str] = []
        for name in requested:
            if name not in ds.variables:
                critical.append(f"compact output lacks requested source field {name!r} in {path}")
                continue
            values = _finite_values(ds.variables[name])
            if values.ndim < 3:
                critical.append(f"compact source field {name!r} has unexpected dimensions in {path}")
                continue
            if values.ndim == 3:
                valid = np.isfinite(values) & wet[None, :, :]
                denominator = wet.sum()
                coverage = valid.sum(axis=(1, 2)) / max(1, denominator)
            elif values.ndim == 4:
                sigma_sources.append(name)
                valid = np.isfinite(values) & wet[None, None, :, :]
                denominator = wet.sum() * values.shape[1]
                coverage = valid.sum(axis=(1, 2, 3)) / max(1, denominator)
            else:
                critical.append(f"compact source field {name!r} has unsupported rank in {path}")
                continue
            minimum_coverage = float(np.min(coverage)) if coverage.size else 0.0
            variable_checks[name] = {"shape": list(values.shape), "minimum_finite_wet_fraction": minimum_coverage, "all_nan_frames": int((coverage == 0).sum())}
            if minimum_coverage < 0.95:
                critical.append(f"{name} finite wet coverage is below 95 percent in {path}")

        for name in sigma_sources:
            for view in request["vertical_views"]:
                derived_name = f"{name}_{_view_suffix(view)}"
                if derived_name not in ds.variables:
                    critical.append(f"compact output lacks requested vertical view {derived_name!r} in {path}")
                    continue
                values = _finite_values(ds.variables[derived_name])
                if values.shape != (len(times),) + wet.shape:
                    critical.append(f"compact vertical view {derived_name!r} has an unexpected shape in {path}")
                    continue
                coverage = (np.isfinite(values) & wet[None, :, :]).sum(axis=(1, 2)) / max(1, wet.sum())
                minimum_coverage = float(np.min(coverage)) if coverage.size else 0.0
                variable_checks[derived_name] = {
                    "shape": list(values.shape),
                    "minimum_finite_wet_fraction": minimum_coverage,
                    "all_nan_frames": int((coverage == 0).sum()),
                }
                if minimum_coverage < 0.95:
                    critical.append(f"{derived_name} finite wet coverage is below 95 percent in {path}")

        for view in request["vertical_views"]:
            suffix = _view_suffix(view)
            if "u" in requested and "v" in requested:
                names = [f"u_{suffix}", f"v_{suffix}", f"current_speed_{suffix}"]
                missing_names = [name for name in names if name not in ds.variables]
                if missing_names:
                    critical.append(f"compact output lacks velocity view variables {missing_names} in {path}")
                    continue
                u = _finite_values(ds.variables[names[0]])
                v = _finite_values(ds.variables[names[1]])
                speed = _finite_values(ds.variables[names[2]])
                if u.shape != v.shape or u.shape != speed.shape:
                    critical.append(f"paired velocity shapes differ for {suffix} in {path}")
                else:
                    expected_speed = np.hypot(u, v)
                    finite = np.isfinite(expected_speed) & np.isfinite(speed)
                    max_error = float(np.max(np.abs(expected_speed[finite] - speed[finite]))) if finite.any() else None
                    variable_checks[names[2]] = {
                        "shape": list(speed.shape),
                        "speed_max_abs_error": max_error,
                        "minimum_finite_wet_fraction": float(np.min((np.isfinite(speed) & wet[None, :, :]).sum(axis=(1, 2)) / max(1, wet.sum()))),
                    }
                    if variable_checks[names[2]]["minimum_finite_wet_fraction"] < 0.95:
                        critical.append(f"{names[2]} finite wet coverage is below 95 percent in {path}")
                    if not finite.any() or (max_error is not None and max_error > 1e-5):
                        critical.append(f"current speed is inconsistent with paired u/v for {suffix} in {path}")
                    finite_speed = speed[np.isfinite(speed)]
                    if finite_speed.size and (float(finite_speed.min()) < 0 or float(finite_speed.max()) > 10):
                        warnings.append(f"current speed outside broad 0..10 m/s range for {suffix} in {path}")
        if "air_u" in requested and "air_v" in requested:
            if "wind_speed" not in ds.variables:
                critical.append(f"compact output lacks wind_speed in {path}")
            else:
                wind = _finite_values(ds.variables["wind_speed"])
                expected_wind = np.hypot(_finite_values(ds.variables["air_u"]), _finite_values(ds.variables["air_v"]))
                finite = np.isfinite(wind) & np.isfinite(expected_wind)
                error = float(np.max(np.abs(wind[finite] - expected_wind[finite]))) if finite.any() else None
                variable_checks["wind_speed"] = {"shape": list(wind.shape), "speed_max_abs_error": error}
                if not finite.any() or (error is not None and error > 1e-5):
                    critical.append(f"wind_speed is inconsistent with air_u/air_v in {path}")
                if finite.any() and float(wind[finite].max()) > 100:
                    warnings.append(f"wind speed exceeds broad 100 m/s warning limit in {path}")
        if "zeta" in ds.variables:
            zeta = _finite_values(ds.variables["zeta"])
            finite_zeta = zeta[np.isfinite(zeta)]
            if finite_zeta.size and float(np.max(np.abs(finite_zeta))) > 15:
                warnings.append(f"water-surface elevation exceeds broad +/-15 m warning limit in {path}")
        report["variables"] = variable_checks

        if plots_dir is not None:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                plots_dir.mkdir(parents=True, exist_ok=True)
                figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
                colored = axis.pcolormesh(lon, lat, np.where(wet, depth, np.nan), shading="auto")
                figure.colorbar(colored, ax=axis, label="Depth (m)")
                axis.set_title(f"NYOFS {output.get('grid')} compact-grid health")
                axis.set_xlabel("Longitude")
                axis.set_ylabel("Latitude")
                latitude = lat[wet]
                if latitude.size:
                    axis.set_aspect(1.0 / max(0.2, math.cos(math.radians(float(np.nanmean(latitude))))))
                plot_path = plots_dir / f"{output.get('grid')}_grid_health.png"
                figure.savefig(plot_path, dpi=150)
                plt.close(figure)
                report["health_plot"] = str(plot_path.resolve())
            except Exception as exc:
                warnings.append(f"could not create optional health plot for {path}: {exc}")
    return report, critical, warnings


def _verify_transfers(
    estimate: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    critical: list[str] = []
    warnings: list[str] = []
    expected = {item["key"]: item for item in estimate.get("objects", [])}
    outcomes = {item["key"]: item for item in manifest.get("outcomes", [])}
    rows: list[dict[str, Any]] = []
    for key, source in expected.items():
        outcome = outcomes.get(key)
        if outcome is None:
            critical.append(f"fetch manifest has no outcome for {key}")
            continue
        row = {"key": key, "status": outcome.get("status"), "expected_size": source.get("size"), "expected_etag": _clean_etag(source.get("etag"))}
        if outcome.get("status") not in {"downloaded", "cache_hit"}:
            critical.append(f"transfer did not complete for {key}")
            rows.append(row)
            continue
        path = Path(outcome["local_path"])
        row["path"] = str(path)
        if not path.is_file():
            critical.append(f"downloaded object is missing locally: {path}")
            rows.append(row)
            continue
        actual_size = path.stat().st_size
        actual_hash = _sha256(path)
        row.update({"actual_size": actual_size, "sha256": actual_hash})
        if actual_size != int(source["size"]):
            critical.append(f"download size mismatch for {key}")
        if actual_hash != outcome.get("sha256"):
            critical.append(f"download SHA-256 mismatch for {key}")
        if _clean_etag(outcome.get("etag")) != _clean_etag(source.get("etag")):
            critical.append(f"download ETag provenance mismatch for {key}")
        sidecar = _download_sidecar(path)
        row["sidecar"] = str(sidecar)
        if not sidecar.is_file():
            critical.append(f"cache metadata sidecar is missing for {key}")
        else:
            try:
                metadata = _read_json(sidecar)
                if int(metadata.get("size", -1)) != int(source["size"]):
                    critical.append(f"cache sidecar size mismatch for {key}")
                if _clean_etag(metadata.get("etag")) != _clean_etag(source.get("etag")):
                    critical.append(f"cache sidecar ETag mismatch for {key}")
                if metadata.get("sha256") != actual_hash:
                    critical.append(f"cache sidecar SHA-256 mismatch for {key}")
            except Exception as exc:
                critical.append(f"cache sidecar cannot be parsed for {key}: {exc}")
        rows.append(row)
    extra = sorted(set(outcomes) - set(expected))
    if extra:
        warnings.append(f"fetch manifest contains {len(extra)} objects absent from the current estimate")
    return {"objects": rows, "expected_count": len(expected), "verified_count": len(rows)}, critical, warnings


def _delete_raw_cache(run_dir: Path, manifest: MutableMapping[str, Any]) -> dict[str, Any]:
    deleted: list[dict[str, Any]] = []
    for outcome in manifest.get("outcomes", []):
        path = Path(str(outcome.get("local_path", "")))
        if not path.is_file():
            continue
        size = path.stat().st_size
        path.unlink()
        sidecar = _download_sidecar(path)
        sidecar_size = sidecar.stat().st_size if sidecar.is_file() else 0
        sidecar.unlink(missing_ok=True)
        deleted.append({"path": str(path), "bytes": size, "sidecar_bytes": sidecar_size})
    raw_root = run_dir / "cache" / "raw"
    if raw_root.is_dir():
        for directory in sorted((path for path in raw_root.rglob("*") if path.is_dir()), key=lambda p: len(p.parts), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            raw_root.rmdir()
        except OSError:
            pass
    return {
        "deleted": True,
        "files": deleted,
        "source_bytes": sum(item["bytes"] for item in deleted),
        "sidecar_bytes": sum(item["sidecar_bytes"] for item in deleted),
        "recovery": "re-download from public AWS using download_estimate.json",
    }


def evaluate_health(
    request: Mapping[str, Any] | str | Path,
    run_dir: str | Path,
    *,
    output: str | Path | None = None,
    plots_dir: str | Path | None = None,
) -> dict[str, Any]:
    normalized = load_request(request) if isinstance(request, (str, Path)) else validate_request(request)
    run_path = Path(run_dir)
    estimate_path = run_path / "download_estimate.json"
    manifest_path = run_path / "fetch_manifest.json"
    if not estimate_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("download_estimate.json and fetch_manifest.json are required for health checking")
    estimate = _read_json(estimate_path)
    manifest = _read_json(manifest_path)
    transfer, critical, warnings = _verify_transfers(estimate, manifest)
    raw, raw_critical, raw_warnings = _raw_consistency(manifest.get("outcomes", []), normalized)
    critical.extend(raw_critical)
    warnings.extend(raw_warnings)
    compact_checks: list[dict[str, Any]] = []
    extraction: dict[str, Any] | None = None
    if normalized["product"] == "fields":
        extraction_path = run_path / "extraction_manifest.json"
        if not extraction_path.is_file():
            critical.append("extraction_manifest.json is required for fields health checking")
        else:
            extraction = _read_json(extraction_path)
            for compact_output in extraction.get("outputs", []):
                check, findings, cautions = _check_compact(
                    compact_output,
                    normalized,
                    Path(plots_dir) if plots_dir is not None else None,
                )
                compact_checks.append(check)
                critical.extend(findings)
                warnings.extend(cautions)
            expected_outputs = 2 if normalized["grid"] == "both" else 1
            if len(compact_checks) != expected_outputs:
                critical.append(f"expected {expected_outputs} compact grid outputs, found {len(compact_checks)}")
    deletion: dict[str, Any] | None = None
    passed_before_deletion = not critical
    if passed_before_deletion and normalized["cache_policy"] == "delete_after_extract":
        deletion = _delete_raw_cache(run_path, manifest)
        if extraction is not None:
            extraction["raw_cache_deleted"] = True
            extraction["raw_cache_deletion"] = deletion
            write_json_atomic(run_path / "extraction_manifest.json", extraction)
    report = {
        "schema_version": "nyofs_health_check_v1",
        "created_utc": _iso(datetime.now(UTC)),
        "request": normalized,
        "status": "pass" if not critical else "fail",
        "critical_findings": critical,
        "warnings": warnings,
        "transfer_integrity": transfer,
        "raw_source_consistency": raw,
        "compact_outputs": compact_checks,
        "raw_cache_deletion": deletion,
    }
    write_json_atomic(output or run_path / "health_check.json", report)
    return report


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request", required=True, help="Path to a nyofs_request_v1 JSON file")
    parser.add_argument("--run-dir", required=True, help="Run/evidence directory outside the skill package")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory_parser = subparsers.add_parser("inventory", help="List matching public S3 objects")
    _add_common_arguments(inventory_parser)
    plan_parser = subparsers.add_parser("plan", help="Select objects and write an exact storage estimate")
    _add_common_arguments(plan_parser)
    plan_parser.add_argument("--output", help="Optional estimate JSON path")
    fetch_parser = subparsers.add_parser("fetch", help="Plan and download selected objects")
    _add_common_arguments(fetch_parser)
    inspect_parser = subparsers.add_parser("inspect", help="Inspect cached NetCDF metadata")
    _add_common_arguments(inspect_parser)
    extract_parser = subparsers.add_parser("extract", help="Concatenate and derive compact POM fields")
    _add_common_arguments(extract_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inventory":
            result = inventory_request(args.request, args.run_dir)
        elif args.command == "plan":
            result = plan_request(args.request, args.run_dir, output=args.output)
        elif args.command == "fetch":
            result = fetch_request(args.request, args.run_dir)
        elif args.command == "inspect":
            result = inspect_request(args.request, args.run_dir)
        elif args.command == "extract":
            result = extract_request(args.request, args.run_dir)
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "ok", "command": args.command, "summary": _json_clean(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
