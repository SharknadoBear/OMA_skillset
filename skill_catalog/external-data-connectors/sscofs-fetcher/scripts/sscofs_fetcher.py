#!/usr/bin/env python3
"""Plan, fetch, inspect, and extract public NOAA SSCOFS NetCDF data."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote
import xml.etree.ElementTree as ET


UTC = timezone.utc
SCHEMA_VERSION = "sscofs_request_v1"
SOFTWARE_VERSION = "1.0.0"
BUCKET = "noaa-nos-ofs-pds"
S3_ROOT = f"https://{BUCKET}.s3.amazonaws.com"
LIST_URL = f"{S3_ROOT}/"
ARCHIVE_ROOT = "sscofs/netcdf"
CYCLE_HOURS = {3, 9, 15, 21}
PRODUCTS = {"fields", "stations", "regulargrid"}
GUIDANCE = {"nowcast", "forecast"}
NAMED_VIEWS = {"surface", "near_surface", "bottom", "depth_average"}

_CURRENT_FIELD = re.compile(
    r"sscofs\.t(?P<cycle>\d{2})z\.(?P<date>\d{8})\."
    r"(?P<product>fields|regulargrid)\.(?P<code>[nf])(?P<lead>\d{3})\.nc$",
    re.IGNORECASE,
)
_CURRENT_STATION = re.compile(
    r"sscofs\.t(?P<cycle>\d{2})z\.(?P<date>\d{8})\.stations\."
    r"(?P<guidance>nowcast|forecast)\.nc$",
    re.IGNORECASE,
)
_LEGACY_FIELD = re.compile(
    r"nos\.sscofs\.(?P<product>fields|regulargrid)\."
    r"(?P<code>[nf])(?P<lead>\d{3})\.(?P<date>\d{8})\.t(?P<cycle>\d{2})z\.nc$",
    re.IGNORECASE,
)
_LEGACY_STATION = re.compile(
    r"nos\.sscofs\.stations\.(?P<guidance>nowcast|forecast)\."
    r"(?P<date>\d{8})\.t(?P<cycle>\d{2})z\.nc$",
    re.IGNORECASE,
)


def _requests_module():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - environment failure.
        raise RuntimeError("The requests package is required for live SSCOFS access.") from exc
    return requests


def _parse_utc(value: Any, name: str = "timestamp") -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO-8601 timestamp with a UTC offset") from exc
    else:
        raise ValueError(f"{name} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include an explicit UTC offset")
    return parsed.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _hourly_range(start: datetime, end: datetime) -> list[datetime]:
    values: list[datetime] = []
    cursor = start
    while cursor < end:
        values.append(cursor)
        cursor += timedelta(hours=1)
    return values


def _view_suffix(view: str | int) -> str:
    return f"sigma_{view:03d}" if isinstance(view, int) else str(view)


def load_request(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("SSCOFS request JSON must contain an object")
    return validate_request(value)


def validate_request(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize an ``sscofs_request_v1`` mapping."""

    allowed = {
        "schema_version", "schema", "start_utc", "end_utc_exclusive", "product",
        "guidance", "run_cycle_utc", "variables", "vertical_views", "missing_policy",
        "cache_policy", "max_workers",
    }
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown request properties: {', '.join(unknown)}")
    schema = mapping.get("schema_version", mapping.get("schema"))
    if schema != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    start = _parse_utc(mapping.get("start_utc"), "start_utc")
    end = _parse_utc(mapping.get("end_utc_exclusive"), "end_utc_exclusive")
    if end <= start:
        raise ValueError("end_utc_exclusive must be later than start_utc")
    product = str(mapping.get("product", "")).lower()
    guidance = str(mapping.get("guidance", "")).lower()
    if product not in PRODUCTS:
        raise ValueError(f"product must be one of {sorted(PRODUCTS)}")
    if guidance not in GUIDANCE:
        raise ValueError(f"guidance must be one of {sorted(GUIDANCE)}")
    if product in {"fields", "regulargrid"} and any(
        (value.minute, value.second, value.microsecond) != (0, 0, 0) for value in (start, end)
    ):
        raise ValueError("Hourly field requests must start and end on whole UTC hours")

    run_cycle: datetime | None = None
    if mapping.get("run_cycle_utc") is not None:
        run_cycle = _parse_utc(mapping["run_cycle_utc"], "run_cycle_utc")
        if run_cycle.hour not in CYCLE_HOURS or any((run_cycle.minute, run_cycle.second, run_cycle.microsecond)):
            raise ValueError("run_cycle_utc must be a 03, 09, 15, or 21 UTC cycle")
    if guidance == "forecast" and run_cycle is None:
        raise ValueError("run_cycle_utc is required for forecast requests")

    variables_present = "variables" in mapping
    views_present = "vertical_views" in mapping
    if product != "fields" and (variables_present or views_present):
        raise ValueError("variables and vertical_views are supported only for product='fields'")
    variables = mapping.get("variables", ["salinity", "u", "v"] if product == "fields" else [])
    if not isinstance(variables, list) or any(not isinstance(item, str) or not item.strip() for item in variables):
        raise ValueError("variables must be a nonempty array of NetCDF variable names")
    variables = list(dict.fromkeys(item.strip() for item in variables))
    if product == "fields" and not variables:
        raise ValueError("fields requests require at least one variable")
    views = mapping.get("vertical_views", ["surface", "bottom", "depth_average"] if product == "fields" else [])
    if not isinstance(views, list) or (product == "fields" and not views):
        raise ValueError("vertical_views must be a nonempty array for fields requests")
    normalized_views: list[str | int] = []
    for view in views:
        if isinstance(view, bool):
            raise ValueError("Boolean values are not valid sigma indices")
        if isinstance(view, int):
            if view < 0:
                raise ValueError("Explicit sigma indices must be nonnegative")
            item: str | int = int(view)
        elif isinstance(view, str) and view.lower() in NAMED_VIEWS:
            item = view.lower()
        else:
            raise ValueError(f"Unsupported vertical view: {view!r}")
        if item not in normalized_views:
            normalized_views.append(item)

    missing_policy = str(mapping.get("missing_policy", "error")).lower()
    cache_policy = str(mapping.get("cache_policy", "keep")).lower()
    if missing_policy not in {"error", "skip"}:
        raise ValueError("missing_policy must be 'error' or 'skip'")
    if cache_policy not in {"keep", "delete_after_extract"}:
        raise ValueError("cache_policy must be 'keep' or 'delete_after_extract'")
    max_workers = int(mapping.get("max_workers", 4))
    if max_workers < 1 or max_workers > 32:
        raise ValueError("max_workers must be between 1 and 32")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "start_utc": _iso(start),
        "end_utc_exclusive": _iso(end),
        "product": product,
        "guidance": guidance,
        "missing_policy": missing_policy,
        "cache_policy": cache_policy,
        "max_workers": max_workers,
    }
    if run_cycle is not None:
        result["run_cycle_utc"] = _iso(run_cycle)
    if product == "fields":
        result["variables"] = variables
        result["vertical_views"] = normalized_views
    return result


def _layout_for_key(key: str) -> str:
    if re.search(r"/\d{4}/\d{2}/\d{2}/", key):
        return "nested_daily"
    if re.search(r"/\d{6}/", key):
        return "legacy_monthly"
    return "unknown"


def parse_object_key(key: str) -> dict[str, Any] | None:
    """Parse a current or legacy SSCOFS S3 object key."""

    name = Path(key).name
    match = _CURRENT_FIELD.search(name) or _LEGACY_FIELD.search(name)
    if match:
        groups = match.groupdict()
        cycle = int(groups["cycle"])
        if cycle not in CYCLE_HOURS:
            return None
        run = datetime.strptime(groups["date"], "%Y%m%d").replace(hour=cycle, tzinfo=UTC)
        lead = int(groups["lead"])
        guidance = "nowcast" if groups["code"].lower() == "n" else "forecast"
        valid = run + timedelta(hours=(lead - 6 if guidance == "nowcast" else lead))
        return {
            "key": key,
            "product": groups["product"].lower(),
            "guidance": guidance,
            "lead_hour": lead,
            "run_time_utc": _iso(run),
            "valid_time_utc": _iso(valid),
            "layout": _layout_for_key(key),
            "naming": "legacy" if name.lower().startswith("nos.") else "current",
        }
    match = _CURRENT_STATION.search(name) or _LEGACY_STATION.search(name)
    if match:
        groups = match.groupdict()
        cycle = int(groups["cycle"])
        if cycle not in CYCLE_HOURS:
            return None
        run = datetime.strptime(groups["date"], "%Y%m%d").replace(hour=cycle, tzinfo=UTC)
        return {
            "key": key,
            "product": "stations",
            "guidance": groups["guidance"].lower(),
            "lead_hour": None,
            "run_time_utc": _iso(run),
            "valid_time_utc": None,
            "layout": _layout_for_key(key),
            "naming": "legacy" if name.lower().startswith("nos.") else "current",
        }
    return None


def _xml_text(node: ET.Element, name: str) -> str | None:
    child = node.find(f"{{*}}{name}")
    return child.text if child is not None else None


def list_s3_objects(
    prefix: str,
    *,
    session: Any | None = None,
    timeout: float = 60.0,
    max_keys: int = 1000,
) -> list[dict[str, Any]]:
    """List and parse all anonymous S3 objects beneath *prefix*."""

    own_session = session is None
    if session is None:
        session = _requests_module().Session()
    results: list[dict[str, Any]] = []
    token: str | None = None
    try:
        while True:
            params: dict[str, Any] = {"list-type": "2", "prefix": prefix, "max-keys": max_keys}
            if token:
                params["continuation-token"] = token
            response = session.get(LIST_URL, params=params, timeout=timeout)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for item in root.findall("{*}Contents"):
                key = _xml_text(item, "Key")
                size_text = _xml_text(item, "Size")
                if not key or not size_text or int(size_text) <= 0:
                    continue
                parsed = parse_object_key(key)
                if parsed is None:
                    continue
                parsed.update(
                    {
                        "size_bytes": int(size_text),
                        "size": int(size_text),
                        "etag": (_xml_text(item, "ETag") or "").strip('"'),
                        "last_modified": _xml_text(item, "LastModified"),
                        "url": f"{S3_ROOT}/{quote(key, safe='/')}",
                    }
                )
                results.append(parsed)
            truncated = (_xml_text(root, "IsTruncated") or "false").lower() == "true"
            if not truncated:
                break
            token = _xml_text(root, "NextContinuationToken")
            if not token:
                raise RuntimeError("S3 listing was truncated without a continuation token")
    finally:
        if own_session and hasattr(session, "close"):
            session.close()
    return results


def _month_starts(start: datetime, end: datetime) -> Iterable[datetime]:
    cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    stop = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cursor <= stop:
        yield cursor
        cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)


def _discovery_prefixes(request: Mapping[str, Any]) -> list[str]:
    start = _parse_utc(request["start_utc"])
    end = _parse_utc(request["end_utc_exclusive"])
    look_start = start - timedelta(days=1)
    look_end = end + timedelta(days=1)
    days = max(1, math.ceil((look_end - look_start).total_seconds() / 86400))
    prefixes: list[str] = []
    if days <= 14:
        cursor = look_start.replace(hour=0, minute=0, second=0, microsecond=0)
        while cursor < look_end:
            prefixes.append(f"{ARCHIVE_ROOT}/{cursor:%Y/%m/%d}/")
            cursor += timedelta(days=1)
    else:
        prefixes.extend(f"{ARCHIVE_ROOT}/{month:%Y/%m}/" for month in _month_starts(look_start, look_end))
    prefixes.extend(f"{ARCHIVE_ROOT}/{month:%Y%m}/" for month in _month_starts(look_start, look_end))
    # Forecast objects are stored under the run date, which can be as much as
    # 72 hours earlier than the requested valid-time window.
    if request.get("guidance") == "forecast" and request.get("run_cycle_utc"):
        run = _parse_utc(request["run_cycle_utc"], "run_cycle_utc")
        prefixes.append(f"{ARCHIVE_ROOT}/{run:%Y/%m/%d}/")
        prefixes.append(f"{ARCHIVE_ROOT}/{run:%Y%m}/")
    return list(dict.fromkeys(prefixes))


def discover_objects(
    request: Mapping[str, Any], *, session: Any | None = None, timeout: float = 60.0
) -> list[dict[str, Any]]:
    """Discover live objects from both supported archive layouts."""

    normalized = validate_request(request)
    objects: dict[str, dict[str, Any]] = {}
    own_session = session is None
    if session is None:
        session = _requests_module().Session()
    try:
        for prefix in _discovery_prefixes(normalized):
            for item in list_s3_objects(prefix, session=session, timeout=timeout):
                objects[item["key"]] = item
    finally:
        if own_session and hasattr(session, "close"):
            session.close()
    return sorted(objects.values(), key=lambda item: item["key"])


def _preference(item: Mapping[str, Any]) -> tuple[int, int, int, str]:
    lead = item.get("lead_hour")
    return (
        # At a cycle boundary n006 is the continuous nowcast record and n000
        # is its duplicate in the following run.  Guidance continuity outranks
        # which archive layout happens to contain the copy.
        1 if lead == 6 else (0 if lead == 0 else -1),
        1 if item.get("layout") == "nested_daily" else 0,
        1 if item.get("naming") == "current" else 0,
        str(item.get("key", "")),
    )


def select_objects(request: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Filter, deduplicate, and coverage-check discovered objects."""

    normalized = validate_request(request)
    start = _parse_utc(normalized["start_utc"])
    end = _parse_utc(normalized["end_utc_exclusive"])
    filtered = [
        dict(item) for item in objects
        if item.get("product") == normalized["product"] and item.get("guidance") == normalized["guidance"]
    ]
    if normalized["guidance"] == "forecast":
        cycle = normalized["run_cycle_utc"]
        filtered = [item for item in filtered if item.get("run_time_utc") == cycle]

    duplicates: list[dict[str, Any]] = []
    missing: list[datetime] = []
    if normalized["product"] == "stations":
        selected: list[dict[str, Any]] = []
        for item in filtered:
            run = _parse_utc(item["run_time_utc"])
            if normalized["guidance"] == "nowcast":
                overlaps = run >= start and (run - timedelta(hours=6)) < end
            else:
                overlaps = True
            if overlaps:
                selected.append(item)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in selected:
            grouped.setdefault(str(item["run_time_utc"]), []).append(item)
        selected = []
        for run_time, group in grouped.items():
            keep = max(group, key=_preference)
            selected.append(keep)
            duplicates.extend(
                {"run_time_utc": run_time, "kept_key": keep["key"], "discarded_key": other["key"]}
                for other in group if other is not keep
            )
        selected.sort(key=lambda item: item["run_time_utc"])
        if not selected and normalized["missing_policy"] == "error":
            raise ValueError("No station object covers the requested window")
    else:
        grouped = {}
        for item in filtered:
            value = item.get("valid_time_utc")
            if not value:
                continue
            valid = _parse_utc(value)
            if start <= valid < end:
                grouped.setdefault(_iso(valid), []).append(item)
        selected = []
        for valid_time, group in grouped.items():
            keep = max(group, key=_preference)
            selected.append(keep)
            duplicates.extend(
                {"valid_time_utc": valid_time, "kept_key": keep["key"], "discarded_key": other["key"]}
                for other in group if other is not keep
            )
        selected.sort(key=lambda item: item["valid_time_utc"])
        expected = _hourly_range(start, end)
        observed = {_parse_utc(item["valid_time_utc"]) for item in selected}
        missing = [item for item in expected if item not in observed]
        if missing and normalized["missing_policy"] == "error":
            raise ValueError(
                "Missing required SSCOFS hours: " + ", ".join(_iso(item) or "" for item in missing[:12])
            )
    return {
        "objects": selected,
        "selected_objects": selected,
        "missing_times": [_iso(item) for item in missing],
        "duplicate_records": duplicates,
        "candidate_count": len(filtered),
        "selected_count": len(selected),
    }


def write_json_atomic(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(part, target)
    return target


def _disk_free(path: Path) -> int:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return int(shutil.disk_usage(candidate).free)


def inventory_request(
    request: Mapping[str, Any], *, objects: Sequence[Mapping[str, Any]] | None = None,
    session: Any | None = None,
) -> dict[str, Any]:
    normalized = validate_request(request)
    found = list(objects) if objects is not None else discover_objects(normalized, session=session)
    relevant = [
        dict(item) for item in found
        if item.get("product") == normalized["product"] and item.get("guidance") == normalized["guidance"]
    ]
    return {
        "schema_version": "sscofs_inventory_v1",
        "generated_utc": _iso(datetime.now(UTC)),
        "source": {"bucket": BUCKET, "access": "anonymous_https_listobjectsv2"},
        "request": normalized,
        "prefixes": _discovery_prefixes(normalized),
        "object_count": len(relevant),
        "objects": relevant,
    }


def plan_request(
    request: Mapping[str, Any],
    run_dir: str | Path | None = None,
    *,
    objects: Sequence[Mapping[str, Any]] | None = None,
    session: Any | None = None,
) -> dict[str, Any]:
    """Resolve exact objects, bytes, gaps, and the OMA storage route."""

    normalized = validate_request(request)
    found = list(objects) if objects is not None else discover_objects(normalized, session=session)
    selection = select_objects(normalized, found)
    selected = selection["objects"]
    total = int(sum(int(item["size_bytes"]) for item in selected))
    path = Path(run_dir or ".")
    path.mkdir(parents=True, exist_ok=True)
    free = _disk_free(path)
    local_ok = bool(selected) and free > 4 * total
    route = "local" if local_ok else "kestrel"
    reason = (
        f"Local free space ({free} bytes) exceeds four times the exact request ({4 * total} bytes)."
        if local_ok else
        f"Local free space ({free} bytes) does not exceed four times the exact request ({4 * total} bytes)."
    )
    estimate = {
        "schema_version": "sscofs_download_estimate_v1",
        "generated_utc": _iso(datetime.now(UTC)),
        "software_version": SOFTWARE_VERSION,
        "request": normalized,
        "source": {"bucket": BUCKET, "region": "us-east-1", "access": "anonymous_https"},
        "objects": selected,
        "selected_object_count": len(selected),
        "candidate_object_count": selection["candidate_count"],
        "total_bytes": total,
        "total_gib": total / 1024 ** 3,
        "missing_times": selection["missing_times"],
        "duplicate_records": selection["duplicate_records"],
        "local_free_bytes": free,
        "required_free_bytes": 4 * total,
        "routing_decision": route,
        "routing_reason": reason,
        "download_gate": "approved" if local_ok else "route_to_kestrel",
        "kestrel_scratch_path": "/scratch/yhuang168/oma_external_data_connectors/sscofs-fetcher/<run-id>",
    }
    if run_dir is not None:
        write_json_atomic(path / "download_estimate.json", estimate)
    return estimate


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _object_size(item: Mapping[str, Any]) -> int:
    return int(item.get("size_bytes", item.get("size", 0)))


def _clean_etag(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _download_sidecar(destination: Path) -> Path:
    return destination.with_name(destination.name + ".download.json")


def _cache_result(item: Mapping[str, Any], destination: Path) -> dict[str, Any] | None:
    if not destination.is_file() or destination.stat().st_size != _object_size(item):
        return None
    sidecar_path = _download_sidecar(destination)
    sidecar: dict[str, Any] = {}
    if sidecar_path.is_file():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            sidecar = {}
    if not sidecar:
        # Size alone cannot prove that an existing file belongs to this S3
        # object, especially when the source ETag is multipart.
        return None
    expected_etag = _clean_etag(item.get("etag"))
    if sidecar and (
        int(sidecar.get("size_bytes", -1)) != _object_size(item)
        or _clean_etag(sidecar.get("etag")) != expected_etag
    ):
        return None
    recorded_digest = str(sidecar.get("sha256", ""))
    digest = _sha256(destination)
    if len(recorded_digest) == 64 and digest.lower() != recorded_digest.lower():
        return None
    result = {
        "key": item["key"],
        "url": item["url"],
        "local_path": str(destination.resolve()),
        "status": "cache_hit",
        "cache_hit": True,
        "size_bytes": _object_size(item),
        "etag": expected_etag,
        "sha256": digest,
        "retries": 0,
        "resumed_bytes": 0,
        "valid_time_utc": item.get("valid_time_utc"),
        "run_time_utc": item.get("run_time_utc"),
        "product": item.get("product"),
        "guidance": item.get("guidance"),
    }
    write_json_atomic(sidecar_path, result)
    return result


def download_object(
    item: Mapping[str, Any],
    destination: str | Path,
    *,
    session: Any | None = None,
    max_retries: int = 4,
    timeout: float = 120.0,
    chunk_size: int = 8 * 1024 * 1024,
    retry_delay: float = 1.0,
) -> dict[str, Any]:
    """Download one object atomically with safe range resumption."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = _cache_result(item, target)
    if cached is not None:
        return cached
    expected_size = _object_size(item)
    if expected_size <= 0:
        raise ValueError(f"Object {item.get('key')} has an invalid size")
    expected_etag = _clean_etag(item.get("etag"))
    part = target.with_name(target.name + ".part")
    if part.exists() and part.stat().st_size > expected_size:
        part.unlink()
    own_session = session is None
    if session is None:
        session = _requests_module().Session()
    retries = 0
    original_part_size = part.stat().st_size if part.exists() else 0
    last_error: Exception | None = None
    try:
        for attempt in range(max_retries + 1):
            try:
                start = part.stat().st_size if part.exists() else 0
                headers = {"Range": f"bytes={start}-"} if start else {}
                response = session.get(item["url"], headers=headers, stream=True, timeout=timeout)
                response.raise_for_status()
                status = int(getattr(response, "status_code", 200))
                response_etag = _clean_etag(getattr(response, "headers", {}).get("ETag"))
                if expected_etag and response_etag and response_etag != expected_etag:
                    raise IOError(f"ETag changed for {item['key']}: {response_etag} != {expected_etag}")
                if start and status == 206:
                    content_range = getattr(response, "headers", {}).get("Content-Range", "")
                    if content_range and not content_range.startswith(f"bytes {start}-"):
                        raise IOError(f"Server returned an incompatible Content-Range: {content_range}")
                    mode = "ab"
                elif start and status == 200:
                    mode = "wb"
                    start = 0
                elif not start and status in {200, 206}:
                    mode = "wb"
                else:
                    raise IOError(f"Unexpected HTTP status {status} for {item['key']}")
                with part.open(mode) as handle:
                    for block in response.iter_content(chunk_size=chunk_size):
                        if block:
                            handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
                actual_size = part.stat().st_size
                if actual_size != expected_size:
                    raise IOError(f"Downloaded size {actual_size} does not match listed size {expected_size}")
                digest = _sha256(part)
                os.replace(part, target)
                result = {
                    "key": item["key"],
                    "url": item["url"],
                    "local_path": str(target.resolve()),
                    "status": "downloaded",
                    "cache_hit": False,
                    "size_bytes": expected_size,
                    "etag": expected_etag,
                    "sha256": digest,
                    "retries": retries,
                    "resumed_bytes": original_part_size,
                    "valid_time_utc": item.get("valid_time_utc"),
                    "run_time_utc": item.get("run_time_utc"),
                    "product": item.get("product"),
                    "guidance": item.get("guidance"),
                }
                write_json_atomic(_download_sidecar(target), result)
                return result
            except Exception as exc:  # Network/filesystem retries are recorded.
                last_error = exc
                if attempt >= max_retries:
                    break
                retries += 1
                time.sleep(retry_delay * (2 ** attempt))
    finally:
        if own_session and hasattr(session, "close"):
            session.close()
    raise RuntimeError(f"Failed to download {item.get('key')} after {retries} retries: {last_error}")


def fetch_request(
    request_or_plan: Mapping[str, Any],
    run_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    max_retries: int = 4,
    force_route: bool = False,
) -> dict[str, Any]:
    """Fetch all objects in a request/plan and write ``fetch_manifest.json``."""

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    if request_or_plan.get("schema_version") == "sscofs_download_estimate_v1":
        plan = dict(request_or_plan)
        request = validate_request(plan["request"])
        if not (run_path / "download_estimate.json").is_file():
            write_json_atomic(run_path / "download_estimate.json", plan)
    else:
        request = validate_request(request_or_plan)
        plan = plan_request(request, run_dir=run_path)
    if plan.get("routing_decision") != "local" and not force_route:
        raise RuntimeError(
            "The four-times-free-space gate recommends Kestrel; pass --force-route only after explicit review."
        )
    raw_dir = Path(cache_dir) if cache_dir else run_path / "cache" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    objects = list(plan.get("objects", []))
    started = datetime.now(UTC)
    records: list[dict[str, Any]] = []

    def job(item: Mapping[str, Any]) -> dict[str, Any]:
        target = raw_dir / Path(str(item["key"])).name
        try:
            return download_object(item, target, max_retries=max_retries)
        except Exception as exc:
            return {
                "key": item.get("key"), "url": item.get("url"),
                "local_path": str(target.resolve()), "status": "failed",
                "cache_hit": False, "size_bytes": _object_size(item),
                "etag": _clean_etag(item.get("etag")), "error": str(exc),
                "valid_time_utc": item.get("valid_time_utc"),
                "run_time_utc": item.get("run_time_utc"),
                "product": item.get("product"), "guidance": item.get("guidance"),
            }

    with ThreadPoolExecutor(max_workers=request["max_workers"]) as pool:
        futures = {pool.submit(job, item): item for item in objects}
        for future in as_completed(futures):
            records.append(future.result())
            records.sort(key=lambda item: (str(item.get("valid_time_utc")), str(item.get("key"))))
            checkpoint = {
                "schema_version": "sscofs_fetch_manifest_v1",
                "generated_utc": _iso(datetime.now(UTC)),
                "software_version": SOFTWARE_VERSION,
                "request": request,
                "estimate_path": str((run_path / "download_estimate.json").resolve()),
                "source": plan.get("source"),
                "records": records,
                "complete": len(records) == len(objects) and all(item["status"] != "failed" for item in records),
            }
            write_json_atomic(run_path / "fetch_manifest.json", checkpoint)
    failures = [item for item in records if item["status"] == "failed"]
    manifest = {
        "schema_version": "sscofs_fetch_manifest_v1",
        "generated_utc": _iso(datetime.now(UTC)),
        "started_utc": _iso(started),
        "software_version": SOFTWARE_VERSION,
        "request": request,
        "estimate_path": str((run_path / "download_estimate.json").resolve()),
        "source": plan.get("source"),
        "selected_object_count": len(objects),
        "successful_object_count": len(records) - len(failures),
        "failure_count": len(failures),
        "downloaded_count": sum(item["status"] == "downloaded" for item in records),
        "cache_hit_count": sum(item["status"] == "cache_hit" for item in records),
        "records": records,
        "complete": len(records) == len(objects) and not failures,
    }
    write_json_atomic(run_path / "fetch_manifest.json", manifest)
    if failures and request["missing_policy"] == "error":
        raise RuntimeError(f"{len(failures)} required SSCOFS object download(s) failed")
    return manifest


def _filled(value: Any, dtype: Any = float):
    import numpy as np
    if np.ma.isMaskedArray(value):
        return np.ma.filled(value.astype(dtype), np.nan)
    return np.asarray(value, dtype=dtype)


def _decode_times(ds: Any) -> list[datetime]:
    import numpy as np
    if "Times" in ds.variables:
        values = ds.variables["Times"][:]
        rows = values if np.ndim(values) > 1 else np.asarray(values).reshape(1, -1)
        result: list[datetime] = []
        for row in rows:
            parts = []
            for item in row:
                if isinstance(item, bytes):
                    parts.append(item.decode("ascii", errors="ignore"))
                else:
                    parts.append(str(item))
            text = "".join(parts).replace("\x00", "").strip()
            if text:
                text = text.replace("_", "T")
                result.append(_parse_utc(text + ("Z" if not re.search(r"(?:Z|[+-]\d\d:\d\d)$", text) else ""), "Times"))
        if result:
            return result
    if "time" in ds.variables:
        var = ds.variables["time"]
        units = getattr(var, "units", None)
        if units:
            from netCDF4 import num2date
            values = num2date(var[:], units=units, calendar=getattr(var, "calendar", "standard"))
            return [datetime(item.year, item.month, item.day, item.hour, item.minute, item.second, tzinfo=UTC) for item in values]
    return []


def inspect_file(path: str | Path) -> dict[str, Any]:
    """Inspect NetCDF dimensions, variables, centering, and decoded times."""

    from netCDF4 import Dataset
    source = Path(path)
    with Dataset(source) as ds:
        dimensions = {name: len(dim) for name, dim in ds.dimensions.items()}
        variables: dict[str, Any] = {}
        for name, var in ds.variables.items():
            variables[name] = {
                "dimensions": list(var.dimensions),
                "shape": list(var.shape),
                "dtype": str(var.dtype),
                "units": getattr(var, "units", None),
                "long_name": getattr(var, "long_name", None),
            }
        return {
            "path": str(source.resolve()),
            "size_bytes": source.stat().st_size,
            "dimensions": dimensions,
            "variables": variables,
            "times": [_iso(item) for item in _decode_times(ds)],
            "mesh": {
                "node_count": dimensions.get("node"),
                "element_count": dimensions.get("nele"),
                "sigma_layer_count": dimensions.get("siglay"),
                "has_connectivity": "nv" in ds.variables,
            },
        }


def thickness_weights(siglev: Any):
    """Return nonnegative sigma-layer thickness fractions."""

    import numpy as np
    values = _filled(siglev, float)
    if values.ndim < 1 or values.shape[0] < 2:
        raise ValueError("siglev must contain at least two interfaces on its first axis")
    weights = np.abs(np.diff(values, axis=0))
    weights[~np.isfinite(weights)] = np.nan
    return weights


def weighted_vertical_average(data: Any, weights: Any, wet_mask: Any | None = None):
    """Average the penultimate (sigma-layer) axis over finite wet layers."""

    import numpy as np
    values = _filled(data, float)
    layer_weights = _filled(weights, float)
    if values.ndim < 2:
        raise ValueError("data must include layer and horizontal axes")
    if layer_weights.ndim != 2:
        raise ValueError("weights must be shaped (layer, horizontal)")
    if values.shape[-2:] != layer_weights.shape:
        raise ValueError(f"data layer/horizontal shape {values.shape[-2:]} != weights {layer_weights.shape}")
    reshape = (1,) * (values.ndim - 2) + layer_weights.shape
    broadcast_weights = np.broadcast_to(layer_weights.reshape(reshape), values.shape)
    valid = np.isfinite(values) & np.isfinite(broadcast_weights) & (broadcast_weights > 0)
    if wet_mask is not None:
        wet = np.asarray(wet_mask, dtype=bool)
        if wet.shape != values.shape[:-2] + (values.shape[-1],):
            try:
                wet = np.broadcast_to(wet, values.shape[:-2] + (values.shape[-1],))
            except ValueError as exc:
                raise ValueError("wet_mask is not broadcastable to data columns") from exc
        valid &= np.expand_dims(wet, axis=-2)
    numerator = np.sum(np.where(valid, values * broadcast_weights, 0.0), axis=-2)
    denominator = np.sum(np.where(valid, broadcast_weights, 0.0), axis=-2)
    result = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result




def _normalize_connectivity(values: Any, nnode: int):
    import numpy as np
    nv = np.asarray(values, dtype=np.int64).squeeze()
    if nv.ndim != 2:
        raise ValueError(f"nv must be two-dimensional; got {nv.shape}")
    if nv.shape[0] != 3 and nv.shape[1] == 3:
        nv = nv.T
    if nv.shape[0] != 3:
        raise ValueError(f"nv must have a three-node axis; got {nv.shape}")
    minimum = int(np.nanmin(nv))
    if minimum == 1:
        nv = nv - 1
    elif minimum != 0:
        raise ValueError(f"nv has unsupported starting index {minimum}")
    if int(np.nanmin(nv)) < 0 or int(np.nanmax(nv)) >= nnode:
        raise ValueError("nv connectivity contains indices outside the node dimension")
    if np.any((nv[0] == nv[1]) | (nv[1] == nv[2]) | (nv[0] == nv[2])):
        raise ValueError("nv connectivity contains repeated nodes within an element")
    return nv.astype(np.int32)


def _variable_time_axis(variable: Any) -> int | None:
    for index, dim in enumerate(variable.dimensions):
        if "time" in dim.lower():
            return index
    return None


def _read_record(variable: Any, time_index: int):
    axis = _variable_time_axis(variable)
    if axis is None:
        return _filled(variable[:], float if variable.dtype.kind in "fc" else variable.dtype)
    selection = [slice(None)] * variable.ndim
    selection[axis] = time_index
    dtype = float if variable.dtype.kind in "fc" else variable.dtype
    return _filled(variable[tuple(selection)], dtype)


def _copy_attributes(source: Any, destination: Any, *, skip: set[str] | None = None) -> None:
    omitted = {"_FillValue"}.union(skip or set())
    for name in source.ncattrs():
        if name not in omitted:
            try:
                destination.setncattr(name, source.getncattr(name))
            except (TypeError, ValueError):
                destination.setncattr(name, str(source.getncattr(name)))


def _read_wet_mask(ds: Any, location: str, time_index: int, count: int):
    import numpy as np
    names = (
        ("wet_nodes", "wet_nodes_prev_int", "wet_nodes_prev_ext", "wet_node_mask")
        if location == "node" else
        ("wet_cells", "wet_cells_prev_int", "wet_cells_prev_ext", "wet_cell_mask")
    )
    for name in names:
        if name not in ds.variables:
            continue
        values = np.asarray(_read_record(ds.variables[name], time_index)).squeeze().reshape(-1)
        if values.size == count:
            return np.isfinite(values) & (values > 0), name
    return np.ones(count, dtype=bool), None


def _requested_source_names(request: Mapping[str, Any], available: set[str]) -> list[str]:
    by_lower = {name.lower(): name for name in available}
    names: list[str] = []
    for requested in request.get("variables", []):
        lower = requested.lower()
        if lower == "salt":
            candidates = ["salinity"]
        elif lower in {"velocity", "current", "current_speed"}:
            candidates = ["u", "v"]
        else:
            candidates = [lower]
        for candidate in candidates:
            if candidate not in by_lower:
                raise KeyError(f"Requested field variable {requested!r} is unavailable; examples: {sorted(available)[:25]}")
            actual = by_lower[candidate]
            if actual not in names:
                names.append(actual)
    if ("u" in names) != ("v" in names):
        other = "v" if "u" in names else "u"
        if other not in by_lower:
            raise KeyError("Velocity extraction requires paired native u and v variables")
        names.append(by_lower[other])
    return names


def _records_from_files(paths: Sequence[str | Path], request: Mapping[str, Any]) -> list[dict[str, Any]]:
    from netCDF4 import Dataset
    normalized = validate_request(request)
    start = _parse_utc(normalized["start_utc"])
    end = _parse_utc(normalized["end_utc_exclusive"])
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with Dataset(path) as ds:
            times = _decode_times(ds)
        if not times:
            raise ValueError(f"Could not decode Times/time in {path}")
        parsed = parse_object_key(path.name)
        for index, valid in enumerate(times):
            if parsed and parsed.get("valid_time_utc") and len(times) == 1:
                expected = _parse_utc(parsed["valid_time_utc"])
                if abs((valid - expected).total_seconds()) > 1:
                    raise ValueError(f"Filename/NetCDF valid-time mismatch in {path}: {_iso(expected)} != {_iso(valid)}")
            if start <= valid < end:
                records.append({"path": path, "time_index": index, "time": valid})
    records.sort(key=lambda item: item["time"])
    observed = [item["time"] for item in records]
    if len(set(observed)) != len(observed):
        raise ValueError("Input files contain duplicate verified Times values")
    expected = _hourly_range(start, end)
    missing = [item for item in expected if item not in set(observed)]
    if missing and normalized["missing_policy"] == "error":
        raise ValueError("Downloaded NetCDF files are missing requested hours: " + ", ".join(_iso(item) or "" for item in missing[:12]))
    if not records:
        raise ValueError("No downloaded NetCDF records fall inside the request window")
    return records


def _geometry_arrays(ds: Any) -> dict[str, Any]:
    required = ("lon", "lat", "nv", "siglay", "siglev")
    missing = [name for name in required if name not in ds.variables]
    if missing:
        raise KeyError("Native fields file lacks required FVCOM geometry: " + ", ".join(missing))
    names = (
        "lon", "lat", "x", "y", "h", "lonc", "latc", "xc", "yc", "nv", "nbe",
        "siglay", "siglev", "art1", "art2",
    )
    return {name: _filled(ds.variables[name][:], float if ds.variables[name].dtype.kind in "fc" else ds.variables[name].dtype) for name in names if name in ds.variables}


def _assert_same_geometry(reference: Mapping[str, Any], ds: Any, path: Path) -> None:
    import numpy as np
    current = _geometry_arrays(ds)
    if set(current) != set(reference):
        raise ValueError(f"Static geometry variable set changed in {path}")
    for name, expected in reference.items():
        actual = current[name]
        if np.shape(actual) != np.shape(expected):
            raise ValueError(f"Geometry shape for {name} changed in {path}")
        if np.issubdtype(np.asarray(expected).dtype, np.floating):
            same = np.allclose(actual, expected, rtol=1.0e-7, atol=1.0e-8, equal_nan=True)
        else:
            same = np.array_equal(actual, expected)
        if not same:
            raise ValueError(f"Static geometry/topology variable {name} changed in {path}")


def _create_compressed_variable(ds: Any, name: str, dtype: Any, dimensions: Sequence[str], **kwargs: Any):
    numeric = str(dtype)[0] not in {"S", "U", "|"}
    options = dict(kwargs)
    if numeric and dimensions:
        options.update({"zlib": True, "complevel": 4, "shuffle": True})
    return ds.createVariable(name, dtype, tuple(dimensions), **options)


def extract_fields(
    paths: Sequence[str | Path],
    request: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Concatenate verified native fields and derive requested sigma views."""

    import numpy as np
    from netCDF4 import Dataset, date2num

    normalized = validate_request(request)
    if normalized["product"] != "fields":
        raise ValueError("extract_fields supports only product='fields'")
    records = _records_from_files(paths, normalized)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    if part.exists():
        part.unlink()

    with Dataset(records[0]["path"]) as first:
        geometry = _geometry_arrays(first)
        nnode = int(np.asarray(geometry["lon"]).size)
        connectivity = _normalize_connectivity(geometry["nv"], nnode)
        nele = connectivity.shape[1]
        siglay = np.asarray(geometry["siglay"], dtype=float)
        siglev = np.asarray(geometry["siglev"], dtype=float)
        if siglay.ndim == 1:
            siglay = np.repeat(siglay[:, None], nnode, axis=1)
        if siglev.ndim == 1:
            siglev = np.repeat(siglev[:, None], nnode, axis=1)
        if siglay.shape[1] != nnode or siglev.shape[1] != nnode:
            raise ValueError("siglay/siglev horizontal dimension does not match node")
        node_weights = thickness_weights(siglev)
        if node_weights.shape != siglay.shape:
            raise ValueError("siglev differences do not align with siglay")
        element_weights = np.nanmean(node_weights[:, connectivity], axis=1)
        layer_mean = np.nanmean(siglay, axis=1)
        order = np.argsort(layer_mean)[::-1]
        surface_index = int(order[0])
        near_surface_index = int(order[1]) if len(order) > 1 else surface_index
        bottom_index = int(order[-1])
        for view in normalized["vertical_views"]:
            if isinstance(view, int) and view >= siglay.shape[0]:
                raise ValueError(f"Explicit sigma index {view} is outside 0..{siglay.shape[0] - 1}")
        available = set(first.variables)
        source_names = _requested_source_names(normalized, available)
        source_schema = {name: (first.variables[name].dimensions, first.variables[name].dtype.str) for name in source_names}
        first_dimensions = {name: len(dim) for name, dim in first.dimensions.items()}
        geometry_schema = {name: first.variables[name] for name in geometry}

    with Dataset(part, "w", format="NETCDF4") as output:
        output.setncattr("title", "Compact NOAA SSCOFS native fields")
        output.setncattr("source", "NOAA noaa-nos-ofs-pds public AWS archive")
        output.setncattr("software", f"sscofs-fetcher {SOFTWARE_VERSION}")
        output.setncattr("request_json", json.dumps(normalized, sort_keys=True))
        output.setncattr("source_files_json", json.dumps([str(item["path"].resolve()) for item in records]))
        output.setncattr("surface_sigma_index", surface_index)
        output.setncattr("near_surface_sigma_index", near_surface_index)
        output.setncattr("bottom_sigma_index", bottom_index)

        # Preserve every dimension needed by geometry and requested source variables.
        needed_dims: set[str] = {"time", "DateStrLen"}
        with Dataset(records[0]["path"]) as first:
            for name in geometry:
                needed_dims.update(first.variables[name].dimensions)
            for name in source_names:
                needed_dims.update(first.variables[name].dimensions)
            time_dim_names = {dim for dim in needed_dims if "time" in dim.lower()}
            for name in sorted(needed_dims):
                if name in time_dim_names:
                    if name not in output.dimensions:
                        output.createDimension(name, len(records))
                elif name == "DateStrLen":
                    if name not in output.dimensions:
                        output.createDimension(name, max(20, first_dimensions.get(name, 20)))
                elif name not in output.dimensions:
                    output.createDimension(name, first_dimensions[name])

            # Ensure canonical horizontal/layer dimensions exist even in unusual files.
            canonical_dims = {
                "node": nnode, "nele": nele, "three": 3,
                "siglay": siglay.shape[0], "siglev": siglev.shape[0],
            }
            for name, size in canonical_dims.items():
                if name not in output.dimensions:
                    output.createDimension(name, size)

            for name in geometry:
                source = first.variables[name]
                fill_value = getattr(source, "_FillValue", None)
                options = {"fill_value": fill_value} if fill_value is not None else {}
                destination = _create_compressed_variable(output, name, source.dtype, source.dimensions, **options)
                _copy_attributes(source, destination)
                destination[:] = source[:]

            time_dim = next((name for name in output.dimensions if name.lower() == "time"), "time")
            if "time" in output.variables:
                del_name = None  # A geometry/source time variable is never copied above.
            time_var = output.createVariable("time", "f8", (time_dim,))
            time_var.units = "seconds since 1970-01-01 00:00:00 +00:00"
            time_var.calendar = "standard"
            time_var.standard_name = "time"
            time_var[:] = date2num([item["time"] for item in records], time_var.units, time_var.calendar)
            date_dim = "DateStrLen"
            times_var = output.createVariable("Times", "S1", (time_dim, date_dim))
            times_var.long_name = "UTC valid time"
            for index, item in enumerate(records):
                text = (_iso(item["time"]) or "")[: len(output.dimensions[date_dim])]
                encoded = np.full(len(output.dimensions[date_dim]), b" ", dtype="S1")
                chars = np.frombuffer(text.encode("ascii"), dtype="S1")
                encoded[: chars.size] = chars
                times_var[index, :] = encoded

            wet_nodes_var = _create_compressed_variable(output, "wet_nodes", "i1", (time_dim, "node"))
            wet_nodes_var.flag_values = np.asarray([0, 1], dtype=np.int8)
            wet_cells_var = _create_compressed_variable(output, "wet_cells", "i1", (time_dim, "nele"))
            wet_cells_var.flag_values = np.asarray([0, 1], dtype=np.int8)

            source_outputs: dict[str, Any] = {}
            for name in source_names:
                source = first.variables[name]
                fill_value = getattr(source, "_FillValue", None)
                if fill_value is None and source.dtype.kind in "fc":
                    fill_value = np.nan
                options = {"fill_value": fill_value} if fill_value is not None else {}
                destination = _create_compressed_variable(output, name, source.dtype, source.dimensions, **options)
                _copy_attributes(source, destination)
                destination.setncattr("source_variable", name)
                source_outputs[name] = destination

        derived: dict[str, Any] = {}
        lower_to_actual = {name.lower(): name for name in source_names}
        has_salinity = "salinity" in lower_to_actual
        has_velocity = "u" in lower_to_actual and "v" in lower_to_actual
        for view in normalized["vertical_views"]:
            suffix = _view_suffix(view)
            if has_salinity:
                var = _create_compressed_variable(output, f"salinity_{suffix}", "f4", (time_dim, "node"), fill_value=np.nan)
                var.units = getattr(source_outputs[lower_to_actual["salinity"]], "units", "1")
                var.long_name = f"salinity at {suffix.replace('_', ' ')}"
                derived[f"salinity_{suffix}"] = var
            if has_velocity:
                for component in ("u", "v"):
                    var = _create_compressed_variable(output, f"{component}_{suffix}", "f4", (time_dim, "nele"), fill_value=np.nan)
                    var.units = getattr(source_outputs[lower_to_actual[component]], "units", "m s-1")
                    var.long_name = f"{component} velocity at {suffix.replace('_', ' ')}"
                    derived[f"{component}_{suffix}"] = var
                speed = _create_compressed_variable(output, f"current_speed_{suffix}", "f4", (time_dim, "nele"), fill_value=np.nan)
                speed.units = "m s-1"
                speed.long_name = f"current speed at {suffix.replace('_', ' ')}"
                derived[f"current_speed_{suffix}"] = speed

        for output_index, record in enumerate(records):
            with Dataset(record["path"]) as source_ds:
                _assert_same_geometry(geometry, source_ds, record["path"])
                for name, schema in source_schema.items():
                    if name not in source_ds.variables:
                        raise ValueError(f"Requested variable {name} disappeared in {record['path']}")
                    current_schema = (source_ds.variables[name].dimensions, source_ds.variables[name].dtype.str)
                    if current_schema != schema:
                        raise ValueError(f"Schema for requested variable {name} changed in {record['path']}")
                wet_nodes, wet_node_source = _read_wet_mask(source_ds, "node", record["time_index"], nnode)
                wet_cells, wet_cell_source = _read_wet_mask(source_ds, "element", record["time_index"], nele)
                wet_nodes_var[output_index, :] = wet_nodes.astype(np.int8)
                wet_cells_var[output_index, :] = wet_cells.astype(np.int8)
                if wet_node_source is None:
                    wet_nodes_var.setncattr("inferred_all_wet", "true; no recognized source mask")
                if wet_cell_source is None:
                    wet_cells_var.setncattr("inferred_all_wet", "true; no recognized source mask")

                loaded: dict[str, Any] = {}
                for name in source_names:
                    source_var = source_ds.variables[name]
                    values = _read_record(source_var, record["time_index"])
                    loaded[name.lower()] = np.asarray(values, dtype=float) if source_var.dtype.kind in "fc" else values
                    destination = source_outputs[name]
                    axis = _variable_time_axis(destination)
                    if axis is None:
                        if output_index == 0:
                            destination[:] = values
                    else:
                        selection = [slice(None)] * destination.ndim
                        selection[axis] = output_index
                        destination[tuple(selection)] = values

                salinity = loaded.get("salinity")
                u = loaded.get("u")
                v = loaded.get("v")
                for view in normalized["vertical_views"]:
                    suffix = _view_suffix(view)
                    if view == "depth_average":
                        salt_view = weighted_vertical_average(salinity, node_weights, wet_nodes) if salinity is not None else None
                        u_view = weighted_vertical_average(u, element_weights, wet_cells) if u is not None else None
                        v_view = weighted_vertical_average(v, element_weights, wet_cells) if v is not None else None
                    else:
                        if view == "surface":
                            layer = surface_index
                        elif view == "near_surface":
                            layer = near_surface_index
                        elif view == "bottom":
                            layer = bottom_index
                        else:
                            layer = int(view)
                        salt_view = np.asarray(salinity[layer, :], dtype=float) if salinity is not None else None
                        u_view = np.asarray(u[layer, :], dtype=float) if u is not None else None
                        v_view = np.asarray(v[layer, :], dtype=float) if v is not None else None
                        if salt_view is not None:
                            salt_view = np.where(wet_nodes, salt_view, np.nan)
                        if u_view is not None:
                            u_view = np.where(wet_cells, u_view, np.nan)
                        if v_view is not None:
                            v_view = np.where(wet_cells, v_view, np.nan)
                    if salt_view is not None:
                        derived[f"salinity_{suffix}"][output_index, :] = salt_view
                    if u_view is not None and v_view is not None:
                        derived[f"u_{suffix}"][output_index, :] = u_view
                        derived[f"v_{suffix}"][output_index, :] = v_view
                        derived[f"current_speed_{suffix}"][output_index, :] = np.hypot(u_view, v_view)

    os.replace(part, target)
    return {
        "schema_version": "sscofs_extraction_v1",
        "output_path": str(target.resolve()),
        "record_count": len(records),
        "start_utc": _iso(records[0]["time"]),
        "end_utc": _iso(records[-1]["time"]),
        "source_files": [str(item["path"].resolve()) for item in records],
        "source_variables": source_names,
        "derived_variables": sorted(derived),
        "surface_sigma_index": surface_index,
        "near_surface_sigma_index": near_surface_index,
        "bottom_sigma_index": bottom_index,
        "sha256": _sha256(target),
        "size_bytes": target.stat().st_size,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _manifest_paths(run_dir: Path) -> list[Path]:
    manifest_path = run_dir / "fetch_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing fetch manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    paths = [
        Path(item["local_path"]) for item in manifest.get("records", [])
        if item.get("status") in {"downloaded", "cache_hit"} and item.get("local_path")
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest raw file is absent: {missing[0]}")
    return paths


def _delete_raw_after_extract(run_dir: Path) -> int:
    manifest_path = run_dir / "fetch_manifest.json"
    manifest = _read_json(manifest_path)
    deleted = 0
    run_resolved = run_dir.resolve()
    for record in manifest.get("records", []):
        if record.get("status") not in {"downloaded", "cache_hit"} or not record.get("local_path"):
            continue
        path = Path(record["local_path"]).resolve()
        try:
            path.relative_to(run_resolved)
        except ValueError as exc:
            raise RuntimeError(f"Refusing to delete cache outside run directory: {path}") from exc
        if path.is_file():
            path.unlink()
            deleted += 1
        sidecar = _download_sidecar(path)
        if sidecar.is_file():
            sidecar.unlink()
        record["status"] = "deleted_after_extract"
        record["deleted_utc"] = _iso(datetime.now(UTC))
    manifest["cache_cleanup"] = {
        "policy": "delete_after_extract", "deleted_file_count": deleted,
        "completed_utc": _iso(datetime.now(UTC)),
    }
    write_json_atomic(manifest_path, manifest)
    return deleted


def _add_common_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request", required=True, help="Path to an sscofs_request_v1 JSON file.")
    parser.add_argument("--run-dir", required=True, help="Run directory for estimates, manifests, cache, and products.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Discover matching anonymous S3 objects.")
    _add_common_request_arguments(inventory)
    inventory.add_argument("--output", default=None, help="Defaults to <run-dir>/inventory.json.")

    plan = subparsers.add_parser("plan", help="Resolve exact files, bytes, gaps, and storage route.")
    _add_common_request_arguments(plan)
    plan.add_argument("--output", default=None, help="Defaults to <run-dir>/download_estimate.json.")

    fetch = subparsers.add_parser("fetch", help="Download a planned request with resume/cache support.")
    _add_common_request_arguments(fetch)
    fetch.add_argument("--cache-dir", default=None)
    fetch.add_argument("--max-retries", type=int, default=4)
    fetch.add_argument("--force-route", action="store_true", help="Override a non-local routing recommendation after review.")

    inspect = subparsers.add_parser("inspect", help="Inspect cached NetCDF variables and dimensions.")
    _add_common_request_arguments(inspect)
    inspect.add_argument("--path", action="append", default=None, help="Explicit NetCDF path; repeat as needed.")
    inspect.add_argument("--output", default=None, help="Defaults to <run-dir>/inspection.json.")

    extract = subparsers.add_parser("extract", help="Create a compact native-fields NetCDF product.")
    _add_common_request_arguments(extract)
    extract.add_argument("--path", action="append", default=None, help="Explicit raw NetCDF path; repeat as needed.")
    extract.add_argument("--output", default=None, help="Defaults to <run-dir>/compact/sscofs_fields_compact.nc.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        request = load_request(args.request)
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.command == "inventory":
            result = inventory_request(request)
            output = Path(args.output) if args.output else run_dir / "inventory.json"
            write_json_atomic(output, result)
            summary = {"object_count": result["object_count"], "output": str(output)}
        elif args.command == "plan":
            result = plan_request(request, run_dir=run_dir)
            output = Path(args.output) if args.output else run_dir / "download_estimate.json"
            if output != run_dir / "download_estimate.json":
                write_json_atomic(output, result)
            summary = {
                "selected_object_count": result["selected_object_count"],
                "total_bytes": result["total_bytes"],
                "routing_decision": result["routing_decision"],
                "output": str(output),
            }
        elif args.command == "fetch":
            estimate_path = run_dir / "download_estimate.json"
            estimate = _read_json(estimate_path) if estimate_path.is_file() else plan_request(request, run_dir=run_dir)
            result = fetch_request(
                estimate, run_dir, cache_dir=args.cache_dir,
                max_retries=args.max_retries, force_route=args.force_route,
            )
            summary = {
                "complete": result["complete"], "downloaded_count": result["downloaded_count"],
                "cache_hit_count": result["cache_hit_count"], "failure_count": result["failure_count"],
                "output": str(run_dir / "fetch_manifest.json"),
            }
        elif args.command == "inspect":
            paths = [Path(path) for path in args.path] if args.path else _manifest_paths(run_dir)
            result = {
                "schema_version": "sscofs_inspection_v1",
                "generated_utc": _iso(datetime.now(UTC)),
                "files": [inspect_file(path) for path in paths],
            }
            output = Path(args.output) if args.output else run_dir / "inspection.json"
            write_json_atomic(output, result)
            summary = {"file_count": len(paths), "output": str(output)}
        else:
            paths = [Path(path) for path in args.path] if args.path else _manifest_paths(run_dir)
            output = Path(args.output) if args.output else run_dir / "compact" / "sscofs_fields_compact.nc"
            result = extract_fields(paths, request, output)
            write_json_atomic(run_dir / "extraction_manifest.json", result)
            deleted = _delete_raw_after_extract(run_dir) if request["cache_policy"] == "delete_after_extract" else 0
            summary = {
                "record_count": result["record_count"], "output": str(output),
                "size_bytes": result["size_bytes"], "deleted_raw_count": deleted,
                "manifest": str(run_dir / "extraction_manifest.json"),
            }
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"sscofs-fetcher {getattr(args, 'command', 'command')} failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
