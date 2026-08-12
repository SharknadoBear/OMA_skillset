#!/usr/bin/env python3
"""Plan, fetch, inspect, and extract NOAA SJROFS EFDC data from AWS/NCEI."""

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

try:
    from . import ofs_archive_sources as archive_sources
except ImportError:
    import ofs_archive_sources as archive_sources

BUCKET = "noaa-nos-ofs-pds"
S3_ENDPOINT = f"https://{BUCKET}.s3.amazonaws.com"
SCHEMA_VERSION = "sjrofs_request_v2"
LEGACY_SCHEMA_VERSION = "sjrofs_request_v1"
COMPACT_SCHEMA_VERSION = "efdc_compact_fields_v1"
CYCLE_HOURS = {5, 11, 17, 23}
DEFAULT_VARIABLES = ["zeta", "salt", "u", "v"]
DEFAULT_VIEWS: list[str | int] = ["surface", "bottom", "depth_average"]
UTC = timezone.utc
SOURCE_POLICIES = {"aws_then_ncei", "aws_only", "ncei_only"}

_CURRENT_RE = re.compile(
    r"^(?P<model>sjrofs)\.t(?P<hour>\d{2})z\."
    r"(?P<date>\d{8})\.(?P<product>fields|stations)\."
    r"(?P<guidance>nowcast|forecast)\.nc$",
    re.IGNORECASE,
)
_LEGACY_RE = re.compile(
    r"^(?:nos\.)?(?P<model>sjrofs)\."
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


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(_json_clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _view_suffix(view: str | int) -> str:
    return f"sigma_{view}" if isinstance(view, int) else str(view)


def load_request(path: str | Path) -> dict[str, Any]:
    mapping = _read_json(path)
    if not isinstance(mapping, Mapping):
        raise ValueError("request root must be a JSON object")
    return validate_request(mapping)


def request_migration(mapping: Mapping[str, Any]) -> dict[str, Any]:
    original = mapping.get("schema_version")
    return {
        "original_schema_version": original,
        "normalized_schema_version": SCHEMA_VERSION,
        "migrated": original == LEGACY_SCHEMA_VERSION,
        "defaults_applied": ["source_policy=aws_then_ncei"]
        if original == LEGACY_SCHEMA_VERSION and "source_policy" not in mapping else [],
    }


def validate_request(mapping: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "start_utc",
        "end_utc_exclusive",
        "product",
        "guidance",
        "run_cycle_utc",
        "variables",
        "vertical_views",
        "missing_policy",
        "cache_policy",
        "max_workers",
        "source_policy",
    }
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown request properties: {', '.join(unknown)}")
    if mapping.get("schema_version") not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r} or {LEGACY_SCHEMA_VERSION!r}")
    if mapping.get("schema_version") == LEGACY_SCHEMA_VERSION and "source_policy" in mapping:
        raise ValueError("source_policy is a v2-only field; omit it from v1 requests and migrate first")
    start = _parse_utc(mapping.get("start_utc"), "start_utc")
    end = _parse_utc(mapping.get("end_utc_exclusive"), "end_utc_exclusive")
    if end <= start:
        raise ValueError("end_utc_exclusive must be later than start_utc")
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
    source_policy = mapping.get("source_policy", "aws_then_ncei")
    if source_policy not in SOURCE_POLICIES:
        raise ValueError(f"source_policy must be one of {sorted(SOURCE_POLICIES)}")

    normalized: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "start_utc": _iso(start),
        "end_utc_exclusive": _iso(end),
        "product": product,
        "guidance": guidance,
        "missing_policy": missing_policy,
        "cache_policy": cache_policy,
        "max_workers": max_workers,
        "source_policy": source_policy,
    }
    if run_cycle is not None:
        normalized["run_cycle_utc"] = _iso(run_cycle)
    if product == "fields":
        normalized["variables"] = variables
        normalized["vertical_views"] = views
    return normalized


def _layout_for_key(key: str) -> str:
    if key.startswith(archive_sources.get_source_descriptor("ncei_long_term", "sjrofs")["root_prefix"]):
        return "ncei_monthly"
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
    if run.hour not in CYCLE_HOURS:
        return None
    model = values["model"].lower()
    grid = "native"
    product = values["product"].lower()
    guidance = values["guidance"].lower()
    lead = None
    valid = None
    aggregate = True
    cycle_center = run + timedelta(minutes=30)
    if guidance == "nowcast" and product == "fields":
        expected_start = cycle_center - timedelta(hours=5)
        expected_end = cycle_center + timedelta(hours=1)
    elif guidance == "nowcast":
        expected_start = cycle_center - timedelta(hours=5, minutes=54)
        expected_end = cycle_center + timedelta(minutes=6)
    elif product == "fields":
        expected_start = cycle_center
        expected_end = cycle_center + timedelta(hours=49)
    else:
        expected_start = cycle_center
        expected_end = cycle_center + timedelta(hours=48, minutes=6)
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
    if endpoint != S3_ENDPOINT:
        # Retain the injected endpoint hook only for the pre-v2 offline fixture.
        descriptor_endpoint = archive_sources.AWS_ENDPOINT
        if endpoint != descriptor_endpoint:
            return _legacy_test_listing(prefix, session=session, endpoint=endpoint, timeout=timeout)
    return archive_sources.list_objects_v2(
        "aws_operational", "sjrofs", prefix, session=session, timeout=timeout,
    )


def _legacy_test_listing(prefix: str, *, session: Any, endpoint: str, timeout: float) -> list[dict[str, Any]]:
    """Compatibility shim for the existing in-memory pagination test only."""
    token: str | None = None
    objects: list[dict[str, Any]] = []
    while True:
        params: dict[str, Any] = {"list-type": "2", "prefix": prefix, "max-keys": 1000}
        if token:
            params["continuation-token"] = token
        response = session.get(endpoint + "/", params=params, timeout=timeout)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for node in root.iter():
            if node.tag.rsplit("}", 1)[-1] == "Contents":
                key, size = _xml_text(node, "Key"), _xml_text(node, "Size")
                if key and size:
                    objects.append({"key": key, "size": int(size), "etag": (_xml_text(node, "ETag") or "").strip('"'),
                                    "last_modified": _xml_text(node, "LastModified"), "url": endpoint + "/" + quote(key, safe="/")})
        truncated = next((node.text for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "IsTruncated"), "false")
        if str(truncated).lower() != "true":
            return objects
        token = next((node.text for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "NextContinuationToken"), None)
        if not token:
            raise RuntimeError("S3 listing is truncated but has no continuation token")


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
    daily = [f"sjrofs/netcdf/{d:%Y/%m/%d}/" for d in _days(first, last)]
    monthly = [f"sjrofs/netcdf/{m:%Y%m}/" for m in _month_starts(first, last)]
    return daily + monthly


def _source_ids(request: Mapping[str, Any]) -> list[str]:
    return ["aws_operational"] if request["source_policy"] == "aws_only" else (["ncei_long_term"] if request["source_policy"] == "ncei_only" else ["aws_operational", "ncei_long_term"])


def _ncei_capability(request: Mapping[str, Any]) -> tuple[bool, str | None]:
    if request["product"] == "fields" and request["guidance"] == "forecast":
        return False, "NCEI does not provide verified SJROFS field-forecast fallback"
    return True, None


def _ncei_prefixes(request: Mapping[str, Any]) -> list[str]:
    descriptor = archive_sources.get_source_descriptor("ncei_long_term", "sjrofs")
    if request["guidance"] == "forecast":
        center = _parse_utc(request["run_cycle_utc"])
        first, last = center - timedelta(days=1), center + timedelta(days=1)
    else:
        first = _parse_utc(request["start_utc"]) - timedelta(days=1)
        last = _parse_utc(request["end_utc_exclusive"]) + timedelta(days=1)
    return [f"{descriptor['root_prefix']}{month:%Y/%m}/" for month in _month_starts(first, last)]


def _decorate_source(item: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    descriptor = archive_sources.get_source_descriptor(source_id, "sjrofs")
    value = {**descriptor, **dict(item), "source_id": source_id}
    if str(value.get("key", "")).startswith(descriptor["root_prefix"]):
        value["url"] = archive_sources.canonical_object_url(source_id, "sjrofs", str(value["key"]))
    value.setdefault("size", int(value.get("size_bytes", 0)))
    value["naming_era"] = value.get("naming")
    identity = {
        "model": "sjrofs", "product": value.get("product"),
        "guidance": value.get("guidance"), "run_time": value.get("run_time"),
        "valid_time": value.get("valid_time"), "aggregate": value.get("aggregate"),
    }
    value["semantic_identity"] = identity
    value["semantic_identity_digest"] = archive_sources.semantic_identity_digest(identity)
    value["source_identity"] = archive_sources.source_identity_digest(value)
    return value


def probe_legacy_ncei_coverage(
    item: Mapping[str, Any], *, session: Any | None = None, timeout: float = 60.0,
    max_probe_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Decode ambiguous legacy SJROFS times from a bounded HTTP range probe."""
    if item.get("source_id") != "ncei_long_term" or item.get("naming") != "legacy_aggregate" or item.get("product") != "fields":
        return dict(item)
    own = session is None
    client = session or _requests_module().Session()
    size = int(item["size"])
    if size > max_probe_bytes:
        raise RuntimeError(
            f"ambiguous legacy SJROFS object is {size} bytes, above the reviewed "
            f"{max_probe_bytes}-byte metadata-probe ceiling"
        )
    errors: list[str] = []
    transferred_bytes = 0
    request_count = 0
    try:
        probe_size = min(size, 1024 * 1024)
        while probe_size <= min(size, max_probe_bytes):
            response = client.get(
                str(item["url"]), headers={"Range": f"bytes=0-{probe_size - 1}"},
                timeout=timeout,
            )
            try:
                response.raise_for_status()
                remote_etag = _clean_etag(response.headers.get("ETag"))
                if remote_etag != _clean_etag(item.get("etag")):
                    raise RuntimeError("NCEI probe ETag differs from listing")
                content_range = archive_sources.parse_content_range(response.headers.get("Content-Range"))
                if response.status_code == 206 and content_range != (0, probe_size - 1, size):
                    raise RuntimeError("NCEI probe Content-Range differs from the planned object")
                content = bytes(response.content)
                request_count += 1
                transferred_bytes += len(content)
                if response.status_code == 200:
                    if len(content) != size:
                        raise RuntimeError("NCEI server ignored Range but did not return the complete listed object")
                elif len(content) != probe_size:
                    raise RuntimeError("NCEI probe returned an unexpected byte count")
                try:
                    import netCDF4
                    with netCDF4.Dataset("sjrofs_probe.nc", memory=content) as ds:
                        decoded = decode_times(ds, "fields")
                    if not decoded:
                        raise RuntimeError("bounded probe decoded no time records")
                    times = [record.normalized for record in decoded]
                    result = dict(item)
                    result["aggregate"] = len(times) > 1
                    result["naming"] = "legacy_aggregate" if len(times) > 1 else "legacy_single"
                    result["valid_time"] = _iso(times[0]) if len(times) == 1 else None
                    result["expected_start_utc"] = _iso(min(times))
                    result["expected_end_utc_exclusive"] = _iso(max(times) + timedelta(hours=1))
                    result["coverage_probe"] = {
                        "method": "bounded_http_range_netcdf_time_decode",
                        "bytes": transferred_bytes,
                        "request_count": request_count,
                        "final_response_bytes": len(content),
                        "server_honored_range": response.status_code == 206,
                        "record_count": len(times),
                        "first_utc": _iso(min(times)), "last_utc": _iso(max(times)),
                        "etag": remote_etag,
                    }
                    return _decorate_source(result, "ncei_long_term")
                except Exception as exc:
                    errors.append(f"{probe_size} bytes: {exc}")
                    if response.status_code == 200:
                        raise RuntimeError(
                            "could not decode ambiguous legacy SJROFS coverage from the complete "
                            "server-returned object: " + errors[-1]
                        ) from exc
            finally:
                if hasattr(response, "close"):
                    response.close()
            if probe_size == size or probe_size == max_probe_bytes:
                break
            probe_size = min(size, max_probe_bytes, probe_size * 2)
        raise RuntimeError("could not decode ambiguous legacy SJROFS coverage: " + "; ".join(errors[-3:]))
    finally:
        if own and hasattr(client, "close"):
            client.close()


def _nominally_relevant_before_probe(item: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    """Cheap semantic/window gate applied before any ambiguous NetCDF probe."""
    if item.get("product") != request["product"] or item.get("guidance") != request["guidance"]:
        return False
    if request["guidance"] == "forecast" and item.get("run_time") != request.get("run_cycle_utc"):
        return False
    start = _parse_utc(request["start_utc"])
    end = _parse_utc(request["end_utc_exclusive"])
    return _overlaps(item, start, end)


def _coverage_probe_summary(objects: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    probes = [item.get("coverage_probe") for item in objects if isinstance(item.get("coverage_probe"), Mapping)]
    return {
        "object_count": len(probes),
        "bytes": sum(int(item.get("bytes", 0)) for item in probes),
    }


def _discover_source(request: Mapping[str, Any], source_id: str, *, session: Any | None = None) -> list[dict[str, Any]]:
    prefixes = discovery_prefixes(request) if source_id == "aws_operational" else _ncei_prefixes(request)
    found: dict[str, dict[str, Any]] = {}
    for prefix in prefixes:
        for raw in archive_sources.list_objects_v2(source_id, "sjrofs", prefix, session=session):
            parsed = parse_object_key(raw["key"])
            if parsed is None:
                continue
            item = _decorate_source({**raw, **parsed}, source_id)
            if not _nominally_relevant_before_probe(item, request):
                continue
            archive_sources.validate_source_object("sjrofs", item, expected_source_id=source_id)
            # The decoded time coordinate is authoritative and may narrow the
            # conservative pre-probe span out of the requested window.
            if not _nominally_relevant_before_probe(item, request):
                continue
            found[item["key"]] = item
    return sorted(found.values(), key=lambda x: (x["run_time"], x["key"]))


def _selection_probe(request: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probe = dict(request)
    probe["missing_policy"] = "skip"
    return select_objects(probe, objects)


def _scientific_fallback_times(
    request: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Find station boundaries covered only by a following cycle's initial record."""
    if request["product"] != "stations" or request["guidance"] != "nowcast":
        return []
    start = _parse_utc(request["start_utc"])
    end = _parse_utc(request["end_utc_exclusive"])
    candidates = [
        item for item in objects
        if item.get("product") == "stations" and item.get("guidance") == "nowcast"
    ]
    unresolved: list[str] = []
    for stamp in _expected_times(start, end, 360):
        covering = [item for item in candidates if stamp in _nominal_item_times(item, "stations")]
        if (
            covering
            and any(_parse_utc(item["expected_start_utc"]) == stamp for item in covering)
            and not any(_parse_utc(item["run_time"]) == stamp for item in covering)
        ):
            unresolved.append(_iso(stamp))
    return unresolved


def validate_fallback_decision(
    request: Mapping[str, Any], trace: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Validate source policy, provider order, and fallback trigger evidence."""
    failures: list[str] = []
    policy = request.get("source_policy")
    selected_sources = {str(item.get("source_id")) for item in selected}
    aws = trace.get("aws") if isinstance(trace.get("aws"), Mapping) else {}
    ncei = trace.get("ncei") if isinstance(trace.get("ncei"), Mapping) else {}
    if trace.get("policy") != policy:
        failures.append("source-discovery policy differs from the normalized request")
    if trace.get("injected_inventory"):
        failures.append("reviewed transfer plan contains injected rather than provider discovery evidence")
    if policy == "aws_only":
        if "ncei_long_term" in selected_sources or ncei.get("status") not in {None, "not_requested"}:
            failures.append("aws_only plan contains or attempted NCEI objects")
        if aws.get("status") != "success":
            failures.append("aws_only plan lacks successful AWS discovery evidence")
    elif policy == "ncei_only":
        if "aws_operational" in selected_sources or aws.get("status") not in {None, "not_requested"}:
            failures.append("ncei_only plan contains or attempted AWS objects")
        if ncei.get("status") != "success":
            failures.append("ncei_only plan lacks successful NCEI discovery evidence")
    elif policy == "aws_then_ncei":
        if aws.get("status") != "success":
            failures.append("aws_then_ncei plan lacks successful primary AWS discovery evidence")
        if "ncei_long_term" in selected_sources:
            before = trace.get("coverage_before_fallback")
            missing = before if isinstance(before, list) else (
                before.get("missing_times") if isinstance(before, Mapping) else None
            )
            scientific = trace.get("scientific_precedence_before_fallback")
            if trace.get("fallback_triggered") is not True:
                failures.append("NCEI selection lacks an explicit fallback trigger")
            if ncei.get("status") != "success":
                failures.append("NCEI selection lacks successful NCEI discovery evidence")
            if not ((isinstance(missing, list) and missing) or (isinstance(scientific, list) and scientific)):
                failures.append("NCEI fallback lacks an unresolved AWS semantic/scientific record")
    else:
        failures.append("source policy is invalid")
    return failures


def discover_objects(
    request: Mapping[str, Any],
    *,
    session: Any | None = None,
    endpoint: str = S3_ENDPOINT,
    with_trace: bool = False,
) -> Any:
    normalized = validate_request(request)
    policy = normalized["source_policy"]
    trace: dict[str, Any] = {
        "policy": policy, "aws": {"status": "not_requested", "object_count": 0},
        "ncei": {"status": "not_requested", "object_count": 0},
        "fallback_triggered": False, "fallback_reason": None,
    }
    if policy == "ncei_only":
        capable, reason = _ncei_capability(normalized)
        if not capable:
            raise ValueError(reason)
        ncei = _discover_source(normalized, "ncei_long_term", session=session)
        trace["ncei"] = {"status": "success", "object_count": len(ncei), "prefixes": _ncei_prefixes(normalized),
                         "coverage_probe": _coverage_probe_summary(ncei)}
        trace["coverage_after_fallback"] = _selection_probe(normalized, ncei)["missing_times"]
        return (ncei, trace) if with_trace else ncei
    # Any AWS listing exception propagates; it never triggers fallback.
    aws = _discover_source(normalized, "aws_operational", session=session)
    trace["aws"] = {"status": "success", "object_count": len(aws), "prefixes": discovery_prefixes(normalized)}
    before = _selection_probe(normalized, aws)
    unresolved = list(before["missing_times"])
    scientific = _scientific_fallback_times(normalized, aws)
    trace["scientific_precedence_before_fallback"] = scientific
    trace["coverage_before_fallback"] = unresolved
    combined = list(aws)
    if policy == "aws_then_ncei" and (unresolved or scientific):
        capable, reason = _ncei_capability(normalized)
        if capable:
            trace["fallback_triggered"] = True
            trace["fallback_reason"] = "AWS discovery succeeded but semantic coverage remained unresolved"
            ncei = _discover_source(normalized, "ncei_long_term", session=session)
            trace["ncei"] = {"status": "success", "object_count": len(ncei), "prefixes": _ncei_prefixes(normalized),
                             "coverage_probe": _coverage_probe_summary(ncei)}
            combined.extend(ncei)
        else:
            trace["ncei"] = {"status": "unsupported", "object_count": 0, "reason": reason}
            trace["fallback_reason"] = reason
    trace["coverage_after_fallback"] = _selection_probe(normalized, combined)["missing_times"]
    return (combined, trace) if with_trace else combined


def _preference(item: Mapping[str, Any]) -> tuple[int, int, int, str]:
    return (
        1 if item.get("naming") == "current_aggregate" else 0,
        1 if item.get("layout") == "daily" else 0,
        1 if item.get("source_id", "aws_operational") == "aws_operational" else 0,
        str(item.get("key", "")),
    )


def _overlaps(item: Mapping[str, Any], start: datetime, end: datetime) -> bool:
    # Aggregate interval bounds are conservative. Selection is by the discrete
    # nominal points so a prior cycle ending at 23:30 cannot leak into a
    # request that begins at 00:00.
    return any(start <= stamp < end for stamp in _nominal_item_times(item, str(item["product"])))


def _expected_times(start: datetime, end: datetime, step_seconds: int, phase_seconds: int = 0) -> list[datetime]:
    first_epoch = math.ceil((start.timestamp() - phase_seconds) / step_seconds) * step_seconds + phase_seconds
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
    return _expected_times(start, end, step, 1800 if product == "fields" else 0)


def select_objects(request: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    start = _parse_utc(request["start_utc"])
    end = _parse_utc(request["end_utc_exclusive"])
    candidates: list[dict[str, Any]] = []
    for raw in objects:
        item = dict(raw)
        if item.get("product") != request["product"] or item.get("guidance") != request["guidance"]:
            continue
        if request["guidance"] == "forecast":
            if item.get("run_time") != request["run_cycle_utc"]:
                continue
        if _overlaps(item, start, end):
            candidates.append(item)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in candidates:
        if item.get("aggregate"):
            identity = (item["product"], item["guidance"], item["run_time"])
        else:
            identity = (item["product"], item["guidance"], item.get("valid_time"))
        grouped.setdefault(identity, []).append(item)
    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for group in grouped.values():
        top_rank = max(_preference(item)[:3] for item in group)
        same_rank = [item for item in group if _preference(item)[:3] == top_rank]
        remote_identities = {
            (item.get("size"), _clean_etag(item.get("etag")), item.get("last_modified"))
            for item in same_rank
        }
        if len(same_rank) > 1 and len(remote_identities) > 1:
            raise RuntimeError("same-rank SJROFS semantic aliases have conflicting remote identities")
        winner = max(group, key=_preference)
        selected.append(winner)
        duplicates.extend(item for item in group if item is not winner)
    selected.sort(key=lambda x: (x["run_time"], x["key"]))

    expected = _expected_times(start, end, 3600 if request["product"] == "fields" else 360, 1800 if request["product"] == "fields" else 0)
    requested_grids = ["native"]
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
    if not selected and request["missing_policy"] == "error":
        raise RuntimeError("source inventory did not contain any matching SJROFS objects")
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
    raw_request = _read_json(request) if isinstance(request, (str, Path)) else request
    lineage = request_migration(raw_request)
    normalized = validate_request(raw_request)
    if objects is not None:
        discovered = [dict(item) for item in objects]
        trace = {"policy": normalized["source_policy"], "injected_inventory": True}
    else:
        discovered, trace = discover_objects(normalized, session=session, endpoint=endpoint, with_trace=True)
    report = {
        "schema_version": "sjrofs_inventory_v2",
        "created_utc": _iso(datetime.now(UTC)),
        "request": normalized,
        "request_lineage": lineage,
        "source_discovery": trace,
        "object_count": len(discovered),
        "objects": discovered,
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
    raw_request = _read_json(request) if isinstance(request, (str, Path)) else request
    lineage = request_migration(raw_request)
    normalized = validate_request(raw_request)
    if objects is not None:
        discovered = [dict(item) for item in objects]
        trace = {"policy": normalized["source_policy"], "injected_inventory": True}
    else:
        discovered, trace = discover_objects(normalized, session=session, endpoint=endpoint, with_trace=True)
    selection = select_objects(normalized, discovered)
    selected = [_decorate_source(item, str(item.get("source_id") or "aws_operational")) for item in selection["selected"]]
    incomplete = [item["key"] for item in selected if not isinstance(item.get("size"), int) or item["size"] <= 0 or not item.get("etag") or not item.get("last_modified")]
    total = sum(int(item.get("size", 0)) for item in selected)
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
        "schema_version": "sjrofs_download_estimate_v2",
        "created_utc": _iso(datetime.now(UTC)),
        "request": normalized,
        "request_lineage": lineage,
        "normalized_request_sha256": _canonical_json_sha256(normalized),
        "source_discovery": trace,
        "objects": selected,
        "object_count": len(selected),
        "selected_objects_sha256": _canonical_json_sha256(sorted(selected, key=lambda item: (str(item.get("source_id")), str(item.get("key"))))),
        "total_bytes": total,
        "total_gib": total / 1024**3,
        "incomplete_size_keys": incomplete,
        "missing_times": selection["missing_times"],
        "duplicate_times": selection["duplicate_times"],
        "duplicate_objects": selection["duplicate_objects"],
        "source_totals": {
            source_id: {
                "object_count": sum(item.get("source_id") == source_id for item in selected),
                "bytes": sum(int(item["size"]) for item in selected if item.get("source_id") == source_id),
            }
            for source_id in ("aws_operational", "ncei_long_term")
        },
        "nominal_time_count": selection["nominal_time_count"],
        "nominal_time_count_by_grid": selection["nominal_time_count_by_grid"],
        "coverage_note": selection["coverage_note"],
        "local_free_bytes": free,
        "required_free_bytes": required,
        "routing_decision": route,
        "routing_reason": decision,
        "kestrel_stage_hint": "/scratch/yhuang168/oma_external_data_connectors/sjrofs-fetcher/<run-id>",
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


def _validate_netcdf_payload(path: Path) -> None:
    """Open a completed payload before atomic promotion; size alone is insufficient."""
    netCDF4, _ = _netcdf_modules()
    try:
        with netCDF4.Dataset(path) as ds:
            if "time" not in ds.variables or not ds.dimensions:
                raise RuntimeError("payload lacks required time variable or dimensions")
    except Exception as exc:
        raise RuntimeError(f"completed payload is not a valid SJROFS NetCDF: {path}: {exc}") from exc


def _clean_etag(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _download_sidecar(destination: Path) -> Path:
    return destination.with_name(destination.name + ".download.json")


def _partial_sidecar(destination: Path) -> Path:
    return destination.with_name(destination.name + ".part.json")


def _destination_for_key(run_dir: Path, key: str) -> Path:
    prefix = "sjrofs/netcdf/"
    relative = key[len(prefix) :] if key.startswith(prefix) else Path(key).name
    parts = Path(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts) or Path(relative).is_absolute():
        raise ValueError(f"unsafe S3 key path: {key!r}")
    destination = run_dir / "cache" / "raw"
    for part in parts:
        destination /= part
    return destination


def _destination_for_object(run_dir: Path, item: Mapping[str, Any]) -> Path:
    return run_dir / "cache" / "raw" / archive_sources.cache_relpath(item)


def _legacy_aws_cache_result(item: Mapping[str, Any], destination: Path) -> dict[str, Any] | None:
    """Strictly bind and reuse a historical v1 AWS cache in place."""
    if item.get("source_id") != "aws_operational" or not destination.is_file():
        return None
    sidecar_path = _download_sidecar(destination)
    if not sidecar_path.is_file() or destination.stat().st_size != int(item["size"]):
        return None
    try:
        sidecar = _read_json(sidecar_path)
    except Exception:
        return None
    digest = _sha256(destination)
    if any((
        sidecar.get("schema_version") != "sjrofs_cached_object_v1",
        sidecar.get("key") != item.get("key"), sidecar.get("url") != item.get("url"),
        int(sidecar.get("size", -1)) != int(item["size"]),
        _clean_etag(sidecar.get("etag")) != _clean_etag(item.get("etag")),
        sidecar.get("last_modified") != item.get("last_modified"),
        sidecar.get("sha256") != digest,
    )):
        return None
    try:
        netCDF4, _ = _netcdf_modules()
        with netCDF4.Dataset(destination) as ds:
            if not ds.dimensions or not ds.variables or "time" not in ds.variables:
                return None
    except Exception:
        return None
    return {
        "key": item["key"], "url": item["url"], "local_path": str(destination.resolve()),
        "status": "cache_hit", "size": int(item["size"]), "etag": _clean_etag(item.get("etag")),
        "sha256": digest, "resumed": False, "resumed_from_bytes": 0, "retry_count": 0,
        "legacy_cache_reused": True, "cache_layout": "legacy_aws_v1", "source": dict(item),
        "source_id": "aws_operational", "source_identity": archive_sources.source_identity_digest(item),
    }


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
    expected_identity = archive_sources.source_identity_digest(item) if item.get("source_id") else None
    if (
        metadata.get("schema_version") != "sjrofs_cached_object_v2"
        or metadata.get("source_id") != item.get("source_id")
        or metadata.get("source_identity") != expected_identity
        or metadata.get("key") != item.get("key")
        or metadata.get("url") != item.get("url")
        or metadata.get("last_modified") != item.get("last_modified")
        or metadata.get("etag_semantics") != "opaque_provenance"
    ):
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
        "source_id": item.get("source_id"),
        "source_identity": expected_identity,
    }


def download_object(
    item: Mapping[str, Any],
    destination: str | Path,
    *,
    session: Any | None = None,
    timeout: float = 120.0,
    max_attempts: int = 4,
    chunk_size: int = 4 * 1024 * 1024,
    revalidate_remote: bool = False,
) -> dict[str, Any]:
    requests = _requests_module()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = _cache_result(item, destination)
    if cached is not None:
        return cached
    expected_size = int(item.get("size", -1))
    if expected_size <= 0:
        raise RuntimeError(f"object has no exact positive size: {item.get('key')}")
    expected_etag = _clean_etag(item.get("etag"))
    # V2 plans always carry a validated source_id. Preserve direct helper
    # compatibility for synthetic/offline callers that predate provider IDs.
    if item.get("source_id") is not None:
        archive_sources.validate_source_object("sjrofs", item, expected_source_id=str(item["source_id"]))
        if revalidate_remote:
            exact = [candidate for candidate in archive_sources.list_objects_v2(
                str(item["source_id"]), "sjrofs", str(item["key"]), session=session, max_keys=2,
            ) if candidate.get("key") == item.get("key")]
            if len(exact) != 1:
                raise RuntimeError("planned source object is no longer uniquely listed; replan before fetching")
            archive_sources.validate_remote_metadata(item, exact[0])
    partial = destination.with_name(destination.name + ".part")
    partial_metadata_path = _partial_sidecar(destination)
    partial_metadata = {
        "schema_version": "sjrofs_partial_object_v2",
        "key": item["key"],
        "url": item["url"],
        "size": expected_size,
        "etag": expected_etag,
        "source_id": item.get("source_id"),
        "source_identity": archive_sources.source_identity_digest(item) if item.get("source_id") else None,
        "last_modified": item.get("last_modified"),
    }
    if partial.exists():
        try:
            existing_partial_metadata = _read_json(partial_metadata_path)
        except Exception:
            existing_partial_metadata = None
        if not isinstance(existing_partial_metadata, Mapping) or any(
            existing_partial_metadata.get(name) != partial_metadata[name]
            for name in ("key", "url", "size", "etag", "source_id", "source_identity", "last_modified")
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
        _validate_netcdf_payload(partial)
        digest = _sha256(partial)
        os.replace(partial, destination)
        partial_metadata_path.unlink(missing_ok=True)
        metadata = {
            "schema_version": "sjrofs_cached_object_v2",
            "key": item["key"],
            "url": item["url"],
            "size": expected_size,
            "etag": expected_etag,
            "etag_is_multipart": "-" in expected_etag,
            "last_modified": item.get("last_modified"),
            "sha256": digest,
            "completed_utc": _iso(datetime.now(UTC)),
            "source_id": item.get("source_id"),
            "source_identity": archive_sources.source_identity_digest(item) if item.get("source_id") else None,
            "etag_semantics": "opaque_provenance",
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
            "source_id": item.get("source_id"),
            "source_identity": archive_sources.source_identity_digest(item),
        }
    client = session or requests.Session()
    own_client = session is None
    errors: list[str] = []
    try:
        for attempt in range(max_attempts):
            current = partial.stat().st_size if partial.exists() else 0
            headers = archive_sources.build_resume_headers(current)
            try:
                response = client.get(item["url"], headers=headers, stream=True, timeout=timeout)
                try:
                    response.raise_for_status()
                    archive_sources.validate_download_response(response, item, offset=current)
                    mode = "ab" if current else "wb"
                    with partial.open(mode) as stream:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                stream.write(chunk)
                finally:
                    if hasattr(response, "close"):
                        response.close()
                actual_size = partial.stat().st_size
                if actual_size != expected_size:
                    raise RuntimeError(f"downloaded size mismatch: {actual_size} != {expected_size}")
                _validate_netcdf_payload(partial)
                digest = _sha256(partial)
                os.replace(partial, destination)
                partial_metadata_path.unlink(missing_ok=True)
                metadata = {
                "schema_version": "sjrofs_cached_object_v2",
                "key": item["key"],
                "url": item["url"],
                "size": expected_size,
                "etag": expected_etag,
                "etag_is_multipart": "-" in expected_etag,
                "last_modified": item.get("last_modified"),
                "sha256": digest,
                "completed_utc": _iso(datetime.now(UTC)),
                "source_id": item.get("source_id"),
                "source_identity": archive_sources.source_identity_digest(item) if item.get("source_id") else None,
                "etag_semantics": "opaque_provenance",
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
                "source_id": item.get("source_id"),
                "source_identity": archive_sources.source_identity_digest(item) if item.get("source_id") else None,
                }
            except Exception as exc:  # preserve .part for a later retry/run
                errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                if attempt + 1 < max_attempts:
                    time.sleep(min(2**attempt, 4))
    finally:
        if own_client and hasattr(client, "close"):
            client.close()
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
    if not isinstance(request, (str, Path)):
        raise TypeError("fetch requires an existing reviewed plan path, not an in-memory mapping")
    plan_path = Path(request).resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"reviewed plan path does not exist: {plan_path}")
    estimate = _read_json(plan_path)
    if estimate.get("schema_version") != "sjrofs_download_estimate_v2":
        raise ValueError("fetch requires a reviewed sjrofs_download_estimate_v2 plan")
    normalized = validate_request(estimate["request"])
    run_path = Path(run_dir)
    selected = list(estimate.get("objects", []))
    if estimate.get("normalized_request_sha256") != _canonical_json_sha256(normalized):
        raise ValueError("reviewed plan request digest is invalid")
    selected_sha = _canonical_json_sha256(sorted(selected, key=lambda item: (str(item.get("source_id")), str(item.get("key")))))
    if estimate.get("selected_objects_sha256") != selected_sha:
        raise ValueError("reviewed plan selected-object digest is invalid")
    if int(estimate.get("object_count", -1)) != len(selected) or int(estimate.get("total_bytes", -1)) != sum(int(item["size"]) for item in selected):
        raise ValueError("reviewed plan object count or byte binding is invalid")
    fallback_failures = validate_fallback_decision(normalized, estimate.get("source_discovery") or {}, selected)
    if fallback_failures:
        raise ValueError("reviewed plan fallback decision is invalid: " + "; ".join(fallback_failures))
    for item in selected:
        archive_sources.validate_source_object("sjrofs", item, expected_source_id=str(item.get("source_id")))
    if estimate["routing_decision"] != "local" or _disk_free(run_path) <= 4 * int(estimate["total_bytes"]):
        raise RuntimeError(
            f"local fetch is not approved by reviewed estimate/current storage: {estimate['routing_decision']} "
            f"({estimate['routing_reason']})"
        )
    plan_sha256 = _sha256(plan_path)
    outcomes: list[dict[str, Any]] = []

    def transfer(item: Mapping[str, Any]) -> dict[str, Any]:
        destination = _destination_for_object(run_path, item)
        cached = _cache_result(item, destination)
        if cached is not None:
            return cached
        legacy = _legacy_aws_cache_result(item, _destination_for_key(run_path, str(item["key"])))
        if legacy is not None:
            return legacy
        return download_object(item, destination, session=session, revalidate_remote=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=normalized["max_workers"]) as pool:
        futures = [pool.submit(transfer, item) for item in selected]
        for future in concurrent.futures.as_completed(futures):
            outcomes.append(future.result())
    outcomes.sort(key=lambda x: str(x.get("key")))
    failures = [item for item in outcomes if item["status"] == "failed"]
    manifest = {
        "schema_version": "sjrofs_fetch_manifest_v2",
        "created_utc": _iso(datetime.now(UTC)),
        "request": normalized,
        "estimate_path": str(plan_path),
        "reviewed_plan_sha256": plan_sha256,
        "normalized_request_sha256": estimate["normalized_request_sha256"],
        "selected_objects_sha256": estimate["selected_objects_sha256"],
        "selected_object_count_binding": len(selected),
        "selected_total_bytes_binding": int(estimate["total_bytes"]),
        "outcomes": outcomes,
        "counts": {
            "objects": len(outcomes),
            "downloaded": sum(item["status"] == "downloaded" for item in outcomes),
            "cache_hits": sum(item["status"] == "cache_hit" for item in outcomes),
            "failed": len(failures),
            "resumed": sum(bool(item.get("resumed")) for item in outcomes),
        },
        "source_discovery": estimate.get("source_discovery"),
        "source_totals": estimate.get("source_totals"),
    }
    write_json_atomic(run_path / "fetch_manifest.json", manifest)
    if failures:
        raise RuntimeError(f"{len(failures)} SJROFS object transfers failed; inspect fetch_manifest.json")
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


def normalize_time(
    value: datetime, cadence_seconds: int, tolerance_seconds: float = 60.0,
    phase_seconds: int = 0,
) -> tuple[datetime, float]:
    epoch = value.timestamp()
    nominal_epoch = round((epoch - phase_seconds) / cadence_seconds) * cadence_seconds + phase_seconds
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
        normalized, adjustment = normalize_time(original, cadence, phase_seconds=1800 if product == "fields" else 0)
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


def _verified_manifest_outcomes(run_dir: Path, request: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest_path = run_dir / "fetch_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("v2 reviewed plan and fetch manifest are required for extraction")
    manifest = _read_json(manifest_path)
    plan_path = Path(str(manifest.get("estimate_path") or "")).resolve()
    if not plan_path.is_file():
        raise RuntimeError("fetch manifest reviewed plan path does not exist")
    plan = _read_json(plan_path)
    if plan.get("schema_version") != "sjrofs_download_estimate_v2" or manifest.get("schema_version") != "sjrofs_fetch_manifest_v2":
        raise RuntimeError("v2 reviewed plan and fetch manifest are required for extraction")
    normalized = validate_request(request)
    selected = list(plan.get("objects", []))
    selected_sha = _canonical_json_sha256(sorted(selected, key=lambda item: (str(item.get("source_id")), str(item.get("key")))))
    if plan.get("normalized_request_sha256") != _canonical_json_sha256(normalized) or plan.get("selected_objects_sha256") != selected_sha:
        raise RuntimeError("reviewed plan request/object binding is invalid")
    if manifest.get("reviewed_plan_sha256") != _sha256(plan_path) or manifest.get("selected_objects_sha256") != selected_sha:
        raise RuntimeError("fetch manifest reviewed-plan binding is invalid")
    expected = {f"{item.get('source_id')}:{item.get('key')}": item for item in selected}
    outcomes = list(manifest.get("outcomes", []))
    if len(outcomes) != len(selected) or any(item.get("status") not in {"downloaded", "cache_hit"} for item in outcomes):
        raise RuntimeError("fetch manifest is incomplete")
    raw_root = (run_dir / "cache" / "raw").resolve()
    for outcome in outcomes:
        item = expected.get(f"{outcome.get('source_id')}:{outcome.get('key')}")
        if item is None:
            raise RuntimeError("fetch manifest contains an unplanned source object")
        if outcome.get("source_identity") != archive_sources.source_identity_digest(item):
            raise RuntimeError("fetch outcome source identity differs from the reviewed plan")
        outcome_source = outcome.get("source")
        if not isinstance(outcome_source, Mapping) or _canonical_json_sha256(outcome_source) != _canonical_json_sha256(item):
            raise RuntimeError("fetch outcome source descriptor differs from the reviewed plan")
        path = Path(str(outcome.get("local_path", ""))).resolve()
        provider_path = raw_root / archive_sources.cache_relpath(item)
        uses_legacy = bool(outcome.get("legacy_cache_reused"))
        expected_path = _destination_for_key(run_dir, str(item["key"])).resolve() if uses_legacy else provider_path
        try:
            path.relative_to(raw_root)
        except ValueError as exc:
            raise RuntimeError(f"cache path escapes the run directory: {path}") from exc
        if path != expected_path or not path.is_file() or path.stat().st_size != int(item["size"]) or _sha256(path) != outcome.get("sha256"):
            raise RuntimeError(f"cache payload binding is invalid: {path}")
        if uses_legacy:
            if _legacy_aws_cache_result(item, path) is None:
                raise RuntimeError(f"legacy AWS cache binding is invalid: {path}")
        else:
            sidecar = _read_json(_download_sidecar(path))
            if any((
                sidecar.get("schema_version") != "sjrofs_cached_object_v2",
                sidecar.get("source_identity") != archive_sources.source_identity_digest(item),
                sidecar.get("key") != item.get("key"), sidecar.get("url") != item.get("url"),
                sidecar.get("last_modified") != item.get("last_modified"),
                _clean_etag(sidecar.get("etag")) != _clean_etag(item.get("etag")),
                sidecar.get("sha256") != outcome.get("sha256"),
            )):
                raise RuntimeError(f"cache sidecar binding is invalid: {path}")
    return outcomes


def inspect_request(request: Mapping[str, Any] | str | Path, run_dir: str | Path) -> dict[str, Any]:
    normalized = load_request(request) if isinstance(request, (str, Path)) else validate_request(request)
    run_path = Path(run_dir)
    paths = _manifest_paths(run_path)
    if not paths:
        raise RuntimeError("fetch_manifest.json contains no available downloaded files")
    files = [inspect_file(path, normalized["product"]) for path in paths]
    report = {
        "schema_version": "sjrofs_inspection_v1",
        "created_utc": _iso(datetime.now(UTC)),
        "request": normalized,
        "file_count": len(files),
        "files": files,
    }
    write_json_atomic(run_path / "inspection.json", report)
    return report


def efdc_layer_top_weights(sigma: Any):
    """Return layer fractions for EFDC layer-top sigma values in source order."""
    _, np = _netcdf_modules()
    values = np.asarray(sigma, dtype=float).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("sigma must contain at least two finite points")
    order = np.argsort(values)
    sorted_values = values[order]
    if abs(float(sorted_values[0])) > 1.0e-6:
        raise ValueError("EFDC layer-top sigma must begin near zero")
    if np.any(np.diff(sorted_values) <= 0) or np.any(sorted_values < 0) or np.any(sorted_values >= 1):
        raise ValueError("EFDC layer-top sigma must be unique values in [0, 1)")
    sorted_weights = np.diff(np.concatenate((sorted_values, [1.0])))
    if np.any(sorted_weights <= 0):
        raise ValueError("EFDC layer-top sigma produces non-positive layer weights")
    weights = np.empty_like(sorted_weights)
    weights[order] = np.abs(sorted_weights)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("EFDC layer-top weights have a non-positive sum")
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
        raise ValueError(f"EFDC fields file is missing geometry variables: {', '.join(missing)}")
    lon_var = ds.variables["lon"]
    if len(lon_var.dimensions) != 2:
        raise ValueError("lon must be a two-dimensional curvilinear coordinate")
    ydim, xdim = lon_var.dimensions
    arrays = {name: np.ma.asarray(ds.variables[name][:]).filled(np.nan) for name in required}
    shape = arrays["lon"].shape
    if arrays["lat"].shape != shape or arrays["mask"].shape != shape or arrays["depth"].shape != shape:
        raise ValueError("lon, lat, mask, and depth must have identical two-dimensional shapes")
    mask_finite = arrays["mask"][np.isfinite(arrays["mask"])]
    mask_values = set(float(x) for x in np.unique(mask_finite))
    if 5.0 not in mask_values or any(value not in {0.0, 5.0} and value >= 0.0 for value in mask_values):
        raise ValueError("SJROFS mask does not unambiguously use active-water code 5")
    wet = np.isclose(arrays["mask"], 5.0)
    negative_count = int(np.sum(np.isfinite(arrays["mask"]) & (arrays["mask"] < 0)))
    if negative_count > 1:
        raise ValueError("SJROFS mask contains more than the one permitted inactive padding sentinel")
    if not wet.any():
        raise ValueError("SJROFS mask has no wet cells")
    if shape == (105, 188) and int(wet.sum()) != 2210:
        raise ValueError("current 188x105 SJROFS grid does not contain exactly 2,210 active cells")
    if not np.isfinite(arrays["lon"][wet]).all() or not np.isfinite(arrays["lat"][wet]).all():
        raise ValueError("SJROFS wet cells contain non-finite coordinates")
    sigma = np.asarray(arrays["sigma"], dtype=float).reshape(-1)
    efdc_layer_top_weights(sigma)
    return {
        "lon": np.asarray(arrays["lon"], dtype=float),
        "lat": np.asarray(arrays["lat"], dtype=float),
        "mask": np.asarray(arrays["mask"], dtype=float),
        "wet": wet,
        "depth": np.asarray(arrays["depth"], dtype=float),
        "sigma": sigma,
        "ydim": ydim,
        "xdim": xdim,
        "inactive_negative_sentinel_count": negative_count,
    }


def _validate_dynamic_wet_footprint(ds: Any, geometry: Mapping[str, Any]) -> None:
    """Require hydrodynamic variables, not atmospheric forcing, to be wet-only."""
    _, np = _netcdf_modules()
    wet = np.asarray(geometry["wet"], dtype=bool)
    ydim, xdim = geometry["ydim"], geometry["xdim"]
    for name in ("zeta", "salt", "u", "v", "temp"):
        if name not in ds.variables:
            continue
        variable = ds.variables[name]
        dimensions = tuple(variable.dimensions)
        if "time" not in dimensions or ydim not in dimensions or xdim not in dimensions:
            continue
        values = np.ma.asarray(variable[:], dtype=float).filled(np.nan)
        y_axis, x_axis = dimensions.index(ydim), dimensions.index(xdim)
        order = [axis for axis in range(values.ndim) if axis not in (y_axis, x_axis)] + [y_axis, x_axis]
        horizontal = np.transpose(values, order)
        if np.any(np.isfinite(horizontal) & ~wet.reshape((1,) * (horizontal.ndim - 2) + wet.shape)):
            raise ValueError(f"hydrodynamic variable {name!r} contains valid values outside source mask==5 active cells")


def _assert_geometry(reference: Mapping[str, Any], candidate: Mapping[str, Any], path: Path) -> None:
    _, np = _netcdf_modules()
    for name in ("lon", "lat", "mask", "depth", "sigma"):
        left = np.asarray(reference[name])
        right = np.asarray(candidate[name])
        if left.shape != right.shape or not np.allclose(left, right, rtol=1e-6, atol=1e-7, equal_nan=True):
            raise RuntimeError(f"SJROFS geometry drift for {name} in {path}")


def _canonical_dynamic_dimensions(variable: Any, geometry: Mapping[str, Any]) -> tuple[str, ...]:
    dimensions = tuple(name for name in variable.dimensions if name != "time")
    ydim, xdim = geometry["ydim"], geometry["xdim"]
    if dimensions == (ydim, xdim):
        return ("y", "x")
    if dimensions == ("sigma", ydim, xdim):
        return ("sigma", "y", "x")
    raise ValueError(
        f"variable {variable.name!r} has unsupported EFDC dimensions {variable.dimensions}; "
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
    finite = np.isfinite(values)
    wet_only = variable.name in {"zeta", "salt", "u", "v", "temp"}
    if wet_only and values.ndim == 2 and np.any(finite & ~wet):
        raise ValueError(f"{variable.name!r} contains valid values outside source mask==5 active cells")
    if wet_only and values.ndim == 3 and np.any(finite & ~wet[None, :, :]):
        raise ValueError(f"{variable.name!r} contains valid values outside source mask==5 active cells")
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
        source = outcome["source"]
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
                            "source_id": source.get("source_id", outcome.get("source_id", "unknown")),
                            "source_url": source.get("url", outcome.get("url")),
                        }
                    )
    records: list[dict[str, Any]] = []
    for stamp, group in candidates.items():
        # For a boundary duplicate, the preceding (earlier) cycle owns the terminal record.
        winner = min(group, key=lambda item: (item["run_time"], str(item["key"])))
        records.append(winner)
    records.sort(key=lambda item: item["time"])
    expected = _expected_times(start, end, 3600, 1800)
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


def _decode_char_rows(values: Any) -> list[str]:
    """Decode fixed-width NetCDF character rows without lossy coercion."""
    _, np = _netcdf_modules()
    array = np.asanyarray(values)
    rows = array if array.ndim > 1 else array.reshape(1, -1)
    result: list[str] = []
    for row in rows:
        pieces = [
            value.decode("utf-8", errors="strict") if isinstance(value, bytes) else str(value)
            for value in row
        ]
        result.append("".join(pieces).replace("\x00", "").strip())
    return result


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
                "signature": {
                    key: _json_clean(getattr(variable, key, None))
                    for key in ("units", "standard_name", "positive", "missing_value")
                },
            }
        if "u" in requested and "v" in requested:
            _validate_vector_metadata(first_ds.variables["u"], first_ds.variables["v"])
        for path, ds in datasets.items():
            geometry = _geometry(ds)
            _assert_geometry(reference_geometry, geometry, path)
            _validate_dynamic_wet_footprint(ds, geometry)
            for name, schema in source_schema.items():
                if name not in ds.variables:
                    raise RuntimeError(f"schema drift: {name!r} is missing from {path}")
                variable = ds.variables[name]
                if tuple(variable.dimensions) != schema["source_dimensions"] or str(variable.dtype) != schema["dtype"]:
                    raise RuntimeError(f"schema drift for {name!r} in {path}")
                signature = {
                    key: _json_clean(getattr(variable, key, None))
                    for key in ("units", "standard_name", "positive", "missing_value")
                }
                if signature != schema["signature"]:
                    raise RuntimeError(f"metadata schema drift for {name!r} in {path}")
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
        if "zeta" in arrays:
            columns = reference_geometry["depth"][None, :, :] + arrays["zeta"]
            invalid_columns = reference_geometry["wet"][None, :, :] & (
                ~np.isfinite(columns) | (columns <= 0)
            )
            if np.any(invalid_columns):
                raise RuntimeError("SJROFS contains non-positive or missing wet water-column thickness")
        sigma = reference_geometry["sigma"]
        weights = efdc_layer_top_weights(sigma)
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
        for suffix in views:
            if f"salt_{suffix}" in derived:
                derived[f"salinity_{suffix}"] = derived[f"salt_{suffix}"].copy()
            if f"u_{suffix}" in derived:
                derived[f"eastward_velocity_{suffix}"] = derived[f"u_{suffix}"].copy()
            if f"v_{suffix}" in derived:
                derived[f"northward_velocity_{suffix}"] = derived[f"v_{suffix}"].copy()
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
        destination = compact_dir / "sjrofs_fields.nc"
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.unlink(missing_ok=True)
        fill_value = np.float32(-99999.0)
        with netCDF4.Dataset(temporary, "w", format="NETCDF4_CLASSIC") as output:
            output.createDimension("time", len(records))
            output.createDimension("sigma", len(sigma))
            output.createDimension("y", reference_geometry["lon"].shape[0])
            output.createDimension("x", reference_geometry["lon"].shape[1])
            source_width = max(1, max(len(str(record["key"]).encode("utf-8")) for record in records))
            archive_width = max(1, max(len(str(record["source_id"]).encode("utf-8")) for record in records))
            output.createDimension("source_key_strlen", source_width)
            output.createDimension("source_archive_strlen", archive_width)
            time_units_width = max(1, max(len(record["source_time_units"].encode("utf-8")) for record in records))
            calendar_width = max(1, max(len(record["source_time_calendar"].encode("utf-8")) for record in records))
            output.createDimension("source_time_units_strlen", time_units_width)
            output.createDimension("source_time_calendar_strlen", calendar_width)
            output.setncatts(
                {
                    "schema_version": COMPACT_SCHEMA_VERSION,
                    "title": "NOAA SJROFS EFDC compact fields",
                    "Conventions": "CF-1.10",
                    "source_system": "SJROFS",
                    "source_model": "EFDC",
                    "source_grid": "native",
                    "grid_type": "curvilinear",
                    "vector_components": "earth_relative",
                    "wet_mask_contract": "source mask equals 5",
                    "vertical_method": "efdc_layer_top_sigma_with_bed_edge_1",
                    "time_coverage_start": request["start_utc"],
                    "time_coverage_end_exclusive": request["end_utc_exclusive"],
                    "source_keys_json": json.dumps(sorted({str(record["key"]) for record in records})),
                    "source_archives_json": json.dumps(sorted({str(record["source_id"]) for record in records})),
                    "source_summary_json": json.dumps({
                        source_id: sum(record["source_id"] == source_id for record in records)
                        for source_id in sorted({record["source_id"] for record in records})
                    }, sort_keys=True),
                    "source_objects_json": json.dumps([
                        {"source_id": record["source_id"], "key": record["key"], "url": record["source_url"]}
                        for record in records
                    ], sort_keys=True),
                    "created_utc": _iso(datetime.now(UTC)),
                    "history": "Created by OMA sjrofs-fetcher",
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
            key_var.long_name = "source archive object key"
            _write_char_rows(key_var, [str(record["key"]) for record in records], source_width)
            archive_var = output.createVariable("source_archive", "S1", ("time", "source_archive_strlen"))
            archive_var.long_name = "source archive identifier"
            _write_char_rows(archive_var, [str(record["source_id"]) for record in records], archive_width)

            for name, data, units, long_name in (
                ("lon", reference_geometry["lon"], "degrees_east", "longitude"),
                ("lat", reference_geometry["lat"], "degrees_north", "latitude"),
                ("depth", reference_geometry["depth"], "m", "positive-down bathymetry"),
            ):
                variable = output.createVariable(name, "f4", ("y", "x"), zlib=True, complevel=4, shuffle=True, fill_value=fill_value)
                variable.units = units
                variable.long_name = long_name
                variable[:] = np.where(np.isfinite(data), data, fill_value).astype(np.float32)
            mask_var = output.createVariable("mask", "f4", ("y", "x"), zlib=True, complevel=4, shuffle=True)
            mask_var.setncatts({"long_name": "source EFDC activity mask", "active_water_code": 5.0})
            mask_var[:] = reference_geometry["mask"].astype(np.float32)
            wet_var = output.createVariable("wet_mask", "i1", ("y", "x"), zlib=True, complevel=4, shuffle=True)
            wet_var.setncatts({"long_name": "derived active water mask", "flag_values": np.asarray([0, 1], dtype=np.int8), "flag_meanings": "inactive active_water"})
            wet_var[:] = reference_geometry["wet"].astype(np.int8)
            sigma_var = output.createVariable("sigma", "f4", ("sigma",))
            sigma_var.setncatts({"long_name": "EFDC layer-top sigma fraction", "positive": "down", "units": "1", "bounds_method": "append bed edge 1.0"})
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
                elif name.startswith("salinity_"):
                    variable.setncatts({"long_name": name.replace("_", " "), "units": "1e-3", "standard_name": "sea_water_practical_salinity"})
                elif name.startswith("eastward_velocity_"):
                    variable.setncatts({"long_name": name.replace("_", " "), "units": "m s-1", "standard_name": "eastward_sea_water_velocity"})
                elif name.startswith("northward_velocity_"):
                    variable.setncatts({"long_name": name.replace("_", " "), "units": "m s-1", "standard_name": "northward_sea_water_velocity"})
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
            "source_records": [
                {"source_id": record["source_id"], "key": record["key"], "url": record["source_url"]}
                for record in records
            ],
            "source_summary": {
                source_id: sum(record["source_id"] == source_id for record in records)
                for source_id in sorted({record["source_id"] for record in records})
            },
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
    outcomes = _verified_manifest_outcomes(run_path, normalized)
    outputs = [_extract_grid(normalized, outcomes, "native", run_path)]
    report = {
        "schema_version": "sjrofs_extraction_manifest_v2",
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
    objects: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    start = _parse_utc(request["start_utc"])
    end = _parse_utc(request["end_utc_exclusive"])
    cadence = 3600 if request["product"] == "fields" else 360
    for outcome in outcomes:
        if outcome.get("status") not in {"downloaded", "cache_hit"}:
            continue
        source = outcome.get("source") or {}
        path = Path(outcome["local_path"])
        try:
            with netCDF4.Dataset(path) as ds:
                records = decode_times(ds, request["product"])
                objects.append(
                    {
                        "path": str(path),
                        "key": outcome.get("key"),
                        "count": len(records),
                        "times": records,
                        "run_time": _parse_utc(source["run_time"]),
                        "source_id": source.get("source_id", outcome.get("source_id")),
                    }
                )
                if request["product"] == "fields":
                    candidate = _geometry(ds)
                    _validate_dynamic_wet_footprint(ds, candidate)
                    if "u" in ds.variables or "v" in ds.variables:
                        if "u" not in ds.variables or "v" not in ds.variables:
                            raise RuntimeError("raw SJROFS velocity components are unpaired")
                        _validate_vector_metadata(ds.variables["u"], ds.variables["v"])
                    if reference is not None:
                        try:
                            _assert_geometry(reference, candidate, path)
                        except RuntimeError as exc:
                            critical.append(str(exc))
                    else:
                        reference = candidate
        except Exception as exc:
            critical.append(f"cannot inspect raw object {path}: {type(exc).__name__}: {exc}")
    summary: dict[str, Any] = {}
    expected = _expected_times(start, end, cadence, 1800 if request["product"] == "fields" else 0)
    for grid in ["native"]:
        candidates: dict[datetime, list[dict[str, Any]]] = {}
        for item in objects:
            for time_index, record in enumerate(item["times"]):
                if start <= record.normalized < end:
                    candidates.setdefault(record.normalized, []).append(
                        {"record": record, "run_time": item["run_time"], "key": item["key"],
                         "source_id": item["source_id"], "time_index": time_index}
                    )
        unique = sorted(candidates)
        missing = [stamp for stamp in expected if stamp not in candidates]
        duplicate_count = sum(max(0, len(group) - 1) for group in candidates.values())
        selected_records = []
        for stamp in unique:
            winner = min(candidates[stamp], key=lambda item: (item["run_time"], str(item["key"])))
            selected_records.append({
                "normalized_time_utc": _iso(stamp),
                "source_id": winner["source_id"],
                "source_key": winner["key"],
                "source_cycle_utc": _iso(winner["run_time"]),
                "source_time_index": winner["time_index"],
                "candidate_count": len(candidates[stamp]),
                "deduplication_rank": "earliest_cycle_terminal_before_following_cycle_initial",
            })
        nonmonotonic = any(right <= left for left, right in zip(unique, unique[1:]))
        summary[grid] = {
            "object_count": len(objects),
            "unique_requested_time_count": len(unique),
            "expected_time_count": len(expected),
            "missing_times": [_iso(stamp) for stamp in missing],
            "source_duplicate_record_count": duplicate_count,
            "deduplication": "preceding cycle terminal record wins",
            "selected_time_records": selected_records,
            "monotonic": not nonmonotonic,
            "first_time_utc": _iso(unique[0]) if unique else None,
            "last_time_utc": _iso(unique[-1]) if unique else None,
        }
        if not objects:
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
        if getattr(ds, "vertical_method", None) != "efdc_layer_top_sigma_with_bed_edge_1":
            critical.append(f"compact vertical method is invalid in {path}")
        try:
            compact_sources = json.loads(str(getattr(ds, "source_objects_json")))
        except Exception as exc:
            compact_sources = []
            critical.append(f"compact output source_objects_json is invalid in {path}: {exc}")
        expected_sources = [
            {"source_id": item.get("source_id"), "key": item.get("key"), "url": item.get("url")}
            for item in output.get("source_records", [])
        ]
        if compact_sources != expected_sources:
            critical.append(f"compact NetCDF source records differ from extraction manifest in {path}")
        try:
            compact_archives = json.loads(str(getattr(ds, "source_archives_json")))
            compact_summary = json.loads(str(getattr(ds, "source_summary_json")))
        except Exception as exc:
            compact_archives, compact_summary = [], {}
            critical.append(f"compact output global source provenance is invalid in {path}: {exc}")
        expected_summary = output.get("source_summary", {})
        if compact_archives != sorted(expected_summary):
            critical.append(f"compact source_archives_json differs from extraction manifest in {path}")
        if compact_summary != expected_summary:
            critical.append(f"compact source_summary_json differs from extraction manifest in {path}")
        report["source_records"] = compact_sources
        required_geometry = {
            "lon",
            "lat",
            "mask",
            "wet_mask",
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
            "source_archive",
        }
        absent_geometry = sorted(required_geometry - set(ds.variables))
        if absent_geometry:
            critical.append(f"compact output lacks required coordinates: {', '.join(absent_geometry)}")
            return report, critical, warnings
        archive_rows = _decode_char_rows(ds.variables["source_archive"][:])
        key_rows = _decode_char_rows(ds.variables["source_key"][:])
        if archive_rows != [str(item.get("source_id") or "") for item in output.get("source_records", [])]:
            critical.append(f"compact source_archive rows differ from extraction manifest in {path}")
        if key_rows != [str(item.get("key") or "") for item in output.get("source_records", [])]:
            critical.append(f"compact source_key rows differ from extraction manifest in {path}")
        lon = _finite_values(ds.variables["lon"])
        lat = _finite_values(ds.variables["lat"])
        mask = _finite_values(ds.variables["mask"])
        depth = _finite_values(ds.variables["depth"])
        sigma = np.asarray(ds.variables["sigma"][:], dtype=float)
        wet = np.isclose(_finite_values(ds.variables["wet_mask"]), 1.0)
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
            weights = efdc_layer_top_weights(sigma)
            geometry["sigma_weights"] = [float(value) for value in weights]
            geometry["sigma_weight_sum"] = float(weights.sum())
        except ValueError as exc:
            critical.append(f"invalid sigma coordinate in {path}: {exc}")
        report["geometry"] = geometry
        if lon.ndim != 2 or lat.shape != lon.shape or mask.shape != lon.shape or depth.shape != lon.shape:
            critical.append(f"compact EFDC geometry shapes are inconsistent in {path}")
        if 5.0 not in mask_values or not wet.any() or not np.array_equal(wet, np.isclose(mask, 5.0)):
            critical.append(f"compact EFDC mask is invalid in {path}")
        if geometry["coordinate_finite_wet_fraction"] < 1.0:
            critical.append(f"compact wet coordinates are non-finite in {path}")
        if geometry["positive_depth_wet_fraction"] < 1.0:
            critical.append(f"compact wet bathymetry is not strictly positive in {path}")
        if lon.shape == (105, 188) and geometry["wet_cells"] != 2210:
            critical.append(f"current compact grid does not contain exactly 2,210 wet cells in {path}")

        times = _compact_times(ds)
        expected_times = _expected_times(_parse_utc(request["start_utc"]), _parse_utc(request["end_utc_exclusive"]), 3600, 1800)
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
            if "salt" in requested:
                aliases = [f"salt_{suffix}", f"salinity_{suffix}"]
                if any(name not in ds.variables for name in aliases):
                    critical.append(f"compact output lacks salinity alias for {suffix} in {path}")
                else:
                    left, right = (_finite_values(ds.variables[name]) for name in aliases)
                    if not np.allclose(left, right, equal_nan=True):
                        critical.append(f"salinity alias is inconsistent for {suffix} in {path}")
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
                    for alias, raw in ((f"eastward_velocity_{suffix}", names[0]), (f"northward_velocity_{suffix}", names[1])):
                        if alias not in ds.variables or not np.allclose(
                            _finite_values(ds.variables[alias]), _finite_values(ds.variables[raw]), equal_nan=True
                        ):
                            critical.append(f"earth-relative velocity alias is missing or inconsistent for {suffix} in {path}")
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
            columns = depth[None, :, :] + zeta
            if np.any(wet[None, :, :] & (~np.isfinite(columns) | (columns <= 0))):
                critical.append(f"compact wet water-column thickness is invalid in {path}")
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
                axis.set_title(f"SJROFS {output.get('grid')} compact-grid health")
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
    estimate: Mapping[str, Any], manifest: Mapping[str, Any], run_dir: Path | None = None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    critical: list[str] = []
    warnings: list[str] = []
    if estimate.get("schema_version") == "sjrofs_download_estimate_v2":
        normalized = validate_request(estimate.get("request", {}))
        selected = list(estimate.get("objects", []))
        raw_outcomes = manifest.get("outcomes", [])
        if manifest.get("schema_version") != "sjrofs_fetch_manifest_v2":
            critical.append("fetch manifest schema is not sjrofs_fetch_manifest_v2")
        try:
            manifest_request = validate_request(manifest.get("request", {}))
        except Exception as exc:
            critical.append(f"fetch manifest request is invalid: {exc}")
        else:
            if _canonical_json_sha256(manifest_request) != _canonical_json_sha256(normalized):
                critical.append("fetch manifest request differs from reviewed plan request")
        if not isinstance(raw_outcomes, list) or any(not isinstance(item, Mapping) for item in raw_outcomes):
            critical.append("fetch manifest outcomes are not a valid list")
            raw_outcomes = []
        identities = [(str(item.get("source_id")), str(item.get("key"))) for item in raw_outcomes]
        if len(set(identities)) != len(identities):
            critical.append("fetch manifest contains duplicate source outcomes")
        if len(raw_outcomes) != len(selected):
            critical.append("fetch manifest outcome cardinality differs from reviewed plan")
        if int(manifest.get("selected_object_count_binding", -1)) != len(selected):
            critical.append("fetch manifest selected-object count binding is invalid")
        if int(manifest.get("selected_total_bytes_binding", -1)) != sum(int(item["size"]) for item in selected):
            critical.append("fetch manifest selected-byte binding is invalid")
        counts = manifest.get("counts") if isinstance(manifest.get("counts"), Mapping) else {}
        expected_counts = {
            "objects": len(raw_outcomes),
            "downloaded": sum(item.get("status") == "downloaded" for item in raw_outcomes),
            "cache_hits": sum(item.get("status") == "cache_hit" for item in raw_outcomes),
            "failed": sum(item.get("status") == "failed" for item in raw_outcomes),
            "resumed": sum(bool(item.get("resumed")) for item in raw_outcomes),
        }
        if any(counts.get(name) != value for name, value in expected_counts.items()):
            critical.append("fetch manifest counts do not match its exact outcomes")
        expected_selected_sha = _canonical_json_sha256(sorted(selected, key=lambda item: (str(item.get("source_id")), str(item.get("key")))))
        if estimate.get("normalized_request_sha256") != _canonical_json_sha256(normalized):
            critical.append("download estimate normalized-request binding is invalid")
        if estimate.get("selected_objects_sha256") != expected_selected_sha:
            critical.append("download estimate selected-object binding is invalid")
        critical.extend(validate_fallback_decision(
            normalized, estimate.get("source_discovery") or {}, selected
        ))
        expected_totals = {
            source_id: {
                "object_count": sum(item.get("source_id") == source_id for item in selected),
                "bytes": sum(int(item["size"]) for item in selected if item.get("source_id") == source_id),
            }
            for source_id in ("aws_operational", "ncei_long_term")
        }
        if estimate.get("source_totals") != expected_totals:
            critical.append("download estimate per-source object/byte totals are invalid")
        if manifest.get("source_totals") != expected_totals:
            critical.append("fetch manifest per-source object/byte totals differ from reviewed plan")
        estimate_path = Path(str(manifest.get("estimate_path", "")))
        if not estimate_path.is_file() or manifest.get("reviewed_plan_sha256") != _sha256(estimate_path):
            critical.append("fetch manifest reviewed-plan path/SHA binding is invalid")
        for field in ("normalized_request_sha256", "selected_objects_sha256"):
            if manifest.get(field) != estimate.get(field):
                critical.append(f"fetch manifest {field} differs from reviewed plan")
    expected = {item["key"]: item for item in estimate.get("objects", [])}
    raw_outcomes = manifest.get("outcomes", []) if isinstance(manifest.get("outcomes"), list) else []
    outcomes = {item["key"]: item for item in raw_outcomes if isinstance(item, Mapping) and item.get("key")}
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
        if run_dir is not None:
            raw_root = (run_dir / "cache" / "raw").resolve()
            resolved = path.resolve()
            try:
                resolved.relative_to(raw_root)
            except ValueError:
                critical.append(f"cache path escapes the run directory for {key}")
            if estimate.get("schema_version") == "sjrofs_download_estimate_v2":
                uses_legacy = bool(outcome.get("legacy_cache_reused"))
                expected_path = (
                    _destination_for_key(run_dir, str(source["key"])).resolve()
                    if uses_legacy else raw_root / archive_sources.cache_relpath(source)
                )
                if resolved != expected_path:
                    critical.append(f"cache path is not the provider-isolated planned path for {key}")
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
                expected_identity = archive_sources.source_identity_digest(source) if source.get("source_id") else None
                uses_legacy = bool(outcome.get("legacy_cache_reused"))
                if estimate.get("schema_version") != "sjrofs_download_estimate_v2":
                    if metadata.get("schema_version") != "sjrofs_cached_object_v1":
                        critical.append(f"legacy cache sidecar schema is invalid for {key}")
                elif uses_legacy:
                    if _legacy_aws_cache_result(source, path) is None:
                        critical.append(f"legacy AWS cache sidecar binding is invalid for {key}")
                elif metadata.get("schema_version") != "sjrofs_cached_object_v2":
                    critical.append(f"cache sidecar schema is not v2 for {key}")
                elif metadata.get("source_id") != source.get("source_id") or metadata.get("source_identity") != expected_identity:
                    critical.append(f"cache sidecar source identity mismatch for {key}")
                if metadata.get("last_modified") != source.get("last_modified") or metadata.get("url") != source.get("url"):
                    critical.append(f"cache sidecar source metadata mismatch for {key}")
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
        critical.append(f"fetch manifest contains {len(extra)} objects absent from the reviewed plan")
    return {"objects": rows, "expected_count": len(expected), "verified_count": len(rows)}, critical, warnings


def _delete_raw_cache(run_dir: Path, manifest: MutableMapping[str, Any]) -> dict[str, Any]:
    deleted: list[dict[str, Any]] = []
    for outcome in manifest.get("outcomes", []):
        path = Path(str(outcome.get("local_path", ""))).resolve()
        try:
            path.relative_to((run_dir / "cache" / "raw").resolve())
        except ValueError as exc:
            raise RuntimeError(f"refusing to delete cache outside run directory: {path}") from exc
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
        "recovery": "re-download from the provider-bound source in download_estimate.json",
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
    manifest_path = run_path / "fetch_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("fetch_manifest.json is required for health checking")
    manifest = _read_json(manifest_path)
    estimate_path = Path(str(manifest.get("estimate_path") or "")).resolve()
    if not estimate_path.is_file():
        raise RuntimeError("fetch manifest reviewed plan path does not exist")
    estimate = _read_json(estimate_path)
    if _canonical_json_sha256(normalized) != _canonical_json_sha256(validate_request(estimate.get("request", {}))):
        raise RuntimeError("health request differs from the reviewed plan request")
    if _canonical_json_sha256(normalized) != _canonical_json_sha256(validate_request(manifest.get("request", {}))):
        raise RuntimeError("health request differs from the fetch manifest request")
    transfer, critical, warnings = _verify_transfers(estimate, manifest, run_path)
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
            if extraction.get("schema_version") != "sjrofs_extraction_manifest_v2":
                critical.append("extraction manifest schema is invalid")
            try:
                extraction_request = validate_request(extraction.get("request", {}))
            except Exception as exc:
                critical.append(f"extraction manifest request is invalid: {exc}")
            else:
                if _canonical_json_sha256(extraction_request) != _canonical_json_sha256(normalized):
                    critical.append("extraction manifest request differs from the reviewed plan request")
            for compact_output in extraction.get("outputs", []):
                check, findings, cautions = _check_compact(
                    compact_output,
                    normalized,
                    Path(plots_dir) if plots_dir is not None else None,
                )
                compact_checks.append(check)
                critical.extend(findings)
                warnings.extend(cautions)
                compact_path = Path(str(compact_output.get("path", "")))
                if not compact_path.is_file() or compact_output.get("sha256") != _sha256(compact_path):
                    critical.append("extraction manifest compact-output hash binding is invalid")
            expected_outputs = 1
            if len(compact_checks) != expected_outputs:
                critical.append(f"expected {expected_outputs} compact grid outputs, found {len(compact_checks)}")
            planned_sources = {
                (str(item.get("source_id")), str(item.get("key")), str(item.get("url")))
                for item in estimate.get("objects", [])
            }
            extracted_sources = {
                (str(item.get("source_id")), str(item.get("key")), str(item.get("url")))
                for output_item in extraction.get("outputs", [])
                for item in output_item.get("source_records", [])
            }
            if extracted_sources != planned_sources:
                critical.append("compact extraction source records do not exactly match the reviewed plan")
    deletion: dict[str, Any] | None = None
    passed_before_deletion = not critical
    if passed_before_deletion and normalized["cache_policy"] == "delete_after_extract":
        deletion = _delete_raw_cache(run_path, manifest)
        if extraction is not None:
            extraction["raw_cache_deleted"] = True
            extraction["raw_cache_deletion"] = deletion
            write_json_atomic(run_path / "extraction_manifest.json", extraction)
    report = {
        "schema_version": "sjrofs_health_check_v2",
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
    parser.add_argument("--request", required=True, help="Path to a sjrofs_request_v2 (or migratable v1) JSON file")
    parser.add_argument("--run-dir", required=True, help="Run/evidence directory outside the skill package")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory_parser = subparsers.add_parser("inventory", help="List matching public S3 objects")
    _add_common_arguments(inventory_parser)
    plan_parser = subparsers.add_parser("plan", help="Select objects and write an exact storage estimate")
    _add_common_arguments(plan_parser)
    plan_parser.add_argument("--output", help="Optional estimate JSON path")
    fetch_parser = subparsers.add_parser("fetch", help="Fetch the exact objects in a reviewed v2 plan")
    fetch_parser.add_argument("--plan", required=True, help="Reviewed sjrofs_download_estimate_v2 JSON path")
    fetch_parser.add_argument("--run-dir", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="Inspect cached NetCDF metadata")
    _add_common_arguments(inspect_parser)
    extract_parser = subparsers.add_parser("extract", help="Concatenate and derive compact EFDC fields")
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
            result = fetch_request(args.plan, args.run_dir)
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
