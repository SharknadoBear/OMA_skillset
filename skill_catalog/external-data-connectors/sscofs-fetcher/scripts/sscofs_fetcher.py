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

try:
    from . import ofs_archive_sources as archive_sources
except ImportError:
    import ofs_archive_sources as archive_sources


UTC = timezone.utc
SCHEMA_VERSION = "sscofs_request_v2"
LEGACY_SCHEMA_VERSION = "sscofs_request_v1"
SOFTWARE_VERSION = "2.0.0"
BUCKET = "noaa-nos-ofs-pds"
S3_ROOT = f"https://{BUCKET}.s3.amazonaws.com"
LIST_URL = f"{S3_ROOT}/"
ARCHIVE_ROOT = "sscofs/netcdf"
CYCLE_HOURS = {3, 9, 15, 21}
PRODUCTS = {"fields", "stations", "regulargrid"}
GUIDANCE = {"nowcast", "forecast"}
SOURCE_POLICIES = {"aws_then_ncei", "aws_only", "ncei_only"}
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


def request_migration(mapping: Mapping[str, Any]) -> dict[str, Any]:
    original = mapping.get("schema_version", mapping.get("schema"))
    return {
        "original_schema_version": original,
        "normalized_schema_version": SCHEMA_VERSION,
        "migrated": original == LEGACY_SCHEMA_VERSION,
        "defaults_applied": ["source_policy=aws_then_ncei"]
        if original == LEGACY_SCHEMA_VERSION and "source_policy" not in mapping else [],
    }


def validate_request(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Validate v2 or migrate a v1 request to normalized v2."""

    allowed = {
        "schema_version", "schema", "start_utc", "end_utc_exclusive", "product",
        "guidance", "run_cycle_utc", "variables", "vertical_views", "missing_policy",
        "cache_policy", "max_workers", "source_policy",
    }
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown request properties: {', '.join(unknown)}")
    schema = mapping.get("schema_version", mapping.get("schema"))
    if schema not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r} or {LEGACY_SCHEMA_VERSION!r}")
    if schema == LEGACY_SCHEMA_VERSION and "source_policy" in mapping:
        raise ValueError("source_policy is a v2-only field; omit it from v1 requests and migrate first")
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
    if guidance == "nowcast" and run_cycle is not None:
        raise ValueError("run_cycle_utc is permitted only for forecast requests")

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
    source_policy = str(mapping.get("source_policy", "aws_then_ncei")).lower()
    if source_policy not in SOURCE_POLICIES:
        raise ValueError(f"source_policy must be one of {sorted(SOURCE_POLICIES)}")

    result: dict[str, Any] = {
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
        result["run_cycle_utc"] = _iso(run_cycle)
    if product == "fields":
        result["variables"] = variables
        result["vertical_views"] = normalized_views
    return result


def _layout_for_key(key: str) -> str:
    if key.startswith(archive_sources.get_source_descriptor("ncei_long_term", "sscofs")["root_prefix"]):
        return "ncei_monthly"
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
    results: list[dict[str, Any]] = []
    for raw in archive_sources.list_objects_v2(
        "aws_operational", "sscofs", prefix, session=session,
        timeout=timeout, max_keys=max_keys,
    ):
        parsed = parse_object_key(raw["key"])
        if parsed is not None:
            results.append({**raw, **parsed})
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


def _ncei_capability(request: Mapping[str, Any]) -> tuple[bool, str | None]:
    if request["product"] == "regulargrid":
        return False, "NCEI does not provide verified SSCOFS regular-grid fallback"
    if request["product"] == "fields" and request["guidance"] == "forecast":
        return False, "NCEI does not provide verified SSCOFS field-forecast fallback"
    return True, None


def _ncei_prefixes(request: Mapping[str, Any]) -> list[str]:
    descriptor = archive_sources.get_source_descriptor("ncei_long_term", "sscofs")
    if request.get("guidance") == "forecast" and request.get("run_cycle_utc"):
        center = _parse_utc(request["run_cycle_utc"])
        first, last = center - timedelta(days=1), center + timedelta(days=1)
    else:
        first = _parse_utc(request["start_utc"]) - timedelta(days=1)
        last = _parse_utc(request["end_utc_exclusive"]) + timedelta(days=1)
    return [f"{descriptor['root_prefix']}{month:%Y/%m}/" for month in _month_starts(first, last)]


def _discover_source(
    request: Mapping[str, Any], source_id: str, *, session: Any | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    prefixes = _discovery_prefixes(request) if source_id == "aws_operational" else _ncei_prefixes(request)
    found: dict[str, dict[str, Any]] = {}
    for prefix in prefixes:
        for raw in archive_sources.list_objects_v2(
            source_id, "sscofs", prefix, session=session, timeout=timeout,
        ):
            parsed = parse_object_key(raw["key"])
            if parsed is None:
                continue
            item = {**raw, **parsed, "source_id": source_id}
            item["naming_era"] = item.get("naming")
            item["semantic_identity"] = {
                "model": "sscofs", "product": item.get("product"),
                "guidance": item.get("guidance"), "run_time_utc": item.get("run_time_utc"),
                "valid_time_utc": item.get("valid_time_utc"), "lead_hour": item.get("lead_hour"),
            }
            item["semantic_identity_digest"] = archive_sources.semantic_identity_digest(item["semantic_identity"])
            archive_sources.validate_source_object("sscofs", item, expected_source_id=source_id)
            found[item["key"]] = item
    return sorted(found.values(), key=lambda item: item["key"])


def _selection_without_missing_error(request: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probe = dict(request)
    probe["missing_policy"] = "skip"
    return select_objects(probe, objects)


def _scientific_fallback_times(
    request: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Find lower-ranked records that must not suppress an archive fallback."""
    if request["guidance"] != "nowcast":
        return []
    if request["product"] != "stations":
        start = _parse_utc(request["start_utc"])
        end = _parse_utc(request["end_utc_exclusive"])
        relevant = [
            item for item in objects
            if item.get("product") == request["product"]
            and item.get("guidance") == "nowcast"
            and item.get("valid_time_utc")
            and start <= _parse_utc(str(item["valid_time_utc"])) < end
        ]
        preferred_times = {
            _iso(_parse_utc(str(item["valid_time_utc"])))
            for item in relevant if item.get("lead_hour") == 6
        }
        # n000 is the following cycle's duplicate of the preceding cycle's
        # terminal n006.  Query the long-term archive only when an in-window
        # n000 is all AWS supplied for that valid time.  Broad discovery also
        # returns records outside the request window; those must never trigger
        # fallback for an otherwise complete request.
        return sorted({
            _iso(_parse_utc(str(item["valid_time_utc"]))) or ""
            for item in relevant
            if item.get("lead_hour") == 0
            and _iso(_parse_utc(str(item["valid_time_utc"]))) not in preferred_times
        })
    start = _parse_utc(request["start_utc"])
    end = _parse_utc(request["end_utc_exclusive"])
    stations = [
        item for item in objects
        if item.get("product") == "stations" and item.get("guidance") == "nowcast"
    ]
    unresolved: list[str] = []
    stamp = start
    while stamp < end:
        covering = []
        for item in stations:
            run = _parse_utc(item["run_time_utc"])
            if run - timedelta(hours=6) <= stamp <= run:
                covering.append(item)
        if (
            covering
            and any(_parse_utc(item["run_time_utc"]) - timedelta(hours=6) == stamp for item in covering)
            and not any(_parse_utc(item["run_time_utc"]) == stamp for item in covering)
        ):
            unresolved.append(_iso(stamp) or "")
        stamp += timedelta(minutes=6)
    return unresolved


def validate_fallback_decision(
    request: Mapping[str, Any], trace: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Validate source policy, ordered discovery, and the reason for fallback."""
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
            missing = before.get("missing_times") if isinstance(before, Mapping) else None
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
    request: Mapping[str, Any], *, session: Any | None = None, timeout: float = 60.0,
    with_trace: bool = False,
) -> Any:
    """Discover AWS first and query NCEI only for unresolved semantic coverage."""

    normalized = validate_request(request)
    policy = normalized["source_policy"]
    trace: dict[str, Any] = {
        "policy": policy, "aws": {"status": "not_requested", "object_count": 0},
        "ncei": {"status": "not_requested", "object_count": 0},
        "fallback_triggered": False, "fallback_reason": None,
        "coverage_before_fallback": None, "coverage_after_fallback": None,
    }
    if policy == "ncei_only":
        capable, reason = _ncei_capability(normalized)
        if not capable:
            raise ValueError(reason)
        ncei = _discover_source(normalized, "ncei_long_term", session=session, timeout=timeout)
        trace["ncei"] = {"status": "success", "object_count": len(ncei), "prefixes": _ncei_prefixes(normalized)}
        after = _selection_without_missing_error(normalized, ncei)
        trace["coverage_after_fallback"] = {"missing_times": after["missing_times"], "selected_count": after["selected_count"]}
        return (ncei, trace) if with_trace else ncei

    # Exceptions intentionally propagate: an AWS listing error is never empty coverage.
    aws = _discover_source(normalized, "aws_operational", session=session, timeout=timeout)
    trace["aws"] = {"status": "success", "object_count": len(aws), "prefixes": _discovery_prefixes(normalized)}
    before = _selection_without_missing_error(normalized, aws)
    unresolved = list(before["missing_times"])
    scientific = _scientific_fallback_times(normalized, aws)
    trace["scientific_precedence_before_fallback"] = scientific
    trace["coverage_before_fallback"] = {"missing_times": unresolved, "selected_count": before["selected_count"]}
    combined = list(aws)
    if policy == "aws_then_ncei" and (unresolved or scientific):
        capable, reason = _ncei_capability(normalized)
        if capable:
            trace["fallback_triggered"] = True
            trace["fallback_reason"] = "AWS discovery succeeded but semantic coverage remained unresolved"
            ncei = _discover_source(normalized, "ncei_long_term", session=session, timeout=timeout)
            trace["ncei"] = {"status": "success", "object_count": len(ncei), "prefixes": _ncei_prefixes(normalized)}
            combined.extend(ncei)
        else:
            trace["ncei"] = {"status": "unsupported", "object_count": 0, "reason": reason}
            trace["fallback_reason"] = reason
    after = _selection_without_missing_error(normalized, combined)
    trace["coverage_after_fallback"] = {"missing_times": after["missing_times"], "selected_count": after["selected_count"]}
    return (combined, trace) if with_trace else combined


def _preference(item: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    lead = item.get("lead_hour")
    return (
        # At a cycle boundary n006 is the continuous nowcast record and n000
        # is its duplicate in the following run.  Guidance continuity outranks
        # which archive layout happens to contain the copy.
        1 if lead == 6 else (0 if lead == 0 else -1),
        1 if item.get("source_id", "aws_operational") == "aws_operational" else 0,
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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_id(item: Mapping[str, Any]) -> str:
    return str(item.get("source_id") or "aws_operational")


def _decorate_object(item: Mapping[str, Any], source_id: str = "aws_operational") -> dict[str, Any]:
    descriptor = archive_sources.get_source_descriptor(source_id, "sscofs")
    result = {**descriptor, **dict(item), "source_id": source_id}
    if str(result.get("key", "")).startswith(descriptor["root_prefix"]):
        result["url"] = archive_sources.canonical_object_url(source_id, "sscofs", str(result["key"]))
    result.setdefault("size", _object_size(result))
    result.setdefault("size_bytes", int(result["size"]))
    result.setdefault("naming_era", result.get("naming"))
    result["semantic_identity"] = {
        "model": "sscofs", "product": result.get("product"), "guidance": result.get("guidance"),
        "run_time_utc": result.get("run_time_utc"), "valid_time_utc": result.get("valid_time_utc"),
        "lead_hour": result.get("lead_hour"),
    }
    identity_digest = archive_sources.source_identity_digest(result)
    result["source_identity"] = identity_digest
    result["source_identity_digest"] = identity_digest
    return result


def _request_lineage(request: Mapping[str, Any]) -> dict[str, Any]:
    return request_migration(request)


def inventory_request(
    request: Mapping[str, Any], *, objects: Sequence[Mapping[str, Any]] | None = None,
    session: Any | None = None,
) -> dict[str, Any]:
    lineage = request_migration(request)
    normalized = validate_request(request)
    if objects is not None:
        found = [dict(item) for item in objects]
        trace = {"policy": normalized["source_policy"], "injected_inventory": True}
    else:
        found, trace = discover_objects(normalized, session=session, with_trace=True)
    relevant = [
        dict(item) for item in found
        if item.get("product") == normalized["product"] and item.get("guidance") == normalized["guidance"]
    ]
    return {
        "schema_version": "sscofs_inventory_v2",
        "generated_utc": _iso(datetime.now(UTC)),
        "request_lineage": lineage,
        "source_discovery": trace,
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

    lineage = request_migration(request)
    normalized = validate_request(request)
    if objects is not None:
        found = [dict(item) for item in objects]
        trace = {"policy": normalized["source_policy"], "injected_inventory": True}
    else:
        found, trace = discover_objects(normalized, session=session, with_trace=True)
    selection = select_objects(normalized, found)
    selected = [_decorate_object(item, _source_id(item)) for item in selection["objects"]]
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
        "schema_version": "sscofs_download_estimate_v2",
        "generated_utc": _iso(datetime.now(UTC)),
        "software_version": SOFTWARE_VERSION,
        "request": normalized,
        "request_lineage": lineage,
        "normalized_request_sha256": _canonical_sha256(normalized),
        "source_discovery": trace,
        "objects": selected,
        "selected_object_count": len(selected),
        "selected_objects_sha256": _canonical_sha256(sorted(selected, key=lambda item: (_source_id(item), str(item.get("key"))))),
        "candidate_object_count": selection["candidate_count"],
        "total_bytes": total,
        "total_gib": total / 1024 ** 3,
        "missing_times": selection["missing_times"],
        "duplicate_records": selection["duplicate_records"],
        "source_totals": {
            source_id: {
                "object_count": sum(_source_id(item) == source_id for item in selected),
                "bytes": sum(_object_size(item) for item in selected if _source_id(item) == source_id),
            }
            for source_id in ("aws_operational", "ncei_long_term")
        },
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


def _partial_sidecar(destination: Path) -> Path:
    return destination.with_name(destination.name + ".part.json")


def _legacy_aws_destination(run_dir: Path, item: Mapping[str, Any]) -> Path:
    """Return the flat v1 AWS cache location without changing it in place."""
    return run_dir / "cache" / "raw" / Path(str(item.get("key", ""))).name


def _legacy_aws_cache_result(
    item: Mapping[str, Any], destination: Path, *, validate_netcdf: bool = False,
) -> dict[str, Any] | None:
    """Strictly bind and reuse a historical v1 AWS cache in its original path."""
    if _source_id(item) != "aws_operational" or not destination.is_file():
        return None
    sidecar_path = _download_sidecar(destination)
    if not sidecar_path.is_file() or destination.stat().st_size != _object_size(item):
        return None
    try:
        sidecar = _read_json(sidecar_path)
    except Exception:
        return None
    digest = _sha256(destination)
    if any((
        sidecar.get("schema_version") != "sscofs_cached_object_v1",
        sidecar.get("key") != item.get("key"), sidecar.get("url") != item.get("url"),
        int(sidecar.get("size_bytes", sidecar.get("size", -1))) != _object_size(item),
        _clean_etag(sidecar.get("etag")) != _clean_etag(item.get("etag")),
        sidecar.get("last_modified") != item.get("last_modified"),
        sidecar.get("sha256") != digest,
    )):
        return None
    if validate_netcdf:
        try:
            _validate_netcdf_payload(destination, item)
        except RuntimeError:
            return None
    return {
        "key": item["key"], "url": item["url"], "local_path": str(destination.resolve()),
        "status": "cache_hit", "cache_hit": True, "legacy_cache_reused": True,
        "cache_layout": "legacy_aws_v1", "size_bytes": _object_size(item),
        "etag": _clean_etag(item.get("etag")), "sha256": digest, "retries": 0,
        "resumed_bytes": 0, "valid_time_utc": item.get("valid_time_utc"),
        "run_time_utc": item.get("run_time_utc"), "product": item.get("product"),
        "guidance": item.get("guidance"), "source_id": "aws_operational",
        "source_identity": archive_sources.source_identity_digest(item), "source": dict(item),
    }


def _validate_netcdf_payload(path: Path, item: Mapping[str, Any]) -> None:
    try:
        from netCDF4 import Dataset
        with Dataset(path) as ds:
            if not ds.dimensions or not ds.variables:
                raise RuntimeError("NetCDF payload has no dimensions or variables")
            if item.get("product") == "fields" and not ({"time", "Times"} & set(ds.variables)):
                raise RuntimeError("SSCOFS fields payload has no time coordinate")
    except Exception as exc:
        raise RuntimeError(f"downloaded object is not a usable SSCOFS NetCDF: {exc}") from exc


def _cache_result(item: Mapping[str, Any], destination: Path, *, validate_netcdf: bool = False) -> dict[str, Any] | None:
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
    expected_identity = archive_sources.source_identity_digest(item)
    if sidecar and (
        sidecar.get("schema_version") != "sscofs_cached_object_v2"
        or sidecar.get("source_id") != _source_id(item)
        or sidecar.get("source_identity") != expected_identity
        or sidecar.get("key") != item.get("key")
        or sidecar.get("url") != item.get("url")
        or sidecar.get("last_modified") != item.get("last_modified")
        or sidecar.get("etag_semantics") != "opaque_provenance"
        or
        int(sidecar.get("size_bytes", -1)) != _object_size(item)
        or _clean_etag(sidecar.get("etag")) != expected_etag
    ):
        return None
    recorded_digest = str(sidecar.get("sha256", ""))
    digest = _sha256(destination)
    if len(recorded_digest) == 64 and digest.lower() != recorded_digest.lower():
        return None
    if validate_netcdf:
        try:
            _validate_netcdf_payload(destination, item)
        except RuntimeError:
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
        "source_id": _source_id(item),
        "source_identity": expected_identity,
        "source": dict(item),
    }
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
    validate_netcdf: bool = False,
) -> dict[str, Any]:
    """Download one object atomically with safe range resumption."""

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = _cache_result(item, target, validate_netcdf=validate_netcdf)
    if cached is not None:
        return cached
    expected_size = _object_size(item)
    if expected_size <= 0:
        raise ValueError(f"Object {item.get('key')} has an invalid size")
    expected_etag = _clean_etag(item.get("etag"))
    archive_sources.validate_source_object("sscofs", item, expected_source_id=_source_id(item))
    part = target.with_name(target.name + ".part")
    part_sidecar = _partial_sidecar(target)
    partial_metadata = {
        "schema_version": "sscofs_partial_object_v2",
        "source_id": _source_id(item), "source_identity": archive_sources.source_identity_digest(item),
        "key": item.get("key"), "url": item.get("url"), "size_bytes": expected_size,
        "etag": expected_etag, "last_modified": item.get("last_modified"),
    }
    if part.exists():
        try:
            previous = _read_json(part_sidecar)
        except Exception:
            previous = None
        if previous != partial_metadata:
            part.unlink(missing_ok=True)
            part_sidecar.unlink(missing_ok=True)
    if part.exists() and part.stat().st_size > expected_size:
        part.unlink()
    if not part.exists():
        write_json_atomic(part_sidecar, partial_metadata)
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
                headers = archive_sources.build_resume_headers(start)
                response = session.get(item["url"], headers=headers, stream=True, timeout=timeout)
                response.raise_for_status()
                archive_sources.validate_download_response(response, item, offset=start)
                mode = "ab" if start else "wb"
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
                if validate_netcdf:
                    _validate_netcdf_payload(part, item)
                os.replace(part, target)
                part_sidecar.unlink(missing_ok=True)
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
                    "source_id": _source_id(item),
                    "source_identity": archive_sources.source_identity_digest(item),
                    "source": dict(item),
                }
                write_json_atomic(_download_sidecar(target), {
                    **result,
                    "schema_version": "sscofs_cached_object_v2",
                    "size_bytes": expected_size,
                    "last_modified": item.get("last_modified"),
                    "etag_semantics": "opaque_provenance",
                })
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
    request_or_plan: Mapping[str, Any] | str | Path,
    run_dir: str | Path,
    *,
    max_retries: int = 4,
) -> dict[str, Any]:
    """Fetch all objects in a request/plan and write ``fetch_manifest.json``."""

    if not isinstance(request_or_plan, (str, Path)):
        raise TypeError("fetch requires an existing reviewed plan path, not an in-memory mapping")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    plan_path = Path(request_or_plan).resolve()
    if not plan_path.is_file():
        raise FileNotFoundError(f"reviewed plan path does not exist: {plan_path}")
    plan = _read_json(plan_path)
    if plan.get("schema_version") != "sscofs_download_estimate_v2":
        raise ValueError("fetch requires a reviewed sscofs_download_estimate_v2 plan")
    request = validate_request(plan["request"])
    objects = list(plan.get("objects", []))
    if plan.get("normalized_request_sha256") != _canonical_sha256(request):
        raise ValueError("reviewed plan request digest is invalid")
    expected_objects_sha = _canonical_sha256(sorted(objects, key=lambda item: (_source_id(item), str(item.get("key")))))
    if plan.get("selected_objects_sha256") != expected_objects_sha:
        raise ValueError("reviewed plan selected-object digest is invalid")
    if int(plan.get("selected_object_count", -1)) != len(objects):
        raise ValueError("reviewed plan object count is invalid")
    if int(plan.get("total_bytes", -1)) != sum(_object_size(item) for item in objects):
        raise ValueError("reviewed plan exact byte total is invalid")
    fallback_failures = validate_fallback_decision(request, plan.get("source_discovery") or {}, objects)
    if fallback_failures:
        raise ValueError("reviewed plan fallback decision is invalid: " + "; ".join(fallback_failures))
    for item in objects:
        archive_sources.validate_source_object("sscofs", item, expected_source_id=_source_id(item))
    free_now = _disk_free(run_path)
    if plan.get("routing_decision") != "local" or free_now <= 4 * int(plan["total_bytes"]):
        raise RuntimeError("local fetch is not approved by the reviewed plan and current four-times-free-space gate")
    raw_dir = run_path / "cache" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    plan_sha256 = _sha256(plan_path)
    started = datetime.now(UTC)
    records: list[dict[str, Any]] = []

    def revalidate_remote(item: Mapping[str, Any]) -> None:
        matches = archive_sources.list_objects_v2(
            _source_id(item), "sscofs", str(item["key"]), max_keys=2,
        )
        exact = [candidate for candidate in matches if candidate.get("key") == item.get("key")]
        if len(exact) != 1:
            raise RuntimeError("planned source object is no longer uniquely listed")
        archive_sources.validate_remote_metadata(item, exact[0])

    def job(item: Mapping[str, Any]) -> dict[str, Any]:
        target = raw_dir / archive_sources.cache_relpath(item)
        try:
            cached = _cache_result(item, target, validate_netcdf=True)
            if cached is not None:
                return cached
            legacy = _legacy_aws_cache_result(
                item, _legacy_aws_destination(run_path, item), validate_netcdf=True
            )
            if legacy is not None:
                return legacy
            revalidate_remote(item)
            return download_object(item, target, max_retries=max_retries, validate_netcdf=True)
        except Exception as exc:
            return {
                "key": item.get("key"), "url": item.get("url"),
                "local_path": str(target.resolve()), "status": "failed",
                "cache_hit": False, "size_bytes": _object_size(item),
                "etag": _clean_etag(item.get("etag")), "error": str(exc),
                "valid_time_utc": item.get("valid_time_utc"),
                "run_time_utc": item.get("run_time_utc"),
                "product": item.get("product"), "guidance": item.get("guidance"),
                "source_id": _source_id(item), "source_identity": archive_sources.source_identity_digest(item),
                "source": dict(item),
            }

    with ThreadPoolExecutor(max_workers=request["max_workers"]) as pool:
        futures = {pool.submit(job, item): item for item in objects}
        for future in as_completed(futures):
            records.append(future.result())
            records.sort(key=lambda item: (str(item.get("valid_time_utc")), str(item.get("key"))))
            checkpoint = {
                "schema_version": "sscofs_fetch_manifest_v2",
                "generated_utc": _iso(datetime.now(UTC)),
                "software_version": SOFTWARE_VERSION,
                "request": request,
                "estimate_path": str(plan_path),
                "reviewed_plan_sha256": plan_sha256,
                "normalized_request_sha256": plan["normalized_request_sha256"],
                "selected_objects_sha256": plan["selected_objects_sha256"],
                "records": records,
                "complete": len(records) == len(objects) and all(item["status"] != "failed" for item in records),
            }
            write_json_atomic(run_path / "fetch_manifest.json", checkpoint)
    failures = [item for item in records if item["status"] == "failed"]
    manifest = {
        "schema_version": "sscofs_fetch_manifest_v2",
        "generated_utc": _iso(datetime.now(UTC)),
        "started_utc": _iso(started),
        "software_version": SOFTWARE_VERSION,
        "request": request,
        "estimate_path": str(plan_path),
        "reviewed_plan_sha256": plan_sha256,
        "normalized_request_sha256": plan["normalized_request_sha256"],
        "selected_objects_sha256": plan["selected_objects_sha256"],
        "selected_object_count_binding": len(objects),
        "selected_total_bytes_binding": int(plan["total_bytes"]),
        "source_discovery": plan.get("source_discovery"),
        "source_totals": plan.get("source_totals"),
        "selected_object_count": len(objects),
        "successful_object_count": len(records) - len(failures),
        "failure_count": len(failures),
        "downloaded_count": sum(item["status"] == "downloaded" for item in records),
        "cache_hit_count": sum(item["status"] == "cache_hit" for item in records),
        "records": records,
        "complete": len(records) == len(objects) and not failures,
    }
    write_json_atomic(run_path / "fetch_manifest.json", manifest)
    if failures:
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


def _records_from_files(
    paths: Sequence[str | Path], request: Mapping[str, Any],
    source_by_path: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
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
        if source_by_path is not None and str(path.resolve()) not in source_by_path:
            raise ValueError(f"Input file is not bound to the reviewed fetch manifest: {path}")
        provenance = dict((source_by_path or {}).get(str(path.resolve()), {}))
        parsed = parse_object_key(str(provenance.get("key") or path.name))
        for index, valid in enumerate(times):
            if parsed and parsed.get("valid_time_utc") and len(times) == 1:
                expected = _parse_utc(parsed["valid_time_utc"])
                if abs((valid - expected).total_seconds()) > 1:
                    raise ValueError(f"Filename/NetCDF valid-time mismatch in {path}: {_iso(expected)} != {_iso(valid)}")
            if start <= valid < end:
                records.append({
                    "path": path, "time_index": index, "time": valid,
                    "source_id": provenance.get("source_id", "explicit_input"),
                    "source_key": provenance.get("key", path.name),
                    "source_url": provenance.get("url"),
                })
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
    *,
    source_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Concatenate verified native fields and derive requested sigma views."""

    import numpy as np
    from netCDF4 import Dataset, date2num

    normalized = validate_request(request)
    if normalized["product"] != "fields":
        raise ValueError("extract_fields supports only product='fields'")
    source_by_path = None if source_records is None else {
        str(Path(str(item["local_path"])).resolve()): item
        for item in source_records if item.get("local_path")
    }
    records = _records_from_files(paths, normalized, source_by_path)
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

    record_source_summary = {
        source_id: sum(item["source_id"] == source_id for item in records)
        for source_id in sorted({item["source_id"] for item in records})
    }
    with Dataset(part, "w", format="NETCDF4") as output:
        output.setncattr("title", "Compact NOAA SSCOFS native fields")
        output.setncattr("source", "NOAA operational AWS and NCEI long-term OFS archives")
        output.setncattr("software", f"sscofs-fetcher {SOFTWARE_VERSION}")
        output.setncattr("request_json", json.dumps(normalized, sort_keys=True))
        output.setncattr("source_files_json", json.dumps([str(item["path"].resolve()) for item in records]))
        output.setncattr("source_archives_json", json.dumps(sorted({item["source_id"] for item in records})))
        output.setncattr("source_objects_json", json.dumps([
            {"source_id": item["source_id"], "key": item["source_key"], "url": item["source_url"]}
            for item in records
        ], sort_keys=True))
        output.setncattr("source_summary_json", json.dumps(record_source_summary, sort_keys=True))
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

            source_id_width = max(1, max(len(str(item["source_id"])) for item in records))
            source_key_width = max(1, max(len(str(item["source_key"])) for item in records))
            output.createDimension("source_id_strlen", source_id_width)
            output.createDimension("source_key_strlen", source_key_width)
            source_id_var = output.createVariable("source_archive", "S1", (time_dim, "source_id_strlen"))
            source_key_var = output.createVariable("source_key", "S1", (time_dim, "source_key_strlen"))
            for index, item in enumerate(records):
                for variable, value, width in (
                    (source_id_var, str(item["source_id"]), source_id_width),
                    (source_key_var, str(item["source_key"]), source_key_width),
                ):
                    row = np.full(width, b" ", dtype="S1")
                    encoded = np.frombuffer(value.encode("utf-8")[:width], dtype="S1")
                    row[: encoded.size] = encoded
                    variable[index, :] = row

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
        "schema_version": "sscofs_extraction_v2",
        "output_path": str(target.resolve()),
        "record_count": len(records),
        "start_utc": _iso(records[0]["time"]),
        "end_utc": _iso(records[-1]["time"]),
        "source_files": [str(item["path"].resolve()) for item in records],
        "source_records": [
            {"source_id": item["source_id"], "key": item["source_key"], "url": item["source_url"]}
            for item in records
        ],
        "source_summary": record_source_summary,
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


def _verified_manifest_records(run_dir: Path, request: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest_path = run_dir / "fetch_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("reviewed plan and fetch manifest are required")
    manifest = _read_json(manifest_path)
    plan_path = Path(str(manifest.get("estimate_path") or "")).resolve()
    if not plan_path.is_file():
        raise RuntimeError("fetch manifest reviewed plan path does not exist")
    plan = _read_json(plan_path)
    if plan.get("schema_version") != "sscofs_download_estimate_v2" or manifest.get("schema_version") != "sscofs_fetch_manifest_v2":
        raise RuntimeError("v2 reviewed plan and fetch manifest are required for extraction")
    normalized = validate_request(request)
    selected = list(plan.get("objects", []))
    if plan.get("normalized_request_sha256") != _canonical_sha256(normalized):
        raise RuntimeError("request does not match the reviewed plan")
    selected_sha = _canonical_sha256(sorted(selected, key=lambda item: (_source_id(item), str(item.get("key")))))
    if plan.get("selected_objects_sha256") != selected_sha or manifest.get("selected_objects_sha256") != selected_sha:
        raise RuntimeError("selected-object binding is invalid")
    if manifest.get("reviewed_plan_sha256") != _sha256(plan_path):
        raise RuntimeError("fetch manifest does not bind the reviewed plan file")
    expected = {f"{_source_id(item)}:{item.get('key')}": item for item in selected}
    records = list(manifest.get("records", []))
    if len(records) != len(selected) or any(record.get("status") not in {"downloaded", "cache_hit"} for record in records):
        raise RuntimeError("fetch manifest is incomplete")
    raw_root = (run_dir / "cache" / "raw").resolve()
    for record in records:
        item = expected.get(f"{record.get('source_id')}:{record.get('key')}")
        if item is None:
            raise RuntimeError("fetch manifest contains an unplanned source object")
        path = Path(str(record.get("local_path", ""))).resolve()
        try:
            path.relative_to(raw_root)
        except ValueError as exc:
            raise RuntimeError(f"cache path escapes the run directory: {path}") from exc
        provider_path = raw_root / archive_sources.cache_relpath(item)
        legacy_path = _legacy_aws_destination(run_dir, item).resolve()
        uses_legacy = bool(record.get("legacy_cache_reused"))
        expected_path = legacy_path if uses_legacy else provider_path
        if path != expected_path or not path.is_file() or path.stat().st_size != _object_size(item) or _sha256(path) != record.get("sha256"):
            raise RuntimeError(f"cache payload binding is invalid: {path}")
        if uses_legacy:
            if _legacy_aws_cache_result(item, path, validate_netcdf=True) is None:
                raise RuntimeError(f"legacy AWS cache binding is invalid: {path}")
        else:
            sidecar = _read_json(_download_sidecar(path))
            if any((
                sidecar.get("schema_version") != "sscofs_cached_object_v2",
                sidecar.get("source_identity") != archive_sources.source_identity_digest(item),
                sidecar.get("key") != item.get("key"), sidecar.get("url") != item.get("url"),
                sidecar.get("last_modified") != item.get("last_modified"),
                _clean_etag(sidecar.get("etag")) != _clean_etag(item.get("etag")),
                sidecar.get("sha256") != record.get("sha256"),
            )):
                raise RuntimeError(f"cache sidecar binding is invalid: {path}")
    return records


def _delete_raw_after_extract(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "fetch_manifest.json"
    manifest = _read_json(manifest_path)
    deleted: list[dict[str, Any]] = []
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
            size = path.stat().st_size
            digest = record.get("sha256") or _sha256(path)
            path.unlink()
            deleted.append({
                "path": str(path), "key": record.get("key"),
                "source_id": record.get("source_id"), "bytes": size, "sha256": digest,
            })
        sidecar = _download_sidecar(path)
        if sidecar.is_file():
            sidecar.unlink()
    record_bindings = [
        {name: item.get(name) for name in ("source_id", "key", "url", "local_path", "size_bytes", "sha256")}
        for item in manifest.get("records", [])
    ]
    cleanup = {
        "schema_version": "sscofs_cache_cleanup_v2",
        "policy": "delete_after_extract", "deleted_file_count": len(deleted),
        "deleted_bytes": sum(item["bytes"] for item in deleted), "files": deleted,
        "completed_utc": _iso(datetime.now(UTC)),
        "recovery": "re-download provider-bound objects from the reviewed plan",
        "reviewed_plan_sha256": manifest.get("reviewed_plan_sha256"),
        "normalized_request_sha256": manifest.get("normalized_request_sha256"),
        "selected_objects_sha256": manifest.get("selected_objects_sha256"),
        "fetch_records_sha256": _canonical_sha256(record_bindings),
    }
    manifest["cache_cleanup"] = cleanup
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(run_dir / "cache_cleanup.json", cleanup)
    return cleanup


def _add_common_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request", required=True, help="Path to an sscofs_request_v2 (or migratable v1) JSON file.")
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
    fetch.add_argument("--plan", required=True, help="Reviewed sscofs_download_estimate_v2 JSON path.")
    fetch.add_argument("--run-dir", required=True)
    fetch.add_argument("--max-retries", type=int, default=4)

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
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        request = None if args.command == "fetch" else load_request(args.request)
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
                (run_dir / "download_estimate.json").unlink(missing_ok=True)
            summary = {
                "selected_object_count": result["selected_object_count"],
                "total_bytes": result["total_bytes"],
                "routing_decision": result["routing_decision"],
                "output": str(output),
            }
        elif args.command == "fetch":
            result = fetch_request(
                args.plan, run_dir, max_retries=args.max_retries,
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
            verified_records = _verified_manifest_records(run_dir, request) if not args.path else None
            result = extract_fields(paths, request, output, source_records=verified_records)
            write_json_atomic(run_dir / "extraction_manifest.json", result)
            summary = {
                "record_count": result["record_count"], "output": str(output),
                "size_bytes": result["size_bytes"], "raw_cleanup": (
                    "deferred_until_health_passes" if request["cache_policy"] == "delete_after_extract" else "kept"
                ),
                "manifest": str(run_dir / "extraction_manifest.json"),
            }
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"sscofs-fetcher {getattr(args, 'command', 'command')} failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
