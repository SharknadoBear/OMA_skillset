#!/usr/bin/env python3
"""Shared anonymous-AWS planning and transfer core for NOAA ROMS OFS skills."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

try:
    from . import ofs_archive_sources as archive_sources
except ImportError:
    import ofs_archive_sources as archive_sources

BUCKET = "noaa-nos-ofs-pds"
S3_ENDPOINT = f"https://{BUCKET}.s3.amazonaws.com"
UTC = timezone.utc
_NETCDF_METADATA_LOCK = threading.Lock()
SOURCE_POLICIES = {"aws_then_ncei", "aws_only", "ncei_only"}


@dataclass(frozen=True)
class ModelConfig:
    """Small model-specific surface for the generic AWS core."""

    model: str
    request_schema: str
    connector_name: str
    cycle_hours: tuple[int, ...] = (0, 6, 12, 18)
    default_variables: tuple[str, ...] = ("zeta", "salt", "u", "v")
    default_views: tuple[str | int, ...] = ("surface",)


def _source_ids(request: Mapping[str, Any]) -> list[str]:
    return ({"aws_only": ["aws_operational"],
             "ncei_only": ["ncei_long_term"]}
            .get(request["source_policy"], ["aws_operational", "ncei_long_term"]))


def _ncei_capability(request: Mapping[str, Any]) -> tuple[bool, str | None]:
    if request["product"] == "regulargrid":
        return False, "ncei_long_term does not support regulargrid"
    if request["product"] == "fields" and request["guidance"] == "forecast":
        return False, "ncei_long_term does not support native field forecasts"
    return True, None


def _decorate_source(config: ModelConfig, item: Mapping[str, Any],
                     source_id: str = "aws_operational") -> dict[str, Any]:
    descriptor = archive_sources.get_source_descriptor(source_id, config.model)
    value = {**descriptor, **dict(item), "source_id": source_id,
             "provider": descriptor["provider"], "archive_role": descriptor["archive_role"],
             "container": descriptor["container"], "endpoint": descriptor["endpoint"],
             "listing_endpoint": descriptor["listing_endpoint"]}
    value.setdefault("url", archive_sources.canonical_object_url(source_id, config.model, str(value["key"])))
    parsed = parse_object_key(str(value["key"]), config)
    if parsed is not None:
        semantic = {
            "model": config.model, "product": parsed["product"],
            "guidance": parsed["guidance"], "run_time": parsed["run_time"],
            "valid_time": parsed["valid_time"], "lead": parsed["lead"],
            "aggregate": parsed["aggregate"],
        }
        value["naming_era"] = parsed["naming"]
        value["semantic_identity"] = semantic
        value["semantic_identity_digest"] = archive_sources.semantic_identity_digest(semantic)
    value["source_identity"] = archive_sources.source_identity_digest(value)
    return value


def _requests_module():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("requests is required for public HTTPS access") from exc
    return requests


def parse_utc(value: Any, name: str = "timestamp") -> datetime:
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


def iso_utc(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def json_clean(value: Any) -> Any:
    if isinstance(value, datetime):
        return iso_utc(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_clean(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return json_clean(value.item())
        if isinstance(value, np.ndarray):
            return json_clean(value.tolist())
    except ImportError:
        pass
    return value


def write_json_atomic(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(json_clean(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_json_sha256(value: Any) -> str:
    """Return a stable digest for an in-memory JSON-compatible value."""
    payload = json.dumps(
        json_clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_objects_sha256(objects: Sequence[Mapping[str, Any]]) -> str:
    """Hash complete approved source records independent of list ordering."""
    ordered = sorted((dict(item) for item in objects), key=lambda item: str(item.get("key", "")))
    return canonical_json_sha256(ordered)


def load_request(path: str | Path, config: ModelConfig) -> dict[str, Any]:
    raw = read_json(path)
    if not isinstance(raw, Mapping):
        raise ValueError("request root must be a JSON object")
    return validate_request(raw, config)


def validate_request(mapping: Mapping[str, Any], config: ModelConfig) -> dict[str, Any]:
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
    input_schema = mapping.get("schema_version")
    legacy_schema = f"{config.model}_request_v1"
    if input_schema not in {legacy_schema, config.request_schema}:
        raise ValueError(
            f"schema_version must be {legacy_schema!r} or {config.request_schema!r}"
        )
    if input_schema == legacy_schema and "source_policy" in mapping:
        raise ValueError(
            "source_policy is not permitted in a v1 request; v1 always migrates "
            "to source_policy=aws_then_ncei"
        )
    start = parse_utc(mapping.get("start_utc"), "start_utc")
    end = parse_utc(mapping.get("end_utc_exclusive"), "end_utc_exclusive")
    if end <= start:
        raise ValueError("end_utc_exclusive must be later than start_utc")
    product = mapping.get("product")
    if product not in {"fields", "stations", "regulargrid"}:
        raise ValueError("product must be fields, stations, or regulargrid")
    guidance = mapping.get("guidance")
    if guidance not in {"nowcast", "forecast"}:
        raise ValueError("guidance must be nowcast or forecast")
    run_cycle: datetime | None = None
    if guidance == "forecast":
        if "run_cycle_utc" not in mapping:
            raise ValueError("run_cycle_utc is required for forecast requests")
        run_cycle = parse_utc(mapping["run_cycle_utc"], "run_cycle_utc")
        if (
            run_cycle.minute
            or run_cycle.second
            or run_cycle.microsecond
            or run_cycle.hour not in config.cycle_hours
        ):
            cycle_text = ", ".join(f"{hour:02d}" for hour in config.cycle_hours)
            raise ValueError(f"run_cycle_utc must be exactly one of {cycle_text} UTC")
    elif "run_cycle_utc" in mapping:
        raise ValueError("run_cycle_utc is permitted only for forecast requests")

    if product != "fields":
        if "variables" in mapping or "vertical_views" in mapping:
            raise ValueError(f"{product} is passthrough-only; variables and vertical_views are not allowed")
        variables: list[str] = []
        views: list[str | int] = []
    else:
        raw_variables = mapping.get("variables", list(config.default_variables))
        if (
            not isinstance(raw_variables, list)
            or not raw_variables
            or any(not isinstance(item, str) or not item.strip() for item in raw_variables)
        ):
            raise ValueError("variables must be a non-empty array of names")
        aliases = {"salinity": "salt", "temperature": "temp"}
        variables = [aliases.get(item.strip().lower(), item.strip()) for item in raw_variables]
        if len(set(variables)) != len(variables):
            raise ValueError("variables must be unique after alias normalization")
        if ("u" in variables) != ("v" in variables):
            raise ValueError("u and v must be requested together")
        raw_views = mapping.get("vertical_views", list(config.default_views))
        if not isinstance(raw_views, list) or not raw_views:
            raise ValueError("vertical_views must be a non-empty array")
        valid_named = {"surface", "near_surface", "bottom", "depth_average"}
        views = []
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
        raise ValueError(
            "source_policy must be aws_then_ncei, aws_only, or ncei_only"
        )
    result: dict[str, Any] = {
        "schema_version": config.request_schema,
        "start_utc": iso_utc(start),
        "end_utc_exclusive": iso_utc(end),
        "product": product,
        "guidance": guidance,
        "missing_policy": missing_policy,
        "cache_policy": cache_policy,
        "max_workers": max_workers,
        "source_policy": source_policy,
    }
    if run_cycle is not None:
        result["run_cycle_utc"] = iso_utc(run_cycle)
    if product == "fields":
        result["variables"] = variables
        result["vertical_views"] = views
    return result


def request_migration(request: Mapping[str, Any] | str | Path,
                      config: ModelConfig) -> dict[str, Any]:
    """Describe the lossless v1-to-v2 normalization recorded in v2 artifacts."""
    raw = read_json(request) if isinstance(request, (str, Path)) else dict(request)
    input_schema = raw.get("schema_version")
    migrated = input_schema != config.request_schema
    return {
        "input_schema_version": input_schema,
        "normalized_schema_version": config.request_schema,
        "migration": (
            {
                "applied": True,
                "name": f"{config.model}_request_v1_to_v2",
                "defaults_applied": {"source_policy": "aws_then_ncei"},
            }
            if migrated else {"applied": False, "name": None, "defaults_applied": {}}
        ),
    }


def _layout_for_key(key: str) -> str:
    if re.search(r"/\d{4}/\d{2}/\d{2}/", key):
        return "daily"
    if re.search(r"/\d{6}/", key):
        return "monthly"
    if re.search(r"/access/[^/]+/\d{4}/\d{2}/", key):
        return "ncei_monthly"
    return "unknown"


def _approved_source_contract(item: Mapping[str, Any], config: ModelConfig) -> dict[str, Any]:
    """Validate one reviewed-plan object against its exact NOAA source scope."""
    source_id = str(item.get("source_id") or "aws_operational")
    key = item.get("key")
    prefix = archive_sources.get_source_descriptor(source_id, config.model)["root_prefix"]
    if not isinstance(key, str) or not key.startswith(prefix):
        raise RuntimeError(f"approved plan key is outside {prefix!r}: {key!r}")
    parsed = parse_object_key(key, config)
    if parsed is None:
        raise RuntimeError(f"approved plan contains an invalid {config.model.upper()} object key")
    relative = key[len(prefix):]
    daily = re.fullmatch(r"(\d{4})/(\d{2})/(\d{2})/([^/]+)", relative)
    monthly = re.fullmatch(r"(\d{6})/([^/]+)", relative)
    ncei_monthly = re.fullmatch(r"(\d{4})/(\d{2})/([^/]+)", relative)
    run = parse_utc(parsed["run_time"])
    if daily is not None:
        archive_date = "".join(daily.group(index) for index in (1, 2, 3))
        if archive_date != run.strftime("%Y%m%d"):
            raise RuntimeError(f"approved plan daily-layout key date disagrees with filename run date: {key}")
    elif monthly is not None:
        if monthly.group(1) != run.strftime("%Y%m"):
            raise RuntimeError(f"approved plan monthly-layout key month disagrees with filename run date: {key}")
    elif source_id == "ncei_long_term" and ncei_monthly is not None:
        if ncei_monthly.group(1) + ncei_monthly.group(2) != run.strftime("%Y%m"):
            raise RuntimeError(f"approved plan NCEI key month disagrees with filename run date: {key}")
    else:
        raise RuntimeError(f"approved plan key has no exact daily or monthly archive layout: {key}")
    expected_url = archive_sources.canonical_object_url(source_id, config.model, key)
    if item.get("url") != expected_url:
        raise RuntimeError(f"approved plan URL is not the exact NOAA archive object URL for {key}")
    archive_sources.validate_source_object(
        config.model, _decorate_source(config, item, source_id),
        expected_source_id=source_id, require_metadata=True,
    )
    if not _clean_etag(item.get("etag")):
        raise RuntimeError(f"approved plan object has no nonempty ETag: {key}")
    if not isinstance(item.get("last_modified"), str) or not item["last_modified"].strip():
        raise RuntimeError(f"approved plan object has no Last-Modified provenance: {key}")
    return parsed


def parse_object_key(key: str, config: ModelConfig) -> dict[str, Any] | None:
    model = re.escape(config.model)
    name = Path(key).name
    patterns = [
        (
            re.compile(
                rf"^(?P<model>{model})\.t(?P<hour>\d{{2}})z\.(?P<date>\d{{8}})\."
                r"(?P<product>fields|regulargrid)\.(?P<code>[nf])(?P<lead>\d{3})\.nc$",
                re.IGNORECASE,
            ),
            "current",
        ),
        (
            re.compile(
                rf"^(?P<model>{model})\.t(?P<hour>\d{{2}})z\.(?P<date>\d{{8}})\."
                r"(?P<product>stations)\.(?P<aggregate_guidance>nowcast|forecast)\.nc$",
                re.IGNORECASE,
            ),
            "current",
        ),
        (
            re.compile(
                rf"^(?:nos\.)?(?P<model>{model})\.(?P<product>fields|regulargrid)\."
                r"(?P<code>[nf])(?P<lead>\d{3})\.(?P<date>\d{8})\.t(?P<hour>\d{2})z\.nc$",
                re.IGNORECASE,
            ),
            "legacy",
        ),
        (
            re.compile(
                rf"^(?:nos\.)?(?P<model>{model})\.(?P<product>stations)\."
                r"(?P<aggregate_guidance>nowcast|forecast)\.(?P<date>\d{8})\.t(?P<hour>\d{2})z\.nc$",
                re.IGNORECASE,
            ),
            "legacy",
        ),
    ]
    match = None
    naming = ""
    for pattern, candidate_naming in patterns:
        match = pattern.match(name)
        if match is not None:
            naming = candidate_naming
            break
    if match is None:
        return None
    values = match.groupdict()
    run = datetime.strptime(values["date"] + values["hour"], "%Y%m%d%H").replace(tzinfo=UTC)
    if run.hour not in config.cycle_hours:
        return None
    product = values["product"].lower()
    if values.get("aggregate_guidance"):
        guidance = values["aggregate_guidance"].lower()
        lead: int | None = None
        valid: datetime | None = None
        aggregate = True
        if guidance == "nowcast":
            expected_start = run - timedelta(hours=6)
            expected_end = run + timedelta(minutes=6)
        else:
            expected_start = run
            expected_end = run + timedelta(hours=48, minutes=6)
    else:
        code = values["code"].lower()
        lead = int(values["lead"])
        source_id = "ncei_long_term" if "/access/" in key else "aws_operational"
        if (code == "n" and not ((1 <= lead <= 6) or (lead == 0 and source_id == "ncei_long_term"))) or (code == "f" and not 1 <= lead <= 48):
            return None
        guidance = "nowcast" if code == "n" else "forecast"
        if guidance == "nowcast":
            valid = run + timedelta(hours=lead - 6)
        else:
            valid = run + timedelta(hours=lead)
        expected_start = valid
        expected_end = valid + timedelta(hours=1)
        aggregate = False
    return {
        "key": key,
        "name": name,
        "layout": _layout_for_key(key),
        "naming": naming,
        "aggregate": aggregate,
        "model": config.model,
        "product": product,
        "guidance": guidance,
        "run_time": iso_utc(run),
        "cycle_hour": run.hour,
        "lead": lead,
        "valid_time": iso_utc(valid),
        "expected_start_utc": iso_utc(expected_start),
        "expected_end_utc_exclusive": iso_utc(expected_end),
        "semantic_id": f"{config.model}:{product}:{guidance}:{iso_utc(run)}:{iso_utc(valid) or 'aggregate'}",
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
            objects.append({
                "key": key,
                "size": int(size),
                "etag": (_xml_text(node, "ETag") or "").strip('"'),
                "last_modified": _xml_text(node, "LastModified"),
                "storage_class": _xml_text(node, "StorageClass"),
                "url": endpoint + "/" + quote(key, safe="/"),
            })
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


def _months(start: datetime, end: datetime) -> Iterable[datetime]:
    cursor = datetime(start.year, start.month, 1, tzinfo=UTC)
    final = datetime(end.year, end.month, 1, tzinfo=UTC)
    while cursor <= final:
        yield cursor
        cursor = datetime(
            cursor.year + (cursor.month == 12),
            1 if cursor.month == 12 else cursor.month + 1,
            1,
            tzinfo=UTC,
        )


def discovery_prefixes(request: Mapping[str, Any], config: ModelConfig) -> list[str]:
    if request["guidance"] == "forecast":
        center = parse_utc(request["run_cycle_utc"])
        first, last = center - timedelta(days=1), center + timedelta(days=1)
    else:
        first = parse_utc(request["start_utc"]) - timedelta(days=1)
        last = parse_utc(request["end_utc_exclusive"]) + timedelta(days=1)
    daily = [f"{config.model}/netcdf/{day:%Y/%m/%d}/" for day in _days(first, last)]
    monthly = [f"{config.model}/netcdf/{month:%Y%m}/" for month in _months(first, last)]
    return daily + monthly


def source_discovery_prefixes(request: Mapping[str, Any], config: ModelConfig,
                              source_id: str) -> list[str]:
    if source_id == "aws_operational":
        return discovery_prefixes(request, config)
    descriptor = archive_sources.get_source_descriptor(source_id, config.model)
    if request["guidance"] == "forecast":
        center = parse_utc(request["run_cycle_utc"])
        first, last = center - timedelta(days=1), center + timedelta(days=1)
    else:
        first = parse_utc(request["start_utc"]) - timedelta(days=1)
        last = parse_utc(request["end_utc_exclusive"]) + timedelta(days=1)
    return [f"{descriptor['root_prefix']}{month:%Y/%m}/" for month in _months(first, last)]


def discover_objects(
    request: Mapping[str, Any],
    config: ModelConfig,
    *,
    session: Any | None = None,
    endpoint: str = S3_ENDPOINT,
) -> list[dict[str, Any]]:
    return discover_objects_with_evidence(
        request, config, session=session, endpoint=endpoint)[0]


def _discover_one_source(request: Mapping[str, Any], config: ModelConfig,
                         source_id: str, *, session: Any | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    prefixes = source_discovery_prefixes(request, config, source_id)
    for prefix in prefixes:
        for raw in archive_sources.list_objects_v2(
                source_id, config.model, prefix, session=session):
            parsed = parse_object_key(raw["key"], config)
            if parsed is None:
                continue
            item = _decorate_source(config, {**raw, **parsed}, source_id)
            if item["product"] == request["product"] and item["guidance"] == request["guidance"]:
                found[item["key"]] = item
    objects = sorted(found.values(), key=lambda item: (item["run_time"], item.get("valid_time") or "", item["key"]))
    return objects, {"source_id": source_id, "status": "success",
                     "prefixes": prefixes, "object_count": len(objects), "error": None}


def _coverage_gaps(request: Mapping[str, Any], objects: Sequence[Mapping[str, Any]],
                   config: ModelConfig) -> list[str]:
    relaxed = {**request, "missing_policy": "skip"}
    try:
        return list(select_objects(relaxed, objects, config)["missing_times"])
    except RuntimeError as exc:
        if "did not contain matching" not in str(exc):
            raise
        start, end = parse_utc(request["start_utc"]), parse_utc(request["end_utc_exclusive"])
        step = 360 if request["product"] == "stations" else 3600
        return [iso_utc(stamp) for stamp in _expected_times(start, end, step)]


def _scientific_fallback_times(request: Mapping[str, Any],
                               objects: Sequence[Mapping[str, Any]]) -> list[str]:
    """Find station boundaries covered only by the following cycle's first record."""
    if request["product"] != "stations" or request["guidance"] != "nowcast":
        return []
    start, end = parse_utc(request["start_utc"]), parse_utc(request["end_utc_exclusive"])
    candidates = [item for item in objects
                  if item.get("product") == "stations" and item.get("guidance") == "nowcast"]
    unresolved: list[str] = []
    for stamp in _expected_times(start, end, 360):
        covering = [item for item in candidates if stamp in _nominal_item_times(item)]
        if (covering
                and any(parse_utc(item["expected_start_utc"]) == stamp for item in covering)
                and not any(parse_utc(item["run_time"]) == stamp for item in covering)):
            unresolved.append(iso_utc(stamp))
    return unresolved


def validate_fallback_decision(request: Mapping[str, Any],
                               artifact: Mapping[str, Any],
                               selected: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return violations of the provider-ordering evidence contract."""
    failures: list[str] = []
    policy = request.get("source_policy")
    selected_sources = {str(item.get("source_id")) for item in selected}
    attempts = artifact.get("source_attempts")
    if not isinstance(attempts, list) or any(not isinstance(item, Mapping) for item in attempts):
        attempts = []
    attempted = {str(item.get("source_id")) for item in attempts}
    successful = {str(item.get("source_id")) for item in attempts
                  if item.get("status") == "success"}
    if policy == "aws_only":
        if "ncei_long_term" in selected_sources or "ncei_long_term" in attempted:
            failures.append("aws_only plan contains or attempted an NCEI object")
        if "aws_operational" not in successful:
            failures.append("aws_only plan lacks successful AWS discovery evidence")
    elif policy == "ncei_only":
        if "aws_operational" in selected_sources or "aws_operational" in attempted:
            failures.append("ncei_only plan contains or attempted an AWS object")
        if "ncei_long_term" not in successful:
            failures.append("ncei_only plan lacks successful NCEI discovery evidence")
    elif policy == "aws_then_ncei":
        if "aws_operational" not in successful:
            failures.append("aws_then_ncei plan lacks successful AWS discovery evidence")
        if "ncei_long_term" in selected_sources:
            semantic_gap = artifact.get("coverage_before_fallback")
            scientific_gap = artifact.get("scientific_precedence_before_fallback")
            if "ncei_long_term" not in successful:
                failures.append("NCEI selection lacks successful NCEI discovery evidence")
            if artifact.get("fallback_triggered") is not True:
                failures.append("NCEI selection lacks an explicit fallback trigger")
            if not ((isinstance(semantic_gap, list) and semantic_gap)
                    or (isinstance(scientific_gap, list) and scientific_gap)):
                failures.append("NCEI fallback lacks an unresolved AWS semantic/scientific record")
    else:
        failures.append("source policy is invalid")
    return failures


def discover_objects_with_evidence(request: Mapping[str, Any], config: ModelConfig, *,
                                   session: Any | None = None,
                                   endpoint: str = S3_ENDPOINT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del endpoint
    policy = request["source_policy"]
    attempts: list[dict[str, Any]] = []
    if policy == "ncei_only":
        capable, reason = _ncei_capability(request)
        if not capable:
            raise ValueError(reason)
        objects, evidence = _discover_one_source(request, config, "ncei_long_term", session=session)
        attempts.append(evidence)
        before, after = _coverage_gaps(request, [], config), _coverage_gaps(request, objects, config)
        return objects, {"source_attempts": attempts, "fallback_triggered": False,
                         "fallback_reason": None, "coverage_before_fallback": before,
                         "coverage_after_fallback": after, "ncei_filled_times": sorted(set(before) - set(after)),
                         "unresolved_times": after}
    aws, evidence = _discover_one_source(request, config, "aws_operational", session=session)
    attempts.append(evidence)
    before = _coverage_gaps(request, aws, config)
    scientific_before = _scientific_fallback_times(request, aws)
    combined, triggered, reason = list(aws), False, None
    capable, capability_reason = _ncei_capability(request)
    if policy == "aws_then_ncei" and (before or scientific_before) and capable:
        ncei, evidence = _discover_one_source(request, config, "ncei_long_term", session=session)
        attempts.append(evidence); combined.extend(ncei)
        triggered = True
        reason = ("aws_scientific_precedence_unresolved" if scientific_before and not before
                  else "aws_semantic_coverage_unresolved")
    elif policy == "aws_then_ncei" and (before or scientific_before):
        attempts.append({"source_id": "ncei_long_term", "status": "not_capable",
                         "prefixes": [], "object_count": 0, "error": capability_reason})
        reason = capability_reason
    after = _coverage_gaps(request, combined, config)
    return combined, {"source_attempts": attempts, "fallback_triggered": triggered,
                      "fallback_reason": reason, "coverage_before_fallback": before,
                      "scientific_precedence_before_fallback": scientific_before,
                      "coverage_after_fallback": after, "ncei_filled_times": sorted(set(before) - set(after)),
                      "unresolved_times": after}


def _preference(item: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        1 if item.get("lead") != 0 else 0,
        1 if item.get("source_id", "aws_operational") == "aws_operational" else 0,
        1 if item.get("naming") == "current" else 0,
        1 if item.get("layout") == "daily" else 0,
    )


def _expected_times(start: datetime, end: datetime, step_seconds: int) -> list[datetime]:
    first_epoch = math.ceil(start.timestamp() / step_seconds) * step_seconds
    cursor = datetime.fromtimestamp(first_epoch, tz=UTC)
    result: list[datetime] = []
    while cursor < end:
        result.append(cursor)
        cursor += timedelta(seconds=step_seconds)
    return result


def expected_times(start: datetime, end: datetime, step_seconds: int) -> list[datetime]:
    """Return exact cadence points in the half-open interval ``[start, end)``."""
    return _expected_times(start, end, step_seconds)


def _nominal_item_times(item: Mapping[str, Any]) -> list[datetime]:
    return _expected_times(
        parse_utc(item["expected_start_utc"]),
        parse_utc(item["expected_end_utc_exclusive"]),
        360 if item["product"] == "stations" else 3600,
    )


def _same_remote(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return int(a.get("size", -1)) == int(b.get("size", -2)) and str(a.get("etag", "")) == str(b.get("etag", ""))


def select_objects(
    request: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    config: ModelConfig,
) -> dict[str, Any]:
    start, end = parse_utc(request["start_utc"]), parse_utc(request["end_utc_exclusive"])
    candidates: list[dict[str, Any]] = []
    for raw in objects:
        item = dict(raw)
        if item.get("product") != request["product"] or item.get("guidance") != request["guidance"]:
            continue
        if request["guidance"] == "forecast" and item.get("run_time") != request["run_cycle_utc"]:
            continue
        item_start = parse_utc(item["expected_start_utc"])
        item_end = parse_utc(item["expected_end_utc_exclusive"])
        if item_start < end and item_end > start:
            candidates.append(item)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in candidates:
        identity = (
            item["product"],
            item["guidance"],
            item["run_time"] if item["aggregate"] else item["valid_time"],
        )
        grouped.setdefault(identity, []).append(item)
    selected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for identity, group in grouped.items():
        best_rank = max(_preference(item) for item in group)
        best = [item for item in group if _preference(item) == best_rank]
        if len(best) > 1 and any(not _same_remote(best[0], item) for item in best[1:]):
            keys = ", ".join(item["key"] for item in best)
            raise RuntimeError(f"equal-rank conflicting semantic duplicates for {identity}: {keys}")
        winner = sorted(best, key=lambda item: item["key"])[0]
        selected.append(winner)
        duplicates.extend({**item, "rejected_in_favor_of": winner["key"],
                           "rejection_reason": "lower_archive_or_layout_preference"}
                          for item in group if item is not winner)
    selected.sort(key=lambda item: (item.get("valid_time") or item["run_time"], item["key"]))

    # Aggregate station cycles share their boundary timestamp. Walk cycles from
    # oldest to newest so the preceding cycle owns that record, and omit an
    # aggregate that would contribute no new requested time at all.
    if request["product"] == "stations":
        covered: set[datetime] = set()
        needed: list[dict[str, Any]] = []
        for item in selected:
            contribution = {
                stamp for stamp in _nominal_item_times(item)
                if start <= stamp < end
            }
            if contribution - covered:
                needed.append(item)
                covered.update(contribution)
            else:
                duplicates.append({**item, "rejected_in_favor_of": "preceding cycle terminal record"})
        selected = needed

    cadence = 360 if request["product"] == "stations" else 3600
    expected = _expected_times(start, end, cadence)
    coverage: dict[datetime, list[str]] = {}
    for item in selected:
        for stamp in _nominal_item_times(item):
            if start <= stamp < end:
                coverage.setdefault(stamp, []).append(item["key"])
    missing = [iso_utc(stamp) for stamp in expected if stamp not in coverage]
    duplicate_times = [
        {"time_utc": iso_utc(stamp), "keys": keys}
        for stamp, keys in sorted(coverage.items())
        if len(keys) > 1
    ]
    if missing and request["missing_policy"] == "error":
        raise RuntimeError(
            f"source inventory is missing {len(missing)} required timestamps: {', '.join(missing[:8])}"
        )
    if not selected:
        raise RuntimeError(f"source inventory did not contain matching {config.model.upper()} objects")
    return {
        "selected": selected,
        "duplicate_objects": duplicates,
        "missing_times": missing,
        "duplicate_times": duplicate_times,
        "nominal_time_count": len(expected),
        "coverage_note": "filename times are nominal; downloaded NetCDF coordinates are authoritative",
    }


def normalize_request_input(request: Mapping[str, Any] | str | Path, config: ModelConfig) -> dict[str, Any]:
    return load_request(request, config) if isinstance(request, (str, Path)) else validate_request(request, config)


def inventory_request(
    request: Mapping[str, Any] | str | Path,
    config: ModelConfig,
    *,
    output: str | Path | None = None,
    objects: Sequence[Mapping[str, Any]] | None = None,
    session: Any | None = None,
    endpoint: str = S3_ENDPOINT,
) -> dict[str, Any]:
    migration = request_migration(request, config)
    normalized = normalize_request_input(request, config)
    if objects is None:
        raw_objects, discovery = discover_objects_with_evidence(
            normalized, config, session=session, endpoint=endpoint)
    else:
        raw_objects = list(objects)
        gaps = _coverage_gaps(normalized, raw_objects, config)
        discovery = {"source_attempts": [{"source_id": "provided", "status": "provided",
                                           "prefixes": [], "object_count": len(raw_objects), "error": None}],
                     "fallback_triggered": False, "fallback_reason": None,
                     "coverage_before_fallback": gaps, "coverage_after_fallback": gaps,
                     "ncei_filled_times": [], "unresolved_times": gaps}
    discovered = [_decorate_source(config, item, str(item.get("source_id") or "aws_operational"))
                  for item in raw_objects]
    capable, reason = _ncei_capability(normalized)
    report = {
        "schema_version": f"{config.model}_inventory_v2",
        "created_utc": iso_utc(datetime.now(UTC)),
        "request": normalized,
        **migration,
        "source_policy": normalized["source_policy"],
        **discovery,
        "prefixes": discovery_prefixes(normalized, config),
        "object_count": len(discovered),
        "objects": discovered,
        "source": {"policy": normalized["source_policy"], "access": "anonymous_https_listobjectsv2"},
    }
    if output:
        write_json_atomic(output, report)
    return report


def plan_request(
    request: Mapping[str, Any] | str | Path,
    run_dir: str | Path,
    config: ModelConfig,
    *,
    output: str | Path | None = None,
    objects: Sequence[Mapping[str, Any]] | None = None,
    session: Any | None = None,
    endpoint: str = S3_ENDPOINT,
) -> dict[str, Any]:
    migration = request_migration(request, config)
    normalized = normalize_request_input(request, config)
    capable, reason = _ncei_capability(normalized)
    if normalized["source_policy"] == "ncei_only" and not capable:
        raise ValueError(reason)
    if objects is None:
        raw_objects, discovery = discover_objects_with_evidence(
            normalized, config, session=session, endpoint=endpoint)
    else:
        raw_objects = list(objects)
        gaps = _coverage_gaps(normalized, raw_objects, config)
        discovery = {"source_attempts": [{"source_id": "provided", "status": "provided",
                                           "prefixes": [], "object_count": len(raw_objects), "error": None}],
                     "fallback_triggered": False, "fallback_reason": None,
                     "coverage_before_fallback": gaps, "coverage_after_fallback": gaps,
                     "ncei_filled_times": [], "unresolved_times": gaps}
    discovered = [_decorate_source(config, item, str(item.get("source_id") or "aws_operational"))
                  for item in raw_objects]
    selection = select_objects(normalized, discovered, config)
    def exact_positive_size(item: Mapping[str, Any]) -> int | None:
        size = item.get("size")
        return size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else None

    sizes = [exact_positive_size(item) for item in selection["selected"]]
    incomplete = [item["key"] for item, size in zip(selection["selected"], sizes) if size is None]
    metadata_incomplete: list[dict[str, str]] = []
    source_contract_errors: list[dict[str, str]] = []
    for item in selection["selected"]:
        key = str(item.get("key", ""))
        if not _clean_etag(item.get("etag")):
            metadata_incomplete.append({"key": key, "field": "etag"})
        if not isinstance(item.get("last_modified"), str) or not item["last_modified"].strip():
            metadata_incomplete.append({"key": key, "field": "last_modified"})
        try:
            _approved_source_contract(item, config)
        except (RuntimeError, ValueError) as exc:
            source_contract_errors.append({"key": key, "error": str(exc)})
    known_bytes = sum(size for size in sizes if size is not None)
    total = None if incomplete else known_bytes
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(run_path).free)
    required = None if total is None else 4 * total
    if incomplete:
        route, reason = "review", "estimate_incomplete"
    elif metadata_incomplete:
        route, reason = "review", "source_metadata_incomplete"
    elif source_contract_errors:
        route, reason = "review", "source_contract_invalid"
    elif required is not None and free > required:
        route, reason = "local", "local_free_bytes_exceeds_four_times_exact_request_bytes"
    else:
        route, reason = "kestrel", "local_free_space_gate_failed"
    report = {
        "schema_version": f"{config.model}_download_estimate_v2",
        "created_utc": iso_utc(datetime.now(UTC)),
        "request": normalized,
        "request_sha256": canonical_json_sha256(normalized),
        "objects_sha256": source_objects_sha256(selection["selected"]),
        **migration,
        "source_policy": normalized["source_policy"],
        **discovery,
        "selected_source_counts": {
            source_id: sum(item.get("source_id") == source_id for item in selection["selected"])
            for source_id in ("aws_operational", "ncei_long_term")
        },
        "source_totals": {
            source_id: {
                "object_count": sum(item.get("source_id") == source_id
                                    for item in selection["selected"]),
                "bytes": sum(size for item, size in zip(selection["selected"], sizes)
                             if item.get("source_id") == source_id and size is not None),
            }
            for source_id in ("aws_operational", "ncei_long_term")
        },
        "source": {"policy": normalized["source_policy"], "access": "anonymous_https_listobjectsv2"},
        "objects": selection["selected"],
        "object_count": len(selection["selected"]),
        "total_bytes": total,
        "total_gib": None if total is None else total / 1024**3,
        "known_bytes": known_bytes,
        "incomplete_size_keys": incomplete,
        "incomplete_source_metadata": metadata_incomplete,
        "source_contract_errors": source_contract_errors,
        "missing_times": selection["missing_times"],
        "duplicate_times": selection["duplicate_times"],
        "duplicate_objects": selection["duplicate_objects"],
        "nominal_time_count": selection["nominal_time_count"],
        "coverage_note": selection["coverage_note"],
        "local_free_bytes": free,
        "required_free_bytes": required,
        "routing_decision": route,
        "routing_reason": reason,
        "kestrel_stage_hint": f"/scratch/yhuang168/oma_external_data_connectors/{config.connector_name}/<run-id>",
    }
    write_json_atomic(output or run_path / "download_estimate.json", report)
    return report


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_etag(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _validate_netcdf_payload(path: str | Path) -> None:
    """Reject exact-size/hash-matching bytes that are not an openable NetCDF."""
    source = Path(path)
    with source.open("rb") as stream:
        signature = stream.read(8)
    classic = signature[:4] in {b"CDF\x01", b"CDF\x02", b"CDF\x05"}
    hdf5 = signature == b"\x89HDF\r\n\x1a\n"
    if not (classic or hdf5):
        raise RuntimeError(f"downloaded object has no NetCDF/HDF5 signature: {source}")
    try:
        import netCDF4
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("netCDF4 is required for NetCDF payload validation") from exc
    try:
        # netCDF4/HDF5 metadata opens are not reliably thread-safe in the
        # Windows build used by this skill. Serialize only this short open /
        # traverse / close section; HTTP streaming and SHA-256 remain parallel.
        with _NETCDF_METADATA_LOCK:
            with netCDF4.Dataset(source) as dataset:
                # Force metadata traversal so a magic prefix alone cannot make
                # a truncated object look cache-valid.
                tuple(dataset.dimensions)
                tuple(dataset.variables)
    except Exception as exc:
        raise RuntimeError(f"downloaded object is not an openable NetCDF: {source}") from exc


def _validate_object_payload(item: Mapping[str, Any], path: str | Path) -> None:
    if str(item.get("key", "")).lower().endswith(".nc"):
        _validate_netcdf_payload(path)


@contextmanager
def _destination_lock(destination: Path):
    """Fail fast when another process owns this destination's partial file."""
    lock_path = destination.with_name(destination.name + ".transfer.lock")
    owner_token = uuid.uuid4().hex
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"another transfer owns {destination}; lock exists: {lock_path}"
        ) from exc
    owned_stat = os.fstat(descriptor)
    try:
        payload = json.dumps({
            "pid": os.getpid(),
            "created_utc": iso_utc(datetime.now(UTC)),
            "destination": str(destination.resolve()),
            "owner_token": owner_token,
        }) + "\n"
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
        yield lock_path
    finally:
        os.close(descriptor)
        # Never unlink a path that was replaced while this context was active.
        # The token protects filesystems with weak inode reporting; the file
        # identity protects against a copied/replayed token.
        try:
            current_stat = lock_path.stat()
            current = read_json(lock_path)
            same_identity = (current_stat.st_dev, current_stat.st_ino) == (
                owned_stat.st_dev, owned_stat.st_ino)
            if (same_identity and isinstance(current, Mapping)
                    and current.get("owner_token") == owner_token):
                lock_path.unlink(missing_ok=True)
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
            pass


def _destination_for_key(run_dir: Path, key: str, config: ModelConfig,
                         item: Mapping[str, Any] | None = None) -> Path:
    prefix = f"{config.model}/netcdf/"
    relative = archive_sources.cache_relpath(item) if item is not None else (
        key[len(prefix) :] if key.startswith(prefix) else Path(key).name)
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe S3 key path: {key!r}")
    return run_dir.joinpath("cache", "raw", *parts)


def legacy_aws_cache_result(config: ModelConfig, item: Mapping[str, Any],
                            run_dir: str | Path) -> dict[str, Any] | None:
    """Reuse a fully verified pre-v2 AWS cache in place, without rewriting it."""
    prefix = f"{config.model}/netcdf/"
    if item.get("source_id") != "aws_operational" or not str(item.get("key", "")).startswith(prefix):
        return None
    relative = Path(str(item["key"])[len(prefix):])
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    destination = Path(run_dir) / "cache" / "raw" / relative
    sidecar_path = _sidecar(destination)
    if not destination.is_file() or not sidecar_path.is_file():
        return None
    try:
        metadata = read_json(sidecar_path)
    except Exception:
        return None
    expected = {
        "schema_version": f"{config.model}_cached_object_v1",
        "key": item.get("key"), "url": item.get("url"), "size": item.get("size"),
        "etag": _clean_etag(item.get("etag")),
        "last_modified": item.get("last_modified"),
    }
    if any((_clean_etag(metadata.get(name)) if name == "etag" else metadata.get(name)) != value
           for name, value in expected.items()):
        return None
    digest = metadata.get("sha256")
    if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or destination.stat().st_size != item.get("size")
            or sha256_file(destination) != digest):
        return None
    try:
        _validate_object_payload(item, destination)
    except Exception:
        return None
    return {
        "key": item["key"], "url": item["url"],
        "local_path": str(destination.resolve()), "status": "cache_hit",
        "cache_location": "legacy_aws_v1", "size": item["size"],
        "etag": _clean_etag(item["etag"]), "sha256": digest,
        "resumed": False, "resumed_from_bytes": 0, "retry_count": 0,
        "source": dict(item),
    }


def _sidecar(destination: Path) -> Path:
    return destination.with_name(destination.name + ".download.json")


def _partial_sidecar(destination: Path) -> Path:
    return destination.with_name(destination.name + ".part.json")


def cache_result(item: Mapping[str, Any], destination: Path) -> dict[str, Any] | None:
    if not destination.is_file() or not _sidecar(destination).is_file():
        return None
    try:
        metadata = read_json(_sidecar(destination))
    except Exception:
        return None
    expected_size = int(item.get("size", -1))
    expected_etag = _clean_etag(item.get("etag"))
    if not expected_etag:
        raise RuntimeError(f"object has no planned ETag: {item.get('key')}")
    if destination.stat().st_size != expected_size or int(metadata.get("size", -2)) != expected_size:
        return None
    if expected_etag and _clean_etag(metadata.get("etag")) != expected_etag:
        return None
    if (metadata.get("source_id") != item.get("source_id")
            or metadata.get("source_identity") != item.get("source_identity")):
        return None
    expected_hash = str(metadata.get("sha256", ""))
    if len(expected_hash) != 64 or sha256_file(destination) != expected_hash:
        return None
    try:
        _validate_object_payload(item, destination)
    except Exception:
        return None
    return {
        "key": item["key"], "url": item["url"], "local_path": str(destination.resolve()),
        "status": "cache_hit", "size": expected_size, "etag": expected_etag,
        "sha256": expected_hash, "resumed": False, "resumed_from_bytes": 0,
        "retry_count": 0, "source": dict(item),
    }


def _download_object_locked(
    item: Mapping[str, Any],
    destination: str | Path,
    *,
    session: Any | None = None,
    timeout: float = 120.0,
    max_attempts: int = 4,
    chunk_size: int = 4 * 1024 * 1024,
    schema_prefix: str = "roms_ofs",
) -> dict[str, Any]:
    requests = _requests_module()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_size = item.get("size")
    if not isinstance(raw_size, int) or isinstance(raw_size, bool) or raw_size <= 0:
        raise RuntimeError(f"object has no exact positive size: {item.get('key')}")
    expected_size = raw_size
    cached = cache_result(item, destination)
    if cached is not None:
        return cached
    expected_etag = _clean_etag(item.get("etag"))
    if not expected_etag:
        raise RuntimeError(f"object has no planned ETag: {item.get('key')}")
    partial = destination.with_name(destination.name + ".part")
    partial_metadata_path = _partial_sidecar(destination)
    partial_metadata = {
        "schema_version": f"{schema_prefix}_partial_object_v1",
        "source_id": item.get("source_id"),
        "source_identity": item.get("source_identity"),
        "key": item["key"], "url": item["url"], "size": expected_size,
        "etag": expected_etag, "last_modified": item.get("last_modified"),
    }
    if partial.exists():
        try:
            existing = read_json(partial_metadata_path)
        except Exception:
            existing = None
        if not isinstance(existing, Mapping) or any(
                existing.get(key) != partial_metadata[key] for key in partial_metadata):
            partial.unlink(missing_ok=True)
            partial_metadata_path.unlink(missing_ok=True)
    if not partial.exists():
        write_json_atomic(partial_metadata_path, partial_metadata)
    initial_partial_bytes = partial.stat().st_size if partial.exists() else 0
    if initial_partial_bytes > expected_size:
        partial.unlink()
        initial_partial_bytes = 0
    client = session or requests.Session()
    errors: list[str] = []
    last_resume_from = initial_partial_bytes
    discarded_invalid_partial = False
    for attempt in range(max_attempts):
        current = partial.stat().st_size if partial.exists() else 0
        if current > expected_size:
            partial.unlink(missing_ok=True)
            current = 0
        resume_used = current
        last_resume_from = resume_used
        response = None
        try:
            if current == expected_size:
                response = None
            else:
                headers = {"Range": f"bytes={current}-"} if current else {}
                response = client.get(item["url"], headers=headers, stream=True, timeout=timeout)
                response.raise_for_status()
                if current and response.status_code != 206:
                    partial.unlink(missing_ok=True)
                    current = 0
                    resume_used = 0
                    last_resume_from = 0
                mode = "ab" if current and response.status_code == 206 else "wb"
                response_etag = _clean_etag(response.headers.get("ETag"))
                if not response_etag:
                    raise RuntimeError("transfer response has no ETag")
                if response_etag != expected_etag:
                    raise RuntimeError(f"ETag changed during transfer: {response_etag} != {expected_etag}")
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    expected_response = expected_size - current if mode == "ab" else expected_size
                    if int(content_length) != expected_response:
                        raise RuntimeError(f"Content-Length mismatch: {content_length} != {expected_response}")
                if mode == "ab":
                    content_range = response.headers.get("Content-Range")
                    expected_range = f"bytes {current}-{expected_size - 1}/{expected_size}"
                    if content_range is not None and content_range != expected_range:
                        raise RuntimeError(f"Content-Range mismatch: {content_range} != {expected_range}")
                archive_sources.validate_download_response(response, item, offset=current)
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            stream.write(chunk)
                close = getattr(response, "close", None)
                if close is not None:
                    close()
                response = None
            if partial.stat().st_size != expected_size:
                raise RuntimeError(f"downloaded size mismatch: {partial.stat().st_size} != {expected_size}")
            digest = sha256_file(partial)
            try:
                _validate_object_payload(item, partial)
            except Exception:
                # A provenance-matched complete partial can still contain
                # corrupt bytes. Discard it so the next attempt transfers from
                # byte zero instead of revalidating the same payload.
                partial.unlink(missing_ok=True)
                partial_metadata_path.unlink(missing_ok=True)
                discarded_invalid_partial = True
                if attempt + 1 < max_attempts:
                    write_json_atomic(partial_metadata_path, partial_metadata)
                raise
            os.replace(partial, destination)
            partial_metadata_path.unlink(missing_ok=True)
            write_json_atomic(_sidecar(destination), {
                "schema_version": f"{schema_prefix}_cached_object_v1",
                "model": schema_prefix,
                "source_id": item.get("source_id"),
                "provider": item.get("provider"),
                "archive_role": item.get("archive_role"),
                "container": item.get("container"),
                "endpoint": item.get("endpoint"),
                "listing_endpoint": item.get("listing_endpoint"),
                "source_identity": item.get("source_identity"),
                "key": item["key"], "url": item["url"], "size": expected_size,
                "etag": expected_etag, "etag_is_multipart": "-" in expected_etag,
                "etag_semantics": "opaque_provenance",
                "last_modified": item.get("last_modified"), "sha256": digest,
                "netcdf_openable": str(item.get("key", "")).lower().endswith(".nc"),
                "completed_utc": iso_utc(datetime.now(UTC)),
            })
            return {
                "key": item["key"], "url": item["url"], "local_path": str(destination.resolve()),
                "status": "downloaded", "size": expected_size, "etag": expected_etag,
                "sha256": digest, "resumed": resume_used > 0,
                "resumed_from_bytes": resume_used, "retry_count": attempt,
                "discarded_invalid_partial": discarded_invalid_partial, "source": dict(item),
            }
        except Exception as exc:
            errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            if attempt + 1 < max_attempts:
                time.sleep(min(2**attempt, 4))
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if close is not None:
                    close()
    return {
        "key": item.get("key"), "url": item.get("url"), "local_path": str(destination.resolve()),
        "status": "failed", "size": expected_size, "etag": expected_etag,
        "resumed": last_resume_from > 0, "resumed_from_bytes": last_resume_from,
        "retry_count": max(0, max_attempts - 1),
        "discarded_invalid_partial": discarded_invalid_partial,
        "errors": errors, "source": dict(item),
    }


def download_object(
    item: Mapping[str, Any],
    destination: str | Path,
    *,
    session: Any | None = None,
    timeout: float = 120.0,
    max_attempts: int = 4,
    chunk_size: int = 4 * 1024 * 1024,
    schema_prefix: str = "roms_ofs",
) -> dict[str, Any]:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = cache_result(item, destination)
    if cached is not None:
        return cached
    with _destination_lock(destination):
        # A winning process may have completed between the optimistic cache
        # check above and this process acquiring the exclusive lock.
        cached = cache_result(item, destination)
        if cached is not None:
            return cached
        return _download_object_locked(
            item,
            destination,
            session=session,
            timeout=timeout,
            max_attempts=max_attempts,
            chunk_size=chunk_size,
            schema_prefix=schema_prefix,
        )


def fetch_from_plan(
    plan: Mapping[str, Any] | str | Path,
    run_dir: str | Path,
    config: ModelConfig,
    *,
    session: Any | None = None,
) -> dict[str, Any]:
    """Transfer only the objects in a previously written, locally approved plan."""
    if not isinstance(plan, (str, Path)):
        raise RuntimeError("fetch requires an existing reviewed plan file path, not an in-memory mapping")
    plan_path = Path(plan).resolve()
    if not plan_path.is_file():
        raise RuntimeError("fetch requires an existing reviewed plan file")
    estimate = read_json(plan_path)
    if estimate.get("schema_version") != f"{config.model}_download_estimate_v2":
        raise RuntimeError("fetch requires a connector-generated download estimate")
    normalized = validate_request(estimate.get("request", {}), config)
    if estimate.get("source_policy") != normalized["source_policy"]:
        raise RuntimeError("approved plan source_policy does not match its request")
    request_sha256 = canonical_json_sha256(normalized)
    if estimate.get("request_sha256") != request_sha256:
        raise RuntimeError("approved plan request_sha256 does not match its normalized request")
    if estimate.get("routing_decision") != "local":
        raise RuntimeError(
            f"local fetch is not approved: {estimate.get('routing_decision')} "
            f"({estimate.get('routing_reason')})"
        )
    objects = estimate.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RuntimeError("approved plan contains no source objects")
    if estimate.get("object_count") != len(objects):
        raise RuntimeError("approved plan object_count does not match its source objects")
    sizes: list[int] = []
    for item in objects:
        if not isinstance(item, Mapping):
            raise RuntimeError("approved plan contains a non-object source entry")
        size = item.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise RuntimeError("approved plan does not contain exact positive object sizes")
        try:
            _approved_source_contract(item, config)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"approved plan contains invalid source provenance: {exc}") from exc
        sizes.append(size)
    exact_total = sum(sizes)
    objects_sha256 = source_objects_sha256(objects)
    if estimate.get("objects_sha256") != objects_sha256:
        raise RuntimeError("approved plan objects_sha256 does not match its selected source objects")
    expected_source_totals = {
        source_id: {
            "object_count": sum(item.get("source_id") == source_id for item in objects),
            "bytes": sum(item["size"] for item in objects if item.get("source_id") == source_id),
        }
        for source_id in ("aws_operational", "ncei_long_term")
    }
    if estimate.get("source_totals") != expected_source_totals:
        raise RuntimeError("approved plan per-source object/byte totals are inconsistent")
    fallback_failures = validate_fallback_decision(normalized, estimate, objects)
    if fallback_failures:
        raise RuntimeError("approved plan fallback decision is invalid: " + "; ".join(fallback_failures))
    exact_required = 4 * exact_total
    if estimate.get("total_bytes") != exact_total:
        raise RuntimeError("approved plan total_bytes does not match its source objects")
    if estimate.get("required_free_bytes") != exact_required:
        raise RuntimeError("approved plan required_free_bytes is not exactly four times total_bytes")
    reselection = select_objects(normalized, objects, config)
    if [item["key"] for item in reselection["selected"]] != [item["key"] for item in objects]:
        raise RuntimeError("approved plan object selection is inconsistent with its request")
    plan_sha256 = sha256_file(plan_path)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    current_free = int(shutil.disk_usage(run_path).free)
    if current_free <= exact_required:
        raise RuntimeError(
            f"local free-space gate failed immediately before transfer: "
            f"{current_free} <= {exact_required} bytes"
        )

    def transfer(item: Mapping[str, Any]) -> dict[str, Any]:
        destination = _destination_for_key(run_path, str(item["key"]), config, item)
        cached = cache_result(item, destination)
        if cached is not None:
            return cached
        legacy_cached = legacy_aws_cache_result(config, item, run_path)
        if legacy_cached is not None:
            return legacy_cached
        source_id = str(item["source_id"])
        exact = [candidate for candidate in archive_sources.list_objects_v2(
            source_id, config.model, str(item["key"]), session=session, max_keys=2,
        ) if candidate.get("key") == item.get("key")]
        if len(exact) != 1:
            raise RuntimeError(
                "planned source object is no longer uniquely listed; replan before fetching")
        archive_sources.validate_remote_metadata(item, exact[0])
        return download_object(
            item,
            destination,
            session=session,
            schema_prefix=config.model,
        )

    outcomes: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=normalized["max_workers"]) as pool:
        futures = [pool.submit(transfer, item) for item in estimate["objects"]]
        for future in concurrent.futures.as_completed(futures):
            outcomes.append(future.result())
    outcomes.sort(key=lambda item: str(item.get("key")))
    failures = [item for item in outcomes if item["status"] == "failed"]
    manifest = {
        "schema_version": f"{config.model}_fetch_manifest_v2",
        "created_utc": iso_utc(datetime.now(UTC)),
        "request": normalized,
        "source_policy": normalized["source_policy"],
        "approved_plan": {
            "path": str(plan_path),
            "sha256": plan_sha256,
            "schema_version": estimate["schema_version"],
            "request_sha256": request_sha256,
            "objects_sha256": objects_sha256,
            "object_count": len(objects),
            "total_bytes": exact_total,
        },
        "transfer_storage_gate": {
            "total_bytes": exact_total,
            "required_free_bytes": exact_required,
            "current_free_bytes": current_free,
            "rule": "current_free_bytes > 4 * exact_total_bytes",
            "decision": "local",
        },
        "outcomes": outcomes,
        "counts": {
            "objects": len(outcomes),
            "downloaded": sum(item["status"] == "downloaded" for item in outcomes),
            "cache_hits": sum(item["status"] == "cache_hit" for item in outcomes),
            "failed": len(failures),
            "resumed": sum(bool(item.get("resumed")) for item in outcomes),
        },
        "source_provenance": {
            "policy": normalized["source_policy"],
            "source_counts": {source_id: sum(item.get("source", {}).get("source_id") == source_id for item in outcomes)
                              for source_id in ("aws_operational", "ncei_long_term")},
            "access": "anonymous_https",
        },
    }
    write_json_atomic(run_path / "fetch_manifest.json", manifest)
    if failures:
        raise RuntimeError(f"{len(failures)} {config.model.upper()} transfers failed; inspect fetch_manifest.json")
    return manifest


def fetch_request(
    request: Mapping[str, Any] | str | Path,
    run_dir: str | Path,
    config: ModelConfig,
    *,
    objects: Sequence[Mapping[str, Any]] | None = None,
    session: Any | None = None,
    endpoint: str = S3_ENDPOINT,
) -> dict[str, Any]:
    raise RuntimeError(
        "direct request-to-transfer is disabled; write and review a plan, then use fetch_from_plan"
    )


def verify_legacy_v1_transfers(run_path: Path, manifest: Mapping[str, Any],
                               manifest_path: Path,
                               expected_request: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read and integrity-check immutable pre-v2 AWS evidence; never authorizes transfer."""
    failures: list[str] = []
    try:
        normalized = validate_request(manifest.get("request", {}), ModelConfig(
            model="dbofs", request_schema="dbofs_request_v2",
            connector_name="dbofs-fetcher"))
    except Exception as exc:
        return {"status": "fail", "objects": [], "failures": [f"legacy request is invalid: {exc}"]}
    if expected_request is not None and canonical_json_sha256(normalized) != canonical_json_sha256(expected_request):
        failures.append("legacy fetch-manifest request does not match health request")
    approved = manifest.get("approved_plan")
    if not isinstance(approved, Mapping) or approved.get("schema_version") != "dbofs_download_estimate_v1":
        failures.append("legacy approved plan schema is invalid")
    else:
        plan_path = Path(str(approved.get("path", ""))).resolve()
        if not plan_path.is_file() or sha256_file(plan_path) != approved.get("sha256"):
            failures.append("legacy approved plan path/SHA-256 is invalid")
        else:
            plan = read_json(plan_path)
            if plan.get("schema_version") != "dbofs_download_estimate_v1":
                failures.append("legacy approved plan file schema is invalid")
    outcomes = manifest.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        failures.append("legacy fetch manifest has no outcomes")
        outcomes = []
    raw_root = (run_path / "cache" / "raw").resolve()
    checks: list[dict[str, Any]] = []
    for outcome in outcomes:
        source = outcome.get("source") if isinstance(outcome, Mapping) else None
        key = source.get("key") if isinstance(source, Mapping) else None
        parsed = parse_object_key(str(key or ""), ModelConfig(
            model="dbofs", request_schema="dbofs_request_v2",
            connector_name="dbofs-fetcher"))
        path = Path(str(outcome.get("local_path", ""))).resolve() if isinstance(outcome, Mapping) else Path()
        reason = None
        try:
            path.relative_to(raw_root)
        except ValueError:
            reason = "legacy local path is outside raw cache"
        expected_url = S3_ENDPOINT + "/" + quote(str(key), safe="/") if key else None
        if parsed is None or source.get("url") != expected_url:
            reason = reason or "legacy AWS source key/URL is invalid"
        size, digest = outcome.get("size"), outcome.get("sha256")
        sidecar_path = path.with_name(path.name + ".download.json")
        if not path.is_file() or not sidecar_path.is_file():
            reason = reason or "legacy raw file/sidecar is missing"
        else:
            sidecar = read_json(sidecar_path)
            expected = {
                "schema_version": "dbofs_cached_object_v1", "key": key,
                "url": expected_url, "size": size,
                "etag": _clean_etag(outcome.get("etag")),
                "last_modified": source.get("last_modified"), "sha256": digest,
            }
            if any((_clean_etag(sidecar.get(name)) if name == "etag" else sidecar.get(name)) != value
                   for name, value in expected.items()):
                reason = reason or "legacy cache sidecar provenance mismatch"
            elif path.stat().st_size != size or sha256_file(path) != digest:
                reason = reason or "legacy raw size/SHA-256 mismatch"
            else:
                try:
                    _validate_object_payload(source, path)
                except Exception as exc:
                    reason = reason or f"legacy raw NetCDF is invalid: {exc}"
        check = {"key": key, "local_path": str(path),
                 "status": "fail" if reason else "pass", "size": size,
                 "sha256": digest, "etag": outcome.get("etag"),
                 "source_id": "aws_operational", "source_archive": "aws_operational",
                 "endpoint": S3_ENDPOINT, "url": expected_url,
                 "legacy_evidence_schema": "v1"}
        if reason:
            check["reason"] = reason
            failures.append(f"{key}: {reason}")
        checks.append(check)
    return {"status": "pass" if not failures else "fail", "objects": checks,
            "failures": failures, "legacy_read_only": True,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path), "request": normalized}


def verify_transfers(
    run_dir: str | Path,
    expected_request: Mapping[str, Any] | None = None,
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    manifest_path = Path(manifest_path).resolve() if manifest_path else run_path / "fetch_manifest.json"
    if not manifest_path.is_file():
        return {"status": "missing", "objects": [], "failures": ["fetch_manifest.json is missing"]}
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        return {"status": "fail", "objects": [], "failures": [f"fetch manifest is unreadable: {type(exc).__name__}: {exc}"]}
    if manifest.get("schema_version") == "dbofs_fetch_manifest_v1":
        return verify_legacy_v1_transfers(
            run_path, manifest, manifest_path, expected_request)
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    manifest_request = manifest.get("request")
    request_schema = manifest_request.get("schema_version") if isinstance(manifest_request, Mapping) else None
    model = str(request_schema).removesuffix("_request_v2") if isinstance(request_schema, str) else ""
    if not model or manifest.get("schema_version") != f"{model}_fetch_manifest_v2":
        failures.append("fetch manifest schema_version is missing or inconsistent")
    if expected_request is not None and canonical_json_sha256(manifest_request) != canonical_json_sha256(expected_request):
        failures.append("fetch manifest request does not match the requested health/extraction contract")
    approved = manifest.get("approved_plan")
    approved_plan: Mapping[str, Any] | None = None
    approved_objects: list[Mapping[str, Any]] | None = None
    if not isinstance(approved, Mapping) or len(str(approved.get("sha256", ""))) != 64:
        failures.append("fetch manifest has no approved-plan SHA-256 provenance")
    elif not isinstance(approved.get("path"), str) or not approved.get("path"):
        failures.append("fetch manifest has no approved-plan file path")
    else:
        approved_path = Path(approved["path"])
        if not approved_path.is_file():
            failures.append("approved plan file is missing")
        elif sha256_file(approved_path) != approved.get("sha256"):
            failures.append("approved plan SHA-256 mismatch")
        else:
            try:
                loaded_plan = read_json(approved_path)
            except Exception as exc:
                failures.append(
                    f"approved plan is unreadable: {type(exc).__name__}: {exc}"
                )
            else:
                if not isinstance(loaded_plan, Mapping):
                    failures.append("approved plan root is not a JSON object")
                else:
                    approved_plan = loaded_plan
                    expected_plan_schema = f"{model}_download_estimate_v2" if model else None
                    if (not expected_plan_schema
                            or approved_plan.get("schema_version") != expected_plan_schema
                            or approved.get("schema_version") != expected_plan_schema):
                        failures.append("approved plan schema_version is inconsistent")
                    plan_request = approved_plan.get("request")
                    if (not isinstance(plan_request, Mapping)
                            or not isinstance(manifest_request, Mapping)
                            or canonical_json_sha256(plan_request) != canonical_json_sha256(manifest_request)):
                        failures.append("approved plan request does not match the fetch manifest request")
                    elif (approved_plan.get("request_sha256") != canonical_json_sha256(plan_request)
                          or approved.get("request_sha256") != approved_plan.get("request_sha256")):
                        failures.append("approved plan request_sha256 is missing or inconsistent")
                    plan_objects = approved_plan.get("objects")
                    if not isinstance(plan_objects, list) or not plan_objects:
                        failures.append("approved plan has no selected source objects")
                    elif not all(isinstance(item, Mapping) for item in plan_objects):
                        failures.append("approved plan contains a non-object source entry")
                    else:
                        approved_objects = plan_objects
                        for source_object in approved_objects:
                            try:
                                _approved_source_contract(source_object, ModelConfig(
                                    model=model, request_schema=f"{model}_request_v2",
                                    connector_name=f"{model}-fetcher"))
                            except Exception as exc:
                                failures.append(f"approved plan source object is invalid: {exc}")
                        if isinstance(plan_request, Mapping):
                            failures.extend(validate_fallback_decision(
                                plan_request, approved_plan, approved_objects))
                        if approved_plan.get("object_count") != len(approved_objects):
                            failures.append("approved plan object_count is inconsistent")
                        sizes = [item.get("size") for item in approved_objects]
                        if (not all(isinstance(size, int) and not isinstance(size, bool) and size > 0
                                    for size in sizes)
                                or approved_plan.get("total_bytes") != sum(sizes)):
                            failures.append("approved plan total_bytes/source sizes are inconsistent")
                        if (approved_plan.get("objects_sha256") != source_objects_sha256(approved_objects)
                                or approved.get("objects_sha256") != approved_plan.get("objects_sha256")):
                            failures.append("approved plan objects_sha256 is missing or inconsistent")
                        if approved.get("object_count") != approved_plan.get("object_count"):
                            failures.append("fetch manifest approved object_count is inconsistent")
                        if approved.get("total_bytes") != approved_plan.get("total_bytes"):
                            failures.append("fetch manifest approved total_bytes is inconsistent")
                        expected_source_totals = {
                            source_id: {
                                "object_count": sum(item.get("source_id") == source_id
                                                    for item in approved_objects),
                                "bytes": sum(item.get("size", 0) for item in approved_objects
                                             if item.get("source_id") == source_id),
                            }
                            for source_id in ("aws_operational", "ncei_long_term")
                        }
                        if approved_plan.get("source_totals") != expected_source_totals:
                            failures.append("approved plan per-source object/byte totals are inconsistent")
    outcomes = manifest.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        failures.append("fetch manifest has no transfer outcomes")
        outcomes = []
    counts = manifest.get("counts", {})
    if isinstance(counts, Mapping) and counts.get("objects") != len(outcomes):
        failures.append("fetch manifest object count is inconsistent")
    if approved_objects is not None:
        plan_keys = [item.get("key") for item in approved_objects]
        outcome_keys = [item.get("key") for item in outcomes]
        if (not all(isinstance(key, str) and key for key in plan_keys + outcome_keys)
                or len(set(plan_keys)) != len(plan_keys)
                or len(set(outcome_keys)) != len(outcome_keys)
                or sorted(plan_keys) != sorted(outcome_keys)):
            failures.append("approved plan selected keys do not match fetch manifest outcomes")
        else:
            plan_by_key = {item["key"]: item for item in approved_objects}
            for outcome in outcomes:
                source = plan_by_key[outcome["key"]]
                if (outcome.get("size") != source.get("size")
                        or _clean_etag(outcome.get("etag")) != _clean_etag(source.get("etag"))
                        or outcome.get("url") != source.get("url")):
                    failures.append(
                        f"{outcome['key']}: fetch outcome does not match approved-plan provenance"
                    )
    raw_root = (run_path / "cache" / "raw").resolve()
    for outcome in outcomes:
        path = Path(str(outcome.get("local_path", ""))).resolve()
        source = outcome.get("source")
        check = {"key": outcome.get("key"), "local_path": str(path), "status": "pass"}
        try:
            path.relative_to(raw_root)
        except ValueError:
            check["status"] = "fail"
            check["reason"] = "local path is outside the run raw cache"
        size = outcome.get("size")
        digest = outcome.get("sha256")
        if check["status"] == "pass" and outcome.get("status") not in {"downloaded", "cache_hit"}:
            check["status"] = "fail"
            check["reason"] = "transfer outcome failed"
        elif check["status"] == "pass" and (not isinstance(size, int) or isinstance(size, bool) or size <= 0):
            check["status"] = "fail"
            check["reason"] = "manifest size is not an exact positive integer"
        elif check["status"] == "pass" and (not isinstance(digest, str) or len(digest) != 64):
            check["status"] = "fail"
            check["reason"] = "manifest SHA-256 is invalid"
        elif check["status"] == "pass" and not path.is_file():
            check["status"] = "fail"
            check["reason"] = "local file missing"
        elif check["status"] == "pass" and path.stat().st_size != size:
            check["status"] = "fail"
            check["reason"] = "size mismatch"
        elif check["status"] == "pass" and sha256_file(path) != digest:
            check["status"] = "fail"
            check["reason"] = "SHA-256 mismatch"
        elif check["status"] == "pass":
            sidecar = path.with_name(path.name + ".download.json")
            try:
                metadata = read_json(sidecar)
            except Exception as exc:
                check["status"] = "fail"
                check["reason"] = f"cache sidecar missing or unreadable: {type(exc).__name__}"
            else:
                legacy = outcome.get("cache_location") == "legacy_aws_v1"
                expected_sidecar = ({
                    "schema_version": f"{model}_cached_object_v1",
                    "key": outcome.get("key"), "url": outcome.get("url"),
                    "size": size, "etag": _clean_etag(outcome.get("etag")),
                    "last_modified": source.get("last_modified") if isinstance(source, Mapping) else None,
                    "sha256": digest,
                } if legacy else {
                    "schema_version": f"{model}_cached_object_v1",
                    "model": model,
                    "source_id": source.get("source_id") if isinstance(source, Mapping) else None,
                    "provider": source.get("provider") if isinstance(source, Mapping) else None,
                    "archive_role": source.get("archive_role") if isinstance(source, Mapping) else None,
                    "container": source.get("container") if isinstance(source, Mapping) else None,
                    "endpoint": source.get("endpoint") if isinstance(source, Mapping) else None,
                    "listing_endpoint": source.get("listing_endpoint") if isinstance(source, Mapping) else None,
                    "source_identity": source.get("source_identity") if isinstance(source, Mapping) else None,
                    "key": outcome.get("key"),
                    "url": outcome.get("url"),
                    "size": size,
                    "etag": _clean_etag(outcome.get("etag")),
                    "last_modified": source.get("last_modified") if isinstance(source, Mapping) else None,
                    "etag_semantics": "opaque_provenance",
                    "sha256": digest,
                })
                sidecar_matches = all(
                    (_clean_etag(metadata.get(name)) if name == "etag" else metadata.get(name)) == value
                    for name, value in expected_sidecar.items()
                )
                if legacy and (not isinstance(source, Mapping)
                               or source.get("source_id") != "aws_operational"):
                    sidecar_matches = False
                if sidecar_matches and not legacy:
                    try:
                        archive_sources.validate_source_object(
                            model, metadata,
                            expected_source_id=str(expected_sidecar["source_id"]),
                        )
                    except Exception:
                        sidecar_matches = False
                if not sidecar_matches:
                    check["status"] = "fail"
                    check["reason"] = "cache sidecar provenance mismatch"
        if check["status"] == "pass":
            if not isinstance(source, Mapping):
                check["status"] = "fail"
                check["reason"] = "fetch outcome has no source provenance"
            else:
                try:
                    descriptor = archive_sources.get_source_descriptor(
                        str(source.get("source_id")), model)
                    _approved_source_contract(source, ModelConfig(
                        model=model, request_schema=f"{model}_request_v2",
                        connector_name=f"{model}-fetcher"))
                except Exception as exc:
                    check["status"] = "fail"
                    check["reason"] = f"fetch source provenance is invalid: {exc}"
        if check["status"] == "pass":
            check["size"] = path.stat().st_size
            check["sha256"] = digest
            check["etag"] = outcome.get("etag")
            check["source_id"] = source.get("source_id")
            check["source_archive"] = source.get("source_id")
            check["archive_role"] = descriptor["archive_role"]
            check["container"] = descriptor["container"]
            check["endpoint"] = descriptor["endpoint"]
            check["listing_endpoint"] = descriptor["listing_endpoint"]
            check["url"] = source.get("url")
            check["last_modified"] = source.get("last_modified")
            check["source_identity"] = source.get("source_identity")
            check["cache_location"] = outcome.get("cache_location", "source_isolated_v2")
        checks.append(check)
        if check["status"] != "pass":
            failures.append(f"{check['key']}: {check.get('reason')}")
    return {
        "status": "pass" if not failures else "fail",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "request": manifest_request,
        "objects": checks,
        "failures": failures,
    }


def manifest_paths(
    run_dir: str | Path,
    expected_request: Mapping[str, Any] | None = None,
    *,
    manifest_path: str | Path | None = None,
) -> list[Path]:
    verification = verify_transfers(
        run_dir, expected_request=expected_request, manifest_path=manifest_path,
    )
    if verification["status"] != "pass":
        raise RuntimeError("fetch manifest/integrity gate failed: " + "; ".join(verification["failures"]))
    return [Path(item["local_path"]) for item in verification["objects"]]
