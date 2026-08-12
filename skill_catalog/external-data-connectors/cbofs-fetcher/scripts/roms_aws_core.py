#!/usr/bin/env python3
"""Shared anonymous-AWS inventory, planning, download, and inspection core.

This module is intentionally model-configured so CBOFS and DBOFS connector
packages can carry byte-identical copies of the transfer implementation.
"""

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
CYCLE_HOURS = {0, 6, 12, 18}
PRODUCTS = {"fields", "stations", "regulargrid"}
DEFAULT_VARIABLES = ["zeta", "salt", "u", "v"]
DEFAULT_VIEWS: list[str | int] = ["surface"]
SOURCE_POLICIES = {"aws_then_ncei", "aws_only", "ncei_only"}
_NETCDF_METADATA_LOCK = threading.Lock()


@dataclass(frozen=True)
class ModelConfig:
    model: str
    schema_version: str
    compact_filename: str
    display_name: str

    @property
    def prefix(self) -> str:
        return f"{self.model}/netcdf/"


def _source_ids(request: Mapping[str, Any]) -> list[str]:
    policy = request["source_policy"]
    if policy == "aws_only":
        return ["aws_operational"]
    if policy == "ncei_only":
        return ["ncei_long_term"]
    return ["aws_operational", "ncei_long_term"]


def _ncei_capability(request: Mapping[str, Any]) -> tuple[bool, str | None]:
    product, guidance = request["product"], request["guidance"]
    if product == "regulargrid":
        return False, "ncei_long_term does not support regulargrid"
    if product == "fields" and guidance == "forecast":
        return False, "ncei_long_term does not support native field forecasts"
    return True, None


def _decorate_source(config: ModelConfig, item: Mapping[str, Any],
                     source_id: str = "aws_operational") -> dict[str, Any]:
    descriptor = archive_sources.get_source_descriptor(source_id, config.model)
    value = {**descriptor, **dict(item), "source_id": source_id}
    value["provider"] = descriptor["provider"]
    value["archive_role"] = descriptor["archive_role"]
    value["container"] = descriptor["container"]
    value["endpoint"] = descriptor["endpoint"]
    value["listing_endpoint"] = descriptor["listing_endpoint"]
    value.setdefault("url", archive_sources.canonical_object_url(source_id, config.model, str(value["key"])))
    parsed = parse_object_key(config, str(value["key"]))
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


def _netcdf_modules():
    try:
        import netCDF4
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("netCDF4 and numpy are required for NetCDF inspection") from exc
    return netCDF4, np


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


def iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def json_clean(value: Any) -> Any:
    if isinstance(value, datetime):
        return iso(value)
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
    """Hash the complete approved source records independent of list ordering."""
    ordered = sorted((dict(item) for item in objects), key=lambda item: str(item.get("key", "")))
    return canonical_json_sha256(ordered)


def _expected_object_url(key: str) -> str:
    return S3_ENDPOINT + "/" + quote(key, safe="/")


def validate_source_object(config: ModelConfig, item: Mapping[str, Any], *,
                           require_remote_metadata: bool = True) -> dict[str, Any]:
    """Fail closed unless an object is an exact NOAA key and provenance record."""
    source_id = str(item.get("source_id") or "aws_operational")
    key = item.get("key")
    descriptor = archive_sources.get_source_descriptor(source_id, config.model)
    if not isinstance(key, str) or not key.startswith(descriptor["root_prefix"]):
        raise ValueError(f"source key must start with {descriptor['root_prefix']!r}")
    parsed = parse_object_key(config, key)
    if parsed is None:
        raise ValueError(f"invalid {config.model.upper()} object key: {key!r}")
    relative = key[len(descriptor["root_prefix"]):]
    run = parse_utc(parsed["run_time"])
    daily = f"{run:%Y/%m/%d}/{parsed['name']}"
    monthly = f"{run:%Y%m}/{parsed['name']}"
    ncei_monthly = f"{run:%Y/%m}/{parsed['name']}"
    if relative == daily:
        expected_layout = "daily"
    elif relative == monthly:
        expected_layout = "monthly"
    elif source_id == "ncei_long_term" and relative == ncei_monthly:
        expected_layout = "ncei_monthly"
    else:
        raise ValueError("source key layout/date does not match its filename run date")
    if parsed["layout"] != expected_layout:
        raise ValueError("parsed source layout is inconsistent")
    for name in ("model", "layout", "naming", "product", "guidance", "aggregate",
                 "run_time", "cycle_hour", "lead", "valid_time",
                 "expected_start_utc", "expected_end_utc_exclusive"):
        if name in item and item.get(name) != parsed.get(name):
            raise ValueError(f"source object {name} conflicts with its key")
    expected_url = archive_sources.canonical_object_url(source_id, config.model, key)
    url = item.get("url")
    if url is not None and url != expected_url:
        raise ValueError("source URL is not the exact NOAA archive URL for its key")
    archive_sources.validate_source_object(
        config.model, _decorate_source(config, item, source_id),
        expected_source_id=source_id, require_metadata=require_remote_metadata,
    )
    size = item.get("size")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size <= 0):
        raise ValueError("source size must be an exact positive integer")
    etag = _clean_etag(item.get("etag"))
    modified = item.get("last_modified")
    if require_remote_metadata:
        if url != expected_url:
            raise ValueError("source URL is missing")
        if size is None:
            raise ValueError("source size is missing")
        if not etag:
            raise ValueError("source ETag is missing")
        if not isinstance(modified, str) or not modified.strip():
            raise ValueError("source Last-Modified is missing")
        parse_utc(modified, "last_modified")
    return {**parsed, "url": expected_url, "size": size, "etag": etag,
            "last_modified": modified}


def load_request(config: ModelConfig, path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("request root must be a JSON object")
    return validate_request(config, value)


def validate_request(config: ModelConfig, mapping: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "schema_version", "start_utc", "end_utc_exclusive", "product",
        "guidance", "run_cycle_utc", "variables", "vertical_views",
        "missing_policy", "cache_policy", "max_workers", "source_policy",
    }
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown request properties: {', '.join(unknown)}")
    input_schema = mapping.get("schema_version")
    legacy_schema = f"{config.model}_request_v1"
    if input_schema not in {legacy_schema, config.schema_version}:
        raise ValueError(
            f"schema_version must be {legacy_schema!r} or {config.schema_version!r}"
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
    if product not in PRODUCTS:
        raise ValueError("product must be fields, stations, or regulargrid")
    guidance = mapping.get("guidance")
    if guidance not in {"nowcast", "forecast"}:
        raise ValueError("guidance must be nowcast or forecast")
    run_cycle: datetime | None = None
    if guidance == "forecast":
        if "run_cycle_utc" not in mapping:
            raise ValueError("run_cycle_utc is required for forecast requests")
        run_cycle = parse_utc(mapping["run_cycle_utc"], "run_cycle_utc")
        if (run_cycle.hour not in CYCLE_HOURS or run_cycle.minute or
                run_cycle.second or run_cycle.microsecond):
            raise ValueError("run_cycle_utc must be exactly 00, 06, 12, or 18 UTC")
    elif "run_cycle_utc" in mapping:
        raise ValueError("run_cycle_utc is permitted only for forecast requests")

    if product != "fields":
        if "variables" in mapping or "vertical_views" in mapping:
            raise ValueError(f"{product} is passthrough-only; variables and vertical_views are not allowed")
        variables: list[str] = []
        views: list[str | int] = []
    else:
        raw_variables = mapping.get("variables", DEFAULT_VARIABLES)
        if (not isinstance(raw_variables, list) or not raw_variables or
                any(not isinstance(v, str) or not v.strip() for v in raw_variables)):
            raise ValueError("variables must be a non-empty array of source names or aliases")
        aliases = {"salinity": "salt", "temperature": "temp"}
        variables = [aliases.get(v.strip().lower(), v.strip()) for v in raw_variables]
        if len(set(variables)) != len(variables):
            raise ValueError("variables must be unique after alias normalization")
        if ("u" in variables) != ("v" in variables):
            raise ValueError("u and v must be requested together")
        raw_views = mapping.get("vertical_views", DEFAULT_VIEWS)
        if not isinstance(raw_views, list) or not raw_views:
            raise ValueError("vertical_views must be a non-empty array")
        views = []
        valid = {"surface", "near_surface", "bottom", "depth_average"}
        for view in raw_views:
            if isinstance(view, bool) or not isinstance(view, (str, int)):
                raise ValueError("vertical views must be named strings or non-negative indices")
            if isinstance(view, int) and view < 0:
                raise ValueError("explicit sigma indices must be non-negative")
            if isinstance(view, str) and view not in valid:
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
    workers = mapping.get("max_workers", 4)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("max_workers must be a positive integer")
    source_policy = mapping.get("source_policy", "aws_then_ncei")
    if source_policy not in SOURCE_POLICIES:
        raise ValueError(
            "source_policy must be aws_then_ncei, aws_only, or ncei_only"
        )
    normalized: dict[str, Any] = {
        "schema_version": config.schema_version,
        "start_utc": iso(start), "end_utc_exclusive": iso(end),
        "product": product, "guidance": guidance,
        "missing_policy": missing_policy, "cache_policy": cache_policy,
        "max_workers": workers, "source_policy": source_policy,
    }
    if run_cycle:
        normalized["run_cycle_utc"] = iso(run_cycle)
    if product == "fields":
        normalized["variables"] = variables
        normalized["vertical_views"] = views
    return normalized


def request_migration(config: ModelConfig,
                      request: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Describe the lossless v1-to-v2 normalization recorded in v2 artifacts."""
    raw = read_json(request) if isinstance(request, (str, Path)) else dict(request)
    input_schema = raw.get("schema_version")
    migrated = input_schema != config.schema_version
    return {
        "input_schema_version": input_schema,
        "normalized_schema_version": config.schema_version,
        "migration": (
            {
                "applied": True,
                "name": f"{config.model}_request_v1_to_v2",
                "defaults_applied": {"source_policy": "aws_then_ncei"},
            }
            if migrated else {"applied": False, "name": None, "defaults_applied": {}}
        ),
    }


def _layout(key: str) -> str:
    if re.search(r"/\d{4}/\d{2}/\d{2}/", key):
        return "daily"
    if re.search(r"/access/[^/]+/\d{4}/\d{2}/", key):
        return "ncei_monthly"
    if re.search(r"/\d{6}/", key):
        return "monthly"
    return "unknown"


def parse_object_key(config: ModelConfig, key: str) -> dict[str, Any] | None:
    model = re.escape(config.model)
    current = re.compile(
        rf"^{model}\.t(?P<hour>\d{{2}})z\.(?P<date>\d{{8}})\."
        rf"(?P<product>fields|stations|regulargrid)\."
        rf"(?:(?P<code>[nf])(?P<lead>\d{{3}})|(?P<aggregate>nowcast|forecast))\.nc$",
        re.IGNORECASE,
    )
    legacy = re.compile(
        rf"^(?:nos\.)?{model}\.(?P<product>fields|stations|regulargrid)\."
        rf"(?:(?P<code>[nf])(?P<lead>\d{{3}})|(?P<aggregate>nowcast|forecast))\."
        rf"(?P<date>\d{{8}})\.t(?P<hour>\d{{2}})z\.nc$",
        re.IGNORECASE,
    )
    name = Path(key).name
    match = current.match(name)
    naming = "current"
    if match is None:
        match = legacy.match(name)
        naming = "legacy"
    if match is None:
        return None
    values = match.groupdict()
    try:
        run = datetime.strptime(values["date"] + values["hour"], "%Y%m%d%H").replace(tzinfo=UTC)
    except ValueError:
        return None
    product = values["product"].lower()
    if run.hour not in CYCLE_HOURS:
        return None
    if values.get("code"):
        code = values["code"].lower()
        lead = int(values["lead"])
        # Native field and regular-grid objects are hourly single-record
        # products.  Station products are cycle aggregates, not n/f objects.
        if product == "stations":
            return None
        source_id = "ncei_long_term" if "/access/" in key else "aws_operational"
        if (code == "n" and not ((1 <= lead <= 6) or (lead == 0 and source_id == "ncei_long_term"))) or (code == "f" and lead < 1):
            return None
        guidance = "nowcast" if code == "n" else "forecast"
        valid = run + timedelta(hours=(lead - 6 if code == "n" else lead))
        expected_start = valid
        expected_end = valid + timedelta(hours=1)
        aggregate = False
    else:
        if product != "stations":
            return None
        guidance = values["aggregate"].lower()
        lead = None
        valid = None
        aggregate = True
        if guidance == "nowcast":
            expected_start, expected_end = run - timedelta(hours=6), run + timedelta(minutes=6)
        else:
            expected_start, expected_end = run, run + timedelta(hours=48, minutes=6)
    return {
        "key": key, "name": name, "model": config.model, "layout": _layout(key),
        "naming": naming, "product": product, "guidance": guidance,
        "aggregate": aggregate, "run_time": iso(run), "cycle_hour": run.hour,
        "lead": lead, "valid_time": iso(valid),
        "expected_start_utc": iso(expected_start),
        "expected_end_utc_exclusive": iso(expected_end),
        "semantic_id": f"{config.model}:{product}:{guidance}:{iso(run)}:{iso(valid) or 'aggregate'}",
    }


def _xml_text(node: ET.Element, name: str) -> str | None:
    for child in node:
        if child.tag.rsplit("}", 1)[-1] == name:
            return child.text
    return None


def list_s3_objects(prefix: str, *, session: Any | None = None,
                    endpoint: str = S3_ENDPOINT, timeout: float = 60.0) -> list[dict[str, Any]]:
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
            key, size = _xml_text(node, "Key"), _xml_text(node, "Size")
            if not key or size is None:
                continue
            objects.append({
                "key": key, "size": int(size),
                "etag": (_xml_text(node, "ETag") or "").strip('"'),
                "last_modified": _xml_text(node, "LastModified"),
                "storage_class": _xml_text(node, "StorageClass"),
                "url": endpoint + "/" + quote(key, safe="/"),
            })
        truncated = next((node.text for node in root.iter()
                          if node.tag.rsplit("}", 1)[-1] == "IsTruncated"), "false")
        if str(truncated).lower() != "true":
            break
        token = next((node.text for node in root.iter()
                      if node.tag.rsplit("}", 1)[-1] == "NextContinuationToken"), None)
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
        cursor = datetime(cursor.year + (cursor.month == 12),
                          1 if cursor.month == 12 else cursor.month + 1, 1, tzinfo=UTC)


def discovery_prefixes(config: ModelConfig, request: Mapping[str, Any]) -> list[str]:
    if request["guidance"] == "forecast":
        center = parse_utc(request["run_cycle_utc"])
        first, last = center - timedelta(days=1), center + timedelta(days=1)
    else:
        first = parse_utc(request["start_utc"]) - timedelta(days=1)
        last = parse_utc(request["end_utc_exclusive"]) + timedelta(days=1)
    daily = [f"{config.prefix}{day:%Y/%m/%d}/" for day in _days(first, last)]
    monthly = [f"{config.prefix}{month:%Y%m}/" for month in _months(first, last)]
    return daily + monthly


def source_discovery_prefixes(config: ModelConfig, request: Mapping[str, Any],
                              source_id: str) -> list[str]:
    if source_id == "aws_operational":
        return discovery_prefixes(config, request)
    descriptor = archive_sources.get_source_descriptor(source_id, config.model)
    if request["guidance"] == "forecast":
        center = parse_utc(request["run_cycle_utc"])
        first, last = center - timedelta(days=1), center + timedelta(days=1)
    else:
        first = parse_utc(request["start_utc"]) - timedelta(days=1)
        last = parse_utc(request["end_utc_exclusive"]) + timedelta(days=1)
    return [f"{descriptor['root_prefix']}{month:%Y/%m}/" for month in _months(first, last)]


def _discover_one_source(config: ModelConfig, request: Mapping[str, Any], source_id: str,
                         *, session: Any | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    prefixes = source_discovery_prefixes(config, request, source_id)
    for prefix in prefixes:
        for raw in archive_sources.list_objects_v2(
                source_id, config.model, prefix, session=session):
            parsed = parse_object_key(config, raw["key"])
            if parsed is None:
                continue
            item = _decorate_source(config, {**raw, **parsed}, source_id)
            if item["product"] == request["product"] and item["guidance"] == request["guidance"]:
                found[item["key"]] = item
    objects = sorted(found.values(), key=lambda item: (item["run_time"], item["key"]))
    return objects, {"source_id": source_id, "status": "success",
                     "prefixes": prefixes, "object_count": len(objects), "error": None}


def _coverage_gaps(request: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]) -> list[str]:
    relaxed = {**request, "missing_policy": "skip"}
    try:
        return list(select_objects(relaxed, objects)["missing_times"])
    except RuntimeError as exc:
        if "did not contain matching" not in str(exc):
            raise
        start, end = parse_utc(request["start_utc"]), parse_utc(request["end_utc_exclusive"])
        step = 360 if request["product"] == "stations" else 3600
        return [iso(stamp) for stamp in expected_times(start, end, step)]


def _scientific_fallback_times(request: Mapping[str, Any],
                               objects: Sequence[Mapping[str, Any]]) -> list[str]:
    """Find station boundaries covered only by the following cycle's first record."""
    if request["product"] != "stations" or request["guidance"] != "nowcast":
        return []
    start, end = parse_utc(request["start_utc"]), parse_utc(request["end_utc_exclusive"])
    candidates = [item for item in objects
                  if item.get("product") == "stations" and item.get("guidance") == "nowcast"]
    unresolved: list[str] = []
    for stamp in expected_times(start, end, 360):
        covering = [item for item in candidates if stamp in _nominal_times(item)]
        if (covering
                and any(parse_utc(item["expected_start_utc"]) == stamp for item in covering)
                and not any(parse_utc(item["run_time"]) == stamp for item in covering)):
            unresolved.append(iso(stamp))
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


def discover_objects_with_evidence(config: ModelConfig, request: Mapping[str, Any], *,
                                   session: Any | None = None,
                                   endpoint: str = S3_ENDPOINT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del endpoint  # endpoints are fixed by the shared source descriptors
    policy = request["source_policy"]
    attempts: list[dict[str, Any]] = []
    if policy == "ncei_only":
        capable, reason = _ncei_capability(request)
        if not capable:
            raise ValueError(reason)
        objects, evidence = _discover_one_source(config, request, "ncei_long_term", session=session)
        attempts.append(evidence)
        before = _coverage_gaps(request, [])
        after = _coverage_gaps(request, objects)
        return objects, {"source_attempts": attempts, "fallback_triggered": False,
                         "fallback_reason": None, "coverage_before_fallback": before,
                         "coverage_after_fallback": after, "ncei_filled_times": sorted(set(before) - set(after)),
                         "unresolved_times": after}
    aws, aws_evidence = _discover_one_source(config, request, "aws_operational", session=session)
    attempts.append(aws_evidence)
    before = _coverage_gaps(request, aws)
    scientific_before = _scientific_fallback_times(request, aws)
    combined = list(aws)
    triggered = False
    reason = None
    capable, capability_reason = _ncei_capability(request)
    if policy == "aws_then_ncei" and (before or scientific_before) and capable:
        ncei, ncei_evidence = _discover_one_source(config, request, "ncei_long_term", session=session)
        attempts.append(ncei_evidence)
        combined.extend(ncei)
        triggered = True
        reason = ("aws_scientific_precedence_unresolved" if scientific_before and not before
                  else "aws_semantic_coverage_unresolved")
    elif policy == "aws_then_ncei" and (before or scientific_before):
        attempts.append({"source_id": "ncei_long_term", "status": "not_capable",
                         "prefixes": [], "object_count": 0, "error": capability_reason})
        reason = capability_reason
    after = _coverage_gaps(request, combined)
    return combined, {"source_attempts": attempts, "fallback_triggered": triggered,
                      "fallback_reason": reason, "coverage_before_fallback": before,
                      "scientific_precedence_before_fallback": scientific_before,
                      "coverage_after_fallback": after,
                      "ncei_filled_times": sorted(set(before) - set(after)),
                      "unresolved_times": after}


def discover_objects(config: ModelConfig, request: Mapping[str, Any], *,
                     session: Any | None = None, endpoint: str = S3_ENDPOINT) -> list[dict[str, Any]]:
    return discover_objects_with_evidence(
        config, request, session=session, endpoint=endpoint)[0]


def expected_times(start: datetime, end: datetime, seconds: int) -> list[datetime]:
    first = math.ceil(start.timestamp() / seconds) * seconds
    cursor = datetime.fromtimestamp(first, tz=UTC)
    result: list[datetime] = []
    while cursor < end:
        result.append(cursor)
        cursor += timedelta(seconds=seconds)
    return result


def _preference(item: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (1 if item.get("lead") != 0 else 0,
            1 if item.get("source_id", "aws_operational") == "aws_operational" else 0,
            1 if item.get("naming") == "current" else 0,
            1 if item.get("layout") == "daily" else 0)


def _overlaps(item: Mapping[str, Any], start: datetime, end: datetime) -> bool:
    return parse_utc(item["expected_start_utc"]) < end and parse_utc(item["expected_end_utc_exclusive"]) > start


def _nominal_times(item: Mapping[str, Any]) -> list[datetime]:
    step = 360 if item["product"] == "stations" else 3600
    return expected_times(parse_utc(item["expected_start_utc"]),
                          parse_utc(item["expected_end_utc_exclusive"]), step)


def select_objects(request: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    start, end = parse_utc(request["start_utc"]), parse_utc(request["end_utc_exclusive"])
    candidates = []
    for raw in objects:
        item = dict(raw)
        if item.get("product") != request["product"] or item.get("guidance") != request["guidance"]:
            continue
        if request["guidance"] == "forecast" and item.get("run_time") != request["run_cycle_utc"]:
            continue
        if _overlaps(item, start, end):
            candidates.append(item)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in candidates:
        identity = ((item["product"], item["guidance"], item["run_time"])
                    if item["aggregate"] else
                    (item["product"], item["guidance"], item["valid_time"]))
        groups.setdefault(identity, []).append(item)
    selected, rejected = [], []
    for identity, group in groups.items():
        best_rank = max(_preference(item) for item in group)
        winners = [item for item in group if _preference(item) == best_rank]
        if len(winners) > 1:
            identities = {(item.get("size"), _clean_etag(item.get("etag")))
                          for item in winners}
            if (len(identities) != 1 or next(iter(identities))[0] is None
                    or not next(iter(identities))[1]):
                details = ", ".join(str(item["key"]) for item in winners)
                raise RuntimeError(f"same-rank conflicting source objects for {identity}: {details}")
            winner = min(winners, key=lambda item: str(item["key"]))
            for item in winners:
                if item is not winner:
                    rejected.append({**item,
                                     "rejection_reason": "same_rank_identical_remote_metadata",
                                     "preferred_key": winner["key"]})
        else:
            winner = winners[0]
        selected.append(winner)
        rejected.extend({**item, "rejection_reason": "lower_archive_or_layout_preference",
                         "preferred_key": winner["key"]}
                        for item in group if item not in winners)
    selected.sort(key=lambda item: (item.get("valid_time") or item["run_time"], item["key"]))
    # Station cycle aggregates overlap at their boundary. Walk oldest to
    # newest so a preceding cycle's terminal record owns the timestamp; an
    # aggregate that contributes no new requested timestamp is unnecessary.
    if request["product"] == "stations":
        covered: set[datetime] = set()
        needed: list[dict[str, Any]] = []
        for item in selected:
            contribution = {
                stamp for stamp in _nominal_times(item) if start <= stamp < end
            }
            if contribution - covered:
                needed.append(item)
                covered.update(contribution)
            else:
                rejected.append({
                    **item,
                    "rejection_reason": "following_cycle_initial_duplicate",
                    "preferred_key": "preceding cycle terminal record",
                })
        selected = needed
    step = 360 if request["product"] == "stations" else 3600
    expected = expected_times(start, end, step)
    coverage: dict[datetime, list[str]] = {}
    for item in selected:
        for stamp in _nominal_times(item):
            if start <= stamp < end:
                coverage.setdefault(stamp, []).append(item["key"])
    missing = [iso(stamp) for stamp in expected if stamp not in coverage]
    duplicate_times = [{"time_utc": iso(stamp), "keys": keys,
                        "preferred_key": keys[0]}
                       for stamp, keys in sorted(coverage.items()) if len(keys) > 1]
    if missing and request["missing_policy"] == "error":
        raise RuntimeError(f"source inventory is missing {len(missing)} required timestamps: {', '.join(missing[:8])}")
    if not selected:
        raise RuntimeError("source inventory did not contain matching objects")
    return {
        "selected": selected, "duplicate_objects": rejected,
        "missing_times": missing, "duplicate_times": duplicate_times,
        "nominal_time_count": len(expected),
        "coverage_note": "filename-derived coverage is nominal; downloaded NetCDF time coordinates are authoritative",
    }


def inventory_request(config: ModelConfig, request: Mapping[str, Any] | str | Path,
                      run_dir: str | Path, *, output: str | Path | None = None,
                      objects: Sequence[Mapping[str, Any]] | None = None,
                      session: Any | None = None, endpoint: str = S3_ENDPOINT) -> dict[str, Any]:
    migration = request_migration(config, request)
    normalized = load_request(config, request) if isinstance(request, (str, Path)) else validate_request(config, request)
    if objects is None:
        raw_objects, discovery = discover_objects_with_evidence(
            config, normalized, session=session, endpoint=endpoint)
    else:
        raw_objects = list(objects)
        gaps = _coverage_gaps(normalized, raw_objects)
        discovery = {"source_attempts": [{"source_id": "provided", "status": "provided",
                                           "prefixes": [], "object_count": len(raw_objects), "error": None}],
                     "fallback_triggered": False, "fallback_reason": None,
                     "coverage_before_fallback": gaps, "coverage_after_fallback": gaps,
                     "ncei_filled_times": [], "unresolved_times": gaps}
    discovered = [_decorate_source(config, item, str(item.get("source_id") or "aws_operational"))
                  for item in raw_objects]
    capable, reason = _ncei_capability(normalized)
    report = {
        "schema_version": f"{config.model}_inventory_v2", "created_utc": iso(datetime.now(UTC)),
        "request": normalized, "prefixes": discovery_prefixes(config, normalized),
        **migration,
        "source_policy": normalized["source_policy"],
        **discovery,
        "object_count": len(discovered), "objects": discovered,
        "source": {"policy": normalized["source_policy"], "access": "anonymous_https_listobjectsv2"},
    }
    write_json_atomic(output or Path(run_dir) / "inventory.json", report)
    return report


def plan_request(config: ModelConfig, request: Mapping[str, Any] | str | Path,
                 run_dir: str | Path, *, output: str | Path | None = None,
                 objects: Sequence[Mapping[str, Any]] | None = None,
                 session: Any | None = None, endpoint: str = S3_ENDPOINT) -> dict[str, Any]:
    migration = request_migration(config, request)
    normalized = load_request(config, request) if isinstance(request, (str, Path)) else validate_request(config, request)
    capable, reason = _ncei_capability(normalized)
    if normalized["source_policy"] == "ncei_only" and not capable:
        raise ValueError(reason)
    if objects is None:
        raw_objects, discovery = discover_objects_with_evidence(
            config, normalized, session=session, endpoint=endpoint)
    else:
        raw_objects = list(objects)
        gaps = _coverage_gaps(normalized, raw_objects)
        discovery = {"source_attempts": [{"source_id": "provided", "status": "provided",
                                           "prefixes": [], "object_count": len(raw_objects), "error": None}],
                     "fallback_triggered": False, "fallback_reason": None,
                     "coverage_before_fallback": gaps, "coverage_after_fallback": gaps,
                     "ncei_filled_times": [], "unresolved_times": gaps}
    discovered = [_decorate_source(config, item, str(item.get("source_id") or "aws_operational"))
                  for item in raw_objects]
    selection = select_objects(normalized, discovered)
    invalid_source_metadata: list[dict[str, str]] = []
    for item in selection["selected"]:
        try:
            validate_source_object(config, item, require_remote_metadata=True)
        except ValueError as exc:
            # Missing remote identity cannot authorize a local transfer. A
            # malformed or off-scope key/URL is more serious and is rejected.
            message = str(exc)
            if any(token in message for token in ("missing", "source size must")):
                invalid_source_metadata.append({"key": str(item.get("key")),
                                                "reason": message})
            else:
                raise
    def exact_positive_size(item: Mapping[str, Any]) -> int | None:
        size = item.get("size")
        return size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else None

    sizes = [exact_positive_size(item) for item in selection["selected"]]
    incomplete = [item["key"] for item, size in zip(selection["selected"], sizes) if size is None]
    known_bytes = sum(size for size in sizes if size is not None)
    total = None if incomplete else known_bytes
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(run_path).free)
    required = None if total is None else total * 4
    if incomplete or invalid_source_metadata:
        route, reason = "review", "estimate_incomplete"
    elif free > required:
        route, reason = "local", "local_free_bytes_exceeds_four_times_exact_request_bytes"
    else:
        route, reason = "kestrel", "local_free_space_gate_failed"
    report = {
        "schema_version": f"{config.model}_download_estimate_v2",
        "created_utc": iso(datetime.now(UTC)), "request": normalized,
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
        "objects": selection["selected"], "object_count": len(selection["selected"]),
        "total_bytes": total, "total_gib": None if total is None else total / 1024**3,
        "known_bytes": known_bytes,
        "incomplete_size_keys": incomplete, "missing_times": selection["missing_times"],
        "incomplete_source_metadata": invalid_source_metadata,
        "duplicate_times": selection["duplicate_times"], "duplicate_objects": selection["duplicate_objects"],
        "nominal_time_count": selection["nominal_time_count"], "coverage_note": selection["coverage_note"],
        "local_free_bytes": free, "required_free_bytes": required,
        "routing_decision": route, "routing_reason": reason,
        "kestrel_stage_hint": f"/scratch/yhuang168/oma_external_data_connectors/{config.model}-fetcher/<run-id>",
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
    netCDF4, _ = _netcdf_modules()
    try:
        # Some Windows netCDF4/HDF5 builds are not safe when independent
        # Dataset handles are opened and closed from fetch-worker threads.
        # Serialize only this metadata-open probe; signatures, hashing, and
        # network transfers remain concurrent.
        with _NETCDF_METADATA_LOCK:
            with netCDF4.Dataset(source) as dataset:
                # Force the library to traverse the root metadata, not just
                # open a file descriptor with plausible magic bytes.
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
        payload = json.dumps({"pid": os.getpid(), "created_utc": iso(datetime.now(UTC)),
                              "destination": str(destination.resolve()),
                              "owner_token": owner_token}) + "\n"
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


def destination_for_key(config: ModelConfig, run_dir: str | Path, key: str,
                        item: Mapping[str, Any] | None = None) -> Path:
    value = dict(item or {"key": key, "source_id": "aws_operational"})
    if item is not None:
        relative = archive_sources.cache_relpath(value)
    else:
        relative = key[len(config.prefix):] if key.startswith(config.prefix) else Path(key).name
    parts = Path(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts) or Path(relative).is_absolute():
        raise ValueError(f"unsafe S3 key path: {key!r}")
    destination = Path(run_dir) / "cache" / "raw"
    for part in parts:
        destination /= part
    return destination


def legacy_aws_cache_result(config: ModelConfig, item: Mapping[str, Any],
                            run_dir: str | Path) -> dict[str, Any] | None:
    """Reuse a fully verified pre-v2 AWS cache in place, without rewriting it."""
    if item.get("source_id") != "aws_operational" or not str(item.get("key", "")).startswith(config.prefix):
        return None
    relative = Path(str(item["key"])[len(config.prefix):])
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    destination = Path(run_dir) / "cache" / "raw" / relative
    sidecar_path = destination.with_name(destination.name + ".download.json")
    if not destination.is_file() or not sidecar_path.is_file():
        return None
    try:
        metadata = read_json(sidecar_path)
    except Exception:
        return None
    expected = {
        "schema_version": "roms_cached_object_v1", "model": config.model,
        "key": item.get("key"), "url": item.get("url"), "size": item.get("size"),
        "etag": _clean_etag(item.get("etag")),
        "last_modified": item.get("last_modified"),
        "etag_semantics": "opaque_provenance",
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


def _cache_result(item: Mapping[str, Any], destination: Path) -> dict[str, Any] | None:
    sidecar = destination.with_name(destination.name + ".download.json")
    if not destination.is_file() or not sidecar.is_file():
        return None
    try:
        metadata = read_json(sidecar)
    except Exception:
        return None
    expected_size, expected_etag = int(item.get("size", -1)), _clean_etag(item.get("etag"))
    if destination.stat().st_size != expected_size or int(metadata.get("size", -2)) != expected_size:
        return None
    if (metadata.get("schema_version") != "roms_cached_object_v1"
            or metadata.get("model") != item.get("model")
            or metadata.get("source_id") != item.get("source_id")
            or metadata.get("source_identity") != item.get("source_identity")
            or metadata.get("key") != item.get("key")
            or metadata.get("url") != item.get("url")
            or metadata.get("last_modified") != item.get("last_modified")
            or metadata.get("etag_semantics") != "opaque_provenance"):
        return None
    if expected_etag and _clean_etag(metadata.get("etag")) != expected_etag:
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


def _download_object_locked(config: ModelConfig, item: Mapping[str, Any], destination: str | Path,
                            *, session: Any | None = None, timeout: float = 120.0,
                            max_attempts: int = 4,
                            chunk_size: int = 4 * 1024 * 1024) -> dict[str, Any]:
    requests = _requests_module()
    raw_size = item.get("size")
    if not isinstance(raw_size, int) or isinstance(raw_size, bool) or raw_size <= 0:
        raise RuntimeError(f"object has no exact positive size: {item.get('key')}")
    expected_size, expected_etag = raw_size, _clean_etag(item.get("etag"))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = _cache_result(item, destination)
    if cached:
        return cached
    partial = destination.with_name(destination.name + ".part")
    part_sidecar = destination.with_name(destination.name + ".part.json")
    part_meta = {
        "schema_version": "roms_partial_object_v1",
        "source_id": item.get("source_id"),
        "source_identity": item.get("source_identity"),
        "key": item["key"], "url": item["url"], "size": expected_size,
        "etag": expected_etag, "last_modified": item.get("last_modified"),
    }
    if partial.exists():
        try:
            prior = read_json(part_sidecar)
        except Exception:
            prior = None
        if not isinstance(prior, Mapping) or any(
                prior.get(k) != part_meta[k] for k in part_meta):
            partial.unlink(missing_ok=True)
            part_sidecar.unlink(missing_ok=True)
    if not partial.exists():
        write_json_atomic(part_sidecar, part_meta)
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
        try:
            response = None
            if current == expected_size:
                # A complete, provenance-matched .part is promotable without
                # another network request.
                pass
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
                remote_etag = _clean_etag(response.headers.get("ETag"))
                if not remote_etag:
                    raise RuntimeError("transfer response has no ETag")
                if remote_etag != expected_etag:
                    raise RuntimeError(f"ETag changed during transfer: {remote_etag} != {expected_etag}")
                content_length = response.headers.get("Content-Length")
                expected_response = expected_size - current if mode == "ab" else expected_size
                if content_length is not None and int(content_length) != expected_response:
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
                # On Windows a live requests response can retain handles long
                # enough to interfere with the atomic promotion below.
                response.close()
                response = None
            if partial.stat().st_size != expected_size:
                raise RuntimeError(f"downloaded size mismatch: {partial.stat().st_size} != {expected_size}")
            digest = sha256_file(partial)
            try:
                _validate_object_payload(item, partial)
            except Exception:
                # A provenance-matched complete partial can still contain
                # corrupt bytes (for example after an interrupted concurrent
                # writer). Do not revalidate those same bytes on every retry:
                # discard them and make the next attempt a clean transfer.
                partial.unlink(missing_ok=True)
                part_sidecar.unlink(missing_ok=True)
                discarded_invalid_partial = True
                if attempt + 1 < max_attempts:
                    write_json_atomic(part_sidecar, part_meta)
                raise
            os.replace(partial, destination)
            part_sidecar.unlink(missing_ok=True)
            metadata = {
                "schema_version": "roms_cached_object_v1", "model": config.model,
                "source_id": item.get("source_id"),
                "source_identity": item.get("source_identity"),
                "key": item["key"], "url": item["url"], "size": expected_size,
                "etag": expected_etag, "etag_is_multipart": "-" in expected_etag,
                "etag_semantics": "opaque_provenance", "last_modified": item.get("last_modified"),
                "sha256": digest, "netcdf_openable": str(item.get("key", "")).lower().endswith(".nc"),
                "completed_utc": iso(datetime.now(UTC)),
            }
            write_json_atomic(destination.with_name(destination.name + ".download.json"), metadata)
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
                response.close()
    return {
        "key": item.get("key"), "url": item.get("url"),
        "local_path": str(destination.resolve()), "status": "failed",
        "size": expected_size, "etag": expected_etag, "resumed": last_resume_from > 0,
        "resumed_from_bytes": last_resume_from, "retry_count": max_attempts - 1,
        "discarded_invalid_partial": discarded_invalid_partial,
        "errors": errors, "source": dict(item),
    }


def download_object(config: ModelConfig, item: Mapping[str, Any], destination: str | Path,
                    *, session: Any | None = None, timeout: float = 120.0,
                    max_attempts: int = 4, chunk_size: int = 4 * 1024 * 1024) -> dict[str, Any]:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = _cache_result(item, destination)
    if cached:
        return cached
    with _destination_lock(destination):
        # The winning process may have completed between our initial cache
        # check and lock acquisition.
        cached = _cache_result(item, destination)
        if cached:
            return cached
        return _download_object_locked(
            config, item, destination, session=session, timeout=timeout,
            max_attempts=max_attempts, chunk_size=chunk_size,
        )


def fetch_from_plan(config: ModelConfig, plan: Mapping[str, Any] | str | Path,
                    run_dir: str | Path, *, session: Any | None = None) -> dict[str, Any]:
    if not isinstance(plan, (str, Path)):
        raise RuntimeError("fetch requires an existing reviewed plan file path, not an in-memory mapping")
    plan_path = Path(plan).resolve()
    if not plan_path.is_file():
        raise RuntimeError("fetch requires an existing reviewed plan file")
    estimate = read_json(plan_path)
    if estimate.get("schema_version") != f"{config.model}_download_estimate_v2":
        raise RuntimeError("fetch requires a connector-generated download estimate")
    request = validate_request(config, estimate["request"])
    if estimate.get("source_policy") != request["source_policy"]:
        raise RuntimeError("approved plan source_policy does not match its request")
    request_sha256 = canonical_json_sha256(request)
    if estimate.get("request_sha256") != request_sha256:
        raise RuntimeError("approved plan request_sha256 does not match its normalized request")
    if estimate.get("routing_decision") != "local":
        raise RuntimeError(f"local fetch not approved: {estimate.get('routing_decision')} ({estimate.get('routing_reason')})")
    objects = estimate.get("objects")
    if not isinstance(objects, list) or not objects:
        raise RuntimeError("approved plan contains no source objects")
    sizes = [item.get("size") for item in objects if isinstance(item, Mapping)]
    if (len(sizes) != len(objects) or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0
                                         for size in sizes)):
        raise RuntimeError("approved plan does not contain exact positive object sizes")
    total = sum(sizes)
    if (estimate.get("total_bytes") != total or estimate.get("object_count") != len(objects)
            or estimate.get("required_free_bytes") != total * 4
            or estimate.get("incomplete_size_keys") not in ([], ())
            or estimate.get("incomplete_source_metadata") not in (None, [], ())):
        raise RuntimeError("approved plan accounting does not match its source objects")
    expected_source_totals = {
        source_id: {
            "object_count": sum(item.get("source_id") == source_id for item in objects),
            "bytes": sum(item["size"] for item in objects if item.get("source_id") == source_id),
        }
        for source_id in ("aws_operational", "ncei_long_term")
    }
    if estimate.get("source_totals") != expected_source_totals:
        raise RuntimeError("approved plan per-source object/byte totals are inconsistent")
    for item in objects:
        try:
            validate_source_object(config, item, require_remote_metadata=True)
        except ValueError as exc:
            raise RuntimeError(f"approved plan contains invalid source provenance: {exc}") from exc
    objects_sha256 = source_objects_sha256(objects)
    if estimate.get("objects_sha256") != objects_sha256:
        raise RuntimeError("approved plan objects_sha256 does not match its selected source objects")
    fallback_failures = validate_fallback_decision(request, estimate, objects)
    if fallback_failures:
        raise RuntimeError("approved plan fallback decision is invalid: " + "; ".join(fallback_failures))
    reselection = select_objects(request, objects)
    if [item["key"] for item in reselection["selected"]] != [item["key"] for item in objects]:
        raise RuntimeError("approved plan object selection is inconsistent with its request")
    plan_sha256 = sha256_file(plan_path)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    if int(shutil.disk_usage(run_path).free) <= total * 4:
        raise RuntimeError("current local free space no longer exceeds four times the approved source bytes")
    outcomes: list[dict[str, Any]] = []
    def transfer(item: Mapping[str, Any]) -> dict[str, Any]:
        destination = destination_for_key(config, run_path, item["key"], item)
        cached = _cache_result(item, destination)
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
        return download_object(config, item, destination, session=session)
    with concurrent.futures.ThreadPoolExecutor(max_workers=request["max_workers"]) as pool:
        futures = [pool.submit(transfer, item) for item in objects]
        for future in concurrent.futures.as_completed(futures):
            outcomes.append(future.result())
    outcomes.sort(key=lambda item: str(item.get("key")))
    failures = [item for item in outcomes if item["status"] == "failed"]
    manifest = {
        "schema_version": f"{config.model}_fetch_manifest_v2", "created_utc": iso(datetime.now(UTC)),
        "request": request,
        "source_policy": request["source_policy"],
        "approved_plan": {
            "path": str(plan_path),
            "sha256": plan_sha256,
            "schema_version": estimate["schema_version"],
            "request_sha256": request_sha256,
            "objects_sha256": objects_sha256,
            "object_count": len(objects),
            "total_bytes": total,
        },
        "outcomes": outcomes,
        "counts": {"objects": len(outcomes),
                   "downloaded": sum(item["status"] == "downloaded" for item in outcomes),
                   "cache_hits": sum(item["status"] == "cache_hit" for item in outcomes),
                   "failed": len(failures), "resumed": sum(bool(item.get("resumed")) for item in outcomes)},
        "source_provenance": {
            "policy": request["source_policy"],
            "source_counts": {source_id: sum(item.get("source", {}).get("source_id") == source_id for item in outcomes)
                              for source_id in ("aws_operational", "ncei_long_term")},
            "access": "anonymous_https",
        },
    }
    write_json_atomic(run_path / "fetch_manifest.json", manifest)
    if failures:
        raise RuntimeError(f"{len(failures)} {config.display_name} transfers failed; inspect fetch_manifest.json")
    return manifest


def verify_approved_plan_provenance(config: ModelConfig,
                                    manifest: Mapping[str, Any]) -> list[str]:
    """Verify that a fetch manifest remains bound to its reviewed plan."""
    failures: list[str] = []
    try:
        request = validate_request(config, manifest.get("request", {}))
    except Exception as exc:
        return [f"manifest request is invalid: {type(exc).__name__}: {exc}"]
    approved = manifest.get("approved_plan")
    if not isinstance(approved, Mapping):
        return ["fetch manifest has no approved-plan provenance"]
    expected_schema = f"{config.model}_download_estimate_v2"
    if approved.get("schema_version") != expected_schema:
        failures.append("approved plan schema_version is missing or inconsistent")
    if approved.get("request_sha256") != canonical_json_sha256(request):
        failures.append("approved plan request digest does not match the fetch request")
    outcomes = manifest.get("outcomes")
    outcome_sources = [item.get("source") for item in outcomes
                       if isinstance(item, Mapping)] if isinstance(outcomes, list) else []
    if (not outcome_sources or any(not isinstance(item, Mapping) for item in outcome_sources)
            or approved.get("objects_sha256") != source_objects_sha256(outcome_sources)):
        failures.append("approved plan objects do not match fetch outcome provenance")
    if approved.get("object_count") != len(outcome_sources):
        failures.append("approved plan object_count does not match fetch outcomes")
    outcome_bytes = sum(item.get("size", 0) for item in outcomes
                        if isinstance(item, Mapping) and isinstance(item.get("size"), int)
                        and not isinstance(item.get("size"), bool)) if isinstance(outcomes, list) else -1
    if approved.get("total_bytes") != outcome_bytes:
        failures.append("approved plan total_bytes does not match fetch outcomes")
    digest = approved.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        failures.append("approved plan SHA-256 is missing or invalid")
    plan_value: Mapping[str, Any] | None = None
    if approved.get("path"):
        plan_path = Path(str(approved["path"])).resolve()
        if not plan_path.is_file():
            failures.append("approved plan file is missing")
        elif sha256_file(plan_path) != digest:
            failures.append("approved plan SHA-256 mismatch")
        else:
            try:
                candidate = read_json(plan_path)
                if not isinstance(candidate, Mapping):
                    raise ValueError("plan root is not an object")
                plan_value = candidate
            except Exception as exc:
                failures.append(f"approved plan is unreadable: {type(exc).__name__}: {exc}")
    else:
        failures.append("approved plan file path is missing")
    if plan_value is not None:
        if plan_value.get("schema_version") != expected_schema:
            failures.append("approved plan file schema does not match the connector")
        try:
            plan_request = validate_request(config, plan_value.get("request", {}))
        except Exception as exc:
            failures.append(f"approved plan request is invalid: {type(exc).__name__}: {exc}")
        else:
            if canonical_json_sha256(plan_request) != canonical_json_sha256(request):
                failures.append("approved plan file request does not match the fetch request")
            if plan_value.get("request_sha256") != approved.get("request_sha256"):
                failures.append("approved plan file request_sha256 is missing or inconsistent")
        plan_objects = plan_value.get("objects")
        if (not isinstance(plan_objects, list)
                or any(not isinstance(item, Mapping) for item in plan_objects)):
            failures.append("approved plan file has invalid source objects")
        elif source_objects_sha256(plan_objects) != approved.get("objects_sha256"):
            failures.append("approved plan file objects do not match the fetch outcomes")
        else:
            if plan_value.get("objects_sha256") != approved.get("objects_sha256"):
                failures.append("approved plan file objects_sha256 is missing or inconsistent")
            if plan_value.get("object_count") != approved.get("object_count"):
                failures.append("approved plan file object_count is missing or inconsistent")
            if plan_value.get("total_bytes") != approved.get("total_bytes"):
                failures.append("approved plan file total_bytes is missing or inconsistent")
            expected_source_totals = {
                source_id: {
                    "object_count": sum(item.get("source_id") == source_id for item in plan_objects),
                    "bytes": sum(item.get("size", 0) for item in plan_objects
                                 if item.get("source_id") == source_id),
                }
                for source_id in ("aws_operational", "ncei_long_term")
            }
            if plan_value.get("source_totals") != expected_source_totals:
                failures.append("approved plan per-source object/byte totals are inconsistent")
            for item in plan_objects:
                try:
                    validate_source_object(config, item, require_remote_metadata=True)
                except Exception as exc:
                    failures.append(f"approved plan source object is invalid: {exc}")
            failures.extend(validate_fallback_decision(request, plan_value, plan_objects))
        source = plan_value.get("source")
        if (not isinstance(source, Mapping)
                or source.get("policy") != request.get("source_policy")):
            failures.append("approved plan source policy is missing or inconsistent")
    if isinstance(outcomes, list):
        seen: set[str] = set()
        for outcome in outcomes:
            if not isinstance(outcome, Mapping):
                failures.append("fetch outcome is not an object")
                continue
            source = outcome.get("source")
            if not isinstance(source, Mapping):
                failures.append("fetch outcome has no source provenance")
                continue
            try:
                validate_source_object(config, source, require_remote_metadata=True)
            except Exception as exc:
                failures.append(f"fetch outcome source is invalid: {exc}")
                continue
            key = str(source.get("key"))
            if key in seen:
                failures.append(f"fetch outcomes contain duplicate key: {key}")
            seen.add(key)
            if outcome.get("status") not in {"downloaded", "cache_hit"}:
                failures.append(f"fetch outcome is not successful: {key}")
            for name in ("key", "url", "size"):
                if outcome.get(name) != source.get(name):
                    failures.append(f"fetch outcome {name} does not match source: {key}")
            if _clean_etag(outcome.get("etag")) != _clean_etag(source.get("etag")):
                failures.append(f"fetch outcome ETag does not match source: {key}")
        counts = manifest.get("counts")
        if not isinstance(counts, Mapping) or counts.get("objects") != len(outcomes):
            failures.append("fetch-manifest counts do not match outcomes")
        elif counts.get("failed") != 0:
            failures.append("fetch manifest records failed outcomes")
    return failures


def verify_legacy_v1_manifest(config: ModelConfig, manifest: Mapping[str, Any],
                              manifest_path: str | Path,
                              expected_request: Mapping[str, Any]) -> dict[str, Any]:
    """Read and integrity-check immutable pre-v2 AWS evidence; never authorizes transfer."""
    failures: list[str] = []
    if manifest.get("schema_version") != f"{config.model}_fetch_manifest_v1":
        return {"status": "fail", "objects": [], "failures": ["not a legacy v1 fetch manifest"]}
    try:
        normalized = validate_request(config, manifest.get("request", {}))
    except Exception as exc:
        return {"status": "fail", "objects": [], "failures": [f"legacy request is invalid: {exc}"]}
    if normalized != expected_request:
        failures.append("legacy fetch-manifest request does not match health request")
    approved = manifest.get("approved_plan")
    if not isinstance(approved, Mapping) or approved.get("schema_version") != f"{config.model}_download_estimate_v1":
        failures.append("legacy approved plan schema is invalid")
        plan = None
    else:
        plan_path = Path(str(approved.get("path", ""))).resolve()
        if not plan_path.is_file() or sha256_file(plan_path) != approved.get("sha256"):
            failures.append("legacy approved plan path/SHA-256 is invalid")
            plan = None
        else:
            plan = read_json(plan_path)
            if plan.get("schema_version") != f"{config.model}_download_estimate_v1":
                failures.append("legacy approved plan file schema is invalid")
    outcomes = manifest.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        failures.append("legacy fetch manifest has no outcomes")
        outcomes = []
    root = Path(manifest_path).resolve().parent
    raw_root = (root / "cache" / "raw").resolve()
    objects: list[dict[str, Any]] = []
    for outcome in outcomes:
        source = outcome.get("source") if isinstance(outcome, Mapping) else None
        key = source.get("key") if isinstance(source, Mapping) else None
        parsed = parse_object_key(config, str(key or ""))
        path = Path(str(outcome.get("local_path", ""))).resolve() if isinstance(outcome, Mapping) else Path()
        finding = None
        try:
            path.relative_to(raw_root)
        except ValueError:
            finding = "legacy local path is outside raw cache"
        expected_url = _expected_object_url(str(key)) if key else None
        if parsed is None or source.get("url") != expected_url:
            finding = finding or "legacy AWS source key/URL is invalid"
        size, digest = outcome.get("size"), outcome.get("sha256")
        sidecar_path = path.with_name(path.name + ".download.json")
        if not path.is_file() or not sidecar_path.is_file():
            finding = finding or "legacy raw file/sidecar is missing"
        else:
            sidecar = read_json(sidecar_path)
            expected = {
                "schema_version": "roms_cached_object_v1", "model": config.model,
                "key": key, "url": expected_url, "size": size,
                "etag": _clean_etag(outcome.get("etag")),
                "last_modified": source.get("last_modified"), "sha256": digest,
                "etag_semantics": "opaque_provenance",
            }
            if any((_clean_etag(sidecar.get(name)) if name == "etag" else sidecar.get(name)) != value
                   for name, value in expected.items()):
                finding = finding or "legacy cache sidecar provenance mismatch"
            elif (path.stat().st_size != size or sha256_file(path) != digest):
                finding = finding or "legacy raw size/SHA-256 mismatch"
            else:
                try:
                    _validate_object_payload(source, path)
                except Exception as exc:
                    finding = finding or f"legacy raw NetCDF is invalid: {exc}"
        record = {"key": key, "local_path": str(path),
                  "status": "fail" if finding else "pass",
                  "size": size, "sha256": digest, "etag": outcome.get("etag"),
                  "source_archive": "aws_operational", "url": expected_url,
                  "legacy_evidence_schema": "v1"}
        if finding:
            record["reason"] = finding
            failures.append(f"{key}: {finding}")
        objects.append(record)
    return {"status": "pass" if not failures else "fail", "objects": objects,
            "failures": failures, "legacy_read_only": True,
            "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": sha256_file(manifest_path)}


def verify_cached_outcome(config: ModelConfig, outcome: Mapping[str, Any], *,
                          raw_root: str | Path) -> tuple[Path | None, list[str], dict[str, Any] | None]:
    """Verify file, sidecar, outcome, and source identity as one unit."""
    failures: list[str] = []
    source = outcome.get("source")
    if not isinstance(source, Mapping):
        return None, ["outcome has no source object"], None
    try:
        canonical = validate_source_object(config, source, require_remote_metadata=True)
    except Exception as exc:
        return None, [f"source object is invalid: {exc}"], None
    key = canonical["key"]
    for name in ("key", "url", "size"):
        if outcome.get(name) != source.get(name):
            failures.append(f"outcome {name} does not match source")
    if _clean_etag(outcome.get("etag")) != canonical["etag"]:
        failures.append("outcome ETag does not match source")
    digest = outcome.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        failures.append("outcome SHA-256 is missing or invalid")
    path = Path(str(outcome.get("local_path", ""))).resolve()
    root = Path(raw_root).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        failures.append("outcome path is outside the run raw cache")
    sidecar_path = path.with_name(path.name + ".download.json")
    sidecar: Mapping[str, Any] | None = None
    if not path.is_file():
        failures.append("raw object is missing")
    if not sidecar_path.is_file():
        failures.append("raw object download sidecar is missing")
    else:
        try:
            value = read_json(sidecar_path)
            if not isinstance(value, Mapping):
                raise ValueError("sidecar root is not an object")
            sidecar = value
        except Exception as exc:
            failures.append(f"raw object download sidecar is invalid: {exc}")
    if sidecar is not None:
        expected = ({
            "schema_version": "roms_cached_object_v1", "model": config.model,
            "key": key, "url": canonical["url"], "size": canonical["size"],
            "etag": canonical["etag"], "last_modified": canonical["last_modified"],
            "sha256": digest, "etag_semantics": "opaque_provenance",
        } if outcome.get("cache_location") == "legacy_aws_v1" else {
            "schema_version": "roms_cached_object_v1", "model": config.model,
            "source_id": canonical.get("source_id", source.get("source_id")),
            "source_identity": source.get("source_identity"),
            "key": key, "url": canonical["url"], "size": canonical["size"],
            "etag": canonical["etag"], "last_modified": canonical["last_modified"],
            "sha256": digest, "etag_semantics": "opaque_provenance",
        })
        if (outcome.get("cache_location") == "legacy_aws_v1"
                and source.get("source_id") != "aws_operational"):
            failures.append("legacy cache reuse is permitted only for AWS objects")
        for name, value in expected.items():
            actual = _clean_etag(sidecar.get(name)) if name == "etag" else sidecar.get(name)
            if actual != value:
                label = "ETag" if name == "etag" else ("SHA-256" if name == "sha256" else name)
                failures.append(f"download sidecar {label} mismatch")
    if path.is_file() and canonical["size"] is not None:
        if path.stat().st_size != canonical["size"]:
            failures.append("raw object size mismatch")
        elif isinstance(digest, str) and sha256_file(path) != digest:
            failures.append("raw object SHA-256 mismatch")
        else:
            try:
                _validate_object_payload(source, path)
            except Exception as exc:
                failures.append(f"raw object payload is invalid: {exc}")
    descriptor = archive_sources.get_source_descriptor(
        str(source.get("source_id")), config.model)
    record = None if failures else {
        "source_id": source.get("source_id"),
        "source_archive": source.get("source_id"),
        "archive_role": descriptor["archive_role"],
        "container": descriptor["container"],
        "endpoint": descriptor["endpoint"],
        "listing_endpoint": descriptor["listing_endpoint"],
        "source_identity": source.get("source_identity"),
        "key": key, "url": canonical["url"], "local_path": str(path),
        "size": canonical["size"], "etag": canonical["etag"],
        "last_modified": canonical["last_modified"], "sha256": digest,
        "sidecar_path": str(sidecar_path),
        "cache_location": outcome.get("cache_location", "source_isolated_v2"),
    }
    return (path if not failures else None), failures, record


def manifest_fetch_binding(config: ModelConfig, manifest: Mapping[str, Any],
                           manifest_path: str | Path) -> dict[str, Any]:
    """Build the exact extraction/health binding from verified outcomes."""
    normalized = validate_request(config, manifest.get("request", {}))
    approved = manifest.get("approved_plan")
    outcomes = manifest.get("outcomes")
    if not isinstance(approved, Mapping) or not isinstance(outcomes, list):
        raise RuntimeError("manifest cannot produce a fetch binding")
    objects = []
    for outcome in outcomes:
        source = outcome.get("source")
        if not isinstance(source, Mapping):
            raise RuntimeError("manifest outcome has no source object")
        canonical = validate_source_object(config, source, require_remote_metadata=True)
        descriptor = archive_sources.get_source_descriptor(
            str(source.get("source_id")), config.model)
        objects.append({
            "source_id": source.get("source_id"),
            "source_archive": source.get("source_id"),
            "archive_role": descriptor["archive_role"],
            "container": descriptor["container"],
            "endpoint": descriptor["endpoint"],
            "listing_endpoint": descriptor["listing_endpoint"],
            "source_identity": source.get("source_identity"),
            "key": canonical["key"], "url": canonical["url"],
            "local_path": str(Path(str(outcome.get("local_path", ""))).resolve()),
            "size": canonical["size"], "etag": canonical["etag"],
            "last_modified": canonical["last_modified"],
            "sha256": outcome.get("sha256"),
            "sidecar_path": str(Path(str(outcome.get("local_path", ""))).resolve().with_name(
                Path(str(outcome.get("local_path", ""))).name + ".download.json")),
            "cache_location": outcome.get("cache_location", "source_isolated_v2"),
        })
    objects.sort(key=lambda item: item["key"])
    path = Path(manifest_path).resolve()
    return {
        "schema_version": f"{config.model}_verified_fetch_binding_v2",
        "verified": True,
        "request_sha256": canonical_json_sha256(normalized),
        "fetch_manifest_path": str(path), "fetch_manifest_sha256": sha256_file(path),
        "approved_plan_path": approved.get("path"),
        "approved_plan_sha256": approved.get("sha256"),
        "objects": objects,
    }


def verified_manifest_inputs(config: ModelConfig, manifest_path: str | Path, *,
                             request: Mapping[str, Any] | str | Path | None = None,
                             run_dir: str | Path | None = None) -> tuple[list[Path], dict[str, Any]]:
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise RuntimeError(f"fetch manifest is absent: {path}")
    manifest = read_json(path)
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != f"{config.model}_fetch_manifest_v2":
        raise RuntimeError("fetch manifest schema is invalid")
    normalized = validate_request(config, manifest.get("request", {}))
    if request is not None:
        expected = load_request(config, request) if isinstance(request, (str, Path)) else validate_request(config, request)
        if normalized != expected:
            raise RuntimeError("fetch-manifest request does not match extraction request")
    provenance_failures = verify_approved_plan_provenance(config, manifest)
    if provenance_failures:
        raise RuntimeError("approved-plan provenance failed: " + "; ".join(provenance_failures))
    root = Path(run_dir).resolve() if run_dir is not None else path.parent
    raw_root = (root / "cache" / "raw").resolve()
    outcomes = manifest.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise RuntimeError("fetch manifest has no outcomes")
    paths: list[Path] = []
    objects: list[dict[str, Any]] = []
    failures: list[str] = []
    for outcome in outcomes:
        verified_path, findings, record = verify_cached_outcome(config, outcome, raw_root=raw_root)
        if findings:
            failures.extend(f"{outcome.get('key')}: {item}" for item in findings)
        elif verified_path is not None and record is not None:
            paths.append(verified_path)
            objects.append(record)
    if failures:
        raise RuntimeError("raw manifest verification failed: " + "; ".join(failures))
    binding = manifest_fetch_binding(config, manifest, path)
    if objects != binding["objects"]:
        raise RuntimeError("verified raw objects do not match manifest binding")
    return paths, binding


def fetch_request(config: ModelConfig, request: Mapping[str, Any] | str | Path,
                  run_dir: str | Path, *, objects: Sequence[Mapping[str, Any]] | None = None,
                  session: Any | None = None, endpoint: str = S3_ENDPOINT) -> dict[str, Any]:
    raise RuntimeError(
        "direct request-to-transfer is disabled; write and review a plan, then use fetch_from_plan"
    )


def as_utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return datetime(int(value.year), int(value.month), int(value.day), int(value.hour),
                    int(value.minute), int(value.second), int(getattr(value, "microsecond", 0)), tzinfo=UTC)


def normalize_time(value: datetime, cadence_seconds: int, tolerance_seconds: float = 60.0) -> tuple[datetime, float]:
    nearest = round(value.timestamp() / cadence_seconds) * cadence_seconds
    candidate = datetime.fromtimestamp(nearest, tz=UTC)
    offset = (candidate - value).total_seconds()
    return (candidate, offset) if abs(offset) <= tolerance_seconds else (value, 0.0)


CALENDAR_ALIASES = {
    # Historical NOAA/NCEI ROMS files use this non-CF spelling.
    "gregorian_proleptic": "proleptic_gregorian",
}


def calendar_metadata(variable: Any) -> dict[str, Any]:
    """Return lossless source metadata plus the decoder-safe calendar name."""
    source_calendar = str(getattr(variable, "calendar", "standard"))
    decoder_calendar = CALENDAR_ALIASES.get(source_calendar.strip().casefold(),
                                            source_calendar)
    return {
        "source_time_units": str(variable.units),
        "source_calendar": source_calendar,
        "decoder_calendar": decoder_calendar,
        "calendar_alias_applied": decoder_calendar != source_calendar,
    }


def decode_times(dataset: Any) -> list[dict[str, Any]]:
    netCDF4, _ = _netcdf_modules()
    name = next((candidate for candidate in ("ocean_time", "time") if candidate in dataset.variables), None)
    if name is None:
        raise RuntimeError("NetCDF has no ocean_time or time coordinate")
    variable = dataset.variables[name]
    if not hasattr(variable, "units"):
        raise RuntimeError(f"{name} has no CF units")
    metadata = calendar_metadata(variable)
    values = netCDF4.num2date(variable[:], variable.units,
                             calendar=metadata["decoder_calendar"],
                             only_use_cftime_datetimes=False,
                             only_use_python_datetimes=True)
    cadence = 360 if len(values) > 1 and abs((as_utc_datetime(values[1]) - as_utc_datetime(values[0])).total_seconds()) < 1000 else 3600
    records = []
    for index, raw in enumerate(values):
        original = as_utc_datetime(raw)
        normalized, adjustment = normalize_time(original, cadence)
        records.append({"index": index, "original_time_utc": iso(original),
                        "normalized_time_utc": iso(normalized), "adjustment_seconds": adjustment,
                        **metadata})
    return records


def _array_digest(value: Any) -> str:
    _, np = _netcdf_modules()
    array = np.ma.filled(value, np.nan)
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _coordinate_digest(variable: Any) -> str:
    """Hash stored coordinate values without applying valid_min/max masks."""
    _, np = _netcdf_modules()
    variable.set_auto_mask(False)
    try:
        array = np.asarray(variable[:], dtype=np.float64)
    finally:
        variable.set_auto_mask(True)
    for name in ("_FillValue", "missing_value"):
        if hasattr(variable, name):
            for marker in np.asarray(getattr(variable, name)).reshape(-1):
                array = np.where(array == marker, np.nan, array)
    return _array_digest(array)


def inspect_file(path: str | Path) -> dict[str, Any]:
    netCDF4, _ = _netcdf_modules()
    source = Path(path)
    with netCDF4.Dataset(source) as dataset:
        variables = {
            name: {"dimensions": list(var.dimensions), "shape": list(var.shape),
                   "dtype": str(var.dtype), "standard_name": getattr(var, "standard_name", None),
                   "units": getattr(var, "units", None)}
            for name, var in dataset.variables.items()
        }
        coordinate_hashes = {name: _coordinate_digest(dataset.variables[name])
                             for name in ("lon_rho", "lat_rho", "h", "angle", "mask_rho",
                                          "lon_u", "lat_u", "mask_u", "lon_v", "lat_v", "mask_v",
                                          "s_rho", "s_w", "Cs_r", "Cs_w")
                             if name in dataset.variables}
        times = decode_times(dataset)
        return {
            "schema_version": "roms_file_inspection_v1", "path": str(source.resolve()),
            "size": source.stat().st_size, "sha256": sha256_file(source),
            "dimensions": {name: len(dim) for name, dim in dataset.dimensions.items()},
            "variables": variables, "times": times, "coordinate_hashes": coordinate_hashes,
            "Vtransform": int(getattr(dataset, "Vtransform", dataset.variables["Vtransform"][:]
                                  if "Vtransform" in dataset.variables else -1)),
            "Vstretching": int(getattr(dataset, "Vstretching", dataset.variables["Vstretching"][:]
                                   if "Vstretching" in dataset.variables else -1)),
        }


def audit_time_records(config: ModelConfig, request: Mapping[str, Any],
                       paths: Sequence[str | Path]) -> dict[str, Any]:
    """Verify normalized cadence and resolve cycle-boundary duplicates."""
    netCDF4, _ = _netcdf_modules()
    normalized = validate_request(config, request)
    start, end = parse_utc(normalized["start_utc"]), parse_utc(normalized["end_utc_exclusive"])
    cadence = 360 if normalized["product"] == "stations" else 3600
    records: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        parsed = parse_object_key(config, f"{config.prefix}{path.name}")
        run_time = parse_utc(parsed["run_time"]) if parsed else datetime.max.replace(tzinfo=UTC)
        with netCDF4.Dataset(path) as dataset:
            for record in decode_times(dataset):
                stamp = parse_utc(record["normalized_time_utc"])
                if start <= stamp < end:
                    records.append({**record, "time": stamp, "run_time": run_time,
                                    "path": str(path.resolve())})
    groups: dict[datetime, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["time"], []).append(record)
    selected, duplicates = [], []
    for stamp, group in sorted(groups.items()):
        # At a station cycle boundary the preceding cycle has the earlier run.
        winner = min(group, key=lambda item: (item["run_time"], item["path"]))
        selected.append(winner)
        if len(group) > 1:
            duplicates.append({"time_utc": iso(stamp), "preferred": winner["path"],
                               "rejected": [item["path"] for item in group if item is not winner],
                               "rule": "preceding_cycle_terminal_record"})
    expected = expected_times(start, end, cadence)
    selected_times = [item["time"] for item in selected]
    missing = [iso(stamp) for stamp in expected if stamp not in set(selected_times)]
    adjustments = [float(item["adjustment_seconds"]) for item in selected]
    return {
        "cadence_seconds": cadence, "expected_count": len(expected),
        "selected_count": len(selected), "times_utc": [iso(stamp) for stamp in selected_times],
        "missing_times": missing, "duplicate_records": duplicates,
        "unique": len(selected_times) == len(set(selected_times)),
        "strictly_monotonic": all(a < b for a, b in zip(selected_times, selected_times[1:])),
        "max_absolute_normalization_seconds": max((abs(value) for value in adjustments), default=0.0),
    }


def manifest_paths(config: ModelConfig, run_dir: str | Path, *,
                   request: Mapping[str, Any] | str | Path | None = None,
                   manifest_path: str | Path | None = None) -> list[Path]:
    """Return only files verified through plan, manifest, sidecar, and bytes."""
    path = Path(manifest_path) if manifest_path is not None else Path(run_dir) / "fetch_manifest.json"
    paths, _ = verified_manifest_inputs(config, path, request=request, run_dir=run_dir)
    return paths
