#!/usr/bin/env python3
"""Provider-neutral archive metadata helpers for NOAA OFS connectors.

The module is intentionally dependency-light and copied byte-for-byte into each
self-contained OFS skill.  Model-specific filename parsing and scientific
processing remain in the connector that imports this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping
from urllib.parse import quote
import xml.etree.ElementTree as ET


AWS_ENDPOINT = "https://noaa-nos-ofs-pds.s3.amazonaws.com"
NCEI_ENDPOINT = "https://www.ncei.noaa.gov/oa/prod-model"
NCEI_BASE = (
    "operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/"
    "access/"
)
MODEL_SLUGS = {
    "cbofs": "chesapeake-bay-operational-forecast-system-cbofs",
    "dbofs": "delaware-bay-operational-forecast-system-dbofs",
    "nyofs": "port-of-new-york-and-new-jersey-operational-forecast-system-nyofs",
    "sscofs": "salish-sea-and-columbia-river-operational-forecast-system-sscofs",
    "sjrofs": "st-johns-river-operational-forecast-system-sjrofs",
}
SOURCE_IDS = {"aws_operational", "ncei_long_term"}


def _requests_module():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("requests is required for anonymous NOAA archive access") from exc
    return requests


def _clean_etag(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _size(record: Mapping[str, Any]) -> int | None:
    value = record.get("size", record.get("size_bytes"))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_source_descriptor(source_id: str, model: str) -> dict[str, str]:
    """Return the immutable source contract for *source_id* and *model*."""

    source_id = str(source_id).lower()
    model = str(model).lower()
    if source_id not in SOURCE_IDS:
        raise ValueError(f"unsupported OFS source_id: {source_id!r}")
    if model not in MODEL_SLUGS:
        raise ValueError(f"unsupported OFS model: {model!r}")
    if source_id == "aws_operational":
        return {
            "source_id": source_id,
            "provider": "NOAA",
            "archive_role": "operational",
            "container": "noaa-nos-ofs-pds",
            "endpoint": AWS_ENDPOINT,
            "listing_endpoint": AWS_ENDPOINT + "/",
            "root_prefix": f"{model}/netcdf/",
            "access": "anonymous_https_listobjectsv2",
        }
    return {
        "source_id": source_id,
        "provider": "NOAA NCEI",
        "archive_role": "long_term",
        "container": "prod-model",
        "endpoint": NCEI_ENDPOINT,
        "listing_endpoint": NCEI_ENDPOINT,
        "root_prefix": f"{NCEI_BASE}{MODEL_SLUGS[model]}/",
        "access": "anonymous_https_listobjectsv2",
    }


def canonical_object_url(source_id: str, model: str, key: str) -> str:
    """Build the one approved anonymous HTTPS URL for an archive key."""

    descriptor = get_source_descriptor(source_id, model)
    if not isinstance(key, str) or not key or key.startswith("/"):
        raise ValueError("archive key must be a nonempty relative path")
    path = PurePosixPath(key)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("archive key contains an unsafe path component")
    return descriptor["endpoint"] + "/" + quote(key, safe="/")


def _filename_date(name: str) -> str | None:
    matches = re.findall(r"(?<!\d)(20\d{6})(?!\d)", name)
    return matches[-1] if matches else None


def validate_source_object(
    model: str,
    record: Mapping[str, Any],
    *,
    expected_source_id: str | None = None,
    require_metadata: bool = True,
) -> dict[str, Any]:
    """Fail closed on model/source/path/URL and remote identity metadata."""

    if not isinstance(record, Mapping):
        raise ValueError("source object must be a mapping")
    model = str(model).lower()
    source_id = str(record.get("source_id") or expected_source_id or "")
    if expected_source_id is not None and source_id != expected_source_id:
        raise ValueError("source_id does not match the expected archive")
    descriptor = get_source_descriptor(source_id, model)
    key = record.get("key")
    if not isinstance(key, str) or not key.startswith(descriptor["root_prefix"]):
        raise ValueError(f"source key is outside the approved {model} archive prefix")
    relative = key[len(descriptor["root_prefix"]):]
    if not relative or relative.startswith("/") or any(
        part in {"", ".", ".."} for part in PurePosixPath(relative).parts
    ):
        raise ValueError("source key has an unsafe archive-relative path")
    name = PurePosixPath(relative).name
    if model not in name.lower() or not name.lower().endswith(".nc"):
        raise ValueError("source filename does not identify the requested OFS model NetCDF")
    date = _filename_date(name)
    if date is None:
        raise ValueError("source filename has no unambiguous YYYYMMDD run date")
    parts = PurePosixPath(relative).parts
    if source_id == "aws_operational":
        daily = len(parts) == 4 and re.fullmatch(r"\d{4}", parts[0]) and re.fullmatch(
            r"\d{2}", parts[1]
        ) and re.fullmatch(r"\d{2}", parts[2])
        monthly = len(parts) == 2 and re.fullmatch(r"\d{6}", parts[0])
        if not (daily or monthly):
            raise ValueError("AWS key must use the approved daily or legacy monthly layout")
        path_date = "".join(parts[:3]) if daily else str(parts[0])
        if not date.startswith(path_date):
            raise ValueError("AWS archive path does not match the filename run date")
    else:
        monthly = (
            len(parts) == 3
            and re.fullmatch(r"\d{4}", parts[0])
            and re.fullmatch(r"\d{2}", parts[1])
        )
        if not monthly or not date.startswith(str(parts[0]) + str(parts[1])):
            raise ValueError("NCEI key must use YYYY/MM matching the filename run date")
    expected_url = canonical_object_url(source_id, model, key)
    if record.get("url") != expected_url:
        label = "NOAA S3" if source_id == "aws_operational" else "NOAA NCEI"
        raise ValueError(f"source URL is not the exact {label} URL")
    for field in ("provider", "archive_role", "container", "endpoint", "listing_endpoint"):
        if record.get(field) != descriptor[field]:
            raise ValueError(f"source {field} does not match the approved archive")
    if require_metadata:
        if _size(record) is None:
            raise ValueError("source size must be an exact positive integer")
        if not _clean_etag(record.get("etag")):
            raise ValueError("source object is missing ETag provenance")
        if not str(record.get("last_modified") or "").strip():
            raise ValueError("source object is missing Last-Modified provenance")
    identity = record.get("source_identity")
    if not isinstance(identity, str) or identity != source_identity_digest(record):
        raise ValueError("source_identity does not match canonical source metadata")
    return descriptor


def source_identity_digest(record: Mapping[str, Any]) -> str:
    """Hash provider-local identity; cross-provider ETags are never compared."""

    return _canonical_sha256({
        "source_id": record.get("source_id"),
        "container": record.get("container"),
        "endpoint": record.get("endpoint"),
        "key": record.get("key"),
        "url": record.get("url"),
        "size": _size(record),
        "etag": _clean_etag(record.get("etag")),
        "last_modified": record.get("last_modified"),
    })


def semantic_identity_digest(identity: Mapping[str, Any] | Iterable[Any]) -> str:
    """Hash a connector-defined semantic identity deterministically."""

    return _canonical_sha256(identity)


def cache_relpath(record: Mapping[str, Any]) -> str:
    """Return a source-isolated, collision-resistant cache-relative path."""

    source_id = str(record.get("source_id") or "")
    if source_id not in SOURCE_IDS:
        raise ValueError("cache record has no supported source_id")
    key = str(record.get("key") or "")
    name = PurePosixPath(key).name
    if not name or name in {".", ".."}:
        raise ValueError("cache record has no safe basename")
    date = _filename_date(name)
    if date is None:
        raise ValueError("cache record filename has no YYYYMMDD date")
    short_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"{source_id}/{date[:4]}/{date[4:6]}/{short_hash}-{name}"


def _xml_text(node: ET.Element, name: str) -> str | None:
    child = node.find(f"{{*}}{name}")
    return None if child is None else child.text


def iter_list_objects_v2(
    source_id: str,
    model: str,
    prefix: str,
    *,
    session: Any | None = None,
    timeout: float = 60.0,
    max_keys: int = 1000,
) -> Iterable[dict[str, Any]]:
    """Yield every positive-size object from a bounded anonymous XML listing."""

    descriptor = get_source_descriptor(source_id, model)
    if not isinstance(prefix, str) or not prefix.startswith(descriptor["root_prefix"]):
        raise ValueError("listing prefix is outside the approved model archive root")
    own_session = session is None
    client = session or _requests_module().Session()
    token: str | None = None
    try:
        while True:
            params: dict[str, Any] = {
                "list-type": "2", "prefix": prefix, "max-keys": int(max_keys),
            }
            if token:
                params["continuation-token"] = token
            response = client.get(descriptor["listing_endpoint"], params=params, timeout=timeout)
            try:
                response.raise_for_status()
                content = getattr(response, "content", None)
                if content is None:
                    content = str(getattr(response, "text", "")).encode("utf-8")
                root = ET.fromstring(content)
                for item in root.findall("{*}Contents"):
                    key = _xml_text(item, "Key")
                    size_text = _xml_text(item, "Size")
                    if not key or not size_text:
                        continue
                    try:
                        size = int(size_text)
                    except ValueError:
                        continue
                    if size <= 0:
                        continue
                    value: dict[str, Any] = {
                        **descriptor,
                        "key": key,
                        "url": canonical_object_url(source_id, model, key),
                        "size": size,
                        "size_bytes": size,
                        "etag": _clean_etag(_xml_text(item, "ETag")),
                        "last_modified": _xml_text(item, "LastModified"),
                    }
                    value["source_identity"] = source_identity_digest(value)
                    yield value
                truncated = str(_xml_text(root, "IsTruncated") or "false").lower() == "true"
                if not truncated:
                    break
                token = _xml_text(root, "NextContinuationToken")
                if not token:
                    raise RuntimeError("archive listing was truncated without a continuation token")
            finally:
                if hasattr(response, "close"):
                    response.close()
    finally:
        if own_session and hasattr(client, "close"):
            client.close()


def list_objects_v2(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return list(iter_list_objects_v2(*args, **kwargs))


_CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)


def parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    match = _CONTENT_RANGE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"invalid Content-Range: {value!r}")
    start, end, total = (int(part) for part in match.groups())
    if start < 0 or end < start or total <= end:
        raise ValueError(f"inconsistent Content-Range: {value!r}")
    return start, end, total


def build_resume_headers(offset: int) -> dict[str, str]:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("resume offset must be a nonnegative integer")
    return {} if offset == 0 else {"Range": f"bytes={offset}-"}


def validate_download_response(
    response: Any,
    record: Mapping[str, Any],
    *,
    offset: int = 0,
) -> dict[str, Any]:
    """Validate status, ETag, and byte-range metadata before streaming."""

    size = _size(record)
    if size is None:
        raise ValueError("planned object has no exact positive size")
    expected_etag = _clean_etag(record.get("etag"))
    if not expected_etag:
        raise ValueError("planned object has no ETag")
    status = int(getattr(response, "status_code", 0))
    expected_status = 206 if offset else 200
    if status != expected_status:
        raise RuntimeError(f"unexpected HTTP status {status}; expected {expected_status}")
    headers = getattr(response, "headers", {})
    remote_etag = _clean_etag(headers.get("ETag"))
    if not remote_etag:
        raise RuntimeError("transfer response has no ETag")
    if remote_etag != expected_etag:
        raise RuntimeError(
            f"ETag changed during transfer: {remote_etag} != {expected_etag}"
        )
    remaining = size - offset
    length = headers.get("Content-Length")
    if length is not None and int(length) != remaining:
        raise RuntimeError(f"Content-Length mismatch: {length} != {remaining}")
    parsed_range = parse_content_range(headers.get("Content-Range"))
    if offset:
        expected_range = (offset, size - 1, size)
        if parsed_range != expected_range:
            raise RuntimeError(f"Content-Range mismatch: {parsed_range} != {expected_range}")
    elif parsed_range is not None and parsed_range != (0, size - 1, size):
        raise RuntimeError("unexpected full-transfer Content-Range")
    return {"status_code": status, "etag": remote_etag, "remaining_bytes": remaining}


def validate_remote_metadata(
    record: Mapping[str, Any], remote: Mapping[str, Any]
) -> None:
    """Require HEAD/list metadata to repeat a planned provider-local identity."""

    planned_size = _size(record)
    remote_size = _size(remote)
    if planned_size is None or remote_size != planned_size:
        raise RuntimeError("remote Content-Length differs from the reviewed plan")
    if _clean_etag(remote.get("etag")) != _clean_etag(record.get("etag")):
        raise RuntimeError("remote ETag differs from the reviewed plan")
    if str(remote.get("last_modified") or "") != str(record.get("last_modified") or ""):
        raise RuntimeError("remote Last-Modified differs from the reviewed plan")


__all__ = [
    "AWS_ENDPOINT", "NCEI_ENDPOINT", "NCEI_BASE", "MODEL_SLUGS", "SOURCE_IDS",
    "get_source_descriptor", "iter_list_objects_v2", "list_objects_v2",
    "canonical_object_url", "validate_source_object", "source_identity_digest",
    "semantic_identity_digest", "cache_relpath", "parse_content_range",
    "build_resume_headers", "validate_download_response", "validate_remote_metadata",
]
