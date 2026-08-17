#!/usr/bin/env python3
"""Estimate-first acquisition and QA for native Argo GDAC profile files."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import requests

try:
    from .download_monitor import atomic_write_json, launch_monitor, safe_message, utc_now, write_status
except ImportError:
    from download_monitor import atomic_write_json, launch_monitor, safe_message, utc_now, write_status

REQUEST_SCHEMA = "argo_fetch_request_v1"
PLAN_SCHEMA = "argo_download_plan_v1"
DOI = "10.17882/42182"
PRIMARY_BASE = "https://data-argo.ifremer.fr"
S3_BASE = "https://argo-gdac-sandbox.s3.eu-west-3.amazonaws.com/pub"
PRODUCTS = {
    "core": "ar_index_global_prof.txt.gz",
    "synthetic": "argo_synthetic-profile_index.txt.gz",
    "bio": "argo_bio-profile_index.txt.gz",
}
USER_AGENT = "OMA-Argo-Fetcher/1.0 (+https://www.argodatamgt.org/DataAccess.html)"
PLAN_HOURS = 24
MAX_WORKERS = 16
DEFAULT_WORKERS = 8
ESTIMATE_SAMPLE = 64
EXACT_HEAD_LIMIT = 100


class ArgoError(RuntimeError):
    """Raised for an operator-actionable contract or acquisition failure."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, target)


def atomic_write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["file"]
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def parse_utc(value: str, field: str) -> datetime:
    text = str(value).strip()
    if re.fullmatch(r"\d{8}(\d{6})?", text):
        fmt = "%Y%m%d%H%M%S" if len(text) == 14 else "%Y%m%d"
        return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArgoError(f"{field} is not a valid UTC date/time: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def compact_argo_time(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))[:14]
    return digits.ljust(14, "0") if len(digits) >= 8 else ""


def normalize_lon(value: float) -> float:
    result = ((float(value) + 180.0) % 360.0) - 180.0
    return 180.0 if math.isclose(result, -180.0) and float(value) > 0 else result


def file_parts(relative_file: str) -> tuple[str, str, str, str]:
    parts = Path(relative_file.replace("\\", "/")).parts
    if len(parts) < 4 or parts[-2].lower() != "profiles":
        raise ArgoError(f"Unexpected GDAC profile path: {relative_file!r}")
    dac, wmo, name = parts[0].lower(), parts[1], parts[-1]
    mode_at = 1 if name[:1].upper() in {"B", "S"} else 0
    mode = name[mode_at : mode_at + 1].upper()
    if mode not in {"R", "D"}:
        raise ArgoError(f"Cannot determine R/D filename mode: {relative_file!r}")
    return dac, wmo, name, mode


def safe_relative_profile_path(relative_file: str) -> Path:
    clean = relative_file.replace("\\", "/").lstrip("/")
    path = Path(clean)
    if path.is_absolute() or ".." in path.parts:
        raise ArgoError(f"Unsafe GDAC path: {relative_file!r}")
    file_parts(clean)
    return Path("raw") / "dac" / path


def _selector_signature(request: Mapping[str, Any]) -> dict[str, Any]:
    if "geojson" in request:
        path = Path(str(request["geojson"]))
        return {"type": "geojson", "name": path.name, "sha256": sha256_file(path)}
    if "mesh_2dm" in request:
        path = Path(str(request["mesh_2dm"]))
        return {"type": "mesh_2dm", "name": path.name, "sha256": sha256_file(path)}
    if "bbox" in request:
        return {"type": "bbox", "value": [float(v) for v in request["bbox"]]}
    return {"type": "global"}


def validate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize an argo_fetch_request_v1 document."""
    if request.get("schema") != REQUEST_SCHEMA:
        raise ArgoError(f"request.schema must be {REQUEST_SCHEMA!r}")
    products = list(dict.fromkeys(str(v).lower() for v in request.get("products", [])))
    if not products or any(value not in PRODUCTS for value in products):
        raise ArgoError(f"products must contain one or more of {sorted(PRODUCTS)}")

    all_time = request.get("all_time") is True
    if all_time:
        if "start" in request or "end" in request:
            raise ArgoError("Use either explicit start/end or all_time=true, not both")
        start, end = datetime.min.replace(tzinfo=timezone.utc), datetime.max.replace(tzinfo=timezone.utc)
    else:
        if "start" not in request or "end" not in request:
            raise ArgoError("Inclusive start and end are required unless all_time=true")
        start, end = parse_utc(str(request["start"]), "start"), parse_utc(str(request["end"]), "end")
        if end < start:
            raise ArgoError("end must be greater than or equal to start")

    selectors = [key for key in ("bbox", "geojson", "mesh_2dm", "global") if key in request]
    if len(selectors) != 1 or (selectors == ["global"] and request.get("global") is not True):
        raise ArgoError("Specify exactly one of bbox, geojson, mesh_2dm, or global=true")
    if "bbox" in request:
        bbox = request["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ArgoError("bbox must be [west, south, east, north]")
        west, south, east, north = map(float, bbox)
        if not (-90 <= south < north <= 90) or not all(-180 <= v <= 180 for v in (west, east)):
            raise ArgoError("bbox longitude must be within [-180,180] and south < north within [-90,90]")
    if "geojson" in request and not Path(str(request["geojson"])).is_file():
        raise ArgoError("geojson selector does not exist")
    if "mesh_2dm" in request and not Path(str(request["mesh_2dm"])).is_file():
        raise ArgoError("mesh_2dm selector does not exist")

    file_modes = list(dict.fromkeys(str(v).upper() for v in request.get("file_modes", [])))
    if any(value not in {"R", "D"} for value in file_modes):
        raise ArgoError("file_modes may contain only R and D; adjusted A mode is internal")
    parameters = list(dict.fromkeys(str(v).upper() for v in request.get("parameters", [])))
    if any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", value) for value in parameters):
        raise ArgoError("parameters must contain uppercase Argo parameter names")
    parameter_match = str(request.get("parameter_match", "all")).lower()
    if parameter_match not in {"all", "any"}:
        raise ArgoError("parameter_match must be all or any")
    if parameters and not any(product in {"bio", "synthetic"} for product in products):
        raise ArgoError("BGC parameter filters require bio or synthetic products")

    wmos = list(dict.fromkeys(str(v).strip() for v in request.get("wmos", [])))
    if any(not re.fullmatch(r"\d+", value) for value in wmos):
        raise ArgoError("wmos must contain numeric platform identifiers")
    dacs = list(dict.fromkeys(str(v).lower() for v in request.get("dacs", [])))
    if any(not re.fullmatch(r"[a-z0-9_-]+", value) for value in dacs):
        raise ArgoError("dacs contain an invalid identifier")

    canonical: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "products": sorted(products),
        "all_time": all_time,
        "selector": _selector_signature(request),
        "wmos": sorted(wmos),
        "dacs": sorted(dacs),
        "file_modes": sorted(file_modes),
        "parameters": sorted(parameters),
        "parameter_match": parameter_match,
    }
    if not all_time:
        canonical["start"] = start.isoformat().replace("+00:00", "Z")
        canonical["end"] = end.isoformat().replace("+00:00", "Z")
    return canonical


def _geojson_geometry(path: str | Path):
    from shapely.geometry import shape
    from shapely.ops import unary_union

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = document.get("type")
    if kind == "Feature":
        geometry = shape(document["geometry"])
    elif kind == "FeatureCollection":
        geometry = unary_union([shape(item["geometry"]) for item in document.get("features", [])])
    else:
        geometry = shape(document)
    if geometry.geom_type not in {"Polygon", "MultiPolygon"} or geometry.is_empty or not geometry.is_valid:
        raise ArgoError("GeoJSON selector must contain valid nonempty Polygon/MultiPolygon geometry")
    return geometry


def _mesh_geometry(path: str | Path):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    nodes: dict[int, tuple[float, float]] = {}
    elements: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            code = fields[0].upper()
            if code == "ND" and len(fields) >= 4:
                nodes[int(fields[1])] = (float(fields[2]), float(fields[3]))
            elif code == "E3T" and len(fields) >= 5:
                elements.append([int(v) for v in fields[2:5]])
            elif code == "E4Q" and len(fields) >= 6:
                elements.append([int(v) for v in fields[2:6]])
    if not nodes or not elements:
        raise ArgoError("2DM selector requires ND nodes and E3T/E4Q wet elements")
    polygons = []
    for element in elements:
        try:
            polygon = Polygon([nodes[node] for node in element])
        except KeyError as exc:
            raise ArgoError(f"2DM element references missing node {exc.args[0]}") from exc
        if polygon.is_valid and polygon.area > 0:
            polygons.append(polygon)
    geometry = unary_union(polygons)
    if geometry.is_empty or not geometry.is_valid:
        raise ArgoError("2DM wet-element union is empty or invalid")
    return geometry


def spatial_mask(frame: pd.DataFrame, request: Mapping[str, Any]) -> np.ndarray:
    lon = np.asarray([normalize_lon(v) for v in frame["longitude"]], dtype=float)
    lat = frame["latitude"].to_numpy(dtype=float)
    if request.get("global") is True:
        return np.ones(len(frame), dtype=bool)
    if "bbox" in request:
        west, south, east, north = map(float, request["bbox"])
        lon_ok = (lon >= west) & (lon <= east) if west <= east else (lon >= west) | (lon <= east)
        return lon_ok & (lat >= south) & (lat <= north)

    from shapely.geometry import Point
    from shapely.prepared import prep

    geometry = _geojson_geometry(request["geojson"]) if "geojson" in request else _mesh_geometry(request["mesh_2dm"])
    minx, miny, maxx, maxy = geometry.bounds
    point_lon = lon.copy()
    if minx >= 0 and maxx > 180:
        point_lon = np.where(point_lon < 0, point_lon + 360.0, point_lon)
    candidates = (point_lon >= minx) & (point_lon <= maxx) & (lat >= miny) & (lat <= maxy)
    prepared = prep(geometry)
    result = np.zeros(len(frame), dtype=bool)
    for index in np.flatnonzero(candidates):
        result[index] = prepared.covers(Point(float(point_lon[index]), float(lat[index])))
    return result


def _index_update_timestamp(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            if "date of update" in line.lower():
                match = re.search(r"(\d{8,14})", line)
                if match:
                    return match.group(1)
    return ""


def load_index(path: str | Path, product: str) -> pd.DataFrame:
    target = Path(path)
    frame = pd.read_csv(
        target,
        compression="infer",
        comment="#",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    required = {"file", "date", "latitude", "longitude", "date_update"}
    missing = required.difference(frame.columns)
    if missing:
        raise ArgoError(f"{PRODUCTS[product]} is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["date_compact"] = frame["date"].map(compact_argo_time)
    frame["date_update"] = frame["date_update"].map(compact_argo_time)
    frame = frame[
        frame["file"].ne("")
        & frame["date_compact"].str.len().eq(14)
        & frame["latitude"].notna()
        & frame["longitude"].notna()
    ].copy()
    parsed = [file_parts(value) for value in frame["file"]]
    frame["dac"] = [value[0] for value in parsed]
    frame["wmo"] = [value[1] for value in parsed]
    frame["file_mode"] = [value[3] for value in parsed]
    frame["product"] = product
    if "parameters" not in frame:
        frame["parameters"] = ""
    if "parameter_data_mode" not in frame:
        frame["parameter_data_mode"] = ""
    return frame.reset_index(drop=True)


def select_profiles(frame: pd.DataFrame, request: Mapping[str, Any], product: str) -> pd.DataFrame:
    validate_request(request)
    if product not in request["products"]:
        return frame.iloc[0:0].copy()
    if request.get("all_time") is True:
        time_ok = np.ones(len(frame), dtype=bool)
    else:
        start = parse_utc(str(request["start"]), "start").strftime("%Y%m%d%H%M%S")
        end = parse_utc(str(request["end"]), "end").strftime("%Y%m%d%H%M%S")
        time_ok = frame["date_compact"].between(start, end, inclusive="both").to_numpy()
    mask = time_ok & spatial_mask(frame, request)
    if request.get("wmos"):
        mask &= frame["wmo"].isin({str(v) for v in request["wmos"]}).to_numpy()
    if request.get("dacs"):
        mask &= frame["dac"].isin({str(v).lower() for v in request["dacs"]}).to_numpy()
    if request.get("file_modes"):
        mask &= frame["file_mode"].isin({str(v).upper() for v in request["file_modes"]}).to_numpy()
    parameters = {str(v).upper() for v in request.get("parameters", [])}
    if parameters and product in {"bio", "synthetic"}:
        available = frame["parameters"].map(lambda value: {v.upper() for v in str(value).split()})
        if str(request.get("parameter_match", "all")).lower() == "all":
            mask &= available.map(parameters.issubset).to_numpy()
        else:
            mask &= available.map(lambda value: bool(value & parameters)).to_numpy()
    return frame.loc[mask].sort_values(["date_compact", "file"], kind="stable").reset_index(drop=True)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _remote_metadata(session: requests.Session, url: str, timeout: float) -> dict[str, Any]:
    response = session.head(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    length = response.headers.get("Content-Length")
    return {
        "etag": response.headers.get("ETag", "").strip('"'),
        "last_modified": response.headers.get("Last-Modified", ""),
        "content_length": int(length) if length and length.isdigit() else None,
    }


def _download_atomic(session: requests.Session, url: str, target: Path, timeout: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with session.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if partial.stat().st_size <= 0:
        raise ArgoError(f"Downloaded zero bytes for {target.name}")
    os.replace(partial, target)


def ensure_index(
    product: str,
    cache_dir: str | Path,
    *,
    refresh: bool = False,
    allow_stale_offline: bool = False,
    index_dir: str | Path | None = None,
    timeout: float = 60.0,
    session: requests.Session | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Return a validated local official index and provenance metadata."""
    if product not in PRODUCTS:
        raise ArgoError(f"Unknown Argo product: {product}")
    name = PRODUCTS[product]
    if index_dir is not None:
        source = Path(index_dir) / name
        if not source.is_file() or source.stat().st_size <= 0:
            raise ArgoError(f"Local index fixture is missing: {name}")
        load_index(source, product)
        return source, {
            "product": product,
            "name": name,
            "source": "local_fixture",
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
            "index_update": _index_update_timestamp(source),
            "retrieved_utc": None,
            "etag": "",
            "last_modified": "",
        }

    root = Path(cache_dir) / "indexes"
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    sidecar = root / f"{name}.metadata.json"
    previous: dict[str, Any] = {}
    if sidecar.exists():
        with contextlib.suppress(Exception):
            previous = json.loads(sidecar.read_text(encoding="utf-8"))
    url = f"{PRIMARY_BASE}/{name}"
    client = session or _session()
    remote: dict[str, Any] | None = None
    remote_error = ""
    try:
        remote = _remote_metadata(client, url, timeout)
    except Exception as exc:
        remote_error = f"{type(exc).__name__}: {exc}"

    valid_local = target.is_file() and target.stat().st_size > 0
    changed = refresh or not valid_local
    if remote and valid_local and not changed:
        comparisons = [
            remote.get("etag") and previous.get("etag") and remote["etag"] != previous["etag"],
            remote.get("last_modified") and previous.get("last_modified") and remote["last_modified"] != previous["last_modified"],
            remote.get("content_length") and remote["content_length"] != target.stat().st_size,
        ]
        changed = any(comparisons)
    if remote is None and not (valid_local and allow_stale_offline):
        raise ArgoError(f"Cannot verify {name}; use --allow-stale-offline only for deliberate offline work: {remote_error}")
    if changed:
        if remote is None:
            raise ArgoError(f"Cannot refresh {name} while offline")
        _download_atomic(client, url, target, timeout)
    load_index(target, product)
    metadata = {
        "product": product,
        "name": name,
        "source": "coriolis_https",
        "url": url,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "index_update": _index_update_timestamp(target),
        "retrieved_utc": utc_now() if changed else previous.get("retrieved_utc"),
        "checked_utc": utc_now(),
        "etag": (remote or previous).get("etag", ""),
        "last_modified": (remote or previous).get("last_modified", ""),
        "offline_stale_used": remote is None,
    }
    atomic_write_json(sidecar, metadata)
    return target, metadata


def inventory_products(
    products: Sequence[str],
    cache_dir: str | Path,
    *,
    refresh: bool = False,
    allow_stale_offline: bool = False,
    index_dir: str | Path | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    entries = []
    for product in products:
        path, meta = ensure_index(
            product,
            cache_dir,
            refresh=refresh,
            allow_stale_offline=allow_stale_offline,
            index_dir=index_dir,
            timeout=timeout,
        )
        frame = load_index(path, product)
        dates = frame["date_compact"]
        entries.append(
            {
                **meta,
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "date_min": dates.min() if len(frame) else "",
                "date_max": dates.max() if len(frame) else "",
                "latitude_min": float(frame["latitude"].min()) if len(frame) else None,
                "latitude_max": float(frame["latitude"].max()) if len(frame) else None,
                "longitude_min": float(frame["longitude"].min()) if len(frame) else None,
                "longitude_max": float(frame["longitude"].max()) if len(frame) else None,
                "dacs": sorted(frame["dac"].unique().tolist()),
                "file_modes": sorted(frame["file_mode"].unique().tolist()),
            }
        )
    return {
        "schema": "argo_inventory_v1",
        "created_utc": utc_now(),
        "official_doi": DOI,
        "products": entries,
    }


def _profile_urls(relative_file: str) -> tuple[str, str]:
    relative = relative_file.replace("\\", "/").lstrip("/")
    return f"{PRIMARY_BASE}/dac/{relative}", f"{S3_BASE}/dac/{relative}"


def _head_size(session: requests.Session, url: str, timeout: float) -> int | None:
    try:
        response = session.head(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        value = response.headers.get("Content-Length")
        if value and value.isdigit() and int(value) > 0:
            return int(value)
    except requests.RequestException:
        pass
    try:
        response = session.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout, stream=True)
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        match = re.search(r"/(\d+)$", content_range)
        if match:
            return int(match.group(1))
        value = response.headers.get("Content-Length")
        return int(value) if value and value.isdigit() and int(value) > 1 else None
    except requests.RequestException:
        return None
    finally:
        with contextlib.suppress(Exception):
            response.close()  # type: ignore[possibly-undefined]


def deterministic_sample(rows: Sequence[Mapping[str, Any]], maximum: int = ESTIMATE_SAMPLE) -> list[Mapping[str, Any]]:
    """Return a deterministic round-robin sample across product/DAC/mode strata."""
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["product"]), str(row["dac"]), str(row["file_mode"]))
        groups.setdefault(key, []).append(row)
    for values in groups.values():
        values.sort(key=lambda row: hashlib.sha256(str(row["file"]).encode()).hexdigest())
    selected: list[Mapping[str, Any]] = []
    offset = 0
    keys = sorted(groups)
    while len(selected) < min(maximum, len(rows)):
        progressed = False
        for key in keys:
            if offset < len(groups[key]) and len(selected) < maximum:
                selected.append(groups[key][offset])
                progressed = True
        if not progressed:
            break
        offset += 1
    return selected


def estimate_remote_bytes(
    rows: Sequence[Mapping[str, Any]],
    *,
    timeout: float = 30.0,
    session: requests.Session | None = None,
    size_lookup: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if not rows:
        return {"method": "empty", "files_probed": 0, "exact_bytes": 0, "conservative_bytes": 0, "credible": True}
    exact = len(rows) <= EXACT_HEAD_LIMIT
    sample = list(rows) if exact else deterministic_sample(rows)
    client = session or _session()
    sizes: list[int] = []
    failed: list[str] = []
    for row in sample:
        relative = str(row["file"])
        size = int(size_lookup[relative]) if size_lookup and relative in size_lookup else None
        if size is None:
            primary, fallback = _profile_urls(relative)
            size = _head_size(client, primary, timeout) or _head_size(client, fallback, timeout)
        if size is None or size <= 0:
            failed.append(relative)
        else:
            sizes.append(size)
    if failed or not sizes:
        return {
            "method": "exact_head" if exact else "stratified_head_sample",
            "files_probed": len(sample),
            "failed_probes": len(failed),
            "credible": False,
            "conservative_bytes": None,
        }
    if exact:
        total = sum(sizes)
        return {"method": "exact_head", "files_probed": len(sizes), "exact_bytes": total, "conservative_bytes": total, "credible": True}
    ordered = sorted(sizes)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    conservative_each = max(p95, max(ordered))
    return {
        "method": "deterministic_product_dac_mode_sample",
        "files_probed": len(sizes),
        "sample_p95_bytes": p95,
        "sample_max_bytes": max(ordered),
        "conservative_per_file_bytes": conservative_each,
        "conservative_bytes": conservative_each * len(rows),
        "credible": True,
    }


def transport_probe(
    relative_file: str,
    *,
    timeout: float = 30.0,
    session: requests.Session | None = None,
    probe_bytes: int = 65536,
) -> dict[str, Any]:
    client = session or _session()
    primary, fallback = _profile_urls(relative_file)
    errors: list[str] = []
    for mirror, url in (("coriolis_https", primary), ("argo_s3", fallback)):
        started = time.perf_counter()
        try:
            response = client.get(url, headers={"Range": f"bytes=0-{probe_bytes - 1}"}, timeout=timeout, stream=True)
            response.raise_for_status()
            received = 0
            for chunk in response.iter_content(16384):
                received += len(chunk)
                if received >= probe_bytes:
                    break
            response.close()
            elapsed = max(time.perf_counter() - started, 1e-6)
            if received <= 0:
                raise ArgoError("zero-byte probe")
            return {"ok": True, "mirror": mirror, "bytes": received, "seconds": elapsed, "bytes_per_second": received / elapsed}
        except Exception as exc:
            errors.append(f"{mirror}:{type(exc).__name__}")
    return {"ok": False, "errors": errors, "bytes_per_second": None}


def _plan_rows(frames: Sequence[pd.DataFrame]) -> list[dict[str, Any]]:
    columns = [
        "product", "file", "date_compact", "latitude", "longitude", "dac", "wmo",
        "file_mode", "parameters", "parameter_data_mode", "date_update",
    ]
    rows: list[dict[str, Any]] = []
    for frame in frames:
        for _, item in frame.iterrows():
            row = {key: item.get(key, "") for key in columns}
            row["date"] = row.pop("date_compact")
            row["latitude"] = float(row["latitude"])
            row["longitude"] = float(row["longitude"])
            row["local_path"] = safe_relative_profile_path(str(row["file"])).as_posix()
            rows.append(row)
    return sorted(rows, key=lambda row: (row["product"], row["date"], row["file"]))


def build_download_plan(
    request: Mapping[str, Any],
    run_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    refresh_indexes: bool = False,
    allow_stale_offline: bool = False,
    index_dir: str | Path | None = None,
    timeout: float = 60.0,
    size_lookup: Mapping[str, int] | None = None,
    probe_result: Mapping[str, Any] | None = None,
    free_bytes_override: int | None = None,
) -> dict[str, Any]:
    canonical = validate_request(request)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir else root / "cache"
    metas: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    for product in canonical["products"]:
        path, meta = ensure_index(
            product,
            cache,
            refresh=refresh_indexes,
            allow_stale_offline=allow_stale_offline,
            index_dir=index_dir,
            timeout=timeout,
        )
        metas.append(meta)
        selected_frames.append(select_profiles(load_index(path, product), request, product))
    rows = _plan_rows(selected_frames)
    estimate = estimate_remote_bytes(rows, timeout=timeout, size_lookup=size_lookup)
    if probe_result is None:
        probe = transport_probe(rows[0]["file"], timeout=timeout) if rows else {"ok": True, "bytes_per_second": None, "bytes": 0, "seconds": 0}
    else:
        probe = dict(probe_result)
    conservative = estimate.get("conservative_bytes")
    credible = bool(estimate.get("credible")) and (not rows or probe.get("ok") is True)
    runtime = None
    if conservative is not None and probe.get("bytes_per_second"):
        runtime = float(conservative) / max(float(probe["bytes_per_second"]), 1.0)
    free_bytes = int(free_bytes_override) if free_bytes_override is not None else shutil.disk_usage(root.resolve()).free
    required_free = int(conservative) * 4 if conservative is not None else None
    storage_ok = required_free is not None and free_bytes > required_free
    created = datetime.now(timezone.utc)
    empty_selection = len(rows) == 0
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "created_utc": created.isoformat().replace("+00:00", "Z"),
        "expires_utc": (created + timedelta(hours=PLAN_HOURS)).isoformat().replace("+00:00", "Z"),
        "request": canonical,
        "request_hash": sha256_value(canonical),
        "indexes": metas,
        "selection_count": len(rows),
        "selected_rows": rows,
        "estimate": estimate,
        "transport_probe": probe,
        "estimated_runtime_seconds": runtime,
        "storage": {
            "local_free_bytes": free_bytes,
            "required_free_bytes": required_free,
            "multiplier": 4,
            "passes": storage_ok,
        },
        "blocked": empty_selection or not credible or not storage_ok,
        "block_reasons": [
            *([] if not empty_selection else ["selection_empty"]),
            *([] if credible else ["no_credible_size_or_transport_estimate"]),
            *([] if storage_ok else ["local_free_space_not_greater_than_four_times_estimate"]),
        ],
        "provenance": {"doi": DOI, "access_date_utc": created.date().isoformat(), "primary": PRIMARY_BASE, "fallback": S3_BASE},
    }
    plan["plan_hash"] = sha256_value(plan)
    atomic_write_json(root / "selection.json", {"schema": "argo_selection_v1", "request_hash": plan["request_hash"], "rows": rows})
    atomic_write_csv(root / "selection.csv", rows)
    return plan


def verify_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ArgoError(f"plan.schema must be {PLAN_SCHEMA!r}")
    supplied = str(plan.get("plan_hash", ""))
    body = dict(plan)
    body.pop("plan_hash", None)
    if not supplied or supplied != sha256_value(body):
        raise ArgoError("Download plan hash does not match its content")
    if parse_utc(str(plan["expires_utc"]), "expires_utc") <= datetime.now(timezone.utc):
        raise ArgoError("Download plan has expired; estimate again")


def _selected_revision_map(plan: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    return {(str(row["product"]), str(row["file"])): str(row.get("date_update", "")) for row in plan["selected_rows"]}


def recheck_selected_rows(
    plan: Mapping[str, Any],
    cache_dir: str | Path,
    *,
    index_dir: str | Path | None = None,
    allow_stale_offline: bool = False,
    timeout: float = 60.0,
) -> None:
    expected = _selected_revision_map(plan)
    current: dict[tuple[str, str], str] = {}
    products = sorted({product for product, _ in expected})
    for product in products:
        path, _ = ensure_index(
            product,
            cache_dir,
            refresh=False,
            allow_stale_offline=allow_stale_offline,
            index_dir=index_dir,
            timeout=timeout,
        )
        frame = load_index(path, product)
        wanted = {file for prod, file in expected if prod == product}
        for _, row in frame.loc[frame["file"].isin(wanted)].iterrows():
            current[(product, str(row["file"]))] = str(row["date_update"])
    missing = sorted(set(expected).difference(current))
    changed = sorted(key for key in expected if key in current and expected[key] != current[key])
    if missing or changed:
        details = []
        if missing:
            details.append(f"{len(missing)} selected rows disappeared")
        if changed:
            details.append(f"{len(changed)} selected date_update values changed")
        raise ArgoError("Selected GDAC rows require replanning: " + "; ".join(details))


def _decode_chars(value: Any) -> str:
    array = np.asarray(value)
    if array.dtype.kind == "S":
        return b"".join(array.reshape(-1).tolist()).decode("ascii", errors="ignore").strip(" \x00")
    if array.dtype.kind == "U":
        return "".join(array.reshape(-1).tolist()).strip(" \x00")
    return str(value).strip()


def internal_date_update(path: str | Path) -> str:
    import xarray as xr

    with xr.open_dataset(path, engine="netcdf4", decode_cf=False, mask_and_scale=False) as dataset:
        if "DATE_UPDATE" in dataset.variables:
            return compact_argo_time(_decode_chars(dataset["DATE_UPDATE"].values))
        return compact_argo_time(dataset.attrs.get("date_update", ""))


def netcdf_opens(path: str | Path) -> tuple[bool, str]:
    try:
        import xarray as xr

        with xr.open_dataset(path, engine="netcdf4", decode_cf=False, mask_and_scale=False) as dataset:
            if not dataset.sizes:
                return False, "no_dimensions"
            _ = list(dataset.variables)
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"


def _manifest_map(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["file"]: row for row in csv.DictReader(handle) if row.get("file")}


def reusable_file(path: Path, row: Mapping[str, Any], record: Mapping[str, Any] | None) -> tuple[bool, str]:
    if not record or not path.is_file() or path.stat().st_size <= 0:
        return False, "missing_or_unrecorded"
    if str(record.get("index_date_update", "")) != str(row.get("date_update", "")):
        return False, "index_revision_changed"
    try:
        if int(record.get("bytes", "0")) != path.stat().st_size:
            return False, "size_changed"
    except ValueError:
        return False, "invalid_recorded_size"
    if record.get("sha256") != sha256_file(path):
        return False, "hash_changed"
    opened, reason = netcdf_opens(path)
    return (True, "validated_cache") if opened else (False, reason)


def _download_with_resume(
    session: requests.Session,
    url: str,
    partial: Path,
    *,
    timeout: float,
    progress_callback=None,
) -> tuple[int, str]:
    partial.parent.mkdir(parents=True, exist_ok=True)
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with session.get(url, headers=headers, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        last_modified = response.headers.get("Last-Modified", "")
        append = existing > 0 and response.status_code == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            total = existing if append else 0
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    total += len(chunk)
                    if progress_callback:
                        progress_callback(len(chunk))
    return partial.stat().st_size, last_modified


def _download_one(
    row: Mapping[str, Any],
    root: Path,
    *,
    timeout: float,
    retries: int,
    existing_record: Mapping[str, Any] | None,
    sleep=time.sleep,
) -> dict[str, Any]:
    relative_local = Path(str(row["local_path"]))
    target = root / relative_local
    reusable, reuse_reason = reusable_file(target, row, existing_record)
    if reusable:
        return {
            "product": row["product"], "file": row["file"], "status": "reused_validated",
            "attempts": 0, "mirror": existing_record.get("mirror", "cache"),
            "bytes": target.stat().st_size, "sha256": sha256_file(target),
            "index_date_update": row.get("date_update", ""), "internal_date_update": internal_date_update(target),
            "local_path": relative_local.as_posix(), "error": "",
        }
    primary, fallback = _profile_urls(str(row["file"]))
    partial = target.with_name(target.name + ".part")
    errors: list[str] = []
    attempts = 0
    chosen = ""
    for attempt in range(1, retries + 1):
        attempts = attempt
        mirror, url = ("coriolis_https", primary) if attempt <= math.ceil(retries / 2) else ("argo_s3", fallback)
        chosen = mirror
        try:
            client = _session()
            _, object_last_modified = _download_with_resume(client, url, partial, timeout=timeout)
            if partial.stat().st_size <= 0:
                raise ArgoError("zero-byte partial")
            opened, reason = netcdf_opens(partial)
            if not opened:
                raise ArgoError(f"NetCDF validation failed: {reason}")
            internal = internal_date_update(partial)
            selected_update = compact_argo_time(row.get("date_update", ""))
            if mirror == "argo_s3":
                from email.utils import parsedate_to_datetime

                try:
                    object_update = parsedate_to_datetime(object_last_modified).astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
                except Exception as exc:
                    raise ArgoError("S3 fallback lacks valid object Last-Modified evidence") from exc
                if object_update < selected_update or not internal or internal < selected_update:
                    raise ArgoError("S3 fallback object or internal DATE_UPDATE is older than the selected index row")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(partial, target)
            return {
                "product": row["product"], "file": row["file"], "status": "downloaded",
                "attempts": attempts, "mirror": mirror, "bytes": target.stat().st_size,
                "sha256": sha256_file(target), "index_date_update": row.get("date_update", ""),
                "internal_date_update": internal, "local_path": relative_local.as_posix(),
                "error": "", "replaced_cache_reason": reuse_reason,
            }
        except Exception as exc:
            errors.append(f"{chosen}:{type(exc).__name__}:{safe_message(exc)}")
            if attempt < retries:
                sleep(min(2 ** (attempt - 1), 16))
    return {
        "product": row["product"], "file": row["file"], "status": "failed",
        "attempts": attempts, "mirror": chosen, "bytes": partial.stat().st_size if partial.exists() else 0,
        "sha256": "", "index_date_update": row.get("date_update", ""), "internal_date_update": "",
        "local_path": relative_local.as_posix(), "error": " | ".join(errors),
    }


def fetch_plan(
    plan: Mapping[str, Any],
    run_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    index_dir: str | Path | None = None,
    allow_stale_offline: bool = False,
    timeout: float = 90.0,
    workers: int = DEFAULT_WORKERS,
    retries: int = 5,
    recheck: bool = True,
) -> list[dict[str, Any]]:
    verify_plan(plan)
    if plan.get("blocked"):
        raise ArgoError("Plan is blocked: " + ", ".join(plan.get("block_reasons", [])))
    if not 1 <= workers <= MAX_WORKERS:
        raise ArgoError(f"workers must be between 1 and {MAX_WORKERS}")
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    required_free = plan.get("storage", {}).get("required_free_bytes")
    if required_free is None or shutil.disk_usage(root.resolve()).free <= int(required_free):
        raise ArgoError("Current local free space is not greater than four times the planned conservative size")
    cache = Path(cache_dir) if cache_dir else root / "cache"
    if recheck:
        recheck_selected_rows(plan, cache, index_dir=index_dir, allow_stale_offline=allow_stale_offline, timeout=timeout)
    manifest_path = root / "download_manifest.csv"
    previous = _manifest_map(manifest_path)
    rows = list(plan.get("selected_rows", []))
    write_status(root, phase="fetch", state="running", total_files=len(rows), completed_files=0, failed_files=0, message="Starting bounded Argo transfer")
    monitor = None
    if float(plan.get("estimated_runtime_seconds") or 0) >= 600:
        monitor = launch_monitor(root)
    results: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _download_one, row, root, timeout=timeout, retries=retries,
                    existing_record=previous.get(str(row["file"])),
                ): row
                for row in rows
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # defensive containment for a single file
                    result = {
                        "product": row["product"], "file": row["file"], "status": "failed",
                        "attempts": 0, "mirror": "", "bytes": 0, "sha256": "",
                        "index_date_update": row.get("date_update", ""), "internal_date_update": "",
                        "local_path": row["local_path"], "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                    }
                results.append(result)
                completed += 1
                failed += result["status"] == "failed"
                atomic_write_csv(manifest_path, sorted(results, key=lambda value: (value["product"], value["file"])))
                write_status(
                    root, phase="fetch", state="running", total_files=len(rows), completed_files=completed,
                    failed_files=failed, current_file=Path(str(row["file"])).name,
                    transferred_bytes=sum(int(item.get("bytes", 0)) for item in results),
                )
    finally:
        if monitor:
            monitor["server"].shutdown()
            monitor["server"].server_close()
    state = "complete" if failed == 0 and completed == len(rows) else "failed"
    write_status(root, phase="fetch", state=state, total_files=len(rows), completed_files=completed, failed_files=failed, message="Transfer finished")
    return sorted(results, key=lambda value: (value["product"], value["file"]))


def _strings_along_first(value: Any, count: int) -> list[str]:
    array = np.asarray(value)
    if count <= 0:
        return []
    if array.ndim == 0:
        return [_decode_chars(array)] * count
    if array.shape[0] == count:
        return [_decode_chars(array[index]) for index in range(count)]
    text = _decode_chars(array)
    return [text] * count


def _flag_counts(value: Any) -> dict[str, int]:
    array = np.asarray(value)
    if np.ma.isMaskedArray(array):
        array = array.compressed()
    if array.dtype.kind in {"S", "U", "O"}:
        characters = []
        for item in array.reshape(-1):
            if item is None or (isinstance(item, (float, np.floating)) and not np.isfinite(item)):
                continue
            text = _decode_chars(item)
            characters.extend(char for char in text if char.strip())
    else:
        characters = [str(item) for item in array.reshape(-1) if np.isfinite(item)]
    result: dict[str, int] = {}
    for item in characters:
        result[item] = result.get(item, 0) + 1
    return result


def _aligned_parameter_modes(dataset, n_prof: int) -> dict[str, dict[str, int]]:
    """Preserve STATION_PARAMETERS/PARAMETER_DATA_MODE positional alignment."""
    if "STATION_PARAMETERS" not in dataset or "PARAMETER_DATA_MODE" not in dataset:
        return {}
    parameters = np.asarray(dataset["STATION_PARAMETERS"].values)
    modes = np.asarray(dataset["PARAMETER_DATA_MODE"].values)
    if parameters.ndim < 2 or modes.ndim < 2:
        return {}
    result: dict[str, dict[str, int]] = {}
    for profile in range(min(n_prof, parameters.shape[0], modes.shape[0])):
        count = min(parameters.shape[1], modes.shape[1])
        for index in range(count):
            parameter = _decode_chars(parameters[profile, index]).strip().upper()
            mode = _decode_chars(modes[profile, index]).strip().upper()
            if parameter and mode:
                result.setdefault(parameter, {})[mode] = result.setdefault(parameter, {}).get(mode, 0) + 1
    return result


def _merge_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _finite_count(dataset, variable: str) -> int:
    if variable not in dataset.variables:
        return 0
    values = np.asarray(dataset[variable].values)
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    if np.issubdtype(values.dtype, np.number):
        return int(np.isfinite(values).sum())
    return 0


def _requested_parameter_presence(dataset, parameters: Sequence[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for parameter in parameters:
        raw = parameter in dataset.variables
        adjusted_name = f"{parameter}_ADJUSTED"
        adjusted = adjusted_name in dataset.variables
        finite = _finite_count(dataset, parameter) + _finite_count(dataset, adjusted_name)
        result[parameter] = {
            "raw_available": raw,
            "adjusted_available": adjusted,
            "adjusted_error_available": f"{parameter}_ADJUSTED_ERROR" in dataset.variables,
            "finite_observations": finite,
        }
    return result


def _profile_metadata(dataset, count: int) -> tuple[list[float], list[float], list[str]]:
    lat = np.asarray(dataset["LATITUDE"].values).reshape(-1) if "LATITUDE" in dataset else np.full(count, np.nan)
    lon = np.asarray(dataset["LONGITUDE"].values).reshape(-1) if "LONGITUDE" in dataset else np.full(count, np.nan)
    times: list[str] = [""] * count
    if "JULD" in dataset:
        raw = np.asarray(dataset["JULD"].values).reshape(-1)
        for index in range(min(count, len(raw))):
            value = raw[index]
            if not pd.isna(value):
                with contextlib.suppress(Exception):
                    times[index] = pd.Timestamp(value).isoformat()
    latitudes = [float(lat[i]) if i < len(lat) and np.isfinite(lat[i]) else math.nan for i in range(count)]
    longitudes = [float(lon[i]) if i < len(lon) and np.isfinite(lon[i]) else math.nan for i in range(count)]
    return latitudes, longitudes, times


def inspect_native_file(path: Path, row: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import xarray as xr

    with xr.open_dataset(path, engine="netcdf4", mask_and_scale=True) as dataset:
        n_prof = int(dataset.sizes.get("N_PROF", 0))
        level_dims = {name: int(size) for name, size in dataset.sizes.items() if name.startswith("N_LEVELS")}
        data_modes = _strings_along_first(dataset["DATA_MODE"].values, n_prof) if "DATA_MODE" in dataset else [""] * n_prof
        parameter_modes: dict[str, int] = {}
        if "PARAMETER_DATA_MODE" in dataset:
            _merge_counts(parameter_modes, _flag_counts(dataset["PARAMETER_DATA_MODE"].values))
        aligned_parameter_modes = _aligned_parameter_modes(dataset, n_prof)
        qc: dict[str, dict[str, int]] = {}
        for name in dataset.variables:
            if name.endswith("_QC") or name in {"POSITION_QC", "JULD_QC"}:
                qc[name] = _flag_counts(dataset[name].values)
        requested = [str(value) for value in row.get("requested_parameters", [])]
        if not requested:
            requested = [value for value in ("PRES", "TEMP", "PSAL") if value in dataset.variables or f"{value}_ADJUSTED" in dataset.variables]
        presence = _requested_parameter_presence(dataset, requested)
        latitudes, longitudes, times = _profile_metadata(dataset, n_prof)
        incomplete = {
            "coordinates": sum(not (np.isfinite(latitudes[i]) and np.isfinite(longitudes[i])) for i in range(n_prof)),
            "times": sum(not value for value in times),
            "requested_variables": sum(any(item["finite_observations"] <= 0 for item in presence.values()) for _ in range(1)) if presence else 0,
        }
        raw_available = sorted(name for name in dataset.data_vars if not name.endswith(("_ADJUSTED", "_ADJUSTED_QC", "_ADJUSTED_ERROR")))
        adjusted_available = sorted(name for name in dataset.data_vars if name.endswith("_ADJUSTED"))
        adjusted_errors = sorted(name for name in dataset.data_vars if name.endswith("_ADJUSTED_ERROR"))
        profiles = []
        platform = _strings_along_first(dataset["PLATFORM_NUMBER"].values, n_prof) if "PLATFORM_NUMBER" in dataset else [str(row.get("wmo", ""))] * n_prof
        for index in range(n_prof):
            profiles.append(
                {
                    "product": row["product"], "file": row["file"], "profile_index": index,
                    "wmo": platform[index] if index < len(platform) else str(row.get("wmo", "")),
                    "latitude": latitudes[index], "longitude": longitudes[index], "time": times[index],
                    "data_mode": data_modes[index] if index < len(data_modes) else "",
                    "has_complete_coordinates": np.isfinite(latitudes[index]) and np.isfinite(longitudes[index]),
                    "has_time": bool(times[index]),
                }
            )
        summary = {
            "n_prof": n_prof,
            "level_dimensions": level_dims,
            "data_mode_counts": _flag_counts(np.asarray(data_modes, dtype="U1")),
            "parameter_data_mode_counts": parameter_modes,
            "parameter_data_mode_by_parameter": aligned_parameter_modes,
            "qc_counts": qc,
            "requested_parameters": presence,
            "raw_variables": raw_available,
            "adjusted_variables": adjusted_available,
            "adjusted_error_variables": adjusted_errors,
            "incomplete": incomplete,
        }
        return summary, profiles


def _write_health_plots(profiles: Sequence[Mapping[str, Any]], health: Mapping[str, Any], plots_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    frame = pd.DataFrame(profiles)

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if not frame.empty:
        for product, subset in frame.groupby("product"):
            axis.scatter(subset["longitude"], subset["latitude"], s=16, alpha=0.7, label=product)
        axis.legend()
    axis.set(xlabel="Longitude", ylabel="Latitude", title="Selected Argo profile coverage")
    axis.grid(alpha=0.25)
    path = plots_dir / "spatial_coverage.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    outputs.append(path.name)

    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if not frame.empty:
        times = pd.to_datetime(frame["time"], errors="coerce", utc=True).dropna()
        if len(times):
            counts = times.dt.tz_localize(None).dt.to_period("M").value_counts().sort_index()
            axis.plot([str(value) for value in counts.index], counts.values, marker=".")
            if len(counts) > 12:
                axis.set_xticks(range(0, len(counts), max(1, len(counts) // 10)))
            axis.tick_params(axis="x", rotation=45)
    axis.set(xlabel="Month", ylabel="Profiles", title="Argo time coverage")
    axis.grid(alpha=0.25)
    path = plots_dir / "time_coverage.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    outputs.append(path.name)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    data_modes = health.get("summary", {}).get("data_mode_counts", {})
    qc_counts = health.get("summary", {}).get("all_qc_flag_counts", {})
    axes[0].bar(list(data_modes), list(data_modes.values()))
    axes[0].set(title="DATA_MODE", xlabel="Mode", ylabel="Profiles")
    axes[1].bar(list(qc_counts), list(qc_counts.values()))
    axes[1].set(title="All reported QC flags", xlabel="Flag", ylabel="Occurrences")
    path = plots_dir / "qc_data_mode_summary.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    outputs.append(path.name)
    return outputs


def health_check(
    plan: Mapping[str, Any],
    run_dir: str | Path,
    *,
    output: str | Path | None = None,
    plots_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate native files, revisions, dimensions, requested data, and QA summaries."""
    verify_plan(plan)
    root = Path(run_dir)
    manifest_path = root / "download_manifest.csv"
    manifest = _manifest_map(manifest_path)
    selected = list(plan.get("selected_rows", []))
    failures: list[dict[str, str]] = []
    file_reports: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    data_modes: dict[str, int] = {}
    parameter_modes: dict[str, int] = {}
    aligned_parameter_modes: dict[str, dict[str, int]] = {}
    all_qc: dict[str, int] = {}
    request_parameters = list(plan.get("request", {}).get("parameters", []))

    if not selected:
        failures.append({"code": "empty_selection", "detail": "The plan selected no native profile files"})

    if len(manifest) != len(selected):
        failures.append({"code": "selection_download_count_mismatch", "detail": f"selected={len(selected)} manifest={len(manifest)}"})
    for row in selected:
        file_name = str(row["file"])
        record = manifest.get(file_name)
        path = root / Path(str(row["local_path"]))
        if record is None:
            failures.append({"code": "missing_manifest_record", "detail": file_name})
            continue
        if record.get("status") == "failed" or not path.is_file():
            failures.append({"code": "missing_selected_file", "detail": file_name})
            continue
        if path.stat().st_size <= 0:
            failures.append({"code": "zero_byte_file", "detail": file_name})
            continue
        actual_hash = sha256_file(path)
        if actual_hash != record.get("sha256"):
            failures.append({"code": "sha256_mismatch", "detail": file_name})
            continue
        opened, open_reason = netcdf_opens(path)
        if not opened:
            failures.append({"code": "netcdf_open_failure", "detail": f"{file_name}: {open_reason}"})
            continue
        internal = internal_date_update(path)
        if not internal or internal < compact_argo_time(row.get("date_update", "")):
            failures.append({"code": "stale_native_file", "detail": file_name})
        inspect_row = dict(row)
        inspect_row["requested_parameters"] = request_parameters if row["product"] in {"bio", "synthetic"} else []
        try:
            summary, profiles = inspect_native_file(path, inspect_row)
        except Exception as exc:
            failures.append({"code": "netcdf_open_failure", "detail": f"{file_name}: {type(exc).__name__}"})
            continue
        if summary["n_prof"] <= 0 or not summary["level_dimensions"] or max(summary["level_dimensions"].values(), default=0) <= 0:
            failures.append({"code": "empty_profile_or_level_dimension", "detail": file_name})
        for parameter, evidence in summary["requested_parameters"].items():
            if not evidence["raw_available"] and not evidence["adjusted_available"]:
                failures.append({"code": "requested_parameter_absent", "detail": f"{file_name}:{parameter}"})
            elif evidence["finite_observations"] <= 0:
                failures.append({"code": "no_finite_requested_observations", "detail": f"{file_name}:{parameter}"})
        _merge_counts(data_modes, summary["data_mode_counts"])
        _merge_counts(parameter_modes, summary["parameter_data_mode_counts"])
        for parameter, modes in summary["parameter_data_mode_by_parameter"].items():
            aligned_parameter_modes.setdefault(parameter, {})
            _merge_counts(aligned_parameter_modes[parameter], modes)
        for counts in summary["qc_counts"].values():
            _merge_counts(all_qc, counts)
        profile_rows.extend(profiles)
        file_reports.append(
            {
                "product": row["product"], "file": file_name, "bytes": path.stat().st_size,
                "sha256": actual_hash, "index_date_update": row.get("date_update", ""),
                "internal_date_update": internal, **summary,
            }
        )

    health: dict[str, Any] = {
        "schema": "argo_health_check_v1",
        "created_utc": utc_now(),
        "status": "pass" if not failures else "fail",
        "selected_files": len(selected),
        "validated_files": len(file_reports),
        "profiles": len(profile_rows),
        "failures": failures,
        "summary": {
            "data_mode_counts": data_modes,
            "parameter_data_mode_counts": parameter_modes,
            "parameter_data_mode_by_parameter": aligned_parameter_modes,
            "all_qc_flag_counts": all_qc,
            "files_with_incomplete_coordinates": sum(report["incomplete"]["coordinates"] > 0 for report in file_reports),
            "files_with_incomplete_times": sum(report["incomplete"]["times"] > 0 for report in file_reports),
            "files_with_incomplete_requested_variables": sum(report["incomplete"]["requested_variables"] > 0 for report in file_reports),
        },
        "files": file_reports,
        "provenance": plan.get("provenance", {"doi": DOI}),
    }
    inventory_path = root / "profile_inventory.csv"
    atomic_write_csv(inventory_path, profile_rows)
    plot_root = Path(plots_dir) if plots_dir else root / "health_plots"
    health["plots"] = _write_health_plots(profile_rows, health, plot_root)
    output_path = Path(output) if output else root / "health_check.json"
    atomic_write_json(output_path, health)
    write_status(root, phase="health", state=health["status"], total_files=len(selected), completed_files=len(file_reports), failed_files=len(failures), message="Health check finished")
    return health


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Refresh/cache and summarize official GDAC indexes")
    inventory.add_argument("--products", nargs="+", choices=sorted(PRODUCTS), required=True)
    inventory.add_argument("--cache-dir", required=True)
    inventory.add_argument("--output", required=True)
    inventory.add_argument("--refresh", action="store_true")
    inventory.add_argument("--allow-stale-offline", action="store_true")
    inventory.add_argument("--index-dir", help="Local official-index fixture directory")
    inventory.add_argument("--timeout", type=float, default=60.0)

    def add_estimate_arguments(command):
        command.add_argument("--request", required=True)
        command.add_argument("--run-dir", required=True)
        command.add_argument("--output", help="Plan path; defaults to <run-dir>/download_plan.json")
        command.add_argument("--cache-dir")
        command.add_argument("--index-dir", help="Local official-index fixture directory")
        command.add_argument("--refresh-indexes", action="store_true")
        command.add_argument("--allow-stale-offline", action="store_true")
        command.add_argument("--timeout", type=float, default=60.0)

    estimate = subparsers.add_parser("estimate", help="Select rows and create a hash-bound plan")
    add_estimate_arguments(estimate)

    fetch = subparsers.add_parser("fetch", help="Execute or resume an unexpired plan")
    fetch.add_argument("--plan", required=True)
    fetch.add_argument("--run-dir", required=True)
    fetch.add_argument("--cache-dir")
    fetch.add_argument("--index-dir")
    fetch.add_argument("--allow-stale-offline", action="store_true")
    fetch.add_argument("--timeout", type=float, default=90.0)
    fetch.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    fetch.add_argument("--retries", type=int, default=5)

    health = subparsers.add_parser("health", help="Validate downloaded native files and write QA")
    health.add_argument("--plan", required=True)
    health.add_argument("--run-dir", required=True)
    health.add_argument("--output")
    health.add_argument("--plots-dir")

    run = subparsers.add_parser("run", help="Estimate and optionally execute the bounded request")
    add_estimate_arguments(run)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    run.add_argument("--retries", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        result = inventory_products(
            args.products, args.cache_dir, refresh=args.refresh,
            allow_stale_offline=args.allow_stale_offline, index_dir=args.index_dir,
            timeout=args.timeout,
        )
        atomic_write_json(args.output, result)
        _print_json(result)
        return 0

    if args.command in {"estimate", "run"}:
        request = _load_json(args.request)
        plan = build_download_plan(
            request, args.run_dir, cache_dir=args.cache_dir, refresh_indexes=args.refresh_indexes,
            allow_stale_offline=args.allow_stale_offline, index_dir=args.index_dir, timeout=args.timeout,
        )
        output = Path(args.output) if args.output else Path(args.run_dir) / "download_plan.json"
        atomic_write_json(output, plan)
        _print_json(plan)
        if args.command == "estimate" or not args.execute:
            return 2 if plan["blocked"] else 0
        if plan["blocked"]:
            return 2
        fetch_plan(
            plan, args.run_dir, cache_dir=args.cache_dir, index_dir=args.index_dir,
            allow_stale_offline=args.allow_stale_offline, timeout=args.timeout,
            workers=args.workers, retries=args.retries,
        )
        result = health_check(plan, args.run_dir)
        _print_json(result)
        return 0 if result["status"] == "pass" else 3

    if args.command == "fetch":
        plan = _load_json(args.plan)
        results = fetch_plan(
            plan, args.run_dir, cache_dir=args.cache_dir, index_dir=args.index_dir,
            allow_stale_offline=args.allow_stale_offline, timeout=args.timeout,
            workers=args.workers, retries=args.retries,
        )
        _print_json({"files": len(results), "failed": sum(row["status"] == "failed" for row in results)})
        return 0 if all(row["status"] != "failed" for row in results) else 3

    if args.command == "health":
        result = health_check(_load_json(args.plan), args.run_dir, output=args.output, plots_dir=args.plots_dir)
        _print_json(result)
        return 0 if result["status"] == "pass" else 3
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
