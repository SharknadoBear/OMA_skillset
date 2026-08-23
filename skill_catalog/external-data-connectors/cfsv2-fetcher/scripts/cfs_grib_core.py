#!/usr/bin/env python3
"""Shared NCEI-first CFSR/CFSv2 atmospheric acquisition core.

This file is intentionally byte-identical in both skill packages.  Model-specific
entry points pass ``cfsr`` or ``cfsv2`` and retain their legacy compatibility API.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urljoin, urlsplit, urlunsplit
import warnings
import xml.etree.ElementTree as ET

import netCDF4 as nc4
import numpy as np
import requests

try:
    from .download_monitor import DownloadStatus, atomic_write_json, launch_monitor, write_monitor_html
except ImportError:
    from download_monitor import DownloadStatus, atomic_write_json, launch_monitor, write_monitor_html


SCHEMA_REQUEST = "cfs_atmospheric_request_v2"
SCHEMA_PLAN = "cfs_model_download_plan_v2"
SCHEMA_FIELDS = "cfs_atmospheric_fields_v2"
SCHEMA_ROUTING = "cfs_family_routing_manifest_v1"
ERA_SPLIT = datetime(2011, 4, 1, tzinfo=timezone.utc)
ERA_START = {
    "cfsr": datetime(1979, 1, 1, tzinfo=timezone.utc),
    "cfsv2": ERA_SPLIT,
}
ERA_END = {"cfsr": datetime(2011, 3, 31, 23, tzinfo=timezone.utc), "cfsv2": None}

MODEL_CONFIG: dict[str, dict[str, str]] = {
    "cfsr": {
        "connector": "cfsr-fetcher",
        "catalog_root": "https://www.ncei.noaa.gov/thredds/catalog/model-cfs_reanl_ts/catalog.xml",
        "month_catalog": "https://www.ncei.noaa.gov/thredds/catalog/model-cfs_reanl_ts/{yyyymm}/catalog.xml",
        "file": "https://www.ncei.noaa.gov/thredds/fileServer/model-cfs_reanl_ts/{yyyymm}/{stem}.gdas.{yyyymm}.grb2",
        "extension": "grb2",
    },
    "cfsv2": {
        "connector": "cfsv2-fetcher",
        "catalog_root": "https://www.ncei.noaa.gov/thredds/catalog/model-cfs_v2_anl_ts/catalog.xml",
        "month_catalog": "https://www.ncei.noaa.gov/thredds/catalog/model-cfs_v2_anl_ts/{year}/{yyyymm}/catalog.xml",
        "file": "https://www.ncei.noaa.gov/thredds/fileServer/model-cfs_v2_anl_ts/{year}/{yyyymm}/{stem}.gdas.{yyyymm}.grib2",
        "extension": "grib2",
    },
}

# Names are deliberately provider-neutral.  Each field matcher also validates
# its native level; filename matching alone is not accepted.
PRODUCTS: dict[str, dict[str, Any]] = {
    "wind_10m": {
        "stem": "wnd10m",
        "fields": [
            {"name": "eastward_wind", "units": "m s-1", "short": {"ugrd", "10u", "u10"}, "contains": ("u-component of wind", "u component of wind"), "level": {"heightaboveground"}, "value": 10.0},
            {"name": "northward_wind", "units": "m s-1", "short": {"vgrd", "10v", "v10"}, "contains": ("v-component of wind", "v component of wind"), "level": {"heightaboveground"}, "value": 10.0},
        ],
    },
    "surface_pressure": {
        "stem": "pressfc",
        "fields": [{"name": "absolute_air_pressure", "units": "Pa", "short": {"pres", "sp"}, "contains": ("pressure",), "level": {"surface", "sfc"}}],
    },
    "air_temperature_2m": {
        "stem": "tmp2m",
        "fields": [{"name": "air_temperature_2m", "units": "degree_Celsius", "short": {"tmp", "2t", "t2m"}, "contains": ("temperature",), "level": {"heightaboveground"}, "value": 2.0}],
    },
    "specific_humidity_2m": {
        "stem": "q2m",
        "fields": [{"name": "specific_humidity_2m", "units": "kg kg-1", "short": {"spfh", "2sh", "q"}, "contains": ("specific humidity",), "level": {"heightaboveground"}, "value": 2.0}],
    },
    "precipitation_rate": {
        "stem": "prate",
        "fields": [{"name": "precipitation_rate", "units": "kg m-2 s-1", "short": {"prate"}, "contains": ("precipitation rate",), "level": {"surface", "sfc"}}],
    },
    "downward_shortwave_surface_flux": {
        "stem": "dswsfc",
        "fields": [{"name": "surface_downwelling_shortwave_flux", "units": "W m-2", "short": {"dswrf", "ssrd"}, "contains": ("downward short-wave", "downward shortwave"), "level": {"surface", "sfc"}}],
    },
    "downward_longwave_surface_flux": {
        "stem": "dlwsfc",
        "fields": [{"name": "surface_downwelling_longwave_flux", "units": "W m-2", "short": {"dlwrf", "strd"}, "contains": ("downward long-wave", "downward longwave"), "level": {"surface", "sfc"}}],
    },
    "surface_wind_stress": {
        "stem": "wndstrs",
        "fields": [
            {"name": "eastward_wind_stress", "units": "N m-2", "short": {"uflx"}, "contains": ("momentum flux, u component", "u-component momentum flux"), "level": {"surface", "sfc"}},
            {"name": "northward_wind_stress", "units": "N m-2", "short": {"vflx"}, "contains": ("momentum flux, v component", "v-component momentum flux"), "level": {"surface", "sfc"}},
        ],
    },
    "surface_temperature": {
        "stem": "tmpsfc",
        "fields": [{"name": "surface_temperature", "units": "degree_Celsius", "short": {"tmp", "skt"}, "contains": ("temperature",), "level": {"surface", "sfc"}}],
    },
    "latent_heat_net_flux": {
        "stem": "lhtfl",
        "fields": [{"name": "latent_heat_net_flux", "units": "W m-2", "short": {"lhtfl"}, "contains": ("latent heat",), "level": {"surface", "sfc"}}],
    },
    "sensible_heat_net_flux": {
        "stem": "shtfl",
        "fields": [{"name": "sensible_heat_net_flux", "units": "W m-2", "short": {"shtfl"}, "contains": ("sensible heat",), "level": {"surface", "sfc"}}],
    },
}

PRODUCT_ALIASES = {
    "surface_downwelling_shortwave_flux": "downward_shortwave_surface_flux",
    "surface_downwelling_longwave_flux": "downward_longwave_surface_flux",
    "surface_pressure": "surface_pressure",
    "10m_wind": "wind_10m",
}

# Only scientifically exact mappings are eligible for automatic fallback.
HYCOM_MAP: dict[str, dict[str, dict[str, Any]]] = {
    "cfsr": {
        "surface_pressure": {"subdataset": "sfcprs", "fields": {"absolute_air_pressure": ("airprs", "pressure_departure")}},
    },
    "cfsv2": {
        "wind_10m": {"subdataset": "uv-10m", "fields": {"eastward_wind": ("wndewd", "identity"), "northward_wind": ("wndnwd", "identity")}},
        "surface_pressure": {"subdataset": "sfcprs", "fields": {"absolute_air_pressure": ("airprs", "pressure_departure")}},
        "downward_shortwave_surface_flux": {"subdataset": "dswsfc", "fields": {"surface_downwelling_shortwave_flux": ("dswflx", "identity")}},
        "downward_longwave_surface_flux": {"subdataset": "dlwsfc", "fields": {"surface_downwelling_longwave_flux": ("dlwflx", "identity")}},
        "surface_wind_stress": {"subdataset": "strblk", "fields": {"eastward_wind_stress": ("tauewd", "identity"), "northward_wind_stress": ("taunwd", "identity")}},
        "precipitation_rate": {"subdataset": "precip", "fields": {"precipitation_rate": ("precip", "identity")}},
        "surface_temperature": {"subdataset": "surtmp", "fields": {"surface_temperature": ("surtmp", "temperature")}},
    },
}


class CfsAtmosphericError(RuntimeError):
    """Raised when a CFS-family request cannot be fulfilled safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    result = datetime.fromisoformat(text)
    if result.tzinfo is None:
        raise ValueError(f"Timestamp {value!r} must include UTC or an offset")
    return result.astimezone(timezone.utc).replace(microsecond=0)


def time_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def public_url(value: str) -> str:
    parsed = urlsplit(str(value))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Expected a public HTTP(S) URL, got {value!r}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Source URLs cannot contain credentials, queries, or fragments")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def month_keys(start: datetime, end: datetime) -> list[str]:
    cursor = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    final = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    result: list[str] = []
    while cursor <= final:
        result.append(cursor.strftime("%Y%m"))
        cursor = datetime(cursor.year + int(cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1, tzinfo=timezone.utc)
    return result


def normalize_request(payload: Mapping[str, Any], model: str) -> dict[str, Any]:
    if model not in MODEL_CONFIG:
        raise ValueError(f"Unknown CFS model {model!r}")
    start = parse_utc(str(payload.get("start") or payload.get("start_utc")))
    end = parse_utc(str(payload.get("end") or payload.get("end_utc")))
    if end < start:
        raise ValueError("Request end precedes start")
    if start.minute or start.second or end.minute or end.second:
        raise ValueError("CFS atmospheric requests must use exact UTC hours")
    raw_products = payload.get("products")
    if raw_products is None:
        raw_products = [payload.get("product", "surface_pressure")]
    if not isinstance(raw_products, Sequence) or isinstance(raw_products, (str, bytes)):
        raise ValueError("products must be a non-empty list")
    products: list[str] = []
    for raw in raw_products:
        name = PRODUCT_ALIASES.get(str(raw), str(raw))
        if name not in PRODUCTS:
            raise ValueError(f"Unknown product {raw!r}; choose from {', '.join(PRODUCTS)}")
        if name not in products:
            products.append(name)
    if not products:
        raise ValueError("products must not be empty")
    bbox = payload.get("bbox") or payload.get("bbox_0_360")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        raise ValueError("bbox must be [west,south,east,north]")
    west, south, east, north = [float(value) for value in bbox]
    if not (0 <= west < east <= 360 and -90 <= south < north <= 90):
        raise ValueError("bbox must be increasing and use 0-360 longitude")
    provider = str(payload.get("provider", "auto")).lower()
    if provider not in {"auto", "ncei", "hycom"}:
        raise ValueError("provider must be auto, ncei, or hycom")
    order = [str(value).lower() for value in payload.get("provider_order", ["ncei", "hycom"])]
    if provider != "auto":
        order = [provider]
    if not order or len(set(order)) != len(order) or any(item not in {"ncei", "hycom"} for item in order):
        raise ValueError("provider_order must contain unique ncei/hycom values")
    output = Path(str(payload.get("output") or f"{model}_atmospheric_fields.nc")).name
    return {
        "schema_version": SCHEMA_REQUEST,
        "model": model,
        "start": time_text(start),
        "end": time_text(end),
        "products": products,
        "bbox": [west, south, east, north],
        "halo_cells": max(0, int(payload.get("halo_cells", 1))),
        "provider": provider,
        "provider_order": order,
        "max_retries": max(1, int(payload.get("max_retries", 5))),
        "retry_delay_seconds": max(0.0, float(payload.get("retry_delay_seconds", 2.0))),
        "backoff": max(1.0, float(payload.get("backoff", 2.0))),
        "output": output,
        "routing_depth": max(0, int(payload.get("routing_depth", 0))),
        "catalog_root": public_url(str(payload.get("catalog_root", MODEL_CONFIG[model]["catalog_root"]))),
        "month_catalog_template": public_url(str(payload.get("month_catalog_template", MODEL_CONFIG[model]["month_catalog"]))),
        "ncei_file_template": public_url(str(payload.get("ncei_file_template", MODEL_CONFIG[model]["file"]))),
        "hycom_sources": {str(k): public_url(str(v)) for k, v in dict(payload.get("hycom_sources", {})).items()},
    }


def classify_era(start: datetime, end: datetime) -> list[tuple[str, datetime, datetime]]:
    if end < ERA_SPLIT:
        return [("cfsr", start, end)]
    if start >= ERA_SPLIT:
        return [("cfsv2", start, end)]
    return [("cfsr", start, ERA_END["cfsr"]), ("cfsv2", ERA_START["cfsv2"], end)]


def validate_in_era(request: Mapping[str, Any], model: str) -> None:
    start, end = parse_utc(str(request["start"])), parse_utc(str(request["end"]))
    segments = classify_era(start, end)
    if len(segments) != 1 or segments[0][0] != model:
        raise CfsAtmosphericError(f"Request belongs to {','.join(x[0] for x in segments)}; route it before accessing {model} providers")


def _request(session: requests.Session, method: str, url: str, *, retries: int = 3, timeout: tuple[float, float] = (20, 120), **kwargs: Any) -> requests.Response:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.request(method, public_url(url), timeout=timeout, allow_redirects=True, **kwargs)
            if response.status_code >= 500 or response.status_code in {408, 429}:
                raise requests.HTTPError(f"HTTP {response.status_code} for {url}", response=response)
            response.raise_for_status()
            return response
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(min(20.0, 2.0 ** (attempt - 1)))
    raise CfsAtmosphericError(str(last)) from last


def _catalog_refs(xml: bytes, base: str) -> tuple[list[str], list[str]]:
    root = ET.fromstring(xml)
    paths: list[str] = []
    refs: list[str] = []
    for element in root.iter():
        path = element.attrib.get("urlPath")
        if path:
            paths.append(path)
        href = next((value for key, value in element.attrib.items() if key.endswith("href")), None)
        if href:
            refs.append(urljoin(base, href.replace("catalog.html", "catalog.xml")))
    return paths, refs


def discover_available_months(model: str, *, session: requests.Session | None = None, catalog_root: str | None = None) -> dict[str, Any]:
    """Discover archive bounds and internal gaps from live THREDDS catalogs."""
    own = session is None
    client = session or requests.Session()
    root_url = public_url(catalog_root or MODEL_CONFIG[model]["catalog_root"])
    try:
        response = _request(client, "GET", root_url, retries=3)
        paths, refs = _catalog_refs(response.content, root_url)
        months = set(re.findall(r"(?<!\d)((?:19|20)\d{4})(?!\d)", " ".join(paths + refs)))
        year_refs = [ref for ref in refs if re.search(r"/(?:19|20)\d{2}/catalog\.xml$", ref)]
        for ref in year_refs:
            child = _request(client, "GET", ref, retries=3)
            child_paths, child_refs = _catalog_refs(child.content, ref)
            months.update(re.findall(r"(?<!\d)((?:19|20)\d{4})(?!\d)", " ".join(child_paths + child_refs)))
        ordered = sorted(months)
        if not ordered:
            raise CfsAtmosphericError(f"No YYYYMM entries were found in the {model} catalog")
        first = datetime.strptime(ordered[0], "%Y%m").replace(tzinfo=timezone.utc)
        last = datetime.strptime(ordered[-1], "%Y%m").replace(tzinfo=timezone.utc)
        expected = set(month_keys(first, last))
        return {
            "schema_version": "cfs_ncei_coverage_v1",
            "model": model,
            "catalog_url": root_url,
            "discovered_utc": utc_now(),
            "first_month": ordered[0],
            "last_month": ordered[-1],
            "month_count": len(ordered),
            "missing_months": sorted(expected - set(ordered)),
            "available_months": ordered,
        }
    finally:
        if own:
            client.close()


def _range_supported(client: requests.Session, url: str, retries: int) -> tuple[int, str | None, bool]:
    response = _request(client, "HEAD", url, retries=retries)
    length = int(response.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        raise CfsAtmosphericError(f"NCEI did not report a positive Content-Length for {Path(url).name}")
    accepted = response.headers.get("Accept-Ranges", "").lower() == "bytes"
    if not accepted:
        probe = _request(client, "GET", url, retries=retries, headers={"Range": "bytes=0-0"})
        accepted = probe.status_code == 206 and len(probe.content) == 1
    return length, response.headers.get("Last-Modified"), accepted


def ncei_inventory(model: str, request: Mapping[str, Any], *, session: requests.Session | None = None, discover_coverage: bool = True) -> dict[str, Any]:
    validate_in_era(request, model)
    own = session is None
    client = session or requests.Session()
    started = time.monotonic()
    try:
        coverage = discover_available_months(model, session=client, catalog_root=str(request["catalog_root"])) if discover_coverage else None
        required_months = month_keys(parse_utc(str(request["start"])), parse_utc(str(request["end"])))
        if coverage:
            available = set(coverage["available_months"])
            absent = [month for month in required_months if month not in available]
            if absent:
                raise CfsAtmosphericError(f"NCEI {model} catalog lacks requested month(s): {', '.join(absent)}")
        rows: list[dict[str, Any]] = []
        for month in required_months:
            catalog_url = str(request["month_catalog_template"]).format(yyyymm=month, year=month[:4], month=month[4:])
            catalog = _request(client, "GET", catalog_url, retries=int(request["max_retries"]))
            paths, _ = _catalog_refs(catalog.content, catalog_url)
            names = {Path(path).name for path in paths if ".l." not in Path(path).name.lower()}
            for product in request["products"]:
                stem = str(PRODUCTS[product]["stem"])
                expected = f"{stem}.gdas.{month}.{MODEL_CONFIG[model]['extension']}"
                if expected not in names:
                    raise CfsAtmosphericError(f"Full-resolution NCEI file {expected} is absent from {catalog_url}")
                url = str(request["ncei_file_template"]).format(yyyymm=month, year=month[:4], month=month[4:], stem=stem)
                length, modified, ranges = _range_supported(client, url, int(request["max_retries"]))
                if not ranges:
                    raise CfsAtmosphericError(f"NCEI file {expected} does not support byte-range resume")
                rows.append({"id": f"{month}:{product}", "month": month, "product": product, "url": public_url(url), "bytes": length, "last_modified": modified, "accept_ranges": True})
        return {
            "schema_version": "cfs_ncei_inventory_v2",
            "model": model,
            "provider": "ncei",
            "coverage": coverage,
            "source_units": rows,
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        if own:
            client.close()


def _plan_gate(transfer_bytes: int, free_bytes: int, conservative_seconds: float) -> dict[str, Any]:
    storage = free_bytes > 4 * transfer_bytes
    if not storage:
        state, reason = "blocked", "Local free space is not greater than four times the planned raw transfer."
    elif not math.isfinite(conservative_seconds) or conservative_seconds <= 0:
        state, reason = "blocked", "A positive duration estimate was not established."
    elif conservative_seconds >= 600:
        state, reason = "long_run_monitor_required", "Conservative duration is at least ten minutes."
    else:
        state, reason = "ready", "Time and storage gates passed."
    return {"state": state, "reason": reason, "storage_passed": storage, "storage_rule": "free_bytes > 4 * raw_transfer_bytes", "monitor_threshold_seconds": 600}


def _hycom_candidates(model: str, year: int, subdataset: str) -> list[str]:
    if model == "cfsr":
        return [f"https://tds.hycom.org/thredds/dodsC/datasets/force/ncep_cfsr/netcdf/cfsr-sec_{year}_01hr_{subdataset}.nc"]
    base = "https://tds.hycom.org/thredds/dodsC/datasets/force/ncep_cfsv2/netcdf/"
    return [f"{base}{prefix}_{year}_01hr_{subdataset}.nc" for prefix in ("cfsv2-sec2", "cfsv2-sec", "cfsv2-sea")]


def hycom_eligibility(model: str, products: Sequence[str]) -> dict[str, Any]:
    unsupported = [product for product in products if product not in HYCOM_MAP[model]]
    return {"eligible": not unsupported, "unsupported_products": unsupported, "rule": "all requested products require scientifically exact mappings"}


def build_hycom_plan(model: str, request: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    eligibility = hycom_eligibility(model, request["products"])
    if not eligibility["eligible"]:
        raise CfsAtmosphericError("HYCOM whole-request fallback is unavailable for: " + ", ".join(eligibility["unsupported_products"]))
    # OPeNDAP transfer is a decoded subset, so use a conservative float32 cube estimate.
    hours = int((parse_utc(str(request["end"])) - parse_utc(str(request["start"]))).total_seconds() // 3600) + 1
    variables = sum(len(PRODUCTS[p]["fields"]) for p in request["products"])
    estimated = max(4096, hours * variables * 256 * 256 * 4)
    conservative = max(60.0, estimated / (256 * 1024) + 30 * len(request["products"]))
    free = int(shutil.disk_usage(run_dir).free)
    return {
        "provider": "hycom",
        "eligibility": eligibility,
        "raw_transfer_bytes": estimated,
        "expected_hours": hours,
        "duration_estimate_seconds": {"central": round(conservative / 1.5, 3), "conservative": round(conservative, 3)},
        "gate": _plan_gate(estimated, free, conservative),
        "source_units": [],
        "note": "HYCOM URLs are resolved only after the whole-request provider lock is selected.",
    }


def build_plan(model: str, payload: Mapping[str, Any], run_dir: str | Path, *, mode: str = "fetch", discover_coverage: bool = True, free_bytes_override: int | None = None) -> dict[str, Any]:
    if mode not in {"fetch", "snapshot"}:
        raise ValueError("mode must be fetch or snapshot")
    request = normalize_request(payload, model)
    validate_in_era(request, model)
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    provider_plan: dict[str, Any] | None = None
    for provider in request["provider_order"]:
        try:
            if provider == "ncei":
                inventory = ncei_inventory(model, request, discover_coverage=discover_coverage)
                client = requests.Session()
                unit = inventory["source_units"][0]
                sample_size = min(1024 * 1024, int(unit["bytes"]))
                began = time.monotonic()
                sample = _request(client, "GET", str(unit["url"]), retries=int(request["max_retries"]), headers={"Range": f"bytes=0-{sample_size - 1}"})
                elapsed = max(time.monotonic() - began, 1e-6)
                client.close()
                if len(sample.content) != sample_size:
                    raise CfsAtmosphericError("NCEI bounded transfer probe returned an unexpected byte count")
                rate = len(sample.content) / elapsed
                if mode == "snapshot":
                    transfer = min(int(inventory["total_bytes"]), max(4 * 1024 * 1024 * sum(len(PRODUCTS[p]["fields"]) for p in request["products"]), sample_size))
                else:
                    transfer = int(inventory["total_bytes"])
                central = transfer / max(1.0, rate) + 30 * len(inventory["source_units"])
                conservative = central * 1.5 + 60 * len(inventory["source_units"])
                free = int(free_bytes_override if free_bytes_override is not None else shutil.disk_usage(directory).free)
                provider_plan = {
                    "provider": "ncei",
                    "inventory": inventory,
                    "source_units": inventory["source_units"],
                    "raw_transfer_bytes": transfer,
                    "full_month_archive_bytes": int(inventory["total_bytes"]),
                    "expected_hours": int((parse_utc(request["end"]) - parse_utc(request["start"])).total_seconds() // 3600) + 1,
                    "transfer_probe": {"method": "HTTP Range", "bytes": len(sample.content), "elapsed_seconds": round(elapsed, 6), "bytes_per_second": round(rate, 3)},
                    "duration_estimate_seconds": {"central": round(central, 3), "conservative": round(conservative, 3)},
                    "gate": _plan_gate(transfer, free, conservative),
                }
            else:
                provider_plan = build_hycom_plan(model, request, directory)
            attempts.append({"provider": provider, "state": "selected"})
            break
        except Exception as exc:
            attempts.append({"provider": provider, "state": "failed", "error": str(exc)[:1000]})
    if provider_plan is None:
        raise CfsAtmosphericError("No whole-request provider passed planning: " + " | ".join(f"{x['provider']}: {x.get('error')}" for x in attempts))
    body = {
        "schema_version": SCHEMA_PLAN,
        "connector": MODEL_CONFIG[model]["connector"],
        "model": model,
        "mode": mode,
        "created_utc": utc_now(),
        "request": request,
        "request_hash": hash_payload(request),
        "provider_attempts": attempts,
        **provider_plan,
    }
    body["plan_hash"] = hash_payload(body)
    atomic_write_json(directory / "download_plan.json", body)
    return body


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != SCHEMA_PLAN:
        raise ValueError(f"Expected {SCHEMA_PLAN}")
    stored = str(plan.get("plan_hash", ""))
    body = dict(plan)
    body.pop("plan_hash", None)
    if stored != hash_payload(body):
        raise ValueError("Plan hash mismatch")
    if plan.get("gate", {}).get("state") == "blocked":
        raise ValueError(f"Plan is blocked: {plan.get('gate', {}).get('reason')}")
    if hash_payload(plan["request"]) != plan.get("request_hash"):
        raise ValueError("Request hash mismatch")


def _require_eccodes() -> Any:
    try:
        import eccodes
    except ImportError as exc:
        raise CfsAtmosphericError("eccodes is required for native NCEI GRIB2") from exc
    return eccodes


def runtime_preflight() -> dict[str, Any]:
    imports: dict[str, Any] = {}
    for name in ("numpy", "netCDF4", "requests", "rasterio", "xarray", "eccodes"):
        try:
            module = __import__(name)
            imports[name] = {"available": True, "version": str(getattr(module, "__version__", "unknown"))}
        except Exception as exc:
            imports[name] = {"available": False, "error": str(exc)[:500]}
    raster = imports["rasterio"].get("available", False)
    if raster:
        try:
            import rasterio
            with rasterio.Env() as environment:
                imports["rasterio"]["grib_driver"] = "GRIB" in environment.drivers()
        except Exception as exc:
            imports["rasterio"]["grib_driver"] = False
            imports["rasterio"]["driver_error"] = str(exc)[:500]
    passed = all(value.get("available") for value in imports.values()) and bool(imports["rasterio"].get("grib_driver"))
    return {"schema_version": "cfs_grib_runtime_preflight_v1", "passed": passed, "python": sys.version, "executable": str(Path(sys.executable).resolve()), "imports": imports, "checked_utc": utc_now()}


def _codes_get(eccodes: Any, gid: Any, key: str, default: Any = None) -> Any:
    try:
        return eccodes.codes_get(gid, key)
    except Exception:
        return default


def _message_metadata(eccodes: Any, gid: Any) -> dict[str, Any]:
    date = int(_codes_get(eccodes, gid, "validityDate", 0) or 0)
    clock = int(_codes_get(eccodes, gid, "validityTime", 0) or 0)
    valid = datetime.strptime(f"{date:08d}{clock:04d}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    short = str(_codes_get(eccodes, gid, "shortName", "")).strip()
    name = str(_codes_get(eccodes, gid, "name", _codes_get(eccodes, gid, "parameterName", ""))).strip()
    pdt = _codes_get(eccodes, gid, "productDefinitionTemplateNumber", -1)
    return {
        "valid_time": time_text(valid),
        "short_name": short,
        "name": name,
        "units": str(_codes_get(eccodes, gid, "units", "")).strip(),
        "type_of_level": str(_codes_get(eccodes, gid, "typeOfLevel", "")).strip(),
        "level": float(_codes_get(eccodes, gid, "level", 0.0) or 0.0),
        "product_definition_template": int(-1 if pdt is None else pdt),
        "forecast_time": float(_codes_get(eccodes, gid, "forecastTime", 0.0) or 0.0),
        "step_range": str(_codes_get(eccodes, gid, "stepRange", "")),
        "data_date": int(_codes_get(eccodes, gid, "dataDate", 0) or 0),
        "data_time": int(_codes_get(eccodes, gid, "dataTime", 0) or 0),
    }


def _match_field(meta: Mapping[str, Any], spec: Mapping[str, Any]) -> bool:
    short = str(meta["short_name"]).lower()
    name = str(meta["name"]).lower()
    level_type = str(meta["type_of_level"]).lower()
    parameter = short in spec["short"] or any(text in name for text in spec["contains"])
    level_ok = level_type in spec["level"]
    value_ok = "value" not in spec or abs(float(meta["level"]) - float(spec["value"])) < 1e-6
    return parameter and level_ok and value_ok


def _regular_grid(eccodes: Any, gid: Any, decoded_values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ni = int(eccodes.codes_get(gid, "Ni"))
    nj = int(eccodes.codes_get(gid, "Nj"))
    values = np.asarray(eccodes.codes_get_array(gid, "values") if decoded_values is None else decoded_values, dtype=np.float64).reshape(-1)
    lats = np.asarray(eccodes.codes_get_array(gid, "latitudes"), dtype=np.float64)
    lons = np.mod(np.asarray(eccodes.codes_get_array(gid, "longitudes"), dtype=np.float64), 360.0)
    if not (values.size == lats.size == lons.size == ni * nj):
        raise CfsAtmosphericError("GRIB grid dimensions do not match decoded arrays")
    for rows, cols, transpose in ((nj, ni, False), (ni, nj, True)):
        lat2, lon2, data = lats.reshape(rows, cols), lons.reshape(rows, cols), values.reshape(rows, cols)
        if transpose:
            lat2, lon2, data = lat2.T, lon2.T, data.T
        if np.allclose(lat2, lat2[:, :1], atol=1e-7) and np.allclose(lon2, lon2[:1, :], atol=1e-7):
            return lat2[:, 0], lon2[0, :], data
    raise CfsAtmosphericError("NCEI GRIB grid is not separable latitude/longitude")


def _decode_values(eccodes: Any, gid: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        return _regular_grid(eccodes, gid)
    except Exception as first:
        try:
            import rasterio
            from rasterio.io import MemoryFile
            payload = eccodes.codes_get_message(gid)
            with MemoryFile(payload) as memory:
                with memory.open() as raster:
                    values = raster.read(1)
            return _regular_grid(eccodes, gid, values)
        except Exception as second:
            raise CfsAtmosphericError(f"Neither ecCodes nor rasterio decoded GRIB values: {first}; {second}") from second


def _bbox_indices(axis: np.ndarray, low: float, high: float, halo: int) -> np.ndarray:
    found = np.flatnonzero(np.isfinite(axis) & (axis >= low) & (axis <= high))
    if found.size == 0:
        raise CfsAtmosphericError("Requested bbox lies outside source coverage")
    return np.arange(max(0, int(found.min()) - halo), min(axis.size, int(found.max()) + halo + 1))


def _convert(values: np.ndarray, meta: Mapping[str, Any], canonical: str) -> tuple[np.ndarray, str]:
    units = str(meta["units"]).lower().replace(" ", "")
    data = np.asarray(values, dtype=np.float64)
    if canonical == "absolute_air_pressure":
        if units in {"pa", "pascal", "pascals"}:
            return data, "identity_Pa"
        if units in {"hpa", "mb", "mbar"}:
            return data * 100.0, "hPa_times_100_to_Pa"
        raise CfsAtmosphericError(f"Unsupported pressure units {meta['units']!r}")
    if canonical in {"air_temperature_2m", "surface_temperature"}:
        if units in {"k", "kelvin"}:
            return data - 273.15, "K_minus_273.15_to_degree_Celsius"
        if "c" in units:
            return data, "identity_degree_Celsius"
        raise CfsAtmosphericError(f"Unsupported temperature units {meta['units']!r}")
    return data, "identity"


def _message_at_or_after(client: requests.Session, url: str, offset: int, total: int, retries: int) -> tuple[int, int, bytes]:
    header = _request(client, "GET", url, retries=retries, headers={"Range": f"bytes={offset}-{min(total - 1, offset + 15)}"}).content
    if len(header) < 16 or header[:4] != b"GRIB" or header[7] != 2:
        search = _request(client, "GET", url, retries=retries, headers={"Range": f"bytes={offset}-{min(total - 1, offset + 4 * 1024 * 1024 - 1)}"}).content
        found = search.find(b"GRIB")
        if found < 0:
            raise EOFError
        offset += found
        header = search[found:found + 16]
    length = int.from_bytes(header[8:16], "big")
    if length <= 16 or offset + length > total:
        raise CfsAtmosphericError(f"Invalid GRIB2 message length {length} at byte {offset}")
    response = _request(client, "GET", url, retries=retries, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
    if len(response.content) != length:
        raise CfsAtmosphericError("NCEI range response did not contain one complete GRIB message")
    return offset, offset + length, response.content


def _next_message(client: requests.Session, url: str, offset: int, total: int, retries: int) -> tuple[int, bytes]:
    _start, next_offset, payload = _message_at_or_after(client, url, offset, total, retries)
    return next_offset, payload


def _decode_message_bytes(payload: bytes) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    eccodes = _require_eccodes()
    gid = eccodes.codes_new_from_message(payload)
    if gid is None:
        raise CfsAtmosphericError("ecCodes did not find a GRIB message")
    try:
        meta = _message_metadata(eccodes, gid)
        lat, lon, values = _decode_values(eccodes, gid)
    finally:
        eccodes.codes_release(gid)
    return meta, lat, lon, values


def _decode_message_metadata_bytes(payload: bytes) -> dict[str, Any]:
    eccodes = _require_eccodes()
    gid = eccodes.codes_new_from_message(payload)
    if gid is None:
        raise CfsAtmosphericError("ecCodes did not find a GRIB message")
    try:
        return _message_metadata(eccodes, gid)
    finally:
        eccodes.codes_release(gid)


def _snapshot_messages(unit: Mapping[str, Any], target: datetime, specs: Sequence[Mapping[str, Any]], request: Mapping[str, Any], raw_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Path]]:
    client = requests.Session()
    offset = 0
    selected: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Path]] = {}
    target_text = time_text(target)
    month_start = datetime(target.year, target.month, 1, tzinfo=timezone.utc)
    if target > month_start:
        low, high, best = 0, int(unit["bytes"]) - 16, 0
        for _ in range(24):
            if high - low < 1024 * 1024:
                break
            probe = (low + high) // 2
            try:
                message_start, next_offset, payload = _message_at_or_after(client, str(unit["url"]), probe, int(unit["bytes"]), int(request["max_retries"]))
            except EOFError:
                high = probe
                continue
            meta = _decode_message_metadata_bytes(payload)
            if parse_utc(meta["valid_time"]) <= target:
                best = message_start
                low = max(next_offset, probe + 1)
            else:
                high = probe
        # Back up far enough to include paired vector components at the same valid time.
        offset = max(0, best - 8 * 1024 * 1024)
    max_messages = 192
    try:
        for index in range(max_messages):
            try:
                next_offset, payload = _next_message(client, str(unit["url"]), offset, int(unit["bytes"]), int(request["max_retries"]))
            except EOFError:
                break
            meta, lat, lon, values = _decode_message_bytes(payload)
            offset = next_offset
            if meta["valid_time"] > target_text and selected:
                break
            if meta["valid_time"] != target_text:
                continue
            for spec in specs:
                if _match_field(meta, spec):
                    candidate_score = (int(lat.size * lon.size), -float(meta["forecast_time"]))
                    prior = selected.get(spec["name"])
                    prior_score = (int(prior[0].size * prior[1].size), -float(prior[3]["forecast_time"])) if prior else None
                    if prior_score is not None and candidate_score <= prior_score:
                        continue
                    path = raw_dir / f"{unit['month']}_{unit['product']}_{spec['name']}.grib2"
                    temporary = path.with_suffix(".part")
                    temporary.write_bytes(payload)
                    os.replace(temporary, path)
                    selected[spec["name"]] = (lat, lon, values, meta, path)
    finally:
        client.close()
    missing = [spec["name"] for spec in specs if spec["name"] not in selected]
    if missing:
        raise CfsAtmosphericError(f"No exact {target_text} GRIB message found in {unit['id']} for {', '.join(missing)}")
    return selected


def _write_canonical(path: Path, times: Sequence[datetime], latitude: np.ndarray, longitude: np.ndarray, fields: Mapping[str, np.ndarray], metadata: Mapping[str, Mapping[str, Any]], attrs: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with nc4.Dataset(temporary, "w", format="NETCDF4") as dataset:
        dataset.createDimension("time", len(times))
        dataset.createDimension("latitude", int(latitude.size))
        dataset.createDimension("longitude", int(longitude.size))
        tv = dataset.createVariable("time", "i8", ("time",))
        tv.units, tv.calendar = "seconds since 1970-01-01 00:00:00 UTC", "gregorian"
        tv[:] = np.asarray([int(value.timestamp()) for value in times], dtype=np.int64)
        yv = dataset.createVariable("latitude", "f8", ("latitude",)); yv.units = "degrees_north"; yv[:] = latitude
        xv = dataset.createVariable("longitude", "f8", ("longitude",)); xv.units = "degrees_east"; xv[:] = longitude
        for name, values in fields.items():
            variable = dataset.createVariable(name, "f4", ("time", "latitude", "longitude"), zlib=True, complevel=2, fill_value=np.float32(9.96921e36))
            variable[:] = np.asarray(values, dtype=np.float32)
            detail = dict(metadata[name])
            variable.units = str(detail.pop("canonical_units"))
            variable.long_name = name.replace("_", " ")
            variable.source_grib_metadata = canonical_json(detail)
        dataset.schema_version = SCHEMA_FIELDS
        dataset.Conventions = "CF-1.10"
        for key, value in attrs.items():
            setattr(dataset, str(key), canonical_json(value) if isinstance(value, (dict, list)) else str(value))
    os.replace(temporary, path)


def fetch_ncei_snapshot(model: str, plan: Mapping[str, Any], run_dir: Path, destination: Path, status: DownloadStatus) -> None:
    request = plan["request"]
    target = parse_utc(request["start"])
    if target != parse_utc(request["end"]):
        raise CfsAtmosphericError("snapshot mode requires start == end")
    raw_dir = run_dir / "raw_messages"
    raw_dir.mkdir(parents=True, exist_ok=True)
    units = {row["product"]: row for row in plan["source_units"] if row["month"] == target.strftime("%Y%m")}
    fields: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    canonical_lat: np.ndarray | None = None
    canonical_lon: np.ndarray | None = None
    for product in request["products"]:
        status.update(message=f"Reading ranged GRIB snapshot for {product}", active_chunk=product)
        selected = _snapshot_messages(units[product], target, PRODUCTS[product]["fields"], request, raw_dir)
        for name, (lat, lon, values, meta, raw_path) in selected.items():
            yi = _bbox_indices(lat, request["bbox"][1], request["bbox"][3], int(request["halo_cells"]))
            xi = _bbox_indices(lon, request["bbox"][0], request["bbox"][2], int(request["halo_cells"]))
            local_lat, local_lon = lat[yi], lon[xi]
            if canonical_lat is None:
                canonical_lat, canonical_lon = local_lat, local_lon
            elif not (np.allclose(canonical_lat, local_lat) and np.allclose(canonical_lon, local_lon)):
                raise CfsAtmosphericError("Requested NCEI products do not share one native grid")
            converted, conversion = _convert(values[np.ix_(yi, xi)], meta, name)
            fields[name] = converted[None, :, :]
            spec = next(field for field in PRODUCTS[product]["fields"] if field["name"] == name)
            metadata[name] = {**meta, "canonical_units": spec["units"], "conversion": conversion, "source_file": Path(str(units[product]["url"])).name, "raw_message": raw_path.name, "raw_sha256": sha256_file(raw_path)}
    assert canonical_lat is not None and canonical_lon is not None
    _write_canonical(destination, [target], canonical_lat, canonical_lon, fields, metadata, {"connector": MODEL_CONFIG[model]["connector"], "model": model, "source_provider": "ncei", "request_hash": plan["request_hash"], "provider_lock": "whole_request", "request_bbox_0_360": request["bbox"]})


def _download_unit(unit: Mapping[str, Any], raw_dir: Path, request: Mapping[str, Any], status: DownloadStatus) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / Path(str(unit["url"])).name
    if destination.exists() and destination.stat().st_size == int(unit["bytes"]):
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    client = requests.Session()
    try:
        response = _request(client, "GET", str(unit["url"]), retries=int(request["max_retries"]), headers=headers, stream=True, timeout=(30, 300))
        if start and response.status_code != 206:
            partial.unlink(missing_ok=True)
            start = 0
            response.close()
            response = _request(client, "GET", str(unit["url"]), retries=int(request["max_retries"]), stream=True, timeout=(30, 300))
        mode = "ab" if start else "wb"
        completed = start
        with partial.open(mode) as stream:
            for block in response.iter_content(1024 * 1024):
                if block:
                    stream.write(block)
                    completed += len(block)
                    status.update(completed_bytes=int(status.data.get("completed_bytes", 0)) + len(block), active_chunk=unit["id"])
        if completed != int(unit["bytes"]):
            raise CfsAtmosphericError(f"Downloaded {completed} bytes for {unit['id']}; expected {unit['bytes']}")
        os.replace(partial, destination)
        return destination
    finally:
        client.close()


def _iter_grib(path: Path) -> Iterable[tuple[Any, Any]]:
    eccodes = _require_eccodes()
    stream = path.open("rb")
    try:
        while True:
            gid = eccodes.codes_grib_new_from_file(stream)
            if gid is None:
                break
            try:
                yield eccodes, gid
            finally:
                eccodes.codes_release(gid)
    finally:
        stream.close()


def fetch_ncei_full(model: str, plan: Mapping[str, Any], run_dir: Path, destination: Path, status: DownloadStatus, cleanup_raw: bool = False) -> None:
    request = plan["request"]
    raw_dir = run_dir / "raw"
    downloaded: dict[str, Path] = {}
    for unit in plan["source_units"]:
        status.update(message=f"Downloading {unit['id']}", active_chunk=unit["id"])
        downloaded[unit["id"]] = _download_unit(unit, raw_dir, request, status)
    source_hashes = {unit_id: sha256_file(path) for unit_id, path in downloaded.items()}
    start, end = parse_utc(request["start"]), parse_utc(request["end"])
    records: dict[str, dict[datetime, tuple[tuple[int, float], np.ndarray, dict[str, Any], np.ndarray, np.ndarray]]] = defaultdict(dict)
    for unit in plan["source_units"]:
        specs = PRODUCTS[unit["product"]]["fields"]
        for eccodes, gid in _iter_grib(downloaded[unit["id"]]):
            meta = _message_metadata(eccodes, gid)
            valid = parse_utc(meta["valid_time"])
            if valid < start or valid > end:
                continue
            matched = next((spec for spec in specs if _match_field(meta, spec)), None)
            if matched is None:
                continue
            lat, lon, values = _decode_values(eccodes, gid)
            yi = _bbox_indices(lat, request["bbox"][1], request["bbox"][3], int(request["halo_cells"]))
            xi = _bbox_indices(lon, request["bbox"][0], request["bbox"][2], int(request["halo_cells"]))
            local_lat, local_lon = lat[yi], lon[xi]
            data, conversion = _convert(values[np.ix_(yi, xi)], meta, matched["name"])
            lead = float(meta["forecast_time"])
            prior = records[matched["name"]].get(valid)
            detail = {**meta, "canonical_units": matched["units"], "conversion": conversion, "source_file": downloaded[unit["id"]].name, "source_sha256": source_hashes[unit["id"]]}
            score = (int(lat.size * lon.size), -lead)
            if prior is None or score > prior[0]:
                records[matched["name"]][valid] = (score, data, detail, local_lat, local_lon)
    times = [start]
    while times[-1] < end:
        times.append(datetime.fromtimestamp(times[-1].timestamp() + 3600, tz=timezone.utc))
    fields: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    canonical_lat: np.ndarray | None = None
    canonical_lon: np.ndarray | None = None
    for product in request["products"]:
        for spec in PRODUCTS[product]["fields"]:
            missing = [time_text(value) for value in times if value not in records[spec["name"]]]
            if missing:
                raise CfsAtmosphericError(f"Missing exact hourly {spec['name']} records: {', '.join(missing[:8])}")
            chosen = [records[spec["name"]][value] for value in times]
            for record in chosen:
                local_lat, local_lon = record[3], record[4]
                if canonical_lat is None:
                    canonical_lat, canonical_lon = local_lat, local_lon
                elif not (np.allclose(canonical_lat, local_lat) and np.allclose(canonical_lon, local_lon)):
                    raise CfsAtmosphericError("Requested NCEI products do not share one dominant native grid")
            fields[spec["name"]] = np.stack([record[1] for record in chosen])
            metadata[spec["name"]] = chosen[0][2]
    assert canonical_lat is not None and canonical_lon is not None
    _write_canonical(destination, times, canonical_lat, canonical_lon, fields, metadata, {"connector": MODEL_CONFIG[model]["connector"], "model": model, "source_provider": "ncei", "request_hash": plan["request_hash"], "provider_lock": "whole_request", "request_bbox_0_360": request["bbox"]})
    if cleanup_raw:
        for path in downloaded.values():
            path.unlink(missing_ok=True)


def _mt_epoch(values: np.ndarray) -> np.ndarray:
    epoch = np.datetime64("1900-12-31T00:00:00", "s")
    return (epoch + np.rint(np.asarray(values, dtype=np.float64) * 86400.0).astype("timedelta64[s]")).astype("datetime64[s]").astype(np.int64)


def _open_hycom(model: str, year: int, product: str, request: Mapping[str, Any]) -> tuple[Any, str]:
    try:
        import xarray as xr
    except ImportError as exc:
        raise CfsAtmosphericError("xarray is required for HYCOM fallback") from exc
    mapping = HYCOM_MAP[model][product]
    explicit = request.get("hycom_sources", {}).get(product)
    candidates = [explicit.format(year=year)] if explicit else _hycom_candidates(model, year, mapping["subdataset"])
    errors: list[str] = []
    for candidate in candidates:
        try:
            return xr.open_dataset(candidate, engine="netcdf4", decode_times=False, cache=False), public_url(candidate)
        except Exception as exc:
            errors.append(f"{Path(candidate).name}: {str(exc)[:200]}")
    raise CfsAtmosphericError("No HYCOM source opened: " + " | ".join(errors))


def fetch_hycom(model: str, plan: Mapping[str, Any], run_dir: Path, destination: Path, status: DownloadStatus) -> None:
    request = plan["request"]
    start, end = parse_utc(request["start"]), parse_utc(request["end"])
    start_s, end_s = int(start.timestamp()), int(end.timestamp())
    pieces: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}
    canonical_lat: np.ndarray | None = None
    canonical_lon: np.ndarray | None = None
    source_urls: set[str] = set()
    for product in request["products"]:
        mapping = HYCOM_MAP[model][product]
        for year in range(start.year, end.year + 1):
            dataset, url = _open_hycom(model, year, product, request)
            source_urls.add(url)
            try:
                epoch = _mt_epoch(dataset["MT"].values)
                ti = np.flatnonzero((epoch >= start_s) & (epoch <= end_s))
                if not ti.size:
                    continue
                lat = np.asarray(dataset["Latitude"].values, dtype=float)
                lon = np.mod(np.asarray(dataset["Longitude"].values, dtype=float), 360.0)
                yi = _bbox_indices(lat, request["bbox"][1], request["bbox"][3], int(request["halo_cells"]))
                xi = _bbox_indices(lon, request["bbox"][0], request["bbox"][2], int(request["halo_cells"]))
                local_lat, local_lon = lat[yi], lon[xi]
                if canonical_lat is None:
                    canonical_lat, canonical_lon = local_lat, local_lon
                elif not (np.allclose(canonical_lat, local_lat) and np.allclose(canonical_lon, local_lon)):
                    raise CfsAtmosphericError("HYCOM fallback products do not share one native grid")
                for canonical, (source_name, conversion) in mapping["fields"].items():
                    values = np.asarray(dataset[source_name].isel(MT=ti, Latitude=yi, Longitude=xi).values, dtype=np.float64)
                    units = str(dataset[source_name].attrs.get("units", ""))
                    if conversion == "pressure_departure":
                        values = (values + 1000.0) * 100.0 if units.lower() not in {"pa", "pascal", "pascals"} else values + 100000.0
                    elif conversion == "temperature" and units.lower() in {"k", "kelvin"}:
                        values -= 273.15
                    pieces[canonical].append((epoch[ti], values))
                    spec = next(field for field in PRODUCTS[product]["fields"] if field["name"] == canonical)
                    metadata[canonical] = {"canonical_units": spec["units"], "source_variable": source_name, "source_units": units, "conversion": conversion, "source_url": url, "product_definition_template": "not_applicable_hycom"}
            finally:
                dataset.close()
    expected = np.arange(start_s, end_s + 1, 3600, dtype=np.int64)
    fields: dict[str, np.ndarray] = {}
    for name, parts in pieces.items():
        epochs = np.concatenate([part[0] for part in parts])
        values = np.concatenate([part[1] for part in parts], axis=0)
        order = np.argsort(epochs)
        epochs, values = epochs[order], values[order]
        unique, index = np.unique(epochs, return_index=True)
        epochs, values = unique, values[index]
        if not np.array_equal(epochs, expected):
            raise CfsAtmosphericError(f"HYCOM fallback did not provide the exact hourly axis for {name}")
        fields[name] = values
    required = [field["name"] for product in request["products"] for field in PRODUCTS[product]["fields"]]
    if sorted(fields) != sorted(required):
        raise CfsAtmosphericError("HYCOM fallback did not provide every canonical field")
    assert canonical_lat is not None and canonical_lon is not None
    times = [datetime.fromtimestamp(int(value), tz=timezone.utc) for value in expected]
    _write_canonical(destination, times, canonical_lat, canonical_lon, fields, metadata, {"connector": MODEL_CONFIG[model]["connector"], "model": model, "source_provider": "hycom", "source_urls": sorted(source_urls), "request_hash": plan["request_hash"], "provider_lock": "whole_request", "request_bbox_0_360": request["bbox"]})


def health(path: str | Path, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source = Path(path)
    checks: dict[str, bool] = {"exists": source.exists(), "nonempty": source.exists() and source.stat().st_size > 0}
    dimensions: dict[str, int] = {}
    variables: list[str] = []
    provider = "unknown"
    if checks["nonempty"]:
        try:
            with nc4.Dataset(source) as dataset:
                checks["schema"] = str(getattr(dataset, "schema_version", "")) == SCHEMA_FIELDS
                dimensions = {name: len(dimension) for name, dimension in dataset.dimensions.items()}
                provider = str(getattr(dataset, "source_provider", "unknown"))
                variables = [name for name in dataset.variables if name not in {"time", "latitude", "longitude"}]
                checks["coordinate_dimensions"] = all(dimensions.get(name, 0) > 0 for name in ("time", "latitude", "longitude"))
                checks["finite_fields"] = all(np.all(np.isfinite(np.ma.filled(dataset[name][:], np.nan))) for name in variables)
                if request:
                    model = str(request.get("model") or "cfsv2")
                    normalized = normalize_request(request, model)
                    required = [field["name"] for product in normalized["products"] for field in PRODUCTS[product]["fields"]]
                    checks["requested_variables"] = all(name in variables for name in required)
                    expected = int((parse_utc(normalized["end"]) - parse_utc(normalized["start"])).total_seconds() // 3600) + 1
                    checks["exact_time_count"] = dimensions.get("time") == expected
        except Exception:
            checks["readable"] = False
        else:
            checks["readable"] = True
    payload = {"schema_version": "cfs_atmospheric_health_v2", "path": str(source.resolve()) if source.exists() else str(source), "passed": bool(checks) and all(checks.values()), "source_provider": provider, "dimensions": dimensions, "variables": variables, "checks": checks, "checked_utc": utc_now()}
    if source.exists():
        payload.update({"bytes": source.stat().st_size, "sha256": sha256_file(source)})
    return payload


def fetch_plan(model: str, plan: Mapping[str, Any], run_dir: str | Path, *, output: str | Path | None = None, open_monitor: bool = True, cleanup_raw: bool = False) -> dict[str, Any]:
    validate_plan(plan)
    if plan.get("model") != model:
        raise ValueError(f"Plan belongs to {plan.get('model')}, not {model}")
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = Path(output) if output else directory / Path(plan["request"]["output"]).name
    if not destination.is_absolute():
        destination = directory / destination.name
    provider = str(plan["provider"])
    atomic_write_json(directory / "source_provider_lock.json", {"schema_version": "cfs_source_provider_lock_v2", "provider": provider, "request_hash": plan["request_hash"], "mixed_providers": False, "locked_utc": utc_now(), "provider_attempts": plan.get("provider_attempts", [])})
    estimate = float(plan["duration_estimate_seconds"]["conservative"])
    status = DownloadStatus(directory / "download_status.json", connector=MODEL_CONFIG[model]["connector"], provider=provider, request_hash=plan["request_hash"], total_chunks=len(plan.get("source_units", [])) or len(plan["request"]["products"]), expected_bytes=int(plan["raw_transfer_bytes"]), estimate_seconds=estimate, expected_hours=int(plan["expected_hours"]), completed_bytes=0, decoded_hours=0, artifacts={"output": destination.name, "health": "health_check.json", "monitor": "download_monitor.html"})
    write_monitor_html(directory)
    monitor = launch_monitor(directory, open_browser=open_monitor) if estimate >= 600 else {"launched": False, "reason": "below_threshold", "html": str(directory / "download_monitor.html")}
    status.start()
    try:
        if provider == "ncei" and plan["mode"] == "snapshot":
            fetch_ncei_snapshot(model, plan, directory, destination, status)
        elif provider == "ncei":
            fetch_ncei_full(model, plan, directory, destination, status, cleanup_raw=cleanup_raw)
        else:
            fetch_hycom(model, plan, directory, destination, status)
        report = health(destination, plan["request"])
        atomic_write_json(directory / "health_check.json", report)
        if not report["passed"]:
            raise CfsAtmosphericError("Canonical output failed health validation")
        status.update(completed_bytes=int(plan["raw_transfer_bytes"]), completed_chunks=len(plan.get("source_units", [])) or len(plan["request"]["products"]), decoded_hours=int(plan["expected_hours"]), message="Health checks passed")
        status.finish("complete", "Output published and health-checked")
        return {"schema_version": "cfs_atmospheric_fetch_result_v2", "model": model, "provider": provider, "output": str(destination.resolve()), "sha256": report["sha256"], "health": str((directory / "health_check.json").resolve()), "status": str((directory / "download_status.json").resolve()), "monitor": monitor, "request_hash": plan["request_hash"]}
    except Exception as exc:
        status.finish("failed", str(exc))
        raise


def _sibling_script(current_script: Path, sibling_model: str) -> Path:
    skill_root = current_script.resolve().parent.parent
    skills_root = skill_root.parent
    sibling_name = MODEL_CONFIG[sibling_model]["connector"]
    candidate = skills_root / sibling_name / "scripts" / f"{sibling_model}_fetcher.py"
    if not candidate.exists():
        raise CfsAtmosphericError(f"Sibling skill entry point is unavailable: {candidate}")
    return candidate


def _invoke_child(script: Path, request: Mapping[str, Any], run_dir: Path, *, snapshot: bool) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "routed_request.json"
    atomic_write_json(request_path, dict(request))
    command = [sys.executable, str(script), "run", "--request", str(request_path), "--run-dir", str(run_dir), "--no-route"]
    if snapshot:
        command.append("--snapshot")
    process = subprocess.run(command, capture_output=True, text=True, timeout=7200)
    result_path = run_dir / "run_result.json"
    if process.returncode or not result_path.exists():
        raise CfsAtmosphericError(f"Routed child failed ({process.returncode}): {(process.stderr or process.stdout)[-1000:]}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def execute_request(model: str, payload: Mapping[str, Any], run_dir: str | Path, current_script: str | Path, *, snapshot: bool = False, no_route: bool = False, open_monitor: bool = True) -> dict[str, Any]:
    request = normalize_request(payload, model)
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    segments = classify_era(parse_utc(request["start"]), parse_utc(request["end"]))
    if no_route:
        if len(segments) != 1 or segments[0][0] != model:
            raise CfsAtmosphericError("Routing recursion guard rejected an out-of-era child request")
        plan = build_plan(model, request, directory, mode="snapshot" if snapshot else "fetch")
        result = fetch_plan(model, plan, directory, open_monitor=open_monitor)
        atomic_write_json(directory / "run_result.json", result)
        return result
    if len(segments) == 1 and segments[0][0] == model:
        plan = build_plan(model, request, directory, mode="snapshot" if snapshot else "fetch")
        result = fetch_plan(model, plan, directory, open_monitor=open_monitor)
        atomic_write_json(directory / "run_result.json", result)
        return result
    if int(request.get("routing_depth", 0)) > 0:
        raise CfsAtmosphericError("Routing depth exceeded; refusing a sibling loop")
    stem = Path(request["output"]).stem
    children: list[dict[str, Any]] = []
    for child_model, start, end in segments:
        child_dir = directory / child_model
        child = dict(request)
        child.update({"model": child_model, "start": time_text(start), "end": time_text(end), "output": f"{stem}_{child_model}.nc", "routing_depth": 1})
        if child_model != model:
            child.update({
                "catalog_root": MODEL_CONFIG[child_model]["catalog_root"],
                "month_catalog_template": MODEL_CONFIG[child_model]["month_catalog"],
                "ncei_file_template": MODEL_CONFIG[child_model]["file"],
                "hycom_sources": {},
            })
        if child_model == model:
            child_plan = build_plan(model, child, child_dir, mode="snapshot" if snapshot else "fetch")
            result = fetch_plan(model, child_plan, child_dir, open_monitor=open_monitor)
            atomic_write_json(child_dir / "run_result.json", result)
        else:
            existing_path = child_dir / "run_result.json"
            expected_hash = hash_payload(normalize_request(child, child_model))
            result = json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.exists() else None
            if not result or result.get("request_hash") != expected_hash or not Path(str(result.get("output", ""))).exists():
                result = _invoke_child(_sibling_script(Path(current_script), child_model), child, child_dir, snapshot=snapshot)
        children.append({"model": child_model, "start": child["start"], "end": child["end"], "request_hash": result["request_hash"], "provider": result["provider"], "output": result["output"], "sha256": result["sha256"], "health": result["health"]})
    if len(children) == 1:
        result = {**children[0], "schema_version": "cfs_atmospheric_fetch_result_v2", "routed_by": model}
        atomic_write_json(directory / "run_result.json", result)
        return result
    manifest = {"schema_version": SCHEMA_ROUTING, "created_utc": utc_now(), "parent_model": model, "parent_request_hash": hash_payload(request), "no_regridding_or_concatenation": True, "segments": children}
    manifest["manifest_hash"] = hash_payload(manifest)
    path = directory / f"{stem}_routing_manifest.json"
    atomic_write_json(path, manifest)
    result = {"schema_version": "cfs_atmospheric_fetch_result_v2", "model": "cfs-family", "provider": "segmented", "output": str(path.resolve()), "sha256": sha256_file(path), "health": [child["health"] for child in children], "request_hash": manifest["parent_request_hash"], "segments": children}
    atomic_write_json(directory / "run_result.json", result)
    return result


def legacy_cross_era_warning() -> None:
    warnings.warn("A cross-era legacy call returns a routing-manifest Path; migrate to cfs_atmospheric_request_v2.", DeprecationWarning, stacklevel=2)


__all__ = [
    "CfsAtmosphericError", "ERA_END", "ERA_SPLIT", "ERA_START", "HYCOM_MAP", "MODEL_CONFIG", "PRODUCTS",
    "SCHEMA_FIELDS", "SCHEMA_PLAN", "SCHEMA_REQUEST", "SCHEMA_ROUTING", "build_plan", "classify_era",
    "discover_available_months", "execute_request", "fetch_plan", "health", "hycom_eligibility", "legacy_cross_era_warning",
    "month_keys", "ncei_inventory", "normalize_request", "parse_utc", "runtime_preflight", "sha256_file", "validate_in_era", "validate_plan",
]
