#!/usr/bin/env python3
"""Provider-neutral HRRR inventory, ranged transfer, decode, and health core."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
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
from typing import Any
from urllib.parse import urlsplit, urlunsplit

try:
    from .download_monitor import DownloadStatus, atomic_write_json, utc_now
except ImportError:
    from download_monitor import DownloadStatus, atomic_write_json, utc_now


SCHEMA_REQUEST = "hrrr_request_v1"
SCHEMA_INVENTORY = "hrrr_inventory_v1"
SCHEMA_PLAN = "hrrr_download_plan_v1"
SCHEMA_FIELDS = "hrrr_fields_v1"
SCHEMA_MANIFEST = "hrrr_run_manifest_v1"
SCHEMA_HEALTH = "hrrr_health_report_v1"

PROVIDERS = {
    "aws": "https://noaa-hrrr-bdp-pds.s3.amazonaws.com",
    "gcp": "https://storage.googleapis.com/high-resolution-rapid-refresh",
    "azure": "https://noaahrrr.blob.core.windows.net/hrrr",
    "nomads": "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod",
}
DEFAULT_PROVIDERS = ["aws", "gcp", "azure", "nomads"]
FAMILIES = {"wrfsfc", "wrfprs", "wrfnat", "wrfsubh"}
ECCODES_SHORT_NAMES = {
    "APCP": {"tp", "apcp"},
    "DLWRF": {"sdlwrf", "dlwrf"},
    "DPT": {"d", "2d", "dpt"},
    "DSWRF": {"sdswrf", "dswrf"},
    "GUST": {"gust"},
    "LHTFL": {"slhtf", "lhtfl"},
    "PRES": {"sp", "pres"},
    "PRATE": {"prate"},
    "SHTFL": {"ishf", "shtfl"},
    "SPFH": {"q", "2sh", "spfh"},
    "TMP": {"t", "2t", "tmp"},
    "UGRD": {"u", "10u", "80u", "ugrd"},
    "VGRD": {"v", "10v", "80v", "vgrd"},
}
REQUIRED_RUNTIME = {
    "eccodes": "2.47.0",
    "netCDF4": "1.7.4",
    "numpy": "2.5.1",
    "pyproj": "3.7.2",
    "rasterio": "1.5.1",
    "requests": "2.34.2",
    "xarray": "2026.7.0",
}
DOMAIN_INFO = {
    "conus": {
        "archive_start": "2014-07-30T18:00:00Z",
        "analysis_step_hours": 1,
        "nx": 1799,
        "ny": 1059,
        "grid_template": 30,
        "suffix": "",
    },
    "alaska": {
        "archive_start": "2018-07-11T18:00:00Z",
        "analysis_step_hours": 3,
        "nx": 1299,
        "ny": 919,
        "grid_template": 20,
        "suffix": ".ak",
    },
}


def _field(
    output_name: str,
    short_name: str,
    level_text: str | None = None,
    *,
    level_contains: str | None = None,
    step_type: str = "instant",
    family: str = "wrfsfc",
    component: str | None = None,
    vector_group: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "output_name": output_name,
        "short_name": short_name,
        "family": family,
        "step_type": step_type,
        "canonical_alias": True,
    }
    if level_text is not None:
        row["level_text"] = level_text
    if level_contains is not None:
        row["level_contains"] = level_contains
    if component:
        row["component"] = component
    if vector_group:
        row["vector_group"] = vector_group
        row["rotate_to_earth"] = True
    return row


ALIASES: dict[str, list[dict[str, Any]]] = {
    "wind_10m": [
        _field("eastward_wind_10m", "UGRD", "10 m above ground", component="u", vector_group="wind_10m"),
        _field("northward_wind_10m", "VGRD", "10 m above ground", component="v", vector_group="wind_10m"),
    ],
    "wind_80m": [
        _field("eastward_wind_80m", "UGRD", "80 m above ground", component="u", vector_group="wind_80m"),
        _field("northward_wind_80m", "VGRD", "80 m above ground", component="v", vector_group="wind_80m"),
    ],
    "surface_pressure": [_field("surface_air_pressure", "PRES", "surface")],
    "mean_sea_level_pressure": [_field("air_pressure_at_mean_sea_level", "MSLMA", "mean sea level")],
    "air_temperature_2m": [_field("air_temperature_2m", "TMP", "2 m above ground")],
    "specific_humidity_2m": [_field("specific_humidity_2m", "SPFH", "2 m above ground")],
    "dew_point_temperature_2m": [_field("dew_point_temperature_2m", "DPT", "2 m above ground")],
    "surface_temperature": [_field("surface_temperature", "TMP", "surface")],
    "precipitation_rate": [_field("precipitation_rate", "PRATE", "surface")],
    "total_precipitation": [_field("total_precipitation", "APCP", "surface", step_type="accum")],
    "downward_shortwave_flux": [_field("surface_downwelling_shortwave_flux", "DSWRF", "surface")],
    "downward_longwave_flux": [_field("surface_downwelling_longwave_flux", "DLWRF", "surface")],
    "sensible_heat_flux": [_field("surface_sensible_heat_flux", "SHTFL", "surface")],
    "latent_heat_flux": [_field("surface_latent_heat_flux", "LHTFL", "surface")],
    "wind_gust": [_field("wind_speed_of_gust", "GUST", "surface")],
    "visibility": [_field("visibility_in_air", "VIS", "surface")],
    "total_cloud_cover": [_field("cloud_area_fraction", "TCDC", "entire atmosphere")],
    "precipitable_water": [_field("atmosphere_mass_content_of_water_vapor", "PWAT", level_contains="entire atmosphere")],
    "composite_reflectivity": [_field("composite_reflectivity", "REFC", "entire atmosphere")],
    "aerosol_optical_thickness": [_field("atmosphere_optical_thickness_due_to_ambient_aerosol", "AOTK", level_contains="entire atmosphere")],
    "column_smoke_mass_density": [_field("column_smoke_mass_density", "COLMD", level_contains="entire atmosphere")],
}
CF_STANDARD_NAMES = {
    "air_pressure_at_mean_sea_level": "air_pressure_at_mean_sea_level",
    "air_temperature_2m": "air_temperature",
    "atmosphere_mass_content_of_water_vapor": "atmosphere_mass_content_of_water_vapor",
    "cloud_area_fraction": "cloud_area_fraction",
    "dew_point_temperature_2m": "dew_point_temperature",
    "precipitation_rate": "precipitation_flux",
    "specific_humidity_2m": "specific_humidity",
    "surface_air_pressure": "surface_air_pressure",
    "surface_downwelling_longwave_flux": "surface_downwelling_longwave_flux_in_air",
    "surface_downwelling_shortwave_flux": "surface_downwelling_shortwave_flux_in_air",
    "surface_temperature": "surface_temperature",
    "visibility_in_air": "visibility_in_air",
    "wind_speed_of_gust": "wind_speed_of_gust",
}


class HrrrError(RuntimeError):
    """Machine-actionable HRRR workflow failure."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO UTC timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must include UTC timezone: {value!r}")
    return parsed.astimezone(timezone.utc).replace(microsecond=0)


def time_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def public_url(value: str) -> str:
    parsed = urlsplit(str(value))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def parse_period(value: str) -> int:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", str(value).upper())
    if not match or (match.group(1) is None and match.group(2) is None):
        raise ValueError(f"Forecast period must be ISO-8601 PT#H/PT#M: {value!r}")
    minutes = int(match.group(1) or 0) * 60 + int(match.group(2) or 0)
    if minutes < 0:
        raise ValueError("Forecast period cannot be negative")
    return minutes


def period_text(minutes: int) -> str:
    hours, remainder = divmod(int(minutes), 60)
    if remainder == 0:
        return f"PT{hours}H"
    if hours == 0:
        return f"PT{remainder}M"
    return f"PT{hours}H{remainder}M"


def _hours(start: datetime, end: datetime, step_hours: int) -> list[datetime]:
    if end < start:
        raise ValueError("End must not precede start")
    rows: list[datetime] = []
    value = start
    while value <= end:
        rows.append(value)
        value += timedelta(hours=step_hours)
    if rows[-1] != end:
        raise ValueError("Time range is not divisible by the requested cadence")
    return rows


def hrrr_version(value: datetime) -> str:
    gates = [
        (parse_utc("2020-12-02T00:00:00Z"), "v4"),
        (parse_utc("2018-07-12T00:00:00Z"), "v3"),
        (parse_utc("2016-08-23T00:00:00Z"), "v2"),
        (parse_utc("2014-09-30T00:00:00Z"), "v1"),
    ]
    for gate, label in gates:
        if value >= gate:
            return label
    return "pre-operational"


def _normalize_lon(value: float) -> float:
    lon = ((float(value) + 180.0) % 360.0) - 180.0
    return 180.0 if lon == -180.0 and float(value) > 0 else lon


def normalize_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version", SCHEMA_REQUEST) != SCHEMA_REQUEST:
        raise ValueError(f"Expected {SCHEMA_REQUEST}")
    domain = str(payload.get("domain", "conus")).lower()
    if domain not in DOMAIN_INFO:
        raise ValueError("domain must be conus or alaska")
    mode = str(payload.get("mode", "analysis")).lower()
    if mode not in {"analysis", "forecast"}:
        raise ValueError("mode must be analysis or forecast")
    products = payload.get("products")
    if not isinstance(products, list) or not products:
        raise ValueError("products must be a non-empty list")
    provider_override = payload.get("provider_override")
    providers = (
        [str(provider_override).lower()]
        if provider_override is not None
        else [str(item).lower() for item in payload.get("provider_order", DEFAULT_PROVIDERS)]
    )
    if not providers or len(providers) != len(set(providers)) or any(item not in PROVIDERS for item in providers):
        raise ValueError(f"provider_order must contain unique names from {sorted(PROVIDERS)}")
    longitude_convention = str(payload.get("longitude_convention", "-180_180")).lower()
    if longitude_convention not in {"-180_180", "0_360"}:
        raise ValueError("longitude_convention must be -180_180 or 0_360")
    bbox_input = payload.get("bbox", [-180.0, -90.0, 180.0, 90.0])
    if not isinstance(bbox_input, list) or len(bbox_input) != 4:
        raise ValueError("bbox must be [west,south,east,north]")
    raw_west, raw_east = float(bbox_input[0]), float(bbox_input[2])
    if abs(raw_east - raw_west) >= 360.0 - 1e-9:
        west, east = -180.0, 180.0
    else:
        west, east = _normalize_lon(raw_west), _normalize_lon(raw_east)
    bbox = [west, float(bbox_input[1]), east, float(bbox_input[3])]
    if not (-90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError("bbox latitude bounds are invalid")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_REQUEST,
        "domain": domain,
        "mode": mode,
        "products": json.loads(json.dumps(products)),
        "bbox": bbox,
        "longitude_convention": longitude_convention,
        "halo_cells": int(payload.get("halo_cells", 1)),
        "missing_policy": str(payload.get("missing_policy", "error")).lower(),
        "retain_raw_messages": bool(payload.get("retain_raw_messages", True)),
        "provider_order": providers,
        "connect_timeout_seconds": float(payload.get("connect_timeout_seconds", 15.0)),
        "read_timeout_seconds": float(payload.get("read_timeout_seconds", 90.0)),
        "max_retries": int(payload.get("max_retries", 3)),
    }
    if result["halo_cells"] < 0:
        raise ValueError("halo_cells cannot be negative")
    if result["missing_policy"] not in {"error", "skip"}:
        raise ValueError("missing_policy must be error or skip")
    if result["connect_timeout_seconds"] <= 0 or result["read_timeout_seconds"] <= 0:
        raise ValueError("network timeouts must be positive")
    if result["max_retries"] < 0:
        raise ValueError("max_retries cannot be negative")
    if mode == "analysis":
        start = parse_utc(str(payload["start"]))
        end = parse_utc(str(payload["end"]))
        step = int(payload.get("step_hours", DOMAIN_INFO[domain]["analysis_step_hours"]))
        required = int(DOMAIN_INFO[domain]["analysis_step_hours"])
        if step <= 0 or step % required:
            raise ValueError(f"{domain} analysis step_hours must be a positive multiple of {required}")
        if start.minute or start.second or end.minute or end.second:
            raise ValueError("Analysis times must be exact UTC hours")
        if domain == "alaska" and (start.hour % 3 or end.hour % 3):
            raise ValueError("Alaska analysis cycles must be 00,03,06,09,12,15,18,21 UTC")
        if start < parse_utc(str(DOMAIN_INFO[domain]["archive_start"])):
            raise ValueError(f"{domain} HRRR archive begins {DOMAIN_INFO[domain]['archive_start']}")
        _hours(start, end, step)
        result.update({"start": time_text(start), "end": time_text(end), "step_hours": step})
    else:
        start = parse_utc(str(payload["cycle_start"]))
        end = parse_utc(str(payload.get("cycle_end", payload["cycle_start"])))
        step = int(payload.get("cycle_step_hours", DOMAIN_INFO[domain]["analysis_step_hours"]))
        required = int(DOMAIN_INFO[domain]["analysis_step_hours"])
        if step <= 0 or step % required:
            raise ValueError(f"{domain} cycle_step_hours must be a positive multiple of {required}")
        if start.minute or start.second or end.minute or end.second:
            raise ValueError("Forecast cycles must be exact UTC hours")
        cycles = _hours(start, end, step)
        if domain == "alaska" and any(value.hour % 3 for value in cycles):
            raise ValueError("Alaska forecast cycles occur every three hours")
        if start < parse_utc(str(DOMAIN_INFO[domain]["archive_start"])):
            raise ValueError(f"{domain} HRRR archive begins {DOMAIN_INFO[domain]['archive_start']}")
        periods = sorted({parse_period(str(item)) for item in payload.get("forecast_periods", [])})
        if not periods:
            raise ValueError("forecast mode requires forecast_periods")
        result.update({
            "cycle_start": time_text(start),
            "cycle_end": time_text(end),
            "cycle_step_hours": step,
            "forecast_periods": [period_text(item) for item in periods],
        })
    _expand_products(result["products"])
    return result


def _level_text(type_of_level: str, value: Any) -> str:
    number = float(value)
    shown = str(int(number)) if number.is_integer() else str(number)
    mapping = {
        "isobaricinhpa": f"{shown} mb",
        "heightaboveground": f"{shown} m above ground",
        "hybrid": f"{shown} hybrid level",
        "surface": "surface",
        "meansea": "mean sea level",
    }
    key = str(type_of_level).replace("_", "").lower()
    if key not in mapping:
        raise ValueError(f"Use level_text for unsupported type_of_level {type_of_level!r}")
    return mapping[key]


def _safe_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    if not name or name[0].isdigit():
        name = f"field_{name}"
    return name


def _expand_products(products: Sequence[Any]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for product in products:
        if isinstance(product, str):
            key = product.lower()
            if key not in ALIASES:
                raise ValueError(f"Unknown HRRR product alias {product!r}; use products command or an exact selector")
            expanded.extend(json.loads(json.dumps(ALIASES[key])))
            continue
        if not isinstance(product, Mapping):
            raise ValueError("Each product must be an alias string or selector object")
        if "alias" in product:
            key = str(product["alias"]).lower()
            if key not in ALIASES:
                raise ValueError(f"Unknown HRRR product alias {product['alias']!r}")
            family_override = product.get("family")
            if family_override is not None and str(family_override).lower() not in FAMILIES:
                raise ValueError(f"Alias family override must be one of {sorted(FAMILIES)}")
            rows = json.loads(json.dumps(ALIASES[key]))
            for row in rows:
                if family_override is not None:
                    row["family"] = str(family_override).lower()
                if "output_name" in product:
                    suffix = "_u" if row.get("component") == "u" else "_v" if row.get("component") == "v" else ""
                    row["output_name"] = _safe_name(str(product["output_name"])) + suffix
            expanded.extend(rows)
            continue
        family = str(product.get("family", "")).lower()
        short = str(product.get("short_name", "")).upper()
        if family not in FAMILIES or not short:
            raise ValueError("Exact selectors require family and short_name")
        levels = product.get("levels")
        if levels is None:
            levels = [product.get("level")] if "level" in product else [None]
        if not isinstance(levels, list) or not levels:
            raise ValueError("levels must be a non-empty list")
        base_name = _safe_name(str(product.get("output_name", short)))
        for level in levels:
            row = {
                "family": family,
                "short_name": short,
                "output_name": base_name,
                "step_type": str(product.get("step_type", "instant")).lower(),
                "canonical_alias": False,
            }
            if short in {"UGRD", "VGRD"}:
                if not product.get("vector_group"):
                    raise ValueError("Exact UGRD/VGRD selectors require vector_group and must be requested as a pair")
                row["component"] = "u" if short == "UGRD" else "v"
                row["vector_group"] = str(product["vector_group"])
                row["rotate_to_earth"] = False
            if row["step_type"] not in {"instant", "accum", "avg", "max", "min"}:
                raise ValueError("step_type must be instant, accum, avg, max, or min")
            if "level_text" in product:
                row["level_text"] = str(product["level_text"])
            elif "type_of_level" in product:
                row["type_of_level"] = str(product["type_of_level"])
                row["level_value"] = level
                row["level_text"] = _level_text(row["type_of_level"], level)
            else:
                raise ValueError("Exact selectors require level_text or type_of_level with level/levels")
            if "step_contains" in product:
                row["step_contains"] = str(product["step_contains"])
            if "canonical_units" in product:
                raise ValueError("hrrr-fetcher preserves GRIB units; perform unit conversion downstream")
            expanded.append(row)
    identifiers: set[str] = set()
    source_selectors: set[tuple[str, str, str, str, str]] = set()
    for index, row in enumerate(expanded):
        row["selector_id"] = f"s{index:04d}_{row['output_name']}_{_safe_name(str(row.get('level_text', '')))}"
        if row["selector_id"] in identifiers:
            raise ValueError(f"Duplicate selector {row['selector_id']}")
        identifiers.add(row["selector_id"])
        source_selector = (
            str(row["family"]), str(row["short_name"]), str(row.get("level_text", "")),
            str(row.get("step_type", "instant")), str(row.get("step_contains", "")),
        )
        if source_selector in source_selectors:
            raise ValueError(f"Duplicate source selector {source_selector}")
        source_selectors.add(source_selector)
    vector_components: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in expanded:
        if row.get("vector_group"):
            key = (str(row["family"]), str(row["vector_group"]), str(row.get("level_text", "")))
            vector_components[key].add(str(row["component"]))
    incomplete = [key for key, components in vector_components.items() if components != {"u", "v"}]
    if incomplete:
        raise ValueError(f"U/V selectors must be paired at every family and level: {incomplete}")
    return expanded


def product_catalog() -> dict[str, Any]:
    return {
        "schema_version": "hrrr_product_catalog_v1",
        "providers": PROVIDERS,
        "default_provider_order": DEFAULT_PROVIDERS,
        "families": {
            "wrfsfc": {"description": "2-D surface and diagnostic fields", "current_record_guide": {"conus": 170, "alaska": 169}},
            "wrfprs": {"description": "3-D isobaric fields", "pressure_levels": 39, "range_hpa": [50, 1000]},
            "wrfnat": {"description": "3-D native hybrid and inline-smoke fields", "hybrid_levels": 50},
            "wrfsubh": {"description": "15-minute selected surface/diagnostic fields", "maximum_hours": 18},
        },
        "domains": DOMAIN_INFO,
        "aliases": ALIASES,
        "bufr": {"documented": True, "decoded": False},
    }


def object_key(domain: str, cycle: datetime, family: str, lead_hour: int) -> str:
    suffix = str(DOMAIN_INFO[domain]["suffix"])
    return f"hrrr.{cycle:%Y%m%d}/{domain}/hrrr.t{cycle:%H}z.{family}f{lead_hour:02d}{suffix}.grib2"


def provider_url(provider: str, key: str) -> str:
    return f"{PROVIDERS[provider]}/{key}"


def _period_step(minutes: int) -> str:
    if minutes == 0:
        return "anl"
    if minutes % 60 == 0:
        hours = minutes // 60
        return f"{hours} hour fcst"
    return f"{minutes} min fcst"


def _requirements(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    domain = str(request["domain"])
    specs = _expand_products(request["products"])
    groups: dict[tuple[str, str, int], dict[str, Any]] = {}
    if request["mode"] == "analysis":
        cycles = _hours(parse_utc(str(request["start"])), parse_utc(str(request["end"])), int(request["step_hours"]))
        periods = [0]
    else:
        cycles = _hours(parse_utc(str(request["cycle_start"])), parse_utc(str(request["cycle_end"])), int(request["cycle_step_hours"]))
        periods = [parse_period(item) for item in request["forecast_periods"]]
    for cycle in cycles:
        for minutes in periods:
            for spec in specs:
                family = str(spec["family"])
                if family == "wrfsubh":
                    if request["mode"] == "analysis":
                        raise ValueError("wrfsubh is forecast-only; use wrfsfc f00 for strict analyses")
                    if minutes % 15 or minutes > 18 * 60:
                        raise ValueError("wrfsubh periods must be 15-minute multiples through PT18H")
                    lead = int(math.ceil(minutes / 60.0)) if minutes else 0
                else:
                    if minutes % 60:
                        if request["mode"] == "forecast":
                            continue
                        raise ValueError(f"{family} requires whole-hour periods")
                    lead = minutes // 60
                    maximum = 48 if cycle.hour in {0, 6, 12, 18} else 18
                    if lead > maximum:
                        raise ValueError(f"Cycle {time_text(cycle)} supports {family} only through f{maximum:02d}")
                key = object_key(domain, cycle, family, lead)
                group_key = (time_text(cycle), family, lead)
                group = groups.setdefault(group_key, {
                    "id": f"{time_text(cycle)}_{family}_f{lead:02d}",
                    "key": key,
                    "domain": domain,
                    "family": family,
                    "cycle": time_text(cycle),
                    "lead_hour": lead,
                    "cadence_group": "subhourly" if family == "wrfsubh" else "hourly",
                    "hrrr_version": hrrr_version(cycle),
                    "targets": [],
                })
                target = dict(spec)
                target["forecast_period_minutes"] = minutes
                target["forecast_period"] = period_text(minutes)
                target["expected_step"] = "anl" if minutes == 0 else f"{minutes} min fcst" if family == "wrfsubh" else _period_step(minutes)
                group["targets"].append(target)
    if not groups:
        raise ValueError(
            "No source objects match the request. Use wrfsubh aliases/selectors for subhourly periods "
            "and hourly families for whole-hour periods."
        )
    return sorted(groups.values(), key=lambda row: (row["cycle"], row["family"], row["lead_hour"]))


def parse_idx(text: str, total_bytes: int) -> list[dict[str, Any]]:
    if int(total_bytes) <= 0:
        raise HrrrError("HRRR object size must be positive")
    rows: list[dict[str, Any]] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 6:
            raise HrrrError(f"Malformed HRRR idx line: {line[:200]}")
        try:
            number, offset = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise HrrrError(f"Malformed HRRR idx offset: {line[:200]}") from exc
        rows.append({
            "message_number": number,
            "offset": offset,
            "reference": parts[2],
            "short_name": parts[3],
            "level_text": parts[4],
            "step": parts[5],
            "extra": ":".join(parts[6:]).strip(":"),
            "idx_line": line,
        })
    if not rows:
        raise HrrrError("HRRR idx was empty")
    if int(rows[0]["offset"]) != 0:
        raise HrrrError("HRRR idx first offset is not zero")
    if len({int(row["message_number"]) for row in rows}) != len(rows):
        raise HrrrError("HRRR idx message numbers are not unique")
    if any(rows[index]["offset"] >= rows[index + 1]["offset"] for index in range(len(rows) - 1)):
        raise HrrrError("HRRR idx offsets are not strictly increasing")
    for index, row in enumerate(rows):
        end = rows[index + 1]["offset"] - 1 if index + 1 < len(rows) else int(total_bytes) - 1
        if end < int(row["offset"]):
            raise HrrrError("HRRR idx range extends beyond the object size")
        row["end"] = end
        row["bytes"] = end - int(row["offset"]) + 1
    return rows


def _step_matches(row: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    step = str(row["step"]).lower().strip()
    if target.get("step_contains"):
        return str(target["step_contains"]).lower() in step
    kind = str(target.get("step_type", "instant"))
    minutes = int(target["forecast_period_minutes"])
    tokens = {"accum": "acc", "avg": "ave", "max": "max", "min": "min"}
    if kind == "instant":
        return step == str(target["expected_step"]).lower()
    if tokens[kind] not in step:
        return False
    if minutes == 0:
        return step.startswith("0-0")
    if minutes % 60 == 0:
        return bool(re.search(rf"(?:^|-)\s*{minutes // 60}\s+hour\s+{tokens[kind]}\s+fcst$", step))
    return bool(re.search(rf"(?:^|-)\s*{minutes}\s+min\s+{tokens[kind]}\s+fcst$", step))


def select_idx(rows: Sequence[Mapping[str, Any]], target: Mapping[str, Any]) -> dict[str, Any]:
    matches = []
    for row in rows:
        if str(row["short_name"]).upper() != str(target["short_name"]).upper():
            continue
        level = str(row["level_text"]).lower()
        if target.get("level_text") is not None and level != str(target["level_text"]).lower():
            continue
        if target.get("level_contains") is not None and str(target["level_contains"]).lower() not in level:
            continue
        if not _step_matches(row, target):
            continue
        matches.append(dict(row))
    if not matches:
        raise HrrrError(
            f"No exact idx field for {target['short_name']}:{target.get('level_text', target.get('level_contains'))}:"
            f"{target['expected_step']} ({target.get('step_type')})"
        )
    if len(matches) > 1:
        raise HrrrError(f"Ambiguous idx field for selector {target['selector_id']}: {len(matches)} matches")
    return matches[0]


def _request(session: Any, method: str, url: str, request: Mapping[str, Any], **kwargs: Any) -> Any:
    timeout = (float(request["connect_timeout_seconds"]), float(request["read_timeout_seconds"]))
    last: Exception | None = None
    for attempt in range(int(request["max_retries"]) + 1):
        response = None
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            status_code = getattr(response, "status_code", None)
            if response is not None:
                response.close()
            last = exc
            if status_code is not None and 400 <= int(status_code) < 500 and int(status_code) not in {408, 429}:
                break
            if attempt >= int(request["max_retries"]):
                break
            time.sleep(min(0.5 * (2**attempt), 4.0))
    raise HrrrError(f"{method} failed for {public_url(url)}: {last}") from last


def probe_provider(session: Any, provider: str, requirement: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    url = provider_url(provider, str(requirement["key"]))
    idx_url = url + ".idx"
    idx_response = _request(session, "GET", idx_url, request)
    range_response = _request(session, "GET", url, request, headers={"Range": "bytes=0-0"})
    try:
        if range_response.status_code != 206:
            raise HrrrError(f"Provider {provider} ignored HTTP Range for {requirement['key']}")
        content_range = range_response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes 0-0/(\d+)", content_range)
        if not match:
            raise HrrrError(f"Provider {provider} returned invalid Content-Range {content_range!r}")
        total = int(match.group(1))
        idx_payload = idx_response.content
        idx_text = idx_response.text
        rows = parse_idx(idx_text, total)
        etag = range_response.headers.get("ETag")
    finally:
        idx_response.close()
        range_response.close()
    selected_by_message: dict[int, dict[str, Any]] = {}
    for target in requirement["targets"]:
        found = select_idx(rows, target)
        number = int(found["message_number"])
        selected = selected_by_message.setdefault(number, {**found, "targets": []})
        selected["targets"].append(dict(target))
    selected_rows = sorted(selected_by_message.values(), key=lambda row: int(row["offset"]))
    signature_body = [
        {
            "message_number": row["message_number"],
            "offset": row["offset"],
            "end": row["end"],
            "short_name": row["short_name"],
            "level_text": row["level_text"],
            "step": row["step"],
        }
        for row in selected_rows
    ]
    return {
        "provider": provider,
        "url": public_url(url),
        "idx_url": public_url(idx_url),
        "total_bytes": total,
        "idx_records": len(rows),
        "selected_messages": selected_rows,
        "selected_bytes": sum(int(row["bytes"]) for row in selected_rows),
        "selected_signature": hash_payload(signature_body),
        "idx_sha256": hashlib.sha256(idx_payload).hexdigest(),
        "idx_text": idx_text,
        "etag": etag,
    }


def build_inventory(payload: Mapping[str, Any], *, session: Any | None = None) -> dict[str, Any]:
    import requests

    request = normalize_request(payload)
    requirements = _requirements(request)
    client = session or requests.Session()
    close = session is None
    objects: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    try:
        for requirement in requirements:
            locked: dict[str, Any] | None = None
            for provider in request["provider_order"]:
                started = time.monotonic()
                try:
                    probe = probe_provider(client, provider, requirement, request)
                    attempts.append({
                        "object_id": requirement["id"],
                        "provider": provider,
                        "available": True,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "url": probe["url"],
                    })
                    locked = {**dict(requirement), **probe, "provider_lock": provider}
                    break
                except Exception as exc:
                    attempts.append({
                        "object_id": requirement["id"],
                        "provider": provider,
                        "available": False,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "reason": str(exc)[:1000],
                        "url": provider_url(provider, str(requirement["key"])),
                    })
            if locked is None:
                gaps.append({"object_id": requirement["id"], "key": requirement["key"], "targets": requirement["targets"]})
            else:
                objects.append(locked)
    finally:
        if close:
            client.close()
    selected_bytes = sum(int(item["selected_bytes"]) for item in objects)
    return {
        "schema_version": SCHEMA_INVENTORY,
        "request": request,
        "request_hash": hash_payload(request),
        "created_utc": utc_now(),
        "objects": objects,
        "gaps": gaps,
        "provider_attempts": attempts,
        "selected_bytes": selected_bytes,
        "object_count": len(objects),
        "message_count": sum(len(item["selected_messages"]) for item in objects),
        "upper_cycle_discovered": max((item["cycle"] for item in objects), default=None),
        "pre_operational": any(item["hrrr_version"] == "pre-operational" for item in objects),
    }


def build_plan(
    payload: Mapping[str, Any],
    run_dir: str | Path,
    *,
    session: Any | None = None,
    free_bytes_override: int | None = None,
) -> dict[str, Any]:
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(payload, session=session)
    request = inventory["request"]
    raw = int(inventory["selected_bytes"])
    info = DOMAIN_INFO[str(request["domain"])]
    full_grid_decoded = int(info["nx"]) * int(info["ny"]) * 8 * max(1, int(inventory["message_count"]))
    calculated = raw * 2 + full_grid_decoded * 2 + 64 * 1024 * 1024
    required = max(raw * 4, calculated)
    free = int(free_bytes_override) if free_bytes_override is not None else int(shutil.disk_usage(directory).free)
    reasons: list[str] = []
    if inventory["gaps"] and request["missing_policy"] == "error":
        reasons.append(f"{len(inventory['gaps'])} required HRRR objects are unavailable from every provider")
    if not inventory["objects"]:
        reasons.append("no required HRRR objects are available; missing_policy=skip cannot publish an empty dataset")
    if free < required:
        reasons.append(f"free workspace {free} bytes is below required {required} bytes")
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_PLAN,
        "request": request,
        "request_hash": inventory["request_hash"],
        "inventory": inventory,
        "created_utc": utc_now(),
        "run_dir_name": directory.name,
        "transfer_bytes": raw,
        "full_grid_scratch_bytes": full_grid_decoded,
        "required_free_bytes": required,
        "available_free_bytes": free,
        "conservative_seconds": round(raw / (5 * 1024 * 1024) + inventory["object_count"] * 2.0, 1),
        "gate": {"state": "blocked" if reasons else "ready", "reasons": reasons},
    }
    plan["plan_hash"] = hash_payload(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != SCHEMA_PLAN:
        raise ValueError(f"Expected {SCHEMA_PLAN}")
    body = dict(plan)
    stored = str(body.pop("plan_hash", ""))
    if stored != hash_payload(body):
        raise ValueError("Plan hash mismatch")
    if hash_payload(plan["request"]) != plan.get("request_hash"):
        raise ValueError("Plan request hash mismatch")
    if plan.get("gate", {}).get("state") != "ready":
        raise ValueError(f"Plan is blocked: {plan.get('gate', {}).get('reasons')}")


def runtime_preflight() -> dict[str, Any]:
    imports: dict[str, Any] = {}
    for name, expected in REQUIRED_RUNTIME.items():
        try:
            module = __import__(name)
            version = str(getattr(module, "__version__", "unknown"))
            imports[name] = {"available": True, "version": version, "expected_version": expected, "version_match": version == expected}
        except Exception as exc:
            imports[name] = {"available": False, "expected_version": expected, "version_match": False, "error": str(exc)[:500]}
    if imports.get("rasterio", {}).get("available"):
        try:
            import rasterio

            with rasterio.Env() as environment:
                imports["rasterio"]["grib_driver"] = "GRIB" in environment.drivers()
        except Exception as exc:
            imports["rasterio"]["grib_driver"] = False
            imports["rasterio"]["driver_error"] = str(exc)[:500]
    passed = all(item.get("available") and item.get("version_match") for item in imports.values()) and bool(imports.get("rasterio", {}).get("grib_driver"))
    return {
        "schema_version": "hrrr_grib_runtime_preflight_v1",
        "passed": passed,
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "imports": imports,
        "checked_utc": utc_now(),
    }


def _atomic_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(20):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))


def _valid_grib(path: Path, expected_bytes: int | None = None) -> bool:
    try:
        size = path.stat().st_size
        if expected_bytes is not None and size != int(expected_bytes):
            return False
        if size < 20:
            return False
        with path.open("rb") as stream:
            header = stream.read(16)
            stream.seek(-4, os.SEEK_END)
            ending = stream.read(4)
        return (
            header[:4] == b"GRIB"
            and header[7] == 2
            and int.from_bytes(header[8:16], "big") == size
            and ending == b"7777"
        )
    except OSError:
        return False


def _coalesce_messages(messages: Sequence[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    """Coalesce selected messages only when their byte ranges are exactly adjacent."""
    groups: list[list[dict[str, Any]]] = []
    for message in sorted((dict(item) for item in messages), key=lambda row: int(row["offset"])):
        if groups and int(groups[-1][-1]["end"]) + 1 == int(message["offset"]):
            groups[-1].append(message)
        else:
            groups.append([message])
    return groups


def _download_range(
    session: Any,
    url: str,
    start: int,
    end: int,
    destination: Path,
    request: Mapping[str, Any],
    status: DownloadStatus,
    *,
    require_single_grib: bool = True,
) -> dict[str, Any]:
    expected = int(end) - int(start) + 1
    complete = _valid_grib(destination, expected) if require_single_grib else destination.exists() and destination.stat().st_size == expected
    if complete:
        return {"path": destination, "bytes": expected, "sha256": sha256_file(destination), "resumed": True}
    part = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if part.exists() and part.stat().st_size > expected:
        part.unlink()
    retries = 0
    for attempt in range(int(request["max_retries"]) + 1):
        completed = part.stat().st_size if part.exists() else 0
        if completed == expected:
            break
        request_start = int(start) + completed
        headers = {"Range": f"bytes={request_start}-{int(end)}"}
        try:
            response = session.get(
                url,
                headers=headers,
                timeout=(float(request["connect_timeout_seconds"]), float(request["read_timeout_seconds"])),
                stream=True,
            )
            response.raise_for_status()
            wanted = f"bytes {request_start}-{int(end)}/"
            if response.status_code != 206 or not str(response.headers.get("Content-Range", "")).startswith(wanted):
                response.close()
                raise HrrrError(f"Invalid ranged response for {public_url(url)}: {response.status_code} {response.headers.get('Content-Range')}")
            with part.open("ab") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        stream.write(chunk)
            response.close()
        except Exception as exc:
            retries += 1
            status.update(retries=int(status.data.get("retries", 0)) + 1, message=f"Range retry {attempt + 1}: {exc}")
            if attempt >= int(request["max_retries"]):
                raise HrrrError(f"Range download failed for {public_url(url)} bytes {start}-{end}: {exc}") from exc
            time.sleep(min(0.5 * (2**attempt), 4.0))
    if part.stat().st_size != expected:
        raise HrrrError(f"Ranged message has {part.stat().st_size} bytes; expected {expected}")
    _atomic_replace(part, destination)
    if require_single_grib and not _valid_grib(destination, expected):
        raise HrrrError(f"Downloaded range is not one complete GRIB2 message: {destination.name}")
    return {"path": destination, "bytes": expected, "sha256": sha256_file(destination), "resumed": retries > 0}


def _download_object(
    session: Any,
    item: Mapping[str, Any],
    request: Mapping[str, Any],
    raw_root: Path,
    status: DownloadStatus,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    object_hash = hashlib.sha256(str(item["key"]).encode("utf-8")).hexdigest()[:16]
    final_dir = raw_root / object_hash
    state_path = final_dir / "object.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("selected_signature") == item.get("selected_signature") and all(
                _valid_grib(final_dir / row["file"], int(row["bytes"])) for row in state.get("messages", [])
            ):
                rows = [{**row, "path": str(final_dir / row["file"])} for row in state["messages"]]
                return state, rows, [{"provider": state["provider"], "state": "resumed_complete"}]
        except (OSError, json.JSONDecodeError):
            pass
    provider_order = [str(item["provider_lock"])] + [name for name in request["provider_order"] if name != item["provider_lock"]]
    provider_attempts: list[dict[str, Any]] = []
    last: Exception | None = None
    for provider in provider_order:
        started = time.monotonic()
        try:
            probe = dict(item) if provider == item["provider_lock"] else probe_provider(session, provider, item, request)
            if int(probe["total_bytes"]) != int(item["total_bytes"]) or probe["selected_signature"] != item["selected_signature"]:
                raise HrrrError("Mirror object size or normalized selected-index signature differs")
            stage = raw_root / ".staging" / object_hash / provider
            stage.mkdir(parents=True, exist_ok=True)
            downloaded: list[dict[str, Any]] = []
            groups = _coalesce_messages(probe["selected_messages"])
            for position, group in enumerate(groups, start=1):
                first, last_message = group[0], group[-1]
                cached_paths = [stage / f"message_{int(message['message_number']):04d}.grib2" for message in group]
                if all(_valid_grib(path, int(message["bytes"])) for path, message in zip(cached_paths, group, strict=True)):
                    range_digest = hashlib.sha256()
                    for path in cached_paths:
                        with path.open("rb") as stream:
                            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                                range_digest.update(chunk)
                    for path, message in zip(cached_paths, group, strict=True):
                        downloaded.append({
                            "file": path.name,
                            "message_number": int(message["message_number"]),
                            "offset": int(message["offset"]),
                            "end": int(message["end"]),
                            "bytes": int(message["bytes"]),
                            "sha256": sha256_file(path),
                            "range_sha256": range_digest.hexdigest(),
                            "short_name": message["short_name"],
                            "level_text": message["level_text"],
                            "step": message["step"],
                            "targets": message["targets"],
                        })
                    status.update(message=f"Resumed completed {provider} object range {position}/{len(groups)}")
                    continue
                span_path = stage / f"range_{int(first['message_number']):04d}_{int(last_message['message_number']):04d}.partset"
                status.update(
                    active_chunk=f"{item['id']}:{first['message_number']}-{last_message['message_number']}",
                    message=f"Downloading {provider} object range {position}/{len(groups)}",
                )
                transfer = _download_range(
                    session,
                    str(probe["url"]),
                    int(first["offset"]),
                    int(last_message["end"]),
                    span_path,
                    request,
                    status,
                    require_single_grib=len(group) == 1,
                )
                span_payload = span_path.read_bytes()
                for message in group:
                    filename = f"message_{int(message['message_number']):04d}.grib2"
                    path = stage / filename
                    relative_start = int(message["offset"]) - int(first["offset"])
                    relative_end = int(message["end"]) - int(first["offset"]) + 1
                    if path != span_path:
                        temporary = path.with_suffix(path.suffix + ".part")
                        temporary.write_bytes(span_payload[relative_start:relative_end])
                        _atomic_replace(temporary, path)
                    if not _valid_grib(path, int(message["bytes"])):
                        raise HrrrError(f"Coalesced range did not frame GRIB message {message['message_number']}")
                    downloaded.append({
                        "file": filename,
                        "message_number": int(message["message_number"]),
                        "offset": int(message["offset"]),
                        "end": int(message["end"]),
                        "bytes": int(message["bytes"]),
                        "sha256": sha256_file(path),
                        "range_sha256": transfer["sha256"],
                        "short_name": message["short_name"],
                        "level_text": message["level_text"],
                        "step": message["step"],
                        "targets": message["targets"],
                    })
                span_path.unlink()
            idx_path = stage / "selected.idx"
            idx_path.write_text("\n".join(str(row["idx_line"]) for row in probe["selected_messages"]) + "\n", encoding="utf-8", newline="\n")
            (stage / "source.idx").write_text(str(probe["idx_text"]), encoding="utf-8", newline="\n")
            state = {
                "schema_version": "hrrr_object_lock_v1",
                "object_id": item["id"],
                "key": item["key"],
                "provider": provider,
                "url": probe["url"],
                "idx_url": probe["idx_url"],
                "total_bytes": probe["total_bytes"],
                "selected_signature": probe["selected_signature"],
                "idx_sha256": probe["idx_sha256"],
                "messages": downloaded,
                "completed_utc": utc_now(),
            }
            atomic_write_json(stage / "object.json", state)
            if final_dir.exists():
                stale = raw_root / ".staging" / object_hash / f"stale_{int(time.time())}"
                _atomic_replace(final_dir, stale)
            _atomic_replace(stage, final_dir)
            provider_attempts.append({"provider": provider, "state": "complete", "elapsed_seconds": round(time.monotonic() - started, 3), "url": probe["url"]})
            rows = [{**row, "path": str(final_dir / row["file"])} for row in downloaded]
            return state, rows, provider_attempts
        except Exception as exc:
            last = exc
            provider_attempts.append({
                "provider": provider,
                "state": "failed",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "reason": str(exc)[:1000],
                "url": provider_url(provider, str(item["key"])),
            })
            status.update(failed_chunks=int(status.data.get("failed_chunks", 0)) + 1, message=f"Provider {provider} failed for {item['id']}: {exc}")
    raise HrrrError(f"Every provider failed while fetching {item['key']}: {last}") from last


def _codes_get(eccodes: Any, gid: Any, key: str, default: Any = None) -> Any:
    try:
        return eccodes.codes_get(gid, key)
    except Exception:
        return default


def _codes_int(eccodes: Any, gid: Any, key: str, default: int = -1) -> int:
    value = _codes_get(eccodes, gid, key, default)
    return int(default if value is None else value)


def _metadata(path: Path) -> dict[str, Any]:
    import eccodes

    payload = path.read_bytes()
    gid = eccodes.codes_new_from_message(payload)
    if gid is None:
        raise HrrrError(f"ecCodes did not decode {path.name}")
    try:
        date = int(_codes_get(eccodes, gid, "validityDate", 0) or 0)
        clock = int(_codes_get(eccodes, gid, "validityTime", 0) or 0)
        valid = datetime.strptime(f"{date:08d}{clock:04d}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        data_date = int(_codes_get(eccodes, gid, "dataDate", 0) or 0)
        data_time = int(_codes_get(eccodes, gid, "dataTime", 0) or 0)
        reference = datetime.strptime(f"{data_date:08d}{data_time:04d}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        pdt = _codes_get(eccodes, gid, "productDefinitionTemplateNumber", -1)
        nx = _codes_get(eccodes, gid, "Nx", _codes_get(eccodes, gid, "Ni", 0))
        ny = _codes_get(eccodes, gid, "Ny", _codes_get(eccodes, gid, "Nj", 0))
        return {
            "valid_time": time_text(valid),
            "forecast_reference_time": time_text(reference),
            "short_name": str(_codes_get(eccodes, gid, "shortName", "")),
            "name": str(_codes_get(eccodes, gid, "name", "")),
            "units": str(_codes_get(eccodes, gid, "units", "")),
            "type_of_level": str(_codes_get(eccodes, gid, "typeOfLevel", "")),
            "level": float(_codes_get(eccodes, gid, "level", 0.0) or 0.0),
            "step_type": str(_codes_get(eccodes, gid, "stepType", "")),
            "step_range": str(_codes_get(eccodes, gid, "stepRange", "")),
            "forecast_time": float(_codes_get(eccodes, gid, "forecastTime", 0.0) or 0.0),
            "product_definition_template": int(-1 if pdt is None else pdt),
            "data_date": data_date,
            "data_time": data_time,
            "discipline": _codes_int(eccodes, gid, "discipline"),
            "parameter_category": _codes_int(eccodes, gid, "parameterCategory"),
            "parameter_number": _codes_int(eccodes, gid, "parameterNumber"),
            "centre": str(_codes_get(eccodes, gid, "centre", "")),
            "sub_centre": _codes_int(eccodes, gid, "subCentre", 0),
            "type_of_generating_process": _codes_int(eccodes, gid, "typeOfGeneratingProcess"),
            "generating_process_identifier": _codes_int(eccodes, gid, "generatingProcessIdentifier"),
            "uv_relative_to_grid": bool(int(_codes_get(eccodes, gid, "uvRelativeToGrid", 0) or 0)),
            "scanning_mode": int(_codes_get(eccodes, gid, "scanningMode", 0) or 0),
            "i_scans_negatively": bool(int(_codes_get(eccodes, gid, "iScansNegatively", 0) or 0)),
            "j_scans_positively": bool(int(_codes_get(eccodes, gid, "jScansPositively", 0) or 0)),
            "j_points_are_consecutive": bool(int(_codes_get(eccodes, gid, "jPointsAreConsecutive", 0) or 0)),
            "alternative_row_scanning": bool(int(_codes_get(eccodes, gid, "alternativeRowScanning", 0) or 0)),
            "grid_type": str(_codes_get(eccodes, gid, "gridType", "")),
            "grid_definition_template": int(_codes_get(eccodes, gid, "gridDefinitionTemplateNumber", -1) or -1),
            "nx": int(nx or 0),
            "ny": int(ny or 0),
        }
    finally:
        eccodes.codes_release(gid)


def _bbox_subset(latitude: Any, longitude: Any, bbox: Sequence[float], halo: int) -> tuple[slice, slice, Any]:
    import numpy as np

    west, south, east, north = map(float, bbox)
    lon = ((np.asarray(longitude, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lat = np.asarray(latitude, dtype=np.float64)
    lon_mask = (lon >= west) & (lon <= east) if west <= east else ((lon >= west) | (lon <= east))
    mask = np.isfinite(lat) & np.isfinite(lon) & (lat >= south) & (lat <= north) & lon_mask
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise HrrrError("Requested bbox does not intersect the HRRR grid")
    y0, y1 = max(0, int(rows.min()) - halo), min(lat.shape[0], int(rows.max()) + halo + 1)
    x0, x1 = max(0, int(cols.min()) - halo), min(lat.shape[1], int(cols.max()) + halo + 1)
    return slice(y0, y1), slice(x0, x1), mask[y0:y1, x0:x1]


def _tangent_basis(transformer: Any, x: Any, y: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    import numpy as np

    xx, yy = np.meshgrid(x, y)
    lon, lat = transformer.transform(xx, yy)
    lon_x, lat_x = transformer.transform(xx + 1000.0, yy)
    lon_y, lat_y = transformer.transform(xx, yy + 1000.0)

    def basis(lon1: Any, lat1: Any) -> tuple[Any, Any]:
        dlon = ((np.asarray(lon1) - np.asarray(lon) + 180.0) % 360.0) - 180.0
        east = np.deg2rad(dlon) * np.cos(np.deg2rad((np.asarray(lat1) + np.asarray(lat)) / 2.0))
        north = np.deg2rad(np.asarray(lat1) - np.asarray(lat))
        norm = np.hypot(east, north)
        return east / norm, north / norm

    ex, nx = basis(lon_x, lat_x)
    ey, ny = basis(lon_y, lat_y)
    return np.asarray(lat), ((np.asarray(lon) + 180.0) % 360.0) - 180.0, ex, nx, ey, ny


def _decode_message(
    path: Path,
    target: Mapping[str, Any],
    request: Mapping[str, Any],
    grid_cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    import numpy as np
    from pyproj import CRS, Transformer
    from rasterio.io import MemoryFile

    metadata = _metadata(path)
    idx_short = str(target["short_name"]).upper()
    decoded_short = str(metadata["short_name"]).lower()
    accepted_short_names = ECCODES_SHORT_NAMES.get(idx_short)
    if accepted_short_names is not None and decoded_short not in accepted_short_names:
        raise HrrrError(f"Decoded shortName {metadata['short_name']} does not match selected {target['short_name']}")
    metadata["idx_short_name"] = idx_short
    if str(metadata["step_type"]).lower() != str(target.get("step_type", "instant")).lower():
        raise HrrrError(f"Decoded step type {metadata['step_type']} does not match {target.get('step_type', 'instant')}")
    expected = DOMAIN_INFO[str(request["domain"])]
    if (metadata["nx"], metadata["ny"], metadata["grid_definition_template"]) != (
        int(expected["nx"]), int(expected["ny"]), int(expected["grid_template"])
    ):
        raise HrrrError(
            f"Unexpected {request['domain']} grid {metadata['nx']}x{metadata['ny']} template {metadata['grid_definition_template']}"
        )
    if target.get("type_of_level") and str(metadata["type_of_level"]).lower() != str(target["type_of_level"]).lower():
        raise HrrrError(f"Decoded level type {metadata['type_of_level']} does not match {target['type_of_level']}")
    if target.get("level_value") is not None and abs(float(metadata["level"]) - float(target["level_value"])) > 1e-6:
        raise HrrrError(f"Decoded level {metadata['level']} does not match {target['level_value']}")
    payload = path.read_bytes()
    with MemoryFile(payload) as memory:
        with memory.open() as raster:
            values = np.asarray(raster.read(1), dtype=np.float32)
            transform = raster.transform
            if raster.crs is None:
                raise HrrrError("Rasterio did not expose the HRRR projected CRS")
            crs = CRS.from_user_input(raster.crs)
    if values.shape != (int(expected["ny"]), int(expected["nx"])):
        raise HrrrError(f"Raster shape {values.shape} does not match expected HRRR grid")
    signature = hash_payload({"shape": values.shape, "transform": tuple(transform), "crs": crs.to_wkt()})
    grid = grid_cache.get(signature)
    if grid is None:
        x_full = transform.c + transform.a * (np.arange(values.shape[1], dtype=np.float64) + 0.5)
        y_full = transform.f + transform.e * (np.arange(values.shape[0], dtype=np.float64) + 0.5)
        transformer = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        latitude, longitude, ex, nx, ey, ny = _tangent_basis(transformer, x_full, y_full)
        ys, xs, mask = _bbox_subset(latitude, longitude, request["bbox"], int(request["halo_cells"]))
        grid = {
            "signature": signature,
            "x": x_full[xs],
            "y": y_full[ys],
            "latitude": latitude[ys, xs],
            "longitude": longitude[ys, xs],
            "bbox_mask": mask,
            "basis_ex": ex[ys, xs],
            "basis_nx": nx[ys, xs],
            "basis_ey": ey[ys, xs],
            "basis_ny": ny[ys, xs],
            "y_slice": [ys.start, ys.stop],
            "x_slice": [xs.start, xs.stop],
            "crs_wkt": crs.to_wkt(),
        }
        grid_cache[signature] = grid
    ys = slice(*grid["y_slice"])
    xs = slice(*grid["x_slice"])
    data = values[ys, xs]
    data[np.abs(data) > 9.0e35] = np.nan
    return metadata, data, grid


def _vertical_dimension(target: Mapping[str, Any], metadata: Mapping[str, Any]) -> tuple[str | None, float | None]:
    level_type = str(target.get("type_of_level", metadata.get("type_of_level", ""))).lower()
    level = float(metadata.get("level", 0.0))
    if level_type == "isobaricinhpa":
        return "pressure", level * 100.0
    if level_type == "hybrid":
        return "hybrid_level", level
    return None, None


def _rotate_winds(records: list[dict[str, Any]], grid: Mapping[str, Any]) -> None:
    import numpy as np

    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        target = record["target"]
        if target.get("vector_group"):
            key = (
                record["cadence_group"],
                record["cycle"],
                record["forecast_period_minutes"],
                target["vector_group"],
                record.get("vertical_value"),
            )
            grouped[key][str(target["component"])] = record
    for key, pair in grouped.items():
        if set(pair) != {"u", "v"}:
            raise HrrrError(f"Canonical vector pair is incomplete: {key}")
        u_record, v_record = pair["u"], pair["v"]
        if bool(u_record["metadata"].get("uv_relative_to_grid")) != bool(v_record["metadata"].get("uv_relative_to_grid")):
            raise HrrrError(f"U/V orientation flags disagree: {key}")
        if not bool(u_record["target"].get("rotate_to_earth")):
            label = "preserved_grid_relative" if bool(u_record["metadata"].get("uv_relative_to_grid")) else "preserved_earth_relative"
            u_record["metadata"]["wind_rotation"] = label
            v_record["metadata"]["wind_rotation"] = label
            continue
        if not bool(u_record["metadata"].get("uv_relative_to_grid")):
            u_record["metadata"]["wind_rotation"] = "source_earth_relative"
            v_record["metadata"]["wind_rotation"] = "source_earth_relative"
            continue
        u = np.load(u_record["scratch_path"])
        v = np.load(v_record["scratch_path"])
        east = u * grid["basis_ex"] + v * grid["basis_ey"]
        north = u * grid["basis_nx"] + v * grid["basis_ny"]
        np.save(u_record["scratch_path"], east.astype(np.float32))
        np.save(v_record["scratch_path"], north.astype(np.float32))
        for record in (u_record, v_record):
            record["metadata"]["wind_rotation"] = "grid_to_earth_tangent_basis"
            record["metadata"]["source_uv_relative_to_grid"] = True


def _write_output(
    destination: Path,
    request: Mapping[str, Any],
    request_hash: str,
    records: Sequence[Mapping[str, Any]],
    grid: Mapping[str, Any],
    provider_locks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import netCDF4 as nc4
    import numpy as np

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    mode = str(request["mode"])
    cycles = sorted({str(row["cycle"]) for row in records})
    periods = sorted({int(row["forecast_period_minutes"]) for row in records})
    variable_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        variable_groups[str(record["target"]["output_name"])].append(record)
    pressure_values = sorted({float(row["vertical_value"]) for row in records if row.get("vertical_dimension") == "pressure"})
    hybrid_values = sorted({float(row["vertical_value"]) for row in records if row.get("vertical_dimension") == "hybrid_level"})
    with nc4.Dataset(temporary, "w", format="NETCDF4") as dataset:
        dataset.createDimension("y", len(grid["y"]))
        dataset.createDimension("x", len(grid["x"]))
        if mode == "analysis":
            valid_times = sorted({time_text(parse_utc(str(row["cycle"])) + timedelta(minutes=int(row["forecast_period_minutes"]))) for row in records})
            dataset.createDimension("time", len(valid_times))
            tv = dataset.createVariable("time", "i8", ("time",))
            tv.units = "seconds since 1970-01-01 00:00:00 UTC"
            tv.calendar = "gregorian"
            tv[:] = [int(parse_utc(value).timestamp()) for value in valid_times]
            time_index = {value: index for index, value in enumerate(valid_times)}
            temporal_dimensions = ("time",)
        else:
            dataset.createDimension("forecast_reference_time", len(cycles))
            dataset.createDimension("forecast_period", len(periods))
            cv = dataset.createVariable("forecast_reference_time", "i8", ("forecast_reference_time",))
            cv.units = "seconds since 1970-01-01 00:00:00 UTC"
            cv.calendar = "gregorian"
            cv[:] = [int(parse_utc(value).timestamp()) for value in cycles]
            pv = dataset.createVariable("forecast_period", "i4", ("forecast_period",))
            pv.units = "minutes"
            pv[:] = periods
            vv = dataset.createVariable("valid_time", "i8", ("forecast_reference_time", "forecast_period"))
            vv.units = cv.units
            vv.calendar = cv.calendar
            vv[:] = np.asarray([[int((parse_utc(cycle) + timedelta(minutes=period)).timestamp()) for period in periods] for cycle in cycles])
            cycle_index = {value: index for index, value in enumerate(cycles)}
            period_index = {value: index for index, value in enumerate(periods)}
            temporal_dimensions = ("forecast_reference_time", "forecast_period")
        if pressure_values:
            dataset.createDimension("pressure", len(pressure_values))
            pv2 = dataset.createVariable("pressure", "f8", ("pressure",))
            pv2.units = "Pa"
            pv2.positive = "down"
            pv2[:] = pressure_values
        if hybrid_values:
            dataset.createDimension("hybrid_level", len(hybrid_values))
            hv = dataset.createVariable("hybrid_level", "f8", ("hybrid_level",))
            hv.units = "1"
            hv.positive = "down"
            hv[:] = hybrid_values
        xv = dataset.createVariable("x", "f8", ("x",)); xv.units = "m"; xv.standard_name = "projection_x_coordinate"; xv[:] = grid["x"]
        yv = dataset.createVariable("y", "f8", ("y",)); yv.units = "m"; yv.standard_name = "projection_y_coordinate"; yv[:] = grid["y"]
        latv = dataset.createVariable("latitude", "f8", ("y", "x")); latv.units = "degrees_north"; latv.standard_name = "latitude"; latv[:] = grid["latitude"]
        lonv = dataset.createVariable("longitude", "f8", ("y", "x")); lonv.units = "degrees_east"; lonv.standard_name = "longitude"; lonv[:] = grid["longitude"]
        maskv = dataset.createVariable("bbox_mask", "i1", ("y", "x")); maskv.long_name = "requested bounding box mask before halo"; maskv[:] = np.asarray(grid["bbox_mask"], dtype=np.int8)
        crs = dataset.createVariable("crs", "i4")
        crs.spatial_ref = str(grid["crs_wkt"])
        crs.crs_wkt = str(grid["crs_wkt"])
        try:
            from pyproj import CRS

            for attribute, value in CRS.from_wkt(str(grid["crs_wkt"])).to_cf().items():
                setattr(crs, attribute, value)
        except Exception:
            pass
        pressure_index = {value: index for index, value in enumerate(pressure_values)}
        hybrid_index = {value: index for index, value in enumerate(hybrid_values)}
        for name, rows in sorted(variable_groups.items()):
            vertical = rows[0].get("vertical_dimension")
            if any(row.get("vertical_dimension") != vertical for row in rows):
                raise HrrrError(f"Variable {name} mixes vertical coordinate types")
            dimensions = temporal_dimensions + ((str(vertical),) if vertical else ()) + ("y", "x")
            variable = dataset.createVariable(name, "f4", dimensions, zlib=True, complevel=2, fill_value=np.float32(9.96921e36))
            variable.units = str(rows[0]["metadata"].get("units", "1"))
            variable.long_name = str(rows[0]["metadata"].get("name", name))
            variable.grid_mapping = "crs"
            variable.coordinates = "latitude longitude"
            if name.startswith("eastward_wind"):
                variable.standard_name = "eastward_wind"
            elif name.startswith("northward_wind"):
                variable.standard_name = "northward_wind"
            elif name in CF_STANDARD_NAMES:
                variable.standard_name = CF_STANDARD_NAMES[name]
            variable.source_grib_metadata = canonical_json([row["metadata"] for row in rows])
            for row in rows:
                data = np.load(row["scratch_path"])
                if mode == "analysis":
                    valid = time_text(parse_utc(str(row["cycle"])) + timedelta(minutes=int(row["forecast_period_minutes"])))
                    index: tuple[Any, ...] = (time_index[valid],)
                else:
                    index = (cycle_index[str(row["cycle"])], period_index[int(row["forecast_period_minutes"])])
                if vertical == "pressure":
                    index += (pressure_index[float(row["vertical_value"])],)
                elif vertical == "hybrid_level":
                    index += (hybrid_index[float(row["vertical_value"])],)
                variable[index + (slice(None), slice(None))] = data
        dataset.schema_version = SCHEMA_FIELDS
        dataset.Conventions = "CF-1.10"
        dataset.connector = "hrrr-fetcher"
        dataset.domain = str(request["domain"])
        dataset.time_mode = mode
        dataset.request_hash = request_hash
        dataset.request_bbox = canonical_json(request["bbox"])
        dataset.input_longitude_convention = str(request["longitude_convention"])
        dataset.output_longitude_convention = "-180_180"
        dataset.halo_cells = int(request["halo_cells"])
        dataset.native_nx = int(DOMAIN_INFO[str(request["domain"])]["nx"])
        dataset.native_ny = int(DOMAIN_INFO[str(request["domain"])]["ny"])
        dataset.native_grid_definition_template = int(DOMAIN_INFO[str(request["domain"])]["grid_template"])
        dataset.native_x_slice = canonical_json(grid.get("x_slice"))
        dataset.native_y_slice = canonical_json(grid.get("y_slice"))
        dataset.provider_locks = canonical_json(list(provider_locks))
        dataset.history = f"Created {utc_now()} by hrrr-fetcher without regridding"
    _atomic_replace(temporary, destination)
    return {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256_file(destination), "variables": sorted(variable_groups)}


def execute_plan(plan: Mapping[str, Any], run_dir: str | Path) -> dict[str, Any]:
    import numpy as np
    import requests

    validate_plan(plan)
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    request = plan["request"]
    atomic_write_json(directory / "request.normalized.json", request)
    atomic_write_json(directory / "inventory.json", plan["inventory"])
    atomic_write_json(directory / "download_plan.json", plan)
    raw_root = directory / "raw_messages"
    scratch = directory / "scratch"
    raw_root.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    message_total = sum(len(item["selected_messages"]) for item in plan["inventory"]["objects"])
    status = DownloadStatus(directory / "download_status.json", request_hash=str(plan["request_hash"]), total_chunks=message_total, expected_bytes=int(plan["transfer_bytes"]))
    status.start()
    client = requests.Session()
    grid_cache: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    locks: list[dict[str, Any]] = []
    provider_attempts: list[dict[str, Any]] = []
    raw_artifacts: list[dict[str, Any]] = []
    completed_bytes = 0
    completed_chunks = 0
    try:
        for item in plan["inventory"]["objects"]:
            state, messages, attempts = _download_object(client, item, request, raw_root, status)
            provider_attempts.extend({"object_id": item["id"], **row} for row in attempts)
            locks.append({"object_id": item["id"], "key": item["key"], "provider": state["provider"], "url": state["url"], "selected_signature": state["selected_signature"]})
            for message in messages:
                path = Path(message["path"])
                completed_bytes += int(message["bytes"])
                completed_chunks += 1
                status.update(completed_bytes=completed_bytes, completed_chunks=completed_chunks)
                raw_artifacts.append({key: value for key, value in message.items() if key != "targets"})
                for target in message["targets"]:
                    metadata, data, grid = _decode_message(path, target, request, grid_cache)
                    expected_valid = parse_utc(str(item["cycle"])) + timedelta(minutes=int(target["forecast_period_minutes"]))
                    if metadata["valid_time"] != time_text(expected_valid):
                        raise HrrrError(f"Decoded valid time {metadata['valid_time']} does not match requested {time_text(expected_valid)}")
                    scratch_name = f"{hashlib.sha256((item['id'] + target['selector_id'] + target['forecast_period']).encode()).hexdigest()[:20]}.npy"
                    scratch_path = scratch / scratch_name
                    np.save(scratch_path, data.astype(np.float32))
                    vertical_dimension, vertical_value = _vertical_dimension(target, metadata)
                    records.append({
                        "cadence_group": item["cadence_group"],
                        "cycle": item["cycle"],
                        "forecast_period_minutes": int(target["forecast_period_minutes"]),
                        "target": target,
                        "metadata": {
                            **metadata,
                            "provider": state["provider"],
                            "source_key": item["key"],
                            "source_url": state["url"],
                            "byte_offset": message["offset"],
                            "byte_end": message["end"],
                            "message_sha256": message["sha256"],
                            "source_idx_level": message["level_text"],
                            "source_idx_step": message["step"],
                        },
                        "vertical_dimension": vertical_dimension,
                        "vertical_value": vertical_value,
                        "scratch_path": str(scratch_path),
                    })
        if not records:
            raise HrrrError("No HRRR messages were decoded")
        if len(grid_cache) != 1:
            raise HrrrError(f"Request resolved to {len(grid_cache)} native grids; one domain must remain one grid")
        grid = next(iter(grid_cache.values()))
        _rotate_winds(records, grid)
        outputs: list[dict[str, Any]] = []
        hourly = [row for row in records if row["cadence_group"] == "hourly"]
        subhourly = [row for row in records if row["cadence_group"] == "subhourly"]
        if hourly:
            outputs.append(_write_output(directory / "hrrr_fields.nc", request, str(plan["request_hash"]), hourly, grid, locks))
        if subhourly:
            outputs.append(_write_output(directory / "hrrr_subhourly_fields.nc", request, str(plan["request_hash"]), subhourly, grid, locks))
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_MANIFEST,
            "request_hash": plan["request_hash"],
            "plan_hash": plan["plan_hash"],
            "created_utc": utc_now(),
            "domain": request["domain"],
            "mode": request["mode"],
            "provider_locks": locks,
            "provider_attempts": [*plan["inventory"]["provider_attempts"], *provider_attempts],
            "gaps": plan["inventory"]["gaps"],
            "raw_messages": raw_artifacts,
            "outputs": outputs,
            "records": [
                {
                    "output_name": row["target"]["output_name"],
                    "cycle": row["cycle"],
                    "forecast_period_minutes": row["forecast_period_minutes"],
                    "cadence_group": row["cadence_group"],
                    "vertical_dimension": row["vertical_dimension"],
                    "vertical_value": row["vertical_value"],
                    "metadata": row["metadata"],
                }
                for row in records
            ],
        }
        manifest["manifest_hash"] = hash_payload(manifest)
        atomic_write_json(directory / "run_manifest.json", manifest)
        health = health_run(directory)
        if not health["passed"]:
            raise HrrrError(f"HRRR health check failed: {health['issues']}")
        if not bool(request["retain_raw_messages"]):
            for lock in locks:
                object_hash = hashlib.sha256(str(lock["key"]).encode("utf-8")).hexdigest()[:16]
                shutil.rmtree(raw_root / object_hash, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)
        status.finish("complete", "HRRR download and health check completed")
        return {"manifest": str(directory / "run_manifest.json"), "health": str(directory / "health_check.json"), "outputs": outputs}
    except Exception as exc:
        status.finish("failed", str(exc))
        raise
    finally:
        client.close()


def health_run(run_dir: str | Path) -> dict[str, Any]:
    import netCDF4 as nc4
    import numpy as np

    directory = Path(run_dir)
    manifest_path = directory / "run_manifest.json"
    issues: list[str] = []
    checks: list[dict[str, Any]] = []
    if not manifest_path.exists():
        report = {"schema_version": SCHEMA_HEALTH, "passed": False, "checked_utc": utc_now(), "issues": ["run_manifest.json is missing"], "checks": []}
        atomic_write_json(directory / "health_check.json", report)
        return report
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_MANIFEST:
        issues.append("manifest schema is invalid")
    body = dict(manifest)
    stored = body.pop("manifest_hash", "")
    if stored != hash_payload(body):
        issues.append("manifest hash mismatch")
    requested_names = {str(row["output_name"]) for row in manifest.get("records", [])}
    found_names: set[str] = set()
    for output in manifest.get("outputs", []):
        path = Path(str(output["path"]))
        if not path.is_absolute():
            path = directory / path
        row: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            issues.append(f"missing output {path.name}")
            checks.append(row)
            continue
        actual_hash = sha256_file(path)
        row["sha256"] = actual_hash
        if actual_hash != output.get("sha256"):
            issues.append(f"output hash mismatch for {path.name}")
        try:
            with nc4.Dataset(path) as dataset:
                row["schema_version"] = getattr(dataset, "schema_version", None)
                if row["schema_version"] != SCHEMA_FIELDS:
                    issues.append(f"invalid schema in {path.name}")
                for coordinate in ("x", "y", "latitude", "longitude", "bbox_mask", "crs"):
                    if coordinate not in dataset.variables:
                        issues.append(f"{path.name} lacks {coordinate}")
                for attribute in ("halo_cells", "native_nx", "native_ny", "native_grid_definition_template", "native_x_slice", "native_y_slice"):
                    if not hasattr(dataset, attribute):
                        issues.append(f"{path.name} lacks {attribute} metadata")
                if "bbox_mask" in dataset.variables and not np.any(dataset.variables["bbox_mask"][:] == 1):
                    issues.append(f"{path.name} bbox mask is empty")
                output_records = [
                    record
                    for record in manifest.get("records", [])
                    if ("subhourly" in path.name) == (record.get("cadence_group") == "subhourly")
                ]
                if manifest.get("mode") == "analysis":
                    expected_times = sorted({
                        int((parse_utc(str(record["cycle"])) + timedelta(minutes=int(record["forecast_period_minutes"]))).timestamp())
                        for record in output_records
                    })
                    actual_times = [int(value) for value in np.asarray(dataset.variables.get("time", [])[:]).ravel()]
                    if actual_times != expected_times:
                        issues.append(f"{path.name} analysis time coverage differs from the manifest")
                else:
                    expected_cycles = sorted({int(parse_utc(str(record["cycle"])).timestamp()) for record in output_records})
                    expected_periods = sorted({int(record["forecast_period_minutes"]) for record in output_records})
                    actual_cycles = [int(value) for value in np.asarray(dataset.variables.get("forecast_reference_time", [])[:]).ravel()]
                    actual_periods = [int(value) for value in np.asarray(dataset.variables.get("forecast_period", [])[:]).ravel()]
                    if actual_cycles != expected_cycles or actual_periods != expected_periods:
                        issues.append(f"{path.name} forecast axes differ from the manifest")
                    elif "valid_time" not in dataset.variables:
                        issues.append(f"{path.name} lacks two-dimensional valid_time")
                    else:
                        expected_valid = np.asarray([[cycle + period * 60 for period in expected_periods] for cycle in expected_cycles])
                        if not np.array_equal(np.asarray(dataset.variables["valid_time"][:]), expected_valid):
                            issues.append(f"{path.name} valid_time differs from forecast_reference_time + forecast_period")
                for name in output.get("variables", []):
                    found_names.add(str(name))
                    if name not in dataset.variables:
                        issues.append(f"{path.name} lacks requested variable {name}")
                        continue
                    values = np.ma.asarray(dataset.variables[name][:])
                    finite = np.isfinite(values.compressed())
                    if finite.size == 0 or not finite.any():
                        issues.append(f"{path.name}:{name} has no finite values")
                row["dimensions"] = {name: len(value) for name, value in dataset.dimensions.items()}
                row["variables"] = list(output.get("variables", []))
        except Exception as exc:
            issues.append(f"cannot inspect {path.name}: {exc}")
        checks.append(row)
    missing_variables = sorted(requested_names - found_names)
    if missing_variables:
        issues.append(f"manifest variables missing from outputs: {missing_variables}")
    report = {
        "schema_version": SCHEMA_HEALTH,
        "passed": not issues,
        "checked_utc": utc_now(),
        "request_hash": manifest.get("request_hash"),
        "issues": issues,
        "checks": checks,
        "provider_locks": manifest.get("provider_locks", []),
        "gaps": manifest.get("gaps", []),
    }
    atomic_write_json(directory / "health_check.json", report)
    return report
