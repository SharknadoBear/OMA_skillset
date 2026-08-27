from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .features import standardize_feature_doc

from .normalization import request_text


NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "fvcom-region-bpoly/1.0 (PNNL research preprocessing; named-place discovery)"
WATER_TYPES = {
    "archipelago",
    "bay",
    "channel",
    "estuary",
    "fjord",
    "gulf",
    "island",
    "lake",
    "ocean",
    "river",
    "sea",
    "sound",
    "strait",
    "water",
}


class PlaceDiscoveryError(RuntimeError):
    pass


def extract_named_region_query(request: dict[str, Any] | str) -> str:
    """Extract a concise geographic target from a natural modeling request."""
    raw = request_text(request).strip()
    patterns = [
        r"\bmodel\s+(?:of|for)\s+(.+?)(?=\s+to\s+study\b|\s+for\s+(?:studying|investigating|simulating)\b|[.;]|$)",
        r"\b(?:region|domain)\s+(?:of|for|covering)\s+(.+?)(?=\s+to\s+study\b|[.;]|$)",
        r"\b(?:define|cover|model)\s+(.+?)(?=\s+to\s+study\b|[.;]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            query = match.group(1).strip(" ,:-")
            query = re.sub(r"^(?:an?|the)\s+", "", query, flags=re.IGNORECASE)
            if query:
                return query

    first_sentence = re.split(r"[.;]", raw, maxsplit=1)[0]
    first_sentence = re.sub(
        r"^(?:please\s+)?(?:i\s+am\s+developing|we\s+are\s+developing|use\s+fvcom\s+regionbpoly\s+to\s+define)\s+",
        "",
        first_sentence,
        flags=re.IGNORECASE,
    )
    return first_sentence.strip(" ,:-")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:64] or "discovered_region"


def _valid_bbox(bbox: list[float]) -> bool:
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        return False
    west, south, east, north = bbox
    return -180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0


def normalize_seed_bbox(bbox: list[float], domain_type: str, minimum_span_km: float | None = None) -> list[float]:
    """Expand point-like geocoder results into a usable initial mission frame."""
    if not _valid_bbox(bbox):
        raise PlaceDiscoveryError(f"invalid discovered bbox: {bbox}")
    west, south, east, north = bbox
    center_lon = (west + east) / 2.0
    center_lat = (south + north) / 2.0
    if minimum_span_km is None:
        minimum_span_km = {"lake": 140.0, "island": 180.0, "coastal": 120.0}.get(domain_type, 120.0)
    min_lat_span = minimum_span_km / 111.0
    lon_scale = max(0.15, math.cos(math.radians(center_lat)))
    min_lon_span = min(12.0, minimum_span_km / (111.0 * lon_scale))
    lon_span = max(east - west, min_lon_span)
    lat_span = max(north - south, min_lat_span)
    pad_lon = max(0.05, lon_span * 0.08)
    pad_lat = max(0.05, lat_span * 0.08)
    out = [
        max(-180.0, center_lon - lon_span / 2.0 - pad_lon),
        max(-89.9, center_lat - lat_span / 2.0 - pad_lat),
        min(180.0, center_lon + lon_span / 2.0 + pad_lon),
        min(89.9, center_lat + lat_span / 2.0 + pad_lat),
    ]
    if not _valid_bbox(out):
        raise PlaceDiscoveryError(f"normalized discovery bbox is invalid: {out}")
    return [round(value, 7) for value in out]


def _candidate_bbox(candidate: dict[str, Any]) -> list[float]:
    raw = candidate.get("boundingbox")
    if not isinstance(raw, list) or len(raw) != 4:
        raise PlaceDiscoveryError("geocoder candidate is missing boundingbox")
    south, north, west, east = (float(value) for value in raw)
    return [west, south, east, north]


def _candidate_score(candidate: dict[str, Any], query: str) -> tuple[float, float, str]:
    display = str(candidate.get("display_name", "")).lower()
    query_tokens = {token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 2}
    token_fraction = sum(1 for token in query_tokens if token in display) / max(1, len(query_tokens))
    category = str(candidate.get("category") or candidate.get("class") or "").lower()
    place_type = str(candidate.get("type") or candidate.get("addresstype") or "").lower()
    water_bonus = 2.0 if category == "natural" or place_type in WATER_TYPES else 0.0
    importance = float(candidate.get("importance") or 0.0)
    return (water_bonus + token_fraction * 3.0 + importance, importance, display)


def _cache_path(cache_dir: Path, query: str) -> Path:
    digest = hashlib.sha256(query.casefold().encode("utf-8")).hexdigest()
    return cache_dir / f"nominatim_{digest}.json"


def _nominatim_search(query: str, timeout_s: float, cache_dir: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_path = _cache_path(cache_dir, query) if cache_dir else None
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return list(cached.get("results", [])), {"cache_hit": True, "cache_path": str(cache_path)}

    params = urlencode({"q": query, "format": "jsonv2", "addressdetails": 1, "limit": 5})
    endpoint = f"{NOMINATIM_SEARCH_URL}?{params}"
    request = Request(endpoint, headers={"User-Agent": NOMINATIM_USER_AGENT, "Accept-Language": "en"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            results = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # urllib raises several transport-specific subclasses
        raise PlaceDiscoveryError(f"online named-place lookup failed for {query!r}: {exc}") from exc
    if not isinstance(results, list):
        raise PlaceDiscoveryError("online named-place lookup returned a non-list response")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"query": query, "results": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results, {"cache_hit": False, "cache_path": str(cache_path) if cache_path else None}


def feature_doc_from_bbox(
    request: dict[str, Any] | str,
    bbox: list[float],
    label: str,
    domain_type: str,
    source: str,
    discovery: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_bbox = normalize_seed_bbox([float(value) for value in bbox], domain_type)
    feature_id = f"{_slug(label)}_discovered_extent"
    source_kind = "agent_supplied_bbox" if source == "agent_supplied_place_discovery" else "web_discovery"
    geometry_status = "inferred_seed" if source_kind == "agent_supplied_bbox" else "discovered_seed"
    feature_doc = standardize_feature_doc({
        "schema_version": "target_region_features_v1",
        "source": source,
        "request_text": request_text(request),
        "domain_scale": "regional",
        "domain_variant": "discovered_named_region",
        "considerations": {},
        "features": [
            {
                "id": feature_id,
                "label": label,
                "role": "discovered_geographic_seed",
                "category": "target_region",
                "type": "bbox",
                "geometry": seed_bbox,
                "required": True,
                "notes": "Initial mission frame derived from named-place discovery; visual review must refine scope and confirm the offshore side.",
            }
        ],
        "place_discovery": discovery,
    }, request, source_kind=source_kind, source_key=discovery.get("query"), geometry_status=geometry_status)
    discovery_record = dict(discovery)
    discovery_record.update(
        {
            "schema_version": "region_place_discovery_v1",
            "source": source,
            "request_text": request_text(request),
            "domain_type_assumption": domain_type,
            "raw_bbox": [float(value) for value in bbox],
            "normalized_seed_bbox": seed_bbox,
            "feature_id": feature_id,
            "requires_visual_scope_confirmation": True,
            "requires_visual_offshore_side_confirmation": domain_type == "coastal",
        }
    )
    return feature_doc, discovery_record


def discover_named_region_features(
    request: dict[str, Any] | str,
    domain_type: str,
    *,
    bbox_override: list[float] | None = None,
    label_override: str | None = None,
    query_override: str | None = None,
    timeout_s: float = 20.0,
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    query = (query_override or extract_named_region_query(request)).strip()
    if not query:
        raise PlaceDiscoveryError("could not extract a named geographic target from the request")

    if bbox_override is not None:
        label = (label_override or query).strip()
        discovery = {
            "provider": "agent_or_user_supplied_bbox",
            "query": query,
            "selected_display_name": label,
            "selection_method": "explicit_bbox_override",
            "attribution": None,
        }
        return feature_doc_from_bbox(request, bbox_override, label, domain_type, "agent_supplied_place_discovery", discovery)

    results, cache = _nominatim_search(query, timeout_s, cache_dir)
    usable: list[dict[str, Any]] = []
    for candidate in results:
        try:
            _candidate_bbox(candidate)
        except (PlaceDiscoveryError, TypeError, ValueError):
            continue
        usable.append(candidate)
    if not usable:
        raise PlaceDiscoveryError(f"online named-place lookup returned no bbox for {query!r}")
    selected = max(usable, key=lambda item: _candidate_score(item, query))
    raw_bbox = _candidate_bbox(selected)
    label = str(selected.get("display_name") or label_override or query)
    discovery = {
        "provider": "OpenStreetMap Nominatim",
        "provider_url": NOMINATIM_SEARCH_URL,
        "query": query,
        "candidate_count": len(results),
        "usable_candidate_count": len(usable),
        "selected_display_name": label,
        "selected_category": selected.get("category") or selected.get("class"),
        "selected_type": selected.get("type") or selected.get("addresstype"),
        "selected_importance": selected.get("importance"),
        "selected_osm_type": selected.get("osm_type"),
        "selected_osm_id": selected.get("osm_id"),
        "selected_place_id": selected.get("place_id"),
        "selection_method": "water_and_name_match_then_importance",
        "attribution": selected.get("licence") or "Data © OpenStreetMap contributors, ODbL 1.0",
        **cache,
    }
    return feature_doc_from_bbox(request, raw_bbox, label, domain_type, "online_named_place_discovery", discovery)
