"""Small bbox OpenStreetMap coastline fallback via Overpass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
from shapely.geometry import LineString, box

from .progress import ProgressReporter, normalize_timeout


OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
OSM_ATTRIBUTION = "OpenStreetMap contributors, ODbL 1.0"


def _cache_key(bbox: tuple[float, float, float, float], name: str = "") -> str:
    text = f"{name}|{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_overpass_query(
    bbox: tuple[float, float, float, float],
    *,
    overpass_timeout_s: float | int | None = 0,
) -> str:
    """Build an Overpass coastline query for W,S,E,N bbox."""

    west, south, east, north = bbox
    timeout = normalize_timeout(overpass_timeout_s)
    header = "[out:json]"
    if timeout is not None:
        header += f"[timeout:{int(timeout)}]"
    header += ";\n"
    return header + (
        "(\n"
        f'  way["natural"="coastline"]({south},{west},{north},{east});\n'
        ");\n"
        "out geom;\n"
    )


def fetch_overpass_json(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
    *,
    name: str = "",
    endpoints: tuple[str, ...] = OVERPASS_ENDPOINTS,
    refresh: bool = False,
    client_timeout_s: float | int | None = 0,
    overpass_timeout_s: float | int | None = 0,
    reporter: ProgressReporter | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Fetch or load cached Overpass JSON for a coastline bbox query."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"osm_coastline_{_cache_key(bbox, name)}.json"
    query = build_overpass_query(bbox, overpass_timeout_s=overpass_timeout_s)

    if cache_path.exists() and not refresh:
        if reporter:
            reporter.event(
                "osm-cache",
                "using cached Overpass JSON",
                path=str(cache_path),
                size_mb=round(cache_path.stat().st_size / 1048576.0, 3),
            )
        return json.loads(cache_path.read_text(encoding="utf-8")), {
            "mode": "cache",
            "cache_path": str(cache_path),
            "query": query,
            "endpoint": None,
        }

    errors: list[str] = []
    client_timeout = normalize_timeout(client_timeout_s)
    for endpoint in endpoints:
        request = Request(
            endpoint,
            data=urlencode({"data": query}).encode("utf-8"),
            headers={"User-Agent": "fvcom-cusp-coastline/2.0"},
        )
        started = time.time()
        try:
            if reporter:
                reporter.event(
                    "osm-overpass",
                    "submitting OSM coastline query",
                    endpoint=endpoint,
                    client_timeout_s=client_timeout,
                    overpass_timeout_s=normalize_timeout(overpass_timeout_s),
                )
            chunks: list[bytes] = []
            bytes_read = 0
            kwargs = {"timeout": client_timeout} if client_timeout is not None else {}
            heartbeat = reporter.background_heartbeat(
                "osm-overpass",
                "waiting for/downloading Overpass response",
                endpoint=endpoint,
            ) if reporter else None
            if heartbeat:
                heartbeat.__enter__()
            try:
                with urlopen(request, **kwargs) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        bytes_read += len(chunk)
                        if reporter:
                            reporter.heartbeat(
                                "osm-overpass",
                                "downloaded Overpass JSON",
                                endpoint=endpoint,
                                bytes_mb=round(bytes_read / 1048576.0, 3),
                            )
            finally:
                if heartbeat:
                    heartbeat.__exit__(None, None, None)
            raw = b"".join(chunks)
            data = json.loads(raw.decode("utf-8"))
            cache_path.write_text(json.dumps(data), encoding="utf-8")
            if reporter:
                reporter.event(
                    "osm-overpass",
                    "cached Overpass JSON",
                    endpoint=endpoint,
                    path=str(cache_path),
                    bytes_mb=round(len(raw) / 1048576.0, 3),
                    seconds=round(time.time() - started, 3),
                )
            return data, {
                "mode": "live",
                "cache_path": str(cache_path),
                "query": query,
                "endpoint": endpoint,
                "bytes": len(raw),
                "seconds": round(time.time() - started, 3),
            }
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError("All Overpass endpoints failed: " + "; ".join(errors))


def overpass_json_to_lines(
    data: dict[str, object],
    bbox: tuple[float, float, float, float],
    *,
    reporter: ProgressReporter | None = None,
) -> gpd.GeoDataFrame:
    """Convert Overpass JSON coastline ways to clipped EPSG:4326 line features."""

    rows: list[dict[str, object]] = []
    bbox_poly = box(*bbox)
    elements = list(data.get("elements", []))
    total = len(elements)
    if reporter:
        reporter.event("osm-convert", "converting Overpass elements to line geometries", elements=total)
    for index, element in enumerate(elements, start=1):
        if reporter:
            reporter.heartbeat("osm-convert", "processed OSM elements", processed=index, total=total, rows=len(rows))
        if not isinstance(element, dict) or element.get("type") != "way":
            continue
        coords = []
        for point in element.get("geometry", []):
            if "lon" in point and "lat" in point:
                coords.append((float(point["lon"]), float(point["lat"])))
        if len(coords) < 2:
            continue
        tags = element.get("tags", {}) if isinstance(element.get("tags"), dict) else {}
        rows.append(
            {
                "osm_id": element.get("id"),
                "osm_source": tags.get("source"),
                "osm_tags": json.dumps(tags, sort_keys=True),
                "fvcom_source": "osm_overpass",
                "source_rank": 2,
                "source_status": "fallback_candidate",
                "source_license": OSM_ATTRIBUTION,
                "merge_action": "candidate",
                "geometry": LineString(coords).intersection(bbox_poly),
            }
        )

    if not rows:
        if reporter:
            reporter.event("osm-convert", "no OSM coastline geometries converted", elements=total)
        return gpd.GeoDataFrame(
            {
                "osm_id": [],
                "osm_source": [],
                "osm_tags": [],
                "fvcom_source": [],
                "source_rank": [],
                "source_status": [],
                "source_license": [],
                "merge_action": [],
            },
            geometry=[],
            crs="EPSG:4326",
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if gdf.empty:
        if reporter:
            reporter.event("osm-convert", "no OSM coastline geometries converted", elements=total)
        return gdf
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    if reporter:
        reporter.event("osm-convert", "converted OSM coastline geometries", elements=total, features=len(gdf))
    return gdf.reset_index(drop=True)


def fetch_osm_coastline(
    bbox: tuple[float, float, float, float],
    cache_dir: str | Path,
    *,
    name: str = "",
    refresh: bool = False,
    client_timeout_s: float | int | None = 0,
    overpass_timeout_s: float | int | None = 0,
    reporter: ProgressReporter | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    """Fetch OSM coastline candidates and return lines plus fetch metadata."""

    data, fetch_meta = fetch_overpass_json(
        bbox,
        cache_dir,
        name=name,
        refresh=refresh,
        client_timeout_s=client_timeout_s,
        overpass_timeout_s=overpass_timeout_s,
        reporter=reporter,
    )
    gdf = overpass_json_to_lines(data, bbox, reporter=reporter)
    fetch_meta["feature_count"] = int(len(gdf))
    fetch_meta["attribution"] = OSM_ATTRIBUTION
    return gdf, fetch_meta
