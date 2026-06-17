"""Build and read local NOAA CUDEM tile indexes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

import requests

from .tiles import COLLECTION_INFO, COLLECTION_ORDER, TileRecord, parse_cudem_tile_name

THREDDS_CATALOG_BASE = "https://www.ngdc.noaa.gov/thredds/catalog/tiles"
THREDDS_DAP_BASE = "https://www.ngdc.noaa.gov/thredds/dodsC"
THREDDS_FILE_BASE = "https://www.ngdc.noaa.gov/thredds/fileServer"

DIGITAL_COAST_NINTH_ROOT = (
    "https://chs.coast.noaa.gov/htdata/raster2/elevation/"
    "NCEI_ninth_Topobathy_2014_8483"
)

DEFAULT_DIGITAL_COAST_REGIONS = (
    "AK",
    "AL_nwFL",
    "CA",
    "FL",
    "LA_MS",
    "MA_NH_ME",
    "NC",
    "OR",
    "TX",
    "chesapeake_bay",
    "columbia_river",
    "northeast_sandy",
    "rima",
    "southeast",
    "wash_bellingham",
    "wash_juandefuca",
    "wash_outercoast",
    "wash_pugetsound",
)

DEFAULT_DIGITAL_COAST_URLLISTS = (
    "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/"
    "NCEI_ninth_Topobathy_2014_8483/urllist8483.txt",
    "https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/"
    "NCEI_third_Topobathy_2014_8580/urllist8580.txt",
)

CATALOG_VERSION = 1
USER_AGENT = "fvcom-cudem-bathy/0.1 (+https://www.noaa.gov/)"
NS = {"t": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}


def build_tile_index(
    *,
    collections: tuple[str, ...] = COLLECTION_ORDER,
    include_thredds: bool = True,
    include_digital_coast: bool = True,
    include_urllists: bool = True,
    digital_regions: tuple[str, ...] = DEFAULT_DIGITAL_COAST_REGIONS,
    digital_urllists: tuple[str, ...] = DEFAULT_DIGITAL_COAST_URLLISTS,
    timeout: int = 60,
) -> dict:
    """Build a CUDEM tile index from remote machine-readable catalogs."""

    tiles: list[TileRecord] = []
    warnings: list[str] = []
    if include_thredds:
        for collection in collections:
            try:
                tiles.extend(fetch_thredds_collection(collection, timeout=timeout))
            except Exception as exc:  # keep one bad catalog from blocking others
                warnings.append(f"THREDDS {collection}: {type(exc).__name__}: {exc}")

    if include_digital_coast:
        for region in digital_regions:
            try:
                tiles.extend(fetch_digital_coast_region(region, timeout=timeout))
            except Exception as exc:
                warnings.append(
                    f"Digital Coast {region}: {type(exc).__name__}: {exc}"
                )

    if include_urllists:
        for url in digital_urllists:
            try:
                tiles.extend(fetch_digital_coast_urllist(url, timeout=timeout))
            except Exception as exc:
                warnings.append(
                    f"Digital Coast urllist {url}: {type(exc).__name__}: {exc}"
                )

    tiles = _dedupe_records(tiles)
    return {
        "version": CATALOG_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "thredds_catalog_base": THREDDS_CATALOG_BASE,
            "digital_coast_ninth_root": DIGITAL_COAST_NINTH_ROOT,
            "digital_coast_urllists": list(digital_urllists),
        },
        "warnings": warnings,
        "tiles": [tile.to_dict() for tile in tiles],
    }


def fetch_thredds_collection(collection: str, *, timeout: int = 60) -> list[TileRecord]:
    """Fetch one root-level THREDDS CUDEM collection catalog."""

    if collection not in COLLECTION_INFO:
        raise ValueError(f"Unsupported collection: {collection}")

    catalog_url = f"{THREDDS_CATALOG_BASE}/{collection}/catalog.xml"
    text = _get_text(catalog_url, timeout=timeout)
    root = ET.fromstring(text)
    tiles: list[TileRecord] = []
    for ds in root.findall(".//t:dataset[@urlPath]", NS):
        name = ds.attrib.get("name", "")
        try:
            parsed = parse_cudem_tile_name(name)
        except ValueError:
            continue
        if parsed["collection"] != collection:
            continue
        size_mb = _read_datasize_mb(ds)
        modified = _read_modified(ds)
        url_path = ds.attrib["urlPath"]
        tiles.append(
            TileRecord(
                name=parsed["name"],
                collection=parsed["collection"],
                source_mode="opendap_netcdf",
                url=f"{THREDDS_DAP_BASE}/{url_path}",
                west=parsed["west"],
                south=parsed["south"],
                east=parsed["east"],
                north=parsed["north"],
                resolution_arcsec=parsed["resolution_arcsec"],
                source="NOAA NCEI THREDDS",
                year=parsed["year"],
                version=parsed["version"],
                size_mb=size_mb,
                modified=modified,
                catalog_url=catalog_url,
                file_server_url=f"{THREDDS_FILE_BASE}/{url_path}",
            )
        )
    return tiles


def fetch_digital_coast_region(region: str, *, timeout: int = 60) -> list[TileRecord]:
    """Fetch one Digital Coast CUDEM GeoTIFF region listing."""

    page_url = f"{DIGITAL_COAST_NINTH_ROOT}/{region}/index.html"
    text = _get_text(page_url, timeout=timeout)
    tiles: list[TileRecord] = []
    for url, name, size_mb in _extract_geotiff_links(text, page_url):
        try:
            parsed = parse_cudem_tile_name(name)
        except ValueError:
            continue
        if parsed["collection"] != "tiled_19as":
            # The current Digital Coast ninth root should only expose 1/9 tiles.
            continue
        tiles.append(
            TileRecord(
                name=parsed["name"],
                collection=parsed["collection"],
                source_mode="https_geotiff",
                url=url,
                west=parsed["west"],
                south=parsed["south"],
                east=parsed["east"],
                north=parsed["north"],
                resolution_arcsec=parsed["resolution_arcsec"],
                source="NOAA Digital Coast HTTPS",
                year=parsed["year"],
                version=parsed["version"],
                size_mb=size_mb,
                modified=None,
                region=region,
                catalog_url=page_url,
            )
        )
    return tiles


def fetch_digital_coast_urllist(
    urllist_url: str, *, timeout: int = 60
) -> list[TileRecord]:
    """Fetch a Digital Coast text file containing GeoTIFF URLs."""

    text = _get_text(urllist_url, timeout=timeout)
    tiles: list[TileRecord] = []
    for raw in text.splitlines():
        url = raw.strip()
        if not url.lower().endswith((".tif", ".tiff")):
            continue
        name = url.rsplit("/", 1)[-1]
        try:
            parsed = parse_cudem_tile_name(name)
        except ValueError:
            continue
        region = _region_from_digital_url(url)
        tiles.append(
            TileRecord(
                name=parsed["name"],
                collection=parsed["collection"],
                source_mode="https_geotiff",
                url=url,
                west=parsed["west"],
                south=parsed["south"],
                east=parsed["east"],
                north=parsed["north"],
                resolution_arcsec=parsed["resolution_arcsec"],
                source="NOAA Digital Coast HTTPS urllist",
                year=parsed["year"],
                version=parsed["version"],
                size_mb=None,
                modified=None,
                region=region,
                catalog_url=urllist_url,
            )
        )
    return tiles


def save_tile_index(index: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_tile_index(path: str | Path) -> list[TileRecord]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tiles = raw["tiles"] if isinstance(raw, dict) and "tiles" in raw else raw
    return [TileRecord.from_dict(item) for item in tiles]


def summarize_index(index: dict | list[TileRecord]) -> dict:
    tiles = index["tiles"] if isinstance(index, dict) else [x.to_dict() for x in index]
    summary: dict[str, dict[str, int]] = {}
    for item in tiles:
        collection = item["collection"]
        source_mode = item["source_mode"]
        summary.setdefault(collection, {})
        summary[collection][source_mode] = summary[collection].get(source_mode, 0) + 1
    return summary


def _get_text(url: str, *, timeout: int) -> str:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.text


def _read_datasize_mb(ds: ET.Element) -> float | None:
    for child in ds:
        if _strip_ns(child.tag) == "dataSize":
            try:
                value = float((child.text or "").strip())
            except ValueError:
                return None
            units = (child.attrib.get("units") or "").lower()
            if units.startswith("k"):
                return value / 1024.0
            if units.startswith("g"):
                return value * 1024.0
            return value
    return None


def _read_modified(ds: ET.Element) -> str | None:
    for child in ds:
        if _strip_ns(child.tag) == "date" and child.attrib.get("type") == "modified":
            return (child.text or "").strip() or None
    return None


def _extract_geotiff_links(text: str, page_url: str) -> list[tuple[str, str, float | None]]:
    links: list[tuple[str, str, float | None]] = []
    pattern = re.compile(
        r'href="(?P<href>[^"]*ncei\d+_[^"]+?\.tif)"[^>]*>(?P<label>[^<]+)</a>'
        r"\s*(?:\((?P<size>[0-9.]+)\s*(?P<unit>[KMG]B|Kbytes|Mbytes|Gbytes)\))?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        href = match.group("href")
        url = urljoin(page_url, href)
        name = match.group("label").strip().split("/")[-1]
        size_mb = _parse_size_mb(match.group("size"), match.group("unit"))
        links.append((url, name, size_mb))
    return links


def _parse_size_mb(value: str | None, unit: str | None) -> float | None:
    if value is None:
        return None
    size = float(value)
    unit = (unit or "MB").lower()
    if unit.startswith("k"):
        return size / 1024.0
    if unit.startswith("g"):
        return size * 1024.0
    return size


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _region_from_digital_url(url: str) -> str | None:
    parts = url.split("/")
    try:
        idx = next(i for i, part in enumerate(parts) if part.startswith("NCEI_"))
    except StopIteration:
        return None
    if idx + 1 < len(parts) - 1:
        return parts[idx + 1]
    return None


def _dedupe_records(records: list[TileRecord]) -> list[TileRecord]:
    best: dict[tuple[str, str], TileRecord] = {}
    for rec in records:
        key = (rec.source_mode, rec.url)
        old = best.get(key)
        if old is None or _record_score(rec) > _record_score(old):
            best[key] = rec
    return list(best.values())


def _record_score(rec: TileRecord) -> tuple[int, int, int]:
    return (
        1 if rec.size_mb is not None else 0,
        1 if rec.region is not None else 0,
        rec.version,
    )
