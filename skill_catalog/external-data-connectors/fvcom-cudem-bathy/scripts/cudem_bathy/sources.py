"""Generic bathymetry source records for CUDEM-first fallback workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import requests

from .catalog import build_tile_index
from .tiles import TileRecord, bbox_intersects, normalize_bbox, parse_cudem_tile_name

THREDDS_DAP_BASE = "https://www.ngdc.noaa.gov/thredds/dodsC"
THREDDS_CRM_CURRENT = "https://www.ngdc.noaa.gov/thredds/catalog/crm/cudem/catalog.xml"
THREDDS_CRM_LEGACY = "https://www.ngdc.noaa.gov/thredds/catalog/crm/catalog.xml"
THREDDS_ETOPO_15S = (
    "https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/"
    "15s/15s_bed_elev_netcdf/catalog.xml"
)
THREDDS_ETOPO_15S_SURFACE = (
    "https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/"
    "15s/15s_surface_elev_netcdf/catalog.xml"
)

CRM_CITATION = (
    "NOAA National Centers for Environmental Information. Coastal Relief Models "
    "(CRMs). NOAA National Centers for Environmental Information. Not for navigation."
)
ETOPO_CITATION = (
    "NOAA National Centers for Environmental Information. 2022: ETOPO 2022 "
    "15 Arc-Second Global Relief Model. DOI: 10.25921/fd45-gt74. Not for navigation."
)

SOURCE_PRIORITY = {"cudem": 1, "crm": 2, "etopo": 3}
SOURCE_ID = {"none": 0, "cudem": 1, "crm": 2, "etopo": 3}
SOURCE_LABELS = {value: key for key, value in SOURCE_ID.items()}

USER_AGENT = "fvcom-cudem-bathy/0.2 (+https://www.noaa.gov/)"
NS = {"t": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}


@dataclass(frozen=True)
class BathySourceRecord:
    """One bathymetry raster source advertised by NOAA services."""

    name: str
    source_name: str
    source_mode: str
    url: str
    west: float
    south: float
    east: float
    north: float
    resolution_arcsec: float
    variable: str
    priority: int
    citation: str
    horizontal_datum: str
    vertical_datum: str
    units: str = "meters"
    size_mb: float | None = None
    modified: str | None = None
    catalog_url: str | None = None
    source_id: int | None = None
    notes: str | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def to_dict(self) -> dict:
        item = asdict(self)
        item["source_id"] = self.source_id or SOURCE_ID.get(self.source_name, -1)
        return item

    @classmethod
    def from_dict(cls, item: dict) -> "BathySourceRecord":
        data = dict(item)
        return cls(**data)


def cudem_tile_to_source(tile: TileRecord) -> BathySourceRecord:
    """Convert an existing CUDEM tile record into a generic source record."""

    return BathySourceRecord(
        name=tile.name,
        source_name="cudem",
        source_mode=tile.source_mode,
        url=tile.url,
        west=tile.west,
        south=tile.south,
        east=tile.east,
        north=tile.north,
        resolution_arcsec=tile.resolution_arcsec,
        variable="auto",
        priority=SOURCE_PRIORITY["cudem"],
        citation=(
            "Cooperative Institute for Research in Environmental Sciences (CIRES) "
            "at the University of Colorado, Boulder. Continuously Updated Digital "
            "Elevation Model (CUDEM). NOAA National Centers for Environmental "
            "Information. Not for navigation."
        ),
        horizontal_datum="NAD83 / EPSG:4269 where provided by CUDEM tiles",
        vertical_datum="NAVD88 where provided by CUDEM tiles",
        size_mb=tile.size_mb,
        modified=tile.modified,
        catalog_url=tile.catalog_url,
        source_id=SOURCE_ID["cudem"],
        notes=tile.source,
    )


def build_bathy_source_index(
    *,
    include_cudem: bool = True,
    include_crm: bool = True,
    include_etopo: bool = True,
    timeout: int = 60,
) -> dict:
    """Build a combined CUDEM, CRM, and ETOPO source index."""

    records: list[BathySourceRecord] = []
    warnings: list[str] = []

    if include_cudem:
        try:
            cudem = build_tile_index(timeout=timeout)
            records.extend(cudem_tile_to_source(TileRecord.from_dict(tile)) for tile in cudem["tiles"])
            warnings.extend(f"CUDEM: {warning}" for warning in cudem.get("warnings", []))
        except Exception as exc:
            warnings.append(f"CUDEM index: {type(exc).__name__}: {exc}")

    if include_crm:
        for catalog_url, selector in (
            (THREDDS_CRM_CURRENT, _is_current_crm),
            (THREDDS_CRM_LEGACY, _is_legacy_crm),
        ):
            try:
                records.extend(fetch_crm_sources(catalog_url, selector=selector, timeout=timeout))
            except Exception as exc:
                warnings.append(f"CRM {catalog_url}: {type(exc).__name__}: {exc}")

    if include_etopo:
        try:
            records.extend(fetch_etopo_sources(timeout=timeout))
        except Exception as exc:
            warnings.append(f"ETOPO 2022: {type(exc).__name__}: {exc}")

    records = _dedupe_sources(records)
    return {
        "version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_priority": SOURCE_PRIORITY,
        "source_ids": SOURCE_ID,
        "sources": {
            "crm_current_catalog": THREDDS_CRM_CURRENT,
            "crm_legacy_catalog": THREDDS_CRM_LEGACY,
            "etopo_15s_catalog": THREDDS_ETOPO_15S,
            "etopo_15s_surface_catalog": THREDDS_ETOPO_15S_SURFACE,
        },
        "warnings": warnings,
        "records": [record.to_dict() for record in records],
    }


def fetch_crm_sources(
    catalog_url: str,
    *,
    selector,
    timeout: int = 60,
) -> list[BathySourceRecord]:
    """Read CRM netCDF datasets from a THREDDS catalog."""

    text = _get_text(catalog_url, timeout=timeout)
    root = ET.fromstring(text)
    records: list[BathySourceRecord] = []
    for ds in root.findall(".//t:dataset[@urlPath]", NS):
        name = ds.attrib.get("name", "")
        url_path = ds.attrib.get("urlPath", "")
        if not selector(name, url_path):
            continue
        url = f"{THREDDS_DAP_BASE}/{url_path}"
        try:
            west, south, east, north, resolution_arcsec, var, horizontal, vertical = _opendap_grid_info(
                url, timeout=timeout
            )
        except Exception:
            west, south, east, north, resolution_arcsec, var, horizontal, vertical = _crm_static_info(name)
        records.append(
            BathySourceRecord(
                name=name,
                source_name="crm",
                source_mode="opendap_netcdf",
                url=url,
                west=west,
                south=south,
                east=east,
                north=north,
                resolution_arcsec=resolution_arcsec,
                variable=var,
                priority=SOURCE_PRIORITY["crm"],
                citation=CRM_CITATION,
                horizontal_datum=horizontal,
                vertical_datum=vertical,
                size_mb=_read_datasize_mb(ds),
                modified=_read_modified(ds),
                catalog_url=catalog_url,
                source_id=SOURCE_ID["crm"],
                notes="NOAA Coastal Relief Model fallback source",
            )
        )
    return records


def fetch_etopo_sources(*, timeout: int = 60) -> list[BathySourceRecord]:
    """Read ETOPO 2022 15 arc-second bedrock elevation tiles."""

    records: list[BathySourceRecord] = []
    for catalog_url, suffix, note in (
        (THREDDS_ETOPO_15S, "_bed.nc", "ETOPO 2022 bedrock elevation fallback"),
        (
            THREDDS_ETOPO_15S_SURFACE,
            "_surface.nc",
            "ETOPO 2022 surface elevation global fallback; ocean bathymetry is unchanged from bed outside ice sheets",
        ),
    ):
        text = _get_text(catalog_url, timeout=timeout)
        root = ET.fromstring(text)
        for ds in root.findall(".//t:dataset[@urlPath]", NS):
            name = ds.attrib.get("name", "")
            if not name.endswith(suffix):
                continue
            url_path = ds.attrib.get("urlPath", "")
            try:
                west, south, east, north = parse_etopo_tile_bbox(name)
            except ValueError:
                west, south, east, north, _res, _var, _h, _v = _opendap_grid_info(
                    f"{THREDDS_DAP_BASE}/{url_path}", timeout=timeout
                )
            records.append(
                BathySourceRecord(
                    name=name,
                    source_name="etopo",
                    source_mode="opendap_netcdf",
                    url=f"{THREDDS_DAP_BASE}/{url_path}",
                    west=west,
                    south=south,
                    east=east,
                    north=north,
                    resolution_arcsec=15.0,
                    variable="z",
                    priority=SOURCE_PRIORITY["etopo"],
                    citation=ETOPO_CITATION,
                    horizontal_datum="WGS84 / EPSG:4326",
                    vertical_datum="EGM2008 height / EPSG:3855",
                    size_mb=_read_datasize_mb(ds),
                    modified=_read_modified(ds),
                    catalog_url=catalog_url,
                    source_id=SOURCE_ID["etopo"],
                    notes=note,
                )
            )
    return records


ETOPO_TILE_RE = re.compile(
    r"ETOPO_2022_v1_15s_(?P<lat>[NS]\d{2})(?P<lon>[EW]\d{3})_(?:bed|surface)\.nc$"
)


def parse_etopo_tile_bbox(name: str) -> tuple[float, float, float, float]:
    """Parse ETOPO 2022 15-degree tile names into west/south/east/north."""

    match = ETOPO_TILE_RE.match(name)
    if not match:
        raise ValueError(f"Not an ETOPO 2022 tile name: {name}")
    lat_token = match.group("lat")
    lon_token = match.group("lon")
    north = float(lat_token[1:]) if lat_token[0] == "N" else -float(lat_token[1:])
    south = max(-90.0, north - 15.0)
    west = float(lon_token[1:]) if lon_token[0] == "E" else -float(lon_token[1:])
    east = min(180.0, west + 15.0)
    return west, south, east, north


def save_bathy_source_index(index: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_bathy_source_index(path: str | Path) -> list[BathySourceRecord]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw["records"] if isinstance(raw, dict) and "records" in raw else raw
    return [BathySourceRecord.from_dict(item) for item in items]


def select_sources(
    records: list[BathySourceRecord],
    bbox: tuple[float, float, float, float],
    *,
    source_names: tuple[str, ...] = ("cudem", "crm", "etopo"),
) -> list[BathySourceRecord]:
    """Return records intersecting a bbox in priority/resolution order."""

    bbox = normalize_bbox(bbox)
    wanted = set(source_names)
    selected = [
        record
        for record in records
        if record.source_name in wanted and _record_intersects_bbox(record, bbox)
    ]
    selected.sort(key=lambda r: (r.priority, r.resolution_arcsec, r.name))
    return selected


def _record_intersects_bbox(
    record: BathySourceRecord, bbox: tuple[float, float, float, float]
) -> bool:
    """Intersect source bboxes, including sources that advertise 0-360 longitudes."""

    west, south, east, north = bbox
    rw, rs, re, rn = record.bbox
    if rw > 180.0 or re > 180.0:
        bw = west + 360.0 if west < 0.0 else west
        be = east + 360.0 if east <= 0.0 else east
    else:
        bw, be = west, east
    return rw < be and re > bw and rs < north and rn > south


def summarize_bathy_index(index: dict | list[BathySourceRecord]) -> dict:
    records = index["records"] if isinstance(index, dict) else [x.to_dict() for x in index]
    summary: dict[str, dict] = {}
    for item in records:
        source = item["source_name"]
        bucket = summary.setdefault(
            source,
            {
                "count": 0,
                "min_resolution_arcsec": None,
                "max_resolution_arcsec": None,
            },
        )
        bucket["count"] += 1
        res = float(item["resolution_arcsec"])
        bucket["min_resolution_arcsec"] = (
            res if bucket["min_resolution_arcsec"] is None else min(bucket["min_resolution_arcsec"], res)
        )
        bucket["max_resolution_arcsec"] = (
            res if bucket["max_resolution_arcsec"] is None else max(bucket["max_resolution_arcsec"], res)
        )
    return summary


def _opendap_grid_info(
    url: str, *, timeout: int
) -> tuple[float, float, float, float, float, str, str, str]:
    html = _get_text(f"{url}.html", timeout=timeout)
    lon_range = _actual_range(html, "lon")
    lat_range = _actual_range(html, "lat")
    var = "z" if re.search(r"\bz\s*:\s*Grid", html) else "auto"
    resolution = _resolution_from_ranges(html, lon_range, lat_range)
    horizontal = "WGS84 / EPSG:4326" if "EPSG&quot;,&quot;4326" in html or "EPSG\",\"4326" in html else "geographic lon/lat"
    vertical = _vertical_from_html(html)
    return (
        min(lon_range),
        min(lat_range),
        max(lon_range),
        max(lat_range),
        resolution,
        var,
        horizontal,
        vertical,
    )


def _actual_range(html: str, name: str) -> tuple[float, float]:
    pattern = re.compile(
        rf"{re.escape(name)}:.*?actual_range:\s*([-+0-9.Ee]+)\s*,\s*([-+0-9.Ee]+)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(html)
    if not match:
        raise ValueError(f"No actual_range for {name}")
    return float(match.group(1)), float(match.group(2))


def _resolution_from_ranges(
    html: str, lon_range: tuple[float, float], lat_range: tuple[float, float]
) -> float:
    lon_size = _array_size(html, "lon")
    lat_size = _array_size(html, "lat")
    lon_res = abs(lon_range[1] - lon_range[0]) / max(1, lon_size - 1) * 3600.0
    lat_res = abs(lat_range[1] - lat_range[0]) / max(1, lat_size - 1) * 3600.0
    return float(max(lon_res, lat_res))


def _array_size(html: str, name: str) -> int:
    dim_match = re.search(rf"dods_{re.escape(name)}\.add_dim\((\d+)\)", html)
    if dim_match:
        return int(dim_match.group(1))
    match = re.search(
        rf"{re.escape(name)}:\s*Array.*?\[\s*{re.escape(name)}\s*=\s*0\.\.(\d+)\s*\]",
        html,
        re.DOTALL,
    )
    if not match:
        return 2
    return int(match.group(1)) + 1


def _vertical_from_html(html: str) -> str:
    match = re.search(r"vert_crs_name:\s*([^<\n]+)", html)
    if match:
        return match.group(1).strip()
    if "EGM2008" in html:
        return "EGM2008 height / EPSG:3855"
    if "Mean Sea Level" in html or "Sea Level" in html:
        return "Mean Sea Level / Sea Level"
    return "source vertical datum; see NOAA metadata"


def _crm_static_info(name: str) -> tuple[float, float, float, float, float, str, str, str]:
    static = {
        "crm_southak.nc": (170.0, 48.5, 230.0, 66.5, 24.0, "z", "WGS84 geographic lon/lat", "Sea Level"),
        "crm_socal_1as_vers2.nc": (-122.0, 30.0, -115.0, 38.0, 1.0, "z", "NAD83 geographic lon/lat", "Mean Sea Level"),
        "crm_socal_3as_vers2.nc": (-122.0, 30.0, -115.0, 38.0, 3.0, "z", "NAD83 geographic lon/lat", "Mean Sea Level"),
    }
    if name in static:
        return static[name]
    vol = re.search(r"crm_vol(?P<num>\d+)(?:_2023)?\.nc", name)
    if vol:
        # Coarse fallback bboxes keep selection usable if THREDDS metadata is temporarily unavailable.
        bounds = {
            "1": (-77.0, 39.0, -65.0, 46.0),
            "2": (-82.0, 31.0, -73.0, 40.0),
            "3": (-89.0, 24.0, -79.0, 31.0),
            "4": (-94.0, 25.0, -86.0, 31.0),
            "5": (-98.0, 25.0, -91.0, 31.0),
            "7": (-161.0, 17.0, -152.0, 25.0),
            "8": (-180.0, 17.0, -161.0, 25.0),
            "9": (-68.0, 17.0, -62.0, 20.0),
            "10": (-161.0, 18.0, -154.0, 23.0),
        }.get(vol.group("num"))
        if bounds:
            return (*bounds, 1.0 if "_2023" in name else 3.0, "z", "WGS84 geographic lon/lat", "EGM2008 or Mean Sea Level")
    raise ValueError(f"No static CRM metadata for {name}")


def _get_text(url: str, *, timeout: int) -> str:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.text


def _read_datasize_mb(ds: ET.Element) -> float | None:
    for child in ds:
        if child.tag.rsplit("}", 1)[-1] == "dataSize":
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
        if child.tag.rsplit("}", 1)[-1] == "date" and child.attrib.get("type") == "modified":
            return (child.text or "").strip() or None
    return None


def _is_current_crm(name: str, url_path: str) -> bool:
    return name.startswith("crm_") and name.endswith("_2023.nc") or name.endswith("_2025.nc")


def _is_legacy_crm(name: str, url_path: str) -> bool:
    return name in {"crm_southak.nc", "crm_socal_1as_vers2.nc", "crm_socal_3as_vers2.nc"}


def _dedupe_sources(records: list[BathySourceRecord]) -> list[BathySourceRecord]:
    best: dict[tuple[str, str], BathySourceRecord] = {}
    for rec in records:
        key = (rec.source_name, rec.url)
        old = best.get(key)
        if old is None or _record_score(rec) > _record_score(old):
            best[key] = rec
    return sorted(best.values(), key=lambda r: (r.priority, r.resolution_arcsec, r.name))


def _record_score(rec: BathySourceRecord) -> tuple[int, int]:
    return (1 if rec.size_mb is not None else 0, 1 if rec.modified else 0)
