"""Generic bathymetry source records for CUDEM-first fallback workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
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
NBS_BUCKET_ROOT = "https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com"
NBS_BLUETOPO_SCHEME_PREFIX = "BlueTopo/_BlueTopo_Tile_Scheme/"
NBS_BLUETOPO_SCHEME_LIST = (
    f"{NBS_BUCKET_ROOT}/?list-type=2&prefix={NBS_BLUETOPO_SCHEME_PREFIX}"
)

CRM_CITATION = (
    "NOAA National Centers for Environmental Information. Coastal Relief Models "
    "(CRMs). NOAA National Centers for Environmental Information. Not for navigation."
)
ETOPO_CITATION = (
    "NOAA National Centers for Environmental Information. 2022: ETOPO 2022 "
    "15 Arc-Second Global Relief Model. DOI: 10.25921/fd45-gt74. Not for navigation."
)
NBS_BLUETOPO_CITATION = (
    "NOAA Office of Coast Survey. National Bathymetric Source: BlueTopo. "
    "Public AWS Open Data Registry bucket noaa-ocs-nationalbathymetry-pds. "
    "Not for navigation."
)

SOURCE_PRIORITY = {"cudem": 1, "nbs_bluetopo": 2, "crm": 3, "etopo": 4}
SOURCE_ID = {"none": 0, "cudem": 1, "nbs_bluetopo": 2, "crm": 3, "etopo": 4, "regional_candidate": 5}
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
    resolution_m: float | None = None
    size_mb: float | None = None
    modified: str | None = None
    catalog_url: str | None = None
    source_id: int | None = None
    notes: str | None = None
    utm_zone: str | None = None
    rat_url: str | None = None
    checksum: str | None = None
    rat_checksum: str | None = None
    delivered_date: str | None = None

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
        allowed = {field.name for field in fields(cls)}
        data = {key: value for key, value in data.items() if key in allowed}
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
    include_nbs: bool = True,
    include_crm: bool = True,
    include_etopo: bool = True,
    timeout: int = 60,
) -> dict:
    """Build a combined CUDEM, NBS BlueTopo, CRM, and ETOPO source index."""

    records: list[BathySourceRecord] = []
    warnings: list[str] = []

    if include_cudem:
        try:
            cudem = build_tile_index(timeout=timeout)
            records.extend(cudem_tile_to_source(TileRecord.from_dict(tile)) for tile in cudem["tiles"])
            warnings.extend(f"CUDEM: {warning}" for warning in cudem.get("warnings", []))
        except Exception as exc:
            warnings.append(f"CUDEM index: {type(exc).__name__}: {exc}")

    if include_nbs:
        try:
            records.extend(fetch_nbs_bluetopo_sources(timeout=timeout))
        except Exception as exc:
            warnings.append(f"NBS BlueTopo index: {type(exc).__name__}: {exc}")

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
            "nbs_bluetopo_bucket": NBS_BUCKET_ROOT,
            "nbs_bluetopo_tile_scheme_prefix": NBS_BLUETOPO_SCHEME_PREFIX,
            "crm_current_catalog": THREDDS_CRM_CURRENT,
            "crm_legacy_catalog": THREDDS_CRM_LEGACY,
            "etopo_15s_catalog": THREDDS_ETOPO_15S,
            "etopo_15s_surface_catalog": THREDDS_ETOPO_15S_SURFACE,
        },
        "warnings": warnings,
        "records": [record.to_dict() for record in records],
    }


def fetch_nbs_bluetopo_sources(*, timeout: int = 60) -> list[BathySourceRecord]:
    """Read the current NOAA NBS BlueTopo tile-scheme GeoPackage from AWS."""

    key, modified, size_mb = _latest_bluetopo_scheme(timeout=timeout)
    url = f"{NBS_BUCKET_ROOT}/{key}"
    with tempfile.TemporaryDirectory(prefix="fvcom_bluetopo_") as tmp:
        gpkg_path = Path(tmp) / Path(key).name
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        gpkg_path.write_bytes(resp.content)
        try:
            import geopandas as gpd

            gdf = gpd.read_file(gpkg_path)
        except Exception as exc:
            raise RuntimeError(
                "Could not read BlueTopo tile scheme GeoPackage. Install geopandas/pyogrio."
            ) from exc

    records: list[BathySourceRecord] = []
    for row in gdf.itertuples(index=False):
        geom = getattr(row, "geometry", None)
        url_value = _row_value(row, "GeoTIFF_Link")
        if geom is None or geom.is_empty or not url_value:
            continue
        url_value = _normalize_nbs_url(str(url_value))
        west, south, east, north = (float(x) for x in geom.bounds)
        resolution_m = _parse_resolution_m(_row_value(row, "Resolution"))
        resolution_arcsec = _meters_to_arcsec_equivalent(resolution_m)
        tile = str(_row_value(row, "tile") or Path(str(url_value)).stem)
        records.append(
            BathySourceRecord(
                name=tile,
                source_name="nbs_bluetopo",
                source_mode="nbs_bluetopo_geotiff",
                url=url_value,
                west=west,
                south=south,
                east=east,
                north=north,
                resolution_arcsec=resolution_arcsec,
                variable="Elevation",
                priority=SOURCE_PRIORITY["nbs_bluetopo"],
                citation=NBS_BLUETOPO_CITATION,
                horizontal_datum="NAD83 / UTM zone provided by BlueTopo tile metadata",
                vertical_datum="NAVD88 where provided by BlueTopo tile metadata",
                resolution_m=resolution_m,
                size_mb=None,
                modified=modified,
                catalog_url=url,
                source_id=SOURCE_ID["nbs_bluetopo"],
                notes=(
                    "NOAA NBS BlueTopo GeoTIFF source. Bands commonly include "
                    "Elevation, Uncertainty, and Contributor."
                ),
                utm_zone=str(_row_value(row, "UTM") or "") or None,
                rat_url=_normalize_nbs_url(_row_value(row, "RAT_Link")),
                checksum=_optional_text(_row_value(row, "GeoTIFF_SHA256_Checksum")),
                rat_checksum=_optional_text(_row_value(row, "RAT_SHA256_Checksum")),
                delivered_date=_optional_text(_row_value(row, "Delivered_Date")),
            )
        )
    return records


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
    source_names: tuple[str, ...] = ("cudem", "nbs_bluetopo", "crm", "etopo"),
    resolution_policy: str = "source-priority",
) -> list[BathySourceRecord]:
    """Return records intersecting a bbox in priority/resolution order."""

    bbox = normalize_bbox(bbox)
    wanted = set(source_names)
    selected = [
        record
        for record in records
        if record.source_name in wanted and _record_intersects_bbox(record, bbox)
    ]
    selected.sort(key=lambda r: source_sort_key(r, resolution_policy=resolution_policy))
    return selected


def native_resolution_m(record: BathySourceRecord) -> float:
    """Return the best native spacing estimate in meters for source arbitration."""

    if record.resolution_m is not None:
        return float(record.resolution_m)
    return float(record.resolution_arcsec) * 30.87


def source_sort_key(
    record: BathySourceRecord, *, resolution_policy: str = "source-priority"
) -> tuple[float, float, int, str]:
    """Sort source records for conservative or finest-available workflows."""

    policy = resolution_policy.lower().strip()
    if policy == "source-priority":
        return (float(record.priority), native_resolution_m(record), 0, record.name)
    if policy == "finest":
        return (0.0, native_resolution_m(record), int(record.priority), record.name)
    raise ValueError("resolution_policy must be source-priority or finest")


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
                "min_resolution_m": None,
                "max_resolution_m": None,
            },
        )
        bucket["count"] += 1
        res = float(item["resolution_arcsec"])
        res_m = item.get("resolution_m")
        bucket["min_resolution_arcsec"] = (
            res if bucket["min_resolution_arcsec"] is None else min(bucket["min_resolution_arcsec"], res)
        )
        bucket["max_resolution_arcsec"] = (
            res if bucket["max_resolution_arcsec"] is None else max(bucket["max_resolution_arcsec"], res)
        )
        if res_m is not None:
            res_m = float(res_m)
            bucket["min_resolution_m"] = (
                res_m if bucket["min_resolution_m"] is None else min(bucket["min_resolution_m"], res_m)
            )
            bucket["max_resolution_m"] = (
                res_m if bucket["max_resolution_m"] is None else max(bucket["max_resolution_m"], res_m)
            )
    return summary


def _latest_bluetopo_scheme(*, timeout: int) -> tuple[str, str | None, float | None]:
    text = _get_text(NBS_BLUETOPO_SCHEME_LIST, timeout=timeout)
    root = ET.fromstring(text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    candidates: list[tuple[str, str | None, float | None]] = []
    for contents in root.findall("s3:Contents", ns):
        key = _s3_child_text(contents, "Key", ns)
        if not key or not key.lower().endswith(".gpkg"):
            continue
        modified = _s3_child_text(contents, "LastModified", ns)
        size_text = _s3_child_text(contents, "Size", ns)
        size_mb = float(size_text) / 1024.0 / 1024.0 if size_text else None
        candidates.append((key, modified, size_mb))
    if not candidates:
        raise RuntimeError("No BlueTopo tile-scheme GeoPackage found in NOAA NBS AWS bucket.")
    return sorted(candidates, key=lambda item: (item[1] or "", item[0]))[-1]


def _s3_child_text(element: ET.Element, name: str, ns: dict[str, str]) -> str | None:
    child = element.find(f"s3:{name}", ns)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _row_value(row, name: str):
    return getattr(row, name, None)


def _parse_resolution_m(value) -> float | None:
    if value is None:
        return None
    match = re.search(r"([-+0-9.]+)", str(value))
    if not match:
        return None
    return float(match.group(1))


def _normalize_nbs_url(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("s3://noaa-ocs-nationalbathymetry-pds/"):
        key = text.split("s3://noaa-ocs-nationalbathymetry-pds/", 1)[1]
        return f"{NBS_BUCKET_ROOT}/{key}"
    return text


def _optional_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _meters_to_arcsec_equivalent(resolution_m: float | None) -> float:
    if resolution_m is None:
        return 1.0
    return float(resolution_m) / 30.87


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
