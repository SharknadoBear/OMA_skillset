"""CUDEM tile metadata parsing and selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Iterable, Sequence

COLLECTION_ORDER = ("tiled_19as", "tiled_13as", "tiled_1as", "tiled_3as")

COLLECTION_INFO = {
    "tiled_19as": {
        "label": "1/9 arc-second",
        "resolution_arcsec": 1.0 / 9.0,
        "rank": 0,
    },
    "tiled_13as": {
        "label": "1/3 arc-second",
        "resolution_arcsec": 1.0 / 3.0,
        "rank": 1,
    },
    "tiled_1as": {
        "label": "1 arc-second",
        "resolution_arcsec": 1.0,
        "rank": 2,
    },
    "tiled_3as": {
        "label": "3 arc-second",
        "resolution_arcsec": 3.0,
        "rank": 3,
    },
}

TILE_RE = re.compile(
    r"(?P<prefix>ncei(?P<res>\d+))_"
    r"(?P<lat>[ns]\d+[xX]\d+)_"
    r"(?P<lon>[ew]\d+[xX]\d+)_"
    r"(?P<year>\d{4})v(?P<version>\d+)\.(?P<ext>nc|tif|tiff)$",
    re.IGNORECASE,
)


class NoCoverageError(RuntimeError):
    """Raised when the CUDEM index has no usable coverage for a bbox."""


@dataclass(frozen=True)
class TileRecord:
    """One CUDEM tile advertised by THREDDS or Digital Coast."""

    name: str
    collection: str
    source_mode: str
    url: str
    west: float
    south: float
    east: float
    north: float
    resolution_arcsec: float
    source: str
    year: int
    version: int
    size_mb: float | None = None
    modified: str | None = None
    region: str | None = None
    catalog_url: str | None = None
    file_server_url: str | None = None

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    @property
    def tile_key(self) -> tuple[str, float, float]:
        return (self.collection, round(self.north, 6), round(self.west, 6))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, item: dict) -> "TileRecord":
        return cls(**item)


def parse_cudem_tile_name(name: str) -> dict:
    """Parse a CUDEM tile filename and return collection, bbox, and version."""

    base = name.split("/")[-1]
    match = TILE_RE.match(base)
    if not match:
        raise ValueError(f"Not a recognized CUDEM tile filename: {name}")

    res = match.group("res")
    collection = f"tiled_{res}as"
    if collection not in COLLECTION_INFO:
        raise ValueError(f"Unsupported CUDEM collection in {name}: {collection}")

    lat_hemi, lat_mag = _split_coord_token(match.group("lat"))
    lon_hemi, lon_mag = _split_coord_token(match.group("lon"))

    if lat_hemi == "n":
        north = lat_mag
        south = north - 0.25
    else:
        south = -lat_mag
        north = south + 0.25

    if lon_hemi == "w":
        west = -lon_mag
        east = west + 0.25
    else:
        west = lon_mag
        east = west + 0.25

    return {
        "name": base,
        "collection": collection,
        "west": round(west, 8),
        "south": round(south, 8),
        "east": round(east, 8),
        "north": round(north, 8),
        "resolution_arcsec": COLLECTION_INFO[collection]["resolution_arcsec"],
        "year": int(match.group("year")),
        "version": int(match.group("version")),
        "extension": match.group("ext").lower(),
    }


def normalize_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must be four values: west south east north")
    west, south, east, north = (float(x) for x in bbox)
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise ValueError("CUDEM v1 expects longitudes in [-180, 180].")
    if west >= east:
        raise ValueError("CUDEM v1 does not support dateline-crossing bboxes.")
    if south >= north:
        raise ValueError("bbox south must be less than north.")
    return west, south, east, north


def bbox_intersects(
    a: Sequence[float], b: Sequence[float], *, touch_counts: bool = True
) -> bool:
    aw, asouth, ae, anorth = normalize_bbox(a)
    bw, bsouth, be, bnorth = normalize_bbox(b)
    if touch_counts:
        return aw <= be and ae >= bw and asouth <= bnorth and anorth >= bsouth
    return aw < be and ae > bw and asouth < bnorth and anorth > bsouth


def select_tiles(
    index: Sequence[TileRecord | dict],
    bbox: Sequence[float],
    *,
    resolution: str = "auto",
    max_tiles: int = 48,
    source_preference: Sequence[str] = ("opendap_netcdf", "https_geotiff"),
) -> list[TileRecord]:
    """Select CUDEM tiles for bbox using one resolution tier only."""

    bbox = normalize_bbox(bbox)
    records = [x if isinstance(x, TileRecord) else TileRecord.from_dict(x) for x in index]
    if resolution == "auto":
        collections = COLLECTION_ORDER
    else:
        collections = (_resolution_to_collection(resolution),)

    by_collection: dict[str, list[TileRecord]] = {}
    for rec in records:
        if rec.collection not in collections:
            continue
        if bbox_intersects(rec.bbox, bbox, touch_counts=True):
            by_collection.setdefault(rec.collection, []).append(rec)

    errors: list[str] = []
    for collection in collections:
        candidates = by_collection.get(collection, [])
        if not candidates:
            errors.append(f"{collection}: 0 intersecting tiles")
            continue
        selected = _deduplicate(candidates, source_preference)
        if len(selected) > max_tiles:
            errors.append(
                f"{collection}: {len(selected)} tiles exceeds max_tiles={max_tiles}"
            )
            continue
        selected.sort(key=lambda r: (r.north, r.west, r.source_mode))
        return selected

    details = "; ".join(errors) if errors else "no intersecting CUDEM tiles"
    raise NoCoverageError(
        f"No usable CUDEM coverage for bbox {bbox} with resolution={resolution}: {details}"
    )


def coverage_fraction(records: Sequence[TileRecord], bbox: Sequence[float]) -> float:
    """Approximate bbox coverage fraction from selected tile rectangles."""

    west, south, east, north = normalize_bbox(bbox)
    area = (east - west) * (north - south)
    if area <= 0:
        return 0.0
    try:
        from shapely.geometry import box
        from shapely.ops import unary_union

        target = box(west, south, east, north)
        pieces = [
            box(rec.west, rec.south, rec.east, rec.north).intersection(target)
            for rec in records
        ]
        union = unary_union([p for p in pieces if not p.is_empty])
        return float(min(1.0, max(0.0, union.area / area)))
    except Exception:
        # Conservative fallback; overlaps may inflate, so clamp to 1.
        overlap_area = 0.0
        for rec in records:
            ow = max(west, rec.west)
            oe = min(east, rec.east)
            os = max(south, rec.south)
            on = min(north, rec.north)
            if ow < oe and os < on:
                overlap_area += (oe - ow) * (on - os)
        return float(min(1.0, max(0.0, overlap_area / area)))


def _split_coord_token(token: str) -> tuple[str, float]:
    hemi = token[0].lower()
    if hemi not in {"n", "s", "e", "w"}:
        raise ValueError(f"Bad coordinate token: {token}")
    value = float(token[1:].replace("X", ".").replace("x", "."))
    if not math.isfinite(value):
        raise ValueError(f"Bad coordinate value: {token}")
    return hemi, value


def _resolution_to_collection(value: str) -> str:
    key = value.lower().strip()
    aliases = {
        "auto": "auto",
        "1/9": "tiled_19as",
        "1/9as": "tiled_19as",
        "1/9 arc-second": "tiled_19as",
        "tiled_19as": "tiled_19as",
        "19": "tiled_19as",
        "1/3": "tiled_13as",
        "1/3as": "tiled_13as",
        "1/3 arc-second": "tiled_13as",
        "tiled_13as": "tiled_13as",
        "13": "tiled_13as",
        "1": "tiled_1as",
        "1as": "tiled_1as",
        "tiled_1as": "tiled_1as",
        "3": "tiled_3as",
        "3as": "tiled_3as",
        "tiled_3as": "tiled_3as",
    }
    if key not in aliases or aliases[key] == "auto":
        raise ValueError(
            "resolution must be auto, tiled_19as, tiled_13as, tiled_1as, tiled_3as, "
            "or an alias such as 1/9, 1/3, 1, 3"
        )
    return aliases[key]


def _deduplicate(
    records: Iterable[TileRecord], source_preference: Sequence[str]
) -> list[TileRecord]:
    pref = {name: i for i, name in enumerate(source_preference)}
    best: dict[tuple[str, float, float], TileRecord] = {}
    for rec in records:
        key = rec.tile_key
        old = best.get(key)
        if old is None or _sort_key(rec, pref) > _sort_key(old, pref):
            best[key] = rec
    return list(best.values())


def _sort_key(rec: TileRecord, pref: dict[str, int]) -> tuple[int, int, int, str]:
    # Higher is better. Source preference is inverted so index 0 sorts higher.
    source_rank = -pref.get(rec.source_mode, len(pref))
    return (source_rank, rec.year, rec.version, rec.modified or "")
