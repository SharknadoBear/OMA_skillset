"""NOAA NSDE CUSP source registry and region selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

from .progress import ProgressReporter, normalize_timeout


NSDE_URL = "https://nsde.ngs.noaa.gov/"
INPORT_URL = "https://www.fisheries.noaa.gov/inport/item/60812"


@dataclass(frozen=True)
class CuspRegion:
    """One official NSDE regional CUSP ZIP source."""

    key: str
    label: str
    url: str
    bbox_wsen: tuple[float, float, float, float]
    production: bool = True


CUSP_REGIONS: tuple[CuspRegion, ...] = (
    CuspRegion("alaska", "Alaska", "https://geodesy.noaa.gov/dist_shoreline/Alaska.zip", (-180.0, 50.0, -128.0, 73.5)),
    CuspRegion(
        "gulf_of_america",
        "Gulf of America",
        "https://geodesy.noaa.gov/dist_shoreline/Gulf_Of_America.zip",
        (-99.5, 18.0, -80.0, 31.5),
    ),
    CuspRegion(
        "north_atlantic",
        "North Atlantic",
        "https://geodesy.noaa.gov/dist_shoreline/North_Atlantic.zip",
        (-82.0, 35.0, -58.0, 47.5),
    ),
    CuspRegion(
        "pacific_islands",
        "Pacific Islands",
        "https://geodesy.noaa.gov/dist_shoreline/Pacific_Islands.zip",
        (-180.0, -20.0, -130.0, 30.0),
    ),
    CuspRegion(
        "southeast_caribbean",
        "Southeast Caribbean",
        "https://geodesy.noaa.gov/dist_shoreline/Southeast_Caribbean.zip",
        (-69.5, 16.0, -63.0, 19.5),
    ),
    CuspRegion("west", "West", "https://geodesy.noaa.gov/dist_shoreline/West.zip", (-130.0, 30.0, -114.0, 49.5)),
    CuspRegion(
        "great_lakes",
        "Great Lakes",
        "https://geodesy.noaa.gov/dist_shoreline/Great_Lakes.zip",
        (-93.0, 41.0, -75.0, 50.0),
    ),
    CuspRegion(
        "planned_projects",
        "Planned Projects",
        "https://geodesy.noaa.gov/dist_shoreline/CUSP_IN_PROGRESS.zip",
        (-180.0, -20.0, -58.0, 73.5),
        production=False,
    ),
)


def normalize_region_key(label_or_url_stem: str) -> str:
    """Return a stable lower-snake region key."""

    stem = Path(label_or_url_stem).stem
    key = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_").lower()
    if key == "gulf_of_america":
        return key
    return key


def _head(
    url: str,
    *,
    client_timeout_s: float | int | None = 0,
    reporter: ProgressReporter | None = None,
) -> dict[str, str | int | float | None]:
    request = Request(url, method="HEAD")
    timeout = normalize_timeout(client_timeout_s)
    kwargs = {"timeout": timeout} if timeout is not None else {}
    heartbeat = reporter.background_heartbeat("cusp-index", "waiting for CUSP source HEAD", url=url) if reporter else None
    if heartbeat:
        heartbeat.__enter__()
    try:
        response_ctx = urlopen(request, **kwargs)
    finally:
        if heartbeat:
            heartbeat.__exit__(None, None, None)
    with response_ctx as response:
        length = response.headers.get("Content-Length")
        return {
            "status": int(response.status),
            "content_length": int(length) if length and length.isdigit() else None,
            "size_mb": round(int(length) / 1048576.0, 3) if length and length.isdigit() else None,
            "content_type": response.headers.get("Content-Type"),
            "last_modified": response.headers.get("Last-Modified"),
        }


def scrape_nsde_region_urls(
    nsde_url: str = NSDE_URL,
    *,
    client_timeout_s: float | int | None = 0,
    reporter: ProgressReporter | None = None,
) -> dict[str, str]:
    """Return region ZIP URLs advertised by the NSDE page."""

    timeout = normalize_timeout(client_timeout_s)
    kwargs = {"timeout": timeout} if timeout is not None else {}
    if reporter:
        reporter.event("cusp-index", "scraping NSDE CUSP region URLs", url=nsde_url, client_timeout_s=timeout)
    heartbeat = reporter.background_heartbeat("cusp-index", "waiting for NSDE source page", url=nsde_url) if reporter else None
    if heartbeat:
        heartbeat.__enter__()
    try:
        response_ctx = urlopen(nsde_url, **kwargs)
    finally:
        if heartbeat:
            heartbeat.__exit__(None, None, None)
    with response_ctx as response:
        html = response.read().decode("utf-8", errors="replace")
    matches = re.findall(r'href="(https://geodesy\.noaa\.gov/dist_shoreline/[^"]+\.zip)"', html)
    if reporter:
        reporter.event("cusp-index", "found NSDE CUSP region URLs", count=len(matches))
    return {normalize_region_key(url): url for url in matches}


def build_region_index(
    *,
    include_head: bool = True,
    client_timeout_s: float | int | None = 0,
    reporter: ProgressReporter | None = None,
) -> dict[str, object]:
    """Build a serializable CUSP region index with optional HEAD metadata."""

    scraped: dict[str, str]
    try:
        scraped = scrape_nsde_region_urls(client_timeout_s=client_timeout_s, reporter=reporter)
    except (OSError, URLError):
        scraped = {}

    regions: list[dict[str, object]] = []
    for index, region in enumerate(CUSP_REGIONS, start=1):
        url = scraped.get(region.key, region.url)
        item = asdict(region)
        item["url"] = url
        item["source_page"] = NSDE_URL
        item["citation"] = "National Geodetic Survey, NOAA NGS Continually Updated Shoreline Product (CUSP)"
        item["metadata_url"] = INPORT_URL
        item["zip_name"] = Path(url).name
        if include_head:
            try:
                if reporter:
                    reporter.heartbeat("cusp-index", "checking CUSP region source metadata", processed=index, total=len(CUSP_REGIONS), region=region.key)
                item["http"] = _head(url, client_timeout_s=client_timeout_s, reporter=reporter)
            except Exception as exc:  # pragma: no cover - network-specific
                item["http"] = {"error": str(exc)}
        regions.append(item)

    return {
        "schema_version": 1,
        "source_page": NSDE_URL,
        "metadata_url": INPORT_URL,
        "regions": regions,
    }


def save_region_index(index: dict[str, object], path: str | Path) -> Path:
    """Write a region index JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return path


def load_region_index(path: str | Path) -> dict[str, object]:
    """Read a region index JSON."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _intersection_area(a: Iterable[float], b: Iterable[float]) -> float:
    aw, as_, ae, an = tuple(a)
    bw, bs, be, bn = tuple(b)
    width = max(0.0, min(ae, be) - max(aw, bw))
    height = max(0.0, min(an, bn) - max(as_, bs))
    return width * height


def validate_bbox(bbox: Iterable[float]) -> tuple[float, float, float, float]:
    """Validate and normalize a bbox tuple."""

    values = tuple(float(x) for x in bbox)
    if len(values) != 4:
        raise ValueError("bbox must have four values: W S E N")
    west, south, east, north = values
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise ValueError(f"longitude values out of range: {values}")
    if not (-90.0 <= south <= 90.0 and -90.0 <= north <= 90.0):
        raise ValueError(f"latitude values out of range: {values}")
    if west >= east or south >= north:
        raise ValueError(f"bbox must satisfy west < east and south < north: {values}")
    return values


def select_region(index: dict[str, object], bbox: Iterable[float], requested: str = "auto") -> dict[str, object]:
    """Select the best production CUSP region for a bbox."""

    bbox = validate_bbox(bbox)
    regions = list(index.get("regions", []))
    if requested != "auto":
        key = normalize_region_key(requested)
        for region in regions:
            if region.get("key") == key or normalize_region_key(str(region.get("label", ""))) == key:
                return region
        raise ValueError(f"requested region {requested!r} not found in index")

    candidates = []
    for region in regions:
        if not region.get("production", True):
            continue
        area = _intersection_area(region.get("bbox_wsen", ()), bbox)
        if area > 0.0:
            candidates.append((area, region))
    if not candidates:
        raise ValueError(f"no production CUSP region intersects bbox {bbox}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]
