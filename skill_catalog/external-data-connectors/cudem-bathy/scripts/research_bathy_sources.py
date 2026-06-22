#!/usr/bin/env python
"""Write a lightweight regional bathymetry-source discovery report for a bbox."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))


USER_AGENT = "fvcom-cudem-bathy/0.3 (+https://www.noaa.gov/)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    bbox = tuple(float(x) for x in args.bbox)
    candidates = build_candidate_report(bbox, timeout=args.timeout)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "case": args.name,
        "bbox_wsen": list(bbox),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "This report is advisory. Regional candidates are not injected into the "
            "fetch stack until Bear/agent review confirms datum, quality, and access method."
        ),
        "candidates": candidates,
    }
    json_path = run_dir / f"{args.name}_regional_bathy_research.json"
    md_path = run_dir / f"{args.name}_regional_bathy_research.md"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(data), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


def build_candidate_report(bbox: tuple[float, float, float, float], *, timeout: int) -> list[dict]:
    west, south, east, north = bbox
    is_se_ak = west < -128.0 and east > -141.0 and north > 54.0 and south < 60.5
    candidates = [
        {
            "name": "NOAA NBS BlueTopo",
            "source_type": "default_candidate",
            "access": "AWS S3/HTTPS GeoTIFF tiles using BlueTopo tile-scheme GeoPackage",
            "url": "https://registry.opendata.aws/noaa-bathymetry/",
            "expected_resolution": "4 m, 16 m, and coarser by tile",
            "vertical_datum": "NAVD88 where provided by tile metadata",
            "automation_difficulty": "low",
            "default_stack_role": "first-class nbs_bluetopo source",
            "status": check_url("https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/?list-type=2&prefix=BlueTopo/_BlueTopo_Tile_Scheme/", timeout),
        },
        {
            "name": "NOAA NBS S-102",
            "source_type": "future_candidate",
            "access": "AWS S3/HTTPS HDF5 S-102 products",
            "url": "https://noaa-s102-pds.s3.amazonaws.com/README.html",
            "expected_resolution": "product-dependent; often high-resolution gridded bathymetry",
            "vertical_datum": "product metadata; not harmonized by this skill",
            "automation_difficulty": "medium-high because S-102/HDF5 parsing is more complex than GeoTIFF",
            "default_stack_role": "documented future source, not automatic v1 fetch",
            "status": check_url("https://noaa-s102-pds.s3.amazonaws.com/README.html", timeout),
        },
        {
            "name": "NCEI BAG bathymetry image service",
            "source_type": "regional_candidate",
            "access": "ArcGIS ImageServer exportImage",
            "url": "https://gis.ngdc.noaa.gov/arcgis/rest/services/bag_bathymetry/ImageServer",
            "expected_resolution": "service mosaic dependent",
            "vertical_datum": "mixed survey/product metadata",
            "automation_difficulty": "medium; good for discovery/preview, needs careful source provenance",
            "default_stack_role": "review-only candidate",
            "status": check_url("https://gis.ngdc.noaa.gov/arcgis/rest/services/bag_bathymetry/ImageServer?f=json", timeout),
        },
        {
            "name": "NCEI multibeam mosaic image service",
            "source_type": "regional_candidate",
            "access": "ArcGIS ImageServer exportImage",
            "url": "https://gis.ngdc.noaa.gov/arcgis/rest/services/multibeam_mosaics/multibeam_mosaic_combined/ImageServer",
            "expected_resolution": "service mosaic dependent",
            "vertical_datum": "mixed survey/product metadata",
            "automation_difficulty": "medium; useful for comparison but not yet production fill",
            "default_stack_role": "review-only candidate",
            "status": check_url(
                "https://gis.ngdc.noaa.gov/arcgis/rest/services/multibeam_mosaics/multibeam_mosaic_combined/ImageServer?f=json",
                timeout,
            ),
        },
        {
            "name": "NOAA Fisheries Alaska bathymetry",
            "source_type": "regional_candidate" if is_se_ak else "regional_note",
            "access": "ArcGIS MapServer when available",
            "url": "https://alaskafisheries.noaa.gov/arcgis/rest/services/bathy_40m/MapServer",
            "expected_resolution": "about 20-40 m product family",
            "vertical_datum": "source metadata review required",
            "automation_difficulty": "medium; endpoint availability has been intermittent",
            "default_stack_role": "review-only candidate for Alaska",
            "status": check_url("https://alaskafisheries.noaa.gov/arcgis/rest/services/bathy_40m/MapServer?f=json", timeout),
        },
        {
            "name": "SE Alaska 8 arc-second coastal DEM",
            "source_type": "regional_candidate" if is_se_ak else "regional_note",
            "access": "NCEI OPeNDAP/NetCDF/WCS",
            "url": "https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ngdc.mgg.dem%3A575",
            "expected_resolution": "8 arc-second",
            "vertical_datum": "Mean Higher High Water in source metadata",
            "automation_difficulty": "medium; useful regional fallback but datum differs",
            "default_stack_role": "review-only candidate for SE-AK",
            "status": check_url(
                "https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ngdc.mgg.dem%3A575",
                timeout,
            ),
        },
    ]
    return candidates


def check_url(url: str, timeout: int) -> dict:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT}, stream=True)
        return {"ok": resp.ok, "status_code": resp.status_code, "content_type": resp.headers.get("content-type")}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def render_markdown(data: dict) -> str:
    lines = [
        f"# Regional Bathymetry Source Review: {data['case']}",
        "",
        f"- Generated UTC: `{data['generated_at_utc']}`",
        f"- Bbox WSEN: `{data['bbox_wsen']}`",
        f"- Decision rule: {data['decision']}",
        "",
        "## Candidates",
        "",
    ]
    for item in data["candidates"]:
        lines.extend(
            [
                f"### {item['name']}",
                "",
                f"- Type: `{item['source_type']}`",
                f"- Access: {item['access']}",
                f"- URL: {item['url']}",
                f"- Expected resolution: {item['expected_resolution']}",
                f"- Vertical datum: {item['vertical_datum']}",
                f"- Automation difficulty: {item['automation_difficulty']}",
                f"- Stack role: {item['default_stack_role']}",
                f"- Endpoint status: `{item['status']}`",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
