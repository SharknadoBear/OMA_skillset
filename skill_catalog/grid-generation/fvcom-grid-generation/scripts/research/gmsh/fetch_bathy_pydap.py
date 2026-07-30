#!/usr/bin/env python3
"""Run the CUDEM bathymetry fetcher with a recorded PyDAP OPeNDAP fallback.

The generic connector remains authoritative.  This research wrapper only
works around Windows netCDF4/curl failures by retrying the same bounded NOAA
OPeNDAP subset through PyDAP; it does not alter source order or fill policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SKILL_CATALOG = Path(__file__).resolve().parents[5]
CUDEM_SCRIPTS = (
    SKILL_CATALOG / "external-data-connectors" / "cudem-bathy" / "scripts"
)
sys.path.insert(0, str(CUDEM_SCRIPTS))

from cudem_bathy import bathy_fetch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument(
        "--fallback-policy",
        default="cudem-nbs-crm-etopo",
        choices=(
            "cudem-only",
            "cudem-crm",
            "cudem-crm-etopo",
            "cudem-nbs-crm-etopo",
        ),
    )
    parser.add_argument(
        "--resolution-policy",
        default="source-priority",
        choices=("source-priority", "finest"),
    )
    parser.add_argument("--target-spacing-arcsec", type=float, required=True)
    parser.add_argument("--max-sources", type=int, default=256)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    original_open_dataset = bathy_fetch.xr.open_dataset
    pydap_fallback_urls: list[str] = []

    def resilient_open_dataset(*open_args, **open_kwargs):
        try:
            return original_open_dataset(*open_args, **open_kwargs)
        except (OSError, RuntimeError):
            if not open_args:
                raise
            retry_kwargs = dict(open_kwargs)
            retry_kwargs["engine"] = "pydap"
            dataset = original_open_dataset(*open_args, **retry_kwargs)
            pydap_fallback_urls.append(str(open_args[0]))
            return dataset

    bathy_fetch.xr.open_dataset = resilient_open_dataset
    try:
        result = bathy_fetch.fetch_bathy_bbox(
            args.index,
            args.bbox,
            run_dir=args.run_dir,
            name=args.name,
            fallback_policy=args.fallback_policy,
            resolution_policy=args.resolution_policy,
            target_spacing_arcsec=args.target_spacing_arcsec,
            max_sources=args.max_sources,
            make_plot=not args.no_plot,
        )
    finally:
        bathy_fetch.xr.open_dataset = original_open_dataset

    metadata = dict(result.metadata)
    metadata["research_transport_wrapper"] = {
        "netcdf4_first": True,
        "pydap_fallback_used": bool(pydap_fallback_urls),
        "pydap_fallback_urls": sorted(set(pydap_fallback_urls)),
        "source_and_subset_unchanged": True,
    }
    result.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
