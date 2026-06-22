#!/usr/bin/env python
"""Fetch CUDEM/NBS-first bathymetry with CRM and ETOPO fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cudem_bathy.bathy_fetch import fetch_bathy_bbox  # noqa: E402
from cudem_bathy.sources import build_bathy_source_index, save_bathy_source_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--index", required=True, help="Combined bathymetry source index JSON.")
    parser.add_argument(
        "--fallback-policy",
        default="cudem-nbs-crm-etopo",
        choices=("cudem-only", "cudem-crm", "cudem-crm-etopo", "cudem-nbs-crm-etopo"),
    )
    parser.add_argument(
        "--resolution-policy",
        default="source-priority",
        choices=("source-priority", "finest"),
        help=(
            "source-priority keeps CUDEM/NBS/CRM/ETOPO family order; finest lets "
            "the finest usable local native resolution win across sources."
        ),
    )
    parser.add_argument(
        "--target-spacing-arcsec",
        type=float,
        default=3.0,
        help="Output grid spacing. Use 0 for native finest selected source spacing.",
    )
    parser.add_argument("--max-sources", type=int, default=256)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    index_path = Path(args.index)
    if args.rebuild_index or not index_path.exists():
        index = build_bathy_source_index()
        save_bathy_source_index(index, index_path)

    result = fetch_bathy_bbox(
        index_path,
        args.bbox,
        run_dir=args.run_dir,
        name=args.name,
        fallback_policy=args.fallback_policy,
        resolution_policy=args.resolution_policy,
        target_spacing_arcsec=None if args.target_spacing_arcsec == 0 else args.target_spacing_arcsec,
        max_sources=args.max_sources,
        make_plot=not args.no_plot,
    )
    print(json.dumps(result.metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
