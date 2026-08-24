#!/usr/bin/env python3
"""Fetch and clip GSHHG/GSHHS shoreline polygons for a bbox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gshhs_coastline.fetch import fetch_gshhs_bbox  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="Exact generic clip bbox.")
    selector.add_argument("--model-bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"), help="FVCOM model bbox centered inside an expanded topology footprint.")
    parser.add_argument("--coverage-factor", type=float, default=3.0, help="Centered FVCOM topology coverage factor; minimum 2.0.")
    parser.add_argument("--lookahead-km", type=float, default=0.0, help="Symmetric minimum halo for FVCOM topology feedback.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--resolution", default="auto", choices=("auto", "c", "l", "i", "h", "f"))
    parser.add_argument("--levels", default="1", help="Comma-separated GSHHS levels, default 1.")
    parser.add_argument("--cache-dir", default="Workspace/Preprocessing/fvcom-gshhs-coastline/cache/gshhg")
    parser.add_argument("--formats", default="gpkg,geojson")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--allow-no-basemap", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = fetch_gshhs_bbox(
        tuple(args.bbox) if args.bbox else None,
        model_bbox=tuple(args.model_bbox) if args.model_bbox else None,
        coverage_factor=args.coverage_factor,
        lookahead_km=args.lookahead_km,
        run_dir=args.run_dir,
        name=args.name,
        resolution=args.resolution,
        levels=args.levels,
        cache_dir=args.cache_dir,
        formats=args.formats,
        force_download=args.force_download,
        make_plot=not args.no_plot,
        allow_no_basemap=args.allow_no_basemap,
        quiet=args.quiet,
    )
    print(json.dumps(result.manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
