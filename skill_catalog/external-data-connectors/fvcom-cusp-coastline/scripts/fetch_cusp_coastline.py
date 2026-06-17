"""Fetch NOAA CUSP shoreline for a bbox and write vector/PNG/JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cusp_coastline.fetch import fetch_cusp_bbox  # noqa: E402
from cusp_coastline.sources import build_region_index, save_region_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    parser.add_argument("--run-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--name", required=True, help="Case/output stem.")
    parser.add_argument("--region", default="auto", help="auto or a region key such as north_atlantic, west, alaska.")
    parser.add_argument("--index", help="Existing cusp_region_index.json path.")
    parser.add_argument("--formats", default="shapefile,gpkg,geojson")
    parser.add_argument("--basemap-provider", default="Esri.WorldImagery")
    parser.add_argument("--allow-no-basemap", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--fallback-policy", default="none", choices=("none", "osm-overpass", "auto"))
    parser.add_argument("--merge-tolerance-m", type=float, default=75.0)
    parser.add_argument("--snap-tolerance-m", type=float, default=100.0)
    parser.add_argument("--min-fallback-fragment-m", type=float, default=100.0)
    parser.add_argument("--refresh-osm", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--client-timeout-s", type=float, default=0.0, help="0 means no hard client timeout.")
    parser.add_argument("--overpass-timeout-s", type=float, default=0.0, help="0 means omit Overpass server timeout.")
    parser.add_argument("--progress-jsonl", help="Progress JSONL path. Defaults under the run directory.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress text; JSONL is still written.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    index_path = Path(args.index) if args.index else run_dir / "cusp_region_index.json"
    if args.rebuild_index or not index_path.exists():
        save_region_index(build_region_index(), index_path)

    result = fetch_cusp_bbox(
        index_path,
        tuple(args.bbox),
        run_dir=run_dir,
        name=args.name,
        region=args.region,
        formats=args.formats,
        basemap_provider=args.basemap_provider,
        allow_no_basemap=args.allow_no_basemap,
        make_plot=not args.no_plot,
        fallback_policy=args.fallback_policy,
        merge_tolerance_m=args.merge_tolerance_m,
        snap_tolerance_m=args.snap_tolerance_m,
        min_fallback_fragment_m=args.min_fallback_fragment_m,
        refresh_osm=args.refresh_osm,
        heartbeat_seconds=args.heartbeat_seconds,
        client_timeout_s=args.client_timeout_s,
        overpass_timeout_s=args.overpass_timeout_s,
        progress_jsonl=args.progress_jsonl,
        quiet=args.quiet,
    )
    print(json.dumps(result.metadata, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
