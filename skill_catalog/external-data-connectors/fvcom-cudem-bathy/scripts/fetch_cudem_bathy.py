"""Fetch NOAA CUDEM bathymetry for a bbox and write NetCDF/PNG/JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cudem_bathy.catalog import build_tile_index, save_tile_index  # noqa: E402
from cudem_bathy.fetch import fetch_cudem_bbox  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    parser.add_argument("--run-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--name", required=True, help="Case/output stem.")
    parser.add_argument("--index", help="Existing cudem_tile_index.json path.")
    parser.add_argument("--resolution", default="auto", help="auto, tiled_19as, tiled_13as, tiled_1as, tiled_3as.")
    parser.add_argument("--max-tiles", type=int, default=48)
    parser.add_argument(
        "--target-spacing-arcsec",
        type=float,
        default=3.0,
        help="Output mosaic spacing. Source still uses selected CUDEM resolution.",
    )
    parser.add_argument(
        "--source-preference",
        default="opendap_netcdf,https_geotiff",
        help="Comma-separated source priority.",
    )
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    index_path = Path(args.index) if args.index else run_dir / "cudem_tile_index.json"
    if args.rebuild_index or not index_path.exists():
        index = build_tile_index()
        save_tile_index(index, index_path)

    result = fetch_cudem_bbox(
        index_path,
        args.bbox,
        run_dir=run_dir,
        name=args.name,
        resolution=args.resolution,
        max_tiles=args.max_tiles,
        target_spacing_arcsec=args.target_spacing_arcsec,
        source_preference=tuple(x.strip() for x in args.source_preference.split(",") if x.strip()),
        make_plot=not args.no_plot,
    )
    print(json.dumps(result.metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
