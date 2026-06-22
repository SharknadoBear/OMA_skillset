"""Build a local NOAA CUDEM tile index JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cudem_bathy.catalog import (  # noqa: E402
    DEFAULT_DIGITAL_COAST_REGIONS,
    build_tile_index,
    DEFAULT_DIGITAL_COAST_URLLISTS,
    save_tile_index,
    summarize_index,
)
from cudem_bathy.tiles import COLLECTION_ORDER  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output cudem_tile_index.json path.")
    parser.add_argument(
        "--collections",
        default=",".join(COLLECTION_ORDER),
        help="Comma-separated THREDDS collections to index.",
    )
    parser.add_argument(
        "--digital-regions",
        default=",".join(DEFAULT_DIGITAL_COAST_REGIONS),
        help="Comma-separated Digital Coast regions to index.",
    )
    parser.add_argument(
        "--digital-urllists",
        default=",".join(DEFAULT_DIGITAL_COAST_URLLISTS),
        help="Comma-separated Digital Coast urllist text files to index.",
    )
    parser.add_argument("--no-thredds", action="store_true", help="Skip THREDDS XML catalogs.")
    parser.add_argument(
        "--no-digital-coast", action="store_true", help="Skip Digital Coast GeoTIFF listings."
    )
    parser.add_argument("--no-urllists", action="store_true", help="Skip Digital Coast urllist text files.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    args = parser.parse_args()

    index = build_tile_index(
        collections=tuple(x.strip() for x in args.collections.split(",") if x.strip()),
        include_thredds=not args.no_thredds,
        include_digital_coast=not args.no_digital_coast,
        include_urllists=not args.no_urllists,
        digital_regions=tuple(x.strip() for x in args.digital_regions.split(",") if x.strip()),
        digital_urllists=tuple(x.strip() for x in args.digital_urllists.split(",") if x.strip()),
        timeout=args.timeout,
    )
    save_tile_index(index, args.output)
    print(json.dumps({"output": str(Path(args.output)), "summary": summarize_index(index)}, indent=2))
    if index.get("warnings"):
        print(json.dumps({"warnings": index["warnings"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
