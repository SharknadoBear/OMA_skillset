from __future__ import annotations

import argparse
from region_bbox.io import read_json, write_json
from region_bbox.mesh_io import read_2dm_bbox


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark RegionBox against a reference 2DM bbox.")
    ap.add_argument("--region-box-json", required=True)
    ap.add_argument("--reference-2dm", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    region = read_json(args.region_box_json)
    ref = read_2dm_bbox(args.reference_2dm)
    write_json(args.output, {"region_box_bbox": region.get("envelope_bbox"), "reference_2dm_bbox": ref})
    print(f"Wrote benchmark: {args.output}")


if __name__ == "__main__":
    main()

