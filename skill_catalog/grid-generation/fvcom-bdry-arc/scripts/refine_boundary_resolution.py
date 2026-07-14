#!/usr/bin/env python3
"""Build an opt-in adaptive coastal boundary-resolution package."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_bdry_arc import boundary_resolution_config, build_boundary_resolution  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-boundary-loops-gpkg", required=True)
    parser.add_argument("--model-boundary-loop-manifest")
    parser.add_argument("--region-bpoly-json")
    parser.add_argument("--coastline-gpkg")
    parser.add_argument(
        "--reuse-boundary-resolution-manifest",
        help="For v2 only, reuse an accepted adaptive-v1 resolved island topology and rebuild v2 prevention metadata.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--profile",
        choices=("adaptive-coastal-v1", "adaptive-coastal-v2"),
        default="adaptive-coastal-v1",
    )
    parser.add_argument(
        "--passage-max-width-m",
        type=float,
        help="V2-only maximum width for special narrow-passage inventory; wider gaps use ordinary sizing.",
    )
    args = parser.parse_args()
    config = boundary_resolution_config(args.profile)
    if args.passage_max_width_m is not None:
        if args.profile != "adaptive-coastal-v2" or args.passage_max_width_m <= 0.0:
            parser.error("--passage-max-width-m requires adaptive-coastal-v2 and a positive value")
        config = replace(config, passage_max_width_m=float(args.passage_max_width_m))
    manifest = build_boundary_resolution(
        args.model_boundary_loops_gpkg,
        args.model_boundary_loop_manifest,
        args.region_bpoly_json,
        args.coastline_gpkg,
        args.run_dir,
        args.name,
        config,
        reuse_boundary_resolution_manifest=args.reuse_boundary_resolution_manifest,
    )
    print(json.dumps({"final_status": manifest["final_status"], "outputs": manifest["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
