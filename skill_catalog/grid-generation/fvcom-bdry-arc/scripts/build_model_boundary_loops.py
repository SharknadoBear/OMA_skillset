#!/usr/bin/env python3
"""Build continuous model boundary loops from an fvcom-bdry-arc package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_bdry_arc import build_model_boundary_loops  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdry-arc-gpkg", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--target-resolution-m", type=float)
    parser.add_argument("--min-island-area-m2", type=float, default=0.0)
    parser.add_argument("--mode", default="execute", choices=("execute", "test"))
    args = parser.parse_args()

    manifest = build_model_boundary_loops(
        bdry_arc_gpkg=args.bdry_arc_gpkg,
        manifest_json=args.manifest_json,
        run_dir=args.run_dir,
        name=args.name,
        target_resolution_m=args.target_resolution_m,
        min_island_area_m2=args.min_island_area_m2,
        mode=args.mode,
    )
    print(json.dumps({"final_status": manifest["final_status"], "outputs": manifest["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
