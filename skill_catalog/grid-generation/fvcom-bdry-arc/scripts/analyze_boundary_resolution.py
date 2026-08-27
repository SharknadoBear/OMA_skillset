#!/usr/bin/env python3
"""Analyze island shapes and boundary-resolution burden without changing inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_bdry_arc import analyze_boundary_resolution, boundary_resolution_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-boundary-loops-gpkg", required=True)
    parser.add_argument("--region-bpoly-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = analyze_boundary_resolution(
        args.model_boundary_loops_gpkg,
        region_bpoly_json=args.region_bpoly_json,
        config=boundary_resolution_config(),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "island_count": report["island_count"], "class_counts": report["class_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
