#!/usr/bin/env python3
"""Write a boundary-versus-interior size-contract diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fvcom_grid_generation.boundary_size_contract import (
    diagnose_boundary_size_contract,
    write_boundary_size_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether source boundary targets and fvcom_size_field_v4 "
            "request compatible spacing at source boundary vertices."
        )
    )
    parser.add_argument("--boundary-geojson", required=True, type=Path)
    parser.add_argument("--size-field-nc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ratio-tolerance", type=float, default=2.0)
    args = parser.parse_args()
    report = diagnose_boundary_size_contract(
        args.boundary_geojson,
        args.size_field_nc,
        ratio_tolerance=args.ratio_tolerance,
    )
    output = write_boundary_size_contract(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "conflict_vertex_count": report["conflict_vertex_count"],
                "source_boundary_vertex_count": report[
                    "source_boundary_vertex_count"
                ],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
