#!/usr/bin/env python
"""Plot boundary/field transition diagnostics for one raw portfolio mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fvcom_grid_generation.raw_transition_diagnostics import (  # noqa: E402
    DEFAULT_MAX_PLOT_TRIANGLES,
    write_raw_transition_diagnostics,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write read-only whole-mesh and boundary/first-ring L/h maps "
            "for a raw mesher-portfolio candidate."
        )
    )
    parser.add_argument("--mesh-2dm", required=True)
    parser.add_argument("--canonical-boundary-geojson", required=True)
    parser.add_argument("--canonical-size-field-nc", required=True)
    parser.add_argument("--quality-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title")
    parser.add_argument(
        "--max-plot-triangles",
        type=int,
        default=DEFAULT_MAX_PLOT_TRIANGLES,
    )
    parser.add_argument("--transition-graph-rings", type=int, default=2)
    parser.add_argument(
        "--boundary-match-tolerance-m",
        type=float,
        default=0.05,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = write_raw_transition_diagnostics(
        args.mesh_2dm,
        args.canonical_boundary_geojson,
        args.canonical_size_field_nc,
        args.quality_json,
        args.output_dir,
        title=args.title,
        max_plot_triangles=args.max_plot_triangles,
        transition_graph_rings=args.transition_graph_rings,
        boundary_match_tolerance_m=args.boundary_match_tolerance_m,
    )
    print(
        json.dumps(
            {
                "diagnostic_status": report["diagnostic_status"],
                "report_path": report["report_path"],
                "report_sha256": report["report_sha256"],
                "whole_mesh_map": report["artifacts"]["whole_mesh_map"][
                    "path"
                ],
                "boundary_first_ring_map": report["artifacts"][
                    "boundary_first_ring_map"
                ]["path"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
