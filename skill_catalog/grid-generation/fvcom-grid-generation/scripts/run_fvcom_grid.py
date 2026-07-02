#!/usr/bin/env python3
"""Run FVCOM grid generation from boundary-loop and bathymetry artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation import GridConfig, run_fvcom_grid  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-text")
    parser.add_argument("--region-bpoly-json")
    parser.add_argument("--offshore-artifacts-json")
    parser.add_argument("--bdry-arc-manifest")
    parser.add_argument("--boundary-loops-gpkg")
    parser.add_argument("--bathy-nc")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=("execute", "test"), default="execute")
    parser.add_argument("--coarse-smoke", action="store_true")
    parser.add_argument("--land-spacing-m", type=float, default=50.0)
    parser.add_argument("--open-spacing-m", type=float, default=3000.0)
    parser.add_argument("--gradation", type=float, default=0.15)
    parser.add_argument("--target-timestep-s", default="auto")
    parser.add_argument("--max-interior-points", type=int, default=80_000)
    parser.add_argument("--refine-iterations", type=int, default=3)
    parser.add_argument("--smooth-iterations", type=int, default=8)
    parser.add_argument(
        "--bathy-fallback-policy",
        choices=("cudem-only", "cudem-crm", "cudem-crm-etopo", "cudem-nbs-crm-etopo"),
        default="cudem-nbs-crm-etopo",
    )
    parser.add_argument(
        "--bathy-resolution-policy",
        choices=("source-priority", "finest"),
        default="source-priority",
    )
    parser.add_argument("--bathy-target-spacing-arcsec", type=float, default=1.0)
    parser.add_argument("--bathy-max-sources", type=int, default=256)
    parser.add_argument("--progress-interval-s", type=float, default=10.0)
    parser.add_argument("--size-field-max-cells", type=int, default=1_500_000)
    args = parser.parse_args()

    manifest = run_fvcom_grid(
        run_dir=args.run_dir,
        name=args.name,
        config=GridConfig(
            mode=args.mode,
            land_spacing_m=args.land_spacing_m,
            open_spacing_m=args.open_spacing_m,
            coarse_smoke=args.coarse_smoke,
            gradation=args.gradation,
            target_timestep_s=args.target_timestep_s,
            max_interior_points=args.max_interior_points,
            refine_iterations=args.refine_iterations,
            smooth_iterations=args.smooth_iterations,
            bathy_fallback_policy=args.bathy_fallback_policy,
            bathy_resolution_policy=args.bathy_resolution_policy,
            bathy_target_spacing_arcsec=args.bathy_target_spacing_arcsec,
            bathy_max_sources=args.bathy_max_sources,
            progress_interval_s=args.progress_interval_s,
            size_field_max_cells=args.size_field_max_cells,
        ),
        request_text=args.request_text,
        region_bpoly_json=args.region_bpoly_json,
        offshore_artifacts_json=args.offshore_artifacts_json,
        bdry_arc_manifest=args.bdry_arc_manifest,
        boundary_loops_gpkg=args.boundary_loops_gpkg,
        bathy_nc=args.bathy_nc,
    )
    print(json.dumps({"final_status": manifest["final_status"], "outputs": manifest["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
