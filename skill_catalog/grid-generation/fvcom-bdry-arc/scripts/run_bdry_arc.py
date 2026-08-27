#!/usr/bin/env python3
"""Create an FVCOM boundary-arc package from RegionBPoly and coastline data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_bdry_arc import BdryArcConfig, run_bdry_arc  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-bpoly-json", required=True)
    parser.add_argument("--offshore-artifacts-json", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--coastline-gpkg")
    parser.add_argument("--fetch-coastline", action="store_true")
    parser.add_argument("--coastline-source", default="gshhs", choices=("gshhs", "generic-gpkg", "cusp-legacy"))
    parser.add_argument("--mode", default="execute", choices=("execute", "test"))
    parser.add_argument("--target-resolution-m", type=float, default=250.0)
    parser.add_argument("--review-depth", default="auto", choices=("auto", "fast", "full"))
    parser.add_argument("--coastline-buffer-km", type=float, default=10.0)
    parser.add_argument("--seed-mode", default="auto", choices=("auto", "manual-json"))
    parser.add_argument("--manual-seed-json")
    parser.add_argument("--cusp-skill-dir")
    parser.add_argument("--gshhs-skill-dir")
    parser.add_argument("--gshhs-resolution", default="f", choices=("auto", "c", "l", "i", "h", "f"))
    parser.add_argument("--gshhs-levels", default="1")
    parser.add_argument("--gshhs-coverage-factor", type=float, default=3.0, help="Centered topology source coverage factor; minimum 2.0.")
    parser.add_argument("--fallback-policy", default="auto", choices=("none", "osm-overpass", "auto"))
    parser.add_argument("--topology-mode", default="gshhs-vector", choices=("gshhs-vector", "island-loop", "iterative-raster", "vector-only"))
    parser.add_argument("--raster-resolution-m", type=float)
    parser.add_argument("--max-topology-iterations", type=int, default=4)
    parser.add_argument("--convergence-area-frac", type=float, default=0.01)
    parser.add_argument("--convergence-anchor-m", type=float)
    parser.add_argument("--progress-interval-s", type=float, default=30.0)
    parser.add_argument("--heuristic-mode", default="auto", choices=("auto", "memory", "unknown"), help="auto uses text memory in execute and disables text-only routing in test.")
    parser.add_argument("--topology-time-budget-s", type=float, default=900.0, help="Maximum seconds to spend evaluating full-resolution topology candidates before returning needs_review.")
    parser.add_argument(
        "--boundary-resolution-profile",
        default="adaptive-coastal-v2",
        choices=("adaptive-coastal-v2",),
        help="Deprecated compatibility selector; Adaptive v2 is the sole active profile.",
    )
    parser.add_argument(
        "--expected-obc-count",
        type=int,
        help="Requested delivered OBC-chain count; defaults to the RegionBPoly/offshore contract.",
    )
    parser.add_argument(
        "--obc-placement-policy",
        default="offshore-first",
        choices=("offshore-first", "mouth-first"),
        help="Prefer a complete offshore OBC, or try a compact mouth OBC first for coastal estuaries.",
    )
    parser.add_argument(
        "--frame-clip-policy",
        default="reject-unintended",
        choices=("reject-unintended", "report-only"),
        help="Reject residual open-exterior length or retain diagnostic report-only behavior.",
    )
    parser.add_argument(
        "--residual-boundary-policy",
        default="solid-default",
        choices=("solid-default", "strict-reject"),
        help="Classify residual frame-water segments as solid by default, or retain the historical strict length veto.",
    )
    parser.add_argument(
        "--frame-clip-tolerance-m",
        type=float,
        help="Absolute residual-frame tolerance; default max(250 m, 0.05 * target resolution).",
    )
    args = parser.parse_args()

    config = BdryArcConfig(
        mode=args.mode,
        target_resolution_m=args.target_resolution_m,
        review_depth=args.review_depth,
        coastline_buffer_km=args.coastline_buffer_km,
        seed_mode=args.seed_mode,
        manual_seed_json=args.manual_seed_json,
        fetch_coastline=args.fetch_coastline,
        coastline_source=args.coastline_source,
        cusp_skill_dir=args.cusp_skill_dir,
        gshhs_skill_dir=args.gshhs_skill_dir,
        gshhs_resolution=args.gshhs_resolution,
        gshhs_levels=args.gshhs_levels,
        gshhs_coverage_factor=args.gshhs_coverage_factor,
        fallback_policy=args.fallback_policy,
        topology_mode=args.topology_mode,
        raster_resolution_m=args.raster_resolution_m,
        max_topology_iterations=args.max_topology_iterations,
        convergence_area_frac=args.convergence_area_frac,
        convergence_anchor_m=args.convergence_anchor_m,
        progress_interval_s=args.progress_interval_s,
        heuristic_mode=args.heuristic_mode,
        topology_time_budget_s=args.topology_time_budget_s,
        boundary_resolution_profile=args.boundary_resolution_profile,
        expected_obc_count=args.expected_obc_count,
        frame_clip_policy=args.frame_clip_policy,
        residual_boundary_policy=args.residual_boundary_policy,
        frame_clip_tolerance_m=args.frame_clip_tolerance_m,
        obc_placement_policy=args.obc_placement_policy,
    )
    manifest = run_bdry_arc(
        region_bpoly_json=args.region_bpoly_json,
        offshore_artifacts_json=args.offshore_artifacts_json,
        run_dir=args.run_dir,
        name=args.name,
        coastline_gpkg=args.coastline_gpkg,
        config=config,
    )
    print(json.dumps({"final_status": manifest["final_status"], "outputs": manifest["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
