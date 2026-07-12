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
    parser.add_argument("--boundary-resolution-manifest")
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
    parser.add_argument("--boundary-resolution-profile", choices=("legacy", "adaptive-coastal-v1"), default="legacy")
    parser.add_argument(
        "--bathy-gradient-policy",
        choices=("auto", "global", "coastal", "off"),
        default="auto",
        help="Bathymetric-gradient sizing policy; auto uses coastal gating for adaptive boundaries and global behavior for legacy grids.",
    )
    parser.add_argument("--coastal-gradient-distance-m", type=float, default=25_000.0)
    parser.add_argument("--regional-spring-relaxation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spring-relax-iterations", type=int, default=20)
    parser.add_argument("--spring-relax-quality-threshold", type=float, default=0.40)
    parser.add_argument("--spring-relax-min-angle-deg", type=float, default=28.0)
    parser.add_argument("--spring-relax-ring-layers", type=int, default=3)
    parser.add_argument("--spring-relax-shape-weight", type=float, default=0.20)
    parser.add_argument("--thin-triangle-repair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--thin-triangle-quality-threshold", type=float, default=0.25)
    parser.add_argument("--thin-triangle-min-angle-deg", type=float, default=20.0)
    parser.add_argument("--thin-triangle-max-passes", type=int, default=2)
    parser.add_argument("--thin-triangle-max-flips", type=int, default=200)
    parser.add_argument("--thin-triangle-max-insertions", type=int, default=50)
    parser.add_argument(
        "--area-transition-relaxation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply guarded local spring patches to excessive adjacent-area transitions after thin repair.",
    )
    parser.add_argument("--area-transition-max-patches", type=int, default=12)
    parser.add_argument("--area-transition-area-change-threshold", type=float, default=0.50)
    parser.add_argument("--area-transition-target-gradient-threshold", type=float, default=0.10)
    parser.add_argument(
        "--postprocess-profile",
        choices=("none", "rpw2019", "projection-medium"),
        default="none",
    )
    parser.add_argument(
        "--postprocess-boundary-policy",
        choices=("protect-all", "protect-open"),
        default="protect-all",
    )
    parser.add_argument("--postprocess-max-passes", type=int, default=8)
    parser.add_argument("--postprocess-connectivity-limit", default="auto")
    args = parser.parse_args()
    if args.postprocess_profile != "none":
        parser.error("Integrated cleanup is disabled; run scripts/postprocess_fvcom_mesh.py on the completed .2dm")
    connectivity_limit = (
        None
        if str(args.postprocess_connectivity_limit).lower() == "auto"
        else int(args.postprocess_connectivity_limit)
    )

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
            boundary_resolution_profile=args.boundary_resolution_profile,
            bathymetry_gradient_policy=args.bathy_gradient_policy,
            coastal_gradient_distance_m=args.coastal_gradient_distance_m,
            regional_spring_relaxation=args.regional_spring_relaxation,
            spring_relax_iterations=args.spring_relax_iterations,
            spring_relax_quality_threshold=args.spring_relax_quality_threshold,
            spring_relax_min_angle_deg=args.spring_relax_min_angle_deg,
            spring_relax_ring_layers=args.spring_relax_ring_layers,
            spring_relax_shape_weight=args.spring_relax_shape_weight,
            thin_triangle_repair=args.thin_triangle_repair,
            thin_triangle_quality_threshold=args.thin_triangle_quality_threshold,
            thin_triangle_min_angle_deg=args.thin_triangle_min_angle_deg,
            thin_triangle_max_passes=args.thin_triangle_max_passes,
            thin_triangle_max_flips=args.thin_triangle_max_flips,
            thin_triangle_max_insertions=args.thin_triangle_max_insertions,
            area_transition_relaxation=args.area_transition_relaxation,
            area_transition_max_patches=args.area_transition_max_patches,
            area_transition_area_change_threshold=args.area_transition_area_change_threshold,
            area_transition_target_gradient_threshold=args.area_transition_target_gradient_threshold,
            postprocess_profile=args.postprocess_profile,
            postprocess_boundary_policy=args.postprocess_boundary_policy,
            postprocess_max_passes=args.postprocess_max_passes,
            postprocess_connectivity_limit=connectivity_limit,
        ),
        request_text=args.request_text,
        region_bpoly_json=args.region_bpoly_json,
        offshore_artifacts_json=args.offshore_artifacts_json,
        bdry_arc_manifest=args.bdry_arc_manifest,
        boundary_loops_gpkg=args.boundary_loops_gpkg,
        boundary_resolution_manifest=args.boundary_resolution_manifest,
        bathy_nc=args.bathy_nc,
    )
    print(json.dumps({"final_status": manifest["final_status"], "outputs": manifest["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
