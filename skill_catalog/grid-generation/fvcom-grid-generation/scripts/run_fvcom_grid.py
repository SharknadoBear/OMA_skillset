#!/usr/bin/env python3
"""Run FVCOM grid generation from boundary-loop and bathymetry artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation import GridConfig, run_fvcom_grid  # noqa: E402
from fvcom_grid_generation.node_budget import (  # noqa: E402
    DEFAULT_HARD_NODE_LIMIT,
    DEFAULT_MAX_INTERIOR_POINTS,
    DEFAULT_NODE_BUDGET_STOP_FRACTION,
)
from fvcom_grid_generation.systematic_v6_policy import (  # noqa: E402
    FIXED_GATE_POLICIES,
)


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
    parser.add_argument("--gradation", type=float, default=0.20)
    parser.add_argument("--slope-elements", type=float, default=10.0)
    parser.add_argument("--coastal-distance-m", type=float, default=25_000.0)
    parser.add_argument("--hydraulic-elements-across-min", type=float, default=3.0)
    parser.add_argument("--hydraulic-elements-across-max", type=float, default=8.0)
    parser.add_argument("--hydraulic-max-width-m", type=float, default=20_000.0)
    parser.add_argument("--hydraulic-bank-angle-deg", type=float, default=110.0)
    parser.add_argument("--hydraulic-longitudinal-gradation", type=float, default=0.10)
    parser.add_argument("--obc-hold-distance-m", type=float, default=10_000.0)
    parser.add_argument("--obc-transition-distance-m", type=float, default=60_000.0)
    parser.add_argument("--target-timestep-s", default="auto")
    parser.add_argument(
        "--max-interior-points",
        type=int,
        default=DEFAULT_MAX_INTERIOR_POINTS,
        help=(
            "Clean-room interior seed ceiling "
            f"(default: {DEFAULT_MAX_INTERIOR_POINTS:,})."
        ),
    )
    parser.add_argument(
        "--max-total-nodes",
        type=int,
        default=DEFAULT_HARD_NODE_LIMIT,
        help=(
            "Maximum delivered mesh node count "
            f"(default: {DEFAULT_HARD_NODE_LIMIT:,})."
        ),
    )
    parser.add_argument(
        "--node-budget-stop-fraction",
        type=float,
        default=DEFAULT_NODE_BUDGET_STOP_FRACTION,
        help=(
            "Pre-triangulation planning fraction of the hard cap "
            f"(default: {DEFAULT_NODE_BUDGET_STOP_FRACTION:.2f})."
        ),
    )
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
    parser.add_argument(
        "--boundary-resolution-profile",
        choices=("legacy", "adaptive-coastal-v1", "adaptive-coastal-v2"),
        default="legacy",
    )
    parser.add_argument("--regional-spring-relaxation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spring-relax-iterations", type=int, default=20)
    parser.add_argument("--spring-relax-quality-threshold", type=float, default=0.40)
    parser.add_argument("--spring-relax-min-angle-deg", type=float, default=28.0)
    parser.add_argument("--spring-relax-ring-layers", type=int, default=3)
    parser.add_argument("--spring-relax-shape-weight", type=float, default=0.20)
    parser.add_argument("--thin-triangle-repair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--thin-repair-profile",
        choices=("guarded-v1", "systematic-v2", "systematic-v3", "systematic-v5", "systematic-v6", "none"),
        default="guarded-v1",
        help=(
            "Extreme-tail topology profile. Systematic v2/v3, "
            "connectivity-restricted v5, and coupled exact-zero v6 are "
            "opt-in; auto behavior is unchanged."
        ),
    )
    parser.add_argument(
        "--systematic-v3-obc-policy",
        choices=("preserve", "redistribute"),
        default="redistribute",
        help="Preserve the OBC node count or allow source-arc redistribution when systematic-v3 is selected.",
    )
    parser.add_argument("--systematic-v5-total-iterations", type=int, default=1000)
    parser.add_argument("--systematic-v5-max-cycles", type=int, default=6)
    parser.add_argument("--systematic-v5-max-burst", type=int, default=250)
    parser.add_argument("--systematic-v5-thin-trigger", type=int, default=25)
    parser.add_argument("--systematic-v5-checkpoint-interval", type=int, default=10)
    parser.add_argument("--systematic-v5-wall-time-s", type=float, default=21600.0)
    parser.add_argument(
        "--systematic-v5-connectivity-restriction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable research-only persistent allowed-edge closure under systematic-v5.",
    )
    parser.add_argument(
        "--systematic-v5-max-connectivity-transactions",
        type=int,
        default=32,
    )
    parser.add_argument("--systematic-v6-total-iterations", type=int, default=1000)
    parser.add_argument("--systematic-v6-max-cycles", type=int, default=12)
    parser.add_argument("--systematic-v6-max-closure-rounds", type=int, default=8)
    parser.add_argument("--systematic-v6-max-burst", type=int, default=100)
    parser.add_argument("--systematic-v6-checkpoint-interval", type=int, default=10)
    parser.add_argument("--systematic-v6-wall-time-s", type=float, default=28800.0)
    parser.add_argument("--systematic-v6-final-audit-reserve-s", type=float, default=3600.0)
    parser.add_argument(
        "--systematic-v6-gate-policy",
        choices=FIXED_GATE_POLICIES,
        default="strict-v6",
        help=(
            "Fixed whole-mesh V6 closure policy. Adaptive policy ladders "
            "remain research-driver orchestration."
        ),
    )
    parser.add_argument(
        "--systematic-v6-passage-removal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable research-only authorized whole-passage removal. "
            "Disabled by default."
        ),
    )
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
        "--conditioning-profile",
        choices=(
            "auto",
            "minimal-topology-v1",
            "guarded-v1",
            "aggressive-local-v2",
            "none",
        ),
        default="auto",
        help=(
            "Auto selects minimal-topology-v1. Explicit legacy profiles "
            "retain their existing spring/thin/area behavior."
        ),
    )
    parser.add_argument(
        "--minimal-conditioning-wall-time-s",
        type=float,
        default=3_600.0,
    )
    parser.add_argument("--aggressive-conditioning-rounds", type=int, default=4)
    parser.add_argument(
        "--aggressive-boundary-edit-policy",
        choices=("kind-aware-envelope", "split-only", "none"),
        default="kind-aware-envelope",
    )
    parser.add_argument("--aggressive-max-prunes-per-round", type=int, default=500)
    parser.add_argument("--aggressive-max-valence-repairs-per-round", type=int, default=500)
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
            slope_elements=args.slope_elements,
            coastal_distance_m=args.coastal_distance_m,
            hydraulic_elements_across_min=args.hydraulic_elements_across_min,
            hydraulic_elements_across_max=args.hydraulic_elements_across_max,
            hydraulic_max_width_m=args.hydraulic_max_width_m,
            hydraulic_bank_angle_deg=args.hydraulic_bank_angle_deg,
            hydraulic_longitudinal_gradation=(
                args.hydraulic_longitudinal_gradation
            ),
            obc_hold_distance_m=args.obc_hold_distance_m,
            obc_transition_distance_m=args.obc_transition_distance_m,
            target_timestep_s=args.target_timestep_s,
            max_interior_points=args.max_interior_points,
            max_total_nodes=args.max_total_nodes,
            node_budget_stop_fraction=args.node_budget_stop_fraction,
            refine_iterations=args.refine_iterations,
            smooth_iterations=args.smooth_iterations,
            bathy_fallback_policy=args.bathy_fallback_policy,
            bathy_resolution_policy=args.bathy_resolution_policy,
            bathy_target_spacing_arcsec=args.bathy_target_spacing_arcsec,
            bathy_max_sources=args.bathy_max_sources,
            progress_interval_s=args.progress_interval_s,
            size_field_max_cells=args.size_field_max_cells,
            boundary_resolution_profile=args.boundary_resolution_profile,
            regional_spring_relaxation=args.regional_spring_relaxation,
            spring_relax_iterations=args.spring_relax_iterations,
            spring_relax_quality_threshold=args.spring_relax_quality_threshold,
            spring_relax_min_angle_deg=args.spring_relax_min_angle_deg,
            spring_relax_ring_layers=args.spring_relax_ring_layers,
            spring_relax_shape_weight=args.spring_relax_shape_weight,
            thin_triangle_repair=args.thin_triangle_repair,
            thin_repair_profile=args.thin_repair_profile,
            systematic_v3_obc_policy=args.systematic_v3_obc_policy,
            systematic_v5_total_iterations=args.systematic_v5_total_iterations,
            systematic_v5_max_cycles=args.systematic_v5_max_cycles,
            systematic_v5_max_burst=args.systematic_v5_max_burst,
            systematic_v5_thin_trigger=args.systematic_v5_thin_trigger,
            systematic_v5_checkpoint_interval=args.systematic_v5_checkpoint_interval,
            systematic_v5_wall_time_s=args.systematic_v5_wall_time_s,
            systematic_v5_connectivity_restriction=(
                args.systematic_v5_connectivity_restriction
            ),
            systematic_v5_max_connectivity_transactions=(
                args.systematic_v5_max_connectivity_transactions
            ),
            systematic_v6_total_iterations=args.systematic_v6_total_iterations,
            systematic_v6_max_cycles=args.systematic_v6_max_cycles,
            systematic_v6_max_closure_rounds=args.systematic_v6_max_closure_rounds,
            systematic_v6_max_burst=args.systematic_v6_max_burst,
            systematic_v6_checkpoint_interval=args.systematic_v6_checkpoint_interval,
            systematic_v6_wall_time_s=args.systematic_v6_wall_time_s,
            systematic_v6_final_audit_reserve_s=(
                args.systematic_v6_final_audit_reserve_s
            ),
            systematic_v6_gate_policy=args.systematic_v6_gate_policy,
            systematic_v6_passage_removal=(
                args.systematic_v6_passage_removal
            ),
            thin_triangle_quality_threshold=args.thin_triangle_quality_threshold,
            thin_triangle_min_angle_deg=args.thin_triangle_min_angle_deg,
            thin_triangle_max_passes=args.thin_triangle_max_passes,
            thin_triangle_max_flips=args.thin_triangle_max_flips,
            thin_triangle_max_insertions=args.thin_triangle_max_insertions,
            area_transition_relaxation=args.area_transition_relaxation,
            area_transition_max_patches=args.area_transition_max_patches,
            area_transition_area_change_threshold=args.area_transition_area_change_threshold,
            area_transition_target_gradient_threshold=args.area_transition_target_gradient_threshold,
            conditioning_profile=args.conditioning_profile,
            minimal_conditioning_wall_time_s=(
                args.minimal_conditioning_wall_time_s
            ),
            aggressive_conditioning_rounds=args.aggressive_conditioning_rounds,
            aggressive_boundary_edit_policy=args.aggressive_boundary_edit_policy,
            aggressive_max_prunes_per_round=args.aggressive_max_prunes_per_round,
            aggressive_max_valence_repairs_per_round=args.aggressive_max_valence_repairs_per_round,
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
