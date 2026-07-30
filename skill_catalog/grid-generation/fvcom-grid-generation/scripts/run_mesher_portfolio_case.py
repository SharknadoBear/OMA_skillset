#!/usr/bin/env python3
"""Run one immutable, generator-neutral FVCOM raw-candidate case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from fvcom_grid_generation.portfolio_case import (
    CANDIDATE_ALIASES,
    DEFAULT_CANDIDATES,
    DEFAULT_HARD_NODE_LIMIT,
    DEFAULT_PREFLIGHT_NODE_LIMIT,
    PortfolioCaseConfig,
    run_portfolio_case,
)


def _workspace_root() -> Path:
    candidates = [
        Path.cwd(),
        *Path.cwd().parents,
        *Path(__file__).resolve().parents,
    ]
    for candidate in candidates:
        if (
            (candidate / "Workspace").is_dir()
            and (candidate / "Agent_skill_dev").is_dir()
        ):
            return candidate
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build one canonical fvcom_size_field_v4 input bundle and run "
            "selected clean-room/Gmsh raw mesh candidates. This command does "
            "not perform or claim common post-conditioning."
        )
    )
    parser.add_argument("--case-manifest", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=_workspace_root())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=sorted(CANDIDATE_ALIASES),
        help=(
            "Candidate aliases. Default policy order: "
            + ", ".join(DEFAULT_CANDIDATES)
            + "."
        ),
    )
    parser.add_argument(
        "--preflight-node-limit",
        type=int,
        default=DEFAULT_PREFLIGHT_NODE_LIMIT,
        help=(
            "Planning threshold for common h_u selection "
            f"(default: {DEFAULT_PREFLIGHT_NODE_LIMIT:,})."
        ),
    )
    parser.add_argument(
        "--hard-node-limit",
        type=int,
        default=DEFAULT_HARD_NODE_LIMIT,
        help=(
            "Maximum delivered nodes for every candidate "
            f"(default: {DEFAULT_HARD_NODE_LIMIT:,})."
        ),
    )
    parser.add_argument(
        "--size-field-max-cells",
        type=int,
        default=1_500_000,
    )
    parser.add_argument("--land-spacing-m", type=float, default=50.0)
    parser.add_argument("--open-spacing-m", type=float, default=3_000.0)
    parser.add_argument("--maximum-size-m", type=float, default=8_000.0)
    parser.add_argument("--gradation", type=float, default=0.10)
    parser.add_argument(
        "--boundary-reconciliation-max-iterations",
        type=int,
        default=8,
        help=(
            "Maximum boundary/2-D-field fixed-point passes before the common "
            "input bundle is rejected."
        ),
    )
    parser.add_argument(
        "--boundary-metric-edge",
        type=float,
        default=1.0,
        help="Target maximum metric length L/h for reconciled boundary edges.",
    )
    parser.add_argument(
        "--boundary-field-compatibility-factor",
        type=float,
        default=1.5,
        help="Hard boundary-target versus rebuilt-field compatibility factor.",
    )
    parser.add_argument(
        "--boundary-target-combination",
        choices=("sampled_field", "minimum"),
        default="sampled_field",
        help=(
            "How delivered boundary targets combine with the sampled 2-D "
            "callback. Default sampled_field follows H exactly; with default "
            "geometry continuity H includes the chord-derived boundary trace. "
            "minimum additionally locks any finer source target."
        ),
    )
    parser.add_argument(
        "--disable-boundary-geometry-continuity",
        action="store_true",
        help=(
            "Disable the default chord-derived boundary target that makes the "
            "first interior ring respond to the realized source geometry."
        ),
    )
    parser.add_argument(
        "--boundary-geometry-metric-ratio",
        type=float,
        default=1.0,
        help=(
            "Desired realized boundary-edge L/h used to derive geometry-aware "
            "targets (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--boundary-trace-samples-per-target",
        type=float,
        default=4.0,
        help=(
            "Deterministic trace samples per local target spacing; endpoints "
            "and midpoints are always included."
        ),
    )
    parser.add_argument(
        "--boundary-trace-nearest-sample-count",
        type=int,
        default=16,
        help=(
            "Initial nearest trace samples; the exact search expands "
            "adaptively until no unseen sample can improve the result."
        ),
    )
    parser.add_argument(
        "--disable-case-budget-spacing-policy",
        action="store_true",
        help=(
            "Use literal land/open spacing arguments instead of resolving the "
            "case bathymetry floor and common budget-selected h_u policy. "
            "The 25 m h_u increment is a numerical search quantum, not "
            "bathymetry raster resolution."
        ),
    )
    parser.add_argument("--slope-elements", type=float, default=10.0)
    parser.add_argument("--coastal-distance-m", type=float, default=25_000.0)
    parser.add_argument(
        "--clean-room-refine-iterations",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--clean-room-smooth-iterations",
        type=int,
        default=8,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = PortfolioCaseConfig(
            preflight_node_limit=args.preflight_node_limit,
            hard_node_limit=args.hard_node_limit,
            size_field_max_cells=args.size_field_max_cells,
            land_spacing_m=args.land_spacing_m,
            open_spacing_m=args.open_spacing_m,
            maximum_size_m=args.maximum_size_m,
            gradation=args.gradation,
            boundary_reconciliation_max_iterations=(
                args.boundary_reconciliation_max_iterations
            ),
            boundary_metric_edge=args.boundary_metric_edge,
            boundary_field_compatibility_factor=(
                args.boundary_field_compatibility_factor
            ),
            boundary_target_combination=args.boundary_target_combination,
            boundary_geometry_continuity=(
                not args.disable_boundary_geometry_continuity
            ),
            boundary_geometry_metric_ratio=(
                args.boundary_geometry_metric_ratio
            ),
            boundary_trace_samples_per_target=(
                args.boundary_trace_samples_per_target
            ),
            boundary_trace_nearest_sample_count=(
                args.boundary_trace_nearest_sample_count
            ),
            use_case_budget_spacing_policy=(
                not args.disable_case_budget_spacing_policy
            ),
            slope_elements=args.slope_elements,
            coastal_distance_m=args.coastal_distance_m,
            clean_room_refine_iterations=args.clean_room_refine_iterations,
            clean_room_smooth_iterations=args.clean_room_smooth_iterations,
        )
        manifest = run_portfolio_case(
            args.case_manifest,
            args.workspace_root,
            args.output_dir,
            candidate_ids=args.candidates,
            config=config,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["status"] == "pass":
        return 0
    if manifest["status"] == "needs_review":
        return 3
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
