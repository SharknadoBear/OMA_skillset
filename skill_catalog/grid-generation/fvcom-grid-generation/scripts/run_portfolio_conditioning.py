#!/usr/bin/env python3
"""Apply the generator-neutral FVCOM portfolio conditioner to one raw 2DM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.portfolio_conditioning import (  # noqa: E402
    PortfolioConditioningConfig,
    condition_portfolio_mesh,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--size-field-nc", required=True, type=Path)
    parser.add_argument("--bathymetry-nc", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name")
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
    )
    parser.add_argument("--boundary-contract-json", type=Path)
    parser.add_argument("--source-boundary-metadata-json", type=Path)
    parser.add_argument(
        "--scientific-input-status",
        choices=("valid", "invalid"),
        default="valid",
    )
    parser.add_argument("--scientific-input-note")
    parser.add_argument("--wall-time-s", type=float, default=3_600.0)
    parser.add_argument("--primary-rounds", type=int, default=4)
    parser.add_argument("--terminal-rounds", type=int, default=1)
    parser.add_argument("--max-prunes-per-round", type=int, default=500)
    parser.add_argument(
        "--max-valence-repairs-per-round",
        type=int,
        default=500,
    )
    parser.add_argument("--max-valence-flip-batch", type=int, default=64)
    parser.add_argument(
        "--max-valence-cluster-merges-per-round",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--max-valence-l-over-h-count-increase",
        type=int,
        default=0,
    )
    parser.add_argument("--micro-relax-cycles", type=int, default=3)
    parser.add_argument(
        "--area-transition-max-patches",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--area-transition-area-change-threshold",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--area-transition-target-gradient-threshold",
        type=float,
        default=0.10,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    boundary_contract = (
        json.loads(args.boundary_contract_json.read_text(encoding="utf-8"))
        if args.boundary_contract_json is not None
        else None
    )
    source_boundary_metadata = (
        json.loads(
            args.source_boundary_metadata_json.read_text(encoding="utf-8")
        )
        if args.source_boundary_metadata_json is not None
        else None
    )
    config = PortfolioConditioningConfig(
        conditioning_profile=str(args.conditioning_profile),
        primary_rounds=int(args.primary_rounds),
        terminal_rounds=int(args.terminal_rounds),
        max_prunes_per_round=int(args.max_prunes_per_round),
        max_valence_repairs_per_round=int(
            args.max_valence_repairs_per_round
        ),
        max_valence_flip_batch=int(args.max_valence_flip_batch),
        max_valence_cluster_merges_per_round=int(
            args.max_valence_cluster_merges_per_round
        ),
        max_valence_l_over_h_count_increase=int(
            args.max_valence_l_over_h_count_increase
        ),
        micro_relax_cycles=int(args.micro_relax_cycles),
        area_transition_max_patches=int(
            args.area_transition_max_patches
        ),
        area_transition_raw_threshold=float(
            args.area_transition_area_change_threshold
        ),
        area_transition_target_gradient_threshold=float(
            args.area_transition_target_gradient_threshold
        ),
        wall_time_s=float(args.wall_time_s),
    )
    try:
        report = condition_portfolio_mesh(
            args.mesh,
            args.size_field_nc,
            args.bathymetry_nc,
            args.output_dir,
            name=args.name,
            config=config,
            boundary_contract=boundary_contract,
            source_boundary_metadata=source_boundary_metadata,
            scientific_input_valid=(
                str(args.scientific_input_status) == "valid"
            ),
            scientific_input_note=args.scientific_input_note,
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
    print(
        json.dumps(
            {
                "status": report["status"],
                "conditioned_2dm": report["outputs"]["conditioned_2dm"]["path"],
                "conditioning_report": report["outputs"][
                    "conditioning_report_json"
                ],
                "quality_accepted": report["quality_accepted"],
                "minimal_local_debt_closed": report[
                    "minimal_local_debt_closed"
                ],
                "fvcom_ready": report["fvcom_ready"],
                "quality_failure_taxonomy": report[
                    "quality_failure_taxonomy"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
