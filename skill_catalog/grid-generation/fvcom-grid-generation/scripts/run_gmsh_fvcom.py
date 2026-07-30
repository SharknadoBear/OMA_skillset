#!/usr/bin/env python3
"""Research-only CLI for the six-case Gmsh FVCOM grid experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from fvcom_grid_generation.gmsh_experiment import (
    BudgetConfig,
    check_case_readiness,
    run_gmsh_experiment,
)


def _workspace_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "Workspace").is_dir() and (candidate / "Agent_skill_dev").is_dir():
            return candidate
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or audit the isolated Gmsh 4.15.2 FVCOM mesh experiment. "
            "This does not enable a production backend."
        )
    )
    parser.add_argument("--case-manifest", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=_workspace_root())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--boundary-loop-package", type=Path)
    parser.add_argument("--adaptive-resolution-manifest", type=Path)
    parser.add_argument("--bathymetry-netcdf", type=Path)
    parser.add_argument("--preflight-node-threshold", type=int, default=135_000)
    parser.add_argument("--hard-node-cap", type=int, default=150_000)
    parser.add_argument("--integration-max-cells", type=int, default=250_000)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Audit declared case readiness without starting Gmsh.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.check_only:
        report = check_case_readiness(args.case_manifest, args.workspace_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "ready" else 2
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --check-only is used")
    if args.boundary_loop_package and args.adaptive_resolution_manifest:
        raise SystemExit(
            "Provide at most one of --boundary-loop-package and "
            "--adaptive-resolution-manifest"
        )
    if args.preflight_node_threshold <= 0 or args.hard_node_cap <= 0:
        raise SystemExit("Node thresholds must be positive")
    if args.preflight_node_threshold >= args.hard_node_cap:
        raise SystemExit(
            "--preflight-node-threshold must be smaller than --hard-node-cap"
        )
    config = BudgetConfig(
        max_nodes=args.hard_node_cap,
        preflight_nodes=args.preflight_node_threshold,
        integration_max_cells=args.integration_max_cells,
    )
    try:
        manifest = run_gmsh_experiment(
            args.case_manifest,
            args.workspace_root,
            args.output_dir,
            bathymetry_override=args.bathymetry_netcdf,
            boundary_loop_override=args.boundary_loop_package,
            adaptive_resolution_manifest=args.adaptive_resolution_manifest,
            budget_config=config,
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
    return 0 if manifest["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
