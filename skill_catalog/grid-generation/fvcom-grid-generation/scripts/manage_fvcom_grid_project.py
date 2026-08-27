#!/usr/bin/env python3
"""Initialize, promote, publish, and validate a standardized FVCOM grid project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.grid_project import init_project, promote, publish, validate  # noqa: E402
from fvcom_grid_generation.open_exterior import GRID_BOUNDARY_GATE_POLICIES  # noqa: E402


def _bool(value: str) -> bool:
    lowered = value.lower()
    if lowered not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return lowered == "true"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--project", required=True, type=Path)
    init.add_argument("--name", required=True)
    promote_p = sub.add_parser("promote")
    promote_p.add_argument("--project", required=True, type=Path)
    promote_p.add_argument("--stage", required=True)
    promote_p.add_argument("--source", required=True, type=Path)
    promote_p.add_argument("--artifact-name", required=True)
    promote_p.add_argument(
        "--generator-manifest",
        type=Path,
        help=(
            "Required when promoting raw_mesh.2dm; must be the project-local "
            "Gmsh Frontal-Delaunay-6 candidate_manifest.json."
        ),
    )
    publication = sub.add_parser("publish")
    publication.add_argument("--project", required=True, type=Path)
    publication.add_argument("--mesh", type=Path)
    for key in ("mesh-quality", "mesh-conditioning", "boundary-nodes", "obc-remap-manifest", "roundtrip-audit", "mesh-review-map"):
        publication.add_argument("--" + key, type=Path)
    publication.add_argument("--fvcom-ready", type=_bool)
    publication.add_argument("--submission-eligible", type=_bool)
    publication.add_argument("--obc-status", default="unknown")
    publication.add_argument("--forcing-status", default="unknown")
    publication.add_argument("--failure", action="append", default=[])
    publication.add_argument("--open-exterior-source", type=Path)
    publication.add_argument("--boundary-resolution-source", type=Path)
    publication.add_argument(
        "--boundary-gate-policy",
        choices=GRID_BOUNDARY_GATE_POLICIES,
        default="strict",
    )
    publication.add_argument(
        "--basemap-provider",
        default="topo",
        choices=("topo", "offline"),
    )
    validation = sub.add_parser("validate")
    validation.add_argument("--project", required=True, type=Path)
    validation.add_argument("--require-submission-ready", action="store_true")
    validation.add_argument("--require-benchmark-ready", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "init":
        result = init_project(args.project, args.name)
    elif args.command == "promote":
        result = promote(
            args.project,
            args.stage,
            args.source,
            args.artifact_name,
            generator_manifest=args.generator_manifest,
        )
    elif args.command == "publish":
        companions = {
            "mesh_quality": args.mesh_quality,
            "mesh_conditioning": args.mesh_conditioning,
            "boundary_nodes": args.boundary_nodes,
            "obc_remap_manifest": args.obc_remap_manifest,
            "roundtrip_audit": args.roundtrip_audit,
            "mesh_review_map": args.mesh_review_map,
        }
        companions = {key: value for key, value in companions.items() if value is not None}
        result = publish(
            args.project,
            mesh=args.mesh,
            companions=companions,
            fvcom_ready=args.fvcom_ready,
            submission_eligible=args.submission_eligible,
            obc_status=args.obc_status,
            forcing_status=args.forcing_status,
            failures=args.failure,
            open_exterior_source=args.open_exterior_source,
            boundary_resolution_source=args.boundary_resolution_source,
            boundary_gate_policy=args.boundary_gate_policy,
            basemap_provider=args.basemap_provider,
        )
    else:
        result = validate(
            args.project,
            require_benchmark_ready=args.require_benchmark_ready,
            require_submission_ready=args.require_submission_ready,
        )
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
