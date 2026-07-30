#!/usr/bin/env python3
"""Plan, execute, and compare generator-neutral FVCOM mesher candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))

from fvcom_grid_generation.mesher_bakeoff import (  # noqa: E402
    COMMON_CONDITIONED,
    RAW,
    BakeoffContractError,
    compare_results,
    execute_stage,
    load_declaration,
    lock_input_bundle,
    plan_bakeoff,
    write_comparison,
    write_input_bundle,
)


def _json_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    lock = subparsers.add_parser(
        "lock-bundle", help="Hash and freeze one common mesher input bundle."
    )
    lock.add_argument("--case-id", required=True)
    lock.add_argument("--boundary", required=True, type=Path)
    lock.add_argument("--bathymetry", required=True, type=Path)
    lock.add_argument("--size-field", required=True, type=Path)
    lock.add_argument("--projection-json", required=True, type=Path)
    lock.add_argument("--node-budget-json", required=True, type=Path)
    lock.add_argument("--output", required=True, type=Path)

    plan = subparsers.add_parser(
        "plan", help="Create a fresh immutable bakeoff run plan."
    )
    plan.add_argument("--declaration", required=True, type=Path)
    plan.add_argument("--output-dir", required=True, type=Path)

    execute = subparsers.add_parser(
        "execute", help="Execute one isolated RAW or COMMON_CONDITIONED stage."
    )
    execute.add_argument("--run-manifest", required=True, type=Path)
    execute.add_argument("--candidate", required=True)
    execute.add_argument(
        "--stage", required=True, choices=(RAW, COMMON_CONDITIONED)
    )

    compare = subparsers.add_parser(
        "compare", help="Write a no-composite, per-metric comparison."
    )
    compare.add_argument(
        "--run-manifest", required=True, type=Path, action="append"
    )
    compare.add_argument(
        "--stage", required=True, choices=(RAW, COMMON_CONDITIONED)
    )
    compare.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.action == "lock-bundle":
            bundle = lock_input_bundle(
                case_id=args.case_id,
                boundary=args.boundary,
                bathymetry=args.bathymetry,
                canonical_size_field=args.size_field,
                projection=_json_object(args.projection_json),
                node_budget=_json_object(args.node_budget_json),
            )
            write_input_bundle(args.output, bundle)
            result = bundle
        elif args.action == "plan":
            result = plan_bakeoff(
                load_declaration(args.declaration), args.output_dir
            )
        elif args.action == "execute":
            result = execute_stage(
                args.run_manifest, args.candidate, args.stage
            )
        else:
            result = compare_results(args.run_manifest, stage=args.stage)
            write_comparison(args.output, result)
    except (BakeoffContractError, FileNotFoundError, ValueError) as exc:
        code = getattr(exc, "code", type(exc).__name__)
        print(
            json.dumps(
                {"status": "failed", "failure_code": code, "message": str(exc)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.action == "execute" and result["status"] == "failed":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
