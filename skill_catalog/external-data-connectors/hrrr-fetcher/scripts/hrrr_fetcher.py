#!/usr/bin/env python3
"""Command-line and importable interface for resilient NOAA HRRR access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

try:
    from .download_monitor import atomic_write_json
    from .hrrr_core import (
        HrrrError,
        build_inventory,
        build_plan,
        execute_plan,
        health_run,
        normalize_request,
        product_catalog,
        runtime_preflight,
    )
except ImportError:
    from download_monitor import atomic_write_json
    from hrrr_core import (
        HrrrError,
        build_inventory,
        build_plan,
        execute_plan,
        health_run,
        normalize_request,
        product_catalog,
        runtime_preflight,
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    if str(path) == "-":
        value = json.load(sys.stdin)
    else:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def _emit(payload: Mapping[str, Any], output: str | Path | None = None) -> dict[str, Any]:
    result = dict(payload)
    if output:
        atomic_write_json(Path(output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def inventory_hrrr(payload: Mapping[str, Any]) -> dict[str, Any]:
    return build_inventory(payload)


def estimate_hrrr_request(payload: Mapping[str, Any], run_dir: str | Path) -> dict[str, Any]:
    return build_plan(payload, run_dir)


def fetch_hrrr(plan: Mapping[str, Any], run_dir: str | Path) -> dict[str, Any]:
    return execute_plan(plan, run_dir)


def snapshot_hrrr(payload: Mapping[str, Any], run_dir: str | Path) -> dict[str, Any]:
    request = normalize_request(payload)
    if request["mode"] == "analysis" and request["start"] != request["end"]:
        raise ValueError("snapshot analysis requires start == end")
    if request["mode"] == "forecast" and request["cycle_start"] != request["cycle_end"]:
        raise ValueError("snapshot forecast requires cycle_start == cycle_end")
    plan = build_plan(request, run_dir)
    _persist_plan(plan, run_dir)
    return execute_plan(plan, run_dir)


def health_hrrr_run(run_dir: str | Path) -> dict[str, Any]:
    return health_run(run_dir)


def _persist_plan(plan: Mapping[str, Any], run_dir: str | Path) -> None:
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / "request.normalized.json", plan["request"])
    atomic_write_json(directory / "inventory.json", plan["inventory"])
    atomic_write_json(directory / "download_plan.json", dict(plan))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory, estimate, range-fetch, decode, and validate NOAA HRRR GRIB2 fields."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    products = subparsers.add_parser("products", help="Print aliases, source families, domains, and providers")
    products.add_argument("--output", help="Optional JSON output path")

    preflight = subparsers.add_parser("preflight", help="Check the pinned GRIB decoding runtime")
    preflight.add_argument("--output", help="Optional JSON output path")

    inventory = subparsers.add_parser("inventory", help="Discover every required object and exact message range")
    inventory.add_argument("--request", required=True, help="hrrr_request_v1 JSON path, or - for stdin")
    inventory.add_argument("--output", help="Optional inventory JSON output path")

    estimate = subparsers.add_parser("estimate", help="Inventory and apply the transfer/storage gate")
    estimate.add_argument("--request", required=True)
    estimate.add_argument("--run-dir", required=True)
    estimate.add_argument("--output", help="Optional plan JSON output path; the run directory always receives one")

    fetch = subparsers.add_parser("fetch", help="Execute a ready hrrr_download_plan_v1")
    fetch.add_argument("--plan", required=True)
    fetch.add_argument("--run-dir", required=True)

    snapshot = subparsers.add_parser("snapshot", help="Plan and execute one analysis time or forecast cycle")
    snapshot.add_argument("--request", required=True)
    snapshot.add_argument("--run-dir", required=True)

    health = subparsers.add_parser("health", help="Validate a completed run and rewrite health_check.json")
    health.add_argument("--run-dir", required=True)
    health.add_argument("--output", help="Optional second report path")

    run = subparsers.add_parser("run", help="Plan and execute a bounded HRRR request")
    run.add_argument("--request", required=True)
    run.add_argument("--run-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "products":
            _emit(product_catalog(), args.output)
        elif args.command == "preflight":
            report = runtime_preflight()
            _emit(report, args.output)
            if not report["passed"]:
                return 2
        elif args.command == "inventory":
            _emit(build_inventory(_read_json(args.request)), args.output)
        elif args.command == "estimate":
            plan = build_plan(_read_json(args.request), args.run_dir)
            _persist_plan(plan, args.run_dir)
            _emit(plan, args.output)
            if plan["gate"]["state"] != "ready":
                return 2
        elif args.command == "fetch":
            _emit(execute_plan(_read_json(args.plan), args.run_dir))
        elif args.command == "snapshot":
            _emit(snapshot_hrrr(_read_json(args.request), args.run_dir))
        elif args.command == "health":
            report = health_run(args.run_dir)
            _emit(report, args.output)
            if not report["passed"]:
                return 2
        elif args.command == "run":
            plan = build_plan(_read_json(args.request), args.run_dir)
            _persist_plan(plan, args.run_dir)
            if plan["gate"]["state"] != "ready":
                _emit(plan)
                return 2
            _emit(execute_plan(plan, args.run_dir))
        return 0
    except (HrrrError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "hrrr_error_v1",
                    "command": args.command,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
