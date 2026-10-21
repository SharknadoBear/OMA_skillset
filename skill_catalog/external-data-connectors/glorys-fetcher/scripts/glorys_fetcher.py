"""CLI and public Python helpers for glorys-fetcher."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from glorys_core import (DAILY, MONTHLY, GLORYSError, check_runtime, inventory,
    build_plan, save_plan, snapshot, fetch, inspect, health, read_json, write_json, OfficialBackend)


def plan(request_path, run_dir, backend=None, toolbox_estimate=False):
    result = save_plan(request_path, run_dir, backend=backend)
    if toolbox_estimate:
        backend = backend or OfficialBackend(service=result["source"]["service"])
        estimates = [backend.subset(result, c, Path(run_dir) / "estimate.nc", dry_run=True) for c in result["chunks"]]
        result["toolbox_estimate"] = {"chunks": estimates, "note": "Toolbox output and maximum transfer estimates in MB; observed network transfer is not measured"}
        write_json(Path(run_dir) / "download_plan.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("check-runtime"); p.add_argument("--access", action="store_true"); p.add_argument("--output")
    p = commands.add_parser("inventory"); p.add_argument("--dataset-id", default=DAILY, choices=[DAILY, MONTHLY]); p.add_argument("--output"); p.add_argument("--service", default="arco-geo-series", choices=["arco-geo-series","arco-time-series"])
    for name in ("plan", "snapshot"):
        p = commands.add_parser(name); p.add_argument("--request", required=True); p.add_argument("--run-dir", required=True)
        if name == "plan": p.add_argument("--toolbox-estimate", action="store_true")
    p = commands.add_parser("fetch"); p.add_argument("--run-dir", required=True); p.add_argument("--max-chunks", type=int)
    p = commands.add_parser("inspect"); p.add_argument("--file", required=True); p.add_argument("--output")
    p = commands.add_parser("health"); p.add_argument("--run-dir", required=True); p.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "check-runtime": result = check_runtime(args.access)
        elif args.command == "inventory": result = inventory(args.dataset_id, service=args.service)
        elif args.command == "plan": result = plan(args.request, args.run_dir, toolbox_estimate=args.toolbox_estimate)
        elif args.command == "snapshot": result = snapshot(args.request, args.run_dir)
        elif args.command == "fetch":
            if args.max_chunks is not None and args.max_chunks < 1: parser.error("--max-chunks must be positive")
            result = fetch(args.run_dir, max_chunks=args.max_chunks)
        elif args.command == "inspect": result = inspect(args.file)
        else: result = health(args.run_dir)
        if getattr(args, "output", None): write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == "check-runtime": return 0 if result["runtime_ready"] and result.get("access", {}).get("ready", True) else 2
        if args.command == "health": return 0 if result["pass"] else 2
        return 0
    except GLORYSError as exc:
        print(json.dumps({"error_type": type(exc).__name__, "error": str(exc)}))
        return 2
    except (OSError, ValueError, KeyError) as exc:
        # Do not propagate raw API/credential content through tracebacks.
        print(json.dumps({"error_type": type(exc).__name__, "error": "Local input or output could not be read; verify paths and JSON"}))
        return 2


if __name__ == "__main__": raise SystemExit(main())
