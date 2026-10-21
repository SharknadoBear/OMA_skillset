"""Command-line entry point and public helper exports."""
import argparse
import json
import importlib.metadata
from pathlib import Path

try:
    from era5_core import (ERA5Error, assemble_year, build_plan, check_runtime, health,
                           normalize_request, run, save_plan, snapshot, write_json)
except ModuleNotFoundError as exc:
    missing = []
    for name in ("cdsapi", "ecmwf-datastores-client", "numpy", "xarray", "netCDF4", "requests", "PyYAML", "matplotlib"):
        try:
            importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    print(json.dumps({"runtime_ready": False, "missing_packages": missing,
                      "message": "Install references/requirements.txt in an isolated environment"}, indent=2))
    raise SystemExit(2) from None


def main():
    parser = argparse.ArgumentParser(description="Regional hourly ERA5 wind and pressure downloader")
    commands = parser.add_subparsers(dest="command", required=True)
    runtime = commands.add_parser("check-runtime")
    runtime.add_argument("--access", action="store_true")
    runtime.add_argument("--output")
    for name in ("plan", "snapshot", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--request", required=True)
        sub.add_argument("--run-dir", required=True)
        if name == "run":
            sub.add_argument("--max-chunks", type=int)
    sub = commands.add_parser("health")
    sub.add_argument("--run-dir", required=True)
    sub = commands.add_parser("assemble-year")
    sub.add_argument("--run-dir", required=True)
    sub.add_argument("--year", type=int, required=True)
    sub.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "check-runtime":
            result = check_runtime(args.access)
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, indent=2))
            return 0 if result["runtime_ready"] and (not args.access or result["access"]["ready"]) else 2
        if args.command == "plan":
            result = save_plan(args.request, args.run_dir)
            print(json.dumps({k: result[k] for k in ("plan_hash", "total_hours", "chunk_count", "estimate", "storage")} | {"area": result["request"]["area"]}, indent=2))
            return 0 if result["storage"]["ready"] else 2
        if args.command == "snapshot":
            result = snapshot(args.request, args.run_dir)
        elif args.command == "run":
            save_plan(args.request, args.run_dir)
            result = run(args.run_dir, max_chunks=args.max_chunks)
        elif args.command == "health":
            result = health(args.run_dir)
        else:
            result = assemble_year(args.run_dir, args.year, args.output)
        if "health" in result:
            print(json.dumps({"new_chunks": result["new_chunks"], "reused_chunks": result["reused_chunks"],
                              "pass_all": result["health"]["pass_all"], "hours": result["health"]["completed_hours"]}, indent=2))
            return 0 if result["health"]["pass_all"] else 2
        print(json.dumps(result, indent=2))
        return 0 if result.get("pass_all", True) else 2
    except ERA5Error as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
