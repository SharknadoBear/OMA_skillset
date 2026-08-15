from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and plot an existing FVCOM time-series elevation boundary forcing file."
    )
    parser.add_argument("--forcing", required=True, help="FVCOM elevation forcing NetCDF")
    grid = parser.add_mutually_exclusive_group(required=True)
    grid.add_argument("--mesh-2dm", help="Geographic SMS 2DM mesh")
    grid.add_argument("--grd", help="FVCOM _grd.dat file")
    parser.add_argument("--open-ns", nargs="+", type=int, help="Open nodestring ids for --mesh-2dm")
    parser.add_argument("--obc", help="FVCOM _obc.dat file required with --grd")
    parser.add_argument("--qa-dir", required=True, help="Directory for plots and validation_report.json")
    parser.add_argument("--tidal-min-hours", type=float, default=4.0)
    parser.add_argument("--tidal-max-hours", type=float, default=34.0)
    parser.add_argument("--vlf-min-days", type=float, default=90.0)
    return parser


def _load_runtime():
    try:
        import waterlevel_core
        import waterlevel_validation
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing Python dependency {exc.name!r}. Install numpy, scipy, netCDF4, and matplotlib on this equipment."
        ) from exc
    return waterlevel_core, waterlevel_validation


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    core, validation = _load_runtime()
    if args.mesh_2dm:
        if not args.open_ns:
            raise ValueError("--mesh-2dm requires --open-ns")
        if args.obc:
            raise ValueError("--obc is only valid with --grd")
        boundary = core.read_boundary_2dm(args.mesh_2dm, args.open_ns)
    else:
        if not args.obc:
            raise ValueError("--grd requires --obc")
        if args.open_ns:
            raise ValueError("--open-ns is only valid with --mesh-2dm")
        boundary = core.read_boundary_dat(args.grd, args.obc)
    report = validation.validate_forcing(
        args.forcing,
        boundary,
        args.qa_dir,
        tidal_min_hours=args.tidal_min_hours,
        tidal_max_hours=args.tidal_max_hours,
        vlf_min_days=args.vlf_min_days,
    )
    report_path = Path(args.qa_dir) / "validation_report.json"
    core.write_json_atomic(report_path, report)
    print(f"[PASS] FVCOM boundary forcing validation: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
