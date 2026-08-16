#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import matplotlib  # noqa: F401
    import netCDF4  # noqa: F401
    import numpy  # noqa: F401
    import scipy  # noqa: F401

    from surface_fluxes_core import parse_utc_ms, read_mesh_2dm, read_mesh_grd, write_json_atomic
    from surface_fluxes_validation import validate_forcing_files
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency {exc.name!r}. Install numpy, scipy, netCDF4, and matplotlib on this equipment before continuing."
    ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate existing FVCOM surface forcing files and regenerate scientific QA.")
    parser.add_argument("--forcing", nargs="+", required=True, help="One combined file or all split bundle members")
    parser.add_argument("--qa-dir", required=True)
    parser.add_argument("--report", help="Optional validation JSON")
    parser.add_argument("--model-start", help="Optional required start coverage, explicit UTC")
    parser.add_argument("--model-end", help="Optional required end coverage, explicit UTC")
    mesh = parser.add_mutually_exclusive_group()
    mesh.add_argument("--mesh-2dm", help="Optional geographic SMS mesh for native-grid alignment")
    mesh.add_argument("--grd", help="Optional FVCOM _grd.dat for native-grid alignment")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mesh = read_mesh_2dm(args.mesh_2dm) if args.mesh_2dm else read_mesh_grd(args.grd) if args.grd else None
    report = validate_forcing_files(
        args.forcing,
        args.qa_dir,
        model_start_ms=parse_utc_ms(args.model_start),
        model_end_ms=parse_utc_ms(args.model_end),
        mesh=mesh,
    )
    if args.report:
        write_json_atomic(Path(args.report), report)
    print(f"[PASS] Validated {len(args.forcing)} FVCOM surface forcing file(s); QA: {args.qa_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
