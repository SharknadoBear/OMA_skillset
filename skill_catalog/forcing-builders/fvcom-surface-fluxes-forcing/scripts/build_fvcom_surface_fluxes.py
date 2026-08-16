#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

try:
    import matplotlib  # noqa: F401
    import netCDF4  # noqa: F401
    import numpy  # noqa: F401
    import scipy  # noqa: F401

    import surface_fluxes_core as core
    from surface_fluxes_validation import validate_forcing_files
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency {exc.name!r}. Install numpy, scipy, netCDF4, and matplotlib on this equipment before continuing."
    ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and scientifically validate modular FVCOM surface forcing files.")
    parser.add_argument("--source", required=True, help="Prepared neutral NetCDF")
    parser.add_argument("--layout", required=True, choices=("structured", "fvcom"))
    parser.add_argument("--packages", nargs="+", required=True, help="Any subset of wind heat freshwater pressure; commas are accepted")
    parser.add_argument("--wind-mode", choices=("speed", "stress"), default="speed")
    parser.add_argument("--heat-mode", choices=("direct", "bulk"), default="direct")
    parser.add_argument("--coare-version", choices=("COARE26Z", "COARE40VN"), default="COARE26Z")
    parser.add_argument("--file-layout", choices=("auto", "combined", "split"), default="auto")
    mesh = parser.add_mutually_exclusive_group()
    mesh.add_argument("--mesh-2dm", help="Geographic SMS 2DM mesh for native layout")
    mesh.add_argument("--grd", help="FVCOM _grd.dat mesh for native layout")
    parser.add_argument("--time-var")
    parser.add_argument("--lat-var")
    parser.add_argument("--lon-var")
    parser.add_argument("--var", action="append", default=[], metavar="ROLE=NAME", help="Prepared-variable override; repeat as needed")
    parser.add_argument("--pressure-reference", choices=("absolute",), help="Explicitly confirm absolute source pressure")
    parser.add_argument("--assume-utc", action="store_true", help="Treat timezone-free source timestamps as UTC after external confirmation")
    parser.add_argument("--external-wind-speed", action="store_true", help="Confirm bulk heat will use a separate compatible FVCOM wind-speed forcing")
    parser.add_argument("--model-start", help="Optional required FVCOM start coverage, explicit UTC")
    parser.add_argument("--model-end", help="Optional required FVCOM end coverage, explicit UTC")
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qa-dir", help="Default: OUTPUT_DIR/CASE_qa")
    parser.add_argument("--report", help="Default: OUTPUT_DIR/CASE_health_report.json")
    parser.add_argument("--namelist", help="Default: OUTPUT_DIR/CASE_surface_forcing.nml")
    return parser


def _packages(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(part.strip().lower() for part in value.split(",") if part.strip())
    return output


def _var_map(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--var must use ROLE=NAME syntax, found {value!r}")
        role, name = value.split("=", 1)
        role, name = role.strip(), name.strip()
        if role not in core.ROLE_ALIASES or not name:
            raise ValueError(f"Invalid variable override {value!r}")
        result[role] = name
    return result


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    packages = _packages(args.packages)
    if args.layout == "fvcom":
        if args.mesh_2dm:
            mesh = core.read_mesh_2dm(args.mesh_2dm)
        elif args.grd:
            mesh = core.read_mesh_grd(args.grd)
        else:
            raise ValueError("--layout fvcom requires --mesh-2dm or --grd")
    else:
        if args.mesh_2dm or args.grd:
            raise ValueError("Mesh options are only valid with --layout fvcom")
        mesh = None
    prepared = core.read_prepared_netcdf(
        args.source,
        layout=args.layout,
        packages=packages,
        wind_mode=args.wind_mode,
        heat_mode=args.heat_mode,
        coare_version=args.coare_version,
        mesh=mesh,
        var_map=_var_map(args.var),
        time_var=args.time_var,
        lat_var=args.lat_var,
        lon_var=args.lon_var,
        pressure_reference=args.pressure_reference,
        assume_utc=args.assume_utc,
    )
    model_start = core.parse_utc_ms(args.model_start)
    model_end = core.parse_utc_ms(args.model_end)
    output_dir = Path(args.output_dir)
    result = core.write_prepared_bundle(
        prepared,
        output_dir,
        case_name=args.case_name,
        packages=packages,
        wind_mode=args.wind_mode,
        heat_mode=args.heat_mode,
        coare_version=args.coare_version,
        file_layout=args.file_layout,
        external_wind_speed=args.external_wind_speed,
        model_start_ms=model_start,
        model_end_ms=model_end,
    )
    qa_dir = Path(args.qa_dir) if args.qa_dir else output_dir / f"{args.case_name}_qa"
    validation = validate_forcing_files(
        result.files.values(), qa_dir, model_start_ms=model_start, model_end_ms=model_end, mesh=prepared.mesh
    )
    namelist_path = Path(args.namelist) if args.namelist else output_dir / f"{args.case_name}_surface_forcing.nml"
    _write_text_atomic(namelist_path, result.namelist)
    report_path = Path(args.report) if args.report else output_dir / f"{args.case_name}_health_report.json"
    report = {
        "schema_version": "fvcom_surface_flux_bundle_v1",
        "status": "pass",
        "source": {
            "path": str(Path(args.source).resolve()),
            "sha256": prepared.source_sha256,
            "layout": prepared.layout,
            "fields": {
                role: {"shape": list(values.shape), "canonical_units": prepared.field_units[role]}
                for role, values in prepared.fields.items()
            },
        },
        "request": {
            "packages": packages,
            "wind_mode": args.wind_mode,
            "heat_mode": args.heat_mode,
            "coare_version": args.coare_version,
            "file_layout": args.file_layout,
            "layout_decision": result.layout_decision,
            "model_start_utc": args.model_start,
            "model_end_utc": args.model_end,
        },
        "time": {
            "start_utc": core.iso_utc(prepared.times_ms[[0]])[0],
            "end_utc": core.iso_utc(prepared.times_ms[[-1]])[0],
            "records": len(prepared.times_ms),
        },
        "transformations": list(result.transformations),
        "outputs": {
            label: {"path": str(path.resolve()), "sha256": core.sha256_file(path), "bytes": path.stat().st_size}
            for label, path in result.files.items()
        },
        "package_files": {package: str(path.resolve()) for package, path in result.package_files.items()},
        "namelist": {"path": str(namelist_path.resolve()), "text": result.namelist},
        "validation": validation,
    }
    core.write_json_atomic(report_path, report)
    print(f"[PASS] FVCOM surface forcing bundle: {', '.join(str(path) for path in result.files.values())}")
    print(f"[PASS] Health report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
