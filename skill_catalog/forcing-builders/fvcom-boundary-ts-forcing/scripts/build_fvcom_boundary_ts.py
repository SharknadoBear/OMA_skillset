from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and scientifically validate an FVCOM temperature/salinity open-boundary forcing file "
            "from a sigma-ready, per-boundary-node NetCDF source."
        )
    )
    parser.add_argument("--source", required=True, help="Sigma-ready per-node temperature/salinity NetCDF")
    parser.add_argument("--output", required=True, help="Output FVCOM T/S boundary forcing NetCDF")
    parser.add_argument("--case-name", required=True, help="FVCOM case/title attribute")
    grid = parser.add_mutually_exclusive_group(required=True)
    grid.add_argument("--mesh-2dm", help="Geographic SMS 2DM mesh with positive-down depth in ND records")
    grid.add_argument("--grd", help="FVCOM _grd.dat file")
    parser.add_argument("--open-ns", nargs="+", type=int, help="Open-boundary nodestring ids for --mesh-2dm")
    parser.add_argument("--obc", help="FVCOM _obc.dat file required with --grd")
    parser.add_argument("--temp-var", help="Temperature variable override")
    parser.add_argument("--salt-var", help="Salinity variable override")
    parser.add_argument("--time-var", help="Time variable override")
    parser.add_argument("--node-id-var", help="Boundary node-ID variable override")
    parser.add_argument("--siglay-var", help="Sigma-layer variable override")
    parser.add_argument("--siglev-var", help="Sigma-interface variable override")
    parser.add_argument("--temp-units", help="Temperature units override: Celsius or Kelvin")
    parser.add_argument("--salt-units", help="Salinity units override: PSU or recognized practical-salinity unit")
    parser.add_argument("--assume-utc", action="store_true", help="Treat timezone-free timestamps as UTC only after confirmation")
    parser.add_argument("--start", help="Target UTC start; requires --end and --dt-seconds")
    parser.add_argument("--end", help="Target UTC end; requires --start and --dt-seconds")
    parser.add_argument("--dt-seconds", type=float, help="Target regular timestep in seconds")
    parser.add_argument("--max-gap-factor", type=float, default=3.0, help="Reject resampling across larger source-time gaps")
    parser.add_argument("--temperature-min", type=float, default=-5.0)
    parser.add_argument("--temperature-max", type=float, default=45.0)
    parser.add_argument("--salinity-min", type=float, default=0.0)
    parser.add_argument("--salinity-max", type=float, default=50.0)
    parser.add_argument("--qa-dir", help="QA directory; defaults to <output-stem>_qa")
    return parser


def _load_runtime():
    try:
        import matplotlib  # noqa: F401
        import netCDF4  # noqa: F401
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import ts_core
        import ts_validation
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing Python dependency {exc.name!r}. Install numpy, scipy, netCDF4, and matplotlib on this equipment before continuing."
        ) from exc
    return ts_core, ts_validation


def _boundary_from_args(args: argparse.Namespace, core: object):
    if args.mesh_2dm:
        if not args.open_ns:
            raise ValueError("--mesh-2dm requires one or more --open-ns ids")
        if args.obc:
            raise ValueError("--obc is only valid with --grd")
        return core.read_boundary_2dm(args.mesh_2dm, args.open_ns)
    if not args.obc:
        raise ValueError("--grd requires --obc")
    if args.open_ns:
        raise ValueError("--open-ns is only valid with --mesh-2dm")
    return core.read_boundary_dat(args.grd, args.obc)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    core, validation_module = _load_runtime()
    if args.temperature_min >= args.temperature_max or args.salinity_min >= args.salinity_max:
        raise ValueError("Each physical minimum must be less than its maximum")
    output = Path(args.output).resolve()
    qa_dir = Path(args.qa_dir).resolve() if args.qa_dir else output.with_name(f"{output.stem}_qa")
    output.parent.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    boundary = _boundary_from_args(args, core)
    source = core.read_sigma_ready_source(
        args.source,
        boundary,
        temp_var=args.temp_var,
        salt_var=args.salt_var,
        time_var=args.time_var,
        node_var=args.node_id_var,
        siglay_var=args.siglay_var,
        siglev_var=args.siglev_var,
        temp_units=args.temp_units,
        salt_units=args.salt_units,
        assume_utc=args.assume_utc,
    )
    temperature, temperature_codes, temperature_repair = core.repair_missing(
        source.temperature_c, source.times_ms, source.siglay, boundary, "temperature"
    )
    salinity, salinity_codes, salinity_repair = core.repair_missing(
        source.salinity, source.times_ms, source.siglay, boundary, "salinity"
    )
    target_times = core.build_target_times(
        source.times_ms,
        start=args.start,
        end=args.end,
        dt_seconds=args.dt_seconds,
        assume_utc=args.assume_utc,
    )
    temperature, temperature_time = core.temporal_resample(
        source.times_ms, temperature, target_times, max_gap_factor=args.max_gap_factor
    )
    salinity, salinity_time = core.temporal_resample(
        source.times_ms, salinity, target_times, max_gap_factor=args.max_gap_factor
    )
    source_hash = core.sha256_file(args.source)
    fd, stage_name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".validation.nc", dir=output.parent)
    os.close(fd)
    os.unlink(stage_name)
    stage = Path(stage_name)
    try:
        core.write_fvcom_ts_forcing(
            stage,
            boundary,
            target_times,
            source.siglay,
            source.siglev,
            temperature,
            salinity,
            case_name=args.case_name,
            source_name=args.source,
            source_sha256=source_hash,
            sigma_orientation=source.sigma_orientation,
        )
        repairs = {"temperature": temperature_repair, "salinity": salinity_repair}
        validation = validation_module.validate_forcing(
            stage,
            boundary,
            qa_dir,
            temperature_min=args.temperature_min,
            temperature_max=args.temperature_max,
            salinity_min=args.salinity_min,
            salinity_max=args.salinity_max,
            repair_codes={"temperature": temperature_codes, "salinity": salinity_codes},
            repair_times_ms=source.times_ms,
            repair_reports=repairs,
        )
        validation["forcing_file"] = str(output)
        validation["forcing_sha256"] = core.sha256_file(stage)
        boundary_hashes = (
            {"mesh_2dm_sha256": core.sha256_file(args.mesh_2dm)}
            if args.mesh_2dm
            else {"grd_sha256": core.sha256_file(args.grd), "obc_sha256": core.sha256_file(args.obc)}
        )
        repaired_total = temperature_repair["repaired_total"] + salinity_repair["repaired_total"]
        report = {
            "status": "pass_with_repairs" if repaired_total else "pass",
            "builder": {
                "source_file": str(Path(args.source).resolve()),
                "source_sha256": source_hash,
                "boundary_input_hashes": boundary_hashes,
                "source_variables": source.source_variables,
                "source_units": source.source_units,
                "source_time_representation": source.time_source,
                "source_time_start_utc": core.iso_utc(source.times_ms[[0]])[0],
                "source_time_end_utc": core.iso_utc(source.times_ms[[-1]])[0],
                "source_time_count": int(len(source.times_ms)),
                "ignored_extra_source_node_ids": [int(value) for value in source.extra_node_ids],
                "boundary_source": boundary.source,
                "boundary_node_count": int(len(boundary.node_ids)),
                "boundary_arc_count": int(len(boundary.arcs)),
                "boundary_geometry": {
                    "longitude_span": [float(boundary.lon.min()), float(boundary.lon.max())],
                    "latitude_span": [float(boundary.lat.min()), float(boundary.lat.max())],
                    "depth_span_m": [float(boundary.depth_m.min()), float(boundary.depth_m.max())],
                    "arc_node_counts": [int(len(arc)) for arc in boundary.arcs],
                    "arc_lengths_km": [float(core.arc_distances_km(boundary, arc)[-1]) for arc in boundary.arcs],
                },
                "sigma_orientation": source.sigma_orientation,
                "repairs": repairs,
                "temporal_mapping": {"temperature": temperature_time, "salinity": salinity_time},
                "physical_bounds": {
                    "temperature_celsius": [args.temperature_min, args.temperature_max],
                    "salinity_psu": [args.salinity_min, args.salinity_max],
                },
                "output_file": str(output),
                "output_sha256": core.sha256_file(stage),
            },
            "validation": validation,
        }
        core.write_json_atomic(qa_dir / "health_report.json", report)
        os.replace(stage, output)
    except Exception:
        try:
            stage.unlink()
        except FileNotFoundError:
            pass
        raise
    print(f"[PASS] FVCOM T/S forcing: {output}")
    print(f"[PASS] QA report: {qa_dir / 'health_report.json'}")
    if repaired_total:
        print(f"[WARN] Repaired {repaired_total} source values; inspect missing_data_repair_diagnostics.png")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
