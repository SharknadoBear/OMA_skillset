from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate one existing, already-combined water-level time series to FVCOM open-boundary "
            "nodes, write an elevation forcing NetCDF, and create the default QA suite."
        )
    )
    parser.add_argument("--source", required=True, help="Input NetCDF or tidy CSV water-level file")
    parser.add_argument("--output", required=True, help="Output FVCOM elevation forcing NetCDF")
    parser.add_argument("--case-name", required=True, help="FVCOM case/title attribute")
    parser.add_argument("--datum", default="unspecified", help="Known source vertical datum; never inferred")

    grid = parser.add_mutually_exclusive_group(required=True)
    grid.add_argument("--mesh-2dm", help="Geographic SMS 2DM mesh")
    grid.add_argument("--grd", help="FVCOM _grd.dat file")
    parser.add_argument("--open-ns", nargs="+", type=int, help="Open-boundary nodestring ids for --mesh-2dm")
    parser.add_argument("--obc", help="FVCOM _obc.dat file required with --grd")

    parser.add_argument("--value-var", help="Water-level variable or CSV column")
    parser.add_argument("--time-var", help="Time variable or CSV column")
    parser.add_argument("--lon-var", help="Longitude variable or CSV column")
    parser.add_argument("--lat-var", help="Latitude variable or CSV column")
    parser.add_argument("--node-id-var", help="Boundary node-id variable or CSV column")
    parser.add_argument("--units", help="Override/input units: m, cm, or mm; required for CSV")
    parser.add_argument(
        "--assume-utc", action="store_true", help="Treat timezone-free input/target timestamps as UTC"
    )
    parser.add_argument(
        "--broadcast-single-series",
        action="store_true",
        help="Explicitly apply one non-spatial series identically to every boundary node",
    )
    parser.add_argument("--station-power", type=float, default=2.0, help="IDW power for station collections")
    parser.add_argument(
        "--max-nearest-km",
        type=float,
        help="Maximum bounded nearest-wet distance; default is twice source median spacing",
    )
    parser.add_argument("--start", help="Target UTC start (ISO 8601); use with --end and --dt-seconds")
    parser.add_argument("--end", help="Target UTC end (ISO 8601); use with --start and --dt-seconds")
    parser.add_argument("--dt-seconds", type=float, help="Target regular timestep in seconds")
    parser.add_argument(
        "--max-gap-factor", type=float, default=3.0, help="Reject interpolation over gaps above this native-cadence factor"
    )
    parser.add_argument("--qa-dir", help="QA directory; defaults to <output-stem>_qa")
    parser.add_argument("--tidal-min-hours", type=float, default=4.0)
    parser.add_argument("--tidal-max-hours", type=float, default=34.0)
    parser.add_argument("--vlf-min-days", type=float, default=90.0)
    return parser


def _load_runtime() -> tuple[object, object]:
    try:
        import waterlevel_core
        import waterlevel_validation
    except ModuleNotFoundError as exc:
        package = exc.name or "a required package"
        raise SystemExit(
            f"Missing Python dependency {package!r}. Install numpy, scipy, netCDF4, and matplotlib on this equipment."
        ) from exc
    return waterlevel_core, waterlevel_validation


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
    output = Path(args.output).resolve()
    qa_dir = Path(args.qa_dir).resolve() if args.qa_dir else output.with_name(f"{output.stem}_qa")
    output.parent.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    boundary = _boundary_from_args(args, core)
    source = core.read_source(
        args.source,
        value_var=args.value_var,
        time_var=args.time_var,
        lon_var=args.lon_var,
        lat_var=args.lat_var,
        node_var=args.node_id_var,
        units_override=args.units,
        assume_utc=args.assume_utc,
    )
    spatial_values, spatial_report = core.spatial_interpolate(
        source,
        boundary,
        broadcast_single=args.broadcast_single_series,
        station_power=args.station_power,
        max_nearest_km=args.max_nearest_km,
    )
    target_times = core.build_target_times(
        source.times_ms,
        start=args.start,
        end=args.end,
        dt_seconds=args.dt_seconds,
        assume_utc=args.assume_utc,
    )
    elevation, temporal_report = core.temporal_interpolate(
        source.times_ms, spatial_values, target_times, max_gap_factor=args.max_gap_factor
    )
    source_hash = core.sha256_file(args.source)

    fd, stage_name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".validation.nc", dir=output.parent)
    os.close(fd)
    os.unlink(stage_name)
    stage = Path(stage_name)
    try:
        core.write_fvcom_forcing(
            stage,
            boundary,
            target_times,
            elevation,
            case_name=args.case_name,
            vertical_datum=args.datum,
            source_name=args.source,
            source_sha256=source_hash,
            spatial_method=spatial_report["method"],
        )
        validation = validation_module.validate_forcing(
            stage,
            boundary,
            qa_dir,
            tidal_min_hours=args.tidal_min_hours,
            tidal_max_hours=args.tidal_max_hours,
            vlf_min_days=args.vlf_min_days,
        )
        os.replace(stage, output)
    except Exception:
        try:
            stage.unlink()
        except FileNotFoundError:
            pass
        raise

    validation["forcing_file"] = str(output)
    boundary_hashes = (
        {"mesh_2dm_sha256": core.sha256_file(args.mesh_2dm)}
        if args.mesh_2dm
        else {
            "grd_sha256": core.sha256_file(args.grd),
            "obc_sha256": core.sha256_file(args.obc),
        }
    )
    report = {
        "status": "pass",
        "builder": {
            "source_file": str(Path(args.source).resolve()),
            "source_sha256": source_hash,
            "boundary_input_hashes": boundary_hashes,
            "source_layout": source.layout,
            "source_variable": source.source_variable,
            "source_time_representation": source.time_source,
            "source_time_start_utc": core.iso_utc(source.times_ms[[0]])[0],
            "source_time_end_utc": core.iso_utc(source.times_ms[[-1]])[0],
            "source_time_count": int(len(source.times_ms)),
            "source_units_normalized": "meters",
            "vertical_datum": args.datum,
            "boundary_source": boundary.source,
            "boundary_node_count": int(len(boundary.node_ids)),
            "boundary_arc_count": int(len(boundary.arcs)),
            "boundary_geometry": {
                "longitude_span": [float(boundary.lon.min()), float(boundary.lon.max())],
                "latitude_span": [float(boundary.lat.min()), float(boundary.lat.max())],
                "arc_node_counts": [int(len(arc)) for arc in boundary.arcs],
                "arc_lengths_km": [float(core.arc_distances_km(boundary, arc)[-1]) for arc in boundary.arcs],
            },
            "spatial_interpolation": spatial_report,
            "temporal_interpolation": temporal_report,
            "output_file": str(output),
            "output_sha256": core.sha256_file(output),
        },
        "validation": validation,
    }
    report_path = qa_dir / "health_report.json"
    core.write_json_atomic(report_path, report)
    print(f"[PASS] FVCOM forcing: {output}")
    print(f"[PASS] QA report: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
