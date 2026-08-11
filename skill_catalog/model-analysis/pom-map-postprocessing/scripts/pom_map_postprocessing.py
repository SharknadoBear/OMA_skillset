"""CLI for inspecting and mapping POM curvilinear NetCDF fields."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from netCDF4 import Dataset

try:
    from .pom_map_tools import colormap_for_variable, quantile_limits, save_pom_scalar_map
    from .pom_output import inspect_inputs, load_scalar_series, parse_time, validate_vector_components
except ImportError:  # Direct script execution.
    from pom_map_tools import colormap_for_variable, quantile_limits, save_pom_scalar_map
    from pom_output import inspect_inputs, load_scalar_series, parse_time, validate_vector_components


def sha256_file(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def _iso(value: np.datetime64) -> str:
    return np.datetime_as_string(value.astype("datetime64[s]"), unit="s") + "Z"


def _input_provenance(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
        "sha256": sha256_file(path),
    }


def command_inspect(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.input)
    if source.resolve() == Path(args.output).resolve():
        raise ValueError("Inspection output must not overwrite the NetCDF input.")
    report = inspect_inputs([source])
    report["schema_version"] = "pom_inspection_v1"
    write_json_atomic(args.output, report)
    return {
        "status": report["status"],
        "output": str(Path(args.output).resolve()),
        "grid_shape": report["geometry"]["shape"],
        "time_count": report["combined_time"]["unique_record_count"],
    }


def _select_time_index(series, *, time: str | None, time_index: int | None) -> int:
    if (time is None) == (time_index is None):
        raise ValueError("Specify exactly one of --time or --time-index.")
    if time_index is not None:
        index = int(time_index)
        if index < 0:
            index += len(series.times)
        if not 0 <= index < len(series.times):
            raise IndexError(f"time-index {time_index} is outside 0..{len(series.times) - 1}.")
        return index
    requested = parse_time(time)
    matches = np.flatnonzero(series.times == requested)
    if matches.size:
        return int(matches[0])
    distances = np.abs(series.times.astype("int64") - requested.astype("int64")) / 1.0e9
    nearest = int(np.argmin(distances))
    raise KeyError(f"Time {time!r} is unavailable; nearest is {_iso(series.times[nearest])} ({distances[nearest]:.3f} s away).")


def _source_field_metadata(path: Path, source_variables: Sequence[str]) -> tuple[str, str]:
    with Dataset(path) as ds:
        variables = [ds.variables[name] for name in source_variables if name in ds.variables]
        if not variables:
            return "", ""
        units = str(getattr(variables[0], "units", ""))
        long_name = str(getattr(variables[0], "long_name", source_variables[0]))
        return units, long_name


def _grid_label(path: Path, source_payload: dict[str, Any]) -> str:
    value = str(source_payload.get("source_grid") or "").lower()
    if value in {"coarse", "fine"}:
        return value
    lower = path.name.lower()
    return "fine" if "nyofs_fg" in lower or "_fine_" in lower else "coarse"


def command_map(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.input)
    output = Path(args.output)
    report_path = Path(args.report)
    resolved_paths = {source.resolve(), output.resolve(), report_path.resolve()}
    if len(resolved_paths) != 3:
        raise ValueError("--input, --output, and --report must be three different paths.")
    if output.suffix.lower() != ".png":
        raise ValueError("--output must use a .png extension.")

    series = load_scalar_series([source], variable=args.variable, layer=args.layer)
    index = _select_time_index(series, time=args.time, time_index=args.time_index)
    frame = series.values[index]
    limit_values = series.values if args.limits_scope == "series" else frame
    limits = list(quantile_limits(limit_values, float(args.quantiles[0]), float(args.quantiles[1])))
    if args.vmin is not None:
        limits[0] = float(args.vmin)
    if args.vmax is not None:
        limits[1] = float(args.vmax)
    if not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError(f"Resolved color limits must satisfy vmin < vmax; got {limits}.")

    quiver_u = quiver_v = None
    rendered_quiver_count = 0
    quiver_metadata: dict[str, Any] = {"mode": "none"}
    if args.quiver != "none":
        first, second = (("u", "v") if args.quiver == "current" else ("air_u", "air_v"))
        vector_layer = args.layer if args.quiver == "current" else "surface"
        u_series = load_scalar_series([source], variable=first, layer=vector_layer)
        v_series = load_scalar_series([source], variable=second, layer=vector_layer)
        validate_vector_components(
            source,
            u_series.resolution["source_variables"][0],
            v_series.resolution["source_variables"][0],
            wind=args.quiver == "wind",
        )
        if not np.array_equal(u_series.times, v_series.times) or u_series.grid.geometry_sha256 != v_series.grid.geometry_sha256:
            raise ValueError(f"{first}/{second} vector components do not share normalized times and geometry.")
        vector_index = _select_time_index(u_series, time=_iso(series.times[index]), time_index=None)
        quiver_u, quiver_v = u_series.values[vector_index], v_series.values[vector_index]
        stride_slice = (slice(None, None, args.quiver_stride), slice(None, None, args.quiver_stride))
        rendered_quiver_count = int(
            np.count_nonzero(
                (series.grid.mask[stride_slice] == 1)
                & np.isfinite(quiver_u[stride_slice])
                & np.isfinite(quiver_v[stride_slice])
            )
        )
        quiver_metadata = {
            "mode": args.quiver,
            "source_variables": [u_series.resolution["source_variables"], v_series.resolution["source_variables"]],
            "layer": vector_layer,
        }

    time_label = _iso(series.times[index])
    grid = _grid_label(source, series.sources[0])
    title = args.title or f"{grid.title()} NYOFS {args.variable} ({args.layer})\n{time_label}"
    units, long_name = _source_field_metadata(source, series.resolution["source_variables"])
    selected_cmap = colormap_for_variable(args.variable) if args.cmap == "auto" else args.cmap
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    result = save_pom_scalar_map(
        str(temporary),
        lon=series.grid.lon,
        lat=series.grid.lat,
        mask=series.grid.mask,
        values=frame,
        title=title,
        vmin=limits[0],
        vmax=limits[1],
        cmap=selected_cmap,
        method=args.style,
        colorbar_label=f"{long_name}{f' [{units}]' if units else ''}",
        quiver_u=quiver_u,
        quiver_v=quiver_v,
        quiver_stride=args.quiver_stride,
        quiver_scale=args.quiver_scale,
        figure_size=tuple(args.figsize),
        dpi=args.dpi,
    )
    try:
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    output_stat = output.stat()
    manifest: dict[str, Any] = {
        "schema_version": "pom_map_manifest_v1",
        "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": _input_provenance(source),
        "request": {
            "variable": args.variable,
            "layer": args.layer,
            "time": args.time,
            "time_index": args.time_index,
            "style": args.style,
            "cmap": args.cmap,
            "limits_scope": args.limits_scope,
            "quantiles": [float(item) for item in args.quantiles],
            "requested_vmin": args.vmin,
            "requested_vmax": args.vmax,
            "dpi": args.dpi,
        },
        "selection": {
            "grid": grid,
            "grid_shape": [int(item) for item in series.grid.shape],
            "geometry_sha256": series.grid.geometry_sha256,
            "normalized_time_utc": time_label,
            "original_time_utc": _iso(series.original_times[index]),
            "normalization_offset_seconds": float(series.time_offsets_seconds[index]),
            "series_record_count": int(series.times.size),
            "cadence_seconds": series.resolution["normalized_cadence_seconds"],
            "duplicates_removed": series.duplicate_times_removed,
            "variable_requested": args.variable,
            "source_variables": series.resolution["source_variables"],
            "units": units,
            "long_name": long_name,
            "layer": args.layer,
            "resolution": series.resolution,
        },
        "rendering": {
            "style": args.style,
            "cmap": selected_cmap,
            "vmin": float(limits[0]),
            "vmax": float(limits[1]),
            "title": title,
            "figsize_inches": [float(item) for item in args.figsize],
            "wet_cells": result.wet_cell_count,
            "finite_wet_cells": result.finite_wet_count,
            "finite_wet_coverage": result.finite_wet_fraction,
            "quiver_count": rendered_quiver_count if result.quiver is not None else None,
            "quiver_stride": args.quiver_stride if result.quiver is not None else None,
            "quiver_scale": args.quiver_scale,
        },
        "quiver": quiver_metadata,
        "output": {
            "path": str(output.resolve()),
            "bytes": int(output_stat.st_size),
            "sha256": sha256_file(output),
            "format": output.suffix.lstrip(".").lower(),
        },
    }
    write_json_atomic(report_path, manifest)
    return {
        "status": "pass",
        "output": str(output.resolve()),
        "report": str(report_path.resolve()),
        "time": time_label,
        "vmin": limits[0],
        "vmax": limits[1],
        "finite_wet_coverage": result.finite_wet_fraction,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pom_map_postprocessing.py",
        description="Inspect and render native-grid POM curvilinear NetCDF fields.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inventory grid, time, sigma, and variables.")
    inspect_parser.add_argument("--input", required=True, help="POM NetCDF input.")
    inspect_parser.add_argument("--output", required=True, help="Inspection JSON output.")
    inspect_parser.set_defaults(handler=command_inspect)

    map_parser = subparsers.add_parser("map", help="Render one static scalar PNG.")
    map_parser.add_argument("--input", required=True, help="Raw or compact POM NetCDF input.")
    map_parser.add_argument("--variable", required=True, help="Source field, current_speed, or wind_speed.")
    group = map_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--time", help="Exact normalized UTC time, for example 2026-07-20T12:00:00Z.")
    group.add_argument("--time-index", type=int, help="Zero-based record index; negative values count backward.")
    map_parser.add_argument("--layer", default="surface", help="surface, near_surface, bottom, depth_average, or index:N.")
    map_parser.add_argument("--output", required=True, help="PNG output path.")
    map_parser.add_argument("--report", required=True, help="Map manifest JSON path.")
    map_parser.add_argument("--style", choices=("pcolormesh", "contourf"), default="pcolormesh")
    map_parser.add_argument("--cmap", default="auto", help="Matplotlib colormap name or auto.")
    map_parser.add_argument("--limits-scope", choices=("frame", "series"), default="frame")
    map_parser.add_argument("--quantiles", nargs=2, type=float, metavar=("LOW", "HIGH"), default=(2.0, 98.0))
    map_parser.add_argument("--vmin", type=float)
    map_parser.add_argument("--vmax", type=float)
    map_parser.add_argument("--quiver", choices=("none", "current", "wind"), default="none")
    map_parser.add_argument("--quiver-stride", type=int, default=8)
    map_parser.add_argument("--quiver-scale", type=float)
    map_parser.add_argument("--title")
    map_parser.add_argument("--figsize", nargs=2, type=float, default=(8.2, 7.0), metavar=("WIDTH", "HEIGHT"))
    map_parser.add_argument("--dpi", type=int, default=150)
    map_parser.set_defaults(handler=command_map)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "quantiles") and not (0 <= args.quantiles[0] < args.quantiles[1] <= 100):
        parser.error("--quantiles must satisfy 0 <= LOW < HIGH <= 100.")
    if hasattr(args, "vmin") and (args.vmin is None) != (args.vmax is None):
        parser.error("Specify both --vmin and --vmax, or neither.")
    if hasattr(args, "quiver_stride") and args.quiver_stride < 1:
        parser.error("--quiver-stride must be at least 1.")
    if hasattr(args, "figsize") and any(value <= 0 for value in args.figsize):
        parser.error("--figsize values must be positive.")
    if hasattr(args, "dpi") and args.dpi < 50:
        parser.error("--dpi must be at least 50.")
    try:
        summary = args.handler(args)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc), "type": type(exc).__name__}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
