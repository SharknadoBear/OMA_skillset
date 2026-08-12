#!/usr/bin/env python3
"""Inspect and map raw or compact staggered ROMS NetCDF fields."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
from netCDF4 import Dataset

try:
    from .roms_map_tools import colormap_for_variable, quantile_limits, save_roms_scalar_map
    from .roms_output import inspect_inputs, load_current_series, load_scalar_series, parse_time
except ImportError:
    from roms_map_tools import colormap_for_variable, quantile_limits, save_roms_scalar_map
    from roms_output import inspect_inputs, load_current_series, load_scalar_series, parse_time


def _flatten(values: Iterable[str | Path | Sequence[str | Path]]) -> list[Path]:
    result = []
    for value in values:
        if isinstance(value, (str, Path)):
            result.append(Path(value).expanduser().resolve())
        else:
            result.extend(Path(item).expanduser().resolve() for item in value)
    if not result:
        raise ValueError("At least one --input file is required.")
    return result


def _jsonable(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _iso(value: np.datetime64) -> str:
    return np.datetime_as_string(np.datetime64(value, "ns"), unit="ns") + "Z"


def _select_index(times: np.ndarray, requested: str | None, index: int | None) -> int:
    if (requested is None) == (index is None):
        raise ValueError("Specify exactly one of --time or --time-index.")
    if index is not None:
        selected = int(index)
        if selected < 0:
            selected += len(times)
        if not 0 <= selected < len(times):
            raise IndexError(f"time-index {index} is outside 0..{len(times) - 1}.")
        return selected
    target = parse_time(requested)
    matches = np.flatnonzero(times == target)
    if matches.size:
        return int(matches[0])
    seconds = np.abs(times.astype("int64") - target.astype("int64")) / 1.0e9
    nearest = int(np.argmin(seconds))
    raise KeyError(f"Time {requested!r} is unavailable; nearest is {_iso(times[nearest])} ({seconds[nearest]:.3f} seconds away).")


def _source_label(series, index: int) -> tuple[str, str]:
    source_variables = series.resolution.get("source_variables", [])
    requested = str(series.resolution.get("requested_variable", ""))
    if not source_variables or not series.record_sources:
        return "", requested
    source = Path(series.record_sources[index])
    if not source.is_file():
        return "", str(series.resolution.get("requested_variable", ""))
    with Dataset(source) as ds:
        variable = ds.variables.get(source_variables[0])
        if variable is None:
            return "", requested
        units = str(getattr(variable, "units", ""))
        key = requested.strip().lower()
        current_labels = {
            "current_speed": "Earth-relative current speed",
            "eastward_velocity": "Eastward sea-water velocity",
            "eastward_sea_water_velocity": "Eastward sea-water velocity",
            "northward_velocity": "Northward sea-water velocity",
            "northward_sea_water_velocity": "Northward sea-water velocity",
        }
        if key in current_labels:
            return units, current_labels[key]
        label = getattr(variable, "long_name", getattr(variable, "standard_name", source_variables[0]))
        return units, str(label)


def inspect_command(args) -> dict[str, Any]:
    paths = _flatten(args.input)
    output = Path(args.output).expanduser().resolve()
    if output in set(paths):
        raise ValueError("Inspection output must not collide with any input path.")
    payload = inspect_inputs(paths)
    payload["created_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    destination = _write_json(output, payload)
    return {"status": payload["status"], "output": str(destination),
            "input_count": len(paths), "time_count": payload["combined_time"]["unique_record_count"]}


def map_command(args) -> dict[str, Any]:
    paths = _flatten(args.input)
    output, report = Path(args.output).expanduser().resolve(), Path(args.report).expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("--output must end in .png.")
    protected = {path for path in paths}
    if output in protected or report in protected or output == report:
        raise ValueError("Inputs, PNG output, and JSON report must use distinct paths.")
    series = load_scalar_series(paths, variable=args.variable, layer=args.layer)
    index = _select_index(series.times, args.time, args.time_index)
    frame = series.values[index]
    coordinate_wet = ((series.grid.mask == 1) & np.isfinite(series.grid.lon)
                      & np.isfinite(series.grid.lat))
    finite_wet_count = int(np.count_nonzero(coordinate_wet & np.isfinite(frame)))
    if finite_wet_count == 0:
        raise ValueError("Selected ROMS map frame has zero finite wet coverage and cannot be rendered.")
    limit_values = series.values if args.limits_scope == "series" else frame
    if args.vmin is None:
        vmin, vmax = quantile_limits(limit_values, *args.quantiles)
        limit_method = f"{args.limits_scope}_quantiles"
    else:
        vmin, vmax = float(args.vmin), float(args.vmax)
        limit_method = "explicit"
    if not np.isfinite([vmin, vmax]).all() or vmin >= vmax:
        raise ValueError("Resolved color limits must be finite with vmin < vmax.")

    quiver_u = quiver_v = None
    quiver_count = 0
    quiver_metadata = {"mode": "none"}
    if args.quiver == "current":
        vectors = load_current_series(paths, layer=args.layer)
        vector_index = _select_index(vectors.times, _iso(series.times[index]), None)
        if vectors.grid.geometry_sha256 != series.grid.geometry_sha256:
            raise ValueError("Scalar and current-vector grids differ.")
        quiver_u, quiver_v = vectors.east[vector_index], vectors.north[vector_index]
        sample = np.s_[::args.quiver_stride, ::args.quiver_stride]
        quiver_count = int(np.count_nonzero((series.grid.mask[sample] == 1)
                           & np.isfinite(quiver_u[sample]) & np.isfinite(quiver_v[sample])))
        quiver_metadata = {"mode": "current", "layer": args.layer, "resolution": vectors.resolution}

    units, long_name = _source_label(series, index)
    timestamp = _iso(series.times[index])
    title = args.title or f"ROMS {args.variable} ({args.layer})\n{timestamp}"
    cmap = colormap_for_variable(args.variable) if args.cmap == "auto" else args.cmap
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=output.suffix,
                                                   dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        result = save_roms_scalar_map(
            str(temporary), lon=series.grid.lon, lat=series.grid.lat, mask=series.grid.mask,
            values=frame, vmin=vmin, vmax=vmax, cmap=cmap, title=title,
            colorbar_label=f"{long_name}{f' [{units}]' if units else ''}", method=args.style,
            quiver_u=quiver_u, quiver_v=quiver_v, quiver_stride=args.quiver_stride,
            quiver_scale=args.quiver_scale, figure_size=tuple(args.figsize), dpi=args.dpi)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "schema_version": "roms_map_manifest_v1", "status": "pass",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": list(series.sources),
        "request": {"variable": args.variable, "layer": args.layer, "time": args.time,
                    "time_index": args.time_index, "limits_scope": args.limits_scope,
                    "quantiles": list(args.quantiles), "requested_vmin": args.vmin,
                    "requested_vmax": args.vmax},
        "selection": {
            "normalized_time_utc": timestamp, "original_time_utc": _iso(series.original_times[index]),
            "normalization_offset_seconds": float(series.time_offsets_seconds[index]),
            "source": series.record_sources[index],
            "source_record_index": int(series.record_indices[index]),
            "series_record_count": len(series.times), "duplicates_removed": series.duplicate_times_removed,
            "geometry_sha256": series.grid.geometry_sha256, "grid_shape": list(series.grid.shape),
            "angle_units": series.grid.angle_units,
            "angle_convention": series.grid.angle_convention,
            "resolution": series.resolution, "units": units, "long_name": long_name,
        },
        "rendering": {
            "style": args.style, "cmap": cmap, "vmin": float(vmin), "vmax": float(vmax),
            "color_limit_method": limit_method, "quantiles_percent": list(args.quantiles) if args.vmin is None else None,
            "finite_wet_coverage": result.finite_wet_fraction, "wet_cells": result.wet_cell_count,
            "finite_wet_cells": result.finite_wet_count, "quiver_count": quiver_count if result.quiver else None,
            "quiver_stride": args.quiver_stride if result.quiver else None,
            "quiver_scale": args.quiver_scale, "title": title, "dpi": args.dpi,
        },
        "quiver": quiver_metadata,
        "output": {"path": str(output), "size_bytes": output.stat().st_size,
                   "sha256": __import__("hashlib").sha256(output.read_bytes()).hexdigest(), "format": "png"},
    }
    _write_json(report, manifest)
    return {"status": "pass", "output": str(output), "report": str(report), "time": timestamp,
            "vmin": float(vmin), "vmax": float(vmax), "finite_wet_coverage": result.finite_wet_fraction}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roms_map_postprocessing.py",
                                     description="Inspect and map staggered ROMS curvilinear fields.")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="Inspect ROMS geometry, times, and fields.")
    inspect_parser.add_argument("--input", action="append", nargs="+", required=True)
    inspect_parser.add_argument("--output", required=True)
    inspect_parser.set_defaults(handler=inspect_command)
    map_parser = commands.add_parser("map", help="Render one static ROMS scalar PNG.")
    map_parser.add_argument("--input", action="append", nargs="+", required=True)
    map_parser.add_argument("--variable", required=True)
    selection = map_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--time")
    selection.add_argument("--time-index", type=int)
    map_parser.add_argument("--layer", default="surface")
    map_parser.add_argument("--limits-scope", choices=("frame", "series"), default="frame")
    map_parser.add_argument("--quantiles", nargs=2, type=float, default=(2.0, 98.0), metavar=("LOW", "HIGH"))
    map_parser.add_argument("--vmin", type=float)
    map_parser.add_argument("--vmax", type=float)
    map_parser.add_argument("--style", choices=("pcolormesh", "contourf"), default="pcolormesh")
    map_parser.add_argument("--cmap", default="auto")
    map_parser.add_argument("--quiver", choices=("none", "current"), default="none")
    map_parser.add_argument("--quiver-stride", type=int, default=8)
    map_parser.add_argument("--quiver-scale", type=float)
    map_parser.add_argument("--title")
    map_parser.add_argument("--figsize", nargs=2, type=float, default=(8.2, 7.0))
    map_parser.add_argument("--dpi", type=int, default=150)
    map_parser.add_argument("--output", required=True)
    map_parser.add_argument("--report", required=True)
    map_parser.set_defaults(handler=map_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "quantiles") and not 0 <= args.quantiles[0] < args.quantiles[1] <= 100:
        parser.error("--quantiles must satisfy 0 <= LOW < HIGH <= 100.")
    if hasattr(args, "vmin") and (args.vmin is None) != (args.vmax is None):
        parser.error("Specify both --vmin and --vmax, or neither.")
    if hasattr(args, "quiver_stride") and args.quiver_stride < 1:
        parser.error("--quiver-stride must be at least 1.")
    try:
        result = args.handler(args)
    except Exception as error:
        print(json.dumps({"status": "error", "type": type(error).__name__, "error": str(error)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
