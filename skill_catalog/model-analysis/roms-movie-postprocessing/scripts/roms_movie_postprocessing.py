#!/usr/bin/env python3
"""Inspect ROMS files and create fixed-color-scale scalar GIFs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from .roms_map_tools import colormap_for_variable, plot_roms_scalar, quantile_limits
    from .roms_output import inspect_inputs as inspect_roms_inputs
    from .roms_output import load_scalar_series
except ImportError:
    from roms_map_tools import colormap_for_variable, plot_roms_scalar, quantile_limits
    from roms_output import inspect_inputs as inspect_roms_inputs
    from roms_output import load_scalar_series


def _flatten(values: Iterable[str | Path | Sequence[str | Path]]) -> list[Path]:
    paths = []
    for value in values:
        if isinstance(value, (str, Path)):
            paths.append(Path(value).expanduser().resolve())
        else:
            paths.extend(Path(item).expanduser().resolve() for item in value)
    if not paths:
        raise ValueError("At least one --input file is required.")
    return paths


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _iso(value: np.datetime64) -> str:
    return np.datetime_as_string(np.datetime64(value, "ns"), unit="ns") + "Z"


def _display_label(variable: str) -> str:
    key = str(variable).strip().lower()
    return {
        "current_speed": "Earth-relative current speed",
        "eastward_velocity": "Eastward sea-water velocity",
        "eastward_sea_water_velocity": "Eastward sea-water velocity",
        "northward_velocity": "Northward sea-water velocity",
        "northward_sea_water_velocity": "Northward sea-water velocity",
    }.get(key, str(variable))


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


def inspect_inputs(inputs: Sequence[str | Path]) -> dict[str, Any]:
    payload = inspect_roms_inputs(inputs)
    payload["schema_version"] = "roms_movie_inspection_v1"
    payload["created_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return payload


def create_gif(
    inputs: Sequence[str | Path],
    *,
    variable: str,
    layer: str = "surface",
    start: str | None = None,
    end_exclusive: str | None = None,
    fps: float = 4.0,
    quantiles: tuple[float, float] = (2.0, 98.0),
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    output: str | Path,
    report: str | Path | None = None,
    dpi: int = 120,
    figure_size: tuple[float, float] = (7.2, 6.0),
    title_template: str = "{variable} {layer} | {time}",
) -> dict[str, Any]:
    """Render a fixed-scale scalar GIF from raw or compact ROMS inputs."""

    paths = _flatten(inputs)
    if not variable.strip():
        raise ValueError("variable must not be empty.")
    if fps <= 0:
        raise ValueError("fps must be greater than zero.")
    if dpi < 40:
        raise ValueError("dpi must be at least 40.")
    low, high = float(quantiles[0]), float(quantiles[1])
    if not 0 <= low < high <= 100:
        raise ValueError("quantiles must satisfy 0 <= low < high <= 100.")
    if (vmin is None) != (vmax is None):
        raise ValueError("Provide both vmin and vmax, or neither.")
    output_path = Path(output).expanduser().resolve()
    report_path = None if report is None else Path(report).expanduser().resolve()
    if output_path.suffix.lower() != ".gif":
        raise ValueError("output must end in .gif.")
    targets = [output_path] + ([] if report_path is None else [report_path])
    if len(set(targets)) != len(targets) or any(target in set(paths) for target in targets):
        raise ValueError("Input, GIF output, and report paths must be distinct.")

    series = load_scalar_series(paths, variable=variable, layer=layer, start=start,
                                end_exclusive=end_exclusive, snap_tolerance_seconds=60.0)
    values = np.asarray(series.values, dtype=float)
    times = np.asarray(series.times, dtype="datetime64[ns]")
    if values.ndim != 3 or values.shape[0] != len(times):
        raise ValueError(f"Expected scalar frames (time,eta_rho,xi_rho); got {values.shape} and {len(times)} times.")
    if len(times) > 1 and np.any(np.diff(times.astype("int64")) <= 0):
        raise ValueError("Selected ROMS times are not unique and strictly increasing.")
    wet = (series.grid.mask == 1) & np.isfinite(series.grid.lon) & np.isfinite(series.grid.lat)
    wet_count = int(np.count_nonzero(wet))
    if wet_count == 0:
        raise ValueError("ROMS grid has zero finite wet coordinate cells.")
    coverage = np.asarray([np.count_nonzero(np.isfinite(frame) & wet) / wet_count for frame in values])
    if np.any(coverage == 0):
        raise ValueError(f"All-NaN wet frames are not renderable: {np.flatnonzero(coverage == 0).tolist()}.")
    finite = values[:, wet]
    finite = finite[np.isfinite(finite)]
    if vmin is None:
        vmin, vmax = quantile_limits(finite, low, high)
        color_method = "full_series_quantiles"
    else:
        vmin, vmax = float(vmin), float(vmax)
        color_method = "explicit"
    if not np.isfinite([vmin, vmax]).all() or vmin >= vmax:
        raise ValueError("Fixed color limits must be finite with vmin < vmax.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_cmap = cmap or colormap_for_variable(variable)
    duration_ms = max(1, int(round(1000.0 / fps)))
    images = []
    temporary_cleaned = False
    temporary_name = None
    frame_hashes = []
    frame_count = 0
    size = []
    observed_duration = duration_ms
    try:
        with tempfile.TemporaryDirectory(prefix=".roms_movie_", dir=output_path.parent) as directory:
            temporary_name = directory
            for index, (timestamp, frame) in enumerate(zip(times, values, strict=True)):
                fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
                png = Path(directory) / f"frame_{index:05d}.png"
                try:
                    title = title_template.format(variable=variable, layer=layer, time=_iso(timestamp), index=index)
                    plot_roms_scalar(
                        ax, lon=series.grid.lon, lat=series.grid.lat, mask=series.grid.mask,
                        values=frame, vmin=vmin, vmax=vmax, cmap=selected_cmap, title=title,
                        colorbar_label=_display_label(variable))
                    fig.savefig(png, dpi=dpi)
                finally:
                    plt.close(fig)
                with Image.open(png) as image:
                    images.append(image.convert("P", palette=Image.Palette.ADAPTIVE).copy())
            staged_output = Path(directory) / "animation.gif"
            images[0].save(staged_output, format="GIF", save_all=True, append_images=images[1:],
                           duration=duration_ms, loop=0, optimize=False, disposal=2)
            with Image.open(staged_output) as gif:
                frame_count = int(getattr(gif, "n_frames", 1))
                size = [int(gif.width), int(gif.height)]
                for index in range(frame_count):
                    gif.seek(index)
                    frame_hashes.append(hashlib.sha256(gif.convert("RGB").tobytes()).hexdigest())
                observed_duration = int(gif.info.get("duration", duration_ms))
            if frame_count != len(times):
                raise RuntimeError(f"GIF contains {frame_count} frames; expected {len(times)}.")
            os.replace(staged_output, output_path)
        temporary_cleaned = temporary_name is None or not Path(temporary_name).exists()
    finally:
        for image in images:
            image.close()
        plt.close("all")

    minimum_coverage = float(np.min(coverage))
    warnings = []
    if minimum_coverage < 0.95:
        warnings.append(f"Minimum finite wet coverage is {minimum_coverage:.3%}, below 95%.")
    frames = [{
        "index": index, "time_utc": _iso(times[index]),
        "original_time_utc": _iso(series.original_times[index]),
        "normalization_offset_seconds": float(series.time_offsets_seconds[index]),
        "finite_wet_fraction": float(coverage[index]), "source": series.record_sources[index],
        "source_record_index": int(series.record_indices[index]), "rendered_frame_sha256": frame_hashes[index],
    } for index in range(len(times))]
    manifest = {
        "schema_version": "roms_movie_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "pass_with_warnings" if warnings else "pass", "warnings": warnings,
        "inputs": list(series.sources),
        "request": {"variable": variable, "layer": layer, "start_utc": start,
                    "end_utc_exclusive": end_exclusive},
        "resolved": series.resolution,
        "grid": {"shape": list(series.grid.shape), "wet_cell_count": wet_count,
                 "geometry_sha256": series.grid.geometry_sha256,
                 "Vtransform": series.grid.vtransform, "Vstretching": series.grid.vstretching,
                 "angle_units": series.grid.angle_units,
                 "angle_convention": series.grid.angle_convention},
        "selection": {"frame_count": len(times), "first_time_utc": _iso(times[0]),
                      "last_time_utc": _iso(times[-1]), "unique_monotonic": True,
                      "duplicate_times_removed": series.duplicate_times_removed,
                      "distinct_rendered_frame_count": len(set(frame_hashes)), "snap_tolerance_seconds": 60.0},
        "fixed_color_limits": {"method": color_method, "vmin": float(vmin), "vmax": float(vmax),
                               "quantiles_percent": [low, high] if color_method == "full_series_quantiles" else None},
        "coverage": {"minimum_finite_wet_fraction": minimum_coverage,
                     "mean_finite_wet_fraction": float(np.mean(coverage)), "all_nan_frame_count": 0},
        "rendering": {"format": "GIF", "fps_requested": float(fps), "frame_duration_ms": duration_ms,
                      "cmap": selected_cmap, "dpi": int(dpi), "figure_size_inches": list(figure_size),
                      "movie_quivers": False, "temporary_frames_cleaned": temporary_cleaned},
        "frames": frames,
        "output": {"path": str(output_path), "size_bytes": output_path.stat().st_size,
                   "sha256": _sha256(output_path), "frame_count": frame_count,
                   "pixel_size": size, "frame_duration_ms_observed": observed_duration},
    }
    if report_path is not None:
        manifest["report_path"] = str(report_path)
        _write_json(report_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roms_movie_postprocessing.py",
                                     description="Inspect and animate staggered ROMS curvilinear fields.")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--input", action="append", nargs="+", required=True)
    inspect_parser.add_argument("--output", required=True)
    gif_parser = commands.add_parser("gif")
    gif_parser.add_argument("--input", action="append", nargs="+", required=True)
    gif_parser.add_argument("--variable", required=True)
    gif_parser.add_argument("--layer", default="surface")
    gif_parser.add_argument("--start")
    gif_parser.add_argument("--end-exclusive")
    gif_parser.add_argument("--fps", type=float, default=4.0)
    gif_parser.add_argument("--quantiles", nargs=2, type=float, default=(2.0, 98.0), metavar=("LOW", "HIGH"))
    gif_parser.add_argument("--vmin", type=float)
    gif_parser.add_argument("--vmax", type=float)
    gif_parser.add_argument("--cmap")
    gif_parser.add_argument("--dpi", type=int, default=120)
    gif_parser.add_argument("--figure-size", nargs=2, type=float, default=(7.2, 6.0))
    gif_parser.add_argument("--title-template", default="{variable} {layer} | {time}")
    gif_parser.add_argument("--output", required=True)
    gif_parser.add_argument("--report", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        paths = _flatten(args.input)
        if args.command == "inspect":
            inspection_output = Path(args.output).expanduser().resolve()
            if inspection_output in set(paths):
                raise ValueError("Inspection output must not collide with any input path.")
            payload = inspect_inputs(paths)
            destination = _write_json(inspection_output, payload)
            result = {"status": payload["status"], "output": str(destination), "input_count": len(paths)}
        else:
            manifest = create_gif(
                paths, variable=args.variable, layer=args.layer, start=args.start,
                end_exclusive=args.end_exclusive, fps=args.fps, quantiles=tuple(args.quantiles),
                vmin=args.vmin, vmax=args.vmax, cmap=args.cmap, dpi=args.dpi,
                figure_size=tuple(args.figure_size), title_template=args.title_template,
                output=args.output, report=args.report)
            result = {"status": manifest["status"], "output": manifest["output"],
                      "report": manifest["report_path"]}
    except Exception as error:
        print(json.dumps({"status": "error", "type": type(error).__name__, "error": str(error)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
