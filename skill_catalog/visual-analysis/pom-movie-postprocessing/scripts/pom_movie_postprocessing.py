#!/usr/bin/env python3
"""Inspect POM files and render fixed-color-scale scalar GIFs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from .pom_map_tools import colormap_for_variable, plot_pom_scalar, quantile_limits
    from .pom_output import inspect_inputs as inspect_pom_inputs
    from .pom_output import load_scalar_series
except ImportError:  # Direct script execution.
    from pom_map_tools import colormap_for_variable, plot_pom_scalar, quantile_limits
    from pom_output import inspect_inputs as inspect_pom_inputs
    from pom_output import load_scalar_series


MANIFEST_VERSION = "pom_movie_manifest_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value.astype("datetime64[ns]"), unit="ns") + "Z"
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
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, destination)
    return destination


def _sha256(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten_inputs(inputs: Iterable[str | Path | Sequence[str | Path]]) -> list[Path]:
    flat: list[Path] = []
    for item in inputs:
        if isinstance(item, (str, Path)):
            flat.append(Path(item).expanduser().resolve())
        else:
            flat.extend(Path(value).expanduser().resolve() for value in item)
    if not flat:
        raise ValueError("At least one --input NetCDF file is required.")
    missing = [str(path) for path in flat if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input files do not exist: {missing}")
    return flat


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _iso_utc(value: np.datetime64) -> str:
    text = np.datetime_as_string(np.datetime64(value, "ns"), unit="ns")
    return f"{text}Z"


def _expanded_limits(vmin: float, vmax: float) -> tuple[float, float]:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Color limits must be finite.")
    if vmin > vmax:
        raise ValueError(f"vmin ({vmin}) must be less than vmax ({vmax}).")
    if vmin == vmax:
        delta = max(abs(vmin) * 0.01, 1.0e-12)
        return vmin - delta, vmax + delta
    return vmin, vmax


def inspect_inputs(inputs: Sequence[str | Path]) -> dict[str, Any]:
    """Inspect POM inputs without loading a movie into memory."""

    paths = _flatten_inputs(inputs)
    report = inspect_pom_inputs(paths)
    return {
        "schema_version": "pom_movie_inspection_v1",
        "created_utc": _utc_now(),
        "input_count": len(paths),
        "inputs": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in paths
        ],
        "pom": _jsonable(report),
    }


def create_gif(
    inputs: Sequence[str | Path],
    *,
    variable: str,
    layer: str = "surface",
    output: str | Path,
    report: str | Path | None = None,
    start: str | None = None,
    end_exclusive: str | None = None,
    fps: float = 4.0,
    quantiles: tuple[float, float] = (2.0, 98.0),
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    dpi: int = 120,
    figure_size: tuple[float, float] = (7.2, 6.0),
    title_template: str = "{variable} {layer} | {time}",
) -> dict[str, Any]:
    """Create a fixed-scale POM GIF and return its provenance manifest."""

    paths = _flatten_inputs(inputs)
    if not variable.strip():
        raise ValueError("variable must not be empty.")
    if fps <= 0:
        raise ValueError("fps must be greater than zero.")
    if dpi < 40:
        raise ValueError("dpi must be at least 40.")
    q_low, q_high = (float(quantiles[0]), float(quantiles[1]))
    if not (0.0 <= q_low < q_high <= 100.0):
        raise ValueError("quantiles must satisfy 0 <= low < high <= 100.")
    if (vmin is None) != (vmax is None):
        raise ValueError("Provide both --vmin and --vmax, or neither.")

    series = load_scalar_series(
        paths,
        variable=variable,
        layer=layer,
        start=start,
        end_exclusive=end_exclusive,
        snap_tolerance_seconds=60.0,
    )
    values = np.asarray(_field(series, "values"), dtype=float)
    times = np.asarray(_field(series, "times")).astype("datetime64[ns]")
    original_times = np.asarray(_field(series, "original_times", times)).astype("datetime64[ns]")
    offsets = np.asarray(_field(series, "time_offsets_seconds", np.zeros(len(times))), dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Expected scalar frames shaped (time, y, x); received {values.shape}.")
    if values.shape[0] == 0 or len(times) == 0:
        raise ValueError("No frames remain after time selection.")
    if values.shape[0] != len(times):
        raise ValueError("The loaded time and scalar-frame counts differ.")
    if len(times) > 1 and np.any(np.diff(times.astype("int64")) <= 0):
        raise ValueError("Selected timestamps are not unique and strictly increasing.")

    grid = _field(series, "grid")
    lon = np.asarray(_field(grid, "lon"), dtype=float)
    lat = np.asarray(_field(grid, "lat"), dtype=float)
    mask = np.asarray(_field(grid, "mask", np.ones_like(lon)), dtype=float)
    if lon.shape != lat.shape or lon.shape != mask.shape or values.shape[1:] != lon.shape:
        raise ValueError(
            f"Grid/frame mismatch: lon={lon.shape}, lat={lat.shape}, mask={mask.shape}, frames={values.shape}."
        )
    wet = (mask == 1) & np.isfinite(lon) & np.isfinite(lat)
    wet_count = int(np.count_nonzero(wet))
    if wet_count == 0:
        raise ValueError("The POM grid contains no finite cells with mask == 1.")
    coverage = np.asarray(
        [np.count_nonzero(np.isfinite(frame) & wet) / wet_count for frame in values], dtype=float
    )
    all_nan = np.flatnonzero(coverage == 0.0)
    if all_nan.size:
        raise ValueError(f"All-NaN wet frames are not renderable: {all_nan.tolist()}")

    finite = values[:, wet]
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("No finite wet values are available for color limits.")
    if vmin is None:
        auto_min, auto_max = quantile_limits(finite, q_low, q_high)
        color_method = "full_series_quantiles"
        vmin, vmax = _expanded_limits(float(auto_min), float(auto_max))
    else:
        color_method = "explicit"
        vmin, vmax = _expanded_limits(float(vmin), float(vmax))

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = cmap or colormap_for_variable(variable)
    duration_ms = max(1, int(round(1000.0 / fps)))
    images: list[Image.Image] = []
    temporary_frames_cleaned = False
    temporary_path: str | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="pom_movie_") as tmp_name:
            temporary_path = tmp_name
            tmp_dir = Path(tmp_name)
            for index, (timestamp, frame) in enumerate(zip(times, values, strict=True)):
                fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
                title = title_template.format(
                    variable=variable,
                    layer=layer,
                    time=_iso_utc(timestamp),
                    index=index,
                )
                plot_pom_scalar(
                    ax,
                    lon=lon,
                    lat=lat,
                    mask=mask,
                    values=frame,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    title=title,
                    colorbar_label=variable,
                )
                png_path = tmp_dir / f"frame_{index:05d}.png"
                fig.savefig(png_path, dpi=dpi)
                plt.close(fig)
                with Image.open(png_path) as frame_image:
                    images.append(frame_image.convert("P", palette=Image.Palette.ADAPTIVE).copy())

            images[0].save(
                output_path,
                format="GIF",
                save_all=True,
                append_images=images[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
                disposal=2,
            )
        temporary_frames_cleaned = temporary_path is None or not Path(temporary_path).exists()
    finally:
        for frame_image in images:
            frame_image.close()
        plt.close("all")

    with Image.open(output_path) as gif:
        gif_frame_count = int(getattr(gif, "n_frames", 1))
        gif_size = [int(gif.width), int(gif.height)]
        gif_duration = int(gif.info.get("duration", duration_ms))
    if gif_frame_count != len(times):
        raise RuntimeError(f"GIF contains {gif_frame_count} frames; expected {len(times)}.")

    record_sources = _field(series, "record_sources", [None] * len(times))
    record_indices = _field(series, "record_indices", [None] * len(times))
    frames = [
        {
            "index": index,
            "time_utc": _iso_utc(times[index]),
            "original_time_utc": _iso_utc(original_times[index]),
            "normalization_offset_seconds": float(offsets[index]),
            "finite_wet_fraction": float(coverage[index]),
            "source": None if record_sources is None else str(record_sources[index]),
            "source_record_index": None if record_indices is None else int(record_indices[index]),
        }
        for index in range(len(times))
    ]
    source_metadata = _field(series, "sources", [])
    warnings: list[str] = []
    minimum_coverage = float(np.min(coverage))
    if minimum_coverage < 0.95:
        warnings.append(
            f"Minimum finite wet coverage is {minimum_coverage:.3%}, below the 95% acceptance guideline."
        )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_VERSION,
        "created_utc": _utc_now(),
        "status": "pass" if not warnings else "pass_with_warnings",
        "warnings": warnings,
        "inputs": _jsonable(source_metadata)
        if source_metadata
        else [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in paths
        ],
        "request": {
            "variable": variable,
            "layer": layer,
            "start_utc": start,
            "end_utc_exclusive": end_exclusive,
        },
        "resolved": _jsonable(_field(series, "resolution", {})),
        "grid": {
            "shape": [int(lon.shape[0]), int(lon.shape[1])],
            "wet_cell_count": wet_count,
            "geometry_sha256": _field(grid, "geometry_sha256"),
            "longitude_range": [float(np.nanmin(lon[wet])), float(np.nanmax(lon[wet]))],
            "latitude_range": [float(np.nanmin(lat[wet])), float(np.nanmax(lat[wet]))],
        },
        "selection": {
            "frame_count": len(times),
            "first_time_utc": _iso_utc(times[0]),
            "last_time_utc": _iso_utc(times[-1]),
            "unique_monotonic": True,
            "duplicate_times_removed": int(_field(series, "duplicate_times_removed", 0)),
            "snap_tolerance_seconds": 60.0,
        },
        "fixed_color_limits": {
            "method": color_method,
            "vmin": float(vmin),
            "vmax": float(vmax),
            "quantiles_percent": [q_low, q_high] if color_method == "full_series_quantiles" else None,
        },
        "coverage": {
            "minimum_finite_wet_fraction": minimum_coverage,
            "mean_finite_wet_fraction": float(np.mean(coverage)),
            "all_nan_frame_count": 0,
        },
        "rendering": {
            "format": "GIF",
            "fps_requested": float(fps),
            "frame_duration_ms": duration_ms,
            "cmap": cmap,
            "dpi": int(dpi),
            "figure_size_inches": [float(figure_size[0]), float(figure_size[1])],
            "movie_quivers": False,
            "temporary_frames_cleaned": temporary_frames_cleaned,
        },
        "frames": frames,
        "output": {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
            "frame_count": gif_frame_count,
            "pixel_size": gif_size,
            "frame_duration_ms_observed": gif_duration,
        },
    }
    if report is not None:
        report_path = Path(report).expanduser().resolve()
        manifest["report_path"] = str(report_path)
        _write_json(report_path, manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and animate scalar fields on NOAA POM curvilinear grids."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect POM inputs and write JSON metadata.")
    inspect_parser.add_argument("--input", action="append", nargs="+", required=True, help="NetCDF input(s).")
    inspect_parser.add_argument("--output", required=True, help="Inspection JSON path.")

    gif_parser = subparsers.add_parser("gif", help="Create a fixed-scale scalar GIF.")
    gif_parser.add_argument("--input", action="append", nargs="+", required=True, help="NetCDF input(s).")
    gif_parser.add_argument("--variable", required=True, help="Scalar name, current_speed, or wind_speed.")
    gif_parser.add_argument("--layer", default="surface", help="surface, near_surface, bottom, depth_average, or index:N.")
    gif_parser.add_argument("--start", help="Inclusive UTC selection bound.")
    gif_parser.add_argument("--end-exclusive", help="Exclusive UTC selection bound.")
    gif_parser.add_argument("--fps", type=float, default=4.0, help="Playback frames per second (default: 4).")
    gif_parser.add_argument("--quantiles", nargs=2, type=float, metavar=("LOW", "HIGH"), default=(2.0, 98.0))
    gif_parser.add_argument("--vmin", type=float, help="Explicit fixed minimum; requires --vmax.")
    gif_parser.add_argument("--vmax", type=float, help="Explicit fixed maximum; requires --vmin.")
    gif_parser.add_argument("--cmap", help="Matplotlib colormap (default selected by variable).")
    gif_parser.add_argument("--dpi", type=int, default=120)
    gif_parser.add_argument("--figure-size", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"), default=(7.2, 6.0))
    gif_parser.add_argument("--title-template", default="{variable} {layer} | {time}")
    gif_parser.add_argument("--output", required=True, help="Output GIF path.")
    gif_parser.add_argument("--report", required=True, help="Output manifest JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    inputs = _flatten_inputs(args.input)
    if args.command == "inspect":
        payload = inspect_inputs(inputs)
        path = _write_json(args.output, payload)
        print(json.dumps({"status": "pass", "output": str(path), "input_count": len(inputs)}, indent=2))
        return 0
    manifest = create_gif(
        inputs,
        variable=args.variable,
        layer=args.layer,
        start=args.start,
        end_exclusive=args.end_exclusive,
        fps=args.fps,
        quantiles=tuple(args.quantiles),
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        dpi=args.dpi,
        figure_size=tuple(args.figure_size),
        title_template=args.title_template,
        output=args.output,
        report=args.report,
    )
    print(json.dumps({"status": "pass", "output": manifest["output"], "report": manifest["report_path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
