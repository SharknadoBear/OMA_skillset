"""GIF creation for FVCOM scalar output fields using gridded map rendering."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Mapping, Sequence
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from PIL import Image

try:
    from .fvcom_map_tools import plot_fvcom_scalar_map, quantile_limits, transform_values
    from .fvcom_output import decode_fvcom_time, parse_time_like
    from .mesh_tools import mesh_from_output
except ImportError:
    from fvcom_map_tools import plot_fvcom_scalar_map, quantile_limits, transform_values
    from fvcom_output import decode_fvcom_time, parse_time_like
    from mesh_tools import mesh_from_output


ALIASES = {
    "salinity": ("salinity", "salt", "ssl"),
    "salt": ("salinity", "salt", "ssl"),
    "temperature": ("temperature", "temp", "tsl"),
    "temp": ("temperature", "temp", "tsl"),
    "mp": ("mp1", "microplastic", "plastic"),
    "mp1": ("mp1",),
    "floc": ("coarse_sand_1", "coarse_sand_2", "coarse_sand_3"),
}


def resolve_variable_names(ds: Dataset, requested: str | Sequence[str]) -> list[str]:
    """Resolve variable aliases against an open FVCOM dataset."""

    items = [requested] if isinstance(requested, str) else list(requested)
    available = {name.lower(): name for name in ds.variables}
    resolved: list[str] = []
    for item in items:
        key = str(item).lower()
        if key.endswith("*"):
            prefix = key[:-1]
            resolved.extend(name for low, name in available.items() if low.startswith(prefix))
            continue
        candidates = ALIASES.get(key, (item,))
        found = [available[c.lower()] for c in candidates if c.lower() in available]
        if not found and key == "floc":
            found = [name for low, name in available.items() if low.startswith("coarse_sand_")]
        if not found:
            raise KeyError(f"Variable {item!r} not found. Available examples: {list(ds.variables)[:20]}")
        resolved.extend(found)
    out: list[str] = []
    for name in resolved:
        if name not in out:
            out.append(name)
    return out


def sigma_layer_index(ds: Dataset, dim_name: str, layer: str | int) -> int | None:
    """Resolve a vertical layer selector for an FVCOM sigma dimension."""

    if isinstance(layer, int):
        return layer
    layer_key = str(layer).lower()
    if layer_key == "depth_mean":
        return None
    if dim_name in ds.variables:
        sigma = np.asarray(ds.variables[dim_name][:], dtype=float)
        if sigma.ndim > 1:
            sigma = np.nanmean(sigma, axis=tuple(range(1, sigma.ndim)))
        if layer_key == "surface":
            return int(np.nanargmax(sigma))
        if layer_key == "bottom":
            return int(np.nanargmin(sigma))
    if layer_key == "surface":
        return -1
    if layer_key == "bottom":
        return 0
    raise ValueError(f"Unknown layer selector: {layer!r}")


def read_scalar_frame(ds: Dataset, variable: str, time_index: int, layer: str | int = "surface") -> np.ndarray:
    """Read one scalar frame from an FVCOM variable."""

    var = ds.variables[variable]
    dims = var.dimensions
    arr = var[:]
    if "time" in dims:
        axis = dims.index("time")
        arr = np.take(arr, time_index, axis=axis)
        dims = tuple(dim for dim in dims if dim != "time")
    elif "Time" in dims:
        axis = dims.index("Time")
        arr = np.take(arr, time_index, axis=axis)
        dims = tuple(dim for dim in dims if dim != "Time")

    vertical_dims = [dim for dim in dims if dim.lower().startswith("sig")]
    if vertical_dims:
        vdim = vertical_dims[0]
        axis = dims.index(vdim)
        if str(layer).lower() == "depth_mean":
            arr = np.nanmean(arr, axis=axis)
        else:
            idx = sigma_layer_index(ds, vdim, layer)
            arr = np.take(arr, idx, axis=axis)
    return np.asarray(arr, dtype=float).squeeze()


def default_scale(variable: str) -> str:
    """Return default display scale. Use linear unless explicitly overridden."""

    del variable
    return "linear"


def plot_scalar_map(
    ax,
    mesh: Mapping[str, np.ndarray],
    values: np.ndarray,
    title: str = "",
    zoom: str | Sequence[float] = "full",
    cmap: str | None = "auto",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Backward-compatible scalar map wrapper returning the Matplotlib artist."""

    result = plot_fvcom_scalar_map(
        mesh["lon"],
        mesh["lat"],
        mesh["tri"],
        values,
        ax=ax,
        title=title,
        zoom=zoom,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    return result.artist


def _selected_records(files: Sequence[Path], start=None, end=None) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    records: list[tuple[Path, np.ndarray, np.ndarray]] = []
    start64 = parse_time_like(start)
    end64 = parse_time_like(end)
    for path in files:
        with Dataset(path) as ds:
            times = decode_fvcom_time(ds)
        mask = np.ones(len(times), dtype=bool)
        if start64 is not None:
            mask &= times >= start64
        if end64 is not None:
            mask &= times < end64
        idx = np.flatnonzero(mask)
        if len(idx):
            records.append((path, idx, times[idx]))
    return records


def _selected_flat_records(
    files: Sequence[Path],
    *,
    start=None,
    end=None,
    every: int = 6,
    max_frames: int = 400,
) -> list[tuple[Path, int, np.datetime64]]:
    records = _selected_records(files, start=start, end=end)
    flat: list[tuple[Path, int, np.datetime64]] = []
    for path, indices, times in records:
        flat.extend((path, int(i), times[k]) for k, i in enumerate(indices))
    if every > 1:
        flat = flat[::every]
    if max_frames and len(flat) > max_frames:
        pick = np.linspace(0, len(flat) - 1, max_frames).round().astype(int)
        flat = [flat[i] for i in pick]
    return flat


def make_scalar_gif(
    nc_files: str | Path | Sequence[str | Path],
    variable: str,
    out_path: str | Path,
    mesh: Mapping[str, np.ndarray] | None = None,
    layer: str | int = "surface",
    zoom: str | Sequence[float] = "full",
    start=None,
    end=None,
    every: int = 6,
    max_frames: int = 400,
    fps: int = 8,
    cmap: str | None = "auto",
    scale: str | None = None,
    quantiles: tuple[float, float] | None = (2.0, 98.0),
    contour_lines: bool = False,
    background: str = "none",
    grid_spacing_km: float | None = None,
    max_grid_points: int = 1_500_000,
    edge_quantile: float = 5.0,
) -> Path:
    """Create a GIF movie for a scalar FVCOM output variable.

    Color limits are estimated once from all selected frames and reused for
    every map, so the GIF colorbar is fixed through time.
    """

    files = [Path(nc_files)] if isinstance(nc_files, (str, Path)) else [Path(p) for p in nc_files]
    if not files:
        raise FileNotFoundError("No NetCDF files were supplied.")

    flat = _selected_flat_records(files, start=start, end=end, every=every, max_frames=max_frames)
    if not flat:
        raise ValueError("No frames selected for GIF.")

    with Dataset(files[0]) as ds0:
        variable = resolve_variable_names(ds0, variable)[0]
    if mesh is None:
        mesh = mesh_from_output(files[0])
    scale = scale or default_scale(variable)

    raw_frames: list[np.ndarray] = []
    transformed_for_limits: list[np.ndarray] = []
    for path, idx, _ in flat:
        with Dataset(path) as ds:
            frame = read_scalar_frame(ds, variable, idx, layer=layer)
        transformed, _ = transform_values(frame, scale)
        raw_frames.append(frame)
        transformed_for_limits.append(transformed)

    finite_parts = [arr[np.isfinite(arr)].ravel() for arr in transformed_for_limits if np.isfinite(arr).any()]
    if finite_parts:
        finite = np.concatenate(finite_parts)
        vmin, vmax = quantile_limits(finite, quantiles)
    else:
        warnings.warn(f"All selected values for {variable} are non-finite after {scale} transform.", RuntimeWarning)
        vmin, vmax = None, None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(fps, 1)))
    images: list[Image.Image] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for n, ((_, _, timestamp), frame) in enumerate(zip(flat, raw_frames)):
            fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
            title = f"{variable} {layer} {np.datetime_as_string(timestamp, unit='h')}"
            label = variable if scale == "linear" else f"log10({variable})"
            plot_fvcom_scalar_map(
                mesh["lon"],
                mesh["lat"],
                mesh["tri"],
                frame,
                ax=ax,
                variable_name=variable,
                title=title,
                zoom=zoom,
                cmap=cmap,
                scale=scale,
                vmin=vmin,
                vmax=vmax,
                contour_lines=contour_lines,
                background=background,
                grid_spacing_km=grid_spacing_km,
                max_grid_points=max_grid_points,
                edge_quantile=edge_quantile,
                colorbar_label=label,
            )
            frame_path = tmp_dir / f"frame_{n:05d}.png"
            fig.savefig(frame_path, dpi=130)
            plt.close(fig)
            images.append(Image.open(frame_path).convert("P", palette=Image.Palette.ADAPTIVE))

        images[0].save(
            out_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    return out_path


def make_scalar_gif_from_frames(
    lon: np.ndarray,
    lat: np.ndarray,
    nv: np.ndarray,
    frames: np.ndarray,
    out_path: str | Path,
    *,
    variable_name: str = "scalar",
    time_labels: Sequence[str] | None = None,
    zoom: str | Sequence[float] = "full",
    fps: int = 8,
    cmap: str | None = "auto",
    scale: str = "linear",
    quantiles: tuple[float, float] | None = (2.0, 98.0),
    vmin: float | None = None,
    vmax: float | None = None,
    contour_lines: bool = False,
    background: str = "none",
    background_zoom: int | str = "auto",
    filled_alpha: float | None = None,
    grid_spacing_km: float | None = None,
    max_grid_points: int = 1_500_000,
    edge_quantile: float = 5.0,
    colorbar_label: str | None = None,
    title_template: str | None = "",
) -> Path:
    """Create a fixed-colorbar GIF from already-loaded node-frame data.

    ``frames`` must be shaped ``(time, node)``.  Color limits are estimated
    once from all selected frames unless ``vmin``/``vmax`` are supplied.
    """

    arr = np.asarray(frames, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"frames must be 2-D as (time, node); got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("frames contains no time records.")

    transformed, _ = transform_values(arr, scale)
    if vmin is None or vmax is None:
        finite = transformed[np.isfinite(transformed)]
        auto_vmin, auto_vmax = quantile_limits(finite, quantiles) if finite.size else (None, None)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(fps, 1)))
    images: list[Image.Image] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i_time, frame in enumerate(arr):
            fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
            if title_template is None:
                title = ""
            elif title_template:
                label = "" if time_labels is None or i_time >= len(time_labels) else str(time_labels[i_time])
                title = title_template.format(variable=variable_name, time=label, index=i_time)
            else:
                title = ""
            plot_fvcom_scalar_map(
                lon,
                lat,
                nv,
                frame,
                ax=ax,
                variable_name=variable_name,
                title=title,
                zoom=zoom,
                cmap=cmap,
                scale=scale,
                quantiles=quantiles,
                vmin=vmin,
                vmax=vmax,
                contour_lines=contour_lines,
                background=background,
                background_zoom=background_zoom,
                filled_alpha=filled_alpha,
                grid_spacing_km=grid_spacing_km,
                max_grid_points=max_grid_points,
                edge_quantile=edge_quantile,
                colorbar_label=colorbar_label,
            )
            frame_path = tmp_dir / f"frame_{i_time:05d}.png"
            fig.savefig(frame_path, dpi=130)
            plt.close(fig)
            images.append(Image.open(frame_path).convert("P", palette=Image.Palette.ADAPTIVE))

        images[0].save(
            out_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    return out_path
