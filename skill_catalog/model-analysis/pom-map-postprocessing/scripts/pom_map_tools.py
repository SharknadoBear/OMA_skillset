"""Shared native-curvilinear plotting helpers for POM maps and movies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class POMPlotResult:
    """Artists and QA metadata returned by a POM scalar rendering."""

    artist: Any
    colorbar: Any
    quiver: Any | None
    finite_wet_fraction: float
    wet_cell_count: int
    finite_wet_count: int


def quantile_limits(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    """Return finite robust limits, expanding a constant field slightly."""

    if not (0.0 <= low < high <= 100.0):
        raise ValueError("Quantiles must satisfy 0 <= low < high <= 100.")
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot determine color limits from all-non-finite values.")
    vmin, vmax = (float(value) for value in np.percentile(finite, [low, high]))
    if vmin == vmax:
        delta = max(abs(vmin) * 0.01, 1.0e-12)
        vmin -= delta
        vmax += delta
    return vmin, vmax


def colormap_for_variable(variable: str) -> str:
    """Return a dependency-free Matplotlib colormap for common POM fields."""

    name = str(variable).lower()
    if any(token in name for token in ("speed", "wind", "sal")):
        return "viridis"
    if any(token in name for token in ("zeta", "elevation", "velocity")) or name in {"u", "v"}:
        return "coolwarm"
    if "temp" in name:
        return "plasma"
    return "viridis"


def geographic_aspect(latitude: np.ndarray) -> float:
    """Return a stable y/x display ratio for geographic degree coordinates."""

    finite = np.asarray(latitude, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    cosine = abs(float(np.cos(np.deg2rad(np.nanmean(finite)))))
    return 1.0 / max(cosine, 1.0e-3)


def _validate_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    mask_arr = np.asarray(mask, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    shapes = {lon_arr.shape, lat_arr.shape, mask_arr.shape, value_arr.shape}
    if len(shapes) != 1 or lon_arr.ndim != 2:
        raise ValueError(
            f"lon, lat, mask, and values must share a two-dimensional shape; got "
            f"{lon_arr.shape}, {lat_arr.shape}, {mask_arr.shape}, {value_arr.shape}."
        )
    coordinate_wet = (mask_arr == 1) & np.isfinite(lon_arr) & np.isfinite(lat_arr)
    if not np.any(coordinate_wet):
        raise ValueError("POM grid contains no finite cells with mask == 1.")
    finite_wet = coordinate_wet & np.isfinite(value_arr)
    display = np.ma.masked_where(~finite_wet, value_arr)
    return lon_arr, lat_arr, mask_arr, display, coordinate_wet


def plot_pom_scalar(
    ax,
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
    values: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    title: str = "",
    colorbar_label: str | None = None,
    method: str = "pcolormesh",
    contour_levels: int = 16,
    add_colorbar: bool = True,
    quiver_u: np.ndarray | None = None,
    quiver_v: np.ndarray | None = None,
    quiver_stride: int = 8,
    quiver_scale: float | None = None,
) -> POMPlotResult:
    """Render one masked scalar frame on its native POM curvilinear grid."""

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise ValueError(f"Color limits must be finite with vmin < vmax; got {vmin}, {vmax}.")
    lon_arr, lat_arr, _, display, coordinate_wet = _validate_grid(lon, lat, mask, values)
    method_key = method.lower()
    if method_key == "pcolormesh":
        artist = ax.pcolormesh(
            lon_arr,
            lat_arr,
            display,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
    elif method_key == "contourf":
        if contour_levels < 2:
            raise ValueError("contour_levels must be at least two.")
        levels = np.linspace(vmin, vmax, contour_levels + 1)
        artist = ax.contourf(
            lon_arr,
            lat_arr,
            display,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extend="both",
        )
    else:
        raise ValueError("method must be 'pcolormesh' or 'contourf'.")

    colorbar = None
    if add_colorbar:
        colorbar = ax.figure.colorbar(artist, ax=ax, pad=0.025, shrink=0.88)
        if colorbar_label:
            colorbar.set_label(colorbar_label)

    quiver = None
    if (quiver_u is None) != (quiver_v is None):
        raise ValueError("Provide both quiver_u and quiver_v, or neither.")
    if quiver_u is not None:
        u = np.asarray(quiver_u, dtype=float)
        v = np.asarray(quiver_v, dtype=float)
        if u.shape != lon_arr.shape or v.shape != lon_arr.shape:
            raise ValueError("Quiver components must share the native POM grid shape.")
        stride = max(1, int(quiver_stride))
        selection = np.s_[::stride, ::stride]
        valid = coordinate_wet[selection] & np.isfinite(u[selection]) & np.isfinite(v[selection])
        qlon = np.where(valid, lon_arr[selection], np.nan)
        qlat = np.where(valid, lat_arr[selection], np.nan)
        qu = np.where(valid, u[selection], np.nan)
        qv = np.where(valid, v[selection], np.nan)
        quiver = ax.quiver(
            qlon,
            qlat,
            qu,
            qv,
            color="black",
            alpha=0.75,
            angles="xy",
            scale_units="xy",
            scale=quiver_scale,
            width=0.0022,
            zorder=4,
        )

    valid_lon = lon_arr[coordinate_wet]
    valid_lat = lat_arr[coordinate_wet]
    lon_span = float(np.nanmax(valid_lon) - np.nanmin(valid_lon))
    lat_span = float(np.nanmax(valid_lat) - np.nanmin(valid_lat))
    lon_pad = max(lon_span * 0.02, 1.0e-6)
    lat_pad = max(lat_span * 0.02, 1.0e-6)
    ax.set_xlim(float(np.nanmin(valid_lon) - lon_pad), float(np.nanmax(valid_lon) + lon_pad))
    ax.set_ylim(float(np.nanmin(valid_lat) - lat_pad), float(np.nanmax(valid_lat) + lat_pad))
    ax.set_aspect(geographic_aspect(valid_lat), adjustable="box")
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_title(title)
    ax.grid(True, color="0.75", linewidth=0.45, alpha=0.55)

    finite_wet_count = int(np.ma.count(display))
    wet_cell_count = int(np.count_nonzero(coordinate_wet))
    return POMPlotResult(
        artist=artist,
        colorbar=colorbar,
        quiver=quiver,
        finite_wet_fraction=finite_wet_count / wet_cell_count,
        wet_cell_count=wet_cell_count,
        finite_wet_count=finite_wet_count,
    )


def save_pom_scalar_map(
    output: str,
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
    values: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    title: str = "",
    colorbar_label: str | None = None,
    dpi: int = 140,
    figure_size: tuple[float, float] = (7.2, 6.0),
    **kwargs,
) -> POMPlotResult:
    """Save one standalone POM scalar map and return its QA metadata."""

    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    try:
        result = plot_pom_scalar(
            ax,
            lon=lon,
            lat=lat,
            mask=mask,
            values=values,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title=title,
            colorbar_label=colorbar_label,
            **kwargs,
        )
        fig.savefig(output, dpi=dpi)
        return result
    finally:
        plt.close(fig)
