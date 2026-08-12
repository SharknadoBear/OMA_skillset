"""Shared native-curvilinear plotting helpers for ROMS maps and movies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ROMSPlotResult:
    artist: Any
    colorbar: Any
    quiver: Any | None
    finite_wet_fraction: float
    wet_cell_count: int
    finite_wet_count: int


def quantile_limits(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("Quantiles must satisfy 0 <= low < high <= 100.")
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("Cannot determine color limits from all-non-finite values.")
    vmin, vmax = (float(value) for value in np.percentile(finite, [low, high]))
    if vmin == vmax:
        delta = max(abs(vmin) * 0.01, 1.0e-12)
        vmin, vmax = vmin - delta, vmax + delta
    return vmin, vmax


def colormap_for_variable(variable: str) -> str:
    name = str(variable).lower()
    if "sal" in name or "speed" in name:
        return "viridis"
    if any(token in name for token in ("zeta", "elevation", "velocity")) or name in {"u", "v"}:
        return "coolwarm"
    if "temp" in name:
        return "plasma"
    return "viridis"


def geographic_aspect(latitude: np.ndarray) -> float:
    finite = np.asarray(latitude, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 1.0
    cosine = abs(float(np.cos(np.deg2rad(np.mean(finite)))))
    return 1.0 / max(cosine, 1.0e-3)


def _validated(lon, lat, mask, values):
    longitude = np.asarray(lon, dtype=float)
    latitude = np.asarray(lat, dtype=float)
    wet_mask = np.asarray(mask, dtype=float)
    scalar = np.asarray(values, dtype=float)
    if longitude.ndim != 2 or len({longitude.shape, latitude.shape, wet_mask.shape, scalar.shape}) != 1:
        raise ValueError("lon, lat, mask, and values must share one two-dimensional rho-grid shape.")
    coordinate_wet = (wet_mask == 1) & np.isfinite(longitude) & np.isfinite(latitude)
    if not np.any(coordinate_wet):
        raise ValueError("ROMS rho grid contains no finite wet coordinates.")
    finite_wet = coordinate_wet & np.isfinite(scalar)
    if not np.any(finite_wet):
        raise ValueError("ROMS scalar has zero finite wet coverage and cannot be rendered.")
    return longitude, latitude, coordinate_wet, np.ma.masked_where(~finite_wet, scalar)


def plot_roms_scalar(
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
) -> ROMSPlotResult:
    """Render one masked scalar frame and optional earth-relative current quivers."""

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise ValueError(f"Color limits require finite vmin < vmax; got {vmin}, {vmax}.")
    longitude, latitude, coordinate_wet, display = _validated(lon, lat, mask, values)
    if method == "pcolormesh":
        artist = ax.pcolormesh(longitude, latitude, display, shading="auto", cmap=cmap,
                               vmin=vmin, vmax=vmax, rasterized=True)
    elif method == "contourf":
        if contour_levels < 2:
            raise ValueError("contour_levels must be at least two.")
        levels = np.linspace(vmin, vmax, contour_levels + 1)
        artist = ax.contourf(longitude, latitude, display, levels=levels, cmap=cmap,
                             vmin=vmin, vmax=vmax, extend="both")
    else:
        raise ValueError("method must be pcolormesh or contourf.")
    colorbar = ax.figure.colorbar(artist, ax=ax, pad=0.025, shrink=0.88) if add_colorbar else None
    if colorbar is not None and colorbar_label:
        colorbar.set_label(colorbar_label)

    quiver = None
    if (quiver_u is None) != (quiver_v is None):
        raise ValueError("Provide both earth-relative quiver components or neither.")
    if quiver_u is not None:
        east, north = np.asarray(quiver_u, dtype=float), np.asarray(quiver_v, dtype=float)
        if east.shape != longitude.shape or north.shape != longitude.shape:
            raise ValueError("Quiver components must share the ROMS rho-grid shape.")
        stride = max(1, int(quiver_stride))
        selection = np.s_[::stride, ::stride]
        valid = coordinate_wet[selection] & np.isfinite(east[selection]) & np.isfinite(north[selection])
        quiver = ax.quiver(
            np.where(valid, longitude[selection], np.nan),
            np.where(valid, latitude[selection], np.nan),
            np.where(valid, east[selection], np.nan),
            np.where(valid, north[selection], np.nan),
            color="black", alpha=0.75, angles="xy", scale_units="xy", scale=quiver_scale,
            width=0.0022, zorder=4,
        )

    valid_lon, valid_lat = longitude[coordinate_wet], latitude[coordinate_wet]
    lon_span, lat_span = float(np.ptp(valid_lon)), float(np.ptp(valid_lat))
    lon_pad, lat_pad = max(0.02 * lon_span, 1.0e-6), max(0.02 * lat_span, 1.0e-6)
    ax.set_xlim(float(np.min(valid_lon) - lon_pad), float(np.max(valid_lon) + lon_pad))
    ax.set_ylim(float(np.min(valid_lat) - lat_pad), float(np.max(valid_lat) + lat_pad))
    ax.set_aspect(geographic_aspect(valid_lat), adjustable="box")
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_title(title)
    ax.grid(True, color="0.75", linewidth=0.45, alpha=0.55)
    finite_count, wet_count = int(np.ma.count(display)), int(np.count_nonzero(coordinate_wet))
    return ROMSPlotResult(artist, colorbar, quiver, finite_count / wet_count, wet_count, finite_count)


def save_roms_scalar_map(output: str, *, lon, lat, mask, values, vmin, vmax,
                         cmap="viridis", title="", colorbar_label=None, dpi=140,
                         figure_size=(7.2, 6.0), **kwargs) -> ROMSPlotResult:
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    try:
        result = plot_roms_scalar(
            ax, lon=lon, lat=lat, mask=mask, values=values, vmin=vmin, vmax=vmax,
            cmap=cmap, title=title, colorbar_label=colorbar_label, **kwargs)
        fig.savefig(output, dpi=dpi)
        return result
    finally:
        plt.close(fig)
