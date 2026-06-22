"""Matplotlib-only CUDEM diagnostic maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_bathymetry_map(
    ds: xr.Dataset,
    output_png: str | Path,
    *,
    title: str = "CUDEM Bathymetry",
    bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    """Write a non-interactive elevation/depth map PNG."""

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    lon = ds["lon"].values
    lat = ds["lat"].values
    elevation = ds["elevation_m"].values

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    if np.isfinite(elevation).any():
        vmin = float(np.nanpercentile(elevation, 2))
        vmax = float(np.nanpercentile(elevation, 98))
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0
    else:
        vmin, vmax = -1.0, 1.0

    mesh = ax.pcolormesh(lon, lat, elevation, shading="auto", cmap="terrain", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
    cbar.set_label("Elevation relative to source datum (m)")
    try:
        ax.contour(lon, lat, elevation, levels=[0.0], colors="black", linewidths=0.7)
    except Exception:
        pass
    if bbox is not None:
        west, south, east, north = bbox
        ax.plot(
            [west, east, east, west, west],
            [south, south, north, north, south],
            color="red",
            linewidth=1.2,
        )
    ax.set_title(title)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png


def plot_source_id_map(
    ds: xr.Dataset,
    output_png: str | Path,
    *,
    title: str = "Bathymetry Source Coverage",
    bbox: tuple[float, float, float, float] | None = None,
) -> Path:
    """Write a categorical source-id map PNG."""

    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    lon = ds["lon"].values
    lat = ds["lat"].values
    source_id = ds["source_id"].values

    labels = {0: "none", 1: "CUDEM", 2: "CRM", 3: "ETOPO"}
    colors = ["#d9d9d9", "#0072b2", "#009e73", "#d55e00"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.pcolormesh(lon, lat, source_id, shading="auto", cmap=cmap, norm=norm)
    handles = [
        Patch(facecolor=colors[i], edgecolor="black", linewidth=0.4, label=labels[i])
        for i in range(len(colors))
        if np.any(source_id == i)
    ]
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    if bbox is not None:
        west, south, east, north = bbox
        ax.plot(
            [west, east, east, west, west],
            [south, south, north, north, south],
            color="red",
            linewidth=1.2,
        )
    ax.set_title(title)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_png
