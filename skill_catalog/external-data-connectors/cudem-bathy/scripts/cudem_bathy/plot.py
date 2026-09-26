"""Matplotlib-only CUDEM diagnostic maps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _source_legend(ds: xr.Dataset) -> dict[int, str]:
    """Prefer variable metadata; global metadata is the older fallback."""
    raw = ds["source_id"].attrs.get("legend", ds.attrs.get("source_legend"))
    if raw is None:
        return {}
    try:
        declared = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        if not isinstance(declared, dict):
            raise ValueError("expected an object mapping integer IDs to names")
        labels = {}
        for key, name in declared.items():
            source = int(str(key))
            if not isinstance(name, str) or not name.strip() or source in labels:
                raise ValueError("expected unique integer IDs and nonempty source names")
            labels[source] = name.strip()
        return labels
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid source_id legend metadata: {exc}") from exc


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
    """Plot declared source identities, including legacy and sparse encodings.

    Missing metadata/IDs are labeled as undeclared, never guessed from a
    current or historical numeric encoding. Nonfinite pixels remain masked.
    """

    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    lon = ds["lon"].values
    lat = ds["lat"].values
    source_id = ds["source_id"].values

    declared = _source_legend(ds)
    present = np.unique(source_id[np.isfinite(source_id)])
    if np.any(present != np.floor(present)):
        raise ValueError("source_id contains noninteger categories")
    # Colors follow source names, not IDs: CRM is 2 in legacy files and 3
    # in the current writer. Dense display indices avoid sparse-ID clipping.
    styles = {
        "none": ("none", "#d9d9d9"),
        "cudem": ("CUDEM", "#0072b2"),
        "nbs_bluetopo": ("NBS BlueTopo", "#56b4e9"),
        "crm": ("CRM", "#009e73"),
        "etopo": ("ETOPO", "#d55e00"),
        "regional_candidate": ("regional candidate", "#cc79a7"),
    }
    labels, colors = [], []
    categories = np.full(source_id.shape, np.nan)
    for index, source in enumerate(present):
        name = declared.get(int(source), f"source ID {int(source)} (undeclared)")
        label, color = styles.get(name.lower(), (name, plt.get_cmap("tab20")(index % 20)))
        labels.append(label)
        colors.append(color)
        categories[source_id == source] = index
    cmap = ListedColormap(colors or ["#d9d9d9"])
    norm = BoundaryNorm(np.arange(cmap.N + 1) - 0.5, cmap.N)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.pcolormesh(lon, lat, np.ma.masked_invalid(categories), shading="auto", cmap=cmap, norm=norm)
    handles = [
        Patch(facecolor=colors[i], edgecolor="black", linewidth=0.4, label=labels[i])
        for i in range(len(colors))
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
