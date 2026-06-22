"""Satellite-background CUSP diagnostic maps."""

from __future__ import annotations

from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box


def _choose_zoom(bbox: tuple[float, float, float, float]) -> int:
    west, south, east, north = bbox
    span = max(east - west, north - south)
    if span > 2.0:
        return 8
    if span > 1.0:
        return 9
    if span > 0.5:
        return 10
    return 11


def plot_coastline_satellite(
    gdf: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    output_png: str | Path,
    *,
    title: str,
    basemap_provider: str = "Esri.WorldImagery",
    allow_no_basemap: bool = False,
) -> tuple[Path, list[str]]:
    """Plot clipped CUSP lines over a satellite basemap."""

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    west, south, east, north = bbox
    bbox_gdf = gpd.GeoDataFrame(geometry=[box(west, south, east, north)], crs="EPSG:4326")
    plot_gdf = gdf.to_crs(3857) if not gdf.empty else gdf.set_crs("EPSG:4326", allow_override=True).to_crs(3857)
    plot_bbox = bbox_gdf.to_crs(3857)
    minx, miny, maxx, maxy = plot_bbox.total_bounds

    xpad = max((maxx - minx) * 0.04, 500.0)
    ypad = max((maxy - miny) * 0.04, 500.0)

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)

    try:
        provider = cx.providers
        for part in basemap_provider.split("."):
            provider = getattr(provider, part)
        cx.add_basemap(ax, source=provider, zoom=_choose_zoom(bbox), attribution_size=6)
    except Exception as exc:
        message = f"Satellite basemap failed: {exc}"
        warnings.append(message)
        if not allow_no_basemap:
            plt.close(fig)
            raise RuntimeError(message) from exc
        ax.set_facecolor("#d7e3ea")

    if not plot_gdf.empty:
        plot_gdf.plot(ax=ax, color="#00ffff", linewidth=1.1, alpha=0.95)
        plot_gdf.plot(ax=ax, color="#002b36", linewidth=0.25, alpha=0.7)
    plot_bbox.boundary.plot(ax=ax, color="red", linewidth=1.4)

    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png, warnings


def plot_merged_coastline_satellite(
    cusp_gdf: gpd.GeoDataFrame,
    fallback_gdf: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    output_png: str | Path,
    *,
    title: str,
    basemap_provider: str = "Esri.WorldImagery",
    allow_no_basemap: bool = False,
) -> tuple[Path, list[str]]:
    """Plot primary CUSP and retained fallback lines over satellite imagery."""

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    west, south, east, north = bbox
    bbox_gdf = gpd.GeoDataFrame(geometry=[box(west, south, east, north)], crs="EPSG:4326")
    plot_bbox = bbox_gdf.to_crs(3857)
    minx, miny, maxx, maxy = plot_bbox.total_bounds

    xpad = max((maxx - minx) * 0.04, 500.0)
    ypad = max((maxy - miny) * 0.04, 500.0)

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)

    try:
        provider = cx.providers
        for part in basemap_provider.split("."):
            provider = getattr(provider, part)
        cx.add_basemap(ax, source=provider, zoom=_choose_zoom(bbox), attribution_size=6)
    except Exception as exc:
        message = f"Satellite basemap failed: {exc}"
        warnings.append(message)
        if not allow_no_basemap:
            plt.close(fig)
            raise RuntimeError(message) from exc
        ax.set_facecolor("#d7e3ea")

    if not cusp_gdf.empty:
        cusp_gdf.to_crs(3857).plot(ax=ax, color="#00ffff", linewidth=1.05, alpha=0.95, label="CUSP")
    if not fallback_gdf.empty:
        fallback_gdf.to_crs(3857).plot(ax=ax, color="#ff9f1a", linewidth=1.15, alpha=0.95, label="OSM fallback")
    plot_bbox.boundary.plot(ax=ax, color="red", linewidth=1.4)
    if not cusp_gdf.empty or not fallback_gdf.empty:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.8)

    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png, warnings
