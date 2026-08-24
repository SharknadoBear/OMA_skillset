from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_gshhs_map(
    land_gdf: gpd.GeoDataFrame,
    coastline_gdf: gpd.GeoDataFrame,
    bbox_gdf: gpd.GeoDataFrame,
    output_png: str | Path,
    *,
    model_bbox_gdf: gpd.GeoDataFrame | None = None,
    source_frame_gdf: gpd.GeoDataFrame | None = None,
    title: str,
    allow_no_basemap: bool = False,
) -> tuple[Path, list[str]]:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    bbox_3857 = bbox_gdf.to_crs(3857)
    minx, miny, maxx, maxy = bbox_3857.total_bounds
    xpad = max((maxx - minx) * 0.04, 500.0)
    ypad = max((maxy - miny) * 0.04, 500.0)

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.set_xlim(minx - xpad, maxx + xpad)
    ax.set_ylim(miny - ypad, maxy + ypad)
    ax.set_facecolor("#d8edf7")

    try:
        import contextily as cx

        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom="auto", attribution_size=6)
    except Exception as exc:
        message = f"Basemap unavailable, using vector-only background: {exc}"
        warnings.append(message)
        if not allow_no_basemap:
            # The vector map is still scientifically useful, so keep it rather than failing.
            pass

    if not land_gdf.empty:
        land_gdf.to_crs(3857).plot(ax=ax, facecolor="#eee6d6", edgecolor="#475569", linewidth=0.5, alpha=0.9)
    if not coastline_gdf.empty:
        coastline_gdf.to_crs(3857).plot(ax=ax, color="#0f172a", linewidth=0.75, alpha=0.9)
    bbox_3857.boundary.plot(ax=ax, color="#dc2626", linewidth=1.2, linestyle="--", label="source footprint")
    if source_frame_gdf is not None and not source_frame_gdf.empty:
        source_frame_gdf.to_crs(3857).plot(ax=ax, color="#dc2626", linewidth=0.8, alpha=0.75)
    if model_bbox_gdf is not None and not model_bbox_gdf.empty:
        model_bbox_gdf.to_crs(3857).boundary.plot(
            ax=ax, color="#2563eb", linewidth=1.8, linestyle="-.", label="model bbox"
        )

    ax.set_title(title)
    ax.set_axis_off()
    if model_bbox_gdf is not None and not model_bbox_gdf.empty:
        ax.legend(loc="best")
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    return output_png, warnings
