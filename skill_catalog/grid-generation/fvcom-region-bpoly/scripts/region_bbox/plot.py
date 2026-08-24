from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .basemap import add_basemap
from .geometry import RegionBox, bbox_to_points
from .scoring import ingredient_points


def _unwrap_lon(lon: float, origin: float) -> float:
    x = float(lon)
    while x - origin > 180.0:
        x -= 360.0
    while origin - x > 180.0:
        x += 360.0
    return x


def _wrap_lon(lon: float) -> float:
    x = float(lon)
    while x > 180.0:
        x -= 360.0
    while x < -180.0:
        x += 360.0
    return x


def _display_origin(region: RegionBox) -> float | None:
    if not getattr(region, "crosses_antimeridian", lambda: False)():
        return None
    if hasattr(region, "_local_origin"):
        return float(region._local_origin()[0])
    return float(region.center_lon)


def _to_display_lon(lon: float, origin: float | None) -> float:
    return _unwrap_lon(lon, origin) if origin is not None else float(lon)


def _to_display_point(point, origin: float | None) -> list[float]:
    return [_to_display_lon(point[0], origin), float(point[1])]


def _display_bbox_from_points(points: list[list[float]], pad_fraction: float = 0.15) -> list[float]:
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)
    dx = max(0.2, (east - west) * pad_fraction)
    dy = max(0.2, (north - south) * pad_fraction)
    return [west - dx, south - dy, east + dx, north + dy]


def _display_bbox(region: RegionBox, ingredients: list[dict], bbox, origin: float | None) -> list[float]:
    if bbox is not None:
        west, south, east, north = map(float, bbox)
        west_d = _to_display_lon(west, origin)
        east_d = _to_display_lon(east, origin)
        if east_d < west_d:
            east_d += 360.0
        return [west_d, south, east_d, north]

    points = [_to_display_point(p, origin) for p in region.polygon_lonlat()[:-1]]
    for item in ingredients:
        points.extend(_to_display_point(p, origin) for p in ingredient_points(item))
    return _display_bbox_from_points(points)


def _format_lon_tick(value: float, _pos=None) -> str:
    wrapped = _wrap_lon(value)
    if abs(wrapped) < 1e-9:
        return "0"
    suffix = "E" if wrapped > 0 else "W"
    return f"{abs(wrapped):.0f}{suffix}"


def _plot_box(
    ax,
    region: RegionBox,
    *,
    origin: float | None = None,
    color="tab:red",
    label="RegionBPoly",
    linestyle="-",
    linewidth=2.2,
    alpha=1.0,
    show_reference=True,
    show_envelope=True,
) -> None:
    pts = [_to_display_point(p, origin) for p in region.polygon_lonlat()]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=linewidth, ls=linestyle, alpha=alpha, label=label)
    center_x = _to_display_lon(region.center_lon, origin)
    ax.scatter([center_x], [region.center_lat], color=color, s=22, zorder=5, alpha=alpha)
    if show_reference:
        mid = _to_display_point(region.offshore_edge_midpoint_lonlat(), origin)
        ax.scatter([mid[0]], [mid[1]], color="tab:orange", s=45, marker="*", zorder=6, label="offshore side reference")
    if show_envelope and hasattr(region, "envelope_bbox"):
        bbox = region.envelope_bbox()
        bb = bbox_to_points(bbox) + [bbox_to_points(bbox)[0]]
        bb = [_to_display_point(p, origin) for p in bb]
        ax.plot([p[0] for p in bb], [p[1] for p in bb], color="tab:purple", lw=1.0, ls=":", label="derived fetch envelope")
    for side in region.sides():
        p = _to_display_point(side["midpoint_lonlat"], origin)
        ax.text(p[0], p[1], side["side_name"], fontsize=7, color=color, ha="center", va="center")


def _plot_ingredients(ax, ingredients: list[dict], origin: float | None = None) -> None:
    for item in ingredients:
        pts = ingredient_points(item)
        if not pts:
            continue
        pts2 = pts + [pts[0]]
        pts2 = [_to_display_point(p, origin) for p in pts2]
        color = "tab:green" if item.get("required", True) else "0.5"
        ax.plot([p[0] for p in pts2], [p[1] for p in pts2], color=color, lw=1.0, alpha=0.75)
        cx = sum(p[0] for p in pts2[:-1]) / max(1, len(pts2) - 1)
        cy = sum(p[1] for p in pts2[:-1]) / max(1, len(pts2) - 1)
        ax.text(cx, cy, item.get("id", ""), fontsize=6, color=color)


def plot_region_map(
    path: str | Path,
    region: RegionBox,
    ingredients=None,
    title="",
    bbox=None,
    basemap_provider="topo",
    basemap_zoom: int | None = None,
    open_boundary_reference: dict | None = None,
    comparison_region: RegionBox | None = None,
    comparison_label: str = "previous RegionBPoly",
) -> dict:
    ingredients = ingredients or []
    origin = _display_origin(region)
    display_bbox = _display_bbox(region, ingredients, bbox, origin)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(display_bbox[0], display_bbox[2])
    ax.set_ylim(display_bbox[1], display_bbox[3])
    output_path = Path(path)
    base = add_basemap(
        ax,
        display_bbox,
        provider=basemap_provider,
        zoom=basemap_zoom,
        search_roots=[output_path.parent],
    )
    base["display_frame"] = {
        "longitude_origin": origin,
        "uses_unwrapped_longitude": origin is not None,
        "bbox": display_bbox,
        "lon_span_deg": display_bbox[2] - display_bbox[0],
    }

    _plot_ingredients(ax, ingredients, origin)
    if comparison_region is not None:
        _plot_box(
            ax,
            comparison_region,
            origin=origin,
            color="0.25",
            label=comparison_label,
            linestyle="--",
            linewidth=1.8,
            alpha=0.8,
            show_reference=False,
            show_envelope=False,
        )
    _plot_box(ax, region, origin=origin)
    if open_boundary_reference:
        ref = open_boundary_reference.get("snapped") or open_boundary_reference
        if isinstance(ref, dict) and "lon" in ref and "lat" in ref:
            ref_x = _to_display_lon(ref["lon"], origin)
            ax.scatter([ref_x], [ref["lat"]], color="black", s=55, marker="x", zorder=8, label="offshore point")
            ax.annotate("offshore point", (ref_x, ref["lat"]), xytext=(6, 6), textcoords="offset points", fontsize=8, color="black")

    ax.set_xlabel("Longitude" if origin is None else "Longitude (antimeridian display frame)")
    ax.set_ylabel("Latitude")
    ax.set_title(title or "FVCOM RegionBPoly")
    if origin is not None:
        ax.xaxis.set_major_formatter(FuncFormatter(_format_lon_tick))
    ax.grid(True, color="0.85", lw=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.08, top=0.92)
    p = output_path
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return base


def side_focus_records(
    region: RegionBox,
    run_dir: Path,
    name: str,
    side_indices: list[int],
    fractions: list[float],
    radius_km=45.0,
    basemap_provider="topo",
    basemap_zoom: int | None = None,
) -> list[dict]:
    records = []
    side_dir = run_dir / "side_focus"
    side_dir.mkdir(parents=True, exist_ok=True)
    sides = region.sides()
    for idx in side_indices:
        side = sides[idx]
        p0, p1 = side["start_lonlat"], side["end_lonlat"]
        p1_lon = p1[0]
        if p1_lon - p0[0] > 180:
            p1_lon -= 360
        elif p0[0] - p1_lon > 180:
            p1_lon += 360
        for frac in fractions:
            lon = p0[0] + frac * (p1_lon - p0[0])
            while lon > 180:
                lon -= 360
            while lon < -180:
                lon += 360
            lat = p0[1] + frac * (p1[1] - p0[1])
            half_lon = radius_km / 111.32
            half_lat = radius_km / 111.32
            bbox = [lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat]
            label = {0.125: "q1", 0.15: "start", 0.375: "q2", 0.5: "middle", 0.625: "q3", 0.85: "end", 0.875: "q4"}.get(frac, f"{frac:.2f}")
            map_path = side_dir / f"s{idx}_{label}.png"
            base = plot_region_map(map_path, region, [], title=f"{name} side {idx} {label}", bbox=bbox, basemap_provider=basemap_provider, basemap_zoom=basemap_zoom)
            records.append(
                {
                    "side_index": idx,
                    "side_name": side["side_name"],
                    "position": label,
                    "fraction": frac,
                    "center_lonlat": [lon, lat],
                    "radius_km": radius_km,
                    "bbox": bbox,
                    "display_frame": base.get("display_frame"),
                    "side_start_lonlat": p0,
                    "side_end_lonlat": p1,
                    "map_path": str(map_path),
                    "basemap": base,
                }
            )
    return records
