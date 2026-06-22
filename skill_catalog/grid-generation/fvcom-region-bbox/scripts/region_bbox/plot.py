from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .basemap import add_basemap
from .geometry import RegionBox, bbox_to_points
from .scoring import ingredient_points


def _plot_box(ax, region: RegionBox, color="tab:red", label="RegionBox"):
    pts = region.polygon_lonlat()
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2.2, label=label)
    ax.scatter([region.center_lon], [region.center_lat], color=color, s=22, zorder=5)
    mid = region.offshore_edge_midpoint_lonlat()
    ax.scatter([mid[0]], [mid[1]], color="tab:orange", s=45, marker="*", zorder=6, label="open-boundary reference side")


def _plot_ingredients(ax, ingredients):
    for item in ingredients:
        pts = ingredient_points(item)
        if not pts:
            continue
        pts2 = pts + [pts[0]]
        color = "tab:green" if item.get("required", True) else "0.5"
        ax.plot([p[0] for p in pts2], [p[1] for p in pts2], color=color, lw=1.0, alpha=0.75)
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        ax.text(cx, cy, item.get("id", ""), fontsize=6, color=color)


def plot_region_map(path: str | Path, region: RegionBox, ingredients=None, title="", bbox=None, basemap_provider="street") -> dict:
    ingredients = ingredients or []
    if bbox is None:
        bbox = region.envelope_bbox()
        dx = max(0.2, (bbox[2] - bbox[0]) * 0.15)
        dy = max(0.2, (bbox[3] - bbox[1]) * 0.15)
        bbox = [bbox[0] - dx, bbox[1] - dy, bbox[2] + dx, bbox[3] + dy]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    base = add_basemap(ax, bbox, provider=basemap_provider)
    _plot_ingredients(ax, ingredients)
    _plot_box(ax, region)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or "FVCOM RegionBox")
    ax.grid(True, color="0.85", lw=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return base


def side_focus_records(region: RegionBox, run_dir: Path, name: str, side_indices: list[int], fractions: list[float], radius_km=45.0, basemap_provider="street") -> list[dict]:
    records = []
    side_dir = run_dir / f"{name}_side_focus"
    side_dir.mkdir(parents=True, exist_ok=True)
    sides = region.sides()
    for idx in side_indices:
        side = sides[idx]
        p0, p1 = side["start_lonlat"], side["end_lonlat"]
        for frac in fractions:
            lon = p0[0] + frac * (p1[0] - p0[0])
            lat = p0[1] + frac * (p1[1] - p0[1])
            half_lon = radius_km / 111.32
            half_lat = radius_km / 111.32
            bbox = [lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat]
            label = {0.125: "q1", 0.15: "start", 0.375: "q2", 0.5: "middle", 0.625: "q3", 0.85: "end", 0.875: "q4"}.get(frac, f"{frac:.2f}")
            map_path = side_dir / f"{name}_side_{idx}_{label}_focus_map.png"
            base = plot_region_map(map_path, region, [], title=f"{name} side {idx} {label}", bbox=bbox, basemap_provider=basemap_provider)
            records.append(
                {
                    "side_index": idx,
                    "side_name": f"side_{idx}",
                    "position": label,
                    "fraction": frac,
                    "center_lonlat": [lon, lat],
                    "radius_km": radius_km,
                    "bbox": bbox,
                    "side_start_lonlat": p0,
                    "side_end_lonlat": p1,
                    "map_path": str(map_path),
                    "basemap": base,
                }
            )
    return records

