from __future__ import annotations

from .geometry import RegionBox, bbox_to_points


def ingredient_points(item: dict) -> list[list[float]]:
    geom = item.get("geometry", [])
    if item.get("type") == "bbox":
        return bbox_to_points(geom)
    return [list(map(float, p)) for p in geom]


def score_region_box(region: RegionBox, ingredients: list[dict]) -> dict:
    out = []
    missing = []
    for item in ingredients:
        pts = ingredient_points(item)
        inside_count = sum(1 for lon, lat in pts if region.contains_lonlat(lon, lat))
        inside = inside_count == len(pts)
        rec = dict(item)
        rec.update(
            {
                "inside": inside,
                "inside_point_count": inside_count,
                "total_point_count": len(pts),
                "missing_reason": "" if inside else "not all ingredient geometry points are inside the RegionBox",
            }
        )
        out.append(rec)
        if item.get("required", True) and not inside:
            missing.append(item.get("id", "unknown"))
    required_count = sum(1 for i in ingredients if i.get("required", True))
    return {
        "all_required_inside": not missing,
        "required_count": required_count,
        "ingredient_count": len(ingredients),
        "missing_required_ids": missing,
        "ingredients": out,
    }

