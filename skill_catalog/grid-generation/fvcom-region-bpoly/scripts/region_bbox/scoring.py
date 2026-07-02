from __future__ import annotations

from .geometry import RegionBox, bbox_to_points
from .normalization import canonical_region_key


def ingredient_points(item: dict) -> list[list[float]]:
    geom = item.get("geometry", [])
    if item.get("type") == "bbox":
        return bbox_to_points(geom)
    if item.get("type") == "point":
        return [list(map(float, geom))]
    return [list(map(float, p)) for p in geom]


def _on_region_boundary(region: RegionBox, lon: float, lat: float, tol_km: float = 1e-6) -> bool:
    px, py = region.local_xy_km(lon, lat)
    for side in region.sides():
        ax, ay = region.local_xy_km(*side["start_lonlat"])
        bx, by = region.local_xy_km(*side["end_lonlat"])
        vx, vy = bx - ax, by - ay
        den = vx * vx + vy * vy
        if den <= 0:
            continue
        t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / den))
        sx, sy = ax + t * vx, ay + t * vy
        if ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5 <= tol_km:
            return True
    return False


def score_region_box(region: RegionBox, ingredients: list[dict]) -> dict:
    out = []
    missing = []
    for item in ingredients:
        pts = ingredient_points(item)
        inside_count = sum(1 for lon, lat in pts if region.contains_lonlat(lon, lat) or _on_region_boundary(region, lon, lat))
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


def _polygon_area_km2(region: RegionBox) -> float:
    xy = [region.local_xy_km(*p) for p in region.polygon_lonlat()[:-1]]
    area = 0.0
    for i, p0 in enumerate(xy):
        p1 = xy[(i + 1) % len(xy)]
        area += p0[0] * p1[1] - p1[0] * p0[1]
    return abs(area) / 2.0


def _required_feature_extent_area_km2(region: RegionBox, ingredients: list[dict]) -> float:
    pts = []
    for item in ingredients:
        if item.get("required", True):
            pts.extend(region.local_xy_km(*p) for p in ingredient_points(item))
    if not pts:
        return 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def _unwrap_lon(lon: float, origin: float) -> float:
    x = float(lon)
    while x - origin > 180.0:
        x -= 360.0
    while origin - x > 180.0:
        x += 360.0
    return x


def _compact_lon_span_deg(region: RegionBox) -> float:
    pts = region.polygon_lonlat()[:-1]
    if not pts:
        return 0.0
    origin = pts[0][0]
    lons = [_unwrap_lon(p[0], origin) for p in pts]
    return max(lons) - min(lons)


def _bbox_dict(region: RegionBox) -> dict:
    bbox = region.envelope_bbox()
    return {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]}


def _domain_scale(request: dict | str, ingredients: list[dict]) -> str:
    key = canonical_region_key(request)
    if key == "murderkill":
        return "small_estuary"
    if any(item.get("domain_scale") == "small_estuary" for item in ingredients):
        return "small_estuary"
    return "regional"


def _is_obstruction_guard(item: dict) -> bool:
    return item.get("role") == "offshore_boundary_exclusion" or item.get("category") in {
        "offshore_boundary_exclusion",
        "obstruction_guard",
    }


def _feature_xy_bounds(region: RegionBox, item: dict) -> tuple[float, float, float, float]:
    pts = ingredient_points(item)
    xs = []
    ys = []
    for lon, lat in pts:
        x, y = region.local_xy_km(lon, lat)
        xs.append(x)
        ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def _feature_center(item: dict) -> tuple[float, float] | None:
    if item.get("type") != "bbox" or len(item.get("geometry", [])) != 4:
        return None
    west, south, east, north = [float(v) for v in item["geometry"]]
    return ((west + east) / 2.0, (south + north) / 2.0)


def _point_in_xy_bounds(x: float, y: float, bounds: tuple[float, float, float, float]) -> bool:
    xmin, ymin, xmax, ymax = bounds
    return xmin <= x <= xmax and ymin <= y <= ymax


def _ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    o1 = _ccw(a, b, c)
    o2 = _ccw(a, b, d)
    o3 = _ccw(c, d, a)
    o4 = _ccw(c, d, b)
    if o1 == 0 and _point_in_xy_bounds(c[0], c[1], (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))):
        return True
    if o2 == 0 and _point_in_xy_bounds(d[0], d[1], (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))):
        return True
    if o3 == 0 and _point_in_xy_bounds(a[0], a[1], (min(c[0], d[0]), min(c[1], d[1]), max(c[0], d[0]), max(c[1], d[1]))):
        return True
    if o4 == 0 and _point_in_xy_bounds(b[0], b[1], (min(c[0], d[0]), min(c[1], d[1]), max(c[0], d[0]), max(c[1], d[1]))):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _segment_intersects_rect(
    a: tuple[float, float],
    b: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> bool:
    xmin, ymin, xmax, ymax = bounds
    if _point_in_xy_bounds(a[0], a[1], bounds) or _point_in_xy_bounds(b[0], b[1], bounds):
        return True
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    return any(_segments_intersect(a, b, c, d) for c, d in edges)


def _distance_point_to_segment_km(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    den = vx * vx + vy * vy
    if den <= 0:
        return ((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / den))
    sx, sy = a[0] + t * vx, a[1] + t * vy
    return ((p[0] - sx) ** 2 + (p[1] - sy) ** 2) ** 0.5


def _guard_intersection_record(region: RegionBox, guard: dict, offshore_side: dict) -> dict:
    bounds = _feature_xy_bounds(region, guard)
    guard_points = list(ingredient_points(guard))
    center = _feature_center(guard)
    if center:
        guard_points.append([center[0], center[1]])
    guard_point_inside_region = any(region.contains_lonlat(lon, lat) or _on_region_boundary(region, lon, lat) for lon, lat in guard_points)
    region_vertex_inside_guard = False
    for lon, lat in region.polygon_lonlat()[:-1]:
        x, y = region.local_xy_km(lon, lat)
        if _point_in_xy_bounds(x, y, bounds):
            region_vertex_inside_guard = True
            break

    side_a = region.local_xy_km(*offshore_side["start_lonlat"])
    side_b = region.local_xy_km(*offshore_side["end_lonlat"])
    side_intersects_guard = _segment_intersects_rect(side_a, side_b, bounds)
    guard_xy = [region.local_xy_km(lon, lat) for lon, lat in guard_points]
    min_side_distance_km = min((_distance_point_to_segment_km(p, side_a, side_b) for p in guard_xy), default=None)
    guard_distance_km = float(guard.get("guard_distance_km") or 0.0)
    side_corridor_near_guard = min_side_distance_km is not None and guard_distance_km > 0 and min_side_distance_km <= guard_distance_km
    intersects_region = guard_point_inside_region or region_vertex_inside_guard
    blocks = bool(guard.get("blocks_final_pass", True)) and (intersects_region or side_intersects_guard)
    return {
        "guard_id": guard.get("id"),
        "label": guard.get("label"),
        "role": guard.get("role"),
        "category": guard.get("category"),
        "guard_distance_km": guard_distance_km,
        "intersects_region": intersects_region,
        "offshore_side_intersects_guard": side_intersects_guard,
        "offshore_side_corridor_near_guard": side_corridor_near_guard,
        "min_side_distance_km": min_side_distance_km,
        "blocks_final_pass": blocks,
        "notes": guard.get("notes", ""),
    }


def score_bpoly_quality(
    region: RegionBox,
    ingredients: list[dict],
    request: dict | str,
    domain_type: str,
    boundary_policy: str,
    open_boundary_reference: dict | None,
    basemap_meta: dict | None = None,
) -> dict:
    """Score practical RegionBPoly quality beyond required-feature containment."""
    key = canonical_region_key(request)
    bbox = _bbox_dict(region)
    region_area = _polygon_area_km2(region)
    feature_area = _required_feature_extent_area_km2(region, ingredients)
    tightness_ratio = feature_area / region_area if region_area > 0 else 0.0
    oversize_factor = region_area / feature_area if feature_area > 0 else None
    domain_scale = _domain_scale(request, ingredients)
    length_width = region.approximate_length_width_km() if hasattr(region, "approximate_length_width_km") else None
    if isinstance(length_width, dict):
        approx_length_km = length_width.get("length_km")
        approx_width_km = length_width.get("width_km")
    elif isinstance(length_width, (tuple, list)) and len(length_width) >= 2:
        approx_length_km = float(length_width[0])
        approx_width_km = float(length_width[1])
    else:
        approx_length_km = None
        approx_width_km = None
    taxonomy: list[dict] = []

    tightness_status = "good"
    if feature_area > 0 and tightness_ratio < 0.08:
        tightness_status = "fail"
        taxonomy.append({"code": "loose_feature_fit", "severity": "fail", "message": "RegionBPoly area is far larger than required feature extent."})
    elif feature_area > 0 and tightness_ratio < 0.16:
        tightness_status = "needs_review"
        taxonomy.append({"code": "loose_feature_fit", "severity": "review", "message": "RegionBPoly is loose relative to required features; visual tightening may be useful."})
    small_estuary_limits = None
    if domain_scale == "small_estuary":
        small_estuary_limits = {"max_region_area_km2": 1700.0, "max_width_km": 36.0, "max_oversize_factor": 2.6}
        small_fail = region_area > small_estuary_limits["max_region_area_km2"]
        if approx_width_km is not None:
            small_fail = small_fail or approx_width_km > small_estuary_limits["max_width_km"]
        if oversize_factor is not None:
            small_fail = small_fail or oversize_factor > small_estuary_limits["max_oversize_factor"]
        if small_fail:
            tightness_status = "fail"
            taxonomy.append(
                {
                    "code": "small_estuary_box_too_large",
                    "severity": "fail",
                    "message": "Small-estuary RegionBPoly is too broad for creek/estuary-scale modeling.",
                }
            )

    wrong_region_warnings: list[str] = []
    if key == "delaware":
        if bbox["west"] < -76.0 and bbox["south"] < 39.3:
            wrong_region_warnings.append("Delaware domain extends far enough west/south to risk including Chesapeake Bay context.")
            taxonomy.append({"code": "chesapeake_overreach_risk", "severity": "review", "message": "Delaware box may be oversized toward Chesapeake Bay; offshore side can still be acceptable."})
    elif key == "southeast_alaska":
        if bbox["west"] < -150.0 or bbox["east"] > -120.0 or bbox["south"] < 48.0 or bbox["north"] > 65.0:
            wrong_region_warnings.append("Southeast Alaska normalization appears to have fallen back to a continent-scale region.")
            taxonomy.append({"code": "continent_scale_fallback", "severity": "fail", "message": "Southeast Alaska box is outside regional archipelago scale."})
    elif key == "hawaii_island":
        if bbox["north"] > 20.75 or bbox["west"] < -157.0 or bbox["east"] > -153.8:
            wrong_region_warnings.append("Hawaii Island-only box may be stepping toward neighboring islands instead of staying cleanly around the Big Island.")
            taxonomy.append({"code": "hawaii_island_scope_ambiguity", "severity": "review", "message": "Hawaii Island-only scope needs visual confirmation against nearby islands."})
    elif key == "cook_inlet":
        if bbox["east"] > -148.3 and bbox["north"] > 59.0:
            wrong_region_warnings.append("Cook Inlet wave-fetch bpoly risks including Prince William Sound on the east side.")
            taxonomy.append(
                {
                    "code": "prince_william_sound_overreach_risk",
                    "severity": "review",
                    "message": "Cook Inlet bpoly should avoid Prince William Sound unless the prompt explicitly asks for it.",
                }
            )
    elif key == "mobile_bay":
        if bbox["east"] > -87.45:
            wrong_region_warnings.append("Mobile Bay bpoly risks unnecessary Perdido Bay / Wolf Bay inclusion.")
            taxonomy.append(
                {
                    "code": "perdido_wolf_bay_overreach_risk",
                    "severity": "review",
                    "message": "Mobile Bay bpoly should not include Perdido Bay or Wolf Bay unless explicitly requested.",
                }
            )
        if bbox["west"] > -88.85:
            wrong_region_warnings.append("Mobile Bay Gulf-facing gate may not extend far enough west to land beyond Horn Island.")
            taxonomy.append(
                {
                    "code": "open_gate_landing_blocked_by_horn_island",
                    "severity": "fail",
                    "message": "Extend the Mobile Bay offshore gate west enough that the downstream arc can land on solid coast rather than being cut by Horn Island.",
                }
            )

    expected_domain = "lake" if key.startswith("lake_") else ("island" if key in {"aleutian", "hawaii_state", "hawaii_island"} else "coastal")
    domain_status = "good" if domain_type == expected_domain else "fail"
    if domain_status == "fail":
        taxonomy.append({"code": "domain_type_mismatch", "severity": "fail", "message": f"Domain type {domain_type!r} does not match expected {expected_domain!r}."})

    offshore_side_index = region.offshore_side_index()
    offshore_side = region.sides()[offshore_side_index]
    offshore_warnings: list[str] = []
    obstruction_guards = [_guard_intersection_record(region, item, offshore_side) for item in ingredients if _is_obstruction_guard(item)]
    if domain_type == "lake":
        if boundary_policy != "no_open_boundary":
            offshore_warnings.append("Lake domain should not be labeled with an ocean open-boundary policy.")
            taxonomy.append({"code": "lake_open_boundary_policy", "severity": "fail", "message": "Lake domain has an ocean-style open boundary policy."})
        if open_boundary_reference:
            offshore_warnings.append("Lake domain should not emit an ocean open-boundary reference point.")
            taxonomy.append({"code": "lake_open_boundary_reference", "severity": "review", "message": "Lake domain emitted an open-boundary reference point."})
    else:
        if not open_boundary_reference:
            offshore_warnings.append("Coastal/island domain is missing the offshore point used for downstream coastline-anchor snapping.")
            taxonomy.append({"code": "missing_offshore_point", "severity": "fail", "message": "No offshore side reference point was emitted."})
        elif open_boundary_reference.get("side_index") not in {None, offshore_side_index}:
            offshore_warnings.append("Snapped offshore point is not on the side selected by offshore azimuth.")
            taxonomy.append({"code": "offshore_snap_side_mismatch", "severity": "review", "message": "Offshore point snapped to a different side than the offshore azimuth selector."})
        if key in {"aleutian", "hawaii_state", "southeast_alaska"}:
            offshore_warnings.append("Island/archipelago offshore side requires visual confirmation that no broad-region island cuts the open side.")
            taxonomy.append({"code": "island_crossing_visual_check", "severity": "review", "message": "Confirm offshore side is not cut by islands in the broad region."})
        if key == "hawaii_island" and offshore_side["midpoint_lonlat"][1] > 20.35:
            offshore_warnings.append("Hawaii Island-only offshore side is too close to the neighboring-island side.")
            taxonomy.append({"code": "hawaii_offshore_side_near_neighbors", "severity": "fail", "message": "Selected offshore side is on the northern/neighbor-island side."})
        for guard in obstruction_guards:
            if guard.get("blocks_final_pass"):
                offshore_warnings.append(f"Offshore side or domain intersects obstruction guard: {guard.get('guard_id')}.")
                taxonomy.append(
                    {
                        "code": "offshore_obstruction_guard_intersection",
                        "severity": "fail",
                        "message": f"Selected RegionBPoly conflicts with offshore obstruction guard {guard.get('guard_id')}.",
                        "guard_id": guard.get("guard_id"),
                    }
                )

    compact_span = _compact_lon_span_deg(region)
    antimeridian_status = "good"
    if region.crosses_antimeridian() and compact_span > 80.0:
        antimeridian_status = "needs_review"
        taxonomy.append({"code": "wide_antimeridian_frame", "severity": "review", "message": "Antimeridian domain is broad; generated maps must be checked in compact longitude frame."})
    if not region.crosses_antimeridian() and key == "aleutian":
        antimeridian_status = "fail"
        taxonomy.append({"code": "missing_antimeridian_crossing", "severity": "fail", "message": "Aleutian request should use antimeridian-safe longitude handling."})

    basemap_meta = basemap_meta or {}
    display_frame = basemap_meta.get("display_frame", {})
    display_lon_span = display_frame.get("lon_span_deg")
    map_warnings: list[str] = []
    if not basemap_meta.get("enabled", False):
        map_warnings.append("Map background is missing.")
        taxonomy.append({"code": "missing_background_map", "severity": "fail", "message": "A visible background map is required."})
    if display_lon_span is not None and display_lon_span > 120.0:
        map_warnings.append("Map frame is too wide to visually review the target region.")
        taxonomy.append({"code": "unreviewable_map_extent", "severity": "fail", "message": "Map longitude span is too wide for practical visual QA."})

    blocking_failure = any(item["severity"] == "fail" for item in taxonomy)
    return {
        "schema_version": "bpoly_quality_score_v1",
        "canonical_region_key": key,
        "tight_feature_fit": {
            "status": tightness_status,
            "domain_scale": domain_scale,
            "feature_extent_area_km2": feature_area,
            "region_area_km2": region_area,
            "feature_to_region_area_ratio": tightness_ratio,
            "oversize_factor": oversize_factor,
            "approx_length_km": approx_length_km,
            "approx_width_km": approx_width_km,
            "small_estuary_limits": small_estuary_limits,
        },
        "wrong_region_inclusion": {
            "warnings": wrong_region_warnings,
            "bbox": bbox,
        },
        "domain_type_qa": {
            "expected_domain_type": expected_domain,
            "actual_domain_type": domain_type,
            "status": domain_status,
        },
        "offshore_side_qa": {
            "selected_side_index": offshore_side_index,
            "selected_side_name": offshore_side.get("side_name"),
            "boundary_policy": boundary_policy,
            "warnings": offshore_warnings,
            "obstruction_guards": obstruction_guards,
        },
        "antimeridian_qa": {
            "status": antimeridian_status,
            "crosses_antimeridian": region.crosses_antimeridian(),
            "compact_lon_span_deg": compact_span,
            "map_display_lon_span_deg": display_lon_span,
        },
        "map_usability_qa": {
            "enabled": basemap_meta.get("enabled", False),
            "source": basemap_meta.get("source"),
            "status": basemap_meta.get("status"),
            "display_frame": display_frame,
            "warnings": map_warnings,
        },
        "failure_taxonomy": taxonomy,
        "blocking_failure": blocking_failure,
    }
