from __future__ import annotations

import math
from typing import Any

from .geometry import KM_PER_DEG_LAT, RegionBPoly, _az_to_vec, _lon_scale, _wrap_lon


def _xy_to_lonlat(region: RegionBPoly, x_km: float, y_km: float) -> list[float]:
    origin_lon, origin_lat = region._local_origin()
    return [_wrap_lon(origin_lon + x_km / _lon_scale(origin_lat)), origin_lat + y_km / KM_PER_DEG_LAT]


def _polygon_area_xy(xy: list[tuple[float, float]]) -> float:
    area = 0.0
    for i, p0 in enumerate(xy):
        p1 = xy[(i + 1) % len(xy)]
        area += p0[0] * p1[1] - p1[0] * p0[1]
    return area / 2.0


def _orientation(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _validate_xy(xy: list[tuple[float, float]]) -> None:
    if len(xy) != 4:
        raise ValueError("RegionBPoly adjustment requires exactly four vertices")
    if abs(_polygon_area_xy(xy)) < 1e-6:
        raise ValueError("Adjusted polygon has near-zero area")
    if _segments_intersect(xy[0], xy[1], xy[2], xy[3]) or _segments_intersect(xy[1], xy[2], xy[3], xy[0]):
        raise ValueError("Adjusted polygon is self-intersecting")


def _pivot_xy(region: RegionBPoly, op: dict[str, Any]) -> tuple[float, float]:
    if "pivot_lonlat" in op:
        lon, lat = op["pivot_lonlat"]
        return region.local_xy_km(float(lon), float(lat))
    pivot = str(op.get("pivot", "center")).lower()
    if pivot in {"center", "centroid"}:
        return (0.0, 0.0)
    if pivot in {"offshore_midpoint", "offshore_side_midpoint"}:
        return region.local_xy_km(*region.offshore_edge_midpoint_lonlat())
    raise ValueError(f"Unsupported pivot {pivot!r}")


def _rotate_xy(xy: list[tuple[float, float]], op: dict[str, Any], pivot: tuple[float, float]) -> list[tuple[float, float]]:
    angle = math.radians(float(op.get("angle_deg", op.get("degrees", 0.0))))
    ca, sa = math.cos(angle), math.sin(angle)
    px, py = pivot
    out = []
    for x, y in xy:
        dx, dy = x - px, y - py
        out.append((px + ca * dx - sa * dy, py + sa * dx + ca * dy))
    return out


def _scale_xy(region: RegionBPoly, xy: list[tuple[float, float]], op: dict[str, Any], pivot: tuple[float, float]) -> list[tuple[float, float]]:
    if "factor" in op:
        along_factor = across_factor = float(op["factor"])
    else:
        along_factor = float(op.get("along_factor", op.get("length_factor", 1.0)))
        across_factor = float(op.get("across_factor", op.get("width_factor", 1.0)))
    angle = float(op.get("axis_angle_deg", region.approximate_orientation_deg()))
    ux, uy = _az_to_vec(angle)
    vx, vy = uy, -ux
    px, py = pivot
    out = []
    for x, y in xy:
        dx, dy = x - px, y - py
        along = dx * ux + dy * uy
        across = dx * vx + dy * vy
        out.append((px + along * along_factor * ux + across * across_factor * vx, py + along * along_factor * uy + across * across_factor * vy))
    return out


def _reshape_xy(xy: list[tuple[float, float]], op: dict[str, Any]) -> list[tuple[float, float]]:
    deltas = op.get("vertex_delta_km", op.get("vertex_deltas_km"))
    if deltas is None:
        raise ValueError("reshape operation requires vertex_delta_km")
    if isinstance(deltas, dict):
        out = list(xy)
        for key, delta in deltas.items():
            idx = int(key)
            dx, dy = delta
            out[idx] = (out[idx][0] + float(dx), out[idx][1] + float(dy))
        return out
    if len(deltas) != 4:
        raise ValueError("reshape vertex_delta_km must contain four [east_km, north_km] deltas")
    return [(x + float(delta[0]), y + float(delta[1])) for (x, y), delta in zip(xy, deltas)]


def _expand_side_xy(
    region: RegionBPoly,
    xy: list[tuple[float, float]],
    op: dict[str, Any],
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Move one selected side outward in the local projected frame."""
    side_index = int(op.get("side_index", -1))
    if side_index not in range(4):
        raise ValueError("expand_side requires side_index in 0..3")
    offshore_side_index = region.offshore_side_index()
    distance_km = float(op.get("distance_km", 0.0))
    if not math.isfinite(distance_km) or distance_km <= 0.0:
        raise ValueError("expand_side requires a positive finite distance_km")

    start_index = side_index
    end_index = (side_index + 1) % 4
    x0, y0 = xy[start_index]
    x1, y1 = xy[end_index]
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("expand_side cannot move a zero-length side")

    # Pick the normal pointing away from the polygon centroid. This is stable for
    # clockwise and counter-clockwise vertex orderings.
    nx, ny = -dy / length, dx / length
    cx = sum(x for x, _ in xy) / 4.0
    cy = sum(y for _, y in xy) / 4.0
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if nx * (mx - cx) + ny * (my - cy) < 0.0:
        nx, ny = -nx, -ny

    out = list(xy)
    shift_x, shift_y = nx * distance_km, ny * distance_km
    out[start_index] = (x0 + shift_x, y0 + shift_y)
    out[end_index] = (x1 + shift_x, y1 + shift_y)
    details = {
        "side_index": side_index,
        "offshore_side_index": offshore_side_index,
        "distance_km": distance_km,
        "outward_unit_normal_xy": [nx, ny],
        "moved_vertex_indices": [start_index, end_index],
    }
    return out, details


def apply_adjustment_manifest(region: RegionBPoly, manifest: dict[str, Any]) -> tuple[RegionBPoly, list[dict[str, Any]]]:
    operations = manifest.get("operations")
    if operations is None:
        operations = [manifest]
    xy = [region.local_xy_km(*p) for p in region.polygon_lonlat()[:-1]]
    history: list[dict[str, Any]] = []
    offshore = float(manifest.get("offshore_azimuth_deg", region.offshore_azimuth_deg))

    for op in operations:
        op_type = str(op.get("operation", op.get("type", ""))).lower()
        if op_type == "rotate":
            pivot = _pivot_xy(region, op)
            xy = _rotate_xy(xy, op, pivot)
            details = {}
        elif op_type in {"scale", "resize", "enlarge", "shrink"}:
            pivot = _pivot_xy(region, op)
            xy = _scale_xy(region, xy, op, pivot)
            details = {}
        elif op_type == "reshape":
            xy = _reshape_xy(xy, op)
            details = {}
        elif op_type == "expand_side":
            xy, details = _expand_side_xy(region, xy, op)
        else:
            raise ValueError(f"Unsupported adjustment operation {op_type!r}")
        _validate_xy(xy)
        history.append({"operation": op_type, "parameters": op, **details})
        if "offshore_azimuth_deg" in op:
            offshore = float(op["offshore_azimuth_deg"])

    pts = [_xy_to_lonlat(region, x, y) for x, y in xy]
    return RegionBPoly(pts, offshore, edge_labels=region.edge_labels), history
