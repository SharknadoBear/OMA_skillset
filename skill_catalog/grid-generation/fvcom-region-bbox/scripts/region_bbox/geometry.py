from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

KM_PER_DEG_LAT = 111.32


def _lon_scale(lat: float) -> float:
    return max(10.0, KM_PER_DEG_LAT * math.cos(math.radians(lat)))


def _az_to_vec(az_deg: float) -> tuple[float, float]:
    a = math.radians(az_deg)
    return math.sin(a), math.cos(a)


def _vec_to_az(east: float, north: float) -> float:
    return (math.degrees(math.atan2(east, north)) + 360.0) % 360.0


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _unwrap_pair(a: float, b: float) -> tuple[float, float]:
    if b - a > 180:
        b -= 360
    elif a - b > 180:
        b += 360
    return a, b


@dataclass
class RegionBox:
    center_lon: float
    center_lat: float
    length_km: float
    width_km: float
    orientation_deg: float
    offshore_azimuth_deg: float

    def _offset_lonlat(self, east_km: float, north_km: float) -> list[float]:
        return [
            self.center_lon + east_km / _lon_scale(self.center_lat),
            self.center_lat + north_km / KM_PER_DEG_LAT,
        ]

    def local_xy_km(self, lon: float, lat: float) -> tuple[float, float]:
        return (
            (lon - self.center_lon) * _lon_scale(self.center_lat),
            (lat - self.center_lat) * KM_PER_DEG_LAT,
        )

    def polygon_lonlat(self) -> list[list[float]]:
        ux, uy = _az_to_vec(self.orientation_deg)
        vx, vy = uy, -ux
        hl = self.length_km / 2.0
        hw = self.width_km / 2.0
        pts = []
        for a, b in [(hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)]:
            pts.append(self._offset_lonlat(a * ux + b * vx, a * uy + b * vy))
        pts.append(pts[0])
        return pts

    def sides(self) -> list[dict]:
        pts = self.polygon_lonlat()[:-1]
        out = []
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i + 1) % 4]
            x0, y0 = self.local_xy_km(*p0)
            x1, y1 = self.local_xy_km(*p1)
            dx, dy = x1 - x0, y1 - y0
            # Polygon is clockwise in local coordinates; outward normal is left.
            nx, ny = -dy, dx
            norm = math.hypot(nx, ny) or 1.0
            out.append(
                {
                    "side_index": i,
                    "side_name": f"side_{i}",
                    "start_lonlat": p0,
                    "end_lonlat": p1,
                    "midpoint_lonlat": [(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0],
                    "outward_azimuth_deg": _vec_to_az(nx / norm, ny / norm),
                }
            )
        return out

    def offshore_side_index(self) -> int:
        u = _az_to_vec(self.offshore_azimuth_deg)
        best_i, best_score = 0, -999.0
        for side in self.sides():
            v = _az_to_vec(side["outward_azimuth_deg"])
            score = _dot(u, v)
            if score > best_score:
                best_i, best_score = side["side_index"], score
        return best_i

    def offshore_edge_midpoint_lonlat(self) -> list[float]:
        return self.sides()[self.offshore_side_index()]["midpoint_lonlat"]

    def envelope_bbox(self) -> list[float]:
        pts = self.polygon_lonlat()[:-1]
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        return [min(lons), min(lats), max(lons), max(lats)]

    def crosses_antimeridian(self) -> bool:
        pts = self.polygon_lonlat()
        return any(abs(pts[i + 1][0] - pts[i][0]) > 180 for i in range(len(pts) - 1))

    def map_visibility_warnings(self) -> list[str]:
        if not self.crosses_antimeridian():
            return []
        return [
            "RegionBox crosses the antimeridian; final map must visibly show the complete polygon and downstream tools should use polygon_lonlat/RegionBox geometry rather than a naive lon/lat bbox."
        ]

    def contains_lonlat(self, lon: float, lat: float) -> bool:
        x, y = self.local_xy_km(lon, lat)
        poly = [self.local_xy_km(*p) for p in self.polygon_lonlat()]
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y):
                x_int = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
                if x < x_int:
                    inside = not inside
            j = i
        return inside

    def snap_point_to_edge(self, lon: float, lat: float) -> dict:
        px, py = self.local_xy_km(lon, lat)
        best = None
        for side in self.sides():
            a = self.local_xy_km(*side["start_lonlat"])
            b = self.local_xy_km(*side["end_lonlat"])
            vx, vy = b[0] - a[0], b[1] - a[1]
            den = vx * vx + vy * vy or 1.0
            t = max(0.0, min(1.0, ((px - a[0]) * vx + (py - a[1]) * vy) / den))
            sx, sy = a[0] + t * vx, a[1] + t * vy
            d_km = math.hypot(px - sx, py - sy)
            snapped = self._offset_lonlat(sx, sy)
            if best is None or d_km < best["distance_km"]:
                best = {
                    "snapped": {"lon": snapped[0], "lat": snapped[1]},
                    "distance_km": d_km,
                    "snap_distance_m": d_km * 1000.0,
                    "side_index": side["side_index"],
                    "fraction": t,
                }
        return best or {}

    def to_dict(self) -> dict:
        return {
            "center_lon": self.center_lon,
            "center_lat": self.center_lat,
            "length_km": self.length_km,
            "width_km": self.width_km,
            "orientation_deg": self.orientation_deg,
            "offshore_azimuth_deg": self.offshore_azimuth_deg,
            "polygon_lonlat": self.polygon_lonlat(),
            "envelope_bbox": self.envelope_bbox(),
            "offshore_edge_midpoint_lonlat": self.offshore_edge_midpoint_lonlat(),
            "crosses_antimeridian": self.crosses_antimeridian(),
            "map_visibility_warnings": self.map_visibility_warnings(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RegionBox":
        return cls(
            center_lon=float(data["center_lon"]),
            center_lat=float(data["center_lat"]),
            length_km=float(data["length_km"]),
            width_km=float(data["width_km"]),
            orientation_deg=float(data["orientation_deg"]),
            offshore_azimuth_deg=float(data.get("offshore_azimuth_deg", data.get("orientation_deg", 90.0))),
        )


def bbox_to_points(bbox: Iterable[float]) -> list[list[float]]:
    w, s, e, n = map(float, bbox)
    return [[w, s], [e, s], [e, n], [w, n]]
