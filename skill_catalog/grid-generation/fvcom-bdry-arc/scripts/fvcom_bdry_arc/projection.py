"""Small projection helpers used by the boundary-arc workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer
from shapely.ops import transform as shapely_transform


@dataclass(frozen=True)
class LocalProjection:
    """Lon/lat to local projected CRS transformers."""

    crs: CRS
    to_xy: Transformer
    to_lonlat: Transformer
    longitude_origin: float | None = None

    @property
    def epsg(self) -> int | None:
        return self.crs.to_epsg()


def local_utm_projection(bbox_wsen: tuple[float, float, float, float]) -> LocalProjection:
    """Choose a local UTM CRS for a lon/lat bbox."""
    west, south, east, north = bbox_wsen
    east_for_center = east + 360.0 if east < west else east
    lon0_unwrapped = 0.5 * (west + east_for_center)
    lon0 = ((lon0_unwrapped + 180.0) % 360.0) - 180.0
    lat0 = 0.5 * (south + north)
    zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
    zone = min(max(zone, 1), 60)
    epsg = 32600 + zone if lat0 >= 0.0 else 32700 + zone
    crs = CRS.from_epsg(epsg)
    return LocalProjection(
        crs=crs,
        to_xy=Transformer.from_crs("EPSG:4326", crs, always_xy=True),
        to_lonlat=Transformer.from_crs(crs, "EPSG:4326", always_xy=True),
        longitude_origin=lon0,
    )


def project_geometry(geometry, projection: LocalProjection):
    """Project a Shapely geometry from lon/lat to local CRS."""
    return shapely_transform(projection.to_xy.transform, geometry)


def unproject_geometry(geometry, projection: LocalProjection):
    """Project a Shapely geometry from local CRS to lon/lat."""
    return shapely_transform(projection.to_lonlat.transform, geometry)


def project_xy(points_lonlat: np.ndarray, projection: LocalProjection) -> np.ndarray:
    """Project lon/lat points to local meters."""
    points_lonlat = np.asarray(points_lonlat, dtype=float)
    x, y = projection.to_xy.transform(points_lonlat[:, 0], points_lonlat[:, 1])
    return np.column_stack([x, y])


def unproject_xy(points_xy: np.ndarray, projection: LocalProjection) -> np.ndarray:
    """Project local-meter points to lon/lat."""
    points_xy = np.asarray(points_xy, dtype=float)
    lon, lat = projection.to_lonlat.transform(points_xy[:, 0], points_xy[:, 1])
    return np.column_stack([lon, lat])


def unwrap_longitude(lon: float, origin: float) -> float:
    x = float(lon)
    while x - origin > 180.0:
        x -= 360.0
    while origin - x > 180.0:
        x += 360.0
    return x


def unwrap_geometry_longitudes(geometry, origin: float):
    """Unwrap lon coordinates around origin so antimeridian edges stay local."""

    def _transform(x, y, z=None):
        arr = np.asarray(x, dtype=float)
        out = arr.copy()
        out = np.where(out - origin > 180.0, out - 360.0, out)
        out = np.where(origin - out > 180.0, out + 360.0, out)
        if z is None:
            return out, y
        return out, y, z

    return shapely_transform(_transform, geometry)
