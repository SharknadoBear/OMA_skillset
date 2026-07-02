"""Local projection helpers for coastal mesh generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pyproj import CRS, Transformer
from shapely.ops import transform


@dataclass(frozen=True)
class LocalProjection:
    """A local projected CRS and reversible transforms."""

    crs: CRS
    to_xy: Transformer
    to_lonlat: Transformer
    epsg: int
    lon0: float
    lat0: float


def local_utm_projection(bbox_wsen: tuple[float, float, float, float]) -> LocalProjection:
    """Choose a UTM CRS from a W/S/E/N lon-lat bbox."""
    west, south, east, north = bbox_wsen
    if east < west:
        east_for_center = east + 360.0
        lon0 = west + 0.5 * (east_for_center - west)
        lon0 = ((lon0 + 180.0) % 360.0) - 180.0
    else:
        lon0 = 0.5 * (west + east)
    lat0 = 0.5 * (south + north)
    zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
    zone = min(max(zone, 1), 60)
    epsg = (32600 if lat0 >= 0 else 32700) + zone
    crs = CRS.from_epsg(epsg)
    return LocalProjection(
        crs=crs,
        to_xy=Transformer.from_crs("EPSG:4326", crs, always_xy=True),
        to_lonlat=Transformer.from_crs(crs, "EPSG:4326", always_xy=True),
        epsg=epsg,
        lon0=float(lon0),
        lat0=float(lat0),
    )


def project_geometry(geometry: Any, projection: LocalProjection):
    """Project a shapely geometry from lon-lat to local meters."""
    return transform(projection.to_xy.transform, geometry)


def unproject_geometry(geometry: Any, projection: LocalProjection):
    """Project a shapely geometry from local meters to lon-lat."""
    return transform(projection.to_lonlat.transform, geometry)


def project_points(lonlat: np.ndarray, projection: LocalProjection) -> np.ndarray:
    """Project N x 2 lon-lat coordinates to local x-y."""
    x, y = projection.to_xy.transform(lonlat[:, 0], lonlat[:, 1])
    return np.column_stack([x, y]).astype(float)


def unproject_points(xy: np.ndarray, projection: LocalProjection) -> np.ndarray:
    """Project N x 2 x-y coordinates to lon-lat."""
    lon, lat = projection.to_lonlat.transform(xy[:, 0], xy[:, 1])
    return np.column_stack([lon, lat]).astype(float)
