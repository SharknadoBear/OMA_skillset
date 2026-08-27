"""Small projection helpers used by the boundary-arc workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon
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


def projection_from_crs(
    crs_like,
    *,
    longitude_origin: float | None = None,
) -> LocalProjection:
    """Build direct native-lon/lat transformers for an explicit projected CRS."""
    crs = CRS.from_user_input(crs_like)
    if not crs.is_projected:
        raise ValueError(f"Expected a projected CRS, received {crs.to_string()!r}")
    return LocalProjection(
        crs=crs,
        to_xy=Transformer.from_crs("EPSG:4326", crs, always_xy=True),
        to_lonlat=Transformer.from_crs(crs, "EPSG:4326", always_xy=True),
        longitude_origin=(float(longitude_origin) if longitude_origin is not None else None),
    )


def projection_from_manifest(
    manifest: dict,
    fallback_bbox_wsen: tuple[float, float, float, float],
) -> LocalProjection:
    """Prefer a producer-recorded CRS and otherwise select one from the native bbox."""
    record = manifest.get("projection") if isinstance(manifest, dict) else None
    record = record if isinstance(record, dict) else {}
    crs_like = record.get("crs") or (
        f"EPSG:{int(record['epsg'])}" if record.get("epsg") is not None else None
    )
    if crs_like:
        return projection_from_crs(
            crs_like,
            longitude_origin=record.get("longitude_origin"),
        )
    return local_utm_projection(fallback_bbox_wsen)


def project_geometry(geometry, projection: LocalProjection):
    """Project a Shapely geometry from lon/lat to local CRS."""
    return shapely_transform(projection.to_xy.transform, geometry)


def project_geometry_densified(
    geometry,
    projection: LocalProjection,
    *,
    maximum_segment_degrees: float = 0.25,
):
    """Densify sparse native geographic edges, then project without longitude warping."""
    return project_geometry(
        densify_native_geographic_geometry(
            geometry,
            maximum_segment_degrees=maximum_segment_degrees,
        ),
        projection,
    )


def densify_native_geographic_geometry(
    geometry,
    *,
    maximum_segment_degrees: float = 0.25,
):
    """Add native lon/lat vertices along shortest circular-longitude segments."""
    maximum_segment_degrees = float(maximum_segment_degrees)
    if maximum_segment_degrees <= 0.0:
        raise ValueError("maximum_segment_degrees must be positive")
    if geometry is None or geometry.is_empty or isinstance(geometry, Point):
        return geometry
    if isinstance(geometry, LineString):
        return LineString(
            _densify_native_coordinate_sequence(
                list(geometry.coords), maximum_segment_degrees
            )
        )
    if isinstance(geometry, Polygon):
        exterior = _densify_native_coordinate_sequence(
            list(geometry.exterior.coords), maximum_segment_degrees
        )
        interiors = [
            _densify_native_coordinate_sequence(list(ring.coords), maximum_segment_degrees)
            for ring in geometry.interiors
        ]
        return Polygon(exterior, interiors)
    if isinstance(geometry, MultiLineString):
        return MultiLineString(
            [
                densify_native_geographic_geometry(
                    part, maximum_segment_degrees=maximum_segment_degrees
                )
                for part in geometry.geoms
            ]
        )
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon(
            [
                densify_native_geographic_geometry(
                    part, maximum_segment_degrees=maximum_segment_degrees
                )
                for part in geometry.geoms
            ]
        )
    if isinstance(geometry, GeometryCollection):
        return GeometryCollection(
            [
                densify_native_geographic_geometry(
                    part, maximum_segment_degrees=maximum_segment_degrees
                )
                for part in geometry.geoms
            ]
        )
    return geometry


def _densify_native_coordinate_sequence(
    coordinates,
    maximum_segment_degrees: float,
) -> list[tuple[float, float]]:
    if len(coordinates) < 2:
        return [(float(coord[0]), float(coord[1])) for coord in coordinates]
    output: list[tuple[float, float]] = []
    for start, end in zip(coordinates[:-1], coordinates[1:]):
        lon0, lat0 = float(start[0]), float(start[1])
        lon1, lat1 = float(end[0]), float(end[1])
        delta_lon = lon1 - lon0
        if delta_lon > 180.0:
            delta_lon -= 360.0
        elif delta_lon < -180.0:
            delta_lon += 360.0
        delta_lat = lat1 - lat0
        segments = max(
            1,
            int(
                np.ceil(
                    max(abs(delta_lon), abs(delta_lat))
                    / maximum_segment_degrees
                )
            ),
        )
        for index in range(segments):
            fraction = float(index) / float(segments)
            longitude = lon0 + fraction * delta_lon
            longitude = ((longitude + 180.0) % 360.0) - 180.0
            output.append((longitude, lat0 + fraction * delta_lat))
    output.append((float(coordinates[-1][0]), float(coordinates[-1][1])))
    return output


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
