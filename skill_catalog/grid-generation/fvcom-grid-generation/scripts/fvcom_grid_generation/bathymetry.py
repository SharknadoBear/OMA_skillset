"""Local bathymetry loading and interpolation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class BathymetryGrid:
    """Structured bathymetry in lon/lat with FVCOM positive-down depth."""

    lon: np.ndarray
    lat: np.ndarray
    depth: np.ndarray
    source: str
    lon_name: str = "lon"
    lat_name: str = "lat"
    depth_name: str = "depth"

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            float(np.nanmin(self.lon)),
            float(np.nanmin(self.lat)),
            float(np.nanmax(self.lon)),
            float(np.nanmax(self.lat)),
        )

    def interpolator(self, fill_value: float | None = None) -> RegularGridInterpolator:
        """Return a lat,lon interpolator for positive-down depth."""
        if fill_value is None:
            finite = self.depth[np.isfinite(self.depth)]
            fill_value = float(np.nanmedian(finite)) if finite.size else 1.0
        return RegularGridInterpolator(
            (self.lat, self.lon),
            self.depth,
            bounds_error=False,
            fill_value=fill_value,
        )

    def sample(self, lon: np.ndarray, lat: np.ndarray, fill_value: float | None = None) -> np.ndarray:
        pts = np.column_stack([np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)])
        return np.asarray(self.interpolator(fill_value)(pts), dtype=float)


def load_bathymetry(
    path: str | Path,
    lon_name: str | None = None,
    lat_name: str | None = None,
    depth_name: str | None = None,
) -> BathymetryGrid:
    """Load NetCDF or GeoTIFF bathymetry and return positive-down depth."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".nc", ".nc4", ".cdf"}:
        return _load_netcdf(path, lon_name=lon_name, lat_name=lat_name, depth_name=depth_name)
    if suffix in {".tif", ".tiff"}:
        return _load_geotiff(path)
    raise ValueError(f"Unsupported bathymetry format: {path.suffix}")


def _load_netcdf(path: Path, lon_name: str | None, lat_name: str | None, depth_name: str | None) -> BathymetryGrid:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        lon_name = lon_name or _find_name(ds.coords, ("lon", "longitude", "x"))
        lat_name = lat_name or _find_name(ds.coords, ("lat", "latitude", "y"))
        if lon_name is None or lat_name is None:
            raise ValueError("Could not identify longitude/latitude coordinates.")

        depth_name = depth_name or _find_depth_name(ds, lat_name, lon_name)
        if depth_name is None:
            raise ValueError("Could not identify a 2D bathymetry/depth variable.")

        field = ds[depth_name]
        if lat_name in field.dims and lon_name in field.dims:
            field = field.transpose(lat_name, lon_name)
        else:
            raise ValueError(f"{depth_name!r} must have latitude and longitude dimensions.")

        lon = np.asarray(ds[lon_name].values, dtype=float).ravel()
        lat = np.asarray(ds[lat_name].values, dtype=float).ravel()
        values = np.asarray(field.values, dtype=float)

    lon = np.where(lon > 180.0, lon - 360.0, lon)
    lon_order = np.argsort(lon)
    lat_order = np.argsort(lat)
    lon = lon[lon_order]
    lat = lat[lat_order]
    values = values[np.ix_(lat_order, lon_order)]
    depth = _to_positive_down(values)

    return BathymetryGrid(
        lon=lon,
        lat=lat,
        depth=depth,
        source=str(path),
        lon_name=lon_name,
        lat_name=lat_name,
        depth_name=depth_name,
    )


def _load_geotiff(path: Path) -> BathymetryGrid:
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        rows, cols = np.indices(arr.shape)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
        lon2 = np.asarray(xs, dtype=float)
        lat2 = np.asarray(ys, dtype=float)

    lon = lon2[0, :]
    lat = lat2[:, 0]
    lon_order = np.argsort(lon)
    lat_order = np.argsort(lat)
    return BathymetryGrid(
        lon=lon[lon_order],
        lat=lat[lat_order],
        depth=_to_positive_down(arr[np.ix_(lat_order, lon_order)]),
        source=str(path),
        lon_name="lon",
        lat_name="lat",
        depth_name="band1",
    )


def _find_name(names: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _find_depth_name(ds, lat_name: str, lon_name: str) -> str | None:
    preferred = (
        "depth",
        "h",
        "z",
        "elevation",
        "topo",
        "topography",
        "bathymetry",
        "Band1",
    )
    for name in preferred:
        if name in ds.data_vars:
            return name
    for name, var in ds.data_vars.items():
        if lat_name in var.dims and lon_name in var.dims and var.ndim == 2:
            return name
    return None


def _to_positive_down(values: np.ndarray) -> np.ndarray:
    """Convert elevation-style data to FVCOM positive-down depth."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    # Elevation/topography sources are commonly negative below sea level.
    if np.nanmedian(finite) < 0.0 or np.nanpercentile(finite, 10) < 0.0:
        depth = -values
    else:
        depth = values.copy()
    depth[depth < 0.0] = np.nan
    return depth
