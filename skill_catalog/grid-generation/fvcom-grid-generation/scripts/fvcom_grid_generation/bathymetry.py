"""Bathymetry loading and sampling for FVCOM mesh generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class BathymetryGrid:
    """Structured lon-lat bathymetry grid with positive-down depth."""

    lon: np.ndarray
    lat: np.ndarray
    depth: np.ndarray
    source_path: str | None = None
    metadata: dict[str, Any] | None = None

    def sample(self, lon: np.ndarray, lat: np.ndarray, fill_value: float | None = None) -> np.ndarray:
        """Sample positive-down depth at lon/lat points."""
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        finite = np.isfinite(self.depth)
        fallback = float(np.nanmedian(self.depth[finite])) if np.any(finite) else 2.0
        fill = fallback if fill_value is None else float(fill_value)
        interp = RegularGridInterpolator(
            (self.lat, self.lon),
            self.depth,
            bounds_error=False,
            fill_value=fill,
        )
        return np.asarray(interp(np.column_stack([lat, lon])), dtype=float)


def coarsen_for_size_field(bathy: BathymetryGrid, max_cells: int = 1_500_000) -> BathymetryGrid:
    """Return a bounded-cell bathymetry grid for size-field operations."""
    max_cells = max(int(max_cells), 1)
    cell_count = int(bathy.depth.size)
    if cell_count <= max_cells:
        return bathy
    stride = int(np.ceil(np.sqrt(cell_count / max_cells)))
    lat_idx = _stride_indices(bathy.lat.size, stride)
    lon_idx = _stride_indices(bathy.lon.size, stride)
    metadata = dict(bathy.metadata or {})
    metadata["coarsened_for_size_field"] = {
        "source_cell_count": cell_count,
        "max_cells": max_cells,
        "stride": stride,
        "size_field_cell_count": int(len(lat_idx) * len(lon_idx)),
    }
    return BathymetryGrid(
        lon=bathy.lon[lon_idx],
        lat=bathy.lat[lat_idx],
        depth=bathy.depth[np.ix_(lat_idx, lon_idx)],
        source_path=bathy.source_path,
        metadata=metadata,
    )


def load_bathymetry(path: str | Path) -> BathymetryGrid:
    """Load a CUDEM-style NetCDF or a simple NPZ bathymetry product."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            lon = np.asarray(data["lon"], dtype=float)
            lat = np.asarray(data["lat"], dtype=float)
            depth = np.asarray(data["depth"], dtype=float)
        return BathymetryGrid(lon=lon, lat=lat, depth=_normalize_depth(depth), source_path=str(path), metadata={"format": "npz"})

    ds = xr.open_dataset(path)
    lon_name = _find_coord(ds, ("lon", "longitude", "x"))
    lat_name = _find_coord(ds, ("lat", "latitude", "y"))
    depth_name = _find_var(
        ds,
        (
            "depth_m",
            "depth",
            "fvcom_depth",
            "positive_down_depth",
            "bathymetry",
            "elevation_m",
            "elevation",
            "z",
            "Band1",
        ),
    )
    if lon_name is None or lat_name is None or depth_name is None:
        raise ValueError(f"Could not identify lon/lat/depth variables in {path}")
    lon = np.asarray(ds[lon_name].values, dtype=float)
    lat = np.asarray(ds[lat_name].values, dtype=float)
    var = ds[depth_name]
    if lat_name in var.dims and lon_name in var.dims:
        values = np.asarray(var.transpose(lat_name, lon_name).values, dtype=float)
        used_named_dims = True
    else:
        values = np.asarray(var.values, dtype=float)
        used_named_dims = False
    if values.ndim > 2:
        values = np.squeeze(values)
    if not used_named_dims and values.shape == (lon.size, lat.size):
        values = values.T
    if values.shape != (lat.size, lon.size):
        raise ValueError(f"Bathymetry variable {depth_name} shape {values.shape} does not match lat/lon grid")
    if _is_positive_up(ds[depth_name], depth_name):
        depth = -values
    else:
        depth = values
    metadata = {
        "format": "netcdf",
        "variables": list(ds.data_vars),
        "lon_name": lon_name,
        "lat_name": lat_name,
        "depth_name": depth_name,
        "source_id_present": "source_id" in ds.data_vars,
        "source_resolution_present": "source_resolution_arcsec" in ds.data_vars,
        "global_attrs": {str(key): str(value) for key, value in ds.attrs.items()},
    }
    return BathymetryGrid(lon=lon, lat=lat, depth=_normalize_depth(depth), source_path=str(path), metadata=metadata)


def write_synthetic_bathymetry(path: str | Path, bbox_wsen: tuple[float, float, float, float], nx: int = 80, ny: int = 80) -> Path:
    """Write a small smooth positive-down bathymetry NetCDF for tests/smokes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = bbox_wsen
    lon = np.linspace(west, east, nx)
    lat = np.linspace(south, north, ny)
    x = np.linspace(0.0, 1.0, nx)[None, :]
    y = np.linspace(0.0, 1.0, ny)[:, None]
    depth = 4.0 + 25.0 * x + 8.0 * np.sin(np.pi * y) ** 2
    ds = xr.Dataset(
        {"depth": (("lat", "lon"), depth.astype(float))},
        coords={"lon": lon.astype(float), "lat": lat.astype(float)},
        attrs={"summary": "Synthetic positive-down bathymetry for fvcom-grid-generation tests."},
    )
    ds.to_netcdf(path)
    return path


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    out = np.asarray(depth, dtype=float)
    out = np.where(np.isfinite(out), out, np.nan)
    out = np.maximum(out, 0.5)
    return out


def _stride_indices(size: int, stride: int) -> np.ndarray:
    indices = list(range(0, int(size), max(int(stride), 1)))
    if indices[-1] != size - 1:
        indices.append(size - 1)
    return np.asarray(indices, dtype=int)


def _find_coord(ds: xr.Dataset, names: tuple[str, ...]) -> str | None:
    lower = {name.lower(): name for name in list(ds.coords) + list(ds.variables)}
    for name in names:
        if name in lower:
            return lower[name]
    return None


def _find_var(ds: xr.Dataset, names: tuple[str, ...]) -> str | None:
    lower = {name.lower(): name for name in ds.data_vars}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    for name in ds.data_vars:
        values = np.asarray(ds[name].values)
        if values.ndim >= 2 and np.issubdtype(values.dtype, np.number):
            return name
    return None


def _is_positive_up(var: xr.DataArray, name: str) -> bool:
    positive = str(var.attrs.get("positive", "")).strip().lower()
    if positive == "up":
        return True
    if positive == "down":
        return False
    lower = name.lower()
    return lower in {"elevation", "elevation_m", "z", "band1"}
