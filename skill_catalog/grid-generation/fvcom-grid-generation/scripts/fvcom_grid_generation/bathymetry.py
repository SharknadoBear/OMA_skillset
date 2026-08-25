"""Bathymetry loading and sampling for FVCOM mesh generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString, MultiLineString, Polygon


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


def bathymetry_coverage_report(
    bathymetry: BathymetryGrid,
    domain_lonlat: Polygon,
    open_boundary_lonlat: LineString | MultiLineString | None = None,
) -> dict[str, Any]:
    """Audit domain coverage and require complete finite support on every OBC."""
    lon = np.asarray(bathymetry.lon, dtype=float)
    lat = np.asarray(bathymetry.lat, dtype=float)
    depth = np.asarray(bathymetry.depth, dtype=float)
    lon_monotonic = bool(np.all(np.diff(lon) > 0.0) or np.all(np.diff(lon) < 0.0))
    lat_monotonic = bool(np.all(np.diff(lat) > 0.0) or np.all(np.diff(lat) < 0.0))
    lon_step = float(np.quantile(np.abs(np.diff(lon)), 0.95)) if len(lon) > 1 else 0.0
    lat_step = float(np.quantile(np.abs(np.diff(lat)), 0.95)) if len(lat) > 1 else 0.0
    west, south, east, north = domain_lonlat.bounds
    bbox_covers = bool(
        float(np.min(lon)) <= west + lon_step
        and float(np.max(lon)) >= east - lon_step
        and float(np.min(lat)) <= south + lat_step
        and float(np.max(lat)) >= north - lat_step
    )
    stride = max(1, int(np.ceil(np.sqrt(depth.size / 500_000))))
    lon_sample = lon[::stride]
    lat_sample = lat[::stride]
    depth_sample = depth[::stride, ::stride]
    llon, llat = np.meshgrid(lon_sample, lat_sample)
    wet = np.asarray(shapely.contains_xy(domain_lonlat, llon, llat), dtype=bool)
    wet_count = int(np.count_nonzero(wet))
    finite_fraction = (
        float(np.count_nonzero(np.isfinite(depth_sample[wet])) / wet_count)
        if wet_count
        else 0.0
    )

    obc_points = _densify_open_boundary_for_bathymetry(
        open_boundary_lonlat,
        lon,
        lat,
    )
    obc_required = bool(len(obc_points))
    if obc_required:
        sampled_points = _snap_roundoff_to_raster_bounds(obc_points, lon, lat)
        sampled_obc = bathymetry.sample(
            sampled_points[:, 0],
            sampled_points[:, 1],
            fill_value=np.nan,
        )
        finite_obc = np.isfinite(sampled_obc)
        unsupported_obc_count = int(np.count_nonzero(~finite_obc))
        finite_obc_fraction = float(np.count_nonzero(finite_obc) / len(finite_obc))
        obc_support_passed = unsupported_obc_count == 0
    else:
        unsupported_obc_count = 0
        finite_obc_fraction = 1.0
        obc_support_passed = True

    failures: list[str] = []
    if not lon_monotonic or not lat_monotonic:
        failures.append("bathymetry_coordinate_monotonicity_failed")
    if not bbox_covers:
        failures.append("bathymetry_domain_bbox_incomplete")
    if finite_fraction < 0.95:
        failures.append("bathymetry_domain_finite_coverage_below_95pct")
    if not obc_support_passed:
        failures.append("open_boundary_bathymetry_support_incomplete")

    return {
        "lon_monotonic": lon_monotonic,
        "lat_monotonic": lat_monotonic,
        "raster_bbox_wsen": [
            float(np.min(lon)),
            float(np.min(lat)),
            float(np.max(lon)),
            float(np.max(lat)),
        ],
        "domain_bbox_wsen": [float(west), float(south), float(east), float(north)],
        "bbox_covers_domain_with_one_cell_tolerance": bbox_covers,
        "wet_sample_count": wet_count,
        "finite_wet_fraction": finite_fraction,
        "open_boundary_required": obc_required,
        "open_boundary_sample_count": int(len(obc_points)),
        "finite_open_boundary_fraction": finite_obc_fraction,
        "unsupported_open_boundary_sample_count": unsupported_obc_count,
        "open_boundary_bathymetry_support_passed": obc_support_passed,
        "open_boundary_sampling_policy": "every_segment_at_no_more_than_half_minimum_raster_axis_step",
        "open_boundary_outside_support_fill": "nan",
        "open_boundary_raster_edge_tolerance": "1e-8_of_minimum_axis_step",
        "failure_taxonomy": failures,
        "passed": not failures,
    }


def _densify_open_boundary_for_bathymetry(
    geometry: LineString | MultiLineString | None,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Sample every OBC chord at no more than half a raster-axis cell."""
    parts = _line_parts(geometry)
    if not parts:
        return np.empty((0, 2), dtype=float)
    lon_step = _minimum_positive_axis_step(lon)
    lat_step = _minimum_positive_axis_step(lat)
    sampled: list[tuple[float, float]] = []
    for line in parts:
        coords = list(line.coords)
        for start, end in zip(coords[:-1], coords[1:]):
            ratios = []
            if np.isfinite(lon_step):
                ratios.append(abs(float(end[0]) - float(start[0])) / lon_step)
            if np.isfinite(lat_step):
                ratios.append(abs(float(end[1]) - float(start[1])) / lat_step)
            intervals = max(1, int(np.ceil(2.0 * max(ratios, default=1.0))))
            if not sampled or sampled[-1] != (float(start[0]), float(start[1])):
                sampled.append((float(start[0]), float(start[1])))
            for index in range(1, intervals + 1):
                fraction = index / intervals
                sampled.append(
                    (
                        float(start[0]) + fraction * (float(end[0]) - float(start[0])),
                        float(start[1]) + fraction * (float(end[1]) - float(start[1])),
                    )
                )
    return np.asarray(sampled, dtype=float).reshape((-1, 2))


def _minimum_positive_axis_step(values: np.ndarray) -> float:
    delta = np.abs(np.diff(np.asarray(values, dtype=float)))
    positive = delta[np.isfinite(delta) & (delta > 0.0)]
    return float(np.min(positive)) if len(positive) else float("inf")


def _snap_roundoff_to_raster_bounds(
    points: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Snap projection roundoff at a raster edge without masking real overrun."""
    output = np.asarray(points, dtype=float).copy()
    for column, axis in ((0, lon), (1, lat)):
        lower = float(np.min(axis))
        upper = float(np.max(axis))
        step = _minimum_positive_axis_step(axis)
        tolerance = max(
            (step * 1.0e-8 if np.isfinite(step) else 0.0),
            64.0 * np.finfo(float).eps * max(abs(lower), abs(upper), 1.0),
        )
        near_lower = np.abs(output[:, column] - lower) <= tolerance
        near_upper = np.abs(output[:, column] - upper) <= tolerance
        output[near_lower, column] = lower
        output[near_upper, column] = upper
    return output


def _line_parts(geometry: LineString | MultiLineString | None) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    parts: list[LineString] = []
    for item in getattr(geometry, "geoms", []):
        if isinstance(item, LineString) and not item.is_empty:
            parts.append(item)
    return parts


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
