"""RPWCW2019-inspired mesh-size fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .bathymetry import BathymetryGrid


EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class SizeFieldConfig:
    """Mesh-size controls in meters."""

    min_size: float = 1_000.0
    max_size: float = 20_000.0
    nearshore_depth: float = 50.0
    shelf_depth: float = 250.0
    nearshore_max_size: float = 2_000.0
    shelf_max_size: float = 8_000.0
    gradation: float = 0.15
    slope_elements: float = 20.0
    min_gradient: float = 1.0e-5
    gradation_iterations: int = 12


@dataclass(frozen=True)
class SizeField:
    lon: np.ndarray
    lat: np.ndarray
    size: np.ndarray
    slope: np.ndarray

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        interp = RegularGridInterpolator(
            (self.lat, self.lon),
            self.size,
            bounds_error=False,
            fill_value=float(np.nanmax(self.size)),
        )
        return np.asarray(interp(np.column_stack([lat, lon])), dtype=float)


def build_size_field(bathy: BathymetryGrid, config: SizeFieldConfig | None = None) -> SizeField:
    """Build a gridded target element-size field from depth and slope."""
    config = config or SizeFieldConfig()
    depth = np.asarray(bathy.depth, dtype=float)
    grad = bathymetric_gradient(bathy)

    size = np.full(depth.shape, config.max_size, dtype=float)
    size = np.where(depth <= config.shelf_depth, np.minimum(size, config.shelf_max_size), size)
    size = np.where(depth <= config.nearshore_depth, np.minimum(size, config.nearshore_max_size), size)

    topo_length = depth / np.maximum(grad, config.min_gradient)
    slope_size = (2.0 * np.pi / config.slope_elements) * topo_length
    slope_size = np.where(depth > config.nearshore_depth, slope_size, np.inf)
    size = np.minimum(size, slope_size)
    size = np.clip(size, config.min_size, config.max_size)
    size = np.where(np.isfinite(depth), size, config.max_size)
    size = apply_gradation_limit(bathy.lon, bathy.lat, size, config)

    return SizeField(lon=bathy.lon, lat=bathy.lat, size=size, slope=grad)


def bathymetric_gradient(bathy: BathymetryGrid) -> np.ndarray:
    """Estimate bathymetric slope magnitude as meters per meter."""
    lon = bathy.lon
    lat = bathy.lat
    depth = np.asarray(bathy.depth, dtype=float)
    lat0 = float(np.nanmean(lat))
    dx = np.gradient(lon) * np.pi / 180.0 * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    dy = np.gradient(lat) * np.pi / 180.0 * EARTH_RADIUS_M
    dz_dy, dz_dx = np.gradient(depth)
    dz_dx = dz_dx / np.maximum(dx[None, :], 1.0)
    dz_dy = dz_dy / np.maximum(dy[:, None], 1.0)
    return np.hypot(dz_dx, dz_dy)


def apply_gradation_limit(
    lon: np.ndarray,
    lat: np.ndarray,
    size: np.ndarray,
    config: SizeFieldConfig,
) -> np.ndarray:
    """Limit neighbor-to-neighbor size expansion using a Persson-style gradient bound."""
    out = np.asarray(size, dtype=float).copy()
    lat0 = float(np.nanmean(lat))
    dx_m = float(np.nanmedian(np.diff(lon))) * np.pi / 180.0 * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    dy_m = float(np.nanmedian(np.diff(lat))) * np.pi / 180.0 * EARTH_RADIUS_M
    dx_m = abs(dx_m) if np.isfinite(dx_m) and dx_m != 0 else 1.0
    dy_m = abs(dy_m) if np.isfinite(dy_m) and dy_m != 0 else 1.0

    for _ in range(max(config.gradation_iterations, 0)):
        prev = out.copy()
        out[:, 1:] = np.minimum(out[:, 1:], prev[:, :-1] + config.gradation * dx_m)
        out[:, :-1] = np.minimum(out[:, :-1], prev[:, 1:] + config.gradation * dx_m)
        out[1:, :] = np.minimum(out[1:, :], prev[:-1, :] + config.gradation * dy_m)
        out[:-1, :] = np.minimum(out[:-1, :], prev[1:, :] + config.gradation * dy_m)
    return np.clip(out, config.min_size, config.max_size)
