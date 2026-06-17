"""RPWCW2019-inspired mesh-size fields."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

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
    gradation_iterations: int = 40
    gradation_tolerance_fraction: float = 0.01


@dataclass(frozen=True)
class SizeField:
    lon: np.ndarray
    lat: np.ndarray
    size: np.ndarray
    slope: np.ndarray
    raw_size: np.ndarray | None = None
    gradation_report: dict | None = None

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
    raw_size = size.copy()
    size, report = apply_gradation_limit_with_report(bathy.lon, bathy.lat, size, config)

    return SizeField(lon=bathy.lon, lat=bathy.lat, size=size, slope=grad, raw_size=raw_size, gradation_report=report)


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
    limited, _ = _apply_gradation_limit_impl(lon, lat, size, config)
    return limited


def apply_gradation_limit_with_report(
    lon: np.ndarray,
    lat: np.ndarray,
    size: np.ndarray,
    config: SizeFieldConfig,
) -> tuple[np.ndarray, dict]:
    """Return a gradation-limited size field and reproducibility diagnostics."""
    return _apply_gradation_limit_impl(lon, lat, size, config)


def _apply_gradation_limit_impl(
    lon: np.ndarray,
    lat: np.ndarray,
    size: np.ndarray,
    config: SizeFieldConfig,
    return_info: bool = False,
) -> tuple[np.ndarray, dict]:
    """Limit ``|h_i - h_j| / distance(i, j)`` without coarsening fine cells."""
    raw = np.asarray(size, dtype=float)
    out = raw.copy()
    lat0 = float(np.nanmean(lat))
    dx_m = float(np.nanmedian(np.diff(lon))) * np.pi / 180.0 * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    dy_m = float(np.nanmedian(np.diff(lat))) * np.pi / 180.0 * EARTH_RADIUS_M
    dx_m = abs(dx_m) if np.isfinite(dx_m) and dx_m != 0 else 1.0
    dy_m = abs(dy_m) if np.isfinite(dy_m) and dy_m != 0 else 1.0

    finite = np.isfinite(out)
    if not np.any(finite):
        report = {
            "gradation": float(config.gradation),
            "method": "priority_queue_lower_envelope",
            "iterations": 0,
            "relaxations": 0,
            "max_change_m": 0.0,
            "tolerance_m": float(max(float(config.gradation_tolerance_fraction) * float(config.min_size), 0.0)),
            "dx_m": float(dx_m),
            "dy_m": float(dy_m),
            "max_neighbor_gradation": 0.0,
            "converged": True,
        }
        return out, report

    out = np.clip(out, config.min_size, config.max_size)
    heap: list[tuple[float, int, int]] = []
    rows, cols = out.shape
    for j, i in zip(*np.nonzero(finite)):
        heapq.heappush(heap, (float(out[j, i]), int(j), int(i)))

    relaxations = 0
    edge_costs = ((0, -1, dx_m), (0, 1, dx_m), (-1, 0, dy_m), (1, 0, dy_m))
    while heap:
        value, j, i = heapq.heappop(heap)
        if value > out[j, i] + 1.0e-9:
            continue
        for dj, di, distance in edge_costs:
            jj = j + dj
            ii = i + di
            if jj < 0 or jj >= rows or ii < 0 or ii >= cols or not finite[jj, ii]:
                continue
            candidate = value + config.gradation * distance
            if candidate < out[jj, ii]:
                out[jj, ii] = candidate
                relaxations += 1
                heapq.heappush(heap, (float(candidate), int(jj), int(ii)))

    report = {
        "gradation": float(config.gradation),
        "method": "priority_queue_lower_envelope",
        "iterations": 1,
        "relaxations": int(relaxations),
        "max_change_m": float(np.nanmax(np.abs(raw - out))) if out.size else 0.0,
        "tolerance_m": float(max(float(config.gradation_tolerance_fraction) * float(config.min_size), 0.0)),
        "dx_m": float(dx_m),
        "dy_m": float(dy_m),
        "max_neighbor_gradation": float(max_neighbor_gradation(out, dx_m, dy_m)),
        "converged": True,
    }
    return out, report


def max_neighbor_gradation(size: np.ndarray, dx_m: float, dy_m: float) -> float:
    """Return max neighbor ``|dh| / distance`` over cardinal grid edges."""
    values = []
    if size.shape[1] > 1:
        values.append(np.nanmax(np.abs(np.diff(size, axis=1)) / max(dx_m, 1.0)))
    if size.shape[0] > 1:
        values.append(np.nanmax(np.abs(np.diff(size, axis=0)) / max(dy_m, 1.0)))
    return float(np.nanmax(values)) if values else 0.0
