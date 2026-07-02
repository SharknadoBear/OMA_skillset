"""OceanMesh/RPW-style mesh-size fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import heapq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from .bathymetry import BathymetryGrid
from .boundary import BoundaryNodes
from .projection import project_points


EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class SizeFieldConfig:
    land_spacing_m: float = 50.0
    open_spacing_m: float = 3000.0
    max_size_m: float = 20_000.0
    nearshore_depth_m: float = 50.0
    shelf_depth_m: float = 250.0
    nearshore_max_size_m: float = 2_000.0
    shelf_max_size_m: float = 8_000.0
    gradation: float = 0.15
    slope_elements: float = 20.0
    min_gradient: float = 1.0e-5
    target_timestep_s: str = "auto"
    cfl: float = 0.5


@dataclass(frozen=True)
class SizeField:
    lon: np.ndarray
    lat: np.ndarray
    size: np.ndarray
    raw_size: np.ndarray
    depth: np.ndarray
    slope: np.ndarray
    report: dict[str, Any]

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        interp = RegularGridInterpolator(
            (self.lat, self.lon),
            self.size,
            bounds_error=False,
            fill_value=float(np.nanmax(self.size)),
        )
        return np.asarray(interp(np.column_stack([lat, lon])), dtype=float)


def build_size_field(bathy: BathymetryGrid, boundary: BoundaryNodes, config: SizeFieldConfig) -> SizeField:
    """Build a size field from bathymetry, boundary distance, and gradation."""
    depth = np.asarray(bathy.depth, dtype=float)
    slope = bathymetric_gradient(bathy)
    raw = np.full(depth.shape, config.max_size_m, dtype=float)
    raw = np.where(depth <= config.shelf_depth_m, np.minimum(raw, config.shelf_max_size_m), raw)
    raw = np.where(depth <= config.nearshore_depth_m, np.minimum(raw, config.nearshore_max_size_m), raw)

    topo_length = depth / np.maximum(slope, config.min_gradient)
    slope_size = (2.0 * np.pi / max(config.slope_elements, 1.0)) * topo_length
    slope_size = np.where(depth > config.nearshore_depth_m, slope_size, np.inf)
    raw = np.minimum(raw, slope_size)

    shore_size = shoreline_distance_size(bathy, boundary, config)
    raw = np.minimum(raw, shore_size)

    cfl_report = cfl_size_report(depth, raw, config)
    if cfl_report["mode"] == "enforced":
        raw = np.minimum(raw, cfl_report["cfl_size_m"])

    raw = np.where(np.isfinite(depth), raw, config.max_size_m)
    raw = np.clip(raw, min(config.land_spacing_m, config.open_spacing_m), config.max_size_m)
    limited, gradation_report = apply_gradation_limit(bathy.lon, bathy.lat, raw, config.gradation)
    report = {
        "schema_version": "fvcom_size_field_v1",
        "method": "rpw_clean_room_pointwise_minimum",
        "gradation": gradation_report,
        "cfl": {key: value for key, value in cfl_report.items() if key != "cfl_size_m"},
        "min_size_m": float(np.nanmin(limited)),
        "max_size_m": float(np.nanmax(limited)),
        "land_spacing_m": float(config.land_spacing_m),
        "open_spacing_m": float(config.open_spacing_m),
    }
    return SizeField(lon=bathy.lon, lat=bathy.lat, size=limited, raw_size=raw, depth=depth, slope=slope, report=report)


def shoreline_distance_size(bathy: BathymetryGrid, boundary: BoundaryNodes, config: SizeFieldConfig) -> np.ndarray:
    """Create a simple shoreline-distance refinement field."""
    lon2, lat2 = np.meshgrid(bathy.lon, bathy.lat)
    lonlat = np.column_stack([lon2.ravel(), lat2.ravel()])
    xy = project_points(lonlat, boundary.projection)
    land_points = np.asarray([pt for pt, kind in zip(boundary.xy, boundary.kinds) if kind != "open"], dtype=float)
    if land_points.size == 0:
        land_points = boundary.xy
    land_dist = cKDTree(land_points).query(xy, workers=-1)[0].reshape(lon2.shape)
    size = config.land_spacing_m + 0.30 * land_dist
    open_points = np.asarray([boundary.xy[idx] for idx in boundary.open_boundary_indices if idx < len(boundary.xy)], dtype=float)
    if open_points.size:
        open_dist = cKDTree(open_points).query(xy, workers=-1)[0].reshape(lon2.shape)
        open_size = config.open_spacing_m + 0.10 * open_dist
        size = np.minimum(size, open_size)
    return np.clip(size, min(config.land_spacing_m, config.open_spacing_m), config.max_size_m)


def bathymetric_gradient(bathy: BathymetryGrid) -> np.ndarray:
    """Estimate depth gradient magnitude in m/m."""
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


def cfl_size_report(depth: np.ndarray, raw_size: np.ndarray, config: SizeFieldConfig) -> dict[str, Any]:
    """Return CFL limiter diagnostics and optional size cap."""
    grav = 9.807
    safe_depth = np.maximum(np.asarray(depth, dtype=float), 1.0)
    wave_speed = np.sqrt(grav * safe_depth) + np.sqrt(grav / safe_depth)
    stable_dt = config.cfl * raw_size / np.maximum(wave_speed, 1.0e-6)
    report: dict[str, Any] = {
        "target_timestep_s": config.target_timestep_s,
        "recommended_timestep_s": float(np.nanmin(stable_dt)),
        "cfl": float(config.cfl),
        "mode": "reported",
    }
    if str(config.target_timestep_s).lower() != "auto":
        target = float(config.target_timestep_s)
        report["mode"] = "enforced"
        report["target_timestep_s"] = target
        report["cfl_size_m"] = target * wave_speed / max(config.cfl, 1.0e-6)
    return report


def apply_gradation_limit(lon: np.ndarray, lat: np.ndarray, size: np.ndarray, gradation: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Priority-queue lower-envelope gradation limiter."""
    out = np.asarray(size, dtype=float).copy()
    raw = out.copy()
    finite = np.isfinite(out)
    lat0 = float(np.nanmean(lat))
    dx_m = abs(float(np.nanmedian(np.diff(lon))) * np.pi / 180.0 * EARTH_RADIUS_M * np.cos(np.radians(lat0))) or 1.0
    dy_m = abs(float(np.nanmedian(np.diff(lat))) * np.pi / 180.0 * EARTH_RADIUS_M) or 1.0
    heap: list[tuple[float, int, int]] = []
    rows, cols = out.shape
    for j, i in zip(*np.nonzero(finite)):
        heapq.heappush(heap, (float(out[j, i]), int(j), int(i)))
    relaxations = 0
    for value, j, i in iter(lambda: heapq.heappop(heap) if heap else None, None):
        if value > out[j, i] + 1.0e-9:
            continue
        for dj, di, dist in ((0, -1, dx_m), (0, 1, dx_m), (-1, 0, dy_m), (1, 0, dy_m)):
            jj = j + dj
            ii = i + di
            if jj < 0 or jj >= rows or ii < 0 or ii >= cols or not finite[jj, ii]:
                continue
            candidate = value + gradation * dist
            if candidate < out[jj, ii]:
                out[jj, ii] = candidate
                relaxations += 1
                heapq.heappush(heap, (float(candidate), int(jj), int(ii)))
    report = {
        "method": "priority_queue_lower_envelope",
        "gradation": float(gradation),
        "relaxations": int(relaxations),
        "max_reduction_m": float(np.nanmax(raw - out)) if out.size else 0.0,
        "dx_m": float(dx_m),
        "dy_m": float(dy_m),
        "max_neighbor_gradation": float(max_neighbor_gradation(out, dx_m, dy_m)),
        "converged": True,
    }
    return out, report


def max_neighbor_gradation(size: np.ndarray, dx_m: float, dy_m: float) -> float:
    values = []
    if size.shape[1] > 1:
        values.append(np.nanmax(np.abs(np.diff(size, axis=1)) / max(dx_m, 1.0)))
    if size.shape[0] > 1:
        values.append(np.nanmax(np.abs(np.diff(size, axis=0)) / max(dy_m, 1.0)))
    return float(np.nanmax(values)) if values else 0.0


def write_size_field(size_field: SizeField, nc_path: str | Path, png_path: str | Path) -> tuple[Path, Path]:
    """Write size field NetCDF and diagnostic PNG."""
    nc_path = Path(nc_path)
    png_path = Path(png_path)
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            "mesh_size_m": (("lat", "lon"), size_field.size),
            "raw_mesh_size_m": (("lat", "lon"), size_field.raw_size),
            "depth_m": (("lat", "lon"), size_field.depth),
            "slope": (("lat", "lon"), size_field.slope),
        },
        coords={"lon": size_field.lon, "lat": size_field.lat},
        attrs={"schema_version": "fvcom_size_field_v1"},
    )
    ds.to_netcdf(nc_path)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.pcolormesh(size_field.lon, size_field.lat, size_field.size, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="mesh size (m)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("FVCOM target mesh size")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return nc_path, png_path
