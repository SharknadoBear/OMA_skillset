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
    adaptive_boundary: bool = False
    bathymetry_gradient_policy: str = "auto"
    coastal_gradient_distance_m: float = 25_000.0


@dataclass(frozen=True)
class SizeField:
    lon: np.ndarray
    lat: np.ndarray
    size: np.ndarray
    raw_size: np.ndarray
    depth: np.ndarray
    slope: np.ndarray
    report: dict[str, Any]
    boundary_size: np.ndarray | None = None
    slope_size: np.ndarray | None = None
    bathymetry_gradient_mask: np.ndarray | None = None
    coastal_distance_m: np.ndarray | None = None

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
    adaptive_boundary = bool(config.adaptive_boundary or boundary.adaptive_resolution)
    requested_gradient_policy, effective_gradient_policy = _resolve_bathymetry_gradient_policy(config, adaptive_boundary)
    coastal_distance = coastal_boundary_distance(bathy, boundary)
    coastal_threshold = float(config.coastal_gradient_distance_m)
    if coastal_threshold < 0.0:
        raise ValueError("coastal_gradient_distance_m must be non-negative")
    coastal_mask = coastal_distance <= coastal_threshold

    raw = np.full(depth.shape, config.max_size_m, dtype=float)
    if not adaptive_boundary:
        raw = np.where(depth <= config.shelf_depth_m, np.minimum(raw, config.shelf_max_size_m), raw)
        raw = np.where(depth <= config.nearshore_depth_m, np.minimum(raw, config.nearshore_max_size_m), raw)

    topo_length = depth / np.maximum(slope, config.min_gradient)
    slope_candidate = (2.0 * np.pi / max(config.slope_elements, 1.0)) * topo_length
    depth_eligible = np.isfinite(depth) & (depth > config.nearshore_depth_m)
    slope_size = np.where(depth_eligible, slope_candidate, np.nan)
    if effective_gradient_policy == "global":
        gradient_mask = depth_eligible
    elif effective_gradient_policy == "coastal":
        gradient_mask = depth_eligible & coastal_mask
    else:
        gradient_mask = np.zeros(depth.shape, dtype=bool)

    boundary_size = shoreline_distance_size(bathy, boundary, config)
    raw = np.minimum(raw, boundary_size)
    pre_slope_size = raw.copy()
    effective_slope_size = np.where(gradient_mask, slope_candidate, np.inf)
    slope_limited = gradient_mask & np.isfinite(slope_candidate) & (slope_candidate < pre_slope_size)
    offshore_slope_limited = slope_limited & ~coastal_mask
    raw = np.minimum(raw, effective_slope_size)

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
        "adaptive_boundary": adaptive_boundary,
        "bathymetry_gradient_policy_requested": requested_gradient_policy,
        "bathymetry_gradient_policy_effective": effective_gradient_policy,
        "bathymetry_gradient": {
            "requested_policy": requested_gradient_policy,
            "effective_policy": effective_gradient_policy,
            "coastal_gradient_distance_m": coastal_threshold,
            "eligible_depth_cell_count": int(np.count_nonzero(depth_eligible)),
            "active_cell_count": int(np.count_nonzero(gradient_mask)),
            "active_cell_fraction": float(np.count_nonzero(gradient_mask) / max(np.count_nonzero(np.isfinite(depth)), 1)),
            "coastal_cell_count": int(np.count_nonzero(coastal_mask & np.isfinite(depth))),
            "coastal_cell_fraction": float(np.count_nonzero(coastal_mask & np.isfinite(depth)) / max(np.count_nonzero(np.isfinite(depth)), 1)),
            "slope_limited_count": int(np.count_nonzero(slope_limited)),
            "offshore_slope_limited_count": int(np.count_nonzero(offshore_slope_limited)),
            "land_island_boundary_node_count": int(
                sum(str(kind).lower() not in {"open", "open_boundary"} for kind in boundary.kinds)
            ),
        },
    }
    return SizeField(
        lon=bathy.lon,
        lat=bathy.lat,
        size=limited,
        raw_size=raw,
        depth=depth,
        slope=slope,
        report=report,
        boundary_size=boundary_size,
        slope_size=slope_size,
        bathymetry_gradient_mask=gradient_mask,
        coastal_distance_m=coastal_distance,
    )


def _resolve_bathymetry_gradient_policy(config: SizeFieldConfig, adaptive_boundary: bool) -> tuple[str, str]:
    requested = str(config.bathymetry_gradient_policy).strip().lower()
    allowed = {"auto", "global", "coastal", "off"}
    if requested not in allowed:
        raise ValueError(f"bathymetry_gradient_policy must be one of {sorted(allowed)}")
    effective = "coastal" if requested == "auto" and adaptive_boundary else "global" if requested == "auto" else requested
    return requested, effective


def coastal_boundary_distance(bathy: BathymetryGrid, boundary: BoundaryNodes) -> np.ndarray:
    """Return projected distance to the nearest fixed land or island boundary node."""
    lon2, lat2 = np.meshgrid(bathy.lon, bathy.lat)
    solid_points = np.asarray(
        [point for point, kind in zip(boundary.xy, boundary.kinds) if str(kind).lower() not in {"open", "open_boundary"}],
        dtype=float,
    )
    if solid_points.size == 0:
        return np.full(lon2.shape, np.inf, dtype=float)
    lonlat = np.column_stack([lon2.ravel(), lat2.ravel()])
    xy = project_points(lonlat, boundary.projection)
    distance = cKDTree(solid_points).query(xy, workers=-1)[0]
    return np.asarray(distance, dtype=float).reshape(lon2.shape)


def shoreline_distance_size(bathy: BathymetryGrid, boundary: BoundaryNodes, config: SizeFieldConfig) -> np.ndarray:
    """Create a simple shoreline-distance refinement field."""
    lon2, lat2 = np.meshgrid(bathy.lon, bathy.lat)
    lonlat = np.column_stack([lon2.ravel(), lat2.ravel()])
    xy = project_points(lonlat, boundary.projection)
    if config.adaptive_boundary or boundary.adaptive_resolution:
        if not len(boundary.xy):
            return np.full(lon2.shape, config.max_size_m, dtype=float)
        distance, nearest = cKDTree(boundary.xy).query(xy, workers=-1)
        target = np.asarray(boundary.target_spacing_m, dtype=float)[np.asarray(nearest, dtype=int)]
        return np.clip(
            (target + float(config.gradation) * np.asarray(distance, dtype=float)).reshape(lon2.shape),
            float(np.nanmin(boundary.target_spacing_m)),
            config.max_size_m,
        )
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
    variables: dict[str, Any] = {
        "mesh_size_m": (("lat", "lon"), size_field.size),
        "raw_mesh_size_m": (("lat", "lon"), size_field.raw_size),
        "depth_m": (("lat", "lon"), size_field.depth),
        "slope": (("lat", "lon"), size_field.slope),
    }
    if size_field.boundary_size is not None:
        variables["boundary_mesh_size_m"] = (("lat", "lon"), size_field.boundary_size)
    if size_field.slope_size is not None:
        variables["bathymetry_slope_mesh_size_m"] = (("lat", "lon"), size_field.slope_size)
    if size_field.bathymetry_gradient_mask is not None:
        variables["bathymetry_gradient_mask"] = (("lat", "lon"), np.asarray(size_field.bathymetry_gradient_mask, dtype=np.uint8))
    if size_field.coastal_distance_m is not None:
        variables["coastal_distance_m"] = (("lat", "lon"), size_field.coastal_distance_m)
    ds = xr.Dataset(
        variables,
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
