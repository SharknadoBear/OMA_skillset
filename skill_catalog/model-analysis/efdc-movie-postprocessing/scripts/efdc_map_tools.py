"""Shared native-curvilinear plotting helpers for EFDC maps and movies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize


EARTH_RADIUS_METERS = 6_371_008.8


@dataclass(frozen=True)
class WetCellFootprints:
    """Independent EFDC cell polygons inferred only from wet-neighbor spacing."""

    polygons: np.ndarray
    wet_flat_indices: np.ndarray
    fallback_cells: np.ndarray
    grid_shape: tuple[int, int]
    maximum_span_km: float
    spacing_method: str = "efdc_immediate_wet_neighbor_centers"


@dataclass(frozen=True)
class EFDCPlotResult:
    artist: Any
    colorbar: Any
    quiver: Any | None
    finite_wet_fraction: float
    wet_cell_count: int
    finite_wet_count: int
    rendering_method: str
    rendered_cell_count: int
    footprint_fallback_cell_count: int | None
    footprint_maximum_span_km: float | None


def quantile_limits(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("Quantiles must satisfy 0 <= low < high <= 100.")
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("Cannot determine color limits from all-non-finite values.")
    vmin, vmax = (float(value) for value in np.percentile(finite, [low, high]))
    if vmin == vmax:
        delta = max(abs(vmin) * 0.01, 1.0e-12)
        vmin, vmax = vmin - delta, vmax + delta
    return vmin, vmax


def colormap_for_variable(variable: str) -> str:
    name = str(variable).lower()
    if "sal" in name or "speed" in name:
        return "viridis"
    if any(token in name for token in ("zeta", "elevation", "velocity")) or name in {"u", "v"}:
        return "coolwarm"
    if "temp" in name:
        return "plasma"
    return "viridis"


def geographic_aspect(latitude: np.ndarray) -> float:
    finite = np.asarray(latitude, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 1.0
    cosine = abs(float(np.cos(np.deg2rad(np.mean(finite)))))
    return 1.0 / max(cosine, 1.0e-3)


def _validated(lon, lat, mask, values):
    longitude = np.asarray(lon, dtype=float)
    latitude = np.asarray(lat, dtype=float)
    wet_mask = np.asarray(mask, dtype=float)
    scalar = np.asarray(values, dtype=float)
    if longitude.ndim != 2 or len({longitude.shape, latitude.shape, wet_mask.shape, scalar.shape}) != 1:
        raise ValueError("lon, lat, mask, and values must share one two-dimensional curvilinear-grid shape.")
    coordinate_wet = (wet_mask == 1) & np.isfinite(longitude) & np.isfinite(latitude)
    if not np.any(coordinate_wet):
        raise ValueError("EFDC grid contains no finite wet coordinates.")
    finite_wet = coordinate_wet & np.isfinite(scalar)
    if not np.any(finite_wet):
        raise ValueError("EFDC scalar has zero finite wet coverage and cannot be rendered.")
    return longitude, latitude, coordinate_wet, finite_wet, scalar, np.ma.masked_where(~finite_wet, scalar)


def _local_xy(longitude: np.ndarray, latitude: np.ndarray, wet: np.ndarray):
    """Project one regional grid to a local metric plane without spanning the dateline."""

    longitude_radians = np.deg2rad(longitude[wet])
    reference_lon = float(np.rad2deg(np.arctan2(
        np.mean(np.sin(longitude_radians)), np.mean(np.cos(longitude_radians)))))
    reference_lat = float(np.median(latitude[wet]))
    longitude_delta = (longitude - reference_lon + 180.0) % 360.0 - 180.0
    cosine = max(abs(float(np.cos(np.deg2rad(reference_lat)))), 1.0e-6)
    x = EARTH_RADIUS_METERS * cosine * np.deg2rad(longitude_delta)
    y = EARTH_RADIUS_METERS * np.deg2rad(latitude - reference_lat)
    return x, y, reference_lon, reference_lat, cosine


def _incident_spacing(x: np.ndarray, y: np.ndarray, wet: np.ndarray, axis: int) -> np.ndarray:
    """Average distances to immediate wet neighbors along one EFDC index axis."""

    total = np.zeros(x.shape, dtype=float)
    count = np.zeros(x.shape, dtype=np.int8)
    if axis == 1:
        pair = wet[:, :-1] & wet[:, 1:]
        distance = np.hypot(x[:, 1:] - x[:, :-1], y[:, 1:] - y[:, :-1])
        valid = pair & np.isfinite(distance) & (distance > 0.0)
        contribution = np.where(valid, distance, 0.0)
        total[:, :-1] += contribution
        total[:, 1:] += contribution
        count[:, :-1] += valid
        count[:, 1:] += valid
    elif axis == 0:
        pair = wet[:-1, :] & wet[1:, :]
        distance = np.hypot(x[1:, :] - x[:-1, :], y[1:, :] - y[:-1, :])
        valid = pair & np.isfinite(distance) & (distance > 0.0)
        contribution = np.where(valid, distance, 0.0)
        total[:-1, :] += contribution
        total[1:, :] += contribution
        count[:-1, :] += valid
        count[1:, :] += valid
    else:
        raise ValueError("axis must be 0 (eta) or 1 (xi).")
    return np.divide(total, count, out=np.full(x.shape, np.nan), where=count > 0)


def _orientation_from_wet_neighbors(
    x: np.ndarray, y: np.ndarray, wet: np.ndarray, xi_spacing: np.ndarray,
) -> np.ndarray:
    """Infer logical-XI orientation exclusively from immediate wet pairs."""

    vector_x = np.zeros(x.shape, dtype=float)
    vector_y = np.zeros(x.shape, dtype=float)
    count = np.zeros(x.shape, dtype=np.int8)
    pair = wet[:, :-1] & wet[:, 1:]
    delta_x, delta_y = x[:, 1:] - x[:, :-1], y[:, 1:] - y[:, :-1]
    valid = pair & np.isfinite(delta_x) & np.isfinite(delta_y) & (np.hypot(delta_x, delta_y) > 0.0)
    contribution_x, contribution_y = np.where(valid, delta_x, 0.0), np.where(valid, delta_y, 0.0)
    vector_x[:, :-1] += contribution_x
    vector_x[:, 1:] += contribution_x
    vector_y[:, :-1] += contribution_y
    vector_y[:, 1:] += contribution_y
    count[:, :-1] += valid
    count[:, 1:] += valid
    angle = np.arctan2(vector_y, vector_x)
    known = wet & (count > 0) & np.isfinite(xi_spacing)
    if not np.any(known):
        raise ValueError("Cannot infer wet-cell orientation without an immediate wet XI-neighbor pair.")
    mean_angle = float(np.arctan2(np.mean(np.sin(angle[known])), np.mean(np.cos(angle[known]))))
    return np.where(known, angle, mean_angle)


def build_wet_cell_footprints(
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
) -> WetCellFootprints:
    """Build independent wet-cell footprints without consulting dry coordinates.

    Center-to-center spacings are measured only across immediate ``derived wet_mask == 1``
    pairs. Boundary and one-cell-wide channel cells use the median wet-grid aspect
    ratio for a missing axis. Consequently, packed dry coordinates can never pull a
    wet footprint across a tributary seam, as center-coordinate ``pcolormesh`` can.
    """

    longitude = np.asarray(lon, dtype=float)
    latitude = np.asarray(lat, dtype=float)
    wet_mask = np.asarray(mask, dtype=float)
    if longitude.ndim != 2 or len({longitude.shape, latitude.shape, wet_mask.shape}) != 1:
        raise ValueError("lon, lat, and mask must share one two-dimensional grid shape.")
    wet = (wet_mask == 1) & np.isfinite(longitude) & np.isfinite(latitude)
    if not np.any(wet):
        raise ValueError("EFDC rho grid contains no finite wet coordinates.")
    x, y, reference_lon, reference_lat, cosine = _local_xy(longitude, latitude, wet)
    xi_spacing = _incident_spacing(x, y, wet, axis=1)
    eta_spacing = _incident_spacing(x, y, wet, axis=0)
    xi_observed, eta_observed = xi_spacing[wet & np.isfinite(xi_spacing)], eta_spacing[wet & np.isfinite(eta_spacing)]
    if not xi_observed.size and not eta_observed.size:
        raise ValueError("Cannot infer cell footprints from a grid with no adjacent wet cells.")
    xi_median = float(np.median(xi_observed)) if xi_observed.size else float(np.median(eta_observed))
    eta_median = float(np.median(eta_observed)) if eta_observed.size else float(np.median(xi_observed))
    aspect = xi_median / eta_median
    missing_xi = wet & ~np.isfinite(xi_spacing)
    missing_eta = wet & ~np.isfinite(eta_spacing)
    xi_spacing = np.where(missing_xi & np.isfinite(eta_spacing), eta_spacing * aspect, xi_spacing)
    eta_spacing = np.where(missing_eta & np.isfinite(xi_spacing), xi_spacing / aspect, eta_spacing)
    xi_spacing = np.where(wet & ~np.isfinite(xi_spacing), xi_median, xi_spacing)
    eta_spacing = np.where(wet & ~np.isfinite(eta_spacing), eta_median, eta_spacing)
    if np.any(wet & ((xi_spacing <= 0.0) | (eta_spacing <= 0.0))):
        raise ValueError("Wet-cell footprint spacing must be finite and positive.")

    orientation = _orientation_from_wet_neighbors(x, y, wet, xi_spacing)

    half_xi_x = 0.5 * xi_spacing * np.cos(orientation)
    half_xi_y = 0.5 * xi_spacing * np.sin(orientation)
    half_eta_x = -0.5 * eta_spacing * np.sin(orientation)
    half_eta_y = 0.5 * eta_spacing * np.cos(orientation)
    corner_x = np.stack((
        x - half_xi_x - half_eta_x,
        x + half_xi_x - half_eta_x,
        x + half_xi_x + half_eta_x,
        x - half_xi_x + half_eta_x,
    ), axis=-1)
    corner_y = np.stack((
        y - half_xi_y - half_eta_y,
        y + half_xi_y - half_eta_y,
        y + half_xi_y + half_eta_y,
        y - half_xi_y + half_eta_y,
    ), axis=-1)
    corner_lon = reference_lon + np.rad2deg(corner_x / (EARTH_RADIUS_METERS * cosine))
    corner_lat = reference_lat + np.rad2deg(corner_y / EARTH_RADIUS_METERS)
    all_polygons = np.stack((corner_lon, corner_lat), axis=-1)
    wet_flat_indices = np.flatnonzero(wet.ravel())
    polygons = all_polygons.reshape(-1, 4, 2)[wet_flat_indices]
    if not np.isfinite(polygons).all():
        raise ValueError("Inferred wet-cell footprints contain non-finite vertices.")
    fallback = (missing_xi | missing_eta).ravel()[wet_flat_indices]
    spans = np.hypot(xi_spacing[wet], eta_spacing[wet]) / 1000.0
    return WetCellFootprints(
        polygons=polygons,
        wet_flat_indices=wet_flat_indices,
        fallback_cells=np.asarray(fallback, dtype=bool),
        grid_shape=longitude.shape,
        maximum_span_km=float(np.max(spans)),
    )


def plot_efdc_scalar(
    ax,
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    mask: np.ndarray,
    values: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    title: str = "",
    colorbar_label: str | None = None,
    method: str = "wet_cells",
    contour_levels: int = 16,
    add_colorbar: bool = True,
    quiver_u: np.ndarray | None = None,
    quiver_v: np.ndarray | None = None,
    quiver_stride: int = 8,
    quiver_scale: float | None = None,
    footprints: WetCellFootprints | None = None,
) -> EFDCPlotResult:
    """Render one masked scalar frame and optional earth-relative current quivers."""

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise ValueError(f"Color limits require finite vmin < vmax; got {vmin}, {vmax}.")
    longitude, latitude, coordinate_wet, finite_wet, scalar, display = _validated(lon, lat, mask, values)
    fallback_count = None
    maximum_span_km = None
    if method == "wet_cells":
        footprints = footprints or build_wet_cell_footprints(
            longitude, latitude, coordinate_wet.astype(np.int8))
        expected_indices = np.flatnonzero(coordinate_wet.ravel())
        if footprints.grid_shape != longitude.shape or not np.array_equal(footprints.wet_flat_indices, expected_indices):
            raise ValueError("Precomputed wet-cell footprints do not match this EFDC wet grid.")
        rendered = np.isfinite(scalar.ravel()[footprints.wet_flat_indices])
        artist = PolyCollection(
            footprints.polygons[rendered], array=scalar.ravel()[footprints.wet_flat_indices][rendered],
            cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax, clip=False),
            edgecolors="none", linewidths=0.0, antialiaseds=False, rasterized=True,
        )
        ax.add_collection(artist)
        fallback_count = int(np.count_nonzero(footprints.fallback_cells))
        maximum_span_km = footprints.maximum_span_km
    elif method == "pcolormesh":
        if np.any(~coordinate_wet):
            raise ValueError(
                "Center-coordinate pcolormesh is unsafe on a masked EFDC grid because it can infer "
                "corners across dry or packed seams; use method='wet_cells'.")
        artist = ax.pcolormesh(longitude, latitude, display, shading="auto", cmap=cmap,
                               vmin=vmin, vmax=vmax, rasterized=True)
    elif method == "contourf":
        if contour_levels < 2:
            raise ValueError("contour_levels must be at least two.")
        levels = np.linspace(vmin, vmax, contour_levels + 1)
        artist = ax.contourf(longitude, latitude, display, levels=levels, cmap=cmap,
                             vmin=vmin, vmax=vmax, extend="both")
    else:
        raise ValueError("method must be wet_cells, pcolormesh, or contourf.")
    colorbar = ax.figure.colorbar(artist, ax=ax, pad=0.025, shrink=0.88) if add_colorbar else None
    if colorbar is not None and colorbar_label:
        colorbar.set_label(colorbar_label)

    quiver = None
    if (quiver_u is None) != (quiver_v is None):
        raise ValueError("Provide both earth-relative quiver components or neither.")
    if quiver_u is not None:
        east, north = np.asarray(quiver_u, dtype=float), np.asarray(quiver_v, dtype=float)
        if east.shape != longitude.shape or north.shape != longitude.shape:
            raise ValueError("Quiver components must share the EFDC rho-grid shape.")
        stride = max(1, int(quiver_stride))
        selection = np.s_[::stride, ::stride]
        valid = coordinate_wet[selection] & np.isfinite(east[selection]) & np.isfinite(north[selection])
        quiver = ax.quiver(
            np.where(valid, longitude[selection], np.nan),
            np.where(valid, latitude[selection], np.nan),
            np.where(valid, east[selection], np.nan),
            np.where(valid, north[selection], np.nan),
            color="black", alpha=0.75, angles="xy", scale_units="xy", scale=quiver_scale,
            width=0.0022, zorder=4,
        )

    if method == "wet_cells":
        valid_lon = footprints.polygons[:, :, 0].ravel()
        valid_lat = footprints.polygons[:, :, 1].ravel()
    else:
        valid_lon, valid_lat = longitude[coordinate_wet], latitude[coordinate_wet]
    lon_span, lat_span = float(np.ptp(valid_lon)), float(np.ptp(valid_lat))
    lon_pad, lat_pad = max(0.02 * lon_span, 1.0e-6), max(0.02 * lat_span, 1.0e-6)
    ax.set_xlim(float(np.min(valid_lon) - lon_pad), float(np.max(valid_lon) + lon_pad))
    ax.set_ylim(float(np.min(valid_lat) - lat_pad), float(np.max(valid_lat) + lat_pad))
    ax.set_aspect(geographic_aspect(valid_lat), adjustable="box")
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_title(title)
    ax.grid(True, color="0.75", linewidth=0.45, alpha=0.55)
    finite_count, wet_count = int(np.ma.count(display)), int(np.count_nonzero(coordinate_wet))
    return EFDCPlotResult(
        artist, colorbar, quiver, finite_count / wet_count, wet_count, finite_count,
        method, finite_count, fallback_count, maximum_span_km)


def save_efdc_scalar_map(output: str, *, lon, lat, mask, values, vmin, vmax,
                         cmap="viridis", title="", colorbar_label=None, dpi=140,
                         figure_size=(7.2, 6.0), **kwargs) -> EFDCPlotResult:
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    try:
        result = plot_efdc_scalar(
            ax, lon=lon, lat=lat, mask=mask, values=values, vmin=vmin, vmax=vmax,
            cmap=cmap, title=title, colorbar_label=colorbar_label, **kwargs)
        fig.savefig(output, dpi=dpi)
        return result
    finally:
        plt.close(fig)
