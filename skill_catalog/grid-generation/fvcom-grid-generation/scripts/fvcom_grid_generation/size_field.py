"""Hydraulic-skeleton mesh-size field for FVCOM grids.

The production field has one path:

* derive a solid-boundary background and a solid-only raster Voronoi medial
  skeleton; open-boundary segments never act as skeleton banks;
* validate opposing-bank cross-sections and rank them with a wet-distance
  storage-to-cross-sectional-area proxy;
* combine the smooth hydraulic-corridor and bathymetric-gradient targets with
  the solid-boundary target;
* hold the delivered coarse open-boundary target for a physical wet distance,
  then transfer authority in log space to the nearshore target; and
* apply a wet-domain eight-neighbour lower gradation envelope while keeping CFL
  diagnostic only.

The hydraulic skeleton is generated directly from the delivered boundary and
bathymetry; no external drainage-network artifact is accepted or generated.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label as connected_components, minimum_filter
from shapely import contains_xy, points
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
import xarray as xr

from .bathymetry import BathymetryGrid
from .boundary import BoundaryNodes
from .projection import project_points, unproject_points


EARTH_RADIUS_M = 6_371_000.0
OPEN_KINDS = {"open", "open_boundary"}
SOURCE_CODES = {
    0: "uncovered",
    1: "solid_boundary_background",
    2: "bathymetry_slope",
    3: "hydraulic_skeleton_corridor",
}


@dataclass(frozen=True)
class SizeFieldConfig:
    """Configuration for the single hydraulic-skeleton size-field algorithm."""

    land_spacing_m: float = 50.0
    open_spacing_m: float = 3000.0
    max_size_m: float = 20_000.0
    interior_min_size_m: float | None = None
    gradation: float = 0.20
    slope_elements: float = 10.0
    min_gradient: float = 1.0e-5
    coastal_distance_m: float = 25_000.0
    hydraulic_elements_across_min: float = 3.0
    hydraulic_elements_across_max: float = 8.0
    hydraulic_max_width_m: float = 20_000.0
    hydraulic_bank_angle_deg: float = 110.0
    hydraulic_longitudinal_gradation: float = 0.10
    hydraulic_corridor_width_factor: float = 0.55
    obc_hold_distance_m: float = 10_000.0
    obc_transition_distance_m: float = 60_000.0
    target_timestep_s: str | float = "auto"
    cfl: float = 0.5


@dataclass(frozen=True)
class BoundaryDistanceFields:
    """Exact boundary-family distances, targets, and nearest solid segments."""

    query_xy: np.ndarray
    open_distance_m: np.ndarray
    open_target_m: np.ndarray
    land_distance_m: np.ndarray
    land_target_m: np.ndarray
    land_segment_index: np.ndarray
    solid_background_m: np.ndarray
    open_segments: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    land_segments: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    land_segment_chain_id: np.ndarray
    land_segment_arc_mid_m: np.ndarray
    land_segment_chain_length_m: np.ndarray
    report: dict[str, Any]


class WetMaskAwareSizeInterpolator:
    """Sample a cell-centred size raster without sharing dry-cell halos.

    Ordinary bilinear interpolation can mix a wet target with a value stored
    at an inactive land/island cell. Giving that inactive cell a single halo
    value is also unsafe: the cell can be shared by hydraulically separated
    wet sides. This sampler therefore uses normal bilinear interpolation only
    when all four stencil cells are active. At a wet/dry interface it selects
    the positive-weight active stencil corner with the largest bilinear
    weight. If a callback query has no positive-weight active corner (for
    example, an exact CAD boundary query at a dry-cell centre), it falls back
    to the coarsest covered corner. That conservative fallback prevents a fine
    target from the wrong side of a dry barrier from overriding the separate
    boundary-trace sampler.

    This is a raster-interface guard, not a barrier-aware wet-distance solver.
    """

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        values: np.ndarray,
        active_mask: np.ndarray,
        coverage_mask: np.ndarray,
    ) -> None:
        self._lat = np.asarray(lat, dtype=float)
        self._lon = np.asarray(lon, dtype=float)
        self._values = np.asarray(values, dtype=float)
        self._active = np.asarray(active_mask, dtype=bool)
        self._coverage = np.asarray(coverage_mask, dtype=bool)
        expected = (len(self._lat), len(self._lon))
        for name, value in (
            ("values", self._values),
            ("active_mask", self._active),
            ("coverage_mask", self._coverage),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}; got {value.shape}"
                )
        if len(self._lat) < 2 or len(self._lon) < 2:
            raise ValueError(
                "wet-mask-aware interpolation needs at least two cells per axis"
            )
        if (
            np.any(~np.isfinite(self._lat))
            or np.any(~np.isfinite(self._lon))
            or np.any(np.diff(self._lat) <= 0.0)
            or np.any(np.diff(self._lon) <= 0.0)
        ):
            raise ValueError(
                "wet-mask-aware interpolation axes must be finite and "
                "strictly increasing"
            )
        invalid_covered = self._coverage & (
            ~np.isfinite(self._values) | (self._values <= 0.0)
        )
        if np.any(invalid_covered):
            raise ValueError(
                "covered size-field cells must contain finite positive values"
            )

    def sample(
        self,
        query_lat_lon: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Return sampled values, with NaN for out-of-coverage queries."""

        sampled, _support = self.sample_with_active_support(query_lat_lon)
        return sampled

    def sample_with_active_support(
        self,
        query_lat_lon: np.ndarray | Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return values and positive-weight wet-stencil support flags."""

        query = np.asarray(query_lat_lon, dtype=float)
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("size queries must have shape (N, 2) as lat/lon")
        if not len(query):
            return np.empty(0, dtype=float), np.empty(0, dtype=bool)
        finite = np.all(np.isfinite(query), axis=1)
        latitude_scale = max(
            1.0,
            abs(float(self._lat[0])),
            abs(float(self._lat[-1])),
        )
        longitude_scale = max(
            1.0,
            abs(float(self._lon[0])),
            abs(float(self._lon[-1])),
        )
        latitude_tolerance = 64.0 * np.finfo(float).eps * latitude_scale
        longitude_tolerance = 64.0 * np.finfo(float).eps * longitude_scale
        in_bounds = (
            finite
            & (query[:, 0] >= self._lat[0] - latitude_tolerance)
            & (query[:, 0] <= self._lat[-1] + latitude_tolerance)
            & (query[:, 1] >= self._lon[0] - longitude_tolerance)
            & (query[:, 1] <= self._lon[-1] + longitude_tolerance)
        )
        result = np.full(len(query), np.nan, dtype=float)
        active_support = np.zeros(len(query), dtype=bool)
        if not np.any(in_bounds):
            return result, active_support

        selected = np.flatnonzero(in_bounds)
        q_lat = np.clip(
            query[selected, 0],
            self._lat[0],
            self._lat[-1],
        )
        q_lon = np.clip(
            query[selected, 1],
            self._lon[0],
            self._lon[-1],
        )
        row_hi = np.clip(
            np.searchsorted(self._lat, q_lat, side="right"),
            1,
            len(self._lat) - 1,
        )
        col_hi = np.clip(
            np.searchsorted(self._lon, q_lon, side="right"),
            1,
            len(self._lon) - 1,
        )
        row_lo = row_hi - 1
        col_lo = col_hi - 1
        fy = (q_lat - self._lat[row_lo]) / (
            self._lat[row_hi] - self._lat[row_lo]
        )
        fx = (q_lon - self._lon[col_lo]) / (
            self._lon[col_hi] - self._lon[col_lo]
        )

        corner_values = np.column_stack(
            [
                self._values[row_lo, col_lo],
                self._values[row_lo, col_hi],
                self._values[row_hi, col_lo],
                self._values[row_hi, col_hi],
            ]
        )
        corner_active = np.column_stack(
            [
                self._active[row_lo, col_lo],
                self._active[row_lo, col_hi],
                self._active[row_hi, col_lo],
                self._active[row_hi, col_hi],
            ]
        )
        corner_covered = np.column_stack(
            [
                self._coverage[row_lo, col_lo],
                self._coverage[row_lo, col_hi],
                self._coverage[row_hi, col_lo],
                self._coverage[row_hi, col_hi],
            ]
        )
        weights = np.column_stack(
            [
                (1.0 - fy) * (1.0 - fx),
                (1.0 - fy) * fx,
                fy * (1.0 - fx),
                fy * fx,
            ]
        )
        local = np.full(len(selected), np.nan, dtype=float)
        all_active = np.all(corner_active, axis=1)
        if np.any(all_active):
            local[all_active] = np.sum(
                weights[all_active] * corner_values[all_active],
                axis=1,
            )

        positive_weight_active = corner_active & (
            weights > 64.0 * np.finfo(float).eps
        )
        interface = ~all_active & np.any(
            positive_weight_active,
            axis=1,
        )
        if np.any(interface):
            active_weights = np.where(
                positive_weight_active[interface],
                weights[interface],
                -np.inf,
            )
            chosen = np.argmax(active_weights, axis=1)
            local[interface] = corner_values[interface, chosen]
        local_active_support = all_active | interface

        no_positive_active = ~np.any(positive_weight_active, axis=1)
        covered_fallback = no_positive_active & np.any(
            corner_covered,
            axis=1,
        )
        if np.any(covered_fallback):
            covered_values = np.where(
                corner_covered[covered_fallback],
                corner_values[covered_fallback],
                np.nan,
            )
            local[covered_fallback] = np.nanmax(
                covered_values,
                axis=1,
            )
        result[selected] = local
        active_support[selected] = local_active_support
        return result, active_support


class WetMaskAwareSizeInterpolatorV1Replay(WetMaskAwareSizeInterpolator):
    """Faithfully replay archived ``fvcom_wet_mask_sampling_v1`` rasters.

    Version 1 treated any active stencil corner as support, including a corner
    with zero bilinear weight at an exact grid coordinate. When no corner was
    active, it selected the covered corner with greatest bilinear weight.
    These semantics are retained only for immutable provenance replay; new
    size fields use :class:`WetMaskAwareSizeInterpolator`.
    """

    def sample_with_active_support(
        self,
        query_lat_lon: np.ndarray | Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query_lat_lon, dtype=float)
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("size queries must have shape (N, 2) as lat/lon")
        if not len(query):
            return np.empty(0, dtype=float), np.empty(0, dtype=bool)
        finite = np.all(np.isfinite(query), axis=1)
        in_bounds = (
            finite
            & (query[:, 0] >= self._lat[0])
            & (query[:, 0] <= self._lat[-1])
            & (query[:, 1] >= self._lon[0])
            & (query[:, 1] <= self._lon[-1])
        )
        result = np.full(len(query), np.nan, dtype=float)
        active_support = np.zeros(len(query), dtype=bool)
        if not np.any(in_bounds):
            return result, active_support

        selected = np.flatnonzero(in_bounds)
        q_lat = query[selected, 0]
        q_lon = query[selected, 1]
        row_hi = np.clip(
            np.searchsorted(self._lat, q_lat, side="right"),
            1,
            len(self._lat) - 1,
        )
        col_hi = np.clip(
            np.searchsorted(self._lon, q_lon, side="right"),
            1,
            len(self._lon) - 1,
        )
        row_lo = row_hi - 1
        col_lo = col_hi - 1
        fy = (q_lat - self._lat[row_lo]) / (
            self._lat[row_hi] - self._lat[row_lo]
        )
        fx = (q_lon - self._lon[col_lo]) / (
            self._lon[col_hi] - self._lon[col_lo]
        )
        corner_values = np.column_stack(
            [
                self._values[row_lo, col_lo],
                self._values[row_lo, col_hi],
                self._values[row_hi, col_lo],
                self._values[row_hi, col_hi],
            ]
        )
        corner_active = np.column_stack(
            [
                self._active[row_lo, col_lo],
                self._active[row_lo, col_hi],
                self._active[row_hi, col_lo],
                self._active[row_hi, col_hi],
            ]
        )
        corner_covered = np.column_stack(
            [
                self._coverage[row_lo, col_lo],
                self._coverage[row_lo, col_hi],
                self._coverage[row_hi, col_lo],
                self._coverage[row_hi, col_hi],
            ]
        )
        weights = np.column_stack(
            [
                (1.0 - fy) * (1.0 - fx),
                (1.0 - fy) * fx,
                fy * (1.0 - fx),
                fy * fx,
            ]
        )
        local = np.full(len(selected), np.nan, dtype=float)
        all_active = np.all(corner_active, axis=1)
        if np.any(all_active):
            local[all_active] = np.sum(
                weights[all_active] * corner_values[all_active],
                axis=1,
            )

        interface = ~all_active & np.any(corner_active, axis=1)
        if np.any(interface):
            active_weights = np.where(
                corner_active[interface],
                weights[interface],
                -np.inf,
            )
            chosen = np.argmax(active_weights, axis=1)
            local[interface] = corner_values[interface, chosen]

        no_active = ~np.any(corner_active, axis=1)
        covered_fallback = no_active & np.any(corner_covered, axis=1)
        if np.any(covered_fallback):
            covered_weights = np.where(
                corner_covered[covered_fallback],
                weights[covered_fallback],
                -np.inf,
            )
            chosen = np.argmax(covered_weights, axis=1)
            local[covered_fallback] = corner_values[
                covered_fallback,
                chosen,
            ]

        result[selected] = local
        active_support[selected] = all_active | interface
        return result, active_support


class LegacyCoveredBilinearSizeInterpolator:
    """Replay historical canonical rasters that predate wet-mask sampling."""

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        values: np.ndarray,
        coverage_mask: np.ndarray,
    ) -> None:
        self._lat = np.asarray(lat, dtype=float)
        self._lon = np.asarray(lon, dtype=float)
        self._values_array = np.asarray(values, dtype=float)
        self._coverage_array = np.asarray(coverage_mask, dtype=bool)
        expected = (len(self._lat), len(self._lon))
        if (
            self._values_array.shape != expected
            or self._coverage_array.shape != expected
        ):
            raise ValueError(
                "legacy size raster and coverage must match lat/lon axes"
            )
        self._values = RegularGridInterpolator(
            (self._lat, self._lon),
            self._values_array,
            bounds_error=False,
            fill_value=np.nan,
        )
        self._coverage = RegularGridInterpolator(
            (self._lat, self._lon),
            self._coverage_array.astype(np.uint8),
            method="nearest",
            bounds_error=False,
            fill_value=0,
        )

    def sample(
        self,
        query_lat_lon: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        sampled, _support = self.sample_with_active_support(query_lat_lon)
        return sampled

    def sample_with_active_support(
        self,
        query_lat_lon: np.ndarray | Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query_lat_lon, dtype=float)
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError("size queries must have shape (N, 2) as lat/lon")
        sampled = np.asarray(self._values(query), dtype=float)
        covered = np.asarray(self._coverage(query), dtype=float) >= 0.5
        invalid = ~covered | ~np.isfinite(sampled) | (sampled <= 0.0)
        sampled[invalid] = np.nan
        return sampled, covered & ~invalid


def recorded_size_interpolator(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    coverage_mask: np.ndarray,
    domain_mask: np.ndarray,
    sampling_interface_schema_version: str | None,
) -> (
    WetMaskAwareSizeInterpolator
    | WetMaskAwareSizeInterpolatorV1Replay
    | LegacyCoveredBilinearSizeInterpolator
):
    """Dispatch an exported canonical raster by its recorded sampler schema."""

    schema = str(sampling_interface_schema_version or "").strip()
    if schema == "fvcom_wet_mask_sampling_v1":
        return WetMaskAwareSizeInterpolatorV1Replay(
            lat,
            lon,
            values,
            np.asarray(coverage_mask, dtype=bool)
            & np.asarray(domain_mask, dtype=bool),
            coverage_mask,
        )
    if schema == "fvcom_wet_mask_sampling_v2":
        return WetMaskAwareSizeInterpolator(
            lat,
            lon,
            values,
            np.asarray(coverage_mask, dtype=bool)
            & np.asarray(domain_mask, dtype=bool),
            coverage_mask,
        )
    if schema in {
        "",
        "legacy_unspecified",
        "fvcom_size_sampling_halo_v1",
    }:
        return LegacyCoveredBilinearSizeInterpolator(
            lat,
            lon,
            values,
            coverage_mask,
        )
    raise ValueError(
        f"unsupported canonical sampling interface schema {schema!r}"
    )


def linear_target_metric_edge_fractions(
    length_m: float,
    target_start_m: float,
    target_end_m: float,
    *,
    samples_per_target: float,
    include_end: bool = False,
    maximum_sample_count: int | None = None,
) -> np.ndarray:
    """Return deterministic trace fractions for a linear endpoint target.

    Fractions are equidistributed in ``ds / h(s)`` and always include the
    physical midpoint used by the boundary reconciliation audit. Production
    oriented chains omit the end because the next edge supplies it; unordered
    diagnostic edge sets may include it without changing the min-plus field.
    """

    length = float(length_m)
    ha = float(target_start_m)
    hb = float(target_end_m)
    density = float(samples_per_target)
    if not np.isfinite(length) or length < 0.0:
        raise ValueError("boundary trace edge length must be finite and nonnegative")
    if (
        not np.isfinite(ha)
        or ha <= 0.0
        or not np.isfinite(hb)
        or hb <= 0.0
    ):
        raise ValueError("boundary trace endpoint targets must be finite and positive")
    if not np.isfinite(density) or density < 2.0:
        raise ValueError("samples_per_target must be at least two")
    sample_limit = (
        None
        if maximum_sample_count is None
        else int(maximum_sample_count)
    )
    if sample_limit is not None and sample_limit < 1:
        raise ValueError(
            "boundary trace sampling exceeds the safety limit before allocation"
        )
    target_scale = max(abs(ha), abs(hb), 1.0)
    nearly_constant = abs(hb - ha) <= (
        64.0 * np.finfo(float).eps * target_scale
    )
    metric_length = (
        length / ha
        if nearly_constant
        else length * np.log(hb / ha) / (hb - ha)
    )
    intervals = max(2, int(np.ceil(metric_length * density)))
    if intervals % 2:
        intervals += 1
    stop = intervals + 1 if include_end else intervals
    if sample_limit is not None and stop > sample_limit:
        raise ValueError(
            "boundary trace sampling exceeds the safety limit "
            f"of {sample_limit} points before allocation"
        )
    metric_fraction = np.arange(stop, dtype=float) / intervals
    if nearly_constant:
        fraction = metric_fraction
    else:
        fraction = (
            ha
            * np.expm1(metric_fraction * np.log(hb / ha))
            / (hb - ha)
        )
    if not np.any(fraction == 0.5):
        if sample_limit is not None and len(fraction) >= sample_limit:
            raise ValueError(
                "boundary trace sampling exceeds the safety limit "
                f"of {sample_limit} points before midpoint insertion"
            )
        fraction = np.concatenate(
            [fraction, np.asarray([0.5], dtype=float)]
        )
    return np.unique(fraction)


@dataclass(frozen=True)
class SizeField:
    lon: np.ndarray
    lat: np.ndarray
    size: np.ndarray
    raw_size: np.ndarray
    depth: np.ndarray
    slope: np.ndarray
    report: dict[str, Any]
    boundary_size: np.ndarray
    slope_size: np.ndarray
    hydraulic_size: np.ndarray
    nearshore_size: np.ndarray
    obc_transition_size: np.ndarray
    open_boundary_distance_m: np.ndarray
    wet_obc_distance_m: np.ndarray
    wet_obc_target_m: np.ndarray
    land_boundary_distance_m: np.ndarray
    transition_fraction: np.ndarray
    coastal_mask: np.ndarray
    hydraulic_skeleton_mask: np.ndarray
    hydraulic_corridor_mask: np.ndarray
    hydraulic_width_m: np.ndarray
    hydraulic_importance: np.ndarray
    hydraulic_storage_area_m2: np.ndarray
    hydraulic_cross_section_area_m2: np.ndarray
    coverage_mask: np.ndarray
    domain_mask: np.ndarray
    source_attribution: np.ndarray

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Sample the field and reject every point outside explicit coverage."""
        lon, lat = np.broadcast_arrays(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        query = np.column_stack([lat.ravel(), lon.ravel()])
        size_interp = WetMaskAwareSizeInterpolator(
            self.lat,
            self.lon,
            self.size,
            self.coverage_mask & self.domain_mask,
            self.coverage_mask,
        )
        sampled = np.asarray(size_interp.sample(query), dtype=float)
        uncovered = ~np.isfinite(sampled)
        if np.any(uncovered):
            first = int(np.flatnonzero(uncovered)[0])
            raise ValueError(
                "Size-field sampling requested outside explicit coverage: "
                f"{int(np.count_nonzero(uncovered))} point(s); first lon/lat="
                f"({float(lon.ravel()[first]):.8f}, {float(lat.ravel()[first]):.8f})"
            )
        return sampled.reshape(lon.shape)


def build_size_field(
    bathy: BathymetryGrid,
    boundary: BoundaryNodes,
    config: SizeFieldConfig,
    *,
    coverage_mask: np.ndarray | None = None,
    domain_mask: np.ndarray | None = None,
) -> SizeField:
    """Build the single hydraulic-skeleton FVCOM production size field."""
    _validate_config(config)
    depth = np.asarray(bathy.depth, dtype=float)
    expected_shape = (len(bathy.lat), len(bathy.lon))
    if depth.shape != expected_shape:
        raise ValueError(f"Bathymetry depth must have shape {expected_shape}; got {depth.shape}")

    coverage = np.isfinite(depth)
    if coverage_mask is not None:
        supplied_coverage = _grid_mask(coverage_mask, expected_shape, "coverage_mask")
        coverage &= supplied_coverage
    if not np.any(coverage):
        raise ValueError("The size field has no covered finite bathymetry cells")
    boundary_coverage = _boundary_coverage_report(bathy, boundary, coverage)
    if boundary_coverage["uncovered_boundary_node_count"]:
        raise ValueError(
            "Boundary is outside explicit size-field coverage: "
            f"{boundary_coverage['uncovered_boundary_node_count']} of "
            f"{boundary_coverage['boundary_node_count']} node(s)"
        )

    if domain_mask is None:
        model_mask = model_domain_mask(bathy, boundary)
    else:
        model_mask = _grid_mask(domain_mask, expected_shape, "domain_mask")
    active = coverage & model_mask
    if not np.any(active):
        raise ValueError("The model-domain mask contains no covered cells")

    boundary_minimum = _minimum_size(boundary, config)
    minimum = _interior_minimum_size(boundary, config)
    boundary_fields = boundary_distance_fields(bathy, boundary, config)
    background = np.where(coverage, boundary_fields.solid_background_m, np.nan)
    open_distance = boundary_fields.open_distance_m
    land_distance = boundary_fields.land_distance_m
    coastal = active & (land_distance <= float(config.coastal_distance_m))
    slope = bathymetric_gradient(bathy)
    safe_depth = np.maximum(depth, 1.0)
    slope_size = (
        (2.0 * np.pi / float(config.slope_elements))
        * safe_depth
        / np.maximum(slope, float(config.min_gradient))
    )

    dx_m, dy_m = _grid_spacing_m(bathy.lon, bathy.lat)
    has_open = bool(len(boundary_fields.open_segments[0]))
    wet_obc_distance = np.full(expected_shape, np.inf, dtype=float)
    wet_obc_target = np.full(expected_shape, np.nan, dtype=float)
    open_seed_count = 0
    if has_open:
        support = float(np.hypot(dx_m, dy_m))
        open_seeds = active & np.isfinite(open_distance) & (
            open_distance <= 1.5 * support
        )
        if not np.any(open_seeds):
            active_indices = np.flatnonzero(active.ravel())
            nearest = int(
                active_indices[
                    np.argmin(open_distance.ravel()[active_indices])
                ]
            )
            open_seeds.ravel()[nearest] = True
        open_seed_count = int(np.count_nonzero(open_seeds))
        wet_obc_distance, open_source = _wet_graph_distance_and_labels(
            active,
            open_seeds,
            dx_m,
            dy_m,
        )
        reachable_open = active & np.isfinite(wet_obc_distance) & (
            open_source >= 0
        )
        wet_obc_target[reachable_open] = (
            boundary_fields.open_target_m.ravel()[
                open_source[reachable_open]
            ]
        )

    (
        hydraulic_size,
        hydraulic_skeleton,
        hydraulic_corridor,
        hydraulic_width,
        hydraulic_importance,
        hydraulic_storage,
        hydraulic_cross_section,
        hydraulic_report,
    ) = hydraulic_skeleton_size(
        bathy,
        boundary,
        config,
        boundary_fields,
        active,
        coastal,
        wet_obc_distance,
    )

    nearshore_raw = np.asarray(background, dtype=float).copy()
    source = np.where(active, 1, 0).astype(np.int16)
    for code, candidate in (
        (2, np.where(coastal, slope_size, np.inf)),
        (3, np.where(hydraulic_corridor, hydraulic_size, np.inf)),
    ):
        selected = active & np.isfinite(candidate) & (candidate < nearshore_raw)
        nearshore_raw[selected] = candidate[selected]
        source[selected] = code

    nearshore_raw[active] = np.clip(
        nearshore_raw[active],
        minimum,
        float(config.max_size_m),
    )
    nearshore_active, nearshore_gradation_report = apply_gradation_limit(
        bathy.lon,
        bathy.lat,
        np.where(active, nearshore_raw, np.nan),
        float(config.gradation),
        connectivity=8,
    )
    nearshore = nearshore_raw.copy()
    nearshore[active] = nearshore_active[active]

    transition_fraction = np.full(expected_shape, np.nan, dtype=float)
    obc_transition_size = nearshore.copy()
    transition_report: dict[str, Any]
    if has_open:
        open_target = np.clip(
            wet_obc_target,
            minimum,
            float(config.max_size_m),
        )
        transfer_mask = (
            active
            & np.isfinite(wet_obc_distance)
            & np.isfinite(open_target)
        )
        if not np.any(transfer_mask):
            raise ValueError(
                "Open-boundary segments exist, but no wet raster cell is "
                "connected to an open-boundary seed"
            )
        requested_transition = float(config.obc_transition_distance_m)
        required_transition = _required_obc_transition_distance(
            open_target[transfer_mask],
            nearshore[transfer_mask],
            float(config.gradation),
        )
        available_transition = max(
            float(np.max(wet_obc_distance[transfer_mask]))
            - float(config.obc_hold_distance_m),
            0.0,
        )
        # Treat the configured distance as a minimum.  A shorter log-transfer
        # cannot satisfy the declared wet-domain gradation, and previously
        # created an avoidable OBC-to-bulk resolution kink even though ample
        # wet distance was available.
        effective_transition = max(
            requested_transition,
            float(required_transition),
        )
        xi = np.clip(
            (
                wet_obc_distance
                - float(config.obc_hold_distance_m)
            )
            / max(effective_transition, 1.0e-12),
            0.0,
            1.0,
        )
        transition_fraction = _quintic_smootherstep(xi)
        obc_transition_size[transfer_mask] = np.exp(
            (1.0 - transition_fraction[transfer_mask])
            * np.log(open_target[transfer_mask])
            + transition_fraction[transfer_mask]
            * np.log(nearshore[transfer_mask])
        )
        transition_fraction[active & ~transfer_mask] = np.nan
        transition_report = {
            "method": "wet_distance_quintic_log_authority_transfer",
            "open_seed_cell_count": int(open_seed_count),
            "hold_distance_m": float(config.obc_hold_distance_m),
            "requested_transition_distance_m": requested_transition,
            "required_transition_distance_m": float(required_transition),
            "effective_transition_distance_m": float(effective_transition),
            "available_transition_distance_m": float(available_transition),
            "configured_transition_shorter_than_derivative_requirement": bool(
                requested_transition + 1.0e-9 < required_transition
            ),
            "full_transition_fits_available_wet_distance": bool(
                effective_transition <= available_transition + 1.0e-9
            ),
            "effective_distance_auto_extended_for_gradation": bool(
                effective_transition > requested_transition + 1.0e-9
            ),
            "wet_distance_reachable_cell_count": int(
                np.count_nonzero(active & np.isfinite(wet_obc_distance))
            ),
            "wet_distance_unreachable_cell_count": int(
                np.count_nonzero(active & ~transfer_mask)
            ),
            "unreachable_cell_policy": (
                "retain_nearshore_target; no offshore authority is invented"
            ),
            "hold_cell_count": int(
                np.count_nonzero(
                    active
                    & (
                        wet_obc_distance
                        <= float(config.obc_hold_distance_m) + 1.0e-9
                    )
                )
            ),
        }
    else:
        transition_report = {
            "method": "closed_domain_no_open_boundary_transfer",
            "open_seed_cell_count": 0,
            "hold_distance_m": None,
            "requested_transition_distance_m": None,
            "required_transition_distance_m": None,
            "effective_transition_distance_m": None,
            "available_transition_distance_m": None,
            "configured_transition_shorter_than_derivative_requirement": False,
            "full_transition_fits_available_wet_distance": False,
            "wet_distance_reachable_cell_count": 0,
            "wet_distance_unreachable_cell_count": 0,
            "unreachable_cell_policy": None,
            "hold_cell_count": 0,
        }

    raw = obc_transition_size.copy()
    raw[coverage] = np.clip(raw[coverage], minimum, float(config.max_size_m))
    raw[~coverage] = np.nan
    limited_active, gradation_report = apply_gradation_limit(
        bathy.lon,
        bathy.lat,
        np.where(active, raw, np.nan),
        float(config.gradation),
        connectivity=8,
    )
    limited = raw.copy()
    limited[active] = limited_active[active]
    limited[~coverage] = np.nan
    if np.any(limited[active] > raw[active] + 1.0e-9):
        raise RuntimeError("Gradation limiter coarsened at least one wet-domain cell")
    sampling_interface_report = {
        "schema_version": "fvcom_wet_mask_sampling_v2",
        "method": (
            "bilinear_only_for_all_wet_stencil_else_highest_weight_"
            "active_corner"
        ),
        "shared_inactive_halo_used": False,
        "dry_or_zero_weight_fallback": "coarsest_covered_corner",
        "axis_roundtrip_tolerance": "64_machine_eps_times_axis_scale",
        "covered_fallback_when_no_active_stencil_corner": True,
        "budget_domain": "active_wet_cells_only",
        "barrier_aware_wet_distance": False,
        "interpretation": (
            "prevents a shared dry-cell halo from transmitting a target "
            "between wet sides; does not replace a barrier-aware field"
        ),
    }

    hold_mask = (
        active
        & has_open
        & (
            wet_obc_distance
            <= float(config.obc_hold_distance_m) + 1.0e-9
        )
    )
    hold_debt = np.zeros(expected_shape, dtype=float)
    hold_debt[hold_mask] = np.maximum(
        raw[hold_mask] - limited[hold_mask],
        0.0,
    )
    transition_report["post_gradation_hold_debt_cell_count"] = int(
        np.count_nonzero(hold_debt > 1.0e-6)
    )
    transition_report["post_gradation_max_hold_debt_m"] = float(
        np.max(hold_debt, initial=0.0)
    )
    transition_report["hold_preserved"] = bool(
        transition_report["post_gradation_hold_debt_cell_count"] == 0
    )

    cfl_report = cfl_size_report(depth, limited, config, mask=active)
    budget = estimate_node_budget(
        bathy.lon,
        bathy.lat,
        limited,
        coverage_mask=coverage,
        domain_mask=model_mask,
    )
    source_counts = {
        SOURCE_CODES[code]: int(np.count_nonzero(coverage & (source == code)))
        for code in sorted(SOURCE_CODES)
    }
    report = {
        "schema_version": "fvcom_size_field_v4",
        "method": "solid_slope_hydraulic_skeleton_obc_log_transition",
        "coverage": {
            "policy": "strict",
            "covered_cell_count": int(np.count_nonzero(coverage)),
            "uncovered_cell_count": int(coverage.size - np.count_nonzero(coverage)),
            "covered_cell_fraction": float(np.count_nonzero(coverage) / max(coverage.size, 1)),
            **boundary_coverage,
        },
        "domain": {
            "active_cell_count": int(np.count_nonzero(active)),
            "coastal_cell_count": int(np.count_nonzero(coastal)),
            "coastal_distance_m": float(config.coastal_distance_m),
        },
        "boundary": boundary_fields.report,
        "hydraulic_skeleton": hydraulic_report,
        "open_boundary_transition": transition_report,
        "candidate_formulas": {
            "slope": "(2*pi/N_s)*max(depth,1)/max(abs(grad_b),epsilon)",
            "hydraulic_skeleton": "clip(width/(N_min+(N_max-N_min)*importance),h_min,h_max)",
            "hydraulic_corridor": "exp((1-P(v))*log(h_solid)+P(v)*log(h_skeleton)); P(v)=6v^5-15v^4+10v^3",
            "obc_transfer": "exp((1-alpha)*log(h_open)+alpha*log(h_nearshore)); alpha=P(clip((d_wet-L_hold)/L_transition))",
        },
        "source_attribution_codes": {str(key): value for key, value in SOURCE_CODES.items()},
        "source_attribution_cell_counts": source_counts,
        "source_attribution_stage": "nearshore_pointwise_minimum_before_obc_transfer_and_gradation",
        "nearshore_gradation": nearshore_gradation_report,
        "gradation": gradation_report,
        "sampling_interface": sampling_interface_report,
        "cfl": cfl_report,
        "node_budget_estimate": budget,
        "min_size_m": float(np.nanmin(limited[coverage])),
        "min_active_size_m": float(np.nanmin(limited[active])),
        "max_size_m": float(np.nanmax(limited[coverage])),
        "configured_min_size_m": float(minimum),
        "configured_boundary_trace_min_size_m": float(
            boundary_minimum
        ),
        "interior_minimum_separated_from_boundary_geometry": bool(
            minimum > boundary_minimum + 1.0e-9
        ),
        "land_spacing_m": float(config.land_spacing_m),
        "open_spacing_m": float(config.open_spacing_m),
        "slope_active_cell_count": int(np.count_nonzero(coastal & np.isfinite(slope_size))),
        "shallow_slope_active_cell_count": int(
            np.count_nonzero(coastal & np.isfinite(slope_size) & (depth <= 50.0))
        ),
    }
    return SizeField(
        lon=np.asarray(bathy.lon, dtype=float),
        lat=np.asarray(bathy.lat, dtype=float),
        size=limited,
        raw_size=raw,
        depth=depth,
        slope=slope,
        report=report,
        boundary_size=background,
        slope_size=np.where(coastal, slope_size, np.nan),
        hydraulic_size=np.where(
            hydraulic_corridor & np.isfinite(hydraulic_size),
            hydraulic_size,
            np.nan,
        ),
        nearshore_size=np.where(coverage, nearshore, np.nan),
        obc_transition_size=np.where(coverage, obc_transition_size, np.nan),
        open_boundary_distance_m=open_distance,
        wet_obc_distance_m=wet_obc_distance,
        wet_obc_target_m=wet_obc_target,
        land_boundary_distance_m=land_distance,
        transition_fraction=transition_fraction,
        coastal_mask=coastal,
        hydraulic_skeleton_mask=hydraulic_skeleton,
        hydraulic_corridor_mask=hydraulic_corridor,
        hydraulic_width_m=hydraulic_width,
        hydraulic_importance=hydraulic_importance,
        hydraulic_storage_area_m2=hydraulic_storage,
        hydraulic_cross_section_area_m2=hydraulic_cross_section,
        coverage_mask=coverage,
        domain_mask=model_mask,
        source_attribution=source,
    )


def boundary_distance_fields(
    bathy: BathymetryGrid,
    boundary: BoundaryNodes,
    config: SizeFieldConfig,
) -> BoundaryDistanceFields:
    """Return exact open/solid distances and the solid-boundary background."""
    lon2, lat2 = np.meshgrid(bathy.lon, bathy.lat)
    query_xy = project_points(
        np.column_stack([lon2.ravel(), lat2.ravel()]),
        boundary.projection,
    )
    families = _boundary_family_segments(boundary)
    (
        land_chain_id,
        land_arc_mid_m,
        land_chain_length_m,
    ) = _solid_segment_lineage(boundary)
    if len(land_chain_id) != len(families["land"][0]):
        raise RuntimeError(
            "Solid-boundary segment lineage is not aligned with segment geometry"
        )
    open_distance, open_target, _, _ = _nearest_segment_details(
        query_xy,
        *families["open"],
    )
    land_distance, land_target, land_segment_index, _ = _nearest_segment_details(
        query_xy,
        *families["land"],
    )
    has_open = bool(len(families["open"][0]))
    has_land = bool(len(families["land"][0]))
    shape = lon2.shape
    if has_land:
        land_dynamic = (
            land_target
            if boundary.adaptive_resolution
            else np.maximum(land_target, float(config.land_spacing_m))
        )
        background = land_dynamic + float(config.gradation) * land_distance
        mode = "solid_segment_target_plus_metric_distance"
    else:
        background = np.full(len(query_xy), float(config.max_size_m), dtype=float)
        land_target = np.full(len(query_xy), float(config.max_size_m), dtype=float)
        mode = "no_solid_segments_maximum"

    background = np.clip(
        background,
        _interior_minimum_size(boundary, config),
        float(config.max_size_m),
    )
    report = {
        "method": mode,
        "open_segment_count": int(len(families["open"][0])),
        "land_segment_count": int(len(families["land"][0])),
        "open_segments_excluded_from_medial_skeleton": True,
        "adaptive_boundary_targets_used_directly": bool(
            boundary.adaptive_resolution
        ),
        "legacy_land_dynamic_floor_m": (
            None
            if boundary.adaptive_resolution
            else float(config.land_spacing_m)
        ),
        "delivered_land_target_min_m": (
            float(np.nanmin(land_target)) if has_land else None
        ),
        "delivered_land_target_max_m": (
            float(np.nanmax(land_target)) if has_land else None
        ),
    }
    return BoundaryDistanceFields(
        query_xy=query_xy.reshape((*shape, 2)),
        open_distance_m=open_distance.reshape(shape),
        open_target_m=open_target.reshape(shape),
        land_distance_m=land_distance.reshape(shape),
        land_target_m=land_target.reshape(shape),
        land_segment_index=land_segment_index.reshape(shape),
        solid_background_m=background.reshape(shape),
        open_segments=families["open"],
        land_segments=families["land"],
        land_segment_chain_id=land_chain_id,
        land_segment_arc_mid_m=land_arc_mid_m,
        land_segment_chain_length_m=land_chain_length_m,
        report=report,
    )


def hydraulic_skeleton_size(
    bathy: BathymetryGrid,
    boundary: BoundaryNodes,
    config: SizeFieldConfig,
    fields: BoundaryDistanceFields,
    active: np.ndarray,
    coastal: np.ndarray,
    wet_obc_distance: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Build a paired-bank medial-skeleton corridor target."""
    shape = np.asarray(bathy.depth).shape
    hydraulic = np.full(shape, np.inf, dtype=float)
    skeleton = np.zeros(shape, dtype=bool)
    corridor = np.zeros(shape, dtype=bool)
    width_grid = np.full(shape, np.nan, dtype=float)
    importance_grid = np.full(shape, np.nan, dtype=float)
    storage_grid = np.full(shape, np.nan, dtype=float)
    cross_section_grid = np.full(shape, np.nan, dtype=float)
    land_segments = fields.land_segments
    if not len(land_segments[0]):
        return (
            hydraulic,
            skeleton,
            corridor,
            width_grid,
            importance_grid,
            storage_grid,
            cross_section_grid,
            {
                "method": "solid_only_raster_voronoi_medial_skeleton",
                "status": "no_solid_segments",
                "skeleton_cell_count": 0,
                "corridor_cell_count": 0,
            },
        )

    dx_m, dy_m = _grid_spacing_m(bathy.lon, bathy.lat)
    (
        skeleton,
        width_grid,
        bank_a,
        bank_b,
        bank_angle,
        ridge_balance,
        detection_report,
    ) = _detect_solid_medial_skeleton(
        fields.query_xy,
        active & coastal,
        fields.land_segment_index,
        land_segments[0],
        land_segments[1],
        fields.land_segment_chain_id,
        fields.land_segment_arc_mid_m,
        fields.land_segment_chain_length_m,
        config,
        dx_m,
        dy_m,
        _interior_minimum_size(boundary, config),
    )

    candidate_indices = np.flatnonzero(skeleton.ravel())
    cross_section_report: dict[str, Any] = {
        "candidate_count": int(len(candidate_indices)),
        "validated_count": 0,
        "rejected_cross_land_count": 0,
    }
    if len(candidate_indices):
        query = fields.query_xy.reshape((-1, 2))[candidate_indices]
        segment_a = bank_a.ravel()[candidate_indices]
        segment_b = bank_b.ravel()[candidate_indices]
        _, contact_a = _closest_points_on_segments(
            query,
            segment_a,
            land_segments[0],
            land_segments[1],
        )
        _, contact_b = _closest_points_on_segments(
            query,
            segment_b,
            land_segments[0],
            land_segments[1],
        )
        fractions = np.linspace(0.05, 0.95, 9, dtype=float)
        section_xy = (
            contact_a[:, None, :]
            + fractions[None, :, None]
            * (contact_b - contact_a)[:, None, :]
        )
        section_wet = contains_xy(
            boundary.domain_polygon_xy,
            section_xy[..., 0].ravel(),
            section_xy[..., 1].ravel(),
        ).reshape((len(candidate_indices), len(fractions)))
        valid_section = np.mean(section_wet, axis=1) >= 8.0 / 9.0
        rejected = candidate_indices[~valid_section]
        skeleton.ravel()[rejected] = False
        cross_section_report["rejected_cross_land_count"] = int(len(rejected))

        valid_indices = candidate_indices[valid_section]
        if len(valid_indices):
            section_xy = section_xy[valid_section]
            section_lonlat = unproject_points(
                section_xy.reshape((-1, 2)),
                boundary.projection,
            )
            section_depth = bathy.sample(
                section_lonlat[:, 0],
                section_lonlat[:, 1],
                fill_value=np.nan,
            ).reshape((len(valid_indices), len(fractions)))
            section_depth = np.maximum(
                np.where(np.isfinite(section_depth), section_depth, 1.0),
                0.5,
            )
            section_width = width_grid.ravel()[valid_indices]
            section_x = fractions[None, :] * section_width[:, None]
            cross_area = np.trapezoid(section_depth, x=section_x, axis=1)
            cross_section_grid.ravel()[valid_indices] = cross_area
            cross_section_report["validated_count"] = int(len(valid_indices))

    skeleton = _remove_small_skeleton_components(skeleton, minimum_cells=3)
    removed_after_validation = (
        np.isfinite(cross_section_grid) & ~skeleton
    )
    cross_section_grid[removed_after_validation] = np.nan
    width_grid[~skeleton] = np.nan
    bank_angle[~skeleton] = np.nan
    ridge_balance[~skeleton] = np.nan

    skeleton_indices = np.flatnonzero(skeleton.ravel())
    if not len(skeleton_indices):
        return (
            hydraulic,
            skeleton,
            corridor,
            width_grid,
            importance_grid,
            storage_grid,
            cross_section_grid,
            {
                "method": "solid_only_raster_voronoi_medial_skeleton",
                "status": "no_valid_paired_bank_skeleton",
                "detection": detection_report,
                "cross_sections": cross_section_report,
                "skeleton_cell_count": 0,
                "corridor_cell_count": 0,
            },
        )

    cell_area = _metric_cell_area(bathy.lon, bathy.lat)
    cumulative_storage = _cumulative_storage_proxy(
        active,
        wet_obc_distance,
        cell_area,
    )
    storage_values = cumulative_storage.ravel()[skeleton_indices]
    if np.any(~np.isfinite(storage_values)):
        total_area = float(np.sum(cell_area[active]))
        storage_values = np.where(
            np.isfinite(storage_values),
            storage_values,
            total_area,
        )
    storage_grid.ravel()[skeleton_indices] = storage_values
    cross_area = np.maximum(
        cross_section_grid.ravel()[skeleton_indices],
        1.0,
    )
    velocity_proxy = storage_values / cross_area
    log_proxy = np.log(np.maximum(velocity_proxy, 1.0e-12))
    q_low, q_high = np.quantile(log_proxy, [0.10, 0.90])
    if q_high > q_low + 1.0e-12:
        importance = np.clip(
            (log_proxy - q_low) / (q_high - q_low),
            0.0,
            1.0,
        )
    else:
        importance = np.full(len(log_proxy), 0.5, dtype=float)
    importance_grid.ravel()[skeleton_indices] = importance

    elements_across = (
        float(config.hydraulic_elements_across_min)
        + (
            float(config.hydraulic_elements_across_max)
            - float(config.hydraulic_elements_across_min)
        )
        * importance
    )
    skeleton_target = np.clip(
        width_grid.ravel()[skeleton_indices] / elements_across,
        _interior_minimum_size(boundary, config),
        float(config.max_size_m),
    )
    skeleton_target_grid = np.full(shape, np.nan, dtype=float)
    skeleton_target_grid.ravel()[skeleton_indices] = skeleton_target
    smoothed_target, longitudinal_report = apply_gradation_limit(
        bathy.lon,
        bathy.lat,
        skeleton_target_grid,
        float(config.hydraulic_longitudinal_gradation),
        connectivity=8,
    )

    skeleton_distance, skeleton_source = _wet_graph_distance_and_labels(
        active,
        skeleton,
        dx_m,
        dy_m,
    )
    reachable = active & np.isfinite(skeleton_distance) & (skeleton_source >= 0)
    source_width = np.full(shape, np.nan, dtype=float)
    source_target = np.full(shape, np.nan, dtype=float)
    source_width[reachable] = width_grid.ravel()[
        skeleton_source[reachable]
    ]
    source_target[reachable] = smoothed_target.ravel()[
        skeleton_source[reachable]
    ]
    corridor = (
        reachable
        & coastal
        & (
            skeleton_distance
            <= float(config.hydraulic_corridor_width_factor) * source_width
        )
    )
    transverse_fraction = np.divide(
        fields.land_distance_m,
        fields.land_distance_m + skeleton_distance,
        out=np.zeros(shape, dtype=float),
        where=(
            corridor
            & np.isfinite(fields.land_distance_m)
            & np.isfinite(skeleton_distance)
            & ((fields.land_distance_m + skeleton_distance) > 1.0e-9)
        ),
    )
    blend = _quintic_smootherstep(
        np.clip(transverse_fraction, 0.0, 1.0)
    )
    hydraulic[corridor] = np.exp(
        (1.0 - blend[corridor])
        * np.log(fields.solid_background_m[corridor])
        + blend[corridor]
        * np.log(source_target[corridor])
    )

    return (
        hydraulic,
        skeleton,
        corridor,
        width_grid,
        importance_grid,
        storage_grid,
        cross_section_grid,
        {
            "method": "solid_only_raster_voronoi_medial_skeleton",
            "status": "complete",
            "open_boundary_used_as_bank": False,
            "detection": detection_report,
            "cross_sections": cross_section_report,
            "importance": {
                "method": "wet_distance_storage_ranking_proxy_over_cross_section_area",
                "interpretation": (
                    "ranking proxy for tidal exchange, not solved hydrodynamic velocity"
                ),
                "branch_and_loop_ambiguity": True,
                "log_proxy_quantile_10": float(q_low),
                "log_proxy_quantile_90": float(q_high),
                "minimum": float(np.min(importance)),
                "maximum": float(np.max(importance)),
            },
            "sizing": {
                "elements_across_min": float(
                    config.hydraulic_elements_across_min
                ),
                "elements_across_max": float(
                    config.hydraulic_elements_across_max
                ),
                "longitudinal_gradation": float(
                    config.hydraulic_longitudinal_gradation
                ),
                "corridor_width_factor": float(
                    config.hydraulic_corridor_width_factor
                ),
                "minimum_skeleton_target_m": float(
                    np.nanmin(smoothed_target[skeleton])
                ),
                "maximum_skeleton_target_m": float(
                    np.nanmax(smoothed_target[skeleton])
                ),
                "longitudinal_limiter": longitudinal_report,
            },
            "working_grid": {
                "dx_m": float(dx_m),
                "dy_m": float(dy_m),
                "cell_count": int(np.prod(shape)),
            },
            "skeleton_cell_count": int(np.count_nonzero(skeleton)),
            "corridor_cell_count": int(np.count_nonzero(corridor)),
            "minimum_width_m": float(np.nanmin(width_grid[skeleton])),
            "maximum_width_m": float(np.nanmax(width_grid[skeleton])),
            "minimum_bank_angle_deg": float(
                np.nanmin(bank_angle[skeleton])
            ),
            "maximum_ridge_balance_m": float(
                np.nanmax(ridge_balance[skeleton])
            ),
        },
    )


def _validate_config(config: SizeFieldConfig) -> None:
    positive = {
        "land_spacing_m": config.land_spacing_m,
        "open_spacing_m": config.open_spacing_m,
        "max_size_m": config.max_size_m,
        "slope_elements": config.slope_elements,
        "min_gradient": config.min_gradient,
        "hydraulic_elements_across_min": config.hydraulic_elements_across_min,
        "hydraulic_elements_across_max": config.hydraulic_elements_across_max,
        "hydraulic_max_width_m": config.hydraulic_max_width_m,
        "hydraulic_longitudinal_gradation": config.hydraulic_longitudinal_gradation,
        "hydraulic_corridor_width_factor": config.hydraulic_corridor_width_factor,
        "obc_transition_distance_m": config.obc_transition_distance_m,
        "cfl": config.cfl,
    }
    for name, value in positive.items():
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if config.interior_min_size_m is not None:
        interior_minimum = float(config.interior_min_size_m)
        if not np.isfinite(interior_minimum) or interior_minimum <= 0.0:
            raise ValueError(
                "interior_min_size_m must be finite and positive"
            )
        if interior_minimum > float(config.max_size_m):
            raise ValueError(
                "interior_min_size_m cannot exceed max_size_m"
            )
    if not np.isfinite(float(config.gradation)) or float(config.gradation) < 0.0:
        raise ValueError("gradation must be finite and non-negative")
    if not np.isfinite(float(config.coastal_distance_m)) or float(config.coastal_distance_m) < 0.0:
        raise ValueError("coastal_distance_m must be finite and non-negative")
    if (
        float(config.hydraulic_elements_across_max)
        < float(config.hydraulic_elements_across_min)
    ):
        raise ValueError(
            "hydraulic_elements_across_max must be at least hydraulic_elements_across_min"
        )
    angle = float(config.hydraulic_bank_angle_deg)
    if not np.isfinite(angle) or not (90.0 <= angle < 180.0):
        raise ValueError("hydraulic_bank_angle_deg must be in [90, 180)")
    if (
        not np.isfinite(float(config.obc_hold_distance_m))
        or float(config.obc_hold_distance_m) < 0.0
    ):
        raise ValueError("obc_hold_distance_m must be finite and non-negative")


def _minimum_size(boundary: BoundaryNodes, config: SizeFieldConfig) -> float:
    values = [float(config.land_spacing_m), float(config.open_spacing_m)]
    targets = np.asarray(boundary.target_spacing_m, dtype=float)
    values.extend(targets[np.isfinite(targets) & (targets > 0.0)].tolist())
    return float(min(values))


def _interior_minimum_size(
    boundary: BoundaryNodes,
    config: SizeFieldConfig,
) -> float:
    """Return the 2-D raster floor, independent of subgrid source chords."""

    if config.interior_min_size_m is None:
        return _minimum_size(boundary, config)
    return float(config.interior_min_size_m)


def _grid_mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    return array.copy()


def _boundary_coverage_report(
    bathy: BathymetryGrid,
    boundary: BoundaryNodes,
    coverage: np.ndarray,
) -> dict[str, Any]:
    count = int(len(boundary.lonlat))
    if count == 0:
        return {
            "boundary_node_count": 0,
            "uncovered_boundary_node_count": 0,
            "uncovered_boundary_node_indices": [],
        }
    interp = RegularGridInterpolator(
        (bathy.lat, bathy.lon),
        np.asarray(coverage, dtype=np.uint8),
        bounds_error=False,
        fill_value=0,
    )
    query = np.column_stack([boundary.lonlat[:, 1], boundary.lonlat[:, 0]])
    covered = np.asarray(interp(query), dtype=float) >= 1.0 - 1.0e-9
    missing = np.flatnonzero(~covered)
    return {
        "boundary_node_count": count,
        "uncovered_boundary_node_count": int(len(missing)),
        "uncovered_boundary_node_indices": [int(value) for value in missing[:100]],
    }


def model_domain_mask(bathy: BathymetryGrid, boundary: BoundaryNodes) -> np.ndarray:
    """Return the projected model-polygon mask at raster cell centres."""
    lon2, lat2 = np.meshgrid(bathy.lon, bathy.lat)
    xy = project_points(
        np.column_stack([lon2.ravel(), lat2.ravel()]),
        boundary.projection,
    )
    inside = contains_xy(boundary.domain_polygon_xy, xy[:, 0], xy[:, 1])
    return np.asarray(inside, dtype=bool).reshape(lon2.shape)


def _boundary_family_segments(
    boundary: BoundaryNodes,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Split boundary chains into pure open and non-open segment families.

    An edge is open only when both endpoints are open.  Every mixed edge stays
    in the solid family, so it and the first/last pure-open edge meet at the
    contract's actual hard landfall anchor rather than at a synthetic midpoint.
    """
    xy = np.asarray(boundary.xy, dtype=float)
    targets = np.asarray(boundary.target_spacing_m, dtype=float)
    kinds = [str(value).strip().lower() for value in boundary.kinds]
    if len(xy) != len(targets) or len(kinds) != len(xy):
        raise ValueError("Boundary xy, kinds, and target_spacing_m lengths must match")
    if np.any(~np.isfinite(targets)) or np.any(targets <= 0.0):
        raise ValueError("Boundary targets must be finite and positive")
    families: dict[str, list[tuple[np.ndarray, np.ndarray, float, float]]] = {
        "open": [],
        "land": [],
    }
    chains = boundary.constraint_chains or ([list(range(len(xy)))] if len(xy) else [])
    for chain in chains:
        clean = [int(value) for value in chain if 0 <= int(value) < len(xy)]
        if len(clean) < 2:
            continue
        pairs = list(zip(clean[:-1], clean[1:]))
        if clean[-1] != clean[0]:
            pairs.append((clean[-1], clean[0]))
        for index_a, index_b in pairs:
            a = xy[index_a]
            b = xy[index_b]
            if index_a == index_b or float(np.linalg.norm(b - a)) <= 1.0e-9:
                continue
            target_a = float(targets[index_a])
            target_b = float(targets[index_b])
            family = (
                "open"
                if kinds[index_a] in OPEN_KINDS and kinds[index_b] in OPEN_KINDS
                else "land"
            )
            families[family].append((a, b, target_a, target_b))

    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for family, records in families.items():
        if records:
            result[family] = (
                np.asarray([record[0] for record in records], dtype=float),
                np.asarray([record[1] for record in records], dtype=float),
                np.asarray([record[2] for record in records], dtype=float),
                np.asarray([record[3] for record in records], dtype=float),
            )
        else:
            result[family] = (
                np.empty((0, 2), dtype=float),
                np.empty((0, 2), dtype=float),
                np.empty(0, dtype=float),
                np.empty(0, dtype=float),
            )
    return result


def _solid_segment_lineage(
    boundary: BoundaryNodes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return chain and cyclic arclength metadata aligned to solid segments."""
    xy = np.asarray(boundary.xy, dtype=float)
    kinds = [str(value).strip().lower() for value in boundary.kinds]
    chains = boundary.constraint_chains or (
        [list(range(len(xy)))] if len(xy) else []
    )
    chain_ids: list[int] = []
    arc_midpoints: list[float] = []
    chain_lengths: list[float] = []
    for chain_id, chain in enumerate(chains):
        clean = [int(value) for value in chain if 0 <= int(value) < len(xy)]
        if len(clean) < 2:
            continue
        pairs = list(zip(clean[:-1], clean[1:]))
        if clean[-1] != clean[0]:
            pairs.append((clean[-1], clean[0]))
        lengths = np.asarray(
            [
                float(np.linalg.norm(xy[index_b] - xy[index_a]))
                for index_a, index_b in pairs
            ],
            dtype=float,
        )
        total = float(np.sum(lengths))
        cumulative = 0.0
        for (index_a, index_b), length in zip(pairs, lengths):
            if index_a == index_b or length <= 1.0e-9:
                cumulative += float(length)
                continue
            is_open = (
                kinds[index_a] in OPEN_KINDS
                and kinds[index_b] in OPEN_KINDS
            )
            if not is_open:
                chain_ids.append(int(chain_id))
                arc_midpoints.append(float(cumulative + 0.5 * length))
                chain_lengths.append(total)
            cumulative += float(length)
    return (
        np.asarray(chain_ids, dtype=np.int32),
        np.asarray(arc_midpoints, dtype=float),
        np.asarray(chain_lengths, dtype=float),
    )


def _nearest_segment_details(
    query_xy: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest distance, target, segment index, and contact point."""
    query = np.asarray(query_xy, dtype=float)
    if not len(seg_a):
        return (
            np.full(len(query), np.inf, dtype=float),
            np.full(len(query), np.nan, dtype=float),
            np.full(len(query), -1, dtype=np.int32),
            np.full((len(query), 2), np.nan, dtype=float),
        )
    if not (
        len(seg_a) == len(seg_b) == len(target_a) == len(target_b)
    ):
        raise ValueError("Segment coordinate and target arrays must have equal lengths")
    geometries = [
        LineString([tuple(a), tuple(b)])
        for a, b in zip(np.asarray(seg_a, dtype=float), np.asarray(seg_b, dtype=float))
    ]
    tree = STRtree(geometries)
    nearest_distance = np.empty(len(query), dtype=float)
    nearest_target = np.empty(len(query), dtype=float)
    nearest_segment = np.empty(len(query), dtype=np.int32)
    nearest_contact = np.empty((len(query), 2), dtype=float)
    chunk_size = 100_000
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        indices, distance = tree.query_nearest(
            points(query[start:stop]),
            all_matches=False,
            return_distance=True,
        )
        input_index = np.asarray(indices[0], dtype=int)
        segment_index = np.asarray(indices[1], dtype=int)
        # query_nearest returns one row per input but does not promise ordering.
        order = np.argsort(input_index)
        segment_index = segment_index[order]
        distance = np.asarray(distance, dtype=float)[order]
        q = query[start:stop]
        a = np.asarray(seg_a, dtype=float)[segment_index]
        vector = np.asarray(seg_b, dtype=float)[segment_index] - a
        denominator = np.sum(vector * vector, axis=1)
        fraction = np.divide(
            np.sum((q - a) * vector, axis=1),
            denominator,
            out=np.zeros(len(q), dtype=float),
            where=denominator > 0.0,
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        nearest_distance[start:stop] = distance
        nearest_segment[start:stop] = segment_index.astype(np.int32)
        nearest_contact[start:stop] = a + fraction[:, None] * vector
        nearest_target[start:stop] = (
            (1.0 - fraction) * np.asarray(target_a, dtype=float)[segment_index]
            + fraction * np.asarray(target_b, dtype=float)[segment_index]
        )
    return (
        nearest_distance,
        nearest_target,
        nearest_segment,
        nearest_contact,
    )


def _closest_points_on_segments(
    query_xy: np.ndarray,
    segment_index: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project each query point onto its indexed line segment."""
    query = np.asarray(query_xy, dtype=float)
    index = np.asarray(segment_index, dtype=int)
    if len(query) != len(index):
        raise ValueError("query_xy and segment_index lengths must match")
    if np.any(index < 0) or np.any(index >= len(seg_a)):
        raise ValueError("segment_index contains an invalid solid-boundary segment")
    a = np.asarray(seg_a, dtype=float)[index]
    vector = np.asarray(seg_b, dtype=float)[index] - a
    denominator = np.sum(vector * vector, axis=1)
    fraction = np.divide(
        np.sum((query - a) * vector, axis=1),
        denominator,
        out=np.zeros(len(query), dtype=float),
        where=denominator > 0.0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    contact = a + fraction[:, None] * vector
    distance = np.linalg.norm(contact - query, axis=1)
    return distance, contact


def _detect_solid_medial_skeleton(
    query_xy: np.ndarray,
    evaluation_mask: np.ndarray,
    nearest_segment: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    segment_chain_id: np.ndarray,
    segment_arc_mid_m: np.ndarray,
    segment_chain_length_m: np.ndarray,
    config: SizeFieldConfig,
    dx_m: float,
    dy_m: float,
    minimum_size_m: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Detect raster Voronoi ridges with geometrically opposing solid contacts."""
    shape = evaluation_mask.shape
    rows, cols = shape
    best_balance = np.full(shape, np.inf, dtype=float)
    best_width = np.full(shape, np.nan, dtype=float)
    best_a = np.full(shape, -1, dtype=np.int32)
    best_b = np.full(shape, -1, dtype=np.int32)
    best_angle = np.full(shape, np.nan, dtype=float)
    grid_diagonal = float(np.hypot(dx_m, dy_m))
    ridge_tolerance = 1.75 * grid_diagonal
    minimum_width = max(2.5 * max(dx_m, dy_m), 2.0 * minimum_size_m)
    maximum_width = float(config.hydraulic_max_width_m)
    angle_threshold = float(config.hydraulic_bank_angle_deg)
    nonlocal_arc_separation = max(
        1_500.0,
        8.0 * max(dx_m, dy_m),
    )
    neighbor_offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    comparisons = 0
    accepted_pairs = 0

    for dj, di in neighbor_offsets:
        current_j = slice(max(0, -dj), min(rows, rows - dj))
        current_i = slice(max(0, -di), min(cols, cols - di))
        neighbor_j = slice(current_j.start + dj, current_j.stop + dj)
        neighbor_i = slice(current_i.start + di, current_i.stop + di)
        current_mask = evaluation_mask[current_j, current_i]
        neighbor_mask = evaluation_mask[neighbor_j, neighbor_i]
        current_label = nearest_segment[current_j, current_i]
        neighbor_label = nearest_segment[neighbor_j, neighbor_i]
        pair_mask = (
            current_mask
            & neighbor_mask
            & (current_label >= 0)
            & (neighbor_label >= 0)
            & (current_label != neighbor_label)
        )
        if not np.any(pair_mask):
            continue
        local_positions = np.flatnonzero(pair_mask.ravel())
        comparisons += int(len(local_positions))
        query = query_xy[current_j, current_i].reshape((-1, 2))[
            local_positions
        ]
        segment_a = current_label.ravel()[local_positions]
        segment_b = neighbor_label.ravel()[local_positions]
        distance_a, contact_a = _closest_points_on_segments(
            query,
            segment_a,
            seg_a,
            seg_b,
        )
        distance_b, contact_b = _closest_points_on_segments(
            query,
            segment_b,
            seg_a,
            seg_b,
        )
        vector_a = contact_a - query
        vector_b = contact_b - query
        denominator = np.maximum(distance_a * distance_b, 1.0e-12)
        cosine = np.clip(
            np.sum(vector_a * vector_b, axis=1) / denominator,
            -1.0,
            1.0,
        )
        angle = np.degrees(np.arccos(cosine))
        balance = np.abs(distance_a - distance_b)
        width = distance_a + distance_b
        contact_span = np.linalg.norm(contact_b - contact_a, axis=1)
        chain_a = segment_chain_id[segment_a]
        chain_b = segment_chain_id[segment_b]
        arc_difference = np.abs(
            segment_arc_mid_m[segment_a]
            - segment_arc_mid_m[segment_b]
        )
        same_chain = chain_a == chain_b
        cyclic_separation = np.minimum(
            arc_difference,
            np.maximum(
                segment_chain_length_m[segment_a] - arc_difference,
                0.0,
            ),
        )
        nonlocal_contacts = (~same_chain) | (
            cyclic_separation >= nonlocal_arc_separation
        )
        valid = (
            nonlocal_contacts
            & (angle >= angle_threshold)
            & (balance <= ridge_tolerance)
            & (width >= minimum_width)
            & (width <= maximum_width)
            & (contact_span >= 0.80 * width)
        )
        if not np.any(valid):
            continue
        accepted_pairs += int(np.count_nonzero(valid))
        current_flat = np.arange(
            evaluation_mask[current_j, current_i].size,
            dtype=int,
        )[local_positions[valid]]
        sub_balance = best_balance[current_j, current_i].ravel()
        improve = balance[valid] < sub_balance[current_flat]
        if not np.any(improve):
            continue
        target_positions = current_flat[improve]
        sub_balance[target_positions] = balance[valid][improve]
        best_balance[current_j, current_i] = sub_balance.reshape(
            best_balance[current_j, current_i].shape
        )
        for destination, values in (
            (best_width, width[valid][improve]),
            (best_a, segment_a[valid][improve]),
            (best_b, segment_b[valid][improve]),
            (best_angle, angle[valid][improve]),
        ):
            sub = destination[current_j, current_i].ravel()
            sub[target_positions] = values
            destination[current_j, current_i] = sub.reshape(
                destination[current_j, current_i].shape
            )

    candidate = np.isfinite(best_balance) & evaluation_mask
    local_minimum = minimum_filter(
        np.where(candidate, best_balance, np.inf),
        size=3,
        mode="constant",
        cval=np.inf,
    )
    skeleton = candidate & (
        best_balance <= local_minimum + 0.25 * grid_diagonal
    )
    skeleton = _remove_small_skeleton_components(skeleton, minimum_cells=3)
    best_width[~skeleton] = np.nan
    best_a[~skeleton] = -1
    best_b[~skeleton] = -1
    best_angle[~skeleton] = np.nan
    best_balance[~skeleton] = np.nan
    return (
        skeleton,
        best_width,
        best_a,
        best_b,
        best_angle,
        best_balance,
        {
            "ridge_method": "nearest_solid_segment_label_discontinuity",
            "neighbor_comparison_count": int(comparisons),
            "accepted_opposing_pair_count": int(accepted_pairs),
            "prevalidation_skeleton_cell_count": int(
                np.count_nonzero(skeleton)
            ),
            "minimum_resolved_width_m": float(minimum_width),
            "maximum_width_m": float(maximum_width),
            "minimum_bank_angle_deg": float(angle_threshold),
            "same_chain_nonlocal_arc_separation_m": float(
                nonlocal_arc_separation
            ),
            "ridge_balance_tolerance_m": float(ridge_tolerance),
            "minimum_component_cells": 3,
        },
    )


def _remove_small_skeleton_components(
    skeleton: np.ndarray,
    *,
    minimum_cells: int,
) -> np.ndarray:
    """Remove isolated raster-ridge fragments without altering larger branches."""
    mask = np.asarray(skeleton, dtype=bool)
    if not np.any(mask):
        return mask.copy()
    labels, count = connected_components(
        mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= max(int(minimum_cells), 1)
    keep[0] = False
    return keep[labels]


def _wet_graph_distance_and_labels(
    wet_mask: np.ndarray,
    source_mask: np.ndarray,
    dx_m: float,
    dy_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eight-neighbour wet-path distance and source flat index."""
    wet = np.asarray(wet_mask, dtype=bool)
    sources = np.asarray(source_mask, dtype=bool) & wet
    if wet.shape != sources.shape:
        raise ValueError("wet_mask and source_mask shapes must match")
    rows, cols = wet.shape
    distance = np.full(wet.shape, np.inf, dtype=float)
    source = np.full(wet.shape, -1, dtype=np.int64)
    heap: list[tuple[float, int, int, int]] = []
    for j, i in zip(*np.nonzero(sources)):
        flat = int(j * cols + i)
        distance[j, i] = 0.0
        source[j, i] = flat
        heapq.heappush(heap, (0.0, int(j), int(i), flat))
    if not heap:
        return distance, source
    diagonal = float(np.hypot(dx_m, dy_m))
    neighbors = (
        (-1, 0, float(dy_m)),
        (1, 0, float(dy_m)),
        (0, -1, float(dx_m)),
        (0, 1, float(dx_m)),
        (-1, -1, diagonal),
        (-1, 1, diagonal),
        (1, -1, diagonal),
        (1, 1, diagonal),
    )
    while heap:
        value, j, i, label = heapq.heappop(heap)
        if value > distance[j, i] + 1.0e-9:
            continue
        for dj, di, step in neighbors:
            jj = j + dj
            ii = i + di
            if (
                jj < 0
                or jj >= rows
                or ii < 0
                or ii >= cols
                or not wet[jj, ii]
            ):
                continue
            candidate = value + step
            if candidate < distance[jj, ii] - 1.0e-9:
                distance[jj, ii] = candidate
                source[jj, ii] = label
                heapq.heappush(
                    heap,
                    (float(candidate), int(jj), int(ii), int(label)),
                )
    return distance, source


def _metric_cell_area(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Return regular-grid cell-centre quadrature area in square metres."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon_width = (
        np.abs(np.gradient(lon))
        if len(lon) > 1
        else np.ones(1, dtype=float)
    )
    lat_width = (
        np.abs(np.gradient(lat))
        if len(lat) > 1
        else np.ones(1, dtype=float)
    )
    dx = (
        lon_width[None, :]
        * np.pi
        / 180.0
        * EARTH_RADIUS_M
        * np.maximum(np.cos(np.radians(lat))[:, None], 1.0e-6)
    )
    dy = lat_width[:, None] * np.pi / 180.0 * EARTH_RADIUS_M
    return dx * dy


def _cumulative_storage_proxy(
    wet_mask: np.ndarray,
    wet_obc_distance: np.ndarray,
    cell_area_m2: np.ndarray,
) -> np.ndarray:
    """Approximate landward storage rank; branches and loops remain ambiguous."""
    wet = np.asarray(wet_mask, dtype=bool)
    distance = np.asarray(wet_obc_distance, dtype=float)
    area = np.asarray(cell_area_m2, dtype=float)
    storage = np.full(wet.shape, np.nan, dtype=float)
    reachable = wet & np.isfinite(distance)
    if not np.any(reachable):
        storage[wet] = float(np.sum(area[wet]))
        return storage
    indices = np.flatnonzero(reachable.ravel())
    order = np.argsort(distance.ravel()[indices], kind="mergesort")[::-1]
    ranked = indices[order]
    cumulative = np.cumsum(area.ravel()[ranked])
    storage.ravel()[ranked] = cumulative
    return storage


def _quintic_smootherstep(value: np.ndarray) -> np.ndarray:
    """Return P(v)=6v^5-15v^4+10v^3 for clipped v."""
    v = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return 6.0 * v**5 - 15.0 * v**4 + 10.0 * v**3


def _required_obc_transition_distance(
    open_target_m: np.ndarray,
    nearshore_target_m: np.ndarray,
    gradation: float,
) -> float:
    """Estimate the log-transfer length required by the configured gradation."""
    open_target = np.asarray(open_target_m, dtype=float)
    nearshore = np.asarray(nearshore_target_m, dtype=float)
    valid = (
        np.isfinite(open_target)
        & np.isfinite(nearshore)
        & (open_target > 0.0)
        & (nearshore > 0.0)
    )
    if not np.any(valid):
        return 0.0
    log_ratio = np.log(nearshore[valid] / open_target[valid])
    if np.max(np.abs(log_ratio), initial=0.0) <= 1.0e-12:
        return 0.0
    if gradation <= 0.0:
        raise ValueError(
            "A nonzero gradation is required when open and nearshore targets differ"
        )
    maximum_numerator = 0.0
    for xi in np.linspace(0.0, 1.0, 65):
        alpha = 6.0 * xi**5 - 15.0 * xi**4 + 10.0 * xi**3
        derivative = 30.0 * xi**2 * (1.0 - xi) ** 2
        value = (
            open_target[valid]
            * np.exp(alpha * log_ratio)
            * np.abs(log_ratio)
            * derivative
        )
        maximum_numerator = max(
            maximum_numerator,
            float(np.max(value, initial=0.0)),
        )
    return float(maximum_numerator / float(gradation))


def _grid_spacing_m(lon: np.ndarray, lat: np.ndarray) -> tuple[float, float]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lat0 = float(np.nanmean(lat))
    dx = (
        abs(float(np.nanmedian(np.diff(lon))))
        * np.pi
        / 180.0
        * EARTH_RADIUS_M
        * max(float(np.cos(np.radians(lat0))), 1.0e-6)
        if len(lon) > 1
        else 1.0
    )
    dy = (
        abs(float(np.nanmedian(np.diff(lat))))
        * np.pi
        / 180.0
        * EARTH_RADIUS_M
        if len(lat) > 1
        else 1.0
    )
    return max(dx, 1.0), max(dy, 1.0)


def bathymetric_gradient(bathy: BathymetryGrid) -> np.ndarray:
    """Estimate positive-down depth-gradient magnitude in m/m."""
    depth = np.asarray(bathy.depth, dtype=float)
    dx_m, dy_m = _grid_spacing_m(bathy.lon, bathy.lat)
    dz_dy, dz_dx = np.gradient(depth, dy_m, dx_m)
    return np.hypot(dz_dx, dz_dy)


def cfl_size_report(
    depth: np.ndarray,
    size: np.ndarray,
    config: SizeFieldConfig,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return diagnostic-only external-mode CFL information."""
    safe_depth = np.maximum(np.asarray(depth, dtype=float), 1.0)
    wave_speed = np.sqrt(9.807 * safe_depth)
    stable_dt = float(config.cfl) * np.asarray(size, dtype=float) / np.maximum(wave_speed, 1.0e-12)
    active = np.isfinite(stable_dt)
    if mask is not None:
        active &= _grid_mask(mask, stable_dt.shape, "cfl mask")
    values = stable_dt[active]
    report: dict[str, Any] = {
        "mode": "diagnostic_only",
        "cfl_modifies_size": False,
        "cfl": float(config.cfl),
        "wave_speed_assumption": "external_mode_sqrt(g*max(depth,1)); no current velocity supplied",
        "target_timestep_s": config.target_timestep_s,
        "recommended_timestep_s": float(np.nanmin(values)) if values.size else None,
    }
    if str(config.target_timestep_s).strip().lower() != "auto":
        target = float(config.target_timestep_s)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("target_timestep_s must be 'auto' or a finite positive number")
        report["target_timestep_s"] = target
        report["cells_below_target_timestep"] = int(np.count_nonzero(active & (stable_dt < target)))
    return report


def apply_gradation_limit(
    lon: np.ndarray,
    lat: np.ndarray,
    size: np.ndarray,
    gradation: float,
    *,
    connectivity: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a metric priority-queue lower envelope without coarsening cells."""
    if connectivity not in {4, 8}:
        raise ValueError("gradation connectivity must be 4 or 8")
    if not np.isfinite(float(gradation)) or float(gradation) < 0.0:
        raise ValueError("gradation must be finite and non-negative")
    out = np.asarray(size, dtype=float).copy()
    raw = out.copy()
    finite = np.isfinite(out)
    dx_m, dy_m = _grid_spacing_m(lon, lat)
    heap: list[tuple[float, int, int]] = []
    rows, cols = out.shape
    for j, i in zip(*np.nonzero(finite)):
        heapq.heappush(heap, (float(out[j, i]), int(j), int(i)))
    neighbors = [(0, -1, dx_m), (0, 1, dx_m), (-1, 0, dy_m), (1, 0, dy_m)]
    if connectivity == 8:
        diagonal = float(np.hypot(dx_m, dy_m))
        neighbors.extend(
            [(-1, -1, diagonal), (-1, 1, diagonal), (1, -1, diagonal), (1, 1, diagonal)]
        )
    relaxations = 0
    while heap:
        value, j, i = heapq.heappop(heap)
        if value > out[j, i] + 1.0e-9:
            continue
        for dj, di, distance in neighbors:
            jj = j + dj
            ii = i + di
            if jj < 0 or jj >= rows or ii < 0 or ii >= cols or not finite[jj, ii]:
                continue
            candidate = value + float(gradation) * distance
            if candidate < out[jj, ii]:
                out[jj, ii] = candidate
                relaxations += 1
                heapq.heappush(heap, (float(candidate), int(jj), int(ii)))
    return out, {
        "method": (
            "priority_queue_lower_envelope"
            if connectivity == 4
            else "priority_queue_8_neighbor_lower_envelope"
        ),
        "connectivity": int(connectivity),
        "gradation": float(gradation),
        "relaxations": int(relaxations),
        "max_reduction_m": float(np.nanmax(raw - out)) if np.any(finite) else 0.0,
        "dx_m": float(dx_m),
        "dy_m": float(dy_m),
        "max_neighbor_gradation": float(
            max_neighbor_gradation(out, dx_m, dy_m, connectivity=connectivity)
        ),
        "never_coarsened": bool(np.all(out[finite] <= raw[finite] + 1.0e-9)),
        "converged": True,
    }


def max_neighbor_gradation(
    size: np.ndarray,
    dx_m: float,
    dy_m: float,
    *,
    connectivity: int = 8,
) -> float:
    values: list[float] = []

    def append_finite_max(difference: np.ndarray, distance_m: float) -> None:
        finite = np.isfinite(difference)
        if np.any(finite):
            values.append(
                float(
                    np.max(
                        np.abs(difference[finite])
                        / max(float(distance_m), 1.0)
                    )
                )
            )

    if size.shape[1] > 1:
        append_finite_max(np.diff(size, axis=1), dx_m)
    if size.shape[0] > 1:
        append_finite_max(np.diff(size, axis=0), dy_m)
    if connectivity == 8 and size.shape[0] > 1 and size.shape[1] > 1:
        diagonal = max(float(np.hypot(dx_m, dy_m)), 1.0)
        append_finite_max(size[1:, 1:] - size[:-1, :-1], diagonal)
        append_finite_max(size[1:, :-1] - size[:-1, 1:], diagonal)
    return float(max(values)) if values else 0.0


def estimate_node_budget(
    lon: np.ndarray,
    lat: np.ndarray,
    size_m: np.ndarray,
    *,
    coverage_mask: np.ndarray | None = None,
    domain_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate triangular-lattice node demand by metric-cell quadrature."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    size = np.asarray(size_m, dtype=float)
    expected_shape = (len(lat), len(lon))
    if size.shape != expected_shape:
        raise ValueError(f"size_m must have shape {expected_shape}; got {size.shape}")
    lon_width = np.abs(np.gradient(lon)) if len(lon) > 1 else np.ones(1, dtype=float)
    lat_width = np.abs(np.gradient(lat)) if len(lat) > 1 else np.ones(1, dtype=float)
    dx = (
        lon_width[None, :]
        * np.pi
        / 180.0
        * EARTH_RADIUS_M
        * np.maximum(np.cos(np.radians(lat))[:, None], 1.0e-6)
    )
    dy = lat_width[:, None] * np.pi / 180.0 * EARTH_RADIUS_M
    cell_area = dx * dy
    active = np.isfinite(size) & (size > 0.0)
    if coverage_mask is not None:
        active &= _grid_mask(coverage_mask, expected_shape, "coverage_mask")
    domain_limited = domain_mask is not None
    if domain_mask is not None:
        active &= _grid_mask(domain_mask, expected_shape, "domain_mask")
    node_density = np.zeros(expected_shape, dtype=float)
    node_density[active] = 2.0 / (
        np.sqrt(3.0) * np.maximum(size[active], 1.0e-9) ** 2
    )
    estimate = float(np.sum(cell_area * node_density))
    return {
        "method": "triangular_lattice_metric_cell_quadrature",
        "estimated_interior_node_count_float": estimate,
        "estimated_interior_node_count": int(np.ceil(estimate)),
        "active_cell_count": int(np.count_nonzero(active)),
        "active_area_m2": float(np.sum(cell_area[active])),
        "domain_mask_applied": bool(domain_limited),
        "interpretation": (
            "rectangular-coverage upper estimate"
            if not domain_limited
            else "domain-masked estimate"
        ),
    }


def boundary_front_seed_points(
    boundary: BoundaryNodes,
    *,
    offset_factor: float = 0.65,
    minimum_boundary_clearance_factor: float = 0.20,
    minimum_seed_separation_factor: float = 0.35,
    include_hard_anchors: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create inward segment-front and hard-anchor-bisector seed candidates."""
    if offset_factor <= 0.0:
        raise ValueError("offset_factor must be positive")
    domain = boundary.domain_polygon_xy
    candidates: list[tuple[int, np.ndarray, float, str]] = []
    chains = boundary.constraint_chains or (
        [list(range(len(boundary.xy)))] if len(boundary.xy) else []
    )
    hard = np.asarray(
        boundary.hard_anchor_mask
        if boundary.hard_anchor_mask is not None
        else np.zeros(len(boundary.xy), dtype=bool),
        dtype=bool,
    )
    skipped_outside = 0
    skipped_clearance = 0

    def add_interior_candidate(
        origin: np.ndarray,
        directions: list[np.ndarray],
        target: float,
        priority: int,
        kind: str,
    ) -> None:
        nonlocal skipped_outside, skipped_clearance
        for direction in directions:
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-12:
                continue
            candidate = origin + float(offset_factor) * target * direction / norm
            point = Point(float(candidate[0]), float(candidate[1]))
            if not domain.contains(point):
                continue
            clearance = float(domain.boundary.distance(point))
            if clearance < float(minimum_boundary_clearance_factor) * target:
                skipped_clearance += 1
                continue
            candidates.append((priority, candidate, target, kind))
            return
        skipped_outside += 1

    for chain in chains:
        clean = [int(value) for value in chain if 0 <= int(value) < len(boundary.xy)]
        if len(clean) < 2:
            continue
        pairs = list(zip(clean, clean[1:]))
        if clean[-1] != clean[0]:
            pairs.append((clean[-1], clean[0]))
        for index_a, index_b in pairs:
            a = np.asarray(boundary.xy[index_a], dtype=float)
            b = np.asarray(boundary.xy[index_b], dtype=float)
            tangent = b - a
            if np.linalg.norm(tangent) <= 1.0e-12:
                continue
            target = 0.5 * (
                float(boundary.target_spacing_m[index_a])
                + float(boundary.target_spacing_m[index_b])
            )
            midpoint = 0.5 * (a + b)
            normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
            add_interior_candidate(midpoint, [normal, -normal], target, 1, "segment_front")

        if include_hard_anchors:
            for position, index in enumerate(clean):
                if index >= len(hard) or not hard[index]:
                    continue
                previous = clean[position - 1]
                following = clean[(position + 1) % len(clean)]
                origin = np.asarray(boundary.xy[index], dtype=float)
                previous_vector = np.asarray(boundary.xy[previous], dtype=float) - origin
                following_vector = np.asarray(boundary.xy[following], dtype=float) - origin
                previous_norm = max(float(np.linalg.norm(previous_vector)), 1.0e-12)
                following_norm = max(float(np.linalg.norm(following_vector)), 1.0e-12)
                bisector = (
                    previous_vector / previous_norm + following_vector / following_norm
                )
                target = float(boundary.target_spacing_m[index])
                add_interior_candidate(
                    origin,
                    [bisector, -bisector],
                    target,
                    0,
                    "hard_anchor_bisector",
                )

    accepted: list[np.ndarray] = []
    accepted_targets: list[float] = []
    accepted_kinds: list[str] = []
    rejected_separation = 0
    for _, point, target, kind in sorted(
        candidates,
        key=lambda value: (value[0], value[1][1], value[1][0]),
    ):
        if accepted:
            accepted_points = np.asarray(accepted, dtype=float)
            distances = np.linalg.norm(accepted_points - point[None, :], axis=1)
            thresholds = float(minimum_seed_separation_factor) * np.minimum(
                np.asarray(accepted_targets, dtype=float),
                target,
            )
            if np.any(distances < thresholds):
                rejected_separation += 1
                continue
        accepted.append(point)
        accepted_targets.append(float(target))
        accepted_kinds.append(kind)
    seed_points = (
        np.asarray(accepted, dtype=float).reshape((-1, 2))
        if accepted
        else np.empty((0, 2), dtype=float)
    )
    return seed_points, {
        "method": "segment_normal_and_hard_anchor_bisector",
        "candidate_count": int(len(candidates)),
        "accepted_count": int(len(seed_points)),
        "segment_front_count": int(
            sum(kind == "segment_front" for kind in accepted_kinds)
        ),
        "hard_anchor_bisector_count": int(
            sum(kind == "hard_anchor_bisector" for kind in accepted_kinds)
        ),
        "skipped_outside_count": int(skipped_outside),
        "skipped_clearance_count": int(skipped_clearance),
        "rejected_seed_separation_count": int(rejected_separation),
        "offset_factor": float(offset_factor),
        "minimum_boundary_clearance_factor": float(
            minimum_boundary_clearance_factor
        ),
        "minimum_seed_separation_factor": float(
            minimum_seed_separation_factor
        ),
    }


def write_size_field(
    size_field: SizeField,
    nc_path: str | Path,
    png_path: str | Path,
) -> tuple[Path, Path, Path]:
    """Write the v4 NetCDF, combined map, and six-panel component map."""
    nc_path = Path(nc_path)
    png_path = Path(png_path)
    components_path = png_path.with_name("size_field_components.png")
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    variables: dict[str, Any] = {
        "mesh_size_m": (("lat", "lon"), size_field.size),
        "raw_mesh_size_m": (("lat", "lon"), size_field.raw_size),
        "depth_m": (("lat", "lon"), size_field.depth),
        "slope": (("lat", "lon"), size_field.slope),
        "solid_boundary_background_mesh_size_m": (
            ("lat", "lon"),
            size_field.boundary_size,
        ),
        "bathymetry_slope_mesh_size_m": (("lat", "lon"), size_field.slope_size),
        "hydraulic_corridor_mesh_size_m": (
            ("lat", "lon"),
            size_field.hydraulic_size,
        ),
        "nearshore_mesh_size_m": (("lat", "lon"), size_field.nearshore_size),
        "obc_transition_mesh_size_m": (
            ("lat", "lon"),
            size_field.obc_transition_size,
        ),
        "open_boundary_euclidean_distance_m": (
            ("lat", "lon"),
            size_field.open_boundary_distance_m,
        ),
        "wet_obc_distance_m": (
            ("lat", "lon"),
            size_field.wet_obc_distance_m,
        ),
        "wet_obc_source_target_m": (
            ("lat", "lon"),
            size_field.wet_obc_target_m,
        ),
        "land_boundary_distance_m": (
            ("lat", "lon"),
            size_field.land_boundary_distance_m,
        ),
        "obc_transition_fraction": (
            ("lat", "lon"),
            size_field.transition_fraction,
        ),
        "coastal_estuary_mask": (
            ("lat", "lon"),
            np.asarray(size_field.coastal_mask, dtype=np.uint8),
        ),
        "hydraulic_skeleton_mask": (
            ("lat", "lon"),
            np.asarray(size_field.hydraulic_skeleton_mask, dtype=np.uint8),
        ),
        "hydraulic_corridor_mask": (
            ("lat", "lon"),
            np.asarray(size_field.hydraulic_corridor_mask, dtype=np.uint8),
        ),
        "hydraulic_bank_width_m": (
            ("lat", "lon"),
            size_field.hydraulic_width_m,
        ),
        "hydraulic_importance": (
            ("lat", "lon"),
            size_field.hydraulic_importance,
        ),
        "hydraulic_storage_ranking_area_m2": (
            ("lat", "lon"),
            size_field.hydraulic_storage_area_m2,
        ),
        "hydraulic_cross_section_area_m2": (
            ("lat", "lon"),
            size_field.hydraulic_cross_section_area_m2,
        ),
        "size_field_coverage_mask": (
            ("lat", "lon"),
            np.asarray(size_field.coverage_mask, dtype=np.uint8),
        ),
        "model_domain_mask": (
            ("lat", "lon"),
            np.asarray(size_field.domain_mask, dtype=np.uint8),
        ),
        "nearshore_size_source_attribution": (
            ("lat", "lon"),
            np.asarray(size_field.source_attribution, dtype=np.int16),
        ),
    }
    schema_version = str(
        size_field.report.get("schema_version", "fvcom_size_field_v4")
    )
    sampling_interface = dict(
        size_field.report.get("sampling_interface") or {}
    )
    dataset = xr.Dataset(
        variables,
        coords={"lon": size_field.lon, "lat": size_field.lat},
        attrs={
            "schema_version": schema_version,
            "coverage_policy": "strict",
            "interior_min_size_m": float(
                size_field.report.get("configured_min_size_m", np.nan)
            ),
            "boundary_trace_min_size_m": float(
                size_field.report.get(
                    "configured_boundary_trace_min_size_m",
                    np.nan,
                )
            ),
            "sampling_interface_schema_version": str(
                sampling_interface.get(
                    "schema_version",
                    "not_configured",
                )
            ),
            "sampling_interface_shared_inactive_halo_used": int(
                bool(
                    sampling_interface.get(
                        "shared_inactive_halo_used",
                        False,
                    )
                )
            ),
            "sampling_interface_report_json": json.dumps(
                sampling_interface,
                sort_keys=True,
            ),
        },
    )
    dataset["nearshore_size_source_attribution"].attrs["codes"] = json.dumps(
        size_field.report.get("source_attribution_codes", {}),
        sort_keys=True,
    )
    dataset["hydraulic_storage_ranking_area_m2"].attrs["warning"] = (
        "wet-distance ranking proxy; branched and looped passage allocation is ambiguous"
    )
    dataset.to_netcdf(nc_path)

    positive = size_field.size[
        np.isfinite(size_field.size) & (size_field.size > 0.0)
    ]
    vmin = float(np.nanpercentile(positive, 1.0))
    vmax = float(np.nanpercentile(positive, 99.0))
    if vmax <= vmin:
        vmax = max(vmin * 1.01, vmin + 1.0)
    norm = LogNorm(vmin=max(vmin, 1.0e-6), vmax=vmax)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.pcolormesh(
        size_field.lon,
        size_field.lat,
        size_field.size,
        shading="auto",
        cmap="viridis",
        norm=norm,
    )
    fig.colorbar(image, ax=ax, label="mesh size (m)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("FVCOM combined hydraulic-skeleton target mesh size")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    panels = (
        ("Solid-boundary background $h_S$", size_field.boundary_size),
        ("Bathymetric-gradient target $h_G$", size_field.slope_size),
        ("Hydraulic corridor target $h_H$", size_field.hydraulic_size),
        ("Nearshore minimum $h_N$", size_field.nearshore_size),
        ("OBC hold / log transfer $h_T$", size_field.obc_transition_size),
        ("Final wet-domain gradated $h$", size_field.size),
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    last_image = None
    lon2, lat2 = np.meshgrid(size_field.lon, size_field.lat)
    for axis, (title, values) in zip(axes.ravel(), panels):
        last_image = axis.pcolormesh(
            size_field.lon,
            size_field.lat,
            values,
            shading="auto",
            cmap="viridis",
            norm=norm,
        )
        if "Hydraulic" in title or "Final" in title:
            skeleton_points = np.nonzero(size_field.hydraulic_skeleton_mask)
            if len(skeleton_points[0]):
                axis.scatter(
                    size_field.lon[skeleton_points[1]],
                    size_field.lat[skeleton_points[0]],
                    s=1.0,
                    c="magenta",
                    linewidths=0.0,
                    alpha=0.8,
                )
        if "OBC" in title or "Final" in title:
            finite_distance = np.isfinite(size_field.wet_obc_distance_m)
            if np.any(finite_distance):
                transition = size_field.report.get(
                    "open_boundary_transition",
                    {},
                )
                hold = transition.get("hold_distance_m")
                width = transition.get("effective_transition_distance_m")
                levels = []
                if hold is not None:
                    levels.append(float(hold))
                if hold is not None and width is not None:
                    levels.append(float(hold) + float(width))
                if levels:
                    axis.contour(
                        lon2,
                        lat2,
                        np.where(
                            finite_distance,
                            size_field.wet_obc_distance_m,
                            np.nan,
                        ),
                        levels=sorted(set(levels)),
                        colors=("white", "orange")[: len(set(levels))],
                        linewidths=0.8,
                    )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes.ravel().tolist(),
            label="mesh size (m), common logarithmic scale",
            shrink=0.88,
        )
    fig.suptitle(
        "FVCOM hydraulic-skeleton size-field components",
        fontsize=14,
    )
    fig.savefig(components_path, dpi=150)
    plt.close(fig)
    return nc_path, png_path, components_path
