"""Unified OceanMesh-style mesh-size field for FVCOM grids.

The production field has one path:

* blend exact open- and solid-boundary targets through a smooth logarithmic
  distance-ratio transition (or use the traditional solid-boundary distance
  field for a closed domain);
* evaluate feature-size, M2 wavelength, bathymetric-slope, and optional
  supplied-channel candidates only in the coastal/estuary mask;
* take the pointwise lower envelope, clip it, and apply an eight-neighbour
  lower-envelope gradation pass; and
* report CFL stability without allowing CFL to change the mesh size.

Channel extraction is deliberately outside this module.  A reusable upstream
workflow supplies projected flow-line geometry and ``SegOrder`` values through
``ChannelFlowline``.
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
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from shapely import contains_xy, points
from shapely.geometry import LineString, MultiLineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
import xarray as xr

from .bathymetry import BathymetryGrid
from .boundary import BoundaryNodes
from .projection import project_points


EARTH_RADIUS_M = 6_371_000.0
OPEN_KINDS = {"open", "open_boundary"}
SOURCE_CODES = {
    0: "uncovered",
    1: "open_land_background",
    2: "oceanmesh_feature",
    3: "m2_wavelength",
    4: "bathymetry_slope",
    5: "channel_stencil",
}


@dataclass(frozen=True)
class SizeFieldConfig:
    """Configuration for the single production size-field algorithm."""

    land_spacing_m: float = 50.0
    open_spacing_m: float = 3000.0
    max_size_m: float = 20_000.0
    gradation: float = 0.20
    slope_elements: float = 10.0
    min_gradient: float = 1.0e-5
    coastal_distance_m: float = 12_000.0
    feature_elements: float = 3.0
    wavelength_period_s: float = 44_714.0
    wavelength_elements: float = 20.0
    channel_reslope_angle_deg: float = 60.0
    channel_elements_per_depth: float = 1.0
    channel_min_size_m: float | None = None
    target_timestep_s: str | float = "auto"
    cfl: float = 0.5


@dataclass(frozen=True)
class ChannelFlowline:
    """A projected flow line supplied by the upstream flownet skill.

    ``geometry_xy`` must use the same projected metre coordinates as
    ``BoundaryNodes.xy``.  ``seg_order`` is used only to attribute overlapping
    channel stencils; it does not alter the mesh-size formula.
    """

    geometry_xy: BaseGeometry
    seg_order: int


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
    feature_size: np.ndarray
    wavelength_size: np.ndarray
    slope_size: np.ndarray
    channel_size: np.ndarray
    open_boundary_distance_m: np.ndarray
    land_boundary_distance_m: np.ndarray
    transition_fraction: np.ndarray
    coastal_mask: np.ndarray
    channel_seg_order: np.ndarray
    coverage_mask: np.ndarray
    domain_mask: np.ndarray
    source_attribution: np.ndarray

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Sample the field and reject every point outside explicit coverage."""
        lon, lat = np.broadcast_arrays(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        query = np.column_stack([lat.ravel(), lon.ravel()])
        size_interp = RegularGridInterpolator(
            (self.lat, self.lon),
            self.size,
            bounds_error=False,
            fill_value=np.nan,
        )
        coverage_interp = RegularGridInterpolator(
            (self.lat, self.lon),
            np.asarray(self.coverage_mask, dtype=np.uint8),
            method="nearest",
            bounds_error=False,
            fill_value=0,
        )
        sampled = np.asarray(size_interp(query), dtype=float)
        uncovered = ~np.isfinite(sampled)
        uncovered |= np.asarray(coverage_interp(query), dtype=float) < 0.5
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
    flowlines: Sequence[ChannelFlowline] | None = None,
    coverage_mask: np.ndarray | None = None,
    domain_mask: np.ndarray | None = None,
) -> SizeField:
    """Build the single FVCOM production size field."""
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

    (
        background,
        open_distance,
        land_distance,
        transition_fraction,
        background_report,
    ) = boundary_background_size(bathy, boundary, config)
    background = np.where(coverage, background, np.nan)

    coastal = active & (land_distance <= float(config.coastal_distance_m))
    slope = bathymetric_gradient(bathy)
    wet = active & (depth > 0.0)

    feature_size, feature_report = oceanmesh_feature_size(
        bathy,
        boundary,
        wet,
        background,
        config,
    )
    safe_depth = np.maximum(depth, 1.0)
    wavelength_size = (
        float(config.wavelength_period_s)
        * np.sqrt(9.807 * safe_depth)
        / float(config.wavelength_elements)
    )
    slope_size = (
        (2.0 * np.pi / float(config.slope_elements))
        * safe_depth
        / np.maximum(slope, float(config.min_gradient))
    )
    channel_size, channel_order, channel_report = channel_stencil_size(
        bathy,
        boundary,
        flowlines or (),
        config,
        evaluation_mask=coastal,
    )

    candidates = [
        np.asarray(background, dtype=float),
        np.where(coastal, feature_size, np.inf),
        np.where(coastal, wavelength_size, np.inf),
        np.where(coastal, slope_size, np.inf),
        np.where(coastal, channel_size, np.inf),
    ]
    raw = candidates[0].copy()
    source = np.where(coverage, 1, 0).astype(np.int16)
    for code, candidate in enumerate(candidates[1:], start=2):
        selected = coverage & np.isfinite(candidate) & (candidate < raw)
        raw[selected] = candidate[selected]
        source[selected] = code

    minimum = _minimum_size(boundary, config)
    raw[coverage] = np.clip(raw[coverage], minimum, float(config.max_size_m))
    raw[~coverage] = np.nan
    limited, gradation_report = apply_gradation_limit(
        bathy.lon,
        bathy.lat,
        raw,
        float(config.gradation),
        connectivity=8,
    )
    limited[~coverage] = np.nan
    if np.any(limited[coverage] > raw[coverage] + 1.0e-9):
        raise RuntimeError("Gradation limiter coarsened at least one covered cell")

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
        "schema_version": "fvcom_size_field_v3",
        "method": "unified_oceanmesh_coastal_lower_envelope",
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
        "background": background_report,
        "feature": feature_report,
        "channel": channel_report,
        "candidate_formulas": {
            "wavelength": "T*sqrt(g*max(depth,1))/N_w",
            "slope": "(2*pi/N_s)*max(depth,1)/max(abs(grad_b),epsilon)",
            "channel": "max(channel_min_size,depth/channel_elements_per_depth)",
        },
        "source_attribution_codes": {str(key): value for key, value in SOURCE_CODES.items()},
        "source_attribution_cell_counts": source_counts,
        "source_attribution_stage": "raw_pointwise_minimum_before_clip_and_gradation",
        "gradation": gradation_report,
        "cfl": cfl_report,
        "node_budget_estimate": budget,
        "min_size_m": float(np.nanmin(limited[coverage])),
        "max_size_m": float(np.nanmax(limited[coverage])),
        "configured_min_size_m": float(minimum),
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
        feature_size=np.where(coastal, feature_size, np.nan),
        wavelength_size=np.where(coastal, wavelength_size, np.nan),
        slope_size=np.where(coastal, slope_size, np.nan),
        channel_size=np.where(coastal & np.isfinite(channel_size), channel_size, np.nan),
        open_boundary_distance_m=open_distance,
        land_boundary_distance_m=land_distance,
        transition_fraction=transition_fraction,
        coastal_mask=coastal,
        channel_seg_order=np.where(coastal, channel_order, 0).astype(np.int32),
        coverage_mask=coverage,
        domain_mask=model_mask,
        source_attribution=source,
    )


def boundary_background_size(
    bathy: BathymetryGrid,
    boundary: BoundaryNodes,
    config: SizeFieldConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the open/land background field and its distance diagnostics.

    With both boundary families,

    ``phi=d_open/(d_open+d_land)``,
    ``S=3*phi**2-2*phi**3``, and
    ``h=exp((1-S)*log(h_open)+S*log(h_land_dynamic))``.

    At a coincident landfall ``phi`` is defined as one half.  A closed domain
    uses the traditional OceanMesh distance background
    ``h_land_dynamic + gradation*d_land``.
    """
    lon2, lat2 = np.meshgrid(bathy.lon, bathy.lat)
    query_xy = project_points(
        np.column_stack([lon2.ravel(), lat2.ravel()]),
        boundary.projection,
    )
    families = _boundary_family_segments(boundary)
    open_distance, open_target = _nearest_segment_target(
        query_xy,
        *families["open"],
    )
    land_distance, land_target = _nearest_segment_target(
        query_xy,
        *families["land"],
    )
    has_open = bool(len(families["open"][0]))
    has_land = bool(len(families["land"][0]))
    shape = lon2.shape
    transition = np.full(len(query_xy), np.nan, dtype=float)

    if has_open and has_land:
        land_dynamic = np.maximum(land_target, float(config.land_spacing_m))
        denominator = open_distance + land_distance
        transition = np.divide(
            open_distance,
            denominator,
            out=np.full_like(denominator, 0.5),
            where=denominator > 1.0e-9,
        )
        transition = np.clip(transition, 0.0, 1.0)
        smooth = 3.0 * transition**2 - 2.0 * transition**3
        background = np.exp(
            (1.0 - smooth) * np.log(open_target)
            + smooth * np.log(land_dynamic)
        )
        mode = "open_land_log_smoothstep"
    elif has_land:
        land_dynamic = np.maximum(land_target, float(config.land_spacing_m))
        background = land_dynamic + float(config.gradation) * land_distance
        mode = "closed_domain_land_distance"
    elif has_open:
        background = open_target + float(config.gradation) * open_distance
        mode = "degenerate_open_only_distance"
    else:
        background = np.full(len(query_xy), float(config.max_size_m), dtype=float)
        mode = "no_boundary_segments_maximum"

    background = np.clip(
        background,
        _minimum_size(boundary, config),
        float(config.max_size_m),
    )
    report = {
        "method": mode,
        "open_segment_count": int(len(families["open"][0])),
        "land_segment_count": int(len(families["land"][0])),
        "coincident_landfall_cell_count": int(
            np.count_nonzero((open_distance + land_distance) <= 1.0e-9)
        ),
        "land_dynamic_floor_m": float(config.land_spacing_m),
    }
    return (
        background.reshape(shape),
        open_distance.reshape(shape),
        land_distance.reshape(shape),
        transition.reshape(shape),
        report,
    )


def oceanmesh_feature_size(
    bathy: BathymetryGrid,
    boundary: BoundaryNodes,
    wet_mask: np.ndarray,
    fallback_size: np.ndarray,
    config: SizeFieldConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Approximate wet-feature width using OceanMesh medial-axis geometry.

    The wet-domain distance gradient identifies medial singularities.  Isolated
    singularities are pruned with the same four-neighbour continuity idea used
    by OceanMesh2D.  The candidate is ``2*(d_medial+d_wet_boundary)/N_f``.
    If fewer than thirteen continuous medial points survive, the deterministic
    fallback is the supplied land-distance background.
    """
    wet = _grid_mask(wet_mask, np.asarray(bathy.depth).shape, "wet_mask")
    fallback = np.asarray(fallback_size, dtype=float)
    if fallback.shape != wet.shape:
        raise ValueError(f"fallback_size must have shape {wet.shape}; got {fallback.shape}")
    dx_m, dy_m = _grid_spacing_m(bathy.lon, bathy.lat)
    padded = np.pad(wet, 1, mode="constant", constant_values=False)
    wet_distance = distance_transform_edt(padded, sampling=(dy_m, dx_m))[1:-1, 1:-1]
    grad_y, grad_x = np.gradient(wet_distance, dy_m, dx_m)
    grad_mag = np.hypot(grad_x, grad_y)
    support = max(dx_m, dy_m)
    medial_mask = wet & (wet_distance > 0.5 * support) & (grad_mag < 0.90)

    lon2, lat2 = np.meshgrid(bathy.lon, bathy.lat)
    query_xy = project_points(
        np.column_stack([lon2.ravel(), lat2.ravel()]),
        boundary.projection,
    )
    medial_xy = query_xy[medial_mask.ravel()]
    initial_count = int(len(medial_xy))
    if len(medial_xy) > 12:
        tree = cKDTree(medial_xy)
        distances = np.asarray(tree.query(medial_xy, k=4, workers=-1)[0], dtype=float)
        cutoff = 2.0 * np.sqrt(2.0) * support
        keep = (
            (distances[:, 1] <= cutoff)
            & (distances[:, 2] <= 2.0 * cutoff)
            & (distances[:, 3] <= 3.0 * cutoff)
        )
        medial_xy = medial_xy[keep]
    final_count = int(len(medial_xy))

    if final_count <= 12:
        result = fallback.copy()
        fallback_used = True
    else:
        distance_to_medial = cKDTree(medial_xy).query(query_xy, workers=-1)[0].reshape(wet.shape)
        width = distance_to_medial + wet_distance
        result = 2.0 * width / float(config.feature_elements)
        result[~wet] = np.inf
        fallback_used = False
    return result, {
        "method": "oceanmesh_distance_gradient_medial_axis",
        "formula": "2*(distance_to_medial+distance_to_wet_boundary)/feature_elements",
        "initial_medial_point_count": initial_count,
        "retained_medial_point_count": final_count,
        "fallback_used": fallback_used,
        "fallback_method": "open_land_background",
        "feature_elements": float(config.feature_elements),
        "grid_support_m": float(support),
    }


def channel_stencil_size(
    bathy: BathymetryGrid,
    boundary: BoundaryNodes,
    flowlines: Sequence[ChannelFlowline],
    config: SizeFieldConfig,
    *,
    evaluation_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rasterize supplied flow lines with the OceanMesh channel stencil."""
    shape = np.asarray(bathy.depth).shape
    candidate = np.full(shape, np.inf, dtype=float)
    attribution = np.zeros(shape, dtype=np.int32)
    if not flowlines:
        return candidate, attribution, {
            "method": "supplied_segorder_oceanmesh_stencil",
            "flowline_count": 0,
            "active_cell_count": 0,
            "seg_orders": [],
        }

    segments_by_order: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for flowline in flowlines:
        order = int(flowline.seg_order)
        if order < 1:
            raise ValueError("ChannelFlowline.seg_order must be a positive integer")
        segments = _geometry_segments(flowline.geometry_xy)
        if not segments:
            continue
        segments_by_order.setdefault(order, []).extend(segments)
    if not segments_by_order:
        return candidate, attribution, {
            "method": "supplied_segorder_oceanmesh_stencil",
            "flowline_count": int(len(flowlines)),
            "active_cell_count": 0,
            "seg_orders": [],
        }

    lon2, lat2 = np.meshgrid(bathy.lon, bathy.lat)
    evaluation = (
        np.ones(shape, dtype=bool)
        if evaluation_mask is None
        else _grid_mask(evaluation_mask, shape, "channel evaluation_mask")
    )
    evaluation_indices = np.flatnonzero(evaluation.ravel())
    query_xy = project_points(
        np.column_stack(
            [lon2.ravel()[evaluation_indices], lat2.ravel()[evaluation_indices]]
        ),
        boundary.projection,
    )
    depth = np.maximum(np.asarray(bathy.depth, dtype=float), 1.0)
    support = max(_grid_spacing_m(bathy.lon, bathy.lat))
    radius_grid = np.maximum(
        support,
        np.tan(np.radians(float(config.channel_reslope_angle_deg))) * depth,
    )
    radius = radius_grid.ravel()[evaluation_indices]
    channel_minimum = (
        float(config.land_spacing_m)
        if config.channel_min_size_m is None
        else float(config.channel_min_size_m)
    )
    target = np.maximum(
        channel_minimum,
        depth / float(config.channel_elements_per_depth),
    )
    union = np.zeros(len(query_xy), dtype=bool)
    evaluation_attribution = np.zeros(len(query_xy), dtype=np.int32)
    # Descending order means the highest SegOrder owns attribution wherever
    # stencils overlap.  Candidate size itself is independent of SegOrder.
    for order in sorted(segments_by_order, reverse=True):
        pairs = segments_by_order[order]
        seg_a = np.asarray([pair[0] for pair in pairs], dtype=float)
        seg_b = np.asarray([pair[1] for pair in pairs], dtype=float)
        distance, _ = _nearest_segment_target(
            query_xy,
            seg_a,
            seg_b,
            np.ones(len(seg_a), dtype=float),
            np.ones(len(seg_a), dtype=float),
        )
        inside = distance <= radius
        union |= inside
        evaluation_attribution[(evaluation_attribution == 0) & inside] = int(order)
    union_grid = np.zeros(shape, dtype=bool)
    union_grid.ravel()[evaluation_indices] = union
    attribution.ravel()[evaluation_indices] = evaluation_attribution
    candidate[union_grid] = target[union_grid]
    return candidate, attribution, {
        "method": "supplied_segorder_oceanmesh_stencil",
        "formula": {
            "radius": "max(grid_support,tan(reslope_angle)*depth)",
            "target": "max(channel_min_size,depth/channel_elements_per_depth)",
        },
        "flowline_count": int(len(flowlines)),
        "active_cell_count": int(np.count_nonzero(union)),
        "evaluated_cell_count": int(len(evaluation_indices)),
        "seg_orders": [int(value) for value in sorted(segments_by_order)],
        "overlap_attribution": "highest_segorder",
        "segorder_changes_size": False,
        "grid_support_m": float(support),
        "channel_min_size_m": float(channel_minimum),
    }


def _validate_config(config: SizeFieldConfig) -> None:
    positive = {
        "land_spacing_m": config.land_spacing_m,
        "open_spacing_m": config.open_spacing_m,
        "max_size_m": config.max_size_m,
        "slope_elements": config.slope_elements,
        "min_gradient": config.min_gradient,
        "feature_elements": config.feature_elements,
        "wavelength_period_s": config.wavelength_period_s,
        "wavelength_elements": config.wavelength_elements,
        "channel_elements_per_depth": config.channel_elements_per_depth,
        "cfl": config.cfl,
    }
    for name, value in positive.items():
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(float(config.gradation)) or float(config.gradation) < 0.0:
        raise ValueError("gradation must be finite and non-negative")
    if not np.isfinite(float(config.coastal_distance_m)) or float(config.coastal_distance_m) < 0.0:
        raise ValueError("coastal_distance_m must be finite and non-negative")
    angle = float(config.channel_reslope_angle_deg)
    if not np.isfinite(angle) or angle < 0.0 or angle >= 90.0:
        raise ValueError("channel_reslope_angle_deg must be in [0, 90)")
    if config.channel_min_size_m is not None and (
        not np.isfinite(float(config.channel_min_size_m))
        or float(config.channel_min_size_m) <= 0.0
    ):
        raise ValueError("channel_min_size_m must be finite and positive when supplied")


def _grid_mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    return array.copy()


def _minimum_size(boundary: BoundaryNodes, config: SizeFieldConfig) -> float:
    values = [float(config.land_spacing_m), float(config.open_spacing_m)]
    targets = np.asarray(boundary.target_spacing_m, dtype=float)
    values.extend(targets[np.isfinite(targets) & (targets > 0.0)].tolist())
    if config.channel_min_size_m is not None:
        values.append(float(config.channel_min_size_m))
    return float(min(values))


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


def _nearest_segment_target(
    query_xy: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact nearest-segment distance and interpolated target."""
    query = np.asarray(query_xy, dtype=float)
    if not len(seg_a):
        return (
            np.full(len(query), np.inf, dtype=float),
            np.full(len(query), np.nan, dtype=float),
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
        nearest_target[start:stop] = (
            (1.0 - fraction) * np.asarray(target_a, dtype=float)[segment_index]
            + fraction * np.asarray(target_b, dtype=float)[segment_index]
        )
    return nearest_distance, nearest_target


def _geometry_segments(geometry: BaseGeometry) -> list[tuple[np.ndarray, np.ndarray]]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        lines = [geometry]
    elif isinstance(geometry, MultiLineString):
        lines = list(geometry.geoms)
    else:
        raise TypeError("ChannelFlowline.geometry_xy must be a LineString or MultiLineString")
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for line in lines:
        coordinates = np.asarray(line.coords, dtype=float)
        for a, b in zip(coordinates[:-1], coordinates[1:]):
            if float(np.linalg.norm(b - a)) > 1.0e-9:
                segments.append((a, b))
    return segments


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
    if size.shape[1] > 1:
        values.append(float(np.nanmax(np.abs(np.diff(size, axis=1)) / max(dx_m, 1.0))))
    if size.shape[0] > 1:
        values.append(float(np.nanmax(np.abs(np.diff(size, axis=0)) / max(dy_m, 1.0))))
    if connectivity == 8 and size.shape[0] > 1 and size.shape[1] > 1:
        diagonal = max(float(np.hypot(dx_m, dy_m)), 1.0)
        values.append(
            float(np.nanmax(np.abs(size[1:, 1:] - size[:-1, :-1]) / diagonal))
        )
        values.append(
            float(np.nanmax(np.abs(size[1:, :-1] - size[:-1, 1:]) / diagonal))
        )
    return float(np.nanmax(values)) if values else 0.0


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
) -> tuple[Path, Path]:
    """Write the v3 size-field NetCDF and diagnostic map."""
    nc_path = Path(nc_path)
    png_path = Path(png_path)
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    variables: dict[str, Any] = {
        "mesh_size_m": (("lat", "lon"), size_field.size),
        "raw_mesh_size_m": (("lat", "lon"), size_field.raw_size),
        "depth_m": (("lat", "lon"), size_field.depth),
        "slope": (("lat", "lon"), size_field.slope),
        "background_mesh_size_m": (("lat", "lon"), size_field.boundary_size),
        "oceanmesh_feature_mesh_size_m": (("lat", "lon"), size_field.feature_size),
        "m2_wavelength_mesh_size_m": (("lat", "lon"), size_field.wavelength_size),
        "bathymetry_slope_mesh_size_m": (("lat", "lon"), size_field.slope_size),
        "channel_mesh_size_m": (("lat", "lon"), size_field.channel_size),
        "open_boundary_distance_m": (
            ("lat", "lon"),
            size_field.open_boundary_distance_m,
        ),
        "land_boundary_distance_m": (
            ("lat", "lon"),
            size_field.land_boundary_distance_m,
        ),
        "open_land_transition_fraction": (
            ("lat", "lon"),
            size_field.transition_fraction,
        ),
        "coastal_estuary_mask": (
            ("lat", "lon"),
            np.asarray(size_field.coastal_mask, dtype=np.uint8),
        ),
        "channel_seg_order": (
            ("lat", "lon"),
            np.asarray(size_field.channel_seg_order, dtype=np.int32),
        ),
        "size_field_coverage_mask": (
            ("lat", "lon"),
            np.asarray(size_field.coverage_mask, dtype=np.uint8),
        ),
        "model_domain_mask": (
            ("lat", "lon"),
            np.asarray(size_field.domain_mask, dtype=np.uint8),
        ),
        "raw_size_source_attribution": (
            ("lat", "lon"),
            np.asarray(size_field.source_attribution, dtype=np.int16),
        ),
    }
    schema_version = str(size_field.report.get("schema_version", "fvcom_size_field_v3"))
    dataset = xr.Dataset(
        variables,
        coords={"lon": size_field.lon, "lat": size_field.lat},
        attrs={"schema_version": schema_version, "coverage_policy": "strict"},
    )
    dataset["raw_size_source_attribution"].attrs["codes"] = json.dumps(
        size_field.report.get("source_attribution_codes", {}),
        sort_keys=True,
    )
    dataset["channel_seg_order"].attrs["meaning"] = (
        "highest supplied SegOrder where channel stencils overlap"
    )
    dataset.to_netcdf(nc_path)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.pcolormesh(
        size_field.lon,
        size_field.lat,
        size_field.size,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(image, ax=ax, label="mesh size (m)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("FVCOM unified target mesh size")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return nc_path, png_path
