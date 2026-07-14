"""OceanMesh/RPW-style mesh-size fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import heapq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from shapely import contains_xy
from shapely.geometry import Point

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
    size_field_profile: str = "v1"
    coverage_policy: str = "auto"
    junction_floor_m: float | None = None
    junction_transition_distance_m: float | None = None


@dataclass(frozen=True)
class SizeFieldSemantics:
    """Optional grid-aligned scientific priorities for the v2 size field.

    Every array must either be scalar or have the bathymetry grid shape.  Soft
    boundary, depth, and slope targets may be relaxed upward inside
    ``junction_mask``.  Channel, mission, CFL, and caller-supplied hard targets
    always retain priority over that relaxation.
    """

    junction_mask: np.ndarray | None = None
    junction_floor_m: float | np.ndarray | None = None
    channel_size_m: np.ndarray | None = None
    mission_size_m: np.ndarray | None = None
    hard_size_m: np.ndarray | None = None
    coverage_mask: np.ndarray | None = None
    domain_mask: np.ndarray | None = None


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
    soft_size: np.ndarray | None = None
    hard_size: np.ndarray | None = None
    junction_mask: np.ndarray | None = None
    coverage_mask: np.ndarray | None = None
    domain_mask: np.ndarray | None = None
    source_attribution: np.ndarray | None = None
    boundary_source_attribution: np.ndarray | None = None
    coverage_policy: str = "legacy-max"

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        lon, lat = np.broadcast_arrays(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        query = np.column_stack([lat.ravel(), lon.ravel()])
        if self.coverage_policy == "legacy-max":
            interp = RegularGridInterpolator(
                (self.lat, self.lon),
                self.size,
                bounds_error=False,
                fill_value=float(np.nanmax(self.size)),
            )
            return np.asarray(interp(query), dtype=float).reshape(lon.shape)
        interp = RegularGridInterpolator(
            (self.lat, self.lon),
            self.size,
            bounds_error=False,
            fill_value=np.nan,
        )
        sampled = np.asarray(interp(query), dtype=float)
        uncovered = ~np.isfinite(sampled)
        if self.coverage_mask is not None:
            coverage_interp = RegularGridInterpolator(
                (self.lat, self.lon),
                np.asarray(self.coverage_mask, dtype=np.uint8),
                method="nearest",
                bounds_error=False,
                fill_value=0,
            )
            uncovered |= np.asarray(coverage_interp(query), dtype=float) < 0.5
        if np.any(uncovered) and self.coverage_policy == "raise":
            first = int(np.flatnonzero(uncovered)[0])
            raise ValueError(
                "Size-field sampling requested outside explicit coverage: "
                f"{int(np.count_nonzero(uncovered))} point(s); first lon/lat="
                f"({float(lon.ravel()[first]):.8f}, {float(lat.ravel()[first]):.8f})"
            )
        if np.any(uncovered) and self.coverage_policy == "nearest":
            valid = np.isfinite(self.size)
            if self.coverage_mask is not None:
                valid &= np.asarray(self.coverage_mask, dtype=bool)
            if not np.any(valid):
                raise ValueError("Cannot apply nearest coverage policy: size field has no covered cells")
            grid_lon, grid_lat = np.meshgrid(self.lon, self.lat)
            scale = max(float(np.cos(np.radians(np.nanmean(self.lat)))), 1.0e-6)
            tree_points = np.column_stack([grid_lon[valid] * scale, grid_lat[valid]])
            lookup = cKDTree(tree_points).query(
                np.column_stack([lon.ravel()[uncovered] * scale, lat.ravel()[uncovered]]),
                workers=-1,
            )[1]
            sampled[uncovered] = np.asarray(self.size[valid], dtype=float)[np.asarray(lookup, dtype=int)]
        elif np.any(uncovered):
            sampled[uncovered] = np.nan
        return sampled.reshape(lon.shape)


def build_size_field(
    bathy: BathymetryGrid,
    boundary: BoundaryNodes,
    config: SizeFieldConfig,
    *,
    semantics: SizeFieldSemantics | Mapping[str, Any] | None = None,
) -> SizeField:
    """Build a size field from bathymetry, boundary distance, and gradation."""
    profile = _resolve_size_field_profile(config.size_field_profile)
    if profile == "adaptive-coastal-v2":
        return _build_size_field_v2(bathy, boundary, config, _coerce_semantics(semantics))
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


def _build_size_field_v2(
    bathy: BathymetryGrid,
    boundary: BoundaryNodes,
    config: SizeFieldConfig,
    semantics: SizeFieldSemantics,
) -> SizeField:
    """Build an opt-in, segment-aware field with explicit scientific priorities."""
    depth = np.asarray(bathy.depth, dtype=float)
    if depth.shape != (len(bathy.lat), len(bathy.lon)):
        raise ValueError("Bathymetry depth shape must be (lat, lon)")
    coverage_policy = _resolve_coverage_policy(config.coverage_policy, "adaptive-coastal-v2")
    coverage = np.isfinite(depth)
    semantic_coverage = _as_grid(semantics.coverage_mask, depth.shape, "coverage_mask", dtype=bool)
    if semantic_coverage is not None:
        coverage &= semantic_coverage
    if not np.any(coverage):
        raise ValueError("The v2 size field has no covered finite bathymetry cells")
    domain_mask = _as_grid(semantics.domain_mask, depth.shape, "domain_mask", dtype=bool)
    if domain_mask is None:
        domain_mask = model_domain_mask(bathy, boundary)
    boundary_coverage = _boundary_coverage_report(bathy, boundary, coverage)
    if coverage_policy == "raise" and boundary_coverage["uncovered_boundary_node_count"]:
        raise ValueError(
            "Adaptive v2 boundary is outside explicit size-field coverage: "
            f"{boundary_coverage['uncovered_boundary_node_count']} of "
            f"{boundary_coverage['boundary_node_count']} node(s)"
        )

    slope = bathymetric_gradient(bathy)
    adaptive_boundary = bool(config.adaptive_boundary or boundary.adaptive_resolution)
    requested_gradient_policy, effective_gradient_policy = _resolve_bathymetry_gradient_policy(config, adaptive_boundary)
    coastal_distance = coastal_boundary_distance(bathy, boundary)
    coastal_threshold = float(config.coastal_gradient_distance_m)
    if coastal_threshold < 0.0:
        raise ValueError("coastal_gradient_distance_m must be non-negative")
    coastal_mask = coastal_distance <= coastal_threshold

    soft = np.full(depth.shape, float(config.max_size_m), dtype=float)
    source = np.zeros(depth.shape, dtype=np.int16)
    if not adaptive_boundary:
        shelf = coverage & (depth <= config.shelf_depth_m) & (float(config.shelf_max_size_m) < soft)
        soft[shelf] = float(config.shelf_max_size_m)
        source[shelf] = 2
        nearshore = coverage & (depth <= config.nearshore_depth_m) & (float(config.nearshore_max_size_m) < soft)
        soft[nearshore] = float(config.nearshore_max_size_m)
        source[nearshore] = 2

    boundary_size, _, _, boundary_source = segment_boundary_distance_size(bathy, boundary, config)
    boundary_limited = coverage & np.isfinite(boundary_size) & (boundary_size < soft)
    soft[boundary_limited] = boundary_size[boundary_limited]
    source[boundary_limited] = 1

    topo_length = depth / np.maximum(slope, config.min_gradient)
    slope_candidate = (2.0 * np.pi / max(config.slope_elements, 1.0)) * topo_length
    depth_eligible = coverage & (depth > config.nearshore_depth_m)
    slope_size = np.where(depth_eligible, slope_candidate, np.nan)
    if effective_gradient_policy == "global":
        gradient_mask = depth_eligible
    elif effective_gradient_policy == "coastal":
        gradient_mask = depth_eligible & coastal_mask
    else:
        gradient_mask = np.zeros(depth.shape, dtype=bool)
    slope_limited = gradient_mask & np.isfinite(slope_candidate) & (slope_candidate < soft)
    soft[slope_limited] = slope_candidate[slope_limited]
    source[slope_limited] = 3

    semantic_junction_mask = _as_grid(semantics.junction_mask, depth.shape, "junction_mask", dtype=bool)
    auto_junction_mask, auto_junction_floor, auto_junction_report = land_open_junction_semantics(
        bathy,
        boundary,
        config,
    )
    junction_mask = auto_junction_mask if semantic_junction_mask is None else semantic_junction_mask
    junction_mask &= coverage
    semantic_floor = semantics.junction_floor_m
    if semantic_floor is None:
        semantic_floor = config.junction_floor_m
    if semantic_floor is None:
        semantic_floor = (
            auto_junction_floor
            if semantic_junction_mask is None and np.any(auto_junction_mask)
            else float(config.open_spacing_m)
        )
    junction_floor = _as_grid(semantic_floor, depth.shape, "junction_floor_m", dtype=float)
    if junction_floor is None:
        junction_floor = np.full(depth.shape, float(config.open_spacing_m), dtype=float)
    raised = junction_mask & np.isfinite(junction_floor) & (junction_floor > soft)
    soft[raised] = junction_floor[raised]
    source[raised] = 8

    hard = np.full(depth.shape, np.inf, dtype=float)
    hard_source = np.zeros(depth.shape, dtype=np.int16)
    for candidate_value, code, name in (
        (semantics.channel_size_m, 5, "channel_size_m"),
        (semantics.mission_size_m, 6, "mission_size_m"),
        (semantics.hard_size_m, 7, "hard_size_m"),
    ):
        candidate = _as_grid(candidate_value, depth.shape, name, dtype=float)
        if candidate is None:
            continue
        selected = coverage & np.isfinite(candidate) & (candidate < hard)
        hard[selected] = candidate[selected]
        hard_source[selected] = code

    cfl_report = cfl_size_report(depth, soft, config)
    if cfl_report["mode"] == "enforced":
        cfl_size = np.asarray(cfl_report["cfl_size_m"], dtype=float)
        selected = coverage & np.isfinite(cfl_size) & (cfl_size < hard)
        hard[selected] = cfl_size[selected]
        hard_source[selected] = 4

    raw = np.minimum(soft, hard)
    hard_wins = hard < soft
    source[hard_wins] = hard_source[hard_wins]
    minimum_candidates = [float(config.land_spacing_m), float(config.open_spacing_m)]
    explicit_boundary_targets = np.asarray(boundary.target_spacing_m, dtype=float)
    explicit_boundary_targets = explicit_boundary_targets[
        np.isfinite(explicit_boundary_targets) & (explicit_boundary_targets > 0.0)
    ]
    if explicit_boundary_targets.size:
        minimum_candidates.append(float(np.min(explicit_boundary_targets)))
    finite_hard = hard[np.isfinite(hard) & (hard > 0.0)]
    if finite_hard.size:
        minimum_candidates.append(float(np.min(finite_hard)))
    min_size = min(minimum_candidates)
    raw[coverage] = np.clip(raw[coverage], min_size, float(config.max_size_m))
    raw[~coverage] = np.nan
    limited, gradation_report = apply_gradation_limit(bathy.lon, bathy.lat, raw, config.gradation, connectivity=8)
    limited[~coverage] = np.nan
    floor_protected = junction_mask & ~hard_wins & np.isfinite(junction_floor)
    floor_conflict = floor_protected & (limited + 1.0e-9 < junction_floor)

    budget = estimate_node_budget(
        bathy.lon,
        bathy.lat,
        limited,
        coverage_mask=coverage,
        domain_mask=domain_mask,
    )
    source_codes = {
        "0": "maximum_size",
        "1": "boundary_segment",
        "2": "depth_band",
        "3": "bathymetry_slope",
        "4": "cfl_hard",
        "5": "channel_hard",
        "6": "mission_hard",
        "7": "semantic_hard",
        "8": "junction_floor",
    }
    source_counts = {
        label: int(np.count_nonzero(coverage & (source == int(code))))
        for code, label in source_codes.items()
    }
    report = {
        "schema_version": "fvcom_size_field_v2",
        "profile": "adaptive-coastal-v2",
        "method": "segment_lower_envelope_hard_soft_priority",
        "coverage": {
            "policy": coverage_policy,
            "covered_cell_count": int(np.count_nonzero(coverage)),
            "uncovered_cell_count": int(coverage.size - np.count_nonzero(coverage)),
            "covered_cell_fraction": float(np.count_nonzero(coverage) / max(coverage.size, 1)),
            **boundary_coverage,
        },
        "gradation": gradation_report,
        "cfl": {key: value for key, value in cfl_report.items() if key != "cfl_size_m"},
        "node_budget_estimate": budget,
        "source_attribution_codes": source_codes,
        "source_attribution_cell_counts": source_counts,
        "boundary_source_attribution_codes": {
            "0": "none",
            "1": "land",
            "2": "open",
            "3": "island",
            "4": "frame",
            "5": "kind_transition",
            "6": "other",
        },
        "junction": {
            "cell_count": int(np.count_nonzero(junction_mask)),
            "soft_cells_raised": int(np.count_nonzero(raised)),
            "hard_override_cell_count": int(np.count_nonzero(junction_mask & hard_wins)),
            "post_gradation_floor_conflict_count": int(np.count_nonzero(floor_conflict)),
            "configured_floor_m": None if config.junction_floor_m is None else float(config.junction_floor_m),
            "effective_floor_min_m": (
                float(np.nanmin(junction_floor[junction_mask])) if np.any(junction_mask) else None
            ),
            "effective_floor_max_m": (
                float(np.nanmax(junction_floor[junction_mask])) if np.any(junction_mask) else None
            ),
            "automatic": auto_junction_report,
        },
        "hard_priority": {
            "finite_cell_count": int(np.count_nonzero(np.isfinite(hard) & coverage)),
            "winning_cell_count": int(np.count_nonzero(hard_wins & coverage)),
        },
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
            "slope_limited_count": int(np.count_nonzero(slope_limited)),
            "offshore_slope_limited_count": int(np.count_nonzero(slope_limited & ~coastal_mask)),
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
        soft_size=soft,
        hard_size=hard,
        junction_mask=junction_mask,
        coverage_mask=coverage,
        domain_mask=domain_mask,
        source_attribution=source,
        boundary_source_attribution=boundary_source,
        coverage_policy=coverage_policy,
    )


def _resolve_size_field_profile(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"v1", "legacy", "adaptive-coastal-v1", "fvcom_size_field_v1"}:
        return "v1"
    if normalized in {"v2", "adaptive-coastal-v2", "fvcom_size_field_v2"}:
        return "adaptive-coastal-v2"
    raise ValueError("size_field_profile must be v1 or adaptive-coastal-v2")


def _resolve_coverage_policy(value: str, profile: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return "raise" if profile == "adaptive-coastal-v2" else "legacy-max"
    allowed = {"raise", "nan", "nearest", "legacy-max"}
    if normalized not in allowed:
        raise ValueError(f"coverage_policy must be auto or one of {sorted(allowed)}")
    return normalized


def _coerce_semantics(value: SizeFieldSemantics | Mapping[str, Any] | None) -> SizeFieldSemantics:
    if value is None:
        return SizeFieldSemantics()
    if isinstance(value, SizeFieldSemantics):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("size-field semantics must be SizeFieldSemantics, a mapping, or None")
    allowed = set(SizeFieldSemantics.__dataclass_fields__)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown size-field semantic keys: {unknown}")
    return SizeFieldSemantics(**{key: value[key] for key in allowed if key in value})


def _as_grid(
    value: Any,
    shape: tuple[int, int],
    name: str,
    *,
    dtype: type,
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return np.full(shape, array.item(), dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"{name} must be scalar or have grid shape {shape}; got {array.shape}")
    return array.copy()


def _boundary_coverage_report(
    bathy: BathymetryGrid,
    boundary: BoundaryNodes,
    coverage: np.ndarray,
) -> dict[str, Any]:
    count = int(len(boundary.lonlat))
    if count == 0:
        return {"boundary_node_count": 0, "uncovered_boundary_node_count": 0, "uncovered_boundary_node_indices": []}
    interp = RegularGridInterpolator(
        (bathy.lat, bathy.lon),
        np.asarray(coverage, dtype=np.uint8),
        bounds_error=False,
        fill_value=0,
    )
    covered = np.asarray(interp(np.column_stack([boundary.lonlat[:, 1], boundary.lonlat[:, 0]])), dtype=float) >= 1.0 - 1.0e-9
    missing = np.flatnonzero(~covered)
    return {
        "boundary_node_count": count,
        "uncovered_boundary_node_count": int(len(missing)),
        "uncovered_boundary_node_indices": [int(value) for value in missing[:100]],
    }


def model_domain_mask(bathy: BathymetryGrid, boundary: BoundaryNodes) -> np.ndarray:
    """Return a grid-cell-center mask for the projected model polygon."""
    lon2, lat2 = np.meshgrid(bathy.lon, bathy.lat)
    xy = project_points(np.column_stack([lon2.ravel(), lat2.ravel()]), boundary.projection)
    inside = contains_xy(boundary.domain_polygon_xy, xy[:, 0], xy[:, 1])
    return np.asarray(inside, dtype=bool).reshape(lon2.shape)


def land_open_junction_semantics(
    bathy: BathymetryGrid,
    boundary: BoundaryNodes,
    config: SizeFieldConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Derive a graded soft-size floor around land/open kind transitions."""
    xy = np.asarray(boundary.xy, dtype=float)
    targets = np.asarray(boundary.target_spacing_m, dtype=float)
    kinds = [str(value).strip().lower() for value in boundary.kinds]
    open_kinds = {"open", "open_boundary"}
    junction_xy: list[np.ndarray] = []
    junction_target: list[float] = []
    chains = boundary.constraint_chains[:1] if boundary.constraint_chains else [list(range(len(xy)))]
    for chain in chains:
        clean = [int(value) for value in chain if 0 <= int(value) < len(xy)]
        if len(clean) < 2:
            continue
        pairs = list(zip(clean, clean[1:]))
        if clean[-1] != clean[0]:
            pairs.append((clean[-1], clean[0]))
        for index_a, index_b in pairs:
            a_open = kinds[index_a] in open_kinds
            b_open = kinds[index_b] in open_kinds
            if a_open == b_open:
                continue
            open_index = index_a if a_open else index_b
            junction_xy.append(np.asarray(xy[open_index], dtype=float))
            junction_target.append(float(targets[open_index]))

    shape = (len(bathy.lat), len(bathy.lon))
    if not junction_xy:
        return (
            np.zeros(shape, dtype=bool),
            np.full(shape, np.nan, dtype=float),
            {
                "method": "boundary_kind_transition_grade",
                "junction_point_count": 0,
                "cell_count": 0,
                "transition_distance_m": config.junction_transition_distance_m,
            },
        )
    if config.gradation <= 0.0:
        raise ValueError("adaptive-coastal-v2 automatic junction grading requires positive gradation")
    lon2, lat2 = np.meshgrid(bathy.lon, bathy.lat)
    query_xy = project_points(np.column_stack([lon2.ravel(), lat2.ravel()]), boundary.projection)
    anchors = np.asarray(junction_xy, dtype=float)
    anchor_targets = np.asarray(junction_target, dtype=float)
    floor = np.full(len(query_xy), -np.inf, dtype=float)
    nearest_distance = np.full(len(query_xy), np.inf, dtype=float)
    for anchor, target in zip(anchors, anchor_targets):
        distance = np.linalg.norm(query_xy - anchor[None, :], axis=1)
        floor = np.maximum(floor, float(target) - float(config.gradation) * distance)
        nearest_distance = np.minimum(nearest_distance, distance)
    minimum = min(float(config.land_spacing_m), float(config.open_spacing_m))
    mask = floor > minimum + 1.0e-9
    if config.junction_transition_distance_m is not None:
        transition_distance = float(config.junction_transition_distance_m)
        if transition_distance < 0.0:
            raise ValueError("junction_transition_distance_m must be non-negative")
        mask &= nearest_distance <= transition_distance
    floor[~mask] = np.nan
    return (
        mask.reshape(shape),
        floor.reshape(shape),
        {
            "method": "boundary_kind_transition_grade",
            "junction_point_count": int(len(anchors)),
            "cell_count": int(np.count_nonzero(mask)),
            "transition_distance_m": config.junction_transition_distance_m,
            "junction_target_min_m": float(np.min(anchor_targets)),
            "junction_target_max_m": float(np.max(anchor_targets)),
        },
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
    if _resolve_size_field_profile(config.size_field_profile) == "adaptive-coastal-v2":
        return segment_boundary_distance_size(bathy, boundary, config)[0]
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


def segment_boundary_distance_size(
    bathy: BathymetryGrid,
    boundary: BoundaryNodes,
    config: SizeFieldConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate target size from the nearest boundary segment.

    Returns ``(size, distance, interpolated_boundary_target, source_code)`` on
    the bathymetry grid.  Candidate lookup is accelerated with grid-scale
    support points and exact point-to-segment projection for the nearby
    candidates, avoiding the sampling-phase jumps of nearest-node assignment.
    """
    lon2, lat2 = np.meshgrid(bathy.lon, bathy.lat)
    lonlat = np.column_stack([lon2.ravel(), lat2.ravel()])
    query_xy = project_points(lonlat, boundary.projection)
    seg_a, seg_b, target_a, target_b, source_code = _boundary_segments(boundary)
    if not len(seg_a):
        shape = lon2.shape
        return (
            np.full(shape, float(config.max_size_m), dtype=float),
            np.full(shape, np.inf, dtype=float),
            np.full(shape, float(config.max_size_m), dtype=float),
            np.zeros(shape, dtype=np.int16),
        )
    if not np.all(np.isfinite(target_a)) or not np.all(np.isfinite(target_b)) or np.any(target_a <= 0.0) or np.any(target_b <= 0.0):
        raise ValueError("Adaptive v2 boundary segment targets must be finite and positive")

    grid_support_spacing = _projected_grid_support_spacing(bathy, boundary)
    support_points: list[np.ndarray] = []
    support_segment: list[np.ndarray] = []
    lengths = np.linalg.norm(seg_b - seg_a, axis=1)
    for idx, length in enumerate(lengths):
        sample_count = max(2, int(np.ceil(float(length) / grid_support_spacing)) + 1)
        fractions = np.linspace(0.0, 1.0, sample_count)
        support_points.append(seg_a[idx][None, :] + fractions[:, None] * (seg_b[idx] - seg_a[idx])[None, :])
        support_segment.append(np.full(sample_count, idx, dtype=int))
    supports = np.vstack(support_points)
    support_to_segment = np.concatenate(support_segment)
    candidate_count = min(12, len(supports))
    tree = cKDTree(supports)
    nearest_distance = np.full(len(query_xy), np.inf, dtype=float)
    nearest_target = np.full(len(query_xy), float(config.max_size_m), dtype=float)
    nearest_source = np.zeros(len(query_xy), dtype=np.int16)
    chunk_size = 100_000
    for start in range(0, len(query_xy), chunk_size):
        stop = min(start + chunk_size, len(query_xy))
        query = query_xy[start:stop]
        candidate = tree.query(query, k=candidate_count, workers=-1)[1]
        candidate = np.asarray(candidate, dtype=int)
        if candidate.ndim == 1:
            candidate = candidate[:, None]
        segment_index = support_to_segment[candidate]
        a = seg_a[segment_index]
        vector = seg_b[segment_index] - a
        denom = np.sum(vector * vector, axis=2)
        relative = query[:, None, :] - a
        fraction = np.divide(
            np.sum(relative * vector, axis=2),
            denom,
            out=np.zeros_like(denom),
            where=denom > 0.0,
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        projected = a + fraction[:, :, None] * vector
        distance = np.linalg.norm(query[:, None, :] - projected, axis=2)
        winner = np.argmin(distance, axis=1)
        row = np.arange(len(query))
        selected_segment = segment_index[row, winner]
        selected_fraction = fraction[row, winner]
        nearest_distance[start:stop] = distance[row, winner]
        nearest_target[start:stop] = (
            (1.0 - selected_fraction) * target_a[selected_segment]
            + selected_fraction * target_b[selected_segment]
        )
        nearest_source[start:stop] = source_code[selected_segment]

    size = nearest_target + float(config.gradation) * nearest_distance
    finite_targets = np.asarray(boundary.target_spacing_m, dtype=float)
    finite_targets = finite_targets[np.isfinite(finite_targets) & (finite_targets > 0.0)]
    lower = float(np.nanmin(finite_targets)) if finite_targets.size else min(config.land_spacing_m, config.open_spacing_m)
    shape = lon2.shape
    return (
        np.clip(size, lower, float(config.max_size_m)).reshape(shape),
        nearest_distance.reshape(shape),
        nearest_target.reshape(shape),
        nearest_source.reshape(shape),
    )


def _boundary_segments(
    boundary: BoundaryNodes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xy = np.asarray(boundary.xy, dtype=float)
    targets = np.asarray(boundary.target_spacing_m, dtype=float)
    kinds = [str(value).strip().lower() for value in boundary.kinds]
    chains = boundary.constraint_chains or ([list(range(len(xy)))] if len(xy) else [])
    pairs: list[tuple[int, int]] = []
    for chain in chains:
        clean = [int(value) for value in chain if 0 <= int(value) < len(xy)]
        if len(clean) < 2:
            continue
        pairs.extend(zip(clean[:-1], clean[1:]))
        if clean[-1] != clean[0]:
            pairs.append((clean[-1], clean[0]))
    pairs = [(a, b) for a, b in pairs if a != b and np.linalg.norm(xy[b] - xy[a]) > 1.0e-9]
    if not pairs:
        empty_xy = np.empty((0, 2), dtype=float)
        empty = np.empty(0, dtype=float)
        return empty_xy, empty_xy.copy(), empty, empty.copy(), np.empty(0, dtype=np.int16)

    index_a = np.asarray([pair[0] for pair in pairs], dtype=int)
    index_b = np.asarray([pair[1] for pair in pairs], dtype=int)
    code = np.asarray(
        [_segment_kind_code(kinds[a], kinds[b]) for a, b in pairs],
        dtype=np.int16,
    )
    return xy[index_a], xy[index_b], targets[index_a], targets[index_b], code


def _segment_kind_code(kind_a: str, kind_b: str) -> int:
    if kind_a != kind_b:
        return 5
    if kind_a in {"land", "coast", "coastline"}:
        return 1
    if kind_a in {"open", "open_boundary"}:
        return 2
    if kind_a == "island":
        return 3
    if kind_a in {"frame", "frame_clip_boundary"}:
        return 4
    return 6


def _projected_grid_support_spacing(bathy: BathymetryGrid, boundary: BoundaryNodes) -> float:
    lon_center = float(np.nanmean(bathy.lon))
    lat_center = float(np.nanmean(bathy.lat))
    lon_step = abs(float(np.nanmedian(np.diff(bathy.lon)))) if len(bathy.lon) > 1 else 0.0
    lat_step = abs(float(np.nanmedian(np.diff(bathy.lat)))) if len(bathy.lat) > 1 else 0.0
    samples = np.asarray(
        [
            [lon_center, lat_center],
            [lon_center + lon_step, lat_center],
            [lon_center, lat_center + lat_step],
        ],
        dtype=float,
    )
    projected = project_points(samples, boundary.projection)
    distances = np.linalg.norm(projected[1:] - projected[0], axis=1)
    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    return max(float(np.nanmax(positive)) if positive.size else 1000.0, 1.0)


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


def apply_gradation_limit(
    lon: np.ndarray,
    lat: np.ndarray,
    size: np.ndarray,
    gradation: float,
    *,
    connectivity: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Priority-queue lower-envelope gradation limiter."""
    if connectivity not in {4, 8}:
        raise ValueError("gradation connectivity must be 4 or 8")
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
        neighbors = [(0, -1, dx_m), (0, 1, dx_m), (-1, 0, dy_m), (1, 0, dy_m)]
        if connectivity == 8:
            diagonal = float(np.hypot(dx_m, dy_m))
            neighbors.extend([(-1, -1, diagonal), (-1, 1, diagonal), (1, -1, diagonal), (1, 1, diagonal)])
        for dj, di, dist in neighbors:
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
        "method": (
            "priority_queue_lower_envelope"
            if connectivity == 4
            else "priority_queue_8_neighbor_lower_envelope"
        ),
        "connectivity": int(connectivity),
        "gradation": float(gradation),
        "relaxations": int(relaxations),
        "max_reduction_m": float(np.nanmax(raw - out)) if out.size else 0.0,
        "dx_m": float(dx_m),
        "dy_m": float(dy_m),
        "max_neighbor_gradation": float(max_neighbor_gradation(out, dx_m, dy_m, connectivity=connectivity)),
        "converged": True,
    }
    return out, report


def max_neighbor_gradation(
    size: np.ndarray,
    dx_m: float,
    dy_m: float,
    *,
    connectivity: int = 4,
) -> float:
    values = []
    if size.shape[1] > 1:
        values.append(np.nanmax(np.abs(np.diff(size, axis=1)) / max(dx_m, 1.0)))
    if size.shape[0] > 1:
        values.append(np.nanmax(np.abs(np.diff(size, axis=0)) / max(dy_m, 1.0)))
    if connectivity == 8 and size.shape[0] > 1 and size.shape[1] > 1:
        diagonal = max(float(np.hypot(dx_m, dy_m)), 1.0)
        values.append(np.nanmax(np.abs(size[1:, 1:] - size[:-1, :-1]) / diagonal))
        values.append(np.nanmax(np.abs(size[1:, :-1] - size[:-1, 1:]) / diagonal))
    return float(np.nanmax(values)) if values else 0.0


def estimate_node_budget(
    lon: np.ndarray,
    lat: np.ndarray,
    size_m: np.ndarray,
    *,
    coverage_mask: np.ndarray | None = None,
    domain_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate triangular-lattice node demand by metric cell quadrature."""
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
        coverage = np.asarray(coverage_mask, dtype=bool)
        if coverage.shape != expected_shape:
            raise ValueError(f"coverage_mask must have shape {expected_shape}; got {coverage.shape}")
        active &= coverage
    domain_limited = domain_mask is not None
    if domain_mask is not None:
        domain = np.asarray(domain_mask, dtype=bool)
        if domain.shape != expected_shape:
            raise ValueError(f"domain_mask must have shape {expected_shape}; got {domain.shape}")
        active &= domain
    density = np.zeros(size.shape, dtype=float)
    density[active] = 2.0 / (np.sqrt(3.0) * np.square(size[active]))
    estimate = float(np.sum(cell_area * density))
    return {
        "method": "triangular_lattice_metric_quadrature",
        "estimated_interior_node_count": int(np.ceil(estimate)),
        "estimated_interior_node_count_float": estimate,
        "active_cell_count": int(np.count_nonzero(active)),
        "active_area_m2": float(np.sum(cell_area[active])),
        "domain_mask_applied": bool(domain_limited),
        "interpretation": "rectangular-coverage upper estimate" if not domain_limited else "domain-masked estimate",
    }


def boundary_front_seed_points(
    boundary: BoundaryNodes,
    *,
    offset_factor: float = 0.65,
    minimum_boundary_clearance_factor: float = 0.20,
    minimum_seed_separation_factor: float = 0.35,
    include_hard_anchors: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create inward boundary-front and hard-anchor bisector seed candidates.

    Physical-boundary distance is evaluated against the polygon boundary, not
    against the nearest boundary vertex.  This helper is intentionally separate
    from meshing so callers can budget and audit the candidates before insertion.
    """
    if offset_factor <= 0.0:
        raise ValueError("offset_factor must be positive")
    domain = boundary.domain_polygon_xy
    candidates: list[tuple[int, np.ndarray, float, str]] = []
    chains = boundary.constraint_chains or ([list(range(len(boundary.xy)))] if len(boundary.xy) else [])
    hard = np.asarray(
        boundary.hard_anchor_mask if boundary.hard_anchor_mask is not None else np.zeros(len(boundary.xy), dtype=bool),
        dtype=bool,
    )
    skipped_outside = 0
    skipped_clearance = 0

    def add_interior_candidate(origin: np.ndarray, directions: list[np.ndarray], target: float, priority: int, kind: str) -> None:
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
                float(boundary.target_spacing_m[index_a]) + float(boundary.target_spacing_m[index_b])
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
                bisector = previous_vector / previous_norm + following_vector / following_norm
                target = float(boundary.target_spacing_m[index])
                add_interior_candidate(origin, [bisector, -bisector], target, 0, "hard_anchor_bisector")

    accepted: list[np.ndarray] = []
    accepted_targets: list[float] = []
    accepted_kinds: list[str] = []
    rejected_separation = 0
    for _, point, target, kind in sorted(candidates, key=lambda value: (value[0], value[1][1], value[1][0])):
        if accepted:
            points = np.asarray(accepted, dtype=float)
            distances = np.linalg.norm(points - point[None, :], axis=1)
            thresholds = float(minimum_seed_separation_factor) * np.minimum(
                np.asarray(accepted_targets, dtype=float), target
            )
            if np.any(distances < thresholds):
                rejected_separation += 1
                continue
        accepted.append(point)
        accepted_targets.append(float(target))
        accepted_kinds.append(kind)
    points = np.asarray(accepted, dtype=float).reshape((-1, 2)) if accepted else np.empty((0, 2), dtype=float)
    report = {
        "method": "segment_normal_and_hard_anchor_bisector",
        "candidate_count": int(len(candidates)),
        "accepted_count": int(len(points)),
        "segment_front_count": int(sum(kind == "segment_front" for kind in accepted_kinds)),
        "hard_anchor_bisector_count": int(sum(kind == "hard_anchor_bisector" for kind in accepted_kinds)),
        "skipped_outside_count": int(skipped_outside),
        "skipped_clearance_count": int(skipped_clearance),
        "rejected_seed_separation_count": int(rejected_separation),
        "offset_factor": float(offset_factor),
        "minimum_boundary_clearance_factor": float(minimum_boundary_clearance_factor),
        "minimum_seed_separation_factor": float(minimum_seed_separation_factor),
    }
    return points, report


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
    if size_field.soft_size is not None:
        variables["soft_priority_mesh_size_m"] = (("lat", "lon"), size_field.soft_size)
    if size_field.hard_size is not None:
        variables["hard_priority_mesh_size_m"] = (("lat", "lon"), size_field.hard_size)
    if size_field.junction_mask is not None:
        variables["land_open_junction_mask"] = (("lat", "lon"), np.asarray(size_field.junction_mask, dtype=np.uint8))
    if size_field.coverage_mask is not None:
        variables["size_field_coverage_mask"] = (("lat", "lon"), np.asarray(size_field.coverage_mask, dtype=np.uint8))
    if size_field.domain_mask is not None:
        variables["model_domain_mask"] = (("lat", "lon"), np.asarray(size_field.domain_mask, dtype=np.uint8))
    if size_field.source_attribution is not None:
        variables["size_source_attribution"] = (("lat", "lon"), np.asarray(size_field.source_attribution, dtype=np.int16))
    if size_field.boundary_source_attribution is not None:
        variables["boundary_source_attribution"] = (
            ("lat", "lon"),
            np.asarray(size_field.boundary_source_attribution, dtype=np.int16),
        )
    schema_version = str(size_field.report.get("schema_version", "fvcom_size_field_v1"))
    ds = xr.Dataset(
        variables,
        coords={"lon": size_field.lon, "lat": size_field.lat},
        attrs={"schema_version": schema_version, "coverage_policy": str(size_field.coverage_policy)},
    )
    if "size_source_attribution" in ds:
        ds["size_source_attribution"].attrs["codes"] = str(size_field.report.get("source_attribution_codes", {}))
    if "boundary_source_attribution" in ds:
        ds["boundary_source_attribution"].attrs["codes"] = str(
            size_field.report.get("boundary_source_attribution_codes", {})
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
