"""Generator-neutral raw-candidate runner for regional FVCOM mesh research.

This module builds one immutable ``fvcom_size_field_v4`` bundle, then routes
that exact boundary, bathymetry and size field to the clean-room generator and
to the explicitly supported Gmsh algorithm portfolio.  It deliberately stops
at raw meshes: common conditioning and cross-candidate ranking belong to the
separate bakeoff layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import unary_union

from .bathymetry import coarsen_for_size_field
from .boundary import (
    BoundaryNodes,
    OpenBoundaryChain,
    boundary_nodes_geojson,
    load_boundary_resolution,
)
from .boundary_size_reconciliation import (
    BoundarySizeReconciliationConfig,
    audit_reconciled_boundary_size_field,
    reconcile_boundary_size_field,
)
from .edge_size_audit import audit_edge_target_sizes
from .gmsh_experiment import (
    BudgetConfig,
    PreparedCase,
    SourceOpenBoundary,
    _backend_geometry,
    _delivered_lineage_manifest,
    _native_quality_report,
    assert_readiness_manifest_binding,
    bathymetry_coverage_report,
    check_case_readiness,
    file_sha256,
    integration_samples,
    prepare_case,
    select_uniform_target_m,
    source_open_boundary_lonlat,
)
from .mesh import MeshConfig, generate_mesh
from .node_budget import (
    DEFAULT_HARD_NODE_LIMIT,
    DEFAULT_PREFLIGHT_NODE_LIMIT,
    DEFAULT_SPACING_QUANTUM_M,
    delivered_node_budget_report,
)
from .projection import (
    project_geometry,
    project_points,
    unproject_geometry,
    unproject_points,
)
from .quality import evaluate_mesh_quality
from .size_field import (
    SizeField,
    SizeFieldConfig,
    WetMaskAwareSizeInterpolator,
    boundary_front_seed_points,
    build_size_field,
    estimate_node_budget,
    linear_target_metric_edge_fractions,
    write_size_field,
)
from .sms_2dm import read_2dm, write_2dm


SCHEMA_VERSION = "fvcom_mesher_portfolio_case_v2"
INPUT_BUNDLE_SCHEMA = "fvcom_mesher_input_bundle_v2"

CANDIDATE_ALIASES: Mapping[str, str] = {
    "clean-room": "clean_room_raw",
    "clean-room-raw": "clean_room_raw",
    "clean_room_raw": "clean_room_raw",
    "gmsh-1": "gmsh_meshadapt_1",
    "gmsh-meshadapt": "gmsh_meshadapt_1",
    "gmsh_meshadapt_1": "gmsh_meshadapt_1",
    "gmsh-5": "gmsh_delaunay_5",
    "gmsh-delaunay": "gmsh_delaunay_5",
    "gmsh_delaunay_5": "gmsh_delaunay_5",
    "gmsh-6": "gmsh_frontal_delaunay_6",
    "gmsh-frontal-delaunay": "gmsh_frontal_delaunay_6",
    "gmsh_frontal_delaunay_6": "gmsh_frontal_delaunay_6",
}
DEFAULT_PRIMARY_CANDIDATE = "gmsh_frontal_delaunay_6"
DEFAULT_FALLBACK_CANDIDATES = (
    "gmsh_delaunay_5",
    "clean_room_raw",
    "gmsh_meshadapt_1",
)
# New operational runs execute one deterministic candidate by default.  The
# challenger/control candidates remain available only when a caller names
# them explicitly; a Gmsh-6 failure must not silently change the generator.
DEFAULT_CANDIDATES = (DEFAULT_PRIMARY_CANDIDATE,)
GMSH_CANDIDATE_ALGORITHMS: Mapping[str, int] = {
    "gmsh_meshadapt_1": 1,
    "gmsh_delaunay_5": 5,
    "gmsh_frontal_delaunay_6": 6,
}


@dataclass(frozen=True)
class PortfolioCaseConfig:
    """Common raw-stage controls shared by every candidate."""

    preflight_node_limit: int = DEFAULT_PREFLIGHT_NODE_LIMIT
    hard_node_limit: int = DEFAULT_HARD_NODE_LIMIT
    size_field_max_cells: int = 1_500_000
    land_spacing_m: float = 50.0
    open_spacing_m: float = 3_000.0
    maximum_size_m: float = 8_000.0
    gradation: float = 0.10
    slope_elements: float = 10.0
    coastal_distance_m: float = 25_000.0
    hydraulic_elements_across_min: float = 3.0
    hydraulic_elements_across_max: float = 8.0
    hydraulic_max_width_m: float = 20_000.0
    hydraulic_bank_angle_deg: float = 110.0
    hydraulic_longitudinal_gradation: float = 0.10
    obc_hold_distance_m: float = 10_000.0
    obc_transition_distance_m: float = 60_000.0
    target_timestep_s: str = "auto"
    boundary_reconciliation_max_iterations: int = 8
    boundary_metric_edge: float = 1.0
    boundary_field_compatibility_factor: float = 1.5
    boundary_target_combination: str = "sampled_field"
    boundary_geometry_continuity: bool = True
    boundary_geometry_metric_ratio: float = 1.0
    boundary_trace_samples_per_target: float = 4.0
    boundary_trace_nearest_sample_count: int = 16
    use_case_budget_spacing_policy: bool = True
    clean_room_refine_iterations: int = 3
    clean_room_smooth_iterations: int = 8

    def __post_init__(self) -> None:
        if int(self.preflight_node_limit) <= 0:
            raise ValueError("preflight_node_limit must be positive")
        if int(self.hard_node_limit) <= int(self.preflight_node_limit):
            raise ValueError(
                "hard_node_limit must be greater than preflight_node_limit"
            )
        if int(self.size_field_max_cells) <= 0:
            raise ValueError("size_field_max_cells must be positive")
        for name in (
            "land_spacing_m",
            "open_spacing_m",
            "maximum_size_m",
            "slope_elements",
            "hydraulic_elements_across_min",
            "hydraulic_elements_across_max",
            "hydraulic_max_width_m",
            "hydraulic_bank_angle_deg",
            "obc_transition_distance_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(float(self.gradation)) or float(self.gradation) <= 0.0:
            raise ValueError("gradation must be finite and positive")
        if int(self.clean_room_refine_iterations) < 0:
            raise ValueError("clean_room_refine_iterations must be nonnegative")
        if int(self.clean_room_smooth_iterations) < 0:
            raise ValueError("clean_room_smooth_iterations must be nonnegative")
        if int(self.boundary_reconciliation_max_iterations) < 1:
            raise ValueError(
                "boundary_reconciliation_max_iterations must be positive"
            )
        if (
            not np.isfinite(float(self.boundary_metric_edge))
            or float(self.boundary_metric_edge) <= 0.0
        ):
            raise ValueError("boundary_metric_edge must be finite and positive")
        if (
            not np.isfinite(float(self.boundary_field_compatibility_factor))
            or float(self.boundary_field_compatibility_factor) <= 1.0
        ):
            raise ValueError(
                "boundary_field_compatibility_factor must be finite and > 1"
            )
        if (
            not np.isfinite(float(self.boundary_geometry_metric_ratio))
            or float(self.boundary_geometry_metric_ratio) <= 0.0
            or float(self.boundary_geometry_metric_ratio) > 1.0
        ):
            raise ValueError(
                "boundary_geometry_metric_ratio must be finite and in (0, 1]"
            )
        if self.boundary_target_combination not in {
            "minimum",
            "sampled_field",
        }:
            raise ValueError(
                "boundary_target_combination must be 'minimum' or "
                "'sampled_field'"
            )
        if (
            not np.isfinite(float(self.boundary_trace_samples_per_target))
            or float(self.boundary_trace_samples_per_target) < 2.0
        ):
            raise ValueError(
                "boundary_trace_samples_per_target must be finite and at "
                "least two"
            )
        if int(self.boundary_trace_nearest_sample_count) < 1:
            raise ValueError(
                "boundary_trace_nearest_sample_count must be positive"
            )


def _case_budget_spacing_policy(
    prepared: PreparedCase,
    readiness: Mapping[str, Any],
    config: PortfolioCaseConfig,
) -> tuple[PortfolioCaseConfig, dict[str, Any]]:
    """Resolve the case bathymetry floor and common budget-selected ``h_u``."""

    if not bool(config.use_case_budget_spacing_policy):
        return config, {
            "schema_version": "fvcom_case_budget_spacing_policy_v1",
            "status": "disabled",
            "applied": False,
        }
    bathymetry = readiness.get("bathymetry", {})
    floor_value = (
        bathymetry.get("bathymetry_floor_m")
        if isinstance(bathymetry, Mapping)
        else None
    )
    if floor_value is None:
        raise ValueError(
            "case budget spacing policy requires bathymetry_floor_m readiness"
        )
    floor_m = float(floor_value)
    sizing = prepared.manifest.get("sizing") or {}
    budget = prepared.manifest.get("budget") or {}
    near_size_m = float(
        sizing.get("near_obc_size_m")
        or config.maximum_size_m
    )
    near_distance_m = float(
        sizing.get("near_obc_distance_m")
        or config.obc_hold_distance_m
    )
    far_distance_m = float(
        sizing.get("far_obc_distance_m")
        or (
            config.obc_hold_distance_m
            + config.obc_transition_distance_m
        )
    )
    step_m = float(
        budget.get("hu_increment_m") or DEFAULT_SPACING_QUANTUM_M
    )
    quadrature_cells = min(int(config.size_field_max_cells), 250_000)
    quadrature = integration_samples(
        prepared,
        max_cells=quadrature_cells,
    )
    source_boundary_count = int(
        len(prepared.exterior_xy)
        + sum(len(values) for values in prepared.holes_xy)
    )
    budget_config = BudgetConfig(
        max_nodes=int(config.hard_node_limit),
        preflight_nodes=int(config.preflight_node_limit),
        near_size_m=near_size_m,
        near_distance_m=near_distance_m,
        far_distance_m=far_distance_m,
        step_m=step_m,
        integration_max_cells=quadrature_cells,
    )
    h_uniform_m, estimated_total = select_uniform_target_m(
        floor_m,
        source_boundary_count,
        quadrature["area_weights_m2"],
        quadrature["distance_to_obc_m"],
        has_open_boundary=bool(prepared.open_boundaries),
        config=budget_config,
    )
    effective = replace(
        config,
        land_spacing_m=float(h_uniform_m),
        open_spacing_m=(
            near_size_m
            if prepared.open_boundaries
            else float(h_uniform_m)
        ),
    )
    return effective, {
        "schema_version": "fvcom_case_budget_spacing_policy_v1",
        "status": "resolved",
        "applied": True,
        "method": (
            "bathymetry_floor_plus_quantized_uniform_target_bisection_under_"
            "common_preflight_budget"
        ),
        "bathymetry_floor_m": floor_m,
        "selected_h_uniform_m": float(h_uniform_m),
        "solid_and_island_target_m": float(h_uniform_m),
        "open_boundary_target_m": (
            near_size_m if prepared.open_boundaries else None
        ),
        "source_boundary_node_count": source_boundary_count,
        "simple_metric_estimated_total_nodes": float(estimated_total),
        "preflight_node_limit": int(config.preflight_node_limit),
        "hard_node_limit": int(config.hard_node_limit),
        "increment_m": step_m,
        "quadrature": {
            key: value
            for key, value in quadrature.items()
            if key
            not in {
                "x",
                "y",
                "area_weights_m2",
                "distance_to_obc_m",
            }
        },
        "geometry_forced_subgrid_policy": (
            "preserve every source vertex and topology; never delete short "
            "source chords; report rather than conceal targets forced below "
            "the bathymetry-supported scale"
        ),
    }


def _apply_case_budget_targets(
    boundary: BoundaryNodes,
    spacing_policy: Mapping[str, Any],
    *,
    geometry_continuity: bool = True,
    geometry_metric_ratio: float = 1.0,
) -> tuple[BoundaryNodes, dict[str, Any]]:
    """Assign budget targets and reconcile them with realized source chords.

    The source geometry is immutable: every original vertex must remain in the
    delivered CAD loops.  A short chord can therefore be much finer than a
    bathymetry- or budget-selected target.  When geometry continuity is
    enabled, each boundary vertex receives the conservative minimum of its
    policy target and the two incident chord scales

    ``h_geo = L_edge / geometry_metric_ratio``.

    The resulting targets seed both the canonical wet-domain field and the
    continuous boundary trace.  This makes the first interior ring respond to
    the actual one-dimensional discretization instead of merely reporting a
    geometry-forced jump after meshing.
    """

    budget_applied = bool(spacing_policy.get("applied"))
    open_nodes = {
        int(node)
        for chain in (boundary.open_boundaries or [])
        for node in chain.node_indices
    }
    if budget_applied:
        solid_target: float | None = float(
            spacing_policy["solid_and_island_target_m"]
        )
        open_target_value = spacing_policy.get("open_boundary_target_m")
        open_target: float | None = (
            float(open_target_value)
            if open_target_value is not None
            else solid_target
        )
        policy_targets = np.full(
            len(boundary.xy),
            solid_target,
            dtype=float,
        )
        if open_nodes:
            policy_targets[
                np.asarray(sorted(open_nodes), dtype=int)
            ] = open_target
    else:
        solid_target = None
        open_target = None
        policy_targets = np.asarray(
            boundary.target_spacing_m,
            dtype=float,
        ).copy()
        if policy_targets.shape != (len(boundary.xy),):
            raise ValueError(
                "boundary target spacing must have one value per vertex "
                "when case-budget spacing is disabled"
            )
        if np.any(~np.isfinite(policy_targets)) or np.any(
            policy_targets <= 0.0
        ):
            raise ValueError(
                "boundary target spacing must be finite and positive "
                "when case-budget spacing is disabled"
            )

    if (
        not np.isfinite(float(geometry_metric_ratio))
        or float(geometry_metric_ratio) <= 0.0
        or float(geometry_metric_ratio) > 1.0
    ):
        raise ValueError(
            "geometry_metric_ratio must be finite and in (0, 1]"
        )
    geometry_targets = np.full(len(boundary.xy), np.inf, dtype=float)
    edge_lengths: list[float] = []
    for raw_chain in boundary.constraint_chains:
        chain = [int(value) for value in raw_chain]
        for start, end in zip(chain, chain[1:] + chain[:1]):
            length = float(
                np.linalg.norm(boundary.xy[end] - boundary.xy[start])
            )
            if not np.isfinite(length) or length <= 0.0:
                raise ValueError(
                    f"constraint edge {start}->{end} has invalid length"
                )
            edge_lengths.append(length)
            edge_target = length / float(geometry_metric_ratio)
            geometry_targets[start] = min(
                float(geometry_targets[start]),
                edge_target,
            )
            geometry_targets[end] = min(
                float(geometry_targets[end]),
                edge_target,
            )
    if np.any(~np.isfinite(geometry_targets)):
        missing = np.flatnonzero(~np.isfinite(geometry_targets))
        raise ValueError(
            "every boundary node must be incident to a constraint edge; "
            f"missing {missing[:10].tolist()}"
        )
    targets = (
        np.minimum(policy_targets, geometry_targets)
        if bool(geometry_continuity)
        else policy_targets.copy()
    )

    forced_edges = 0
    forced_lengths: list[float] = []
    for raw_chain in boundary.constraint_chains:
        chain = [int(value) for value in raw_chain]
        for start, end in zip(chain, chain[1:] + chain[:1]):
            length = float(
                np.linalg.norm(boundary.xy[end] - boundary.xy[start])
            )
            target = min(
                float(policy_targets[start]),
                float(policy_targets[end]),
            )
            if length + 1.0e-9 < target:
                forced_edges += 1
                forced_lengths.append(length / target)
    metadata = dict(boundary.metadata or {})
    metadata["pre_geometry_target_spacing_m"] = policy_targets.copy()
    if budget_applied:
        metadata["case_budget_target_spacing_m"] = (
            policy_targets.copy()
        )
    metadata["geometry_continuity_target_spacing_m"] = (
        geometry_targets.copy()
    )
    metadata["effective_boundary_target_spacing_m"] = targets.copy()
    profile_suffixes: list[str] = []
    if budget_applied:
        profile_suffixes.append("case_budget_hu")
    if geometry_continuity:
        profile_suffixes.append("geometry_continuity")
    updated = (
        replace(
            boundary,
            target_spacing_m=targets,
            metadata=metadata,
            resolution_profile=(
                boundary.resolution_profile
                + "".join(f"+{value}" for value in profile_suffixes)
            ),
        )
        if profile_suffixes
        else boundary
    )
    return updated, {
        "applied": bool(budget_applied),
        "solid_and_island_target_m": solid_target,
        "open_boundary_target_m": (
            open_target if open_nodes else None
        ),
        "open_boundary_node_count": int(len(open_nodes)),
        "geometry_continuity_applied": bool(geometry_continuity),
        "geometry_metric_ratio": float(geometry_metric_ratio),
        "geometry_target_minimum_m": float(np.min(geometry_targets)),
        "geometry_target_p50_m": float(np.median(geometry_targets)),
        "geometry_target_p95_m": float(
            np.percentile(geometry_targets, 95.0)
        ),
        "geometry_target_maximum_m": float(np.max(geometry_targets)),
        "geometry_limited_node_count": int(
            np.count_nonzero(
                geometry_targets < policy_targets - 1.0e-9
            )
            if geometry_continuity
            else 0
        ),
        "geometry_limited_node_fraction": float(
            np.count_nonzero(
                geometry_targets < policy_targets - 1.0e-9
            )
            / max(len(targets), 1)
            if geometry_continuity
            else 0.0
        ),
        "source_edge_length_minimum_m": float(min(edge_lengths)),
        "source_edge_length_p50_m": float(np.median(edge_lengths)),
        "source_edge_length_p95_m": float(
            np.percentile(edge_lengths, 95.0)
        ),
        "source_edge_length_maximum_m": float(max(edge_lengths)),
        "geometry_forced_subgrid_edge_count": int(forced_edges),
        "geometry_forced_subgrid_edge_fraction": float(
            forced_edges
            / max(
                sum(len(chain) for chain in boundary.constraint_chains),
                1,
            )
        ),
        "geometry_forced_edge_l_over_target_minimum": (
            float(min(forced_lengths)) if forced_lengths else None
        ),
        "geometry_forced_edge_l_over_target_p50": (
            float(np.median(forced_lengths)) if forced_lengths else None
        ),
    }


@dataclass
class _CandidateMesh:
    nodes_xy: np.ndarray
    nodes_lonlat: np.ndarray
    triangles_1based: np.ndarray
    depths: np.ndarray
    constraint_chains_zero: list[list[int]]
    open_boundary_chains_1based: list[list[int]]
    open_boundary_cyclic: list[bool]
    constraint_report: dict[str, Any]
    boundary_metadata: dict[str, Any]
    generator_report: dict[str, Any]
    extra_quality: dict[str, Any]
    raw_msh_path: Path | None = None
    logger_output: tuple[str, ...] = ()


class ProjectedSizeSampler:
    """Persistent strict sampler for one structured canonical size field.

    Interpolators and the projection transformer are cached once.  This avoids
    constructing ``SizeField.sample`` machinery for every scalar Gmsh callback.
    """

    def __init__(self, size_field: SizeField, projection: Any) -> None:
        self._projection = projection
        coverage = np.asarray(size_field.coverage_mask, dtype=bool)
        domain = np.asarray(
            getattr(size_field, "domain_mask", coverage),
            dtype=bool,
        )
        self._size = WetMaskAwareSizeInterpolator(
            np.asarray(size_field.lat, dtype=float),
            np.asarray(size_field.lon, dtype=float),
            np.asarray(size_field.size, dtype=float),
            coverage & domain,
            coverage,
        )
        sampling_interface = dict(
            (getattr(size_field, "report", {}) or {}).get(
                "sampling_interface"
            )
            or {}
        )
        self.report = {
            "schema_version": "fvcom_projected_size_sampler_v2",
            "sampling_interface": sampling_interface,
            "sampling_interface_schema_version": str(
                sampling_interface.get(
                    "schema_version",
                    "legacy_unspecified",
                )
            ),
            "strict_coverage": True,
        }

    def sample_xy(self, values_xy: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        sampled, _active_support = self.sample_xy_with_active_support(
            values_xy
        )
        return sampled

    def sample_xy_with_active_support(
        self,
        values_xy: np.ndarray | Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return strict raster values and wet-stencil support flags."""

        values = np.asarray(values_xy, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("projected size queries must have shape (n, 2)")
        if not np.all(np.isfinite(values)):
            raise ValueError("projected size queries contain non-finite coordinates")
        lon, lat = self._projection.to_lonlat.transform(values[:, 0], values[:, 1])
        query = np.column_stack([lat, lon])
        sampled, active_support = self._size.sample_with_active_support(query)
        sampled = np.asarray(sampled, dtype=float)
        invalid = ~np.isfinite(sampled) | (sampled <= 0.0)
        if np.any(invalid):
            first = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                "canonical size query is outside strict coverage or invalid: "
                f"{int(np.count_nonzero(invalid))} point(s); first projected "
                f"coordinate=({values[first, 0]:.6f}, {values[first, 1]:.6f})"
            )
        return sampled, np.asarray(active_support, dtype=bool)

    def __call__(self, x: float, y: float) -> float:
        return float(self.sample_xy(np.asarray([[x, y]], dtype=float))[0])


class BoundaryTraceSizeSampler:
    """Add a sampled gradated boundary trace to a raster size sampler.

    ``fvcom_size_field_v4`` is stored at wet-cell centers. Directly
    interpolating that raster back to the exact shoreline can therefore
    coarsen the apparent target by several factors. This wrapper evaluates
    the point-sampled approximation

    ``min(H_raster(x), min_i(h_gamma_i + g * |x - x_i|))``

    using deterministic samples that include every delivered boundary vertex
    and every edge midpoint. The nearest-neighbour search expands until a
    global lower bound proves that no unvisited sample can reduce the result.
    It therefore returns the exact minimum over the deterministic trace sample
    set, rather than a fixed-``k`` approximation, while retaining the raster
    field away from the boundary.
    """

    def __init__(
        self,
        base_sampler: ProjectedSizeSampler,
        boundary: BoundaryNodes,
        *,
        gradation: float,
        samples_per_target: float = 4.0,
        nearest_sample_count: int = 16,
        maximum_total_sample_count: int = 5_000_000,
        query_chunk_size: int = 4_096,
    ) -> None:
        if not np.isfinite(float(gradation)) or float(gradation) <= 0.0:
            raise ValueError("boundary trace gradation must be positive")
        if (
            not np.isfinite(float(samples_per_target))
            or float(samples_per_target) < 2.0
        ):
            raise ValueError("samples_per_target must be at least two")
        if int(nearest_sample_count) < 1:
            raise ValueError("nearest_sample_count must be positive")
        if int(maximum_total_sample_count) < 1:
            raise ValueError(
                "maximum_total_sample_count must be positive"
            )
        if int(query_chunk_size) < 1:
            raise ValueError("query_chunk_size must be positive")
        points: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        sample_spacings: list[float] = []
        total_sample_count = 0
        for raw_chain in boundary.constraint_chains:
            chain = [int(value) for value in raw_chain]
            for position, start in enumerate(chain):
                end = chain[(position + 1) % len(chain)]
                a = np.asarray(boundary.xy[start], dtype=float)
                b = np.asarray(boundary.xy[end], dtype=float)
                ha = float(boundary.target_spacing_m[start])
                hb = float(boundary.target_spacing_m[end])
                length = float(np.linalg.norm(b - a))
                remaining_sample_count = (
                    int(maximum_total_sample_count) - total_sample_count
                )
                fraction = linear_target_metric_edge_fractions(
                    length,
                    ha,
                    hb,
                    samples_per_target=float(samples_per_target),
                    include_end=False,
                    maximum_sample_count=remaining_sample_count,
                )
                total_sample_count += len(fraction)
                sample_spacings.append(
                    length
                    * float(
                        np.max(
                            np.diff(
                                np.concatenate(
                                    [fraction, np.asarray([1.0])]
                                )
                            )
                        )
                    )
                )
                points.append(
                    a[None, :] + fraction[:, None] * (b - a)[None, :]
                )
                targets.append((1.0 - fraction) * ha + fraction * hb)
        if not points:
            raise ValueError("boundary trace requires at least one edge")
        self._base = base_sampler
        self._gradation = float(gradation)
        self._points = np.vstack(points)
        self._targets = np.concatenate(targets)
        self._tree = cKDTree(self._points)
        self._initial_nearest_sample_count = min(
            int(nearest_sample_count),
            len(self._points),
        )
        self._query_chunk_size = int(query_chunk_size)
        self._minimum_target = float(np.min(self._targets))
        maximum_sample_spacing = max(sample_spacings, default=0.0)
        self.report = {
            "schema_version": "fvcom_boundary_trace_sampler_v2",
            "method": (
                "raster_min_deterministic_boundary_point_"
                "euclidean_gradation_extension_adaptive_exact_sample_minimum"
            ),
            "boundary_sample_count": int(len(self._points)),
            "samples_per_target": float(samples_per_target),
            "sample_distribution": (
                "linear_endpoint_target_metric_equidistribution"
            ),
            "maximum_total_sample_count": int(
                maximum_total_sample_count
            ),
            "query_chunk_size": int(self._query_chunk_size),
            "memory_bounded_query_chunks": True,
            # Retain the v1 key as a backward-compatible alias. In v2 this is
            # the initial search size, not a hard truncation.
            "nearest_sample_count": int(
                self._initial_nearest_sample_count
            ),
            "initial_nearest_sample_count": int(
                self._initial_nearest_sample_count
            ),
            "gradation": float(self._gradation),
            "endpoint_midpoint_exact_by_construction": True,
            "adaptive_neighbor_expansion": True,
            "exact_discrete_trace_sample_minimum": True,
            "global_unvisited_lower_bound": (
                "minimum_boundary_target_plus_gradation_times_"
                "current_kth_nearest_distance"
            ),
            "maximum_trace_sample_spacing_m": float(
                maximum_sample_spacing
            ),
            "maximum_continuous_trace_overestimate_bound_m": float(
                self._gradation * maximum_sample_spacing
            ),
            "sample_query_count": 0,
            "expanded_sample_query_count": 0,
            "maximum_neighbors_examined": 0,
            "base_query_without_positive_active_support_count": 0,
            "no_active_support_policy": "boundary_trace_authoritative",
            "operational_counter_scope": "since_most_recent_reset",
            "distance_metric": "straight_euclidean",
            "barrier_aware": False,
            "base_raster_sampler": dict(
                getattr(base_sampler, "report", {})
            ),
        }

    def reset_operational_counters(self) -> None:
        """Start a fresh, candidate-local sampler measurement interval."""

        self.report["sample_query_count"] = 0
        self.report["expanded_sample_query_count"] = 0
        self.report["maximum_neighbors_examined"] = 0
        self.report["base_query_without_positive_active_support_count"] = 0

    def sample_xy(
        self,
        values_xy: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        values = np.asarray(values_xy, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("boundary trace query must have shape (N, 2)")
        if not len(values):
            return np.empty(0, dtype=float)
        base, active_support = self._sample_base_with_support(values)
        effective_base = np.where(active_support, base, np.inf)
        self.report[
            "base_query_without_positive_active_support_count"
        ] = int(
            self.report[
                "base_query_without_positive_active_support_count"
            ]
        ) + int(np.count_nonzero(~active_support))
        return self._sample_trace_reduction(
            values,
            upper_bound=np.asarray(effective_base, dtype=float),
            update_counters=True,
        )

    def sample_trace_xy(
        self,
        values_xy: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Return the exact discrete trace extension without the raster min."""

        values = np.asarray(values_xy, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("boundary trace query must have shape (N, 2)")
        if not len(values):
            return np.empty(0, dtype=float)
        return self._sample_trace_reduction(
            values,
            upper_bound=None,
            update_counters=False,
        )

    def sample_components_xy(
        self,
        values_xy: np.ndarray | Sequence[Sequence[float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return strict raster and exact trace components independently."""

        values = np.asarray(values_xy, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("boundary trace query must have shape (N, 2)")
        if not len(values):
            empty = np.empty(0, dtype=float)
            return empty, empty
        base, active_support = self._sample_base_with_support(values)
        base = np.where(active_support, base, np.inf)
        trace = self._sample_trace_reduction(
            values,
            upper_bound=None,
            update_counters=False,
        )
        return base, trace

    def _sample_base_with_support(
        self,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        method = getattr(
            self._base,
            "sample_xy_with_active_support",
            None,
        )
        if callable(method):
            base, support = method(values)
            return (
                np.asarray(base, dtype=float),
                np.asarray(support, dtype=bool),
            )
        base = np.asarray(self._base.sample_xy(values), dtype=float)
        return base, np.ones(len(values), dtype=bool)

    @property
    def gradation(self) -> float:
        return float(self._gradation)

    @property
    def minimum_trace_target_m(self) -> float:
        return float(self._minimum_target)

    def _sample_trace_reduction(
        self,
        values: np.ndarray,
        *,
        upper_bound: np.ndarray | None,
        update_counters: bool,
    ) -> np.ndarray:
        if len(values) > self._query_chunk_size:
            chunks: list[np.ndarray] = []
            for begin in range(0, len(values), self._query_chunk_size):
                end = min(len(values), begin + self._query_chunk_size)
                chunks.append(
                    self._sample_trace_reduction(
                        values[begin:end],
                        upper_bound=(
                            None
                            if upper_bound is None
                            else np.asarray(upper_bound)[begin:end]
                        ),
                        update_counters=update_counters,
                    )
                )
            return np.concatenate(chunks)
        result = (
            np.full(len(values), np.inf, dtype=float)
            if upper_bound is None
            else np.asarray(upper_bound, dtype=float).copy()
        )
        best_extension = np.full(len(values), np.inf, dtype=float)
        unresolved = np.arange(len(values), dtype=int)
        neighbor_count = int(self._initial_nearest_sample_count)
        expanded_queries = 0
        maximum_examined = 0
        tolerance = 32.0 * np.finfo(float).eps

        while len(unresolved):
            distance, indices = self._tree.query(
                values[unresolved],
                k=neighbor_count,
            )
            distance_array = np.asarray(distance, dtype=float)
            index_array = np.asarray(indices, dtype=int)
            if distance_array.ndim == 1:
                distance_array = distance_array[:, None]
                index_array = index_array[:, None]
            local_extension = np.min(
                self._targets[index_array]
                + self._gradation * distance_array,
                axis=1,
            )
            best_extension[unresolved] = np.minimum(
                best_extension[unresolved],
                local_extension,
            )
            if upper_bound is None:
                result[unresolved] = best_extension[unresolved]
            else:
                result[unresolved] = np.minimum(
                    upper_bound[unresolved],
                    best_extension[unresolved],
                )
            maximum_examined = max(maximum_examined, neighbor_count)
            if neighbor_count >= len(self._points):
                break

            # Every unvisited sample is at least as distant as the current
            # kth neighbour and has target >= the global minimum target.
            # Once the incumbent is no larger than that bound, the exact
            # discrete min-plus value is proven without visiting the rest.
            unseen_lower_bound = (
                self._minimum_target
                + self._gradation * distance_array[:, -1]
            )
            scale = np.maximum(
                1.0,
                np.maximum(
                    np.abs(result[unresolved]),
                    np.abs(unseen_lower_bound),
                ),
            )
            resolved_here = (
                result[unresolved]
                <= unseen_lower_bound + tolerance * scale
            )
            remaining = unresolved[~resolved_here]
            if not len(remaining):
                break
            expanded_queries += int(len(remaining))
            unresolved = remaining
            neighbor_count = min(
                len(self._points),
                max(neighbor_count + 1, 2 * neighbor_count),
            )

        if update_counters:
            self.report["sample_query_count"] = int(
                self.report["sample_query_count"]
            ) + int(len(values))
            self.report["expanded_sample_query_count"] = int(
                self.report["expanded_sample_query_count"]
            ) + int(expanded_queries)
            self.report["maximum_neighbors_examined"] = max(
                int(self.report["maximum_neighbors_examined"]),
                int(maximum_examined),
            )
        return result

    def __call__(self, x: float, y: float) -> float:
        return float(self.sample_xy(np.asarray([[x, y]], dtype=float))[0])


def normalize_candidate_ids(values: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize aliases, preserve order, and reject unknown candidates."""

    supplied = DEFAULT_CANDIDATES if values is None else tuple(values)
    normalized: list[str] = []
    for raw in supplied:
        key = str(raw).strip().lower()
        candidate = CANDIDATE_ALIASES.get(key)
        if candidate is None:
            allowed = ", ".join(sorted(CANDIDATE_ALIASES))
            raise ValueError(f"unknown candidate {raw!r}; choose from {allowed}")
        if candidate not in normalized:
            normalized.append(candidate)
    if not normalized:
        raise ValueError("at least one mesher candidate is required")
    return tuple(normalized)


def capability_routing(prepared: PreparedCase) -> dict[str, Any]:
    """Return explicit topology capabilities without attempting generation."""

    open_count = len(prepared.open_boundaries)
    cyclic = any(item.cyclic for item in prepared.open_boundaries)
    clean_supported = bool(open_count <= 1 and not cyclic)
    clean_reasons: list[str] = []
    if open_count > 1:
        clean_reasons.append("multiple_open_boundaries_unsupported")
    if cyclic:
        clean_reasons.append("cyclic_open_boundary_unsupported")
    return {
        "schema_version": "fvcom_mesher_capability_routing_v2",
        "case_open_boundary_count": int(open_count),
        "case_has_cyclic_open_boundary": bool(cyclic),
        "default_raw_candidate": DEFAULT_PRIMARY_CANDIDATE,
        "fallback_order": [],
        "explicit_research_candidates": list(DEFAULT_FALLBACK_CANDIDATES),
        "promotion_status": "gmsh_frontal_delaunay_6_operational_default",
        "selection_basis": (
            "topology_complete_primary_then_metric_by_metric_challengers"
        ),
        "candidates": {
            "clean_room_raw": {
                "supported": clean_supported,
                "policy_role": "explicit_research_control_when_supported",
                "supports_zero_obc": True,
                "supports_plural_obc": False,
                "supports_cyclic_obc": False,
                "reasons": clean_reasons,
            },
            **{
                candidate: {
                    "supported": True,
                    "policy_role": (
                        "operational_default_raw"
                        if candidate == DEFAULT_PRIMARY_CANDIDATE
                        else (
                            "delaunay_challenger"
                            if candidate == "gmsh_delaunay_5"
                            else "meshadapt_diagnostic"
                        )
                    ),
                    "supports_zero_obc": True,
                    "supports_plural_obc": True,
                    "supports_cyclic_obc": True,
                    "reasons": [],
                }
                for candidate in GMSH_CANDIDATE_ALGORITHMS
            },
        },
    }


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return {
            str(key): _json_ready(item)
            for key, item in asdict(value).items()
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _validate_output_path_budget(output: Path) -> None:
    """Fail early when a Windows run root cannot hold portable artifacts."""

    if os.name != "nt":
        return
    representative = (
        output
        / "candidates"
        / "gmsh_frontal_delaunay_6"
        / "boundary_metadata.json"
    )
    # Fiona, NetCDF, and some Gmsh/Python builds still cross legacy Win32 APIs.
    # Keep a small margin below MAX_PATH rather than failing after meshing.
    if len(str(representative)) >= 248:
        raise ValueError(
            "portfolio output path is too long for portable Windows artifact "
            f"writers ({len(str(representative))} characters for "
            f"{representative}); shorten --output-dir"
        )


def _hash_artifacts(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": int(path.stat().st_size),
        }
        for name, path in paths.items()
    }


def _scientific_bundle_sha256(
    *,
    case_id: str,
    source_hashes: Mapping[str, Mapping[str, Any]],
    canonical_boundary_sha256: str,
    canonical_field_sha256: str,
    projection_epsg: int,
    portfolio_config: PortfolioCaseConfig,
    size_field_config: SizeFieldConfig,
    preflight: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Hash only scientific content and controls, excluding paths/timestamps."""

    node_budget = {
        key: preflight.get(key)
        for key in (
            "canonical_size_field_schema",
            "estimated_interior_node_count",
            "upstream_source_boundary_node_count",
            "canonical_reconciled_boundary_node_count",
            "explicit_source_boundary_node_count",
            "common_boundary_node_count",
            "boundary_front_seed_count",
            "estimated_total_node_count",
            "preflight_node_limit",
            "hard_node_limit",
            "passed",
        )
    }
    payload = {
        "schema_version": "fvcom_scientific_input_bundle_hash_v2",
        "case_id": str(case_id),
        "source_artifact_sha256": {
            str(name): str(values["sha256"])
            for name, values in sorted(source_hashes.items())
        },
        "canonical_boundary_sha256": str(canonical_boundary_sha256),
        "canonical_size_field_sha256": str(canonical_field_sha256),
        "projection_epsg": int(projection_epsg),
        "portfolio_config": _json_ready(asdict(portfolio_config)),
        "size_field_config": _json_ready(asdict(size_field_config)),
        "node_budget": _json_ready(node_budget),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), payload


def _source_obc_node_indices(
    source: Any,
    exterior_count: int,
) -> tuple[int, ...]:
    segments = tuple(int(value) for value in source.exterior_segment_indices)
    if source.orientation == "source":
        nodes = list(segments)
        if not source.cyclic:
            nodes.append((segments[-1] + 1) % exterior_count)
    elif source.orientation == "reverse":
        nodes = [(segments[0] + 1) % exterior_count]
        nodes.extend(segments)
        if source.cyclic:
            nodes.pop()
    else:
        raise ValueError(
            f"unsupported source OBC orientation {source.orientation!r}"
        )
    if len(nodes) != len(set(nodes)):
        raise ValueError(f"source OBC {source.chain_id!r} repeats a node")
    return tuple(int(value) for value in nodes)


def _open_and_land_geometries(
    prepared: PreparedCase,
) -> tuple[LineString | MultiLineString, LineString | MultiLineString]:
    exterior = np.asarray(prepared.exterior_xy, dtype=float)
    open_indices = {
        int(index)
        for boundary in prepared.open_boundaries
        for index in boundary.exterior_segment_indices
    }
    open_lines = [
        LineString(
            [
                exterior[index],
                exterior[(index + 1) % len(exterior)],
            ]
        )
        for index in sorted(open_indices)
    ]
    land_lines = [
        LineString(
            [
                exterior[index],
                exterior[(index + 1) % len(exterior)],
            ]
        )
        for index in range(len(exterior))
        if index not in open_indices
    ]
    for hole in prepared.holes_xy:
        closed = np.vstack([hole, hole[0]])
        land_lines.append(LineString(closed))
    return (
        unary_union(open_lines) if open_lines else LineString(),
        unary_union(land_lines) if land_lines else LineString(),
    )


def _reproject_passage_diagnostics(
    diagnostics: list[dict[str, Any]] | None,
    source_projection: Any,
    target_projection: Any,
) -> list[dict[str, Any]] | None:
    if diagnostics is None:
        return None
    result: list[dict[str, Any]] = []
    for record in diagnostics:
        copied = dict(record)
        geometry = copied.get("geometry_xy")
        if geometry is not None and source_projection.epsg != target_projection.epsg:
            copied["geometry_xy"] = project_geometry(
                unproject_geometry(geometry, source_projection),
                target_projection,
            )
        result.append(copied)
    return result


def _load_canonical_boundary(
    prepared: PreparedCase,
    config: PortfolioCaseConfig,
) -> tuple[BoundaryNodes, dict[str, Any]]:
    """Rebase authoritative node metadata onto the exact prepared case loops."""

    loops = (
        np.asarray(prepared.exterior_xy, dtype=float),
        *(np.asarray(values, dtype=float) for values in prepared.holes_xy),
    )
    chains: list[list[int]] = []
    offset = 0
    for values in loops:
        chains.append(list(range(offset, offset + len(values))))
        offset += len(values)
    xy = np.vstack(loops)
    input_kind = str(prepared.manifest["boundary"]["input_kind"])
    resolution_payload: dict[str, Any] = {}
    metadata: dict[str, np.ndarray] | None = None
    passage_diagnostics: list[dict[str, Any]] | None = None

    if input_kind == "adaptive_v2":
        manifest_path = prepared.input_paths.get("boundary_manifest")
        if manifest_path is None:
            raise FileNotFoundError(
                "prepared adaptive-v2 case has no boundary_manifest input"
            )
        _package, loaded, resolution_payload = load_boundary_resolution(
            manifest_path
        )
        if str(resolution_payload.get("final_status")) != "pass":
            raise ValueError("adaptive-v2 boundary manifest is not pass")
        if len(loaded.constraint_chains) != len(chains):
            raise ValueError(
                "adaptive-v2 chain count differs from prepared source loops"
            )
        loaded_order = [
            int(index)
            for chain in loaded.constraint_chains
            for index in chain
        ]
        if [len(value) for value in loaded.constraint_chains] != [
            len(value) for value in chains
        ]:
            raise ValueError(
                "adaptive-v2 chain lengths differ from prepared source loops"
            )
        kinds = [str(loaded.kinds[index]) for index in loaded_order]
        targets = np.asarray(
            [loaded.target_spacing_m[index] for index in loaded_order],
            dtype=float,
        )
        hard = np.asarray(
            [
                (
                    bool(loaded.hard_anchor_mask[index])
                    if loaded.hard_anchor_mask is not None
                    else False
                )
                for index in loaded_order
            ],
            dtype=bool,
        )
        if loaded.metadata is not None:
            metadata = {
                name: np.asarray(values)[np.asarray(loaded_order, dtype=int)]
                for name, values in loaded.metadata.items()
            }
        passage_diagnostics = _reproject_passage_diagnostics(
            loaded.passage_diagnostics,
            loaded.projection,
            prepared.projection,
        )
        adaptive = True
        resolution_profile = str(resolution_payload.get("profile"))
        source_manifest = str(manifest_path)
    else:
        kinds = []
        targets_list: list[float] = []
        for loop_index, values in enumerate(loops):
            if loop_index:
                kinds.extend(["island"] * len(values))
                targets_list.extend([float(config.land_spacing_m)] * len(values))
            else:
                for segment_kind in prepared.exterior_segment_kinds:
                    kind = "open" if str(segment_kind).lower() == "open" else "land"
                    kinds.append(kind)
                    targets_list.append(
                        float(config.open_spacing_m)
                        if kind == "open"
                        else float(config.land_spacing_m)
                    )
        targets = np.asarray(targets_list, dtype=float)
        hard = np.zeros(len(xy), dtype=bool)
        hard[np.asarray(prepared.hard_anchor_vertex_indices, dtype=int)] = True
        adaptive = False
        resolution_profile = "legacy"
        source_manifest = None

    if len(kinds) != len(xy) or len(targets) != len(xy):
        raise ValueError("canonical boundary metadata length mismatch")
    if not np.all(np.isfinite(targets)) or np.any(targets <= 0.0):
        raise ValueError("canonical boundary targets must be finite and positive")
    domain = Polygon(
        prepared.exterior_xy,
        holes=[values.tolist() for values in prepared.holes_xy],
    )
    if not domain.is_valid or domain.is_empty:
        raise ValueError("prepared source loops do not form a valid wet domain")
    open_geometry, land_geometry = _open_and_land_geometries(prepared)
    open_boundaries = [
        OpenBoundaryChain(
            chain_id=value.chain_id,
            node_indices=_source_obc_node_indices(
                value,
                len(prepared.exterior_xy),
            ),
            kind=value.kind,
            cyclic=value.cyclic,
            orientation=value.orientation,
        )
        for value in prepared.open_boundaries
    ]
    open_indices = [
        int(index)
        for chain in open_boundaries
        for index in chain.node_indices
    ]
    lonlat = unproject_points(xy, prepared.projection)
    boundary = BoundaryNodes(
        xy=xy,
        lonlat=lonlat,
        kinds=kinds,
        target_spacing_m=targets,
        exterior_indices=list(chains[0]),
        open_boundary_indices=list(dict.fromkeys(open_indices)),
        constraint_chains=chains,
        domain_polygon_xy=domain,
        open_boundary_xy=open_geometry,
        land_boundary_xy=land_geometry,
        island_polygons_xy=[
            Polygon(np.asarray(values, dtype=float))
            for values in prepared.holes_xy
        ],
        projection=prepared.projection,
        hard_anchor_mask=hard,
        adaptive_resolution=adaptive,
        source_resolution_manifest=source_manifest,
        resolution_profile=resolution_profile,
        metadata=metadata,
        passage_diagnostics=passage_diagnostics,
        open_boundaries=open_boundaries,
        topology_compensation=(
            loaded.topology_compensation if input_kind == "adaptive_v2" else None
        ),
    )
    report = {
        "schema_version": "fvcom_portfolio_canonical_boundary_v1",
        "input_kind": input_kind,
        "resolution_profile": resolution_profile,
        "source_resolution_manifest": source_manifest,
        "node_count": int(len(xy)),
        "constraint_chain_count": int(len(chains)),
        "open_boundary_count": int(len(open_boundaries)),
        "open_boundary_ids": [value.chain_id for value in open_boundaries],
        "open_boundary_cyclic": [bool(value.cyclic) for value in open_boundaries],
        "hard_anchor_count": int(np.count_nonzero(hard)),
        "target_spacing_minimum_m": float(np.min(targets)),
        "target_spacing_maximum_m": float(np.max(targets)),
        "adaptive_manifest_summary": {
            "schema_version": resolution_payload.get("schema_version"),
            "final_status": resolution_payload.get("final_status"),
            "profile": resolution_payload.get("profile"),
        },
        "boundary_topology_compensation": (
            loaded.topology_compensation.report
            if input_kind == "adaptive_v2" and loaded.topology_compensation is not None
            else None
        ),
    }
    return boundary, report


def _prepared_case_on_boundary(
    prepared: PreparedCase,
    boundary: BoundaryNodes,
) -> PreparedCase:
    """Rebase Gmsh source entities onto one common delivered boundary.

    ``PreparedCase`` stores OBCs as exterior *segment* indices, while the
    generator-neutral boundary contract stores ordered node chains.  This
    adapter reconstructs segment indices without changing chain identity,
    traversal direction, loop orientation, or the immutable case inputs.
    """

    if not boundary.constraint_chains:
        raise ValueError("reconciled boundary has no constraint chains")
    chains = [
        [int(value) for value in chain]
        for chain in boundary.constraint_chains
    ]
    exterior = chains[0]
    if len(exterior) < 3 or len(exterior) != len(set(exterior)):
        raise ValueError("reconciled exterior chain is invalid")
    exterior_position = {
        int(node_index): int(position)
        for position, node_index in enumerate(exterior)
    }
    exterior_count = len(exterior)
    open_boundaries: list[SourceOpenBoundary] = []
    for chain in boundary.open_boundaries or []:
        nodes = [int(value) for value in chain.node_indices]
        if chain.cyclic:
            pairs = list(zip(nodes, nodes[1:] + nodes[:1]))
        else:
            pairs = list(zip(nodes[:-1], nodes[1:]))
        if not pairs:
            raise ValueError(
                f"reconciled OBC {chain.chain_id!r} has no segments"
            )
        orientation = str(chain.orientation).strip().lower()
        if orientation == "forward":
            orientation = "source"
        if orientation not in {"source", "reverse"}:
            raise ValueError(
                f"unsupported reconciled OBC orientation {chain.orientation!r}"
            )
        segment_indices: list[int] = []
        for node_a, node_b in pairs:
            if node_a not in exterior_position or node_b not in exterior_position:
                raise ValueError(
                    f"reconciled OBC {chain.chain_id!r} leaves the exterior"
                )
            position_a = exterior_position[node_a]
            position_b = exterior_position[node_b]
            if position_b == (position_a + 1) % exterior_count:
                segment_index = position_a
                traversal = "source"
            elif position_a == (position_b + 1) % exterior_count:
                segment_index = position_b
                traversal = "reverse"
            else:
                raise ValueError(
                    f"reconciled OBC {chain.chain_id!r} is not contiguous"
                )
            if traversal != orientation:
                raise ValueError(
                    f"reconciled OBC {chain.chain_id!r} changes orientation"
                )
            segment_indices.append(int(segment_index))
        open_boundaries.append(
            SourceOpenBoundary(
                chain_id=str(chain.chain_id),
                kind=str(chain.kind),
                cyclic=bool(chain.cyclic),
                orientation=orientation,
                exterior_segment_indices=tuple(segment_indices),
            )
        )

    hard = np.asarray(
        boundary.hard_anchor_mask
        if boundary.hard_anchor_mask is not None
        else np.zeros(len(boundary.xy), dtype=bool),
        dtype=bool,
    )
    open_segment_indices = {
        int(segment_index)
        for chain in open_boundaries
        for segment_index in chain.exterior_segment_indices
    }
    return replace(
        prepared,
        exterior_xy=np.asarray(boundary.xy[exterior], dtype=float),
        holes_xy=tuple(
            np.asarray(boundary.xy[chain], dtype=float)
            for chain in chains[1:]
        ),
        exterior_segment_kinds=tuple(
            "open" if position in open_segment_indices else "land"
            for position, _node_index in enumerate(exterior)
        ),
        hard_anchor_vertex_indices=tuple(
            position
            for position, node_index in enumerate(exterior)
            if hard[node_index]
        ),
        open_boundaries=tuple(open_boundaries),
    )


def _reconciliation_changed_obc_sequence(
    prepared: PreparedCase,
    boundary: BoundaryNodes,
) -> bool:
    """Return whether reconciliation changed any forcing-chain node sequence.

    The result is deliberately conservative: missing reconciliation lineage is
    treated as a changed forcing sequence instead of inferring compatibility
    from node counts alone.
    """

    original = list(prepared.open_boundaries)
    delivered = list(boundary.open_boundaries or [])
    if len(original) != len(delivered):
        return True
    if not original:
        return False
    metadata = boundary.metadata or {}
    inserted = np.asarray(
        metadata.get("reconciliation_inserted", []),
        dtype=bool,
    )
    source_node = np.asarray(
        metadata.get(
            "reconciliation_source_node_index_zero_based",
            [],
        ),
        dtype=int,
    )
    if len(inserted) != len(boundary.xy) or len(source_node) != len(
        boundary.xy
    ):
        return True
    for source, chain in zip(original, delivered):
        source_orientation = str(source.orientation).strip().lower()
        chain_orientation = str(chain.orientation).strip().lower()
        if chain_orientation == "forward":
            chain_orientation = "source"
        delivered_indices = np.asarray(chain.node_indices, dtype=int)
        if (
            np.any(delivered_indices < 0)
            or np.any(delivered_indices >= len(boundary.xy))
        ):
            return True
        expected_nodes = _source_obc_node_indices(
            source,
            len(prepared.exterior_xy),
        )
        exact_source_nodes = tuple(
            int(value)
            for value in source_node[delivered_indices]
            if int(value) >= 0
        )
        if (
            str(source.chain_id) != str(chain.chain_id)
            or str(source.kind) != str(chain.kind)
            or bool(source.cyclic) != bool(chain.cyclic)
            or source_orientation != chain_orientation
            or bool(np.any(inserted[delivered_indices]))
            or exact_source_nodes != expected_nodes
        ):
            return True
    return False


def _size_field_config(
    config: PortfolioCaseConfig,
) -> SizeFieldConfig:
    if float(config.land_spacing_m) > float(config.maximum_size_m):
        raise ValueError(
            "budget-selected land/interior spacing exceeds maximum_size_m; "
            "increase --maximum-size-m so the two-dimensional raster can "
            "honor h_u without silently lowering its budget-compatible floor"
        )
    return SizeFieldConfig(
        land_spacing_m=float(config.land_spacing_m),
        open_spacing_m=float(config.open_spacing_m),
        max_size_m=float(config.maximum_size_m),
        interior_min_size_m=float(config.land_spacing_m),
        gradation=float(config.gradation),
        slope_elements=float(config.slope_elements),
        coastal_distance_m=float(config.coastal_distance_m),
        hydraulic_elements_across_min=float(
            config.hydraulic_elements_across_min
        ),
        hydraulic_elements_across_max=float(
            config.hydraulic_elements_across_max
        ),
        hydraulic_max_width_m=float(config.hydraulic_max_width_m),
        hydraulic_bank_angle_deg=float(config.hydraulic_bank_angle_deg),
        hydraulic_longitudinal_gradation=float(
            config.hydraulic_longitudinal_gradation
        ),
        obc_hold_distance_m=float(config.obc_hold_distance_m),
        obc_transition_distance_m=float(config.obc_transition_distance_m),
        target_timestep_s=str(config.target_timestep_s),
    )


def _reconcile_boundary_and_size_field(
    size_bathymetry: Any,
    source_boundary: BoundaryNodes,
    field_config: SizeFieldConfig,
    config: PortfolioCaseConfig,
) -> tuple[
    BoundaryNodes,
    SizeField,
    ProjectedSizeSampler | BoundaryTraceSizeSampler,
    dict[str, Any],
]:
    """Solve the common boundary/field contract before any triangulation."""

    policy = BoundarySizeReconciliationConfig(
        target_metric_edge=float(config.boundary_metric_edge),
        compatibility_factor=float(
            config.boundary_field_compatibility_factor
        ),
        maximum_spacing_gradient=float(config.gradation),
        maximum_boundary_l_over_h=1.55,
        enforce_sampled_field_compatibility=False,
        target_combination=str(config.boundary_target_combination),
        sampler_id="fvcom_size_field_v4_projected_sampler",
    )
    current_field = build_size_field(
        size_bathymetry,
        source_boundary,
        field_config,
    )
    current_raster_sampler = ProjectedSizeSampler(
        current_field,
        source_boundary.projection,
    )
    # The first pass must already see the immutable source-chord scale.  If it
    # starts from raster-cell-center interpolation alone, the reconciliation
    # can erase a fine realized boundary target before the trace exists.
    current_sampler = (
        BoundaryTraceSizeSampler(
            current_raster_sampler,
            source_boundary,
            gradation=float(config.gradation),
            samples_per_target=float(
                config.boundary_trace_samples_per_target
            ),
            nearest_sample_count=int(
                config.boundary_trace_nearest_sample_count
            ),
        )
        if bool(config.boundary_geometry_continuity)
        else current_raster_sampler
    )
    final_boundary = source_boundary
    final_field = current_field
    final_sampler = current_sampler
    final_audit: dict[str, Any] = {
        "schema_version": "fvcom_reconciled_boundary_final_field_audit_v1",
        "status": "not_run",
        "passed": False,
        "failure_taxonomy": ["boundary_reconciliation_not_run"],
    }
    iteration_reports: list[dict[str, Any]] = []
    converged_iteration: int | None = None

    for iteration in range(
        1,
        int(config.boundary_reconciliation_max_iterations) + 1,
    ):
        # Always rebase on the authoritative upstream boundary.  Feeding a
        # previously inserted boundary back into the reconciler would silently
        # replace its source-segment lineage.
        reconciled = reconcile_boundary_size_field(
            source_boundary,
            current_sampler,
            config=policy,
        )
        candidate_boundary = reconciled.boundary
        candidate_field = build_size_field(
            size_bathymetry,
            candidate_boundary,
            field_config,
        )
        candidate_raster_sampler = ProjectedSizeSampler(
            candidate_field,
            candidate_boundary.projection,
        )
        candidate_sampler = (
            BoundaryTraceSizeSampler(
                candidate_raster_sampler,
                candidate_boundary,
                gradation=float(config.gradation),
                samples_per_target=float(
                    config.boundary_trace_samples_per_target
                ),
                nearest_sample_count=int(
                    config.boundary_trace_nearest_sample_count
                ),
            )
            if bool(config.boundary_geometry_continuity)
            else candidate_raster_sampler
        )
        post_rebuild = audit_reconciled_boundary_size_field(
            candidate_boundary,
            candidate_sampler,
            config=policy,
        )
        iteration_reports.append(
            {
                "iteration": int(iteration),
                "input_field_node_budget_estimate": dict(
                    current_field.report.get("node_budget_estimate", {})
                ),
                "reconciliation_against_input_field": reconciled.audit,
                "rebuilt_field_node_budget_estimate": dict(
                    candidate_field.report.get("node_budget_estimate", {})
                ),
                "post_rebuild_boundary_field_audit": post_rebuild,
            }
        )
        final_boundary = candidate_boundary
        final_field = candidate_field
        final_sampler = candidate_sampler
        final_audit = post_rebuild
        if bool(reconciled.audit.get("passed")) and bool(
            post_rebuild.get("passed")
        ):
            converged_iteration = int(iteration)
            break
        current_field = candidate_field
        current_sampler = candidate_sampler

    passed = converged_iteration is not None
    if isinstance(final_sampler, BoundaryTraceSizeSampler):
        final_field.report["boundary_trace_sampler"] = dict(
            final_sampler.report
        )
    report = {
        "schema_version": "fvcom_boundary_size_fixed_point_v1",
        "status": "pass" if passed else "needs_review",
        "passed": bool(passed),
        "method": (
            (
                "authoritative_source_geometry_trace_resampling_plus_"
                "rebuilt_field_fixed_point"
            )
            if bool(config.boundary_geometry_continuity)
            else (
                "authoritative_source_resampling_plus_rebuilt_field_"
                "fixed_point"
            )
        ),
        "source_boundary_node_count": int(len(source_boundary.xy)),
        "reconciled_boundary_node_count": int(len(final_boundary.xy)),
        "inserted_boundary_node_count": int(
            len(final_boundary.xy) - len(source_boundary.xy)
        ),
        "source_constraint_chain_count": int(
            len(source_boundary.constraint_chains)
        ),
        "reconciled_constraint_chain_count": int(
            len(final_boundary.constraint_chains)
        ),
        "source_open_boundary_count": int(
            len(source_boundary.open_boundaries or [])
        ),
        "reconciled_open_boundary_count": int(
            len(final_boundary.open_boundaries or [])
        ),
        "maximum_iterations": int(
            config.boundary_reconciliation_max_iterations
        ),
        "iterations_executed": int(len(iteration_reports)),
        "converged_iteration": converged_iteration,
        "policy": asdict(policy),
        "iterations": iteration_reports,
        "final_boundary_field_audit": final_audit,
        "failure_taxonomy": (
            []
            if passed
            else sorted(
                set(
                    list(final_audit.get("failure_taxonomy", []))
                    + ["boundary_size_fixed_point_not_converged"]
                )
            )
        ),
        "lineage_policy": (
            "every pass starts from the immutable upstream boundary; every "
            "inserted node maps to one upstream segment and interpolation weight"
        ),
        "method_scope": (
            (
                "boundary targets follow the sampled, already "
                "wet-domain-gradated fvcom_size_field_v4"
                if policy.target_combination == "sampled_field"
                else "boundary targets are the pointwise minimum of the "
                "source target and sampled fvcom_size_field_v4"
            )
            + (
                "; every pass includes the deterministic realized-geometry "
                "boundary trace before resampling; the trace uses straight "
                "Euclidean distance and is not a barrier-aware wet-distance "
                "min-plus solve"
                if bool(config.boundary_geometry_continuity)
                else (
                    "; realized-geometry boundary trace is explicitly "
                    "disabled for this reproducibility run"
                )
            )
        ),
        # Freeze this evidence before preflight and candidate-local resets
        # mutate the live sampler's operational counters.
        "boundary_trace_sampler": (
            dict(final_sampler.report)
            if isinstance(final_sampler, BoundaryTraceSizeSampler)
            else None
        ),
    }
    return final_boundary, final_field, final_sampler, report


def _sampler_adjusted_node_budget(
    size_field: SizeField,
    boundary: BoundaryNodes,
    sampler: ProjectedSizeSampler | BoundaryTraceSizeSampler,
    *,
    chunk_size: int = 50_000,
) -> dict[str, Any]:
    """Conservatively integrate the final callback over active raster cells.

    A center-only sample can miss the trace minimum at a shoreline inside the
    cell. The raster component is bounded below by the minimum of neighboring
    *active* centers. The min-plus trace is ``gradation``-Lipschitz, so its
    per-subcell lower bound is the exact subcell-center value minus a
    conservative subcell radius, never below the global trace target minimum.
    Cells refine adaptively only when that release is material relative to the
    local target. Their pointwise minimum is therefore no coarser than the
    callback anywhere in each charged subcell without spreading an isolated
    fine trace point over a complete coarse raster cell.
    """

    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    raster = np.asarray(size_field.size, dtype=float)
    active = (
        np.asarray(size_field.coverage_mask, dtype=bool)
        & np.asarray(size_field.domain_mask, dtype=bool)
        & np.isfinite(raster)
        & (raster > 0.0)
    )
    flat_active = np.flatnonzero(active.ravel())
    adjusted = raster.copy()
    finite_raster = np.where(
        active,
        raster,
        np.inf,
    )
    raster_stencil_lower_bound = minimum_filter(
        finite_raster,
        size=3,
        mode="constant",
        cval=np.inf,
    )
    lon_axis = np.asarray(size_field.lon, dtype=float)
    lat_axis = np.asarray(size_field.lat, dtype=float)
    maximum_lon_step = float(np.max(np.diff(lon_axis)))
    maximum_lat_step = float(np.max(np.diff(lat_axis)))
    # 111.32 km/degree overestimates longitude distance away from the equator;
    # the 5% factor also covers the regional projected-coordinate scale.
    cell_radius_upper_bound_m = float(
        0.5
        * 111_320.0
        * np.hypot(maximum_lon_step, maximum_lat_step)
        * 1.05
    )
    lon_grid, lat_grid = np.meshgrid(
        lon_axis,
        lat_axis,
    )
    callback_center = np.full(raster.shape, np.inf, dtype=float)
    trace_center = np.full(raster.shape, np.inf, dtype=float)
    reduced_count = 0
    maximum_reduction = 0.0
    for begin in range(0, len(flat_active), int(chunk_size)):
        indices = flat_active[begin : begin + int(chunk_size)]
        lonlat = np.column_stack(
            [
                lon_grid.ravel()[indices],
                lat_grid.ravel()[indices],
            ]
        )
        xy = project_points(lonlat, boundary.projection)
        if isinstance(sampler, BoundaryTraceSizeSampler):
            base_values, trace_values = sampler.sample_components_xy(xy)
            values = np.minimum(base_values, trace_values)
            trace_center.ravel()[indices] = trace_values
        else:
            values = np.asarray(sampler.sample_xy(xy), dtype=float)
        original = raster.ravel()[indices]
        values = np.minimum(values, original)
        callback_center.ravel()[indices] = values
        trace_reduction = original - values
        reduced_count += int(
            np.count_nonzero(trace_reduction > 1.0e-9)
        )
        maximum_reduction = max(
            maximum_reduction,
            float(np.max(trace_reduction, initial=0.0)),
        )
    subdivision_count_by_axis: dict[int, int] = {1: int(len(flat_active))}
    subcell_trace_query_count = 0
    final_values = raster_stencil_lower_bound.ravel()[flat_active].copy()
    if isinstance(sampler, BoundaryTraceSizeSampler):
        raster_lower = raster_stencil_lower_bound.ravel()[flat_active]
        center_trace = trace_center.ravel()[flat_active]
        release = float(
            sampler.gradation * cell_radius_upper_bound_m
        )
        candidate = (
            np.maximum(
                center_trace - release,
                sampler.minimum_trace_target_m,
            )
            < raster_lower
        )
        local_scale = np.minimum(center_trace, raster_lower)
        target_release = np.maximum(0.25 * local_scale, 0.5)
        needed = np.maximum(
            1,
            np.ceil(release / target_release).astype(int),
        )
        powers = np.asarray([1, 2, 4, 8, 16, 32], dtype=int)
        selected_power = powers[
            np.minimum(
                np.searchsorted(powers, needed, side="left"),
                len(powers) - 1,
            )
        ]
        selected_power[~candidate] = 1
        subdivision_count_by_axis = {
            int(value): int(np.count_nonzero(selected_power == value))
            for value in powers
            if np.any(selected_power == value)
        }
        lon_width = np.abs(np.gradient(lon_axis))
        lat_width = np.abs(np.gradient(lat_axis))
        for subdivisions in powers:
            group_positions = np.flatnonzero(
                selected_power == int(subdivisions)
            )
            if not len(group_positions):
                continue
            group_indices = flat_active[group_positions]
            if int(subdivisions) == 1:
                lower = np.maximum(
                    center_trace[group_positions] - release,
                    sampler.minimum_trace_target_m,
                )
                final_values[group_positions] = np.minimum(
                    raster_lower[group_positions],
                    lower,
                )
                continue
            fractions = (
                (np.arange(int(subdivisions), dtype=float) + 0.5)
                / float(subdivisions)
                - 0.5
            )
            offset_x, offset_y = np.meshgrid(fractions, fractions)
            offset_x = offset_x.ravel()
            offset_y = offset_y.ravel()
            samples_per_cell = int(subdivisions) ** 2
            cells_per_chunk = max(
                1,
                int(chunk_size) // samples_per_cell,
            )
            subcell_radius = (
                cell_radius_upper_bound_m / float(subdivisions)
            )
            for begin in range(0, len(group_indices), cells_per_chunk):
                indices = group_indices[
                    begin : begin + cells_per_chunk
                ]
                positions = group_positions[
                    begin : begin + cells_per_chunk
                ]
                rows, columns = np.unravel_index(indices, raster.shape)
                query_lon = (
                    lon_axis[columns, None]
                    + lon_width[columns, None] * offset_x[None, :]
                )
                query_lat = (
                    lat_axis[rows, None]
                    + lat_width[rows, None] * offset_y[None, :]
                )
                lonlat = np.column_stack(
                    [query_lon.ravel(), query_lat.ravel()]
                )
                xy = project_points(lonlat, boundary.projection)
                trace_values = sampler.sample_trace_xy(xy).reshape(
                    len(indices),
                    samples_per_cell,
                )
                trace_lower = np.maximum(
                    trace_values
                    - sampler.gradation * subcell_radius,
                    sampler.minimum_trace_target_m,
                )
                cell_lower = np.minimum(
                    raster_stencil_lower_bound.ravel()[indices, None],
                    trace_lower,
                )
                mean_inverse_square = np.mean(
                    1.0 / np.square(cell_lower),
                    axis=1,
                )
                final_values[positions] = 1.0 / np.sqrt(
                    mean_inverse_square
                )
                subcell_trace_query_count += int(trace_values.size)
    adjusted.ravel()[flat_active] = final_values
    stencil_reduction = (
        callback_center.ravel()[flat_active] - final_values
    )
    stencil_reduced_count = int(
        np.count_nonzero(stencil_reduction > 1.0e-9)
    )
    maximum_stencil_reduction = float(
        np.max(stencil_reduction, initial=0.0)
    )
    budget = estimate_node_budget(
        np.asarray(size_field.lon, dtype=float),
        np.asarray(size_field.lat, dtype=float),
        adjusted,
        coverage_mask=np.asarray(size_field.coverage_mask, dtype=bool),
        domain_mask=np.asarray(size_field.domain_mask, dtype=bool),
    )
    budget.update(
        {
            "schema_version": "fvcom_sampler_adjusted_node_budget_v3",
            "callback_schema": (
                sampler.report.get("schema_version")
                if isinstance(sampler, BoundaryTraceSizeSampler)
                else "fvcom_raster_size_sampler_v1"
            ),
            "active_cell_center_sample_count": int(len(flat_active)),
            "trace_reduced_cell_count": int(reduced_count),
            "trace_reduced_cell_fraction": float(
                reduced_count / max(len(flat_active), 1)
            ),
            "maximum_trace_reduction_m": float(maximum_reduction),
            "stencil_support": (
                "active_three_by_three_raster_minimum_plus_adaptive_"
                "trace_lipschitz_subcell_lower_bounds"
            ),
            "inactive_cells_excluded_from_raster_stencil": True,
            "trace_lipschitz_lower_bound_applied": bool(
                isinstance(sampler, BoundaryTraceSizeSampler)
            ),
            "trace_cell_radius_upper_bound_m": float(
                cell_radius_upper_bound_m
            ),
            "trace_cell_radius_projection_safety_factor": 1.05,
            "trace_subdivision_count_by_axis": {
                str(key): int(value)
                for key, value in subdivision_count_by_axis.items()
            },
            "trace_maximum_subdivision_count_by_axis": 32,
            "trace_subcell_query_count": int(
                subcell_trace_query_count
            ),
            "trace_subdivision_release_fraction": 0.25,
            "stencil_reduced_cell_count": int(stencil_reduced_count),
            "stencil_reduced_cell_fraction": float(
                stencil_reduced_count / max(len(flat_active), 1)
            ),
            "maximum_stencil_reduction_m": float(
                maximum_stencil_reduction
            ),
            "chunk_size": int(chunk_size),
        }
    )
    return budget


def _preflight_report(
    size_field: SizeField,
    boundary: BoundaryNodes,
    config: PortfolioCaseConfig,
    *,
    sampler: ProjectedSizeSampler | BoundaryTraceSizeSampler | None = None,
    upstream_source_boundary_node_count: int | None = None,
    gmsh_boundary_node_count: int | None = None,
) -> dict[str, Any]:
    raster_interior = int(
        size_field.report.get("node_budget_estimate", {}).get(
            "estimated_interior_node_count",
            0,
        )
    )
    sampler_budget = (
        _sampler_adjusted_node_budget(
            size_field,
            boundary,
            sampler,
        )
        if sampler is not None
        else None
    )
    interior = int(
        sampler_budget.get(
            "estimated_interior_node_count",
            raster_interior,
        )
        if sampler_budget is not None
        else raster_interior
    )
    _seeds, front = boundary_front_seed_points(boundary)
    explicit_boundary = int(len(boundary.xy))
    common_boundary = max(
        explicit_boundary,
        int(gmsh_boundary_node_count or 0),
    )
    estimated_total = int(
        interior + common_boundary + int(front.get("accepted_count", 0))
    )
    return {
        "schema_version": "fvcom_portfolio_node_budget_preflight_v3",
        "canonical_size_field_schema": size_field.report.get("schema_version"),
        "estimated_interior_node_count": interior,
        "raster_only_estimated_interior_node_count": raster_interior,
        "final_callback_estimated_interior_node_count": interior,
        "final_callback_budget": sampler_budget,
        "upstream_source_boundary_node_count": int(
            upstream_source_boundary_node_count
            if upstream_source_boundary_node_count is not None
            else explicit_boundary
        ),
        "canonical_reconciled_boundary_node_count": explicit_boundary,
        # Backward-compatible name: in v2 this is the common canonical
        # boundary, which can be denser than the upstream source.
        "explicit_source_boundary_node_count": explicit_boundary,
        "gmsh_measured_boundary_node_count": gmsh_boundary_node_count,
        "gmsh_common_boundary_lock_passed": bool(
            gmsh_boundary_node_count is None
            or int(gmsh_boundary_node_count) == explicit_boundary
        ),
        "common_boundary_node_count": common_boundary,
        "boundary_front_seed_count": int(front.get("accepted_count", 0)),
        "estimated_total_node_count": estimated_total,
        "preflight_node_limit": int(config.preflight_node_limit),
        "hard_node_limit": int(config.hard_node_limit),
        "passed": bool(
            estimated_total <= config.preflight_node_limit
            and (
                gmsh_boundary_node_count is None
                or int(gmsh_boundary_node_count) == explicit_boundary
            )
        ),
        "boundary_front": front,
    }


def _sample_depths_strict(
    prepared: PreparedCase,
    nodes_lonlat: np.ndarray,
) -> np.ndarray:
    bathy = prepared.bathymetry
    interpolator = RegularGridInterpolator(
        (
            np.asarray(bathy.lat, dtype=float),
            np.asarray(bathy.lon, dtype=float),
        ),
        np.asarray(bathy.depth, dtype=float),
        bounds_error=False,
        fill_value=np.nan,
    )
    values = np.asarray(nodes_lonlat, dtype=float)
    depths = np.asarray(
        interpolator(np.column_stack([values[:, 1], values[:, 0]])),
        dtype=float,
    )
    invalid = ~np.isfinite(depths) | (depths <= 0.0)
    if np.any(invalid):
        raise ValueError(
            "immutable bathymetry produced non-finite or non-positive raw "
            f"candidate depths at {int(np.count_nonzero(invalid))} node(s)"
        )
    return depths


def _clean_boundary_metadata(
    prepared: PreparedCase,
    boundary: BoundaryNodes,
    mesh: Any,
) -> dict[str, Any]:
    open_chains = (
        [mesh.open_boundary_nodes.tolist()]
        if len(prepared.open_boundaries) == 1
        else []
    )
    source_chains = [
        [int(index) + 1 for index in value.node_indices]
        for value in (boundary.open_boundaries or [])
    ]
    reconciled_unchanged = bool(
        len(open_chains) == len(source_chains)
        and all(actual == source for actual, source in zip(open_chains, source_chains))
    )
    reconciliation_changed_obc = _reconciliation_changed_obc_sequence(
        prepared,
        boundary,
    )
    source_nodes_retained = bool(
        len(mesh.nodes_xy) >= len(boundary.xy)
        and np.allclose(
            mesh.nodes_xy[: len(boundary.xy)],
            boundary.xy,
            rtol=0.0,
            atol=1.0e-7,
        )
    )
    boundary_policy_stratum = (
        "COMMON_LOCKED"
        if int(mesh.boundary_node_count) == int(len(boundary.xy))
        else "RECOVERY_NATIVE"
    )
    return {
        "schema_version": "fvcom_portfolio_boundary_delivery_v1",
        "generator": "clean_room_raw",
        "upstream_source_vertex_count": int(
            len(prepared.exterior_xy)
            + sum(len(values) for values in prepared.holes_xy)
        ),
        "source_vertex_count": int(len(boundary.xy)),
        "reconciled_boundary_vertex_count": int(len(boundary.xy)),
        "delivered_boundary_node_count": int(mesh.boundary_node_count),
        "boundary_discretization_mode": "constraint_midpoint_recovery",
        "boundary_policy_stratum": boundary_policy_stratum,
        "boundary_discretization_matched_to_source": bool(
            int(mesh.boundary_node_count) == int(len(boundary.xy))
        ),
        "all_source_vertices_retained": source_nodes_retained,
        "hard_anchor_count": int(np.count_nonzero(boundary.hard_anchor_mask)),
        "hard_anchors_retained": source_nodes_retained,
        "reconciled_obc_sequence_changed": bool(
            reconciliation_changed_obc
        ),
        "forcing_compatible": bool(
            reconciled_unchanged and not reconciliation_changed_obc
        ),
        "forcing_interpolation_performed": False,
        "constraint_chains_1based": [
            [int(index) + 1 for index in chain]
            for chain in mesh.constraint_chains
        ],
        "open_boundaries": [
            {
                "chain_id": source.chain_id,
                "kind": source.kind,
                "cyclic": bool(source.cyclic),
                "orientation": source.orientation,
                "source_node_ids": expected,
                "delivered_node_ids": actual,
                "source_sequence_unchanged": bool(
                    actual == expected and not reconciliation_changed_obc
                ),
            }
            for source, expected, actual in zip(
                prepared.open_boundaries,
                source_chains,
                open_chains,
            )
        ],
    }


def _run_clean_room_candidate(
    prepared: PreparedCase,
    boundary: BoundaryNodes,
    size_field: SizeField,
    config: PortfolioCaseConfig,
    sampler: ProjectedSizeSampler | BoundaryTraceSizeSampler | None = None,
) -> _CandidateMesh:
    maximum_interior = max(
        1,
        int(config.preflight_node_limit) - int(len(boundary.xy)),
    )
    mesh = generate_mesh(
        boundary,
        size_field,
        MeshConfig(
            refine_iterations=int(config.clean_room_refine_iterations),
            smooth_iterations=int(config.clean_room_smooth_iterations),
            max_interior_points=maximum_interior,
            adaptive_seed=bool(boundary.adaptive_resolution),
            regional_spring_relaxation=False,
            thin_triangle_repair=False,
            thin_repair_profile="none",
            area_transition_relaxation=False,
            conditioning_profile="none",
        ),
        size_sampler_xy=(
            sampler.sample_xy if sampler is not None else None
        ),
    )
    depths = _sample_depths_strict(prepared, mesh.nodes_lonlat)
    open_chains = (
        [mesh.open_boundary_nodes.tolist()]
        if len(prepared.open_boundaries) == 1
        else []
    )
    metadata = _clean_boundary_metadata(prepared, boundary, mesh)
    return _CandidateMesh(
        nodes_xy=np.asarray(mesh.nodes_xy, dtype=float),
        nodes_lonlat=np.asarray(mesh.nodes_lonlat, dtype=float),
        triangles_1based=np.asarray(mesh.triangles, dtype=int),
        depths=depths,
        constraint_chains_zero=[
            [int(value) for value in chain]
            for chain in mesh.constraint_chains
        ],
        open_boundary_chains_1based=open_chains,
        open_boundary_cyclic=[
            bool(value.cyclic) for value in prepared.open_boundaries
        ],
        constraint_report=dict(mesh.report.get("constraint_recovery") or {}),
        boundary_metadata=metadata,
        generator_report={
            "schema_version": "fvcom_portfolio_generator_report_v1",
            "backend": "scipy_delaunay_clean_room",
            "boundary_discretization_mode": "constraint_midpoint_recovery",
            "boundary_policy_stratum": str(
                metadata["boundary_policy_stratum"]
            ),
            "source_boundary_vertex_count": int(len(boundary.xy)),
            "delivered_boundary_node_count": int(mesh.boundary_node_count),
            "raw_stage": True,
            "common_conditioning_applied": False,
            "canonical_size_callback_used": bool(sampler is not None),
            "mesh_report": mesh.report,
        },
        extra_quality={},
    )


def _validate_gmsh_open_boundaries(
    prepared: PreparedCase,
    result: Any,
) -> None:
    expected = [
        (
            value.chain_id,
            value.kind,
            bool(value.cyclic),
            value.orientation,
            tuple(value.exterior_segment_indices),
        )
        for value in prepared.open_boundaries
    ]
    delivered = [
        (
            value.chain_id,
            value.kind,
            bool(value.cyclic),
            value.orientation,
            tuple(value.source_segment_indices),
        )
        for value in result.open_boundaries
    ]
    if delivered != expected:
        raise ValueError(
            "Gmsh delivered OBC chain identities/order different from source"
        )


def _run_gmsh_candidate(
    candidate_id: str,
    prepared: PreparedCase,
    sampler: ProjectedSizeSampler,
    output_dir: Path,
    *,
    boundary: BoundaryNodes | None = None,
) -> _CandidateMesh:
    from .gmsh_backend import GmshConfig, run_gmsh_attempt

    algorithm = int(GMSH_CANDIDATE_ALGORITHMS[candidate_id])
    mesher_prepared = (
        _prepared_case_on_boundary(prepared, boundary)
        if boundary is not None
        else prepared
    )
    geometry = _backend_geometry(mesher_prepared)
    # ``h_uniform_m`` remains a required legacy field but is not consulted
    # when the canonical projected callback is installed.
    result = run_gmsh_attempt(
        geometry,
        GmshConfig(
            h_uniform_m=1_000.0,
            algorithm=algorithm,
            canonical_size_callback=sampler,
            constant_field=not bool(prepared.open_boundaries),
            preserve_source_boundary_discretization=True,
            model_name=f"{prepared.manifest['case_id']}_{candidate_id}",
        ),
        output_dir / "raw_mesh.msh",
    )
    _validate_gmsh_open_boundaries(mesher_prepared, result)
    nodes_lonlat = unproject_points(result.nodes_xy, prepared.projection)
    depths = _sample_depths_strict(prepared, nodes_lonlat)
    lineage = _delivered_lineage_manifest(mesher_prepared, result)
    if boundary is not None:
        reconciliation_changed_obc = _reconciliation_changed_obc_sequence(
            prepared,
            boundary,
        )
        lineage["upstream_source_vertex_count"] = int(
            len(prepared.exterior_xy)
            + sum(len(values) for values in prepared.holes_xy)
        )
        lineage["reconciled_boundary_vertex_count"] = int(len(boundary.xy))
        lineage["reconciled_obc_sequence_changed"] = bool(
            reconciliation_changed_obc
        )
        lineage["forcing_compatible"] = bool(
            lineage.get("forcing_compatible", False)
            and not reconciliation_changed_obc
        )
        if reconciliation_changed_obc:
            for record in lineage.get("open_boundaries", []):
                record["source_sequence_unchanged"] = False
    lineage["delivered_boundary_node_count"] = int(
        result.boundary_node_count_1d
    )
    lineage["boundary_discretization_mode"] = str(
        result.boundary_discretization_mode
    )
    lineage["boundary_policy_stratum"] = "COMMON_LOCKED"
    lineage["boundary_discretization_matched_to_source"] = bool(
        int(result.boundary_node_count_1d)
        == int(lineage["source_vertex_count"])
    )
    expected_loop_lengths = [
        len(mesher_prepared.exterior_xy),
        *[len(values) for values in mesher_prepared.holes_xy],
    ]
    delivered_loop_lengths = [
        len(value.node_ids) for value in result.delivered_loops
    ]
    expected_common_boundary_count = int(
        len(boundary.xy)
        if boundary is not None
        else lineage["source_vertex_count"]
    )
    if (
        int(result.boundary_node_count_1d)
        != expected_common_boundary_count
        or delivered_loop_lengths != expected_loop_lengths
        or not bool(lineage["all_source_vertices_retained"])
        or not bool(lineage["boundary_discretization_matched_to_source"])
    ):
        raise ValueError(
            "COMMON_LOCKED boundary invariant failed: measured/delivered "
            "boundary nodes do not exactly match the canonical reconciled loops"
        )
    return _CandidateMesh(
        nodes_xy=np.asarray(result.nodes_xy, dtype=float),
        nodes_lonlat=nodes_lonlat,
        triangles_1based=np.asarray(result.triangles_1based, dtype=int),
        depths=depths,
        constraint_chains_zero=[
            [int(value) - 1 for value in delivered.node_ids]
            for delivered in result.delivered_loops
        ],
        open_boundary_chains_1based=[
            [int(value) for value in delivered.node_ids]
            for delivered in result.open_boundaries
        ],
        open_boundary_cyclic=[
            bool(value.cyclic) for value in result.open_boundaries
        ],
        constraint_report={
            "boundary_constraint_recovered": bool(
                lineage["all_source_vertices_retained"]
                and lineage["hard_anchors_retained"]
                and len(result.delivered_loops) == 1 + len(prepared.holes_xy)
            ),
            "all_source_vertices_retained": bool(
                lineage["all_source_vertices_retained"]
            ),
            "hard_anchors_retained": bool(lineage["hard_anchors_retained"]),
            "expected_loop_count": int(1 + len(prepared.holes_xy)),
            "delivered_loop_count": int(len(result.delivered_loops)),
        },
        boundary_metadata=lineage,
        generator_report={
            "schema_version": "fvcom_portfolio_generator_report_v1",
            "backend": "gmsh",
            "gmsh_version": result.gmsh_version,
            "algorithm": result.algorithm,
            "algorithm_name": result.algorithm_name,
            "size_field_mode": result.size_field_mode,
            "boundary_discretization_mode": result.boundary_discretization_mode,
            "boundary_policy_stratum": "COMMON_LOCKED",
            "source_boundary_vertex_count": int(
                lineage["source_vertex_count"]
            ),
            "delivered_boundary_node_count": int(
                result.boundary_node_count_1d
            ),
            "boundary_node_count_1d": int(result.boundary_node_count_1d),
            "raw_stage": True,
            "common_conditioning_applied": False,
            "native_smoothing_steps": 8,
            "thread_count": 1,
            "random_seed": 1,
            "algorithm_fallback_enabled": False,
        },
        extra_quality={
            "gmsh_native_quality": _native_quality_report(result),
        },
        raw_msh_path=result.msh_path,
        logger_output=result.logger_output,
    )


def _roundtrip_report(
    path: Path,
    prepared: PreparedCase,
    mesh: _CandidateMesh,
    candidate_id: str,
) -> dict[str, Any]:
    write_2dm(
        path,
        mesh.nodes_lonlat,
        mesh.depths,
        mesh.triangles_1based,
        np.empty(0, dtype=int),
        mesh_name=f"{prepared.manifest['case_id']}_{candidate_id}_raw",
        open_boundary_chains=mesh.open_boundary_chains_1based,
        open_boundary_ids=range(
            1,
            len(mesh.open_boundary_chains_1based) + 1,
        ),
    )
    parsed = read_2dm(path)
    parsed_xy = project_points(parsed.nodes_lonlat, prepared.projection)
    shifts = np.linalg.norm(parsed_xy - mesh.nodes_xy, axis=1)
    maximum_shift = float(np.max(shifts)) if len(shifts) else 0.0
    chains_equal = bool(
        len(parsed.open_boundary_chains)
        == len(mesh.open_boundary_chains_1based)
        and all(
            np.array_equal(actual, np.asarray(expected, dtype=int))
            for actual, expected in zip(
                parsed.open_boundary_chains,
                mesh.open_boundary_chains_1based,
            )
        )
    )
    triangle_equal = bool(
        np.array_equal(parsed.triangles, mesh.triangles_1based)
    )
    node_count_equal = bool(len(parsed.nodes_lonlat) == len(mesh.nodes_xy))
    zero_nodestring_contract = bool(
        len(prepared.open_boundaries) != 0
        or len(parsed.open_boundary_chains) == 0
    )
    return {
        "schema_version": "fvcom_portfolio_2dm_roundtrip_v1",
        "passed": bool(
            chains_equal
            and triangle_equal
            and node_count_equal
            and zero_nodestring_contract
            and maximum_shift < 0.01
        ),
        "open_boundary_chain_count": int(len(parsed.open_boundary_chains)),
        "nodestring_ids": list(parsed.open_boundary_ids),
        "open_boundary_order_exact": chains_equal,
        "triangle_connectivity_exact": triangle_equal,
        "node_count_exact": node_count_equal,
        "zero_nodestring_contract_passed": zero_nodestring_contract,
        "maximum_projected_coordinate_shift_m": maximum_shift,
        "coordinate_shift_threshold_m": 0.01,
    }


def _quality_report(
    prepared: PreparedCase,
    boundary: BoundaryNodes,
    mesh: _CandidateMesh,
    sampler: ProjectedSizeSampler,
    roundtrip: dict[str, Any],
    config: PortfolioCaseConfig,
    input_bundle_sha256: str,
) -> dict[str, Any]:
    triangles_zero = np.asarray(mesh.triangles_1based, dtype=int) - 1
    centroids = np.mean(mesh.nodes_xy[triangles_zero], axis=1)
    target_sizes = sampler.sample_xy(centroids)
    legacy_open = (
        np.asarray(mesh.open_boundary_chains_1based[0], dtype=int)
        if len(mesh.open_boundary_chains_1based) == 1
        else np.empty(0, dtype=int)
    )
    quality = evaluate_mesh_quality(
        mesh.nodes_xy,
        mesh.depths,
        mesh.triangles_1based,
        legacy_open,
        mesh.constraint_report,
        constraint_chains=mesh.constraint_chains_zero,
        open_boundary_chains=mesh.open_boundary_chains_1based,
        open_boundary_cyclic=mesh.open_boundary_cyclic,
        require_open_boundary=bool(prepared.open_boundaries),
        expected_open_boundary_count=int(
            prepared.manifest["boundary"]["expected_open_boundary_count"]
        ),
        enforce_size_error=True,
        enforce_no_unused_nodes=True,
        target_size_by_triangle=target_sizes,
    )
    boundary_targets = np.full(len(mesh.nodes_xy), np.nan, dtype=float)
    if (
        len(mesh.nodes_xy) >= len(boundary.xy)
        and np.allclose(
            mesh.nodes_xy[: len(boundary.xy)],
            boundary.xy,
            rtol=0.0,
            atol=1.0e-7,
        )
    ):
        boundary_targets[: len(boundary.xy)] = np.asarray(
            boundary.target_spacing_m,
            dtype=float,
        )
    delivered_loops = mesh.boundary_metadata.get("loops", [])
    if len(delivered_loops) == len(boundary.constraint_chains):
        for chain, delivered in zip(
            boundary.constraint_chains,
            delivered_loops,
        ):
            records = delivered.get("nodes", [])
            if len(records) != len(chain):
                continue
            for boundary_index, record in zip(chain, records):
                mesh_index = int(record["mesh_node_id"]) - 1
                if 0 <= mesh_index < len(boundary_targets):
                    boundary_targets[mesh_index] = float(
                        boundary.target_spacing_m[int(boundary_index)]
                    )
    edge_size = audit_edge_target_sizes(
        mesh.nodes_xy,
        triangles_zero,
        [
            {
                "chain_id": f"constraint_{index:03d}",
                "nodes": chain,
                "cyclic": True,
            }
            for index, chain in enumerate(
                mesh.constraint_chains_zero,
                start=1,
            )
        ],
        sampler.sample_xy,
        boundary_target_by_node=boundary_targets,
        transition_graph_rings=2,
        thresholds=(1.55, 2.0),
        boundary_gradation_limit=float(config.gradation),
        boundary_field_ratio_limit=float(
            config.boundary_field_compatibility_factor
        ),
    )
    external_failures: list[str] = []
    if len(mesh.nodes_xy) > int(config.hard_node_limit):
        external_failures.append("hard_node_cap_exceeded")
    if not roundtrip["passed"]:
        external_failures.append("sms_2dm_roundtrip_failed")
    if not mesh.boundary_metadata.get("all_source_vertices_retained", False):
        external_failures.append("source_boundary_vertex_lost")
    if not mesh.boundary_metadata.get("hard_anchors_retained", False):
        external_failures.append("protected_anchor_lost")
    if (
        str(
            mesh.boundary_metadata.get(
                "boundary_policy_stratum",
                "",
            )
        )
        == "COMMON_LOCKED"
        and not mesh.boundary_metadata.get(
            "boundary_discretization_matched_to_source",
            False,
        )
    ):
        external_failures.append("common_locked_boundary_mismatch")
    edge_all = edge_size["triangle_l_over_h"]["all"]
    edge_p95 = edge_all["quantiles"]["p95"]
    edge_maximum = edge_all["maximum"]
    if edge_all["invalid_count"]:
        external_failures.append("edge_aware_target_size_invalid")
    if edge_p95 is None or float(edge_p95) > 1.55:
        external_failures.append("edge_aware_target_size_p95_exceeded")
    if edge_maximum is None or float(edge_maximum) > 2.0:
        external_failures.append("edge_aware_target_size_maximum_exceeded")
    if not bool(edge_size.get("passed", False)):
        external_failures.extend(
            str(value)
            for value in edge_size.get(
                "failure_taxonomy",
                ["boundary_field_interface_audit_failed"],
            )
        )
    if external_failures:
        quality["failure_taxonomy"] = sorted(
            set(quality["failure_taxonomy"] + external_failures)
        )
        quality["accepted"] = False
    quality["raw_stage"] = True
    quality["common_conditioning_applied"] = False
    quality["input_bundle_sha256"] = str(input_bundle_sha256)
    quality["canonical_size_field_schema"] = "fvcom_size_field_v4"
    quality["canonical_size_sampling"] = dict(sampler.report)
    quality["edge_aware_size_error"] = edge_size
    quality["sms_2dm_roundtrip"] = roundtrip
    delivered_budget = delivered_node_budget_report(
        len(mesh.nodes_xy),
        config.hard_node_limit,
    )
    # Preserve the archived portfolio aliases while making the shared
    # delivered-budget schema authoritative.
    delivered_budget["actual_node_count"] = int(len(mesh.nodes_xy))
    delivered_budget["hard_node_limit"] = int(config.hard_node_limit)
    quality["node_budget"] = delivered_budget
    quality.update(mesh.extra_quality)
    return quality


def _failed_candidate_artifacts(
    candidate_dir: Path,
    candidate_id: str,
    *,
    status: str,
    failure: str,
    input_bundle_sha256: str,
    exception: BaseException | None = None,
) -> dict[str, Any]:
    quality_path = _write_json(
        candidate_dir / "quality.json",
        {
            "schema_version": "fvcom_mesh_quality_v2",
            "candidate_id": candidate_id,
            "input_bundle_sha256": str(input_bundle_sha256),
            "raw_stage": True,
            "accepted": False,
            "failure_taxonomy": [failure],
            "evaluation_completed": False,
        },
    )
    manifest = {
        "schema_version": "fvcom_mesher_candidate_manifest_v1",
        "candidate_id": candidate_id,
        "input_bundle_sha256": str(input_bundle_sha256),
        "status": status,
        "raw_stage": True,
        "common_conditioning_applied": False,
        "failure_taxonomy": [failure],
        "quality_accepted": False,
        "artifacts": {
            "quality_json": {
                "path": str(quality_path),
                "sha256": file_sha256(quality_path),
            }
        },
    }
    if exception is not None:
        manifest["exception"] = {
            "type": type(exception).__name__,
            "message": str(exception),
            "traceback": traceback.format_exc(),
        }
    manifest_path = _write_json(candidate_dir / "candidate_manifest.json", manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


def _execute_candidate(
    candidate_id: str,
    candidate_dir: Path,
    prepared: PreparedCase,
    boundary: BoundaryNodes,
    size_field: SizeField,
    sampler: ProjectedSizeSampler | BoundaryTraceSizeSampler,
    config: PortfolioCaseConfig,
    input_bundle_sha256: str,
) -> dict[str, Any]:
    execution_started_at_utc = _utc_now()
    execution_clock = time.perf_counter()
    if candidate_id == "clean_room_raw":
        mesh = _run_clean_room_candidate(
            prepared,
            boundary,
            size_field,
            config,
            sampler=sampler,
        )
    else:
        mesh = _run_gmsh_candidate(
            candidate_id,
            prepared,
            sampler,
            candidate_dir,
            boundary=boundary,
        )
    boundary_path = _write_json(
        candidate_dir / "boundary_metadata.json",
        mesh.boundary_metadata,
    )
    generator_path = _write_json(
        candidate_dir / "generator_report.json",
        mesh.generator_report,
    )
    if mesh.logger_output:
        logger_path = candidate_dir / "gmsh.log"
        logger_path.write_text(
            "\n".join(mesh.logger_output) + "\n",
            encoding="utf-8",
        )
    else:
        logger_path = None
    # Keep the leaf name short: regional output roots can already approach the
    # legacy Windows MAX_PATH limit, and a long case+algorithm filename caused
    # a completed Gmsh run to fail only when opening its 2DM artifact.
    output_2dm = candidate_dir / "raw_mesh.2dm"
    roundtrip = _roundtrip_report(
        output_2dm,
        prepared,
        mesh,
        candidate_id,
    )
    roundtrip_path = _write_json(
        candidate_dir / "roundtrip.json",
        roundtrip,
    )
    quality = _quality_report(
        prepared,
        boundary,
        mesh,
        sampler,
        roundtrip,
        config,
        input_bundle_sha256,
    )
    delivered_budget_path = _write_json(
        candidate_dir / "node_budget_delivered.json",
        quality["node_budget"],
    )
    quality_path = _write_json(candidate_dir / "quality.json", quality)
    execution_wall_seconds = float(time.perf_counter() - execution_clock)
    artifact_paths: dict[str, Path] = {
        "sms_2dm": output_2dm,
        "boundary_metadata": boundary_path,
        "generator_report": generator_path,
        "roundtrip": roundtrip_path,
        "node_budget_delivered": delivered_budget_path,
        "quality_json": quality_path,
    }
    if mesh.raw_msh_path is not None:
        artifact_paths["raw_msh_4_1"] = mesh.raw_msh_path
    if logger_path is not None:
        artifact_paths["gmsh_logger"] = logger_path
    status = "pass" if quality["accepted"] else "needs_review"
    manifest = {
        "schema_version": "fvcom_mesher_candidate_manifest_v1",
        "case_id": prepared.manifest["case_id"],
        "candidate_id": candidate_id,
        "input_bundle_sha256": str(input_bundle_sha256),
        "status": status,
        "execution_started_at_utc": execution_started_at_utc,
        "execution_wall_seconds": execution_wall_seconds,
        "completed_at_utc": _utc_now(),
        "raw_stage": True,
        "common_conditioning_applied": False,
        "node_count": int(len(mesh.nodes_xy)),
        "triangle_count": int(len(mesh.triangles_1based)),
        "source_boundary_vertex_count": int(
            mesh.boundary_metadata.get("source_vertex_count", 0)
        ),
        "delivered_boundary_node_count": int(
            mesh.boundary_metadata.get("delivered_boundary_node_count", 0)
        ),
        "boundary_discretization_mode": str(
            mesh.boundary_metadata.get(
                "boundary_discretization_mode",
                "unknown",
            )
        ),
        "boundary_policy_stratum": str(
            mesh.boundary_metadata.get(
                "boundary_policy_stratum",
                "unknown",
            )
        ),
        "boundary_discretization_matched_to_source": bool(
            mesh.boundary_metadata.get(
                "boundary_discretization_matched_to_source",
                False,
            )
        ),
        "open_boundary_chain_count": int(
            len(mesh.open_boundary_chains_1based)
        ),
        "quality_accepted": bool(quality["accepted"]),
        "failure_taxonomy": list(quality["failure_taxonomy"]),
        "forcing_compatible": bool(
            mesh.boundary_metadata.get("forcing_compatible", False)
        ),
        "artifacts": _hash_artifacts(artifact_paths),
    }
    manifest_path = _write_json(
        candidate_dir / "candidate_manifest.json",
        manifest,
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


def _build_input_bundle(
    output: Path,
    prepared: PreparedCase,
    boundary: BoundaryNodes,
    boundary_report: dict[str, Any],
    size_field: SizeField,
    size_field_config: SizeFieldConfig,
    config: PortfolioCaseConfig,
    preflight: dict[str, Any],
    readiness: dict[str, Any],
    *,
    source_boundary: BoundaryNodes | None = None,
    boundary_reconciliation: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    bundle = output / "input_bundle"
    bundle.mkdir(parents=True, exist_ok=False)
    case_snapshot = bundle / "case_manifest.snapshot.json"
    shutil.copy2(prepared.manifest_path, case_snapshot)
    if file_sha256(case_snapshot) != prepared.manifest_sha256:
        raise RuntimeError("case manifest snapshot hash mismatch")
    boundary_report_path = _write_json(
        bundle / "canonical_boundary.json",
        boundary_report,
    )
    boundary_geojson_path = _write_json(
        bundle / "canonical_boundary_nodes.geojson",
        boundary_nodes_geojson(boundary),
    )
    source_boundary_path = (
        _write_json(
            bundle / "upstream_source_boundary_nodes.geojson",
            boundary_nodes_geojson(source_boundary),
        )
        if source_boundary is not None
        else None
    )
    reconciliation_path = (
        _write_json(
            bundle / "boundary_size_reconciliation.json",
            boundary_reconciliation,
        )
        if boundary_reconciliation is not None
        else None
    )
    readiness_path = _write_json(bundle / "case_readiness.json", readiness)
    size_report_path = _write_json(
        bundle / "canonical_size_field_report.json",
        size_field.report,
    )
    preflight_path = _write_json(
        bundle / "node_budget_preflight.json",
        preflight,
    )
    size_nc, size_png, components_png = write_size_field(
        size_field,
        bundle / "canonical_size_field_v4.nc",
        bundle / "canonical_size_field_v4.png",
    )
    original_inputs = {
        "case_manifest_source": {
            "path": str(prepared.manifest_path),
            "sha256": prepared.manifest_sha256,
            "bytes": int(prepared.manifest_path.stat().st_size),
        },
        **{
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": int(path.stat().st_size),
            }
            for name, path in prepared.input_paths.items()
        },
    }
    bundle_artifact_paths = {
            "case_manifest_snapshot": case_snapshot,
            "canonical_boundary": boundary_report_path,
            "canonical_boundary_nodes": boundary_geojson_path,
            "case_readiness": readiness_path,
            "canonical_size_field_report": size_report_path,
            "node_budget_preflight": preflight_path,
            "canonical_size_field_netcdf": size_nc,
            "canonical_size_field_map": size_png,
            "canonical_size_field_components_map": components_png,
    }
    if source_boundary_path is not None:
        bundle_artifact_paths["upstream_source_boundary_nodes"] = (
            source_boundary_path
        )
    if reconciliation_path is not None:
        bundle_artifact_paths["boundary_size_reconciliation"] = (
            reconciliation_path
        )
    bundle_artifacts = _hash_artifacts(bundle_artifact_paths)
    input_bundle_sha256, hash_contract = _scientific_bundle_sha256(
        case_id=str(prepared.manifest["case_id"]),
        source_hashes=original_inputs,
        canonical_boundary_sha256=bundle_artifacts[
            "canonical_boundary_nodes"
        ]["sha256"],
        canonical_field_sha256=bundle_artifacts[
            "canonical_size_field_netcdf"
        ]["sha256"],
        projection_epsg=int(prepared.projection.epsg),
        portfolio_config=config,
        size_field_config=size_field_config,
        preflight=preflight,
    )
    manifest = {
        "schema_version": INPUT_BUNDLE_SCHEMA,
        "case_id": prepared.manifest["case_id"],
        "created_at_utc": _utc_now(),
        "immutable_raw_stage_input": True,
        "input_bundle_sha256": input_bundle_sha256,
        "input_bundle_hash_contract": hash_contract,
        "case_projection_epsg": int(prepared.projection.epsg),
        "canonical_size_field_schema": size_field.report.get("schema_version"),
        "canonical_size_field_method": size_field.report.get("method"),
        "boundary_size_reconciliation": (
            dict(boundary_reconciliation)
            if boundary_reconciliation is not None
            else None
        ),
        "size_field_config": asdict(size_field_config),
        "portfolio_config": asdict(config),
        "bathymetry_depth_policy": (
            "each raw candidate samples the immutable source bathymetry "
            "strictly at its delivered nodes"
        ),
        "original_input_hashes": original_inputs,
        "artifacts": bundle_artifacts,
    }
    manifest_path = _write_json(bundle / "input_bundle_manifest.json", manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest_path, manifest


def _nested_value(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _first_nested_value(
    payload: Mapping[str, Any],
    *paths: Sequence[str],
) -> Any:
    """Return the first present nested value, preserving falsey values."""

    for path in paths:
        value = _nested_value(payload, *path)
        if value is not None:
            return value
    return None


def _write_raw_metric_comparison(
    output: Path,
    candidate_manifests: Sequence[Mapping[str, Any]],
    routing: Mapping[str, Any],
    input_bundle_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Write an unranked, hard-gate-oriented raw metric comparison."""

    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for manifest in candidate_manifests:
        candidate_id = str(manifest["candidate_id"])
        capability = routing["candidates"][candidate_id]
        if not bool(capability["supported"]):
            unsupported.append(
                {
                    "candidate_id": candidate_id,
                    "status": manifest["status"],
                    "reasons": list(capability.get("reasons", [])),
                }
            )
            continue
        quality_artifact = manifest.get("artifacts", {}).get("quality_json", {})
        quality_path = Path(str(quality_artifact.get("path", "")))
        quality = (
            json.loads(quality_path.read_text(encoding="utf-8"))
            if quality_path.is_file()
            else {}
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "execution_status": manifest.get("status"),
                "execution_wall_seconds": manifest.get(
                    "execution_wall_seconds"
                ),
                "hard_gates_passed": bool(quality.get("accepted", False)),
                "node_count": quality.get("node_count"),
                "triangle_count": quality.get("triangle_count"),
                "source_boundary_vertex_count": manifest.get(
                    "source_boundary_vertex_count"
                ),
                "delivered_boundary_node_count": manifest.get(
                    "delivered_boundary_node_count"
                ),
                "boundary_discretization_mode": manifest.get(
                    "boundary_discretization_mode"
                ),
                "boundary_policy_stratum": manifest.get(
                    "boundary_policy_stratum"
                ),
                "boundary_discretization_matched_to_source": manifest.get(
                    "boundary_discretization_matched_to_source"
                ),
                "q_l3_sigma": _nested_value(
                    quality,
                    "oceanmesh_quality",
                    "q_l3_sigma",
                ),
                "q_min": _nested_value(
                    quality,
                    "oceanmesh_quality",
                    "q_min",
                ),
                "count_q_below_0_10": _nested_value(
                    quality,
                    "oceanmesh_quality",
                    "count_q_below_0_10",
                ),
                "min_angle_deg": quality.get("min_angle_deg"),
                "max_angle_deg": quality.get("max_angle_deg"),
                "max_bathymetric_slope": quality.get(
                    "max_bathymetric_slope"
                ),
                "max_adjacent_area_change": quality.get(
                    "max_adjacent_area_change"
                ),
                "max_node_valence": quality.get("max_node_valence"),
                "singly_connected_triangle_count": _nested_value(
                    quality,
                    "topology",
                    "singly_connected_triangle_count",
                ),
                "nonmanifold_edge_count": _nested_value(
                    quality,
                    "topology",
                    "nonmanifold_edge_count",
                ),
                "unused_node_count": _nested_value(
                    quality,
                    "topology",
                    "unused_node_count",
                ),
                "target_size_l_over_h_p95": _nested_value(
                    quality,
                    "size_error_l_over_h",
                    "quantiles",
                    "p95",
                ),
                "target_size_l_over_h_maximum": _nested_value(
                    quality,
                    "size_error_l_over_h",
                    "maximum",
                ),
                "centroid_target_size_l_over_h_p95": _nested_value(
                    quality,
                    "size_error_l_over_h",
                    "quantiles",
                    "p95",
                ),
                "centroid_target_size_l_over_h_maximum": _nested_value(
                    quality,
                    "size_error_l_over_h",
                    "maximum",
                ),
                "edge_target_size_l_over_h_p95": _nested_value(
                    quality,
                    "edge_aware_size_error",
                    "triangle_l_over_h",
                    "all",
                    "quantiles",
                    "p95",
                ),
                "edge_target_size_l_over_h_maximum": _nested_value(
                    quality,
                    "edge_aware_size_error",
                    "triangle_l_over_h",
                    "all",
                    "maximum",
                ),
                **{
                    f"{stratum}_edge_l_over_h_p95": _nested_value(
                        quality,
                        "edge_aware_size_error",
                        "edge_l_over_h",
                        stratum,
                        "quantiles",
                        "p95",
                    )
                    for stratum in (
                        "boundary",
                        "first_ring",
                        "transition",
                        "true_interior",
                    )
                },
                **{
                    f"{stratum}_edge_l_over_h_maximum": _nested_value(
                        quality,
                        "edge_aware_size_error",
                        "edge_l_over_h",
                        stratum,
                        "maximum",
                    )
                    for stratum in (
                        "boundary",
                        "first_ring",
                        "transition",
                        "true_interior",
                    )
                },
                **{
                    f"{stratum}_triangle_l_over_h_p95": _nested_value(
                        quality,
                        "edge_aware_size_error",
                        "triangle_l_over_h",
                        stratum,
                        "quantiles",
                        "p95",
                    )
                    for stratum in (
                        "boundary",
                        "first_ring",
                        "transition",
                        "true_interior",
                    )
                },
                **{
                    f"{stratum}_triangle_l_over_h_maximum": _nested_value(
                        quality,
                        "edge_aware_size_error",
                        "triangle_l_over_h",
                        stratum,
                        "maximum",
                    )
                    for stratum in (
                        "boundary",
                        "first_ring",
                        "transition",
                        "true_interior",
                    )
                },
                "boundary_field_interface_ratio_p95": _nested_value(
                    quality,
                    "edge_aware_size_error",
                    "boundary_field_interface",
                    "symmetric_ratio",
                    "quantiles",
                    "p95",
                ),
                "boundary_field_interface_ratio_maximum": _nested_value(
                    quality,
                    "edge_aware_size_error",
                    "boundary_field_interface",
                    "symmetric_ratio",
                    "maximum",
                ),
                "boundary_field_interface_ratio_limit_exceedance_count": (
                    _first_nested_value(
                        quality,
                        (
                            "edge_aware_size_error",
                            "boundary_field_interface",
                            "ratio_limit_exceedance_count",
                        ),
                        (
                            "edge_aware_size_error",
                            "boundary_field_interface",
                            "factor_two_exceedance_count",
                        ),
                    )
                ),
                # Backward-compatible comparison alias for archived readers.
                "boundary_field_interface_factor_two_exceedance_count": (
                    _first_nested_value(
                        quality,
                        (
                            "edge_aware_size_error",
                            "boundary_field_interface",
                            "ratio_limit_exceedance_count",
                        ),
                        (
                            "edge_aware_size_error",
                            "boundary_field_interface",
                            "factor_two_exceedance_count",
                        ),
                    )
                ),
                "boundary_first_ring_continuity_passed": _nested_value(
                    quality,
                    "edge_aware_size_error",
                    "boundary_first_ring_realized_continuity",
                    "passed",
                ),
                "boundary_first_ring_continuity_p95": _nested_value(
                    quality,
                    "edge_aware_size_error",
                    "boundary_first_ring_realized_continuity",
                    "global",
                    "symmetric_ratio",
                    "quantiles",
                    "p95",
                ),
                "boundary_first_ring_continuity_maximum": _nested_value(
                    quality,
                    "edge_aware_size_error",
                    "boundary_first_ring_realized_continuity",
                    "global",
                    "symmetric_ratio",
                    "maximum",
                ),
                "boundary_first_ring_chain_p95_exceedance_count": (
                    _nested_value(
                        quality,
                        "edge_aware_size_error",
                        "boundary_first_ring_realized_continuity",
                        "global",
                        "chain_p95_exceedance_count",
                    )
                ),
                "boundary_first_ring_chain_maximum_exceedance_count": (
                    _nested_value(
                        quality,
                        "edge_aware_size_error",
                        "boundary_first_ring_realized_continuity",
                        "global",
                        "chain_maximum_exceedance_count",
                    )
                ),
                "roundtrip_passed": _nested_value(
                    quality,
                    "sms_2dm_roundtrip",
                    "passed",
                ),
                "forcing_compatible": manifest.get("forcing_compatible"),
                "failure_taxonomy": list(
                    quality.get("failure_taxonomy", [])
                ),
            }
        )
    payload = {
        "schema_version": "fvcom_raw_mesher_metric_comparison_v2",
        "input_bundle_sha256": str(input_bundle_sha256),
        "raw_stage": True,
        "common_conditioning_applied": False,
        "comparison_policy": "metric_by_metric_hard_gates_only",
        "boundary_policy_stratification_required": True,
        "comparable_boundary_policy_stratum_required": True,
        "composite_score_computed": False,
        "winner_selected": False,
        "ranking_performed": False,
        "candidate_order_semantics": "requested_execution_order_only",
        "supported_candidate_metrics": rows,
        "unsupported_candidates": unsupported,
    }
    json_path = _write_json(output / "raw_metric_comparison.json", payload)
    csv_path = output / "raw_metric_comparison.csv"
    fieldnames = [
        key
        for key in (rows[0].keys() if rows else ())
        if key != "failure_taxonomy"
    ] + (["failure_taxonomy"] if rows else [])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            values = dict(row)
            values["failure_taxonomy"] = "|".join(
                str(value) for value in row["failure_taxonomy"]
            )
            writer.writerow(values)
    return _hash_artifacts(
        {
            "raw_metric_comparison_json": json_path,
            "raw_metric_comparison_csv": csv_path,
        }
    )


def run_portfolio_case(
    case_manifest_path: str | Path,
    workspace_root: str | Path,
    output_dir: str | Path,
    *,
    candidate_ids: Iterable[str] | None = None,
    config: PortfolioCaseConfig | None = None,
) -> dict[str, Any]:
    """Build one canonical bundle and execute selected raw mesher candidates."""

    from .gmsh_backend import GmshConfig, measure_boundary_mesh

    config = config or PortfolioCaseConfig()
    candidates = normalize_candidate_ids(candidate_ids)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"portfolio output must not already exist: {output}"
        )
    _validate_output_path_budget(output)
    output.mkdir(parents=True, exist_ok=False)
    readiness = check_case_readiness(case_manifest_path, workspace_root)
    if readiness["status"] != "ready":
        raise ValueError(
            "case readiness failed: " + ", ".join(readiness["blockers"])
        )
    prepared = prepare_case(case_manifest_path, workspace_root)
    assert_readiness_manifest_binding(
        readiness,
        prepared.manifest_sha256,
    )
    config, case_spacing_policy = _case_budget_spacing_policy(
        prepared,
        readiness,
        config,
    )
    coverage = bathymetry_coverage_report(
        prepared.bathymetry,
        prepared.source_domain_lonlat,
        source_open_boundary_lonlat(
            prepared.exterior_xy,
            prepared.open_boundaries,
            prepared.projection,
        ),
    )
    if not coverage["passed"]:
        raise ValueError(f"bathymetry coverage contract failed: {coverage}")
    source_boundary, source_boundary_report = _load_canonical_boundary(
        prepared,
        config,
    )
    source_boundary, target_assignment = _apply_case_budget_targets(
        source_boundary,
        case_spacing_policy,
        geometry_continuity=bool(
            config.boundary_geometry_continuity
        ),
        geometry_metric_ratio=float(
            config.boundary_geometry_metric_ratio
        ),
    )
    case_spacing_policy["boundary_target_assignment"] = target_assignment
    size_bathy = coarsen_for_size_field(
        prepared.bathymetry,
        max_cells=int(config.size_field_max_cells),
    )
    field_config = _size_field_config(config)
    (
        boundary,
        size_field,
        sampler,
        reconciliation,
    ) = _reconcile_boundary_and_size_field(
        size_bathy,
        source_boundary,
        field_config,
        config,
    )
    if size_field.report.get("schema_version") != "fvcom_size_field_v4":
        raise RuntimeError("canonical builder did not emit fvcom_size_field_v4")
    boundary_report = {
        **source_boundary_report,
        "schema_version": "fvcom_portfolio_canonical_boundary_v2",
        "node_count": int(len(boundary.xy)),
        "upstream_source_node_count": int(len(source_boundary.xy)),
        "reconciled_node_count": int(len(boundary.xy)),
        "target_spacing_minimum_m": float(
            np.min(boundary.target_spacing_m)
        ),
        "target_spacing_maximum_m": float(
            np.max(boundary.target_spacing_m)
        ),
        "boundary_size_reconciliation": reconciliation,
        "case_budget_spacing_policy": case_spacing_policy,
    }
    routing = capability_routing(prepared)
    any_gmsh = any(value in GMSH_CANDIDATE_ALGORITHMS for value in candidates)
    gmsh_boundary_count = None
    if any_gmsh:
        mesher_prepared = _prepared_case_on_boundary(prepared, boundary)
        preflight_result = measure_boundary_mesh(
            _backend_geometry(mesher_prepared),
            GmshConfig(
                h_uniform_m=1_000.0,
                algorithm=6,
                canonical_size_callback=sampler,
                constant_field=not bool(prepared.open_boundaries),
                preserve_source_boundary_discretization=True,
                model_name=f"{prepared.manifest['case_id']}_portfolio_preflight",
            ),
        )
        gmsh_boundary_count = int(preflight_result.boundary_node_count)
    preflight = _preflight_report(
        size_field,
        boundary,
        config,
        sampler=sampler,
        upstream_source_boundary_node_count=len(source_boundary.xy),
        gmsh_boundary_node_count=gmsh_boundary_count,
    )
    bundle_path, bundle = _build_input_bundle(
        output,
        prepared,
        boundary,
        boundary_report,
        size_field,
        field_config,
        config,
        preflight,
        readiness,
        source_boundary=source_boundary,
        boundary_reconciliation=reconciliation,
    )
    input_bundle_sha256 = str(bundle["input_bundle_sha256"])
    routing_path = _write_json(output / "capability_routing.json", routing)
    if not preflight["passed"] or not reconciliation["passed"]:
        rejection_failures: list[str] = []
        if not reconciliation["passed"]:
            rejection_failures.extend(
                reconciliation.get(
                    "failure_taxonomy",
                    ["boundary_size_fixed_point_not_converged"],
                )
            )
        if not preflight["passed"]:
            if (
                int(preflight["estimated_total_node_count"])
                > int(preflight["preflight_node_limit"])
            ):
                rejection_failures.append(
                    "common_node_budget_preflight_exceeded"
                )
            if not bool(
                preflight.get("gmsh_common_boundary_lock_passed", True)
            ):
                rejection_failures.append(
                    "gmsh_common_boundary_lock_preflight_failed"
                )
        failure_manifest = {
            "schema_version": SCHEMA_VERSION,
            "case_id": prepared.manifest["case_id"],
            "input_bundle_sha256": input_bundle_sha256,
            "status": "preflight_rejected",
            "raw_stage": True,
            "common_conditioning_applied": False,
            "failure_taxonomy": sorted(set(rejection_failures)),
            "input_bundle": {
                "path": str(bundle_path),
                "sha256": file_sha256(bundle_path),
            },
            "capability_routing": {
                "path": str(routing_path),
                "sha256": file_sha256(routing_path),
            },
            "candidates": [],
            "comparison_policy": "metric_by_metric_only_no_composite_winner",
        }
        _write_json(output / "portfolio_case_manifest.json", failure_manifest)
        if not reconciliation["passed"]:
            raise ValueError(
                "boundary/size-field fixed-point preflight failed: "
                + ", ".join(rejection_failures)
            )
        raise ValueError(
            "common portfolio preflight failed: "
            + ", ".join(sorted(set(rejection_failures)))
        )

    candidate_manifests: list[dict[str, Any]] = []
    candidates_root = output / "candidates"
    candidates_root.mkdir()
    for candidate_id in candidates:
        candidate_dir = candidates_root / candidate_id
        candidate_dir.mkdir()
        capability = routing["candidates"][candidate_id]
        if not capability["supported"]:
            candidate_manifests.append(
                _failed_candidate_artifacts(
                    candidate_dir,
                    candidate_id,
                    status="unsupported",
                    failure="capability_route_unsupported_for_case_topology",
                    input_bundle_sha256=input_bundle_sha256,
                )
            )
            continue
        if isinstance(sampler, BoundaryTraceSizeSampler):
            # Preflight and earlier candidates must not leak operational
            # sampler counters into this candidate's immutable quality report.
            sampler.reset_operational_counters()
        try:
            manifest = _execute_candidate(
                candidate_id,
                candidate_dir,
                prepared,
                boundary,
                size_field,
                sampler,
                config,
                input_bundle_sha256,
            )
        except Exception as exc:
            manifest = _failed_candidate_artifacts(
                candidate_dir,
                candidate_id,
                status="failed",
                failure="raw_candidate_execution_failed",
                input_bundle_sha256=input_bundle_sha256,
                exception=exc,
            )
        candidate_manifests.append(manifest)

    comparison_artifacts = _write_raw_metric_comparison(
        output,
        candidate_manifests,
        routing,
        input_bundle_sha256,
    )
    if file_sha256(prepared.manifest_path) != prepared.manifest_sha256:
        raise RuntimeError("case manifest changed during portfolio execution")
    failed = [
        value
        for value in candidate_manifests
        if value["status"] == "failed"
    ]
    unsupported = [
        value
        for value in candidate_manifests
        if value["status"] == "unsupported"
    ]
    needs_review = [
        value
        for value in candidate_manifests
        if value["status"] == "needs_review"
    ]
    status = (
        "partial_failure"
        if failed
        else (
            "needs_review"
            if needs_review
            else ("capability_limited" if unsupported else "pass")
        )
    )
    case_manifest = {
        "schema_version": SCHEMA_VERSION,
        "case_id": prepared.manifest["case_id"],
        "display_name": prepared.manifest.get("display_name"),
        "completed_at_utc": _utc_now(),
        "status": status,
        "research_only": True,
        "raw_stage": True,
        "common_conditioning_applied": False,
        "conditioning_owner": "external_generic_bakeoff_stage",
        "canonical_size_field_schema": "fvcom_size_field_v4",
        "boundary_size_reconciliation": {
            "status": reconciliation["status"],
            "passed": bool(reconciliation["passed"]),
            "source_boundary_node_count": int(
                reconciliation["source_boundary_node_count"]
            ),
            "reconciled_boundary_node_count": int(
                reconciliation["reconciled_boundary_node_count"]
            ),
            "converged_iteration": reconciliation[
                "converged_iteration"
            ],
            "policy": reconciliation["policy"],
            "method_scope": reconciliation["method_scope"],
            "final_boundary_field_audit": reconciliation[
                "final_boundary_field_audit"
            ],
        },
        "case_budget_spacing_policy": case_spacing_policy,
        "input_bundle_sha256": input_bundle_sha256,
        "preflight_node_limit": int(config.preflight_node_limit),
        "hard_node_limit": int(config.hard_node_limit),
        "bathymetry_coverage": coverage,
        "input_bundle": {
            "path": str(bundle_path),
            "sha256": file_sha256(bundle_path),
        },
        "input_bundle_manifest": bundle,
        "capability_routing": {
            "path": str(routing_path),
            "sha256": file_sha256(routing_path),
        },
        "requested_candidates": list(candidates),
        "comparison_artifacts": comparison_artifacts,
        "candidates": [
            {
                "candidate_id": value["candidate_id"],
                "status": value["status"],
                "quality_accepted": bool(value.get("quality_accepted", False)),
                "failure_taxonomy": list(value.get("failure_taxonomy", [])),
                "manifest_path": value.get("manifest_path"),
                "manifest_sha256": value.get("manifest_sha256"),
            }
            for value in candidate_manifests
        ],
        "comparison_policy": "metric_by_metric_only_no_composite_winner",
        "boundary_policy_stratification_required": True,
        "composite_winner": None,
    }
    _write_json(output / "portfolio_case_manifest.json", case_manifest)
    return case_manifest


__all__ = [
    "BoundaryTraceSizeSampler",
    "CANDIDATE_ALIASES",
    "DEFAULT_CANDIDATES",
    "DEFAULT_HARD_NODE_LIMIT",
    "DEFAULT_FALLBACK_CANDIDATES",
    "DEFAULT_PRIMARY_CANDIDATE",
    "DEFAULT_PREFLIGHT_NODE_LIMIT",
    "GMSH_CANDIDATE_ALGORITHMS",
    "PortfolioCaseConfig",
    "ProjectedSizeSampler",
    "SCHEMA_VERSION",
    "capability_routing",
    "normalize_candidate_ids",
    "run_portfolio_case",
]
