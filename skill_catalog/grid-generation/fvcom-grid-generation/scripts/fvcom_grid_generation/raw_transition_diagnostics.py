"""Read-only visual diagnostics for raw mesher-portfolio candidates.

The portfolio quality JSON contains the authoritative aggregate gates.  This
module adds spatial evidence for two failure modes that are difficult to
understand from aggregates alone:

* target-size mismatch across the one-dimensional boundary/two-dimensional
  field interface; and
* excessive area change across adjacent triangles.

Inputs are never modified.  Triangle ``L/h`` follows :mod:`edge_size_audit`:
each edge uses the minimum target sampled at both endpoints and its midpoint,
and each triangle receives the maximum ratio of its three edges.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.tri as mtri
import numpy as np
from scipy.spatial import cKDTree
import xarray as xr

from .edge_size_audit import audit_edge_target_sizes
from .metrics import build_edge_topology, triangle_geometry
from .projection import local_utm_projection, project_points
from .size_field import (
    linear_target_metric_edge_fractions,
    recorded_size_interpolator,
)
from .sms_2dm import read_2dm


SCHEMA_VERSION = "fvcom_raw_transition_diagnostics_v1"
DEFAULT_MAX_PLOT_TRIANGLES = 100_000
DEFAULT_INTERFACE_RATIO_LIMIT = 2.0
DEFAULT_AREA_CHANGE_LIMIT = 0.5
DEFAULT_BOUNDARY_MATCH_TOLERANCE_M = 0.05
BOUNDARY_TRACE_SAMPLER_V1 = "fvcom_boundary_trace_sampler_v1"
BOUNDARY_TRACE_SAMPLER_V2 = "fvcom_boundary_trace_sampler_v2"
BOUNDARY_TRACE_SAMPLER_SCHEMAS = frozenset(
    {
        BOUNDARY_TRACE_SAMPLER_V1,
        BOUNDARY_TRACE_SAMPLER_V2,
    }
)


class _CanonicalSizeSampler:
    """Strict-coverage regular-grid sampler for ``fvcom_size_field_v4``."""

    def __init__(self, path: Path, projection: Any) -> None:
        with xr.open_dataset(path) as dataset:
            schema = str(dataset.attrs.get("schema_version", "")).strip()
            if schema != "fvcom_size_field_v4":
                raise ValueError(
                    "canonical size field requires "
                    f"schema_version=fvcom_size_field_v4, received {schema!r}"
                )
            for name in ("lon", "lat", "mesh_size_m"):
                if name not in dataset:
                    raise ValueError(
                        f"canonical size field is missing {name!r}"
                    )
            lon = np.asarray(dataset["lon"].values, dtype=float)
            lat = np.asarray(dataset["lat"].values, dtype=float)
            values = np.asarray(
                dataset["mesh_size_m"].transpose("lat", "lon").values,
                dtype=float,
            )
            if "size_field_coverage_mask" in dataset:
                coverage = np.asarray(
                    dataset["size_field_coverage_mask"]
                    .transpose("lat", "lon")
                    .values,
                    dtype=bool,
                )
            else:
                coverage = np.isfinite(values) & (values > 0.0)
            if "model_domain_mask" in dataset:
                domain = np.asarray(
                    dataset["model_domain_mask"]
                    .transpose("lat", "lon")
                    .values,
                    dtype=bool,
                )
            else:
                domain = coverage.copy()
            sampling_interface_schema = str(
                dataset.attrs.get(
                    "sampling_interface_schema_version",
                    "legacy_unspecified",
                )
            ).strip()
        if lon.ndim != 1 or lat.ndim != 1:
            raise ValueError("size-field lon and lat must be one dimensional")
        if len(lon) < 2 or len(lat) < 2:
            raise ValueError("size-field grid must be at least 2 by 2")
        if values.shape != (len(lat), len(lon)):
            raise ValueError("mesh_size_m dimensions do not match lat/lon")
        if coverage.shape != values.shape:
            raise ValueError(
                "size_field_coverage_mask dimensions do not match mesh_size_m"
            )
        if domain.shape != values.shape:
            raise ValueError(
                "model_domain_mask dimensions do not match mesh_size_m"
            )
        if np.all(np.diff(lon) < 0.0):
            lon = lon[::-1]
            values = values[:, ::-1]
            coverage = coverage[:, ::-1]
            domain = domain[:, ::-1]
        if np.all(np.diff(lat) < 0.0):
            lat = lat[::-1]
            values = values[::-1, :]
            coverage = coverage[::-1, :]
            domain = domain[::-1, :]
        if (
            np.any(~np.isfinite(lon))
            or np.any(~np.isfinite(lat))
            or np.any(np.diff(lon) <= 0.0)
            or np.any(np.diff(lat) <= 0.0)
        ):
            raise ValueError("size-field axes must be finite and increasing")
        if np.any(coverage & (~np.isfinite(values) | (values <= 0.0))):
            raise ValueError(
                "canonical size field has invalid values inside coverage"
            )
        self.schema_version = schema
        self.path = path
        self.projection = projection
        self.lon = lon
        self.lat = lat
        self.values = values
        self.coverage = coverage
        self.domain = domain
        self.sampling_interface_schema_version = sampling_interface_schema
        self._raster = recorded_size_interpolator(
            lat,
            lon,
            values,
            coverage,
            domain,
            sampling_interface_schema,
        )
        self._trace_tree: cKDTree | None = None
        self._trace_targets: np.ndarray | None = None
        self._trace_gradation: float | None = None
        self._trace_nearest_count = 0
        self._trace_minimum_target: float | None = None
        self._trace_adaptive_neighbor_expansion = False
        self._trace_no_active_support_policy = "raster_min"
        self._trace_query_chunk_size = 4_096
        self.trace_report: dict[str, Any] = {
            "enabled": False,
            "method": "raster_only",
            "base_raster_sampling_interface_schema_version": (
                self.sampling_interface_schema_version
            ),
        }

    def enable_boundary_trace(
        self,
        nodes_xy: np.ndarray,
        boundary_edges: Sequence[tuple[int, int]],
        target_by_node: np.ndarray,
        *,
        gradation: float,
        samples_per_target: float = 4.0,
        nearest_sample_count: int = 16,
        maximum_total_sample_count: int = 5_000_000,
        schema_version: str = BOUNDARY_TRACE_SAMPLER_V1,
        no_active_support_policy: str | None = None,
        query_chunk_size: int = 4_096,
    ) -> None:
        """Reconstruct the portfolio's sampled boundary-trace extension."""

        nodes = np.asarray(nodes_xy, dtype=float)
        targets_by_node = np.asarray(target_by_node, dtype=float)
        schema = str(schema_version).strip()
        if schema not in BOUNDARY_TRACE_SAMPLER_SCHEMAS:
            raise ValueError(
                "unsupported boundary trace sampler schema "
                f"{schema_version!r}"
            )
        if not np.isfinite(gradation) or gradation <= 0.0:
            raise ValueError("boundary trace gradation must be positive")
        if int(nearest_sample_count) < 1:
            raise ValueError("nearest_sample_count must be positive")
        if int(maximum_total_sample_count) < 1:
            raise ValueError("maximum_total_sample_count must be positive")
        if int(query_chunk_size) < 1:
            raise ValueError("query_chunk_size must be positive")
        support_policy = str(
            no_active_support_policy or "raster_min"
        ).strip()
        if support_policy not in {
            "raster_min",
            "boundary_trace_authoritative",
        }:
            raise ValueError(
                "unsupported boundary trace no-active-support policy "
                f"{support_policy!r}"
            )
        points: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        sample_spacings: list[float] = []
        total_sample_count = 0
        for raw_start, raw_end in sorted(boundary_edges):
            start = int(raw_start)
            end = int(raw_end)
            ha = float(targets_by_node[start])
            hb = float(targets_by_node[end])
            if not np.isfinite(ha) or ha <= 0.0 or not np.isfinite(hb) or hb <= 0.0:
                continue
            a = nodes[start]
            b = nodes[end]
            length = float(np.linalg.norm(b - a))
            remaining_sample_count = (
                int(maximum_total_sample_count) - total_sample_count
            )
            if schema == BOUNDARY_TRACE_SAMPLER_V1:
                intervals = max(
                    2,
                    int(
                        np.ceil(
                            length
                            * float(samples_per_target)
                            / min(ha, hb)
                        )
                    ),
                )
                if intervals % 2:
                    intervals += 1
                required = intervals + 1
                if required > remaining_sample_count:
                    raise ValueError(
                        "boundary trace sampling exceeds the safety limit "
                        "before allocation"
                    )
                fraction = np.linspace(
                    0.0,
                    1.0,
                    required,
                    endpoint=True,
                    dtype=float,
                )
            else:
                fraction = linear_target_metric_edge_fractions(
                    length,
                    ha,
                    hb,
                    samples_per_target=float(samples_per_target),
                    include_end=True,
                    maximum_sample_count=remaining_sample_count,
                )
            total_sample_count += len(fraction)
            if total_sample_count > int(maximum_total_sample_count):
                raise ValueError(
                    "boundary trace sampling exceeds the safety limit "
                    f"of {int(maximum_total_sample_count)} points"
                )
            sample_spacings.append(
                length * float(np.max(np.diff(fraction), initial=0.0))
            )
            points.append(
                a[None, :] + fraction[:, None] * (b - a)[None, :]
            )
            targets.append((1.0 - fraction) * ha + fraction * hb)
        if not points:
            raise ValueError("no finite boundary edges are available for trace sampling")
        trace_points = np.vstack(points)
        trace_targets = np.concatenate(targets)
        if schema == BOUNDARY_TRACE_SAMPLER_V1:
            _unique, first = np.unique(
                np.column_stack([trace_points, trace_targets]),
                axis=0,
                return_index=True,
            )
            keep = np.sort(first)
            trace_points = trace_points[keep]
            trace_targets = trace_targets[keep]
        self._trace_tree = cKDTree(trace_points)
        self._trace_targets = trace_targets
        self._trace_gradation = float(gradation)
        self._trace_nearest_count = min(
            int(nearest_sample_count),
            len(trace_points),
        )
        self._trace_minimum_target = float(np.min(self._trace_targets))
        self._trace_adaptive_neighbor_expansion = (
            schema == BOUNDARY_TRACE_SAMPLER_V2
        )
        self._trace_no_active_support_policy = support_policy
        self._trace_query_chunk_size = int(query_chunk_size)
        report = {
            "enabled": True,
            "schema_version": schema,
            "boundary_sample_count": int(len(trace_points)),
            "samples_per_target": float(samples_per_target),
            "sample_distribution": (
                "linear_endpoint_target_metric_equidistribution"
                if schema == BOUNDARY_TRACE_SAMPLER_V2
                else "uniform_physical_arclength"
            ),
            "maximum_total_sample_count": int(
                maximum_total_sample_count
            ),
            "query_chunk_size": int(self._trace_query_chunk_size),
            "memory_bounded_query_chunks": True,
            "nearest_sample_count": int(self._trace_nearest_count),
            "gradation": float(gradation),
            "endpoint_midpoint_exact_by_construction": True,
            "distance_metric": "straight_euclidean",
            "barrier_aware": False,
            "base_raster_sampling_interface_schema_version": (
                self.sampling_interface_schema_version
            ),
            "base_query_without_positive_active_support_count": 0,
            "no_active_support_policy": support_policy,
        }
        if schema == BOUNDARY_TRACE_SAMPLER_V2:
            maximum_sample_spacing = max(sample_spacings, default=0.0)
            report.update(
                {
                    "method": (
                        "raster_min_deterministic_boundary_point_"
                        "euclidean_gradation_extension_"
                        "adaptive_exact_sample_minimum"
                    ),
                    "initial_nearest_sample_count": int(
                        self._trace_nearest_count
                    ),
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
                        float(gradation) * maximum_sample_spacing
                    ),
                    "sample_query_count": 0,
                    "expanded_sample_query_count": 0,
                    "maximum_neighbors_examined": 0,
                }
            )
        else:
            report["method"] = (
                "raster_min_deterministic_boundary_point_"
                "euclidean_gradation_extension"
            )
        self.trace_report = report

    def sample_lonlat(self, lonlat: np.ndarray) -> np.ndarray:
        sampled, _active_support = (
            self.sample_lonlat_with_active_support(lonlat)
        )
        return sampled

    def sample_lonlat_with_active_support(
        self,
        lonlat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(lonlat, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("size queries must have shape (n, 2)")
        query = np.column_stack((points[:, 1], points[:, 0]))
        sampled, support = self._raster.sample_with_active_support(query)
        return (
            np.asarray(sampled, dtype=float),
            np.asarray(support, dtype=bool),
        )

    def sample_xy(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("projected size queries must have shape (n, 2)")
        if len(points) > self._trace_query_chunk_size:
            return np.concatenate(
                [
                    self.sample_xy(
                        points[
                            begin : min(
                                len(points),
                                begin + self._trace_query_chunk_size,
                            )
                        ]
                    )
                    for begin in range(
                        0,
                        len(points),
                        self._trace_query_chunk_size,
                    )
                ]
            )
        lon, lat = self.projection.to_lonlat.transform(
            points[:, 0],
            points[:, 1],
        )
        sampled, active_support = self.sample_lonlat_with_active_support(
            np.column_stack((lon, lat))
        )
        invalid = ~np.isfinite(sampled) | (sampled <= 0.0)
        if np.any(invalid):
            first = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                "canonical size query is outside strict coverage or invalid: "
                f"{int(np.count_nonzero(invalid))} point(s); first projected "
                f"coordinate=({points[first, 0]:.6f}, "
                f"{points[first, 1]:.6f})"
            )
        if (
            self._trace_tree is None
            or self._trace_targets is None
            or self._trace_gradation is None
        ):
            return sampled
        effective_sampled = (
            np.where(active_support, sampled, np.inf)
            if self._trace_no_active_support_policy
            == "boundary_trace_authoritative"
            else np.asarray(sampled, dtype=float)
        )
        self.trace_report[
            "base_query_without_positive_active_support_count"
        ] = int(
            self.trace_report[
                "base_query_without_positive_active_support_count"
            ]
        ) + int(np.count_nonzero(~active_support))
        if self._trace_adaptive_neighbor_expansion:
            if self._trace_minimum_target is None:
                raise RuntimeError(
                    "adaptive boundary trace is missing its minimum target"
                )
            result = np.asarray(effective_sampled, dtype=float).copy()
            best_extension = np.full(len(points), np.inf, dtype=float)
            unresolved = np.arange(len(points), dtype=int)
            neighbor_count = int(self._trace_nearest_count)
            expanded_queries = 0
            maximum_examined = 0
            tolerance = 32.0 * np.finfo(float).eps

            while len(unresolved):
                distance, indices = self._trace_tree.query(
                    points[unresolved],
                    k=neighbor_count,
                )
                distance_array = np.asarray(distance, dtype=float)
                index_array = np.asarray(indices, dtype=int)
                if distance_array.ndim == 1:
                    distance_array = distance_array[:, None]
                    index_array = index_array[:, None]
                local_extension = np.min(
                    self._trace_targets[index_array]
                    + self._trace_gradation * distance_array,
                    axis=1,
                )
                best_extension[unresolved] = np.minimum(
                    best_extension[unresolved],
                    local_extension,
                )
                result[unresolved] = np.minimum(
                    effective_sampled[unresolved],
                    best_extension[unresolved],
                )
                maximum_examined = max(maximum_examined, neighbor_count)
                if neighbor_count >= len(self._trace_targets):
                    break

                unseen_lower_bound = (
                    self._trace_minimum_target
                    + self._trace_gradation * distance_array[:, -1]
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
                    len(self._trace_targets),
                    max(neighbor_count + 1, 2 * neighbor_count),
                )

            self.trace_report["sample_query_count"] = int(
                self.trace_report["sample_query_count"]
            ) + int(len(points))
            self.trace_report["expanded_sample_query_count"] = int(
                self.trace_report["expanded_sample_query_count"]
            ) + int(expanded_queries)
            self.trace_report["maximum_neighbors_examined"] = max(
                int(self.trace_report["maximum_neighbors_examined"]),
                int(maximum_examined),
            )
            return result
        distance, indices = self._trace_tree.query(
            points,
            k=self._trace_nearest_count,
        )
        distance_array = np.asarray(distance, dtype=float)
        index_array = np.asarray(indices, dtype=int)
        if distance_array.ndim == 1:
            distance_array = distance_array[:, None]
            index_array = index_array[:, None]
        extension = np.min(
            self._trace_targets[index_array]
            + self._trace_gradation * distance_array,
            axis=1,
        )
        return np.minimum(effective_sampled, extension)


def write_raw_transition_diagnostics(
    mesh_2dm: str | Path,
    canonical_boundary_geojson: str | Path,
    canonical_size_field_nc: str | Path,
    quality_json: str | Path,
    output_dir: str | Path,
    *,
    title: str | None = None,
    max_plot_triangles: int = DEFAULT_MAX_PLOT_TRIANGLES,
    transition_graph_rings: int = 2,
    boundary_match_tolerance_m: float = DEFAULT_BOUNDARY_MATCH_TOLERANCE_M,
) -> dict[str, Any]:
    """Create whole-mesh and boundary/first-ring raw-quality maps.

    The returned document is also serialized as
    ``raw_transition_diagnostics.json`` in ``output_dir``.  Existing outputs
    are never overwritten.
    """

    paths = {
        "mesh_2dm": Path(mesh_2dm).expanduser().resolve(),
        "canonical_boundary_geojson": Path(canonical_boundary_geojson)
        .expanduser()
        .resolve(),
        "canonical_size_field_nc": Path(canonical_size_field_nc)
        .expanduser()
        .resolve(),
        "quality_json": Path(quality_json).expanduser().resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if int(max_plot_triangles) != max_plot_triangles or max_plot_triangles < 1:
        raise ValueError("max_plot_triangles must be a positive integer")
    if (
        int(transition_graph_rings) != transition_graph_rings
        or transition_graph_rings < 0
    ):
        raise ValueError(
            "transition_graph_rings must be a non-negative integer"
        )
    if (
        not np.isfinite(boundary_match_tolerance_m)
        or boundary_match_tolerance_m <= 0.0
    ):
        raise ValueError(
            "boundary_match_tolerance_m must be finite and positive"
        )

    output = Path(output_dir).expanduser().resolve()
    os.makedirs(_extended_path(output), exist_ok=True)
    whole_path = output / "whole_mesh_l_over_h.png"
    transition_path = output / "boundary_first_ring_transition.png"
    report_path = output / "raw_transition_diagnostics.json"
    for path in (whole_path, transition_path, report_path):
        if _path_exists(path):
            raise FileExistsError(f"refusing to overwrite {path}")

    mesh = read_2dm(paths["mesh_2dm"])
    nodes_lonlat = np.asarray(mesh.nodes_lonlat, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=np.int64) - 1
    _validate_mesh(nodes_lonlat, triangles)
    west = float(np.min(nodes_lonlat[:, 0]))
    east = float(np.max(nodes_lonlat[:, 0]))
    south = float(np.min(nodes_lonlat[:, 1]))
    north = float(np.max(nodes_lonlat[:, 1]))
    projection = local_utm_projection((west, south, east, north))
    nodes_xy = project_points(nodes_lonlat, projection)

    quality = _read_json_object(paths["quality_json"], "quality JSON")
    thresholds = _quality_thresholds(quality)
    sampler = _CanonicalSizeSampler(
        paths["canonical_size_field_nc"],
        projection,
    )
    topology = build_edge_topology(len(nodes_xy), triangles)
    constraint_chains, boundary_trace = _trace_boundary_chains(
        topology.boundary_edges
    )
    boundary_targets, target_mapping = _map_boundary_targets(
        paths["canonical_boundary_geojson"],
        nodes_xy,
        projection,
        boundary_match_tolerance_m,
        set(topology.boundary_edges),
    )
    sampling_contract = quality.get("canonical_size_sampling", {})
    if not (
        isinstance(sampling_contract, Mapping)
        and sampling_contract.get("schema_version")
        in BOUNDARY_TRACE_SAMPLER_SCHEMAS
    ):
        reconciliation_path = (
            paths["canonical_boundary_geojson"].parent
            / "boundary_size_reconciliation.json"
        )
        if reconciliation_path.is_file():
            reconciliation = _read_json_object(
                reconciliation_path,
                "boundary reconciliation JSON",
            )
            fallback_contract = reconciliation.get(
                "boundary_trace_sampler",
                {},
            )
            if (
                isinstance(fallback_contract, Mapping)
                and fallback_contract.get("schema_version")
                in BOUNDARY_TRACE_SAMPLER_SCHEMAS
            ):
                sampling_contract = fallback_contract
                paths["boundary_size_reconciliation"] = reconciliation_path
    if (
        isinstance(sampling_contract, Mapping)
        and sampling_contract.get("schema_version")
        in BOUNDARY_TRACE_SAMPLER_SCHEMAS
    ):
        sampling_schema = str(sampling_contract["schema_version"])
        sampler.enable_boundary_trace(
            nodes_xy,
            topology.boundary_edges,
            boundary_targets,
            gradation=float(
                sampling_contract.get(
                    "gradation",
                    thresholds["gradation_limit"],
                )
            ),
            samples_per_target=float(
                sampling_contract.get("samples_per_target", 4.0)
            ),
            nearest_sample_count=int(
                sampling_contract.get(
                    "initial_nearest_sample_count",
                    sampling_contract.get("nearest_sample_count", 16),
                )
            ),
            schema_version=sampling_schema,
            no_active_support_policy=str(
                sampling_contract.get(
                    "no_active_support_policy",
                    "raster_min",
                )
            ),
            query_chunk_size=int(
                sampling_contract.get("query_chunk_size", 4_096)
            ),
        )

    arrays = _edge_triangle_arrays(
        nodes_lonlat,
        nodes_xy,
        triangles,
        topology.edge_to_triangles,
        set(topology.boundary_edges),
        boundary_targets,
        sampler.sample_xy,
        thresholds["interface_ratio_limit"],
        thresholds["area_change_limit"],
    )
    triangle_distance = _triangle_graph_distance(
        len(triangles),
        topology.edge_to_triangles,
        set(topology.boundary_edges),
    )
    strata = _strata(triangle_distance, int(transition_graph_rings))

    edge_audit = audit_edge_target_sizes(
        nodes_xy,
        triangles,
        [
            {
                "chain_id": f"mesh_boundary_{index:03d}",
                "nodes": chain["nodes"],
                "cyclic": bool(chain["cyclic"]),
            }
            for index, chain in enumerate(constraint_chains, start=1)
        ],
        sampler.sample_xy,
        boundary_target_by_node=boundary_targets,
        transition_graph_rings=int(transition_graph_rings),
        thresholds=(
            thresholds["l_over_h_p95_limit"],
            thresholds["l_over_h_maximum_limit"],
        ),
        boundary_gradation_limit=thresholds["gradation_limit"],
        boundary_field_ratio_limit=thresholds["interface_ratio_limit"],
    )

    whole_indices = _deterministic_sample(
        np.arange(len(triangles), dtype=np.int64),
        int(max_plot_triangles),
    )
    boundary_first_indices = _deterministic_sample(
        np.flatnonzero(triangle_distance <= 1),
        int(max_plot_triangles),
    )
    map_title = title or mesh.mesh_name or paths["mesh_2dm"].stem
    _write_whole_map(
        whole_path,
        nodes_lonlat,
        triangles,
        arrays,
        whole_indices,
        map_title,
        quality,
        thresholds,
    )
    _write_transition_map(
        transition_path,
        nodes_lonlat,
        triangles,
        arrays,
        boundary_first_indices,
        triangle_distance,
        map_title,
        thresholds,
    )

    interface_hotspots = arrays["interface_hotspot_edge_indices"]
    area_hotspots = arrays["area_hotspot_edge_indices"]
    diagnostic_failures = _diagnostic_failure_taxonomy(
        edge_audit=edge_audit,
        interface_hotspot_count=int(len(interface_hotspots)),
        area_hotspot_count=int(len(area_hotspots)),
        unmatched_boundary_node_count=int(
            target_mapping["unmatched_mesh_boundary_node_count"]
        ),
        invalid_triangle_l_over_h_count=int(
            arrays["invalid_triangle_l_over_h_count"]
        ),
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_status": (
            "pass" if not diagnostic_failures else "needs_review"
        ),
        "failure_taxonomy": diagnostic_failures,
        "read_only_inputs": True,
        "method": {
            "length_coordinates": (
                f"local UTM EPSG:{projection.epsg} selected from mesh bbox"
            ),
            "edge_l_over_h": (
                "edge_length / minimum_target_at_endpoints_and_midpoint"
            ),
            "triangle_l_over_h": "maximum_of_three_incident_edge_ratios",
            "boundary_target_precedence": (
                "canonical_1d_boundary_target_then_2d_field_fallback"
            ),
            "size_field_sampling": sampler.trace_report,
            "boundary_field_interface_ratio": (
                "maximum pointwise symmetric h_gamma/H ratio at both "
                "endpoints and the midpoint"
            ),
            "adjacent_area_change": "abs(A1-A2)/max(A1,A2)",
            "plot_triangle_sampling": (
                "deterministic_equal-index_stride_without_replacement"
            ),
        },
        "inputs": {
            label: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for label, path in paths.items()
        },
        "mesh": {
            "mesh_name": mesh.mesh_name,
            "node_count": int(len(nodes_lonlat)),
            "triangle_count": int(len(triangles)),
            "unique_edge_count": int(len(arrays["edges"])),
            "topological_boundary_edge_count": int(
                len(topology.boundary_edges)
            ),
            "open_boundary_chain_count": int(
                len(mesh.open_boundary_chains)
            ),
        },
        "quality_source": _quality_summary(quality),
        "thresholds": thresholds,
        "boundary_trace": boundary_trace,
        "boundary_target_mapping": target_mapping,
        "triangle_strata": {
            name: int(np.count_nonzero(strata == name))
            for name in (
                "boundary",
                "first_ring",
                "transition",
                "true_interior",
            )
        },
        "triangle_l_over_h": _numeric_summary(
            arrays["triangle_l_over_h"]
        ),
        "boundary_field_interface_hotspots": {
            "definition": (
                "strict symmetric boundary/field target ratio above limit"
            ),
            "limit": float(thresholds["interface_ratio_limit"]),
            "count": int(len(interface_hotspots)),
            "symmetric_ratio": _numeric_summary(
                arrays["interface_ratio"][arrays["is_boundary"]]
            ),
            "incomplete_boundary_target_edge_count": int(
                arrays["incomplete_boundary_target_edge_count"]
            ),
            "incomplete_2d_field_edge_count": int(
                arrays["incomplete_field_target_edge_count"]
            ),
            "top_hotspots": _hotspot_records(
                arrays,
                interface_hotspots,
                "interface_ratio",
            ),
        },
        "adjacent_area_change_hotspots": {
            "definition": "strict abs(A1-A2)/max(A1,A2) above limit",
            "limit": float(thresholds["area_change_limit"]),
            "count": int(len(area_hotspots)),
            "maximum": _finite_maximum(arrays["area_change"]),
            "top_hotspots": _hotspot_records(
                arrays,
                area_hotspots,
                "area_change",
            ),
        },
        "invalid_triangle_l_over_h_count": int(
            arrays["invalid_triangle_l_over_h_count"]
        ),
        "authoritative_edge_size_audit": edge_audit,
        "plot_sampling": {
            "maximum_triangles_per_map": int(max_plot_triangles),
            "whole_mesh_eligible_count": int(len(triangles)),
            "whole_mesh_plotted_count": int(len(whole_indices)),
            "boundary_first_ring_eligible_count": int(
                np.count_nonzero(triangle_distance <= 1)
            ),
            "boundary_first_ring_plotted_count": int(
                len(boundary_first_indices)
            ),
        },
        "artifacts": {
            "whole_mesh_map": {
                "path": str(whole_path),
                "sha256": _sha256(whole_path),
            },
            "boundary_first_ring_map": {
                "path": str(transition_path),
                "sha256": _sha256(transition_path),
            },
        },
    }
    with open(_extended_path(report_path), "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
    report["report_path"] = str(report_path)
    report["report_sha256"] = _sha256(report_path)
    return report


def _validate_mesh(nodes_lonlat: np.ndarray, triangles: np.ndarray) -> None:
    if (
        nodes_lonlat.ndim != 2
        or nodes_lonlat.shape[1] != 2
        or len(nodes_lonlat) < 3
        or np.any(~np.isfinite(nodes_lonlat))
    ):
        raise ValueError("2DM must contain at least three finite lon/lat nodes")
    if (
        triangles.ndim != 2
        or triangles.shape[1] != 3
        or len(triangles) == 0
        or np.any(triangles < 0)
        or np.any(triangles >= len(nodes_lonlat))
    ):
        raise ValueError("2DM must contain valid first-order triangles")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return document


def _quality_thresholds(quality: Mapping[str, Any]) -> dict[str, float]:
    values = quality.get("thresholds")
    source = values if isinstance(values, Mapping) else {}
    edge = quality.get("edge_aware_size_error")
    edge_source = edge if isinstance(edge, Mapping) else {}
    method = edge_source.get("method")
    edge_method = method if isinstance(method, Mapping) else {}
    interface = edge_source.get("boundary_field_interface")
    interface_source = interface if isinstance(interface, Mapping) else {}
    result = {
        "l_over_h_p95_limit": float(
            source.get("max_size_error_p95", 1.55)
        ),
        "l_over_h_maximum_limit": float(
            source.get("max_size_error", 2.0)
        ),
        "interface_ratio_limit": float(
            interface_source.get(
                "ratio_limit",
                interface_source.get(
                    "factor_two_limit",
                    edge_method.get(
                        "boundary_field_ratio_limit",
                        DEFAULT_INTERFACE_RATIO_LIMIT,
                    ),
                ),
            )
        ),
        "area_change_limit": float(
            source.get("max_area_change", DEFAULT_AREA_CHANGE_LIMIT)
        ),
        "gradation_limit": float(
            edge_method.get("boundary_gradation_limit", 0.20)
        ),
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in result.values()):
        raise ValueError("quality JSON contains an invalid positive threshold")
    if result["interface_ratio_limit"] <= 1.0:
        raise ValueError("boundary-field interface ratio limit must exceed one")
    return result


def _trace_boundary_chains(
    boundary_edges: Sequence[tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edges = {
        _canonical_edge(int(a), int(b))
        for a, b in boundary_edges
        if int(a) != int(b)
    }
    if not edges:
        raise ValueError("mesh has no topological boundary edges")
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    abnormal = sorted(
        node for node, neighbors in adjacency.items() if len(neighbors) != 2
    )
    if abnormal:
        chains = [
            {"nodes": [a, b], "cyclic": False}
            for a, b in sorted(edges)
        ]
        return chains, {
            "mode": "individual_edges_due_to_noncycle_boundary",
            "chain_count": int(len(chains)),
            "degree_not_two_node_count": int(len(abnormal)),
            "degree_not_two_nodes_zero_based_preview": abnormal[:50],
            "all_boundary_edges_represented": True,
        }

    unvisited = set(edges)
    chains: list[dict[str, Any]] = []
    while unvisited:
        component_nodes = _edge_component_nodes(min(unvisited), unvisited)
        start = min(component_nodes)
        current = start
        previous: int | None = None
        nodes = [start]
        while True:
            candidates = sorted(
                neighbor
                for neighbor in adjacency[current]
                if neighbor != previous
            )
            if not candidates:
                raise ValueError("boundary-cycle tracing reached a dead end")
            if previous is None:
                next_node = candidates[0]
            else:
                unused = [
                    node
                    for node in candidates
                    if _canonical_edge(current, node) in unvisited
                ]
                next_node = unused[0] if unused else candidates[0]
            edge = _canonical_edge(current, next_node)
            unvisited.discard(edge)
            if next_node == start:
                break
            nodes.append(next_node)
            previous, current = current, next_node
            if len(nodes) > len(component_nodes):
                raise ValueError("boundary-cycle tracing did not close")
        chains.append({"nodes": nodes, "cyclic": True})
    represented = {
        _canonical_edge(a, b)
        for chain in chains
        for a, b in _chain_pairs(chain["nodes"], bool(chain["cyclic"]))
    }
    return chains, {
        "mode": "closed_topological_cycles",
        "chain_count": int(len(chains)),
        "degree_not_two_node_count": 0,
        "all_boundary_edges_represented": bool(represented == edges),
    }


def _edge_component_nodes(
    seed: tuple[int, int],
    edges: set[tuple[int, int]],
) -> set[int]:
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen = {int(seed[0])}
    queue = deque([int(seed[0])])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency[current]):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def _map_boundary_targets(
    path: Path,
    mesh_xy: np.ndarray,
    projection: Any,
    tolerance_m: float,
    boundary_edges: set[tuple[int, int]],
) -> tuple[np.ndarray, dict[str, Any]]:
    document = _read_json_object(path, "canonical boundary GeoJSON")
    features = document.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("canonical boundary GeoJSON has no features")
    mesh_boundary_nodes = {
        node for edge in boundary_edges for node in edge
    }
    tree = cKDTree(mesh_xy)
    targets = np.full(len(mesh_xy), np.nan, dtype=float)
    matched_distances: list[float] = []
    unmatched_features = 0
    nonpoint_features = 0
    duplicate_matches = 0
    off_topological_boundary = 0
    for feature in features:
        if not isinstance(feature, Mapping):
            nonpoint_features += 1
            continue
        geometry = feature.get("geometry")
        props = feature.get("properties")
        if (
            not isinstance(geometry, Mapping)
            or geometry.get("type") != "Point"
            or not isinstance(props, Mapping)
        ):
            nonpoint_features += 1
            continue
        coordinates = geometry.get("coordinates")
        raw_target = props.get("target_spacing_m")
        if (
            not isinstance(coordinates, Sequence)
            or len(coordinates) < 2
            or raw_target is None
        ):
            nonpoint_features += 1
            continue
        lonlat = np.asarray(
            [[float(coordinates[0]), float(coordinates[1])]],
            dtype=float,
        )
        target = float(raw_target)
        if np.any(~np.isfinite(lonlat)) or not np.isfinite(target) or target <= 0:
            nonpoint_features += 1
            continue
        feature_xy = project_points(lonlat, projection)[0]
        suggested = props.get("node_index_zero_based")
        matched: int | None = None
        distance = float("inf")
        if suggested is not None:
            index = int(suggested)
            if 0 <= index < len(mesh_xy):
                candidate_distance = float(
                    np.linalg.norm(mesh_xy[index] - feature_xy)
                )
                if candidate_distance <= tolerance_m:
                    matched = index
                    distance = candidate_distance
        if matched is None:
            nearest_distance, nearest = tree.query(feature_xy, k=1)
            distance = float(nearest_distance)
            if distance <= tolerance_m:
                matched = int(nearest)
        if matched is None:
            unmatched_features += 1
            continue
        if matched not in mesh_boundary_nodes:
            off_topological_boundary += 1
            continue
        if np.isfinite(targets[matched]):
            duplicate_matches += 1
            targets[matched] = min(float(targets[matched]), target)
        else:
            targets[matched] = target
        matched_distances.append(distance)
    unmatched_mesh = sorted(
        node for node in mesh_boundary_nodes if not np.isfinite(targets[node])
    )
    return targets, {
        "canonical_feature_count": int(len(features)),
        "matched_unique_mesh_boundary_node_count": int(
            np.count_nonzero(np.isfinite(targets))
        ),
        "unmatched_canonical_feature_count": int(unmatched_features),
        "invalid_or_nonpoint_feature_count": int(nonpoint_features),
        "duplicate_mesh_node_match_count": int(duplicate_matches),
        "matched_off_topological_boundary_count": int(off_topological_boundary),
        "mesh_topological_boundary_node_count": int(len(mesh_boundary_nodes)),
        "unmatched_mesh_boundary_node_count": int(len(unmatched_mesh)),
        "unmatched_mesh_boundary_nodes_zero_based_preview": unmatched_mesh[:50],
        "match_tolerance_m": float(tolerance_m),
        "maximum_match_distance_m": (
            float(max(matched_distances)) if matched_distances else None
        ),
    }


def _edge_triangle_arrays(
    nodes_lonlat: np.ndarray,
    nodes_xy: np.ndarray,
    triangles: np.ndarray,
    edge_to_triangles: Mapping[tuple[int, int], Sequence[int]],
    boundary_edges: set[tuple[int, int]],
    boundary_targets_by_node: np.ndarray,
    size_sampler_xy: Callable[[np.ndarray], np.ndarray],
    interface_limit: float,
    area_change_limit: float,
) -> dict[str, Any]:
    edges = np.asarray(sorted(edge_to_triangles), dtype=np.int64)
    edge_lookup = {
        (int(edge[0]), int(edge[1])): index
        for index, edge in enumerate(edges)
    }
    endpoint_a = nodes_xy[edges[:, 0]]
    endpoint_b = nodes_xy[edges[:, 1]]
    midpoint = 0.5 * (endpoint_a + endpoint_b)
    stacked = np.vstack((endpoint_a, endpoint_b, midpoint))
    sampled = np.asarray(size_sampler_xy(stacked), dtype=float)
    edge_count = len(edges)
    field_samples = np.column_stack(
        (
            sampled[:edge_count],
            sampled[edge_count : 2 * edge_count],
            sampled[2 * edge_count :],
        )
    )
    field_complete = np.all(
        np.isfinite(field_samples) & (field_samples > 0.0),
        axis=1,
    )
    field_target = _positive_row_min(field_samples)

    boundary_samples = np.column_stack(
        (
            boundary_targets_by_node[edges[:, 0]],
            boundary_targets_by_node[edges[:, 1]],
            0.5
            * (
                boundary_targets_by_node[edges[:, 0]]
                + boundary_targets_by_node[edges[:, 1]]
            ),
        )
    )
    boundary_complete = np.all(
        np.isfinite(boundary_samples) & (boundary_samples > 0.0),
        axis=1,
    )
    boundary_target = _positive_row_min(boundary_samples)
    is_boundary = np.asarray(
        [
            (int(edge[0]), int(edge[1])) in boundary_edges
            for edge in edges
        ],
        dtype=bool,
    )
    selected_target = field_target.copy()
    has_boundary_target = is_boundary & np.isfinite(boundary_target)
    selected_target[has_boundary_target] = boundary_target[
        has_boundary_target
    ]
    lengths = np.linalg.norm(endpoint_b - endpoint_a, axis=1)
    edge_l_over_h = np.full(edge_count, np.nan, dtype=float)
    valid = (
        np.isfinite(lengths)
        & (lengths > 0.0)
        & np.isfinite(selected_target)
        & (selected_target > 0.0)
    )
    edge_l_over_h[valid] = lengths[valid] / selected_target[valid]

    pointwise_interface_ratio = np.full(
        boundary_samples.shape,
        np.nan,
        dtype=float,
    )
    interface_ratio = np.full(edge_count, np.nan, dtype=float)
    interface_evaluated = is_boundary & boundary_complete & field_complete
    pointwise_interface_ratio[interface_evaluated] = np.maximum(
        boundary_samples[interface_evaluated]
        / field_samples[interface_evaluated],
        field_samples[interface_evaluated]
        / boundary_samples[interface_evaluated],
    )
    interface_ratio[interface_evaluated] = np.max(
        pointwise_interface_ratio[interface_evaluated],
        axis=1,
    )
    interface_hotspots = np.flatnonzero(
        interface_evaluated & (interface_ratio > interface_limit)
    )

    triangle_edges = np.asarray(
        [
            [
                edge_lookup[_canonical_edge(int(tri[0]), int(tri[1]))],
                edge_lookup[_canonical_edge(int(tri[1]), int(tri[2]))],
                edge_lookup[_canonical_edge(int(tri[2]), int(tri[0]))],
            ]
            for tri in triangles
        ],
        dtype=np.int64,
    )
    triangle_edge_values = edge_l_over_h[triangle_edges]
    triangle_l_over_h = np.full(len(triangles), np.nan, dtype=float)
    finite_any = np.any(np.isfinite(triangle_edge_values), axis=1)
    triangle_l_over_h[finite_any] = np.nanmax(
        triangle_edge_values[finite_any],
        axis=1,
    )

    geometry = triangle_geometry(nodes_xy, triangles)
    areas = np.asarray(geometry["area"], dtype=float)
    area_change = np.full(edge_count, np.nan, dtype=float)
    for edge_index, edge in enumerate(edges):
        attached = edge_to_triangles[(int(edge[0]), int(edge[1]))]
        if len(attached) != 2:
            continue
        a, b = map(int, attached)
        denominator = max(float(areas[a]), float(areas[b]))
        if denominator > 0.0 and np.isfinite(denominator):
            area_change[edge_index] = abs(
                float(areas[a]) - float(areas[b])
            ) / denominator
    area_hotspots = np.flatnonzero(
        np.isfinite(area_change) & (area_change > area_change_limit)
    )
    edge_mid_lonlat = 0.5 * (
        nodes_lonlat[edges[:, 0]] + nodes_lonlat[edges[:, 1]]
    )
    return {
        "edges": edges,
        "edge_to_triangles": edge_to_triangles,
        "edge_l_over_h": edge_l_over_h,
        "triangle_l_over_h": triangle_l_over_h,
        "field_target": field_target,
        "boundary_target": boundary_target,
        "is_boundary": is_boundary,
        "interface_ratio": interface_ratio,
        "area_change": area_change,
        "interface_hotspot_edge_indices": interface_hotspots,
        "area_hotspot_edge_indices": area_hotspots,
        "incomplete_boundary_target_edge_count": int(
            np.count_nonzero(is_boundary & ~boundary_complete)
        ),
        "incomplete_field_target_edge_count": int(
            np.count_nonzero(is_boundary & ~field_complete)
        ),
        "invalid_triangle_l_over_h_count": int(
            np.count_nonzero(~np.isfinite(triangle_l_over_h))
        ),
        "edge_mid_lonlat": edge_mid_lonlat,
    }


def _triangle_graph_distance(
    triangle_count: int,
    edge_to_triangles: Mapping[tuple[int, int], Sequence[int]],
    boundary_edges: set[tuple[int, int]],
) -> np.ndarray:
    adjacency = [set() for _ in range(triangle_count)]
    seeds: set[int] = set()
    for edge, attached_values in edge_to_triangles.items():
        attached = [int(value) for value in attached_values]
        if edge in boundary_edges:
            seeds.update(attached)
        if len(attached) == 2:
            a, b = attached
            adjacency[a].add(b)
            adjacency[b].add(a)
    distance = np.full(triangle_count, -1, dtype=np.int64)
    queue: deque[int] = deque()
    for seed in sorted(seeds):
        distance[seed] = 0
        queue.append(seed)
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency[current]):
            if distance[neighbor] < 0:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    distance[distance < 0] = np.iinfo(np.int32).max
    return distance


def _strata(distance: np.ndarray, transition_rings: int) -> np.ndarray:
    result = np.full(len(distance), "true_interior", dtype=object)
    result[distance <= 1 + transition_rings] = "transition"
    result[distance == 1] = "first_ring"
    result[distance == 0] = "boundary"
    return result


def _write_whole_map(
    path: Path,
    nodes_lonlat: np.ndarray,
    triangles: np.ndarray,
    arrays: Mapping[str, Any],
    triangle_indices: np.ndarray,
    title: str,
    quality: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> None:
    fig, ax = plt.subplots(
        figsize=_figure_size(nodes_lonlat),
        constrained_layout=True,
    )
    _plot_triangle_values(
        fig,
        ax,
        nodes_lonlat,
        triangles,
        triangle_indices,
        np.asarray(arrays["triangle_l_over_h"], dtype=float),
        label="triangle max edge L/h",
        threshold=float(thresholds["l_over_h_maximum_limit"]),
        render_centroids=bool(len(triangle_indices) < len(triangles)),
    )
    _plot_boundary_and_hotspots(
        ax,
        nodes_lonlat,
        arrays,
        area_edge_filter=None,
    )
    ax.set_title(f"{title}\nRaw whole-mesh target-size conformity")
    _map_axes(ax)
    summary = _quality_summary(quality)
    q_value = summary.get("q_l3_sigma")
    q_text = "n/a" if q_value is None else f"{float(q_value):.4f}"
    ax.text(
        0.01,
        0.01,
        (
            f"render: {'sampled triangle centroids' if len(triangle_indices) < len(triangles) else 'triangle faces'}\n"
            f"input quality accepted: {summary.get('accepted')}\n"
            f"q_L3sigma: {q_text}\n"
            f"boundary/field > {thresholds['interface_ratio_limit']:g}: "
            f"{len(arrays['interface_hotspot_edge_indices'])}\n"
            f"area change > {thresholds['area_change_limit']:g}: "
            f"{len(arrays['area_hotspot_edge_indices'])}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#666666"},
    )
    _hotspot_legend(ax)
    fig.savefig(_extended_path(path), dpi=180)
    plt.close(fig)


def _write_transition_map(
    path: Path,
    nodes_lonlat: np.ndarray,
    triangles: np.ndarray,
    arrays: Mapping[str, Any],
    triangle_indices: np.ndarray,
    triangle_distance: np.ndarray,
    title: str,
    thresholds: Mapping[str, float],
) -> None:
    fig, ax = plt.subplots(
        figsize=_figure_size(nodes_lonlat),
        constrained_layout=True,
    )
    _plot_transition_edge_values(
        fig,
        ax,
        nodes_lonlat,
        arrays,
        set(map(int, np.asarray(triangle_indices, dtype=np.int64))),
        maximum_edges=100_000,
        label="boundary + first-ring edge L/h",
        threshold=float(thresholds["l_over_h_maximum_limit"]),
    )
    boundary_first = set(
        map(int, np.flatnonzero(triangle_distance <= 1))
    )
    _plot_boundary_and_hotspots(
        ax,
        nodes_lonlat,
        arrays,
        area_edge_filter=boundary_first,
        draw_boundary=False,
    )
    ax.set_title(
        f"{title}\nBoundary and first-ring target-size transition"
    )
    _map_axes(ax)
    ax.text(
        0.01,
        0.01,
        (
            "Edges touching triangles at graph distance 0-1 from a "
            "topological boundary are colored.\n"
            "Red: configured ratio-limit 1-D/2-D target jump. "
            "Orange: adjacent-area-change hotspot touching this strip."
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#666666"},
    )
    _hotspot_legend(ax)
    fig.savefig(_extended_path(path), dpi=180)
    plt.close(fig)


def _plot_triangle_values(
    fig: Any,
    ax: Any,
    nodes_lonlat: np.ndarray,
    triangles: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    threshold: float,
    render_centroids: bool = False,
) -> None:
    if len(indices) == 0:
        ax.text(
            0.5,
            0.5,
            "No eligible triangles",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return
    selected_values = np.asarray(values[indices], dtype=float)
    finite = selected_values[np.isfinite(selected_values)]
    upper = float(threshold)
    if len(finite):
        upper = max(
            upper,
            min(float(np.quantile(finite, 0.99)), 4.0 * float(threshold)),
        )
    if render_centroids:
        centroids = np.mean(
            nodes_lonlat[np.asarray(triangles[indices], dtype=np.int64)],
            axis=1,
        )
        artist = ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c=np.ma.masked_invalid(selected_values),
            s=2.2,
            marker=".",
            linewidths=0.0,
            cmap="viridis",
            vmin=0.0,
            vmax=upper,
            rasterized=True,
        )
    else:
        triangulation = mtri.Triangulation(
            nodes_lonlat[:, 0],
            nodes_lonlat[:, 1],
            triangles=np.asarray(triangles[indices], dtype=np.int64),
        )
        artist = ax.tripcolor(
            triangulation,
            facecolors=np.ma.masked_invalid(selected_values),
            shading="flat",
            cmap="viridis",
            vmin=0.0,
            vmax=upper,
            edgecolors="none",
            rasterized=True,
        )
    colorbar = fig.colorbar(artist, ax=ax, shrink=0.86, pad=0.02)
    colorbar.set_label(label)


def _plot_transition_edge_values(
    fig: Any,
    ax: Any,
    nodes_lonlat: np.ndarray,
    arrays: Mapping[str, Any],
    eligible_triangles: set[int],
    *,
    maximum_edges: int,
    label: str,
    threshold: float,
) -> None:
    edges = np.asarray(arrays["edges"], dtype=np.int64)
    edge_to_triangles = arrays["edge_to_triangles"]
    eligible = np.asarray(
        [
            index
            for index, edge in enumerate(edges)
            if any(
                int(triangle) in eligible_triangles
                for triangle in edge_to_triangles[
                    (int(edge[0]), int(edge[1]))
                ]
            )
        ],
        dtype=np.int64,
    )
    eligible = _deterministic_sample(eligible, int(maximum_edges))
    if len(eligible) == 0:
        ax.text(
            0.5,
            0.5,
            "No eligible boundary/first-ring edges",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return
    values = np.asarray(arrays["edge_l_over_h"], dtype=float)[eligible]
    finite = values[np.isfinite(values)]
    upper = float(threshold)
    if len(finite):
        upper = max(
            upper,
            min(float(np.quantile(finite, 0.99)), 4.0 * float(threshold)),
        )
    collection = LineCollection(
        [nodes_lonlat[edges[index]] for index in eligible],
        array=np.ma.masked_invalid(values),
        cmap="viridis",
        linewidths=0.72,
        alpha=0.95,
        rasterized=True,
        zorder=4,
    )
    collection.set_clim(0.0, upper)
    ax.add_collection(collection)
    colorbar = fig.colorbar(collection, ax=ax, shrink=0.86, pad=0.02)
    colorbar.set_label(label)


def _plot_boundary_and_hotspots(
    ax: Any,
    nodes_lonlat: np.ndarray,
    arrays: Mapping[str, Any],
    *,
    area_edge_filter: set[int] | None,
    draw_boundary: bool = True,
) -> None:
    edges = np.asarray(arrays["edges"], dtype=np.int64)
    edge_to_triangles = arrays["edge_to_triangles"]
    boundary_segments = [
        nodes_lonlat[edge]
        for edge in edges
        if len(edge_to_triangles[(int(edge[0]), int(edge[1]))]) == 1
    ]
    if boundary_segments and draw_boundary:
        ax.add_collection(
            LineCollection(
                boundary_segments,
                colors="#222222",
                linewidths=0.35,
                alpha=0.85,
                zorder=4,
            )
        )
    interface_indices = _deterministic_sample(
        np.asarray(
            arrays["interface_hotspot_edge_indices"],
            dtype=np.int64,
        ),
        20_000,
    )
    if len(interface_indices):
        ax.add_collection(
            LineCollection(
                [nodes_lonlat[edges[index]] for index in interface_indices],
                colors="#d62728",
                linewidths=2.0,
                alpha=0.95,
                zorder=8,
            )
        )
    area_indices = np.asarray(
        arrays["area_hotspot_edge_indices"],
        dtype=np.int64,
    )
    if area_edge_filter is not None:
        area_indices = np.asarray(
            [
                index
                for index in area_indices
                if any(
                    int(triangle) in area_edge_filter
                    for triangle in edge_to_triangles[
                        (int(edges[index, 0]), int(edges[index, 1]))
                    ]
                )
            ],
            dtype=np.int64,
        )
    area_indices = _deterministic_sample(area_indices, 20_000)
    if len(area_indices):
        ax.add_collection(
            LineCollection(
                [nodes_lonlat[edges[index]] for index in area_indices],
                colors="#ff7f0e",
                linewidths=0.9,
                alpha=0.78,
                zorder=7,
            )
        )


def _hotspot_legend(ax: Any) -> None:
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#222222",
                linewidth=1.0,
                label="topological boundary",
            ),
            Line2D(
                [0],
                [0],
                color="#d62728",
                linewidth=2.0,
                label="boundary/field ratio-limit hotspot",
            ),
            Line2D(
                [0],
                [0],
                color="#ff7f0e",
                linewidth=1.2,
                label="adjacent-area-change hotspot",
            ),
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )


def _map_axes(ax: Any) -> None:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#999999", linewidth=0.35, alpha=0.25)


def _figure_size(nodes_lonlat: np.ndarray) -> tuple[float, float]:
    lon = np.asarray(nodes_lonlat[:, 0], dtype=float)
    lat = np.asarray(nodes_lonlat[:, 1], dtype=float)
    lon_span = max(float(np.ptp(lon)), 1.0e-9)
    lat_span = max(float(np.ptp(lat)), 1.0e-9)
    corrected_aspect = (
        lon_span
        * max(float(np.cos(np.radians(float(np.mean(lat))))), 0.1)
        / lat_span
    )
    width = 14.0
    height = float(np.clip(width / max(corrected_aspect, 0.4), 5.5, 10.5))
    return width, height


def _quality_summary(quality: Mapping[str, Any]) -> dict[str, Any]:
    oceanmesh = quality.get("oceanmesh_quality")
    oceanmesh_source = oceanmesh if isinstance(oceanmesh, Mapping) else {}
    failures = quality.get("failure_taxonomy")
    return {
        "schema_version": quality.get("schema_version"),
        "accepted": bool(quality.get("accepted", False)),
        "evaluation_completed": quality.get("evaluation_completed", True),
        "node_count": _optional_int(quality.get("node_count")),
        "triangle_count": _optional_int(quality.get("triangle_count")),
        "q_l3_sigma": _optional_float(
            oceanmesh_source.get(
                "q_l3_sigma",
                quality.get("q_l3_sigma"),
            )
        ),
        "max_adjacent_area_change": _optional_float(
            quality.get("max_adjacent_area_change")
        ),
        "failure_taxonomy": (
            [str(value) for value in failures]
            if isinstance(failures, list)
            else []
        ),
    }


def _diagnostic_failure_taxonomy(
    *,
    edge_audit: Mapping[str, Any],
    interface_hotspot_count: int,
    area_hotspot_count: int,
    unmatched_boundary_node_count: int,
    invalid_triangle_l_over_h_count: int,
) -> list[str]:
    """Combine every hard transition diagnostic into one status contract."""

    failures = {
        str(value)
        for value in edge_audit.get("failure_taxonomy", [])
    }
    if not bool(edge_audit.get("passed", False)) and not failures:
        failures.add("authoritative_edge_size_audit_failed")
    if int(interface_hotspot_count) > 0:
        failures.add("boundary_field_interface_ratio_limit_exceeded")
    if int(area_hotspot_count) > 0:
        failures.add("adjacent_area_change_above_threshold")
    if int(unmatched_boundary_node_count) > 0:
        failures.add("boundary_target_mapping_incomplete")
    if int(invalid_triangle_l_over_h_count) > 0:
        failures.add("edge_aware_target_size_invalid")
    return sorted(failures)


def _hotspot_records(
    arrays: Mapping[str, Any],
    edge_indices: np.ndarray,
    value_name: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    values = np.asarray(arrays[value_name], dtype=float)
    edges = np.asarray(arrays["edges"], dtype=np.int64)
    midpoints = np.asarray(arrays["edge_mid_lonlat"], dtype=float)
    ranked = sorted(
        (int(index) for index in edge_indices),
        key=lambda index: (-float(values[index]), index),
    )[:limit]
    return [
        {
            "edge_nodes_1based": [
                int(edges[index, 0]) + 1,
                int(edges[index, 1]) + 1,
            ],
            "value": float(values[index]),
            "midpoint_lonlat": [
                float(midpoints[index, 0]),
                float(midpoints[index, 1]),
            ],
        }
        for index in ranked
    ]


def _numeric_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        quantiles = {
            name: None
            for name in ("minimum", "p01", "p05", "p50", "p95", "p99")
        }
        maximum = None
    else:
        sampled = np.quantile(
            finite,
            [0.0, 0.01, 0.05, 0.50, 0.95, 0.99],
        )
        quantiles = {
            name: float(value)
            for name, value in zip(
                ("minimum", "p01", "p05", "p50", "p95", "p99"),
                sampled,
            )
        }
        maximum = float(np.max(finite))
    return {
        "count": int(len(array)),
        "finite_count": int(len(finite)),
        "invalid_count": int(len(array) - len(finite)),
        "quantiles": quantiles,
        "maximum": maximum,
    }


def _positive_row_min(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    valid = np.isfinite(source) & (source > 0.0)
    safe = np.where(valid, source, np.inf)
    result = np.min(safe, axis=1)
    result[~np.any(valid, axis=1)] = np.nan
    return result


def _deterministic_sample(
    raw_indices: np.ndarray,
    maximum_count: int,
) -> np.ndarray:
    indices = np.asarray(raw_indices, dtype=np.int64).reshape(-1)
    if len(indices) <= int(maximum_count):
        return indices.copy()
    positions = (
        np.arange(int(maximum_count), dtype=np.int64) * len(indices)
    ) // int(maximum_count)
    return indices[positions]


def _chain_pairs(
    nodes: Sequence[int],
    cyclic: bool,
) -> list[tuple[int, int]]:
    values = [int(value) for value in nodes]
    pairs = list(zip(values[:-1], values[1:]))
    if cyclic and len(values) > 1:
        pairs.append((values[-1], values[0]))
    return pairs


def _canonical_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _finite_maximum(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if len(finite) else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if np.isfinite(converted) else None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_extended_path(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extended_path(path: Path) -> str:
    """Return an extended-length Windows path without changing manifests."""

    resolved = str(Path(path).resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved.lstrip("\\")
    return "\\\\?\\" + resolved


def _path_exists(path: Path) -> bool:
    return os.path.exists(_extended_path(path))


__all__ = [
    "DEFAULT_MAX_PLOT_TRIANGLES",
    "SCHEMA_VERSION",
    "write_raw_transition_diagnostics",
]
