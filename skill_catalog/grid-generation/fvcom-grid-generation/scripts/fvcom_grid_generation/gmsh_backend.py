"""Isolated Gmsh 4.15.2 backend for the research FVCOM grid experiment.

This module deliberately does not participate in the production meshing route.
The Gmsh import is lazy and exact-version checked, and each call owns a fresh
Gmsh session so that configuration cannot leak between experiment runs.

Coordinates supplied to this module must already be projected in meters.
Boundary loops omit the repeated closing vertex: segment ``i`` joins vertex
``i`` to ``(i + 1) % n``.  Every supplied vertex becomes a distinct CAD point
and every supplied segment becomes a distinct straight CAD line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from pathlib import Path
import re
from types import MappingProxyType, ModuleType
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


PINNED_GMSH_VERSION = "4.15.2"
GMSH_2D_ALGORITHM_NAMES: Mapping[int, str] = MappingProxyType(
    {
        1: "MeshAdapt",
        5: "Delaunay",
        6: "Frontal-Delaunay",
    }
)
CanonicalSizeCallback = Callable[[float, float], float]


class GmshBackendError(RuntimeError):
    """Raised when the isolated research backend cannot complete a run."""

    def __init__(self, message: str, *, logger_output: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.logger_output = tuple(str(item) for item in logger_output)


def gmsh_algorithm_name(algorithm: int) -> str:
    """Return the stable provenance name for an allowed 2-D algorithm."""

    if isinstance(algorithm, bool):
        raise ValueError("Gmsh 2-D algorithm must be one of 1, 5, or 6")
    try:
        normalized = int(algorithm)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Gmsh 2-D algorithm must be one of 1, 5, or 6") from exc
    if normalized != algorithm or normalized not in GMSH_2D_ALGORITHM_NAMES:
        allowed = ", ".join(str(item) for item in GMSH_2D_ALGORITHM_NAMES)
        raise ValueError(f"Gmsh 2-D algorithm must be one of {allowed}")
    return GMSH_2D_ALGORITHM_NAMES[normalized]


def _readonly_xy(values: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2), got {array.shape}")
    if array.shape[0] < 3:
        raise ValueError(f"{name} must contain at least three vertices")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite coordinates")
    if np.linalg.norm(array[0] - array[-1]) <= 1.0e-10:
        raise ValueError(
            f"{name} repeats its first vertex; pass an open coordinate array "
            "because loop closure is implicit"
        )
    edge_lengths = np.linalg.norm(np.roll(array, -1, axis=0) - array, axis=1)
    if np.any(edge_lengths <= 1.0e-10):
        bad = np.flatnonzero(edge_lengths <= 1.0e-10).tolist()
        raise ValueError(f"{name} has zero-length source segments at indices {bad}")
    copied = np.array(array, dtype=float, copy=True)
    copied.setflags(write=False)
    return copied


def _signed_area(vertices_xy: np.ndarray) -> float:
    x = vertices_xy[:, 0]
    y = vertices_xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


@dataclass(frozen=True)
class BoundaryLoopGeometry:
    """One source boundary loop, in its authoritative source orientation."""

    loop_id: str
    vertices_xy: np.ndarray
    role: str = "island"
    island_id: str | None = None
    segment_kinds: tuple[str, ...] = ()
    source_vertex_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        loop_id = str(self.loop_id).strip()
        if not loop_id:
            raise ValueError("loop_id cannot be empty")
        role = str(self.role).strip().lower()
        if role not in {"exterior", "island"}:
            raise ValueError("loop role must be 'exterior' or 'island'")
        if role == "exterior" and self.island_id is not None:
            raise ValueError("the exterior loop cannot have an island_id")
        island_id = None if self.island_id is None else str(self.island_id).strip()
        if role == "island" and not island_id:
            island_id = loop_id
        segment_count = int(np.asarray(self.vertices_xy).shape[0])
        segment_kinds = tuple(str(item).strip() for item in self.segment_kinds)
        if segment_kinds and len(segment_kinds) != segment_count:
            raise ValueError(
                f"segment_kinds for loop {loop_id!r} must be empty or have "
                f"{segment_count} entries"
            )
        if not segment_kinds:
            default_kind = "land" if role == "exterior" else "island"
            segment_kinds = (default_kind,) * segment_count
        if any(not item for item in segment_kinds):
            raise ValueError(f"loop {loop_id!r} has an empty segment kind")
        source_vertex_ids = tuple(
            str(item).strip() for item in self.source_vertex_ids
        )
        if source_vertex_ids and len(source_vertex_ids) != segment_count:
            raise ValueError(
                f"source_vertex_ids for loop {loop_id!r} must be empty or have "
                f"{segment_count} entries"
            )
        if not source_vertex_ids:
            source_vertex_ids = tuple(
                f"{loop_id}:{index}" for index in range(segment_count)
            )
        if any(not item for item in source_vertex_ids):
            raise ValueError(f"loop {loop_id!r} has an empty source vertex ID")
        if len(set(source_vertex_ids)) != len(source_vertex_ids):
            raise ValueError(f"loop {loop_id!r} repeats a source vertex ID")
        object.__setattr__(self, "loop_id", loop_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "island_id", island_id)
        object.__setattr__(self, "segment_kinds", segment_kinds)
        object.__setattr__(self, "source_vertex_ids", source_vertex_ids)
        object.__setattr__(
            self,
            "vertices_xy",
            _readonly_xy(self.vertices_xy, f"vertices_xy for loop {loop_id!r}"),
        )
        if abs(_signed_area(self.vertices_xy)) <= 1.0e-8:
            raise ValueError(f"loop {loop_id!r} has zero signed area")

    @property
    def segment_count(self) -> int:
        return int(self.vertices_xy.shape[0])

    @property
    def signed_area_m2(self) -> float:
        return _signed_area(self.vertices_xy)

    @property
    def xy(self) -> np.ndarray:
        """Compatibility alias used by the experiment orchestrator."""

        return self.vertices_xy


@dataclass(frozen=True)
class SourceLoop:
    """Orchestrator-facing source loop with explicit source metadata."""

    loop_id: str
    xy: np.ndarray
    segment_kinds: tuple[str, ...] = ()
    source_vertex_ids: tuple[str, ...] = ()
    role: str = "island"
    island_id: str | None = None

    def as_boundary_loop(self, *, role: str | None = None) -> BoundaryLoopGeometry:
        return BoundaryLoopGeometry(
            loop_id=self.loop_id,
            vertices_xy=self.xy,
            role=self.role if role is None else role,
            island_id=self.island_id,
            segment_kinds=self.segment_kinds,
            source_vertex_ids=self.source_vertex_ids,
        )


@dataclass(frozen=True)
class OpenBoundaryGeometry:
    """An ordered OBC chain expressed as source exterior-segment indices.

    ``orientation='source'`` traverses each segment from vertex ``i`` to
    ``i + 1``.  ``orientation='reverse'`` traverses it in the opposite
    direction.  Segment indices themselves must be listed in traversal order.
    """

    chain_id: str
    segment_indices: tuple[int, ...]
    kind: str = "open"
    cyclic: bool = False
    orientation: str = "source"

    def __post_init__(self) -> None:
        chain_id = str(self.chain_id).strip()
        kind = str(self.kind).strip()
        orientation = str(self.orientation).strip().lower()
        indices = tuple(int(index) for index in self.segment_indices)
        if not chain_id:
            raise ValueError("open-boundary chain_id cannot be empty")
        if not kind:
            raise ValueError(f"open-boundary {chain_id!r} has an empty kind")
        if not indices:
            raise ValueError(f"open-boundary {chain_id!r} has no source segments")
        if len(set(indices)) != len(indices):
            raise ValueError(f"open-boundary {chain_id!r} repeats a source segment")
        if orientation not in {"source", "reverse"}:
            raise ValueError("open-boundary orientation must be 'source' or 'reverse'")
        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "orientation", orientation)
        object.__setattr__(self, "segment_indices", indices)


@dataclass(frozen=True)
class SourceOpenBoundary:
    """Orchestrator-facing OBC definition over exterior source segments."""

    chain_id: str
    exterior_segment_indices: tuple[int, ...]
    kind: str = "open"
    cyclic: bool = False
    orientation: str = "source"

    def as_open_boundary(self) -> OpenBoundaryGeometry:
        return OpenBoundaryGeometry(
            chain_id=self.chain_id,
            segment_indices=self.exterior_segment_indices,
            kind=self.kind,
            cyclic=self.cyclic,
            orientation=self.orientation,
        )


@dataclass(frozen=True)
class GmshGeometry:
    """Wet-domain BRep input: one exterior, zero or more island holes."""

    exterior: BoundaryLoopGeometry | SourceLoop
    holes: tuple[BoundaryLoopGeometry | SourceLoop, ...] = ()
    open_boundaries: tuple[OpenBoundaryGeometry | SourceOpenBoundary, ...] = ()

    def __post_init__(self) -> None:
        exterior = (
            self.exterior.as_boundary_loop(role="exterior")
            if isinstance(self.exterior, SourceLoop)
            else self.exterior
        )
        holes = tuple(
            hole.as_boundary_loop(role="island") if isinstance(hole, SourceLoop) else hole
            for hole in self.holes
        )
        open_boundaries = tuple(
            chain.as_open_boundary()
            if isinstance(chain, SourceOpenBoundary)
            else chain
            for chain in self.open_boundaries
        )
        if exterior.role != "exterior":
            raise ValueError("GmshGeometry.exterior must have role='exterior'")
        loop_ids = [exterior.loop_id]
        for hole in holes:
            if hole.role != "island":
                raise ValueError(f"hole {hole.loop_id!r} must have role='island'")
            loop_ids.append(hole.loop_id)
        if len(set(loop_ids)) != len(loop_ids):
            raise ValueError("boundary loop IDs must be unique")

        chain_ids = [chain.chain_id for chain in open_boundaries]
        if len(set(chain_ids)) != len(chain_ids):
            raise ValueError("open-boundary chain IDs must be unique")

        segment_count = exterior.segment_count
        claimed: dict[int, str] = {}
        for chain in open_boundaries:
            for index in chain.segment_indices:
                if not 0 <= index < segment_count:
                    raise ValueError(
                        f"open-boundary {chain.chain_id!r} references exterior "
                        f"segment {index}, outside [0, {segment_count})"
                    )
                if index in claimed:
                    raise ValueError(
                        f"exterior segment {index} belongs to both "
                        f"{claimed[index]!r} and {chain.chain_id!r}"
                    )
                claimed[index] = chain.chain_id
            _validate_chain_contiguity(chain, segment_count)
        source_vertex_ids = [
            source_id
            for loop in (exterior, *holes)
            for source_id in loop.source_vertex_ids
        ]
        if len(set(source_vertex_ids)) != len(source_vertex_ids):
            raise ValueError("source vertex IDs must be globally unique")
        object.__setattr__(self, "exterior", exterior)
        object.__setattr__(self, "holes", holes)
        object.__setattr__(self, "open_boundaries", open_boundaries)

    @property
    def loops(self) -> tuple[BoundaryLoopGeometry, ...]:
        return (self.exterior, *self.holes)


def _validate_chain_contiguity(chain: OpenBoundaryGeometry, segment_count: int) -> None:
    step = 1 if chain.orientation == "source" else -1
    for left, right in zip(chain.segment_indices[:-1], chain.segment_indices[1:]):
        expected = (left + step) % segment_count
        if right != expected:
            raise ValueError(
                f"open-boundary {chain.chain_id!r} is not contiguous in "
                f"{chain.orientation!r} orientation: segment {left} should be "
                f"followed by {expected}, not {right}"
            )
    if chain.cyclic:
        if len(chain.segment_indices) != segment_count:
            raise ValueError(
                f"cyclic open-boundary {chain.chain_id!r} must cover the full "
                "exterior loop"
            )
        expected = (chain.segment_indices[-1] + step) % segment_count
        if chain.segment_indices[0] != expected:
            raise ValueError(
                f"cyclic open-boundary {chain.chain_id!r} does not close in "
                "its declared orientation"
            )


@dataclass(frozen=True)
class GmshMeshingConfig:
    """Deterministic first-order configuration for the research portfolio.

    The default remains Gmsh's Frontal-Delaunay algorithm 6.  Algorithms 1
    (MeshAdapt) and 5 (Delaunay) are explicit research alternatives.  When
    ``canonical_size_callback`` is supplied it receives projected ``(x, y)``
    coordinates in meters and supersedes the legacy Threshold/constant field
    for both boundary preflight and two-dimensional generation.
    """

    uniform_target_m: float
    obc_near_size_m: float = 8_000.0
    obc_near_distance_m: float = 10_000.0
    obc_far_distance_m: float = 70_000.0
    algorithm: int = 6
    smoothing_steps: int = 8
    random_seed: int = 1
    model_name: str = "fvcom_gmsh_research"
    constant_field: bool = False
    preserve_source_boundary_discretization: bool = False
    canonical_size_callback: CanonicalSizeCallback | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        numeric = {
            "uniform_target_m": self.uniform_target_m,
            "obc_near_size_m": self.obc_near_size_m,
            "obc_near_distance_m": self.obc_near_distance_m,
            "obc_far_distance_m": self.obc_far_distance_m,
        }
        for name, value in numeric.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.obc_far_distance_m <= self.obc_near_distance_m:
            raise ValueError("obc_far_distance_m must exceed obc_near_distance_m")
        gmsh_algorithm_name(self.algorithm)
        algorithm = int(self.algorithm)
        if int(self.smoothing_steps) != 8:
            raise ValueError("the research contract fixes eight native smoothing steps")
        if int(self.random_seed) < 0:
            raise ValueError("random_seed must be non-negative")
        model_name = str(self.model_name).strip()
        if not model_name:
            raise ValueError("model_name cannot be empty")
        callback = self.canonical_size_callback
        if callback is not None and not callable(callback):
            raise ValueError("canonical_size_callback must be callable")
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "smoothing_steps", 8)
        object.__setattr__(self, "random_seed", int(self.random_seed))
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "constant_field", bool(self.constant_field))
        object.__setattr__(
            self,
            "preserve_source_boundary_discretization",
            bool(self.preserve_source_boundary_discretization),
        )

    @property
    def algorithm_name(self) -> str:
        return gmsh_algorithm_name(self.algorithm)

    @property
    def h_uniform_m(self) -> float:
        return float(self.uniform_target_m)

    @property
    def near_size_m(self) -> float:
        return float(self.obc_near_size_m)

    @property
    def dist_min_m(self) -> float:
        return float(self.obc_near_distance_m)

    @property
    def dist_max_m(self) -> float:
        return float(self.obc_far_distance_m)


@dataclass(frozen=True)
class GmshConfig:
    """Compact orchestrator-facing name for :class:`GmshMeshingConfig`."""

    h_uniform_m: float
    near_size_m: float = 8_000.0
    dist_min_m: float = 10_000.0
    dist_max_m: float = 70_000.0
    constant_field: bool = False
    algorithm: int = 6
    smoothing_steps: int = 8
    random_seed: int = 1
    model_name: str = "fvcom_gmsh_research"
    preserve_source_boundary_discretization: bool = False
    canonical_size_callback: CanonicalSizeCallback | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def as_meshing_config(self) -> GmshMeshingConfig:
        return GmshMeshingConfig(
            uniform_target_m=self.h_uniform_m,
            obc_near_size_m=self.near_size_m,
            obc_near_distance_m=self.dist_min_m,
            obc_far_distance_m=self.dist_max_m,
            algorithm=self.algorithm,
            smoothing_steps=self.smoothing_steps,
            random_seed=self.random_seed,
            model_name=self.model_name,
            constant_field=self.constant_field,
            preserve_source_boundary_discretization=(
                self.preserve_source_boundary_discretization
            ),
            canonical_size_callback=self.canonical_size_callback,
        )

    @property
    def algorithm_name(self) -> str:
        return gmsh_algorithm_name(self.algorithm)


@dataclass(frozen=True)
class BoundaryNodeLineage:
    """Source-segment mapping for one delivered boundary node."""

    mesh_node_id: int
    gmsh_node_tag: int
    loop_id: str
    source_segment_index: int
    source_segment_kind: str
    interpolation_weight: float
    loop_normalized_arclength: float
    chain_normalized_arclength: float | None = None
    is_source_vertex: bool = False


@dataclass(frozen=True)
class DeliveredBoundaryLoop:
    """Ordered delivered nodes for a closed exterior or island loop."""

    loop_id: str
    role: str
    island_id: str | None
    source_orientation: str
    node_ids: tuple[int, ...]
    gmsh_node_tags: tuple[int, ...]
    lineage: tuple[BoundaryNodeLineage, ...]


@dataclass(frozen=True)
class DeliveredOpenBoundary:
    """Ordered delivered nodes for one noncyclic or cyclic OBC chain."""

    chain_id: str
    kind: str
    cyclic: bool
    orientation: str
    source_segment_indices: tuple[int, ...]
    node_ids: tuple[int, ...]
    gmsh_node_tags: tuple[int, ...]
    lineage: tuple[BoundaryNodeLineage, ...]


@dataclass(frozen=True)
class QualityStatistics:
    minimum: float
    p05: float
    median: float
    mean: float
    p95: float
    maximum: float


@dataclass(frozen=True)
class GmshElementQualityReport:
    """Native Gmsh SICN and gamma measures for all delivered triangles."""

    element_tags: tuple[int, ...]
    sicn: QualityStatistics
    gamma: QualityStatistics
    sicn_values: tuple[float, ...] = field(repr=False)
    gamma_values: tuple[float, ...] = field(repr=False)


@dataclass(frozen=True)
class GmshBoundaryPreflight:
    """Measured one-dimensional boundary mesh for node-budget planning."""

    gmsh_version: str
    boundary_node_count: int
    loop_node_counts: Mapping[str, int]
    open_boundary_node_counts: Mapping[str, int]
    logger_output: tuple[str, ...]
    algorithm: int = 6
    algorithm_name: str = "Frontal-Delaunay"
    size_field_mode: str = "gmsh_distance_threshold"
    boundary_discretization_mode: str = "gmsh_size_driven_insertion"


@dataclass(frozen=True)
class GmshMeshResult:
    """Dense first-order triangular mesh and its boundary provenance."""

    gmsh_version: str
    msh_path: Path
    nodes_xy: np.ndarray
    gmsh_node_tags: tuple[int, ...]
    triangles: np.ndarray
    triangle_gmsh_element_tags: tuple[int, ...]
    boundary_node_count_1d: int
    delivered_loops: tuple[DeliveredBoundaryLoop, ...]
    open_boundaries: tuple[DeliveredOpenBoundary, ...]
    source_vertex_node_ids: Mapping[str, int]
    physical_groups: Mapping[str, tuple[int, int]]
    element_quality: GmshElementQualityReport
    logger_output: tuple[str, ...]
    algorithm: int = 6
    algorithm_name: str = "Frontal-Delaunay"
    size_field_mode: str = "gmsh_distance_threshold"
    boundary_discretization_mode: str = "gmsh_size_driven_insertion"

    @property
    def triangles_1based(self) -> np.ndarray:
        return self.triangles

    @property
    def delivered_loop_chains_1based(self) -> Mapping[str, tuple[int, ...]]:
        return {loop.loop_id: loop.node_ids for loop in self.delivered_loops}

    @property
    def delivered_open_boundaries_1based(self) -> Mapping[str, tuple[int, ...]]:
        return {
            boundary.chain_id: boundary.node_ids
            for boundary in self.open_boundaries
        }

    @property
    def lineage(self) -> tuple[BoundaryNodeLineage, ...]:
        return tuple(
            item
            for loop in self.delivered_loops
            for item in loop.lineage
        )


def load_pinned_gmsh() -> ModuleType:
    """Import Gmsh lazily and reject every version except 4.15.2."""

    try:
        gmsh = importlib.import_module("gmsh")
    except Exception as exc:  # Import may fail while loading the native library.
        raise GmshBackendError(
            "Gmsh Python bindings are unavailable; install exactly gmsh==4.15.2 "
            "inside the isolated experiment environment"
        ) from exc
    version = str(getattr(gmsh, "__version__", "")).strip()
    if version != PINNED_GMSH_VERSION:
        raise GmshBackendError(
            f"expected gmsh=={PINNED_GMSH_VERSION}, found {version or 'unknown'}"
        )
    return gmsh


def threshold_target_size(
    distance_m: np.ndarray | Sequence[float] | float,
    config: GmshMeshingConfig | GmshConfig,
) -> np.ndarray:
    """Evaluate the experiment's linear Distance/Threshold size contract.

    The name ``SizeMin`` in Gmsh means the value at ``DistMin``; it is not a
    numerical ordering constraint.  Consequently this function intentionally
    supports ``obc_near_size_m > uniform_target_m``.
    """

    config = _normalized_config(config)
    distance = np.asarray(distance_m, dtype=float)
    fraction = (
        (distance - config.obc_near_distance_m)
        / (config.obc_far_distance_m - config.obc_near_distance_m)
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    return (
        config.obc_near_size_m
        + fraction * (config.uniform_target_m - config.obc_near_size_m)
    )


def measure_boundary_mesh(
    geometry: GmshGeometry,
    config: GmshMeshingConfig | GmshConfig,
) -> GmshBoundaryPreflight:
    """Generate only the one-dimensional mesh and measure its unique nodes."""

    config = _normalized_config(config)
    gmsh = load_pinned_gmsh()
    session = _run_session(
        gmsh,
        geometry,
        config,
        generate_2d=False,
        msh_path=None,
        overwrite=False,
    )
    return GmshBoundaryPreflight(
        gmsh_version=PINNED_GMSH_VERSION,
        boundary_node_count=session.boundary_node_count,
        loop_node_counts={
            loop.loop_id: len(session.curve_samples_by_loop[loop.loop_id].loop_tags)
            for loop in geometry.loops
        },
        open_boundary_node_counts={
            chain.chain_id: len(
                _ordered_chain_samples(
                    geometry.exterior,
                    chain,
                    session.curve_samples_by_loop[geometry.exterior.loop_id],
                )
            )
            for chain in geometry.open_boundaries
        },
        logger_output=session.logger_output,
        algorithm=config.algorithm,
        algorithm_name=config.algorithm_name,
        size_field_mode=_size_field_mode(geometry, config),
        boundary_discretization_mode=_boundary_discretization_mode(config),
    )


def generate_gmsh_mesh(
    geometry: GmshGeometry,
    config: GmshMeshingConfig | GmshConfig,
    msh_path: str | Path,
    *,
    overwrite: bool = False,
) -> GmshMeshResult:
    """Generate the deterministic research mesh and write native MSH 4.1."""

    config = _normalized_config(config)
    output_path = Path(msh_path).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing Gmsh artifact: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh = load_pinned_gmsh()
    session = _run_session(
        gmsh,
        geometry,
        config,
        generate_2d=True,
        msh_path=output_path,
        overwrite=overwrite,
    )
    if session.nodes_xy is None or session.triangles is None:
        raise GmshBackendError("internal error: final Gmsh session has no 2-D mesh")

    dense_by_tag = {
        int(tag): index + 1 for index, tag in enumerate(session.gmsh_node_tags)
    }
    delivered_loops = tuple(
        _delivered_loop(
            loop,
            session.curve_samples_by_loop[loop.loop_id],
            dense_by_tag,
        )
        for loop in geometry.loops
    )
    delivered_open_boundaries = tuple(
        _delivered_chain(
            geometry.exterior,
            chain,
            session.curve_samples_by_loop[geometry.exterior.loop_id],
            dense_by_tag,
        )
        for chain in geometry.open_boundaries
    )
    source_vertex_node_ids = _source_vertex_node_ids(
        geometry,
        session.curve_samples_by_loop,
        dense_by_tag,
    )
    return GmshMeshResult(
        gmsh_version=PINNED_GMSH_VERSION,
        msh_path=output_path,
        nodes_xy=session.nodes_xy,
        gmsh_node_tags=session.gmsh_node_tags,
        triangles=session.triangles,
        triangle_gmsh_element_tags=session.triangle_element_tags,
        boundary_node_count_1d=session.boundary_node_count,
        delivered_loops=delivered_loops,
        open_boundaries=delivered_open_boundaries,
        source_vertex_node_ids=source_vertex_node_ids,
        physical_groups=dict(session.physical_groups),
        element_quality=session.element_quality,
        logger_output=session.logger_output,
        algorithm=config.algorithm,
        algorithm_name=config.algorithm_name,
        size_field_mode=_size_field_mode(geometry, config),
        boundary_discretization_mode=_boundary_discretization_mode(config),
    )


def run_gmsh_attempt(
    geometry: GmshGeometry,
    config: GmshMeshingConfig | GmshConfig,
    msh_path: str | Path,
    *,
    overwrite: bool = False,
) -> GmshMeshResult:
    """Orchestrator-facing alias for one final deterministic Gmsh attempt."""

    return generate_gmsh_mesh(
        geometry,
        config,
        msh_path,
        overwrite=overwrite,
    )


def _normalized_config(
    config: GmshMeshingConfig | GmshConfig,
) -> GmshMeshingConfig:
    if isinstance(config, GmshConfig):
        return config.as_meshing_config()
    if isinstance(config, GmshMeshingConfig):
        return config
    raise TypeError(
        "config must be GmshMeshingConfig or the compact GmshConfig adapter"
    )


@dataclass
class _CurveSamples:
    """Per-loop Gmsh nodes grouped by authoritative source segment."""

    samples_by_segment: tuple[tuple[tuple[int, float], ...], ...]
    segment_lengths: np.ndarray
    cumulative_lengths: np.ndarray
    perimeter: float
    loop_tags: tuple[int, ...]


@dataclass
class _SessionResult:
    boundary_node_count: int
    curve_samples_by_loop: Mapping[str, _CurveSamples]
    physical_groups: Mapping[str, tuple[int, int]]
    logger_output: tuple[str, ...]
    gmsh_node_tags: tuple[int, ...] = ()
    nodes_xy: np.ndarray | None = None
    triangles: np.ndarray | None = None
    triangle_element_tags: tuple[int, ...] = ()
    element_quality: GmshElementQualityReport | None = None


@dataclass
class _CadModel:
    surface_tag: int
    point_tags_by_loop: Mapping[str, tuple[int, ...]]
    line_tags_by_loop: Mapping[str, tuple[int, ...]]
    physical_groups: Mapping[str, tuple[int, int]]


def _run_session(
    gmsh: ModuleType,
    geometry: GmshGeometry,
    config: GmshMeshingConfig,
    *,
    generate_2d: bool,
    msh_path: Path | None,
    overwrite: bool,
) -> _SessionResult:
    if bool(gmsh.isInitialized()):
        raise GmshBackendError(
            "the isolated backend requires ownership of a fresh Gmsh session"
    )
    logger_started = False
    logger_output: tuple[str, ...] = ()
    size_callback_guard: Callable[
        [int, int, float, float, float, float],
        float,
    ] | None = None
    try:
        gmsh.initialize(
            [config.model_name],
            readConfigFiles=False,
            run=False,
            interruptible=False,
        )
        gmsh.logger.start()
        logger_started = True
        _set_deterministic_options(gmsh, config)
        gmsh.model.add(config.model_name)
        cad = _build_cad(gmsh, geometry)
        _configure_boundary_discretization(gmsh, cad, config)
        # Keep the Python callback alive for the full 1-D/2-D session.  Gmsh
        # stores a C callback pointer, not an independently owned Python copy.
        size_callback_guard = _configure_size_field(gmsh, geometry, cad, config)

        gmsh.model.mesh.generate(1)
        curves_1d = _extract_all_curve_samples(gmsh, geometry, cad)
        boundary_tags_1d = {
            tag
            for samples in curves_1d.values()
            for tag in samples.loop_tags
        }
        boundary_count = len(boundary_tags_1d)
        if boundary_count == 0:
            raise GmshBackendError("Gmsh generated an empty one-dimensional boundary")

        if not generate_2d:
            logger_output = _read_logger(gmsh)
            return _SessionResult(
                boundary_node_count=boundary_count,
                curve_samples_by_loop=curves_1d,
                physical_groups=cad.physical_groups,
                logger_output=logger_output,
            )

        gmsh.model.mesh.generate(2)
        curves_2d = _extract_all_curve_samples(gmsh, geometry, cad)
        boundary_tags_2d = {
            tag
            for samples in curves_2d.values()
            for tag in samples.loop_tags
        }
        if boundary_tags_2d != boundary_tags_1d:
            raise GmshBackendError(
                "the delivered 2-D boundary node set differs from the measured "
                "1-D preflight node set"
            )
        (
            gmsh_node_tags,
            nodes_xy,
            triangle_element_tags,
            triangles,
        ) = _extract_dense_triangles(gmsh, cad.surface_tag)
        quality = _extract_element_quality(gmsh, triangle_element_tags)

        if msh_path is None:
            raise GmshBackendError("a final two-dimensional run requires msh_path")
        if msh_path.exists() and not overwrite:
            raise FileExistsError(
                f"refusing to overwrite existing Gmsh artifact: {msh_path}"
            )
        gmsh.write(str(msh_path))
        logger_output = _read_logger(gmsh)
        return _SessionResult(
            boundary_node_count=boundary_count,
            curve_samples_by_loop=curves_2d,
            physical_groups=cad.physical_groups,
            logger_output=logger_output,
            gmsh_node_tags=gmsh_node_tags,
            nodes_xy=nodes_xy,
            triangles=triangles,
            triangle_element_tags=triangle_element_tags,
            element_quality=quality,
        )
    except Exception as exc:
        if logger_started:
            logger_output = _read_logger(gmsh)
        if isinstance(exc, GmshBackendError):
            if not exc.logger_output:
                exc.logger_output = logger_output
            raise
        raise GmshBackendError(
            f"Gmsh {PINNED_GMSH_VERSION} backend failed: {exc}",
            logger_output=logger_output,
        ) from exc
    finally:
        if logger_started:
            try:
                gmsh.logger.stop()
            except Exception:
                pass
        if bool(gmsh.isInitialized()):
            gmsh.finalize()
        size_callback_guard = None


def _read_logger(gmsh: ModuleType) -> tuple[str, ...]:
    try:
        return tuple(str(item) for item in gmsh.logger.get())
    except Exception:
        return ()


def _set_deterministic_options(gmsh: ModuleType, config: GmshMeshingConfig) -> None:
    options = {
        "General.Terminal": 0,
        "General.NumThreads": 1,
        "Mesh.MaxNumThreads1D": 1,
        "Mesh.MaxNumThreads2D": 1,
        "Mesh.MaxNumThreads3D": 1,
        "Mesh.Reproducible": 1,
        "Mesh.RandomSeed": config.random_seed,
        "Mesh.Algorithm": config.algorithm,
        "Mesh.AlgorithmSwitchOnFailure": 0,
        "Mesh.Smoothing": config.smoothing_steps,
        "Mesh.ElementOrder": 1,
        "Mesh.RecombineAll": 0,
        "Mesh.Optimize": 0,
        "Mesh.OptimizeNetgen": 0,
        "Mesh.MeshSizeFromPoints": 0,
        "Mesh.MeshSizeFromCurvature": 0,
        "Mesh.MeshSizeExtendFromBoundary": 0,
        "Mesh.MshFileVersion": 4.1,
        "Mesh.Binary": 0,
        "Mesh.SaveAll": 1,
    }
    for name, value in options.items():
        gmsh.option.setNumber(name, float(value))


def _surface_curve_tags(line_tags: Sequence[int], signed_area: float, *, hole: bool) -> list[int]:
    desired_positive = not hole
    is_positive = signed_area > 0.0
    if is_positive == desired_positive:
        return [int(tag) for tag in line_tags]
    return [-int(tag) for tag in reversed(line_tags)]


def _build_cad(gmsh: ModuleType, geometry: GmshGeometry) -> _CadModel:
    point_tags_by_loop: dict[str, tuple[int, ...]] = {}
    line_tags_by_loop: dict[str, tuple[int, ...]] = {}
    curve_loop_tags: list[int] = []

    for loop in geometry.loops:
        point_tags = tuple(
            int(gmsh.model.geo.addPoint(float(x), float(y), 0.0))
            for x, y in loop.vertices_xy
        )
        line_tags = tuple(
            int(
                gmsh.model.geo.addLine(
                    point_tags[index],
                    point_tags[(index + 1) % len(point_tags)],
                )
            )
            for index in range(len(point_tags))
        )
        oriented_tags = _surface_curve_tags(
            line_tags,
            loop.signed_area_m2,
            hole=(loop.role == "island"),
        )
        curve_loop_tags.append(int(gmsh.model.geo.addCurveLoop(oriented_tags)))
        point_tags_by_loop[loop.loop_id] = point_tags
        line_tags_by_loop[loop.loop_id] = line_tags

    surface_tag = int(gmsh.model.geo.addPlaneSurface(curve_loop_tags))
    gmsh.model.geo.synchronize()

    physical_groups: dict[str, tuple[int, int]] = {}
    _add_physical_group(
        gmsh,
        physical_groups,
        dim=2,
        entity_tags=[surface_tag],
        name="WET_DOMAIN",
    )

    exterior_line_tags = line_tags_by_loop[geometry.exterior.loop_id]
    open_indices: set[int] = set()
    for chain in geometry.open_boundaries:
        tags = [exterior_line_tags[index] for index in chain.segment_indices]
        open_indices.update(chain.segment_indices)
        _add_physical_group(
            gmsh,
            physical_groups,
            dim=1,
            entity_tags=tags,
            name=f"OBC_{_physical_token(chain.chain_id)}",
        )

    land_tags = [
        tag for index, tag in enumerate(exterior_line_tags) if index not in open_indices
    ]
    if land_tags:
        _add_physical_group(
            gmsh,
            physical_groups,
            dim=1,
            entity_tags=land_tags,
            name="LAND_COASTLINE",
        )

    for hole in geometry.holes:
        _add_physical_group(
            gmsh,
            physical_groups,
            dim=1,
            entity_tags=line_tags_by_loop[hole.loop_id],
            name=f"ISLAND_{_physical_token(hole.island_id or hole.loop_id)}",
        )

    return _CadModel(
        surface_tag=surface_tag,
        point_tags_by_loop=point_tags_by_loop,
        line_tags_by_loop=line_tags_by_loop,
        physical_groups=physical_groups,
    )


def _physical_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip()).strip("_")
    return token or "UNNAMED"


def _add_physical_group(
    gmsh: ModuleType,
    registry: dict[str, tuple[int, int]],
    *,
    dim: int,
    entity_tags: Iterable[int],
    name: str,
) -> None:
    if name in registry:
        raise GmshBackendError(f"duplicate physical-group name {name!r}")
    tags = [int(tag) for tag in entity_tags]
    if not tags:
        raise GmshBackendError(f"physical group {name!r} has no entities")
    physical_tag = int(gmsh.model.addPhysicalGroup(dim, tags))
    gmsh.model.setPhysicalName(dim, physical_tag, name)
    registry[name] = (int(dim), physical_tag)


def _configure_size_field(
    gmsh: ModuleType,
    geometry: GmshGeometry,
    cad: _CadModel,
    config: GmshMeshingConfig,
) -> Callable[[int, int, float, float, float, float], float] | None:
    if config.canonical_size_callback is not None:
        callback = _gmsh_size_callback(config.canonical_size_callback)
        gmsh.model.mesh.setSizeCallback(callback)
        return callback
    if geometry.open_boundaries and config.constant_field:
        raise GmshBackendError(
            "constant_field=True is reserved for closed, zero-OBC domains"
        )
    if geometry.open_boundaries:
        exterior_lines = cad.line_tags_by_loop[geometry.exterior.loop_id]
        obc_lines = sorted(
            {
                int(exterior_lines[index])
                for chain in geometry.open_boundaries
                for index in chain.segment_indices
            }
        )
        distance_field = int(gmsh.model.mesh.field.add("Distance"))
        gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", obc_lines)

        threshold_field = int(gmsh.model.mesh.field.add("Threshold"))
        gmsh.model.mesh.field.setNumber(
            threshold_field, "InField", float(distance_field)
        )
        gmsh.model.mesh.field.setNumber(
            threshold_field, "SizeMin", float(config.obc_near_size_m)
        )
        gmsh.model.mesh.field.setNumber(
            threshold_field, "SizeMax", float(config.uniform_target_m)
        )
        gmsh.model.mesh.field.setNumber(
            threshold_field, "DistMin", float(config.obc_near_distance_m)
        )
        gmsh.model.mesh.field.setNumber(
            threshold_field, "DistMax", float(config.obc_far_distance_m)
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    else:
        constant_field = int(gmsh.model.mesh.field.add("MathEval"))
        gmsh.model.mesh.field.setString(
            constant_field,
            "F",
            format(float(config.uniform_target_m), ".17g"),
        )
        gmsh.model.mesh.field.setAsBackgroundMesh(constant_field)
    return None


def _configure_boundary_discretization(
    gmsh: ModuleType,
    cad: _CadModel,
    config: GmshMeshingConfig,
) -> None:
    """Optionally retain exactly the two source endpoints of every CAD line."""

    if not config.preserve_source_boundary_discretization:
        return
    for line_tags in cad.line_tags_by_loop.values():
        for line_tag in line_tags:
            gmsh.model.mesh.setTransfiniteCurve(int(line_tag), 2)


def _boundary_discretization_mode(config: GmshMeshingConfig) -> str:
    return (
        "preserve_source_segments_two_endpoints"
        if config.preserve_source_boundary_discretization
        else "gmsh_size_driven_insertion"
    )


def _gmsh_size_callback(
    canonical_callback: CanonicalSizeCallback,
) -> Callable[[int, int, float, float, float, float], float]:
    """Adapt a projected ``h(x, y)`` sampler to Gmsh's six-argument API."""

    def callback(
        dim: int,
        tag: int,
        x: float,
        y: float,
        z: float,
        lc: float,
    ) -> float:
        del dim, tag, z, lc
        x_value = float(x)
        y_value = float(y)
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise GmshBackendError(
                "Gmsh requested canonical size at non-finite projected coordinates"
            )
        try:
            size = float(canonical_callback(x_value, y_value))
        except Exception as exc:
            raise GmshBackendError(
                "canonical projected size callback failed at "
                f"({x_value:.17g}, {y_value:.17g})"
            ) from exc
        if not math.isfinite(size) or size <= 0.0:
            raise GmshBackendError(
                "canonical projected size callback returned a non-finite or "
                f"non-positive value {size!r} at "
                f"({x_value:.17g}, {y_value:.17g})"
            )
        return size

    return callback


def _size_field_mode(
    geometry: GmshGeometry,
    config: GmshMeshingConfig,
) -> str:
    if config.canonical_size_callback is not None:
        return "canonical_projected_callback"
    if geometry.open_boundaries:
        return "gmsh_distance_threshold"
    return "gmsh_constant"


def _extract_all_curve_samples(
    gmsh: ModuleType,
    geometry: GmshGeometry,
    cad: _CadModel,
) -> dict[str, _CurveSamples]:
    return {
        loop.loop_id: _extract_curve_samples(
            gmsh,
            loop,
            cad.line_tags_by_loop[loop.loop_id],
        )
        for loop in geometry.loops
    }


def _extract_curve_samples(
    gmsh: ModuleType,
    loop: BoundaryLoopGeometry,
    line_tags: Sequence[int],
) -> _CurveSamples:
    samples_by_segment: list[tuple[tuple[int, float], ...]] = []
    segment_lengths = np.linalg.norm(
        np.roll(loop.vertices_xy, -1, axis=0) - loop.vertices_xy,
        axis=1,
    )
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    unique_loop_tags: set[int] = set()

    for index, line_tag in enumerate(line_tags):
        node_tags_raw, coordinates_raw, _ = gmsh.model.mesh.getNodes(
            1,
            int(line_tag),
            True,
            False,
        )
        node_tags = np.asarray(node_tags_raw, dtype=np.int64).reshape(-1)
        coordinates = np.asarray(coordinates_raw, dtype=float).reshape(-1, 3)[:, :2]
        if node_tags.size == 0:
            raise GmshBackendError(
                f"source segment {index} of loop {loop.loop_id!r} has no mesh nodes"
            )
        if coordinates.shape[0] != node_tags.size:
            raise GmshBackendError("Gmsh returned inconsistent curve node arrays")
        start = loop.vertices_xy[index]
        vector = loop.vertices_xy[(index + 1) % loop.segment_count] - start
        length_squared = float(np.dot(vector, vector))
        weights = ((coordinates - start) @ vector) / length_squared
        if np.any(weights < -1.0e-7) or np.any(weights > 1.0 + 1.0e-7):
            raise GmshBackendError(
                f"curve nodes leave source segment {index} of loop {loop.loop_id!r}"
            )
        weights = np.clip(weights, 0.0, 1.0)
        by_tag: dict[int, float] = {}
        for tag, weight in zip(node_tags.tolist(), weights.tolist()):
            tag_int = int(tag)
            weight_float = float(weight)
            previous = by_tag.get(tag_int)
            if previous is None or abs(weight_float - 0.5) < abs(previous - 0.5):
                by_tag[tag_int] = weight_float
        ordered = tuple(
            sorted(by_tag.items(), key=lambda item: (item[1], item[0]))
        )
        if ordered[0][1] > 1.0e-7 or ordered[-1][1] < 1.0 - 1.0e-7:
            raise GmshBackendError(
                f"source endpoints are missing from segment {index} of "
                f"loop {loop.loop_id!r}"
            )
        samples_by_segment.append(ordered)
        unique_loop_tags.update(by_tag)

    loop_samples = _ordered_loop_samples(
        tuple(samples_by_segment),
        segment_lengths,
        cumulative,
    )
    if len({tag for tag, _, _ in loop_samples}) != len(unique_loop_tags):
        raise GmshBackendError(
            f"not every delivered node of loop {loop.loop_id!r} received lineage"
        )
    return _CurveSamples(
        samples_by_segment=tuple(samples_by_segment),
        segment_lengths=np.asarray(segment_lengths, dtype=float),
        cumulative_lengths=np.asarray(cumulative, dtype=float),
        perimeter=float(cumulative[-1]),
        loop_tags=tuple(tag for tag, _, _ in loop_samples),
    )


def _ordered_loop_samples(
    samples_by_segment: tuple[tuple[tuple[int, float], ...], ...],
    segment_lengths: np.ndarray,
    cumulative: np.ndarray,
) -> tuple[tuple[int, int, float], ...]:
    del segment_lengths, cumulative
    ordered: list[tuple[int, int, float]] = []
    for segment_index, samples in enumerate(samples_by_segment):
        for tag, weight in samples:
            if weight >= 1.0 - 1.0e-9:
                continue
            ordered.append((int(tag), segment_index, float(weight)))
    return tuple(ordered)


def _ordered_chain_samples(
    exterior: BoundaryLoopGeometry,
    chain: OpenBoundaryGeometry,
    curve_samples: _CurveSamples,
) -> tuple[tuple[int, int, float, float], ...]:
    ordered: list[tuple[int, int, float, float]] = []
    chain_lengths = [
        float(curve_samples.segment_lengths[index])
        for index in chain.segment_indices
    ]
    chain_total = float(sum(chain_lengths))
    cumulative = 0.0
    forward = chain.orientation == "source"

    for position, segment_index in enumerate(chain.segment_indices):
        samples = curve_samples.samples_by_segment[segment_index]
        traversal_samples = samples if forward else tuple(reversed(samples))
        is_last = position == len(chain.segment_indices) - 1
        for tag, source_weight in traversal_samples:
            traversal_weight = source_weight if forward else 1.0 - source_weight
            is_segment_end = traversal_weight >= 1.0 - 1.0e-9
            if is_segment_end and (not is_last or chain.cyclic):
                continue
            chain_fraction = (
                cumulative
                + traversal_weight * curve_samples.segment_lengths[segment_index]
            ) / chain_total
            if chain.cyclic and chain_fraction >= 1.0 - 1.0e-12:
                chain_fraction = 0.0
            ordered.append(
                (
                    int(tag),
                    int(segment_index),
                    float(source_weight),
                    float(chain_fraction),
                )
            )
        cumulative += curve_samples.segment_lengths[segment_index]

    tags = [sample[0] for sample in ordered]
    if len(tags) != len(set(tags)):
        raise GmshBackendError(
            f"open-boundary {chain.chain_id!r} contains repeated delivered nodes"
        )
    if not ordered:
        raise GmshBackendError(
            f"open-boundary {chain.chain_id!r} has no delivered nodes"
        )
    del exterior
    return tuple(ordered)


def _loop_orientation(loop: BoundaryLoopGeometry) -> str:
    return "counterclockwise" if loop.signed_area_m2 > 0.0 else "clockwise"


def _delivered_loop(
    loop: BoundaryLoopGeometry,
    samples: _CurveSamples,
    dense_by_tag: Mapping[int, int],
) -> DeliveredBoundaryLoop:
    raw = _ordered_loop_samples(
        samples.samples_by_segment,
        samples.segment_lengths,
        samples.cumulative_lengths,
    )
    lineage: list[BoundaryNodeLineage] = []
    for tag, segment_index, weight in raw:
        loop_fraction = (
            samples.cumulative_lengths[segment_index]
            + weight * samples.segment_lengths[segment_index]
        ) / samples.perimeter
        lineage.append(
            BoundaryNodeLineage(
                mesh_node_id=int(dense_by_tag[tag]),
                gmsh_node_tag=int(tag),
                loop_id=loop.loop_id,
                source_segment_index=int(segment_index),
                source_segment_kind=loop.segment_kinds[segment_index],
                interpolation_weight=float(weight),
                loop_normalized_arclength=float(loop_fraction),
                is_source_vertex=bool(weight <= 1.0e-9),
            )
        )
    return DeliveredBoundaryLoop(
        loop_id=loop.loop_id,
        role=loop.role,
        island_id=loop.island_id,
        source_orientation=_loop_orientation(loop),
        node_ids=tuple(item.mesh_node_id for item in lineage),
        gmsh_node_tags=tuple(item.gmsh_node_tag for item in lineage),
        lineage=tuple(lineage),
    )


def _delivered_chain(
    exterior: BoundaryLoopGeometry,
    chain: OpenBoundaryGeometry,
    samples: _CurveSamples,
    dense_by_tag: Mapping[int, int],
) -> DeliveredOpenBoundary:
    raw = _ordered_chain_samples(exterior, chain, samples)
    lineage: list[BoundaryNodeLineage] = []
    for tag, segment_index, source_weight, chain_fraction in raw:
        loop_fraction = (
            samples.cumulative_lengths[segment_index]
            + source_weight * samples.segment_lengths[segment_index]
        ) / samples.perimeter
        lineage.append(
            BoundaryNodeLineage(
                mesh_node_id=int(dense_by_tag[tag]),
                gmsh_node_tag=int(tag),
                loop_id=exterior.loop_id,
                source_segment_index=int(segment_index),
                source_segment_kind=exterior.segment_kinds[segment_index],
                interpolation_weight=float(source_weight),
                loop_normalized_arclength=float(loop_fraction),
                chain_normalized_arclength=float(chain_fraction),
                is_source_vertex=bool(
                    source_weight <= 1.0e-9 or source_weight >= 1.0 - 1.0e-9
                ),
            )
        )
    return DeliveredOpenBoundary(
        chain_id=chain.chain_id,
        kind=chain.kind,
        cyclic=bool(chain.cyclic),
        orientation=chain.orientation,
        source_segment_indices=chain.segment_indices,
        node_ids=tuple(item.mesh_node_id for item in lineage),
        gmsh_node_tags=tuple(item.gmsh_node_tag for item in lineage),
        lineage=tuple(lineage),
    )


def _source_vertex_node_ids(
    geometry: GmshGeometry,
    curve_samples_by_loop: Mapping[str, _CurveSamples],
    dense_by_tag: Mapping[int, int],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for loop in geometry.loops:
        samples = curve_samples_by_loop[loop.loop_id]
        for vertex_index, segment_samples in enumerate(samples.samples_by_segment):
            candidates = [
                tag for tag, weight in segment_samples if weight <= 1.0e-7
            ]
            if len(candidates) != 1:
                raise GmshBackendError(
                    f"source vertex {vertex_index} of loop {loop.loop_id!r} "
                    f"maps to {len(candidates)} Gmsh nodes"
                )
            result[loop.source_vertex_ids[vertex_index]] = int(
                dense_by_tag[int(candidates[0])]
            )
    return result


def _extract_dense_triangles(
    gmsh: ModuleType,
    surface_tag: int,
) -> tuple[tuple[int, ...], np.ndarray, tuple[int, ...], np.ndarray]:
    node_tags_raw, coordinates_raw, _ = gmsh.model.mesh.getNodes(-1, -1, False, False)
    node_tags = np.asarray(node_tags_raw, dtype=np.int64).reshape(-1)
    coordinates = np.asarray(coordinates_raw, dtype=float).reshape(-1, 3)
    if node_tags.size == 0:
        raise GmshBackendError("Gmsh generated no mesh nodes")
    order = np.argsort(node_tags, kind="stable")
    node_tags = node_tags[order]
    nodes_xy = coordinates[order, :2]
    dense_by_tag = {int(tag): index + 1 for index, tag in enumerate(node_tags)}

    element_types, element_tags_sets, element_nodes_sets = gmsh.model.mesh.getElements(
        2,
        int(surface_tag),
    )
    triangle_tags: list[int] = []
    triangle_nodes: list[tuple[int, int, int]] = []
    for element_type, tags_raw, nodes_raw in zip(
        element_types,
        element_tags_sets,
        element_nodes_sets,
    ):
        properties = gmsh.model.mesh.getElementProperties(int(element_type))
        name = str(properties[0])
        order_value = int(properties[2])
        node_count = int(properties[3])
        primary_count = int(properties[5])
        tags = np.asarray(tags_raw, dtype=np.int64).reshape(-1)
        if tags.size == 0:
            continue
        if order_value != 1 or node_count != 3 or primary_count != 3:
            raise GmshBackendError(
                f"unexpected 2-D element type {element_type} ({name}); "
                "the experiment permits first-order triangles only"
            )
        nodes = np.asarray(nodes_raw, dtype=np.int64).reshape(-1, node_count)
        for element_tag, row in zip(tags.tolist(), nodes.tolist()):
            triangle_tags.append(int(element_tag))
            try:
                triangle_nodes.append(tuple(dense_by_tag[int(tag)] for tag in row))
            except KeyError as exc:
                raise GmshBackendError(
                    f"triangle references unknown Gmsh node tag {exc.args[0]}"
                ) from exc
    if not triangle_tags:
        raise GmshBackendError("Gmsh generated no first-order triangles")

    element_order = np.argsort(np.asarray(triangle_tags), kind="stable")
    triangle_tags_array = np.asarray(triangle_tags, dtype=np.int64)[element_order]
    triangles = np.asarray(triangle_nodes, dtype=np.int64)[element_order]
    zero_based = triangles - 1
    a = nodes_xy[zero_based[:, 0]]
    b = nodes_xy[zero_based[:, 1]]
    c = nodes_xy[zero_based[:, 2]]
    twice_area = (
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    if np.any(np.abs(twice_area) <= 1.0e-12):
        raise GmshBackendError("Gmsh produced zero-area triangles")
    clockwise = twice_area < 0.0
    if np.any(clockwise):
        swapped = triangles[clockwise, 1].copy()
        triangles[clockwise, 1] = triangles[clockwise, 2]
        triangles[clockwise, 2] = swapped

    nodes_xy = np.asarray(nodes_xy, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    nodes_xy.setflags(write=False)
    triangles.setflags(write=False)
    return (
        tuple(int(tag) for tag in node_tags.tolist()),
        nodes_xy,
        tuple(int(tag) for tag in triangle_tags_array.tolist()),
        triangles,
    )


def _quality_statistics(values: np.ndarray, name: str) -> QualityStatistics:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise GmshBackendError(f"Gmsh returned invalid {name} quality values")
    return QualityStatistics(
        minimum=float(np.min(array)),
        p05=float(np.quantile(array, 0.05)),
        median=float(np.median(array)),
        mean=float(np.mean(array)),
        p95=float(np.quantile(array, 0.95)),
        maximum=float(np.max(array)),
    )


def _extract_element_quality(
    gmsh: ModuleType,
    triangle_element_tags: Sequence[int],
) -> GmshElementQualityReport:
    element_tags = np.asarray(triangle_element_tags, dtype=np.int64)
    sicn = np.asarray(
        gmsh.model.mesh.getElementQualities(element_tags, "minSICN"),
        dtype=float,
    )
    gamma = np.asarray(
        gmsh.model.mesh.getElementQualities(element_tags, "gamma"),
        dtype=float,
    )
    if sicn.size != element_tags.size or gamma.size != element_tags.size:
        raise GmshBackendError("Gmsh returned incomplete element-quality arrays")
    return GmshElementQualityReport(
        element_tags=tuple(int(tag) for tag in element_tags.tolist()),
        sicn=_quality_statistics(sicn, "SICN"),
        gamma=_quality_statistics(gamma, "gamma"),
        sicn_values=tuple(float(value) for value in sicn.tolist()),
        gamma_values=tuple(float(value) for value in gamma.tolist()),
    )


__all__ = [
    "CanonicalSizeCallback",
    "GMSH_2D_ALGORITHM_NAMES",
    "PINNED_GMSH_VERSION",
    "BoundaryLoopGeometry",
    "BoundaryNodeLineage",
    "DeliveredBoundaryLoop",
    "DeliveredOpenBoundary",
    "GmshBackendError",
    "GmshBoundaryPreflight",
    "GmshConfig",
    "GmshElementQualityReport",
    "GmshGeometry",
    "GmshMeshResult",
    "GmshMeshingConfig",
    "OpenBoundaryGeometry",
    "QualityStatistics",
    "SourceLoop",
    "SourceOpenBoundary",
    "generate_gmsh_mesh",
    "gmsh_algorithm_name",
    "load_pinned_gmsh",
    "measure_boundary_mesh",
    "run_gmsh_attempt",
    "threshold_target_size",
]
