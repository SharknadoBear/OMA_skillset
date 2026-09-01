"""Boundary-loop ingestion and boundary-node assignment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union

from .boundary_topology import (
    BoundaryTopologyCompensation,
    normalize_boundary_topology,
)
from .projection import LocalProjection, local_utm_projection, project_geometry, project_points, unproject_geometry, unproject_points


@dataclass(frozen=True)
class BoundaryConfig:
    land_spacing_m: float = 50.0
    open_spacing_m: float = 3000.0
    island_spacing_m: float = 50.0


@dataclass(frozen=True)
class BoundaryPackage:
    domain_polygon_lonlat: Polygon
    open_boundary_lonlat: LineString | MultiLineString
    land_boundary_lonlat: LineString | MultiLineString
    frame_boundary_lonlat: LineString | MultiLineString
    island_polygons_lonlat: list[Polygon]
    source_gpkg: str
    projection: LocalProjection


@dataclass(frozen=True)
class OpenBoundaryChain:
    """Ordered open-boundary contract used by multi-gate research routes."""

    chain_id: str
    node_indices: tuple[int, ...]
    kind: str = "exchange"
    cyclic: bool = False
    orientation: str = "forward"


@dataclass
class BoundaryNodes:
    xy: np.ndarray
    lonlat: np.ndarray
    kinds: list[str]
    target_spacing_m: np.ndarray
    exterior_indices: list[int]
    open_boundary_indices: list[int]
    constraint_chains: list[list[int]]
    domain_polygon_xy: Polygon
    open_boundary_xy: LineString | MultiLineString
    land_boundary_xy: LineString | MultiLineString
    island_polygons_xy: list[Polygon]
    projection: LocalProjection
    hard_anchor_mask: np.ndarray | None = None
    adaptive_resolution: bool = False
    source_resolution_manifest: str | None = None
    resolution_profile: str = "legacy"
    metadata: dict[str, np.ndarray] | None = None
    passage_diagnostics: list[dict[str, Any]] | None = None
    open_boundaries: list[OpenBoundaryChain] | None = None
    topology_compensation: BoundaryTopologyCompensation | None = None


def load_boundary_package(path: str | Path) -> BoundaryPackage:
    """Read the model-boundary-loop GeoPackage from fvcom-bdry-arc."""
    path = Path(path)
    layers = set(gpd.list_layers(path)["name"])
    if "model_domain_polygon" not in layers:
        raise ValueError(f"{path} does not contain model_domain_polygon")
    domain = gpd.read_file(path, layer="model_domain_polygon").to_crs("EPSG:4326").geometry.iloc[0]
    if not isinstance(domain, Polygon):
        raise ValueError("model_domain_polygon must contain a Polygon")
    segments = gpd.read_file(path, layer="model_outer_boundary_segments").to_crs("EPSG:4326") if "model_outer_boundary_segments" in layers else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    open_lines = []
    land_lines = []
    frame_lines = []
    if not segments.empty and "segment_class" in segments:
        for _, row in segments.iterrows():
            geom = row.geometry
            klass = str(row.get("segment_class", "")).lower()
            if geom is None or geom.is_empty:
                continue
            if klass == "open_boundary":
                open_lines.append(geom)
            elif klass == "frame_clip_boundary":
                frame_lines.append(geom)
            else:
                land_lines.append(geom)
    if not open_lines and "source_open_boundary_arc" in layers:
        src = gpd.read_file(path, layer="source_open_boundary_arc").to_crs("EPSG:4326")
        open_lines = [geom for geom in src.geometry if geom is not None and not geom.is_empty]
    islands = []
    if "island_boundary_polygons" in layers:
        island_gdf = gpd.read_file(path, layer="island_boundary_polygons").to_crs("EPSG:4326")
        islands = [geom for geom in island_gdf.geometry if isinstance(geom, Polygon) and not geom.is_empty]
    bbox = tuple(float(v) for v in domain.bounds)
    projection = local_utm_projection((bbox[0], bbox[1], bbox[2], bbox[3]))
    return BoundaryPackage(
        domain_polygon_lonlat=domain,
        open_boundary_lonlat=unary_union(open_lines) if open_lines else LineString(),
        land_boundary_lonlat=unary_union(land_lines) if land_lines else LineString(domain.exterior.coords),
        frame_boundary_lonlat=unary_union(frame_lines) if frame_lines else LineString(),
        island_polygons_lonlat=islands,
        source_gpkg=str(path),
        projection=projection,
    )


def prepare_boundary_nodes(package: BoundaryPackage, config: BoundaryConfig) -> BoundaryNodes:
    """Densify model-domain boundary and classify open/land/island nodes."""
    projection = package.projection
    domain_xy = project_geometry(package.domain_polygon_lonlat, projection)
    simplify_tol = max(0.0, 0.35 * min(config.land_spacing_m, config.island_spacing_m))
    if simplify_tol > 0.0:
        domain_xy = domain_xy.simplify(simplify_tol, preserve_topology=True).buffer(0)
    open_xy = project_geometry(package.open_boundary_lonlat, projection) if not package.open_boundary_lonlat.is_empty else LineString()
    land_xy = project_geometry(package.land_boundary_lonlat, projection) if not package.land_boundary_lonlat.is_empty else LineString(domain_xy.exterior.coords)
    islands_xy = [project_geometry(poly, projection).simplify(simplify_tol, preserve_topology=True).buffer(0) for poly in package.island_polygons_lonlat]
    open_tol = max(config.land_spacing_m * 2.0, config.open_spacing_m * 0.15, 25.0)

    points: list[tuple[float, float]] = []
    kinds: list[str] = []
    exterior_indices: list[int] = []
    open_indices: list[int] = []
    constraint_chains: list[list[int]] = []

    exterior_coords = list(domain_xy.exterior.coords)
    exterior_chain: list[int] = []
    for a, b in zip(exterior_coords[:-1], exterior_coords[1:]):
        seg = LineString([a, b])
        midpoint = seg.interpolate(0.5, normalized=True)
        kind = "open" if (not open_xy.is_empty and open_xy.distance(midpoint) <= open_tol) else "land"
        spacing = config.open_spacing_m if kind == "open" else config.land_spacing_m
        samples = _sample_segment(seg, spacing, include_end=False)
        for xy in samples:
            idx = _append_point(points, kinds, xy, kind)
            exterior_chain.append(idx)
            exterior_indices.append(idx)
            if kind == "open":
                open_indices.append(idx)
    if exterior_chain:
        constraint_chains.append(exterior_chain)

    for poly in islands_xy:
        ring = list(poly.exterior.coords)
        chain: list[int] = []
        for a, b in zip(ring[:-1], ring[1:]):
            for xy in _sample_segment(LineString([a, b]), config.island_spacing_m, include_end=False):
                idx = _append_point(points, kinds, xy, "island")
                chain.append(idx)
        if chain:
            constraint_chains.append(chain)

    xy_arr = np.asarray(points, dtype=float)
    lonlat = unproject_points(xy_arr, projection) if len(xy_arr) else np.empty((0, 2), dtype=float)
    hard_anchor_mask = _boundary_hard_anchor_mask(kinds, constraint_chains)
    open_boundaries = _open_boundary_chains(exterior_chain, kinds)
    return BoundaryNodes(
        xy=xy_arr,
        lonlat=lonlat,
        kinds=kinds,
        target_spacing_m=np.asarray(
            [config.open_spacing_m if kind == "open" else config.island_spacing_m if kind == "island" else config.land_spacing_m for kind in kinds],
            dtype=float,
        ),
        exterior_indices=exterior_indices,
        open_boundary_indices=_ordered_unique(open_indices),
        constraint_chains=constraint_chains,
        domain_polygon_xy=domain_xy,
        open_boundary_xy=open_xy,
        land_boundary_xy=land_xy,
        island_polygons_xy=islands_xy,
        projection=projection,
        hard_anchor_mask=hard_anchor_mask,
        open_boundaries=open_boundaries,
    )


def load_boundary_resolution(manifest_path: str | Path) -> tuple[BoundaryPackage, BoundaryNodes, dict[str, Any]]:
    """Load an explicit adaptive boundary package emitted by fvcom-bdry-arc."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    profile = str(manifest.get("profile", ""))
    if profile not in {"adaptive-coastal-v1", "adaptive-coastal-v2"}:
        raise ValueError("Boundary resolution manifest must use profile adaptive-coastal-v1 or adaptive-coastal-v2")
    gpkg = _resolve_manifest_output(
        manifest_path,
        manifest["outputs"]["boundary_resolution_gpkg"],
    )
    layers = set(gpd.list_layers(gpkg)["name"])
    required = {"resolved_domain_polygon", "resolved_open_boundary", "boundary_nodes"}
    missing = required - layers
    if missing:
        raise ValueError(f"Boundary resolution package is missing layers: {sorted(missing)}")
    domain_gdf = gpd.read_file(gpkg, layer="resolved_domain_polygon").to_crs("EPSG:4326")
    domain = next(geom for geom in domain_gdf.geometry if isinstance(geom, Polygon) and not geom.is_empty)
    open_gdf = gpd.read_file(gpkg, layer="resolved_open_boundary").to_crs("EPSG:4326")
    open_boundary = unary_union([geom for geom in open_gdf.geometry if geom is not None and not geom.is_empty])
    projection = local_utm_projection(tuple(float(v) for v in domain.bounds))
    source_domain_xy = project_geometry(domain, projection)
    nodes_gdf = gpd.read_file(gpkg, layer="boundary_nodes").to_crs("EPSG:4326")
    nodes_gdf = nodes_gdf.sort_values(["chain_id", "chain_position"]).reset_index(drop=True)
    grouped = [
        (str(chain_id), group.reset_index(drop=True))
        for chain_id, group in nodes_gdf.groupby("chain_id", sort=True)
    ]
    if not grouped:
        raise ValueError("Boundary resolution package contains no boundary chains")
    exterior_group = grouped[0][1]
    exterior_lonlat = np.asarray(
        [[float(point.x), float(point.y)] for point in exterior_group.geometry],
        dtype=float,
    )
    exterior_xy = project_points(exterior_lonlat, projection)
    source_island_xy: list[np.ndarray] = []
    source_island_targets: list[np.ndarray] = []
    source_island_ids: list[str] = []
    for chain_id, group in grouped[1:]:
        group_lonlat = np.asarray(
            [[float(point.x), float(point.y)] for point in group.geometry],
            dtype=float,
        )
        source_island_xy.append(project_points(group_lonlat, projection))
        source_island_targets.append(np.asarray(group["target_spacing_m"], dtype=float))
        source_island_ids.append(chain_id)
    protected_exterior_contract = {
        "coordinates_xy": exterior_xy.tolist(),
        "boundary_kind": [str(value) for value in exterior_group["boundary_kind"]],
        "target_spacing_m": [float(value) for value in exterior_group["target_spacing_m"]],
        "is_hard_anchor": [
            bool(value)
            for value in (
                exterior_group["is_hard_anchor"]
                if "is_hard_anchor" in exterior_group
                else np.zeros(len(exterior_group), dtype=bool)
            )
        ],
        "open_boundary_chains": manifest.get("open_boundary_chains") or [],
    }
    compensation = normalize_boundary_topology(
        exterior_xy,
        source_island_xy,
        source_island_targets,
        source_chain_ids=source_island_ids,
        reference_holes_xy=[np.asarray(ring.coords, dtype=float) for ring in source_domain_xy.interiors],
        source_resolution_manifest=manifest_path,
        source_boundary_gpkg=gpkg,
        protected_exterior_contract=protected_exterior_contract,
    )

    reserved_columns = {
        "geometry",
        "node_index_zero_based",
        "chain_id",
        "chain_position",
        "boundary_kind",
        "target_spacing_m",
        "is_hard_anchor",
    }
    metadata_columns = [
        str(column) for column in nodes_gdf.columns if column not in reserved_columns
    ]
    metadata_values: dict[str, list[Any]] = {column: [] for column in metadata_columns}
    xy_parts: list[np.ndarray] = []
    kinds: list[str] = []
    target_parts: list[np.ndarray] = []
    hard_parts: list[np.ndarray] = []
    chains: list[list[int]] = []
    source_node_to_loaded: dict[int, int] = {}
    offset = 0

    def append_source_group(group: gpd.GeoDataFrame, group_xy: np.ndarray) -> list[int]:
        nonlocal offset
        count = len(group_xy)
        chain = list(range(offset, offset + count))
        xy_parts.append(np.asarray(group_xy, dtype=float))
        kinds.extend(str(value) for value in group["boundary_kind"])
        target_parts.append(np.asarray(group["target_spacing_m"], dtype=float))
        hard_parts.append(
            np.asarray(group["is_hard_anchor"], dtype=bool)
            if "is_hard_anchor" in group
            else np.zeros(count, dtype=bool)
        )
        for column in metadata_columns:
            metadata_values[column].extend(group[column].tolist())
        offset += count
        return chain

    exterior = append_source_group(exterior_group, exterior_xy)
    for loaded, source in zip(
        exterior,
        exterior_group.get("node_index_zero_based", np.arange(len(exterior_group))),
    ):
        source_node_to_loaded[int(source)] = int(loaded)
    chains.append(exterior)

    for delivered in compensation.delivered_islands:
        if delivered.unchanged and len(delivered.source_indices) == 1:
            source_group = grouped[int(delivered.source_indices[0]) + 1][1]
            chain = append_source_group(source_group, delivered.xy)
        else:
            count = len(delivered.xy)
            chain = list(range(offset, offset + count))
            xy_parts.append(np.asarray(delivered.xy, dtype=float))
            kinds.extend(["island"] * count)
            target_parts.append(np.asarray(delivered.target_spacing_m, dtype=float))
            hard_parts.append(np.zeros(count, dtype=bool))
            for column in metadata_columns:
                metadata_values[column].extend([None] * count)
            offset += count
        chains.append(chain)

    xy = np.vstack(xy_parts) if xy_parts else np.empty((0, 2), dtype=float)
    lonlat = unproject_points(xy, projection) if len(xy) else np.empty((0, 2), dtype=float)
    targets = np.concatenate(target_parts) if target_parts else np.empty(0, dtype=float)
    hard_anchors = np.concatenate(hard_parts) if hard_parts else np.empty(0, dtype=bool)
    metadata = {
        column: np.asarray(values, dtype=object)
        for column, values in metadata_values.items()
    }
    passage_diagnostics: list[dict[str, Any]] = []
    if "passage_diagnostics" in layers:
        passage_gdf = gpd.read_file(gpkg, layer="passage_diagnostics").to_crs("EPSG:4326")
        for _, row in passage_gdf.iterrows():
            geometry = row.geometry
            if geometry is None or geometry.is_empty:
                continue
            record = {
                str(column): row[column]
                for column in passage_gdf.columns
                if column != "geometry"
            }
            record["geometry_xy"] = project_geometry(geometry, projection)
            passage_diagnostics.append(record)
    open_indices = [idx for idx in exterior if kinds[idx] == "open"]
    open_boundaries = _manifest_open_boundary_chains(
        manifest,
        source_node_to_loaded,
        kinds,
    )
    if not open_boundaries:
        open_boundaries = _open_boundary_chains(exterior, kinds)
    domain_xy = compensation.wet_domain_xy
    islands_xy = [Polygon(value.xy) for value in compensation.delivered_islands]
    domain_lonlat = unproject_geometry(domain_xy, projection)
    islands_lonlat = [unproject_geometry(poly, projection) for poly in islands_xy]
    package = BoundaryPackage(
        domain_polygon_lonlat=domain_lonlat,
        open_boundary_lonlat=open_boundary,
        land_boundary_lonlat=LineString(domain_lonlat.exterior.coords),
        frame_boundary_lonlat=LineString(),
        island_polygons_lonlat=islands_lonlat,
        source_gpkg=str(gpkg),
        projection=projection,
    )
    nodes = BoundaryNodes(
        xy=xy,
        lonlat=lonlat,
        kinds=kinds,
        target_spacing_m=targets,
        exterior_indices=list(exterior),
        open_boundary_indices=_ordered_unique(open_indices),
        constraint_chains=chains,
        domain_polygon_xy=domain_xy,
        open_boundary_xy=project_geometry(open_boundary, projection),
        land_boundary_xy=LineString(domain_xy.exterior.coords),
        island_polygons_xy=islands_xy,
        projection=projection,
        hard_anchor_mask=hard_anchors,
        adaptive_resolution=True,
        source_resolution_manifest=str(manifest_path),
        resolution_profile=profile,
        metadata=metadata,
        passage_diagnostics=passage_diagnostics,
        open_boundaries=open_boundaries,
        topology_compensation=compensation,
    )
    return package, nodes, manifest


def _resolve_manifest_output(
    manifest_path: Path,
    value: str | Path,
) -> Path:
    """Resolve a portable manifest artifact independently of process CWD."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    artifact = Path(value).expanduser()
    candidates: list[Path] = []
    if artifact.is_absolute():
        candidates.append(artifact)
    else:
        candidates.extend(
            [
                manifest_path.parent / artifact,
                Path.cwd() / artifact,
            ]
        )
        candidates.extend(parent / artifact for parent in manifest_path.parents)
    # Archived manifests commonly contain a workspace-relative or stale
    # absolute path even though the immutable artifact travels beside them.
    candidates.append(manifest_path.parent / artifact.name)
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved).casefold()
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return resolved
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Manifest output {value!s} is unavailable; tried: {attempted}"
    )


def boundary_nodes_geojson(nodes: BoundaryNodes) -> dict[str, Any]:
    """Return a GeoJSON FeatureCollection for boundary nodes."""
    features = []
    open_set = set(nodes.open_boundary_indices)
    chain_membership: dict[int, tuple[str, int, bool, str]] = {}
    for chain in normalized_open_boundaries(nodes):
        for position, node_index in enumerate(chain.node_indices):
            chain_membership[int(node_index)] = (
                str(chain.chain_id),
                int(position),
                bool(chain.cyclic),
                str(chain.orientation),
            )
    hard = np.asarray(nodes.hard_anchor_mask if nodes.hard_anchor_mask is not None else np.zeros(len(nodes.lonlat), dtype=bool), dtype=bool)
    for idx, ((lon, lat), kind) in enumerate(zip(nodes.lonlat, nodes.kinds)):
        semantic = {}
        for key, values in (nodes.metadata or {}).items():
            if idx >= len(values):
                continue
            value = values[idx]
            if isinstance(value, np.generic):
                value = value.item()
            if value is None or isinstance(value, (str, int, float, bool)):
                semantic[str(key)] = value
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "node_index_zero_based": int(idx),
                    "boundary_kind": kind,
                    "target_spacing_m": float(nodes.target_spacing_m[idx]),
                    "is_open_boundary": bool(idx in open_set),
                    "is_hard_anchor": bool(hard[idx]),
                    "resolution_profile": str(nodes.resolution_profile),
                    "open_boundary_chain_id": chain_membership.get(idx, (None, None, False, None))[0],
                    "open_boundary_chain_position": chain_membership.get(idx, (None, None, False, None))[1],
                    "open_boundary_cyclic": chain_membership.get(idx, (None, None, False, None))[2],
                    "open_boundary_orientation": chain_membership.get(idx, (None, None, False, None))[3],
                    **semantic,
                },
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def evaluate_boundary_contract_v2(
    nodes: BoundaryNodes,
    *,
    gradation: float = 0.15,
    maximum_l_over_h: float = 1.55,
    maximum_adjacent_target_ratio: float = 1.50,
    expected_open_boundary_count: int | None = None,
) -> dict[str, Any]:
    """Audit the adaptive-v2 boundary contract before triangulation."""
    targets = np.asarray(nodes.target_spacing_m, dtype=float)
    hard = np.asarray(
        nodes.hard_anchor_mask if nodes.hard_anchor_mask is not None else np.zeros(len(nodes.xy), dtype=bool),
        dtype=bool,
    )
    failures: list[str] = []
    if len(targets) != len(nodes.xy) or np.any(~np.isfinite(targets)) or np.any(targets <= 0.0):
        failures.append("boundary_target_spacing_invalid")
    open_boundaries = normalized_open_boundaries(nodes)
    open_values = [node for chain in open_boundaries for node in chain.node_indices]
    expected_count = 1 if expected_open_boundary_count is None else int(expected_open_boundary_count)
    if len(open_boundaries) != expected_count:
        failures.append("open_boundary_chain_count_mismatch")
    chain_reports: list[dict[str, Any]] = []
    for chain in open_boundaries:
        values = [int(value) for value in chain.node_indices]
        chain_failures: list[str] = []
        if len(values) != len(set(values)):
            chain_failures.append("duplicate_open_boundary_node")
        if chain.cyclic:
            if len(values) < 3:
                chain_failures.append("cyclic_open_boundary_too_short")
            if len(values) > 1 and values[0] == values[-1]:
                chain_failures.append("cyclic_open_boundary_repeats_first_node")
        elif len(values) < 2 or not hard[values[0]] or not hard[values[-1]]:
            chain_failures.append("open_boundary_landfall_anchor_missing")
        failures.extend(chain_failures)
        chain_reports.append(
            {
                "chain_id": str(chain.chain_id),
                "kind": str(chain.kind),
                "cyclic": bool(chain.cyclic),
                "orientation": str(chain.orientation),
                "node_count": int(len(values)),
                "passed": bool(not chain_failures),
                "failure_taxonomy": chain_failures,
            }
        )

    maximum_lh = 0.0
    maximum_gradient = 0.0
    maximum_ratio = 1.0
    transition_maximum_ratio = 1.0
    edge_count = 0
    for chain in nodes.constraint_chains:
        values = list(map(int, chain))
        if len(values) < 2:
            continue
        for a, b in zip(values, values[1:] + values[:1]):
            length = float(np.linalg.norm(nodes.xy[a] - nodes.xy[b]))
            if length <= 1.0e-10:
                failures.append("duplicate_boundary_vertex")
                continue
            ha = float(targets[a])
            hb = float(targets[b])
            harmonic = 2.0 / (1.0 / ha + 1.0 / hb)
            maximum_lh = max(maximum_lh, length / max(harmonic, 1.0e-12))
            maximum_gradient = max(maximum_gradient, abs(ha - hb) / length)
            ratio = max(ha, hb) / max(min(ha, hb), 1.0e-12)
            maximum_ratio = max(maximum_ratio, ratio)
            if str(nodes.kinds[a]) != str(nodes.kinds[b]):
                transition_maximum_ratio = max(transition_maximum_ratio, ratio)
            edge_count += 1
    if maximum_lh > float(maximum_l_over_h) + 1.0e-9:
        failures.append("boundary_edge_target_ratio_exceeded")
    if maximum_gradient > float(gradation) + 1.0e-9:
        failures.append("boundary_spacing_gradation_exceeded")
    if transition_maximum_ratio > float(maximum_adjacent_target_ratio) + 1.0e-9:
        failures.append("open_land_junction_spacing_jump")

    metadata = nodes.metadata or {}
    if "is_hard_anchor" in metadata:
        semantic_hard = np.asarray(metadata["is_hard_anchor"], dtype=bool)
        if len(semantic_hard) == len(hard) and np.any(semantic_hard & ~hard):
            failures.append("required_boundary_anchor_lost")
    return {
        "schema_version": "fvcom_boundary_contract_v2",
        "passed": bool(not failures),
        "failure_taxonomy": sorted(set(failures)),
        "edge_count": int(edge_count),
        "hard_anchor_count": int(np.count_nonzero(hard)),
        "open_boundary_node_count": int(len(open_values)),
        "open_boundary_chain_count": int(len(open_boundaries)),
        "expected_open_boundary_chain_count": int(expected_count),
        "open_boundaries": chain_reports,
        "maximum_l_over_h": float(maximum_lh),
        "maximum_spacing_gradient": float(maximum_gradient),
        "maximum_adjacent_target_ratio": float(maximum_ratio),
        "maximum_kind_transition_target_ratio": float(transition_maximum_ratio),
        "thresholds": {
            "maximum_l_over_h": float(maximum_l_over_h),
            "gradation": float(gradation),
            "maximum_adjacent_target_ratio": float(maximum_adjacent_target_ratio),
        },
    }


def _boundary_hard_anchor_mask(kinds: list[str], chains: list[list[int]]) -> np.ndarray:
    hard = np.zeros(len(kinds), dtype=bool)
    if not chains:
        return hard
    exterior = chains[0]
    for position, node in enumerate(exterior):
        previous = exterior[position - 1]
        following = exterior[(position + 1) % len(exterior)]
        hard[int(node)] = bool(kinds[int(previous)] != kinds[int(node)] or kinds[int(following)] != kinds[int(node)])
    return hard


def normalized_open_boundaries(nodes: BoundaryNodes) -> list[OpenBoundaryChain]:
    """Return plural OBCs while preserving the legacy flat single-chain input."""
    if nodes.open_boundaries is not None:
        return list(nodes.open_boundaries)
    values = tuple(_ordered_unique([int(value) for value in nodes.open_boundary_indices]))
    if not values:
        return []
    return [OpenBoundaryChain(chain_id="obc_001", node_indices=values)]


def _open_boundary_chains(exterior: list[int], kinds: list[str]) -> list[OpenBoundaryChain]:
    """Split ordered exterior nodes into distinct open runs, merging wraparound."""
    values = [int(value) for value in exterior]
    if not values:
        return []
    mask = [str(kinds[value]).lower() == "open" for value in values]
    if all(mask):
        return [
            OpenBoundaryChain(
                chain_id="obc_001",
                node_indices=tuple(values),
                kind="cyclic_offshore",
                cyclic=True,
            )
        ]
    runs: list[list[int]] = []
    current: list[int] = []
    for value, is_open in zip(values, mask):
        if is_open:
            current.append(value)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if len(runs) > 1 and mask[0] and mask[-1]:
        runs[0] = runs[-1] + runs[0]
        runs.pop()
    return [
        OpenBoundaryChain(
            chain_id=f"obc_{index:03d}",
            node_indices=tuple(run),
            kind="exchange",
            cyclic=False,
        )
        for index, run in enumerate(runs, start=1)
    ]


def _manifest_open_boundary_chains(
    manifest: dict[str, Any],
    source_node_to_loaded: dict[int, int],
    kinds: list[str],
) -> list[OpenBoundaryChain]:
    """Consume v2's authoritative per-OBC node sequences without concatenation."""
    records = manifest.get("open_boundary_chains") or []
    output: list[OpenBoundaryChain] = []
    for record in sorted(records, key=lambda item: int(item.get("obc_id", 0))):
        obc_id = int(record.get("obc_id", len(output)))
        source_values = [int(value) for value in record.get("node_sequence_zero_based", [])]
        try:
            values = tuple(source_node_to_loaded[value] for value in source_values)
        except KeyError as exc:
            raise ValueError(
                f"Boundary resolution OBC {obc_id} references a missing boundary node: {exc.args[0]}"
            ) from exc
        if not values or any(str(kinds[value]).lower() != "open" for value in values):
            raise ValueError(f"Boundary resolution OBC {obc_id} has an invalid node sequence")
        cyclic = bool(record.get("is_closed", False))
        if cyclic and len(values) > 1 and values[0] == values[-1]:
            values = values[:-1]
        output.append(
            OpenBoundaryChain(
                chain_id=f"obc_{obc_id:03d}",
                node_indices=values,
                kind="cyclic_offshore" if cyclic else "exchange",
                cyclic=cyclic,
                orientation="source",
            )
        )
    return output


def _sample_segment(seg: LineString, spacing: float, include_end: bool) -> list[tuple[float, float]]:
    length = float(seg.length)
    if length == 0.0:
        pt = seg.coords[0]
        return [(float(pt[0]), float(pt[1]))]
    n = max(1, int(np.ceil(length / max(spacing, 1.0))))
    distances = np.linspace(0.0, length, n + 1)
    if not include_end:
        distances = distances[:-1]
    out = []
    for distance in distances:
        point = seg.interpolate(float(distance))
        out.append((float(point.x), float(point.y)))
    return out


def _append_point(points: list[tuple[float, float]], kinds: list[str], xy: tuple[float, float], kind: str) -> int:
    if points and np.hypot(points[-1][0] - xy[0], points[-1][1] - xy[1]) < 1.0e-7:
        if kinds[-1] != "open" and kind == "open":
            kinds[-1] = "open"
        return len(points) - 1
    points.append((float(xy[0]), float(xy[1])))
    kinds.append(kind)
    return len(points) - 1


def _ordered_unique(values: list[int]) -> list[int]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out
