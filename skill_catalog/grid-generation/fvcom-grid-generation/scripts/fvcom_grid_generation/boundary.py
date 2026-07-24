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

from .projection import LocalProjection, local_utm_projection, project_geometry, project_points, unproject_points


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
        hard_anchor_mask=_boundary_hard_anchor_mask(kinds, constraint_chains),
    )


def load_boundary_resolution(manifest_path: str | Path) -> tuple[BoundaryPackage, BoundaryNodes, dict[str, Any]]:
    """Load an explicit adaptive boundary package emitted by fvcom-bdry-arc."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    profile = str(manifest.get("profile", ""))
    if profile not in {"adaptive-coastal-v1", "adaptive-coastal-v2"}:
        raise ValueError("Boundary resolution manifest must use profile adaptive-coastal-v1 or adaptive-coastal-v2")
    gpkg = Path(manifest["outputs"]["boundary_resolution_gpkg"])
    layers = set(gpd.list_layers(gpkg)["name"])
    required = {"resolved_domain_polygon", "resolved_open_boundary", "boundary_nodes"}
    missing = required - layers
    if missing:
        raise ValueError(f"Boundary resolution package is missing layers: {sorted(missing)}")
    domain_gdf = gpd.read_file(gpkg, layer="resolved_domain_polygon").to_crs("EPSG:4326")
    domain = next(geom for geom in domain_gdf.geometry if isinstance(geom, Polygon) and not geom.is_empty)
    open_gdf = gpd.read_file(gpkg, layer="resolved_open_boundary").to_crs("EPSG:4326")
    open_boundary = unary_union([geom for geom in open_gdf.geometry if geom is not None and not geom.is_empty])
    islands = []
    if "resolved_island_polygons" in layers:
        island_gdf = gpd.read_file(gpkg, layer="resolved_island_polygons").to_crs("EPSG:4326")
        islands = [geom for geom in island_gdf.geometry if isinstance(geom, Polygon) and not geom.is_empty]
    projection = local_utm_projection(tuple(float(v) for v in domain.bounds))
    package = BoundaryPackage(
        domain_polygon_lonlat=domain,
        open_boundary_lonlat=open_boundary,
        land_boundary_lonlat=LineString(domain.exterior.coords),
        frame_boundary_lonlat=LineString(),
        island_polygons_lonlat=islands,
        source_gpkg=str(gpkg),
        projection=projection,
    )
    nodes_gdf = gpd.read_file(gpkg, layer="boundary_nodes").to_crs("EPSG:4326")
    nodes_gdf = nodes_gdf.sort_values(["chain_id", "chain_position"]).reset_index(drop=True)
    lonlat = np.asarray([[float(point.x), float(point.y)] for point in nodes_gdf.geometry], dtype=float)
    xy = project_points(lonlat, projection)
    kinds = [str(value) for value in nodes_gdf["boundary_kind"]]
    targets = np.asarray(nodes_gdf["target_spacing_m"], dtype=float)
    hard_anchors = (
        np.asarray(nodes_gdf["is_hard_anchor"], dtype=bool)
        if "is_hard_anchor" in nodes_gdf
        else np.zeros(len(nodes_gdf), dtype=bool)
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
    metadata: dict[str, np.ndarray] = {}
    for column in nodes_gdf.columns:
        if column in reserved_columns:
            continue
        values = nodes_gdf[column].to_numpy(copy=True)
        metadata[str(column)] = values
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
    chains: list[list[int]] = []
    for _, group in nodes_gdf.groupby("chain_id", sort=True):
        chains.append([int(value) for value in group.index])
    exterior = chains[0] if chains else []
    open_indices = [idx for idx in exterior if kinds[idx] == "open"]
    domain_xy = project_geometry(domain, projection).buffer(0)
    islands_xy = [project_geometry(poly, projection).buffer(0) for poly in islands]
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
    )
    return package, nodes, manifest


def boundary_nodes_geojson(nodes: BoundaryNodes) -> dict[str, Any]:
    """Return a GeoJSON FeatureCollection for boundary nodes."""
    features = []
    open_set = set(nodes.open_boundary_indices)
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
    open_values = list(map(int, nodes.open_boundary_indices))
    if len(open_values) < 2 or not hard[open_values[0]] or not hard[open_values[-1]]:
        failures.append("open_boundary_landfall_anchor_missing")

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
