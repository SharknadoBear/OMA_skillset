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
    adaptive_resolution: bool = False
    source_resolution_manifest: str | None = None


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
    )


def load_boundary_resolution(manifest_path: str | Path) -> tuple[BoundaryPackage, BoundaryNodes, dict[str, Any]]:
    """Load an explicit adaptive boundary package emitted by fvcom-bdry-arc."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("profile") != "adaptive-coastal-v1":
        raise ValueError("Boundary resolution manifest must use profile adaptive-coastal-v1")
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
        adaptive_resolution=True,
        source_resolution_manifest=str(manifest_path),
    )
    return package, nodes, manifest


def boundary_nodes_geojson(nodes: BoundaryNodes) -> dict[str, Any]:
    """Return a GeoJSON FeatureCollection for boundary nodes."""
    features = []
    open_set = set(nodes.open_boundary_indices)
    for idx, ((lon, lat), kind) in enumerate(zip(nodes.lonlat, nodes.kinds)):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "node_index_zero_based": int(idx),
                    "boundary_kind": kind,
                    "target_spacing_m": float(nodes.target_spacing_m[idx]),
                    "is_open_boundary": bool(idx in open_set),
                },
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


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
