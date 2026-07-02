"""Boundary-loop ingestion and boundary-node assignment."""

from __future__ import annotations

from dataclasses import dataclass
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
    exterior_indices: list[int]
    open_boundary_indices: list[int]
    constraint_chains: list[list[int]]
    domain_polygon_xy: Polygon
    open_boundary_xy: LineString | MultiLineString
    land_boundary_xy: LineString | MultiLineString
    island_polygons_xy: list[Polygon]
    projection: LocalProjection


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
        exterior_indices=exterior_indices,
        open_boundary_indices=_ordered_unique(open_indices),
        constraint_chains=constraint_chains,
        domain_polygon_xy=domain_xy,
        open_boundary_xy=open_xy,
        land_boundary_xy=land_xy,
        island_polygons_xy=islands_xy,
        projection=projection,
    )


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
