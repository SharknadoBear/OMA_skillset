"""Triangular FVCOM mesh generation with a SciPy/Shapely backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon

from .bathymetry import BathymetryGrid
from .domain import DomainBoundary, build_elliptical_domain
from .mesh_quality import lonlat_to_local_xy, triangle_areas
from .size_field import SizeFieldConfig, build_size_field
from .sms_2dm import write_2dm


@dataclass(frozen=True)
class MeshBuildConfig:
    mesh_name: str = "fvcom_grid"
    min_depth: float = 2.0
    max_candidate_points: int = 6_000
    boundary_points: int = 192
    smoothing_iterations: int = 6
    ellipse_buffer_fraction: float = 0.18
    open_arc_fraction: float = 0.28
    size: SizeFieldConfig = SizeFieldConfig()


@dataclass(frozen=True)
class TriMesh:
    nodes: np.ndarray
    depths: np.ndarray
    triangles: np.ndarray
    open_boundary: np.ndarray
    domain: DomainBoundary

    def write_2dm(self, path: str | Path, mesh_name: str = "fvcom_grid") -> Path:
        return write_2dm(path, self.nodes, self.depths, self.triangles, self.open_boundary, mesh_name)


def build_mesh(
    bathy: BathymetryGrid,
    domain: DomainBoundary | None = None,
    config: MeshBuildConfig | None = None,
    output_2dm: str | Path | None = None,
) -> TriMesh:
    """Build a triangular mesh and optionally write it as SMS 2DM."""
    config = config or MeshBuildConfig()
    domain = domain or build_elliptical_domain(
        bathy,
        buffer_fraction=config.ellipse_buffer_fraction,
        n_boundary=config.boundary_points,
        open_arc_fraction=config.open_arc_fraction,
    )
    size_field = build_size_field(bathy, config.size)

    boundary_nodes = domain.points
    interior_nodes = _interior_candidate_points(bathy, domain, size_field, config)
    nodes = np.vstack([boundary_nodes, interior_nodes])

    nodes = _smooth_interior_nodes(nodes, len(boundary_nodes), domain, config.smoothing_iterations)

    tri = Delaunay(nodes)
    triangles0 = _filter_triangles(nodes, tri.simplices, domain)
    triangles0 = _ensure_ccw(nodes, triangles0)
    triangles = triangles0 + 1

    depths = bathy.sample(nodes[:, 0], nodes[:, 1], fill_value=config.min_depth)
    depths = np.where(np.isfinite(depths), depths, config.min_depth)
    depths = np.maximum(depths, config.min_depth)

    open_boundary = domain.open_indices + 1
    mesh = TriMesh(
        nodes=nodes,
        depths=depths.astype(float),
        triangles=triangles.astype(int),
        open_boundary=open_boundary.astype(int),
        domain=domain,
    )
    if output_2dm is not None:
        mesh.write_2dm(output_2dm, mesh_name=config.mesh_name)
    return mesh


def _interior_candidate_points(
    bathy: BathymetryGrid,
    domain: DomainBoundary,
    size_field,
    config: MeshBuildConfig,
) -> np.ndarray:
    polygon = Polygon(domain.points)
    lon_min, lat_min, lon_max, lat_max = polygon.bounds
    center_lat = 0.5 * (lat_min + lat_max)
    meters_per_lon = 111_320.0 * max(np.cos(np.radians(center_lat)), 0.2)
    meters_per_lat = 110_540.0
    span_m = max((lon_max - lon_min) * meters_per_lon, (lat_max - lat_min) * meters_per_lat)
    nominal = max(config.size.min_size, span_m / 90.0)
    estimated = max(1, int((span_m / nominal) ** 2))
    if estimated > config.max_candidate_points:
        nominal *= np.sqrt(estimated / config.max_candidate_points)

    dlon = nominal / meters_per_lon
    dlat = nominal / meters_per_lat
    lons = np.arange(lon_min + dlon, lon_max, dlon)
    lats = np.arange(lat_min + dlat, lat_max, dlat)
    pts = []
    for j, lat in enumerate(lats):
        offset = 0.5 * dlon if j % 2 else 0.0
        for lon in lons + offset:
            point = Point(float(lon), float(lat))
            if polygon.contains(point):
                boundary_distance_deg = polygon.boundary.distance(point)
                if boundary_distance_deg < 0.75 * min(dlon, dlat):
                    continue
                depth = bathy.sample(np.asarray([lon]), np.asarray([lat]))[0]
                if not np.isfinite(depth) or depth <= 0.0:
                    continue
                target = size_field.sample(np.asarray([lon]), np.asarray([lat]))[0]
                stride = max(1, int(round(target / nominal)))
                ii = int(round((lon - lon_min) / dlon))
                jj = int(round((lat - lat_min) / dlat))
                if (ii + jj) % stride == 0:
                    pts.append((float(lon), float(lat)))
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2), dtype=float)


def _filter_triangles(nodes: np.ndarray, triangles0: np.ndarray, domain: DomainBoundary) -> np.ndarray:
    polygon = Polygon(domain.points)
    kept = []
    for tri in triangles0:
        centroid = nodes[tri].mean(axis=0)
        if polygon.contains(Point(float(centroid[0]), float(centroid[1]))):
            kept.append(tri)
    return np.asarray(kept, dtype=int)


def _ensure_ccw(nodes: np.ndarray, triangles0: np.ndarray) -> np.ndarray:
    xy = lonlat_to_local_xy(nodes)
    areas = triangle_areas(xy, triangles0)
    out = triangles0.copy()
    flip = areas < 0.0
    if np.any(flip):
        out[flip] = out[flip][:, [0, 2, 1]]
    return out


def _smooth_interior_nodes(
    nodes: np.ndarray,
    n_boundary: int,
    domain: DomainBoundary,
    iterations: int,
) -> np.ndarray:
    """Laplacian smooth interior nodes while preserving the boundary nodestring."""
    if iterations <= 0 or len(nodes) <= n_boundary:
        return nodes
    polygon = Polygon(domain.points)
    out = np.asarray(nodes, dtype=float).copy()
    for _ in range(iterations):
        tri = Delaunay(out)
        triangles0 = _filter_triangles(out, tri.simplices, domain)
        neighbors: list[list[int]] = [[] for _ in range(len(out))]
        for tri_nodes in triangles0:
            a, b, c = [int(v) for v in tri_nodes]
            neighbors[a].extend([b, c])
            neighbors[b].extend([a, c])
            neighbors[c].extend([a, b])
        updated = out.copy()
        for idx in range(n_boundary, len(out)):
            unique = sorted(set(neighbors[idx]))
            if len(unique) < 3:
                continue
            candidate = 0.55 * out[idx] + 0.45 * np.mean(out[unique], axis=0)
            if polygon.contains(Point(float(candidate[0]), float(candidate[1]))):
                updated[idx] = candidate
        out = updated
    return out
