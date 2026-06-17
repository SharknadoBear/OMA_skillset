"""Gmsh constrained meshing for coastline-aware FVCOM domains."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString, Point, Polygon

from .bathymetry import load_bathymetry
from .coastline_domain import assert_domain_review_passed
from .mesh_quality import QualityThresholds, evaluate_mesh_quality, lonlat_to_local_xy
from .projection import local_utm_projection, project_geometry, unproject_points
from .size_field import SizeFieldConfig, build_size_field
from .sms_2dm import write_2dm


@dataclass(frozen=True)
class CoastlineMeshResult:
    """Generated mesh paths and diagnostics."""

    output_2dm: Path
    quality_json: Path
    summary_json: Path
    quality: dict


def require_gmsh():
    """Import gmsh or raise a clear dependency message."""
    if importlib.util.find_spec("gmsh") is None:
        raise RuntimeError(
            "The coastline-aware generator requires the Python gmsh package. "
            "Install it with `python -m pip install gmsh` and rerun; no SciPy fallback is used for this workflow."
        )
    import gmsh  # type: ignore

    return gmsh


def generate_coastline_mesh(
    domain_metadata_json: str | Path,
    bathymetry: str | Path,
    output_2dm: str | Path,
    mesh_name: str,
    quality_json: str | Path,
    review_json: str | Path | None = None,
    max_attempts: int = 5,
    include_island_holes: bool = False,
) -> CoastlineMeshResult:
    """Generate a constrained triangular FVCOM mesh using Gmsh."""
    domain_metadata_json = Path(domain_metadata_json)
    metadata = json.loads(domain_metadata_json.read_text(encoding="utf-8"))
    review_json = Path(review_json or metadata["outputs"]["visual_review_json"])
    assert_domain_review_passed(review_json)
    gmsh = require_gmsh()

    gpkg = Path(metadata["outputs"]["domain_gpkg"])
    domain_gdf = gpd.read_file(gpkg, layer="domain").to_crs("EPSG:4326")
    open_gdf = gpd.read_file(gpkg, layer="open_boundary").to_crs("EPSG:4326")
    domain_polygon = domain_gdf.geometry.iloc[0]
    open_boundary = open_gdf.geometry.iloc[0]
    if not isinstance(domain_polygon, Polygon):
        raise ValueError("Domain layer must contain a single Polygon.")
    if not isinstance(open_boundary, LineString):
        raise ValueError("Open-boundary layer must contain a LineString.")
    original_hole_count = len(domain_polygon.interiors)
    if not include_island_holes and original_hole_count:
        domain_polygon = Polygon(domain_polygon.exterior.coords)

    bathy = load_bathymetry(bathymetry)
    target_resolution = float(metadata["target_resolution_m"])
    gradation = float((metadata.get("gradation_report") or {}).get("gradation", 0.15))
    size_config = SizeFieldConfig(
        min_size=target_resolution,
        gradation=gradation,
        gradation_iterations=40,
    )
    size_field = build_size_field(bathy, size_config)

    bbox = tuple(float(v) for v in metadata["bbox_wsen"])
    projection = local_utm_projection(bbox)
    domain_xy = project_geometry(domain_polygon, projection)
    simplify_tol = max(0.05 * target_resolution, 1.0)
    domain_xy = domain_xy.simplify(simplify_tol, preserve_topology=True).buffer(0)
    if not isinstance(domain_xy, Polygon):
        raise ValueError("Simplified domain is not a single Polygon; choose a coarser target or inspect domain geometry.")
    open_xy = project_geometry(open_boundary, projection)
    open_spacing = float(metadata["open_boundary_spacing_m"])
    coastline_spacing = target_resolution
    open_tag_tolerance = max(2.0 * coastline_spacing, 1000.0)

    output_2dm = Path(output_2dm)
    output_2dm.parent.mkdir(parents=True, exist_ok=True)
    quality_json = Path(quality_json)
    summary_json = output_2dm.with_suffix(".summary.json")

    last_quality = {}
    for attempt in range(1, max(1, max_attempts) + 1):
        nodes_lonlat, triangles, open_boundary_nodes = _run_gmsh_once(
            gmsh,
            domain_xy,
            open_xy,
            projection,
            size_field,
            coastline_spacing=coastline_spacing,
            open_spacing=open_spacing,
            open_tag_tolerance=open_tag_tolerance,
            mesh_name=mesh_name,
            attempt=attempt,
        )
        depths = bathy.sample(nodes_lonlat[:, 0], nodes_lonlat[:, 1], fill_value=2.0)
        depths = np.where(np.isfinite(depths), depths, 2.0)
        depths = np.maximum(depths, 2.0)
        write_2dm(output_2dm, nodes_lonlat, depths, triangles, open_boundary_nodes, mesh_name=mesh_name)
        quality = evaluate_mesh_quality(nodes_lonlat, depths, triangles, open_boundary_nodes, QualityThresholds())
        serializable = _jsonable_quality(quality)
        serializable["realized_gradation"] = realized_triangle_gradation(nodes_lonlat, triangles)
        serializable["attempt"] = attempt
        serializable["accepted"] = _passes_quality(serializable)
        last_quality = serializable
        quality_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        if serializable["accepted"]:
            break

    summary = {
        "domain_metadata_json": str(domain_metadata_json),
        "bathymetry": str(bathymetry),
        "output_2dm": str(output_2dm),
        "quality_json": str(quality_json),
        "mesh_name": mesh_name,
        "max_attempts": int(max_attempts),
        "final_attempt": int(last_quality.get("attempt", 0)),
        "accepted": bool(last_quality.get("accepted", False)),
        "target_resolution_m": target_resolution,
        "gradation": gradation,
        "include_island_holes": bool(include_island_holes),
        "domain_hole_count": int(original_hole_count),
        "open_tag_tolerance_m": float(open_tag_tolerance),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return CoastlineMeshResult(output_2dm=output_2dm, quality_json=quality_json, summary_json=summary_json, quality=last_quality)


def _run_gmsh_once(
    gmsh,
    domain_xy: Polygon,
    open_xy: LineString,
    projection,
    size_field,
    coastline_spacing: float,
    open_spacing: float,
    open_tag_tolerance: float,
    mesh_name: str,
    attempt: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gmsh.initialize()
    try:
        gmsh.model.add(f"{mesh_name}_attempt_{attempt}")
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)

        exterior = _polygon_ring_vertices(domain_xy.exterior)
        exterior_curves, exterior_loop, open_curves = _add_ring(gmsh, exterior, open_xy, open_tag_tolerance)
        hole_loops = []
        for ring in domain_xy.interiors:
            pts = _polygon_ring_vertices(ring)
            _, loop, _ = _add_ring(gmsh, pts, open_xy, open_tag_tolerance, force_land=True)
            hole_loops.append(loop)

        surface = gmsh.model.geo.addPlaneSurface([exterior_loop] + hole_loops)
        gmsh.model.geo.synchronize()
        if open_curves:
            gmsh.model.addPhysicalGroup(1, open_curves, tag=1, name="open_boundary")
        land_curves = [curve for curve in exterior_curves if curve not in set(open_curves)]
        if land_curves:
            gmsh.model.addPhysicalGroup(1, land_curves, tag=2, name="land_boundary")
        gmsh.model.addPhysicalGroup(2, [surface], tag=10, name="water_domain")

        interp = _size_interpolator(size_field)

        def size_callback(_dim, _tag, x, y, _z, _lc):
            lon, lat = projection.to_lonlat.transform(x, y)
            value = float(interp([[lat, lon]])[0])
            if not np.isfinite(value):
                value = float(np.nanmax(size_field.size))
            return value

        gmsh.model.mesh.setSizeCallback(size_callback)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")

        tag_to_idx, nodes_xy = _extract_nodes(gmsh)
        triangles = _extract_triangles(gmsh, tag_to_idx)
        nodes_lonlat = unproject_points(nodes_xy, projection)
        open_boundary_nodes = _extract_open_boundary_nodes(gmsh, open_curves, tag_to_idx, nodes_xy, open_xy)
        return nodes_lonlat, triangles, open_boundary_nodes
    finally:
        gmsh.finalize()


def _ring_points(line: LineString, spacing: float) -> np.ndarray:
    length = max(line.length, spacing)
    n = max(4, int(np.ceil(length / max(spacing, 1.0))))
    distances = np.linspace(0.0, length, n, endpoint=False)
    points = [line.interpolate(float(distance % line.length)) for distance in distances]
    return np.asarray([[point.x, point.y] for point in points], dtype=float)


def _polygon_ring_vertices(ring) -> np.ndarray:
    """Return ring vertices without the repeated closing coordinate."""
    pts = np.asarray(ring.coords, dtype=float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        raise ValueError("Polygon ring has fewer than three unique vertices.")
    return pts


def _add_ring(gmsh, points: np.ndarray, open_xy: LineString, open_tag_tolerance: float, force_land: bool = False):
    point_tags = []
    for x, y in points:
        point_tags.append(gmsh.model.geo.addPoint(float(x), float(y), 0.0))
    curves = []
    open_curves = []
    for i, a in enumerate(point_tags):
        b = point_tags[(i + 1) % len(point_tags)]
        curve = gmsh.model.geo.addLine(a, b)
        curves.append(curve)
        if not force_land:
            mid = Point(float(0.5 * (points[i, 0] + points[(i + 1) % len(points), 0])), float(0.5 * (points[i, 1] + points[(i + 1) % len(points), 1])))
            if open_xy.distance(mid) <= max(open_tag_tolerance, 1.0):
                open_curves.append(curve)
    return curves, gmsh.model.geo.addCurveLoop(curves), open_curves


def _size_interpolator(size_field):
    return RegularGridInterpolator(
        (size_field.lat, size_field.lon),
        size_field.size,
        bounds_error=False,
        fill_value=float(np.nanmax(size_field.size)),
    )


def _extract_nodes(gmsh):
    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords, dtype=float).reshape((-1, 3))[:, :2]
    tag_to_idx = {int(tag): i + 1 for i, tag in enumerate(tags)}
    return tag_to_idx, coords


def _extract_triangles(gmsh, tag_to_idx: dict[int, int]) -> np.ndarray:
    element_types, _, element_nodes = gmsh.model.mesh.getElements(2)
    if len(element_types) == 0:
        element_types, _, element_nodes = gmsh.model.mesh.getElements()
    tris = []
    diagnostics = []
    for etype, nodes in zip(element_types, element_nodes):
        props = gmsh.model.mesh.getElementProperties(int(etype))
        name = str(props[0])
        dim = int(props[1])
        order = int(props[2])
        n_nodes = int(props[3])
        diagnostics.append(
            {
                "etype": int(etype),
                "name": name,
                "dim": dim,
                "order": order,
                "nodes_per_element": n_nodes,
                "element_count": int(len(nodes) / max(n_nodes, 1)),
            }
        )
        if "Triangle" not in name or dim != 2:
            continue
        arr = np.asarray(nodes, dtype=int).reshape((-1, n_nodes))
        for tri in arr:
            # For higher-order triangles, the first three nodes are the vertices.
            tris.append([tag_to_idx[int(tag)] for tag in tri[:3]])
    if not tris:
        raise RuntimeError(f"Gmsh generated no triangular surface elements. Element diagnostics: {diagnostics}")
    return np.asarray(tris, dtype=int)


def _extract_open_boundary_nodes(gmsh, open_curves: list[int], tag_to_idx: dict[int, int], nodes_xy: np.ndarray, open_xy: LineString) -> np.ndarray:
    if not open_curves:
        raise RuntimeError("No exterior curves were tagged as open_boundary; adjust offshore line/domain placement.")
    tags = set()
    for curve in open_curves:
        node_tags, _, _ = gmsh.model.mesh.getNodes(1, curve, includeBoundary=True)
        tags.update(int(tag) for tag in node_tags)
    if len(tags) < 2:
        raise RuntimeError("Open boundary has fewer than two mesh nodes.")
    node_indices = np.asarray([tag_to_idx[tag] for tag in tags if tag in tag_to_idx], dtype=int)
    xy = nodes_xy[node_indices - 1]
    order = np.argsort([open_xy.project(Point(float(x), float(y))) for x, y in xy])
    return node_indices[order]


def realized_triangle_gradation(nodes_lonlat: np.ndarray, triangles: np.ndarray) -> dict:
    """Compute adjacent-triangle realized gradation from circumradius and centroid distance."""
    xy = lonlat_to_local_xy(nodes_lonlat)
    tri0 = np.asarray(triangles, dtype=int) - 1
    pts = xy[tri0]
    centers = pts.mean(axis=1)
    radii = _circumradii(pts)
    edge_to_tri: dict[tuple[int, int], list[int]] = {}
    for tid, tri in enumerate(tri0):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_tri.setdefault(tuple(sorted((int(a), int(b)))), []).append(tid)
    values = []
    failed_pairs = []
    for tids in edge_to_tri.values():
        if len(tids) != 2:
            continue
        i, j = tids
        dist = max(float(np.linalg.norm(centers[i] - centers[j])), 1.0)
        value = abs(float(radii[i] - radii[j])) / dist
        values.append(value)
        if value > 0.15:
            failed_pairs.append([int(i + 1), int(j + 1), float(value)])
    return {
        "max": float(np.nanmax(values)) if values else 0.0,
        "failed_pair_count_at_15pct": int(len(failed_pairs)),
        "failed_pairs_at_15pct": failed_pairs[:1000],
    }


def _circumradii(pts: np.ndarray) -> np.ndarray:
    a = np.linalg.norm(pts[:, 1] - pts[:, 2], axis=1)
    b = np.linalg.norm(pts[:, 0] - pts[:, 2], axis=1)
    c = np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1)
    area2 = np.abs(np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]))
    return (a * b * c) / np.maximum(2.0 * area2, 1.0)


def _jsonable_quality(quality: dict) -> dict:
    return {key: (value.tolist() if hasattr(value, "tolist") else value) for key, value in quality.items()}


def _passes_quality(quality: dict) -> bool:
    return (
        quality.get("min_angle", 0.0) >= 30.0
        and quality.get("max_angle", 180.0) <= 130.0
        and quality.get("max_slope", 1.0) <= 0.1
        and quality.get("max_area_change", 1.0) <= 0.5
        and quality.get("max_connecting_elements", 99) <= 8
        and quality.get("open_boundary_max_normal_deviation", 99.0) <= 30.0
        and quality.get("realized_gradation", {}).get("max", 99.0) <= 0.15
    )
