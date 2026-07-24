"""Diagnostic plotting for FVCOM grid generation."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .metrics import element_metric_arrays


def write_mesh_review_map(
    path: str | Path,
    nodes_lonlat: np.ndarray,
    triangles_1based: np.ndarray,
    depths: np.ndarray,
    open_boundary_nodes: np.ndarray,
    domain_polygon: Polygon,
    title: str,
) -> Path:
    """Write a diagnostic map with mesh, depth, domain, and open boundary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tris = np.asarray(triangles_1based, dtype=int) - 1
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if len(tris):
        ax.triplot(nodes_lonlat[:, 0], nodes_lonlat[:, 1], tris, color="#4f5965", linewidth=0.25, alpha=0.7)
        sc = ax.scatter(nodes_lonlat[:, 0], nodes_lonlat[:, 1], c=depths, s=4, cmap="viridis", zorder=3)
        fig.colorbar(sc, ax=ax, label="depth positive down (m)")
    gpd.GeoSeries([domain_polygon], crs="EPSG:4326").boundary.plot(ax=ax, color="#1f77b4", linewidth=1.4)
    if open_boundary_nodes.size:
        open_xy = nodes_lonlat[np.asarray(open_boundary_nodes, dtype=int) - 1]
        ax.plot(open_xy[:, 0], open_xy[:, 1], color="#d62728", linewidth=2.0, label="open boundary NS")
        ax.scatter(open_xy[:, 0], open_xy[:, 1], color="#d62728", s=10, zorder=5)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    if open_boundary_nodes.size:
        ax.legend(loc="best")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_mesh_gpkg(
    path: str | Path,
    nodes_lonlat: np.ndarray,
    triangles_1based: np.ndarray,
    depths: np.ndarray,
) -> Path:
    """Write mesh nodes/elements to a GeoPackage for inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    node_records = [
        {"node_id": idx, "depth_m": float(depth), "geometry": Point(float(lon), float(lat))}
        for idx, ((lon, lat), depth) in enumerate(zip(nodes_lonlat, depths), start=1)
    ]
    gpd.GeoDataFrame(node_records, geometry="geometry", crs="EPSG:4326").to_file(path, layer="nodes", driver="GPKG")
    elem_records = []
    for idx, tri in enumerate(np.asarray(triangles_1based, dtype=int), start=1):
        coords = [tuple(nodes_lonlat[node - 1]) for node in tri]
        elem_records.append({"element_id": idx, "n1": int(tri[0]), "n2": int(tri[1]), "n3": int(tri[2]), "geometry": Polygon(coords)})
    gpd.GeoDataFrame(elem_records, geometry="geometry", crs="EPSG:4326").to_file(path, layer="elements", driver="GPKG")
    return path


def write_mesh_quality_gpkg(
    path: str | Path,
    preclean_nodes_lonlat: np.ndarray,
    preclean_triangles_1based: np.ndarray,
    postclean_nodes_lonlat: np.ndarray,
    postclean_triangles_1based: np.ndarray,
) -> Path:
    """Write final-only or genuine before/after per-element geometric QA layers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    same_mesh = bool(
        np.array_equal(np.asarray(preclean_triangles_1based), np.asarray(postclean_triangles_1based))
        and np.array_equal(np.asarray(preclean_nodes_lonlat), np.asarray(postclean_nodes_lonlat))
    )
    if same_mesh:
        _write_quality_layer(path, "final_elements", postclean_nodes_lonlat, postclean_triangles_1based)
        return path
    _write_quality_layer(path, "preclean_elements", preclean_nodes_lonlat, preclean_triangles_1based)
    _write_quality_layer(path, "postclean_elements", postclean_nodes_lonlat, postclean_triangles_1based)
    return path


def _write_quality_layer(path: Path, layer: str, nodes_lonlat: np.ndarray, triangles_1based: np.ndarray) -> None:
    triangles = np.asarray(triangles_1based, dtype=int) - 1
    # The quality metric needs projected coordinates, but this layer is a
    # visualization artifact. Use a local equirectangular projection only for
    # element-shape scalars while retaining WGS84 geometries in the output.
    lonlat = np.asarray(nodes_lonlat, dtype=float)
    lat0 = float(np.nanmean(lonlat[:, 1])) if len(lonlat) else 0.0
    radius = 6_371_000.0
    xy = np.column_stack(
        [
            np.radians(lonlat[:, 0] - np.nanmean(lonlat[:, 0])) * radius * np.cos(np.radians(lat0)),
            np.radians(lonlat[:, 1] - lat0) * radius,
        ]
    )
    metrics = element_metric_arrays(xy, triangles)
    records = []
    for index, tri in enumerate(triangles):
        coords = [tuple(lonlat[node]) for node in tri]
        records.append(
            {
                "element_id": index + 1,
                "quality_q": float(metrics["quality_q"][index]),
                "min_angle_deg": float(metrics["min_angle_deg"][index]),
                "max_angle_deg": float(metrics["max_angle_deg"][index]),
                "area_m2": float(metrics["area_m2"][index]),
                "geometry": Polygon(coords),
            }
        )
    gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326").to_file(path, layer=layer, driver="GPKG")
