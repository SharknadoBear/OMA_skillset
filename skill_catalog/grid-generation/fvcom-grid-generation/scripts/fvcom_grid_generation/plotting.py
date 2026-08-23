"""Diagnostic plotting for FVCOM grid generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .metrics import element_metric_arrays
from .quality_policy import load_quality_policy, public_policy_binding, sha256_file
from .sms_2dm import read_2dm


def _quality_q_l3_sigma(quality: dict) -> float:
    value = quality.get("oceanmesh_quality", {}).get("q_l3_sigma")
    if value is None:
        value = quality.get("quality_advisories", {}).get(
            "oceanmesh_quality", {}
        ).get("q_l3_sigma")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("mesh quality does not contain finite q_L3_sigma")
    return value


def _bounded_tile_zoom(bbox: tuple[float, float, float, float]) -> int:
    import mercantile

    west, south, east, north = bbox
    for zoom in range(13, 2, -1):
        if len(list(mercantile.tiles(west, south, east, north, [zoom]))) <= 64:
            return zoom
    return 3


def _draw_review_background(
    ax,
    bbox: tuple[float, float, float, float],
    coastline_path: str | Path | None,
    provider: str,
) -> dict:
    west, south, east, north = bbox
    ax.set_facecolor("#d9edf7")
    failures: list[str] = []
    if str(provider).lower() not in {"offline", "none", "off"}:
        try:
            import contextily.tile as tile
            from contextily.plotting import add_attribution, warp_tiles
            from contextily.tile import bounds2img
            import xyzservices.providers as xyz

            original_get = tile.requests.get

            def get_with_timeout(url, **kwargs):
                kwargs.setdefault("timeout", 2.0)
                return original_get(url, **kwargs)

            tile.requests.get = get_with_timeout
            try:
                zoom = _bounded_tile_zoom(bbox)
                image, extent = bounds2img(
                    west,
                    south,
                    east,
                    north,
                    zoom=zoom,
                    source=xyz.Esri.WorldTopoMap,
                    ll=True,
                    wait=0,
                    max_retries=0,
                    n_connections=2,
                    use_cache=True,
                )
            finally:
                tile.requests.get = original_get
            image, extent = warp_tiles(image, extent, t_crs="EPSG:4326")
            ax.imshow(image, extent=extent, interpolation="bilinear", zorder=0)
            attribution = xyz.Esri.WorldTopoMap.get("attribution")
            if attribution:
                add_attribution(ax, attribution, font_size=6)
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)
            return {
                "status": "ok",
                "source": "Esri World Topographic Map",
                "provider": "topo",
                "zoom": zoom,
                "fallback": False,
            }
        except Exception as exc:  # pragma: no cover - online environment varies
            failures.append(str(exc))

    if coastline_path is not None and Path(coastline_path).is_file():
        try:
            coastline = gpd.read_file(coastline_path)
            if coastline.crs is not None:
                coastline = coastline.to_crs("EPSG:4326")
            clipped = coastline.cx[west:east, south:north]
            if not clipped.empty:
                polygonal = any("Polygon" in value for value in clipped.geom_type)
                if polygonal:
                    clipped.plot(
                        ax=ax,
                        facecolor="#eef2e6",
                        edgecolor="#495057",
                        linewidth=0.45,
                        alpha=0.9,
                        zorder=0.5,
                    )
                else:
                    clipped.plot(
                        ax=ax,
                        color="#495057",
                        linewidth=0.45,
                        alpha=0.9,
                        zorder=0.5,
                    )
                ax.set_xlim(west, east)
                ax.set_ylim(south, north)
                return {
                    "status": "ok",
                    "source": "project-local GSHHS coastline",
                    "provider": "offline",
                    "feature_count": int(len(clipped)),
                    "fallback": True,
                    "online_failures": failures,
                }
        except Exception as exc:
            failures.append(str(exc))

    # A geographic water/land frame derived from the delivered domain is the
    # final offline fallback. It is deliberately not a blank plotting canvas.
    ax.fill(
        [west, east, east, west],
        [south, south, north, north],
        color="#d9edf7",
        zorder=0,
    )
    ax.grid(True, color="white", linewidth=0.8, alpha=0.8, zorder=0.2)
    return {
        "status": "fallback_mesh_geographic_frame",
        "source": "delivered mesh geographic extent",
        "provider": "offline",
        "fallback": True,
        "online_failures": failures,
    }


def write_standard_mesh_review_map(
    path: str | Path,
    manifest_path: str | Path,
    *,
    mesh_path: str | Path,
    quality_path: str | Path,
    boundary_nodes_path: str | Path,
    grid_name: str,
    coastline_path: str | Path | None = None,
    basemap_provider: str = "topo",
) -> dict:
    """Render and hash-bind the standard terminal bathymetric mesh map."""
    path = Path(path)
    manifest_path = Path(manifest_path)
    mesh_path = Path(mesh_path).resolve()
    quality_path = Path(quality_path).resolve()
    boundary_nodes_path = Path(boundary_nodes_path).resolve()
    mesh = read_2dm(mesh_path)
    quality = json.loads(quality_path.read_text(encoding="utf-8-sig"))
    policy = load_quality_policy()
    binding = public_policy_binding(policy)
    if quality.get("quality_policy") != binding:
        raise ValueError("mesh quality is not bound to the installed benchmark-first policy")
    if len(mesh.nodes_lonlat) < 3 or not len(mesh.triangles):
        raise ValueError("standard mesh review map requires a nonempty triangular 2DM")
    if len(mesh.depths) != len(mesh.nodes_lonlat) or not np.all(np.isfinite(mesh.depths)):
        raise ValueError("standard mesh review map requires finite node depths")
    if not boundary_nodes_path.is_file():
        raise ValueError("standard mesh review map requires boundary_nodes.geojson")

    lon = np.asarray(mesh.nodes_lonlat[:, 0], dtype=float)
    lat = np.asarray(mesh.nodes_lonlat[:, 1], dtype=float)
    span_lon = max(float(np.ptp(lon)), 1.0e-6)
    span_lat = max(float(np.ptp(lat)), 1.0e-6)
    bbox = (
        float(np.min(lon) - 0.03 * span_lon),
        float(np.min(lat) - 0.03 * span_lat),
        float(np.max(lon) + 0.03 * span_lon),
        float(np.max(lat) + 0.03 * span_lat),
    )
    q_l3_sigma = _quality_q_l3_sigma(quality)
    title = f"{grid_name} | q_L3σ = {q_l3_sigma:.4f}"
    tris = np.asarray(mesh.triangles, dtype=int) - 1
    face_depth = np.mean(np.asarray(mesh.depths, dtype=float)[tris], axis=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    background = _draw_review_background(ax, bbox, coastline_path, basemap_provider)
    colored = ax.tripcolor(
        lon,
        lat,
        tris,
        facecolors=face_depth,
        cmap="viridis_r",
        shading="flat",
        alpha=0.76,
        zorder=2,
        rasterized=True,
    )
    ax.triplot(
        lon,
        lat,
        tris,
        color="#263238",
        linewidth=0.18,
        alpha=0.48,
        zorder=3,
        rasterized=True,
    )
    fig.colorbar(colored, ax=ax, label="Bathymetry, positive down (m)")
    for chain_index, chain in enumerate(mesh.open_boundary_chains):
        indices = np.asarray(chain, dtype=int) - 1
        label = "Open boundary arc" if chain_index == 0 else None
        ax.plot(
            lon[indices],
            lat[indices],
            color="#d7191c",
            linewidth=3.0,
            label=label,
            zorder=6,
        )
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if mesh.open_boundary_chains:
        ax.legend(loc="best")
    temporary = path.with_name(path.name + ".tmp.png")
    fig.savefig(temporary, dpi=180, metadata={"Title": title})
    plt.close(fig)
    os.replace(temporary, path)

    payload = {
        "schema_version": "fvcom_mesh_review_map_v1",
        "title": title,
        "grid_name": str(grid_name),
        "q_l3_sigma": q_l3_sigma,
        "mesh": {"sha256": sha256_file(mesh_path)},
        "quality": {"sha256": sha256_file(quality_path)},
        "boundary_nodes": {"sha256": sha256_file(boundary_nodes_path)},
        "coastline": (
            {"sha256": sha256_file(coastline_path)}
            if coastline_path is not None and Path(coastline_path).is_file()
            else None
        ),
        "quality_policy": binding,
        "basemap": background,
        "open_boundary_chain_count": int(len(mesh.open_boundary_chains)),
        "image": {"sha256": sha256_file(path)},
    }
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return payload


def validate_standard_mesh_review_map(
    map_path: str | Path,
    manifest_path: str | Path,
    *,
    mesh_path: str | Path,
    quality_path: str | Path,
    boundary_nodes_path: str | Path,
) -> dict:
    failures: list[str] = []
    map_path = Path(map_path)
    manifest_path = Path(manifest_path)
    if not map_path.is_file() or not manifest_path.is_file():
        return {"passed": False, "failure_taxonomy": ["mesh_review_map_missing"]}
    source_paths = {
        "mesh": Path(mesh_path),
        "quality": Path(quality_path),
        "boundary_nodes": Path(boundary_nodes_path),
    }
    missing_sources = [
        f"mesh_review_map_{key}_source_missing"
        for key, source in source_paths.items()
        if not source.is_file()
    ]
    if missing_sources:
        return {"passed": False, "failure_taxonomy": missing_sources}
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {"passed": False, "failure_taxonomy": ["mesh_review_map_manifest_invalid"]}
    if document.get("schema_version") != "fvcom_mesh_review_map_v1":
        failures.append("mesh_review_map_schema_invalid")
    expected = {
        "image": sha256_file(map_path),
        "mesh": sha256_file(source_paths["mesh"]),
        "quality": sha256_file(source_paths["quality"]),
        "boundary_nodes": sha256_file(source_paths["boundary_nodes"]),
    }
    for key, digest in expected.items():
        if document.get(key, {}).get("sha256") != digest:
            failures.append(f"mesh_review_map_{key}_hash_mismatch")
    if document.get("quality_policy") != public_policy_binding():
        failures.append("mesh_review_map_policy_stale")
    if not str(document.get("title", "")).endswith(
        f"q_L3σ = {float(document.get('q_l3_sigma', float('nan'))):.4f}"
    ):
        failures.append("mesh_review_map_title_invalid")
    return {
        "passed": not failures,
        "failure_taxonomy": sorted(set(failures)),
        "manifest": document,
    }


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
