#!/usr/bin/env python3
"""Prepare a closed Lake Ontario adaptive-v2 boundary package from GSHHG.

This is a research-only helper for the six-case Gmsh experiment.  It makes the
GSHHG level semantics explicit:

* level 1 supplies surrounding-land context for evidence and shoreline QA;
* level 2 supplies the complete Lake Ontario wet shell;
* level 3 supplies islands that become holes in the wet domain.

No ocean/open-boundary chain is created.  The complete source level-2 feature
is retained (not clipped to the request bbox), which preserves the eastern
Lake Ontario/St. Lawrence outlet context.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable
import urllib.request
import zipfile

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, unary_union


GSHHG_VERSION = "2.3.7"
GSHHG_ARCHIVE = f"gshhg-shp-{GSHHG_VERSION}.zip"
GSHHG_URL = f"https://ftp.soest.hawaii.edu/gshhg/{GSHHG_ARCHIVE}"
GSHHG_ESTIMATED_BYTES = 142 * 1024 * 1024
DEFAULT_BBOX = (-80.2, 43.0, -76.0, 44.6)
LAKE_REFERENCE = (-77.80, 43.72)
OUTLET_REFERENCE = (-76.45, 44.10)
OUTLET_CONTEXT_BBOX = (-76.80, 43.88, -75.95, 44.35)
PROJECTED_CRS = "EPSG:32618"
REQUIRED_LEVELS = (1, 2, 3)
REQUIRED_EXTENSIONS = (".shp", ".shx", ".dbf", ".prj")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_paths(cache_dir: Path, resolution: str, level: int) -> list[Path]:
    stem = cache_dir / "GSHHS_shp" / resolution / f"GSHHS_{resolution}_L{level}"
    return [stem.with_suffix(extension) for extension in REQUIRED_EXTENSIONS]


def _all_source_paths(cache_dir: Path, resolution: str) -> list[Path]:
    return [
        path
        for level in REQUIRED_LEVELS
        for path in _source_paths(cache_dir, resolution, level)
    ]


def _source_inventory(cache_dir: Path, resolution: str) -> dict[str, Any]:
    levels: dict[str, Any] = {}
    for level in REQUIRED_LEVELS:
        paths = _source_paths(cache_dir, resolution, level)
        present = [path for path in paths if path.exists()]
        levels[str(level)] = {
            "complete": len(present) == len(paths),
            "present": [str(path) for path in present],
            "missing": [str(path) for path in paths if not path.exists()],
        }
    return {
        "resolution": resolution,
        "requested_levels": list(REQUIRED_LEVELS),
        "levels": levels,
        "complete": all(record["complete"] for record in levels.values()),
    }


def _write_request_and_estimate(
    output_dir: Path,
    cache_dir: Path,
    resolution: str,
    bbox: tuple[float, float, float, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = _source_inventory(cache_dir, resolution)
    archive_path = cache_dir / GSHHG_ARCHIVE
    download_required = not bool(inventory["complete"]) and not archive_path.exists()
    estimated_bytes = GSHHG_ESTIMATED_BYTES if download_required else 0
    free_bytes = int(shutil.disk_usage(cache_dir.parent if cache_dir.parent.exists() else output_dir).free)
    request = {
        "schema_version": "gshhs_lake_topology_request_v1",
        "created_utc": _utc_now(),
        "case_id": "lake_ontario",
        "bbox_wsen": list(map(float, bbox)),
        "resolution": resolution,
        "levels": list(REQUIRED_LEVELS),
        "level_semantics": {
            "1": "surrounding land context and shoreline QA",
            "2": "Lake Ontario wet shell",
            "3": "islands in the level-2 lake, retained as wet-domain holes",
        },
        "cache_dir": str(cache_dir),
        "source_url": GSHHG_URL,
        "clip_policy": "select by bbox but retain the complete selected level-2 feature",
        "open_boundary_policy": "closed lake; exactly zero OBC chains",
    }
    estimate = {
        "schema_version": "external_data_estimate_v1",
        "created_utc": _utc_now(),
        "skill_name": "gshhs-coastline",
        "case_id": "lake_ontario",
        "request_path": str(output_dir / "gshhs_request.json"),
        "run_dir": str(output_dir),
        "source_url": GSHHG_URL,
        "cache": inventory,
        "archive_path": str(archive_path),
        "archive_present": archive_path.exists(),
        "estimated_requested_bytes": int(estimated_bytes),
        "estimated_requested_mb": round(estimated_bytes / 1024**2, 3),
        "local_free_bytes": free_bytes,
        "local_free_gb": round(free_bytes / 1024**3, 3),
        "routing_recommendation": "local" if free_bytes > 4 * estimated_bytes else "kestrel",
        "routing_reason": (
            "All requested level files or the official archive are cached."
            if estimated_bytes == 0
            else "Local free space exceeds four times the official archive estimate."
            if free_bytes > 4 * estimated_bytes
            else "Local free space does not exceed four times the official archive estimate."
        ),
        "download_gate": {
            "download_required": bool(download_required),
            "rule": "download locally only when local_free_bytes > 4 * estimated_requested_bytes",
            "passed": bool(estimated_bytes == 0 or free_bytes > 4 * estimated_bytes),
        },
    }
    _write_json(output_dir / "gshhs_request.json", request)
    _write_json(output_dir / "download_estimate.json", estimate)
    return request, estimate


def _download_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    print(f"Downloading {url}", flush=True)
    downloaded = 0
    next_report = 16 * 1024 * 1024
    request = urllib.request.Request(url, headers={"User-Agent": "OMA-Gmsh-research/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0") or 0)
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                suffix = f"/{total / 1024**2:.1f} MiB" if total else " MiB"
                print(f"Downloaded {downloaded / 1024**2:.1f}{suffix}", flush=True)
                next_report += 16 * 1024 * 1024
    os.replace(temporary, destination)
    print(f"Downloaded archive: {destination} ({downloaded / 1024**2:.1f} MiB)", flush=True)


def _extract_required_sources(cache_dir: Path, resolution: str) -> dict[str, Any]:
    archive_path = cache_dir / GSHHG_ARCHIVE
    extracted: list[str] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = set(archive.namelist())
        for level in REQUIRED_LEVELS:
            for extension in REQUIRED_EXTENSIONS:
                member = f"GSHHS_shp/{resolution}/GSHHS_{resolution}_L{level}{extension}"
                if member not in members:
                    raise FileNotFoundError(f"Official archive is missing required member {member}")
                target = cache_dir / member
                if not target.exists():
                    archive.extract(member, cache_dir)
                    extracted.append(str(target))
    inventory = _source_inventory(cache_dir, resolution)
    if not inventory["complete"]:
        raise RuntimeError("Required GSHHG L1/L2/L3 source extraction is incomplete.")
    return {
        "archive": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "extracted": extracted,
        "inventory": inventory,
    }


def _ensure_sources(
    cache_dir: Path,
    resolution: str,
    estimate: dict[str, Any],
    *,
    allow_download: bool,
) -> dict[str, Any]:
    inventory = _source_inventory(cache_dir, resolution)
    archive_path = cache_dir / GSHHG_ARCHIVE
    download_performed = False
    if inventory["complete"]:
        extraction = {
            "archive": str(archive_path) if archive_path.exists() else None,
            "archive_sha256": _sha256(archive_path) if archive_path.exists() else None,
            "extracted": [],
            "inventory": inventory,
        }
    else:
        if not archive_path.exists():
            if not allow_download:
                raise RuntimeError(
                    "GSHHG L2/L3 files are absent. Run with --allow-download after reviewing "
                    "download_estimate.json."
                )
            if not bool(estimate["download_gate"]["passed"]):
                raise RuntimeError("Estimate-first local download gate did not pass.")
            _download_archive(GSHHG_URL, archive_path)
            download_performed = True
        extraction = _extract_required_sources(cache_dir, resolution)
    source_files = _all_source_paths(cache_dir, resolution)
    return {
        "schema_version": "gshhg_source_cache_manifest_v1",
        "created_utc": _utc_now(),
        "dataset": "GSHHG/GSHHS",
        "version": GSHHG_VERSION,
        "source_url": GSHHG_URL,
        "resolution": resolution,
        "levels": list(REQUIRED_LEVELS),
        "download_performed": download_performed,
        "archive": extraction["archive"],
        "archive_sha256": extraction["archive_sha256"],
        "source_files": [
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for path in source_files
        ],
    }


def _polygon_parts(geometry: Any) -> list[Polygon]:
    if geometry is None or geometry.is_empty:
        return []
    valid = make_valid(geometry)
    if isinstance(valid, Polygon):
        return [valid]
    if isinstance(valid, MultiPolygon):
        return [part for part in valid.geoms if not part.is_empty]
    if hasattr(valid, "geoms"):
        out: list[Polygon] = []
        for item in valid.geoms:
            out.extend(_polygon_parts(item))
        return out
    return []


def _read_level(path: Path, bbox_values: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path, bbox=bbox_values)
    if frame.empty:
        raise RuntimeError(f"No GSHHG features intersect Lake Ontario request bbox in {path}")
    return frame.to_crs("EPSG:4326")


def _select_lake(level2: gpd.GeoDataFrame) -> tuple[Polygon, dict[str, Any]]:
    reference = Point(*LAKE_REFERENCE)
    candidates: list[tuple[int, Any, Polygon]] = []
    for row_index, row in level2.iterrows():
        for polygon in _polygon_parts(row.geometry):
            candidates.append((int(row_index), row, polygon))
    containing = [item for item in candidates if item[2].covers(reference)]
    pool = containing or candidates
    if not pool:
        raise RuntimeError("No polygonal GSHHG L2 Lake Ontario candidate was found.")
    projected = gpd.GeoSeries([item[2] for item in pool], crs="EPSG:4326").to_crs(PROJECTED_CRS)
    selected_position = int(np.argmax(projected.area.to_numpy()))
    row_index, row, lake = pool[selected_position]
    source_id = row.get("id", row.get("ID", row_index))
    return lake, {
        "selection_reference_lonlat": list(LAKE_REFERENCE),
        "candidate_polygon_count": len(candidates),
        "reference_containing_count": len(containing),
        "selected_source_row": row_index,
        "selected_source_id": str(source_id),
        "selected_source_bounds": list(map(float, lake.bounds)),
        "selected_source_interior_ring_count": len(lake.interiors),
        "complete_source_feature_retained": True,
    }


def _select_islands(
    level3: gpd.GeoDataFrame,
    lake: Polygon,
) -> tuple[list[Polygon], list[dict[str, Any]]]:
    selected: list[Polygon] = []
    records: list[dict[str, Any]] = []
    for row_index, row in level3.iterrows():
        source_id = row.get("id", row.get("ID", row_index))
        for part_index, polygon in enumerate(_polygon_parts(row.geometry)):
            if not lake.covers(polygon.representative_point()):
                continue
            outside_fraction = 0.0
            if polygon.area > 0.0:
                outside_fraction = float(polygon.difference(lake).area / polygon.area)
            if outside_fraction > 1.0e-8:
                continue
            selected.append(polygon)
            records.append(
                {
                    "source_row": int(row_index),
                    "source_id": str(source_id),
                    "source_part": int(part_index),
                    "bounds": list(map(float, polygon.bounds)),
                }
            )
    if not selected:
        return [], []
    merged = unary_union(selected)
    islands = _polygon_parts(merged)
    islands.sort(key=lambda geometry: (-geometry.area, geometry.centroid.x, geometry.centroid.y))
    return islands, records


def _signed_ring_area(coords: Iterable[tuple[float, float]]) -> float:
    values = list(coords)
    return 0.5 * sum(
        float(a[0]) * float(b[1]) - float(b[0]) * float(a[1])
        for a, b in zip(values, values[1:] + values[:1])
    )


def _build_domain(lake: Polygon, islands: list[Polygon]) -> Polygon:
    wet = lake if not islands else lake.difference(unary_union(islands))
    parts = _polygon_parts(wet)
    if len(parts) != 1:
        raise RuntimeError(f"Island subtraction created {len(parts)} wet components; expected exactly one.")
    domain = orient(parts[0], sign=1.0)
    if not domain.is_valid:
        raise RuntimeError("Constructed Lake Ontario wet domain is invalid.")
    return domain


def _projected_metrics(
    domain: Polygon,
    islands: list[Polygon],
    outlet_point: Point,
    level1_context: gpd.GeoDataFrame,
) -> dict[str, Any]:
    domain_xy = gpd.GeoSeries([domain], crs="EPSG:4326").to_crs(PROJECTED_CRS).iloc[0]
    islands_xy = (
        gpd.GeoSeries(islands, crs="EPSG:4326").to_crs(PROJECTED_CRS)
        if islands
        else gpd.GeoSeries([], crs=PROJECTED_CRS)
    )
    outlet_xy = gpd.GeoSeries([outlet_point], crs="EPSG:4326").to_crs(PROJECTED_CRS).iloc[0]
    context_zone = box(*OUTLET_CONTEXT_BBOX)
    zone_xy = gpd.GeoSeries([context_zone], crs="EPSG:4326").to_crs(PROJECTED_CRS).iloc[0]
    level1_match_m = None
    if not level1_context.empty:
        l1_xy = level1_context.to_crs(PROJECTED_CRS)
        l1_boundary = unary_union(list(l1_xy.geometry)).boundary
        level1_match_m = float(domain_xy.exterior.distance(l1_boundary))
    hole_vertex_count = sum(max(0, len(ring.coords) - 1) for ring in domain.interiors)
    return {
        "projected_crs": PROJECTED_CRS,
        "wet_domain_valid": bool(domain.is_valid),
        "wet_component_count": 1,
        "wet_area_km2": float(domain_xy.area / 1.0e6),
        "exterior_perimeter_km": float(domain_xy.exterior.length / 1000.0),
        "island_hole_count": int(len(domain.interiors)),
        "island_area_km2": float(sum(geometry.area for geometry in islands_xy) / 1.0e6),
        "exterior_vertex_count": int(max(0, len(domain.exterior.coords) - 1)),
        "island_vertex_count": int(hole_vertex_count),
        "total_source_boundary_vertex_count": int(
            max(0, len(domain.exterior.coords) - 1) + hole_vertex_count
        ),
        "exterior_orientation": (
            "counterclockwise"
            if _signed_ring_area(list(domain.exterior.coords)[:-1]) > 0.0
            else "clockwise"
        ),
        "island_orientations": [
            "counterclockwise" if _signed_ring_area(list(ring.coords)[:-1]) > 0.0 else "clockwise"
            for ring in domain.interiors
        ],
        "open_boundary_chain_count": 0,
        "open_boundary_node_count": 0,
        "outlet_context_reference_lonlat": list(OUTLET_REFERENCE),
        "outlet_context_anchor_lonlat": [float(outlet_point.x), float(outlet_point.y)],
        "outlet_context_zone_intersection_km2": float(domain_xy.intersection(zone_xy).area / 1.0e6),
        "outlet_anchor_distance_to_boundary_m": float(outlet_xy.distance(domain_xy.boundary)),
        "level1_to_level2_boundary_minimum_distance_m": level1_match_m,
    }


def _boundary_nodes(
    domain: Polygon,
    outlet_point: Point,
) -> tuple[gpd.GeoDataFrame, float, list[dict[str, Any]]]:
    rings = [domain.exterior, *domain.interiors]
    ring_lonlat = [list(ring.coords)[:-1] for ring in rings]
    ring_xy = [
        list(
            gpd.GeoSeries([LineString(coords + [coords[0]])], crs="EPSG:4326")
            .to_crs(PROJECTED_CRS)
            .iloc[0]
            .coords
        )[:-1]
        for coords in ring_lonlat
    ]
    edge_lengths: list[float] = []
    for coords in ring_xy:
        edge_lengths.extend(
            math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
            for a, b in zip(coords, coords[1:] + coords[:1])
        )
    maximum_edge = max(edge_lengths) if edge_lengths else 0.0
    target_spacing = max(1500.0, 25.0 * math.ceil((maximum_edge / 1.50) / 25.0))

    outlet_ring_point = nearest_points(domain.exterior, outlet_point)[0]
    exterior_points = [Point(lon, lat) for lon, lat in ring_lonlat[0]]
    outlet_index = min(
        range(len(exterior_points)),
        key=lambda index: exterior_points[index].distance(outlet_ring_point),
    )
    rows: list[dict[str, Any]] = []
    chain_reports: list[dict[str, Any]] = []
    node_index = 0
    for chain_id, coords in enumerate(ring_lonlat):
        boundary_kind = "land" if chain_id == 0 else "island"
        for chain_position, (lon, lat) in enumerate(coords):
            rows.append(
                {
                    "node_index_zero_based": node_index,
                    "chain_id": chain_id,
                    "chain_position": chain_position,
                    "boundary_kind": boundary_kind,
                    "target_spacing_m": target_spacing,
                    "is_hard_anchor": bool(chain_id == 0 and chain_position == outlet_index),
                    "is_source_vertex": True,
                    "source_level": 2 if chain_id == 0 else 3,
                    "source_vertex_index": chain_position,
                    "outlet_context_anchor": bool(chain_id == 0 and chain_position == outlet_index),
                    "geometry": Point(float(lon), float(lat)),
                }
            )
            node_index += 1
        chain_reports.append(
            {
                "chain_id": chain_id,
                "kind": "outer" if chain_id == 0 else "island",
                "node_count": len(coords),
                "start_node_index_zero_based": node_index - len(coords),
                "end_node_index_zero_based": node_index - 1,
                "hard_anchor_count": 1 if chain_id == 0 else 0,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326"), target_spacing, chain_reports


def _empty_open_boundary() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "chain_id": pd.Series(dtype="str"),
            "kind": pd.Series(dtype="str"),
            "cyclic": pd.Series(dtype="bool"),
            "orientation": pd.Series(dtype="str"),
        },
        geometry=gpd.GeoSeries([], crs="EPSG:4326"),
    )


def _write_package(
    output_dir: Path,
    domain: Polygon,
    islands: list[Polygon],
    nodes: gpd.GeoDataFrame,
    level1_context: gpd.GeoDataFrame,
    lake_source: Polygon,
    outlet_point: Point,
) -> tuple[Path, Path]:
    gpkg_path = output_dir / "boundary_resolution.gpkg"
    if gpkg_path.exists():
        raise FileExistsError(
            f"{gpkg_path} already exists. Use a fresh preparation directory to keep runs immutable."
        )
    gpd.GeoDataFrame(
        {"name": ["lake_ontario_closed_wet_domain"]},
        geometry=[domain],
        crs="EPSG:4326",
    ).to_file(gpkg_path, layer="resolved_domain_polygon", driver="GPKG")
    _empty_open_boundary().to_file(
        gpkg_path,
        layer="resolved_open_boundary",
        driver="GPKG",
    )
    gpd.GeoDataFrame(
        {"boundary_kind": ["land"]},
        geometry=[LineString(domain.exterior.coords)],
        crs="EPSG:4326",
    ).to_file(gpkg_path, layer="resolved_land_boundary", driver="GPKG")
    if islands:
        gpd.GeoDataFrame(
            {
                "island_id": list(range(1, len(islands) + 1)),
                "source_level": [3] * len(islands),
            },
            geometry=islands,
            crs="EPSG:4326",
        ).to_file(gpkg_path, layer="resolved_island_polygons", driver="GPKG")
    nodes.to_file(gpkg_path, layer="boundary_nodes", driver="GPKG")
    gpd.GeoDataFrame(
        {"source_level": [2], "complete_feature": [True]},
        geometry=[lake_source],
        crs="EPSG:4326",
    ).to_file(gpkg_path, layer="source_l2_lake_polygon", driver="GPKG")
    if not level1_context.empty:
        level1_context.to_file(gpkg_path, layer="source_l1_land_context", driver="GPKG")
    gpd.GeoDataFrame(
        {
            "name": ["st_lawrence_outlet_context"],
            "is_open_boundary": [False],
            "protected_context": [True],
        },
        geometry=[outlet_point],
        crs="EPSG:4326",
    ).to_file(gpkg_path, layer="outlet_context", driver="GPKG")
    nodes_geojson = output_dir / "boundary_resolution_nodes.geojson"
    nodes.to_file(nodes_geojson, driver="GeoJSON")
    return gpkg_path, nodes_geojson


def _plot_evidence(
    output_path: Path,
    level1_context: gpd.GeoDataFrame,
    domain: Polygon,
    islands: list[Polygon],
    outlet_point: Point,
    metrics: dict[str, Any],
) -> None:
    figure, axis = plt.subplots(figsize=(12.5, 7.5), constrained_layout=True)
    if not level1_context.empty:
        level1_context.plot(ax=axis, color="#d9d2c3", edgecolor="#81796b", linewidth=0.35)
    gpd.GeoDataFrame(geometry=[domain], crs="EPSG:4326").plot(
        ax=axis,
        color="#70b7d5",
        edgecolor="#174f72",
        linewidth=0.75,
    )
    if islands:
        gpd.GeoDataFrame(geometry=islands, crs="EPSG:4326").plot(
            ax=axis,
            color="#d9d2c3",
            edgecolor="#5f4b32",
            linewidth=0.55,
        )
    outlet_zone = gpd.GeoDataFrame(
        geometry=[box(*OUTLET_CONTEXT_BBOX)],
        crs="EPSG:4326",
    )
    outlet_zone.boundary.plot(ax=axis, color="#d97706", linewidth=1.1, linestyle="--")
    axis.scatter(
        [outlet_point.x],
        [outlet_point.y],
        s=54,
        marker="*",
        color="#b91c1c",
        edgecolors="white",
        linewidths=0.5,
        zorder=5,
        label="protected outlet context anchor (not an OBC)",
    )
    domain_min_x, domain_min_y, domain_max_x, domain_max_y = domain.bounds
    axis.set_xlim(
        min(DEFAULT_BBOX[0], domain_min_x, OUTLET_CONTEXT_BBOX[0]) - 0.10,
        max(DEFAULT_BBOX[2], domain_max_x, OUTLET_CONTEXT_BBOX[2]) + 0.10,
    )
    axis.set_ylim(
        min(DEFAULT_BBOX[1], domain_min_y, OUTLET_CONTEXT_BBOX[1]) - 0.05,
        max(DEFAULT_BBOX[3], domain_max_y, OUTLET_CONTEXT_BBOX[3]) + 0.05,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(
        "Lake Ontario closed-lake boundary: GSHHG L2 shell with L3 island holes\n"
        f"{metrics['wet_area_km2']:.1f} km² wet area; "
        f"{metrics['island_hole_count']} island holes; 0 OBC chains"
    )
    axis.legend(loc="lower left", frameon=True)
    axis.grid(True, linewidth=0.25, alpha=0.35)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _prepare(
    output_dir: Path,
    cache_dir: Path,
    resolution: str,
    bbox_values: tuple[float, float, float, float],
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    level_paths = {
        level: _source_paths(cache_dir, resolution, level)[0]
        for level in REQUIRED_LEVELS
    }
    level1 = _read_level(level_paths[1], bbox_values)
    level2 = _read_level(level_paths[2], bbox_values)
    level3 = _read_level(level_paths[3], bbox_values)
    lake, selection = _select_lake(level2)
    islands, island_source_records = _select_islands(level3, lake)
    domain = _build_domain(lake, islands)

    request_polygon = box(*bbox_values)
    level1_context = gpd.clip(
        level1,
        gpd.GeoDataFrame(geometry=[request_polygon], crs="EPSG:4326"),
        keep_geom_type=True,
    )
    outlet_point = nearest_points(domain.exterior, Point(*OUTLET_REFERENCE))[0]
    metrics = _projected_metrics(domain, islands, outlet_point, level1_context)
    if metrics["island_hole_count"] != len(islands):
        raise RuntimeError(
            "Constructed wet-domain hole count does not match selected GSHHG L3 island count."
        )
    if metrics["outlet_context_zone_intersection_km2"] <= 0.0:
        raise RuntimeError("Selected complete Lake Ontario feature lost the outlet context zone.")
    nodes, target_spacing, chains = _boundary_nodes(domain, outlet_point)
    gpkg_path, nodes_geojson = _write_package(
        output_dir,
        domain,
        islands,
        nodes,
        level1_context,
        lake,
        outlet_point,
    )
    map_path = output_dir / "lake_ontario_boundary_evidence.png"
    _plot_evidence(map_path, level1_context, domain, islands, outlet_point, metrics)

    maximum_edge_ratio = 0.0
    domain_xy = gpd.GeoSeries([domain], crs="EPSG:4326").to_crs(PROJECTED_CRS).iloc[0]
    for ring in [domain_xy.exterior, *domain_xy.interiors]:
        coords = list(ring.coords)[:-1]
        for a, b in zip(coords, coords[1:] + coords[:1]):
            maximum_edge_ratio = max(
                maximum_edge_ratio,
                math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
                / target_spacing,
            )

    report = {
        "schema_version": "lake_ontario_closed_boundary_report_v1",
        "created_utc": _utc_now(),
        "case_id": "lake_ontario",
        "preparation_status": "pass",
        "experiment_status": "proposed",
        "source": source_manifest,
        "selection": selection,
        "level_feature_counts_in_request": {
            "level_1": int(len(level1)),
            "level_2": int(len(level2)),
            "level_3": int(len(level3)),
            "selected_level_3_island_source_parts": int(len(island_source_records)),
        },
        "topology": metrics,
        "boundary_contract": {
            "profile": "adaptive-coastal-v2",
            "closed_lake": True,
            "expected_open_boundary_count": 0,
            "delivered_open_boundary_count": 0,
            "forcing_compatible": True,
            "source_vertices_preserved": True,
            "source_vertex_count": int(len(nodes)),
            "hard_anchor_count": int(nodes["is_hard_anchor"].sum()),
            "target_spacing_m": float(target_spacing),
            "maximum_source_edge_to_target_ratio": float(maximum_edge_ratio),
            "chain_count": int(len(chains)),
        },
        "outputs": {
            "boundary_resolution_gpkg": str(gpkg_path),
            "boundary_resolution_nodes_geojson": str(nodes_geojson),
            "boundary_resolution_review_map": str(map_path),
        },
        "scope_exclusions": [
            "No ETOPO or other bathymetry was fetched.",
            "No central six-case manifest was modified.",
            "No Gmsh regional mesh was generated.",
        ],
    }
    report_path = output_dir / "lake_ontario_topology_report.json"
    _write_json(report_path, report)

    manifest = {
        "schema_version": "fvcom_boundary_resolution_manifest_v2",
        "name": "lake_ontario_closed_lake_gshhg_h_l2_l3",
        "created_utc": _utc_now(),
        "created_by": "research/gmsh/prepare_lake_ontario_boundary.py",
        "profile": "adaptive-coastal-v2",
        "final_status": "pass",
        "failure_taxonomy": [],
        "advisory_taxonomy": [],
        "inputs": {
            "gshhg_source_cache_manifest": str(output_dir / "gshhg_source_cache_manifest.json"),
            "gshhg_resolution": resolution,
            "gshhg_levels": list(REQUIRED_LEVELS),
            "request_bbox_wsen": list(map(float, bbox_values)),
            "selection_reference_lonlat": list(LAKE_REFERENCE),
            "outlet_reference_lonlat": list(OUTLET_REFERENCE),
        },
        "settings": {
            "domain_type": "closed_lake",
            "expected_open_boundary_count": 0,
            "land_spacing_m": float(target_spacing),
            "island_spacing_m": float(target_spacing),
            "open_spacing_m": None,
            "gradation": 0.0,
            "preserve_every_source_vertex": True,
            "complete_level_2_feature_retained": True,
        },
        "qa": {
            "open_boundary_chain_count": 0,
            "open_boundary_node_count": 0,
            "island_boundary_node_count": int(metrics["island_vertex_count"]),
            "total_boundary_node_count": int(len(nodes)),
            "resolved_island_count": int(metrics["island_hole_count"]),
            "resolved_domain_valid": bool(domain.is_valid),
            "wet_component_count": 1,
            "model_domain_area_m2": float(metrics["wet_area_km2"] * 1.0e6),
            "maximum_edge_to_target_ratio": float(maximum_edge_ratio),
            "hard_anchor_count": int(nodes["is_hard_anchor"].sum()),
            "outlet_context_preserved": True,
        },
        "chains": chains,
        "open_boundaries": [],
        "outputs": {
            "boundary_resolution_gpkg": str(gpkg_path),
            "boundary_resolution_diagnostics_json": str(report_path),
            "boundary_resolution_nodes_geojson": str(nodes_geojson),
            "boundary_resolution_review_map": str(map_path),
            "boundary_resolution_manifest": str(output_dir / "boundary_resolution_manifest.json"),
        },
    }
    manifest_path = output_dir / "boundary_resolution_manifest.json"
    _write_json(manifest_path, manifest)
    report["outputs"]["boundary_resolution_manifest"] = str(manifest_path)
    report["outputs"]["topology_report"] = str(report_path)
    _write_json(report_path, report)
    return {
        "manifest": manifest,
        "report": report,
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cache-dir",
        default="Workspace/Preprocessing/fvcom-gshhs-coastline/cache/gshhg",
    )
    parser.add_argument("--resolution", choices=("h", "f"), default="h")
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX)
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Write request/estimate artifacts and stop before any download.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit the estimate-gated official GSHHG ZIP download when L2/L3 are absent.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    workspace_root = Path(args.workspace_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (workspace_root / output_dir).resolve()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (workspace_root / cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    bbox_values = tuple(float(value) for value in args.bbox)
    _, estimate = _write_request_and_estimate(
        output_dir,
        cache_dir,
        args.resolution,
        bbox_values,
    )
    if args.estimate_only:
        print(json.dumps(estimate, indent=2))
        return 0
    source_manifest = _ensure_sources(
        cache_dir,
        args.resolution,
        estimate,
        allow_download=bool(args.allow_download),
    )
    _write_json(output_dir / "gshhg_source_cache_manifest.json", source_manifest)
    result = _prepare(
        output_dir,
        cache_dir,
        args.resolution,
        bbox_values,
        source_manifest,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "manifest": result["manifest_path"],
                "report": result["report_path"],
                "topology": result["report"]["topology"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
