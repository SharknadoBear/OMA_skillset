from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from .plot import plot_gshhs_map
from .quality import summarize_gdf
from .sources import (
    GSHHG_VERSION,
    GSHHG_ZIP_URL,
    choose_resolution,
    ensure_gshhs_cache,
    parse_levels,
    shapefile_path,
    split_bbox_antimeridian,
    write_json,
)


@dataclass
class GshhsFetchResult:
    manifest: dict[str, Any]
    land_gdf: gpd.GeoDataFrame
    coastline_gdf: gpd.GeoDataFrame


def _empty_gdf(crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[], crs=crs)


def _explode(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    return gdf.explode(index_parts=False, ignore_index=True)


def _read_level_clip(path: Path, bbox_parts: list[tuple[float, float, float, float]]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    raw_parts: list[gpd.GeoDataFrame] = []
    land_parts: list[gpd.GeoDataFrame] = []
    for part in bbox_parts:
        bbox_poly = box(*part)
        raw = gpd.read_file(path, bbox=part)
        if raw.empty:
            continue
        raw = raw.to_crs(4326)
        raw_parts.append(raw)
        clipped = gpd.clip(raw, gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326"), keep_geom_type=True)
        if not clipped.empty:
            land_parts.append(clipped)
    if raw_parts:
        raw_gdf = gpd.GeoDataFrame(pd.concat(raw_parts, ignore_index=True), crs="EPSG:4326")
    else:
        raw_gdf = _empty_gdf()
    if land_parts:
        land_gdf = _explode(gpd.GeoDataFrame(pd.concat(land_parts, ignore_index=True), crs="EPSG:4326"))
    else:
        land_gdf = _empty_gdf()
    return raw_gdf, land_gdf


def _derive_coastline(raw_gdf: gpd.GeoDataFrame, bbox_parts: list[tuple[float, float, float, float]]) -> gpd.GeoDataFrame:
    if raw_gdf.empty:
        return _empty_gdf()
    boundaries = gpd.GeoDataFrame(raw_gdf.drop(columns="geometry", errors="ignore"), geometry=raw_gdf.geometry.boundary, crs="EPSG:4326")
    clipped_parts: list[gpd.GeoDataFrame] = []
    for part in bbox_parts:
        bbox_gdf = gpd.GeoDataFrame(geometry=[box(*part)], crs="EPSG:4326")
        clipped = gpd.clip(boundaries, bbox_gdf, keep_geom_type=True)
        if not clipped.empty:
            clipped_parts.append(clipped)
    if not clipped_parts:
        return _empty_gdf()
    out = gpd.GeoDataFrame(pd.concat(clipped_parts, ignore_index=True), crs="EPSG:4326")
    return _explode(out)


def _bbox_gdf(bbox_parts: list[tuple[float, float, float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"part": list(range(1, len(bbox_parts) + 1))},
        geometry=[box(*part) for part in bbox_parts],
        crs="EPSG:4326",
    )


def _write_shapefiles(run_dir: Path, name: str, land_gdf: gpd.GeoDataFrame, coastline_gdf: gpd.GeoDataFrame) -> dict[str, str]:
    out_dir = run_dir / f"{name}_gshhs_shapefiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    land_path = out_dir / "land_polygons.shp"
    coastline_path = out_dir / "coastline_lines.shp"
    land_gdf.to_file(land_path)
    coastline_gdf.to_file(coastline_path)
    return {"land_shapefile": str(land_path), "coastline_shapefile": str(coastline_path)}


def fetch_gshhs_bbox(
    bbox: tuple[float, float, float, float],
    *,
    run_dir: str | Path,
    name: str,
    resolution: str = "auto",
    levels: str | list[int] = "1",
    cache_dir: str | Path | None = None,
    formats: str = "gpkg,geojson",
    force_download: bool = False,
    make_plot: bool = True,
    allow_no_basemap: bool = False,
    quiet: bool = False,
) -> GshhsFetchResult:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    bbox_parts, bbox_metadata = split_bbox_antimeridian(tuple(float(x) for x in bbox))
    level_list = parse_levels(levels)
    source, cache_meta = ensure_gshhs_cache(cache_dir, force_download=force_download, quiet=quiet)
    selected_resolution, resolution_warnings = choose_resolution(resolution, tuple(float(x) for x in bbox), source.available_resolutions)

    raw_by_level: list[gpd.GeoDataFrame] = []
    land_by_level: list[gpd.GeoDataFrame] = []
    warnings = list(resolution_warnings)
    missing_levels: list[int] = []
    for level in level_list:
        shp = shapefile_path(source.gshhs_dir, selected_resolution, level)
        if not shp.exists():
            missing_levels.append(level)
            warnings.append(f"Missing GSHHS shapefile for resolution {selected_resolution!r} level {level}.")
            continue
        raw, land = _read_level_clip(shp, bbox_parts)
        if not raw.empty:
            raw["gshhs_resolution"] = selected_resolution
            raw["gshhs_level"] = int(level)
            raw["source_path"] = str(shp)
            raw_by_level.append(raw)
        if not land.empty:
            land["gshhs_resolution"] = selected_resolution
            land["gshhs_level"] = int(level)
            land["source_path"] = str(shp)
            land_by_level.append(land)

    raw_gdf = gpd.GeoDataFrame(pd.concat(raw_by_level, ignore_index=True), crs="EPSG:4326") if raw_by_level else _empty_gdf()
    land_gdf = gpd.GeoDataFrame(pd.concat(land_by_level, ignore_index=True), crs="EPSG:4326") if land_by_level else _empty_gdf()
    coastline_gdf = _derive_coastline(raw_gdf, bbox_parts)
    if not coastline_gdf.empty:
        coastline_gdf["gshhs_resolution"] = selected_resolution
        coastline_gdf["source"] = "derived_from_gshhs_polygon_boundary"

    bbox_layer = _bbox_gdf(bbox_parts)
    source_footprint = (
        gpd.GeoDataFrame(geometry=[box(*raw_gdf.total_bounds)], crs="EPSG:4326") if not raw_gdf.empty else bbox_layer.copy()
    )

    outputs: dict[str, Any] = {}
    requested_formats = {part.strip().lower() for part in formats.split(",") if part.strip()}
    gpkg_path = run_path / f"{name}_gshhs_land.gpkg"
    if "gpkg" in requested_formats:
        if gpkg_path.exists():
            gpkg_path.unlink()
        land_gdf.to_file(gpkg_path, layer="land_polygons", driver="GPKG")
        coastline_gdf.to_file(gpkg_path, layer="coastline_lines", driver="GPKG")
        bbox_layer.to_file(gpkg_path, layer="request_bbox", driver="GPKG")
        source_footprint.to_file(gpkg_path, layer="source_footprint", driver="GPKG")
        outputs["gpkg"] = str(gpkg_path)

    if "geojson" in requested_formats:
        land_geojson = run_path / f"{name}_gshhs_land.geojson"
        coastline_geojson = run_path / f"{name}_gshhs_coastline.geojson"
        land_gdf.to_file(land_geojson, driver="GeoJSON")
        coastline_gdf.to_file(coastline_geojson, driver="GeoJSON")
        outputs["land_geojson"] = str(land_geojson)
        outputs["coastline_geojson"] = str(coastline_geojson)

    if "shapefile" in requested_formats or "shp" in requested_formats:
        outputs.update(_write_shapefiles(run_path, name, land_gdf, coastline_gdf))

    plot_warnings: list[str] = []
    if make_plot:
        map_path, plot_warnings = plot_gshhs_map(
            land_gdf,
            coastline_gdf,
            bbox_layer,
            run_path / f"{name}_gshhs_map.png",
            title=f"{name} GSHHS {selected_resolution} L{','.join(str(x) for x in level_list)}",
            allow_no_basemap=allow_no_basemap,
        )
        outputs["map_png"] = str(map_path)
        warnings.extend(plot_warnings)

    manifest = {
        "schema_version": "gshhs_coastline_fetch_v1",
        "name": name,
        "run_dir": str(run_path),
        "request": {
            "bbox_wsen": [float(x) for x in bbox],
            "resolution": resolution,
            "levels": level_list,
            "formats": sorted(requested_formats),
            "cache_dir": str(cache_dir) if cache_dir else None,
        },
        "source": {
            "dataset": "GSHHG/GSHHS",
            "version": GSHHG_VERSION,
            "source_url": GSHHG_ZIP_URL,
            "cache": cache_meta,
            "selected_resolution": selected_resolution,
            "selected_levels": level_list,
            "missing_levels": missing_levels,
        },
        "bbox_handling": bbox_metadata,
        "quality": {
            "land_polygons": summarize_gdf(land_gdf),
            "coastline_lines": summarize_gdf(coastline_gdf),
            "raw_source_features": summarize_gdf(raw_gdf),
        },
        "outputs": outputs,
        "warnings": warnings,
    }
    manifest_path = run_path / f"{name}_gshhs_manifest.json"
    write_json(manifest_path, manifest)
    outputs["manifest_json"] = str(manifest_path)
    manifest["outputs"] = outputs
    write_json(manifest_path, manifest)
    if not quiet:
        print(f"Wrote {manifest_path}")
    return GshhsFetchResult(manifest=manifest, land_gdf=land_gdf, coastline_gdf=coastline_gdf)
