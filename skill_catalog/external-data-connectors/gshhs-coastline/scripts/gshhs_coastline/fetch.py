from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.validation import explain_validity

from .plot import plot_gshhs_map
from .quality import summarize_gdf
from .sources import (
    GSHHG_VERSION,
    GSHHG_ZIP_URL,
    choose_resolution,
    ensure_gshhs_cache,
    expand_centered_topology_bbox,
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


def _polygonal_components(geometry):
    """Return only polygonal components from a geometry-repair result."""
    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    polygons = []
    for part in getattr(geometry, "geoms", (geometry,)):
        if part.geom_type == "Polygon":
            polygons.append(part)
        elif part.geom_type == "MultiPolygon":
            polygons.extend(part.geoms)
    return unary_union(polygons) if polygons else None


def _equal_area_m2(geometry) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    return float(gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(6933).area.iloc[0])


def _validate_polygonal_source(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Repair invalid selected source polygons without modifying the cache."""
    out = gdf.copy()
    repaired_flags: list[bool] = []
    repair_methods: list[str | None] = []
    repaired_geometries = []
    details: list[dict[str, Any]] = []
    for source_index, geometry in out.geometry.items():
        if geometry is None or geometry.is_empty:
            repaired_geometries.append(geometry)
            repaired_flags.append(False)
            repair_methods.append(None)
            continue
        if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                f"GSHHS land source feature {source_index!r} has non-polygonal geometry {geometry.geom_type!r}."
            )
        if geometry.is_valid:
            repaired_geometries.append(geometry)
            repaired_flags.append(False)
            repair_methods.append(None)
            continue

        reason = explain_validity(geometry)
        source_area_m2 = _equal_area_m2(geometry)
        repaired = _polygonal_components(make_valid(geometry))
        if repaired is None or repaired.is_empty or repaired.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(
                f"GSHHS land source feature {source_index!r} could not be repaired to polygonal geometry: {reason}"
            )
        if not repaired.is_valid:
            raise ValueError(
                f"GSHHS land source feature {source_index!r} remains invalid after make_valid: "
                f"{explain_validity(repaired)}"
            )
        repaired_area_m2 = _equal_area_m2(repaired)
        denominator = max(abs(source_area_m2), 1.0)
        details.append(
            {
                "source_index": str(source_index),
                "source_geometry_type": geometry.geom_type,
                "source_validity_reason": reason,
                "repair_method": "shapely.make_valid_polygonal_components",
                "repaired_geometry_type": repaired.geom_type,
                "source_equal_area_m2": source_area_m2,
                "repaired_equal_area_m2": repaired_area_m2,
                "relative_area_change": abs(repaired_area_m2 - source_area_m2) / denominator,
            }
        )
        repaired_geometries.append(repaired)
        repaired_flags.append(True)
        repair_methods.append("shapely.make_valid_polygonal_components")

    out["geometry"] = repaired_geometries
    out["source_geometry_repaired"] = repaired_flags
    out["geometry_repair_method"] = repair_methods
    return out, {
        "policy": "validate_selected_source_then_make_valid_invalid_polygonal_features",
        "cache_modified": False,
        "selected_feature_count": int(len(out)),
        "invalid_source_feature_count": int(len(details)),
        "repaired_source_feature_count": int(len(details)),
        "unrepaired_source_feature_count": 0,
        "max_relative_area_change": max((float(item["relative_area_change"]) for item in details), default=0.0),
        "details": details,
    }


def _read_level_clip(
    path: Path,
    bbox_parts: list[tuple[float, float, float, float]],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[dict[str, Any]]]:
    raw_parts: list[gpd.GeoDataFrame] = []
    land_parts: list[gpd.GeoDataFrame] = []
    validation_records: list[dict[str, Any]] = []
    for part in bbox_parts:
        bbox_poly = box(*part)
        raw = gpd.read_file(path, bbox=part)
        if raw.empty:
            continue
        raw = raw.to_crs(4326)
        raw, validation = _validate_polygonal_source(raw)
        validation["bbox_part_wsen"] = [float(value) for value in part]
        validation_records.append(validation)
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
    return raw_gdf, land_gdf, validation_records


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_gdf(bbox_parts: list[tuple[float, float, float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"part": list(range(1, len(bbox_parts) + 1)), "role": ["source_frame"] * len(bbox_parts)},
        geometry=[box(*part).boundary for part in bbox_parts],
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
    bbox: tuple[float, float, float, float] | None = None,
    *,
    model_bbox: tuple[float, float, float, float] | None = None,
    coverage_factor: float = 3.0,
    lookahead_km: float = 0.0,
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
    if (bbox is None) == (model_bbox is None):
        raise ValueError("Provide exactly one of bbox or model_bbox")
    topology_coverage: dict[str, Any] | None = None
    if model_bbox is not None:
        request_bbox, topology_coverage = expand_centered_topology_bbox(
            tuple(float(x) for x in model_bbox),
            coverage_factor=float(coverage_factor),
            lookahead_km=float(lookahead_km),
        )
    else:
        request_bbox = tuple(float(x) for x in bbox or ())
    bbox_parts, bbox_metadata = split_bbox_antimeridian(request_bbox)
    level_list = parse_levels(levels)
    source, cache_meta = ensure_gshhs_cache(cache_dir, force_download=force_download, quiet=quiet)
    resolution_bbox = tuple(float(x) for x in (model_bbox or request_bbox))
    selected_resolution, resolution_warnings = choose_resolution(resolution, resolution_bbox, source.available_resolutions)

    raw_by_level: list[gpd.GeoDataFrame] = []
    land_by_level: list[gpd.GeoDataFrame] = []
    warnings = list(resolution_warnings)
    missing_levels: list[int] = []
    source_component_hashes: dict[str, str] = {}
    source_geometry_validation: list[dict[str, Any]] = []
    for level in level_list:
        shp = shapefile_path(source.gshhs_dir, selected_resolution, level)
        if not shp.exists():
            missing_levels.append(level)
            warnings.append(f"Missing GSHHS shapefile for resolution {selected_resolution!r} level {level}.")
            continue
        for component in sorted(shp.parent.glob(shp.stem + ".*")):
            if component.is_file():
                key = str(component.relative_to(source.gshhs_dir)).replace("\\", "/")
                source_component_hashes[key] = _sha256(component)
        raw, land, validation_records = _read_level_clip(shp, bbox_parts)
        for record in validation_records:
            record["gshhs_resolution"] = selected_resolution
            record["gshhs_level"] = int(level)
            record["source_path"] = str(shp)
            source_geometry_validation.append(record)
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
    source_footprint = bbox_layer.copy()
    source_frame = _frame_gdf(bbox_parts)
    model_bbox_layer = None
    if model_bbox is not None:
        model_parts, model_bbox_metadata = split_bbox_antimeridian(tuple(float(x) for x in model_bbox))
        model_bbox_layer = _bbox_gdf(model_parts)
        topology_coverage["model_bbox_handling"] = model_bbox_metadata
        topology_coverage["source_bbox_handling"] = bbox_metadata
        topology_coverage["source_component_sha256"] = source_component_hashes
        coastline_frame_overlap_m = 0.0
        if not coastline_gdf.empty:
            coastline_frame_overlap_m = float(
                unary_union(coastline_gdf.to_crs(3857).geometry).intersection(
                    unary_union(source_frame.to_crs(3857).geometry)
                ).length
            )
        topology_coverage["physical_coastline_source_frame_overlap_m"] = coastline_frame_overlap_m
        topology_coverage["source_frame_overlap_tolerance_m"] = 1.0
        topology_coverage["downstream_topology_eligible"] = bool(
            topology_coverage.get("downstream_topology_eligible")
            and coastline_frame_overlap_m <= 1.0
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
        source_frame.to_file(gpkg_path, layer="source_frame", driver="GPKG")
        if model_bbox_layer is not None:
            model_bbox_layer.to_file(gpkg_path, layer="model_bbox", driver="GPKG")
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
            model_bbox_gdf=model_bbox_layer,
            source_frame_gdf=source_frame,
            title=f"{name} GSHHS {selected_resolution} L{','.join(str(x) for x in level_list)}",
            allow_no_basemap=allow_no_basemap,
        )
        outputs["map_png"] = str(map_path)
        warnings.extend(plot_warnings)

    manifest = {
        "schema_version": "gshhs_coastline_fetch_v2" if topology_coverage else "gshhs_coastline_fetch_v1",
        "name": name,
        "run_dir": str(run_path),
        "request": {
            "bbox_wsen": [float(x) for x in request_bbox],
            "model_bbox_wsen": [float(x) for x in model_bbox] if model_bbox is not None else None,
            "coverage_factor": float(coverage_factor) if model_bbox is not None else None,
            "lookahead_km": float(lookahead_km) if model_bbox is not None else None,
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
            "selected_source_component_sha256": source_component_hashes,
            "geometry_validation": {
                "policy": "validate_selected_source_then_make_valid_invalid_polygonal_features",
                "cache_modified": False,
                "selected_feature_count": sum(int(item["selected_feature_count"]) for item in source_geometry_validation),
                "invalid_source_feature_count": sum(int(item["invalid_source_feature_count"]) for item in source_geometry_validation),
                "repaired_source_feature_count": sum(int(item["repaired_source_feature_count"]) for item in source_geometry_validation),
                "unrepaired_source_feature_count": sum(int(item["unrepaired_source_feature_count"]) for item in source_geometry_validation),
                "max_relative_area_change": max(
                    (float(item["max_relative_area_change"]) for item in source_geometry_validation),
                    default=0.0,
                ),
                "parts": source_geometry_validation,
            },
        },
        "bbox_handling": bbox_metadata,
        "topology_coverage": topology_coverage,
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
    if gpkg_path.exists():
        outputs["gpkg_sha256"] = _sha256(gpkg_path)
    manifest["outputs"] = outputs
    write_json(manifest_path, manifest)
    if not quiet:
        print(f"Wrote {manifest_path}")
    return GshhsFetchResult(manifest=manifest, land_gdf=land_gdf, coastline_gdf=coastline_gdf)
