"""Merge primary CUSP and fallback coastline segments."""

from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import nearest_points

from .progress import ProgressReporter


def local_utm_epsg(bbox: tuple[float, float, float, float]) -> int:
    """Return a local UTM EPSG code for bbox center."""

    west, south, east, north = bbox
    lon = (west + east) / 2.0
    lat = (south + north) / 2.0
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = max(1, min(zone, 60))
    return (32600 if lat >= 0.0 else 32700) + zone


def _empty_like(columns: list[str], crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    data_columns = [col for col in columns if col != "geometry"]
    return gpd.GeoDataFrame({col: [] for col in data_columns}, geometry=[], crs=crs)


def _length_km(gdf: gpd.GeoDataFrame, epsg: int) -> float:
    if gdf.empty:
        return 0.0
    return float(gdf.to_crs(epsg).length.sum() / 1000.0)


def _nearest_distance_m(geom, target_union) -> float | None:
    if target_union is None or target_union.is_empty or geom is None or geom.is_empty:
        return None
    p1, p2 = nearest_points(geom, target_union)
    return float(p1.distance(p2))


def prepare_primary(cusp: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add FVCOM merge provenance columns to primary CUSP lines."""

    primary = cusp.copy()
    if primary.crs is None:
        primary = primary.set_crs("EPSG:4326")
    primary = primary.to_crs("EPSG:4326")
    primary["fvcom_source"] = "cusp_production"
    primary["source_rank"] = 1
    primary["source_status"] = "primary"
    primary["source_license"] = "NOAA NGS CUSP"
    primary["merge_action"] = "keep_primary"
    primary["distance_to_cusp_m"] = 0.0
    return primary


def merge_with_fallback(
    cusp: gpd.GeoDataFrame,
    fallback: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float],
    *,
    merge_tolerance_m: float = 75.0,
    snap_tolerance_m: float = 100.0,
    min_fallback_fragment_m: float = 100.0,
    reporter: ProgressReporter | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, object]]:
    """Return merged lines, retained fallback fragments, and a merge report."""

    epsg = local_utm_epsg(bbox)
    if reporter:
        reporter.event("merge", "preparing primary and fallback geometries", local_metric_epsg=epsg)
    primary = prepare_primary(cusp)
    fallback = fallback.copy()
    if fallback.crs is None:
        fallback = fallback.set_crs("EPSG:4326")
    fallback = fallback.to_crs("EPSG:4326")

    cusp_m = primary.to_crs(epsg)
    fallback_m = fallback.to_crs(epsg) if not fallback.empty else fallback.to_crs(epsg)
    if reporter:
        reporter.event("merge", "building CUSP union", primary_features=len(cusp_m), fallback_features=len(fallback_m))
    cusp_union = cusp_m.geometry.union_all() if not cusp_m.empty else None
    if reporter:
        reporter.event("merge", "building CUSP duplicate buffer", merge_tolerance_m=merge_tolerance_m)
    cusp_buffer = cusp_union.buffer(merge_tolerance_m) if cusp_union is not None and not cusp_union.is_empty else None

    rows: list[dict[str, object]] = []
    candidate_length = float(fallback_m.length.sum()) if not fallback_m.empty else 0.0
    duplicate_length = 0.0
    dropped_slivers = 0
    unsnapped_endpoints = 0

    total_fallback = int(len(fallback_m))
    if reporter:
        reporter.event("merge", "starting fallback difference loop", fallback_features=total_fallback)
    for processed, (_, row) in enumerate(fallback_m.iterrows(), start=1):
        if reporter:
            reporter.heartbeat(
                "merge",
                "processed fallback merge candidates",
                processed=processed,
                total=total_fallback,
                retained=len(rows),
                dropped_slivers=dropped_slivers,
                duplicate_length_km=round(duplicate_length / 1000.0, 3),
            )
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        duplicate_geom = geom.intersection(cusp_buffer) if cusp_buffer is not None else MultiLineString()
        duplicate_length += float(duplicate_geom.length) if not duplicate_geom.is_empty else 0.0
        keep_geom = geom.difference(cusp_buffer) if cusp_buffer is not None else geom
        if keep_geom.is_empty:
            continue
        temp = gpd.GeoDataFrame([row.drop(labels="geometry").to_dict()], geometry=[keep_geom], crs=epsg)
        temp = temp.explode(index_parts=False, ignore_index=True)
        for _, frag_row in temp.iterrows():
            frag = frag_row.geometry
            if frag is None or frag.is_empty or frag.geom_type not in {"LineString", "MultiLineString"}:
                continue
            frag_length = float(frag.length)
            if frag_length < min_fallback_fragment_m:
                dropped_slivers += 1
                continue
            distance = _nearest_distance_m(frag, cusp_union)
            if distance is not None and distance > snap_tolerance_m:
                unsnapped_endpoints += 1
            out = frag_row.drop(labels="geometry").to_dict()
            out["source_status"] = "fallback_retained"
            out["merge_action"] = "gap_fill"
            out["distance_to_cusp_m"] = distance
            out["retained_length_m"] = frag_length
            out["geometry"] = frag
            rows.append(out)

    if reporter:
        reporter.event(
            "merge",
            "finished fallback difference loop",
            processed=total_fallback,
            retained=len(rows),
            dropped_slivers=dropped_slivers,
            duplicate_length_km=round(duplicate_length / 1000.0, 3),
        )
    retained_m = gpd.GeoDataFrame(rows, crs=epsg) if rows else _empty_like(list(fallback.columns) + ["distance_to_cusp_m", "retained_length_m"], crs=f"EPSG:{epsg}")
    if reporter:
        reporter.event("merge", "projecting retained fallback to EPSG:4326", retained_features=len(retained_m))
    retained = retained_m.to_crs("EPSG:4326")
    common_columns = sorted(set(primary.columns).union(retained.columns))
    for col in common_columns:
        if col not in primary.columns:
            primary[col] = None
        if col not in retained.columns:
            retained[col] = None
    merged = gpd.GeoDataFrame(pd.concat([primary[common_columns], retained[common_columns]], ignore_index=True), crs="EPSG:4326")

    cusp_length_m = float(cusp_m.length.sum()) if not cusp_m.empty else 0.0
    retained_length_m = float(retained_m.length.sum()) if not retained_m.empty else 0.0
    merged_length_m = cusp_length_m + retained_length_m
    fallback_fraction = retained_length_m / merged_length_m if merged_length_m > 0 else 0.0
    warnings: list[str] = []
    if fallback_fraction > 0.25:
        warnings.append("Fallback retained length exceeds 25 percent of merged coastline length; manual review recommended.")
    if unsnapped_endpoints:
        warnings.append(f"{unsnapped_endpoints} fallback fragments remain farther than snap tolerance from CUSP.")

    report = {
        "local_metric_epsg": epsg,
        "merge_tolerance_m": merge_tolerance_m,
        "snap_tolerance_m": snap_tolerance_m,
        "min_fallback_fragment_m": min_fallback_fragment_m,
        "cusp_feature_count": int(len(primary)),
        "fallback_candidate_count": int(len(fallback)),
        "fallback_retained_count": int(len(retained)),
        "cusp_length_km": cusp_length_m / 1000.0,
        "fallback_candidate_length_km": candidate_length / 1000.0,
        "fallback_retained_length_km": retained_length_m / 1000.0,
        "fallback_discarded_duplicate_length_km": duplicate_length / 1000.0,
        "merged_length_km": merged_length_m / 1000.0,
        "fallback_fraction_of_merged_length": fallback_fraction,
        "dropped_sliver_count": dropped_slivers,
        "unsnapped_fragment_count": unsnapped_endpoints,
        "source_attribution": {
            "primary": "NOAA NGS CUSP production regional ZIP",
            "fallback": "OpenStreetMap natural=coastline via Overpass, ODbL 1.0",
        },
        "warnings": warnings,
    }
    if reporter:
        reporter.event(
            "merge",
            "merge report ready",
            retained_features=int(len(retained)),
            fallback_fraction=round(fallback_fraction, 4),
            merged_length_km=round(merged_length_m / 1000.0, 3),
        )
    return merged, retained, report
