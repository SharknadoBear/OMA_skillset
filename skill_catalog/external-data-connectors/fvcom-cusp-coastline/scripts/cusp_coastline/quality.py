"""Quality summaries for clipped CUSP coastline products."""

from __future__ import annotations

import math
from typing import Iterable

import geopandas as gpd


DATE_FIELDS = ("SRC_DATE", "VER_DATE", "GISDATE")
ATTRIBUTE_FIELDS = (
    "SOURCE_ID",
    "SRC_DATE",
    "HOR_ACC",
    "INFORM",
    "ATTRIBUTE",
    "VER_DATE",
    "SRC_RESOLU",
    "DATA_SOURC",
    "EXT_METH",
    "DAT_SET_CR",
    "SRC_CITA",
    "FIPS_ALPHA",
    "NOAA_Regio",
    "GISDATE",
    "Shape_Leng",
)


def _clean_dates(values: Iterable[object]) -> list[str]:
    dates = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            dates.append(text)
    return sorted(set(dates))


def summarize_quality(gdf: gpd.GeoDataFrame, bbox: tuple[float, float, float, float]) -> dict[str, object]:
    """Return feature, geometry, attribute, and date-range metadata."""

    warnings: list[str] = []
    if gdf.empty:
        warnings.append("No CUSP line features intersected the requested bbox.")
        return {
            "feature_count": 0,
            "geometry_types": [],
            "valid_geometry_count": 0,
            "invalid_geometry_count": 0,
            "bounds_wsen": None,
            "bbox_intersects_output": False,
            "total_length_m_web_mercator": 0.0,
            "attribute_nonnull_counts": {},
            "source_date_ranges": {},
            "data_sources": [],
            "warnings": warnings,
        }

    bounds = tuple(float(x) for x in gdf.total_bounds)
    west, south, east, north = bbox
    bbox_intersects_output = not (bounds[2] < west or bounds[0] > east or bounds[3] < south or bounds[1] > north)
    if not bbox_intersects_output:
        warnings.append("Output bounds do not intersect requested bbox.")

    try:
        total_length = float(gdf.to_crs(3857).length.sum())
        if not math.isfinite(total_length):
            total_length = 0.0
    except Exception:
        total_length = 0.0
        warnings.append("Could not compute Web Mercator total length.")

    attr_counts = {}
    for field in ATTRIBUTE_FIELDS:
        if field in gdf.columns:
            attr_counts[field] = int(gdf[field].notna().sum())

    date_ranges = {}
    for field in DATE_FIELDS:
        if field in gdf.columns:
            dates = _clean_dates(gdf[field].tolist())
            date_ranges[field] = {"min": dates[0] if dates else None, "max": dates[-1] if dates else None}

    data_sources = []
    if "DATA_SOURC" in gdf.columns:
        data_sources = sorted(str(x) for x in gdf["DATA_SOURC"].dropna().unique().tolist())

    return {
        "feature_count": int(len(gdf)),
        "geometry_types": sorted(gdf.geometry.geom_type.dropna().unique().tolist()),
        "valid_geometry_count": int(gdf.geometry.is_valid.sum()),
        "invalid_geometry_count": int((~gdf.geometry.is_valid).sum()),
        "bounds_wsen": bounds,
        "bbox_intersects_output": bbox_intersects_output,
        "total_length_m_web_mercator": total_length,
        "attribute_nonnull_counts": attr_counts,
        "source_date_ranges": date_ranges,
        "data_sources": data_sources,
        "warnings": warnings,
    }
