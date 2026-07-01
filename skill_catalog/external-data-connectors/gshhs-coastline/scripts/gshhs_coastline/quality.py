from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd


def summarize_gdf(gdf: gpd.GeoDataFrame, *, expected_crs: str = "EPSG:4326") -> dict[str, Any]:
    warnings: list[str] = []
    if gdf.empty:
        warnings.append("GeoDataFrame is empty.")
        return {
            "feature_count": 0,
            "geometry_types": [],
            "crs": str(gdf.crs),
            "bounds_wsen": None,
            "valid_geometry_count": 0,
            "invalid_geometry_count": 0,
            "warnings": warnings,
        }
    if str(gdf.crs).upper() not in {expected_crs.upper(), "EPSG:4326"}:
        warnings.append(f"Unexpected CRS {gdf.crs}; expected {expected_crs}.")
    return {
        "feature_count": int(len(gdf)),
        "geometry_types": sorted(gdf.geometry.geom_type.dropna().unique().tolist()),
        "crs": str(gdf.crs),
        "bounds_wsen": [float(x) for x in gdf.to_crs(4326).total_bounds],
        "valid_geometry_count": int(gdf.geometry.is_valid.sum()),
        "invalid_geometry_count": int((~gdf.geometry.is_valid).sum()),
        "warnings": warnings,
    }


def read_vector_layers(gpkg: str | Path) -> dict[str, gpd.GeoDataFrame]:
    path = Path(gpkg)
    try:
        import pyogrio

        names = pyogrio.list_layers(path)[:, 0].tolist()
    except Exception:
        names = ["land_polygons"]
    return {name: gpd.read_file(path, layer=name) for name in names}


def summarize_product(gpkg: str | Path) -> dict[str, Any]:
    layers = read_vector_layers(gpkg)
    required = {"land_polygons", "coastline_lines", "request_bbox", "source_footprint"}
    missing = sorted(required.difference(layers))
    layer_summaries = {name: summarize_gdf(gdf) for name, gdf in layers.items()}
    warnings: list[str] = []
    if missing:
        warnings.append(f"Missing required layers: {', '.join(missing)}")
    land = layers.get("land_polygons")
    if land is None or land.empty:
        warnings.append("No land polygons were found in the product.")
    elif any("Polygon" not in geom_type for geom_type in land.geom_type.dropna().unique()):
        warnings.append("land_polygons layer contains non-polygon geometry.")
    coastline = layers.get("coastline_lines")
    if coastline is None or coastline.empty:
        warnings.append("No derived coastline lines were found in the product.")
    return {
        "gpkg": str(gpkg),
        "layers": sorted(layers),
        "missing_required_layers": missing,
        "layer_summaries": layer_summaries,
        "warnings": warnings,
        "status": "pass" if not warnings else "needs_review",
    }
