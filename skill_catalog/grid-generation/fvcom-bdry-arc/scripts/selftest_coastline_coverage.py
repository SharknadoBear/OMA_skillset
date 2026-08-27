#!/usr/bin/env python3
"""Regression tests for centered GSHHS source coverage."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from fvcom_bdry_arc.coastline_coverage import audit_coastline_source_coverage
from fvcom_bdry_arc.projection import local_utm_projection, project_geometry


REGION = Polygon([(-75.8, 38.1), (-74.7, 38.1), (-74.7, 40.4), (-75.8, 40.4)])
PHYSICAL_COAST = [
    LineString([(-75.65, 38.05), (-75.65, 39.1), (-75.45, 40.45)]),
    LineString([(-74.9, 38.05), (-74.9, 39.3), (-75.1, 40.45)]),
]


def _product(root: Path, source_bbox, model_bbox=REGION.bounds, *, include_model_layer: bool = True) -> Path:
    gpkg = root / "synthetic_gshhs_land.gpkg"
    source = box(*source_bbox)
    land = Polygon([(-76.5, 37.0), (-75.65, 37.0), (-75.65, 42.0), (-76.5, 42.0)]).intersection(source)
    gpd.GeoDataFrame(geometry=[land], crs="EPSG:4326").to_file(gpkg, layer="land_polygons", driver="GPKG")
    gpd.GeoDataFrame(geometry=PHYSICAL_COAST, crs="EPSG:4326").to_file(gpkg, layer="coastline_lines", driver="GPKG")
    gpd.GeoDataFrame(geometry=[source], crs="EPSG:4326").to_file(gpkg, layer="source_footprint", driver="GPKG")
    gpd.GeoDataFrame(geometry=[source], crs="EPSG:4326").to_file(gpkg, layer="request_bbox", driver="GPKG")
    gpd.GeoDataFrame(geometry=[source.boundary], crs="EPSG:4326").to_file(gpkg, layer="source_frame", driver="GPKG")
    if include_model_layer:
        gpd.GeoDataFrame(geometry=[box(*model_bbox)], crs="EPSG:4326").to_file(gpkg, layer="model_bbox", driver="GPKG")
    width = source_bbox[2] - source_bbox[0]
    height = source_bbox[3] - source_bbox[1]
    model_width = model_bbox[2] - model_bbox[0]
    model_height = model_bbox[3] - model_bbox[1]
    source_center = ((source_bbox[0] + source_bbox[2]) / 2.0, (source_bbox[1] + source_bbox[3]) / 2.0)
    model_center = ((model_bbox[0] + model_bbox[2]) / 2.0, (model_bbox[1] + model_bbox[3]) / 2.0)
    center_offset = [source_center[0] - model_center[0], source_center[1] - model_center[1]]
    declared_centered = bool(
        abs(center_offset[0]) <= 0.05 * model_width
        and abs(center_offset[1]) <= 0.05 * model_height
    )
    manifest = {
        "schema_version": "gshhs_coastline_fetch_v2",
        "topology_coverage": {
            "schema_version": "gshhs_topology_coverage_v1",
            "coverage_factor_lon": width / model_width,
            "coverage_factor_lat": height / model_height,
            "model_bbox_centrally_contained": declared_centered,
            "source_center_offset_lonlat": center_offset,
            "downstream_topology_eligible": width / model_width >= 2.0 and height / model_height >= 2.0,
        },
    }
    (root / "synthetic_gshhs_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return gpkg


def _audit(gpkg: Path, out: Path):
    projection = local_utm_projection(REGION.bounds)
    physical_xy = unary_union([project_geometry(line, projection) for line in PHYSICAL_COAST])
    anchors = [
        Point(project_geometry(Point(PHYSICAL_COAST[0].coords[0]), projection)),
        Point(project_geometry(Point(PHYSICAL_COAST[1].coords[0]), projection)),
    ]
    return audit_coastline_source_coverage(
        gpkg,
        REGION,
        projection,
        physical_xy,
        anchors_xy=anchors,
        delivered_boundary_xy=project_geometry(REGION.boundary, projection),
        target_resolution_m=1000.0,
        output_dir=out,
        name="synthetic",
    )


def test_delaware_style_incomplete_south_source_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gpkg = _product(root, (-76.05, 38.45, -74.40, 40.55))
        result = _audit(gpkg, root / "audit")
        assert result["downstream_eligible"] is False
        assert "coastline_source_footprint_incomplete" in result["failure_taxonomy"]


def test_centered_threefold_source_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        center_x = (REGION.bounds[0] + REGION.bounds[2]) / 2.0
        center_y = (REGION.bounds[1] + REGION.bounds[3]) / 2.0
        half_x = 1.5 * (REGION.bounds[2] - REGION.bounds[0])
        half_y = 1.5 * (REGION.bounds[3] - REGION.bounds[1])
        gpkg = _product(root, (center_x - half_x, center_y - half_y, center_x + half_x, center_y + half_y))
        result = _audit(gpkg, root / "audit")
        assert result["downstream_eligible"] is True, result
        assert Path(result["maps"]["whole_domain"]["path"]).is_file()
        assert Path(result["maps"]["source_edge_zoom"]["path"]).is_file()


def test_source_frame_landfall_and_dependency_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gpkg = _product(root, REGION.bounds)
        result = _audit(gpkg, root / "audit")
        assert result["downstream_eligible"] is False
        assert "coastline_source_frame_used_as_land_boundary" in result["failure_taxonomy"]


def test_off_center_source_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        center_x = (REGION.bounds[0] + REGION.bounds[2]) / 2.0 + 0.25
        center_y = (REGION.bounds[1] + REGION.bounds[3]) / 2.0
        half_x = 1.5 * (REGION.bounds[2] - REGION.bounds[0])
        half_y = 1.5 * (REGION.bounds[3] - REGION.bounds[1])
        gpkg = _product(root, (center_x - half_x, center_y - half_y, center_x + half_x, center_y + half_y))
        result = _audit(gpkg, root / "audit")
        assert result["downstream_eligible"] is False
        assert "boundary_geometry_outside_coastline_coverage" in result["failure_taxonomy"]


def test_missing_coverage_layer_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        center_x = (REGION.bounds[0] + REGION.bounds[2]) / 2.0
        center_y = (REGION.bounds[1] + REGION.bounds[3]) / 2.0
        half_x = 1.5 * (REGION.bounds[2] - REGION.bounds[0])
        half_y = 1.5 * (REGION.bounds[3] - REGION.bounds[1])
        gpkg = _product(
            root,
            (center_x - half_x, center_y - half_y, center_x + half_x, center_y + half_y),
            include_model_layer=False,
        )
        result = _audit(gpkg, root / "audit")
        assert result["downstream_eligible"] is False
        assert "model_bbox" in result["missing_required_layers"]


def test_valid_external_coverage_can_be_inferred() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        center_x = (REGION.bounds[0] + REGION.bounds[2]) / 2.0
        center_y = (REGION.bounds[1] + REGION.bounds[3]) / 2.0
        half_x = 1.5 * (REGION.bounds[2] - REGION.bounds[0])
        half_y = 1.5 * (REGION.bounds[3] - REGION.bounds[1])
        gpkg = _product(root, (center_x - half_x, center_y - half_y, center_x + half_x, center_y + half_y))
        (root / "synthetic_gshhs_manifest.json").unlink()
        result = _audit(gpkg, root / "audit")
        assert result["downstream_eligible"] is True, result
        assert result["coverage_provenance"] == "geometric_inference"


def main() -> None:
    test_delaware_style_incomplete_south_source_rejected()
    test_centered_threefold_source_passes()
    test_source_frame_landfall_and_dependency_rejected()
    test_off_center_source_rejected()
    test_missing_coverage_layer_rejected()
    test_valid_external_coverage_can_be_inferred()
    print("coastline coverage selftests passed")


if __name__ == "__main__":
    main()
