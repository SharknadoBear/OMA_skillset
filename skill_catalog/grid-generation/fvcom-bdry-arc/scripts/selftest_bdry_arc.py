#!/usr/bin/env python3
"""Lightweight selftests for fvcom-bdry-arc."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_bdry_arc import BdryArcConfig, run_bdry_arc  # noqa: E402
from fvcom_bdry_arc.projection import local_utm_projection, project_geometry  # noqa: E402
from fvcom_bdry_arc.workflow import (  # noqa: E402
    _classify_relevant_lines,
    _raster_connectivity_fill,
    extract_gshhs_vector_wet_domain,
    repair_coastline_graph,
)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _synthetic_inputs(root: Path) -> tuple[Path, Path, Path]:
    bpoly = {
        "name": "synthetic_estuary",
        "domain_type": "coastal",
        "envelope_bbox": [0.0, 0.0, 4.0, 4.0],
        "polygon_lonlat": [[4.0, 0.0], [0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]],
        "qa": {"bpoly_quality": {"canonical_region_key": "synthetic"}},
        "target_region_features": {
            "features": [
                {"id": "seed_water", "required": True, "geometry": [1.2, 1.6, 1.6, 2.0]},
            ]
        },
    }
    offshore = {
        "selected_side_index": 3,
        "selected_side_name": "east_or_right",
        "selected_side_start_lonlat": [4.0, 4.0],
        "selected_side_end_lonlat": [4.0, 0.0],
        "offshore_azimuth_deg": 90.0,
    }
    region_path = root / "region_bpoly.json"
    offshore_path = root / "offshore_boundary_artifacts.json"
    _write_json(region_path, bpoly)
    _write_json(offshore_path, offshore)

    coast = gpd.GeoDataFrame(
        [
            {"kind": "mainland", "geometry": LineString([(1.0, 3.3), (0.7, 2.0), (1.0, 0.7)])},
            {"kind": "island", "geometry": LineString([(1.9, 2.1), (2.1, 2.1), (2.1, 1.9), (1.9, 1.9), (1.9, 2.1)])},
            {"kind": "lake", "geometry": LineString([(0.2, 0.3), (0.35, 0.3), (0.35, 0.15), (0.2, 0.15), (0.2, 0.3)])},
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    gpkg = root / "synthetic_cusp_coastline.gpkg"
    coast.to_file(gpkg, layer="coastline", driver="GPKG")
    return region_path, offshore_path, gpkg


def _synthetic_gshhs_inputs(root: Path) -> tuple[Path, Path, Path]:
    bpoly = {
        "name": "synthetic_gshhs_estuary",
        "domain_type": "coastal",
        "envelope_bbox": [0.0, 0.0, 4.0, 4.0],
        "polygon_lonlat": [[4.0, 0.0], [0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]],
        "qa": {"bpoly_quality": {"canonical_region_key": "synthetic"}},
        "target_region_features": {
            "features": [
                {"id": "seed_water", "required": True, "geometry": [2.0, 1.6, 2.3, 2.0]},
            ]
        },
    }
    offshore = {
        "selected_side_index": 3,
        "selected_side_name": "east_or_right",
        "selected_side_start_lonlat": [4.0, 4.0],
        "selected_side_end_lonlat": [4.0, 0.0],
        "offshore_azimuth_deg": 90.0,
    }
    region_path = root / "region_bpoly.json"
    offshore_path = root / "offshore_boundary_artifacts.json"
    _write_json(region_path, bpoly)
    _write_json(offshore_path, offshore)

    land = gpd.GeoDataFrame(
        [
            {"kind": "mainland", "geometry": Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 4.0), (0.0, 4.0)])},
            {"kind": "island", "geometry": Polygon([(2.6, 2.2), (2.9, 2.2), (2.9, 1.9), (2.6, 1.9)])},
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    coast = gpd.GeoDataFrame(
        land.drop(columns="geometry"),
        geometry=land.geometry.boundary,
        crs="EPSG:4326",
    )
    gpkg = root / "synthetic_gshhs_land.gpkg"
    land.to_file(gpkg, layer="land_polygons", driver="GPKG")
    coast.to_file(gpkg, layer="coastline_lines", driver="GPKG")
    return region_path, offshore_path, gpkg


def test_synthetic_package() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        region, offshore, gpkg = _synthetic_inputs(root)
        manifest = run_bdry_arc(
            region,
            offshore,
            root / "run",
            "synthetic",
            coastline_gpkg=gpkg,
            config=BdryArcConfig(mode="test", target_resolution_m=5000.0, coastline_source="generic-gpkg", topology_mode="vector-only"),
        )
        assert Path(manifest["outputs"]["bdry_arc_package_gpkg"]).exists()
        assert Path(manifest["outputs"]["bdry_arc_segments_geojson"]).exists()
        assert Path(manifest["outputs"]["bdry_arc_review_map"]).exists()
        assert Path(manifest["outputs"]["visual_review_dir"], "preliminary_arc_map.png").exists()
        assert Path(manifest["outputs"]["visual_review_dir"], "arc_candidate_contact_sheet.png").exists()
        assert manifest["wet_domain"]["area_m2"] > 0
        assert manifest["offshore_arc"]["candidate_count"] >= 4
        layers = set(gpd.list_layers(manifest["outputs"]["bdry_arc_package_gpkg"])["name"])
        required = {
            "wet_domain",
            "open_boundary_arc",
            "land_boundary_arcs",
            "island_holes",
            "anchor_points",
            "candidate_arcs",
            "coastline_raw",
            "coastline_repaired",
            "topology_diagnostics",
            "forbidden_regions",
        }
        assert required.issubset(layers), sorted(required - layers)


def test_gshhs_vector_package_prefers_coastline_lines() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        region, offshore, gpkg = _synthetic_gshhs_inputs(root)
        manifest = run_bdry_arc(
            region,
            offshore,
            root / "run",
            "synthetic_gshhs",
            coastline_gpkg=gpkg,
            config=BdryArcConfig(mode="test", target_resolution_m=5000.0, coastline_source="gshhs", topology_mode="gshhs-vector"),
        )
        visual_dir = Path(manifest["outputs"]["visual_review_dir"])
        assert manifest["settings"]["topology_mode_used"] == "gshhs-vector"
        assert manifest["inputs"]["coastline_load"]["selected_coastline_layer"] == "coastline_lines"
        assert manifest["inputs"]["coastline_load"]["selected_land_layer"] == "land_polygons"
        assert manifest["wet_domain"]["land_polygon_count"] == 2
        assert manifest["wet_domain"]["coastline_line_count"] >= 2
        assert Path(manifest["outputs"]["bdry_arc_package_gpkg"]).exists()
        assert (visual_dir / "gshhs_polygon_topology_map.png").exists()
        assert (visual_dir / "gshhs_anchor_arc_map.png").exists()
        assert not list(visual_dir.glob("raster_connectivity_iter_*.png"))


def test_gshhs_land_union_fallback_preserves_seed_water() -> None:
    bpoly = box(0.0, 0.0, 10_000.0, 10_000.0)
    land = [box(0.0, 0.0, 4_000.0, 10_000.0)]
    arc = LineString([(4_000.0, 0.0), (4_000.0, 10_000.0)])
    result = extract_gshhs_vector_wet_domain([], land, arc, bpoly, Point(7_000.0, 5_000.0), 250.0)
    wet = result["wet_domain_xy"]
    assert wet.contains(Point(7_000.0, 5_000.0))
    assert not wet.contains(Point(2_000.0, 5_000.0))
    assert result["metadata"]["gshhs_missing_coastline_lines"] is True
    assert result["metadata"]["gshhs_polygonize_fallback_used"] is True


def test_endpoint_repair_is_conservative() -> None:
    projection = local_utm_projection((0.0, 0.0, 1.0, 1.0))
    bpoly = project_geometry(box(0.0, 0.0, 1.0, 1.0), projection)
    line1 = project_geometry(LineString([(0.2, 0.2), (0.4, 0.4)]), projection)
    line2 = project_geometry(LineString([(0.401, 0.401), (0.6, 0.6)]), projection)
    repaired, meta, bridges = repair_coastline_graph([line1, line2], bpoly, 1000.0)
    assert len(repaired) >= 2
    assert meta["bridge_count"] == len(bridges)


def test_raster_fill_respects_connectivity_barrier() -> None:
    bpoly = box(0.0, 0.0, 10_000.0, 10_000.0)
    barrier = LineString([(5_000.0, 0.0), (5_000.0, 10_000.0)])
    arc = LineString([(9_500.0, 0.0), (9_500.0, 10_000.0)])
    result = _raster_connectivity_fill([barrier], arc, bpoly, Point(2_000.0, 5_000.0), 250.0)
    wet = result["wet_domain_xy"]
    assert wet.contains(Point(2_000.0, 5_000.0))
    assert not wet.contains(Point(8_000.0, 5_000.0))


def test_component_classification_drops_disconnected_lines() -> None:
    wet = box(0.0, 0.0, 5_000.0, 10_000.0)
    arc = LineString([(5_000.0, 0.0), (5_000.0, 10_000.0)])
    keep = LineString([(4_800.0, 1_000.0), (4_800.0, 9_000.0)])
    drop = LineString([(9_000.0, 1_000.0), (9_000.0, 9_000.0)])
    classified = _classify_relevant_lines([keep, drop], wet, arc, 250.0, 500.0)
    assert classified["retained_count"] == 1
    assert classified["dropped_count"] == 1


def main() -> int:
    test_synthetic_package()
    test_gshhs_vector_package_prefers_coastline_lines()
    test_gshhs_land_union_fallback_preserves_seed_water()
    test_endpoint_repair_is_conservative()
    test_raster_fill_respects_connectivity_barrier()
    test_component_classification_drops_disconnected_lines()
    print("fvcom-bdry-arc selftests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
