"""Lightweight tests for the coastline-aware FVCOM workflow."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry import LineString, MultiLineString, Point

from fvcom_grid_generation.bathymetry import BathymetryGrid
from fvcom_grid_generation.coastline_domain import (
    DomainPrepareConfig,
    assert_domain_review_passed,
    open_boundary_from_bbox,
)
from fvcom_grid_generation.open_boundary_designer import design_open_boundary
from fvcom_grid_generation.open_boundary_designer import _line_intersections_on_arc
from fvcom_grid_generation.open_boundary_designer import _bbox_touch_info
from fvcom_grid_generation.open_boundary_designer import _anchor_outer_envelope_report
from fvcom_grid_generation.open_boundary_designer import _build_outer_envelope_index
from fvcom_grid_generation.open_boundary_designer import _classify_endpoint_anchors
from fvcom_grid_generation.open_boundary_designer import _directional_bezier_variant
from fvcom_grid_generation.open_boundary_designer import _endpoint_state_has_valid_pair
from fvcom_grid_generation.projection import local_utm_projection, project_geometry
from fvcom_grid_generation.size_field import SizeFieldConfig, apply_gradation_limit_with_report


def main() -> None:
    test_gradation_limiter()
    test_open_boundary_spacing_policy()
    test_non_reference_open_boundary_design()
    test_directional_anchor_bezier_tangents()
    test_anchor_bbox_touch_variants()
    test_anchor_intersection_counting()
    test_anchor_full_coastline_keeps_short_segments()
    test_outer_envelope_classifier()
    test_endpoint_anchor_classification()
    test_endpoint_anchor_same_side_blocks_convergence()
    test_anchor_iterate_converges_on_synthetic_coast()
    test_anchor_iterate_true_max_iteration_stop()
    test_review_gate()
    print("coastline workflow selftest ok")


def test_gradation_limiter() -> None:
    lon = np.linspace(-75.0, -74.9, 8)
    lat = np.linspace(39.0, 39.1, 8)
    size = np.full((8, 8), 10_000.0)
    size[3, 3] = 100.0
    config = SizeFieldConfig(min_size=100.0, max_size=10_000.0, gradation=0.15, gradation_iterations=80)
    limited, report = apply_gradation_limit_with_report(lon, lat, size, config)
    assert limited[3, 3] == 100.0, "gradation limiter must not coarsen finest cell"
    assert np.all(limited <= size + 1.0e-9), "gradation limiter must only reduce sizes"
    assert report["max_neighbor_gradation"] <= 0.150001, report


def test_open_boundary_spacing_policy() -> None:
    bbox = (-75.8, 37.6, -73.5, 40.2)
    target_resolution = 100.0
    config = DomainPrepareConfig(target_resolution_m=target_resolution)
    factor = float(np.clip(config.open_boundary_spacing_factor, config.open_boundary_spacing_min_factor, config.open_boundary_spacing_max_factor))
    spacing = factor * target_resolution
    line = open_boundary_from_bbox(bbox, "east", spacing)
    projection = local_utm_projection(bbox)
    length = project_geometry(line, projection).length
    nominal_segments = max(1, len(line.coords) - 1)
    realized = length / nominal_segments
    assert 50.0 * target_resolution <= realized <= 110.0 * target_resolution, realized


def test_review_gate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "review.json"
        path.write_text(json.dumps({"decision": "needs_review"}), encoding="utf-8")
        try:
            assert_domain_review_passed(path)
        except PermissionError:
            pass
        else:  # pragma: no cover
            raise AssertionError("review gate did not block needs_review manifest")
        path.write_text(json.dumps({"decision": "pass"}), encoding="utf-8")
        assert assert_domain_review_passed(path)["decision"] == "pass"


def test_non_reference_open_boundary_design() -> None:
    lon = np.linspace(-75.0, -74.7, 40)
    lat = np.linspace(38.8, 39.1, 40)
    lon2, _lat2 = np.meshgrid(lon, lat)
    depth = 5.0 + 40.0 * (lon2 - lon.min()) / (lon.max() - lon.min())
    bathy = BathymetryGrid(lon=lon, lat=lat, depth=depth, source="synthetic")
    bbox_wsen = bathy.bbox
    wet_polygon = box(*bbox_wsen)
    coastline = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    result = design_open_boundary(
        wet_polygon,
        bathy,
        bbox_wsen,
        coastline,
        target_resolution_m=500.0,
        open_spacing_m=37_500.0,
        offshore_side=None,
        mode="auto",
        max_rounds=1,
    )
    assert result.metadata["offshore_side"] == "east", result.metadata
    assert result.metadata["candidate_count"] >= 6, result.metadata
    assert not result.candidates.empty
    selected = result.candidates[result.candidates["selected"].astype(bool)]
    assert len(selected) == 1
    assert selected.iloc[0]["wet_fraction"] > 0.9
    assert result.domain_polygon.is_valid
    assert result.open_boundary.length > 0.0


def test_anchor_intersection_counting() -> None:
    arc = LineString([(0.0, 0.0), (10.0, 0.0)])
    assert len(_line_intersections_on_arc(arc, LineString([(3.0, -1.0), (3.0, 1.0)]), 0.1)) == 1
    coast = LineString([(3.0, -1.0), (3.0, 1.0), (7.0, 1.0), (7.0, -1.0)])
    assert len(_line_intersections_on_arc(arc, coast, 0.1)) == 2
    branchy = LineString([(3.0, -1.0), (3.0, 1.0), (5.0, -1.0), (5.0, 1.0), (7.0, -1.0), (7.0, 1.0)])
    assert len(_line_intersections_on_arc(arc, branchy, 0.1)) > 2
    near_duplicate = LineString([(3.0, -1.0), (3.0, 1.0), (3.05, -1.0), (3.05, 1.0)])
    assert len(_line_intersections_on_arc(arc, near_duplicate, 0.2)) == 1


def test_anchor_full_coastline_keeps_short_segments() -> None:
    arc = LineString([(0.0, 0.0), (10.0, 0.0)])
    full_linework = MultiLineString(
        [
            [(3.0, -1.0), (3.0, 1.0)],
            [(7.0, -0.05), (7.0, 0.05)],
        ]
    )
    assert len(_line_intersections_on_arc(arc, full_linework, 0.001)) == 2


def test_outer_envelope_classifier() -> None:
    coast = MultiLineString(
        [
            [(2.0, 0.0), (2.0, 10.0)],
            [(4.0, 0.0), (4.0, 10.0)],
        ]
    )
    envelope = _build_outer_envelope_index(coast, np.asarray([1.0, 0.0]), target_resolution_m=1.0)
    outer = _anchor_outer_envelope_report(np.asarray([[4.0, 5.0]]), envelope, tolerance_m=0.5)
    inner = _anchor_outer_envelope_report(np.asarray([[2.0, 5.0]]), envelope, tolerance_m=0.5)
    assert outer[0]["passed"], outer
    assert not inner[0]["passed"], inner


def test_endpoint_anchor_classification() -> None:
    arc = LineString([(0.0, 0.0), (10.0, 0.0)])
    envelope = {
        "direction": np.asarray([0.0, 1.0]),
        "perpendicular": np.asarray([1.0, 0.0]),
        "q_min": 0.0,
        "bin_size_m": 1.0,
        "max_s_by_bin": {0: 0.0, 5: 0.0, 9: 0.0, 10: 0.0},
    }
    state = _classify_endpoint_anchors(arc, [Point(0.5, 0.0), Point(5.0, 0.0), Point(9.5, 0.0)], envelope, 0.25)
    assert state["start_anchor_status"] == "valid_outer_envelope", state
    assert state["end_anchor_status"] == "valid_outer_envelope", state
    assert state["middle_extra_intersection_count"] == 1, state
    assert _endpoint_state_has_valid_pair(state), state


def test_endpoint_anchor_same_side_blocks_convergence() -> None:
    arc = LineString([(0.0, 0.0), (10.0, 0.0)])
    envelope = {
        "direction": np.asarray([0.0, 1.0]),
        "perpendicular": np.asarray([1.0, 0.0]),
        "q_min": 0.0,
        "bin_size_m": 1.0,
        "max_s_by_bin": {0: 0.0, 1: 0.0},
    }
    state = _classify_endpoint_anchors(arc, [Point(0.5, 0.0), Point(1.0, 0.0)], envelope, 0.25)
    assert state["start_anchor_status"] == "blocked_by_extras", state
    assert state["end_anchor_status"] == "missing", state
    assert not _endpoint_state_has_valid_pair(state), state


def test_directional_anchor_bezier_tangents() -> None:
    bounds = (0.0, 0.0, 10.0, 10.0)
    p0 = np.asarray([3.0, 2.0])
    p3 = np.asarray([3.0, 8.0])
    direction = np.asarray([1.0, 0.0])
    variant = _directional_bezier_variant(p0, p3, direction, bounds, bow_factor=1.0, target_gap_m=0.1)
    assert variant["endpoint_tangent_error_deg"] < 1.0e-9, variant


def test_anchor_bbox_touch_variants() -> None:
    bounds = (0.0, 0.0, 10.0, 10.0)
    p0 = np.asarray([3.0, 2.0])
    p3 = np.asarray([3.0, 8.0])
    direction = np.asarray([1.0, 0.0])
    small = _directional_bezier_variant(p0, p3, direction, bounds, bow_factor=0.35, target_gap_m=0.2)
    large = _directional_bezier_variant(p0, p3, direction, bounds, bow_factor=2.35, target_gap_m=0.2)
    small_gap, _ = _bbox_touch_info(small["arc"], bounds, "east")
    large_gap, _ = _bbox_touch_info(large["arc"], bounds, "east")
    assert large_gap < small_gap, (small_gap, large_gap)


def test_anchor_iterate_converges_on_synthetic_coast() -> None:
    lon = np.linspace(0.0, 10.0, 50)
    lat = np.linspace(0.0, 10.0, 50)
    depth = np.full((50, 50), 20.0)
    bathy = BathymetryGrid(lon=lon, lat=lat, depth=depth, source="synthetic")
    bbox_wsen = bathy.bbox
    wet_polygon = box(*bbox_wsen)
    coast = gpd.GeoDataFrame(geometry=[LineString([(4.0, 1.0), (4.0, 9.0)])], crs="EPSG:4326")
    result = design_open_boundary(
        wet_polygon,
        bathy,
        bbox_wsen,
        coast,
        target_resolution_m=10_000.0,
        open_spacing_m=50_000.0,
        mode="anchor-iterate",
        ocean_direction=(1.0, 0.0),
        anchor_seeds=(4.8, 2.0, 4.8, 8.0),
        anchor_max_iterations=20,
        anchor_step_factor=1.0,
        anchor_min_step_factor=0.1,
    )
    anchor_meta = result.metadata["anchor_iteration"]
    assert anchor_meta["intersection_count"] == 2, anchor_meta
    assert len(anchor_meta["anchor_points_lonlat"]) == 2, anchor_meta
    assert anchor_meta["endpoint_tangent_error_deg"] < 1.0e-9, anchor_meta


def test_anchor_iterate_true_max_iteration_stop() -> None:
    lon = np.linspace(0.0, 1.0, 20)
    lat = np.linspace(0.0, 1.0, 20)
    depth = np.full((20, 20), 20.0)
    bathy = BathymetryGrid(lon=lon, lat=lat, depth=depth, source="synthetic")
    wet_polygon = box(*bathy.bbox)
    coast = gpd.GeoDataFrame(geometry=[LineString([(-0.25, 0.1), (-0.25, 0.9)])], crs="EPSG:4326")
    result = design_open_boundary(
        wet_polygon,
        bathy,
        bathy.bbox,
        coast,
        target_resolution_m=1_000.0,
        open_spacing_m=50_000.0,
        mode="anchor-iterate",
        ocean_direction=(1.0, 0.0),
        anchor_seeds=(0.8, 0.2, 0.8, 0.8),
        anchor_max_iterations=3,
        anchor_step_factor=0.1,
        anchor_min_step_factor=0.01,
    )
    anchor_meta = result.metadata["anchor_iteration"]
    assert anchor_meta["stop_reason"] == "max_iterations", anchor_meta
    assert len(anchor_meta["history"]) == 3, anchor_meta


if __name__ == "__main__":
    main()
