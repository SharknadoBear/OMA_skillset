#!/usr/bin/env python3
"""Lightweight selftests for fvcom-bdry-arc."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon, box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fvcom_bdry_arc.boundary_resolution as boundary_resolution_module  # noqa: E402

from fvcom_bdry_arc import (  # noqa: E402
    BdryArcConfig,
    BoundaryResolutionConfig,
    BoundaryResolutionV2Config,
    analyze_boundary_resolution,
    boundary_resolution_config,
    build_boundary_resolution,
    build_model_boundary_loops,
    build_open_exterior_contract,
    run_bdry_arc,
)
from fvcom_bdry_arc.boundary_resolution import (  # noqa: E402
    _BoundaryResolutionProgress,
    _enforce_delivered_target_gradation,
    _inventory_narrow_passages,
    _normalize_open_chain_endpoints_on_exterior,
    _passage_gate_taxonomy,
    _sample_closed_open_loop_v2,
    _sample_landward_v2,
    _sample_open_arc_v2,
)
from fvcom_bdry_arc.open_exterior import _read_layer as _read_open_exterior_layer  # noqa: E402
from fvcom_bdry_arc.projection import (  # noqa: E402
    densify_native_geographic_geometry,
    local_utm_projection,
    project_geometry,
    project_geometry_densified,
)
from fvcom_bdry_arc.workflow import (  # noqa: E402
    _classify_relevant_lines,
    _coastline_bpoly_anchor_points,
    _deformed_bpoly_frame,
    _extend_open_arc_to_land_polygons,
    _final_status,
    _gshhs_resolution_policy,
    _normalize_open_arc_to_wet_exterior,
    _promote_delivered_open_arc_landfalls,
    _promote_pre_refinement_landfalls,
    _raster_connectivity_fill,
    _reroute_open_arc_around_blocking_land,
    _route_closed_island_loop_clear_of_land,
    _uses_island_loop_branch,
    extract_gshhs_vector_wet_domain,
    repair_coastline_graph,
    score_and_select_bdry_arc,
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


def _synthetic_lake_gshhs_inputs(root: Path) -> tuple[Path, Path, Path]:
    bpoly = {
        "name": "synthetic_lake",
        "domain_type": "lake",
        "boundary_policy": "no_open_boundary",
        "envelope_bbox": [0.0, 0.0, 4.0, 4.0],
        "polygon_lonlat": [[4.0, 0.0], [0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]],
        "qa": {"bpoly_quality": {"canonical_region_key": "synthetic_lake"}},
        "target_region_features": {
            "features": [
                {"id": "lake_seed", "required": True, "geometry": [2.0, 1.6, 2.3, 2.0]},
            ]
        },
    }
    offshore = {
        "selected_side_index": 3,
        "selected_side_name": "east_or_right",
        "selected_side_start_lonlat": [4.0, 4.0],
        "selected_side_end_lonlat": [4.0, 0.0],
        "offshore_azimuth_deg": 90.0,
        "boundary_policy": "no_open_boundary",
    }
    region_path = root / "region_bpoly.json"
    offshore_path = root / "offshore_boundary_artifacts.json"
    _write_json(region_path, bpoly)
    _write_json(offshore_path, offshore)

    land = gpd.GeoDataFrame(
        [
            {"kind": "shore", "geometry": Polygon([(0.0, 0.0), (0.7, 0.0), (0.7, 4.0), (0.0, 4.0)])},
            {"kind": "island", "geometry": Polygon([(2.6, 2.2), (2.9, 2.2), (2.9, 1.9), (2.6, 1.9)])},
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    coast = gpd.GeoDataFrame(land.drop(columns="geometry"), geometry=land.geometry.boundary, crs="EPSG:4326")
    gpkg = root / "synthetic_lake_gshhs_land.gpkg"
    land.to_file(gpkg, layer="land_polygons", driver="GPKG")
    coast.to_file(gpkg, layer="coastline_lines", driver="GPKG")
    return region_path, offshore_path, gpkg


def _synthetic_island_gshhs_inputs(root: Path) -> tuple[Path, Path, Path]:
    bpoly = {
        "name": "synthetic_island_chain",
        "domain_type": "island",
        "boundary_policy": "offshore_loop_no_land_anchors",
        "envelope_bbox": [0.0, 0.0, 4.0, 4.0],
        "polygon_lonlat": [[4.0, 0.0], [0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]],
        "qa": {"bpoly_quality": {"canonical_region_key": "hawaii_state"}},
        "target_region_features": {
            "features": [
                {"id": "island_seed", "required": True, "geometry": [1.8, 1.8, 2.2, 2.2]},
            ]
        },
    }
    offshore = {
        "selected_side_index": 3,
        "selected_side_name": "east_or_right",
        "selected_side_start_lonlat": [4.0, 4.0],
        "selected_side_end_lonlat": [4.0, 0.0],
        "offshore_azimuth_deg": 90.0,
        "boundary_policy": "offshore_loop_no_land_anchors",
    }
    region_path = root / "region_bpoly.json"
    offshore_path = root / "offshore_boundary_artifacts.json"
    _write_json(region_path, bpoly)
    _write_json(offshore_path, offshore)

    land = gpd.GeoDataFrame(
        [
            {"kind": "island", "geometry": Polygon([(1.7, 1.7), (2.3, 1.7), (2.3, 2.3), (1.7, 2.3)])},
            {"kind": "edge_blocker", "geometry": Polygon([(3.85, 1.7), (4.25, 1.7), (4.25, 2.3), (3.85, 2.3)])},
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    coast = gpd.GeoDataFrame(land.drop(columns="geometry"), geometry=land.geometry.boundary, crs="EPSG:4326")
    gpkg = root / "synthetic_island_gshhs_land.gpkg"
    land.to_file(gpkg, layer="land_polygons", driver="GPKG")
    coast.to_file(gpkg, layer="coastline_lines", driver="GPKG")
    return region_path, offshore_path, gpkg


def _synthetic_loop_package(root: Path, include_boundary_refs: bool = True) -> tuple[Path, Path]:
    exterior = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
    holes = [
        [(3.0, 3.0), (4.0, 3.0), (4.0, 4.0), (3.0, 4.0), (3.0, 3.0)],
        [(6.0, 6.0), (7.0, 6.0), (7.0, 7.0), (6.0, 7.0), (6.0, 6.0)],
    ]
    domain = Polygon(exterior, holes)
    gpkg = root / "synthetic_bdry_arc_package.gpkg"
    gpd.GeoDataFrame([{"geometry": domain}], geometry="geometry", crs="EPSG:4326").to_file(gpkg, layer="wet_domain", driver="GPKG")
    gpd.GeoDataFrame(
        [{"geometry": LineString([(10.0, 10.0), (0.0, 10.0)])}],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(gpkg, layer="open_boundary_arc", driver="GPKG")
    if include_boundary_refs:
        gpd.GeoDataFrame(
            [
                {"geometry": LineString([(10.0, 0.0), (10.0, 10.0)])},
                {"geometry": LineString([(0.0, 10.0), (0.0, 0.0)])},
            ],
            geometry="geometry",
            crs="EPSG:4326",
        ).to_file(gpkg, layer="land_boundary_arcs", driver="GPKG")
        gpd.GeoDataFrame(
            [{"geometry": LineString([(0.0, 0.0), (10.0, 0.0)])}],
            geometry="geometry",
            crs="EPSG:4326",
        ).to_file(gpkg, layer="frame_clip_boundary_arcs", driver="GPKG")
    gpd.GeoDataFrame(
        [{"geometry": LineString(exterior)}],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(gpkg, layer="coastline_repaired", driver="GPKG")
    manifest = root / "bdry_arc_manifest.json"
    _write_json(
        manifest,
        {
            "final_status": "pass",
            "failure_taxonomy": [],
            "settings": {"target_resolution_m": 1000.0},
        },
    )
    return gpkg, manifest


def _resolution_loop_case(
    root: Path,
    exterior: list[tuple[float, float]],
    open_chains: list[LineString],
    expected_obc_count: int,
    *,
    closed: bool = False,
) -> tuple[Path, Path]:
    loops = root / "resolution_loops.gpkg"
    domain = Polygon(exterior)
    gpd.GeoDataFrame(
        [{"geometry": domain}], geometry="geometry", crs="EPSG:4326"
    ).to_file(loops, layer="model_domain_polygon", driver="GPKG")
    segment_rows = [
        {
            "sequence_id": index,
            "segment_class": "open_boundary" if closed else "land_outer_boundary",
            "geometry": LineString([start, end]),
        }
        for index, (start, end) in enumerate(zip(exterior[:-1], exterior[1:]))
    ]
    gpd.GeoDataFrame(segment_rows, geometry="geometry", crs="EPSG:4326").to_file(
        loops, layer="model_outer_boundary_segments", driver="GPKG"
    )
    gpd.GeoDataFrame(
        [
            {
                "obc_id": index,
                "is_closed": bool(closed),
                "segment_class": "delivered_open_boundary_arc",
                "geometry": chain,
            }
            for index, chain in enumerate(open_chains)
        ],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(loops, layer="delivered_open_boundary_arc", driver="GPKG")
    manifest = root / "model_boundary_loop_manifest.json"
    _write_json(
        manifest,
        {
            "final_status": "pass",
            "qa": {
                "expected_obc_count": int(expected_obc_count),
                "delivered_obc_count": int(len(open_chains)),
            },
        },
    )
    return loops, manifest


def _coarse_v2_config() -> BoundaryResolutionConfig:
    return BoundaryResolutionConfig(
        land_spacing_m=5000.0,
        mission_spacing_m=5000.0,
        open_anchor_spacing_m=5000.0,
        open_central_spacing_m=20_000.0,
        compact_spacing_m=5000.0,
        irregular_spacing_m=5000.0,
        elongated_spacing_m=5000.0,
        complex_spacing_m=5000.0,
        passage_search_spacing_m=5000.0,
        passage_max_width_m=100.0,
    )


def test_boundary_resolution_profile_is_v2_only() -> None:
    assert BdryArcConfig().boundary_resolution_profile == "adaptive-coastal-v2"
    assert boundary_resolution_config().profile == "adaptive-coastal-v2"
    assert boundary_resolution_config("adaptive-coastal-v2").profile == "adaptive-coastal-v2"
    for removed in ("legacy", "adaptive-coastal-v1"):
        try:
            boundary_resolution_config(removed)
        except ValueError as exc:
            assert "only supports adaptive-coastal-v2" in str(exc)
        else:
            raise AssertionError(f"removed generation profile was accepted: {removed}")


def test_boundary_resolution_progress_records_cancellation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        progress = _BoundaryResolutionProgress(root, interval_s=0.0)
        progress.emit("start", "start", 0, 1, force=True)
        progress.emit(
            "source_island_metrics",
            "running",
            3,
            10,
            {"island_id": 2},
            force=True,
        )
        progress._record_process_exit()
        state = json.loads(progress.state_path.read_text(encoding="utf-8"))
        assert state["current_phase"] == "source_island_metrics"
        assert state["current_message"] == "cancelled"
        assert state["processed_count"] == 3
        assert state["total_count"] == 10
        assert state["phase_percent"] == 30.0
        assert state["last_details"]["cancellation_reason"] == "process_exit_before_completion"
        records = [
            json.loads(line)
            for line in progress.jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert records[-1]["message"] == "cancelled"
        assert records[-1]["processed_count"] == 3
        progress._record_process_exit()
        assert len(progress.jsonl_path.read_text(encoding="utf-8").splitlines()) == len(records)


def test_boundary_resolution_progress_retries_transient_windows_lock() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        real_replace = boundary_resolution_module.os.replace
        calls = 0

        def flaky_replace(source: str | Path, target: str | Path) -> None:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise PermissionError(5, "synthetic sync-client lock", str(target))
            real_replace(source, target)

        boundary_resolution_module.os.replace = flaky_replace
        try:
            progress = _BoundaryResolutionProgress(root, interval_s=0.0)
            progress.emit("start", "start", 0, 1, force=True)
        finally:
            boundary_resolution_module.os.replace = real_replace
        assert calls == 3
        state = json.loads(progress.state_path.read_text(encoding="utf-8"))
        assert state["current_phase"] == "start"
        assert state["current_message"] == "start"
        progress._record_process_exit()


def test_boundary_resolution_progress_records_unhandled_failure() -> None:
    sentinel = object()
    previous = getattr(sys, "last_exc", sentinel)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            progress = _BoundaryResolutionProgress(Path(tmp), interval_s=0.0)
            progress.emit("quality_gates", "running", 2, 3, force=True)
            sys.last_exc = ValueError("synthetic failure")
            progress._record_process_exit()
            state = json.loads(progress.state_path.read_text(encoding="utf-8"))
            assert state["current_message"] == "failed"
            assert state["last_details"]["failure_reason"] == "unhandled_exception"
            assert state["last_details"]["exception_type"] == "ValueError"
            assert state["last_details"]["exception_message"] == "synthetic failure"
    finally:
        if previous is sentinel:
            delattr(sys, "last_exc")
        else:
            sys.last_exc = previous


def test_target_gradation_projection_converges_on_large_closed_loop() -> None:
    count = 4096
    radius = 1_000_000.0
    entries = []
    for index in range(count):
        angle = 2.0 * math.pi * index / count
        is_anchor = index in {0, count // 2}
        entries.append(
            {
                "xy": (radius * math.cos(angle), radius * math.sin(angle)),
                "target_spacing_m": 500.0 if is_anchor else 8000.0,
                "anchor_type": "open_loop_seam" if index == 0 else (
                    "open_loop_balance" if index == count // 2 else "regular"
                ),
            }
        )
    report = _enforce_delivered_target_gradation(entries, 0.15)
    points = np.asarray([entry["xy"] for entry in entries], dtype=float)
    targets = np.asarray([entry["target_spacing_m"] for entry in entries], dtype=float)
    lengths = np.maximum(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1), 1.0)
    gradients = np.abs(np.roll(targets, -1) - targets) / lengths
    assert report["method"] == "anchor_preserving_cycle_shortest_path_lipschitz_projection"
    assert report["iteration_count"] == 1
    assert report["relaxation_count"] > 0
    assert float(np.max(gradients)) <= 0.15 + 1.0e-9
    assert entries[0]["target_spacing_m"] == 500.0
    assert entries[count // 2]["target_spacing_m"] == 500.0

    incompatible = [
        {"xy": (0.0, 0.0), "target_spacing_m": 500.0, "anchor_type": "open_loop_seam"},
        {"xy": (1000.0, 0.0), "target_spacing_m": 1000.0, "anchor_type": "open_loop_balance"},
    ]
    try:
        _enforce_delivered_target_gradation(incompatible, 0.15)
    except ValueError as exc:
        assert "Fixed Adaptive v2 anchors" in str(exc)
    else:
        raise AssertionError("incompatible fixed anchors were accepted")


def test_two_independent_coastal_obcs_preserve_ids_and_anchors() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exterior = [
            (-75.0, 39.0),
            (-74.9, 39.0),
            (-74.9, 39.1),
            (-75.0, 39.1),
            (-75.0, 39.0),
        ]
        open_chains = [
            LineString([(-74.9, 39.02), (-74.9, 39.04)]),
            LineString([(-75.0, 39.08), (-75.0, 39.06)]),
        ]
        loops, loop_manifest = _resolution_loop_case(root, exterior, open_chains, 2)
        manifest = build_boundary_resolution(
            loops,
            loop_manifest,
            None,
            None,
            root / "resolution",
            "two_obc",
            _coarse_v2_config(),
        )
        assert manifest["final_status"] == "pass", manifest["failure_taxonomy"]
        assert manifest["qa"]["expected_obc_count"] == 2
        assert manifest["qa"]["delivered_obc_count"] == 2
        chains = manifest["open_boundary_chains"]
        assert [item["obc_id"] for item in chains] == [0, 1]
        assert all(item["is_closed"] is False for item in chains)
        assert all(item["open_landfall_hard_anchor_count"] == 2 for item in chains)
        assert all(item["exterior_overlap_fraction"] >= 1.0 - 1.0e-9 for item in chains)
        assert all(item["nonendpoint_land_intersection_m"] <= 1.0e-6 for item in chains)
        assert set(chains[0]["node_sequence_zero_based"]).isdisjoint(
            chains[1]["node_sequence_zero_based"]
        )
        resolved = gpd.read_file(
            manifest["outputs"]["boundary_resolution_gpkg"],
            layer="resolved_open_boundary",
        )
        assert list(resolved.sort_values("obc_id")["obc_id"]) == [0, 1]
        assert len(resolved) == 2


def test_v2_open_endpoint_normalization_is_bounded() -> None:
    exterior = LineString(box(0.0, 0.0, 1000.0, 1000.0).exterior.coords)
    delivered = LineString([(1000.0, 100.0), (1000.0, 500.0), (925.0, 800.0)])
    normalized, report = _normalize_open_chain_endpoints_on_exterior(
        delivered,
        exterior,
        250.0,
    )
    assert report["normalized"] is True
    assert report["endpoint_snap_distance_m"] == [0.0, 75.0]
    assert Point(normalized.coords[0]).distance(exterior) <= 1.0e-8
    assert Point(normalized.coords[-1]).distance(exterior) <= 1.0e-8
    assert list(normalized.coords)[1] == list(delivered.coords)[1]
    try:
        _normalize_open_chain_endpoints_on_exterior(
            LineString([(1000.0, 100.0), (500.0, 500.0)]),
            exterior,
            250.0,
        )
    except ValueError as exc:
        assert "too far from the model exterior" in str(exc)
    else:
        raise AssertionError("An out-of-contract OBC endpoint was normalized")


def test_closed_island_obc_uses_seam_and_balance_without_landfalls() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exterior = [
            (-160.2, 20.8),
            (-159.8, 20.8),
            (-159.8, 21.2),
            (-160.2, 21.2),
            (-160.2, 20.8),
        ]
        loops, loop_manifest = _resolution_loop_case(
            root,
            exterior,
            [LineString(exterior)],
            1,
            closed=True,
        )
        manifest = build_boundary_resolution(
            loops,
            loop_manifest,
            None,
            None,
            root / "resolution",
            "closed_island",
            _coarse_v2_config(),
        )
        assert manifest["final_status"] == "pass", manifest["failure_taxonomy"]
        chain = manifest["open_boundary_chains"][0]
        assert chain["is_closed"] is True
        assert chain["open_landfall_hard_anchor_count"] == 0
        assert chain["open_loop_seam_hard_anchor_count"] == 1
        assert chain["open_loop_balance_hard_anchor_count"] == 1
        assert chain["exterior_overlap_fraction"] >= 1.0 - 1.0e-9
        assert chain["nonendpoint_land_intersection_m"] <= 1.0e-6


def test_antimeridian_closed_obc_uses_compact_projection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exterior = [
            (179.0, 50.0),
            (-179.0, 50.0),
            (-179.0, 52.0),
            (179.0, 52.0),
            (179.0, 50.0),
        ]
        loops, loop_manifest = _resolution_loop_case(
            root,
            exterior,
            [LineString(exterior)],
            1,
            closed=True,
        )
        config = BoundaryResolutionConfig(
            **{
                **_coarse_v2_config().__dict__,
                "land_spacing_m": 25_000.0,
                "open_anchor_spacing_m": 25_000.0,
                "open_central_spacing_m": 50_000.0,
            }
        )
        manifest = build_boundary_resolution(
            loops,
            loop_manifest,
            None,
            None,
            root / "resolution",
            "antimeridian_loop",
            config,
        )
        assert manifest["final_status"] == "pass", manifest["failure_taxonomy"]
        chain = manifest["open_boundary_chains"][0]
        assert chain["is_closed"] is True
        assert chain["open_landfall_hard_anchor_count"] == 0
        assert chain["open_loop_seam_hard_anchor_count"] == 1
        assert chain["open_loop_balance_hard_anchor_count"] == 1
        assert chain["source_length_m"] < 1_000_000.0


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
        assert Path(manifest["outputs"]["model_boundary_loop_manifest"]).exists()
        assert Path(manifest["outputs"]["model_boundary_loops_gpkg"]).exists()
        assert Path(manifest["outputs"]["model_boundary_segments_geojson"]).exists()
        assert Path(manifest["outputs"]["model_boundary_colored_map"]).exists()
        assert manifest["model_boundary_loops"]["final_status"] in {"pass", "needs_review"}
        assert Path(manifest["outputs"]["open_exterior_contract"]).exists()
        assert "region_bpoly_arc_feedback" not in manifest
        assert Path(manifest["outputs"]["visual_review_dir"], "preliminary_arc_map.png").exists()
        assert Path(manifest["outputs"]["visual_review_dir"], "arc_candidate_contact_sheet.png").exists()
        assert manifest["wet_domain"]["area_m2"] > 0
        assert manifest["offshore_arc"]["candidate_count"] >= 4
        layers = set(gpd.list_layers(manifest["outputs"]["bdry_arc_package_gpkg"])["name"])
        required = {
            "wet_domain",
            "open_boundary_arc",
            "land_boundary_arcs",
            "frame_clip_boundary_arcs",
            "land_patch_boundary_arcs",
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
        assert manifest["wet_domain"]["closure_method"] == "coastline_anchor_seaward_bpoly_chain"
        assert manifest["anchors"]["source"] == "coastline_bpoly_intersection"
        assert manifest["anchors"]["start_role"] == "exact_physical_land_polygon_landfall"
        assert manifest["anchors"]["end_role"] == "exact_physical_land_polygon_landfall"
        assert manifest["anchors"]["start_anchor_found"] is True
        assert manifest["anchors"]["end_anchor_found"] is True
        assert manifest["anchors"]["landfall_acceptance_tolerance_m"] is None
        assert manifest["anchors"]["landfall_construction_stage"] == "before_adaptive_arc_refinement"
        assert abs(manifest["anchors"]["start_lonlat"][0] - 1.0) < 0.01
        assert abs(manifest["anchors"]["start_lonlat"][1] - 4.0) < 0.01
        assert abs(manifest["anchors"]["end_lonlat"][0] - 1.0) < 0.01
        assert abs(manifest["anchors"]["end_lonlat"][1] - 0.0) < 0.01
        assert len(manifest["anchors"]["seaward_chain_lonlat"]) == 4
        open_arc = gpd.read_file(manifest["outputs"]["bdry_arc_package_gpkg"], layer="open_boundary_arc").geometry.iloc[0]
        assert Point(open_arc.coords[0]).distance(Point(manifest["anchors"]["start_lonlat"])) < 1.0e-8
        assert Point(open_arc.coords[-1]).distance(Point(manifest["anchors"]["end_lonlat"])) < 1.0e-8
        assert Path(manifest["outputs"]["bdry_arc_package_gpkg"]).exists()
        assert Path(manifest["outputs"]["model_boundary_loop_manifest"]).exists()
        assert Path(manifest["outputs"]["model_boundary_loops_gpkg"]).exists()
        assert Path(manifest["outputs"]["model_boundary_segments_geojson"]).exists()
        assert Path(manifest["outputs"]["model_boundary_colored_map"]).exists()
        assert manifest["model_boundary_loops"]["qa"]["outer_boundary_closed"] is True
        assert Path(root / "run" / "bdry_arc_progress_state.json").exists()
        assert (visual_dir / "gshhs_polygon_topology_map.png").exists()
        assert (visual_dir / "gshhs_anchor_arc_map.png").exists()
        assert not list(visual_dir.glob("raster_connectivity_iter_*.png"))


def test_island_loop_branch_avoids_coastline_anchor_failures() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        region, offshore, gpkg = _synthetic_island_gshhs_inputs(root)
        manifest = run_bdry_arc(
            region,
            offshore,
            root / "run",
            "synthetic_island",
            coastline_gpkg=gpkg,
            config=BdryArcConfig(mode="test", target_resolution_m=5000.0, coastline_source="gshhs", topology_mode="gshhs-vector"),
        )
        assert manifest["settings"]["topology_mode_used"] == "island-loop"
        assert manifest["wet_domain"]["closure_method"] == "island_archipelago_offshore_loop"
        assert manifest["anchors"]["source"] == "offshore_loop_no_land_anchors"
        assert "start_coastline_bpoly_anchor_missing" not in manifest["failure_taxonomy"]
        assert "end_coastline_bpoly_anchor_missing" not in manifest["failure_taxonomy"]
        assert manifest["wet_domain"]["land_patch_policy"] == "land_patch"
        layers = set(gpd.list_layers(manifest["outputs"]["bdry_arc_package_gpkg"])["name"])
        assert "land_patch_boundary_arcs" in layers
        open_arc = gpd.read_file(manifest["outputs"]["bdry_arc_package_gpkg"], layer="open_boundary_arc").geometry.iloc[0]
        assert Point(open_arc.coords[0]).distance(Point(open_arc.coords[-1])) < 1.0e-8


def test_gshhs_resolution_policy_no_silent_downgrade() -> None:
    policy = _gshhs_resolution_policy(BdryArcConfig(gshhs_resolution="f"), {"gshhs_selected_resolution": "h"})
    assert policy["downgraded_without_explicit_request"] is True
    policy = _gshhs_resolution_policy(
        BdryArcConfig(gshhs_resolution="f"),
        {
            "gshhs_requested_resolution": "h",
            "gshhs_selected_resolution": "h",
        },
    )
    assert policy["downgraded_without_explicit_request"] is False
    assert policy["explicit_lower_resolution_requested"] is True
    policy = _gshhs_resolution_policy(BdryArcConfig(gshhs_resolution="h"), {"gshhs_selected_resolution": "h"})
    assert policy["downgraded_without_explicit_request"] is False
    assert policy["explicit_lower_resolution_requested"] is True


def test_memory_off_disables_canonical_only_island_routing() -> None:
    region = {
        "domain_type": "coastal",
        "boundary_policy": "coastal_arc_with_land_anchors",
        "qa": {"bpoly_quality": {"canonical_region_key": "hawaii_state"}},
    }
    offshore = {"boundary_policy": "coastal_arc_with_land_anchors"}
    config = BdryArcConfig(mode="test", coastline_source="gshhs", topology_mode="gshhs-vector")
    assert _uses_island_loop_branch(region, offshore, config, place_memory_enabled=False) is False
    assert _uses_island_loop_branch(region, offshore, config, place_memory_enabled=True) is True
    explicit_region = {**region, "domain_type": "island"}
    assert _uses_island_loop_branch(explicit_region, offshore, config, place_memory_enabled=False) is True
    explicit_offshore = {"boundary_policy": "offshore_loop_no_land_anchors"}
    assert _uses_island_loop_branch(region, explicit_offshore, config, place_memory_enabled=False) is True


def test_antimeridian_projection_uses_compact_longitude_frame() -> None:
    projection = local_utm_projection((172.0, 48.0, -158.0, 58.0))
    assert projection.longitude_origin is not None
    assert projection.epsg != 32632
    assert projection.longitude_origin > 150.0 or projection.longitude_origin < -150.0


def test_antimeridian_sparse_edges_are_densified_without_longitude_warping() -> None:
    projection = local_utm_projection((172.0, 51.0, -158.0, 55.5))
    native = Polygon(
        [(172.0, 51.0), (-158.0, 51.0), (-158.0, 55.5), (172.0, 55.5), (172.0, 51.0)]
    )
    densified = densify_native_geographic_geometry(native, maximum_segment_degrees=0.25)
    native_longitudes = [float(lon) for lon, _lat in densified.exterior.coords]
    assert min(native_longitudes) >= -180.0
    assert max(native_longitudes) <= 180.0
    projected = project_geometry_densified(native, projection)
    north_outside = project_geometry(Point(-160.25, 55.68), projection)
    assert projected.is_valid
    assert not projected.covers(north_outside)
    assert north_outside.distance(projected) > 10_000.0


def test_closed_island_loop_routing_preserves_protected_land_and_excludes_mainland() -> None:
    mission = box(0.0, 0.0, 10_000.0, 10_000.0)
    protected_island = box(9_000.0, 4_000.0, 10_500.0, 5_500.0)
    detached_protected_island = box(13_000.0, 7_500.0, 13_500.0, 8_300.0)
    external_mainland = box(-2_000.0, 3_000.0, 1_000.0, 7_000.0)
    nearby_external_island = box(13_000.0, 8_400.0, 13_500.0, 8_800.0)
    land = [protected_island, detached_protected_island, external_mainland, nearby_external_island]
    land_union = unary_union(land).buffer(0)
    source_loop = LineString(mission.exterior.coords)
    routed, report = _route_closed_island_loop_clear_of_land(
        source_loop,
        land,
        land_union,
        mission,
        Point(5_000.0, 5_000.0),
        250.0,
        protected_island_regions_xy=[
            box(8_500.0, 3_500.0, 11_000.0, 6_000.0),
            box(12_500.0, 7_000.0, 14_000.0, 8_350.0),
        ],
    )
    frame = Polygon(routed.coords)
    assert routed.is_ring and routed.is_simple
    assert report["land_free"] is True
    assert report["seed_preserved"] is True
    assert report["retained_inside_component_fraction"] == 1.0
    assert frame.buffer(5.0).covers(protected_island)
    assert frame.buffer(5.0).covers(detached_protected_island)
    assert any(action["accepted"] for action in report["protected_component_connection_actions"])
    assert not frame.covers(external_mainland.representative_point())
    assert routed.intersection(land_union).length <= 1.0e-6


def test_closed_island_loop_batch_connects_intervening_protected_components() -> None:
    mission = box(0.0, 0.0, 10_000.0, 10_000.0)
    outer_protected = box(13_000.0, 4_500.0, 15_000.0, 6_500.0)
    intervening_protected = box(11_500.0, 3_500.0, 12_000.0, 7_500.0)
    external_mainland = box(-2_000.0, 3_000.0, 1_000.0, 7_000.0)
    land = [outer_protected, intervening_protected, external_mainland]
    land_union = unary_union(land).buffer(0)
    routed, report = _route_closed_island_loop_clear_of_land(
        LineString(mission.exterior.coords),
        land,
        land_union,
        mission,
        Point(5_000.0, 5_000.0),
        250.0,
        protected_island_regions_xy=[
            box(12_500.0, 4_000.0, 15_500.0, 7_000.0),
            box(11_000.0, 3_000.0, 12_500.0, 8_000.0),
        ],
    )
    frame = Polygon(routed.coords)
    assert report["land_free"] is True
    assert report["retained_inside_component_fraction"] == 1.0
    assert report["protected_component_batch_connection_report"]["accepted"] is True
    assert report["protected_component_connection_pass_count"] == 0
    assert report["protected_component_unresolved_ids"] == []
    assert frame.covers(outer_protected)
    assert frame.covers(intervening_protected)
    assert routed.intersection(land_union).length <= 1.0e-6


def test_clearance_connected_required_fragment_inherits_external_land_role() -> None:
    mission = box(0.0, 0.0, 10_000.0, 10_000.0)
    required_fragment = box(10_006.0, 4_000.0, 10_500.0, 4_500.0)
    external_mainland = box(10_506.0, 1_000.0, 20_000.0, 9_000.0)
    land = [required_fragment, external_mainland]
    land_union = unary_union(land).buffer(0)
    routed, report = _route_closed_island_loop_clear_of_land(
        LineString(mission.exterior.coords),
        land,
        land_union,
        mission,
        Point(5_000.0, 5_000.0),
        250.0,
        protected_island_regions_xy=[box(10_000.0, 3_500.0, 10_505.0, 5_000.0)],
    )
    role_report = report["clearance_connected_land_role_report"]
    assert role_report["reclassified_component_count"] == 1
    assert role_report["reclassified_components"][0]["gap_to_external_land_m"] == 6.0
    assert report["retained_inside_component_count"] == 0
    assert report["retained_inside_component_fraction"] == 1.0
    assert report["land_free"] is True
    assert routed.intersection(land_union).length <= 1.0e-6


def test_lake_closed_boundary_no_false_open_arc() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        region, offshore, gpkg = _synthetic_lake_gshhs_inputs(root)
        manifest = run_bdry_arc(
            region,
            offshore,
            root / "run",
            "synthetic_lake",
            coastline_gpkg=gpkg,
            config=BdryArcConfig(mode="test", target_resolution_m=5000.0, coastline_source="gshhs", topology_mode="gshhs-vector"),
        )
        assert manifest["settings"]["topology_mode_used"] == "lake-closed-boundary"
        assert manifest["settings"]["lake_closed_boundary_branch"] is True
        assert manifest["wet_domain"]["closure_method"] == "lake_closed_boundary_no_open_arc"
        assert manifest["wet_domain"]["no_ocean_open_boundary"] is True
        assert "open_arc_not_on_final_boundary" not in manifest["failure_taxonomy"]
        open_arc = gpd.read_file(manifest["outputs"]["bdry_arc_package_gpkg"], layer="open_boundary_arc")
        usable_open_arcs = [geom for geom in open_arc.geometry if geom is not None and not geom.is_empty]
        assert not usable_open_arcs
        loop_manifest = json.loads(Path(manifest["outputs"]["model_boundary_loop_manifest"]).read_text(encoding="utf-8"))
        assert loop_manifest["settings"]["lake_no_open_boundary"] is True
        assert "open_boundary_not_sufficiently_on_model_exterior" not in loop_manifest["failure_taxonomy"]


def test_upstream_review_status_is_nonblocking() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        region_path, offshore_path, gpkg = _synthetic_gshhs_inputs(root)
        region = json.loads(region_path.read_text(encoding="utf-8"))
        region.update(
            {
                "schema_version": "region_bpoly_final_v1",
                "final_status": "needs_review",
                "output_package": {"package_state": "internal_review", "delivery_ready": False},
            }
        )
        region.setdefault("qa", {})["land_side_visual_gate"] = {"status": "unresolved"}
        _write_json(region_path, region)
        manifest = run_bdry_arc(
            region_path,
            offshore_path,
            root / "run",
            "review_status_nonblocking",
            coastline_gpkg=gpkg,
            config=BdryArcConfig(
                mode="test",
                target_resolution_m=5000.0,
                coastline_source="gshhs",
                topology_mode="gshhs-vector",
            ),
        )
        upstream = manifest["inputs"]["upstream_region_bpoly"]
        assert upstream["accepted_for_boundary_generation"] is True
        assert upstream["status_fields_are_nonblocking"] is True
        assert upstream["final_status"] == "needs_review"
        assert manifest["settings"]["topology_mode_used"] != "upstream-unresolved"
        assert "upstream_region_bpoly_review_status_nonblocking" in manifest["advisory_taxonomy"]


def test_coastline_anchor_seaward_chain_closes_boundary() -> None:
    bpoly = box(0.0, 0.0, 10_000.0, 10_000.0)
    land = [
        box(0.0, 0.0, 2_000.0, 10_000.0),
        box(4_000.0, 4_000.0, 4_800.0, 4_800.0),
    ]
    coast = [poly.boundary for poly in land]
    selected_side = LineString([(10_000.0, 10_000.0), (10_000.0, 0.0)])
    anchors = _coastline_bpoly_anchor_points(coast[0], selected_side, bpoly, 250.0)
    anchors["landfall_construction_stage"] = "before_adaptive_arc_refinement"
    arc = LineString([(2_000.0, 10_000.0), (11_000.0, 8_000.0), (11_000.0, 2_000.0), (2_000.0, 0.0)])
    result = extract_gshhs_vector_wet_domain(coast, land, arc, bpoly, Point(4_000.0, 5_000.0), 250.0, anchors=anchors)
    wet = result["wet_domain_xy"]
    metadata = result["metadata"]
    assert wet.contains(Point(4_000.0, 5_000.0))
    assert not wet.contains(Point(1_000.0, 5_000.0))
    assert metadata["source"] == "coastline_anchor_seaward_bpoly_chain"
    assert metadata["closure_method"] == "coastline_anchor_seaward_bpoly_chain"
    assert metadata["deformed_frame_valid"] is True
    assert metadata["open_arc_boundary_overlap_fraction"] >= 0.98
    assert metadata["frame_clip_boundary_length_m"] >= 0.0


def test_open_arc_crossing_blocker_is_rerouted() -> None:
    bpoly = box(0.0, 0.0, 10_000.0, 10_000.0)
    land = [box(9_500.0, 3_000.0, 10_500.0, 7_000.0)]
    coast = [box(0.0, 0.0, 2_000.0, 10_000.0).boundary]
    selected_side = LineString([(10_000.0, 10_000.0), (10_000.0, 0.0)])
    anchors = _coastline_bpoly_anchor_points(coast[0], selected_side, bpoly, 250.0)
    anchors["landfall_construction_stage"] = "before_adaptive_arc_refinement"
    arc = LineString([(2_000.0, 10_000.0), (10_700.0, 5_000.0), (2_000.0, 0.0)])
    result = extract_gshhs_vector_wet_domain([], land, arc, bpoly, Point(7_000.0, 5_000.0), 250.0, anchors=anchors)
    assert result["metadata"]["arc_land_intersection"] is False
    assert result["metadata"]["open_arc_blocking_land_rerouted"] is True
    assert result["metadata"]["open_arc_blocking_land_unresolved_count"] == 0
    status, failures = _final_status(
        {"selected": {"metrics": {"extra_coastline_intersection": True}}},
        result,
        {**anchors, "start_distance_m": 0.0, "end_distance_m": 0.0},
        [],
    )
    assert status == "pass"
    assert "gshhs_open_arc_crosses_land" not in failures
    assert "open_arc_intersects_extra_coastline" not in failures


def test_source_arc_tail_trims_to_delivered_landfalls() -> None:
    source_arc = LineString([(-2.0, 0.0), (0.0, 0.0), (10.0, 0.0), (12.0, 0.0)])
    wet_domain = box(0.0, -5.0, 10.0, 0.0)
    land_boundary = MultiLineString(
        [
            LineString([(0.0, -2.0), (0.0, 2.0)]),
            LineString([(10.0, -2.0), (10.0, 2.0)]),
        ]
    )
    delivered, report = _normalize_open_arc_to_wet_exterior(
        source_arc,
        wet_domain,
        land_boundary,
        10.0,
    )
    assert report["open_arc_trimmed_to_wet_exterior"] is True
    assert report["discarded_source_open_arc_length_m"] == 4.0
    assert list(delivered.coords)[0] == (0.0, 0.0)
    assert list(delivered.coords)[-1] == (10.0, 0.0)

    anchors = _promote_delivered_open_arc_landfalls(
        {
            "source": "coastline_bpoly_intersection",
            "start_xy": (-2.0, 0.0),
            "end_xy": (12.0, 0.0),
            "selected_side_start_corner_xy": (-2.0, 0.0),
            "selected_side_end_corner_xy": (12.0, 0.0),
            "start_distance_m": 0.0,
            "end_distance_m": 0.0,
            "landfall_construction_stage": "before_adaptive_arc_refinement",
        },
        delivered,
        land_boundary,
        10.0,
    )
    assert anchors["start_xy"] == (0.0, 0.0)
    assert anchors["end_xy"] == (10.0, 0.0)
    assert anchors["start_anchor_method"] == "wet_exterior_land_intersection"
    status, failures = _final_status(
        {"selected": {"metrics": {"extra_coastline_intersection": True}}},
        {
            "metadata": {
                "source": "coastline_anchor_seaward_bpoly_chain",
                "closure_method": "coastline_anchor_seaward_bpoly_chain",
                "deformed_frame_valid": True,
                "open_arc_boundary_overlap_fraction": 1.0,
                "open_arc_trimmed_to_wet_exterior": True,
                "arc_land_intersection": False,
                "seed_inside": True,
                "forbidden_overlap": [],
            }
        },
        anchors,
        [],
    )
    assert status == "pass"
    assert "open_arc_intersects_extra_coastline" not in failures


def test_blocking_island_is_rerouted_inside_seeded_frame() -> None:
    bpoly = box(0.0, 0.0, 10_000.0, 10_000.0)
    source_arc = LineString([(0.0, 0.0), (10_000.0, 0.0)])
    blocker = box(4_000.0, -1_000.0, 6_000.0, 1_000.0)
    seed = Point(5_000.0, 5_000.0)

    rerouted, report = _reroute_open_arc_around_blocking_land(
        source_arc,
        [blocker],
        blocker,
        bpoly,
        seed,
        250.0,
        None,
    )

    assert report["open_arc_blocking_land_initial_count"] == 1
    assert report["open_arc_blocking_land_rerouted"] is True
    assert report["open_arc_blocking_land_reroute_count"] == 1
    assert report["open_arc_blocking_land_unresolved_count"] == 0
    assert rerouted.is_simple
    assert rerouted.intersection(blocker.buffer(2.0)).length <= 2.0
    frame, _ = _deformed_bpoly_frame(bpoly, rerouted, None)
    assert frame.covers(blocker.representative_point())
    assert frame.difference(blocker).buffer(2.0).contains(seed)


def test_unprotected_passage_is_advisory_only() -> None:
    failures, advisories = _passage_gate_taxonomy(
        {
            "protected_unresolved_count": 0,
            "unprotected_unresolved_count": 3,
        }
    )
    assert failures == []
    assert advisories == ["unprotected_passage_underresolved"]

    failures, advisories = _passage_gate_taxonomy(
        {
            "protected_unresolved_count": 1,
            "unprotected_unresolved_count": 3,
        }
    )
    assert failures == ["protected_passage_underresolved"]
    assert advisories == ["unprotected_passage_underresolved"]


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


def test_coastline_bpoly_anchor_selection_rules() -> None:
    bpoly = box(0.0, 0.0, 10_000.0, 10_000.0)
    selected_side = LineString([(10_000.0, 10_000.0), (10_000.0, 0.0)])
    coastline = LineString([(2_000.0, 10_000.0), (3_000.0, 10_000.0), (3_000.0, 0.0), (2_000.0, 0.0)])
    anchors = _coastline_bpoly_anchor_points(coastline, selected_side, bpoly, 250.0)
    assert anchors["source"] == "coastline_bpoly_intersection"
    assert anchors["start_anchor_method"] == "exact_intersection"
    assert anchors["end_anchor_method"] == "exact_intersection"
    assert Point(anchors["start_xy"]).distance(Point(3_000.0, 10_000.0)) < 1.0e-8
    assert Point(anchors["end_xy"]).distance(Point(3_000.0, 0.0)) < 1.0e-8
    assert anchors["seaward_chain_xy"][1] == (10_000.0, 10_000.0)
    assert anchors["seaward_chain_xy"][2] == (10_000.0, 0.0)
    assert anchors["landfall_acceptance_tolerance_m"] is None
    assert anchors["provisional_side_reference_only"] is True


def test_exact_pre_refinement_landfall_extension_has_no_snap_tolerance() -> None:
    land = unary_union(
        [
            box(-5.0, -5.0, 0.0, 5.0),
            box(10.0, -5.0, 15.0, 5.0),
        ]
    )
    provisional = LineString([(2.0, 0.0), (5.0, 0.0), (8.0, 0.0)])
    extended, report = _extend_open_arc_to_land_polygons(provisional, land)
    assert list(extended.coords)[0] == (0.0, 0.0)
    assert list(extended.coords)[-1] == (10.0, 0.0)
    assert report["acceptance_tolerance_m"] is None
    assert report["construction_stage"] == "before_adaptive_arc_refinement"
    assert math.isclose(report["start"]["extension_length_m"], 2.0, abs_tol=1.0e-12)
    assert math.isclose(report["end"]["extension_length_m"], 2.0, abs_tol=1.0e-12)
    assert report["start"]["distance_to_land_polygon_boundary_m"] == 0.0
    assert report["end"]["distance_to_land_polygon_boundary_m"] == 0.0
    assert extended.is_simple
    config = BoundaryResolutionV2Config(
        open_anchor_spacing_m=2.0,
        open_central_spacing_m=4.0,
    )
    _coords, _sizes, metadata, sampling = _sample_open_arc_v2(
        extended,
        config,
        land_union=land,
    )
    assert sampling["feature_anchor_count"] == 0
    assert sum(item["anchor_type"] == "open_landfall" for item in metadata) == 2


def test_exact_pre_refinement_inside_land_tails_trim_to_first_waterward_exits() -> None:
    land = unary_union(
        [
            box(-5.0, -5.0, 0.0, 5.0),
            box(10.0, -5.0, 15.0, 5.0),
        ]
    )
    provisional = LineString([(-3.0, 0.0), (-1.0, 0.0), (2.0, 0.0), (8.0, 0.0), (12.0, 0.0)])
    prepared, report = _extend_open_arc_to_land_polygons(provisional, land)
    assert list(prepared.coords) == [(0.0, 0.0), (2.0, 0.0), (8.0, 0.0), (10.0, 0.0)]
    assert report["acceptance_tolerance_m"] is None
    assert report["extension_length_m"] == 0.0
    assert math.isclose(report["source_trim_length_m"], 5.0, abs_tol=1.0e-12)
    assert report["start"]["method"] == "source_chain_first_waterward_land_polygon_exit"
    assert report["end"]["method"] == "source_chain_first_waterward_land_polygon_exit"
    assert math.isclose(report["start"]["trim_length_m"], 3.0, abs_tol=1.0e-12)
    assert math.isclose(report["end"]["trim_length_m"], 2.0, abs_tol=1.0e-12)
    assert report["start"]["distance_to_land_polygon_boundary_m"] == 0.0
    assert report["end"]["distance_to_land_polygon_boundary_m"] == 0.0
    config = BoundaryResolutionV2Config(
        open_anchor_spacing_m=2.0,
        open_central_spacing_m=4.0,
    )
    _coords, _sizes, metadata, sampling = _sample_open_arc_v2(
        prepared,
        config,
        land_union=land,
    )
    assert sampling["feature_anchor_count"] == 0
    assert sum(item["anchor_type"] == "open_landfall" for item in metadata) == 2


def test_inside_land_tail_without_waterward_exit_is_typed_blocker() -> None:
    land = box(-5.0, -5.0, 15.0, 5.0)
    provisional = LineString([(-3.0, 0.0), (2.0, 0.0), (12.0, 0.0)])
    try:
        _extend_open_arc_to_land_polygons(provisional, land)
    except ValueError as exc:
        assert "no exact waterward land-polygon exit" in str(exc)
    else:
        raise AssertionError("Expected an inside-land source chain without a water exit to be rejected")


def test_inside_land_predicate_accepts_exact_endpoint_crossing_without_tolerance() -> None:
    land = unary_union(
        [
            box(-5.0, -5.0, 0.0, 5.0),
            box(10.0, -5.0, 15.0, 5.0),
        ]
    )
    provisional = LineString([(0.0, 0.0), (4.0, 0.0), (8.0, 0.0), (12.0, 0.0)])
    prepared, report = _extend_open_arc_to_land_polygons(provisional, land)
    assert list(prepared.coords) == [(0.0, 0.0), (4.0, 0.0), (8.0, 0.0), (10.0, 0.0)]
    assert report["acceptance_tolerance_m"] is None
    assert report["start"]["extension_length_m"] == 0.0
    assert report["end"]["method"] == "source_chain_first_waterward_land_polygon_exit"


def test_missing_exact_land_polygon_intersection_needs_review() -> None:
    bpoly = box(0.0, 0.0, 10_000.0, 10_000.0)
    selected_side = LineString([(10_000.0, 10_000.0), (10_000.0, 0.0)])
    coastline = LineString([(-10_000.0, -10_000.0), (-9_000.0, -9_000.0)])
    anchors = _coastline_bpoly_anchor_points(coastline, selected_side, bpoly, 250.0)
    assert anchors["start_anchor_found"] is True
    assert anchors["end_anchor_found"] is True
    status, failures = _final_status(
        {"selected": {"metrics": {"extra_coastline_intersection": False}}},
        {
            "metadata": {
                "closure_method": "coastline_anchor_seaward_bpoly_chain",
                "deformed_frame_valid": True,
                "open_arc_boundary_overlap_fraction": 1.0,
                "seed_inside": True,
                "forbidden_overlap": [],
            }
        },
        anchors,
        [],
    )
    assert status == "needs_review"
    assert "exact_pre_refinement_landfall_construction_missing" in failures


def test_model_boundary_loop_package() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gpkg, source_manifest = _synthetic_loop_package(root, include_boundary_refs=True)
        manifest = build_model_boundary_loops(
            gpkg,
            source_manifest,
            root / "loops",
            "synthetic_loop",
            mode="test",
        )
        assert manifest["final_status"] == "pass"
        assert manifest["qa"]["outer_boundary_closed"] is True
        assert manifest["qa"]["island_count"] == 2
        assert manifest["qa"]["open_boundary_exterior_overlap_fraction"] >= 0.98
        out_gpkg = Path(manifest["outputs"]["model_boundary_loops_gpkg"])
        assert out_gpkg.exists()
        assert Path(manifest["outputs"]["model_boundary_segments_geojson"]).exists()
        assert Path(manifest["outputs"]["model_boundary_colored_map"]).exists()
        layers = set(gpd.list_layers(out_gpkg)["name"])
        required = {
            "model_domain_polygon",
            "model_outer_boundary",
            "model_outer_boundary_segments",
            "island_boundary_polygons",
            "island_boundary_lines",
            "delivered_open_boundary_arc",
            "source_open_boundary_arc",
        }
        assert required.issubset(layers), sorted(required - layers)
        segments = gpd.read_file(out_gpkg, layer="model_outer_boundary_segments")
        classes = set(segments["segment_class"])
        assert {"open_boundary", "land_outer_boundary", "frame_clip_boundary"}.issubset(classes)
        islands = gpd.read_file(out_gpkg, layer="island_boundary_polygons")
        assert len(islands) == 2


def test_model_boundary_loop_unclassified_needs_review() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gpkg, source_manifest = _synthetic_loop_package(root, include_boundary_refs=False)
        manifest = build_model_boundary_loops(
            gpkg,
            source_manifest,
            root / "loops",
            "synthetic_loop_unclassified",
            mode="test",
        )
        assert manifest["final_status"] == "needs_review"
        assert "unclassified_outer_boundary_length_nontrivial" in manifest["failure_taxonomy"]
        assert manifest["qa"]["unclassified_outer_boundary_length_m"] > manifest["settings"]["unclassified_length_threshold_m"]


def test_adaptive_boundary_resolution_package() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exterior = [(-75.0, 39.0), (-74.9, 39.0), (-74.9, 39.1), (-75.0, 39.1), (-75.0, 39.0)]
        holes = [
            [(-74.970, 39.040), (-74.965, 39.040), (-74.965, 39.045), (-74.970, 39.045), (-74.970, 39.040)],
            [(-74.930, 39.060), (-74.925, 39.060), (-74.925, 39.063), (-74.930, 39.063), (-74.930, 39.060)],
        ]
        domain = Polygon(exterior, holes)
        loops = root / "loops.gpkg"
        gpd.GeoDataFrame([{"geometry": domain}], geometry="geometry", crs="EPSG:4326").to_file(loops, layer="model_domain_polygon", driver="GPKG")
        records = []
        for index, (start, end) in enumerate(zip(exterior[:-1], exterior[1:])):
            records.append(
                {
                    "sequence_id": index,
                    "segment_class": "open_boundary" if index == 1 else "land_outer_boundary",
                    "geometry": LineString([start, end]),
                }
            )
        gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326").to_file(loops, layer="model_outer_boundary_segments", driver="GPKG")
        gpd.GeoDataFrame(
            [{"island_id": index, "area_m2": 1.0, "geometry": Polygon(hole)} for index, hole in enumerate(holes)],
            geometry="geometry",
            crs="EPSG:4326",
        ).to_file(loops, layer="island_boundary_polygons", driver="GPKG")
        region = root / "region.json"
        _write_json(
            region,
            {
                "target_region_features": {
                    "features": [
                        {"role": "target_water_body", "geometry": [-75.0, 39.0, -74.95, 39.08]},
                    ]
                }
            },
        )
        coast = root / "coast.gpkg"
        gpd.GeoDataFrame([{"geometry": box(-74.89, 39.0, -74.88, 39.1)}], geometry="geometry", crs="EPSG:4326").to_file(coast, layer="land_polygons", driver="GPKG")
        analysis = analyze_boundary_resolution(loops, region)
        assert analysis["island_count"] == 2
        assert analysis["protected_count"] >= 1
        manifest = build_boundary_resolution(
            loops,
            None,
            region,
            coast,
            root / "resolution",
            "synthetic_adaptive",
            BoundaryResolutionConfig(progress_interval_s=0.0),
        )
        assert manifest["profile"] == "adaptive-coastal-v2"
        assert manifest["final_status"] == "pass"
        assert manifest["qa"]["open_boundary_node_count"] >= 2
        assert manifest["qa"]["open_arc_land_intersection_m"] <= 1.0e-6
        assert manifest["qa"]["open_arc_exterior_overlap_fraction"] >= 1.0 - 1.0e-9
        assert manifest["qa"]["topology_absolute_area_change_fraction"] <= 0.005 + 1.0e-12
        diagnostics = json.loads(Path(manifest["outputs"]["boundary_resolution_diagnostics_json"]).read_text(encoding="utf-8"))
        protected = [item for item in diagnostics["resolved_islands"] if item["protected_mission"]]
        assert protected
        assert all(item["generalized_area_error_fraction"] <= 1.0e-12 for item in protected)
        assert all(item["principal_orientation_change_deg"] <= 1.0e-9 for item in protected)
        assert all(item["target_spacing_m"] <= 150.0 for item in protected)
        layers = set(gpd.list_layers(manifest["outputs"]["boundary_resolution_gpkg"])["name"])
        assert {"resolved_domain_polygon", "resolved_open_boundary", "resolved_island_polygons", "boundary_nodes", "island_diagnostics"}.issubset(layers)

        progress_state = json.loads(
            Path(manifest["outputs"]["boundary_resolution_progress_state"]).read_text(encoding="utf-8")
        )
        assert progress_state["current_phase"] == "complete"
        assert progress_state["current_message"] == "complete"
        assert progress_state["overall_percent"] == 100.0
        progress_records = [
            json.loads(line)
            for line in Path(manifest["outputs"]["boundary_resolution_progress_jsonl"])
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        overall = [float(item["overall_percent"]) for item in progress_records]
        assert overall == sorted(overall)
        island_metric_records = [
            item for item in progress_records if item["phase"] == "source_island_metrics"
        ]
        assert island_metric_records[0]["processed_count"] == 0
        assert island_metric_records[0]["total_count"] == 2
        assert [item["processed_count"] for item in island_metric_records[-2:]] == [1, 2]
        assert island_metric_records[-1]["phase_percent"] == 100.0

        assert manifest["qa"]["open_landfall_hard_anchor_count"] == 2
        node_doc = json.loads(Path(manifest["outputs"]["boundary_resolution_nodes_geojson"]).read_text(encoding="utf-8"))
        landfalls = [
            feature
            for feature in node_doc["features"]
            if feature["properties"].get("anchor_type") == "open_landfall"
        ]
        assert len(landfalls) == 2
        assert all(feature["properties"]["is_hard_anchor"] for feature in landfalls)
        v2_diagnostics = json.loads(
            Path(manifest["outputs"]["boundary_resolution_diagnostics_json"]).read_text(encoding="utf-8")
        )
        assert len(v2_diagnostics["boundary_sampling"]["junctions"]) == 2
        assert all(item["hard_anchor"] for item in v2_diagnostics["boundary_sampling"]["junctions"])


def test_adaptive_uses_exact_delivered_obc_not_proximity_tails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exterior = [
            (-75.0, 39.0),
            (-74.9, 39.0),
            (-74.9, 39.1),
            (-75.0, 39.1),
            (-75.0, 39.0),
        ]
        domain = Polygon(exterior)
        loops = root / "loops.gpkg"
        gpd.GeoDataFrame(
            [{"geometry": domain}], geometry="geometry", crs="EPSG:4326"
        ).to_file(loops, layer="model_domain_polygon", driver="GPKG")
        records = []
        for index, (start, end) in enumerate(zip(exterior[:-1], exterior[1:])):
            records.append(
                {
                    "sequence_id": index,
                    "segment_class": "open_boundary" if index == 1 else "land_outer_boundary",
                    "geometry": LineString([start, end]),
                }
            )
        gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326").to_file(
            loops, layer="model_outer_boundary_segments", driver="GPKG"
        )
        exact_delivered = LineString([(-74.9, 39.02), (-74.9, 39.08)])
        gpd.GeoDataFrame(
            [{"segment_class": "delivered_open_boundary_arc", "geometry": exact_delivered}],
            geometry="geometry",
            crs="EPSG:4326",
        ).to_file(loops, layer="delivered_open_boundary_arc", driver="GPKG")
        coast = root / "coast.gpkg"
        gpd.GeoDataFrame(
            [{"geometry": box(-75.01, 39.0, -75.005, 39.1)}],
            geometry="geometry",
            crs="EPSG:4326",
        ).to_file(coast, layer="land_polygons", driver="GPKG")
        manifest = build_boundary_resolution(
            loops,
            None,
            None,
            coast,
            root / "resolution",
            "exact_delivered_obc",
            BoundaryResolutionV2Config(passage_max_width_m=100.0),
        )
        assert manifest["final_status"] == "pass", manifest["failure_taxonomy"]
        assert manifest["qa"]["open_arc_source"] == "exact_delivered_open_boundary_arc"
        assert manifest["qa"]["proximity_classified_excess_length_m"] > 0.0
        resolved = gpd.read_file(
            manifest["outputs"]["boundary_resolution_gpkg"],
            layer="resolved_open_boundary",
        ).to_crs("EPSG:4326").geometry.iloc[0]
        assert abs(resolved.coords[0][1] - 39.02) <= 1.0e-8
        assert abs(resolved.coords[-1][1] - 39.08) <= 1.0e-8
        assert min(point[1] for point in resolved.coords) >= 39.02 - 1.0e-8
        assert max(point[1] for point in resolved.coords) <= 39.08 + 1.0e-8


def test_v2_feature_anchors_and_junction_spacing() -> None:
    config = BoundaryResolutionV2Config(
        land_spacing_m=150.0,
        open_anchor_spacing_m=500.0,
        open_central_spacing_m=1500.0,
    )
    open_line = LineString([(0.0, 0.0), (1000.0, 0.0), (1200.0, 600.0), (2600.0, 600.0)])
    open_nodes, open_h, open_meta, report = _sample_open_arc_v2(open_line, config)
    assert report["feature_anchor_count"] >= 1
    assert open_meta[0]["anchor_type"] == "open_landfall" and open_meta[0]["is_hard_anchor"]
    assert open_meta[-1]["anchor_type"] == "open_landfall" and open_meta[-1]["is_hard_anchor"]
    assert any(meta["anchor_type"] in {"sharp_turn", "spit_tip"} for meta in open_meta)
    assert any(Point(node).distance(Point(1000.0, 0.0)) <= 1.0e-8 for node in open_nodes)
    assert open_h[0] == 500.0 and open_h[-1] == 500.0

    land_line = LineString([(0.0, 0.0), (3000.0, 0.0), (3000.0, 2000.0), (6500.0, 2000.0)])
    land_nodes, land_h, land_meta, land_report = _sample_landward_v2(land_line, [], config)
    assert land_h[0] == 500.0 and land_h[-1] == 500.0
    assert min(land_h) <= 150.0 + 1.0e-9
    assert land_report["junction_transition_length_m"] == (500.0 - 150.0) / 0.15
    assert any(meta["anchor_type"] in {"sharp_turn", "spit_tip"} for meta in land_meta)
    assert any(Point(node).distance(Point(3000.0, 0.0)) <= 1.0e-8 for node in land_nodes)


def test_closed_open_loop_sampling_refines_land_crossing_shortcuts() -> None:
    line = LineString(
        [
            (-10.0, -10.0),
            (10.0, -10.0),
            (10.0, 10.0),
            (-10.0, 10.0),
            (-10.0, -10.0),
        ]
    )
    land = box(-1.0, -1.0, 1.0, 1.0)
    config = BoundaryResolutionV2Config(
        open_anchor_spacing_m=1000.0,
        open_central_spacing_m=1000.0,
        sharp_turn_threshold_deg=181.0,
        spit_turn_threshold_deg=181.0,
        anchor_chord_error_fraction=100.0,
    )
    unguarded = [
        tuple(line.interpolate(0.0).coords[0]),
        tuple(line.interpolate(0.5 * line.length).coords[0]),
        tuple(line.interpolate(line.length).coords[0]),
    ]
    guarded, _, metadata, report = _sample_closed_open_loop_v2(
        line,
        config,
        land_union=land,
    )
    assert LineString(unguarded).intersection(land).length > 0.0
    assert not LineString(unguarded).is_simple
    assert LineString(guarded).intersection(land).length <= 1.0e-6
    assert LineString(guarded).is_simple
    safety = report["land_safety_refinement"]
    assert safety["added_node_count"] >= 2
    assert safety["remaining_unsafe_chord_count"] == 0
    assert safety["remaining_land_intersection_m"] <= 1.0e-6
    assert safety["remaining_self_intersection_pair_count"] == 0
    assert safety["sampled_chain_simple"] is True
    anchor_types = {item["anchor_type"] for item in metadata if item["is_hard_anchor"]}
    assert {"open_loop_seam", "open_loop_balance"}.issubset(anchor_types)


def test_v2_passage_inventory_harmonizes_or_gates_without_closure() -> None:
    projection = local_utm_projection((-75.1, 38.9, -74.9, 39.1))
    origin = project_geometry(Point(-75.0, 39.0), projection)
    x0, y0 = float(origin.x), float(origin.y)
    outer = [(x0, y0), (x0 + 10_000.0, y0), (x0 + 10_000.0, y0 + 10_000.0), (x0, y0 + 10_000.0)]
    landward = LineString([outer[2], outer[3], outer[0], outer[1]])
    first = box(x0 + 3500.0, y0 + 4000.0, x0 + 4500.0, y0 + 6000.0)
    second = box(x0 + 5500.0, y0 + 4000.0, x0 + 6500.0, y0 + 6000.0)
    domain = Polygon(outer, holes=[list(first.exterior.coords), list(second.exterior.coords)])
    mission = box(x0 + 4300.0, y0 + 3500.0, x0 + 5700.0, y0 + 6500.0)
    report, controls, island_targets = _inventory_narrow_passages(
        landward,
        [first, second],
        domain,
        mission,
        BoundaryResolutionV2Config(passage_max_width_m=1500.0),
        projection,
    )
    paired = [item for item in report["passages"] if item["bank_a"] == "island" and item["bank_b"] == "island"]
    assert paired
    assert paired[0]["protected_mission"] is True
    assert paired[0]["action"] == "harmonize_paired_spacing"
    assert abs(paired[0]["required_target_spacing_m"] - 250.0) <= 1.0e-8
    assert island_targets[0] == 250.0 and island_targets[1] == 250.0
    assert report["minimum_spacing_policy"] == "adaptive_from_minimum_protected_passage_width"
    assert abs(report["minimum_protected_passage_width_m"] - 1000.0) <= 1.0e-8
    assert abs(report["minimum_permitted_spacing_m"] - 250.0) <= 1.0e-8
    assert report["automatic_topology_operation_count"] == 0
    assert controls == []

    narrow_second = box(x0 + 4850.0, y0 + 4000.0, x0 + 5850.0, y0 + 6000.0)
    narrow_domain = Polygon(outer, holes=[list(first.exterior.coords), list(narrow_second.exterior.coords)])
    narrow_report, _, narrow_targets = _inventory_narrow_passages(
        landward,
        [first, narrow_second],
        narrow_domain,
        mission,
        BoundaryResolutionV2Config(passage_max_width_m=1000.0),
        projection,
    )
    assert narrow_report["protected_unresolved_count"] == 0
    assert narrow_report["minimum_spacing_policy"] == "adaptive_from_minimum_protected_passage_width"
    assert abs(narrow_report["minimum_protected_passage_width_m"] - 350.0) <= 1.0e-8
    assert abs(narrow_report["minimum_permitted_spacing_m"] - 87.5) <= 1.0e-8
    assert all(item["action"] == "harmonize_paired_spacing" for item in narrow_report["passages"])
    assert narrow_targets[0] == 87.5 and narrow_targets[1] == 87.5
    assert narrow_report["automatic_topology_operation_count"] == 0

    override_report, _, _ = _inventory_narrow_passages(
        landward,
        [first, narrow_second],
        narrow_domain,
        mission,
        BoundaryResolutionV2Config(passage_max_width_m=1000.0, passage_min_spacing_m=100.0),
        projection,
    )
    assert override_report["minimum_spacing_policy"] == "explicit_configuration"
    assert override_report["protected_unresolved_count"] >= 1
    assert any(item["action"] == "retain_needs_review" for item in override_report["passages"])

    no_protected_config = BoundaryResolutionV2Config(passage_max_width_m=1000.0)
    no_protected_report, _, _ = _inventory_narrow_passages(
        landward,
        [first, narrow_second],
        narrow_domain,
        None,
        no_protected_config,
        projection,
    )
    assert no_protected_report["minimum_spacing_policy"] == "configured_land_spacing_no_protected_passage"
    assert no_protected_report["minimum_permitted_spacing_m"] == no_protected_config.land_spacing_m
    assert no_protected_report["protected_unresolved_count"] == 0
    assert no_protected_report["unprotected_unresolved_count"] >= 1


def test_v2_passage_inventory_uses_conservative_sparse_broad_phase() -> None:
    projection = local_utm_projection((-75.5, 38.5, -72.5, 41.5))
    origin = project_geometry(Point(-75.0, 39.0), projection)
    x0, y0 = float(origin.x), float(origin.y)
    islands = [
        box(x0 + 10_000.0 * column, y0 + 10_000.0 * row, x0 + 10_000.0 * column + 500.0, y0 + 10_000.0 * row + 500.0)
        for row in range(5)
        for column in range(5)
    ]
    outer = [
        (x0 - 5_000.0, y0 - 5_000.0),
        (x0 + 46_000.0, y0 - 5_000.0),
        (x0 + 46_000.0, y0 + 46_000.0),
        (x0 - 5_000.0, y0 + 46_000.0),
    ]
    domain = Polygon(outer, holes=[list(island.exterior.coords) for island in islands])
    report, controls, island_targets = _inventory_narrow_passages(
        [],
        islands,
        domain,
        None,
        BoundaryResolutionV2Config(passage_max_width_m=1_000.0),
        projection,
    )
    assert report["component_pair_index_policy"] == (
        "expanded_envelope_broad_phase_then_exact_distance_and_wet_connector"
    )
    assert report["wet_connector_domain_buffer_policy"] == (
        "exact_domain_buffers_prepared_once_per_inventory"
    )
    assert report["all_component_pair_count"] == 300
    assert report["spatially_indexed_component_pair_count"] == 0
    assert report["passage_count"] == 0
    assert controls == []
    assert island_targets == {}


def test_coastal_obc_scoring_is_compact_not_bpoly_containment_driven() -> None:
    bpoly = box(0.0, 0.0, 10.0, 10.0)
    compact = {
        "candidate_id": "compact",
        "geometry": LineString([(10.0, 2.0), (12.0, 5.0), (10.0, 8.0)]),
        "bow_distance_m": 2.0,
    }
    excessive = {
        "candidate_id": "excessive",
        "geometry": LineString([(10.0, 2.0), (18.0, 5.0), (10.0, 8.0)]),
        "bow_distance_m": 8.0,
    }
    scored = score_and_select_bdry_arc(
        [excessive, compact],
        None,
        bpoly,
        target_resolution_m=0.25,
    )
    assert scored["selected"]["candidate_id"] == "compact"
    assert scored["selected"]["metrics"]["outside_bpoly_fraction"] > 0.0
    assert scored["selected"]["metrics"]["offshore_obc_bpoly_containment_required"] is False


def test_coastal_obc_self_intersection_remains_blocking() -> None:
    status, failures = _final_status(
        {"selected": {"metrics": {"extra_coastline_intersection": False}}},
        {
            "open_arc_xy": LineString([(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]),
            "metadata": {
                "closure_method": "coastline_anchor_seaward_bpoly_chain",
                "deformed_frame_valid": True,
                "seed_inside": True,
                "open_arc_boundary_overlap_fraction": 1.0,
                "arc_land_intersection": False,
                "forbidden_overlap": [],
            },
        },
        {"source": "synthetic", "start_distance_m": 0.0, "end_distance_m": 0.0},
        [],
    )
    assert status == "needs_review"
    assert "open_arc_self_intersects_or_branches" in failures


def test_open_exterior_contract_is_non_mutating_and_obc_unbound() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        region_path = root / "region_bpoly.json"
        offshore_path = root / "offshore_boundary_artifacts.json"
        region_doc = {
            "name": "unbound_obc",
            "domain_type": "coastal",
            "boundary_policy": "coastal_arc_with_land_anchors",
            "expected_obc_count": 1,
            "polygon_lonlat": [[4.0, 0.0], [0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]],
            "region_bpoly": {
                "polygon_lonlat": [[4.0, 0.0], [0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]],
                "edge_labels": ["south", "west", "north", "east"],
            },
        }
        _write_json(region_path, region_doc)
        region_before = region_path.read_bytes()
        _write_json(offshore_path, {"expected_obc_count": 1, "selected_side_index": 3})
        package = root / "package.gpkg"
        wet = Polygon([(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)])
        obc = LineString([(6.0, 0.0), (6.0, 4.0)])
        gpd.GeoDataFrame([{"geometry": wet}], geometry="geometry", crs="EPSG:4326").to_file(package, layer="wet_domain", driver="GPKG")
        gpd.GeoDataFrame([{"geometry": obc}], geometry="geometry", crs="EPSG:4326").to_file(package, layer="open_boundary_arc", driver="GPKG")
        gpd.GeoDataFrame([{"geometry": obc}], geometry="geometry", crs="EPSG:4326").to_file(package, layer="frame_clip_boundary_arcs", driver="GPKG")

        land_path = root / "land.gpkg"
        land = gpd.GeoDataFrame(
            [
                {"geometry": Polygon([(-1.0, -1.0), (7.0, -1.0), (7.0, 0.0), (-1.0, 0.0)])},
                {"geometry": Polygon([(-1.0, 4.0), (7.0, 4.0), (7.0, 5.0), (-1.0, 5.0)])},
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )
        land.to_file(land_path, layer="land_polygons", driver="GPKG")
        land.boundary.to_frame("geometry").set_crs("EPSG:4326").to_file(land_path, layer="coastline_lines", driver="GPKG")
        loop_path = root / "loop.json"
        _write_json(loop_path, {"final_status": "pass", "failure_taxonomy": [], "qa": {"open_boundary_exterior_overlap_fraction": 1.0}})
        contract = build_open_exterior_contract(
            region_path,
            offshore_path,
            package,
            land_path,
            loop_path,
            root / "open_exterior",
            {
                "inputs": {"coastline_source": "gshhs", "coastline_load": {"source_version": "GSHHG 2.3.7"}},
                "settings": {"target_resolution_m": 5000.0, "gshhs_resolution": "f", "gshhs_levels": "1", "obc_placement_policy": "offshore-first"},
                "wet_domain": {"arc_land_intersection_length_m": 0.0},
                "coastline_source_coverage": {"downstream_eligible": True, "failure_taxonomy": []},
            },
            frame_clip_policy="reject-unintended",
            residual_boundary_policy="solid-default",
            frame_clip_tolerance_m=250.0,
        )
        assert contract["schema_version"] == "fvcom_open_exterior_contract_v2"
        assert contract["boundary_lengths"]["open_boundary_outside_region_bpoly_fraction"] > 0.99
        assert contract["offshore_obc_bpoly_containment_required"] is False
        assert "boundary_completeness" not in contract
        assert region_path.read_bytes() == region_before
        assert Path(contract["outputs"]["open_exterior_contract"]).exists()
        assert Path(contract["outputs"]["open_exterior_review_map"]).exists()
        assert not list(root.rglob("*feedback*"))


def test_open_exterior_reader_drops_empty_geometry_placeholders() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.gpkg"
        gpd.GeoDataFrame([{"geometry": None}], geometry="geometry", crs="EPSG:4326").to_file(path, layer="frame_clip_boundary_arcs", driver="GPKG")
        assert _read_open_exterior_layer(path, "frame_clip_boundary_arcs").empty


def main() -> int:
    test_boundary_resolution_profile_is_v2_only()
    test_boundary_resolution_progress_records_cancellation()
    test_boundary_resolution_progress_records_unhandled_failure()
    test_target_gradation_projection_converges_on_large_closed_loop()
    test_two_independent_coastal_obcs_preserve_ids_and_anchors()
    test_v2_open_endpoint_normalization_is_bounded()
    test_closed_island_obc_uses_seam_and_balance_without_landfalls()
    test_antimeridian_closed_obc_uses_compact_projection()
    test_synthetic_package()
    test_gshhs_vector_package_prefers_coastline_lines()
    test_island_loop_branch_avoids_coastline_anchor_failures()
    test_gshhs_resolution_policy_no_silent_downgrade()
    test_memory_off_disables_canonical_only_island_routing()
    test_antimeridian_projection_uses_compact_longitude_frame()
    test_antimeridian_sparse_edges_are_densified_without_longitude_warping()
    test_closed_island_loop_routing_preserves_protected_land_and_excludes_mainland()
    test_closed_island_loop_batch_connects_intervening_protected_components()
    test_clearance_connected_required_fragment_inherits_external_land_role()
    test_lake_closed_boundary_no_false_open_arc()
    test_upstream_review_status_is_nonblocking()
    test_coastline_anchor_seaward_chain_closes_boundary()
    test_open_arc_crossing_blocker_is_rerouted()
    test_source_arc_tail_trims_to_delivered_landfalls()
    test_blocking_island_is_rerouted_inside_seeded_frame()
    test_unprotected_passage_is_advisory_only()
    test_endpoint_repair_is_conservative()
    test_raster_fill_respects_connectivity_barrier()
    test_component_classification_drops_disconnected_lines()
    test_coastline_bpoly_anchor_selection_rules()
    test_exact_pre_refinement_landfall_extension_has_no_snap_tolerance()
    test_exact_pre_refinement_inside_land_tails_trim_to_first_waterward_exits()
    test_inside_land_tail_without_waterward_exit_is_typed_blocker()
    test_inside_land_predicate_accepts_exact_endpoint_crossing_without_tolerance()
    test_missing_exact_land_polygon_intersection_needs_review()
    test_model_boundary_loop_package()
    test_model_boundary_loop_unclassified_needs_review()
    test_adaptive_boundary_resolution_package()
    test_adaptive_uses_exact_delivered_obc_not_proximity_tails()
    test_v2_feature_anchors_and_junction_spacing()
    test_closed_open_loop_sampling_refines_land_crossing_shortcuts()
    test_v2_passage_inventory_harmonizes_or_gates_without_closure()
    test_v2_passage_inventory_uses_conservative_sparse_broad_phase()
    test_coastal_obc_scoring_is_compact_not_bpoly_containment_driven()
    test_coastal_obc_self_intersection_remains_blocking()
    test_open_exterior_contract_is_non_mutating_and_obc_unbound()
    test_open_exterior_reader_drops_empty_geometry_placeholders()
    print("fvcom-bdry-arc selftests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
