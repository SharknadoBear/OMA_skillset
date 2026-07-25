#!/usr/bin/env python3
"""Selftests for the rebuilt fvcom-grid-generation skill."""

from __future__ import annotations

import json
from copy import deepcopy
import tempfile
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.bathymetry import BathymetryGrid, coarsen_for_size_field, write_synthetic_bathymetry  # noqa: E402
from fvcom_grid_generation.boundary import BoundaryConfig, load_boundary_package, load_boundary_resolution, prepare_boundary_nodes  # noqa: E402
from fvcom_grid_generation.mesh import _ordered_boundary_kind_group  # noqa: E402
from fvcom_grid_generation.local_topology import AggressiveConditioningConfig, condition_mesh_aggressive, inventory_high_valence  # noqa: E402
from fvcom_grid_generation.metrics import compute_mesh_metrics, triangle_geometry  # noqa: E402
from fvcom_grid_generation.postprocess import PostprocessConfig, _stage_acceptance, postprocess_mesh  # noqa: E402
from fvcom_grid_generation.quality import evaluate_mesh_quality  # noqa: E402
from fvcom_grid_generation.regional_conditioning import (  # noqa: E402
    AreaTransitionRelaxConfig,
    SpringRelaxConfig,
    ThinTriangleRepairConfig,
    relax_mesh_area_transitions,
    relax_mesh_spring,
    repair_thin_triangles,
)
from fvcom_grid_generation.size_field import SizeFieldConfig, build_size_field  # noqa: E402
from fvcom_grid_generation.bathymetry import load_bathymetry  # noqa: E402
from fvcom_grid_generation.sms_2dm import read_2dm, write_2dm  # noqa: E402
from fvcom_grid_generation.workflow import (  # noqa: E402
    GridConfig,
    _bathy_fetch_command,
    _channel_flownet_command,
    _load_channel_flownet_manifest,
    _parse_required_source_count,
    _select_channel_surface,
    _write_analysis_mask,
    run_fvcom_grid,
)


def _synthetic_boundary_package(path: Path) -> Path:
    exterior = [(-75.10, 39.00), (-74.90, 39.00), (-74.90, 39.16), (-75.10, 39.16), (-75.10, 39.00)]
    hole = [(-75.015, 39.070), (-74.990, 39.070), (-74.990, 39.095), (-75.015, 39.095), (-75.015, 39.070)]
    domain = Polygon(exterior, [hole])
    gpd.GeoDataFrame([{"geometry": domain}], geometry="geometry", crs="EPSG:4326").to_file(path, layer="model_domain_polygon", driver="GPKG")
    segments = gpd.GeoDataFrame(
        [
            {"segment_class": "land_outer_boundary", "geometry": LineString([exterior[0], exterior[1]])},
            {"segment_class": "open_boundary", "geometry": LineString([exterior[1], exterior[2]])},
            {"segment_class": "land_outer_boundary", "geometry": LineString([exterior[2], exterior[3]])},
            {"segment_class": "land_outer_boundary", "geometry": LineString([exterior[3], exterior[4]])},
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    segments.to_file(path, layer="model_outer_boundary_segments", driver="GPKG")
    gpd.GeoDataFrame([{"geometry": Polygon(hole)}], geometry="geometry", crs="EPSG:4326").to_file(path, layer="island_boundary_polygons", driver="GPKG")
    gpd.GeoDataFrame([{"geometry": LineString([exterior[1], exterior[2]])}], geometry="geometry", crs="EPSG:4326").to_file(path, layer="source_open_boundary_arc", driver="GPKG")
    return path


def _simple_synthetic_boundary_package(path: Path) -> Path:
    exterior = [
        (-75.10, 39.00),
        (-74.90, 39.00),
        (-74.90, 39.16),
        (-75.10, 39.16),
        (-75.10, 39.00),
    ]
    domain = Polygon(exterior)
    gpd.GeoDataFrame(
        [{"geometry": domain}],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(path, layer="model_domain_polygon", driver="GPKG")
    gpd.GeoDataFrame(
        [
            {
                "segment_class": "land_outer_boundary",
                "geometry": LineString([exterior[0], exterior[1]]),
            },
            {
                "segment_class": "open_boundary",
                "geometry": LineString([exterior[1], exterior[2]]),
            },
            {
                "segment_class": "open_boundary",
                "geometry": LineString([exterior[2], exterior[3]]),
            },
            {
                "segment_class": "land_outer_boundary",
                "geometry": LineString([exterior[3], exterior[4]]),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(path, layer="model_outer_boundary_segments", driver="GPKG")
    gpd.GeoDataFrame(
        [{"geometry": LineString([exterior[1], exterior[2], exterior[3]])}],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(path, layer="source_open_boundary_arc", driver="GPKG")
    return path


def _synthetic_resolution_package(root: Path, loops: Path) -> Path:
    package = load_boundary_package(loops)
    nodes = prepare_boundary_nodes(package, BoundaryConfig(land_spacing_m=700.0, open_spacing_m=2500.0, island_spacing_m=700.0))
    gpkg = root / "boundary_resolution.gpkg"
    gpd.GeoDataFrame([{"profile": "adaptive-coastal-v1", "geometry": package.domain_polygon_lonlat}], crs="EPSG:4326").to_file(gpkg, layer="resolved_domain_polygon", driver="GPKG")
    gpd.GeoDataFrame([{"segment_class": "open_boundary", "geometry": package.open_boundary_lonlat}], crs="EPSG:4326").to_file(gpkg, layer="resolved_open_boundary", driver="GPKG")
    if package.island_polygons_lonlat:
        gpd.GeoDataFrame([{"resolved_island_id": i, "geometry": polygon} for i, polygon in enumerate(package.island_polygons_lonlat)], crs="EPSG:4326").to_file(gpkg, layer="resolved_island_polygons", driver="GPKG")
    rows = []
    for chain_id, chain in enumerate(nodes.constraint_chains):
        for position, node in enumerate(chain):
            lon, lat = nodes.lonlat[node]
            target = 2500.0 if nodes.kinds[node] == "open" else 700.0
            rows.append(
                {
                    "node_index_zero_based": int(node),
                    "chain_id": int(chain_id),
                    "chain_position": int(position),
                    "boundary_kind": nodes.kinds[node],
                    "target_spacing_m": target,
                    "is_hard_anchor": False,
                    "geometry": Point(float(lon), float(lat)),
                }
            )
    gpd.GeoDataFrame(rows, crs="EPSG:4326").to_file(gpkg, layer="boundary_nodes", driver="GPKG")
    manifest = root / "boundary_resolution_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "fvcom_boundary_resolution_manifest_v1",
                "profile": "adaptive-coastal-v1",
                "final_status": "pass",
                "inputs": {"model_boundary_loops_gpkg": str(loops)},
                "outputs": {"boundary_resolution_gpkg": str(gpkg)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def test_boundary_ingestion_and_densification() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gpkg = _synthetic_boundary_package(root / "loops.gpkg")
        pkg = load_boundary_package(gpkg)
        nodes = prepare_boundary_nodes(pkg, BoundaryConfig(land_spacing_m=500.0, open_spacing_m=2500.0, island_spacing_m=500.0))
        assert nodes.xy.shape[0] > 20
        assert len(nodes.open_boundary_indices) >= 2
        assert sum(1 for kind in nodes.kinds if kind in {"land", "island"}) > len(nodes.open_boundary_indices)
        assert len(nodes.constraint_chains) == 2


def test_size_field_limiter_never_coarsens_fine_cells() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gpkg = _synthetic_boundary_package(root / "loops.gpkg")
        pkg = load_boundary_package(gpkg)
        nodes = prepare_boundary_nodes(pkg, BoundaryConfig(land_spacing_m=700.0, open_spacing_m=2500.0, island_spacing_m=700.0))
        bathy_path = write_synthetic_bathymetry(root / "bathy.nc", (-75.11, 38.99, -74.89, 39.17), nx=35, ny=35)
        bathy = load_bathymetry(bathy_path)
        size = build_size_field(bathy, nodes, SizeFieldConfig(land_spacing_m=700.0, open_spacing_m=2500.0, gradation=0.15))
        assert np.nanmin(size.size) >= 700.0 - 1.0e-6
        assert np.all(size.size <= size.raw_size + 1.0e-6)
        assert size.report["gradation"]["max_neighbor_gradation"] <= 0.151


def test_unified_oceanmesh_candidates_are_coastal_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gpkg = _synthetic_boundary_package(root / "loops.gpkg")
        package = load_boundary_package(gpkg)
        nodes = prepare_boundary_nodes(package, BoundaryConfig(land_spacing_m=700.0, open_spacing_m=2500.0, island_spacing_m=700.0))
        nodes.adaptive_resolution = True
        lon = np.linspace(-75.11, -74.89, 67)
        lat = np.linspace(38.99, 39.17, 55)
        lon2, lat2 = np.meshgrid(lon, lat)
        ridge = 140.0 * np.exp(-((lon2 + 74.902) / 0.008) ** 2 - ((lat2 - 39.08) / 0.020) ** 2)
        bathy = BathymetryGrid(lon=lon, lat=lat, depth=100.0 + ridge)
        narrow = build_size_field(
            bathy,
            nodes,
            SizeFieldConfig(
                land_spacing_m=700.0,
                open_spacing_m=2500.0,
                max_size_m=8000.0,
                coastal_distance_m=1500.0,
            ),
        )
        wide = build_size_field(
            bathy,
            nodes,
            SizeFieldConfig(
                land_spacing_m=700.0,
                open_spacing_m=2500.0,
                max_size_m=8000.0,
                coastal_distance_m=15_000.0,
            ),
        )
        center = (int(np.argmin(np.abs(lat - 39.08))), int(np.argmin(np.abs(lon + 74.902))))
        assert narrow.report["method"] == "unified_oceanmesh_coastal_lower_envelope"
        assert not bool(narrow.coastal_mask[center])
        assert np.isnan(narrow.slope_size[center])
        assert bool(wide.coastal_mask[center])
        assert np.isfinite(wide.slope_size[center])


def test_regional_spring_relaxation_preserves_boundary_and_improves_quality() -> None:
    points = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.35, 0.90]])
    triangles = np.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=int)
    fixed = np.asarray([True, True, True, True, False])
    before = triangle_geometry(points, triangles)
    result = relax_mesh_spring(
        points,
        triangles,
        fixed,
        constraint_chains=[[0, 1, 2, 3]],
        open_boundary_nodes_zero_based=np.asarray([0, 1]),
        config=SpringRelaxConfig(quality_threshold=0.95, min_angle_deg=55.0, iterations=30, shape_weight=0.35),
    )
    after = triangle_geometry(result.nodes_xy, triangles)
    assert result.report["accepted"] is True
    assert np.array_equal(result.nodes_xy[:4], points[:4])
    assert float(np.min(after["quality"])) > float(np.min(before["quality"]))
    assert result.report["final_energy"] < result.report["initial_energy"]
    assert np.all(after["signed_area"] > 0.0)


def test_area_transition_relaxation_is_eulerian_guarded_and_boundary_fixed() -> None:
    points = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.35, 0.90]])
    triangles = np.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=int)
    fixed = np.asarray([True, True, True, True, False])
    sampled_locations: list[np.ndarray] = []

    def target_sampler(locations: np.ndarray) -> np.ndarray:
        sampled_locations.append(np.asarray(locations, dtype=float).copy())
        return np.full(len(locations), 3.0, dtype=float)

    result = relax_mesh_area_transitions(
        points,
        triangles,
        fixed,
        target_spacing_sampler=target_sampler,
        constraint_chains=[[0, 1, 2, 3]],
        open_boundary_nodes_zero_based=np.asarray([0, 1]),
        config=AreaTransitionRelaxConfig(max_patches=4),
    )
    before = result.report["before"]
    after = result.report["after"]
    assert result.report["accepted"] is True
    assert result.report["applied_patch_count"] >= 1
    assert np.array_equal(result.nodes_xy[:4], points[:4])
    assert result.report["boundary_coordinate_max_shift_m"] == 0.0
    assert result.report["constraint_integrity"]["all_protected_edges_present"] is True
    assert after["maximum_adjacent_area_change"] < before["maximum_adjacent_area_change"]
    assert after["transition_severity_sum"] < before["transition_severity_sum"]
    assert after["adjacent_area_change_above_threshold_count"] <= before["adjacent_area_change_above_threshold_count"]
    assert after["l_over_h"]["count_above_threshold"] <= before["l_over_h"]["count_above_threshold"]
    assert result.report["maximum_total_displacement_over_h"] <= 0.25 + 1.0e-12
    assert result.report["target_sampling"]["call_count"] == len(sampled_locations)
    assert result.report["target_sampling"]["call_count"] >= 4
    assert np.allclose(result.target_spacing_m, 3.0)
    assert np.all(triangle_geometry(result.nodes_xy, triangles)["signed_area"] > 0.0)


def test_area_transition_high_gradient_trigger_requires_normalized_excess() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.6]])
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    fixed = np.ones(4, dtype=bool)

    def target_sampler(locations: np.ndarray) -> np.ndarray:
        return 1.5 - 0.75 * np.asarray(locations, dtype=float)[:, 0]

    result = relax_mesh_area_transitions(
        points,
        triangles,
        fixed,
        target_spacing_sampler=target_sampler,
        constraint_chains=[[0, 1, 2, 3]],
        open_boundary_nodes_zero_based=np.asarray([0, 1]),
        config=AreaTransitionRelaxConfig(max_patches=1),
    )
    before = result.report["before"]
    selected = result.report["patches"][0]["selected_pair"]
    assert result.report["accepted"] is False
    assert result.report["reason"] == "no_legal_transition_patch"
    assert before["raw_trigger_pair_count"] == 0
    assert before["high_gradient_trigger_pair_count"] == 1
    assert before["candidate_pair_count"] == 1
    assert selected["area_change"] > 0.375
    assert selected["area_change"] < 0.50
    assert selected["target_gradient"] > 0.10
    assert selected["normalized_log_jump"] > np.log(1.5)
    assert selected["high_gradient_trigger"] is True
    assert np.array_equal(result.nodes_xy, points)


def test_thin_triangle_edge_flip_preserves_boundary() -> None:
    points = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 0.3]])
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    fixed = np.ones(4, dtype=bool)
    result = repair_thin_triangles(
        points,
        triangles,
        fixed,
        [[0, 1, 2, 3]],
        np.asarray([0, 1]),
        target_spacing_m=np.full(4, 2.0),
        config=ThinTriangleRepairConfig(
            quality_threshold=0.20,
            min_angle_deg=8.0,
            max_passes=2,
            max_flips=4,
            max_insertions=0,
        ),
    )
    assert result.report["accepted"] is True
    assert result.report["edge_flip_count"] == 1
    assert np.array_equal(result.nodes_xy, points)
    assert result.report["constraint_integrity"]["all_protected_edges_present"] is True


def test_thin_triangle_long_edge_split_and_local_relaxation() -> None:
    points = np.asarray([[-2.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    triangles = np.asarray([[0, 1, 2], [1, 0, 3]], dtype=int)
    fixed = np.ones(4, dtype=bool)
    result = repair_thin_triangles(
        points,
        triangles,
        fixed,
        [[0, 2, 1, 3]],
        np.asarray([0, 2, 1]),
        target_spacing_m=np.full(4, 1.5),
        config=ThinTriangleRepairConfig(
            quality_threshold=0.60,
            min_angle_deg=20.0,
            max_passes=1,
            max_flips=0,
            max_insertions=2,
            split_target_factor=1.25,
            relaxation_config=SpringRelaxConfig(quality_threshold=0.75, min_angle_deg=30.0, iterations=10),
        ),
    )
    assert result.report["accepted"] is True
    assert result.report["edge_split_count"] == 1
    assert len(result.nodes_xy) == 5
    assert result.inserted_parent_edges == [(4, 0, 1)]
    assert np.array_equal(result.nodes_xy[:4], points)
    assert np.all(triangle_geometry(result.nodes_xy, result.triangles)["signed_area"] > 0.0)
    assert result.report["constraint_integrity"]["all_protected_edges_present"] is True


def test_fallback_bathy_prefers_depth_m() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lon = np.linspace(-75.1, -75.0, 4)
        lat = np.linspace(39.0, 39.1, 3)
        elevation = np.full((3, 4), -12.0)
        depth = np.full((3, 4), 7.5)
        ds = xr.Dataset(
            {
                "elevation_m": (("lat", "lon"), elevation),
                "depth_m": (("lat", "lon"), depth),
                "source_id": (("lat", "lon"), np.ones((3, 4), dtype=np.int16)),
            },
            coords={"lon": lon, "lat": lat},
        )
        path = root / "fallback_bathy.nc"
        ds.to_netcdf(path)
        bathy = load_bathymetry(path)
        assert bathy.metadata["depth_name"] == "depth_m"
        assert np.allclose(bathy.depth, 7.5)
        assert bathy.metadata["source_id_present"] is True


def test_elevation_m_only_is_positive_up() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = xr.Dataset(
            {"elevation_m": (("lat", "lon"), np.asarray([[-3.0, -5.0], [-8.0, -13.0]]))},
            coords={"lon": np.asarray([-75.1, -75.0]), "lat": np.asarray([39.0, 39.1])},
        )
        path = root / "elevation_only.nc"
        ds.to_netcdf(path)
        bathy = load_bathymetry(path)
        assert bathy.metadata["depth_name"] == "elevation_m"
        assert np.allclose(bathy.depth, np.asarray([[3.0, 5.0], [8.0, 13.0]]))


def test_size_field_bathy_coarsening_caps_cells() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bathy_path = write_synthetic_bathymetry(root / "large_bathy.nc", (-75.2, 38.8, -74.8, 39.2), nx=120, ny=100)
        bathy = load_bathymetry(bathy_path)
        coarsened = coarsen_for_size_field(bathy, max_cells=1_500)
        assert bathy.depth.size > 1_500
        assert coarsened.depth.size <= 1_700
        assert coarsened.metadata["coarsened_for_size_field"]["source_cell_count"] == bathy.depth.size


def test_generated_chain_uses_fallback_bathy_command() -> None:
    cmd = _bathy_fetch_command(
        Path("C:/fake/cudem-bathy"),
        [-75.2, 38.8, -74.7, 39.6],
        Path("run/bathy"),
        "case_bathy",
        GridConfig(),
        Path("run/bathy/bathy_source_index.json"),
    )
    text = " ".join(cmd)
    assert "fetch_bathy_sources.py" in text
    assert "fetch_cudem_bathy.py" not in text
    assert "cudem-nbs-crm-etopo" in cmd
    assert "1.0" in cmd
    assert _parse_required_source_count("641 sources intersect bbox, exceeding max_sources=256.") == 641


def test_channel_flownet_manifest_loader_and_command_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        loops = _synthetic_boundary_package(root / "loops.gpkg")
        package = load_boundary_package(loops)
        nodes = prepare_boundary_nodes(
            package,
            BoundaryConfig(
                land_spacing_m=700.0,
                open_spacing_m=2500.0,
                island_spacing_m=700.0,
            ),
        )
        mask_path = _write_analysis_mask(
            root / "analysis_mask.geojson",
            package.domain_polygon_lonlat,
            name="loader_test",
        )
        mask = gpd.read_file(mask_path)
        assert len(mask.geometry.iloc[0].interiors) == 1

        projected = np.asarray(nodes.xy, dtype=float)
        network_path = root / "topobathy_flownet.gpkg"
        geometry = MultiLineString(
            [
                LineString([projected[0], projected[1]]),
                LineString([projected[2], projected[3]]),
            ]
        )
        network_row = {
            "arcid": 1,
            "from_node": 1,
            "to_node": 2,
            "local": 1,
            "downarc": -1,
            "uparc": -1,
            "SELEV": 10.0,
            "EELEV": 5.0,
            "MAXGRID": 100,
            "dz": 5.0,
            "slope": 0.01,
            "meanmsq": 1_000_000.0,
            "segorder": 3,
            "drainage_area_m2": 1_000_000.0,
            "chanclass": 1,
            "hyddepth": 0.05,
            "hydwidth": 1.0,
            "effwidth": 0.2,
            "effdepth": -9999.0,
            "segdepth": -9999.0,
            "Shape_Leng": float(geometry.length),
            "geometry": geometry,
        }
        gpd.GeoDataFrame(
            [network_row],
            geometry="geometry",
            crs=nodes.projection.crs,
        ).to_file(network_path, layer="topobathy_flownet", driver="GPKG")
        health_path = root / "health_check.json"
        health_path.write_text(
            json.dumps(
                {
                    "schema": "topobathy_flownet_v1",
                    "status": "pass",
                    "failed_checks": [],
                }
            ),
            encoding="utf-8",
        )
        manifest_path = root / "run_manifest.json"
        manifest = {
            "schema": "topobathy_flownet_v1",
            "status": "complete",
            "structural_status": "pass",
            "topology_summary": {"arc_count": 1},
            "outputs": {
                "dhsvm_gpkg": str(network_path),
                "health_check": str(health_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        flowlines, report = _load_channel_flownet_manifest(
            manifest_path,
            nodes,
        )
        assert len(flowlines) == 2
        assert all(flowline.seg_order == 3 for flowline in flowlines)
        assert report["arc_count"] == 1
        assert report["flowline_count"] == 2
        assert report["order_counts"] == {"3": 1}
        assert report["flowline_order_counts"] == {"3": 2}
        assert len(report["manifest"]["sha256"]) == 64
        assert len(report["gpkg"]["sha256"]) == 64

        invalid = deepcopy(manifest)
        invalid["status"] = "failed"
        invalid_path = root / "invalid_manifest.json"
        invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
        try:
            _load_channel_flownet_manifest(invalid_path, nodes)
        except ValueError as exc:
            assert "status must be complete" in str(exc)
        else:
            raise AssertionError("A failed channel flownet manifest was accepted")

        surface = root / "surface.nc"
        xr.Dataset(
            {
                "elevation_m": (
                    ("lat", "lon"),
                    np.asarray([[-2.0, -3.0], [-4.0, -5.0]]),
                    {"positive": "up"},
                ),
                "depth_m": (
                    ("lat", "lon"),
                    np.asarray([[2.0, 3.0], [4.0, 5.0]]),
                    {"positive": "down"},
                ),
            },
            coords={"lon": [-75.1, -75.0], "lat": [39.0, 39.1]},
        ).to_netcdf(surface)
        bathy = load_bathymetry(surface)
        variable, positive = _select_channel_surface(surface, bathy)
        assert (variable, positive) == ("elevation_m", "up")
        command = _channel_flownet_command(
            Path("C:/fake/topobathy-flownet"),
            surface,
            mask_path,
            root / "products",
            variable,
            positive,
            GridConfig(
                channel_flownet_source_area_km2=1.25,
                channel_flownet_target_resolution_m=50.0,
            ),
        )
        assert "run_topobathy_flownet.py" in " ".join(command)
        assert command[command.index("--surface-variable") + 1] == "elevation_m"
        assert command[command.index("--surface-positive") + 1] == "up"
        assert command[command.index("--source-area-km2") + 1] == "1.25"
        assert command[command.index("--target-resolution-m") + 1] == "50.0"

        depth_only = write_synthetic_bathymetry(
            root / "depth_only.nc",
            (-75.1, 39.0, -75.0, 39.1),
            nx=4,
            ny=4,
        )
        depth_bathy = load_bathymetry(depth_only)
        assert _select_channel_surface(depth_only, depth_bathy) == (
            "depth",
            "down",
        )


def test_oceanmesh_metrics_and_true_neighbor_valence() -> None:
    height = np.sqrt(3.0) / 2.0
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, height]])
    triangles = np.asarray([[0, 1, 2]], dtype=int)
    geometry = triangle_geometry(points, triangles)
    assert np.isclose(geometry["quality"][0], 1.0)
    metrics = compute_mesh_metrics(points, triangles)
    assert np.isclose(metrics["oceanmesh_quality"]["q_mean"], 1.0)
    assert metrics["valence"]["max_node_valence"] == 2


def test_ordered_open_boundary_group_is_contiguous() -> None:
    chain = [0, 1, 2, 3, 4, 5]
    kinds = ["open", "open", "land", "land", "open", "open"]
    selected = _ordered_boundary_kind_group(chain, kinds, "open")
    assert selected == [4, 5, 0, 1]


def test_constraint_preserving_rpw2019_postprocess() -> None:
    points = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [0.35, 0.85],
        ]
    )
    triangles = np.asarray([[1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]], dtype=int)
    fixed = np.asarray([True, True, True, True, False])
    chains = [[0, 1, 2, 3]]
    original_boundary = points[:4].copy()
    result = postprocess_mesh(
        points,
        triangles,
        fixed,
        chains,
        np.asarray([1, 2]),
        PostprocessConfig(profile="rpw2019", boundary_policy="protect-all", connectivity_limit=6),
    )
    assert np.allclose(result.nodes_xy[np.asarray(result.constraint_chains[0])], original_boundary)
    assert result.report["all_boundary_coordinates_unchanged"] is True
    assert result.report["after"]["constraint_integrity"]["all_protected_edges_present"] is True
    assert result.report["after"]["constraint_integrity"]["open_boundary_ordered"] is True
    assert result.report["after"]["topology"]["nonpositive_signed_area_count"] == 0
    assert result.report["stage_order"] == [
        "fix_consistency",
        "make_boundaries_traversable",
        "repair_singly_connected",
        "bound_connectivity",
        "direct_implicit_smoothing",
    ]


def test_high_valence_cleanup_never_changes_ring_boundary() -> None:
    count = 10
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.sin(angles)])
    points = np.vstack([ring, [[0.05, -0.03]]])
    center = count
    triangles = np.asarray(
        [[index + 1, ((index + 1) % count) + 1, center + 1] for index in range(count)],
        dtype=int,
    )
    fixed = np.asarray([True] * count + [False])
    result = postprocess_mesh(
        points,
        triangles,
        fixed,
        [list(range(count))],
        np.asarray([1, 2]),
        PostprocessConfig(profile="rpw2019", boundary_policy="protect-all", connectivity_limit=6),
    )
    assert np.allclose(result.nodes_xy[np.asarray(result.constraint_chains[0])], ring)
    assert result.report["after"]["valence"]["max_node_valence"] <= 10
    assert result.report["after"]["constraint_integrity"]["all_protected_edges_present"] is True


def test_projection_medium_profile_order_and_guard() -> None:
    points = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.7, 0.9]])
    triangles = np.asarray([[1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]], dtype=int)
    result = postprocess_mesh(
        points,
        triangles,
        np.asarray([True, True, True, True, False]),
        [[0, 1, 2, 3]],
        np.asarray([1, 2]),
        PostprocessConfig(profile="projection-medium", boundary_policy="protect-all", max_passes=1),
    )
    assert result.report["profile_defaults"]["connectivity_limit"] == 8
    assert result.report["stage_order"] == [
        "fix_consistency",
        "pass_1_poor_boundary_repair",
        "pass_1_interior_thin_collapse",
        "pass_1_traversability",
        "pass_1_connectivity",
        "pass_1_smoothing",
    ]
    assert result.report["all_boundary_coordinates_unchanged"] is True


def test_stage_guard_rejects_quality_tail_regression() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    metrics = compute_mesh_metrics(points, np.asarray([[0, 1, 2]], dtype=int))
    metrics["constraint_integrity"]["open_boundary_ordered"] = True
    degraded = deepcopy(metrics)
    degraded["oceanmesh_quality"]["q_l3_sigma"] -= 0.01
    accepted, reason = _stage_acceptance(metrics, degraded, attempted=1, focus="quality_tail")
    assert accepted is False
    assert reason == "rollback_quality_tail_regression"


def test_2dm_writer_preserves_subnanodegree_orientation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "precision.2dm"
        nodes = np.asarray(
            [
                [-75.0, 39.0],
                [-74.99999999996, 39.0],
                [-75.0, 39.00000000004],
            ]
        )
        write_2dm(path, nodes, np.ones(3), np.asarray([[1, 2, 3]]), np.empty(0, dtype=int))
        serialized = read_2dm(path)
        signed_area = triangle_geometry(serialized.nodes_lonlat, serialized.triangles - 1)["signed_area"]
        assert signed_area[0] > 0.0


def test_aggressive_degree_three_pruning_is_target_guarded() -> None:
    points = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.6, 0.6]])
    triangles = np.asarray([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=int)
    result = condition_mesh_aggressive(
        points,
        triangles,
        np.asarray([True, True, True, False]),
        [[0, 1, 2]],
        np.asarray([0, 1]),
        target_spacing_m=np.full(4, 3.0),
        boundary_kinds=["land", "open", "land", "interior"],
        hard_anchor_mask=np.asarray([True, True, True, False]),
        config=AggressiveConditioningConfig(max_rounds=1, micro_relax_cycles=0),
    )
    assert result.report["edit_counts"]["degree-3-vertex-prune"] == 1
    assert len(result.nodes_xy) == 3
    assert len(result.triangles) == 1
    assert result.report["invariants"]["all_protected_edges_present"] is True


def test_aggressive_valence_repair_uses_distributed_steiner_nodes() -> None:
    count = 9
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    ring = np.column_stack((np.cos(angles), np.sin(angles)))
    points = np.vstack([ring, np.asarray([[0.0, 0.0]])])
    triangles = np.asarray([[count, index, (index + 1) % count] for index in range(count)], dtype=int)
    inventory = inventory_high_valence(
        points,
        triangles,
        fixed_node_mask=np.asarray([True] * count + [False]),
        boundary_kinds=["land"] * count + ["interior"],
        max_valence=8,
    )
    assert inventory["violation_count"] == 1
    assert inventory["records"][0]["repair_route_hint"] == "interior_cavity"
    result = condition_mesh_aggressive(
        points,
        triangles,
        np.asarray([True] * count + [False]),
        [list(range(count))],
        np.asarray([0, 1]),
        target_spacing_m=np.full(count + 1, 1.5),
        boundary_kinds=["land"] * count + ["interior"],
        hard_anchor_mask=np.asarray([True] * count + [False]),
        config=AggressiveConditioningConfig(max_rounds=2, max_prunes_per_round=0, micro_relax_cycles=0),
    )
    assert result.report["fvcom_valence_gate_passed"] is True
    assert result.report["after"]["maximum_valence"] <= 8
    assert result.report["edit_counts"]["high-valence-cavity-remove"] == 1
    assert count < len(result.nodes_xy) <= count + 8
    assert np.all(triangle_geometry(result.nodes_xy, result.triangles)["signed_area"] > 0.0)
    final_inventory = inventory_high_valence(result.nodes_xy, result.triangles, max_valence=8)
    assert final_inventory["violation_count"] == 0


def test_full_synthetic_workflow_and_2dm_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gpkg = _synthetic_boundary_package(root / "loops.gpkg")
        bathy = write_synthetic_bathymetry(root / "bathy.nc", (-75.11, 38.99, -74.89, 39.17), nx=45, ny=45)
        manifest = run_fvcom_grid(
            root / "run",
            "synthetic_grid",
            config=GridConfig(
                mode="test",
                land_spacing_m=900.0,
                open_spacing_m=3000.0,
                max_interior_points=800,
                refine_iterations=1,
                smooth_iterations=2,
                channel_flownet=False,
            ),
            boundary_loops_gpkg=gpkg,
            bathy_nc=bathy,
        )
        outputs = manifest["outputs"]
        assert manifest["schema_version"] == "fvcom_grid_generation_manifest_v7"
        assert Path(manifest["outputs"]["obc_remap_manifest_json"]).is_file()
        for key in (
            "fvcom_grid_2dm",
            "fvcom_grid_manifest",
            "mesh_quality_json",
            "mesh_conditioning_json",
            "mesh_edit_ledger_json",
            "mesh_review_map",
            "size_field_nc",
            "size_field_png",
            "node_budget_preflight_json",
            "boundary_nodes_geojson",
            "delivered_boundary_nodes_geojson",
            "mesh_nodes_elements_gpkg",
            "mesh_quality_elements_gpkg",
            "progress_json",
            "progress_jsonl",
        ):
            assert Path(outputs[key]).exists(), key
        assert manifest["inputs"]["upstream"]["source"] == "supplied_artifacts"
        assert manifest["bathymetry"]["loader_metadata"]["depth_name"] == "depth"
        mesh = read_2dm(outputs["fvcom_grid_2dm"])
        assert len(mesh.nodes_lonlat) == manifest["quality"]["roundtrip"]["node_count"]
        assert len(mesh.triangles) > 0
        assert len(mesh.open_boundary_nodes) >= 2
        assert np.all(mesh.depths > 0)
        assert manifest["mesh"]["constraint_recovery"]["boundary_constraint_recovered"] is True
        assert manifest["mesh"]["conditioning"]["stage_order"] == [
            "spring-relax-v1",
            "thin-repair-v1",
            "aggressive-local-disabled",
            "area-transition-relax-v1",
            "systematic-thin-terminal-disabled",
            "terminal-constraint-audit",
        ]
        assert manifest["mesh"]["conditioning"]["area_transition_relaxation"]["profile"] == "area-transition-relax-v1"
        assert manifest["postprocess"]["enabled"] is False
        assert manifest["settings"]["postprocess_profile"] == "none"
        assert manifest["settings"]["gradation"] == 0.20
        assert manifest["settings"]["coastal_distance_m"] == 12_000.0
        assert manifest["channel_flownet"]["source"] == "disabled"
        assert manifest["settings"]["area_transition_relaxation"] is True
        assert manifest["quality"]["constraint_integrity"]["all_protected_edges_present"] is True
        assert manifest["quality"]["roundtrip"]["triangle_connectivity_match"] is True
        assert manifest["quality"]["roundtrip"]["open_boundary_order_match"] is True
        assert manifest["quality"]["roundtrip"]["coordinate_within_tolerance"] is True
        roundtrip = manifest["quality"]["roundtrip"]
        assert roundtrip["positive_signed_areas"] is (roundtrip["nonpositive_signed_area_count"] == 0)
        if not roundtrip["positive_signed_areas"]:
            assert "2dm_roundtrip_nonpositive_area" in manifest["failure_taxonomy"]
            assert "2dm_roundtrip_failed" in manifest["failure_taxonomy"]
            assert manifest["final_status"] == "needs_review"
        delivered_boundary = json.loads(Path(outputs["delivered_boundary_nodes_geojson"]).read_text(encoding="utf-8"))
        delivered_ids = {int(feature["properties"]["node_id_1based"]) for feature in delivered_boundary["features"]}
        delivered_open_ids = {
            int(feature["properties"]["node_id_1based"])
            for feature in delivered_boundary["features"]
            if feature["properties"]["is_open_boundary"]
        }
        assert len(delivered_ids) == manifest["mesh"]["boundary_node_count"]
        assert delivered_open_ids == set(mesh.open_boundary_nodes.tolist())
        assert manifest["final_status"] in {"pass", "needs_review"}


def test_full_synthetic_systematic_v6_workflow() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gpkg = _simple_synthetic_boundary_package(root / "loops.gpkg")
        bathy = write_synthetic_bathymetry(
            root / "bathy.nc",
            (-75.11, 38.99, -74.89, 39.17),
            nx=35,
            ny=35,
        )
        manifest = run_fvcom_grid(
            root / "run",
            "synthetic_systematic_v6",
            config=GridConfig(
                mode="test",
                land_spacing_m=30000.0,
                open_spacing_m=30000.0,
                max_interior_points=100,
                refine_iterations=0,
                smooth_iterations=1,
                conditioning_profile="aggressive-local-v2",
                aggressive_conditioning_rounds=0,
                aggressive_max_prunes_per_round=0,
                aggressive_max_valence_repairs_per_round=25,
                area_transition_relaxation=False,
                thin_repair_profile="systematic-v6",
                systematic_v6_total_iterations=0,
                systematic_v6_max_cycles=0,
                systematic_v6_max_closure_rounds=0,
                systematic_v6_wall_time_s=30.0,
                systematic_v6_final_audit_reserve_s=0.0,
                systematic_v6_gate_policy="strict-v6",
                systematic_v6_passage_removal=False,
                channel_flownet=False,
            ),
            boundary_loops_gpkg=gpkg,
            bathy_nc=bathy,
        )
        conditioning = manifest["mesh"]["conditioning"]
        v6 = conditioning["terminal_systematic_thin_repair"]
        roundtrip = manifest["quality"]["roundtrip"]
        assert "systematic-v6-terminal" in conditioning["stage_order"]
        assert v6["profile"] == "systematic-v6"
        assert v6["settings"]["closure_gate_policy"] == "strict-v6"
        assert v6["settings"]["passage_removal_enabled"] is False
        assert not v6["settings"]["known_passage_node_ids_1based"]
        assert v6["after"]["restricted_edge_violation_count"] == 0
        assert v6["after"]["nonpositive_signed_area_count"] == 0, v6["after"]
        assert v6["after"]["nonmanifold_edge_count"] == 0
        assert (
            manifest["quality"]["constraint_integrity"][
                "all_protected_edges_present"
            ]
            is True
        )
        assert roundtrip["triangle_connectivity_match"] is True
        assert roundtrip["open_boundary_order_match"] is True
        assert roundtrip["coordinate_within_tolerance"] is True
        assert (
            manifest["settings"]["systematic_v6_gate_policy"]
            == "strict-v6"
        )
        assert (
            manifest["settings"]["systematic_v6_passage_removal"]
            is False
        )


def test_adaptive_resolution_workflow_and_quadtree_seed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        loops = _synthetic_boundary_package(root / "loops.gpkg")
        resolution = _synthetic_resolution_package(root, loops)
        package, nodes, document = load_boundary_resolution(resolution)
        assert nodes.adaptive_resolution is True
        assert document["profile"] == "adaptive-coastal-v1"
        bathy = write_synthetic_bathymetry(root / "bathy.nc", (-75.11, 38.99, -74.89, 39.17), nx=45, ny=45)
        manifest = run_fvcom_grid(
            root / "run",
            "synthetic_adaptive_grid",
            config=GridConfig(
                mode="test",
                land_spacing_m=700.0,
                open_spacing_m=2500.0,
                max_interior_points=5000,
                refine_iterations=1,
                smooth_iterations=1,
                boundary_resolution_profile="adaptive-coastal-v1",
                postprocess_profile="none",
                channel_flownet=False,
            ),
            boundary_loops_gpkg=loops,
            boundary_resolution_manifest=resolution,
            bathy_nc=bathy,
        )
        assert manifest["mesh"]["node_count"] > len(nodes.xy)
        assert manifest["size_field"]["method"] == "unified_oceanmesh_coastal_lower_envelope"
        assert manifest["size_field"]["background"]["method"] == "open_land_log_smoothstep"
        assert manifest["quality"]["open_boundary_size_error"]["p95_l_over_h"] <= 1.55
        assert manifest["mesh"]["constraint_recovery"]["final_recovery_applied"] is True
        assert manifest["quality"]["constraint_integrity"]["all_protected_edges_present"] is True
        assert manifest["postprocess"]["enabled"] is False
        assert read_2dm(manifest["outputs"]["fvcom_grid_2dm"]).triangles.size > 0


def test_final_quality_requires_q_l3_sigma_above_075() -> None:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    triangles = np.asarray([[1, 2, 3], [1, 3, 4]], dtype=int)
    common = {
        "depths": np.ones(4, dtype=float),
        "triangles_1based": triangles,
        "open_boundary_nodes": np.asarray([1, 2], dtype=int),
        "constraint_report": {"boundary_constraint_recovered": True},
        "constraint_chains": [[0, 1, 2, 3]],
    }
    regular = evaluate_mesh_quality(nodes, **common)
    assert regular["oceanmesh_quality"]["q_l3_sigma"] > 0.75
    assert "q_l3_sigma_below_threshold" not in regular["failure_taxonomy"]

    degraded = nodes.copy()
    degraded[2] = [1.0, 0.01]
    poor = evaluate_mesh_quality(degraded, **common)
    assert poor["oceanmesh_quality"]["q_l3_sigma"] <= 0.75
    assert "q_l3_sigma_below_threshold" in poor["failure_taxonomy"]


def main() -> int:
    test_boundary_ingestion_and_densification()
    test_size_field_limiter_never_coarsens_fine_cells()
    test_unified_oceanmesh_candidates_are_coastal_only()
    test_regional_spring_relaxation_preserves_boundary_and_improves_quality()
    test_area_transition_relaxation_is_eulerian_guarded_and_boundary_fixed()
    test_area_transition_high_gradient_trigger_requires_normalized_excess()
    test_thin_triangle_edge_flip_preserves_boundary()
    test_thin_triangle_long_edge_split_and_local_relaxation()
    test_fallback_bathy_prefers_depth_m()
    test_elevation_m_only_is_positive_up()
    test_size_field_bathy_coarsening_caps_cells()
    test_generated_chain_uses_fallback_bathy_command()
    test_channel_flownet_manifest_loader_and_command_contract()
    test_oceanmesh_metrics_and_true_neighbor_valence()
    test_ordered_open_boundary_group_is_contiguous()
    test_constraint_preserving_rpw2019_postprocess()
    test_high_valence_cleanup_never_changes_ring_boundary()
    test_projection_medium_profile_order_and_guard()
    test_stage_guard_rejects_quality_tail_regression()
    test_2dm_writer_preserves_subnanodegree_orientation()
    test_aggressive_degree_three_pruning_is_target_guarded()
    test_aggressive_valence_repair_uses_distributed_steiner_nodes()
    test_full_synthetic_workflow_and_2dm_roundtrip()
    test_full_synthetic_systematic_v6_workflow()
    test_adaptive_resolution_workflow_and_quadtree_seed()
    test_final_quality_requires_q_l3_sigma_above_075()
    print("fvcom-grid-generation selftests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
