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
from shapely.geometry import LineString, Point, Polygon
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.bathymetry import coarsen_for_size_field, write_synthetic_bathymetry  # noqa: E402
from fvcom_grid_generation.boundary import BoundaryConfig, load_boundary_package, load_boundary_resolution, prepare_boundary_nodes  # noqa: E402
from fvcom_grid_generation.mesh import _ordered_boundary_kind_group  # noqa: E402
from fvcom_grid_generation.metrics import compute_mesh_metrics, triangle_geometry  # noqa: E402
from fvcom_grid_generation.postprocess import PostprocessConfig, _stage_acceptance, postprocess_mesh  # noqa: E402
from fvcom_grid_generation.size_field import SizeFieldConfig, build_size_field  # noqa: E402
from fvcom_grid_generation.bathymetry import load_bathymetry  # noqa: E402
from fvcom_grid_generation.sms_2dm import read_2dm  # noqa: E402
from fvcom_grid_generation.workflow import GridConfig, _bathy_fetch_command, _parse_required_source_count, run_fvcom_grid  # noqa: E402


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
            ),
            boundary_loops_gpkg=gpkg,
            bathy_nc=bathy,
        )
        outputs = manifest["outputs"]
        for key in (
            "fvcom_grid_2dm",
            "fvcom_grid_manifest",
            "mesh_quality_json",
            "mesh_review_map",
            "size_field_nc",
            "size_field_png",
            "boundary_nodes_geojson",
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
        assert manifest["postprocess"]["enabled"] is False
        assert manifest["settings"]["postprocess_profile"] == "none"
        assert manifest["quality"]["constraint_integrity"]["all_protected_edges_present"] is True
        assert manifest["final_status"] in {"pass", "needs_review"}


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
            ),
            boundary_loops_gpkg=loops,
            boundary_resolution_manifest=resolution,
            bathy_nc=bathy,
        )
        assert manifest["mesh"]["node_count"] > len(nodes.xy)
        assert manifest["size_field"]["adaptive_boundary"] is True
        assert manifest["quality"]["open_boundary_size_error"]["p95_l_over_h"] <= 1.55
        assert manifest["mesh"]["constraint_recovery"]["final_recovery_applied"] is True
        assert manifest["quality"]["constraint_integrity"]["all_protected_edges_present"] is True
        assert manifest["postprocess"]["enabled"] is False
        assert read_2dm(manifest["outputs"]["fvcom_grid_2dm"]).triangles.size > 0


def main() -> int:
    test_boundary_ingestion_and_densification()
    test_size_field_limiter_never_coarsens_fine_cells()
    test_fallback_bathy_prefers_depth_m()
    test_elevation_m_only_is_positive_up()
    test_size_field_bathy_coarsening_caps_cells()
    test_generated_chain_uses_fallback_bathy_command()
    test_oceanmesh_metrics_and_true_neighbor_valence()
    test_ordered_open_boundary_group_is_contiguous()
    test_constraint_preserving_rpw2019_postprocess()
    test_high_valence_cleanup_never_changes_ring_boundary()
    test_projection_medium_profile_order_and_guard()
    test_stage_guard_rejects_quality_tail_regression()
    test_full_synthetic_workflow_and_2dm_roundtrip()
    test_adaptive_resolution_workflow_and_quadtree_seed()
    print("fvcom-grid-generation selftests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
