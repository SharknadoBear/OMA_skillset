from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString
from unittest.mock import patch

import region_bbox.basemap as basemap_module
import region_bbox.discovery as discovery_module
from region_bbox.discovery import discover_named_region_features, extract_named_region_query
from region_bbox.features import infer_target_region_features
from region_bbox.basemap import _coastline_candidates, _draw_offline_coastline
from region_bbox.geometry import RegionBPoly
from region_bbox.io import file_sha256
from region_bbox.map_policy import resolve_basemap_provider
from region_bbox.scoring import score_bpoly_quality

ROOT = Path(__file__).resolve().parent


def run(cmd, expect_ok=True):
    p = subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, text=True, capture_output=True)
    if expect_ok and p.returncode != 0:
        raise AssertionError(f"command failed: {cmd}\nSTDOUT={p.stdout}\nSTDERR={p.stderr}")
    if not expect_ok and p.returncode == 0:
        raise AssertionError(f"command unexpectedly passed: {cmd}")
    return p


def load_raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load(path: Path) -> dict:
    doc = load_raw(path)
    request = doc.get("land_side_visual_review_request") if path.name == "region_bpoly.json" else None
    if (
        request
        and doc.get("domain_type") == "coastal"
        and doc.get("final_status") == "review_pending"
        and not doc.get("land_side_visual_review")
    ):
        cmd: list[object] = [
            "review_region_bpoly.py",
            "--candidate-json",
            path,
            "--problem-detected",
            "no",
            "--problem-description",
            "The initial synthetic candidate is scientifically suitable.",
            "--change-required",
            "no",
            "--geometry-changed",
            "no",
            "--scientifically-useful",
            "yes",
            "--scientific-rationale",
            "Required features, map context, and offshore orientation are suitable.",
            "--map-visibility-status",
            "pass",
            "--map-visibility-notes",
            "All hash-bound maps are readable in the regression fixture.",
        ]
        run(cmd)
        doc = load_raw(path)
    return doc


def review_cmd(
    path: Path,
    *,
    problem_detected: bool,
    scientifically_useful: bool,
    geometry_changed: bool = False,
    before_region_json: Path | None = None,
    no_meaningful_repair_remaining: bool = False,
    problem_class: str = "general_scientific",
) -> list[object]:
    cmd: list[object] = [
        "review_region_bpoly.py",
        "--candidate-json",
        path,
        "--problem-detected",
        "yes" if problem_detected else "no",
        "--problem-class",
        problem_class,
        "--problem-description",
        (
            "The mapped frame has a meaningful scientific coverage or placement problem."
            if problem_detected
            else "The mapped frame has no meaningful scientific coverage or placement problem."
        ),
        "--change-required",
        "yes" if problem_detected else "no",
        "--geometry-changed",
        "yes" if geometry_changed else "no",
        "--change-description",
        "The agent selected and applied a scientifically motivated geometry change." if geometry_changed else "The agent will choose a scientifically motivated repair.",
        "--scientifically-useful",
        "yes" if scientifically_useful else "no",
        "--scientific-rationale",
        "The final mapped frame is scientifically useful." if scientifically_useful else "The current mapped frame is not yet scientifically useful.",
        "--map-visibility-status",
        "pass",
        "--map-visibility-notes",
        "Whole-domain and side maps are readable.",
    ]
    if before_region_json is not None:
        cmd.extend(["--before-region-json", before_region_json])
    if no_meaningful_repair_remaining:
        cmd.append("--no-meaningful-repair-remaining")
    return cmd


def assert_feature_artifacts(candidate: dict, expected_zoom_count: int) -> None:
    feature_path = Path(candidate["target_region_features_path"])
    geojson_path = Path(candidate["target_region_feature_polygons_path"])
    assert feature_path.exists(), feature_path
    assert geojson_path.exists(), geojson_path
    features = load(feature_path)
    geojson = load(geojson_path)
    assert features["schema_version"] == "target_region_features_v1"
    assert features["features"], "feature decomposition should produce feature boxes"
    assert features["source_kind"] in {"catalog_memory", "explicit", "web_discovery", "agent_supplied_bbox"}
    assert features["geometry_status"] in {"heuristic_seed", "user_supplied", "discovered_seed", "inferred_seed"}
    assert all(feature.get("purpose") for feature in features["features"])
    assert all(feature.get("source_kind") == features["source_kind"] for feature in features["features"])
    assert len(geojson["features"]) == len(features["features"])
    feature_ids = {item["id"] for item in features["features"]}
    scored_ids = {item["id"] for item in candidate["ingredient_coverage"]["ingredients"]}
    assert feature_ids == scored_ids
    assert candidate["side_focus_count"] == expected_zoom_count, candidate["side_focus_reviews"]
    assert len(candidate["side_focus_reviews"]) == expected_zoom_count
    assert candidate["basemap"]["enabled"] is True
    assert candidate["basemap"]["required"] is True
    assert all(review["basemap"]["enabled"] is True and review["basemap"]["required"] is True for review in candidate["side_focus_reviews"])


def assert_standard_delivery(run_dir: Path, final: dict, expected_source_kind: str) -> None:
    canonical = {
        "region_bpoly": run_dir / "region_bpoly.json",
        "target_region_features": run_dir / "target_region_features.json",
        "final_map": run_dir / "region_bpoly_final_map.png",
        "offshore_boundary_artifacts": run_dir / "offshore_boundary_artifacts.json",
        "manifest": run_dir / "region_bpoly_manifest.json",
    }
    assert all(path.is_file() for path in canonical.values()), canonical
    feature_doc = load_raw(canonical["target_region_features"])
    assert feature_doc == final["target_region_features"]
    assert feature_doc["source_kind"] == expected_source_kind
    assert feature_doc["source_key"]
    assert all(feature.get("purpose") for feature in feature_doc["features"])
    assert all(feature.get("source_kind") == expected_source_kind for feature in feature_doc["features"])

    package = final["output_package"]
    assert package["schema_version"] == "region_bpoly_output_package_v1"
    assert package["package_complete"] is True
    assert package["delivery_ready"] is (final["final_status"] == "pass")
    assert package["canonical_files"]["target_region_features"] == "target_region_features.json"

    manifest = load_raw(canonical["manifest"])
    assert manifest["schema_version"] == "region_bpoly_delivery_manifest_v1"
    assert manifest["package_complete"] is True
    assert manifest["delivery_ready"] is (final["final_status"] == "pass")
    records = {record["role"]: record for record in manifest["files"]}
    for role, path in canonical.items():
        if role == "manifest":
            continue
        assert records[role]["present"] is True
        assert records[role]["sha256"] == file_sha256(path)


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        temp_root = Path(td)

        # A short-path launcher must not hide a coastline cache located above
        # the project output path.
        cache_root = temp_root / "workspace_root"
        cached_coast = cache_root / "Workspace" / "Preprocessing" / "fvcom-cusp-coastline" / "cache" / "gshhg" / "GSHHS_shp" / "f" / "GSHHS_f_L1.shp"
        cached_coast.parent.mkdir(parents=True)
        # The named GSHHS fixture has global bounds but no Delaware feature, so
        # it is valid water-only coverage evidence if online providers fail.
        gpd.GeoDataFrame(
            {"fixture": [1]},
            geometry=[LineString([(-180.0, -89.0), (180.0, 89.0)])],
            crs="EPSG:4326",
        ).to_file(cached_coast)
        nested_output = cache_root / "Workspace" / "Preprocessing" / "fvcom-grid-generation" / "runs" / "case" / "01_region"
        nested_output.mkdir(parents=True)
        assert cached_coast in _coastline_candidates([nested_output])
        run_dir = cache_root / "Workspace" / "Preprocessing" / "fvcom-grid-generation" / "runs" / "selftest"
        run_dir.mkdir(parents=True)

        # Antimeridian display windows must be split into native tile requests
        # and recomposed without stretching a dateline tile across the globe.
        aleutian_display_bbox = [-194.45, 46.9295, -151.55, 59.3705]
        longitude_segments = basemap_module._display_longitude_segments(aleutian_display_bbox)
        assert len(longitude_segments) == 2, longitude_segments
        assert longitude_segments[0]["native_bbox"][0] == 165.55, longitude_segments
        assert longitude_segments[0]["native_bbox"][2] == 180.0, longitude_segments
        assert longitude_segments[0]["display_shift_deg"] == -360.0, longitude_segments
        assert longitude_segments[1]["native_bbox"][0] == -180.0, longitude_segments
        assert longitude_segments[1]["native_bbox"][2] == -151.55, longitude_segments
        assert longitude_segments[1]["display_shift_deg"] == 0.0, longitude_segments

        synthetic_tiles = np.zeros((64, 64, 4), dtype=np.uint8)
        synthetic_tiles[:, :, 3] = 255
        _, safe_extent = basemap_module._warp_tiles_edge_safe(
            synthetic_tiles,
            (-20037508.342789244, -16280475.52851626, 5009377.08569731, 8766409.899970295),
        )
        assert safe_extent[1] - safe_extent[0] < 60.0, safe_extent
        assert -180.0001 <= safe_extent[0] <= -179.999, safe_extent

        online_calls = []

        def fake_bounds2img(west, south, east, north, **_kwargs):
            online_calls.append([west, south, east, north])
            pixels = np.zeros((8, 8, 4), dtype=np.uint8)
            pixels[:, :, 3] = 255
            return pixels, (west, east, south, north)

        fig, ax = plt.subplots()
        ax.set_xlim(aleutian_display_bbox[0], aleutian_display_bbox[2])
        ax.set_ylim(aleutian_display_bbox[1], aleutian_display_bbox[3])
        with patch("contextily.tile.bounds2img", side_effect=fake_bounds2img), patch.object(
            basemap_module,
            "_warp_tiles_edge_safe",
            side_effect=lambda image, extent, target_crs="EPSG:4326": (image, extent),
        ):
            import xyzservices.providers as xyz

            composed = basemap_module._add_online_tiles(ax, xyz.Esri.WorldTopoMap, None)
        assert online_calls == [segment["native_bbox"] for segment in longitude_segments], online_calls
        assert len(ax.images) == 2
        assert composed["antimeridian_composited"] is True, composed
        assert composed["longitude_segment_count"] == 2, composed
        assert composed["display_coverage_fraction"] == 1.0, composed
        assert composed["longitude_segments"][0]["displayed_extent"][0] == 165.55 - 360.0, composed
        assert composed["longitude_segments"][1]["displayed_extent"][1] == -151.55, composed
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.set_xlim(aleutian_display_bbox[0], aleutian_display_bbox[2])
        ax.set_ylim(aleutian_display_bbox[1], aleutian_display_bbox[3])
        with patch.object(basemap_module, "_coastline_candidates", return_value=[cached_coast]):
            offline_antimeridian = _draw_offline_coastline(ax, aleutian_display_bbox)
        plt.close(fig)
        assert offline_antimeridian is not None, offline_antimeridian
        assert offline_antimeridian["antimeridian_composited"] is True, offline_antimeridian
        assert offline_antimeridian["display_coverage_fraction"] == 1.0, offline_antimeridian
        assert len(offline_antimeridian["longitude_segments"]) == 2, offline_antimeridian

        # Regional maps fall through independent online providers before the
        # verified offline coastline. A transient Esri failure must select the
        # next provider rather than disabling geographic review.
        fig, ax = plt.subplots()
        ax.set_xlim(-75.8, -74.4)
        ax.set_ylim(38.4, 40.5)
        basemap_module._ONLINE_PROVIDER_FAILURES.clear()
        with patch.object(
            basemap_module,
            "_add_online_tiles",
            side_effect=[
                RuntimeError("synthetic Esri outage"),
                {
                    "effective_zoom": 8,
                    "requested_zoom": None,
                    "adjusted": False,
                    "tile_count": 4,
                    "max_tile_count": 64,
                },
            ],
        ):
            failover = basemap_module.add_basemap(
                ax,
                [-75.8, 38.4, -74.4, 40.5],
                provider="topo",
                search_roots=[nested_output],
            )
        plt.close(fig)
        assert failover["selected_provider"] == "OpenTopoMap", failover
        assert failover["provider_failures"][0]["provider"] == "Esri World Topographic Map", failover
        assert failover["geography_usable"] is True
        basemap_module._ONLINE_PROVIDER_FAILURES.clear()

        # A readable but geographically unrelated regional subset must not be
        # mislabeled as verified water-only background coverage.
        incomplete_coast = temp_root / "incomplete" / "GSHHS_f_L1.shp"
        incomplete_coast.parent.mkdir(parents=True)
        gpd.GeoDataFrame(
            {"fixture": [1]},
            geometry=[LineString([(-0.1, -0.1), (0.1, 0.1)])],
            crs="EPSG:4326",
        ).to_file(incomplete_coast)
        fig, ax = plt.subplots()
        ax.set_xlim(-75.8, -74.4)
        ax.set_ylim(38.4, 40.5)
        with patch.object(basemap_module, "_coastline_candidates", return_value=[incomplete_coast]):
            assert _draw_offline_coastline(ax, [-75.8, 38.4, -74.4, 40.5]) is None
        plt.close(fig)

        # A colored coordinate grid is diagnostic evidence, not a geographic
        # basemap and must block RegionBPoly acceptance.
        minimal_quality = score_bpoly_quality(
            RegionBPoly([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], 180.0),
            [],
            {"request": "generic coastal domain"},
            "coastal",
            "coastal_arc_with_land_anchors",
            {"side_index": 0, "side_name": "open_or_south"},
            {
                "enabled": True,
                "status": "fallback_minimal",
                "source": "minimal geographic background",
                "geography_usable": False,
                "display_frame": {"lon_span_deg": 2.0},
            },
        )
        minimal_codes = {item["code"] for item in minimal_quality["failure_taxonomy"]}
        assert "background_geography_unavailable" in minimal_codes, minimal_quality
        assert minimal_quality["blocking_failure"] is True

        # Fast review: one zoom per side.
        murderkill_features = infer_target_region_features({"request": "Murderkill River DE small estuary salinity intrusion"})
        provider, policy = resolve_basemap_provider({"request": "Murderkill River DE small estuary salinity intrusion"}, "auto", murderkill_features)
        assert provider == "road_detail", policy
        assert policy["domain_scale"] == "small_estuary"
        assert policy["target_zoom"] == 13
        assert policy["provider_chain"] == ["Esri.WorldStreetMap", "CartoDB.Voyager", "OpenStreetMap.Mapnik"]

        regional_provider, regional_policy = resolve_basemap_provider(
            {"request": "Delaware Bay regional coastal circulation"},
            "auto",
            {"domain_scale": "regional"},
        )
        assert regional_provider == "topo", regional_policy
        assert regional_policy["provider_chain"] == [
            "Esri.WorldTopoMap",
            "OpenTopoMap",
            "CartoDB.Voyager",
            "OpenStreetMap.Mapnik",
        ], regional_policy

        run(
            [
                "propose_region_bpoly.py",
                "--request-text",
                "Murderkill River DE small estuary salinity intrusion",
                "--run-dir",
                run_dir / "fast",
                "--name",
                "fast",
                "--basemap-provider",
                "none",
                "--review-depth",
                "fast",
            ]
        )
        fast = load(run_dir / "fast" / "fast_region_bpoly_candidate.json")
        assert fast["review_depth"] == "fast"
        assert fast["side_focus_mode"] == "fast_all_sides"
        assert_feature_artifacts(fast, 4)

        # Full review: start/middle/end on each side.
        run(
            [
                "propose_region_bpoly.py",
                "--request-text",
                "Long Island Sound hypoxia model",
                "--run-dir",
                run_dir / "full",
                "--name",
                "full",
                "--basemap-provider",
                "none",
                "--review-depth",
                "full",
            ]
        )
        full = load(run_dir / "full" / "full_region_bpoly_candidate.json")
        assert full["review_depth"] == "full"
        assert full["side_focus_mode"] == "full_all_sides"
        assert_feature_artifacts(full, 12)

        # Auto review escalates for known complex cases.
        auto_cases = {
            "puget": "Puget Sound / Salish Sea tidal energy assessment for all tidal channels",
            "aleut": "Aleutian Islands tidal energy and wave climate model",
            "seak": "South-east Alaska tidal energy assessment across island passages",
        }
        for name, text in auto_cases.items():
            out_dir = run_dir / name
            run(
                [
                    "run_region_bpoly.py",
                    "--request-text",
                    text,
                    "--run-dir",
                    out_dir,
                    "--name",
                    name,
                    "--mode",
                    "test",
                    "--heuristic-mode",
                    "memory",
                    "--basemap-provider",
                    "none",
                ]
            )
            final = load(out_dir / "region_bpoly.json")
            assert final["qa"]["review_depth"] == "full", final["qa"]
            expected_side_maps = 9 if final["domain_type"] == "coastal" else 12
            assert final["qa"]["side_focus_count"] == expected_side_maps, final["qa"]
            if name == "puget":
                assert final["domain_type"] == "coastal"
            if name == "seak":
                bbox = final["envelope_bbox"]
                assert final["domain_type"] == "coastal"
                assert bbox[0] > -150.0 and bbox[2] < -120.0, bbox
            if name == "aleut":
                assert final["region_bpoly"]["crosses_antimeridian"] is True
                assert final["qa"]["bpoly_quality"]["antimeridian_qa"]["map_display_lon_span_deg"] < 80.0
            assert (out_dir / "intermediate" / "visual_review").exists()
            assert (out_dir / "offshore_boundary_artifacts.json").exists()
            assert_standard_delivery(out_dir, final, "catalog_memory")

        # Antimeridian detection also escalates.
        aleut = RegionBPoly([[172.0, 48.9], [-162.0, 49.9], [-161.5, 57.6], [172.0, 56.7]], 172.0)
        assert aleut.crosses_antimeridian()
        assert aleut.contains_lonlat(179.0, 52.0)
        assert aleut.contains_lonlat(-170.0, 53.0)

        # Execute mode keeps final-only outputs after pass.
        exec_dir = run_dir / "execute_case"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Murderkill River DE small estuary salinity intrusion",
                "--run-dir",
                exec_dir,
                "--name",
                "execute_case",
                "--basemap-provider",
                "none",
            ]
        )
        provisional = load_raw(exec_dir / "region_bpoly.json")
        assert provisional["final_status"] == "review_pending"
        assert provisional["status_reasons"] == ["scientific_review_pending"]
        assert provisional["land_side_visual_review"] is None
        assert len(provisional["land_side_visual_review_request"]["required_land_side_indices"]) == 3
        assert provisional["scientific_review_request"]["iteration_policy"] == "agent_decided_no_numeric_limit"
        assert "No operation, side, or iteration count is prescribed" in provisional["scientific_review_request"]["repair_freedom"]
        final = load(exec_dir / "region_bpoly.json")
        assert final["mode"] == "execute"
        assert final["final_status"] == "pass"
        assert final["qa"]["review_depth"] == "full"
        assert final["qa"]["side_focus_count"] == 9
        assert final["final_map_basemap"]["enabled"] is True
        assert final["final_map_basemap"]["required"] is True
        assert (exec_dir / "region_bpoly_final_map.png").exists()
        assert (exec_dir / "offshore_boundary_artifacts.json").exists()
        assert not (exec_dir / "intermediate").exists()
        assert final["offshore_boundary_artifacts_path"].endswith("offshore_boundary_artifacts.json")
        assert final["qa"]["initial_guess_artifacts"]["retained"] is False
        assert_standard_delivery(exec_dir, final, "catalog_memory")
        downstream = final["downstream_contract"]
        assert "only for RegionBPoly-stage coastline-source planning" in downstream["bathymetry_and_coastline_fetch"]
        assert "final model_domain_polygon" in downstream["bathymetry_and_coastline_fetch"]
        assert "delivered offshore OBC may deform outside" in downstream["domain_and_grid_generation"]
        assert "authoritative for downstream grid generation" in downstream["domain_and_grid_generation"]
        tight = final["qa"]["bpoly_quality"]["tight_feature_fit"]
        assert tight["domain_scale"] == "small_estuary", tight
        assert tight["approx_width_km"] <= tight["small_estuary_limits"]["max_width_km"], tight
        assert tight["region_area_km2"] <= tight["small_estuary_limits"]["max_region_area_km2"], tight

        # Natural objective prompts expose a concise named-place query, and a
        # point-like geocoder result is expanded into an auditable initial
        # RegionBPoly seed rather than a terminal needs_review product.
        galveston_prompt = (
            "I am developing an FVCOM model of Galveston Bay to study estuarine "
            "circulation, salinity intrusion, and tidal mixing. Use FVCOM "
            "RegionBPoly to define an appropriate model region."
        )
        assert extract_named_region_query(galveston_prompt) == "Galveston Bay"
        compound_galveston_prompt = (
            "I am developing an FVCOM model of Galveston and Trinity Bays to simulate "
            "tidal circulation, salinity intrusion, and estuarine mixing."
        )
        assert extract_named_region_query(compound_galveston_prompt) == "Galveston and Trinity Bays"
        synthetic_result = {
            "display_name": "Galveston Bay, Chambers County, Texas, United States",
            "category": "natural",
            "type": "bay",
            "importance": 0.1,
            "boundingbox": ["29.5696234", "29.5697234", "-94.9366420", "-94.9365420"],
            "osm_type": "node",
            "osm_id": 1,
            "licence": "Data © OpenStreetMap contributors, ODbL 1.0",
        }
        with patch.object(discovery_module, "_nominatim_search", return_value=([synthetic_result], {"cache_hit": False, "cache_path": None})):
            discovered_features, discovery = discover_named_region_features(galveston_prompt, "coastal")
        discovered_bbox = discovered_features["features"][0]["geometry"]
        assert discovered_features["source"] == "online_named_place_discovery"
        assert discovered_features["source_kind"] == "web_discovery"
        assert discovered_features["geometry_status"] == "discovered_seed"
        assert discovered_features["features"][0]["purpose"] == "discovered_geographic_seed"
        assert discovery["selected_type"] == "bay"
        assert discovered_bbox[2] - discovered_bbox[0] > 1.0, discovered_bbox
        assert discovered_bbox[3] - discovered_bbox[1] > 1.0, discovered_bbox
        assert discovery["requires_visual_offshore_side_confirmation"] is True

        trinity_result = {
            **synthetic_result,
            "display_name": "Trinity Bay, Chambers County, Texas, United States",
            "type": "bay",
            "boundingbox": ["29.6823977", "29.6824977", "-94.7919155", "-94.7918155"],
            "osm_id": 2,
        }
        distant_trinity_result = {
            **synthetic_result,
            "display_name": "Trinity Bay, Newfoundland and Labrador, Canada",
            "type": "bay",
            "importance": 0.9,
            "boundingbox": ["47.5137014", "48.5471167", "-53.9864918", "-52.9423490"],
            "osm_id": 3,
        }
        compound_results = {
            "Galveston and Trinity Bays": [],
            "Galveston Bay": [synthetic_result],
            "Trinity Bay": [distant_trinity_result, trinity_result],
        }

        def compound_search(query, _timeout_s, _cache_dir=None):
            return compound_results[query], {"cache_hit": True, "cache_path": None}

        with patch.object(discovery_module, "_nominatim_search", side_effect=compound_search):
            compound_features, compound_discovery = discover_named_region_features(
                compound_galveston_prompt,
                "coastal",
            )
        assert compound_discovery["selection_method"] == "component_query_union_after_compound_no_bbox"
        assert compound_discovery["component_queries"] == ["Galveston Bay", "Trinity Bay"]
        assert len(compound_discovery["selected_components"]) == 2
        assert compound_discovery["selected_type"] == "multi_feature_extent"
        assert "Texas" in compound_discovery["selected_components"][1]["display_name"]
        assert compound_discovery["component_compactness_max_center_distance_km"] < 100.0
        compound_bbox = compound_features["features"][0]["geometry"]
        assert compound_bbox[2] - compound_bbox[0] > 1.0, compound_bbox
        assert compound_bbox[3] - compound_bbox[1] > 1.0, compound_bbox

        deep_cache = Path("C:/") / ("nested_case_" * 20) / "place-discovery"
        shortened_cache_path = discovery_module._cache_path(deep_cache, "compound waterbody query")
        assert shortened_cache_path.name.startswith("nom_")
        assert len(shortened_cache_path.name) < len("nominatim_" + "0" * 64 + ".json")

        galveston_dir = run_dir / "galveston_discovered_seed"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                galveston_prompt,
                "--run-dir",
                galveston_dir,
                "--name",
                "galveston_discovered_seed",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--discovery-bbox",
                -95.57,
                29.0,
                -94.28,
                30.15,
                "--discovery-label",
                "Galveston Bay, Texas, agent-supplied regression seed",
                "--offshore-azimuth-deg",
                180,
                "--basemap-provider",
                "none",
            ]
        )
        galveston = load(galveston_dir / "region_bpoly.json")
        assert galveston["final_status"] == "pass", galveston
        assert galveston["domain_type"] == "coastal", galveston
        assert galveston["target_region_features"]["source"] == "agent_supplied_place_discovery"
        assert galveston["place_discovery"]["requires_visual_scope_confirmation"] is True
        assert Path(galveston["place_discovery_path"]).exists()
        assert galveston["open_boundary_reference"]["side_index"] == 0, galveston["open_boundary_reference"]
        assert_standard_delivery(galveston_dir, galveston, "agent_supplied_bbox")

        # Unknown or memory-disabled prompts must never pass through the old
        # Delaware/NJ fallback box. With online discovery explicitly disabled,
        # the runner fails without publishing a terminal needs_review product.
        unknown_cases = {
            "pws_unknown": "Prince William Sound tide/current/ocean-exchange modeling domain",
            "galveston_unknown": "Galveston-Trinity Bay complex tide and salinity modeling domain",
            "albemarle_unknown": "Albemarle-Pamlico Sound complex tide and salinity modeling domain",
            "port_royal_unknown": "Port Royal Sound and St. Helena Sound estuarine complex",
            "murderkill_memory_off": "Murderkill River DE small estuary salinity intrusion",
        }
        for name, text in unknown_cases.items():
            out_dir = run_dir / name
            result = run(
                [
                    "run_region_bpoly.py",
                    "--request-text",
                    text,
                    "--run-dir",
                    out_dir,
                    "--name",
                    name,
                    "--mode",
                    "test",
                    "--basemap-provider",
                    "none",
                    "--place-discovery",
                    "off",
                ],
                expect_ok=False,
            )
            assert "region_discovery_failed" in result.stderr
            assert not (out_dir / "region_bpoly.json").exists()

        execute_unknown_dir = run_dir / "execute_unknown_no_fallback"
        execute_result = run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Uncatalogued coastal embayment with no explicit feature boxes",
                "--run-dir",
                execute_unknown_dir,
                "--name",
                "execute_unknown_no_fallback",
                "--mode",
                "execute",
                "--basemap-provider",
                "none",
                "--place-discovery",
                "off",
            ],
            expect_ok=False,
        )
        assert "region_discovery_failed" in execute_result.stderr
        assert not (execute_unknown_dir / "region_bpoly.json").exists()

        explicit_dir = run_dir / "explicit_feature_test_mode"
        explicit_request = {
            "request": "Unknown named estuary supplied with explicit feature boxes",
            "target_region_features": {
                "schema_version": "target_region_features_v1",
                "source": "selftest_explicit_features",
                "request_text": "Unknown named estuary supplied with explicit feature boxes",
                "domain_scale": "regional",
                "domain_variant": None,
                "considerations": {},
                "features": [
                    {
                        "id": "test_estuary_core",
                        "label": "Test estuary core",
                        "role": "target_estuary",
                        "category": "target_region",
                        "type": "bbox",
                        "geometry": [-95.2, 29.1, -94.6, 29.8],
                        "required": True,
                    },
                    {
                        "id": "test_offshore_gate",
                        "label": "Test offshore gate",
                        "role": "offshore_buffer",
                        "category": "offshore_extension",
                        "type": "bbox",
                        "geometry": [-95.4, 28.7, -94.2, 29.2],
                        "required": True,
                    },
                ],
            },
        }
        explicit_json = run_dir / "explicit_request.json"
        explicit_json.write_text(json.dumps(explicit_request, indent=2), encoding="utf-8")
        run(
            [
                "run_region_bpoly.py",
                "--request-json",
                explicit_json,
                "--run-dir",
                explicit_dir,
                "--name",
                "explicit_feature_test_mode",
                "--mode",
                "test",
                "--basemap-provider",
                "none",
            ]
        )
        explicit = load(explicit_dir / "region_bpoly.json")
        assert explicit["final_status"] == "pass", explicit
        assert explicit["heuristic_mode"] == "unknown"
        assert explicit["place_memory_enabled"] is False
        assert explicit["qa"]["ingredient_coverage"]["required_count"] == 2
        assert_standard_delivery(explicit_dir, explicit, "explicit")

        # Test mode keeps visual-review artifacts.
        test_dir = run_dir / "test_case"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Murderkill River DE small estuary salinity intrusion",
                "--run-dir",
                test_dir,
                "--name",
                "test_case",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
            ]
        )
        test_final = load(test_dir / "region_bpoly.json")
        assert test_final["mode"] == "test"
        visual_dir = test_dir / "intermediate" / "visual_review"
        assert visual_dir.exists()
        assert (visual_dir / "test_case_initial_guess_map.png").exists()
        assert (visual_dir / "test_case_initial_guess_region_bpoly.json").exists()
        assert (visual_dir / "target_region_features.json").exists()
        assert (visual_dir / "target_region_feature_polygons.geojson").exists()
        assert (visual_dir / "basemap_comparison" / "basemap_comparison_manifest.json").exists()
        offshore = load(test_dir / "offshore_boundary_artifacts.json")
        assert offshore["side_focus_count"] == 9
        assert all(z["retained"] for z in offshore["zoom_maps_used"])
        assert test_final["final_map_basemap"]["enabled"] is True
        assert test_final["qa"]["bpoly_quality"]["schema_version"] == "bpoly_quality_score_v1"
        assert test_final["map_detail_policy"]["resolved_provider"] == "road_detail"
        assert test_final["map_detail_policy"]["target_zoom"] == 13
        assert test_final["final_map_basemap"].get("requested_zoom", test_final["final_map_basemap"]["zoom"]) == 13
        assert test_final["final_map_basemap"]["zoom"] in {11, 12, 13}

        # Lake domains should not emit an ocean open-boundary reference.
        lake_dir = run_dir / "lake_case"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Lake Superior circulation and exchange model",
                "--run-dir",
                lake_dir,
                "--name",
                "lake_case",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--basemap-provider",
                "none",
            ]
        )
        lake = load(lake_dir / "region_bpoly.json")
        assert lake["domain_type"] == "lake"
        assert lake["boundary_policy"] == "no_open_boundary"
        assert lake["open_boundary_reference"] is None
        assert_standard_delivery(lake_dir, lake, "catalog_memory")

        # Cook Inlet wave-current domains need broad wave-fetch context.
        cook_dir = run_dir / "cook_case"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Cook Inlet wave-current model with Gulf of Alaska forcing",
                "--run-dir",
                cook_dir,
                "--name",
                "cook_case",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--basemap-provider",
                "none",
            ]
        )
        cook = load(cook_dir / "region_bpoly.json")
        assert cook["final_status"] == "pass"
        assert cook["domain_variant"] == "cook_inlet_wave_fetch"
        cook_feature_ids = {f["id"] for f in cook["target_region_features"]["features"]}
        for feature_id in {"cook_inlet_full", "ursus_cove_kamishak", "augustine_island", "kodiak_island_context", "cook_inlet_broad_wave_apron"}:
            assert feature_id in cook_feature_ids, cook_feature_ids
        assert cook["envelope_bbox"][1] < 56.0, cook["envelope_bbox"]
        assert cook["envelope_bbox"][2] <= -148.3, cook["envelope_bbox"]
        cook_wrong_region = cook["qa"]["bpoly_quality"]["wrong_region_inclusion"]["warnings"]
        assert not any("Prince William Sound" in warning for warning in cook_wrong_region), cook_wrong_region

        # Cook tidal-only domains keep the mouth-gate option and Kodiak guard.
        cook_tidal_dir = run_dir / "cook_tidal_case"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Cook Inlet tidal-only FVCOM model with offshore tidal forcing at the inlet mouth",
                "--run-dir",
                cook_tidal_dir,
                "--name",
                "cook_tidal_case",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--basemap-provider",
                "none",
            ]
        )
        cook_tidal = load(cook_tidal_dir / "region_bpoly.json")
        assert cook_tidal["final_status"] == "pass"
        assert cook_tidal["domain_variant"] == "cook_inlet_tidal_mouth"
        cook_guards = cook_tidal["qa"]["bpoly_quality"]["offshore_side_qa"]["obstruction_guards"]
        kodiak = [g for g in cook_guards if g["guard_id"] == "kodiak_island_obstruction"]
        assert kodiak and not kodiak[0]["blocks_final_pass"], kodiak

        # Mobile Bay needs an open-gate landing west of Horn Island without
        # unnecessary Perdido/Wolf Bay overreach.
        mobile_dir = run_dir / "mobile_case"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Mobile Bay tide and salinity model with Mobile-Tensaw delta and Gulf of Mexico open boundary",
                "--run-dir",
                mobile_dir,
                "--name",
                "mobile_case",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--basemap-provider",
                "none",
            ]
        )
        mobile = load(mobile_dir / "region_bpoly.json")
        assert mobile["final_status"] == "pass"
        assert mobile["domain_type"] == "coastal"
        assert mobile["envelope_bbox"][0] <= -88.85, mobile["envelope_bbox"]
        assert mobile["envelope_bbox"][2] <= -87.45, mobile["envelope_bbox"]
        mobile_feature_ids = {f["id"] for f in mobile["target_region_features"]["features"]}
        assert {"mobile_bay_core", "mobile_tensaw_delta", "mobile_gulf_gate", "horn_island_west_landing_context"}.issubset(mobile_feature_ids)
        mobile_taxonomy = [item["code"] for item in mobile["qa"]["bpoly_quality"]["failure_taxonomy"]]
        assert "open_gate_landing_blocked_by_horn_island" not in mobile_taxonomy
        assert "perdido_wolf_bay_overreach_risk" not in mobile_taxonomy

        # Hawaii Island-only scope should stay cleanly around the Big Island.
        hawaii_dir = run_dir / "hawaii_case"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Hawaii Island OTEC ocean model domain",
                "--run-dir",
                hawaii_dir,
                "--name",
                "hawaii_case",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--basemap-provider",
                "none",
            ]
        )
        hawaii = load(hawaii_dir / "region_bpoly.json")
        assert hawaii["domain_type"] == "island"
        assert hawaii["envelope_bbox"][3] < 20.75, hawaii["envelope_bbox"]
        assert hawaii["final_status"] == "pass"
        hawaii_guards = hawaii["qa"]["bpoly_quality"]["offshore_side_qa"]["obstruction_guards"]
        maui_nui = [g for g in hawaii_guards if g["guard_id"] == "maui_nui_neighbor_islands_obstruction"]
        assert maui_nui and not maui_nui[0]["blocks_final_pass"], maui_nui

        # Hawaii State / island-group scope may include the island chain.
        hawaii_state_dir = run_dir / "hawaii_state_case"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Hawaii State island group OTEC ocean model domain",
                "--run-dir",
                hawaii_state_dir,
                "--name",
                "hawaii_state_case",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--basemap-provider",
                "none",
            ]
        )
        hawaii_state = load(hawaii_state_dir / "region_bpoly.json")
        assert hawaii_state["domain_type"] == "island"
        assert any(f["id"] == "hawaiian_chain" for f in hawaii_state["target_region_features"]["features"])

        # Adjustment CLI: old polygon dashed, adjusted polygon solid on the map.
        manifest_path = run_dir / "adjust_manifest.json"
        manifest_path.write_text(json.dumps({"operation": "scale", "factor": 1.05}), encoding="utf-8")
        adjusted_json = run_dir / "adjusted_region_bpoly.json"
        adjustment_map = run_dir / "adjustment_map.png"
        run(
            [
                "adjust_region_bpoly.py",
                "--input-json",
                test_dir / "region_bpoly.json",
                "--adjustment-manifest",
                manifest_path,
                "--output-json",
                adjusted_json,
                "--map-path",
                adjustment_map,
                "--features-json",
                visual_dir / "target_region_features.json",
                "--basemap-provider",
                "none",
            ]
        )
        adjusted = load(adjusted_json)
        assert adjustment_map.exists()
        assert adjusted["adjustment_history"][0]["operation"] == "scale"
        assert adjusted["source_region_bpoly"]["polygon_lonlat"] != adjusted["adjusted_region_bpoly"]["polygon_lonlat"]
        assert adjusted["adjustment_map_basemap"]["enabled"] is True

        # The coastal review records only problem awareness, actual change, and
        # final scientific usefulness. It does not prescribe a side or method.
        repair_dir = run_dir / "land_side_repair"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Delaware River and Delaware Bay with one Atlantic-facing offshore side",
                "--run-dir",
                repair_dir,
                "--name",
                "land_side_repair",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--basemap-provider",
                "none",
            ]
        )
        repair_path = repair_dir / "region_bpoly.json"
        repair = load_raw(repair_path)
        required_sides = repair["land_side_visual_review_request"]["required_land_side_indices"]
        offshore_side = repair["land_side_visual_review_request"]["offshore_side_index"]
        run(
            review_cmd(
                repair_path,
                problem_detected=True,
                scientifically_useful=False,
            )
        )
        reviewed = load_raw(repair_path)
        assert reviewed["final_status"] == "repair_required"
        assert reviewed["output_package"]["delivery_ready"] is False
        assert reviewed["scientific_review"]["problem_detected"] is True
        assert reviewed["scientific_review"]["next_action"]["operation"] == "agent_selected_repair"
        assert "side_index" not in reviewed["scientific_review"]["next_action"]

        # Offshore orientation has exactly one internal repair opportunity.
        orientation_dir = run_dir / "offshore_orientation_i01"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Delaware River and Delaware Bay with one Atlantic-facing offshore side",
                "--run-dir",
                orientation_dir,
                "--name",
                "offshore_orientation_i01",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--basemap-provider",
                "none",
            ]
        )
        orientation_path = orientation_dir / "region_bpoly.json"
        run(
            review_cmd(
                orientation_path,
                problem_detected=True,
                scientifically_useful=False,
                problem_class="offshore_orientation",
            )
        )
        orientation_repair = load_raw(orientation_path)
        assert orientation_repair["final_status"] == "repair_required"
        assert orientation_repair["scientific_review"]["next_action"]["next_iteration"] == 2
        assert "one permitted internal offshore" in orientation_repair["scientific_review"]["next_action"]["constraints"]
        orientation_i02 = run_dir / "offshore_orientation_i02"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Delaware River and Delaware Bay with one Atlantic-facing offshore side",
                "--input-region-json",
                orientation_path,
                "--review-iteration",
                "2",
                "--run-dir",
                orientation_i02,
                "--name",
                "offshore_orientation_i02",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--basemap-provider",
                "none",
            ]
        )
        orientation_i02_path = orientation_i02 / "region_bpoly.json"
        run(
            review_cmd(
                orientation_i02_path,
                problem_detected=True,
                scientifically_useful=False,
                problem_class="offshore_orientation",
            )
        )
        orientation_blocked = load_raw(orientation_i02_path)
        assert orientation_blocked["final_status"] == "blocked"
        assert orientation_blocked["status_reasons"] == [
            "s1_offshore_orientation_unresolved_after_single_repair"
        ]
        assert orientation_blocked["output_package"]["delivery_ready"] is False

        # A repair cycle accepts multiple sides and arbitrary combinations of
        # rotate, contract/scale, reshape, and offshore-side movement.
        multi_manifest = run_dir / "multi_operation_repair.json"
        multi_manifest.write_text(
            json.dumps(
                {
                    "operations": [
                        {"operation": "rotate", "angle_deg": 1.0},
                        {"operation": "scale", "along_factor": 1.03, "across_factor": 0.98},
                        {"operation": "reshape", "vertex_delta_km": {"0": [-1.0, 1.0], "2": [1.0, -1.0]}},
                        {"operation": "expand_side", "side_index": required_sides[0], "distance_km": 2.0},
                        {
                            "operation": "expand_side",
                            "side_index": offshore_side,
                            "distance_km": 2.0,
                            "offshore_azimuth_deg": 95.0,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        repaired_json = run_dir / "agent_repaired_region.json"
        run(
            [
                "adjust_region_bpoly.py",
                "--input-json",
                repair_path,
                "--adjustment-manifest",
                multi_manifest,
                "--output-json",
                repaired_json,
                "--map-path",
                run_dir / "agent_repair_map.png",
                "--basemap-provider",
                "none",
                "--repair-cycle",
            ]
        )
        repaired = load_raw(repaired_json)
        assert [item["operation"] for item in repaired["adjustment_history"]] == [
            "rotate",
            "scale",
            "reshape",
            "expand_side",
            "expand_side",
        ]
        assert repaired["repair_cycle"] is True
        assert repaired["source_region_bpoly"]["polygon_lonlat"] != repaired["adjusted_region_bpoly"]["polygon_lonlat"]

        # The legacy flag remains a nonrestricting compatibility alias.
        rotate_manifest = run_dir / "rotate_legacy_loop.json"
        rotate_manifest.write_text(json.dumps({"operation": "rotate", "angle_deg": 2.0}), encoding="utf-8")
        run(
            [
                "adjust_region_bpoly.py",
                "--input-json",
                repair_path,
                "--adjustment-manifest",
                rotate_manifest,
                "--output-json",
                run_dir / "legacy_loop_rotate.json",
                "--map-path",
                run_dir / "legacy_loop_rotate.png",
                "--basemap-provider",
                "none",
                "--truncation-loop",
            ]
        )

        # There is no numeric repair-cycle cap.
        iteration_dir = run_dir / "agent_repair_iteration12"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Delaware River and Delaware Bay with one Atlantic-facing offshore side",
                "--input-region-json",
                repaired_json,
                "--review-iteration",
                "12",
                "--run-dir",
                iteration_dir,
                "--name",
                "agent_repair_iteration12",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--basemap-provider",
                "none",
            ]
        )
        iteration_path = iteration_dir / "region_bpoly.json"
        iteration_doc = load_raw(iteration_path)
        assert iteration_doc["scientific_review_request"]["iteration"] == 12
        assert iteration_doc["final_status"] == "review_pending"
        run(
            review_cmd(
                iteration_path,
                problem_detected=True,
                scientifically_useful=True,
                geometry_changed=True,
                before_region_json=repair_path,
            )
        )
        accepted = load_raw(iteration_path)
        assert accepted["final_status"] == "pass"
        assert accepted["scientific_review"]["geometry_change_verified"] is True
        assert accepted["scientific_review"]["effective_scientifically_useful"] is True
        assert accepted["qa"]["scientific_review"]["iteration"] == 12
        assert accepted["accepted_s1_geometry_freeze"]["downstream_geometry_repair_permitted"] is False
        assert accepted["accepted_s1_geometry_freeze"]["region_bpoly_sha256"] == accepted["scientific_review"]["region_bpoly_sha256"]
        assert (iteration_dir / "region_bpoly_land_side_review.json").exists()
        assert (iteration_dir / "region_bpoly_land_side_review.png").exists()

        # A later cycle may regenerate a fresh candidate without an adjustment
        # input, which is also how an agent can rebuild from a revised seed.
        regenerated_dir = run_dir / "regenerated_iteration21"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Delaware River and Delaware Bay with one Atlantic-facing offshore side",
                "--review-iteration",
                "21",
                "--run-dir",
                regenerated_dir,
                "--name",
                "regenerated_iteration21",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--basemap-provider",
                "none",
            ]
        )
        regenerated = load_raw(regenerated_dir / "region_bpoly.json")
        assert regenerated["scientific_review_request"]["iteration"] == 21
        assert regenerated["final_status"] == "review_pending"
        regenerated_path = regenerated_dir / "region_bpoly.json"
        run(
            review_cmd(
                regenerated_path,
                problem_detected=True,
                scientifically_useful=True,
            )
        )
        unverified = load_raw(regenerated_path)
        assert unverified["final_status"] == "pass"
        assert unverified["qa"]["scientific_review"]["status"] == "warning"
        assert any("verified geometry change" in item for item in unverified["scientific_review"]["validation_failures"])

        # Stale maps cannot earn a clean scientific pass, but the
        # latest valid geometry is still accepted with explicit warnings.
        stale_dir = run_dir / "stale_map_gate"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Delaware River and Delaware Bay",
                "--run-dir",
                stale_dir,
                "--name",
                "stale_map_gate",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--basemap-provider",
                "none",
            ]
        )
        stale_path = stale_dir / "region_bpoly.json"
        stale_doc = load_raw(stale_path)
        whole_map = Path(stale_doc["land_side_visual_review_request"]["whole_domain_map"]["map_path"])
        whole_map.write_bytes(whole_map.read_bytes() + b"stale")
        run(
            review_cmd(
                stale_path,
                problem_detected=False,
                scientifically_useful=True,
            )
        )
        stale_result = load_raw(stale_path)
        assert stale_result["final_status"] == "pass"
        assert stale_result["qa"]["scientific_review"]["status"] == "warning"
        assert stale_result["qa"]["scientific_review"]["scientifically_useful"] is False
        assert stale_result["output_package"]["package_state"] == "accepted_delivery"
        assert stale_result["output_package"]["delivery_ready"] is True

        # An honest impasse has no numeric trigger and never becomes needs_review.
        impasse_dir = run_dir / "agent_impasse"
        run(
            [
                "run_region_bpoly.py",
                "--request-text",
                "Delaware River and Delaware Bay",
                "--run-dir",
                impasse_dir,
                "--name",
                "agent_impasse",
                "--mode",
                "test",
                "--heuristic-mode",
                "memory",
                "--basemap-provider",
                "none",
            ]
        )
        impasse_path = impasse_dir / "region_bpoly.json"
        run(
            review_cmd(
                impasse_path,
                problem_detected=True,
                scientifically_useful=False,
                no_meaningful_repair_remaining=True,
            )
        )
        impasse = load_raw(impasse_path)
        assert impasse["final_status"] == "pass"
        assert impasse["status_reasons"] == ["scientific_review_accepted_best_effort"]
        assert impasse["scientific_review"]["effective_decision"] == "accepted_best_effort"
        assert impasse["scientific_review"]["effective_scientifically_useful"] is False
        assert "needs_review" not in json.dumps(impasse["status_reasons"])

        # The same free adjustment path is available to lake and island
        # candidates even though they do not require the coastal finalizer.
        for label, source_path, request_text, expected_type, cycle in (
            ("lake_refinement", lake_dir / "region_bpoly.json", "Lake Superior circulation and exchange model", "lake", 8),
            ("island_refinement", hawaii_state_dir / "region_bpoly.json", "Hawaii State island group OTEC ocean model domain", "island", 9),
        ):
            refined_json = run_dir / f"{label}_adjusted.json"
            run(
                [
                    "adjust_region_bpoly.py",
                    "--input-json",
                    source_path,
                    "--adjustment-manifest",
                    rotate_manifest,
                    "--output-json",
                    refined_json,
                    "--map-path",
                    run_dir / f"{label}_adjustment.png",
                    "--basemap-provider",
                    "none",
                    "--repair-cycle",
                ]
            )
            refined_dir = run_dir / label
            run(
                [
                    "run_region_bpoly.py",
                    "--request-text",
                    request_text,
                    "--input-region-json",
                    refined_json,
                    "--review-iteration",
                    str(cycle),
                    "--run-dir",
                    refined_dir,
                    "--name",
                    label,
                    "--mode",
                    "test",
                    "--heuristic-mode",
                    "memory",
                    "--basemap-provider",
                    "none",
                ]
            )
            refined = load_raw(refined_dir / "region_bpoly.json")
            assert refined["domain_type"] == expected_type
            assert refined["final_status"] == "pass"
            assert refined["polygon_lonlat"] != load_raw(source_path)["polygon_lonlat"]

        assert not (ROOT / "apply_arc_feedback.py").exists()

    print("fvcom-region-bpoly selftest passed")


if __name__ == "__main__":
    main()
