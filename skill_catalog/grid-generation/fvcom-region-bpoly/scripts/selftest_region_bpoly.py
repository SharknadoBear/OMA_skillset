from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from region_bbox.features import infer_target_region_features
from region_bbox.geometry import RegionBPoly
from region_bbox.map_policy import resolve_basemap_provider

ROOT = Path(__file__).resolve().parent


def run(cmd, expect_ok=True):
    p = subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, text=True, capture_output=True)
    if expect_ok and p.returncode != 0:
        raise AssertionError(f"command failed: {cmd}\nSTDOUT={p.stdout}\nSTDERR={p.stderr}")
    if not expect_ok and p.returncode == 0:
        raise AssertionError(f"command unexpectedly passed: {cmd}")
    return p


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_feature_artifacts(candidate: dict, expected_zoom_count: int) -> None:
    feature_path = Path(candidate["target_region_features_path"])
    geojson_path = Path(candidate["target_region_feature_polygons_path"])
    assert feature_path.exists(), feature_path
    assert geojson_path.exists(), geojson_path
    features = load(feature_path)
    geojson = load(geojson_path)
    assert features["schema_version"] == "target_region_features_v1"
    assert features["features"], "feature decomposition should produce feature boxes"
    assert len(geojson["features"]) == len(features["features"])
    feature_ids = {item["id"] for item in features["features"]}
    scored_ids = {item["id"] for item in candidate["ingredient_coverage"]["ingredients"]}
    assert feature_ids == scored_ids
    assert candidate["side_focus_count"] == expected_zoom_count, candidate["side_focus_reviews"]
    assert len(candidate["side_focus_reviews"]) == expected_zoom_count
    assert candidate["basemap"]["enabled"] is True
    assert candidate["basemap"]["required"] is True
    assert all(review["basemap"]["enabled"] is True and review["basemap"]["required"] is True for review in candidate["side_focus_reviews"])


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)

        # Fast review: one zoom per side.
        murderkill_features = infer_target_region_features({"request": "Murderkill River DE small estuary salinity intrusion"})
        provider, policy = resolve_basemap_provider({"request": "Murderkill River DE small estuary salinity intrusion"}, "auto", murderkill_features)
        assert provider == "road_detail", policy
        assert policy["domain_scale"] == "small_estuary"
        assert policy["target_zoom"] == 13
        assert policy["provider_chain"] == ["Esri.WorldStreetMap", "CartoDB.Voyager", "OpenStreetMap.Mapnik"]

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
            assert final["qa"]["side_focus_count"] == 12, final["qa"]
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
        final = load(exec_dir / "region_bpoly.json")
        assert final["mode"] == "execute"
        assert final["final_status"] == "pass"
        assert final["qa"]["review_depth"] == "fast"
        assert final["qa"]["side_focus_count"] == 4
        assert final["final_map_basemap"]["enabled"] is True
        assert final["final_map_basemap"]["required"] is True
        assert (exec_dir / "region_bpoly_final_map.png").exists()
        assert (exec_dir / "offshore_boundary_artifacts.json").exists()
        assert not (exec_dir / "intermediate").exists()
        assert final["offshore_boundary_artifacts_path"].endswith("offshore_boundary_artifacts.json")
        assert final["qa"]["initial_guess_artifacts"]["retained"] is False
        tight = final["qa"]["bpoly_quality"]["tight_feature_fit"]
        assert tight["domain_scale"] == "small_estuary", tight
        assert tight["approx_width_km"] <= tight["small_estuary_limits"]["max_width_km"], tight
        assert tight["region_area_km2"] <= tight["small_estuary_limits"]["max_region_area_km2"], tight

        # Unknown or memory-disabled prompts must not pass through the old
        # Delaware/NJ fallback box.
        unknown_cases = {
            "pws_unknown": "Prince William Sound tide/current/ocean-exchange modeling domain",
            "galveston_unknown": "Galveston-Trinity Bay complex tide and salinity modeling domain",
            "albemarle_unknown": "Albemarle-Pamlico Sound complex tide and salinity modeling domain",
            "port_royal_unknown": "Port Royal Sound and St. Helena Sound estuarine complex",
            "murderkill_memory_off": "Murderkill River DE small estuary salinity intrusion",
        }
        for name, text in unknown_cases.items():
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
                    "--basemap-provider",
                    "none",
                ]
            )
            unknown = load(out_dir / "region_bpoly.json")
            assert unknown["final_status"] == "needs_review", unknown
            assert unknown["domain_type"] == "unresolved_autonomous_failure", unknown
            assert unknown["envelope_bbox"] is None, unknown
            assert unknown["qa"]["bpoly_quality"]["canonical_region_key"] == "unknown"
            assert "unknown_region_no_feature_plan" in {
                item["code"] for item in unknown["qa"]["bpoly_quality"]["failure_taxonomy"]
            }

        execute_unknown_dir = run_dir / "execute_unknown_no_fallback"
        run(
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
            ]
        )
        execute_unknown = load(execute_unknown_dir / "region_bpoly.json")
        assert execute_unknown["final_status"] == "needs_review", execute_unknown
        assert execute_unknown["domain_type"] == "unresolved_autonomous_failure", execute_unknown
        assert execute_unknown["envelope_bbox"] is None, execute_unknown
        assert "unknown_region_no_feature_plan" in {
            item["code"] for item in execute_unknown["qa"]["bpoly_quality"]["failure_taxonomy"]
        }

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
        assert offshore["side_focus_count"] == 4
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

    print("fvcom-region-bpoly selftest passed")


if __name__ == "__main__":
    main()
