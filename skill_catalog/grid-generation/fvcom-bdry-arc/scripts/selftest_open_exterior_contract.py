#!/usr/bin/env python3
"""Regression tests for independent open-exterior gates and policy defaults."""

from pathlib import Path
import copy
import json
import sys
import tempfile

import geopandas as gpd
from shapely.geometry import LineString, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_bdry_arc.open_exterior import evaluate_open_exterior_metrics
from fvcom_bdry_arc.workflow import BdryArcConfig
import finalize_open_exterior_decision as finalizer


def _stub_final_geometry(root: Path):
    candidate = root / "bdry_arc_package.gpkg"
    final = root / "bdry_arc_package_final.gpkg"
    loops = root / "model_boundary_loops.gpkg"
    loop_manifest = root / "model_boundary_loops_final" / "model_boundary_loop_manifest.json"
    loop_manifest.parent.mkdir(parents=True, exist_ok=True)
    for path in (candidate, final, loops):
        path.write_bytes(path.name.encode("utf-8"))
    loop_manifest.write_text('{}', encoding="utf-8")
    package = {
        "candidate_bdry_arc_package_gpkg": str(candidate),
        "bdry_arc_package_final_gpkg": str(final),
        "sha256": finalizer.sha256_file(final),
        "delivered_obc_count": 1,
        "chains": [{"obc_id": 0, "is_closed": False}],
        "solid_component_count": 0,
        "secondary_obc_count": 0,
    }
    loop = {
        "final_status": "pass",
        "failure_taxonomy": [],
        "qa": {"expected_obc_count": 1, "delivered_obc_count": 1},
        "outputs": {
            "model_boundary_loops_gpkg": str(loops),
            "model_boundary_loop_manifest": str(loop_manifest),
        },
    }
    return package, loop


def _run_with_stubbed_final_geometry(root: Path, callback):
    package, loop = _stub_final_geometry(root)
    old_package = finalizer._materialize_finalized_package
    old_loop = finalizer._rebuild_final_loops
    old_open_exterior = finalizer._rebuild_final_open_exterior
    finalizer._materialize_finalized_package = lambda *args, **kwargs: package
    finalizer._rebuild_final_loops = lambda *args, **kwargs: loop
    finalizer._rebuild_final_open_exterior = lambda *args, **kwargs: {
        "schema_version": "fvcom_open_exterior_contract_v2",
        "failure_taxonomy": ["open_exterior_agent_decision_required"],
        "hard_metrics": {"all_independent_metric_gates_pass": True},
        "obc_geometry": {"expected_count": 1, "delivered_count": 1},
        "outputs": {
            "open_exterior_contract": str(root / "open_exterior" / "open_exterior_contract.json"),
            "open_exterior_review_map": str(root / "open_exterior" / "open_exterior_review_map.png"),
        },
    }
    try:
        return callback()
    finally:
        finalizer._materialize_finalized_package = old_package
        finalizer._rebuild_final_loops = old_loop
        finalizer._rebuild_final_open_exterior = old_open_exterior


def _test_materialized_residual_roles() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        candidate = root / "bdry_arc_package.gpkg"
        wet = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
        gpd.GeoDataFrame(
            [{"geometry": wet}], geometry="geometry", crs="EPSG:4326"
        ).to_file(candidate, layer="wet_domain", driver="GPKG")
        gpd.GeoDataFrame(
            [{"segment_class": "open_boundary", "geometry": LineString([(4.0, 0.0), (4.0, 4.0)])}],
            geometry="geometry",
            crs="EPSG:4326",
        ).to_file(candidate, layer="open_boundary_arc", driver="GPKG")
        gpd.GeoDataFrame(
            [
                {"segment_id": 0, "geometry": LineString([(0.0, 4.0), (0.0, 3.0)])},
                {"segment_id": 1, "geometry": LineString([(0.0, 1.0), (0.0, 0.0)])},
            ],
            geometry="geometry",
            crs="EPSG:4326",
        ).to_file(candidate, layer="frame_clip_boundary_arcs", driver="GPKG")
        gpd.GeoDataFrame(
            [{"geometry": LineString([(0.0, 3.0), (0.0, 1.0)])}],
            geometry="geometry",
            crs="EPSG:4326",
        ).to_file(candidate, layer="land_boundary_arcs", driver="GPKG")
        candidate_hash = finalizer.sha256_file(candidate)
        contract = {
            "obc_geometry": {"expected_count": 2, "delivered_count": 2},
            "residual_components": [
                {
                    "segment_id": 0,
                    "assigned_role": "secondary_tidal_obc",
                    "geometry_lonlat": [[0.0, 4.0], [0.0, 3.0]],
                },
                {
                    "segment_id": 1,
                    "assigned_role": "solid_lagoon_closure",
                    "geometry_lonlat": [[0.0, 1.0], [0.0, 0.0]],
                },
            ],
        }
        package = finalizer._materialize_finalized_package(
            {"outputs": {"bdry_arc_package_gpkg": str(candidate)}},
            contract,
            root,
        )
        assert finalizer.sha256_file(candidate) == candidate_hash
        assert Path(package["candidate_bdry_arc_package_gpkg"]) == candidate
        final_path = Path(package["bdry_arc_package_final_gpkg"])
        assert final_path.is_file() and final_path != candidate
        final_open = gpd.read_file(final_path, layer="open_boundary_arc").sort_values("obc_id")
        assert list(final_open["obc_id"]) == [0, 1]
        assert list(final_open["residual_role"]) == ["primary_delivered_obc", "secondary_tidal_obc"]
        final_land = gpd.read_file(final_path, layer="land_patch_boundary_arcs")
        assert len(final_land) == 1
        assert final_land.iloc[0]["residual_role"] == "solid_lagoon_closure"
        assert "frame_clip_boundary_arcs" not in set(gpd.list_layers(final_path)["name"])


def _test_secondary_obc_requires_fresh_noaa_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assessed_hash = "synthetic-contract-sha256"
        base = {
            "obc_geometry": {"expected_count": 2, "delivered_count": 1},
            "boundary_lengths": {
                "outer_boundary_length_m": 10_000.0,
                "landward_boundary_length_m": 8_000.0,
            },
            "hard_metrics": {
                "absolute_limit_m": 250.0,
                "fraction_limit": 0.001,
                "coverage_minimum": 0.999,
                "coastline_source_coverage_gate_pass": True,
            },
            "residual_components": [
                {
                    "segment_id": 7,
                    "classification": "unintended_frame_clip",
                    "length_m": 500.0,
                    "solid_role_geometry": {"eligible": True},
                }
            ],
        }
        try:
            finalizer._apply_residual_roles(
                copy.deepcopy(base),
                assessed_contract_sha256=assessed_hash,
                decision="pass",
                requested_roles={7: "secondary_tidal_obc"},
                station_screen_path=None,
            )
        except ValueError as exc:
            assert "NOAA CO-OPS station" in str(exc)
        else:
            raise AssertionError("secondary OBC was accepted without NOAA station evidence")
        screen_path = root / "station_screen.json"
        screen_path.write_text(
            json.dumps(
                {
                    "schema_version": "noaa_coops_tidal_station_screen_v1",
                    "source_contract_sha256": assessed_hash,
                    "eligible_station_count": 1,
                    "components": [
                        {
                            "segment_id": 7,
                            "candidates": [
                                {
                                    "station_id": "9999999",
                                    "name": "Synthetic Connected Tide Station",
                                    "distance_km": 5.0,
                                    "eligible_for_residual_obc": True,
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        accepted = copy.deepcopy(base)
        assert finalizer._apply_residual_roles(
            accepted,
            assessed_contract_sha256=assessed_hash,
            decision="pass",
            requested_roles={7: "secondary_tidal_obc"},
            station_screen_path=screen_path,
        )
        assert accepted["obc_geometry"]["delivered_count"] == 2
        assert accepted["residual_components"][0]["assigned_role"] == "secondary_tidal_obc"
        assert accepted["residual_components"][0]["forcing_eligibility"]["provider"] == "NOAA CO-OPS"


def main() -> None:
    _test_materialized_residual_roles()
    _test_secondary_obc_requires_fresh_noaa_evidence()
    # Tampa-style northern/western bars fail every metric.
    tampa = evaluate_open_exterior_metrics(111_193.677, 903_200.0, 916_100.0, 250.0)
    assert not tampa["passed"]
    assert not tampa["length_gate"] and not tampa["fraction_gate"] and not tampa["coverage_gate"]

    # A small absolute residual cannot waive fractional/coverage gates.
    compact = evaluate_open_exterior_metrics(200.0, 10_000.0, 10_100.0, 250.0)
    assert compact["length_gate"]
    assert not compact["fraction_gate"]
    assert not compact["coverage_gate"]
    assert not compact["passed"]

    clean = evaluate_open_exterior_metrics(0.0, 100_000.0, 110_000.0, 250.0)
    assert clean["passed"]
    defaults = BdryArcConfig()
    assert defaults.frame_clip_policy == "reject-unintended"
    assert defaults.residual_boundary_policy == "solid-default"
    assert defaults.obc_placement_policy == "offshore-first"
    assert defaults.boundary_resolution_profile == "adaptive-coastal-v2"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        evidence = root / "open_exterior"
        evidence.mkdir()
        review_map = evidence / "open_exterior_review_map.png"
        review_map.write_bytes(b"map")
        decision_path = evidence / "open_exterior_agent_decision.json"
        decision_path.write_text("{}", encoding="utf-8")
        contract_path = evidence / "open_exterior_contract.json"
        contract_path.write_text(json.dumps({
            "schema_version": "fvcom_open_exterior_contract_v1",
            "report_only": False,
            "hard_metrics": {
                "absolute_gate_pass": True,
                "fraction_gate_pass": True,
                "coverage_gate_pass": True,
                "all_independent_metric_gates_pass": True,
            },
            "source_hashes": {"region": "abc"},
            "map": {"path": str(review_map)},
            "failure_taxonomy": ["open_exterior_agent_decision_required"],
        }), encoding="utf-8")
        manifest_path = root / "bdry_arc_manifest.json"
        manifest_path.write_text(json.dumps({
            "final_status": "needs_review",
            "failure_taxonomy": ["open_exterior_agent_decision_required"],
            "settings": {"boundary_resolution_profile": "adaptive-coastal-v2"},
            "outputs": {
                "open_exterior_contract": str(contract_path),
                "open_exterior_review_map": str(review_map),
                "open_exterior_agent_decision": str(decision_path),
            },
        }), encoding="utf-8")
        result = _run_with_stubbed_final_geometry(
            root,
            lambda: finalizer.finalize(
                manifest_path,
                "pass",
                "Inspected full map; no frame bar remains.",
                resume_adaptive=False,
            ),
        )
        assert result["open_exterior_contract"]["downstream_eligible"] is True
        assert result["final_status"] == "pass"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        evidence = root / "open_exterior"
        evidence.mkdir()
        review_map = evidence / "open_exterior_review_map.png"
        review_map.write_bytes(b"map")
        decision_path = evidence / "open_exterior_agent_decision.json"
        decision_path.write_text('{"status":"pending"}', encoding="utf-8")
        contract_path = evidence / "open_exterior_contract.json"
        contract_path.write_text(json.dumps({
            "schema_version": "fvcom_open_exterior_contract_v1",
            "report_only": False,
            "hard_metrics": {
                "absolute_gate_pass": True,
                "fraction_gate_pass": True,
                "coverage_gate_pass": True,
                "all_independent_metric_gates_pass": True,
            },
            "source_hashes": {},
            "failure_taxonomy": ["open_exterior_agent_decision_required"],
        }), encoding="utf-8")
        manifest_path = root / "bdry_arc_manifest.json"
        manifest_path.write_text(json.dumps({
            "name": "atomic",
            "final_status": "needs_review",
            "failure_taxonomy": ["open_exterior_agent_decision_required"],
            "settings": {"boundary_resolution_profile": "adaptive-coastal-v2"},
            "inputs": {"region_bpoly_json": "region", "coastline_gpkg": "coast"},
            "outputs": {
                "open_exterior_contract": str(contract_path),
                "open_exterior_review_map": str(review_map),
                "open_exterior_agent_decision": str(decision_path),
                "model_boundary_loops_gpkg": "loops",
                "model_boundary_loop_manifest": "loop_manifest",
            },
        }), encoding="utf-8")
        before = (contract_path.read_bytes(), decision_path.read_bytes(), manifest_path.read_bytes())
        original = finalizer.build_boundary_resolution
        finalizer.build_boundary_resolution = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic interruption"))
        try:
            try:
                _run_with_stubbed_final_geometry(
                    root,
                    lambda: finalizer.finalize(
                        manifest_path, "pass", "Inspected map.", resume_adaptive=True
                    ),
                )
            except RuntimeError as exc:
                assert "synthetic interruption" in str(exc)
            else:
                raise AssertionError("synthetic interruption was not propagated")
        finally:
            finalizer.build_boundary_resolution = original
        assert before == (contract_path.read_bytes(), decision_path.read_bytes(), manifest_path.read_bytes())

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        evidence = root / "open_exterior"
        evidence.mkdir()
        review_map = evidence / "open_exterior_review_map.png"
        component_map = evidence / "residual_component_000_role_map.png"
        review_map.write_bytes(b"whole map")
        component_map.write_bytes(b"component map")
        decision_path = evidence / "open_exterior_agent_decision.json"
        decision_path.write_text('{"status":"pending"}', encoding="utf-8")
        contract_path = evidence / "open_exterior_contract.json"
        contract_path.write_text(json.dumps({
            "schema_version": "fvcom_open_exterior_contract_v2",
            "report_only": False,
            "downstream_eligible": False,
            "hard_metrics": {
                "absolute_residual_length_m": 831.118,
                "absolute_limit_m": 250.0,
                "absolute_gate_pass": False,
                "residual_fraction": 0.0008,
                "fraction_limit": 0.001,
                "fraction_gate_pass": True,
                "coastline_plus_obc_exterior_coverage": 0.9992,
                "coverage_minimum": 0.999,
                "coverage_gate_pass": True,
                "all_independent_metric_gates_pass": False,
                "metric_subject": "unassigned_residual",
            },
            "raw_residual_metrics": {"absolute_residual_length_m": 831.118},
            "boundary_lengths": {"outer_boundary_length_m": 1_100_000.0, "landward_boundary_length_m": 1_000_000.0},
            "obc_geometry": {"expected_count": 1, "delivered_count": 1, "simple_nonbranching": True, "nonendpoint_land_crossing_m": 0.0},
            "residual_components": [{
                "segment_id": 0,
                "classification": "unintended_frame_clip",
                "length_m": 831.118,
                "role_status": "pending",
                "assigned_role": None,
                "solid_role_geometry": {"eligible": True},
            }],
            "component_maps": {"0": {"path": str(component_map), "sha256": finalizer.sha256_file(component_map)}},
            "source_hashes": {"region": "abc"},
            "map": {"path": str(review_map), "sha256": finalizer.sha256_file(review_map)},
            "failure_taxonomy": ["residual_boundary_role_decision_required", "open_exterior_agent_decision_required"],
        }), encoding="utf-8")
        manifest_path = root / "bdry_arc_manifest.json"
        manifest_path.write_text(json.dumps({
            "final_status": "needs_review",
            "failure_taxonomy": ["residual_boundary_role_decision_required", "blocked_by_open_exterior_contract"],
            "settings": {"boundary_resolution_profile": "adaptive-coastal-v2"},
            "outputs": {
                "open_exterior_contract": str(contract_path),
                "open_exterior_review_map": str(review_map),
                "open_exterior_agent_decision": str(decision_path),
            },
        }), encoding="utf-8")
        result = _run_with_stubbed_final_geometry(
            root,
            lambda: finalizer.finalize(
                manifest_path,
                "pass",
                "Inspected whole and component maps; the lagoon closure is simple and no artificial bar remains.",
                resume_adaptive=False,
            ),
        )
        resolved = result["open_exterior_contract"]
        assert resolved["schema_version"] == "fvcom_open_exterior_contract_v2"
        assert resolved["raw_residual_metrics"]["absolute_residual_length_m"] == 831.118
        assert resolved["hard_metrics"]["absolute_residual_length_m"] == 0.0
        assert resolved["residual_components"][0]["assigned_role"] == "solid_lagoon_closure"
        assert resolved["downstream_eligible"] is True

        pending_hash = finalizer.sha256_file(contract_path)
        station_screen = evidence / "station_screen.json"
        station_screen.write_text(json.dumps({
            "schema_version": "noaa_coops_tidal_station_screen_v1",
            "source_contract_sha256": pending_hash,
            "eligible_station_count": 1,
            "components": [{"segment_id": 0, "candidates": [{"station_id": "8570283", "name": "Ocean City Inlet", "distance_km": 11.5, "eligible_for_residual_obc": True}]}],
        }), encoding="utf-8")
        # The requested count is already occupied by the Atlantic OBC, so a
        # nearby station cannot silently create a second opening.
        try:
            _run_with_stubbed_final_geometry(
                root,
                lambda: finalizer.finalize(
                    manifest_path,
                    "pass",
                    "Inspected maps.",
                    resume_adaptive=False,
                    residual_roles={0: "secondary_tidal_obc"},
                    station_screen_path=station_screen,
                ),
            )
        except ValueError as exc:
            assert "Requested OBC count" in str(exc) or "stale" in str(exc)
        else:
            raise AssertionError("secondary OBC must respect the requested count")

    # v3 is historical evidence only and is rejected by active finalization.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        evidence = root / "open_exterior"
        evidence.mkdir()
        review_map = evidence / "open_exterior_review_map.png"
        review_map.write_bytes(b"map")
        decision_path = evidence / "open_exterior_agent_decision.json"
        decision_path.write_text('{"status":"pending"}', encoding="utf-8")
        contract_path = evidence / "open_exterior_contract.json"
        contract_path.write_text(json.dumps({
            "schema_version": "fvcom_open_exterior_contract_v3",
            "report_only": False,
            "hard_metrics": {},
            "source_hashes": {},
        }), encoding="utf-8")
        manifest_path = root / "bdry_arc_manifest.json"
        manifest_path.write_text(json.dumps({
            "final_status": "needs_review",
            "failure_taxonomy": [],
            "settings": {"boundary_resolution_profile": "adaptive-coastal-v2"},
            "outputs": {
                "open_exterior_contract": str(contract_path),
                "open_exterior_review_map": str(review_map),
                "open_exterior_agent_decision": str(decision_path),
            },
        }), encoding="utf-8")
        try:
            finalizer.finalize(manifest_path, "pass", "Inspected complete geometry.", resume_adaptive=False)
        except ValueError as exc:
            assert "unsupported open-exterior contract" in str(exc)
        else:
            raise AssertionError("historical v3 must be rejected by active finalization")
    print("passed strict and solid-default open-exterior policy tests")


if __name__ == "__main__":
    main()
