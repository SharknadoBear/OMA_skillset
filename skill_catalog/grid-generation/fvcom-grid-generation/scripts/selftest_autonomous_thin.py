#!/usr/bin/env python3
"""Focused offline tests for autonomous-thin-v1."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
from shapely.geometry import LineString, Point, Polygon
import geopandas as gpd

from fvcom_grid_generation.autonomous_thin import (
    DECISION_SCHEMA,
    ROUTES,
    boundary_transaction_audit,
    canonical_sha256,
    closure_acceptance,
    derive_cusp_buffer_m,
    interior_topology_plan,
    no_op_closure_report,
    rank_shoreline_candidates,
    regularize_shoreline,
    replace_cyclic_window,
    resolution_feasibility,
    sha256_file,
    shoreline_junction_turns_deg,
    validate_agent_decision,
)


def _decision(image: Path, route: str) -> dict:
    result = {
        "schema_version": DECISION_SCHEMA,
        "diagnostic_sha256": "a" * 64,
        "input_mesh_sha256": "b" * 64,
        "bound_input_hashes": {
            "mesh": "b" * 64,
            "cusp_gpkg": "c" * 64,
            "gshhs_gpkg": "d" * 64,
        },
        "component_id": "thin-1-test",
        "cycle_index_zero_based": 0,
        "decision_actor": {"kind": "codex_agent", "identifier": "offline-test"},
        "route": route,
        "observations": "Inspected the complete and local diagnostic maps.",
        "visual_evidence": [{"path": str(image), "sha256": sha256_file(image)}],
        "source_window": {
            "chain_index_zero_based": 0,
            "source_node_indices_zero_based": [1],
        },
        "protected_feature_check": {
            "obc_touched": False,
            "explicit_mission_feature_touched": False,
            "forcing_anchor_touched": False,
        },
    }
    if route == "resolved_channel_meshing_defect":
        result["resolution_evidence"] = {
            "width_m": 900.0,
            "local_target_m": 400.0,
            "bathymetry_floor_m": 200.0,
            "estimated_nodes_at_required_size": 300000,
        }
    elif route == "subgrid_wet_connection":
        result["resolution_evidence"] = {
            "width_m": 300.0,
            "local_target_m": 200.0,
            "bathymetry_floor_m": 150.0,
            "estimated_nodes_at_required_size": 1200000,
        }
    return result


def main() -> int:
    assert derive_cusp_buffer_m(100.0, 25.0) == 1000.0
    assert derive_cusp_buffer_m(700.0, 4000.0) == 5000.0
    feasible = resolution_feasibility(900.0, 400.0, 200.0, 300_000)
    assert feasible["resolvable"] and not feasible["resolved_at_current_target"]
    infeasible = resolution_feasibility(300.0, 200.0, 150.0, 1_200_000)
    assert not infeasible["resolvable"]
    major_channel = resolution_feasibility(1200.0, 300.0, 100.0, 100_000)
    assert major_channel["resolvable"] and major_channel["resolved_at_current_target"]

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        image = root / "evidence.png"
        image.write_bytes(b"synthetic visual evidence")
        for route in sorted(ROUTES):
            decision = _decision(image, route)
            if route not in {
                "resolved_channel_meshing_defect",
                "subgrid_boundary_spike_or_sliver",
                "subgrid_wet_connection",
            }:
                decision.pop("source_window")
            validate_agent_decision(
                decision,
                diagnostic_sha256="a" * 64,
                mesh_sha256="b" * 64,
                diagnostic_input_hashes=decision["bound_input_hashes"],
            )
        stale = _decision(image, "subgrid_boundary_spike_or_sliver")
        stale["diagnostic_sha256"] = "c" * 64
        try:
            validate_agent_decision(
                stale,
                diagnostic_sha256="a" * 64,
                mesh_sha256="b" * 64,
                diagnostic_input_hashes=stale["bound_input_hashes"],
            )
        except ValueError as exc:
            assert "stale" in str(exc)
        else:
            raise AssertionError("stale decision was accepted")
        human = _decision(image, "subgrid_boundary_spike_or_sliver")
        human["human_review_required"] = True
        try:
            validate_agent_decision(
                human,
                diagnostic_sha256="a" * 64,
                mesh_sha256="b" * 64,
                diagnostic_input_hashes=human["bound_input_hashes"],
            )
        except ValueError as exc:
            assert "human-review" in str(exc)
        else:
            raise AssertionError("human-gated decision was accepted")
        protected = _decision(image, "subgrid_boundary_spike_or_sliver")
        protected["protected_feature_check"] = {
            "obc_touched": False,
            "explicit_mission_feature_touched": True,
            "forcing_anchor_touched": False,
        }
        try:
            validate_agent_decision(
                protected,
                diagnostic_sha256="a" * 64,
                mesh_sha256="b" * 64,
                diagnostic_input_hashes=protected["bound_input_hashes"],
            )
        except ValueError as exc:
            assert "protected" in str(exc)
        else:
            raise AssertionError("protected boundary lineage was accepted")

    original = LineString([(0, 0), (50, 80), (100, 0)])
    candidates = rank_shoreline_candidates(
        [
            LineString([(-5, 0), (50, 10), (105, 0)]),
            LineString([(500, 500), (600, 600)]),
        ],
        (0, 0),
        (100, 0),
        original,
        local_target_m=100.0,
        horizontal_accuracy_m=[2.0, 2.0],
    )
    assert len(candidates) == 1 and candidates[0]["source_feature_index"] == 0
    dated_candidates = rank_shoreline_candidates(
        [
            LineString([(-5, 0), (50, 10), (105, 0)]),
            LineString([(-5, 0), (50, 10), (105, 0)]),
        ],
        (0, 0),
        (100, 0),
        original,
        local_target_m=100.0,
        horizontal_accuracy_m=[2.0, 2.0],
        source_dates=["2012-01-01", "2025-01-01"],
    )
    assert dated_candidates[0]["source_feature_index"] == 1
    regularized = regularize_shoreline(
        candidates[0]["geometry"], (0, 0), (100, 0), local_target_m=50.0
    )
    assert regularized.coords[0] == (0.0, 0.0)
    assert regularized.coords[-1] == (100.0, 0.0)
    smooth_turns = shoreline_junction_turns_deg(
        LineString([(0, 0), (50, 0), (100, 0)]), (1, 0), (1, 0)
    )
    reverse_turns = shoreline_junction_turns_deg(
        LineString([(0, 0), (-50, 0), (100, 0)]), (1, 0), (1, 0)
    )
    assert smooth_turns["maximum_turn_deg"] == 0.0
    assert reverse_turns["maximum_turn_deg"] == 180.0

    no_op = no_op_closure_report({
        "component_count": 0,
        "superthin_triangle_count": 0,
        "input_hashes": {"mesh": "e" * 64},
        "fvcom_ready": False,
        "fvcom_readiness_failure_taxonomy": ["forcing_incompatible"],
    })
    assert no_op["autonomous_thin_closed"] and no_op["minimal_local_debt_closed"]

    with tempfile.TemporaryDirectory() as temporary:
        interior_image = Path(temporary) / "interior.png"
        interior_image.write_bytes(b"synthetic interior evidence")
        interior_decision = _decision(interior_image, "interior_topology_defect")
        interior_diagnostic = {
            "input_hashes": {"mesh": "b" * 64},
            "components": [{"component_id": "thin-1-test"}],
        }
        local_plan = interior_topology_plan(interior_decision, interior_diagnostic)
        assert local_plan["schema_version"] == "fvcom_visual_superthin_repair_plan_v1"
        assert local_plan["acceptance"]["fixed_boundary_coordinates"]
        assert all(
            action["maximum_support_nodes"] == 0
            for action in local_plan["actions"]
        )

    # Generated shape anchors may be demoted inside a proven patch; protected
    # anchors and open nodes may not be removed by the same transaction.
    from run_autonomous_thin_closure import (
        _replace_frame_window,
        _retain_rejected_boundary_candidates,
    )
    frame = gpd.GeoDataFrame(
        {
            "node_index_zero_based": [0, 1, 2, 3, 4],
            "boundary_kind": ["land"] * 5,
            "is_hard_anchor": [False, True, False, False, False],
            "anchor_type": ["", "sharp_turn", "", "", ""],
            "anchor_id": ["", "generated", "", "", ""],
            "target_spacing_m": [100.0] * 5,
        },
        geometry=[Point(0, 0), Point(1, -1), Point(2, 0), Point(2, 2), Point(0, 2)],
        crs="EPSG:4326",
    )
    from fvcom_grid_generation.projection import local_utm_projection, project_points
    projection = local_utm_projection((0.0, 0.0, 2.0, 2.0))
    xy = project_points(np.asarray([[p.x, p.y] for p in frame.geometry]), projection)
    replacement = LineString([xy[0], xy[2]])
    changed, removed, blocked = _replace_frame_window(
        frame, 0, 2, replacement, projection,
        target_spacing_m=100.0, source_tag="offline-test",
    )
    assert removed == [1] and not blocked and "generated" not in changed["anchor_id"].tolist()
    protected_frame = frame.copy()
    protected_frame.loc[1, "anchor_type"] = "river_forcing_anchor"
    try:
        _replace_frame_window(
            protected_frame, 0, 2, replacement, projection,
            target_spacing_m=100.0, source_tag="offline-test",
        )
    except ValueError as exc:
        assert "nondemotable_hard_anchor" in str(exc)
    else:
        raise AssertionError("protected anchor was demoted")
    with tempfile.TemporaryDirectory() as temporary:
        retained_root = Path(temporary)
        source_manifest = retained_root / "source.json"
        source_manifest.write_text("{}\n", encoding="utf-8")
        retained = _retain_rejected_boundary_candidates(
            retained_root,
            route="subgrid_boundary_spike_or_sliver",
            candidate_records=[{"candidate": 1, "rejection_reason": "invalid_polygon"}],
            source_manifest_path=source_manifest,
        )
        retained_document = json.loads(retained.read_text(encoding="utf-8"))
        assert retained_document["status"] == "rejected"
        assert retained_document["candidate_ranking"][0]["rejection_reason"] == "invalid_polygon"

    coordinates = [[0, 0], [1, 0], [1, 1], [0, 1]]
    replaced, removed = replace_cyclic_window(
        coordinates, 0, 2, LineString([(0, 0), (0.5, 0.2), (1, 1)])
    )
    assert removed == [1] and len(replaced) == 4

    lobe = np.asarray([
        [0, 0], [4, 0], [4, 4], [3, 4], [3, 5],
        [2.9, 6], [2.8, 5], [2.8, 4], [0, 4],
    ], dtype=float)
    closed_lobe, removed_lobe = replace_cyclic_window(
        lobe, 3, 7, LineString([lobe[3], lobe[7]])
    )
    before_lobe = Polygon(lobe)
    after_lobe = Polygon(closed_lobe)
    assert removed_lobe == [4, 5, 6]
    assert after_lobe.is_valid and after_lobe.area < before_lobe.area
    assert after_lobe.geom_type == "Polygon"

    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    changed = Polygon([(0, 0), (10, 0), (10, 10), (1, 10)])
    audit = boundary_transaction_audit(
        square,
        changed,
        expected_hole_count=0,
        obc_before=LineString([(0, 0), (10, 0)]),
        obc_after=LineString([(0, 0), (10, 0)]),
        protected_points=[Point(0.25, 9.75)],
    )
    assert not audit["passed"] and "protected_feature_lost" in audit["failure_taxonomy"]

    accepted = closure_acceptance(
        {"superthin_triangle_count": 3},
        {
            "superthin_triangle_count": 0,
            "connected_component_count": 1,
            "singly_connected_triangle_count": 0,
            "nonmanifold_edge_count": 0,
            "nonpositive_area_count": 0,
            "open_boundary_chain_count": 1,
            "open_boundary_ordered": True,
            "forcing_compatible": True,
        },
        expected_open_boundary_count=1,
        roundtrip_passed=True,
    )
    assert accepted["autonomous_thin_closed"]
    delaware_rejection = closure_acceptance(
        {"superthin_triangle_count": 4},
        {
            "superthin_triangle_count": 0,
            "connected_component_count": 2,
            "singly_connected_triangle_count": 136,
            "nonmanifold_edge_count": 0,
            "nonpositive_area_count": 0,
            "open_boundary_chain_count": 1,
            "open_boundary_ordered": True,
            "forcing_compatible": True,
        },
        expected_open_boundary_count=1,
        roundtrip_passed=True,
    )
    assert not delaware_rejection["passed"]
    assert "multiple_mesh_components" in delaware_rejection["failure_taxonomy"]
    assert "singly_connected_elements_present" not in delaware_rejection["failure_taxonomy"]
    assert any(
        value["code"] == "singly_connected_elements_present"
        for value in delaware_rejection["regional_refinement_debt"]
    )

    first = canonical_sha256({"b": 2, "a": 1})
    second = canonical_sha256({"a": 1, "b": 2})
    assert first == second
    print(json.dumps({"status": "pass", "routes": sorted(ROUTES)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
