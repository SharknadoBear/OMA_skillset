from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile

import numpy as np
from shapely.geometry import box


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fvcom_grid_generation.boundary import load_boundary_resolution  # noqa: E402
from fvcom_grid_generation.boundary_topology import (  # noqa: E402
    SCHEMA_VERSION,
    expected_hole_count_matches,
    normalize_boundary_topology,
    write_boundary_topology_compensation,
)
from fvcom_grid_generation.gmsh_experiment import prepare_case  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection  # noqa: E402


def _ring(minx: float, miny: float, maxx: float, maxy: float) -> np.ndarray:
    return np.asarray(box(minx, miny, maxx, maxy).exterior.coords[:-1], dtype=float)


def _targets(ring: np.ndarray, value: float) -> np.ndarray:
    return np.full(len(ring), float(value), dtype=float)


def _normalize(islands: list[np.ndarray], spacing: list[float] | None = None):
    values = spacing or [2.0] * len(islands)
    return normalize_boundary_topology(
        _ring(0.0, 0.0, 100.0, 100.0),
        islands,
        [_targets(ring, value) for ring, value in zip(islands, values)],
        source_chain_ids=[f"chain_{index + 1}" for index in range(len(islands))],
        protected_exterior_contract={"obc_id": "obc_001", "coordinates": [[0.0, 0.0], [100.0, 0.0]]},
    )


def test_overlap_touch_and_subresolution_merge() -> None:
    for islands in (
        [_ring(10.0, 10.0, 20.0, 20.0), _ring(19.0, 12.0, 25.0, 18.0)],
        [_ring(10.0, 10.0, 20.0, 20.0), _ring(20.0, 12.0, 25.0, 18.0)],
        [_ring(10.0, 10.0, 20.0, 20.0), _ring(21.5, 12.0, 25.0, 18.0)],
    ):
        result = _normalize(islands)
        assert result.report["schema_version"] == SCHEMA_VERSION
        assert result.report["counts"]["source_island_chain_count"] == 2
        assert result.report["counts"]["delivered_island_chain_count"] == 1
        assert result.report["counts"]["merge_action_count"] == 1
        assert result.report["validity"]["exact_chain_hole_agreement"] is True
        assert result.report["invariants"]["obc_coordinates_order_ids_segmentation_hard_anchors_unchanged"] is True


def test_above_threshold_gap_stays_separate() -> None:
    result = _normalize(
        [_ring(10.0, 10.0, 20.0, 20.0), _ring(22.1, 12.0, 25.0, 18.0)]
    )
    assert result.report["changed"] is False
    assert result.report["counts"]["delivered_island_chain_count"] == 2


def test_pair_threshold_uses_finer_member_spacing() -> None:
    islands = [_ring(10.0, 10.0, 20.0, 20.0), _ring(23.0, 12.0, 25.0, 18.0)]
    result = _normalize(islands, spacing=[2.0, 8.0])
    assert result.report["changed"] is False
    result = _normalize(islands, spacing=[4.0, 8.0])
    assert result.report["counts"]["delivered_island_chain_count"] == 1
    action = result.report["actions"][0]
    assert action["eligible_pair_evidence"][0]["threshold_m"] == 4.0


def test_transitive_merge_and_lineage() -> None:
    result = _normalize(
        [
            _ring(10.0, 10.0, 20.0, 20.0),
            _ring(21.0, 12.0, 25.0, 18.0),
            _ring(26.0, 12.0, 30.0, 18.0),
        ],
        spacing=[2.0, 3.0, 4.0],
    )
    assert result.report["counts"]["delivered_island_chain_count"] == 1
    lineage = result.report["source_to_delivered_chains"][0]
    assert lineage["source_indices_one_based"] == [1, 2, 3]
    assert lineage["source_chain_ids"] == ["chain_1", "chain_2", "chain_3"]
    assert lineage["delivered_minimum_target_spacing_m"] == 2.0


def test_exterior_touch_and_cross_remove_complete_islands() -> None:
    result = _normalize(
        [
            _ring(0.0, 10.0, 5.0, 20.0),
            _ring(95.0, 10.0, 105.0, 20.0),
            _ring(40.0, 40.0, 45.0, 45.0),
        ]
    )
    assert result.report["counts"]["removed_source_chain_count"] == 2
    assert result.report["counts"]["delivered_island_chain_count"] == 1
    assert all(
        action["action_type"] == "remove_exterior_conflict"
        for action in result.report["actions"]
    )


def test_idempotence_and_expected_hole_policy() -> None:
    first = _normalize(
        [_ring(10.0, 10.0, 20.0, 20.0), _ring(21.0, 12.0, 25.0, 18.0)]
    )
    second = normalize_boundary_topology(
        first.exterior_xy,
        [value.xy for value in first.delivered_islands],
        [value.target_spacing_m for value in first.delivered_islands],
        source_chain_ids=["merged_delivery"],
    )
    assert second.report["changed"] is False
    assert second.report["counts"]["delivered_island_chain_count"] == 1
    assert expected_hole_count_matches(2, first) == (
        True,
        "matched_source_before_authorized_compensation",
    )
    assert expected_hole_count_matches(1, first) == (
        True,
        "matched_delivered_after_authorized_compensation",
    )
    assert expected_hole_count_matches(3, first)[0] is False
    with tempfile.TemporaryDirectory() as tmp:
        outputs = write_boundary_topology_compensation(
            tmp,
            first,
            local_utm_projection((-82.8, 27.5, -82.2, 28.1)),
        )
        assert set(outputs) == {"report_json", "overview_map", "zoom_map"}
        assert all(Path(value).is_file() and Path(value).stat().st_size > 0 for value in outputs.values())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archived_three_case_regression() -> None:
    workspace_candidates = [
        Path.cwd().resolve().parent / "Workspace" / "Preprocessing" / "fvcom-grid-generation" / "run",
        HERE.parents[4] / "Workspace" / "Preprocessing" / "fvcom-grid-generation" / "run",
    ]
    workspace = next((value for value in workspace_candidates if value.is_dir()), workspace_candidates[0])
    cases = [
        (
            workspace / "lfx1" / "tb" / "project" / "03_boundary" / "_work" / "s2" / "boundary_resolution" / "boundary_resolution_manifest.json",
            106,
            105,
            "merge_island_group",
            [14, 15],
        ),
        (
            workspace / "t6v3" / "gb" / "s2-boundary-arc-attempt2" / "run" / "boundary_resolution" / "boundary_resolution_manifest.json",
            29,
            28,
            "remove_exterior_conflict",
            [29],
        ),
        (
            workspace / "t6v3" / "hi" / "project" / "03_boundary" / "_work" / "s2_boundary_arc" / "boundary_resolution" / "boundary_resolution_manifest.json",
            25,
            24,
            "merge_island_group",
            [10, 16],
        ),
    ]
    for manifest, source_count, delivered_count, action_type, source_indices in cases:
        assert manifest.is_file(), manifest
        gpkg = manifest.with_name("boundary_resolution.gpkg")
        before = {manifest: _sha256(manifest), gpkg: _sha256(gpkg)}
        package, nodes, _payload = load_boundary_resolution(manifest)
        compensation = nodes.topology_compensation
        assert compensation is not None
        report = compensation.report
        assert report["counts"]["source_island_chain_count"] == source_count
        assert report["counts"]["delivered_island_chain_count"] == delivered_count
        matching = [
            action
            for action in report["actions"]
            if action["action_type"] == action_type
            and action["source_indices_one_based"] == source_indices
        ]
        assert matching, (manifest, report["actions"])
        assert len(nodes.constraint_chains) - 1 == delivered_count
        assert len(package.domain_polygon_lonlat.interiors) == delivered_count
        assert report["validity"]["wet_component_count"] == 1
        assert report["invariants"]["source_exterior_geometry_sha256"] == report["invariants"]["delivered_exterior_geometry_sha256"]
        assert before == {manifest: _sha256(manifest), gpkg: _sha256(gpkg)}

    galveston_case = workspace / "t6v3" / "gb" / "project" / "05_mesh_intent" / "case_manifest.json"
    assert galveston_case.is_file(), galveston_case
    prepared = prepare_case(galveston_case, galveston_case.parents[1])
    assert prepared.topology_compensation is not None
    assert len(prepared.holes_xy) == 28
    assert prepared.boundary_revalidation["expected_island_holes_match_mode"] == (
        "matched_source_before_authorized_compensation"
    )
    assert [
        (action["action_type"], action["source_indices_one_based"])
        for action in prepared.topology_compensation.report["actions"]
    ] == [("remove_exterior_conflict", [29])]


def main() -> int:
    test_overlap_touch_and_subresolution_merge()
    test_above_threshold_gap_stays_separate()
    test_pair_threshold_uses_finer_member_spacing()
    test_transitive_merge_and_lineage()
    test_exterior_touch_and_cross_remove_complete_islands()
    test_idempotence_and_expected_hole_policy()
    test_archived_three_case_regression()
    print("boundary topology compensation selftests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
