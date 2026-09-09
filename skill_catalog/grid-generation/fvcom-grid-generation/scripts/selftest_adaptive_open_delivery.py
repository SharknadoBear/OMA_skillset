#!/usr/bin/env python3
"""Regression tests for exact Adaptive-v2 OBC/exterior delivery."""

from __future__ import annotations

from pathlib import Path
import copy
import tempfile

import numpy as np
from shapely.geometry import LineString, Polygon

from fvcom_grid_generation.boundary_topology import BoundaryTopologyCompensation
from fvcom_grid_generation.gmsh_experiment import (
    SourceOpenBoundary,
    _delivered_open_boundary_membership_report,
    _validate_adaptive_boundary_evidence,
    _open_boundaries_from_adaptive_chains,
    _open_boundary_node_indices,
)


EXTERIOR = np.asarray(
    [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
    dtype=float,
)
KINDS = ("open", "open", "land", "land")
HARD = (0, 2)


def _manifest(*, prose: str = "") -> dict:
    return {
        "case_id": "synthetic_adaptive_delivery",
        "boundary": {
            "expected_open_boundary_count": 1,
            "expected_island_holes": 0,
            "open_boundaries": [
                {
                    "id": "ocean",
                    "kind": "ocean_exchange",
                    "cyclic": False,
                    "orientation": "source",
                }
            ],
            "required_revalidation": [prose] if prose else [],
            "build_policy": [prose] if prose else [],
        },
    }


def _resolution() -> dict:
    return {
        "qa": {"wet_component_count": 1, "resolved_domain_valid": True},
        "open_boundary_chains": [
            {
                "obc_id": 0,
                "is_closed": False,
                "node_count": 3,
                "hard_anchor_count": 2,
            }
        ],
    }


def _obc(indices: tuple[int, ...] = (0, 1), *, chain_id: str = "ocean") -> tuple[SourceOpenBoundary, ...]:
    return (
        SourceOpenBoundary(
            chain_id=chain_id,
            kind="ocean_exchange",
            cyclic=False,
            orientation="source",
            exterior_segment_indices=indices,
        ),
    )


def _report(
    *,
    manifest: dict | None = None,
    kinds: tuple[str, ...] = KINDS,
    hard: tuple[int, ...] = HARD,
    obc: tuple[SourceOpenBoundary, ...] | None = None,
) -> dict:
    return _delivered_open_boundary_membership_report(
        manifest or _manifest(),
        EXTERIOR,
        kinds,
        hard,
        obc or _obc(),
        _resolution(),
    )


def test_exact_membership_passes() -> None:
    report = _report()
    assert report["passed"] is True
    assert report["complete_open_segment_coverage"] is True
    assert report["segments_covered_exactly_once"] is True


def test_free_text_and_low_proxy_cannot_activate_gate() -> None:
    manifest = _manifest(
        prose="open-boundary exterior overlap is at least 0.98 and must be revalidated"
    )
    polygon = Polygon(EXTERIOR)
    compensation = BoundaryTopologyCompensation(
        exterior_xy=EXTERIOR.copy(),
        source_islands_xy=(),
        delivered_islands=(),
        wet_domain_xy=polygon,
        report={
            "counts": {
                "source_island_chain_count": 0,
                "delivered_island_chain_count": 0,
            }
        },
    )
    with tempfile.TemporaryDirectory(prefix="adaptive_open_delivery_") as temporary:
        source_manifest = Path(temporary) / "boundary_resolution_manifest.json"
        source_manifest.write_text("{}\n", encoding="utf-8")
        report, _ = _validate_adaptive_boundary_evidence(
            manifest,
            Path(temporary),
            source_manifest,
            _resolution(),
            polygon,
            LineString([[100.0, 100.0], [110.0, 100.0]]),
            (),
            EXTERIOR,
            KINDS,
            HARD,
            _obc(),
            compensation,
        )
    assert report["passed"] is True
    assert report["independent_open_boundary_exterior_overlap_fraction"] == 0.0
    assert report["independent_open_boundary_exterior_overlap_role"].startswith(
        "diagnostic_only"
    )


def test_land_segment_is_rejected() -> None:
    report = _report(obc=_obc((0, 2)))
    assert report["passed"] is False
    assert any("land_segment_in_open_boundary" in value for value in report["failure_taxonomy"])


def test_reordered_segments_are_rejected() -> None:
    report = _report(obc=_obc((1, 0)))
    assert report["passed"] is False
    assert any("not_contiguous" in value for value in report["failure_taxonomy"])


def test_duplicate_segments_are_rejected() -> None:
    report = _report(obc=_obc((0, 0, 1)))
    assert report["passed"] is False
    assert any("duplicate_segment_index" in value for value in report["failure_taxonomy"])


def test_incomplete_open_segment_coverage_is_rejected() -> None:
    report = _report(obc=_obc((0,)))
    assert report["passed"] is False
    assert "delivered_open_boundary_segments_do_not_exactly_cover_open_exterior" in report[
        "failure_taxonomy"
    ]


def test_wrong_id_is_rejected() -> None:
    report = _report(obc=_obc(chain_id="wrong"))
    assert report["passed"] is False
    assert "delivered_open_boundary_ids_do_not_match_manifest" in report["failure_taxonomy"]


def test_hard_anchor_signature_is_rejected() -> None:
    report = _report(hard=(0,))
    assert report["passed"] is False
    assert "adaptive_v2_delivered_obc_signature_mismatch" in report["failure_taxonomy"]


def _source_fixture():
    manifest = {"case_id": "arbitrary_source_ids", "boundary": {
        "expected_open_boundary_count": 2,
        "open_boundaries": [{"id": "1", "source_obc_id": 1, "cyclic": False},
                            {"id": "0", "source_obc_id": 0, "cyclic": False}],
    }}
    resolution = {"open_boundary_chains": [
        {"obc_id": 0, "is_closed": False, "node_count": 3, "hard_anchor_count": 2,
         "node_sequence_zero_based": [100, 101, 102]},
        {"obc_id": 1, "is_closed": False, "node_count": 3, "hard_anchor_count": 2,
         "node_sequence_zero_based": [104, 105, 106]},
    ]}
    return manifest, resolution, list(range(100, 108)), ("open", "open", "land", "land", "open", "open", "land", "land")


def test_source_ids_and_declared_order_are_exact() -> None:
    manifest, resolution, ids, kinds = _source_fixture()
    result = _open_boundaries_from_adaptive_chains(manifest, resolution, ids, kinds)
    assert [x.chain_id for x in result] == ["1", "0"]
    assert [x.exterior_segment_indices for x in result] == [(4, 5), (0, 1)]
    assert [_open_boundary_node_indices(x, 8) for x in result] == [[4, 5, 6], [0, 1, 2]]
    # Labels deliberately contradict compass-based legacy naming.
    manifest["boundary"]["open_boundaries"][0]["id"] = "river_west"
    named = _open_boundaries_from_adaptive_chains(manifest, resolution, ids, kinds)
    assert named[0].exterior_segment_indices == (4, 5)
    assert resolution["open_boundary_chains"][0]["node_sequence_zero_based"] == [100, 101, 102]


def test_source_reversed_wrapped_and_cyclic_order() -> None:
    for nodes, cyclic, expected_indices, expected_orientation in [
        ([2, 1, 0], False, (1, 0), "reverse"),
        ([7, 0, 1], False, (7, 0), "source"),
        ([1, 0, 7], False, (0, 7), "reverse"),
        ([2, 3, 0, 1], True, (2, 3, 0, 1), "source"),
        ([2, 1, 0, 3], True, (1, 0, 3, 2), "reverse"),
    ]:
        count = 4 if cyclic else 8
        manifest = {"case_id": "order", "boundary": {"expected_open_boundary_count": 1,
            "open_boundaries": [{"id": "ocean", "cyclic": cyclic, "orientation": "source"}]}}
        source = {"open_boundary_chains": [{"obc_id": 5, "is_closed": cyclic,
            "node_count": len(nodes), "node_sequence_zero_based": nodes,
            "source_direction_reversed_for_exterior": expected_orientation == "reverse",
            "hard_anchor_count": 2}]}
        kinds = tuple("open" if i in expected_indices else "land" for i in range(count))
        chains = _open_boundaries_from_adaptive_chains(manifest, source, list(range(count)), kinds)
        assert chains[0].exterior_segment_indices == expected_indices
        assert chains[0].orientation == expected_orientation
        traversal = _open_boundary_node_indices(chains[0], count)
        assert traversal == nodes + (nodes[:1] if cyclic else [])
        # The downstream Gmsh adapter and membership gate must honor that direction.
        from fvcom_grid_generation.gmsh_backend import OpenBoundaryGeometry, _validate_chain_contiguity
        backend = OpenBoundaryGeometry(chain_id="ocean", kind="ocean_exchange", cyclic=cyclic,
            orientation=chains[0].orientation, segment_indices=chains[0].exterior_segment_indices)
        _validate_chain_contiguity(backend, count)
        report = _delivered_open_boundary_membership_report(manifest, np.zeros((count, 2)),
            kinds, (nodes[0], nodes[-1]), chains, source)
        assert report["passed"], report["failure_taxonomy"]


def test_invalid_source_bindings_fail_closed() -> None:
    base = _source_fixture()
    changes = [
        lambda m, r, n, k: m["boundary"]["open_boundaries"][1].update(source_obc_id=1),
        lambda m, r, n, k: m["boundary"]["open_boundaries"][1].update(source_obc_id=99),
        lambda m, r, n, k: m["boundary"]["open_boundaries"][1].update(cyclic=True),
        lambda m, r, n, k: r["open_boundary_chains"][0].update(node_count=2),
        lambda m, r, n, k: r["open_boundary_chains"][0].update(node_sequence_zero_based=[100, 103, 102]),
        lambda m, r, n, k: r["open_boundary_chains"][0].update(node_sequence_zero_based=[100, 101, 101]),
        lambda m, r, n, k: r["open_boundary_chains"][0].update(node_sequence_zero_based=[999, 101, 102]),
        lambda m, r, n, k: r["open_boundary_chains"][0].update(obc_id=1),
        lambda m, r, n, k: n.__setitem__(0, n[1]),
        lambda m, r, n, k: r["open_boundary_chains"][0].update(node_sequence_zero_based=[104, 105, 106]),
        lambda m, r, n, k: m["boundary"]["open_boundaries"].pop(),
    ]
    for change in changes:
        values = copy.deepcopy(base)
        change(*values)
        try:
            _open_boundaries_from_adaptive_chains(*values)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid or ambiguous source binding was accepted")


def test_closed_domain_has_no_invented_obc() -> None:
    manifest = {"boundary": {"expected_open_boundary_count": 0, "open_boundaries": []}}
    assert _open_boundaries_from_adaptive_chains(manifest, {"open_boundary_chains": []},
        [0, 1, 2, 3], ("land",) * 4) == ()


TESTS = (
    test_exact_membership_passes,
    test_free_text_and_low_proxy_cannot_activate_gate,
    test_land_segment_is_rejected,
    test_reordered_segments_are_rejected,
    test_duplicate_segments_are_rejected,
    test_incomplete_open_segment_coverage_is_rejected,
    test_wrong_id_is_rejected,
    test_hard_anchor_signature_is_rejected,
    test_source_ids_and_declared_order_are_exact,
    test_source_reversed_wrapped_and_cyclic_order,
    test_invalid_source_bindings_fail_closed,
    test_closed_domain_has_no_invented_obc,
)


def main() -> int:
    failures: list[tuple[str, BaseException]] = []
    for test in TESTS:
        try:
            test()
        except BaseException as exc:
            failures.append((test.__name__, exc))
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    if failures:
        print(f"{len(failures)} of {len(TESTS)} Adaptive open-delivery tests failed")
        return 1
    print(f"All {len(TESTS)} Adaptive open-delivery tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
