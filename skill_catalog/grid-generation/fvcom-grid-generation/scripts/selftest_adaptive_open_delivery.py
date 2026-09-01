#!/usr/bin/env python3
"""Regression tests for exact Adaptive-v2 OBC/exterior delivery."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from shapely.geometry import LineString, Polygon

from fvcom_grid_generation.boundary_topology import BoundaryTopologyCompensation
from fvcom_grid_generation.gmsh_experiment import (
    SourceOpenBoundary,
    _delivered_open_boundary_membership_report,
    _validate_adaptive_boundary_evidence,
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


TESTS = (
    test_exact_membership_passes,
    test_free_text_and_low_proxy_cannot_activate_gate,
    test_land_segment_is_rejected,
    test_reordered_segments_are_rejected,
    test_duplicate_segments_are_rejected,
    test_incomplete_open_segment_coverage_is_rejected,
    test_wrong_id_is_rejected,
    test_hard_anchor_signature_is_rejected,
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
