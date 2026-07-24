#!/usr/bin/env python3
"""Focused tests for the adaptive-v2 boundary handshake."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from shapely.geometry import LineString, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.boundary import BoundaryNodes, evaluate_boundary_contract_v2  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection  # noqa: E402


def _nodes(targets: list[float], hard: list[bool]) -> BoundaryNodes:
    xy = np.asarray([[0.0, 0.0], [500.0, 0.0], [1000.0, 0.0], [1000.0, 500.0], [0.0, 500.0]])
    return BoundaryNodes(
        xy=xy,
        lonlat=np.zeros_like(xy),
        kinds=["open", "open", "open", "land", "land"],
        target_spacing_m=np.asarray(targets, dtype=float),
        exterior_indices=list(range(5)),
        open_boundary_indices=[0, 1, 2],
        constraint_chains=[list(range(5))],
        domain_polygon_xy=Polygon(xy),
        open_boundary_xy=LineString(xy[:3]),
        land_boundary_xy=LineString([xy[2], xy[3], xy[4], xy[0]]),
        island_polygons_xy=[],
        projection=local_utm_projection((-75.0, 39.0, -74.0, 40.0)),
        hard_anchor_mask=np.asarray(hard, dtype=bool),
        adaptive_resolution=True,
        resolution_profile="adaptive-coastal-v2",
    )


def test_contract_passes_smooth_anchored_chain() -> None:
    report = evaluate_boundary_contract_v2(
        _nodes([700.0, 700.0, 700.0, 700.0, 700.0], [True, False, True, False, False])
    )
    assert report["passed"] is True, report
    assert report["maximum_l_over_h"] <= 1.55


def test_contract_rejects_missing_second_landfall_anchor() -> None:
    report = evaluate_boundary_contract_v2(
        _nodes([700.0, 700.0, 700.0, 700.0, 700.0], [True, False, False, False, False])
    )
    assert "open_boundary_landfall_anchor_missing" in report["failure_taxonomy"]


def test_contract_rejects_kind_transition_jump() -> None:
    report = evaluate_boundary_contract_v2(
        _nodes([500.0, 500.0, 500.0, 150.0, 150.0], [True, False, True, False, False])
    )
    assert "open_land_junction_spacing_jump" in report["failure_taxonomy"]


def main() -> int:
    tests = [
        test_contract_passes_smooth_anchored_chain,
        test_contract_rejects_missing_second_landfall_anchor,
        test_contract_rejects_kind_transition_jump,
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} adaptive-v2 boundary contract tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
