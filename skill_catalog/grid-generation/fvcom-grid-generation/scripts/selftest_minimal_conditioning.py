#!/usr/bin/env python3
"""Focused tests for the minimal-topology-v1 conditioning contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.local_topology import (  # noqa: E402
    AggressiveConditioningConfig,
    condition_mesh_aggressive,
)
from fvcom_grid_generation.metrics import chain_edges  # noqa: E402
from fvcom_grid_generation.portfolio_conditioning import (  # noqa: E402
    PortfolioConditioningConfig,
    _primary_topology_config,
)


def _minimal_config(**overrides: object) -> AggressiveConditioningConfig:
    base = AggressiveConditioningConfig(
        profile_name="minimal-topology-v1",
        stage_order="valence-before-thin",
        enable_pruning=False,
        enable_thin_repair=True,
        enable_valence_repair=True,
        max_rounds=4,
        max_prunes_per_round=0,
        boundary_edit_policy="none",
        max_boundary_edits_per_round=0,
        max_boundary_welds_per_round=0,
        max_boundary_ear_removals_per_round=0,
        micro_relax_cycles=0,
    )
    return replace(base, **overrides)


def _nine_spoke_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[list[int]],
    list[str],
]:
    count = 9
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    ring = np.column_stack((np.cos(angles), np.sin(angles)))
    points = np.vstack((ring, np.asarray([[0.0, 0.0]])))
    triangles = np.asarray(
        [[count, index, (index + 1) % count] for index in range(count)],
        dtype=int,
    )
    return (
        points,
        triangles,
        np.asarray([True] * count + [False]),
        [list(range(count))],
        ["land"] * count + ["interior"],
    )


def _run(
    points: np.ndarray,
    triangles: np.ndarray,
    fixed: np.ndarray,
    chains: list[list[int]],
    kinds: list[str],
    targets: np.ndarray,
    config: AggressiveConditioningConfig | None = None,
):
    return condition_mesh_aggressive(
        points,
        triangles,
        fixed,
        chains,
        np.empty(0, dtype=int),
        target_spacing_m=targets,
        boundary_kinds=kinds,
        hard_anchor_mask=fixed,
        config=config or _minimal_config(),
    )


def _canonical_triangles(values: np.ndarray) -> list[tuple[int, int, int]]:
    return sorted(
        tuple(sorted(map(int, triangle)))
        for triangle in np.asarray(values, dtype=int)
    )


def test_auto_resolver_is_minimal_and_disables_broad_edits() -> None:
    resolved = _primary_topology_config(
        PortfolioConditioningConfig(conditioning_profile="auto"),
        effective_profile="minimal-topology-v1",
    )
    assert resolved.profile_name == "minimal-topology-v1"
    assert resolved.stage_order == "valence-before-thin"
    assert not resolved.enable_pruning
    assert resolved.boundary_edit_policy == "none"
    assert resolved.max_boundary_edits_per_round == 0
    assert resolved.max_boundary_welds_per_round == 0
    assert resolved.max_boundary_ear_removals_per_round == 0
    assert resolved.micro_relax_cycles == 0


def test_valence_only_repair_closes_without_boundary_movement() -> None:
    points, triangles, fixed, chains, kinds = _nine_spoke_fixture()
    result = _run(
        points,
        triangles,
        fixed,
        chains,
        kinds,
        np.ones(len(points), dtype=float),
    )
    assert result.report["before"]["count_valence_above_limit"] == 1
    assert result.report["after"]["count_valence_above_limit"] == 0
    assert result.report["after"]["superthin_triangle_count"] == 0
    assert result.report["minimal_local_debt_closed"]
    assert result.report["edit_counts"].get(
        "high-valence-cavity-remove",
        0,
    ) == 1
    assert np.array_equal(result.nodes_xy[:9], points[:9])
    assert chain_edges(result.constraint_chains) == chain_edges(chains)


def test_superthin_only_repair_is_an_atomic_flip() -> None:
    points = np.asarray(
        [[-1.0, 0.0], [0.0, -0.01], [1.0, 0.0], [0.0, 2.0]],
        dtype=float,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    fixed = np.ones(4, dtype=bool)
    chains = [[0, 1, 2, 3]]
    result = _run(
        points,
        triangles,
        fixed,
        chains,
        ["land"] * 4,
        np.asarray([0.25, 0.25, 0.25, 0.50]),
    )
    assert result.report["before"]["superthin_triangle_count"] == 1
    assert result.report["after"]["superthin_triangle_count"] == 0
    assert result.report["minimal_local_debt_closed"]
    assert result.report["edit_counts"] == {"superthin-edge-flip": 1}
    assert len(result.triangles) == len(triangles)
    assert np.array_equal(result.nodes_xy, points)
    assert result.constraint_chains == chains


def test_protected_superthin_boundary_is_reported_not_deleted() -> None:
    points = np.asarray(
        [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0e-4]],
        dtype=float,
    )
    triangles = np.asarray([[0, 1, 2]], dtype=int)
    fixed = np.ones(3, dtype=bool)
    result = _run(
        points,
        triangles,
        fixed,
        [[0, 1, 2]],
        ["land"] * 3,
        np.ones(3, dtype=float),
    )
    assert not result.report["minimal_local_debt_closed"]
    assert result.report["after"]["superthin_triangle_count"] == 1
    assert len(result.triangles) == 1
    assert np.array_equal(result.nodes_xy, points)
    assert _canonical_triangles(result.triangles) == [(0, 1, 2)]


def test_expired_deadline_stops_before_first_transaction() -> None:
    points, triangles, fixed, chains, kinds = _nine_spoke_fixture()
    result = _run(
        points,
        triangles,
        fixed,
        chains,
        kinds,
        np.ones(len(points), dtype=float),
        _minimal_config(deadline_monotonic_s=time.perf_counter() - 1.0),
    )
    assert result.report["deadline_reached"]
    assert result.report["rounds"] == []
    assert result.report["after"]["count_valence_above_limit"] == 1
    assert not result.report["minimal_local_debt_closed"]
    assert np.array_equal(result.nodes_xy, points)
    assert _canonical_triangles(result.triangles) == _canonical_triangles(
        triangles
    )


def test_deterministic_replay_matches_connectivity_and_lineage() -> None:
    fixture = _nine_spoke_fixture()
    first = _run(
        *fixture,
        np.ones(len(fixture[0]), dtype=float),
    )
    second = _run(
        *fixture,
        np.ones(len(fixture[0]), dtype=float),
    )
    assert np.array_equal(first.nodes_xy, second.nodes_xy)
    assert np.array_equal(first.triangles, second.triangles)
    assert np.array_equal(first.node_lineage, second.node_lineage)
    assert first.edit_ledger == second.edit_ledger


TESTS: tuple[Callable[[], None], ...] = (
    test_auto_resolver_is_minimal_and_disables_broad_edits,
    test_valence_only_repair_closes_without_boundary_movement,
    test_superthin_only_repair_is_an_atomic_flip,
    test_protected_superthin_boundary_is_reported_not_deleted,
    test_expired_deadline_stops_before_first_transaction,
    test_deterministic_replay_matches_connectivity_and_lineage,
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
        return 1
    print("All minimal-conditioning tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
