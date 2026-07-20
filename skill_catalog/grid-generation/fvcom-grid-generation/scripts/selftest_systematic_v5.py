"""Deterministic smoke tests for interaction relaxation and the V5 loop."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from fvcom_grid_generation.interaction_relaxation import (  # noqa: E402
    InteractionRelaxationConfig,
    interaction_metrics,
    relax_mesh_interaction,
)
from fvcom_grid_generation.local_topology import AggressiveConditioningConfig  # noqa: E402
from fvcom_grid_generation.systematic_v5 import (  # noqa: E402
    SystematicV5LoopConfig,
    run_systematic_v5_loop,
)


def _fan() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 0.35],
        ],
        dtype=float,
    )
    triangles = np.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=int,
    )
    fixed = np.asarray([True, True, True, True, False])
    return points, triangles, fixed


def test_fixed_connectivity_interaction_is_repeatable_and_keeps_boundary_exact() -> None:
    points, triangles, fixed = _fan()
    config = InteractionRelaxationConfig(
        iterations=4,
        checkpoint_interval=1,
        superthin_trigger=100,
    )
    first = relax_mesh_interaction(
        points,
        triangles,
        fixed,
        target_spacing_m=np.full(len(points), 1.5),
        config=config,
    )
    second = relax_mesh_interaction(
        points,
        triangles,
        fixed,
        target_spacing_m=np.full(len(points), 1.5),
        config=config,
    )
    assert np.array_equal(first.nodes_xy, second.nodes_xy)
    assert np.array_equal(first.nodes_xy[fixed], points[fixed])
    assert first.report["global_delaunay_rebuild"] is False
    assert first.report["after"]["nonpositive_signed_area_count"] == 0


def test_unified_debt_uses_quality_or_minimum_angle() -> None:
    points = np.asarray([[0.0, 0.0], [2.0, 0.0], [1.0, 0.05]], dtype=float)
    triangles = np.asarray([[0, 1, 2]], dtype=int)
    metrics = interaction_metrics(
        points,
        triangles,
        InteractionRelaxationConfig(),
    )
    assert metrics["superthin_triangle_count"] == 1
    assert metrics["superthin_severity_sum"] > 0.0


def test_rejected_cycle_restores_zero_debt_champion_exactly() -> None:
    points, triangles, fixed = _fan()
    result = run_systematic_v5_loop(
        points,
        triangles,
        fixed,
        [[0, 1, 2, 3]],
        np.empty(0, dtype=int),
        target_spacing_m=np.full(len(points), 1.5),
        boundary_kinds=["land"] * 4 + ["interior"],
        hard_anchor_mask=np.asarray([True, True, True, True, False]),
        topology_config=AggressiveConditioningConfig(
            thin_repair_profile="systematic-v5",
            enable_pruning=False,
            enable_valence_repair=False,
            micro_relax_cycles=0,
        ),
        loop_config=SystematicV5LoopConfig(
            total_iterations=3,
            maximum_cycles=1,
            burst_ladder=(3,),
            maximum_burst=3,
            superthin_trigger=20,
            checkpoint_interval=1,
            wall_clock_seconds=30.0,
            minimum_champion_gain=1.0,
            target_q_l3_sigma=2.0,
        ),
    )
    loop = result.report["systematic_v5_loop"]
    assert loop["committed_cycle_count"] == 0
    assert result.report["after"]["superthin_triangle_count"] == 0
    assert np.array_equal(result.nodes_xy, points)
    assert np.array_equal(result.triangles, triangles)


def test_deadline_returns_last_structurally_valid_checkpoint() -> None:
    points, triangles, fixed = _fan()
    result = run_systematic_v5_loop(
        points,
        triangles,
        fixed,
        [[0, 1, 2, 3]],
        np.empty(0, dtype=int),
        target_spacing_m=np.full(len(points), 1.5),
        boundary_kinds=["land"] * 4 + ["interior"],
        hard_anchor_mask=np.asarray([True, True, True, True, False]),
        topology_config=replace(
            AggressiveConditioningConfig(thin_repair_profile="systematic-v5"),
            deadline_monotonic_s=time.perf_counter() + 1.0,
        ),
        loop_config=SystematicV5LoopConfig(
            total_iterations=100,
            maximum_cycles=6,
            wall_clock_seconds=0.01,
            deadline_monotonic_s=time.perf_counter() + 1.0,
        ),
    )
    assert result.report["after"]["nonpositive_signed_area_count"] == 0
    assert np.array_equal(result.nodes_xy[fixed], points[fixed])


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"systematic-v5 selftests passed ({len(tests)} tests)")


if __name__ == "__main__":
    main()
