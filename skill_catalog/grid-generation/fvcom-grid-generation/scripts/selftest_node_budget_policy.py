#!/usr/bin/env python3
"""Focused regression tests for shared node-budget and spacing defaults."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fvcom_grid_generation.gmsh_experiment import (  # noqa: E402
    BudgetConfig,
    DEFAULT_NODE_LIMIT,
    DEFAULT_PREFLIGHT_LIMIT,
    select_uniform_target_m,
)
from fvcom_grid_generation.mesh import MeshConfig  # noqa: E402
from fvcom_grid_generation.node_budget import (  # noqa: E402
    DEFAULT_HARD_NODE_LIMIT,
    DEFAULT_MAX_INTERIOR_POINTS,
    DEFAULT_NODE_BUDGET_STOP_FRACTION,
    DEFAULT_PREFLIGHT_NODE_LIMIT,
    DEFAULT_SPACING_QUANTUM_M,
    delivered_node_budget_report,
)
from fvcom_grid_generation.portfolio_case import (  # noqa: E402
    PortfolioCaseConfig,
)
from fvcom_grid_generation.workflow import (  # noqa: E402
    GridConfig,
    _apply_delivered_node_budget_gate,
)
from run_gmsh_fvcom import build_parser as build_gmsh_parser  # noqa: E402
from run_mesher_portfolio_case import (  # noqa: E402
    build_parser as build_portfolio_parser,
)


def test_shared_default_budget() -> None:
    assert DEFAULT_HARD_NODE_LIMIT == 5_000_000
    assert DEFAULT_PREFLIGHT_NODE_LIMIT == 4_500_000
    assert DEFAULT_NODE_BUDGET_STOP_FRACTION == 0.90
    assert DEFAULT_MAX_INTERIOR_POINTS == 4_500_000
    assert DEFAULT_SPACING_QUANTUM_M == 25.0

    gmsh = BudgetConfig()
    assert DEFAULT_NODE_LIMIT == gmsh.max_nodes == 5_000_000
    assert DEFAULT_PREFLIGHT_LIMIT == gmsh.preflight_nodes == 4_500_000
    assert gmsh.step_m == 25.0

    portfolio = PortfolioCaseConfig()
    assert portfolio.preflight_node_limit == 4_500_000
    assert portfolio.hard_node_limit == 5_000_000

    production = GridConfig()
    assert production.max_total_nodes == 5_000_000
    assert production.node_budget_stop_fraction == 0.90
    assert (
        production.max_total_nodes * production.node_budget_stop_fraction
        == 4_500_000
    )
    assert production.max_interior_points == 4_500_000
    assert MeshConfig().max_interior_points == 4_500_000

    policy_path = (
        SCRIPTS.parent / "references" / "fvcom_grid_quality_policy_v1.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy["thresholds"]["planning_node_limit"] == DEFAULT_PREFLIGHT_NODE_LIMIT
    assert policy["thresholds"]["hard_node_limit"] == DEFAULT_HARD_NODE_LIMIT


def test_cli_defaults_and_legacy_overrides() -> None:
    gmsh = build_gmsh_parser().parse_args(["--case-manifest", "case.json"])
    assert gmsh.preflight_node_threshold == 4_500_000
    assert gmsh.hard_node_cap == 5_000_000
    gmsh_legacy = build_gmsh_parser().parse_args(
        [
            "--case-manifest",
            "case.json",
            "--preflight-node-threshold",
            "135000",
            "--hard-node-cap",
            "150000",
        ]
    )
    assert gmsh_legacy.preflight_node_threshold == 135_000
    assert gmsh_legacy.hard_node_cap == 150_000

    portfolio = build_portfolio_parser().parse_args(
        ["--case-manifest", "case.json", "--output-dir", "output"]
    )
    assert portfolio.preflight_node_limit == 4_500_000
    assert portfolio.hard_node_limit == 5_000_000
    portfolio_legacy = build_portfolio_parser().parse_args(
        [
            "--case-manifest",
            "case.json",
            "--output-dir",
            "output",
            "--preflight-node-limit",
            "135000",
            "--hard-node-limit",
            "150000",
        ]
    )
    assert portfolio_legacy.preflight_node_limit == 135_000
    assert portfolio_legacy.hard_node_limit == 150_000
    PortfolioCaseConfig(
        preflight_node_limit=portfolio_legacy.preflight_node_limit,
        hard_node_limit=portfolio_legacy.hard_node_limit,
    )


def test_spacing_quantum_is_configurable_search_granularity() -> None:
    area_weights_m2 = np.asarray([13_000_000.0])
    distance_to_obc_m = np.asarray([0.0])
    selected_25m, estimate_25m = select_uniform_target_m(
        101.0,
        100,
        area_weights_m2,
        distance_to_obc_m,
        has_open_boundary=False,
        config=BudgetConfig(
            preflight_nodes=1_000,
            max_nodes=2_000,
            step_m=25.0,
        ),
    )
    assert selected_25m == 150.0
    assert estimate_25m <= 1_000

    selected_10m, estimate_10m = select_uniform_target_m(
        101.0,
        100,
        area_weights_m2,
        distance_to_obc_m,
        has_open_boundary=False,
        config=BudgetConfig(
            preflight_nodes=1_000,
            max_nodes=2_000,
            step_m=10.0,
        ),
    )
    assert selected_10m == 130.0
    assert estimate_10m <= 1_000


def test_delivered_hard_cap_is_an_acceptance_gate() -> None:
    passing = delivered_node_budget_report(5_000_000, 5_000_000)
    assert passing["passed"] is True
    assert passing["remaining_node_capacity"] == 0
    assert passing["failure_taxonomy"] == []

    failing = delivered_node_budget_report(5_000_001, 5_000_000)
    assert failing["passed"] is False
    assert failing["remaining_node_capacity"] == -1
    assert failing["failure_taxonomy"] == ["hard_node_cap_exceeded"]

    quality: dict[str, object] = {
        "accepted": True,
        "failure_taxonomy": [],
    }
    _apply_delivered_node_budget_gate(quality, failing)
    assert quality["accepted"] is False
    assert quality["node_budget"] == failing
    assert quality["failure_taxonomy"] == ["hard_node_cap_exceeded"]


def test_current_case_manifests_declare_future_run_defaults() -> None:
    root = SCRIPTS / "research" / "gmsh"
    schema = json.loads((root / "case-manifest.schema.json").read_text())
    budget_schema = schema["$defs"]["budget"]["properties"]
    assert budget_schema["preflight_node_threshold"]["minimum"] == 1
    assert budget_schema["hard_node_cap"]["minimum"] == 1
    assert budget_schema["hu_increment_m"]["const"] == 25

    cases = sorted((root / "cases").glob("*.json"))
    assert len(cases) == 6
    for case in cases:
        payload = json.loads(case.read_text())
        assert payload["budget"]["preflight_node_threshold"] == 4_500_000
        assert payload["budget"]["hard_node_cap"] == 5_000_000
        assert payload["budget"]["hu_increment_m"] == 25


def test_lake_superior_continuity_manifest_contract() -> None:
    path = (
        SCRIPTS
        / "research"
        / "gmsh"
        / "continuity_cases"
        / "lake_superior.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["case_id"] == "lake_superior"
    assert payload["domain_type"] == "closed_lake"
    assert payload["boundary"]["expected_open_boundary_count"] == 0
    assert payload["boundary"]["open_boundaries"] == []
    assert payload["boundary"]["expected_island_holes"] == 45
    assert "prep_boundary_v5" in payload["boundary"]["resolution_manifest"]
    assert payload["bathymetry"]["netcdf"].endswith(
        "lake_superior_fvcom_depth_v2.nc"
    )
    assert payload["budget"]["preflight_node_threshold"] == 4_500_000
    assert payload["budget"]["hard_node_cap"] == 5_000_000
    assert payload["budget"]["hu_increment_m"] == 25
    assert payload["readiness"]["status"] == "ready"


TESTS: tuple[Callable[[], None], ...] = (
    test_shared_default_budget,
    test_cli_defaults_and_legacy_overrides,
    test_spacing_quantum_is_configurable_search_granularity,
    test_delivered_hard_cap_is_an_acceptance_gate,
    test_current_case_manifests_declare_future_run_defaults,
    test_lake_superior_continuity_manifest_contract,
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
        print(f"{len(failures)} of {len(TESTS)} node-budget tests failed")
        return 1
    print(f"All {len(TESTS)} node-budget tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
