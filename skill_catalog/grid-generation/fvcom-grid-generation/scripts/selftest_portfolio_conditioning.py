#!/usr/bin/env python3
"""Synthetic closed-lake and plural-OBC tests for portfolio conditioning."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Callable

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation import portfolio_conditioning as conditioning_module  # noqa: E402
from fvcom_grid_generation.portfolio_conditioning import (  # noqa: E402
    PortfolioConditioningConfig,
    condition_portfolio_mesh,
)
from fvcom_grid_generation.projection import (  # noqa: E402
    local_utm_projection,
    unproject_points,
)
from fvcom_grid_generation.sms_2dm import read_2dm, write_2dm  # noqa: E402


def _fixture_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projection = local_utm_projection((-77.1, 43.9, -76.9, 44.1))
    center_lonlat = np.asarray([[-77.0, 44.0]], dtype=float)
    center_x, center_y = projection.to_xy.transform(
        center_lonlat[:, 0],
        center_lonlat[:, 1],
    )
    center = np.asarray([center_x[0], center_y[0]], dtype=float)
    radius = 1_000.0
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    boundary = center[None, :] + radius * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    points_xy = np.vstack((center, boundary))
    lonlat = unproject_points(points_xy, projection)
    triangles = np.asarray(
        [
            [1, 2 + index, 2 + ((index + 1) % 6)]
            for index in range(6)
        ],
        dtype=int,
    )
    return lonlat, points_xy, triangles


def _write_inputs(
    root: Path,
    *,
    open_boundary_chains: list[list[int]],
    open_boundary_ids: list[int],
) -> tuple[Path, Path, Path, np.ndarray]:
    lonlat, _points_xy, triangles = _fixture_geometry()
    mesh = root / "raw.2dm"
    write_2dm(
        mesh,
        lonlat,
        np.full(len(lonlat), 999.0, dtype=float),
        triangles,
        np.empty(0, dtype=int),
        mesh_name="portfolio_fixture",
        open_boundary_chains=open_boundary_chains,
        open_boundary_ids=open_boundary_ids,
    )
    west = float(np.min(lonlat[:, 0]) - 0.02)
    east = float(np.max(lonlat[:, 0]) + 0.02)
    south = float(np.min(lonlat[:, 1]) - 0.02)
    north = float(np.max(lonlat[:, 1]) + 0.02)
    lon = np.linspace(west, east, 31)
    lat = np.linspace(south, north, 31)
    target = np.full((len(lat), len(lon)), 1_000.0, dtype=float)
    coverage = np.ones_like(target, dtype=np.uint8)
    size_field = root / "size_field.nc"
    xr.Dataset(
        {
            "mesh_size_m": (("lat", "lon"), target),
            "size_field_coverage_mask": (("lat", "lon"), coverage),
        },
        coords={"lon": lon, "lat": lat},
        attrs={
            "schema_version": "fvcom_size_field_v4",
            "coverage_policy": "strict",
        },
    ).to_netcdf(size_field)
    bathymetry = root / "bathymetry.nc"
    xr.Dataset(
        {"depth_m": (("lat", "lon"), np.full_like(target, 20.0))},
        coords={"lon": lon, "lat": lat},
        attrs={"vertical_convention": "positive_down"},
    ).to_netcdf(bathymetry)
    return mesh, size_field, bathymetry, lonlat


def test_closed_lake_common_conditioning() -> None:
    with tempfile.TemporaryDirectory(prefix="portfolio_closed_lake_") as temp:
        root = Path(temp)
        mesh, size_field, bathymetry, raw_lonlat = _write_inputs(
            root,
            open_boundary_chains=[],
            open_boundary_ids=[],
        )
        output = root / "conditioned"
        report = condition_portfolio_mesh(
            mesh,
            size_field,
            bathymetry,
            output,
            name="closed_lake_conditioned",
            config=PortfolioConditioningConfig(
                primary_rounds=2,
                terminal_rounds=1,
                area_transition_max_patches=2,
            ),
        )
        assert report["status"] == "pass", json.dumps(report, indent=2)
        assert report["policy"]["requested_profile"] == "auto"
        assert report["policy"]["effective_profile"] == "minimal-topology-v1"
        assert report["minimal_local_debt_closed"]
        assert report["fvcom_ready"]
        assert report["area_transition"]["enabled"] is False
        assert (
            report["area_transition"]["reason"]
            == "disabled_by_minimal_topology_v1"
        )
        assert report["inputs"]["canonical_size_field"][
            "sampling_interface_schema_version"
        ] == "legacy_unspecified"
        quality = json.loads(
            (output / "mesh_quality.json").read_text(encoding="utf-8")
        )
        assert quality["canonical_inputs"][
            "sampling_interface_schema_version"
        ] == "legacy_unspecified"
        delivered = read_2dm(output / "conditioned.2dm")
        assert len(delivered.open_boundary_chains) == 0
        assert delivered.open_boundary_ids == ()
        assert np.allclose(delivered.depths, 20.0, atol=1.0e-4)
        assert not np.allclose(delivered.depths, 999.0)
        assert np.max(
            np.linalg.norm(
                delivered.nodes_lonlat[1:] - raw_lonlat[1:],
                axis=1,
            )
        ) <= 1.0e-10
        assert report["final_global_audit"]["boundary_edge_set_exact"]
        assert report["roundtrip"]["passed"]
        assert (output / "conditioning_report.json").is_file()
        assert (output / "delivered_boundary_nodes.geojson").is_file()
        assert (output / "obc_remap_manifest.json").is_file()
        assert (output / "mesh_quality.json").is_file()
        replay = root / "conditioned_replay"
        replay_report = condition_portfolio_mesh(
            mesh,
            size_field,
            bathymetry,
            replay,
            name="closed_lake_conditioned",
            config=PortfolioConditioningConfig(
                primary_rounds=2,
                terminal_rounds=1,
                area_transition_max_patches=2,
            ),
        )
        assert replay_report["minimal_local_debt_closed"]
        assert (output / "conditioned.2dm").read_bytes() == (
            replay / "conditioned.2dm"
        ).read_bytes()
        try:
            condition_portfolio_mesh(
                mesh,
                size_field,
                bathymetry,
                output,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing output directory was overwritten")


def test_two_obc_ids_order_and_boundary_are_preserved() -> None:
    with tempfile.TemporaryDirectory(prefix="portfolio_two_obc_") as temp:
        root = Path(temp)
        source_chains = [[2, 3], [5, 6]]
        source_ids = [17, 23]
        mesh, size_field, bathymetry, raw_lonlat = _write_inputs(
            root,
            open_boundary_chains=source_chains,
            open_boundary_ids=source_ids,
        )
        output = root / "conditioned"
        report = condition_portfolio_mesh(
            mesh,
            size_field,
            bathymetry,
            output,
            name="two_obc_conditioned",
            config=PortfolioConditioningConfig(
                primary_rounds=2,
                terminal_rounds=1,
                area_transition_max_patches=2,
            ),
            boundary_contract={
                "open_boundary_count": 2,
                "open_boundary_ids": source_ids,
                "open_boundary_cyclic": [False, False],
            },
        )
        assert report["status"] == "pass", json.dumps(report, indent=2)
        delivered = read_2dm(output / "conditioned.2dm")
        assert delivered.open_boundary_ids == tuple(source_ids)
        assert [
            values.tolist() for values in delivered.open_boundary_chains
        ] == source_chains
        assert np.allclose(delivered.depths, 20.0, atol=1.0e-4)
        assert np.max(
            np.linalg.norm(
                delivered.nodes_lonlat[1:] - raw_lonlat[1:],
                axis=1,
            )
        ) <= 1.0e-10
        remap = json.loads(
            (output / "obc_remap_manifest.json").read_text(encoding="utf-8")
        )
        assert remap["source_chain_count"] == 2
        assert remap["delivered_chain_count"] == 2
        assert remap["source_nodestring_ids"] == source_ids
        assert remap["delivered_nodestring_ids"] == source_ids
        assert remap["forcing_compatible"]
        assert all(
            chain["orientation_preserved"] for chain in remap["chains"]
        )
        assert (
            remap["cyclicity_contract"]["policy"]
            == "external_cyclicity_sidecar"
        )
        assert report["final_global_audit"]["open_boundary_chain_count"] == 2
        assert report["roundtrip"]["open_boundary_chain_order_match"]
        assert report["roundtrip"]["open_boundary_id_match"]


def test_cyclic_obc_and_source_forcing_remain_readiness_failures() -> None:
    with tempfile.TemporaryDirectory(prefix="portfolio_cyclic_obc_") as temp:
        root = Path(temp)
        source_chains = [[2, 3, 4, 5, 6, 7]]
        source_ids = [41]
        mesh, size_field, bathymetry, _raw_lonlat = _write_inputs(
            root,
            open_boundary_chains=source_chains,
            open_boundary_ids=source_ids,
        )
        output = root / "conditioned"
        report = condition_portfolio_mesh(
            mesh,
            size_field,
            bathymetry,
            output,
            name="cyclic_obc_conditioned",
            boundary_contract={
                "open_boundary_count": 1,
                "open_boundary_ids": ["cyclic_exchange"],
                "open_boundary_cyclic": [True],
            },
            source_boundary_metadata={"forcing_compatible": False},
        )
        assert report["minimal_local_debt_closed"]
        assert report["benchmark_grid_baseline_ready"]
        assert report["fvcom_ready"]
        assert report["status"] == "pass"
        assert not report["submission_eligible"]
        failures = set(report["submission_failure_taxonomy"])
        assert "cyclic_obc_not_self_describing_in_sms_2dm" in failures
        assert "open_boundary_forcing_incompatible" in failures
        assert (
            report["open_boundary_cyclicity_contract"]["chains"][0]["cyclic"]
            is True
        )
        remap = json.loads(
            (output / "obc_remap_manifest.json").read_text(encoding="utf-8")
        )
        assert remap["chains"][0]["cyclic"]
        assert not remap["forcing_compatible"]


def test_scientific_input_failure_is_separate_from_local_closure() -> None:
    with tempfile.TemporaryDirectory(prefix="portfolio_science_gate_") as temp:
        root = Path(temp)
        mesh, size_field, bathymetry, _raw_lonlat = _write_inputs(
            root,
            open_boundary_chains=[],
            open_boundary_ids=[],
        )
        report = condition_portfolio_mesh(
            mesh,
            size_field,
            bathymetry,
            root / "conditioned",
            scientific_input_valid=False,
            scientific_input_note="synthetic rejected boundary placement",
        )
        assert report["minimal_local_debt_closed"]
        assert not report["fvcom_ready"]
        assert "scientific_input_invalid" in report[
            "fvcom_readiness_failure_taxonomy"
        ]
        assert report["scientific_input"]["note"] == (
            "synthetic rejected boundary placement"
        )


def test_rejected_primary_candidate_is_retained_deterministically() -> None:
    with tempfile.TemporaryDirectory(prefix="portfolio_rejected_candidate_") as temp:
        root = Path(temp)
        mesh, size_field, bathymetry, _raw_lonlat = _write_inputs(
            root,
            open_boundary_chains=[],
            open_boundary_ids=[],
        )
        original = conditioning_module._stage_regressions

        def force_remaining_gate(
            before: dict[str, object],
            after: dict[str, object],
            *,
            minimal_policy: bool = False,
        ) -> list[str]:
            failures = original(
                before,
                after,
                minimal_policy=minimal_policy,
            )
            if minimal_policy:
                failures.append("synthetic_remaining_hard_gate")
            return sorted(set(failures))

        conditioning_module._stage_regressions = force_remaining_gate
        try:
            reports = [
                condition_portfolio_mesh(
                    mesh,
                    size_field,
                    bathymetry,
                    root / output_name,
                    name="rejected_candidate_fixture",
                )
                for output_name in ("first", "second")
            ]
        finally:
            conditioning_module._stage_regressions = original

        for report in reports:
            primary = report["primary_topology"]
            assert primary["rollback_applied"]
            assert "synthetic_remaining_hard_gate" in primary["rollback_reasons"]
            evidence = primary["rejected_candidate_evidence"]
            assert evidence["status"] == "rejected"
            for key in (
                "candidate_2dm",
                "mesh_quality_json",
                "boundary_nodes_geojson",
                "edit_ledger_json",
                "rollback_manifest_json",
            ):
                assert Path(evidence[key]["path"]).is_file()
            quality = json.loads(
                Path(evidence["mesh_quality_json"]["path"]).read_text(
                    encoding="utf-8"
                )
            )
            assert quality["artifact_role"] == "rejected_primary_candidate"
            assert not quality["accepted_for_delivery"]
            assert quality["serialized_roundtrip"]["passed"]

        first = reports[0]["primary_topology"]["rejected_candidate_evidence"]
        second = reports[1]["primary_topology"]["rejected_candidate_evidence"]
        for key in (
            "candidate_2dm",
            "mesh_quality_json",
            "boundary_nodes_geojson",
            "edit_ledger_json",
        ):
            assert first[key]["sha256"] == second[key]["sha256"]


def _authorized_midpoint_audit_fixture() -> tuple[
    conditioning_module._ConditioningState,
    np.ndarray,
    list[list[int]],
    list[dict[str, object]],
]:
    raw_points = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    points = np.vstack((raw_points, np.asarray([[1.0, 0.0]])))
    triangles = np.asarray(
        [
            [0, 5, 4],
            [5, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=int,
    )
    fixed = np.asarray([True, True, True, True, False, True])
    hard = np.asarray([True, True, True, True, False, False])
    state = conditioning_module._ConditioningState(
        points=points,
        triangles=triangles,
        fixed=fixed,
        constraint_chains=[[0, 5, 1, 2, 3]],
        boundary_kinds=[
            "fixed_boundary",
            "fixed_boundary",
            "fixed_boundary",
            "fixed_boundary",
            "interior",
            "fixed_boundary",
        ],
        hard=hard,
        targets=np.full(len(points), 1.0, dtype=float),
        raw_lineage=np.asarray([0, 1, 2, 3, 4, -1], dtype=int),
    )
    ledger: list[dict[str, object]] = [
        {
            "operation": "minimal-fixed-hard-fan-source-arc-refinement",
            "component_id": "thin-0-fixture",
            "automatic": True,
            "review_required": False,
            "inserted_boundary_node_count": 1,
            "inserted_support_node_count": 0,
            "removed_movable_node_count": 0,
        },
        {
            "operation": "minimal-fixed-hard-fan-transaction-accepted",
            "component_id": "thin-0-fixture",
            "source_edge_lineage": [0, 1],
        },
    ]
    return state, raw_points, [[0, 1, 2, 3]], ledger


def _run_midpoint_audit(
    state: conditioning_module._ConditioningState,
    raw_points: np.ndarray,
    raw_boundary_chains: list[list[int]],
    ledger: list[dict[str, object]] | None,
    *,
    raw_obc_chains: list[list[int]] | None = None,
) -> dict[str, object]:
    obc = raw_obc_chains or []
    return conditioning_module._global_structural_audit(
        state,
        raw_points,
        raw_boundary_chains,
        obc,
        list(range(1, len(obc) + 1)),
        [False] * len(obc),
        lambda values: np.ones(len(values), dtype=float),
        boundary_refinement_ledger=ledger,
    )


def test_ledger_authorizes_exact_non_obc_boundary_midpoint() -> None:
    state, raw_points, raw_boundary_chains, ledger = (
        _authorized_midpoint_audit_fixture()
    )
    audit = _run_midpoint_audit(
        state,
        raw_points,
        raw_boundary_chains,
        ledger,
    )
    assert audit["core_passed"], audit
    assert audit["raw_boundary_refinement_authorized"]
    assert audit["authorized_boundary_insertion_count"] == 1
    assert audit["raw_boundary_lineage_topology_preserved"]
    assert audit["boundary_edge_set_exact"]


def test_unlogged_or_invalid_boundary_midpoint_is_rejected() -> None:
    state, raw_points, raw_boundary_chains, ledger = (
        _authorized_midpoint_audit_fixture()
    )
    unlogged = _run_midpoint_audit(
        state,
        raw_points,
        raw_boundary_chains,
        None,
    )
    assert not unlogged["core_passed"]
    assert "unauthorized_boundary_refinement" in unlogged["core_failures"]

    shifted = state.clone()
    shifted.points[5] = np.asarray([1.0, 0.05])
    bad_midpoint = _run_midpoint_audit(
        shifted,
        raw_points,
        raw_boundary_chains,
        ledger,
    )
    assert not bad_midpoint["core_passed"]
    assert (
        "inserted_boundary_node_is_not_exact_midpoint"
        in bad_midpoint["boundary_refinement_failures"]
    )

    obc_touch = _run_midpoint_audit(
        state,
        raw_points,
        raw_boundary_chains,
        ledger,
        raw_obc_chains=[[0, 1]],
    )
    assert not obc_touch["core_passed"]
    assert (
        "open_boundary_edge_refinement_not_allowed"
        in obc_touch["boundary_refinement_failures"]
    )


TESTS: tuple[Callable[[], None], ...] = (
    test_closed_lake_common_conditioning,
    test_two_obc_ids_order_and_boundary_are_preserved,
    test_cyclic_obc_and_source_forcing_remain_readiness_failures,
    test_scientific_input_failure_is_separate_from_local_closure,
    test_rejected_primary_candidate_is_retained_deterministically,
    test_ledger_authorizes_exact_non_obc_boundary_midpoint,
    test_unlogged_or_invalid_boundary_midpoint_is_rejected,
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
    print("All portfolio-conditioning tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
