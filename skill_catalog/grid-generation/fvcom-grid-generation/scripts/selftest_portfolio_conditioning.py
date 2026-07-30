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
            == "ordered_noncyclic_arcs_only"
        )
        assert report["final_global_audit"]["open_boundary_chain_count"] == 2
        assert report["roundtrip"]["open_boundary_chain_order_match"]
        assert report["roundtrip"]["open_boundary_id_match"]


TESTS: tuple[Callable[[], None], ...] = (
    test_closed_lake_common_conditioning,
    test_two_obc_ids_order_and_boundary_are_preserved,
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
