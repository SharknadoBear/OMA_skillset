#!/usr/bin/env python3
"""Offline tests for the standard bathymetric mesh review map."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation import plotting
from fvcom_grid_generation.plotting import (
    validate_standard_mesh_review_map,
    write_standard_mesh_review_map,
)
from fvcom_grid_generation.quality_policy import public_policy_binding
from fvcom_grid_generation.sms_2dm import write_2dm


def fixture(root: Path) -> tuple[Path, Path, Path]:
    mesh = root / "mesh.2dm"
    write_2dm(
        mesh,
        np.asarray(
            [[-75.2, 39.0], [-75.0, 39.0], [-75.0, 39.2], [-75.2, 39.2]],
            dtype=float,
        ),
        np.asarray([2.0, 4.0, 8.0, 6.0], dtype=float),
        np.asarray([[1, 2, 3], [1, 3, 4]], dtype=int),
        np.empty(0, dtype=int),
        mesh_name="Delaware fixture",
        open_boundary_chains=[[1, 2]],
        open_boundary_ids=[1],
    )
    quality = root / "quality.json"
    quality.write_text(
        json.dumps(
            {
                "schema_version": "fvcom_mesh_quality_v3",
                "quality_policy": public_policy_binding(),
                "oceanmesh_quality": {"q_l3_sigma": 0.812345},
                "all_quality_findings": [],
                "benchmark_grid_baseline_ready": True,
                "failure_taxonomy": [],
                "regional_refinement_debt": [],
            }
        ),
        encoding="utf-8",
    )
    boundary = root / "boundary.geojson"
    boundary.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}),
        encoding="utf-8",
    )
    return mesh, quality, boundary


def test_offline_map_and_stale_rejection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mesh, quality, boundary = fixture(root)
        image = root / "mesh_review_map.png"
        manifest = root / "mesh_review_map_manifest.json"
        report = write_standard_mesh_review_map(
            image,
            manifest,
            mesh_path=mesh,
            quality_path=quality,
            boundary_nodes_path=boundary,
            grid_name="Delaware River Estuary",
            basemap_provider="offline",
        )
        assert image.is_file() and image.stat().st_size > 10_000
        assert report["title"] == (
            "Delaware River Estuary | q_L3σ = 0.8123"
        )
        assert report["open_boundary_chain_count"] == 1
        assert report["basemap"]["provider"] == "offline"
        assert validate_standard_mesh_review_map(
            image,
            manifest,
            mesh_path=mesh,
            quality_path=quality,
            boundary_nodes_path=boundary,
        )["passed"]
        image.write_bytes(image.read_bytes() + b"stale")
        stale = validate_standard_mesh_review_map(
            image,
            manifest,
            mesh_path=mesh,
            quality_path=quality,
            boundary_nodes_path=boundary,
        )
        assert "mesh_review_map_image_hash_mismatch" in stale[
            "failure_taxonomy"
        ]


def test_online_route_without_network_dependency() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mesh, quality, boundary = fixture(root)
        original = plotting._draw_review_background

        def fake_online(ax, bbox, coastline_path, provider):
            assert provider == "topo"
            ax.set_facecolor("#abcdef")
            return {
                "status": "ok",
                "source": "synthetic online topographic fixture",
                "provider": "topo",
                "fallback": False,
            }

        plotting._draw_review_background = fake_online
        try:
            report = write_standard_mesh_review_map(
                root / "online.png",
                root / "online.json",
                mesh_path=mesh,
                quality_path=quality,
                boundary_nodes_path=boundary,
                grid_name="Online route fixture",
                basemap_provider="topo",
            )
        finally:
            plotting._draw_review_background = original
        assert not report["basemap"]["fallback"]
        assert report["basemap"]["provider"] == "topo"


def main() -> None:
    test_offline_map_and_stale_rejection()
    print("PASS test_offline_map_and_stale_rejection")
    test_online_route_without_network_dependency()
    print("PASS test_online_route_without_network_dependency")
    print("All standard mesh-review-map tests passed.")


if __name__ == "__main__":
    main()
