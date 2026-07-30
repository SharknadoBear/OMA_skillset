#!/usr/bin/env python
"""Focused self-test for raw boundary/field transition visual diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile

import matplotlib
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fvcom_grid_generation.raw_transition_diagnostics import (  # noqa: E402
    SCHEMA_VERSION,
    write_raw_transition_diagnostics,
)
from fvcom_grid_generation.sms_2dm import write_2dm  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        mesh_path = root / "raw.2dm"
        boundary_path = root / "canonical_boundary_nodes.geojson"
        size_path = root / "canonical_size_field_v4.nc"
        quality_path = root / "quality.json"
        output = root / "diagnostics"

        # Four triangles around an off-centre node create a deterministic
        # adjacent-area jump.  Boundary nodes 1-2 carry a 3:1 interface jump.
        nodes = np.asarray(
            [
                [-75.0010, 39.9990],
                [-74.9990, 39.9990],
                [-74.9990, 40.0010],
                [-75.0010, 40.0010],
                [-75.0008, 39.9992],
            ],
            dtype=float,
        )
        triangles = np.asarray(
            [
                [1, 2, 5],
                [2, 3, 5],
                [3, 4, 5],
                [4, 1, 5],
            ],
            dtype=int,
        )
        write_2dm(
            mesh_path,
            nodes,
            np.full(len(nodes), 5.0),
            triangles,
            np.empty(0, dtype=int),
            mesh_name="raw_transition_fixture",
        )

        targets = [300.0, 300.0, 100.0, 100.0]
        features = []
        for index, target in enumerate(targets):
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "node_index_zero_based": index,
                        "target_spacing_m": target,
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": nodes[index].tolist(),
                    },
                }
            )
        boundary_path.write_text(
            json.dumps(
                {"type": "FeatureCollection", "features": features},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        lon = np.linspace(-75.002, -74.998, 5)
        lat = np.linspace(39.998, 40.002, 5)
        dataset = xr.Dataset(
            data_vars={
                "mesh_size_m": (
                    ("lat", "lon"),
                    np.full((len(lat), len(lon)), 100.0),
                ),
                "size_field_coverage_mask": (
                    ("lat", "lon"),
                    np.ones((len(lat), len(lon)), dtype=np.uint8),
                ),
            },
            coords={"lon": lon, "lat": lat},
            attrs={"schema_version": "fvcom_size_field_v4"},
        )
        dataset.to_netcdf(size_path)
        quality_path.write_text(
            json.dumps(
                {
                    "schema_version": "fvcom_mesh_quality_v2",
                    "accepted": False,
                    "node_count": 5,
                    "triangle_count": 4,
                    "oceanmesh_quality": {"q_l3_sigma": 0.5},
                    "max_adjacent_area_change": 0.9,
                    "failure_taxonomy": [
                        "adjacent_area_change_above_threshold"
                    ],
                    "canonical_size_sampling": {
                        "schema_version": "fvcom_boundary_trace_sampler_v1",
                        "gradation": 0.2,
                        "samples_per_target": 4.0,
                        "nearest_sample_count": 16,
                    },
                    "thresholds": {
                        "max_size_error_p95": 1.55,
                        "max_size_error": 2.0,
                        "max_area_change": 0.5,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        input_hashes = {
            path: _sha256(path)
            for path in (mesh_path, boundary_path, size_path, quality_path)
        }

        report = write_raw_transition_diagnostics(
            mesh_path,
            boundary_path,
            size_path,
            quality_path,
            output,
            max_plot_triangles=2,
        )

        assert report["schema_version"] == SCHEMA_VERSION
        assert report["method"]["size_field_sampling"]["enabled"]
        assert report["method"]["size_field_sampling"][
            "endpoint_midpoint_exact_by_construction"
        ]
        assert report["diagnostic_status"] == "needs_review"
        assert (
            report["boundary_field_interface_hotspots"]["count"] >= 1
        )
        assert report["adjacent_area_change_hotspots"]["count"] >= 1
        assert (
            report["boundary_target_mapping"][
                "unmatched_mesh_boundary_node_count"
            ]
            == 0
        )
        assert report["plot_sampling"]["whole_mesh_plotted_count"] == 2
        assert (
            report["plot_sampling"][
                "boundary_first_ring_plotted_count"
            ]
            <= 2
        )
        assert report["authoritative_edge_size_audit"]["counts"][
            "constraint_edges"
        ] == 4
        assert np.isclose(
            report["triangle_l_over_h"]["maximum"],
            report["authoritative_edge_size_audit"]["triangle_l_over_h"][
                "all"
            ]["maximum"],
        )
        assert np.isclose(
            report["boundary_field_interface_hotspots"][
                "symmetric_ratio"
            ]["maximum"],
            report["authoritative_edge_size_audit"][
                "boundary_field_interface"
            ]["symmetric_ratio"]["maximum"],
        )
        assert str(matplotlib.get_backend()).lower() == "agg"
        for artifact in (
            output / "whole_mesh_l_over_h.png",
            output / "boundary_first_ring_transition.png",
            output / "raw_transition_diagnostics.json",
        ):
            assert artifact.is_file()
            assert artifact.stat().st_size > 0
        assert {
            path: _sha256(path)
            for path in (mesh_path, boundary_path, size_path, quality_path)
        } == input_hashes

    print("raw transition diagnostics self-test: 1/1 passed")


if __name__ == "__main__":
    main()
