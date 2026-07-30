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
    _CanonicalSizeSampler,
    _diagnostic_failure_taxonomy,
    write_raw_transition_diagnostics,
)
from fvcom_grid_generation.projection import (  # noqa: E402
    local_utm_projection,
    project_points,
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
        legacy_size_path = root / "legacy_halo_size_field_v4.nc"
        quality_path = root / "quality.json"
        quality_v2_path = root / "quality_v2.json"
        output = root / "diagnostics"
        output_v2 = root / "diagnostics_v2"

        # Four triangles around an off-centre node create a deterministic
        # adjacent-area jump. Boundary nodes 1-2 carry a large interface jump,
        # while the remote low targets force the v2 replay to expand beyond
        # its first nearest trace sample.
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

        targets = [300.0, 300.0, 20.0, 20.0]
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
                "model_domain_mask": (
                    ("lat", "lon"),
                    np.ones((len(lat), len(lon)), dtype=np.uint8),
                ),
            },
            coords={"lon": lon, "lat": lat},
            attrs={
                "schema_version": "fvcom_size_field_v4",
                "sampling_interface_schema_version": (
                    "fvcom_wet_mask_sampling_v2"
                ),
            },
        )
        dataset.to_netcdf(size_path)
        legacy_values = np.full((3, 3), 5.0, dtype=float)
        legacy_values[:, 0] = 100.0
        legacy_values[:, 2] = 1000.0
        xr.Dataset(
            data_vars={
                "mesh_size_m": (("lat", "lon"), legacy_values),
                "size_field_coverage_mask": (
                    ("lat", "lon"),
                    np.ones((3, 3), dtype=np.uint8),
                ),
                "model_domain_mask": (
                    ("lat", "lon"),
                    np.asarray(
                        [
                            [1, 0, 1],
                            [1, 0, 1],
                            [1, 0, 1],
                        ],
                        dtype=np.uint8,
                    ),
                ),
            },
            coords={
                "lon": [-75.002, -75.001, -75.000],
                "lat": [39.999, 40.000, 40.001],
            },
            attrs={
                "schema_version": "fvcom_size_field_v4",
                "sampling_interface_schema_version": (
                    "fvcom_size_sampling_halo_v1"
                ),
            },
        ).to_netcdf(legacy_size_path)
        # Node four is the larger endpoint of both of its normalized incident
        # boundary edges: (0, 3) and (2, 3). The undirected diagnostic replay
        # must still include it exactly.
        projection = local_utm_projection(
            (
                float(np.min(nodes[:, 0])),
                float(np.min(nodes[:, 1])),
                float(np.max(nodes[:, 0])),
                float(np.max(nodes[:, 1])),
            )
        )
        nodes_xy = project_points(nodes, projection)
        endpoint_sampler = _CanonicalSizeSampler(size_path, projection)
        endpoint_sampler.enable_boundary_trace(
            nodes_xy,
            [(0, 1), (1, 2), (2, 3), (0, 3)],
            np.asarray(targets, dtype=float),
            gradation=0.2,
            samples_per_target=4.0,
            nearest_sample_count=16,
            schema_version="fvcom_boundary_trace_sampler_v2",
            no_active_support_policy="boundary_trace_authoritative",
            query_chunk_size=2,
        )
        assert endpoint_sampler.trace_report["sample_distribution"] == (
            "linear_endpoint_target_metric_equidistribution"
        )
        assert endpoint_sampler.trace_report[
            "base_raster_sampling_interface_schema_version"
        ] == "fvcom_wet_mask_sampling_v2"
        assert endpoint_sampler.trace_report[
            "no_active_support_policy"
        ] == "boundary_trace_authoritative"
        assert endpoint_sampler.trace_report["query_chunk_size"] == 2
        assert endpoint_sampler.trace_report["memory_bounded_query_chunks"]
        assert np.isclose(
            endpoint_sampler.sample_xy(nodes_xy[[3]])[0],
            targets[3],
            rtol=0.0,
            atol=1.0e-12,
        )
        reference_sampler = _CanonicalSizeSampler(size_path, projection)
        reference_sampler.enable_boundary_trace(
            nodes_xy,
            [(0, 1), (1, 2), (2, 3), (0, 3)],
            np.asarray(targets, dtype=float),
            gradation=0.2,
            samples_per_target=4.0,
            nearest_sample_count=16,
            schema_version="fvcom_boundary_trace_sampler_v2",
            no_active_support_policy="boundary_trace_authoritative",
            query_chunk_size=1_000,
        )
        assert np.allclose(
            endpoint_sampler.sample_xy(nodes_xy),
            reference_sampler.sample_xy(nodes_xy),
            rtol=0.0,
            atol=1.0e-12,
        )
        pathological_xy = np.asarray(
            [[0.0, 0.0], [1.0e12, 0.0]],
            dtype=float,
        )
        for trace_schema in (
            "fvcom_boundary_trace_sampler_v1",
            "fvcom_boundary_trace_sampler_v2",
        ):
            cap_sampler = _CanonicalSizeSampler(size_path, projection)
            try:
                cap_sampler.enable_boundary_trace(
                    pathological_xy,
                    [(0, 1)],
                    np.ones(2, dtype=float),
                    gradation=0.2,
                    maximum_total_sample_count=1_000,
                    schema_version=trace_schema,
                )
            except ValueError as exc:
                assert "before allocation" in str(exc)
            else:
                raise AssertionError(
                    "diagnostic trace safety cap did not fail for "
                    f"{trace_schema}"
                )

        legacy_projection = local_utm_projection(
            (-75.002, 39.999, -75.000, 40.001)
        )
        legacy_sampler = _CanonicalSizeSampler(
            legacy_size_path,
            legacy_projection,
        )
        legacy_query = project_points(
            np.asarray([[-75.0015, 40.000]], dtype=float),
            legacy_projection,
        )
        assert np.isclose(
            legacy_sampler.sample_xy(legacy_query)[0],
            52.5,
            atol=1.0e-6,
        )
        quality_payload = {
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
        }
        quality_path.write_text(
            json.dumps(
                quality_payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        quality_v2_payload = json.loads(json.dumps(quality_payload))
        quality_v2_payload["canonical_size_sampling"] = {
            "schema_version": "fvcom_boundary_trace_sampler_v2",
            "gradation": 0.2,
            "samples_per_target": 4.0,
            "nearest_sample_count": 1,
            "initial_nearest_sample_count": 1,
            "adaptive_neighbor_expansion": True,
            "exact_discrete_trace_sample_minimum": True,
            "no_active_support_policy": "boundary_trace_authoritative",
        }
        quality_v2_path.write_text(
            json.dumps(
                quality_v2_payload,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        input_hashes = {
            path: _sha256(path)
            for path in (
                mesh_path,
                boundary_path,
                size_path,
                quality_path,
                quality_v2_path,
            )
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
        assert report["method"]["size_field_sampling"][
            "no_active_support_policy"
        ] == "raster_min"
        assert report["diagnostic_status"] == "needs_review"
        assert (
            set(
                report["authoritative_edge_size_audit"][
                    "failure_taxonomy"
                ]
            )
            <= set(report["failure_taxonomy"])
        )
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

        report_v2 = write_raw_transition_diagnostics(
            mesh_path,
            boundary_path,
            size_path,
            quality_v2_path,
            output_v2,
            max_plot_triangles=2,
        )
        v2_sampling = report_v2["method"]["size_field_sampling"]
        assert v2_sampling["enabled"]
        assert (
            v2_sampling["schema_version"]
            == "fvcom_boundary_trace_sampler_v2"
        )
        assert v2_sampling["adaptive_neighbor_expansion"]
        assert v2_sampling["exact_discrete_trace_sample_minimum"]
        assert v2_sampling["no_active_support_policy"] == (
            "boundary_trace_authoritative"
        )
        assert v2_sampling["expanded_sample_query_count"] > 0
        assert v2_sampling["maximum_neighbors_examined"] > 1
        assert report_v2["boundary_field_interface_hotspots"]["count"] >= 1
        assert str(matplotlib.get_backend()).lower() == "agg"
        for artifact in (
            output / "whole_mesh_l_over_h.png",
            output / "boundary_first_ring_transition.png",
            output / "raw_transition_diagnostics.json",
            output_v2 / "whole_mesh_l_over_h.png",
            output_v2 / "boundary_first_ring_transition.png",
            output_v2 / "raw_transition_diagnostics.json",
        ):
            assert artifact.is_file()
            assert artifact.stat().st_size > 0
        assert {
            path: _sha256(path)
            for path in (
                mesh_path,
                boundary_path,
                size_path,
                quality_path,
                quality_v2_path,
            )
        } == input_hashes

        edge_only_failures = _diagnostic_failure_taxonomy(
            edge_audit={
                "passed": False,
                "failure_taxonomy": [
                    "edge_aware_target_size_maximum_exceeded"
                ],
            },
            interface_hotspot_count=0,
            area_hotspot_count=0,
            unmatched_boundary_node_count=0,
            invalid_triangle_l_over_h_count=0,
        )
        assert edge_only_failures == [
            "edge_aware_target_size_maximum_exceeded"
        ]

    print("raw transition diagnostics self-test: 2/2 passed")


if __name__ == "__main__":
    main()
