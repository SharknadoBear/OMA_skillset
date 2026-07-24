#!/usr/bin/env python3
"""Audit or repair causal superthin connectivity in an FVCOM 2DM mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from condition_mesh_local import (  # noqa: E402
    _bbox,
    _boundary_metadata,
    _json_safe,
    _remap_depths,
    _serialized_roundtrip_audit,
    _target_sizes,
)
from fvcom_grid_generation.connectivity_restriction import (  # noqa: E402
    ConnectivityRestrictionConfig,
    audit_superthin_connectivity,
)
from fvcom_grid_generation.local_topology import (  # noqa: E402
    AggressiveConditioningConfig,
    condition_mesh_aggressive,
)
from fvcom_grid_generation.projection import (  # noqa: E402
    local_utm_projection,
    project_points,
    unproject_points,
)
from fvcom_grid_generation.sms_2dm import read_2dm, write_2dm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "repair"), default="audit")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output-mesh")
    parser.add_argument("--report", required=True)
    parser.add_argument("--boundary-nodes-geojson")
    parser.add_argument("--boundary-resolution-manifest")
    parser.add_argument("--size-field-nc")
    parser.add_argument("--target-spacing-m", type=float)
    parser.add_argument("--max-transactions", type=int, default=32)
    parser.add_argument("--wall-time-s", type=float, default=21600.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    report_path = Path(args.report)
    output_path = Path(args.output_mesh) if args.output_mesh else None
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if args.mode == "repair" and output_path is None:
        parser.error("--output-mesh is required in repair mode")
    if output_path is not None and output_path.resolve() == mesh_path.resolve():
        parser.error("--output-mesh must differ from --mesh")
    if args.max_transactions < 0:
        parser.error("--max-transactions must be nonnegative")
    if args.wall_time_s <= 0.0:
        parser.error("--wall-time-s must be positive")
    if args.target_spacing_m is not None and args.target_spacing_m <= 0.0:
        parser.error("--target-spacing-m must be positive")
    outputs = [
        path
        for path in (report_path, output_path)
        if path is not None and path.exists()
    ]
    if outputs and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing outputs: {outputs}"
        )

    mesh = read_2dm(mesh_path)
    projection = local_utm_projection(_bbox(mesh.nodes_lonlat))
    points = project_points(mesh.nodes_lonlat, projection)
    triangles = np.asarray(mesh.triangles, dtype=int) - 1
    open_nodes = np.asarray(mesh.open_boundary_nodes, dtype=int) - 1
    chains, kinds, hard, explicit_targets = _boundary_metadata(
        len(points),
        triangles,
        open_nodes,
        args.boundary_nodes_geojson,
        args.boundary_resolution_manifest,
    )
    fixed = np.zeros(len(points), dtype=bool)
    for chain in chains:
        fixed[np.asarray(chain, dtype=int)] = True
    targets = _target_sizes(
        mesh.nodes_lonlat,
        points,
        triangles,
        args.size_field_nc,
    )
    if args.target_spacing_m is not None:
        targets[:] = float(args.target_spacing_m)
    explicit = np.isfinite(explicit_targets) & (explicit_targets > 0.0)
    targets[explicit] = explicit_targets[explicit]

    policy_config = ConnectivityRestrictionConfig(
        maximum_transactions=int(args.max_transactions),
    )
    before = audit_superthin_connectivity(
        points,
        triangles,
        targets,
        chains,
        config=policy_config,
    )
    document: dict[str, object] = {
        "schema_version": "fvcom_superthin_connectivity_restriction_v1",
        "mode": str(args.mode),
        "input_mesh": str(mesh_path),
        "projection_epsg": int(projection.epsg),
        "settings": {
            "maximum_transactions": int(args.max_transactions),
            "wall_time_seconds": float(args.wall_time_s),
            "topology_only": True,
            "boundary_window_fallback": False,
        },
        "before": before,
    }
    if args.mode == "repair":
        result = condition_mesh_aggressive(
            points,
            triangles,
            fixed,
            chains,
            open_nodes,
            target_spacing_m=targets,
            boundary_kinds=kinds,
            hard_anchor_mask=hard,
            config=AggressiveConditioningConfig(
                thin_repair_profile="systematic-v5",
                systematic_gate_scope="loop-end",
                systematic_v5_connectivity_only=True,
                systematic_v5_enable_connectivity_restriction=True,
                systematic_v5_max_connectivity_transactions_per_round=int(
                    args.max_transactions
                ),
                systematic_v5_enable_boundary_window_fallback=False,
                deadline_monotonic_s=(
                    time.perf_counter() + float(args.wall_time_s)
                ),
                max_rounds=1,
                enable_pruning=False,
                enable_thin_repair=True,
                enable_valence_repair=False,
                max_prunes_per_round=0,
                max_valence_removals_per_round=0,
                micro_relax_cycles=0,
            ),
        )
        assert output_path is not None
        depths = _remap_depths(
            mesh.depths,
            points,
            result.nodes_xy,
            result.node_lineage,
        )
        written = write_2dm(
            output_path,
            unproject_points(result.nodes_xy, projection),
            depths,
            result.triangles + 1,
            result.open_boundary_nodes_zero_based + 1,
            mesh_name=f"{mesh.mesh_name}_connectivity_restricted_v1",
        )
        roundtrip = _serialized_roundtrip_audit(
            written,
            result,
            projection,
        )
        after = audit_superthin_connectivity(
            result.nodes_xy,
            result.triangles,
            result.target_spacing_m,
            result.constraint_chains,
            node_lineage=result.node_lineage,
            restricted_lineage_edges=result.restricted_lineage_edges,
            config=policy_config,
        )
        document.update(
            {
                "output_mesh": str(written),
                "after": after,
                "conditioning_report": result.report,
                "edit_ledger": result.edit_ledger,
                "restricted_lineage_edges": [
                    list(map(int, edge))
                    for edge in sorted(
                        result.restricted_lineage_edges
                    )
                ],
                "serialized_roundtrip": roundtrip,
            }
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_json_safe(document), indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mode": str(args.mode),
                "report": str(report_path),
                "output_mesh": (
                    str(output_path)
                    if output_path is not None
                    and args.mode == "repair"
                    else None
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
