#!/usr/bin/env python3
"""Regenerate passage-removal audit figures from accepted checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from apply_thin_passage_removal import _plot_transaction  # noqa: E402
from condition_mesh_local import _bbox, _boundary_metadata, _target_sizes  # noqa: E402
from fvcom_grid_generation.local_topology import _inventory_superthin_components  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection, project_points  # noqa: E402
from fvcom_grid_generation.sms_2dm import read_2dm  # noqa: E402
from fvcom_grid_generation.thin_passage import ThinPassageRemovalConfig  # noqa: E402
from fvcom_grid_generation.visual_superthin import create_visual_state  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--only-component-id")
    args = parser.parse_args()
    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    plan = json.loads(Path(report["plan"]).read_text(encoding="utf-8-sig"))
    restrictions = {
        tuple(sorted(map(int, edge))) for edge in plan.get("restricted_lineage_edges", [])
    }
    config = ThinPassageRemovalConfig(**plan.get("config", {}))
    mesh_path = Path(report["input_mesh"])
    boundary_path = Path(report["boundary_nodes_geojson"])
    lineage_path: Path | None = None

    outputs: list[str] = []
    for transaction in report["transactions"]:
        if args.only_component_id and str(transaction["component_id"]) != str(args.only_component_id):
            checkpoint = transaction["checkpoint"]
            mesh_path = Path(checkpoint["mesh"])
            boundary_path = Path(checkpoint["boundary_nodes_geojson"])
            lineage_path = Path(checkpoint["node_lineage"])
            continue
        before_state, conditioning = _load_state(
            mesh_path,
            boundary_path,
            Path(report["size_field_nc"]),
            lineage_path,
            restrictions,
        )
        matching = [
            item
            for item in _inventory_superthin_components(before_state, conditioning)
            if str(item["component_id"]) == str(transaction["component_id"])
        ]
        if len(matching) != 1:
            raise ValueError(f"component unavailable while replotting: {transaction['component_id']}")
        checkpoint = transaction["checkpoint"]
        after_state, _ = _load_state(
            Path(checkpoint["mesh"]),
            Path(checkpoint["boundary_nodes_geojson"]),
            Path(report["size_field_nc"]),
            Path(checkpoint["node_lineage"]),
            restrictions,
        )
        selected = transaction["attempts"][int(transaction["selected_candidate_index"])]
        output = Path(transaction["output_plot"])
        _plot_transaction(
            output,
            before_state,
            after_state,
            matching[0],
            selected,
            transaction,
            config=config,
            dpi=int(args.dpi),
        )
        outputs.append(str(output))
        mesh_path = Path(checkpoint["mesh"])
        boundary_path = Path(checkpoint["boundary_nodes_geojson"])
        lineage_path = Path(checkpoint["node_lineage"])
    print(json.dumps({"plots": outputs}, indent=2))
    return 0


def _load_state(
    mesh_path: Path,
    boundary_path: Path,
    size_path: Path,
    lineage_path: Path | None,
    restrictions: set[tuple[int, int]],
):
    mesh = read_2dm(mesh_path)
    projection = local_utm_projection(_bbox(mesh.nodes_lonlat))
    points = project_points(mesh.nodes_lonlat, projection)
    triangles = np.asarray(mesh.triangles, dtype=int) - 1
    open_nodes = np.asarray(mesh.open_boundary_nodes, dtype=int) - 1
    chains, kinds, hard, explicit_targets = _boundary_metadata(
        len(points), triangles, open_nodes, str(boundary_path), None
    )
    fixed = np.zeros(len(points), dtype=bool)
    for chain in chains:
        fixed[np.asarray(chain, dtype=int)] = True
    targets = _target_sizes(mesh.nodes_lonlat, points, triangles, str(size_path))
    explicit = np.isfinite(explicit_targets) & (explicit_targets > 0.0)
    targets[explicit] = explicit_targets[explicit]
    if lineage_path is None:
        lineage = np.arange(len(points), dtype=int)
    else:
        document = json.loads(lineage_path.read_text(encoding="utf-8-sig"))
        if "source_node_index_zero_based" in document:
            values = document["source_node_index_zero_based"]
        else:
            values = [
                item["source_node_index_zero_based"]
                for item in document["node_ids_1based_to_source_lineage"]
            ]
        lineage = np.asarray(values, dtype=int)
    state, conditioning, _ = create_visual_state(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
        boundary_kinds=kinds,
        hard_anchor_mask=hard,
        node_lineage=lineage,
        restricted_lineage_edges=restrictions,
    )
    return state, conditioning


if __name__ == "__main__":
    raise SystemExit(main())
