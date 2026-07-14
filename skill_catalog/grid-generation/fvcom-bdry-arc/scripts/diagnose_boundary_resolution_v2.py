#!/usr/bin/env python3
"""Print the worst delivered spacing gradients and passage-gate inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    gpkg = Path(manifest["outputs"]["boundary_resolution_gpkg"])
    nodes = gpd.read_file(gpkg, layer="boundary_nodes")
    if nodes.crs is None:
        raise ValueError("boundary_nodes layer has no CRS")
    projected = nodes.estimate_utm_crs()
    nodes = nodes.to_crs(projected).sort_values(["chain_id", "chain_position"])
    rows = []
    for chain_id, group in nodes.groupby("chain_id", sort=True):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue
        xy = np.asarray([[geometry.x, geometry.y] for geometry in group.geometry], dtype=float)
        target = np.asarray(group["target_spacing_m"], dtype=float)
        lengths = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
        gradients = np.abs(np.roll(target, -1) - target) / np.maximum(lengths, 1.0)
        harmonic = 2.0 / (1.0 / target + 1.0 / np.roll(target, -1))
        ratios = lengths / np.maximum(harmonic, 1.0)
        selected_indices = set(np.argsort(gradients)[::-1][: max(1, int(args.top))].tolist())
        selected_indices.update(np.argsort(ratios)[::-1][: max(1, int(args.top))].tolist())
        for index in sorted(selected_indices):
            following = (int(index) + 1) % len(group)
            rows.append(
                {
                    "gradient": float(gradients[index]),
                    "l_over_h": float(ratios[index]),
                    "chain_id": int(chain_id),
                    "chain_position_a": int(group.iloc[index]["chain_position"]),
                    "chain_position_b": int(group.iloc[following]["chain_position"]),
                    "edge_length_m": float(lengths[index]),
                    "target_a_m": float(target[index]),
                    "target_b_m": float(target[following]),
                    "kind_a": str(group.iloc[index]["boundary_kind"]),
                    "kind_b": str(group.iloc[following]["boundary_kind"]),
                    "anchor_type_a": str(group.iloc[index].get("anchor_type", "")),
                    "anchor_type_b": str(group.iloc[following].get("anchor_type", "")),
                }
            )
    diagnostics = json.loads(
        Path(manifest["outputs"]["boundary_resolution_diagnostics_json"]).read_text(encoding="utf-8-sig")
    )
    passages = diagnostics.get("channel_passages", {})
    output = {
        "status": manifest.get("final_status"),
        "failure_taxonomy": manifest.get("failure_taxonomy", []),
        "worst_spacing_gradients": sorted(rows, key=lambda item: item["gradient"], reverse=True)[: args.top],
        "worst_edge_target_ratios": sorted(rows, key=lambda item: item["l_over_h"], reverse=True)[: args.top],
        "passages": {
            key: passages.get(key)
            for key in (
                "passage_count",
                "protected_unresolved_count",
                "unprotected_unresolved_count",
                "maximum_inventory_width_m",
                "all_component_pair_count",
                "spatially_indexed_component_pair_count",
            )
        },
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
