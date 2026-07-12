#!/usr/bin/env python3
"""Inventory FVCOM valence violations and plot local repair-candidate maps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.local_topology import inventory_high_valence  # noqa: E402
from fvcom_grid_generation.metrics import build_edge_topology  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection, project_points  # noqa: E402
from fvcom_grid_generation.sms_2dm import read_2dm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boundary-nodes-geojson")
    parser.add_argument("--conditioning-report", help="Optional repair report whose rejected node lineage should be plotted first.")
    parser.add_argument("--map-count", type=int, default=6)
    parser.add_argument("--graph-rings", type=int, default=3)
    parser.add_argument("--max-valence", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    output_dir = Path(args.output_dir)
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    products = {
        "summary_json": output_dir / "high_valence_inventory.json",
        "records_csv": output_dir / "high_valence_nodes.csv",
        "records_geojson": output_dir / "high_valence_nodes.geojson",
        "overview_png": output_dir / "high_valence_overview.png",
        "local_png": output_dir / "high_valence_local_examples.png",
    }
    existing = [path for path in products.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = read_2dm(mesh_path)
    projection = local_utm_projection(_bbox(mesh.nodes_lonlat))
    points = project_points(mesh.nodes_lonlat, projection)
    triangles = np.asarray(mesh.triangles, dtype=int) - 1
    chains, fixed, kinds, hard, lineage = _boundary_metadata(
        len(points),
        args.boundary_nodes_geojson,
    )
    inventory = inventory_high_valence(
        points,
        triangles,
        constraint_chains=chains,
        fixed_node_mask=fixed,
        boundary_kinds=kinds,
        hard_anchor_mask=hard,
        node_lineage=lineage,
        max_valence=int(args.max_valence),
    )
    records = inventory["records"]
    for record in records:
        node = int(record["node_index_zero_based"])
        record["longitude"] = float(mesh.nodes_lonlat[node, 0])
        record["latitude"] = float(mesh.nodes_lonlat[node, 1])
    rejected_lineage, rejected_indices = _rejected_nodes(args.conditioning_report)
    selected = _select_examples(records, rejected_lineage, rejected_indices, max(1, int(args.map_count)))

    _write_csv(products["records_csv"], records)
    products["records_geojson"].write_text(json.dumps(_geojson(records), indent=2), encoding="utf-8")
    _plot_overview(products["overview_png"], points, triangles, records, int(args.dpi))
    _plot_local_examples(
        products["local_png"],
        points,
        triangles,
        records,
        selected,
        max(1, int(args.graph_rings)),
        int(args.dpi),
    )
    document = {
        "schema_version": "fvcom_high_valence_inventory_v1",
        "mesh": str(mesh_path),
        "projection_epsg": int(projection.epsg),
        "conditioning_report": str(args.conditioning_report) if args.conditioning_report else None,
        "rejected_node_lineage": rejected_lineage,
        "rejected_node_indices_zero_based": rejected_indices,
        "selected_local_examples": [int(record["node_index_zero_based"]) for record in selected],
        "summary": {key: value for key, value in inventory.items() if key != "records"},
        "records": records,
        "outputs": {key: str(value) for key, value in products.items()},
    }
    products["summary_json"].write_text(json.dumps(_json_safe(document), indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "violation_count": int(inventory["violation_count"]),
                "maximum_valence": int(inventory["maximum_valence"]),
                "route_counts": inventory["route_counts"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    return 0


def _boundary_metadata(
    node_count: int,
    path_value: str | None,
) -> tuple[list[list[int]], np.ndarray, list[str], np.ndarray, np.ndarray]:
    fixed = np.zeros(node_count, dtype=bool)
    hard = np.zeros(node_count, dtype=bool)
    kinds = ["interior"] * node_count
    lineage = np.arange(node_count, dtype=int)
    if not path_value:
        return [], fixed, kinds, hard, lineage
    document = json.loads(Path(path_value).read_text(encoding="utf-8-sig"))
    grouped: dict[int, list[tuple[int, int]]] = {}
    for feature in document.get("features", []):
        props = feature.get("properties", {})
        node = int(props.get("node_index_zero_based", int(props["node_id_1based"]) - 1))
        if not 0 <= node < node_count:
            continue
        chain = int(props["constraint_chain_id"])
        position = int(props["constraint_chain_position"])
        grouped.setdefault(chain, []).append((position, node))
        fixed[node] = True
        hard[node] = bool(props.get("is_hard_anchor", False))
        kinds[node] = str(props.get("boundary_kind", "open" if props.get("is_open_boundary") else "land"))
        source = props.get("source_node_index_zero_based")
        if source is not None:
            lineage[node] = int(source)
    chains = [[node for _, node in sorted(grouped[key])] for key in sorted(grouped)]
    return chains, fixed, kinds, hard, lineage


def _rejected_nodes(path_value: str | None) -> tuple[list[int], list[int]]:
    if not path_value:
        return [], []
    document = json.loads(Path(path_value).read_text(encoding="utf-8-sig"))
    conditioning = document.get("conditioning", document)
    lineage: list[int] = []
    indices: list[int] = []
    for round_doc in conditioning.get("rounds", []):
        stage = round_doc.get("high_valence_repair", {})
        lineage.extend(int(value["node_lineage"]) for value in stage.get("rejected_cases", []) if "node_lineage" in value)
        indices.extend(int(value["node_index_zero_based"]) for value in stage.get("rejected_cases", []) if "node_index_zero_based" in value)
    return list(dict.fromkeys(lineage)), list(dict.fromkeys(indices))


def _select_examples(
    records: list[dict[str, Any]],
    rejected_lineage: list[int],
    rejected_indices: list[int],
    count: int,
) -> list[dict[str, Any]]:
    by_lineage = {int(record["node_lineage"]): record for record in records}
    by_index = {int(record["node_index_zero_based"]): record for record in records}
    selected = [by_index[value] for value in rejected_indices if value in by_index]
    selected.extend(by_lineage[value] for value in rejected_lineage if value in by_lineage and by_lineage[value] not in selected)
    ordering = sorted(
        records,
        key=lambda value: (
            -int(value["valence"]),
            float(value["minimum_incident_quality"]),
            int(value["node_index_zero_based"]),
        ),
    )
    seen = {int(value["node_index_zero_based"]) for value in selected}
    for record in ordering:
        if int(record["node_index_zero_based"]) not in seen:
            selected.append(record)
            seen.add(int(record["node_index_zero_based"]))
        if len(selected) >= count:
            break
    return selected[:count]


def _plot_overview(path: Path, points: np.ndarray, triangles: np.ndarray, records: list[dict[str, Any]], dpi: int) -> None:
    topology = build_edge_topology(len(points), triangles)
    fig, ax = plt.subplots(figsize=(10.5, 8.0), constrained_layout=True)
    boundary_segments = np.asarray([[points[a], points[b]] for a, b in topology.boundary_edges], dtype=float)
    if len(boundary_segments):
        ax.add_collection(LineCollection(boundary_segments / 1000.0, colors="#64748b", linewidths=0.35, alpha=0.55))
    if records:
        nodes = np.asarray([int(record["node_index_zero_based"]) for record in records], dtype=int)
        values = np.asarray([int(record["valence"]) for record in records], dtype=float)
        scatter = ax.scatter(points[nodes, 0] / 1000.0, points[nodes, 1] / 1000.0, c=values, s=18, cmap="plasma", vmin=9, vmax=max(11, float(np.max(values))), edgecolors="black", linewidths=0.2)
        fig.colorbar(scatter, ax=ax, label="Unique-neighbor valence")
    ax.set_title(f"High-valence inventory: {len(records)} nodes above 8")
    ax.set_xlabel("Projected easting (km)")
    ax.set_ylabel("Projected northing (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_local_examples(
    path: Path,
    points: np.ndarray,
    triangles: np.ndarray,
    records: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    graph_rings: int,
    dpi: int,
) -> None:
    topology = build_edge_topology(len(points), triangles)
    valence = np.asarray([len(values) for values in topology.node_neighbors], dtype=int)
    record_nodes = {int(value["node_index_zero_based"]) for value in records}
    columns = 2
    rows = max(1, int(np.ceil(len(selected) / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(12.0, 5.2 * rows), squeeze=False, constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, record in zip(axes.ravel(), selected, strict=False):
        ax.axis("on")
        center = int(record["node_index_zero_based"])
        patch = _graph_patch(topology.node_neighbors, center, graph_rings)
        edges = {
            tuple(sorted((int(a), int(b))))
            for node in patch
            for a, b in ((node, neighbor) for neighbor in topology.node_neighbors[node])
            if int(a) in patch and int(b) in patch
        }
        segments = np.asarray([[points[a], points[b]] for a, b in sorted(edges)], dtype=float)
        origin = points[center]
        if len(segments):
            ax.add_collection(LineCollection((segments - origin) / 1000.0, colors="#94a3b8", linewidths=0.65, alpha=0.9))
        patch_values = np.asarray(sorted(patch), dtype=int)
        ax.scatter((points[patch_values, 0] - origin[0]) / 1000.0, (points[patch_values, 1] - origin[1]) / 1000.0, s=10, c="#475569", zorder=2)
        nearby_bad = np.asarray(sorted(record_nodes & patch), dtype=int)
        if len(nearby_bad):
            ax.scatter((points[nearby_bad, 0] - origin[0]) / 1000.0, (points[nearby_bad, 1] - origin[1]) / 1000.0, s=38, c=valence[nearby_bad], cmap="plasma", vmin=9, vmax=max(11, int(np.max(valence[nearby_bad]))), edgecolors="black", linewidths=0.4, zorder=3)
        ax.scatter([0.0], [0.0], marker="*", s=180, c="#dc2626", edgecolors="black", linewidths=0.6, zorder=4)
        for node in sorted(topology.node_neighbors[center]) + [center]:
            xy = (points[node] - origin) / 1000.0
            ax.text(float(xy[0]), float(xy[1]), f"{node + 1}\nν={valence[node]}", fontsize=6.5, ha="center", va="bottom")
        ax.set_title(
            f"Node {center + 1}: ν={record['valence']}, route={record['repair_route_hint']}\n"
            f"qmin={record['minimum_incident_quality']:.3f}, angle min={record['minimum_incident_angle_deg']:.2f}°"
        )
        ax.set_xlabel("Local easting offset (km)")
        ax.set_ylabel("Local northing offset (km)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.18)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _graph_patch(neighbors: list[set[int]], center: int, rings: int) -> set[int]:
    selected = {int(center)}
    frontier = {int(center)}
    for _ in range(max(0, int(rings))):
        following = {int(value) for node in frontier for value in neighbors[node]} - selected
        selected.update(following)
        frontier = following
        if not frontier:
            break
    return selected


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = list(records[0]) if records else ["node_index_zero_based", "node_id_1based", "valence"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _geojson(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {key: _json_safe(value) for key, value in record.items() if key not in {"longitude", "latitude"}},
                "geometry": {"type": "Point", "coordinates": [float(record["longitude"]), float(record["latitude"])]},
            }
            for record in records
        ],
    }


def _bbox(lonlat: np.ndarray) -> tuple[float, float, float, float]:
    return float(np.min(lonlat[:, 0])), float(np.min(lonlat[:, 1])), float(np.max(lonlat[:, 0])), float(np.max(lonlat[:, 1]))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
