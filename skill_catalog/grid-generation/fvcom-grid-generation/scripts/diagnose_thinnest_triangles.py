#!/usr/bin/env python3
"""Create read-only zoom diagnostics for the thinnest triangles in an FVCOM 2DM mesh."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import condition_mesh_local as mesh_cli  # noqa: E402
from fvcom_grid_generation.local_topology import _edge_target  # noqa: E402
from fvcom_grid_generation.metrics import build_edge_topology, chain_edges, triangle_geometry  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection, project_points  # noqa: E402
from fvcom_grid_generation.regional_conditioning import (  # noqa: E402
    ThinTriangleRepairConfig,
    _edge_flip_candidate,
    _split_improves_patch,
)
from fvcom_grid_generation.sms_2dm import read_2dm  # noqa: E402


def _triangle_edges(triangle: np.ndarray) -> list[tuple[int, int]]:
    a, b, c = map(int, triangle)
    return [tuple(sorted(edge)) for edge in ((a, b), (b, c), (c, a))]


def _ring_patch(topology: Any, triangles: np.ndarray, seed_nodes: set[int], rings: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = set(map(int, seed_nodes))
    frontier = set(nodes)
    for _ in range(max(0, int(rings))):
        following = {
            int(neighbor)
            for node in frontier
            for neighbor in topology.node_neighbors[int(node)]
            if int(neighbor) not in nodes
        }
        nodes.update(following)
        frontier = following
        if not frontier:
            break
    triangle_ids = np.where(
        np.asarray([any(int(node) in nodes for node in tri) for tri in triangles], dtype=bool)
    )[0]
    return triangle_ids, np.asarray(sorted(nodes), dtype=int)


def _context_classification(
    *,
    fixed_count: int,
    hard_count: int,
    protected_count: int,
    patch_kinds: set[str],
    patch_chain_count: int,
    flip_eligible: list[list[int]],
    split_eligible: list[list[int]],
) -> tuple[str, str]:
    normalized = {str(value).lower() for value in patch_kinds}
    if "open" in normalized and "land" in normalized:
        return (
            "land_open_boundary_junction",
            "Defer to the authorized junction-size review; preserve exact landfall and OBC order.",
        )
    if patch_chain_count >= 2:
        return (
            "narrow_passage_or_island_gap",
            "Review whether the wet gap is resolvable at the target size before any connectivity edit.",
        )
    if protected_count and hard_count:
        return (
            "protected_boundary_hard_anchor_fan",
            "Do not delete hard anchors; audit anchor density or add/reposition the first inward front node.",
        )
    if protected_count:
        return (
            "protected_boundary_sliver",
            "Use a guarded boundary ear removal, source-arc weld, or boundary split; a protected edge cannot be flipped.",
        )
    if fixed_count:
        return (
            "boundary_front_transition",
            "Rebalance the first inward seed/front or use a local unprotected flip without moving the source boundary.",
        )
    if flip_eligible:
        return "interior_flip_candidate", "Prefer the legal local edge flip and re-audit valence and q-tail atomically."
    if split_eligible:
        return "interior_split_candidate", "Split the long interior edge only if the local q-and-angle patch improves."
    return (
        "interior_connectivity_or_size_transition",
        "The current local operations are blocked; revisit local seeding/target transition before broader smoothing.",
    )


def _plot_zoom(
    ax: Any,
    *,
    record: dict[str, Any],
    points: np.ndarray,
    triangles: np.ndarray,
    topology: Any,
    geometry: dict[str, np.ndarray],
    selected_mask: np.ndarray,
    protected: set[tuple[int, int]],
    fixed: np.ndarray,
    hard: np.ndarray,
    kinds: list[str],
    rings: int,
    legend: bool,
) -> None:
    index = int(record["triangle_index_zero_based"])
    tri = np.asarray(triangles[index], dtype=int)
    patch_triangles, patch_nodes = _ring_patch(topology, triangles, set(map(int, tri)), rings)
    x0, y0 = np.mean(points[tri], axis=0)
    local_edges = np.asarray(geometry["edge_lengths"])[patch_triangles]
    local_scale = max(
        float(np.max(geometry["edge_lengths"][index])),
        float(np.median(local_edges)) if local_edges.size else 1.0,
    )
    radius = max(2.8 * local_scale, 40.0)

    for patch_index in patch_triangles:
        polygon = points[np.r_[triangles[patch_index], triangles[patch_index, 0]]]
        other_selected = bool(selected_mask[int(patch_index)] and int(patch_index) != index)
        ax.plot(
            polygon[:, 0],
            polygon[:, 1],
            color="#e67e22" if other_selected else "#aeb6bf",
            linewidth=1.3 if other_selected else 0.55,
            zorder=1,
        )
    target_polygon = points[np.r_[tri, tri[0]]]
    ax.fill(target_polygon[:, 0], target_polygon[:, 1], color="#e74c3c", alpha=0.36, zorder=2)
    ax.plot(target_polygon[:, 0], target_polygon[:, 1], color="#c0392b", linewidth=2.4, zorder=3)

    patch_node_set = set(map(int, patch_nodes.tolist()))
    for edge in protected:
        if edge[0] not in patch_node_set and edge[1] not in patch_node_set:
            continue
        xy = points[list(edge)]
        kind = str(kinds[int(edge[0])]).lower()
        ax.plot(xy[:, 0], xy[:, 1], color="#1565c0" if kind == "open" else "#111111", linewidth=2.0, zorder=4)
    fixed_patch = patch_nodes[fixed[patch_nodes]]
    if len(fixed_patch):
        ax.scatter(
            points[fixed_patch, 0],
            points[fixed_patch, 1],
            marker="s",
            s=17,
            c="#26c6da",
            edgecolors="#006064",
            linewidths=0.4,
            zorder=5,
        )
    hard_patch = patch_nodes[hard[patch_nodes]]
    if len(hard_patch):
        ax.scatter(
            points[hard_patch, 0],
            points[hard_patch, 1],
            marker="*",
            s=78,
            c="#fdd835",
            edgecolors="#5d4037",
            linewidths=0.5,
            zorder=6,
        )
    for node in tri:
        ax.annotate(
            str(int(node) + 1),
            points[int(node)],
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color="#4a148c",
            zorder=7,
        )

    ax.set_xlim(x0 - radius, x0 + radius)
    ax.set_ylim(y0 - radius, y0 + radius)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#eeeeee", linewidth=0.4)
    ax.set_title(
        f"#{record['rank_by_quality']}  T{record['triangle_id_1based']}  "
        f"q={record['quality']:.3g}, min={record['minimum_angle_deg']:.3g}°\n"
        f"{record['context_class']}",
        fontsize=9,
    )
    ax.text(
        0.02,
        0.02,
        record["recommended_route"],
        transform=ax.transAxes,
        fontsize=6.7,
        va="bottom",
        ha="left",
        wrap=True,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb", "pad": 2.0},
    )
    ax.set_xlabel("UTM easting (m)", fontsize=7)
    ax.set_ylabel("UTM northing (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    if legend:
        ax.plot([], [], color="#c0392b", linewidth=3, label="target triangle")
        ax.plot([], [], color="#e67e22", linewidth=2, label="other selected triangle")
        ax.plot([], [], color="#1565c0", linewidth=2, label="protected open boundary")
        ax.plot([], [], color="#111111", linewidth=2, label="other protected boundary")
        ax.scatter([], [], marker="s", s=20, c="#26c6da", label="fixed boundary node")
        ax.scatter([], [], marker="*", s=55, c="#fdd835", label="hard anchor")
        ax.legend(fontsize=6, loc="upper right")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="Input SMS 2DM mesh.")
    parser.add_argument("--boundary-nodes-geojson", required=True, help="Delivered boundary-node metadata GeoJSON.")
    parser.add_argument("--size-field", required=True, help="Size-field NetCDF used by the mesh.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--rings", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    individual_dir = output_dir / "individual_zooms"
    individual_dir.mkdir(parents=True, exist_ok=True)
    mesh = read_2dm(args.mesh)
    projection = local_utm_projection(mesh_cli._bbox(mesh.nodes_lonlat))
    points = project_points(mesh.nodes_lonlat, projection)
    triangles = np.asarray(mesh.triangles, dtype=int) - 1
    open_nodes = np.asarray(mesh.open_boundary_nodes, dtype=int) - 1
    chains, kinds, hard, explicit_targets = mesh_cli._boundary_metadata(
        len(points), triangles, open_nodes, args.boundary_nodes_geojson, None
    )
    fixed = np.zeros(len(points), dtype=bool)
    node_chains: list[set[int]] = [set() for _ in range(len(points))]
    for chain_index, chain in enumerate(chains):
        nodes = np.asarray(chain, dtype=int)
        fixed[nodes] = True
        for node in nodes:
            node_chains[int(node)].add(int(chain_index))
    targets = mesh_cli._target_sizes(mesh.nodes_lonlat, points, triangles, args.size_field)
    explicit = np.isfinite(explicit_targets) & (explicit_targets > 0.0)
    targets[explicit] = explicit_targets[explicit]

    geometry = triangle_geometry(points, triangles)
    minimum_angles = np.min(geometry["angles_deg"], axis=1)
    ordering = np.asarray(
        sorted(
            range(len(triangles)),
            key=lambda index: (float(geometry["quality"][index]), float(minimum_angles[index]), int(index)),
        )[: max(1, int(args.count))],
        dtype=int,
    )
    selected_mask = np.zeros(len(triangles), dtype=bool)
    selected_mask[ordering] = True
    topology = build_edge_topology(len(points), triangles)
    protected = chain_edges(chains)

    records: list[dict[str, Any]] = []
    for rank, index in enumerate(ordering, start=1):
        tri = np.asarray(triangles[int(index)], dtype=int)
        tri_edges = _triangle_edges(tri)
        patch_triangles, patch_nodes = _ring_patch(topology, triangles, set(map(int, tri)), args.rings)
        patch_kinds = {str(kinds[int(node)]) for node in patch_nodes if fixed[int(node)]}
        patch_chains = {chain_index for node in patch_nodes for chain_index in node_chains[int(node)]}
        interchain_pairs: list[tuple[float, int, int, float]] = []
        fixed_patch_nodes = [int(node) for node in patch_nodes if fixed[int(node)]]
        for first_offset, first in enumerate(fixed_patch_nodes):
            for second in fixed_patch_nodes[first_offset + 1 :]:
                if not node_chains[first] or not node_chains[second] or node_chains[first] & node_chains[second]:
                    continue
                distance = float(np.linalg.norm(points[first] - points[second]))
                target = float(_edge_target(targets, (first, second)))
                interchain_pairs.append((distance, first, second, distance / max(target, 1.0e-12)))
        minimum_interchain = min(interchain_pairs, default=None, key=lambda value: value[0])
        flip_eligible: list[list[int]] = []
        split_eligible: list[list[int]] = []
        edge_records: list[dict[str, Any]] = []
        for edge in tri_edges:
            attached = list(map(int, topology.edge_to_triangles.get(edge, [])))
            length = float(np.linalg.norm(points[edge[0]] - points[edge[1]]))
            target = float(_edge_target(targets, edge))
            l_over_h = length / max(target, 1.0e-12)
            is_protected = edge in protected
            if is_protected:
                flip_status = "blocked: protected edge"
            elif len(attached) != 2:
                flip_status = f"blocked: {len(attached)} attached triangle(s)"
            else:
                candidate = _edge_flip_candidate(points, triangles, edge, attached[0], attached[1])
                if candidate is None:
                    flip_status = "blocked: invalid/nonconvex/duplicate flip"
                else:
                    _, _, old_q, new_q, old_angle, new_angle, _ = candidate
                    if new_q > old_q + 1.0e-8 and new_angle > old_angle + 0.05:
                        flip_status = (
                            f"eligible: q {old_q:.4g}->{new_q:.4g}; "
                            f"angle {old_angle:.4g}->{new_angle:.4g}"
                        )
                        flip_eligible.append([int(edge[0]) + 1, int(edge[1]) + 1])
                    else:
                        flip_status = (
                            f"blocked: joint improvement q {old_q:.4g}->{new_q:.4g}; "
                            f"angle {old_angle:.4g}->{new_angle:.4g}"
                        )
            if is_protected:
                split_status = "blocked: protected edge"
            elif len(attached) != 2:
                split_status = f"blocked: {len(attached)} attached triangle(s)"
            elif length <= 1.25 * target:
                split_status = f"blocked: L/h={l_over_h:.4g} <= 1.25"
            elif _split_improves_patch(points, triangles, edge, attached, ThinTriangleRepairConfig()):
                split_status = f"eligible: L/h={l_over_h:.4g} and patch improves"
                split_eligible.append([int(edge[0]) + 1, int(edge[1]) + 1])
            else:
                split_status = f"blocked: midpoint split does not improve patch (L/h={l_over_h:.4g})"
            edge_records.append(
                {
                    "node_ids_1based": [int(edge[0]) + 1, int(edge[1]) + 1],
                    "length_m": length,
                    "target_m": target,
                    "l_over_h": l_over_h,
                    "protected": bool(is_protected),
                    "attached_triangle_ids_1based": [value + 1 for value in attached],
                    "flip_status": flip_status,
                    "split_status": split_status,
                }
            )
        fixed_count = int(np.count_nonzero(fixed[tri]))
        hard_count = int(np.count_nonzero(hard[tri]))
        protected_count = int(sum(edge in protected for edge in tri_edges))
        context, route = _context_classification(
            fixed_count=fixed_count,
            hard_count=hard_count,
            protected_count=protected_count,
            patch_kinds=patch_kinds,
            patch_chain_count=len(patch_chains),
            flip_eligible=flip_eligible,
            split_eligible=split_eligible,
        )
        records.append(
            {
                "rank_by_quality": int(rank),
                "triangle_index_zero_based": int(index),
                "triangle_id_1based": int(index) + 1,
                "node_ids_1based": [int(node) + 1 for node in tri],
                "centroid_lon": float(np.mean(mesh.nodes_lonlat[tri, 0])),
                "centroid_lat": float(np.mean(mesh.nodes_lonlat[tri, 1])),
                "quality": float(geometry["quality"][index]),
                "minimum_angle_deg": float(minimum_angles[index]),
                "area_m2": float(geometry["area"][index]),
                "edge_lengths_m": [float(value) for value in geometry["edge_lengths"][index]],
                "maximum_l_over_h": float(max(value["l_over_h"] for value in edge_records)),
                "fixed_node_count": fixed_count,
                "hard_anchor_count": hard_count,
                "protected_edge_count": protected_count,
                "triangle_neighbor_count": int(topology.triangle_neighbor_count[index]),
                "patch_boundary_kinds": sorted(patch_kinds),
                "patch_constraint_chain_count": int(len(patch_chains)),
                "minimum_patch_interchain_distance_m": (
                    float(minimum_interchain[0]) if minimum_interchain is not None else None
                ),
                "minimum_patch_interchain_node_ids_1based": (
                    [int(minimum_interchain[1]) + 1, int(minimum_interchain[2]) + 1]
                    if minimum_interchain is not None
                    else []
                ),
                "minimum_patch_interchain_gap_over_target": (
                    float(minimum_interchain[3]) if minimum_interchain is not None else None
                ),
                "patch_triangle_count": int(len(patch_triangles)),
                "flip_eligible_edges": flip_eligible,
                "split_eligible_edges": split_eligible,
                "context_class": context,
                "recommended_route": route,
                "edges": edge_records,
            }
        )

    diagnostic = {
        "schema_version": "fvcom_thinnest_triangle_diagnostic_v2",
        "mesh": str(Path(args.mesh)),
        "selection": f"lowest {len(records)} triangles ordered by equilateral quality then minimum angle",
        "triangle_count": len(records),
        "superthin_definition": "q < 0.10 or minimum angle < 5 degrees",
        "selected_superthin_count": int(
            sum(record["quality"] < 0.10 or record["minimum_angle_deg"] < 5.0 for record in records)
        ),
        "context_class_counts": {
            key: int(sum(record["context_class"] == key for record in records))
            for key in sorted({record["context_class"] for record in records})
        },
        "triangles": records,
    }
    (output_dir / "thinnest_triangles_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "thinnest_triangles_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "rank_by_quality",
            "triangle_id_1based",
            "node_ids_1based",
            "centroid_lon",
            "centroid_lat",
            "quality",
            "minimum_angle_deg",
            "area_m2",
            "maximum_l_over_h",
            "fixed_node_count",
            "hard_anchor_count",
            "protected_edge_count",
            "patch_constraint_chain_count",
            "context_class",
            "recommended_route",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})

    column_count = min(3, len(records))
    row_count = int(np.ceil(len(records) / column_count))
    fig, axes = plt.subplots(row_count, column_count, figsize=(5.0 * column_count, 4.6 * row_count), constrained_layout=True)
    axes_array = np.atleast_1d(axes).ravel()
    for panel, record in zip(axes_array, records):
        _plot_zoom(
            panel,
            record=record,
            points=points,
            triangles=triangles,
            topology=topology,
            geometry=geometry,
            selected_mask=selected_mask,
            protected=protected,
            fixed=fixed,
            hard=hard,
            kinds=kinds,
            rings=args.rings,
            legend=record["rank_by_quality"] == 1,
        )
    for panel in axes_array[len(records) :]:
        panel.axis("off")
    fig.suptitle(
        f"FVCOM mesh: zoomed diagnosis of the {len(records)} thinnest triangles\n"
        "Red = selected element; node labels are 1-based SMS IDs",
        fontsize=15,
    )
    fig.savefig(output_dir / "thinnest_triangles_zoom_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    for record in records:
        fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
        _plot_zoom(
            ax,
            record=record,
            points=points,
            triangles=triangles,
            topology=topology,
            geometry=geometry,
            selected_mask=selected_mask,
            protected=protected,
            fixed=fixed,
            hard=hard,
            kinds=kinds,
            rings=args.rings,
            legend=True,
        )
        fig.savefig(
            individual_dir / f"rank_{record['rank_by_quality']:02d}_triangle_{record['triangle_id_1based']}.png",
            dpi=240,
            bbox_inches="tight",
        )
        plt.close(fig)
    print(json.dumps({"output_dir": str(output_dir), "context_class_counts": diagnostic["context_class_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
