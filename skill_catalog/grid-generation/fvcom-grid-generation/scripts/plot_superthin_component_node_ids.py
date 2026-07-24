#!/usr/bin/env python3
"""Plot a superthin component with plain one-based 2DM node IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from condition_mesh_local import _bbox, _boundary_metadata, _target_sizes  # noqa: E402
from fvcom_grid_generation.local_topology import (  # noqa: E402
    _expand_triangle_patch,
    _inventory_superthin_components,
    _ordered_patch_boundary,
)
from fvcom_grid_generation.metrics import build_edge_topology, chain_edges  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection, project_points  # noqa: E402
from fvcom_grid_generation.sms_2dm import read_2dm  # noqa: E402
from fvcom_grid_generation.visual_superthin import create_visual_state  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--boundary-nodes-geojson", required=True)
    parser.add_argument("--size-field-nc", required=True)
    parser.add_argument("--component-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rings", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--restricted-lineage-edge", nargs=2, type=int, action="append", default=[])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    boundary_path = Path(args.boundary_nodes_geojson)
    size_path = Path(args.size_field_nc)
    output_path = Path(args.output)
    for path in (mesh_path, boundary_path, size_path):
        if not path.is_file():
            raise FileNotFoundError(path)

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
    restrictions = {tuple(sorted(map(int, edge))) for edge in args.restricted_lineage_edge}
    state, config, _ = create_visual_state(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
        boundary_kinds=kinds,
        hard_anchor_mask=hard,
        restricted_lineage_edges=restrictions,
    )
    components = _inventory_superthin_components(state, config)
    matches = [item for item in components if item["component_id"] == args.component_id]
    if len(matches) != 1:
        available = [item["component_id"] for item in components]
        raise ValueError(f"component {args.component_id!r} not found uniquely; available={available}")

    topology = build_edge_topology(len(points), triangles)
    protected = chain_edges(chains)
    open_edges = {
        tuple(sorted((int(a), int(b))))
        for a, b in zip(open_nodes[:-1], open_nodes[1:])
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    node_ids = _plot_plain_node_ids(
        output_path,
        state,
        matches[0],
        topology,
        protected,
        open_edges,
        rings=int(args.rings),
        dpi=int(args.dpi),
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "component_id": str(args.component_id),
                "patch_rings": int(args.rings),
                "node_numbering": "one-based current SMS 2DM node IDs",
                "node_ids": node_ids,
            },
            indent=2,
        )
    )
    return 0


def _plot_plain_node_ids(
    path: Path,
    state: Any,
    component: dict[str, Any],
    topology: Any,
    protected: set[tuple[int, int]],
    open_edges: set[tuple[int, int]],
    *,
    rings: int,
    dpi: int,
) -> list[int]:
    patches = {
        value: _expand_triangle_patch(
            state.triangles, topology, component["triangle_indices"], value
        )
        for value in (1, 2, 4)
    }
    display_patch = patches[rings]
    display_nodes = sorted(set(map(int, np.unique(state.triangles[display_patch]))))
    component_triangles = set(map(int, component["triangle_indices"]))
    component_nodes = set(
        map(
            int,
            np.unique(
                state.triangles[np.asarray(component["triangle_indices"], dtype=int)]
            ),
        )
    )
    coords = state.points[np.asarray(display_nodes, dtype=int)]
    span = np.ptp(coords, axis=0)
    pad = max(float(np.max(span)) * 0.10, 5.0)

    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    for triangle_index in display_patch:
        tri = state.triangles[int(triangle_index)]
        polygon = state.points[np.r_[tri, tri[0]]]
        selected = int(triangle_index) in component_triangles
        ax.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor="#ffcc80" if selected else "#eceff1",
            edgecolor="#e65100" if selected else "#90a4ae",
            linewidth=2.2 if selected else 0.8,
            alpha=0.92 if selected else 0.48,
            zorder=1,
        )

    display_node_set = set(display_nodes)
    for edge in protected:
        if edge[0] not in display_node_set or edge[1] not in display_node_set:
            continue
        values = state.points[np.asarray(edge, dtype=int)]
        ax.plot(
            values[:, 0],
            values[:, 1],
            color="#1565c0" if edge in open_edges else "#111111",
            linewidth=3.0,
            zorder=3,
        )

    for value, linestyle, color in (
        (1, "--", "#00838f"),
        (2, "-.", "#6a1b9a"),
        (4, ":", "#455a64"),
    ):
        if value > rings:
            continue
        ring = _ordered_patch_boundary(state.triangles, patches[value])
        if ring is None:
            continue
        values = state.points[np.asarray([*ring, ring[0]], dtype=int)]
        ax.plot(
            values[:, 0],
            values[:, 1],
            linestyle=linestyle,
            color=color,
            linewidth=1.1,
            label=f"{value}-ring patch",
            zorder=2,
        )

    interior_nodes = [node for node in display_nodes if not state.fixed[node]]
    fixed_nodes = [node for node in display_nodes if state.fixed[node]]
    hard_nodes = [node for node in display_nodes if state.hard[node]]
    if interior_nodes:
        values = state.points[np.asarray(interior_nodes, dtype=int)]
        ax.scatter(
            values[:, 0], values[:, 1], s=28, facecolor="white", edgecolor="#37474f", zorder=4
        )
    if fixed_nodes:
        values = state.points[np.asarray(fixed_nodes, dtype=int)]
        ax.scatter(
            values[:, 0], values[:, 1], marker="s", s=48, color="#26c6da", edgecolor="#006064", zorder=5
        )
    if hard_nodes:
        values = state.points[np.asarray(hard_nodes, dtype=int)]
        ax.scatter(
            values[:, 0], values[:, 1], marker="*", s=150, color="#fdd835", edgecolor="#6d4c41", zorder=6
        )
    if component_nodes:
        values = state.points[np.asarray(sorted(component_nodes), dtype=int)]
        ax.scatter(
            values[:, 0], values[:, 1], s=105, facecolor="none", edgecolor="#d32f2f", linewidth=1.7, zorder=7
        )

    center = np.mean(coords, axis=0)
    bottom_limit = float(np.min(coords[:, 1]) + 0.04 * max(span[1], 1.0))
    bottom_nodes = sorted(
        (node for node in display_nodes if state.points[node, 1] <= bottom_limit),
        key=lambda node: float(state.points[node, 0]),
    )
    bottom_offsets = (18, -20, 36, -38, 54, -56)
    bottom_offset_by_node = {
        node: bottom_offsets[index % len(bottom_offsets)]
        for index, node in enumerate(bottom_nodes)
    }
    for ordinal, node in enumerate(display_nodes):
        delta = state.points[node] - center
        if float(np.linalg.norm(delta)) < 1.0e-12:
            delta = np.asarray([1.0, 1.0])
        dx = 7 if delta[0] >= 0 else -7
        dy = 7 if delta[1] >= 0 else -7
        if node in bottom_offset_by_node:
            dx = 0
            dy = bottom_offset_by_node[node]
        elif ordinal % 2:
            dy *= -1
        ax.annotate(
            str(node + 1),
            xy=(state.points[node, 0], state.points[node, 1]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
            va="bottom" if dy > 0 else "top",
            fontsize=8.5,
            fontweight="bold" if node in component_nodes else "normal",
            color="#b71c1c" if node in component_nodes else "#102027",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.80},
            arrowprops={"arrowstyle": "-", "color": "#78909c", "linewidth": 0.45},
            zorder=8,
        )

    ax.set_xlim(float(np.min(coords[:, 0])) - pad, float(np.max(coords[:, 0])) + pad)
    ax.set_ylim(float(np.min(coords[:, 1])) - pad, float(np.max(coords[:, 1])) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.set_xlabel("UTM easting (m)")
    ax.set_ylabel("UTM northing (m)")
    ax.set_title(
        f"{component['component_id']} — {rings}-ring connectivity map\n"
        "Plain labels are one-based 2DM node IDs; orange triangles are superthin"
    )
    ax.legend(fontsize=8, loc="best")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [node + 1 for node in display_nodes]


if __name__ == "__main__":
    raise SystemExit(main())
