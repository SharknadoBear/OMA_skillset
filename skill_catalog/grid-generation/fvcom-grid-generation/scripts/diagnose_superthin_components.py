#!/usr/bin/env python3
"""Create a visual/numerical atlas of every connected superthin component."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from fvcom_grid_generation.metrics import build_edge_topology, chain_edges, triangle_geometry  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection, project_points  # noqa: E402
from fvcom_grid_generation.sms_2dm import read_2dm  # noqa: E402
from fvcom_grid_generation.visual_superthin import (  # noqa: E402
    create_visual_state,
    preview_action_points,
)


COLORS = {
    "inward_front_support": "#2ca02c",
    "passage_centerline_support": "#9467bd",
    "source_arc_insertion": "#d62728",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--boundary-nodes-geojson", required=True)
    parser.add_argument("--size-field-nc", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conditioning-report")
    parser.add_argument(
        "--node-lineage-json",
        help="Optional JSON list or {'node_lineage': [...]} for source-ID labels.",
    )
    parser.add_argument(
        "--restricted-lineage-json",
        help="Optional JSON list or {'restricted_lineage_edges': [[a,b], ...]}.",
    )
    parser.add_argument(
        "--ring-ladder",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 12],
    )
    parser.add_argument("--restricted-lineage-edge", nargs=2, type=int, action="append", default=[])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    boundary_path = Path(args.boundary_nodes_geojson)
    size_path = Path(args.size_field_nc)
    output_dir = Path(args.output_dir)
    for path in (mesh_path, boundary_path, size_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary_path = output_dir / "component_atlas.json"
    if summary_path.exists() and not bool(args.overwrite):
        raise FileExistsError(summary_path)
    component_dir = output_dir / "components"
    component_dir.mkdir(parents=True, exist_ok=True)
    ring_ladder = tuple(
        sorted(set(max(1, int(value)) for value in args.ring_ladder))
    )
    conditioning_report = (
        json.loads(
            Path(args.conditioning_report).read_text(encoding="utf-8-sig")
        )
        if args.conditioning_report
        else None
    )
    node_lineage = None
    if args.node_lineage_json:
        lineage_document = json.loads(
            Path(args.node_lineage_json).read_text(encoding="utf-8-sig")
        )
        if isinstance(lineage_document, dict):
            lineage_document = lineage_document["node_lineage"]
        node_lineage = np.asarray(lineage_document, dtype=int)

    mesh = read_2dm(mesh_path)
    projection = local_utm_projection(_bbox(mesh.nodes_lonlat))
    points = project_points(mesh.nodes_lonlat, projection)
    triangles = np.asarray(mesh.triangles, dtype=int) - 1
    open_nodes = np.asarray(mesh.open_boundary_nodes, dtype=int) - 1
    chains, kinds, hard, explicit_targets = _boundary_metadata(
        len(points),
        triangles,
        open_nodes,
        str(boundary_path),
        None,
    )
    fixed = np.zeros(len(points), dtype=bool)
    for chain in chains:
        fixed[np.asarray(chain, dtype=int)] = True
    targets = _target_sizes(mesh.nodes_lonlat, points, triangles, str(size_path))
    explicit = np.isfinite(explicit_targets) & (explicit_targets > 0.0)
    targets[explicit] = explicit_targets[explicit]
    restrictions = {
        tuple(sorted(map(int, edge))) for edge in args.restricted_lineage_edge
    }
    if args.restricted_lineage_json:
        restriction_document = json.loads(
            Path(args.restricted_lineage_json).read_text(encoding="utf-8-sig")
        )
        if isinstance(restriction_document, dict):
            restriction_document = restriction_document[
                "restricted_lineage_edges"
            ]
        restrictions.update(
            tuple(sorted(map(int, edge))) for edge in restriction_document
        )
    if node_lineage is not None and len(node_lineage) != len(points):
        raise ValueError(
            "node-lineage JSON must contain one value per delivered mesh node"
        )
    state, config, _ = create_visual_state(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
        boundary_kinds=kinds,
        hard_anchor_mask=hard,
        node_lineage=node_lineage,
        restricted_lineage_edges=restrictions,
    )
    components = _inventory_superthin_components(state, config)
    geometry = triangle_geometry(points, triangles)
    topology = build_edge_topology(len(points), triangles)
    protected = chain_edges(chains)
    open_edges = {
        tuple(sorted((int(a), int(b))))
        for a, b in zip(open_nodes[:-1], open_nodes[1:])
    }
    valence = np.asarray([len(values) for values in topology.node_neighbors], dtype=int)
    records: list[dict[str, Any]] = []
    for rank, component in enumerate(components, start=1):
        preview = preview_action_points(state, component, patch_rings=2)
        record = {
            key: value
            for key, value in component.items()
            if key != "triangle_indices"
        }
        record.update(
            {
                "rank": int(rank),
                "triangle_indices_zero_based": list(map(int, component["triangle_indices"])),
                "triangle_ids_1based": [int(value) + 1 for value in component["triangle_indices"]],
                "maximum_local_valence": int(
                    np.max(
                        valence[
                            np.asarray(
                                sorted(
                                    set(
                                        map(
                                            int,
                                            np.unique(triangles[np.asarray(component["triangle_indices"], dtype=int)]),
                                        )
                                    )
                                ),
                                dtype=int,
                            )
                        ]
                    )
                ),
                "preview_candidate_points_xy": preview,
                "recommended_tool_sequence": _route(component["classification"]),
                "failure_evidence": _component_failure_evidence(
                    conditioning_report,
                    str(component["component_id"]),
                ),
                "image": str(component_dir / f"{rank:02d}_{component['component_id']}.png"),
            }
        )
        _plot_component(
            Path(record["image"]),
            state,
            component,
            geometry,
            topology,
            protected,
            open_edges,
            valence,
            preview,
            ring_ladder=ring_ladder,
            dpi=int(args.dpi),
        )
        records.append(record)
    document = {
        "schema_version": "fvcom_conditioning_component_atlas_v1",
        "mesh": str(mesh_path),
        "mesh_sha256": _sha256(mesh_path),
        "boundary_nodes_geojson": str(boundary_path),
        "size_field_nc": str(size_path),
        "projection_epsg": int(projection.epsg),
        "superthin_definition": "q < 0.10 or minimum angle < 5 degrees",
        "component_count": int(len(records)),
        "superthin_triangle_count": int(sum(item["triangle_count"] for item in records)),
        "restricted_lineage_edges": [list(map(int, edge)) for edge in sorted(restrictions)],
        "ring_ladder": list(map(int, ring_ladder)),
        "conditioning_report": str(args.conditioning_report or ""),
        "node_lineage_json": str(args.node_lineage_json or ""),
        "restricted_lineage_json": str(
            args.restricted_lineage_json or ""
        ),
        "components": records,
        "visual_review_required": bool(records),
        "deterministic_classification_is_not_visual_review": True,
    }
    summary_path.write_text(json.dumps(_json_safe(document), indent=2), encoding="utf-8")
    _write_csv(output_dir / "component_atlas.csv", records)
    (output_dir / "visual_repair_plan_templates.json").write_text(
        json.dumps(
            [_plan_template(document, item) for item in records],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "component_count": len(records),
                "superthin_triangle_count": document["superthin_triangle_count"],
                "component_images": [item["image"] for item in records],
            },
            indent=2,
        )
    )
    return 0


def _plot_component(
    path: Path,
    state: Any,
    component: dict[str, Any],
    geometry: dict[str, np.ndarray],
    topology: Any,
    protected: set[tuple[int, int]],
    open_edges: set[tuple[int, int]],
    valence: np.ndarray,
    preview: dict[str, list[list[float]]],
    *,
    ring_ladder: tuple[int, ...],
    dpi: int,
) -> None:
    patches: dict[int, np.ndarray] = {}
    for rings in ring_ladder:
        patches[rings] = _expand_triangle_patch(
            state.triangles,
            topology,
            component["triangle_indices"],
            rings,
        )
    display_patch = patches[max(ring_ladder)]
    display_nodes = sorted(set(map(int, np.unique(state.triangles[display_patch]))))
    coords = state.points[np.asarray(display_nodes, dtype=int)]
    span = np.ptp(coords, axis=0)
    pad = max(float(np.max(span)) * 0.12, 5.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), constrained_layout=True)
    for ax, field in zip(axes, ("quality", "minimum_angle", "target")):
        for triangle_index in display_patch:
            tri = state.triangles[int(triangle_index)]
            polygon = state.points[np.r_[tri, tri[0]]]
            color = "#ef6c00" if int(triangle_index) in set(component["triangle_indices"]) else "#b0bec5"
            width = 2.4 if int(triangle_index) in set(component["triangle_indices"]) else 0.6
            ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=width, zorder=2)
        for edge in protected:
            if edge[0] not in display_nodes or edge[1] not in display_nodes:
                continue
            values = state.points[np.asarray(edge, dtype=int)]
            ax.plot(
                values[:, 0],
                values[:, 1],
                color="#1565c0" if edge in open_edges else "#111111",
                linewidth=2.2,
                zorder=3,
            )
        styles = ("--", "-.", ":", (0, (3, 1, 1, 1)), (0, (1, 1)))
        colors = ("#00838f", "#6a1b9a", "#455a64", "#2e7d32", "#ad1457")
        for rings, linestyle, color in zip(ring_ladder, styles, colors):
            ring = _ordered_patch_boundary(state.triangles, patches[rings])
            if ring is None:
                continue
            values = state.points[np.asarray([*ring, ring[0]], dtype=int)]
            ax.plot(values[:, 0], values[:, 1], linestyle=linestyle, color=color, linewidth=1.0, label=f"{rings}-ring")
        fixed_nodes = [node for node in display_nodes if state.fixed[node]]
        hard_nodes = [node for node in display_nodes if state.hard[node]]
        if fixed_nodes:
            values = state.points[np.asarray(fixed_nodes, dtype=int)]
            ax.scatter(values[:, 0], values[:, 1], marker="s", s=22, color="#26c6da", zorder=4)
        if hard_nodes:
            values = state.points[np.asarray(hard_nodes, dtype=int)]
            ax.scatter(values[:, 0], values[:, 1], marker="*", s=90, color="#fdd835", edgecolors="#6d4c41", zorder=5)
        for node in sorted(set(map(int, np.unique(state.triangles[np.asarray(component["triangle_indices"], dtype=int)])))):
            ax.text(
                state.points[node, 0],
                state.points[node, 1],
                (
                    f"{int(state.lineage[node]) + 1}"
                    if int(state.lineage[node]) >= 0
                    else f"new:{int(state.lineage[node])}"
                )
                + f"\nν={valence[node]}",
                fontsize=7,
                color="#4a148c",
                zorder=6,
            )
        for tool, points in preview.items():
            if not points:
                continue
            values = np.asarray(points, dtype=float)
            ax.scatter(
                values[:, 0],
                values[:, 1],
                marker="x" if tool != "source_arc_insertion" else "+",
                s=42,
                color=COLORS[tool],
                label=tool.replace("_", " "),
                zorder=7,
            )
        ax.set_xlim(float(np.min(coords[:, 0])) - pad, float(np.max(coords[:, 0])) + pad)
        ax.set_ylim(float(np.min(coords[:, 1])) - pad, float(np.max(coords[:, 1])) + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.set_xlabel("UTM easting (m)")
        if field == "quality":
            ax.set_title(
                f"qmin={component['minimum_quality']:.5f}; severity={component['severity']:.3f}\n"
                "orange = selected superthin component"
            )
        elif field == "minimum_angle":
            ax.set_title(
                f"minimum angle={component['minimum_angle_deg']:.3f}°\n"
                f"fixed={component['fixed_node_count']}; hard={component['hard_anchor_count']}"
            )
        else:
            ax.set_title(
                f"class={component['classification']}\n"
                f"passage={component.get('passage_width_m')} m; gap/h={component.get('gap_over_h')}"
            )
        ax.legend(fontsize=6, loc="best")
    fig.suptitle(
        f"{component['component_id']} — {component['triangle_count']} superthin triangle(s)",
        fontsize=14,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _component_failure_evidence(
    document: Any,
    component_id: str,
) -> list[dict[str, Any]]:
    """Collect gate failures attached to one component from a V6 report."""
    output: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            matches = str(value.get("component_id", "")) == str(component_id)
            failures = value.get("failures", value.get("rejection_gates", []))
            if matches and failures:
                output.append(
                    {
                        "path": path,
                        "status": str(value.get("status", "")),
                        "failures": list(map(str, failures)),
                    }
                )
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    if document is not None:
        visit(document, "")
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for value in output:
        key = (str(value["status"]), tuple(value["failures"]))
        if key not in seen:
            unique.append(value)
            seen.add(key)
    return unique


def _route(classification: str) -> list[str]:
    routes = {
        "fixed-boundary-hard-anchor-fan": [
            "inward_front_support",
            "constrained_retriangulation",
            "source_arc_insertion",
        ],
        "fixed-boundary-fan": [
            "constrained_retriangulation",
            "inward_front_support",
            "source_arc_insertion",
        ],
        "under-resolved-passage": [
            "passage_centerline_support",
            "constrained_retriangulation",
        ],
        "interior-connectivity-transition": [
            "constrained_retriangulation",
            "inward_front_support",
        ],
    }
    return routes.get(str(classification), ["constrained_retriangulation"])


def _plan_template(atlas: dict[str, Any], component: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "fvcom_visual_superthin_repair_plan_v1",
        "input_mesh_sha256": atlas["mesh_sha256"],
        "review": {
            "status": "pending_visual_review",
            "reviewed_by": "",
            "reviewed_at_utc": "",
            "manageable": False,
            "visual_evidence": [component["image"]],
            "observations": "",
        },
        "component": {
            "component_id": component["component_id"],
            "classification": component["classification"],
            "node_lineage": component["node_lineage"],
        },
        "actions": [
            {
                "tool": tool,
                "patch_rings": [1, 2, 4],
                "maximum_support_nodes": 2,
                "local_relaxation": True,
            }
            for tool in component["recommended_tool_sequence"]
        ],
        "restricted_lineage_edges": atlas["restricted_lineage_edges"],
        "acceptance": {
            "require_strict_superthin_reduction": True,
            "allow_valence_change": True,
            "preserve_quality_tails": True,
            "preserve_existing_boundary_coordinates": True,
        },
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "component_id",
        "classification",
        "triangle_count",
        "minimum_quality",
        "minimum_angle_deg",
        "severity",
        "fixed_node_count",
        "hard_anchor_count",
        "passage_width_m",
        "gap_over_h",
        "maximum_local_valence",
        "image",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow({key: item.get(key) for key in fields})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
