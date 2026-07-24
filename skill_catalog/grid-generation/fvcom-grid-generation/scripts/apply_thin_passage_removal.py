#!/usr/bin/env python3
"""Apply reviewed whole-passage removals to an FVCOM 2DM transactionally."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from condition_mesh_local import (  # noqa: E402
    _bbox,
    _boundary_geojson,
    _boundary_metadata,
    _json_safe,
    _remap_depths,
    _serialized_roundtrip_audit,
    _target_sizes,
)
from fvcom_grid_generation.local_topology import (  # noqa: E402
    _expand_triangle_patch,
    _inventory_superthin_components,
)
from fvcom_grid_generation.metrics import build_edge_topology, chain_edges, triangle_geometry  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection, project_points, unproject_points  # noqa: E402
from fvcom_grid_generation.sms_2dm import read_2dm, write_2dm  # noqa: E402
from fvcom_grid_generation.thin_passage import (  # noqa: E402
    REPORT_SCHEMA,
    ThinPassageRemovalConfig,
    config_as_dict,
    infer_passage_removal_candidates,
    try_remove_thin_passage,
)
from fvcom_grid_generation.visual_superthin import create_visual_state  # noqa: E402


PLAN_SCHEMA = "fvcom_thin_passage_removal_plan_v1"
RUN_SCHEMA = "fvcom_thin_passage_removal_run_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--boundary-nodes-geojson", required=True)
    parser.add_argument("--size-field-nc", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    boundary_path = Path(args.boundary_nodes_geojson)
    size_path = Path(args.size_field_nc)
    plan_path = Path(args.plan)
    run_dir = Path(args.run_dir)
    for path in (mesh_path, boundary_path, size_path, plan_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    _validate_plan(plan, mesh_path)
    summary_path = run_dir / "passage_removal_report.json"
    if summary_path.exists() and not bool(args.overwrite):
        raise FileExistsError(summary_path)
    run_dir.mkdir(parents=True, exist_ok=True)

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
    restrictions = {
        tuple(sorted(map(int, edge)))
        for edge in plan.get("restricted_lineage_edges", [])
    }
    state, conditioning, _ = create_visual_state(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
        boundary_kinds=kinds,
        hard_anchor_mask=hard,
        node_lineage=np.arange(len(points), dtype=int),
        restricted_lineage_edges=restrictions,
    )
    config = ThinPassageRemovalConfig(**plan.get("config", {}))
    records: list[dict[str, Any]] = []
    all_accepted = True

    for transaction_index, transaction in enumerate(plan["transactions"], start=1):
        before_state = state.clone()
        component_id = str(transaction["component_id"])
        components = _inventory_superthin_components(before_state, conditioning)
        matching = [item for item in components if str(item["component_id"]) == component_id]
        if len(matching) != 1:
            raise ValueError(f"transaction component is absent or stale: {component_id}")
        component = matching[0]
        mode = str(transaction["mode"])
        if mode == "human_approved_nodes":
            node_ids = list(map(int, transaction["remove_node_ids_1based_source"]))
            candidates = [[value - 1 for value in node_ids]]
            inference = {
                "mode": mode,
                "human_example": True,
                "approved_node_ids_1based_source": node_ids,
            }
            human_approved = True
        elif mode == "infer_resolution_cluster":
            candidates, inference = infer_passage_removal_candidates(
                before_state,
                component,
                config=config,
            )
            inference["mode"] = mode
            human_approved = bool(transaction.get("human_approved_topology_change", False))
        else:
            raise ValueError(f"unsupported passage-removal mode: {mode}")

        state, report = try_remove_thin_passage(
            before_state,
            component_id,
            candidates,
            expected_boundary_component_delta=transaction.get(
                "expected_boundary_component_delta"
            ),
            expected_wet_component_delta=transaction.get(
                "expected_wet_component_delta"
            ),
            human_approved=human_approved,
            inference_evidence=inference,
            config=config,
        )
        report["transaction_index"] = int(transaction_index)
        report["transaction_label"] = str(transaction.get("label", component_id))
        accepted = bool(report["accepted"])
        all_accepted = bool(all_accepted and accepted)
        if not accepted:
            records.append(report)
            break

        selected = report["attempts"][int(report["selected_candidate_index"])]
        checkpoint_stem = str(
            transaction.get(
                "checkpoint_stem",
                f"checkpoint_{transaction_index:02d}_{component_id}",
            )
        )
        checkpoint = _write_checkpoint(
            run_dir,
            checkpoint_stem,
            mesh,
            points,
            state,
            projection,
        )
        report["checkpoint"] = checkpoint
        if not bool(checkpoint["serialized_roundtrip"]["passed"]):
            report["accepted"] = False
            report["status"] = "roundtrip_failure"
            report["attempts"][int(report["selected_candidate_index"])]["failures"].append(
                "serialized_roundtrip_failure"
            )
            state = before_state
            all_accepted = False
            records.append(report)
            break

        plot_path = Path(str(transaction["output_plot"]))
        report["output_plot"] = str(plot_path)
        transaction_report = run_dir / f"{checkpoint_stem}_audit.json"
        report["audit_json"] = str(transaction_report)
        transaction_report.write_text(
            json.dumps(_json_safe(report), indent=2), encoding="utf-8"
        )
        if not bool(args.skip_plots):
            _plot_transaction(
                plot_path,
                before_state,
                state,
                component,
                selected,
                report,
                config=config,
                dpi=int(transaction.get("dpi", 300)),
            )
        report["plot_rendered_in_this_run"] = bool(not args.skip_plots)
        transaction_report.write_text(
            json.dumps(_json_safe(report), indent=2), encoding="utf-8"
        )
        records.append(report)
        gc.collect()

    final = _write_checkpoint(
        run_dir,
        "final_passage_removed",
        mesh,
        points,
        state,
        projection,
    )
    final_summary = records[-1]["after"] if records else {}
    zero_superthin = bool(
        all_accepted and int(final_summary.get("superthin_triangle_count", -1)) == 0
    )
    model_ready = bool(
        zero_superthin
        and int(final_summary.get("count_valence_above_limit", 1)) == 0
        and float(final_summary.get("q_l3_sigma", 0.0)) > 0.75
    )
    document = {
        "schema_version": RUN_SCHEMA,
        "operator_report_schema": REPORT_SCHEMA,
        "input_mesh": str(mesh_path),
        "input_mesh_sha256": _sha256(mesh_path),
        "boundary_nodes_geojson": str(boundary_path),
        "size_field_nc": str(size_path),
        "plan": str(plan_path),
        "config": config_as_dict(config),
        "all_transactions_accepted": bool(all_accepted),
        "visual_zero_superthin_pass": bool(zero_superthin),
        "fvcom_model_ready": bool(model_ready),
        "forcing_compatible": True,
        "forcing_invalidation_required": False,
        "transactions": records,
        "final": final,
        "final_summary": final_summary,
    }
    summary_path.write_text(
        json.dumps(_json_safe(document), indent=2), encoding="utf-8"
    )
    _write_markdown(run_dir / "acceptance_summary.md", document)
    print(
        json.dumps(
            {
                "all_transactions_accepted": bool(all_accepted),
                "visual_zero_superthin_pass": bool(zero_superthin),
                "fvcom_model_ready": bool(model_ready),
                "final_mesh": final["mesh"],
                "report": str(summary_path),
                "plots": [item.get("output_plot") for item in records],
            },
            indent=2,
        )
    )
    return 0 if all_accepted else 2


def _write_checkpoint(
    run_dir: Path,
    stem: str,
    source_mesh: Any,
    original_points: np.ndarray,
    state: Any,
    projection: Any,
) -> dict[str, Any]:
    mesh_path = run_dir / f"{stem}.2dm"
    boundary_path = run_dir / f"{stem}_boundary_nodes.geojson"
    lineage_path = run_dir / f"{stem}_node_lineage.json"
    depths = _remap_depths(
        source_mesh.depths,
        original_points,
        state.points,
        state.lineage,
    )
    output = write_2dm(
        mesh_path,
        unproject_points(state.points, projection),
        depths,
        state.triangles + 1,
        state.open_nodes + 1,
        mesh_name=f"{source_mesh.mesh_name}_thin_passage_removed",
    )
    proxy = SimpleNamespace(
        nodes_xy=state.points,
        triangles=state.triangles,
        open_boundary_nodes_zero_based=state.open_nodes,
    )
    roundtrip = _serialized_roundtrip_audit(output, proxy, projection)
    lonlat = unproject_points(state.points, projection)
    boundary_path.write_text(
        json.dumps(
            _boundary_geojson(
                lonlat,
                state.chains,
                state.open_nodes,
                state.kinds,
                state.hard,
                state.lineage,
                state.targets,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    lineage_path.write_text(
        json.dumps(
            {
                "schema_version": "fvcom_node_lineage_v2",
                "indexing": "array position is delivered node_index_zero_based",
                "source_node_index_zero_based": [
                    int(value) for value in state.lineage
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "mesh": str(output),
        "mesh_sha256": _sha256(output),
        "boundary_nodes_geojson": str(boundary_path),
        "node_lineage": str(lineage_path),
        "serialized_roundtrip": roundtrip,
        "obc_forcing_compatible": True,
        "forcing_invalidation_required": False,
    }


def _plot_transaction(
    path: Path,
    before_state: Any,
    after_state: Any,
    component: dict[str, Any],
    selected: dict[str, Any],
    report: dict[str, Any],
    *,
    config: ThinPassageRemovalConfig,
    dpi: int,
) -> None:
    topology = build_edge_topology(len(before_state.points), before_state.triangles)
    patch = _expand_triangle_patch(
        before_state.triangles,
        topology,
        component["triangle_indices"],
        int(config.patch_rings),
    )
    patch_nodes = sorted(set(map(int, np.unique(before_state.triangles[patch]))))
    coords = before_state.points[np.asarray(patch_nodes, dtype=int)]
    span = np.ptp(coords, axis=0)
    pad = max(float(np.max(span)) * 0.10, 5.0)
    xlim = (float(np.min(coords[:, 0])) - pad, float(np.max(coords[:, 0])) + pad)
    ylim = (float(np.min(coords[:, 1])) - pad, float(np.max(coords[:, 1])) + pad)
    removed_lineages = set(map(int, selected["edit"]["removed_node_lineages"]))
    removed_nodes = [
        index
        for index, lineage in enumerate(before_state.lineage)
        if int(lineage) in removed_lineages
    ]
    removed_triangles = set(
        map(
            int,
            np.where(
                np.any(
                    np.isin(before_state.triangles, np.asarray(removed_nodes, dtype=int)),
                    axis=1,
                )
            )[0],
        )
    )
    selected_triangles = set(map(int, component["triangle_indices"]))

    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    _draw_mesh_panel(
        axes[0],
        before_state,
        xlim,
        ylim,
        removed_triangles=removed_triangles,
        selected_triangles=selected_triangles,
        old_boundary_lineage_edges=None,
    )
    axes[0].set_title(
        f"Before: {len(removed_nodes)} passage-core nodes selected\n"
        f"orange = causal superthin; red = complete deleted node stars"
    )
    old_boundary_lineage_edges = _boundary_lineage_edges(before_state)
    _draw_mesh_panel(
        axes[1],
        after_state,
        xlim,
        ylim,
        removed_triangles=set(),
        selected_triangles=set(),
        old_boundary_lineage_edges=old_boundary_lineage_edges,
    )
    axes[1].set_title(
        "After: passage deleted; magenta = newly exposed cut boundary\n"
        f"global superthin {report['before']['superthin_triangle_count']} → {report['after']['superthin_triangle_count']}; "
        f"boundary loops {report['before']['boundary_component_count']} → {report['after']['boundary_component_count']}"
    )
    label_offsets = ((-8, 18), (0, -24), (8, 38), (-8, -42), (8, 58), (0, -62))
    ordered_removed_nodes = sorted(
        removed_nodes,
        key=lambda node: (float(before_state.points[node, 0]), float(before_state.points[node, 1])),
    )
    for ax in axes:
        if removed_nodes:
            values = before_state.points[np.asarray(removed_nodes, dtype=int)]
            ax.scatter(
                values[:, 0], values[:, 1], marker="x", s=90, linewidth=2.2,
                color="#c62828", zorder=8,
            )
            for label_index, node in enumerate(ordered_removed_nodes):
                dx, dy = label_offsets[label_index % len(label_offsets)]
                ax.annotate(
                    str(int(before_state.lineage[node]) + 1),
                    xy=before_state.points[node],
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
                    va="bottom" if dy > 0 else "top",
                    fontsize=8,
                    fontweight="bold",
                    color="#b71c1c",
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
                    arrowprops={"arrowstyle": "-", "color": "#b0bec5", "linewidth": 0.5},
                    zorder=9,
                )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.set_xlabel("UTM easting (m)")
    axes[0].set_ylabel("UTM northing (m)")
    fig.suptitle(
        f"{component['component_id']} — whole thin-passage removal audit",
        fontsize=15,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _draw_mesh_panel(
    ax: Any,
    state: Any,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    removed_triangles: set[int],
    selected_triangles: set[int],
    old_boundary_lineage_edges: set[tuple[int, int]] | None,
) -> None:
    points = state.points
    inside = (
        (points[:, 0] >= xlim[0])
        & (points[:, 0] <= xlim[1])
        & (points[:, 1] >= ylim[0])
        & (points[:, 1] <= ylim[1])
    )
    triangle_indices = np.where(np.any(inside[state.triangles], axis=1))[0]
    for index in triangle_indices:
        triangle = state.triangles[int(index)]
        polygon = points[np.r_[triangle, triangle[0]]]
        if int(index) in selected_triangles:
            face, edge, width, alpha = "#ffcc80", "#e65100", 2.3, 0.95
        elif int(index) in removed_triangles:
            face, edge, width, alpha = "#ef9a9a", "#c62828", 1.1, 0.72
        else:
            face, edge, width, alpha = "#eceff1", "#90a4ae", 0.7, 0.48
        ax.fill(
            polygon[:, 0], polygon[:, 1], facecolor=face, edgecolor=edge,
            linewidth=width, alpha=alpha, zorder=1,
        )
    topology = build_edge_topology(len(points), state.triangles)
    for edge in topology.boundary_edges:
        values = points[np.asarray(edge, dtype=int)]
        if not np.any(
            (values[:, 0] >= xlim[0])
            & (values[:, 0] <= xlim[1])
            & (values[:, 1] >= ylim[0])
            & (values[:, 1] <= ylim[1])
        ):
            continue
        lineage_edge = tuple(sorted((int(state.lineage[edge[0]]), int(state.lineage[edge[1]]))))
        new_cut = old_boundary_lineage_edges is not None and lineage_edge not in old_boundary_lineage_edges
        ax.plot(
            values[:, 0], values[:, 1],
            color="#d81b60" if new_cut else "#111111",
            linewidth=3.2 if new_cut else 2.3,
            zorder=4,
        )


def _boundary_lineage_edges(state: Any) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(state.lineage[a]), int(state.lineage[b]))))
        for a, b in chain_edges(state.chains)
    }


def _validate_plan(plan: dict[str, Any], mesh_path: Path) -> None:
    if str(plan.get("schema_version")) != PLAN_SCHEMA:
        raise ValueError(f"plan schema must be {PLAN_SCHEMA}")
    if str(plan.get("input_mesh_sha256", "")).upper() != _sha256(mesh_path):
        raise ValueError("stale thin-passage plan: input mesh hash does not match")
    transactions = plan.get("transactions")
    if not isinstance(transactions, list) or not transactions:
        raise ValueError("thin-passage plan requires at least one transaction")
    for item in transactions:
        if str(item.get("mode")) not in {"human_approved_nodes", "infer_resolution_cluster"}:
            raise ValueError(f"invalid thin-passage transaction: {item!r}")
        if not str(item.get("component_id", "")):
            raise ValueError("each thin-passage transaction requires component_id")
        if not str(item.get("output_plot", "")):
            raise ValueError("each thin-passage transaction requires output_plot")
        if str(item["mode"]) == "human_approved_nodes" and not item.get(
            "remove_node_ids_1based_source"
        ):
            raise ValueError("human-approved transaction requires remove_node_ids_1based_source")


def _write_markdown(path: Path, document: dict[str, Any]) -> None:
    lines = [
        "# Thin Passage Removal Acceptance Summary",
        "",
        f"- All transactions accepted: `{document['all_transactions_accepted']}`",
        f"- Zero superthin triangles: `{document['visual_zero_superthin_pass']}`",
        f"- FVCOM model ready: `{document['fvcom_model_ready']}`",
        f"- OBC forcing compatible: `{document['forcing_compatible']}`",
        f"- Final wet components: `{document['final_summary'].get('connected_component_count')}`",
        f"- Final singly connected triangles: `{document['final_summary'].get('singly_connected_triangle_count')}`",
        f"- Final maximum valence: `{document['final_summary'].get('maximum_valence')}`",
        f"- Final q_l3_sigma: `{document['final_summary'].get('q_l3_sigma')}`",
        "",
        "| Case | Status | Removed nodes (source IDs) | Superthin | Boundary loops |",
        "|---|---|---:|---:|---:|",
    ]
    for item in document["transactions"]:
        selected = (
            item["attempts"][int(item["selected_candidate_index"])]
            if item.get("selected_candidate_index") is not None
            else {}
        )
        ids = selected.get("edit", {}).get("removed_node_ids_1based_source", [])
        lines.append(
            f"| {item['component_id']} | {item['status']} | {', '.join(map(str, ids))} | "
            f"{item['before']['superthin_triangle_count']} -> {item['after']['superthin_triangle_count']} | "
            f"{item['before']['boundary_component_count']} -> {item['after']['boundary_component_count']} |"
        )
    lines.extend(
        [
            "",
            "The passage cut is a human-approved research topology change. It is not enabled in normal automatic profiles.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
