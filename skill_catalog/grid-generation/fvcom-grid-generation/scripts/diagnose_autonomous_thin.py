#!/usr/bin/env python3
"""Build autonomous-thin-v1 geometry diagrams and pending Codex decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent))

from condition_mesh_local import _bbox, _target_sizes  # noqa: E402
from fvcom_grid_generation.autonomous_thin import (  # noqa: E402
    DIAGNOSTIC_SCHEMA,
    AutonomousThinConfig,
    decision_template,
    derive_cusp_buffer_m,
    json_safe,
    sha256_file,
)
from fvcom_grid_generation.metrics import triangle_geometry  # noqa: E402
from fvcom_grid_generation.projection import (  # noqa: E402
    local_utm_projection,
    project_points,
    unproject_points,
)
from fvcom_grid_generation.sms_2dm import read_2dm  # noqa: E402


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2) + "\n", encoding="utf-8")


def _boundary_source_evidence(
    boundary: gpd.GeoDataFrame,
    component_nodes: np.ndarray,
) -> dict[str, Any]:
    if "node_index_zero_based" not in boundary:
        return {
            "source_chain_index_zero_based": None,
            "source_node_indices_zero_based": [],
            "boundary_rows": [],
        }
    selected = boundary[
        boundary["node_index_zero_based"].astype(int).isin(
            [int(value) for value in component_nodes]
        )
    ].copy()
    rows: list[dict[str, Any]] = []
    source_nodes: set[int] = set()
    source_chains: list[int] = []
    for _, row in selected.iterrows():
        chain_value = row.get("reconciliation_source_chain_index_zero_based", 0)
        try:
            source_chains.append(int(chain_value))
        except (TypeError, ValueError):
            pass
        for column in (
            "reconciliation_source_node_index_zero_based",
            "reconciliation_source_segment_start_zero_based",
            "reconciliation_source_segment_end_zero_based",
        ):
            value = row.get(column)
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed >= 0:
                source_nodes.add(parsed)
        rows.append(
            {
                "mesh_node_index_zero_based": int(row["node_index_zero_based"]),
                "boundary_kind": str(row.get("boundary_kind", "")),
                "is_hard_anchor": bool(row.get("is_hard_anchor", False)),
                "anchor_type": str(row.get("anchor_type", "") or ""),
                "anchor_id": str(row.get("anchor_id", "") or ""),
                "target_spacing_m": float(row.get("target_spacing_m", np.nan)),
                "source_chain_index_zero_based": (
                    int(chain_value) if str(chain_value).strip() else 0
                ),
            }
        )
    chain = min(source_chains) if source_chains else None
    return {
        "source_chain_index_zero_based": chain,
        "source_node_indices_zero_based": sorted(source_nodes),
        "boundary_rows": rows,
        "touches_open_boundary": any(
            value["boundary_kind"].lower() == "open" for value in rows
        ),
        "generated_anchor_types": sorted(
            {
                value["anchor_type"]
                for value in rows
                if value["anchor_type"] in {"sharp_turn", "spit_tip"}
            }
        ),
        "nondemotable_anchor_count": sum(
            bool(value["is_hard_anchor"])
            and value["anchor_type"] not in {"sharp_turn", "spit_tip"}
            for value in rows
        ),
    }


def _read_line_context(path: str | None, projection: Any) -> list[LineString]:
    if not path:
        return []
    source = Path(path)
    if not source.is_file():
        return []
    lines: list[LineString] = []
    for layer in gpd.list_layers(source)["name"].tolist():
        try:
            frame = gpd.read_file(source, layer=layer).to_crs(projection.crs)
        except Exception:
            continue
        for geometry in frame.geometry:
            if geometry is None or geometry.is_empty:
                continue
            if geometry.geom_type == "LineString":
                lines.append(geometry)
            elif geometry.geom_type == "MultiLineString":
                lines.extend(list(geometry.geoms))
            elif geometry.geom_type == "Polygon":
                lines.append(LineString(geometry.exterior.coords))
    return lines


def _plot_whole(
    path: Path,
    points: np.ndarray,
    triangles: np.ndarray,
    components: list[dict[str, Any]],
    boundary_xy: np.ndarray,
    *,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    stride = max(1, int(np.ceil(len(triangles) / 50_000)))
    ax.triplot(
        points[:, 0],
        points[:, 1],
        triangles[::stride],
        color="#b8c5cc",
        linewidth=0.12,
        alpha=0.45,
    )
    ax.plot(boundary_xy[:, 0], boundary_xy[:, 1], color="black", linewidth=0.7)
    for rank, component in enumerate(components, start=1):
        ids = np.asarray(component["triangle_indices_zero_based"], dtype=int)
        centre = np.mean(points[triangles[ids].ravel()], axis=0)
        ax.scatter([centre[0]], [centre[1]], s=65, color="#ff6f00", zorder=5)
        ax.annotate(
            f"{rank}: {component['component_id']}",
            centre,
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color="#8b3100",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Autonomous thin-component locations — complete model domain")
    ax.set_xlabel("projected easting (m)")
    ax.set_ylabel("projected northing (m)")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_decision(
    path: Path,
    points: np.ndarray,
    triangles: np.ndarray,
    component: dict[str, Any],
    boundary_xy: np.ndarray,
    cusp_lines: list[LineString],
    gshhs_lines: list[LineString],
    *,
    dpi: int,
) -> None:
    ids = np.asarray(component["triangle_indices_zero_based"], dtype=int)
    node_ids = np.unique(triangles[ids].ravel())
    centre = np.mean(points[node_ids], axis=0)
    diameter = max(float(component["component_diameter_m"]), 1.0)
    radius = max(6.0 * float(component["local_target_m"]), 3.0 * diameter, 500.0)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    ax = axes[0, 0]
    stride = max(1, int(np.ceil(len(triangles) / 40_000)))
    ax.triplot(points[:, 0], points[:, 1], triangles[::stride], color="#c4cdd2", linewidth=0.1)
    ax.plot(boundary_xy[:, 0], boundary_xy[:, 1], color="black", linewidth=0.6)
    ax.scatter([centre[0]], [centre[1]], color="#ff6f00", s=70)
    ax.set_title("A. Complete-domain location")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[0, 1]
    local_triangles = np.where(
        np.any(
            (np.linalg.norm(points[triangles] - centre[None, None, :], axis=2) <= radius),
            axis=1,
        )
    )[0]
    ax.triplot(points[:, 0], points[:, 1], triangles[local_triangles], color="#9fb1bb", linewidth=0.45)
    ax.triplot(points[:, 0], points[:, 1], triangles[ids], color="#ff6f00", linewidth=2.2)
    ax.plot(boundary_xy[:, 0], boundary_xy[:, 1], color="black", linewidth=1.5, label="delivered boundary")
    for line in gshhs_lines:
        if line.distance(LineString([centre, centre + [1.0, 0.0]])) <= 2.0 * radius:
            x, y = line.xy
            ax.plot(x, y, color="#3274a1", linewidth=0.9, alpha=0.7, label="GSHHS" if "GSHHS" not in ax.get_legend_handles_labels()[1] else None)
    for line in cusp_lines:
        if line.distance(LineString([centre, centre + [1.0, 0.0]])) <= 2.0 * radius:
            x, y = line.xy
            ax.plot(x, y, color="#2ca02c", linewidth=1.1, alpha=0.8, label="CUSP" if "CUSP" not in ax.get_legend_handles_labels()[1] else None)
    for node in node_ids:
        ax.annotate(str(int(node) + 1), points[node], fontsize=8, color="#4b1d8f")
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_title("B. Mesh, boundary, and shoreline evidence")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    boundary_rows = component.get("boundary_rows", [])
    ax.axis("off")
    table_rows = [
        [
            row["mesh_node_index_zero_based"] + 1,
            row["boundary_kind"],
            row["is_hard_anchor"],
            row["anchor_type"] or "—",
            f"{row['target_spacing_m']:.1f}",
        ]
        for row in boundary_rows
    ]
    if table_rows:
        table = ax.table(
            cellText=table_rows,
            colLabels=["2DM node", "kind", "hard", "anchor", "h (m)"],
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.3)
    ax.set_title("C. Boundary lineage and scale")

    ax = axes[1, 1]
    ax.axis("off")
    text = (
        f"component: {component['component_id']}\n"
        f"triangles: {component['triangle_count']}\n"
        f"qmin: {component.get('minimum_quality', component.get('q_minimum'))}\n"
        f"minimum angle: {component.get('minimum_angle_deg')}°\n"
        f"diameter: {component['component_diameter_m']:.1f} m\n"
        f"local target h: {component['local_target_m']:.1f} m\n"
        f"CUSP buffer: {component['cusp_request_buffer_m']:.1f} m\n"
        f"source chain: {component.get('source_chain_index_zero_based')}\n"
        f"source nodes: {component.get('source_node_indices_zero_based')}\n\n"
        "Agent must choose exactly one route:\n"
        "• interior_topology_defect\n"
        "• resolved_channel_meshing_defect\n"
        "• subgrid_boundary_spike_or_sliver\n"
        "• subgrid_wet_connection\n"
        "• protected_or_source_conflict\n\n"
        "No isolated triangle deletion is permitted."
    )
    ax.text(0.02, 0.98, text, va="top", family="monospace", fontsize=10)
    ax.set_title("D. Autonomous routing contract")
    fig.suptitle(f"Autonomous thin decision: {component['component_id']}", fontsize=15)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--boundary-nodes-geojson", required=True, type=Path)
    parser.add_argument("--size-field-nc", required=True, type=Path)
    parser.add_argument("--bathymetry-nc", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--conditioning-report", type=Path)
    parser.add_argument("--cusp-gpkg", type=Path)
    parser.add_argument("--gshhs-gpkg", type=Path)
    parser.add_argument("--region-bpoly-json", type=Path)
    parser.add_argument("--case-manifest-json", type=Path)
    parser.add_argument("--boundary-resolution-manifest", type=Path)
    parser.add_argument("--source-boundary-metadata-json", type=Path)
    parser.add_argument("--boundary-contract-json", type=Path)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.output_dir.resolve()
    diagnostic_path = output / "thin_v2.json"
    if diagnostic_path.exists() and not args.overwrite:
        raise FileExistsError(diagnostic_path)
    output.mkdir(parents=True, exist_ok=True)
    # Keep internal names short enough for legacy Windows image writers. The
    # schema, not a verbose filename, identifies every artifact.
    atlas_dir = output / "a"
    command = [
        sys.executable,
        str(Path(__file__).with_name("diagnose_superthin_components.py")),
        "--mesh", str(args.mesh.resolve()),
        "--boundary-nodes-geojson", str(args.boundary_nodes_geojson.resolve()),
        "--size-field-nc", str(args.size_field_nc.resolve()),
        "--output-dir", str(atlas_dir),
    ]
    if args.conditioning_report:
        command.extend(["--conditioning-report", str(args.conditioning_report.resolve())])
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)
    atlas = _json(atlas_dir / "component_atlas.json")

    mesh = read_2dm(args.mesh)
    projection = local_utm_projection(_bbox(mesh.nodes_lonlat))
    points = project_points(mesh.nodes_lonlat, projection)
    triangles = np.asarray(mesh.triangles, dtype=int) - 1
    geometry = triangle_geometry(points, triangles)
    targets = _target_sizes(mesh.nodes_lonlat, points, triangles, str(args.size_field_nc))
    boundary = gpd.read_file(args.boundary_nodes_geojson).to_crs("EPSG:4326")
    boundary_xy = project_points(
        np.asarray([[p.x, p.y] for p in boundary.geometry], dtype=float),
        projection,
    )
    whole_map = output / "whole.png"
    components = [dict(value) for value in atlas.get("components", [])]
    _plot_whole(whole_map, points, triangles, components, boundary_xy, dpi=args.dpi)
    cusp_lines = _read_line_context(str(args.cusp_gpkg) if args.cusp_gpkg else None, projection)
    gshhs_lines = _read_line_context(str(args.gshhs_gpkg) if args.gshhs_gpkg else None, projection)
    decision_dir = output / "d"
    decision_dir.mkdir(parents=True, exist_ok=True)
    for component_rank, component in enumerate(components, start=1):
        ids = np.asarray(component["triangle_indices_zero_based"], dtype=int)
        nodes = np.unique(triangles[ids].ravel())
        coords = points[nodes]
        diameter = float(np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2).max())
        local_target = float(np.median(targets[nodes]))
        evidence = _boundary_source_evidence(boundary, nodes)
        component.update(evidence)
        component["component_diameter_m"] = diameter
        component["local_target_m"] = local_target
        component["cusp_request_buffer_m"] = derive_cusp_buffer_m(local_target, diameter)
        centre = np.mean(coords, axis=0)
        buffer_m = float(component["cusp_request_buffer_m"])
        bbox_xy = np.asarray(
            [
                [centre[0] - buffer_m, centre[1] - buffer_m],
                [centre[0] + buffer_m, centre[1] + buffer_m],
            ],
            dtype=float,
        )
        bbox_ll = unproject_points(bbox_xy, projection)
        component["cusp_request_bbox_wsen"] = [
            float(bbox_ll[0, 0]),
            float(bbox_ll[0, 1]),
            float(bbox_ll[1, 0]),
            float(bbox_ll[1, 1]),
        ]
        component["minimum_quality"] = float(np.min(geometry["quality"][ids]))
        component["minimum_angle_deg"] = float(
            np.min(geometry["angles_deg"][ids])
        )
        diagram = output / f"c{component_rank:02d}.png"
        component["decision_diagram"] = str(diagram)
        _plot_decision(
            diagram,
            points,
            triangles,
            component,
            boundary_xy,
            cusp_lines,
            gshhs_lines,
            dpi=args.dpi,
        )

    document = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "profile": "autonomous-thin-v1",
        "status": "pass" if not components else "agent_visual_classification_required",
        "input_hashes": {
            "mesh": sha256_file(args.mesh),
            "boundary_nodes_geojson": sha256_file(args.boundary_nodes_geojson),
            "size_field_nc": sha256_file(args.size_field_nc),
            "bathymetry_nc": sha256_file(args.bathymetry_nc) if args.bathymetry_nc else None,
            "cusp_gpkg": sha256_file(args.cusp_gpkg) if args.cusp_gpkg else None,
            "gshhs_gpkg": sha256_file(args.gshhs_gpkg) if args.gshhs_gpkg else None,
            "region_bpoly_json": sha256_file(args.region_bpoly_json) if args.region_bpoly_json else None,
            "case_manifest_json": sha256_file(args.case_manifest_json) if args.case_manifest_json else None,
            "boundary_resolution_manifest": sha256_file(args.boundary_resolution_manifest) if args.boundary_resolution_manifest else None,
            "source_boundary_metadata_json": sha256_file(args.source_boundary_metadata_json) if args.source_boundary_metadata_json else None,
            "boundary_contract_json": sha256_file(args.boundary_contract_json) if args.boundary_contract_json else None,
        },
        "input_paths": {
            "mesh": str(args.mesh.resolve()),
            "boundary_nodes_geojson": str(args.boundary_nodes_geojson.resolve()),
            "size_field_nc": str(args.size_field_nc.resolve()),
            "bathymetry_nc": str(args.bathymetry_nc.resolve()) if args.bathymetry_nc else None,
            "boundary_contract_json": str(args.boundary_contract_json.resolve()) if args.boundary_contract_json else None,
            "source_boundary_metadata_json": str(args.source_boundary_metadata_json.resolve()) if args.source_boundary_metadata_json else None,
            "boundary_resolution_manifest": str(args.boundary_resolution_manifest.resolve()) if args.boundary_resolution_manifest else None,
            "region_bpoly_json": str(args.region_bpoly_json.resolve()) if args.region_bpoly_json else None,
            "case_manifest_json": str(args.case_manifest_json.resolve()) if args.case_manifest_json else None,
            "cusp_gpkg": str(args.cusp_gpkg.resolve()) if args.cusp_gpkg else None,
            "gshhs_gpkg": str(args.gshhs_gpkg.resolve()) if args.gshhs_gpkg else None,
        },
        "projection_epsg": int(projection.epsg),
        "whole_domain_map": str(whole_map),
        "component_count": len(components),
        "superthin_triangle_count": int(sum(value["triangle_count"] for value in components)),
        "components": components,
        "policy": {
            "minimum_elements_across": 3,
            "isolated_triangle_deletion_permitted": False,
            "routine_human_review_gate": False,
            "maximum_candidates_per_component": AutonomousThinConfig().maximum_candidates_per_component,
            "maximum_remesh_cycles": AutonomousThinConfig().maximum_remesh_cycles,
        },
    }
    _write_json(diagnostic_path, document)
    for component_rank, component in enumerate(components, start=1):
        template = decision_template(document, diagnostic_path, component)
        _write_json(decision_dir / f"c{component_rank:02d}.pending.json", template)
    print(json.dumps({
        "status": document["status"],
        "diagnostic": str(diagnostic_path),
        "whole_domain_map": str(whole_map),
        "component_count": len(components),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
