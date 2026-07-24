#!/usr/bin/env python3
"""Apply one agent-reviewed visual superthin repair plan to an FVCOM 2DM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

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
from fvcom_grid_generation.projection import (  # noqa: E402
    local_utm_projection,
    project_points,
    unproject_points,
)
from fvcom_grid_generation.sms_2dm import read_2dm, write_2dm  # noqa: E402
from fvcom_grid_generation.visual_superthin import (  # noqa: E402
    VisualSuperthinConfig,
    apply_visual_superthin_plan,
    validate_visual_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--boundary-nodes-geojson", required=True)
    parser.add_argument("--size-field-nc", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-mesh", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-boundary-nodes", required=True)
    parser.add_argument("--output-obc-remap-manifest", required=True)
    parser.add_argument("--progress-jsonl")
    parser.add_argument("--maximum-support-nodes", type=int, default=2)
    parser.add_argument("--maximum-boundary-insertions", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = {
        "mesh": Path(args.mesh),
        "boundary": Path(args.boundary_nodes_geojson),
        "size": Path(args.size_field_nc),
        "plan": Path(args.plan),
        "output_mesh": Path(args.output_mesh),
        "report": Path(args.report),
        "output_boundary": Path(args.output_boundary_nodes),
        "output_obc": Path(args.output_obc_remap_manifest),
    }
    for key in ("mesh", "boundary", "size", "plan"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    outputs = [
        paths["output_mesh"],
        paths["report"],
        paths["output_boundary"],
        paths["output_obc"],
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not bool(args.overwrite):
        raise FileExistsError(f"refusing to overwrite visual outputs: {existing}")
    if int(args.maximum_support_nodes) not in {0, 1, 2}:
        parser.error("--maximum-support-nodes must be 0, 1, or 2")
    if not 0 <= int(args.maximum_boundary_insertions) <= 2:
        parser.error("--maximum-boundary-insertions must be between 0 and 2")

    mesh_hash = _sha256(paths["mesh"])
    plan = json.loads(paths["plan"].read_text(encoding="utf-8-sig"))
    validate_visual_plan(plan, input_sha256=mesh_hash)
    progress_path = Path(args.progress_jsonl) if args.progress_jsonl else None
    _progress(
        progress_path,
        "visual_plan_started",
        {
            "mesh": str(paths["mesh"]),
            "mesh_sha256": mesh_hash,
            "plan": str(paths["plan"]),
            "component_id": plan["component"]["component_id"],
        },
    )

    mesh = read_2dm(paths["mesh"])
    projection = local_utm_projection(_bbox(mesh.nodes_lonlat))
    points = project_points(mesh.nodes_lonlat, projection)
    triangles = np.asarray(mesh.triangles, dtype=int) - 1
    open_nodes = np.asarray(mesh.open_boundary_nodes, dtype=int) - 1
    chains, kinds, hard, explicit_targets = _boundary_metadata(
        len(points),
        triangles,
        open_nodes,
        str(paths["boundary"]),
        None,
    )
    fixed = np.zeros(len(points), dtype=bool)
    for chain in chains:
        fixed[np.asarray(chain, dtype=int)] = True
    targets = _target_sizes(
        mesh.nodes_lonlat,
        points,
        triangles,
        str(paths["size"]),
    )
    explicit = np.isfinite(explicit_targets) & (explicit_targets > 0.0)
    targets[explicit] = explicit_targets[explicit]
    restrictions = {
        tuple(sorted(map(int, edge)))
        for edge in plan.get("restricted_lineage_edges", [])
    }

    result = apply_visual_superthin_plan(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
        plan=plan,
        boundary_kinds=kinds,
        hard_anchor_mask=hard,
        node_lineage=np.arange(len(points), dtype=int),
        restricted_lineage_edges=restrictions,
        visual_config=VisualSuperthinConfig(
            maximum_support_nodes=int(args.maximum_support_nodes),
            maximum_boundary_insertions=int(args.maximum_boundary_insertions),
        ),
    )
    depths = _remap_depths(
        mesh.depths,
        points,
        result.nodes_xy,
        result.node_lineage,
    )
    paths["output_mesh"].parent.mkdir(parents=True, exist_ok=True)
    output_mesh = write_2dm(
        paths["output_mesh"],
        unproject_points(result.nodes_xy, projection),
        depths,
        result.triangles + 1,
        result.open_boundary_nodes_zero_based + 1,
        mesh_name=f"{mesh.mesh_name}_visual_superthin",
    )
    roundtrip = _serialized_roundtrip_audit(output_mesh, result, projection)
    paths["output_boundary"].parent.mkdir(parents=True, exist_ok=True)
    paths["output_boundary"].write_text(
        json.dumps(
            _boundary_geojson(
                unproject_points(result.nodes_xy, projection),
                result.constraint_chains,
                result.open_boundary_nodes_zero_based,
                result.boundary_kinds,
                result.hard_anchor_mask,
                result.node_lineage,
                result.target_spacing_m,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["output_obc"].write_text(
        json.dumps(_json_safe(result.obc_remap_manifest), indent=2),
        encoding="utf-8",
    )
    document = {
        "schema_version": "fvcom_visual_superthin_cli_v1",
        "input_mesh": str(paths["mesh"]),
        "input_mesh_sha256": mesh_hash,
        "boundary_nodes_geojson": str(paths["boundary"]),
        "size_field_nc": str(paths["size"]),
        "plan": str(paths["plan"]),
        "output_mesh": str(output_mesh),
        "output_boundary_nodes": str(paths["output_boundary"]),
        "output_obc_remap_manifest": str(paths["output_obc"]),
        "projection_epsg": int(projection.epsg),
        "repair": result.report,
        "edit_ledger": result.edit_ledger,
        "obc_remap_manifest": result.obc_remap_manifest,
        "serialized_roundtrip": roundtrip,
    }
    paths["report"].write_text(
        json.dumps(_json_safe(document), indent=2),
        encoding="utf-8",
    )
    _progress(
        progress_path,
        "visual_plan_finished",
        {
            "status": result.report["status"],
            "accepted": bool(result.report["accepted"]),
            "superthin_before": int(result.report["before"]["superthin_triangle_count"]),
            "superthin_after": int(result.report["after"]["superthin_triangle_count"]),
            "report": str(paths["report"]),
        },
    )
    print(
        json.dumps(
            {
                "status": result.report["status"],
                "accepted": bool(result.report["accepted"]),
                "superthin_before": int(result.report["before"]["superthin_triangle_count"]),
                "superthin_after": int(result.report["after"]["superthin_triangle_count"]),
                "output_mesh": str(output_mesh),
                "report": str(paths["report"]),
            },
            indent=2,
        )
    )
    return 0 if bool(result.report["accepted"]) else 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _progress(path: Path | None, event: str, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "fvcom_visual_superthin_progress_v1",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "event": str(event),
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
