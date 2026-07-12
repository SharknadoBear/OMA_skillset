#!/usr/bin/env python3
"""Apply aggressive local FVCOM topology conditioning to an existing 2DM mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import geopandas as gpd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.local_topology import AggressiveConditioningConfig, condition_mesh_aggressive  # noqa: E402
from fvcom_grid_generation.metrics import build_edge_topology, triangle_geometry  # noqa: E402
from fvcom_grid_generation.postprocess import boundary_chains_from_mesh  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection, project_points, unproject_points  # noqa: E402
from fvcom_grid_generation.sms_2dm import read_2dm, write_2dm  # noqa: E402


def main_with_mode(forced_mode: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output-mesh", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-boundary-nodes")
    parser.add_argument("--boundary-nodes-geojson")
    parser.add_argument("--boundary-resolution-manifest")
    parser.add_argument("--size-field-nc")
    parser.add_argument("--target-spacing-m", type=float, help="Uniform fallback/override target spacing for standalone tests.")
    parser.add_argument("--mode", choices=("all", "valence", "prune"), default=forced_mode or "all")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--boundary-edit-policy", choices=("kind-aware-envelope", "split-only", "none"), default="kind-aware-envelope")
    parser.add_argument("--max-prunes-per-round", type=int, default=500)
    parser.add_argument("--max-valence-repairs-per-round", type=int, default=500)
    parser.add_argument("--max-valence-flip-batch", type=int, default=64)
    parser.add_argument("--max-valence-cluster-merges-per-round", type=int, default=25)
    parser.add_argument(
        "--max-valence-l-over-h-count-increase",
        type=int,
        default=0,
        help="Explicit transaction budget for new triangles above L/h=1.55 while closing hard valence defects.",
    )
    parser.add_argument(
        "--only-node-id-1based",
        type=int,
        action="append",
        default=[],
        help="Restrict valence repair to one or more current-mesh node IDs; repeat the option for multiple nodes.",
    )
    parser.add_argument("--micro-relax-cycles", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if forced_mode is not None:
        args.mode = forced_mode
    if any(value <= 0 for value in args.only_node_id_1based):
        parser.error("--only-node-id-1based values must be positive")
    mesh_path = Path(args.mesh)
    output_path = Path(args.output_mesh)
    report_path = Path(args.report)
    boundary_output = Path(args.output_boundary_nodes) if args.output_boundary_nodes else None
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if output_path.resolve() == mesh_path.resolve():
        raise ValueError("Refusing to overwrite the input mesh")
    existing = [path for path in (output_path, report_path, boundary_output) if path is not None and path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")

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
    targets = _target_sizes(mesh.nodes_lonlat, points, triangles, args.size_field_nc)
    if args.target_spacing_m is not None:
        if args.target_spacing_m <= 0.0:
            parser.error("--target-spacing-m must be positive")
        targets[:] = float(args.target_spacing_m)
    valid_explicit = np.isfinite(explicit_targets) & (explicit_targets > 0.0)
    targets[valid_explicit] = explicit_targets[valid_explicit]

    mode = str(args.mode)
    config = AggressiveConditioningConfig(
        max_rounds=int(args.rounds),
        boundary_edit_policy=str(args.boundary_edit_policy),
        max_prunes_per_round=int(args.max_prunes_per_round) if mode in {"all", "prune"} else 0,
        max_collapses_per_round=100 if mode == "all" else 0,
        max_boundary_edits_per_round=25 if mode == "all" else 0,
        max_valence_removals_per_round=int(args.max_valence_repairs_per_round) if mode in {"all", "valence"} else 0,
        max_valence_flip_batch=int(args.max_valence_flip_batch) if mode in {"all", "valence"} else 0,
        max_valence_cluster_merges_per_round=(
            int(args.max_valence_cluster_merges_per_round) if mode in {"all", "valence"} else 0
        ),
        max_valence_l_over_h_count_increase=(
            int(args.max_valence_l_over_h_count_increase) if mode in {"all", "valence"} else 0
        ),
        valence_node_lineage_filter=tuple(int(value) - 1 for value in args.only_node_id_1based) if mode in {"all", "valence"} else (),
        micro_relax_cycles=int(args.micro_relax_cycles),
    )
    result = condition_mesh_aggressive(
        points,
        triangles,
        fixed,
        chains,
        open_nodes,
        target_spacing_m=targets,
        boundary_kinds=kinds,
        hard_anchor_mask=hard,
        config=config,
    )
    depths = _remap_depths(mesh.depths, points, result.nodes_xy, result.node_lineage)
    output_mesh = write_2dm(
        output_path,
        unproject_points(result.nodes_xy, projection),
        depths,
        result.triangles + 1,
        result.open_boundary_nodes_zero_based + 1,
        mesh_name=f"{mesh.mesh_name}_{mode}_local_v2",
    )
    roundtrip = _serialized_roundtrip_audit(output_mesh, result, projection)
    if boundary_output is not None:
        boundary_output.parent.mkdir(parents=True, exist_ok=True)
        boundary_output.write_text(
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
    document = {
        "schema_version": "fvcom_local_conditioning_cli_v2",
        "mode": mode,
        "input_mesh": str(mesh_path),
        "output_mesh": str(output_mesh),
        "output_boundary_nodes": str(boundary_output) if boundary_output is not None else None,
        "projection_epsg": int(projection.epsg),
        "boundary_metadata_supplied": bool(args.boundary_nodes_geojson),
        "size_field_supplied": bool(args.size_field_nc),
        "serialized_roundtrip": roundtrip,
        "conditioning": result.report,
        "edit_ledger": result.edit_ledger,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_json_safe(document), indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_mesh": str(output_mesh),
                "report": str(report_path),
                "edit_count": int(len(result.edit_ledger)),
                "fvcom_valence_gate_passed": bool(result.report["fvcom_valence_gate_passed"]),
                "serialized_roundtrip_passed": bool(roundtrip["passed"]),
            },
            indent=2,
        )
    )
    return 0 if roundtrip["passed"] and (mode == "prune" or result.report["fvcom_valence_gate_passed"]) else 2


def _serialized_roundtrip_audit(output_mesh: Path, result: Any, projection: Any) -> dict[str, Any]:
    written = read_2dm(output_mesh)
    expected_triangles = np.asarray(result.triangles, dtype=int) + 1
    expected_open = np.asarray(result.open_boundary_nodes_zero_based, dtype=int) + 1
    node_count_match = bool(len(written.nodes_lonlat) == len(result.nodes_xy))
    triangle_connectivity_match = bool(np.array_equal(written.triangles, expected_triangles))
    open_boundary_order_match = bool(np.array_equal(written.open_boundary_nodes, expected_open))
    finite_positive_depths = bool(np.all(np.isfinite(written.depths)) and np.all(written.depths > 0.0))
    coordinate_max_shift_m = float("inf")
    positive_signed_areas = False
    minimum_signed_area_m2 = float("nan")
    if node_count_match:
        projected = project_points(written.nodes_lonlat, projection)
        coordinate_max_shift_m = float(np.max(np.linalg.norm(projected - result.nodes_xy, axis=1))) if len(projected) else 0.0
        if triangle_connectivity_match and len(written.triangles):
            signed = triangle_geometry(projected, written.triangles - 1)["signed_area"]
            minimum_signed_area_m2 = float(np.min(signed))
            positive_signed_areas = bool(np.all(signed > 0.0))
    passed = bool(
        node_count_match
        and triangle_connectivity_match
        and open_boundary_order_match
        and finite_positive_depths
        and coordinate_max_shift_m <= 0.01
        and positive_signed_areas
    )
    return {
        "passed": passed,
        "node_count_match": node_count_match,
        "triangle_connectivity_match": triangle_connectivity_match,
        "open_boundary_order_match": open_boundary_order_match,
        "finite_positive_depths": finite_positive_depths,
        "coordinate_max_shift_m": coordinate_max_shift_m,
        "coordinate_tolerance_m": 0.01,
        "positive_signed_areas": positive_signed_areas,
        "minimum_signed_area_m2": minimum_signed_area_m2,
    }


def _boundary_metadata(
    node_count: int,
    triangles: np.ndarray,
    open_nodes: np.ndarray,
    boundary_geojson: str | None,
    resolution_manifest: str | None,
) -> tuple[list[list[int]], list[str], np.ndarray, np.ndarray]:
    kinds = ["interior"] * int(node_count)
    hard = np.zeros(int(node_count), dtype=bool)
    targets = np.full(int(node_count), np.nan, dtype=float)
    chain_position_to_node: dict[tuple[int, int], int] = {}
    if boundary_geojson:
        document = json.loads(Path(boundary_geojson).read_text(encoding="utf-8-sig"))
        groups: dict[int, list[tuple[int, int]]] = {}
        for feature in document.get("features", []):
            props = feature.get("properties", {})
            node = int(props.get("node_index_zero_based", int(props["node_id_1based"]) - 1))
            chain = int(props["constraint_chain_id"])
            position = int(props["constraint_chain_position"])
            chain_position_to_node[(chain, position)] = node
            groups.setdefault(chain, []).append((position, node))
            if node < node_count:
                kinds[node] = str(props.get("boundary_kind", "open" if props.get("is_open_boundary") else "island" if chain > 0 else "land"))
                hard[node] = bool(props.get("is_hard_anchor", False))
                value = props.get("target_spacing_m")
                if value is not None:
                    targets[node] = float(value)
        chains = [[node for _, node in sorted(groups[key])] for key in sorted(groups)]
    else:
        chains = boundary_chains_from_mesh(triangles + 1)
        for chain_index, chain in enumerate(chains):
            for node in chain:
                kinds[int(node)] = "island" if chain_index > 0 else "land"
    open_set = set(map(int, open_nodes.tolist()))
    for node in open_set:
        if 0 <= node < node_count:
            kinds[node] = "open"
    if len(open_nodes):
        hard[int(open_nodes[0])] = True
        hard[int(open_nodes[-1])] = True
    # Enrich sparse delivered metadata by chain identity.  Never overwrite by
    # row position after node compaction, because terminal node indices may no
    # longer match the upstream package's row index.
    if resolution_manifest:
        manifest = json.loads(Path(resolution_manifest).read_text(encoding="utf-8-sig"))
        gpkg = Path(manifest["outputs"]["boundary_resolution_gpkg"])
        nodes = gpd.read_file(gpkg, layer="boundary_nodes").sort_values(["chain_id", "chain_position"]).reset_index(drop=True)
        for index, row in nodes.iterrows():
            key = (int(row["chain_id"]), int(row["chain_position"]))
            node = chain_position_to_node.get(key, int(index) if not boundary_geojson and int(index) < node_count else -1)
            if not 0 <= node < node_count:
                continue
            kinds[node] = str(row["boundary_kind"])
            hard[node] = bool(row.get("is_hard_anchor", hard[node]))
            # The adaptive boundary-resolution package is authoritative.  A
            # delivered mesh GeoJSON may contain raster-sampled fallback values
            # that are useful without the upstream package but must not shadow
            # its explicit along-chain targets when the manifest is supplied.
            targets[node] = float(row["target_spacing_m"])
    return chains, kinds, hard, targets


def _target_sizes(lonlat: np.ndarray, points: np.ndarray, triangles: np.ndarray, size_field_nc: str | None) -> np.ndarray:
    if size_field_nc:
        with xr.open_dataset(size_field_nc) as dataset:
            interpolator = RegularGridInterpolator(
                (np.asarray(dataset["lat"].values, dtype=float), np.asarray(dataset["lon"].values, dtype=float)),
                np.asarray(dataset["mesh_size_m"].values, dtype=float),
                bounds_error=False,
                fill_value=np.nan,
            )
            values = interpolator(np.column_stack((lonlat[:, 1], lonlat[:, 0])))
        if np.any(np.isfinite(values) & (values > 0.0)):
            fallback = float(np.nanmedian(values[np.isfinite(values) & (values > 0.0)]))
            return np.where(np.isfinite(values) & (values > 0.0), values, fallback)
    topology = build_edge_topology(len(points), triangles)
    values = np.full(len(points), np.nan, dtype=float)
    for node, neighbors in enumerate(topology.node_neighbors):
        if neighbors:
            values[node] = float(np.median([np.linalg.norm(points[node] - points[neighbor]) for neighbor in neighbors]))
    fallback = float(np.nanmedian(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else 1.0
    return np.where(np.isfinite(values) & (values > 0.0), values, fallback)


def _remap_depths(original: np.ndarray, original_points: np.ndarray, new_points: np.ndarray, lineage: np.ndarray) -> np.ndarray:
    depths = np.full(len(new_points), np.nan, dtype=float)
    lineage = np.asarray(lineage, dtype=int)
    surviving = (lineage >= 0) & (lineage < len(original))
    depths[surviving] = np.asarray(original, dtype=float)[lineage[surviving]]
    missing = np.where(~np.isfinite(depths))[0]
    if len(missing):
        tree = cKDTree(np.asarray(original_points, dtype=float))
        distance, indices = tree.query(np.asarray(new_points, dtype=float)[missing], k=min(3, len(original_points)))
        distance = np.atleast_2d(distance) if np.ndim(distance) == 1 else distance
        indices = np.atleast_2d(indices) if np.ndim(indices) == 1 else indices
        weights = 1.0 / np.maximum(distance, 1.0e-6)
        depths[missing] = np.sum(weights * np.asarray(original, dtype=float)[indices], axis=1) / np.sum(weights, axis=1)
    return np.maximum(depths, 0.5)


def _boundary_geojson(
    lonlat: np.ndarray,
    chains: list[list[int]],
    open_nodes: np.ndarray,
    kinds: list[str],
    hard: np.ndarray,
    lineage: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    open_set = set(map(int, open_nodes.tolist()))
    features = []
    for chain_id, chain in enumerate(chains):
        for position, node in enumerate(chain):
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "node_index_zero_based": int(node),
                        "node_id_1based": int(node) + 1,
                        "constraint_chain_id": int(chain_id),
                        "constraint_chain_position": int(position),
                        "is_open_boundary": bool(node in open_set),
                        "boundary_kind": str(kinds[node]),
                        "is_hard_anchor": bool(hard[node]),
                        "source_node_index_zero_based": int(lineage[node]) if lineage[node] >= 0 else None,
                        "target_spacing_m": float(targets[node]),
                    },
                    "geometry": {"type": "Point", "coordinates": [float(lonlat[node, 0]), float(lonlat[node, 1])]},
                }
            )
    return {"type": "FeatureCollection", "features": features}


def _bbox(lonlat: np.ndarray) -> tuple[float, float, float, float]:
    return float(np.min(lonlat[:, 0])), float(np.min(lonlat[:, 1])), float(np.max(lonlat[:, 0])), float(np.max(lonlat[:, 1]))


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


def main() -> int:
    return main_with_mode(None)


if __name__ == "__main__":
    raise SystemExit(main())
