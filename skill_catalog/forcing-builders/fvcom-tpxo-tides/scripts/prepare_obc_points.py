#!/usr/bin/env python3
"""Create an exact-order geographic FVCOM OBC point table from a grid contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA = "fvcom_grid_delivery_contract_v1"
OUTPUT_SCHEMA = "fvcom_obc_points_manifest_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_nodes_sha256(node_ids: list[int]) -> str:
    payload = json.dumps(node_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def read_mesh(path: Path) -> tuple[dict[int, tuple[float, float]], int]:
    nodes: dict[int, tuple[float, float]] = {}
    element_count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            parts = raw.split()
            if not parts:
                continue
            record = parts[0].upper()
            if record == "ND":
                if len(parts) < 5:
                    raise ValueError(f"Malformed ND at {path}:{line_number}")
                node_id = int(parts[1])
                if node_id in nodes:
                    raise ValueError(f"Duplicate node {node_id} at {path}:{line_number}")
                nodes[node_id] = (float(parts[2]), float(parts[3]))
            elif record == "E3T":
                element_count += 1
    if not nodes or not element_count:
        raise ValueError("Mesh must contain ND and E3T records")
    return nodes, element_count


def to_geographic(
    nodes: dict[int, tuple[float, float]], source_crs: str
) -> dict[int, tuple[float, float]]:
    key = source_crs.strip().upper().replace(" ", "")
    if key in {"EPSG:4326", "EPSG:4269"}:
        result = dict(nodes)
    else:
        try:
            from pyproj import Transformer  # type: ignore
        except ImportError as exc:
            raise ValueError(
                f"Transforming OBC coordinates from {source_crs!r} requires pyproj"
            ) from exc
        transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        result = {
            node_id: tuple(map(float, transformer.transform(x, y)))
            for node_id, (x, y) in nodes.items()
        }
    for node_id, (longitude, latitude) in result.items():
        if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
            raise ValueError(f"Invalid geographic coordinate for node {node_id}")
    return result


def prepare(contract_file: str | Path, output_file: str | Path, report_file: str | Path) -> dict[str, Any]:
    contract_path = Path(contract_file).expanduser().resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise ValueError(f"Grid contract must use schema_version {CONTRACT_SCHEMA}")
    mesh_spec = contract.get("mesh")
    if not isinstance(mesh_spec, dict):
        raise ValueError("Grid contract requires a mesh object")
    for key in ("path", "sha256", "node_count", "triangle_count", "coordinate_crs"):
        if key not in mesh_spec:
            raise ValueError(f"Grid contract mesh is missing {key}")
    mesh_path = Path(str(mesh_spec["path"]))
    if not mesh_path.is_absolute():
        mesh_path = contract_path.parent / mesh_path
    mesh_path = mesh_path.resolve()
    if sha256_file(mesh_path) != str(mesh_spec["sha256"]).lower():
        raise ValueError("Grid contract mesh SHA-256 does not match the file")
    mesh_nodes, element_count = read_mesh(mesh_path)
    if len(mesh_nodes) != int(mesh_spec["node_count"]):
        raise ValueError("Grid contract node_count does not match the mesh")
    if element_count != int(mesh_spec["triangle_count"]):
        raise ValueError("Grid contract triangle_count does not match the mesh")

    boundaries = contract.get("open_boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("Grid contract requires a non-empty open_boundaries list")
    flattened: list[int] = []
    seen: set[int] = set()
    boundary_records: list[dict[str, Any]] = []
    for index, boundary in enumerate(boundaries, start=1):
        if not isinstance(boundary, dict):
            raise ValueError(f"open_boundaries[{index - 1}] must be an object")
        node_ids = [int(value) for value in boundary.get("node_ids", [])]
        if not node_ids:
            raise ValueError(f"open_boundaries[{index - 1}] has no nodes")
        expected_hash = str(boundary.get("node_order_sha256", "")).lower()
        actual_hash = ordered_nodes_sha256(node_ids)
        if expected_hash != actual_hash:
            raise ValueError(f"Open-boundary node-order hash mismatch for boundary {index}")
        missing = [node_id for node_id in node_ids if node_id not in mesh_nodes]
        if missing:
            raise ValueError(f"Open boundary references missing mesh node(s): {missing[:5]}")
        duplicates = [node_id for node_id in node_ids if node_id in seen]
        if duplicates:
            raise ValueError(f"Open boundaries overlap at node(s): {duplicates[:5]}")
        seen.update(node_ids)
        flattened.extend(node_ids)
        boundary_records.append(
            {
                "obc_id": str(boundary.get("obc_id") or f"obc_{index:03d}"),
                "node_count": len(node_ids),
                "node_order_sha256": actual_hash,
            }
        )

    geographic = to_geographic(mesh_nodes, str(mesh_spec["coordinate_crs"]))
    destination = Path(output_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("node_id", "longitude", "latitude"))
        for node_id in flattened:
            longitude, latitude = geographic[node_id]
            writer.writerow((node_id, f"{longitude:.12f}", f"{latitude:.12f}"))
    os.replace(partial, destination)
    longitudes = [geographic[node_id][0] for node_id in flattened]
    latitudes = [geographic[node_id][1] for node_id in flattened]
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "ready",
        "grid_contract_sha256": sha256_file(contract_path),
        "mesh_sha256": sha256_file(mesh_path),
        "coordinate_crs": str(mesh_spec["coordinate_crs"]),
        "open_boundaries": boundary_records,
        "obc_node_count": len(flattened),
        "flattened_node_order_sha256": ordered_nodes_sha256(flattened),
        "obc_points_sha256": sha256_file(destination),
        "longitude_range": [min(longitudes), max(longitudes)],
        "latitude_range": [min(latitudes), max(latitudes)],
        "output": str(destination),
    }
    report = Path(report_file).expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    report_partial = report.with_name(f".{report.name}.partial")
    report_partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(report_partial, report)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-contract", required=True)
    parser.add_argument("--output", required=True, help="Ordered OBC CSV output")
    parser.add_argument("--report", required=True, help="Hash/order manifest JSON")
    args = parser.parse_args()
    print(json.dumps(prepare(args.grid_contract, args.output, args.report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
