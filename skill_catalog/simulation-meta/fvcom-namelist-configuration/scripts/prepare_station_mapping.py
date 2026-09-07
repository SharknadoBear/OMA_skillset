#!/usr/bin/env python3
"""Map eligible NOAA CO-OPS stations to exact FVCOM cells and local depths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mesh(path: Path) -> tuple[dict[int, tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    nodes: dict[int, tuple[float, float, float]] = {}
    elements: list[tuple[int, int, int, int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            parts = raw.split()
            if not parts:
                continue
            if parts[0].upper() == "ND":
                if len(parts) < 5:
                    raise ValueError(f"Malformed ND at {path}:{line_number}")
                nodes[int(parts[1])] = (float(parts[2]), float(parts[3]), float(parts[4]))
            elif parts[0].upper() == "E3T":
                if len(parts) < 5:
                    raise ValueError(f"Malformed E3T at {path}:{line_number}")
                elements.append(tuple(map(int, parts[1:5])))
    if not nodes or not elements:
        raise ValueError("Mesh must contain nodes and triangular elements")
    return nodes, elements


def barycentric(
    point: tuple[float, float], triangle: tuple[tuple[float, float, float], ...]
) -> tuple[float, float, float] | None:
    x, y = point
    a, b, c = triangle
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if denominator == 0.0:
        return None
    u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
    v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
    return u, v, 1.0 - u - v


def locate(
    point: tuple[float, float],
    nodes: dict[int, tuple[float, float, float]],
    elements: list[tuple[int, int, int, int]],
) -> tuple[int, float]:
    x, y = point
    matches: list[tuple[int, float]] = []
    for element_id, n1, n2, n3 in elements:
        triangle = (nodes[n1], nodes[n2], nodes[n3])
        if x < min(item[0] for item in triangle) or x > max(item[0] for item in triangle):
            continue
        if y < min(item[1] for item in triangle) or y > max(item[1] for item in triangle):
            continue
        weights = barycentric(point, triangle)
        if weights is None or min(weights) < -1.0e-10:
            continue
        depth = sum(weight * abs(vertex[2]) for weight, vertex in zip(weights, triangle, strict=True))
        if not math.isfinite(depth) or depth <= 0.0:
            raise ValueError(f"Station maps to nonpositive depth in element {element_id}")
        matches.append((element_id, depth))
    if not matches:
        raise ValueError("Station marked eligible is not inside any FVCOM element")
    return min(matches, key=lambda item: item[0])


def station_coordinates(longitude: float, latitude: float, mesh_crs: str) -> tuple[float, float]:
    key = mesh_crs.strip().upper().replace(" ", "")
    if key in {"EPSG:4326", "EPSG:4269"}:
        return longitude, latitude
    try:
        from pyproj import Transformer  # type: ignore
    except ImportError as exc:
        raise ValueError(f"Mapping stations into {mesh_crs!r} requires pyproj") from exc
    return tuple(map(float, Transformer.from_crs("EPSG:4326", mesh_crs, always_xy=True).transform(longitude, latitude)))


def build(grid_contract: str | Path, station_inventory: str | Path, output: str | Path) -> dict[str, Any]:
    contract_path = Path(grid_contract).expanduser().resolve()
    inventory_path = Path(station_inventory).expanduser().resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    if contract.get("schema_version") != "fvcom_grid_delivery_contract_v1":
        raise ValueError("Unsupported grid delivery contract")
    if inventory.get("schema") != "noaa_coops_validation_station_inventory_v1":
        raise ValueError("Unsupported station inventory")
    mesh_spec = contract.get("mesh")
    if not isinstance(mesh_spec, dict):
        raise ValueError("Grid contract has no mesh object")
    mesh_path = Path(str(mesh_spec["path"]))
    if not mesh_path.is_absolute():
        mesh_path = contract_path.parent / mesh_path
    mesh_path = mesh_path.resolve()
    mesh_hash = sha256_file(mesh_path)
    if mesh_hash != str(mesh_spec["sha256"]).lower():
        raise ValueError("Grid contract mesh hash mismatch")
    if str(inventory.get("mesh_sha256", "")).lower() != mesh_hash:
        raise ValueError("Station inventory was not screened against this exact mesh")
    nodes, elements = read_mesh(mesh_path)
    if len(nodes) != int(mesh_spec["node_count"]) or len(elements) != int(mesh_spec["triangle_count"]):
        raise ValueError("Grid contract mesh counts do not match the file")

    mapped: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for station in inventory.get("stations", []):
        if not isinstance(station, dict):
            continue
        station_id = str(station.get("id", "")).strip()
        role = str(station.get("role", ""))
        if not station.get("eligible"):
            excluded.append({"station_id": station_id, "reason": str(station.get("exclusion_reason") or "not_eligible")})
            continue
        longitude = float(station["longitude"])
        latitude = float(station["latitude"])
        point = station_coordinates(longitude, latitude, str(mesh_spec["coordinate_crs"]))
        cell_id, depth = locate(point, nodes, elements)
        mapped.append(
            {
                "station_id": station_id,
                "longitude": longitude,
                "latitude": latitude,
                "cell_id": cell_id,
                "depth_m": depth,
                "role": role,
                "name": station.get("name"),
            }
        )
    if not mapped:
        raise ValueError("No eligible stations mapped into the wet mesh")
    mapped.sort(key=lambda item: (str(item["role"]), str(item["station_id"])))
    payload = {
        "schema": "fvcom_station_mapping_v1",
        "status": "ready",
        "grid_contract_sha256": sha256_file(contract_path),
        "mesh_sha256": mesh_hash,
        "station_inventory_sha256": sha256_file(inventory_path),
        "selection_policy": "eligible NOAA stations strictly inside the exact wet mesh; downward all-bin currents only",
        "station_count": len(mapped),
        "stations": mapped,
        "excluded": excluded,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-contract", required=True)
    parser.add_argument("--station-inventory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.grid_contract, args.station_inventory, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
