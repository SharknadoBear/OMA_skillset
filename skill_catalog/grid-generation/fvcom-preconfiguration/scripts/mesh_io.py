from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class Node:
    id: int
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Element:
    id: int
    n1: int
    n2: int
    n3: int
    material: int | None = None


@dataclass(frozen=True)
class NodeString:
    id: int
    nodes: tuple[int, ...]


@dataclass
class Mesh2DM:
    path: Path
    mesh_name: str | None
    nodes: dict[int, Node]
    elements: dict[int, Element]
    nodestrings: dict[int, NodeString]

    @property
    def sorted_nodes(self) -> list[Node]:
        return [self.nodes[i] for i in sorted(self.nodes)]

    @property
    def sorted_elements(self) -> list[Element]:
        return [self.elements[i] for i in sorted(self.elements)]

    def depths(self, mode: str = "auto", constant_depth: float | None = None) -> dict[int, float]:
        if constant_depth is not None:
            if constant_depth <= 0:
                raise ValueError("--constant-depth must be positive")
            return {node_id: float(constant_depth) for node_id in self.nodes}

        mode_key = mode.lower().replace("_", "-")
        out: dict[int, float] = {}
        for node_id, node in self.nodes.items():
            if mode_key == "auto":
                depth = -node.z if node.z <= 0.0 else node.z
            elif mode_key == "negate-z":
                depth = -node.z
            elif mode_key == "positive-z":
                depth = node.z
            else:
                raise ValueError(f"Unsupported depth mode: {mode}")
            if not math.isfinite(depth) or depth <= 0.0:
                raise ValueError(
                    f"Non-positive FVCOM depth at node {node_id}: {depth}. "
                    "Use --constant-depth or check the 2DM z convention."
                )
            out[node_id] = depth
        return out


def parse_2dm(path: str | Path) -> Mesh2DM:
    mesh_path = Path(path)
    nodes: dict[int, Node] = {}
    elements: dict[int, Element] = {}
    nodestrings: dict[int, NodeString] = {}
    mesh_name: str | None = None
    next_nodestring_id = 1

    with mesh_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            record = parts[0].upper()

            if record == "MESHNAME":
                mesh_name = line.partition(" ")[2].strip().strip('"') or None
            elif record == "E3T":
                if len(parts) < 5:
                    raise ValueError(f"Malformed E3T at {mesh_path}:{line_number}")
                elem_id, n1, n2, n3 = map(int, parts[1:5])
                material = int(parts[5]) if len(parts) > 5 else None
                elements[elem_id] = Element(elem_id, n1, n2, n3, material)
            elif record == "ND":
                if len(parts) < 5:
                    raise ValueError(f"Malformed ND at {mesh_path}:{line_number}")
                node_id = int(parts[1])
                nodes[node_id] = Node(node_id, float(parts[2]), float(parts[3]), float(parts[4]))
            elif record == "NS":
                values = [int(v) for v in parts[1:]]
                ns_nodes: list[int] = []
                nodestring_id: int | None = None
                for idx, value in enumerate(values):
                    if value < 0:
                        ns_nodes.append(abs(value))
                        if idx + 1 < len(values):
                            nodestring_id = values[idx + 1]
                        break
                    ns_nodes.append(value)
                if not ns_nodes:
                    raise ValueError(f"Malformed NS at {mesh_path}:{line_number}")
                if nodestring_id is None:
                    nodestring_id = next_nodestring_id
                nodestrings[nodestring_id] = NodeString(nodestring_id, tuple(ns_nodes))
                next_nodestring_id = max(next_nodestring_id, nodestring_id + 1)

    mesh = Mesh2DM(mesh_path, mesh_name, nodes, elements, nodestrings)
    validate_mesh(mesh)
    return mesh


def validate_mesh(mesh: Mesh2DM) -> None:
    if not mesh.nodes:
        raise ValueError(f"No ND records found in {mesh.path}")
    if not mesh.elements:
        raise ValueError(f"No E3T records found in {mesh.path}")

    missing: list[tuple[int, int]] = []
    for element in mesh.elements.values():
        for node_id in (element.n1, element.n2, element.n3):
            if node_id not in mesh.nodes:
                missing.append((element.id, node_id))
    if missing:
        first = missing[0]
        raise ValueError(f"Element {first[0]} references missing node {first[1]}")

    for ns_id, nodestring in mesh.nodestrings.items():
        absent = [node_id for node_id in nodestring.nodes if node_id not in mesh.nodes]
        if absent:
            raise ValueError(f"Nodestring {ns_id} references missing nodes: {absent[:5]}")


def edge_lengths_for_nodes(mesh: Mesh2DM, node_ids: Iterable[int]) -> list[float]:
    ids = list(node_ids)
    lengths: list[float] = []
    for left, right in zip(ids[:-1], ids[1:]):
        a = mesh.nodes[left]
        b = mesh.nodes[right]
        lengths.append(math.hypot(a.x - b.x, a.y - b.y))
    return lengths


def estimate_sponge(mesh: Mesh2DM, nodestring_id: int, default_coeff: float = 0.0025) -> dict[str, float | int]:
    nodestring = mesh.nodestrings.get(nodestring_id)
    if nodestring is None:
        raise ValueError(f"Nodestring {nodestring_id} not found. Available: {sorted(mesh.nodestrings)}")
    lengths = edge_lengths_for_nodes(mesh, nodestring.nodes)
    if not lengths:
        raise ValueError(f"Nodestring {nodestring_id} needs at least two nodes for sponge estimation")
    radius = median(lengths)
    return {
        "nodestring_id": nodestring_id,
        "node_count": len(nodestring.nodes),
        "edge_count": len(lengths),
        "edge_length_min_m": min(lengths),
        "edge_length_median_m": radius,
        "edge_length_max_m": max(lengths),
        "sponge_radius_m": radius,
        "sponge_coefficient": float(default_coeff),
    }


def obc_type_code(value: str) -> int:
    key = str(value).strip().lower()
    mapping = {
        "prescribed": 1,
        "specified": 1,
        "1": 1,
        "radiation": 3,
        "3": 3,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported OBC type {value!r}; use prescribed/1 or radiation/3")
    return mapping[key]


def coriolis_values(mesh: Mesh2DM, mode: str, latitude_deg: float | None = None) -> dict[int, float]:
    key = mode.lower().replace("_", "-")
    if key == "zero":
        value_by_node = {node.id: 0.0 for node in mesh.sorted_nodes}
    elif key in {"latitude", "constant-latitude"}:
        if latitude_deg is None:
            raise ValueError("--latitude-deg is required for coriolis-mode latitude")
        value_by_node = {node.id: float(latitude_deg) for node in mesh.sorted_nodes}
    elif key in {"y-coordinate", "node-y"}:
        value_by_node = {node.id: node.y for node in mesh.sorted_nodes}
    else:
        raise ValueError(f"Unsupported coriolis mode: {mode}")
    return value_by_node


def write_fvcom_dat(
    mesh: Mesh2DM,
    out_dir: str | Path,
    prefix: str,
    open_ns: int,
    river_ns: int | None = None,
    obc_type: str = "prescribed",
    depth_mode: str = "auto",
    constant_depth: float | None = None,
    coriolis_mode: str = "zero",
    latitude_deg: float | None = None,
    sponge_mode: str = "estimate",
    sponge_coeff: float = 0.0025,
    sponge_radius: float | None = None,
) -> dict[str, object]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if open_ns not in mesh.nodestrings:
        raise ValueError(f"Open nodestring {open_ns} not found. Available: {sorted(mesh.nodestrings)}")
    if river_ns is not None and river_ns not in mesh.nodestrings:
        raise ValueError(f"River nodestring {river_ns} not found. Available: {sorted(mesh.nodestrings)}")

    depths = mesh.depths(mode=depth_mode, constant_depth=constant_depth)
    cor = coriolis_values(mesh, coriolis_mode, latitude_deg=latitude_deg)
    open_nodes = mesh.nodestrings[open_ns].nodes
    obc_code = obc_type_code(obc_type)

    sponge_estimate = estimate_sponge(mesh, open_ns, default_coeff=sponge_coeff)
    if sponge_mode.lower() == "estimate":
        radius = float(sponge_estimate["sponge_radius_m"])
    elif sponge_mode.lower() == "constant":
        if sponge_radius is None or sponge_radius <= 0:
            raise ValueError("--sponge-radius must be positive when --sponge-mode constant")
        radius = float(sponge_radius)
    else:
        raise ValueError("--sponge-mode must be estimate or constant")

    files = {
        "grd": out_path / f"{prefix}_grd.dat",
        "dep": out_path / f"{prefix}_dep.dat",
        "cor": out_path / f"{prefix}_cor.dat",
        "obc": out_path / f"{prefix}_obc.dat",
        "spg": out_path / f"{prefix}_spg.dat",
    }

    with files["grd"].open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"Node Number = {len(mesh.nodes)}\n")
        handle.write(f"Cell Number = {len(mesh.elements)}\n")
        for element in mesh.sorted_elements:
            handle.write(f"{element.id} {element.n1} {element.n2} {element.n3}\n")
        for node in mesh.sorted_nodes:
            handle.write(f"{node.id} {node.x:.6f} {node.y:.6f} {depths[node.id]:.6f}\n")

    with files["dep"].open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"Node Number = {len(mesh.nodes)}\n")
        for node in mesh.sorted_nodes:
            handle.write(f"{node.x:.6f} {node.y:.6f} {depths[node.id]:.6f}\n")

    with files["cor"].open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"Node Number = {len(mesh.nodes)}\n")
        for node in mesh.sorted_nodes:
            handle.write(f"{node.x:.6f} {node.y:.6f} {cor[node.id]:.6f}\n")

    with files["obc"].open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"OBC Node Number = {len(open_nodes)}\n")
        for counter, node_id in enumerate(open_nodes, start=1):
            handle.write(f"{counter} {node_id} {obc_code}\n")

    with files["spg"].open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"Sponge Node Number = {len(open_nodes)}\n")
        for node_id in open_nodes:
            handle.write(f"{node_id} {radius:.6f} {float(sponge_coeff):.6f} \n")

    z_values = [node.z for node in mesh.nodes.values()]
    depth_values = list(depths.values())
    manifest: dict[str, object] = {
        "mesh": str(mesh.path),
        "mesh_name": mesh.mesh_name,
        "prefix": prefix,
        "node_count": len(mesh.nodes),
        "triangle_count": len(mesh.elements),
        "nodestring_count": len(mesh.nodestrings),
        "nodestrings": {
            str(ns_id): {"node_count": len(ns.nodes), "nodes": list(ns.nodes)}
            for ns_id, ns in sorted(mesh.nodestrings.items())
        },
        "open_nodestring": open_ns,
        "river_nodestring_excluded_from_obc_spg": river_ns,
        "obc_type_code": obc_code,
        "depth_mode": depth_mode,
        "constant_depth": constant_depth,
        "z_min": min(z_values),
        "z_max": max(z_values),
        "depth_min": min(depth_values),
        "depth_max": max(depth_values),
        "coriolis_mode": coriolis_mode,
        "latitude_deg": latitude_deg,
        "sponge_mode": sponge_mode,
        "sponge_radius_m": radius,
        "sponge_coefficient": float(sponge_coeff),
        "sponge_estimate": sponge_estimate,
        "generated_files": {key: str(path) for key, path in files.items()},
        "notes": [
            "_cor.dat stores latitude degrees for geophysical meter-grid cases; FVCOM converts to physical f internally.",
            "Sponge values are initial calibration seeds and should be revisited after smoke-run diagnostics.",
        ],
    }
    manifest_path = out_path / f"{prefix}_fvcom_dat_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest

