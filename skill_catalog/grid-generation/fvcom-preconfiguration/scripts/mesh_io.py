from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

try:
    from .fvcom_sig import sigma_text
except ImportError:  # pragma: no cover - direct script execution
    from fvcom_sig import sigma_text


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
    pending_nodestring_nodes: list[int] = []

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
                nodestring_id: int | None = None
                for idx, value in enumerate(values):
                    if value < 0:
                        pending_nodestring_nodes.append(abs(value))
                        if idx + 1 < len(values):
                            nodestring_id = values[idx + 1]
                        break
                    pending_nodestring_nodes.append(value)
                if not pending_nodestring_nodes:
                    raise ValueError(f"Malformed NS at {mesh_path}:{line_number}")
                if any(value < 0 for value in values):
                    if nodestring_id is None:
                        nodestring_id = next_nodestring_id
                    if nodestring_id in nodestrings:
                        raise ValueError(f"Duplicate nodestring id {nodestring_id} at {mesh_path}:{line_number}")
                    nodestrings[nodestring_id] = NodeString(
                        nodestring_id, tuple(pending_nodestring_nodes)
                    )
                    pending_nodestring_nodes = []
                    next_nodestring_id = max(next_nodestring_id, nodestring_id + 1)

    if pending_nodestring_nodes:
        raise ValueError(f"Unterminated NS record at end of {mesh_path}")

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


def edge_lengths_for_nodes(
    mesh: Mesh2DM,
    node_ids: Iterable[int],
    xy_by_node: dict[int, tuple[float, float]] | None = None,
) -> list[float]:
    ids = list(node_ids)
    lengths: list[float] = []
    for left, right in zip(ids[:-1], ids[1:]):
        if xy_by_node is None:
            a = mesh.nodes[left]
            b = mesh.nodes[right]
            left_xy = (a.x, a.y)
            right_xy = (b.x, b.y)
        else:
            left_xy = xy_by_node[left]
            right_xy = xy_by_node[right]
        lengths.append(math.hypot(left_xy[0] - right_xy[0], left_xy[1] - right_xy[1]))
    return lengths


def estimate_sponge(
    mesh: Mesh2DM,
    nodestring_id: int,
    default_coeff: float = 0.0025,
    radius_scale: float = 3.0,
    xy_by_node: dict[int, tuple[float, float]] | None = None,
) -> dict[str, float | int]:
    nodestring = mesh.nodestrings.get(nodestring_id)
    if nodestring is None:
        raise ValueError(f"Nodestring {nodestring_id} not found. Available: {sorted(mesh.nodestrings)}")
    lengths = edge_lengths_for_nodes(mesh, nodestring.nodes, xy_by_node=xy_by_node)
    if not lengths:
        raise ValueError(f"Nodestring {nodestring_id} needs at least two nodes for sponge estimation")
    if not math.isfinite(radius_scale) or radius_scale <= 0:
        raise ValueError("Sponge radius scale must be finite and positive")
    median_edge = median(lengths)
    radius = radius_scale * median_edge
    return {
        "nodestring_id": nodestring_id,
        "node_count": len(nodestring.nodes),
        "edge_count": len(lengths),
        "edge_length_min_m": min(lengths),
        "edge_length_median_m": median_edge,
        "edge_length_max_m": max(lengths),
        "sponge_radius_m": radius,
        "sponge_radius_scale": radius_scale,
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


def _utm_latitude(easting: float, northing: float, zone: int, northern: bool) -> float:
    """Inverse WGS84 UTM latitude without requiring an optional GIS package."""
    if not 1 <= zone <= 60:
        raise ValueError(f"Invalid UTM zone {zone}")
    if not math.isfinite(easting) or not math.isfinite(northing):
        raise ValueError("UTM coordinates must be finite")
    a = 6378137.0
    ecc_sq = 0.0066943799901413165
    ecc_prime_sq = ecc_sq / (1.0 - ecc_sq)
    k0 = 0.9996
    x = easting - 500000.0
    y = northing if northern else northing - 10000000.0
    m = y / k0
    mu = m / (a * (1.0 - ecc_sq / 4.0 - 3.0 * ecc_sq**2 / 64.0 - 5.0 * ecc_sq**3 / 256.0))
    e1 = (1.0 - math.sqrt(1.0 - ecc_sq)) / (1.0 + math.sqrt(1.0 - ecc_sq))
    j1 = 3.0 * e1 / 2.0 - 27.0 * e1**3 / 32.0
    j2 = 21.0 * e1**2 / 16.0 - 55.0 * e1**4 / 32.0
    j3 = 151.0 * e1**3 / 96.0
    j4 = 1097.0 * e1**4 / 512.0
    fp = mu + j1 * math.sin(2.0 * mu) + j2 * math.sin(4.0 * mu) + j3 * math.sin(6.0 * mu) + j4 * math.sin(8.0 * mu)
    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)
    c1 = ecc_prime_sq * cos_fp**2
    t1 = tan_fp**2
    n1 = a / math.sqrt(1.0 - ecc_sq * sin_fp**2)
    r1 = a * (1.0 - ecc_sq) / (1.0 - ecc_sq * sin_fp**2) ** 1.5
    d = x / (n1 * k0)
    latitude = fp - (n1 * tan_fp / r1) * (
        d**2 / 2.0
        - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1**2 - 9.0 * ecc_prime_sq) * d**4 / 24.0
        + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1**2 - 252.0 * ecc_prime_sq - 3.0 * c1**2) * d**6 / 720.0
    )
    return math.degrees(latitude)


def _utm_xy(longitude: float, latitude: float, zone: int, northern: bool) -> tuple[float, float]:
    """Forward WGS84 geographic coordinates to UTM."""
    if not 1 <= zone <= 60:
        raise ValueError(f"Invalid UTM zone {zone}")
    if not -80.0 <= latitude <= 84.0:
        raise ValueError(f"Latitude {latitude} is outside the standard UTM domain")
    a = 6378137.0
    ecc_sq = 0.0066943799901413165
    ecc_prime_sq = ecc_sq / (1.0 - ecc_sq)
    k0 = 0.9996
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    lon_origin = math.radians((zone - 1) * 6 - 180 + 3)
    n = a / math.sqrt(1.0 - ecc_sq * math.sin(lat) ** 2)
    t = math.tan(lat) ** 2
    c = ecc_prime_sq * math.cos(lat) ** 2
    aa = math.cos(lat) * (lon - lon_origin)
    m = a * (
        (1.0 - ecc_sq / 4.0 - 3.0 * ecc_sq**2 / 64.0 - 5.0 * ecc_sq**3 / 256.0) * lat
        - (3.0 * ecc_sq / 8.0 + 3.0 * ecc_sq**2 / 32.0 + 45.0 * ecc_sq**3 / 1024.0) * math.sin(2.0 * lat)
        + (15.0 * ecc_sq**2 / 256.0 + 45.0 * ecc_sq**3 / 1024.0) * math.sin(4.0 * lat)
        - 35.0 * ecc_sq**3 / 3072.0 * math.sin(6.0 * lat)
    )
    easting = k0 * n * (
        aa
        + (1.0 - t + c) * aa**3 / 6.0
        + (5.0 - 18.0 * t + t**2 + 72.0 * c - 58.0 * ecc_prime_sq) * aa**5 / 120.0
    ) + 500000.0
    northing = k0 * (
        m
        + n
        * math.tan(lat)
        * (
            aa**2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c**2) * aa**4 / 24.0
            + (61.0 - 58.0 * t + t**2 + 600.0 * c - 330.0 * ecc_prime_sq) * aa**6 / 720.0
        )
    )
    if not northern:
        northing += 10000000.0
    return easting, northing


def _utm_definition(crs: str) -> tuple[int, bool] | None:
    key = crs.strip().upper().replace(" ", "")
    if not key.startswith("EPSG:"):
        return None
    epsg = int(key.split(":", 1)[1])
    if 32601 <= epsg <= 32660:
        return epsg - 32600, True
    if 32701 <= epsg <= 32760:
        return epsg - 32700, False
    return None


def projected_xy_values(
    mesh: Mesh2DM, source_crs: str, target_crs: str
) -> dict[int, tuple[float, float]]:
    source_key = source_crs.strip().upper().replace(" ", "")
    target_key = target_crs.strip().upper().replace(" ", "")
    if source_key == target_key:
        values = {node.id: (node.x, node.y) for node in mesh.sorted_nodes}
    else:
        target_utm = _utm_definition(target_crs)
        if source_key in {"EPSG:4326", "EPSG:4269"} and target_utm:
            zone, northern = target_utm
            values = {
                node.id: _utm_xy(node.x, node.y, zone, northern)
                for node in mesh.sorted_nodes
            }
        else:
            try:
                from pyproj import Transformer  # type: ignore
            except ImportError as exc:
                raise ValueError(
                    f"Transform {source_crs!r} to {target_crs!r} requires pyproj"
                ) from exc
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            values = {
                node.id: tuple(map(float, transformer.transform(node.x, node.y)))
                for node in mesh.sorted_nodes
            }
    for node_id, (x, y) in values.items():
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Nonfinite projected coordinate for node {node_id}")
    return values


def _crs_latitudes(mesh: Mesh2DM, source_crs: str) -> dict[int, float]:
    key = source_crs.strip().upper().replace(" ", "")
    if key in {"EPSG:4326", "EPSG:4269"}:
        values = {node.id: node.y for node in mesh.sorted_nodes}
        if any(not -90.0 <= value <= 90.0 for value in values.values()):
            raise ValueError("Geographic mesh contains an invalid latitude")
        return values
    if key.startswith("EPSG:"):
        epsg = int(key.split(":", 1)[1])
        if 32601 <= epsg <= 32660:
            zone, northern = epsg - 32600, True
        elif 32701 <= epsg <= 32760:
            zone, northern = epsg - 32700, False
        else:
            zone = 0
            northern = True
        if zone:
            return {
                node.id: _utm_latitude(node.x, node.y, zone, northern)
                for node in mesh.sorted_nodes
            }
    try:
        from pyproj import Transformer  # type: ignore
    except ImportError as exc:
        raise ValueError(
            f"CRS {source_crs!r} is not WGS84 UTM and pyproj is unavailable"
        ) from exc
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    values: dict[int, float] = {}
    for node in mesh.sorted_nodes:
        _, latitude = transformer.transform(node.x, node.y)
        if not math.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
            raise ValueError(f"Invalid transformed latitude for node {node.id}: {latitude}")
        values[node.id] = float(latitude)
    return values


def coriolis_values(
    mesh: Mesh2DM,
    mode: str,
    latitude_deg: float | None = None,
    source_crs: str | None = None,
) -> dict[int, float]:
    key = mode.lower().replace("_", "-")
    if key == "zero":
        value_by_node = {node.id: 0.0 for node in mesh.sorted_nodes}
    elif key in {"latitude", "constant-latitude"}:
        if latitude_deg is None:
            raise ValueError("--latitude-deg is required for coriolis-mode latitude")
        value_by_node = {node.id: float(latitude_deg) for node in mesh.sorted_nodes}
    elif key in {"y-coordinate", "node-y"}:
        value_by_node = {node.id: node.y for node in mesh.sorted_nodes}
    elif key in {"crs-latitude", "projected-latitude"}:
        if not source_crs:
            raise ValueError("source_crs is required for coriolis-mode crs-latitude")
        value_by_node = _crs_latitudes(mesh, source_crs)
    else:
        raise ValueError(f"Unsupported coriolis mode: {mode}")
    return value_by_node


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_nodes_sha256(node_ids: Iterable[int]) -> str:
    payload = json.dumps(list(node_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def audit_open_boundary_topology(
    mesh: Mesh2DM,
    open_boundaries: Iterable[dict[str, object]],
    tge_source_path: str | Path | None = None,
) -> dict[str, object]:
    """Reproduce FVCOM 4.3.1 TGE.F's ISONB sum gate for every triangle."""
    edge_owners: dict[tuple[int, int], list[int]] = {}
    element_edges: dict[int, tuple[tuple[int, int], ...]] = {}
    for element in mesh.sorted_elements:
        nodes = (element.n1, element.n2, element.n3)
        edges = tuple(
            tuple(sorted((nodes[index], nodes[(index + 1) % 3])))
            for index in range(3)
        )
        element_edges[element.id] = edges
        for edge in edges:
            edge_owners.setdefault(edge, []).append(element.id)

    open_edge_owner: dict[tuple[int, int], str] = {}
    for boundary in open_boundaries:
        obc_id = str(boundary["obc_id"])
        node_ids = [int(value) for value in boundary["node_ids"]]  # type: ignore[index]
        if len(node_ids) < 2:
            raise ValueError(f"Open boundary {obc_id} must contain at least two nodes")
        for left, right in zip(node_ids[:-1], node_ids[1:]):
            edge = tuple(sorted((left, right)))
            prior = open_edge_owner.get(edge)
            if prior is not None:
                raise ValueError(f"Open edge {edge} is declared by both {prior} and {obc_id}")
            open_edge_owner[edge] = obc_id

    non_exterior_open_edges = []
    for edge, obc_id in sorted(open_edge_owner.items()):
        owners = edge_owners.get(edge, [])
        if len(owners) != 1:
            non_exterior_open_edges.append(
                {"edge_node_ids": list(edge), "obc_id": obc_id, "element_ids": owners}
            )

    isonb = {node_id: 0 for node_id in mesh.nodes}
    for edge, owners in edge_owners.items():
        if len(owners) == 1:
            isonb[edge[0]] = 1
            isonb[edge[1]] = 1
    for boundary in open_boundaries:
        for node_id in boundary["node_ids"]:  # type: ignore[index]
            isonb[int(node_id)] = 2

    source_binding = None
    if tge_source_path is not None:
        source_path = Path(tge_source_path).resolve()
        source_text = source_path.read_text(encoding="utf-8", errors="replace")
        compact = re.sub(r"\s+", "", source_text.upper())
        required = {
            "initialize_solid_edge_1": "IF(NBE(I,1)==0)THEN",
            "initialize_solid_edge_2": "IF(NBE(I,2)==0)THEN",
            "initialize_solid_edge_3": "IF(NBE(I,3)==0)THEN",
            "override_obc_nodes": "ISONB(I_OBC_N(I))=2",
            "accept_open_cell": "IF(SUM(ISONB(NV(I,1:3)))==4)THEN",
            "reject_bad_cell": "ELSEIF(SUM(ISONB(NV(I,1:3)))>4)THEN",
        }
        missing = [name for name, expression in required.items() if expression not in compact]
        if missing:
            raise ValueError(
                "Supplied TGE.F does not match the implemented FVCOM ISONB gate; "
                f"missing expressions: {missing}"
            )
        source_binding = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "verified_expressions": sorted(required),
        }

    conflicts = []
    accepted_open_cells = []
    for element in mesh.sorted_elements:
        edges = element_edges[element.id]
        exterior = [edge for edge in edges if len(edge_owners[edge]) == 1]
        opened = [edge for edge in edges if edge in open_edge_owner]
        node_ids = [element.n1, element.n2, element.n3]
        node_flags = [isonb[node_id] for node_id in node_ids]
        flag_sum = sum(node_flags)
        cell = {
            "element_id": element.id,
            "element_node_ids": node_ids,
            "isonb_values": node_flags,
            "isonb_sum": flag_sum,
            "open_edges": [list(edge) for edge in opened],
            "exterior_edges": [list(edge) for edge in exterior],
        }
        if flag_sum == 4:
            accepted_open_cells.append(element.id)
        elif flag_sum > 4:
            conflicts.append(
                {
                    **cell,
                    "obc_ids": sorted({open_edge_owner[edge] for edge in opened}),
                    "reason": "tge_isonb_sum_greater_than_4",
                }
            )

    blocking = bool(non_exterior_open_edges or conflicts)
    return {
        "schema_version": "fvcom_obc_topology_audit_v1",
        "status": "blocking" if blocking else "ready",
        "declared_open_edge_count": len(open_edge_owner),
        "non_exterior_open_edges": non_exterior_open_edges,
        "conflicting_boundary_cells": conflicts,
        "accepted_open_boundary_cell_count": len(accepted_open_cells),
        "accepted_open_boundary_cell_ids": accepted_open_cells,
        "algorithm": {
            "source": "FVCOM 4.3.1 TRIANGLE_GRID_EDGE (TGE.F)",
            "initial_exterior_node_flag": 1,
            "open_boundary_node_override": 2,
            "accepted_open_cell_expression": "SUM(ISONB(NV(I,1:3))) == 4",
            "fatal_cell_expression": "SUM(ISONB(NV(I,1:3))) > 4",
        },
        "tge_source_binding": source_binding,
        "blocking_reason": (
            "FVCOM TGE rejects triangles whose post-OBC ISONB node-flag sum is greater "
            "than four; trim/redefine OBC junctions or repair the boundary topology"
            if blocking
            else None
        ),
    }


def write_fvcom_dat(
    mesh: Mesh2DM,
    out_dir: str | Path,
    prefix: str,
    open_ns: int | Iterable[int] | None = None,
    river_ns: int | Iterable[int] | None = None,
    obc_type: str = "prescribed",
    depth_mode: str = "auto",
    constant_depth: float | None = None,
    coriolis_mode: str = "zero",
    latitude_deg: float | None = None,
    source_crs: str | None = None,
    target_crs: str | None = None,
    horizontal_units: str = "m",
    sponge_mode: str = "estimate",
    sponge_coeff: float = 0.0025,
    sponge_radius: float | None = None,
    sponge_radius_scale: float = 3.0,
    sponge_profile: dict[str, object] | None = None,
    sponge_profile_path: str | Path | None = None,
    open_boundaries: list[dict[str, object]] | None = None,
    sigma_levels: int = 10,
    sigma_type: str = "UNIFORM",
    grid_contract_path: str | Path | None = None,
    grid_contract_sha256: str | None = None,
    attempt_id: str | None = None,
    tge_source_path: str | Path | None = None,
) -> dict[str, object]:
    if horizontal_units.lower() not in {"m", "meter", "metre", "meters", "metres"}:
        raise ValueError("FVCOM _grd.dat must use projected metric x/y coordinates")
    if not source_crs or not target_crs:
        if source_crs and not target_crs:
            target_crs = source_crs
        elif target_crs and not source_crs:
            source_crs = target_crs
        else:
            source_crs = target_crs = "mesh_native_unspecified_metric"
    target_key = target_crs.strip().upper().replace(" ", "")
    if target_key in {"EPSG:4326", "EPSG:4269"}:
        raise ValueError("FVCOM output_grid_crs must be projected; geographic degrees are not metric")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if open_boundaries is None:
        if open_ns is None:
            raise ValueError("At least one open nodestring is required")
        if isinstance(open_ns, int):
            open_ids = [open_ns]
        else:
            open_ids = [int(value) for value in open_ns]
        open_boundaries = [
            {"obc_id": f"obc_{index:03d}", "nodestring_id": ns_id, "obc_type": obc_type}
            for index, ns_id in enumerate(open_ids, start=1)
        ]

    normalized_boundaries: list[dict[str, object]] = []
    all_open_nodes: list[int] = []
    seen_nodes: set[int] = set()
    for index, supplied in enumerate(open_boundaries, start=1):
        obc_id = str(supplied.get("obc_id") or f"obc_{index:03d}")
        nodestring_id = int(supplied["nodestring_id"])
        if nodestring_id not in mesh.nodestrings:
            raise ValueError(
                f"Open nodestring {nodestring_id} not found. Available: {sorted(mesh.nodestrings)}"
            )
        actual_nodes = list(mesh.nodestrings[nodestring_id].nodes)
        contract_nodes = supplied.get("node_ids")
        if contract_nodes is not None:
            expected_nodes = [int(value) for value in contract_nodes]  # type: ignore[arg-type]
            if expected_nodes != actual_nodes:
                raise ValueError(
                    f"Open-boundary node order mismatch for {obc_id}: contract has "
                    f"{expected_nodes[:8]}..., mesh nodestring {nodestring_id} has {actual_nodes[:8]}..."
                )
        order_hash = ordered_nodes_sha256(actual_nodes)
        expected_order_hash = supplied.get("node_order_sha256")
        if expected_order_hash and str(expected_order_hash).lower() != order_hash:
            raise ValueError(f"Open-boundary node-order hash mismatch for {obc_id}")
        duplicates = [node_id for node_id in actual_nodes if node_id in seen_nodes]
        if duplicates:
            raise ValueError(f"Open boundaries overlap at node(s) {duplicates[:5]}")
        seen_nodes.update(actual_nodes)
        all_open_nodes.extend(actual_nodes)
        normalized_boundaries.append(
            {
                "obc_id": obc_id,
                "nodestring_id": nodestring_id,
                "node_ids": actual_nodes,
                "node_count": len(actual_nodes),
                "node_order_sha256": order_hash,
                "obc_type_code": obc_type_code(str(supplied.get("obc_type", obc_type))),
                "flat_start_index_1based": len(all_open_nodes) - len(actual_nodes) + 1,
                "flat_end_index_1based": len(all_open_nodes),
            }
        )

    obc_topology_audit = audit_open_boundary_topology(
        mesh, normalized_boundaries, tge_source_path=tge_source_path
    )
    if obc_topology_audit["status"] != "ready":
        conflict_ids = [
            item["element_id"]
            for item in obc_topology_audit["conflicting_boundary_cells"]  # type: ignore[index]
        ]
        raise ValueError(
            "FVCOM-invalid open-boundary topology: TGE boundary cells "
            f"{conflict_ids[:8]} have SUM(ISONB(NV(I,1:3))) > 4; "
            "repair the OBC endpoint/topology before generating DAT files"
        )

    if river_ns is None:
        river_ids: list[int] = []
    elif isinstance(river_ns, int):
        river_ids = [river_ns]
    else:
        river_ids = [int(value) for value in river_ns]
    for river_id in river_ids:
        if river_id not in mesh.nodestrings:
            raise ValueError(f"River nodestring {river_id} not found. Available: {sorted(mesh.nodestrings)}")
        if river_id in {int(item["nodestring_id"]) for item in normalized_boundaries}:
            raise ValueError(f"Nodestring {river_id} cannot be both an OBC and a river exclusion")

    depths = mesh.depths(mode=depth_mode, constant_depth=constant_depth)
    if source_crs == "mesh_native_unspecified_metric":
        metric_xy = {node.id: (node.x, node.y) for node in mesh.sorted_nodes}
    else:
        metric_xy = projected_xy_values(mesh, source_crs, target_crs)
    cor = coriolis_values(
        mesh, coriolis_mode, latitude_deg=latitude_deg, source_crs=source_crs
    )

    profile_rules: dict[str, dict[str, object]] = {}
    if sponge_profile:
        if sponge_profile.get("schema_version") != "fvcom_sponge_profile_v1":
            raise ValueError("Sponge profile must use schema_version fvcom_sponge_profile_v1")
        profile_attempt = sponge_profile.get("attempt_id")
        if not profile_attempt:
            raise ValueError("Attempt-specific sponge profile requires attempt_id")
        if attempt_id and str(profile_attempt) != attempt_id:
            raise ValueError("Sponge profile attempt_id does not match requested attempt_id")
        attempt_id = str(profile_attempt)
        raw_rules = sponge_profile.get("open_boundaries", {})
        if isinstance(raw_rules, list):
            for rule in raw_rules:
                if not isinstance(rule, dict) or not rule.get("obc_id"):
                    raise ValueError("Each sponge boundary rule requires obc_id")
                profile_rules[str(rule["obc_id"])] = rule
        elif isinstance(raw_rules, dict):
            profile_rules = {
                str(key): value for key, value in raw_rules.items() if isinstance(value, dict)
            }
            if len(profile_rules) != len(raw_rules):
                raise ValueError("Every sponge-profile boundary value must be an object")
        else:
            raise ValueError("Sponge profile open_boundaries must be an object or list")
        known_ids = {str(item["obc_id"]) for item in normalized_boundaries}
        unknown_ids = sorted(set(profile_rules) - known_ids)
        if unknown_ids:
            raise ValueError(f"Sponge profile contains unknown OBC ids: {unknown_ids}")

    sponge_by_obc: list[dict[str, object]] = []
    for boundary in normalized_boundaries:
        estimate = estimate_sponge(
            mesh,
            int(boundary["nodestring_id"]),
            default_coeff=sponge_coeff,
            radius_scale=sponge_radius_scale,
            xy_by_node=metric_xy,
        )
        if sponge_mode.lower() == "estimate":
            radius = float(estimate["sponge_radius_m"])
        elif sponge_mode.lower() == "constant":
            if sponge_radius is None or sponge_radius <= 0:
                raise ValueError("--sponge-radius must be positive when --sponge-mode constant")
            radius = float(sponge_radius)
        else:
            raise ValueError("--sponge-mode must be estimate or constant")
        coefficient = float(sponge_coeff)
        rule = profile_rules.get(str(boundary["obc_id"]), {})
        if "radius_m" in rule and "radius_multiplier" in rule:
            raise ValueError("Use only radius_m or radius_multiplier in a sponge rule")
        if "coefficient" in rule and "coefficient_multiplier" in rule:
            raise ValueError("Use only coefficient or coefficient_multiplier in a sponge rule")
        if "radius_m" in rule:
            radius = float(rule["radius_m"])
        elif "radius_multiplier" in rule:
            radius *= float(rule["radius_multiplier"])
        if "coefficient" in rule:
            coefficient = float(rule["coefficient"])
        elif "coefficient_multiplier" in rule:
            coefficient *= float(rule["coefficient_multiplier"])
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(f"Sponge radius for {boundary['obc_id']} must be finite and positive")
        if not math.isfinite(coefficient) or coefficient <= 0:
            raise ValueError(f"Sponge coefficient for {boundary['obc_id']} must be finite and positive")
        sponge_by_obc.append(
            {
                "obc_id": boundary["obc_id"],
                "nodestring_id": boundary["nodestring_id"],
                "radius_m": radius,
                "coefficient": coefficient,
                "baseline_estimate": estimate,
                "override": rule or None,
            }
        )

    files = {
        "grd": out_path / f"{prefix}_grd.dat",
        "dep": out_path / f"{prefix}_dep.dat",
        "cor": out_path / f"{prefix}_cor.dat",
        "obc": out_path / f"{prefix}_obc.dat",
        "spg": out_path / f"{prefix}_spg.dat",
        "sig": out_path / f"{prefix}_sig.dat",
    }

    with files["grd"].open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"Node Number = {len(mesh.nodes)}\n")
        handle.write(f"Cell Number = {len(mesh.elements)}\n")
        for element in mesh.sorted_elements:
            handle.write(f"{element.id} {element.n1} {element.n2} {element.n3}\n")
        for node in mesh.sorted_nodes:
            x, y = metric_xy[node.id]
            handle.write(f"{node.id} {x:.6f} {y:.6f} {depths[node.id]:.6f}\n")

    with files["dep"].open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"Node Number = {len(mesh.nodes)}\n")
        for node in mesh.sorted_nodes:
            x, y = metric_xy[node.id]
            handle.write(f"{x:.6f} {y:.6f} {depths[node.id]:.6f}\n")

    with files["cor"].open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"Node Number = {len(mesh.nodes)}\n")
        for node in mesh.sorted_nodes:
            x, y = metric_xy[node.id]
            handle.write(f"{x:.6f} {y:.6f} {cor[node.id]:.6f}\n")

    with files["obc"].open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"OBC Node Number = {len(all_open_nodes)}\n")
        counter = 0
        for boundary in normalized_boundaries:
            for node_id in boundary["node_ids"]:  # type: ignore[union-attr]
                counter += 1
                handle.write(f"{counter} {node_id} {boundary['obc_type_code']}\n")

    with files["spg"].open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"Sponge Node Number = {len(all_open_nodes)}\n")
        for boundary, controls in zip(normalized_boundaries, sponge_by_obc):
            for node_id in boundary["node_ids"]:  # type: ignore[union-attr]
                handle.write(
                    f"{node_id} {float(controls['radius_m']):.6f} "
                    f"{float(controls['coefficient']):.6f} \n"
                )

    files["sig"].write_text(
        sigma_text(levels=sigma_levels, sigma_type=sigma_type),
        encoding="ascii",
        newline="\n",
    )

    z_values = [node.z for node in mesh.nodes.values()]
    depth_values = list(depths.values())
    manifest: dict[str, object] = {
        "schema_version": "fvcom_preconfiguration_manifest_v1",
        "status": "ready",
        "mesh": str(mesh.path),
        "mesh_sha256": sha256_file(mesh.path),
        "mesh_name": mesh.mesh_name,
        "prefix": prefix,
        "node_count": len(mesh.nodes),
        "triangle_count": len(mesh.elements),
        "nodestring_count": len(mesh.nodestrings),
        "nodestrings": {
            str(ns_id): {"node_count": len(ns.nodes), "nodes": list(ns.nodes)}
            for ns_id, ns in sorted(mesh.nodestrings.items())
        },
        "open_boundaries": normalized_boundaries,
        "open_boundary_count": len(normalized_boundaries),
        "open_boundary_node_count": len(all_open_nodes),
        "flattened_obc_node_order_sha256": ordered_nodes_sha256(all_open_nodes),
        "obc_topology_audit": obc_topology_audit,
        "river_nodestrings_excluded_from_obc_spg": river_ids,
        "depth_mode": depth_mode,
        "constant_depth": constant_depth,
        "z_min": min(z_values),
        "z_max": max(z_values),
        "depth_min": min(depth_values),
        "depth_max": max(depth_values),
        "coriolis_mode": coriolis_mode,
        "latitude_deg": latitude_deg,
        "mesh_coordinate_crs": source_crs,
        "output_grid_crs": target_crs,
        "horizontal_units": "m",
        "coriolis_latitude_min_deg": min(cor.values()),
        "coriolis_latitude_max_deg": max(cor.values()),
        "sponge_mode": sponge_mode,
        "sponge_radius_scale": sponge_radius_scale,
        "sponge_by_obc": sponge_by_obc,
        "sponge_profile": {
            "path": str(sponge_profile_path) if sponge_profile_path else None,
            "sha256": sha256_file(sponge_profile_path) if sponge_profile_path else None,
            "attempt_id": attempt_id,
        },
        "sigma": {"levels": sigma_levels, "type": sigma_type.upper()},
        "grid_contract": {
            "path": str(grid_contract_path) if grid_contract_path else None,
            "sha256": grid_contract_sha256,
        },
        "generated_files": {key: str(path) for key, path in files.items()},
        "generated_file_sha256": {key: sha256_file(path) for key, path in files.items()},
        "notes": [
            "_cor.dat stores latitude degrees for geophysical meter-grid cases; FVCOM converts to physical f internally.",
            "Sponge values are initial calibration seeds and should be revisited after smoke-run diagnostics.",
        ],
    }
    manifest_path = out_path / f"{prefix}_fvcom_dat_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest
