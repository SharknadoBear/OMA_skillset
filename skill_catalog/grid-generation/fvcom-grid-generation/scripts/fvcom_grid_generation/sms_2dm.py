"""SMS 2DM reader/writer for FVCOM triangular meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Mesh2DM:
    name: str
    nodes: np.ndarray
    depths: np.ndarray
    triangles: np.ndarray
    open_boundaries: list[np.ndarray]


def read_2dm(path: str | Path) -> Mesh2DM:
    """Read MESH2D, MESHNAME, E3T, ND, and NS records from an SMS 2DM file."""
    path = Path(path)
    name = path.stem
    nodes_by_id: dict[int, tuple[float, float, float]] = {}
    triangles: list[tuple[int, int, int]] = []
    open_boundaries: list[np.ndarray] = []
    current_ns: list[int] = []

    for raw in path.read_text(errors="replace").splitlines():
        parts = raw.split()
        if not parts:
            continue
        rec = parts[0].upper()
        if rec == "MESHNAME":
            name = raw.split(" ", 1)[1].strip().strip('"') if " " in raw else name
        elif rec == "E3T":
            triangles.append((int(parts[2]), int(parts[3]), int(parts[4])))
        elif rec == "ND":
            nodes_by_id[int(parts[1])] = (float(parts[2]), float(parts[3]), float(parts[4]))
        elif rec == "NS":
            for token in parts[1:]:
                value = int(float(token))
                current_ns.append(abs(value))
                if value < 0:
                    open_boundaries.append(np.asarray(current_ns, dtype=int))
                    current_ns = []
            # SMS often appends the first node after a negative end marker.
            if open_boundaries and current_ns and current_ns[-1] == open_boundaries[-1][0]:
                current_ns = []

    if current_ns:
        open_boundaries.append(np.asarray(current_ns, dtype=int))
    if not nodes_by_id:
        raise ValueError(f"No ND records found in {path}")

    max_id = max(nodes_by_id)
    nodes = np.full((max_id, 2), np.nan, dtype=float)
    depths = np.full(max_id, np.nan, dtype=float)
    for node_id, (lon, lat, depth) in nodes_by_id.items():
        nodes[node_id - 1] = (lon, lat)
        depths[node_id - 1] = depth
    return Mesh2DM(
        name=name,
        nodes=nodes,
        depths=depths,
        triangles=np.asarray(triangles, dtype=int),
        open_boundaries=open_boundaries,
    )


def write_2dm(
    path: str | Path,
    nodes: np.ndarray,
    depths: np.ndarray,
    triangles: np.ndarray,
    open_boundary: np.ndarray | None = None,
    mesh_name: str = "fvcom_grid",
) -> Path:
    """Write an SMS 2DM file with one optional open-boundary nodestring."""
    path = Path(path)
    nodes = np.asarray(nodes, dtype=float)
    depths = np.asarray(depths, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("MESH2D\n")
        handle.write(f'MESHNAME "{mesh_name}"\n')
        for eid, tri in enumerate(triangles, start=1):
            handle.write(f"E3T {eid} {int(tri[0])} {int(tri[1])} {int(tri[2])} 1\n")
        for nid, ((lon, lat), depth) in enumerate(zip(nodes, depths), start=1):
            handle.write(f"ND {nid} {lon:.8e} {lat:.8e} {depth:.8e}\n")
        if open_boundary is not None and len(open_boundary) > 0:
            tokens = [int(v) for v in open_boundary]
            if len(tokens) >= 2:
                tokens = tokens[:-1] + [-abs(tokens[-1]), tokens[0]]
            for start in range(0, len(tokens), 10):
                chunk = tokens[start : start + 10]
                handle.write("NS  " + " ".join(str(v) for v in chunk) + "\n")
    return path
