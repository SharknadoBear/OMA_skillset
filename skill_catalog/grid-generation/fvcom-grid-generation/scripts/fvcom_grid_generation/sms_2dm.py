"""Minimal SMS 2DM read/write helpers for FVCOM grids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Mesh2DM:
    nodes_lonlat: np.ndarray
    depths: np.ndarray
    triangles: np.ndarray
    open_boundary_nodes: np.ndarray
    mesh_name: str


def write_2dm(
    path: str | Path,
    nodes_lonlat: np.ndarray,
    depths: np.ndarray,
    triangles: np.ndarray,
    open_boundary_nodes: np.ndarray,
    mesh_name: str = "fvcom_grid",
) -> Path:
    """Write a FVCOM/SMS-style 2DM mesh."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes_lonlat = np.asarray(nodes_lonlat, dtype=float)
    depths = np.asarray(depths, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    open_boundary_nodes = np.asarray(open_boundary_nodes, dtype=int)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("MESH2D\n")
        handle.write(f'MESHNAME "{mesh_name}"\n')
        for idx, tri in enumerate(triangles, start=1):
            handle.write(f"E3T {idx} {int(tri[0])} {int(tri[1])} {int(tri[2])} 1\n")
        for idx, ((lon, lat), depth) in enumerate(zip(nodes_lonlat, depths), start=1):
            handle.write(f"ND {idx} {lon:.10f} {lat:.10f} {float(depth):.4f}\n")
        if open_boundary_nodes.size:
            ids = [int(v) for v in open_boundary_nodes if int(v) > 0]
            for start in range(0, len(ids), 10):
                chunk = ids[start : start + 10]
                if start + 10 >= len(ids):
                    chunk = chunk[:-1] + [-chunk[-1], 1] if chunk else []
                handle.write("NS " + " ".join(str(v) for v in chunk) + "\n")
    return path


def read_2dm(path: str | Path) -> Mesh2DM:
    """Read the subset of SMS 2DM records written by this skill."""
    nodes: dict[int, tuple[float, float, float]] = {}
    tris: list[tuple[int, int, int]] = []
    ns: list[int] = []
    mesh_name = "fvcom_grid"
    ns_terminated = False
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if not parts:
            continue
        tag = parts[0].upper()
        if tag == "MESHNAME" and len(parts) > 1:
            mesh_name = " ".join(parts[1:]).strip('"')
        elif tag == "E3T" and len(parts) >= 5:
            tris.append((int(parts[2]), int(parts[3]), int(parts[4])))
        elif tag == "ND" and len(parts) >= 5:
            nodes[int(parts[1])] = (float(parts[2]), float(parts[3]), float(parts[4]))
        elif tag == "NS":
            for item in parts[1:]:
                value = int(item)
                if ns_terminated and value == 1:
                    continue
                if value < 0:
                    ns_terminated = True
                ns.append(abs(value))
    ordered = [nodes[idx] for idx in sorted(nodes)]
    arr = np.asarray([[lon, lat] for lon, lat, _depth in ordered], dtype=float)
    depths = np.asarray([depth for _lon, _lat, depth in ordered], dtype=float)
    return Mesh2DM(
        nodes_lonlat=arr,
        depths=depths,
        triangles=np.asarray(tris, dtype=int),
        open_boundary_nodes=np.asarray(ns, dtype=int),
        mesh_name=mesh_name,
    )
