"""Minimal SMS 2DM read/write helpers for FVCOM grids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Mesh2DM:
    nodes_lonlat: np.ndarray
    depths: np.ndarray
    triangles: np.ndarray
    open_boundary_nodes: np.ndarray
    mesh_name: str
    open_boundary_chains: tuple[np.ndarray, ...] = ()
    open_boundary_ids: tuple[int, ...] = ()


def write_2dm(
    path: str | Path,
    nodes_lonlat: np.ndarray,
    depths: np.ndarray,
    triangles: np.ndarray,
    open_boundary_nodes: np.ndarray,
    mesh_name: str = "fvcom_grid",
    *,
    open_boundary_chains: Iterable[Iterable[int]] | None = None,
    open_boundary_ids: Iterable[int] | None = None,
) -> Path:
    """Write a FVCOM/SMS-style 2DM mesh.

    ``open_boundary_nodes`` is the legacy single-chain interface.  Callers
    with more than one exchange boundary should pass ``open_boundary_chains``;
    one independently terminated ``NS`` nodestring is written for each chain.
    A repeated first node is omitted because SMS does not encode cyclicity.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes_lonlat = np.asarray(nodes_lonlat, dtype=float)
    depths = np.asarray(depths, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    open_boundary_nodes = np.asarray(open_boundary_nodes, dtype=int)
    if open_boundary_chains is None:
        chains = [open_boundary_nodes] if open_boundary_nodes.size else []
    else:
        if open_boundary_nodes.size:
            raise ValueError(
                "Pass either legacy open_boundary_nodes or plural "
                "open_boundary_chains, not both"
            )
        chains = [np.asarray(list(values), dtype=int) for values in open_boundary_chains]
    ids = [
        int(value)
        for value in (
            open_boundary_ids
            if open_boundary_ids is not None
            else range(1, len(chains) + 1)
        )
    ]
    if (
        len(ids) != len(chains)
        or any(value <= 0 for value in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError(
            "open_boundary_ids must contain one unique positive ID per chain"
        )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("MESH2D\n")
        handle.write(f'MESHNAME "{mesh_name}"\n')
        for idx, tri in enumerate(triangles, start=1):
            handle.write(f"E3T {idx} {int(tri[0])} {int(tri[1])} {int(tri[2])} 1\n")
        for idx, ((lon, lat), depth) in enumerate(zip(nodes_lonlat, depths), start=1):
            handle.write(f"ND {idx} {lon:.12f} {lat:.12f} {float(depth):.4f}\n")
        for values, nodestring_id in zip(chains, ids):
            chain_node_ids = [int(v) for v in values if int(v) > 0]
            if len(chain_node_ids) > 1 and chain_node_ids[-1] == chain_node_ids[0]:
                chain_node_ids.pop()
            for start in range(0, len(chain_node_ids), 10):
                chunk = chain_node_ids[start : start + 10]
                if start + 10 >= len(chain_node_ids):
                    chunk = chunk[:-1] + [-chunk[-1], nodestring_id] if chunk else []
                handle.write("NS " + " ".join(str(v) for v in chunk) + "\n")
    return path


def read_2dm(path: str | Path) -> Mesh2DM:
    """Read the subset of SMS 2DM records written by this skill."""
    nodes: dict[int, tuple[float, float, float]] = {}
    tris: list[tuple[int, int, int]] = []
    ns_chains: list[list[int]] = []
    ns_ids: list[int] = []
    current_ns: list[int] = []
    mesh_name = "fvcom_grid"
    awaiting_ns_sentinel = False
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
                if awaiting_ns_sentinel:
                    ns_ids.append(value)
                    awaiting_ns_sentinel = False
                    continue
                if value < 0:
                    current_ns.append(abs(value))
                    ns_chains.append(current_ns)
                    current_ns = []
                    awaiting_ns_sentinel = True
                else:
                    current_ns.append(value)
    if current_ns:
        ns_chains.append(current_ns)
    if awaiting_ns_sentinel:
        ns_ids.append(len(ns_ids) + 1)
    while len(ns_ids) < len(ns_chains):
        ns_ids.append(len(ns_ids) + 1)
    ordered = [nodes[idx] for idx in sorted(nodes)]
    arr = np.asarray([[lon, lat] for lon, lat, _depth in ordered], dtype=float)
    depths = np.asarray([depth for _lon, _lat, depth in ordered], dtype=float)
    chain_arrays = tuple(np.asarray(values, dtype=int) for values in ns_chains if values)
    legacy_open = chain_arrays[0].copy() if chain_arrays else np.empty(0, dtype=int)
    return Mesh2DM(
        nodes_lonlat=arr,
        depths=depths,
        triangles=np.asarray(tris, dtype=int),
        open_boundary_nodes=legacy_open,
        mesh_name=mesh_name,
        open_boundary_chains=chain_arrays,
        open_boundary_ids=tuple(int(value) for value in ns_ids[: len(chain_arrays)]),
    )
