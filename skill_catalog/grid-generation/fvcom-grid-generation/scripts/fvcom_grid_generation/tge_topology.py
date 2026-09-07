"""Reproduce the FVCOM 4.3.1 TGE open-boundary junction test.

FVCOM first assigns ``ISONB=1`` to every node on a topological exterior edge,
then overwrites listed open-boundary nodes with ``ISONB=2``.  TGE classifies a
cell as open when the three-node sum is exactly four and stops for a sum above
four.  This module deliberately follows those operations instead of replacing
them with a geometric approximation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


ALGORITHM_ID = "fvcom_4_3_1_tge_isonb_cell_sum_v1"


def bind_tge_source(path: str | Path) -> dict[str, Any]:
    """Verify that a TGE source file contains the operations we reproduce."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw_text = source.read_text(encoding="utf-8", errors="replace")
    code_lines: list[str] = []
    for line in raw_text.splitlines():
        if line.lstrip().startswith("!"):
            continue
        code_lines.append(line.split("!", 1)[0])
    text = "\n".join(code_lines).upper()
    checks = {
        "initialize_isonb_zero": r"\bISONB\s*=\s*0\b",
        "mark_exterior_nodes_one": r"ISONB\s*\(\s*NV\s*\(\s*I\s*,\s*[123]\s*\)\s*\)\s*=\s*1",
        "overwrite_obc_nodes_two": r"ISONB\s*\(\s*I_OBC_N\s*\(\s*I\s*\)\s*\)\s*=\s*2",
        "classify_sum_equal_four": r"SUM\s*\(\s*ISONB\s*\(\s*NV\s*\(\s*I\s*,\s*1\s*:\s*3\s*\)\s*\)\s*\)\s*==\s*4",
        "stop_sum_above_four": r"SUM\s*\(\s*ISONB\s*\(\s*NV\s*\(\s*I\s*,\s*1\s*:\s*3\s*\)\s*\)\s*\)\s*>\s*4",
    }
    matched = {
        name: bool(re.search(pattern, text, flags=re.MULTILINE))
        for name, pattern in checks.items()
    }
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "status": "source_bound" if all(matched.values()) else "mismatch",
        "source_path": str(source),
        "source_sha256": digest,
        "algorithm_id": ALGORITHM_ID,
        "checks": matched,
        "passed": bool(all(matched.values())),
    }


def audit_tge_boundary_junctions(
    n_nodes: int,
    triangles_zero_based: np.ndarray,
    open_boundary_chains_zero_based: Iterable[Iterable[int]],
    *,
    tge_source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the exact serial TGE ISONB cell-sum decision for a mesh."""

    triangles = np.asarray(triangles_zero_based, dtype=int)
    if n_nodes < 3:
        raise ValueError("n_nodes must be at least three")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("triangles_zero_based must be a nonempty (n, 3) array")
    if int(np.min(triangles)) < 0 or int(np.max(triangles)) >= int(n_nodes):
        raise ValueError("triangle connectivity is outside zero-based node range")
    if any(len(set(map(int, row))) != 3 for row in triangles):
        raise ValueError("triangle connectivity contains a repeated node")

    edge_counts: dict[tuple[int, int], int] = {}
    for first, second, third in triangles:
        for left, right in ((first, second), (second, third), (third, first)):
            edge = tuple(sorted((int(left), int(right))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    exterior_edges = sorted(edge for edge, count in edge_counts.items() if count == 1)
    exterior_nodes = sorted({node for edge in exterior_edges for node in edge})

    chains = [list(map(int, chain)) for chain in open_boundary_chains_zero_based]
    open_nodes = sorted({node for chain in chains for node in chain})
    if any(node < 0 or node >= int(n_nodes) for node in open_nodes):
        raise ValueError("open-boundary node is outside zero-based node range")

    # These assignments intentionally mirror TGE.F rather than infer edge roles.
    isonb = np.zeros(int(n_nodes), dtype=int)
    isonb[np.asarray(exterior_nodes, dtype=int)] = 1
    if open_nodes:
        isonb[np.asarray(open_nodes, dtype=int)] = 2
    cell_sums = np.sum(isonb[triangles], axis=1)
    open_cells = np.flatnonzero(cell_sums == 4)
    fatal_cells = np.flatnonzero(cell_sums > 4)

    fatal_details: list[dict[str, Any]] = []
    exterior_edge_set = set(exterior_edges)
    open_node_set = set(open_nodes)
    for cell in fatal_cells:
        nodes = [int(value) for value in triangles[int(cell)]]
        cell_edges = [
            tuple(sorted((nodes[0], nodes[1]))),
            tuple(sorted((nodes[1], nodes[2]))),
            tuple(sorted((nodes[2], nodes[0]))),
        ]
        fatal_details.append(
            {
                "element_id_1based": int(cell) + 1,
                "node_ids_1based": [node + 1 for node in nodes],
                "isonb_values": [int(isonb[node]) for node in nodes],
                "isonb_sum": int(cell_sums[int(cell)]),
                "exterior_edges_node_ids_1based": [
                    [edge[0] + 1, edge[1] + 1]
                    for edge in cell_edges
                    if edge in exterior_edge_set
                ],
                "open_boundary_node_ids_1based": [
                    node + 1 for node in nodes if node in open_node_set
                ],
            }
        )

    source_binding = (
        bind_tge_source(tge_source_path)
        if tge_source_path is not None
        else {
            "status": "algorithm_only",
            "source_path": None,
            "source_sha256": None,
            "algorithm_id": ALGORITHM_ID,
            "checks": {},
            "passed": None,
        }
    )
    passed = bool(not len(fatal_cells))
    if tge_source_path is not None:
        passed = bool(passed and source_binding["passed"])
    return {
        "schema_version": "fvcom_tge_boundary_junction_audit_v1",
        "algorithm_id": ALGORITHM_ID,
        "passed": passed,
        "tge_would_pstop": bool(len(fatal_cells)),
        "node_count": int(n_nodes),
        "element_count": int(len(triangles)),
        "exterior_edge_count": int(len(exterior_edges)),
        "exterior_node_count": int(len(exterior_nodes)),
        "open_boundary_chain_count": int(len(chains)),
        "open_boundary_node_count": int(len(open_nodes)),
        "open_boundary_nodes_not_exterior_1based": [
            node + 1 for node in open_nodes if node not in set(exterior_nodes)
        ],
        "open_cell_sum_equal_four_count": int(len(open_cells)),
        "open_cell_ids_1based": [int(cell) + 1 for cell in open_cells],
        "fatal_cell_sum_above_four_count": int(len(fatal_cells)),
        "fatal_cells": fatal_details,
        "source_binding": source_binding,
        "decision_rule": {
            "exterior_node_value": 1,
            "open_boundary_node_override": 2,
            "open_cell_sum": 4,
            "fatal_cell_sum": ">4",
        },
    }


__all__ = ["ALGORITHM_ID", "audit_tge_boundary_junctions", "bind_tge_source"]
