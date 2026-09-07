#!/usr/bin/env python3
"""Focused regression tests for the FVCOM TGE boundary-junction gate."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import tempfile

import numpy as np

_MODULE_PATH = (
    Path(__file__).resolve().parent
    / "fvcom_grid_generation"
    / "tge_topology.py"
)
_SPEC = importlib.util.spec_from_file_location("tge_topology", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
audit_tge_boundary_junctions = _MODULE.audit_tge_boundary_junctions


def main() -> int:
    # Two triangles form a square.  OBC 1-2 ends in an all-exterior triangle,
    # so its adjacent cell has the fatal sum 2 + 2 + 1 = 5.
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    failing = audit_tge_boundary_junctions(4, triangles, [[0, 1]])
    assert not failing["passed"]
    assert failing["tge_would_pstop"]
    assert failing["fatal_cell_sum_above_four_count"] == 1
    assert failing["fatal_cells"][0]["element_id_1based"] == 1
    assert failing["fatal_cells"][0]["isonb_sum"] == 5

    # Removing the junction endpoint from the OBC reproduces the accepted trim:
    # no cell sum is above four.
    passing = audit_tge_boundary_junctions(4, triangles, [[1]])
    assert passing["passed"]
    assert not passing["tge_would_pstop"]

    with tempfile.TemporaryDirectory(prefix="tge_source_") as temp:
        source = Path(temp) / "tge.F"
        source.write_text(
            """
            ISONB = 0
            ISONB(NV(I,2)) = 1
            ISONB(I_OBC_N(I))=2
            IF(SUM(ISONB(NV(I,1:3))) == 4) THEN
            ELSE IF(SUM(ISONB(NV(I,1:3))) > 4) THEN
            END IF
            """,
            encoding="utf-8",
        )
        bound = audit_tge_boundary_junctions(
            4,
            triangles,
            [[1]],
            tge_source_path=source,
        )
        assert bound["passed"]
        assert bound["source_binding"]["status"] == "source_bound"
        assert len(bound["source_binding"]["source_sha256"]) == 64
    print("selftest_tge_topology: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
