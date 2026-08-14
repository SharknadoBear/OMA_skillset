#!/usr/bin/env python3
"""Regression checks for high-valence boundary metadata compatibility."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from diagnose_high_valence import _boundary_metadata


def _feature(properties: dict) -> dict:
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
    }


def test_canonical_portfolio_boundary_metadata() -> None:
    document = {
        "type": "FeatureCollection",
        "features": [
            _feature(
                {
                    "node_index_zero_based": 0,
                    "reconciliation_source_chain_index_zero_based": 0,
                    "reconciliation_source_node_index_zero_based": 10,
                    "boundary_kind": "open",
                    "is_hard_anchor": True,
                }
            ),
            _feature(
                {
                    "node_index_zero_based": 1,
                    "reconciliation_source_chain_index_zero_based": 0,
                    "reconciliation_source_node_index_zero_based": -1,
                    "boundary_kind": "open",
                }
            ),
            _feature(
                {
                    "node_index_zero_based": 2,
                    "reconciliation_source_chain_index_zero_based": 1,
                    "reconciliation_source_node_index_zero_based": 22,
                    "boundary_kind": "land",
                }
            ),
        ],
    }
    with TemporaryDirectory() as folder:
        path = Path(folder) / "canonical.geojson"
        path.write_text(json.dumps(document), encoding="utf-8")
        chains, fixed, kinds, hard, lineage = _boundary_metadata(4, str(path))
    assert chains == [[0, 1], [2]]
    assert np.array_equal(fixed, [True, True, True, False])
    assert kinds == ["open", "open", "land", "interior"]
    assert np.array_equal(hard, [True, False, False, False])
    assert np.array_equal(lineage, [10, -1, 22, 3])


def test_legacy_delivered_boundary_metadata() -> None:
    document = {
        "type": "FeatureCollection",
        "features": [
            _feature(
                {
                    "node_id_1based": 2,
                    "constraint_chain_id": 7,
                    "constraint_chain_position": 1,
                    "source_node_index_zero_based": 31,
                    "is_open_boundary": False,
                }
            ),
            _feature(
                {
                    "node_id_1based": 1,
                    "constraint_chain_id": 7,
                    "constraint_chain_position": 0,
                    "source_node_index_zero_based": 30,
                    "is_open_boundary": True,
                }
            ),
        ],
    }
    with TemporaryDirectory() as folder:
        path = Path(folder) / "legacy.geojson"
        path.write_text(json.dumps(document), encoding="utf-8")
        chains, fixed, kinds, hard, lineage = _boundary_metadata(3, str(path))
    assert chains == [[0, 1]]
    assert np.array_equal(fixed, [True, True, False])
    assert kinds == ["open", "land", "interior"]
    assert np.array_equal(hard, [False, False, False])
    assert np.array_equal(lineage, [30, 31, 2])


if __name__ == "__main__":
    test_canonical_portfolio_boundary_metadata()
    test_legacy_delivered_boundary_metadata()
    print("high-valence diagnostic metadata self-test passed")
