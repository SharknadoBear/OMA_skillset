#!/usr/bin/env python3
"""Audit an SMS 2DM with FVCOM 4.3.1 TGE's exact ISONB sum test."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent / "fvcom_grid_generation"


def _load_module(name: str, filename: str):
    path = _PACKAGE_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


read_2dm = _load_module("standalone_sms_2dm", "sms_2dm.py").read_2dm
audit_tge_boundary_junctions = _load_module(
    "standalone_tge_topology", "tge_topology.py"
).audit_tge_boundary_junctions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--tge-source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    mesh = read_2dm(args.mesh)
    report = audit_tge_boundary_junctions(
        len(mesh.nodes_lonlat),
        np.asarray(mesh.triangles, dtype=int) - 1,
        [
            np.asarray(chain, dtype=int) - 1
            for chain in mesh.open_boundary_chains
        ],
        tge_source_path=args.tge_source,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
