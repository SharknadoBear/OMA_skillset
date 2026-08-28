#!/usr/bin/env python3
"""Validate the generic FVCOM regional-refinement request contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(payload: dict) -> list[str]:
    errors: list[str] = []
    required = ["source_mesh", "refinement_polygon", "projected_crs", "output_workspace", "protected_node_ids", "patch", "size_field", "mesher"]
    for name in required:
        if name not in payload:
            errors.append(f"missing required field: {name}")
    if errors:
        return errors
    if len(payload["protected_node_ids"]) != len(set(payload["protected_node_ids"])):
        errors.append("protected_node_ids must be unique")
    if any(int(value) < 1 for value in payload["protected_node_ids"]):
        errors.append("protected node IDs must be one-based positive integers")
    patch = payload["patch"]
    if float(patch.get("initial_buffer_m", 0)) <= 0 or float(patch.get("increment_m", 0)) <= 0:
        errors.append("patch buffers and increments must be positive")
    if float(patch.get("maximum_buffer_m", 0)) < float(patch.get("initial_buffer_m", 0)):
        errors.append("maximum_buffer_m must be at least initial_buffer_m")
    size = payload["size_field"]
    if not 0 < float(size.get("gradation", 0)) <= 0.10:
        errors.append("gradation must be in (0, 0.10]")
    if float(size.get("minimum_m", 0)) <= 0 or float(size.get("maximum_core_m", 0)) < float(size.get("minimum_m", 0)):
        errors.append("size bounds are invalid")
    mesher = payload["mesher"]
    fixed = {"algorithm": 6, "threads": 1, "seed": 1, "element_order": 1, "smoothing_steps": 8, "algorithm_fallback": False}
    for name, expected in fixed.items():
        if mesher.get(name) != expected:
            errors.append(f"mesher.{name} must equal {expected!r}")
    cap = int(mesher.get("node_cap", 0))
    if not 0 < cap <= 1_000_000:
        errors.append("mesher.node_cap must be in 1..1000000")
    obc = set(payload.get("obc_node_ids", []))
    river = set(payload.get("river_node_ids", []))
    if obc & river:
        errors.append("OBC and river node contracts overlap")
    if not (obc | river).issubset(set(payload["protected_node_ids"])):
        errors.append("every OBC and river node must be protected")
    boundary = payload.get("physical_boundary_refinement", {"mode": "locked"})
    mode = boundary.get("mode", "locked")
    if mode not in {"locked", "selected_exact_chord_split"}:
        errors.append("physical_boundary_refinement.mode must be locked or selected_exact_chord_split")
    if mode == "selected_exact_chord_split" and not boundary.get("selector_geojson"):
        errors.append("selector_geojson is required for selected_exact_chord_split")
    if float(boundary.get("selection_tolerance_m", 25.0)) < 0:
        errors.append("physical boundary selection_tolerance_m must be nonnegative")
    if float(boundary.get("maximum_edge_to_target_ratio", 1.5)) <= 0:
        errors.append("maximum_edge_to_target_ratio must be positive")
    if boundary.get("preserve_source_vertices", True) is not True:
        errors.append("preserve_source_vertices must be true")
    if boundary.get("forbid_protected_incidence", True) is not True:
        errors.append("forbid_protected_incidence must be true")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    errors = validate(payload)
    print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
