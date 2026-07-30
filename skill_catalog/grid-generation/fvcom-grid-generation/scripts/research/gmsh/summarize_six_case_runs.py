#!/usr/bin/env python3
"""Aggregate immutable six-case Gmsh run artifacts into one result matrix."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CASE_ORDER = {
    "san_francisco_bay": 1,
    "delaware_bay": 2,
    "galveston_trinity_bay": 3,
    "long_island_sound": 4,
    "lake_ontario": 5,
    "hawaiian_islands": 6,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def artifact_path(run_dir: Path, run: dict[str, Any], key: str, fallback: str) -> Path:
    declared = ((run.get("artifacts") or {}).get(key) or {}).get("path")
    path = Path(declared) if declared else run_dir / fallback
    return path if path.is_absolute() else run_dir / path


def row_for_run(run_manifest_path: Path) -> dict[str, Any]:
    run_dir = run_manifest_path.parent
    run = read_json(run_manifest_path)
    quality_path = artifact_path(run_dir, run, "quality_json", "quality.json")
    preflight_path = artifact_path(
        run_dir, run, "node_budget_preflight", "node_budget_preflight.json"
    )
    quality = read_json(quality_path)
    preflight = read_json(preflight_path)
    ocean = quality["oceanmesh_quality"]
    size = quality["size_error_l_over_h"]
    topology = quality["topology"]
    native = quality["gmsh_native_quality"]
    leakage = quality["euclidean_through_land_sizing_leakage"]
    normality = quality["obc_normality"]
    roundtrip = quality["sms_2dm_roundtrip"]
    attempt = run["attempts"][-1]
    revalidation = dict(run.get("boundary_revalidation") or {})
    snapshot = ((run.get("artifacts") or {}).get("case_manifest_snapshot") or {})
    return {
        "order": CASE_ORDER[str(run["case_id"])],
        "case_id": run["case_id"],
        "display_name": run["display_name"],
        "status": run["status"],
        "accepted": bool(quality["accepted"]),
        "h_uniform_m": run["selected_h_uniform_m"],
        "bathymetry_floor_m": preflight["bathymetry_resolution_floor"][
            "selected_floor_m"
        ],
        "boundary_node_count_preflight": preflight["boundary_mesh_at_floor"][
            "node_count"
        ],
        "estimated_total_nodes": preflight["estimated_total_nodes"],
        "node_count": quality["node_count"],
        "triangle_count": quality["triangle_count"],
        "overflow_rerun_count": max(0, len(run["attempts"]) - 1),
        "open_boundary_chain_count": quality["open_boundary_chain_count"],
        "open_boundary_node_count": quality["open_boundary_node_count"],
        "forcing_compatible": bool(run["forcing_compatible"]),
        "boundary_revalidation_passed": bool(revalidation.get("passed", False)),
        "case_manifest_snapshot_sha256": snapshot.get("sha256"),
        "q_l3_sigma": ocean["q_l3_sigma"],
        "q_min": ocean["q_min"],
        "count_q_below_0_10": ocean["count_q_below_0_10"],
        "min_angle_deg": quality["min_angle_deg"],
        "max_angle_deg": quality["max_angle_deg"],
        "max_bathymetric_slope": quality["max_bathymetric_slope"],
        "max_adjacent_area_change": quality["max_adjacent_area_change"],
        "max_node_valence": quality["max_node_valence"],
        "connected_component_count": topology["connected_component_count"],
        "nonmanifold_edge_count": topology["nonmanifold_edge_count"],
        "singly_connected_triangle_count": topology[
            "singly_connected_triangle_count"
        ],
        "size_l_over_h_p95": size["quantiles"]["p95"],
        "size_l_over_h_max": size["maximum"],
        "roundtrip_passed": bool(roundtrip["passed"]),
        "roundtrip_max_shift_m": roundtrip[
            "maximum_projected_coordinate_shift_m"
        ],
        "gmsh_sicn_median": native["sicn"]["median"],
        "gmsh_gamma_median": native["gamma"]["median"],
        "obc_normality_p95_deviation_deg": normality["p95_deviation_deg"],
        "euclidean_leakage_candidate_fraction": (
            leakage.get("candidate_fraction") if leakage.get("applicable") else None
        ),
        "failure_taxonomy": list(quality["failure_taxonomy"]),
        "msh_sha256": attempt["msh_sha256"],
        "run_manifest": str(run_manifest_path.resolve()),
        "quality_json": str(quality_path.resolve()),
        "mesh_map": str(
            artifact_path(run_dir, run, "mesh_map", "mesh_map.png").resolve()
        ),
        "quality_map": str(
            artifact_path(run_dir, run, "quality_map", "quality_map.png").resolve()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_csv.exists():
        raise FileExistsError("Summary outputs must be fresh.")

    manifests = sorted(args.run_root.rglob("run_manifest.json"))
    rows = sorted((row_for_run(path) for path in manifests), key=lambda row: row["order"])
    ids = [row["case_id"] for row in rows]
    missing = sorted(set(CASE_ORDER) - set(ids))
    duplicate = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if missing or duplicate:
        raise RuntimeError(
            f"Expected one immutable run for each case; missing={missing}, duplicate={duplicate}"
        )

    payload = {
        "schema_version": "gmsh_six_case_result_matrix_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(args.run_root.resolve()),
        "case_count": len(rows),
        "pass_count": sum(row["status"] == "pass" for row in rows),
        "needs_review_count": sum(row["status"] == "needs_review" for row in rows),
        "all_structural_roundtrips_passed": all(
            row["roundtrip_passed"]
            and row["connected_component_count"] == 1
            and row["nonmanifold_edge_count"] == 0
            for row in rows
        ),
        "all_boundary_revalidations_passed": all(
            row["boundary_revalidation_passed"] for row in rows
        ),
        "production_promotion_eligible": all(
            row["status"] == "pass"
            and row["boundary_revalidation_passed"]
            for row in rows
        ),
        "promotion_policy": (
            "all six topology cases must pass automatic boundary revalidation "
            "and every hard mesh gate"
        ),
        "cases": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["failure_taxonomy"] = ";".join(row["failure_taxonomy"])
        for key in ("run_manifest", "quality_json", "mesh_map", "quality_map"):
            flat.pop(key)
        flat_rows.append(flat)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
