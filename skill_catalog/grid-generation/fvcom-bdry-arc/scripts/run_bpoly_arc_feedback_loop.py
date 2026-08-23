#!/usr/bin/env python3
"""Run a bounded RegionBPoly -> GSHHS -> boundary-arc feedback loop."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from pyproj import Geod


GEOD = Geod(ellps="WGS84")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _run(command: list[str], cwd: Path, log_path: Path, allow_nonzero: bool = False) -> int:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND\n" + subprocess.list2cmdline(command) + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0 and not allow_nonzero:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log_path}")
    return int(completed.returncode)


def _arc_command(args, region: Path, offshore: Path, output_dir: Path, name: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_bdry_arc.py"),
        "--region-bpoly-json",
        str(region),
        "--offshore-artifacts-json",
        str(offshore),
        "--run-dir",
        str(output_dir),
        "--name",
        name,
        "--coastline-source",
        "gshhs",
        "--mode",
        "test",
        "--target-resolution-m",
        str(args.target_resolution_m),
        "--review-depth",
        "full",
        "--coastline-buffer-km",
        str(args.gshhs_lookahead_km),
        "--gshhs-resolution",
        args.gshhs_resolution,
        "--gshhs-levels",
        args.gshhs_levels,
        "--topology-mode",
        "gshhs-vector",
        "--heuristic-mode",
        "unknown",
        "--topology-time-budget-s",
        str(args.topology_time_budget_s),
        "--boundary-resolution-profile",
        args.boundary_resolution_profile,
        "--frame-clip-policy",
        "reject-unintended",
        "--residual-boundary-policy",
        args.residual_boundary_policy,
        "--feedback-candidate-max-km",
        str(args.max_side_expansion_km),
    ]
    if args.frame_clip_tolerance_m is not None:
        command.extend(["--frame-clip-tolerance-m", str(args.frame_clip_tolerance_m)])
    if args.coastline_gpkg:
        command.extend(["--coastline-gpkg", str(Path(args.coastline_gpkg).resolve())])
    else:
        command.append("--fetch-coastline")
    if args.gshhs_skill_dir:
        command.extend(["--gshhs-skill-dir", str(Path(args.gshhs_skill_dir).resolve())])
    return command


def _run_arc(args, region: Path, offshore: Path, output_dir: Path, name: str) -> tuple[dict, dict, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to rerun into nonempty arc directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(_arc_command(args, region, offshore, output_dir, name), output_dir, output_dir / "command.log")
    manifest_path = output_dir / "bdry_arc_manifest.json"
    manifest = _read_json(manifest_path)
    feedback_path = Path(manifest.get("outputs", {}).get("feedback_json", ""))
    if not feedback_path.is_file():
        raise FileNotFoundError("Boundary arc run did not produce region_bpoly_arc_feedback_v1.json")
    return manifest, _read_json(feedback_path), feedback_path


def _row(iteration: int, trial: str, manifest: dict, feedback: dict, candidate: dict | None, selected: bool = False) -> dict:
    metrics = feedback.get("metrics", {})
    return {
        "iteration": iteration,
        "trial": trial,
        "candidate_id": candidate.get("candidate_id") if candidate else None,
        "side_index": candidate.get("side_index") if candidate else None,
        "profile": candidate.get("profile") if candidate else None,
        "displacement_km": candidate.get("displacement_km") if candidate else 0.0,
        "region_status": None,
        "arc_status": manifest.get("final_status"),
        "feedback_status": feedback.get("status"),
        "adaptive_status": metrics.get("adaptive_status"),
        "unintended_frame_clip_length_m": metrics.get("unintended_frame_clip_length_m"),
        "unintended_frame_clip_fraction": metrics.get("unintended_frame_clip_fraction"),
        "intended_exterior_coverage_fraction": metrics.get("intended_land_open_exterior_coverage_fraction"),
        "wet_component_count": metrics.get("wet_component_count"),
        "delivered_obc_count": metrics.get("delivered_obc_count"),
        "open_boundary_land_intersection_m": metrics.get("open_boundary_land_intersection_m"),
        "selected": selected,
        "failure_taxonomy": feedback.get("failure_taxonomy", []),
    }


def _full_pass(manifest: dict, feedback: dict, adaptive_profile: str) -> bool:
    if manifest.get("final_status") != "pass" or feedback.get("status") != "pass":
        return False
    if adaptive_profile == "legacy":
        return True
    return manifest.get("boundary_resolution", {}).get("final_status") == "pass"


def _clip_length(feedback: dict) -> float:
    return float(feedback.get("metrics", {}).get("unintended_frame_clip_length_m", float("inf")) or 0.0)


def _clip_closed_without_structural_regression(feedback: dict) -> bool:
    """Return true when only the downstream adaptive gate still blocks a trial."""
    if feedback.get("diagnostic_status") != "pass":
        return False
    adaptive_failures = set(feedback.get("adaptive", {}).get("failure_taxonomy", []))
    remaining = [
        item
        for item in feedback.get("failure_taxonomy", [])
        if item not in adaptive_failures and not str(item).startswith("adaptive_boundary_resolution")
    ]
    return not remaining


def _candidate_max_vertex_displacement_km(candidate: dict) -> float:
    deltas = candidate.get("vertex_delta_km", {})
    return max(
        (float(dx) ** 2 + float(dy) ** 2) ** 0.5
        for dx, dy in deltas.values()
    ) if deltas else float(candidate.get("displacement_km", float("inf")))


def _region_area_m2(region: dict) -> float:
    points = region.get("polygon_lonlat") or region.get("region_bpoly", {}).get("polygon_lonlat")
    if not points:
        return 0.0
    if points[0] != points[-1]:
        points = [*points, points[0]]
    area, _ = GEOD.polygon_area_perimeter(
        [float(point[0]) for point in points],
        [float(point[1]) for point in points],
    )
    return abs(float(area))


def _area_growth_fraction(source: dict, adjusted: dict) -> float:
    source_area = _region_area_m2(source)
    adjusted_area = _region_area_m2(adjusted)
    return max(0.0, adjusted_area / source_area - 1.0) if source_area > 0.0 else float("inf")


def _trial_rank(item: dict) -> tuple[float, float, float]:
    """Rank comparable hard-gate outcomes by residual, area growth, then movement."""
    return (
        _clip_length(item["feedback"]),
        float(item.get("area_growth_fraction", float("inf"))),
        _candidate_max_vertex_displacement_km(item["candidate"]),
    )


def _select_trial(
    trial_results: list[dict],
    current_feedback: dict,
    adaptive_profile: str,
) -> tuple[dict | None, str, str]:
    """Select a monotonic trial without trading clip closure for adaptive status."""
    passing = [
        item
        for item in trial_results
        if _full_pass(item["manifest"], item["feedback"], adaptive_profile)
    ]
    if passing:
        return min(passing, key=_trial_rank), "pass", "full_boundary_gate_passed"

    # Once a candidate closes the frame-clipping and structural gates, an
    # adaptive-only failure is not evidence for another bbox displacement.
    # Preserve the smallest geometry change and stop for explicit review.
    clip_closed = [
        item
        for item in trial_results
        if _clip_closed_without_structural_regression(item["feedback"])
    ]
    if clip_closed:
        return (
            min(clip_closed, key=_trial_rank),
            "input_needs_review",
            "adaptive_failure_after_frame_clip_closed",
        )

    adjustable = [
        item for item in trial_results if item["feedback"].get("status") == "adjust_bpoly"
    ]
    if not adjustable:
        return None, "input_needs_review", "no_adjustable_candidate"
    selected = min(adjustable, key=_trial_rank)
    old_clip = _clip_length(current_feedback)
    new_clip = _clip_length(selected["feedback"])
    tolerance = float(
        selected["feedback"].get("policy", {}).get("frame_clip_tolerance_m", 0.0) or 0.0
    )
    if not (new_clip <= tolerance or new_clip <= 0.95 * old_clip):
        return None, "input_needs_review", "no_monotonic_clip_improvement"
    return selected, "input_needs_review", "monotonic_bbox_adjustment_selected"


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "iteration",
        "trial",
        "candidate_id",
        "side_index",
        "profile",
        "displacement_km",
        "region_status",
        "arc_status",
        "feedback_status",
        "adaptive_status",
        "unintended_frame_clip_length_m",
        "unintended_frame_clip_fraction",
        "intended_exterior_coverage_fraction",
        "wet_component_count",
        "delivered_obc_count",
        "open_boundary_land_intersection_m",
        "selected",
        "failure_taxonomy",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            copy = dict(row)
            copy["failure_taxonomy"] = ";".join(copy.get("failure_taxonomy", []))
            writer.writerow(copy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-bpoly-json", required=True)
    parser.add_argument("--offshore-artifacts-json", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--coastline-gpkg")
    parser.add_argument("--gshhs-skill-dir")
    parser.add_argument("--region-bpoly-skill-dir")
    parser.add_argument("--gshhs-resolution", default="h", choices=("c", "l", "i", "h", "f"))
    parser.add_argument("--gshhs-levels", default="1")
    parser.add_argument("--gshhs-lookahead-km", type=float, default=100.0)
    parser.add_argument("--target-resolution-m", type=float, default=8000.0)
    parser.add_argument("--frame-clip-tolerance-m", type=float)
    parser.add_argument("--residual-boundary-policy", default="solid-default", choices=("solid-default", "strict-reject"))
    parser.add_argument("--boundary-resolution-profile", default="adaptive-coastal-v2", choices=("legacy", "adaptive-coastal-v1", "adaptive-coastal-v2"))
    parser.add_argument("--topology-time-budget-s", type=float, default=900.0)
    parser.add_argument("--max-adjustments", type=int, default=4)
    parser.add_argument("--max-candidates-per-adjustment", type=int, default=3)
    parser.add_argument("--max-side-expansion-km", type=float, default=100.0)
    args = parser.parse_args()

    root = Path(args.run_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty feedback-loop directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    # Keep physical paths compact for GDAL/SQLite on Windows. Human-readable
    # iteration and candidate identifiers remain in the JSON/CSV manifests.
    seed_dir = root / "i00" / "region"
    seed_dir.mkdir(parents=True, exist_ok=True)
    current_region = seed_dir / "region_bpoly.json"
    current_offshore = seed_dir / "offshore_boundary_artifacts.json"
    shutil.copy2(Path(args.region_bpoly_json).resolve(), current_region)
    shutil.copy2(Path(args.offshore_artifacts_json).resolve(), current_offshore)

    skill_parent = Path(__file__).resolve().parents[2]
    region_skill = Path(args.region_bpoly_skill_dir).resolve() if args.region_bpoly_skill_dir else skill_parent / "fvcom-region-bpoly"
    apply_script = region_skill / "scripts" / "apply_arc_feedback.py"
    if not apply_script.is_file():
        raise FileNotFoundError(f"Could not locate RegionBPoly feedback applier: {apply_script}")

    rows: list[dict] = []
    accepted_adjustments: list[dict] = []
    cumulative_by_side: dict[int, float] = {}
    current_manifest, current_feedback, current_feedback_path = _run_arc(
        args,
        current_region,
        current_offshore,
        root / "i00" / "arc",
        "fb_i00",
    )
    rows.append(_row(0, "seed", current_manifest, current_feedback, None, selected=True))
    final_status = "pass" if _full_pass(current_manifest, current_feedback, args.boundary_resolution_profile) else "input_needs_review"
    stop_reason = "seed_passed" if final_status == "pass" else None

    for adjustment_index in range(1, args.max_adjustments + 1):
        if final_status == "pass":
            break
        if current_feedback.get("status") != "adjust_bpoly":
            stop_reason = "feedback_not_adjustable"
            break
        candidates = []
        for candidate in current_feedback.get("candidate_recommendations", []):
            side = int(candidate["side_index"])
            cumulative = cumulative_by_side.get(side, 0.0) + float(candidate["displacement_km"])
            if cumulative <= args.max_side_expansion_km + 1.0e-9:
                candidates.append(candidate)
            if len(candidates) >= args.max_candidates_per_adjustment:
                break
        if not candidates:
            stop_reason = "side_expansion_cap_exhausted"
            break

        trial_results = []
        iteration_dir = root / f"i{adjustment_index:02d}"
        for trial_index, candidate in enumerate(candidates, start=1):
            trial_dir = iteration_dir / f"t{trial_index:02d}"
            trial_dir.mkdir(parents=True, exist_ok=False)
            region_dir = trial_dir / "region"
            apply_command = [
                sys.executable,
                str(apply_script),
                "--input-json",
                str(current_region),
                "--feedback-json",
                str(current_feedback_path),
                "--candidate-id",
                candidate["candidate_id"],
                "--output-dir",
                str(region_dir),
                "--basemap-provider",
                "none",
            ]
            apply_code = _run(apply_command, trial_dir, trial_dir / "apply_feedback.log", allow_nonzero=True)
            region_json = region_dir / "region_bpoly.json"
            offshore_json = region_dir / "offshore_boundary_artifacts.json"
            if not region_json.is_file() or not offshore_json.is_file():
                rows.append(
                    {
                        **_row(adjustment_index, f"trial_{trial_index:02d}", {}, {"status": "input_needs_review", "failure_taxonomy": ["region_adjustment_failed"]}, candidate),
                        "region_status": "failed",
                    }
                )
                continue
            region_doc = _read_json(region_json)
            if apply_code != 0 or region_doc.get("final_status") != "pass":
                rows.append(
                    {
                        **_row(adjustment_index, f"trial_{trial_index:02d}", {}, {"status": "input_needs_review", "failure_taxonomy": ["region_adjustment_qa_failed"]}, candidate),
                        "region_status": region_doc.get("final_status"),
                    }
                )
                continue
            manifest, feedback, feedback_path = _run_arc(
                args,
                region_json,
                offshore_json,
                trial_dir / "arc",
                f"fb_i{adjustment_index:02d}t{trial_index:02d}",
            )
            row = _row(adjustment_index, f"trial_{trial_index:02d}", manifest, feedback, candidate)
            row["region_status"] = region_doc.get("final_status")
            rows.append(row)
            trial_results.append(
                {
                    "candidate": candidate,
                    "region": region_json,
                    "offshore": offshore_json,
                    "manifest": manifest,
                    "feedback": feedback,
                    "feedback_path": feedback_path,
                    "row": row,
                    "area_growth_fraction": _area_growth_fraction(
                        _read_json(current_region), region_doc
                    ),
                }
            )

        selected, final_status, stop_reason = _select_trial(
            trial_results,
            current_feedback,
            args.boundary_resolution_profile,
        )
        if selected is None:
            break

        selected["row"]["selected"] = True
        candidate = selected["candidate"]
        side = int(candidate["side_index"])
        cumulative_by_side[side] = cumulative_by_side.get(side, 0.0) + float(candidate["displacement_km"])
        accepted_adjustments.append(
            {
                "iteration": adjustment_index,
                "candidate": candidate,
                "cumulative_side_expansion_km": cumulative_by_side[side],
                "unintended_frame_clip_length_m": _clip_length(selected["feedback"]),
                "area_growth_fraction": selected["area_growth_fraction"],
                "selection_reason": stop_reason,
            }
        )
        current_region = selected["region"]
        current_offshore = selected["offshore"]
        current_manifest = selected["manifest"]
        current_feedback = selected["feedback"]
        current_feedback_path = selected["feedback_path"]
        if stop_reason == "adaptive_failure_after_frame_clip_closed":
            break

    summary = {
        "schema_version": "region_bpoly_arc_feedback_loop_v1",
        "name": args.name,
        "final_status": final_status,
        "stop_reason": stop_reason or "maximum_adjustments_exhausted",
        "policy": {
            "max_adjustments": args.max_adjustments,
            "max_candidates_per_adjustment": args.max_candidates_per_adjustment,
            "max_side_expansion_km": args.max_side_expansion_km,
            "gshhs_lookahead_km": args.gshhs_lookahead_km,
            "gshhs_resolution": args.gshhs_resolution,
            "gshhs_levels": args.gshhs_levels,
            "boundary_resolution_profile": args.boundary_resolution_profile,
            "residual_boundary_policy": args.residual_boundary_policy,
        },
        "accepted_adjustments": accepted_adjustments,
        "cumulative_side_expansion_km": cumulative_by_side,
        "selected_outputs": {
            "region_bpoly_json": str(current_region),
            "offshore_artifacts_json": str(current_offshore),
            "bdry_arc_manifest": str(Path(current_manifest.get("outputs", {}).get("feedback_json", "")).parents[1] / "bdry_arc_manifest.json") if current_manifest else None,
            "feedback_json": str(current_feedback_path),
            **current_manifest.get("outputs", {}),
        },
        "final_feedback": current_feedback,
        "trials": rows,
    }
    summary_path = root / "feedback_loop_summary.json"
    csv_path = root / "feedback_loop_summary.csv"
    _write_json(summary_path, summary)
    _write_csv(csv_path, rows)
    print(json.dumps({"final_status": final_status, "stop_reason": summary["stop_reason"], "summary_json": str(summary_path), "summary_csv": str(csv_path)}, indent=2))
    return 0 if final_status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
