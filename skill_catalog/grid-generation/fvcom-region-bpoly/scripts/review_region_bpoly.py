from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from region_bbox.delivery import write_standard_delivery
from region_bbox.geometry import RegionBPoly
from region_bbox.io import canonical_sha256, file_sha256, read_json, utc_now, write_json


def _yes(value: str) -> bool:
    return value == "yes"


def _verify_map(binding: dict, label: str, failures: list[str]) -> None:
    path = Path(binding.get("map_path", ""))
    expected = binding.get("map_sha256")
    if not path.is_file():
        failures.append(f"{label} map is missing: {path}")
        return
    if file_sha256(path) != expected:
        failures.append(f"{label} map hash is stale")
    if not binding.get("geography_usable", False):
        failures.append(f"{label} map has unusable background geography")


def _verify_request(final: dict, request: dict) -> list[str]:
    failures: list[str] = []
    if request.get("schema_version") != "region_bpoly_scientific_review_request_v1":
        failures.append("scientific review request schema is missing or unsupported")
        return failures
    iteration = int(request.get("iteration", 0))
    if iteration < 1:
        failures.append("scientific review iteration must be a positive integer")
    if request.get("iteration_policy") != "agent_decided_no_numeric_limit":
        failures.append("scientific review does not declare the agent-decided iteration policy")
    serialized_region = final.get("region_bpoly") or RegionBPoly.from_dict(final).to_dict()
    if canonical_sha256(serialized_region) != request.get("region_bpoly_sha256"):
        failures.append("RegionBPoly geometry is stale relative to the review request")
    candidate_path = Path(request.get("candidate_json", ""))
    if not candidate_path.is_file():
        failures.append(f"candidate JSON is missing: {candidate_path}")
    elif file_sha256(candidate_path) != request.get("candidate_json_sha256"):
        failures.append("candidate JSON hash is stale")
    _verify_map(request.get("whole_domain_map", {}), "whole-domain", failures)
    required_positions = set(request.get("required_positions", []))
    if required_positions != {"start", "middle", "end"}:
        failures.append("coastal scientific review does not provide start/middle/end context")
    for idx in request.get("required_land_side_indices", []):
        records = request.get("side_views", {}).get(str(idx), [])
        if {record.get("position") for record in records} != required_positions:
            failures.append(f"side {idx} does not have the expected contextual views")
        for record in records:
            _verify_map(record, f"side {idx} {record.get('position')}", failures)
    return failures


def _geometry_hash(path: Path) -> str:
    return canonical_sha256(RegionBPoly.from_dict(read_json(path)).to_dict())


def _compact_review_map(path: Path, request: dict) -> None:
    panels: list[tuple[str, Path]] = [("whole domain", Path(request["whole_domain_map"]["map_path"]))]
    order = {"start": 0, "middle": 1, "end": 2}
    for idx in request.get("required_land_side_indices", []):
        records = sorted(
            request.get("side_views", {}).get(str(idx), []),
            key=lambda item: order.get(item.get("position"), 9),
        )
        for record in records:
            panels.append((f"side {idx} {record.get('position')}", Path(record["map_path"])))
    cols = 3
    rows = max(1, math.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.8 * rows))
    flat = list(getattr(axes, "flat", [axes]))
    for ax, (label, image_path) in zip(flat, panels):
        ax.imshow(plt.imread(image_path))
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    for ax in flat[len(panels) :]:
        ax.axis("off")
    fig.suptitle(f"RegionBPoly scientific review, cycle {request.get('iteration')}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_final_documents(run_dir: Path, name: str, final: dict) -> None:
    write_standard_delivery(run_dir, name, final, write_name_alias=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Record the coarse, agent-directed RegionBPoly scientific review.")
    ap.add_argument("--candidate-json", required=True, help="Provisional top-level region_bpoly.json from run_region_bpoly.py.")
    ap.add_argument("--problem-detected", required=True, choices=["yes", "no"])
    ap.add_argument("--problem-description", required=True, help="Concise scientific problem statement, or why none was found.")
    ap.add_argument("--change-required", required=True, choices=["yes", "no"])
    ap.add_argument("--geometry-changed", required=True, choices=["yes", "no"])
    ap.add_argument("--before-region-json", help="Previous RegionBPoly JSON used to verify a reported geometry change.")
    ap.add_argument("--change-description", default="", help="Concise description of the chosen or intended change; the method is not graded.")
    ap.add_argument("--scientifically-useful", required=True, choices=["yes", "no"])
    ap.add_argument("--scientific-rationale", required=True)
    ap.add_argument("--map-visibility-status", required=True, choices=["pass", "fail"])
    ap.add_argument("--map-visibility-notes", default="")
    ap.add_argument(
        "--no-meaningful-repair-remaining",
        action="store_true",
        help="Stop autonomously and deliver the latest valid geometry as accepted best effort.",
    )
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    input_path = Path(args.candidate_json)
    final = read_json(input_path)
    run_dir = input_path.parent
    name = final.get("name", "region")
    if final.get("domain_type") != "coastal":
        raise SystemExit("The hash-bound finalizer currently applies to coastal RegionBPoly candidates")
    request = final.get("scientific_review_request") or final.get("land_side_visual_review_request") or {}
    iteration = int(request.get("iteration", 0))
    failures = _verify_request(final, request)

    problem_detected = _yes(args.problem_detected)
    change_required = _yes(args.change_required)
    reported_changed = _yes(args.geometry_changed)
    reported_useful = _yes(args.scientifically_useful)
    if not args.problem_description.strip():
        failures.append("problem description must be nonempty")
    if not args.scientific_rationale.strip():
        failures.append("scientific rationale must be nonempty")
    if problem_detected != change_required:
        failures.append("a meaningful detected problem and the need for geometry change must agree")
    if reported_changed and not args.change_description.strip():
        failures.append("a reported geometry change requires a concise change description")

    current_hash = canonical_sha256(RegionBPoly.from_dict(final).to_dict())
    before_hash = None
    verified_changed = False
    if args.before_region_json:
        before_path = Path(args.before_region_json)
        if not before_path.is_file():
            failures.append(f"before-region JSON is missing: {before_path}")
        else:
            before_hash = _geometry_hash(before_path)
            verified_changed = before_hash != current_hash
    if reported_changed and before_hash is None:
        failures.append("geometry_changed=yes requires --before-region-json")
    if before_hash is not None and reported_changed != verified_changed:
        failures.append("reported geometry_changed does not match the before/current geometry hashes")
    if problem_detected and reported_useful and not verified_changed:
        failures.append("a detected problem cannot be accepted as resolved without a verified geometry change")
    if not problem_detected and reported_changed:
        failures.append("a geometry change must be tied to an explicitly recognized problem")
    if not reported_useful and not change_required and not args.no_meaningful_repair_remaining:
        failures.append("a non-useful candidate must request change or declare that no meaningful repair remains")

    coverage = final.get("qa", {}).get("ingredient_coverage", {})
    if not coverage.get("all_required_inside", False):
        failures.append("required feature coverage is incomplete")
    if not final.get("open_boundary_reference"):
        failures.append("coastal domain is missing its offshore-side reference")
    if args.map_visibility_status != "pass":
        failures.append("reviewer did not find the maps usable")

    clean_pass = reported_useful and not failures
    repair_requested = (
        not reported_useful
        and change_required
        and not args.no_meaningful_repair_remaining
        and not failures
    )
    accepted_with_warnings = not clean_pass and not repair_requested
    effective_useful = bool(clean_pass)

    review_map = run_dir / f"region_bpoly_land_side_review_i{iteration:02d}.png"
    maps_ok = not any("map" in failure.lower() or "geography" in failure.lower() for failure in failures)
    if maps_ok and request:
        _compact_review_map(review_map, request)
    review = {
        "schema_version": "region_bpoly_scientific_review_v1",
        "name": name,
        "reviewed_at_utc": utc_now(),
        "reviewer": "codex-agent-scientific-inspection",
        "iteration": iteration,
        "iteration_policy": "agent_decided_no_numeric_limit",
        "decision": "repair" if repair_requested else "accept" if clean_pass else "stop_best_effort",
        "effective_decision": "repair_required" if repair_requested else "pass" if clean_pass else "accepted_best_effort",
        "candidate_json": str(input_path),
        "region_bpoly_sha256": current_hash,
        "review_request": request,
        "problem_detected": problem_detected,
        "problem_description": args.problem_description,
        "change_required": change_required,
        "geometry_changed": reported_changed,
        "geometry_change_verified": verified_changed,
        "change_description": args.change_description,
        "geometry_evidence": {
            "before_region_json": args.before_region_json,
            "before_region_bpoly_sha256": before_hash,
            "current_region_bpoly_sha256": current_hash,
        },
        "scientifically_useful": reported_useful,
        "effective_scientifically_useful": effective_useful,
        "scientific_rationale": args.scientific_rationale,
        "map_visibility_status": args.map_visibility_status,
        "map_visibility_notes": args.map_visibility_notes,
        "no_meaningful_repair_remaining": bool(args.no_meaningful_repair_remaining),
        "review_map_path": str(review_map) if review_map.is_file() else None,
        "review_map_sha256": file_sha256(review_map) if review_map.is_file() else None,
        "validation_failures": failures,
        "notes": args.notes,
    }
    if repair_requested:
        review["next_action"] = {
            "operation": "agent_selected_repair",
            "next_iteration": iteration + 1,
            "constraints": "Choose any scientifically defensible geometry change; preserve only valid four-corner geometry and required mission features.",
        }

    review_path = write_json(run_dir / f"region_bpoly_land_side_review_i{iteration:02d}.json", review)
    final["scientific_review"] = review
    final["scientific_review_path"] = str(review_path)
    final["land_side_visual_review"] = review
    final["land_side_visual_review_path"] = str(review_path)
    if repair_requested:
        final["final_status"] = "repair_required"
        final["status_reasons"] = ["agent_selected_geometry_repair_required"]
        final.setdefault("qa", {})["scientific_review"] = {
            "status": "repair_required",
            "iteration": iteration,
            "review_json": str(review_path),
            "next_action": review["next_action"],
        }
        _write_final_documents(run_dir, name, final)
        print(f"Wrote scientific review requesting an agent-selected repair: {review_path}")
        return

    final["final_status"] = "pass"
    final["status_reasons"] = [] if clean_pass else ["scientific_review_accepted_best_effort"]
    warning_messages = list(final.get("delivery_warnings", []))
    if accepted_with_warnings:
        warning_messages.extend(f"scientific review: {failure}" for failure in failures)
        if not reported_useful:
            warning_messages.append("scientific review: latest valid RegionBPoly was not judged scientifically useful")
        if args.no_meaningful_repair_remaining:
            warning_messages.append("scientific review: agent reported no meaningful repair remaining")
    final["delivery_warnings"] = list(dict.fromkeys(warning_messages))
    final.setdefault("qa", {})["delivery_warnings"] = final["delivery_warnings"]
    final["qa"]["scientific_review"] = {
        "status": "pass" if clean_pass else "warning",
        "outcome": "pass" if clean_pass else "accepted_best_effort",
        "scientifically_useful": effective_useful,
        "iteration": iteration,
        "review_json": str(review_path),
        "review_map": str(review_map) if review_map.is_file() else None,
        "validation_failures": failures,
    }
    final["qa"]["land_side_visual_gate"] = dict(final["qa"]["scientific_review"])
    final_review_path = write_json(run_dir / "region_bpoly_land_side_review.json", review)
    final["scientific_review_path"] = str(final_review_path)
    final["land_side_visual_review_path"] = str(final_review_path)
    if review_map.is_file():
        final_map_alias = run_dir / "region_bpoly_land_side_review.png"
        shutil.copyfile(review_map, final_map_alias)
        final["scientific_review_map_path"] = str(final_map_alias)
        final["land_side_visual_review_map_path"] = str(final_map_alias)
    else:
        final["scientific_review_map_path"] = None
        final["land_side_visual_review_map_path"] = None
    offshore_path = Path(final.get("offshore_boundary_artifacts_path", ""))
    if offshore_path.is_file():
        offshore = read_json(offshore_path)
        offshore["final_status"] = "pass"
        offshore["scientific_review_status"] = "pass" if clean_pass else "warning"
        offshore["scientific_review_path"] = str(final_review_path)
        offshore["land_side_visual_review_status"] = offshore["scientific_review_status"]
        offshore["land_side_visual_review_path"] = str(final_review_path)
        write_json(offshore_path, offshore)
    _write_final_documents(run_dir, name, final)
    if clean_pass and final.get("mode") == "execute":
        intermediate = Path(final.get("intermediate_dir") or run_dir / "intermediate")
        if intermediate.is_dir():
            shutil.rmtree(intermediate)
        final["intermediate_dir"] = None
        _write_final_documents(run_dir, name, final)
    if clean_pass:
        print(f"Wrote scientifically accepted RegionBPoly: {run_dir / 'region_bpoly.json'}")
    else:
        print(f"Wrote best-effort accepted RegionBPoly with scientific warnings: {run_dir / 'region_bpoly.json'}")


if __name__ == "__main__":
    main()
