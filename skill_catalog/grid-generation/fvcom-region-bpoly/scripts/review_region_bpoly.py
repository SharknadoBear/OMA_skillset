from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from region_bbox.delivery import write_standard_delivery
from region_bbox.geometry import RegionBPoly
from region_bbox.io import canonical_sha256, file_sha256, read_json, utc_now, write_json


ALLOWED_SIDE_STATUSES = {"pass", "expand_required", "unresolved"}


def _mission_required(candidate: dict) -> bool:
    return any(n.get("status") in {"requires_review", "requires_island_chain"} for n in candidate.get("mission_scope_notes", []))


def _parse_indexed(values: list[str], label: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for value in values:
        if ":" not in value:
            raise SystemExit(f"{label} must use SIDE_INDEX:value")
        raw_idx, payload = value.split(":", 1)
        idx = int(raw_idx)
        if idx in out:
            raise SystemExit(f"duplicate {label} for side {idx}")
        out[idx] = payload.strip()
    return out


def _verify_map(binding: dict, label: str, failures: list[str]) -> None:
    path = Path(binding.get("map_path", ""))
    expected = binding.get("map_sha256")
    if not path.is_file():
        failures.append(f"{label} map is missing: {path}")
        return
    actual = file_sha256(path)
    if actual != expected:
        failures.append(f"{label} map hash is stale")
    if not binding.get("geography_usable", False):
        failures.append(f"{label} map has unusable background geography")


def _verify_request(final: dict, request: dict) -> list[str]:
    failures: list[str] = []
    if request.get("schema_version") != "region_bpoly_land_side_visual_review_request_v1":
        failures.append("land-side review request schema is missing or unsupported")
        return failures
    iteration = int(request.get("iteration", 0))
    if iteration not in {1, 2, 3}:
        failures.append("land-side review iteration must be 1, 2, or 3")
    if int(request.get("maximum_iterations", 0)) != 3:
        failures.append("land-side review maximum_iterations binding is not 3")
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
        failures.append("review request does not require start/middle/end views")
    for idx in request.get("required_land_side_indices", []):
        records = request.get("side_views", {}).get(str(idx), [])
        if {record.get("position") for record in records} != required_positions:
            failures.append(f"side {idx} does not have exactly the required positions")
        for record in records:
            _verify_map(record, f"side {idx} {record.get('position')}", failures)
    offshore = int(request.get("offshore_side_index", -1))
    if offshore in {int(i) for i in request.get("required_land_side_indices", [])}:
        failures.append("selected offshore side is incorrectly included in the land-side gate")
    return failures


def _compact_review_map(path: Path, request: dict, statuses: dict[int, str], notes: dict[int, str]) -> None:
    panels: list[tuple[str, Path]] = [("whole domain", Path(request["whole_domain_map"]["map_path"]))]
    order = {"start": 0, "middle": 1, "end": 2}
    for idx in request.get("required_land_side_indices", []):
        records = sorted(request.get("side_views", {}).get(str(idx), []), key=lambda item: order.get(item.get("position"), 9))
        for record in records:
            panels.append((f"side {idx} {record.get('position')}", Path(record["map_path"])))
    cols = 3
    rows = max(1, math.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.8 * rows))
    flat = list(getattr(axes, "flat", [axes]))
    for ax, (label, image_path) in zip(flat, panels):
        ax.imshow(plt.imread(image_path))
        side_text = ""
        if label.startswith("side "):
            idx = int(label.split()[1])
            side_text = f" | {statuses.get(idx)}: {notes.get(idx, '')}"
        ax.set_title(label + side_text, fontsize=9)
        ax.axis("off")
    for ax in flat[len(panels) :]:
        ax.axis("off")
    fig.suptitle(f"RegionBPoly land-side visual review, iteration {request.get('iteration')}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_final_documents(run_dir: Path, name: str, final: dict) -> None:
    write_standard_delivery(run_dir, name, final, write_name_alias=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Finalize the hash-bound RegionBPoly coastal land-side visual gate.")
    ap.add_argument("--candidate-json", required=True, help="Provisional top-level region_bpoly.json from run_region_bpoly.py.")
    ap.add_argument("--decision", required=True, choices=["pass", "revise", "fail"])
    ap.add_argument("--map-visibility-status", required=True, choices=["pass", "fail"])
    ap.add_argument("--map-visibility-notes", default="")
    ap.add_argument("--mission-scope-status", choices=["pass", "revise", "not_applicable"], default="not_applicable")
    ap.add_argument("--mission-scope-notes", default="")
    ap.add_argument("--side-status", action="append", default=[], help="SIDE_INDEX:pass|expand_required|unresolved")
    ap.add_argument("--side-note", action="append", default=[], help="SIDE_INDEX:concise geographic evidence")
    ap.add_argument("--single-open-boundary-status", choices=["pass", "revise", "not_applicable"], default="not_applicable")
    ap.add_argument("--single-open-boundary-notes", default="")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    input_path = Path(args.candidate_json)
    final = read_json(input_path)
    run_dir = input_path.parent
    name = final.get("name", "region")
    if final.get("domain_type") != "coastal":
        raise SystemExit("The strict land-side visual finalizer applies only to coastal RegionBPoly candidates")
    request = final.get("land_side_visual_review_request") or {}
    iteration = int(request.get("iteration", 0))
    statuses = _parse_indexed(args.side_status, "--side-status")
    notes = _parse_indexed(args.side_note, "--side-note")
    required_sides = {int(idx) for idx in request.get("required_land_side_indices", [])}
    offshore_side = int(request.get("offshore_side_index", -1))
    failures = _verify_request(final, request)

    for idx, status in statuses.items():
        if status not in ALLOWED_SIDE_STATUSES:
            failures.append(f"side {idx} has unsupported status {status!r}")
        if idx == offshore_side and status == "expand_required":
            failures.append("the selected offshore side cannot request expansion")
        if idx not in required_sides:
            failures.append(f"side {idx} is not a required land side")
    if set(statuses) != required_sides:
        failures.append("every required land side must have exactly one status")
    if set(notes) != required_sides or any(not value.strip() for value in notes.values()):
        failures.append("every required land side must have concise nonempty geographic evidence")
    if args.map_visibility_status != "pass":
        failures.append("reviewer did not pass map visibility")
    coverage = final.get("qa", {}).get("ingredient_coverage", {})
    if not coverage.get("all_required_inside", False):
        failures.append("required feature coverage is incomplete")
    if not final.get("open_boundary_reference"):
        failures.append("coastal domain is missing its offshore-side reference")
    candidate_path = Path(request.get("candidate_json", ""))
    candidate = read_json(candidate_path) if candidate_path.is_file() else {}
    if _mission_required(candidate) and args.mission_scope_status != "pass":
        failures.append("mission-scope gate requires pass")
    if args.single_open_boundary_status == "revise":
        failures.append("offshore-side selection requires revision")

    nonpass = {idx: status for idx, status in statuses.items() if status != "pass"}
    if args.decision == "pass" and nonpass:
        failures.append("pass decision requires every land side to pass")
    if args.decision == "revise" and not any(status == "expand_required" for status in statuses.values()):
        failures.append("revise decision requires at least one expand_required land side")
    if args.decision == "fail" and not any(status == "unresolved" for status in statuses.values()):
        failures.append("fail decision requires at least one unresolved land side")
    if len([status for status in statuses.values() if status == "expand_required"]) > 1:
        failures.append("one repair iteration may expand only one named land side")

    exhausted = bool(nonpass) and iteration >= 3
    if exhausted:
        failures.append("land-side truncation remains unresolved after the third review attempt")

    clean_pass = args.decision == "pass" and not failures
    repair_requested = args.decision == "revise" and not failures and iteration < 3
    accepted_with_warnings = not clean_pass and not repair_requested

    review_map = run_dir / f"region_bpoly_land_side_review_i{iteration:02d}.png"
    maps_ok = not any("map" in failure.lower() or "geography" in failure.lower() for failure in failures)
    if maps_ok and request:
        _compact_review_map(review_map, request, statuses, notes)
    review = {
        "schema_version": "region_bpoly_land_side_visual_review_v1",
        "name": name,
        "reviewed_at_utc": utc_now(),
        "reviewer": "codex-agent-visual-inspection",
        "iteration": iteration,
        "maximum_iterations": 3,
        "decision": args.decision,
        "effective_decision": (
            "pass"
            if clean_pass
            else "expand_required"
            if repair_requested
            else "accepted_with_warnings"
        ),
        "candidate_json": str(input_path),
        "region_bpoly_sha256": request.get("region_bpoly_sha256"),
        "review_request": request,
        "map_visibility_status": args.map_visibility_status,
        "map_visibility_notes": args.map_visibility_notes,
        "mission_scope_status": args.mission_scope_status,
        "mission_scope_notes": args.mission_scope_notes,
        "side_reviews": [
            {"side_index": idx, "status": statuses.get(idx), "geographic_evidence": notes.get(idx, "")}
            for idx in sorted(required_sides)
        ],
        "single_open_boundary_status": args.single_open_boundary_status,
        "single_open_boundary_notes": args.single_open_boundary_notes,
        "review_map_path": str(review_map) if review_map.is_file() else None,
        "review_map_sha256": file_sha256(review_map) if review_map.is_file() else None,
        "validation_failures": failures,
        "notes": args.notes,
    }
    if repair_requested:
        expand_side = next(idx for idx, status in statuses.items() if status == "expand_required")
        review["next_action"] = {
            "operation": "expand_side",
            "side_index": expand_side,
            "next_iteration": iteration + 1,
            "constraints": "Move only this complete land side outward; do not rotate or globally scale inside the loop.",
        }

    review_path = write_json(run_dir / f"region_bpoly_land_side_review_i{iteration:02d}.json", review)
    final["land_side_visual_review"] = review
    final["land_side_visual_review_path"] = str(review_path)
    if repair_requested:
        final["final_status"] = "repair_required"
        final["status_reasons"] = ["land_side_expansion_required"]
        final["qa"]["land_side_visual_gate"] = {
            "status": "expand_required",
            "iteration": iteration,
            "review_json": str(review_path),
            "next_action": review["next_action"],
        }
        _write_final_documents(run_dir, name, final)
        print(f"Wrote land-side review requiring the authorized repair: {review_path}")
        return

    final["final_status"] = "pass"
    final["status_reasons"] = [] if clean_pass else ["land_side_visual_review_accepted_best_effort"]
    warning_messages = list(final.get("delivery_warnings", []))
    if accepted_with_warnings:
        warning_messages.extend(f"land-side review: {failure}" for failure in failures)
        warning_messages.extend(
            f"land-side review: side {idx} remained {status}"
            for idx, status in sorted(nonpass.items())
        )
        if not failures and not nonpass:
            warning_messages.append(f"land-side review decision {args.decision!r} was accepted as best effort")
    final["delivery_warnings"] = list(dict.fromkeys(warning_messages))
    final.setdefault("qa", {})["delivery_warnings"] = final["delivery_warnings"]
    final["qa"]["land_side_visual_gate"] = {
        "status": "pass" if clean_pass else "warning",
        "outcome": "pass" if clean_pass else "accepted_best_effort",
        "iteration": iteration,
        "review_json": str(review_path),
        "review_map": str(review_map) if review_map.is_file() else None,
        "validation_failures": failures,
    }
    final_review_path = write_json(run_dir / "region_bpoly_land_side_review.json", review)
    final["land_side_visual_review_path"] = str(final_review_path)
    if review_map.is_file():
        final_map_alias = run_dir / "region_bpoly_land_side_review.png"
        shutil.copyfile(review_map, final_map_alias)
        final["land_side_visual_review_map_path"] = str(final_map_alias)
    else:
        final["land_side_visual_review_map_path"] = None
    offshore_path = Path(final.get("offshore_boundary_artifacts_path", ""))
    if offshore_path.is_file():
        offshore = read_json(offshore_path)
        offshore["final_status"] = "pass"
        offshore["land_side_visual_review_status"] = "pass" if clean_pass else "warning"
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
        print(f"Wrote accepted RegionBPoly: {run_dir / 'region_bpoly.json'}")
    else:
        print(f"Wrote best-effort accepted RegionBPoly with retained review warnings: {run_dir / 'region_bpoly.json'}")


if __name__ == "__main__":
    main()
