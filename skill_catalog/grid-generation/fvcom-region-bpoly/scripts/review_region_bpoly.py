from __future__ import annotations

import argparse
from pathlib import Path

from region_bbox.geometry import RegionBPoly
from region_bbox.io import read_json, utc_now, write_json


def _mission_required(candidate: dict) -> bool:
    return any(n.get("status") in {"requires_review", "requires_island_chain"} for n in candidate.get("mission_scope_notes", []))


def _parse_side_statuses(values: list[str]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for value in values:
        if ":" in value:
            idx, status = value.split(":", 1)
            out[int(idx)] = {"side_index": int(idx), "status": status, "notes": ""}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Review a RegionBPoly candidate.")
    ap.add_argument("--candidate-json", required=True)
    ap.add_argument("--decision", required=True, choices=["pass", "revise", "fail"])
    ap.add_argument("--domain-type-note-json")
    ap.add_argument("--map-visibility-status", choices=["pass", "fail"])
    ap.add_argument("--map-visibility-notes", default="")
    ap.add_argument("--mission-scope-status", choices=["pass", "revise", "not_applicable"], default="not_applicable")
    ap.add_argument("--mission-scope-notes", default="")
    ap.add_argument("--side-status", action="append", default=[], help="SIDE_INDEX:pass|expand_required|unresolved_autonomous_failure")
    ap.add_argument("--side-review-all-pass", action="store_true")
    ap.add_argument("--single-open-boundary-status", choices=["pass", "revise", "not_applicable"], default="not_applicable")
    ap.add_argument("--single-open-boundary-notes", default="")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    candidate = read_json(args.candidate_json)
    bpoly = RegionBPoly.from_dict(candidate["region_bpoly"])
    name = candidate["name"]
    run_dir = Path(args.candidate_json).parent
    side_statuses = _parse_side_statuses(args.side_status)
    required_sides = candidate.get("side_focus_required_side_indices", [])
    if args.side_review_all_pass:
        for idx in required_sides:
            side_statuses[idx] = {"side_index": idx, "status": "pass", "notes": "Marked pass by reviewer."}

    blocks = []
    if args.decision == "pass":
        if not candidate.get("ingredient_coverage", {}).get("all_required_inside", False):
            blocks.append("required ingredients are missing")
        if args.map_visibility_status != "pass":
            blocks.append("map visibility has not passed")
        if bpoly.crosses_antimeridian() and args.map_visibility_status != "pass":
            blocks.append("antimeridian/crossing candidate needs explicit map visibility pass")
        for idx in required_sides:
            if side_statuses.get(idx, {}).get("status") != "pass":
                blocks.append(f"required side {idx} is not pass")
        if _mission_required(candidate) and args.mission_scope_status != "pass":
            blocks.append("mission-scope gate requires pass")
        note = read_json(args.domain_type_note_json) if args.domain_type_note_json else {}
        if note.get("domain_type") == "coastal" and not note.get("open_boundary_reference"):
            blocks.append("coastal domain missing open-boundary reference")
        if args.single_open_boundary_status not in {"pass", "not_applicable"}:
            blocks.append("single-open-boundary status is not pass")
        if blocks:
            raise SystemExit("Cannot pass RegionBPoly: " + "; ".join(blocks))

    note = read_json(args.domain_type_note_json) if args.domain_type_note_json else {}
    review = {
        "name": name,
        "reviewed_at_utc": utc_now(),
        "reviewer": "codex-agent-visual-inspection",
        "decision": args.decision,
        "candidate_json": args.candidate_json,
        "domain_type_note_json": args.domain_type_note_json,
        "domain_type": note.get("domain_type"),
        "boundary_policy": note.get("boundary_policy"),
        "open_boundary_reference": note.get("open_boundary_reference"),
        "map_path": candidate.get("map_path"),
        "focus_map_path": candidate.get("focus_map_path"),
        "map_visibility_status": args.map_visibility_status,
        "map_visibility_notes": args.map_visibility_notes,
        "mission_scope_status": args.mission_scope_status,
        "mission_scope_notes": args.mission_scope_notes,
        "mission_scope_candidate_notes": candidate.get("mission_scope_notes", []),
        "side_reviews": [side_statuses[i] for i in sorted(side_statuses)],
        "single_open_boundary_status": args.single_open_boundary_status,
        "single_open_boundary_notes": args.single_open_boundary_notes,
        "ingredient_coverage": candidate.get("ingredient_coverage"),
        "notes": args.notes,
    }
    review_path = write_json(run_dir / f"{name}_visual_review.json", review)
    md = run_dir / f"{name}_visual_review.md"
    md.write_text(
        f"# RegionBPoly Visual Review\n\n- Decision: `{args.decision}`\n- Map visibility: `{args.map_visibility_status}`\n- Mission scope: `{args.mission_scope_status}`\n- Single open boundary: `{args.single_open_boundary_status}`\n\n{args.notes}\n",
        encoding="utf-8",
    )
    if args.decision == "pass":
        final = candidate["region_bpoly"]
        final.update(
            {
                "review_json": str(review_path),
                "domain_type_note_json": args.domain_type_note_json,
                "domain_type": note.get("domain_type"),
                "boundary_policy": note.get("boundary_policy"),
                "open_boundary_reference": note.get("open_boundary_reference"),
                "map_visibility_status": args.map_visibility_status,
                "mission_scope_status": args.mission_scope_status,
                "mission_scope_notes": candidate.get("mission_scope_notes", []),
            }
        )
        out = write_json(run_dir / f"{name}_region_bpoly.json", final)
        write_json(run_dir / "region_bpoly.json", final)
        print(f"Wrote accepted RegionBPoly: {out}")
    else:
        print(f"Wrote visual review: {review_path}")


if __name__ == "__main__":
    main()

