from __future__ import annotations

import argparse
from pathlib import Path

from region_bbox.geometry import RegionBox
from region_bbox.io import read_json, utc_now, write_json
from region_bbox.plot import plot_region_map


POLICY = {
    "coastal": "coastal_arc_with_land_anchors",
    "island": "offshore_loop_no_land_anchors",
    "lake": "no_open_boundary",
    "unresolved_autonomous_failure": "unresolved",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Write RegionBox domain-type note.")
    ap.add_argument("--candidate-json", required=True)
    ap.add_argument("--domain-type", required=True, choices=list(POLICY))
    ap.add_argument("--open-boundary-reference", nargs=2, type=float, metavar=("LON", "LAT"))
    ap.add_argument("--notes", default="")
    ap.add_argument("--reviewer", default="codex-agent-visual-inspection")
    args = ap.parse_args()
    candidate = read_json(args.candidate_json)
    region = RegionBox.from_dict(candidate["region_box"])
    name = candidate["name"]
    run_dir = Path(args.candidate_json).parent
    if args.domain_type == "coastal" and not args.open_boundary_reference:
        raise SystemExit("coastal domain requires --open-boundary-reference lon lat")
    ref = None
    if args.open_boundary_reference:
        lon, lat = args.open_boundary_reference
        snap = region.snap_point_to_edge(lon, lat)
        ref = {
            "role": "arc_reference_point",
            "source": args.reviewer,
            "guess": {"lon": lon, "lat": lat},
            "snapped": snap.get("snapped"),
            "snap_distance_m": snap.get("snap_distance_m"),
            "side_index": snap.get("side_index"),
            "notes": "Approximate visual guide for later mesh/domain tools, not an exact boundary anchor.",
        }
    note = {
        "name": name,
        "created_at_utc": utc_now(),
        "candidate_json": args.candidate_json,
        "domain_type": args.domain_type,
        "boundary_policy": POLICY[args.domain_type],
        "open_boundary_required": args.domain_type in {"coastal", "island"},
        "land_anchors_required": args.domain_type == "coastal",
        "river_inputs_on_closed_boundary": args.domain_type in {"coastal", "lake"},
        "open_boundary_reference": ref,
        "reviewer": args.reviewer,
        "notes": args.notes,
        "ingredient_coverage": candidate.get("ingredient_coverage"),
        "warnings": [],
    }
    out = write_json(run_dir / f"{name}_domain_type_note.json", note)
    png = run_dir / f"{name}_domain_type_review.png"
    plot_region_map(png, region, [], title=f"{name} domain type: {args.domain_type}", basemap_provider="street")
    md = run_dir / f"{name}_domain_type_review.md"
    md.write_text(f"# Domain Type Review\n\n- Domain type: `{args.domain_type}`\n- Boundary policy: `{POLICY[args.domain_type]}`\n- Note JSON: `{out}`\n", encoding="utf-8")
    print(f"Wrote domain-type note: {out}")


if __name__ == "__main__":
    main()

