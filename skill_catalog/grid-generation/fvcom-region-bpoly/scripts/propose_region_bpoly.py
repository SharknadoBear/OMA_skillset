from __future__ import annotations

import argparse
from pathlib import Path

from region_bbox.geometry import RegionBPoly, RegionBox
from region_bbox.ingredients import mission_scope_notes, request_text, required_ingredients
from region_bbox.io import read_json, utc_now, write_json
from region_bbox.plot import plot_region_map, side_focus_records
from region_bbox.scoring import score_region_box


def guess_region_box(request: dict | str) -> RegionBox:
    text = request_text(request).lower()
    if "puget" in text or "salish" in text:
        if any(k in text for k in ["tidal energy", "all tidal", "all the tidal", "connect", "salish"]):
            return RegionBox(-123.35, 48.65, 650, 430, 10, 270)
        return RegionBox(-123.0, 47.85, 260, 180, 15, 270)
    if "long island" in text or "hypoxia" in text:
        return RegionBox(-72.72, 40.85, 290, 240, 90, 180)
    if "murderkill" in text:
        return RegionBox(-75.35, 39.05, 45, 35, 25, 120)
    if "delaware" in text:
        return RegionBox(-74.65, 39.0, 270, 220, 20, 120)
    if "aleut" in text or "aleuc" in text:
        return RegionBox(-178.0, 53.2, 2200, 1050, 82, 172)
    if "hawaiian islands" in text or "hawaii islands" in text or "hawaii state" in text:
        return RegionBox(-157.8, 20.6, 760, 420, 120, 210)
    if "hawaii island" in text or "big island" in text or "otec" in text:
        return RegionBox(-155.34, 19.43, 240, 260, 0, 225)
    if "lake superior" in text:
        return RegionBox(-88.4, 47.7, 650, 300, 82, 0)
    if "cook inlet" in text:
        return RegionBox(-151.7, 59.6, 460, 260, 25, 170)
    if "southeast alaska" in text or "se ak" in text:
        return RegionBox(-134.0, 55.5, 1050, 620, 130, 215)
    return RegionBox(-75.0, 39.0, 200, 150, 0, 90)


def deform_bpoly(base: RegionBPoly, request: dict | str) -> tuple[RegionBPoly, list[str]]:
    """Apply simple four-corner deformations for known topology risks."""
    text = request_text(request).lower()
    notes: list[str] = []
    pts = base.polygon_lonlat()[:-1]
    offshore = base.offshore_azimuth_deg

    if "puget" in text and any(k in text for k in ["tidal energy", "all tidal", "salish", "connect"]):
        pts = [[-125.65, 46.8], [-121.25, 46.85], [-121.35, 50.65], [-125.25, 50.9]]
        offshore = 270.0
        notes.append("Deformed to a Salish-Sea-scale quadrilateral so San Juan/Strait of Georgia context is inside.")
    elif "long island" in text or "hypoxia" in text:
        pts = [[-70.98, 39.77], [-74.44, 39.77], [-74.44, 41.93], [-70.99, 41.93]]
        offshore = 180.0
        notes.append("Uses validated LIS broad Atlantic-facing quadrilateral.")
    elif "murderkill" in text:
        pts = [[-75.05, 38.85], [-75.70, 38.85], [-75.70, 39.32], [-75.05, 39.32]]
        offshore = 120.0
        notes.append("Deformed to include Murderkill River context plus Delaware Bay connection.")
    elif "aleut" in text or "aleuc" in text:
        pts = [[172.0, 48.9], [-162.0, 49.9], [-161.5, 57.6], [172.0, 56.7]]
        offshore = 172.0
        notes.append("Uses antimeridian-crossing Aleutian bpoly; maps must visibly show full polygon.")
    elif "hawaii island" in text or "big island" in text or "otec" in text:
        pts = [[-156.55, 18.45], [-154.2, 18.55], [-154.15, 20.25], [-156.35, 20.55]]
        offshore = 225.0
        notes.append("Deformed around Big Island OTEC corridors while staying south of Maui.")
    elif "lake superior" in text:
        pts = [[-92.6, 46.1], [-84.0, 46.6], [-84.0, 49.35], [-92.4, 49.05]]
        offshore = 0.0
        notes.append("Lake bpoly follows Lake Superior broad orientation; no ocean open boundary expected.")

    return RegionBPoly(pts, offshore, edge_labels=["open_or_south", "west_or_left", "north_or_inner", "east_or_right"]), notes


def load_request(args) -> dict:
    if args.request_json:
        return read_json(args.request_json)
    if args.request_text:
        return {"request": args.request_text}
    return {"request": ""}


def main() -> None:
    ap = argparse.ArgumentParser(description="Propose a four-sided RegionBPoly for FVCOM preprocessing.")
    ap.add_argument("--request-json")
    ap.add_argument("--request-text")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--polygon-lonlat", help="JSON list of four [lon,lat] vertices")
    ap.add_argument("--offshore-azimuth-deg", type=float)
    ap.add_argument("--basemap-provider", default="street")
    ap.add_argument("--full-side-review", action="store_true")
    ap.add_argument("--iteration", type=int, default=1)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    request = load_request(args)
    req_path = write_json(run_dir / f"{args.name}_request.json", request)

    if args.polygon_lonlat:
        import json

        bpoly = RegionBPoly(json.loads(args.polygon_lonlat), args.offshore_azimuth_deg or 90.0)
        deformation_notes = ["User supplied explicit four-corner polygon."]
    else:
        bpoly, deformation_notes = deform_bpoly(RegionBPoly.from_region_box(guess_region_box(request)), request)

    ingredients = required_ingredients(request)
    coverage = score_region_box(bpoly, ingredients)
    notes = mission_scope_notes(request)
    warnings = list(bpoly.map_visibility_warnings())
    for note in notes:
        if note.get("status") in {"requires_review", "requires_island_chain"}:
            warnings.append(note["message"])

    map_path = run_dir / f"{args.name}_candidate_map.png"
    basemap = plot_region_map(map_path, bpoly, ingredients, title=f"{args.name} RegionBPoly", basemap_provider=args.basemap_provider)
    focus_path = run_dir / f"{args.name}_candidate_focus_map.png"
    plot_region_map(focus_path, bpoly, ingredients, title=f"{args.name} focus", bbox=bpoly.envelope_bbox(), basemap_provider=args.basemap_provider)

    if args.full_side_review:
        side_indices, fractions, mode = [0, 1, 2, 3], [0.15, 0.5, 0.85], "full_all_sides"
    else:
        side_indices, fractions, mode = [bpoly.offshore_side_index()], [0.125, 0.375, 0.625, 0.875], "fast_open_side"
    side_reviews = side_focus_records(bpoly, run_dir, args.name, side_indices, fractions, basemap_provider=args.basemap_provider)

    score = {
        "ingredient_coverage": coverage,
        "warnings": warnings,
        "mission_scope_notes": notes,
        "deformation_notes": deformation_notes,
        "map_visibility": {
            "candidate_map_path": str(map_path),
            "crosses_antimeridian": bpoly.crosses_antimeridian(),
            "warnings": bpoly.map_visibility_warnings(),
            "requires_visible_box_review": True,
        },
    }
    score_path = write_json(run_dir / f"{args.name}_candidate_score.json", score)
    ingredient_path = write_json(run_dir / f"{args.name}_ingredient_coverage.json", coverage)
    candidate = {
        "name": args.name,
        "created_at_utc": utc_now(),
        "iteration": args.iteration,
        "review_status": "needs_review",
        "request_path": str(req_path),
        "region_bpoly": bpoly.to_dict(),
        "region_box_compatibility": bpoly.to_dict(),
        "ingredient_coverage": coverage,
        "ingredient_coverage_path": str(ingredient_path),
        "mission_scope_notes": notes,
        "deformation_notes": deformation_notes,
        "map_visibility": score["map_visibility"],
        "map_path": str(map_path),
        "focus_map_path": str(focus_path),
        "basemap": basemap,
        "score_path": str(score_path),
        "side_focus_mode": mode,
        "side_focus_count": len(side_reviews),
        "side_focus_required_side_indices": side_indices,
        "side_focus_reviews": side_reviews,
    }
    cand_path = write_json(run_dir / f"{args.name}_region_bpoly_candidate.json", candidate)
    write_json(run_dir / "region_bpoly_candidate.json", candidate)
    summary = run_dir / f"{args.name}_candidate_summary.md"
    summary.write_text(
        "\n".join(
            [
                f"# {args.name} RegionBPoly Candidate",
                f"- Candidate JSON: `{cand_path}`",
                f"- Required ingredients inside: `{coverage['all_required_inside']}`",
                f"- Missing required IDs: `{coverage['missing_required_ids']}`",
                f"- Side review mode: `{mode}`",
                f"- Crosses antimeridian: `{bpoly.crosses_antimeridian()}`",
                f"- Deformation notes: `{deformation_notes}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote candidate: {cand_path}")


if __name__ == "__main__":
    main()
