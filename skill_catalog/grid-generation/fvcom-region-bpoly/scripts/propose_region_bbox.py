from __future__ import annotations

import argparse
from pathlib import Path

from region_bbox.geometry import RegionBox
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
    if "chesapeake" in text:
        return RegionBox(-76.1, 38.1, 390, 250, 15, 115)
    if "aleut" in text or "aleuc" in text:
        return RegionBox(-175.0, 53.0, 2100, 360, 80, 180)
    if "hawaiian islands" in text or "hawaii islands" in text or "hawaii state" in text:
        return RegionBox(-157.8, 20.6, 760, 420, 120, 210)
    if "hawaii island" in text or "big island" in text or "otec" in text:
        return RegionBox(-155.45, 19.65, 230, 210, 25, 160)
    if "lake superior" in text:
        return RegionBox(-88.4, 47.7, 650, 300, 82, 0)
    if "lake ontario" in text:
        return RegionBox(-78.2, 43.8, 360, 160, 75, 0)
    if "lake erie" in text:
        return RegionBox(-81.1, 42.1, 430, 150, 75, 0)
    if "cook inlet" in text:
        return RegionBox(-151.7, 59.6, 460, 260, 25, 170)
    if "southeast alaska" in text or "se ak" in text:
        return RegionBox(-134.0, 55.5, 1050, 620, 130, 215)
    if "columbia" in text:
        return RegionBox(-124.15, 46.15, 140, 110, 82, 270)
    if "hudson" in text:
        return RegionBox(-73.95, 41.25, 240, 110, 10, 165)
    if "san francisco" in text:
        return RegionBox(-122.35, 37.75, 240, 190, 30, 250)
    return RegionBox(-75.0, 39.0, 200, 150, 0, 90)


def load_request(args) -> dict:
    if args.request_json:
        return read_json(args.request_json)
    if args.request_text:
        return {"request": args.request_text}
    return {"request": ""}


def main() -> None:
    ap = argparse.ArgumentParser(description="Propose a RegionBox for FVCOM preprocessing.")
    ap.add_argument("--request-json")
    ap.add_argument("--request-text")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--center-lon", type=float)
    ap.add_argument("--center-lat", type=float)
    ap.add_argument("--length-km", type=float)
    ap.add_argument("--width-km", type=float)
    ap.add_argument("--orientation-deg", type=float)
    ap.add_argument("--offshore-azimuth-deg", type=float)
    ap.add_argument("--basemap-provider", default="topo")
    ap.add_argument("--full-side-review", action="store_true")
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--override-iteration-cap", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    request = load_request(args)
    req_path = write_json(run_dir / f"{args.name}_request.json", request)

    if args.center_lon is not None:
        region = RegionBox(
            args.center_lon,
            args.center_lat,
            args.length_km,
            args.width_km,
            args.orientation_deg,
            args.offshore_azimuth_deg,
        )
    else:
        region = guess_region_box(request)

    ingredients = required_ingredients(request)
    coverage = score_region_box(region, ingredients)
    notes = mission_scope_notes(request)
    warnings = list(region.map_visibility_warnings())
    for note in notes:
        if note.get("status") in {"requires_review", "requires_island_chain"}:
            warnings.append(note["message"])

    map_path = run_dir / f"{args.name}_candidate_map.png"
    basemap = plot_region_map(map_path, region, ingredients, title=f"{args.name} RegionBox", basemap_provider=args.basemap_provider)
    focus_path = run_dir / f"{args.name}_candidate_focus_map.png"
    focus_bbox = region.envelope_bbox()
    plot_region_map(focus_path, region, ingredients, title=f"{args.name} focus", bbox=focus_bbox, basemap_provider=args.basemap_provider)

    if args.full_side_review:
        side_indices = [0, 1, 2, 3]
        fractions = [0.15, 0.5, 0.85]
        mode = "full_all_sides"
    else:
        side_indices = [region.offshore_side_index()]
        fractions = [0.125, 0.375, 0.625, 0.875]
        mode = "fast_open_side"
    side_reviews = side_focus_records(region, run_dir, args.name, side_indices, fractions, basemap_provider=args.basemap_provider)

    score = {
        "ingredient_coverage": coverage,
        "warnings": warnings,
        "mission_scope_notes": notes,
        "map_visibility": {
            "candidate_map_path": str(map_path),
            "crosses_antimeridian": region.crosses_antimeridian(),
            "warnings": region.map_visibility_warnings(),
            "requires_visible_box_review": True,
        },
    }
    score_path = write_json(run_dir / f"{args.name}_candidate_score.json", score)
    ingredient_path = write_json(run_dir / f"{args.name}_ingredient_coverage.json", coverage)
    candidate = {
        "name": args.name,
        "created_at_utc": utc_now(),
        "iteration": args.iteration,
        "default_iteration_cap": {"max_iterations": 2, "rule": "one proposal, one revision, then final decision unless required ingredients are missing"},
        "review_status": "review_pending",
        "request_path": str(req_path),
        "region_box": region.to_dict(),
        "ingredient_coverage": coverage,
        "ingredient_coverage_path": str(ingredient_path),
        "mission_scope_notes": notes,
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
    cand_path = write_json(run_dir / f"{args.name}_region_box_candidate.json", candidate)
    summary = run_dir / f"{args.name}_candidate_summary.md"
    summary.write_text(
        "\n".join(
            [
                f"# {args.name} RegionBox Candidate",
                f"- Candidate JSON: `{cand_path}`",
                f"- Required ingredients inside: `{coverage['all_required_inside']}`",
                f"- Missing required IDs: `{coverage['missing_required_ids']}`",
                f"- Side review mode: `{mode}`",
                f"- Crosses antimeridian: `{region.crosses_antimeridian()}`",
                f"- Mission-scope notes: `{notes}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote candidate: {cand_path}")


if __name__ == "__main__":
    main()
