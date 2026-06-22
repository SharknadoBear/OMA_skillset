from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from propose_region_bpoly import deform_bpoly, guess_region_box, load_request
from region_bbox.geometry import RegionBPoly
from region_bbox.ingredients import mission_scope_notes, required_ingredients, request_text
from region_bbox.io import utc_now, write_json
from region_bbox.plot import plot_region_map, side_focus_records
from region_bbox.scoring import score_region_box


POLICY = {
    "coastal": "coastal_arc_with_land_anchors",
    "island": "offshore_loop_no_land_anchors",
    "lake": "no_open_boundary",
    "unresolved_autonomous_failure": "unresolved",
}


def infer_domain_type(request: dict | str) -> str:
    text = request_text(request).lower()
    if "lake " in text or "lake_" in text or "lake superior" in text or "lake ontario" in text or "lake erie" in text:
        return "lake"
    if any(k in text for k in ["island", "aleut", "aleuc", "hawaii", "otec"]):
        return "island"
    return "coastal"


def write_intermediate(intermediate: Path, name: str, request: dict, bpoly: RegionBPoly, ingredients: list[dict], coverage: dict, mission_notes: list[dict], deformation_notes: list[str], basemap_provider: str, full_side_review: bool) -> dict:
    intermediate.mkdir(parents=True, exist_ok=True)
    request_path = write_json(intermediate / f"{name}_request.json", request)
    map_path = intermediate / f"{name}_candidate_map.png"
    basemap = plot_region_map(map_path, bpoly, ingredients, title=f"{name} RegionBPoly candidate", basemap_provider=basemap_provider)
    focus_path = intermediate / f"{name}_candidate_focus_map.png"
    plot_region_map(focus_path, bpoly, ingredients, title=f"{name} candidate focus", bbox=bpoly.envelope_bbox(), basemap_provider=basemap_provider)

    if full_side_review:
        side_indices, fractions, mode = [0, 1, 2, 3], [0.15, 0.5, 0.85], "full_all_sides"
    else:
        side_indices, fractions, mode = [bpoly.offshore_side_index()], [0.125, 0.375, 0.625, 0.875], "fast_open_side"
    side_reviews = side_focus_records(bpoly, intermediate, name, side_indices, fractions, basemap_provider=basemap_provider)

    candidate = {
        "name": name,
        "created_at_utc": utc_now(),
        "review_status": "intermediate",
        "request_path": str(request_path),
        "region_bpoly": bpoly.to_dict(),
        "ingredient_coverage": coverage,
        "mission_scope_notes": mission_notes,
        "deformation_notes": deformation_notes,
        "map_path": str(map_path),
        "focus_map_path": str(focus_path),
        "basemap": basemap,
        "side_focus_mode": mode,
        "side_focus_count": len(side_reviews),
        "side_focus_required_side_indices": side_indices,
        "side_focus_reviews": side_reviews,
    }
    write_json(intermediate / f"{name}_region_bpoly_candidate.json", candidate)
    write_json(intermediate / f"{name}_candidate_score.json", {"ingredient_coverage": coverage, "mission_scope_notes": mission_notes, "deformation_notes": deformation_notes})
    write_json(intermediate / f"{name}_ingredient_coverage.json", coverage)
    return candidate


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the streamlined RegionBPoly workflow.")
    ap.add_argument("--request-json")
    ap.add_argument("--request-text")
    ap.add_argument("--run-dir", required=True, help="Final case output folder.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["execute", "test"], default="execute")
    ap.add_argument("--domain-type", choices=list(POLICY))
    ap.add_argument("--open-boundary-reference", nargs=2, type=float, metavar=("LON", "LAT"))
    ap.add_argument("--basemap-provider", default="street")
    ap.add_argument("--full-side-review", action="store_true")
    args = ap.parse_args()

    case_dir = Path(args.run_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    intermediate = case_dir / "intermediate"
    if intermediate.exists():
        shutil.rmtree(intermediate)

    request = load_request(args)
    bpoly, deformation_notes = deform_bpoly(RegionBPoly.from_region_box(guess_region_box(request)), request)
    ingredients = required_ingredients(request)
    coverage = score_region_box(bpoly, ingredients)
    mission_notes = mission_scope_notes(request)
    domain_type = args.domain_type or infer_domain_type(request)

    ref_guess = args.open_boundary_reference
    if not ref_guess and domain_type == "coastal":
        ref_guess = bpoly.offshore_edge_midpoint_lonlat()
    open_ref = None
    if ref_guess:
        snap = bpoly.snap_point_to_edge(float(ref_guess[0]), float(ref_guess[1]))
        open_ref = {
            "role": "arc_reference_point",
            "source": "run_region_bpoly",
            "guess": {"lon": float(ref_guess[0]), "lat": float(ref_guess[1])},
            "snapped": snap.get("snapped"),
            "snap_distance_m": snap.get("snap_distance_m"),
            "side_index": snap.get("side_index"),
            "side_name": snap.get("side_name"),
            "notes": "Approximate guide for downstream domain/open-boundary tools.",
        }

    candidate = write_intermediate(intermediate, args.name, request, bpoly, ingredients, coverage, mission_notes, deformation_notes, args.basemap_provider, args.full_side_review)

    blocks: list[str] = []
    if not coverage.get("all_required_inside", False):
        blocks.append("required ingredients missing: " + ", ".join(coverage.get("missing_required_ids", [])))
    if domain_type == "coastal" and not open_ref:
        blocks.append("coastal domain missing open-boundary reference")
    if bpoly.crosses_antimeridian():
        blocks.append("antimeridian map visibility requires agent visual confirmation")

    final_status = "pass" if not blocks else "needs_review"
    final_map = case_dir / "region_bpoly_final_map.png"
    basemap = plot_region_map(
        final_map,
        bpoly,
        ingredients,
        title=f"{args.name} final RegionBPoly",
        basemap_provider=args.basemap_provider,
        open_boundary_reference=open_ref,
    )

    final = {
        "schema_version": "region_bpoly_final_v1",
        "object_type": "RegionBPolyFinal",
        "name": args.name,
        "created_at_utc": utc_now(),
        "mode": args.mode,
        "final_status": final_status,
        "status_reasons": blocks,
        "region_bpoly": bpoly.to_dict(),
        "polygon_lonlat": bpoly.polygon_lonlat(),
        "envelope_bbox": bpoly.envelope_bbox(),
        "domain_type": domain_type,
        "boundary_policy": POLICY[domain_type],
        "open_boundary_reference": open_ref,
        "final_map_path": str(final_map),
        "final_map_basemap": basemap,
        "intermediate_dir": str(intermediate) if args.mode == "test" or final_status != "pass" else None,
        "qa": {
            "ingredient_coverage": {
                "all_required_inside": coverage.get("all_required_inside", False),
                "missing_required_ids": coverage.get("missing_required_ids", []),
                "required_count": coverage.get("required_count", 0),
                "ingredient_count": coverage.get("ingredient_count", 0),
            },
            "mission_scope_notes": mission_notes,
            "map_visibility_warnings": bpoly.map_visibility_warnings(),
            "side_focus_mode": candidate.get("side_focus_mode"),
            "side_focus_count": candidate.get("side_focus_count"),
        },
        "deformation_notes": deformation_notes,
        "downstream_contract": {
            "bathymetry_and_coastline_fetch": "Use envelope_bbox.",
            "domain_and_grid_generation": "Use polygon_lonlat / region_bpoly as controlling geometry.",
        },
    }
    out = write_json(case_dir / "region_bpoly.json", final)
    if args.mode == "execute" and final_status == "pass" and intermediate.exists():
        shutil.rmtree(intermediate)
    print(f"Wrote final RegionBPoly: {out}")
    if final_status != "pass":
        print("Final status needs review: " + "; ".join(blocks))


if __name__ == "__main__":
    main()

