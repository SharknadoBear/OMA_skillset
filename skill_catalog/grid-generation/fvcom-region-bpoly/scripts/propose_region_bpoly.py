from __future__ import annotations

import argparse
from pathlib import Path

from region_bbox.geometry import RegionBPoly, RegionBox
from region_bbox.features import (
    bpoly_from_feature_boxes,
    cook_inlet_domain_variant,
    features_as_ingredients,
    features_to_geojson,
    fit_bpoly_to_feature_boxes,
    infer_target_region_features,
    is_complex_feature_request,
)
from region_bbox.ingredients import mission_scope_notes, request_text, required_ingredients
from region_bbox.io import read_json, utc_now, write_json
from region_bbox.map_policy import resolve_basemap_provider, side_focus_radius_km
from region_bbox.normalization import canonical_region_key, normalize_request_text
from region_bbox.plot import plot_region_map, side_focus_records
from region_bbox.scoring import score_bpoly_quality, score_region_box


def guess_region_box(request: dict | str) -> RegionBox:
    text = normalize_request_text(request)
    key = canonical_region_key(request)
    if key == "puget_salish":
        if any(k in text for k in ["tidal energy", "all tidal", "all the tidal", "connect", "salish"]):
            return RegionBox(-123.35, 48.65, 650, 430, 10, 270)
        return RegionBox(-123.0, 47.85, 260, 180, 15, 270)
    if key == "long_island_sound":
        return RegionBox(-72.72, 40.85, 290, 240, 90, 180)
    if key == "murderkill":
        return RegionBox(-75.38, 39.08, 38, 30, 25, 120)
    if key == "delaware":
        return RegionBox(-74.65, 39.0, 270, 220, 20, 120)
    if key == "aleutian":
        return RegionBox(-178.0, 53.2, 2200, 1050, 82, 172)
    if key == "hawaii_state":
        return RegionBox(-157.8, 20.6, 760, 420, 120, 210)
    if key == "hawaii_island":
        return RegionBox(-155.35, 19.45, 235, 245, 0, 225)
    if key == "lake_superior":
        return RegionBox(-88.4, 47.7, 650, 300, 82, 0)
    if key == "cook_inlet":
        return RegionBox(-151.7, 59.6, 460, 260, 25, 170)
    if key == "mobile_bay":
        return RegionBox(-88.25, 30.55, 180, 145, 5, 180)
    if key == "southeast_alaska":
        return RegionBox(-134.0, 55.5, 1050, 620, 130, 215)
    raise ValueError("unknown_region_no_feature_plan")


def _effective_request(request: dict | str, place_memory_enabled: bool) -> dict | str:
    if isinstance(request, dict):
        out = dict(request)
        out["_place_memory_enabled"] = bool(place_memory_enabled)
        return out
    return {"request": request, "_place_memory_enabled": bool(place_memory_enabled)}


def deform_bpoly(base: RegionBPoly, request: dict | str) -> tuple[RegionBPoly, list[str]]:
    """Apply simple four-corner deformations for known topology risks."""
    text = normalize_request_text(request)
    key = canonical_region_key(request)
    notes: list[str] = []
    pts = base.polygon_lonlat()[:-1]
    offshore = base.offshore_azimuth_deg

    if key == "puget_salish" and any(k in text for k in ["tidal energy", "all tidal", "salish", "connect"]):
        pts = [[-121.25, 46.85], [-125.65, 46.8], [-125.25, 50.9], [-121.35, 50.65]]
        offshore = 270.0
        notes.append("Deformed to a Salish-Sea-scale quadrilateral so San Juan/Strait of Georgia context is inside.")
    elif key == "long_island_sound":
        pts = [[-70.98, 39.77], [-74.44, 39.77], [-74.44, 41.93], [-70.99, 41.93]]
        offshore = 180.0
        notes.append("Uses validated LIS broad Atlantic-facing quadrilateral.")
    elif key == "murderkill":
        pts = [[-75.18, 38.96], [-75.62, 38.96], [-75.62, 39.23], [-75.18, 39.23]]
        offshore = 120.0
        notes.append("V2 small-estuary deformation tightens around the Murderkill river path and immediate Delaware Bay mouth.")
    elif key == "aleutian":
        pts = [[-162.0, 49.9], [172.0, 48.9], [172.0, 56.7], [-161.5, 57.6]]
        offshore = 172.0
        notes.append("Uses antimeridian-crossing Aleutian bpoly; maps must visibly show full polygon.")
    elif key == "hawaii_island":
        pts = [[-154.05, 18.15], [-156.45, 18.15], [-156.45, 20.36], [-154.05, 20.36]]
        offshore = 225.0
        notes.append("V2 Big-Island-only deformation keeps the northern edge below the Maui Nui neighboring-island guard.")
    elif key == "cook_inlet":
        if cook_inlet_domain_variant(request) == "cook_inlet_wave_fetch":
            pts = [[-148.35, 55.25], [-157.30, 55.30], [-156.15, 62.05], [-148.65, 62.10]]
            offshore = 180.0
            notes.append("V4 Cook Inlet wave-fetch deformation includes Kodiak, Augustine Island, Ursus Cove/Kamishak context, and a broad Gulf of Alaska wave apron while avoiding Prince William Sound overreach.")
        else:
            pts = [[-148.55, 58.88], [-153.35, 58.95], [-153.65, 61.75], [-148.75, 61.85]]
            offshore = 165.0
            notes.append("V3 Cook Inlet tidal-mouth deformation keeps the offshore side north/east of the Kodiak Island obstruction guard.")
    elif key == "lake_superior":
        pts = [[-84.0, 46.6], [-92.6, 46.1], [-92.4, 49.05], [-84.0, 49.35]]
        offshore = 0.0
        notes.append("Lake bpoly follows Lake Superior broad orientation; no ocean open boundary expected.")
    elif key == "mobile_bay":
        pts = [[-87.55, 29.92], [-88.95, 29.90], [-88.82, 31.25], [-87.70, 31.20]]
        offshore = 180.0
        notes.append("V4 Mobile Bay deformation keeps a Gulf-facing gate west enough to land beyond Horn Island while avoiding Perdido/Wolf Bay overreach.")

    labels = ["open_or_south", "west_or_left", "north_or_inner", "east_or_right"]
    if key.startswith("lake_"):
        labels = ["south_lake_edge", "west_lake_edge", "north_lake_edge", "east_lake_edge"]
    return RegionBPoly(pts, offshore, edge_labels=labels), notes


def known_repair_candidates(request: dict | str) -> list[dict]:
    """Return deterministic four-sided candidates for known v2 QA failure modes."""
    key = canonical_region_key(request)
    specs: list[tuple[str, list[list[float]], float, str]] = []
    if key == "cook_inlet":
        if cook_inlet_domain_variant(request) == "cook_inlet_wave_fetch":
            specs.append(
                (
                    "cook_inlet_wave_fetch_broad_repair",
                    [[-148.35, 55.25], [-157.30, 55.30], [-156.15, 62.05], [-148.65, 62.10]],
                    180.0,
                    "Includes Kodiak, Augustine Island, Ursus Cove/Kamishak west-side Cook Inlet, and a broad Gulf of Alaska wave-fetch apron without Prince William Sound overreach.",
                )
            )
        else:
            specs.append(
                (
                    "cook_inlet_tidal_mouth_kodiak_obstruction_repair",
                    [[-148.55, 58.88], [-153.35, 58.95], [-153.65, 61.75], [-148.75, 61.85]],
                    165.0,
                    "Keeps the selected offshore side out of the Kodiak Island obstruction guard while retaining Cook Inlet and a Gulf of Alaska mouth gate.",
                )
            )
    elif key == "murderkill":
        specs.append(
            (
                "murderkill_small_estuary_tight_repair",
                [[-75.18, 38.96], [-75.62, 38.96], [-75.62, 39.23], [-75.18, 39.23]],
                120.0,
                "Tightens the polygon around Murderkill River, tidal-creek upstream gate, and immediate Delaware Bay mouth.",
            )
        )
    elif key == "hawaii_island":
        specs.append(
            (
                "hawaii_big_island_neighbor_guard_repair",
                [[-154.05, 18.15], [-156.45, 18.15], [-156.45, 20.36], [-154.05, 20.36]],
                225.0,
                "Keeps a Big-Island-only bpoly below Maui Nui and neighboring-island obstruction guards.",
            )
        )
    elif key == "mobile_bay":
        specs.append(
            (
                "mobile_bay_gate_landing_repair",
                [[-87.55, 29.92], [-88.95, 29.90], [-88.82, 31.25], [-87.70, 31.20]],
                180.0,
                "Extends the Gulf-facing gate west enough for a solid coastal landing while avoiding unnecessary Perdido/Wolf Bay inclusion.",
            )
        )
    return [
        {
            "id": candidate_id,
            "region_bpoly": RegionBPoly(points, offshore, edge_labels=["open_or_south", "west_or_left", "north_or_inner", "east_or_right"]),
            "notes": note,
        }
        for candidate_id, points, offshore, note in specs
    ]


def load_request(args) -> dict:
    if args.request_json:
        return read_json(args.request_json)
    if args.request_text:
        return {"request": args.request_text}
    return {"request": ""}


def select_review_depth(
    request: dict | str,
    features_doc: dict,
    coverage: dict,
    bpoly: RegionBPoly,
    requested: str,
    first_coverage_failed: bool = False,
) -> tuple[str, list[str]]:
    if requested in {"fast", "full"}:
        return requested, [f"explicit review-depth={requested}"]
    reasons: list[str] = []
    complex_case, complex_reasons = is_complex_feature_request(request, features_doc)
    reasons.extend(complex_reasons)
    if first_coverage_failed:
        reasons.append("required feature coverage failed before feature-envelope refit")
    if not coverage.get("all_required_inside", False):
        reasons.append("required feature coverage failed")
    if bpoly.crosses_antimeridian():
        reasons.append("bpoly crosses antimeridian")
    return ("full" if complex_case or reasons else "fast"), reasons or ["auto selected fast review"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Propose a four-sided RegionBPoly for FVCOM preprocessing.")
    ap.add_argument("--request-json")
    ap.add_argument("--request-text")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--polygon-lonlat", help="JSON list of four [lon,lat] vertices")
    ap.add_argument("--offshore-azimuth-deg", type=float)
    ap.add_argument("--basemap-provider", default="auto", help="auto/topo/road/street/satellite/offline provider; none/off still uses the required offline background fallback.")
    ap.add_argument("--full-side-review", action="store_true")
    ap.add_argument("--review-depth", choices=["auto", "fast", "full"], default="auto")
    ap.add_argument("--heuristic-mode", choices=["memory", "unknown"], default="memory", help="memory uses built-in place heuristics; unknown disables them and requires explicit geometry.")
    ap.add_argument("--iteration", type=int, default=1)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    request = load_request(args)
    place_memory_enabled = args.heuristic_mode == "memory"
    effective_request = _effective_request(request, place_memory_enabled)
    req_path = write_json(run_dir / f"{args.name}_request.json", request)

    features_doc = infer_target_region_features(effective_request, use_place_memory=place_memory_enabled)
    ingredients = features_as_ingredients(features_doc) or required_ingredients(effective_request, use_place_memory=place_memory_enabled)
    if args.polygon_lonlat:
        import json

        bpoly = RegionBPoly(json.loads(args.polygon_lonlat), args.offshore_azimuth_deg or 90.0)
        deformation_notes = ["User supplied explicit four-corner polygon."]
    elif not place_memory_enabled:
        bpoly, feature_seed_note = bpoly_from_feature_boxes(ingredients, args.offshore_azimuth_deg or 90.0)
        if bpoly is None:
            raise ValueError("unknown_region_no_feature_plan: test/unknown heuristic mode requires explicit target_region_features, required_ingredients, or --polygon-lonlat")
        deformation_notes = [feature_seed_note] if feature_seed_note else []
    else:
        bpoly, deformation_notes = deform_bpoly(RegionBPoly.from_region_box(guess_region_box(effective_request)), effective_request)

    basemap_provider, map_detail_policy = resolve_basemap_provider(effective_request, args.basemap_provider, features_doc)
    basemap_zoom = map_detail_policy.get("target_zoom") if map_detail_policy else None
    side_radius_km = side_focus_radius_km(effective_request, features_doc)
    coverage = score_region_box(bpoly, ingredients)
    initial_guess_map = run_dir / f"{args.name}_initial_guess_map.png"
    initial_guess_basemap = plot_region_map(
        initial_guess_map,
        bpoly,
        ingredients,
        title=f"{args.name} initial RegionBPoly guess",
        basemap_provider=basemap_provider,
        basemap_zoom=basemap_zoom,
    )
    initial_guess_basemap["map_detail_policy"] = map_detail_policy
    initial_guess_payload = {
        "schema_version": "initial_region_bpoly_guess_v1",
        "name": args.name,
        "created_at_utc": utc_now(),
        "request": request,
        "region_bpoly": bpoly.to_dict(),
        "ingredient_coverage": coverage,
        "deformation_notes": deformation_notes,
        "map_path": str(initial_guess_map),
        "basemap": initial_guess_basemap,
        "map_detail_policy": map_detail_policy,
    }
    initial_guess_json = write_json(run_dir / f"{args.name}_initial_guess_region_bpoly.json", initial_guess_payload)
    first_coverage_failed = not coverage.get("all_required_inside", False)
    if first_coverage_failed and features_doc.get("features"):
        bpoly, refit_note = fit_bpoly_to_feature_boxes(bpoly, ingredients)
        if refit_note:
            deformation_notes.append(refit_note)
            coverage = score_region_box(bpoly, ingredients)
    notes = mission_scope_notes(effective_request)
    review_depth_arg = "full" if args.full_side_review else args.review_depth
    review_depth, review_reasons = select_review_depth(effective_request, features_doc, coverage, bpoly, review_depth_arg, first_coverage_failed)
    warnings = list(bpoly.map_visibility_warnings())
    for note in notes:
        if note.get("status") in {"requires_review", "requires_island_chain"}:
            warnings.append(note["message"])

    map_path = run_dir / f"{args.name}_candidate_map.png"
    basemap = plot_region_map(map_path, bpoly, ingredients, title=f"{args.name} RegionBPoly", basemap_provider=basemap_provider, basemap_zoom=basemap_zoom)
    basemap["map_detail_policy"] = map_detail_policy
    focus_path = run_dir / f"{args.name}_candidate_focus_map.png"
    plot_region_map(focus_path, bpoly, ingredients, title=f"{args.name} focus", bbox=bpoly.envelope_bbox(), basemap_provider=basemap_provider, basemap_zoom=basemap_zoom)

    feature_path = write_json(run_dir / "target_region_features.json", features_doc)
    feature_geojson_path = write_json(run_dir / "target_region_feature_polygons.geojson", features_to_geojson(features_doc))

    if review_depth == "full":
        side_indices, fractions, mode = [0, 1, 2, 3], [0.15, 0.5, 0.85], "full_all_sides"
    else:
        side_indices, fractions, mode = [0, 1, 2, 3], [0.5], "fast_all_sides"
    side_reviews = side_focus_records(bpoly, run_dir, args.name, side_indices, fractions, basemap_provider=basemap_provider, radius_km=side_radius_km, basemap_zoom=basemap_zoom)
    open_ref = None
    if not any(f.get("category") == "lake_connecting_channels" for f in features_doc.get("features", [])):
        snap = bpoly.snap_point_to_edge(*bpoly.offshore_edge_midpoint_lonlat())
        open_ref = {
            "role": "arc_reference_point",
            "source": "propose_region_bpoly",
            "snapped": snap.get("snapped"),
            "snap_distance_m": snap.get("snap_distance_m"),
            "side_index": snap.get("side_index"),
            "side_name": snap.get("side_name"),
            "notes": "Identifies intended offshore side for downstream coastline-anchor snapping only; boundary arc generation is outside this skill.",
        }
    key = canonical_region_key(effective_request)
    if key.startswith("lake_"):
        domain_type = "lake"
    elif key in {"aleutian", "hawaii_state", "hawaii_island"}:
        domain_type = "island"
    else:
        domain_type = "coastal"
    boundary_policy = {"coastal": "coastal_arc_with_land_anchors", "island": "offshore_loop_no_land_anchors", "lake": "no_open_boundary"}[domain_type]
    quality_score = score_bpoly_quality(bpoly, ingredients, effective_request, domain_type, boundary_policy, open_ref, basemap)

    score = {
        "ingredient_coverage": coverage,
        "bpoly_quality": quality_score,
        "warnings": warnings,
        "mission_scope_notes": notes,
        "target_region_features_path": str(feature_path),
        "target_region_feature_polygons_path": str(feature_geojson_path),
        "domain_variant": features_doc.get("domain_variant"),
        "heuristic_mode": args.heuristic_mode,
        "place_memory_enabled": place_memory_enabled,
        "review_depth": review_depth,
        "review_depth_reasons": review_reasons,
        "deformation_notes": deformation_notes,
        "map_detail_policy": map_detail_policy,
        "initial_guess_artifacts": {
            "json_path": str(initial_guess_json),
            "map_path": str(initial_guess_map),
            "basemap": initial_guess_basemap,
            "map_detail_policy": map_detail_policy,
        },
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
        "target_region_features": features_doc,
        "target_region_features_path": str(feature_path),
        "target_region_feature_polygons_path": str(feature_geojson_path),
        "review_depth": review_depth,
        "review_depth_reasons": review_reasons,
        "deformation_notes": deformation_notes,
        "initial_guess_artifacts": {
            "json_path": str(initial_guess_json),
            "map_path": str(initial_guess_map),
            "basemap": initial_guess_basemap,
            "map_detail_policy": map_detail_policy,
        },
        "bpoly_quality": quality_score,
        "domain_type": domain_type,
        "boundary_policy": boundary_policy,
        "open_boundary_reference": open_ref,
        "map_visibility": score["map_visibility"],
        "map_path": str(map_path),
        "focus_map_path": str(focus_path),
        "basemap": basemap,
        "map_detail_policy": map_detail_policy,
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
                f"- Review depth: `{review_depth}`",
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
