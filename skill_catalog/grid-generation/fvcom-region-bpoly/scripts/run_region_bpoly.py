from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from region_bbox.features import (
    bpoly_from_feature_boxes,
    features_as_ingredients,
    features_to_geojson,
    fit_bpoly_to_feature_boxes,
    infer_target_region_features,
    is_complex_feature_request,
)
from region_bbox.discovery import PlaceDiscoveryError, discover_named_region_features
from region_bbox.delivery import write_standard_delivery
from propose_region_bpoly import deform_bpoly, guess_region_box, known_repair_candidates, load_request
from region_bbox.geometry import RegionBPoly
from region_bbox.ingredients import mission_scope_notes, required_ingredients, request_text
from region_bbox.io import canonical_sha256, file_sha256, read_json, utc_now, write_json
from region_bbox.map_policy import resolve_basemap_provider, side_focus_radius_km
from region_bbox.normalization import canonical_region_key, normalize_request_text
from region_bbox.plot import plot_region_map, side_focus_records
from region_bbox.scoring import score_bpoly_quality, score_region_box


POLICY = {
    "coastal": "coastal_arc_with_land_anchors",
    "island": "offshore_loop_no_land_anchors",
    "lake": "no_open_boundary",
}


def _resolve_heuristic_mode(cli_mode: str, run_mode: str) -> tuple[str, bool]:
    if cli_mode == "auto":
        resolved = "memory" if run_mode == "execute" else "unknown"
    else:
        resolved = cli_mode
    return resolved, resolved == "memory"


def _effective_request(request: dict | str, place_memory_enabled: bool) -> dict | str:
    if isinstance(request, dict):
        out = dict(request)
        out["_place_memory_enabled"] = bool(place_memory_enabled)
        return out
    return {"request": request, "_place_memory_enabled": bool(place_memory_enabled)}


def infer_domain_type(request: dict | str) -> str:
    text = normalize_request_text(request)
    key = canonical_region_key(request)
    if key.startswith("lake_"):
        return "lake"
    if key in {"puget_salish", "long_island_sound", "delaware", "murderkill", "cook_inlet", "southeast_alaska", "columbia", "hudson", "san_francisco"}:
        return "coastal"
    if key in {"aleutian", "hawaii_state", "hawaii_island"}:
        return "island"
    if re.search(r"\blake\b", text):
        return "lake"
    if any(term in text for term in ["island", "archipelago", "atoll"]):
        return "island"
    return "coastal"


def _default_place_cache_dir(case_dir: Path) -> Path:
    for parent in [case_dir, *case_dir.parents]:
        if parent.name.casefold() == "fvcom-region-bpoly":
            return parent / "cache" / "place-discovery"
    return case_dir.parent / ".place-discovery-cache"


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


def write_initial_guess_artifacts(
    visual_dir: Path,
    name: str,
    request: dict,
    bpoly: RegionBPoly,
    ingredients: list[dict],
    coverage: dict,
    deformation_notes: list[str],
    basemap_provider: str,
    map_detail_policy: dict | None = None,
    basemap_zoom: int | None = None,
) -> dict:
    visual_dir.mkdir(parents=True, exist_ok=True)
    map_path = visual_dir / f"{name}_initial_guess_map.png"
    basemap = plot_region_map(
        map_path,
        bpoly,
        ingredients,
        title=f"{name} initial RegionBPoly guess",
        basemap_provider=basemap_provider,
        basemap_zoom=basemap_zoom,
    )
    payload = {
        "schema_version": "initial_region_bpoly_guess_v1",
        "name": name,
        "created_at_utc": utc_now(),
        "request": request,
        "region_bpoly": bpoly.to_dict(),
        "ingredient_coverage": coverage,
        "deformation_notes": deformation_notes,
        "map_path": str(map_path),
        "basemap": basemap,
        "map_detail_policy": map_detail_policy,
    }
    json_path = write_json(visual_dir / f"{name}_initial_guess_region_bpoly.json", payload)
    return {"json_path": str(json_path), "map_path": str(map_path), "basemap": basemap, "map_detail_policy": map_detail_policy}


def write_intermediate(
    intermediate: Path,
    name: str,
    request: dict,
    bpoly: RegionBPoly,
    features_doc: dict,
    ingredients: list[dict],
    coverage: dict,
    mission_notes: list[dict],
    deformation_notes: list[str],
    basemap_provider: str,
    map_detail_policy: dict | None,
    review_depth: str,
    review_depth_reasons: list[str],
    domain_type: str,
    initial_guess_artifacts: dict | None = None,
    side_focus_radius: float = 45.0,
    basemap_zoom: int | None = None,
) -> dict:
    intermediate.mkdir(parents=True, exist_ok=True)
    visual_dir = intermediate / "visual_review"
    visual_dir.mkdir(parents=True, exist_ok=True)
    request_path = write_json(visual_dir / f"{name}_request.json", request)
    feature_path = write_json(visual_dir / "target_region_features.json", features_doc)
    feature_geojson_path = write_json(visual_dir / "target_region_feature_polygons.geojson", features_to_geojson(features_doc))
    map_path = visual_dir / f"{name}_candidate_map.png"
    basemap = plot_region_map(map_path, bpoly, ingredients, title=f"{name} RegionBPoly candidate", basemap_provider=basemap_provider, basemap_zoom=basemap_zoom)
    focus_path = visual_dir / f"{name}_candidate_focus_map.png"
    focus_basemap = plot_region_map(focus_path, bpoly, ingredients, title=f"{name} candidate focus", bbox=bpoly.envelope_bbox(), basemap_provider=basemap_provider, basemap_zoom=basemap_zoom)

    offshore_side_index = bpoly.offshore_side_index()
    if domain_type == "coastal":
        side_indices = [idx for idx in range(4) if idx != offshore_side_index]
        fractions, mode = [0.15, 0.5, 0.85], "coastal_land_sides_start_middle_end"
    elif review_depth == "full":
        side_indices, fractions, mode = [0, 1, 2, 3], [0.15, 0.5, 0.85], "full_all_sides"
    else:
        side_indices, fractions, mode = [0, 1, 2, 3], [0.5], "fast_all_sides"
    side_reviews = side_focus_records(bpoly, visual_dir, name, side_indices, fractions, radius_km=side_focus_radius, basemap_provider=basemap_provider, basemap_zoom=basemap_zoom)

    candidate = {
        "name": name,
        "created_at_utc": utc_now(),
        "review_status": "intermediate",
        "request_path": str(request_path),
        "region_bpoly": bpoly.to_dict(),
        "target_region_features": features_doc,
        "target_region_features_path": str(feature_path),
        "target_region_feature_polygons_path": str(feature_geojson_path),
        "initial_guess_artifacts": initial_guess_artifacts,
        "ingredient_coverage": coverage,
        "mission_scope_notes": mission_notes,
        "deformation_notes": deformation_notes,
        "map_path": str(map_path),
        "focus_map_path": str(focus_path),
        "basemap": basemap,
        "focus_basemap": focus_basemap,
        "map_detail_policy": map_detail_policy,
        "review_depth": review_depth,
        "review_depth_reasons": review_depth_reasons,
        "side_focus_mode": mode,
        "side_focus_count": len(side_reviews),
        "side_focus_required_side_indices": side_indices,
        "offshore_side_index": offshore_side_index,
        "side_focus_reviews": side_reviews,
    }
    write_json(visual_dir / f"{name}_region_bpoly_candidate.json", candidate)
    write_json(
        visual_dir / f"{name}_candidate_score.json",
        {
            "ingredient_coverage": coverage,
            "mission_scope_notes": mission_notes,
            "target_region_features_path": str(feature_path),
            "target_region_feature_polygons_path": str(feature_geojson_path),
            "review_depth": review_depth,
            "review_depth_reasons": review_depth_reasons,
            "deformation_notes": deformation_notes,
            "map_detail_policy": map_detail_policy,
        },
    )
    write_json(visual_dir / f"{name}_ingredient_coverage.json", coverage)
    return candidate


def _land_side_visual_review_request(
    *,
    candidate: dict,
    candidate_json: Path,
    final_map: Path,
    final_map_basemap: dict,
    bpoly: RegionBPoly,
    iteration: int,
) -> dict:
    required_sides = list(candidate.get("side_focus_required_side_indices", []))
    side_views: dict[str, list[dict]] = {}
    for idx in required_sides:
        records = []
        for record in candidate.get("side_focus_reviews", []):
            if int(record.get("side_index", -1)) != int(idx):
                continue
            map_path = Path(record["map_path"])
            records.append(
                {
                    "position": record.get("position"),
                    "fraction": record.get("fraction"),
                    "map_path": str(map_path),
                    "map_sha256": file_sha256(map_path),
                    "geography_usable": bool(record.get("basemap", {}).get("geography_usable", False)),
                }
            )
        side_views[str(idx)] = records
    return {
        "schema_version": "region_bpoly_land_side_visual_review_request_v1",
        "iteration": iteration,
        "maximum_iterations": 3,
        "candidate_json": str(candidate_json),
        "candidate_json_sha256": file_sha256(candidate_json),
        "region_bpoly_sha256": canonical_sha256(bpoly.to_dict()),
        "offshore_side_index": bpoly.offshore_side_index(),
        "required_land_side_indices": required_sides,
        "required_positions": ["start", "middle", "end"],
        "whole_domain_map": {
            "map_path": str(final_map),
            "map_sha256": file_sha256(final_map),
            "geography_usable": bool(final_map_basemap.get("geography_usable", False)),
        },
        "side_views": side_views,
        "allowed_statuses": ["pass", "expand_required", "unresolved"],
        "gate_policy": (
            "Inspect the whole-domain map and start/middle/end maps for every required land side. "
            "The selected offshore side is excluded and cannot request expansion."
        ),
    }


def write_basemap_comparison(
    visual_dir: Path,
    name: str,
    bpoly: RegionBPoly,
    ingredients: list[dict],
    features_doc: dict,
    basemap_zoom: int | None,
) -> list[dict]:
    if features_doc.get("domain_scale") != "small_estuary":
        return []
    comparison_dir = visual_dir / "basemap_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    providers = [
        ("esri_street", "Esri Street"),
        ("cartodb_voyager", "CARTO Voyager"),
        ("osm", "OSM Mapnik"),
        ("topo", "Esri Topo"),
        ("offline", "Offline fallback"),
    ]
    records = []
    for provider, label in providers:
        map_path = comparison_dir / f"{name}_{provider}.png"
        basemap = plot_region_map(
            map_path,
            bpoly,
            ingredients,
            title=f"{name} basemap comparison - {label}",
            basemap_provider=provider,
            basemap_zoom=basemap_zoom,
        )
        records.append({"provider": provider, "label": label, "map_path": str(map_path), "basemap": basemap})
    write_json(comparison_dir / "basemap_comparison_manifest.json", {"schema_version": "basemap_comparison_v1", "records": records})
    return records


def _provisional_open_ref(name: str, bpoly: RegionBPoly, domain_type: str) -> dict | None:
    if domain_type not in {"coastal", "island"}:
        return None
    guess_lon, guess_lat = bpoly.offshore_edge_midpoint_lonlat()
    snap = bpoly.snap_point_to_edge(guess_lon, guess_lat)
    return {
        "role": "arc_reference_point",
        "source": f"{name}_candidate_repair_precheck",
        "guess": {"lon": float(guess_lon), "lat": float(guess_lat)},
        "snapped": snap.get("snapped"),
        "snap_distance_m": snap.get("snap_distance_m"),
        "side_index": snap.get("side_index"),
        "side_name": snap.get("side_name"),
        "notes": "Precheck only; identifies intended offshore side for downstream coastline-anchor snapping.",
    }


def _candidate_basemap_meta(bpoly: RegionBPoly, map_detail_policy: dict | None = None) -> dict:
    bbox = bpoly.envelope_bbox()
    lon_span = abs(float(bbox[2]) - float(bbox[0]))
    if bpoly.crosses_antimeridian():
        lons = [p[0] for p in bpoly.polygon_lonlat()[:-1]]
        lon_span = min(lon_span, abs(max(lons) - min(lons)))
    return {
        "enabled": True,
        "status": "candidate_repair_precheck",
        "source": "candidate_repair_precheck",
        "geography_usable": True,
        "display_frame": {"lon_span_deg": lon_span},
        "map_detail_policy": map_detail_policy,
    }


def _blocking_failure_count(quality: dict) -> int:
    return sum(1 for item in quality.get("failure_taxonomy", []) if item.get("severity") == "fail")


def apply_candidate_repairs(
    name: str,
    request: dict | str,
    bpoly: RegionBPoly,
    ingredients: list[dict],
    coverage: dict,
    domain_type: str,
    map_detail_policy: dict | None,
) -> tuple[RegionBPoly, dict, list[str]]:
    candidates = known_repair_candidates(request)
    if not candidates:
        return bpoly, coverage, []

    current_quality = score_bpoly_quality(
        bpoly,
        ingredients,
        request,
        domain_type,
        POLICY[domain_type],
        _provisional_open_ref(name, bpoly, domain_type),
        _candidate_basemap_meta(bpoly, map_detail_policy),
    )
    current_fail_count = _blocking_failure_count(current_quality)
    current_area = current_quality.get("tight_feature_fit", {}).get("region_area_km2") or float("inf")
    best = {
        "bpoly": bpoly,
        "coverage": coverage,
        "quality": current_quality,
        "fail_count": current_fail_count,
        "area": current_area,
        "candidate_id": "current",
        "notes": None,
    }

    for candidate in candidates:
        candidate_bpoly = candidate["region_bpoly"]
        candidate_coverage = score_region_box(candidate_bpoly, ingredients)
        if not candidate_coverage.get("all_required_inside", False):
            continue
        candidate_quality = score_bpoly_quality(
            candidate_bpoly,
            ingredients,
            request,
            domain_type,
            POLICY[domain_type],
            _provisional_open_ref(name, candidate_bpoly, domain_type),
            _candidate_basemap_meta(candidate_bpoly, map_detail_policy),
        )
        fail_count = _blocking_failure_count(candidate_quality)
        area = candidate_quality.get("tight_feature_fit", {}).get("region_area_km2") or float("inf")
        improves_failures = fail_count < best["fail_count"]
        improves_tightness = fail_count == best["fail_count"] and area < best["area"] * 0.95
        if improves_failures or improves_tightness:
            best = {
                "bpoly": candidate_bpoly,
                "coverage": candidate_coverage,
                "quality": candidate_quality,
                "fail_count": fail_count,
                "area": area,
                "candidate_id": candidate.get("id"),
                "notes": candidate.get("notes"),
            }

    if best["candidate_id"] == "current":
        return bpoly, coverage, []
    note = f"Candidate repair selected {best['candidate_id']}: {best['notes']}"
    return best["bpoly"], best["coverage"], [note]


def build_offshore_boundary_artifacts(
    path: Path,
    name: str,
    bpoly: RegionBPoly,
    domain_type: str,
    boundary_policy: str,
    open_ref: dict | None,
    candidate: dict,
    retained_intermediate: bool,
    quality_score: dict | None = None,
) -> dict:
    side_index = bpoly.offshore_side_index()
    side = bpoly.sides()[side_index]
    side_reviews = candidate.get("side_focus_reviews", [])
    zoom_records = []
    for record in side_reviews:
        zoom_records.append(
            {
                "side_index": record.get("side_index"),
                "side_name": record.get("side_name"),
                "position": record.get("position"),
                "fraction": record.get("fraction"),
                "center_lonlat": record.get("center_lonlat"),
                "bbox": record.get("bbox"),
                "map_path": record.get("map_path") if retained_intermediate else None,
                "retained": retained_intermediate,
            }
        )
    artifacts = {
        "schema_version": "offshore_boundary_artifacts_v1",
        "name": name,
        "domain_type": domain_type,
        "boundary_policy": boundary_policy,
        "selected_side_index": side_index,
        "selected_side_name": side.get("side_name"),
        "selected_side_start_lonlat": side.get("start_lonlat"),
        "selected_side_end_lonlat": side.get("end_lonlat"),
        "selected_side_midpoint_lonlat": side.get("midpoint_lonlat"),
        "offshore_azimuth_deg": bpoly.offshore_azimuth_deg,
        "open_boundary_reference": open_ref,
        "offshore_point_purpose": "Identifies the intended offshore side for downstream coastline-anchor snapping. Boundary arc generation is outside fvcom-region-bpoly.",
        "review_depth": candidate.get("review_depth"),
        "review_depth_reasons": candidate.get("review_depth_reasons", []),
        "side_focus_mode": candidate.get("side_focus_mode"),
        "side_focus_count": candidate.get("side_focus_count"),
        "zoom_maps_used": zoom_records,
        "warnings": bpoly.map_visibility_warnings()
        + (quality_score or {}).get("offshore_side_qa", {}).get("warnings", [])
        + (quality_score or {}).get("map_usability_qa", {}).get("warnings", []),
        "offshore_side_qa": (quality_score or {}).get("offshore_side_qa", {}),
        "failure_taxonomy": (quality_score or {}).get("failure_taxonomy", []),
    }
    write_json(path, artifacts)
    return artifacts


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the streamlined RegionBPoly workflow.")
    ap.add_argument("--request-json")
    ap.add_argument("--request-text")
    ap.add_argument("--input-region-json", help="Adjusted RegionBPoly input for a land-side repair iteration.")
    ap.add_argument("--land-side-review-iteration", type=int, choices=[1, 2, 3], default=1)
    ap.add_argument("--run-dir", required=True, help="Final case output folder.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["execute", "test"], default="execute")
    ap.add_argument("--heuristic-mode", choices=["auto", "memory", "unknown"], default="auto", help="auto uses place memory in execute and disables it in test; unknown bypasses catalog geometry but still permits named-place discovery.")
    ap.add_argument("--place-discovery", choices=["auto", "off"], default="auto", help="When catalog or explicit geometry is absent, auto performs one cached named-place lookup; off requires --discovery-bbox or explicit request geometry.")
    ap.add_argument("--discovery-bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="Agent/user-supplied initial geographic bbox used only when no feature plan exists.")
    ap.add_argument("--discovery-label", help="Label and provenance description for --discovery-bbox.")
    ap.add_argument("--discovery-query", help="Override the named-place query extracted from the modeling request.")
    ap.add_argument("--place-discovery-cache-dir", help="Persistent cache for online named-place responses.")
    ap.add_argument("--place-discovery-timeout-s", type=float, default=20.0)
    ap.add_argument("--review-depth", choices=["auto", "fast", "full"], default="auto")
    ap.add_argument("--domain-type", choices=list(POLICY))
    ap.add_argument("--open-boundary-reference", nargs=2, type=float, metavar=("LON", "LAT"))
    ap.add_argument("--offshore-azimuth-deg", type=float)
    ap.add_argument("--basemap-provider", default="auto", help="auto/topo/road/street/satellite/offline provider; none/off still uses the required offline background fallback.")
    ap.add_argument("--full-side-review", action="store_true")
    args = ap.parse_args()

    case_dir = Path(args.run_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    intermediate = case_dir / "intermediate"
    if intermediate.exists():
        shutil.rmtree(intermediate)

    request = load_request(args)
    heuristic_mode, place_memory_enabled = _resolve_heuristic_mode(args.heuristic_mode, args.mode)
    effective_request = _effective_request(request, place_memory_enabled)
    domain_type = args.domain_type or infer_domain_type(effective_request)
    features_doc = infer_target_region_features(effective_request, use_place_memory=place_memory_enabled)
    ingredients = features_as_ingredients(features_doc) or required_ingredients(effective_request, use_place_memory=place_memory_enabled)
    place_discovery: dict | None = None
    place_discovery_path: Path | None = None
    if not ingredients:
        if args.place_discovery == "off" and args.discovery_bbox is None:
            raise SystemExit(
                "region_discovery_failed: no catalog or explicit feature geometry exists and online discovery is off; "
                "supply --discovery-bbox WEST SOUTH EAST NORTH or target_region_features"
            )
        cache_dir = Path(args.place_discovery_cache_dir) if args.place_discovery_cache_dir else _default_place_cache_dir(case_dir)
        try:
            features_doc, place_discovery = discover_named_region_features(
                effective_request,
                domain_type,
                bbox_override=args.discovery_bbox,
                label_override=args.discovery_label,
                query_override=args.discovery_query,
                timeout_s=args.place_discovery_timeout_s,
                cache_dir=cache_dir,
            )
        except PlaceDiscoveryError as exc:
            raise SystemExit(
                f"region_discovery_failed: {exc}. Research or infer an initial frame and rerun with "
                "--discovery-bbox WEST SOUTH EAST NORTH; no unresolved RegionBPoly was delivered."
            ) from exc
        place_discovery_path = write_json(case_dir / "region_place_discovery.json", place_discovery)
        ingredients = features_as_ingredients(features_doc)
    bpoly: RegionBPoly | None = None
    deformation_notes: list[str] = []
    adjusted_source: dict | None = None
    if args.input_region_json:
        adjusted_source = read_json(args.input_region_json)
        bpoly = RegionBPoly.from_dict(adjusted_source)
        deformation_notes.append(f"Started land-side review iteration {args.land_side_review_iteration} from {args.input_region_json}.")
        if args.land_side_review_iteration == 1:
            raise SystemExit("--input-region-json is only valid for land-side review iterations 2 or 3")
        prior_review = adjusted_source.get("source_land_side_visual_review") or adjusted_source.get("land_side_visual_review")
        prior_iteration = int((prior_review or {}).get("iteration", 0))
        if prior_iteration != args.land_side_review_iteration - 1:
            raise SystemExit("Adjusted input is not bound to the immediately preceding land-side review iteration")
        if (prior_review or {}).get("decision") != "revise":
            raise SystemExit("Adjusted input does not follow an expand_required review decision")
    elif args.land_side_review_iteration != 1:
        raise SystemExit("Review iterations 2 and 3 require --input-region-json")
    elif place_memory_enabled:
        try:
            bpoly, deformation_notes = deform_bpoly(RegionBPoly.from_region_box(guess_region_box(effective_request)), effective_request)
        except ValueError:
            bpoly = None
    if bpoly is None:
        seed_azimuth = float(args.offshore_azimuth_deg) if args.offshore_azimuth_deg is not None else (0.0 if domain_type == "lake" else 90.0)
        edge_labels = (
            ["south_lake_edge", "west_lake_edge", "north_lake_edge", "east_lake_edge"]
            if domain_type == "lake"
            else ["open_or_south", "west_or_left", "north_or_inner", "east_or_right"]
        )
        bpoly, feature_seed_note = bpoly_from_feature_boxes(ingredients, seed_azimuth, edge_labels=edge_labels)
        if feature_seed_note:
            deformation_notes.append(feature_seed_note)
    if bpoly is None:
        raise SystemExit(
            "region_discovery_failed: no valid four-corner seed could be built from catalog, explicit, "
            "or discovered geometry; no unresolved RegionBPoly was delivered"
        )
    basemap_provider, map_detail_policy = resolve_basemap_provider(effective_request, args.basemap_provider, features_doc)
    basemap_zoom = map_detail_policy.get("target_zoom") if map_detail_policy else None
    side_radius_km = side_focus_radius_km(effective_request, features_doc)
    coverage = score_region_box(bpoly, ingredients)
    initial_guess_artifacts = write_initial_guess_artifacts(
        intermediate / "visual_review",
        args.name,
        request,
        bpoly,
        ingredients,
        coverage,
        deformation_notes,
        basemap_provider,
        map_detail_policy,
        basemap_zoom,
    )
    first_coverage_failed = not coverage.get("all_required_inside", False)
    if first_coverage_failed and features_doc.get("features"):
        bpoly, refit_note = fit_bpoly_to_feature_boxes(bpoly, ingredients)
        if refit_note:
            deformation_notes.append(refit_note)
            coverage = score_region_box(bpoly, ingredients)
    mission_notes = mission_scope_notes(effective_request)
    repair_notes: list[str] = []
    if adjusted_source is None:
        bpoly, coverage, repair_notes = apply_candidate_repairs(
            args.name,
            effective_request,
            bpoly,
            ingredients,
            coverage,
            domain_type,
            map_detail_policy,
        )
    if repair_notes:
        deformation_notes.extend(repair_notes)
    if args.offshore_azimuth_deg is not None:
        bpoly = RegionBPoly(
            bpoly.polygon_lonlat()[:-1],
            float(args.offshore_azimuth_deg),
            edge_labels=bpoly.edge_labels,
        )
        deformation_notes.append(f"Set offshore azimuth to {float(args.offshore_azimuth_deg):g} degrees from CLI override.")
    requested_review_depth = "full" if args.full_side_review else args.review_depth
    review_depth, review_depth_reasons = select_review_depth(effective_request, features_doc, coverage, bpoly, requested_review_depth, first_coverage_failed)
    if domain_type == "coastal" and review_depth != "full":
        review_depth = "full"
        review_depth_reasons.append("coastal land-side truncation gate requires start/middle/end views")

    ref_guess = args.open_boundary_reference
    if not ref_guess and domain_type in {"coastal", "island"}:
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
            "notes": "Identifies intended offshore side for downstream coastline-anchor snapping only; boundary arc generation is outside this skill.",
        }

    candidate = write_intermediate(
        intermediate,
        args.name,
        effective_request,
        bpoly,
        features_doc,
        ingredients,
        coverage,
        mission_notes,
        deformation_notes,
        basemap_provider,
        map_detail_policy,
        review_depth,
        review_depth_reasons,
        domain_type,
        initial_guess_artifacts,
        side_radius_km,
        basemap_zoom,
    )
    basemap_comparison = []
    if args.mode == "test":
        basemap_comparison = write_basemap_comparison(
            intermediate / "visual_review",
            args.name,
            bpoly,
            ingredients,
            features_doc,
            basemap_zoom,
        )
        if basemap_comparison:
            candidate["basemap_comparison"] = basemap_comparison

    final_map = case_dir / "region_bpoly_final_map.png"
    basemap = plot_region_map(
        final_map,
        bpoly,
        ingredients,
        title=f"{args.name} final RegionBPoly",
        basemap_provider=basemap_provider,
        basemap_zoom=basemap_zoom,
        open_boundary_reference=open_ref,
    )
    basemap["map_detail_policy"] = map_detail_policy
    quality_score = score_bpoly_quality(
        bpoly,
        ingredients,
        request,
        domain_type,
        POLICY[domain_type],
        open_ref,
        basemap,
    )

    delivery_warnings: list[str] = []
    if not coverage.get("all_required_inside", False):
        delivery_warnings.append("required ingredients missing: " + ", ".join(coverage.get("missing_required_ids", [])))
    if domain_type == "coastal" and not open_ref:
        delivery_warnings.append("coastal domain missing open-boundary reference")
    for item in quality_score.get("failure_taxonomy", []):
        if item.get("severity") == "fail":
            delivery_warnings.append(f"{item.get('code')}: {item.get('message')}")

    review_basemaps = [
        ("initial", (initial_guess_artifacts or {}).get("basemap", {})),
        ("candidate", candidate.get("basemap", {})),
        ("candidate_focus", candidate.get("focus_basemap", {})),
        *[
            (f"side_{record.get('side_index')}_{record.get('position')}", record.get("basemap", {}))
            for record in candidate.get("side_focus_reviews", [])
        ],
        ("final", basemap),
    ]
    unusable_maps = [label for label, metadata in review_basemaps if not metadata.get("geography_usable", False)]
    if unusable_maps:
        delivery_warnings.append("background geography unavailable in review maps: " + ", ".join(unusable_maps))

    final_status = "needs_review" if domain_type == "coastal" else "pass"
    retained_intermediate = args.mode == "test"
    offshore_artifacts_path = case_dir / "offshore_boundary_artifacts.json"
    offshore_artifacts = build_offshore_boundary_artifacts(
        offshore_artifacts_path,
        args.name,
        bpoly,
        domain_type,
        POLICY[domain_type],
        open_ref,
        candidate,
        retained_intermediate,
        quality_score,
    )
    candidate_json = intermediate / "visual_review" / f"{args.name}_region_bpoly_candidate.json"
    land_side_review_request = None
    if domain_type == "coastal":
        land_side_review_request = _land_side_visual_review_request(
            candidate=candidate,
            candidate_json=candidate_json,
            final_map=final_map,
            final_map_basemap=basemap,
            bpoly=bpoly,
            iteration=args.land_side_review_iteration,
        )

    final = {
        "schema_version": "region_bpoly_final_v1",
        "object_type": "RegionBPolyFinal",
        "name": args.name,
        "created_at_utc": utc_now(),
        "mode": args.mode,
        "heuristic_mode": heuristic_mode,
        "place_memory_enabled": place_memory_enabled,
        "place_discovery": place_discovery,
        "place_discovery_path": str(place_discovery_path) if place_discovery_path else None,
        "final_status": final_status,
        "status_reasons": (["land_side_visual_review_required"] if domain_type == "coastal" else []),
        "delivery_warnings": delivery_warnings,
        "region_bpoly": bpoly.to_dict(),
        "polygon_lonlat": bpoly.polygon_lonlat(),
        "envelope_bbox": bpoly.envelope_bbox(),
        "target_region_features": features_doc,
        "domain_variant": features_doc.get("domain_variant"),
        "domain_type": domain_type,
        "boundary_policy": POLICY[domain_type],
        "open_boundary_reference": open_ref,
        "offshore_boundary_artifacts_path": str(offshore_artifacts_path),
        "land_side_visual_review_request": land_side_review_request,
        "land_side_visual_review": None,
        "final_map_path": str(final_map),
        "final_map_basemap": basemap,
        "map_detail_policy": map_detail_policy,
        "intermediate_dir": str(intermediate) if retained_intermediate else None,
        "qa": {
            "ingredient_coverage": {
                "all_required_inside": coverage.get("all_required_inside", False),
                "missing_required_ids": coverage.get("missing_required_ids", []),
                "required_count": coverage.get("required_count", 0),
                "ingredient_count": coverage.get("ingredient_count", 0),
            },
            "target_region_features": {
                "feature_count": len(features_doc.get("features", [])),
                "categories": sorted({f.get("category") for f in features_doc.get("features", []) if f.get("category")}),
                "retained_path": str(case_dir / "target_region_features.json"),
                "retained_geojson_path": candidate.get("target_region_feature_polygons_path") if retained_intermediate else None,
            },
            "mission_scope_notes": mission_notes,
            "map_visibility_warnings": bpoly.map_visibility_warnings(),
            "bpoly_quality": quality_score,
            "map_detail_policy": map_detail_policy,
            "basemap_comparison": basemap_comparison if retained_intermediate else [],
            "review_depth": review_depth,
            "review_depth_reasons": review_depth_reasons,
            "side_focus_mode": candidate.get("side_focus_mode"),
            "side_focus_count": candidate.get("side_focus_count"),
            "initial_guess_artifacts": initial_guess_artifacts if retained_intermediate else {"retained": False},
            "delivery_policy": (
                "coastal_land_side_visual_review_is_blocking"
                if domain_type == "coastal"
                else "resolved_region_bpoly_qa_is_nonblocking"
            ),
            "delivery_warnings": delivery_warnings,
        },
        "offshore_boundary_artifacts": {
            "selected_side_index": offshore_artifacts.get("selected_side_index"),
            "selected_side_name": offshore_artifacts.get("selected_side_name"),
            "review_depth": offshore_artifacts.get("review_depth"),
            "side_focus_count": offshore_artifacts.get("side_focus_count"),
        },
        "deformation_notes": deformation_notes,
        "downstream_contract": {
            "bathymetry_and_coastline_fetch": (
                "Use envelope_bbox only for RegionBPoly-stage coastline-source planning. "
                "Fetch downstream bathymetry from the final model_domain_polygon plus the configured halo."
            ),
            "domain_and_grid_generation": (
                "Use polygon_lonlat / region_bpoly as the mission and land-boundary completeness frame. "
                "The delivered offshore OBC may deform outside it; the assembled model_domain_polygon "
                "is authoritative for downstream grid generation."
            ),
            "offshore_point": "Use the offshore point only to identify the intended offshore side for later coastline-anchor snapping; do not interpret it as a generated boundary arc.",
        },
    }
    out = write_standard_delivery(case_dir, args.name, final)
    if args.mode == "execute" and intermediate.exists() and final_status == "pass":
        shutil.rmtree(intermediate)
    print(f"Wrote final RegionBPoly: {out}")
    if final_status == "needs_review":
        print("Final status needs review: inspect and finalize the hash-bound coastal land-side maps.")
    if delivery_warnings:
        print("Delivered resolved RegionBPoly with QA warnings: " + "; ".join(delivery_warnings))


if __name__ == "__main__":
    main()
