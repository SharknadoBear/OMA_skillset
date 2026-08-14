#!/usr/bin/env python3
"""Apply one geometry-only boundary-arc feedback candidate to a RegionBPoly."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from region_bbox.adjustments import apply_adjustment_manifest
from region_bbox.features import features_as_ingredients
from region_bbox.geometry import RegionBPoly
from region_bbox.io import read_json, utc_now, write_json
from region_bbox.plot import plot_region_map
from region_bbox.scoring import score_bpoly_quality, score_region_box


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate(feedback: dict, candidate_id: str) -> dict:
    candidates = list(feedback.get("candidate_recommendations", []))
    for item in candidates:
        if item.get("candidate_id") == candidate_id:
            return item
    raise ValueError(f"Feedback candidate not found: {candidate_id}")


def _request_for_scoring(source: dict, features: dict) -> dict:
    return {
        "request": features.get("request_text") or source.get("name") or "RegionBPoly arc-feedback adjustment",
        "region": source.get("name"),
        "target_region_features": features,
        "_place_memory_enabled": bool(source.get("place_memory_enabled", False)),
    }


def _open_reference(source: dict, region: RegionBPoly) -> dict | None:
    if source.get("domain_type") == "lake" or source.get("boundary_policy") == "no_open_boundary":
        return None
    previous = source.get("open_boundary_reference") or {}
    guess = previous.get("guess") or {}
    lon = guess.get("lon")
    lat = guess.get("lat")
    if lon is None or lat is None:
        lon, lat = region.offshore_edge_midpoint_lonlat()
    snap = region.snap_point_to_edge(float(lon), float(lat))
    return {
        "role": "arc_reference_point",
        "source": "apply_arc_feedback",
        "guess": {"lon": float(lon), "lat": float(lat)},
        "snapped": snap.get("snapped"),
        "snap_distance_m": snap.get("snap_distance_m"),
        "side_index": snap.get("side_index"),
        "side_name": snap.get("side_name"),
        "notes": "Identifies the intended offshore side after geometry-only arc-feedback adjustment.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--feedback-json", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--basemap-provider", default="none")
    args = parser.parse_args()

    input_path = Path(args.input_json).resolve()
    feedback_path = Path(args.feedback_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = read_json(input_path)
    feedback = read_json(feedback_path)
    expected_hash = feedback.get("input_sha256", {}).get("region_bpoly_json")
    actual_hash = _sha256(input_path)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("Arc feedback is stale for the supplied RegionBPoly")
    candidate = _candidate(feedback, args.candidate_id)
    if candidate.get("semantic_feature_changes"):
        raise ValueError("Arc-feedback candidates may not change semantic features")

    original = RegionBPoly.from_dict(source)
    adjustment_manifest = {
        "operation": "reshape",
        "vertex_delta_km": candidate.get("vertex_delta_km"),
    }
    adjusted, history = apply_adjustment_manifest(original, adjustment_manifest)
    features = copy.deepcopy(source.get("target_region_features", {}))
    feature_hash_before = _canonical_hash(features)
    ingredients = features_as_ingredients(features)
    coverage = score_region_box(adjusted, ingredients)
    open_ref = _open_reference(source, adjusted)

    comparison_map = output_dir / "region_bpoly_arc_adjustment_map.png"
    basemap = plot_region_map(
        comparison_map,
        adjusted,
        ingredients,
        title=f"{source.get('name', 'RegionBPoly')} arc-feedback adjustment",
        basemap_provider=args.basemap_provider,
        comparison_region=original,
        comparison_label="previous RegionBPoly",
        open_boundary_reference=open_ref,
    )
    domain_type = source.get("domain_type", "coastal")
    boundary_policy = source.get("boundary_policy", "coastal_arc_with_land_anchors")
    quality = score_bpoly_quality(
        adjusted,
        ingredients,
        _request_for_scoring(source, features),
        domain_type,
        boundary_policy,
        open_ref,
        basemap,
    )
    blocks: list[str] = []
    if not coverage.get("all_required_inside", False):
        blocks.append("required ingredients missing: " + ", ".join(coverage.get("missing_required_ids", [])))
    if domain_type == "coastal" and not open_ref:
        blocks.append("coastal domain missing open-boundary reference")
    blocks.extend(
        f"{item.get('code')}: {item.get('message')}"
        for item in quality.get("failure_taxonomy", [])
        if item.get("severity") == "fail"
    )
    feature_hash_after = _canonical_hash(features)
    if feature_hash_after != feature_hash_before:
        blocks.append("target feature document changed during geometry-only adjustment")
    final_status = "pass" if not blocks else "needs_review"

    result = copy.deepcopy(source)
    result.update(
        {
            "schema_version": "region_bpoly_final_v1",
            "object_type": "RegionBPolyFinal",
            "created_at_utc": utc_now(),
            "final_status": final_status,
            "status_reasons": blocks,
            "region_bpoly": adjusted.to_dict(),
            "polygon_lonlat": adjusted.polygon_lonlat(),
            "envelope_bbox": adjusted.envelope_bbox(),
            "target_region_features": features,
            "open_boundary_reference": open_ref,
            "final_map_path": str(comparison_map),
            "final_map_basemap": basemap,
            "deformation_notes": list(source.get("deformation_notes", []))
            + [f"Applied geometry-only arc feedback candidate {args.candidate_id}."],
            "arc_feedback_lineage": {
                "schema_version": "region_bpoly_arc_adjustment_lineage_v1",
                "source_region_bpoly_json": str(input_path),
                "source_region_bpoly_sha256": actual_hash,
                "source_feedback_json": str(feedback_path),
                "source_feedback_sha256": _sha256(feedback_path),
                "candidate_id": args.candidate_id,
                "candidate": candidate,
                "adjustment_history": history,
                "target_region_features_sha256_before": feature_hash_before,
                "target_region_features_sha256_after": feature_hash_after,
                "semantic_feature_changes": [],
            },
        }
    )
    qa = copy.deepcopy(source.get("qa", {}))
    qa["ingredient_coverage"] = {
        "all_required_inside": coverage.get("all_required_inside", False),
        "missing_required_ids": coverage.get("missing_required_ids", []),
        "required_count": coverage.get("required_count", 0),
        "ingredient_count": coverage.get("ingredient_count", 0),
    }
    qa["bpoly_quality"] = quality
    qa["arc_feedback_adjustment"] = result["arc_feedback_lineage"]
    result["qa"] = qa

    offshore = {
        "schema_version": "offshore_boundary_artifacts_v1",
        "name": result.get("name"),
        "domain_type": domain_type,
        "boundary_policy": boundary_policy,
        "selected_side_index": adjusted.offshore_side_index(),
        "selected_side_name": adjusted.sides()[adjusted.offshore_side_index()].get("side_name"),
        "selected_side_start_lonlat": adjusted.sides()[adjusted.offshore_side_index()].get("start_lonlat"),
        "selected_side_end_lonlat": adjusted.sides()[adjusted.offshore_side_index()].get("end_lonlat"),
        "selected_side_midpoint_lonlat": adjusted.sides()[adjusted.offshore_side_index()].get("midpoint_lonlat"),
        "offshore_azimuth_deg": adjusted.offshore_azimuth_deg,
        "open_boundary_reference": open_ref,
        "offshore_point_purpose": "Identifies the intended offshore side after geometry-only arc-feedback adjustment.",
        "review_depth": source.get("qa", {}).get("review_depth"),
        "warnings": quality.get("offshore_side_qa", {}).get("warnings", []),
        "offshore_side_qa": quality.get("offshore_side_qa", {}),
        "failure_taxonomy": quality.get("failure_taxonomy", []),
        "arc_feedback_lineage": result["arc_feedback_lineage"],
    }
    offshore_path = write_json(output_dir / "offshore_boundary_artifacts.json", offshore)
    result["offshore_boundary_artifacts_path"] = str(offshore_path)
    result["offshore_boundary_artifacts"] = {
        "selected_side_index": offshore["selected_side_index"],
        "selected_side_name": offshore["selected_side_name"],
        "review_depth": offshore.get("review_depth"),
        "side_focus_count": source.get("offshore_boundary_artifacts", {}).get("side_focus_count"),
    }
    output_path = write_json(output_dir / "region_bpoly.json", result)
    print(json.dumps({"final_status": final_status, "region_bpoly_json": str(output_path), "offshore_artifacts_json": str(offshore_path)}, indent=2))
    return 0 if final_status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
