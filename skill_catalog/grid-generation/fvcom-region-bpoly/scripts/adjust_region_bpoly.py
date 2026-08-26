from __future__ import annotations

import argparse
from pathlib import Path

from region_bbox.adjustments import apply_adjustment_manifest
from region_bbox.features import features_as_ingredients
from region_bbox.geometry import RegionBPoly
from region_bbox.io import read_json, utc_now, write_json
from region_bbox.plot import plot_region_map


def _load_region(path: Path) -> tuple[RegionBPoly, dict]:
    data = read_json(path)
    return RegionBPoly.from_dict(data), data


def _load_ingredients(path: Path | None) -> list[dict]:
    if not path:
        return []
    return features_as_ingredients(read_json(path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply explicit RegionBPoly edits, including deterministic single-land-side expansion.")
    ap.add_argument("--input-json", required=True, help="Input region_bpoly.json, candidate JSON, or raw RegionBPoly JSON.")
    ap.add_argument("--adjustment-manifest", required=True, help="JSON manifest with one operation or an operations list.")
    ap.add_argument("--output-json", help="Adjusted RegionBPoly JSON path.")
    ap.add_argument("--map-path", help="Comparison map path with old polygon dashed and adjusted polygon solid.")
    ap.add_argument("--features-json", help="Optional target_region_features.json to overlay feature boxes.")
    ap.add_argument("--basemap-provider", default="topo", help="topo/street/satellite/offline provider.")
    ap.add_argument(
        "--truncation-loop",
        action="store_true",
        help="Enforce the strict repair-loop contract: one review-authorized expand_side operation only.",
    )
    args = ap.parse_args()

    input_path = Path(args.input_json)
    manifest_path = Path(args.adjustment_manifest)
    output_json = Path(args.output_json) if args.output_json else input_path.with_name("region_bpoly_adjusted.json")
    map_path = Path(args.map_path) if args.map_path else output_json.with_name("region_bpoly_adjustment_map.png")

    original, source_doc = _load_region(input_path)
    manifest = read_json(manifest_path)
    operations = manifest.get("operations", [manifest])
    if args.truncation_loop:
        if len(operations) != 1 or str(operations[0].get("operation", operations[0].get("type", ""))).lower() != "expand_side":
            raise SystemExit("The truncation loop permits exactly one expand_side operation; rotate, scale, and reshape are explicit-edit operations only")
        review = source_doc.get("land_side_visual_review") or {}
        next_action = review.get("next_action") or {}
        if review.get("decision") != "revise" or next_action.get("operation") != "expand_side":
            raise SystemExit("The truncation-loop input lacks a current expand_required review decision")
        if int(operations[0].get("side_index", -1)) != int(next_action.get("side_index", -2)):
            raise SystemExit("expand_side does not match the land side authorized by the current visual review")
    adjusted, history = apply_adjustment_manifest(original, manifest)
    ingredients = _load_ingredients(Path(args.features_json) if args.features_json else None)
    basemap = plot_region_map(
        map_path,
        adjusted,
        ingredients,
        title="RegionBPoly adjustment",
        basemap_provider=args.basemap_provider,
        comparison_region=original,
        comparison_label="old RegionBPoly",
    )

    payload = {
        "schema_version": "region_bpoly_adjustment_v1",
        "created_at_utc": utc_now(),
        "input_json": str(input_path),
        "adjustment_manifest": str(manifest_path),
        "source_region_bpoly": original.to_dict(),
        "adjusted_region_bpoly": adjusted.to_dict(),
        "region_bpoly": adjusted.to_dict(),
        "polygon_lonlat": adjusted.polygon_lonlat(),
        "adjustment_history": history,
        "adjustment_map_path": str(map_path),
        "adjustment_map_basemap": basemap,
        "source_document_name": source_doc.get("name"),
        "source_land_side_visual_review": source_doc.get("land_side_visual_review"),
        "source_land_side_visual_review_request": source_doc.get("land_side_visual_review_request"),
        "target_region_features": source_doc.get("target_region_features"),
        "domain_type": source_doc.get("domain_type"),
        "boundary_policy": source_doc.get("boundary_policy"),
        "open_boundary_reference": source_doc.get("open_boundary_reference"),
        "truncation_loop": bool(args.truncation_loop),
    }
    out = write_json(output_json, payload)
    print(f"Wrote adjusted RegionBPoly: {out}")
    print(f"Wrote adjustment map: {map_path}")


if __name__ == "__main__":
    main()
