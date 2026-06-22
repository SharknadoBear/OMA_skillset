"""Run SE-AK CUSP+OSM fallback smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cusp_coastline.fetch import fetch_cusp_bbox  # noqa: E402
from cusp_coastline.sources import build_region_index, save_region_index  # noqa: E402
from cusp_coastline.visual_qa import visual_review_passed  # noqa: E402


SEAK_CASES = {
    "se_ak_icy_strait": (-136.20, 58.10, -134.80, 58.60),
    "se_ak_sumner_strait": (-133.50, 55.60, -131.70, 56.40),
    "se_ak_icy_west_gap": (-136.20, 58.10, -135.612, 58.60),
    "se_ak_cross_sound_outer_coast": (-137.20, 57.85, -136.15, 58.55),
}


def _check(path: str | Path, errors: list[str], label: str) -> None:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        errors.append(f"{label} missing or empty: {path}")


def _validate_case(name: str, metadata: dict[str, object], *, require_visual_review: bool = False) -> list[str]:
    errors: list[str] = []
    outputs = metadata["outputs"]
    for key in ("cusp_primary_gpkg", "fallback_candidates_gpkg", "merged_coastline_gpkg", "merge_report_json", "merged_satellite_png"):
        _check(outputs[key], errors, key)
    for key in ("visual_review_json", "visual_review_md"):
        if key not in outputs:
            errors.append(f"{key} missing from outputs")
            continue
        _check(outputs[key], errors, key)
    if require_visual_review and not visual_review_passed(outputs):
        errors.append("agent visual review has not recorded a pass")

    report = metadata.get("merge_report") or {}
    merged = gpd.read_file(outputs["merged_coastline_gpkg"], layer="coastline", engine="pyogrio")
    if merged.empty:
        errors.append("merged coastline is empty")
    if name == "se_ak_icy_west_gap":
        if report.get("cusp_feature_count") != 0:
            errors.append("Icy west gap should have zero production CUSP features")
        if report.get("fallback_retained_count", 0) <= 0:
            errors.append("Icy west gap should retain OSM fallback features")
    if name == "se_ak_icy_strait" and report.get("fallback_retained_count", 0) <= 0:
        errors.append("Icy Strait should retain fallback features in the western gap")
    if name == "se_ak_sumner_strait" and report.get("fallback_fraction_of_merged_length", 0.0) > 0.25:
        errors.append("Sumner fallback fraction exceeded 25 percent")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--index", help="Existing or output cusp_region_index.json path.")
    parser.add_argument("--fallback-policy", default="auto", choices=("osm-overpass", "auto"))
    parser.add_argument("--allow-no-basemap", action="store_true")
    parser.add_argument("--refresh-osm", action="store_true")
    parser.add_argument("--require-visual-review", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--client-timeout-s", type=float, default=0.0, help="0 means no hard client timeout.")
    parser.add_argument("--overpass-timeout-s", type=float, default=0.0, help="0 means omit Overpass server timeout.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case progress text; JSONL is still written.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index) if args.index else run_dir / "cusp_region_index.json"
    if not index_path.exists():
        save_region_index(build_region_index(), index_path)

    summary = {"cases": {}, "warnings": [], "errors": []}
    for name, bbox in SEAK_CASES.items():
        case_dir = run_dir / name
        result = fetch_cusp_bbox(
            index_path,
            bbox,
            run_dir=case_dir,
            name=name,
            fallback_policy=args.fallback_policy,
            allow_no_basemap=args.allow_no_basemap,
            refresh_osm=args.refresh_osm,
            heartbeat_seconds=args.heartbeat_seconds,
            client_timeout_s=args.client_timeout_s,
            overpass_timeout_s=args.overpass_timeout_s,
            quiet=args.quiet,
        )
        errors = _validate_case(name, result.metadata, require_visual_review=args.require_visual_review)
        report = result.metadata["merge_report"]
        summary["cases"][name] = {
            "bbox_wsen": bbox,
            "cusp_feature_count": report["cusp_feature_count"],
            "fallback_candidate_count": report["fallback_candidate_count"],
            "fallback_retained_count": report["fallback_retained_count"],
            "cusp_length_km": report["cusp_length_km"],
            "fallback_candidate_length_km": report["fallback_candidate_length_km"],
            "fallback_retained_length_km": report["fallback_retained_length_km"],
            "fallback_fraction_of_merged_length": report["fallback_fraction_of_merged_length"],
            "outputs": result.metadata["outputs"],
            "progress": result.metadata.get("progress"),
            "visual_review": result.metadata["outputs"].get("visual_review_json"),
            "warnings": result.metadata["warnings"],
            "errors": errors,
        }
        summary["warnings"].extend(f"{name}: {w}" for w in result.metadata["warnings"])
        summary["errors"].extend(f"{name}: {e}" for e in errors)

    summary_path = run_dir / "seak_fallback_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
