"""Run NOAA CUSP coastline smoke tests for representative FVCOM regions."""

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


SMOKE_CASES = {
    "delaware_bay": (-75.35, 38.75, -74.95, 39.10),
    "long_island_sound": (-73.30, 40.80, -72.40, 41.25),
    "puget_sound": (-123.15, 47.45, -122.35, 48.05),
    "se_ak_icy_strait": (-136.20, 58.10, -134.80, 58.60),
    "se_ak_sumner_strait": (-133.50, 55.60, -131.70, 56.40),
}


def _validate_outputs(metadata: dict[str, object], *, require_visual_review: bool = False) -> list[str]:
    errors: list[str] = []
    quality = metadata["quality"]
    outputs = metadata["outputs"]
    if quality["feature_count"] <= 0:
        errors.append("feature_count is zero")
    if not quality["bbox_intersects_output"]:
        errors.append("output bounds do not intersect bbox")
    for key in ("shapefile", "gpkg", "geojson", "satellite_png", "metadata_json"):
        path = Path(outputs[key])
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"{key} missing or empty: {path}")
    for key in ("visual_review_json", "visual_review_md"):
        if key not in outputs:
            errors.append(f"{key} missing from outputs")
            continue
        path = Path(outputs[key])
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"{key} missing or empty: {path}")
    if require_visual_review and not visual_review_passed(outputs):
        errors.append("agent visual review has not recorded a pass")
    try:
        readback = gpd.read_file(outputs["gpkg"], layer="coastline", engine="pyogrio")
        if len(readback) != quality["feature_count"]:
            errors.append("GeoPackage readback count mismatch")
    except Exception as exc:
        errors.append(f"GeoPackage readback failed: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--index", help="Existing or output cusp_region_index.json path.")
    parser.add_argument("--allow-no-basemap", action="store_true")
    parser.add_argument("--require-visual-review", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--client-timeout-s", type=float, default=0.0, help="0 means no hard client timeout.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-case progress text; JSONL is still written.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index) if args.index else run_dir / "cusp_region_index.json"
    if not index_path.exists():
        save_region_index(build_region_index(), index_path)

    summary = {"cases": {}, "warnings": [], "errors": []}
    for name, bbox in SMOKE_CASES.items():
        case_dir = run_dir / name
        result = fetch_cusp_bbox(
            index_path,
            bbox,
            run_dir=case_dir,
            name=name,
            allow_no_basemap=args.allow_no_basemap,
            heartbeat_seconds=args.heartbeat_seconds,
            client_timeout_s=args.client_timeout_s,
            quiet=args.quiet,
        )
        errors = _validate_outputs(result.metadata, require_visual_review=args.require_visual_review)
        summary["cases"][name] = {
            "bbox_wsen": bbox,
            "selected_region": result.metadata["selected_region"],
            "feature_count": result.metadata["quality"]["feature_count"],
            "source_date_ranges": result.metadata["quality"]["source_date_ranges"],
            "outputs": result.metadata["outputs"],
            "progress": result.metadata.get("progress"),
            "visual_review": result.metadata["outputs"].get("visual_review_json"),
            "warnings": result.metadata["warnings"],
            "errors": errors,
        }
        summary["warnings"].extend(f"{name}: {w}" for w in result.metadata["warnings"])
        summary["errors"].extend(f"{name}: {e}" for e in errors)

    summary_path = run_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
