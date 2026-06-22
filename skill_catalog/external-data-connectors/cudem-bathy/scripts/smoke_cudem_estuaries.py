"""Run required CUDEM smoke tests for selected U.S. coastal regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cudem_bathy.catalog import build_tile_index, save_tile_index  # noqa: E402
from cudem_bathy.fetch import fetch_cudem_bbox  # noqa: E402

SMOKE_CASES = {
    "se_ak_icy_strait_juneau": (-136.00, 58.25, -135.50, 58.49),
    "delaware_bay": (-75.35, 38.75, -74.95, 39.10),
    "long_island_sound": (-73.30, 40.80, -72.40, 41.25),
    "puget_sound": (-123.15, 47.45, -122.35, 48.05),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--index")
    parser.add_argument("--max-tiles", type=int, default=48)
    parser.add_argument(
        "--target-spacing-arcsec",
        type=float,
        default=9.0,
        help="Smoke-test output spacing; keeps live tests small while using CUDEM sources.",
    )
    parser.add_argument("--only", choices=sorted(SMOKE_CASES), nargs="*")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index) if args.index else run_dir / "cudem_tile_index.json"
    if args.rebuild_index or not index_path.exists():
        index = build_tile_index()
        save_tile_index(index, index_path)

    names = args.only or list(SMOKE_CASES)
    summary: dict[str, dict] = {}
    for name in names:
        case_dir = run_dir / name
        try:
            result = fetch_cudem_bbox(
                index_path,
                SMOKE_CASES[name],
                run_dir=case_dir,
                name=name,
                resolution="auto",
                max_tiles=args.max_tiles,
                target_spacing_arcsec=args.target_spacing_arcsec,
            )
            summary[name] = {
                "status": "passed",
                "metadata": str(result.metadata_path),
                "netcdf": str(result.netcdf_path),
                "png": str(result.png_path),
                "source_modes": result.metadata.get("source_modes"),
                "n_tiles": result.metadata.get("n_tiles"),
                "finite_output_fraction": result.metadata.get("finite_output_fraction"),
            }
        except Exception as exc:
            summary[name] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
    summary_path = run_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "cases": summary}, indent=2))
    return 0 if all(item["status"] == "passed" for item in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
