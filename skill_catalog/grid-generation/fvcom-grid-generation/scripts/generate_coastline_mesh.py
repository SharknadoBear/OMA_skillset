"""Generate an FVCOM 2DM mesh from an approved coastline-aware domain."""

from __future__ import annotations

import argparse
import json

from fvcom_grid_generation.gmsh_builder import generate_coastline_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a coastline-aware FVCOM mesh with Gmsh.")
    parser.add_argument("domain_metadata_json", help="Prepared domain metadata JSON.")
    parser.add_argument("bathymetry", help="Bathymetry used for depths and size field.")
    parser.add_argument("--output-2dm", required=True)
    parser.add_argument("--mesh-name", default="fvcom_coastline_grid")
    parser.add_argument("--quality-json", required=True)
    parser.add_argument("--review-json", default=None)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--include-island-holes", action="store_true", help="Include filtered island holes in the Gmsh surface.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = generate_coastline_mesh(
            args.domain_metadata_json,
            args.bathymetry,
            args.output_2dm,
            args.mesh_name,
            args.quality_json,
            review_json=args.review_json,
            max_attempts=args.max_attempts,
            include_island_holes=args.include_island_holes,
        )
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    print(json.dumps({"output_2dm": str(result.output_2dm), "quality_json": str(result.quality_json), "summary_json": str(result.summary_json), "accepted": result.quality.get("accepted", False)}, indent=2))


if __name__ == "__main__":
    main()
