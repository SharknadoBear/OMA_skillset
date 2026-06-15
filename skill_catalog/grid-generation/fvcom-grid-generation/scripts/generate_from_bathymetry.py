"""Generate an FVCOM-ready 2DM mesh from a local bathymetry file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fvcom_grid_generation import (
    MeshBuildConfig,
    QualityThresholds,
    build_elliptical_domain,
    build_mesh,
    evaluate_mesh_quality,
    load_bathymetry,
)
from fvcom_grid_generation.size_field import SizeFieldConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an FVCOM/SMS 2DM mesh from local bathymetry.")
    parser.add_argument("bathymetry", help="Local NetCDF or GeoTIFF bathymetry.")
    parser.add_argument("--output-2dm", default="fvcom_grid.2dm")
    parser.add_argument("--mesh-name", default="fvcom_grid")
    parser.add_argument("--offshore-side", choices=["east", "west", "north", "south"], default=None)
    parser.add_argument("--min-size", type=float, default=1_000.0)
    parser.add_argument("--max-size", type=float, default=20_000.0)
    parser.add_argument("--gradation", type=float, default=0.15)
    parser.add_argument("--boundary-points", type=int, default=192)
    parser.add_argument("--quality-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bathy = load_bathymetry(args.bathymetry)
    size = SizeFieldConfig(min_size=args.min_size, max_size=args.max_size, gradation=args.gradation)
    config = MeshBuildConfig(mesh_name=args.mesh_name, size=size, boundary_points=args.boundary_points)
    domain = build_elliptical_domain(
        bathy,
        offshore_side=args.offshore_side,
        n_boundary=args.boundary_points,
    )
    mesh = build_mesh(bathy, domain=domain, config=config, output_2dm=args.output_2dm)
    quality = evaluate_mesh_quality(mesh.nodes, mesh.depths, mesh.triangles, mesh.open_boundary, QualityThresholds())
    serializable = {
        key: (value.tolist() if hasattr(value, "tolist") else value)
        for key, value in quality.items()
    }
    print(json.dumps(serializable, indent=2))
    if args.quality_json:
        Path(args.quality_json).write_text(json.dumps(serializable, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
