"""Create a synthetic sloping-shelf bathymetry and generate a test 2DM mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from fvcom_grid_generation import (
    MeshBuildConfig,
    QualityThresholds,
    build_elliptical_domain,
    build_mesh,
    evaluate_mesh_quality,
    load_bathymetry,
)
from fvcom_grid_generation.size_field import SizeFieldConfig
from fvcom_grid_generation.view_mesh import plot_2dm


def create_synthetic_bathy(path: Path) -> Path:
    lon = np.linspace(-75.9, -73.4, 140)
    lat = np.linspace(37.5, 40.4, 150)
    lon2, lat2 = np.meshgrid(lon, lat)
    east = (lon2 - lon.min()) / (lon.max() - lon.min())
    north = (lat2 - lat.min()) / (lat.max() - lat.min())
    shelf = 3.0 + 90.0 * east**1.6
    channel = 22.0 * np.exp(-((lat2 - (38.3 + 0.9 * east)) ** 2) / 0.01)
    bank = -12.0 * np.exp(-((east - 0.65) ** 2 + (north - 0.55) ** 2) / 0.025)
    depth = np.maximum(shelf + channel + bank, 2.0)
    ds = xr.Dataset({"depth": (("lat", "lon"), depth)}, coords={"lon": lon, "lat": lat})
    ds.to_netcdf(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic FVCOM grid-generation smoke test.")
    parser.add_argument("--run-dir", default="Workspace/Grid_preprocessing/runs/synthetic_smoke")
    parser.add_argument("--min-size", type=float, default=9_000.0)
    parser.add_argument("--max-size", type=float, default=32_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    bathy_path = create_synthetic_bathy(run_dir / "synthetic_sloping_shelf.nc")
    bathy = load_bathymetry(bathy_path)
    config = MeshBuildConfig(
        mesh_name="synthetic_fvcom_grid",
        boundary_points=160,
        size=SizeFieldConfig(
            min_size=args.min_size,
            max_size=args.max_size,
            nearshore_max_size=args.min_size,
            shelf_max_size=0.6 * args.max_size,
        ),
    )
    domain = build_elliptical_domain(bathy, offshore_side="east", n_boundary=config.boundary_points)
    mesh = build_mesh(bathy, domain=domain, config=config, output_2dm=run_dir / "synthetic_fvcom_grid.2dm")
    quality = evaluate_mesh_quality(mesh.nodes, mesh.depths, mesh.triangles, mesh.open_boundary, QualityThresholds())
    serializable = {
        key: (value.tolist() if hasattr(value, "tolist") else value)
        for key, value in quality.items()
    }
    (run_dir / "quality_report.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    plot_2dm(run_dir / "synthetic_fvcom_grid.2dm", run_dir / "synthetic_fvcom_grid.png")
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
