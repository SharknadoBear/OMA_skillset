#!/usr/bin/env python3
"""Apply constraint-preserving OceanMesh-style cleanup to an existing 2DM mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.bathymetry import load_bathymetry  # noqa: E402
from fvcom_grid_generation.boundary import load_boundary_package  # noqa: E402
from fvcom_grid_generation.comparison import compare_quality_documents, write_quality_comparison_plot  # noqa: E402
from fvcom_grid_generation.postprocess import PostprocessConfig, boundary_chains_from_mesh, postprocess_mesh  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection, project_points, unproject_points  # noqa: E402
from fvcom_grid_generation.quality import evaluate_mesh_quality  # noqa: E402
from fvcom_grid_generation.sms_2dm import read_2dm, write_2dm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--boundary-loops-gpkg")
    parser.add_argument("--bathy-nc")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--postprocess-profile", choices=("rpw2019", "projection-medium"), default="rpw2019")
    parser.add_argument("--postprocess-boundary-policy", choices=("protect-all", "protect-open"), default="protect-all")
    parser.add_argument("--postprocess-max-passes", type=int, default=8)
    parser.add_argument("--postprocess-connectivity-limit", default="auto")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    mesh = read_2dm(args.mesh)
    if args.boundary_loops_gpkg:
        projection = load_boundary_package(args.boundary_loops_gpkg).projection
    else:
        bbox = (
            float(np.min(mesh.nodes_lonlat[:, 0])),
            float(np.min(mesh.nodes_lonlat[:, 1])),
            float(np.max(mesh.nodes_lonlat[:, 0])),
            float(np.max(mesh.nodes_lonlat[:, 1])),
        )
        projection = local_utm_projection(bbox)
    xy = project_points(mesh.nodes_lonlat, projection)
    chains = boundary_chains_from_mesh(mesh.triangles)
    fixed = np.zeros(len(xy), dtype=bool)
    for chain in chains:
        fixed[np.asarray(chain, dtype=int)] = True
    limit = None if args.postprocess_connectivity_limit == "auto" else int(args.postprocess_connectivity_limit)
    pre_quality = evaluate_mesh_quality(
        xy,
        mesh.depths,
        mesh.triangles,
        mesh.open_boundary_nodes,
        {"boundary_constraint_recovered": True},
        constraint_chains=chains,
    )
    result = postprocess_mesh(
        xy,
        mesh.triangles,
        fixed,
        chains,
        mesh.open_boundary_nodes,
        PostprocessConfig(
            profile=args.postprocess_profile,
            boundary_policy=args.postprocess_boundary_policy,
            max_passes=args.postprocess_max_passes,
            connectivity_limit=limit,
        ),
    )
    lonlat = unproject_points(result.nodes_xy, projection)
    if args.bathy_nc:
        bathy = load_bathymetry(args.bathy_nc)
        depths = bathy.sample(lonlat[:, 0], lonlat[:, 1], fill_value=float(np.nanmedian(bathy.depth)))
    else:
        nearest = cKDTree(xy).query(result.nodes_xy)[1]
        depths = mesh.depths[np.asarray(nearest, dtype=int)]
    depths = np.maximum(np.where(np.isfinite(depths), depths, 2.0), 0.5)
    post_quality = evaluate_mesh_quality(
        result.nodes_xy,
        depths,
        result.triangles,
        result.open_boundary_nodes,
        {"boundary_constraint_recovered": True},
        constraint_chains=result.constraint_chains,
    )
    write_2dm(run_dir / "fvcom_grid_preclean.2dm", mesh.nodes_lonlat, mesh.depths, mesh.triangles, mesh.open_boundary_nodes, mesh_name=f"{args.name}_preclean")
    write_2dm(run_dir / "fvcom_grid.2dm", lonlat, depths, result.triangles, result.open_boundary_nodes, mesh_name=args.name)
    (run_dir / "mesh_quality_preclean.json").write_text(json.dumps(pre_quality, indent=2), encoding="utf-8")
    (run_dir / "mesh_quality.json").write_text(json.dumps(post_quality, indent=2), encoding="utf-8")
    (run_dir / "mesh_cleanup_report.json").write_text(json.dumps(result.report, indent=2), encoding="utf-8")
    (run_dir / "mesh_cleanup_history.jsonl").write_text(
        "".join(json.dumps({key: value for key, value in entry.items() if key != "after_full"}) + "\n" for entry in result.history),
        encoding="utf-8",
    )
    comparison = compare_quality_documents(pre_quality, post_quality)
    (run_dir / "mesh_quality_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    write_quality_comparison_plot(run_dir / "mesh_quality_comparison.png", pre_quality, post_quality, f"{args.name} post-generation cleanup")
    print(json.dumps({"run_dir": str(run_dir), "comparison": comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
