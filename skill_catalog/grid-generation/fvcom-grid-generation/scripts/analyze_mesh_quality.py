#!/usr/bin/env python3
"""Analyze an existing FVCOM 2DM mesh without modifying the mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fvcom_grid_generation.boundary import load_boundary_package  # noqa: E402
from fvcom_grid_generation.postprocess import boundary_chains_from_mesh  # noqa: E402
from fvcom_grid_generation.projection import local_utm_projection, project_points  # noqa: E402
from fvcom_grid_generation.quality import evaluate_mesh_quality  # noqa: E402
from fvcom_grid_generation.sms_2dm import read_2dm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--boundary-loops-gpkg")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mesh = read_2dm(args.mesh)
    if args.boundary_loops_gpkg:
        projection = load_boundary_package(args.boundary_loops_gpkg).projection
    else:
        bounds = (
            float(np.min(mesh.nodes_lonlat[:, 0])),
            float(np.min(mesh.nodes_lonlat[:, 1])),
            float(np.max(mesh.nodes_lonlat[:, 0])),
            float(np.max(mesh.nodes_lonlat[:, 1])),
        )
        projection = local_utm_projection(bounds)
    xy = project_points(mesh.nodes_lonlat, projection)
    chains = boundary_chains_from_mesh(mesh.triangles)
    quality = evaluate_mesh_quality(
        xy,
        mesh.depths,
        mesh.triangles,
        mesh.open_boundary_nodes,
        {"boundary_constraint_recovered": True, "source": "topological_boundary_from_2dm"},
        constraint_chains=chains,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "accepted": quality["accepted"],
                "q_l3_sigma": quality["oceanmesh_quality"]["q_l3_sigma"],
                "count_q_below_0_25": quality["oceanmesh_quality"]["count_q_below_0_25"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
