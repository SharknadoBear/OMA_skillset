#!/usr/bin/env python3
"""Exercise both installed command-line interfaces with a small analytic source fixture."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from netCDF4 import Dataset
from remora_grid import REGION_SCRIPTS, read_json, write_json, validate_grid


def run(script, *args, success=True):
    p = subprocess.run(
        [sys.executable, str(script), *map(str, args)], capture_output=True, text=True
    )
    if success and p.returncode:
        raise AssertionError(p.stdout + "\n" + p.stderr)
    if not success and not p.returncode:
        raise AssertionError("Expected the invalid input to fail")
    return p


def main(v2=False):
    grid = Path(__file__).with_name("remora_grid.py")
    region = REGION_SCRIPTS / "remora_regions.py"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gpd.GeoDataFrame(geometry=[box(-75.8, 38.2, -75.5, 39.8)], crs=4326).to_file(
            root / "land.geojson", driver="GeoJSON"
        )
        reg = {
            "name": "integration_fixture",
            "objective": "Test complete-request orchestration, not a scientific regional product",
            "coastline": {"path": "land.geojson", "bbox_wsen": [-76, 38, -74, 40]},
            "regions": [
                {
                    "id": "outer",
                    "purpose": "cover fixture water",
                    "vertices_lonlat": [
                        [-75.6, 38.5],
                        [-75.1, 38.5],
                        [-75.1, 39],
                        [-75.6, 39],
                    ],
                    "required_features": [],
                }
            ],
        }
        write_json(root / "region_request.json", reg)
        with Dataset(root / "bathy.nc", "w") as d:
            d.createDimension("lon", 51)
            d.createDimension("lat", 51)
            d.createVariable("lon", "f8", ("lon",))[:] = np.linspace(-76, -74, 51)
            d.createVariable("lat", "f8", ("lat",))[:] = np.linspace(38, 40, 51)
            v = d.createVariable("elevation_m", "f8", ("lat", "lon"))
            v.units = "m"
            v.positive = "up"
            v[:] = -20
        req = {
            "region_request": "region_request.json",
            "parameters": {"spacing_m": 5000, "N": 8},
            "bathymetry": {
                "path": "bathy.nc",
                "variable": "elevation_m",
                "positive": "up",
            },
        }
        if v2:
            reg["modeling_purpose"]="barotropic_tide"
            reg["science_rationale"]="Synthetic constant-density geometry integration test"
            write_json(root/"region_request.json",reg)
            req["science"]={"regime":"barotropic_tide","rationale":reg["science_rationale"]}
            req["design"]={
                "coverage_rationale":"Synthetic fixture coverage",
                "resolution_rationale":"Exercise curved and graded coordinates",
                "cell_budget_rationale":"Small analytic integration fixture",
                "boundary_intents":{side:{"intent":"undetermined","rationale":"Synthetic reader fixture"}
                                    for side in ("xlo","xhi","ylo","yhi")}}
            req["parameters"].update(vertical_rationale="Eight layers for constant-density fixture",
                                     theta_s=2,theta_b=0,hc_m=10)
            req["geometry"]={"kind":"conformal_graded","nx":32,"ny":32,
                             "xi_control":[[0,0],[.5,.6],[1,1]],
                             "eta_control":[[0,0],[1,1]],"c2":[0,.025],"c3":[0,0]}
        write_json(root / "request.json", req)
        case_root = root / "case"
        case = case_root / "attempts/attempt_001"
        write_json(case_root / "science_request.json", {"location":"Synthetic fixture","modeling_purpose":"CLI validation"})
        run(grid, "prepare", "--request", root / "request.json", "--output-dir", case)
        # Expected pause: producing maps alone must not satisfy visual review.
        run(
            grid,
            "fit",
            "--request",
            case / "grid_request.json",
            "--output-dir",
            case / "02_fit",
            success=False,
        )
        delivery = case / "01_regions/region_delivery.json"
        run(
            region,
            "review",
            "--delivery",
            delivery,
            "--decision",
            "accepted",
            "--rationale",
            "Automated review-binding fixture, not a scientific map review",
        )
        run(
            grid,
            "fit",
            "--request",
            case / "grid_request.json",
            "--output-dir",
            case / "02_fit",
        )
        fit = case / "02_fit/grid_fit.json"
        run(
            grid, "build", "--fit", fit, "--output-dir", case / "03_grid", success=False
        )
        run(
            grid,
            "review-fit",
            "--artifact",
            fit,
            "--decision",
            "accepted",
            "--rationale",
            "Automated fixture for the fit review handoff",
        )
        run(grid, "build", "--fit", fit, "--output-dir", case / "03_grid")
        gd = case / "03_grid/grid_delivery.json"
        run(
            grid,
            "review",
            "--artifact",
            gd,
            "--decision",
            "accepted",
            "--rationale",
            "Automated fixture for the final review handoff",
        )
        run(grid, "validate", "--delivery", gd, "--require-reviewed")
        run(grid, "validate", "--delivery", gd, "--require-reader", success=False)
        run(grid, "finalize", "--case-root", case_root, "--delivery", gd, "--release-revision", "unpublished")
        final = case_root / "final/case_delivery.json"
        run(grid, "validate-case", "--delivery", final)
        run(grid, "finalize", "--case-root", case_root, "--delivery", gd, "--release-revision", "unpublished", success=False)
        run(grid, "finalize", "--case-root", root/"unrelated", "--delivery", gd, "--release-revision", "unpublished", success=False)
        with Dataset(case_root / "final/remora_grid.nc", "r+") as ds:
            ds["h"][2,2] = 900
        run(grid, "validate-case", "--delivery", final, success=False)
        reg["objective"] = "changed intent"
        write_json(root / "region_request.json", reg)
        run(
            grid,
            "prepare",
            "--request",
            root / "request.json",
            "--output-dir",
            case,
            success=False,
        )
        print(
            json.dumps(
                {
                    "status": "pass",
                    "entrypoints": [
                        "prepare",
                        "region build/review",
                        "fit/review-fit",
                        "build/review",
                        "validate", "finalize", "validate-case",
                    ],
                    "error_paths": [
                        "unreviewed region",
                        "unreviewed fit",
                        "missing reader evidence",
                        "changed source request", "final overwrite", "outside case directory", "modified final grid",
                    ],
                }
            )
        )


if __name__ == "__main__":
    main(v2="--v2" in sys.argv)
