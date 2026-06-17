"""Interpolate a CUDEM bathymetry NetCDF to FVCOM mesh nodes or points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cudem_bathy.interp import interpolate_to_points, write_points_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bathy-netcdf", required=True)
    parser.add_argument("--output-csv", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mesh-2dm")
    group.add_argument("--grid-dat")
    group.add_argument("--points-csv")
    parser.add_argument("--method", default="linear", choices=["linear", "nearest"])
    args = parser.parse_args()

    result = interpolate_to_points(
        args.bathy_netcdf,
        mesh_2dm=args.mesh_2dm,
        grid_dat=args.grid_dat,
        csv_path=args.points_csv,
        method=args.method,
    )
    write_points_csv(result, args.output_csv)
    ok = np.asarray(result["status"]) == "ok"
    print(
        json.dumps(
            {
                "output_csv": args.output_csv,
                "points": int(len(ok)),
                "ok_points": int(ok.sum()),
                "missing_points": int((~ok).sum()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
