#!/usr/bin/env python3
"""Derive positive-down Lake Ontario depth from an ETOPO elevation subset.

ETOPO stores bed elevation relative to a sea-level vertical datum. FVCOM needs
water depth below the lake surface. This research-only conversion retains the
source elevation and derives depth from the declared Lake Ontario chart datum.
It does not claim to transform EGM2008 heights into IGLD 1985.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="ETOPO source mosaic NetCDF.")
    parser.add_argument("--output", required=True, help="Fresh lake-depth NetCDF.")
    parser.add_argument("--metadata", required=True, help="Conversion metadata JSON.")
    parser.add_argument(
        "--chart-datum-m",
        type=float,
        default=74.2,
        help="Lake Ontario chart datum/LWD in metres (IGLD 1985).",
    )
    parser.add_argument(
        "--minimum-depth-m",
        type=float,
        default=0.1,
        help="Positive shoreline floor applied where a boundary cell is at/above datum.",
    )
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    metadata_path = Path(args.metadata).resolve()
    if source == output:
        raise ValueError("Input is immutable; choose a fresh output path.")
    if args.minimum_depth_m <= 0:
        raise ValueError("--minimum-depth-m must be positive.")

    with xr.open_dataset(source, decode_times=False) as opened:
        ds = opened.load()
    if "elevation_m" not in ds:
        raise KeyError("Input must contain elevation_m.")

    elevation = np.asarray(ds["elevation_m"].values, dtype=np.float32)
    finite = np.isfinite(elevation)
    raw_depth = np.float32(args.chart_datum_m) - elevation
    depth = np.where(
        finite,
        np.maximum(raw_depth, np.float32(args.minimum_depth_m)),
        np.nan,
    ).astype(np.float32)
    clipped = finite & (raw_depth < args.minimum_depth_m)

    ds["depth_m"] = (ds["elevation_m"].dims, depth)
    ds["depth_m"].attrs.update(
        {
            "long_name": "Lake Ontario positive-down depth below declared chart datum",
            "units": "m",
            "positive": "down",
            "lake_surface_reference_m": float(args.chart_datum_m),
            "lake_surface_reference_datum": "IGLD 1985 chart datum/LWD",
            "minimum_depth_floor_m": float(args.minimum_depth_m),
        }
    )
    if "wet_mask" in ds:
        ds["wet_mask"] = (
            ds["elevation_m"].dims,
            (finite & (elevation < args.chart_datum_m)).astype(np.int8),
        )
    ds.attrs.update(
        {
            "title": "Lake Ontario ETOPO elevation with FVCOM positive-down depth",
            "lake_depth_conversion": (
                "depth_m=max(74.2 m - elevation_m, 0.1 m); apply only inside "
                "the independently defined GSHHG lake wet domain"
            ),
            "lake_chart_datum_m": float(args.chart_datum_m),
            "lake_chart_datum_reference": (
                "NOAA Great Lakes Low Water Datum, Lake Ontario 74.2 m IGLD 1985"
            ),
            "vertical_datum_caveat": (
                "ETOPO elevation is referenced to EGM2008 while the 74.2 m lake "
                "surface is IGLD 1985; no geodetic vertical transformation was "
                "applied in this research experiment."
            ),
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output)
    report = {
        "schema_version": "gmsh_lake_depth_conversion_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(source),
        "input_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "chart_datum_m": float(args.chart_datum_m),
        "chart_datum_vertical_reference": "IGLD 1985",
        "source_vertical_reference": "ETOPO metadata declares EGM2008",
        "minimum_depth_m": float(args.minimum_depth_m),
        "finite_cell_count": int(finite.sum()),
        "clipped_cell_count_full_bbox": int(clipped.sum()),
        "clipped_cell_fraction_full_bbox": (
            float(clipped.sum() / finite.sum()) if finite.any() else None
        ),
        "depth_min_m": float(np.nanmin(depth)),
        "depth_max_m": float(np.nanmax(depth)),
        "datum_caveat": (
            "This is a documented datum interpretation, not an EGM2008-to-IGLD85 "
            "vertical transformation. Boundary-domain coverage is audited downstream."
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
