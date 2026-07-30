#!/usr/bin/env python3
"""Derive positive-down Lake Superior depth from an ETOPO elevation subset.

ETOPO 2022 stores elevation relative to EGM2008.  FVCOM needs positive-down
water depth relative to the modeled lake surface.  This research conversion
uses the NOAA Lake Superior Low Water Datum of 183.2 m on IGLD 1985, retains
the source elevation, and explicitly does not claim an EGM2008-to-IGLD 1985
vertical transformation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely import contains_xy
import xarray as xr


NOAA_LWD_URL = "https://tidesandcurrents.noaa.gov/gldatums.html"
ETOPO_DOI = "10.25921/fd45-gt74"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_fresh_output_files(
    *,
    source: Path,
    domain_gpkg: Path,
    output: Path,
    metadata_path: Path,
) -> None:
    """Reject aliases and existing products before any conversion write."""
    if output == metadata_path:
        raise ValueError("NetCDF output and metadata JSON must be different paths.")
    immutable_inputs = {source, domain_gpkg}
    for label, path in (
        ("NetCDF output", output),
        ("metadata JSON", metadata_path),
    ):
        if path in immutable_inputs:
            raise ValueError(f"{label} cannot overwrite an immutable input: {path}")
        if path.exists():
            raise FileExistsError(f"{label} must not already exist: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="ETOPO source mosaic NetCDF.")
    parser.add_argument("--output", required=True, help="Fresh lake-depth NetCDF.")
    parser.add_argument("--metadata", required=True, help="Conversion metadata JSON.")
    parser.add_argument(
        "--domain-gpkg",
        required=True,
        help=(
            "Accepted Lake Superior boundary package; its resolved wet domain "
            "defines the audit mask."
        ),
    )
    parser.add_argument(
        "--chart-datum-m",
        type=float,
        default=183.2,
        help="Lake Superior chart datum/LWD in metres (IGLD 1985).",
    )
    parser.add_argument(
        "--minimum-depth-m",
        type=float,
        default=0.1,
        help="Positive shoreline floor applied where a cell is at/above datum.",
    )
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    metadata_path = Path(args.metadata).resolve()
    domain_gpkg = Path(args.domain_gpkg).resolve()
    _require_fresh_output_files(
        source=source,
        domain_gpkg=domain_gpkg,
        output=output,
        metadata_path=metadata_path,
    )
    if args.minimum_depth_m <= 0:
        raise ValueError("--minimum-depth-m must be positive.")

    with xr.open_dataset(source, decode_times=False) as opened:
        ds = opened.load()
    if "elevation_m" not in ds:
        raise KeyError("Input must contain elevation_m.")
    longitude_name = next(
        (name for name in ("longitude", "lon", "x") if name in ds.coords),
        None,
    )
    latitude_name = next(
        (name for name in ("latitude", "lat", "y") if name in ds.coords),
        None,
    )
    if longitude_name is None or latitude_name is None:
        raise KeyError("Input must expose longitude/latitude coordinates.")
    domain_frame = gpd.read_file(
        domain_gpkg,
        layer="resolved_domain_polygon",
    ).to_crs("EPSG:4326")
    if len(domain_frame) != 1:
        raise ValueError(
            f"Expected one resolved wet domain, found {len(domain_frame)}."
        )
    domain = domain_frame.geometry.iloc[0]
    longitude = np.asarray(ds[longitude_name].values, dtype=float)
    latitude = np.asarray(ds[latitude_name].values, dtype=float)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    wet_domain_mask = np.asarray(
        contains_xy(domain, lon_grid, lat_grid),
        dtype=bool,
    )
    if not wet_domain_mask.any():
        raise ValueError("Accepted wet domain contains no bathymetry cells.")

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
            "long_name": "Lake Superior positive-down depth below declared chart datum",
            "units": "m",
            "positive": "down",
            "lake_surface_reference_m": float(args.chart_datum_m),
            "lake_surface_reference_datum": "IGLD 1985 chart datum/LWD",
            "minimum_depth_floor_m": float(args.minimum_depth_m),
        }
    )
    ds["fvcom_wet_domain_mask"] = (
        ds["elevation_m"].dims,
        wet_domain_mask.astype(np.int8),
    )
    ds["fvcom_wet_domain_mask"].attrs.update(
        {
            "long_name": (
                "accepted GSHHG Lake Superior wet-domain center mask"
            ),
            "flag_values": "0 1",
            "flag_meanings": "outside_wet_domain inside_wet_domain",
            "source_boundary_gpkg": str(domain_gpkg),
            "source_boundary_gpkg_sha256": sha256(domain_gpkg),
        }
    )
    if "wet_mask" in ds:
        ds["wet_mask"] = (
            ds["elevation_m"].dims,
            (finite & (elevation < args.chart_datum_m)).astype(np.int8),
        )
    ds.attrs.update(
        {
            "title": "Lake Superior ETOPO elevation with FVCOM positive-down depth",
            "lake_depth_conversion": (
                f"depth_m=max({args.chart_datum_m} m - elevation_m, "
                f"{args.minimum_depth_m} m); apply only inside the independently "
                "defined GSHHG Lake Superior wet domain"
            ),
            "lake_chart_datum_m": float(args.chart_datum_m),
            "lake_chart_datum_reference": (
                "NOAA Great Lakes Low Water Datum, Lake Superior "
                "183.2 m IGLD 1985"
            ),
            "lake_chart_datum_source_url": NOAA_LWD_URL,
            "source_dataset": "NOAA NCEI ETOPO 2022 15 arc-second surface elevation",
            "source_doi": ETOPO_DOI,
            "vertical_datum_caveat": (
                "ETOPO elevation is referenced to EGM2008 while the 183.2 m "
                "lake surface is IGLD 1985; no geodetic vertical transformation "
                "was applied in this research experiment."
            ),
            "wet_domain_mask_policy": (
                "fvcom_wet_domain_mask identifies accepted wet cell centers; "
                "finite depth outside the mask is retained only so strict "
                "near-boundary interpolation has complete raster support"
            ),
            "lake_depth_conversion_metadata_json": str(metadata_path),
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output)
    report = {
        "schema_version": "gmsh_lake_superior_depth_conversion_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(source),
        "input_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "domain_gpkg": str(domain_gpkg),
        "domain_gpkg_sha256": sha256(domain_gpkg),
        "chart_datum_m": float(args.chart_datum_m),
        "chart_datum_vertical_reference": "IGLD 1985",
        "chart_datum_source_url": NOAA_LWD_URL,
        "source_dataset": "NOAA NCEI ETOPO 2022 15 arc-second surface elevation",
        "source_doi": ETOPO_DOI,
        "source_vertical_reference": "ETOPO metadata declares EGM2008",
        "minimum_depth_m": float(args.minimum_depth_m),
        "finite_cell_count": int(finite.sum()),
        "wet_domain_cell_count": int(wet_domain_mask.sum()),
        "wet_domain_finite_depth_fraction": float(
            np.isfinite(depth[wet_domain_mask]).mean()
        ),
        "wet_domain_positive_depth_fraction": float(
            (depth[wet_domain_mask] > 0.0).mean()
        ),
        "wet_domain_depth_min_m": float(
            np.nanmin(depth[wet_domain_mask])
        ),
        "wet_domain_depth_max_m": float(
            np.nanmax(depth[wet_domain_mask])
        ),
        "clipped_cell_count_full_bbox": int(clipped.sum()),
        "clipped_cell_fraction_full_bbox": (
            float(clipped.sum() / finite.sum()) if finite.any() else None
        ),
        "depth_min_m": float(np.nanmin(depth)),
        "depth_max_m": float(np.nanmax(depth)),
        "all_finite_source_cells_have_positive_depth": bool(
            np.all(depth[finite] > 0.0)
        ),
        "datum_caveat": (
            "This is a documented datum interpretation, not an "
            "EGM2008-to-IGLD85 vertical transformation. Full wet-domain "
            "coverage is audited by validate_lake_superior_preparation.py."
        ),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
