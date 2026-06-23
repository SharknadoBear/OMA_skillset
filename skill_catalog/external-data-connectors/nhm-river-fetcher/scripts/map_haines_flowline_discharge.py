#!/usr/bin/env python3
"""Scaffold a Haines NHM flowline discharge diagnostic map."""

import argparse
from pathlib import Path
from typing import List, Optional


DEFAULT_BBOX = (-136.5, -134.5, 58.5, 59.75)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True, help="NHM segment geometry file readable by geopandas.")
    parser.add_argument("--netcdf", type=Path, required=True, help="seg_outflow.nc path.")
    parser.add_argument("--segment-id-field", default=None)
    parser.add_argument("--variable", default="seg_outflow")
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX, metavar=("MINLON", "MAXLON", "MINLAT", "MAXLAT"))
    parser.add_argument("--out-map", type=Path, default=Path("outputs/nhm_prms_ak/maps/haines_flowline_discharge.png"))
    args = parser.parse_args(argv)

    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import xarray as xr
    except ImportError as exc:
        raise SystemExit("map_haines_flowline_discharge.py requires geopandas, xarray, and matplotlib") from exc

    gdf = gpd.read_file(args.geometry)
    if gdf.crs is None:
        raise SystemExit("geometry has no CRS; assign one before mapping")
    gdf_ll = gdf.to_crs("EPSG:4326")
    min_lon, max_lon, min_lat, max_lat = args.bbox
    subset = gdf_ll.cx[min_lon:max_lon, min_lat:max_lat].copy()
    if subset.empty:
        raise SystemExit("no flowlines intersect requested bbox")

    ds = xr.open_dataset(args.netcdf, decode_times=False)
    if args.variable not in ds:
        raise SystemExit(f"variable {args.variable!r} not found in {args.netcdf}")
    discharge = ds[args.variable]
    reduce_dims = [dim for dim in discharge.dims if "time" in dim.lower()]
    discharge_mean = discharge.mean(dim=reduce_dims) if reduce_dims else discharge

    seg_field = args.segment_id_field
    if seg_field is None:
        candidates = [col for col in subset.columns if col.lower() in {"seg_id", "nhm_seg", "segment_id", "poi_id", "id"}]
        if not candidates:
            raise SystemExit("provide --segment-id-field; no obvious segment id field found")
        seg_field = candidates[0]

    segment_dim = discharge_mean.dims[-1]
    segment_ids = ds[segment_dim].values if segment_dim in ds.coords else range(discharge_mean.size)
    value_by_segment = dict(zip([str(value) for value in segment_ids], discharge_mean.values.ravel()))
    subset["nhm_discharge"] = subset[seg_field].astype(str).map(value_by_segment)

    args.out_map.parent.mkdir(parents=True, exist_ok=True)
    ax = subset.plot(column="nhm_discharge", legend=True, linewidth=1.5, figsize=(8, 8), missing_kwds={"color": "lightgray"})
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_title("NHM-PRMS Flowline Discharge - Haines / Upper Lynn Canal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(args.out_map, dpi=180)
    print(f"wrote {args.out_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
