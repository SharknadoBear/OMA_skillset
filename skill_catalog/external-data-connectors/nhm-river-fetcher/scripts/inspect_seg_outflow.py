#!/usr/bin/env python3
"""Inspect NHM-PRMS seg_outflow.nc without assuming dimension names."""

import argparse
import json
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--netcdf", type=Path, required=True)
    parser.add_argument("--variable", default="seg_outflow")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/nhm_prms_ak/tables/seg_outflow_inventory.json"))
    args = parser.parse_args(argv)

    try:
        import xarray as xr
    except ImportError as exc:
        raise SystemExit("inspect_seg_outflow.py requires xarray for NetCDF inspection") from exc

    ds = xr.open_dataset(args.netcdf, decode_times=False)
    var_name = args.variable if args.variable in ds.data_vars else None
    if var_name is None:
        candidates = [name for name in ds.data_vars if "outflow" in name.lower()]
        var_name = candidates[0] if candidates else next(iter(ds.data_vars))
    da = ds[var_name]

    report = {
        "path": str(args.netcdf),
        "dimensions": {name: int(size) for name, size in ds.sizes.items()},
        "data_variables": {
            name: {
                "dims": list(ds[name].dims),
                "dtype": str(ds[name].dtype),
                "attrs": {key: str(value) for key, value in ds[name].attrs.items()},
            }
            for name in ds.data_vars
        },
        "coordinates": {
            name: {
                "dims": list(ds[name].dims),
                "dtype": str(ds[name].dtype),
                "attrs": {key: str(value) for key, value in ds[name].attrs.items()},
            }
            for name in ds.coords
        },
        "selected_variable": var_name,
        "selected_variable_dims": list(da.dims),
        "selected_variable_attrs": {key: str(value) for key, value in da.attrs.items()},
    }

    for coord in ds.coords:
        if "time" in coord.lower():
            values = ds[coord].values
            if values.size:
                report["time_coordinate"] = {
                    "name": coord,
                    "first_raw": str(values[0]),
                    "last_raw": str(values[-1]),
                    "attrs": {key: str(value) for key, value in ds[coord].attrs.items()},
                }
            break

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:6000])
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
