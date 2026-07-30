#!/usr/bin/env python3
"""Overlay a staged NOAA CRM volume onto an existing ETOPO fallback mosaic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--crm", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.metadata.exists():
        raise FileExistsError("Output and metadata paths must be fresh")

    base = xr.open_dataset(args.base).load()
    crm = xr.open_dataset(args.crm, decode_times=False)
    elevation_name = "z" if "z" in crm.data_vars else next(iter(crm.data_vars))
    target_lon = np.asarray(base["lon"].values, dtype=float)
    target_lat = np.asarray(base["lat"].values, dtype=float)
    crm = crm.sortby("lat").sortby("lon")
    west = max(float(target_lon.min()), float(crm["lon"].min()))
    east = min(float(target_lon.max()), float(crm["lon"].max()))
    south = max(float(target_lat.min()), float(crm["lat"].min()))
    north = min(float(target_lat.max()), float(crm["lat"].max()))
    if west >= east or south >= north:
        raise ValueError("Staged CRM volume does not overlap the target mosaic")
    subset = crm.sel(lat=slice(south, north), lon=slice(west, east))
    source_spacing = max(
        float(np.median(np.abs(np.diff(subset["lon"].values)))),
        float(np.median(np.abs(np.diff(subset["lat"].values)))),
    )
    target_spacing = max(
        float(np.median(np.abs(np.diff(target_lon)))),
        float(np.median(np.abs(np.diff(target_lat)))),
    )
    stride = max(1, int(round(target_spacing / source_spacing)))
    subset = subset.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
    crm_on_target = subset[elevation_name].interp(
        lon=base["lon"],
        lat=base["lat"],
        method="linear",
    ).load()
    crm_values = np.asarray(crm_on_target.values, dtype=np.float32)
    crm_valid = np.isfinite(crm_values)
    elevation = np.asarray(base["elevation_m"].values, dtype=np.float32)
    elevation[crm_valid] = crm_values[crm_valid]
    depth = np.where(np.isfinite(elevation), np.maximum(-elevation, 0.0), np.nan)
    wet = np.isfinite(elevation) & (elevation < 0.0)
    source_id = np.asarray(base["source_id"].values, dtype=np.int16)
    source_resolution = np.asarray(
        base["source_resolution_arcsec"].values,
        dtype=np.float32,
    )
    source_id[crm_valid] = 3
    source_resolution[crm_valid] = np.float32(source_spacing * 3600.0)

    output = xr.Dataset(
        {
            "elevation_m": (("lat", "lon"), elevation),
            "depth_m": (("lat", "lon"), depth.astype(np.float32)),
            "wet_mask": (("lat", "lon"), wet.astype(np.int8)),
            "source_id": (("lat", "lon"), source_id),
            "source_resolution_arcsec": (
                ("lat", "lon"),
                source_resolution,
            ),
        },
        coords={
            "lon": np.asarray(base["lon"].values, dtype=float),
            "lat": np.asarray(base["lat"].values, dtype=float),
        },
        attrs={
            **dict(base.attrs),
            "title": "CRM-first Hawaii bathymetry with ETOPO fallback",
            "source_priority": "NOAA Coastal Relief Model -> ETOPO 2022",
            "transport_note": (
                "CRM was staged from the bounded NOAA THREDDS fileServer "
                "endpoint after the equivalent large OPeNDAP request returned 403."
            ),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(args.output)
    counts = {
        "crm": int(np.count_nonzero(source_id == 3)),
        "etopo": int(np.count_nonzero(source_id == 4)),
        "none": int(np.count_nonzero(source_id == 0)),
    }
    total = int(source_id.size)
    metadata = {
        "schema_version": "gmsh_bathymetry_priority_merge_v1",
        "policy": "crm_first_etopo_fallback",
        "inputs": {
            "base_etopo_mosaic": {
                "path": str(args.base.resolve()),
                "sha256": sha256(args.base),
            },
            "staged_crm_volume": {
                "path": str(args.crm.resolve()),
                "sha256": sha256(args.crm),
            },
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
        },
        "crm_source_spacing_arcsec": float(source_spacing * 3600.0),
        "target_spacing_arcsec": float(target_spacing * 3600.0),
        "crm_sampling_stride": int(stride),
        "coverage_cells": counts,
        "coverage_fraction": {
            key: float(value / total) for key, value in counts.items()
        },
        "finite_fraction": float(np.mean(np.isfinite(elevation))),
        "datum_warning": base.attrs.get("datum_warning"),
        "transport_fallback": (
            "Single advertised CRM file staged from NOAA fileServer; no "
            "collection bulk download."
        ),
    }
    args.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
