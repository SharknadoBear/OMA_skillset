#!/usr/bin/env python3
"""Small deterministic checks for the boundary/interior size contract."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from fvcom_grid_generation.boundary_size_contract import (
    diagnose_boundary_size_contract,
)


def _fixture(root: Path, boundary_target: float, interior_target: float) -> tuple[Path, Path]:
    boundary = root / "boundary.geojson"
    boundary.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [float(lon), float(lat)],
                        },
                        "properties": {
                            "target_spacing_m": float(boundary_target),
                        },
                    }
                    for lon, lat in ((-76.0, 43.0), (-75.9, 43.1))
                ],
            }
        ),
        encoding="utf-8",
    )
    field = root / "field.nc"
    xr.Dataset(
        data_vars={
            "mesh_size_m": (
                ("lat", "lon"),
                np.full((2, 2), float(interior_target)),
            )
        },
        coords={"lat": [43.0, 43.1], "lon": [-76.0, -75.9]},
        attrs={"schema_version": "fvcom_size_field_v4"},
    ).to_netcdf(field)
    return boundary, field


def test_matched_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="boundary_size_match_") as value:
        boundary, field = _fixture(Path(value), 100.0, 100.0)
        report = diagnose_boundary_size_contract(boundary, field)
        assert report["status"] == "pass"
        assert report["conflict_vertex_count"] == 0
        assert report["target_size_attribution_valid"]


def test_conflicting_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="boundary_size_conflict_") as value:
        boundary, field = _fixture(Path(value), 100.0, 10.0)
        report = diagnose_boundary_size_contract(boundary, field)
        assert report["status"] == "conflict_detected"
        assert report["boundary_coarser_than_interior_count"] == 2
        assert not report["target_size_attribution_valid"]


if __name__ == "__main__":
    tests = [test_matched_contract, test_conflicting_contract]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")
