#!/usr/bin/env python3
"""Offline synthetic contract tests for the TPXO9v5 connector core."""

from __future__ import annotations

import tempfile
from pathlib import Path

import netCDF4 as nc4
import numpy as np

from extract_tpxo9v5 import load_points_csv
from tpxo9v5.interpolation import complex_to_amplitude_phase, interpolate_complex_field
from tpxo9v5.io import HarmonicField
from tpxo9v5.outputs import validate_product, write_point_product


NAMES = [
    "M2", "S2", "N2", "K2", "2N2", "MU2", "NU2", "L2", "K1", "O1", "P1",
    "Q1", "2Q1", "J1", "OO1", "S1", "MF", "MM", "M4", "MN4", "MS4", "M3",
]


def field_with_dry_corner() -> HarmonicField:
    phase = np.deg2rad(np.asarray([359.0, 1.0]))
    coefficient = np.empty((22, 2, 2), dtype=np.complex128)
    for index in range(22):
        coefficient[index] = (index + 1) * 0.01 * np.exp(-1j * phase)[None, :]
    coefficient[:, 1, 1] = np.nan + 1j * np.nan
    return HarmonicField(
        name="elevation",
        grid="z",
        constituents=NAMES,
        longitude=np.asarray([280.0, 281.0]),
        latitude=np.asarray([28.0, 29.0]),
        coefficient=coefficient,
        units="m",
        depth=np.ones((2, 2)),
        mask=np.ones((2, 2)),
        source_basename="synthetic.nc",
        source_variables=("hRe", "hIm"),
        source_span={},
        actual_span={},
    )


def run() -> None:
    field = field_with_dry_corner()
    values, flags = interpolate_complex_field(
        field,
        np.asarray([-79.5, -79.0]),
        np.asarray([28.0, 29.0]),
        nearest_wet_max_degrees=2.0,
    )
    assert values.shape == (22, 2)
    assert set(np.unique(flags)).issubset({0, 1})
    assert np.all(flags[:, 1] == 1), "dry target must use bounded nearest-wet fallback"
    amplitude, phase = complex_to_amplitude_phase(
        np.asarray([np.exp(-1j * np.deg2rad(359.0)), np.exp(-1j * np.deg2rad(1.0))])
    )
    assert np.allclose(amplitude, 1.0)
    assert np.allclose(phase, [359.0, 1.0])

    with tempfile.TemporaryDirectory(prefix="tpxo9v5_selftest_") as temp:
        root = Path(temp)
        csv_path = root / "points.csv"
        csv_path.write_text(
            "node_id,longitude,latitude\n153,-79.5,28.0\n7,-79.0,29.0\n",
            encoding="utf-8",
        )
        lon, lat, shape, ids = load_points_csv(csv_path)
        assert ids == ["153", "7"] and shape == (2,)
        output = root / "points.nc"
        write_point_product(
            output,
            [(field, values, flags)],
            lon,
            lat,
            shape,
            {"test": "synthetic"},
            target_ids=ids,
        )
        health = validate_product(output)
        assert health["status"] == "pass"
        with nc4.Dataset(output) as ds:
            assert len(ds.dimensions["constituent"]) == 22
            assert [str(value) for value in ds["target_id"][:]] == ["153", "7"]
            assert list(ds["constituent_name"][:]) == NAMES
    print("PASS: TPXO9v5 connector synthetic tests")


if __name__ == "__main__":
    run()
