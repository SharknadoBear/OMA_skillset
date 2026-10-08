#!/usr/bin/env python3
"""Numerical and error-path regression tests; no model runtime or live data."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from grid_math import (
    fit_domain,
    coordinates,
    connected_mask,
    smooth_depth,
    roughness,
    depths,
    haney,
    staggered_masks,
    sample_netcdf,
)
from remora_grid import export_grid, readback, parameters, anchor_indices


class GridTests(unittest.TestCase):
    vertices = [[-75.4, 38.7], [-75, 38.7], [-75, 39.1], [-75.4, 39.1]]

    def test_rotations_and_staggering(self):
        for rotation in [0, 30, 70, -70]:
            f = fit_domain(self.vertices, 2000, rotation=rotation)
            a = coordinates(f)
            ny, nx = f["ny"], f["nx"]
            self.assertEqual(a["lon_rho"].shape, (ny + 2, nx + 2))
            self.assertEqual(a["lon_u"].shape, (ny + 2, nx + 1))
            self.assertEqual(a["lon_v"].shape, (ny + 1, nx + 2))
            self.assertEqual(a["lon_psi"].shape, (ny + 1, nx + 1))
            self.assertLess(a["orthogonality_error_deg"].max(), 0.001)
            np.testing.assert_allclose(1 / a["pm"], 2000, rtol=0.001)
            self.assertAlmostEqual(a["x_psi"][0, 0], 0)
            self.assertAlmostEqual(a["x_psi"][0, -1], nx * 2000)
            # Geographic orientation matches a positive Jacobian, including rotated grids.
            lon = a["lon_psi"]
            lat = a["lat_psi"]
            det = (
                np.diff(lon, axis=1)[:-1] * np.diff(lat, axis=0)[:, :-1]
                - np.diff(lat, axis=1)[:-1] * np.diff(lon, axis=0)[:, :-1]
            )
            self.assertTrue(np.all(det > 0))

    def test_budget(self):
        with self.assertRaises(ValueError):
            fit_domain(self.vertices, 10, max_cells=100)

    def test_invalid_spacing(self):
        with self.assertRaises(ValueError):
            fit_domain(self.vertices, 0)

    def test_channel_preserved(self):
        m = np.zeros((7, 9), bool)
        m[3, :] = 1
        m[0, 0] = 1
        final, r = connected_mask(m, [(3, 0), (3, 8)])
        self.assertTrue(final[3].all())
        self.assertFalse(final[0, 0])
        self.assertEqual(r["removed_cells"], 1)

    def test_disconnected_required_features(self):
        m = np.eye(5, dtype=bool)
        with self.assertRaises(ValueError):
            connected_mask(m, [(0, 0), (4, 4)])

    def test_volume_and_roughness(self):
        h = np.array([[2, 100, 4, 50], [10, 40, 2, 90]], float)
        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], float)
        m = np.ones(h.shape, bool)
        sm, r = smooth_depth(h, m, a)
        self.assertLessEqual(roughness(sm, m), 0.2 + 1e-8)
        self.assertLess(abs(r["volume_relative_change"]), 1e-12)
        self.assertGreaterEqual(sm.min(), h.min())

    def test_smoothing_failure(self):
        with self.assertRaises(ValueError):
            smooth_depth(
                np.array([[2.0, 100.0]]),
                np.ones((1, 2), bool),
                np.ones((1, 2)),
                max_iterations=0,
            )

    def test_vertical_limits(self):
        h = np.array([[2.0, 20.0], [100.0, 1000.0]])
        for ts, tb in [(0, 0), (0, 2), (6, 0), (6, 2)]:
            zr, zw, _ = depths(h, 40, ts, tb, 20)
            self.assertTrue(np.isfinite(zw).all())
            self.assertTrue((np.diff(zw, axis=0) > 0).all())
            np.testing.assert_allclose(zw[0], -h)
            np.testing.assert_allclose(zw[-1], 0)
            self.assertTrue((zr > zw[:-1]).all() and (zr < zw[1:]).all())

    def test_haney_is_layer_aware(self):
        h = np.array([[10.0, 20.0], [10.0, 20.0]])
        _, zw, _ = depths(h, 20)
        self.assertGreater(
            haney(zw, np.ones(h.shape, bool)), roughness(h, np.ones(h.shape, bool))
        )

    def test_psi_masks_all_patterns(self):
        # Independent truth table from pinned REMORA branch definitions.
        expected = [0, 0, 0, 2, 0, 2, 0, 1, 0, 0, 2, 1, 2, 1, 1, 1]
        for bits, want in enumerate(expected):
            m = np.array([(bits >> i) & 1 for i in range(4)]).reshape(2, 2)
            self.assertEqual(int(staggered_masks(m)["mask_psi"][0, 0]), want)

    def test_bathy_sign_gaps_and_units(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bathy.nc"
            with Dataset(p, "w") as d:
                d.createDimension("lon", 2)
                d.createDimension("lat", 2)
                d.createVariable("lon", "f8", ("lon",))[:] = [0, 1]
                d.createVariable("lat", "f8", ("lat",))[:] = [0, 1]
                z = d.createVariable("elevation", "f8", ("lat", "lon"))
                z[:] = [[-10, 10], [-10, 10]]
                z.units = "m"
                z.positive = "up"
            z = sample_netcdf(
                p,
                "elevation",
                np.array([[0.0, 1.0, 2.0]]),
                np.array([[0.5, 0.5, 0.5]]),
                "up",
            )
            self.assertEqual(z[0, 0], -10)
            self.assertEqual(z[0, 1], 10)
            self.assertTrue(np.isnan(z[0, 2]))
            with self.assertRaises(ValueError):
                sample_netcdf(
                    p, "elevation", np.zeros((1, 1)), np.zeros((1, 1)), "down"
                )

    def test_cdf5_and_stale_mask(self):
        with tempfile.TemporaryDirectory() as td:
            f = fit_domain(self.vertices, 10000)
            a = coordinates(f)
            h = np.full(a["pm"].shape, 20.0)
            m = np.ones(h.shape, bool)
            p = parameters({"parameters": {"spacing_m": 10000}})
            path = Path(td) / "grid.nc"
            export_grid(path, a, h, m, p, {"geometry": f}, h)
            self.assertEqual(readback(path)["format"], "NETCDF3_64BIT_DATA")
            with Dataset(path, "r+") as d:
                d["mask_u"][2, 2] = 0
            with self.assertRaises(ValueError):
                readback(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
