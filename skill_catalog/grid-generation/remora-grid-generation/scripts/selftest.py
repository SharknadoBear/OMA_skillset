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
    vertical_diagnostics,
    vertical_section,
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

    def test_log_mean_and_roughness(self):
        h = np.array([[2, 100, 4, 50], [10, 40, 2, 90]], float)
        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], float)
        m = np.ones(h.shape, bool)
        sm, r = smooth_depth(h, m, a)
        self.assertLessEqual(roughness(sm, m), 0.2 + 1e-8)
        self.assertLess(r["volume_relative_change"], 0)
        self.assertAlmostEqual(float(np.sum(a*np.log(h))), float(np.sum(a*np.log(sm))), places=10)
        self.assertGreaterEqual(sm.min(), h.min())

    def test_analytical_log_pair(self):
        # Weighted mean of logarithms is fixed; final ratio is (1+r)/(1-r).
        h=np.array([[2.,100.]])
        for area in [np.array([[1.,1.]]), np.array([[1.,3.]])]:
            mean=float(np.sum(area*np.log(h))/area.sum())
            jump=np.log(1.5)
            expected=np.exp([[mean-area[0,1]/area.sum()*jump,
                              mean+area[0,0]/area.sum()*jump]])
            sm,r=smooth_depth(h,np.ones_like(h,bool),area)
            np.testing.assert_allclose(sm,expected,rtol=1e-13)
            self.assertAlmostEqual(r['volume_relative_change'],float((np.sum(sm*area)-np.sum(h*area))/np.sum(h*area)))
            self.assertEqual(r['space'],'natural_log_depth')

    def test_acceptable_and_land_barriers(self):
        h=np.array([[2.,2.1,1000.,100.]])
        m=np.array([[True,True,False,True]])
        sm,r=smooth_depth(h,m,np.ones_like(h))
        np.testing.assert_array_equal(sm,h)
        self.assertEqual(r['iterations'],0)
        changed=np.array([[2.,100.,777.,5.,6.]])
        mask=np.array([[True,True,False,True,True]])
        sm,_=smooth_depth(changed,mask,np.ones_like(changed))
        np.testing.assert_array_equal(sm[0,2:],changed[0,2:])
        self.assertGreaterEqual(sm[mask].min(),2.)

    def test_smoothing_invalid_inputs(self):
        h=np.array([[2.,100.]])
        m=np.ones_like(h,bool)
        for invalid in [0.,-1.,np.nan,np.inf]:
            with self.subTest(depth=invalid),self.assertRaises(ValueError):
                smooth_depth(np.array([[invalid,10.]]),m,np.ones_like(h))
            with self.subTest(area=invalid),self.assertRaises(ValueError):
                smooth_depth(h,m,np.array([[invalid,1.]]))
        with self.assertRaises(ValueError): smooth_depth(h,np.zeros_like(m),np.ones_like(h))
        with self.assertRaises(ValueError): smooth_depth(h,m,np.ones((2,2)))
        with self.assertRaises(ValueError): smooth_depth(h,m,np.ones_like(h),rmax=np.nan)
        with self.assertRaises(ValueError): smooth_depth(h,m,np.ones_like(h),max_iterations=-1)

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

    def test_blocked_vertical_matches_full_transform(self):
        # Sharp eta jumps straddle block edges; land pairs must stay excluded.
        h = np.geomspace(2., 1200., 77).reshape(11, 7)
        h[3:6] *= 4
        mask = np.ones(h.shape, bool)
        mask[4:7, 2:4] = False
        surface = np.linspace(-0.5, 1.2, h.size).reshape(h.shape)
        for n, ts, tb in [(1, 0, 0), (8, 0, 2), (40, 6, 2)]:
            _, zw, _ = depths(h, n, ts, tb, 20, surface)
            for block_rows in [1, 3, 32]:
                got = vertical_diagnostics(h, n, ts, tb, 20, surface, mask, block_rows)
                self.assertEqual(got['minimum_layer_thickness_m'], float(np.diff(zw, axis=0).min()))
                self.assertEqual(got['haney_rx1_diagnostic'], haney(zw, mask))
            for axis, index in [(0, 0), (0, 10), (1, 0), (1, 6)]:
                section = vertical_section(h, axis, index, n, ts, tb, 20, surface)
                expected = zw[:, index, :] if axis == 0 else zw[:, :, index]
                np.testing.assert_array_equal(section, expected)
        self.assertEqual(vertical_diagnostics(np.ones((1, 1)), mask=np.ones((1, 1), bool))['haney_rx1_diagnostic'], 0.)

    def test_blocked_vertical_rejects_invalid_inputs(self):
        h = np.ones((4, 3))
        for bad in [np.zeros_like(h), -h, h * np.nan, np.ones(3), np.ones((0, 3))]:
            with self.assertRaises(ValueError): vertical_diagnostics(bad)
        for size in [0, -1, 1.5]:
            with self.assertRaises(ValueError): vertical_diagnostics(h, block_rows=size)
        with self.assertRaises(ValueError): vertical_diagnostics(h, mask=np.ones((2, 2)))
        with self.assertRaises(ValueError): vertical_diagnostics(h, zeta=-1.)
        with self.assertRaises(ValueError): vertical_diagnostics(h, zeta=np.nan)
        for axis, index in [(2, 0), (0, -1), (0, 4), (1, 3), (0, 1.5)]:
            with self.assertRaises(ValueError): vertical_section(h, axis, index)

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
                self.assertEqual(d.smoothing_space,"natural_log_depth")
                d["mask_u"][2, 2] = 0
            with self.assertRaises(ValueError):
                readback(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
