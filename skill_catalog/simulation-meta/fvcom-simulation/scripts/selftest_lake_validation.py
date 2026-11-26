"""Focused lake science and native-output reader regressions. Run directly with Python."""
import tempfile
import unittest
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from lake_native_io import at_depth, compact, on_clock, station_clock, thermocline, wet_mean
from lake_validation import clock_coverage, daily_surface, reference_metrics, station_metrics


class LakeValidationTests(unittest.TestCase):
    def test_native_float32_station_time_needs_agreeing_internal_steps(self):
        with tempfile.TemporaryDirectory() as work, netCDF4.Dataset(Path(work)/"station.nc","w") as ds:
            ds.createDimension("time",4)
            t=ds.createVariable("time","f4",("time",))
            t.units="days since 2018-01-01 00:00:00"
            t[:]=2738+np.arange(4)/24
            ds.createVariable("iint","i4",("time",))[:]=np.arange(4)*720
            recovered,info=station_clock(ds,3600)
            np.testing.assert_array_equal(recovered-recovered[0],[0,3600,7200,10800])
            self.assertEqual(info["adjusted_records"],2)
            self.assertEqual(info["seconds_per_internal_step"],5.)
            ds["iint"][2]=1430
            with self.assertRaisesRegex(ValueError,"internal model steps"):
                station_clock(ds,3600)
            ds["iint"][2]=1440
            t[2]=float(t[2])+40/86400
            with self.assertRaisesRegex(ValueError,"packing precision"):
                station_clock(ds,3600)

    def test_native_spherical_area_and_wet_mask(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work)
            metric_path = path/"native_areas.npz"
            np.savez(metric_path, node_id=[1, 2, 3], node_control_area_m2=[2., 5., 8.],
                     lon=[-90., -91., -89.], lat=[47., 48., 49.])
            history = path/"history.nc"
            with netCDF4.Dataset(history, "w") as ds:
                for dim, size in [("time", 2), ("siglay", 2), ("node", 3)]:
                    ds.createDimension(dim, size)
                for name, dims, unit, data in [
                    ("time", ("time",), "seconds since 1970-01-01 00:00:00", [0., 60.]),
                    ("lon", ("node",), "degrees_east", [270., 269., 271.]),
                    ("lat", ("node",), "degrees_north", [47., 48., 49.]),
                    ("h", ("node",), "m", [10., 20., 30.]),
                    ("zeta", ("time", "node"), "m", np.zeros((2, 3))),
                    ("temp", ("time", "siglay", "node"), "degrees_C",
                     [[[2., 100., 12.], [0., 0., 0.]], [[2., 10., 12.], [0., 0., 0.]]]),
                    ("wet_nodes", ("time", "node"), "1", [[1, 0, 1], [0, 1, 0]])
                ]:
                    v = ds.createVariable(name, "f8", dims)
                    v.units = unit
                    v[:] = data
            table = compact([history], metric_path, 0, 120)
            np.testing.assert_allclose(table.temperature_c, [10., 10.])
            np.testing.assert_allclose(table.wet_area_fraction, [2/3, 1/3])
            with netCDF4.Dataset(history, "r+") as ds:
                ds["lon"][0] = 271.
            with self.assertRaisesRegex(ValueError, "geography/order"):
                compact([history], metric_path, 0, 120)
        with self.assertRaisesRegex(ValueError, "wet nodes"):
            wet_mean(np.array([np.nan, 2.]), np.ones(2), np.array([1, 1]))

    def test_depth_and_clock_support(self):
        model = {"time": np.array([0, 60, 120]), "depth": [20.], "sigma": np.array([[-.25], [-.75]]),
                 "zeta": np.array([[0.], [0.], [4.]]), "temp": np.array([[[10.], [4.]]]*3)}
        result, half = at_depth(model, 0, np.array([0, 120]), 12., 60)
        np.testing.assert_allclose(result, [5.8, 7.])
        self.assertFalse(half.any())
        surface, half = at_depth(model, 0, np.array([0]), 1., 60)
        self.assertEqual(surface[0], 10.)
        self.assertTrue(half[0])
        self.assertTrue(np.isnan(at_depth(model, 0, np.array([0]), 21., 60)[0][0]))
        sparse_time = np.array([0, 60, 240])
        interp = on_clock(sparse_time, np.array([2., 4., 8.]), np.array([30, 180, 240, 300]), 60)
        np.testing.assert_allclose(interp, [3., np.nan, 8., np.nan], equal_nan=True)

    def test_coverage_and_thermocline(self):
        times = np.arange(0, 86400, 21600)
        self.assertTrue(clock_coverage(times, 0, 86400, 21600)["complete"])
        jitter = clock_coverage(np.array([0, 359, 720]), 0, 1080, 360)
        self.assertTrue(jitter["complete"])
        self.assertEqual(jitter["jittered_records"], 1)
        self.assertFalse(clock_coverage(np.array([0, 359, 360]), 0, 1080, 360)["complete"])
        self.assertFalse(clock_coverage(np.array([0, 358, 720]), 0, 1080, 360)["complete"])
        area = pd.DataFrame({"epoch": list(times)+[86400], "temperature_c": [2., 4., 6., 8., 20.],
                             "wet_area_fraction": [1.]*5})
        obs = pd.DataFrame({"date": ["1970-01-01", "1970-01-02"], "observed": [4., 5.]})
        daily = daily_surface(area, obs, 0, 2*86400, 21600)
        self.assertEqual(daily.complete.tolist(), [True, False])
        self.assertEqual(daily.model.iloc[0], 5.)
        result = thermocline(np.array([1., 5., 10., 20.]), np.array([[15., 14., 7., 6.], [6., 6., 6., 6.]]))
        np.testing.assert_allclose(result, [7.5, np.nan], equal_nan=True)

    def test_datum_and_separate_noaa_metrics(self):
        clock = np.array([0, 60, 120])
        model = {"time": clock, "ids": ["O1", "R1"], "depth": [10., 10.],
                 "sigma": np.array([[-.25, -.25], [-.75, -.75]]),
                 "zeta": np.array([[0., 1.], [1., 2.], [2., 3.]]), "temp": np.ones((3, 2, 2))*8.}
        gauges = {"1": pd.DataFrame(dict(utc=pd.to_datetime(clock, unit="s", utc=True),
                                         elevation_m_IGLD85=[183.2, 184.2, 185.2]))}
        temp = pd.DataFrame(dict(station="1", epoch=clock, depth_m=1., temperature_c=7.))
        scores, _, _ = station_metrics(model, "O", gauges, temp, 0, 180, 60, 183.2)
        self.assertEqual(next(s for s in scores if s["variable"] == "absolute_elevation_m_IGLD85")["rmse"], 0.)
        no_datum, _, _ = station_metrics(model, "O", gauges, temp, 0, 180, 60, None)
        self.assertFalse(any(s["variable"] == "absolute_elevation_m_IGLD85" for s in no_datum))
        ref = dict(time=clock, ids=["1"], zeta=np.zeros((3, 1)), temp=np.ones((3, 2, 1))*5.,
                   sigma=np.array([[-.25], [-.75]]))
        comparisons = reference_metrics(model, ref, 0, 180, 60)
        self.assertEqual(comparisons[comparisons.variable == "elevation_m"].bias.iloc[0], 2.)
        self.assertTrue((comparisons[comparisons.variable == "temperature_degC"].bias == 3.).all())


if __name__ == "__main__":
    unittest.main()
