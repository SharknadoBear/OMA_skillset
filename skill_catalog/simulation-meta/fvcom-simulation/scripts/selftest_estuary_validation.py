"""Check the boundary, dry-gap and timing conditions that distinguish a salt-front comparison."""

import unittest

import numpy as np
from estuary_budget import open_edges
from estuary_gauges import crossings, interp_time, salinity_metrics, station_times
from estuary_sections import salt_front, sample_vertical, weights_at
from netCDF4 import Dataset


class EstuaryValidationTests(unittest.TestCase):
    def test_single_precision_station_clock(self):
        with Dataset("clock", "w", diskless=True, persist=False) as ds:
            ds.createDimension("time", 3)
            v = ds.createVariable("time", "f4", "time")
            v.units = "days since 1858-11-17 00:00:00"
            v[:] = 60752.0 + np.arange(3) * 360 / 86400
            ds.createVariable("iint", "i4", "time")[:] = [0, 300, 600]
            clock = station_times(ds, "2025-03-18T00:00:00Z", 0, 1.2)
            self.assertEqual(str(clock[1]), "2025-03-18T00:06:00.000")
            np.testing.assert_equal(np.diff(clock).astype(int), [360000, 360000])
            with self.assertRaises(ValueError):
                station_times(ds, "2025-03-17T00:00:00Z", 0, 1.2)

    def test_velocity_normal_and_event_gaps(self):
        xy = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        for cells in ([[0, 1, 2]], [[2, 1, 0]]):
            _, _, normal = open_edges(xy, np.array(cells), [0, 1])
            np.testing.assert_equal(normal, [[0.0, -2.0]])
        t = np.datetime64("2025-03-18") + np.array([0, 6, 18], dtype="timedelta64[m]")
        event = crossings(t, [2.0, 0.0, 3.0], 1.0, 540)
        self.assertEqual(
            event, [{"utc": "2025-03-18T00:03:00.000Z", "direction": "decreasing"}]
        )

    def test_connected_front(self):
        distance = np.arange(5.0)
        self.assertEqual(
            salt_front(distance, [4, 3, 0, 6, 5], 1), (1 + 2 / 3, "crossing")
        )
        self.assertEqual(
            salt_front(distance, [4, 3, np.nan, 6, 5], 1), (1.0, "lower_bound_gap")
        )
        self.assertEqual(
            salt_front(distance, [4, 3, 3, 6, 5], 1), (4.0, "lower_bound_end")
        )

    def test_physical_bounds(self):
        z = np.array([-1.0, -5.0, -9.0])
        s = np.array([5.0, 12.0, 24.0])
        self.assertEqual(sample_vertical(z, s, -9.5, -10.0, 0.0), 24.0)
        self.assertEqual(sample_vertical(z, s, -7.0, -10.0, 0.0), 18.0)
        self.assertTrue(np.isnan(sample_vertical(z, s, -10.5, -10.0, 0.0)))

    def test_native_triangle(self):
        xy = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        ci, _, weights = weights_at(
            xy, np.array([[0, 1, 2]]), np.array([[0.5, 0.5], [1.5, 1.5]])
        )
        np.testing.assert_equal(ci, [0, -1])
        np.testing.assert_allclose(weights[0], [0.5, 0.25, 0.25])
        self.assertTrue(np.isnan(weights[1]).all())

    def test_clock_interpolation_and_centering(self):
        start = np.datetime64("2025-03-18T00:00:00")

        def minutes(v):
            return start + np.asarray(v, dtype="timedelta64[m]")

        values = interp_time(
            minutes([0, 6, 18]), [1.0, 3.0, 4.0], minutes([-1, 3, 6, 12, 18]), 360
        )
        np.testing.assert_allclose(
            values, [np.nan, 2.0, 3.0, np.nan, 4.0], equal_nan=True
        )
        metrics = salinity_metrics([3.0, 4.0, np.nan], [1.0, 2.0, 3.0])
        self.assertEqual(
            (metrics["count"], metrics["bias"], metrics["centered_rmse"]), (2, 2.0, 0.0)
        )


if __name__ == "__main__":
    unittest.main()
