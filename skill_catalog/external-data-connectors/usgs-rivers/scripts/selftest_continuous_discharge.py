"""Offline tests for continuous-discharge identity, cadence and unit conversion."""
import unittest
from datetime import timedelta

from fetch_continuous_discharge import observations, utc


class SeriesTest(unittest.TestCase):
    def setUp(self):
        self.start, self.end = utc("2025-03-28T00:00:00Z"), utc("2025-03-28T01:00:00Z")
        self.data = [{"properties": {"monitoring_location_id": "USGS-14246900",
                      "parameter_code": "72137", "time_series_id": "one",
                      "statistic_id": "00011", "approval_status": "Provisional",
                      "qualifier": None, "value": "294000", "unit_of_measure": "ft^3/s",
                      "time": instant.isoformat()}} for instant in (self.end, self.start)]

    def parse(self):
        return observations(self.data, "USGS-14246900", "72137", self.start,
                            self.end, timedelta(hours=1))

    def test_sorted_si_series(self):
        rows, identity = self.parse()
        self.assertEqual(identity, ("one", "00011"))
        self.assertEqual(rows[0][0], self.start)
        self.assertAlmostEqual(rows[0][2], 294000 * 0.028316846592)
        self.assertEqual(rows[0][3], "Provisional")

    def test_invalid_identity(self):
        self.data[0]["properties"]["parameter_code"] = "00060"
        with self.assertRaisesRegex(ValueError, "site or parameter"):
            self.parse()

    def test_conflicting_series(self):
        self.data[0]["properties"]["time_series_id"] = "two"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.parse()

    def test_wrong_units(self):
        self.data[0]["properties"]["unit_of_measure"] = "ft"
        with self.assertRaisesRegex(ValueError, "unit"):
            self.parse()

    def test_duplicate(self):
        self.data.append(self.data[0])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self.parse()

    def test_truncated_coverage(self):
        self.data.pop()
        with self.assertRaisesRegex(ValueError, "endpoints"):
            self.parse()


if __name__ == "__main__":
    unittest.main()
