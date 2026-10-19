"""Offline discovery regressions; catalog presence never certifies period data."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import coops_currents as api


class DiscoveryCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mesh = Path(self.tmp.name) / "mesh.2dm"
        self.mesh.write_text("MESH2D\nND 1 0 0 5\nND 2 1 0 5\nND 3 0 1 5\nE3T 1 1 2 3 1\n")
        def station(sid, x=.2, y=.2): return {"id": sid, "lng": x, "lat": y, "name": sid}
        self.catalogs = {"waterlevels": [station("dual")], "tidepredictions": [
            station("dual", .8, .8), station("prediction"), station("outside", .8, .8), station("far", 2, 2)],
            "currents": [station("current"), station("outside_current", .8, .8), station("up"), station("old"), station("unknown")],
            "historiccurrents": [station("current"), station("historic"), station("prediction")]}

    def discover(self, policy="strict_inside"):
        def meta(sid):
            dates = {} if sid == "unknown" else {"first_good_data": "2020-01-01", "last_good_data": "2021-01-01" if sid == "old" else "2026-01-01"}
            return {"deployments": {"orientation": "up" if sid == "up" else "down", **dates}, "bins": [], "units": "meters"}
        with patch.object(api, "metadata_stations", side_effect=lambda kind: self.catalogs[kind]) as calls, \
                patch.object(api, "current_metadata", side_effect=meta):
            result = api.discover(self.mesh, "EPSG:4326", "2025-04-01T00:00:00Z", "2025-05-01T00:00:00Z", policy)
        return result, [c.args[0] for c in calls.call_args_list]

    def test_prediction_catalog_and_first_coordinate_provenance(self):
        result, calls = self.discover()
        self.assertEqual(calls, ["waterlevels", "tidepredictions", "currents", "historiccurrents"])
        rows = {(r["id"], r["role"]): r for r in result["stations"]}
        water = rows[("prediction", "water_level")]
        self.assertTrue(water["eligible"])
        self.assertEqual(water["source_data_eligibility"], "pending_period_scalar_data_checks")
        self.assertNotIn("observation_eligible", water)
        dual = rows[("dual", "water_level")]
        self.assertEqual(dual["longitude"], .2)
        self.assertEqual([s["catalog_type"] for s in dual["inventory_sources"]], ["waterlevels", "tidepredictions"])
        self.assertEqual(len(result["stations"]), len(rows))
        self.assertNotIn(("far", "water_level"), rows)
        self.assertIn(("prediction", "current"), rows)

    def test_proxy_preserves_containment_and_current_gates(self):
        strict, _ = self.discover()
        proxy, _ = self.discover("containing_cell_or_nearest_wet_cell")
        a = {(r["id"], r["role"]): r for r in strict["stations"]}
        b = {(r["id"], r["role"]): r for r in proxy["stations"]}
        self.assertFalse(a[("outside", "water_level")]["eligible"])
        outside = b[("outside", "water_level")]
        self.assertTrue(outside["eligible"])
        self.assertFalse(outside["inside_wet_mesh"])
        self.assertEqual(outside["mapping_required"], "nearest_wet_cell_proxy")
        for key in a:
            if key[1] == "current": self.assertEqual(a[key], b[key])
        for sid in ("outside_current", "up", "old", "unknown"):
            self.assertFalse(b[(sid, "current")]["eligible"])
        self.assertTrue(b[("historic", "current")]["eligible"])
        self.assertEqual([s["catalog_type"] for s in b[("current", "current")]["inventory_sources"]], ["currents", "historiccurrents"])

    def test_unavailable_prediction_catalog_is_not_absence(self):
        def fetch(kind):
            if kind == "tidepredictions": raise RuntimeError("catalog unavailable")
            return self.catalogs[kind]
        with patch.object(api, "metadata_stations", side_effect=fetch), \
                patch.object(api, "current_metadata", side_effect=AssertionError("unexpected current lookup")), \
                self.assertRaisesRegex(RuntimeError, "catalog unavailable"):
            api.discover(self.mesh, "EPSG:4326", "2025-04-01T00:00:00Z", "2025-05-01T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
