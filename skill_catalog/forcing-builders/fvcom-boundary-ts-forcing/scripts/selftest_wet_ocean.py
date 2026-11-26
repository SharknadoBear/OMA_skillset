"""Offline regressions for ocean-connected donor selection and the 2025 HYCOM archive layout."""

import json
import tempfile
import unittest
from pathlib import Path

import netCDF4 as nc
import numpy as np

from hycom_salinity import content_hash, load_hycom
from remap_glorys_sigma import check_startup_sigma, donor_bank, indices
from ts_core import sha256_file


class WetOceanTests(unittest.TestCase):
    def test_startup_sigma_order(self):
        with nc.Dataset("memory.nc", mode="w", diskless=True, persist=False) as ds:
            layers = np.array([-0.25, -0.75])
            levels = np.array([0, -0.5, -1])
            for key, n in (("node", 2), ("siglay", 2), ("siglev", 3)):
                ds.createDimension(key, n)
            ds.createVariable("siglay", "f4", ("siglay",))[:] = layers
            interfaces = ds.createVariable("siglev", "f4", ("siglev", "node"))
            interfaces[:] = np.broadcast_to(levels[:, None], (3, 2))
            ids = np.array([1, 2])
            check_startup_sigma(ds, ids, layers, levels)
            interfaces[:, 1] = levels[::-1]
            with self.assertRaisesRegex(ValueError, "sigma order"):
                check_startup_sigma(ds, ids, layers, levels)

    def test_common_vertical_wet_support(self):
        axis = np.arange(5) * 0.01
        valid = np.ones((2, 5, 5), bool)
        valid[0, :, 2] = False
        needed = np.ones((2, 2), bool)
        sites = np.array([[0.007, 0.004], [0.031, 0.012]])
        ids, w, dist, methods, basin = donor_bank(axis, axis, valid, sites, [0, 0], needed, 10.0)
        self.assertTrue((methods == [[0, 2], [0, 2]]).all())
        np.testing.assert_allclose(w.sum(axis=-1), 1.0)
        self.assertTrue(np.all(w >= 0))
        self.assertTrue(np.all(ids[w > 0] % 5 < 2))
        self.assertFalse(basin[1, 2, 4])
        np.testing.assert_allclose((ids[0, 0] % 5 * 0.01) @ w[0, 0], sites[0, 0])
        self.assertGreater(dist.max(), 1)
        with self.assertRaisesRegex(ValueError, "distant"):
            donor_bank(axis, axis, valid, sites, [0, 0], needed, 1.0)

    def test_renormalized_and_dry_seed(self):
        axis = np.arange(3) * 0.01
        valid = np.ones((1, 3, 3), bool)
        valid[0, 0, 1] = False
        args = (axis, axis, valid, np.array([[0.005, 0.005]]))
        ids, w, _, method, _ = donor_bank(*args, [0, 0], np.ones((1, 1), bool), 3.0)
        self.assertEqual(method[0, 0], 1)
        np.testing.assert_allclose(w[w > 0], np.ones(3) / 3)
        self.assertNotIn(1, ids[w > 0])
        with self.assertRaisesRegex(ValueError, "seed dry"):
            donor_bank(*args, [0.01, 0], np.ones((1, 1), bool), 3.0)
        with self.assertRaises(ValueError):
            indices(np.array([2, 1]), np.array([1.5]))

    def test_native_hycom_salinity_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            request = {
                "source": "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/s3z/2025",
                "variables": ["salinity"],
            }
            plan = dict(
                schema_version="hycom_download_plan_v1",
                request=request,
                request_hash=content_hash(request),
                source_inventory_hash="abc",
            )
            plan["plan_hash"] = content_hash(plan)
            (root / "download_plan.json").write_text(json.dumps(plan))
            path = root / "subset.nc"
            with nc.Dataset(path, "w") as ds:
                for key, values in {
                    "time": [0, 3, 6],
                    "depth": [0, 10],
                    "lat": [45, 46],
                    "lon": [235, 235.08],
                }.items():
                    ds.createDimension(key, len(values))
                    v = ds.createVariable(key, "f8", (key,))
                    v[:] = values
                ds["time"].units = "hours since 2025-03-18 00:00:00"
                ds["depth"].units = "m"
                ds["depth"].positive = "down"
                s = ds.createVariable("salinity", "i2", ("time", "depth", "lat", "lon"), fill_value=-30000)
                s.units = "psu"
                s.scale_factor = 0.001
                s.add_offset = 20.0
                s[:] = 31.5
                s[0, 0, 0, 0] = np.ma.masked
                ds.connector = "hycom-fetcher"
                ds.request_hash = plan["request_hash"]
                ds.field_type = "instantaneous"
            qa = dict(
                schema_version="hycom_health_v1", passed=True, sha256=sha256_file(path), input_name=path.name
            )
            (root / "health_check.json").write_text(json.dumps(qa))
            lon, _, z, clock, salt, _, _ = load_hycom(root)
            np.testing.assert_allclose(lon, [-125, -124.92])
            self.assertTrue(np.isnan(salt[0, 0, 0, 0]))
            self.assertAlmostEqual(salt[-1, -1, -1, -1], 31.5)
            self.assertEqual(z[0], 0.0)
            self.assertTrue(np.all(np.diff(clock) == 10_800_000))
            qa["sha256"] = "a" * 64
            (root / "health_check.json").write_text(json.dumps(qa))
            with self.assertRaisesRegex(ValueError, "changed"):
                load_hycom(root)


if __name__ == "__main__":
    unittest.main()
