"""Test native vector contrasts, wet support, NOAA support and matched forcing identities."""

import json
import tempfile
import unittest
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

from factorial_fields import MEMBERS, effects, spatial
from factorial_validation import common_scores
from run_case_validation import sha256


class FourWayTests(unittest.TestCase):
    def test_effects_and_common_observations(self):
        np.testing.assert_allclose(effects([[5], [7], [8], [14]]), [[3], [7], [2], [6], [4]])
        np.testing.assert_allclose(effects([[5], [7], [8], [10]])[-1], 0)
        times = pd.date_range("2025-04-01", periods=6, freq="6min", tz="UTC")
        baseline = np.arange(len(times), dtype=float)
        data = pd.DataFrame({k: baseline + i for i, k in enumerate(MEMBERS)}, index=times)
        data["observed"] = [0, 1, np.nan, 3, np.nan, 5]
        data["predicted"] = [-1, 0, 1, np.nan, 3, 4]
        data.loc[times[4], "combined"] = np.nan
        result = common_scores(data, {"station": "one"})
        for member in MEMBERS:
            rows = {r["reference"]: r for r in result if r["member"] == member}
            self.assertEqual(
                [rows[k]["samples"] for k in ("predicted", "observed", "residual")], [4, 4, 3]
            )
        data["observed"] = np.nan
        self.assertEqual({r["reference"] for r in common_scores(data, {})}, {"predicted"})

    def test_four_native_wet_masks_and_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = pd.Timestamp("2025-04-01", tz="UTC")
            cases = {}
            tri = np.array([[1, 2, 3], [2, 4, 3]], dtype=int)
            for k, amplitude in zip(MEMBERS, [2, 5, 7, 14]):
                path = root / (k + ".nc")
                with nc.Dataset(path, "w") as ds:
                    for name, n in (("time", 2), ("node", 4), ("nele", 2), ("three", 3)):
                        ds.createDimension(name, n)

                    def put(name, dims, vals, dtype="f8"):
                        ds.createVariable(name, dtype, dims)[:] = vals

                    for name, vals in (
                        ("x", [0, 1000, 0, 1000]),
                        ("y", [0, 0, 1000, 1000]),
                        ("h", [5] * 4),
                    ):
                        put(name, ("node",), vals)
                    put("nv", ("three", "nele"), tri.T, "i4")
                    put("iint", ("time",), [0, 10800], "i4")
                    put("zeta", ("time", "node"), np.full((2, 4), amplitude))
                    put("ua", ("time", "nele"), np.full((2, 2), -amplitude))
                    put("va", ("time", "nele"), np.full((2, 2), 2 * amplitude))
                    mask = np.ones((2, 4))
                    cells = np.ones((2, 2))
                    if k == "river":
                        mask[0, 3] = cells[0, 1] = 0
                    put("wet_nodes", ("time", "node"), mask, "i4")
                    put("wet_cells", ("time", "nele"), cells, "i4")
                audit = root / (k + ".json")
                audit.write_text(
                    json.dumps({"full_grid_netcdf": [{"path": str(path), "sha256": sha256(path)}]})
                )
                nml = root / (k + ".nml")
                nml.write_text(
                    "EXTSTEP_SECONDS=1\nISPLIT=1\nSTART_DATE='2025-04-01'\nEND_DATE='2025-04-02'\n"
                )
                mapping = root / "stations.json"
                mapping.write_text(
                    json.dumps({"stations": [{"station_id": "A", "role": "water_level", "cell_id": 2}]})
                )
                cases[k] = dict(
                    start=start,
                    end=start + pd.Timedelta(hours=6),
                    run_start=start,
                    attempt=root,
                    audit_path=audit,
                    namelist=nml,
                    mapping_path=mapping,
                )
            report = spatial(cases, root)
            self.assertEqual(report["timestamps"], 2)
            with nc.Dataset(root / "native_contrasts.nc") as ds:
                self.assertEqual(ds["zeta"][1, 4, 0], 4)
                self.assertEqual(ds["ua"][1, 4, 0], -4)
                self.assertEqual(ds["va"][1, 4, 0], 8)
                self.assertTrue(np.ma.getmaskarray(ds["zeta"][0, :, 3]).all())
                np.testing.assert_array_equal(ds["wet_samples_nele"][:], [2, 1])
                np.testing.assert_allclose(ds["mean_zeta"][4, :], 4)
            wetness = pd.read_csv(root / "gauge_wetness.csv")
            self.assertEqual(wetness.loc[wetness.member == "river", "wet_cell"].tolist(), [0, 1])
            with self.assertRaisesRegex(ValueError, "unmatched|Missing"):
                cases["river"]["end"] = start + pd.Timedelta(hours=3)
                spatial(cases, root)


if __name__ == "__main__":
    unittest.main()
