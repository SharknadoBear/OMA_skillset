"""Offline integration and rejection tests for mesh-bound river forcing."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import netCDF4
import numpy as np

from build_barotropic_river import build, digest, second


class RiverTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mesh = self.root / "grid.dat"
        self.dep = self.root / "depth.dat"
        self.obc = self.root / "obc.dat"
        xyh = ["0 0 3", "1 0 3", "2 0 2", "0 1 3", "1 1 3", "2 1 4"]
        self.mesh.write_text("Node Number = 6\nCell Number = 4\n"
                             "1 1 2 5\n2 1 5 4\n3 2 3 6\n4 2 6 5\n" +
                             "".join(f"{i} {line}\n" for i, line in enumerate(xyh, 1)))
        self.dep.write_text("Node Number = 6\n" + "\n".join(xyh) + "\n")
        self.obc.write_text("OBC Node Number = 2\n1 1 1\n2 4 1\n")
        self.mapping = {"mesh_sha256": digest(self.mesh), "obc_sha256": digest(self.obc),
                        "wet_depth_floor_m": 0.1, "cutoff_description": "east closure",
                        "gauge_to_cutoff_km": 1, "nodes": [{"node_id": 3, "width_m": .5},
                                                           {"node_id": 6, "width_m": .5}]}
        self.rows = [[f"2025-03-29T0{h}:00:00Z", x, x * 0.028316846592]
                     for h, x in enumerate([100, 300, 200])]
        self.start, self.end = second(self.rows[0][0]), second(self.rows[-1][0])
        self.source = {"schema": "usgs_continuous_discharge_v1", "site": "USGS-14246900",
                       "parameter_code": "72137", "units": "m3/s", "original_units": "ft^3/s",
                       "approval_counts": {"Provisional": 3}, "records": 3, "interval_seconds": 3600}

    def run_builder(self):
        mapping_file = self.root / "mapping.json"
        mapping_file.write_text(json.dumps(self.mapping))
        table = self.root / "discharge.csv"
        with table.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_utc", "discharge_ft3s", "discharge_m3s"])
            writer.writerows(self.rows)
        raw = self.root / "response.json"
        raw.write_text("[]")
        self.source["series"] = {"path": table.name, "sha256": digest(table)}
        self.source["raw"] = {"path": raw.name, "sha256": digest(raw)}
        manifest = self.root / "source.json"
        manifest.write_text(json.dumps(self.source))
        return build(manifest, mapping_file, self.mesh, self.dep, self.obc,
                     self.start, self.end, 360, self.root / "product")

    def test_written_forcing(self):
        result = self.run_builder()
        self.assertEqual(result["river_number"], 2)
        self.assertAlmostEqual(result["nodes"][1]["weight"], 2 / 3)
        self.assertAlmostEqual(result["checks"]["source_volume_m3"], 450 * 3600 * 0.028316846592)
        self.assertAlmostEqual(result["checks"]["written_volume_m3"], result["checks"]["source_volume_m3"], places=2)
        nml = (self.root / "product/river.nml").read_text()
        self.assertEqual(nml.count("RIVER_VERTICAL_DISTRIBUTION = 'uniform'"), 2)
        with netCDF4.Dataset(self.root / "product/river.nc") as ds:
            self.assertEqual(ds.variables["time"].dtype, np.float64)
            np.testing.assert_allclose(ds["river_flux"][10], np.array([100, 200]) * 0.028316846592)
        with self.assertRaises(FileExistsError):
            self.run_builder()

    def test_obc_overlap(self):
        self.mapping["nodes"][0]["node_id"] = 1
        with self.assertRaisesRegex(ValueError, "solid-boundary"):
            self.run_builder()

    def test_dry_node(self):
        self.mapping["wet_depth_floor_m"] = 2.1
        with self.assertRaisesRegex(ValueError, "must be wet"):
            self.run_builder()

    def test_changed_mesh(self):
        self.mesh.write_text(self.mesh.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.run_builder()

    def test_bad_weights(self):
        self.mapping["nodes"][0]["width_m"] = -1
        with self.assertRaisesRegex(ValueError, "positive finite"):
            self.run_builder()

    def test_missing_coverage(self):
        self.end += 1
        with self.assertRaisesRegex(ValueError, "bracket"):
            self.run_builder()

    def test_source_gaps(self):
        self.rows.pop(1)
        self.source["records"] = 2
        with self.assertRaisesRegex(ValueError, "missing"):
            self.run_builder()

    def test_negative_flow(self):
        self.rows[1][2] *= -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.run_builder()

    def test_wrong_input_units(self):
        self.rows[1][2] = self.rows[1][1]
        with self.assertRaisesRegex(ValueError, "conversion"):
            self.run_builder()


if __name__ == "__main__":
    unittest.main()
