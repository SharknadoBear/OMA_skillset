"""Exercise final CLI scoring without catalog memory; maps are synthetic fixtures."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_region_bpoly as runner
from region_bbox.geometry import RegionBPoly
from region_bbox.scoring import score_bpoly_quality, score_region_box


class UnknownScoringTests(unittest.TestCase):
    def run_cli(self, heuristic="unknown", mode="execute", *, usable=True, guard=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            geometry = RegionBPoly([[-88.6, 30], [-88.6, 31], [-87.8, 31], [-87.8, 30]], 180)
            ingredients = [{"id": "explicit_core", "type": "bbox", "role": "target_estuary",
                            "category": "target_region", "geometry": [-88.5, 30.1, -87.9, 30.9], "required": True}]
            if guard:
                ingredients.append({"id": "explicit_guard", "type": "bbox", "role": "offshore_boundary_exclusion",
                                    "geometry": [-88.4, 29.9, -88.2, 30.2], "required": False})
            request = {"request": "Mobile Bay tidal model", "target_region_features": {
                "schema_version": "target_region_features_v1", "domain_scale": "regional", "features": ingredients}}
            (root / "request.json").write_text(json.dumps(request))
            (root / "geometry.json").write_text(json.dumps(geometry.to_dict()))
            def fake_map(path, *args, **kwargs):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"synthetic map fixture; no scientific acceptance")
                return {"enabled": True, "geography_usable": usable, "source": "synthetic_test", "status": "ok"}
            args = ["run_region_bpoly.py", "--request-json", str(root / "request.json"),
                    "--input-region-json", str(root / "geometry.json"), "--run-dir", str(root / "out"),
                    "--name", "fixture", "--heuristic-mode", heuristic, "--mode", mode,
                    "--place-discovery", "off", "--basemap-provider", "none"]
            with patch.object(sys, "argv", args), patch.object(runner, "plot_region_map", fake_map), \
                    patch.object(runner, "side_focus_records", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            doc = json.loads((root / "out/region_bpoly.json").read_text())
            self.assertEqual(doc["region_bpoly"]["polygon_lonlat"], geometry.to_dict()["polygon_lonlat"])
            self.assertEqual(doc["final_status"], "review_pending")
            return doc

    def test_unknown_final_cli_has_no_catalog_warning(self):
        doc = self.run_cli()
        self.assertFalse(doc["place_memory_enabled"])
        self.assertEqual(doc["qa"]["bpoly_quality"]["canonical_region_key"], "unknown")
        self.assertNotIn("Horn Island", json.dumps(doc["delivery_warnings"]))
        self.assertNotIn("open_gate_landing_blocked_by_horn_island", json.dumps(doc["qa"]))

    def test_auto_test_mode_bypasses_catalog_scoring(self):
        doc = self.run_cli("auto", "test")
        self.assertEqual(doc["qa"]["bpoly_quality"]["canonical_region_key"], "unknown")

    def test_memory_and_auto_execute_retain_catalog_scoring(self):
        for mode in ("memory", "auto"):
            with self.subTest(mode=mode):
                doc = self.run_cli(mode)
                self.assertEqual(doc["qa"]["bpoly_quality"]["canonical_region_key"], "mobile_bay")
                self.assertIn("open_gate_landing_blocked_by_horn_island", json.dumps(doc["qa"]))

    def test_unknown_retains_explicit_obstruction_geometry(self):
        doc = self.run_cli(guard=True)
        self.assertIn("offshore_obstruction_guard_intersection", json.dumps(doc["qa"]))
        self.assertNotIn("open_gate_landing_blocked_by_horn_island", json.dumps(doc["qa"]))

    def test_unknown_retains_map_usability_check(self):
        doc = self.run_cli(usable=False)
        self.assertIn("background_geography_unavailable", json.dumps(doc["qa"]))

    def test_direct_explicit_coverage_and_offshore_checks_remain(self):
        region = RegionBPoly([[0, 0], [0, 1], [1, 1], [1, 0]], 180)
        outside = [{"id": "required_outside", "type": "point", "geometry": [2, 2], "required": True}]
        self.assertEqual(score_region_box(region, outside)["missing_required_ids"], ["required_outside"])
        result = score_bpoly_quality(region, [], {"request": "Mobile Bay", "_place_memory_enabled": False},
                                     "coastal", "coastal_arc_with_land_anchors", None,
                                     {"enabled": True, "geography_usable": True})
        self.assertIn("missing_offshore_point", [x["code"] for x in result["failure_taxonomy"]])
        self.assertEqual(result["canonical_region_key"], "unknown")


if __name__ == "__main__":
    unittest.main()
