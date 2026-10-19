"""Regression for island/lake pass deliveries missing their native S1 freeze."""
from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from region_bbox.delivery import write_standard_delivery
from region_bbox.geometry import RegionBPoly
from region_bbox.io import canonical_sha256, file_sha256, write_json


class NoncoastalFreezeTests(unittest.TestCase):
    def fixture(self, root: Path, domain: str) -> dict:
        region = RegionBPoly([[-160, 19], [-160, 22], [-154, 22], [-154, 19]], 90)
        (root / "intermediate").mkdir()
        (root / "region_bpoly_final_map.png").write_bytes(b"retained map fixture")
        write_json(root / "offshore_boundary_artifacts.json", {"domain_type": domain})
        return {
            "name": "fixture", "domain_type": domain, "final_status": "pass",
            "region_bpoly": region.to_dict(), "polygon_lonlat": region.polygon_lonlat(),
            "open_boundary_reference": None if domain == "lake" else {"side_index": 0},
            "offshore_boundary_artifacts": {"selected_side_index": None if domain == "lake" else 0},
            "final_map_path": str(root / "region_bpoly_final_map.png"),
            "final_map_basemap": {"geography_usable": True},
            "scientific_review": None, "delivery_warnings": ["retained diagnostic warning"],
            "qa": {"ingredient_coverage": {"all_required_inside": True}},
            "target_region_features": {"features": []},
        }

    def test_island_lake_freeze_survives_execute_cleanup_without_science_claim(self):
        for domain in ("island", "lake"):
            with self.subTest(domain=domain), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                final = self.fixture(root, domain)
                geometry = copy.deepcopy(final["region_bpoly"])
                write_standard_delivery(root, "fixture", final)
                freeze = final["accepted_s1_geometry_freeze"]
                self.assertEqual(final["region_bpoly"], geometry)
                self.assertEqual(freeze["region_bpoly_sha256"], canonical_sha256(geometry))
                self.assertFalse(freeze["downstream_geometry_repair_permitted"])
                self.assertIsNone(freeze["scientific_review_json"])
                self.assertIsNone(freeze["scientific_review_json_sha256"])
                self.assertIsNone(final["scientific_review"])
                self.assertEqual(final["delivery_warnings"], ["retained diagnostic warning"])
                # run_region_bpoly execute mode removes this directory after packaging.
                shutil.rmtree(root / "intermediate")
                for key in ("candidate_json", "final_map"):
                    path = Path(freeze[key])
                    self.assertTrue(path.is_file())
                    self.assertEqual(file_sha256(path), freeze[key + "_sha256"])
                manifest = json.loads((root / "region_bpoly_manifest.json").read_text())
                candidates = [r for r in manifest["files"] if r["role"] == "geometry_candidate"]
                self.assertEqual(len(candidates), 1)
                self.assertTrue(candidates[0]["required_for_state"])
                self.assertTrue(manifest["delivery_ready"])
                self.assertEqual(freeze["offshore_side_index"], None if domain == "lake" else 0)
                before = Path(freeze["candidate_json"]).read_bytes()
                write_standard_delivery(root, "fixture", final)
                self.assertEqual(Path(freeze["candidate_json"]).read_bytes(), before)

    def test_existing_frozen_geometry_cannot_be_silently_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = self.fixture(root, "island")
            write_standard_delivery(root, "fixture", final)
            before = (root / "region_bpoly_geometry_candidate.json").read_bytes()
            final["region_bpoly"]["polygon_lonlat"][0][0] += 0.25
            with self.assertRaisesRegex(ValueError, "new run directory"):
                write_standard_delivery(root, "fixture", final)
            self.assertEqual((root / "region_bpoly_geometry_candidate.json").read_bytes(), before)

    def test_coastal_and_nonpass_do_not_get_automatic_freeze(self):
        for domain, status in (("coastal", "review_pending"), ("coastal", "pass"), ("island", "blocked")):
            with self.subTest(domain=domain, status=status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                final = self.fixture(root, domain)
                final["final_status"] = status
                write_standard_delivery(root, "fixture", final)
                self.assertNotIn("accepted_s1_geometry_freeze", final)


if __name__ == "__main__":
    unittest.main()
