#!/usr/bin/env python3
"""Offline public-interface tests for hrrr-fetcher."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from hrrr_core import normalize_request, product_catalog
from hrrr_fetcher import health_hrrr_run, snapshot_hrrr


class PublicInterfaceTests(unittest.TestCase):
    def test_catalog_contract(self) -> None:
        catalog = product_catalog()
        self.assertEqual(catalog["default_provider_order"], ["aws", "gcp", "azure", "nomads"])
        self.assertFalse(catalog["bufr"]["decoded"])
        self.assertEqual(catalog["families"]["wrfprs"]["pressure_levels"], 39)
        self.assertEqual(catalog["families"]["wrfnat"]["hybrid_levels"], 50)

    def test_provider_override_and_longitude_normalization(self) -> None:
        request = normalize_request({
            "domain": "conus",
            "start": "2024-01-15T12:00:00Z",
            "end": "2024-01-15T12:00:00Z",
            "products": ["surface_pressure"],
            "bbox": [283, 36, 285, 40],
            "longitude_convention": "0_360",
            "provider_override": "gcp",
        })
        self.assertEqual(request["bbox"], [-77.0, 36.0, -75.0, 40.0])
        self.assertEqual(request["provider_order"], ["gcp"])

    def test_health_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            report = health_hrrr_run(Path(value))
            self.assertFalse(report["passed"])
            self.assertTrue((Path(value) / "health_check.json").exists())

    def test_snapshot_rejects_multiple_analysis_times_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            with self.assertRaisesRegex(ValueError, "start == end"):
                snapshot_hrrr({
                    "domain": "conus",
                    "start": "2024-01-15T12:00:00Z",
                    "end": "2024-01-15T13:00:00Z",
                    "products": ["surface_pressure"],
                }, value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
