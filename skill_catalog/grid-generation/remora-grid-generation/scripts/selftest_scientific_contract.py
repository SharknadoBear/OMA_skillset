#!/usr/bin/env python3
"""Tests of scientific feature handoff and closure guards in the grid CLI."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest

import geopandas as gpd
import numpy as np
from shapely.geometry import box

import remora_grid as grid


class ScientificContractTests(unittest.TestCase):
    def request(self):
        return {
            "science": {"regime": "barotropic_tide", "rationale": "Synthetic tide test"},
            "design": {
                "coverage_rationale": "Protect synthetic essential water",
                "resolution_rationale": "Resolve synthetic inlet",
                "cell_budget_rationale": "Small numerical test",
                "boundary_intents": {
                    side: {"intent": "open", "rationale": "Synthetic boundary"}
                    for side in ("xlo", "xhi", "ylo", "yhi")
                },
            },
            "parameters": {
                "N": 10,
                "vertical_rationale": "Ten layers for a barotropic test",
            },
            "protected_wet_features": [
                {"name": "inlet", "point_lonlat": [-75.03, 38.02], "max_snap_m": 10.}
            ],
            "bathymetry": {"path": "synthetic.nc", "positive": "up"},
        }

    def region(self):
        return {
            "id": "outer",
            "features": [
                {
                    "name": "inlet",
                    "point_lonlat": [-75.03, 38.02],
                    "priority": "essential",
                    "feature_type": "water",
                },
                {
                    "name": "lagoon",
                    "point_lonlat": [-75.01, 38.04],
                    "priority": "optional",
                    "feature_type": "water",
                },
            ],
        }

    def test_essential_water_requires_named_canonical_anchor(self):
        req, region = self.request(), self.region()
        grid.require_scientific_features(req, region)
        for protection in (
            [],
            [{"name": "other", "point_lonlat": [-75.03, 38.02]}],
            [{"name": "inlet", "point_lonlat": [-75.02, 38.02]}],
            [{"name": "inlet", "point_lonlat": [np.nan, 38.02]}],
            req["protected_wet_features"] * 2,
        ):
            candidate = deepcopy(req)
            candidate["protected_wet_features"] = protection
            with self.subTest(protection=protection), self.assertRaises(ValueError):
                grid.require_scientific_features(candidate, region)

    def test_land_and_coverage_features_do_not_become_wet_anchors(self):
        req = self.request()
        req["protected_wet_features"] = []
        region = {
            "features": [
                {"name": "island", "feature_type": "land",
                 "point_lonlat": [-75.03, 38.02], "priority": "essential"},
                {"name": "mission", "bbox_wsen": [-75.04, 38.01, -75.01, 38.04],
                 "priority": "essential"},
                {"name": "optional water", "point_lonlat": [-75.01, 38.04],
                 "priority": "optional"},
            ]
        }
        grid.require_scientific_features(req, region)

    def test_essential_water_bbox_anchor_must_be_inside(self):
        req = self.request()
        region = {"features": [{
            "name": "inlet", "feature_type": "water", "priority": "essential",
            "bbox_wsen": [-75.04, 38.01, -75.02, 38.03],
        }]}
        grid.require_scientific_features(req, region)
        req["protected_wet_features"][0]["point_lonlat"] = [-75.01, 38.02]
        with self.assertRaisesRegex(ValueError, "outside"):
            grid.require_scientific_features(req, region)

    def test_legacy_defaults_remain_and_new_tide_n40_is_rejected(self):
        legacy = grid.parameters({})
        self.assertEqual(legacy["N"], 40)
        self.assertNotIn("rx1_max", legacy)
        grid.require_scientific_features({}, self.region())
        req = self.request()
        self.assertEqual(grid.parameters(req)["N"], 10)
        req["parameters"]["N"] = 40
        with self.assertRaisesRegex(ValueError, "8"):
            grid.parameters(req)
        req["science"]["regime"] = "baroclinic"
        self.assertEqual(grid.parameters(req)["N"], 40)
        req["design"]["boundary_intents"].pop("xlo")
        with self.assertRaisesRegex(ValueError, "all four"):
            grid.parameters(req)

    def rejection_from_build(self, request, region, extra_regions=()):
        """Run the real build closure/anchor logic with only source IO stubbed."""
        lon, lat = np.meshgrid(np.linspace(-75.06, -75.00, 7),
                              np.linspace(38., 38.04, 5))
        arrays = {"lon_rho": lon, "lat_rho": lat,
                  "pm": np.full(lon.shape, .001), "pn": np.full(lon.shape, .001)}
        land = gpd.GeoDataFrame(geometry=[box(-76., 37., -75.9, 37.1)], crs=4326)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fitdoc = {"request": {"path": "request.json"}, "geometry": {}}
            loaded = (request, grid.parameters(request), root / "region.json",
                      {"source": {"path": "land.geojson"}},
                      {"regions": [region, *extra_regions]}, region)
            with patch.object(grid, "check_fit", return_value=fitdoc), \
                 patch.object(grid, "load_request", return_value=loaded), \
                 patch.object(grid, "coordinates", return_value=arrays), \
                 patch.object(grid, "land_frame", return_value=land), \
                 patch.object(grid, "sample_netcdf", return_value=np.full(lon.shape, -10.)):
                with self.assertRaises(ValueError) as caught:
                    grid.build(root / "fit.json", root / "grid")
                self.assertFalse((root / "grid").exists())
            return str(caught.exception)

    def closure(self, name):
        return {
            "action": "close_optional",
            "feature_name": name,
            # This explicitly attempts to close the original inlet anchor.
            "vertices_lonlat": [[-75.035, 38.015], [-75.025, 38.015],
                               [-75.025, 38.025], [-75.035, 38.025]],
            "rationale": "Deliberately invalid synthetic closure",
            "evidence": [{"source": "synthetic fixture", "finding": "Guard test"}],
        }

    def test_optional_closure_cannot_dry_essential_original_anchor(self):
        req, region = self.request(), self.region()
        req["mask_adjustments"] = [self.closure("lagoon")]
        error = self.rejection_from_build(req, region)
        self.assertIn("dry or disconnected", error)

    def test_other_region_optional_name_cannot_override_essential(self):
        req, region = self.request(), self.region()
        req["mask_adjustments"] = [self.closure("inlet")]
        other = {"id": "different_root", "features": [
            {"name": "inlet", "priority": "optional", "point_lonlat": [-74., 37.]}
        ]}
        error = self.rejection_from_build(req, region, [other])
        self.assertIn("declared optional", error)

    def test_build_rejects_missing_essential_before_source_processing(self):
        req = self.request()
        req["protected_wet_features"] = []
        req["mask_adjustments"] = [self.closure("lagoon")]
        self.assertIn("named protected anchor", self.rejection_from_build(req, self.region()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
