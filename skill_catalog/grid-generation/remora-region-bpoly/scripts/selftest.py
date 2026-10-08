#!/usr/bin/env python3
"""Behavioral hierarchy, geometry and stale-evidence checks; no network."""

import copy
import tempfile
import unittest
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box
from remora_regions import (
    polygon,
    validate_regions,
    write_json,
    build,
    review,
    validate_delivery,
)


def region(rid="outer", parent=None, bounds=(-75.8, 38.5, -74.5, 40.0)):
    w, s, e, n = bounds
    return {
        "id": rid,
        "parent_id": parent,
        "purpose": "test geographic coverage",
        "vertices_lonlat": [[w, s], [e, s], [e, n], [w, n]],
        "required_features": [],
    }


class Regions(unittest.TestCase):
    def test_invalid_shapes(self):
        for vertices in [
            [[0, 0], [1, 1], [0, 1], [1, 0]],
            [[0, 0], [1, 0], [2, 0], [3, 0]],
            [[179, 0], [-179, 0], [-179, 1], [179, 1]],
            [[0, 0], [1, 0], [1, float("nan")], [0, 1]],
        ]:
            with self.subTest(vertices=vertices), self.assertRaises(ValueError):
                polygon(vertices)

    def test_child_coverage(self):
        validate_regions([region(), region("bay", "outer", (-75.5, 39, -75, 39.5))])

    def test_missing_parent(self):
        with self.assertRaises(ValueError):
            validate_regions([region("bay", "absent")])

    def test_cycle(self):
        with self.assertRaises(ValueError):
            validate_regions([region("a", "b"), region("b", "a")])

    def test_child_outside(self):
        with self.assertRaises(ValueError):
            validate_regions([region(), region("bay", "outer", (-76, 39, -75, 39.5))])

    def test_required_feature(self):
        r = region()
        r["required_features"] = [{"name": "required bay", "point_lonlat": [-70, 40]}]
        with self.assertRaises(ValueError):
            validate_regions([r])

    def test_unique_ids(self):
        with self.assertRaises(ValueError):
            validate_regions([region(), region()])

    def test_delivery_and_changed_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            land = root / "land.geojson"
            gpd.GeoDataFrame(geometry=[box(-76, 38, -75.7, 41)], crs=4326).to_file(
                land, driver="GeoJSON"
            )
            request = {
                "name": "fixture",
                "objective": "test",
                "coastline": {"path": "land.geojson", "bbox_wsen": [-76, 38, -74, 41]},
                "regions": [region()],
            }
            write_json(root / "request.json", request)
            delivery = build(root / "request.json", root / "out")
            with self.assertRaises(FileNotFoundError):
                validate_delivery(delivery, True)
            review(
                delivery,
                "accepted",
                "Automated fixture exercising review binding; not a scientific case review",
            )
            validate_delivery(delivery, True)
            (root / "out/maps/region_overview.png").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                validate_delivery(delivery, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
