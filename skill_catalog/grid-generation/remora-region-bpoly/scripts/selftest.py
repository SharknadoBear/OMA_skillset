#!/usr/bin/env python3
"""Behavioral hierarchy, geometry and stale-evidence checks; no network."""

import copy
import argparse
import contextlib
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
    validate_science,
    read_json,
)

ARTIFACT_ROOT = None


@contextlib.contextmanager
def artifact_directory(name):
    if ARTIFACT_ROOT is None:
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)
    else:
        root = ARTIFACT_ROOT / name
        root.mkdir(parents=True, exist_ok=False)
        yield root


def fixture(root, **extras):
    land = root / "land.geojson"
    gpd.GeoDataFrame(geometry=[box(-76, 38, -75.7, 41)], crs=4326).to_file(
        land, driver="GeoJSON"
    )
    write_json(root / "coast_manifest.json", {"source": "synthetic land", "version": 1})
    request = {
        "name": "fixture",
        "objective": "test",
        "coastline": {
            "path": "land.geojson",
            "bbox_wsen": [-76, 38, -74, 41],
            "manifest": "coast_manifest.json",
        },
        "regions": [region()],
        **extras,
    }
    write_json(root / "request.json", request)
    return build(root / "request.json", root / "out"), request


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

    def test_priority_and_disposition(self):
        r = region()
        r["features"] = [
            {"name": "entrance", "point_lonlat": [-75, 39], "priority": "essential"},
            {
                "name": "small peripheral embayment",
                "point_lonlat": [-74.3, 39],
                "priority": "optional",
                "disposition": "excluded",
                "rationale": "Outside the transport pathway in this synthetic task",
                "evidence": [{
                    "source": "synthetic morphology",
                    "finding": "Peripheral closed embayment outside the essential connection",
                }],
            },
            {
                "name": "later passage",
                "point_lonlat": [-75, 39.4],
                "priority": "future_refinement",
                "rationale": "Future nesting interest; no present resolution claim",
            },
        ]
        validate_regions([r])
        for mutation in [
            {"priority": "unrecognized"},
            {"priority": "essential", "disposition": "excluded", "rationale": "Must fail"},
            {"priority": "optional", "disposition": "excluded", "rationale": ""},
            {"priority": "optional", "disposition": "excluded", "rationale": "No evidence"},
            {"priority": "future_refinement", "disposition": "included"},
        ]:
            bad = copy.deepcopy(r)
            bad["features"][0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_regions([bad])

    def test_optional_included_feature_requires_coverage(self):
        r = region()
        r["features"] = [
            {"name": "outlying channel", "point_lonlat": [-70, 40], "priority": "optional"}
        ]
        with self.assertRaises(ValueError):
            validate_regions([r])
        r["features"][0].update(
            disposition="simplified", rationale="Coarse portrayal",
            evidence=[{"source": "synthetic map", "finding": "Optional secondary channel"}],
        )
        with self.assertRaises(ValueError):
            validate_regions([r])

    def test_legacy_features_remain_essential(self):
        r = region()
        r["required_features"] = [{"name": "entrance", "point_lonlat": [-75, 39]}]
        validate_regions([r])
        r["features"] = [{"name": "entrance", "point_lonlat": [-75, 39.1]}]
        with self.assertRaises(ValueError):
            validate_regions([r])

    def test_feature_geometry_not_ambiguous(self):
        r = region()
        r["features"] = [{
            "name": "ambiguous",
            "point_lonlat": [-75, 39],
            "bbox_wsen": [-75.1, 38.9, -74.9, 39.1],
        }]
        with self.assertRaises(ValueError):
            validate_regions([r])

    def test_water_land_and_coverage_feature_types(self):
        r = region()
        r["features"] = [
            {"name": "wet entrance", "point_lonlat": [-75, 39], "feature_type": "water"},
            {"name": "real island", "point_lonlat": [-75.7, 39], "feature_type": "land"},
            {"name": "exchange area", "bbox_wsen": [-75.4, 39, -75, 39.4],
             "feature_type": "coverage"},
        ]
        validate_regions([r])
        r["features"][0]["feature_type"] = "anything"
        with self.assertRaises(ValueError):
            validate_regions([r])

    def test_known_role_parent_semantics_and_legacy_role(self):
        outer = region()
        outer["role"] = "outer_domain"
        child = region("bay", "outer", (-75.5, 39, -75, 39.5))
        child["role"] = "nesting_interest"
        validate_regions([outer, child])
        child["role"] = "refinement_interest"
        validate_regions([outer, child])
        child["parent_id"] = None
        with self.assertRaises(ValueError):
            validate_regions([outer, child])
        child["parent_id"] = "outer"
        child["role"] = "outer_domain"
        with self.assertRaises(ValueError):
            validate_regions([outer, child])
        child["role"] = "legacy_special_role"
        validate_regions([outer, child])

    def test_multi_open_side_boundary_intents(self):
        r = region()
        r["boundary_intents"] = [
            {"side_index": i, "intent": "open", "rationale": "Exchange with shelf waters"}
            for i in (0, 1, 2)
        ]
        validate_regions([r])
        r["boundary_intents"].append(copy.deepcopy(r["boundary_intents"][0]))
        with self.assertRaises(ValueError):
            validate_regions([r])
        r["boundary_intents"] = [{"side_index": 4, "intent": "open", "rationale": "bad side"}]
        with self.assertRaises(ValueError):
            validate_regions([r])

    def test_scientific_purpose_enum_is_optional(self):
        validate_science({})
        for purpose in ("barotropic_tide", "baroclinic"):
            validate_science({"modeling_purpose": purpose})
        with self.assertRaises(ValueError):
            validate_science({"modeling_purpose": "tide_and_everything"})

    def test_science_and_feature_intent_retention(self):
        with artifact_directory("science_retention") as root:
            r = region()
            r["role"] = "outer_domain"
            r["features"] = [{
                "name": "main passage", "point_lonlat": [-75, 39],
                "priority": "essential", "rationale": "Transport connection",
            }]
            r["boundary_intents"] = [
                {"side_index": 1, "intent": "open", "rationale": "Shelf exchange"},
                {"side_index": 2, "intent": "open", "rationale": "Alongshore exchange"},
            ]
            delivery, req = fixture(
                root, regions=[r], modeling_purpose="barotropic_tide",
                science_rationale="Resolve depth-integrated tidal exchange",
            )
            d, plan = validate_delivery(delivery)
            self.assertEqual(plan["regions"], req["regions"])
            self.assertEqual(plan["modeling_purpose"], req["modeling_purpose"])
            self.assertEqual(plan["science_rationale"], req["science_rationale"])
            geojson = read_json(root / "out" / d["polygons"]["path"])
            self.assertEqual(geojson["features"][0]["properties"]["features"], r["features"])
            self.assertEqual(len(d["maps"]), 2)

    def test_changed_source_manifest_invalidates_delivery(self):
        with artifact_directory("source_integrity") as root:
            delivery, _ = fixture(root)
            validate_delivery(delivery)
            write_json(root / "coast_manifest.json", {"source": "different coastline"})
            with self.assertRaisesRegex(ValueError, "Changed evidence"):
                validate_delivery(delivery)

    def test_changed_coastline_invalidates_delivery(self):
        with artifact_directory("coastline_integrity") as root:
            delivery, _ = fixture(root)
            validate_delivery(delivery)
            (root / "land.geojson").write_text("changed source", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Changed evidence"):
                validate_delivery(delivery)

    def test_changed_request_invalidates_delivery(self):
        with artifact_directory("request_integrity") as root:
            delivery, req = fixture(root)
            req["modeling_purpose"] = "baroclinic"
            write_json(root / "request.json", req)
            with self.assertRaisesRegex(ValueError, "Changed evidence"):
                validate_delivery(delivery)

    def test_delivery_and_changed_evidence(self):
        with artifact_directory("map_integrity") as root:
            delivery, _ = fixture(root)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path)
    args, remaining = parser.parse_known_args()
    ARTIFACT_ROOT = args.artifact_dir
    unittest.main(argv=[__file__, *remaining], verbosity=2)
