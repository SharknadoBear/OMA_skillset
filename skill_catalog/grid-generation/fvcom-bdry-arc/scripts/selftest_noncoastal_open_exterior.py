"""Real noncoastal producer fixtures and decision-gate regressions.

The finalizer test stubs only downstream materialization (covered by the owner
suite); it consumes the real generated contract, hashes and geographic map.
"""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import LineString, box

from fvcom_bdry_arc.open_exterior import build_open_exterior_contract, file_sha256
import finalize_open_exterior_decision as finalizer


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class NoncoastalOpenExteriorTests(unittest.TestCase):
    def fixture(self, root, domain, frame="reject-unintended", residual="solid-default"):
        outer = box(-158.1, 21.2, -158.0, 21.3)
        land = box(-158.07, 21.23, -158.03, 21.27)
        count = 0 if domain == "lake" else 1
        wet = outer if not count else outer.difference(land)
        if not count:
            land = outer.buffer(0.02).difference(outer)
        region = root / "region.json"
        offshore = root / "offshore.json"
        loop = root / "loop.json"
        package = root / "candidate.gpkg"
        coast = root / "coast.gpkg"
        write(region, {"domain_type": domain, "polygon_lonlat": list(outer.exterior.coords),
                       "expected_obc_count": count, "boundary_policy": "no_open_boundary" if not count else "island_closed_loop"})
        write(offshore, {"expected_obc_count": count})
        write(loop, {"failure_taxonomy": [], "qa": {"open_boundary_exterior_overlap_fraction": 1.0}})
        gpd.GeoDataFrame(geometry=[wet], crs=4326).to_file(package, layer="wet_domain", driver="GPKG")
        if count:
            gpd.GeoDataFrame(geometry=[LineString(outer.exterior.coords)], crs=4326).to_file(package, layer="open_boundary_arc", driver="GPKG")
        gpd.GeoDataFrame(geometry=[land], crs=4326).to_file(coast, layer="land_polygons", driver="GPKG")
        gpd.GeoDataFrame(geometry=[land.boundary], crs=4326).to_file(coast, layer="coastline_lines", driver="GPKG")
        before = [file_sha256(p) for p in (region, package, coast)]
        contract = build_open_exterior_contract(region, offshore, package, coast, loop, root / "evidence",
            {"inputs": {"coastline_source": "gshhs"}, "settings": {"target_resolution_m": 1000.0},
             "wet_domain": {"arc_land_intersection_length_m": 0.0},
             "coastline_source_coverage": {"downstream_eligible": True}},
            frame_clip_policy=frame, residual_boundary_policy=residual, frame_clip_tolerance_m=250.0,
            expected_obc_count=count)
        self.assertEqual(before, [file_sha256(p) for p in (region, package, coast)])
        return contract, package, loop, count

    def finalize_fixture(self, root, contract, package, loop, count, **kwargs):
        finalized = root / "final.gpkg"
        shutil.copyfile(package, finalized)
        manifest = root / "manifest.json"
        write(manifest, {"outputs": contract["outputs"], "settings": {"boundary_resolution_profile": "adaptive-coastal-v2"}})
        geometry = {"candidate_bdry_arc_package_gpkg": str(package), "bdry_arc_package_final_gpkg": str(finalized),
                    "sha256": file_sha256(finalized), "delivered_obc_count": count,
                    "chains": [{"obc_id": 0, "is_closed": True}] if count else [],
                    "solid_component_count": 0, "secondary_obc_count": 0}
        loop_result = {"final_status": "pass", "failure_taxonomy": [], "qa": {},
                       "outputs": {"model_boundary_loops_gpkg": str(finalized), "model_boundary_loop_manifest": str(loop)}}
        # The finalizer binds its canonical final-loop manifest path.
        canonical_loop = root / "model_boundary_loops_final/model_boundary_loop_manifest.json"
        canonical_loop.parent.mkdir(exist_ok=True)
        shutil.copyfile(loop, canonical_loop)
        with patch.object(finalizer, "_materialize_finalized_package", return_value=geometry), \
             patch.object(finalizer, "_rebuild_final_loops", return_value=loop_result), \
             patch.object(finalizer, "_rebuild_final_open_exterior", return_value=copy.deepcopy(contract)):
            return finalizer.finalize(manifest, "pass", "Synthetic regression fixture map and geometry inspected.",
                                      resume_adaptive=False, **kwargs)

    def test_default_noncoastal_v2_and_explicit_decision(self):
        for domain in ("island", "archipelago", "lake"):
            with self.subTest(domain=domain), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                contract, package, loop, count = self.fixture(root, domain)
                self.assertEqual(contract["schema_version"], "fvcom_open_exterior_contract_v2")
                self.assertFalse(contract["downstream_eligible"])
                self.assertEqual(contract["agent_decision"]["status"], "pending")
                self.assertIsNone(contract["residual_boundary_policy"]["default_role"])
                self.assertFalse(contract["residual_boundary_policy"]["role_assignment_enabled"])
                self.assertEqual(contract["residual_components"], [])
                self.assertEqual(contract["obc_geometry"]["delivered_count"], count)
                result = self.finalize_fixture(root, contract, package, loop, count)
                decided = result["open_exterior_contract"]
                self.assertEqual(decided["schema_version"], "fvcom_open_exterior_contract_v2")
                self.assertTrue(decided["downstream_eligible"])
                self.assertEqual(decided["obc_geometry"]["delivered_count"], count)
                self.assertEqual(decided["residual_components"], [])

    def test_failed_metrics_and_role_injection_remain_rejected(self):
        for failure in ("metrics", "roles"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                contract, package, loop, count = self.fixture(root, "island")
                kwargs = {}
                if failure == "metrics":
                    contract["hard_metrics"]["all_independent_metric_gates_pass"] = False
                    write(Path(contract["outputs"]["open_exterior_contract"]), contract)
                else:
                    kwargs["residual_roles"] = {0: "secondary_tidal_obc"}
                with self.assertRaises(ValueError):
                    self.finalize_fixture(root, contract, package, loop, count, **kwargs)

    def test_legacy_strict_and_report_only_routes_remain_v1(self):
        for frame, residual in (("report-only", "solid-default"), ("reject-unintended", "strict-reject")):
            with self.subTest(frame=frame, residual=residual), tempfile.TemporaryDirectory() as tmp:
                contract, *_ = self.fixture(Path(tmp), "island", frame, residual)
                self.assertEqual(contract["schema_version"], "fvcom_open_exterior_contract_v1")
                self.assertFalse(contract["downstream_eligible"])


if __name__ == "__main__":
    unittest.main()
