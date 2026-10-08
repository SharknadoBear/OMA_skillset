"""Offline behavior tests. Synthetic data do not establish authenticated live readiness."""
import contextlib
import copy
import datetime as dt
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import xarray as xr
import glorys_core as g


class FakeBackend:
    def __init__(self):
        self.calls = 0; self.failures = []; self.change = None; self.alter = None; self.service="arco-geo-series"
        self.times = [f"2025-{m:02d}-{d:02d}T12:00:00Z" for m, d in [(8,29),(8,30),(8,31),(9,1),(9,2),(9,3)]]
        self.coords = {"time": self.times, "depth": [0.5, 10., 100., 200., 250.],
            "latitude": [34., 34.5, 35., 35.5, 36.], "longitude": [-122., -121.5, -121., -120.5, -120.]}
        self.fields = {}
        for n, unit in [("zos", "m"), ("thetao", "degrees_C"), ("so", "1e-3"), ("uo", "m s-1"), ("vo", "m s-1"), ("bottomT", "degrees_C")]:
            self.fields[n] = {"dtype": "float32", "dimensions": ["time"] + ([] if n in ("zos", "bottomT") else ["depth"]) + ["latitude", "longitude"],
                "attributes": {"units": unit, "long_name": n, "custom_published_attribute": "preserve me"}}
    def inventory(self, dataset_id, dataset_version=None):
        inv = {"schema_version": "glorys_inventory_v1", "product_id": g.PRODUCT, "dataset_id": dataset_id,
            "dataset_version": "202311", "dataset_part": "default", "service": self.service, "toolbox_version": "2.5.0",
            "coordinates_verified": True, "coordinates": copy.deepcopy(self.coords), "variables": copy.deepcopy(self.fields),
            "coordinate_attributes": {"depth": {"positive": "down", "units": "m"}, "latitude": {"units": "degrees_north"}, "longitude": {"units": "degrees_east"}, "time": {"standard_name": "time"}}}
        if self.change: self.change(inv)
        return inv
    def subset(self, plan, chunk, target, dry_run=False):
        if dry_run: return {"estimated_size": 1000}
        self.calls += 1
        if self.failures: raise self.failures.pop(0)
        c = copy.deepcopy(plan["coordinates"]); c["time"] = chunk["times"]
        data = {}
        for n, field in plan["variables"].items():
            shape = tuple(len(c[d]) for d in field["dimensions"])
            value = {"zos": .2, "thetao": 12., "so": 34., "uo": .1, "vo": -.2, "bottomT": 7.}[n]
            a = np.full(shape, value, dtype="float32")
            a[..., 0, 0] = np.nan  # land
            if "depth" in field["dimensions"] and len(c["depth"]) > 1: a[:, -1, -1, -1] = np.nan  # below seabed
            data[n] = (field["dimensions"], a, field["attributes"])
        coords = {n: (n, np.array([x.rstrip("Z") for x in v], dtype="datetime64[s]") if n == "time" else v, plan["coordinate_attributes"].get(n, {})) for n, v in c.items()}
        ds = xr.Dataset(data, coords=coords, attrs={"source": "OFFLINE SYNTHETIC FIXTURE"})
        if self.alter: ds = self.alter(ds)
        ds.to_netcdf(target, engine="h5netcdf")


def request():
    return {"schema_version": "glorys_request_v1", "variables": ["water_level", "temperature", "salinity", "currents_3d"],
        "start": "2025-08-30T00:00:00Z", "end": "2025-09-03T00:00:00Z", "bbox": [-121.2,34.7,-120.6,35.2], "depth": [0,200], "halo_cells": 1}


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name); self.backend = FakeBackend()
        self.r = request(); self.rfile = self.root / "request.json"; g.write_json(self.rfile, self.r)
        self.run = self.root / "run"
    def tearDown(self): self.tmp.cleanup()
    def plan(self): return g.save_plan(self.rfile, self.run, backend=self.backend)
    def fetch(self, **kw): return g.fetch(self.run, backend=self.backend, retry_delay=0, **kw)
    def test_aliases_and_mixed_dimensions(self):
        p = self.plan(); self.assertEqual(p["request"]["variables"], ["zos","thetao","so","uo","vo"])
        self.assertNotIn("depth", p["variables"]["zos"]["dimensions"]); self.assertIn("depth", p["variables"]["thetao"]["dimensions"])
    def test_half_open_daily_clocks(self):
        self.r["end"] = "2025-09-02T12:00:00Z"; g.write_json(self.rfile, self.r)
        p = self.plan(); self.assertEqual(len(p["coordinates"]["time"]), 3)
        self.assertEqual(p["coordinates"]["time"][-1], "2025-09-01T12:00:00Z")
    def test_native_depths_and_halo(self):
        p = self.plan(); self.assertEqual(p["coordinates"]["depth"], [.5,10.,100.,200.])
        self.assertEqual(len(p["coordinates"]["longitude"]), 5)
    def test_mesh_all_nodes_and_hash(self):
        mesh = self.root / "fort.14"; mesh.write_text("test\n2 4\n1 -121.2 34.7 5\n2 -120.6 34.7 5\n3 -120.6 35.2 5\n4 -121.0 35.8 5\n", encoding="utf-8")
        self.r.pop("bbox"); self.r["mesh"] = {"path": "fort.14", "crs": "EPSG:4326"}; g.write_json(self.rfile,self.r)
        p = self.plan(); self.assertEqual(p["request"]["bbox"][3],35.8); self.assertEqual(p["request"]["mesh"]["sha256"],g.file_hash(mesh))
        mesh.write_text(mesh.read_text()+"changed\n")
        with self.assertRaises(g.ValidationError): self.fetch()
    def test_unavailable_field(self):
        self.r["variables"] = ["wo"]; g.write_json(self.rfile,self.r)
        with self.assertRaises(g.RequestError): self.plan()
    def test_individual_catalogue_variable(self):
        self.r["variables"] = ["bottomT"]; g.write_json(self.rfile,self.r); self.plan(); self.fetch()
        with xr.open_dataset(self.run/"subset.nc") as ds: self.assertEqual(set(ds.data_vars),{"bottomT"})
    def test_resume_repeat_no_retrieval(self):
        self.plan(); a = self.fetch(max_chunks=1); self.assertFalse(a["complete"]); self.assertEqual(self.backend.calls,1)
        b = self.fetch(); self.assertTrue(b["complete"]); self.assertEqual(self.backend.calls,2)
        c = self.fetch(); self.assertEqual(c["new_chunks"],0); self.assertEqual(c["reused_chunks"],2); self.assertEqual(self.backend.calls,2)
        self.assertTrue(g.health(self.run)["pass"])
    def test_source_masks_values_and_units(self):
        self.plan(); self.fetch()
        with xr.open_dataset(self.run/"subset.nc") as ds:
            self.assertTrue(np.isnan(ds.thetao.values[...,0,0]).all()); self.assertTrue(np.isnan(ds.thetao.values[:,-1,-1,-1]).all())
            self.assertEqual(ds.thetao.attrs["units"],"degrees_C"); self.assertEqual(ds.so.attrs["units"],"1e-3")
            self.assertEqual(float(ds.thetao.values[0,0,1,1]),12.); self.assertEqual(ds.sizes["time"],4)
    def test_transient_three_attempts(self):
        self.plan(); self.backend.failures = [g.TransientError("transport"),g.TransientError("transport")]
        self.fetch(); self.assertEqual(self.backend.calls,4)
        r=g.read_json(g.chunk_paths(self.run,g.read_json(self.run/"download_plan.json")["chunks"][0])[1]); self.assertEqual(r["attempts"],3)
    def test_transient_exhaustion(self):
        self.plan(); self.backend.failures = [g.TransientError("transport")]*3
        with self.assertRaises(g.TransientError): self.fetch()
        self.assertEqual(self.backend.calls,3); self.assertEqual(list((self.run/"chunks").glob("*.nc")),[])
    def test_auth_failure_no_retry(self):
        self.plan(); self.backend.failures=[g.AccessError("Access rejected")]
        with self.assertRaises(g.AccessError): self.fetch()
        self.assertEqual(self.backend.calls,1)
    def test_validation_failure_no_retry(self):
        self.plan(); self.backend.alter=lambda ds: ds.assign_coords(time=ds.time + np.timedelta64(1,"h"))
        with self.assertRaises(g.ValidationError): self.fetch()
        self.assertEqual(self.backend.calls,1)
    def test_corrupt_committed_chunk(self):
        p=self.plan(); self.fetch(max_chunks=1); path,_=g.chunk_paths(self.run,p["chunks"][0]); path.write_bytes(b"bad")
        with self.assertRaises(g.ValidationError): self.fetch()
        self.assertEqual(self.backend.calls,1)
    def test_corrupt_assembly(self):
        self.plan(); self.fetch(); (self.run/"subset.nc").write_bytes(b"bad")
        with self.assertRaises(g.ValidationError): g.health(self.run)
        with self.assertRaises(g.ValidationError): self.fetch()
    def test_changed_request_refused(self):
        self.plan(); self.r["depth"]=[0,100]; g.write_json(self.rfile,self.r)
        with self.assertRaises(g.ValidationError): self.plan()
    def test_changed_source_refused(self):
        self.plan(); self.backend.change=lambda inv: inv["variables"]["so"]["attributes"].update(units="changed")
        with self.assertRaises(g.ValidationError): self.fetch()
        self.assertEqual(self.backend.calls,0)
    def test_extended_source_does_not_change_selection(self):
        self.plan(); self.backend.coords["time"].append("2025-09-04T12:00:00Z"); self.fetch(); self.assertTrue(g.health(self.run)["pass"])
    def test_plan_tamper_refused(self):
        p=self.plan(); p["coordinates"]["depth"]=[1]; g.write_json(self.run/"download_plan.json",p)
        with self.assertRaises(g.ValidationError): self.fetch()
    def test_interrupted_uncommitted_chunk_reacquired(self):
        p=self.plan(); path,_=g.chunk_paths(self.run,p["chunks"][0]); path.parent.mkdir(); path.write_bytes(b"partial")
        stale=path.with_name(path.stem+"."+"a"*32+".part.nc"); stale.write_bytes(b"partial transport")
        self.fetch(); self.assertTrue(g.health(self.run)["pass"])
        self.assertFalse(stale.exists())
    def test_credentials_redacted(self):
        secret="SECRET_USERNAME:SECRET_PASSWORD?token=SECRET_TOKEN"
        e=g.classify_failure(RuntimeError("authentication rejected "+secret))
        self.assertIsInstance(e,g.AccessError); self.assertNotIn("SECRET",str(e))
        class BadLogin:
            def login(self,**kw): print(secret); raise RuntimeError(secret)
        with patch.object(g,"credentials_present",return_value=True), patch.dict("sys.modules",{"copernicusmarine":BadLogin()}):
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream): result=g.check_runtime(access=True)
            self.assertNotIn("SECRET",str(result)+stream.getvalue()); self.assertFalse(result["access"]["ready"])
    def test_unavailable_time_and_depth(self):
        self.r["end"]="2025-10-01T00:00:00Z"; g.write_json(self.rfile,self.r)
        with self.assertRaises(g.RequestError): self.plan()
        self.r=request(); self.r["depth"]=[201,249]; g.write_json(self.rfile,self.r)
        with self.assertRaises(g.RequestError): self.plan()
    def test_daily_time_gap_refused(self):
        self.backend.coords["time"].remove("2025-08-31T12:00:00Z")
        with self.assertRaises(g.ValidationError): self.plan()
    def test_monthly_selection_and_coverage(self):
        self.r["dataset_id"]=g.MONTHLY; self.r["start"]="2025-08-01T00:00:00Z"; self.r["end"]="2025-10-01T00:00:00Z"
        self.backend.coords["time"]=["2025-08-16T12:00:00Z","2025-09-16T00:00:00Z"]
        g.write_json(self.rfile,self.r); p=self.plan(); self.assertEqual(len(p["chunks"]),2); self.fetch()
    def test_no_wet_ocean_refused(self):
        self.plan(); self.backend.alter=lambda ds: ds.assign(zos=xr.full_like(ds.zos,np.nan))
        with self.assertRaises(g.ValidationError): self.fetch()
    def test_plausibility_excursions_preserved(self):
        self.plan(); self.backend.alter=lambda ds: ds.assign(zos=ds.zos*100)
        self.fetch(); h=g.health(self.run); self.assertTrue(h["plausibility_excursions"])
        with xr.open_dataset(self.run/"subset.nc") as ds: self.assertEqual(float(ds.zos.values[0,1,1]),20.)
    def test_writer_lock(self):
        self.plan(); (self.run/"writer.lock").write_text("{}")
        with self.assertRaises(g.ValidationError): self.fetch()
        self.assertEqual(self.backend.calls,0)
    def test_snapshot_first_clock(self):
        g.snapshot(self.rfile,self.run,backend=self.backend)
        with xr.open_dataset(self.run/"subset.nc") as ds:
            self.assertEqual(ds.sizes["time"],1); self.assertEqual(g.iso(ds.time.values[0]),"2025-08-30T12:00:00Z")
    def test_monthly_missing_first_requested_month(self):
        self.r.update(dataset_id=g.MONTHLY,start="2025-08-01T00:00:00Z",end="2025-10-01T00:00:00Z")
        self.backend.coords["time"]=["2025-07-16T12:00:00Z","2025-09-16T00:00:00Z"]
        g.write_json(self.rfile,self.r)
        with self.assertRaises(g.ValidationError): self.plan()
    def test_two_dimensional_request_has_no_depth(self):
        self.r["variables"]=["zos"]; g.write_json(self.rfile,self.r); p=self.plan()
        self.assertNotIn("depth",p["coordinates"]); self.fetch(); self.assertTrue(g.health(self.run)["pass"])
    def test_manifest_corruption_is_visible(self):
        self.plan(); self.fetch(); manifest=g.read_json(self.run/"manifest.json")
        manifest["chunks"][0]["sha256"]="changed"; g.write_json(self.run/"manifest.json",manifest)
        with self.assertRaises(g.ValidationError): g.health(self.run)
    def test_subsecond_half_open_boundary(self):
        self.r["start"]="2025-08-30T12:00:00.001Z"; g.write_json(self.rfile,self.r)
        p=self.plan(); self.assertEqual(p["coordinates"]["time"][0],"2025-08-31T12:00:00Z")
        self.assertEqual(len(p["coordinates"]["time"]),3)
    def test_inserted_source_depth_refused_before_retrieval(self):
        self.plan(); self.backend.coords["depth"].insert(2,50.)
        with self.assertRaises(g.ValidationError): self.fetch()
        self.assertEqual(self.backend.calls,0)
    def test_bounded_batch_crosses_months_and_reuses(self):
        self.r.update(service="arco-time-series",chunk_mode="bounded_batch",variables=["water_level"])
        self.backend.service="arco-time-series"; g.write_json(self.rfile,self.r)
        p=self.plan(); self.assertEqual(len(p["chunks"]),1); self.assertEqual(p["source"]["service"],"arco-time-series")
        self.fetch(); self.assertTrue(g.health(self.run)["pass"]); self.assertEqual(self.fetch()["new_chunks"],0)
        self.assertEqual(self.backend.calls,1)
    def test_service_mismatch_refused(self):
        self.r["service"]="arco-time-series"; g.write_json(self.rfile,self.r)
        with self.assertRaises(g.ValidationError): self.plan()
    def test_unknown_batch_or_service_refused(self):
        self.r["chunk_mode"]="unknown"; g.write_json(self.rfile,self.r)
        with self.assertRaises(g.RequestError): self.plan()
        self.r.pop("chunk_mode"); self.r["service"]="unknown"; g.write_json(self.rfile,self.r)
        with self.assertRaises(g.RequestError): self.plan()


if __name__ == "__main__":
    import argparse
    import time
    parser=argparse.ArgumentParser(); parser.add_argument("--output")
    args=parser.parse_args(); began=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    report={"kind":"offline synthetic fixtures; not live readiness", "at_utc":g.now(),
        "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "pass":result.wasSuccessful(),"elapsed_seconds":time.monotonic()-began}
    if args.output: g.write_json(args.output,report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
