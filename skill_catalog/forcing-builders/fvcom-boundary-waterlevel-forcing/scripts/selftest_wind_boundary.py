"""Regression tests for the wind boundary product, including the staged simulation entry point."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import netCDF4 as nc
import numpy as np

from glorys_obc import DATASET, PRODUCT, daily_midpoints, monthly_days, sample_common_stencil
from waterlevel_core import Boundary, DAY_MS, MJD_EPOCH_MS, sha256_file, write_fvcom_forcing
from wind_boundary import MANIFEST, verify_wind_boundary


def ms(values):
    return np.asarray(values, dtype="datetime64[ms]").astype("i8")


def source(folder, dates, freq):
    folder.mkdir()
    path = folder/"subset.nc"
    with nc.Dataset(path, "w") as ds:
        for n, vals, units in (("time",dates,"milliseconds since 1970-01-01"), ("latitude",[44,45],"degrees_north"),
                               ("longitude",[-125,-124],"degrees_east")):
            ds.createDimension(n,len(vals))
            v = ds.createVariable(n,"f8",n)
            v.units = units
            v[:] = vals
        z = ds.createVariable("zos","f8",("time","latitude","longitude"))
        z.units = "m"
        z.standard_name = "sea_surface_height_above_geoid"
        z[:] = np.arange(len(dates))[:,None,None]*0.001 + np.arange(4).reshape((2,2))*0.01
    receipt = dict(complete=True, plan_hash="verified-plan", source=dict(product_id=PRODUCT,
                   dataset_id=DATASET+freq+"-m", dataset_version="test"),
                   assembled=dict(file=path.name,sha256=sha256_file(path)))
    (folder/"manifest.json").write_text(json.dumps(receipt))
    (folder/"health_report.json").write_text(json.dumps({"pass":True,"plan_hash":"verified-plan"}))


class BoundaryTests(unittest.TestCase):
    def test_duplicate_month_and_daily_gap(self):
        months = ms(np.arange("1993-01","2013-01",dtype="datetime64[M]"))
        self.assertEqual(monthly_days(months).sum(), 7305)
        months[1] = months[0]
        with self.assertRaises(ValueError):
            monthly_days(months)
        with self.assertRaises(ValueError):
            daily_midpoints(ms(["2025-03-16","2025-03-18"]))

    def test_common_wet_mapping(self):
        gx, gy = np.array([-125.,-124.]), np.array([44.,45.])
        field = np.ones((2,2,2))
        field[1,1,1] = np.nan
        sample, mapping = sample_common_stencil(gx,gy,field,np.array([[-124.99,44.01]]),10)
        self.assertEqual(mapping[0]["method"],"nearest_common_wet")
        self.assertTrue((sample == 1).all())
        for point in ([-124.5,44.5],[-125.1,44.01]):
            with self.assertRaises(ValueError):
                sample_common_stencil(gx,gy,field,np.array([point]),10)

    def test_staged_readback_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            daily, monthly = base/"daily", base/"monthly"
            days = ms(["2025-03-17","2025-03-18","2025-03-19"])
            months = ms(np.arange("1993-01","2013-01",dtype="datetime64[M]"))
            source(daily, days, "P1D")
            source(monthly, months, "P1M")
            inputdir = base/"input"
            inputdir.mkdir()
            mesh, obc, pressure = [inputdir/name for name in ("mesh.dat","obc.dat","pressure.nc")]
            mesh.write_text("synthetic mesh identity")
            obc.write_text("OBC Node Number = 2\n1 2 1\n2 1 1\n")
            ids = np.array([2,1])
            tide = base/"elev.nc"
            pts = base/"points.csv"
            pts.write_text("node_id,longitude,latitude\n2,-124.7,44.3\n1,-124.8,44.4\n")
            start = ms("2025-03-18")
            time = start + np.arange(11)*360000
            bn = Boundary(ids,np.array([-124.7,-124.8]),np.array([44.3,44.4]),(np.arange(2),),"fixture")
            write_fvcom_forcing(tide,bn,time,np.zeros((11,2)),case_name="synthetic",vertical_datum="tide",
                                source_name="fixture",source_sha256="test",spatial_method="fixture")
            with nc.Dataset(tide,"a") as ds:
                ds.obc_points_sha256 = sha256_file(pts)
            tide_manifest = base/"tide.json"
            tide_manifest.write_text(json.dumps(dict(status="ready",health={"status":"pass"},
                                          hashes={"forcing_sha256":sha256_file(tide)})))
            with nc.Dataset(pressure,"w") as ds:
                ds.createDimension("time",2)
                ds.createDimension("node",2)
                ds.createVariable("node_id","i4","node")[:] = [1,2]
                ds.createVariable("lon","f8","node")[:] = bn.lon[::-1]
                ds.createVariable("lat","f8","node")[:] = bn.lat[::-1]
                ds.createVariable("Itime","i4","time")[:] = (start-MJD_EPOCH_MS)//DAY_MS
                ds.createVariable("Itime2","i4","time")[:] = [0,3600000]
                pr = ds.createVariable("air_pressure","f4",("time","node"))
                pr.units = "Pa"
                pr[:] = [[100000,101325],[101000,102000]]
            owner = dict(schema="fvcom_atmosphere_manifest_v1",status="ready",provider="synthetic",
                         pressure_reference="absolute_mean_sea_level", mesh_sha256=sha256_file(mesh),
                         obc_sha256=sha256_file(obc), files={"surface":{"path":pressure.name,"sha256":sha256_file(pressure)}})
            (inputdir/"atmosphere_manifest.json").write_text(json.dumps(owner))
            entry = [sys.executable,str(Path(__file__).with_name("wind_boundary.py")), "--tide",str(tide),
                     "--tide-manifest",str(tide_manifest),"--points",str(pts),"--pressure",str(pressure),
                     "--mesh",str(mesh),"--obc",str(obc),"--nearest-limit-km","5","--output-dir",str(inputdir)]
            passed = subprocess.run(entry+["--daily",str(daily),"--monthly",str(monthly)],capture_output=True,text=True)
            self.assertEqual(passed.returncode,0,passed.stderr)
            manifest = json.loads((inputdir/MANIFEST).read_text())
            values = {"NML_SURFACE_FORCING":dict(WIND_ON="T",AIRPRESSURE_ON="T",AIRPRESSURE_FILE="'pressure.nc'"),
                      "NML_GRID_COORDINATES":{"GRID_FILE":"'mesh.dat'"},
                      "NML_OPEN_BOUNDARY_CONTROL":dict(OBC_ELEVATION_FORCING_ON="T",OBC_NODE_LIST_FILE="'obc.dat'",
                                                      OBC_ELEVATION_FILE="'elev.nc'")}
            self.assertEqual(verify_wind_boundary(inputdir,values), inputdir/"pure_tpxo.nc")
            with nc.Dataset(inputdir/"wind_boundary_components.nc") as ds:
                self.assertAlmostEqual(ds["inverse_barometer"][0,0], 0.)
                self.assertAlmostEqual(ds["inverse_barometer"][0,1], 1325/(1025*9.81))
            values["NML_SURFACE_FORCING"]["AIRPRESSURE_ON"] = "F"
            with self.assertRaisesRegex(ValueError,"AIRPRESSURE_ON"):
                verify_wind_boundary(inputdir,values)
            values["NML_SURFACE_FORCING"]["AIRPRESSURE_ON"] = "T"
            # Give a modified sum fresh file hashes: staged arithmetic must still reject it.
            with nc.Dataset(inputdir/"elev.nc","a") as ds:
                ds["elevation"][0,1] += 1325/(1025*9.81)
            manifest["files"]["combined"]["sha256"] = sha256_file(inputdir/"elev.nc")
            (inputdir/MANIFEST).write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"one pressure head"):
                verify_wind_boundary(inputdir,values)

            evidence = base/"source_conventions.txt"
            evidence.write_text("Synthetic regression: source height includes model tide and equilibrium pressure response in run and reference.")
            def ref(path):
                return {"path":str(path), "sha256":sha256_file(path)}
            spec = {"provider":"HYCOM","schema":"hycom_offshore_reference_v1"}
            for period,folder in (("reference",monthly),("run",daily)):
                path = folder/"subset.nc"
                with nc.Dataset(path,"a") as ds:
                    for term, value in (("astronomical",0.1),("pressure_head",0.04)):
                        field = ds.createVariable(term,"f8",("time","latitude","longitude"))
                        field.units = "m"
                        field[:] = value
                item = dict(experiment="synthetic",vertical_reference="synthetic common zero",
                         coordinates=dict(longitude="longitude",latitude="latitude",time="time"),ssh_variable="zos",
                         clock="calendar_month_mean" if period=="reference" else "native_utc_sample",
                         evidence_summary=evidence.read_text(),evidence_url="https://example.org/synthetic-fixture",
                         evidence_file=ref(evidence),file=ref(path),physics={"tide":{"convention":"present_subtract","variable":"astronomical"},
                         "inverse_barometer":{"convention":"present_subtract","variable":"pressure_head"}})
                spec[period] = item
            spec["run"]["cadence_seconds"] = 86400
            package = base/"hycom.json"
            package.write_text(json.dumps(spec))
            passed = subprocess.run(entry+["--hycom-package",str(package)],capture_output=True,text=True)
            self.assertEqual(passed.returncode,0,passed.stderr)
            self.assertEqual(verify_wind_boundary(inputdir,values),inputdir/"pure_tpxo.nc")
            with nc.Dataset(inputdir/"wind_boundary_components.nc") as ds:
                np.testing.assert_allclose(ds["daily_zos"][:],ds["hycom_run_raw"][:]-0.14)
            # Distinct physical terms cannot both remove the same aligned variable.
            spec["run"]["physics"]["inverse_barometer"]["variable"] = "astronomical"
            package.write_text(json.dumps(spec))
            bad = subprocess.run(entry+["--hycom-package",str(package)],capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)
            self.assertIn("duplicate",bad.stderr)
            spec["run"]["physics"]["inverse_barometer"] = {"convention":"unknown"}
            package.write_text(json.dumps(spec))
            bad = subprocess.run(entry+["--hycom-package",str(package)],capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)


if __name__ == "__main__":
    unittest.main()
