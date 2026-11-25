import json
import shutil
import tempfile
import unittest
from pathlib import Path

import netCDF4 as nc
import numpy as np

from freshwater import anomaly
from lake_initial import DIRECT, FLAGS, barotropic, build, save, sha
from restart_handoff import verify


class InitializationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.source=self.root/"analysis.nc"
        with nc.Dataset(self.source,"w") as f:
            for name,size in (("node",3),("nele",1),("three",3),("siglay",2),("siglev",3),("time",1),("DateStrLen",26)):
                f.createDimension(name,size)
            f.CoordinateSystem="GeoReferenced"; f.CoordinateProjection="UTM"; f.source="test"
            for name,dims,data in (("nv",("three","nele"),[[1],[2],[3]]),("h",("node",),[3,6,9]),
                    ("siglay",("siglay","node"),[[-.25]*3,[-.75]*3]),("siglev",("siglev","node"),[[0]*3,[-.5]*3,[-1]*3])):
                f.createVariable(name,"i4" if name=="nv" else "f4",dims)[:]=data
            for name in ("lon","lat","x","y","lonc","latc","xc","yc"):
                f.createVariable(name,"f4",("nele" if name.endswith("c") else "node",))[:]=[5] if name.endswith("c") else [4,5,6]
            clock=f.createVariable("time","f8",("time",))
            clock.units="hours since 2025-07-01 00:00:00"; clock[:]=[0]
            f.createVariable("iint","i4",("time",))[:]=99
            for name in DIRECT:
                axis="nele" if name in ("u","v") or "cells" in name else "node"
                vertical=() if name in ("zeta",*FLAGS) else ("siglay",) if name in ("u","v","salinity","temp") else ("siglev",)
                var=f.createVariable(name,"i4" if name in FLAGS else "f4",("time",*vertical,axis))
                var[:]=1
            f["temp"][0]=[[10,11,12],[4,5,6]]; f["salinity"][:]=0.; f["zeta"][:]=.1
            f["u"][0]=[[.5],[-.25]]; f["u"].standard_name="eastward_sea_water_velocity"
            f["v"][0]=[[.1],[-.05]]; f["v"].standard_name="northward_sea_water_velocity"
        self.reference=self.root/"reference.nc"
        shutil.copyfile(self.source,self.reference)
        g=self.root/"grid.dat"; g.write_text("Node Number = 3\nCell Number = 1\n1 1 3 2 1\n1 4 4 3\n2 5 5 6\n3 6 6 9\n")
        d=self.root/"depth.dat"; d.write_text("Node Number = 3\n4 4 3\n5 5 6\n6 6 9\n")
        s=self.root/"sigma.dat"; s.write_text("NUMBER OF SIGMA LEVELS = 3\nSIGMA COORDINATE TYPE = UNIFORM\n")
        self.contract=self.root/"grid.json"
        save(self.contract,dict(status="ready",source_sha256=sha(self.reference),files={k:dict(name=p.name,sha256=sha(p)) for k,p in (("grd",g),("dep",d),("sig",s))},
             nodes=3,elements=1,sigma_layers=2,sigma_interfaces=3,sigma_interface_values=[0,-.5,-1],geometry="spherical",
             grid_file_units="degrees",ascii_to_native_nv_one_based=[1,3,2]))
        self.pressure=self.root/"p.nc"
        with nc.Dataset(self.pressure,"w") as f:
            f.createDimension("time",1); f.createDimension("node",3)
            c=f.createVariable("time","f8",("time",)); c.units="hours since 2025-07-01 00:00:00"; c[:]=0
            p=f.createVariable("air_pressure","f4",("time","node")); p.units="Pa"; p[:]=101000
            f.grid_sha256=sha(g)

    def run_build(self):
        receipt=self.root/"source.json"
        save(receipt,dict(status="ready",target_utc="2025-07-01T00:00:00Z",object=dict(product="fields",key="synthetic"),record=0,
                          sha256=sha(self.source),path=str(self.source)))
        return build(receipt,self.reference,self.contract,self.pressure,self.root/"state")

    def test_build_and_continuation(self):
        result=self.run_build()
        out=self.root/"state"
        with nc.Dataset(out/"lake_initial.nc") as f:
            self.assertEqual(float(f["ua"][0,0]),.125)
            np.testing.assert_array_equal(f["u"][0,:,0],[.5,-.25])
            self.assertGreater(np.ptp(f["temp"][0],axis=0).max(),0)
            self.assertEqual(float(f["salinity"][:].max()),0)
        checkpoint=self.root/"restart.nc"
        shutil.copyfile(out/"lake_initial.nc",checkpoint)
        with nc.Dataset(checkpoint,"r+") as f:
            f["Times"][0]=np.asarray(list("2025/07/02 00:00:00.000000"),dtype="S1")
            f["iint"][0]=8640
            # The NML selects a 2018 reference for native output, while the original startup uses MJD.
            f["Itime"].units="days since 2018-01-01 00:00:00"
            f["Itime"][0]=int(nc.date2num(nc.num2date(f["time"][0]+1,f["time"].units),f["Itime"].units))
            f["time"][0]+=1
        initial=out/"initialization_manifest.json"
        receipt=verify(checkpoint,initial,"2025-07-02T00:00:00Z","a"*64,"test-only",sha(checkpoint))
        self.assertEqual(receipt["source_iint"],8640)
        with self.assertRaises(ValueError):
            verify(checkpoint,initial,"2025-07-03T00:00:00Z","a"*64,"test-only",sha(checkpoint))
        self.assertEqual(result["new_iint"],0)

    def test_missing_wet_temperature(self):
        with nc.Dataset(self.source,"r+") as f: f["temp"][0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"temp"): self.run_build()

    def test_reject_wrong_mesh(self):
        with nc.Dataset(self.source,"r+") as f: f["nv"][:]=[[2],[1],[3]]
        with self.assertRaisesRegex(ValueError,"native nv"): self.run_build()

    def test_reject_layer_reversal(self):
        with nc.Dataset(self.source,"r+") as f: f["siglay"][:]=f["siglay"][::-1,:]
        with self.assertRaisesRegex(ValueError,"siglay"): self.run_build()

    def test_freshwater_density_maximum(self):
        t=np.asarray([0,4,8,20],dtype="f4")
        atmospheric=1000*(1+anomaly(t,np.zeros(4,"f4")))
        np.testing.assert_allclose(atmospheric,[999.8426,999.9750,999.8509,998.2063],atol=.012,rtol=0)
        self.assertEqual(np.argmax(atmospheric),1)
        self.assertTrue(np.all(anomaly(t,np.full(4,300.))>anomaly(t,np.zeros(4))))


if __name__=="__main__": unittest.main()
