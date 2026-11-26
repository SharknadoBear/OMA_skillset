"""Regress native interval arithmetic, failure on missing cycles, and whole-provider routing."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import netCDF4 as nc
import numpy as np

from acquire_wind_pressure import digest, read, write
from lake_interval_surface import KNOT_SECONDS, native_interval, nonnegative_rain, one_hour_mean, shared_boundary
from lake_surface import precipitation_shape
from ranked_lake_surface import run


class IntervalTests(unittest.TestCase):
    def test_grib_bounds_and_integral_differences(self):
        ref=datetime(2008,1,1,tzinfo=timezone.utc)
        def bounds(lo,hi,cycle=ref):
            end=cycle+timedelta(hours=hi)
            row=dict(data_date=int(cycle.strftime("%Y%m%d")),data_time=int(cycle.strftime("%H%M")),
                     step_range=f"{lo}-{hi}",step_units="1",step_type="avg",valid_time=end.isoformat())
            return native_interval(row,end)
        a,b,c=bounds(0,1),bounds(0,2),bounds(0,3)
        self.assertEqual(one_hour_mean(5,a),5)
        self.assertEqual(one_hour_mean(8,b,(5,a)),11)
        self.assertEqual(one_hour_mean(6,c,(8,b)),2)
        self.assertEqual(one_hour_mean(-11,bounds(0,1,ref+timedelta(hours=6))),-11)
        for prior in (None,(5,c),(5,bounds(0,1,ref-timedelta(hours=6)))):
            with self.assertRaisesRegex(ValueError,"prior cumulative"):
                one_hour_mean(8,b,prior)

    def test_conservative_signed_flux_and_water(self):
        vals=np.array([-10.,20.,0.,-2.])
        for left,cur,right in zip(vals,vals[1:],np.roll(vals,-2)):
            knots=precipitation_shape(cur,shared_boundary(left,cur),shared_boundary(cur,right))
            self.assertAlmostEqual(np.trapezoid(knots,x=KNOT_SECONDS)/3600,cur)
            self.assertTrue(np.all(np.array(knots)*cur>=0))

    def test_quantized_rain_reports_correction(self):
        rates,adjustment=nonnegative_rain(np.array([-1e-9,2e-9]),1.1e-9)
        np.testing.assert_array_equal(rates,[0,2e-9])
        np.testing.assert_array_equal(adjustment,[1e-9,0])
        with self.assertRaisesRegex(ValueError,"packing precision"):
            nonnegative_rain(np.array([-2e-9]),1e-9)

    def test_end_anchored_era5_and_rejection(self):
        with TemporaryDirectory() as tmp:
            base=Path(tmp)
            era=base/"sources"/"era5"
            era.mkdir(parents=True)
            mesh=base/"mesh.nc"
            with nc.Dataset(mesh,"w") as d:
                for n,j in (("node",3),("nele",1),("three",3)):
                    d.createDimension(n,j)
                for n,v in (("lon",[0,.5,0]),("lat",[0,0,.5]),("art1",[1,1,1])):
                    d.createVariable(n,"f8",("node",))[:]=v
                for n in ("lonc","latc"):
                    d.createVariable(n,"f8",("nele",))[:]=1/6
                d.createVariable("nv","i4",("three","nele"))[:]=[[1],[2],[3]]
            law=base/"bulk.F"
            law.write_text("subroutine psimple_drag(\ntx=spdx(i)\nty=spdy(i)\nwds=sqrt(tx*tx+ty*ty)\ncd=1.2e-3\n"
                           "if(wds>=11.0_sp)cd=(0.49_sp+0.065_sp*wds)*1.e-3_sp\n"
                           "if(wds>=25.0_sp)cd=(0.49_sp+0.065_sp*25.0_sp)*1.e-3_sp\n"
                           "strx(i)=1.2_sp*cd*tx*wds\nstry(i)=1.2_sp*cd*ty*wds\nend subroutine\n")
            path=era/"fields.nc"
            with nc.Dataset(path,"w") as d:
                d.schema_version="era5_fields_v1"
                d.request_hash="fixture"
                for n,v in (("time",[0,1,2]),("latitude",[-1,1]),("longitude",[-1,1])):
                    d.createDimension(n,len(v))
                    var=d.createVariable(n,"f8",(n,))
                    var[:]=v
                    if n=="time":
                        var.units="hours since 2025-06-30 23:00:00"
                    else:
                        var.units="degrees_north" if n=="latitude" else "degrees_east"
                for n,unit,values in (("u10","m s-1",[1,2,3]),("v10","m s-1",[0,0,0]),
                           ("msl","Pa",[101000,101100,101200]),("ssr","J m-2",[360000,36000,72000]),
                           ("str","J m-2",[0,0,0]),("slhf","J m-2",[0,-9000,18000]),("sshf","J m-2",[0,0,0]),
                           ("tp","m",[.5,.001,.002])):
                    v=d.createVariable(n,"f8",("time","latitude","longitude"))
                    v.units=unit
                    v[:]=np.broadcast_to(np.array(values)[:,None,None],(3,2,2))
                    if n in ("u10","v10"):
                        v.standard_name="eastward_wind" if n=="u10" else "northward_wind"
                    if n in ("ssr","str","sshf","slhf","tp"):
                        v.interval_seconds,v.interval_anchor=3600,"end"
            write(era/"run_manifest.json",dict(request_hash="fixture",plan_hash="pq",chunks={"0":dict(
                  state="complete",output_path=path.name,output_sha256=digest(path)),"old":dict(state="split")},splits={"old":["0"]}))
            write(era/"health_check.json",dict(pass_all=True,completed_chunks=1,expected_chunks=1,request_hash="fixture",plan_hash="pq"))
            args=SimpleNamespace(mesh=mesh,mod_bulk=law,source_root=era.parent,output=base/"out",
                       start="2025-06-30T23:00:00Z",end="2025-07-01T01:00:00Z",projected_crs=None,
                       hrrr_analysis=None,hrrr_precipitation=None,acquire=False,fetch_worker=None)
            run(args)
            selected=read(args.output/"provider_attempts.json")
            self.assertEqual([r["status"] for r in selected["attempts"]],["rejected","selected"])
            with nc.Dataset(args.output/"superior_atmosphere.nc") as d, nc.Dataset(args.output/"superior_pe.nc") as p:
                for k,solar,heat,rain,loss in ((0,10,7.5,.001,-1e-9),(1,20,25,.002+3600*2e-9,0)):
                    sl=slice(3*k,3*k+4)
                    for v,w in ((d["short_wave"],solar),(d["net_heat_flux"],heat),(p["precip"],rain/3600),(p["evap"],loss)):
                        np.testing.assert_allclose(np.trapezoid(v[sl,0],x=KNOT_SECONDS)/3600,w,rtol=2e-7,atol=1e-13)
            with nc.Dataset(path,"a") as d:
                d["tp"].interval_anchor="start"
            args.output=base/"bad_hash"
            with self.assertRaisesRegex(ValueError,"No complete"):
                run(args)
            self.assertIn("changed",read(args.output/"provider_attempts.json")["attempts"][1]["reason"])
            obj=read(era/"run_manifest.json")
            obj["chunks"]["0"]["output_sha256"]=digest(path)
            write(era/"run_manifest.json",obj)
            args.output=base/"bad_interval"
            with self.assertRaisesRegex(ValueError,"No complete"):
                run(args)
            self.assertIn("preceding-hour",read(args.output/"provider_attempts.json")["attempts"][1]["reason"])


if __name__=="__main__":
    unittest.main()
