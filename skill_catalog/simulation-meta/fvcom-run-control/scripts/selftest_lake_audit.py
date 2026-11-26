"""Check retained UTC clocks and the native direct heat, stress, pressure and P/E reader replay."""
from contextlib import ExitStack
import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import netCDF4 as nc
import numpy as np
import fvcom_lake_audit as audit


class NativeReadback(unittest.TestCase):
    def test_forcing_clocks_signs_and_interpolation(self):
        with TemporaryDirectory() as tmp, ExitStack() as opened:
            files=[opened.enter_context(nc.Dataset(Path(tmp)/n,"w")) for n in ("history.nc","met.nc","pe.nc")]
            for ds in files:
                ds.createDimension("time",None)
                ds.createDimension("node",3)
                ds.createDimension("nele",2)
            history,met,pe=files
            origin=audit.clock("2025/07/01 00:00:00.000000")
            met_times=[origin,origin+dt.timedelta(hours=2)]
            pe_times=[origin+dt.timedelta(minutes=k) for k in (59,60,61)]
            sources={}
            for native,(param,var,units,_) in audit.FORCING.items():
                source,times=(pe,pe_times) if native in {"precip","evap"} else (met,met_times)
                sources[param]=(source,times)
                dim="nele" if native in {"uwind_stress","vwind_stress"} else "node"
                x=source.createVariable(var,"f4",("time",dim))
                y=history.createVariable(native,"f4",("time",dim))
                x.units=y.units=sorted(units)[0]
                sign=-1 if native=="evap" else 1
                for i in range(len(times)):
                    x[i]=sign*(i+1)
                y[0]=sign*(2 if source is pe else 1.5)
            self.assertTrue(all(v==0 for v in audit.compare_forcing(history,0,pe_times[1],sources).values()))
            history["evap"][0]=2
            with self.assertRaisesRegex(ValueError,"native forcing differs.*evap"):
                audit.compare_forcing(history,0,pe_times[1],sources)

    def test_precise_time_and_scalar_rounding(self):
        with TemporaryDirectory() as tmp, nc.Dataset(Path(tmp)/"station.nc","w") as ds:
            ds.createDimension("time",1)
            day=ds.createVariable("Itime","i4",("time",))
            day.units="days since 2018-01-01 00:00:00"
            day[:]=2738
            sub=ds.createVariable("Itime2","i4",("time",))
            sub.units="msec since 00:00:00"
            sub[:]=600000
            coarse=ds.createVariable("time","f4",("time",))
            coarse.units=day.units
            coarse[:]=2738+600/86400
            self.assertEqual(audit.axis(ds),[audit.clock("2025-07-01T00:10:00Z")])
            coarse[:]=2739
            with self.assertRaisesRegex(ValueError,"scalar clock"):
                audit.axis(ds)


if __name__=="__main__":
    unittest.main()
