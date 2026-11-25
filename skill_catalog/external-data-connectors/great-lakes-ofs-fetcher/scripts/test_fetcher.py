import hashlib
import tempfile
import unittest
from pathlib import Path

import netCDF4 as nc

from great_lakes_ofs import cache, fetch_epoch, ranked, utc
from station_series import select


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        with nc.Dataset(self.root/"original.nc","w") as ds:
            ds.createDimension("time",1)
            clock=ds.createVariable("time","f8",("time",))
            clock.units="hours since 2025-07-01 00:00:00"
            clock[:]=[0]
        self.data=(self.root/"original.nc").read_bytes()
        self.t="2025-07-01T00:00:00Z"
        self.obj=dict(key="lsofs/synthetic.nc",product="fields",stream="nowcast",cycle=self.t,
                      candidate_start=self.t,candidate_end=self.t,size=len(self.data),
                      etag=hashlib.md5(self.data).hexdigest())

    def test_interrupted_range_resume_and_immutable_cache(self):
        seen=[]
        def interrupted(obj,start,end):
            seen.append(start)
            if start: raise ConnectionError("interrupted")
            return self.data[start:end+1]
        with self.assertRaises(ConnectionError): cache(self.obj,self.root,block=64,fetch_range=interrupted)
        seen.clear()
        def resumed(obj,start,end):
            seen.append(start)
            return self.data[start:end+1]
        receipt=cache(self.obj,self.root,block=64,fetch_range=resumed)
        self.assertEqual(seen[0],64)
        self.assertEqual(receipt["clock_utc"],[self.t])
        self.assertEqual(receipt["sha256"],hashlib.sha256(self.data).hexdigest())
        self.assertEqual(fetch_epoch([self.obj],self.t,"fields",self.root)["record"],0)
        Path(receipt["path"]).write_bytes(b"damaged")
        with self.assertRaises(ValueError): cache(self.obj,self.root)

    def test_nowcast_and_cycle_precedence(self):
        forecast={**self.obj,"stream":"forecast","key":"forecast"}
        later={**self.obj,"cycle":"2025-07-01T06:00:00Z","key":"next"}
        self.assertEqual(ranked([later,forecast,self.obj],self.t,"fields",True),[self.obj,later,forecast])
        selected,overlap=select([dict(object=o,clock_utc=[self.t]) for o in [later,forecast,self.obj]],
                                utc(self.t),utc("2025-07-02T00:00:00Z"))
        self.assertEqual(selected[self.t][0]["object"],self.obj)
        self.assertEqual(overlap,2)


if __name__=="__main__": unittest.main()
