"""Exercise the native depth floor and wet-node flags at a prescribed boundary."""
from pathlib import Path
import tempfile
import unittest

import netCDF4 as nc
import numpy as np

from audit_columbia_wind_boundary import check_boundary


def clock(ds):
    ds.createDimension("time", 2)
    ds.createVariable("Itime", "i4", ("time",))[:] = [60000, 60000]
    ds.createVariable("Itime2", "i4", ("time",))[:] = [0, 360000]


class BoundaryTest(unittest.TestCase):
    def setUp(self):
        sandbox = tempfile.TemporaryDirectory()
        self.addCleanup(sandbox.cleanup)
        self.root = Path(sandbox.name)
        self.nml = self.root/"case.nml"
        self.grid = self.root/"grid.nc"
        self.nml.write_text("""STARTUP_TYPE = 'coldstart'
START_DATE = '2023-02-25 00:00:00'
END_DATE = '2023-02-25 00:06:00'
EXTSTEP_SECONDS = 1
ISPLIT = 1
IRAMP = 0
MIN_DEPTH = 0.05
WETTING_DRYING_ON = T
OBC_NODE_LIST_FILE = 'obc.dat'
DEPTH_FILE = 'dep.dat'
OBC_ELEVATION_FILE = 'elev.nc'
""")
        (self.root/"obc.dat").write_text("OBC Node Number = 2\n1 1 1\n2 3 1\n")
        (self.root/"dep.dat").write_text("Node Number = 4\n0 0 .5\n0 1 2\n1 1 3\n1 0 2\n")
        with nc.Dataset(self.root/"elev.nc","w") as ds:
            clock(ds)
            ds.createDimension("nobc", 2)
            ds.createVariable("obc_nodes","i4",("nobc",))[:] = [1,3]
            ds.createVariable("elevation","f4",("time","nobc"))[:] = [[0,0],[-1.1,.2]]
        with nc.Dataset(self.grid,"w") as ds:
            clock(ds)
            for dim,size in (("node",4),("three",3),("nele",2)):
                ds.createDimension(dim,size)
            ds.createVariable("iint","i4",("time",))[:] = [0,360]
            ds.createVariable("h","f4",("node",))[:] = [.5,2,3,2]
            ds.createVariable("nv","i4",("nele","three"))[:] = [[1,2,4],[2,3,4]]
            ds.createVariable("zeta","f4",("time","node"))[:] = [[0,0,0,0],[-.45,0,.2,0]]
            ds.createVariable("wet_nodes","i4",("time","node"))[:] = [[1,1,1,1],[0,1,0,1]]
            ds.createVariable("wet_cells","i4",("time","nele"))[:] = [[1,1],[1,0]]

    def audit(self):
        return check_boundary(self.nml,self.root,[self.grid])

    def test_floor_and_adjacent_dry_cell(self):
        result = self.audit()
        self.assertEqual(result["checked_node_samples"],4)
        self.assertEqual(result["dry_node_ids"],[1,3])
        self.assertEqual(result["dry_node_samples"],2)
        self.assertEqual(result["floor_limited_node_samples"],1)
        self.assertAlmostEqual(result["max_floor_adjustment_m"],.65,places=6)

    def test_dry_elevation_remains_checked(self):
        with nc.Dataset(self.grid,"r+") as ds:
            ds["zeta"][1,0] = -.4
        with self.assertRaisesRegex(ValueError,"expected saved elevation"):
            self.audit()

    def test_impossible_dry_flag(self):
        with nc.Dataset(self.grid,"r+") as ds:
            ds["wet_cells"][1,1] = 1
        with self.assertRaisesRegex(ValueError,"wet-node flags"):
            self.audit()

    def test_masked_node_flag(self):
        with nc.Dataset(self.grid,"r+") as ds:
            ds["wet_nodes"][1,0] = np.ma.masked
        with self.assertRaisesRegex(ValueError,"masked"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
