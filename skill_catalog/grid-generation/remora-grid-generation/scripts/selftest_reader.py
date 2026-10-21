#!/usr/bin/env python3
"""Reader-contract tests; synthetic files are temporary and prove no model run."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from grid_math import fit_domain, coordinates
from remora_grid import export_grid, parameters, sha
from reader_fixture import create
from check_reader_output import (compare_field, metric_with_outer_ghosts, check,
                                 PINNED_SOURCE, FIELDS)


class ReaderTests(unittest.TestCase):
    def test_metric_first_halo_is_not_interior_extrapolation(self):
        j, i = np.mgrid[:7, :9]
        field = 1e-3 + i*1e-5 + j*2e-5 + i*j*1e-7
        for name in ("pm", "pn"):
            self.assertTrue(compare_field(name, field, field.copy())["halo_exact"])
            bad = np.pad(field[1:-1, 1:-1], 1, mode="edge")
            with self.assertRaises(AssertionError):
                compare_field(name, field, bad)
            for where in ((0, 0), (0, 4), (6, 8), (3, 4)):
                bad = field.copy()
                bad[where] = np.nextafter(bad[where], np.inf)
                with self.assertRaises(AssertionError):
                    compare_field(name, field, bad)
        extended = metric_with_outer_ghosts(field, 2)
        np.testing.assert_array_equal(extended[2:-2, 2:-2], field)
        self.assertEqual(extended[0, 0], field[0, 0])
        np.testing.assert_array_equal(extended[0, 2:-2], field[0])

    def test_coriolis_requires_explicit_halo(self):
        field = np.arange(63).reshape(7, 9)*1e-7
        expected = np.pad(field[1:-1, 1:-1], 1, mode="edge")
        self.assertTrue(compare_field("f", field, expected)["halo_exact"])
        with self.assertRaises(AssertionError):
            compare_field("f", field, field)
        bad = expected.copy()
        bad[0, 0] += 1e-9
        with self.assertRaises(AssertionError):
            compare_field("f", field, bad)

    def make_grid(self, root):
        fit = fit_domain([[-75.4,38.7],[-75,38.7],[-75,39.1],[-75.4,39.1]],10000)
        arrays = coordinates(fit)
        h = np.full(arrays["pm"].shape,20.)
        p = parameters({"parameters":{"spacing_m":10000,"N":4}})
        grid = root/"grid.nc"
        export_grid(grid,arrays,h,np.ones(h.shape,bool),p,{"geometry":fit},h)
        # Force extents to differ from nominal spacing to catch the former bug.
        with Dataset(grid,"r+") as ds:
            ds["xl"][...] = 12345.125
            ds["el"][...] = 23456.25
        return grid

    def test_exact_copy_extents_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            grid=self.make_grid(root)
            before=sha(grid)
            receipt=create(grid,root/"case")
            self.assertEqual(receipt["grid_sha256"],before)
            self.assertEqual(sha(root/"case/remora_grid.nc"),before)
            self.assertEqual(receipt["logical_extent_m"],[12345.125,23456.25])
            text=(root/"case/inputs.reader").read_text()
            self.assertIn("remora.prob_hi = 12345.125 23456.25 0",text)
            self.assertIn("remora.max_step = 0",text)
            self.assertIn("remora.stop_time = 0",text)
            for name in ("inputs.reader", "fixture.json"):
                data=(root/"case"/name).read_bytes()
                self.assertNotIn(b"\r",data, "HPC payload text must use LF: "+name)
                self.assertTrue(data.endswith(b"\n"))
                self.assertFalse(data.startswith(b"\xef\xbb\xbf"))
            lines=(root/"case/inputs.reader").read_bytes().split(b"\n")
            self.assertEqual(lines.count(b"remora.max_step = 0"),1)
            self.assertEqual(lines.count(b"remora.stop_time = 0"),1)
            self.assertEqual(receipt["inputs_sha256"],sha(root/"case/inputs.reader"))
            with self.assertRaises(ValueError):
                create(grid,root/"case")

    def evidence(self, root):
        """SYNTHETIC CPU evidence for validation error paths, never GPU proof."""
        grid=self.make_grid(root)
        ret=root/"returned"
        create(grid,ret/"case")
        hist=ret/"case/reader_initialized_his_d01.nc"
        with Dataset(grid) as src,Dataset(hist,"w",format="NETCDF3_64BIT_DATA") as dst:
            for key,dim in src.dimensions.items():
                dst.createDimension(key,len(dim))
            dst.createDimension("ocean_time",1)
            dst.createVariable("ocean_time","f8",("ocean_time",))[:]=0
            for name in FIELDS:
                v=src[name]
                data=np.asarray(v[:])
                if name=="f":
                    data=np.pad(data[1:-1,1:-1],1,mode="edge")
                dst.createVariable(name,v.dtype,v.dimensions)[:]=data
        (ret/"case/staged.sha256").write_text("".join(
            f"{sha(ret/'case'/name)}  {name}\n" for name in
            ("remora_grid.nc","reader_initial.nc","inputs.reader")))
        (ret/"initialized.sha256").write_text(f"{sha(hist)}  case/{hist.name}\n")
        (ret/"source_revision.txt").write_text(PINNED_SOURCE+"\n")
        (ret/"executable.sha256").write_text("a"*64+"  /synthetic/test-only\n")
        (ret/"submodules.txt").write_text(" "+"b"*40+" Submodules/AMReX\n")
        (ret/"modules.txt").write_text("SYNTHETIC UNIT TEST NO EXECUTION\n")
        (ret/"reader_accounting.txt").write_text(
            "123|synthetic|COMPLETED|0:0|0|\n123.0|synthetic|COMPLETED|0:0|0|\n")
        (ret/"reader.123.log").write_text(
            "SYNTHETIC UNIT TEST NO EXECUTION\nAMReX (test) initialized\n"+
            "\n".join("Loading "+item+" from NetCDF file remora_grid.nc" for item in
                      ("initial bathymetry","grid variables","masks","initial coriolis"))+
            "\nAMReX finalized\n")
        return grid,ret

    def test_identity_and_gpu_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            grid,ret=self.evidence(root)
            out=root/"unit_test_only.json"
            result=check(grid,ret,"123",out)
            self.assertFalse(result["gpu_runtime_verified"])
            with self.assertRaises(ValueError):
                check(grid,ret,"123",out,expected_executable_sha256="c"*64)
            with self.assertRaises(ValueError):
                check(grid,ret,"123",out,expected_source_revision="d"*40)
            with self.assertRaises(ValueError):
                check(grid,ret,"123",out,require_gpu=True,
                      expected_executable_sha256="a"*64,expected_source_revision=PINNED_SOURCE,
                      expected_amrex_revision="b"*40)
            with (ret/"reader.123.log").open("a") as f:
                f.write("Initializing CUDA...\nCUDA initialized with 1 device.\n")
            with self.assertRaises(ValueError):
                check(grid,ret,"123",out,require_gpu=True,
                      expected_executable_sha256="a"*64,expected_source_revision=PINNED_SOURCE,
                      expected_amrex_revision="b"*40)
            inp=ret/"case/inputs.reader"
            inp.write_text(inp.read_text()+"\nremora.max_step = 2\n")
            fixture=json.loads((ret/"case/fixture.json").read_text())
            fixture["inputs_sha256"]=sha(inp)
            (ret/"case/fixture.json").write_text(json.dumps(fixture))
            with self.assertRaises(ValueError):
                check(grid,ret,"123",out)


if __name__=="__main__":
    unittest.main(verbosity=2)
