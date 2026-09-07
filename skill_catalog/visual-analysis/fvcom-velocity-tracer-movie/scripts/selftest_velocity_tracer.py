"""Numerical and input-contract tests; synthetic artifacts live in a temporary directory."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from netCDF4 import Dataset, stringtochar
from fvcom_velocity_tracer_movie import prepare, reconstruct, native_mesh, epoch, write_html


def fixture(path, times=(0, 10800, 21600), shifted=False, geo_zero=True, uv_offset=0, layers=False):
    with Dataset(path,"w") as ds:
        for key,size in dict(node=4,nele=2,three=3,time=len(times),DateStrLen=26,siglay=2).items():
            ds.createDimension(key,size)
        x=np.array([300000,300100,300100,300000],dtype=float)+(10 if shifted else 0)
        y=np.array([3250000,3250000,3250100,3250100],dtype=float)
        for k,arr,dim in [("x",x,"node"),("y",y,"node"),("lon",np.zeros(4),"node"),("lat",np.zeros(4),"node")]:
            ds.createVariable(k,"f8",(dim,))[:]=arr
        ds.createVariable("nv","i4",("three","nele"))[:]=np.array([[1,2,3],[1,3,4]]).T
        ds.createVariable("Itime","i4",("time",))[:]=60766
        ds.createVariable("Itime2","i4",("time",))[:]=np.asarray(times)*1000
        for name,value in [("ua",1+uv_offset),("va",0)]:
            v=ds.createVariable(name,"f4",("time","nele"));v.units="m s-1";v[:]=value
        ds.createVariable("wet_cells","i4",("time","nele"))[:]=1
        if layers:
            ds.createVariable("siglay_center","f4",("siglay","nele"))[:]=[[-.25,-.25],[-.75,-.75]]
            for name,scale in [("u",1),("v",2)]:
                v=ds.createVariable(name,"f4",("time","siglay","nele"));v.units="m s-1"
                v[:]=np.broadcast_to(np.array([[1,1],[3,3]])*scale,(len(times),2,2))
    return path


class InputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.path=fixture(self.root/"one.nc")
    def tearDown(self):self.tmp.cleanup()
    def load(self,**kw):
        args=dict(inputs=[self.path],crs="EPSG:32615",vector_basis="grid");args.update(kw)
        return prepare(**args)
    def test_zero_lon_recovered(self):
        arr,info=self.load();self.assertTrue(info["invalid_geographic_arrays"])
        self.assertTrue(np.all((arr["geo"][:,0]<-94)&(arr["geo"][:,0]>-96)))
        self.assertEqual(info["finite_wet_counts"],[2,2,2])
    def test_missing_crs(self):
        with self.assertRaisesRegex(ValueError,"CRS"):self.load(crs=None)
    def test_ambiguous_vectors(self):
        with self.assertRaisesRegex(ValueError,"basis"):self.load(vector_basis=None)
    def test_layer_missing(self):
        with self.assertRaisesRegex(ValueError,"absent"):self.load(layer="surface")
    def test_layers(self):
        fixture(self.path,layers=True)
        for layer,expected in [("surface",[1,2]),("bottom",[3,6]),("index:0",[1,2])]:
            arr,_=self.load(layer=layer);np.testing.assert_array_equal(arr["uv"][0,0],expected)
        with self.assertRaisesRegex(ValueError,"bounds"):self.load(layer="index:8")
    def test_geometry_mismatch(self):
        p=fixture(self.root/"two.nc",shifted=True)
        with self.assertRaisesRegex(ValueError,"inconsistent"):self.load(inputs=[self.path,p])
    def test_duplicate_conflict(self):
        p=fixture(self.root/"two.nc",uv_offset=2)
        with self.assertRaisesRegex(ValueError,"Conflicting"):self.load(inputs=[self.path,p])
    def test_duplicate_identity(self):
        arr,info=self.load(inputs=[self.path,self.path]);self.assertEqual(len(info["times"]),3)
    def test_missing_record_gap(self):
        fixture(self.path,times=(0,10800,32400));_,info=self.load();self.assertEqual(info["gap_after_indices"],[1])
    def test_unavailable_endpoint(self):
        with self.assertRaisesRegex(ValueError,"endpoints"):self.load(start="2025-04-01T00:01:00Z")
    def test_invalid_connectivity(self):
        with Dataset(self.path,"a") as ds:ds["nv"][0,0]=999
        with self.assertRaisesRegex(ValueError,"connectivity"):self.load()
    def test_degenerate_geometry(self):
        with Dataset(self.path,"a") as ds:ds["x"][1]=ds["x"][0];ds["y"][1]=ds["y"][0]
        with self.assertRaisesRegex(ValueError,"Degenerate"):self.load()
    def test_bad_wet_mask(self):
        with Dataset(self.path,"a") as ds:ds["wet_cells"][0,0]=5
        with self.assertRaisesRegex(ValueError,"0/1"):self.load()
    def test_invalid_vectors_excluded(self):
        with Dataset(self.path,"a") as ds:ds["ua"][0,0]=np.nan
        arr,info=self.load();self.assertEqual(info["invalid_wet_vectors_excluded"],1);self.assertEqual(arr["wet"][0,0],0)
    def test_obc_validation(self):
        p=self.root/"obc.dat";p.write_text("OBC Node Number = 2\n1 1 1\n2 2 1\n")
        arr,info=self.load(obc=p);self.assertEqual(arr["boundaryType"].sum(),1)
        p.write_text("OBC Node Number = 2\n1 1 1\n2 99 1\n")
        with self.assertRaisesRegex(ValueError,"OBC"):self.load(obc=p)
    def test_uniform_reconstruction(self):
        arr,_=self.load();out=reconstruct(arr,arr["uv"][0],arr["wet"][0]);np.testing.assert_allclose(out[:,:,0],1);np.testing.assert_allclose(out[:,:,1],0)
    def test_wet_mask_reconstruction(self):
        arr,_=self.load();out=reconstruct(arr,np.array([[1,0],[10,0]]),np.array([1,0]));np.testing.assert_allclose(out[0,:,0],1);self.assertFalse(out[1].any())
    def test_east_north_transform(self):
        arr,_=self.load(vector_basis="east-north");self.assertTrue(np.all(arr["uv"][0,:,0]>.999));self.assertTrue(np.all(np.abs(arr["uv"][0,:,1])<.1))
        np.testing.assert_array_equal(arr["sourceUv"][0],[[1,0],[1,0]])
    def test_output_collision(self):
        arr,info=self.load()
        with self.assertRaisesRegex(ValueError,"overwrite"):write_html(arr,info,self.path)
    def test_timezone(self):self.assertEqual(epoch("2025-04-01T03:00:00+03:00"),epoch("2025-04-01T00:00:00Z"))


if __name__=="__main__":unittest.main(verbosity=2)
