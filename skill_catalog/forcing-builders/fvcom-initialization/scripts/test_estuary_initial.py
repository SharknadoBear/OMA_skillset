"""Focused cross-grid FVCOM mapping and state regressions."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import netCDF4 as nc
import numpy as np
from pyproj import CRS,Transformer
from estuary_remap import containing, dens2, edges, extend, fit_elevation, vertical, wet_state
from estuary_initial import east_unit, map_fields
from lake_initial import sha


class Estuary(unittest.TestCase):
    def test_wet_triangle_hole(self):
        vertices=np.array([[0,0],[1,0],[0,1],[2,0],[2,1],[3,0]],dtype=float)
        tri=np.array([[0,1,2],[3,4,5]])
        target=np.array([[.25,.25],[1.5,.25],[2.25,.25]])
        ids,w=containing(vertices,tri,np.array([0,1]),target)
        np.testing.assert_array_equal(ids,[0,-1,1])
        np.testing.assert_allclose(w[0],[.5,.25,.25])
        # Two donor components separated by a dry gap cannot seed each other.
        with self.assertRaisesRegex(ValueError,"without connected source"):
            extend((np.array([0,-1,-1]),w),target,np.array([[0,1],[1,2]]),np.array([True,False,True]))

    def test_vertical_physical_depth_and_endpoints(self):
        z=np.array([[-1.,-2.],[-3.,-4.],[-5.,-8.]])
        values=3*z+2
        target=np.array([[0.,-2.],[-2.,-6.],[-40.,-20.]])
        np.testing.assert_allclose(vertical(z,values,target),[[-1,-4],[-4,-16],[-13,-22]])
        with self.assertRaises(ValueError): vertical(z[::-1],values,target)

    def test_wet_dry_flags_and_clamp(self):
        h=np.array([1.,1.,.02,.02])
        tri=np.array([[0,1,2],[1,2,3]])
        z,wn,wc=wet_state(h,np.array([1.,0.,0.,0.]),tri,.05)
        np.testing.assert_array_equal(wn,[True,True,False,False])
        np.testing.assert_array_equal(wc,[True,False])
        np.testing.assert_allclose(z,[1,0,.03,.03])

    def test_dens2_density(self):
        # Independent double-precision evaluation of the published FVCOM DENS2 polynomial.
        t=np.array([[4.,12.,20.]])
        salt=np.array([[0.,17.,35.]])
        expected=[]
        for s,T in zip(salt[0],t[0]):
            b=.814876577*s-.00048249614*s**2+.00000676786136*s**3
            thermal=1-.004786*T+.000098185*T**2-.0000010843*T**3
            factor=T*(.00001803+T*(-.0000008164+.00000001667*T))
            expected.append(((b+.03895414)*(thermal+(b-.22584586)*factor)
                             -(T-3.98)**2*(T+283)/(503.57*(T+67.26)))/1000)
        np.testing.assert_allclose(dens2(salt,t),[expected],rtol=3e-7)

    def test_rotation(self):
        crs=CRS(32610)
        xy=np.column_stack(Transformer.from_crs(4326,crs,always_xy=True).transform([-124.,-123.,-122.],[46.,46.,46.]))
        east,north=east_unit(crs,xy)
        np.testing.assert_allclose(east**2+north**2,1)
        self.assertLess(north[0],0)
        self.assertAlmostEqual(north[1],0,6)
        self.assertGreater(north[2],0)

    def test_local_boundary_and_dry_wetting_limit(self):
        xy=np.array([[i,j] for i in range(5) for j in (0,1)],dtype=float)
        tri=np.array([t for i in range(0,8,2) for t in ([i,i+1,i+2],[i+1,i+2,i+3])])
        h=np.ones(10)
        initial=np.ones(10)
        fitted,dist=fit_elevation(initial,h,xy,tri,np.array([0,1]),np.array([.3,-4.]),2.5,.1)
        np.testing.assert_allclose(fitted[:2],[.3,-.9])
        np.testing.assert_array_equal(fitted[dist>=2.5],initial[dist>=2.5])
        self.assertTrue((fitted[2:]>-.9).all() and (fitted[2:]<=1).all())
        self.assertEqual(len(edges(tri)),17)

    def test_temperature_priority_and_stationary_field_identity(self):
        with TemporaryDirectory() as tmp, nc.Dataset("source","w",diskless=True) as s, nc.Dataset("target","w",diskless=True) as t:
            for d in (s,t):
                for name,size in dict(node=3,nele=1,time=1,siglev=3,siglay=2).items():
                    d.createDimension(name,size)
                d.createVariable("h","f4",("node",))[:]=4.
                d.createVariable("siglay","f4",("siglay","node"))[:]=[[-.25]*3,[-.75]*3]
                d.createVariable("siglev","f4",("siglev","node"))[:]=[[0.]*3,[-.5]*3,[-1.]*3]
            for name,dims in dict(zeta=("time","node"),salinity=("time","siglay","node"),
                                  u=("time","siglay","nele"),v=("time","siglay","nele"),
                                  omega=("time","siglev","node"),ww=("time","siglay","nele"),tauc=("time","nele")).items():
                s.createVariable(name,"f4",dims)[:]=1. if name=="salinity" else 0.
            s["u"].standard_name="eastward_sea_water_velocity"
            s["v"].standard_name="northward_sea_water_velocity"
            xy=np.array([[500000,5100000],[500100,5100000],[500000,5100100]])
            tri=np.array([[0,1,2]])
            mapping=dict(node_source_cell=np.ones(3,int),node_weights=np.eye(3),cell_source_cell=np.ones(1,int))
            def choose(spec):
                state,_,diag=map_fields(s,0,t,CRS(32610),xy,tri,tri,np.zeros(3),np.ones(3,bool),
                                         np.ones(1,bool),mapping,spec,"frozen_grid")
                return state["temp"],diag["temperature_mode"]
            with self.assertRaisesRegex(ValueError,"No specified temperature"): choose({})
            self.assertEqual(choose({"uniform_degrees_C":5})[1],"explicit_uniform")
            stationary=Path(tmp)/"stationary.nc"
            climate=np.array([[14.,15.,17.],[6.,7.,8.]],dtype="f4")
            with nc.Dataset(stationary,"w") as f:
                f.createDimension("siglay",2)
                f.createDimension("node",3)
                f.createVariable("siglay","f4",("siglay","node"))[:]=t["siglay"][:]
                f.createVariable("temp","f4",("siglay","node"))[:]=climate
                f["temp"].units="degrees_C"
                f.grid_sha256="frozen_grid"
            spec=dict(documented_3d_file=str(stationary),provenance="verified March climatology",
                      sha256=sha(stationary),uniform_degrees_C=5)
            temperature,mode=choose(spec)
            self.assertEqual(mode,"documented_stationary_3d")
            np.testing.assert_array_equal(temperature,climate)
            with self.assertRaisesRegex(ValueError,"frozen bytes"):
                choose({**spec,"sha256":"incorrect"})
            s.createVariable("temp","f4",("time","siglay","node"))[0]=climate+1
            s["temp"].units="degrees_C"
            temperature,mode=choose(spec)
            self.assertEqual(mode,"time_matched_native_3d")
            np.testing.assert_array_equal(temperature,climate+1)


if __name__=="__main__":
    unittest.main()
