"""Independent geometry checks; no case-specific geographic expectations."""
import unittest
import numpy as np
from curvilinear import fit_curvilinear, coordinates, projected_at, axis_map
from grid_math import GEOD


class CurvilinearTests(unittest.TestCase):
    def fit(self, **kw):
        d=dict(kind="conformal_graded",nx=90,ny=70)
        d.update(kw)
        return fit_curvilinear([[-75.3,38.8],[-74.7,38.8],[-74.7,39.3],[-75.3,39.3]],
                               500,1000,23,100000,d)

    def test_uniform_metrics_staggers(self):
        f=self.fit()
        a=coordinates(f)
        self.assertEqual(a["lon_rho"].shape,(72,92))
        self.assertEqual(a["lon_u"].shape,(72,91))
        self.assertEqual(a["lon_v"].shape,(71,92))
        self.assertEqual(a["lon_psi"].shape,(71,91))
        self.assertLess(a["orthogonality_error_deg"].max(),.001)
        # Independent geodesic length between u points equals rho width.
        _,_,length=GEOD.inv(a["lon_u"][:,:-1],a["lat_u"][:,:-1],
                           a["lon_u"][:,1:],a["lat_u"][:,1:])
        np.testing.assert_allclose(length,1/a["pm"][:,1:-1],rtol=1e-9)
        for j,i in ((2,2),(30,40),(65,85)):
            lo=a["lon_psi"][j:j+2,i:i+2]
            la=a["lat_psi"][j:j+2,i:i+2]
            area,_=GEOD.polygon_area_perimeter(lo[[0,0,1,1],[0,1,1,0]],la[[0,0,1,1],[0,1,1,0]])
            metric=1/(a["pm"][j+1,i+1]*a["pn"][j+1,i+1])
            self.assertAlmostEqual(abs(area)/metric,1,places=5)

    def test_curved_graded_orthogonal_and_coverage(self):
        f=self.fit(c2=[0,.055],c3=[.01,0],
                   xi_control=[[0,0],[.4,.22],[1,1]],
                   eta_control=[[0,0],[.6,.7],[1,1]])
        a=coordinates(f)
        self.assertGreater((1/a["pm"]).max()/(1/a["pm"]).min(),1.8)
        self.assertLess(a["orthogonality_error_deg"].max(),.005)
        self.assertGreater(a["jacobian_m2"].min(),0)
        self.assertGreater(a["physical_cell_area_projected_m2"].min(),0)
        # Direct differential orthogonality and positive orientation.
        _,_,tx,ty=projected_at(f,np.linspace(0,1,19),np.linspace(1,0,19))
        np.testing.assert_allclose(np.real(tx*np.conj(ty))/(abs(tx)*abs(ty)),0,atol=2e-15)

    def test_invalid_geometry(self):
        for kw in (dict(c2=[2,0]),dict(nx=0),dict(c3=[np.nan,0]),
                   dict(xi_control=[[0,0],[.6,.8],[.5,.9],[1,1]])):
            with self.assertRaises(ValueError):
                self.fit(**kw)
        with self.assertRaises(ValueError):
            coordinates(self.fit(xi_control=[[0,0],[.8,.05],[1,1]],max_neighbor_ratio=1.01))

    def test_axis_and_halo(self):
        value,derivative=axis_map(np.linspace(-.1,1.1,201),[[0,0],[.4,.2],[1,1]])
        self.assertTrue(np.all(np.diff(value)>0))
        self.assertTrue(np.all(derivative>0))
        for invalid in ([[0,0],[1,0]],[[.1,0],[1,1]],[[0,0],[1,np.nan]]):
            with self.assertRaises(ValueError):
                axis_map([.5],invalid)


if __name__=="__main__":
    unittest.main()
