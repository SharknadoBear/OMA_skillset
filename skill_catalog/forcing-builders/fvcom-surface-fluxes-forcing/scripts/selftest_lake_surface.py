"""Water and heat signs, energy and hourly water conservation."""
import numpy as np
from lake_surface import unit_fluxes,precipitation_shape
from lake_water_support import LakeWaterSupport
from ranked_lake_surface import mask_samples


def check_shoreline():
    y=np.array([2.,1.,0.])
    x=np.arange(6,dtype=float)
    water=np.zeros((3,6),bool)
    water[:,:2]=True
    water[1,4]=True  # A separate water body cannot supply heat to the target lake.
    nodes=np.array([[1.5,.5],[1.5,1.6],[1.0,3.9],[0.,1.]])
    field=np.array([[4.,8.,35.,35.,50.,35.],[8.,12.,35.,35.,50.,35.],[10.,14.,35.,35.,50.,35.]])
    support=LakeWaterSupport((y,x),nodes,water,seed=(1.,0.),maximum_m=3.)
    original=np.array([8.,25.,50.,14.])
    np.testing.assert_allclose(support.apply(field,original.copy()),[8.,10.,12.,14.])
    assert [support.info[k] for k in ("fully_supported","renormalized","nearest_same_lake")]==[2,1,1]
    try: LakeWaterSupport((y,x),nodes,water,seed=(1.,0.),maximum_m=2.)
    except ValueError: pass
    else: raise AssertionError("distant water donor passed")

def main():
    check_shoreline()
    all_masks = mask_samples("2025-06-03T00:00:00Z","2025-08-01T00:00:00Z")
    assert [t.strftime("%Y%m%d") for t in all_masks] == [
        "20250603","20250617","20250619","20250624","20250626","20250701","20250801"]
    short_masks = mask_samples("2025-06-24T00:00:00Z","2025-06-24T03:00:00Z")
    assert [t.hour for t in short_masks] == [0,3]
    sw,heat,e=unit_fluxes(dict(down_sw=200,up_sw=20,down_lw=300,up_lw=380,sensible=10,latent=25))
    assert (sw,heat)==(180,65) and np.isclose(e,1e-8,rtol=1e-12,atol=0)
    rates=np.array([0,3,2,8,0])*1e-7
    for k,val in enumerate(rates):
        left=min(val,rates[max(0,k-1)]);right=min(val,rates[min(len(rates)-1,k+1)])
        shaped=precipitation_shape(val,left,right)
        np.testing.assert_allclose(np.trapezoid(shaped,x=[0,60,3540,3600]),3600*val,rtol=1e-12)
        assert min(shaped)>=0
    print("Surface energy, signed water flux and interval conservation passed.")
if __name__=="__main__": main()
