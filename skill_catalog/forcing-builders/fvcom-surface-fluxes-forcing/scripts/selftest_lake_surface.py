"""Water and heat signs, energy and hourly water conservation."""
import numpy as np
from lake_surface import unit_fluxes,precipitation_shape

def main():
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
