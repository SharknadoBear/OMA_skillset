"""Check freshwater COARE26Z thermodynamics, supersaturation, and raw rain closure."""
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
import json

import netCDF4 as nc
import numpy as np

from lake_coare_diagnostic import THERMO, checked_thermodynamics, dewpoint, forcing_clock, humidity, rain_knots, saturation
from surface_fluxes_core import sha256_file


def main():
    t=np.array([0.,10.,25.]);p=np.array([100000.,98000.,101300.])
    # Independent evaluation at 0°C and 1000 hPa with QSAT26's documented enhancement.
    enhancement=1.0007+.00346
    expected=.622*6.112*enhancement/(1000-.378*6.112*enhancement)
    np.testing.assert_allclose(saturation(t,p)[0],expected,rtol=1e-13)
    source=saturation(t,p)*np.array([.6,1.,1.07])
    tc,rh=humidity(t+273.15,source,p)
    np.testing.assert_allclose(tc,t,atol=1e-12)
    np.testing.assert_allclose(rh,[60,100,107],atol=1e-12)
    np.testing.assert_allclose(saturation(tc,p)*rh*.01,source,atol=1e-14)
    require_dewpoint=dewpoint(source,p)
    assert require_dewpoint[0]<tc[0] and require_dewpoint[2]>tc[2]
    try:humidity(t,source,p)
    except ValueError:pass
    else:raise AssertionError("Celsius mislabelled as native HRRR Kelvin was accepted")
    cur=np.array([0.,4e-7,2.e-7]);left=np.array([1e-6,0.,3e-7]);right=np.array([0.,8e-7,1e-7])
    a=rain_knots(cur,left,right)
    assert np.min(a)>=0
    np.testing.assert_allclose(np.trapezoid(a,x=[0,60,3540,3600],axis=0),3600*cur,rtol=2e-15,atol=1e-15)
    np.testing.assert_array_equal(a[-1],rain_knots(right,cur,left)[0])
    assert np.all(np.asarray(a)[:,0]==0) and rh[-1]>100
    clock=forcing_clock(SimpleNamespace(start="2025-06-26T00:00:00Z",end="2025-06-27T00:00:00Z"))
    assert len(clock)==27 and np.all(np.diff(clock)==3600000)
    time="2025-06-26T00:00:00Z"
    with TemporaryDirectory() as tmp:
        d=Path(tmp);file=d/"hrrr_fields.nc"
        with nc.Dataset(file,"w") as ds:
            ds.createDimension("time",1)
            for name,(code,units,level) in THERMO.items():
                v=ds.createVariable(name,"f4",("time",));v.units=units
                v.source_grib_metadata=json.dumps([dict(idx_short_name=code,step_type="instant",source_idx_level=level,
                    valid_time=time,forecast_reference_time=time,message_sha256="a"*64)])
        health=d/"health_check.json";health.write_text(json.dumps(dict(passed=True,checks=[dict(sha256=sha256_file(file))])))
        checked_thermodynamics(file,time)
        try:checked_thermodynamics(file,"2025-06-26T01:00:00Z")
        except ValueError:pass
        else:raise AssertionError("Wrong native analysis clock accepted")
    print("Passed Q2/T/P, dewpoint and supersaturation, zero raw rain under condensation, conservative APCP ramps, source-clock checks.")
if __name__=="__main__":main()
