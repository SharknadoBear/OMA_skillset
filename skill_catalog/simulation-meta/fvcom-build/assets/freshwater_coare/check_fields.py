"""Independent application checks for native freshwater COARE output at one archived clock."""
from datetime import datetime,timedelta,timezone
import numpy as np

def axis(d):
    v=d["Itime"];epoch=datetime.fromisoformat(v.units[11:].strip().replace("/","-")).replace(tzinfo=timezone.utc)
    return [epoch+timedelta(days=int(a),milliseconds=int(b)) for a,b in zip(v[:],d["Itime2"][:])]
def field(ds,k,i):
    v=np.ma.asarray(ds[k][i])
    if np.ma.getmaskarray(v).any() or not np.isfinite(v.data).all():raise ValueError("nonfinite or masked "+k)
    return np.asarray(v,dtype=float)
def interp(src,k,t):
    times=axis(src)
    import bisect
    p=bisect.bisect_left(times,t)
    if p<len(times) and times[p]==t:return field(src,k,p)
    if p==0 or p>=len(times):raise ValueError("unbracketed native forcing "+k)
    w=(t-times[p-1])/(times[p]-times[p-1])
    return (1-w)*field(src,k,p-1)+w*field(src,k,p)
def checks(d,i,bulk,rain,hydro):
    t=axis(d)[i]
    wet=field(d,"wet_nodes",i)==1
    if not wet.any():raise ValueError("empty wet support")
    tri=np.asarray(d["nv"][:],int).T-1;num=len(wet)
    speed=np.hypot(interp(bulk,"U10",t),interp(bulk,"V10",t))
    face=np.bincount(tri.ravel(),weights=np.repeat(speed,3),minlength=num)/np.bincount(tri.ravel(),minlength=num)
    hs,hl,lw,sw,net=[field(d,k,i) for k in ("sensible_heat_flux","latent_heat_flux","long_wave","short_wave","net_heat_flux")]
    sst=field(d,"coare_input_sst",i);raw=field(d,"coare_raw_rain",i);vap=field(d,"coare_signed_vapor",i)
    def check(name,actual,expected,support,atol,rtol):
        au=actual[support];ex=expected[support];err=float(np.max(np.abs(au-ex)))
        if not np.allclose(au,ex,atol=atol,rtol=rtol):raise ValueError(f"{name}: {err}, expected range {ex.min()}..{ex.max()}")
        return err
    results={}
    pairs=[("coare_wind_speed",field(d,"coare_wind_speed",i),face,wet,2e-5,1e-5),
           ("short_wave",sw,interp(bulk,"short_wave",t),wet,1e-4,2e-6),
           ("heat_addition_Wm2",net,sw+hs+hl+lw,wet,.0001,2e-6),
           ("latent_mass_Wm2",hl,1000*(2.501-.00237*sst)*1e6*vap,wet,.0001,2e-6),
           ("raw_rain",raw,interp(rain,"precip",t),wet,1e-12,2e-6),
           ("rain_and_dew",field(d,"precip",i),raw+np.maximum(vap,0),wet,1e-12,2e-6),
           ("evap",field(d,"evap",i),np.minimum(vap,0),wet,1e-12,2e-6),
           ("hydro_mslp",field(d,"atmos_press",i),interp(hydro,"air_pressure",t),wet,.02,2e-6)]
    for k in ("uwind_stress","vwind_stress"):pairs.append((k,field(d,k,i),interp(hydro,k,t),slice(None),1.e-7,2.e-6))
    for k,a,b,m,atol,rtol in pairs:results[k]=check(k,a,b,m,atol,rtol)
    return dict(max_abs_error=results,wet_nodes=int(wet.sum()),evaporating_wet_nodes=int(np.sum((vap<0)&wet)),
                condensing_wet_nodes=int(np.sum((vap>0)&wet)),minimum_exchange_Wm2=float(hl[wet].min()),maximum_exchange_Wm2=float(hl[wet].max()))
