"""Zero-salinity branches of FVCOM_MP f8e4b50 eqs_of_state.F, in native anomaly units."""
import numpy as np


def gravity(latitude):
    r=np.deg2rad(np.asarray(latitude,dtype="f8"))
    return (9.780327*(1+0.0053024*np.sin(r)**2-0.0000058*np.sin(2*r)**2)).astype("f4")


def gradient(t,p):
    # Bryden freshwater adiabatic temperature gradient; salinity departure from 35 is -35.
    ds=-35.
    return (((-2.1687e-16*t+1.8676e-14)*t-4.6206e-13)*p+
            ((2.7759e-12*t-1.1351e-10)*ds+((-5.4481e-14*t+8.733e-12)*t-6.7795e-10)*t+1.8741e-8))*p+(
            -4.2393e-8*t+1.8932e-6)*ds+((6.6228e-10*t-6.836e-8)*t+8.5258e-6)*t+3.5803e-5


def potential(t,p):
    h=-p
    q=h*gradient(t,p)
    t=t+.5*q
    p=p+.5*h
    x=h*gradient(t,p)
    t=t+.29289322*(x-q)
    q=.58578644*x+.121320344*q
    x=h*gradient(t,p)
    t=t+1.707106781*(x-q)
    q=3.414213562*x-4.121320344*q
    x=h*gradient(t,p+.5*h)
    return t+(x-2*q)/6


def anomaly(temperature,pressure_dbar):
    t=np.asarray(temperature,dtype="f4")
    p=np.asarray(pressure_dbar,dtype="f4")
    if not np.isfinite(t).all() or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError("Freshwater EOS requires finite temperature and nonnegative pressure")
    t=potential(t,p)
    p=p/10
    sig=((((6.536332e-9*t-1.120083e-6)*t+1.001685e-4)*t-9.095290e-3)*t+6.793952e-2)*t-28.263737
    v=np.float32(1)/np.float32(1028.1063)
    sva=-sig*v/(1028.1063+sig)
    bw=(5.2787e-8*t-6.12293e-6)*t+3.47718e-5
    aw=((-5.77905e-7*t+1.16092e-4)*t+1.43713e-3)*t-.1194975
    kw=(((-5.155288e-5*t+1.360477e-2)*t-2.327105)*t+148.4206)*t-1930.06
    dk=(bw*p+aw)*p+kw
    k35=(5.03217e-5*p+3.359406)*p+21582.27
    gamma=p/k35
    pk=1-gamma
    sva=sva*pk+(v+sva)*p*dk/(k35*(k35+dk))
    ref_volume=v*pk
    sigma=28.106331+gamma/ref_volume-sva/(ref_volume*(ref_volume+sva))
    return np.where(p==0,sig+28.106331,sigma)*np.float32(.001)


def reference_columns(sigma,depth,zeta,wet,fields,levels=601):
    """Average across wet nodes on fixed geometric heights, then map the common reference back to native layers."""
    indices=np.flatnonzero(wet)
    top=float(zeta[indices].max())
    bottom=float(depth[indices].max())
    grid=np.linspace(-bottom,top,levels)
    # All sigma columns are independent. Interpolate in finite blocks and never average outside a physical water column.
    z=sigma*(depth+zeta)[None,:]+zeta[None,:]
    counts=np.zeros(levels,dtype=int)
    sums=[np.zeros(levels,dtype=float) for _ in fields]
    for j in indices:
        active=(grid>=-depth[j])&(grid<=zeta[j])
        counts[active]+=1
        for values,total in zip(fields,sums):
            total[active]+=np.interp(grid[active],z[::-1,j],values[::-1,j])
    valid=counts>0
    if valid.sum()<2: raise ValueError("Insufficient water columns for fixed-height reference profile")
    profiles=[total[valid]/counts[valid] for total in sums]
    mapped=[np.interp(z,grid[valid],v).astype("f4") for v in profiles]
    diagnostic=dict(z_m=grid[valid].tolist(),wet_nodes=counts[valid].tolist(),values=[v.tolist() for v in profiles],
                    method="common arithmetic wet-node mean at fixed geometric heights; nearest layer in surface/bed half layers")
    return mapped,diagnostic
