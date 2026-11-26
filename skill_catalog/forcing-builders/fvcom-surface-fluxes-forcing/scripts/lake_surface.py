"""Build closed-lake stress, direct heat, pressure and P/E from checked HRRR connector output."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from prepare_wind_pressure import (array, checked_fields, coordinates, hour_axis, model_basis, psimple_drag,
                                   require, source_law, source_time, stamp, utc)
from surface_fluxes_core import MeshGeometry, _time_variables, sha256_file
from pyproj import CRS

LATENT_HEAT = 2.5e6  # J/kg, summer liquid water
WATER_DENSITY = 1000.0  # kg/m3
FLUXES = {"down_sw":("surface_downwelling_shortwave_flux","DSWRF"),
          "up_sw":("upward_shortwave_flux","USWRF"),
          "down_lw":("surface_downwelling_longwave_flux","DLWRF"),
          "up_lw":("upward_longwave_flux","ULWRF"),
          "sensible":("surface_sensible_heat_flux","SHTFL"),
          "latent":("surface_latent_heat_flux","LHTFL")}

def native_geometry(path):
    with nc.Dataset(path) as ds:
        lon,lat,lonc,latc=[array(ds[n]) for n in ("lon","lat","lonc","latc")]
        tri=np.asarray(ds["nv"][:],dtype=int).T-1
        require(tri.shape==(len(lonc),3) and tri.min()==0 and tri.max()==len(lon)-1,"invalid native topology")
        mesh=MeshGeometry(np.arange(1,len(lon)+1),np.arange(1,len(lonc)+1),
                          (lon+180)%360-180,lat,(lonc+180)%360-180,latc,tri,str(path))
        # FVCOM art1 is the node-centred tracer control-volume area, m².
        area=array(ds["art1"])
        require(area.shape==lon.shape and np.all(area>0),"invalid native tracer control-volume areas")
        return mesh,area

def create_file(path,mesh,times,variables,frame,provider="HRRR"):
    ds=nc.Dataset(path,"w",format="NETCDF3_64BIT_OFFSET")
    ds.source="FVCOM grid (unstructured) surface forcing"
    ds.type="FVCOM SURFACE FORCING FILE"
    ds.source_provider=provider
    ds.wind_vector_frame=frame
    if any(name=="air_pressure" for name,_,_ in variables):
        ds.pressure_reference="absolute_mean_sea_level"
    ds.createDimension("time",None)
    ds.createDimension("DateStrLen",26)
    for dim,idx,lon,lat,cn in (("node",mesh.node_ids,mesh.lon,mesh.lat,""),("nele",mesh.element_ids,mesh.lonc,mesh.latc,"c")):
        ds.createDimension(dim,len(idx))
        ds.createVariable("node_id" if dim=="node" else "element_id","i4",(dim,))[:]=idx
        for n,values,units in (("lon"+cn,lon,"degrees_east"),("lat"+cn,lat,"degrees_north")):
            var=ds.createVariable(n,"f4",(dim,));var.units=units;var[:]=values
    for name,dim,unit in variables:
        var=ds.createVariable(name,"f4",("time",dim));var.units=unit;var.grid="fvcom_grid"
    _time_variables(ds,times)
    return ds

def source_index(folder,clock,forecast=False):
    """Read and hash exactly the published connector files that supply the requested records."""
    index={}
    receipts={}
    wanted=set(map(int,clock))
    for t in sorted(wanted):
        dt=utc(stamp(t))
        key=dt.replace(hour=6*(dt.hour//6)).strftime("%Y%m%d%H")
        if key in receipts: continue
        root=folder/key
        path=root/"hrrr_fields.nc"
        health=json.loads((root/"health_check.json").read_bytes())
        digest=sha256_file(path)
        require(health["passed"] and any(c.get("sha256")==digest for c in health["checks"]),
                f"source health or object digest differs in {root}")
        with nc.Dataset(path) as ds:
            if forecast:
                rt=ds["forecast_reference_time"];fp=ds["forecast_period"];valid=ds["valid_time"]
                require(rt.units.startswith("seconds since 1970-01-01") and fp.units=="minutes" and
                        np.array_equal(fp[:],[60]) and valid.units==rt.units and
                        np.array_equal(valid[:,0],rt[:]+3600),"expected hourly HRRR f01 accumulations")
                times=np.asarray(rt[:],dtype="i8")*1000
                v=ds["total_precipitation"]
                rows=json.loads(v.source_grib_metadata)
                require(v.units in ("kg m**-2","kg m-2","mm") and len(rows)==len(times),"unexpected precipitation")
                for ts,row in zip(times,rows):
                    require(row["idx_short_name"]=="APCP" and row["step_range"]=="0-1" and row["step_type"]=="accum"
                            and utc(row["valid_time"])==utc(stamp(ts+3600000)),"ambiguous precipitation interval")
            else:
                times=source_time(ds)
                checked_fields(ds,"hrrr",times)
                for field,grib in FLUXES.values():
                    v=ds[field];rows=json.loads(v.source_grib_metadata)
                    require(v.units in ("W m**-2","W m-2") and len(rows)==len(times),"invalid surface flux units/clock")
                    for ts,row in zip(times,rows):
                        require(row["idx_short_name"]==grib and row["source_idx_level"]=="surface" and
                                row["step_type"]=="instant" and utc(row["valid_time"])==utc(stamp(ts)) and
                                utc(row["forecast_reference_time"])==utc(stamp(ts)),"expected surface f00 flux")
            for i,ts in enumerate(times):
                if int(ts) in wanted:
                    require(int(ts) not in index,"duplicate source time")
                    index[int(ts)]=(path,i)
        receipts[key]=dict(sha256=digest,health_sha256=sha256_file(root/"health_check.json"))
    require(set(index)==wanted,"missing source timestamps")
    return index,receipts

def interpolate(ds,var,i,points,axes,forecast=False):
    raw=ds[var][i,0] if forecast else ds[var][i]
    values=np.ma.asarray(raw)
    require(not np.ma.getmaskarray(values).any() and np.isfinite(values).all(),f"missing values in {var}")
    return RegularGridInterpolator(axes,np.asarray(raw),bounds_error=True)(points)

def unit_fluxes(inputs):
    sw=inputs["down_sw"]-inputs["up_sw"]
    total=sw+inputs["down_lw"]-inputs["up_lw"]-inputs["sensible"]-inputs["latent"]
    return sw,total,inputs["latent"]/(LATENT_HEAT*WATER_DENSITY)

def precipitation_shape(mean,left,right):
    """One-minute linear transitions conserve the original one-hour accumulation."""
    plateau=(3600*mean-30*(left+right))/3540
    return left,plateau,plateau,right

def build(args):
    clock=hour_axis(args.start,args.end)
    require(clock.size>=2 or args.pressure_only,"surface interval requires two endpoints")
    analyses,a_receipts=source_index(args.analysis,clock)
    if not args.pressure_only:
        rain,p_receipts=source_index(args.precipitation,clock[:-1],True)
    mesh,areas=native_geometry(args.mesh)
    source_law(args.mod_bulk)
    args.output.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=args.output) as tmp:
        temp=Path(tmp)
        path=analyses[int(clock[0])][0]
        with nc.Dataset(path) as d:
            axes,cells,_,nodes=coordinates(d,"hrrr",mesh)
            x,y=np.array(d["x"][:]),np.array(d["y"][:])
            wkt=d["crs"].crs_wkt
        def same_grid(d):
            require(np.array_equal(d["x"][:],x) and np.array_equal(d["y"][:],y) and d["crs"].crs_wkt==wkt,
                    "atmospheric source grid changed")
        frame="earth_east_north" if not args.projected_crs else "model_xy"
        c,s=(1.,0.) if not args.projected_crs else model_basis(CRS(args.projected_crs),mesh.lonc,mesh.latc)
        variables=[("air_pressure","node","Pa")]
        if not args.pressure_only: variables.extend([("uwind_stress","nele","Pa"),("vwind_stress","nele","Pa"),
                                                     ("short_wave","node","W m-2"),("net_heat_flux","node","W m-2")])
        atm=temp/("pressure.nc" if args.pressure_only else "superior_atmosphere.nc")
        with create_file(atm,mesh,clock,variables,frame) as d:
            # Keep hourly signed vapour and heat integrals outside the atmosphere file.
            if not args.pressure_only:
                vapor=np.lib.format.open_memmap(temp/"vapor.npy",mode="w+",dtype="f4",shape=(len(clock),len(areas)))
                heat=np.zeros(len(clock))
            for k,ts in enumerate(clock):
                path,i=analyses[int(ts)]
                with nc.Dataset(path) as src:
                    same_grid(src)
                    pressure=interpolate(src,"air_pressure_at_mean_sea_level",i,nodes,axes)
                    require(pressure.min()>75000 and pressure.max()<115000,"unphysical absolute MSLP")
                    d["air_pressure"][k]=pressure
                    if args.pressure_only: continue
                    u,v=[interpolate(src,n,i,cells,axes) for n in ("eastward_wind_10m","northward_wind_10m")]
                    d["uwind_stress"][k],d["vwind_stress"][k]=psimple_drag(c*u-s*v,s*u+c*v)
                    fluxes={key:interpolate(src,n,i,nodes,axes) for key,(n,_) in FLUXES.items()}
                    sw,total,ev=unit_fluxes(fluxes)
                    d["short_wave"][k],d["net_heat_flux"][k],vapor[k]=sw,total,ev
                    heat[k]=np.sum(total*areas)
        files=[atm]
        if not args.pressure_only:
            def rate(k):
                path,i=rain[int(clock[k])]
                with nc.Dataset(path) as src:
                    same_grid(src)
                    acc=interpolate(src,"total_precipitation",i,nodes,axes,True)
                    require(acc.min()>=0,"negative accumulated precipitation")
                    return acc/(3600.*WATER_DENSITY)
            pe=temp/"superior_pe.nc";budget=temp/"surface_budget.csv"
            pe_clock=np.r_[np.column_stack((clock[:-1],clock[:-1]+60000,clock[1:]-60000)).ravel(),clock[-1]]
            with create_file(pe,mesh,pe_clock,[("precip","node","m s-1"),("evap","node","m s-1")],frame) as ds, budget.open("w",newline="") as out:
                writer=csv.writer(out)
                writer.writerow(("start_utc","end_utc","rain_m3","evaporated_m3","condensed_m3","net_heat_j"))
                cur=left=rate(0)
                for k in range(len(clock)-1):
                    following=rate(k+1) if k+2<len(clock) else cur
                    right=np.minimum(cur,following)
                    E=[];C=[]
                    for j,(p,sec) in enumerate(zip(precipitation_shape(cur,left,right),(0,60,3540,3600))):
                        ev=(1.-sec/3600)*vapor[k]+(sec/3600)*vapor[k+1]
                        loss=np.maximum(ev,0);dew=np.maximum(-ev,0)
                        ds["precip"][3*k+j]=p+dew;ds["evap"][3*k+j]=-loss
                        E.append(np.sum(loss*areas));C.append(np.sum(dew*areas))
                    writer.writerow((stamp(clock[k]),stamp(clock[k+1]),3600*np.sum(cur*areas),np.trapezoid(E,x=[0,60,3540,3600]),
                                     np.trapezoid(C,x=[0,60,3540,3600]),1800*(heat[k]+heat[k+1])))
                    left,cur=right,following
            del vapor
            files.extend([pe,budget])
        for file in files:
            if file.suffix==".nc":
                with nc.Dataset(file) as d:
                    t=np.asarray(d["Itime"][:],dtype="i8")*86400000+d["Itime2"][:]
                    expected=pe_clock if file.name=="superior_pe.nc" else clock
                    require(np.array_equal(t,expected+40587*86400000),"readback clock differs")
                    for var in d.variables.values():
                        if var.dimensions[:1]==("time",) and var.dtype.kind=="f":
                            for first in range(0,len(t),12):
                                v=np.ma.asarray(var[first:first+12])
                                require(not np.ma.getmaskarray(v).any() and np.isfinite(v).all(),"nonfinite stored forcing")
        result=dict(schema="fvcom_lake_surface_v1",start_utc=stamp(clock[0]),end_utc=stamp(clock[-1]),wind_frame=frame,
                    mesh_sha256=sha256_file(args.mesh),mod_bulk_sha256=sha256_file(args.mod_bulk),provider="hrrr",
                    analyses=a_receipts,precipitation={} if args.pressure_only else p_receipts,
                    files={f.name:sha256_file(f) for f in files},area_m2=float(areas.sum()),
                    latent_heat_j_kg=LATENT_HEAT,water_density_kg_m3=WATER_DENSITY)
        manifest=temp/"surface_manifest.json"
        manifest.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
        for f in files+[manifest]:
            dst=args.output/f.name
            if dst.exists():
                require(sha256_file(dst)==sha256_file(f),"different forcing already published")
            else: f.rename(dst)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("mesh","analysis","mod-bulk","output"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--precipitation",type=Path);p.add_argument("--start",required=True);p.add_argument("--end",required=True)
    p.add_argument("--pressure-only",action="store_true");p.add_argument("--projected-crs")
    args=p.parse_args()
    if not args.pressure_only and not args.precipitation: p.error("--precipitation is required with heat and wind")
    print(json.dumps(build(args)["files"],indent=2))
