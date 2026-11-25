"""Build an FVCOM_MP freshwater hotstart from a verified spherical, same-grid OFS analysis."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import netCDF4 as nc
import numpy as np

from freshwater import anomaly, gravity, reference_columns

GEOMETRY = ("lon", "lat", "x", "y", "lonc", "latc", "xc", "yc", "nv", "h", "siglay", "siglev")
FLAGS = ("wet_nodes", "wet_cells", "wet_nodes_prev_int", "wet_cells_prev_int", "wet_cells_prev_ext")
DIRECT = ("temp", "salinity", "zeta", "u", "v", "omega", "q2", "q2l", "l", "km", "kh", "kq", *FLAGS)
MJD = datetime(1858, 11, 17, tzinfo=timezone.utc)
SOURCE_COMMIT = "f8e4b50d4fd51409febcefc763cbc5dca11870e2"


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def utc(value):
    t = datetime.fromisoformat(str(value).replace("/", "-").replace("Z", "+00:00"))
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def native_times(ds):
    integers=None
    if "Itime" in ds.variables and "Itime2" in ds.variables:
        days=ds["Itime"]
        if not days.units.startswith("days since "): raise ValueError("Unrecognized FVCOM integer date reference")
        origin=utc(days.units.removeprefix("days since "))
        integers=[origin+timedelta(days=int(d),milliseconds=int(ms)) for d,ms in zip(days[:],ds["Itime2"][:])]
    if "Times" in ds.variables:
        times = [utc(b"".join(row).decode().strip(" \x00")) for row in np.ma.asarray(ds["Times"][:]).filled(b" ") if len(row)]
    elif integers is not None:
        times=integers
    else:
        values = nc.num2date(ds["time"][:], ds["time"].units)
        times = [datetime(t.year,t.month,t.day,t.hour,t.minute,t.second,t.microsecond,tzinfo=timezone.utc) for t in values]
    if len(times) != len(set(times)) or times != sorted(times):
        raise ValueError("Invalid native clock")
    if integers is not None and times!=integers:
        raise ValueError("Native integer and character clocks differ")
    return times


def complete(var, index=slice(None)):
    array = np.ma.asarray(var[index])
    if np.ma.getmaskarray(array).any() or not np.isfinite(array.data).all():
        raise ValueError(f"Invalid or unrepresentable source values in {var.name}")
    return np.asarray(array.data)


def check_geometry(ds, ref, contract, directory):
    for name in GEOMETRY:
        if ds[name].dimensions!=ref[name].dimensions or not np.array_equal(complete(ds[name]), complete(ref[name])):
            raise ValueError(f"Source and accepted reference differ at native {name}")
    n, e, k = (len(ds.dimensions[key]) for key in ("node","nele","siglay"))
    if (n,e,k,len(ds.dimensions["siglev"]))!=(contract["nodes"],contract["elements"],contract["sigma_layers"],contract["sigma_interfaces"]):
        raise ValueError("Dimensions differ from the runtime grid")
    if contract["geometry"]!="spherical" or contract["grid_file_units"]!="degrees" or ds.CoordinateSystem!="GeoReferenced":
        raise ValueError("Expected spherical native grid and earth-frame vectors")
    for key in ("grd","dep","sig"):
        entry=contract["files"][key]
        if sha(directory/entry["name"])!=entry["sha256"]:
            raise ValueError(f"Frozen runtime {key} changed")
    with (directory/contract["files"]["grd"]["name"]).open() as stream:
        counts=[int(stream.readline().split("=")[-1]) for _ in range(2)]
        if counts!=[n,e]: raise ValueError("Runtime grid counts differ")
        tri=np.loadtxt(stream,max_rows=e,dtype=int,ndmin=2)
        coords=np.loadtxt(stream,ndmin=2)
    perm=np.asarray(contract["ascii_to_native_nv_one_based"])-1
    if (not np.array_equal(tri[:,0],np.arange(1,e+1)) or not np.array_equal(coords[:,0],np.arange(1,n+1))
            or not np.array_equal(tri[:,1:4][:,perm].T,ds["nv"][:])):
        raise ValueError("Runtime element/node ordering differs")
    dep=np.loadtxt(directory/contract["files"]["dep"]["name"],skiprows=1)
    # Archive longitude was stored as single precision after adding 360 degrees.
    for xy in (coords[:,1:3],dep[:,:2]):
        delta=(xy[:,0]-np.asarray(ds["lon"][:],dtype="f8")+180)%360-180
        if np.max(abs(delta))>2e-5 or np.max(abs(xy[:,1]-ds["lat"][:]))>4e-6:
            raise ValueError("Runtime longitude/latitude differ")
    if not np.array_equal(dep[:,2].astype(ds["h"].dtype),ds["h"][:]):
        raise ValueError("Runtime positive-down bathymetry differs")
    interfaces=complete(ds["siglev"])
    layers=complete(ds["siglay"])
    levels=np.asarray(contract["sigma_interface_values"])[:,None]
    if (not np.allclose(interfaces,levels,atol=1e-7,rtol=0) or not np.all(np.diff(interfaces,axis=0)<0)
            or not np.allclose(interfaces[[0,-1]],[[0],[-1]]) or not np.allclose(layers,(interfaces[1:]+interfaces[:-1])/2,atol=1e-7,rtol=0)):
        raise ValueError("Sigma interfaces or midpoint order differ from runtime")
    if not all(abs(v)<1e-6 for v in (interfaces[0,0],interfaces[-1,0]+1)): raise ValueError("Bad sigma endpoints")
    return {key:contract["files"][key]["sha256"] for key in ("grd","dep","sig")}


def barotropic(u,v,siglev,depth,nv):
    total=depth[nv].mean(axis=0,dtype="f8")
    if np.any(total<=0): raise ValueError("Nonpositive cell water column")
    thickness=-np.diff(siglev,axis=0)*depth
    weights=sum(thickness[:,vertices].astype("f8") for vertices in nv)/(3*total)
    if not np.allclose(weights.sum(axis=0),1,rtol=1e-6) or np.any(weights<=0):
        raise ValueError("Invalid element layer thickness")
    return tuple(np.sum(field*weights,axis=0).astype("f4") for field in (u,v))


def pressure_at(path, target, ds, record, grid_sha):
    if path is None:
        return complete(ds["atmos_press"],record),dict(mode="archived_analysis_reader_probe",variable="atmos_press")
    with nc.Dataset(path) as p:
        times=native_times(p)
        if target not in times: raise ValueError("No exact runtime pressure sample")
        v=p["air_pressure"]
        if v.dimensions!=("time","node") or v.units.lower() not in ("pa","pascals","pascal"):
            raise ValueError("Expected absolute air_pressure(time,node) in pascals")
        if "lon" in p.variables and "lat" in p.variables:
            delta=(np.asarray(p["lon"][:],dtype="f8")-ds["lon"][:]+180)%360-180
            if np.max(abs(delta))>2e-5 or np.max(abs(p["lat"][:]-ds["lat"][:]))>4e-6:
                raise ValueError("Pressure node coordinates differ")
        elif getattr(p,"grid_sha256","")!=grid_sha:
            raise ValueError("Pressure requires ordered lon/lat or grid_sha256 matching runtime grd")
        if "node_id" in p.variables and not np.array_equal(p["node_id"][:],np.arange(1,len(ds.dimensions["node"])+1)):
            raise ValueError("Pressure node order differs")
        data=complete(v,times.index(target))
        if len(data)!=len(ds.dimensions["node"]): raise ValueError("Pressure node count differs")
        return data,dict(mode="runtime_forcing",variable="air_pressure",path=str(path.resolve()),sha256=sha(path),record=times.index(target),
                         sample_sha256=hashlib.sha256(np.asarray(data,dtype="<f4").tobytes()).hexdigest())


def state(ds, record, pressure):
    fields={name:complete(ds[name],record) for name in DIRECT}
    expected={"temp":("siglay","node"),"salinity":("siglay","node"),"u":("siglay","nele"),"v":("siglay","nele")}
    expected.update({name:("siglev","node") for name in ("omega","q2","q2l","l","km","kh","kq")})
    expected.update({name:("node" if name in ("zeta","wet_nodes","wet_nodes_prev_int") else "nele",) for name in ("zeta",*FLAGS)})
    for name,dims in expected.items():
        if ds[name].dimensions!=("time",*dims): raise ValueError(f"Invalid native axes for {name}")
    if not (ds["u"].standard_name.lower().startswith("eastward") and ds["v"].standard_name.lower().startswith("northward")):
        raise ValueError("Native spherical east/north velocities required")
    if any(not np.isin(fields[name],[0,1]).all() for name in FLAGS): raise ValueError("Invalid wet/dry flags")
    if np.any(fields["salinity"]!=0): raise ValueError("DENS1 freshwater branch requires native zero salinity")
    if not np.all((pressure>80000)&(pressure<110000)): raise ValueError("Pressure outside absolute Pa range")
    wet=fields["wet_nodes"].astype(bool)
    h=complete(ds["h"])
    depth=h+fields["zeta"]
    if np.any(depth[wet]<=0): raise ValueError("Nonpositive wet-node water column")
    sig=complete(ds["siglay"])
    g=gravity(complete(ds["lat"]))
    fields["ua"],fields["va"]=barotropic(fields["u"],fields["v"],complete(ds["siglev"]),depth,complete(ds["nv"])-1)
    fields["et"]=fields["zeta"].copy()
    fields["el_press"]=-(pressure-101325)/(1025*g)
    p=-g*1.025*sig*depth*.1
    fields["rho1"]=anomaly(fields["temp"],p)
    mean,profile=reference_columns(sig,h,fields["zeta"],wet,[fields["temp"],fields["rho1"]])
    fields["tmean1"],fields["rmean1"]=mean
    fields["smean1"]=fields["salinity"].copy()
    deepest=int(np.argmax(h))
    reports=dict(wet_nodes=int(wet.sum()),wet_cells=int(fields["wet_cells"].sum()),min_wet_depth_m=float(depth[wet].min()),
                 temp_range_C=[float(fields["temp"].min()),float(fields["temp"].max())],
                 salinity_range=[float(fields["salinity"].min()),float(fields["salinity"].max())],
                 max_top_bottom_temperature_difference_C=float(abs(fields["temp"][0]-fields["temp"][-1]).max()),
                 max_top_bottom_current_shear_m_s=float(np.hypot(fields["u"][0]-fields["u"][-1],fields["v"][0]-fields["v"][-1]).max()),
                 density_kg_m3=[float(1000*(1+fields["rho1"].min())),float(1000*(1+fields["rho1"].max()))],
                 deepest_node_id=deepest+1,deepest_node_temperature_C=fields["temp"][:,deepest].tolist(),
                 deepest_node_layer_elevation_m=(sig[:,deepest]*depth[deepest]+fields["zeta"][deepest]).tolist())
    return fields,profile,reports


def write_state(path,ds,fields,t):
    derived={"tmean1":("siglay","node"),"smean1":("siglay","node"),"rmean1":("siglay","node"),"rho1":("time","siglay","node"),
             "ua":("time","nele"),"va":("time","nele"),"el_press":("time","node"),"et":("time","node")}
    with nc.Dataset(path,"w",format="NETCDF3_64BIT_OFFSET") as out:
        for name in ("node","nele","siglay","siglev","three","DateStrLen","time"):
            out.createDimension(name,None if name=="time" else len(ds.dimensions[name]))
        out.CoordinateSystem=ds.CoordinateSystem
        out.CoordinateProjection=ds.CoordinateProjection
        out.title="FVCOM same-grid freshwater analysis hotstart"
        out.source=f"FVCOM_MP {SOURCE_COMMIT} reader; {ds.source} analysis"
        # Define the complete classic header before writing the first unlimited record.
        for name in (*GEOMETRY,*DIRECT):
            original=ds[name]
            target=out.createVariable(name,original.dtype,original.dimensions)
            target.setncatts({k:original.getncattr(k) for k in original.ncattrs() if k!="_FillValue"})
        for name,dims in derived.items():
            target=out.createVariable(name,"f4",dims)
            target.units={"ua":"m s-1","va":"m s-1","el_press":"m","et":"m","tmean1":"degrees_C"}.get(name,"1")
        elapsed=t-MJD
        date=t.strftime("%Y/%m/%d %H:%M:%S.%f")
        out.createVariable("iint","i4",("time",))
        clock=out.createVariable("time","f8",("time",))
        clock.units="days since 1858-11-17 00:00:00"
        clock.time_zone="UTC"
        clock.format="modified julian day (MJD)"
        for key,units in (("Itime",clock.units),("Itime2","msec since 00:00:00")):
            c=out.createVariable(key,"i4",("time",))
            c.units=units
            c.time_zone="UTC"
            c.format=clock.format
        c=out.createVariable("Times","S1",("time","DateStrLen"))
        c.time_zone="UTC"
        out["iint"][0]=0
        clock[0]=elapsed.total_seconds()/86400
        out["Itime"][0]=elapsed.days
        out["Itime2"][0]=elapsed.seconds*1000+elapsed.microseconds//1000
        c[0]=np.asarray(list(date),dtype="S1")
        for name in GEOMETRY: out[name][:]=ds[name][:]
        for name,values in fields.items():
            target=out[name]
            if "time" in target.dimensions: target[0]=values
            else: target[:]=values


def build(source,reference,contract_path,pressure,output):
    receipt=json.loads(source.read_text()); contract=json.loads(contract_path.read_text())
    path=Path(receipt["path"]); target=utc(receipt["target_utc"]); record=receipt["record"]
    if receipt["status"]!="ready" or receipt["object"]["product"]!="fields" or sha(path)!=receipt["sha256"]:
        raise ValueError("Native snapshot receipt or bytes changed")
    if contract["status"]!="ready" or sha(reference)!=contract["source_sha256"]:
        raise ValueError("Native grid reference differs from accepted delivery")
    output.mkdir(parents=True,exist_ok=True)
    if (output/"initialization_manifest.json").exists(): raise ValueError("Use a fresh output directory for each immutable version")
    with TemporaryDirectory(dir=output) as tmp:
        stage=Path(tmp); file=stage/"lake_initial.nc"
        with nc.Dataset(path) as ds,nc.Dataset(reference) as ref:
            if native_times(ds)[record]!=target: raise ValueError("Snapshot exact UTC no longer matches")
            hashes=check_geometry(ds,ref,contract,contract_path.parent)
            data,pressure_info=pressure_at(pressure,target,ds,record,hashes["grd"])
            fields,profile,reports=state(ds,record,data)
            write_state(file,ds,fields,target)
            with nc.Dataset(file) as out:
                if native_times(out)!=[target] or out["iint"][0]!=0: raise ValueError("New run clock mismatch")
                for name,expected in fields.items():
                    v=out[name]; array=complete(v,0 if "time" in v.dimensions else slice(None))
                    if not np.array_equal(array,expected): raise ValueError(f"Native field {name} failed readback")
                for name in GEOMETRY:
                    if not np.array_equal(out[name][:],ds[name][:]): raise ValueError("Written geometry mismatch")
            manifest=dict(schema="fvcom_initialization_v1",status="ready" if pressure else "reader_probe",mode="analysis_hotstart",
                          case_epoch_utc=receipt["target_utc"],source_commit=SOURCE_COMMIT,source_key=receipt["object"]["key"],source_sha256=receipt["sha256"],
                          source_receipt_sha256=sha(source),source_record=record,source_iint=int(ds["iint"][record]),new_iint=0,
                          dimensions={k:len(ds.dimensions[k]) for k in ("node","nele","siglay","siglev")},runtime_grid=hashes,
                          geometry="spherical",velocity_components="earth east and north, source values unchanged",runtime_grid_contract_sha256=sha(contract_path),
                          pressure=pressure_info,eos="dens1",barotropic="thickness-weighted native element velocities",
                          pressure_head="-(P-101325)/(1025*g(latitude))",density="native (rho kg m-3 - 1000)/1000 at in situ pressure",
                          eta_derivation="et equals sampled zeta at initialization; analysis contains no independent split-mode et",
                          namelist={"STARTUP_TYPE":"hotstart","STARTUP_FILE":file.name,"STARTUP_TS_TYPE":"set values",
                                    "STARTUP_UV_TYPE":"set values","STARTUP_TURB_TYPE":"set values"},checks=reports,
                          file={"path":file.name,"sha256":sha(file),"bytes":file.stat().st_size})
        save(stage/"reference_profile.json",profile)
        manifest["reference_profile_sha256"]=sha(stage/"reference_profile.json")
        save(stage/"initialization_manifest.json",manifest)
        for name in ("lake_initial.nc","reference_profile.json","initialization_manifest.json"):
            os.replace(stage/name,output/name)
    return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("source","reference","grid-contract","output"): p.add_argument("--"+arg,type=Path,required=True)
    q=p.add_mutually_exclusive_group(required=True)
    q.add_argument("--pressure",type=Path,help="Native runtime air_pressure in Pa on ordered nodes")
    q.add_argument("--probe-use-analysis-pressure",action="store_true",help="Produce a reader-only probe from archived analysis pressure")
    a=p.parse_args()
    print(json.dumps(build(a.source,a.reference,a.grid_contract,a.pressure,a.output)["file"]))
