"""Build a Cartesian FVCOM_MP estuary hotstart from one native, wet-triangle OFS nowcast."""
import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import netCDF4 as nc
import numpy as np
from pyproj import CRS, Geod, Transformer

from estuary_remap import containing, dens2, edges, extend, face_edges, fit_elevation, open_component, vertical, wet_state
from freshwater import reference_columns
from lake_initial import MJD, SOURCE_COMMIT, barotropic, complete, native_times, save, sha, utc, verify_reader_layout


def stats(values):
    a=np.asarray(values,dtype="f8")
    if not a.size or not np.isfinite(a).all():
        raise ValueError("Missing or nonfinite values")
    return dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()),mean=float(a.mean()),
                rms=float(np.sqrt(np.mean(a*a))))


def runtime_grid(t, contract, directory):
    with open(contract,encoding="utf-8") as f:
        c=json.load(f)
    crs=CRS(c["output_grid_crs"])
    if not crs.is_projected or crs.axis_info[0].unit_name!="metre":
        raise ValueError("Target must have a projected metre CRS")
    # Resolve basenames portably even when reading a Windows-originated manifest on POSIX.
    files={k:directory/str(v).replace(chr(92),"/").split("/")[-1] for k,v in c["generated_files"].items()}
    hashes={k:sha(v) for k,v in files.items()}
    for k in ("grd","dep","sig","obc","cor"):
        if hashes[k]!=c["generated_file_sha256"][k]:
            raise ValueError(f"Retained grid identity differs for {k}")
    with files["grd"].open() as f:
        n,e=(int(f.readline().split("=")[-1]) for _ in range(2))
        faces=np.loadtxt(f,dtype=int,max_rows=e)
        points=np.loadtxt(f)
    if (n,e)!=(len(t.dimensions["node"]),len(t.dimensions["nele"])) or not np.array_equal(faces[:,0],np.arange(1,e+1)) or not np.array_equal(points[:,0],np.arange(1,n+1)):
        raise ValueError("Native target ID counts or order differ")
    xy=points[:,1:3]
    tri=complete(t["nv"]).T.astype(int)-1
    if not np.array_equal(tri+1,faces[:,[1,3,2]]):
        raise ValueError("Target orientation or connectivity differs")
    target_xy=np.column_stack([complete(t[k]) for k in ("x","y")])
    # The output stores single-precision native coordinates. Allow only their rounding error.
    if np.any(abs(xy-target_xy)>np.spacing(np.abs(target_xy))*.501+1e-5):
        raise ValueError("Target projection coordinates differ from frozen GRD")
    dep=np.loadtxt(files["dep"],skiprows=1)
    if not np.array_equal(dep[:,:2],xy) or not np.array_equal(dep[:,2].astype("f4"),complete(t["h"])):
        raise ValueError("Target native bathymetry differs")
    obc=np.loadtxt(files["obc"],skiprows=1,dtype=int,ndmin=2)
    if not np.array_equal(obc[:,1],complete(t["obc_nodes"])) or not np.array_equal(obc[:,2],complete(t["obc_type"])):
        raise ValueError("Native OBC ordering or type differs")
    sigma=complete(t["siglev"])
    sigspec=files["sig"].read_text().splitlines()
    if len(sigspec)<2 or int(sigspec[0].split("=")[-1])!=len(sigma):
        raise ValueError("Native and configured sigma level counts differ")
    if sigspec[1].split("=")[-1].strip().upper()=="UNIFORM" and not np.allclose(sigma,np.linspace(0,-1,len(sigma))[:,None],atol=2e-7):
        raise ValueError("Target native interfaces are not the configured uniform grid")
    if (not np.allclose(sigma[[0,-1]],[[0],[-1]]) or not (np.diff(sigma,axis=0)<0).all()
            or not np.allclose(.5*(sigma[1:]+sigma[:-1]),complete(t["siglay"]),atol=2e-7)):
        raise ValueError("Target sigma ordering differs")
    return crs,target_xy.astype("f8"),tri,hashes


def field(v,record=None):
    a=v[:] if record is None else v[record]
    return np.ma.asarray(a).astype("f4").filled(np.nan)


def checkpoints(source,t,record,crs,xy,tri,minimum,cap,boundary=None,width=None):
    forward=Transformer.from_crs(4326,crs,always_xy=True)
    lon=(complete(source["lon"]).astype("f8")+180)%360-180
    source_xy=np.column_stack(forward.transform(lon,complete(source["lat"])))
    st,sw,labels=open_component(source,record)
    center=xy[tri].mean(axis=1)
    nm=containing(source_xy,st,np.flatnonzero(sw),xy)
    obc=complete(t["obc_nodes"]).astype(int)-1
    faces=nm[0][obc]
    components=np.unique(labels[faces[faces>=0]])
    if len(components)!=1:
        raise ValueError("Order-bound target ocean boundary does not select one wet source component")
    selected=np.flatnonzero(sw & (labels==components[0]))
    nm=containing(source_xy,st,selected,xy)
    cm=containing(source_xy,st,selected,center)
    elinks,nlinks=face_edges(tri),edges(tri)
    # Bootstrap zeta on the target topology before deriving the model water mask.
    first,_=extend(nm,xy,nlinks,np.ones(len(xy),bool))
    eta_source=field(source["zeta"],record)
    def elevation(anchor):
        return np.sum(eta_source[st[nm[0][anchor]]]*nm[1][anchor],axis=1)
    h=complete(t["h"])
    def surface(anchor):
        raw=elevation(anchor)
        return fit_elevation(raw,h,xy,tri,obc,boundary,width,minimum)[0] if width is not None else raw
    eta,wn,wc=wet_state(h,surface(first),tri,minimum)
    # Re-evaluate on the wet graph. Dry slots only retain the finite bootstrap value.
    for _ in range(8):
        active,dist=extend(nm,xy,edges(tri[wc]),wn)
        anchor=np.where(wn,active,first)
        eta,wn2,wc2=wet_state(h,surface(anchor),tri,minimum)
        if np.array_equal(wn2,wn) and np.array_equal(wc2,wc):
            break
        wn,wc=wn2,wc2
    else:
        raise ValueError("Wet-mask donor selection did not converge")
    first_c,_=extend(cm,center,elinks,np.ones(len(center),bool))
    active_c,dist_c=extend(cm,center,elinks,wc)
    anchor_c=np.where(wc,active_c,first_c)
    def report(m,d,w):
        direct=(m[0]>=0)&w
        return dict(wet_count=int(w.sum()),direct_count=int(direct.sum()),extended_count=int((w&~direct).sum()),
                    path_distance_m=stats(d[w]),direct_coverage_fraction=float(direct.sum()/w.sum()))
    if max(dist[wn].max(),dist_c[wc].max())>cap:
        raise ValueError("Wet-graph donor exceeds declared maximum extension")
    mapping=dict(node_source_cell=nm[0][anchor]+1,node_weights=nm[1][anchor],node_anchor=anchor+1,
                 node_direct=nm[0]>=0,node_distance_m=dist,cell_source_cell=cm[0][anchor_c]+1,
                 cell_anchor=anchor_c+1,cell_direct=cm[0]>=0,cell_distance_m=dist_c,wet_node=wn,wet_cell=wc)
    qa=dict(node=report(nm,dist,wn),cell=report(cm,dist_c,wc),
            ocean_direct_count=int((nm[0][obc]>=0).sum()),ocean_total=len(obc),source_connected_wet_faces=len(selected))
    if width is not None:
        raw=elevation(anchor).astype("f4")
        _,path=fit_elevation(raw,h,xy,tri,obc,boundary,width,minimum)
        delta=eta-raw
        slope=np.abs(np.diff(delta[nlinks],axis=1)[:,0])/np.linalg.norm(xy[nlinks[:,0]]-xy[nlinks[:,1]],axis=1)
        mapping.update(raw_source_zeta_m=raw,boundary_offset_m=delta,path_to_obc_m=path)
        qa["boundary_adjustment"]=dict(method="inverse-edge-length harmonic Dirichlet extension with compact cosine taper",
            maximum_path_m=width,changed_nodes=int(np.count_nonzero(delta)),offset_m=stats(delta),
            maximum_edge_gradient_m_per_m=float(slope.max()),
            raw_minus_effective_obc_m=stats(raw[obc]-np.maximum(boundary,minimum-h[obc])),
            altitude_after_minus_effective_obc_m=stats(eta[obc]-np.maximum(boundary,minimum-h[obc])))
    return st,eta,wn,wc,mapping,qa


def east_unit(crs,xy):
    lon,lat=Transformer.from_crs(crs,4326,always_xy=True).transform(*xy.T)
    lon2,lat2,_=Geod(ellps="WGS84").fwd(lon,lat,np.full(len(xy),90.),np.full(len(xy),10.))
    dest=np.column_stack(Transformer.from_crs(4326,crs,always_xy=True).transform(lon2,lat2))
    dx=dest-xy
    return (dx/np.linalg.norm(dx,axis=1)[:,None]).T


def map_fields(s,r,t,crs,xy,tri,st,eta,wn,wc,mapping,temp_request,grid_sha):
    source_h=field(s["h"])
    height=source_h+field(s["zeta"],r)
    h=complete(t["h"])
    target_depth=h+eta
    layers=complete(t["siglay"])*target_depth
    interfaces=complete(t["siglev"])*target_depth
    source_lay=field(s["siglay"])*height
    source_lev=field(s["siglev"])*height
    source_vertices=st[mapping["node_source_cell"]-1]
    weights=mapping["node_weights"]
    def nodal(name,z_source,z_target):
        values=field(s[name],r)
        result=np.zeros(z_target.shape,dtype="f8")
        for j in range(3):
            ids=source_vertices[:,j]
            result+=weights[:,j]*vertical(z_source[:,ids],values[:,ids],z_target)
        return result.astype("f4")
    # Depth below the local free surface removes a spurious geoid transformation in tracer interpolation.
    temp_mode="time_matched_native_3d"
    if "temp" in s.variables:
        if s["temp"].units.lower() not in ("degrees_c","degree_celsius","degrees_celsius","celsius"):
            raise ValueError("Unsupported source temperature units")
        temp=nodal("temp",source_lay,layers)
    elif temp_request.get("documented_3d_file"):
        p=Path(temp_request["documented_3d_file"])
        if not temp_request.get("provenance") or sha(p)!=temp_request.get("sha256"):
            raise ValueError("Documented stationary temperature requires provenance and frozen bytes")
        with nc.Dataset(p) as f:
            v=f["temp"]
            if (v.dimensions!=("siglay","node") or v.shape!=layers.shape or getattr(f,"grid_sha256","")!=grid_sha
                    or not np.array_equal(f["siglay"][:],t["siglay"][:]) or v.units.lower() not in ("degrees_c","degrees_celsius")):
                raise ValueError("Stationary 3D temperature grid or units differ")
            temp=complete(v)
        temp_mode="documented_stationary_3d"
    elif "uniform_degrees_C" in temp_request:
        temp=np.full(layers.shape,float(temp_request["uniform_degrees_C"]),dtype="f4")
        temp_mode="explicit_uniform"
    else:
        raise ValueError("No specified temperature source")
    salt=nodal("salinity",source_lay,layers)
    if not np.isfinite(temp).all() or temp.min() < -3 or temp.max()>45 or salt.min() < -0.001 or salt.max()>45:
        raise ValueError("Out-of-range temperature or salinity")
    omega=nodal("omega",source_lev,interfaces)
    omega[:,~wn]=0
    source_cell=mapping["cell_source_cell"]-1
    cell_z=source_lay[:,st[source_cell]].mean(axis=2)
    target_z=layers[:,tri].mean(axis=2)
    def cell(name):
        v=vertical(cell_z,field(s[name],r)[:,source_cell],target_z)
        v[:,~wc]=0
        return v
    if not (s["u"].standard_name.lower().startswith("eastward") and s["v"].standard_name.lower().startswith("northward")):
        raise ValueError("Source velocity is not identified as east/north")
    east,north=cell("u"),cell("v")
    ex,ey=east_unit(crs,xy[tri].mean(axis=1))
    u,v=(ex*east-ey*north).astype("f4"),(ey*east+ex*north).astype("f4")
    ua,va=barotropic(u,v,complete(t["siglev"]),target_depth,tri.T)
    rho=dens2(salt,temp)
    means,profile=reference_columns(complete(t["siglay"]),h,eta,wn,[temp,salt,rho])
    tauc=np.where(wc,field(s["tauc"],r)[source_cell],0)
    state=dict(temp=temp,salinity=salt,rho1=rho,tmean1=means[0],smean1=means[1],rmean1=means[2],
               zeta=eta,et=eta.copy(),u=u,v=v,ua=ua,va=va,omega=omega,ww=cell("ww"),tauc=tauc,
               wet_nodes=wn,wet_nodes_prev_int=wn,wet_cells=wc,wet_cells_prev_int=wc,wet_cells_prev_ext=wc,
               inundation_cells=wc & (h[tri].mean(axis=1)<0))
    bottom=np.sum(weights*height[source_vertices],axis=1)
    diag=dict(temperature_mode=temp_mode,temperature_C=stats(temp[:,wn]),salinity=stats(salt[:,wn]),
              surface_minus_bed_salinity=stats((salt[0]-salt[-1])[wn]),speed_m_s=stats(np.hypot(u,v)[:,wc]),
              target_minus_source_column_m=stats((target_depth-bottom)[wn]),
              below_donor_bed=dict(layer_nodes=int(np.count_nonzero((-layers[:,wn])>bottom[wn])),
                                   definition="barycentric donor column depth, target layer centers"),
              source_u_v_frame="true east/north",target_u_v_frame=crs.to_string(),surface_elevation_m=stats(eta[wn]))
    return state,profile,diag


def read_forcing(path,name,target,t,crs,xy):
    with nc.Dataset(path) as f:
        times=native_times(f)
        i=times.index(target)
        if name=="elevation":
            if not np.array_equal(f["obc_nodes"][:],t["obc_nodes"][:]) or f[name].dimensions!=("time","nobc"):
                raise ValueError("Boundary node metadata differs")
        else:
            if not np.array_equal(f["node_id"][:],np.arange(1,len(xy)+1)) or f[name].units.lower() not in ("pa","pascals","pascal"):
                raise ValueError("Pressure IDs or units differ")
            lon,lat=Transformer.from_crs(crs,4326,always_xy=True).transform(*xy.T)
            if np.max(abs((f["lon"][:]-lon+180)%360-180))>2e-5 or np.max(abs(f["lat"][:]-lat))>2e-5:
                raise ValueError("Pressure geographic grid differs from projected mesh")
        return complete(f[name],i)


def build(request, directory):
    with request.open(encoding="utf-8-sig") as f:
        q=json.load(f)
    if q.get("schema")!="fvcom_estuary_initial_request_v1" or q["min_depth_m"]<=0 or q["max_wet_path_m"]<0:
        raise ValueError("Invalid request schema, wetting depth or path cap")
    paths={k:(request.parent/v).resolve() for k,v in q["paths"].items()}
    hashes={k:sha(v) for k,v in paths.items() if k!="grid_directory"}
    if hashes!=q["required_sha256"]:
        raise ValueError("A frozen input has changed")
    if not q.get("vertical_reference_statement"):
        raise ValueError("Document source and target vertical reference and any transfer assumption")
    target=utc(q["utc"])
    directory.mkdir(parents=True,exist_ok=True)
    stem=q["output_basename"]
    if Path(stem).name!=stem or not stem.endswith(".nc"):
        raise ValueError("Expected a local NetCDF basename")
    products=[stem,"donor_mapping.npz","density_reference.json","initialization_manifest.json"]
    if any((directory/p).exists() for p in products):
        raise FileExistsError("Use a new output directory for a different build")
    with TemporaryDirectory(dir=directory) as staging:
        stage=Path(staging)
        with nc.Dataset(paths["native_source"]) as s,nc.Dataset(paths["restart_template"]) as t:
            verify_reader_layout(t)
            if getattr(s,"CoordinateSystem","")!="GeoReferenced":
                raise ValueError("Source requires native geographic coordinates and true east/north velocities")
            for name,dims in dict(salinity=("time","siglay","node"),zeta=("time","node"),h=("node",),
                                  u=("time","siglay","nele"),v=("time","siglay","nele"),omega=("time","siglev","node"),
                                  ww=("time","siglay","nele"),siglev=("siglev","node"),siglay=("siglay","node"),
                                  wet_nodes=("time","node"),wet_cells=("time","nele"),nbe=("three","nele"),
                                  nv=("three","nele")).items():
                if s[name].dimensions!=dims: raise ValueError(f"Unexpected source dimension order for {name}")
            if "temp" in s.variables and s["temp"].dimensions!=("time","siglay","node"):
                raise ValueError("Unexpected source temperature dimension order")
            if s["salinity"].units.lower() not in ("1e-3","psu","1","ppt"):
                raise ValueError("Unrecognized practical salinity units")
            if len(t.dimensions["time"])!=1:
                raise ValueError("Supply a single-record native restart template")
            record=native_times(s).index(target)
            source_t=nc.num2date(s["time"][record],s["time"].units,only_use_cftime_datetimes=False)
            if abs((utc(source_t)-target).total_seconds())>1:
                raise ValueError("Native source numeric and character clocks disagree")
            crs,xy,tri,grid_hashes=runtime_grid(t,paths["preconfiguration"],paths["grid_directory"])
            if grid_hashes!=q["runtime_grid_sha256"]:
                raise ValueError("Runtime DAT inputs differ from request")
            tide=read_forcing(paths["tidal_elevation"],"elevation",target,t,crs,xy)
            policy=q.get("boundary_alignment")
            width=None
            if policy is not None:
                if policy.get("type")!="type1_bundled_elevation_unramped" or set(complete(t["obc_type"]))!={1}:
                    raise ValueError("Boundary fit requires type-1 time-series elevation with IRAMP=0")
                width=float(policy["width_m"])
            st,eta,wn,wc,mapping,coverage=checkpoints(s,t,record,crs,xy,tri,q["min_depth_m"],q["max_wet_path_m"],tide,width)
            temperature=dict(q.get("temperature",{}))
            if temperature.get("documented_3d_file"):
                temperature["documented_3d_file"]=str((request.parent/temperature["documented_3d_file"]).resolve())
            state,profile,diag=map_fields(s,record,t,crs,xy,tri,st,eta,wn,wc,mapping,temperature,grid_hashes["grd"])
            p=read_forcing(paths["surface_pressure"],"air_pressure",target,t,crs,xy)
            if p.min()<80000 or p.max()>110000: raise ValueError("Expected absolute pressure in Pa")
            state["el_press"]=(-(p-101325.)/(1025.*9.81)).astype("f4")
            obc=complete(t["obc_nodes"]).astype(int)-1
            diag["initial_zeta_minus_boundary_m"]=stats(eta[obc]-tide)
            diag["initial_zeta_minus_boundary_after_removing_mean_m"]=stats(eta[obc]-tide-np.mean(eta[obc]-tide))
            diag["target_elevation_boundary_m"]=stats(tide)
            # Native SET_DEFAULT_TURB initializes interface boundaries to zero and interiors as below.
            turbulence=q["turbulence"]
            umol,pr=turbulence["vertical_mixing_coefficient"],turbulence["vertical_prandtl_number"]
            if min(umol,pr)<=0: raise ValueError("Invalid vertical mixing settings")
            for name,interior in dict(km=2*umol,kq=2*umol,kh=2*umol/pr,l=1,q2=1e-8,q2l=1e-8).items():
                a=np.zeros(complete(t["siglev"]).shape,dtype="f4")
                a[1:-1]=interior
                state[name]=a
            for name in ("viscofm","viscofh"):
                state[name]=np.zeros(t[name].shape[1:],dtype="f4")
            shutil.copyfile(paths["restart_template"],stage/stem)
            n,e=len(xy),len(tri)
            cloned=[name for name in t.variables if "time" not in t[name].dimensions and name not in state]
        clock=target.strftime("%Y-%m-%dT%H:%M:%S.%f")
        offset=target-MJD
        state.update(iint=0,time=offset.total_seconds()/86400,Itime=offset.days,
                     Itime2=(offset.seconds*1000+offset.microseconds//1000))
        with nc.Dataset(stage/stem,"r+") as out:
            for name,value in state.items():
                if not np.isfinite(value).all(): raise ValueError(f"Nonfinite startup state {name}")
                v=out[name]
                if "time" in v.dimensions: v[0]=value
                else: v[:]=value
            for name in ("Times","file_date"):
                out[name][0]=np.frombuffer(clock.encode(),dtype="S1")
            for name in list(out.ncattrs()):
                if name.endswith("_Forcing"): out.delncattr(name)
            out.title="Cross-grid estuarine baroclinic initialization"
            out.source="Native OFS analysis and retained target FVCOM restart geometry"
            out.history="Built for "+clock+" UTC"
            out.vertical_reference=q["vertical_reference_statement"]
        with nc.Dataset(stage/stem) as out,nc.Dataset(paths["restart_template"]) as template:
            verify_reader_layout(out)
            if native_times(out)!=[target] or abs(nc.date2num(target.replace(tzinfo=None),out["time"].units)-out["time"][0])>1e-5:
                raise ValueError("Inconsistent startup date encoding")
            for name,value in state.items():
                v=out[name]
                a=complete(v,0) if "time" in v.dimensions else complete(v)
                if not np.array_equal(a,np.asarray(value,dtype=v.dtype)):
                    raise ValueError(f"Round-trip differs for {name}")
            if any(not np.array_equal(complete(out[k]),complete(template[k])) for k in cloned):
                raise ValueError("A static geometry field changed")
            accounted=set(state)|set(cloned)|{"file_date","Times"}
            if accounted!=set(out.variables):
                raise ValueError(f"Unclassified restart variables {set(out.variables)-accounted}")
        np.savez_compressed(stage/products[1],**mapping)
        save(stage/products[2],profile)
        manifest=dict(schema="fvcom_estuary_initialization_v1",status="verified",source_commit=SOURCE_COMMIT,
                      target_utc=q["utc"],record=record,paths={k:str(v) for k,v in paths.items()},sha256=hashes,
                      request_sha256=sha(request),required_runtime_grid_sha256=grid_hashes,
                      nodes=n,cells=e,obc_count=len(obc),sigma_layers=state["temp"].shape[0],wet_coverage=coverage,
                      diagnostics=diag,reference_pressure_pa=101325,gravity_m_s2=9.81,rho_reference_kg_m3=1025,
                      projection=crs.to_string(),vertical_reference=q["vertical_reference_statement"],
                      source_only_vertical_endpoint="constant nearest source layer outside donor layer-center support",
                      native_geometry_copied=cloned,replaced_state=sorted(state),turbulence=turbulence,
                      startup_turb_type="default",temperature_prognostic=False,
                      products={p:sha(stage/p) for p in products[:-1]})
        save(stage/products[-1],manifest)
        for name in products: (stage/name).rename(directory/name)
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    result=build(args.request,args.output_dir)
    print(json.dumps(dict(status=result["status"],products=result["products"]),indent=2))
