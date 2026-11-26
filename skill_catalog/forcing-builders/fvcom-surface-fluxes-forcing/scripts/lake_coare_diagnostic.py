"""Prepare water-supported HRRR inputs for the source-qualified FVCOM freshwater COARE26Z diagnostic."""
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import netCDF4 as nc
import numpy as np

from lake_surface import (coordinates, create_file, hour_axis, interpolate, native_geometry,
                          precipitation_shape, source_index, stamp)
from lake_water_support import from_checked_masks
from prepare_wind_pressure import require, utc
from surface_fluxes_core import _decode_times, sha256_file


HOUR_MS = 3_600_000
BULK = [("air_temperature", "node", "Celsius"), ("relative_humidity", "node", "percent"),
        ("air_pressure", "node", "Pa"), ("short_wave", "node", "W m-2"), ("long_wave", "node", "W m-2"),
        ("U10", "nele", "m s-1"), ("V10", "nele", "m s-1")]
RAIN = [("precip", "node", "m s-1"), ("evap", "node", "m s-1")]
# HRRR connector output name, native GRIB short name, units, vertical level.
THERMO = {"air_temperature_2m": ("TMP", "K", "2 m above ground"),
          "dew_point_temperature_2m": ("DPT", "K", "2 m above ground"),
          "specific_humidity_2m": ("SPFH", "kg kg**-1", "2 m above ground"),
          "relative_humidity_2m": ("RH", "%", "2 m above ground"),
          "surface_air_pressure": ("PRES", "Pa", "surface")}
PRODUCTS = ["air_temperature_2m", "dew_point_temperature_2m", "specific_humidity_2m", "surface_pressure",
            dict(family="wrfsfc", short_name="RH", level_text="2 m above ground", step_type="instant",
                 output_name="relative_humidity_2m")]


def saturation(t_c, p_pa):
    """Specific humidity at saturation in kg/kg using the freshwater COARE26Z QSAT26."""
    p = p_pa/100.
    e = 6.112*np.exp(17.502*t_c/(t_c+241.))*(1.0007+3.46e-6*p)
    return .622*e/(p-.378*e)


def humidity(t_k, q, p):
    require(np.all((t_k>200)&(t_k<339)) and np.all((q>0)&(q<.05)) and np.all((p>80000)&(p<112000)),
            "Invalid summer lake HRRR 2-m temperature (K), specific humidity (kg/kg), or local pressure (Pa)")
    temp = t_k-273.15
    rh = 100*q/saturation(temp,p)
    require(np.isfinite(rh).all(), "Invalid derived COARE humidity")
    return temp,rh


def dewpoint(q,p_pa):
    p = p_pa/100.
    e = q*p/(.622+.378*q)
    log_ratio = np.log(e/(6.112*(1.0007+3.46e-6*p)))
    return 241*log_ratio/(17.502-log_ratio)


def checked_thermodynamics(path, timestamp):
    digest=sha256_file(path)
    health=path.parent/"health_check.json"
    status=json.loads(health.read_bytes())
    require(status["passed"] and any(c.get("sha256")==digest for c in status["checks"]), "Unverified HRRR thermodynamic data")
    messages={}
    with nc.Dataset(path) as ds:
        require(len(ds.dimensions["time"])==1,"Expected one source analysis per hourly thermodynamics file")
        for key,(code,units,level) in THERMO.items():
            v=ds[key];records=json.loads(v.source_grib_metadata)
            require(len(records)==1,"Unexpected number of raw source records for "+key)
            m=records[0]
            require((m["idx_short_name"],m["step_type"],m["source_idx_level"],m["valid_time"],m["forecast_reference_time"],v.units)==
                    (code,"instant",level,timestamp,timestamp,units),"Wrong native HRRR metadata at "+timestamp+" "+key)
            messages[key]=m["message_sha256"]
    return dict(sha256=digest,health_sha256=sha256_file(health),source_message_sha256=messages)


def forcing_clock(args):
    core=hour_axis(args.start,args.end)
    require(len(core)>=2,"Require at least one complete bulk forcing interval")
    return np.arange(core[0]-HOUR_MS,core[-1]+2*HOUR_MS,HOUR_MS,dtype="i8")


def fetch_thermodynamics(args):
    sys.path.insert(0,str(args.skill_root/"hrrr-fetcher"/"scripts"))
    import hrrr_core as core
    require(args.bbox and core.runtime_preflight()["passed"],"Provide an HRRR bounding box and the connector's GRIB Python")
    for tick in forcing_clock(args):
        time=stamp(tick);folder=args.hrrr_thermodynamics/utc(time).strftime("%Y%m%d%H")
        folder.mkdir(parents=True,exist_ok=True)
        request=dict(schema_version="hrrr_request_v1",domain="conus",mode="analysis",start=time,end=time,
                     products=PRODUCTS,bbox=args.bbox,halo_cells=2,retain_raw_messages=True,missing_policy="error")
        if (folder/"run_manifest.json").is_file():
            require(json.loads((folder/"request.normalized.json").read_bytes())==core.normalize_request(request),
                    "Existing thermodynamic cache belongs to another request")
        else:
            plan=core.build_plan(request,folder)
            core.atomic_write_json(folder/"download_plan.json",plan)
            core.execute_plan(plan,folder)
        require(core.health_run(folder)["passed"],"Invalid HRRR acquisition at "+time)


def rain_knots(start_rates, left_rate, right_rate):
    require(np.all(start_rates>=0) and np.all(left_rate>=0) and np.all(right_rate>=0),"Negative native APCP rate")
    return precipitation_shape(start_rates,np.minimum(left_rate,start_rates),np.minimum(start_rates,right_rate))


def check_grid(ds,ref):
    require(ds["crs"].crs_wkt==ref["crs"].crs_wkt and all(np.array_equal(ds[n][:],ref[n][:]) for n in ("x","y")),
            "HRRR sources use different grid axes or map projections")


def native_clock(ds):
    integer,_=_decode_times(ds,"Itime",True)
    character,_=_decode_times(ds,"Times",True)
    require(np.array_equal(integer,character),"Character and integer FVCOM time coordinates disagree")
    return integer


def parent_file(folder,manifest,name):
    path=folder/name
    require(Path(name).name==name and sha256_file(path)==manifest["files"][name],"Parent surface hash mismatch: "+name)
    return path


def build(args):
    require(args.parent_surface_manifest and args.coriolis_file and args.rain_context,
            "Provide the qualified parent surface manifest, native latitude file and parent rain clock")
    require(args.lake_water_mask and args.lake_seed_lonlat and args.max_water_distance_km,"Checked lake-water donor policy is required")
    require(not args.projected_crs,"This freshwater diagnostic expects spherical native east/north wind components")
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]+",args.case_name)),"Case name must be a plain basename")
    parent=json.loads(args.parent_surface_manifest.read_bytes());folder=args.parent_surface_manifest.parent
    require(parent["provider"]=="hrrr" and parent["wind_frame"]=="earth_east_north" and sha256_file(args.mesh)==parent["mesh_sha256"],
            "Parent HRRR mesh or native spherical frame differs")
    bound=hour_axis(*args.rain_context)
    declared=parent.get("rain_context_utc",[parent["start_utc"],parent["end_utc"]])
    require(np.array_equal(bound,hour_axis(*declared)),
            "Declared rain context differs from qualified parent")
    clock=forcing_clock(args)
    require(bound[0]<=clock[0]<clock[-1]<=bound[-1],"Missing parent rain-clock guards; diagnostic needs one hour on both ends")
    names=[n for n in parent["files"] if n.endswith("_atmosphere.nc")]
    require(len(names)==1,"Parent surface manifest must identify one hydrodynamic atmosphere")
    hydro=parent_file(folder,parent,names[0]);csvfile=parent_file(folder,parent,"surface_budget.csv")
    pe_names=[n for n in parent["files"] if n.endswith("_pe.nc")]
    require(len(pe_names)==1,"Parent must include native P/E with its declared ramp clock")
    pe=parent_file(folder,parent,pe_names[0])
    with csvfile.open(newline="") as src:reference={r["start_utc"]:float(r["rain_m3"]) for r in csv.DictReader(src)}
    times=hour_axis(parent["start_utc"],parent["end_utc"])
    require(times[0]<=clock[0] and clock[-1]<=times[-1],"Parent hydrodynamic data must include both guards")
    offset=int((clock[0]-times[0])//HOUR_MS)
    analysis,analyses=source_index(args.hrrr_analysis,clock)
    rates=bound[:-1][(bound[:-1]>=clock[0]-HOUR_MS)&(bound[:-1]<=clock[-1])]
    rain,rain_sources=source_index(args.hrrr_precipitation,rates,True)
    pe_clock=np.r_[np.column_stack((clock[:-1],clock[:-1]+60000,clock[1:]-60000)).ravel(),clock[-1]]
    mesh,area=native_geometry(args.mesh)
    latitudes=np.loadtxt(args.coriolis_file,skiprows=1,ndmin=2)
    require(latitudes.shape==(len(area),3) and np.allclose(latitudes[:,2],mesh.lat,rtol=0,atol=5e-5),
            "The native COARE Coriolis-file third column must contain node latitude in degrees")
    thermo_sources={};humidity_checks=[];rain_budget=[]
    out=args.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    bulk_name=args.case_name+"_bulk.nc";rain_name=args.case_name+"_rain.nc";cor_name=args.case_name+"_cor.dat"
    with TemporaryDirectory(dir=out.parent) as temporary:
        work=Path(temporary)
        with nc.Dataset(analysis[int(clock[0])][0]) as ref,nc.Dataset(hydro) as parent_atm,nc.Dataset(pe) as parent_pe:
            require(np.array_equal(native_clock(parent_atm),times),"Parent atmospheric clock differs from manifest")
            parent_ts=native_clock(parent_pe)
            require(np.array_equal(parent_ts[(parent_ts>=pe_clock[0])&(parent_ts<=pe_clock[-1])],pe_clock),
                    "Raw rain knots differ from actual parent P/E clock")
            axes,cells,_,nodes=coordinates(ref,"hrrr",mesh)
            nw=from_checked_masks(args,ref,axes,nodes);ew=from_checked_masks(args,ref,axes,cells)
            require(all(nw.info[k]==parent["lake_water_support"][k] for k in
                        ("component_sha256","binary_water_sha256","fully_supported","renormalized","nearest_same_lake","maximum_used_distance_m")),
                    "Parent hydrometeorology uses another lake or shoreline mask")
            with create_file(work/bulk_name,mesh,clock,BULK,"earth_east_north") as ds:
                ds.pressure_reference="local_absolute_surface_pressure"
                ds["air_pressure"].long_name="absolute HRRR local surface pressure for freshwater COARE"
                ds["relative_humidity"].long_name="percent of local COARE26Z saturation specific humidity derived from observed HRRR 2 m specific humidity"
                ds["short_wave"].long_name="net solar radiation entering lake from downwelling minus reflected shortwave"
                ds["long_wave"].long_name="incoming downwelling longwave radiation"
                for k,tick in enumerate(clock):
                    time=stamp(tick);mp=args.hrrr_thermodynamics/utc(time).strftime("%Y%m%d%H")/"hrrr_fields.nc"
                    thermo_sources[time]=checked_thermodynamics(mp,time)
                    path,i=analysis[int(tick)]
                    with nc.Dataset(mp) as m,nc.Dataset(path) as h:
                        check_grid(m,ref);check_grid(h,ref)
                        def node(key,src,idx):return interpolate(src,key,idx,nodes,axes,water_support=nw)
                        tk,td,q,rh2,p=[node(key,m,0) for key in THERMO]
                        temp,rh=humidity(tk,q,p)
                        sw=node("surface_downwelling_shortwave_flux",h,i)-node("upward_shortwave_flux",h,i)
                        values=dict(air_temperature=temp,relative_humidity=rh,air_pressure=p,short_wave=sw,
                            long_wave=node("surface_downwelling_longwave_flux",h,i),
                            U10=interpolate(h,"eastward_wind_10m",i,cells,axes,water_support=ew),
                            V10=interpolate(h,"northward_wind_10m",i,cells,axes,water_support=ew))
                        for key,_,_ in BULK:
                            require(np.isfinite(values[key]).all(),"Nonfinite FVCOM "+key)
                            ds[key][k]=values[key]
                        require(np.array_equal(sw.astype("f4"),parent_atm["short_wave"][k+offset]),"Parent net solar flux differs")
                        mslp=interpolate(h,"air_pressure_at_mean_sea_level",i,nodes,axes)
                        require(np.array_equal(mslp.astype("f4"),parent_atm["air_pressure"][k+offset]),"Parent hydrodynamic MSLP differs")
                        stored=lambda key:np.asarray(ds[key][k],dtype=float)
                        qerr=np.max(abs(q-stored("relative_humidity")*.01*saturation(stored("air_temperature"),stored("air_pressure"))))
                        require(qerr<2e-8 and np.all((td>200)&(td<339)),"Failed native humidity replay or dewpoint unit check")
                        humidity_checks.append(dict(utc=time,Q2_max_abs_error_kgkg=float(qerr),supersaturated_nodes=int((rh>100).sum()),
                            RH_percent_range=[float(rh.min()),float(rh.max())],native_RH2_max_difference_pct=float(np.max(abs(rh-rh2))),
                            dewpoint_Q2_minus_native_max_abs_C=float(np.max(abs(dewpoint(q,p)-(td-273.15))))))
            def rate(t):
                p,i=rain[int(t)]
                with nc.Dataset(p) as f:
                    check_grid(f,ref)
                    return interpolate(f,"total_precipitation",i,nodes,axes,True)/(1000.*3600.)
            with create_file(work/rain_name,mesh,pe_clock,RAIN,"earth_east_north") as ds:
                ds["precip"].long_name="raw HRRR hourly accumulated rain reconstructed on original 0,1,59,60-minute knots"
                ds["evap"].long_name="zero placeholder; native signed water vapor comes from freshwater bulk latent heat"
                current=rate(clock[0]);previous=rate(clock[0]-HOUR_MS) if int(clock[0]-HOUR_MS) in rain else current
                for k,tick in enumerate(clock[:-1]):
                    following=rate(tick+HOUR_MS) if int(tick+HOUR_MS) in rain else current
                    for j,r in enumerate(rain_knots(current,previous,following)):
                        ds["precip"][3*k+j]=r;ds["evap"][3*k+j]=0.
                    integrated=np.trapezoid(np.asarray(ds["precip"][3*k:3*k+4],float),x=[0,60,3540,3600],axis=0)
                    expected=float(3600*current@area);readback=float(integrated@area)
                    require(np.max(abs(integrated-3600*current))<1e-8 and np.isclose(reference[stamp(tick)],expected,rtol=5e-13,atol=1e-5),
                            "APCP water-volume closure or parent rain mismatch")
                    rain_budget.append(dict(start_utc=stamp(tick),end_utc=stamp(tick+HOUR_MS),raw_rain_m3=expected,reader_rain_m3=readback))
                    previous,current=current,following
        shutil.copyfile(args.coriolis_file,work/cor_name)
        manifest=dict(schema="fvcom_lake_hrrr_coare26z_diagnostic_v1",status="prepared_for_native_qualification",provider="hrrr",
            model_start_utc=args.start,model_end_utc=args.end,guarded_utc=[stamp(clock[0]),stamp(clock[-1])],rain_context=list(args.rain_context),
            mesh_sha256=sha256_file(args.mesh),parent_surface_manifest_sha256=sha256_file(args.parent_surface_manifest),
            hydrodynamic_reference=dict(file=names[0],sha256=parent["files"][names[0]],used_fields=["air_pressure","uwind_stress","vwind_stress"]),
            parent_rain_clock_sha256=parent["files"][pe_names[0]],signed_vapor="Native heat-coupled HLB/(1000*Lv(preheat SST)), positive condensation and negative evaporation",
            source_receipts=dict(thermodynamics=thermo_sources,analysis=analyses,rain=rain_sources),
            bulk_support=dict(nodes={**nw.info,"fields":"2-m thermodynamics, local pressure, incoming LW and net SW"},
                              elements={**ew.info,"fields":"earth-relative U10 and V10 for heat exchange"}),
            humidity_checks=humidity_checks,raw_rain_budget=rain_budget,files={n:sha256_file(work/n) for n in (bulk_name,rain_name,cor_name)})
        (work/"coare_diagnostic_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        out.mkdir(parents=True,exist_ok=True)
        products=[work/n for n in (bulk_name,rain_name,cor_name,"coare_diagnostic_manifest.json")]
        for f in products:
            dest=out/f.name
            if dest.exists():
                require(sha256_file(dest)==sha256_file(f),"Different diagnostic output already exists: "+str(dest))
        for f in products:
            try:
                os.link(f,out/f.name)
            except FileExistsError:
                require(sha256_file(out/f.name)==sha256_file(f),"Concurrent diagnostic output differs: "+f.name)
    return manifest
