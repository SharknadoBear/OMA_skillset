"""Verify native Lake Superior outputs on scratch and publish validated restart objects."""
import argparse
from bisect import bisect_left
from contextlib import ExitStack
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re

import netCDF4 as nc
import numpy as np


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src,"sha256").hexdigest()


def clock(value):
    stamp=dt.datetime.fromisoformat(value.strip().replace("/","-").replace("Z","+00:00"))
    return stamp.replace(tzinfo=dt.timezone.utc) if stamp.tzinfo is None else stamp.astimezone(dt.timezone.utc)


def iso(stamp):
    return stamp.isoformat().replace("+00:00","Z")


def read(path):
    return json.loads(path.read_bytes())


def publish_json(path, value):
    payload=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    try:
        with path.open("xb") as f:
            f.write(payload)
    except FileExistsError:
        check(path.read_bytes()==payload, "conflicting frozen receipt "+str(path))


def key(nml, name):
    matches=re.findall(r"(?im)^\s*"+name+r"\s*=\s*([^!\r\n]+)",nml)
    check(len(matches)==1, "missing unique namelist key "+name)
    return matches[0].strip().rstrip(",").strip().strip("'\x22")


def cadence(nml, name):
    unit,number=key(nml,name).split("=")
    seconds=float(number)*{"days":86400,"seconds":1}[unit.strip().lower()]
    check(seconds>0 and seconds.is_integer(), "invalid native cadence")
    return int(seconds)


def scheduled(first, end, seconds):
    return [first+dt.timedelta(seconds=i*seconds) for i in range(math.floor((end-first).total_seconds()/seconds)+1)]


def characters(row):
    return np.asarray(np.ma.filled(row,b" "),dtype="S1").tobytes().decode().strip("\x00 ")


def dimensions(ds, expected):
    check(all(len(ds.dimensions[k])==v for k,v in expected.items()),"native dimensions differ from frozen geometry")


def values(ds, name, i=0):
    var=ds[name]
    idx=tuple(i if dim=="time" else slice(None) for dim in var.dimensions)
    return np.ma.asarray(var[idx])


def axis(ds, expected=None):
    """Use native units per file, with dtype-limited UTC precision for scalar time."""
    v=ds["time"]
    numeric=np.asarray(v[:])
    check(np.isfinite(numeric).all(),"nonfinite native time")
    exact=None
    if "Itime" in ds.variables:
        days,ms=ds["Itime"],ds["Itime2"]
        check(days.units.lower().startswith("days since ") and "sec" in ms.units.lower(),"invalid native integer clock units")
        origin=clock(days.units[len("days since "):])
        exact=[origin+dt.timedelta(days=int(d),milliseconds=int(m)) for d,m in zip(days[:],ms[:])]
    if "Times" in ds.variables:
        stamps=[clock(characters(row)) for row in ds["Times"][:]]
        check(exact is None or stamps==exact,"integer and character clocks disagree")
        exact=stamps
    if expected is not None:
        check(exact is None or exact==expected,"native exact clock differs from schedule")
        exact=expected
    check(exact is not None and len(numeric)==len(exact),"missing exact clock or wrong native output count")
    check(v.units.lower().startswith(("days since ","seconds since ")),"unknown native real-time units: "+v.units)
    ideal=np.asarray(nc.date2num([x.replace(tzinfo=None) for x in exact],v.units,calendar="standard"),dtype=np.float64)
    # FVCOM station time is stored as single precision. The IINT independently resolves exact output seconds.
    tol=np.abs(np.spacing(ideal.astype(v.dtype)).astype("float64"))*.51 + 1.e-9
    check((np.abs(numeric.astype("float64")-ideal)<=tol).all(),"scalar clock disagrees with native units")
    return exact


def finite_fields(ds, names, i, ranges, masks=None):
    for name in names:
        v=values(ds,name,i)
        if masks:
            dims=ds[name].dimensions
            for dim,mask in masks.items():
                if dim in dims:
                    axis_no=[d for d in dims if d!="time"].index(dim)
                    v=np.take(v,np.flatnonzero(mask),axis=axis_no)
                    break
        check(v.size and not np.ma.getmaskarray(v).any() and np.isfinite(v.data).all(),
              "masked or nonfinite values on required domain in "+name)
        lo,hi=float(v.min()),float(v.max())
        prev=ranges.get(name,[lo,hi])
        ranges[name]=[min(prev[0],lo),max(prev[1],hi)]


def check_initial(start, ds, i):
    delta={}
    for k in ("zeta","temp","salinity","ua","va","u","v"):
        base=values(start,k)
        actual=values(ds,k,i)
        dim="node" if "node" in start[k].dimensions else "nele"
        mask=values(start,"wet_nodes" if dim=="node" else "wet_cells").astype(bool).filled(False)
        mask &= values(ds,"wet_nodes" if dim=="node" else "wet_cells",i).astype(bool).filled(False)
        check(mask.any() and base.shape==actual.shape,"incorrect support for initial "+k)
        a,b=base[...,mask],actual[...,mask]
        check(not np.ma.getmaskarray(a).any() and not np.ma.getmaskarray(b).any(),"masked initial "+k)
        err=float(np.max(np.abs(a.data-b.data)))
        delta[k]=dict(max_abs=err,rms=float(np.sqrt(np.mean((a.data.astype("float64")-b.data)**2))),wet_count=int(mask.sum()))
        check(np.allclose(a.data,b.data,rtol=3e-6,atol=2.e-6),"native zero-time field differs from hotstart: "+k+" "+str(delta[k]))
    for k in ("wet_nodes","wet_cells"):
        base=values(start,k)
        actual=values(ds,k,i)
        check(base.shape==actual.shape,"wrong zero-time mask shape")
        delta[k]={"changed":int(np.count_nonzero(base!=actual)),"total":int(base.size)}
    return delta


def inspect_water(ds,i,river,ranges,kind,mesh,*,diagnostic=False):
    dimensions(ds,mesh)
    masks={}
    for d,k in (("node","wet_nodes"),("nele","wet_cells")):
        f=values(ds,k,i)
        check(not np.ma.getmaskarray(f).any() and np.isin(f,[0,1]).all(),"invalid native wet mask "+k)
        masks[d]=np.asarray(f,dtype=bool)
    cells=np.asarray([e["cell_id"]-1 for e in river["entries"]])
    endpoints=np.asarray([np.asarray(e["edge_nodes"])-1 for e in river["entries"]])
    check(masks["nele"][cells].all() and masks["node"][endpoints].all(),"dry river source/donor edge")
    edge_cells=np.asarray(ds["nv"][:]).T[cells]-1 if "nv" in ds.variables else None
    if edge_cells is not None:
        check(all(set(nodes).issubset(tri) for nodes,tri in zip(endpoints,edge_cells)),"source/donor endpoints disagree with FVCOM connectivity")
    extra=[k for k in ("ww","omega","q2","q2l","l","km","kq","kh","rho1") if k in ds.variables]
    finite_fields(ds,("zeta","temp","salinity","ua","va","u","v",*extra),i,ranges,masks)
    if not diagnostic:
        check(ranges["temp"][0]>=-2 and ranges["temp"][1]<=40,"wet temperature outside liquid water bounds")
    check(max(abs(x) for x in ranges["salinity"])<=1e-5,"freshwater salinity has drifted")
    check(np.all((values(ds,"h")+values(ds,"zeta",i))[masks["node"]]>0),"nonpositive total depth on wet support")
    if kind=="history":
        finite_fields(ds,("net_heat_flux","short_wave"),i,ranges,masks)
    return dict(wet_nodes=int(masks["node"].sum()),wet_cells=int(masks["nele"].sum()),wet_river_edges=len(cells))


FORCING = {
    "short_wave":("HEATING_FILE","short_wave",{"W m-2"},1.e-4),
    "net_heat_flux":("HEATING_FILE","net_heat_flux",{"W m-2"},1.e-4),
    "uwind_stress":("WIND_FILE","uwind_stress",{"Pa","pascals"},1.e-7),
    "vwind_stress":("WIND_FILE","vwind_stress",{"Pa","pascals"},1.e-7),
    "atmos_press":("AIRPRESSURE_FILE","air_pressure",{"Pa","pascals"},0.02),
    "precip":("PRECIPITATION_FILE","precip",{"m s-1"},1.e-11),
    "evap":("PRECIPITATION_FILE","evap",{"m s-1"},1.e-11),
}


def compare_forcing(ds, i, stamp, sources):
    differences={}
    for native,(param,src,units,tol) in FORCING.items():
        source,times=sources[param]
        index=bisect_left(times,stamp)
        check(index<len(times) and (index>0 or times[0]==stamp),"forcing does not cover native output")
        expected=values(source,src,index)
        if times[index]!=stamp:
            weight=(stamp-times[index-1])/(times[index]-times[index-1])
            expected=weight*expected+(1-weight)*values(source,src,index-1)
        actual=values(ds,native,i)
        check(ds[native].units in units and source[src].units in units,"forcing units differ: "+native)
        check(actual.shape==expected.shape and not np.ma.getmaskarray(actual).any()
              and not np.ma.getmaskarray(expected).any(),"invalid forcing support: "+native)
        error=np.abs(actual.data.astype("float64")-expected.data)
        maximum=float(error.max())
        check(np.isfinite(error).all() and np.allclose(expected.data,actual.data,atol=tol,rtol=2.e-6),
              "native forcing differs from time-aligned input: "+native+" max_abs="+str(maximum))
        differences[native]=maximum
    return differences


def audit(a):
    folder=a.attempt.resolve()
    manifest=read(folder/"attempt_manifest.json")
    diagnostic=getattr(a,"command",None)=="lake-coare-audit"
    check(diagnostic==(manifest.get("schema")=="fvcom_coare_diagnostic_attempt_v1"),
          "Use lake-coare-audit for an opt-in COARE diagnostic and lake-audit for the direct lake workflow")
    stage=manifest["stage"]
    launch=read(folder/"launch_identity.json")
    receipt=read(a.submission)
    job=str(receipt["job_id"])
    check(receipt["workdir"]==manifest["remote_run"]==str(folder) and launch["job_id"]==job
          and launch["executable_sha256"]==manifest["executable_sha256"] and launch["mpi_ranks"]==manifest["mpi_ranks"]
          and receipt["script_sha256"]==manifest["job_sha256"]==sha(folder/"job.sbatch"),"wrong model submission")
    with a.accounting.open() as f:
        rows=list(csv.DictReader(f,delimiter="|"))
    allocation=[r for r in rows if r.get("JobID")==job]
    check(len(allocation)==1 and allocation[0]["State"]=="COMPLETED" and allocation[0]["ExitCode"]=="0:0",
          "model allocation did not complete")
    app=[r for r in rows if r.get("JobID")==job+".0"]
    nodes=manifest.get("nodes",(manifest["mpi_ranks"]+127)//128)
    check(len(app)==1 and app[0]["State"]=="COMPLETED" and app[0]["ExitCode"]=="0:0"
          and int(app[0]["NNodes"])==nodes and int(app[0]["NTasks"])==launch["mpi_ranks"],
          "native MPI step failed or used the wrong resources")
    stdout=(folder/"stdout.log").read_text(errors="replace")
    check("TADA!" in stdout and not re.search(r"(?i)\b(blowup|fatal error|segmentation fault)\b",stdout),"model log did not finish cleanly")
    check(re.fullmatch(r"[a-zA-Z0-9_]+",a.case_name),"invalid case basename")
    nmlfile=folder/(a.case_name+"_run.nml")
    nml=nmlfile.read_text()
    check(sha(nmlfile)==manifest["run_namelist_sha256"],"runtime namelist changed")
    if diagnostic:
        import fvcom_coare_output as coupled
        check(launch["site"]=="perlmutter" and launch["constraint"]=="cpu" and launch["threads_per_rank"]==1
              and receipt["account"]==allocation[0]["Account"]==manifest["account"], "Diagnostic site/account mismatch")
        field_reader=coupled.bind(a,folder,manifest,launch,nml)
        phase_lookup=coupled.phase_table()
        phases={}
        previous=None
    first,last=map(clock,(stage["start_utc"],stage["end_utc"]))
    check(clock(key(nml,"START_DATE"))==first and clock(key(nml,"END_DATE"))==last,"frozen stage clock mismatch")
    step=float(key(nml,"EXTSTEP_SECONDS"))*int(key(nml,"ISPLIT"))
    river=read(a.river)
    geometry=read(a.geometry)
    mesh={native:int(geometry[k]) for native,k in (("node","nodes"),("nele","elements"),
                                                  ("siglay","sigma_layers"),("siglev","sigma_interfaces"))}
    inputs=folder/"input"
    source=inputs/key(nml,"STARTUP_FILE")
    check(sha(source)==manifest["files"][source.name],"actual startup SHA changed")
    rf=inputs/key(nml,"RIVER_INFO_FILE")
    check(all(sha(inputs/k)==manifest["files"][k]==v for k,v in river["files"].items()),
          "river-package SHA mismatch")
    check(river.get("status")=="ready" and river["mesh_sha256"]==manifest["files"][key(nml,"GRID_FILE")],
          "signed river manifest differs from frozen geometry")
    receivers=[int(x) for x in re.findall(r"(?im)^\s*RIVER_GRID_LOCATION\s*=\s*(\d+)",rf.read_text())]
    check(receivers==[e["cell_id"] for e in river["entries"]] and len(receivers)==int(key(nml,"RIVER_NUMBER")),
          "signed river receiver order mismatch")
    hist=list(sorted(folder.glob(a.case_name+"_[0-9][0-9][0-9][0-9].nc")))
    rest=list(sorted(folder.glob(a.case_name+"_restart*.nc")))
    station=folder/(a.case_name+"_station_timeseries.nc")
    planned=[line.split()[-1] for line in (inputs/key(nml,"STATION_FILE")).read_text().splitlines()[1:] if line.strip()]
    outputs={}
    ranges={}
    masks=[]
    verified_restarts={}
    forcing_readback={}
    forcing_inputs={}
    with ExitStack() as opened:
        startup=opened.enter_context(nc.Dataset(source))
        dimensions(startup,mesh)
        tri=np.asarray(startup["nv"][:]).T-1
        coriolis=2*7.292e-5*np.sin(np.deg2rad(np.asarray(startup["lat"][:])[tri].mean(axis=1)))
        check(startup["cor"].dimensions==("nele",) and np.allclose(startup["cor"][:],coriolis,rtol=3e-6,atol=1e-9),
              "hotstart Coriolis differs from spherical mesh")
        check(startup["cc_sponge"].dimensions==("nele",) and
              (geometry["sponge_nodes"] or np.all(startup["cc_sponge"][:]==0)),"closed-lake hotstart sponge differs")
        check(int(key(nml,"IRAMP"))==0,"direct lake forcing comparison requires IRAMP=0")
        sources={}
        params=("HEATING_CALCULATE_FILE","PRECIPITATION_FILE","AIRPRESSURE_FILE") if diagnostic else sorted(set(x[0] for x in FORCING.values()))
        for param in params:
            filename=coupled.coare.value(nml,param) if diagnostic else key(nml,param)
            path=inputs/filename
            if path.name not in forcing_inputs:
                check(sha(path)==manifest["files"][path.name],"forcing input changed: "+path.name)
                source_data=opened.enter_context(nc.Dataset(path))
                stamps=axis(source_data)
                check(all(x<y for x,y in zip(stamps,stamps[1:])),"non-increasing forcing clock")
                forcing_inputs[path.name]=(source_data,stamps)
            sources[param]=forcing_inputs[path.name]
        axis(startup,[first])
        initial_iint=int(values(startup,"iint"))
        def iterations(ds, stamps):
            expected=[initial_iint+round((t-first).total_seconds()/step) for t in stamps]
            check(np.array_equal(ds["iint"][:],expected),"native iteration clock inconsistent with actual startup IINT")
        for kind,files,start_key,interval in (("history",hist,"NC_FIRST_OUT","NC_OUT_INTERVAL"),
                                              ("restart",rest,"RST_FIRST_OUT","RST_OUT_INTERVAL")):
            expected=scheduled(clock(key(nml,start_key)),last,cadence(nml,interval))
            seen=[]
            for path in files:
                with nc.Dataset(path) as ds:
                    stamps=axis(ds)
                    iterations(ds,stamps)
                    check(stamps,"empty native stack")
                    digest=sha(path)
                    outputs[path.name]=dict(sha256=digest,count=len(stamps),first=iso(stamps[0]),last=iso(stamps[-1]))
                    for i,t in enumerate(stamps):
                        masks.append(inspect_water(ds,i,river,ranges,kind,mesh,diagnostic=diagnostic))
                        if kind=="history" and t==first:
                            initial=check_initial(startup,ds,i)
                        elif kind=="history" and diagnostic:
                            forcing_readback[iso(t)]=field_reader.checks(ds,i,*(sources[k][0] for k in params))
                        elif kind=="history":
                            forcing_readback[iso(t)]=compare_forcing(ds,i,t,sources)
                        if diagnostic and kind=="history":
                            pressure=field_reader.interp(sources["AIRPRESSURE_FILE"][0],"air_pressure",t)
                            phases[iso(t)],previous=coupled.wet_phase(ds,i,phase_lookup,pressure,previous)
                        if kind=="restart":
                            check(len(stamps)==1,"downstream hotstart expects single-record restart stacks")
                            verified_restarts[t]=(path,digest)
                    seen.extend(stamps)
            check(seen==expected,"missing, out-of-order or duplicate "+kind+" record")
        with nc.Dataset(station) as ds:
            dimensions(ds,{"siglay":mesh["siglay"],"siglev":mesh["siglev"],"station":len(planned)})
            stamps=axis(ds,scheduled(first,last,cadence(nml,"OUT_INTERVAL")))
            iterations(ds,stamps)
            names=[characters(row) for row in ds["name_station"][:]]
            check(names==planned,"station identities/order changed")
            for i in range(len(stamps)):
                finite_fields(ds,("temp","salinity","zeta","ua","va","u","v"),i,ranges)
            outputs[station.name]=dict(sha256=sha(station),count=len(stamps),first=iso(stamps[0]),last=iso(stamps[-1]))
    check(last in verified_restarts,"missing final restart")
    a.output.mkdir(parents=True,exist_ok=True)
    all_receipts={}
    selected={"terminal_restart":last}
    for item in a.checkpoint:
        label,value=item.split("=",1)
        check(re.fullmatch(r"[a-zA-Z0-9_]+",label) and label not in selected,"invalid additional restart label")
        selected[label]=clock(value)
    check(all(t in verified_restarts for t in selected.values()),"requested checkpoint missing from native outputs")
    for label,stamp in selected.items():
        path,digest=verified_restarts[stamp]
        dest=a.cas/digest
        try:
            os.link(path,dest)
        except FileExistsError:
            check(os.path.samefile(path,dest) or sha(dest)==digest,"CAS digest conflict")
        record=dict(schema="fvcom_lake_restart_v1",status="verified",utc=iso(stamp),job_id=job,
                    sha256=digest,path=str(dest),native_path=str(path),attempt=manifest["name"])
        publish_json(a.output/(label+".json"),record)
        all_receipts[label]=record
    summary=dict(schema="fvcom_lake_native_audit_v1",status="verified",job_id=job,stage=stage["name"],
                 attempt=manifest["name"],attempt_sha256=sha(folder/"attempt_manifest.json"),
                 executable_sha256=manifest["executable_sha256"],startup_sha256=manifest["files"][source.name],
                 first_utc=iso(first),last_utc=iso(last),initial_iint=initial_iint,initial_field_differences=initial,
                 output_files=outputs,field_ranges=ranges,geometry_sha256=sha(a.geometry),river_manifest_sha256=sha(a.river),
                 forcing_reader=dict(status="verified" if forcing_readback else "no_post_initial_grid_record",
                    inputs_sha256={k:manifest["files"][k] for k in forcing_inputs},maximum_absolute_errors=forcing_readback,
                    zero_time="Initial diagnostics precede the first physics step"),
                 river_mask_records=len(masks),minimum_wet_river_edges=min(m["wet_river_edges"] for m in masks),
                 minimum_wet_cells=min(m["wet_cells"] for m in masks),minimum_wet_nodes=min(m["wet_nodes"] for m in masks),
                 restart_receipts=all_receipts)
    if diagnostic:
        hours=int((last-first).total_seconds()/3600)
        ledgers,points=coupled.ledgers(folder,launch["mpi_ranks"],hours,initial_iint)
        summary.update(schema="fvcom_coare_native_audit_v1",scope="diagnostic",wet_phase=phases,
            phase_screen_passed=all(not phase[k]["wet_layers"] for phase in phases.values()
                                    for k in ("substantial","persistent","gross_hot")),
            model_accounting=allocation[0],mpi_accounting=app[0],wet_field_ranges=ranges,
            forcing_reader=forcing_readback,ledger_sha256=ledgers,native_point_ledger=points,
            checkpoints={iso(t):dict(utc=iso(t),model_job_id=job,path=str(path),sha256=digest,
                         executable_sha256=launch["executable_sha256"]) for t,(path,digest) in verified_restarts.items()})
    publish_json(a.output/"native_audit.json",summary)
    return dict(status="verified",job_id=job,stage=stage["name"],records=len(masks),output=str(a.output/"native_audit.json"))


def arguments(p):
    for name in ("attempt","submission","accounting","river","cas","output","geometry"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--case-name",required=True)
    p.add_argument("--checkpoint",action="append",default=[],metavar="LABEL=UTC")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    arguments(p)
    print(json.dumps(audit(p.parse_args()),indent=2))
