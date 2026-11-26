"""Compare native states from all matched Perlmutter replicates on common wet supports."""
from contextlib import ExitStack
from datetime import timedelta
from itertools import combinations
import hashlib
import math
from pathlib import Path

import netCDF4 as nc
import numpy as np
from fvcom_benchmark import namelist_value,parse_datetime,sha256
from perlmutter_benchmark import analyze_cpu,read,require


FIELDS={"zeta":("node","m"),"temp":("node","degC"),"salinity":("node","1e-3"),
        "ua":("nele","m s-1"),"va":("nele","m s-1"),"u":("nele","m s-1"),"v":("nele","m s-1")}


def relative(base,s):
    path=Path(s)
    return path if path.is_absolute() else base/path


def wet_difference(left, right, wet):
    require(left.shape==right.shape and left.shape[-1]==len(wet) and wet.any(),"incompatible wet geometry")
    a,b=left[...,wet],right[...,wet]
    require(not np.ma.getmaskarray(a).any() and not np.ma.getmaskarray(b).any(),"masked value on common wet support")
    delta=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    require(np.isfinite(delta).all(),"nonfinite wet-state difference")
    squared=float(np.einsum("i,i->",delta.ravel(),delta.ravel()))
    return {"count":int(delta.size),"rms":math.sqrt(squared/delta.size),"max_abs":float(np.max(np.abs(delta)))},squared


def grid_digest(ds):
    digest=hashlib.sha256()
    for name in ("nv","lon","lat","h","siglev"):
        value=np.ascontiguousarray(ds[name][:])
        digest.update(str((name,value.shape,value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def native_times(ds):
    days=ds["Itime"]
    require(days.units.lower().startswith("days since "),"invalid FVCOM day origin")
    origin=parse_datetime(days.units[len("days since "):])
    return [origin+timedelta(days=int(d),milliseconds=int(m)) for d,m in zip(days[:],ds["Itime2"][:])]


def evaluate(plan_path, records, inventory_path):
    plan=read(plan_path)
    science=analyze_cpu(plan,records,sha256(plan_path))
    inventory=read(inventory_path)
    rows={r["stage"]:r for r in records}
    require(set(inventory["members"])==set(rows),"expected all nine native benchmark attempts")
    base=inventory_path.parent
    stamps={}
    hashes={}
    grid_hashes=set()
    names=sorted(rows)
    pairs=list(combinations(names,2))
    per_pair={a+"__"+b:{"left":a,"right":b,"by_utc":{},"pooled":{},"mask_disagreements":{}} for a,b in pairs}
    with ExitStack() as files:
        for name,row in rows.items():
            meta=inventory["members"][name]
            folder=relative(base,meta["attempt"])
            audit_path=relative(base,meta["audit"])
            audit=read(audit_path)
            attempt=read(folder/"attempt_manifest.json")
            require(audit["status"]=="verified" and audit["attempt"]==attempt["name"]==row["attempt"]
                    and audit["job_id"]==row["job_id"] and attempt["stage"]["name"]==name
                    and audit["attempt_sha256"]==sha256(folder/"attempt_manifest.json")==row["evidence_sha256"]["attempt"]
                    and audit["executable_sha256"]==row["executable_sha256"]
                    and audit["startup_sha256"]==row["restart_sha256"],"benchmark output lineage mismatch: "+name)
            nmls=list(folder.glob("*_run.nml"))
            require(len(nmls)==1 and sha256(nmls[0])==row["namelist_sha256"],"namelist mismatch")
            nml=nmls[0].read_text()
            case=nmls[0].name[:-len("_run.nml")]
            unit,value=namelist_value(nml,"NC_OUT_INTERVAL").split("=")
            interval=timedelta(seconds=float(value)*{"seconds":1,"days":86400}[unit])
            first,last=map(parse_datetime,(plan["simulated_start"],plan["simulated_end"]))
            require(interval.total_seconds()>0 and parse_datetime(namelist_value(nml,"NC_FIRST_OUT"))==first,
                    "invalid benchmark grid clock")
            expected=[first+i*interval for i in range((last-first)//interval+1)]
            found={}
            science_files={}
            partitions=set()
            ownership={}
            for p in sorted(folder.glob(case+"_[0-9][0-9][0-9][0-9].nc")):
                h=sha256(p)
                require(audit["output_files"][p.name]["sha256"]==h,"history file changed: "+str(p))
                ds=files.enter_context(nc.Dataset(p))
                owner=np.asarray(ds["partition"][:],dtype="<i4")
                ranks=int(np.asarray(ds["nprocs"][:]).item())
                require(owner.shape==(len(ds.dimensions["nele"]),) and ranks==row["ranks"],
                        "native partition dimensions or process count mismatch")
                ids,counts=np.unique(owner,return_counts=True)
                require(ids.size and ((ids[0]>=0 and ids[-1]<ranks) or (ids[0]>=1 and ids[-1]<=ranks)),
                        "native partition owner outside the launched MPI layout")
                partitions.add(hashlib.sha256(owner.tobytes()).hexdigest())
                ownership={str(r):int(n) for r,n in zip(ids,counts)}
                if not found:
                    grid_hashes.add(grid_digest(ds))
                for i,time in enumerate(native_times(ds)):
                    require(time not in found,"duplicate history date")
                    found[time]=(ds,i)
                science_files[p.name]=h
            require(list(found)==expected,"incomplete benchmark history: "+name)
            stamps[name]=found
            require(len(partitions)==1,"element ownership changed between native stacks")
            hashes[name]={"audit_sha256":sha256(audit_path),"native_histories":science_files,
                          "partition_sha256":next(iter(partitions)),"elements_by_native_partition":ownership}
        require(len(grid_hashes)==1,"ordered mesh, bathymetry or sigma geometry differs across native histories")
        require(all(list(s)==list(stamps[names[0]]) for s in stamps.values()),"benchmark clocks differ")
        for t in stamps[names[0]]:
            instant=t.isoformat().replace("+00:00","Z")
            at={n:stamps[n][t] for n in names}
            masks={}
            for n,(ds,i) in at.items():
                masks[n]={}
                for d,var in (("node","wet_nodes"),("nele","wet_cells")):
                    raw=np.ma.asarray(ds[var][i])
                    require(np.isin(raw,[0,1]).all() and not np.ma.getmaskarray(raw).any(),"invalid wet flags")
                    masks[n][d]=np.asarray(raw,dtype=bool)
            for a,b in pairs:
                pair=per_pair[a+"__"+b]
                sample=pair["by_utc"][instant]={"fields":{},"masks":{}}
                for d in ("node","nele"):
                    count=int(np.count_nonzero(masks[a][d]!=masks[b][d]))
                    sample["masks"][d]={"disagreement_count":count,"total":len(masks[a][d]),
                                        "common_wet":int(np.count_nonzero(masks[a][d]&masks[b][d]))}
                    pair["mask_disagreements"][d]=count+pair["mask_disagreements"].get(d,0)
            for field,(d,_) in FIELDS.items():
                states={n:np.ma.asarray(ds[field][i]) for n,(ds,i) in at.items()}
                for a,b in pairs:
                    pair=per_pair[a+"__"+b]
                    metrics,squared=wet_difference(states[a],states[b],masks[a][d]&masks[b][d])
                    pair["by_utc"][instant]["fields"][field]=metrics
                    pooled=pair["pooled"].setdefault(field,{"count":0,"sum_squared":0.0,"max_abs":0.0})
                    pooled["count"]+=metrics["count"]
                    pooled["sum_squared"]+=squared
                    pooled["max_abs"]=max(pooled["max_abs"],metrics["max_abs"])
    for pair in per_pair.values():
        pair["identical_element_partition"]=(hashes[pair["left"]]["partition_sha256"]
                                             ==hashes[pair["right"]]["partition_sha256"])
        for field in pair["pooled"].values():
            field["rms"]=math.sqrt(field.pop("sum_squared")/field["count"])
    return dict(schema="fvcom_benchmark_native_differences_v1",status="verified",execution_site="perlmutter",
                run_count=len(names),pair_count=len(per_pair),timestamps=len(stamps[names[0]]),
                units={k:v[1] for k,v in FIELDS.items()},
                weighting="Equal weight per common-wet native node or element, sigma layer and retained timestamp",
                lineage=science["lineage"],records_sha256=science["records_sha256"],grid_sha256=next(iter(grid_hashes)),
                distinct_partitions_per_layout={str(r):len({hashes[n]["partition_sha256"] for n in names
                                                            if rows[n]["ranks"]==r}) for r in {x["ranks"] for x in records}},
                source_hashes=hashes,comparisons=list(per_pair.values()))
