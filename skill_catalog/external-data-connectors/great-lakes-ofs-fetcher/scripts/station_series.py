"""Assemble a UTC station reference on native station/layer axes from overlapping OFS cycles."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import netCDF4 as nc
import numpy as np

from great_lakes_ofs import SYSTEMS, cache, digest, listing, save, stamp, utc


def preference(item,t):
    c=utc(item["cycle"])
    return (item["stream"]!="nowcast",abs((c-utc(t)).total_seconds()),-c.timestamp(),item["key"])


def select(receipts,start,end):
    rows={}
    overlaps=0
    for item in receipts:
        for i,t in enumerate(item["clock_utc"]):
            if start<=utc(t)<end:
                if t in rows: overlaps+=1
                if t not in rows or preference(item["object"],t)<preference(rows[t][0]["object"],t):
                    rows[t]=(item,i)
    return dict(sorted(rows.items())),overlaps


def series(system,start,end,cache_root,target,cadence=360,workers=4):
    start,end=utc(start),utc(end)
    if end<=start or cadence<=0: raise ValueError("Positive bounded time span and cadence required")
    days=[start.date()+timedelta(days=i) for i in range((end.date()-start.date()).days+2)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        cycles=[o for group in pool.map(lambda d:listing(system,d),days) for o in group if o["product"]=="stations"
                and o["stream"]=="nowcast" and utc(o["candidate_end"])>=start and utc(o["candidate_start"])<end]
        # NetCDF4/HDF5 is not thread-safe in every workstation build. Network/cache concurrency uses separate processes.
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        receipts=list(pool.map(cache,cycles,[cache_root]*len(cycles)))
    selected,overlaps=select(receipts,start,end)
    times=list(selected)
    expected=[stamp(start+timedelta(seconds=i)) for i in range(0,int((end-start).total_seconds()),cadence)]
    # Some NOAA cycles stamp their nominal 04:06 sample at 04:05:59.
    # Preserve the exact native time field and record each offset from nominal.
    if not expected or len(expected)!=len(times): raise ValueError("Station archive contains missing or extra samples")
    offsets=[(utc(actual)-utc(nominal)).total_seconds() for actual,nominal in zip(times,expected)]
    if any(abs(value)>1 for value in offsets): raise ValueError("Native sample lies beyond one second of nominal cadence")
    indices={}
    for n,(receipt,idx) in enumerate(selected.values()):
        indices.setdefault(receipt["path"],[]).append((n,idx))
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as tmp:
        out=Path(tmp)/target.name
        with nc.Dataset(next(iter(indices))) as first,nc.Dataset(out,"w") as dst:
            for name,dim in first.dimensions.items(): dst.createDimension(name,len(times) if name=="time" else len(dim))
            dst.setncatts({k:first.getncattr(k) for k in first.ncattrs() if k not in ("history","_NCProperties")})
            dst.history="Native OFS fields, UTC deduplicated by archived nowcast cycle"
            for name,v in first.variables.items():
                kw={"fill_value":v._FillValue} if "_FillValue" in v.ncattrs() else {}
                w=dst.createVariable(name,v.dtype,v.dimensions,zlib=True,complevel=2,**kw)
                w.setncatts({k:v.getncattr(k) for k in v.ncattrs() if k!="_FillValue"})
                if "time" not in v.dimensions: w[:]=v[:]
            for path,pairs in indices.items():
                dest,source=np.asarray(pairs).T
                with nc.Dataset(path) as src:
                    if set(src.variables)!=set(first.variables): raise ValueError("Station variable inventory changed")
                    for name,v in src.variables.items():
                        if v.dimensions!=first[name].dimensions or getattr(v,"units",None)!=getattr(first[name],"units",None):
                            raise ValueError("Station variable axes or units changed")
                        if "time" in v.dimensions:
                            if v.dimensions[0]!="time": raise ValueError("Unexpected native station time axis")
                            dst[name][dest]=v[source]
                        elif not np.array_equal(v[:],first[name][:]): raise ValueError("Station IDs, depths or geometry changed")
        with nc.Dataset(out) as ds:
            from great_lakes_ofs import cf_clocks
            if cf_clocks(ds["time"][:],ds["time"].units)!=times: raise ValueError("Station time roundtrip mismatch")
            dimensions={k:len(v) for k,v in ds.dimensions.items()}
        os.replace(out,target)
    result=dict(schema="great_lakes_ofs_station_reference_v1",status="ready",system=system,start_utc=stamp(start),end_exclusive_utc=stamp(end),
                cadence_seconds=cadence,dimensions=dimensions,duplicate_records_arbitrated=overlaps,sha256=digest(target),source_count=len(receipts),
                off_nominal_native_times=[dict(actual_utc=t,nominal_utc=n,offset_seconds=d) for t,n,d in zip(times,expected,offsets) if d],
                sources=[dict(key=r["object"]["key"],sha256=r["sha256"],etag=r["object"]["etag"],retained_records=len(indices.get(r["path"],[]))) for r in receipts])
    save(target.with_suffix(".json"),result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system",choices=SYSTEMS,required=True)
    p.add_argument("--start",required=True)
    p.add_argument("--end",required=True,help="Exclusive")
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cadence-seconds",type=int,default=360)
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    print(json.dumps(series(a.system,a.start,a.end,a.cache,a.output,a.cadence_seconds,a.workers)["dimensions"]))
