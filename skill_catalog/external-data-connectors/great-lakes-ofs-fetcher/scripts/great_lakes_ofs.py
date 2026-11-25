"""Inventory, range-inspect, cache and time-select NOAA Great Lakes OFS NetCDF data."""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import time
from contextlib import contextmanager
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote

import h5py
import netCDF4 as nc
import numpy as np
import requests

SYSTEMS = ("lsofs","lmhofs","leofs","loofs")
ROOT = "https://noaa-nos-ofs-pds.s3.amazonaws.com"
NAME = re.compile(r"(?:nos\.)?(?P<sys>lsofs|lmhofs|leofs|loofs)\.t(?P<cycle>\d{2})z\.(?P<day>\d{8})\.(?P<product>fields|stations)\.(?P<stream>[nf]\d{3}|nowcast|forecast)\.nc$")


def utc(value):
    parsed=datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset()!=timedelta(0):
        raise ValueError("Supply explicit UTC timestamps")
    return parsed.astimezone(timezone.utc)


def stamp(value):
    return value.isoformat(timespec="seconds").replace("+00:00","Z")


def save(path, data):
    path=Path(path)
    path.parent.mkdir(exist_ok=True,parents=True)
    with NamedTemporaryFile("w",dir=path.parent,delete=False,encoding="utf-8") as f:
        json.dump(data,f,indent=2,allow_nan=False)
        f.write("\n")
    os.replace(f.name,path)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def listing(system,date,session=None):
    if system not in SYSTEMS:
        raise ValueError("Unknown Great Lakes FVCOM system")
    session=session or requests
    prefix=f"{system}/netcdf/{date:%Y/%m/%d}/"
    if date.year<2025:
        prefix=f"{system}/netcdf/{date:%Y%m}/"
    params={"list-type":"2","prefix":prefix}
    rows=[]
    while True:
        r=session.get(ROOT,params=params,timeout=45)
        r.raise_for_status()
        doc=ET.fromstring(r.content)
        for elem in doc.findall("{*}Contents"):
            key=elem.findtext("{*}Key")
            match=NAME.fullmatch(key.rsplit("/",1)[-1])
            if not match or match["sys"]!=system or match["day"]!=date.strftime("%Y%m%d"):
                continue
            cycle=datetime.strptime(match["day"]+match["cycle"],"%Y%m%d%H").replace(tzinfo=timezone.utc)
            token=match["stream"]
            now=token.startswith("n")
            if token[-1].isdigit():
                t=cycle+timedelta(hours=int(token[1:])-(6 if now else 0))
                start=end=t
            else:
                start=cycle-timedelta(hours=6) if now else cycle
                end=cycle if now else cycle+timedelta(hours=120)
            rows.append(dict(system=system,key=key,url=ROOT+"/"+quote(key,safe="/"),size=int(elem.findtext("{*}Size")),
                             etag=elem.findtext("{*}ETag").strip('"'),last_modified=elem.findtext("{*}LastModified"),
                             product=match["product"],stream="nowcast" if now else "forecast",cycle=stamp(cycle),
                             candidate_start=stamp(start),candidate_end=stamp(end)))
        cont=doc.findtext("{*}NextContinuationToken")
        if not cont:
            break
        params["continuation-token"]=cont
    return rows


def ranked(rows,target,product,allow_forecast=False):
    """Rank candidates; acceptance still requires a matching internal valid time."""
    when=utc(target)
    candidates=[o for o in rows if o["product"]==product and (allow_forecast or o["stream"]=="nowcast")
                and utc(o["candidate_start"])<=when<=utc(o["candidate_end"])]
    return sorted(candidates,key=lambda o:(o["stream"]!="nowcast",abs((utc(o["cycle"])-when).total_seconds()),
                                         -utc(o["cycle"]).timestamp(),o["key"]))


def get_range(obj,start,end,session=None):
    headers={"Range":f"bytes={start}-{end}","If-Match":'"'+obj["etag"]+'"',"Accept-Encoding":"identity"}
    for attempt in range(4):
        try:
            resp=(session or requests).get(obj["url"],headers=headers,timeout=(15,120))
            if resp.status_code in (429,500,502,503,504):
                raise requests.ConnectionError(f"Transient HTTP {resp.status_code}")
            resp.raise_for_status()
            if (resp.status_code!=206 or resp.headers.get("Content-Range")!=f"bytes {start}-{end}/{obj['size']}" or
                    resp.headers.get("ETag","").strip('"')!=obj["etag"] or len(resp.content)!=end-start+1):
                raise ValueError("Range, size or version identity differs from inventory")
            return resp.content
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            if attempt==3:
                raise
            time.sleep(2**attempt)


class RangeReader(io.RawIOBase):
    """Seekable, version-locked block reader for small HDF5 metadata inspections."""
    def __init__(self,obj,block=384*1024):
        self.obj=obj
        self.block=block
        self.pos=0
        self.blocks={}
        self.session=requests.Session()

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=(0,self.pos,self.obj["size"])[whence]+offset
        if self.pos<0:
            raise ValueError("Negative file offset")
        return self.pos

    def readinto(self,buf):
        limit=min(len(buf),self.obj["size"]-self.pos)
        copied=0
        while copied<limit:
            k,off=divmod(self.pos,self.block)
            if k not in self.blocks:
                begin=k*self.block
                self.blocks[k]=get_range(self.obj,begin,min(begin+self.block,self.obj["size"])-1,self.session)
            piece=self.blocks[k][off:off+limit-copied]
            buf[copied:copied+len(piece)]=piece
            copied+=len(piece)
            self.pos+=len(piece)
        return max(0,copied)

    def close(self):
        self.session.close()
        super().close()


def strings(raw):
    a=np.asarray(raw)
    return [b"".join(row).decode("ascii").strip().strip("\x00") for row in a]


def internal_clocks(raw):
    values=[]
    for s in strings(raw):
        value=datetime.fromisoformat(s.replace("/","-").replace("Z","+00:00"))
        if value.tzinfo is None: value=value.replace(tzinfo=timezone.utc)
        values.append(stamp(value))
    if not values or values!=sorted(set(values)):
        raise ValueError("Invalid or duplicate native UTC clock")
    return values


def cf_clocks(values,units,calendar="standard"):
    decoded=nc.num2date(values,units,calendar)
    if any(v.microsecond for v in decoded):
        raise ValueError("Unrepresentable fractional-second native sample")
    clocks=[stamp(datetime(v.year,v.month,v.day,v.hour,v.minute,v.second,tzinfo=timezone.utc)) for v in decoded]
    if not clocks or clocks!=sorted(set(clocks)): raise ValueError("Invalid CF source clock")
    return clocks


def metadata(obj):
    with RangeReader(obj) as reader, h5py.File(reader,"r") as f:
        variables={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in f.items() if isinstance(v,h5py.Dataset)}
        if "Times" in f:
            clocks=internal_clocks(f["Times"][:])
        else:
            units=f["time"].attrs["units"]
            clocks=cf_clocks(f["time"][:],units.decode() if isinstance(units,bytes) else units)
        return dict(object=obj,variables=variables,utc=clocks,bytes_read=sum(map(len,reader.blocks.values())))


def verify_local(path,obj):
    if path.stat().st_size!=obj["size"]:
        raise ValueError("Cached source has the wrong length")
    if re.fullmatch("[0-9a-f]{32}",obj["etag"]):
        with path.open("rb") as f:
            if hashlib.file_digest(f,"md5").hexdigest()!=obj["etag"]:
                raise ValueError("Single-part S3 ETag does not match local data")
    with nc.Dataset(path) as ds:
        variable=ds["time"]
        t=internal_clocks(ds["Times"][:]) if "Times" in ds.variables else cf_clocks(variable[:],variable.units,getattr(variable,"calendar","standard"))
        cf=nc.num2date(variable[:],variable.units,getattr(variable,"calendar","standard"))
        if any(abs((utc(actual)-datetime(v.year,v.month,v.day,v.hour,v.minute,v.second,tzinfo=timezone.utc)).total_seconds())>
               max(1,abs(float(np.spacing(np.array(variable[i],dtype=variable.dtype))))*(86400 if variable.units.startswith("days") else 1))
               for i,(actual,v) in enumerate(zip(t,cf))):
            raise ValueError("Native Times and quantitative CF time disagree")
    return dict(sha256=digest(path),size=path.stat().st_size,clock_utc=t)


@contextmanager
def cache_lock(path):
    """OS releases the per-source lock on process exit, including interrupted downloads."""
    with path.open("a+b") as f:
        f.seek(0,2)
        if not f.tell():
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name=="nt":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
                    break
                except OSError as error:
                    if error.winerror not in (33,36) and error.errno!=13: raise
                    time.sleep(.25)
            try: yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(f,fcntl.LOCK_UN)


def cache(obj,root,existing=None,block=8*1024*1024,fetch_range=get_range):
    root=Path(root)
    path=root/obj["key"]
    path.parent.mkdir(parents=True,exist_ok=True)
    with cache_lock(path.with_suffix(".lock")):
        return cache_version(obj,path,existing,block,fetch_range)


def cache_version(obj,path,existing,block,fetch_range):
    receipt=path.with_suffix(".receipt.json")
    if existing:
        candidate=Path(existing).resolve()
        evidence=verify_local(candidate,obj)
        if "-" in obj["etag"]:
            # S3 multipart ETags bind object parts, not the concatenated bytes. Verify the full file by ranges once.
            with candidate.open("rb") as f:
                for first in range(0,obj["size"],block):
                    if f.read(block)!=fetch_range(obj,first,min(first+block,obj["size"])-1):
                        raise ValueError("External cached bytes differ from the listed S3 version")
        return dict(path=str(candidate),object=obj,**evidence)
    if path.exists():
        if not receipt.is_file():
            raise ValueError("Unreceipted final cache file")
        prior=json.loads(receipt.read_text())
        if prior["object"]!=obj or prior["sha256"]!=digest(path):
            raise ValueError("Cached immutable source differs from its version or checksum")
        return prior
    part=path.with_suffix(".part")
    state=path.with_suffix(".partial.json")
    partial=json.loads(state.read_text()) if state.is_file() else {"key":obj["key"],"etag":obj["etag"],"blocks":[]}
    if (partial["etag"],partial["key"])!=(obj["etag"],obj["key"]):
        raise ValueError("Incomplete cache belongs to another source version")
    with part.open("a+b") as f:
        f.seek(0)
        offset=0
        for length,checksum in partial["blocks"]:
            if hashlib.sha256(f.read(length)).hexdigest()!=checksum:
                raise ValueError("Incomplete cache block mutated")
            offset+=length
        f.truncate(offset)
        f.seek(offset)
        if shutil.disk_usage(path.parent).free<obj["size"]-offset:
            raise ValueError("Insufficient cache space for remaining native bytes")
        while offset<obj["size"]:
            last=min(offset+block,obj["size"])-1
            raw=fetch_range(obj,offset,last)
            require_len=last-offset+1
            if len(raw)!=require_len: raise ValueError("Short source block")
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
            partial["blocks"].append([len(raw),hashlib.sha256(raw).hexdigest()])
            save(state,partial)
            offset+=len(raw)
    evidence=verify_local(part,obj)
    result=dict(path=str(path.resolve()),object=obj,**evidence)
    # Persist validated receipt first: interruption before final publication safely resumes from verified blocks.
    save(receipt,result)
    os.replace(part,path)
    state.unlink(missing_ok=True)
    return result


def fetch_epoch(rows,target,product,root,allow_forecast=False,existing=None):
    target=stamp(utc(target))
    rejected=[]
    for obj in ranked(rows,target,product,allow_forecast):
        info=cache(obj,root,existing=existing)
        if target in info["clock_utc"]:
            return dict(schema="great_lakes_ofs_snapshot_v1",status="ready",target_utc=target,record=info["clock_utc"].index(target),
                        selection="nowcast first, closest cycle, latest cycle at a tie; internal UTC exact",rejected=rejected,**info)
        rejected.append(obj["key"])
        existing=None
    raise ValueError("No archived source with the exact internally verified target time")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("inventory","metadata","fetch"))
    p.add_argument("--system",choices=SYSTEMS,required=True)
    p.add_argument("--time",required=True)
    p.add_argument("--product",choices=("fields","stations"),default="fields")
    p.add_argument("--cache",type=Path)
    p.add_argument("--existing",type=Path)
    p.add_argument("--allow-forecast",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    t=utc(a.time)
    # Include cycles on the adjacent calendar day to bracket nowcast archives around 18–24Z.
    dates=[(t+timedelta(days=i)).date() for i in range(-5 if a.allow_forecast else 0,2)]
    rows=[row for d in dates for row in listing(a.system,d)]
    candidates=ranked(rows,a.time,a.product,a.allow_forecast)
    if a.action=="inventory":
        result=dict(schema="great_lakes_ofs_inventory_v1",system=a.system,target_utc=stamp(t),
                    returned_objects=len(rows),ranked_candidates=candidates,
                    first_choice_download_bytes=candidates[0]["size"] if candidates else 0,
                    candidate_download_bytes=sum(item["size"] for item in candidates))
    elif a.action=="metadata":
        result=None
        for obj in candidates:
            info=metadata(obj)
            if stamp(t) in info["utc"]:
                result=info
                break
        if result is None: raise ValueError("No internally confirmed candidate")
    else:
        if not a.cache: p.error("fetch needs --cache")
        result=fetch_epoch(rows,a.time,a.product,a.cache,a.allow_forecast,a.existing)
    save(a.output,result)
    print(a.output)


if __name__=="__main__":
    main()
