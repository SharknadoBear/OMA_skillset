"""Verify an intact FVCOM native checkpoint at a continuation epoch."""
import argparse
import json
from pathlib import Path

import netCDF4 as nc
import numpy as np

from lake_initial import DIRECT, FLAGS, complete, native_times, save, sha, utc, verify_reader_layout


def verify(restart, initial_manifest, when, executable_sha256, source_job_id, expected_sha256):
    manifest=json.loads(initial_manifest.read_text())
    if manifest["status"]!="ready":
        raise ValueError("Continuation requires a production-eligible initialization")
    initial=initial_manifest.parent/manifest["file"]["path"]
    if sha(initial)!=manifest["file"]["sha256"]:
        raise ValueError("Initial-state bytes disagree with receipt")
    if sha(restart)!=expected_sha256:
        raise ValueError("Checkpoint differs from the originating job's checksum")
    if len(executable_sha256)!=64 or any(c not in "0123456789abcdef" for c in executable_sha256) or not str(source_job_id).strip():
        raise ValueError("Missing source executable SHA-256 or source job ID")
    t=utc(when)
    if t<=utc(manifest["case_epoch_utc"]):
        raise ValueError("Checkpoint must be later than the parent analysis")
    with nc.Dataset(restart) as ds,nc.Dataset(initial) as prior:
        verify_reader_layout(ds)
        times=native_times(ds)
        if t not in times: raise ValueError("Missing exact continuation timestamp")
        record=times.index(t)
        iteration=int(complete(ds["iint"],record))
        if iteration<=0: raise ValueError("A continuation must retain its positive native step counter")
        for key in ("node","nele","siglay","siglev"):
            if len(ds.dimensions[key])!=manifest["dimensions"][key]:
                raise ValueError("Native restart dimensions changed")
        for key in ("nv","h","siglay","siglev","lon","lat","cor","cc_sponge"):
            old,new=complete(prior[key]),complete(ds[key])
            if prior[key].dimensions!=ds[key].dimensions: raise ValueError(f"Checkpoint axes differ at {key}")
            if key=="lon": ok=np.max(abs((new.astype(float)-old+180)%360-180))<3.2e-5
            elif key=="lat": ok=np.max(abs(new-old))<4e-6
            else: ok=np.allclose(new,old,rtol=0,atol=1e-7) if key.startswith("sig") else np.array_equal(new,old)
            if not ok: raise ValueError(f"Checkpoint geometry differs at {key}")
        for key in (*DIRECT,"et","ua","va","el_press","rho1","tmean1","smean1","rmean1"):
            if key not in ds.variables or prior[key].dimensions!=ds[key].dimensions:
                raise ValueError(f"Checkpoint lacks native reader field/axes {key}")
            values=complete(ds[key],record if "time" in ds[key].dimensions else slice(None))
            if key in FLAGS and not np.isin(values,[0,1]).all(): raise ValueError("Checkpoint wetness flags invalid")
            if key in ("salinity","smean1") and np.any(abs(values)>1e-7): raise ValueError("Freshwater salinity changed")
            if key in ("rho1","rmean1") and np.any(abs(values)>.05): raise ValueError("Native density anomaly units invalid")
        wet=complete(ds["wet_nodes"],record).astype(bool)
        if not wet.any() or np.any((complete(ds["h"])+complete(ds["zeta"],record))[wet]<=0):
            raise ValueError("Invalid wet-node water column")
    return dict(schema="fvcom_native_restart_handoff_v1",status="ready",source_job_id=str(source_job_id),executable_sha256=executable_sha256,
                initialization_manifest_sha256=sha(initial_manifest),case_epoch_utc=when,source_record=record,source_iint=iteration,
                file=dict(path=str(restart.resolve()),sha256=expected_sha256,bytes=restart.stat().st_size),
                namelist={**manifest["namelist"],"STARTUP_FILE":restart.name},transition="preserve every native checkpoint byte and original iteration")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("restart","initial-manifest","output"):
        p.add_argument("--"+arg,type=Path,required=True)
    for arg in ("time","executable-sha256","source-job-id","checkpoint-sha256"):
        p.add_argument("--"+arg,required=True)
    a=p.parse_args()
    save(a.output,verify(a.restart,a.initial_manifest,a.time,a.executable_sha256,a.source_job_id,a.checkpoint_sha256))
