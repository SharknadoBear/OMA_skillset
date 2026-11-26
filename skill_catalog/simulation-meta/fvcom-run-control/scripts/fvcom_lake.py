"""Resolve and freeze the baroclinic Lake Superior experiment."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
import re

MODE = "baroclinic-temperature-wind-river"
PROFILE = "lake_superior_summer"
GRID_POLICY = "operational_lake_mesh"
ROLE_ROOTS = dict(lead="agents/00_lead",grid="agents/01_grid",
                  forcing_initialization="agents/02_forcing",configuration_build="agents/03_configuration_build")
PROVIDERS = ["hrrr","era5","cfsv2","cfsr"]
START = datetime(2025,7,1,tzinfo=timezone.utc)
END = datetime(2025,8,1,tzinfo=timezone.utc)
SOURCE_COMMIT = "f8e4b50d4fd51409febcefc763cbc5dca11870e2"

def utc(t):
    return t.isoformat().replace("+00:00","Z")


def resolve(req):
    """The named regional profile fixes the scientific comparison before any job runs."""
    req=deepcopy(req)
    if req.get("mode")!=MODE or req.get("profile")!=PROFILE:
        raise ValueError("the Lake Superior profile requires the baroclinic temperature, wind and river mode")
    if req.get("execution",{}).get("site")!="perlmutter" or "kestrel" in req:
        raise ValueError("the Lake Superior pilot requires the Perlmutter CPU site")
    site=req.get("perlmutter",{})
    if not re.fullmatch(r"[A-Za-z0-9_-]+",site.get("account","")):
        raise ValueError("supply the Perlmutter CPU charging account")
    for key in ("project_dir","source_dir"):
        p=site.get(key,"")
        if not re.fullmatch(r"/[a-zA-Z0-9_./+-]+",p) or ".." in PurePosixPath(p).parts:
            raise ValueError(f"supply a resolved absolute perlmutter.{key}")
    if req.get("tpxo") or req.get("accepted_grid"):
        raise ValueError("Lake Superior uses native LSOFS geometry and signed tributary/outlet forcing")
    defaults=dict(grid_policy=GRID_POLICY,
        physics=dict(formulation="three_dimensional_baroclinic",water="freshwater",temperature=True,
                     stress=True,atmospheric_pressure=True,heat=True,precipitation=True,evaporation=True,
                     tide=False,ice=False,sigma_interfaces=21,sigma_layers=20),
        mesh=dict(provider="lsofs",coordinates="spherical",frame="east_north"),
        river=dict(provider="usgs_eccc",outlet="st_marys",outlet_tracers="local_donor"),
        initialization=dict(provider="lsofs",type="native_mesh_nowcast"),
        atmosphere=dict(provider_order=PROVIDERS,heat_mode="direct",wind_mode="stress"),
        period=dict(analysis_start=utc(START),analysis_end=utc(END),spinup_candidates_days=[7,14,28]),
        observations=dict(provider="ndbc_glsea_coops",model_reference="lsofs"),
        benchmark=dict(enabled=True,ranks=[64,128,256],repeats=3,segment_hours=24,
                       cores_per_node=128,extend=False,cost_margin=0.15),
        execution=dict(site="perlmutter",stages=["smoke","spinup","benchmark","production"],
                       smoke_ranks=128,threads_per_rank=1),
        perlmutter=dict(source_commit=SOURCE_COMMIT,constraint="cpu"),
        output=dict(station_seconds=3600,grid_seconds=21600,restart_seconds=86400))
    def merge(target, fixed, prefix=""):
        for key,value in fixed.items():
            path=prefix+key
            if isinstance(value,dict):
                branch=target.setdefault(key,{})
                if not isinstance(branch,dict): raise ValueError(path+" must be an object")
                merge(branch,value,path+".")
            elif key in target and (target[key]!=value or type(target[key])!=type(value)):
                raise ValueError(f"{PROFILE} fixes {path}={value!r}")
            else:
                target[key]=deepcopy(value)
    merge(req,defaults)
    return req


def schedule(req):
    req=resolve(req)
    trials=[]
    def add(name,first,last,ranks,state):
        trials.append(dict(name=name,start_utc=utc(first),end_utc=utc(last),ranks=ranks,startup=state))
    add("startup_probe",START,START+timedelta(minutes=10),128,"noaa_nowcast")
    add("smoke",START,START+timedelta(days=1),128,"noaa_nowcast")
    add("restart_probe",START+timedelta(days=1),START+timedelta(days=1,hours=1),128,"smoke_terminal_restart")
    for days in req["period"]["spinup_candidates_days"]:
        add(f"spinup_{days:02d}d",START-timedelta(days=days),START+timedelta(days=3),128,"noaa_nowcast")
    for rep in range(1,req["benchmark"]["repeats"]+1):
        for ranks in req["benchmark"]["ranks"]:
            add(f"benchmark_{ranks:03d}_r{rep}",START,START+timedelta(days=1),ranks,"selected_july_restart")
    add("production",START,END,None,"selected_july_restart")
    return dict(schema="fvcom_lake_schedule_v1",case_id=req["case_id"],stages=trials,
                comparison_start_utc=utc(START),comparison_end_utc=utc(START+timedelta(days=3)),
                convergence=dict(volume_weighted_temperature_rms_c=0.25,upper_20m_temperature_rms_c=0.5,
                                 median_thermocline_abs_difference_m=2.0,lake_mean_elevation_difference_m=0.02))
