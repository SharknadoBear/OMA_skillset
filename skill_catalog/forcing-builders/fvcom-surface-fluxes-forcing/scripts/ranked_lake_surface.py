"""Select the first complete lake surface package in HRRR, ERA5, CFSv2, CFSR order."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from acquire_wind_pressure import HOUR, ORDER, collect, iso, write

BASE_PRODUCTS = ["wind_10m","mean_sea_level_pressure"]
CFS_PRODUCTS = ["downward_shortwave_surface_flux","upward_shortwave_surface_flux",
                "downward_longwave_surface_flux","upward_longwave_surface_flux",
                "sensible_heat_net_flux","latent_heat_net_flux","precipitation_rate"]
HRRR_PRODUCTS = BASE_PRODUCTS + ["downward_shortwave_flux","downward_longwave_flux",
                                 "sensible_heat_flux","latent_heat_flux"] + [
    dict(family="wrfsfc",short_name=code,level_text="surface",step_type="instant",output_name=name)
    for code,name in (("USWRF","upward_shortwave_flux"),("ULWRF","upward_longwave_flux"))]


def hrrr_lake_arguments(parser):
    parser.add_argument("--lake-water-mask", type=Path, action="append",
                        help="Checked HRRR LAND connector NetCDF. Repeat for dated mask checks.")
    parser.add_argument("--lake-seed-lonlat", nargs=2, type=float, metavar=("LONGITUDE", "LATITUDE"))
    parser.add_argument("--max-water-distance-km", type=float,
                        help="Maximum distance from a model node to a replacement donor")
    parser.add_argument("--rain-context", nargs=2, metavar=("START_UTC", "END_UTC"),
                        help="Parent HRRR integration window for identical precipitation transitions at segment edges")


def acquire(args, provider):
    from datetime import datetime, timezone
    start,end = (datetime.fromisoformat(v.replace("Z","+00:00")).astimezone(timezone.utc) for v in (args.start,args.end))
    if not args.bbox:
        raise ValueError("Fetching source fields requires --bbox WEST SOUTH EAST NORTH")
    if provider != "hrrr":
        products = BASE_PRODUCTS + (["surface_heat_and_precipitation"] if provider=="era5" else CFS_PRODUCTS)
        collect(SimpleNamespace(**{**vars(args),"output":args.source_root}),provider,start,end,products)
        return
    sys.path.insert(0,str(args.skill_root/"hrrr-fetcher"/"scripts"))
    import hrrr_core as core
    if not core.runtime_preflight()["passed"]:
        raise ValueError("HRRR requires its pinned GRIB runtime")
    rain_start,rain_end = start,end-HOUR
    if args.rain_context:
        outer_start,outer_end = (datetime.fromisoformat(v.replace("Z","+00:00")).astimezone(timezone.utc) for v in args.rain_context)
        if not (outer_start<=start and end<=outer_end):
            raise ValueError("Segment exceeds its rain context")
        rain_start,rain_end=max(outer_start,start-HOUR),min(outer_end-HOUR,end)
    for forecast,cache,first,stop in ((False,args.hrrr_analysis,start,end),(True,args.hrrr_precipitation,rain_start,rain_end)):
        tick = first.replace(hour=6*(first.hour//6))
        while tick<=stop:
            begin,finish = max(tick,first),min(tick+5*HOUR,stop)
            root = cache/tick.strftime("%Y%m%d%H")
            root.mkdir(parents=True,exist_ok=True)
            req = dict(schema_version="hrrr_request_v1",domain="conus",bbox=args.bbox,halo_cells=2,
                       missing_policy="error",retain_raw_messages=False)
            if forecast:
                req.update(mode="forecast",cycle_start=iso(begin),cycle_end=iso(finish),cycle_step_hours=1,
                           forecast_periods=["PT1H"],products=["total_precipitation"])
            else:
                req.update(mode="analysis",start=iso(begin),end=iso(finish),products=HRRR_PRODUCTS)
            if not (root/"health_check.json").exists() or not core.health_run(root)["passed"]:
                plan = core.build_plan(req,root)
                core.atomic_write_json(root/"download_plan.json",plan)
                core.execute_plan(plan,root)
                if not core.health_run(root)["passed"]:
                    raise ValueError("HRRR source health failed")
            tick += 6*HOUR


def run(args):
    args.source_root=args.source_root.resolve()
    args.hrrr_analysis = args.hrrr_analysis or args.source_root/"hrrr"
    args.hrrr_precipitation = args.hrrr_precipitation or args.source_root/"hrrr_precip"
    if args.fetch_worker:
        acquire(args,args.fetch_worker)
        return
    from lake_interval_surface import build as build_intervals
    from lake_surface import build as build_hrrr
    from prepare_wind_pressure import hour_axis, require
    from acquire_wind_pressure import digest
    clock = hour_axis(args.start,args.end)
    require(len(clock)>=2,"Need at least one complete forcing interval")
    args.output.mkdir(parents=True,exist_ok=True)
    attempts=[]
    for provider in ORDER:
        root = args.source_root/provider
        try:
            available = ((args.hrrr_analysis.is_dir() and args.hrrr_precipitation.is_dir()) if provider=="hrrr"
                         else (root/"health_check.json").is_file())
            def prepare(staging):
                selected = SimpleNamespace(**{**vars(args),"output":staging})
                if provider=="hrrr":
                    selected.analysis,selected.precipitation = args.hrrr_analysis,args.hrrr_precipitation
                    selected.pressure_only = False
                    return build_hrrr(selected)
                return build_intervals(selected,root,provider)
            with TemporaryDirectory(dir=args.output) as tmp:
                staging = Path(tmp)
                if available:
                    try:
                        manifest = prepare(staging)
                    except (ValueError,OSError,KeyError,AttributeError) as exc:
                        if not args.acquire:
                            raise
                        attempts.append(dict(provider=provider,status="cache_rejected",reason=str(exc)))
                        available=False
                if not available:
                    require(args.acquire,f"Missing checked {provider} source cache")
                    args.source_root.mkdir(parents=True,exist_ok=True)
                    interpreter = getattr(args,("cfs" if provider in ("cfsv2","cfsr") else provider)+"_python")
                    command=[str(interpreter or sys.executable),str(Path(__file__).resolve()),*sys.argv[1:],"--fetch-worker",provider]
                    with (args.output/f"{provider}_acquire.log").open("w",encoding="utf8") as log:
                        status = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
                    require(status==0,f"{provider} archive worker exited {status}, see {provider}_acquire.log")
                    manifest = prepare(staging)
                products = [staging/name for name in [*manifest["files"],"surface_manifest.json"]]
                for p in products:
                    dest = args.output/p.name
                    if dest.exists() and digest(dest)!=digest(p):
                        raise FileExistsError(f"Different forcing already exists at {dest}; use a fresh output directory")
                for p in products:
                    if not (args.output/p.name).exists():
                        p.rename(args.output/p.name)
            attempts.append(dict(provider=provider,status="selected"))
            write(args.output/"provider_attempts.json",dict(schema="fvcom_lake_source_selection_v1",
                  selected=provider,complete_bundle=True,attempts=attempts,manifest_sha256=digest(args.output/"surface_manifest.json")))
            print(provider,args.output/"surface_manifest.json")
            return manifest
        except FileExistsError:
            raise
        except (ValueError,OSError,RuntimeError,KeyError,AttributeError) as exc:
            attempts.append(dict(provider=provider,status="rejected",reason=str(exc)))
            write(args.output/"provider_attempts.json",dict(schema="fvcom_lake_source_selection_v1",
                  selected=None,complete_bundle=False,attempts=attempts))
    raise ValueError("No complete lake surface provider; see provider_attempts.json")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("mesh","mod-bulk","output","source-root"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("start","end"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--projected-crs")
    hrrr_lake_arguments(parser)
    parser.add_argument("--acquire",action="store_true",help="Use the archive connectors for missing or rejected caches")
    parser.add_argument("--bbox",type=float,nargs=4,metavar=("WEST","SOUTH","EAST","NORTH"))
    for name in ("hrrr-analysis","hrrr-precipitation","hrrr-python","era5-python","cfs-python"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--skill-root",type=Path,default=Path(os.environ.get("CODEX_HOME",Path.home()/".codex"))/"skills")
    parser.add_argument("--fetch-worker",choices=ORDER,help=argparse.SUPPRESS)
    run(parser.parse_args())
