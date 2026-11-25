"""Freeze a Lake Superior stage, reusing a single immutable copy of each runtime input."""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import hashlib
import json
import re
import shutil

from fvcom_lake import resolve,schedule

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def materialize(args):
    from fvcom_run_control import attempt_sbatch, load_data
    req=resolve(load_data(args.project/"request.json"))
    stages={s["name"]:s for s in schedule(req)["stages"]}
    spec=stages[args.name]
    binding=load_data(args.binding)
    binary=binding["native_executable"]
    modules=binding["build_environment"]["modules"]
    if (binding.get("status")!="frozen" or binding.get("site")!="perlmutter"
            or binding["source_commit"]!=req["perlmutter"]["source_commit"]
            or not re.fullmatch(r"[0-9a-f]{64}",binding["executable_sha256"])):
        raise ValueError("expected the frozen Perlmutter lake build")
    if not re.fullmatch(r"/[a-zA-Z0-9_./+-]+",binary) or ".." in PurePosixPath(binary).parts:
        raise ValueError("unsafe runtime executable path")
    if args.revision<0 or not re.fullmatch(r"[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]",args.walltime):
        raise ValueError("invalid revision or Slurm time")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+",args.case_name):
        raise ValueError("invalid case basename")
    ranks=spec["ranks"]
    if ranks is None:
        summary=load_data(args.benchmark_summary)
        ranks=summary["production_layout"]["ranks"]
        if ranks not in req["benchmark"]["ranks"]: raise ValueError("unmeasured production layout")
    args.ranks=ranks
    args.nodes=(ranks+127)//128
    args.grid_case="native_lsofs"
    site=req["perlmutter"]
    args.execution_site="perlmutter"
    args.account=site["account"]
    args.site_configuration={**site,"qos":args.qos,"threads_per_rank":1}
    args.module=modules
    args.remote_executable=binary
    args.job_prefix="lsofs"
    aid=f"attempt_{args.name}_{args.revision:04d}"
    root=Path("run")/args.grid_case/"attempts"/aid
    files=load_data(args.inputs)
    if not files or not all(re.fullmatch(r"[A-Za-z0-9_.-]+",k) and k not in {".",".."} and
                            re.fullmatch(r"[0-9a-f]{64}",v) for k,v in files.items()):
        raise ValueError("inputs must be an exact map of filenames to content SHA-256")
    nml=args.namelist.read_text(encoding="utf-8")
    def key(k):
        matches=re.findall(r"(?im)^\s*"+k+r"\s*=\s*([^!\r\n]+)",nml)
        if len(matches)!=1: raise ValueError("required single namelist key "+k)
        return matches[0].strip().rstrip(",").strip().strip("'\x22")
    for param,expected in (("START_DATE",spec["start_utc"]),("END_DATE",spec["end_utc"])):
        actual=datetime.fromisoformat(key(param).replace("Z","+00:00"))
        if actual.tzinfo is None: actual=actual.replace(tzinfo=timezone.utc)
        if actual!=datetime.fromisoformat(expected.replace("Z","+00:00")):
            raise ValueError(param+" differs from stage plan")
    if key("INPUT_DIR").rstrip("/")!="./input" or key("OUTPUT_DIR").rstrip("/")!=".":
        raise ValueError("lake attempt uses stage-local ./input/ and ./")
    for k in ("STARTUP_FILE","GRID_FILE","DEPTH_FILE","SIGMA_LEVELS_FILE","WIND_FILE","HEATING_FILE",
              "AIRPRESSURE_FILE","PRECIPITATION_FILE","STATION_FILE","RIVER_INFO_FILE"):
        if key(k) not in files: raise ValueError("unfrozen input "+k)
    if key("STARTUP_TYPE").lower()!="hotstart": raise ValueError("lake stage must read its dated startup")
    facts=dict(schema="fvcom_lake_attempt_v1",name=aid,stage=spec,mpi_ranks=ranks,nodes=args.nodes,
               site="perlmutter",account=args.account,files=files,executable_sha256=binding["executable_sha256"],
               executable=binary,build_binding_sha256=digest(args.binding),run_namelist_sha256=digest(args.namelist),
               remote_run=(PurePosixPath(site["project_dir"])/root.as_posix()).as_posix())
    out=args.project/root
    out.parent.mkdir(parents=True,exist_ok=True)
    script=attempt_sbatch(args,aid,None)
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage=Path(tmp)/"stage";stage.mkdir()
        shutil.copyfile(args.namelist,stage/(args.case_name+"_run.nml"))
        checks=[(v,"input/"+k) for k,v in sorted(files.items())]
        checks.extend([(facts["run_namelist_sha256"],args.case_name+"_run.nml"),(facts["executable_sha256"],binary)])
        (stage/"runtime_checksums.sha256").write_text("".join(f"{v}  {k}\n" for v,k in checks),newline="\n")
        (stage/"job.sbatch").write_text(script,encoding="utf8",newline="\n")
        facts["job_sha256"]=digest(stage/"job.sbatch")
        (stage/"attempt_manifest.json").write_text(json.dumps(facts,indent=2,sort_keys=True)+"\n",encoding="utf8")
        if out.exists():
            if any(not (out/p.name).is_file() or digest(p)!=digest(out/p.name) for p in stage.iterdir()):
                raise FileExistsError(f"conflicting lake attempt {out}")
        else: stage.rename(out)
    return facts
