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
    import fvcom_lake_coare as coare
    req=resolve(load_data(args.project/"request.json"))
    binding=load_data(args.binding)
    binary=binding["native_executable"]
    modules=binding["build_environment"]["modules"]
    if (binding.get("status")!="frozen" or binding.get("site")!="perlmutter"
            or binding["source_commit"]!=req["perlmutter"]["source_commit"]
            or not re.fullmatch(r"[0-9a-f]{64}",binding["executable_sha256"])):
        raise ValueError("expected the frozen Perlmutter lake build")
    nml=args.namelist.read_text(encoding="utf-8")
    diagnostic=bool(getattr(args,"coare_build",None))
    if diagnostic != coare.enabled(nml):
        raise ValueError("Calculated heat requires the opt-in COARE diagnostic build and namelist together")
    inputs=load_data(args.inputs)
    evidence={}
    if diagnostic:
        if args.case_name!="superior" or not args.coare_forcing:
            raise ValueError("This Lake Superior recipe requires --case-name superior and --coare-forcing")
        spec=coare.configuration(nml,args.name)
        spec["ranks"]=req["execution"]["smoke_ranks"]
        derived=load_data(args.coare_build)
        coare.verify_build(derived,coare.build_assets(args.build_skill))
        if (derived.get("baseline_binding_sha256")!=digest(args.binding)
                or derived["baseline_executable_sha256"]!=binding["executable_sha256"]):
            raise ValueError("COARE build did not derive from the supplied native module/baseline binding")
        binary=derived["exe"]
        executable_sha=derived["sha256"]
        evidence={"coare_build.json":args.coare_build,"coare_forcing.json":args.coare_forcing}
        if args.name!="coare_reader":
            if not args.coare_reader:
                raise ValueError("Cold, warm and continuation windows require --coare-reader")
            coare.verify_reader(load_data(args.coare_reader),derived)
            evidence["coare_reader.json"]=args.coare_reader
        sources=coare.checked_sources(inputs,PurePosixPath(req["perlmutter"]["project_dir"]))
        files={k:v["sha256"] for k,v in sources.items()}
        coare.verify_forcing(nml,files,load_data(args.coare_forcing),spec)
    else:
        if any(getattr(args,k,None) for k in ("coare_reader","coare_forcing")):
            raise ValueError("COARE evidence requires an opt-in diagnostic build")
        spec={s["name"]:s for s in schedule(req)["stages"]}[args.name]
        executable_sha=binding["executable_sha256"]
        files=inputs
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
    if not files or not all(re.fullmatch(r"[A-Za-z0-9_.-]+",k) and k not in {".",".."} and
                            re.fullmatch(r"[0-9a-f]{64}",v) for k,v in files.items()):
        raise ValueError("inputs must be an exact map of filenames to content SHA-256")
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
    heat_keys=() if diagnostic else ("HEATING_FILE",)
    for k in ("STARTUP_FILE","GRID_FILE","DEPTH_FILE","SIGMA_LEVELS_FILE","WIND_FILE",*heat_keys,
              "AIRPRESSURE_FILE","PRECIPITATION_FILE","STATION_FILE","RIVER_INFO_FILE"):
        if key(k) not in files: raise ValueError("unfrozen input "+k)
    if key("STARTUP_TYPE").lower()!="hotstart": raise ValueError("lake stage must read its dated startup")
    facts=dict(schema="fvcom_lake_attempt_v1",name=aid,stage=spec,mpi_ranks=ranks,nodes=args.nodes,
               site="perlmutter",account=args.account,files=files,executable_sha256=executable_sha,
               executable=binary,build_binding_sha256=digest(args.binding),run_namelist_sha256=digest(args.namelist),
               remote_run=(PurePosixPath(site["project_dir"])/root.as_posix()).as_posix())
    out=args.project/root
    out.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        stage=Path(tmp)/"stage";stage.mkdir()
        shutil.copyfile(args.namelist,stage/(args.case_name+"_run.nml"))
        checks=[(v,"input/"+k) for k,v in sorted(files.items())]
        checks.extend([(facts["run_namelist_sha256"],args.case_name+"_run.nml"),(facts["executable_sha256"],binary)])
        if diagnostic:
            facts["schema"]="fvcom_coare_diagnostic_attempt_v1"
            facts["qualification"]={name:digest(path) for name,path in evidence.items()}
            case=dict(schema="fvcom_diagnostic_coare_case_v1",kind=args.name,mpi_ranks=ranks,inputs=sources,
                      start_utc=spec["start_utc"],end_utc=spec["end_utc"],executable=derived,
                      forcing_manifest_sha256=digest(args.coare_forcing),nml_sha256=facts["run_namelist_sha256"])
            (stage/"case.json").write_bytes((json.dumps(case,indent=2,sort_keys=True)+"\n").encode())
            facts["case_sha256"]=digest(stage/"case.json")
            checks.append((facts["case_sha256"],"case.json"))
            for name,path in evidence.items():
                shutil.copyfile(path,stage/name)
                checks.append((facts["qualification"][name],name))
        (stage/"runtime_checksums.sha256").write_text("".join(f"{v}  {k}\n" for v,k in checks),newline="\n")
        if diagnostic:
            args.coare_case_sha256=facts["case_sha256"]
            args.coare_inputs=sources
            args.coare_checks_sha256=digest(stage/"runtime_checksums.sha256")
        script=attempt_sbatch(args,aid,None)
        (stage/"job.sbatch").write_text(script,encoding="utf8",newline="\n")
        facts["job_sha256"]=digest(stage/"job.sbatch")
        (stage/"attempt_manifest.json").write_text(json.dumps(facts,indent=2,sort_keys=True)+"\n",encoding="utf8")
        if out.exists():
            if any(not (out/p.name).is_file() or digest(p)!=digest(out/p.name) for p in stage.iterdir()):
                raise FileExistsError(f"conflicting lake attempt {out}")
        else: stage.rename(out)
    return facts
