"""Prepare a Perlmutter CPU freshwater COARE diagnostic build or verify its native readers."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import tarfile
import tempfile


def asset_root():
    installed = Path(__file__).resolve().parents[1] / "assets" / "freshwater_coare"
    return installed if installed.is_dir() else Path(__file__).resolve().parent


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def require(test, message):
    if not test:
        raise ValueError(message)


def safe_path(value):
    p = PurePosixPath(value)
    require(p.is_absolute() and ".." not in p.parts and "\n" not in value and "\r" not in value,
            "expected an absolute native compute path")
    return p


def check_changes(source, pin):
    exchange = "IF(PAR)CALL AEXCHANGE(NC,MYID,NPROCS,COARE_WIND_SPEED)"
    boundary = (source/"bcond_gcn.F").read_text(encoding="latin1")
    require(boundary.count(exchange) == 1, "native nodal wind diagnostic lacks its single MPI halo exchange")
    wrong = [name for name, record in pin["changed"].items()
             if digest(source / name) != record["after"]]
    require(not wrong, "transformed source differs from the qualified operator: " + ", ".join(wrong))
    require(digest(source / "coare26z.F") == pin["coare26z_sha256"], "COARE26Z source differs")


def check_binding(binding, pin):
    env = binding["build_environment"]
    require(binding["status"] == "frozen" and binding["execution_site"] == "perlmutter"
            and env["compiler"] == "ftn" and env["programming_environment"] == "PrgEnv-gnu"
            and binding["source_commit"] == pin["source_commit"], "expected the pinned Perlmutter GNU freshwater baseline")
    flags = set(binding["feature_defines"])
    require({"AIR_PRESSURE", "DOUBLE_PRECISION", "FRESHWATER_LAKE", "MULTIPROCESSOR",
             "SPHERICAL", "WET_DRY", "SINGLE_OUTPUT"}.issubset(flags)
            and not flags.intersection({"HEATING_CALCULATED", "LAKE_COARE_DIAG", "TWO_D_MODEL"}),
            "expected the frozen three-dimensional direct-heat baseline")
    require(all(re.fullmatch("[A-Za-z0-9_./+-]+", item) for item in env["modules"]), "invalid native module name")
    for key in ("executable_sha256", "native_freeze_sha256", "lake_physics_probe_sha256"):
        require(re.fullmatch("[a-f0-9]{64}", binding[key]), "invalid baseline digest: " + key)
    safe_path(binding["native_executable"])


def package(args):
    root = asset_root()
    pin = read(root/"source_pin.json")
    binding = read(args.binding)
    check_binding(binding, pin)
    base, evidence, work = map(safe_path, (args.baseline_source, args.baseline_evidence, args.remote_workdir))
    require(all(work != v and v not in work.parents and work not in v.parents for v in (base, evidence)),
            "use a separate diagnostic work directory")
    require(re.fullmatch("[A-Za-z][A-Za-z0-9_-]+", args.account) and not args.account.endswith("_g"),
            "expected a Perlmutter CPU allocation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        out = Path(temp)
        files = {n:root/n for n in ("patch_coare.py", "ledger.F", "source_pin.json", "check_fields.py")}
        files.update({"freshwater_coare.py":Path(__file__), "perlmutter_dependencies.py":root.parent/"perlmutter_dependencies.py"})
        for name, source in files.items():
            (out/name).write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
        (out/"base_binding.json").write_bytes(args.binding.read_bytes())
        body = (root/"build.sh").read_text()
        replacements = {"@ACCOUNT@":args.account, "@QOS@":args.qos, "@WORK@":shlex.quote(str(work)),
                        "@BASE@":shlex.quote(str(base)), "@EVIDENCE@":shlex.quote(str(evidence)),
                        "@MODULES@":" ".join(map(shlex.quote, binding["build_environment"]["modules"]))}
        for key,value in replacements.items():
            require(body.count(key) == 1, "build template key differs: " + key)
            body = body.replace(key, value)
        (out/"build.sbatch").write_text(body, encoding="utf8", newline="\n")
        ledger = "".join(f"{digest(f)}  {f.name}\n" for f in sorted(out.iterdir()))
        (out/"inputs.sha256").write_text(ledger, encoding="ascii", newline="\n")
        with tarfile.open(args.output, "x:gz") as archive:
            for f in sorted(out.iterdir()):
                archive.add(f, arcname=f.name, recursive=False)
    return dict(status="prepared", archive_sha256=digest(args.output), remote_workdir=str(work),
                baseline_executable_sha256=binding["executable_sha256"], source_commit=pin["source_commit"])


def baseline(args):
    binding, pin = read(args.binding), read(asset_root()/"source_pin.json")
    check_binding(binding, pin)
    evidence, source = args.evidence, args.source
    require(digest(evidence/"freeze.json") == binding["native_freeze_sha256"], "unbound baseline freeze")
    freeze = read(evidence/"freeze.json")
    revision = read(evidence/"source_revision.json")
    require(revision["source_commit"] == freeze["source_commit"] == pin["source_commit"]
            and revision["source_repository"] == pin["source_repository"]
            and freeze["executable_sha256"] == binding["executable_sha256"] == digest(binding["native_executable"])
            and freeze["build_source_dir"] == str(source.resolve())
            and digest(evidence/"make.inc.after") == digest(source/"make.inc"),
            "baseline source directory, revision, environment or binary differs")
    require(digest(evidence/"lake_physics.log") == binding["lake_physics_probe_sha256"]
            and "FVCOM_MPI_NETCDF_PASS ranks=2 value=3" in (evidence/"mpi_netcdf_probe.log").read_text(),
            "missing baseline EOS, signed edge or MPI/netCDF evidence")
    for name, record in pin["changed"].items():
        require(digest(source/name) == record["before"], "baseline source differs: " + name)
    require(digest(source/"coare26z.F") == pin["coare26z_sha256"], "baseline COARE26Z mismatch")
    return dict(status="baseline_verified", executable_sha256=binding["executable_sha256"])


def finish(args):
    pin, binding = read(asset_root()/"source_pin.json"), read(args.binding)
    check_binding(binding, pin)
    check_changes(args.source, pin)
    changes = read(args.evidence/"changes.json")
    require(changes == pin["changed"], "unexpected changed source inventory")
    defined = read(args.evidence/"active_defines.json")["active_defines"]
    require(set(defined) == set(binding["feature_defines"]) | {"HEATING_CALCULATED", "LAKE_COARE_DIAG"},
            "COARE compile flags differ")
    log = (args.evidence/"build.log").read_text(errors="replace").splitlines()
    for name in (*pin["changed"], "coare26z.F"):
        require(any(name in row.split() and "cpp " in row and "-DLAKE_COARE_DIAG " in row
                    and "-DHEATING_CALCULATED " in row for row in log), "missing preprocessed native unit: " + name)
    blank=(args.evidence/"blank_namelist.nml").read_text().upper()
    require("&NML_HEATING_CALCULATED" in blank and all(key in blank for key in
            ("HEATING_CALCULATE_ON","HEATING_FRESHWATER","COARE_VERSION","ZUU","ZTT","ZQQ")),
            "native blank lacks the calculated heat interface")
    binary = args.source/"fvcom_plased"
    require(digest(binary) != binding["executable_sha256"] and "not found" not in (args.evidence/"ldd.txt").read_text().lower(),
            "diagnostic binary or dynamic linkage invalid")
    return dict(schema="fvcom_coare_diagnostic_build_v1", scope="diagnostic", status="frozen",
                source_commit=pin["source_commit"], source_repository=pin["source_repository"], build_job=args.job_id,
                changed=changes, cpp=defined, exe=str(binary.resolve()), sha256=digest(binary),
                baseline_executable_sha256=binding["executable_sha256"], baseline_binding_sha256=digest(args.binding),
                native_blank_sha256=digest(args.evidence/"blank_namelist.nml"), reader_status="pending")


def verify_reader(args):
    import netCDF4 as nc
    sys.path.insert(0, str(asset_root()))
    from check_fields import axis, checks
    pin, freeze, launch = read(asset_root()/"source_pin.json"), read(args.freeze), read(args.launch)
    require(freeze["status"] == "frozen" and freeze["scope"] == "diagnostic" and freeze["changed"] == pin["changed"]
            and freeze["source_commit"] == pin["source_commit"] and freeze["sha256"] == launch["executable_sha256"]
            and {"HEATING_CALCULATED", "LAKE_COARE_DIAG"}.issubset(freeze["cpp"])
            and launch["site"] == "perlmutter" and int(launch["mpi_ranks"]) > 1,
            "native launch differs from qualified parallel diagnostic executable")
    case=read(args.case)
    require(launch["case_sha256"] == digest(args.case) and case["executable"]["sha256"] == freeze["sha256"]
            and case["mpi_ranks"] == launch["mpi_ranks"] and str(args.history.resolve().parent) == str(args.launch.resolve().parent),
            "case or history differs from native launch")
    source_hashes={}
    for file in (args.bulk, args.rain, args.hydrodynamic):
        source_hashes[file.name] = digest(file)
        require(case["inputs"][file.name]["sha256"] == source_hashes[file.name], "input digest differs: " + file.name)
    with nc.Dataset(args.history) as out, nc.Dataset(args.bulk) as bulk, nc.Dataset(args.rain) as rain, nc.Dataset(args.hydrodynamic) as hydro:
        times=axis(out)
        require(len(times)>=2 and all(a<b for a,b in zip(times,times[1:]))
                and times[0].isoformat().replace("+00:00","Z")==case["start_utc"]
                and times[-1].isoformat().replace("+00:00","Z")==case["end_utc"], "supply a closed benchmark reader stack covering its registered interval")
        results={t.isoformat():checks(out,i,bulk,rain,hydro) for i,t in enumerate(times) if i}
    return dict(status="verified", scope="diagnostic", job_id=launch["job_id"], executable_sha256=freeze["sha256"],
                input_sha256=source_hashes, history_sha256=digest(args.history), records=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("prepare")
    for key in ("baseline-source","baseline-evidence","remote-workdir","account"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--qos",choices=("debug","regular"),default="debug")
    p.add_argument("--binding",type=Path,required=True)
    p.set_defaults(fn=package)
    for cmd, fn in (("check-baseline",baseline),("freeze",finish)):
        p=commands.add_parser(cmd)
        for key in ("source","binding","evidence"):p.add_argument("--"+key,type=Path,required=True)
        if cmd=="freeze":p.add_argument("--job-id",required=True)
        p.set_defaults(fn=fn)
    p=commands.add_parser("verify-reader")
    for key in ("freeze","launch","case","history","bulk","rain","hydrodynamic"):
        p.add_argument("--"+key,type=Path,required=True)
    p.set_defaults(fn=verify_reader)
    args=parser.parse_args()
    print(json.dumps(args.fn(args),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
