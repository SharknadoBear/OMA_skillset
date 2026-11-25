"""Freeze a compiled Perlmutter freshwater lake executable and validate native physics probes."""
import argparse
from pathlib import Path, PurePosixPath
import re
import shutil

import fvcom_build as b


def finalize(evidence, binary, output, source_dir):
    read=lambda name:(evidence/name).read_text(encoding="utf-8",errors="replace")
    for name in b.MANDATORY_EVIDENCE + ["source_revision.json","lake_signed_donor.diff","lake_surface_units.diff",
                                       "perlmutter_lake_probe.f90","lake_physics.log","lake_reader.log"]:
        if not (evidence/name).is_file() or not (evidence/name).stat().st_size:
            raise ValueError("missing native lake build evidence "+name)
    freeze=b.load_json(evidence/"freeze.json")
    revision=b.load_json(evidence/"source_revision.json")
    env=freeze["build_environment"]
    job=str(freeze["job_id"])
    family=env["programming_environment"].removeprefix("PrgEnv-")
    if (not job.isdigit() or read("build_job_id.txt").strip()!=job or read("nersc_host.txt").strip()!="perlmutter"
            or "JobId="+job not in read("scheduler_job.txt").split() or "Features=cpu" not in read("scheduler_job.txt")
            or freeze.get("execution_site")!="perlmutter" or env.get("site")!="perlmutter" or env.get("compiler")!="ftn"
            or family not in {"gnu","cray"} or any(module not in read("modules.txt") for module in env["modules"])):
        raise ValueError("site, native CPU compiler and scheduler binding disagree")
    if (revision["source_commit"]!="f8e4b50d4fd51409febcefc763cbc5dca11870e2" or
            revision["source_repository"]!="https://github.com/SharknadoBear/FVCOM_MP.git" or
            any(freeze[key]!=revision[key] for key in ("source_repository","source_commit"))):
        raise ValueError("freshwater patches require their exact FVCOM_MP source pin")
    sha=b.sha256(binary)
    if (sha!=b.parse_checksum_file(evidence/"executable.sha256")[0] or sha!=freeze["executable_sha256"]
            or freeze.get("source_edits_after_freeze_allowed") is not False or "not found" in read("executable.ldd.txt").lower()):
        raise ValueError("executable or dynamic-link verification failed")
    build_dir=str(PurePosixPath(source_dir)/f"build_{job}_{family}")
    make=b.audit_make_inc(evidence/"make.inc.after",build_dir,air_pressure=True,spherical=True)
    if (freeze.get("build_source_dir")!=build_dir or make["status"]!="pass" or
            make["sha256"]!=b.load_json(evidence/"make_inc_audit.json")["sha256"] or
            "FRESHWATER_LAKE" not in make["active_defines"]):
        raise ValueError("native build must enable spherical freshwater pressure and 3-D dynamics")
    assets=Path(__file__).resolve().parents[1]/"assets"
    for name,archived in (("fvcom_mp_lake_signed_donor.patch","lake_signed_donor.diff"),
                          ("fvcom_mp_lake_surface_units.patch","lake_surface_units.diff"),
                          ("perlmutter_lake_probe.f90","perlmutter_lake_probe.f90")):
        if (assets/name).read_bytes().replace(b"\r\n",b"\n")!=(evidence/archived).read_bytes():
            raise ValueError("native source/probe differs from the freshwater skill: "+name)
    commands=read("build.log").splitlines()
    for unit in ("bcond_gcn.F","mod_main.F","mod_prec.F","adv_t.F","adv_s.F","extel_edge.F","eqs_of_state.F"):
        if not any("cpp " in line and "-DFRESHWATER_LAKE " in line and "-DSPHERICAL " in line and unit in line.split() for line in commands):
            raise ValueError("missing compiled freshwater source "+unit)
    if "FVCOM_MPI_NETCDF_PASS ranks=2 value=3" not in read("mpi_netcdf_probe.log").splitlines():
        raise ValueError("MPI/netCDF two-rank readback failed")
    physics=read("lake_physics.log").splitlines()
    if ("FVCOM_LAKE_EOS_SIGNED_EDGE_PASS" not in physics or
            sum(line.startswith("EOS ") for line in physics)!=33 or sum(line.startswith("SIGNED_EDGE ") for line in physics)!=6):
        raise ValueError("native EOS or signed edge tracer probe failed")
    reader=read("lake_reader.log").lower()
    if ("subroutine ends: read_name_list;" not in reader or not re.search(
            r"expected_read_end\.nc; can not be opend by nf90_open|expected_read_end\.dat not found", reader)):
        raise ValueError("native freshwater namelist reader did not reach its sentinel")
    output.mkdir(parents=True,exist_ok=True)
    blank=evidence/"blank_namelist.nml"
    binding=dict(schema="fvcom_executable_reuse_binding_v1",status="frozen",execution_site="perlmutter",site="perlmutter",
                 native_executable=str(PurePosixPath(source_dir)/"fvcom_plased"),build_job_id=job,job_id=job,
                 executable_sha256=sha,source_commit=revision["source_commit"],feature_defines=make["active_defines"],
                 build_environment=env,blank_template_sha256=b.sha256(blank),native_freeze_sha256=b.sha256(evidence/"freeze.json"),
                 lake_physics_probe_sha256=b.sha256(evidence/"lake_physics.log"))
    path=output/"executable_reuse_binding.json"
    if path.exists() and b.load_json(path)!=binding:
        raise ValueError("target contains a different frozen build")
    b.write_json(path,binding)
    target=output/"blank_namelist.nml"
    if target.exists() and b.sha256(target)!=binding["blank_template_sha256"]:
        raise ValueError("target contains a different blank")
    if not target.exists():
        shutil.copyfile(blank,target)
    return binding


def cli(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("evidence","binary","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--source-dir",required=True)
    args=p.parse_args(argv)
    print(finalize(args.evidence,args.binary,args.output,args.source_dir)["executable_sha256"])
    return 0


if __name__=="__main__":
    raise SystemExit(cli())
