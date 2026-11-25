"""Exercise site selection, native Slurm rendering, stage limits and immutable launch hashes."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from freeze_attempt_inputs import freeze_attempt
from fvcom_profiles import COMBINED_MODE, FACTORIAL_PROFILE, execution_site, resolve_request
from fvcom_run_control import attempt_sbatch, build_parser, create_attempt, init_project, sha256


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def request():
    return dict(
        schema="simulation_request_v1", mode=COMBINED_MODE, profile=FACTORIAL_PROFILE, case_id="pilot",
        river=dict(provider="usgs", station_id="14246900", parameter_code="72137"),
        execution=dict(site="perlmutter"), perlmutter=dict(
            account="test-project", project_dir="/test/pilot/combined", source_dir="/test/pilot/source",
            build_environment=dict(compiler="ftn",modules=["PrgEnv-gnu/1","cray-netcdf/1"]),
        ),
    )


class SiteTests(unittest.TestCase):
    def test_native_launch_receipt_before_mpi(self):
        bash = shutil.which("bash")
        if os.name == "nt":
            git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
            if git_bash.is_file():
                bash = str(git_bash)
        if not bash:
            self.skipTest("A Bash runtime is needed for a local Slurm-script replay")
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp).resolve()
            run = project / "run/retained_columbia/attempts/attempt_smoke_0000"
            run.mkdir(parents=True)
            binary = project / "fvcom"
            binary.write_bytes(b"replay binary")
            science = run / "tide.nc"
            science.write_bytes(b"replay forcing")
            cfg = resolve_request(request())["perlmutter"]
            cfg["project_dir"] = project.as_posix()
            args = Namespace(execution_site="perlmutter",site_configuration=cfg,job_prefix="pilot",
                             grid_case="retained_columbia",account=cfg["account"],nodes=1,ranks=52,
                             walltime=cfg["walltime"],case_name="columbia",module=["PrgEnv-gnu/1"],
                             remote_executable=binary.as_posix())
            job = run / "job.sbatch"
            job.write_text(attempt_sbatch(args,run.name,None),encoding="utf-8",newline="\n")
            (run/"runtime_checksums.sha256").write_text(
                f"{sha256(binary)}  {binary.as_posix()}\n{sha256(science)}  {science.name}\n",newline="\n")
            shell = r'''
module() {
  case "$1:$2" in
   reset:) LOADEDMODULES=gpu/1:cudatoolkit/1 ;;
   is-loaded:gpu|is-loaded:cudatoolkit|is-loaded:cpu) [[ ":${LOADEDMODULES}:" == *:"$2"/*:* ]] ;;
   swap:gpu) LOADEDMODULES=cpu/1:cudatoolkit/1 ;;
   unload:cudatoolkit) LOADEDMODULES=cpu/1 ;;
   load:PrgEnv-gnu/1) LOADEDMODULES="$LOADEDMODULES:PrgEnv-gnu/1${TEST_ACCEL:-}" ;;
  esac
}
scontrol() { echo "JobId=$SLURM_JOB_ID Features=${TEST_CONSTRAINT:-cpu}"; }
srun() {
  test -s launch_identity.json
  printf '%s\n' "$@" > mpi_argv
}
source ./job.sbatch
'''
            env = dict(os.environ,NERSC_HOST="perlmutter",SLURM_JOB_ID="456",SLURM_NTASKS="52")
            def execute(**extras):
                return subprocess.run([bash,"-c",shell],cwd=run,env=dict(env,**extras),
                    capture_output=True,text=True,check=False)
            for overrides in (dict(TEST_ACCEL=":craype-accel-nvidia80"),
                              dict(TEST_CONSTRAINT="gpu"),dict(SLURM_NTASKS="51")):
                with self.subTest(overrides=overrides):
                    self.assertNotEqual(execute(**overrides).returncode,0)
                    self.assertFalse((run/"launch_identity.json").exists())
                    self.assertFalse((run/"mpi_argv").exists())
            science.write_bytes(b"modified tide")
            self.assertNotEqual(execute().returncode,0)
            self.assertFalse((run/"launch_identity.json").exists())
            science.write_bytes(b"replay forcing")
            result = execute()
            self.assertEqual(result.returncode,0,result.stderr)
            evidence=json.loads((run/"launch_identity.json").read_bytes())
            self.assertEqual((evidence["site"],evidence["job_id"],evidence["mpi_ranks"],
                              evidence["executable_sha256"]),("perlmutter","456",52,sha256(binary)))
            self.assertIn("--ntasks=52",(run/"mpi_argv").read_text().splitlines())
            self.assertNotEqual(execute(SLURM_JOB_ID="457").returncode,0)
            self.assertEqual(json.loads((run/"launch_identity.json").read_bytes()),evidence)

    def test_select_site_without_changing_science(self):
        payload = request()
        native = resolve_request(payload)
        ref = resolve_request({k:v for k,v in payload.items() if k not in ("execution","perlmutter")})
        self.assertEqual(execution_site(ref), "kestrel")
        self.assertEqual(ref["kestrel"]["os_preference"],"rh9")
        self.assertEqual(native["perlmutter"]["source_commit"],ref["kestrel"]["source_commit"])
        self.assertNotIn("kestrel",native)
        self.assertEqual(native["execution"]["stages"],["smoke"])
        self.assertEqual(native["physics"],ref["physics"])
        self.assertEqual(native["period"],ref["period"])
        self.assertEqual(native, resolve_request(native))
        for edits in ({"kestrel":{}}, {"execution":{"site":"unknown"}},
                      {"execution":{"site":"perlmutter","stages":["smoke","production"]}},
                      {"perlmutter":dict(payload["perlmutter"],constraint="gpu")},
                      {"perlmutter":dict(payload["perlmutter"],project_dir="$SCRATCH/pilot")},):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                resolve_request(dict(payload,**edits))

    def test_attempt_and_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "combined"
            spec = request()
            meta = init_project(save(root/"request.json",spec),project)
            self.assertEqual(meta["perlmutter_root"],spec["perlmutter"]["project_dir"])
            self.assertNotIn("kestrel_root",meta)
            resolved=resolve_request(spec)
            binary="/test/pilot/source/fvcom"
            binding=dict(status="frozen",native_executable=binary,execution_site="perlmutter",
                executable_sha256="1"*64,source_commit=resolved["perlmutter"]["source_commit"],
                feature_defines=["MULTIPROCESSOR","AIR_PRESSURE","WET_DRY","GCN"],
                build_environment=spec["perlmutter"]["build_environment"])
            binding_path=save(project/"run/build/executable_reuse_binding.json",binding)
            receipt=save(root/"installed.json",dict(commit="a"*40,installed_root=str(root),
                files={"request.json":sha256(root/"request.json")}))
            base=project/"input/retained_columbia/base"
            (base/"columbia_spg.dat").write_text("Sponge Node Number = 1\n1 100.0 0.0025\n")
            (base/"station_mapping.json").write_text("{}")
            nml=root/"run.nml"
            nml.write_text("START_DATE='2025-03-18'\nEND_DATE='2025-03-19'\nSTARTUP_TYPE='coldstart'\n"
                           "INPUT_DIR='./'\nOUTPUT_DIR='./'\nEXTSTEP_SECONDS=1.2\nISPLIT=1\n")
            ctl=save(root/"controls.json",dict(extstep_seconds=1.2,isplit=1))
            args=["attempt","--project",str(project),"--grid-case","retained_columbia","--index","0",
                  "--stage","smoke","--controls",str(ctl),"--base-input",str(base),"--base-namelist",str(nml),
                  "--remote-executable",binary,"--case-name","columbia","--skill-installation",str(receipt)]

            def attempt(extra=()):
                return create_attempt(build_parser().parse_args(args + list(extra)))

            # Other integration tests exercise the physical reader against real forcing.
            with patch("fvcom_run_control.verify_stage_forcing") as verify, patch("freeze_attempt_inputs.verify_stage_forcing"):
                for extra in (["--stage","production"],["--index","1"],["--account","wrong"],
                              ["--partition","debug"],["--ranks","104"],["--remote-executable","/test/old/kestrel"]):
                    with self.subTest(extra=extra), self.assertRaises(ValueError):
                        attempt(extra)
                save(binding_path,dict(binding,execution_site="kestrel"))
                with self.assertRaises(ValueError):
                    attempt()
                save(binding_path,binding)
                manifest=attempt()
                self.assertTrue(verify.called)
                a=project/manifest["run_dir"]
                job=(a/"job.sbatch").read_text()
                for present in ("#!/bin/bash -l","#SBATCH --constraint=cpu","#SBATCH --qos=debug",
                                "#SBATCH --account=test-project","#SBATCH --nodes=1","#SBATCH --ntasks=52",
                                "#SBATCH --ntasks-per-node=52","#SBATCH --cpus-per-task=1",
                                "#SBATCH --time=00:30:00","export OMP_NUM_THREADS=1","--cpu-bind=cores",
                                "module load PrgEnv-gnu/1","sha256sum --check --quiet runtime_checksums.sha256",
                                "#SBATCH --chdir=/test/pilot/combined/run/retained_columbia/",binary):
                    self.assertIn(present,job)
                self.assertNotIn("#SBATCH --partition",job)
                self.assertNotIn("oneapi",job)
                self.assertEqual(freeze_attempt(project,a),freeze_attempt(project,a))
                with self.assertRaises(FileExistsError):
                    attempt()
                (a/"runtime_checksums.sha256").write_text("wrong")
                with self.assertRaisesRegex(ValueError,"checksum"):
                    freeze_attempt(project,a)


if __name__ == "__main__":
    unittest.main()
