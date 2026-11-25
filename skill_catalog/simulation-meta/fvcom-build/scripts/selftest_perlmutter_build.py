"""Offline contract regressions for native Perlmutter CPU build evidence."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tarfile
import tempfile
import unittest

import fvcom_build as fb
from prepare_perlmutter_build import package


class CpuBuildContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.artifacts = self.root / "artifacts"
        base = "/pscratch/sd/u/user/pilot"
        self.env = {
            "site": "perlmutter", "os": "cpe", "compiler": "ftn",
            "programming_environment": "PrgEnv-gnu",
            "modules": ["cpu/1", "PrgEnv-gnu/8", "cray-mpich/9",
                        "cray-hdf5-parallel/1", "cray-netcdf-hdf5parallel/4"],
            "compiler_options": ["-O2"],
        }
        self.request = fb.resolve_request({
            "case_id": "cpu_pilot", "mode": fb.COMBINED_MODE, "profile": fb.FACTORIAL_PROFILE,
            "river": {"provider": "usgs", "station_id": "14246900", "parameter_code": "72137"},
            "execution": {"site": "perlmutter"},
            "perlmutter": {"account": "project", "pair_root": base, "project_dir": base+"/combined",
                           "source_dir": base+"/source", "build_environment": self.env},
        })
        config = self.request["perlmutter"]
        revision = {k: config[k] for k in ("source_repository", "source_commit")}
        fb.write_json(self.evidence/"source_revision.json", revision)
        self.source = base + "/source/build_42_gnu"
        self.freeze = dict(revision, build_source_dir=self.source, build_environment=self.env,
                           source_edits_after_freeze_allowed=False, executable_sha256="a"*64,
                           job_id="42", execution_site="perlmutter")
        for key in fb.MANDATORY_EVIDENCE:
            (self.evidence/key).write_text("fixture\n")
        fb.write_json(self.evidence/"freeze.json", self.freeze)
        groups = dict(fb.PAPER_TEMPLATE_FIELDS, NML_CASE=(),
                      NML_RIVER_TYPE=("RIVER_NUMBER", "RIVER_TS_SETTING", "RIVER_INFLOW_LOCATION",
                                     "RIVER_KIND", "RIVER_INFO_FILE"),
                      NML_SURFACE_FORCING=("WIND_ON", "WIND_TYPE", "WIND_KIND", "WIND_FILE",
                                           "AIRPRESSURE_ON", "AIRPRESSURE_KIND", "AIRPRESSURE_VALUE",
                                           "AIRPRESSURE_FILE"))
        blank = "".join("&"+name+"\n"+"".join(key+" = 0\n" for key in keys)+"/\n" for name, keys in groups.items())
        reader = "Subroutine Ends: Read_Name_List;\nexpected_read_end.dat NOT FOUND\n"
        self.probes = {}
        for member in ("control", "combined"):
            wind = member == "combined"
            surface = ("WIND_ON=T\nWIND_TYPE='stress'\nWIND_KIND='variable'\nWIND_FILE='fields.nc'\n"
                       "AIRPRESSURE_KIND='variable'\nAIRPRESSURE_FILE='fields.nc'" if wind else
                       "WIND_ON=F\nAIRPRESSURE_KIND='constant'\nAIRPRESSURE_VALUE=101325.0")
            river = ("RIVER_NUMBER=19\nRIVER_KIND='variable'\nRIVER_TS_SETTING='calculated'\n"
                     "RIVER_INFLOW_LOCATION='node'\nRIVER_INFO_FILE='river.nml'" if wind else "RIVER_NUMBER=0")
            self.probes[member] = ("&NML_PHYSICS\nBAROTROPIC=T\n/\n&NML_SURFACE_FORCING\nAIRPRESSURE_ON=T\n"
                                   "HEATING_ON=F\nPRECIPITATION_ON=F\n"+surface+"\n/\n&NML_RIVER_TYPE\n"+river+"\n/\n"
                                   "INPUT_DIR='.'\nOUTPUT_DIR='.'\nOBC_NODE_LIST_FILE='grid.dat'\n")
        values = {
            "blank_namelist.nml": blank, "compiler.txt": "GNU Fortran",
            "smoke.log": "FVCOM_INPUT_SMOKE_PASS", "executable.file.txt": "ELF executable",
            "build_commands.txt": "ftn", "modules.txt": "\n".join(self.env["modules"]),
            "os-release.txt": "SUSE", "nersc_host.txt": "perlmutter", "build_job_id.txt": "42",
            "scheduler_job.txt": "JobId=42 Features=cpu", "wrapper_link.txt": "ftn -craype-verbose",
            "python_runtime.txt": "/usr/bin/python3.11\n3.11.10 (fixture)",
            "mpi_netcdf_probe.log": "FVCOM_MPI_NETCDF_PASS ranks=2 value=3", "perlmutter_mpi_netcdf.f90": "probe",
            "cpu_probe.nc": "fixture", "control_reader.log": reader, "combined_reader.log": reader,
            "probe_control_run.nml": self.probes["control"], "probe_combined_run.nml": self.probes["combined"],
            "executable.sha256": "a"*64+"  fvcom_plased",
            "build.log": "\n".join("cpp -DAIR_PRESSURE "+source for source in
                                   ("mod_obcs.F", "advave_edge_gcn.F", "external_step.F")),
        }
        for name, value in values.items():
            (self.evidence/name).write_text(value)
        self.make_inc("GFORTRAN")

    def make_inc(self, compiler):
        path = self.evidence/"make.inc.after"
        flags = fb.REQUIRED_DEFINES | {"AIR_PRESSURE", compiler}
        path.write_text(f"TOPDIR = {self.source}\nDEFS = "+" ".join("-D"+flag for flag in sorted(flags)))
        fb.write_json(self.evidence/"make_inc_audit.json", fb.audit_make_inc(path, self.source, True))

    def finalize(self):
        return fb.finalize(self.request, self.evidence, self.artifacts)

    def test_native_evidence_and_plan(self):
        self.assertEqual(self.finalize()["status"], "ready")
        plan = fb.build_plan(self.request, self.artifacts, str(self.source))
        self.assertEqual(plan["execution_site"], "perlmutter")
        self.assertIn("--ntasks=1", plan["authoritative_blank_namelist_command"])
        self.assertNotIn("hindcastra", plan["authoritative_blank_namelist_command"])

    def test_native_negative_controls(self):
        corrupt = [
            ("nersc_host.txt", "other"), ("mpi_netcdf_probe.log", "MPI initialized"),
            ("control_reader.log", "solver banner"), ("scheduler_job.txt", "JobId=142 Features=cpu"),
            ("build_job_id.txt", "142"), ("scheduler_job.txt", "JobId=42 Features=gpu"),
            ("modules.txt", "\n".join(self.env["modules"]+["craype-accel-nvidia80"])),
            ("compiler.txt", "Cray Fortran"),
            ("python_runtime.txt", "/usr/bin/python3\n3.6.15 (fixture)"),
            ("probe_combined_run.nml", self.probes["control"]),
        ]
        for filename, value in corrupt:
            with self.subTest(filename=filename, value=value):
                path = self.evidence/filename
                original = path.read_bytes()
                path.write_text(value)
                self.assertEqual(self.finalize()["status"], "blocked_compile")
                path.write_bytes(original)
        self.make_inc("CRAY")
        self.assertEqual(self.finalize()["status"], "blocked_compile")

    def test_site_module_and_binary_isolation(self):
        request = copy.deepcopy(self.request)
        env = request["perlmutter"]["build_environment"]
        env["modules"] = [item for item in env["modules"] if not item.startswith("cpu/")]
        with self.assertRaisesRegex(fb.ContractError, "Perlmutter modules"):
            fb.build_environment(request)
        binding = self.root/"old_binding.json"
        fb.write_json(binding, {"status": "frozen"})
        with self.assertRaisesRegex(fb.ContractError, "across execution sites"):
            fb.reuse_build(self.request, binding, self.evidence, self.root/"old.tgz", self.root/"bin",
                           self.root/"blank.nml", self.artifacts)

    def test_active_fortran_module_order(self):
        path = Path(__file__).resolve().parents[1]/"assets/perlmutter_dependencies.py"
        spec = importlib.util.spec_from_file_location("dependencies", path)
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        sources = [("mod_par.F", "MODULE MOD_PAR\nend module"),
                   ("particle.F", "module particle_class\nuse MOD_PAR, only: node\nend module"),
                   ("mod_esmf_nesting.F", "module nest\nuse mod_par ! lower case\nuse, intrinsic :: iso_c_binding\nend module")]
        self.assertEqual(tool.dependencies(sources),
                         {"particle.o": ["mod_par.o"], "mod_esmf_nesting.o": ["mod_par.o"], "mod_par.o": []})
        with self.assertRaisesRegex(ValueError, "Duplicate module"):
            tool.dependencies(sources+[("conflict.F", "module MOD_PAR")])

    def test_source_packaging_and_submit_layout(self):
        src = self.root/"source_package"
        src.mkdir()
        raw = src/"makefile"
        raw.write_text("all:\n\n")
        with tarfile.open(src/"pinned_source.tgz", "w:gz") as tar:
            tar.add(raw, arcname="makefile")
        revision = {"source_commit": self.request["perlmutter"]["source_commit"],
                    "source_archive_sha256": fb.sha256(src/"pinned_source.tgz"), "git_blobs": 1}
        fb.write_json(src/"source_revision.json", revision)
        (src/"source_inventory_before.sha256").write_text(f"{fb.sha256(raw)}  makefile\n")
        (src/"no_plastic_guard.patch").write_text("fixture")
        control, combined = src/"control.nml", src/"combined.nml"
        control.write_text(self.probes["control"])
        combined.write_text(self.probes["combined"])
        output = self.root/"payload.tgz"
        receipt = package(src, control, combined, combined, output, "project")
        self.assertEqual(receipt["sha256"], fb.sha256(output))
        self.assertFalse(receipt["compute_job_submitted"])
        with tarfile.open(output) as tar:
            job = tar.extractfile("build_cpu.sbatch").read().decode()
            self.assertIn("#SBATCH --constraint=cpu", job)
            self.assertIn("#SBATCH --account=project\n", job)
            for name in ("probe_control_run.nml", "probe_combined_run.nml"):
                self.assertIn(b"expected_read_end.dat", tar.extractfile(name).read())
            for line in tar.extractfile("payload.sha256").read().decode().splitlines():
                digest, name = line.split()
                self.assertEqual(hashlib.file_digest(tar.extractfile(name), "sha256").hexdigest(), digest)
        with self.assertRaises(FileExistsError):
            package(src, control, combined, combined, output, "project")
        (src/"source_inventory_before.sha256").write_text("0"*64+"  makefile\n")
        with self.assertRaisesRegex(ValueError, "Git-blob"):
            package(src, control, combined, combined, self.root/"changed.tgz", "project")


if __name__ == "__main__":
    unittest.main()
