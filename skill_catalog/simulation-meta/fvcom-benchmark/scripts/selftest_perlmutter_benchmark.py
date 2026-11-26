"""Exercise CPU accounting, restart lineage and production selection with synthetic jobs."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import fvcom_benchmark as benchmark
import perlmutter_benchmark as cpu


class PerlmutterCPU(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = benchmark.make_plan(site="perlmutter", start_utc="2025-07-01T00:00:00Z")
        self.plan_path = self.root / "plan.json"
        benchmark.write_json(self.plan_path, self.plan)

    def collect(self, slot, model, wall, step_exit="0:0"):
        r, n = slot["ranks"], slot["nodes"]
        job = str(10 * r + slot["repeat"])
        name = "attempt_" + slot["stage"] + "_0000"
        att = self.root / name
        att.mkdir(exist_ok=True)
        def put(name, value):
            path = att / name
            benchmark.write_json(path, value)
            return path
        nml = att / "superior_run.nml"
        nml.write_text("START_DATE='2025-07-01 00:00:00'\nEND_DATE='2025-07-02 00:00:00'\n"
                       "STARTUP_TYPE='hotstart'\nSTARTUP_FILE='a.nc'\nEXTSTEP_SECONDS=2.5\nISPLIT=2\n"
                       "NC_FIRST_OUT='2025-07-01 00:00:00'\nNC_OUT_INTERVAL='seconds=21600.0'\n")
        script = att / "job.sbatch"
        script.write_text("srun --ntasks=" + str(r))
        script_sha, nml_sha = map(benchmark.sha256, (script, nml))
        remote = "/scratch/" + name
        files = {"a.nc":"a"*64, "grid.dat":"b"*64}
        put("attempt_manifest.json", dict(stage=dict(name=slot["stage"], ranks=r,
            start_utc=self.plan["simulated_start"],end_utc=self.plan["simulated_end"]),
            site="perlmutter", name=name, account="m3780", remote_run=remote, mpi_ranks=r,
            nodes=n, job_sha256=script_sha, run_namelist_sha256=nml_sha, files=files,
            executable="/scratch/exe",executable_sha256="e"*64, build_binding_sha256="d"*64))
        (att/"runtime_checksums.sha256").write_text("a"*64+"  input/a.nc\n"+"b"*64+"  input/grid.dat\n"
                           +nml_sha+"  superior_run.nml\n"+"e"*64+"  /scratch/exe\n")
        sub = put("slurm.json",dict(job_id=job,account="m3780",workdir=remote,script=remote+"/job.sbatch",
                                    script_sha256=script_sha))
        launch = put("launch.json",dict(site="perlmutter",job_id=job,mpi_ranks=r,constraint="cpu",
                                        threads_per_rank=1,executable_sha256="e"*64))
        terminal = put("terminal.json",dict(status="verified",utc=self.plan["simulated_end"],job_id=job,sha256="f"*64))
        stdout, sacct = att/"stdout.log",att/"sacct.psv"
        stdout.write_text("! 17200 2025-07-01T23:53:20\n"
                          "! 17280 2025-07-02T00:00:00.000000 0000:00:00:00 0.0039  |====|\nTADA!\n")
        sacct.write_text("JobID|State|ExitCode|NNodes|NTasks|ElapsedRaw|MaxRSS\n"
                         f"{job}|COMPLETED|0:0|{n}||{wall}|\n{job}.0|COMPLETED|{step_exit}|{n}|{r}|{model}|1M\n")
        restart = put("selected.json",dict(status="verified",utc=self.plan["simulated_start"],sha256="a"*64))
        return cpu.collect_cpu(self.plan_path,att,sub,launch,sacct,stdout,restart,terminal)

    def test_selection(self):
        # The two-node run is fastest but costs 160 node-seconds versus a 115 ceiling.
        # The 128-rank layout has 110 charged seconds and 72 model seconds.
        timings = {64:(95,100),128:(72,110),256:(65,80)}
        rows = [self.collect(s,*timings[s["ranks"]]) for s in self.plan["runs"]]
        self.assertEqual(rows[0]["fvcom_iterations"],17280)
        self.assertEqual(rows[0]["native_loop_seconds_per_iteration_at_end"],0.0039)
        self.assertEqual(rows[0]["peak_memory_bytes_per_task"],1048576)
        self.assertIsNone(rows[0]["max_disk_write_bytes_per_task"])
        result = benchmark.analyze(rows,self.plan)
        self.assertEqual(result["production_layout"],{"ranks":128,"nodes":1})
        self.assertEqual(benchmark.resume_plan(self.plan,rows[:-1])["pending"],[self.plan["runs"][-1]])
        with self.assertRaisesRegex(ValueError,"nine verified"):
            benchmark.analyze(rows[:-1],self.plan)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            benchmark.analyze([*rows,rows[0]],self.plan)
        altered = deepcopy(rows)
        altered[0]["job_id"]=altered[-1]["job_id"]
        with self.assertRaisesRegex(ValueError,"duplicate"):
            benchmark.analyze(altered,self.plan)
        altered[0]["job_id"]="42"
        altered[0]["restart_sha256"]="c"*64
        with self.assertRaisesRegex(ValueError,"lineage differs"):
            benchmark.analyze(altered,self.plan)
        altered[0]["restart_sha256"]=None
        with self.assertRaisesRegex(ValueError,"missing native lineage"):
            benchmark.analyze(altered,self.plan)
        with self.assertRaisesRegex(ValueError,"dated plan"):
            benchmark.analyze(rows,self.plan,"0"*64)

    def test_exit_signal(self):
        with self.assertRaisesRegex(ValueError,"signal"):
            self.collect(self.plan["runs"][0],60,100,"0:9")

    def test_plan_and_render(self):
        self.assertEqual(benchmark.make_plan()["cores_per_node"],104)
        with self.assertRaisesRegex(ValueError,"three repeats"):
            benchmark.make_plan(site="perlmutter",repeats=1)
        with self.assertRaisesRegex(ValueError,"lake-attempt"):
            benchmark.render_jobs(self.plan_path,None,None,"","", "", "", "", "", [])


if __name__=="__main__":
    unittest.main()
