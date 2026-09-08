from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path

import netCDF4 as nc4
import numpy as np

from fvcom_run_control import align_external_step, audit_run, build_parser, create_attempt, derive_controls, evaluate_preparation_join, record_command, render_report, stability_plan


def main() -> int:
    assert align_external_step(1.3, 4, 360) == 1.2
    assert align_external_step(1.2, 1, 360) == 1.2
    assert align_external_step(0.9, 2, 360) == 0.9
    aligned = stability_plan({"extstep_seconds": 1.2, "isplit": 4,
                              "time_alignment": {"quantum_seconds": 360}})
    for a in aligned["attempts"]:
        assert 3600 % (round(a["extstep_seconds"] * 10) * a["isplit"]) == 0
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mesh = root / "tiny.2dm"
        mesh.write_text(
            "MESH2D\nE3T 1 1 2 3 1\nE3T 2 1 3 4 1\n"
            "ND 1 0 0 5\nND 2 100 0 5\nND 3 100 100 10\nND 4 0 100 10\n"
            "NS 1 -2\nNS 3 -4\n", encoding="utf-8"
        )
        controls = derive_controls(mesh)
        assert controls["extstep_seconds"] > 0
        assert controls["isplit"] >= 1
        assert len(controls["sponge"]) == 2
        plan = stability_plan(controls)
        assert plan["attempts"][0]["extstep_multiplier"] == 1.0
        keys = {(a["extstep_multiplier"], a["isplit"], a["sponge_radius_multiplier"], a["sponge_coefficient_multiplier"]) for a in plan["attempts"]}
        assert len(keys) == plan["attempt_count"]
        project = root / "project"
        base_input = root / "base_input"; base_input.mkdir()
        (base_input / "tiny_spg.dat").write_text(
            "Sponge Node Number = 2\n1 100.000000 0.002500 \n2 100.000000 0.002500 \n",
            encoding="ascii",
        )
        base_nml = root / "case_run.nml"
        base_nml.write_text(
            "&NML_IO\n INPUT_DIR = 'shared/base',\n OUTPUT_DIR = 'shared/output',\n/\n"
            "&NML_INTEGRATION\n EXTSTEP_SECONDS = 9.9,\n ISPLIT = 9,\n/\n",
            encoding="utf-8",
        )
        plan_file = root / "plan.json"; plan_file.write_text(json.dumps(plan), encoding="utf-8")
        created = create_attempt(Namespace(
            project=str(project), plan=str(plan_file), index=0, grid_case="accepted_t6v6",
            base_input=str(base_input), base_namelist=str(base_nml),
            stage_namelist=[],
            revision=0, module=[],
            remote_executable="/scratch/fvcom", case_name="tiny", account="test",
            partition="debug", nodes=1, ranks=1, walltime="00:05:00",
        ))
        attempt_nml = (project / created["run_dir"] / "tiny_run.nml").read_text(encoding="utf-8")
        assert f"EXTSTEP_SECONDS = {plan['attempts'][0]['extstep_seconds']:.1f}" in attempt_nml
        assert f"ISPLIT = {plan['attempts'][0]['isplit']}" in attempt_nml
        assert "INPUT_DIR = '../../../../input/accepted_t6v6/attempts/attempt_0000'" in attempt_nml
        assert "OUTPUT_DIR = '.'" in attempt_nml
        assert created["namelist_input_dir"] == "../../../../input/accepted_t6v6/attempts/attempt_0000"
        assert (project / created["input_dir"] / "tiny_spg.dat").read_text(encoding="ascii").count("0.002500") == 2
        assert created["execution_layout"]["cpu_bind"] is None
        assert not created["execution_layout"]["exclusive"]
        staged_project = root / "staged_project"
        staged = create_attempt(Namespace(
            project=str(staged_project), plan=str(plan_file), index=0, grid_case="fresh_reproduction",
            base_input=str(base_input), base_namelist=None,
            stage_namelist=[f"smoke={base_nml}", f"canary={base_nml}", f"spinup={base_nml}"],
            revision=1, module=["intel/2023.2.0", "netcdf-fortran/4.6.1-intel"], cpu_bind="cores", exclusive=True,
            remote_executable="/scratch/fvcom", case_name="tiny", account="test",
            partition="debug", nodes=1, ranks=1, walltime="00:05:00",
        ))
        assert staged["stage_order"] == ["smoke", "canary", "spinup"]
        assert staged["revision"] == 1
        assert staged["execution_layout"] == {"ranks": 1, "nodes": 1, "cpu_bind": "cores", "exclusive": True}
        assert staged["namelist_input_dir"] == "../../../../../input/fresh_reproduction/attempts/attempt_0000_r001"
        for stage in staged["stage_order"]:
            stage_dir = staged_project / staged["stages"][stage]["run_dir"]
            staged_nml = (stage_dir / "tiny_run.nml").read_text(encoding="utf-8")
            assert staged["namelist_input_dir"] in staged_nml
            assert (stage_dir / "job.sbatch").is_file()
            sbatch = (stage_dir / "job.sbatch").read_text(encoding="utf-8")
            assert "module purge" in sbatch and "module load intel/2023.2.0" in sbatch
            assert "#SBATCH --exclusive\n" in sbatch and "srun --cpu-bind=cores --ntasks=1 " in sbatch
        log = root / "stdout.log"
        log.write_text(
            "END_DATE = 2025-04-01 00:00:00\n! IEND = 1\n"
            "! 1 D000000T00:00: 1.000000\nTADA!\n", encoding="utf-8"
        )
        product = root / "out.nc"; product.write_bytes(b"not-netcdf")
        result = audit_run(log, None, 0, "2025-04-01T00:00:00Z", [], [])
        assert result["stable"]
        iso_log = root / "iso_stdout.log"
        iso_log.write_text(
            "! (Date Time=2025-03-25T01:00:00.000000Z)\n! IEND = 3000\n"
            "! 3000 2025-03-25T01:00:00.000000 0000:00:00:00 0.0060\nTADA!\n",
            encoding="utf-8",
        )
        assert audit_run(iso_log, None, 0, "2025-03-25T01:00:00Z", [], [])["stable"]
        valid_product = root / "valid.nc"
        with nc4.Dataset(valid_product, "w") as ds:
            ds.createDimension("time", 1); ds.createDimension("DateStrLen", 26)
            times = ds.createVariable("Times", "S1", ("time", "DateStrLen"))
            times[0, :] = np.frombuffer(b"2025/04/01 00:00:00.000000", dtype="S1")
        assert audit_run(log, None, 0, "2025-04-01T00:00:00Z", [valid_product], [])["stable"]
        corrupt = audit_run(log, None, 0, "2025-04-01T00:00:00Z", [product], [])
        assert not corrupt["stable"]
        assert "missing_or_corrupt_required_product" in corrupt["failure_reasons"]
        assert corrupt["products"][0]["check"].startswith("unreadable:")
        blow = root / "blow.log"; blow.write_text("NaN blowup\n", encoding="utf-8")
        assert not audit_run(blow, None, 1, "2025-04-01T00:00:00Z", [], [])["stable"]
        tge = root / "tge.log"
        tge.write_text(
            "SORRY, THE BOUNDARY CELL 6269 IS NOT GOOD FOR MODEL.\nSTOP RUNNING...\n",
            encoding="utf-8",
        )
        tge_audit = audit_run(tge, None, 1, "2025-04-01T00:00:00Z", [], [])
        assert tge_audit["blowup_or_fatal"]
        assert "blowup_or_fatal_text" in tge_audit["failure_reasons"]
        timeout = root / "timeout.log"; timeout.write_text("slurmstepd: TIME LIMIT\n", encoding="utf-8")
        assert not audit_run(timeout, None, 1, "2025-04-01T00:00:00Z", [], [])["stable"]
        missing = root / "missing.log"; missing.write_text("TADA!\n", encoding="utf-8")
        assert not audit_run(missing, None, 0, "2025-04-01T00:00:00Z", [], [])["stable"]
        echo_only = root / "echo_only.log"
        echo_only.write_text("END_DATE = 2025-04-01 00:00:00\nTADA!\n", encoding="utf-8")
        echo_audit = audit_run(echo_only, None, 0, "2025-04-01T00:00:00Z", [], [])
        assert not echo_audit["final_timestamp_present"]
        report_project = root / "report_project"
        nested_status = report_project / "run" / "accepted_t6v6" / "attempts" / "attempt_0000_r001" / "smoke" / "status.json"
        nested_status.parent.mkdir(parents=True)
        nested_status.write_text(json.dumps({"status": "stable", "failure_reasons": []}), encoding="utf-8")
        benchmark_submission = report_project / "run" / "accepted_t6v6" / "benchmark" / "matrix_001" / "benchmark_submission.json"
        benchmark_submission.parent.mkdir(parents=True)
        benchmark_submission.write_text(json.dumps({
            "attempt": 4, "status": "submitted", "execution": "scientific_gate_afterok_sequential",
            "requested_walltime": "24:00:00", "jobs": [{"job_id": "12345"}], "blocking_reasons": [],
        }), encoding="utf-8")
        benchmark_record = benchmark_submission.parent / "rank_0001" / "benchmark_record.json"
        benchmark_record.parent.mkdir()
        benchmark_record.write_text(json.dumps({
            "job_id": "12345", "eligible": False, "slurm_state": "RUNNING", "tada": False,
            "final_timestamp_present": False, "wall_seconds": 30,
        }), encoding="utf-8")
        report_path = render_report(report_project)
        report_text = report_path.read_text(encoding="utf-8")
        assert "attempt_0000_r001" in report_text and "smoke" in report_text
        assert "Benchmark attempts" in report_text and "12345" in report_text and "24:00:00" in report_text
        (root / "commands.jsonl").touch()
        recorded = record_command(root, "test", "fvcom --create_namelist=blank", "success", ["blank.nml"])
        assert recorded["kind"] == "test"
        assert json.loads((root / "commands.jsonl").read_text(encoding="utf-8"))["status"] == "success"
        parsed_record = build_parser().parse_args([
            "record", "--project", str(root), "--kind", "test", "--command", "safe command",
        ])
        assert parsed_record.command == "record" and parsed_record.command_text == "safe command"
        manifests = []
        for index in range(3):
            path = root / f"worker_{index}.json"
            path.write_text(json.dumps({
                "schema": f"worker_{index}_v1", "worker": ["grid", "tpxo", "configuration_build"][index],
                "status": "ready", "artifact_root": f"role_{index}",
                "hashes": {"artifact": str(index) * 64}, "provenance": {}, "warnings": [],
                "blocking_reasons": [], "resume_token": f"role:{index}:ready",
            }), encoding="utf-8")
            manifests.append(path)
        assert evaluate_preparation_join(manifests)["ready"]
        original = json.loads(manifests[1].read_text())
        manifests[1].write_text(json.dumps(dict(original, blocking_reasons=["unresolved scientific input"])))
        assert not evaluate_preparation_join(manifests)["ready"]
        manifests[1].write_text(json.dumps(dict(original, worker="grid")))
        try:
            evaluate_preparation_join(manifests)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate preparation roles accepted")
        manifests[1].write_text(json.dumps(original))
        blocked = json.loads(manifests[1].read_text(encoding="utf-8"))
        blocked["status"] = "blocked"
        blocked["blocking_reasons"] = [{"code": "registered_source_locator_required"}]
        manifests[1].write_text(json.dumps(blocked), encoding="utf-8")
        assert evaluate_preparation_join(manifests)["status"] == "waiting_user"
        print(json.dumps({"status": "pass", "attempt_count": plan["attempt_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
