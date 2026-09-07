from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fvcom_benchmark import analyze, collect_record, load_records, make_plan, render_jobs, resume_plan


def main() -> int:
    plan = make_plan()
    assert plan["layouts"][-1] == {"ranks": 416, "nodes": 4, "repeat": 1}
    records = []
    for ranks, nodes, wall in [(1, 1, 1000), (52, 1, 40), (104, 1, 25), (208, 2, 18), (416, 4, 17)]:
        for repeat in range(1, 4):
            records.append({
                "ranks": ranks, "nodes": nodes, "wall_seconds": wall + repeat - 2,
                "simulated_seconds": 86400, "fvcom_iterations": 1000,
                "queue_seconds": 10, "peak_memory_mb": 1000 + ranks, "io_seconds": 1,
                "tada": True, "final_timestamp_present": True, "exit_code": 0,
                "executable_sha256": "a", "restart_sha256": "b", "input_bundle_sha256": "c",
                "simulated_start": "2025-04-01T00:00:00Z", "simulated_end": "2025-04-02T00:00:00Z",
                "repeat": repeat,
            })
    result = analyze(records)
    assert result["fastest"]["ranks"] == 416
    assert result["least_node_hour"]["ranks"] in {104, 208, 416}
    assert result["extension_required"]
    assert all(item["additional_repeats"] == 0 for item in result["pareto_neighborhood_repeat_plan"])
    resumed = resume_plan(plan, [records[0]])
    assert not resumed["plan_complete"]
    assert not any(item["ranks"] == 1 and item["repeat"] == 1 for item in resumed["pending"])
    assert any(item["ranks"] == 2 and item["repeat"] == 1 for item in resumed["pending"])
    bad = [dict(records[0]), dict(records[1])]
    bad[1]["restart_sha256"] = "changed"
    try:
        analyze(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("lineage mismatch was accepted")
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan_path = root / "plan.json"
        small_plan = make_plan(); small_plan["layouts"] = small_plan["layouts"][:2]
        plan_path.write_text(json.dumps(small_plan), encoding="utf-8")
        bom_records = root / "bom_records.json"
        bom_records.write_text(json.dumps({"records": records[:2]}), encoding="utf-8-sig")
        assert len(load_records(bom_records)) == 2
        bom_jsonl = root / "bom_records.jsonl"
        bom_jsonl.write_text("\n".join(json.dumps(row) for row in records[:2]), encoding="utf-8-sig")
        assert len(load_records(bom_jsonl)) == 2
        nml = root / "benchmark.nml"
        nml.write_text(
            "&NML_CASE\n START_DATE = '2025-04-01 00:00:00',\n END_DATE = '2025-04-02 00:00:00',\n/\n"
            "&NML_STARTUP\n STARTUP_TYPE = 'hotstart',\n STARTUP_FILE = 'restart.nc',\n/\n"
            "&NML_IO\n INPUT_DIR = 'old',\n OUTPUT_DIR = 'old',\n/\n"
            "&NML_INTEGRATION\n EXTSTEP_SECONDS = 1.2,\n/\n",
            encoding="utf-8",
        )
        rendered = render_jobs(plan_path, nml, root / "jobs", "/scratch/case/input", "/scratch/fvcom",
                               "galveston", "test", "debug", "01:00:00", ["intel/2023.2.0"], 2)
        assert len(rendered["jobs"]) == 2
        first = Path(rendered["jobs"][0]["run_dir"])
        assert first.name.endswith("attempt_002")
        assert "#SBATCH --exclusive" in (first / "job.sbatch").read_text(encoding="utf-8")
        job_text = (first / "job.sbatch").read_text(encoding="utf-8")
        assert "grep -Fq 'TADA!' stdout.log" in job_text
        assert "2025-04-02T00:00:00([.]0+)?" in job_text
        assert rendered["post_run_gate"]["failure_exit_codes"]["missing_tada"] == 41
        assert rendered["requested_walltime"] == "01:00:00"
        assert "'/scratch/case/input/'" in (first / "galveston_run.nml").read_text(encoding="utf-8")
        assert rendered["remote_input_dir_characters"] == len("/scratch/case/input/")
        try:
            render_jobs(plan_path, nml, root / "too_long", "/" + "a" * 80, "/scratch/fvcom",
                        "galveston", "test", "debug", "01:00:00", ["intel/2023.2.0"])
        except ValueError as exc:
            assert "CHARACTER(LEN=80)" in str(exc)
        else:
            raise AssertionError("overlength FVCOM INPUT_DIR was accepted")
        try:
            render_jobs(plan_path, nml, root / "bad_walltime", "/scratch/case/input", "/scratch/fvcom",
                        "galveston", "test", "debug", "tomorrow", ["intel/2023.2.0"])
        except ValueError as exc:
            assert "walltime" in str(exc)
        else:
            raise AssertionError("invalid Slurm walltime was accepted")
        stdout = root / "stdout.log"
        stdout.write_text(
            " !   71999 2025-04-01T23:59:58.800000 0000:00:00:00 0.0100 |=================== |\n"
            " !   72000 2025-04-02T00:00:00.000000 0000:00:00:00 0.0100 |====================|\n"
            " TADA!\n",
            encoding="utf-8",
        )
        sacct = root / "sacct.txt"
        sacct.write_text(
            "JobID|State|ExitCode|ElapsedRaw|Submit|Start|End|AllocCPUS|AllocNodes|MaxRSS|MaxDiskRead|MaxDiskWrite\n"
            "123|COMPLETED|0:0|720|2025-04-01T00:00:00|2025-04-01T00:00:10|2025-04-01T00:12:10|104|1|||\n"
            "123.0|COMPLETED|0:0|715|2025-04-01T00:00:00|2025-04-01T00:00:15|2025-04-01T00:12:10|104|1|245304K|255.93M|250.72M\n",
            encoding="utf-8",
        )
        collected = collect_record(stdout, sacct, nml, 104, 1, 1, "a", "b", "c", None, None)
        assert collected["eligible"] and collected["fvcom_iterations"] == 72000
        assert collected["queue_seconds"] == 10 and collected["peak_memory_mb"] > 239
        assert collected["io_seconds"] is None and collected["io_measurement_method"] == "unavailable"
    print(json.dumps({"status": "pass", "pareto_knee": result["pareto_knee"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
