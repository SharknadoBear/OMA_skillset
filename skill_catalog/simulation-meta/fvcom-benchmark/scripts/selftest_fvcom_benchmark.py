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
    focused_plan = make_plan(ranks=[52, 104, 156, 208], repeats=1, extend=False, max_ranks=416)
    focused_records = [r for r in records if r["ranks"] in (52, 104, 156, 208) and r["repeat"] == 1]
    focused = analyze(focused_records, focused_plan)
    assert focused["speedup_reference"]["ranks"] == 52
    assert focused["layouts"][0]["speedup"] == 1.0
    assert focused["layouts"][0]["parallel_efficiency"] == 1.0
    assert focused["layouts"][1]["speedup"] == 39 / 24
    assert not focused["extension_required"]
    assert all(x["additional_repeats"] == 0 for x in focused["pareto_neighborhood_repeat_plan"])
    singleton = analyze([r for r in records if r["ranks"] == 416 and r["repeat"] == 1], make_plan(ranks=[416], repeats=1, extend=False))
    assert singleton["layouts"][0]["speedup"] == 1.0 and not singleton["extension_required"]
    for failure in ({"eligible": False}, {"fatal_marker_present": True}, {"slurm_state": "FAILED"}, {"wall_seconds": 0, "fvcom_iterations": 0}):
        checked = analyze(focused_records + [dict(focused_records[0], **failure)], focused_plan)
        assert checked["failed_run_count"] == 1
        assert checked["eligible_run_count"] == len(focused_records)
    try:
        analyze([r for r in records if r["ranks"] == 416], focused_plan)
    except ValueError as exc:
        assert "absent" in str(exc)
    else:
        raise AssertionError("unplanned layout entered performance selection")
    for invalid in ([52, 52], [0, 104], []):
        try:
            make_plan(ranks=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid rank selection accepted")
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
            "&NML_INTEGRATION\n EXTSTEP_SECONDS = 1.2,\n ISPLIT = 4,\n/\n",
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
            " !   17999 2025-04-01T23:59:55.200000 0000:00:00:00 0.0100 |=================== |\n"
            " !   18000 2025-04-02T00:00:00.000000 0000:00:00:00 0.0100 |====================|\n"
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
        assert collected["eligible"] and collected["fvcom_iterations"] == 18000
        assert collected["internal_step_seconds"] == 4.8 and collected["last_logged_iint"] == 18000
        assert collected["queue_seconds"] == 10 and collected["peak_memory_mb"] > 239
        assert collected["io_seconds"] is None and collected["io_measurement_method"] == "unavailable"
        assert collected["max_disk_read_mb_per_task"] == 255.93
        original = sacct.read_text()
        for replacement, expected in [("0|0", 0.0), ("|", None)]:
            sacct.write_text(original.replace("255.93M|250.72M", replacement))
            missing = collect_record(stdout, sacct, nml, 104, 1, 1, "a", "b", "c", None, None)
            assert missing["max_disk_read_mb_per_task"] == expected
            assert missing["disk_counter_availability"]["read"] == (expected is not None)
        lines = [line.split('|') for line in original.splitlines()]
        columns = [j for j,k in enumerate(lines[0]) if k not in {"MaxDiskRead", "MaxDiskWrite"}]
        sacct.write_text('\n'.join('|'.join(row[j] for j in columns) for row in lines)+'\n')
        absent = collect_record(stdout, sacct, nml, 104, 1, 1, "a", "b", "c", None, None)
        assert absent["max_disk_read_mb_per_task"] is None and absent["max_disk_write_mb_per_task"] is None
        assert absent["eligible"] and absent["wall_seconds"] == collected["wall_seconds"]
    print(json.dumps({"status": "pass", "pareto_knee": result["pareto_knee"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
