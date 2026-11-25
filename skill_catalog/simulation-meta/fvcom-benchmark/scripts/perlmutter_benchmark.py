"""Compare matched, dated FVCOM restart segments on Perlmutter CPU nodes."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

from fvcom_benchmark import PROGRESS_RE, namelist_value, parse_datetime, parse_scaled_bytes, sha256


RANKS = (64, 128, 256)
LINEAGE = ("executable_sha256", "build_binding_sha256", "restart_sha256",
           "input_bundle_sha256", "namelist_sha256", "plan_sha256")


def utc(value):
    return parse_datetime(value).isoformat().replace("+00:00", "Z")


def make_cpu_plan(start):
    if not start:
        raise ValueError("specify the common restart clock with --start-utc")
    first = parse_datetime(start)
    return {
        "schema": "fvcom_benchmark_plan_v1", "execution_site": "perlmutter",
        "segment_simulated_hours": 24, "cores_per_node": 128,
        "simulated_start": utc(start),
        "simulated_end": (first + dt.timedelta(days=1)).isoformat().replace("+00:00", "Z"),
        "layouts": [{"ranks": r, "nodes": math.ceil(r / 128), "repeat": 3} for r in RANKS],
        "runs": [{"stage": f"benchmark_{r:03d}_r{k}", "ranks": r,
                  "nodes": math.ceil(r / 128), "repeat": k} for k in range(1, 4) for r in RANKS],
        "selection": "Shortest median FVCOM srun elapsed among layouts within 15 percent of minimum median allocation node-hours",
        "allocation_cost_tolerance": 0.15, "renderer": "fvcom-run-control lake-attempt",
    }


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value) is not None


def inputs_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def check_plan(plan):
    require(plan == make_cpu_plan(plan["simulated_start"]), "inconsistent Perlmutter benchmark plan")


def collect_cpu(plan_path, attempt_dir, submission_path, launch_path, sacct_path, stdout_path, restart_path, terminal_path):
    plan = read(plan_path)
    check_plan(plan)
    attempt_path = attempt_dir / "attempt_manifest.json"
    attempt = read(attempt_path)
    spec = attempt["stage"]
    target = next((r for r in plan["runs"] if r["stage"] == spec["name"]), None)
    require(target is not None, "attempt is outside the frozen benchmark plan")
    ranks, nodes = target["ranks"], target["nodes"]
    require(attempt["site"] == "perlmutter" and attempt["mpi_ranks"] == spec["ranks"] == ranks
            and attempt["nodes"] == nodes and attempt["name"] == attempt_dir.name,
            "attempt layout or directory differs from the frozen slot")
    require(re.fullmatch("attempt_" + target["stage"] + r"_\d{4}", attempt["name"]), "invalid frozen attempt name")
    sub, launch, terminal = map(read, (submission_path, launch_path, terminal_path))
    job = str(sub["job_id"])
    require(job.isdecimal() and sub["account"] == attempt["account"] and sub["workdir"] == attempt["remote_run"]
            and sub["script"] == sub["workdir"] + "/job.sbatch"
            and sub["script_sha256"] == attempt["job_sha256"] == sha256(attempt_dir / "job.sbatch"),
            "submission does not bind this frozen job")
    require(launch.get("site") == "perlmutter" and str(launch["job_id"]) == job
            and launch["mpi_ranks"] == ranks and launch["constraint"] == "cpu" and launch["threads_per_rank"] == 1
            and launch["executable_sha256"] == attempt["executable_sha256"], "native launch identity mismatch")
    require(terminal.get("status") == "verified" and str(terminal["job_id"]) == job
            and utc(terminal["utc"]) == plan["simulated_end"] and valid_hash(terminal["sha256"]),
            "a verified terminal restart from this job at the segment end is required")
    namelists = list(attempt_dir.glob("*_run.nml"))
    require(len(namelists) == 1 and sha256(namelists[0]) == attempt["run_namelist_sha256"],
            "frozen namelist mismatch")
    nml = namelists[0].read_text(encoding="utf-8-sig")
    for key, field, clock in (("START_DATE", "start_utc", "simulated_start"),
                              ("END_DATE", "end_utc", "simulated_end")):
        require(utc(namelist_value(nml, key)) == utc(spec[field]) == plan[clock], "segment clock mismatch")
    require(namelist_value(nml, "STARTUP_TYPE").lower() == "hotstart", "benchmark requires a dated restart")
    files = attempt["files"]
    require(bool(files) and all(valid_hash(h) for h in files.values()), "invalid input digest")
    initial = read(restart_path)
    require(initial.get("status") == "verified" and utc(initial["utc"]) == plan["simulated_start"]
            and initial["sha256"] == files[namelist_value(nml, "STARTUP_FILE")],
            "selected restart receipt differs from the frozen input")
    checks = [(h, "input/" + k) for k, h in sorted(files.items())]
    checks += [(attempt["run_namelist_sha256"], namelists[0].name),
               (attempt["executable_sha256"], attempt["executable"])]
    checks_path = attempt_dir / "runtime_checksums.sha256"
    require(checks_path.read_text().splitlines() == [h + "  " + name for h, name in checks],
            "the verified pre-launch inventory differs from the attempt")
    with sacct_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="|"))
    def row_for(identifier):
        found = [r for r in rows if r.get("JobID", r.get("JobIDRaw")) == identifier]
        require(len(found) == 1, "expected a unique sacct row for " + identifier)
        row = found[0]
        require(row["State"] == "COMPLETED" and row["ExitCode"] == "0:0",
                "job and model step must both finish without exit code or signal")
        require(int(row["NNodes"]) == nodes, "measured node count differs from frozen layout")
        return row
    allocation, step = row_for(job), row_for(job + ".0")
    require(int(step["NTasks"]) == ranks, "Slurm model-step rank count differs from frozen layout")
    require(not any(re.fullmatch(job + r"\.[1-9]\d*", r.get("JobID", r.get("JobIDRaw", ""))) for r in rows),
            "more than one application step in benchmark allocation")
    wall, model = int(allocation["ElapsedRaw"]), int(step["ElapsedRaw"])
    require(0 < model <= wall, "invalid job and model-step elapsed accounting")
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    require("TADA!" in stdout and not re.search(r"(?i)\b(blowup|fatal error|segmentation fault|forrtl:)\b", stdout),
            "FVCOM log lacks a clean native completion")
    progress = PROGRESS_RE.findall(stdout)
    require(any(parse_datetime(plan["simulated_start"]) < parse_datetime(t) <= parse_datetime(plan["simulated_end"])
                for _, t in progress), "FVCOM log contains no in-window integration progress")
    extstep, isplit = float(namelist_value(nml, "EXTSTEP_SECONDS")), int(namelist_value(nml, "ISPLIT"))
    require(math.isfinite(extstep) and extstep > 0 and isplit > 0, "invalid time stepping")
    iterations = 86400 / (extstep * isplit)
    require(math.isclose(iterations, round(iterations), rel_tol=0, abs_tol=1e-7), "fractional segment step count")
    def optional_bytes(k):
        return parse_scaled_bytes(step[k]) if step.get(k) else None
    record = dict(schema="fvcom_perlmutter_benchmark_record_v1", execution_site="perlmutter", eligible=True,
                  **target, job_id=job, attempt=attempt["name"], simulated_start=plan["simulated_start"],
                  simulated_end=plan["simulated_end"], measured_nodes=nodes, measured_ranks=ranks,
                  model_step_seconds=model, allocation_seconds=wall, extstep_seconds=extstep, isplit=isplit,
                  fvcom_iterations=round(iterations), last_logged_iint=int(progress[-1][0]),
                  peak_memory_bytes_per_task=optional_bytes("MaxRSS"),
                  max_disk_read_bytes_per_task=optional_bytes("MaxDiskRead"),
                  max_disk_write_bytes_per_task=optional_bytes("MaxDiskWrite"),
                  restart_sha256=files[namelist_value(nml, "STARTUP_FILE")],
                  executable_sha256=attempt["executable_sha256"], build_binding_sha256=attempt["build_binding_sha256"],
                  input_bundle_sha256=inputs_digest(files), namelist_sha256=attempt["run_namelist_sha256"],
                  plan_sha256=sha256(plan_path), plan_content_sha256=inputs_digest(plan),
                  terminal_restart_sha256=terminal["sha256"],
                  evidence_sha256={k:sha256(p) for k, p in
                                   (("attempt",attempt_path), ("launch",launch_path), ("submission",submission_path),
                                    ("sacct",sacct_path), ("stdout",stdout_path), ("terminal",terminal_path),
                                    ("restart",restart_path), ("runtime_checksums",checks_path))})
    require(all(valid_hash(record[k]) for k in LINEAGE), "invalid benchmark lineage")
    return record


def audited_slots(plan, records, plan_file_sha256=None):
    check_plan(plan)
    slots = {r["stage"]: r for r in plan["runs"]}
    complete, jobs = {}, set()
    for row in records:
        stage, job = row["stage"], row["job_id"]
        require(stage in slots and all(slots[stage][k] == row.get(k) for k in slots[stage]),
                "unplanned or mismatched benchmark slot")
        require(stage not in complete and job not in jobs, "duplicate benchmark replicate or Slurm job")
        require(row.get("schema") == "fvcom_perlmutter_benchmark_record_v1" and row.get("eligible") is True
                and row.get("execution_site") == "perlmutter", "unverified benchmark record")
        require(str(job).isdecimal() and all(valid_hash(row.get(k)) for k in (*LINEAGE, "terminal_restart_sha256"))
                and set(row["evidence_sha256"]) == {"attempt","launch","submission","sacct","stdout","restart","terminal","runtime_checksums"}
                and all(valid_hash(h) for h in row["evidence_sha256"].values()), "missing native lineage")
        require(row["simulated_start"] == plan["simulated_start"] and row["simulated_end"] == plan["simulated_end"]
                and row.get("plan_content_sha256") == inputs_digest(plan)
                and (plan_file_sha256 is None or row["plan_sha256"] == plan_file_sha256),
                "record does not bind the dated plan")
        require(row["measured_nodes"] == row["nodes"] and row["measured_ranks"] == row["ranks"]
                and 0 < row["model_step_seconds"] <= row["allocation_seconds"] and row["fvcom_iterations"] > 0
                and row["extstep_seconds"] > 0 and row["isplit"] > 0
                and math.isclose(row["fvcom_iterations"] * row["extstep_seconds"] * row["isplit"], 86400),
                "invalid native accounting or integration clock")
        complete[stage] = row
        jobs.add(job)
    for k in (*LINEAGE, "extstep_seconds", "isplit", "fvcom_iterations"):
        require(len({row[k] for row in records}) <= 1, "benchmark science lineage differs: " + k)
    return complete


def resume_cpu(plan, records, plan_file_sha256=None):
    complete = audited_slots(plan, records, plan_file_sha256)
    pending = [r for r in plan["runs"] if r["stage"] not in complete]
    return dict(schema="fvcom_benchmark_resume_v1", plan_complete=not pending,
                completed_target_count=len(complete), pending=pending)


def analyze_cpu(plan, records, plan_file_sha256=None):
    complete = audited_slots(plan, records, plan_file_sha256)
    require(len(complete) == 9, "nine verified unique benchmark replicates are required before selection")
    layouts = []
    for ranks in RANKS:
        rows = [r for r in records if r["ranks"] == ranks]
        nodes = rows[0]["nodes"]
        elapsed = median(r["model_step_seconds"] for r in rows)
        wall = median(r["allocation_seconds"] for r in rows)
        layouts.append(dict(ranks=ranks, nodes=nodes, repeat_count=len(rows),
                            model_step_seconds_median=elapsed, allocation_seconds_median=wall,
                            model_step_seconds_min=min(r["model_step_seconds"] for r in rows),
                            model_step_seconds_max=max(r["model_step_seconds"] for r in rows),
                            charged_node_hours_median=wall*nodes/3600, simulated_days_per_day_median=86400/elapsed,
                            seconds_per_internal_iteration_median=elapsed/rows[0]["fvcom_iterations"]))
    least = min(x["charged_node_hours_median"] for x in layouts)
    eligible = [x for x in layouts if x["charged_node_hours_median"] <= least * 1.15 + 1e-12]
    winner = min(eligible, key=lambda x:(x["model_step_seconds_median"],x["charged_node_hours_median"],x["ranks"]))
    return dict(schema="fvcom_benchmark_summary_v1", execution_site="perlmutter", status="verified",
                eligible_run_count=9, layouts=layouts, selection=plan["selection"],
                production_layout={"ranks":winner["ranks"],"nodes":winner["nodes"]},
                eligible_cost_ceiling_node_hours=least*1.15,
                records_sha256=inputs_digest(sorted(records,key=lambda r:r["stage"])),
                job_ids={stage:row["job_id"] for stage,row in complete.items()},
                lineage={k:records[0][k] for k in LINEAGE})
