from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RANKS = [1, 2, 4, 8, 13, 26, 52, 104, 156, 208, 312, 416]
PROGRESS_RE = re.compile(r"(?m)^\s*!\s*(\d+)\s+(\d{4}-\d{2}-\d{2}T\S+)")
FVCOM_INPUT_DIR_MAX_CHARS = 80


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_namelist_value(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?im)^(\s*{re.escape(key)}\s*=\s*).*?(\s*,?\s*(?:!.*)?)$")
    updated, count = pattern.subn(rf"\g<1>{value}\g<2>", text)
    if count != 1:
        raise ValueError(f"benchmark namelist must contain exactly one {key}; found {count}")
    return updated


def namelist_value(text: str, key: str) -> str:
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*(.*?)\s*,?\s*(?:!.*)?$", text)
    if not match:
        raise ValueError(f"benchmark namelist is missing {key}")
    value = match.group(1).rstrip(",").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def normalize_remote_input_dir(value: str) -> str:
    if not value.startswith("/") or not re.fullmatch(r"[A-Za-z0-9_./+-]+", value):
        raise ValueError(f"unsafe remote input directory: {value!r}")
    normalized = value.rstrip("/") + "/"
    if len(normalized) > FVCOM_INPUT_DIR_MAX_CHARS:
        raise ValueError(
            "remote input directory exceeds FVCOM CHARACTER(LEN=80) INPUT_DIR after "
            f"adding its required trailing slash: {len(normalized)} characters; use a "
            "short, project-owned alias bound to the same immutable input bundle"
        )
    return normalized


def render_jobs(plan_path: Path, base_namelist: Path, output_root: Path,
                remote_input_dir: str, remote_executable: str, case_name: str,
                account: str, partition: str, walltime: str,
                modules: list[str], attempt: int = 1) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    if plan.get("schema") != "fvcom_benchmark_plan_v1" or plan.get("execution") != "sequential":
        raise ValueError("benchmark plan is invalid or not sequential")
    if int(plan.get("segment_simulated_hours", 0)) != 24:
        raise ValueError("benchmark segment must be exactly 24 simulated hours")
    remote_input_dir = normalize_remote_input_dir(remote_input_dir)
    if not remote_executable.startswith("/") or not re.fullmatch(r"[A-Za-z0-9_./+-]+", remote_executable):
        raise ValueError(f"unsafe remote executable: {remote_executable!r}")
    if attempt < 1:
        raise ValueError("benchmark attempt must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", case_name):
        raise ValueError("unsafe case name")
    if not re.fullmatch(r"(?:(?:[0-9]+)-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}", walltime):
        raise ValueError("walltime must use Slurm [[days-]hours:]minutes:seconds syntax")
    for module in modules:
        if not re.fullmatch(r"[A-Za-z0-9._+/-]+", module):
            raise ValueError(f"unsafe module name: {module}")
    text = base_namelist.read_text(encoding="utf-8-sig")
    if namelist_value(text, "STARTUP_TYPE").lower() != "hotstart":
        raise ValueError("benchmark namelist must be a hotstart")
    start = dt.datetime.fromisoformat(namelist_value(text, "START_DATE").replace("Z", "+00:00"))
    end = dt.datetime.fromisoformat(namelist_value(text, "END_DATE").replace("Z", "+00:00"))
    if abs((end - start).total_seconds() - 86400.0) > 1.0:
        raise ValueError("benchmark namelist does not span exactly 24 hours")
    final_progress_time = end.strftime("%Y-%m-%dT%H:%M:%S")
    rendered_namelist = replace_namelist_value(text, "INPUT_DIR", f"'{remote_input_dir}'")
    rendered_namelist = replace_namelist_value(rendered_namelist, "OUTPUT_DIR", "'.'")
    manifest_path = output_root / "benchmark_jobs_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"immutable benchmark jobs manifest already exists: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for sequence, layout in enumerate(plan["layouts"], 1):
        ranks, nodes = int(layout["ranks"]), int(layout["nodes"])
        repeats = int(layout.get("repeat", 1))
        if ranks < 1 or nodes < 1 or repeats < 1:
            raise ValueError("benchmark ranks, nodes, and repeats must be positive")
        for repeat in range(1, repeats + 1):
            run_id = f"rank_{ranks:04d}_nodes_{nodes:02d}_rep_{repeat:02d}_attempt_{attempt:03d}"
            run_dir = output_root / run_id
            if run_dir.exists():
                raise FileExistsError(f"immutable benchmark run already exists: {run_dir}")
            run_dir.mkdir(parents=True)
            nml_path = run_dir / f"{case_name}_run.nml"
            nml_path.write_text(rendered_namelist, encoding="utf-8", newline="\n")
            module_lines = "module purge\n" + "\n".join(f"module load {name}" for name in modules)
            sbatch = f"""#!/bin/bash
#SBATCH --job-name=gb_b{sequence:03d}_r{ranks:04d}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ranks}
#SBATCH --exclusive
#SBATCH --time={walltime}
#SBATCH --output=stdout.log
#SBATCH --error=stderr.log
set -euo pipefail
{module_lines}
module -t list 2>&1
cd \"$SLURM_SUBMIT_DIR\"
date -u +%Y-%m-%dT%H:%M:%SZ > job_start_utc.txt
srun --cpu-bind=cores --ntasks={ranks} {remote_executable} --casename={case_name}
if ! grep -Fq 'TADA!' stdout.log; then
  echo 'BENCHMARK_GATE_FAILED: FVCOM did not reach TADA!' >&2
  exit 41
fi
if ! grep -Eq '^[[:space:]]*![[:space:]]+[0-9]+[[:space:]]+{final_progress_time}([.]0+)?([[:space:]]|$)' stdout.log; then
  echo 'BENCHMARK_GATE_FAILED: final requested progress timestamp is missing' >&2
  exit 42
fi
date -u +%Y-%m-%dT%H:%M:%SZ > job_end_utc.txt
"""
            sbatch_path = run_dir / "job.sbatch"
            sbatch_path.write_text(sbatch, encoding="utf-8", newline="\n")
            jobs.append({
                "sequence": sequence, "run_id": run_id, "ranks": ranks, "nodes": nodes, "repeat": repeat,
                "status": "prepared", "run_dir": str(run_dir), "namelist_sha256": sha256(nml_path),
                "sbatch_sha256": sha256(sbatch_path),
            })
    result = {
        "schema": "fvcom_benchmark_jobs_v1", "status": "prepared", "generated_at": now(),
        "execution": "sequential", "plan": str(plan_path), "plan_sha256": sha256(plan_path),
        "base_namelist": str(base_namelist), "base_namelist_sha256": sha256(base_namelist),
        "simulated_start": start.isoformat(), "simulated_end": end.isoformat(),
        "remote_input_dir": remote_input_dir,
        "remote_input_dir_characters": len(remote_input_dir),
        "fvcom_input_dir_max_characters": FVCOM_INPUT_DIR_MAX_CHARS,
        "requested_walltime": walltime,
        "post_run_gate": {
            "requires_tada": True,
            "requires_final_progress_timestamp": final_progress_time,
            "failure_exit_codes": {"missing_tada": 41, "missing_final_progress_timestamp": 42},
        },
        "attempt": attempt,
        "remote_executable": remote_executable,
        "case_name": case_name, "modules": modules, "jobs": jobs,
    }
    write_json(manifest_path, result)
    return result


def parse_datetime(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def parse_scaled_bytes(value: str | None) -> float:
    if not value or not value.strip():
        return 0.0
    match = re.fullmatch(r"\s*([0-9.]+)\s*([KMGTPE]?)\s*", value, re.I)
    if not match:
        raise ValueError(f"unrecognized Slurm size: {value!r}")
    factors = {"": 1.0, "K": 1024.0, "M": 1024.0**2, "G": 1024.0**3,
               "T": 1024.0**4, "P": 1024.0**5, "E": 1024.0**6}
    return float(match.group(1)) * factors[match.group(2).upper()]


def collect_record(stdout_path: Path, sacct_path: Path, namelist_path: Path,
                   ranks: int, nodes: int, repeat: int, executable_sha256: str,
                   restart_sha256: str, input_bundle_sha256: str,
                   io_seconds: float | None, io_method: str | None) -> dict[str, Any]:
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    nml = namelist_path.read_text(encoding="utf-8-sig")
    start = parse_datetime(namelist_value(nml, "START_DATE"))
    end = parse_datetime(namelist_value(nml, "END_DATE"))
    extstep = float(namelist_value(nml, "EXTSTEP_SECONDS"))
    isplit = int(namelist_value(nml, "ISPLIT"))
    internal_step = extstep * isplit
    simulated_seconds = (end - start).total_seconds()
    if abs(simulated_seconds - 86400.0) > 1.0:
        raise ValueError("benchmark evidence namelist does not span exactly 24 hours")
    if ranks < 1 or nodes < 1 or repeat < 1 or not math.isfinite(internal_step) or extstep <= 0 or isplit < 1:
        raise ValueError("ranks, nodes, repeat, EXTSTEP_SECONDS, and ISPLIT must be positive and finite")
    iterations = simulated_seconds / internal_step
    if not math.isclose(iterations, round(iterations), rel_tol=0, abs_tol=1e-7):
        raise ValueError("benchmark duration must be an exact internal-step multiple")

    rows = list(csv.DictReader(sacct_path.read_text(encoding="utf-8-sig").splitlines(), delimiter="|"))
    if not rows:
        raise ValueError("sacct evidence contains no rows")
    rows = [{str(k).strip(): str(v or "").strip() for k, v in row.items() if k is not None} for row in rows]
    primary = next((row for row in rows if row.get("JobID") and "." not in row["JobID"]), rows[0])
    elapsed_raw = primary.get("ElapsedRaw", "")
    if not elapsed_raw:
        raise ValueError("sacct evidence is missing ElapsedRaw")
    submit = parse_datetime(primary["Submit"]) if primary.get("Submit") else None
    job_start = parse_datetime(primary["Start"]) if primary.get("Start") else None
    queue_seconds = max(0.0, (job_start - submit).total_seconds()) if submit and job_start else None
    exit_code = int((primary.get("ExitCode") or "1:0").split(":", 1)[0])
    state = primary.get("State", "UNKNOWN").split("+", 1)[0]
    peak_memory_bytes = max((parse_scaled_bytes(row.get("MaxRSS")) for row in rows), default=0.0)
    max_disk_read_bytes = max((parse_scaled_bytes(row.get("MaxDiskRead")) for row in rows), default=0.0)
    max_disk_write_bytes = max((parse_scaled_bytes(row.get("MaxDiskWrite")) for row in rows), default=0.0)

    progress = PROGRESS_RE.findall(stdout)
    last_iint = int(progress[-1][0]) if progress else None
    expected_end = end.strftime("%Y-%m-%dT%H:%M:%S")
    final_present = bool(re.search(rf"(?m)^\s*!\s*\d+\s+{re.escape(expected_end)}(?:\.0+)?\b", stdout))
    fatal = bool(re.search(r"(?i)\b(blowup|fatal error|segmentation fault|forrtl:)\b", stdout))
    record = {
        "schema": "fvcom_benchmark_record_v1", "collected_at": now(),
        "job_id": primary.get("JobID"), "slurm_state": state, "exit_code": exit_code,
        "ranks": ranks, "nodes": nodes, "repeat": repeat,
        "queue_seconds": queue_seconds, "wall_seconds": float(elapsed_raw),
        "simulated_seconds": simulated_seconds,
        "fvcom_iterations": int(round(iterations)),
        "iteration_definition": "internal FVCOM IINT steps",
        "extstep_seconds": extstep, "isplit": isplit, "internal_step_seconds": internal_step,
        "last_logged_iint": last_iint,
        "peak_memory_mb": peak_memory_bytes / 1024.0**2,
        "max_disk_read_mb_per_task": max_disk_read_bytes / 1024.0**2,
        "max_disk_write_mb_per_task": max_disk_write_bytes / 1024.0**2,
        "io_seconds": io_seconds, "io_measurement_method": io_method or "unavailable",
        "tada": "TADA!" in stdout, "final_timestamp_present": final_present,
        "fatal_marker_present": fatal,
        "executable_sha256": executable_sha256, "restart_sha256": restart_sha256,
        "input_bundle_sha256": input_bundle_sha256,
        "simulated_start": start.isoformat().replace("+00:00", "Z"),
        "simulated_end": end.isoformat().replace("+00:00", "Z"),
        "stdout_sha256": sha256(stdout_path), "sacct_sha256": sha256(sacct_path),
        "namelist_sha256": sha256(namelist_path),
    }
    record["eligible"] = bool(
        state == "COMPLETED" and exit_code == 0 and record["tada"]
        and final_present and not fatal
    )
    return record


def make_plan(cores_per_node: int = 104, ranks: list[int] | None = None,
              repeats: int = 3, extend: bool = True,
              max_ranks: int | None = None) -> dict[str, Any]:
    selected = DEFAULT_RANKS if ranks is None else ranks
    if cores_per_node < 1 or repeats < 1 or not selected or any(r < 1 for r in selected):
        raise ValueError("cores, repeats, and ranks must be positive")
    if len(set(selected)) != len(selected):
        raise ValueError("benchmark ranks must be unique")
    if max_ranks is not None and (max_ranks < 1 or max(selected) > max_ranks):
        raise ValueError("benchmark rank exceeds max_ranks")
    return {
        "schema": "fvcom_benchmark_plan_v1", "generated_at": now(),
        "segment_simulated_hours": 24, "execution": "sequential",
        "cores_per_node": cores_per_node,
        "layouts": [{"ranks": r, "nodes": math.ceil(r / cores_per_node), "repeat": 1} for r in sorted(selected)],
        "extension": {"enabled": extend, "max_ranks": max_ranks,
                      "rank_increment": cores_per_node, "stop_after_consecutive_slower": 2},
        "pareto_neighborhood_repeats": repeats,
    }


def resume_plan(plan: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return only benchmark replicates that lack an eligible terminal record."""
    completed = {
        (int(row["ranks"]), int(row["nodes"]), int(row.get("repeat", 1)))
        for row in records
        if bool(row.get("tada")) and bool(row.get("final_timestamp_present"))
        and int(row.get("exit_code", 0)) == 0
    }
    pending = []
    for layout in plan["layouts"]:
        for repeat in range(1, int(layout.get("repeat", 1)) + 1):
            key = (int(layout["ranks"]), int(layout["nodes"]), repeat)
            if key not in completed:
                prior = sum(
                    1 for row in records
                    if int(row["ranks"]) == key[0] and int(row["nodes"]) == key[1]
                    and int(row.get("repeat", 1)) == repeat
                )
                pending.append({
                    "ranks": key[0], "nodes": key[1], "repeat": repeat,
                    "prior_attempt_count": prior, "next_attempt_number": prior + 1,
                })
    return {
        "schema": "fvcom_benchmark_resume_v1", "generated_at": now(),
        "plan_complete": not pending, "completed_target_count": len(completed),
        "pending": pending,
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value["records"] if isinstance(value, dict) else value
    records = []
    for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL line {i}: {exc}") from exc
    return records


def enrich(record: dict[str, Any]) -> dict[str, Any]:
    r = dict(record)
    ranks, nodes = int(r["ranks"]), int(r["nodes"])
    wall = float(r.get("wall_seconds", 0))
    sim = float(r.get("simulated_seconds", 86400.0))
    iterations = int(r.get("fvcom_iterations", 0))
    raw_io = r.get("io_seconds")
    io_s = float(raw_io) if raw_io is not None else None
    valid_numbers = ranks > 0 and nodes > 0 and iterations > 0 and all(math.isfinite(x) and x > 0 for x in (wall, sim))
    eligible = (valid_numbers and r.get("eligible") is not False
                and not r.get("fatal_marker_present", False)
                and r.get("slurm_state", "COMPLETED") == "COMPLETED"
                and bool(r.get("tada")) and bool(r.get("final_timestamp_present"))
                and int(r.get("exit_code", 0)) == 0)
    r.update({
        "ranks": ranks, "nodes": nodes,
        "seconds_per_iteration": wall / iterations if valid_numbers else None,
        "simulated_days_per_day": sim / wall if valid_numbers else None,
        "node_hours": nodes * wall / 3600.0 if valid_numbers else None,
        "io_fraction": io_s / wall if io_s is not None and wall else None,
        "eligible": eligible,
    })
    return r


def analyze(records: list[dict[str, Any]], plan: dict[str, Any] | None = None) -> dict[str, Any]:
    if not records:
        raise ValueError("no benchmark records")
    enriched = [enrich(r) for r in records]
    eligible = [r for r in enriched if r["eligible"]]
    if plan is not None:
        if plan.get("schema") != "fvcom_benchmark_plan_v1":
            raise ValueError("invalid benchmark plan")
        allowed = {(int(x["ranks"]), int(x["nodes"])) for x in plan["layouts"]}
        if any((x["ranks"], x["nodes"]) not in allowed for x in eligible):
            raise ValueError("eligible benchmark layout is absent from the supplied plan; record an immutable plan revision")
    if not eligible:
        raise ValueError("no eligible TADA/final-time records")
    lineage_keys = ["executable_sha256", "restart_sha256", "input_bundle_sha256", "simulated_start", "simulated_end"]
    mismatches = {k: sorted({str(r.get(k)) for r in eligible}) for k in lineage_keys if len({str(r.get(k)) for r in eligible}) != 1}
    if mismatches:
        raise ValueError(f"benchmark science lineage mismatch: {mismatches}")
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in eligible:
        groups[(r["ranks"], r["nodes"])].append(r)
    layouts = []
    for (ranks, nodes), rows in sorted(groups.items()):
        item = {
            "ranks": ranks, "nodes": nodes, "repeat_count": len(rows),
            "wall_seconds_median": statistics.median(x["wall_seconds"] for x in rows),
            "simulated_days_per_day_median": statistics.median(x["simulated_days_per_day"] for x in rows),
            "seconds_per_iteration_median": statistics.median(x["seconds_per_iteration"] for x in rows),
            "node_hours_median": statistics.median(x["node_hours"] for x in rows),
            "queue_seconds_median": statistics.median(float(x.get("queue_seconds", 0.0)) for x in rows),
            "peak_memory_mb_max": max(float(x.get("peak_memory_mb", 0.0)) for x in rows),
            "io_fraction_median": (
                statistics.median(float(x["io_fraction"]) for x in rows if x.get("io_fraction") is not None)
                if any(x.get("io_fraction") is not None for x in rows) else None
            ),
        }
        layouts.append(item)
    reference = layouts[0]
    baseline = reference["wall_seconds_median"]
    for x in layouts:
        x["speedup"] = baseline / x["wall_seconds_median"]
        x["speedup_reference_ranks"] = reference["ranks"]
        x["parallel_efficiency"] = x["speedup"] / (x["ranks"] / reference["ranks"])
    fastest = max(layouts, key=lambda x: (x["simulated_days_per_day_median"], -x["node_hours_median"]))
    cheapest = min(layouts, key=lambda x: (x["node_hours_median"], -x["simulated_days_per_day_median"]))
    pareto = []
    for x in layouts:
        dominated = any(
            y["simulated_days_per_day_median"] >= x["simulated_days_per_day_median"] and
            y["node_hours_median"] <= x["node_hours_median"] and
            (y["simulated_days_per_day_median"] > x["simulated_days_per_day_median"] or y["node_hours_median"] < x["node_hours_median"])
            for y in layouts
        )
        if not dominated:
            pareto.append(x)
    throughputs = [x["simulated_days_per_day_median"] for x in pareto]
    costs = [x["node_hours_median"] for x in pareto]
    tmin, tmax, cmin, cmax = min(throughputs), max(throughputs), min(costs), max(costs)
    def knee_distance(x: dict[str, Any]) -> float:
        nt = (x["simulated_days_per_day_median"] - tmin) / (tmax - tmin) if tmax > tmin else 1.0
        nc = (x["node_hours_median"] - cmin) / (cmax - cmin) if cmax > cmin else 0.0
        return math.hypot(1.0 - nt, nc)
    knee = min(pareto, key=knee_distance)
    ordered = sorted(layouts, key=lambda x: x["ranks"])
    knee_index = ordered.index(knee)
    neighborhood = {knee["ranks"]}
    if knee_index > 0:
        neighborhood.add(ordered[knee_index - 1]["ranks"])
    if knee_index + 1 < len(ordered):
        neighborhood.add(ordered[knee_index + 1]["ranks"])
    neighborhood.update(x["ranks"] for x in pareto)
    repeat_plan = []
    target_repeats = int((plan or {}).get("pareto_neighborhood_repeats", 3))
    if target_repeats < 1:
        raise ValueError("pareto_neighborhood_repeats must be positive")
    for item in ordered:
        if item["ranks"] in neighborhood:
            repeat_plan.append({
                "ranks": item["ranks"], "nodes": item["nodes"], "target_repeats": target_repeats,
                "eligible_repeats": item["repeat_count"],
                "additional_repeats": max(0, target_repeats - item["repeat_count"]),
            })
    slower = 0
    for prev, cur in zip(ordered[:-1], ordered[1:]):
        slower = slower + 1 if cur["simulated_days_per_day_median"] <= prev["simulated_days_per_day_median"] else 0
    policy = (plan or {}).get("extension", {})
    increment = int(policy.get("rank_increment", 104))
    next_ranks = ordered[-1]["ranks"] + increment
    extend = (len(ordered) > 1 and policy.get("enabled", True)
              and ordered[-1]["ranks"] >= 416 and slower < 2
              and ordered[-1]["simulated_days_per_day_median"] > ordered[-2]["simulated_days_per_day_median"]
              and (policy.get("max_ranks") is None or next_ranks <= int(policy["max_ranks"])))
    return {
        "schema": "fvcom_benchmark_summary_v1", "generated_at": now(),
        "eligible_run_count": len(eligible), "failed_run_count": len(enriched) - len(eligible),
        "layouts": layouts, "fastest": {"ranks": fastest["ranks"], "nodes": fastest["nodes"]},
        "speedup_reference": {"ranks": reference["ranks"], "nodes": reference["nodes"],
                              "wall_seconds": baseline, "measured": True,
                              "interpretation": "serial" if reference["ranks"] == 1 else "relative_to_lowest_measured_rank"},
        "least_node_hour": {"ranks": cheapest["ranks"], "nodes": cheapest["nodes"]},
        "pareto": [{"ranks": x["ranks"], "nodes": x["nodes"]} for x in pareto],
        "pareto_knee": {"ranks": knee["ranks"], "nodes": knee["nodes"]},
        "pareto_neighborhood_repeat_plan": repeat_plan,
        "extension_required": extend,
        "next_extension_ranks": next_ranks if extend else None,
        "lineage": {k: eligible[0].get(k) for k in lineage_keys},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        if rows:
            w.writeheader(); w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="FVCOM benchmark planner and analyzer")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("plan"); q.add_argument("--output", required=True); q.add_argument("--cores-per-node", type=int, default=104)
    q.add_argument("--ranks", nargs="+", type=int); q.add_argument("--repeats", type=int, default=3)
    q.add_argument("--no-extend", action="store_true"); q.add_argument("--max-ranks", type=int)
    q = sub.add_parser("analyze"); q.add_argument("--records", required=True); q.add_argument("--output", required=True); q.add_argument("--csv")
    q.add_argument("--plan", help="Apply the recorded repeat and extension policy")
    q = sub.add_parser("resume"); q.add_argument("--plan", required=True); q.add_argument("--records", required=True); q.add_argument("--output", required=True)
    q = sub.add_parser("collect")
    q.add_argument("--stdout", required=True); q.add_argument("--sacct", required=True)
    q.add_argument("--namelist", required=True); q.add_argument("--ranks", type=int, required=True)
    q.add_argument("--nodes", type=int, required=True); q.add_argument("--repeat", type=int, default=1)
    q.add_argument("--executable-sha256", required=True); q.add_argument("--restart-sha256", required=True)
    q.add_argument("--input-bundle-sha256", required=True); q.add_argument("--io-seconds", type=float)
    q.add_argument("--io-method"); q.add_argument("--output", required=True)
    q = sub.add_parser("render")
    q.add_argument("--plan", required=True); q.add_argument("--base-namelist", required=True)
    q.add_argument("--output-root", required=True); q.add_argument("--remote-input-dir", required=True)
    q.add_argument("--remote-executable", required=True); q.add_argument("--case-name", default="galveston")
    q.add_argument("--account", default="hindcastra"); q.add_argument("--partition", default="standard")
    q.add_argument("--walltime", default="04:00:00"); q.add_argument("--module", action="append", default=[])
    q.add_argument("--attempt", type=int, default=1)
    args = p.parse_args()
    if args.command == "plan":
        result = make_plan(args.cores_per_node, args.ranks, args.repeats, not args.no_extend, args.max_ranks)
    elif args.command == "analyze":
        result = analyze(load_records(Path(args.records)),
                         json.loads(Path(args.plan).read_text(encoding="utf-8-sig")) if args.plan else None)
        if args.csv:
            write_csv(Path(args.csv), result["layouts"])
    elif args.command == "resume":
        result = resume_plan(json.loads(Path(args.plan).read_text(encoding="utf-8-sig")), load_records(Path(args.records)))
    elif args.command == "collect":
        result = collect_record(Path(args.stdout), Path(args.sacct), Path(args.namelist),
                                args.ranks, args.nodes, args.repeat, args.executable_sha256,
                                args.restart_sha256, args.input_bundle_sha256,
                                args.io_seconds, args.io_method)
    else:
        result = render_jobs(Path(args.plan), Path(args.base_namelist), Path(args.output_root),
                             args.remote_input_dir, args.remote_executable, args.case_name,
                             args.account, args.partition, args.walltime, args.module, args.attempt)
    if args.command != "render":
        write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
