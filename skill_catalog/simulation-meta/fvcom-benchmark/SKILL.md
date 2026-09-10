---
name: fvcom-benchmark
description: Plan and analyze reproducible FVCOM Kestrel rank/node benchmarks from identical restart segments. Use to compare throughput, memory, node-hours, speedup, efficiency, and I/O; report fastest, least-cost, and Pareto-knee layouts without submitting jobs itself.
---

# FVCOM Benchmark

Preserve `MaxDiskRead` and `MaxDiskWrite` in terminal Slurm accounting. Missing
columns or blank counters are unavailable (`null`), not measured zero. Report
the maximum measured per-task byte counter separately from I/O duration; byte
counters do not measure I/O time. A recorded station-only revision after certified
spin-up may change the mapping and bound station file only; verify the original
bundle first and use one identical revised bundle for all layouts and production.

Use only after a stable, hash-bound restart exists. Invoke `$kestrel-hpc` for all job operations and `$fvcom-run-control` to audit completion. Every candidate must use the same executable, restart, input hashes, simulated 24-hour window, and output cadence.

Create the initial plan:

```powershell
python scripts/fvcom_benchmark.py plan --output benchmark_plan.json
```

The default matrix is `1,2,4,8,13,26,52,104,156,208,312,416` ranks on current 104-core Kestrel CPU nodes. Execute sequentially to reduce cross-job interference. If median throughput improves at 416, add one 104-rank node at a time until two consecutive layouts are slower. Repeat the Pareto-neighborhood layouts three times.

Honor an explicitly bounded campaign with `plan --ranks 52 104 156 208 --repeats 1 --no-extend --max-ranks 416`. Here `--repeats` controls the Pareto-neighborhood target; the initial matrix always contains one replicate. Pass `--plan benchmark_plan.json` to `analyze` to preserve those bounds. Independent high-rank probes are compatibility evidence, not timing records; a failed probe must not prevent other probes. Add successfully probed ranks to a new immutable plan revision before their full benchmark.

Query the actual MPI tag bound on a compute node before interpreting a messaging failure. The frozen FVCOM `genmap.F` may construct rank-dependent tags outside that bound; record the source expression, actual failed tag, module environment, and whether integration began. Do not classify it as a mesh or hydrodynamic failure or change source/environment inside an existing benchmark lineage.

After a stable restart and exact 24-hour hotstart namelist exist, render immutable job directories. Rendering reserves whole nodes, holds the science namelist and output cadence constant, and changes only ranks/nodes:

```powershell
python scripts/fvcom_benchmark.py render --plan benchmark_plan.json --base-namelist benchmark_run.nml --output-root run/case/benchmark --remote-input-dir /scratch/user/case/input --remote-executable /path/to/fvcom --attempt 1 --module intel/2023.2.0 --module intel-oneapi-mpi/2021.11.0-intel
```

FVCOM 4.3.1 declares `INPUT_DIR` as `CHARACTER(LEN=80)` and concatenates it directly with file names. Rendering therefore normalizes the path to a trailing slash and fails closed when the normalized path exceeds 80 characters. When the immutable input bundle has a longer canonical path, create and verify a short project-owned alias to that exact bundle and record both paths and hashes.

Submit the rendered jobs sequentially through `$kestrel-hpc`. Each rendered job exits nonzero unless the runtime log contains both `TADA!` and a progress line at the requested final timestamp, so a strict Slurm `afterok` chain is an execution gate rather than merely a process-exit dependency. Audit the scientific outputs independently after retrieval.

Treat the one-rank layout as a walltime canary. Once its seconds-per-iteration stabilizes, project its full-segment runtime and require the requested Slurm walltime to exceed that projection with a practical margin. If it cannot finish within the active limit, cancel it early, preserve the partial evidence, increment `--attempt`, and rerender with a sufficient allowed walltime. Never overwrite a rendered or failed benchmark run.

Collect one immutable record from the exact namelist, FVCOM log, and a headered
`sacct -P` table containing at least `JobID,State,ExitCode,ElapsedRaw,Submit,Start,End,MaxRSS,MaxDiskRead,MaxDiskWrite`:

```powershell
python scripts/fvcom_benchmark.py collect --stdout stdout.log --sacct sacct.txt --namelist galveston_run.nml --ranks 104 --nodes 1 --repeat 1 --executable-sha256 SHA --restart-sha256 SHA --input-bundle-sha256 SHA --output benchmark_record.json
```

Supply `--io-seconds` and `--io-method` only when a compatible profiler or a
controlled paired-output experiment measured I/O time. Never report Slurm disk
bytes or an iteration-time residual as measured I/O time. If the frozen binary
cannot be instrumented compatibly, retain a null I/O fraction, report the disk
byte counters, and state the instrumentation limitation.

Iteration counts refer to the internal `IINT` loop: divide the segment duration by `EXTSTEP_SECONDS * ISPLIT`. Retain both controls and the last logged `IINT`; dividing by the external step alone misreports seconds per iteration.

After collecting one JSON record per run, analyze them:

```powershell
python scripts/fvcom_benchmark.py analyze --records benchmark_records.jsonl --output benchmark_summary.json --csv benchmark_summary.csv
```

Each successful record includes ranks, nodes, queue seconds, wall seconds, simulated seconds, FVCOM iterations, peak memory, I/O seconds, `TADA!`, final-time status, executable/restart/input hashes, and repeat index. Reject non-identical science hashes. Compute seconds/iteration, simulated-days/day, node-hours, speedup, parallel efficiency, and I/O fraction. Report the fastest median layout, least-node-hour median layout, nondominated layouts, and a normalized-distance Pareto knee. A queue-time observation is reported but excluded from model throughput.

Speedup is relative to the lowest successfully measured rank. With no serial run, disclose that rank and report relative efficiency as speedup divided by the rank-count ratio; never fabricate a one-rank runtime. Single-replicate reports carry no sampling uncertainty estimate.

Do not tune physics or output cadence for a benchmark. A failed run remains evidence but cannot enter performance selection.

Resume without rerunning eligible terminal replicates or overwriting failed evidence:

```powershell
python scripts/fvcom_benchmark.py resume --plan benchmark_plan.json --records benchmark_records.jsonl --output benchmark_resume.json
```

## Validation

```powershell
python scripts/selftest_fvcom_benchmark.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
