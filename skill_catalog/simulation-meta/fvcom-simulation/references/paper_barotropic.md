# paper_barotropic

Use `mode: barotropic_tide` and `profile: paper_barotropic`. This profile replaces
the ordinary run sequence and exploratory retry/benchmark policy. Keep the
common scientific preparation gates in the parent skill. The profile resolver
in Run Control fills the following settings and rejects conflicting overrides.

## Fixed setup

- Generate a fresh grid in empty local and remote trial roots (`fresh_only`).
- Analysis: **2025-04-01 00:00 through 2025-04-15 00:00 UTC, end exclusive**.
- Production: cold start **March 29 00:00**, end April 15 00:00; the first three
  days are physical spin-up within this run. Do not initialize from the smoke restart.
- Three-dimensional barotropic, uniform 20 C and 30 PSU, ten uniform sigma
  levels (nine layers), tide-only, all available TPXO constituents. Preserve
  exact OBC order, six-minute UTC forcing/stations and three-hour full-grid output.
- Use the existing conservative CFL, time-alignment, sponge and two-day ramp
  defaults. A one-day smoke does not certify the full ramp or long-run stability.
- Compare elevation with available NOAA astronomical predictions, retain total
  observations as diagnostics, and compare depth-mean currents only with eligible
  downward-looking all-bin profiles. Disclose unavailable observations and the
  harmonic-resolution limits of a 14-day record; do not claim month-long resolution.

## Workflow

1. **Prepare.** Initialize through Run Control. Use the existing grid, TPXO and
   configuration/build roles and their typed handoffs. Complete the common join,
   source-bound TGE, preconfiguration, forcing, station-mapping and timing gates.
   Keep all project artifacts and caches in this trial. Use a project-owned source
   copy. Prefer a native RH9 build; inspect the actual host/compiler/modules with
   Kestrel HPC and record `kestrel.build_environment`. RH8 requires a documented
   availability or compatibility reason. Never transplant the legacy RH8 module
   pins onto RH9. Verify the selected environment again on the allocated nodes.
2. **Smoke.** Run March 29–30 (24 simulated hours) on debug or shared, using one
   CPU node and a fitting rank count without a speed test. Require successful
   process exit, `TADA!`, the final timestamp and readable required output/restart.
3. **Production.** After smoke passes, cold-start the complete March 29–April 15
   period on **two exclusive CPU nodes, 104 MPI ranks per node (208 total)**.
   Use an eligible CPU queue; `partition: auto` lets Kestrel route by resources
   and walltime. No canary, separate spin-up job, benchmark or compatibility probe.
4. **Validate.** Audit all 17 simulated days against the cold-start zero-IINT
   anchor and independent NetCDF clocks. At the fixed cadences, expect 4,081
   station and 137 full-grid records including both endpoints, plus a valid
   final 3-D restart. The regional validation runner compares only
   **[April 1, April 15)**. Preserve the full model series for interpolation;
   exclude spin-up and the end timestamp from comparison samples and metrics.

## Attempts and stopping

Each stage has **three total attempts: the initial run plus two corrected
reruns**. Smoke and production have separate cumulative counters. An accepted
Slurm submission counts, including startup, runtime, timeout or numerical failure.
An authentication wait or rejected submission with no job ID does not count;
resolve uncertain submission identity before retrying. Never resubmit the same
attempt script, create extra stages, or reset counts after a repair or chat resume.

Use Run Control `attempt --paper-stage smoke|production --index 0|1|2
--controls <diagnosed-controls.json> --base-namelist <stage.nml>` with the normal
project/input/resource arguments. Do not generate an exhaustive stability plan.
Retain each job ID, terminal scheduler evidence, diagnostics and correction in its
trial-local attempt manifest/report. Only one trial-owned integration job may be
active at a time; reconcile it before a rerun.

Correct diagnosed setup, input, parameter, stability or runtime mistakes within
the remaining attempts. Revalidate changed dependencies and preserve old hashes
and evidence. Keep region, dates, forcing physics and validation method fixed;
never tune to improve a score. If a correction invalidates a passed smoke, another
smoke consumes its remaining slot. Build/input-reader checks must not advance
model time outside these counted runs. Do not edit shared skills.

After a third unsuccessful attempt in either stage, set `attempt_limit_reached`,
report what failed and stop. An earlier explicit inability is also a valid terminal
report. Preserve artifacts. Bear controls campaign launches, concurrency, timing,
pause/resume and stop decisions; report external waits without changing the human
clock. Do not create campaign controllers, hooks, timers or scheduled supervision.
On a human stop, stop only this trial's workers and jobs and report anything still
active. Keep workflow completion and physical assessment separate; Bear judges
acceptance. A stable run alone is not validation completion.
