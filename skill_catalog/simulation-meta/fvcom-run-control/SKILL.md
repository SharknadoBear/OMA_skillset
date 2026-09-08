---
name: fvcom-run-control
description: Initialize immutable FVCOM simulation projects, derive conservative time-step and sponge controls, plan stability attempts, generate Slurm attempt artifacts, audit TADA/final-time/output evidence, and maintain live HTML status. Use after a simulation request exists; use kestrel-hpc for every remote action.
---

# FVCOM Run Control

Use this skill for local orchestration artifacts and evidence. It never opens SSH or submits a job; invoke `$kestrel-hpc` for transfer, submission, monitoring, cancellation, and retrieval.

## Workflow

1. Initialize the fixed project from a `simulation_request_v1` JSON document:

```powershell
python scripts/fvcom_run_control.py init --request request.json --project PROJECT
```

2. Derive the baseline controls from the exact metric 2DM and its ordered OBC chains:

```powershell
python scripts/fvcom_run_control.py controls --mesh fvcom_grid.2dm --source-crs EPSG:4326 --metric-crs EPSG:32615 --output controls.json
```

The external step uses `0.5 * minimum_element_altitude / sqrt(g * local_max_depth)` and is rounded downward to 0.1 seconds. Supply source and metric CRS together when the delivered 2DM serializes lon/lat even though FVCOM DAT files are projected. `ISPLIT` keeps the internal step within ten external steps and the conservative `0.5 * altitude / 2 m/s` advective limit. Each OBC radius is three times its median boundary-edge length; the baseline coefficient is 0.0025.

Pass `--time-quantum-seconds 360` for the April six-minute-output experiment. FVCOM advances fixed internal steps and requires output/restart intervals to be exact step multiples. The helper lowers the external step to the largest 0.1-second value whose internal step divides this common time quantum; if necessary it first lowers ISPLIT until a solution exists. It records the original CFL-derived proposal and never raises either control. The stability plan preserves this alignment and deduplicates actual controls. For other schedules supply their common time quantum. The namelist stage gate independently rejects incompatible durations and cadences. Columbia's 1.3 s / ISPLIT 4 proposal fails this gate; 1.2 s / 4 passes without relaxing CFL.

Append every material local or Kestrel operation to the non-secret provenance ledger. Never place a password or OTP in this command:

```powershell
python scripts/fvcom_run_control.py record --project PROJECT --kind preprocessing --command "fvcom_2dm_to_dat.py ..." --artifact input/base/preconfiguration_manifest.json
```

At the preparation barrier, validate exactly three role manifests through their common typed envelope (role-specific schema names are allowed):

```powershell
python scripts/fvcom_run_control.py join --manifest gridding/worker_manifest.json --manifest forcing/shared_source/worker_manifest.json --manifest run/build/worker_manifest.json --output preparation_join.json
```

3. Create the finite, deduplicated stability strategy:

```powershell
python scripts/fvcom_run_control.py plan --controls controls.json --output stability_plan.json
```

Attempt zero is the baseline. The ordered single-axis retries precede the complete cross-product. Only external step, `ISPLIT`, sponge radius, and sponge coefficient may change.

4. Materialize a new attempt without overwriting evidence:

```powershell
python scripts/fvcom_run_control.py attempt --project PROJECT --grid-case accepted_t6v6 --plan stability_plan.json --index 0 --base-input INPUT_BASE --stage-namelist smoke=smoke_run.nml --stage-namelist canary=canary_run.nml --stage-namelist spinup=spinup_run.nml --module intel/2023.2.0 --module intel-oneapi-mpi/2021.11.0-intel --module netcdf-c/4.9.2-cray-mpich-intel --module netcdf-fortran/4.6.1-intel --remote-executable /scratch/path/fvcom --account hindcastra --partition standard
```

For stability work, supply exactly the smoke, canary, and spin-up namelists. The command creates ordered stage subdirectories so the same numerical controls advance only after the prior gate passes. A single `--base-namelist` remains available for non-staged runs.

The attempt command rewrites `INPUT_DIR` to the immutable attempt copy and `OUTPUT_DIR` to each stage run directory. Verify both bindings in `attempt_manifest.json`; otherwise a sponge override could be recorded without being consumed by FVCOM.

Always pass the frozen executable's recorded module-load request. The generated Slurm scripts purge the inherited environment, load that stack, and print the effective module list before `srun`. If a pre-submission audit invalidates an already-materialized attempt without running it, preserve that package and create a deterministic `--revision 1` (then 2, and so on) of the same strategy index; never silently overwrite it.

For production selected from a benchmark, preserve the measured launch layout as well as ranks, nodes and modules. Use `--cpu-bind cores --exclusive` when the selected benchmark used explicit core binding and exclusive nodes. Verify `execution_layout` in the attempt manifest and the generated Slurm script before submission; an unspecified scheduler default is not evidence of matching the measured binding.

5. After `$kestrel-hpc` retrieves logs and products, audit the attempt and refresh the report:

```powershell
python scripts/fvcom_run_control.py audit --stdout stdout.log --stderr stderr.log --expected-final 2025-04-01T00:00:00Z --netcdf output.nc --restart restart.nc --exit-code 0 --output status.json
python scripts/fvcom_run_control.py report --project PROJECT
```

The root report includes preparation manifests, observation preflight, stability attempts, benchmark submissions, and terminal benchmark records. Refresh it immediately after every stability or benchmark attempt is prepared, submitted, retried, or audited.

Stable requires exit code zero, `TADA!`, expected final timestamp evidence, and readable required NetCDF/restart files. Blowup text, fatal input, timeout, NaN, corrupt output, premature time, or missing TADA fails. Keep one stability job active at a time. Never edit the frozen executable or use validation skill scores to choose an attempt.

## Production Output Gate

Use `scripts/audit_fvcom_production.py` in the scientific NumPy/NetCDF4 environment before declaring monthly production complete. Supply logs, exit code, exact namelist/mapping, every station and full-grid stack, final restart, and `--lineage input_freeze.json`. For relocated inputs supply `--startup-restart-netcdf` pointing to the actual frozen startup restart. The `fvcom_input_freeze_v1` file map must include its SHA-256, mapping hash and run namelist hash.

Require 7,201 station and 241 full-grid records for the April endpoint-inclusive experiment, at exact 360/10,800-second cadences. The shared `fvcom_time_anchor.py` reads the startup record's IINT at START_DATE; it never assigns an arbitrary first output record to that date. Full-grid and restart Times, Itime and Itime2 must agree independently. Float32 MJD is checked at its actual precision. Equal stack-boundary duplicates are permitted; shifted, conflicting, missing, masked or nonfinite records fail.

The audit checks flag-enabled full-grid/station variables, all present known physical state fields, and the frozen 3-D restart's velocity, vertical velocity (`ww`/`omega`), temperature/salinity and turbulence state. Optional 3-D output fields may be absent when their namelist flags are false. The final restart must end at the exact requested time and IINT. A passing report records rule version `startup_anchor_exact_clock_3d_v1` and the externally verified time anchor for tidal validation.

## Terminal Rules

Stop for `user_stopped` or after every planned attempt has terminal evidence and none is stable. The latter is `scope_exhausted` only when the report identifies why a grid, forcing formulation, or source change is outside the authorized stability controls.

## Validation

```powershell
python scripts/selftest_fvcom_run_control.py
python scripts/selftest_audit_fvcom_production.py
python scripts/selftest_fvcom_time_anchor.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
