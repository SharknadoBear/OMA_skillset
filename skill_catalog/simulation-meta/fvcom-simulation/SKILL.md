---
name: fvcom-simulation
description: Orchestrate reproducible FVCOM simulations from a science request through build, preparation, Kestrel execution, stability, benchmarking, and validation. Use for complete barotropic-tide studies or FVCOM performance benchmarks; delegate all scripts and remote actions to the named specialist skills.
---

# FVCOM Simulation

Interpret the science request and route the complete workflow. This skill is an orchestration contract only: it contains no scripts, performs no SSH itself, and never replaces the specialist skills that create or inspect artifacts.

Read [simulation request contract](references/simulation_request_v1.md) before initializing a project and [state and manifest contract](references/state_contract.md) before resuming one.

## Modes

- `barotropic_tide`: prepare, stabilize, benchmark, run, and validate a tide-only FVCOM case.
- `benchmark`: use a validated restart and scientifically identical 24-hour segments to compare Kestrel layouts. Do not interpret a benchmark as a validated simulation.

Default the project to `Workspace/Simulation/fvcom-simulation/projects/<case_id>` and its Kestrel mirror to `/scratch/yhuang168/FVCOM_Simulation/<case_id>`. Use the fixed project layout in the request contract. Attempts are immutable; resume a complete stage or create the next numbered attempt.

## Initialize

1. Parse the prompt as `simulation_request_v1`. Preserve explicit dates and settings. For the Galveston tide default use 2025-04-01 through 2025-05-01 UTC for analysis, seven spin-up days beginning 2025-03-25, 20 C, 30 PSU, ten uniform sigma layers, all TPXO constituents, and all eligible NOAA CO-OPS stations inside the retained wet domain.
2. Invoke `$fvcom-run-control` to initialize the project, provenance ledger, `project_status.json`, `commands.jsonl`, and `report.html`.
3. Ask the human only for secure Password+OTP entry after `$kestrel-hpc` displays its credential window. Never receive a password or OTP in chat or write it to an artifact.
4. Preserve accepted inputs. Hash and freeze an accepted mesh package before creating project-owned copies or derivatives.

## Exactly Three Initial Workers

Launch exactly three subagents concurrently and require each to return a typed role manifest satisfying the common worker envelope in the state contract:

1. **Grid worker** invokes `$fvcom-grid-generation` for a fresh case, or verifies and freezes an accepted delivery. It later invokes `$fvcom-preconfiguration` after the join.
2. **TPXO worker** invokes `$tpxo9v5-data-fetcher` to inventory and stage a registered, padded, model-neutral harmonic product. It does not create FVCOM forcing before the exact OBC contract exists.
3. **Configuration/build worker** invokes `$fvcom-namelist-configuration` for a provisional configuration and `$fvcom-build`, which in turn uses `$kestrel-hpc` for the remote source audit and build.

The parent remains the sole coordinator. Do not add a fourth initial worker, split one role across hidden workers, or let workers overwrite one another's artifact roots.

## Join and Preparation Gate

Wait for all three manifests. Advance only when each is `ready`, hashes are present, resume tokens are unique, the executable is frozen, the grid delivery is benchmark-ready, and its `fvcom_tge_boundary_junction_gate.passed` value is true. After the build and grid workers join, invoke `$fvcom-grid-generation`'s `audit_fvcom_tge_topology.py` against the delivered 2DM and the exact frozen-tree `tge.F`; require both the reproduced cell-sum decision and source-pattern/SHA binding to pass. This second, source-bound join gate is mandatory even when an earlier algorithm-only grid audit passed. A registered-data or authentication requirement is `waiting_user`, not a scientific failure.

For each grid case in `accepted_t6v6`, then `fresh_reproduction` order:

1. Invoke `$fvcom-preconfiguration` with the complete grid-delivery contract and the passing source-bound TGE junction audit. Require metric `_grd.dat`, positive-down depth, per-node geodetic latitude in `_cor.dat`, exact plural-OBC identity/order, ten uniform sigma layers, and a hash-bound manifest. Its independent TGE/source check must agree cell-for-cell with the Grid gate.
2. Return the exact OBC nodes to the TPXO worker. Invoke `$fvcom-tpxo-tides` to create the monotonic six-minute UTC elevation forcing for the complete run window. Require all discovered constituents; Galveston expects 22 and a different count blocks until explained.
3. Invoke `$noaa-coops-tides` to discover water-level and current stations inside the actual wet polygon. Quantitative current validation admits only downward-looking all-bin profiles; record why side-looking instruments are excluded.
4. Invoke `$fvcom-namelist-configuration` to generate the final namelist and cell-based station file. Reject any surface, atmospheric-pressure, river, temperature/salinity OBC, mean-flow, wave, ice, biology, sediment, or particle forcing.
5. Recompute every file reference and hash. Do not submit a bundle whose mesh, OBC, forcing, namelist, executable, or station mapping disagrees.

## Numerical Controls and Stability

Invoke `$fvcom-run-control` to derive the initial external step from the minimum element altitude and local gravity-wave speed at CFL 0.5, rounded down to 0.1 s. Choose `ISPLIT` so the initial internal step is no more than ten external steps and no more than the 2 m/s conservative advective limit. Set each OBC sponge radius to three local median boundary-edge lengths and coefficient to 0.0025.

Run, one job at a time through `$kestrel-hpc`: input smoke, three-day canary including the full ramp, and seven-day spin-up. An attempt is stable only when `srun` succeeds, the debug log reaches `TADA!`, the requested final timestamp exists, and required NetCDF/restart files are readable.

On failure, let `$fvcom-run-control` create immutable attempts using only:

1. external-step multipliers 0.75, 0.50, 0.25;
2. deduplicated `ISPLIT` values baseline, half-baseline, 1;
3. sponge-radius multipliers 1, 2, 4;
4. sponge-coefficient multipliers 0.5, 1, 2;
5. the remaining deduplicated cross-product, ordered by increasing departure from baseline.

Update root `report.html` after every attempt. Never tune validation metrics, change the grid, add forcing physics, or edit the frozen source/executable in this loop. Stop only for a user stop or after the permitted finite strategy is exhausted and the evidence requires a grid, forcing-formulation, or source change.

## Benchmark, Production, and Validation

After a stable spin-up, invoke `$fvcom-benchmark` and `$kestrel-hpc` for sequential, identical 24-hour restart segments at ranks `1,2,4,8,13,26,52,104,156,208,312,416` on 104-core CPU nodes. Extend by 104 ranks while performance improves until two successive layouts are slower. Repeat the Pareto neighborhood three times. Report fastest, least-node-hour, and Pareto-knee layouts; use the knee for the 30-day run unless the request chooses another objective.

Invoke `$fvcom-tidal-validation` after retrieving condensed station output locally. Compare model elevation with both NOAA GMT/MSL observations and astronomical predictions. Mean-align only over the common period and record the offset; never imply equal vertical datums. Total water level is mandatory but non-scoring for a tide-only model. Compare `ua/va` only with vertically weighted downward-looking all-bin profiles. Force all constituents but interpret only one-month-resolvable harmonics.

Use two independent terminal fields:

- `workflow_status=validation_complete` when every comparison is reproducible and scientifically valid.
- `scientific_assessment=diagnostic-pass|diagnostic-advisory|invalid` for the physical result.

Informational research thresholds cannot tune or block the first two Galveston workflows.

## Terminal States

Write exactly one current state in `project_status.json`: `initialized`, `awaiting_authentication`, `parallel_preparation`, `preparation_join`, `input_assembly`, `stability_smoke`, `stability_canary`, `stability_spinup`, `benchmarking`, `production`, `validation`, `validation_complete`, `waiting_user`, `user_stopped`, or `scope_exhausted`. `scope_exhausted` requires the complete allowed stability strategy and explicit causal evidence. Never call a staged, submitted, or merely stable case complete.
