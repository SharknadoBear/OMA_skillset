# Freshwater spinup and water and heat budgets

Use Python 3.11+, NumPy 2 and netCDF4. These reducers work on a closed freshwater FVCOM grid with native spherical coordinates, positive-down depth, static sigma layers, spherical node-control areas, direct heat, signed edge rivers, zero forcing ramp, and native six-hourly output. Keep the large simulation files on their current filesystem. The JSON results contain metrics and producing-run identities.

```text
python scripts/prepare_lake_physics.py geometry --source accepted_native_grid.nc --output analysis_geometry.nc
python scripts/prepare_lake_physics.py spinup --inventory spinup_paths.json --output spinup_spec.json
python scripts/lake_physics.py spinup --spec spinup_spec.json --output sensitivity.json
python scripts/prepare_lake_physics.py budget --inventory budget_paths.json --output budget_spec.json
python scripts/lake_physics.py budget --spec budget_spec.json --output budget.json
python scripts/selftest_lake_physics.py
```

For spinup, supply `geometry`, `stage_plan` and `members` in the path inventory. Key the members by integer spinup length in days. Run Control supplies the stage plan with the common UTC comparison limits, dated spinup trials and preregistered daily convergence thresholds. The comparison period must span one or more whole UTC days. For a budget, supply `geometry`, `start_utc`, `end_utc`, `surface_manifest`, `surface_budget`, `river_manifest` and a `member`. Use full paths or paths relative to the directory where the preparer runs. Store its resulting SHA-256 specification with the results.

Each member uses the following structure.

```json
{
  "attempt_directory": "/project/run/native/attempts/attempt_spinup_07d_0000",
  "freeze": "/project/run/build/evidence/freeze.json",
  "native_audit": "/project/analysis/native_qa/native_audit.json",
  "submission": "/project/analysis/native_qa/submission.json",
  "accounting": "/project/analysis/native_qa/sacct.psv",
  "forcing": {
    "atmosphere": "/project/inputs/lake_atmosphere.nc",
    "pe": "/project/inputs/lake_precip_evap.nc",
    "river": "/project/inputs/lake_river.nc"
  }
}
```

The preparer finds the namelist matching the attempt manifest and discovers its native history stacks. Override `history_glob` if the run uses a different naming convention. The reducer requires matching executable, submission, successful native audit, startup hash, parent accounting row and native output SHA-256 identities. It also binds the actual forcing bytes. The comparison requires the same rank count, common inputs and physical namelist settings. Preserve the native QA output names when moving files. The job's elapsed seconds and allocated whole-node hours are recorded when the accounting columns are present.

## Spinup selection

Each day uses a trapezoidal average of the six-hourly snapshots. Temperature RMS uses the common wet 3-D volume and integrates partial sigma layers at 20 m. Current RMS uses native eastward and northward velocities and wet cell volumes. The thermocline lies halfway between layer centers with the strongest temperature decrease per metre. Each profile must have a surface at least 4 °C, at least 4 m water depth, at least 1 °C surface-to-bottom contrast, and a gradient of at least 0.1 °C/m. Report the jointly stratified area fraction. The median displacement uses native area and time weighting. Compare every daily gate with its specified threshold.

Select the shortest spinup agreeing with all longer preregistered trials. If unresolved, select the verified longest spinup and report `advisory_unresolved`. If the longest trial is unusable and no shorter trial passes against every longer one, report `blocked`. Use the selected trial's independently verified restart at the comparison start for subsequent benchmarking or production. The reducer checks the actual restart bytes against the native audit and carries the verified source job, comparison clock, archive path and SHA-256 into the selection. Run the reducer on the filesystem containing the archived restart.

## Wet-temperature eligibility for freshwater spinups

The `lake_superior_summer` profile requires a thermal screen for the entire native run, from startup through its final reported state. Other freshwater profiles can opt in by adding `freshwater_thermal_screen` to their stage plan. The object contains `policy_sha256` and `freezing_lookup_sha256`. Set both before launching trials. The preparer binds an adjacent project `request.json` when present or accepts an explicit `project_request`. A Lake Superior request with an older unqualified schedule requires a new versioned schedule.

Run Control can register `policy` and `freezing_lookup` paths relative to the stage plan directory. For independently staged copies, place `thermal_policy` and `freezing_lookup` paths alongside `geometry` and `stage_plan` in the thermal-screen and spinup inventories. The policy declares positive conservative and substantial temperature deficits under `cold_conservative_margin_C` and `cold_substantial_margin_C` and includes the table's `freezing_lookup_sha256`. The CSV header is `sea_pressure_dbar,air_free_temperature_C,air_saturated_temperature_C`. The table must span all wet pressures. Supply the certified `member` paths for each individual screening run.

```text
python scripts/prepare_lake_physics.py thermal-screen --inventory trial_paths.json --output trial_inputs.json
python scripts/lake_physics.py thermal-screen --spec trial_inputs.json --output wet_thermal_manifest.json
```

Add the resulting `thermal_manifest` path to that member in the spinup inventory, then run the ordinary spinup commands. The receipt identifies the submitted job, audited native fields, initial state, policy and full six-hour time series. The selector independently checks every wet layer against the pressure-dependent air-saturated freshwater freezing point using native pressure, layer thickness and local gravity. A substantial deficit in any sampled wet layer or a conservative deficit in the same wet layer on consecutive samples makes the member ineligible. A missing, foreign, incomplete or conflicting screen also excludes its restart, including the longest trial's advisory fallback. The selector accepts compatible case receipts of schema `fvcom_lake_member_thermal_tails_v1` under the same replay checks.

## Signed heat and water budgets

For a freshwater COARE26Z diagnostic, use the model's hourly integrated heat and external-volume ledgers. On the archived attempt run `python scripts/lake_coare_ledger.py --run ATTEMPT --native-audit QA/native_audit.json --output hourly.json`. The native audit schema `fvcom_coare_native_audit_v1` supplies the checked ledger hashes for every MPI rank. The reducer checks both model clocks, sums rank contributions, differences cumulative totals after each hour and checks energy and signed vapor identities. Join guarded atmospheric heat, heat in surface freshwater, light transmitted to the bed and inflow/outflow enthalpy to the wet-grid storage change at the matching six-hourly clocks. Use the external ledger's actual rain and signed vapor volumes for the water budget. A fresh restart begins a new set of rank files and cumulative totals. Match the absolute internal counter across segments before summing. [Build and native ledger specification](../../fvcom-build/references/freshwater_coare.md)

Use the same `ART1`, `h+zeta`, wet masks and positive layer fractions as the model for storage. The current adapter implements the freshwater model's 1,000 kg/m³ reference density and 4,186 J/(kg K) heat capacity. It integrates hourly surface fluxes, P/E on its native irregular interval clock, and signed river flux. Incoming river heat uses prescribed river temperature. Withdrawals use native endpoint control-area fractions, the corresponding cell layer fractions and local simulated donor temperatures. Surface freshwater carries the local surface-layer temperature. Include shortwave transmitted to the bed using the runtime attenuation parameters.

The result reports water and heat residuals per six hours and over the full window, total heat uptake, and signed inflow/outflow heat relative to 0 °C. The model’s recorded heat field is compared at matching source times. A zero-valued initialization record before the first forcing update is labeled separately. Wet exposure, donor temperature and model surface temperature are reconstructed between native state records. The result records the submitted scalar positivity and bottom-gradient controls. Native temperature clipping and the terrain-following bottom-gradient correction can change heat content each internal step. Six-hourly states do not resolve those separate terms. If an experiment has a material heat residual, compare the same initial state and forcing with a read-only within-step heat ledger and document spatial and basin-integrated replay differences. Independent instrument scoring and operational-model comparisons use the distinct [lake validation workflow](lake_validation.md).
