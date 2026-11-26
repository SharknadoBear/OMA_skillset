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

## Signed heat and water budgets

Use the same `ART1`, `h+zeta`, wet masks and positive layer fractions as the model for storage. The current adapter implements the freshwater model's 1,000 kg/m³ reference density and 4,186 J/(kg K) heat capacity. It integrates hourly surface fluxes, P/E on its native irregular interval clock, and signed river flux. Incoming river heat uses prescribed river temperature. Withdrawals use native endpoint control-area fractions, the corresponding cell layer fractions and local simulated donor temperatures. Surface freshwater carries the local surface-layer temperature. Include shortwave transmitted to the bed using the runtime attenuation parameters.

The result reports water and heat residuals per six hours and over the full window, total heat uptake, and signed inflow/outflow heat relative to 0 °C. The model’s recorded heat field is compared at matching source times. A zero-valued initialization record before the first forcing update is labeled separately. Wet exposure, donor temperature and model surface temperature are reconstructed between native state records. The result records the submitted scalar positivity and bottom-gradient controls. Native temperature clipping and the terrain-following bottom-gradient correction can change heat content each internal step. Six-hourly states do not resolve those separate terms. If an experiment has a material heat residual, compare the same initial state and forcing with a read-only within-step heat ledger and document spatial and basin-integrated replay differences. Independent instrument scoring and operational-model comparisons use the distinct [lake validation workflow](lake_validation.md).
