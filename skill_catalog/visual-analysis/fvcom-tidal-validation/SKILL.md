---
name: fvcom-tidal-validation
description: Condense FVCOM station output and validate tide-only elevation and depth-integrated currents against NOAA CO-OPS observations and predictions. Use for reproducible time alignment, datum-offset disclosure, scalar/vector metrics, resolvable harmonics, tidal ellipses, plots, and validation HTML.
---

# FVCOM Tidal Validation

Run locally after `$kestrel-hpc` retrieves compact station output. Do not install analysis environments or plot on Kestrel.

## Inputs and Scientific Rules

- Require exact model case/executable/input hashes and station mappings.
- Water-level CSV contains UTC `time`, `model`, `observed`, and `predicted` columns. Align each comparison on common timestamps, subtract each series' common-period mean, and record the model-minus-reference mean offset. Do not label the offset a datum conversion.
- NOAA total observed water level is mandatory but non-scoring because tide-only FVCOM omits atmospheric and river residuals. NOAA astronomical prediction is the primary tide comparison.
- Current CSV contains UTC `time`, `model_u`, `model_v`, `observed_u`, and `observed_v` in m/s. Admit only downward-looking all-bin profiles prepared by `$noaa-coops-tides`; compare their documented vertically weighted vector with FVCOM `ua/va`.
- Force all available TPXO constituents. Harmonic validation uses request order as scientific priority, retains one representative per frequency cluster separated by the Rayleigh limit `1/T`, and labels its amplitude/phase as a cluster diagnostic. Report omitted aliases such as P1 relative to K1 and K2 relative to S2 rather than claiming that either pair is independently resolved.

Condense every retrieved FVCOM station stack before joining observations. Use `scripts/condense_fvcom_station.py` with the exact station mapping and run namelist. FVCOM station `iint` counts internal steps, so reconstruct UTC as `START_DATE + (iint - first_iint) * EXTSTEP_SECONDS * ISPLIT`; do not use FVCOM station `time` directly because its float32 MJD values are too coarse to represent six-minute samples reliably near modern dates. The condenser must match station IDs and their order exactly, require readable `zeta`, `ua`, and `va`, reach `END_DATE`, and hash every product.

```powershell
python scripts/condense_fvcom_station.py --station-netcdf galveston_station_timeseries.nc --station-mapping station_mapping.json --run-namelist galveston_run.nml --output-dir output/case/condensed --manifest output/case/condensed/manifest.json
```

Join condensed products to NOAA caches with `scripts/prepare_validation_tables.py`. Water levels use exact UTC timestamp matches. NOAA current profiles are normally centered three minutes between the model's six-minute records, so linearly interpolate model `ua/va` to the observation timestamps and disclose that alignment in the manifest.

```powershell
python scripts/prepare_validation_tables.py --condensation-manifest output/case/condensed/manifest.json --observation-root analysis/observations --output-dir analysis/case/tables --manifest analysis/case/tables_manifest.json
```

Run:

```powershell
python scripts/fvcom_tidal_validation.py validate --water station_water.csv --current station_current.csv --lineage validation_lineage.json --station-inventory analysis/observations/station_inventory_case.json --output-dir analysis/case --report analysis/validation_report.html
```

The report includes bias, MAE, centered RMSE, normalized centered RMSE (by reference standard deviation), correlation, vector RMSE, complex correlation and phase, constituent amplitude/phase, and tidal-current ellipse metrics. It records coverage, interpolation, datum, bin-weighting, and excluded-station evidence.
When a station inventory is supplied, every eligible wet-domain station must have
a matching validation table. Preserve the envelope-discovered but wet-domain-
excluded gauges and non-downward profiler reasons in the final HTML rather than
silently reducing the station count.

Set `workflow_status=validation_complete` when the comparison is complete and reproducible. Set `scientific_assessment` independently to `diagnostic-pass`, `diagnostic-advisory`, or `invalid`. For the first two Galveston cases, thresholds are informational and cannot trigger tuning or prevent workflow completion.

## Validation

```powershell
python scripts/selftest_fvcom_tidal_validation.py
python scripts/selftest_condense_fvcom_station.py
python scripts/selftest_prepare_validation_tables.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
