---
name: fvcom-tidal-validation
description: Condense FVCOM station output and validate tide-only elevation and depth-integrated currents against NOAA CO-OPS observations and predictions. Use for reproducible time alignment, datum-offset disclosure, scalar/vector metrics, resolvable harmonics, tidal ellipses, plots, and validation HTML.
---

# FVCOM Tidal Validation

Run locally after `$kestrel-hpc` retrieves compact station output. Do not install analysis environments or plot on Kestrel.

## Inputs and Scientific Rules

- Require exact model case/executable/input hashes and station mappings.
- Support the request's explicit water-level proxy policy. Preserve original gauge and model-cell coordinates, geometric distance, depth, cell/node IDs, three-node weights and geometry-review hash through condensation, tables and reports. Outside status alone is neither a run nor closure blocker. Label cell-centroid proxies accurately; never tune their selection against agreement scores. Current spatial/profile/period gates remain unchanged.
- Mixed mappings may retain `model_diagnostic` rows with `validation_eligible=false`. Audit every row against full NetCDF station order, retain their CSVs in `diagnostic_products`, and join NOAA tables only from water-level/current comparison `products`. Unknown roles or diagnostics claimed as NOAA comparisons remain invalid.
- For cell-centroid comparisons, `audit_station_cell_sampling.py` checks exact source-node membership and equal weights, then compares station elevation with the three-node mean at shared IINTs after independent UTC/hash certification. The case validation runner requires the audited full-grid files and retains this result. Missing samples, changed connectivity or elevation disagreement fail the output sampling gate.
- Water-level CSV contains UTC `time`, `model`, `observed`, and `predicted` columns. Align each comparison on common timestamps, subtract each series' common-period mean, and record the model-minus-reference mean offset. Do not label the offset a datum conversion.
- NOAA total observed water level is mandatory but non-scoring because tide-only FVCOM omits atmospheric and river residuals. NOAA astronomical prediction is the primary tide comparison.
- Current CSV contains UTC `time`, `model_u`, `model_v`, `observed_u`, and `observed_v` in m/s. Admit only downward-looking all-bin profiles prepared by `$noaa-coops-tides`; compare their documented vertically weighted vector with FVCOM `ua/va`.
- Force all available TPXO constituents. Harmonic validation uses request order as scientific priority, retains one representative per frequency cluster separated by the Rayleigh limit `1/T`, and labels its amplitude/phase as a cluster diagnostic. Report omitted aliases such as P1 relative to K1 and K2 relative to S2 rather than claiming that either pair is independently resolved.

Condense each retrieved station stack using the exact station mapping, immutable run namelist, and verified startup restart. For hotstart, reconstruct UTC as `START_DATE + (iint - startup_iint) * EXTSTEP_SECONDS * ISPLIT`, where `startup_iint` comes from the startup record at START_DATE and is bound to its SHA-256; explicit coldstart uses zero. Use `--startup-restart-netcdf` for relocated inputs. A supplied audited anchor is rechecked against the actual startup restart. Reject missing startup semantics/evidence, masked or nonfinite required station fields, and conflicting duplicate records. Floating MJD time is a precision-aware consistency check. Require exact requested endpoints and the production audit's cadence/count checks before validation. The shared reader lives in `fvcom-run-control/scripts/fvcom_time_anchor.py`.

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
When a station inventory is supplied, every eligible comparison station must have
a matching validation table. Preserve the envelope-discovered but wet-domain-
excluded gauges, proxy geometry and non-downward profiler reasons in the final HTML rather than
silently reducing the station count.

For a complete hash-bound production project, invoke `scripts/run_case_validation.py --project PROJECT --grid-case GRID_CASE --attempt run/GRID_CASE/attempts/PRODUCTION`. Add `--observation-manifest` when observations belong to a variant-specific directory and `--forcing-manifest` for a revised forcing derivative. The runner requires `fvcom_input_freeze_v1`, the frozen executable binding, actual forcing constituent order, eligible station inventory, and a passing `fvcom-run-control` production audit with rule version `startup_anchor_exact_clock_3d_v1`. It verifies actual input/output hashes and the startup restart before condensing, joining and validating. Forcing must carry matching manifest/NetCDF `utide_exact_time_nodal_v1`, all-false UTide flags, version and builder hash; historical midpoint products cannot certify completion under the exact-time contract. Results use a new immutable analysis directory; `--check-only` verifies prerequisites without claiming completion. Run locally in the scientific NumPy/NetCDF4/pandas/matplotlib environment.

Set `workflow_status=validation_complete` when the comparison is complete and reproducible. Set `scientific_assessment` independently to `diagnostic-pass`, `diagnostic-advisory`, or `invalid`. For initial accepted/fresh regional tests, thresholds are informational and cannot trigger tuning or prevent workflow completion. If no eligible current profiles exist for the requested period, retain the inventory and explicitly report current validation unavailable; never substitute incompatible instruments. Missing required water-level comparisons remains a blocker.

## Validation

```powershell
python scripts/selftest_fvcom_tidal_validation.py
python scripts/selftest_condense_fvcom_station.py
python scripts/selftest_prepare_validation_tables.py
python scripts/selftest_run_case_validation.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
