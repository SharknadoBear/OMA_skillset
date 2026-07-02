---
name: glofas-data-fetcher
description: Fetch, validate, and inspect CEMS GloFAS historical river-discharge data through EWDS/CDSAPI. Use when Codex needs GloFAS v4.0 downloads, CDSAPI credential checks, ZIP/NetCDF validation, small smoke tests, annual/resumable regional downloads, Haines-area GloFAS/NWIS comparison products, or full-period mean-discharge grid summaries.
---

# GloFAS Data Fetcher

Use this skill for CEMS GloFAS historical discharge workflows that need reliable EWDS/CDSAPI requests, compact regional downloads, schema checks, and reproducible QC products. Keep credentials outside the skill: use `%USERPROFILE%\.cdsapirc` with EWDS `url:` and `key:` entries, and never print or copy token values.

## Quick Start

- Use `scripts/glofas_smoke_test.py` before a larger request. It downloads a few days, validates the ZIP/NetCDF payload, records the `valid_time - 1 day` flow-date convention, and writes compact evidence.
- Use `scripts/glofas_haines_1979_trial.py` for the original Haines 1979 one-year trial and NWIS comparison.
- Use `scripts/glofas_haines_1979_2025.py` for the Haines annual-resumable 1979-2025 workflow, nearest-pixel NWIS validation, scatter plots, and full-period mean map.

Example smoke test from a project root with a prepared Python environment:

```powershell
.\.venv-glofas\Scripts\python.exe C:\Users\huan111\.codex\skills\glofas-data-fetcher\scripts\glofas_smoke_test.py `
  --start-date 1979-01-01 `
  --end-date 1979-01-03 `
  --area 59.85 -136.30 58.10 -134.50 `
  --output-dir workspace\_runtime\glofas_skill_smoke
```

## Workflow

1. Confirm `.cdsapirc` exists and has nonempty `url:` and `key:` entries without displaying values.
2. Estimate storage before larger requests. Annual Haines ZIPs are small, but EWDS cost limits can reject multi-year batches; prefer one-year requests unless a smaller test proves a batch size is accepted.
3. Request `cems-glofas-historical` with `system_version=version_4_0`, `hydrological_model=lisflood`, `product_type=consolidated`, `variable=river_discharge_in_the_last_24_hours`, `data_format=netcdf`, and `download_format=zip`.
4. Validate every response before downstream processing: reject HTML/error pages, require a valid ZIP, require at least one NetCDF member, and require `dis24`, latitude, longitude, and a time coordinate.
5. Interpret GloFAS `valid_time` as the end of the previous 24-hour period. For daily discharge analyses in this project, use `flow_date = valid_time - 1 day`.
6. Write compact artifacts outside the skill folder: request JSON, ZIP inventory, NetCDF inventory, CSV summaries, plots, and run summaries.

## Haines Defaults

- Area: `[59.85, -136.30, 58.10, -134.50]` as `[north, west, south, east]`.
- Dataset variable: `river_discharge_in_the_last_24_hours`; NetCDF variable is usually `dis24`.
- Units expected from v4.0 historical discharge: metric discharge, typically `m**3 s**-1`.
- Existing Haines products use selected HUC12s and FVCOM boundaries under `workspace/hydropower/dhsvm_flownet/haines`.

## Validation Rules

- Do not continue from a downloaded file unless it is binary ZIP data and contains a NetCDF member.
- Stop and report the actual variables/dimensions if the NetCDF schema differs from `dis24(latitude, longitude, time/valid_time)`.
- Keep raw or large downloads in the project data/runtime area, never inside this skill.
- Do not store EWDS keys, passwords, MFA codes, or token-derived values in scripts, logs, memos, manifests, or examples.
