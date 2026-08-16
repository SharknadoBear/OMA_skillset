---
name: fvcom-surface-fluxes-forcing
description: Package and scientifically validate modular FVCOM surface forcing from already-prepared atmospheric matrices. Use when Codex needs to write any subset of wind speed or stress, prescribed or bulk heat inputs, precipitation/evaporation, and atmospheric pressure on a structured atmospheric grid or the native FVCOM node/element grid, enforce FVCOM variable and time contracts, create combined or split NetCDF files, or generate surface-forcing QA plots and namelist guidance.
---

# FVCOM Surface Fluxes Forcing

Use this skill only after atmospheric fields have been prepared on their final structured grid or exact FVCOM node/element locations. Keep data fetching, horizontal interpolation, temporal resampling, bulk-flux calculation, and physical reconstruction outside this writer.

## Core workflow

1. Read the complete neutral schema and sign conventions in [input_contract.md](references/input_contract.md).
2. Select any non-empty subset of `wind`, `heat`, `freshwater`, and `pressure` from the requested FVCOM configuration.
3. Choose wind `speed` or `stress` and heat `direct` or `bulk`. Default bulk processing to `COARE26Z`.
4. Validate required variables, explicit units, absolute-pressure metadata, spatial shapes, exact mesh IDs, and UTC coverage before writing.
5. Build with `file-layout=auto`: combine compatible packages and split only when requested or required by the COARE40 pressure-unit conflict.
6. Inspect every generated map, time series, package diagnostic, namelist fragment, and JSON health report before accepting the bundle.

## Check dependencies

Require `numpy`, `scipy`, `netCDF4`, and `matplotlib`. If any is missing, notify the user that it must be installed on this equipment before continuing.

```powershell
python -c "import numpy, scipy, netCDF4, matplotlib; print('dependencies OK')"
```

## Build from prepared NetCDF

Structured atmospheric grid:

```powershell
python scripts/build_fvcom_surface_fluxes.py `
  --source prepared_surface.nc --layout structured `
  --packages wind heat freshwater pressure `
  --wind-mode speed --heat-mode direct `
  --pressure-reference absolute `
  --case-name example --output-dir forcing
```

FVCOM-native matrices:

```powershell
python scripts/build_fvcom_surface_fluxes.py `
  --source prepared_native.nc --layout fvcom --grd example_grd.dat `
  --packages wind pressure --wind-mode stress `
  --pressure-reference absolute `
  --model-start 2020-01-01T00:00:00Z --model-end 2020-01-03T00:00:00Z `
  --case-name example --output-dir forcing
```

Use repeated `--var ROLE=NAME` options for unambiguous noncanonical source names. Never use overrides to conceal an uncertain physical meaning.

## Package and safety rules

- Write structured files with `source = "wrf grid (structured) surface forcing"` and native files with `source = "FVCOM grid (unstructured) surface forcing"`.
- Require exact native `node_id` and `element_id` order. Wind is element-centered; heat, freshwater, and pressure are node-centered.
- Require both precipitation and evaporation magnitudes for the freshwater package. Convert evaporation to negative water loss only while writing.
- Require a selected wind-speed package for bulk heat, or use `--external-wind-speed` only after confirming a separate compatible FVCOM wind file.
- Treat direct and bulk heat as mutually exclusive. Do not substitute radiative flux for total net surface heat flux.
- Require `pressure_reference = "absolute"`. Never infer or repair a pressure departure from its magnitude.
- Use COARE26Z pressure in Pa. This FVCOM source passes COARE40VN pressure in hPa, while independent atmospheric pressure remains Pa. Automatic layout therefore splits COARE40VN heat from active inverse-barometer pressure; explicit unsafe combination fails.
- Preserve source timestamps without resampling. Require the forcing to bracket optional model start/end bounds.
- Publish only `NETCDF3_CLASSIC` files whose `Times` and `Itime`/`Itime2` axes agree exactly and whose active fields pass the zero-NaN gate.

## Use the NumPy API

Import `write_surface_forcing_bundle` from `scripts/surface_fluxes_core.py`. Supply UTC timestamps, canonical role-keyed arrays, an equally keyed units mapping, package modes, and either structured latitude/longitude or a parsed `MeshGeometry`. The API applies the same unit, sign, dependency, staging, and FVCOM contract gates as the CLI.

## Validate existing files

```powershell
python scripts/validate_fvcom_surface_fluxes.py `
  --forcing forcing/example_surface.nc `
  --qa-dir forcing/example_qa --report forcing/example_validation.json
```

Pass every split member together so cross-file time alignment is verified. Validation regenerates start/mid/end maps, representative-location time series, wind direction/magnitude, heat-component, freshwater-budget, pressure, and inverse-barometer-equivalent diagnostics as applicable.

## Test the toolbox

```powershell
python scripts/selftest_surface_fluxes.py
```

Retain synthetic artifacts when needed:

```powershell
python scripts/selftest_surface_fluxes.py --work-dir WORK
```

Use an external connector such as `$cfsv2-fetcher` only to prepare a separate validation input. Clearly label any synthetic validation-only field; never present it as observed or model-derived forcing.
