---
name: fvcom-boundary-ts-forcing
description: Generate and scientifically validate FVCOM open-boundary temperature and salinity forcing from an existing sigma-ready NetCDF time series mapped to FVCOM boundary node IDs. Use when Codex needs to package per-node, per-sigma-layer T/S data as an FVCOM TIME SERIES OBC TS FILE, repair auditable missing values, enforce a zero-NaN gate, or create boundary time-series, time-depth, Hovmöller, and vertical-transect QA plots.
---

# FVCOM Boundary Temperature/Salinity Forcing

Use this skill only after temperature and salinity have been placed on every target FVCOM open-boundary node and FVCOM sigma layer. Keep source fetching, horizontal interpolation, z-to-sigma remapping, multi-source merging, and physical reconstruction outside this skill.

## Core rules

- Read the complete contract in [input_contract.md](references/input_contract.md) before preparing or diagnosing an input file.
- Require a geographic SMS `.2dm` mesh with explicit open nodestring ids, or FVCOM `_grd.dat` plus `_obc.dat`.
- Require positive-down boundary depth in metres and exact FVCOM node IDs.
- Preserve source sigma orientation. Accept monotonic surface-to-bottom or bottom-to-surface coordinates; never silently reverse layers.
- Treat time as UTC. Use `--assume-utc` only after confirming timezone-free source timestamps are UTC.
- Preserve a regular source time axis by default. Supply `--start`, `--end`, and `--dt-seconds` together to resample.
- Never extrapolate outside source time coverage or resample across an excessive source gap.
- Keep original finite T/S values unchanged. Audit every repaired value and fail before publishing if any NaN remains.
- Inspect every default figure and the JSON health report before accepting a production forcing file.

## Check the environment

Run with a Python environment containing `numpy`, `scipy`, `netCDF4`, and `matplotlib`. If any package is missing, notify the user that it must be installed on this equipment before continuing.

```powershell
python -c "import numpy, scipy, netCDF4, matplotlib; print('dependencies OK')"
```

## Build forcing

For a geographic SMS mesh:

```powershell
python scripts/build_fvcom_boundary_ts.py `
  --source sigma_ready_boundary_ts.nc `
  --mesh-2dm mesh.2dm --open-ns 1 `
  --case-name example_case `
  --output example_tsobc.nc
```

For FVCOM DAT files and an explicitly resampled target axis:

```powershell
python scripts/build_fvcom_boundary_ts.py `
  --source sigma_ready_boundary_ts.nc `
  --grd example_grd.dat --obc example_obc.dat `
  --start 2020-01-01T00:00:00Z --end 2020-01-31T21:00:00Z --dt-seconds 10800 `
  --case-name example_case `
  --output example_tsobc.nc
```

Use variable-name or unit overrides only when the source metadata is absent or unambiguous external knowledge exists. The builder converts recognized Kelvin temperature to Celsius, validates practical salinity, reorders source nodes by exact ID, and ignores only explicitly reported extra source nodes.

Missing-value repair proceeds deterministically through time, vertical sigma coordinate, and cumulative distance within each disconnected arc. Endpoint and single-valid-value propagation are allowed inside those axes. Repair never crosses arcs, mixes temperature with salinity, changes an original finite value, or injects an arbitrary constant. Any irreparable value fails the build and leaves the requested output untouched.

The output is atomic `NETCDF3_CLASSIC` with `obc_nodes`, `obc_h`, `siglay`, `siglev`, `obc_temp`, `obc_salinity`, and exact FVCOM time representations. The float32 MJD `time` variable is retained for compatibility; use `Times` and `Itime`/`Itime2` as the millisecond-exact UTC representations.

## Review mandatory QA

The builder always reads the staged forcing back and creates:

- A boundary map with five representative nodes per arc.
- Surface, middle, and bottom T/S time series at the representative nodes.
- T/S time-versus-physical-depth curtains at each arc midpoint.
- Complete-boundary surface, middle, and bottom Hovmöller diagrams for both variables.
- Along-boundary vertical T/S transects at the start, middle, and end of the deployment.
- Missing-data repair diagnostics and an explicit final zero-NaN result.
- `health_report.json` with hashes, geometry, exact time checks, units, sigma orientation, ranges, repair methods/counts, and artifact paths.

Treat `pass_with_repairs` as a required scientific-review signal, not the same as an untouched source pass.

## Validate an existing file

```powershell
python scripts/validate_fvcom_boundary_ts.py `
  --forcing example_tsobc.nc `
  --mesh-2dm mesh.2dm --open-ns 1 `
  --qa-dir example_tsobc_qa
```

Validation requires exact boundary-node order and depth, valid sigma coordinates, correct units, regular millisecond-consistent UTC time, reviewed physical ranges, and zero missing values. An existing forcing file cannot reconstruct its pre-build missing mask, so its missing-data figure verifies only the finished zero-NaN state unless a builder repair audit accompanies it.

## Test the skill

Run the synthetic suite after installation or code changes:

```powershell
python scripts/selftest_boundary_ts.py
```

Retain evidence when needed:

```powershell
python scripts/selftest_boundary_ts.py --work-dir WORK
```

Use a source connector such as `$hycom-fetcher-codex` only to prepare an external native-grid test input; perform horizontal and sigma remapping outside this forcing writer. Do not copy acquisition logic into this package.
