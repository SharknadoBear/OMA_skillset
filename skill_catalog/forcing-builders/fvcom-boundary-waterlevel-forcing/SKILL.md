---
name: fvcom-boundary-waterlevel-forcing
description: Generate and scientifically validate FVCOM time-series open-boundary elevation forcing from an existing combined water-level record. Use when Codex needs to map NetCDF or tidy CSV water levels from boundary nodes, stations, or geographic grids onto FVCOM OBC nodes; encode exact FVCOM UTC time variables; inspect spectra; or create sample-node and complete-boundary Hovmöller QA plots.
---

# FVCOM Boundary Water-Level Forcing

Use this skill only after the desired water-level series is scientifically combined and datum-corrected. Do not fetch TPXO/HYCOM/observations, reconstruct harmonics, or manufacture missing tidal, subtidal, or sea-level-rise components here.

## Core rules

- Require a geographic SMS `.2dm` mesh plus explicit open nodestring ids, or FVCOM `_grd.dat` and `_obc.dat` files.
- Accept NetCDF or tidy CSV sources described in [input_formats.md](references/input_formats.md).
- Treat all time as UTC. Pass `--assume-utc` only after confirming timezone-free timestamps are UTC.
- Preserve a regular source time axis by default. Supply `--start`, `--end`, and `--dt-seconds` together to resample.
- Never extrapolate beyond source coverage or interpolate across an excessive gap.
- Require explicit permission to broadcast a single non-spatial series across an entire boundary.
- Record the vertical datum supplied by the user; never infer or silently correct it.
- Accept forcing only after the default read-back, time, spectrum, sample-series, and complete-boundary Hovmöller validation passes.

## Check the environment

Run commands with a Python environment containing `numpy`, `scipy`, `netCDF4`, and `matplotlib`. If a package is missing, notify the user that it must be installed on this equipment before continuing.

```powershell
python -c "import numpy, scipy, netCDF4, matplotlib; print('dependencies OK')"
```

## Build forcing

For a geographic 2DM mesh:

```powershell
python scripts/build_fvcom_boundary_waterlevel.py `
  --source combined_water_level.nc `
  --mesh-2dm mesh.2dm --open-ns 1 `
  --case-name example_case --datum "NAVD88" `
  --output example_obc.nc
```

For existing FVCOM DAT files and a target cadence:

```powershell
python scripts/build_fvcom_boundary_waterlevel.py `
  --source stations.csv --units cm `
  --grd example_grd.dat --obc example_obc.dat `
  --start 2020-01-01T00:00:00Z --end 2020-12-31T23:00:00Z --dt-seconds 3600 `
  --case-name example_case --datum "MSL" `
  --output example_elevation_obc.nc
```

Use variable-name overrides when automatic discovery is ambiguous. Use `--broadcast-single-series` only when applying one record identically to every OBC node is scientifically intended.

The builder writes the forcing only after validation succeeds. It also writes `health_report.json` and these default QA figures:

- Boundary map and representative-node time series.
- A high-resolution energetic tidal window.
- Welch power spectrum with major constituent markers.
- Total, 4–34 h tidal, 34 h–90 d subtidal, and >90 d VLF/SLR-scale Hovmöller diagrams using every boundary node.

Treat the >90 d result as very-low-frequency or mean-sea-level variability. Do not call it detected sea-level rise without independent attribution.

## Validate an existing file

```powershell
python scripts/validate_fvcom_boundary_waterlevel.py `
  --forcing example_elevation_obc.nc `
  --grd example_grd.dat --obc example_obc.dat `
  --qa-dir example_elevation_obc_qa
```

The validator requires `NETCDF3_CLASSIC`, exact boundary-node order, finite elevation in metres, one-based `iint`, and mutually consistent `Times`, `Itime`, `Itime2`, and float MJD `time`. It treats `Times` and `Itime`/`Itime2` as the exact millisecond UTC representations; float32 MJD remains a compatibility field.

## Test the skill

Run the bundled synthetic self-test before using a new environment or after changing the scripts:

```powershell
python scripts/selftest_boundary_waterlevel.py
```

For a retained forward test on a real generated mesh:

```powershell
python scripts/selftest_boundary_waterlevel.py `
  --work-dir WORK `
  --mesh-2dm mesh.2dm --open-ns 1
```

Inspect the plots and JSON report, not only the exit code, before accepting a production forcing file.
