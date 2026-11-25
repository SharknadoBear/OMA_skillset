---
name: fvcom-initialization
description: Build and verify freshwater baroclinic FVCOM spherical-grid hotstarts from native NOAA Great Lakes OFS analysis fields and runtime pressure. Use for same-grid temperature, velocity, turbulence and density initialization, pressure-head and static reference-profile construction, and verified continuation from native FVCOM checkpoints.
---

# Freshwater FVCOM initialization

Use Python 3.11+ with NumPy and netCDF4. This adapter targets the spherical, hydrostatic, DENS1, Mellor–Yamada, AIR_PRESSURE, WET_DRY build of FVCOM_MP at `f8e4b50d4fd51409febcefc763cbc5dca11870e2`. Check the frozen binary's build/namelist and native reader probe when applying it to another study.

## Initialize a run

```text
python scripts/lake_initial.py --source snapshot.json --reference accepted_grid_source.nc --grid-contract lsofs_fvcom_dat_manifest.json --pressure surface_pressure.nc --output state_20250701
python scripts/test_initialization.py
```

Use a verified `great_lakes_ofs_snapshot_v1` fields receipt, an independent frozen native grid analysis, and a `fvcom_retained_spherical_grid_v1` contract with its actual grid, depth, and sigma files. The adapter verifies cell/node ordering, native triangle orientation, the full sigma axes, longitude wrapping, depth and all required state values. It uses the exact archived record and the exact UTC epoch of the run's absolute nodal air pressure in Pa, bound by ordered geographic coordinates or the runtime grid SHA-256. For a reader test before driving pressure is available, `--probe-use-analysis-pressure` emits a distinct `reader_probe` receipt.

Stage `lake_initial.nc` using `STARTUP_TYPE='hotstart'` and `STARTUP_TS_TYPE`, `STARTUP_UV_TYPE`, and `STARTUP_TURB_TYPE` all set to `'set values'`. Each independent analysis starts a new local step count of zero. Original native east/north layer currents, sigma vertical velocity, temperature, zero salinity, turbulence, wet flags, and zeta retain their original axes. Derive depth-averaged currents by integrating native layers with physical thickness. Initialize split-mode et from the instantaneous analyzed zeta. Read the receipt for sample min/max, deepwater temperature and vertical-shear checks.

Compute native DENS1 density anomaly with the source's Fofonoff–Millard pressure-dependent equation and spherical gravity. The static reference is the common wet-node mean profile on 601 fixed physical heights remapped to each native layer. The pressure head uses `-(air_pressure-101325)/(1025*g(latitude))` in metres. The receipt records source, ordered grid, forcing, native result and reference-profile hashes. Successful physical execution requires the separate scheduler log and returned model fields.

## Continue a native checkpoint

```text
python scripts/restart_handoff.py --restart native_restart.nc --initial-manifest state_20250701/initialization_manifest.json --time 2025-07-02T00:00:00Z --executable-sha256 SHA256 --source-job-id JOB_ID --checkpoint-sha256 SOURCE_JOB_CHECKSUM --output continuation.json
```

Obtain the checksum and executable binding from the job that created the checkpoint. Carry the complete native file, including its positive iteration, split-mode state and turbulence, into the continuation. Verify the first returned continuation timestamp and compare common fields to the uninterrupted segment with the same executable, ranks, timestep and forcings.
