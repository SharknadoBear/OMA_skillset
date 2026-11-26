---
name: fvcom-initialization
description: Build and verify same-grid freshwater or cross-grid estuarine baroclinic FVCOM hotstarts from native NOAA OFS analysis fields, retained grid geometry and runtime pressure. Use for wet-connected salinity and temperature mapping, balanced open-boundary startup, density and depth-mean currents, or verified native continuation.
---

# FVCOM initialization

Use Python 3.11+ with NumPy and netCDF4. The lake adapter targets spherical DENS1. The estuary adapter targets projected Cartesian DENS2. Both use hydrostatic FVCOM_MP at `f8e4b50d4fd51409febcefc763cbc5dca11870e2`. Check the frozen build, namelist and native reader probe for the study.

## Initialize a projected estuary

Install SciPy, Shapely 2 and pyproj alongside the common dependencies. Obtain the native fields and their UTC, archive identity and SHA-256 with the appropriate OFS fetcher. Supply one completed projected-target native restart as the static geometry and partition template. Use the same rank count and verified static files as that run.

```text
python scripts/estuary_initial.py --request estuary_request.json --output-dir startup
python scripts/test_estuary_initial.py
```

Set `schema` to `fvcom_estuary_initial_request_v1`. Provide these request fields.

| Field | Contents |
| --- | --- |
| `utc`, `output_basename` | Exact analysis time in UTC and local restart filename ending in .nc |
| `paths` | Relative-to-request or absolute paths for `native_source`, `restart_template`, `preconfiguration`, `grid_directory`, `surface_pressure` and `tidal_elevation` |
| `required_sha256` | SHA-256 for each file in paths. Omit the directory. |
| `runtime_grid_sha256` | Hashes keyed by grd, dep, sig, obc, cor and spg of the scientific run's actual DAT files |
| `min_depth_m`, `max_wet_path_m` | Runtime wetting depth and maximum permitted along-mesh shoreline extension |
| `vertical_reference_statement` | Actual source and target vertical references, any assumed offset and evidence |
| `turbulence` | `vertical_mixing_coefficient` and `vertical_prandtl_number` matching the runtime default turbulence branch |
| `temperature` | Optional fallbacks if native temperature is absent. A target-grid, time-invariant `temp(siglay,node)` file needs `documented_3d_file`, `sha256` and `provenance`, matching siglay and global grid_sha256. An explicit `uniform_degrees_C` has lowest precedence. |
| `boundary_alignment` | Optional object with `type` equal to `type1_bundled_elevation_unramped` and positive `width_m`. Requires type-1 time-series forcing, IRAMP=0 and the complete effective elevation including any external inverse barometer. |

The source is a time-matched georeferenced FVCOM field with native nbe/nv, wet masks, practical salinity, velocity, surface elevation, stress and sigma geometry. Validate the archive clock independently of its filename. A documented, spatially varying 3-D stationary temperature takes precedence over an explicitly chosen constant when analysis temperature is unavailable. The salinity pilot holds the initialized 3-D temperature fixed during integration.

The adapter selects the wet source component at the accepted ordered ocean boundary and locates target nodes and face centers in its native triangles. It extends shoreline gaps along connected wet target edges, records donor IDs and distances and rejects unsupported components or excess path lengths. Nodal profiles combine barycentric source vertices after interpolation by depth below the local instantaneous surface. Cell currents use the containing native face and rotate from east/north into projected axes. The donor endpoint bounds each transferred profile. Inspect wet coverage and bathymetric mismatches in the manifest.

Run the raw transfer in its own directory to measure any initial boundary jump. For the optional aligned transfer the model-native effective type-1 level supplies the boundary, including the wet-depth floor. An edge-weighted harmonic extension tapers to zero over the specified wet-graph distance. Rebuild dependent wet masks, split elevation, physical sigma layers, tracers, density, reference profiles and depth-mean currents. This condition represents a startup boundary constraint. Record datum uncertainties and local correction separately.

Stage the emitted native restart with hotstart, TS/UV `set values` and turbulence `default`. Use the exact bound absolute pressure file. Cartesian pressure head uses gravity 9.81 m/s², 101325 Pa and 1025 kg/m³. Default initialization supplies unavailable turbulence. Horizontal closure diagnostics start at zero. The manifest, donor mapping, density-reference JSON and native file are hash-bound. A native reader run verifies actual binary and namelist behavior.

## Initialize a run

```text
python scripts/lake_initial.py --source snapshot.json --reference accepted_grid_source.nc --grid-contract lsofs_fvcom_dat_manifest.json --pressure surface_pressure.nc --output state_20250701
python scripts/test_initialization.py
```

Use a verified `great_lakes_ofs_snapshot_v1` fields receipt, an independent frozen native grid analysis, and a `fvcom_retained_spherical_grid_v1` contract with its grid, depth, sigma, nodal Coriolis-latitude, and empty sponge files. The adapter verifies cell/node ordering, native triangle orientation, the full sigma axes, longitude wrapping, depth and all required state values. Before dynamic initialization, the hotstart reader requires static `cor(nele)` and `cc_sponge(nele)`. Reproduce the native cold-grid Coriolis conversion from the three nodal latitudes and use the verified zero-sponge closed-lake policy. The adapter uses the exact archived record and the exact UTC epoch of the run's absolute nodal air pressure in Pa, bound by ordered geographic coordinates or the runtime grid SHA-256. For a reader test before driving pressure is available, `--probe-use-analysis-pressure` emits a distinct `reader_probe` receipt.

Stage `lake_initial.nc` using `STARTUP_TYPE='hotstart'` and `STARTUP_TS_TYPE`, `STARTUP_UV_TYPE`, and `STARTUP_TURB_TYPE` all set to `'set values'`. Each independent analysis starts a new local step count of zero. Original native east/north layer currents, sigma vertical velocity, temperature, zero salinity, turbulence, wet flags, and zeta retain their original axes. Derive depth-averaged currents by integrating native layers with physical thickness. Initialize split-mode et from the instantaneous analyzed zeta. Read the receipt for sample min/max, deepwater temperature and vertical-shear checks.

Compute native DENS1 density anomaly with the source's Fofonoff–Millard pressure-dependent equation and spherical gravity. The static reference is the common wet-node mean profile on 601 fixed physical heights remapped to each native layer. The pressure head uses `-(air_pressure-101325)/(1025*g(latitude))` in metres. The receipt records source, ordered grid, forcing, native result and reference-profile hashes. Successful physical execution requires the separate scheduler log and returned model fields.

## Continue a native checkpoint

```text
python scripts/restart_handoff.py --restart native_restart.nc --initial-manifest state_20250701/initialization_manifest.json --time 2025-07-02T00:00:00Z --executable-sha256 SHA256 --source-job-id JOB_ID --checkpoint-sha256 SOURCE_JOB_CHECKSUM --output continuation.json
```

Obtain the checksum and executable binding from the job that created the checkpoint. Carry the complete native file, including its positive iteration, split-mode state and turbulence, into the continuation. Verify the first returned continuation timestamp and compare common fields to the uninterrupted segment with the same executable, ranks, timestep and forcings.
