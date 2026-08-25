# CUDEM Bathymetry Dependency

Use `cudem-bathy` as the bathymetry connector. This skill should not embed CUDEM catalog, source-selection, or download-health logic.

Generated grid runs should call `scripts/fetch_bathy_sources.py`, not the CUDEM-only `fetch_cudem_bathy.py`, unless Bear explicitly requests CUDEM-only behavior.

Default generated-run policy:

- `--fallback-policy cudem-nbs-crm-etopo`
- `--resolution-policy source-priority`
- `--target-spacing-arcsec 1.0`

Expected connector output:

- NetCDF bathymetry file from `fetch_bathy_sources.py`, normally named `<name>_bathy_sources.nc`.
- Metadata JSON with `coverage_by_source`, `no_data_cells`, `finite_output_fraction`, and `outputs.netcdf`.
- Source-id diagnostic PNG and health-check JSON.

`fvcom-grid-generation` consumes the NetCDF and normalizes it to positive-down depth. If no bathymetry file is supplied, `run_fvcom_grid.py` calls the connector using the assembled `model_domain_polygon` buffered in projected coordinates by `--bathy-fetch-halo-m` (2 km by default). RegionBPoly is provenance and initial source-coverage intent, not the bathymetry fetch extent. Before any size-field or mesh work, require at least 95% finite sampled wet-domain coverage and 100% finite support along every densified delivered OBC.

Prefer the NetCDF variable `depth_m` when present. Treat `elevation_m` as positive-up and convert it to positive-down depth only when no depth variable is available.

For broad domains, a 1 arcsec fallback mosaic can contain tens of millions of cells. Keep the downloaded bathymetry product intact for final node-depth sampling, but downsample the in-memory bathymetry used for mesh-size-field and gradation operations with `--size-field-max-cells`.
