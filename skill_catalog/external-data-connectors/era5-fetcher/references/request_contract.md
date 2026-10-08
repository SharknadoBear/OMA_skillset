# ERA5 request and output contract

## Inputs

```json
{
  "schema_version": "era5_request_v1",
  "start": "2025-08-30T00:00:00Z",
  "end": "2025-09-03T00:00:00Z",
  "products": ["wind_10m", "mean_sea_level_pressure"],
  "mesh": {"path": "fort.14", "crs": "EPSG:4326"},
  "halo_cells": 1
}
```

Use `bbox: [west, south, east, north]` instead of `mesh` for an explicit geographic rectangle. Exactly one spatial input is required. Start/end are whole UTC hours, start inclusive and end exclusive. Product aliases expand to `10m_u_component_of_wind`, `10m_v_component_of_wind`, and `mean_sea_level_pressure`. Unknown or repeated products/options fail. No additional ERA5 fields are implemented in v1.

Read every fort.14 node, require ordered IDs 1..NP and declared `crs=EPSG:4326`, retain the source hash, and resolve paths against the request file. Normalize 0..360 or -180..180 longitudes; reject a dateline crossing with guidance to request two regions. Halo is a nonnegative integer, default 1. Expand by halo*0.25 degrees and round outward to the CDS lattice. Require at least two points on each axis. CRS conversion and regridding are outside v1.

## Planning and restart

The plan includes a normalized request, mesh hash/bounds where applicable, request hash, CDS area in north/west/south/east order, exact requested hour count, monthly chunks and their exact API selections, lattice shape, approximate uncompressed float32 bytes, conservative total storage allowance and free-space gate. It does not claim the server currently has every hour. Plans have a deterministic identity hash excluding observational fields such as free space and creation time.

Split at calendar-month boundaries and at partial first/last days to avoid unintended hours from CDS's day/time Cartesian selection. A complete 2011–2025 request has 180 chunks and 131,496 hours. Execution submits one chunk at a time. CDS request-size errors bisect the time interval and record child chunks. One-hour indivisible size failures stop.

Run directories bind to an immutable request. Changing dates, products, mesh bytes or bounds requires a new directory. Verified completed chunks are skipped; corrupt committed chunks fail visibly. Uncommitted partial transfers are replaced. This is chunk-level restart, not guaranteed byte resume or recovery of already queued CDS jobs. Three attempts maximum for transient network/server errors with short bounded backoff; authentication, licence, invalid request and QC errors stop immediately. CDS queue latency is service-controlled and no completion time is promised.

A PID/host lock blocks concurrent writers; dead same-host locks can be reclaimed. Foreign-host locks are not automatically deleted. Atomic replaces retry transient OneDrive/Windows sharing failures. Reports contain sanitized status messages, never credentials or signed download URLs.

## Outputs and validation

- `request.json`, `download_plan.json`, `status.json`, `run_manifest.json`, `health_check.json`.
- `requests/<chunk>.json`: actual CDS dataset and payload selection.
- `raw/<chunk>.payload`: original ZIP/NetCDF response and its checksum.
- `fields/<chunk>.nc`: CF-1.10 `era5_fields_v1` on `(time, latitude, longitude)`, all coordinates increasing; selected `u10`, `v10`, `msl` preserve source values and units.

Identify responses by ZIP/NetCDF signatures, not extension. Reject HTML/error pages. Extract ZIP members only to controlled filenames; reject unsafe paths, unbounded expansion and non-NetCDF contents. Merge separate variable files only on identical time/space coordinates and agreeing shared variables. Never merge conflicting experiment branches.

Scientific gates: every requested hour exactly once, all selected fields, finite values throughout the rectangle, paired wind, m/s for wind and Pa for pressure, positive pressure, exact 0.25-degree support, and complete mesh coverage. Preserve source attributes and available version metadata; no pressure conversion, missing-data filling, interpolation or bias adjustment. Health independently reopens committed files and checks schemas, clocks, units, hashes, support coordinates, raw payload presence and manifest/plan identity. A partial run cannot pass the complete-period health gate.

`assemble-year --run-dir ... --year YYYY --output ...` requires a complete calendar year within a fully healthy run. It streams one chunk at a time, retains a source manifest, verifies the annual clock and output fields, and refuses to overwrite existing output.

## Callable Python API

Add the skill's `scripts` directory to `sys.path`, then import `era5_fetcher`. Public helpers are `check_runtime(access=False)`, `normalize_request(request, base_dir)`, `build_plan(request, base_dir, run_dir)`, `save_plan(request_path, run_dir)`, `snapshot(request_path, run_dir)`, `run(run_dir, max_chunks=None)`, `health(run_dir)` and `assemble_year(run_dir, year, output)`. `run` uses a previously saved plan; `health` returns `pass_all` and explicit errors. Injected test transports in the core API are for isolated offline evaluation only, not alternate providers.

## References

- [CDS API setup](https://cds.climate.copernicus.eu/en/how-to-api)
- [ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
- [ERA5 documentation](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation)
- [Monthly request efficiency](https://confluence.ecmwf.int/spaces/CKB/pages/174856258/Climate+Data+Store+CDS+documentation)
- [NetCDF converter](https://confluence.ecmwf.int/spaces/CKB/pages/450760890/GRIB+to+netCDF+conversion+on+new+CDS+and+ADS+systems)
