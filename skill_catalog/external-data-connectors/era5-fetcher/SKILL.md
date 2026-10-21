---
name: era5-fetcher
description: Plan, download, resume, subset, and health-check hourly ERA5 10 m wind and mean sea-level pressure through CDS. Use for bounded geographic or ADCIRC-mesh requests and multi-year monthly acquisition; keep model interpolation and forcing packaging downstream.
---

# ERA5 Fetcher

Acquire the standard CDS 0.25-degree atmospheric grid with server-side regional trimming. Version 1 supports `wind_10m` (paired earth-relative components) and `mean_sea_level_pressure`. Preserve m/s and Pa; model interpolation, wind stress, and forcing-file conversion are downstream.

## Workflow

1. Read [request_contract.md](references/request_contract.md). Prepare a separate `.venv-era5` using Python 3.11.9 and [requirements-lock.txt](references/requirements-lock.txt), the complete locally tested runtime. [requirements.txt](references/requirements.txt) lists the principal dependencies. Then run `check-runtime`.
2. Write an `era5_request_v1` with hourly UTC endpoints, **start inclusive and end exclusive**, and either a geographic bbox or an EPSG:4326 ADCIRC fort.14. Mesh paths resolve relative to the request JSON. Default to one source-cell halo.
3. Run `plan`. Inspect the area, requested hours, monthly chunks, estimated storage and free-space gate. This is a requested-time inventory, not proof of archive availability.
4. Run `snapshot` for the first hour in a separate run directory, then `run` for the bounded period. Require `health_check.json` to pass before using output.
5. For multiple years, retain monthly partitions. `assemble-year` streams a verified full calendar year into a separate NetCDF. Do not acquire additional years merely because they were included in a planning test.

```powershell
python scripts/era5_fetcher.py check-runtime
python scripts/era5_fetcher.py plan --request request.json --run-dir runs/case
python scripts/era5_fetcher.py snapshot --request request.json --run-dir runs/smoke
python scripts/era5_fetcher.py run --request request.json --run-dir runs/case
python scripts/era5_fetcher.py health --run-dir runs/case
```

## Access and scientific rules

- Use `https://cds.climate.copernicus.eu/api` and dataset `reanalysis-era5-single-levels`. Credentials stay in `.cdsapirc` or CDSAPI environment variables. Inspect readiness only; never print tokens, passwords, or signed result URLs. Do not overwrite an EWDS/ADS config to configure CDS.
- The user must accept the dataset's required licences in CDS before live acquisition. `check-runtime --access` checks authentication and compares required licence IDs/revisions with accepted licences. See [official setup](https://cds.climate.copernicus.eu/en/how-to-api).
- Use modern `data_format=netcdf`, `download_format=zip`, `product_type=reanalysis`, and an explicit area. The CDS regular grid is already interpolated from the IFS model grid; do not call it the native IFS grid.
- Verified completed chunks are reused; interrupted current chunks are reacquired. In-flight CDS jobs are not recovered. Authentication/licence failures stop immediately; transient acquisition failures get bounded retries. Request-size failures bisect the affected time chunk.
- Preserve source responses, API requests, hashes, versions and available expver metadata. Reject simultaneous conflicting experiment branches instead of silently blending ERA5 and ERA5T. Missing version metadata means origin is unknown, not verified final ERA5.
- Keep source cells over both land and sea throughout the rectangular interpolation support; do not fill data gaps. Crossing the dateline is unsupported in v1: use two explicit requests.
- Store downloads and evidence in the project, outside this skill. `--max-chunks 1` stops after one newly completed chunk for a controlled restart test.

## Resources

- `scripts/era5_fetcher.py`: CLI and public Python helper exports.
- `scripts/era5_core.py`: planning, acquisition, atomic resume, normalization, health, annual assembly.
- `scripts/selftest_era5.py`: isolated offline behavioral tests with a simulated transport; these do not count as live download evidence.
