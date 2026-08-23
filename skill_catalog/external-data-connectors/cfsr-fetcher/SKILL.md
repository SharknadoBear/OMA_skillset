---
name: cfsr-fetcher
description: Inventory, estimate, download, resume, decode, and health-check bounded NCEP Climate Forecast System Reanalysis (CFSR) atmospheric fields from NCEI with a whole-run HYCOM fallback. Use when Codex needs hourly historical CFSR surface pressure, an estimate-first storage gate, resumable monthly GRIB2 acquisition, provider-neutral NetCDF subsets, or an automatic HTML download waitbar.
---

# CFSR Fetcher

Acquire native CFSR fields. Keep FVCOM packaging and model-specific interpolation downstream.

## Required workflow

1. Require `numpy`, `netCDF4`, `requests`, `eccodes`, and a JPEG2000-capable `rasterio/GDAL` build. Use `xarray` only for HYCOM fallback.
2. Read [references/request_contract.md](references/request_contract.md), then inventory and estimate the bounded request:

```powershell
python scripts/cfsr_fetcher.py inventory --request request.json --output inventory.json
python scripts/cfsr_fetcher.py estimate --request request.json --run-dir runs/case --output runs/case/download_plan.json
```

3. Do not fetch a blocked plan. Require local free space greater than four times the planned transfer.
4. Fetch with resume. Runs estimated at ten minutes or longer automatically create and launch a loopback HTML waitbar:

```powershell
python scripts/cfsr_fetcher.py fetch --plan runs/case/download_plan.json --run-dir runs/case --resume
```

5. Inspect `health_check.json` before using the output.

## Source rules

- Prefer NCEI `pressfc.gdas.YYYYMM.grb2` from the CFS Reanalysis Time Series archive. Do not substitute `prmsl` or low-resolution `.l` products.
- Treat NCEI surface pressure as absolute pressure and validate its GRIB parameter, surface level, units, and plausible range before writing.
- Use HYCOM `cfsr-sec_{year}_01hr_sfcprs.nc/airprs` only as a whole-run fallback. Convert its 1000 hPa departure with `(airprs + 1000) * 100` Pa.
- Never mix providers in one output. Record the chosen provider in `source_provider_lock.json`.
- Preserve native time and spatial coordinates. Crop with the requested native-cell halo; do not interpolate, smooth, or resample.
- Retain raw downloads and checkpoints unless `--cleanup-raw` is explicitly requested after successful health validation.

## Packaged tools

- `scripts/cfsr_fetcher.py`: inventory, estimate, fetch, status, and health commands.
- `scripts/download_monitor.py`: atomic status and loopback waitbar support.
- `scripts/selftest_cfsr_fetcher.py`: offline gate, resume, normalization, and health tests.
- `references/request_contract.md`: request, output, provider, and monitoring schemas.
