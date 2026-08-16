---
name: cfsv2-fetcher
description: Inventory, estimate, download, resume, and health-check bounded NCEP CFSv2 atmospheric fields from HYCOM THREDDS/OPeNDAP. Use when Codex needs native CFSv2 wind, pressure, heat-flux, freshwater, or surface-temperature subsets with mandatory timed/storage planning and automatic HTML waitbar monitoring for downloads potentially lasting ten minutes or longer.
---

# CFSv2 Fetcher

Use this skill to acquire native atmospheric fields. Keep interpolation, derived fluxes, and FVCOM packaging downstream.

## Required workflow

1. Verify `numpy`, `xarray`, and `netCDF4` are importable. If not, notify Bear with environment-specific installation guidance.
2. Inventory the requested year and subdataset:

```powershell
python scripts/cfsv2_fetcher.py inventory --year 2019 --subdataset uv-10m --output runs/case/inventory.json
```

3. Read [references/request_contract.md](references/request_contract.md), write a bounded request, and run the mandatory estimate gate:

```powershell
python scripts/cfsv2_fetcher.py estimate --request request.json --run-dir runs/case --output runs/case/download_plan.json
```

4. Do not fetch a blocked plan. Use Kestrel when local free space is not greater than four times the selected data size.
5. Execute the plan. Runs conservatively estimated at 600 seconds or longer automatically open the loopback waitbar:

```powershell
python scripts/cfsv2_fetcher.py fetch --plan runs/case/download_plan.json --run-dir runs/case --output runs/case/wind.nc
```

6. Inspect `health_check.json` before downstream transformation.

## Source and compatibility rules

- Use official HYCOM CFSv2 yearly sources. Keep downloads bounded by UTC time, native bbox, subdataset, and variables.
- Use canonical subdataset `dlwsfc`; accept `dlwflx` only as its compatibility alias.
- Treat `airprs` as a departure from 1000 hPa. Use `cfsv2_airprs_to_absolute_pa` for explicit absolute-pressure conversion.
- Preserve `fetch_cfsv2_window`, `fetch_cfsv2_year`, `fetch_wind_year`, and `fetch_pressure_year`; they invoke the estimate gate automatically.
- Retain `window` as a deprecated CLI alias. Do not add model-specific regridding or forcing-file creation.
- Never place credentials, URL queries, personal paths, or field values in plans, monitor artifacts, or examples.

## Packaged tools

- `scripts/cfsv2_fetcher.py`: inventory, estimate, fetch, window, health, and pressure conversion.
- `scripts/download_monitor.py`: atomic status and localhost waitbar protocol.
- `scripts/estimate_data_request.py` and `scripts/check_download_health.py`: standard connector hooks.
- `scripts/selftest_cfsv2_fetcher.py`: offline compatibility and gate tests.
