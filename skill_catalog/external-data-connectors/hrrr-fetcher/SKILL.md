---
name: hrrr-fetcher
description: Inventory, estimate, range-download, resume, subset, decode, and health-check NOAA High-Resolution Rapid Refresh (HRRR) analysis and forecast GRIB2 fields from redundant AWS, Google Cloud, Azure, and NOMADS sources. Use when Codex needs bounded CONUS or Alaska HRRR surface, pressure-level, native-level, subhourly, smoke, wind, precipitation, radiation, heat-flux, pressure, temperature, humidity, visibility, cloud, or reflectivity data.
---

# HRRR Fetcher

Acquire native HRRR fields without regridding. Keep interpolation and model-forcing packaging downstream.

## Required workflow

1. Read [request_contract.md](references/request_contract.md) before authoring a request. Read [product_catalog.md](references/product_catalog.md) when choosing families, fields, levels, or forecast periods.
2. Run `python scripts/check_grib_runtime.py --output runtime.json`. If it fails, create the isolated environment pinned in [requirements-grib.txt](references/requirements-grib.txt).
3. Write a bounded `hrrr_request_v1`. Default to strict `analysis`; use `forecast` only when forecast cycles or subhourly periods are explicitly required.
4. Inventory and estimate before transfer:

```powershell
python scripts/hrrr_fetcher.py inventory --request request.json --output inventory.json
python scripts/hrrr_fetcher.py estimate --request request.json --run-dir runs/case --output runs/case/download_plan.json
```

5. Do not execute a blocked plan. Use `snapshot` for one analysis time or `run` for a bounded request:

```powershell
python scripts/hrrr_fetcher.py snapshot --request request.json --run-dir runs/snapshot
python scripts/hrrr_fetcher.py run --request request.json --run-dir runs/case
```

6. Require `health_check.json` to pass before downstream use.

## Source and scientific rules

- Use AWS NODD first, then Google Cloud, Azure, and NOMADS. Lock every complete GRIB object to one provider; never splice one object across mirrors.
- Never replace a missing analysis with a forecast. Missing objects try every provider, then follow the explicit missing policy.
- Treat live `.idx` inventories and decoded ecCodes metadata as authoritative. Field content changes across HRRR versions.
- Pair grid-relative U/V before rotating canonical wind aliases to earth-relative components. Raw U/V selectors remain native.
- Preserve units, levels, intervals, step type, product-definition template, grid flags, source bytes, URLs, and hashes. Do not silently change flux signs or accumulated quantities.
- Keep CONUS and Alaska on their native projected grids. Never concatenate domains or hourly and subhourly cadences into one file.
- Do not store credentials, URL queries, or personal absolute paths in portable requests, examples, or committed evidence.

## Packaged resources

- `scripts/hrrr_fetcher.py`: CLI and public Python helpers.
- `scripts/hrrr_core.py`: request normalization, source probing, `.idx` selection, ranged transfer, decoding, subsetting, writing, and health logic.
- `scripts/download_monitor.py`: atomic JSON status with Windows/OneDrive rename retry.
- [request_contract.md](references/request_contract.md): schemas, routing, outputs, gates, and failure behavior.
- [product_catalog.md](references/product_catalog.md): providers, filenames, grids, product families, aliases, and coverage.
