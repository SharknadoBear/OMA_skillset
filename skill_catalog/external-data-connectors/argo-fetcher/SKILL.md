---
name: argo-fetcher
description: Inventory, estimate, download, resume, and health-check bounded native Argo GDAC core, BGC B-profile, and synthetic S-profile NetCDF files. Use when Codex needs Argo observations for an explicit time range and bbox, GeoJSON polygon, SMS 2DM wet mesh, global domain, WMO list, DAC list, or BGC parameter selection, with estimate-first storage gating, revision-aware caching, native QC summaries, and Coriolis-to-anonymous-S3 fallback.
---

# Argo Fetcher

Acquire native Argo profile files without transforming, interpolating, or scientifically filtering their observations.

## Required workflow

1. Verify `numpy`, `pandas`, `requests`, `xarray`, `netCDF4`, `matplotlib`, and `shapely` are importable.
2. Read [references/request_contract.md](references/request_contract.md), then create an explicit bounded request.
3. Inventory or refresh the required official GDAC indexes:

```powershell
python scripts/argo_fetcher.py inventory --products core synthetic bio --cache-dir runs/case/cache --output runs/case/inventory.json
```

4. Estimate before downloading. Review the selected count, conservative bytes, runtime, index freshness, and four-times-free-space gate:

```powershell
python scripts/argo_fetcher.py estimate --request request.json --run-dir runs/case --output runs/case/download_plan.json
```

5. Fetch only from the unexpired hash-bound plan. Replan if a selected GDAC row has changed:

```powershell
python scripts/argo_fetcher.py fetch --plan runs/case/download_plan.json --run-dir runs/case
```

6. Inspect `health_check.json`, `profile_inventory.csv`, and every QA plot before using the native files downstream. Treat a failed health status as unusable.

Use `run` for estimate-first orchestration. It stops at the plan unless `--execute` is supplied.

## Rules

- Require inclusive UTC `start` and `end` and exactly one spatial selector. Require explicit `global`; never infer an unbounded request.
- Use `core`, `synthetic`, and `bio` indexes only. Do not use the retired public greylist.
- Preserve the GDAC layout under `raw/dac/<dac>/<wmo>/profiles/` and preserve native NetCDF values, dimensions, attributes, QC flags, adjusted fields, and errors.
- Interpret `R` and `D` from filenames. Inspect `DATA_MODE` and `PARAMETER_DATA_MODE` inside NetCDF; never infer adjusted `A` mode from an index filename.
- Keep BGC `parameters` and `parameter_data_mode` positions aligned.
- Use Coriolis HTTPS first. Use anonymous Argo S3 only after an eligible primary transport failure and reject fallback content older than the selected index row.
- Do not reuse a file based only on existence. Require matching index revision, size, SHA-256, and a successful NetCDF-open check.
- Do not fetch when size cannot be credibly estimated or local free space is not greater than four times the conservative estimate. Route large work separately.
- Never put credentials, signed query strings, absolute personal paths, or observation values in plans, status files, monitor pages, or examples.
- Keep partial files as resumable evidence, but publish valid files atomically.

## Packaged tools

- `scripts/argo_fetcher.py`: `inventory`, `estimate`, `fetch`, `health`, and `run` CLI plus Python API.
- `scripts/estimate_data_request.py` and `scripts/check_download_health.py`: standard connector hooks.
- `scripts/download_monitor.py`: loopback-only progress page for runs estimated at ten minutes or longer.
- `scripts/selftest_argo_fetcher.py`: offline contract, transport, health, and frozen-Guam regression tests.
- `scripts/live_smoke_argo_fetcher.py`: bounded one-file-per-product live smoke test.
