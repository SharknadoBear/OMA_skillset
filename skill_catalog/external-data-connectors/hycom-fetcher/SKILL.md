---
name: hycom-fetcher
description: Inventory, estimate, fetch, resume, point-sample, and health-check arbitrary native HYCOM variables from bounded time, depth, and spatial requests. Use when Codex needs model-neutral HYCOM subsets with live schema discovery, a mandatory timed/storage gate, and automatic HTML monitoring for downloads potentially lasting ten minutes or longer.
---

# HYCOM Fetcher

Use this skill only for native HYCOM data acquisition and optional generic point sampling. Keep FVCOM boundaries, sigma layers, forcing formats, and time reconstruction downstream.

## Required workflow

1. Verify `numpy`, `xarray`, and `netCDF4` are importable. If not, notify Bear and provide environment-specific installation guidance.
2. Inventory the selected alias or public OPeNDAP URL before choosing variables:

```powershell
python scripts/hycom_fetcher.py inventory --source gofs-latest --output runs/case/inventory.json
```

3. Read [references/request_contract.md](references/request_contract.md), write a bounded request JSON, and estimate before every download:

```powershell
python scripts/hycom_fetcher.py estimate --request request.json --run-dir runs/case --output runs/case/download_plan.json
```

4. Do not fetch when the plan is blocked. Use Kestrel when local free space is not greater than four times the estimated request size.
5. Execute the hash-bound plan. A conservative estimate of at least 600 seconds automatically creates and opens the localhost waitbar:

```powershell
python scripts/hycom_fetcher.py fetch --plan runs/case/download_plan.json --run-dir runs/case --output runs/case/subset.nc
```

6. Inspect `health_check.json` and the native output before downstream use. Treat all-missing requested variables, empty dimensions, or nonmonotonic time as failures; ordinary source land masks may remain missing.

## Rules

- Discover variables and coordinate roles from the selected source; never restrict work to a hard-coded SSH/T/S/U/V list.
- Preserve source coordinates, masks, units, attributes, and native layouts. Do not silently convert physical variables.
- Keep every request bounded in time and space. Do not bulk-download a collection without explicit approval.
- Use request-hash-isolated chunks and retain failed-run chunks for diagnosis/resume. Publish final NetCDF atomically.
- Never put credentials, URL query strings, personal paths, or data values in plans, status JSON, monitor pages, logs, or examples.
- Use the official HYCOM THREDDS catalogs as source truth; inventory again when using the evolving `gofs-latest` alias.

## Packaged tools

- `scripts/hycom_fetcher.py`: `inventory`, `estimate`, `fetch`, and `health` CLI plus Python API.
- `scripts/download_monitor.py`: atomic status protocol and loopback HTML monitor.
- `scripts/estimate_data_request.py` and `scripts/check_download_health.py`: standard connector-hook entry points.
- `scripts/selftest_hycom_fetcher.py`: offline contract and regression tests.
