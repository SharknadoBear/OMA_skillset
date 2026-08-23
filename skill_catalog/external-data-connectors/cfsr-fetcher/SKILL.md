---
name: cfsr-fetcher
description: Inventory, estimate, snapshot, download, resume, decode, route, and health-check bounded NCEP Climate Forecast System Reanalysis atmospheric fields with NCEI full-resolution GRIB2 as the primary source and a whole-request HYCOM fallback. Use when Codex needs hourly CFSR wind, pressure, temperature, humidity, precipitation, heat-flux, stress, or surface-temperature fields from 1979-01-01 through 2011-03-31, including automatic CFSv2 era handoff and cross-era manifests.
---

# CFSR Fetcher

Acquire native reanalysis fields. Keep interpolation and forcing-file packaging downstream.

## Required workflow

1. Run `python scripts/check_grib_runtime.py --output runtime.json`. If it fails, use an isolated environment with the pinned packages described in [request_contract.md](references/request_contract.md).
2. Write a bounded `cfs_atmospheric_request_v2`. Use exact UTC hours, 0-360 longitude, and only required products.
3. Inventory and estimate before transfer:

```powershell
python scripts/cfsr_fetcher.py inventory --request request.json --output inventory.json
python scripts/cfsr_fetcher.py estimate --request request.json --run-dir runs/case --output runs/case/download_plan.json
```

4. Do not execute a blocked plan. Require free space greater than four times the planned raw transfer.
5. Use `snapshot` for one exact hour or `run` for routed acquisition. Inspect `health_check.json` before downstream use:

```powershell
python scripts/cfsr_fetcher.py snapshot --request request.json --run-dir runs/snapshot
python scripts/cfsr_fetcher.py run --request request.json --run-dir runs/case
```

## Source and routing rules

- Prefer full-resolution NCEI CFS Reanalysis time-series GRIB2. Reject `.l.` products and verify requested months against the live catalog.
- Route dates from 2011-04-01 onward to the adjacent `$cfsv2-fetcher`. Split a crossing request into two native-grid files plus `cfs_family_routing_manifest_v1`; never concatenate different grids.
- Lock one provider across all products in an era segment. Use HYCOM only when every product has a scientifically exact mapping; never mix providers.
- Preserve GRIB level, units, interval/PDT, forecast lead, source filename, and checksums. Do not silently change flux signs.
- Preserve the historical surface-pressure Python functions and v1 plan reader. NCEI pressure is absolute Pa; HYCOM `airprs` is explicitly converted from its 1000 hPa departure.
- Never store credentials, URL queries, or personal paths in portable requests or examples.

## Packaged resources

- `scripts/cfs_grib_core.py`: shared NCEI/HYCOM v2 acquisition, routing, canonical writing, and health logic.
- `scripts/cfsr_fetcher.py`: CFSR entry point and legacy compatibility API.
- `scripts/download_monitor.py`: atomic status and loopback waitbar, including Windows/OneDrive retry.
- [references/request_contract.md](references/request_contract.md): schemas, products, era behavior, and acceptance rules.
