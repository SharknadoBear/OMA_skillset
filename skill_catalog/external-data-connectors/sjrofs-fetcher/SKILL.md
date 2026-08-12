---
name: sjrofs-fetcher
description: Plan, inventory, anonymously download, concatenate, extract, and health-check NOAA St. Johns River Operational Forecast System (SJROFS) EFDC NetCDF data from operational AWS with bounded NCEI long-term fallback. Use for bounded SJROFS fields, practical salinity, earth-relative currents, stations, exact storage estimates, reviewed transfers, resumable caches, vertical views, or provenance-preserving QA.
---

# SJROFS Fetcher

Use this source-specific connector for NOAA SJROFS. AWS is primary; NCEI can fill only supported unresolved historical records. Keep every request bounded and keep evidence outside this package.

## Follow the reviewed-plan workflow

1. Create `sjrofs_request_v2` JSON using [references/request.schema.json](references/request.schema.json). Read [references/source_contract.md](references/source_contract.md) before interpreting EFDC masks, layers, times, or vectors.
2. Inventory and plan:

```bash
python scripts/sjrofs_fetcher.py inventory --request request.json --run-dir runs/case
python scripts/sjrofs_fetcher.py plan --request request.json --run-dir runs/case
```

3. Review `download_estimate.json`. Fetch accepts only that reviewed plan and only when current local free space still exceeds four times its exact bytes:

```bash
python scripts/sjrofs_fetcher.py fetch --plan runs/case/download_estimate.json --run-dir runs/case
python scripts/sjrofs_fetcher.py inspect --request request.json --run-dir runs/case
python scripts/sjrofs_fetcher.py extract --request request.json --run-dir runs/case
python scripts/check_download_health.py --request request.json --run-dir runs/case --output runs/case/health_check.json --plots-dir runs/case/health_plots
```

Stop on an incomplete estimate, source discovery error, changed remote identity, invalid plan binding, or failing health report. Use `$kestrel-hpc` when the local storage gate routes away from local download.

## Respect the EFDC contract

- Fully extract `fields`; keep `stations` passthrough-only and reject field variables/views for them. There is no SJROFS regular-grid or secondary-grid request.
- Forecasts require an exact 05/11/17/23 UTC `run_cycle_utc`; nowcasts reject it.
- Normalize fields to hourly `HH:30` and stations to six-minute cadence only within 60 seconds. Preserve raw decoded values and offsets.
- Treat exactly `mask == 5` as active water. Preserve the source mask, classify the known negative padding sentinel as inactive, and reject hydrodynamic `zeta/salt/u/v/temp` values outside active cells or ambiguous future active codes. Atmospheric `air_u/air_v` can legitimately cover land; mask them to active water in extracted/derived products.
- Treat positive-down sigma values as EFDC layer-top fractions. Require a first value near zero and unique values in `[0,1)`. Compute layer weights as `diff([sorted_sigma..., 1.0])`, map them back to source order, and renormalize only over finite layers.
- Select surface, near-surface, and bottom by sorted sigma value, independent of storage order. Average `u` and `v` independently before calculating depth-averaged speed.
- Require collocated `u/v` with eastward/northward standard-name metadata. Never destagger or rotate them.
- Download full source objects; variable selection only reduces the compact product.

## Preserve safety and evidence

- Use anonymous HTTPS only. Never request credentials or signed URLs.
- Treat AWS listing errors as failures, not missing coverage. NCEI supports verified field nowcasts and station nowcasts/forecasts, not field forecasts.
- Preserve plan/request hashes, selected objects, provider metadata, opaque ETags, Last-Modified, source-isolated paths, SHA-256, resumes/retries/cache hits, decoded time provenance, source keys, and health findings.
- Keep raw caches by default. `delete_after_extract` takes effect only after passing health.
- Fail under `missing_policy: error`; document partial coverage under `skip`.
- Split requests at incompatible model eras rather than coercing geometry, mask, sigma, variable, or vector conventions.
