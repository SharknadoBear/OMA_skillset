---
name: dbofs-fetcher
description: Plan, inventory, anonymously download, concatenate, extract, and health-check NOAA Delaware Bay Operational Forecast System (DBOFS) ROMS NetCDF data from operational AWS with bounded NCEI long-term fallback. Use when Codex needs bounded DBOFS native staggered-grid fields, station or regular-grid passthrough files, exact byte and storage estimates, resumable source-isolated caches, practical salinity, earth-relative current components, sigma-layer views, depth averages, or provenance-preserving QA artifacts.
---

# DBOFS Fetcher

Use the packaged scripts as a bounded, estimate-first connector for NOAA's
operational AWS and NCEI long-term archives. Native `fields` are fully
processed. Treat
`stations` and `regulargrid` as inspected raw passthrough products.

## Required workflow

1. Write a `dbofs_request_v2` request. Choose `source_policy=aws_then_ncei`
   (default), `aws_only`, or `ncei_only`. Version 1 requests remain accepted
   and are explicitly migrated to v2. Read
   [references/request.schema.json](references/request.schema.json) for the exact
   contract and [references/source_contract.md](references/source_contract.md)
   when archive layout or ROMS conventions matter.
2. Inventory or plan before downloading:

```bash
python scripts/dbofs_fetcher.py inventory --request request.json --output inventory.json
python scripts/dbofs_fetcher.py plan --request request.json --run-dir runs/case
```

3. Review `download_estimate.json`. Download locally only when its routing
   decision is `local`; otherwise stage the request with `$kestrel-hpc`.
4. Fetch, inspect, and extract:

```bash
python scripts/dbofs_fetcher.py fetch --plan runs/case/download_estimate.json --run-dir runs/case
python scripts/dbofs_fetcher.py inspect --input runs/case/cache/raw/*.nc --output inspection.json
python scripts/dbofs_fetcher.py extract --request request.json --run-dir runs/case --output runs/case/dbofs_fields.nc
```

5. Run the finishing gate:

```bash
python scripts/check_download_health.py --request request.json --run-dir runs/case --output runs/case/health_check.json
```

## Processing rules

- Use anonymous HTTPS and S3 ListObjectsV2; do not require AWS credentials,
  `boto3`, the AWS CLI, or `s3fs`.
- Treat downloaded `ocean_time` as authoritative. Filename-derived valid times
  are planning hints: `n001=cycle-5h` through `n006=cycle`, while
  `f001=cycle+1h` onward.
- Preserve native C-grid geometry. Average U and V on their own staggered grids,
  destagger to rho points, rotate through `angle`, and only then calculate
  current speed.
- Resolve surface and bottom from sigma values. Read `Vtransform` from each file,
  include `zeta`, and calculate depth-average weights from W-level thicknesses.
- Reject geometry, vector-pair, angle, or vertical-metadata drift rather than
  silently coercing it.
- Require native binary rho/U/V masks, explicit radian XI-axis angle metadata,
  strict finite monotone vertical coordinates, and stable requested-variable
  dimension/grid signatures across every source file.
- Keep raw files by default. A rerun must use validated cache hits.
- NCEI supports native nowcast fields and station nowcast/forecast only. Field
  forecasts and regular-grid requests remain AWS-only and fail closed under
  `ncei_only`.
- Do not pass field extraction options for station or regular-grid requests.

## Outputs

Planning writes `download_estimate.json`; fetching accepts only that reviewed
plan and writes a plan-hash-bound `fetch_manifest.json`; field extraction writes
a compressed `roms_compact_fields_v1` NetCDF plus an extraction manifest;
passthrough health writes a cropped time-selection artifact. Preserve these
artifacts together as provenance.
Before any reviewed transfer, revalidate exact NOAA key/URL/ETag/Last-Modified
scope and current free space greater than four times the recomputed exact total.

## Legacy compatibility

`scripts/dbofs_fetcher.py` retains deprecated wrappers for the old
`dbofs-boundary` callables. They emit `DeprecationWarning`; use the request/CLI
workflow for new work. Legacy `fhour=0` maps explicitly to the cycle-time
nowcast (`n006`) and never falls through to an unrelated `f001` object.

## Constraints

- Keep downloads request-bounded; do not bulk mirror the archive.
- Do not claim native U/V are earth-relative before rotation.
- Do not silently clip plausible-range excursions.
- Do not put credentials, raw downloads, plots, or run evidence inside the
  skill package.
