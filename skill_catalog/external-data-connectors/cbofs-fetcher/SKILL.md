---
name: cbofs-fetcher
description: Plan, inventory, anonymously download, concatenate, extract, and health-check NOAA Chesapeake Bay Operational Forecast System (CBOFS) ROMS NetCDF data from operational AWS with bounded NCEI long-term fallback. Use when Codex needs bounded CBOFS native fields, practical salinity, earth-relative surface/bottom/depth-average currents, station or regular-grid passthrough files, exact storage estimates, resumable source-isolated caches, or provenance-preserving ROMS QA artifacts.
---

# CBOFS Fetcher

Use the bundled CLI for deterministic NOAA operational-AWS access with bounded
NCEI long-term fallback. Require no AWS
credentials, AWS CLI, boto3, or s3fs. Read
[`references/source_contract.md`](references/source_contract.md) before
changing archive selection, valid-time rules, or ROMS calculations. Validate
requests against [`references/request.schema.json`](references/request.schema.json).

## Workflow

1. Write a bounded `cbofs_request_v2` JSON request with an inclusive start and
   exclusive end. Choose `source_policy=aws_then_ncei` (default), `aws_only`,
   or `ncei_only`. Version 1 requests remain accepted and are explicitly
   migrated to v2.
2. Run `inventory` when source availability or variable packaging is uncertain.
3. Always run `plan` and review `download_estimate.json` before transferring.
   Download locally only when its routing decision is `local`; otherwise stage
   on Kestrel.
4. Run `fetch` only from the reviewed plan. Keep the raw cache unless the
   request explicitly uses `delete_after_extract`.
5. Run `inspect` on a representative object before choosing optional source
   variables.
6. For native `fields`, run `extract`, then run the health checker. Treat
   `stations` and `regulargrid` as raw passthrough products.
   `delete_after_extract` is carried out only after this health gate succeeds;
   the checker writes `cache_cleanup.json` before reporting completion.
7. Preserve the estimate, fetch manifest, compact health JSON, and final health
   report with downstream products.

Use the Python environment that provides `requests`, `numpy`, and `netCDF4`:

```powershell
$python = "<python-with-netcdf4>"
$cli = "<skill>/scripts/cbofs_fetcher.py"

& $python $cli plan --request request.json --run-dir run
& $python $cli fetch --plan run/download_estimate.json --run-dir run
& $python $cli inspect --input run/cache/raw/path/file.nc --output run/inspection.json
& $python $cli extract --request request.json --run-dir run --output run/cbofs_fields.nc
& $python "<skill>/scripts/check_download_health.py" `
  --request request.json --run-dir run --compact run/cbofs_fields.nc
```

For a forecast, specify one `run_cycle_utc` at 00, 06, 12, or 18 UTC. Do not
provide it for a nowcast. Source names are discovered dynamically; common
aliases `salinity` and `temperature` normalize to `salt` and `temp`. Request U
and V together.

## Interpretation guardrails

- Treat downloaded `ocean_time` as authoritative. Filename times are planning
  hints only.
- Treat multipart ETags as opaque source versions, never MD5 checksums.
- Resolve surface and bottom from sigma values; never assume an array order.
- Derive depth averages from W-level thickness including the record's `zeta`.
- Reduce native U/V first, wet-aware destagger them to rho points, rotate with
  `angle`, and then calculate speed.
- Rotate only when `angle` is a finite rho-grid field whose units explicitly
  say radians and whose standard/long-name metadata establishes the ROMS
  XI-axis-from-east convention. Reject degrees, ambiguity, and metadata drift.
- Never describe native U/V as earth-relative or silently clip plausibility
  excursions.
- Reject missing angle/vertical metadata, unpaired vectors, grid drift, or
  finite wet coverage below 95 percent.

Legacy `cbofs-canal` callable names remain as deprecated wrappers in
`scripts/cbofs_fetcher.py`. Use them only to keep old notebooks running; use
the request workflow for new work.

## Outputs

`plan` writes exact source identities, keys, byte counts, gaps, cross-archive
duplicates, capability and fallback decisions, free-space
routing, ETags, and timestamps. `fetch` writes resumable-transfer and cache-hit
provenance with SHA-256 hashes. `extract` writes `roms_compact_fields_v1`,
including native grids and metadata plus requested salinity and earth-relative
velocity views. Extraction through `--run-dir` or `--manifest` verifies the
reviewed plan, request, outcomes, raw bytes, and every `.download.json` sidecar,
then embeds that exact fetch binding in its report and compact file. Explicit
`--input` remains useful for ad hoc work but cannot pass full-run health. The
health checker fails closed without matching plan, manifest, sidecar,
size/SHA-256/ETag, angle, and compact-product provenance. Keep downloads and
generated evidence outside the skill tree.
