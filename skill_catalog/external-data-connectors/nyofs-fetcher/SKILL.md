---
name: nyofs-fetcher
description: Plan, inventory, anonymously download, concatenate, extract, and health-check NOAA New York and New Jersey Operational Forecast System (NYOFS) POM NetCDF data from operational AWS with bounded NCEI long-term fallback. Use when Codex needs coarse or fine NYOFS fields, historical coarse coverage, stations, exact storage estimates, resumable caches, vertical views, or provenance-preserving QA.
---

# NYOFS Fetcher

Use this skill as the source-specific connector for NOAA NYOFS data. Prefer operational AWS and use NCEI only for uncovered, supported semantic records. Keep requests time-bounded, estimate before fetching, and preserve machine-readable evidence.

## Follow the workflow

1. Create a `nyofs_request_v2` JSON request. Read [references/request.schema.json](references/request.schema.json) for the exact contract. Use `source_policy: aws_then_ncei` (default), `aws_only`, or `ncei_only`. V1 requests migrate to v2 with recorded provenance.
2. Read [references/source_contract.md](references/source_contract.md) before interpreting filenames, aggregate-cycle coverage, POM grids, sigma layers, masks, or source variables.
3. Inventory and estimate before downloading:

```bash
python scripts/nyofs_fetcher.py inventory --request request.json --run-dir runs/case
python scripts/estimate_data_request.py --request request.json --run-dir runs/case --output runs/case/download_estimate.json
```

4. Review `download_estimate.json`. Download locally only when `local_free_bytes > 4 * total_bytes`. If the gate fails, recommend `$kestrel-hpc` and stage under `/scratch/yhuang168/oma_external_data_connectors/nyofs-fetcher/<run-id>`. Stop when the estimate is incomplete.
5. Fetch, inspect actual NetCDF metadata, and extract fields:

```bash
python scripts/nyofs_fetcher.py fetch --plan runs/case/download_estimate.json --run-dir runs/case
python scripts/nyofs_fetcher.py inspect --request request.json --run-dir runs/case
python scripts/nyofs_fetcher.py extract --request request.json --run-dir runs/case
```

6. Run the finishing gate and inspect critical findings:

```bash
python scripts/check_download_health.py --request request.json --run-dir runs/case --output runs/case/health_check.json --plots-dir runs/case/health_plots
```

## Use each command deliberately

- Use `inventory` to discover matching objects in both current daily and legacy monthly layouts. Ambiguous historical NCEI coarse-field names require a bounded HTTP Range NetCDF time-coordinate probe after product/grid/window prefiltering; a server may return the complete small aggregate despite the Range request. Review `source_discovery.ncei.coverage_probe.object_count` and `.bytes` as discovery transfer evidence.
- Use `plan` through `nyofs_fetcher.py`, or the compatible `estimate_data_request.py` hook, to select cycle aggregates, calculate exact bytes, flag nominal time gaps, apply the four-times-free-space gate, and write `download_estimate.json`.
- Use `fetch` only after plan review. Preserve resumable `.part` files, validated cache hits, SHA-256 hashes, source ETags, and transfer outcomes in `fetch_manifest.json`.
- Use `inspect` to report actual variables, dimensions, time coordinates, mask values, grid geometry, and sigma ordering. Treat downloaded time coordinates as authoritative.
- Use `extract` only for `product: fields`. Crop by normalized verified time, reject geometry/schema drift, and write one compressed compact NetCDF per requested grid.
- If a long request crosses a model/grid era and triggers geometry, dimensions, masks, sigma, vector, or variable-schema drift, split it at the era boundary; never coerce incompatible records into one product.

Run `python scripts/nyofs_fetcher.py <command> --help` before using unfamiliar options.

## Respect NYOFS/POM boundaries

- Fully process `fields`. Treat `stations` as inventory, estimate, download, inspection, and passthrough QA only; reject `variables` and `vertical_views` for stations.
- Keep coarse `nyofs` and fine `nyofs_fg` grids separate. Never mosaic or interpolate between them.
- Require `run_cycle_utc` for forecasts and reject it for nowcasts. NYOFS cycles are 05, 11, 17, and 23 UTC.
- Normalize hourly fields and six-minute station timestamps only when a decoded value is within 60 seconds of its nearest nominal timestamp. Preserve original time and the adjustment.
- Treat the sigma value nearest zero as surface, the farthest as bottom, and `near_surface` as the second-nearest layer. Compute depth averages with trapezoidal sigma-point weights, renormalized over finite wet layers.
- Average eastward `u` and northward `v` before calculating depth-averaged speed. Derive wind speed from `air_u` and `air_v`.
- Do not apply FVCOM topology, ROMS staggering, or vector rotation. Reject mismatched or explicitly grid-relative velocity components unless usable rotation metadata exists.
- Expect full source objects to be downloaded. Variable selection reduces extracted output, not AWS transfer size.

## Preserve evidence and safety

- Access operational AWS and NCEI `prod-model` anonymously over HTTPS/ListObjectsV2. Treat discovery errors as source failures, not missing coverage. Do not request or store credentials, tokens, passwords, or signed URLs.
- NCEI supports coarse NYOFS nowcast fields and station nowcast/forecast files. Never substitute coarse nowcast data for fine-grid or field-forecast coverage.
- Keep raw cache files by default. Honor `delete_after_extract` only after extraction and a passing health check.
- Preserve source keys/URLs, sizes, opaque ETags, last-modified times, grids, cycles, hashes, retries, cache decisions, normalized request content, and routing decisions.
- Fail on missing required timestamps under `missing_policy: error`; record skipped coverage explicitly under `missing_policy: skip`.
- Surface corrupt objects, invalid masks or geometry, schema drift, non-monotonic or duplicate times, all-NaN frames, unpaired vectors, inconsistent speeds, and finite wet coverage below 95 percent as critical findings. Keep broad physical-range checks as warnings; never silently clip.
- Keep downloads, compact NetCDFs, maps, movies, manifests, and logs in the run workspace, not in the skill directory.
