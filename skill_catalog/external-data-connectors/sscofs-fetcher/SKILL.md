---
name: sscofs-fetcher
description: Plan, inventory, anonymously download, extract, and health-check NOAA Salish Sea and Columbia River Operational Forecast System (SSCOFS) NetCDF data from the public AWS archive. Use when Codex needs bounded SSCOFS native FVCOM fields, station or regular-grid passthrough files, exact byte and storage estimates, resumable cached transfers, sigma-layer surface/bottom/depth-average products, or provenance-preserving QA artifacts.
---

# SSCOFS Fetcher

Use this skill as the source-specific connector for NOAA SSCOFS public AWS data. Keep requests time-bounded, estimate before fetching, and preserve machine-readable evidence for every run.

## Follow the workflow

1. Create an `sscofs_request_v1` JSON request. Read [references/request.schema.json](references/request.schema.json) for the exact contract. Use UTC timestamps with an inclusive `start_utc` and exclusive `end_utc_exclusive`.
2. Read [references/source_contract.md](references/source_contract.md) when interpreting filenames, valid times, FVCOM topology, sigma layers, or source variables.
3. Inventory and plan before downloading:

```bash
python scripts/sscofs_fetcher.py inventory --request request.json --run-dir runs/case
python scripts/estimate_data_request.py --request request.json --run-dir runs/case --output runs/case/download_estimate.json
```

4. Review `download_estimate.json`. Download locally only when `local_free_bytes > 4 * total_bytes`. If that gate fails, recommend `$kestrel-hpc` and stage under `/scratch/yhuang168/oma_external_data_connectors/sscofs-fetcher/<run-id>`. If the estimate is incomplete, stop for review.
5. Start with a small smoke window, then fetch the approved request:

```bash
python scripts/sscofs_fetcher.py fetch --request request.json --run-dir runs/case
```

6. Inspect downloaded NetCDF metadata before extraction, then extract native fields when requested:

```bash
python scripts/sscofs_fetcher.py inspect --request request.json --run-dir runs/case
python scripts/sscofs_fetcher.py extract --request request.json --run-dir runs/case
```

7. Run the finishing gate and inspect critical findings:

```bash
python scripts/check_download_health.py --request request.json --run-dir runs/case --output runs/case/health_check.json --plots-dir runs/case/health_plots
```

## Use each command deliberately

- Use `inventory` to list matching public S3 objects and discover source coverage without downloading NetCDF payloads.
- Use `plan` through `sscofs_fetcher.py`, or the compatible `estimate_data_request.py` hook, to deduplicate valid times, detect gaps, calculate exact bytes, check local free space, and write `download_estimate.json`.
- Use `fetch` only after plan review. Preserve resumable `.part` files, validated cache hits, SHA-256 hashes, source ETags, and transfer outcomes in `fetch_manifest.json`.
- Use `inspect` to report actual variables, dimensions, time coordinates, mesh geometry, and sigma ordering from cached files. Do not rely on a closed variable list.
- Use `extract` only for `product: fields`. Concatenate by verified `Times`, reject geometry or schema drift, and write a compact compressed NetCDF containing requested fields, topology, masks, and derived vertical views.

Run `python scripts/sscofs_fetcher.py <command> --help` before using unfamiliar options.

## Respect product boundaries

- Fully process native unstructured `fields` files.
- Treat `stations` and `regulargrid` as inventory, estimate, and raw-download passthrough products. Do not pass `variables` or `vertical_views` for either product; reject those options instead of ignoring them.
- Require `run_cycle_utc` for forecast requests. Prefer nowcast fields for continuous historical time series and remove redundant `n000` records in favor of the preceding cycle's `n006` record.
- Treat `near_surface` as the second layer from the dynamically detected surface. Keep `surface`, `bottom`, explicit sigma indices, and thickness-weighted `depth_average` distinct.
- Expect full source objects to be downloaded. Variable selection reduces the extracted product, not the AWS transfer size.

## Preserve evidence and safety

- Access `noaa-nos-ofs-pds` anonymously over HTTPS/ListObjectsV2. Do not request or store AWS credentials, tokens, passwords, or signed URLs.
- Keep raw cache files by default. Honor `delete_after_extract` only after a successful extraction and health check.
- Keep source URLs/keys, sizes, ETags, last-modified times, parsed cycles and valid times, hashes, retries, cache decisions, request content, software version, and fallback decisions in run artifacts.
- Fail on missing required hours under `missing_policy: error`; record skipped hours explicitly under `missing_policy: skip`.
- Surface missing coverage, corrupt objects, invalid topology, non-monotonic or duplicate times, all-NaN frames, unpaired velocity components, and finite wet coverage below 95 percent as critical health failures. Report broad physical-range checks as warnings; never silently clip values.
- Keep downloads and generated NetCDF, maps, movies, manifests, and logs in the run workspace, not in the installed skill directory.
