---
name: hycom-obc-core
description: Use shared HYCOM helper routines for gridded ocean data access, point interpolation, z-to-layer remapping, and time conversion. Use when Codex needs a generic reusable Python toolbox core for HYCOM-derived workflows with estimate-first and health-check conventions.
---

# HYCOM OBC Core

Use this skill as a self-contained Python toolbox for external data access. Keep the skill focused on source-specific fetching, cache management, basic transforms, estimate-first planning, and downloaded-data quality gates. Keep model-specific products as downstream or legacy compatibility work.

## Source And Toolbox

- Primary source: Reusable HYCOM helper routines and grid/time interpolation utilities.
- Toolbox focus: shared HYCOM request utilities, coordinate conversion, interpolation helpers, and time conversion.
- Main packaged scripts:
- `scripts/hycom_fetcher.py`
- `scripts/grid_utils.py`
- Standard estimate hook: `scripts/estimate_data_request.py`.
- Standard finishing gate: `scripts/check_download_health.py`.

## Required Workflow

1. Inspect the request or manifest and identify bbox, points, time window, variables, sources, and requested output footprint.
2. Run the estimate hook before any live download:

```bash
python scripts/estimate_data_request.py --request request.json --run-dir runs/case --output runs/case/download_estimate.json
```

3. Use the estimate result to choose storage:
   - download locally only when `local_free_bytes > 4 * estimated_requested_bytes`;
   - if the local disk does not satisfy that rule, use `$kestrel-hpc` and plan work under `/scratch/yhuang168/oma_external_data_connectors/hycom-obc-core/<run-id>`;
   - if the estimate is unknown, do not download until the request is narrowed or explicitly reviewed.
4. Execute the source-specific toolbox script with a small smoke-test window before broader requests.
5. Preserve source URLs, selected files, variable names, coverage, CRS/datum/time metadata, cache paths, and any fallback decisions in run metadata.
6. Run the health gate after download:

```bash
python scripts/check_download_health.py --request request.json --run-dir runs/case --output runs/case/health_check.json --plots-dir runs/case/health_plots
```

7. Surface the health report to Bear only when important caveats exist, such as missing requested coverage, empty variables, all-NaN fields, finite coverage below 95 percent, obvious gaps, or failed diagnostic plots.

## Implementation Rules

- Treat this as a generic data connector; do not make a downstream model file the default output.
- Keep downloads source-bounded and request-bounded. Do not bulk-download whole collections unless the user explicitly approves.
- Do Python plotting and health reports locally. If Kestrel is used, use it for remote download/storage staging, then download compact evidence or products back for local checks.
- Keep legacy model-specific functions and file conventions available as deprecated compatibility aliases where they already exist, but prefer generic names in new docs, manifests, tests, and examples.
- Do not store credentials, personal tokens, passwords, OTPs, or unsupported source-access claims in scripts, logs, metadata, or examples.

## Validation

- Validate the skill with `quick_validate.py`.
- Compile all Python scripts after edits.
- Test `estimate_data_request.py` with local, Kestrel, and unknown-estimate cases.
- Test `check_download_health.py` on a tiny cached or synthetic artifact and confirm JSON plus at least one plot for plottable data.
