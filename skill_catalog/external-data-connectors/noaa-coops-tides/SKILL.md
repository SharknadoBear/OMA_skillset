---
name: noaa-coops-tides
description: Discover, fetch, and inspect NOAA CO-OPS water-level, prediction, temperature, salinity, and current-profile observations. Use for wet-domain station screening, monthly scalar caches, resumable seven-day all-bin current downloads, QC, vector/depth averaging, and harmonic-ready time series.
---

# NOAA CO-OPS Tides

Use this skill as a self-contained Python toolbox for external data access. Keep the skill focused on source-specific fetching, cache management, basic transforms, estimate-first planning, and downloaded-data quality gates. Keep model-specific products as downstream or legacy compatibility work.

## Source And Toolbox

- Primary source: NOAA CO-OPS API observations.
- Toolbox focus: water level, predictions, temperature, salinity, current profiles, station/deployment/bin metadata, harmonic analysis, and time-series cache products.
- Main packaged scripts:
- `scripts/noaa_tides.py`
- `scripts/coops_currents.py`: discover water-level/current stations inside a projected wet 2DM, classify profiler orientation, download `currents` in API-compliant seven-day `bin=0` chunks, and produce east/north depth-integrated currents.
- `scripts/screen_tidal_stations.py`: screen a residual-boundary contract against tidal CO-OPS stations within a bounded radius. Query both NOAA water-level and tide-prediction catalogs, deduplicate station IDs with catalog provenance, and retain product, datum, harmonic and water-component eligibility checks. A station is eligibility evidence only; it never creates an OBC automatically and never substitutes a river gauge.
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
   - if the local disk does not satisfy that rule, use `$kestrel-hpc` and plan work under `/scratch/yhuang168/oma_external_data_connectors/noaa-coops-tides/<run-id>`;
   - if the estimate is unknown, do not download until the request is narrowed or explicitly reviewed.
4. Execute the source-specific toolbox script with a small smoke-test window before broader requests.
5. Preserve source URLs, selected files, variable names, coverage, CRS/datum/time metadata, cache paths, and any fallback decisions in run metadata.
6. Run the health gate after download:

```bash
python scripts/check_download_health.py --request request.json --run-dir runs/case --output runs/case/health_check.json --plots-dir runs/case/health_plots
```

7. Surface the health report to Bear only when important caveats exist, such as missing requested coverage, empty variables, all-NaN fields, finite coverage below 95 percent, obvious gaps, or failed diagnostic plots.

For FVCOM validation, retain actual wet-mesh containment evidence. The default water-level policy is `strict_inside`. When explicitly requested, use `--water-level-mapping-policy containing_cell_or_nearest_wet_cell`: outside water-level gauges within the regional search envelope remain source-data candidates for downstream cell-centroid proxy review. Keep `inside_wet_mesh=false` truthful, and verify requested-period observations/predictions separately. Current eligibility retains strict spatial, downward-looking profile and period gates. This policy does not relax residual-boundary screening.

```bash
python scripts/coops_currents.py discover --mesh fvcom_grid.2dm --mesh-crs EPSG:32615 --period-start 2025-04-01T00:00:00Z --period-end 2025-05-01T00:00:00Z --output station_inventory.json
```

Download an eligible downward-looking profiler in resumable seven-day all-bin chunks:

```bash
python scripts/coops_currents.py fetch --station g06010 --period-start 2025-04-01T00:00:00Z --period-end 2025-05-01T00:00:00Z --cache-dir observations/currents/g06010/cache --output observations/currents/g06010/depth_mean.csv --manifest observations/currents/g06010/manifest.json
```

Metadata-only residual screening is bounded inventory work rather than an observation download. Run it directly against the hash-bound boundary contract and retained wet-domain package:

```bash
python scripts/screen_tidal_stations.py --open-exterior-contract open_exterior_contract.json --wet-domain-gpkg bdry_arc_package.gpkg --output-dir station_screen --radius-km 25
```

Require `tidal=true`, water-level or prediction products, datum evidence, and membership in the same retained wet component. Treat a qualifying station as permission for a later agent decision only. Respect the requested OBC count and keep a residual solid when another OBC is not authorized.

## Implementation Rules

- Treat this as a generic data connector; do not make a downstream model file the default output.
- Keep downloads source-bounded and request-bounded. Do not bulk-download whole collections unless the user explicitly approves.
- The Data API requires `bin=0` all-bin current requests to span no more than seven days. Cache each chunk, verify station metadata and bin IDs, and deduplicate overlapping endpoints when resuming.
- Treat CO-OPS current direction as degrees relative to true north and convert flow-toward vectors with `east=speed*sin(direction)` and `north=speed*cos(direction)`. Metric API current speeds are converted from cm/s to m/s and the conversion is written to the manifest.
- Quantitative depth-integrated validation is limited to downward-looking profilers. Retain side-looking stations in discovery output with an exclusion reason.
- Current eligibility also requires documented deployment overlap with the requested period. Preserve historic stations with their deployment dates and exclusion reason. Missing or ambiguous date metadata is unknown availability and requires a bounded requested-period data probe; never report it as confirmed absence or silently substitute a different year.
- For downward profiles, derive layer interfaces from adjacent valid bin-depth midpoints. Extrapolate the shallowest vector to the surface and deepest valid vector to the bed only when the deployment water depth bounds the bins; record excluded bins, extrapolation, finite-bin fraction, and thickness weights for every timestamp.
- Do Python plotting and health reports locally. If Kestrel is used, use it for remote download/storage staging, then download compact evidence or products back for local checks.
- Keep legacy model-specific functions and file conventions available as deprecated compatibility aliases where they already exist, but prefer generic names in new docs, manifests, tests, and examples.
- Do not store credentials, personal tokens, passwords, OTPs, or unsupported source-access claims in scripts, logs, metadata, or examples.

## Validation

- Validate the skill with `quick_validate.py`.
- Compile all Python scripts after edits.
- Test `estimate_data_request.py` with local, Kestrel, and unknown-estimate cases.
- Test `check_download_health.py` on a tiny cached or synthetic artifact and confirm JSON plus at least one plot for plottable data.
- Run `python scripts/selftest_station_screen.py` and confirm tidal/product/datum/connectivity filtering and river-gauge rejection.
- Run `python scripts/selftest_coops_currents.py` and confirm seven-day chunking, speed/direction conversion, orientation filtering, bin QC, and known-vector vertical weighting.
