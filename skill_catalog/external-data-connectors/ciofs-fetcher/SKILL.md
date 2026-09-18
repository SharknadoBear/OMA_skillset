---
name: ciofs-fetcher
description: "Inventory, estimate, fetch, resume, extract, and health-check bounded NOAA Cook Inlet Operational Forecast System (CIOFS) ROMS point time series. Use for recent nowcast or single-cycle forecast currents, salinity, temperature, and water level at nearest wet cells or exact-coordinate bilinear points, including sigma-layer selection and true-east/north current rotation."
---

# CIOFS Fetcher

Use the bundled, project-independent scripts for point subsets of NOAA's rolling CO-OPS THREDDS archive. Keep harmonic analysis, plotting, comparisons, and engineering interpretation downstream.

## Scope and contract

- Read [source_contract.md](references/source_contract.md) before retrieval or interpreting output. Read [request.schema.json](references/request.schema.json) when preparing a request.
- Version 1 supports hourly native CIOFS nowcasts and one explicitly selected forecast cycle. It does **not** implement historical AWS/NCEI retrieval, full-field downloads, station products, depth averages, or fixed-depth interpolation. Do not claim those capabilities or silently substitute another model/source.
- Inputs are explicit UTC hour-aligned `[start_utc, end_utc_exclusive)` bounds, WGS84 points, and variables. No preferred station, analysis duration, dates, minimum depth, or acceptance criteria are silently inherited from the Cook Inlet tidal-ellipse project.
- Available variables: paired `u`/`v` (`currents`), `salt` (`salinity`), `temp` (`temperature`), and `zeta` (`water_level`). Currents are always processed together.
- Vertical views: `surface` (or `near_surface`), `bottom`, and zero-based `index:N`, resolved from source `s_rho`. These are moving sigma layers, not specified depths in metres. Selecting widely separated views retrieves the contiguous intervening layers within each tiny horizontal stencil.
- Sampling: `nearest_wet` or `bilinear`. The latter requires an enclosing eligible wet quadrilateral; no extrapolation or nearest-cell fallback. All points must be selectable. `required: false` relaxes only the final time-series coverage gate.

## Workflow

1. Confirm the user's variables, valid-time range, near-surface/bottom/index choice, and points. Use explicit reasonable assumptions for routine details. Missing dates or a materially ambiguous science request require clarification.
2. Choose a run directory **outside this skill**. Use an existing Python with dependencies in [requirements.txt](scripts/requirements.txt), or install them in a task-local environment. Do not download datasets into the skill package.
3. Prepare a request using [request.example.json](assets/request.example.json) as a syntax example. Its dates and locations are illustrative, **not defaults**; old dates can expire from the rolling archive. A forecast also requires `run_cycle_utc` at 00/06/12/18 UTC and valid hours cycle+1 through cycle+48.
4. Run `inventory`, then `plan`. Inventory reads bounded day catalogs only. Planning downloads a one-time static geometry probe (roughly tens of MB), selects stencils, records coordinates/offsets/interpolation weights, and estimates the time-series transfer/storage. Inspect `inventory.json`, `points.csv`, `interpolation_support.csv`, and `download_estimate.json` before bulk retrieval. Report missing hours and the estimate to the user. A new approval is not necessary for a reasonable retrieval already authorized by the task.
5. Do not shift requested dates. Default `missing_policy: error` refuses an incomplete source inventory. `skip` must be an explicit request choice; omitted hours remain missing and count against completeness. Resource pressure or unavailable archive data calls for a smaller explicit request, a suitable storage volume, or user direction—not fabricated availability.
6. Run `fetch` against the saved, hash-bound plan. It uses bounded retries and at most eight concurrent source-hour workers (four by default). Rerunning resumes verified caches. Changed/corrupt artifacts cause a failure; investigate and replan into a new run directory rather than suppressing validation.
7. Run `extract`, then `health`. Both work offline once retrieval is complete. Extraction uses per-file CF time and masks; it does not fill gaps. Inspect all required point/view/variable coverage and any timestamp adjustments. Rerunning `fetch` refreshes its manifest, so rerun `extract` and `health` afterward.
8. Deliver the point table, CSV/NetCDF series, coverage, and health/provenance paths. State any exclusions, missing hours, and archive limits. For downstream tides, separately enforce the analysis duration and constituent-resolution requirements; a successful fetch does not establish harmonic-analysis adequacy.

Example commands (replace `PYTHON`, `SKILL`, `REQUEST`, and `RUN` with actual paths; quote paths containing spaces):

```text
PYTHON -B SKILL/scripts/ciofs_fetcher.py inventory --request REQUEST --run-dir RUN
PYTHON -B SKILL/scripts/ciofs_fetcher.py plan --request REQUEST --run-dir RUN
PYTHON -B SKILL/scripts/ciofs_fetcher.py fetch --plan RUN/download_estimate.json --run-dir RUN
PYTHON -B SKILL/scripts/ciofs_fetcher.py extract --run-dir RUN
PYTHON -B SKILL/scripts/ciofs_fetcher.py health --run-dir RUN
```

`inspect --url URL --request REQUEST --output PATH` inspects an explicit canonical native-file URL. `selftest_ciofs_fetcher.py` runs offline tests without downloading model data.

## Scientific and provenance gates

- Validate named C-grid dimensions, binary masks, radian rotation-angle semantics, sigma coordinates, source units, and geometry consistency. Unsupported packing or changed contracts fail closed.
- Destagger valid paired face velocities to rho, rotate each contributing rho vector into true east/north, then interpolate components. Never average headings, speeds, or harmonic phases instead of components. Dry faces or fill values are missing, not zero current.
- Bilinear support weights must be nonnegative, sum to one, and reproduce the requested coordinate. Save contributing indices, coordinates, angles, and weights.
- Use the decoded CF time as authoritative. Deviations up to 60 seconds from the hourly cadence are explicitly retained and audited when normalized; larger deviations and duplicate decoded records fail. Filename times are inventory hints, not analysis timestamps.
- Raw subset binaries, source metadata, request, grid, and plan are hash-bound. Health replays extraction offline and checks output integrity, coordinates, timestamps, values, masks, and per-variable coverage. Hashes describe locally frozen subsets, not whole remote NetCDF files or future upstream immutability.
- The output schema is `ciofs_point_series_v1`, **not** `roms_compact_fields_v1`. Surface velocity is near-surface model-layer velocity, not depth-averaged velocity. Water level retains the model's vertical reference without datum conversion.
