---
name: remora-grid-generation
description: Build and visually validate a single-level REMORA structured grid from a scientific region plan, including rotated orthogonal geometry, bathymetry, automatic mask correction, vertical diagnostics, CDF5 NetCDF, and executable-reader evidence.
---

# REMORA Grid Generation

Own the numerical grid, not the scientific polygon. For a complete scientific request, invoke `$remora-region-bpoly` as a returning subworkflow, inspect its maps and retain its accepted delivery. Consume an existing reviewed region plan directly without generating it twice. Work under the user's REMORA workspace; write project memory to the user's REMORA memory folder.

## Runtime and references

Use Python 3.11+ with prebuilt packages in [requirements.txt](requirements.txt): `python -m pip install --only-binary=:all: -r requirements.txt`. No C/C++ grid generator, source build, Gmsh, or remote grid-generation dependency is required. Install `remora-region-bpoly` beside this skill. See [interfaces](references/interfaces.md) for requests and [numerics](references/numerics.md) for conventions and limitations.

Emma's remora-region-setup is a production-structure/output reference, not imported runtime code. Pinned REMORA source determines reader conventions; [reader verification](references/reader_check.md) describes that separate check.

## Two returning stages

### 1. Region plan

Derive features and research geographic scope, invoke `remora-region-bpoly`, inspect and review its maps, then retain `region_delivery.json`. The command `prepare --request request.json --output-dir case` invokes the polygon CLI for an explicit `region_request` and returns for agent visual review. Continue with its generated `grid_request.json`; do not treat `prepare` as scientific acceptance.

### 2. Grid production

1. Prepare a grid request with region delivery, spacing, vertical parameters and protected wet feature points. The default pilot parameters are 500 m, hmin 2 m, rx0 0.2, N 40, theta_s 6, theta_b 2, hc 20 m. They are configurable, not universal scientific requirements.
2. Run `python scripts/remora_grid.py fit --request request.json --output-dir case/02_fit`. **Open its footprint map.** Verify coverage, orientation, boundary placement and additional area introduced by fitting. Record review with `review-fit --artifact case/02_fit/grid_fit.json --decision accepted --rationale "..."`.
3. Obtain coastline/bathymetry coverage for the fitted footprint plus rho halo. Use `$gshhs-coastline` and `$cudem-bathy`, following estimate, cache, source, datum and health rules. Supplied healthy artifacts may be reused. The v1 bathymetry adapter consumes a rectilinear NetCDF elevation or depth field in metres; use the connector to prepare other formats. Never make missing source data into shallow water.
4. Run `build --fit case/02_fit/grid_fit.json --output-dir case/03_grid`. It classifies wet cells from shoreline and bathymetry, retains the connected component containing all protected wet features, smooths natural-log wet depths with area-weighted pair corrections and reports volume change without rescaling, writes CDF5 and reopens it.
5. **Open every delivered map.** Inspect initial/final masks, corrected cells, feature closeups, bathymetry changes, geometry quality, and vertical sections. If a required feature becomes disconnected or underresolved, improve its source/spacing or make an evidence-backed request adjustment and create a new attempt. Do not fabricate a wet channel through mapped land or silently close a requested passage. Large smoothing changes require scientific examination even if rx0 passes.
6. Record `review --artifact case/03_grid/grid_delivery.json --decision accepted --rationale "..."`, then run `validate --delivery ... --require-reviewed`. This is an agent review; do not insert a routine human approval request.
7. Perform a version-bound REMORA reader/initialization check when runtime access is in scope. Keep `reader_validation` separate from local validation. Use `validate --require-reviewed --require-reader` only with genuine matching executable evidence. Never call generated files simulation-validated because a Python reader succeeded.

## Standard case directory and finalization

When a case directory is supplied, use it exactly. Read [case delivery](references/case_delivery.md) before production. Keep requests and stage outputs in `attempts/attempt_NNN/`, sources in `sources/`, and the scientific request in `science_request.json`. Never place case outputs in installed skills or invent a different root. Reuse healthy shared source data read-only with explicit source records.

After opening every map and recording accepted agent reviews, run `python scripts/remora_grid.py finalize --case-root CASE --delivery CASE/attempts/attempt_001/03_grid/grid_delivery.json --release-revision COMMIT`. Obtain COMMIT from the parent campaign's `release.json`; use `unpublished` only for development fixtures. Run `validate-case --delivery CASE/final/case_delivery.json`. Finalization preserves the original hash-bound evidence, copies the exact NetCDF, and creates a standard HTML review page. It does not grant human acceptance or executable validation.

## Delivery and scope

Return the canonical grid, region and fit references, parameter request, source hashes/datums, validation, mask edits, maps, visual reviews and `inputs.grid`. The snippet requires a complete simulation configuration with IC/BC/forcing; it is not itself a runnable scientific case.

v1 generates one root grid. Plans may contain nesting interests; they do not become independent rotated child grids. Later numerical nesting owns alignment, refinement ratios, mask/bathymetry consistency and coupling. Geographic polygons are never rasterized as artificial land walls.

Run `scripts/selftest.py` for numerical/error-path tests, `scripts/selftest_workflow.py` for both command-line stages and review handoffs, and `remora-region-bpoly/scripts/selftest.py` for hierarchy and region integrity. Keep generated artifacts outside installed skills. Install and validate first; publish matching code only after tests and case review. Publication itself requires user authorization.
