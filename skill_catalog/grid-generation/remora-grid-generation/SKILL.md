---
name: remora-grid-generation
description: Build and visually validate a science-driven single-level REMORA structured grid with conformal curvilinear geometry, smooth graded spacing, bathymetry, mask correction, vertical-coordinate checks, CDF5 NetCDF, and separate executable-reader evidence.
---

# REMORA Grid Generation

Own the numerical grid, beginning with the scientific purpose and reviewed geographic plan. For a complete scientific request, invoke `$remora-region-bpoly` as a returning subworkflow, inspect its maps and retain its accepted delivery. Consume an existing reviewed region plan directly without generating it twice. Work under the user's REMORA workspace; write project memory to the user's REMORA memory folder.

## Scientific design

Read [scientific design guidance](references/scientific_design.md) before selecting resolution and vertical parameters. Preserve the scientific prompt as prose; the case agent chooses and records numerical parameters in the grid request.

- Explicitly choose `barotropic_tide` or `baroclinic`. For barotropic tidal work, choose and justify 8–20 vertical layers; satisfy rx0 ≤ 0.2 and actual layer-aware rx1 < 9, preferably < 7. Thresholds and stretching parameters are configurable scientific choices. A baroclinic grid needs its own vertical-resolution rationale.
- Treat several hundred physical cells per direction as a useful starting scale, not a hard cap. Larger grids need a scientific and computational-budget rationale. Do not globally refine the root grid to resolve a tiny secondary feature.
- Preserve essential water connections and real islands. Optional simplification requires inspected evidence; future refinement means later nesting. Avoid arbitrary upstream, peripheral, or deep-offshore extent.
- Choose footprint shape, grid orientation, smooth axis grading, and open-edge placement together. Several open logical sides are valid. Curvature must improve the scientific fit and pass the geometry checks; its availability is not a reason to distort a suitable rectangle.

## Runtime and references

Use Python 3.11+ with prebuilt packages in [requirements.txt](requirements.txt): `python -m pip install --only-binary=:all: -r requirements.txt`. No C/C++ grid generator, source build, Gmsh, or remote grid-generation dependency is required. Install `remora-region-bpoly` beside this skill. See [interfaces](references/interfaces.md) for requests and [numerics](references/numerics.md) for supported geometry and conventions.

Vertical diagnostics use bounded row blocks and section plots compute only their selected slices. Size the horizontal grid to available memory; do not reduce scientifically chosen vertical layers solely to bypass diagnostic memory use.

Emma's remora-region-setup is a production-structure/output reference, not imported runtime code. Pinned REMORA source determines reader conventions; [reader verification](references/reader_check.md) describes that separate check.

## Two returning stages

### 1. Region plan

Derive features and research geographic scope, invoke `remora-region-bpoly`, inspect and review its maps, then retain `region_delivery.json`. The command `prepare --request request.json --output-dir case` invokes the polygon CLI for an explicit `region_request` and returns for agent visual review. Continue with its generated `grid_request.json`; do not treat `prepare` as scientific acceptance.

### 2. Grid production

1. Prepare a grid request with explicit science and design rationales, numerical geometry, vertical parameters and protected wet feature points. Match every essential water point in the selected region by exact name and coordinates; give each essential water box a named representative anchor, and declare essential passages separately. Mark island features as land. For new designs, use the bounded `conformal_graded` geometry and choose its dimensions, grading and coefficients intentionally; zero curvature and identity grading are supported when justified. Legacy requests retain their earlier behavior for compatibility.
2. Run `python scripts/remora_grid.py fit --request request.json --output-dir case/02_fit`. **Open its footprint map.** Verify essential-feature coverage, orientation, spacing distribution, open-edge alignment, island cuts and additional area introduced by fitting. Check the geometry diagnostics and explain the chosen cell budget. Record review with `review-fit --artifact case/02_fit/grid_fit.json --decision accepted --rationale "..."`.
3. Obtain coastline/bathymetry coverage for the fitted footprint plus rho halo. First inventory local source manifests and healthy cache artifacts read-only; reuse suitable data with explicit provenance. Use `$gshhs-coastline` and `$cudem-bathy` as needed, following estimate, cache, source, datum and health rules. The bathymetry adapter consumes a rectilinear NetCDF elevation or depth field in metres; use the connector to prepare other formats. Never make missing source data into shallow water. Evidence-backed source repairs and optional mask closures use the bounded interfaces in [the request reference](references/interfaces.md).
4. Run `build --fit case/02_fit/grid_fit.json --output-dir case/03_grid`. It classifies wet cells from shoreline and bathymetry, retains the connected component containing all protected wet features, smooths natural-log wet depths with area-weighted pair corrections, reports volume change without rescaling, writes CDF5 and reopens it. New scientific requests include the actual vertical-coordinate rx1 target in conditioning and validation.
5. **Open every delivered map.** Inspect initial/final masks, corrected cells, feature closeups, bathymetry changes, geometry quality, and vertical sections. If an essential feature becomes disconnected or underresolved, improve its source or grid design and create a new attempt. Optional simplification needs evidence and a recorded scientific rationale. Do not fabricate a wet channel through real land or silently close a requested passage. Large smoothing changes require scientific examination even when roughness thresholds pass. Check apparent bridge barriers against actual geography: correct a source artifact only with evidence, retain real islands and causeways, and leave bridge-pier drag and wetting/drying configuration to later modeling.
6. Record `review --artifact case/03_grid/grid_delivery.json --decision accepted --rationale "..."`, then run `validate --delivery ... --require-reviewed`. This is an agent review; do not insert a routine human approval request.
7. Perform a version-bound REMORA reader/initialization check when runtime access is in scope. Keep `reader_validation` separate from local validation. Use `validate --require-reviewed --require-reader` only with genuine matching executable evidence. Never call generated files simulation-validated because a Python reader succeeded.

## Standard case directory and finalization

When a case directory is supplied, use it exactly. Read [case delivery](references/case_delivery.md) before production. Keep requests and stage outputs in `attempts/attempt_NNN/`, sources in `sources/`, and the scientific request in `science_request.json`. Never place case outputs in installed skills or invent a different root. Reuse healthy shared source data read-only with explicit source records.

After opening every map and recording accepted agent reviews, run `python scripts/remora_grid.py finalize --case-root CASE --delivery CASE/attempts/attempt_001/03_grid/grid_delivery.json --release-revision COMMIT`. Obtain COMMIT from the parent campaign's `release.json`; use `unpublished` only for development fixtures. Run `validate-case --delivery CASE/final/case_delivery.json`. Finalization preserves the original hash-bound evidence, copies the exact NetCDF, and creates a standard HTML review page. It does not grant human acceptance or executable validation.

## Delivery and scope

Return the canonical grid, region and fit references, parameter request, source hashes/datums, validation, mask edits, maps, visual reviews and `inputs.grid`. The snippet requires a complete simulation configuration with IC/BC/forcing; it is not itself a runnable scientific case.

This workflow generates one root grid. Plans may contain nesting interests; they do not become independent rotated child grids. Later numerical nesting owns alignment, refinement ratios, mask/bathymetry consistency and coupling. Geographic polygons are never rasterized as artificial land walls. Grid acceptance does not authorize a tidal simulation campaign.

Run `scripts/selftest.py` for numerical/error-path tests, `scripts/selftest_workflow.py` for both command-line stages and review handoffs, and `remora-region-bpoly/scripts/selftest.py` for hierarchy and region integrity. Keep generated artifacts outside installed skills. Install and validate first; publish matching code only after tests and case review. Publication itself requires user authorization.
