---
name: gshhs-coastline
description: Fetch, clip, inspect, and QA GSHHG/GSHHS shoreline polygon data for regional coastal and ocean modeling workflows. Use when Codex needs a robust polygonal coastline/topology base, derived coastline lines, bbox clipping, cache-first GSHHG setup, or health checks before downstream FVCOM boundary-arc and grid-generation work.
---

# GSHHS Coastline

Use this skill as a generic external-data connector for GSHHG/GSHHS shoreline data. Keep it source-focused: download/cache, clip by bbox, preserve provenance, make diagnostic maps, and health-check vector products. Do not mix CUSP refinement into v1 outputs.

## Source And Toolbox

- Primary source: GSHHG/GSHHS shapefile release `2.3.7`, preferably from an existing local cache, otherwise from the SOEST hosted ZIP.
- Product focus: closed land polygons and derived coastline lines. Use polygons as the topology base for downstream workflows.
- Main scripts:
- `scripts/fetch_gshhs_coastline.py`
- `scripts/estimate_data_request.py`
- `scripts/check_download_health.py`
- `scripts/selftest_gshhs_coastline.py`

Read `references/gshhg_sources.md` before changing source URLs, resolution policy, level handling, or topology caveats.

## Required Workflow

1. Inspect the request and identify bbox, resolution, levels, formats, run directory, and cache directory.
2. Run the estimate hook before any live download:

```powershell
python scripts/estimate_data_request.py --request request.json --run-dir runs/case --output runs/case/download_estimate.json
```

3. Prefer local cache. The skill searches:
- the requested `--cache-dir`;
- `Workspace/Preprocessing/fvcom-gshhs-coastline/cache/gshhg`;
- legacy cache `Workspace/Preprocessing/fvcom-cusp-coastline/cache/gshhg`.
4. Fetch/clip the bbox:

```powershell
python scripts/fetch_gshhs_coastline.py --bbox W S E N --run-dir runs/case --name case --resolution h --levels 1
```

For FVCOM topology, pass the controlling RegionBPoly envelope as `--model-bbox`, not `--bbox`. The connector centers it in a 3x source footprint, rejects factors below 2x, and applies any requested symmetric look-ahead halo:

```powershell
python scripts/fetch_gshhs_coastline.py --model-bbox W S E N --coverage-factor 3 --lookahead-km 100 --run-dir runs/case --name case --resolution f --levels 1
```

5. Run the health gate:

```powershell
python scripts/check_download_health.py --run-dir runs/case --output runs/case/health_check.json --plots-dir runs/case/health_plots
```

## Outputs

For a run named `case`, write:

- `case_gshhs_land.gpkg` with `land_polygons`, physical `coastline_lines`, `request_bbox`, `source_footprint`, and `source_frame`; topology requests also include `model_bbox`;
- `case_gshhs_land.geojson` and `case_gshhs_coastline.geojson` when requested;
- optional shapefile folders when requested;
- `case_gshhs_map.png`;
- `case_gshhs_manifest.json`;
- `health_check.json` and `health_plots/`.

## Resolution Policy

Use `--resolution auto|c|l|i|h|f`, default `auto`.

- Use `h` for regional domains such as Delaware Bay and most bpoly-scale topology work.
- Use `f` for small estuary, inlet, or high-detail requests.
- If a requested resolution is missing, download/extract the SOEST ZIP when allowed. If download fails but another requested-compatible cache is available, report the fallback in the manifest.

Use `--levels 1` by default. Level 1 is land. Other levels may be useful for lakes/islands-in-lakes, but downstream FVCOM topology should request them deliberately.

## Implementation Rules

- Treat GSHHS/GSHHG as the robust topology base. Do not treat CUSP as part of this connector.
- Clip land polygons by bbox for wet/land masking, and derive coastline lines from source polygon boundaries clipped to bbox.
- Keep exact `--bbox` behavior for generic extraction. Require `--model-bbox` for new FVCOM topology products, with centered 3x coverage and 2x as the hard minimum.
- Treat `coastline_lines` as the only physical shoreline layer. Never reinterpret the boundary of clipped `land_polygons` as coastline because it contains artificial source-frame edges.
- Preserve CRS, selected resolution, requested levels, source paths, source URL, cache status, bbox split/antimeridian metadata, feature counts, and warnings in the manifest.
- Validate selected source polygons before clipping. Apply `make_valid` only to invalid in-memory features, retain polygonal components, leave cached source files unchanged, and record repair counts, reasons, methods, and equal-area change in the manifest. Derive both clipped land and physical coastline from the validated geometry.
- Handle antimeridian bboxes by splitting the request into two longitude windows and recording that split.
- Preserve native GSHHS longitude coordinates across antimeridian requests.
  For topology QA, project both split windows directly into one compact metric
  CRS, snap only sub-metre projected seams, dissolve the projected polygons,
  and treat the resulting union exterior as the physical source footprint.
  Never warp longitudes or count the internal +/-180-degree split as physical
  coastline or source-frame overlap.
- Keep model-specific topology decisions in downstream skills such as `fvcom-bdry-arc`.

## Validation

From the skill folder:

```powershell
python scripts/selftest_gshhs_coastline.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
