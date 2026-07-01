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

5. Run the health gate:

```powershell
python scripts/check_download_health.py --run-dir runs/case --output runs/case/health_check.json --plots-dir runs/case/health_plots
```

## Outputs

For a run named `case`, write:

- `case_gshhs_land.gpkg` with layers `land_polygons`, `coastline_lines`, `request_bbox`, and `source_footprint`;
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
- Preserve CRS, selected resolution, requested levels, source paths, source URL, cache status, bbox split/antimeridian metadata, feature counts, and warnings in the manifest.
- Handle antimeridian bboxes by splitting the request into two longitude windows and recording that split.
- Keep model-specific topology decisions in downstream skills such as `fvcom-bdry-arc`.

## Validation

From the skill folder:

```powershell
python scripts/selftest_gshhs_coastline.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
