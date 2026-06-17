---
name: fvcom-cusp-coastline
description: Fetch NOAA NGS Continually Updated Shoreline Product (CUSP) coastline vectors for FVCOM preprocessing, with explicit OSM Overpass gap-fill fallback, long-run progress logging, and agent visual QA gates when requested. Use when Codex needs to download/cache official NSDE regional CUSP shapefiles, clip shoreline lines to lon/lat bboxes, export FVCOM-ready GeoPackage/GeoJSON/Shapefile products, merge provenance-tagged fallback coastline, make satellite-background coastline diagnostic maps, babysit long coastline fetch/merge runs, or visually review coastline completeness before grid-generation acceptance.
---

# FVCOM CUSP Coastline

Use this skill to acquire NOAA NGS CUSP shoreline vectors and convert them into local coastline products for FVCOM grid generation.

## Workflow

1. Work in `Workspace/Preprocessing/fvcom-cusp-coastline` unless the user gives another run directory.
2. Build or reuse `cusp_region_index.json` before live fetches.
3. Use CUSP-only data by default. Do not silently fall back to OSM or any non-CUSP source.
4. Prefer official NSDE regional ZIP downloads and clip locally by bbox.
5. Output EPSG:4326 coastline products even though CUSP metadata is NAD83/EPSG:4269; record both CRS values in metadata.
6. Preserve CUSP source attributes, including `SOURCE_ID`, `SRC_DATE`, `HOR_ACC`, `INFORM`, `ATTRIBUTE`, `VER_DATE`, `SRC_RESOLU`, `DATA_SOURC`, `EXT_METH`, `DAT_SET_CR`, `SRC_CITA`, `FIPS_ALPHA`, and `NOAA_Regio` when present.
7. Write Shapefile ZIP, GeoPackage, GeoJSON, metadata JSON, and a satellite-background PNG for each bbox fetch.
8. Treat CUSP as contemporary planning/modeling shoreline, not legal or navigation authority.
9. Do not simplify, polygonize, or build FVCOM open boundaries inside this skill. Leave those steps to grid-generation/refinement skills.
10. When the user explicitly requests fallback, use `--fallback-policy osm-overpass` or `auto`. Query OSM by bbox through Overpass, cache the raw JSON, remove only OSM geometry already near CUSP, and preserve OSM/ODbL attribution in outputs. OSM Overpass is the only implemented fallback source.
11. Treat numeric checks as candidate screening only. Before accepting a coastline product, open each `*_cusp_satellite.png` and `*_merged_satellite.png` with visual inspection, then record a `pass`, `fail`, or `needs_followup` decision in `*_visual_review.json`.
12. Fail the visual gate when a visible land-water boundary, island, fjord, bay, or channel shoreline is missing from the vector overlay, even if feature count, bounds, and total length checks pass.
13. Run long coastline commands with visible stdout/stderr. Relay the latest progress line to the user about every `30 s`, and inspect the process if the latest progress event is older than `2 * heartbeat_seconds`.
14. Preserve partial products and `*_progress.jsonl` after interruption; use them to explain where the previous run stopped and resume from cached CUSP/OSM data.

## Commands

Build the regional index:

```powershell
python scripts\build_cusp_region_index.py --output Workspace\Preprocessing\fvcom-cusp-coastline\cache\cusp_region_index.json
```

Fetch one bbox:

```powershell
python scripts\fetch_cusp_coastline.py --bbox -75.35 38.75 -74.95 39.10 --run-dir Workspace\Preprocessing\fvcom-cusp-coastline\runs\delaware_bay --name delaware_bay --index Workspace\Preprocessing\fvcom-cusp-coastline\cache\cusp_region_index.json
```

Fetch one bbox with explicit OSM fallback and merged outputs:

```powershell
python scripts\fetch_cusp_coastline.py --bbox -136.20 58.10 -134.80 58.60 --run-dir Workspace\Preprocessing\fvcom-cusp-coastline\runs\seak_fallback_v2\se_ak_icy_strait --name se_ak_icy_strait --index Workspace\Preprocessing\fvcom-cusp-coastline\cache\cusp_region_index.json --fallback-policy auto --heartbeat-seconds 30 --client-timeout-s 0 --overpass-timeout-s 0
```

Timeout/progress defaults:

- `--client-timeout-s 0` means no hard Python client timeout.
- `--overpass-timeout-s 0` means omit the Overpass `[timeout:*]` clause.
- `--heartbeat-seconds 30` controls stdout progress and JSONL progress cadence.
- `--progress-jsonl <path>` overrides the default `<run-dir>/<name>_progress.jsonl`.
- Use `--quiet` only when stdout progress is not wanted; JSONL progress still writes.

Run required smoke tests:

```powershell
python scripts\smoke_cusp_regions.py --run-dir Workspace\Preprocessing\fvcom-cusp-coastline\runs\smoke --index Workspace\Preprocessing\fvcom-cusp-coastline\cache\cusp_region_index.json
```

Record visual review after opening a PNG:

```powershell
python scripts\record_visual_review.py --manifest Workspace\Preprocessing\fvcom-cusp-coastline\runs\smoke\se_ak_sumner_strait\se_ak_sumner_strait_visual_review.json --decision fail --reviewer codex-agent --notes "Satellite overlay shows visible shoreline gaps; use fallback or a smaller diagnostic bbox before production use." --fail-reason "Visible shoreline without vector overlay"
```

Run SE-AK fallback smoke tests:

```powershell
python scripts\smoke_seak_fallback.py --run-dir Workspace\Preprocessing\fvcom-cusp-coastline\runs\seak_fallback_v2 --index Workspace\Preprocessing\fvcom-cusp-coastline\cache\cusp_region_index.json
```

Require visual review decisions during smoke validation:

```powershell
python scripts\smoke_cusp_regions.py --run-dir Workspace\Preprocessing\fvcom-cusp-coastline\runs\smoke --index Workspace\Preprocessing\fvcom-cusp-coastline\cache\cusp_region_index.json --require-visual-review
```

## References

- Read `references/noaa_cusp_sources.md` before changing source URLs, region routing, CRS assumptions, CUSP attribute preservation, or smoke-test regions.

## Guardrails

- Cache large regional ZIPs under the workspace, not in the skill folder.
- Keep basemap generation online and explicit; if a smoke test cannot make a satellite map, fail unless the user intentionally passes `--allow-no-basemap`.
- Keep metadata alongside every clipped product.
- Keep `*_visual_review.json` and `*_visual_review.md` alongside every plotted product. A coastline with `needs_agent_review`, `fail`, or `needs_followup` is not production-accepted.
- For complex archipelago/fjord settings such as SE-AK, prefer smaller diagnostic bboxes and explicit fallback tests over one broad bbox that can hide local shoreline gaps.
- Do not download global OSM coastline shapefiles for v2 fallback; use small bbox Overpass queries.
- Treat GSHHG and Natural Earth as non-production diagnostic/future options only; they are not active fallback policies in this skill.
- Use Matplotlib, GeoPandas, Pyogrio, Shapely, PyProj, Rasterio, and Contextily; do not require Cartopy.
