---
name: cusp-coastline-copilot
harness: github-copilot
description: GitHub Copilot variant for NOAA CUSP coastline fetching with OSM fallback. Use run_in_terminal for script execution, read_file for JSON results, view_image for map diagnostics. Scripts shared with sibling cusp-coastline/ folder.
---

# CUSP Coastline — GitHub Copilot Harness

## Source

- Primary: NOAA NGS Continually Updated Shoreline Product (CUSP) vectors.
- Fallback: OSM Overpass API.
- Toolbox: source indexing, bbox clipping, provenance-preserving merge, map diagnostics, visual QA.

## Scripts Location

All scripts in sibling folder:

```
Agent_skill_dev/skill_catalog/external-data-connectors/cusp-coastline/scripts/
```

Key scripts:
- `build_cusp_region_index.py` — Index CUSP regional ZIPs
- `fetch_cusp_coastline.py` — Download bbox → Shapefile/GeoPackage/GeoJSON
- `record_visual_review.py` — Manual QA documentation
- `estimate_data_request.py` — Pre-check disk space
- `check_download_health.py` — Post-download QA

## Required Workflow (Copilot Execution)

### 1. Estimate First

```powershell
# run_in_terminal (mode=sync)
python "Agent_skill_dev\skill_catalog\external-data-connectors\cusp-coastline\scripts\estimate_data_request.py" --request request.json --run-dir runs/case --output runs/case/download_estimate.json
```

`read_file` the estimate. Storage routing:
- `local_free_bytes > 4 * estimated_requested_bytes` → local
- Otherwise → Kestrel `/scratch/yhuang168/oma_external_data_connectors/cusp-coastline/<run-id>`
- Unknown → do NOT download; narrow request first

### 2. Fetch (Smoke Test First)

```powershell
# run_in_terminal (mode=sync)
python "Agent_skill_dev\skill_catalog\external-data-connectors\cusp-coastline\scripts\fetch_cusp_coastline.py" --bbox W S E N --run-dir runs/case --name case_name --formats gpkg
```

Start with a small bbox or `--region auto` to test connectivity.

### 3. Health Check

```powershell
# run_in_terminal (mode=sync)
python "Agent_skill_dev\skill_catalog\external-data-connectors\cusp-coastline\scripts\check_download_health.py" --request request.json --run-dir runs/case --output runs/case/health_check.json --plots-dir runs/case/health_plots
```

- `read_file` on `health_check.json`
- `view_image` on diagnostic plots
- Surface to user only for important caveats

### 4. Visual Review (If Needed)

```powershell
python "Agent_skill_dev\skill_catalog\external-data-connectors\cusp-coastline\scripts\record_visual_review.py" --run-dir runs/case --decision pass --notes "Coastline visually verified"
```

## Copilot Tool Integration

| Step | Tool |
|------|------|
| Run any script | `run_in_terminal` (mode=sync) |
| Read JSON results | `read_file` |
| View map diagnostics | `view_image` |
| Ask user about storage/review | `vscode_askQuestions` |
| Route large downloads to Kestrel | Use kestrel-hpc-copilot bridge |

## Implementation Rules

- Generic data connector — do not produce model-specific output by default.
- Keep downloads source-bounded and request-bounded.
- Do Python plotting and health reports locally.
- Preserve source URLs, CRS, attributes, and fallback decisions in metadata.
- Do not store credentials in scripts, logs, or metadata.
