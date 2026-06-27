---
name: cudem-bathy-copilot
harness: github-copilot
description: GitHub Copilot variant for NOAA CUDEM bathymetry fetching. Use run_in_terminal for script execution, read_file for JSON results, view_image for diagnostic plots. Scripts shared with sibling cudem-bathy/ folder.
---

# CUDEM Bathy — GitHub Copilot Harness

## Source

- Primary: NOAA CUDEM, NBS BlueTopo, NOAA CRM, ETOPO bathymetry/topobathymetry.
- Toolbox: indexing, tile selection, mosaics, sampled depths, coverage diagnostics, health checks.

## Scripts Location

All scripts in sibling folder:

```
Agent_skill_dev/skill_catalog/external-data-connectors/cudem-bathy/scripts/
```

Key scripts:
- `build_cudem_index.py` — Index THREDDS collections
- `build_bathy_source_index.py` — Multi-source index
- `fetch_cudem_bathy.py` — Download bbox → NetCDF/PNG/JSON
- `fetch_bathy_sources.py` — Multi-source priority fallback
- `estimate_data_request.py` — Pre-check disk space
- `check_download_health.py` — Post-download QA

## Required Workflow (Copilot Execution)

### 1. Estimate First

```powershell
# run_in_terminal (mode=sync)
python "Agent_skill_dev\skill_catalog\external-data-connectors\cudem-bathy\scripts\estimate_data_request.py" --request request.json --run-dir runs/case --output runs/case/download_estimate.json
```

Then `read_file` on the estimate JSON. Check storage routing:
- `local_free_bytes > 4 * estimated_requested_bytes` → download locally
- Otherwise → route to Kestrel `/scratch/yhuang168/oma_external_data_connectors/cudem-bathy/<run-id>`
- Unknown estimate → do NOT download; ask user to narrow request

### 2. Fetch (Smoke Test First)

```powershell
# run_in_terminal (mode=sync)
python "Agent_skill_dev\skill_catalog\external-data-connectors\cudem-bathy\scripts\fetch_cudem_bathy.py" --bbox W S E N --run-dir runs/case --name case_name
```

Start with a small bbox to verify connectivity before larger requests.

### 3. Health Check

```powershell
# run_in_terminal (mode=sync)
python "Agent_skill_dev\skill_catalog\external-data-connectors\cudem-bathy\scripts\check_download_health.py" --request request.json --run-dir runs/case --output runs/case/health_check.json --plots-dir runs/case/health_plots
```

Then:
- `read_file` on `health_check.json` for the report
- `view_image` on any diagnostic plots in `health_plots/`
- Surface to user only when: missing coverage, all-NaN fields, coverage <95%, obvious gaps

## Copilot Tool Integration

| Step | Tool |
|------|------|
| Run any script | `run_in_terminal` (mode=sync) |
| Read JSON results | `read_file` |
| View diagnostic plots | `view_image` |
| Ask user about storage routing | `vscode_askQuestions` |
| Download large data to Kestrel | Use kestrel-hpc-copilot bridge |

## Implementation Rules

- Generic data connector — do not produce model-specific output by default.
- Keep downloads source-bounded and request-bounded.
- Do Python plotting and health reports locally.
- If Kestrel is needed for staging, download compact products back for local checks.
- Do not store credentials in scripts, logs, or metadata.
