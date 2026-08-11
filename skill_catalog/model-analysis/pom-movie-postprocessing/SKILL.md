---
name: pom-movie-postprocessing
description: Create fixed-color-scale GIF animations from NOAA POM and NYOFS curvilinear NetCDF fields. Use when Codex needs to inspect one or more POM output files, normalize and deduplicate their timestamps, select surface, near-surface, bottom, depth-average, or explicit sigma-layer scalar fields, and render provenance-preserving time-series movies on the native wet grid.
---

# POM Movie Postprocessing

Create scalar GIF evidence from native POM curvilinear fields without interpolating the source grid. Keep every frame on one color scale so apparent changes reflect the data rather than per-frame autoscaling.

## Workflow

1. Inspect unfamiliar inputs with the `inspect` command before rendering. Confirm the grid, time coverage, available variables, and layer selectors.
2. Supply all cycle aggregates needed for the requested window to `gif`. The loader verifies geometry, sorts records by normalized UTC time, and resolves duplicate timestamps deterministically.
3. Select a scalar variable and layer. Use `current_speed` or `wind_speed` to derive magnitude from paired earthward components when a ready-made compact variable is absent.
4. Prefer explicit `--vmin` and `--vmax` when a related static map established acceptance limits. Otherwise use one full-series quantile range for the complete GIF.
5. Review the JSON report and verify `frame_count`, `duplicate_times_removed`, `geometry_sha256`, `fixed_color_limits`, frame coverage, and output SHA-256.

## Commands

Run from any working directory:

```powershell
python scripts/pom_movie_postprocessing.py inspect `
  --input nyofs_coarse_fields.nc `
  --output inspection.json

python scripts/pom_movie_postprocessing.py gif `
  --input nyofs_coarse_fields.nc `
  --variable current_speed `
  --layer depth_average `
  --start 2026-07-20T00:00:00Z `
  --end-exclusive 2026-07-21T00:00:00Z `
  --fps 4 `
  --output current_speed_depth_average.gif `
  --report current_speed_depth_average.json
```

Repeat `--input` for raw cycle aggregates. Use `--quantiles 2 98` for robust automatic limits or provide both `--vmin` and `--vmax`. The end time is exclusive.

## Guardrails

- Treat `lon`, `lat`, mask, and grid shape as immutable across inputs; stop on drift rather than resampling.
- Preserve POM `u` and `v` as earthward components only when metadata and colocated dimensions support that interpretation. Do not rotate or destagger silently.
- Interpret surface as the sigma value closest to zero and bottom as the farthest from zero; never assume array order.
- Use trapezoidal sigma-point weights for on-demand depth averages and renormalize over finite wet layers.
- Prefer the preceding file's terminal station record when identical boundary timestamps occur; for field-cycle duplicates, prefer the earlier source supplied by deterministic source order.
- Do not create MP4 output or animated quivers in v1.

Read [references/output_contract.md](references/output_contract.md) when diagnosing raw/compact variable resolution or report semantics. Execute `scripts/selftest_pom_movie_postprocessing.py` after modifying any script.
