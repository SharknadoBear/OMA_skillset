---
name: roms-movie-postprocessing
description: Create fixed-color-scale GIF animations from native staggered ROMS C-grid fields or roms_compact_fields_v1 products. Use when Codex needs CBOFS/DBOFS or other ROMS salinity, elevation, current-speed, or scalar time-series movies with normalized and deduplicated timestamps, dynamic sigma views, Vtransform-aware depth averages, U/V destaggering and rotation, and provenance-preserving GIF manifests.
---

# ROMS Movie Postprocessing

Create scalar GIF evidence from raw native ROMS files or compact connector products. Keep every frame on one color scale so visible changes reflect the data.

## Workflow

1. Inspect the input set and confirm grid, vertical metadata, variable availability, and requested time coverage.
2. Pass all required raw hourly objects or one compact NetCDF to `gif`. The loader verifies geometry, normalizes near-nominal timestamps, sorts records, and removes duplicate times deterministically.
3. Select a scalar and layer. Prefer explicit limits established by a companion map; otherwise use one full-series percentile range.
4. Verify the report's frame count, distinct rendered-frame count, geometry hash, resolution method, coverage, fixed limits, temporary cleanup, and output hash.

## Commands

```powershell
python scripts/roms_movie_postprocessing.py inspect `
  --input cbofs_fields.nc `
  --output inspection.json

python scripts/roms_movie_postprocessing.py gif `
  --input cbofs_fields.nc `
  --variable current_speed `
  --layer surface `
  --start 2026-07-20T00:00:00Z `
  --end-exclusive 2026-07-21T00:00:00Z `
  --fps 4 --quantiles 2 98 `
  --output current_speed_surface.gif `
  --report current_speed_surface.json
```

Repeat `--input` or pass several paths after one `--input` for native hourly files. The end time is exclusive.

## Guardrails

- Resolve `surface`, `near_surface`, and `bottom` from `s_rho`; use `index:N` only for an intentional native index.
- For `depth_average`, use actual W-level thicknesses from `h`, `zeta`, `s_w`, `Cs_w`, `hc`, and `Vtransform`.
- Derive current speed only after separately reducing native U/V, applying U/V masks, destaggering to rho points, and rotating with `angle`.
- Reject geometry or vertical-schema drift, missing rotation metadata, unpaired vectors, invalid thickness closure, and all-NaN frames.
- Keep one explicit or full-series quantile scale for every frame. Do not create MP4 output or animated quivers in v1.
- Store GIFs and reports outside this skill package.

Read [output_contract.md](references/output_contract.md) when diagnosing raw/compact field resolution or manifest semantics, and validate reports against [movie_manifest.schema.json](references/movie_manifest.schema.json). `scripts/roms_output.py` and `scripts/roms_map_tools.py` are byte-identical to the map skill's copies; preserve parity.

## Programmatic API and validation

Import `create_gif` from `scripts/roms_movie_postprocessing.py`. Import raw/compact loaders from `scripts/roms_output.py`.

Run:

```powershell
python scripts/selftest_roms_movie_postprocessing.py
```

Then run skill-creator `quick_validate.py` and compile all scripts with bytecode outside the skill tree.
