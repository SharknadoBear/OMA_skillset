---
name: roms-map-postprocessing
description: Inspect and render QA-ready static maps from native staggered ROMS C-grid fields or roms_compact_fields_v1 products. Use when Codex needs CBOFS/DBOFS or other ROMS salinity, elevation, earth-relative current, or scalar maps with dynamic sigma views, Vtransform-aware depth averages, U/V destaggering and rotation, fixed or robust color limits, optional current quivers, and provenance-preserving PNG manifests.
---

# ROMS Map Postprocessing

Render scalar fields on the native ROMS rho grid. Let the bundled loader handle C-grid staggering, masks, sigma orientation, `Vtransform`, and grid-to-earth velocity rotation.

## Workflow

1. Inspect all raw hourly files or one compact connector product:

   ```powershell
   python scripts/roms_map_postprocessing.py inspect `
     --input cbofs.t00z.20260720.fields.n006.nc cbofs.t06z.20260720.fields.n001.nc `
     --output inspection.json
   ```

2. Require consistent geometry and vertical metadata, unique normalized times, finite wet cells, and the intended variables. Read [source_contract.md](references/source_contract.md) before interpreting unfamiliar outputs.

3. Render one exact timestamp using full-series limits when a matching movie is planned:

   ```powershell
   python scripts/roms_map_postprocessing.py map `
     --input cbofs_fields.nc `
     --variable current_speed `
     --time 2026-07-20T12:00:00Z `
     --layer surface `
     --limits-scope series `
     --quiver current --quiver-stride 8 `
     --output current_speed_surface.png `
     --report current_speed_surface.json
   ```

4. Open the PNG and verify its manifest: input hashes, normalized and source time, grid hash, field resolution, vertical transform, destagger/rotation methods, wet coverage, color limits, quiver count, and output hash.

Repeat `--input` or pass several paths after one `--input` for native hourly files.

## ROMS field rules

- Use `surface`, `near_surface`, `bottom`, `depth_average`, or `index:N`. Resolve semantic layers from `s_rho`, never a fixed index.
- Resolve `salinity` to compact `salinity_<view>` or native `salt`.
- Resolve `current_speed` from a compact speed field or paired earth-relative components; otherwise reduce native U/V separately, destagger them to rho points, rotate with `angle`, and then calculate speed.
- For native depth averages, compute W-level depths with the advertised `Vtransform` and time-varying `zeta`, use `abs(diff(z_w))`, and renormalize finite wet layers.
- Apply rho/U/V masks and source fill values. Reject missing angles, unpaired vectors, geometry drift, invalid thickness closure, unavailable exact times, and all-NaN frames.
- Do not interpret native `u`/`v` as earth-relative or draw them directly on the rho grid.

## Rendering choices

- Keep `--style pcolormesh` for native-grid evidence; use `contourf` only intentionally.
- Use the default 2nd/98th percentiles or pass both `--vmin` and `--vmax` for fixed comparisons.
- Use `--limits-scope series` so a midpoint PNG can share a movie's full-series scale.
- Use quivers only for static earth-relative currents and increase the stride until arrows are readable.
- Store PNGs and manifests in the project run directory, not in this skill package.

## Programmatic API

Import `load_scalar_series`, `load_current_series`, and `inspect_inputs` from `scripts/roms_output.py`. Import `quantile_limits`, `plot_roms_scalar`, or `save_roms_scalar_map` from `scripts/roms_map_tools.py`. Those two modules are intentionally byte-identical to the copies in `roms-movie-postprocessing`; preserve parity.

## Validation

Run:

```powershell
python scripts/selftest_roms_map_postprocessing.py
```

Then run skill-creator `quick_validate.py` and compile all scripts with bytecode outside the skill tree.
