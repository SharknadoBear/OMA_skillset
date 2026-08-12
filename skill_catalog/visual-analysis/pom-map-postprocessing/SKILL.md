---
name: pom-map-postprocessing
description: Inspect and render QA-ready static maps from NOAA POM and NYOFS curvilinear NetCDF fields. Use when Codex needs native-grid maps from raw NYOFS cycle aggregates or nyofs-fetcher compact products, including dynamic sigma-layer views, derived current or wind speed, robust or explicit color limits, optional earth-relative current/wind quivers, and provenance-preserving PNG manifests.
---

# POM Map Postprocessing

Render masked POM fields directly on their native two-dimensional curvilinear grid. Use the bundled CLI instead of assuming FVCOM topology, ROMS staggering, regular-grid geometry, or vector rotation.

## Workflow

1. Inspect the input before plotting:

   ```powershell
   python scripts/pom_map_postprocessing.py inspect `
     --input nyofs_coarse_fields.nc `
     --output inspection.json
   ```

2. Confirm `status: pass`, the intended coarse/fine grid, time coverage, sigma ordering, wet-cell count, and requested variables. Read [source_contract.md](references/source_contract.md) when interpreting raw fields, derived views, or vector metadata.

3. Render exactly one normalized timestamp or record index:

   ```powershell
   python scripts/pom_map_postprocessing.py map `
     --input nyofs_coarse_fields.nc `
     --variable current_speed `
     --time 2026-07-20T12:00:00Z `
     --layer surface `
     --limits-scope series `
     --quiver current `
     --quiver-stride 8 `
     --output current_speed_surface.png `
     --report current_speed_surface.json
   ```

4. Open the PNG and review the manifest. Require the selected time/layer, source-variable resolution, grid hash, color limits, finite wet coverage, quiver count, input/output hashes, and output byte count to agree with the request.

## Field and layer rules

- Use `surface`, `near_surface`, `bottom`, `depth_average`, or `index:N` for sigma-dependent variables. Resolve semantic layers from sigma values, not fixed indices.
- Request `current_speed` to use a ready-made suffixed field when present or calculate `sqrt(u^2 + v^2)` from paired earth-relative components.
- Request `wind_speed` to use a ready-made field or calculate it from paired eastward/northward wind components.
- Apply `mask == 1` and source fill values. Never draw land or invalid wet cells as real values.
- Reject geometry drift, missing vector pairs, ambiguous grid-relative vectors, unavailable exact times, and all-NaN frames.
- Treat coarse and nested-fine grids as separate products. Do not mosaic, interpolate, destagger, or rotate them in this skill.

## Rendering choices

- Keep `--style pcolormesh` for faithful native-cell display; use `--style contourf` only when a smoothed visual presentation is intentional.
- Keep the default 2nd/98th percentile limits for a representative frame. Use `--limits-scope series` to make a midpoint map share the full-series scale used by a movie.
- Pass both `--vmin` and `--vmax` when an externally fixed comparison range is required.
- Use `--quiver current` or `--quiver wind` only for static vector evidence. Increase `--quiver-stride` until arrows remain readable.
- Keep generated PNGs and manifests in the project run/evidence directory, not inside the skill package.

## Programmatic API

Import `load_scalar_series` and `inspect_inputs` from `scripts/pom_output.py` for raw/compact loading and concatenation. Import `quantile_limits`, `plot_pom_scalar`, or `save_pom_scalar_map` from `scripts/pom_map_tools.py` for deterministic plotting. These two modules are intentionally byte-identical to the copies in `pom-movie-postprocessing`; preserve parity when changing shared behavior.

## Validation

Run the offline suite after changes:

```powershell
python scripts/selftest_pom_map_postprocessing.py
```

Then run the skill-creator `quick_validate.py`, compile scripts with bytecode outside the skill tree, and forward-test the installed copy on a fresh compact NYOFS file.
