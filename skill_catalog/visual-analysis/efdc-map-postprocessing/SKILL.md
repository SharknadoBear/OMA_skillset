---
name: efdc-map-postprocessing
description: Inspect and render QA-ready static maps from NOAA EFDC curvilinear NetCDF fields. Use for native SJROFS aggregate files or efdc_compact_fields_v1 products when mapping salinity, current speed, wind speed, elevation, sigma views, depth averages, or static earth-relative current/wind quivers without bridging narrow wet branches across dry packed coordinates.
---

# EFDC Map Postprocessing

Render one scientifically selected EFDC record with a provenance manifest. Read [references/source_contract.md](references/source_contract.md) before accepting a new model family or mask/vertical convention.

## Workflow

1. Inspect every input before plotting:

   ```powershell
   python scripts/efdc_map_postprocessing.py inspect --input FILE --output inspection.json
   ```

2. Confirm `mask == 5` defines active water, sigma is positive-down layer-top fractions, and paired native vectors are collocated and CF-declared east/north.
3. Render the independent wet-cell polygons (default):

   ```powershell
   python scripts/efdc_map_postprocessing.py map --input FILE `
     --variable salinity --time 2026-07-20T12:30:00Z --layer surface `
     --limits-scope series --output map.png --report map_manifest.json
   ```

4. Reuse movie `vmin`/`vmax` exactly when matching a benchmark frame. Supply both `--vmin` and `--vmax`.
5. For surface currents add `--quiver current --quiver-stride 8`. Wind quivers use `--quiver wind`.

Repeat `--input` for native aggregates. Time is normalized only within 60 seconds of the half-hour/hourly or six-minute cadence, then deduplicated and sorted. Exact UTC selection is fail-closed.

## Rendering contract

- Default `--style wet_cells` builds one polygon per wet logical cell using only immediate wet neighbors. Dry zeros and negative padding never influence footprints.
- `pcolormesh` is an explicit diagnostic and is rejected when any cell is masked because center-coordinate corner inference can cross land.
- `contourf` is an explicit diagnostic, not the benchmark convention.
- `depth_average` uses `efdc_layer_top_sigma_with_bed_edge_1`: sort layer tops, append the bed edge `1`, difference, and map weights back to source order. Components are averaged separately before speed.
- Native `u/v` are used directly only when metadata proves collocated earth-relative components. Do not destagger or rotate.
- Reject geometry drift, ambiguous mask codes, finite hydrodynamic values outside source `mask == 5`, unpaired vectors, unsafe output collisions, and all-NaN wet frames. Atmospheric fields may cover dry packed cells in the source, but visualization always clips them to active water.

The manifest schema is [references/map_manifest.schema.json](references/map_manifest.schema.json). Run `python scripts/selftest_efdc_map_postprocessing.py` after changes.
