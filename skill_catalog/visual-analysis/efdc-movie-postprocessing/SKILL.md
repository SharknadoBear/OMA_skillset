---
name: efdc-movie-postprocessing
description: Create fixed-color-scale GIF animations from NOAA EFDC curvilinear NetCDF fields. Use for native SJROFS aggregate files or efdc_compact_fields_v1 products when animating salinity, current speed, wind speed, elevation, sigma views, or depth averages on independent mask==5 wet-cell footprints with provenance and frame QA.
---

# EFDC Movie Postprocessing

Create scalar, fixed-scale Pillow GIFs without MP4 or movie quivers. Read [references/output_contract.md](references/output_contract.md) before accepting a new EFDC convention.

## Workflow

1. Inspect repeated inputs:

   ```powershell
   python scripts/efdc_movie_postprocessing.py inspect --input FILE --input FILE2 --output inspection.json
   ```

2. Create the bounded animation:

   ```powershell
   python scripts/efdc_movie_postprocessing.py gif --input FILE --input FILE2 `
     --variable current_speed --layer depth_average `
     --start 2026-07-20T00:00:00Z --end-exclusive 2026-07-21T00:00:00Z `
     --fps 4 --output movie.gif --report movie_manifest.json
   ```

3. Use the default full-series 2nd/98th percentile limits or pass both `--vmin`/`--vmax`. Every frame uses exactly one fixed range.
4. Reuse manifest limits for a matching midpoint map.
5. Verify frame count, distinct frames, minimum wet coverage, output SHA-256, and `temporary_frames_cleaned`.

## Scientific and rendering contract

- Exact source `mask == 5` defines water; compact `wet_mask` must agree. Dry zero coordinates and the negative padding sentinel never influence geometry.
- Independent wet-cell polygons use immediate wet logical neighbors only. This prevents cross-land strips in narrow tributaries.
- `surface`, `near_surface`, and `bottom` resolve by positive-down sigma value, independent of storage order.
- `depth_average` uses bed-closed EFDC layer-top weights. Average vector components before calculating speed.
- Native current and wind components require collocated earth-relative metadata. No destaggering or rotation is allowed.
- Normalize time within 60 seconds, concatenate, sort, and deduplicate; reject geometry/schema drift, finite hydrodynamic fields outside wet cells, all-NaN frames, input/output collisions, and empty windows. Atmospheric source fields may cover dry cells but are clipped to active water before rendering.

Run `python scripts/selftest_efdc_movie_postprocessing.py` and the shared-core hash parity check after changes.
