# ROMS movie output contract

## Input resolution

Accept repeated native ROMS C-grid files or one `roms_compact_fields_v1` product. Preserve the native rho grid; do not mosaic or interpolate. Geometry and vertical metadata must be identical across files. Rho/U/V masks must be finite binary `0`/`1` arrays.

The angle variable must explicitly use radian units and unambiguously describe the ROMS XI-axis-to-east convention through the live ROMS standard name or equivalent `long_name`; wet values outside `[-2*pi, 2*pi]` are rejected. Reports and the geometry hash carry canonical units `radians` and convention `xi_axis_counterclockwise_from_east`.

Resolve `salinity` from `salinity_<view>` or native `salt`. Resolve current components from compact earth-relative rho-grid fields when available. Otherwise:

1. select or depth-average native U and V separately;
2. apply native U/V wet masks;
3. destagger each component to rho points using finite adjacent means and one-sided boundaries;
4. rotate with ROMS `angle`; and
5. calculate speed from the rotated east/north pair.

Depth averages use W-level thicknesses from the advertised `Vtransform`, `h`, time-varying `zeta`, `s_w`, `Cs_w`, and `hc`. Thicknesses must close to `h + zeta`.

Precomputed current fields require the exact compact schema, explicit recognized earth-relative rho-grid provenance, and non-empty `velocity_processing`. Stored speed must be paired with east/north and match `hypot(east,north)` within floating-point tolerance. Raw or unclassified east/north names are ambiguous and rejected. Layer-suffixed precomputed scalar fields also require the exact compact schema.

## Time behavior

Decode CF time metadata, normalize only adjustments within 60 seconds of an hourly or six-minute cadence, retain raw timestamps and offsets, sort, and deterministically remove duplicate normalized times. Apply inclusive `start` and exclusive `end-exclusive` after normalization.

## GIF behavior

- Render every selected scalar frame on the native rho grid.
- Use one explicit or full-series percentile color range for all frames.
- Reject all-NaN wet frames and report finite wet coverage for every frame.
- Write a looping Pillow GIF with the requested FPS.
- Stage and verify the complete GIF before atomically publishing it. If Pillow collapses identical frames, fail without publishing a partial output.
- Do not render quivers or MP4 in v1.
- Remove temporary PNG frames before returning.

## `roms_movie_manifest_v1`

The report contains:

- input sizes, SHA-256 hashes, time coverage, and grid hash;
- requested scalar, layer, and time window;
- resolved source fields, `Vtransform`, destaggering, rotation, and weight method;
- unique time and duplicate-removal counts;
- one fixed color range and its method;
- per-frame source time, normalized time, coverage, source record, and rendered-frame hash;
- GIF frame count, distinct rendered-frame count, pixel size, duration, byte size, and SHA-256;
- confirmation that movie quivers are disabled and temporary frames were cleaned.

See `movie_manifest.schema.json` for the machine-checkable report shape.
