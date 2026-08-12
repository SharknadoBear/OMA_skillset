# ROMS map source contract

## Accepted inputs

The loader accepts either:

- native ROMS C-grid NetCDF files containing `lon_rho`, `lat_rho`, `mask_rho`, `h`, `angle`, `ocean_time`, and requested source fields; or
- `roms_compact_fields_v1` NetCDF products preserving rho geometry and ROMS vertical metadata with derived names such as `salinity_surface`, `eastward_velocity_surface`, `northward_velocity_surface`, and `current_speed_surface`.

Repeat raw files may contain one or several records. Geometry, masks, angle, bathymetry, sigma coordinates, stretching curves, `hc`, and transform metadata must remain identical. Every rho/U/V mask present in a source must contain only finite binary `0` or `1` values; other values are rejected rather than coerced to land.
Variables may store their legal NetCDF dimensions in any order. The loader identifies named time, vertical, eta, and xi axes and moves them to canonical ROMS order before selection, weighting, masking, or broadcast; unrecognized or ambiguous axes are rejected.

The angle variable must explicitly advertise `rad`, `radian`, or `radians` and must identify the XI-axis-to-east convention through `standard_name=grid_angle_of_rotation_from_east_to_y` or an equivalent unambiguous `long_name`. Wet-cell values must be finite and within `[-2*pi, 2*pi]`. The loader canonicalizes this as units `radians` and convention `xi_axis_counterclockwise_from_east`; both values participate in the geometry hash and reports.

## Time and masks

Decode CF numeric or character time coordinates. Snap hourly or six-minute records only when the difference is at most 60 seconds; preserve the original timestamp and adjustment. Sort on normalized UTC and retain the earlier source for duplicate timestamps. Apply source fill values and require `mask_rho == 1`; apply `mask_u` and `mask_v` before destaggering, inferring them from adjacent rho masks only when absent.

## Vertical transforms

For `Vtransform = 1`:

```text
z0 = (s - Cs) * hc + Cs * h
z  = z0 + zeta * (1 + z0 / h)
```

For `Vtransform = 2`:

```text
z0 = (hc * s + h * Cs) / (hc + h)
z  = zeta + (zeta + h) * z0
```

Compute W-level depths from `s_w` and `Cs_w`, use `abs(diff(z_w))`, and require their sum to close to `h + zeta`. Renormalize weights over finite wet layers. Resolve surface as the `s_rho` value closest to zero and bottom as the farthest.

## Current derivation

Native `u` and `v` are grid-relative on different staggered grids. Reduce them on their native grids, then use finite adjacent means with one-sided domain boundaries to place both on rho points. Rotate using angle in radians from the ROMS XI axis to east:

```text
east  = u_rho * cos(angle) - v_rho * sin(angle)
north = u_rho * sin(angle) + v_rho * cos(angle)
speed = hypot(east, north)
```

Never label native U/V as earth-relative. Reject missing angle or unpaired components.

Precomputed earth/north fields are accepted only from an exact `roms_compact_fields_v1` product with a recognized `derived_vector_reference`, `vector_reference`, or controlled `vector_provenance` value declaring earth-relative rho-grid vectors, plus non-empty `velocity_processing`. A stored speed must be paired with east/north and agree with `hypot(east,north)` within floating-point tolerance. Earthward names in raw or unclassified files are rejected as ambiguous. Precomputed layer-suffixed scalar fields likewise require the exact compact schema.

## Map manifest

`roms_map_manifest_v1` records input hashes, requested and source variables, the selected source record, normalized/source time, grid hash, vertical-transform and vector-resolution methods, strictly positive wet coverage, fixed or robust limits, quiver settings, and PNG hash. See `map_manifest.schema.json` for the machine-checkable shape.

## Wet-cell rendering convention

The default renderer does not pass rho-center coordinates to `pcolormesh`. ROMS grids
may pack unrelated dry-coordinate blocks next to narrow wet reaches in index space;
center-based corner inference can therefore stretch a nominally wet face across a dry
seam. Instead, render each finite `mask_rho == 1` cell as an independent footprint.
Measure XI and ETA size only between immediate wet-neighbor centers, orient the footprint
with the verified ROMS `angle`, and use the median wet-grid aspect ratio only where a
one-cell-wide boundary lacks a neighbor along one axis. Never consult a dry center when
constructing a wet footprint. Reports record the rule, fallback count, and maximum
footprint span. `contourf` is an intentional interpolation option; legacy center-coordinate
`pcolormesh` is allowed only when every coordinate cell is wet.
