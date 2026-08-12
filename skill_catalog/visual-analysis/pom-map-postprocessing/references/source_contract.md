# POM curvilinear map source contract

## Supported inputs

The loader accepts NOAA POM/NYOFS NetCDF classic cycle aggregates and compact `nyofs-fetcher` products. A valid input provides:

- matching two-dimensional longitude and latitude arrays;
- a same-shape wet mask where only `mask == 1` is water;
- an hourly or six-minute decodable time coordinate;
- optional same-shape depth and one-dimensional monotonic sigma coordinates;
- requested fields whose spatial shape matches the native grid.

Current live raw NYOFS fields conventionally use:

| Item | Shape | Meaning |
| --- | --- | --- |
| `lon`, `lat`, `mask`, `depth` | `(ny, nx)` | Native curvilinear geometry |
| `sigma` | `(sigma)` | Seven positive-down POM sigma points |
| `zeta`, `air_u`, `air_v` | `(time, ny, nx)` | Elevation and earthward wind |
| `u`, `v` | `(time, sigma, ny, nx)` | Earthward/northward water velocity |

Compact v1 products use `y,x` instead of `ny,nx`, may include ready-made fields such as `current_speed_surface`, `u_depth_average`, and `wind_speed`, and declare `source_grid` plus `vector_components=earth_relative`.

## Time handling

Decode CF `units since ...` numbers or NOAA character timestamps. Infer only hourly (3,600-second) or six-minute (360-second) cadence. Snap each record to the nearest epoch-aligned cadence point only when its absolute adjustment is at most 60 seconds. Preserve the original timestamp and signed normalized-minus-original offset.

Sort normalized records and remove duplicate times deterministically. Prefer the record from the source aggregate with the earlier first timestamp; this retains the preceding cycle's terminal record at a nowcast boundary. Crop with an inclusive start and exclusive end.

Static map `--time` selection is exact after normalization. Use `--time-index` only when index-based inspection is intentional.

## Vertical views

Resolve `surface` as the sigma point closest to zero, `near_surface` as the second closest, and `bottom` as the farthest. `index:N` refers to the source zero-based sigma index.

For `depth_average`, calculate trapezoidal sigma-point weights from `abs(diff(sigma))`, half-weight both endpoints, and renormalize over finite layers at every wet cell. Average `u` and `v` independently before calculating current speed.

## Variable resolution

For a requested field and layer, prefer a ready-made suffixed field such as `current_speed_surface`. Otherwise use the unsuffixed source field and apply the requested sigma view. Derive:

- `current_speed = sqrt(u^2 + v^2)`;
- `wind_speed = sqrt(air_u^2 + air_v^2)`.

Require paired vector dimensions. Accept exact CF eastward/northward standard names, an explicit earth-relative global declaration, or unambiguous eastward/northward long names. Reject components described as grid/xi/eta relative and any pair whose earth orientation cannot be established. Do not silently rotate or destagger.

## Map and manifest contract

Render the masked native grid with `pcolormesh` or `contourf` and correct geographic degree aspect. Choose finite 2nd/98th percentile limits unless explicit bounds are provided. Static current/wind quivers are optional and downsampled by the requested stride.

Write `pom_map_manifest_v1` JSON beside the PNG. It records input size/hash/mtime, request options, normalized and original time, offset, grid geometry hash, source field resolution, fixed limits, finite wet coverage, actual rendered arrow count, output size/hash, and status. See [map_manifest.schema.json](map_manifest.schema.json) for the machine-readable core shape.

Hard failures include missing geometry/time/field, nonmonotonic sigma, geometry drift during concatenation, unproven vectors, an unavailable exact time, invalid limits, empty wet mask, or an all-NaN selected frame.
