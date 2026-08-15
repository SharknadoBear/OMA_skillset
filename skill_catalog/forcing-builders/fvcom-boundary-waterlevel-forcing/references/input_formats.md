# Input formats

The builder consumes one already-combined water-level source per run. It normalizes recognized units to metres before spatial or temporal interpolation.

## NetCDF

Supported layouts are:

- `water_level(time)` for an explicitly broadcast series.
- `elevation(time, nobc)` with an integer `obc_nodes(nobc)` coordinate.
- `water_level(time, point)` with `lon(point)` and `lat(point)` station coordinates.
- `ssh(time, lat, lon)` with one-dimensional coordinates.
- `zeta(time, y, x)` with two-dimensional `lon(y, x)` and `lat(y, x)` coordinates.

Common value names are discovered automatically: `elevation`, `water_level`, `waterlevel`, `ssh`, `zeta`, and `surf_el`. Override ambiguous names with `--value-var`, `--time-var`, `--lon-var`, `--lat-var`, or `--node-id-var`.

Use CF time units, FVCOM `Times`, or FVCOM `Itime` plus `Itime2`. When several FVCOM representations exist, the reader prefers `Times`, then integer MJD plus milliseconds, then numeric `time`. This prevents float32 MJD quantization from corrupting exact timestamps.

The water-level variable must declare metres, centimetres, or millimetres, unless `--units` supplies the correct units.

## Tidy CSV

CSV requires `--units`. Use ISO-8601 timestamps and one of these row layouts:

```text
time,water_level
2020-01-01T00:00:00Z,0.12
```

```text
time,node_id,water_level
2020-01-01T00:00:00Z,101,0.12
```

```text
time,station_id,lon,lat,water_level
2020-01-01T00:00:00Z,A,-122.5,37.7,0.12
```

Every station must have a fixed geographic coordinate. Every time/station or time/node pair must be unique. Wide station tables are intentionally excluded; convert them to tidy rows so identities and coordinates remain explicit.

## Spatial behavior

- Map node-indexed input by exact FVCOM node id and reorder it to match the selected OBC.
- Apply inverse-distance weighting to station collections.
- Apply triangulated linear interpolation to geographic grids and curvilinear fields.
- Use bounded nearest-wet fallback only when linear interpolation is outside the valid stencil or encounters missing values.
- Default the fallback bound to twice the median source spacing; use `--max-nearest-km` to set a reviewed project limit.
- Handle longitude wrapping around the dateline in local geographic-distance calculations.

## Time and datum behavior

- Preserve regular native timestamps when no target grid is supplied.
- Require all three target arguments (`--start`, `--end`, and `--dt-seconds`) for resampling.
- Reject extrapolation and gaps larger than `--max-gap-factor` times the native cadence.
- Round canonical timestamps to milliseconds because FVCOM `Itime2` stores milliseconds since midnight.
- Record `--datum` as provenance. The builder never changes a vertical datum or estimates an offset.
