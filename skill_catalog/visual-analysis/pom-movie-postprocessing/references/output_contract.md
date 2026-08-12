# POM movie input and report contract

## Supported inputs

The scripts accept NOAA POM/NYOFS raw cycle aggregates and compact products produced by `nyofs-fetcher`.

- Geometry: two-dimensional `lon(y,x)` and `lat(y,x)`, plus optional `mask(y,x)` and `depth(y,x)`.
- Time: numeric `time(time)` with CF `units`/`calendar`, NOAA `Times`, or common Modified Julian Day encodings. Near-nominal hourly or six-minute values may be normalized only within 60 seconds; the report retains raw offsets.
- Vertical coordinate: one-dimensional `sigma(sigma)` or a recognized equivalent. Surface is closest to zero; bottom is farthest from zero.
- Compact derived variables: names such as `current_speed_surface`, `current_speed_sigma_3`, and `current_speed_depth_average` are selected directly.
- Raw components: colocated `u(time,sigma,y,x)` and `v(time,sigma,y,x)` may produce current speed; `air_u(time,y,x)` and `air_v(time,y,x)` may produce wind speed.

Land cells are values where `mask != 1`. Source fill values and non-finite samples remain masked. Inputs with differing grid shape, coordinates, mask, depth, sigma, or component dimensions are rejected.

## Layer selectors

- `surface`: sigma closest to zero.
- `near_surface`: the second-closest sigma value to zero.
- `bottom`: sigma farthest from zero.
- `depth_average`: finite-aware trapezoidal sigma-point average.
- `index:N`: explicit zero-based sigma index.

For a compact variable already suffixed with a layer, the requested selector must resolve to that suffix. Two-dimensional time-varying fields such as `zeta` and `wind_speed` ignore vertical selection.

## GIF report

The report is versioned as `pom_movie_manifest_v1`. It records source paths and SHA-256 hashes, the resolved variable/components, layer, original and normalized UTC timestamps, duplicate removal, grid fingerprint, finite wet coverage per frame, fixed color limits and their method, rendering options, temporary-frame cleanup status, GIF frame count, duration, dimensions, and output SHA-256.

Automatic limits use the requested full-series percentiles over finite wet values. Constant fields receive a small symmetric expansion so every frame still has a valid fixed normalization. Explicit limits require `vmin < vmax`.
