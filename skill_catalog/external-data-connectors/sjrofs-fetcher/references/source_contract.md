# SJROFS operational AWS and NCEI source contract

## Source policy and archive roots

Requests normalize to `sjrofs_request_v2` with `aws_then_ncei`, `aws_only`, or `ncei_only`. V1 migrates to v2 with the default policy recorded. AWS uses anonymous ListObjectsV2 under `sjrofs/netcdf/`, accepting current daily `YYYY/MM/DD/`, current monthly `YYYYMM/`, and legacy monthly layouts. NCEI uses bounded `YYYY/MM/` prefixes below:

`operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access/st-johns-river-operational-forecast-system-sjrofs/`

Current names look like `sjrofs.t05z.20200101.fields.nowcast.nc`; legacy names look like `nos.sjrofs.fields.nowcast.20200101.t05z.nc`. Prefer current names over legacy, then daily layout over monthly. Fail conflicting equal-rank aliases. After scientific equivalence, AWS outranks NCEI. Treat size, opaque ETag, and Last-Modified as provider-local identity; bind the reviewed plan and source-isolated cache sidecar to all three plus key, URL, and SHA-256.

AWS is always discovered completely before automatic fallback. Listing/network failure stops; it is not missing coverage. NCEI supports native field nowcasts and station nowcast/forecast aggregates. It does not support field forecasts. Never substitute another product, guidance, model, or grid. Downloaded NetCDF time is authoritative.

## Products and cadence

SJROFS cycles are 05, 11, 17, and 23 UTC. A native nowcast field aggregate contains six hourly records at half-hour timestamps: cycle minus 4.5 hours through cycle plus 0.5 hour. A field forecast contains 49 hourly points beginning at cycle plus 0.5 hour and continuing 48 hours. Station nowcasts use six-minute points through cycle plus 0.5 hour; station forecasts use the same six-minute cadence for 48 hours. Planning uses these discrete points, never continuous overlap at an excluded interval edge.

Normalize a decoded field time only when within 60 seconds of its nearest `HH:30`; normalize stations to the nearest six-minute point on the UTC epoch under the same tolerance. Preserve decoded and normalized timestamps, numeric source time, units/calendar, and offsets. At station boundary duplicates, the preceding cycle terminal record wins.

## Native EFDC fields

Current fields are NetCDF-3 classic on a sparse 188×105 curvilinear grid with six collocated positive-down sigma layers. Discover dimensions dynamically but require stable geometry across concatenated objects. Core variables include:

| Variable | Dimensions | Contract |
|---|---|---|
| `lon`, `lat`, `mask`, `depth` | `(ny,nx)` | native geometry |
| `sigma` | `(sigma)` | EFDC layer-top fractions |
| `zeta`, `air_u`, `air_v` | `(time,ny,nx)` | collocated 2-D fields |
| `salt`, `u`, `v`, `temp` | `(time,sigma,ny,nx)` | collocated 3-D fields |

Exactly source `mask == 5` is active water (2,210 cells in the current grid). Zero-valued dry coordinates and the known negative padding sentinel are inactive and must never define footprints. Preserve the raw mask and emit a derived binary wet mask. Fail if a future nonnegative active code is ambiguous or hydrodynamic `zeta/salt/u/v/temp` has valid values outside `mask == 5`. Atmospheric `air_u/air_v` legitimately cover dry cells in live files; do not infer wetness from them, and mask them to active water before compact output or wind derivation.

Require `u` and `v` to share dimensions and advertise `eastward_sea_water_velocity` and `northward_sea_water_velocity` (or unambiguous equivalent long names). They are already earth-relative and collocated; apply neither ROMS destaggering nor vector rotation.

## Vertical processing and compact output

Sigma values are positive-down layer-top fractions. Sort them, require a first value near zero and unique values in `[0,1)`, append the bed edge `1.0`, and compute positive layer fractions with `diff`. Map weights back to source order. The provenance method is exactly `efdc_layer_top_sigma_with_bed_edge_1`. For incomplete profiles, renormalize over finite layers. Average components independently, then calculate speed. When `zeta` is requested, require positive finite `depth + zeta` at active cells.

Write compressed `compact/sjrofs_fields.nc` using `efdc_compact_fields_v1`. Preserve lon, lat, source mask, derived wet mask, depth, sigma, normalized/raw times and offsets, source archive/key/cycle provenance, requested source variables, and derived `salinity_<view>`, `eastward_velocity_<view>`, `northward_velocity_<view>`, and `current_speed_<view>`. Surface is minimum sigma, near-surface the second minimum, bottom maximum, and an integer view is the native source index.

## Transfer and health gates

Fetch accepts only a reviewed `sjrofs_download_estimate_v2`. Validate plan/request/object hashes, re-list the exact remote source identity, and recheck that local free space is greater than four times exact bytes. Use resumable `.part` transfers with validated Range/Content-Range, atomic rename, and source-isolated caches.

Health fails on broken plan/source/sidecar bindings, bytes/hashes/ETag/Last-Modified mismatch, incomplete or nonunique cadence, geometry/mask/sigma drift, nonpositive weights or water columns, vector metadata/pairing errors, inconsistent speed, valid dry-cell values, all-NaN frames, or under 95 percent finite active-cell coverage. Broad physical excursions are warnings and are never clipped.
