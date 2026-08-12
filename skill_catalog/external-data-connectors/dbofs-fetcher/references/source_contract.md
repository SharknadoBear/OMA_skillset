# DBOFS operational AWS and NCEI source contract

## Archive policy v2

Requests normalize to `dbofs_request_v2` and carry one explicit policy:
`aws_then_ncei`, `aws_only`, or `ncei_only`. A v1 request is accepted only as a
lossless migration to v2 with `aws_then_ncei`. The operational source is
`aws_operational`. The fallback source is `ncei_long_term`, rooted at
`operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access/delaware-bay-operational-forecast-system-dbofs/`
under `https://www.ncei.noaa.gov/oa/prod-model`.

NCEI discovery is month-bounded and supports native nowcast fields (including
historical `n000`) and station nowcast/forecast aggregates. It does not support
field forecasts or regular-grid products. Filename time is only a planning
hint; decoded NetCDF time is authoritative and cross-archive duplicates are
removed after normalization. Scientific duplicate rules run before provider
preference: an equivalent `n006` outranks historical `n000`, and a preceding
station cycle's terminal record outranks the following cycle's initial record.
AWS wins only otherwise-equivalent candidates under `aws_then_ncei`; NCEI
fills uncovered or scientifically lower-ranked AWS coverage. Every selected object,
cache sidecar, plan, fetch manifest, extraction manifest, and health report
binds its source identity. No object may silently change archive at fetch time.

## Archive

- AWS operational source: bucket `noaa-nos-ofs-pds`, anonymous ListObjectsV2
  endpoint `https://noaa-nos-ofs-pds.s3.amazonaws.com/`, canonical object
  endpoint `https://noaa-nos-ofs-pds.s3.amazonaws.com`, current layout
  `dbofs/netcdf/YYYY/MM/DD/`, and legacy layout `dbofs/netcdf/YYYYMM/`.
- NCEI long-term source: container `prod-model`, anonymous ListObjectsV2 and
  canonical object endpoint `https://www.ncei.noaa.gov/oa/prod-model`, and
  month-bounded root
  `operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access/delaware-bay-operational-forecast-system-dbofs/YYYY/MM/`.
- A canonical URL is the approved provider endpoint plus its exact, safely
  encoded key; keys are never translated between providers.
- Cycles: 00, 06, 12, and 18 UTC
- Products: native `fields`, aggregate `stations`, and `regulargrid`

Discover provider-specific prefixes with anonymous ListObjectsV2. Do not encode
a retention window. The listing and downloaded NetCDF metadata are authoritative.

Current fields use names such as
`dbofs.t06z.20260720.fields.n001.nc`; legacy monthly archives can use
`nos.dbofs.fields.n001.20260720.t06z.nc`. Current names outrank legacy names,
and daily layout outranks monthly layout for semantic duplicates. Record every
discarded alternative; fail if equal-rank alternatives disagree.

## Time

Native field objects contain one record. Nowcasts are `n001` through `n006`,
where `n001=cycle-5h` and `n006=cycle`. Forecasts are `f001` through `f048`.
Reject cycle hours other than 00, 06, 12, and 18 UTC and reject other leads.
Always verify `ocean_time` after transfer. Normalize jitter only within 60
seconds and preserve the original value and adjustment.

Historical NCEI files may label the CF proleptic Gregorian calendar as
`gregorian_proleptic`. Decode that exact legacy alias as
`proleptic_gregorian`, while retaining the original calendar string, units,
and alias decision in extraction provenance.

Station files aggregate a cycle at nominal six-minute cadence. At a cycle
boundary, prefer the preceding cycle's terminal record. Crop all products to
the inclusive/exclusive requested window.

## Native ROMS grid

DBOFS fields use a curvilinear staggered ROMS C-grid:

- scalar `salt`, `temp`, `zeta`, `h`, and `angle` use rho points;
- `u` and `v` use their respective staggered edge grids;
- native `mask_rho`, `mask_u`, and `mask_v` are required, finite, correctly
  shaped, and strictly binary (`1=wet`, `0=land`); never synthesize them;
- fill values are source-defined and must become missing values;
- `angle` must be rho-shaped, finite on wet points, explicitly labelled
  `rad`/`radian`/`radians`, and identify the XI-axis rotation from east through
  NOAA's standard-name or equivalent long-name metadata. The canonical
  processing convention is `xi_axis_counterclockwise_from_east_radians`.

Current exports have ten rho layers and advertise `Vtransform=1`, but code must
read all vertical parameters dynamically and support transforms 1 and 2.
`s_rho`, `s_w`, `Cs_r`, and `Cs_w` must be finite, one-dimensional, and
strictly monotonic; `hc` must be finite and nonnegative; `Vtransform` must be 1
or 2; and `Vstretching` must be a positive integer. Requested dynamic-variable
dimension, grid, location, and dtype signatures may not drift between files.

For transform 1:

```text
z0 = (s - Cs) * hc + Cs * h
z  = z0 + zeta * (1 + z0 / h)
```

For transform 2:

```text
z0 = (hc * s + h * Cs) / (hc + h)
z  = zeta + (zeta + h) * z0
```

Calculate W-level depths and `abs(diff(z_w))`. A valid column closes to
`h+zeta` within numeric tolerance. Average U and V over their native grid
columns, destagger finite wet values to rho points, rotate to earth-relative
east/north, then calculate speed.

Reject geometry, dimensions, masks, sigma metadata, vector convention, or
variable-schema drift. Do not coerce incompatible eras; split a long request
at the verified model/grid-era boundary and process each compatible segment.

## Transfer integrity

Only accept keys inside the exact AWS `dbofs/netcdf/` daily/monthly root or the
exact NCEI DBOFS `YYYY/MM/` root, with directory date agreeing with the filename
run date and URL equal to that provider's canonical object URL. An approved
object requires positive size, nonempty opaque ETag, and Last-Modified
provenance. Immediately before transfer, recompute the exact total and require
current free space to exceed four times that total. After any cache miss and
immediately before GET, re-list the exact key from its approved provider and
require current size, ETag, and Last-Modified to match the reviewed plan;
listing failure or drift requires replanning and never changes provider.

Use resumable `.part` files and atomic completion. Validate object byte count
and require each HTTP response to repeat the exact planned ETag; a missing or
changed response ETag fails closed. Preserve Last-Modified and compute SHA-256
locally. Multipart ETags are opaque source versions, not MD5 digests. Reuse a
cache entry only when its file and sidecar validate against the selected remote
object.
