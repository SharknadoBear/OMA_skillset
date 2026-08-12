# DBOFS public-AWS source contract

## Archive

- Bucket: `noaa-nos-ofs-pds`
- HTTPS endpoint: `https://noaa-nos-ofs-pds.s3.amazonaws.com`
- Current layout: `dbofs/netcdf/YYYY/MM/DD/`
- Legacy layout: `dbofs/netcdf/YYYYMM/`
- Cycles: 00, 06, 12, and 18 UTC
- Products: native `fields`, aggregate `stations`, and `regulargrid`

Discover prefixes with anonymous S3 ListObjectsV2. Do not encode a retention
window. The listing and downloaded NetCDF metadata are authoritative.

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

## Transfer integrity

Only accept exact `dbofs/netcdf/` daily or monthly keys whose archive path
agrees with the filename run date and whose URL is the exact NOAA S3 object
URL. An approved object requires positive size, nonempty ETag, and
Last-Modified provenance. Immediately before transfer, recompute the exact
total and require current free space to exceed four times that total.

Use resumable `.part` files and atomic completion. Validate object byte count
and require each HTTP response to repeat the exact planned ETag; a missing or
changed response ETag fails closed. Preserve Last-Modified and compute SHA-256
locally. Multipart ETags are opaque source versions, not MD5 digests. Reuse a
cache entry only when its file and sidecar validate against the selected remote
object.
