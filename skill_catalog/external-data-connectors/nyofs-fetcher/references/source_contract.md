# NOAA NYOFS public-AWS source contract

## Source and access

- Bucket: `noaa-nos-ofs-pds`
- Anonymous object base: `https://noaa-nos-ofs-pds.s3.amazonaws.com/`
- List API: `https://noaa-nos-ofs-pds.s3.amazonaws.com/?list-type=2&prefix=...`
- Current keys: `nyofs/netcdf/YYYY/MM/DD/`
- Legacy discovery path: `nyofs/netcdf/YYYYMM/`

Use unsigned HTTPS and ListObjectsV2. Paginate with `NextContinuationToken`. Do not require `aws`, boto3, s3fs, or credentials. Treat every ETag as opaque provenance; a multipart ETag containing `-` is not an MD5 checksum.

## Models, products, and grids

NYOFS is a Princeton Ocean Model (POM) system with a structured curvilinear coarse grid (`nyofs`) and a separate nested fine grid (`nyofs_fg`). Do not combine the grids or apply FVCOM/ROMS topology rules.

Current aggregate names are:

```text
nyofs.t05z.20260720.fields.nowcast.nc
nyofs_fg.t05z.20260720.fields.nowcast.nc
nyofs.t05z.20260720.stations.forecast.nc
```

Cycles occur at 05, 11, 17, and 23 UTC. A current fields-nowcast aggregate nominally contains six hourly records ending at its cycle. A station-nowcast aggregate nominally spans six hours at six-minute cadence and overlaps the next cycle at the boundary. Prefer the preceding cycle's terminal station record when deduplicating.

Fields forecasts nominally contain 54 hourly records from run plus 1 hour through run plus 54 hours. Station forecasts contain 541 records from the run through run plus 54 hours inclusive (nominal half-open span ending six minutes later). Actual downloaded time coordinates remain authoritative.

Legacy archives may use names such as:

```text
nos.nyofs.fields.nowcast.20260720.t05z.nc
nos.nyofs_fg.fields.forecast.20260720.t05z.nc
nos.nyofs.stations.nowcast.20260720.t05z.nc
```

Legacy NYOFS objects are aggregates too. Inventory may encounter equivalent current/legacy aggregates during transitions; prefer current names, then daily layout over monthly layout. Do not infer generic OFS per-lead naming for NYOFS.

## Live POM field structure

The current coarse fields file is NetCDF-3 classic with dimensions like:

```text
time = 6
sigma = 7
ny = 134
nx = 73
```

The fine grid has independent `ny`/`nx` sizes. Discover dimensions rather than hard-coding them.

Core variables are:

| Variable | Typical dimensions | Meaning |
| --- | --- | --- |
| `time` | `(time)` | float days since 2008-01-01 |
| `lon`, `lat` | `(ny,nx)` | curvilinear geographic coordinates |
| `mask` | `(ny,nx)` | 1 wet, 0 land |
| `depth` | `(ny,nx)` | positive-down bathymetry in metres |
| `sigma` | `(sigma)` | sigma point coordinate, currently 0 surface to 1 bottom |
| `zeta` | `(time,ny,nx)` | water-surface elevation |
| `air_u`, `air_v` | `(time,ny,nx)` | eastward/northward wind |
| `u`, `v`, `w` | `(time,sigma,ny,nx)` | eastward, northward, and vertical water velocity |

Discover other variables and attributes from each file. Current `u` and `v` share the scalar grid and advertise eastward/northward standard names; do not destagger or rotate them. Reject mismatched dimensions or explicit grid-relative metadata unless usable angle metadata and a tested rotation path are added in a later contract.

Station files use a `station` dimension, contain station coordinates/names, and currently contain 61 six-minute-ish records per nowcast aggregate. They are passthrough products in v1.

## Time handling

Decode `time` from its own `units` and optional `calendar`. Float32 day values produce jitter such as `00:59:45.9375`. Round to the nearest hourly timestamp for fields or six-minute timestamp for stations only when the difference is at most 60 seconds. Preserve both the original decoded time and the signed `original - normalized` offset.

Use actual downloaded coordinates for extraction and final QA. Filename-derived aggregate spans are planning hints, especially for forecasts. Crop records using inclusive start/exclusive end and require unique monotonic normalized times.

## Vertical calculations

Find surface and bottom dynamically by absolute distance from zero:

- surface: minimum `abs(sigma)`
- near surface: second-smallest `abs(sigma)`
- bottom: maximum `abs(sigma)`

For a point-coordinate vector `s`, compute trapezoidal quadrature weights in sorted sigma order: endpoints receive half their adjacent interval and interiors receive half the interval between neighbors. Use absolute interval length, map the weights back to source order, and normalize. For every wet horizontal cell, discard non-finite layers and renormalize the remaining weights. Compute depth-averaged `u` and `v` separately, then `sqrt(ubar^2 + vbar^2)`.

## Compact field product

Write one file per grid: `nyofs_coarse_fields.nc` or `nyofs_fine_fields.nc`. Use `nyofs_compact_fields_v1` and preserve:

- canonical hourly `time`, original decoded time, raw numeric time/units/calendar, adjustment, source cycle, and source key;
- `lon(y,x)`, `lat(y,x)`, `mask(y,x)`, `depth(y,x)`, and `sigma(sigma)`;
- requested source variables after applying source fill values and `mask == 1`;
- derived wind speed and requested velocity views using suffixes `surface`, `near_surface`, `bottom`, `sigma_N`, and `depth_average`.

Set global attributes identifying NYOFS, POM, coarse/fine grid, curvilinear geometry, earth-relative vectors, requested window, source keys, and creation time.

## Evidence and finishing gates

`download_estimate.json` is the approval artifact. `fetch_manifest.json` is the transfer ledger. `inspection.json`, `extraction_manifest.json`, and `health_check.json` cover source structure, compact outputs, and final integrity.

Critical findings include incomplete required cadence, corrupt size/hash evidence, grid/schema drift, invalid coordinates/masks/sigma, unpaired velocity components, speed inconsistency, an all-NaN frame, or less than 95 percent finite coverage over wet cells. Salinity and temperature are not assumed to exist. Broad elevation/current/wind limits are warnings and are never used for clipping.
