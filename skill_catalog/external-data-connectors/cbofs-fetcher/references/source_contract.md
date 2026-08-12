# CBOFS operational AWS and NCEI source contract

## Archive policy v2

Requests normalize to `cbofs_request_v2` and carry one explicit policy:
`aws_then_ncei`, `aws_only`, or `ncei_only`. A v1 request is accepted only as a
lossless migration to v2 with `aws_then_ncei`. The operational source is
`aws_operational`. The fallback source is `ncei_long_term`, rooted at
`operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access/chesapeake-bay-operational-forecast-system-cbofs/`
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
cache sidecar, plan, fetch manifest, extraction report, and health report binds
its source identity. No object may silently change archive at fetch time.

## Access and discovery

- AWS operational source: anonymous ListObjectsV2 at bucket
  `noaa-nos-ofs-pds`, listing endpoint
  `https://noaa-nos-ofs-pds.s3.amazonaws.com/`, canonical object endpoint
  `https://noaa-nos-ofs-pds.s3.amazonaws.com`, and daily/monthly prefixes
  `cbofs/netcdf/YYYY/MM/DD/` and `cbofs/netcdf/YYYYMM/`.
- NCEI long-term source: anonymous ListObjectsV2 at container `prod-model`,
  listing/object endpoint `https://www.ncei.noaa.gov/oa/prod-model`, and the
  month-bounded root
  `operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access/chesapeake-bay-operational-forecast-system-cbofs/YYYY/MM/`.
- A canonical object URL is the provider endpoint plus its exact, safely
  encoded key. Never rewrite an AWS key as NCEI or an NCEI key as AWS.
- Accept current names such as `cbofs.t00z.20260720.fields.n006.nc` and
  legacy names such as `nos.cbofs.fields.n006.20260720.t00z.nc`.
- Reject non-cycle hours, AWS `n000`, `n007+`, native `f000`, station n/f
  objects, and field/regular-grid aggregate names. Accept historical `n000`
  only inside the validated NCEI model/month root.
- Prefer current names over legacy names and daily layout over monthly layout.
  Resolve equal-rank duplicates deterministically only when both exact size and
  opaque ETag match; record the rejected twin. Treat other equal-rank metadata
  conflicts as errors.
- Treat ETags, including multipart ETags, as opaque source provenance.
- Approve only exact provider-specific roots whose directory date matches the
  filename run date, and only that provider's canonical URL. Require positive
  size, nonempty opaque ETag, and Last-Modified provenance before local routing.

## Time and products

CBOFS cycles are 00, 06, 12, and 18 UTC. Native `fields` and
`regulargrid` objects use one-hour records. Verified native-field planning
times are:

- `n001 = cycle - 5 h` through `n006 = cycle`.
- `f001 = cycle + 1 h` onward; there is no implied `f000`.

Downloaded `ocean_time` is authoritative. Normalize a coordinate to the
nominal cadence only when it is within 60 seconds and preserve its original
value and adjustment. Station aggregates use six-minute records; when
cycles overlap, retain the preceding cycle's terminal record.

Historical NCEI files may label the CF proleptic Gregorian calendar as
`gregorian_proleptic`. Decode that exact legacy alias as
`proleptic_gregorian`, while retaining the original calendar string, units,
and alias decision in extraction provenance.

Native `fields` are the processed v1 path. `stations` and `regulargrid` are
inventory/download/inspection passthrough products. CBOFS Vibrio probability
products are outside v1.

## ROMS geometry and vectors

Live CBOFS fields use a curvilinear staggered ROMS C-grid with rho, U-edge,
and V-edge coordinates, 20 `s_rho` levels, and complete W-level metadata.
Read `Vtransform`, `Vstretching`, `hc`, `s_rho`, `s_w`, `Cs_r`, `Cs_w`,
`h`, `zeta`, and `angle` from each file. Do not assume a transform version.

Native `u` and `v` are grid-relative. For a requested view, reduce them on
their native grids, destagger to rho points with wet/finite-aware adjacent
averages, rotate by `angle`, and then compute speed. For a depth average,
derive `abs(diff(z_w))` independently on the rho/U/V grids and renormalize
over finite wet layers. Verify that rho-grid thickness sums close to
`h + zeta`.

Before rotating, require `angle` on `(eta_rho, xi_rho)`, finite everywhere,
with magnitude no greater than `2*pi` plus floating tolerance. Units must be
`rad`, `radian`, or `radians`. Semantics must be established by exact CF
standard name `grid_angle_of_rotation_from_east_to_y` or a long name that
identifies the angle between the XI axis and east. The compact convention is
`xi_axis_counterclockwise_from_east_radians`; preserve source metadata and
reject units/semantic drift between files.

Practical salinity is exported as source variable `salt`. Preserve its
source attributes; do not clip it or invent units when the source omits them.
Reject geometry, dimensions, masks, sigma metadata, vector convention, or
variable-schema drift. Do not coerce incompatible eras; split a long request
at the verified model/grid-era boundary and process each compatible segment.

## Storage and integrity

Write an exact, positive-size object estimate before transfer. Permit local download only
when free space is strictly greater than four times the exact source bytes;
otherwise recommend Kestrel. Transfers use resumable `.part` files, atomic
rename, exact size/ETag checks, streaming SHA-256, cache sidecars, and
idempotent validated cache hits. A response must return the exact planned ETag;
missing ETag fails. After any cache miss and immediately before GET, re-list the
exact approved key from its approved provider and require current size, ETag,
and Last-Modified to match the reviewed plan; listing failure or drift requires
replanning and never triggers a silent provider switch. Transfer accepts a reviewed plan only. Extraction from a
manifest verifies its request and approved-plan digest, exact NOAA object
identity, manifest outcome, local cache containment, NetCDF bytes, and matching
sidecar key/URL/size/SHA-256/ETag/Last-Modified fields. The extraction report
binds the exact verified input keys and paths to the fetch-manifest hash; full
run health rejects unbound explicit inputs.
