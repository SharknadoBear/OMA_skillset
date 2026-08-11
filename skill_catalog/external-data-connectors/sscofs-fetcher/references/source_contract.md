# SSCOFS source contract

Use this reference when maintaining inventory parsing, valid-time selection, native FVCOM extraction, or source-aware QA. Discover the live archive and NetCDF schema at runtime; treat the values below as the supported contract and verified operational examples, not as a promise of indefinite retention.

## Authoritative sources

- NOAA SSCOFS system and output documentation: <https://tidesandcurrents.noaa.gov/ofs/sscofs/sscofs_info.html>
- NOAA OFS naming, timing, archive, and FVCOM FAQ: <https://tidesandcurrents.noaa.gov/ofs/ofs_faq.html>
- AWS NOAA OFS public-data documentation: <https://github.com/awslabs/open-data-docs/tree/main/docs/noaa/noaa-ofs-pds>
- Public archive browser: <https://noaa-nos-ofs-pds.s3.amazonaws.com/index.html#sscofs/>

## Anonymous AWS access

- Use bucket `noaa-nos-ofs-pds` in `us-east-1` without credentials.
- List objects with S3 ListObjectsV2 over HTTPS:
  `https://noaa-nos-ofs-pds.s3.amazonaws.com/?list-type=2&prefix=<url-encoded-prefix>`.
- Follow `IsTruncated` and `NextContinuationToken` until pagination completes. URL-encode the continuation token on the next request.
- Download an object with `https://noaa-nos-ofs-pds.s3.amazonaws.com/<url-encoded-key>`. Preserve path separators while encoding key components.
- Use HTTP `Range` for `.part` resumption. Validate status, Content-Length/Content-Range, the listed object size, and source ETag before atomic completion.
- Treat an ETag as source metadata, not a content checksum. A quoted ETag containing `-<part-count>` is multipart and is not an MD5 digest. Compute SHA-256 locally for every completed payload.

Search both observed archive layouts and deduplicate identical keys/valid records:

```text
sscofs/netcdf/YYYY/MM/DD/    current nested day layout
sscofs/netcdf/YYYYMM/        legacy monthly layout
```

Exclude `sscofs/pre_operation/` by default because it contains developmental/pre-operational data. Never hard-code archive start dates, retention, or the date when a layout changed.

## Products and filenames

Support the current NOAA names:

```text
sscofs.tCCz.YYYYMMDD.fields.[n|f]HHH.nc
sscofs.tCCz.YYYYMMDD.regulargrid.[n|f]HHH.nc
sscofs.tCCz.YYYYMMDD.stations.[nowcast|forecast].nc
```

Recognize legacy archive names when discovered:

```text
nos.sscofs.fields.[n|f]HHH.YYYYMMDD.tCCz.nc
nos.sscofs.regulargrid.[n|f]HHH.YYYYMMDD.tCCz.nc
nos.sscofs.stations.[nowcast|forecast].YYYYMMDD.tCCz.nc
```

Interpret `CC` as the cycle hour, `YYYYMMDD` as the cycle date, `n`/`f` as guidance, and `HHH` as the lead code. Accept only parsed SSCOFS output products that agree with the requested product and guidance; do not select files by substring alone.

Fully extract only `fields`. Treat `stations` and `regulargrid` as raw passthrough products in v1 because their variables, time axes, and grids differ from native fields. Inventory and estimate both passthrough products dynamically.

## Run and valid times

- SSCOFS runs at 03, 09, 15, and 21 UTC, four times per day.
- Native fields and regular-grid fields are hourly.
- A nowcast cycle contains `n000` through `n006`. For cycle time `R`, compute `valid_time = R + (HHH - 6) hours`.
- A forecast cycle contains `f000` through `f072`. Compute `valid_time = R + HHH hours`.
- `n000` duplicates `n006` from the preceding cycle. For a continuous nowcast series, prefer the preceding cycle's `n006` and drop the later `n000`.
- `f000` duplicates `n006` from the same cycle. Retain it only when it is needed by an explicitly selected forecast run and report the equivalence in plan metadata.
- Station files contain multiple internal six-minute timestamps; derive their actual coverage from NetCDF time variables after download rather than assigning one valid time from the filename.
- Treat all model times as UTC. Verify each native field's decoded `Times` value against its filename-derived valid time before concatenation.

Select records in the half-open interval `[start_utc, end_utc_exclusive)`. For a forecast request, require one `run_cycle_utc` and select only that run. Report every missing expected timestamp and every discarded duplicate.

## Native FVCOM fields

SSCOFS uses an unstructured triangular FVCOM grid. NOAA currently documents 239,734 nodes, 433,410 elements, and 10 spatially varying terrain-following sigma layers. Validate actual dimensions and geometry in every input; do not encode those counts as acceptance constants.

Common native-field dimensions and variables include:

- `time` and character `Times` for simulation time;
- node coordinates `lon`, `lat`, projected `x`, `y`, and bathymetry `h`;
- element coordinates `lonc`, `latc`, projected `xc`, `yc`;
- connectivity `nv(3, nele)` and neighbor topology such as `nbe`;
- node sigma coordinates `siglay(siglay, node)` and interfaces `siglev(siglev, node)`;
- node fields such as `zeta(time, node)`, `temp(time, siglay, node)`, and `salinity(time, siglay, node)`;
- element fields `u(time, siglay, nele)` and `v(time, siglay, nele)`, plus other dynamically discovered fields;
- wet/dry and inundation masks whose exact names must be discovered and preserved when applicable.

Do not expect native depth-averaged `ua` or `va`. Do not substitute regular-grid aliases such as `salt`, `u_eastward`, or `v_northward` for native variables without an explicit product-aware mapping.

Validate that connectivity indices are in range after normalizing the file's one-based or zero-based convention. Reject changes in node/element counts, coordinates, connectivity, sigma geometry, critical dimensions, or requested variable centering across a concatenated run.

## Vertical views and averages

Determine layer order from `siglay`, not from an assumed index. The surface layer has the sigma midpoint closest to zero; the bottom layer has the most negative/deepest midpoint. Current operational files place them at indices 0 and 9, but reversed or changed ordering must remain valid. Define `near_surface` as the second layer from the detected surface; this corresponds to the layer NOAA uses for its displayed near-surface currents.

For node `j` and layer `k`, compute the nonnegative layer-thickness fraction from interfaces:

```text
d_sigma[k,j] = abs(siglev[k+1,j] - siglev[k,j])
```

For a node-centered field `X`, compute the finite, wet-layer-renormalized average:

```text
X_bar[j] = sum_k(X[k,j] * d_sigma[k,j]) / sum_k(d_sigma[k,j])
```

Include only layers where the value and weight are finite and the applicable wet mask is true. Return missing where the retained denominator is zero. Diagnose unmasked wet columns whose full finite weight sum is materially different from one.

For element `e` with its three nodes from `nv`, map node thickness to the cell:

```text
d_sigma_cell[k,e] = mean(d_sigma[k, nv[:,e]])
```

Use `d_sigma_cell` and the wet-cell mask to compute `u_bar` and `v_bar` independently over finite layers, while requiring paired usable components for current diagnostics. Then compute depth-averaged current speed from the averaged vector:

```text
speed_bar = sqrt(u_bar**2 + v_bar**2)
```

For surface, near-surface, bottom, or explicit sigma-index views, compute speed from the corresponding paired `u` and `v` view. Never average per-layer speed as a substitute for vector averaging.

## Required provenance and QA

Write the request, listed keys, object URLs, sizes, ETags, last-modified timestamps, parsed cycle/lead/valid times, layout and naming convention, duplicate decisions, missing timestamps, transfer status, resume/retry counts, SHA-256 hashes, cache paths, and software identity into JSON run artifacts.

Verify object integrity, unique monotonic cadence, requested time completeness, mesh/topology consistency, sigma weight sums, requested-variable presence and centering, paired `u`/`v`, no all-NaN frame, and at least 95 percent finite coverage over applicable wet nodes/cells. Make broad physical plausibility ranges warnings rather than destructive filters or clipping rules.
