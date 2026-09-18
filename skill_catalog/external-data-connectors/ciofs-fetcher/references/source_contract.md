# CIOFS point connector: source and scientific contract

## Source scope

Version 1 uses anonymous HTTPS access to NOAA CO-OPS THREDDS native CIOFS ROMS hourly field files. NOAA describes the model, four daily cycles, six-hour nowcasts, and 48-hour forecasts at:

- [CIOFS information](https://tidesandcurrents.noaa.gov/ofs/ciofs/ciofs_info.html)
- [Native field catalog](https://opendap.co-ops.nos.noaa.gov/thredds/catalog/NOAA/CIOFS/MODELS/catalog.html)

Do not assume a fixed rolling retention length. Availability is discovered for the exact requested dates, never inferred from today's date or a previous project. There is no historical-source fallback in this version.

Canonical file paths are `NOAA/CIOFS/MODELS/YYYY/MM/DD/ciofs.tHHz.YYYYMMDD.fields.nNNN.nc` for nowcasts and `...fields.fNNN.nc` for forecasts. Cycles HH are 00, 06, 12, or 18 UTC. Nowcast n001 through n006 nominally represent cycle−5 through cycle+0 hours; forecast f001 through f048 represent cycle+1 through cycle+48 hours. The cycle date can differ from valid date. Inventory includes next-day cycles for evening nowcast valid times and deduplicates source identities. A forecast request is bound to exactly one cycle; it never blends lead times from different forecasts.

Day catalogs use `/thredds/catalog/.../catalog.xml`; DAP2 uses `/thredds/dodsC/...nc.dds`, `.das`, and `.dods?constraint`. The server supports primitive numeric arrays. This connector explicitly decodes the verified Float32/Float64/Int32/UInt32 DAP2 representation and rejects unknown declarations, incorrect array counts, truncation, and unconsumed bytes. It does not implement arbitrary DAP2 structures, grids, strings, or DAP4. Metadata and each constrained binary response are retained.

## Request and selection

`ciofs_point_request_v1` uses explicit timezone-aware, hour-aligned start and exclusive end. The normalizer converts timezone offsets to UTC. All bounds are applied again to decoded CF timestamps. Defaults are surface paired currents, minimum static model depth 0 m, nearest wet sampling within 5 km, all points required, completeness 0.95, no gap threshold, error on unavailable source hours, four workers, 60-second network timeout, and four attempts. Dates and point coordinates have no defaults.

`minimum_depth_m` applies to static `h`, not instantaneous total water depth. Points must be interior wet rho cells; velocity requests additionally require both neighboring U and V faces statically wet. At least one eligible cell must lie within each point's `max_distance_m`.

- **Nearest wet:** nearest WGS84 geodesic distance among a 32-candidate spherical-tree shortlist of eligible rho cells. Requested and actual coordinates and offset are separate.
- **Bilinear:** inverse bilinear coordinates in lon/lat on the enclosing four adjacent eligible rho cells. All four contributors must be within `max_distance_m`. Weights are nonnegative, sum to one, and reproduce requested coordinates within 2e−9 degrees. No extrapolation or substitution. Four-cell depth is a weighted static bathymetric value.
- `required: false` permits a selected point to fail final coverage without failing the whole run, provided at least one point/view/variable is accepted. It does not permit an unselectable geographic point.

Selection and verification retain zero-weight contributors for coordinate provenance but skip their values during interpolation. The native grid is locally smooth in Cook Inlet; this is not a global/dateline-safe interpolation service.

## Native ROMS interpretation

Source dimensions must be named in the canonical order:

| Variable | Dimensions |
| --- | --- |
| `u` | ocean_time, s_rho, eta_u, xi_u |
| `v` | ocean_time, s_rho, eta_v, xi_v |
| `salt`, `temp` | ocean_time, s_rho, eta_rho, xi_rho |
| `zeta`, `wetdry_mask_rho` | ocean_time, eta_rho, xi_rho |
| `wetdry_mask_u/v` | ocean_time, eta_u/v, xi_u/v |
| Geometry, static rho mask | eta_rho, xi_rho |

Each source must contain one time record. Source sigma count and dimensions are inspected, not fixed to 30/1044/724. The source angle must explicitly indicate the XI-axis/east rotation in radians. Velocity units must be m/s equivalents. Nonidentity scale_factor/add_offset packing is rejected, not ignored. Metadata layout, units, fill values, and static geometry must remain consistent across the run. CF time epochs may differ and are decoded separately.

Surface is the `s_rho` closest to zero; bottom is closest to −1. Reversed sigma ordering is supported. `index:N` is the native zero-based array index. These are near-surface/bottom sigma layer centres, not the free surface/seabed, depth averages, or fixed metre depths. Exact vertical elevations are not calculated. `zeta` has no vertical layer and is repeated across requested views for a consistent output interface.

At each contributing rho cell, first require a dynamically wet rho cell and two dynamically wet, non-filled neighboring faces for each velocity component. Use arithmetic face averages:

`u_rho = (u_left + u_right)/2`, `v_rho = (v_below + v_above)/2`.

Then rotate using the source rho angle θ:

`east = u_rho*cos(θ) − v_rho*sin(θ)`

`north = u_rho*sin(θ) + v_rho*cos(θ)`.

Interpolate east and north separately at exact points; only afterward compute speed. Never average speed/direction or phases to approximate vector interpolation. Scalars interpolate native rho values. If a nonzero-weight contributor is dry/filled/nonfinite, the output variable is missing; no renormalization around dry cells. Currents become missing as a pair, while valid scalar variables can remain available. There is no numerical high-speed clipping.

Fill values are converted through the source primitive dtype before comparison, avoiding the Float32 `1e37` representation trap. Static and dynamic masks are binary and validated. Missing values are `null` in JSON, blank in CSV, and NaN plus zero validity flags in NetCDF. Reasons distinguish `dry_rho`, `dry_velocity_face`, `source_fill_or_nonfinite`, and `missing_source_hour`.

Salinity/temperature units remain source units. Water level remains the model's native vertical reference; this connector does not establish or transform a tidal/geodetic datum. Do not treat zeta as automatically MSL, MLLW, or NAVD88.

## Time and coverage

CF `ocean_time` and calendar are authoritative. Supported calendars are standard, gregorian, and proleptic_gregorian; the source spelling gregorian_proleptic is normalized. Timestamps within 60 seconds of an exact UTC hour are rounded to that hour, while original decoded time and signed adjustment are retained. Larger offsets fail. Duplicate decoded point/view times fail. Filename mismatch and out-of-request records are audited; the latter are excluded, not shifted into the window. Missing requested hours still count against coverage.

Coverage is per point, view, and variable (paired currents count as one variable). It reports expected/valid counts, completeness, first-to-last valid span, longest missing run, and maximum gap. Maximum gap is the larger of the interval between consecutive valid samples and missing hours at either edge. Thus one interior missing hourly sample creates a two-hour valid-sample gap; a complete multi-hour series has a one-hour interval. No missing samples are filled. Required rows must satisfy the request's thresholds; at least one row must pass. No 29-day/30-day requirement is imposed by the fetcher. Downstream harmonic analysis must separately enforce record span and constituent separability.

## Plan, caching, and outputs

Inventory does not download scientific data. Planning makes one explicit static-grid binary request, preceded by a size probe; this generally costs tens of MB. It also probes constrained response sizes. The plan reports decoded metadata-plus-data bytes, planned request count, an intentionally conservative storage budget, available disk space, and local/insufficient routing. HTTP-compressed wire traffic cannot be predicted exactly. Partial dry masks cannot be known without fetching the time series.

Each hour is a source task; it obtains metadata and requests one small horizontal stencil per point. Multiple vertical selections use the contiguous min/max sigma interval. Workers are bounded (1–8) and requests use bounded retry/backoff. A missing/failed source does not silently disappear from the fetch manifest. Cache files use atomic replacement, paired binary/sidecar digests, source URL and query identity, and plan/metadata bindings. Do not run two simultaneous writers in the same run directory.

A complete cache is reused without contacting NOAA. This preserves the locally acquired snapshot; it does not verify that the remote URL has remained unchanged. ETag/Last-Modified are recorded when supplied but are not whole-file checksums. SHA256 applies to locally saved metadata/subsets, never to the entire remote NetCDF unless explicitly downloaded elsewhere. Request changes require a new run directory. A missing half of an interrupted binary/sidecar pair is fetched again; detected corruption or a changed binding fails for investigation.

Primary artifacts:

| Artifact | Purpose |
| --- | --- |
| `request.json`, `inventory.json`, `download_estimate.json` | Frozen request, available/missing source hours, hash-bound plan and estimate |
| `source_probe.json`, `grid.dods` | Metadata and reference static grid |
| `points.json/csv`, `interpolation_support.csv` | Requested/actual coordinates, offsets, bathymetry, support and weights |
| `cache/metadata`, `cache/subsets`, `fetch_manifest.json` | Raw replayable inputs and retrieval status |
| `point_records.json`, `point_series.csv` | Valid retrieved timestamps, vectors/scalars, source times/URLs, reason-coded masks |
| `point_series.nc` | Full requested hourly axis; time × point × view, missing-hour fill, coordinates, sigma selections, per-field validity |
| `coverage.csv`, `time_audit.json`, `extraction_manifest.json` | Acceptance details, time anomalies, output hashes |
| `health_check.json` | Offline integrity/science replay, coverage status, dependency versions |

NetCDF identifies `ciofs_point_series_v1`; it is **not** a `roms_compact_fields_v1` gridded product. CSV contains retrieved, decoded records, including null masked values, but no fabricated rows for wholly absent source hours. NetCDF includes those missing hours explicitly. The health command verifies planning inputs, caches, output hashes, full independently replayed numeric arrays, coordinates, sigma indices, time axis, validity flags, and speed consistency. All extraction/health operations are offline.

## Provenance of this reusable implementation

The inventory → estimate → fetch/resume → extract → health interface follows the existing CBOFS/DBOFS fetcher design. The tiny-stencil DAP2 and ROMS masking/rotation method was generalized from the Cook Inlet regional tidal-ellipse workflow. No runtime imports, station IDs, data paths, frozen 30-day window, harmonic choices, or map dependencies are inherited from that project. Tests and one example request ship with the skill; downloaded validation data belong in a separate project run directory.
