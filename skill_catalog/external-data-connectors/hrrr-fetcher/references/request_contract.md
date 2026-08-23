# HRRR request and output contract

## Request schema

`hrrr_request_v1` uses one domain and one time mode.

Analysis example:

```json
{
  "schema_version": "hrrr_request_v1",
  "domain": "conus",
  "mode": "analysis",
  "start": "2024-01-15T12:00:00Z",
  "end": "2024-01-15T12:00:00Z",
  "products": ["wind_10m", "surface_pressure", "air_temperature_2m"],
  "bbox": [-77.2, 36.8, -74.5, 39.8]
}
```

Forecast example:

```json
{
  "schema_version": "hrrr_request_v1",
  "domain": "conus",
  "mode": "forecast",
  "cycle_start": "2024-01-15T12:00:00Z",
  "cycle_end": "2024-01-15T12:00:00Z",
  "cycle_step_hours": 1,
  "forecast_periods": ["PT15M", "PT30M", "PT45M", "PT1H"],
  "products": [
    {"alias": "wind_10m", "family": "wrfsubh"},
    {"alias": "precipitation_rate", "family": "wrfsubh"}
  ],
  "bbox": [-77.2, 36.8, -74.5, 39.8]
}
```

Defaults are `mode=analysis`, the domain cadence (one hour for CONUS and three hours for Alaska), `halo_cells=1`, `missing_policy=error`, `retain_raw_messages=true`, and provider order `aws,gcp,azure,nomads`. Longitudes may be -180..180 or 0..360 and are normalized to -180..180. A wrapped bbox uses west greater than east.

Alias objects may override `family`, primarily to request an available alias from `wrfsubh`. Exact selectors are objects with `family`, `short_name`, and a level description. Use `level_text`, or `type_of_level` plus `level`/`levels`. Optional keys are `output_name`, `step_type` (`instant`, `accum`, `avg`, `max`, or `min`), and `step_contains`. Exact UGRD/VGRD selectors require the same `vector_group` and matching levels; they are validated as a pair and retain their source orientation. Unit conversion is deliberately downstream; selectors cannot relabel source units.

```json
{
  "family": "wrfprs",
  "short_name": "TMP",
  "type_of_level": "isobaricInhPa",
  "levels": [1000, 850, 500],
  "output_name": "air_temperature"
}
```

## Time behavior

- Analysis maps each requested valid time to the same initialization and `f00`. It never substitutes an earlier forecast cycle.
- CONUS analysis times are hourly. Alaska analysis times are every three hours.
- Forecast requests use explicit cycles and ISO-8601 periods. Hourly families require whole-hour periods. Forecast-only `wrfsubh` accepts 15-minute periods through 18 hours and maps each period to the enclosing hourly object; strict analyses use `wrfsfc` `f00`.
- Standard forecasts extend through 18 hours. Cycles 00, 06, 12, and 18 extend through 48 hours for hourly families. Subhourly output never extends beyond 18 hours.
- `missing_policy=error` blocks execution after all mirrors fail. `skip` records gaps and proceeds only with present objects.

## Plans, downloads, and fallback

`hrrr_download_plan_v1` embeds the normalized request, inventory, request hash, storage gate, and plan hash. `fetch` rejects changed or blocked plans.

The downloader selects complete GRIB messages from `.idx` byte offsets. A provider owns a complete source object. Resuming uses message-scoped `.part` files and exact `Content-Range` validation. Provider switching is allowed only when object size and normalized selected-index signatures agree; it starts a provider-specific partial and never appends foreign bytes.

Required free space is the larger of four times selected raw bytes or selected raw bytes plus scratch, decoded full-grid working arrays, NetCDF output, and safety margin.

## Canonical output

`hrrr_fields_v1` is CF-1.10 NetCDF on the native grid.

- Analysis dimensions: `time`, optional `pressure`/`hybrid_level`, `y`, `x`.
- Forecast dimensions: `forecast_reference_time`, `forecast_period`, optional vertical dimension, `y`, `x`; `valid_time` is two-dimensional.
- Coordinates include projected `x/y`, two-dimensional `latitude/longitude`, `crs`, and `bbox_mask`.
- Hourly/native/pressure fields use `hrrr_fields.nc`. Subhourly fields use `hrrr_subhourly_fields.nc` when present.
- Canonical wind aliases are earth-relative. Raw U/V variables retain their GRIB orientation flag.
- Variable attributes preserve the source short name, name, units, level, step, interval, PDT, cycle, forecast period, provider, source key, byte range, and message SHA-256.
- Both the NOAA `.idx` label and the ecCodes `shortName` are retained because the two vocabularies use different names for some identical parameters (for example `PRES` and `sp`).

`hrrr_run_manifest_v1` links every request, provider attempt, raw message, output, and SHA-256. `hrrr_health_report_v1` checks hashes, dimensions, coordinates, time coverage, finite data, bbox coverage, and requested variables.

## Archive gates

- CONUS archive begins `2014-07-30T18:00:00Z`. Data before `2014-09-30T00:00:00Z` are marked pre-operational.
- Alaska begins `2018-07-11T18:00:00Z`.
- The live inventory determines the upper bound and gaps.
- Version labels are v1 from 2014-09-30, v2 from 2016-08-23, v3 from 2018-07-12, and v4 from 2020-12-02.
