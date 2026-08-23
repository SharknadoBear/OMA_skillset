# CFS atmospheric v2 contract

## Request

```json
{
  "schema_version": "cfs_atmospheric_request_v2",
  "start": "2019-07-01T00:00:00Z",
  "end": "2019-07-01T00:00:00Z",
  "products": [
    "wind_10m",
    "surface_pressure",
    "air_temperature_2m",
    "specific_humidity_2m",
    "precipitation_rate",
    "downward_shortwave_surface_flux",
    "downward_longwave_surface_flux"
  ],
  "bbox": [283.0, 38.0, 283.3, 38.3],
  "halo_cells": 1,
  "provider": "auto",
  "provider_order": ["ncei", "hycom"],
  "output": "atmospheric_fields.nc"
}
```

Times are inclusive exact UTC hours. The bbox is `[west,south,east,north]` in 0-360 longitude. Additional products are `surface_wind_stress`, `surface_temperature`, `latent_heat_net_flux`, and `sensible_heat_net_flux`.

## Eras and routing

- CFSR: 1979-01-01T00:00Z through 2011-03-31T23:00Z.
- CFSv2: 2011-04-01T00:00Z onward. The live NCEI catalog determines its available upper month and gaps.
- A wrong-era request is delegated to the adjacent sibling skill with the active Python executable.
- A crossing request produces `{stem}_cfsr.nc`, `{stem}_cfsv2.nc`, and `cfs_family_routing_manifest_v1`. The two native grids are never merged.
- A missing month inside the correct era triggers whole-request provider fallback, not an era change.

## Providers and products

NCEI uses full-resolution monthly files and rejects `.l.` products. Each selected GRIB message must match its parameter, native level, and requested valid time. Duplicate valid times retain the shortest forecast lead.

Provider selection is atomic for one era segment. HYCOM is eligible only if every requested product has an exact mapping; unsupported temperature/humidity or composite mappings fail closed instead of being fabricated. Provider attempts and the final lock are recorded.

Canonical fields use `time`, `latitude`, and `longitude`, plus provider-neutral names such as `eastward_wind`, `northward_wind`, `absolute_air_pressure`, `air_temperature_2m`, `specific_humidity_2m`, and `precipitation_rate`. Output variables retain source GRIB units, level, product-definition template, interval/step, forecast lead, filename, conversion, and checksum metadata.

## Gates, resume, and monitoring

- `estimate` hashes the normalized request and immutable plan, probes a bounded HTTP range, and requires `free_bytes > 4 * raw_transfer_bytes`.
- `snapshot` downloads only complete ranged GRIB messages for one valid hour. Production `fetch` resumes full monthly files through `.part` files.
- Runs estimated at ten minutes or longer use the loopback HTML waitbar. JSON publication retries transient Windows/OneDrive locks while remaining atomic.
- Inspect `source_provider_lock.json`, `download_status.json`, and `health_check.json` before downstream use.

## Runtime

Use an isolated Python 3.13 environment when the current interpreter lacks GRIB support. The acceptance environment pins `numpy==2.5.1`, `netCDF4==1.7.4`, `requests==2.34.2`, `rasterio==1.5.1`, `xarray==2026.7.0`, and `eccodes==2.47.0`. Run `scripts/check_grib_runtime.py` and retain its JSON plus `pip freeze` with live evidence.

## CFSR compatibility

The historical surface-pressure request, plan, fetch, resume, and health functions remain importable. NCEI `pressfc` is absolute pressure. HYCOM `airprs` is converted from its departure by `(airprs + 1000) * 100` Pa.
