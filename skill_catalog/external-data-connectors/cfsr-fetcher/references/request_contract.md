# CFSR request contract

## Request JSON

```json
{
  "start": "2008-01-01T00:00:00Z",
  "end": "2009-01-01T00:00:00Z",
  "product": "surface_pressure",
  "bbox": [279.1369, 31.6530, 296.0626, 45.7655],
  "halo_cells": 1,
  "provider": "auto",
  "provider_order": ["ncei", "hycom"],
  "chunk_hours": 168,
  "max_retries": 5,
  "retry_delay_seconds": 5,
  "backoff": 2,
  "output": "cfsr_surface_pressure.nc"
}
```

`bbox` is `[west,south,east,north]` in 0-360 longitude. `surface_pressure` is the initial supported product. Providers are `ncei`, `hycom`, or `auto`; `auto` tests providers in `provider_order` but locks one provider for the complete output.

Optional public source overrides are `ncei_catalog_template`, `ncei_file_template`, and `hycom_source_template`. They may contain documented date placeholders but no credentials, query strings, or fragments.

## Output

The provider-neutral acquisition output is compressed NetCDF4-classic with:

- dimensions `time`, `latitude`, `longitude`;
- `absolute_air_pressure` in Pa with `pressure_reference="absolute"`;
- exact UTC epoch-second `time` and native latitude/longitude coordinates;
- provider, source-variable, conversion, request-hash, source-file, and checksum provenance.

NCEI `pressfc` is decoded as absolute pressure. HYCOM `airprs` is a departure from 1000 hPa and is explicitly converted.

Monthly NCEI files contain alternate-grid and overlapping forecast records. The decoder identifies the dominant forecast-grid signature from the complete monthly file, including when only an inclusive endpoint is requested, retains the shortest forecast lead for each duplicate valid time, and assigns each valid time to its own source month. This produces one fixed native grid and one exact hourly record per UTC hour.

## Gates and monitoring

- `estimate` inventories every source unit, probes a bounded transfer, calculates raw-transfer and output sizes, checks `free_bytes > 4 * raw_transfer_bytes`, and hashes the plan.
- Plans expire after 24 hours and cannot be edited without invalidating the hash.
- NCEI downloads use monthly `.part` files and HTTP Range resume. Monthly decoded checkpoints are atomic and keyed by request hash.
- `download_status.json` follows `external_download_status_v1` and adds provider, active source, decode progress, last-success time, and recent messages.
- `download_monitor.html` polls the status every 30 seconds and includes a manual refresh button.
- Provider fallback is whole-run only. Provider-specific checkpoints live in separate directories.
