# CFSv2 request and run artifacts

## Request JSON

```json
{
  "start": "2019-07-01T00:00:00Z",
  "end": "2019-07-03T00:00:00Z",
  "subdataset": "uv-10m",
  "variables": ["wndewd", "wndnwd"],
  "bbox": [283.0, 36.0, 288.0, 41.0],
  "chunk_hours": 168,
  "max_retries": 5,
  "output": "wind.nc"
}
```

`bbox` is `[west, south, east, north]` in the source 0–360 longitude convention. Inventory the year and subdataset before requesting fields. Omit `variables` to select every documented field in that subdataset.

Use canonical `dlwsfc`; `dlwflx` remains accepted only as a historical subdataset alias. The field within the source remains `dlwflx`.

`source_url` is an advanced public/local override primarily for mirrors and tests. It may contain `{year}` and `{subdataset}` templates but must not contain credentials, query parameters, or fragments.

## Gate and monitoring

- Estimate measures a bounded source probe, exact selected array bytes, chunk latency, working space, and a conservative duration.
- Fetch verifies the plan hash, request hash, 24-hour expiry, and storage gate before opening data variables.
- A conservative estimate of at least 600 seconds automatically creates and opens `download_monitor.html` on `127.0.0.1`.
- `download_status.json` is written atomically during active transfers and contains only progress metadata.
- Checkpoints are isolated by request hash and retained for resume unless explicitly cleaned after success.

## Scientific conventions

The connector preserves native CFSv2 variables. HYCOM `airprs` is a departure from a 1000 hPa base, not absolute pressure. Convert explicitly with `cfsv2_airprs_to_absolute_pa`; downstream forcing writers own all interpolation, flux derivation, and model packaging.
