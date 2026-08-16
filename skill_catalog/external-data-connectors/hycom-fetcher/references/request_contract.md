# HYCOM request and run artifacts

## Request JSON

Use a public HYCOM alias, a public OPeNDAP URL without credentials/query parameters, or a local NetCDF path.

```json
{
  "source": "gofs-latest",
  "variables": ["water_temp", "salinity"],
  "start": "2026-08-01T00:00:00Z",
  "end": "2026-08-02T00:00:00Z",
  "bbox": [-76.0, 36.0, -72.0, 40.0],
  "depth": [0.0, 200.0],
  "coordinate_overrides": {},
  "dimension_bounds": {},
  "chunk_target_mib": 32,
  "max_retries": 5,
  "retry_delay_seconds": 5,
  "backoff": 2,
  "output": "subset.nc"
}
```

`bbox` is `[west, south, east, north]`. Set west greater than east for an explicit dateline crossing. The fetcher also splits Greenwich-crossing requests correctly when converting between longitude conventions.

`dimension_bounds` maps a one-dimensional coordinate name to inclusive lower and upper values. Use `coordinate_overrides` only after inventory when CF metadata or coordinate names are ambiguous.

For a point product, add `points` as objects containing `name`, `lon`, and `lat`. Point sampling uses linear interpolation on a rectilinear native grid and remains model-neutral.

## Plan and status

- The estimate command writes `hycom_download_plan_v1`; fetch verifies its plan hash, request hash, 24-hour expiry, and storage gate.
- The conservative estimate includes measured bounded-probe throughput, request latency, a 1.5 safety multiplier, and five seconds per chunk.
- Runs estimated at 600 seconds or longer create `download_monitor.html` and open a loopback-only monitor.
- `download_status.json` is updated atomically and contains progress only. It must not contain credentials, URL query strings, local absolute paths, or field values.
- Chunk filenames include the request-hash prefix. A checkpoint is reusable only when its embedded request hash and variables match.

## Output boundary

Output is a native HYCOM spatial/time/depth subset or an explicitly requested generic point sample. Do not perform FVCOM node mapping, sigma remapping, forcing generation, MJD conversion, or physical reconstruction in this skill.
