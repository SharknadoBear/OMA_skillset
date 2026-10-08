# Request contract

All times are explicit UTC ISO dates, start inclusive and end exclusive. Bounds are west, south, east, north in nominal geographic degrees and must not cross the date line. A native center crop does not create clipped cells or spatial interpolation.

```json
{
  "schema": "ww3_request_v1",
  "source_family": "multi_1",
  "product": "fields",
  "grid": "at_4m",
  "stations": [],
  "start": "2018-11-01T00:00:00Z",
  "end": "2018-12-01T00:00:00Z",
  "bbox": [-76.35, 36.43, -74.95, 38.10],
  "fields": ["hs", "tp", "dp"],
  "missing_policy": "error",
  "raw_retention": "delete_after_health"
}
```

For spectra use `product: "point_spectra"`, `grid: null`, `stations: ["44014", "44099"]`, `bbox: null`, `fields: []`. Each month acquires SPEC and WMO archives, then retains only selected station products. Transfer size is global, not station-size. Inventory is authoritative. Field codes include hs, tp, dp, wind, phs, ptp, pdir as actually published. Wind and partition files retain distinct GRIB element/level groups as separate NetCDF variables.

`missing_policy` is `error` or `skip` (explicitly report missing monthly files/cadence gaps). Selected stations missing from either archive fail; skip never substitutes stations. `raw_retention` is `keep` or `delete_after_health`.

Products: inventory.json, plan.json, fetch_manifest.json, extraction_manifest.json, health.json, source README snapshot, native field NetCDFs or station spectra NetCDFs with separate point-bulk dimensions, and optional cleanup.json. Request hashes bind stages. Sources retain names inside raw/. Partial downloads retain identity metadata for resumption.
