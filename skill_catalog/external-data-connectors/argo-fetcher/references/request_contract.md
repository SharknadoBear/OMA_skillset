# Argo request and output contract

## Request schema

Use `schema: "argo_fetch_request_v1"` and provide:

```json
{
  "schema": "argo_fetch_request_v1",
  "products": ["core", "synthetic", "bio"],
  "start": "2025-01-01T00:00:00Z",
  "end": "2025-01-31T23:59:59Z",
  "bbox": [170.0, -30.0, -170.0, 30.0],
  "wmos": [5900001],
  "dacs": ["aoml"],
  "file_modes": ["R", "D"],
  "parameters": ["DOXY", "CHLA"],
  "parameter_match": "any"
}
```

`start` and `end` are inclusive UTC instants. Naive values are interpreted as UTC. Use `all_time: true` instead of dates only when an all-time query is intentional.

Choose exactly one spatial selector:

- `bbox: [west, south, east, north]`; `west > east` explicitly crosses the antimeridian.
- `geojson: "polygon.geojson"`; Polygon, MultiPolygon, Feature, and FeatureCollection polygon geometry are accepted.
- `mesh_2dm: "wet_mesh.2dm"`; ND plus E3T/E4Q elements define the exact wet footprint.
- `global: true`.

Optional WMO, DAC, filename mode, and BGC parameter filters intersect the temporal and spatial selection. Polygon and mesh boundaries count as inside. Parameters are uppercase and apply to B-profile and S-profile index rows. With `parameter_match: "all"`, every requested parameter must occur; with `"any"`, at least one must occur.

## Product mapping

| Product | Official index | Native filename family |
| --- | --- | --- |
| `core` | `ar_index_global_prof.txt.gz` | `R...nc`, `D...nc` |
| `synthetic` | `argo_synthetic-profile_index.txt.gz` | `SR...nc`, `SD...nc` |
| `bio` | `argo_bio-profile_index.txt.gz` | `BR...nc`, `BD...nc` |

The plan records the index `date_update` for every selected row. GDAC may revise native files; fetch therefore rechecks selected rows rather than trusting a previously nonempty file.

## Outputs

- `download_plan.json`: immutable `argo_download_plan_v1` selection and estimate.
- `selection.json` and `selection.csv`: selected index rows without personal absolute paths.
- `raw/dac/<dac>/<wmo>/profiles/*.nc`: unchanged native GDAC files.
- `download_manifest.csv`: download, cache, mirror, revision, byte, and SHA-256 evidence.
- `health_check.json` and `profile_inventory.csv`: file/profile health and QC/data-mode summaries.
- `health_plots/spatial_coverage.png`, `time_coverage.png`, and `qc_data_mode_summary.png`.
- `download_status.json` and, for long runs, `download_monitor.html`.

The connector reports QC and availability; it does not silently select good observations, prefer adjusted variables, interpolate profiles, or create model inputs.

## Source provenance

Use the official Coriolis GDAC HTTPS tree as primary and the synchronized anonymous Argo S3 public dataset as fallback. Record DOI `10.17882/42182` and the UTC access date. No credentials or AWS account are required.
