---
name: nhd-flowline-fetcher
description: Fetch, inspect, and map USGS National Hydrography Dataset / NHDFlowline vectors from The National Map ArcGIS REST services. Use when Codex needs region-bounded river or stream centerlines, GeoPackage/Shapefile flowline products, NHD REST estimate-first planning, flowline health checks, or nearest assignment of model/point statistics such as NHM-PRMS discharge values to physical river-path flowlines.
---

# NHD Flowline Fetcher

Use this skill as a generic vector hydrography connector. Keep it focused on source-bounded NHD/NHDFlowline fetching, storage estimation, clipping, health checks, nearest point/statistic assignment, and map diagnostics.

## Source And Toolbox

- Primary v1 source: The National Map NHD REST `Flowline - Large Scale` layer.
- Default layer URL: `https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6`.
- Main packaged scripts:
- `scripts/estimate_data_request.py`
- `scripts/fetch_nhd_flowlines.py`
- `scripts/check_download_health.py`
- `scripts/assign_points_to_flowlines.py`
- `scripts/map_assigned_flowlines.py`

## Required Workflow

1. Identify a bbox as `min_lon max_lon min_lat max_lat`; for named project regions, record the exact bbox in run metadata.
2. Run the estimate hook before live download:

```bash
python scripts/estimate_data_request.py --bbox -136.5 -134.5 58.5 59.75 --output runs/case/download_estimate.json
```

3. Use the estimate to choose storage:
- download locally only when `local_free_bytes > 4 * estimated_requested_bytes`;
- if local disk does not satisfy that rule, use `$kestrel-hpc` for staging;
- if the estimate is unknown or the feature count is unexpectedly large, stop for review.
4. Fetch and save a clipped GeoPackage:

```bash
python scripts/fetch_nhd_flowlines.py --bbox -136.5 -134.5 58.5 59.75 --out-gpkg runs/case/nhd_flowline.gpkg --manifest runs/case/fetch_manifest.json
```

5. Run the health gate:

```bash
python scripts/check_download_health.py --flowlines runs/case/nhd_flowline.gpkg --bbox -136.5 -134.5 58.5 59.75 --output runs/case/health_check.json
```

6. For model/statistic overlays, assign nearest points to flowlines explicitly as a spatial nearest-neighbor product, not a hydrologic crosswalk:

```bash
python scripts/assign_points_to_flowlines.py --flowlines runs/case/nhd_flowline.gpkg --points-csv points.csv --stats-csv segment_summary.csv --out-gpkg runs/case/assigned_flowlines.gpkg --out-csv runs/case/assignment.csv
```

7. Create QC and value maps locally:

```bash
python scripts/map_assigned_flowlines.py --assigned-gpkg runs/case/assigned_flowlines.gpkg --points-csv points.csv --out-qc-map runs/case/qc.png --out-value-map runs/case/mean_discharge.png
```

## Implementation Rules

- Preserve NHD source fields such as `permanent_identifier`, `gnis_name`, `lengthkm`, `reachcode`, `ftype`, `fcode`, and `innetwork`.
- Page through REST results; do not assume one request can return all features.
- Use an Alaska-appropriate projected CRS such as EPSG:3338 for nearest-distance calculations in meters.
- If statistic tables are absent, still produce flowline and nearest-assignment QC outputs with null statistic columns, and skip value-colored discharge maps with an explicit message.
- Do not store credentials, tokens, or unsupported source-access claims in scripts, logs, or metadata.

## Validation

- Validate the skill with `quick_validate.py`.
- Compile all Python scripts after edits.
- Test estimate, fetch, health, assignment, and map scripts on a small bbox before broad use.
