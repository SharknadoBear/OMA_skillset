# glorys_request_v1

```json
{
  "schema_version": "glorys_request_v1",
  "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
  "variables": ["water_level", "temperature", "salinity", "currents_3d"],
  "start": "2025-08-30T00:00:00Z",
  "end": "2025-09-03T00:00:00Z",
  "bbox": [-121.1912, 34.7451, -120.6102, 35.2604],
  "depth": [0, 200],
  "halo_cells": 1,
  "chunk_target_mib": 64
}
```

Supply exactly one spatial input: `bbox` in west/south/east/north order, or `mesh: {"path": "fort.14", "crs": "EPSG:4326"}`. Mesh paths resolve relative to the request file. Every node contributes to the bounds. Geographic mesh node IDs must be ordered 1 through NP. Record and recheck its SHA256; this is a coverage tool, not a full ADCIRC topology validator. Dateline-crossing regions require separate requests.

`dataset_id` defaults to daily and also accepts `cmems_mod_glo_phy_my_0.083deg_P1M-m`. Variables must be a nonempty list. Select individual catalogue field names or aliases: `water_level → zos`, `temperature → thetao`, `salinity → so`, `currents_3d → uo, vo`. Aliases expand and duplicates disappear in original order. Reject unknown fields and unknown request keys. Discovery happens against the resolved source, rather than a hard-coded five-field allowlist.

Dates require UTC `Z` or `+00:00`. Selection is `start <= actual source timestamp < end`. An inclusive Toolbox endpoint is set to the final selected timestamp, preventing accidental extra records. Daily temporal gaps and out-of-coverage requests fail. Monthly records retain the actual producer timestamp and represent whole monthly averaging periods; they are not daily or instantaneous fields.

Depth bounds are inclusive, nonnegative metres, positive down. They select existing levels without interpolation. Omit depth to take all available levels of requested 3D fields. Depth has no effect on 2D fields. The minimum published level is normally below 0 m, so `[0,200]` does not imply a synthetic 0 m level.

Spatial selection first encloses the requested continuous bounds on the regular source grid, then adds `halo_cells` surrounding cells on each side (default 1). A halo may stop at the source edge; the requested domain itself must be covered. `chunk_target_mib` (default 64, range 1–1024) bounds decoded chunk payload. Calendar-month groups split further if needed. An individual timestamp larger than this budget fails before download.

Optional acquisition controls: `service` accepts `arco-geo-series` (default) or `arco-time-series`; `chunk_mode` accepts `calendar_month` (default) or `bounded_batch`. Batched selection groups consecutive source timestamps across calendar months, limited by `chunk_target_mib`. Small multi-year coastal requests often benefit from time-series batches because their source time blocks span years. Compare dry-run estimates before acquisition. The source identity, request hash and plan hash bind the selected route and grouping. Omitting these controls preserves the original month-based behavior and old plan identities. `inventory --service arco-time-series` reads metadata through that service.

## Outputs and identity

`inventory.json` includes catalogue metadata, verified source axes, dimensions, units, field attributes and version. Without credentials, public catalogue inventory remains available but cannot establish exact coordinate selection. `download_plan.json` includes normalized request, source identity, selected axes, counts, calendar chunks and estimates. `mesh_reference.json` binds the input mesh without copying it. `request.json` records the normalized resolved request; it is provenance, not a fresh CLI input (use the original request to replan).

Run files: `chunks/*.nc`, `chunks/*.receipt.json`, `subset.nc`, `progress.json`, `manifest.json`, `health_report.json`. NetCDF chunk commits are atomic and receipts contain content SHA256, field-value and mask fingerprints, exact clocks and validation evidence. Source identity includes dataset version, part, service, Toolbox version, selected coordinate and field metadata hash. Appending unrelated source dates does not invalidate an existing selection. Changing selected metadata, the request or mesh does.

Output estimates count decoded field bytes; conservative peak disk headroom is four times decoded volume plus 64 MiB for metadata and filesystem overhead. Toolbox estimates are recorded separately when requested: estimated output `file_size_MB` and estimated maximum ARCO transfer `data_transfer_size_MB`. Actual network traffic is not measured. Block overhead can be much larger than the final subset; the Cal Poly pilot estimated about 1,745 MB maximum transfer per two-day chunk for about 0.2 MB decoded output. Assembly keeps one chunk in memory and appends time slices to an unlimited NetCDF dimension. Health compares assembled decoded values, attributes and missing patterns against every verified source-subset chunk.

Python: add `scripts` to `sys.path`, import `glorys_fetcher`, then call `plan(request_path, run_dir)`, `fetch(run_dir)`, `health(run_dir)`. `inventory(dataset_id, output=None)`, `check_runtime(access=False)`, `snapshot(request_path, run_dir)`, and `inspect(path)` return JSON-serializable reports. Controlled backend injection supports offline tests.
