# ScienceBase Items for Alaska NHM-PRMS

Use these public ScienceBase item IDs for Alaska NHM/NHM-PRMS data discovery. Always query the item JSON before downloading files:

```text
https://www.sciencebase.gov/catalog/item/<ITEM_ID>?format=json
```

## Required Items

| Role | Item ID | Main use | Current compressed size |
|---|---|---|---:|
| parent | `64a84670d34e70357a27dd86` | Release metadata, dictionaries, gage table | ~0.001 GB |
| output-data | `6723ad59d34e4f57573e8c57` | Three NetCDF output archives and water-balance summaries | ~76.095 GB |
| input-run-files | `64c1cbebd34e70357a32a300` | PRMS inputs, parameters, forcing files | ~16.521 GB |
| gage-simulated-flow | `65cbb0f9d34ef4b119cb3780` | Simulated streamflow/statistics at gages | ~1.738 GB |
| huc12-aggregations | `667b23c8d34e6151c9d6be10` | HUC12 monthly/daily aggregations | ~8.913 GB |
| geofabric | `6644f81ed34e1955f5a42db4` | Alaska geospatial fabric and parameter database | ~0.043 GB |

Observed on 2026-06-22 from ScienceBase JSON:

- `AK_byPOIobs_netcdf.zip`: 30.182 GB / 28.109 GiB.
- `AK_precalibation_netcdf.zip`: 24.358 GB / 22.685 GiB.
- `AK_byHRU_netcdf.zip`: 21.555 GB / 20.075 GiB.
- `param.zip`: 0.043 GB / 0.040 GiB.
- Molly all-listed profile: ~103.31 GB compressed source files.
- Molly all-listed 4x preflight requirement: ~413.24 GB free.
- Local C: observed at ~23.45 GB free on 2026-06-22, not enough for Molly-scale work.

These sizes are cached facts for planning only. Scripts must use live ScienceBase JSON sizes when preparing a manifest.
