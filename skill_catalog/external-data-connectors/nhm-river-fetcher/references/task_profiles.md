# NHM River Fetcher Task Profiles

Profiles define default file selection rules. They are not a substitute for a live manifest.

## `metadata-smoke`

Purpose: validate ScienceBase access, metadata saving, manifest generation, downloader behavior, and storage-preflight logic without large downloads.

Select:

- parent release item metadata JSON;
- tiny parent files such as `AK_output_variables_data_dictionary.csv`, `Table_1_AK_gages.csv`, or XML metadata;
- optionally one tiny output item `.out` or `.xml`.

Do not select archives.

## `geofabric-only`

Purpose: stage NHM geometry/parameter database.

Select:

- geofabric item `6644f81ed34e1955f5a42db4`;
- `param.zip`;
- `AK_paramDB_DataDictionary.csv`;
- `ak_geofabric_paramDB.xml`.

## `byPOIobs-seg-outflow`

Purpose: prepare the first Haines / Upper Lynn Canal flowline-discharge map using `seg_outflow.nc`.

Select:

- output-data item `6723ad59d34e4f57573e8c57`;
- `AK_byPOIobs_netcdf.zip`;
- output item XML and byPOIobs `.out` summary;
- geofabric `param.zip` and dictionaries.

After download, extract only the `seg_outflow.nc` ZIP member unless the user explicitly asks for more.

## `molly-all-listed`

Purpose: full Molly NHM-PRMS staging budget for later broad freshwater/hydropower context.

Select all attached files from:

- parent release item;
- output-data item;
- input/model-run files item;
- simulated streamflow-at-gages item;
- HUC12 aggregation item;
- Alaska geospatial fabric/parameter database item.

This profile is large. Use Kestrel or another HPC/storage target. Do not use local disk unless preflight passes with the default 4x storage multiplier.
